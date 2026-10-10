from __future__ import annotations

import argparse
import copy
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import diagnose as diag_mod
import eval_router
import qscore as qscore_mod
import solve
import video_routing
import common as common_mod
from common import (
    VEH_DIR,
    STORE,
    GATE_NAMES,
    bump_version,
    call_llm,
    cosine,
    embed_texts,
    gate_cfg,
    llm_audit,
    load_cfg,
    load_genome,
    load_questions,
    resolve_capabilities,
    save_genome,
    snapshot_genome,
    version_axes,
)

MAX_EVAL_CANDIDATES = int(os.environ.get("VEH_MAX_EVAL_CANDIDATES", "0"))

_CAND_WORKERS_DEFAULT = 1

def _cand_workers() -> int:
    try:
        n = int(os.environ.get("VEH_CAND_WORKERS", _CAND_WORKERS_DEFAULT))
    except (TypeError, ValueError):
        return _CAND_WORKERS_DEFAULT
    return max(1, n)

def propose(g: dict, diagnosis: dict) -> list[tuple[str, dict]]:
    dom = diagnosis.get("dominant")
    counts = diagnosis.get("diagnosis", {})
    out: list[tuple[str, dict]] = []

    def variant(desc: str, fn) -> None:
        ng = copy.deepcopy(g)
        fn(ng)
        ng["parent_version"] = g["version"]
        ng["note"] = desc
        out.append((desc, ng))

    if dom == "retrieval_miss" or counts.get("retrieval_miss", 0) > 0:
        cur = g["retrieval"]["topk_macro"]
        variant(f"retrieval: topk_macro {cur} -> {cur + 2}", lambda n: n["retrieval"].__setitem__("topk_macro", cur + 2))
        if "macro" not in g["retrieval"]["entries"]:
            variant(
                "retrieval: prepend the macro entry (segment-level coarse locate)",
                lambda n: n["retrieval"].__setitem__("entries", ["macro", *g["retrieval"]["entries"]]),
            )
        elif g["retrieval"]["entries"][0] != "macro":
            variant(
                "retrieval: move the macro entry to the front",
                lambda n: n["retrieval"].__setitem__(
                    "entries", ["macro", *[e for e in g["retrieval"]["entries"] if e != "macro"]]
                ),
            )
        if "ocr" not in g["retrieval"]["entries"]:
            variant(
                "retrieval: add the ocr entry (nodes+ocr merged)",
                lambda n: n["retrieval"].__setitem__("entries", [*g["retrieval"]["entries"], "ocr"]),
            )
        if "enumerate" not in g["retrieval"]["entries"]:
            variant(
                "retrieval: add the enumerate entry",
                lambda n: n["retrieval"].__setitem__("entries", [*g["retrieval"]["entries"], "enumerate"]),
            )
        mc = g["retrieval"]["enumerate_min_cosine"]
        variant(
            f"retrieval: enumerate_min_cosine {mc} -> {round(mc - 0.1, 2)}",
            lambda n: n["retrieval"].__setitem__("enumerate_min_cosine", round(mc - 0.1, 2)),
        )

        if not g["retrieval"].get("rerank"):
            variant(
                "retrieval: rerank candidates by similarity score",
                lambda n: n["retrieval"].__setitem__("rerank", True),
            )
        if not (g["retrieval"].get("adaptive_topk") or {}).get("enabled"):
            cur_topk = g["retrieval"]["topk_macro"]
            variant(
                f"retrieval: adaptive topk (search {cur_topk * 2}, keep only >= 0.85 of best)",
                lambda n: n["retrieval"].__setitem__(
                    "adaptive_topk", {"enabled": True, "max_multiplier": 2, "keep_ratio": 0.85}
                ),
            )
            variant(
                f"retrieval: adaptive topk, stricter (search {cur_topk * 2}, keep >= 0.92 of best)",
                lambda n: n["retrieval"].__setitem__(
                    "adaptive_topk", {"enabled": True, "max_multiplier": 2, "keep_ratio": 0.92}
                ),
            )
        if not g["evidence"].get("primary_only"):
            variant(
                "evidence: locate with all candidates but cite only the best segment",
                lambda n: n["evidence"].__setitem__("primary_only", 1),
            )
            if g["retrieval"]["topk_macro"] >= 4:
                variant(
                    "evidence: cite only the best 2 segments",
                    lambda n: n["evidence"].__setitem__("primary_only", 2),
                )
        if not g["evidence"].get("dedup"):
            variant(
                "evidence: drop duplicate evidence lines",
                lambda n: n["evidence"].__setitem__("dedup", True),
            )

        if not g["retrieval"].get("use_learned_strategy"):
            try:
                import strategy_forge

                verified = [s for s in strategy_forge.load_registry().values() if s.get("verified")]
            except Exception:
                verified = []
            if verified:
                names = ", ".join(s["name"] for s in verified[:3])
                variant(
                    f"retrieval: enable the learned strategy ({names})",
                    lambda n: n["retrieval"].__setitem__("use_learned_strategy", True),
                )

    if counts.get("evidence_truncated", 0) > 0 or dom == "evidence_truncated":
        cur = g["evidence"]["max_chars"]
        variant(
            f"evidence: max_chars {cur} -> {int(cur * 1.6)}",
            lambda n: n["evidence"].__setitem__("max_chars", int(cur * 1.6)),
        )
        if g["evidence"]["max_events"] > 20:
            variant(
                "evidence: cap max_events at 15 (free budget for entities/OCR)",
                lambda n: n["evidence"].__setitem__("max_events", 15),
            )
        if not g["evidence"]["include_neighbor_macro"]:
            variant(
                "evidence: pull in adjacent macros (boundary-straddling events)",
                lambda n: n["evidence"].__setitem__("include_neighbor_macro", True),
            )

    if counts.get("guess_no_anchor", 0) > 0:
        variant(
            "budget: max_tool_rounds +5 (headroom for evidence gathering)",
            lambda n: n["budget"].__setitem__("max_tool_rounds", g["budget"]["max_tool_rounds"] + 5),
        )

    if not out:
        cur = g["retrieval"]["topk_macro"]
        variant(f"retrieval: topk_macro {cur} -> {cur + 1}", lambda n: n["retrieval"].__setitem__("topk_macro", cur + 1))
    return out

def _gen_weight(qtype: str) -> float:
    try:
        import ask
        return float(ask.type_weight(qtype))
    except Exception:
        return 1.0

def propose_questioner(g: dict, qreport: dict | None,
                       solver_trace_by_type: dict[str, dict] | None = None,
                       per_rule_cap: int = 0,
                       ) -> list[tuple[str, dict]]:
    if not qreport:
        return []
    out: list[tuple[str, dict]] = []
    per_rule: dict[str, int] = {}
    q = g["questioner"]
    n_opt = q["n_options"]
    random_base = 1.0 / max(2, n_opt)
    obj = g.get("questioner_objective") or {}
    zpd_lo, zpd_hi = obj.get("zpd_range", [0.3, 0.7])

    def variant(desc: str, fn, rule: str = "other") -> None:
        if per_rule_cap > 0 and per_rule.get(rule, 0) >= per_rule_cap:
            return
        per_rule[rule] = per_rule.get(rule, 0) + 1
        ng = copy.deepcopy(g)
        fn(ng)
        ng["parent_version"] = g["version"]
        ng["note"] = desc
        out.append((desc, ng))

    by_type = qreport.get("by_type", {})
    proposed_windows: set[int] = set()

    overrides = {k: v for k, v in (q.get("type_weight_override") or {}).items()
                 if not k.startswith("_")}
    touched_by_leak: set[str] = set()
    for t, d in by_type.items():
        if d.get("n", 0) < 5 or d.get("blind_acc", 0) <= max(0.6, random_base * 2.4):
            continue
        if not q["type_enabled"].get(t, True):
            continue
        touched_by_leak.add(t)
        if t in overrides:
            variant(
                f"questioner: disable type '{t}' (blind={d['blind_acc']:.2f} still above the line, "
                f"no improvement after down-weighting to {overrides[t]})",
                lambda n, t=t: n["questioner"]["type_enabled"].__setitem__(t, False),
                "leak",
            )
        else:
            cur_w = _gen_weight(t)
            new_w = round(cur_w * 0.3, 4)
            variant(
                f"questioner: down-weight type '{t}' {cur_w} -> {new_w} "
                f"(blind={d['blind_acc']:.2f} vs random={random_base:.2f}; down-weight first, do not disable)",
                lambda n, t=t, w=new_w: n["questioner"].setdefault(
                    "type_weight_override", {}).__setitem__(t, w),
                "leak",
            )

    cur_form = q["form"].get("object_reasoning")
    if by_type.get("object_reasoning", {}).get("blind_acc", 0) > random_base * 1.6:
        alt = "cooccurrence" if cur_form == "description" else "description"
        variant(
            f"questioner: object_reasoning form {cur_form} -> {alt}",
            lambda n, alt=alt: n["questioner"]["form"].__setitem__("object_reasoning", alt),
            "form",
        )

    overall_blind = qreport.get("summary", {}).get("blind_acc", 0.0)

    if overall_blind > random_base * 1.5 and n_opt < 6:
        variant(
            f"questioner: n_options {n_opt} -> {n_opt + 1} (random {random_base:.2f} -> {1/(n_opt+1):.2f})",
            lambda n: n["questioner"].__setitem__("n_options", n_opt + 1),
            "n_options",
        )

    d = q["distractor"]
    if overall_blind > random_base * 1.5:
        if d["adjacent_window"] > 1:
            proposed_windows.add(1)
            variant(
                f"questioner: adjacent_window {d['adjacent_window']} -> 1 (tighter context)",
                lambda n: n["questioner"]["distractor"].__setitem__("adjacent_window", 1),
                "distractor",
            )
        variant(
            f"questioner: same_type_only {d['same_type_only']} -> {not d['same_type_only']}",
            lambda n: n["questioner"]["distractor"].__setitem__(
                "same_type_only", not d["same_type_only"]
            ),
            "distractor",
        )

    for t, tr in sorted((solver_trace_by_type or {}).items()):
        if tr.get("n_scored", tr.get("n", 0)) < 5:
            continue
        metric = "acc_grounded" if tr.get("acc_grounded") is not None else "acc"
        val = tr.get(metric)
        if val is None or val <= zpd_hi:
            continue
        tol = float(q.get("option_len_tolerance", 0.35))
        if tol > 0.1:
            new_tol = round(max(0.1, tol * 0.6), 4)
            variant(
                f"questioner: option_len_tolerance {tol} -> {new_tol} "
                f"('{t}' {metric}={val:.2f} > {zpd_hi}; distractor length too close to the correct answer)",
                lambda n, v=new_tol: n["questioner"].__setitem__("option_len_tolerance", v),
                "harder_tol",
            )
        if d["adjacent_window"] > 1 and (d["adjacent_window"] - 1) not in proposed_windows:
            proposed_windows.add(d["adjacent_window"] - 1)
            variant(
                f"questioner: adjacent_window {d['adjacent_window']} -> {d['adjacent_window'] - 1} "
                f"('{t}' {metric}={val:.2f} > {zpd_hi}; distractors drawn from nearer segments)",
                lambda n, w=d["adjacent_window"] - 1:
                    n["questioner"]["distractor"].__setitem__("adjacent_window", w),
                "harder_window",
            )
        gap = float(q.get("min_gap_sec", 15.0))
        if gap < 60.0:
            new_gap = round(gap * 1.5, 1)
            variant(
                f"questioner: min_gap_sec {gap} -> {new_gap} "
                f"('{t}' {metric}={val:.2f} > {zpd_hi}; widen the span for temporal questions)",
                lambda n, v=new_gap: n["questioner"].__setitem__("min_gap_sec", v),
                "harder_gap",
            )
        break

    too_hard = [t for t, tr in (solver_trace_by_type or {}).items()
                if tr.get("n_scored", tr.get("n", 0)) >= 5
                and (tr.get("acc_grounded") if tr.get("acc_grounded") is not None
                     else tr.get("acc", 1.0)) < zpd_lo]
    if too_hard and float(q.get("option_len_tolerance", 0.35)) < 0.35:
        tol = float(q["option_len_tolerance"])
        new_tol = round(min(0.35, tol / 0.6), 4)
        variant(
            f"questioner: option_len_tolerance {tol} -> {new_tol} "
            f"({len(too_hard)} question types below {zpd_lo}; loosen by one notch)",
            lambda n, v=new_tol: n["questioner"].__setitem__("option_len_tolerance", v),
            "easier",
        )

    dist_by_type = (((qreport.get("summary") or {}).get("store") or {})
                    .get("distribution_by_type") or {})
    if dist_by_type:
        DIST_EPS = 0.05
        cand = [(t, d) for t, d in dist_by_type.items()
                if not d.get("blocked")
                and t not in touched_by_leak]

        over = sorted([c for c in cand if c[1].get("diff", 0) > DIST_EPS],
                      key=lambda c: -c[1]["diff"])
        if over:
            t, d = over[0]
            cur_w = _gen_weight(t)
            new_w = round(max(0.01, cur_w * 0.6), 4)
            if abs(new_w - cur_w) > 1e-9 and overrides.get(t) != new_w:
                variant(
                    f"questioner: quota '{t}' {cur_w} -> {new_w} "
                    f"(realised share {d['share']:.3f} exceeds target {d['target']:.3f} by "
                    f"{d['diff']:+.3f}; lower the quota in favour of under-produced types)",
                    lambda n, t=t, w=new_w: n["questioner"].setdefault(
                        "type_weight_override", {}).__setitem__(t, w),
                    "dist_over",
                )

        blind_ceiling = max(random_base * 1.5, obj.get("blind_target", random_base) * 1.5)
        under = sorted([c for c in cand if c[1].get("diff", 0) < -DIST_EPS],
                       key=lambda c: c[1]["diff"])
        for t, d in under:
            if d.get("n", 0) <= 0:
                continue
            if not q["type_enabled"].get(t, True):
                continue
            b = (by_type.get(t) or {}).get("blind_acc")
            if b is not None and b > blind_ceiling:
                continue
            cur_w = _gen_weight(t)
            new_w = round(cur_w * 1.8, 4)
            if overrides.get(t) == new_w:
                continue
            variant(
                f"questioner: quota '{t}' {cur_w} -> {new_w} "
                f"(realised share {d['share']:.3f} below target {d['target']:.3f} "
                f"{d['diff']:+.3f}, blind={b if b is None else round(b, 3)} "
                f"and does not exceed {blind_ceiling:.3f})",
                lambda n, t=t, w=new_w: n["questioner"].setdefault(
                    "type_weight_override", {}).__setitem__(t, w),
                "dist_under",
            )
            break
    return out

ROUTING_TEMPLATES: dict[str, str] = {
    "causal_edge": "For questions about cause, effect, or why something happened, call get_subgraph and read its relations/edges section. Causal links live there, not in the individual event descriptions.",
    "action_reasoning": "For questions about cause, effect, or why something happened, call get_subgraph and read its relations/edges section. Causal links live there, not in the individual event descriptions.",
    "temporal_order": "For questions about the order of events, call get_subgraph and compare the time_range of each micro-event numerically. Never infer order from the order the list happens to be in.",
    "temporal_reasoning": "For questions about the order of events, call get_subgraph and compare the time_range of each micro-event numerically. Never infer order from the order the list happens to be in.",
    "count_entity": "For counting questions, call get_subgraph once and count the matching entries in its explicit node list. Do not issue further searches: a search returns only its top_k, which undercounts.",
    "counting": "For counting questions, call get_subgraph once and count the matching entries in its explicit node list. Do not issue further searches: a search returns only its top_k, which undercounts.",
    "info_synopsis": "For questions about what a segment is overall about, call get_super_events or get_summary first for the arc, then confirm against one get_subgraph.",
    "spatial": "For questions about where things are relative to each other, call get_subgraph and read the entity visual_grounding and spatial relations.",
    "hallucination": "When asked whether something appears, treat absence from the subgraph as evidence it did not appear. Do not fill gaps from world knowledge.",
}

ROUTING_TEMPLATES["counting_readvideo"] = (
    "For counting questions, get_subgraph first, then call read_video on that exact time "
    "range and count from the frames; the memory record may have missed occurrences."
)
ROUTING_TEMPLATES["ocr_readvideo"] = (
    "If an on-screen-text answer is unclear from search_ocr_text, call read_video on that "
    "time range and read the text off the frames directly."
)

ANTI_PATTERN_TEMPLATES: dict[str, str] = {
    "repeat_subgraph": "Never call get_subgraph twice on the same macro_id. Its output is complete the first time; if you need it later, re-read it from your earlier observations.",
    "answer_from_summary": "Never answer from a segment summary or super-event description alone. Drill into get_subgraph for the concrete lines.",
    "burn_rounds": "If an observation already contains the answer, answer immediately instead of spending another round.",
}

def _policy_chars(rules: list[dict]) -> int:
    return sum(len(r.get("text", "")) for r in rules if not str(r.get("id", "")).startswith("_"))

def validate_policy(g: dict) -> tuple[bool, str]:
    pol = g.get("policy") or {}
    lim = pol.get("limits") or {}
    rules = [r for r in pol.get("rules", []) if not str(r.get("id", "")).startswith("_")]
    if len(rules) > lim.get("max_library_rules", 500):
        return False, f"{len(rules)} rules exceeds max_library_rules={lim.get('max_library_rules')}"
    total = _policy_chars(rules)
    if total > lim.get("max_library_chars", 100000):
        return False, f"{total} chars exceeds max_library_chars={lim.get('max_library_chars')}"
    per = lim.get("max_rule_chars", 200)
    for r in rules:
        if len(r.get("text", "")) > per:
            return False, (
                f"rule {r.get('id')!r} is {len(r.get('text', ''))} chars, over "
                f"max_rule_chars={per}: one long rule would monopolise the injection budget"
            )
    ids = [r.get("id") for r in rules]
    if len(ids) != len(set(ids)):
        return False, "duplicate rule ids"
    for r in rules:
        if r.get("kind") not in ("routing", "anti_pattern"):
            return False, f"rule {r.get('id')!r} has an unknown kind {r.get('kind')!r}"
        if not str(r.get("text", "")).strip():
            return False, f"rule {r.get('id')!r} has empty text"
        if "applies_to" in r and not isinstance(r["applies_to"], list):
            return False, f"rule {r.get('id')!r} has a non-list applies_to"
    if pol.get("tools_enabled", {}).get("search_asr_text"):
        return False, "search_asr_text is frozen off (vision-only setting)"
    if not any(pol.get("tools_enabled", {}).get(t) for t in ("get_subgraph",)):
        return False, "get_subgraph cannot be disabled: it is the only source of concrete facts"

    mt = pol.get("memory_tools") or {}
    if mt.get("enabled"):
        if (g.get("answer") or {}).get("mode") not in ("pipeline", "agent"):
            return False, (f"policy.memory_tools.enabled requires a known answer.mode, got "
                           f"{(g.get('answer') or {}).get('mode')!r}")
        on_fail = str(mt.get("on_embedding_failure", "record_and_continue")).lower()
        if on_fail not in ("record_and_continue", "record_and_reject"):
            return False, (f"policy.memory_tools.on_embedding_failure must be "
                           f"record_and_continue or record_and_reject, got {on_fail!r}")
        try:
            rounds = int(mt.get("max_rounds", 3))
            sel = int(mt.get("max_selected", 3))
            merge_cap = int(mt.get("max_candidates_after_merge", 8))
        except (TypeError, ValueError):
            return False, "policy.memory_tools round/selection limits must be integers"
        if rounds < 1:
            return False, "policy.memory_tools.max_rounds must be >= 1"
        if sel < 1:
            return False, "policy.memory_tools.max_selected must be >= 1"
        if merge_cap < sel:
            return False, (f"policy.memory_tools.max_candidates_after_merge={merge_cap} is "
                           f"below max_selected={sel}: selected candidates would be dropped "
                           f"before replay planning")
        ok_tpl, why_tpl = tool_stage_render_ok(g)
        if not ok_tpl:
            return False, why_tpl
    return True, ""

def near_duplicate(
    text: str, kind: str, rules: list[dict], g: dict, cfg: dict | None = None
) -> tuple[bool, str]:
    dd = (g.get("policy") or {}).get("dedup") or {}
    if not dd.get("enabled", True) or not rules:
        return False, ""
    thr = float(dd.get("max_cosine", 0.78))
    same = [r for r in rules if r.get("kind") == kind and r.get("text")]
    if not same:
        return False, ""
    try:
        vecs = embed_texts([text, *[r["text"] for r in same]], cfg)
    except Exception as exc:
        return False, ""
    q = vecs[0]
    best, best_i = -1.0, 0
    for i, v in enumerate(vecs[1:]):
        c = cosine(q, v)
        if c > best:
            best, best_i = c, i
    if best >= thr:
        dup = same[best_i]
        return True, (
            f"cosine {best:.3f} >= {thr} against same-kind rule {dup.get('id')!r} "
            f"({dup.get('text', '')[:70]}...)"
        )
    return False, ""

def _retire_candidate(rules: list[dict], idle_after: int, cur_version: int) -> int | None:
    scored = [
        (r.get("wins", 0), -(cur_version - r.get("added_at", 0)), i)
        for i, r in enumerate(rules)
        if not str(r.get("id", "")).startswith("_")
    ]
    if not scored:
        return None
    scored.sort()
    wins, neg_age, idx = scored[0]
    if wins == 0 and (cur_version - rules[idx].get("added_at", 0)) >= idle_after:
        return idx
    return idx if wins == 0 else None

def policy_has_effect(g: dict) -> bool:
    ans = g.get("answer") or {}
    if ans.get("mode") == "agent":
        return True
    if ans.get("mode") == "pipeline":
        return bool(((g.get("policy") or {}).get("memory_tools") or {}).get("enabled"))
    return False

def propose_policy(g: dict, diagnosis: dict, records: list[dict] | None = None) -> list[tuple[str, dict]]:
    if not policy_has_effect(g):
        return []
    out: list[tuple[str, dict]] = []
    pol = g["policy"]
    lim = pol["limits"]
    rules = [r for r in pol["rules"] if not str(r.get("id", "")).startswith("_")]
    have_text = {r["text"] for r in rules}

    def variant(desc: str, fn) -> None:
        ng = copy.deepcopy(g)
        fn(ng)
        ok, why = validate_policy(ng)
        if not ok:
            return
        ng["parent_version"] = g["version"]
        ng["note"] = desc
        out.append((desc, ng))

    def _add(rule: dict, desc: str) -> None:
        def fn(n):
            rs = n["policy"]["rules"]
            live = [r for r in rs if not str(r.get("id", "")).startswith("_")]
            if len(live) >= lim.get("max_library_rules", 500):
                raise ValueError("library safety net reached")
            rs.append(rule)

        variant(desc, fn)

    by_type = (diagnosis.get("base") or {}).get("by_type") or {}
    for qtype, d in sorted(by_type.items(), key=lambda kv: kv[1].get("acc", 1.0)):
        if d.get("n", 0) < 2 or d.get("acc", 1.0) > 0.4:
            continue
        tpl = ROUTING_TEMPLATES.get(qtype)
        if not tpl or tpl in have_text:
            continue
        _add(
            {"id": f"r_route_{qtype}", "kind": "routing", "text": tpl,
             "added_at": g["version"], "wins": 0, "trials": 0},
            f"policy ADD routing rule for '{qtype}' (acc={d.get('acc', 0):.2f}, n={d.get('n')})",
        )

    if records:
        dup = 0
        for r in records:
            calls = r.get("tool_calls") or []
            if len(calls) != len(set(calls)):
                dup += 1
        if dup >= max(2, len(records) * 0.15):
            tpl = ANTI_PATTERN_TEMPLATES["repeat_subgraph"]
            if tpl not in have_text:
                _add(
                    {"id": "r_anti_repeat_subgraph", "kind": "anti_pattern", "text": tpl,
                     "added_at": g["version"], "wins": 0, "trials": 0},
                    f"policy ADD anti-pattern: repeated identical tool calls in {dup}/{len(records)} questions",
                )

        maxed = sum(1 for r in records if r.get("tool_rounds", 0) >= g["budget"]["max_tool_rounds"])
        if maxed >= max(2, len(records) * 0.2):
            cur = g["budget"]["max_tool_rounds"]
            if cur < 10:
                variant(
                    f"budget: max_tool_rounds {cur} -> {cur + 2} ({maxed} questions hit the cap)",
                    lambda n: n["budget"].__setitem__("max_tool_rounds", cur + 2),
                )
            w = lim["obs_window"]
            if w < 4:
                variant(
                    f"policy: obs_window {w} -> {w + 1} (less forgetting, more context cost)",
                    lambda n: n["policy"]["limits"].__setitem__("obs_window", w + 1),
                )

    victim = _retire_candidate(rules, lim.get("retire_after_idle_rounds", 3), g["version"])
    if victim is not None and len(rules) > 3:
        vid = rules[victim]["id"]
        variant(
            f"policy DELETE rule '{vid}' (wins=0, idle since v{rules[victim].get('added_at')})",
            lambda n, vid=vid: n["policy"].__setitem__(
                "rules", [r for r in n["policy"]["rules"] if r.get("id") != vid]
            ),
        )
    return out

POLICY_SENTINELS = ("TOOL:", "Answer:", "Evidence:", "{", "}")

def policy_render_ok(g: dict) -> tuple[bool, str]:
    try:
        tool_list, routing, anti = solve.render_policy_sections(g)
        tpl = (VEH_DIR / g["answer"]["agent_prompt_file"]).read_text(encoding="utf-8")
        rendered = tpl.format(
            tool_list=tool_list, routing_rules=routing, anti_patterns=anti,
            question="Q?", options="A. x\nB. y", round=1, max_rounds=4,
            observations="(none)", rounds_left=4,
        )
    except KeyError as exc:
        return False, f"rule text broke prompt formatting (stray brace {exc})"
    except Exception as exc:
        return False, f"render failed: {str(exc)[:120]}"
    for s in ("TOOL:", "Answer:", "Evidence:"):
        if s not in rendered:
            return False, f"rendered prompt lost the {s!r} sentinel; answers would stop parsing"
    return True, ""

TOOL_STAGE_SENTINELS = ("TOOL:", "SELECT:", "DONE")

def tool_stage_render_ok(g: dict) -> tuple[bool, str]:
    mt = ((g.get("policy") or {}).get("memory_tools") or {})
    if not mt.get("enabled"):
        return True, ""
    rel = mt.get("prompt_file") or "prompts/pipeline_tools_v1.txt"
    try:
        import memory_tools

        tpl = (VEH_DIR / rel).read_text(encoding="utf-8")
        cat = memory_tools.build_catalog(g, "", solve.enabled_tools(g), solve.TOOL_SPECS)
        rendered = tpl.format(
            tool_list=memory_tools.render_catalog(cat),
            question="Q?", options="A. x\nB. y",
            retrieved="(evidence)", candidates="macro_0001",
            observations="(none)", round=1, max_rounds=3, rounds_left=3,
            max_selected=int(mt.get("max_selected", 3)),
        )
    except FileNotFoundError:
        return False, f"policy.memory_tools.prompt_file not found: {rel}"
    except KeyError as exc:
        return False, f"tool-stage prompt has an unknown placeholder {exc}"
    except Exception as exc:
        return False, f"tool-stage prompt render failed: {str(exc)[:120]}"
    for s in TOOL_STAGE_SENTINELS:
        if s not in rendered:
            return False, (f"tool-stage prompt lost the {s!r} sentinel; every round would "
                           f"be counted as a protocol error and the stage would silently "
                           f"fall back to the initial retrieval")
    return True, ""

def apply_policy_op(g: dict, op: dict, cfg: dict | None = None) -> tuple[bool, str, dict | None]:
    kind_op = str(op.get("op", "")).upper()
    ng = copy.deepcopy(g)
    rules = ng["policy"]["rules"]
    live = [r for r in rules if not str(r.get("id", "")).startswith("_")]
    lim = ng["policy"]["limits"]

    if kind_op == "ADD":
        text = " ".join(str(op.get("text") or "").split())
        kind = str(op.get("kind") or "routing")
        if len(text) < 20:
            return False, "rule text too short to be a real rule", None
        if any(c in text for c in "{}"):
            return False, "rule text contains braces, which breaks prompt formatting", None
        if text in {r["text"] for r in live}:
            return False, "identical rule already present", None
        dup, why = near_duplicate(text, kind, live, ng, cfg)
        if dup:
            return False, f"near-duplicate of an existing rule: {why}", None
        applies = op.get("applies_to")
        if applies is not None and not isinstance(applies, list):
            return False, "applies_to must be a list of question types", None
        rid = str(op.get("id") or f"r_llm_{abs(hash(text)) % 10000}")
        if rid in {r.get("id") for r in live}:
            rid = f"{rid}_{len(live)}"
        if len(live) >= lim.get("max_library_rules", 500):
            return False, f"library is at max_library_rules={lim.get('max_library_rules')}", None
        rules.append({"id": rid, "kind": kind, "applies_to": applies or [], "text": text,
                      "added_at": g["version"], "wins": 0, "trials": 0})

    elif kind_op == "EDIT":
        rid = str(op.get("id") or "")
        target = next((r for r in rules if r.get("id") == rid), None)
        if target is None:
            return False, f"no rule with id {rid!r}", None
        text = " ".join(str(op.get("text") or "").split())
        if len(text) < 20:
            return False, "replacement text too short", None
        if any(c in text for c in "{}"):
            return False, "replacement text contains braces", None
        others = [r for r in live if r.get("id") != rid]
        dup, why = near_duplicate(text, target.get("kind", "routing"), others, ng, cfg)
        if dup:
            return False, f"edit would duplicate another rule: {why}", None
        target["text"] = text
        if isinstance(op.get("applies_to"), list):
            target["applies_to"] = op["applies_to"]

    elif kind_op == "DELETE":
        rid = str(op.get("id") or "")
        target = next((r for r in live if r.get("id") == rid), None)
        if target is None:
            return False, f"no rule with id {rid!r}", None
        if len(live) <= 3:
            return False, "refusing to shrink the policy below 3 rules", None
        if target.get("protected"):
            return False, (
                f"rule {rid!r} is flagged protected (backed by a direct measurement); "
                f"retire it only by changing that measurement"
            ), None
        if target.get("wins", 0) > 0:
            return False, (
                f"rule {rid!r} has wins={target['wins']}: it has already been credited in "
                f"an accepted patch, so deletion needs held-out evidence, not a trace guess"
            ), None
        ng["policy"]["rules"] = [r for r in rules if r.get("id") != rid]

    else:
        return False, f"unknown op {kind_op!r}", None

    ok, why = validate_policy(ng)
    if not ok:
        return False, why, None
    ok, why = policy_render_ok(ng)
    if not ok:
        return False, why, None
    return True, "", ng

def _failure_traces(records: list[dict], max_per_type: int = 2, max_total: int = 8) -> str:
    by_type: dict[str, list[dict]] = {}
    for r in records:
        if not r.get("correct"):
            by_type.setdefault(r["type"], []).append(r)
    out, n = [], 0
    for qtype, rs in by_type.items():
        for r in rs[:max_per_type]:
            calls = r.get("tool_calls") or []
            dup = " <-- the same call repeated" if len(calls) != len(set(calls)) else ""
            out.append(
                f"- type={qtype} rounds={r.get('tool_rounds', 0)} "
                f"predicted={r.get('pred')} correct_answer={r.get('gold')} "
                f"found_the_right_segment={r.get('gold_macro_hit')}\n"
                f"    tool sequence: {' -> '.join(c.split('(')[0] for c in calls) or '(none)'}{dup}\n"
                f"    full calls: {calls}"
            )
            n += 1
            if n >= max_total:
                return "\n".join(out)
    return "\n".join(out) or "(no failures recorded)"

def propose_policy_llm(
    g: dict,
    diagnosis: dict,
    records: list[dict] | None,
    cfg: dict,
    model: str | None = None,
    max_candidates: int = 4,
) -> list[tuple[str, dict]]:
    if not policy_has_effect(g):
        return []
    records = records or []
    live = [r for r in g["policy"]["rules"] if not str(r.get("id", "")).startswith("_")]
    lim = g["policy"]["limits"]
    inj = g["policy"].get("inject") or {}
    dd = g["policy"].get("dedup") or {}
    tool_list, _, _ = solve.render_policy_sections(g)
    by_type = (diagnosis.get("base") or {}).get("by_type") or {}

    prompt = f"""You are improving the tool-use policy of an agent that answers multiple-choice
questions about long videos by querying a hierarchical graph memory. The agent cannot see
the video. You are looking at its own failures and writing rules to fix them.

## Memory layout
root -> super_events (arcs) -> macro_events (segments) -> subgraph (micro-events,
entities, on-screen texts, relations). Only a subgraph holds concrete facts. Note that
relations/edges exist ONLY inside a subgraph -- they are not in the search index.

## Tools the agent may call
{tool_list}

## The policy it is running now
{json.dumps([{k: r.get(k) for k in ('id', 'kind', 'applies_to', 'text')} for r in live], ensure_ascii=False, indent=2)}

## Accuracy by question type
{json.dumps({t: {'n': v.get('n'), 'acc': v.get('acc')} for t, v in by_type.items()}, ensure_ascii=False)}

## Its actual failing trajectories
{_failure_traces(records)}

## Constraints
- The library itself is not capacity-limited, but only {inj.get('max_rules_per_call', 5)} rules
  ({inj.get('max_chars_per_call', 900)} characters) are injected per call, selected by
  question type. So a rule earns its slot by being specific, not by existing.
- Set "applies_to" to the question types a routing rule is for, e.g. ["causal_edge"].
  Leave it out only for genuinely universal rules -- an over-broad rule wastes a slot on
  every question. anti_pattern rules are always injected, so keep them few and general.
- A near-duplicate of an existing rule will be REJECTED automatically (cosine >= {dd.get('max_cosine', 0.85)}).
  Do not restate a rule in different words; EDIT the original instead.
- kind is either "routing" (which tool to use when) or "anti_pattern" (what never to do).
- Rule text must be one imperative sentence, under {lim.get('max_rule_chars', 200)} characters, no curly braces.
- A rule must be actionable by naming a concrete tool or a concrete thing to read.
  "Think more carefully" is useless. "Read the relations section of the subgraph" is good.

## Task
Diagnose from the trajectories above WHY these questions failed, then propose at most
{max_candidates} operations that would fix it. Each operation is one of:
  {{"op": "ADD", "kind": "routing"|"anti_pattern", "applies_to": ["type"], "text": "...", "reason": "..."}}
  {{"op": "EDIT", "id": "<existing id>", "text": "...", "reason": "..."}}
  {{"op": "DELETE", "id": "<existing id>", "reason": "..."}}

Prefer editing or deleting a rule that the traces show is being misfollowed, over piling
on new ones. Return ONLY a JSON array, no prose.
"""
    try:
        text = call_llm(prompt, cfg, max_tokens=1600, temperature=0.4,
                        role="policy_evolver", model=model)
    except Exception as exc:
        return []

    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return []
    try:
        ops = json.loads(m.group(0))
    except json.JSONDecodeError as exc:
        return []

    out: list[tuple[str, dict]] = []
    for op in ops[: max_candidates * 2]:
        if not isinstance(op, dict):
            continue
        ok, why, ng = apply_policy_op(g, op, cfg)
        label = f"{op.get('op')} {op.get('kind') or op.get('id') or ''}".strip()
        if not ok:
            continue
        ng["parent_version"] = g["version"]
        snippet = (op.get("text") or op.get("id") or "")[:60]
        desc = f"[llm-policy] {label}: {snippet} ({str(op.get('reason', ''))[:60]})"
        ng["note"] = desc
        out.append((desc, ng))
        if len(out) >= max_candidates:
            break
    return out

EVOLVABLE_PATHS: dict[str, tuple] = {
    "routing.max_direct_sec": (float, 60.0, 900.0),
    "routing.max_chunked_sec": (float, 600.0, 7200.0),
    "retrieval.topk_macro": (int, 1, 12),
    "retrieval.search_multiplier": (int, 1, 10),
    "retrieval.enumerate_min_cosine": (float, 0.05, 0.9),
    "retrieval.rerank": (bool, None, None),
    "retrieval.use_learned_strategy": (bool, None, None),
    "retrieval.adaptive_topk.enabled": (bool, None, None),
    "retrieval.adaptive_topk.max_multiplier": (int, 1, 4),
    "retrieval.adaptive_topk.keep_ratio": (float, 0.5, 1.0),
    "evidence.frames.enabled": (bool, None, None),
    "evidence.frames.route_max_min": (float, 0.0, 120.0),
    "evidence.frames.route_max_cosine": (float, 0.0, 1.0),
    "evidence.frames.fps": (float, 0.1, 2.0),
    "evidence.frames.chunk_sec": (float, 30.0, 600.0),
    "evidence.frames.max_chunks": (int, 0, 12),
    "evidence.frames.max_dim": (int, 256, 1024),
    "evidence.primary_only": (int, 0, 8),
    "evidence.dedup": (bool, None, None),
    "evidence.max_chars": (int, 1000, 40000),
    "evidence.max_events": (int, 1, 999),
    "evidence.max_entities": (int, 1, 999),
    "evidence.max_ocr": (int, 1, 999),
    "evidence.include_neighbor_macro": (bool, None, None),
    "evidence.include_summary": (bool, None, None),
    "evidence.summary_budget_share": (float, 0.0, 0.6),
    "policy.read_video.max_window_sec": (float, 4.0, 120.0),
    "policy.read_video.node_window_pad_sec": (float, 0.0, 30.0),
    "policy.read_video.node_window": (bool, None, None),
    "policy.read_video.max_segments": (int, 1, 5),
    "policy.read_video.fps": (float, 0.2, 4.0),
    "policy.read_video.max_frames_per_call": (int, 4, 192),
    "policy.read_video.max_dim": (int, 384, 1024),
    "policy.read_video.max_calls_per_question": (int, 1, 3),
    "policy.read_video.second_look": (bool, None, None),
    "policy.read_video.focus_weight": (float, 1.0, 8.0),
    "budget.max_tool_rounds": (int, 1, 40),
    "answer.temperature": (float, 0.05, 1.0),
    "questioner.n_options": (int, 3, 6),
    "questioner.min_gap_sec": (float, 1.0, 120.0),
    "questioner.tries": (int, 1, 10),
    "questioner.option_len_tolerance": (float, 0.05, 1.0),
    "questioner.max_lexical_overlap_diff": (float, 0.05, 1.0),
    "questioner.per_macro_max": (int, 1, 20),
    "questioner.distractor.adjacent_window": (int, 1, 8),
    "questioner.distractor.same_type_only": (bool, None, None),
    "questioner.distractor.prefer_adjacent_macros": (bool, None, None),
    "questioner.option_text.max_len": (int, 40, 400),
    "policy.memory_tools.enabled": (bool, None, None),
    "policy.memory_tools.max_rounds": (int, 1, 8),
    "policy.memory_tools.max_selected": (int, 1, 5),
    "policy.memory_tools.max_candidates_after_merge": (int, 1, 16),
    "policy.memory_tools.selection_drives_replay": (bool, None, None),
    "policy.memory_tools.max_obs_chars": (int, 200, 4000),
    "policy.memory_tools.obs_window": (int, 1, 4),
    "policy.memory_tools.max_parse_errors": (int, 1, 4),
    "policy.memory_tools.max_tokens": (int, 400, 4000),
    "policy.memory_tools.temperature": (float, 0.0, 1.0),
    "policy.memory_tools.allow_learned_tools": (bool, None, None),
    "policy.memory_tools.max_returned_recorded": (int, 5, 200),
    "policy.read_video.require_replay_on_retrieve": (bool, None, None),
    "policy.read_video.max_frames_total": (int, 0, 400),
    "policy.read_video.second_look_overlap_reject": (float, 0.1, 1.0),
}
ENUM_PATHS: dict[str, list] = {
    "questioner.form.object_reasoning": ["description", "cooccurrence"],
    "questioner.form.info_synopsis": ["description", "label", "both"],
    "questioner.form.attribute_perception": ["attributes", "description"],
    "policy.memory_tools.on_embedding_failure": ["record_and_continue", "record_and_reject"],
}

LIST_PATHS: dict[str, tuple[list, int, int]] = {
    "retrieval.entries": (["macro", "nodes", "ocr", "enumerate", "time"], 1, 4),
}

def _get_path(d: dict, path: str):
    cur = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur

def _set_path(d: dict, path: str, val) -> None:
    keys = path.split(".")
    cur = d
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = val

def validate_patch(path: str, val) -> tuple[bool, str]:
    base_path, route = video_routing.split_scoped(path)
    if route is not None:
        if route not in video_routing.ROUTES:
            return False, f"unknown route {route!r}; expected {video_routing.ROUTES}"
        if base_path not in video_routing.ROUTE_SCOPED_PATHS:
            return False, (f"{base_path} may not be scoped per route; "
                           f"scopable paths: {sorted(video_routing.ROUTE_SCOPED_PATHS)}")
        path = base_path

    if path.startswith("questioner.type_enabled."):
        return (True, "") if isinstance(val, bool) else (False, "type_enabled must be bool")
    if path in ENUM_PATHS:
        return (True, "") if val in ENUM_PATHS[path] else (False, f"must be one of {ENUM_PATHS[path]}")
    if path in LIST_PATHS:
        allowed, lo, hi = LIST_PATHS[path]
        if not isinstance(val, list) or not val:
            return False, "must be a non-empty list"
        if not (lo <= len(val) <= hi):
            return False, f"list length out of range [{lo}, {hi}]"
        bad = [v for v in val if v not in allowed]
        if bad:
            return False, f"invalid entries {bad}; allowed {allowed}"
        if len(set(val)) != len(val):
            return False, "duplicate entries"
        return True, ""
    if path not in EVOLVABLE_PATHS:
        return False, "path not evolvable"
    typ, lo, hi = EVOLVABLE_PATHS[path]
    if typ is bool:
        return (True, "") if isinstance(val, bool) else (False, "must be bool")
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        return False, f"must be {typ.__name__}"
    if not (lo <= val <= hi):
        return False, f"out of range [{lo}, {hi}]"
    return True, ""

def semantic_guard(path: str, val, g: dict, qreport: dict | None) -> tuple[bool, str]:
    if path == "retrieval.use_learned_strategy" and val is True:
        try:
            import strategy_forge
            if not any(s.get("verified") for s in strategy_forge.load_registry().values()):
                return False, ("no verified strategy in the registry -- turning this on "
                               "would change nothing")
        except Exception as exc:
            return False, f"cannot read the strategy registry: {str(exc)[:120]}"

    if path == "evidence.frames.enabled" and val is True:
        fr = (g.get("evidence", {}) or {}).get("frames") or {}
        if float(fr.get("route_max_min") or 0) <= 0:
            return False, ("route_max_min is 0, so no video would qualify -- turning "
                           "frames on would change nothing")
    if path == "evidence.frames.route_max_min":
        fr = (g.get("evidence", {}) or {}).get("frames") or {}
        if float(val) > 0 and not fr.get("enabled"):
            return False, ("evidence.frames.enabled is off, so widening the route "
                           "changes nothing; enable the arm first")
        if float(val) > FRAME_ROUTE_MAX_MIN:
            return False, (f"route_max_min {val} exceeds the {FRAME_ROUTE_MAX_MIN} min "
                           f"ceiling (VEH_FRAME_ROUTE_MAX_MIN); frames are ~1000x the "
                           f"cost of text evidence")

    if not path.startswith("questioner.type_enabled.") or val is not False:
        return True, ""
    qtype = path.rsplit(".", 1)[-1]
    by_type = (qreport or {}).get("by_type", {})
    d = by_type.get(qtype)
    if d is None:
        return False, f"no Blind statistics for '{qtype}'; refusing to disable blindly"

    n_opt = g["questioner"]["n_options"]
    random_base = 1.0 / max(2, n_opt)
    blind = d.get("blind_acc", 0.0)
    if d.get("n", 0) < 5:
        return False, f"only {d.get('n', 0)} samples for '{qtype}'; too few to justify disabling"
    if blind <= random_base * 1.5:
        return False, (
            f"'{qtype}' has blind_acc={blind:.2f} <= {random_base * 1.5:.2f}; "
            f"low Blind means it genuinely requires the video -- disabling it would "
            f"remove a GOOD question type"
        )
    active = [t for t, on in g["questioner"]["type_enabled"].items()
              if on and not t.startswith("_")]
    if len(active) <= 4:
        return False, f"only {len(active)} types active; refusing to shrink diversity further"
    return True, ""

def propose_llm(
    g: dict,
    diagnosis: dict,
    qreport: dict | None,
    cfg: dict,
    model: str | None = None,
    max_candidates: int = 4,
    core_frames: bool = False,
) -> list[tuple[str, dict]]:
    current = {p: _get_path(g, p)
               for p in list(EVOLVABLE_PATHS) + list(ENUM_PATHS) + list(LIST_PATHS)}
    current = {k: v for k, v in current.items() if v is not None}
    enabled = {k: v for k, v in g["questioner"]["type_enabled"].items() if not k.startswith("_")}

    if core_frames:
        core_frames_section = """## Core-arm frame fields (THIS ROUND THEY ARE SEARCHABLE)
These control the arm that answers by reading the RAW VIDEO (routes
`direct_read_video` / `chunked_read_video`), not by retrieving from graph memory.
Questions on those routes are untouched by every retrieval/evidence field above --
for them these are the ONLY genes that matter:
- `evidence.frames.fps`: sampling rate. Higher = more frames = more payload.
- `evidence.frames.chunk_sec` / `.max_chunks`: how the video is cut into chunks and how
  many are read; together they bound both coverage and cost.
- `evidence.frames.max_dim`: per-frame resolution.
- `routing.max_direct_sec` / `routing.max_chunked_sec`: the duration boundaries that
  decide WHICH arm a video goes to. Moving them re-routes questions wholesale.
Propose at least one patch here, as a single-field change so its effect is attributable.
"""
    else:
        core_frames_section = """## Do NOT propose core-arm frame patches this round
`evidence.frames.*` and `routing.*` are NOT searchable this round: any patch touching
them is discarded before evaluation, so proposing one only wastes a slot.
"""

    prompt = f"""You are tuning a long-video question-answering agent. Propose genome patches.

## Current genome values (only these paths may be changed)
{json.dumps(current, ensure_ascii=False, indent=2)}

## Question types currently enabled
{json.dumps(enabled, ensure_ascii=False)}
You may also patch `questioner.type_enabled.<type>` with a boolean.

## Enumerated fields and their allowed values
{json.dumps(ENUM_PATHS, ensure_ascii=False, indent=2)}

## List-valued fields (give the whole list; order matters, no duplicates)
{json.dumps({k: {"allowed": v[0], "min_len": v[1], "max_len": v[2]}
             for k, v in LIST_PATHS.items()}, ensure_ascii=False, indent=2)}

## Localisation fields: these change HOW PRECISELY the answer is located in time
- `retrieval.entries`: which retrieval entries run. `ocr` is the only entry returning
  on-screen-text nodes; their timestamps feed the precise replay window.
- `policy.read_video.max_window_sec`: hard cap on replay window width. The replay
  window is now built around the matched NODE's timestamp, not the parent segment,
  so this cap can go far below the old 60s without losing the answer.
- `policy.read_video.node_window_pad_sec`: seconds of context added on each side of
  the matched node. Small = precise but may clip an answer that sits just outside;
  large = degrades back toward segment-level blur.
- `policy.read_video.max_segments`: how many segments may be watched. Frame budget is
  split across them, so raising it trades frames-per-segment for coverage, not cost.

Locating the segment is not the same as locating the moment: a segment spans far more
time than the answer does, so frame budget spent on the wrong part of it is a total
loss. Narrowing the window is therefore not merely a cost saving.

## Strategy fields: these change HOW evidence is chosen, not how much
- `retrieval.rerank`: order candidate segments by similarity score instead of by the
  order the search entries returned them.
- `retrieval.adaptive_topk.enabled` + `.max_multiplier` + `.keep_ratio`: search
  `topk_macro * max_multiplier` segments, then keep only those scoring at least
  `keep_ratio` of the best one. A wide search with a strict keep_ratio behaves like a
  narrow search when one segment clearly wins, and like a wide one when several tie.
  Note that a low `keep_ratio` keeps almost every candidate, which is just a larger
  `topk_macro` by another name.
- `evidence.primary_only`: N > 0 means locate using every candidate but put only the
  best N segments in the prompt. This is NOT the same as lowering `topk_macro`: the
  retrieval hit-rate metric still counts all candidates.
- `evidence.dedup`: drop evidence lines that repeat a line already included.

Two effects constrain these fields. Retrieving more segments raises the chance that the
answer is somewhere in the prompt, but the extra segments are usually irrelevant and
dilute it, so accuracy does not follow the hit rate. Simply adding more text, without
changing which segments are chosen, tends to cost accuracy rather than gain it. The
useful direction is therefore to keep the search wide but stop the extra segments from
crowding the prompt -- neither widening further nor narrowing back down.

## Solver failure attribution (from ablation)
{json.dumps(diagnosis.get("diagnosis", {}), ensure_ascii=False)}
by question type: {json.dumps(diagnosis.get("by_type", {}), ensure_ascii=False)}

Meaning of each category:
- retrieval_miss: retrieval did not return the macro that holds the answer
- evidence_truncated: the information is in memory but never reached the prompt
- memory_missing_or_reasoning: not in memory at all, or a reasoning failure
- guess_no_anchor: answered correctly but the cited evidence does not support it

## Question-store quality
{json.dumps((qreport or {}).get("summary", {}), ensure_ascii=False)}
per-type Blind accuracy: {json.dumps((qreport or {}).get("by_type", {}), ensure_ascii=False)}

Blind accuracy is how often the question is answered WITHOUT any video evidence.
The ideal value equals pure guessing (1 / n_options). A type whose Blind accuracy stays
much higher is not testing video understanding.

{core_frames_section}
## Task
Propose at most {max_candidates} INDEPENDENT patches, each a minimal single-field change
that targets the dominant problem above. Prefer changes with a clear causal story.

Return ONLY a JSON array, no prose:
[{{"path": "retrieval.topk_macro", "value": 4, "reason": "one short sentence"}}]
"""
    try:
        text = call_llm(prompt, cfg, max_tokens=1200, temperature=0.3, role="evolver", model=model)
    except Exception as exc:
        return []

    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return []
    try:
        items = json.loads(m.group(0))
    except json.JSONDecodeError as exc:
        return []

    out: list[tuple[str, dict]] = []
    for it in items[: max_candidates * 2]:
        if not isinstance(it, dict):
            continue
        path, val = it.get("path"), it.get("value")
        ok, why = validate_patch(str(path), val)
        if not ok:
            continue
        ok, why = semantic_guard(str(path), val, g, qreport)
        if not ok:
            continue
        if _get_path(g, str(path)) == val:
            continue
        ng = copy.deepcopy(g)
        _set_path(ng, str(path), val)
        ng["parent_version"] = g["version"]
        desc = f"[llm] {path}: {_get_path(g, str(path))} -> {val} ({it.get('reason', '')[:70]})"
        ng["note"] = desc
        out.append((desc, ng))
        if len(out) >= max_candidates:
            break
    return out

STRATEGY_MIN_PER_TYPE = 6

def propose_strategy_llm(cfg: dict, g: dict, diagnosis: dict, questions: list[dict],
                         attempts: int = 3,
                         min_per_type: int = STRATEGY_MIN_PER_TYPE,
                         gate5_mode: str = "full",
                         churn_max: float | None = None,
                         ) -> tuple[str, dict, dict] | None:
    try:
        import strategy_forge
    except Exception as exc:
        return None

    try:
        ok, why, spec = strategy_forge.forge(
            cfg, g, questions, diagnosis, None, min_per_type, attempts,
            min_per_type=min_per_type, commit=False,
            gate5_mode=gate5_mode, churn_max=churn_max)
    except Exception as exc:
        return None
    if not ok or not spec:
        return None

    stats = spec.get("_cross_stats") or {}
    n = stats.get("n", 0)
    ng = copy.deepcopy(g)
    ng.setdefault("retrieval", {})["use_learned_strategy"] = True
    ng["parent_version"] = g["version"]
    desc = (f"[strategy] retrieval delegated to the model-authored strategy '{spec['name']}'"
            f" ({str(spec.get('desc', ''))[:60]}; "
            f"gate5 on {n} items: acc {stats.get('baseline_correct', 0)} -> "
            f"{stats.get('strategy_correct', 0)}, "
            f"hit {stats.get('baseline_hit', 0)} -> {stats.get('strategy_hit', 0)})")
    if not spec.get("_gate5_enforced", True):
        desc += "[gate5 not enforced]"
    ng["note"] = desc
    return desc, ng, spec

def propose_tool_llm(cfg: dict, g: dict, diagnosis: dict, questions: list[dict],
                     attempts: int = 3,
                     gate5_mode: str = "full",
                     gate5_min_pass: float = 0.75,
                     model: str | None = None,
                     ) -> tuple[str, dict, dict] | None:
    try:
        import tool_forge
    except Exception as exc:
        return None

    mt = (g.get("policy") or {}).get("memory_tools") or {}
    if not mt.get("enabled"):
        return None
    if not mt.get("allow_learned_tools", True):
        return None

    by_type = (diagnosis.get("base") or {}).get("by_type") or {}
    ranked = sorted(
        (t for t in tool_forge.TYPE_CONTRACTS if t in by_type),
        key=lambda t: by_type[t].get("acc", 1.0))
    qtype = ranked[0] if ranked else next(
        (t for t in tool_forge.TYPE_CONTRACTS
         if any(q.get("type") == t for q in questions)), None)
    if not qtype:
        return None

    same = [q for q in questions if q.get("type") == qtype and q.get("video_key")]
    if not same:
        return None
    video_key = same[0]["video_key"]

    try:
        spec = tool_forge.forge(qtype, video_key, cfg, model=model, attempts=attempts,
                                questions=questions, commit=False,
                                gate5_mode=gate5_mode, gate5_min_pass=gate5_min_pass)
    except Exception as exc:
        return None
    if not spec:
        return None

    ng = copy.deepcopy(g)
    ng.setdefault("policy", {}).setdefault("memory_tools", {})["allow_learned_tools"] = True
    ng["parent_version"] = g["version"]
    desc = (f"[tool] model-authored tool '{spec['name']}' for {qtype}"
            f" ({str(spec.get('desc', ''))[:60]})")
    if not spec.get("_gate5_enforced", True):
        desc += "[gate5 not enforced]"
    ng["note"] = desc
    return desc, ng, spec

def bootstrap_lower(base: list[int], cand: list[int], n_boot: int = 2000, alpha: float = 0.05) -> float:
    if not base or len(base) != len(cand):
        return -1.0
    rnd = random.Random(0)
    n = len(base)
    diffs = []
    for _ in range(n_boot):
        idx = [rnd.randrange(n) for _ in range(n)]
        diffs.append(sum(cand[i] - base[i] for i in idx) / n)
    diffs.sort()
    return diffs[int(n_boot * alpha / 2)]

def score_vector(recs: list[dict], qids: list[str], metric: str = "correct") -> list[int]:
    m = {r["qid"]: r for r in recs}
    if metric == "strict":
        return [int(bool(m[q].get("strict_correct"))) if q in m else 0 for q in qids]
    if metric == "hit":
        return [int(bool(m[q].get("gold_macro_hit"))) if q in m else 0 for q in qids]
    if metric != "correct":
        raise ValueError(f"unknown metric {metric!r}; use 'correct', 'strict' or 'hit'")
    return [0 if (q not in m or m[q].get("correct") is None) else int(m[q]["correct"])
            for q in qids]

MACRO_BUDGET_TOL = float(os.environ.get("VEH_MACRO_BUDGET_TOL", "0.10"))

FRAME_BUDGET_MB = float(os.environ.get("VEH_FRAME_BUDGET_MB", "6.0"))
CORE_FRAME_BUDGET_MB = float(os.environ.get("VEH_CORE_FRAME_BUDGET_MB", "12.0"))
FRAME_ROUTE_MAX_MIN = float(os.environ.get("VEH_FRAME_ROUTE_MAX_MIN", "30.0"))

def mean_frame_mb(recs: list[dict]) -> float:
    vals = [float(r.get("frames_sent_mb") or 0.0) for r in recs]
    return sum(vals) / len(vals) if vals else 0.0

def _touches_frames(g: dict, ng: dict) -> bool:
    def sub(d: dict) -> dict:
        return ((d.get("evidence") or {}).get("frames") or {}) if isinstance(d, dict) else {}

    a, b = sub(g), sub(ng)
    return {k: v for k, v in a.items() if not k.startswith("_")} != \
           {k: v for k, v in b.items() if not k.startswith("_")}

def _touches_memory_tools(g: dict, ng: dict) -> bool:
    def sub(d: dict) -> dict:
        return ((d.get("policy") or {}).get("memory_tools") or {}) if isinstance(d, dict) else {}

    a, b = sub(g), sub(ng)
    return {k: v for k, v in a.items() if not k.startswith("_")} != \
           {k: v for k, v in b.items() if not k.startswith("_")}

def _frame_groups(recs: list[dict]) -> dict[str, float]:
    buckets: dict[str, list[float]] = {"core": [], "memory": []}
    for r in recs:
        key = "core" if (r.get("route") or "") in eval_router.CORE_ROUTES else "memory"
        buckets[key].append(float(r.get("frames_sent_mb") or 0.0))
    return {k: (sum(v) / len(v) if v else 0.0) for k, v in buckets.items()}

def frame_budget_ok(cand_recs: list[dict],
                    tol_mb: float | None = None,
                    core_tol_mb: float | None = None,
                    ) -> tuple[bool, float, dict, list[str]]:
    tol = FRAME_BUDGET_MB if tol_mb is None else tol_mb
    core_tol = CORE_FRAME_BUDGET_MB if core_tol_mb is None else core_tol_mb
    groups = _frame_groups(cand_recs)
    reasons = []
    for arm, mb, cap in (("core", groups["core"], core_tol),
                         ("memory", groups["memory"], tol)):
        if mb > cap:
            reasons.append(f"{arm} arm {mb:.2f} MiB/q > {cap:.1f} MiB cap")
    return (not reasons), mean_frame_mb(cand_recs), groups, reasons

def mean_macros(recs: list[dict]) -> float:
    vals = [len(r.get("located") or []) for r in recs if r.get("located") is not None]
    return sum(vals) / len(vals) if vals else 0.0

def macro_budget_ok(base_recs: list[dict], cand_recs: list[dict],
                    tol: float | None = None) -> tuple[bool, float]:
    t = MACRO_BUDGET_TOL if tol is None else tol
    b, c = mean_macros(base_recs), mean_macros(cand_recs)
    growth = (c - b) / b if b else 0.0
    return growth <= t, growth

def grounded_mask(base_recs: list[dict], qids: list[str]) -> list[bool]:
    m = {r["qid"]: r for r in base_recs}
    keep = []
    for q in qids:
        r = m.get(q)
        if r is None:
            keep.append(True)
            continue
        ao = r.get("anchor_ok")
        keep.append(True if ao is None else not (r.get("correct") and not ao))
    return keep

def gate_report(base_recs: list[dict], cand_recs: list[dict], qids: list[str],
                gates: dict | None = None) -> dict:
    gates = gates or {}
    pb = gates.get("paired_bootstrap") or {}
    boot = {"n_boot": pb.get("n_boot", 2000), "alpha": pb.get("alpha", 0.05)}
    mb = gates.get("macro_budget") or {}
    metric = pb.get("metric", GATE_METRIC)

    out: dict[str, dict] = {}
    for name, m in (("correct", "correct"), ("strict", "strict"), ("hit", "hit")):
        bv = score_vector(base_recs, qids, m)
        cv = score_vector(cand_recs, qids, m)
        out[name] = _flip_stats(bv, cv, qids, **boot)

    tol = mb.get("tol", MACRO_BUDGET_TOL)
    ok, growth = macro_budget_ok(base_recs, cand_recs, tol)
    out["hit"]["macro_growth"] = round(growth, 4)
    out["hit"]["macro_budget_ok"] = ok
    out["hit"]["macro_budget_mode"] = mb.get("mode", "full")
    if not ok and mb.get("mode", "full") == "full":
        out["hit"]["passed"] = False
        out["hit"]["overruled"] = (
            f"retrieved segments grew {growth:+.1%} (> {tol:.0%}); "
            "hit gain attributed to volume, not accuracy"
        )
    elif not ok:
        out["hit"]["not_overruled_because"] = (
            f"gates.macro_budget.mode={mb.get('mode')}: the segment count grew {growth:+.1%}, "
            f"exceeding {tol:.0%}, but this round does not block (acceptance can only be provisional)")

    keep = grounded_mask(base_recs, qids)
    sub = [q for q, k in zip(qids, keep) if k]
    if sub:
        bv = score_vector(base_recs, sub, "correct")
        cv = score_vector(cand_recs, sub, "correct")
        out["grounded"] = _flip_stats(bv, cv, sub, **boot)
        out["grounded"]["n_dropped"] = len(qids) - len(sub)

    routes = route_map_from_records(base_recs + cand_recs)
    if routes:
        m = metric if metric in ("correct", "strict", "hit") else "correct"
        bv = score_vector(base_recs, qids, m)
        cv = score_vector(cand_recs, qids, m)
        out["by_route"] = _flip_stats_by_route(bv, cv, qids, routes)
    return out

BASELINE_RUNS = int(os.environ.get("VEH_BASELINE_RUNS", "1"))

_MAJORITY_FLAGS = ("correct", "strict_correct", "anchor_ok", "gold_macro_hit")

def majority_recs(runs: list[list[dict]], qids: list[str]) -> list[dict]:
    if not runs:
        return []
    if len(runs) == 1:
        return runs[0]
    by_q: dict[str, list[dict]] = {q: [] for q in qids}
    for recs in runs:
        for r in recs:
            if r["qid"] in by_q:
                by_q[r["qid"]].append(r)

    out = []
    for q in qids:
        cands = by_q.get(q) or []
        if not cands:
            continue
        vote = {f: sum(bool(r.get(f)) for r in cands) * 2 > len(cands)
                for f in _MAJORITY_FLAGS}
        rep = max(cands, key=lambda r: sum(bool(r.get(f)) == vote[f]
                                           for f in _MAJORITY_FLAGS))
        merged = dict(rep)
        merged.update(vote)
        merged["_majority_of"] = len(cands)
        out.append(merged)
    return out

def run_baseline(ho: list[dict], g: dict, cfg: dict, workers: int,
                 n_runs: int = None, quiet: bool = True) -> tuple[list[dict], dict]:
    k = BASELINE_RUNS if n_runs is None else n_runs
    if k <= 1:
        recs = eval_router.run_pool(ho, g, cfg, workers, quiet=quiet,
                                    cache_tag="baseline")
        return recs, {}

    qids = [q["qid"] for q in ho]
    runs, accs, hits = [], [], []
    for i in range(k):
        recs = eval_router.run_pool(ho, g, cfg, workers, quiet=quiet,
                                    cache_tag=f"baseline#{i}")
        s = eval_router.summarize_mixed(recs)
        runs.append(recs)
        accs.append(s["acc"])
        hits.append(s["gold_macro_hit_rate"])

    merged = majority_recs(runs, qids)
    n = len(qids) or 1
    drift = {
        "n_runs": k,
        "accs": accs,
        "hits": hits,
        "acc_spread": round(max(accs) - min(accs), 4),
        "acc_spread_questions": round((max(accs) - min(accs)) * n, 1),
        "hit_spread": round(max(hits) - min(hits), 4),
        "majority_acc": round(sum(bool(r.get("correct")) for r in merged) / n, 4),
    }
    return merged, drift

GATE_METRIC = os.environ.get("VEH_GATE_METRIC", "correct").strip().lower()
if GATE_METRIC not in ("correct", "strict", "hit"):
    raise SystemExit(f"VEH_GATE_METRIC={GATE_METRIC!r} is invalid; choose correct / strict / hit")

def drift_ok(lower: float, drift: dict) -> tuple[bool, str]:
    if not drift or drift.get("n_runs", 1) <= 1:
        return True, ""
    spread = drift.get("acc_spread", 0.0)
    if lower <= spread:
        return False, (f"the improvement lower bound {lower:+.4f} does not exceed the baseline's own drift {spread:.4f}"
                       f"({drift.get('acc_spread_questions', 0):.0f} items); "
                       "this 'pass' may merely reflect a baseline that happened to run low")
    return True, ""

def subsample_heldout(ho: list[dict], limit: int) -> list[dict]:
    if limit <= 0 or len(ho) <= limit:
        return ho
    by_route: dict = {}
    for q in ho:
        by_route.setdefault(q.get("route") or "_none", []).append(q)

    order = sorted(by_route, key=lambda r: (-len(by_route[r]), r))
    picked: list[dict] = []
    for idx, route in enumerate(order):
        group = sorted(by_route[route], key=lambda q: str(q.get("qid")))
        remaining_groups = len(order) - idx - 1
        want = max(1, round(limit * len(group) / len(ho)))
        want = min(want, len(group), max(1, limit - len(picked) - remaining_groups))
        picked.extend(group[:want])

    if len(picked) < limit:
        have = {q.get("qid") for q in picked}
        for route in order:
            for q in sorted(by_route[route], key=lambda q: str(q.get("qid"))):
                if len(picked) >= limit:
                    break
                if q.get("qid") not in have:
                    picked.append(q)
                    have.add(q.get("qid"))
            if len(picked) >= limit:
                break
    return sorted(picked[:limit], key=lambda q: str(q.get("qid")))

def gate_verdict(report: dict, drift: dict = None, gates: dict | None = None
                 ) -> tuple[bool, float, str]:
    gates = gates or {}
    pb = gates.get("paired_bootstrap") or {}
    bd = gates.get("baseline_drift") or {}
    metric = pb.get("metric", GATE_METRIC)
    v = report.get(metric) or {}
    lower = v.get("bootstrap_lower", -1.0)
    mode = pb.get("mode", "full")
    if mode == "acc_only":
        base_acc = v.get("base_acc")
        cand_acc = v.get("cand_acc")
        if base_acc is None or cand_acc is None:
            return False, lower, ("[gate acc_only] the report carries no base_acc/cand_acc,"
                                  "so accuracy improvement cannot be judged; treated as a failure")
        gained = cand_acc > base_acc
        delta = cand_acc - base_acc
        why = (f"[gate acc_only] acc {base_acc:.4f} -> {cand_acc:.4f}"
               f"({delta:+.4f}); the lower bound {lower:+.4f} is recorded but not used as a criterion, "
               "acceptance this round can only be provisional and must be re-measured on another question pool")
        if not gained:
            return False, lower, f"{why} -- accuracy did not improve"
        return True, lower, why
    if mode != "full":
        return True, lower, (f"[gate relaxed] paired_bootstrap.mode={mode}: "
                             f"the lower bound {lower:+.4f} is recorded but not used as an admission criterion, "
                             "acceptance this round can only be provisional")
    if v.get("overruled"):
        return False, lower, v["overruled"]
    if not v.get("passed"):
        return False, lower, ""
    if bd.get("mode", "full") != "full":
        return True, lower, (f"[gate relaxed] baseline_drift.mode={bd.get('mode')}: "
                             "never compared against the baseline's own drift")
    ok, why = drift_ok(lower, drift or {})
    if not ok:
        return False, lower, why
    return True, lower, ""

def _flip_stats(base_vec: list[int], cand_vec: list[int], qids: list[str],
                n_boot: int = 2000, alpha: float = 0.05) -> dict:
    lower = bootstrap_lower(base_vec, cand_vec, n_boot=n_boot, alpha=alpha)
    fixed = [q for q, b, c in zip(qids, base_vec, cand_vec) if not b and c]
    broken = [q for q, b, c in zip(qids, base_vec, cand_vec) if b and not c]
    n = len(qids) or 1
    return {
        "n": len(qids),
        "base_acc": round(sum(base_vec) / n, 4),
        "cand_acc": round(sum(cand_vec) / n, 4),
        "bootstrap_lower": lower,
        "passed": lower > 0,
        "n_fixed": len(fixed),
        "n_broken": len(broken),
        "fixed_qids": fixed,
        "broken_qids": broken,
    }

def _flip_stats_by_route(base_vec: list[int], cand_vec: list[int], qids: list[str],
                         route_of_qid: dict[str, str]) -> dict:
    groups: dict[str, list[int]] = {}
    for i, qid in enumerate(qids):
        groups.setdefault(route_of_qid.get(qid) or "unrouted", []).append(i)
    out: dict[str, dict] = {}
    for route, idx in sorted(groups.items()):
        b = [base_vec[i] for i in idx]
        c = [cand_vec[i] for i in idx]
        sub = _flip_stats(b, c, [qids[i] for i in idx])
        sub["net"] = sub["n_fixed"] - sub["n_broken"]
        sub["underpowered"] = sub["n"] < 100
        sub.pop("fixed_qids", None)
        sub.pop("broken_qids", None)
        out[route] = sub
    return out

def route_map_from_records(records: list[dict], questions: list[dict] | None = None) -> dict[str, str]:
    out: dict[str, str] = {}
    for q in (questions or []):
        if q.get("qid") and q.get("route"):
            out[q["qid"]] = q["route"]
    for r in records:
        if r.get("qid") and r.get("route"):
            out[r["qid"]] = r["route"]
    return out

def _is_questioner_patch(g_old: dict, g_new: dict) -> bool:
    for k in set(g_old) | set(g_new):
        if k in ("version", "parent_version", "note", "questioner"):
            continue
        if g_new.get(k) != g_old.get(k):
            return False
    return g_new.get("questioner") != g_old.get("questioner")

def _evaluate_questioner_genome(
    genome: dict,
    workers: int,
    seed: int,
    n_questions: int,
    label: str,
    solver_by_type: dict[str, object],
    videos: list[str] | None = None,
) -> dict:
    run_id = f"qeval_{int(time.time())}_{label}_{abs(hash(json.dumps(genome, sort_keys=True))) % 100000}"
    work = STORE / "runs" / run_id
    store = work / "store"
    work.mkdir(parents=True, exist_ok=False)
    store.mkdir()
    genome_path = work / "genome.json"
    genome_path.write_text(json.dumps(genome, ensure_ascii=False, indent=2), encoding="utf-8")
    env = {
        **os.environ,
        "VEH_GENOME_PATH": str(genome_path),
        "VEH_STORE_DIR": str(store),
        "VEH_REWRITE_CACHE": str(STORE / "runs" / "rewrite_cache"),
        "VEH_CONFIG_FILE": str(VEH_DIR.parent / "qwen-mm-config"),
    }
    ask_cmd = [sys.executable, str(VEH_DIR / "ask.py"),
               "--n", str(n_questions), "--seed", str(seed)]
    if videos:
        ask_cmd += ["--videos", *videos]
    commands = (
        ask_cmd,
        [sys.executable, str(VEH_DIR / "verify.py"), "--workers", str(workers)],
    )
    for cmd in commands:
        code, _tail, err_tail = common_mod.run_streaming(
            cmd, cwd=str(VEH_DIR), env=env, timeout=7200, keep_tail=60,
        )
        if code:
            raise RuntimeError(
                f"questioner evaluation failed: {' '.join(cmd)}\n" + "\n".join(err_tail[-20:])
            )
    verdicts = load_questions(store / "questions_all_verdict.jsonl")
    result = qscore_mod.score(verdicts, solver_by_type=solver_by_type, g=genome)
    (work / "qscore.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    result["artifact_dir"] = str(work)
    return result

def evolve_questioner_round(
    g: dict,
    cfg: dict,
    workers: int,
    solver_by_type: dict[str, object],
    qreport: dict | None,
    n_questions: int | None = None,
    seed: int | None = None,
    apply_patch: bool = True,
    solver_trace_by_type: dict[str, dict] | None = None,
    per_kind_cap: int = 0,
    videos: list[str] | None = None,
) -> tuple[dict, dict]:
    report: dict = {"stage": "questioner", "candidates": [], "accepted": None,
                    "from_version_axes": version_axes(g)}

    difficulty = solver_trace_by_type or solver_by_type
    report["zpd_input"] = "trace" if solver_trace_by_type else "acc_only"

    cands = propose_questioner(g, qreport, solver_trace_by_type,
                               per_rule_cap=per_kind_cap)
    if not cands:
        why = "no questioner candidate (Blind report missing or nothing anomalous)"
        report["skipped"] = why
        return g, report
    if per_kind_cap > 0:
        report["per_rule_cap"] = per_kind_cap

    qgate = gate_cfg(g, "questioner_qscore")
    min_delta = qgate.get("min_delta", 0.0)
    report["gate"] = dict(qgate)

    qs_all = load_questions(STORE / "questions.jsonl") if (STORE / "questions.jsonl").exists() else []
    n = n_questions or max(16, min(len(qs_all) or 48, 96))
    sd = seed if seed is not None else 104729 + int(g["version"])

    for desc, _ in cands:
        pass

    base_q: dict | None = None
    passed: list[tuple[str, dict, dict, dict]] = []
    for desc, ng in cands:
        try:
            if base_q is None:
                base_q = _evaluate_questioner_genome(g, workers, sd, n, "baseline",
                                                     difficulty, videos=videos)
            cand_q = _evaluate_questioner_genome(ng, workers, sd, n, "candidate",
                                                 difficulty, videos=videos)
            delta = cand_q["score"] - base_q["score"]
            report["candidates"].append({"desc": desc, "delta": round(delta, 4),
                                         "baseline_qscore": base_q, "candidate_qscore": cand_q})
            if delta > min_delta:
                passed.append((desc, ng, cand_q, base_q))
            elif qgate["mode"] != "full":
                report["candidates"][-1]["would_reject"] = True
                passed.append((desc, ng, cand_q, base_q))
        except Exception as exc:
            report["candidates"].append({"desc": desc, "error": str(exc)[:300]})

    if not passed:
        return g, report

    desc, ng, cand_q, base_q = max(passed, key=lambda it: it[2]["score"] - it[3]["score"])
    provisional = qgate["mode"] != "full"
    if not apply_patch:
        report["accepted"] = {"desc": desc, "applied": False,
                              "baseline_qscore": base_q, "candidate_qscore": cand_q}
        return g, report

    bump_version(ng, "questioner")
    if provisional:
        ng["provisional"] = True
        ng["provisional_reason"] = (
            f"questioner accepted while questioner_qscore.mode={qgate['mode']} (not judged under the formal criterion)")
    snapshot_genome(ng)
    save_genome(ng)
    if provisional:
        pass
    report["accepted"] = {"desc": desc, "new_version": ng["version"], "kind": "questioner",
                          "applied": True, "new_version_axes": version_axes(ng),
                          "provisional": provisional,
                          "baseline_qscore": base_q, "candidate_qscore": cand_q}
    return ng, report

def _apply_candidate_cap(cands, cap, kind_of=None):
    n_proposed = len(cands)
    if not cap or n_proposed <= cap:
        return list(cands), n_proposed
    if kind_of is None:
        return list(cands)[:cap], n_proposed

    buckets: dict[str, list] = {}
    order: list[str] = []
    for c in cands:
        k = kind_of.get(c[0], "?")
        if k not in buckets:
            buckets[k] = []
            order.append(k)
        buckets[k].append(c)

    kept = []
    idx = 0
    while len(kept) < cap:
        progressed = False
        for k in order:
            if idx < len(buckets[k]):
                kept.append(buckets[k][idx])
                progressed = True
                if len(kept) >= cap:
                    break
        if not progressed:
            break
        idx += 1
    return kept, n_proposed

def _legacy_capability_overrides(use_llm: bool | None,
                                 rule_candidates: bool | None,
                                 strategy_forge: bool | None,
                                 capabilities: dict | None) -> dict[str, str]:
    out: dict[str, str] = dict(capabilities or {})
    if use_llm is not None or rule_candidates is not None:
        want_llm, want_rule = bool(use_llm), bool(rule_candidates)
        mode = ("both" if want_llm and want_rule else
                "llm" if want_llm else
                "rule" if want_rule else "off")
        for kind in ("hyperparam", "skill", "tool_policy"):
            out.setdefault(kind, mode)
    if strategy_forge is not None:
        out.setdefault("strategy_code", "llm" if strategy_forge else "off")
    return out

_T0 = time.time()

def stamp(msg: str = "") -> str:
    el = time.time() - _T0
    return f"[{time.strftime('%H:%M:%S')} +{int(el) // 60:02d}:{int(el) % 60:02d}] {msg}"

def _archive_candidate_code(spec: dict | None, spec_kind: str, desc: str,
                            entry: dict) -> None:
    if not spec or not spec.get("code"):
        return
    if not spec.get("_smoke_tested"):
        return
    try:
        name = re.sub(r"[^\w\-]+", "_", str(spec.get("name") or spec_kind)).strip("_")[:60]
        name = name or spec_kind
        verdict = "pass" if entry.get("gate_passed") else "reject"
        d = (STORE / "runs" / "candidate_code"
             / f"{time.strftime('%Y%m%d_%H%M%S')}_{spec_kind}_{verdict}_{name}")
        d.mkdir(parents=True, exist_ok=True)
        (d / "code.py").write_text(spec["code"], encoding="utf-8")
        (d / "spec.json").write_text(json.dumps(
            {k: v for k, v in spec.items() if k != "code"},
            ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        (d / "verdict.json").write_text(json.dumps({
            "desc": desc, "kind": spec_kind,
            "smoke_tested": spec.get("_smoke_tested"),
            "gate5_passed": spec.get("_gate5_passed"),
            "gate5_enforced": spec.get("_gate5_enforced"),
            "gate5_note": spec.get("_cross_note") or spec.get("_gate5_note"),
            "gate5_stats": spec.get("_cross_stats"),
            **entry,
        }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    except Exception as exc:
        pass

def _genome_diff(old: dict, new: dict, prefix: str = "") -> dict:
    out: dict = {}
    for k in sorted(set(old) | set(new)):
        if k in ("version", "parent_version", "note"):
            continue
        a, b = old.get(k, "<missing>"), new.get(k, "<missing>")
        path = f"{prefix}{k}"
        if isinstance(a, dict) and isinstance(b, dict):
            out.update(_genome_diff(a, b, path + "."))
        elif a != b:
            out[path] = {"from": a, "to": b}
    return out

def _archive_candidate_patch(desc: str, ng: dict | None, base: dict | None,
                             entry: dict) -> None:
    if ng is None:
        return
    try:
        patch = _genome_diff(base or {}, ng) if base else {}
        if base and not patch:
            return
        slug = re.sub(r"[^\w\-]+", "_", str(desc)).strip("_")[:60] or "param"
        verdict = "pass" if entry.get("gate_passed") else "reject"
        d = (STORE / "runs" / "candidate_patch"
             / f"{time.strftime('%Y%m%d_%H%M%S')}_param_{verdict}_{slug}")
        d.mkdir(parents=True, exist_ok=True)
        (d / "patch.json").write_text(json.dumps(
            {"desc": desc, "n_changed": len(patch), "patch": patch},
            ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        (d / "genome.json").write_text(json.dumps(
            ng, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        (d / "verdict.json").write_text(json.dumps(
            {"desc": desc, "kind": "param", **entry},
            ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    except Exception as exc:
        pass

def evolve_round(
    g: dict,
    cfg: dict,
    workers: int,
    questions_path: Path,
    use_llm: bool | None = None,
    evolver_model: str | None = None,
    verdict_path: Path | None = None,
    strategy_forge: bool | None = None,
    strategy_attempts: int = 3,
    strategy_min_per_type: int | None = None,
    rule_candidates: bool | None = None,
    core_frames: bool = False,
    candidate_cap: int = 0,
    capabilities: dict | None = None,
    allow_rule_fallback: bool = False,
    tool_attempts: int = 3,
    code_every: int = 1,
    round_index: int = 0,
    per_kind_cap: int = 0,
    shard_questions: list[dict] | None = None,
    heldout_limit: int = 0,
) -> tuple[dict, dict]:
    overrides = _legacy_capability_overrides(
        use_llm, rule_candidates, strategy_forge, capabilities)

    caps = resolve_capabilities(g, overrides)
    gates = {name: gate_cfg(g, name) for name in GATE_NAMES}
    relaxed = [n for n, c in gates.items() if c["mode"] != "full"]

    ev = load_questions(questions_path, "evolve")
    ho = load_questions(questions_path, "heldout")
    if not ev or not ho:
        raise SystemExit("evolve/heldout pools are empty; run ask.py + verify.py first")

    n_ho_full = len(ho)
    if heldout_limit and heldout_limit < n_ho_full:
        ho = subsample_heldout(ho, heldout_limit)
        import collections as _c
        _dist = _c.Counter(q.get("route") for q in ho)

    n_ev_full = len(ev)
    if shard_questions is not None:
        if not shard_questions:
            raise SystemExit("the evolve pool for this shard is empty: check that shards.json matches the question pool")
        ev = shard_questions

    if relaxed:
        pass

    _t = time.time()
    d = diag_mod.diagnose(ev, g, cfg, workers)

    solver_trace_by_type = d.get("trace_by_type") or {}
    solver_by_type = {t: v["acc"] for t, v in solver_trace_by_type.items()
                      if v.get("acc") is not None}

    qreport = None
    qs_all: list[dict] = []
    vpath = verdict_path or (STORE / "questions_all_verdict.jsonl")
    if vpath.exists():
        qs_all = load_questions(vpath)
        if qs_all and any("verdict" in q for q in qs_all):
            qreport = {
                "summary": qscore_mod.score(qs_all, solver_by_type=solver_trace_by_type, g=g),
                "by_type": qscore_mod.per_type_report(qs_all),
            }

    ho_ids = [q["qid"] for q in ho]

    cands: list[tuple[str, dict]] = []
    proposer_log: dict[str, int] = {}
    kind_of: dict[str, str] = {}

    def _add(name: str, items: list[tuple[str, dict]]) -> None:
        if per_kind_cap > 0 and len(items) > per_kind_cap:
            items = items[:per_kind_cap]
        proposer_log[name] = len(items)
        for desc, _ng in items:
            kind_of.setdefault(desc, name)
        cands.extend(items)

    llm_n = per_kind_cap if per_kind_cap > 0 else 4

    def _rule_candidates() -> list[tuple[str, dict]]:
        out = propose(g, d)
        out += propose_policy(g, d, d.get("base_records"))
        return out

    if caps["hyperparam"] in ("rule", "both") or caps["skill"] in ("rule", "both"):
        _add("rule", _rule_candidates())
    if caps["hyperparam"] in ("llm", "both"):
        _add("hyperparam_llm", propose_llm(g, d, qreport, cfg, model=evolver_model,
                                           max_candidates=llm_n, core_frames=core_frames))
    if caps["skill"] in ("llm", "both"):
        _add("skill_llm", propose_policy_llm(
            g, d, d.get("base_records"), cfg, model=evolver_model, max_candidates=llm_n))

    if caps["tool_policy"] == "off":
        kept, dropped = [], 0
        for c in cands:
            if _touches_memory_tools(g, c[1]):
                dropped += 1
            else:
                kept.append(c)
        if dropped:
            pass
        cands[:] = kept

    if not core_frames:
        kept, dropped = [], 0
        for c in cands:
            if _touches_frames(g, c[1]):
                dropped += 1
            else:
                kept.append(c)
        if dropped:
            pass
        cands[:] = kept

    strategy_specs: dict[str, dict] = {}
    tool_specs: dict[str, dict] = {}

    code_turn = code_every > 0 and (round_index % code_every == 0)
    if caps["strategy_code"] != "off" and not code_turn:
        pass
    if caps["strategy_code"] != "off" and code_turn:
        g5 = gates["strategy_gate5"]
        st = propose_strategy_llm(
            cfg, g, d, ev, attempts=strategy_attempts,
            min_per_type=(strategy_min_per_type if strategy_min_per_type is not None
                          else g5.get("min_per_type", STRATEGY_MIN_PER_TYPE)),
            gate5_mode=g5["mode"], churn_max=g5.get("churn_max"))
        if st is not None:
            desc, ng_s, spec = st
            strategy_specs[desc] = spec
            _add("strategy_code", [(desc, ng_s)])
        else:
            proposer_log["strategy_code"] = 0

    if caps["tool_code"] != "off" and code_turn:
        g5t = gates["tool_gate5"]
        tl = propose_tool_llm(cfg, g, d, ev, attempts=tool_attempts,
                              gate5_mode=g5t["mode"],
                              gate5_min_pass=g5t.get("min_pass", 0.75),
                              model=evolver_model)
        if tl is not None:
            desc, ng_t, spec = tl
            tool_specs[desc] = spec
            _add("tool_code", [(desc, ng_t)])
        else:
            proposer_log["tool_code"] = 0

    no_candidates_reason = None
    if not cands and allow_rule_fallback:
        _add("rule_fallback", _rule_candidates())
    if not cands:
        enabled = [k for k, v in caps.items() if v != "off" and k != "questioner"]
        no_candidates_reason = (
            f"0 candidates this round. Enabled categories: {enabled or '(all off)'}; "
            f"per-proposer output: {json.dumps(proposer_log, ensure_ascii=False)}."
            "possible causes: the model call failed / every patch was rejected by validate_patch or semantic_guard / "
            "there is nothing to propose for this category. Pass --allow-rule-fallback to fall back to rule candidates.")

    seen, uniq = set(), []
    for desc, ng in cands:
        if desc not in seen:
            seen.add(desc)
            uniq.append((desc, ng))
    cands = uniq
    cap = candidate_cap or MAX_EVAL_CANDIDATES
    cands, n_proposed = _apply_candidate_cap(cands, cap, kind_of)
    if n_proposed > len(cands):
        kept_by_kind: dict[str, int] = {}
        for desc, _ in cands:
            k = kind_of.get(desc, "?")
            kept_by_kind[k] = kept_by_kind.get(k, 0) + 1
        dropped_kinds = [k for k, n in proposer_log.items()
                         if n > 0 and kept_by_kind.get(k, 0) == 0]
        if dropped_kinds:
            pass

    for desc, _ in cands:
        pass

    if cands:
        _tb = time.time()
        base_recs, base_drift = run_baseline(ho, g, cfg, workers)
        base_sum = eval_router.summarize_mixed(base_recs)
        base_vec = score_vector(base_recs, ho_ids)
        if base_sum.get("n_timeout"):
            pass
        if base_drift:
            pass
    else:
        base_recs, base_drift, base_sum, base_vec = [], None, {}, []

    best = None
    flips: list[dict] = []

    def _eval_solver(desc: str, ng: dict, spec: dict | None,
                     spec_kind: str = "strategy") -> tuple[dict, tuple | None]:
        import tool_forge as _tf

        if spec is not None and spec_kind == "strategy":
            solve.PENDING_STRATEGY = spec
        elif spec is not None and spec_kind == "tool":
            _tf.PENDING_TOOL = spec
        try:
            recs = eval_router.run_pool(ho, ng, cfg, workers, quiet=True,
                                        cache_tag="heldout",
                                        dump_tag=f"cand:{desc[:60]}")
        finally:
            solve.PENDING_STRATEGY = None
            _tf.PENDING_TOOL = None
        s = eval_router.summarize_mixed(recs)
        cand_vec = score_vector(recs, ho_ids)
        pbg = gates["paired_bootstrap"]
        lower = bootstrap_lower(base_vec, cand_vec,
                                n_boot=pbg.get("n_boot", 2000),
                                alpha=pbg.get("alpha", 0.05))
        fixed = [q for q, b, c in zip(ho_ids, base_vec, cand_vec) if not b and c]
        broken = [q for q, b, c in zip(ho_ids, base_vec, cand_vec) if b and not c]
        alt = gate_report(base_recs, recs, ho_ids, gates=gates)
        passed, gate_lower, overruled = gate_verdict(alt, base_drift, gates=gates)
        fbg = gates["frame_budget"]
        frames_ok, frame_mb, frame_groups, frame_reasons = frame_budget_ok(
            recs,
            tol_mb=fbg.get("memory_mb"),
            core_tol_mb=fbg.get("core_mb"),
        ) if fbg["mode"] != "bypass" else (True, mean_frame_mb(recs), _frame_groups(recs), [])
        if not frames_ok and fbg["mode"] == "full":
            passed = False
            overruled = ("frame payload over ceiling per arm: " + "; ".join(frame_reasons)
                         + " -- gain attributed to payload, not method")
        elif not frames_ok:
            pass
        entry = {
            "desc": desc,
            "acc": s["acc"],
            "hit": s["gold_macro_hit_rate"],
            "bootstrap_lower": lower,
            "gate_metric": gates["paired_bootstrap"].get("metric", GATE_METRIC),
            "gate_lower": gate_lower,
            "gate_passed": passed,
            "gates_relaxed": relaxed or None,
            "gate_note": overruled or None,
            "frame_mb_mean": round(frame_mb, 3),
            "frame_mb_by_arm": {k: round(v, 3) for k, v in frame_groups.items()},
            "frame_budget_ok": frames_ok,
            "pool_by_route": eval_router.route_counts(recs),
            "n_measurable": s["n_measurable"],
            "n_fixed": len(fixed),
            "n_broken": len(broken),
            "fixed_qids": fixed,
            "broken_qids": broken,
            "alt_metrics": {
                k: {kk: vv for kk, vv in v.items()
                    if kk not in ("fixed_qids", "broken_qids")}
                for k, v in alt.items()
            },
        }
        churn = f" | fixed {len(fixed)} broken {len(broken)}" if (fixed or broken) else ""
        gr = alt.get("grounded") or {}
        alt_txt = ""
        if gr:
            alt_txt = (f"\n      control, after dropping the {gr.get('n_dropped', 0)} items the baseline guessed right: "
                       f"lower bound {gr['bootstrap_lower']:+.4f} "
                       f"(fixed {gr['n_fixed']} broken {gr['n_broken']})"
                       f" | strict verdict: lower bound {alt['strict']['bootstrap_lower']:+.4f} "
                       f"(fixed {alt['strict']['n_fixed']} broken {alt['strict']['n_broken']})")
        hv = alt.get("hit") or {}
        if hv:
            alt_txt += (f"\n      retrieval hit: lower bound {hv['bootstrap_lower']:+.4f} "
                        f"(fixed {hv['n_fixed']} broken {hv['n_broken']}, "
                        f"segments {hv.get('macro_growth', 0):+.1%})")
            if hv.get("overruled"):
                alt_txt += f"\n      [overruled] {hv['overruled']}"
        if overruled:
            pass
        _archive_candidate_code(spec, spec_kind, desc, entry)
        if spec is None or not spec.get("code"):
            _archive_candidate_patch(desc, ng, g, entry)
        return entry, ((desc, ng, s, gate_lower) if passed else None)

    plain = [(d, ng) for d, ng in cands
             if d not in tool_specs and d not in strategy_specs]
    coded = [(d, ng) for d, ng in cands
             if d in tool_specs or d in strategy_specs]

    def _run_one(desc, ng):
        spec_kind = "tool" if desc in tool_specs else "strategy"
        spec = tool_specs.get(desc) or strategy_specs.get(desc)
        _te = time.time()
        entry, cand_best = _eval_solver(desc, ng, spec, spec_kind)
        entry["kind"] = ("tool_code" if desc in tool_specs else
                         "strategy_code" if desc in strategy_specs else "patch")
        entry["eval_seconds"] = round(time.time() - _te, 1)
        return entry, cand_best

    cw = _cand_workers()
    results: list[tuple[dict, object]] = []
    if cw > 1 and len(plain) > 1:
        _tp = time.time()
        with ThreadPoolExecutor(max_workers=min(cw, len(plain))) as pool:
            futs = {pool.submit(_run_one, d, ng): d for d, ng in plain}
            for fut in as_completed(futs):
                results.append(fut.result())
        order = {d: i for i, (d, _) in enumerate(plain)}
        results.sort(key=lambda r: order.get(r[0]["desc"], 1 << 30))
    else:
        for desc, ng in plain:
            r = _run_one(desc, ng)
            results.append(r)

    for desc, ng in coded:
        r = _run_one(desc, ng)
        results.append(r)

    for entry, cand_best in results:
        flips.append(entry)
        if cand_best is not None and (best is None or cand_best[2]["acc"] > best[2]["acc"]):
            best = cand_best

    report = {
        "from_version": g["version"],
        "from_version_axes": version_axes(g),
        "diagnosis": d["diagnosis"],
        "baseline": base_sum,
        "baseline_drift": base_drift or None,
        "gate_metric": GATE_METRIC,
        "question_store_quality": (qreport or {}).get("summary"),
        "qreport": qreport,
        "n_candidates_proposed": n_proposed,
        "n_candidates_in_scope": len(cands),
        "candidate_cap": cap or None,
        "candidates": [c[0] for c in cands],
        "candidate_flips": flips,
        "stage": "solver",
        "solver_by_type": solver_by_type,
        "solver_trace_by_type": solver_trace_by_type,
        "solver_by_route": {r: v["acc"] for r, v in (d.get("base_acc_by_route") or {}).items()
                            if isinstance(v, dict) and v.get("acc") is not None},
        "capabilities": caps,
        "gates": {n: dict(c) for n, c in gates.items()},
        "gates_relaxed": relaxed or None,
        "proposer_counts": proposer_log,
        "no_candidates_reason": no_candidates_reason,
        "llm_proposer": any(v in ("llm", "both") for v in caps.values()),
        "llm_primary": all(v != "rule" for v in caps.values()),
        "strategy_forge": caps["strategy_code"] != "off",
        "tool_forge": caps["tool_code"] != "off",
        "evolver_model": evolver_model or cfg.get("QWEN_MM_API_VL_MODEL", "qwen3.7-plus"),
        "llm_audit": llm_audit(),
    }

    if best is None:
        report["accepted"] = None
        return g, report

    desc, ng, s, lower = best
    spec = strategy_specs.get(desc)
    if spec is not None:
        from strategy_forge import register as _register_strategy
        _register_strategy(spec, spec.get("_cross_note", ""), spec.get("_cross_stats") or {})
    tspec = tool_specs.get(desc)
    if tspec is not None:
        import tool_forge as _tf
        _tf.register(tspec, tspec.get("_gate5_note", ""))
    bump_version(ng, "solver")
    if relaxed:
        ng["provisional"] = True
        ng["provisional_reason"] = f"the following gates were not full at acceptance time: {relaxed}"
    else:
        ng.pop("provisional", None)
        ng.pop("provisional_reason", None)
    snapshot_genome(ng)
    save_genome(ng)
    if relaxed:
        pass
    report["accepted"] = {"desc": desc, "new_version": ng["version"], "summary": s,
                          "bootstrap_lower": lower, "kind": "solver",
                          "artifact": ("strategy" if spec else "tool" if tspec else None),
                          "provisional": bool(relaxed),
                          "new_version_axes": version_axes(ng)}
    return ng, report

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--questions", default=str(STORE / "questions_verified.jsonl"))
    ap.add_argument("--verdicts", default=None,
                    help="store WITH verdicts (default: store/questions_all_verdict.jsonl)")
    ap.add_argument("--dry-run", action="store_true", help="only print candidate patches")
    ap.add_argument("--llm-propose", action="store_const", const=True, default=None,
                    help="[legacy] ask the model to propose genome patches. **This is now the default**"
                         "(capabilities default to llm); the flag is kept for compatibility with existing scripts."
                         "Omitting it no longer disables anything -- the genome decides")
    ap.add_argument("--no-llm-propose", dest="llm_propose",
                    action="store_const", const=False,
                    help="Switch hyperparam/skill/tool_policy all to non-LLM (combine with "
                         "--rule-candidates to use rules, otherwise this is equivalent to off)")
    ap.add_argument(
        "--evolver-model",
        default=None,
        help="ABLATION ONLY: run the Evolver on a different (stronger) model. Leaving this "
             "unset keeps the run single-model, so all gains are attributable to the "
             "mechanism rather than to knowledge from a stronger model.",
    )
    ap.add_argument("--strategy-forge", action="store_const", const=True, default=None,
                    help="[legacy] equivalent to --capability strategy_code=llm. Without it the genome decides")
    ap.add_argument("--no-strategy-forge", dest="strategy_forge",
                    action="store_const", const=False,
                    help="equivalent to --capability strategy_code=off")
    ap.add_argument("--strategy-attempts", type=int, default=3,
                    help="how many times the model may retry writing a strategy")
    ap.add_argument("--strategy-min-per-type", type=int, default=None,
                    help=f"gate-5 questions per type (default {STRATEGY_MIN_PER_TYPE}, "
                         "defaults to gates.strategy_gate5.min_per_type)")
    ap.add_argument("--tool-attempts", type=int, default=3,
                    help="how many times the model may retry writing a tool")
    ap.add_argument("--rule-candidates", action="store_const", const=True, default=None,
                    help="[legacy] switch hyperparam/skill/tool_policy to rule-based proposers."
                         "Without it the genome decides (llm by default)")
    ap.add_argument("--capability", action="append", metavar="KIND=MODE",
                    help="Set the proposal permission per category; repeatable."
                         f"KIND in {common_mod.CAPABILITY_KINDS}, "
                         f"MODE in {common_mod.CAPABILITY_MODES}. "
                         "Example: --capability hyperparam=rule --capability tool_code=off")
    ap.add_argument("--gate", action="append", metavar="NAME[.KEY]=VALUE",
                    help="Set the criterion per gate; repeatable. NAME=full|record_only|bypass changes the mode,"
                         "NAME.KEY=VALUE changes a threshold."
                         "Example: --gate paired_bootstrap=record_only --gate macro_budget.tol=0.2."
                         f"available gates: {common_mod.GATE_NAMES}")
    ap.add_argument("--allow-rule-fallback", action="store_true",
                    help="Fall back to rule candidates when the model proposes none (off by default: an empty round"
                         "stays empty, and the cause is recorded in no_candidates_reason)")
    ap.add_argument("--per-kind-candidates", type=int, default=0,
                    help="Maximum candidates proposed per category (0 = unlimited). Unlike --max-candidates:\n"
                         "that one truncates the assembled list, and code candidates are appended last,\n"
                         "so lowering it removes the code category entirely; per-kind lets every category survive")
    ap.add_argument("--code-every", type=int, default=1,
                    help="Generate code-level candidates (strategies/tools) every N rounds. Default 1 = every round."
                         "Each needs one generation pass plus one full held-out run, making them the most expensive candidates")
    ap.add_argument("--max-candidates", type=int, default=0,
                    help="wiring/smoke knob: only evaluate the first N candidate patches this "
                         f"round (0 = no cap, default; env VEH_MAX_EVAL_CANDIDATES={MAX_EVAL_CANDIDATES}). "
                         "Reported as n_candidates_proposed vs n_candidates_in_scope")
    ap.add_argument("--core-frames", action="store_true",
                    help="allow evolution to search the direct-reading arm "
                         "(evidence.frames.*), i.e. appending an official read_video "
                         "skim on top of the retrieved text. Off by default because the "
                         "arm costs ~1000x the payload of text evidence")
    args = ap.parse_args()

    g, cfg = load_genome(), load_cfg()
    vpath = Path(args.verdicts) if args.verdicts else None
    cap_overrides = common_mod.parse_capability_overrides(args.capability)
    gate_overrides = common_mod.parse_gate_overrides(args.gate)
    if gate_overrides:
        g = copy.deepcopy(g)
        merged = dict(g.get("gates") or {})
        for name, patch in gate_overrides.items():
            merged[name] = {**(merged.get(name) or {}), **patch}
        g["gates"] = merged

    if args.evolver_model:
        pass

    if args.dry_run:
        qs = load_questions(Path(args.questions), "evolve")
        d = diag_mod.diagnose(qs, g, cfg, args.workers)
        qreport = None
        vp = vpath or (STORE / "questions_all_verdict.jsonl")
        if vp.exists():
            qs_all = load_questions(vp)
            if qs_all and any("verdict" in q for q in qs_all):
                qreport = {
                    "summary": qscore_mod.score(qs_all, g=g),
                    "by_type": qscore_mod.per_type_report(qs_all),
                }
        cands = propose(g, d) + propose_questioner(
            g, qreport, d.get("trace_by_type") or {})
        if args.llm_propose is not False:
            cands += propose_llm(g, d, qreport, cfg, model=args.evolver_model,
                                 core_frames=args.core_frames)
            cands += propose_policy_llm(
                g, d, d.get("base_records"), cfg, model=args.evolver_model
            )
        for desc, ng in cands:
            kind = "questioner" if _is_questioner_patch(g, ng) else "solver"
        return

    history = []
    for i in range(args.rounds):
        g, rep = evolve_round(
            g, cfg, args.workers, Path(args.questions),
            use_llm=args.llm_propose, evolver_model=args.evolver_model, verdict_path=vpath,
            strategy_forge=args.strategy_forge,
            strategy_attempts=args.strategy_attempts,
            strategy_min_per_type=args.strategy_min_per_type,
            rule_candidates=args.rule_candidates,
            core_frames=args.core_frames,
            candidate_cap=args.max_candidates,
            per_kind_cap=args.per_kind_candidates,
            capabilities=cap_overrides,
            allow_rule_fallback=args.allow_rule_fallback,
            tool_attempts=args.tool_attempts,
            code_every=args.code_every,
            round_index=i,
        )
        history.append(rep)
        if rep["accepted"] is None:
            break

    audit = llm_audit()
    if not audit["single_model"]:
        pass

    p = STORE / "runs" / "evolve_history.json"
    old = json.loads(p.read_text()) if p.exists() else []
    p.write_text(json.dumps(old + history, ensure_ascii=False, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
