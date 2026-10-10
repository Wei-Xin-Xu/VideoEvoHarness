from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import memory_tools
import video_routing
from common import (
    ROOT,
    VEH_DIR,
    STORE,
    ContentFilteredError,
    Memory,
    UpstreamTimeoutError,
    active_memory_dir,
    call_llm,
    fmt_time,
    jload,
    llm_cost,
    load_cfg,
    load_genome,
    load_questions,
    load_toolkit,
    parse_anchor,
    parse_choice,
    reset_cost,
)

EVIDENCE_MODES = ("retrieve", "none", "all", "gold")

def _parse_tr(tr) -> tuple[float, float] | None:
    def one(v) -> float | None:
        if isinstance(v, (int, float)):
            return float(v)
        s = str(v).strip()
        if not s:
            return None
        parts = s.split(":")
        try:
            if len(parts) == 3:
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
            if len(parts) == 2:
                return int(parts[0]) * 60 + float(parts[1])
            return float(s)
        except (TypeError, ValueError):
            return None

    if isinstance(tr, (list, tuple)) and len(tr) >= 2:
        a, b = one(tr[0]), one(tr[1])
    elif isinstance(tr, str) and "-" in tr:
        head, _, tail = tr.rpartition("-")
        a, b = one(head), one(tail)
    else:
        return None
    if a is None or b is None or b < a:
        return None
    return a, b

def locate(tk, question: str, g: dict) -> list[str]:
    return [mid for mid, _ in locate_scored(tk, question, g)]

def locate_scored(
    tk, question: str, g: dict, spans: dict | None = None, micro_ids: list[str] | None = None
) -> list[tuple[str, float]]:
    r = g["retrieval"]
    base_topk, mult = r["topk_macro"], r["search_multiplier"]
    adaptive = r.get("adaptive_topk") or {}
    topk = base_topk * (adaptive.get("max_multiplier", 1) if adaptive.get("enabled") else 1)

    scores: dict[str, float] = {}
    order: list[str] = []
    best_span: dict[str, tuple[float, float]] = {}
    nodes_by_macro: dict[str, list[str]] = {}

    def _keep_span(mid: str, item: dict, score: float, new: bool) -> None:
        if spans is None:
            return
        tr = _parse_tr(item.get("time_range"))
        if tr is None:
            return
        if str(item.get("node_type") or "") == "macro":
            return
        best_span[mid] = tr

    def _collect(raw):
        data = jload(raw)
        results = data.get("results", []) if isinstance(data, dict) else (data or [])
        for item in results:
            if not isinstance(item, dict):
                continue
            pm = item.get("parent_macro") or {}
            mid = pm.get("macro_id") or item.get("macro_id")
            if not mid:
                continue
            if micro_ids is not None:
                nid = item.get("node_id") or item.get("event_id") or item.get("text_id")
                if nid and str(item.get("node_type") or "") != "macro":
                    bucket = nodes_by_macro.setdefault(mid, [])
                    if nid not in bucket:
                        bucket.append(nid)
            raw_score = item.get("cosine")
            if raw_score is None:
                raw_score = item.get("similarity", item.get("score"))
            try:
                score = float(raw_score)
            except (TypeError, ValueError):
                score = 0.0
            if mid not in scores:
                scores[mid] = score
                order.append(mid)
                _keep_span(mid, item, score, new=True)
            elif score > scores[mid]:
                scores[mid] = score
                _keep_span(mid, item, score, new=False)

    for entry in r["entries"]:
        try:
            if entry == "nodes":
                _collect(tk.search_nodes(question, top_k=topk * mult, node_types=r.get("node_types")))
            elif entry == "macro":
                _collect(tk.search_nodes(question, top_k=topk * mult, node_types=["macro"]))
            elif entry == "ocr":
                _collect(tk.search_ocr_text(question, top_k=topk * mult))
            elif entry == "enumerate":
                _collect(
                    tk.enumerate_events(
                        question, min_cosine=r.get("enumerate_min_cosine", 0.35), max_results=topk * mult
                    )
                )
            elif entry == "time":
                _collect(tk.search_by_time(0, 10**9))
        except Exception as exc:
            pass

    ranked = [(mid, scores[mid]) for mid in order]

    def _publish(result: list[tuple[str, float]]) -> list[tuple[str, float]]:
        if spans is not None:
            for mid, _ in result:
                if mid in best_span:
                    spans[mid] = best_span[mid]
        if micro_ids is not None:
            for mid, _ in result:
                for nid in nodes_by_macro.get(mid, []):
                    if nid not in micro_ids:
                        micro_ids.append(nid)
        return result

    if r.get("rerank") or adaptive.get("enabled"):
        ranked.sort(key=lambda kv: -kv[1])

    if adaptive.get("enabled") and ranked:
        best = max(s for _, s in ranked)
        if best <= 0:
            return _publish(ranked[:base_topk])
        floor = best * float(adaptive.get("keep_ratio", 0.85))
        kept = [(mid, s) for mid, s in ranked if s >= floor]
        return _publish(kept[:topk])

    return _publish(ranked[:base_topk])

def _fmt_subgraph(sg: dict, g: dict) -> dict[str, list[str]]:
    e = g["evidence"]
    head = f"[{sg.get('time_range', '')}] {sg.get('label', sg.get('macro_id', ''))}"
    summary = ""
    if e.get("include_summary"):
        summary = " ".join((sg.get("description") or "").split())
    events = [
        f"  - EVENT({ev.get('time_range','')}) {ev.get('action','')}: {ev.get('description','')}"
        for ev in (sg.get("events") or [])[: e["max_events"]]
    ]
    entities = [
        f"  - ENTITY[{en.get('entity_type','')}] {en.get('name','')}: {en.get('description','')}"
        for en in (sg.get("entities") or [])[: e["max_entities"]]
    ]
    ocr = [
        f'  - ON_SCREEN_TEXT({o.get("time_range","")}) "{o.get("text","")}"'
        for o in (sg.get("ocr_texts") or [])[: e["max_ocr"]]
    ]
    return {
        "head": [head],
        "summary": [f"  SEGMENT: {summary}"] if summary else [],
        "events": events,
        "entities": entities,
        "ocr": ocr,
    }

def drill(tk, macro_ids: list[str], g: dict) -> str:
    e = g["evidence"]
    budget = e["max_chars"]
    blocks: list[dict[str, list[str]]] = []
    for mid in macro_ids:
        try:
            sg = jload(tk.get_subgraph(mid))
        except Exception:
            continue
        if isinstance(sg, dict) and "error" not in sg:
            blocks.append(_fmt_subgraph(sg, g))
    if not blocks:
        return ""

    lines: list[str] = []
    used = 0
    for b in blocks:
        for h in b["head"]:
            lines.append(h)
            used += len(h)
    cap = budget * e.get("summary_budget_share", 0.35)
    for b in blocks:
        for s in b.get("summary", []):
            if used + len(s) <= cap:
                lines.append(s)
                used += len(s)
    dedup = bool(e.get("dedup"))
    seen_lines: set[str] = {ln.strip().lower() for ln in lines} if dedup else set()
    idx = {id(b): {k: 0 for k in e["order"]} for b in blocks}
    progressed = True
    while progressed and used < budget:
        progressed = False
        for b in blocks:
            for cat in e["order"]:
                pool = b.get(cat, [])
                i = idx[id(b)][cat]
                if i < len(pool):
                    line = pool[i]
                    if dedup:
                        key = line.strip().lower()
                        if key in seen_lines:
                            idx[id(b)][cat] = i + 1
                            progressed = True
                            continue
                    if used + len(line) > budget:
                        continue
                    lines.append(line)
                    if dedup:
                        seen_lines.add(line.strip().lower())
                    used += len(line)
                    idx[id(b)][cat] = i + 1
                    progressed = True
    return "\n".join(lines)

PENDING_STRATEGY: dict | None = None

def _learned_strategy(q: dict, g: dict, tk, baseline_top: float | None = None
                      ) -> tuple[str, list[str]] | None:
    r = g.get("retrieval", {})
    if not r.get("use_learned_strategy"):
        return None
    ceiling = r.get("strategy_max_cosine")
    if ceiling is not None and baseline_top is not None and baseline_top >= float(ceiling):
        return None
    try:
        import strategy_forge
    except Exception:
        return None
    if PENDING_STRATEGY is not None:
        spec, code = PENDING_STRATEGY, PENDING_STRATEGY.get("code")
    else:
        spec = strategy_forge.strategy_for_type(q.get("type"))
        code = strategy_forge.strategy_code(spec) if spec else None
    if not spec or not code:
        return None
    try:
        return strategy_forge.run_strategy(code, tk, q["question"], g)
    except Exception as exc:
        return None

def build_evidence(q: dict, g: dict, mode: str, tk=None,
                   probe: dict | None = None) -> tuple[str, list[str]]:
    if mode == "none":
        return "", []

    if mode == "all":
        mem = Memory(q["video_key"])
        parts = []
        for mid in mem.all_macro_ids():
            m = mem.macro(mid)
            f = mem.visual_fields(mid)
            tr = m["time_range"]
            parts.append(f"[{fmt_time(tr[0])}-{fmt_time(tr[1])}] {m.get('summary','')}")
            for ev in f["events"]:
                etr = ev.get("time_range") or tr
                parts.append(
                    f"  - EVENT({fmt_time(etr[0])}-{fmt_time(etr[1])}) "
                    f"{ev.get('action','')}: {ev.get('description','')}"
                )
            for en in f["entities"]:
                parts.append(f"  - ENTITY[{en.get('entity_type','')}] {en.get('name','')}: {en.get('description','')}")
            for o in f["ocr"]:
                parts.append(f'  - ON_SCREEN_TEXT "{o.get("text","")}"')
        return "\n".join(parts), mem.all_macro_ids()

    if mode == "gold":
        a = q["anchor"]
        gold = list(a.get("macro_ids") or [a["macro_id"]])
        if probe is not None:
            probe["located_micro"] = list(a.get("micro_ids") or [])
        return drill(tk, gold, g), gold

    spans: dict[str, tuple[float, float]] = {}
    micro_hits: list[str] = []
    ranked = locate_scored(tk, q["question"], g, spans=spans, micro_ids=micro_hits)
    baseline_top = ranked[0][1] if ranked else 0.0
    if probe is not None:
        probe["top_cosine"] = baseline_top
        probe["spans"] = spans
        probe["located_micro"] = list(micro_hits)

    learned = _learned_strategy(q, g, tk, baseline_top) if tk is not None else None
    if learned is not None:
        if probe is not None:
            probe["located_micro"] = None
        return learned

    ids = [mid for mid, _ in ranked]
    if g["evidence"].get("include_neighbor_macro") and ids:
        all_ids = Memory(q["video_key"]).all_macro_ids()
        expanded: list[str] = []
        for mid in ids:
            if mid in all_ids:
                i = all_ids.index(mid)
                for j in (i - 1, i, i + 1):
                    if 0 <= j < len(all_ids) and all_ids[j] not in expanded:
                        expanded.append(all_ids[j])
            elif mid not in expanded:
                expanded.append(mid)
        ids = expanded

    evidence_ids = ids
    n_cite = g["evidence"].get("primary_only")
    if n_cite and ids:
        try:
            keep = max(1, int(n_cite))
        except (TypeError, ValueError):
            keep = 1
        evidence_ids = ids[:keep]
    return drill(tk, evidence_ids, g), ids

def build_prompt(q: dict, evidence: str, g: dict, tail: str = "") -> str:
    tpl = (VEH_DIR / g["answer"]["prompt_file"]).read_text(encoding="utf-8")
    if evidence:
        block = f"VIDEO GRAPH MEMORY EVIDENCE\n{evidence}"
    else:
        block = ("VIDEO GRAPH MEMORY EVIDENCE\n"
                 "(no video evidence is provided this time; judge from general knowledge "
                 "and the options alone)")
    out = tpl.format(evidence_block=block, question=q["question"], options="\n".join(q["options"]))
    return out + tail if tail else out

def anchor_ok(pred_anchor: list[float] | None, gold_anchor: dict | None, tol: float = 60.0) -> bool:
    if not pred_anchor or not gold_anchor:
        return False
    tr = gold_anchor.get("time_range") if isinstance(gold_anchor, dict) else None
    if not tr:
        return False
    gs, ge = float(tr[0]), float(tr[1])
    ps, pe = min(pred_anchor), max(pred_anchor)
    return not (pe < gs - tol or ps > ge + tol)

def micro_anchor_ok(pred_anchor: list[float] | None, gold_anchor: dict | None, tol: float = 60.0) -> bool | None:
    if not isinstance(gold_anchor, dict):
        return None
    tr = gold_anchor.get("micro_time_range")
    if not tr or len(tr) != 2:
        return None
    return anchor_ok(pred_anchor, {"time_range": tr}, tol)

MODE_ROLE = {"retrieve": "solver", "none": "blind", "all": "oracle", "gold": "diagnose_gold"}

TOOL_SPECS: dict[str, dict] = {
    "search_nodes": {
        "params": {"query": str, "top_k": int, "node_types": list},
        "desc": 'semantic search over memory nodes. node_types may include "macro" '
                '(segment level, 11 nodes), "event"/"entity"/"on_screen_text" (fragment '
                'level, 508 nodes). Fragments outnumber segments ~46:1, so a mixed '
                'node_types list is dominated by fragments.',
    },
    "get_subgraph": {
        "params": {"macro_id": str},
        "desc": "full detail of one segment: every micro-event, entity, on-screen text "
                "and relation. This is the only source of concrete facts.",
    },
    "search_ocr_text": {
        "params": {"query": str, "top_k": int},
        "desc": "search on-screen text only (captions, scores, titles, numbers).",
    },
    "enumerate_events": {
        "params": {"query": str, "min_cosine": float, "max_results": int, "node_types": list},
        "desc": "exhaustive recall above a similarity threshold. Use for counting or "
                "for sweeping every occurrence.",
    },
    "search_by_time": {
        "params": {"start_sec": float, "end_sec": float},
        "desc": "list what the memory holds inside a time window.",
    },
    "get_summary": {"params": {}, "desc": "root-level summary of the whole video."},
    "get_super_events": {
        "params": {},
        "desc": "list the narrative arcs (super events) with their time ranges.",
    },
    "get_macro_events": {
        "params": {"super_id": str, "macro_ids": list},
        "desc": "list the segments of one super event, or fetch specific macro_ids.",
    },
    "search_asr_text": {"params": {"query": str, "top_k": int}, "desc": "search spoken content."},
    "read_video": {
        "params": {"start_sec": float, "end_sec": float},
        "desc": "look at the ORIGINAL video frames for a narrowed time window. Use it to verify "
                "what memory only describes: exact counts, on-screen text, colours, positions. "
                "Expensive and rate-limited, so narrow the window with memory tools first.",
    },
}

FROZEN_TOOLS = {"search_asr_text": False}

def enabled_tools(g: dict) -> list[str]:
    te = g["policy"]["tools_enabled"]
    out = []
    for name in TOOL_SPECS:
        if name.startswith("_"):
            continue
        if name in FROZEN_TOOLS:
            if not FROZEN_TOOLS[name]:
                continue
        if te.get(name):
            out.append(name)
    return out

def _select_rules(g: dict, qtype: str | None) -> tuple[list[dict], list[dict]]:
    inj = g["policy"].get("inject") or {}
    rules = [r for r in g["policy"]["rules"] if not str(r.get("id", "")).startswith("_")]

    if inj.get("select_by", "question_type") == "question_type" and qtype:
        rules = [r for r in rules if not r.get("applies_to") or qtype in r["applies_to"]]

    def rank(r: dict):
        trials = r.get("trials", 0) or 0
        rate = (r.get("wins", 0) / trials) if trials else 0.0
        return (-rate, -(r.get("added_at", 0)))

    tagged = [r for r in rules if qtype and qtype in (r.get("applies_to") or [])]
    tagged_ids = {id(r) for r in tagged}
    rest = [r for r in rules if id(r) not in tagged_ids]
    order = (
        sorted(tagged, key=rank)
        + sorted([r for r in rest if r.get("kind") == "anti_pattern"], key=rank)
        + sorted([r for r in rest if r.get("kind") == "routing"], key=rank)
    )

    max_n = int(inj.get("max_rules_per_call", 5))
    max_c = int(inj.get("max_chars_per_call", 900))
    picked: list[dict] = []
    used = 0
    for r in order:
        t = r.get("text", "")
        if len(picked) >= max_n or used + len(t) > max_c:
            continue
        picked.append(r)
        used += len(t)
    return ([r for r in picked if r.get("kind") == "routing"],
            [r for r in picked if r.get("kind") == "anti_pattern"])

def render_policy_sections(g: dict, qtype: str | None = None) -> tuple[str, str, str]:
    tools = enabled_tools(g)
    lines = []
    for t in tools:
        spec = TOOL_SPECS[t]
        args = ", ".join(f"{k}: {v.__name__}" for k, v in spec["params"].items()) or "no arguments"
        lines.append(f"- {t}({args})\n    {spec['desc']}")
    tool_list = "\n".join(lines) if lines else "(no tools enabled)"

    routing, anti = _select_rules(g, qtype)
    fmt = lambda rs: "\n".join(f"{i}. {r['text']}" for i, r in enumerate(rs, 1)) or "(none)"
    return tool_list, fmt(routing), fmt(anti)

def _extract_macro_ids(data) -> list[str]:
    found: list[str] = []

    def walk(x):
        if isinstance(x, dict):
            mid = x.get("macro_id")
            if isinstance(mid, str) and mid.startswith("macro_") and mid not in found:
                found.append(mid)
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(data)
    return found

_MICRO_ID_KEYS = ("node_id", "event_id", "text_id", "entity_id")

def _extract_node_ids(data) -> list[str]:
    found: list[str] = []

    def walk(x):
        if isinstance(x, dict):
            if str(x.get("node_type") or "") != "macro":
                for k in _MICRO_ID_KEYS:
                    v = x.get(k)
                    if isinstance(v, str) and v and v not in found:
                        found.append(v)
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(data)
    return found

_frame_extractor = None

def _get_frame_extractor():
    global _frame_extractor
    if _frame_extractor is None:
        import sys

        cap = ROOT / "Qwen-MM-Plugins" / "src" / "capabilities" / "video-memory"
        bm = cap / "skill" / "script" / "build_memory"
        for p in (bm, cap, ROOT / "Qwen-MM-Plugins" / "src"):
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))
        from build_graph import extract_frames_base64

        _frame_extractor = extract_frames_base64
    return _frame_extractor

def _extract_official(video_path: str, start: float, end: float, *,
                      fps: float, max_dim: int, max_frames: int,
                      total_duration: float = 0.0) -> list[tuple[str, str]]:
    import core_arm

    frames = core_arm.read_video_once(
        video_path, start, end,
        fps=fps,
        budget=core_arm.budget_for_max_dim(max_dim),
        max_frames=max_frames,
    )
    if not frames:
        return []
    api = core_arm._core()
    ref = total_duration if total_duration > 0 else (frames[-1][0] or end)
    return [(f"data:image/jpeg;base64,{b64}", api["format_timestamp"](ts, ref))
            for ts, b64 in frames]

REPLAY_BACKENDS = ("core", "build_memory")

_NEED_MORE_RE = re.compile(
    r"NEED_MORE\s*:\s*(?:none|no|\b0\b)|NEED_MORE\s*:\s*(\d+(?:\.\d+)?)\s*[-–~,]+\s*(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
_NEED_MORE_NONE_RE = re.compile(r"NEED_MORE\s*:\s*(?:none|no|\b0\b)", re.IGNORECASE)

def _parse_need_more(text: str):
    if not text:
        return None
    if _NEED_MORE_NONE_RE.search(text):
        return None
    m = _NEED_MORE_RE.search(text)
    if not m or not m.group(1) or not m.group(2):
        return None
    try:
        a, b = float(m.group(1)), float(m.group(2))
    except (TypeError, ValueError):
        return None
    if b <= a:
        return None
    return [a, b]

def _clip_window(win, max_span: float):
    try:
        s, e = float(win[0]), float(win[1])
    except (TypeError, ValueError, IndexError):
        return None
    if s < 0:
        s = 0.0
    if e <= s:
        return None
    if e - s > max_span:
        e = s + max_span
    return [s, e]

def _split_focus(win, focus, min_side: float = 2.0):
    ws, we = float(win[0]), float(win[1])
    if not focus:
        return [[ws, we]], 0
    fs, fe = max(ws, float(focus[0])), min(we, float(focus[1]))
    if fe <= fs or (fe - fs) >= (we - ws) - min_side:
        return [[ws, we]], 0
    parts, idx = [], 0
    if fs - ws >= min_side:
        parts.append([ws, fs])
        idx = 1
    parts.append([fs, fe])
    if we - fe >= min_side:
        parts.append([fe, we])
    return parts, idx

def exec_read_video(
    q: dict,
    args: dict,
    g: dict,
    used_calls: int,
    located: list[str],
    force: bool = False,
    windows: list[list[float]] | None = None,
    fps_override: float | None = None,
    dim_override: int | None = None,
    frames_override: int | None = None,
    focus: list[list[float]] | None = None,
    focus_stats: dict | None = None,
) -> tuple[str, list]:
    rv = g["policy"].get("read_video") or {}
    route = video_routing.route_of(q, g)

    def rvg(key: str, default):
        return video_routing.gene(g, f"policy.read_video.{key}", route, default)

    if not rv.get("enabled", False) and not force:
        return "ERROR: read_video is disabled by the current policy.", []
    if used_calls >= int(rvg("max_calls_per_question", 1)):
        return (f"ERROR: read_video budget exhausted "
                f"({rvg('max_calls_per_question', 1)} call(s) per question). "
                f"Answer from what you have."), []
    if rv.get("require_narrowed_window", True) and not located:
        return ("ERROR: use a memory tool first to narrow down the time range. "
                "read_video is for verifying a specific window, not for browsing."), []

    max_span = float(rvg("max_window_sec", 60))
    if windows:
        wins = [[float(a), float(b)] for a, b in windows if float(b) > float(a)]
    else:
        try:
            s0 = float(args.get("start_sec"))
            e0 = float(args.get("end_sec"))
        except (TypeError, ValueError):
            return "ERROR: read_video needs numeric start_sec and end_sec.", []
        if e0 <= s0:
            return "ERROR: end_sec must be greater than start_sec.", []
        wins = [[s0, e0]]
    if not wins:
        return "ERROR: read_video got no usable time window.", []

    fps = float(rvg("fps", 1.0)) if fps_override is None else float(fps_override)
    max_dim = int(rvg("max_dim", 768)) if dim_override is None else int(dim_override)
    total = int(rvg("max_frames_per_call", 16)) if frames_override is None else int(frames_override)

    truncated = False
    norm: list[list[float]] = []
    keep: list[int] = []
    for i, (s, e) in enumerate(wins):
        s, e = float(s), float(e)
        if e - s > max_span:
            e = s + max_span
            truncated = True
        norm.append([s, e])
        keep.append(i)

    fw = float(rvg("focus_weight", 1.0))
    sub: list[list[float]] = []
    weight: list[float] = []
    n_focused = 0
    for i, w in zip(keep, norm):
        f = None
        if focus and fw > 1.0 and i < len(focus):
            f = focus[i]
        parts, fidx = _split_focus(w, f)
        if len(parts) > 1:
            n_focused += 1
        for j, p in enumerate(parts):
            sub.append(p)
            weight.append(fw if (len(parts) > 1 and j == fidx) else 1.0)
    if not sub:
        sub, weight = norm, [1.0] * len(norm)

    seg_len = [e - s for s, e in sub]
    wlen = [ln * wt for ln, wt in zip(seg_len, weight)]
    tot_w = sum(wlen) or 1.0

    mem = Memory(q["video_key"])
    if not mem.video_path or not Path(mem.video_path).exists():
        return f"ERROR: the original video file is unavailable ({mem.video_path}).", []

    backend = str(rv.get("backend", "core")).lower()
    if backend not in REPLAY_BACKENDS:
        return (f"ERROR: unknown read_video backend {backend!r}; "
                f"expected one of {REPLAY_BACKENDS}."), []

    if backend == "core":
        total_duration = 0.0
        try:
            import core_arm

            total_duration = core_arm.video_duration(q.get("video_key") or "", mem.video_path)
        except Exception:
            total_duration = 0.0

        def _extract(s: float, e: float, k: int):
            return _extract_official(mem.video_path, s, e, fps=fps, max_dim=max_dim,
                                     max_frames=k, total_duration=total_duration)
    else:
        try:
            raw_extractor = _get_frame_extractor()
        except Exception as exc:
            return f"ERROR: frame extractor unavailable: {str(exc)[:160]}", []

        def _extract(s: float, e: float, k: int):
            return raw_extractor(mem.video_path, s, e, fps=fps, max_dim=max_dim,
                                 max_frames=k)

    items: list = []
    errors: list[str] = []
    for idx, (s, e) in enumerate(sub):
        if len(items) >= total:
            break
        want = int(round((e - s) * fps * weight[idx])) + 1
        share = max(4, int(round(total * wlen[idx] / tot_w)))
        k = max(4, min(want, share, total - len(items)))
        try:
            part = _extract(s, e, min(k, total - len(items)))
        except Exception as exc:
            errors.append(f"{fmt_time(s)}-{fmt_time(e)}: {str(exc)[:80]}")
            continue
        if not part:
            errors.append(f"{fmt_time(s)}-{fmt_time(e)}: no frames")
            continue
        items.extend(part)

    if not items:
        detail = ("; ".join(errors[:3]) or "the extractor returned nothing")
        return f"ERROR: no frames could be extracted ({detail}).", []

    gate_cap = 0
    try:
        import core_arm

        gate_cap = int(core_arm.GATEWAY_MAX_FRAMES)
    except Exception:
        gate_cap = 0
    n_clipped = 0
    if gate_cap > 0 and len(items) > gate_cap:
        n_clipped = len(items)
        step = (len(items) - 1) / (gate_cap - 1) if gate_cap > 1 else 0
        items = [items[round(i * step)] for i in range(gate_cap)]

    spans = ", ".join(f"{fmt_time(s)}-{fmt_time(e)}" for s, e in norm)
    note = (f"Read {len(items)} frames from {len(norm)} segment(s): {spans}"
            + (f" (each window truncated to {max_span:.0f}s by policy)" if truncated else "")
            + (f" | {n_focused} window(s) sampled more densely around the matched moment"
               if n_focused else "")
            + (f" | downsampled from {n_clipped} to fit the gateway image cap"
               if n_clipped else "")
            + (f" | partial failures: {'; '.join(errors[:2])}" if errors else "")
            + ". They are attached to this message -- look at them directly.")
    if isinstance(focus_stats, dict):
        focus_stats["focused_windows"] = n_focused
        focus_stats["sub_ranges"] = len(sub)
        focus_stats["backend"] = backend
        focus_stats["route"] = route
        if n_clipped:
            focus_stats["gateway_clipped_from"] = n_clipped
    return note, items

def _record_frame_discovery(q: dict, window: tuple[float, float], frame_count: int) -> None:
    if q.get("split") != "evolve":
        return
    event = {
        "qid": q.get("qid"),
        "video_key": q.get("video_key"),
        "type": q.get("type"),
        "window": [round(window[0], 3), round(window[1], 3)],
        "frames": frame_count,
        "created_at": time.time(),
    }
    path = STORE / "memory_patch_candidates.jsonl"
    payload = (json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8")
    fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        os.write(fd, payload)
    finally:
        os.close(fd)

def _index_metrics(tk) -> dict:
    index = getattr(tk, "index", None)
    getter = getattr(index, "get_thread_metrics", None) or getattr(index, "get_metrics", None)
    return getter() if callable(getter) else {}

def _metric_delta(before: dict, after: dict) -> dict:
    keys = ("dense_attempts", "dense_successes", "dense_timeouts", "dense_failures",
            "bm25_fallbacks", "embedding_latency_sec")
    return {k: round(float(after.get(k, 0)) - float(before.get(k, 0)), 4) for k in keys}

def _macro_windows(video_key: str, located: list[str], max_span: float,
                   max_segments: int = 1, spans: dict | None = None,
                   pad: float = 8.0, stats: dict | None = None,
                   focus_out: list | None = None) -> list[list[float]]:
    n = max(1, int(max_segments))
    raw: list[list[float]] = []
    n_node = n_macro = 0
    for mid in located or []:
        w = None
        f = None
        tr = (spans or {}).get(mid)
        if tr:
            s, e = float(tr[0]), float(tr[1])
            f = [s, e]
            s, e = max(0.0, s - pad), e + pad
            if e - s > max_span:
                center = (s + e) / 2
                s, e = max(0.0, center - max_span / 2), center + max_span / 2
            if e > s:
                w = (s, e)
                n_node += 1
        if w is None:
            f = None
            w = _macro_window(video_key, [mid], max_span)
            if w:
                n_macro += 1
        if w and float(w[1]) > float(w[0]):
            raw.append([float(w[0]), float(w[1]), f])
        if len(raw) >= n:
            break
    if stats is not None:
        stats["node_spans"] = n_node
        stats["macro_spans"] = n_macro
    raw.sort(key=lambda r: (r[0], r[1]))
    merged: list[list] = []
    for s, e, f in raw:
        if merged and s <= merged[-1][1] + 1.0:
            merged[-1][1] = max(merged[-1][1], e)
            if f:
                mf = merged[-1][2]
                merged[-1][2] = [min(mf[0], f[0]), max(mf[1], f[1])] if mf else list(f)
        else:
            merged.append([s, e, list(f) if f else None])
    merged = merged[:n]
    if focus_out is not None:
        focus_out.clear()
        focus_out.extend(m[2] for m in merged)
    return [[m[0], m[1]] for m in merged]

REPLAY_PLAN_TMPL = """You are about to answer a multiple-choice question about a video.
You have NOT seen any video frames yet. Memory retrieval found these candidate segments
(id, time range, one-line summary):

QUESTION: {question}
{options}

CANDIDATES:
{cands}

Decide which segments you must actually LOOK AT to answer. You may pick 0 to {cap}
segments. Pick several when the answer is spread over more than one place.

Reply with ONLY a JSON object, no prose:
{{"decision": "<one of: answerable_from_text | need_visual | retrieval_missed>",
 "watch": ["macro_0003", "macro_0011"], "why": "one short sentence"}}

How to choose `decision` -- this distinction matters, it is NOT a cosmetic label:
- "answerable_from_text" + watch=[] : the summaries above already contain the answer.
  Use it ONLY then. Staring at frames actively hurts when the question needs whole-video
  reasoning (totals, order of events, overall plot), so this is a real, useful answer.
- "need_visual" + watch=[...ids...] : you must look. Copy ids exactly from CANDIDATES.
- "retrieval_missed" : the answer is NOT in these candidates (they mention none of the
  things the question asks about). This is a statement about the RETRIEVAL being wrong,
  not about you being done. We will then re-search the memory with a wider net and ask
  you again -- so do NOT use this to give up. If you can still name the closest
  segment(s), put them in `watch` as well.

- Never invent an id. Never return more than {cap} ids.
- If your "why" says you need to look, `watch` must NOT be empty.
"""

def _candidate_briefs(video_key: str, ids: list[str], per: int = 90) -> list[tuple[str, float, float, str]]:
    try:
        mem = Memory(video_key)
    except Exception:
        return []
    out: list[tuple[str, float, float, str]] = []
    for mid in ids or []:
        m = mem.macro(mid)
        if not m:
            continue
        tr = m.get("time_range") or []
        s = float(tr[0]) if tr and len(tr) else 0.0
        e = float(tr[1]) if tr and len(tr) > 1 else s
        summ = str(m.get("summary") or m.get("label") or "").replace("\n", " ")[:per]
        out.append((mid, s, e, summ))
    return out

_NEEDS_LOOK = ("need to look", "need to watch", "need to see", "must look", "visual",
               "requires visual", "visual inspection", "inspection is required",
               "need visual", "have to look", "should look")

def _widen_candidates(q: dict, g: dict, tk, seen: list[str], cap_show: int,
                      spans: dict | None = None) -> list[str]:
    seen_list = list(seen)
    out = list(seen_list)
    try:
        mem = Memory(q["video_key"])
        rest = [m for m in mem.all_macro_ids() if m not in set(seen_list)]
    except Exception:
        rest = []

    if tk is not None:
        opts = q.get("options") or []
        if isinstance(opts, dict):
            opts = list(opts.values())
        rich = str(q.get("question", "")) + " " + " ".join(str(o) for o in opts)[:300]
        wide = dict(g)
        wide_spans: dict[str, tuple[float, float]] = {}
        try:
            import copy
            wide = copy.deepcopy(g)
            r = wide.setdefault("retrieval", {})
            r["topk_macro"] = max(int(r.get("topk_macro", 3)) * 3,
                                  int(r.get("topk_macro", 3)) + 6)
            r.pop("adaptive_topk", None)
            ranked = locate_scored(tk, rich, wide, spans=wide_spans)
        except Exception:
            ranked = []
        if spans is not None:
            for mid, tr in wide_spans.items():
                spans.setdefault(mid, tr)
        for mid, _ in ranked:
            if mid not in out:
                out.append(mid)
    for m in rest:
        if m not in out:
            out.append(m)
    return out[: max(cap_show * 3, 24)]

def _ask_planner(q: dict, g: dict, cfg: dict, ids: list[str], cap: int) -> dict:
    rv = g.get("policy", {}).get("read_video") or {}
    briefs = _candidate_briefs(q["video_key"], ids, per=90)
    if not briefs:
        return {"ok": False, "why": "no candidate summary available", "watch": None,
                "decision": "unavailable", "briefs": 0}
    max_show = int(rv.get("planner_max_candidates", 8))
    lines = "\n".join(f"- {mid} [{fmt_time(s)}-{fmt_time(e)}] {summ}"
                      for mid, s, e, summ in briefs[:max_show])
    raw_opts = q.get("options") or []
    if isinstance(raw_opts, dict):
        opts = "\n".join(f"  {k}) {v}" for k, v in raw_opts.items())
    else:
        opts = "\n".join(f"  {chr(ord('A') + i)}) {v}" for i, v in enumerate(raw_opts))
    try:
        text = call_llm(
            REPLAY_PLAN_TMPL.format(question=str(q.get("question", ""))[:600],
                                    options=opts or "(no options given)",
                                    cands=lines, cap=cap),
            cfg,
            int(rv.get("plan_max_tokens", 300)),
            0.0,
            role="replay_plan",
        )
    except Exception as exc:
        return {"ok": False, "why": f"planner call failed: {str(exc)[:80]}",
                "watch": None, "decision": "call_failed", "briefs": len(briefs)}

    obj: dict | None = None
    for i, ch in enumerate(text or ""):
        if ch != "{":
            continue
        try:
            cand, _ = json.JSONDecoder().raw_decode(text[i:])
        except Exception:
            continue
        if isinstance(cand, dict):
            obj = cand
            break
    if obj is None:
        return {"ok": False, "why": f"planner output unusable: {(text or '')[:80]}",
                "watch": None, "decision": "unparsable", "briefs": len(briefs)}

    dec = str(obj.get("decision", "")).strip().lower()
    if dec not in ("answerable_from_text", "need_visual", "retrieval_missed"):
        raw_watch = obj.get("watch")
        dec = "need_visual" if raw_watch else "answerable_from_text"
    why = str(obj.get("why", ""))[:140]
    allowed = {b[0] for b in briefs}
    raw_watch = obj.get("watch")
    watch = [str(x) for x in raw_watch] if isinstance(raw_watch, list) else []
    picked = [x for x in watch if x in allowed]
    picked = list(dict.fromkeys(picked))[:max(1, cap)]

    if dec == "need_visual" and not picked and any(k in why.lower() for k in _NEEDS_LOOK):
        return {"ok": False, "why": f"contradiction (says it must look, watch empty): {why}",
                "watch": None, "decision": "contradiction", "briefs": len(briefs)}
    return {"ok": True, "decision": dec, "watch": picked, "why": why,
            "briefs": len(briefs)}

def _plan_replay_segments(q: dict, g: dict, cfg: dict, ids: list[str],
                          cap: int = 3, tk=None,
                          spans: dict | None = None) -> tuple[list[str] | None, str]:
    rv = g.get("policy", {}).get("read_video") or {}
    attempts = []
    cur_ids = list(ids)
    for rnd in range(2):
        ans = _ask_planner(q, g, cfg, cur_ids, cap)
        attempts.append(f"round{rnd}:{ans.get('decision')}")
        if not ans["ok"]:
            return None, f"planner unusable ({', '.join(attempts)}): {ans['why']}"
        if ans["decision"] == "need_visual":
            note = (f"model picked {len(ans['watch'])}/{ans['briefs']} segment(s): "
                    f"{','.join(ans['watch'])} | why: {ans['why']}")
            if ans["watch"]:
                return ans["watch"], note
            return None, f"need_visual but no valid ids ({', '.join(attempts)})"
        if ans["decision"] == "answerable_from_text":
            return [], f"model says text already answers it | why: {ans['why']}"
        if rnd == 0 and tk is not None and rv.get("widen_on_miss", True):
            widened = _widen_candidates(q, g, tk, cur_ids,
                                        int(rv.get("planner_max_candidates", 8)),
                                        spans=spans)
            if len(widened) > len(cur_ids):
                cur_ids = widened
                continue
        return list(ids)[:cap] or [], (
            f"retrieval_missed even after widening ({', '.join(attempts)}) -> "
            f"watching top-{min(cap, max(1, len(ids)))} anyway | why: {ans['why']}")
    return None, f"planner exhausted ({', '.join(attempts)})"

def _macro_window(video_key: str, located: list[str], max_span: float) -> tuple[float, float] | None:
    if not located:
        return None
    macro = Memory(video_key).macro(located[0]) or {}
    tr = macro.get("time_range") or []
    if not isinstance(tr, (list, tuple)) or len(tr) < 2:
        return None
    try:
        start, end = float(tr[0]), float(tr[1])
    except (TypeError, ValueError):
        return None
    if end - start > max_span:
        center = (start + end) / 2
        start, end = max(0.0, center - max_span / 2), center + max_span / 2
    return start, end

def _run_learned_pipeline_tool(q: dict, tk) -> tuple[str, str | None]:
    try:
        from tool_forge import build_cases_for_type, run_tool, tools_for_type

        cases = build_cases_for_type(q.get("type", ""), [q])
        if not cases:
            return "", None
        for spec in tools_for_type(q.get("type", ""))[:1]:
            value = run_tool(spec["code"], spec["name"], tk, cases[0]["kwargs"], timeout=20.0)
            return f"\nVERIFIED COMPUTATION ({spec['name']}): {value}", spec["name"]
    except Exception as exc:
        return f"\nVERIFIED COMPUTATION unavailable: {str(exc)[:160]}", None
    return "", None

def exec_tool(
    tk, name: str, args: dict, g: dict, node_ids: list[str] | None = None
) -> tuple[str, list[str]]:
    if name not in TOOL_SPECS:
        return f"ERROR: unknown tool {name!r}. Available: {', '.join(enabled_tools(g))}", []
    if name not in enabled_tools(g):
        return f"ERROR: tool {name!r} is disabled by the current policy.", []

    spec = TOOL_SPECS[name]
    clean: dict = {}
    for k, typ in spec["params"].items():
        if k not in args or args[k] is None:
            continue
        v = args[k]
        try:
            if typ is list:
                clean[k] = list(v) if isinstance(v, (list, tuple)) else [v]
            elif typ is int:
                clean[k] = int(v)
            elif typ is float:
                clean[k] = float(v)
            else:
                clean[k] = str(v)
        except (TypeError, ValueError):
            continue

    try:
        raw = getattr(tk, name)(**clean)
    except Exception as exc:
        return f"ERROR calling {name}: {str(exc)[:200]}", []

    data = jload(raw)
    mids = _extract_macro_ids(data)
    if node_ids is not None:
        for nid in _extract_node_ids(data):
            if nid not in node_ids:
                node_ids.append(nid)
    text = raw if isinstance(raw, str) else json.dumps(data, ensure_ascii=False)
    limit = g["policy"]["limits"]["max_obs_chars_per_round"]
    if len(text) > limit:
        text = text[:limit] + f"\n... [truncated at {limit} chars by policy.limits]"
    return text, mids

def _render_observations(history: list[dict], g: dict) -> str:
    if not history:
        return "(nothing yet -- this is your first move)"
    window = g["policy"]["limits"]["obs_window"]
    out = []
    for i, h in enumerate(history, 1):
        head = f"[round {i}] {h['call']}"
        if i > len(history) - window:
            out.append(f"{head}\n{h['obs']}")
        else:
            macros = ", ".join(h["macros"][:6]) or "none"
            out.append(f"{head} -> {h['n_macros']} macro(s): {macros} [detail dropped to save budget]")
    return "\n\n".join(out)

def _parse_tool_call(text: str) -> tuple[str, dict] | None:
    m = re.search(r"TOOL\s*:\s*(\{.*)", text, re.S)
    if not m:
        return None
    frag = m.group(1)
    depth, in_str, esc, end = 0, False, False, -1
    for i, ch in enumerate(frag):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end < 0:
        return None
    try:
        obj = json.loads(frag[:end])
    except json.JSONDecodeError:
        return None
    name = obj.get("name") or obj.get("tool")
    if not isinstance(name, str):
        return None
    args = obj.get("args") or obj.get("arguments") or {}
    return name, (args if isinstance(args, dict) else {})

def solve_one_agent(q: dict, g: dict, cfg: dict, tk) -> dict:
    t0 = time.time()
    dense_before = _index_metrics(tk)
    tpl = (VEH_DIR / g["answer"]["agent_prompt_file"]).read_text(encoding="utf-8")
    tool_list, routing, anti = render_policy_sections(g, q["type"])
    per_type = g["budget"].get("per_type_rounds") or {}
    max_rounds = int(per_type.get(q["type"], g["budget"]["max_tool_rounds"]))

    history: list[dict] = []
    located: list[str] = []
    located_micro: list[str] = []
    text, err = "", None
    error_kind = None
    n_rounds = 0
    rv_calls = 0
    frame_count = 0
    replay_windows: list[list[float]] = []
    retrieval_latency = 0.0
    pending_frames: list = []

    for rnd in range(1, max_rounds + 2):
        left = max(0, max_rounds - len(history))
        prompt = tpl.format(
            tool_list=tool_list,
            routing_rules=routing,
            anti_patterns=anti,
            question=q["question"],
            options="\n".join(q["options"]),
            round=rnd,
            max_rounds=max_rounds,
            observations=_render_observations(history, g),
            rounds_left=left,
        )
        try:
            text = call_llm(
                prompt, cfg, g["answer"]["max_tokens"], g["answer"]["temperature"],
                role="solver_agent", frame_items=pending_frames or None,
            )
            pending_frames = []
        except UpstreamTimeoutError as exc:
            err = str(exc)
            error_kind = "timeout"
            break
        except Exception as exc:
            err = str(exc)
            error_kind = "error"
            break

        if parse_choice(text, q.get("n_options", 4)) and "TOOL:" not in text:
            break
        call = _parse_tool_call(text)
        if not call or left <= 0:
            break

        name, args = call
        if name == "read_video":
            obs, frames = exec_read_video(q, args, g, rv_calls, located)
            mids: list[str] = []
            if frames:
                rv_calls += 1
                frame_count += len(frames)
                try:
                    window = (float(args.get("start_sec")), float(args.get("end_sec")))
                    replay_windows.append([window[0], window[1]])
                    _record_frame_discovery(q, window, len(frames))
                except (TypeError, ValueError):
                    pass
                pending_frames = frames
        else:
            rt0 = time.time()
            obs, mids = exec_tool(tk, name, args, g, node_ids=located_micro)
            retrieval_latency += time.time() - rt0
        n_rounds += 1
        for m in mids:
            if m not in located:
                located.append(m)
        arg_s = ", ".join(f"{k}={v!r}" for k, v in list(args.items())[:3])
        history.append(
            {"call": f"{name}({arg_s})", "obs": obs, "macros": mids, "n_macros": len(mids)}
        )

    pred = parse_choice(text, q.get("n_options", 4))
    pa = parse_anchor(text)
    correct = pred == q["answer"]
    a_ok = anchor_ok(pa, q["anchor"])
    return {
        "qid": q["qid"],
        "video_key": q["video_key"],
        "type": q["type"],
        "videomme_task_type": q.get("videomme_task_type"),
        "cgbench_class": q.get("cgbench_class"),
        "split": q.get("split"),
        "mode": "agent",
        "gold": q["answer"],
        "pred": pred,
        "correct": None if error_kind == "timeout" else bool(correct),
        "error_kind": error_kind,
        "anchor_ok": bool(a_ok),
        "strict_correct": None if error_kind == "timeout" else bool(correct and a_ok),
        "pred_anchor": pa,
        "gold_anchor": q["anchor"],
        "located": located,
        "gold_macro_hit": _gold_hit(q, located),
        "located_micro": located_micro,
        "gold_micro_hit": _gold_micro_hit(q, located_micro),
        "micro_anchor_ok": micro_anchor_ok(pa, q["anchor"]),
        "granularity": q.get("granularity", "macro"),
        "evidence_len": sum(len(h["obs"]) for h in history),
        "tool_rounds": n_rounds,
        "tool_calls": [h["call"] for h in history],
        "llm_calls": n_rounds + 1,
        "read_video_calls": rv_calls,
        "read_video_frames": frame_count,
        "read_video_windows": replay_windows,
        "frame_tokens_est": frame_count * 750,
        "retrieval_latency_s": round(retrieval_latency, 4),
        "retrieval_metrics": _metric_delta(dense_before, _index_metrics(tk)),
        "latency_s": round(time.time() - t0, 2),
        "error": err,
        "raw": text[:600],
    }

def _gold_hit(q: dict, located: list[str] | None) -> bool:
    if not located:
        return False
    a = q.get("anchor") or {}
    needed = a.get("macro_ids")
    if needed:
        return any(m in located for m in needed)
    return a.get("macro_id") in located

def _norm_micro_id(x) -> str:
    return str(x).rsplit(":", 1)[-1]

def _gold_micro_hit(q: dict, located_micro: list[str] | None) -> bool | None:
    a = q.get("anchor") or {}
    needed = a.get("micro_ids")
    if not needed:
        return None
    if not located_micro:
        return False
    got = {_norm_micro_id(x) for x in located_micro}
    return any(_norm_micro_id(m) in got for m in needed)

_ABSTAIN_RE = re.compile(
    r"^(?:n\s*/?\s*a|not\s+(?:available|applicable|mentioned|specified)|"
    r"none\s+of\s+(?:the|these|them|the\s+above)|cannot\s+be\s+determined|"
    r"unknown)\.?$",
    re.I,
)
_OPTION_LABEL_RE = re.compile(r"^\s*[A-Z]\s*[.):]\s*")

def _option_text(opt: object) -> str:
    return _OPTION_LABEL_RE.sub("", str(opt or "")).strip()

def _abstain_letter(q: dict, pred: str | None) -> str | None:
    if not pred:
        return None
    opts = q.get("options") or []
    try:
        idx = ord(str(pred).strip().upper()[:1]) - ord("A")
    except (TypeError, IndexError):
        return None
    if idx < 0 or idx >= len(opts):
        return None
    return str(pred).strip().upper()[:1] if _ABSTAIN_RE.match(_option_text(opts[idx])) else None

def _abstain_followup(q: dict, pred: str) -> str:
    opts = q.get("options") or []
    concrete = [o for i, o in enumerate(opts) if chr(ord("A") + i) != pred]
    listing = ", ".join(f"{chr(ord('A') + i)}" for i in range(len(opts)) if chr(ord("A") + i) != pred)
    samples = "; ".join(_option_text(o) for o in concrete[:4])
    return (
        f"\n\nYour previous answer was {pred}, which is an abstention. Abstentions are only "
        "correct when the evidence positively rules out every concrete option -- a memory that "
        "merely never mentioned the detail rules nothing out, and thin evidence is not a licence "
        "to abstain. Re-answer now: pick the single most likely concrete option "
        f"({listing}" + (f" -- e.g. {samples}" if samples else "") + f"). "
        f"Only if you are certain none of them can be right, repeat {pred}. "
        "Keep the same output format: Evidence: MM:SS-MM:SS on the second-to-last line, "
        "Answer: X on the last line."
    )

def solve_one(q: dict, g: dict, cfg: dict, mode: str, tk=None) -> dict:
    if mode == "retrieve" and g.get("answer", {}).get("mode") == "agent" and tk is not None:
        return solve_one_agent(q, g, cfg, tk)

    t0 = time.time()
    dense_before = _index_metrics(tk) if tk is not None else {}
    retrieval_t0 = time.time()
    probe: dict = {}
    evidence, ids = build_evidence(q, g, mode, tk, probe=probe)
    retrieval_latency = time.time() - retrieval_t0 if mode in ("retrieve", "gold") else 0.0

    frames: list = []
    replay_windows: list[list[float]] = []
    learned_tool = None
    hard_replay = mode == "retrieve" and tk is not None
    replay_gate = "not_required"
    replay_rounds = 0
    replay_need_more = False
    second_look_armed = False
    round2_windows: list[list[float]] = []
    replay_fallback = False
    replay_segments = 0
    replay_declined_override = False
    frame_budget: dict = {"narrow": 0, "second_look": 0, "core_skim": 0,
                          "clipped_by_total_cap": 0, "cap": 0, "skipped": []}
    plan_note = "n/a (no replay)"
    content_filtered = False
    content_filter_error = None
    ret_spans = probe.get("spans") or {}
    window_src: dict = {}
    win_focus: list = []
    focus_src: dict = {}

    tool_stage: dict = {}
    if memory_tools.stage_enabled(g, mode, tk):
        try:
            tool_stage = memory_tools.run_tool_stage(
                q, g, cfg, tk,
                enabled=enabled_tools(g), specs=TOOL_SPECS,
                base_ids=ids, base_spans=ret_spans, evidence=evidence,
            )
        except Exception as exc:
            tool_stage = {"stop_reason": "stage_crashed", "error": str(exc)[:200],
                          "selected": [], "selected_spans": {}, "rounds": []}
        if tool_stage.get("evidence_text"):
            evidence += tool_stage["evidence_text"]
        if tool_stage.get("selected"):
            ids, ret_spans = memory_tools.merge_findings(
                ids, ret_spans, tool_stage,
                cap=int(((g.get("policy") or {}).get("memory_tools") or {})
                        .get("max_candidates_after_merge", 8)),
            )

    q_route = video_routing.route_of(q, g)

    def rvq(key: str, default):
        return video_routing.gene(g, f"policy.read_video.{key}", q_route, default)

    if hard_replay:
        tool_evidence, learned_tool = _run_learned_pipeline_tool(q, tk)
        evidence += tool_evidence
        rv = g.get("policy", {}).get("read_video") or {}
        sel_mode = str(rv.get("selection_mode", "model")).lower()
        cap = max(1, int(rvq("max_segments", 3)))
        stage_sel = [c for c in (tool_stage.get("selected") or []) if c]
        stage_drives = bool(((g.get("policy") or {}).get("memory_tools") or {})
                            .get("selection_drives_replay", True))
        if stage_sel and stage_drives and sel_mode != "top1":
            selected = stage_sel[:cap]
            plan_note = (f"tool stage selected {len(selected)} segment(s): "
                         f"{','.join(selected)} | why: {tool_stage.get('select_why') or 'n/a'}")
        elif sel_mode == "top1":
            selected = list(ids)[:1]
            plan_note = "fixed top-1 (selection_mode=top1)"
        else:
            selected, plan_note = _plan_replay_segments(q, g, cfg, ids, cap, tk=tk,
                                                        spans=ret_spans)
            if selected is None:
                selected, plan_note = list(ids)[:1], f"{plan_note} -> fell back to top-1"
            replay_gate = "model_declined" if not selected else replay_gate
        replay_declined_override = False
        if not selected and bool(rvq("require_replay_on_retrieve", True)):
            selected = list(ids)[:1]
            if selected:
                replay_declined_override = True
                plan_note = (f"{plan_note} -> declined ignored (require_replay_on_retrieve), "
                             f"watching top-1")
                replay_gate = "declined_override"
        if not selected:
            windows = []
            replay_gate = "no_candidates" if not ids else "model_declined"
        else:
            use_node = bool(rv.get("node_window", True))
            windows = _macro_windows(q["video_key"], selected,
                                     float(rvq("max_window_sec", 60)), cap,
                                     spans=ret_spans if use_node else {},
                                     pad=float(rvq("node_window_pad_sec", 8.0)),
                                     stats=window_src,
                                     focus_out=win_focus)
        replay_segments = len(windows)
        if windows:
            note, frames = exec_read_video(
                q, {"start_sec": windows[0][0], "end_sec": windows[0][1]}, g, 0, ids,
                force=True, windows=windows,
                focus=win_focus, focus_stats=focus_src,
            )
            if not frames:
                n0 = int(rvq("max_frames_per_call", 16))
                note2, frames2 = exec_read_video(
                    q, {"start_sec": windows[0][0], "end_sec": windows[0][1]}, g, 0, ids,
                    force=True, windows=windows,
                    fps_override=max(0.2, float(rvq("fps", 1.0)) / 2),
                    dim_override=512,
                    frames_override=max(4, n0 // 2),
                    focus=win_focus, focus_stats=focus_src,
                )
                if frames2:
                    note, frames = note2, frames2
                    replay_gate = "recovered_after_retry"
                else:
                    replay_gate = "extraction_failed"
            else:
                replay_gate = "passed"
            evidence += f"\nFRAME REPLAY: {note}"
            if frames:
                replay_windows = [[round(w[0], 3), round(w[1], 3)] for w in windows]
                _record_frame_discovery(q, windows[0], len(frames))
                replay_rounds = 1
                frame_budget["narrow"] = len(frames)
                if bool(rv.get("second_look", False)) and int(rvq("max_calls_per_question", 1)) >= 2:
                    second_look_armed = True
        else:
            if replay_gate not in ("model_declined", "no_candidates", "declined_override"):
                replay_gate = "no_macro"
            elif replay_gate == "declined_override":
                replay_gate = "no_macro"
        if not frames:
            replay_fallback = True
            reason = {
                "no_macro": "no usable time window could be cut from the retrieved segments",
                "no_candidates": "retrieval returned no candidate segment at all",
                "model_declined": "no frames were requested",
            }.get(replay_gate, "frame extraction failed (retried once at lower quality)")
            evidence += (f"\nFRAME REPLAY UNAVAILABLE ({reason}): you have NOT seen the video. "
                         "Do not invent visual details -- answer from the retrieved memory text "
                         "above and pick the option it supports.")

    core_frames_stats: dict = {}
    fr = (g.get("evidence", {}) or {}).get("frames") or {}
    total_cap = int(rvq("max_frames_total", 0))
    frame_budget["cap"] = total_cap
    if mode == "retrieve" and fr.get("enabled") and not content_filtered:
        if total_cap > 0 and len(frames) >= total_cap:
            frame_budget["skipped"].append("core_skim (total frame cap already reached)")
            core_frames_stats["skipped"] = "total_frame_cap"
            fr = {}
    if mode == "retrieve" and fr.get("enabled") and not content_filtered:
        try:
            import core_arm

            dur = core_arm.video_duration(q["video_key"])
            route_min = float(fr.get("route_max_min") or 0.0)
            ceiling = fr.get("route_max_cosine")
            top_cos = probe.get("top_cosine")
            route_ok = 0 < dur <= route_min * 60
            if route_ok and ceiling is not None and top_cos is not None:
                route_ok = float(top_cos) < float(ceiling)
            if route_ok:
                mem_core = Memory(q["video_key"])
                if mem_core.video_path and Path(mem_core.video_path).exists():
                    max_dim = int(fr.get("max_dim", 512))
                    cf = core_arm.skim_frames(
                        mem_core.video_path, dur,
                        chunk_sec=float(fr.get("chunk_sec", 300.0)),
                        fps=float(fr.get("fps", 1.0)),
                        budget=core_arm.budget_for_max_dim(max_dim),
                        max_chunks=int(fr.get("max_chunks", 0)),
                        stats=core_frames_stats,
                    )
                    if cf:
                        if total_cap > 0 and len(frames) + len(cf) > total_cap:
                            room = max(0, total_cap - len(frames))
                            frame_budget["clipped_by_total_cap"] = len(cf) - room
                            core_frames_stats["clipped_by_total_cap"] = len(cf) - room
                            if room <= 1:
                                cf = list(cf)[:room]
                            else:
                                step = (len(cf) - 1) / (room - 1)
                                cf = [cf[round(i * step)] for i in range(room)]
                        frame_budget["core_skim"] = len(cf)
                    if cf:
                        frames = list(frames) + list(cf)
                        evidence += (
                            f"\nFULL-VIDEO FRAME SKIM (qwen-mm-plugins-core read_video)\n"
                            f"{core_frames_stats.get('source_note', '')}\n"
                            f"{core_frames_stats.get('sampled_note', '')}\n"
                            "These frames cover the WHOLE video at a fixed interval and are "
                            "labelled with their timestamps. Use them to check what the "
                            "retrieved text only describes."
                        )
                        if not fr.get("no_media_info"):
                            mi = core_arm.media_info(mem_core.video_path)
                            if mi:
                                evidence += "\n" + mi
                                core_frames_stats["media_info_chars"] = len(mi)
        except Exception as exc:
            core_frames_stats["error"] = str(exc)[:200]

    sl_instruction = ""
    if second_look_armed:
        sl_instruction = (
            "\n\nSECOND LOOK (answer this BEFORE you answer the question):\n"
            "Do the frames above actually show what the question asks about?\n"
            "- If yes, write exactly: NEED_MORE: none\n"
            "- If no -- the frames are from the wrong moment, or too few, or the thing "
            "asked about is not visible in them -- write: NEED_MORE: START-END\n"
            "  with the seconds you need (e.g. NEED_MORE: 240-285). You may ask for one "
            "extra range of at most "
            f"{float(rv.get('max_window_sec', 60)):.0f} seconds; it will be fetched and "
            "shown to you, then you answer.\n"
            "Write that line first, then Answer: X")
    prompt = build_prompt(q, evidence, g, tail=sl_instruction)
    error_kind = None
    try:
        text = call_llm(
            prompt,
            cfg,
            g["answer"]["max_tokens"],
            g["answer"]["temperature"],
            role=MODE_ROLE.get(mode, mode),
            frame_items=frames or None,
        )
        err = None
    except ContentFilteredError as exc:
        frames = []
        replay_gate = "content_filtered_text_only"
        content_filtered = True
        content_filter_error = str(exc)[:200]
        try:
            text = call_llm(
                prompt,
                cfg,
                g["answer"]["max_tokens"],
                g["answer"]["temperature"],
                role=MODE_ROLE.get(mode, mode),
            )
            err = None
        except Exception as exc2:
            text, err = "", str(exc2)
            error_kind = "timeout" if isinstance(exc2, UpstreamTimeoutError) else "error"
    except UpstreamTimeoutError as exc:
        text, err = "", str(exc)
        error_kind = "timeout"
    except Exception as exc:
        text, err = "", str(exc)
        error_kind = "error"

    if second_look_armed and not err and frames:
        nm = _parse_need_more(text)
        if nm:
            replay_need_more = True
            w2 = _clip_window(nm, float(rvq("max_window_sec", 60)))
            if w2:
                ov_cap = float(rvq("second_look_overlap_reject", 0.8))
                for pw in replay_windows:
                    inter = max(0.0, min(w2[1], pw[1]) - max(w2[0], pw[0]))
                    span = max(1e-6, float(w2[1]) - float(w2[0]))
                    if inter / span >= ov_cap:
                        frame_budget["skipped"].append(
                            f"second_look ({inter / span:.0%} overlap with "
                            f"{pw[0]:.0f}-{pw[1]:.0f}s already read)")
                        w2 = None
                        break
            if w2 and total_cap > 0 and len(frames) >= total_cap:
                frame_budget["skipped"].append("second_look (total frame cap reached)")
                w2 = None
            if w2:
                n3, f3 = exec_read_video(
                    q, {"start_sec": w2[0], "end_sec": w2[1]}, g, 1, ids,
                    force=True, windows=[w2],
                    frames_override=max(4, int(rvq("max_frames_per_call", 16)) // 2),
                )
                if f3:
                    if total_cap > 0 and len(frames) + len(f3) > total_cap:
                        room = max(0, total_cap - len(frames))
                        frame_budget["clipped_by_total_cap"] += len(f3) - room
                        f3 = list(f3)[:room]
                if f3:
                    replay_rounds = 2
                    round2_windows = [[round(w2[0], 3), round(w2[1], 3)]]
                    replay_windows = replay_windows + round2_windows
                    frame_budget["second_look"] = len(f3)
                    frames = list(frames) + list(f3)
                    try:
                        text = call_llm(
                            build_prompt(q, evidence + f"\nFRAME REPLAY (second look): {n3}", g),
                            cfg,
                            g["answer"]["max_tokens"],
                            g["answer"]["temperature"],
                            role=MODE_ROLE.get(mode, mode),
                            frame_items=frames,
                        )
                    except Exception:
                        err = None

    pred = parse_choice(text, q.get("n_options", 4))
    if os.environ.get("VEH_ABSTAIN_TRACE"):
        with open(os.environ["VEH_ABSTAIN_TRACE"], "a", encoding="utf-8") as _tf:
            _tf.write("qid=%s pred=%r err=%r letter=%r nopts=%s opts=%r\n" % (
                q.get("qid"), pred, err, _abstain_letter(q, pred),
                len(q.get("options") or []), (q.get("options") or [])[:2]))
    abstain_rerouted = False
    abstain_reroute_attempted = False
    abstain_reroute_error = None
    if pred and not err and _abstain_letter(q, pred):
        abstain_reroute_attempted = True
        try:
            text2 = call_llm(
                prompt + _abstain_followup(q, pred),
                cfg,
                g["answer"]["max_tokens"],
                g["answer"]["temperature"],
                role=MODE_ROLE.get(mode, mode),
                frame_items=frames or None,
            )
            pred2 = parse_choice(text2, q.get("n_options", 4))
            if pred2 and pred2 != pred:
                abstain_rerouted = True
                pred, text = pred2, text2
        except Exception as exc:
            abstain_reroute_error = str(exc)[:200]
    pa = parse_anchor(text)
    correct = pred == q["answer"]
    a_ok = anchor_ok(pa, q["anchor"])
    return {
        "qid": q["qid"],
        "video_key": q["video_key"],
        "route": q_route,
        "video_duration_sec": q.get("video_duration_sec"),
        "exec_arm": "memory",
        "exec_path": f"solve.solve_one(mode={mode})",
        "exec_memory_dir": (str(active_memory_dir(q.get("video_key", "")))
                            if tk is not None else None),
        "exec_video": (q.get("video_path") if (replay_rounds or 0) > 0 else None),
        "type": q["type"],
        "videomme_task_type": q.get("videomme_task_type"),
        "cgbench_class": q.get("cgbench_class"),
        "split": q.get("split"),
        "mode": mode,
        "gold": q["answer"],
        "pred": pred,
        "correct": None if error_kind == "timeout" else bool(correct),
        "error_kind": error_kind,
        "anchor_ok": bool(a_ok),
        "strict_correct": None if error_kind == "timeout" else bool(correct and a_ok),
        "pred_anchor": pa,
        "gold_anchor": q["anchor"],
        "located": ids,
        "gold_macro_hit": _gold_hit(q, ids),
        "located_micro": probe.get("located_micro"),
        "gold_micro_hit": _gold_micro_hit(q, probe.get("located_micro")),
        "micro_anchor_ok": micro_anchor_ok(pa, q["anchor"]),
        "granularity": q.get("granularity", "macro"),
        "evidence_len": len(evidence),
        "top_cosine": probe.get("top_cosine"),
        "read_video_required": hard_replay,
        "replay_gate_status": replay_gate,
        "replay_declined_override": replay_declined_override,
        "tool_stage_ran": bool(tool_stage),
        "tool_stage_stop_reason": tool_stage.get("stop_reason"),
        "tool_stage_error": tool_stage.get("error"),
        "tool_stage_llm_calls": int(tool_stage.get("llm_calls") or 0),
        "tool_stage_tool_calls": int(tool_stage.get("tool_calls") or 0),
        "tool_stage_parse_errors": int(tool_stage.get("parse_errors") or 0),
        "tool_stage_tool_errors": int(tool_stage.get("tool_errors") or 0),
        "tool_stage_empty_results": int(tool_stage.get("empty_results") or 0),
        "tool_stage_degraded_calls": int(tool_stage.get("degraded_calls") or 0),
        "tool_stage_tools_used": [r.get("tool") for r in (tool_stage.get("rounds") or [])],
        "tool_stage_learned_tools": [r.get("tool") for r in (tool_stage.get("rounds") or [])
                                     if r.get("learned")],
        "tool_stage_selected": list(tool_stage.get("selected") or []),
        "tool_stage_selected_spans": {k: [round(v[0], 2), round(v[1], 2)]
                                      for k, v in (tool_stage.get("selected_spans") or {}).items()},
        "tool_stage_returned_macros": list(tool_stage.get("returned_macro_ids") or []),
        "tool_stage_returned_nodes": list(tool_stage.get("returned_node_ids") or []),
        "tool_stage_returned_macros_total": int(
            tool_stage.get("returned_macro_truncated")
            or len(tool_stage.get("returned_macro_ids") or [])),
        "tool_stage_rejected_selections": list(tool_stage.get("rejected_selections") or []),
        "tool_stage_parse_error_samples": list(tool_stage.get("parse_error_samples") or []),
        "tool_stage_latency_s": float(tool_stage.get("latency_s") or 0.0),
        "frames_from_narrow_replay": int(frame_budget.get("narrow") or 0),
        "frames_from_second_look": int(frame_budget.get("second_look") or 0),
        "frames_from_core_skim": int(frame_budget.get("core_skim") or 0),
        "frames_total_cap": int(frame_budget.get("cap") or 0),
        "frames_clipped_by_total_cap": int(frame_budget.get("clipped_by_total_cap") or 0),
        "frames_budget_skips": list(frame_budget.get("skipped") or []),
        "content_filtered": content_filtered,
        "content_filter_error": content_filter_error,
        "replay_fallback": replay_fallback,
        "replay_segments": replay_segments,
        "replay_plan": plan_note,
        "window_from_node_spans": int(window_src.get("node_spans") or 0),
        "window_from_macro_spans": int(window_src.get("macro_spans") or 0),
        "frames_focused_windows": int(focus_src.get("focused_windows") or 0),
        "frames_sub_ranges": int(focus_src.get("sub_ranges") or 0),
        "frames_backend": focus_src.get("backend"),
        "frames_route": focus_src.get("route"),
        "frame_focus_ranges": [[round(f[0], 2), round(f[1], 2)] for f in win_focus if f],
        "retrieval_spans": {k: [round(v[0], 2), round(v[1], 2)]
                            for k, v in (ret_spans or {}).items()},
        "abstain_rerouted": abstain_rerouted,
        "abstain_reroute_attempted": abstain_reroute_attempted,
        "abstain_reroute_error": abstain_reroute_error,
        "pred_abstain": bool(_abstain_letter(q, pred)),
        "read_video_calls": replay_rounds,
        "read_video_frames": len(frames),
        "frames_core_n": int(core_frames_stats.get("n_frames") or 0),
        "frames_sent_mb": float(core_frames_stats.get("sent_mb") or 0.0),
        "frames_core_fps_effective": core_frames_stats.get("fps_effective"),
        "frames_core_chunks": int(core_frames_stats.get("n_chunks") or 0),
        "frames_core_truncated": bool(core_frames_stats.get("truncated_by_bytes") or False),
        "frames_core_error": core_frames_stats.get("error"),
        "replay_second_look_armed": second_look_armed,
        "replay_need_more": replay_need_more,
        "replay_round2_windows": round2_windows,
        "read_video_windows": replay_windows,
        "frame_tokens_est": len(frames) * 750,
        "learned_tool": learned_tool,
        "retrieval_latency_s": round(retrieval_latency, 4),
        "retrieval_metrics": _metric_delta(dense_before, _index_metrics(tk)) if tk is not None else {},
        "latency_s": round(time.time() - t0, 2),
        "error": err,
        "raw": text[:600],
    }

def _fmt_dur(sec: float) -> str:
    sec = int(max(sec, 0))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"

def _progress_line(done: int, total: int, n_ok: int, n_scored: int,
                   n_timeout: int, n_err: int, t0: float) -> str:
    elapsed = time.time() - t0
    eta = elapsed / done * (total - done) if done else 0.0
    acc = f"{n_ok}/{n_scored}" + (f" ({n_ok / n_scored:.0%})" if n_scored else " (--)")
    bits = [f"acc={acc}"]
    if n_timeout:
        bits.append(f"timeout={n_timeout}")
    if n_err:
        bits.append(f"err={n_err}")
    bits.append(f"elapsed={_fmt_dur(elapsed)}")
    bits.append(f"eta={_fmt_dur(eta)}")
    return f"  [{done}/{total}] {done / total:>5.1%}  " + "  ".join(bits)

def run(
    questions: list[dict],
    g: dict,
    cfg: dict,
    mode: str = "retrieve",
    workers: int = 4,
    quiet: bool = False,
    progress_every: int = 50,
    on_record=None,
) -> list[dict]:
    tks: dict[str, object] = {}
    if mode in ("retrieve", "gold"):
        for vk in {q["video_key"] for q in questions}:
            tks[vk] = load_toolkit(vk)

    total = len(questions)
    t_run = time.time()
    n_ok = n_scored = n_timeout = n_err = 0
    recs: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(solve_one, q, g, cfg, mode, tks.get(q["video_key"])): q for q in questions}
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            recs.append(r)
            if on_record is not None:
                try:
                    on_record(r)
                except Exception as exc:
                    pass
            if not quiet:
                flag = "✓" if r["correct"] else "✗"
                a = "A" if r["anchor_ok"] else "-"
            else:
                if r.get("correct") is not None:
                    n_scored += 1
                    if r["correct"]:
                        n_ok += 1
                if r.get("error_kind") == "timeout":
                    n_timeout += 1
                elif r.get("error"):
                    n_err += 1
                if progress_every and (
                    i % progress_every == 0 or (i == total and total >= progress_every)
                ):
                    pass

    failed = [r for r in recs if r.get("error")]
    if failed:
        timed_out = [r for r in failed if r.get("error_kind") == "timeout"]
        counted = [r for r in failed if r.get("error_kind") != "timeout"]
        rate = len(failed) / len(recs)
        if timed_out:
            pass
        if counted:
            pass
        import collections as _c
        for msg, n in _c.Counter(
            str(r["error"]).split(" for url")[0][:90] for r in failed
        ).most_common(3):
            pass
        if rate > 0.05:
            pass
    return recs

BENCH_DIMS = {
    "by_videomme_task_type": "videomme_task_type",
    "by_cgbench_class": "cgbench_class",
}

def _group_stats(recs: list[dict], key: str) -> dict[str, dict]:
    buckets: dict[str, dict] = {}
    for r in recs:
        k = str(r.get(key) or "unmapped")
        d = buckets.setdefault(k, {"n": 0, "correct": 0, "strict": 0, "hit": 0})
        d["n"] += 1
        d["correct"] += bool(r.get("correct"))
        d["strict"] += bool(r.get("strict_correct"))
        d["hit"] += bool(r.get("gold_macro_hit", False))
        if r.get("gold_micro_hit") is not None:
            d["micro_n"] = d.get("micro_n", 0) + 1
            d["micro_hit"] = d.get("micro_hit", 0) + bool(r["gold_micro_hit"])
    out = {}
    for k, d in sorted(buckets.items(), key=lambda kv: -kv[1]["n"]):
        row = {
            "n": d["n"],
            "acc": round(d["correct"] / d["n"], 3),
            "strict": round(d["strict"] / d["n"], 3),
            "hit": round(d["hit"] / d["n"], 3),
        }
        if d.get("micro_n"):
            row["micro_n"] = d["micro_n"]
            row["micro_hit"] = round(d["micro_hit"] / d["micro_n"], 3)
        out[k] = row
    return out

def summarize(recs: list[dict]) -> dict:
    n = len(recs) or 1
    s = {
        "n": len(recs),
        "acc": round(sum(r["correct"] for r in recs) / n, 4),
        "strict_acc": round(sum(r["strict_correct"] for r in recs) / n, 4),
        "anchor_rate": round(sum(r["anchor_ok"] for r in recs) / n, 4),
        "gold_macro_hit_rate": round(sum(r.get("gold_macro_hit", False) for r in recs) / n, 4),
        "retrieval": {
            "latency_sec_total": round(sum(r.get("retrieval_latency_s", 0) for r in recs), 4),
            "latency_sec_mean": round(sum(r.get("retrieval_latency_s", 0) for r in recs) / n, 4),
            "dense_attempts": sum(r.get("retrieval_metrics", {}).get("dense_attempts", 0) for r in recs),
            "dense_successes": sum(r.get("retrieval_metrics", {}).get("dense_successes", 0) for r in recs),
            "dense_timeouts": sum(r.get("retrieval_metrics", {}).get("dense_timeouts", 0) for r in recs),
            "bm25_fallbacks": sum(r.get("retrieval_metrics", {}).get("bm25_fallbacks", 0) for r in recs),
            "embedding_latency_sec": round(sum(r.get("retrieval_metrics", {}).get("embedding_latency_sec", 0) for r in recs), 4),
        },
        "replay": {
            "required": sum(bool(r.get("read_video_required")) for r in recs),
            "calls": sum(r.get("read_video_calls", 0) for r in recs),
            "frames": sum(r.get("read_video_frames", 0) for r in recs),
            "frame_tokens_est": sum(r.get("frame_tokens_est", 0) for r in recs),
            "gate_passed": sum(1 for r in recs if r.get("replay_gate_status") == "passed"),
            "gate_recovered_after_retry": sum(
                1 for r in recs if r.get("replay_gate_status") == "recovered_after_retry"),
            "gate_extraction_failed": sum(
                1 for r in recs if r.get("replay_gate_status") == "extraction_failed"),
            "gate_no_macro": sum(1 for r in recs if r.get("replay_gate_status") == "no_macro"),
            "gate_no_candidates": sum(
                1 for r in recs if r.get("replay_gate_status") == "no_candidates"),
            "declined_overridden": sum(bool(r.get("replay_declined_override")) for r in recs),
            "answered_from_text_only": sum(bool(r.get("replay_fallback")) for r in recs),
            "segments": sum(r.get("replay_segments", 0) for r in recs),
            "frames_from_narrow_replay": sum(r.get("frames_from_narrow_replay", 0) for r in recs),
            "frames_from_second_look": sum(r.get("frames_from_second_look", 0) for r in recs),
            "frames_from_core_skim": sum(r.get("frames_from_core_skim", 0) for r in recs),
            "frames_clipped_by_total_cap": sum(
                r.get("frames_clipped_by_total_cap", 0) for r in recs),
            "frames_budget_skips": sum(len(r.get("frames_budget_skips") or []) for r in recs),
            "windows_from_node_spans": sum(r.get("window_from_node_spans", 0) for r in recs),
            "windows_from_macro_spans": sum(r.get("window_from_macro_spans", 0) for r in recs),
            "mean_window_sec": round(
                sum((w[1] - w[0]) for r in recs for w in (r.get("read_video_windows") or []))
                / max(1, sum(len(r.get("read_video_windows") or []) for r in recs)), 2),
        },
        "by_type": _group_stats(recs, "type"),
    }
    ran = [r for r in recs if r.get("tool_stage_ran")]
    if ran:
        sel = [r for r in ran if r.get("tool_stage_selected")]
        s["tool_stage"] = {
            "questions": len(ran),
            "llm_calls": sum(r.get("tool_stage_llm_calls", 0) for r in ran),
            "tool_calls": sum(r.get("tool_stage_tool_calls", 0) for r in ran),
            "tool_errors": sum(r.get("tool_stage_tool_errors", 0) for r in ran),
            "parse_errors": sum(r.get("tool_stage_parse_errors", 0) for r in ran),
            "empty_results": sum(r.get("tool_stage_empty_results", 0) for r in ran),
            "embedding_degraded_calls": sum(r.get("tool_stage_degraded_calls", 0) for r in ran),
            "latency_sec_total": round(sum(r.get("tool_stage_latency_s", 0.0) for r in ran), 3),
            "questions_with_a_selection": len(sel),
            "selected_total": sum(len(r.get("tool_stage_selected") or []) for r in ran),
            "returned_macros_total": sum(
                r.get("tool_stage_returned_macros_total", 0) for r in ran),
            "selections_rejected": sum(
                len(r.get("tool_stage_rejected_selections") or []) for r in ran),
            "learned_tool_calls": sum(
                len(r.get("tool_stage_learned_tools") or []) for r in ran),
            "stop_reasons": dict(Counter(
                str(r.get("tool_stage_stop_reason")) for r in ran).most_common()),
            "tools_used": dict(Counter(
                t for r in ran for t in (r.get("tool_stage_tools_used") or [])).most_common()),
        }
    for out_key, field in BENCH_DIMS.items():
        s[out_key] = _group_stats(recs, field)
    micro = _micro_stats(recs)
    if micro:
        s["micro"] = micro
    grans: dict[str, list[dict]] = {}
    for r in recs:
        grans.setdefault(str(r.get("granularity") or "macro"), []).append(r)
    if len(grans) > 1:
        s["by_granularity"] = {
            k: {
                "n": len(v),
                "acc": round(sum(bool(x.get("correct")) for x in v) / len(v), 4),
                "strict_acc": round(sum(bool(x.get("strict_correct")) for x in v) / len(v), 4),
            }
            for k, v in sorted(grans.items())
        }
    return s

def _micro_stats(recs: list[dict]) -> dict | None:
    judged = [r for r in recs if r.get("gold_micro_hit") is not None]
    if not judged:
        return None
    n = len(judged)
    aok = [r for r in judged if r.get("micro_anchor_ok") is not None]
    return {
        "n": n,
        "gold_micro_hit_rate": round(sum(bool(r["gold_micro_hit"]) for r in judged) / n, 4),
        "gold_macro_hit_rate_same_q": round(
            sum(bool(r.get("gold_macro_hit")) for r in judged) / n, 4),
        "micro_anchor_rate": (
            round(sum(bool(r["micro_anchor_ok"]) for r in aok) / len(aok), 4) if aok else None
        ),
        "acc": round(sum(bool(r.get("correct")) for r in judged) / n, 4),
        "strict_acc": round(sum(bool(r.get("strict_correct")) for r in judged) / n, 4),
    }

def print_report(s: dict, title: str = "") -> None:
    if title:
        pass
    ret, replay = s.get("retrieval", {}), s.get("replay", {})
    ts = s.get("tool_stage")
    if ts:
        if ts.get("tools_used"):
            pass
    if replay.get("frames_from_core_skim") or replay.get("frames_clipped_by_total_cap") \
            or replay.get("declined_overridden"):
        pass
    ms = s.get("micro")
    if ms:
        mar = ms["micro_anchor_rate"]
    views = [("internal taxonomy (drives evolution)", "by_type")]
    views += [("Video-MME task type", "by_videomme_task_type"),
              ("CG-Bench class", "by_cgbench_class")]
    for label, key in views:
        rows = s.get(key) or {}
        if not rows:
            continue
        for k, d in rows.items():
            mark = "  <- n<5, not statistically usable" if d["n"] < 5 else ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence", choices=EVIDENCE_MODES, default="retrieve")
    ap.add_argument("--split", default=None, help="evolve / heldout / empty = all")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--routes", default="",
                    help="only solve questions on these routes (comma-separated). "
                         "Empty = no filter. "
                         "Use video_memory to restrict this entry point to the >=30min "
                         "chain it is designed for; <30min is core_arm.py's job.")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--questions", default=str(STORE / "questions_verified.jsonl"),
                    help="default is the VERIFIED store; questions.jsonl is ask.py's raw "
                         "output and has not been through the Blind/Oracle filter")
    ap.add_argument("--genome", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    g = load_genome(args.genome)
    cfg = load_cfg()
    qpath = Path(args.questions)
    qs = load_questions(qpath, args.split)
    if not qs:
        raise SystemExit(f"No questions with split={args.split or 'all'} in {qpath}; run ask.py + verify.py first")
    if args.limit:
        qs = qs[: args.limit]

    if getattr(args, "routes", ""):
        want = {r.strip() for r in args.routes.split(",") if r.strip()}
        bad = want - set(video_routing.ROUTES)
        if bad:
            raise SystemExit(f"unknown route(s) {sorted(bad)}; expected {video_routing.ROUTES}")
        if not any(q.get("route") for q in qs):
            pass
        else:
            before = len(qs)
            dropped = Counter((q.get("route") or "unrouted") for q in qs
                                 if (q.get("route") or "") not in want)
            qs = [q for q in qs if (q.get("route") or "") in want]
            if not qs:
                raise SystemExit(
                    f"no questions left after the route filter; this set is {dict(dropped)}. "
                    f"For <30min videos use harness/core_arm.py instead.")

    if not any(q.get("verdict") or q.get("blind_correct") is not None for q in qs):
        pass
    missing = sum(1 for q in qs if not q.get("videomme_task_type"))
    if missing:
        pass

    reset_cost()
    recs = run(qs, g, cfg, args.evidence, args.workers)
    s = summarize(recs)
    cost = llm_cost()
    print_report(s, f"genome v{g['version']}  mode={args.evidence}  split={args.split or 'all'}")
    t = cost["total"]

    tag = f"{args.evidence}_{args.split or 'all'}" + (f"_n{args.limit}" if args.limit else "")
    out = Path(args.out) if args.out else STORE / "runs" / f"solve_v{g['version']}_{tag}.json"
    out.write_text(
        json.dumps({"genome_version": g["version"], "mode": args.evidence,
                    "split": args.split or "all", "limit": args.limit or None,
                    "summary": s, "cost": cost, "records": recs},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

def protocol_fingerprint(g: dict, cfg: dict, mode: str, split: str,
                         questions: list[dict], workers: int) -> dict:
    def _canon(o) -> str:
        return json.dumps(o, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    qids = sorted(str(q.get("qid") or q.get("question_id") or "") for q in questions)
    parts = {
        "genome": _canon(g),
        "cfg": _canon(cfg or {}),
        "mode": str(mode),
        "split": str(split),
        "workers": str(int(workers or 0)),
        "n_questions": str(len(questions)),
        "qids_sha": hashlib.sha256("\n".join(qids).encode("utf-8")).hexdigest()[:16],
    }
    return {
        "fingerprint": hashlib.sha256(_canon(parts).encode("utf-8")).hexdigest(),
        "parts": parts,
    }

if __name__ == "__main__":
    main()
