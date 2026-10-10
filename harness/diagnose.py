from __future__ import annotations

import argparse
import copy
import json
import re
from collections import Counter
from pathlib import Path

import eval_router
import run_cache
import solve
from common import STORE, load_cfg, load_genome, load_questions

CATEGORIES = ("retrieval_miss", "evidence_truncated", "memory_missing_or_reasoning", "guess_no_anchor")

def _cached_run(questions: list[dict], g: dict, cfg: dict, mode: str,
                workers: int) -> list[dict]:
    if not questions:
        return []
    fp = run_cache.fingerprint(g)
    tag = f"diagnose:{mode}"
    hit, miss = run_cache.split(fp, tag, questions)
    if hit:
        pass
    if not miss:
        return hit
    fresh = solve.run(miss, g, cfg, mode, workers, quiet=True,
                      on_record=lambda r: run_cache.put(fp, tag, r))
    return hit + fresh

def cross_tab(recs: list[dict], key: str) -> dict[str, dict]:
    out: dict[str, Counter] = {}
    for r in recs:
        out.setdefault(str(r.get(key) or "unmapped"), Counter())[r["category"]] += 1
    return {k: {"n_failed": sum(c.values()), **dict(c)}
            for k, c in sorted(out.items(), key=lambda kv: -sum(kv[1].values()))}

def print_cross(rows: dict[str, dict], label: str) -> None:
    if not rows:
        return
    head = "".join(f"{c[:14]:>16}" for c in CATEGORIES)
    for k, d in rows.items():
        cells = "".join(f"{d.get(c, 0):>16}" for c in CATEGORIES)

CORE_FRAME_CAP = 192
OFFICIAL_SKIM_FPS = 1.0

def _parse_failed(rec: dict) -> bool:
    if rec.get("err"):
        return False
    return not str(rec.get("pred") or "").strip()

def _fallback_parsed(rec: dict) -> bool:
    if rec.get("err") or _parse_failed(rec):
        return False
    text = str(rec.get("answer_text") or "")
    if not text.strip():
        return False
    return not re.search(r"Answer\s*:\s*\**\s*[A-Z]", text, re.I)

def _denser_genome(g: dict) -> dict:
    ng = copy.deepcopy(g)
    fr = ng.setdefault("evidence", {}).setdefault("frames", {})
    fr["fps"] = max(2.0, float(fr.get("fps") or OFFICIAL_SKIM_FPS) * 2)
    fr["budget"] = "large"
    return ng

def ablate_core_skim(core_wrong: list[dict], g: dict, cfg: dict,
                     base: dict[str, dict], workers: int) -> dict[str, str]:
    todo = [q for q in core_wrong
            if (base.get(q["qid"]) or {}).get("arm") == "core_skim"
            and not (base.get(q["qid"]) or {}).get("err")]
    if not todo:
        return {}

    ng = _denser_genome(g)
    fp = run_cache.fingerprint(ng)
    tag = "diagnose:core_dense"
    hit, miss = run_cache.split(fp, tag, todo)
    if hit:
        pass
    recs = list(hit)
    if miss:
        recs += eval_router.run_pool(miss, ng, cfg, workers, quiet=True,
                                     cache_tag=tag)

    by_qid = {r.get("qid"): r for r in recs}
    out: dict[str, str] = {}
    for q in todo:
        r = by_qid.get(q["qid"]) or {}
        if r.get("err"):
            continue
        out[q["qid"]] = ("frames_insufficient" if r.get("correct")
                         else "reasoning_or_unseen")
    n_fix = sum(1 for v in out.values() if v == "frames_insufficient")
    return out

def profile_core_failures(core_wrong: list[dict], base: dict[str, dict]) -> dict:
    prof: dict[str, dict] = {}
    for q in core_wrong:
        rec = base.get(q["qid"]) or {}
        arm = rec.get("arm") or "unknown"
        p = prof.setdefault(arm, {
            "n": 0, "by_type": Counter(), "by_duration": Counter(),
            "parse_failed": 0, "fallback_parsed": 0, "infra_error": 0,
            "frame_capped": 0, "bytes_truncated": 0,
            "coarse_failed": 0, "detail_failed": 0, "no_detail_span": 0,
        })
        p["n"] += 1
        p["by_type"][q.get("type") or "?"] += 1
        p["by_duration"][rec.get("duration_bucket") or "?"] += 1

        if rec.get("err"):
            p["infra_error"] += 1
            continue
        if _parse_failed(rec):
            p["parse_failed"] += 1

        if arm == "core_chunked":
            if rec.get("frames_coarse_failed"):
                p["coarse_failed"] += 1
            if rec.get("frames_detail_failed"):
                p["detail_failed"] += 1
            if not rec.get("frames_n_detail_spans"):
                p["no_detail_span"] += 1
        else:
            if _fallback_parsed(rec):
                p["fallback_parsed"] += 1
            if int(rec.get("frames_n_frames") or 0) >= CORE_FRAME_CAP:
                p["frame_capped"] += 1
            if rec.get("frames_truncated_by_bytes"):
                p["bytes_truncated"] += 1
    return prof

def print_core_profile(prof: dict) -> None:
    if not prof:
        return
    for arm in sorted(prof):
        p = prof[arm]
        n = p["n"]
        bits = []
        if p["infra_error"]:
            bits.append(f"call failures {p['infra_error']}")
        if p["parse_failed"]:
            bits.append(f"answers that did not parse {p['parse_failed']}")
        if arm == "core_chunked":
            if p["coarse_failed"]:
                bits.append(f"coarse pass had window failures {p['coarse_failed']}")
            if p["detail_failed"]:
                bits.append(f"detailed pass had segment failures {p['detail_failed']}")
            if p["no_detail_span"]:
                bits.append(f"coarse pass selected no suspicious segment {p['no_detail_span']}")
        else:
            if p["fallback_parsed"]:
                bits.append(f"answers recovered by the fallback parser {p['fallback_parsed']}"
                            f" (no 'Answer: X' written; fixable in the prompt)")
            if p["frame_capped"]:
                bits.append(f"frame count hit the {CORE_FRAME_CAP} cap {p['frame_capped']}")
            if p["bytes_truncated"]:
                bits.append(f"byte-bisecting downsample {p['bytes_truncated']}")
        top_t = "  ".join(f"{k}={v}" for k, v in p["by_type"].most_common(6))
        top_d = "  ".join(f"{k}={v}" for k, v in sorted(p["by_duration"].items()))

def diagnose(questions: list[dict], g: dict, cfg: dict, workers: int = 4) -> dict:
    base = {r["qid"]: r for r in eval_router.run_pool(questions, g, cfg, workers, quiet=True,
                                                      cache_tag="diagnose")}
    wrong = [q for q in questions if not base[q["qid"]]["correct"]]

    if not wrong:
        return {
            "base": eval_router.summarize_mixed(list(base.values())),
            "base_records": list(base.values()),
            "base_acc_by_route": eval_router.acc_by_route(list(base.values())),
            "trace_by_type": eval_router.trace_by_type(list(base.values())),
            "diagnosis": {},
            "records": [],
            "core_wrong_qids": [],
            "core_profile": {},
            "core_ablation": {},
            "core_ablation_counts": {},
        }

    mem_wrong = [q for q in wrong if not eval_router.is_core(q)]
    core_wrong = [q for q in wrong if eval_router.is_core(q)]
    core_profile: dict = {}
    core_ablation: dict[str, str] = {}
    if core_wrong:
        core_profile = profile_core_failures(core_wrong, base)
        print_core_profile(core_profile)
        core_ablation = ablate_core_skim(core_wrong, g, cfg, base, workers)

    gold = {r["qid"]: r for r in _cached_run(mem_wrong, g, cfg, "gold", workers)}

    still = [q for q in mem_wrong if not gold[q["qid"]]["correct"]]
    all_r = {}
    if still:
        all_r = {r["qid"]: r for r in _cached_run(still, g, cfg, "all", workers)}
    else:
        pass

    recs = []
    for q in mem_wrong:
        qid = q["qid"]
        if gold[qid]["correct"]:
            cat = "retrieval_miss"
        elif all_r.get(qid, {}).get("correct"):
            cat = "evidence_truncated"
        else:
            cat = "memory_missing_or_reasoning"
        recs.append(
            {
                "qid": qid,
                "type": q["type"],
                "videomme_task_type": q.get("videomme_task_type"),
                "cgbench_class": q.get("cgbench_class"),
                "category": cat,
                "gold_macro": q["anchor"]["macro_id"],
                "located": base[qid]["located"],
                "evidence_len": base[qid]["evidence_len"],
            }
        )

    lucky = [q for q in questions
             if base[q["qid"]]["correct"]
             and not eval_router.is_core(q)
             and not base[q["qid"]]["anchor_ok"]]
    for q in lucky:
        recs.append({"qid": q["qid"], "type": q["type"],
                     "videomme_task_type": q.get("videomme_task_type"),
                     "cgbench_class": q.get("cgbench_class"),
                     "category": "guess_no_anchor",
                     "gold_macro": q["anchor"]["macro_id"], "located": base[q["qid"]]["located"],
                     "evidence_len": base[q["qid"]]["evidence_len"]})

    counts = Counter(r["category"] for r in recs)

    return {
        "base": eval_router.summarize_mixed(list(base.values())),
        "base_records": list(base.values()),
        "base_acc_by_route": eval_router.acc_by_route(list(base.values())),
        "trace_by_type": eval_router.trace_by_type(list(base.values())),
        "diagnosis": dict(counts),
        "core_wrong_qids": [q["qid"] for q in core_wrong],
        "core_profile": {
            arm: {**{k: v for k, v in p.items()
                     if k not in ("by_type", "by_duration")},
                  "by_type": dict(p["by_type"]),
                  "by_duration": dict(p["by_duration"])}
            for arm, p in core_profile.items()
        },
        "core_ablation": core_ablation,
        "core_ablation_counts": dict(Counter(core_ablation.values())),
        "by_type": cross_tab(recs, "type"),
        "by_videomme_task_type": cross_tab(recs, "videomme_task_type"),
        "by_cgbench_class": cross_tab(recs, "cgbench_class"),
        "dominant": counts.most_common(1)[0][0] if counts else None,
        "records": recs,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="evolve")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--questions", default=str(STORE / "questions_verified.jsonl"))
    args = ap.parse_args()

    g, cfg = load_genome(), load_cfg()
    qs = load_questions(Path(args.questions), args.split)
    if not qs:
        raise SystemExit(f"No questions with split={args.split}; run ask.py + verify.py first")
    if args.limit:
        qs = qs[: args.limit]

    out = diagnose(qs, g, cfg, args.workers)

    solve.print_report(out["base"], f"diagnose baseline  genome v{g['version']}  split={args.split}")

    total_failed = sum(out["diagnosis"].values()) or 1
    for cat, n in sorted(out["diagnosis"].items(), key=lambda kv: -kv[1]):
        pass

    for label, key in ((" by Video-MME task type", "by_videomme_task_type"),
                       (" by CG-Bench class", "by_cgbench_class"),
                       (" by internal type", "by_type")):
        print_cross(out.get(key) or {}, label)

    p = STORE / "runs" / f"diagnose_v{g['version']}_{args.split}.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
