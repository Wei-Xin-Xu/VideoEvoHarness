from __future__ import annotations

import functools
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import core_arm
import run_cache
import solve
import video_routing
from common import STORE

CORE_ROUTES = frozenset({video_routing.ROUTE_DIRECT, video_routing.ROUTE_CHUNKED})

ARM_BY_ROUTE = {
    video_routing.ROUTE_DIRECT: "core_skim",
    video_routing.ROUTE_CHUNKED: "core_chunked",
}

CORE_ARM = "core_skim"

def arm_for(q: dict) -> str:
    return ARM_BY_ROUTE.get(q.get("route") or "", CORE_ARM)

def _group_by_arm(core_q: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for q in core_q:
        out.setdefault(arm_for(q), []).append(q)
    return out

def is_core(q: dict) -> bool:
    return (q.get("route") or "") in CORE_ROUTES

def split_pool(questions: list[dict]) -> tuple[list[dict], list[dict]]:
    core = [q for q in questions if is_core(q)]
    mem = [q for q in questions if not is_core(q)]
    return core, mem

_CORE_PROGRESS_EVERY = 25

def _run_core(core_q: list[dict], g: dict, cfg: dict, workers: int,
              quiet: bool, on_record=None) -> list[dict]:
    recs: list[dict] = []
    total = len(core_q)
    t_run = time.time()
    n_ok = n_scored = n_timeout = n_err = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futs = {pool.submit(core_arm.solve_one_core, q, g, cfg, arm_for(q)): q
                for q in core_q}
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
                continue
            if r.get("correct") is not None:
                n_scored += 1
                if r["correct"]:
                    n_ok += 1
            if r.get("err"):
                n_err += 1
            if i % _CORE_PROGRESS_EVERY == 0 or i == total:
                line = solve._progress_line(i, total, n_ok, n_scored,
                                            n_timeout, n_err, t_run)
    return recs

def run_pool(questions: list[dict], g: dict, cfg: dict, workers: int = 4,
             quiet: bool = True, dump: "Path | str | None" = "auto",
             cache_tag: "str | None" = None,
             dump_tag: "str | None" = None) -> list[dict]:
    core_q, mem_q = split_pool(questions)
    if not core_q and not mem_q:
        return []

    cached: list[dict] = []
    on_rec = None
    on_rec_core = None
    if cache_tag is not None and run_cache.enabled():
        fp = run_cache.fingerprint(g)
        hit_core = []
        rest_core: list[dict] = []
        for arm, group in _group_by_arm(core_q).items():
            hit, miss = run_cache.split(fp, f"{cache_tag}:{arm}", group)
            hit_core += hit
            rest_core += miss
        core_q = rest_core
        hit_mem, mem_q = run_cache.split(fp, cache_tag, mem_q)
        cached = hit_core + hit_mem
        if cached:
            pass
        on_rec = functools.partial(run_cache.put, fp, cache_tag, label=dump_tag)

        def on_rec_core(rec: dict, _fp=fp, _tag=cache_tag, _label=dump_tag) -> None:
            arm = rec.get("arm") or CORE_ARM
            run_cache.put(_fp, f"{_tag}:{arm}", rec, label=_label)

    recs: list[dict] = list(cached)
    extra = {"on_record": on_rec} if on_rec is not None else {}
    extra_core = {"on_record": on_rec_core} if on_rec_core is not None else {}
    if mem_q:
        recs += solve.run(mem_q, g, cfg, "retrieve", workers, quiet=quiet, **extra)
    if core_q:
        recs += _run_core(core_q, g, cfg, workers, quiet, **extra_core)

    order = {q["qid"]: i for i, q in enumerate(questions)}
    recs.sort(key=lambda r: order.get(r.get("qid"), 1 << 30))

    by_qid = {q["qid"]: q for q in questions}
    bad = [r for r in recs if _stamp_exec(r, by_qid.get(r.get("qid")) or {})]
    if bad:
        for r in bad[:5]:
            pass
        if len(bad) > 5:
            pass

    path = _resolve_dump(dump)
    if path is not None:
        _append_dump(path, recs, tag=dump_tag or cache_tag)
    return recs

def _stamp_exec(rec: dict, q: dict) -> bool:
    want = "core" if is_core(q) else "memory"
    rec["exec_expected"] = want
    mismatch = rec.get("exec_arm") != want
    if mismatch:
        rec["exec_mismatch"] = True
    return mismatch

_DUMP_FIELDS = ("qid", "route", "exec_expected", "exec_arm", "exec_path",
                "exec_video", "exec_memory_dir", "correct", "exec_mismatch")

_dump_path: "Path | None" = None
_dump_resolved = False

def _resolve_dump(dump) -> "Path | None":
    global _dump_path, _dump_resolved
    if isinstance(dump, (str, Path)) and str(dump) != "auto":
        return Path(dump)
    if str(os.environ.get("VEH_POOL_DUMP", "")).strip().lower() in ("off", "0", "false", "none"):
        return None
    if dump is None:
        return None
    if _dump_resolved:
        return _dump_path
    _dump_path = (STORE / "runs"
                  / f"pool_records_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}.jsonl")
    _dump_resolved = True
    return _dump_path

def _append_dump(path: Path, recs: list[dict], tag: "str | None" = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for r in recs:
            row = {k: r.get(k) for k in _DUMP_FIELDS}
            row["tag"] = tag
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

def measurable(recs: list[dict]) -> list[dict]:
    return [r for r in recs
            if r.get("gold_macro_hit") is not None or r.get("anchor_ok") is not None]

def summarize_mixed(recs: list[dict]) -> dict:
    scored = [r for r in recs if r.get("correct") is not None]
    n = len(scored) or 1
    sub = [r for r in measurable(recs) if r.get("correct") is not None]
    m = len(sub) or 1
    return {
        "n": len(recs),
        "n_scored": len(scored),
        "n_timeout": len(recs) - len(scored),
        "acc": round(sum(bool(r.get("correct")) for r in scored) / n, 4),
        "strict_acc": round(sum(bool(r.get("strict_correct")) for r in sub) / m, 4),
        "gold_macro_hit_rate": round(sum(bool(r.get("gold_macro_hit")) for r in sub) / m, 4),
        "n_measurable": len(sub),
        "by_route": route_counts(recs),
        "exec_arms": exec_arm_counts(recs),
    }

def exec_arm_counts(recs: list[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in recs:
        out[r.get("exec_arm") or "unreported"] = out.get(r.get("exec_arm") or "unreported", 0) + 1
    mism = sum(1 for r in recs if r.get("exec_mismatch"))
    if mism:
        out["mismatch"] = mism
    return out

def route_counts(recs: list[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in recs:
        k = r.get("route") or "unrouted"
        out[k] = out.get(k, 0) + 1
    return out

def _rate(recs: list[dict], field: str, pred=bool) -> tuple[float | None, int]:
    sub = [r for r in recs if r.get(field) is not None]
    if not sub:
        return None, 0
    return round(sum(bool(pred(r.get(field))) for r in sub) / len(sub), 4), len(sub)

def acc_by_route(recs: list[dict]) -> dict[str, dict]:
    buckets: dict[str, list[dict]] = {}
    for r in recs:
        buckets.setdefault(r.get("route") or "unrouted", []).append(r)
    out: dict[str, dict] = {}
    for route, sub in sorted(buckets.items(), key=lambda kv: -len(kv[1])):
        strict, strict_n = _rate(sub, "strict_correct")
        scored = [r for r in sub if r.get("correct") is not None]
        out[route] = {
            "n": len(sub),
            "n_scored": len(scored),
            "acc": (round(sum(bool(r.get("correct")) for r in scored) / len(scored), 4)
                    if scored else None),
            "strict_acc": strict,
            "n_measurable": strict_n,
        }
    return out

def trace_by_type(recs: list[dict]) -> dict[str, dict]:
    buckets: dict[str, list[dict]] = {}
    for r in recs:
        buckets.setdefault(str(r.get("type") or "unknown"), []).append(r)

    out: dict[str, dict] = {}
    for qtype, sub in sorted(buckets.items()):
        grounded, grounded_n = _rate(sub, "strict_correct")
        replay, replay_n = _rate(sub, "read_video_calls", lambda v: (v or 0) > 0)
        fallback, fallback_n = _rate(sub, "replay_fallback")
        loc = [r for r in sub if r.get("located") is not None]
        scored = [r for r in sub if r.get("correct") is not None]
        out[qtype] = {
            "n": len(sub),
            "n_scored": len(scored),
            "acc": (round(sum(bool(r.get("correct")) for r in scored) / len(scored), 4)
                    if scored else None),
            "acc_grounded": grounded,
            "n_grounded_measurable": grounded_n,
            "replay_rate": replay,
            "n_replay_measurable": replay_n,
            "retrieval_fail_rate": (round(sum(1 for r in loc if not r["located"]) / len(loc), 4)
                                    if loc else None),
            "n_retrieval_measurable": len(loc),
            "fallback_rate": fallback,
            "n_fallback_measurable": fallback_n,
            "by_route": route_counts(sub),
        }
    return out

def _assert_routes_aligned() -> None:
    known = set(video_routing.ROUTES)
    unknown = CORE_ROUTES - known
    if unknown:
        raise AssertionError(
            f"eval_router.CORE_ROUTES contains a route unknown to video_routing: "
            f"route: {sorted(unknown)}")

_assert_routes_aligned()
