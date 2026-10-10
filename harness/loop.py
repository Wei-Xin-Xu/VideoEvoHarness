from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import evolve as evolve_mod
import eval_router
import shard as shard_mod
from common import (
    CAPABILITY_KINDS,
    CAPABILITY_MODES,
    GATE_NAMES,
    MEMORY_ROOT,
    VEH_DIR,
    ROOT,
    STORE,
    gates_relaxed,
    load_cfg,
    load_genome,
    load_questions,
    parse_capability_overrides,
    parse_gate_overrides,
    run_streaming,
    save_genome,
    version_axes,
)

_JUDGE_OFF_VALUES = {"off", "none", "skip", "0"}

SESSION_TS = time.strftime("%Y%m%d_%H%M%S")

def _cache_bucket_names() -> set:
    d = STORE / "runs" / "cache"
    if not d.is_dir():
        return set()
    return {p.name for p in d.glob("*.jsonl")}

def _load_trace() -> list:
    p = STORE / "runs" / "loop_trace.json"
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, ValueError) as exc:
        bak = p.with_name(f"loop_trace.corrupt_{SESSION_TS}.json")
        try:
            os.replace(p, bak)
        except OSError:
            pass
        return []
    if not isinstance(data, list):
        return []
    return data

def _flush_trace(trace: list) -> None:
    p = STORE / "runs" / "loop_trace.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    blob = json.dumps(trace, ensure_ascii=False, indent=2)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(blob, encoding="utf-8")
    os.replace(tmp, p)

    arch = STORE / "runs" / "traces" / f"loop_trace_{SESSION_TS}.json"
    arch.parent.mkdir(parents=True, exist_ok=True)
    atmp = arch.with_suffix(".json.tmp")
    atmp.write_text(blob, encoding="utf-8")
    os.replace(atmp, arch)

def judge_disabled() -> bool:
    return (os.environ.get("VEH_EXTERNAL_JUDGE") or "auto").strip().lower() in _JUDGE_OFF_VALUES

def run_external(g: dict) -> dict | None:
    which = (os.environ.get("VEH_EXTERNAL_JUDGE") or "auto").strip().lower()
    if which in _JUDGE_OFF_VALUES:
        return None
    multi = ROOT / "test_external_lvbench.py"
    single = ROOT / "test_memory.py"

    genome_path = STORE / "runs" / f"external_genome_v{g['version']}.json"
    genome_path.parent.mkdir(parents=True, exist_ok=True)
    genome_path.write_text(json.dumps(g, ensure_ascii=False, indent=2), encoding="utf-8")

    use_multi = which == "lvbench" or (which == "auto" and multi.exists() and _has_lvbench_memories())
    if use_multi:
        script = multi
        limit = os.environ.get("VEH_EXTERNAL_LIMIT", "120")
        ext_workers = os.environ.get("VEH_EXTERNAL_WORKERS", "8")
        cmd = [sys.executable, str(script), "--genome", str(genome_path),
               "--limit-questions", str(limit), "--workers", str(ext_workers)]
        label = f"LV-Bench ({limit} questions sampled from 886, seed-fixed, {ext_workers} workers)"
    elif single.exists():
        script = single
        cmd = [sys.executable, str(script), "--video", "0RxMZBLeqRI", "--genome", str(genome_path)]
        label = "Video-MME, 1 video / 3 questions -- TOO SMALL to detect regressions"
    else:
        return None

    env = {
        **os.environ,
        "QWEN_MM_CONFIG": str(ROOT / "qwen-mm-config"),
        "VEH_CONFIG_FILE": str(ROOT / "qwen-mm-config"),
    }
    if use_multi:
        env["VEH_EXTERNAL_MEMORY_ROOT"] = os.environ.get(
            "VEH_EXTERNAL_MEMORY_ROOT", str(ROOT / "experiment_memory"))
    else:
        env["VEH_MEMORY_ROOT"] = os.environ.get("VEH_EXTERNAL_MEMORY_ROOT", str(ROOT / "memory"))
    code, tail, err_tail = run_streaming(
        cmd, cwd=str(ROOT), env=env, timeout=7200, keep_tail=60,
    )
    if code != 0:
        return {"genome_version": g["version"], "videomme_acc": None,
                "error": f"exit {code}", "tail": tail[-12:]}
    acc = None
    for line in reversed(tail):
        if "accuracy" in line.lower() and "=" in line:
            try:
                acc = float(line.split("=")[-1].strip().rstrip("%")) / 100
            except ValueError:
                pass
            break
    return {"genome_version": g["version"], "videomme_acc": acc,
            "judge": "lvbench" if use_multi else "videomme", "tail": tail[-12:]}

def _has_lvbench_memories(minimum: int = 5) -> bool:
    root = Path(os.environ.get("VEH_EXTERNAL_MEMORY_ROOT", ROOT / "experiment_memory"))
    if not root.is_dir():
        return False
    n = 0
    for d in root.iterdir():
        if d.is_dir() and d.name.startswith("lvbench__") and (d / "graph_memory.json").exists():
            n += 1
            if n >= minimum:
                return True
    return False

_T0 = time.time()

def stamp(msg: str = "") -> str:
    el = time.time() - _T0
    return f"[{time.strftime('%H:%M:%S')} +{int(el) // 60:02d}:{int(el) % 60:02d}] {msg}"

def sh(cmd: list[str]) -> int:
    t0 = time.time()
    rc = subprocess.run(cmd, cwd=str(VEH_DIR)).returncode
    return rc

def _capture_version_pointers() -> dict[str, str | None]:
    paths = [
        VEH_DIR / "tools_learned" / "CURRENT",
        VEH_DIR / "strategies" / "CURRENT",
        VEH_DIR / "prompts" / "CURRENT",
        *((p / "CURRENT" for p in MEMORY_ROOT.iterdir() if p.is_dir()) if MEMORY_ROOT.exists() else ()),
    ]
    return {
        str(path): path.read_text(encoding="utf-8") if path.exists() else None
        for path in paths
    }

def _build_ask_cmd(py: str, args, seed: int | None = None) -> list[str]:
    cmd = [py, "ask.py"]
    if args.n_questions:
        cmd += ["--n", str(args.n_questions)]
    if getattr(args, "smoke_videos", None):
        cmd += ["--videos", *args.smoke_videos]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    return cmd

def _restore_version_pointers(snapshot: dict[str, str | None]) -> None:
    for raw_path, value in snapshot.items():
        path = Path(raw_path)
        if value is None:
            path.unlink(missing_ok=True)
            continue
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(value, encoding="utf-8")
        tmp.replace(path)

def _archive_version_pointers(tag: str, snapshot: dict[str, str | None]) -> Path:
    p = STORE / "runs" / "pointers" / f"{tag}_{SESSION_TS}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    return p

def _freeze_control_genome(g: dict) -> Path:
    p = STORE / "genomes" / f"control_frozen_v{g['version']}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(g, ensure_ascii=False, indent=2), encoding="utf-8")
    os.chmod(p, 0o444)
    return p

def _measure_heldout(g: dict, cfg: dict, qpath: Path, workers: int, label: str,
                     heldout_limit: int = 0) -> dict | None:
    qs = load_questions(qpath, "heldout")
    if heldout_limit:
        qs = evolve_mod.subsample_heldout(qs, heldout_limit)
    if not qs:
        return None
    recs = eval_router.run_pool(qs, g, cfg, workers, quiet=True,
                                cache_tag=f"curve:{label}")
    s = eval_router.summarize_mixed(recs)
    return s

def _curve_point(control_g: dict, g: dict, cfg: dict, qpath: Path, workers: int,
                 store_version: int, questioner_version: int,
                 heldout_limit: int = 0) -> dict:
    a = _measure_heldout(control_g, cfg, qpath, workers, "A frozen", heldout_limit)
    b = _measure_heldout(g, cfg, qpath, workers, "B evolved", heldout_limit)
    acc_a = (a or {}).get("acc")
    acc_b = (b or {}).get("acc")
    point = {
        "store_version": store_version,
        "questioner_version": questioner_version,
        "control_version": control_g["version"],
        "solver_version": version_axes(g)["solver_version"],
        "genome_version": g["version"],
        "acc_A_frozen_solver": acc_a,
        "acc_B_evolved_solver": acc_b,
        "n": (b or {}).get("n"),
        "summary_A": a,
        "summary_B": b,
    }
    if acc_a is None or acc_b is None:
        point["delta"] = None
        point["delta_interpretable"] = False
        point["why"] = "one side not measured"
        return point
    point["delta"] = round(acc_b - acc_a, 4)
    point["delta_interpretable"] = bool(0.2 <= acc_a <= 0.8)
    if not point["delta_interpretable"]:
        point["why"] = (f"acc_A={acc_a:.3f} lies outside 0.2-0.8 (floor or ceiling), "
                        "delta is compressed and therefore not interpretable")
    return point

def _append_curve(point: dict) -> Path:
    p = STORE / "runs" / "curves.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(point, ensure_ascii=False) + "\n")
    return p

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--start-round", type=int, default=0,
                    help="Resume from round N (0-based, default 0 = from the start).\n"
                         "The shard index is round %% n_shards, so restarting mid-run without this flag"
                         "would replay shard 0 forever. To continue with shard 3 after finishing two shards:"
                         "--start-round 2 --rounds 2.\n"
                         "Note that it only affects the round number and does not restore the genome -- the genome"
                         "is always read from its latest on-disk state, and registered strategies remain in place.")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--n-questions", type=int, default=0, help="max questions per round, 0 = unlimited")
    ap.add_argument("--shards", type=int, default=0,
                    help="Split the evolve pool into N shards by video; round i diagnoses only shard i%%N"
                         "(0 = no sharding, the default).\n"
                         "**Only evolve is sharded; heldout always stays complete**: the absolute number of flips"
                         "required by the gates does not shrink with the sample,\nso sharding heldout would close the gate permanently.\n"
                         "The plan is written to STORE/shards.json so it can be reproduced and audited.")
    ap.add_argument("--heldout-limit", type=int, default=0,
                    help="Shrink the acceptance pool to N items (0 = complete, the default). Stratified by route\n"
                         "and selected deterministically by qid, so the membership is identical every round.\n"
                         "**Cost**: the absolute number of flips the gates demand does not shrink proportionally,\n"
                         "so a smaller pool raises the bar. It is normally combined with\n"
                         "gates.paired_bootstrap.mode=acc_only, otherwise candidates become much harder to accept.")
    ap.add_argument("--external-every", type=int, default=1, help="run the external judge every N rounds")
    ap.add_argument("--regen-every", type=int, default=0, help="regenerate questions every N rounds, 0 = generate once")
    ap.add_argument("--external-only", action="store_true")
    ap.add_argument("--skip-questioner", action="store_true",
                    help="Do not evolve the questioner (the pool stays fixed across rounds). Note that the solver\n                         will keep overfitting to the same questions; that is the price of turning this off")
    ap.add_argument("--questioner-propose-only", action="store_true",
                    help="Evaluate questioner candidates as usual but never apply them: useful to observe what it\n                         would propose without changing the pool")
    ap.add_argument("--max-candidates", type=int, default=0,
                    help="wiring/smoke knob: cap the candidate patches evaluated per "
                         "evolution round (0 = no cap, default)")
    ap.add_argument("--smoke-videos", nargs="*", default=None,
                    help="Restrict question generation to these video_keys (for smoke tests). Omitting it means the full set,\n"
                         "which makes post-evolution regeneration scan the entire memory store")
    ap.add_argument("--questioner-n", type=int, default=0,
                    help="How many questions to regenerate per side during paired questioner evaluation (0 = automatic,\n"
                         "roughly the current count). Every questioner candidate reruns ask.py plus verify.py, the most expensive category")
    ap.add_argument("--curves", action="store_true",
                    help="Each round, additionally run a 'frozen solver' over the new questions, producing two\n"
                         "subtractable curves (line A frozen / line B evolved, delta=B-A) in store/runs/curves.jsonl.\n"
                         "Cost: one extra full held-out evaluation per round")
    ap.add_argument("--control-genome", default=None,
                    help="Which genome line A freezes at (default: genome.json as of this loop's start)."
                         "Passing a snapshot such as store/genomes/genome_v3.json selects a later freeze point")

    ap.add_argument("--llm-propose", action="store_const", const=True, default=None,
                    help="[legacy] now the default (capabilities default to llm). Omitting it no longer disables anything")
    ap.add_argument("--no-llm-propose", dest="llm_propose",
                    action="store_const", const=False,
                    help="Switch hyperparam/skill/tool_policy to non-LLM")
    ap.add_argument("--evolver-model", default=None,
                    help="ablation only: use a different model for proposals")
    ap.add_argument("--skip-build", action="store_true", help="skip generation+filtering and reuse the existing store")
    ap.add_argument("--strategy-forge", action="store_const", const=True, default=None,
                    help="[legacy] equivalent to --capability strategy_code=llm")
    ap.add_argument("--no-strategy-forge", dest="strategy_forge",
                    action="store_const", const=False,
                    help="equivalent to --capability strategy_code=off")
    ap.add_argument("--strategy-attempts", type=int, default=3)
    ap.add_argument("--strategy-min-per-type", type=int, default=None)
    ap.add_argument("--tool-attempts", type=int, default=3)
    ap.add_argument("--rule-candidates", action="store_const", const=True, default=None,
                    help="[legacy] switch hyperparam/skill/tool_policy to rule-based proposers")
    ap.add_argument("--capability", action="append", metavar="KIND=MODE",
                    help=f"Set the proposal permission per category; repeatable. KIND in {CAPABILITY_KINDS}. "
                         f"MODE in {CAPABILITY_MODES}")
    ap.add_argument("--gate", action="append", metavar="NAME[.KEY]=VALUE",
                    help=f"Set the criterion per gate; repeatable. NAME=full|record_only|bypass. Available: {GATE_NAMES}")
    ap.add_argument("--allow-rule-fallback", action="store_true",
                    help="Fall back to rule candidates when the model proposes none (off by default)")
    ap.add_argument("--per-kind-candidates", type=int, default=0,
                    help="Maximum candidates proposed per category (0 = unlimited). Unlike --max-candidates:\n"
                         "that one truncates the assembled list, and code candidates are appended last,\n"
                         "so lowering it removes the code category entirely; per-kind lets every category survive")
    ap.add_argument("--code-every", type=int, default=1,
                    help="Generate code-level candidates (strategies/tools) every N rounds")
    ap.add_argument("--core-frames", action="store_true",
                    help="allow evolution to search the direct-reading arm "
                         "(evidence.frames.*)")
    args = ap.parse_args()
    cap_overrides = parse_capability_overrides(args.capability)
    gate_overrides = parse_gate_overrides(args.gate)

    g, cfg = load_genome(), load_cfg()
    py = sys.executable
    qpath = STORE / "questions_verified.jsonl"
    trace = _load_trace()

    if args.external_only:
        return

    if not args.skip_build or not qpath.exists():
        if sh(_build_ask_cmd(py, args)) != 0:
            raise SystemExit("ask.py failed")
        if sh([py, "verify.py", "--workers", str(args.workers)]) != 0:
            raise SystemExit("verify.py failed")

    ext0 = run_external(g)
    trace.append({"stage": "init", "external": ext0})

    if gate_overrides:
        merged = dict(g.get("gates") or {})
        for name, patch in gate_overrides.items():
            merged[name] = {**(merged.get(name) or {}), **patch}
        g["gates"] = merged
        save_genome(g)
    relaxed_gates = gates_relaxed(g)
    if relaxed_gates:
        pass

    store_version = 0
    questioner_version = version_axes(g)["questioner_version"]
    questioner_changed = False

    control_g = None
    if args.curves:
        control_g = load_genome(args.control_genome) if args.control_genome else json.loads(
            json.dumps(g, ensure_ascii=False))
        frozen_at = _freeze_control_genome(control_g)

    shard_plan = None
    if args.shards and args.shards > 1:
        _qs_all = load_questions(qpath)
        shard_plan = shard_mod.build(_qs_all, args.shards)
        shard_mod.save(shard_plan, STORE / "shards.json")
    elif args.shards == 1:
        pass

    start = max(0, int(args.start_round))
    if start:
        pass
    for done, i in enumerate(range(start, start + args.rounds)):

        forced = bool(args.regen_every and done > 0 and i % args.regen_every == 0)
        if questioner_changed or forced:
            why = "questioner evolved" if questioner_changed else f"--regen-every {args.regen_every}"
            if sh(_build_ask_cmd(py, args, seed=i)) != 0:
                raise SystemExit("ask.py failed")
            if sh([py, "verify.py", "--workers", str(args.workers)]) != 0:
                raise SystemExit("verify.py failed")
            store_version += 1
            questioner_changed = False
        entry_store = {"store_version": store_version,
                       "generated_by_questioner_v": questioner_version}

        pointer_snapshot = _capture_version_pointers()
        _archive_version_pointers(f"round{i + 1}_before_v{g['version']}", pointer_snapshot)
        shard_qs = None
        shard_videos = None
        if shard_plan is not None:
            _all = load_questions(qpath)
            shard_qs = shard_mod.select(_all, shard_plan, i)
            shard_videos = shard_mod.videos_of(shard_plan, i)
            si = shard_mod.shard_index(shard_plan, i)

        _round_t0 = time.time()
        _buckets_before = _cache_bucket_names()

        g, rep = evolve_mod.evolve_round(
            g, cfg, args.workers, qpath,
            use_llm=args.llm_propose,
            evolver_model=args.evolver_model,
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
            shard_questions=shard_qs,
            heldout_limit=args.heldout_limit,
        )
        entry = {"stage": f"round{i + 1}", "evolve": rep, **entry_store,
                 "version_axes": version_axes(g)}
        entry["session"] = SESSION_TS
        entry["started_at"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(_round_t0))
        entry["ended_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        entry["cache_buckets_new"] = sorted(_cache_bucket_names() - _buckets_before)
        entry["cache_buckets_all"] = sorted(_cache_bucket_names())
        if shard_plan is not None:
            entry["shard"] = si
            entry["n_shards"] = len(shard_plan.get("shards") or [])
            entry["shard_videos"] = list(shard_videos or [])
            entry["shard_n_questions"] = len(shard_qs or [])
        solver_accepted = rep["accepted"] is not None
        provisional = bool((rep["accepted"] or {}).get("provisional") or g.get("provisional"))
        if provisional:
            entry["provisional"] = True
            entry["provisional_reason"] = g.get("provisional_reason")

        unverified = False
        if rep["accepted"] and not provisional and (i + 1) % args.external_every == 0:
            entry["external"] = run_external(g)
            if entry["external"] is None:
                if judge_disabled():
                    entry["external_disabled"] = True
                else:
                    entry["unverified"] = True
                    unverified = True
            else:
                prev = next((t["external"]["videomme_acc"] for t in reversed(trace)
                             if t.get("external") and t["external"].get("videomme_acc") is not None), None)
                cur = entry["external"].get("videomme_acc")
                if cur is None:
                    rollback_path = STORE / "genomes" / f"genome_v{rep['from_version']}.json"
                    if rollback_path.exists():
                        rejected_version = g["version"]
                        g = load_genome(rollback_path)
                        save_genome(g)
                        _restore_version_pointers(pointer_snapshot)
                        _archive_version_pointers(
                            f"round{i + 1}_rollback_to_v{g['version']}",
                            _capture_version_pointers())
                        entry["rollback"] = {
                            "reason": "external_judge_unavailable",
                            "rejected_version": rejected_version,
                            "restored_version": g["version"],
                        }
                        rep["accepted"] = None
                    else:
                        raise RuntimeError(
                            f"external judge failed and rollback snapshot is missing: {rollback_path}"
                        )
                elif prev is not None and cur < prev - 1e-9:
                    rollback_path = STORE / "genomes" / f"genome_v{rep['from_version']}.json"
                    if not rollback_path.exists():
                        raise RuntimeError(f"rollback snapshot missing: {rollback_path}")
                    rejected_version = g["version"]
                    g = load_genome(rollback_path)
                    save_genome(g)
                    _restore_version_pointers(pointer_snapshot)
                    _archive_version_pointers(
                        f"round{i + 1}_rollback_to_v{g['version']}",
                        _capture_version_pointers())
                    entry["rollback"] = {
                        "reason": "external_regression",
                        "rejected_version": rejected_version,
                        "restored_version": g["version"],
                        "previous_external": prev,
                        "candidate_external": cur,
                    }
                    rep["accepted"] = None

        q_rep = None
        rolled_back = "rollback" in entry
        if args.skip_questioner:
            pass
        elif rolled_back:
            pass
        elif unverified:
            pass
        elif not solver_accepted:
            pass
        else:
            g, q_rep = evolve_mod.evolve_questioner_round(
                g, cfg, args.workers,
                solver_by_type=rep.get("solver_by_type") or {},
                solver_trace_by_type=rep.get("solver_trace_by_type") or {},
                qreport=rep.get("qreport"),
                apply_patch=not args.questioner_propose_only,
                per_kind_cap=args.per_kind_candidates,
                n_questions=args.questioner_n or None,
                videos=shard_videos,
            )
            entry["questioner"] = q_rep
            if (q_rep.get("accepted") or {}).get("applied"):
                questioner_changed = True
                questioner_version = version_axes(g)["questioner_version"]
                _archive_version_pointers(
                    f"round{i + 1}_after_questioner_v{g['version']}",
                    _capture_version_pointers())

        entry["version_axes"] = version_axes(g)
        if args.curves and control_g is not None and not provisional:
            point = _curve_point(control_g, g, cfg, qpath, args.workers,
                                 store_version, questioner_version,
                                 heldout_limit=args.heldout_limit)
            point["round"] = i + 1
            entry["curve"] = point
            _append_curve(point)
        elif args.curves and provisional:
            entry["curve_skipped"] = "provisional: gates were not fully enabled, so this point is not interpretable"

        trace.append(entry)
        _flush_trace(trace)

        try:
            _rc = subprocess.run(
                [sys.executable, str(VEH_DIR.parent / "export_shard_report.py"),
                 "--store", str(STORE)],
                capture_output=True, text=True, timeout=300)
            if _rc.returncode == 0:
                pass
            else:
                pass
        except Exception as _exc:
            pass

        q_accepted = bool((q_rep or {}).get("accepted"))
        if not solver_accepted and not q_accepted:
            if shard_plan is not None:
                pass
            else:
                break

    p = STORE / "runs" / "loop_trace.json"
    _flush_trace(trace)

if __name__ == "__main__":
    main()
