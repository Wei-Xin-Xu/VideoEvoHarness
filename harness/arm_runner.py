from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

VEH_DIR = Path(__file__).resolve().parent
ROOT = VEH_DIR.parent
sys.path.insert(0, str(VEH_DIR))

import state_bundle as sb

ARMS = ("frozen", "evolving")
EVIDENCE_MODES = ("none", "retrieve", "gold", "all")
DEFAULT_MODES = ("retrieve",)

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def write_json(path: Path, value: Any) -> None:
    sb.write_json(path, value)

def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))

def read_jsonl(path: Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]

def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)

def load_kv(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    p = Path(path)
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().strip('"').strip("'")
    return out

def exp_dir(args: argparse.Namespace) -> Path:
    return Path(args.experiment_dir).expanduser().resolve()

def bundle_dir(args: argparse.Namespace) -> Path:
    return exp_dir(args) / "initial_state"

def arm_dir(args: argparse.Namespace, arm: str) -> Path:
    return exp_dir(args) / "arms" / arm

def ensure_layout(root: Path) -> None:
    for child in ("initial_state", "arms/frozen/results", "arms/frozen/logs",
                  "arms/frozen/metrics", "arms/frozen/store",
                  "arms/evolving/results", "arms/evolving/logs",
                  "arms/evolving/metrics", "arms/evolving/store",
                  "arms/evolving/candidates", "comparisons", "reports"):
        (root / child).mkdir(parents=True, exist_ok=True)

def arm_env(args: argparse.Namespace, arm: str, stage: str, variant: str,
            thinking: bool, checkpoint: int) -> dict[str, str]:
    env = os.environ.copy()
    env.update(load_kv(Path(args.config_file)))
    env.update(sb.env_for(
        bundle_dir(args),
        adaptation_mode="frozen" if arm == "frozen" else "evolve",
        arm=arm,
        checkpoint=checkpoint,
        memory_root=args.memory_root,
    ))
    adir = arm_dir(args, arm)
    env.update({
        "QWEN_MM_CONFIG": str(Path(args.config_file).expanduser().resolve()),
        "VEH_CONFIG_FILE": str(Path(args.config_file).expanduser().resolve()),
        "VEH_STORE_DIR": str(adir / "store"),
        "VEH_TOKEN_LOG": str(adir / "metrics/token_calls.jsonl"),
        "VEH_STAGE": stage,
        "VEH_VARIANT": variant,
        "VEH_THINKING": "1" if thinking else "0",
        "PYTHONUNBUFFERED": "1",
    })
    if args.data_split:
        env["VEH_DATA_SPLIT_PATH"] = str(Path(args.data_split).expanduser().resolve())
    for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        env.pop(key, None)
    env["NO_PROXY"] = env["no_proxy"] = "localhost,127.0.0.1"
    return env

def run_command(cmd: list[str], env: dict[str, str], log_path: Path, dry_run: bool = False) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    printable = " ".join(cmd)
    if dry_run:
        return
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{utc_now()}] $ {printable}\n")
        proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            log.write(line)
        code = proc.wait()
    if code:
        raise RuntimeError(f"command failed with exit code {code}: {printable}; log: {log_path}")

def thinking_variants(mode: str) -> list[tuple[str, bool]]:
    if mode == "both":
        return [("no_thinking", False), ("thinking", True)]
    return [("thinking", True)] if mode == "thinking" else [("no_thinking", False)]

def cmd_init(args: argparse.Namespace) -> int:
    root = exp_dir(args)
    ensure_layout(root)
    cfg = load_kv(Path(args.config_file))
    questions = {}
    for item in args.questions or []:
        if "=" not in item:
            raise SystemExit(f"--questions expects name=path, got: {item}")
        name, path = item.split("=", 1)
        questions[name.strip()] = Path(path.strip())
    if not questions:
        raise SystemExit(
            "--questions is required, e.g. "
            "--questions internal_test=<run>/store/questions_verified.jsonl"
        )
    manifest = sb.snapshot(
        bundle_dir(args),
        genome_path=args.genome,
        memory_root=args.memory_root,
        video_keys=args.videos,
        questions=questions,
        config_file=args.config_file,
        model=args.model or cfg.get("QWEN_MM_API_VL_MODEL"),
        embed_model=cfg.get("QWEN_MM_EMBED_MODEL") or cfg.get("EMBED_MODEL"),
        runtime={
            "thinking_mode": args.thinking_mode,
            "solve_workers": args.solve_workers,
            "evidence_modes": list(args.modes),
        },
        note=args.note,
        hash_embeddings=not args.no_embedding_hash,
        overwrite=args.overwrite,
    )
    write_json(root / "experiment.json", {
        "created_at": utc_now(),
        "experiment_dir": str(root),
        "snapshot_id": manifest["snapshot_id"],
        "arms": list(ARMS),
        "memory_root": str(Path(args.memory_root).expanduser().resolve()) if args.memory_root else None,
        "config_file": str(Path(args.config_file).expanduser().resolve()),
        "question_manifests": {k: str(v) for k, v in questions.items()},
        "thinking_mode": args.thinking_mode,
        "evidence_modes": list(args.modes),
        "note": args.note,
        "design": (
            "frozen and evolving share this bundle; frozen is re-run at every checkpoint "
            "to absorb service drift; only equal protocol fingerprints may be paired"
        ),
    })
    return 0

def resolve_questions(args: argparse.Namespace) -> Path:
    manifest = sb.load_manifest(bundle_dir(args))
    entries = manifest.get("question_manifests") or {}
    if args.split_name not in entries:
        raise SystemExit(
            f"question manifest {args.split_name!r} is not in the bundle; "
            f"available: {sorted(entries)}"
        )
    return bundle_dir(args) / entries[args.split_name]["bundle_path"]

def run_arm(args: argparse.Namespace, arm: str) -> list[Path]:
    adir = arm_dir(args, arm)
    adir.mkdir(parents=True, exist_ok=True)
    qpath = resolve_questions(args)
    bundle = bundle_dir(args)

    before = sb.verify(bundle, skip_memory=args.skip_memory_verify,
                       hash_embeddings=not args.no_embedding_hash)
    if arm == "frozen" and not before["state_unchanged"]:
        raise SystemExit(
            "the frozen bundle already drifted BEFORE this run; "
            f"mismatches: {json.dumps(before['mismatches'], ensure_ascii=False)}"
        )

    outputs: list[Path] = []
    for thinking_name, thinking in thinking_variants(args.thinking_mode):
        for evidence in args.modes:
            tag = f"{arm}__ckpt{args.checkpoint}__{thinking_name}__{evidence}__{args.split_name}"
            out = adir / f"results/{tag}.json"
            if out.exists() and not args.force:
                outputs.append(out)
                continue
            cmd = [
                sys.executable, str(VEH_DIR / "solve.py"),
                "--evidence", evidence,
                "--workers", str(args.solve_workers),
                "--questions", str(qpath),
                "--genome", str(bundle / "genome.json"),
                "--out", str(out),
            ]
            if args.split:
                cmd.extend(["--split", args.split])
            if args.limit:
                cmd.extend(["--limit", str(args.limit)])
            run_command(cmd, arm_env(args, arm, "solve", tag, thinking, args.checkpoint),
                        adir / f"logs/{tag}.log", args.dry_run)
            outputs.append(out)

    after = sb.verify(bundle, skip_memory=args.skip_memory_verify,
                      hash_embeddings=not args.no_embedding_hash)
    receipt = {
        "arm": arm,
        "checkpoint": args.checkpoint,
        "finished_at": utc_now(),
        "question_manifest": str(qpath),
        "split": args.split,
        "evidence_modes": list(args.modes),
        "thinking_mode": args.thinking_mode,
        "results": [str(p) for p in outputs],
        "state_hash_before": before["state_hash"],
        "state_hash_after": after["state_hash"],
        "state_unchanged": before["state_hash"] == after["state_hash"] and after["state_unchanged"],
        "mismatches_after": after["mismatches"],
    }
    write_json(adir / f"metrics/receipt_ckpt{args.checkpoint}.json", receipt)
    if arm == "frozen" and not receipt["state_unchanged"] and not args.dry_run:
        raise SystemExit(
            "INVALID FROZEN RUN: pinned state changed during the run; "
            "the results must not be used as a baseline"
        )
    return outputs

def cmd_frozen(args: argparse.Namespace) -> int:
    run_arm(args, "frozen")
    return 0

def cmd_evolving(args: argparse.Namespace) -> int:
    run_arm(args, "evolving")
    return 0

def stratified_bootstrap(
    pairs: list[tuple[str, int]], samples: int = 5000, seed: int = 42
) -> dict:
    if not pairs:
        return {"n": 0, "mean_delta": None, "ci95": [None, None], "lower_bound": None}
    strata: dict[str, list[int]] = defaultdict(list)
    for key, delta in pairs:
        strata[key].append(delta)
    total = sum(len(v) for v in strata.values())
    rng = random.Random(seed)
    means: list[float] = []
    keys = list(strata)
    for _ in range(samples):
        acc = 0.0
        for key in keys:
            values = strata[key]
            n = len(values)
            acc += sum(values[rng.randrange(n)] for _ in range(n))
        means.append(acc / total)
    means.sort()
    lo = means[int(samples * 0.025)]
    hi = means[min(samples - 1, int(samples * 0.975))]
    return {
        "n": total,
        "strata": len(strata),
        "mean_delta": sum(d for _, d in pairs) / total,
        "ci95": [lo, hi],
        "lower_bound": lo,
        "samples": samples,
        "seed": seed,
    }

def _cost_of(payload: dict) -> dict:
    total = ((payload.get("cost") or {}).get("total") or {})
    records = payload.get("records") or []
    n = len(records) or 1
    return {
        "llm_calls": total.get("calls", 0),
        "prompt_tokens": total.get("prompt_tokens", 0),
        "completion_tokens": total.get("completion_tokens", 0),
        "reasoning_tokens": total.get("reasoning_tokens", 0),
        "image_tokens": total.get("image_tokens", 0),
        "total_tokens": total.get("total_tokens", 0),
        "tokens_per_question": round(total.get("total_tokens", 0) / n, 2),
        "llm_wall_sec": total.get("latency_sec", 0),
        "latency_mean_s": round(sum(r.get("latency_s", 0) for r in records) / n, 3),
        "latency_p95_s": (sorted(r.get("latency_s", 0) for r in records)[int(0.95 * (n - 1))]
                          if records else None),
        "read_video_calls": sum(r.get("read_video_calls", 0) for r in records),
        "read_video_frames": sum(r.get("read_video_frames", 0) for r in records),
        "frame_tokens_est": sum(r.get("frame_tokens_est", 0) for r in records),
        "api_errors": sum(bool(r.get("error")) for r in records),
        "unparsed": sum(r.get("pred") is None for r in records),
    }

def compare_pair(baseline: Path, candidate: Path, *, require_fingerprint: bool = True,
                 tolerance: float = 0.0) -> dict:
    b, c = read_json(baseline), read_json(candidate)
    bfp, cfp = b.get("protocol_fingerprint"), c.get("protocol_fingerprint")
    fingerprint_equal = bool(bfp) and bfp == cfp
    diff_keys: list[str] = []
    if not fingerprint_equal:
        bp, cp = b.get("fingerprint_parts") or {}, c.get("fingerprint_parts") or {}
        expected = {"genome_version", "genome_sha256", "retrieval", "evidence", "answer",
                    "read_video", "budget", "prompt_tree_sha256", "tool_registry_sha256",
                    "strategy_tree_sha256", "memory_manifest_sha256"}
        diff_keys = sorted(k for k in set(bp) | set(cp) if bp.get(k) != cp.get(k))
        blocking = sorted(set(diff_keys) - expected)
        if blocking and require_fingerprint:
            raise SystemExit(
                "refusing to compare: these runs differ in fields that define the "
                f"measurement rather than the treatment: {blocking}"
            )

    bm = {r["qid"]: r for r in b.get("records") or []}
    cm = {r["qid"]: r for r in c.get("records") or []}
    common = sorted(set(bm) & set(cm))
    rows: list[dict] = []
    pairs: list[tuple[str, int]] = []
    strict_pairs: list[tuple[str, int]] = []
    for qid in common:
        br, cr = bm[qid], cm[qid]
        delta = int(bool(cr.get("correct"))) - int(bool(br.get("correct")))
        sdelta = int(bool(cr.get("strict_correct"))) - int(bool(br.get("strict_correct")))
        stratum = f"{br.get('video_key')}::{br.get('type')}"
        pairs.append((stratum, delta))
        strict_pairs.append((stratum, sdelta))
        rows.append({
            "qid": qid, "video_key": br.get("video_key"), "type": br.get("type"),
            "stratum": stratum,
            "baseline_correct": bool(br.get("correct")),
            "candidate_correct": bool(cr.get("correct")),
            "delta": delta, "strict_delta": sdelta,
            "outcome": "win" if delta > 0 else "loss" if delta < 0 else "tie",
        })

    bs, cs = b.get("summary") or {}, c.get("summary") or {}
    boot = stratified_bootstrap(pairs)
    by_type: dict[str, dict] = {}
    for row in rows:
        d = by_type.setdefault(row["type"], {"n": 0, "wins": 0, "losses": 0})
        d["n"] += 1
        d["wins"] += row["delta"] > 0
        d["losses"] += row["delta"] < 0
    bcost, ccost = _cost_of(b), _cost_of(c)
    return {
        "compared_at": utc_now(),
        "baseline_file": str(baseline),
        "candidate_file": str(candidate),
        "baseline_arm": b.get("arm"),
        "candidate_arm": c.get("arm"),
        "baseline_checkpoint": b.get("checkpoint"),
        "candidate_checkpoint": c.get("checkpoint"),
        "fingerprint_equal": fingerprint_equal,
        "fingerprint_diff_fields": diff_keys,
        "paired_n": len(common),
        "baseline_only": sorted(set(bm) - set(cm))[:50],
        "candidate_only": sorted(set(cm) - set(bm))[:50],
        "coverage_equal": len(bm) == len(cm) == len(common),
        "baseline_acc": bs.get("acc"),
        "candidate_acc": cs.get("acc"),
        "accuracy_delta": (cs.get("acc") or 0) - (bs.get("acc") or 0),
        "strict_delta": (cs.get("strict_acc") or 0) - (bs.get("strict_acc") or 0),
        "gold_macro_hit_delta": (cs.get("gold_macro_hit_rate") or 0) - (bs.get("gold_macro_hit_rate") or 0),
        "wins": sum(r["delta"] > 0 for r in rows),
        "losses": sum(r["delta"] < 0 for r in rows),
        "ties": sum(r["delta"] == 0 for r in rows),
        "bootstrap": boot,
        "bootstrap_strict": stratified_bootstrap(strict_pairs),
        "by_type": by_type,
        "baseline_cost": bcost,
        "candidate_cost": ccost,
        "cost_ratio_tokens": (round(ccost["total_tokens"] / bcost["total_tokens"], 3)
                              if bcost["total_tokens"] else None),
        "verdict": (
            "improved" if (boot["lower_bound"] or 0) > tolerance and len(bm) == len(cm)
            else "regressed" if (boot["ci95"][1] or 0) < -tolerance
            else "inconclusive"
        ),
        "rows": rows,
    }

def cmd_compare(args: argparse.Namespace) -> int:
    root = exp_dir(args)
    results: list[dict] = []
    for thinking_name, _ in thinking_variants(args.thinking_mode):
        for evidence in args.modes:
            suffix = f"{thinking_name}__{evidence}__{args.split_name}"
            base = arm_dir(args, "frozen") / f"results/frozen__ckpt{args.baseline_checkpoint}__{suffix}.json"
            cand = arm_dir(args, "evolving") / f"results/evolving__ckpt{args.checkpoint}__{suffix}.json"
            if not base.exists() or not cand.exists():
                continue
            out = compare_pair(base, cand, require_fingerprint=not args.allow_protocol_diff,
                               tolerance=args.tolerance)
            tag = f"ckpt{args.checkpoint}__{suffix}"
            write_jsonl(root / f"comparisons/paired_{tag}.jsonl", out.pop("rows"))
            write_json(root / f"comparisons/comparison_{tag}.json", out)
            results.append({"tag": tag, **{k: out[k] for k in (
                "paired_n", "coverage_equal", "baseline_acc", "candidate_acc",
                "accuracy_delta", "wins", "losses", "ties", "verdict")}})
    write_json(root / f"comparisons/summary_ckpt{args.checkpoint}.json",
               {"created_at": utc_now(), "comparisons": results})
    return 0

def cmd_verify(args: argparse.Namespace) -> int:
    report = sb.verify(bundle_dir(args), skip_memory=args.skip_memory_verify,
                       hash_embeddings=not args.no_embedding_hash)
    write_json(exp_dir(args) / "reports/verify_latest.json", report)
    return 0 if report["state_unchanged"] else 3

def cmd_report(args: argparse.Namespace) -> int:
    root = exp_dir(args)
    rows: list[dict] = []
    for arm in ARMS:
        for path in sorted((arm_dir(args, arm) / "results").glob("*.json")):
            try:
                payload = read_json(path)
            except Exception:
                continue
            summary = payload.get("summary") or {}
            cost = _cost_of(payload)
            rows.append({
                "arm": payload.get("arm") or arm,
                "adaptation_mode": payload.get("adaptation_mode"),
                "checkpoint": payload.get("checkpoint"),
                "genome_version": payload.get("genome_version"),
                "evidence": payload.get("mode"),
                "split": payload.get("split"),
                "protocol_fingerprint": (payload.get("protocol_fingerprint") or "")[:16],
                "n": summary.get("n"),
                "acc": summary.get("acc"),
                "strict_acc": summary.get("strict_acc"),
                "anchor_rate": summary.get("anchor_rate"),
                "gold_macro_hit_rate": summary.get("gold_macro_hit_rate"),
                "replay_required": (summary.get("replay") or {}).get("required"),
                "replay_calls": (summary.get("replay") or {}).get("calls"),
                "dense_timeouts": (summary.get("retrieval") or {}).get("dense_timeouts"),
                "bm25_fallbacks": (summary.get("retrieval") or {}).get("bm25_fallbacks"),
                **{f"cost_{k}": v for k, v in cost.items()},
                "file": path.name,
            })
    rows.sort(key=lambda r: (r["checkpoint"] or 0, r["arm"], str(r["evidence"])))
    write_jsonl(root / "reports/all_runs.jsonl", rows)
    if rows:
        fields = list(rows[0])
        with (root / "reports/all_runs.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    comparisons = {p.stem: read_json(p) for p in sorted((root / "comparisons").glob("comparison_*.json"))}
    write_json(root / "reports/report.json", {
        "created_at": utc_now(),
        "experiment": read_json(root / "experiment.json") if (root / "experiment.json").exists() else None,
        "bundle": sb.load_manifest(bundle_dir(args)) if (bundle_dir(args) / sb.MANIFEST_NAME).exists() else None,
        "runs": rows,
        "comparisons": {k: {kk: vv for kk, vv in v.items() if kk != "rows"} for k, v in comparisons.items()},
    })
    for r in rows:
        pass
    return 0

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Frozen vs evolving two-arm driver")
    ap.add_argument("command", choices=["init", "frozen", "evolving", "compare", "verify", "report"])
    ap.add_argument("--experiment-dir", required=True)
    ap.add_argument("--config-file", default=str(ROOT / "qwen-mm-config"))
    ap.add_argument("--memory-root", default=str(ROOT / "experiment_memory"))
    ap.add_argument("--genome", default=None, help="init: genome to freeze (default harness/genome.json)")
    ap.add_argument("--videos", nargs="*", default=None, help="init: pin only these video keys")
    ap.add_argument("--questions", action="append", default=[], help="init: name=path, repeatable")
    ap.add_argument("--data-split", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--note", default="")
    ap.add_argument("--split-name", default="internal_test", help="which pinned manifest to solve")
    ap.add_argument("--split", default=None, help="optional split filter inside the manifest")
    ap.add_argument("--modes", nargs="*", default=list(DEFAULT_MODES), choices=list(EVIDENCE_MODES))
    ap.add_argument("--thinking-mode", choices=["no_thinking", "thinking", "both"], default="no_thinking")
    ap.add_argument("--solve-workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--checkpoint", type=int, default=0)
    ap.add_argument("--baseline-checkpoint", type=int, default=None,
                    help="compare: frozen checkpoint to pair against (default = --checkpoint)")
    ap.add_argument("--tolerance", type=float, default=0.0)
    ap.add_argument("--allow-protocol-diff", action="store_true",
                    help="compare: permit non-treatment protocol differences (documents a caveat)")
    ap.add_argument("--skip-memory-verify", action="store_true",
                    help="skip re-hashing embeddings.npz; faster, weaker guarantee")
    ap.add_argument("--no-embedding-hash", action="store_true")
    ap.add_argument("--force", action="store_true", help="rerun even if the result file exists")
    ap.add_argument("--overwrite", action="store_true", help="init: replace an existing bundle")
    ap.add_argument("--dry-run", action="store_true")
    return ap

def main() -> int:
    args = build_parser().parse_args()
    if args.baseline_checkpoint is None:
        args.baseline_checkpoint = args.checkpoint
    args.modes = tuple(args.modes or DEFAULT_MODES)
    root = exp_dir(args)
    if args.command != "init" and not (bundle_dir(args) / sb.MANIFEST_NAME).exists():
        raise SystemExit(f"no state bundle under {bundle_dir(args)}; run `init` first")
    ensure_layout(root)
    return {
        "init": cmd_init,
        "frozen": cmd_frozen,
        "evolving": cmd_evolving,
        "compare": cmd_compare,
        "verify": cmd_verify,
        "report": cmd_report,
    }[args.command](args)

if __name__ == "__main__":
    raise SystemExit(main())
