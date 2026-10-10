from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

VEH_DIR = Path(__file__).resolve().parent
ROOT = VEH_DIR.parent
sys.path.insert(0, str(VEH_DIR))
sys.path.insert(0, str(ROOT))

import video_routing

import state_bundle as sb

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
ALL_BENCHMARKS = (
    "videomme", "videomme_v2", "cgbench", "longvideobench", "lvbench",
    "mlvu", "mmvu", "video_holmes", "videommmu", "longvideoreason",
    "videoreasonbench",
)
VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".avi", ".mov"}
VIDEO_DATASET = {
    "videomme": "videomme",
    "videomme_v2": "videomme_v2",
    "cgbench": "cgbench",
    "longvideobench": "longvideobench",
    "lvbench": "lvbench",
    "mlvu": "mlvu_test",
    "mmvu": "mmvu",
    "video_holmes": "video_holmes",
    "videommmu": "videommmu",
    "longvideoreason": "longvideoreason_videos",
    "videoreasonbench": "videoreasonbench",
}
VIDEO_EXCLUDE_PARTS = {"assets", "__MACOSX", "subtitle", "_logs"}

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def write_json(path: Path, value: Any) -> None:
    sb.write_json(path, value)

def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)

def read_jsonl(path: Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out

def _safe_key(dataset: str, relative_path: str) -> str:
    stem = str(Path(relative_path).with_suffix(""))
    slug = re.sub(r"[^A-Za-z0-9._-]+", "__", stem.replace("\\", "/")).strip("_.-")
    digest = hashlib.sha1(relative_path.encode("utf-8")).hexdigest()[:10]
    if len(slug) > 120:
        slug = slug[:120].rstrip("_.-")
    return f"{dataset}__{slug}__{digest}"

QUARANTINE_RE = re.compile(r"\.(incomplete|aborted)\.\d+$")

def memory_index(memory_root: Path) -> dict[str, list[str]]:
    index: dict[str, list[str]] = defaultdict(list)
    root = Path(memory_root)
    if not root.is_dir():
        return index
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        if not ((d / "graph_memory.json").exists() or (d / "CURRENT").exists()):
            continue
        if QUARANTINE_RE.search(d.name):
            continue
        index[d.name].append(d.name)
        parts = d.name.split("__")
        if len(parts) >= 3:
            stem = Path("__".join(parts[1:-1]).replace("__", "/")).stem
            index[stem].append(d.name)
        index[Path(d.name).stem].append(d.name)
    return index

def resolve_video_key(index: dict[str, list[str]], dataset: str, video_id: str,
                      relative_path: str | None = None) -> str | None:
    if relative_path:
        key = _safe_key(dataset, relative_path)
        return key if index.get(key) else None

    for probe in filter(None, [video_id, Path(video_id).stem]):
        hits = [h for h in (index.get(probe) or []) if h.split("__", 1)[0] == dataset]
        if len(hits) == 1:
            return hits[0]
    return None

def video_file_index(bench_root: Path, benchmark: str) -> dict[str, dict]:
    dataset = VIDEO_DATASET.get(benchmark, benchmark)
    root = Path(bench_root) / dataset
    out: dict[str, dict] = {}
    if not root.is_dir():
        return out
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in VIDEO_EXTS:
            continue
        rel = path.relative_to(root).as_posix()
        parts = set(Path(rel).parts)
        if parts & VIDEO_EXCLUDE_PARTS or path.name.startswith("._"):
            continue
        record = {
            "dataset": dataset,
            "path": str(path.resolve()),
            "relative_path": rel,
            "video_key": _safe_key(dataset, rel),
            "size_bytes": path.stat().st_size,
        }
        out.setdefault(path.stem, record)
        out[str(record["video_key"])] = record
    return out

def plan_memory(questions: list[dict], bench_root: Path, memory_root: Path,
                benchmarks: list[str], by_route: bool = True) -> list[dict]:
    needed: dict[str, dict[str, dict]] = defaultdict(dict)
    skipped_by_route = 0
    for q in questions:
        if q.get("video_key"):
            continue
        if by_route and not video_routing.ROUTE_NEEDS_MEMORY.get(q.get("route") or "", True):
            skipped_by_route += 1
            continue
        rel = q.get("video_relpath")
        needed[q["benchmark"]][str(rel or q["video_id"])] = {
            "video_id": q["video_id"], "relative_path": rel,
        }
    if skipped_by_route:
        pass

    rows: list[dict] = []
    for name in benchmarks:
        wanted = needed.get(name)
        if not wanted:
            continue
        index = video_file_index(bench_root, name)
        dataset = VIDEO_DATASET.get(name, name)
        for _, ref in sorted(wanted.items()):
            rel, vid = ref["relative_path"], ref["video_id"]
            record = index.get(_safe_key(dataset, rel)) if rel else None
            if record is None and not rel:
                record = index.get(vid) or index.get(Path(vid).stem)
            if not record:
                rows.append({"benchmark": name, "video_id": vid, "relative_path": rel,
                             "status": "video_file_missing", "video_key": None, "path": None})
                continue
            target = Path(memory_root) / record["video_key"]
            ready = ((target / "graph_memory.json").exists()
                     and (target / "embeddings.npz").exists())
            rows.append({
                "benchmark": name, "video_id": vid, "video_key": record["video_key"],
                "dataset": record["dataset"], "path": record["path"],
                "relative_path": record["relative_path"],
                "size_bytes": record["size_bytes"],
                "status": "ready" if ready else "needs_build",
            })
    return rows

def to_internal(
    benchmark: str, qid: Any, video_id: str, question: str, options: list[str],
    gold: str | None, task_type: str, video_key: str | None,
    video_relpath: str | None = None, duration_sec: Any = None, **metadata: Any,
) -> dict:
    options = [re.sub(r"^\s*[A-Z][.)]\s*", "", str(x)).strip() for x in options]
    labelled = [f"{LETTERS[i]}. {value}" for i, value in enumerate(options)]
    scorable = bool(options) and isinstance(gold, str) and gold.upper() in LETTERS[:len(options)]
    parsed_duration = video_routing.parse_duration(duration_sec)
    return {
        "qid": f"{benchmark}::{qid}",
        "source_qid": str(qid),
        "benchmark": benchmark,
        "video_key": video_key or "",
        "video_id": str(video_id),
        "video_relpath": video_relpath,
        "video_duration_sec": parsed_duration,
        "video_duration_source": (video_routing.SOURCE_ANNOTATION if parsed_duration
                                  else video_routing.SOURCE_MISSING),
        "route": None,
        "type": str(task_type or "unknown"),
        "question": str(question).strip(),
        "options": labelled,
        "n_options": len(options),
        "answer": (gold or "").upper() if scorable else (gold if gold is None else str(gold)),
        "anchor": None,
        "split": "external",
        "score_mode": "mc_exact" if scorable else ("normalized_text" if gold else "unscored"),
        "videomme_task_type": metadata.pop("videomme_task_type", None),
        "cgbench_class": metadata.pop("cgbench_class", None),
        "metadata": metadata,
    }

def attach_routes(questions: list[dict], bench_root: Path, genome: dict | None = None,
                  allow_probe: bool = True, workers: int = 8) -> dict:
    lo, hi = video_routing.routing_bounds(genome)

    paths: dict[str, str] = {}
    for q in questions:
        if video_routing.parse_duration(q.get("video_duration_sec")) is not None:
            continue
        rel = (q.get("video_relpath") or "").strip().strip("/")
        if not rel:
            continue
        bench = q.get("benchmark") or ""
        dataset = VIDEO_DATASET.get(bench, bench)
        for base in (bench_root / dataset, bench_root / bench, bench_root):
            cand = base / rel
            if cand.is_file():
                paths[f"{bench}::{rel}"] = str(cand)
                break

    if allow_probe and paths:
        video_routing.probe_missing(sorted(set(paths.values())), workers=workers,
                                    progress=True)
        video_routing.save_cache()

    counts: Counter = Counter()
    routes: Counter = Counter()
    for q in questions:
        rel = (q.get("video_relpath") or "").strip().strip("/")
        key = f"{q.get('benchmark')}::{rel}" if rel else ""
        dur, source = video_routing.duration_for(
            paths.get(key), annotated=q.get("video_duration_sec"), allow_probe=False)
        q["video_duration_sec"] = round(dur, 3) if dur is not None else None
        q["video_duration_source"] = source
        q["route"] = video_routing.route_for(dur, lo, hi)
        counts[source] += 1
        routes[q["route"] or "unrouted"] += 1

    return {
        "max_direct_sec": lo,
        "max_chunked_sec": hi,
        "duration_sources": dict(counts),
        "routes": dict(routes),
        "by_route": video_routing.summarize_routes(questions),
    }

def assert_video_keys(questions: list[dict]) -> list[dict]:
    bad = []
    for q in questions:
        rel, key = q.get("video_relpath"), q.get("video_key")
        if not key or not rel:
            continue
        expected = _safe_key(VIDEO_DATASET.get(q["benchmark"], q["benchmark"]), str(rel))
        if key != expected:
            bad.append({"qid": q["qid"], "benchmark": q["benchmark"],
                        "video_relpath": rel, "video_key": key, "expected": expected})
    return bad

def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, dict):
        return [str(value[k]) for k in sorted(value)]
    if isinstance(value, (list, tuple)):
        return [str(x) for x in value]
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    labelled = re.findall(r"(?:^|\n)\s*[A-Z][.)]\s*(.*?)(?=\n\s*[A-Z][.)]\s*|$)", text, re.S)
    return [x.strip() for x in labelled] if labelled else [text]

def _embedded_mcq(text: str) -> tuple[str, list[str]]:
    matches = list(re.finditer(r"(?:^|\n)\s*\(?([A-Z])\)?[.)]?\s+", text))
    if len(matches) < 2:
        return text.strip(), []
    stem = text[:matches[0].start()].strip()
    options = []
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        options.append(text[match.end():end].strip())
    return stem, options

def load_benchmark(name: str, bench_root: Path, index: dict[str, list[str]]) -> list[dict]:
    import pandas as pd

    rows: list[dict] = []
    base = bench_root / name

    if name == "videomme":
        df = pd.read_parquet(base / "videomme/test-00000-of-00001.parquet")
        for _, r in df.iterrows():
            rel = f"data/{r.videoID}.mp4"
            rows.append(to_internal(
                name, r.question_id, r.videoID, r.question, _as_list(r.options), r.answer,
                r.task_type, resolve_video_key(index, "videomme", str(r.videoID), rel),
                video_relpath=rel,
                videomme_task_type=str(r.task_type), duration=str(r.duration), domain=str(r.domain),
            ))
    elif name == "videomme_v2":
        df = pd.read_parquet(base / "test.parquet")
        for _, r in df.iterrows():
            rel = f"videos/{r.video_id}.mp4"
            rows.append(to_internal(
                name, r.question_id, r.video_id, r.question, _as_list(r.options), r.answer,
                str(r.third_head), resolve_video_key(index, "videomme_v2", str(r.video_id), rel),
                video_relpath=rel,
                level=str(r.level), group_type=str(r.group_type), group_structure=str(r.group_structure),
            ))
    elif name == "cgbench":
        data = json.loads((base / "cgbench.json").read_text(encoding="utf-8"))
        for r in data:
            rel = f"{r['video_uid']}.mp4"
            rows.append(to_internal(
                name, r["qid"], r["video_uid"], r["question"], r["choices"], r["right_answer"],
                r.get("sub_category", "unknown"),
                resolve_video_key(index, "cgbench", str(r["video_uid"]), rel),
                video_relpath=rel,
                cgbench_class=r.get("domain"), duration_sec=r.get("duration"),
            ))
    elif name == "longvideobench":
        df = pd.read_parquet(base / "validation-00000-of-00001.parquet")
        for _, r in df.iterrows():
            options = [r[f"option{i}"] for i in range(5)
                       if str(r.get(f"option{i}", "")).strip() and str(r.get(f"option{i}")).lower() != "nan"]
            rel = f"videos/{r.video_path}"
            rows.append(to_internal(
                name, r.id, r.video_id, r.question, options, LETTERS[int(r.correct_choice)],
                str(r.question_category),
                resolve_video_key(index, "longvideobench", str(r.video_id), rel),
                video_relpath=rel,
                duration_sec=float(r.duration), duration_group=str(r.duration_group),
            ))
    elif name == "lvbench":
        df = pd.read_parquet(base / "data/train-00000-of-00001.parquet")
        for _, r in df.iterrows():
            stem, options = _embedded_mcq(str(r.question))
            qtype = r.question_type.tolist() if hasattr(r.question_type, "tolist") else r.question_type
            if isinstance(qtype, (list, tuple)):
                qtype = "|".join(map(str, qtype))
            rel = f"video_chunks/{r.video_path}"
            rows.append(to_internal(
                name, r.uid, r.key, stem, options, str(r.answer)[0].upper(), str(qtype),
                resolve_video_key(index, "lvbench", str(r.key), rel),
                video_relpath=rel,
                video_type=str(r.type), time_reference=str(r.time_reference),
            ))
    elif name == "mlvu":
        mbase = bench_root / "mlvu_test"
        for gt in (mbase / "test-ground-truth/test_mcq_gt.json",
                   mbase / "test-ground-truth/test_ssc_gt.json",
                   mbase / "test-ground-truth/test_vs_gt.json"):
            if not gt.exists():
                continue
            for i, r in enumerate(json.loads(gt.read_text(encoding="utf-8"))):
                options = list(r.get("candidates") or [])
                answer = r.get("answer")
                gold = LETTERS[options.index(answer)] if options and answer in options else (
                    str(answer) if answer is not None else None)
                rel = f"MLVU_Test/video/{r['video']}"
                rows.append(to_internal(
                    name, r.get("question_id") or f"{gt.stem}_{i}", Path(r["video"]).stem,
                    r["question"], options, gold, str(r.get("question_type") or gt.stem),
                    resolve_video_key(index, "mlvu_test", Path(r["video"]).stem, rel),
                    video_relpath=rel,
                    duration_sec=r.get("duration"),
                ))
    elif name == "mmvu":
        for r in json.loads((base / "validation.json").read_text(encoding="utf-8")):
            marker = "/videos/"
            rel = str(r["video"]).split(marker, 1)[1] if marker in str(r["video"]) else str(r["video"])
            choices = r.get("choices") or {}
            options = [str(choices[k]) for k in sorted(choices) if str(choices[k]).strip()]
            rows.append(to_internal(
                name, r["id"], Path(rel).stem, r["question"], options, r.get("answer"),
                str(r.get("question_type") or "mmvu"),
                resolve_video_key(index, "mmvu", Path(rel).stem, f"videos/{rel}"),
                video_relpath=f"videos/{rel}",
                subject=(r.get("metadata") or {}).get("subject"),
            ))
    elif name == "video_holmes":
        for r in json.loads((base / "test_Video-Holmes.json").read_text(encoding="utf-8")):
            options_map = r.get("Options") or {}
            options = [options_map[k] for k in sorted(options_map)]
            rel = f"videos_cropped/{r['video ID']}.mp4"
            rows.append(to_internal(
                name, r["Question ID"], r["video ID"], r["Question"], options, r["Answer"],
                str(r.get("Question Type", "unknown")),
                resolve_video_key(index, "video_holmes", str(r["video ID"]), rel),
                video_relpath=rel,
            ))
    elif name == "videommmu":
        for split in ("Perception", "Comprehension", "Adaptation"):
            df = pd.read_parquet(base / split / "test-00000-of-00001.parquet")
            for _, r in df.iterrows():
                rows.append(to_internal(
                    name, f"{split}:{r.id}", str(r.id), r.question, _as_list(r.options), r.answer,
                    str(r.qa_type), resolve_video_key(index, "videommmu", str(r.id)),
                    cognitive_stage=split,
                    protocol_note="the Adaptation auxiliary image is NOT injected in the harness path",
                ))
    elif name == "longvideoreason":
        df = pd.read_parquet(base / "data/test-00000-of-00001.parquet")
        for _, r in df.iterrows():
            stem, options = _embedded_mcq(str(r.problem))
            match = re.search(r"<answer>\s*([A-Z])\s*</answer>", str(r.answer), re.I)
            rows.append(to_internal(
                name, r.problem_id, Path(str(r.videos)).stem, stem, options,
                match.group(1).upper() if match else str(r.answer), str(r.problem_type),
                resolve_video_key(index, "longvideoreason_videos", Path(str(r.videos)).stem),
                data_type=str(r.data_type),
            ))
    elif name == "videoreasonbench":
        for category in ("card", "chip", "cup", "file_sys", "grid", "hrd"):
            df = pd.read_parquet(base / category / "test-00000-of-00001.parquet")
            for _, r in df.iterrows():
                rel = str(r.video)
                rows.append(to_internal(
                    name, r.id, Path(rel).stem, r.question, [], r.answer, str(r.dim),
                    resolve_video_key(index, "videoreasonbench", Path(rel).stem, rel),
                    video_relpath=rel,
                    category=category,
                    protocol_note="open-ended; the solver parses option letters, so this is unscored",
                ))
    else:
        raise ValueError(f"unknown benchmark: {name}")
    return rows

def parse_benchmarks(value: str) -> list[str]:
    if value == "all":
        return list(ALL_BENCHMARKS)
    names = [x.strip() for x in value.split(",") if x.strip()]
    unknown = sorted(set(names) - set(ALL_BENCHMARKS))
    if unknown:
        raise ValueError(f"unknown benchmark(s): {unknown}; available: {ALL_BENCHMARKS}")
    return names

def audit(questions: list[dict]) -> dict:
    by_bench: dict[str, dict] = {}
    for q in questions:
        d = by_bench.setdefault(q["benchmark"], {
            "questions": 0, "scorable": 0, "memory_ready": 0, "memory_missing": 0,
            "videos": set(), "videos_with_memory": set(),
        })
        d["questions"] += 1
        d["scorable"] += q["score_mode"] == "mc_exact"
        d["videos"].add(q.get("video_relpath") or q["video_id"])
        if q["video_key"]:
            d["memory_ready"] += 1
            d["videos_with_memory"].add(q["video_key"])
        else:
            d["memory_missing"] += 1
    return {
        "created_at": utc_now(),
        "model_calls": 0,
        "total_questions": len(questions),
        "by_benchmark": {
            name: {
                "questions": d["questions"],
                "scorable_mc": d["scorable"],
                "videos": len(d["videos"]),
                "videos_with_memory": len(d["videos_with_memory"]),
                "questions_with_memory": d["memory_ready"],
                "questions_without_memory": d["memory_missing"],
                "memory_coverage": round(d["memory_ready"] / d["questions"], 4) if d["questions"] else 0,
                "runnable": d["memory_ready"] > 0,
            }
            for name, d in sorted(by_bench.items())
        },
        "note": (
            "questions_without_memory are NOT answered. Build memory for those videos "
            "question-blind first, otherwise the benchmark subset is not the published one."
        ),
    }

def build_memory(args: argparse.Namespace, plan: list[dict], run_dir: Path) -> dict:
    import subprocess

    default_script = (ROOT / "Qwen-MM-Plugins/src/capabilities/video-memory"
                      / "skill/script/build_memory/build_memory.sh")
    script = Path(os.environ.get("VEH_BUILD_MEMORY_SCRIPT") or default_script)
    if not script.exists():
        raise SystemExit("build_memory.sh not found; cannot build benchmark memory")

    todo = [r for r in plan if r["status"] == "needs_build"]
    if args.limit:
        todo = todo[: args.limit]
    memory_root = Path(args.memory_root).expanduser().resolve()
    memory_root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    done, failed = [], []
    for i, row in enumerate(todo, 1):
        key = row["video_key"]
        out = memory_root / key
        log = run_dir / f"logs/memory_{key}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        if args.dry_run:
            continue
        try:
            with log.open("a", encoding="utf-8") as fh:
                proc = subprocess.run(
                    ["bash", str(script), row["path"], "--output-dir", str(out),
                     "--no-asr", "--p2-workers", str(args.p2_workers)],
                    cwd=str(ROOT), env=env, text=True,
                    stdout=fh, stderr=subprocess.STDOUT, check=False,
                )
            if proc.returncode:
                raise RuntimeError(f"build_memory exit {proc.returncode}; log {log}")
            with log.open("a", encoding="utf-8") as fh:
                proc = subprocess.run(
                    [sys.executable, str(VEH_DIR / "enrich_memory.py"), "--videos", key,
                     "--workers", str(args.enrich_workers)],
                    cwd=str(ROOT), env={**env, "VEH_MEMORY_ROOT": str(memory_root)},
                    text=True, stdout=fh, stderr=subprocess.STDOUT, check=False,
                )
            if proc.returncode:
                raise RuntimeError(f"enrich_memory exit {proc.returncode}; log {log}")
            if not ((out / "graph_memory.json").exists() and (out / "embeddings.npz").exists()):
                raise RuntimeError("graph or index missing after build")
            done.append(key)
        except Exception as exc:
            failed.append({"video_key": key, "benchmark": row["benchmark"],
                           "video_id": row["video_id"], "error": str(exc)[:500]})
            if not args.continue_on_error:
                break

    report = {"updated_at": utc_now(), "planned": len(todo),
              "built": len(done), "failed": len(failed), "failures": failed}
    write_json(run_dir / "memory_build_report.json", report)
    write_jsonl(run_dir / "manifests/memory_plan.jsonl", plan)
    return report

def run_benchmark(args: argparse.Namespace, name: str, questions: list[dict],
                  run_dir: Path) -> dict:
    import solve
    from common import adaptation_context, llm_cost, load_cfg, load_genome, reset_cost

    ctx = adaptation_context()
    tag = f"{name}__{ctx['arm']}__ckpt{args.checkpoint}__{args.thinking_mode}"
    out = run_dir / f"results/{tag}.json"
    if out.exists() and not args.force:
        return json.loads(out.read_text(encoding="utf-8")).get("summary") or {}

    runnable = [q for q in questions if q["video_key"] and q["score_mode"] == "mc_exact"]
    if args.limit:
        runnable = runnable[: args.limit]
    attempted = {q["qid"] for q in runnable}
    skipped = [q for q in questions if q["qid"] not in attempted]
    if not runnable:
        return {}

    g, cfg = load_genome(args.genome), load_cfg()
    reset_cost()
    records = solve.run(runnable, g, cfg, "retrieve", args.solve_workers, quiet=args.quiet)
    summary = solve.summarize(records)
    fp = solve.protocol_fingerprint(g, cfg, "retrieve", "external", runnable, args.solve_workers)
    cost = llm_cost()

    n_total = len(questions)
    n_scorable = sum(q["score_mode"] == "mc_exact" for q in questions)
    correct = sum(bool(r.get("correct")) for r in records)
    payload = {
        "benchmark": name,
        "arm": ctx["arm"],
        "adaptation_mode": ctx["adaptation_mode"],
        "checkpoint": ctx["checkpoint"],
        "genome_version": g["version"],
        "mode": "retrieve",
        "split": "external",
        "protocol": "graph_memory_pipeline_vision_only",
        "protocol_fingerprint": fp["fingerprint"],
        "fingerprint_parts": fp["parts"],
        "state": ctx,
        "thinking_mode": args.thinking_mode,
        "coverage": {
            "questions_total": n_total,
            "questions_scorable_mc": n_scorable,
            "questions_attempted": len(records),
            "questions_skipped_no_memory": sum(1 for q in skipped if not q["video_key"]),
            "questions_skipped_unscorable": sum(1 for q in skipped if q["score_mode"] != "mc_exact"),
            "attempt_rate_vs_scorable": round(len(records) / n_scorable, 4) if n_scorable else None,
        },
        "accuracy_completed": round(correct / len(records), 4) if records else None,
        "accuracy_all_scorable": round(correct / n_scorable, 4) if n_scorable else None,
        "officially_comparable": False,
        "caveats": [
            "vision-only, no subtitles or audio",
            "no gold macro/interval: anchor_ok, strict_correct and gold_macro_hit are not measurable",
            "questions whose video has no pre-built memory are skipped, not frame-answered",
        ],
        "summary": summary,
        "cost": cost,
        "records": records,
    }
    write_json(out, payload)
    write_jsonl(run_dir / f"manifests/{name}_skipped.jsonl",
                [{k: v for k, v in q.items() if k != "metadata"} for q in skipped])
    solve.print_report(summary, f"{name}  arm={ctx['arm']}  ckpt{args.checkpoint}")
    return summary

def summarize_run(run_dir: Path) -> dict:
    rows: list[dict] = []
    for path in sorted((run_dir / "results").glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        summary = payload.get("summary") or {}
        coverage = payload.get("coverage") or {}
        rows.append({
            "benchmark": payload.get("benchmark"),
            "arm": payload.get("arm"),
            "checkpoint": payload.get("checkpoint"),
            "genome_version": payload.get("genome_version"),
            "thinking_mode": payload.get("thinking_mode"),
            "protocol_fingerprint": (payload.get("protocol_fingerprint") or "")[:16],
            "questions_total": coverage.get("questions_total"),
            "questions_scorable_mc": coverage.get("questions_scorable_mc"),
            "questions_attempted": coverage.get("questions_attempted"),
            "accuracy_completed": payload.get("accuracy_completed"),
            "accuracy_all_scorable": payload.get("accuracy_all_scorable"),
            "api_errors": summary.get("api_errors"),
            "unparsed": summary.get("unparsed"),
            "replay_required": (summary.get("replay") or {}).get("required"),
            "replay_gate_pass_rate": (summary.get("replay") or {}).get("gate_pass_rate"),
            "dense_timeouts": (summary.get("retrieval") or {}).get("dense_timeouts"),
            "bm25_fallbacks": (summary.get("retrieval") or {}).get("bm25_fallbacks"),
            "total_tokens": ((payload.get("cost") or {}).get("total") or {}).get("total_tokens"),
            "file": path.name,
        })
    out = {"updated_at": utc_now(), "runs": rows}
    write_json(run_dir / "summary.json", out)
    write_jsonl(run_dir / "metrics/benchmark_metrics.jsonl", rows)
    return out

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Run public benchmarks through the solver")
    ap.add_argument("command", choices=["audit", "plan-memory", "build-memory", "run", "summarize"])
    ap.add_argument("--benchmarks", default="videomme", help="all|comma-separated names")
    ap.add_argument("--benchmark-root", default=str(ROOT / "benchmarks"))
    ap.add_argument("--memory-root", default=str(ROOT / "experiment_memory"))
    ap.add_argument("--state-bundle", default=None,
                    help="pinned bundle; required for `run` so the arm is unambiguous")
    ap.add_argument("--adaptation-mode", choices=["frozen", "evolve"], default="frozen")
    ap.add_argument("--arm", default=None)
    ap.add_argument("--checkpoint", type=int, default=0)
    ap.add_argument("--genome", default=None, help="defaults to the bundle genome")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--config-file", default=str(ROOT / "qwen-mm-config"))
    ap.add_argument("--thinking-mode", choices=["no_thinking", "thinking"], default="no_thinking")
    ap.add_argument("--solve-workers", type=int, default=4)
    ap.add_argument("--p2-workers", type=int, default=2)
    ap.add_argument("--enrich-workers", type=int, default=4)
    ap.add_argument("--task-scope", choices=["mc", "all"], default="mc")
    ap.add_argument("--no-route-probe", action="store_true",
                    help="do not run ffprobe for videos with an unknown duration; rely on annotations and the existing cache only, "
                         "so gaps become unrouted and trip the hard gate")
    ap.add_argument("--allow-unrouted", action="store_true",
                    help="allow unroutable questions (written to unrouted_questions.jsonl instead of exiting). "
                         "Such questions are skipped by every entry point at run time; they are **not** treated as short videos by default")
    ap.add_argument("--plan-all-videos", action="store_true",
                    help="plan-memory schedules every video without filtering by route. "
                         "By default memories are built only for videos >=30min (route=video_memory)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--continue-on-error", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    return ap

def main() -> int:
    args = build_parser().parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    for child in ("results", "manifests", "metrics", "logs"):
        (run_dir / child).mkdir(parents=True, exist_ok=True)

    if args.command == "summarize":
        summarize_run(run_dir)
        return 0
    if args.state_bundle:
        bundle = Path(args.state_bundle).expanduser().resolve()
        os.environ.update(sb.env_for(bundle, args.adaptation_mode, args.arm,
                                     args.checkpoint, args.memory_root))
        if not args.genome:
            args.genome = str(bundle / "genome.json")
    elif args.command == "run":
        raise SystemExit(
            "--state-bundle is required for `run`: without it the result cannot be "
            "attributed to a frozen or an evolved state"
        )
    os.environ.setdefault("VEH_MEMORY_ROOT", str(Path(args.memory_root).expanduser().resolve()))
    os.environ["VEH_CONFIG_FILE"] = str(Path(args.config_file).expanduser().resolve())
    os.environ["QWEN_MM_CONFIG"] = os.environ["VEH_CONFIG_FILE"]
    os.environ["VEH_STORE_DIR"] = str(run_dir / "store")
    os.environ["VEH_TOKEN_LOG"] = str(run_dir / "metrics/token_calls.jsonl")
    os.environ["VEH_STAGE"] = "benchmark"
    os.environ["VEH_THINKING"] = "1" if args.thinking_mode == "thinking" else "0"
    for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(key, None)
    os.environ["NO_PROXY"] = os.environ["no_proxy"] = "localhost,127.0.0.1"

    names = parse_benchmarks(args.benchmarks)
    index = memory_index(Path(args.memory_root).expanduser().resolve())
    all_questions: dict[str, list[dict]] = {}
    load_errors: dict[str, str] = {}
    for name in names:
        try:
            rows = load_benchmark(name, Path(args.benchmark_root).expanduser().resolve(), index)
            if args.task_scope == "mc":
                rows = [r for r in rows if r["score_mode"] == "mc_exact"]
            all_questions[name] = rows
        except Exception as exc:
            load_errors[name] = f"{type(exc).__name__}: {exc}"

    flat = [q for rows in all_questions.values() for q in rows]

    mismatched = assert_video_keys(flat)
    if mismatched:
        write_jsonl(run_dir / "manifests/video_key_mismatches.jsonl", mismatched)
        for row in mismatched[:5]:
            pass
        raise SystemExit(
            f"{len(mismatched)} questions have a video_key that does not match their "
            f"video path; see {run_dir / 'manifests/video_key_mismatches.jsonl'}"
        )

    routing_report = attach_routes(
        flat, Path(args.benchmark_root).expanduser().resolve(),
        allow_probe=not args.no_route_probe,
    )

    unrouted = video_routing.assert_routes(flat)
    if unrouted and not args.allow_unrouted:
        write_jsonl(run_dir / "manifests/unrouted_questions.jsonl", unrouted)
        reasons = Counter(r["reason"] for r in unrouted)
        for row in unrouted[:5]:
            pass
        raise SystemExit(
            f"{len(unrouted)} questions cannot be routed ({dict(reasons)}); see "
            f"{run_dir / 'manifests/unrouted_questions.jsonl'}. Pass --allow-unrouted to "
            f"write the manifest anyway (those questions will be skipped at run time)."
        )
    if unrouted:
        write_jsonl(run_dir / "manifests/unrouted_questions.jsonl", unrouted)

    report = audit(flat)
    report["load_errors"] = load_errors
    report["memory_root"] = str(Path(args.memory_root).expanduser().resolve())
    report["state_bundle"] = args.state_bundle
    report["video_key_mismatches"] = 0
    report["routing"] = routing_report
    report["unrouted_questions"] = len(unrouted)
    write_json(run_dir / "audit_summary.json", report)
    write_jsonl(run_dir / "manifests/questions.jsonl",
                [{k: v for k, v in q.items() if k != "metadata"} for q in flat])

    if args.command == "audit":
        return 0

    if args.command in ("plan-memory", "build-memory"):
        plan = plan_memory(flat, Path(args.benchmark_root).expanduser().resolve(),
                           Path(args.memory_root).expanduser().resolve(), names,
                           by_route=not args.plan_all_videos)
        write_jsonl(run_dir / "manifests/memory_plan.jsonl", plan)
        counts = Counter(r["status"] for r in plan)
        by_bench = defaultdict(Counter)
        for r in plan:
            by_bench[r["benchmark"]][r["status"]] += 1
        summary = {
            "created_at": utc_now(),
            "total_videos_planned": len(plan),
            "by_status": dict(counts),
            "by_benchmark": {k: dict(v) for k, v in sorted(by_bench.items())},
            "plan_file": str(run_dir / "manifests/memory_plan.jsonl"),
        }
        write_json(run_dir / "memory_plan_summary.json", summary)
        if args.command == "plan-memory":
            return 0
        if load_errors:
            raise SystemExit(f"adapter load errors; run `audit` first: {load_errors}")
        build_memory(args, plan, run_dir)
        return 0

    if load_errors:
        raise SystemExit(f"adapter load errors; run `audit` first: {load_errors}")

    for name in names:
        run_benchmark(args, name, all_questions.get(name) or [], run_dir)
    summarize_run(run_dir)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
