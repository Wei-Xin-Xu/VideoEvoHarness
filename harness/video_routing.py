from __future__ import annotations

import json
import os
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable

VEH_DIR = Path(__file__).resolve().parent
ROOT = VEH_DIR.parent

DEFAULT_MAX_DIRECT_SEC = 300.0
DEFAULT_MAX_CHUNKED_SEC = 1800.0

ROUTE_DIRECT = "direct_read_video"
ROUTE_CHUNKED = "chunked_read_video"
ROUTE_VIDEO_MEMORY = "video_memory"
ROUTES = (ROUTE_DIRECT, ROUTE_CHUNKED, ROUTE_VIDEO_MEMORY)

ROUTE_NEEDS_MEMORY = {ROUTE_DIRECT: False, ROUTE_CHUNKED: False, ROUTE_VIDEO_MEMORY: True}

SOURCE_ANNOTATION = "annotation"
SOURCE_CACHE = "cache"
SOURCE_FFPROBE = "ffprobe"
SOURCE_MISSING = "missing"

CACHE_FILES = (ROOT / ".cache" / "video_durations.json",)
WRITE_CACHE = ROOT / ".cache" / "video_durations.json"

_cache: dict[str, float] | None = None
_cache_lock = threading.Lock()

def parse_duration(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value) if float(value) > 0 else None
    text = str(value).strip()
    if not text:
        return None
    try:
        num = float(text)
    except ValueError:
        parts = text.split(":")
        if len(parts) in (2, 3) and all(p.strip() for p in parts):
            try:
                nums = [float(p) for p in parts]
            except ValueError:
                return None
            num = 0.0
            for n in nums:
                num = num * 60 + n
        else:
            return None
    return num if num > 0 else None

def _load_cache() -> dict[str, float]:
    global _cache
    if _cache is not None:
        return _cache
    with _cache_lock:
        if _cache is not None:
            return _cache
        merged: dict[str, float] = {}
        for path in CACHE_FILES:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            for key, val in (data or {}).items():
                dur = parse_duration(val)
                if dur is not None:
                    merged[str(key)] = dur
        _cache = merged
        return _cache

def save_cache() -> int:
    cache = _load_cache()
    if not cache:
        return 0
    try:
        WRITE_CACHE.parent.mkdir(parents=True, exist_ok=True)
        WRITE_CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=1, sort_keys=True),
                               encoding="utf-8")
    except Exception:
        return 0
    return len(cache)

def ffprobe_duration(video_path: str | Path, timeout: float = 60.0) -> float | None:
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
           "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)]
    try:
        out = subprocess.run(cmd, capture_output=True, timeout=timeout, text=True)
    except Exception:
        return None
    if out.returncode != 0:
        return None
    return parse_duration(out.stdout.strip())

def probe_missing(paths: Iterable[str | Path], workers: int = 8,
                  progress: bool = False) -> dict[str, float]:
    cache = _load_cache()
    todo = []
    seen: set[str] = set()
    for p in paths:
        key = str(p)
        if key in cache or key in seen:
            continue
        seen.add(key)
        if Path(key).is_file():
            todo.append(key)
    if not todo:
        return {}
    found: dict[str, float] = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for key, dur in zip(todo, pool.map(ffprobe_duration, todo)):
            if dur is not None:
                found[key] = dur
    if found:
        with _cache_lock:
            cache.update(found)
    if progress and found:
        pass
    return found

def duration_for(video_path: str | Path | None, annotated: Any = None,
                 allow_probe: bool = False) -> tuple[float | None, str]:
    dur = parse_duration(annotated)
    if dur is not None:
        return dur, SOURCE_ANNOTATION
    if video_path:
        key = str(video_path)
        cached = _load_cache().get(key)
        if cached is not None:
            return cached, SOURCE_CACHE
        if allow_probe and Path(key).is_file():
            probed = ffprobe_duration(key)
            if probed is not None:
                with _cache_lock:
                    _load_cache()[key] = probed
                return probed, SOURCE_FFPROBE
    return None, SOURCE_MISSING

def route_for(duration_sec: float | None,
              max_direct_sec: float = DEFAULT_MAX_DIRECT_SEC,
              max_chunked_sec: float = DEFAULT_MAX_CHUNKED_SEC) -> str | None:
    if duration_sec is None or duration_sec <= 0:
        return None
    d = float(duration_sec)
    if d <= float(max_direct_sec):
        return ROUTE_DIRECT
    if d >= float(max_chunked_sec):
        return ROUTE_VIDEO_MEMORY
    return ROUTE_CHUNKED

def routing_bounds(genome: dict | None) -> tuple[float, float]:
    node = {}
    if isinstance(genome, dict):
        node = genome.get("routing") or {}
        if not isinstance(node, dict):
            node = {}
    lo = parse_duration(node.get("max_direct_sec")) or DEFAULT_MAX_DIRECT_SEC
    hi = parse_duration(node.get("max_chunked_sec")) or DEFAULT_MAX_CHUNKED_SEC
    if hi <= lo:
        return DEFAULT_MAX_DIRECT_SEC, DEFAULT_MAX_CHUNKED_SEC
    return lo, hi

def route_of(question: dict, genome: dict | None = None) -> str | None:
    recorded = question.get("route")
    if isinstance(recorded, str) and recorded in ROUTES:
        return recorded
    lo, hi = routing_bounds(genome)
    return route_for(parse_duration(question.get("video_duration_sec")), lo, hi)

def needs_memory(question: dict, genome: dict | None = None) -> bool:
    route = route_of(question, genome)
    return ROUTE_NEEDS_MEMORY.get(route or ROUTE_VIDEO_MEMORY, True)

SCOPE_KEY = "__by_route"

ROUTE_SCOPED_PATHS: frozenset[str] = frozenset({
    "policy.read_video.fps",
    "policy.read_video.max_frames_per_call",
    "policy.read_video.max_dim",
    "policy.read_video.max_segments",
    "policy.read_video.max_window_sec",
    "policy.read_video.node_window_pad_sec",
    "policy.read_video.focus_weight",
    "policy.read_video.max_calls_per_question",
    "evidence.frames.fps",
    "evidence.frames.chunk_sec",
    "evidence.frames.max_chunks",
    "evidence.frames.max_dim",
    "evidence.chunked_qa.chunk_seconds",
    "evidence.chunked_qa.coarse_fps",
    "evidence.chunked_qa.detail_fps",
    "evidence.chunked_qa.max_detail_spans",
})

def scoped_path(path: str, route: str) -> str:
    head, _, leaf = path.rpartition(".")
    return f"{head}.{SCOPE_KEY}.{route}.{leaf}" if head else f"{SCOPE_KEY}.{route}.{leaf}"

def split_scoped(path: str) -> tuple[str, str | None]:
    parts = path.split(".")
    if SCOPE_KEY not in parts:
        return path, None
    i = parts.index(SCOPE_KEY)
    if i + 2 > len(parts) - 1:
        return path, None
    route = parts[i + 1]
    base = parts[:i] + parts[i + 2:]
    return ".".join(base), route

def _dig(node: Any, keys: list[str]) -> Any:
    for k in keys:
        if not isinstance(node, dict) or k not in node:
            return None
        node = node[k]
    return node

def gene(genome: dict | None, path: str, route: str | None = None, default: Any = None) -> Any:
    if not isinstance(genome, dict):
        return default
    head, _, leaf = path.rpartition(".")
    if route:
        over = _dig(genome, (head.split(".") if head else []) + [SCOPE_KEY, route, leaf])
        if over is not None:
            return over
    base = _dig(genome, path.split("."))
    return default if base is None else base

def route_overrides(genome: dict | None) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not isinstance(genome, dict):
        return out

    def walk(node: Any, prefix: list[str]) -> None:
        if not isinstance(node, dict):
            return
        for key, val in node.items():
            if key == SCOPE_KEY and isinstance(val, dict):
                for route, leaves in val.items():
                    if not isinstance(leaves, dict):
                        continue
                    for leaf, v in leaves.items():
                        if leaf.startswith("_"):
                            continue
                        out.setdefault(route, {})[".".join(prefix + [leaf])] = v
            elif isinstance(val, dict) and not key.startswith("_"):
                walk(val, prefix + [key])

    walk(genome, [])
    return out

def assert_routes(questions: list[dict], genome: dict | None = None) -> list[dict]:
    lo, hi = routing_bounds(genome)
    bad: list[dict] = []
    for q in questions:
        dur = parse_duration(q.get("video_duration_sec"))
        recorded = q.get("route")
        expected = route_for(dur, lo, hi)
        if dur is None:
            bad.append({"qid": q.get("qid"), "reason": "duration_missing",
                        "benchmark": q.get("benchmark"),
                        "video_relpath": q.get("video_relpath")})
        elif recorded not in ROUTES:
            bad.append({"qid": q.get("qid"), "reason": "route_missing",
                        "duration_sec": dur, "route": recorded})
        elif recorded != expected:
            bad.append({"qid": q.get("qid"), "reason": "route_mismatch",
                        "duration_sec": dur, "route": recorded, "expected": expected})
    return bad

def summarize_routes(questions: list[dict]) -> dict:
    out: dict[str, dict] = {}
    for q in questions:
        route = q.get("route") or "unrouted"
        cell = out.setdefault(route, {
            "questions": 0, "videos": set(), "needs_memory": 0,
            "memory_ready": 0, "min_sec": None, "max_sec": None,
        })
        cell["questions"] += 1
        cell["videos"].add(q.get("video_relpath") or q.get("video_id"))
        if ROUTE_NEEDS_MEMORY.get(route, False):
            cell["needs_memory"] += 1
            if q.get("video_key"):
                cell["memory_ready"] += 1
        dur = parse_duration(q.get("video_duration_sec"))
        if dur is not None:
            cell["min_sec"] = dur if cell["min_sec"] is None else min(cell["min_sec"], dur)
            cell["max_sec"] = dur if cell["max_sec"] is None else max(cell["max_sec"], dur)
    for cell in out.values():
        cell["videos"] = len(cell["videos"])
        for k in ("min_sec", "max_sec"):
            if cell[k] is not None:
                cell[k] = round(float(cell[k]), 1)
    return out
