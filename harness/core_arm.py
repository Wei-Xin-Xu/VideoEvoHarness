from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import solve
import video_routing
from common import (
    ContentFilteredError,
    Memory,
    ROOT,
    VEH_DIR,
    active_memory_dir,
    load_cfg,
    load_genome,
    load_questions,
    reset_cost,
)

try:
    from benchmark_audit import VIDEO_DATASET as _DS
except Exception:
    _DS = {
        "videomme": "videomme", "videomme_v2": "videomme_v2", "cgbench": "cgbench",
        "longvideobench": "longvideobench", "lvbench": "lvbench", "mlvu": "mlvu_test",
        "mmvu": "mmvu", "video_holmes": "video_holmes", "videommmu": "videommmu",
        "longvideoreason": "longvideoreason_videos", "videoreasonbench": "videoreasonbench",
    }
VIDEO_DATASET = _DS

_bench_root = Path(os.environ.get("VEH_BENCH_ROOT")
                   or Path(__file__).resolve().parents[1] / "benchmarks")

_core_api: dict = {}

def _core() -> dict:
    if _core_api:
        return _core_api
    src = ROOT / "Qwen-MM-Plugins" / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from shared.env import (
        DEFAULT_FPS,
        MAX_RESPONSE_BYTES,
        MAX_TOTAL_FRAMES,
        VIDEO_BUDGET_TOKENS,
        VIDEO_MIN_PIXELS,
    )
    from shared.image import budget_to_pixels, smart_resize
    from shared.video import (
        compute_dynamic_fps,
        extract_frames_by_seeking,
        format_timestamp,
        get_video_info,
    )

    _core_api.update(
        DEFAULT_FPS=DEFAULT_FPS,
        MAX_RESPONSE_BYTES=MAX_RESPONSE_BYTES,
        MAX_TOTAL_FRAMES=MAX_TOTAL_FRAMES,
        VIDEO_BUDGET_TOKENS=VIDEO_BUDGET_TOKENS,
        VIDEO_MIN_PIXELS=VIDEO_MIN_PIXELS,
        budget_to_pixels=budget_to_pixels,
        smart_resize=smart_resize,
        compute_dynamic_fps=compute_dynamic_fps,
        extract_frames_by_seeking=extract_frames_by_seeking,
        format_timestamp=format_timestamp,
        get_video_info=get_video_info,
    )
    return _core_api

_media_info_mod = None
_media_info_cache: dict[str, str] = {}

def _media_info_handle():
    global _media_info_mod
    if _media_info_mod is None:
        import importlib.util

        src = ROOT / "Qwen-MM-Plugins" / "src"
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))
        path = src / "capabilities/core/qwen_mm_plugins_core/readers/media_info.py"
        spec = importlib.util.spec_from_file_location("_core_media_info", path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load official media_info from {path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _media_info_mod = mod
    return _media_info_mod

def media_info(video_path: str, skip_media_line: bool = True) -> str:
    if video_path in _media_info_cache:
        return _media_info_cache[video_path]
    out = ""
    try:
        blocks = _media_info_handle().handle({"path": video_path})
        parts = [b.get("text", "") for b in blocks if isinstance(b, dict)]
        out = "\n".join(p for p in parts if p).strip()
        if skip_media_line:
            lines = [ln for ln in out.splitlines() if not ln.startswith("Media: ")]
            out = "\n".join(lines).strip()
    except Exception as exc:
        out = f"(media_info unavailable: {str(exc)[:120]})"
    _media_info_cache[video_path] = out
    return out

OFFICIAL_SKIM_CHUNK_SEC = 300.0
OFFICIAL_SKIM_FPS = 1.0
OFFICIAL_SKIM_BUDGET = "normal"
OFFICIAL_DETAIL_FPS = 2.0
OFFICIAL_DETAIL_BUDGET = "large"

_CHUNK_WORKERS_DEFAULT = 1

def _chunk_workers() -> int:
    try:
        n = int(os.environ.get("VEH_CORE_CHUNK_WORKERS", _CHUNK_WORKERS_DEFAULT))
    except (TypeError, ValueError):
        return _CHUNK_WORKERS_DEFAULT
    return max(1, n)

FRAME_LAYOUT = "interleaved"

def _gateway_max_frames() -> int:
    try:
        import llm_provider
        return llm_provider.max_frames()
    except Exception:
        return int(os.environ.get("VEH_CORE_MAX_FRAMES", "192"))

GATEWAY_MAX_FRAMES = _gateway_max_frames()

CORE_PROMPT_FILE = "prompts/core_frames_v1.txt"

def build_core_prompt(q: dict, note: str, g: dict, evidence: str = "") -> str:
    rel = (g.get("answer", {}) or {}).get("core_prompt_file") or CORE_PROMPT_FILE
    tpl = (VEH_DIR / rel).read_text(encoding="utf-8")

    block = f"SAMPLING\n{note.strip()}" if note.strip() else ""

    if evidence.strip():
        block = (block + "\n\n" if block else "") + f"VIDEO GRAPH MEMORY EVIDENCE\n{evidence.strip()}"

    return tpl.format(evidence_block=block, question=q["question"],
                      options="\n".join(q["options"]))

CORE_CHUNK_OBSERVE_FILE = "prompts/core_chunk_observe_v1.txt"
CORE_CHUNK_SYNTH_FILE = "prompts/core_chunk_synthesize_v1.txt"

CHUNK_OBS_MAX_CHARS = int(os.environ.get("VEH_CORE_CHUNK_OBS_CHARS", "2000"))

def build_chunk_observe_prompt(q: dict, g: dict, chunk_idx: int, n_chunks: int,
                               chunk_span: str, sampling: str) -> str:
    rel = (g.get("answer", {}) or {}).get("core_chunk_observe_file") or CORE_CHUNK_OBSERVE_FILE
    tpl = (VEH_DIR / rel).read_text(encoding="utf-8")
    return tpl.format(chunk_idx=chunk_idx, n_chunks=n_chunks, chunk_span=chunk_span,
                      sampling=sampling.strip(), question=q["question"],
                      options="\n".join(q["options"]))

def build_chunk_synth_prompt(q: dict, g: dict, n_chunks: int, observations: str,
                             evidence: str = "") -> str:
    rel = (g.get("answer", {}) or {}).get("core_chunk_synth_file") or CORE_CHUNK_SYNTH_FILE
    tpl = (VEH_DIR / rel).read_text(encoding="utf-8")
    block = f"SEGMENT OBSERVATIONS\n{observations.strip()}"
    if evidence.strip():
        block += f"\n\nVIDEO GRAPH MEMORY EVIDENCE\n{evidence.strip()}"
    return tpl.format(evidence_block=block, question=q["question"], n_chunks=n_chunks,
                      options="\n".join(q["options"]))

def plan_chunks(duration: float, chunk_sec: float = OFFICIAL_SKIM_CHUNK_SEC,
                max_chunks: int = 0,
                windows: list[list[float]] | None = None) -> list[list[float]]:
    if duration <= 0:
        return []
    if windows:
        chunks = [[max(0.0, float(a)), min(duration, float(b))]
                  for a, b in windows if float(b) > float(a)]
    else:
        n = max(1, math.ceil(duration / chunk_sec))
        chunks = [[i * chunk_sec, min((i + 1) * chunk_sec, duration)] for i in range(n)]
    if max_chunks > 0:
        chunks = chunks[:max_chunks]
    return chunks

def chunk_frames(video_path: str, duration: float, start: float, end: float,
                 fps: float = OFFICIAL_SKIM_FPS, budget: str = OFFICIAL_SKIM_BUDGET,
                 stats: dict | None = None) -> list[tuple[str, str]]:
    api = _core()
    st = stats if stats is not None else {}
    got = read_video_once(video_path, start, end, fps=fps, budget=budget,
                          max_frames=int(api["MAX_TOTAL_FRAMES"]), stats=st)
    if not got:
        return []

    gate_cap = GATEWAY_MAX_FRAMES
    if gate_cap > 0 and len(got) > gate_cap:
        step = (len(got) - 1) / (gate_cap - 1) if gate_cap > 1 else 0
        kept = [got[round(i * step)] for i in range(gate_cap)]
        st["gateway_clipped_from"] = len(got)
        st["sent_mb"] = round(sum(len(b) for _, b in kept) / 1024 / 1024, 3)
        got = kept

    span = end - start
    st["n_frames"] = len(got)
    st["fps_effective"] = round(len(got) / span, 4) if span > 0 else 0.0
    st["fps_requested"] = fps
    st["sampled_note"] = (
        f"Sampled: {len(got)} frames @ {st['fps_effective']:.2f} fps | "
        f"{st.get('frame_wh', ['?', '?'])[0]}x{st.get('frame_wh', ['?', '?'])[1]} (WxH) per frame"
    )

    return [(f"data:image/jpeg;base64,{b64}", api["format_timestamp"](ts_, duration))
            for ts_, b64 in got]

def budget_for_max_dim(max_dim: int) -> str:
    d = int(max_dim or 512)
    if d <= 320:
        return "small"
    if d <= 640:
        return "normal"
    return "large"

class VideoFile:

    __slots__ = ("video_path",)

    def __init__(self, video_path: str | Path):
        self.video_path = str(video_path)

def resolve_video_path(q: dict, bench_root: Path | str | None = None) -> Path | None:
    root = Path(bench_root) if bench_root else _bench_root
    direct = (q.get("video_path") or "").strip()
    if direct and Path(direct).is_file():
        return Path(direct)
    rel = (q.get("video_relpath") or "").strip().strip("/")
    if rel:
        b = (q.get("benchmark") or "").strip()
        ds = VIDEO_DATASET.get(b, b)
        cands: list[Path] = []
        for base in (root / ds, root / b, root):
            if base.is_dir():
                cands.append(base / rel)
        for cand in cands:
            if cand.is_file():
                return cand
        if (root / ds).is_dir():
            target = str(Path(rel)).lower()
            for p in (root / ds).rglob(Path(rel).name):
                if str(p).lower().endswith(target):
                    return p

    vk = (q.get("video_key") or "").strip()
    if vk:
        try:
            p = Memory(vk).video_path
            if p and Path(p).is_file():
                return Path(p)
        except Exception:
            pass
    return None

_duration_cache: dict[str, float] = {}

def video_duration(video_key: str, video_path: str | Path | None = None) -> float:
    ck = f"path::{video_path}" if video_path else f"key::{video_key}"
    if ck in _duration_cache:
        return _duration_cache[ck]

    dur = 0.0
    p = Path(video_path) if video_path else None
    if p is not None and p.is_file():
        try:
            dur = float(_core()["get_video_info"](str(p))["duration"])
        except Exception:
            dur = 0.0
    else:
        try:
            mdir = active_memory_dir(video_key)
            macros = json.loads((mdir / "01_macros.json").read_text(encoding="utf-8"))
            dur = max(float(m["time_range"][1]) for m in macros if m.get("time_range"))
        except Exception:
            try:
                mem = Memory(video_key)
                if mem.video_path and Path(mem.video_path).exists():
                    dur = float(_core()["get_video_info"](mem.video_path)["duration"])
            except Exception:
                dur = 0.0
    _duration_cache[ck] = dur
    return dur

MIN_FRAMES = 2
B64_BYTES_PER_PIXEL = 0.35

def read_video_once(
    video_path: str,
    start_sec: float = 0.0,
    end_sec: float | None = None,
    fps: float = 0.0,
    budget: str = "normal",
    max_frames: int = 0,
    stats: dict | None = None,
) -> list[tuple[float, str]]:
    api = _core()
    info = api["get_video_info"](video_path)
    duration = float(info["duration"])
    if duration <= 0:
        return []

    max_pixels = api["budget_to_pixels"](budget, api["VIDEO_BUDGET_TOKENS"])
    cap_frames = min(max_frames or api["MAX_TOTAL_FRAMES"], api["MAX_TOTAL_FRAMES"])

    start_sec = max(0.0, float(start_sec or 0.0))
    if start_sec >= duration:
        return []
    if end_sec is not None:
        end_sec = min(float(end_sec), duration)
        if end_sec <= start_sec:
            return []
    segment_duration = (end_sec if end_sec is not None else duration) - start_sec

    target_h, target_w = api["smart_resize"](
        info["height"], info["width"], api["VIDEO_MIN_PIXELS"], max_pixels
    )

    if fps and fps > 0:
        eff_fps = min(float(fps), float(info["native_fps"]))
        nframes = int(segment_duration * eff_fps)
        nframes = max(MIN_FRAMES, min(cap_frames, nframes))
    else:
        eff_fps, nframes = api["compute_dynamic_fps"](
            segment_duration, info["native_fps"], MIN_FRAMES, cap_frames, api["DEFAULT_FPS"]
        )

    sample = api["extract_frames_by_seeking"](
        video_path, [start_sec + segment_duration * 0.5], target_h, target_w
    )
    per_frame = len(sample[0][1]) if sample else int(target_h * target_w * B64_BYTES_PER_PIXEL)
    max_safe = max(MIN_FRAMES, int(api["MAX_RESPONSE_BYTES"]) // max(1, per_frame))
    nframes = min(nframes, max_safe)

    seg_end = end_sec if end_sec is not None else duration
    seek_end = min(seg_end, duration - 1.0 / float(info["native_fps"]))
    span = seek_end - start_sec
    if nframes <= 1 or span <= 0:
        timestamps = [start_sec]
    else:
        step = span / (nframes - 1)
        timestamps = [start_sec + i * step for i in range(nframes)]
    frames = api["extract_frames_by_seeking"](video_path, timestamps, target_h, target_w)

    truncated = False
    total = sum(len(b) for _, b in frames)
    while total > api["MAX_RESPONSE_BYTES"] and len(frames) > MIN_FRAMES:
        truncated = True
        keep = max(MIN_FRAMES, int(len(frames) * api["MAX_RESPONSE_BYTES"] / total))
        step2 = (len(frames) - 1) / (keep - 1) if keep > 1 else 0
        frames = [frames[round(i * step2)] for i in range(keep)]
        total = sum(len(b) for _, b in frames)

    if stats is not None:
        actual_fps = len(frames) / segment_duration if segment_duration > 0 else eff_fps
        stats.update(
            n_frames=stats.get("n_frames", 0) + len(frames),
            fps_effective=round(actual_fps, 4),
            fps_requested=fps,
            sent_mb=round(stats.get("sent_mb", 0.0) + total / 1024 / 1024, 3),
            truncated_by_bytes=stats.get("truncated_by_bytes", False) or truncated,
            bytes_per_frame=per_frame,
            frame_wh=[target_w, target_h],
            native_fps=round(float(info["native_fps"]), 2),
            source_note=(
                f"Source: {duration:.1f}s | {info['width']}x{info['height']} (WxH) | "
                f"{info['native_fps']:.1f} fps native"
                + (f" | rotation {int(info.get('rotation') or 0)}°" if info.get("rotation") else "")
            ),
        )
    return frames

def skim_frames(
    video_path: str,
    duration: float,
    chunk_sec: float = OFFICIAL_SKIM_CHUNK_SEC,
    fps: float = OFFICIAL_SKIM_FPS,
    budget: str = OFFICIAL_SKIM_BUDGET,
    max_chunks: int = 0,
    windows: list[list[float]] | None = None,
    stats: dict | None = None,
) -> list[tuple[str, str]]:
    api = _core()
    if duration <= 0:
        return []

    if windows:
        chunks = [[max(0.0, float(a)), min(duration, float(b))]
                  for a, b in windows if float(b) > float(a)]
    else:
        n = max(1, math.ceil(duration / chunk_sec))
        chunks = [[i * chunk_sec, min((i + 1) * chunk_sec, duration)] for i in range(n)]
    if max_chunks > 0:
        chunks = chunks[:max_chunks]
    if not chunks:
        return []

    st = stats if stats is not None else {}
    st.setdefault("n_frames", 0)
    st.setdefault("sent_mb", 0.0)
    st["n_chunks"] = len(chunks)

    span_total = sum(e - s for s, e in chunks)
    total_cap = int(api["MAX_TOTAL_FRAMES"])
    total_bytes = int(api["MAX_RESPONSE_BYTES"])

    out: list[tuple[float, str]] = []
    for s, e in chunks:
        share = (e - s) / span_total if span_total > 0 else 1.0 / len(chunks)
        chunk_cap = max(MIN_FRAMES, int(total_cap * share))
        chunk_bytes = max(1, int(total_bytes * share))
        cst: dict = {}
        got = read_video_once(video_path, s, e, fps=fps, budget=budget,
                              max_frames=chunk_cap, stats=cst)
        tot = sum(len(b) for _, b in got)
        while tot > chunk_bytes and len(got) > MIN_FRAMES:
            st["truncated_by_bytes"] = True
            keep = max(MIN_FRAMES, int(len(got) * chunk_bytes / tot))
            step = (len(got) - 1) / (keep - 1) if keep > 1 else 0
            got = [got[round(i * step)] for i in range(keep)]
            tot = sum(len(b) for _, b in got)
        out += got
        st["n_frames"] = st.get("n_frames", 0) + len(got)
        st["sent_mb"] = round(st.get("sent_mb", 0.0) + tot / 1024 / 1024, 3)
        st["truncated_by_bytes"] = st.get("truncated_by_bytes", False) or cst.get("truncated_by_bytes", False)
        for k in ("bytes_per_frame", "frame_wh", "native_fps", "source_note"):
            if k in cst:
                st[k] = cst[k]

    gate_cap = GATEWAY_MAX_FRAMES
    if gate_cap > 0 and len(out) > gate_cap:
        step = (len(out) - 1) / (gate_cap - 1) if gate_cap > 1 else 0
        kept = [out[round(i * step)] for i in range(gate_cap)]
        st["gateway_clipped_from"] = len(out)
        st["sent_mb"] = round(sum(len(b) for _, b in kept) / 1024 / 1024, 3)
        out = kept

    span_total = sum(e - s for s, e in chunks)
    st["fps_effective"] = round(len(out) / span_total, 4) if span_total > 0 else 0.0
    st["fps_requested"] = fps
    st["sampled_note"] = (
        f"Sampled: {len(out)} frames @ {st['fps_effective']:.2f} fps | "
        f"{st.get('frame_wh', ['?', '?'])[0]}x{st.get('frame_wh', ['?', '?'])[1]} (WxH) per frame"
    )

    last_ts = out[-1][0] if out else duration
    return [(f"data:image/jpeg;base64,{b64}", api["format_timestamp"](ts_, last_ts))
            for ts_, b64 in out]

ARMS = ("retrieve", "core_skim", "both", "core_chunked")

def _solve_core_chunked(q: dict, g: dict, cfg: dict, arm: str, mem, dur: float,
                        stats: dict, t0: float) -> dict:
    q["_exec_path"] = "core_arm._solve_core_chunked"
    import core_chunked_qa as cq

    qa_cfg = cq.ChunkedQAConfig.from_genome(g)
    route = cq.should_use_chunked_read(dur)
    stats["chunked_route"] = route
    if route != cq.ROUTE_CHUNKED:
        return None

    def raw_model_fn(prompt: str, frame_items=None, role: str = "solver") -> str:
        return solve.call_llm(prompt, cfg, g["answer"]["max_tokens"],
                              g["answer"]["temperature"], role=role,
                              frame_items=frame_items,
                              frame_layout=FRAME_LAYOUT if frame_items else "trailing")

    model_fn = cq.with_frame_cap_retry(raw_model_fn, qa_cfg)

    try:
        res = cq.answer_video_question(
            q["question"], mem.video_path, options=q.get("options") or [],
            qtype=str(q.get("type") or ""), cfg=qa_cfg, model_fn=model_fn,
            duration=dur)
    except ContentFilteredError as exc:
        stats["content_filtered"] = str(exc)[:160]
        return _record(q, arm, None, "", dur, stats, err="content_filtered",
                       latency=time.time() - t0)
    except Exception as exc:
        return _record(q, arm, None, "", dur, stats,
                       err=f"chunked_qa failed: {str(exc)[:200]}",
                       latency=time.time() - t0)

    stats.update(
        chunked=True,
        n_chunks=len(res.get("windows") or []),
        n_detail_spans=len(res.get("detail_spans") or []),
        n_model_calls=res.get("n_model_calls"),
        n_read_video_calls=res.get("n_read_video_calls"),
        n_final_calls=res.get("n_final_calls"),
        n_frames=res.get("frames_sent_total"),
        coarse_failed=(res.get("coarse_stats") or {}).get("failed_chunks"),
        coarse_retried=(res.get("coarse_stats") or {}).get("retried_chunks"),
        detail_failed=(res.get("detail_stats") or {}).get("failed_spans"),
        counting=res.get("counting"),
        event_count=res.get("event_count"),
        prompt_variant="chunked_qa:" + qa_cfg.final_prompt_file,
    )
    c_err = (res.get("coarse_stats") or {}).get("errors") or {}
    d_err = (res.get("detail_stats") or {}).get("errors") or {}
    if c_err:
        stats["coarse_errors"] = {str(k): str(v)[:160] for k, v in c_err.items()}
    if d_err:
        stats["detail_errors"] = {str(k): str(v)[:160] for k, v in d_err.items()}
    span = float(dur) or 1.0
    stats["fps_effective"] = round((res.get("frames_sent_total") or 0) / span, 4)

    text = res.get("final_text") or ""
    if not text:
        return _record(q, arm, None, "", dur, stats,
                       err=res.get("error") or "no final answer",
                       latency=time.time() - t0)
    pred = solve.parse_choice(text, q.get("n_options", 4))
    return _record(q, arm, pred, text, dur, stats, latency=time.time() - t0)

def _solve_chunked(q: dict, g: dict, cfg: dict, arm: str, mem, dur: float, fr: dict,
                   chunks: list[list[float]], evidence: str, ids: list,
                   stats: dict, t0: float) -> dict:
    q["_exec_path"] = "core_arm._solve_chunked"
    api = _core()
    fmt = api["format_timestamp"]
    n = len(chunks)

    def _observe(idx: int, s: float, e: float) -> dict:
        span = f"{fmt(s, dur)}-{fmt(e, dur)}"
        cst: dict = {}
        try:
            frames_c = chunk_frames(mem.video_path, dur, s, e,
                                    fps=float(fr.get("fps", OFFICIAL_SKIM_FPS)),
                                    budget=str(fr.get("budget", OFFICIAL_SKIM_BUDGET)),
                                    stats=cst)
        except Exception as exc:
            return {"i": idx, "span": span, "failed": True, "frames": 0, "cst": cst,
                    "obs": f"[segment {idx}/{n} {span}] EXTRACTION FAILED: {str(exc)[:120]}"}
        if not frames_c:
            return {"i": idx, "span": span, "failed": True, "frames": 0, "cst": cst,
                    "obs": f"[segment {idx}/{n} {span}] NOTHING RELEVANT IN THIS SEGMENT "
                           f"(zero frames extracted)"}

        prompt = build_chunk_observe_prompt(q, g, idx, n, span,
                                            cst.get("sampled_note", ""))
        try:
            txt = solve.call_llm(prompt, cfg, g["answer"]["max_tokens"],
                                 g["answer"]["temperature"], role="solver",
                                 frame_items=frames_c, frame_layout=FRAME_LAYOUT)
        except ContentFilteredError as exc:
            return {"i": idx, "span": span, "failed": True, "frames": len(frames_c),
                    "cst": cst, "content_filtered": str(exc)[:160],
                    "obs": f"[segment {idx}/{n} {span}] BLOCKED BY CONTENT FILTER"}
        except Exception as exc:
            return {"i": idx, "span": span, "failed": True, "frames": len(frames_c),
                    "cst": cst,
                    "obs": f"[segment {idx}/{n} {span}] CALL FAILED: {str(exc)[:120]}"}
        return {"i": idx, "span": span, "failed": False, "frames": len(frames_c),
                "cst": cst, "called": True,
                "obs": f"[segment {idx}/{n} {span}]\n{(txt or '').strip()[:CHUNK_OBS_MAX_CHARS]}"}

    cw = _chunk_workers()
    if cw > 1 and n > 1:
        with ThreadPoolExecutor(max_workers=min(cw, n)) as pool:
            out = list(pool.map(lambda t: _observe(t[0], t[1][0], t[1][1]),
                                list(enumerate(chunks, start=1))))
        out.sort(key=lambda r: r["i"])
    else:
        out = [_observe(i, s, e) for i, (s, e) in enumerate(chunks, start=1)]

    obs_parts: list[str] = []
    per_chunk_frames: list[int] = []
    n_frames_tot = 0
    sent_mb_tot = 0.0
    failed = 0
    clipped = False
    calls = 0
    for r in out:
        obs_parts.append(r["obs"])
        if r["failed"]:
            failed += 1
            if r.get("content_filtered"):
                stats["content_filtered"] = r["content_filtered"]
            per_chunk_frames.append(0 if r["frames"] == 0 else r["frames"])
            continue
        per_chunk_frames.append(r["frames"])
        n_frames_tot += r["frames"]
        cst = r["cst"]
        sent_mb_tot += float(cst.get("sent_mb", 0.0))
        clipped = clipped or ("gateway_clipped_from" in cst)
        for k in ("bytes_per_frame", "frame_wh", "native_fps", "source_note"):
            if k in cst:
                stats.setdefault(k, cst[k])
        if r.get("called"):
            calls += 1

    span_total = sum(e - s for s, e in chunks)
    stats.update(
        chunked=True,
        n_chunks=n,
        chunks_failed=failed,
        chunk_frames_each=per_chunk_frames,
        n_frames=n_frames_tot,
        sent_mb=round(sent_mb_tot, 3),
        fps_requested=float(fr.get("fps", OFFICIAL_SKIM_FPS)),
        fps_effective=round(n_frames_tot / span_total, 4) if span_total > 0 else 0.0,
        gateway_clipped_any=clipped,
        sampled_note=(f"Sampled: {n_frames_tot} frames across {n} segments @ "
                      f"{(n_frames_tot / span_total if span_total > 0 else 0):.2f} fps"),
    )

    if failed >= n:
        stats["n_model_calls"] = calls
        return _record(q, arm, None, "", dur, stats, ids=ids,
                       err="all segments failed", latency=time.time() - t0)

    prompt = build_chunk_synth_prompt(q, g, n, "\n\n".join(obs_parts), evidence=evidence)
    stats["prompt_variant"] = "chunked:" + (
        (g.get("answer", {}) or {}).get("core_chunk_synth_file") or CORE_CHUNK_SYNTH_FILE)
    stats["obs_chars"] = sum(len(p) for p in obs_parts)

    text, err = "", None
    try:
        text = solve.call_llm(prompt, cfg, g["answer"]["max_tokens"],
                              g["answer"]["temperature"], role="solver")
        calls += 1
    except ContentFilteredError as exc:
        stats["content_filtered"] = str(exc)[:160]
        err = "content_filtered"
    except Exception as exc:
        err = str(exc)[:200]

    stats["n_model_calls"] = calls
    pred = solve.parse_choice(text, q.get("n_options", 4)) if text else None
    return _record(q, arm, pred, text, dur, stats, err=err, ids=ids,
                   latency=time.time() - t0)

def solve_one_core(q: dict, g: dict, cfg: dict, arm: str, tk=None) -> dict:
    t0 = time.time()
    stats: dict = {}
    evidence, ids = "", []

    if arm == "both":
        probe: dict = {}
        evidence, ids = solve.build_evidence(q, g, "retrieve", tk, probe=probe)

    vpath = resolve_video_path(q)
    if not vpath:
        return _record(q, arm, None, "", 0.0, stats,
                       err=f"video unavailable: relpath={q.get('video_relpath')!r} "
                           f"key={q.get('video_key')!r}")
    mem = VideoFile(vpath)
    q["_video_path"] = str(vpath)

    dur = video_duration(q.get("video_key") or "", vpath)
    fr = (g.get("evidence", {}).get("frames") or {}) if isinstance(g.get("evidence"), dict) else {}

    if arm == "core_chunked":
        got = _solve_core_chunked(q, g, cfg, arm, mem, dur, stats, t0)
        if got is not None:
            return got

    if fr.get("chunked_calls"):
        chunks = plan_chunks(dur, chunk_sec=float(fr.get("chunk_sec", OFFICIAL_SKIM_CHUNK_SEC)),
                             max_chunks=int(fr.get("max_chunks", 0)))
        if len(chunks) > 1:
            return _solve_chunked(q, g, cfg, arm, mem, dur, fr, chunks,
                                  evidence, ids, stats, t0)
        stats["chunked_skipped"] = "single_chunk"

    try:
        frames = skim_frames(
            mem.video_path, dur,
            chunk_sec=float(fr.get("chunk_sec", OFFICIAL_SKIM_CHUNK_SEC)),
            fps=float(fr.get("fps", OFFICIAL_SKIM_FPS)),
            budget=str(fr.get("budget", OFFICIAL_SKIM_BUDGET)),
            max_chunks=int(fr.get("max_chunks", 0)),
            stats=stats,
        )
    except Exception as exc:
        return _record(q, arm, None, "", dur, stats, err=f"extract failed: {str(exc)[:200]}")
    if not frames:
        return _record(q, arm, None, "", dur, stats, err="zero frames extracted")

    note = (f"Video: {mem.video_path}\n"
            f"{stats.get('source_note', '')}\n"
            f"{stats.get('sampled_note', '')}")
    if not fr.get("no_media_info"):
        mi = media_info(mem.video_path)
        if mi:
            note += "\n" + mi
            stats["media_info_chars"] = len(mi)

    if fr.get("legacy_prompt"):
        prompt = solve.build_prompt(q, (evidence + "\n" + note) if evidence else note, g)
        stats["prompt_variant"] = "legacy_solve_v1"
    else:
        prompt = build_core_prompt(q, note, g, evidence=evidence)
        stats["prompt_variant"] = (g.get("answer", {}) or {}).get(
            "core_prompt_file") or CORE_PROMPT_FILE

    text, err = "", None
    try:
        text = solve.call_llm(
            prompt, cfg, g["answer"]["max_tokens"], g["answer"]["temperature"],
            role="solver", frame_items=frames, frame_layout=FRAME_LAYOUT,
        )
    except ContentFilteredError as exc:
        stats["content_filtered"] = str(exc)[:160]
        if arm == "both" and evidence:
            try:
                text = solve.call_llm(prompt, cfg, g["answer"]["max_tokens"],
                                      g["answer"]["temperature"], role="solver")
            except Exception as exc2:
                err = str(exc2)[:200]
        else:
            err = "content_filtered"
    except Exception as exc:
        err = str(exc)[:200]

    pred = solve.parse_choice(text, q.get("n_options", 4)) if text else None
    return _record(q, arm, pred, text, dur, stats, err=err, ids=ids,
                   latency=time.time() - t0)

def _record(q: dict, arm: str, pred, text: str, dur: float, stats: dict,
            err: str | None = None, ids: list | None = None,
            latency: float = 0.0) -> dict:
    measurable = arm == "both"
    return {
        "qid": q["qid"],
        "video_key": q["video_key"],
        "video_relpath": q.get("video_relpath"),
        "video_path": q.get("_video_path"),
        "type": q["type"],
        "split": q.get("split"),
        "route": q.get("route"),
        "video_duration_sec": q.get("video_duration_sec"),
        "exec_arm": "core",
        "exec_path": q.get("_exec_path") or f"core_arm.solve_one_core(arm={arm})",
        "exec_video": q.get("_video_path"),
        "exec_memory_dir": None,
        "arm": arm,
        "mode": "retrieve" if arm == "both" else "core_frames",
        "gold": q["answer"],
        "pred": pred,
        "correct": bool(pred == q["answer"]),
        "err": err,
        "duration_sec": round(dur, 1),
        "duration_bucket": bucket_of(dur),
        "located_macro_ids": list(ids or []),
        "anchor_measurable": measurable,
        "gold_macro_hit": None if not measurable else None,
        "strict_correct": None,
        "latency_s": round(latency, 2),
        "answer_text": (text or "")[:400],
        **{f"frames_{k}": v for k, v in stats.items()},
    }

BUCKETS = ((0, 1), (1, 5), (5, 10), (10, 20), (20, 30), (30, 10**6))

def bucket_of(dur_sec: float) -> str:
    m = dur_sec / 60
    for lo, hi in BUCKETS:
        if lo <= m < hi:
            return f"{lo}-{hi}min" if hi < 10**6 else f"{lo}min+"
    return "unknown"

def stratified(questions: list[dict], per_bucket: int, seed: int = 20260917,
               types: list[str] | None = None) -> list[dict]:
    import random

    rnd = random.Random(seed)
    pool = [q for q in questions if not types or q.get("type") in types]
    by: dict[str, list[dict]] = defaultdict(list)
    for q in pool:
        d = video_duration(q.get("video_key") or "", resolve_video_path(q))
        if d > 0:
            by[bucket_of(d)].append(q)
    out: list[dict] = []
    for name in [f"{lo}-{hi}min" if hi < 10**6 else f"{lo}min+" for lo, hi in BUCKETS]:
        grp = by.get(name) or []
        rnd.shuffle(grp)
        got = grp[:per_bucket]
        out += got
        flag = "" if len(got) == per_bucket else f"  [too few, the pool holds only {len(grp)}]"
    return out

def per_video(questions: list[dict], per_video_n: int, seed: int = 20260917,
              types: list[str] | None = None) -> list[dict]:
    import random

    rnd = random.Random(seed)
    pool = [q for q in questions if not types or q.get("type") in types]
    by: dict[str, list[dict]] = defaultdict(list)
    for q in pool:
        by[q.get("video_relpath") or q["video_key"] or q["qid"]].append(q)
    out: list[dict] = []
    rows: list[tuple[float, str, int]] = []
    for vk, grp in by.items():
        d = video_duration(grp[0].get("video_key") or "", resolve_video_path(grp[0]))
        if d <= 0:
            continue
        rnd.shuffle(grp)
        got = grp[:per_video_n]
        out += got
        rows.append((d / 60, vk, len(got)))
    for m, vk, n in sorted(rows):
        pass
    return out

def summarize_arm(recs: list[dict]) -> dict:
    ok = [r for r in recs if not r.get("err")]
    n = len(ok) or 1
    by_bucket: dict[str, list[dict]] = defaultdict(list)
    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in ok:
        by_bucket[r["duration_bucket"]].append(r)
        by_type[r["type"]].append(r)

    def acc(rs):
        return round(sum(bool(x["correct"]) for x in rs) / max(1, len(rs)), 4)

    return {
        "n": len(recs),
        "n_scored": len(ok),
        "n_error": len(recs) - len(ok),
        "acc": acc(ok),
        "mean_frames": round(sum(r.get("frames_n_frames", 0) for r in ok) / n, 1),
        "mean_sent_mb": round(sum(r.get("frames_sent_mb", 0) for r in ok) / n, 3),
        "mean_fps_effective": round(sum(r.get("frames_fps_effective", 0) for r in ok) / n, 3),
        "n_truncated_by_bytes": sum(1 for r in ok if r.get("frames_truncated_by_bytes")),
        "by_bucket": {k: {"n": len(v), "acc": acc(v)} for k, v in sorted(by_bucket.items())},
        "by_type": {k: {"n": len(v), "acc": acc(v)} for k, v in sorted(by_type.items())},
    }

def run_probe(args: argparse.Namespace) -> dict:
    global _bench_root
    if getattr(args, "benchmark_root", None):
        _bench_root = Path(args.benchmark_root)
    g, cfg = load_genome(), load_cfg()

    if getattr(args, "chunked_calls", False) or getattr(args, "legacy_prompt", False):
        ev = g.setdefault("evidence", {})
        if not isinstance(ev, dict):
            ev = g["evidence"] = {}
        frames_cfg = ev.setdefault("frames", {})
        if args.chunked_calls:
            frames_cfg["chunked_calls"] = True
        if args.legacy_prompt:
            frames_cfg["legacy_prompt"] = True

    qs = load_questions(Path(args.questions), args.split)
    if not qs:
        raise SystemExit(f"question pool is empty: {args.questions} (split={args.split})")

    if args.routes:
        want = {r.strip() for r in args.routes.split(",") if r.strip()}
        bad = want - set(video_routing.ROUTES)
        if bad:
            raise SystemExit(f"unknown route {sorted(bad)}; available {video_routing.ROUTES}")
        has_route = sum(1 for q in qs if q.get("route"))
        if not has_route:
            pass
        else:
            before = len(qs)
            kept = [q for q in qs if (q.get("route") or "") in want]
            dropped = Counter((q.get("route") or "unrouted") for q in qs
                              if (q.get("route") or "") not in want)
            qs = kept
            if not qs:
                raise SystemExit(
                    f"the pool is empty after route filtering. The route distribution of these items is {dict(dropped)}; "
                    f"to run video_memory items use harness/arm_runner.py or harness/solve.py")

    types = args.types.split(",") if args.types else None
    if args.sample_by == "video":
        sample = per_video(qs, args.per_video, args.seed, types=types)
    else:
        sample = stratified(qs, args.per_bucket, args.seed, types=types)
    if not sample:
        raise SystemExit("the sample is empty: possibly every video lacks a memory (duration unreadable)")
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in arms:
        if a not in ARMS:
            raise SystemExit(f"unknown arm {a}; available {ARMS}")

    est = sum(1 for _ in sample) * sum(1 for a in arms if a != "retrieve")
    if args.dry_run:
        return {"sample": len(sample), "arms": arms}

    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    reset_cost()
    results: dict[str, list[dict]] = {}

    for arm in arms:
        t0 = time.time()
        if arm == "retrieve":
            recs = solve.run(sample, g, cfg, "retrieve", args.workers, quiet=args.quiet)
            s = solve.summarize(recs)
            qmap = {q["qid"]: q for q in sample}
            for r in recs:
                qq = qmap.get(r["qid"], {})
                d = video_duration(r["video_key"], resolve_video_path(qq))
                r["arm"], r["duration_sec"], r["duration_bucket"] = arm, round(d, 1), bucket_of(d)
            results[arm] = recs
        else:
            def one(q):
                tk = solve.load_toolkit(q["video_key"]) if arm == "both" else None
                try:
                    return solve_one_core(q, g, cfg, arm, tk)
                except Exception as exc:
                    return _record(q, arm, None, "", 0.0, {}, err=f"unhandled: {str(exc)[:200]}")

            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                recs = list(pool.map(one, sample))
            results[arm] = recs
            s = summarize_arm(recs)
            if s["n_truncated_by_bytes"]:
                pass

        with open(args.out, "a", encoding="utf-8") as f:
            for r in results[arm]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    report = _compare(results, sample)
    rp = Path(args.out).with_suffix(".report.json")
    rp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report

def _compare(results: dict[str, list[dict]], sample: list[dict]) -> dict:
    per_arm = {a: summarize_arm(rs) if a != "retrieve" else _summarize_retrieve(rs)
               for a, rs in results.items()}
    base = results.get("retrieve")
    flips: dict = {}
    if base:
        bmap = {r["qid"]: bool(r["correct"]) for r in base}
        for arm, rs in results.items():
            if arm == "retrieve":
                continue
            fixed = [r["qid"] for r in rs if not bmap.get(r["qid"]) and r["correct"]]
            broken = [r["qid"] for r in rs if bmap.get(r["qid"]) and not r["correct"]]
            per_bucket: dict[str, dict] = defaultdict(lambda: {"fixed": 0, "broken": 0, "n": 0})
            for r in rs:
                b = per_bucket[r["duration_bucket"]]
                b["n"] += 1
                if not bmap.get(r["qid"]) and r["correct"]:
                    b["fixed"] += 1
                if bmap.get(r["qid"]) and not r["correct"]:
                    b["broken"] += 1
            flips[arm] = {
                "n_fixed": len(fixed), "n_broken": len(broken),
                "net": len(fixed) - len(broken),
                "fixed_qids": fixed, "broken_qids": broken,
                "by_bucket": {k: dict(v) for k, v in sorted(per_bucket.items())},
            }
    return {
        "n_questions": len(sample),
        "arms": per_arm,
        "paired_vs_retrieve": flips,
        "caveat": ("small-sample probe; it makes no acceptance decision. gold_macro_hit and strict_acc are "
                   "physically unmeasurable on the core_skim arm (there is no retrieval), so only acc is compared."),
    }

def _summarize_retrieve(recs: list[dict]) -> dict:
    s = solve.summarize(recs)
    if any(r.get("anchor_measurable") is False for r in recs) or all(
            not (r.get("anchor") or {}).get("macro_id") for r in recs if isinstance(r.get("anchor"), dict)):
        s["anchor_rate"] = None
        s["gold_macro_hit_rate"] = None
        s["strict_acc"] = None
        s["anchor_measurable"] = False
    by_bucket: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        by_bucket[r.get("duration_bucket", "unknown")].append(r)
    s["by_bucket"] = {
        k: {"n": len(v),
            "acc": round(sum(bool(x["correct"]) for x in v) / max(1, len(v)), 4)}
        for k, v in sorted(by_bucket.items())
    }
    return s

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true", help="run the stratified probe")
    ap.add_argument("--questions", default=None)
    ap.add_argument("--split", default=None,
                    help="filter by split; without it no filtering is applied. Self-built pools use heldout, "
                         "whereas questions converted from a benchmark carry split=external; the wrong value yields 0 questions")
    ap.add_argument("--sample-by", choices=["video", "bucket"], default="video",
                    help="video = a fixed number of items per video (default; with only 36 videos of similar length, "
                         "stratifying by tier would put two tiers on a single video); bucket = stratify by duration tier")
    ap.add_argument("--per-video", type=int, default=1, help="with --sample-by video, how many per video")
    ap.add_argument("--per-bucket", type=int, default=6, help="with --sample-by bucket, how many per tier")
    ap.add_argument("--arms", default="retrieve,core_skim,both")
    ap.add_argument("--routes", default="direct_read_video,chunked_read_video",
                    help="run only items with these routes (comma separated). Default is the two tiers core owns"
                         " (<30min); video_memory items at >=30min are handled by harness/arm_runner.py. "
                         "Pass an empty string to disable filtering (long videos would then be forced through sparse whole-video sampling, which is not the official pipeline)")
    ap.add_argument("--types", default=None, help="restrict to these question types, comma separated")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20260917)
    ap.add_argument("--out", default=None)
    ap.add_argument("--quiet", action="store_true", default=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--benchmark-root", default=str(_bench_root),
                    help="root directory of the public benchmark videos. The core arm joins the question's video_relpath "
                         "against it to form an absolute path, so short videos (<=30min) can be run in full without building memories")
    ap.add_argument("--chunked-calls", action="store_true",
                    help="chunked answering: one model call per five-minute chunk to view frames, then one final synthesis. "
                         "Videos shorter than chunk_sec fall back to a single call automatically. The call count becomes "
                         "chunks+1, but frame density for videos over five minutes rises several fold")
    ap.add_argument("--legacy-prompt", action="store_true",
                    help="fall back to the solve_v1.txt prompt (the earlier behaviour), for before-and-after comparison")
    args = ap.parse_args()

    if not args.questions:
        args.questions = str(solve.STORE / "questions_verified.jsonl")
    if not args.out:
        ts = time.strftime("%m%d_%H%M%S")
        args.out = str(ROOT / f"experiments/core_probe_{ts}.jsonl")

    if not args.probe:
        raise SystemExit("--probe is currently the only available action")
    run_probe(args)

if __name__ == "__main__":
    main()
