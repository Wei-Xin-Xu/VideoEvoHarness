from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

VEH_DIR = Path(__file__).resolve().parent
ROOT = VEH_DIR.parent

CHUNKED_MIN_DURATION_SEC = 300.0
CHUNKED_MAX_DURATION_SEC = 1800.0

ROUTE_DIRECT = "direct_read_video"
ROUTE_CHUNKED = "chunked_read_video"
ROUTE_VIDEO_MEMORY = "video_memory"


@dataclass
class ChunkedQAConfig:

    chunk_seconds: float = 300.0
    coarse_fps: float = 1.0
    coarse_budget: str = "normal"
    detail_fps: float = 2.0
    detail_budget: str = "large"
    span_context_sec: float = 5.0
    span_merge_gap_sec: float = 2.0
    max_detail_spans: int = 6
    min_detail_span_sec: float = 2.0
    max_detail_span_sec: float = 60.0
    dedup_gap_sec: float = 1.5
    dedup_similarity: float = 0.5
    max_chunk_retries: int = 1
    max_frames_per_model_call: int = int(os.environ.get("VEH_CORE_MAX_FRAMES", "192"))
    max_tokens: int = 2048
    temperature: float = 0.1
    coarse_prompt_file: str = "prompts/core_coarse_evidence_v1.txt"
    detail_prompt_file: str = "prompts/core_detail_evidence_v1.txt"
    final_prompt_file: str = "prompts/core_final_answer_v1.txt"

    @classmethod
    def from_genome(cls, g: dict | None) -> "ChunkedQAConfig":
        cfg = cls()
        if not isinstance(g, dict):
            return cfg
        node = ((g.get("evidence") or {}).get("chunked_qa") or {}) if isinstance(g.get("evidence"), dict) else {}
        for k, v in node.items():
            if k.startswith("_") or not hasattr(cfg, k):
                continue
            setattr(cfg, k, v)
        return cfg


_core_api: dict[str, Any] = {}

_core_lock = threading.Lock()


def _load_by_path(mod_name: str, path: Path):
    import importlib.util

    got = sys.modules.get(mod_name)
    if got is not None:
        return got
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {mod_name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules[mod_name] = mod
    return mod


def _core() -> dict[str, Any]:
    if _core_api:
        return _core_api
    with _core_lock:
        if _core_api:
            return _core_api
        return _load_core_locked()


def _load_core_locked() -> dict[str, Any]:
    src = ROOT / "Qwen-MM-Plugins" / "src"
    readers = src / "capabilities" / "core" / "qwen_mm_plugins_core" / "readers"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    rv_mod = _load_by_path("_qmp_reader_video", readers / "video.py")
    mi_mod = _load_by_path("_qmp_reader_media_info", readers / "media_info.py")
    from shared.env import MAX_RESPONSE_BYTES, MAX_TOTAL_FRAMES
    from shared.video import format_timestamp, get_video_info, parse_time

    _core_api.update(
        read_video=rv_mod.handle,
        read_video_args=rv_mod.ReadVideoArgs,
        media_info=mi_mod.handle,
        parse_time=parse_time,
        format_timestamp=format_timestamp,
        get_video_info=get_video_info,
        MAX_TOTAL_FRAMES=MAX_TOTAL_FRAMES,
        MAX_RESPONSE_BYTES=MAX_RESPONSE_BYTES,
    )
    return _core_api


def probe_video_metadata(video_path: str, *, use_media_info: bool = True) -> dict:
    api = _core()
    if not video_path or not os.path.isfile(video_path):
        return {"ok": False, "error": f"file not found: {video_path}", "duration": 0.0}
    info = api["get_video_info"](video_path)
    out = {
        "ok": float(info.get("duration") or 0) > 0,
        "video_path": video_path,
        "duration": float(info.get("duration") or 0),
        "width": info.get("width"),
        "height": info.get("height"),
        "native_fps": info.get("native_fps"),
        "rotation": info.get("rotation") or 0,
    }
    if use_media_info:
        try:
            blocks = api["media_info"]({"path": video_path})
            out["media_info_text"] = "\n".join(
                b.get("text", "") for b in blocks if b.get("type") == "text"
            ).strip()
        except Exception as exc:
            out["media_info_error"] = str(exc)[:200]
    if not out["ok"]:
        out["error"] = "cannot determine video duration"
    return out


def should_use_chunked_read(duration: float) -> str:
    d = float(duration or 0)
    if d <= CHUNKED_MIN_DURATION_SEC:
        return ROUTE_DIRECT
    if d >= CHUNKED_MAX_DURATION_SEC:
        return ROUTE_VIDEO_MEMORY
    return ROUTE_CHUNKED


def build_video_windows(duration: float, chunk_seconds: float = 300.0) -> list[list[float]]:
    d = float(duration or 0)
    step = float(chunk_seconds or 0)
    if d <= 0 or step <= 0:
        return []
    windows: list[list[float]] = []
    start = 0.0
    while start < d:
        end = min(start + step, d)
        if end <= start:
            break
        windows.append([start, end])
        start = end
    return windows


class ReadVideoError(RuntimeError):
    pass


def _read_video_frames(video_path: str, start: float, end: float, *, fps: float,
                       budget: str, duration: float, cfg: ChunkedQAConfig,
                       read_video_fn: Callable[..., list[dict]] | None = None,
                       ) -> tuple[list[tuple[str, str]], str, dict]:
    api = _core()
    handle = read_video_fn or api["read_video"]
    args = {
        "video_path": video_path,
        "start_time": float(start),
        "end_time": float(end),
        "fps": float(fps),
        "budget": budget,
    }
    blocks = handle(args)
    if not blocks:
        raise ReadVideoError("read_video returned nothing")

    summary = ""
    frames: list[tuple[str, str]] = []
    pending: str | None = None
    for blk in blocks:
        btype = blk.get("type")
        if btype == "text":
            t = str(blk.get("text", ""))
            if t.startswith("Error:"):
                raise ReadVideoError(t[len("Error:"):].strip())
            if t.startswith("<") and t.endswith(">"):
                pending = t[1:-1]
            elif not summary:
                summary = t
        elif btype == "image":
            b64 = blk.get("data") or ""
            if b64:
                frames.append((pending or "", b64))
            pending = None

    if not frames:
        raise ReadVideoError("read_video returned zero frames")

    relabelled: list[tuple[str, str]] = []
    abs_secs: list[float] = []
    for label, b64 in frames:
        try:
            ts = float(api["parse_time"](label))
        except Exception:
            ts = float(start)
        ts = min(max(ts, 0.0), duration if duration > 0 else ts)
        abs_secs.append(ts)
        relabelled.append((f"data:image/jpeg;base64,{b64}",
                           api["format_timestamp"](ts, duration or ts)))

    stats: dict[str, Any] = {
        "n_frames_returned": len(relabelled),
        "abs_ts_first": round(abs_secs[0], 2) if abs_secs else None,
        "abs_ts_last": round(abs_secs[-1], 2) if abs_secs else None,
        "read_video_summary": summary,
    }

    cap = _frame_cap(cfg)
    if cap > 0 and len(relabelled) > cap:
        relabelled = _clip_frames(relabelled, cap)
        stats["model_call_clipped_from"] = stats["n_frames_returned"]

    span = float(end) - float(start)
    stats["n_frames_sent"] = len(relabelled)
    stats["fps_effective"] = round(len(relabelled) / span, 4) if span > 0 else 0.0
    note = (summary.strip() + "\n" if summary.strip() else "") + (
        f"Frames sent to model: {len(relabelled)} @ {stats['fps_effective']:.2f} fps "
        f"(requested fps={fps}, budget={budget})")
    return relabelled, note, stats


ModelFn = Callable[..., str]


_FRAME_CAP_SHRINK = 0.8
_FRAME_CAP_FLOOR = 24
_FRAME_CAP_MAX_SHRINKS = 6
_frame_cap_lock = threading.Lock()
_frame_cap_discovered: int | None = None


def _frame_cap(cfg: ChunkedQAConfig) -> int:
    base = int(cfg.max_frames_per_model_call or 0)
    with _frame_cap_lock:
        got = _frame_cap_discovered
    if got is None:
        return base
    return min(base, got) if base > 0 else got


def _shrink_frame_cap(n_sent: int) -> int:
    global _frame_cap_discovered
    with _frame_cap_lock:
        target = max(_FRAME_CAP_FLOOR, int(n_sent * _FRAME_CAP_SHRINK))
        if _frame_cap_discovered is None or target < _frame_cap_discovered:
            _frame_cap_discovered = target
        return _frame_cap_discovered


def _clip_frames(frames: list[tuple[str, str]], cap: int) -> list[tuple[str, str]]:
    if cap <= 0 or len(frames) <= cap:
        return frames
    if cap == 1:
        return [frames[0]]
    step = (len(frames) - 1) / (cap - 1)
    return [frames[round(i * step)] for i in range(cap)]


def with_frame_cap_retry(raw_call: ModelFn, qa_cfg: ChunkedQAConfig) -> ModelFn:
    from common import PayloadTooLargeError

    def _call(prompt: str, frame_items: list[tuple[str, str]] | None = None,
              role: str = "solver") -> str:
        frames = list(frame_items) if frame_items else frame_items
        last: Exception | None = None
        for _ in range(_FRAME_CAP_MAX_SHRINKS + 1):
            if frames:
                frames = _clip_frames(frames, _frame_cap(qa_cfg))
            try:
                return raw_call(prompt, frames, role)
            except PayloadTooLargeError as exc:
                last = exc
                if not frames or len(frames) <= _FRAME_CAP_FLOOR:
                    raise
                _shrink_frame_cap(len(frames))
        raise last if last else RuntimeError("frame cap retry exhausted")

    return _call


def default_model_fn(cfg_llm: dict, g: dict, qa_cfg: ChunkedQAConfig) -> ModelFn:
    import solve

    def _raw(prompt: str, frame_items: list[tuple[str, str]] | None = None,
             role: str = "solver") -> str:
        return solve.call_llm(prompt, cfg_llm, qa_cfg.max_tokens, qa_cfg.temperature,
                              role=role, frame_items=frame_items,
                              frame_layout="interleaved" if frame_items else "trailing")

    return with_frame_cap_retry(_raw, qa_cfg)


def _render(template_rel: str, mapping: dict[str, Any]) -> str:
    tpl = (VEH_DIR / template_rel).read_text(encoding="utf-8")
    for k, v in mapping.items():
        tpl = tpl.replace("{{" + k + "}}", str(v))
    return tpl


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.S)
_ANSWER_RE = re.compile(r"^\s*answer\s*:\s*[A-Z]\b", re.I | re.M)


def _extract_json(text: str) -> dict:
    if not text:
        raise ValueError("empty model output")
    m = _FENCE_RE.search(text)
    body = m.group(1) if m else text
    try:
        return json.loads(body)
    except Exception:
        pass
    start = body.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(body)):
            ch = body[i]
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
                    try:
                        return json.loads(body[start:i + 1])
                    except Exception:
                        break
        start = body.find("{", start + 1)
    raise ValueError(f"no JSON object in model output: {text[:200]!r}")


def _clamp(v: Any, lo: float, hi: float, default: float = 0.0) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        x = default
    return min(max(x, lo), hi)


def extract_chunk_evidence(question: str, duration: float, idx: int, n_chunks: int,
                           window: Sequence[float], frame_items: list[tuple[str, str]],
                           sampling: str, cfg: ChunkedQAConfig, model_fn: ModelFn,
                           ) -> dict:
    c_start, c_end = float(window[0]), float(window[1])
    prompt = _render(cfg.coarse_prompt_file, {
        "DURATION": f"{duration:.1f}",
        "CHUNK_INDEX": idx,
        "N_CHUNKS": n_chunks,
        "CHUNK_START": f"{c_start:.1f}",
        "CHUNK_END": f"{c_end:.1f}",
        "SAMPLING": sampling,
        "QUESTION": question,
    })
    raw = model_fn(prompt, frame_items, "coarse")
    ev = _normalise_chunk_evidence(_extract_json(raw), idx, c_start, c_end, duration)
    if _ANSWER_RE.search(raw or ""):
        ev["_contained_final_answer"] = True
    ev["_raw_chars"] = len(raw or "")
    return ev


def _normalise_chunk_evidence(d: dict, idx: int, c_start: float, c_end: float,
                              duration: float) -> dict:
    obs_out: list[dict] = []
    for o in (d.get("observations") or []):
        if not isinstance(o, dict):
            continue
        s = _clamp(o.get("start_sec"), c_start, c_end, c_start)
        e = _clamp(o.get("end_sec"), c_start, c_end, s)
        if e < s:
            s, e = e, s
        obs_out.append({
            "start_sec": round(s, 2),
            "end_sec": round(e, 2),
            "event": str(o.get("event", "")).strip(),
            "relation_to_question": str(o.get("relation_to_question", "")).strip(),
            "evidence_type": str(o.get("evidence_type", "visual")).strip() or "visual",
            "confidence": _clamp(o.get("confidence", 0.5), 0.0, 1.0, 0.5),
            "needs_detail": bool(o.get("needs_detail", False)),
        })
    spans_out: list[dict] = []
    for sp in (d.get("candidate_spans") or []):
        if not isinstance(sp, dict):
            continue
        s = _clamp(sp.get("start_sec"), 0.0, duration, c_start)
        e = _clamp(sp.get("end_sec"), 0.0, duration, s)
        if e < s:
            s, e = e, s
        if e <= s:
            continue
        spans_out.append({"start_sec": round(s, 2), "end_sec": round(e, 2),
                          "reason": str(sp.get("reason", "")).strip()})
    relevant = d.get("relevant")
    if relevant is None:
        relevant = bool(obs_out)
    return {
        "chunk_index": idx,
        "chunk_start_sec": round(c_start, 2),
        "chunk_end_sec": round(c_end, 2),
        "relevant": bool(relevant),
        "observations": obs_out,
        "candidate_spans": spans_out,
    }


def read_coarse_chunks(question: str, video_path: str, duration: float,
                       windows: list[list[float]], cfg: ChunkedQAConfig,
                       model_fn: ModelFn,
                       read_video_fn: Callable[..., list[dict]] | None = None,
                       ) -> tuple[list[dict], dict]:
    n = len(windows)
    done: dict[int, dict] = {}
    stats: dict[str, Any] = {"attempts": 0, "read_video_calls": 0, "model_calls": 0,
                             "failed_chunks": [], "retried_chunks": [],
                             "frames_sent_total": 0, "per_chunk_frames": {}}
    pending = list(range(n))
    for attempt in range(int(cfg.max_chunk_retries) + 1):
        if not pending:
            break
        if attempt > 0:
            stats["retried_chunks"].extend(pending)
        still: list[int] = []
        for i in pending:
            if i in done:
                continue
            s, e = windows[i]
            stats["attempts"] += 1
            try:
                frames, note, fstats = _read_video_frames(
                    video_path, s, e, fps=cfg.coarse_fps, budget=cfg.coarse_budget,
                    duration=duration, cfg=cfg, read_video_fn=read_video_fn)
                stats["read_video_calls"] += 1
                ev = extract_chunk_evidence(question, duration, i, n, windows[i],
                                            frames, note, cfg, model_fn)
                stats["model_calls"] += 1
                stats["frames_sent_total"] += fstats.get("n_frames_sent", 0)
                stats["per_chunk_frames"][i] = fstats.get("n_frames_sent", 0)
                ev["_read_stats"] = fstats
                done[i] = ev
            except Exception as exc:
                still.append(i)
                stats.setdefault("errors", {})[i] = str(exc)[:200]
        pending = still
    stats["failed_chunks"] = sorted(pending)
    stats["n_chunks_ok"] = len(done)
    return [done[i] for i in sorted(done)], stats


def collect_candidate_spans(coarse: list[dict], duration: float,
                            cfg: ChunkedQAConfig) -> list[dict]:
    out: list[dict] = []
    for ev in coarse:
        if not ev.get("relevant"):
            continue
        for sp in ev.get("candidate_spans") or []:
            out.append({"start_sec": sp["start_sec"], "end_sec": sp["end_sec"],
                        "reason": sp.get("reason", ""), "source": "candidate_span",
                        "chunk_index": ev["chunk_index"]})
        for o in ev.get("observations") or []:
            if o.get("needs_detail"):
                out.append({"start_sec": o["start_sec"], "end_sec": o["end_sec"],
                            "reason": f"needs_detail: {o.get('event', '')}",
                            "source": "needs_detail", "chunk_index": ev["chunk_index"]})
    for a, b in zip(coarse, coarse[1:]):
        bound = a.get("chunk_end_sec")
        if bound is None:
            continue
        tail = [o for o in (a.get("observations") or [])
                if abs(o["end_sec"] - bound) <= cfg.span_context_sec]
        head = [o for o in (b.get("observations") or [])
                if abs(o["start_sec"] - bound) <= cfg.span_context_sec]
        for o1 in tail:
            for o2 in head:
                if _similar(o1.get("event", ""), o2.get("event", ""), cfg.dedup_similarity):
                    out.append({
                        "start_sec": o1["start_sec"], "end_sec": o2["end_sec"],
                        "reason": "cross-chunk boundary: same event may be double counted",
                        "source": "boundary_conflict", "chunk_index": a["chunk_index"],
                    })
    for sp in out:
        sp["start_sec"] = round(_clamp(sp["start_sec"], 0.0, duration), 2)
        sp["end_sec"] = round(_clamp(sp["end_sec"], 0.0, duration), 2)
    return [sp for sp in out if sp["end_sec"] > sp["start_sec"]]


def merge_candidate_spans(spans: list[dict], duration: float,
                          cfg: ChunkedQAConfig) -> list[dict]:
    if not spans:
        return []
    pad = float(cfg.span_context_sec)
    padded = []
    for sp in spans:
        padded.append({
            "start_sec": _clamp(sp["start_sec"] - pad, 0.0, duration),
            "end_sec": _clamp(sp["end_sec"] + pad, 0.0, duration),
            "reasons": [sp.get("reason", "")] if sp.get("reason") else [],
            "sources": [sp.get("source", "")],
        })
    padded.sort(key=lambda x: (x["start_sec"], x["end_sec"]))

    merged: list[dict] = [padded[0]]
    for sp in padded[1:]:
        last = merged[-1]
        if sp["start_sec"] <= last["end_sec"] + cfg.span_merge_gap_sec:
            last["end_sec"] = max(last["end_sec"], sp["end_sec"])
            last["reasons"].extend(sp["reasons"])
            last["sources"].extend(sp["sources"])
        else:
            merged.append(sp)

    out: list[dict] = []
    for sp in merged:
        s, e = sp["start_sec"], sp["end_sec"]
        if e - s < cfg.min_detail_span_sec:
            e = _clamp(s + cfg.min_detail_span_sec, 0.0, duration)
            if e - s < cfg.min_detail_span_sec:
                s = _clamp(e - cfg.min_detail_span_sec, 0.0, duration)
        if e - s > cfg.max_detail_span_sec:
            e = _clamp(s + cfg.max_detail_span_sec, 0.0, duration)
        out.append({
            "start_sec": round(s, 2), "end_sec": round(e, 2),
            "reason": "; ".join(dict.fromkeys(r for r in sp["reasons"] if r))[:400],
            "sources": sorted(set(x for x in sp["sources"] if x)),
        })
    out.sort(key=lambda x: x["start_sec"])
    return out[: int(cfg.max_detail_spans)] if cfg.max_detail_spans else out


def extract_detail_evidence(question: str, options: Sequence[str], duration: float,
                            span: dict, coarse_claims: str,
                            frame_items: list[tuple[str, str]], sampling: str,
                            cfg: ChunkedQAConfig, model_fn: ModelFn) -> dict:
    s, e = float(span["start_sec"]), float(span["end_sec"])
    prompt = _render(cfg.detail_prompt_file, {
        "DURATION": f"{duration:.1f}",
        "SPAN_START": f"{s:.1f}",
        "SPAN_END": f"{e:.1f}",
        "SAMPLING": sampling,
        "QUESTION": question,
        "OPTIONS": "\n".join(options) if options else "(none)",
        "COARSE_CLAIMS": coarse_claims or "(none)",
    })
    raw = model_fn(prompt, frame_items, "detail")
    d = _extract_json(raw)
    vo: list[dict] = []
    for o in (d.get("verified_observations") or []):
        if not isinstance(o, dict):
            continue
        a = _clamp(o.get("start_sec"), s, e, s)
        b = _clamp(o.get("end_sec"), s, e, a)
        if b < a:
            a, b = b, a
        vo.append({
            "start_sec": round(a, 2), "end_sec": round(b, 2),
            "event": str(o.get("event", "")).strip(),
            "supports": str(o.get("supports", "unclear")).strip() or "unclear",
            "confidence": _clamp(o.get("confidence", 0.5), 0.0, 1.0, 0.5),
        })
    contras = [c for c in (d.get("contradictions") or []) if isinstance(c, dict)]
    out = {
        "source_span": {"start_sec": round(s, 2), "end_sec": round(e, 2)},
        "verified_observations": vo,
        "contradictions": contras,
        "resolved": bool(d.get("resolved", not contras)),
    }
    if _ANSWER_RE.search(raw or ""):
        out["_contained_final_answer"] = True
    return out


def read_detail_spans(question: str, options: Sequence[str], video_path: str,
                      duration: float, spans: list[dict], coarse: list[dict],
                      cfg: ChunkedQAConfig, model_fn: ModelFn,
                      read_video_fn: Callable[..., list[dict]] | None = None,
                      ) -> tuple[list[dict], dict]:
    stats: dict[str, Any] = {"read_video_calls": 0, "model_calls": 0,
                             "failed_spans": [], "frames_sent_total": 0}
    out: list[dict] = []
    for i, sp in enumerate(spans):
        claims = _claims_in_span(coarse, sp["start_sec"], sp["end_sec"])
        try:
            frames, note, fstats = _read_video_frames(
                video_path, sp["start_sec"], sp["end_sec"], fps=cfg.detail_fps,
                budget=cfg.detail_budget, duration=duration, cfg=cfg,
                read_video_fn=read_video_fn)
            stats["read_video_calls"] += 1
            ev = extract_detail_evidence(question, options, duration, sp, claims,
                                          frames, note, cfg, model_fn)
            stats["model_calls"] += 1
            stats["frames_sent_total"] += fstats.get("n_frames_sent", 0)
            ev["_read_stats"] = fstats
            out.append(ev)
        except Exception as exc:
            stats["failed_spans"].append(i)
            stats.setdefault("errors", {})[i] = str(exc)[:200]
    return out, stats


def _claims_in_span(coarse: list[dict], s: float, e: float) -> str:
    lines: list[str] = []
    for ev in coarse:
        for o in ev.get("observations") or []:
            if o["end_sec"] >= s and o["start_sec"] <= e:
                lines.append(f"- [{o['start_sec']:.1f}-{o['end_sec']:.1f}] {o.get('event', '')} "
                             f"(confidence {o.get('confidence', 0):.2f}"
                             f"{', needs_detail' if o.get('needs_detail') else ''})")
    return "\n".join(lines[:20])


_WORD_RE = re.compile(r"[a-z0-9\u4e00-\u9fff]+")


def _tokens(s: str) -> set[str]:
    return set(_WORD_RE.findall((s or "").lower()))


def _similar(a: str, b: str, thresh: float) -> bool:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return False
    inter = len(ta & tb)
    return inter / len(ta | tb) >= thresh


def build_event_instances(coarse: list[dict], detail: list[dict]) -> list[dict]:
    out: list[dict] = []
    for ev in coarse:
        for j, o in enumerate(ev.get("observations") or []):
            out.append({
                "event_id": f"coarse_{ev['chunk_index']:03d}_{j:03d}",
                "start_sec": o["start_sec"], "end_sec": o["end_sec"],
                "description": o.get("event", ""),
                "confidence": o.get("confidence", 0.5),
                "origin": "coarse",
            })
    for k, d in enumerate(detail):
        for j, o in enumerate(d.get("verified_observations") or []):
            out.append({
                "event_id": f"detail_{k:03d}_{j:03d}",
                "start_sec": o["start_sec"], "end_sec": o["end_sec"],
                "description": o.get("event", ""),
                "confidence": o.get("confidence", 0.5),
                "origin": "detail",
            })
    return out


def deduplicate_event_instances(instances: list[dict], cfg: ChunkedQAConfig,
                                ) -> tuple[list[dict], list[dict]]:
    if not instances:
        return [], []
    items = sorted(instances, key=lambda x: (float(x["start_sec"]), float(x["end_sec"])))
    kept: list[dict] = []
    merges: list[dict] = []
    for cur in items:
        placed = False
        for k in kept:
            overlap = float(cur["start_sec"]) <= float(k["end_sec"]) and \
                      float(cur["end_sec"]) >= float(k["start_sec"])
            gap = float(cur["start_sec"]) - float(k["end_sec"])
            adjacent_same = (0 <= gap <= cfg.dedup_gap_sec) and \
                _similar(cur.get("description", ""), k.get("description", ""), cfg.dedup_similarity)
            if (overlap and _similar(cur.get("description", ""), k.get("description", ""),
                                     cfg.dedup_similarity)) or adjacent_same:
                merges.append({"kept": k["event_id"], "dropped": cur["event_id"],
                               "why": "overlap" if overlap else "adjacent_same_event"})
                k["start_sec"] = min(float(k["start_sec"]), float(cur["start_sec"]))
                k["end_sec"] = max(float(k["end_sec"]), float(cur["end_sec"]))
                if cur.get("origin") == "detail" and k.get("origin") != "detail":
                    k["description"] = cur.get("description", k.get("description"))
                    k["confidence"] = cur.get("confidence", k.get("confidence"))
                    k["origin"] = "detail"
                    k["event_id"] = cur["event_id"]
                else:
                    k["confidence"] = max(float(k.get("confidence", 0)),
                                          float(cur.get("confidence", 0)))
                k["merged_count"] = int(k.get("merged_count", 1)) + 1
                placed = True
                break
        if not placed:
            kept.append(dict(cur, merged_count=1))
    kept.sort(key=lambda x: float(x["start_sec"]))
    return kept, merges


def synthesize_final_answer(question: str, options: Sequence[str], duration: float,
                            windows: list[list[float]], coarse: list[dict],
                            detail: list[dict], instances: list[dict],
                            cfg: ChunkedQAConfig, model_fn: ModelFn,
                            counting: bool = False) -> str:
    ev_lines: list[str] = []
    for ev in coarse:
        head = (f"[segment {ev['chunk_index']} {ev['chunk_start_sec']:.1f}-"
                f"{ev['chunk_end_sec']:.1f}s] relevant={str(ev.get('relevant')).lower()}")
        ev_lines.append(head)
        for o in ev.get("observations") or []:
            ev_lines.append(
                f"  - [{o['start_sec']:.1f}-{o['end_sec']:.1f}] {o.get('event', '')}"
                f" | relation: {o.get('relation_to_question', '')}"
                f" | type: {o.get('evidence_type', '')}"
                f" | confidence: {o.get('confidence', 0):.2f}"
                f"{' | needs_detail' if o.get('needs_detail') else ''}")
        if not (ev.get("observations") or []):
            ev_lines.append("  (no observations bearing on the question)")

    d_lines: list[str] = []
    for d in detail:
        sp = d.get("source_span") or {}
        d_lines.append(f"[detail {sp.get('start_sec', 0):.1f}-{sp.get('end_sec', 0):.1f}s] "
                       f"resolved={str(d.get('resolved')).lower()}")
        for o in d.get("verified_observations") or []:
            d_lines.append(f"  - [{o['start_sec']:.1f}-{o['end_sec']:.1f}] {o.get('event', '')}"
                           f" | supports: {o.get('supports', '')}"
                           f" | confidence: {o.get('confidence', 0):.2f}")
        for c in d.get("contradictions") or []:
            d_lines.append(f"  ! contradiction: coarse claimed {c.get('coarse_claim', '')!r}; "
                           f"frames show {c.get('frames_show', '')!r}")
    if not d_lines:
        d_lines.append("(no span needed a detail pass)")

    inst_block = ""
    if counting and instances:
        rows = [f"  {i + 1}. [{x['start_sec']:.1f}-{x['end_sec']:.1f}] {x.get('description', '')}"
                f" (confidence {float(x.get('confidence', 0)):.2f}, from {x.get('origin')})"
                for i, x in enumerate(instances)]
        inst_block = ("DE-DUPLICATED EVENT INSTANCES (use this list, and only this list, "
                      f"for any count — {len(instances)} instances)\n" + "\n".join(rows))

    prompt = _render(cfg.final_prompt_file, {
        "DURATION": f"{duration:.1f}",
        "N_CHUNKS": len(windows),
        "WINDOW_LIST": ", ".join(f"[{a:.0f},{b:.0f}]" for a, b in windows),
        "COARSE_EVIDENCE": "\n".join(ev_lines) or "(none)",
        "DETAIL_EVIDENCE": "\n".join(d_lines),
        "EVENT_INSTANCES": inst_block,
        "QUESTION": question,
        "OPTIONS": "\n".join(options) if options else "(open-ended)",
    })
    return model_fn(prompt, None, "final")


_COUNTING_RE = re.compile(
    r"(how many times|how many|how often|number of times|count of)", re.I)


def is_counting_question(question: str, qtype: str = "") -> bool:
    return bool(_COUNTING_RE.search(question or "")) or "count" in (qtype or "").lower()


def answer_video_question(question: str, video_path: str, *,
                          options: Sequence[str] | None = None,
                          qtype: str = "",
                          cfg: ChunkedQAConfig | None = None,
                          model_fn: ModelFn | None = None,
                          genome: dict | None = None,
                          llm_cfg: dict | None = None,
                          read_video_fn: Callable[..., list[dict]] | None = None,
                          duration: float | None = None,
                          ) -> dict:
    t0 = time.time()
    cfg = cfg or ChunkedQAConfig.from_genome(genome)
    options = list(options or [])
    out: dict[str, Any] = {"question": question, "video_path": video_path}

    meta = probe_video_metadata(video_path) if duration is None else {
        "ok": duration > 0, "duration": float(duration), "video_path": video_path}
    out["metadata"] = {k: v for k, v in meta.items() if k != "media_info_text"}
    if meta.get("media_info_text"):
        out["metadata"]["media_info_chars"] = len(meta["media_info_text"])
    if not meta.get("ok"):
        out["route"] = None
        out["error"] = meta.get("error", "metadata probe failed")
        return out

    dur = float(meta["duration"])
    route = should_use_chunked_read(dur)
    out["route"] = route
    out["duration"] = dur
    if route != ROUTE_CHUNKED:
        out["handled"] = False
        out["reason"] = (f"duration {dur:.1f}s routes to {route}; "
                         "chunked workflow only covers 300 < duration < 1800")
        return out
    out["handled"] = True

    if model_fn is None:
        model_fn = default_model_fn(llm_cfg or {}, genome or {}, cfg)

    windows = build_video_windows(dur, cfg.chunk_seconds)
    out["windows"] = windows

    coarse, cstats = read_coarse_chunks(question, video_path, dur, windows, cfg,
                                        model_fn, read_video_fn=read_video_fn)
    out["coarse_evidence"] = coarse
    out["coarse_stats"] = cstats
    if not coarse:
        out["error"] = "all coarse chunks failed"
        out["latency_sec"] = round(time.time() - t0, 2)
        return out

    raw_spans = collect_candidate_spans(coarse, dur, cfg)
    spans = merge_candidate_spans(raw_spans, dur, cfg)
    out["candidate_spans_raw"] = raw_spans
    out["detail_spans"] = spans

    detail, dstats = read_detail_spans(question, options, video_path, dur, spans,
                                       coarse, cfg, model_fn,
                                       read_video_fn=read_video_fn)
    out["detail_evidence"] = detail
    out["detail_stats"] = dstats

    counting = is_counting_question(question, qtype)
    instances, merges = deduplicate_event_instances(
        build_event_instances(coarse, detail), cfg)
    out["counting"] = counting
    out["event_instances"] = instances
    out["event_merges"] = merges
    if counting:
        out["event_count"] = len(instances)

    out["final_text"] = synthesize_final_answer(
        question, options, dur, windows, coarse, detail, instances, cfg, model_fn,
        counting=counting)
    out["n_final_calls"] = 1
    out["n_model_calls"] = (cstats.get("model_calls", 0) + dstats.get("model_calls", 0) + 1)
    out["n_read_video_calls"] = (cstats.get("read_video_calls", 0)
                                 + dstats.get("read_video_calls", 0))
    out["frames_sent_total"] = (cstats.get("frames_sent_total", 0)
                               + dstats.get("frames_sent_total", 0))
    out["latency_sec"] = round(time.time() - t0, 2)
    return out
