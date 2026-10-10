from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from common import (
    MEMORY_ROOT,
    ROOT,
    VEH_DIR,
    STORE,
    active_memory_dir,
    available_videos,
    cosine,
    embed_texts,
    fmt_time,
    load_cfg,
    load_genome,
)

BACKUP_SUFFIX = ".pre_patch.bak"

def _import_official():
    cap = ROOT / "Qwen-MM-Plugins" / "src" / "capabilities" / "video-memory"
    bm = cap / "skill" / "script" / "build_memory"
    for p in (bm, cap, ROOT / "Qwen-MM-Plugins" / "src"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    import os

    os.environ.setdefault("QWEN_MM_CONFIG", str(ROOT / "qwen-mm-config"))
    import build_graph as bg
    from llm_client import call_vlm_curl, extract_json
    from prompts import SUBGRAPH_CONSTRUCTION_PROMPT

    return bg, call_vlm_curl, extract_json, SUBGRAPH_CONSTRUCTION_PROMPT

def density(macro: dict) -> dict:
    sg = macro.get("subgraph") or {}
    tr = macro.get("time_range") or [0, 0]
    dur = max(1.0, float(tr[1]) - float(tr[0]))
    ne = len(sg.get("micro_events") or [])
    return {
        "macro_id": macro["macro_id"],
        "start": float(tr[0]),
        "end": float(tr[1]),
        "dur": dur,
        "events": ne,
        "entities": len(sg.get("entities") or []),
        "ocr": len(sg.get("on_screen_texts") or []),
        "events_per_min": round(ne / (dur / 60.0), 2),
    }

def scan(video_key: str) -> list[dict]:
    raw = json.loads((active_memory_dir(video_key) / "graph_memory.json").read_text(encoding="utf-8"))
    rows = [density(m) for m in raw.get("macro_events") or []]
    return sorted(rows, key=lambda r: r["events_per_min"])

def find_gaps(macro: dict, min_gap_sec: float) -> list[tuple[float, float]]:
    tr = macro.get("time_range") or [0, 0]
    lo, hi = float(tr[0]), float(tr[1])
    spans: list[tuple[float, float]] = []
    for ev in (macro.get("subgraph") or {}).get("micro_events") or []:
        t = ev.get("time_range")
        if isinstance(t, list) and len(t) == 2:
            try:
                a, b = float(t[0]), float(t[1])
            except (TypeError, ValueError):
                continue
            if b >= a:
                spans.append((max(lo, a), min(hi, b)))
    spans.sort()

    merged: list[list[float]] = []
    for a, b in spans:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])

    gaps: list[tuple[float, float]] = []
    cur = lo
    for a, b in merged:
        if a - cur >= min_gap_sec:
            gaps.append((cur, a))
        cur = max(cur, b)
    if hi - cur >= min_gap_sec:
        gaps.append((cur, hi))
    return gaps

def _dedup_new(new_items: list[dict], old_items: list[dict], key: str, thr: float,
               cfg: dict, time_key: str = "time_range") -> tuple[list[dict], int]:
    if not new_items:
        return [], 0
    old_texts = [str(o.get(key) or "") for o in old_items if o.get(key)]
    kept, dropped = [], 0
    if not old_texts:
        return new_items, 0
    try:
        old_vecs = embed_texts(old_texts, cfg)
    except Exception as exc:
        return new_items, 0

    for it in new_items:
        txt = str(it.get(key) or "")
        if not txt:
            continue
        try:
            v = embed_texts([txt], cfg)[0]
        except Exception:
            kept.append(it)
            continue
        is_dup = False
        for o, ov in zip([o for o in old_items if o.get(key)], old_vecs):
            if cosine(v, ov) < thr:
                continue
            if time_key:
                a, b = it.get(time_key), o.get(time_key)
                if isinstance(a, list) and isinstance(b, list) and len(a) == 2 and len(b) == 2:
                    if float(a[1]) < float(b[0]) or float(a[0]) > float(b[1]):
                        continue
            is_dup = True
            break
        if is_dup:
            dropped += 1
        else:
            kept.append(it)
    return kept, dropped

def patch_macro(video_path: str, macro: dict, g: dict, cfg: dict, bg, call_vlm, extract_json,
                prompt: str, question_hint: str = "",
                window: tuple[float, float] | None = None,
                used_ids: set[str] | None = None) -> dict:
    p = g["memory"]["patch"]
    tr = macro["time_range"]
    m_start, m_end = float(tr[0]), float(tr[1])
    start, end = window if window else (m_start, m_end)

    frame_items = bg.extract_frames_base64(
        video_path, start, end,
        fps=float(p["patch_fps"]),
        max_dim=int(p["max_dim"]),
        max_frames=int(p["max_frames_per_patch"]),
    )
    if not frame_items:
        return {"macro_id": macro["macro_id"], "error": "frame extraction produced nothing"}

    context = (
        f"This is the segment from {fmt_time(start)} to {fmt_time(end)} of a long video, "
        f"re-examined at higher frame density because the existing record for it is sparse. "
        f"Be exhaustive: list every distinct action, entity and on-screen text you can see. "
        f"All timestamps must be RELATIVE to this segment, starting at 0.\n"
    )
    if question_hint:
        context += f"Pay particular attention to details of this kind: {question_hint}\n"

    t0 = time.time()
    try:
        resp = call_vlm(
            context + prompt,
            frame_items=frame_items,
            model=cfg.get("QWEN_MM_API_VL_MODEL", "qwen3.7-plus"),
            api_key=cfg.get("DASHSCOPE_API_KEY", ""),
        )
        data = extract_json(resp)
    except Exception as exc:
        return {"macro_id": macro["macro_id"], "error": f"VLM/parse failed: {str(exc)[:140]}"}
    if not isinstance(data, dict):
        return {"macro_id": macro["macro_id"], "error": "response was not a JSON object"}

    sg = macro.setdefault("subgraph", {})
    sg.setdefault("micro_events", [])
    sg.setdefault("entities", [])
    sg.setdefault("on_screen_texts", [])

    def _to_float(v):
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            try:
                return float(v)
            except ValueError:
                parts = [float(x) for x in v.split(":")] if ":" in v else [0.0]
                out = 0.0
                for x in parts:
                    out = out * 60 + x
                return out
        return 0.0

    for ev in data.get("micro_events") or []:
        t = ev.get("time_range")
        ev["time_range"] = (
            [_to_float(t[0]) + start, _to_float(t[1]) + start]
            if isinstance(t, list) and len(t) == 2 else [start, end]
        )
    for o in data.get("on_screen_texts") or []:
        t = o.get("time_range")
        o["time_range"] = (
            [_to_float(t[0]) + start, _to_float(t[1]) + start]
            if isinstance(t, list) and len(t) == 2 else [start, end]
        )

    thr = float(p["dedup_cosine"])
    new_ev, drop_ev = _dedup_new(data.get("micro_events") or [], sg["micro_events"],
                                 "description", thr, cfg)
    new_en, drop_en = _dedup_new(data.get("entities") or [], sg["entities"],
                                 "name", thr, cfg, time_key="")
    new_oc, drop_oc = _dedup_new(data.get("on_screen_texts") or [], sg["on_screen_texts"],
                                 "text", thr, cfg, time_key="")

    mid = macro["macro_id"]
    used = used_ids if used_ids is not None else set()
    used |= {
        item.get(key)
        for field, key in (("micro_events", "event_id"), ("entities", "entity_id"),
                           ("on_screen_texts", "text_id"))
        for item in (sg.get(field) or [])
        if item.get(key)
    }

    def _mint(kind: str) -> str:
        i = 0
        while True:
            nid = f"patch_{mid}_{kind}{i:03d}"
            if nid not in used:
                used.add(nid)
                return nid
            i += 1

    for ev in new_ev:
        ev["event_id"] = _mint("ev")
    for en in new_en:
        en["entity_id"] = _mint("en")
    for oc in new_oc:
        oc["text_id"] = _mint("oc")

    sg["micro_events"].extend(new_ev)
    sg["entities"].extend(new_en)
    sg["on_screen_texts"].extend(new_oc)

    return {
        "macro_id": mid,
        "window": [round(start, 1), round(end, 1)],
        "frames": len(frame_items),
        "added": {"events": len(new_ev), "entities": len(new_en), "ocr": len(new_oc)},
        "deduped": {"events": drop_ev, "entities": drop_en, "ocr": drop_oc},
        "seconds": round(time.time() - t0, 1),
    }

def _atomic_pointer(path: Path, value: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(value + "\n", encoding="utf-8")
    tmp.replace(path)

def _child_ids(graph: dict) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for macro in graph.get("macro_events") or []:
        out[macro.get("macro_id")] = [
            item.get(key)
            for field, key in (("micro_events", "event_id"), ("entities", "entity_id"),
                               ("on_screen_texts", "text_id"))
            for item in ((macro.get("subgraph") or {}).get(field) or [])
            if item.get(key)
        ]
    return out

def _collisions(by_macro: dict[str, list[str]]) -> set[str]:
    flat = [i for ids in by_macro.values() for i in ids]
    return {i for i in flat if flat.count(i) > 1}

def _validate_candidate(path: Path, baseline: dict | None = None) -> dict:
    graph = json.loads((path / "graph_memory.json").read_text(encoding="utf-8"))
    macros = graph.get("macro_events") or []
    if not macros:
        raise ValueError("candidate has no macro_events")
    ids: list[str] = []
    for macro in macros:
        mid = macro.get("macro_id")
        tr = macro.get("time_range")
        if not mid or not isinstance(tr, list) or len(tr) != 2 or float(tr[1]) <= float(tr[0]):
            raise ValueError(f"invalid macro schema: {mid!r}")
        ids.append(mid)
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate macro IDs")

    cand_by_macro = _child_ids(graph)
    for mid, child in cand_by_macro.items():
        dupes = sorted({x for x in child if child.count(x) > 1})
        base_child = ((baseline and _child_ids(baseline).get(mid)) or [])
        base_dupes = {x for x in base_child if base_child.count(x) > 1}
        new_dupes = [d for d in dupes if d not in base_dupes]
        if new_dupes:
            raise ValueError(f"patch introduced duplicate child IDs in {mid}: {new_dupes[:5]}")
    cand_cross = _collisions(cand_by_macro)
    base_cross = _collisions(_child_ids(baseline)) if baseline else set()
    new_cross = sorted(cand_cross - base_cross)
    if new_cross:
        raise ValueError(f"patch introduced cross-macro ID reuse: {new_cross[:5]}")
    pre_existing = len(cand_cross & base_cross)

    embed = path / "embeddings.npz"
    if not embed.exists() or embed.stat().st_size == 0:
        raise ValueError("embedding index was not rebuilt")
    report = {"macros": len(macros), "embedding_bytes": embed.stat().st_size}
    if pre_existing:
        report["warning"] = (
            f"{pre_existing} node IDs were already reused across macros before this patch; "
            "retrieval merges them, but the patch did not make it worse"
        )
    return report
    embed = path / "embeddings.npz"
    if not embed.exists() or embed.stat().st_size == 0:
        raise ValueError("embedding index was not rebuilt")
    return {"macros": len(macros), "embedding_bytes": embed.stat().st_size}

def _rebuild_and_validate(video_key: str, candidate: Path, baseline: dict | None = None) -> dict:
    staging_root = candidate.parent
    env = {**os.environ, "VEH_MEMORY_ROOT": str(staging_root),
           "VEH_CONFIG_FILE": str(ROOT / "qwen-mm-config")}
    proc = subprocess.run(
        [sys.executable, str(VEH_DIR / "enrich_memory.py"), "--videos", video_key,
         "--allow-added-nodes"],
        cwd=str(VEH_DIR), env=env, capture_output=True, text=True, timeout=7200,
    )
    if proc.returncode:
        raise RuntimeError(f"index rebuild failed: {proc.stderr[-1200:]}")
    report = _validate_candidate(candidate, baseline=baseline)
    report["enrich_tail"] = (proc.stdout or "").splitlines()[-8:]
    return report

def _commit_candidate(video_key: str, candidate: Path, version: str, report: dict) -> Path:
    base = MEMORY_ROOT / video_key
    versions = base / "versions"
    versions.mkdir(parents=True, exist_ok=True)
    dest = versions / version
    if dest.exists():
        raise FileExistsError(dest)
    candidate.replace(dest)
    current = base / "CURRENT"
    previous = current.read_text(encoding="utf-8").strip() if current.exists() else ""
    if previous:
        _atomic_pointer(base / "PREVIOUS", previous)
    elif active_memory_dir(video_key) == base:
        legacy = "legacy_pre_versioning"
        legacy_dir = versions / legacy
        if not legacy_dir.exists():
            legacy_dir.mkdir()
            for name in ("graph_memory.json", "embeddings.npz"):
                src = base / name
                if src.exists():
                    shutil.copy2(src, legacy_dir / name)
        _atomic_pointer(base / "PREVIOUS", legacy)
    (dest / "patch_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _atomic_pointer(current, version)
    return dest

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=None)
    ap.add_argument("--scan", action="store_true", help="density report only, no cost")
    ap.add_argument("--top", type=int, default=None, help="patch the N sparsest macros")
    ap.add_argument("--macros", nargs="*", default=None, help="patch these macro_ids explicitly")
    ap.add_argument("--hint", default="", help="question-guided mode; use evolve-pool questions only")
    ap.add_argument("--from-discoveries", action="store_true",
                    help="consume evolve-only windows recorded by solve.py instead of density gaps")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--restore", action="store_true")
    args = ap.parse_args()

    vks = [args.video] if args.video else available_videos()
    if not vks:
        raise SystemExit("no memory found under memory/")

    if args.restore:
        for vk in vks:
            base = MEMORY_ROOT / vk
            previous = base / "PREVIOUS"
            if previous.exists():
                version = previous.read_text(encoding="utf-8").strip()
                _atomic_pointer(base / "CURRENT", version)
            else:
                pass
        return

    g, cfg = load_genome(), load_cfg()
    p = g.get("memory", {}).get("patch")
    if not p:
        raise SystemExit("genome.json has no memory.patch block")

    for vk in vks:
        rows = scan(vk)
        for r in rows:
            flag = "  <- sparse" if r["events_per_min"] < p["min_events_per_min"] else ""
        if args.scan:
            continue

        source_dir = active_memory_dir(vk)
        raw = json.loads((source_dir / "graph_memory.json").read_text(encoding="utf-8"))
        by_id = {m["macro_id"]: m for m in raw["macro_events"]}

        cand: list[tuple[str, float, float]] = []
        if args.from_discoveries:
            queue = STORE / "memory_patch_candidates.jsonl"
            discoveries = []
            if queue.exists():
                for line in queue.read_text(encoding="utf-8").splitlines():
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if item.get("video_key") == vk and isinstance(item.get("window"), list):
                        discoveries.append(item)
            seen_windows = set()
            for item in discoveries:
                a, b = map(float, item["window"][:2])
                key = (round(a, 1), round(b, 1))
                if key in seen_windows:
                    continue
                seen_windows.add(key)
                best = max(
                    raw["macro_events"],
                    key=lambda m: max(0.0, min(float(m["time_range"][1]), b)
                                    - max(float(m["time_range"][0]), a)),
                )
                if args.macros and best["macro_id"] not in args.macros:
                    continue
                cand.append((best["macro_id"], a, b))
        else:
            for m in raw["macro_events"]:
                if args.macros and m["macro_id"] not in args.macros:
                    continue
                for a, b in find_gaps(m, float(p["min_gap_sec"])):
                    cand.append((m["macro_id"], a, b))
        cand.sort(key=lambda c: c[2] - c[1], reverse=True)
        n = args.top if args.top is not None else int(p["max_gaps_per_round"])
        targets = cand[:n]

        source_label = "frame-discovery windows" if args.from_discoveries else f"uncovered gaps >= {p['min_gap_sec']}s"
        if not targets:
            continue
        est = 0
        for mid, a, b in targets:
            nf = max(4, min(int(p["max_frames_per_patch"]), int((b - a) * float(p["patch_fps"])) + 1))
            est += nf
        if args.dry_run:
            continue

        version = f"patch_{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns() % 1000000:06d}"
        candidate_root = MEMORY_ROOT / vk / "candidates" / version
        candidate = candidate_root / vk
        candidate_root.mkdir(parents=True, exist_ok=False)
        shutil.copytree(source_dir, candidate,
                        ignore=shutil.ignore_patterns("candidates", "versions"))
        gpath = candidate / "graph_memory.json"

        bg, call_vlm, extract_json, prompt = _import_official()
        video_path = raw.get("video_path")
        if not video_path or not Path(video_path).exists():
            raise SystemExit(f"source video not found: {video_path}")

        results = []
        used_ids = {
            item.get(key)
            for macro in raw["macro_events"]
            for field, key in (("micro_events", "event_id"), ("entities", "entity_id"),
                               ("on_screen_texts", "text_id"))
            for item in ((macro.get("subgraph") or {}).get(field) or [])
            if item.get(key)
        }
        for mid, a, b in targets:
            res = patch_macro(video_path, by_id[mid], g, cfg, bg,
                              call_vlm, extract_json, prompt, args.hint, window=(a, b),
                              used_ids=used_ids)
            results.append(res)
            if res.get("error"):
                continue

        successful = [r for r in results if not r.get("error")]
        added_total = sum(sum(r.get("added", {}).values()) for r in successful)
        if not successful or added_total <= 0:
            shutil.rmtree(candidate_root, ignore_errors=True)
            continue

        gpath.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
        report = {
            "video_key": vk,
            "version": version,
            "source": str(source_dir),
            "results": results,
            "added_total": added_total,
        }
        try:
            baseline = json.loads((source_dir / "graph_memory.json").read_text(encoding="utf-8"))
            report["validation"] = _rebuild_and_validate(vk, candidate, baseline=baseline)
            dest = _commit_candidate(vk, candidate, version, report)
            if candidate_root.exists():
                candidate_root.rmdir()
        except Exception as exc:
            report["rejected"] = str(exc)
            for junk in candidate.rglob("*.npz*"):
                junk.unlink(missing_ok=True)
            (candidate_root / "rejection.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            continue

        touched = {mid for mid, _, _ in targets}
        committed_raw = json.loads((dest / "graph_memory.json").read_text(encoding="utf-8"))
        for macro in committed_raw.get("macro_events") or []:
            if macro.get("macro_id") in touched:
                pass
        left = sum(b - a for m in committed_raw["macro_events"]
                   for a, b in find_gaps(m, float(p["min_gap_sec"])))

if __name__ == "__main__":
    main()
