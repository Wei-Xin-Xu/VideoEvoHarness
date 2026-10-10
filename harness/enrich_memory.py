from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from common import (
    MEMORY_ROOT,
    VEH_DIR,
    ROOT,
    available_videos,
    call_llm,
    fmt_time,
    load_cfg,
    load_genome,
)

EVENT_TYPE_VOCAB = [
    "narration",
    "dialogue",
    "action",
    "demonstration",
    "interview",
    "archival_footage",
    "photo_display",
    "illustration_display",
    "map_display",
    "data_visualization",
    "text_overlay",
    "title_card",
    "animation",
    "gameplay",
    "meme_display",
    "scene_transition",
]

BACKUP_SUFFIX = ".pre_enrich.bak"

def _macro_material(macro: dict, include: list[str], max_chars: int) -> str:
    sg = macro.get("subgraph") or {}
    tr = macro.get("time_range") or [0, 0]
    base = float(tr[0])

    def _t(rng, fallback=None):
        r = rng if (isinstance(rng, list) and len(rng) == 2) else fallback
        if not r:
            return ""
        return f"{fmt_time(base + float(r[0]))}-{fmt_time(base + float(r[1]))}"

    cats: dict[str, list[str]] = {}
    if "events" in include:
        cats["events"] = [
            f"  - EVENT({_t(e.get('time_range'), [0, 0])}) {e.get('action', '')}: {e.get('description', '')}"
            for e in (sg.get("micro_events") or [])
            if e.get("description")
        ]
    if "entities" in include:
        cats["entities"] = [
            f"  - ENTITY[{e.get('entity_type', '')}] {e.get('name') or 'unnamed'}: {e.get('description', '')}"
            for e in (sg.get("entities") or [])
            if e.get("description") or e.get("name")
        ]
    if "ocr" in include:
        cats["ocr"] = [
            f'  - ON_SCREEN_TEXT({_t(o.get("time_range"), [0, 0])}) "{o.get("text", "")}"'
            for o in (sg.get("on_screen_texts") or [])
            if o.get("text")
        ]

    headers = {"events": "## Micro-events", "entities": "## Entities", "ocr": "## On-screen texts"}
    used = sum(len(headers[c]) for c in cats)
    picked: dict[str, list[str]] = {c: [] for c in cats}
    idx = {c: 0 for c in cats}
    progressed = True
    while progressed and used < max_chars:
        progressed = False
        for c in cats:
            i = idx[c]
            if i < len(cats[c]):
                line = cats[c][i]
                if used + len(line) > max_chars:
                    continue
                picked[c].append(line)
                used += len(line)
                idx[c] = i + 1
                progressed = True

    out: list[str] = []
    for c in cats:
        if picked[c]:
            out.append(headers[c])
            out.extend(picked[c])
    return "\n".join(out)

def _parse_json_obj(text: str) -> dict | None:
    if not text:
        return None
    start = text.find("{")
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
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
                    obj = json.loads(text[start : i + 1])
                    return obj if isinstance(obj, dict) else None
                except json.JSONDecodeError:
                    return None
    return None

def summarize_macro(macro: dict, g: dict, cfg: dict) -> dict:
    ms = g["memory"]["macro_summary"]
    tpl = (VEH_DIR / ms["prompt_file"]).read_text(encoding="utf-8")
    tr = macro.get("time_range") or [0, 0]
    material = _macro_material(macro, ms["include"], ms["max_material_chars"])
    if not material.strip():
        raise ValueError("empty subgraph material")

    if ms["event_type_vocab"] == "closed":
        instruction = (
            f"Choose 1 to 5 values from this closed list, most representative first. "
            f"Use no other value: {json.dumps(EVENT_TYPE_VOCAB)}"
        )
    else:
        instruction = "List 1 to 5 short snake_case type tags for this segment."

    prompt = tpl.format(
        time_range=f"{fmt_time(tr[0])}-{fmt_time(tr[1])}",
        duration=int(float(tr[1]) - float(tr[0])),
        material=material,
        max_chars=ms["max_chars"],
        event_type_instruction=instruction,
    )

    text = call_llm(prompt, cfg, max_tokens=ms["response_max_tokens"], temperature=0.2, role="summarizer")
    obj = _parse_json_obj(text)
    if not obj:
        raise ValueError(f"unparseable response: {text[:160]!r}")

    label = str(obj.get("label") or "").strip()
    summary = " ".join(str(obj.get("summary") or "").split())
    if not label or not summary:
        raise ValueError(f"missing label/summary: {str(obj)[:160]!r}")

    limit = ms["max_chars"]
    if len(summary) > limit:
        cut = summary[:limit]
        sp = cut.rfind(" ")
        summary = (cut[:sp] if sp > limit * 0.6 else cut).rstrip(" ,;:") + "..."

    types = [str(t).strip() for t in (obj.get("event_types") or []) if str(t).strip()]
    if ms["event_type_vocab"] == "closed":
        allowed = {v.lower(): v for v in EVENT_TYPE_VOCAB}
        types = list(dict.fromkeys(allowed[t.lower()] for t in types if t.lower() in allowed))
    return {"label": label, "summary": summary, "event_types": types[:5]}

def base_nodes(raw: dict) -> list[dict]:
    nodes: list[dict] = []
    for me in raw.get("macro_events") or []:
        sg = me.get("subgraph") or {}
        mid = me["macro_id"]
        for ent in sg.get("entities") or []:
            nodes.append(
                {
                    "node_id": ent.get("entity_id", ""),
                    "node_type": "entity",
                    "text": f"{ent.get('name') or ''}: {ent.get('description', '')}",
                    "macro_id": mid,
                }
            )
        for ev in sg.get("micro_events") or []:
            nodes.append(
                {
                    "node_id": ev.get("event_id", ""),
                    "node_type": "event",
                    "text": f"{ev.get('action', '')}: {ev.get('description', '')}",
                    "macro_id": mid,
                }
            )
        for ocr in sg.get("on_screen_texts") or []:
            nodes.append(
                {
                    "node_id": ocr.get("text_id", ""),
                    "node_type": "on_screen_text",
                    "text": f"{ocr.get('text', '')}: {ocr.get('description', '')}",
                    "macro_id": mid,
                }
            )
    return nodes

def hier_nodes(raw: dict, g: dict) -> list[dict]:
    ms = g["memory"]["macro_summary"]
    out: list[dict] = []
    if not ms["index_macro_nodes"]:
        return out

    for me in raw.get("macro_events") or []:
        summary = (me.get("summary") or "").strip()
        if not summary:
            continue
        label = (me.get("label") or "").strip()
        types = ", ".join(me.get("event_types") or [])
        ents = ", ".join(e.get("name", "") for e in (me.get("key_entities") or [])[:10] if e.get("name"))
        parts = [p for p in (label, summary, f"types: {types}" if types else "", f"entities: {ents}" if ents else "") if p]
        out.append(
            {
                "node_id": f"macrosum_{me['macro_id']}",
                "node_type": "macro",
                "text": " | ".join(parts),
                "macro_id": me["macro_id"],
            }
        )

    if ms["index_super_nodes"]:
        for se in raw.get("super_events") or []:
            desc = (se.get("description") or "").strip()
            if not desc:
                continue
            subs = se.get("sub_macro_ids") or []
            out.append(
                {
                    "node_id": f"supersum_{se.get('super_id', '')}",
                    "node_type": "super",
                    "text": f"{se.get('label', '')} | {desc}",
                    "macro_id": subs[0] if subs else "",
                }
            )
    return out

def verify_base_nodes(built: list[dict], npz_path: Path,
                      allow_added: bool = False) -> tuple[bool, str]:
    if not npz_path.exists():
        return True, "no existing index to compare against"
    try:
        import numpy as np

        d = np.load(npz_path, allow_pickle=True)
        old = json.loads(str(d["nodes"]))
    except Exception as exc:
        return True, f"could not read the existing index ({exc}); skipping comparison"

    old_keep = [n for n in old if n.get("node_type") in {"entity", "event", "on_screen_text"}]
    a = sorted((n["node_id"], n["node_type"], n["text"]) for n in old_keep)
    b = sorted((n["node_id"], n["node_type"], n["text"]) for n in built)
    if allow_added:
        missing = set(a) - set(b)
        if missing:
            return False, f"{len(missing)} existing nodes would be dropped, e.g. {sorted(missing)[0]}"
        return True, f"{len(b) - len(a)} nodes added, all {len(a)} existing nodes preserved"
    if len(old_keep) != len(built):
        return False, f"node count differs: existing {len(old_keep)} vs rebuilt {len(built)}"
    if a != b:
        diff = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), 0)
        return False, f"node content differs, first at #{diff}: {a[diff]} vs {b[diff]}"
    return True, f"{len(built)} bottom-layer nodes match the existing index exactly"

def rebuild_index(nodes: list[dict], out_path: Path) -> None:
    cap = ROOT / "Qwen-MM-Plugins" / "src" / "capabilities" / "video-memory"
    bm = cap / "skill" / "script" / "build_memory"
    for p in (bm, cap, ROOT / "Qwen-MM-Plugins" / "src"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    import os

    os.environ.setdefault("QWEN_MM_CONFIG", str(ROOT / "qwen-mm-config"))
    from embeddings import EmbeddingIndex

    index = EmbeddingIndex()
    index.build(nodes)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    index.save(str(tmp))
    tmp.replace(out_path)

def enrich_video(vk: str, g: dict, cfg: dict, workers: int, dry_run: bool, skip_embed: bool,
                 allow_added_nodes: bool = False, only_empty: bool = False) -> dict:
    d = MEMORY_ROOT / vk
    gpath, npz = d / "graph_memory.json", d / "embeddings.npz"
    raw = json.loads(gpath.read_text(encoding="utf-8"))
    macros = raw.get("macro_events") or []

    before_empty = sum(1 for m in macros if not (m.get("summary") or "").strip())

    targets = [m for m in macros if not (m.get("summary") or "").strip()] if only_empty else macros
    if only_empty:
        pass

    inc = g["memory"]["macro_summary"]["include"]
    cap = g["memory"]["macro_summary"]["max_material_chars"]
    no_material = [m for m in targets if not _macro_material(m, inc, cap).strip()]
    if no_material:
        targets = [m for m in targets if m not in no_material]

    if not targets:
        return {"video": vk, "macros": len(macros), "enriched": 0, "failures": [],
                "no_material": [m.get("macro_id") for m in no_material], "skipped": True}

    if dry_run:
        m = targets[0]
        mat = _macro_material(m, g["memory"]["macro_summary"]["include"],
                              g["memory"]["macro_summary"]["max_material_chars"])
        est = sum(
            len(_macro_material(x, g["memory"]["macro_summary"]["include"],
                                g["memory"]["macro_summary"]["max_material_chars"]))
            for x in targets
        )
        base = base_nodes(raw)
        ok, msg = verify_base_nodes(base, npz)
        return {"video": vk, "dry_run": True, "targets": len(targets)}

    for src in (gpath, npz):
        bak = src.with_suffix(src.suffix + BACKUP_SUFFIX)
        if src.exists() and not bak.exists():
            shutil.copy2(src, bak)

    t0 = time.time()
    results: dict[str, dict] = {}
    failures: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(summarize_macro, m, g, cfg): m["macro_id"] for m in targets}
        for fut in as_completed(futs):
            mid = futs[fut]
            try:
                results[mid] = fut.result()
            except Exception as exc:
                failures.append((mid, str(exc)[:120]))

    if not results:
        raise SystemExit("every macro failed; leaving the memory untouched")

    for m in macros:
        r = results.get(m["macro_id"])
        if not r:
            continue
        m["label"] = r["label"]
        m["summary"] = r["summary"]
        if r["event_types"]:
            m["event_types"] = r["event_types"]

    gpath.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")

    stats = {"video": vk, "macros": len(macros), "targets": len(targets),
             "enriched": len(results), "failures": failures,
             "no_material": [m.get("macro_id") for m in no_material]}
    if skip_embed:
        return stats

    base = base_nodes(raw)
    ok, msg = verify_base_nodes(base, npz, allow_added=allow_added_nodes)
    if not ok:
        raise SystemExit(f"bottom-layer nodes do not match the existing index; refusing to rebuild ({msg})")

    extra = hier_nodes(raw, g)
    nodes = base + extra

    rebuild_index(nodes, npz)
    stats["nodes_total"] = len(nodes)
    stats["nodes_added"] = len(extra)
    return stats

def restore(vk: str) -> None:
    d = MEMORY_ROOT / vk
    n = 0
    for name in ("graph_memory.json", "embeddings.npz"):
        src = d / name
        bak = src.with_suffix(src.suffix + BACKUP_SUFFIX)
        if bak.exists():
            shutil.copy2(bak, src)
            n += 1
    if not n:
        pass

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true", help="print material and cost, call nothing")
    ap.add_argument("--skip-embed", action="store_true", help="write summaries but keep the current index")
    ap.add_argument("--restore", action="store_true", help="roll back from the backups")
    ap.add_argument("--allow-added-nodes", action="store_true",
                    help="accept a rebuilt superset of bottom-layer nodes (memory patch candidates); "
                         "dropping or rewriting existing nodes is still refused")
    ap.add_argument("--only-empty", action="store_true",
                    help="only summarise macros whose summary is empty; leave existing "
                         "summaries byte-for-byte untouched (use this to backfill holes)")
    args = ap.parse_args()

    vks = args.videos or available_videos()
    if not vks:
        raise SystemExit("no memory found under memory/")

    if args.restore:
        for vk in vks:
            restore(vk)
        return

    g, cfg = load_genome(), load_cfg()
    if "memory" not in g or "macro_summary" not in g.get("memory", {}):
        raise SystemExit("genome.json has no memory.macro_summary block")
    if not g["memory"]["macro_summary"]["enabled"]:
        raise SystemExit("memory.macro_summary.enabled is false")

    out = []
    for vk in vks:
        try:
            out.append(enrich_video(vk, g, cfg, args.workers, args.dry_run, args.skip_embed,
                                    allow_added_nodes=args.allow_added_nodes,
                                    only_empty=args.only_empty))
        except SystemExit as exc:
            out.append({"video": vk, "aborted": str(exc)})

if __name__ == "__main__":
    main()
