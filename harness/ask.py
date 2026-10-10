from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

from common import (
    DATA_SPLIT_FILE,
    VEH_DIR,
    Memory,
    assign_split,
    available_videos,
    fmt_time,
    load_cfg,
    load_genome,
    save_questions,
)

TAXONOMY = json.loads((VEH_DIR / "question_taxonomy.json").read_text(encoding="utf-8"))

def load_data_split() -> dict:
    if not DATA_SPLIT_FILE.is_file():
        raise FileNotFoundError(
            "Create harness/data_split.json from harness/data_split.example.json "
            "and list your evolution videos, or set VEH_DATA_SPLIT_PATH."
        )
    data = json.loads(DATA_SPLIT_FILE.read_text(encoding="utf-8"))
    if (not isinstance(data, dict)
            or not isinstance(data.get("evolve_source"), dict)
            or not isinstance(data.get("test_benches"), dict)
            or not isinstance(data["evolve_source"].get("videos"), list)):
        raise ValueError("The data split must contain evolve_source.videos and test_benches.")
    return data

_TS_RE = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b|\b\d+(?:\.\d+)?\s*(?:s|sec|seconds)\b", re.I)
_SPATIAL_RE = re.compile(
    r"\b(left|right|top|bottom|center|centre|behind|front|above|below|beside|next to|"
    r"background|foreground|corner|upper|lower)\b",
    re.I,
)

_Q: dict = {
    "option_max_len": 180,
    "option_min_len": 8,
    "prefer_adjacent_macros": TAXONOMY["distractor_policy"]["prefer_adjacent_macros"],
    "adjacent_window": TAXONOMY["distractor_policy"]["adjacent_window"],
    "fallback_to_random": TAXONOMY["distractor_policy"]["fallback_to_random"],
    "same_type_only": True,
    "form": {},
    "type_enabled": {},
    "weight_override": {},
}

def apply_genome(g: dict) -> dict:
    q = g.get("questioner", {})
    d = q.get("distractor", {})
    o = q.get("option_text", {})
    _Q.update(
        option_max_len=o.get("max_len", _Q["option_max_len"]),
        option_min_len=o.get("min_len_default", _Q["option_min_len"]),
        prefer_adjacent_macros=d.get("prefer_adjacent_macros", _Q["prefer_adjacent_macros"]),
        adjacent_window=d.get("adjacent_window", _Q["adjacent_window"]),
        fallback_to_random=d.get("fallback_to_random", _Q["fallback_to_random"]),
        same_type_only=d.get("same_type_only", _Q["same_type_only"]),
        form={k: v for k, v in q.get("form", {}).items() if not k.startswith("_")},
        type_enabled={k: v for k, v in q.get("type_enabled", {}).items() if not k.startswith("_")},
        weight_override={
            k: v for k, v in q.get("type_weight_override", {}).items() if not k.startswith("_")
        },
    )
    return _Q

def type_weight(qtype: str) -> float:
    if qtype in _Q["weight_override"]:
        return float(_Q["weight_override"][qtype])
    spec = TAXONOMY["types"].get(qtype)
    if not spec:
        return 0.0
    if spec.get("weight_gen") is not None:
        return float(spec["weight_gen"])
    return float(spec.get("weight_long", 0.0))

def target_weight(qtype: str) -> float:
    spec = TAXONOMY["types"].get(qtype)
    return float(spec.get("weight_long", 0.0)) if spec else 0.0

def type_on(qtype: str) -> bool:
    if not bool(_Q["type_enabled"].get(qtype, True)):
        return False
    if qtype in _Q["weight_override"]:
        return float(_Q["weight_override"][qtype]) > 0.0
    spec = TAXONOMY["types"].get(qtype) or {}
    wg = spec.get("weight_gen")
    return True if wg is None else float(wg) > 0.0

def test_bench_video_ids() -> set[str]:
    data_split = load_data_split()
    blacklist: set[str] = set()

    manifest = data_split.get("bench_ids_path")
    if manifest:
        mp = Path(manifest)
        if not mp.is_absolute():
            mp = VEH_DIR.parent / mp
        if mp.exists():
            try:
                blacklist |= {str(x) for x in json.loads(mp.read_text())}
            except Exception as exc:
                pass
        else:
            pass

    vm = data_split["test_benches"].get("videomme_long", {})
    p = Path(vm.get("local_path") or "")
    if not p.is_absolute():
        p = VEH_DIR.parent / p
    if p.exists():
        try:
            import pandas as pd

            df = pd.read_parquet(p)
            blacklist |= set(df["videoID"].astype(str).unique())
        except Exception as exc:
            pass

    for name, spec in data_split["test_benches"].items():
        if name.startswith("_") or not isinstance(spec, dict):
            continue
        for vid in spec.get("video_ids", []) or []:
            blacklist.add(str(vid))
    return blacklist

def _bench_durations() -> dict[str, list[float]]:
    fp = load_data_split().get("bench_durations_path")
    if not fp:
        return {}
    p = Path(fp)
    if not p.is_absolute():
        p = VEH_DIR.parent / p
    if not p.exists():
        return {}
    try:
        raw = json.loads(p.read_text())
        return {
            str(k): [float(x) for x in (v if isinstance(v, list) else [v])]
            for k, v in raw.items()
        }
    except Exception:
        return {}

def leakage_check(video_keys: list[str], allow: bool) -> None:
    bl = test_bench_video_ids()
    candidates: list[tuple[str, str]] = []
    for video_key in video_keys:
        source_id = video_key
        try:
            source_path = Memory(video_key).video_path
            if source_path:
                source_id = Path(source_path).stem
        except Exception:
            pass
        if video_key in bl:
            candidates.append((video_key, video_key))
        elif source_id in bl:
            candidates.append((video_key, source_id))

    hit, spurious = [], []
    durations = _bench_durations()
    for video_key, source_id in candidates:
        bench_ds = durations.get(source_id) or []
        vd = None
        try:
            vd = Memory(video_key).duration_sec()
        except Exception:
            vd = None
        if not bench_ds or vd is None:
            hit.append(video_key)
        elif any(abs(bd - vd) <= 2.0 for bd in bench_ds):
            hit.append(video_key)
        else:
            spurious.append((video_key, vd, bench_ds))

    if spurious:
        pass

    if not hit:
        return
    msg = (
        f"\n{'=' * 66}\n"
        f"DATA LEAKAGE: the following videos belong to a test bench and must not be\n"
        f"used for question generation:\n  {hit}\n\n"
        f"Generating questions on test videos is like practising on the exam paper;\n"
        f"every conclusion drawn from it is invalid.\n"
        f"Switch the evolution source to videos that do not overlap any test bench\n"
        f"(see `evolve_source` in harness/data_split.json).\n{'=' * 66}"
    )
    if not allow:
        raise SystemExit(msg)

def _clean(text: str, limit: int | None = None) -> str:
    t = _TS_RE.sub("", str(text or "")).strip()
    t = re.sub(r"\s+", " ", t)
    lim = limit or _Q["option_max_len"]
    if len(t) > lim:
        cut = t[:lim]
        sp = cut.rfind(" ")
        if sp >= lim * 0.6:
            cut = cut[:sp]
        t = cut
    return t.rstrip(" ,.;:-–—")

def _distinct(options: list[str], min_len: int) -> bool:
    o = [x.strip().lower() for x in options]
    return len(set(o)) == len(o) and all(len(x) >= min_len for x in o)

_WORD_RE = re.compile(r"[a-z]{4,}")

_REJECTED_LEX: dict[str, int] = {}
_REJECTED_LEAK: dict[str, int] = {}

_INFO_SYNOPSIS_FORM_B = False

_TIME_IN_STEM = re.compile(r"\b(\d{1,2}):([0-5]\d)\b")

def _anchor_leaked(question: str, time_range: list[float], tol: float = 1.0) -> bool:
    if not time_range or len(time_range) != 2:
        return False
    hits = _TIME_IN_STEM.findall(question or "")
    if not hits:
        return False
    secs = [int(m) * 60 + int(s) for m, s in hits]
    lo, hi = min(secs), max(secs)
    if hi - lo <= 0:
        return False
    return abs(lo - float(time_range[0])) < tol and abs(hi - float(time_range[1])) < tol

_POS_COUNTER: dict[str, int] = {}

def _pick_distractors(correct: str, pool: list[str], k: int, tol: float) -> list[str]:
    if tol <= 0 or not pool:
        return pool[:k]
    L = len(correct)
    margin = max(tol * L, 4.0)
    near = [d for d in pool if abs(len(d) - L) <= margin]
    if len(near) >= k:
        return near[:k]
    rest = [d for d in pool if abs(len(d) - L) > margin]
    return (near + rest)[:k]

def _lex_shortcut(question: str, correct: str, distractors: list[str], max_diff: float) -> bool:
    if max_diff <= 0:
        return False
    q = set(_WORD_RE.findall(question.lower()))
    if not q:
        return False
    c = len(q & set(_WORD_RE.findall(correct.lower())))
    if not distractors:
        return False
    ds = [len(q & set(_WORD_RE.findall(d.lower()))) for d in distractors]
    return (c - sum(ds) / len(ds)) > max_diff

def _neighbor_macros(mem: Memory, mid: str, window: int | None = None) -> list[str]:
    window = _Q["adjacent_window"] if window is None else window
    ids = mem.all_macro_ids()
    if mid not in ids:
        return [m for m in ids if m != mid]
    i = ids.index(mid)
    near = [ids[j] for j in range(max(0, i - window), min(len(ids), i + window + 1)) if ids[j] != mid]
    far = [m for m in ids if m != mid and m not in near]
    if _Q["prefer_adjacent_macros"]:
        return near + (far if _Q["fallback_to_random"] else [])
    return near + far

def _pool_typed(
    mem: Memory,
    mid: str,
    field: str,
    extract: Callable[[dict], str],
    window: int,
    type_key: str,
    type_val: str,
) -> list[str]:
    tv = str(type_val or "").upper()

    def collect(mids: list[str], same_type: bool) -> list[str]:
        out = []
        for omid in mids:
            for item in mem.visual_fields(omid).get(field, []):
                it = str(item.get(type_key) or "").upper()
                if same_type and it != tv:
                    continue
                if not same_type and it == tv:
                    continue
                s = _clean(extract(item))
                if s:
                    out.append(s)
        return out

    near = _neighbor_macros(mem, mid, window)
    if not _Q["same_type_only"]:
        return _pool(mem, mid, field, extract, window)
    res = collect(near, True)
    if len(res) < 3:
        allm = [m for m in mem.all_macro_ids() if m != mid]
        res += [s for s in collect(allm, True) if s not in res]
    if len(res) < 3:
        allm = [m for m in mem.all_macro_ids() if m != mid]
        res += [s for s in collect(allm, False) if s not in res]
    return res

def _pool(mem: Memory, mid: str, field: str, extract: Callable[[dict], str], window: int) -> list[str]:
    out: list[str] = []
    for omid in _neighbor_macros(mem, mid, window):
        for item in mem.visual_fields(omid).get(field, []):
            s = _clean(extract(item))
            if s:
                out.append(s)
    return out

def _mk(
    mem: Memory,
    qtype: str,
    question: str,
    correct: str,
    distractors: list[str],
    macro_id: str,
    time_range: list[float],
    rnd: random.Random,
    g_or_nopt,
    min_opt_len: int = 8,
    extra_macro_ids: list[str] | None = None,
    micro_ids: list[str] | None = None,
    micro_time_range: list[float] | None = None,
    granularity: str = "macro",
) -> dict | None:
    n_opt = g_or_nopt["questioner"]["n_options"] if isinstance(g_or_nopt, dict) else int(g_or_nopt)
    ds = [d for d in distractors if d.strip().lower() != correct.strip().lower()]
    seen, uniq = set(), []
    for d in ds:
        k = d.strip().lower()
        if k not in seen:
            seen.add(k)
            uniq.append(d)
    if len(uniq) < n_opt - 1 or not correct:
        return None
    tol = 0.35
    if isinstance(g_or_nopt, dict):
        tol = float(g_or_nopt["questioner"].get("option_len_tolerance", 0.35))
    chosen = _pick_distractors(correct, uniq, n_opt - 1, tol)
    if len(chosen) < n_opt - 1:
        return None
    if not _distinct([correct, *chosen], min_opt_len):
        return None
    max_diff = 0.5
    if isinstance(g_or_nopt, dict):
        max_diff = float(g_or_nopt["questioner"].get("max_lexical_overlap_diff", 0.5))
    if _lex_shortcut(question, correct, chosen, max_diff):
        _REJECTED_LEX[qtype] = _REJECTED_LEX.get(qtype, 0) + 1
        return None
    if _anchor_leaked(question, time_range):
        _REJECTED_LEAK[qtype] = _REJECTED_LEAK.get(qtype, 0) + 1
        return None
    rnd.shuffle(chosen)
    slot = _POS_COUNTER.get(qtype, 0) % n_opt
    _POS_COUNTER[qtype] = _POS_COUNTER.get(qtype, 0) + 1
    pool = list(chosen)
    pool.insert(slot, correct)
    idx = slot
    letters = [chr(ord("A") + i) for i in range(len(pool))]
    spec = TAXONOMY["types"][qtype]
    anchor = {
        "macro_id": macro_id,
        "time_range": [float(time_range[0]), float(time_range[1])],
    }
    if extra_macro_ids and len(extra_macro_ids) > 1:
        anchor["macro_ids"] = list(extra_macro_ids)
    if micro_ids:
        ids = [str(m) for m in micro_ids if m]
        if ids:
            anchor["micro_ids"] = ids
    if micro_time_range and len(micro_time_range) == 2:
        try:
            anchor["micro_time_range"] = [
                float(micro_time_range[0]), float(micro_time_range[1])
            ]
        except (TypeError, ValueError):
            pass
    return {
        "qid": f"{mem.video_key}:{qtype}:{rnd.randrange(10**9):09d}",
        "video_key": mem.video_key,
        "type": qtype,
        "videomme_task_type": spec["videomme_task_type"],
        "cgbench_class": spec["cgbench_class"],
        "source": "structural",
        "lang": "en",
        "granularity": granularity,
        "question": question,
        "options": [f"{letters[i]}. {p}" for i, p in enumerate(pool)],
        "answer": letters[idx],
        "anchor": anchor,
        "n_options": len(pool),
    }

def _split_option(opt: str) -> tuple[str, str]:
    m = re.match(r"^\s*([A-Z])\.\s*(.*)$", opt, re.S)
    return (m.group(1), m.group(2).strip()) if m else ("", opt.strip())

def apply_llm_rewrite(qs: list[dict], g: dict, batch_size: int | None = None) -> list[dict]:
    import question_rewrite as qr

    rc = qr.cfg_of(g)
    if not rc["enabled"] or not qs:
        return qs
    bs = int(batch_size or rc["batch_size"])
    if bs <= 0:
        return qs

    n_opt_min = int(g["questioner"].get("min_option_len", 1))
    max_diff = float(g["questioner"].get("max_lexical_overlap_diff", 0.5))
    cfg = load_cfg()

    batches: list[tuple[list[dict], list[tuple]]] = []
    for start in range(0, len(qs), bs):
        items, meta = [], []
        for q in qs[start:start + bs]:
            letters = [_split_option(o)[0] for o in q["options"]]
            texts = [_split_option(o)[1] for o in q["options"]]
            try:
                ai = letters.index(q["answer"])
            except ValueError:
                continue
            items.append({
                "question": q["question"],
                "correct": texts[ai],
                "distractors": [t for i, t in enumerate(texts) if i != ai],
            })
            meta.append((q, letters, texts, ai))
        if items:
            batches.append((items, meta))

    if not batches:
        return qs

    workers = max(1, int(os.environ.get("VEH_REWRITE_WORKERS", "8")))
    results: list[list[dict] | None] = [None] * len(batches)

    def _one(idx: int) -> None:
        try:
            results[idx] = qr.rewrite_batch(batches[idx][0], g, cfg)
        except Exception:
            results[idx] = None

    if workers == 1 or len(batches) == 1:
        for i in range(len(batches)):
            _one(i)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(_one, range(len(batches))))

    for (items, meta), out in zip(batches, results):
        if out is None:
            qr.bump("llm_failed")
            continue
        for (q, letters, texts, ai), new in zip(meta, out):
            new_q = (new.get("question") or q["question"]).strip()
            new_ds = list(new.get("distractors") or [])
            merged = list(texts)
            if len(new_ds) >= len(texts) - 1:
                it = iter(new_ds)
                for i in range(len(merged)):
                    if i != ai:
                        merged[i] = str(next(it)).strip() or merged[i]

            tr = ((q.get("anchor") or {}).get("time_range") or [0.0, 0.0])
            distract = [t for i, t in enumerate(merged) if i != ai]
            if (_distinct(merged, n_opt_min)
                    and not _lex_shortcut(new_q, merged[ai], distract, max_diff)
                    and not _anchor_leaked(new_q, tr)):
                q["question"] = new_q
                q["options"] = [f"{letters[i]}. {t}" for i, t in enumerate(merged)]
                q["llm_rewritten"] = True
            else:
                qr.bump("rejected_by_gate")
    return qs

def _tpl(qtype: str, rnd: random.Random, **kw) -> str:
    t = rnd.choice(TAXONOMY["types"][qtype]["templates"])
    try:
        return t.format(**kw)
    except KeyError as exc:
        raise RuntimeError(
            f"{qtype}: the question template needs parameter {exc}, but _tpl was called without it."
            f"\n  template: {t!r}\n  passed: {sorted(kw)}"
            f"\n  when changing the placeholders in question_taxonomy.json, g_{qtype} must be updated to match."
        ) from exc

def _evs(mem: Memory, mid: str) -> list[dict]:
    return [
        e
        for e in mem.visual_fields(mid)["events"]
        if isinstance(e.get("time_range"), list) and len(e["time_range"]) == 2 and e.get("description")
    ]

def g_object_reasoning(mem, mid, g, rnd, W):
    form = _Q["form"].get("object_reasoning", "cooccurrence")

    if form == "description":
        ents = [e for e in mem.visual_fields(mid)["entities"] if e.get("name") and e.get("description")]
        if not ents:
            return []
        e = rnd.choice(ents)
        correct = _clean(e["description"])
        pool = _pool_typed(
            mem, mid, "entities", lambda x: x.get("description", ""), W,
            "entity_type", e.get("entity_type"),
        )
        q = _tpl("object_reasoning", rnd, entity=e["name"])
        return _one(_mk(mem, "object_reasoning", q, correct, pool, mid,
                        mem.macro(mid)["time_range"], rnd, g))

    ents = [e for e in mem.visual_fields(mid)["entities"] if (e.get("name") or "").strip()]
    if len(ents) < 2:
        return []
    a, b = rnd.sample(ents, 2)
    anchor_name = _clean(a["name"], 60)
    correct = _clean(b["name"], 60)
    if not anchor_name or not correct or anchor_name.lower() == correct.lower():
        return []

    here = {_clean(e.get("name", ""), 60).lower() for e in ents}
    pool = [
        s
        for s in _pool_typed(mem, mid, "entities", lambda x: x.get("name", ""), W,
                             "entity_type", b.get("entity_type"))
        if s.lower() not in here
    ]
    tr = mem.macro(mid)["time_range"]
    q = (
        f"According to the video, which of the following appears in the same segment "
        f"as {anchor_name}?"
    )
    return _one(_mk(mem, "object_reasoning", q, correct, pool, mid, tr, rnd, g, min_opt_len=3))

def g_action_reasoning(mem, mid, g, rnd, W):
    def edges_of(macro_id):
        return [
            x for x in mem.visual_fields(macro_id)["edges"]
            if (x.get("description") or "").strip()
        ]

    all_mids = mem.all_macro_ids()
    causal_local = [x for x in edges_of(mid)
                    if (x.get("relation_type") or "").upper() == "CAUSAL"]
    src_mid, causal = mid, causal_local
    if not causal:
        for omid in sorted(all_mids, key=lambda m: abs(all_mids.index(m) - all_mids.index(mid))):
            c = [x for x in edges_of(omid)
                 if (x.get("relation_type") or "").upper() == "CAUSAL"]
            if c:
                src_mid, causal = omid, c
                break

    if causal:
        edge = rnd.choice(causal)
        correct = _clean(edge["description"])

        i = all_mids.index(src_mid)
        ordered = [src_mid]
        for off in range(1, max(len(all_mids), 2)):
            for j in (i - off, i + off):
                if 0 <= j < len(all_mids):
                    ordered.append(all_mids[j])
        pool: list[str] = []
        for omid in ordered:
            for x in edges_of(omid):
                d = _clean(x["description"])
                if d.lower() != correct.lower() and d not in pool:
                    pool.append(d)
            if len(pool) >= g["questioner"]["n_options"] * 3:
                break

        by_id: dict[str, str] = {}
        for omid in all_mids:
            vf = mem.visual_fields(omid)
            for ev in vf["events"]:
                k = ev.get("event_id") or ev.get("id")
                if k and ev.get("description"):
                    by_id[str(k)] = _clean(ev["description"], 90)
            for en in vf["entities"]:
                k = en.get("entity_id") or en.get("id")
                if k and en.get("name"):
                    by_id[str(k)] = _clean(en["name"], 60)
        src = by_id.get(str(edge.get("source_id") or ""), "")
        tgt = by_id.get(str(edge.get("target_id") or ""), "")
        if src and tgt:
            cue = f"{src} → {tgt}"
        elif src or tgt:
            cue = src or tgt
        else:
            evs_src = _evs(mem, src_mid)
            cue = _clean(rnd.choice(evs_src)["description"], 120) if evs_src else ""

        if cue:
            q = _tpl("action_reasoning", rnd, event=cue)
            tr = mem.macro(src_mid)["time_range"]
            built = _mk(mem, "action_reasoning", q, correct, pool, src_mid, tr, rnd, g)
            if built:
                return [built]

    evs = _evs(mem, mid)
    if len(evs) < 2:
        return []
    evs.sort(key=lambda e: float(e["time_range"][0]))
    i = rnd.randrange(len(evs) - 1)
    cur, nxt = evs[i], evs[i + 1]
    cue = _clean(cur["description"], 120)
    correct = _clean(nxt["description"])
    pool = _pool_typed(
        mem, mid, "events", lambda x: x.get("description", ""), W,
        "event_type", nxt.get("event_type"),
    )
    q = _tpl("action_reasoning", rnd, event=cue,
             subject=cur.get("subject", "the subject"), action="act")
    return _one(
        _mk(mem, "action_reasoning", q, correct, pool, mid,
            [float(cur["time_range"][0]), float(nxt["time_range"][1])], rnd, g)
    )

def g_info_synopsis(mem, mid, g, rnd, W):
    m = mem.macro(mid)
    sid = m.get("super_id")
    se = next((s for s in mem.super_events if s.get("super_id") == sid), None)
    if not se:
        return []
    subs = se.get("sub_macro_ids") or []
    if subs and mid != subs[0]:
        return []

    n_opt = g["questioner"]["n_options"]
    all_mids = mem.all_macro_ids()
    out = []

    sub_mids = [x for x in subs if x in all_mids] or [mid]
    other_labels = [
        _clean(s.get("label", ""), 90)
        for s in mem.super_events
        if s.get("super_id") != sid and len(_clean(s.get("label", ""), 90)) >= 8
    ]
    se_label = _clean(se.get("label", ""), 90)
    if len(sub_mids) >= 2 and len(se_label) >= 8 and len(other_labels) >= n_opt - 1:
        anchors: list[str] = []
        for cmid in sub_mids:
            vf = mem.visual_fields(cmid)
            ents = [e for e in vf["entities"] if (e.get("name") or "").strip()]
            evts = [e for e in vf["events"] if (e.get("description") or "").strip()]
            if ents:
                anchors.append(_clean(rnd.choice(ents)["name"], 50))
            elif evts:
                anchors.append(_clean(rnd.choice(evts)["description"], 60))
        anchors = [a for a in anchors if a]
        if len(anchors) >= 2:
            listed = ", ".join(anchors[:-1]) + f", and {anchors[-1]}"
            tr = se.get("time_range") or m["time_range"]
            q = f"Which of the following segments covers all of the following elements: {listed}?"
            out += _one(_mk(mem, "info_synopsis", q, se_label, other_labels, mid, tr, rnd, g))

    label = _clean(se.get("label", ""), 90)
    summary = _clean(se.get("summary", ""), 240)
    import re as _re
    is_placeholder = not label or bool(_re.fullmatch(r"(scene|macro|segment)[ _]\d+", label, _re.I))
    if _INFO_SYNOPSIS_FORM_B and not is_placeholder and len(label) >= 8:
        others = [s for s in mem.super_events if s.get("super_id") != sid]
        if len(others) >= n_opt - 1:
            tr = se.get("time_range") or m["time_range"]
            pool = [_clean(s.get("label", ""), 90) for s in others if len(_clean(s.get("label",""),90)) >= 8]
            q = f"Which of the following best describes the segment from {fmt_time(tr[0])} to {fmt_time(tr[1])}?"
            out += _one(_mk(mem, "info_synopsis", q, label, pool, mid, tr, rnd, g))

    if mid == all_mids[0] and mem.root:
        root_desc = _clean(mem.root.get("description", ""), 240)
        if len(root_desc) >= 80:
            super_descs = [
                _clean(s.get("description", ""), 240)
                for s in mem.super_events
                if len(_clean(s.get("description", ""), 240)) >= 20
            ]
            if len(super_descs) >= n_opt - 1:
                full = [mem.macro(x)["time_range"] for x in all_mids]
                span = [full[0][0], full[-1][1]]
                out += _one(
                    _mk(mem, "info_synopsis", "What is the video mainly about?",
                        root_desc, super_descs, mid, span, rnd, g)
                )
    return out

def g_temporal_reasoning(mem, mid, g, rnd, W):
    evs = _evs(mem, mid)
    if len(evs) < 2:
        return []
    evs.sort(key=lambda e: float(e["time_range"][0]))
    gap = g["questioner"]["min_gap_sec"]
    pairs = [
        (evs[i], evs[i + 1])
        for i in range(len(evs) - 1)
        if float(evs[i + 1]["time_range"][0]) - float(evs[i]["time_range"][0]) >= gap
    ]
    if not pairs:
        return []
    cur, nxt = rnd.choice(pairs)
    cue, correct = _clean(cur["description"], 120), _clean(nxt["description"])
    pool = _pool(mem, mid, "events", lambda x: x.get("description", ""), W)
    q = _tpl("temporal_reasoning", rnd, event=cue, subject=cur.get("subject", "the subject"), action="act")
    span = [float(cur["time_range"][0]), float(nxt["time_range"][1])]
    return _one(
        _mk(
            mem, "temporal_reasoning", q, correct, pool, mid, span, rnd, g,
            micro_ids=[cur.get("event_id"), nxt.get("event_id")],
            micro_time_range=span,
        )
    )

def _micro_tpl(needle: str, **kw) -> str:
    spec = TAXONOMY["types"]["temporal_reasoning"]
    for pool in (spec.get("templates_micro") or [], spec.get("templates") or []):
        for t in pool:
            if needle in t:
                try:
                    return t.format(**kw)
                except (KeyError, IndexError):
                    return t
    return ""

def g_micro_event(mem, mid, g, rnd, W):
    evs = _evs(mem, mid)
    if len(evs) < 3:
        return []
    evs.sort(key=lambda e: float(e["time_range"][0]))
    i = rnd.randrange(len(evs))
    if i == 0:
        needle = "right after"
    elif i == len(evs) - 1:
        needle = "right before"
    else:
        needle = "right after" if rnd.random() < 0.5 else "right before"
    if needle == "right after":
        cue, tgt = evs[i], evs[i + 1]
    else:
        cue, tgt = evs[i], evs[i - 1]
    stem = _micro_tpl(
        needle, event=_clean(cue["description"], 120),
        subject=cue.get("subject", "the subject"), action="act",
    )
    if not stem:
        return []
    correct = _clean(tgt["description"])
    low = correct.lower()
    same = [
        _clean(e["description"]) for e in evs
        if e is not cue and e is not tgt and _clean(e["description"]).lower() != low
    ]
    rnd.shuffle(same)
    near = [
        s for s in _pool(mem, mid, "events", lambda x: x.get("description", ""), W)
        if s.lower() != low and s.lower() not in {x.lower() for x in same}
    ]
    return _one(_mk(
        mem, "temporal_reasoning", stem, correct, same + near, mid,
        mem.macro(mid)["time_range"], rnd, g,
        micro_ids=[tgt.get("event_id")], micro_time_range=tgt["time_range"],
        granularity="micro",
    ))

def g_action_recognition(mem, mid, g, rnd, W):
    evs = _evs(mem, mid)
    if not evs:
        return []
    ev = rnd.choice(evs)
    correct = _clean(ev.get("action") or ev["description"], 120)
    pool = _pool_typed(
        mem, mid, "events", lambda x: x.get("action") or x.get("description", ""), W,
        "event_type", ev.get("event_type"),
    )
    tr = mem.macro(mid)["time_range"]
    q = _tpl("action_recognition", rnd, subject=ev.get("subject") or "the main subject",
             t0=fmt_time(tr[0]), t1=fmt_time(tr[1]))
    return _one(_mk(mem, "action_recognition", q, correct, pool, mid, ev["time_range"], rnd, g,
                    micro_ids=[ev.get("event_id")], micro_time_range=ev["time_range"]))

def g_object_recognition(mem, mid, g, rnd, W):
    mids = _window_mids(mem, mid, span=1)
    ents = [e for e in mem.visual_fields(mid)["entities"] if (e.get("name") or "").strip()]
    if not ents:
        return []
    tgt = rnd.choice(ents)
    correct = _clean(tgt["name"], 60)
    if not correct:
        return []
    here = set()
    for m in mids:
        for e in mem.visual_fields(m)["entities"]:
            nm = _clean(e.get("name", ""), 60).lower()
            if nm:
                here.add(nm)
    pool = [
        s
        for s in _pool_typed(mem, mid, "entities", lambda x: x.get("name", ""), W,
                             "entity_type", tgt.get("entity_type"))
        if s.lower() not in here
    ]
    trs = [mem.macro(m)["time_range"] for m in mids]
    win = [min(t[0] for t in trs), max(t[1] for t in trs)]
    q = _tpl("object_recognition", rnd, t0=fmt_time(win[0]), t1=fmt_time(win[1]))
    return _one(_mk(mem, "object_recognition", q, correct, pool, mid,
                    mem.macro(mid)["time_range"], rnd, g, min_opt_len=3))

def _window_mids(mem, mid: str, span: int = 1) -> list[str]:
    all_mids = mem.all_macro_ids()
    if mid not in all_mids:
        return [mid]
    i = all_mids.index(mid)
    lo, hi = max(0, i - span), min(len(all_mids), i + span + 1)
    return all_mids[lo:hi]

def g_counting(mem, mid, g, rnd, W):
    mids = _window_mids(mem, mid, span=1)
    by_type: dict[str, set] = {}
    for m in mids:
        for e in mem.visual_fields(m)["entities"]:
            t = (e.get("entity_type") or "").upper()
            nm = (e.get("name") or "").strip().lower()
            if t and nm:
                by_type.setdefault(t, set()).add(nm)
    cands = [(t, v) for t, v in by_type.items() if 2 <= len(v) <= 12]
    if not cands:
        return []
    et, items = rnd.choice(sorted(cands))
    n = len(items)
    trs = [mem.macro(m)["time_range"] for m in mids]
    win = [min(t[0] for t in trs), max(t[1] for t in trs)]
    alt = [x for x in range(max(1, n - 3), n + 4) if x != n]
    rnd.shuffle(alt)
    q = _tpl("counting", rnd, category=f"{et.lower()} entities", event_type=et.lower(),
             t0=fmt_time(win[0]), t1=fmt_time(win[1]))
    return _one(_mk(mem, "counting", q, str(n), [str(x) for x in alt], mid,
                    mem.macro(mid)["time_range"], rnd, g, min_opt_len=1,
                    extra_macro_ids=mids))

def _attr_text(item: dict) -> str:
    a = item.get("attributes")
    if isinstance(a, dict) and a:
        parts = [f"{str(k).replace('_', ' ')}: {v}" for k, v in a.items() if v not in (None, "", [], {})]
        if parts:
            return "; ".join(parts)
    if isinstance(a, list) and a:
        return "; ".join(str(x) for x in a if x)
    return str(item.get("description") or "")

def g_attribute_perception(mem, mid, g, rnd, W):
    ents = [
        e
        for e in mem.visual_fields(mid)["entities"]
        if e.get("name") and (e.get("attributes") or e.get("description"))
    ]
    if not ents:
        return []
    e = rnd.choice(ents)
    correct = _clean(_attr_text(e))
    pool = _pool_typed(mem, mid, "entities", _attr_text, W, "entity_type", e.get("entity_type"))
    q = _tpl("attribute_perception", rnd, entity=e["name"])
    return _one(_mk(mem, "attribute_perception", q, correct, pool, mid, mem.macro(mid)["time_range"], rnd, g))

def g_ocr(mem, mid, g, rnd, W):
    texts = [t for t in mem.visual_fields(mid)["ocr"] if len((t.get("text") or "").strip()) >= 3]
    if not texts:
        return []
    tgt = rnd.choice(texts)
    correct = _clean(tgt["text"], 80)
    here = {_clean(t.get("text", ""), 80).lower() for t in texts}
    pool = [s for s in _pool(mem, mid, "ocr", lambda x: x.get("text", ""), W) if s.lower() not in here]
    tr = tgt.get("time_range") or mem.macro(mid)["time_range"]
    q = _tpl("ocr", rnd, t0=fmt_time(tr[0]))
    return _one(_mk(mem, "ocr", q, correct, pool, mid, tr, rnd, g, min_opt_len=3,
                    micro_ids=[tgt.get("text_id")], micro_time_range=tr))

def g_spatial(mem, mid, g, rnd, W):
    cands = [
        e
        for e in mem.visual_fields(mid)["entities"]
        if e.get("name") and _SPATIAL_RE.search(str(e.get("description") or ""))
    ]
    if not cands:
        return []
    e = rnd.choice(cands)
    correct = _clean(e["description"])
    pool = [
        s
        for s in _pool(mem, mid, "entities", lambda x: x.get("description", ""), W)
        if _SPATIAL_RE.search(s)
    ]
    q = _tpl("spatial", rnd, entity=e["name"])
    return _one(_mk(mem, "spatial", q, correct, pool, mid, mem.macro(mid)["time_range"], rnd, g))

def g_hallucination(mem, mid, g, rnd, W):
    evs = _evs(mem, mid)
    if len(evs) < 3:
        return []
    n_opt = g["questioner"]["n_options"]
    truths = [_clean(e["description"]) for e in rnd.sample(evs, min(len(evs), n_opt - 1))]
    truths = [t for t in truths if t]
    if len(truths) < n_opt - 1:
        return []
    here = {t.lower() for t in truths}
    pool = [s for s in _pool(mem, mid, "events", lambda x: x.get("description", ""), W) if s.lower() not in here]
    if not pool:
        return []
    false_stmt = pool[0]
    tr = mem.macro(mid)["time_range"]
    q = (
        f"Regarding the segment from {fmt_time(tr[0])} to {fmt_time(tr[1])}, "
        + _tpl("hallucination", rnd).lower().replace("according to the video", "according to the video")
    )
    return _one(_mk(mem, "hallucination", q, false_stmt, truths, mid, tr, rnd, g))

def _one(q):
    return [q] if q else []

GENERATORS: dict[str, Callable] = {
    "object_reasoning": g_object_reasoning,
    "action_reasoning": g_action_reasoning,
    "info_synopsis": g_info_synopsis,
    "temporal_reasoning": g_temporal_reasoning,
    "action_recognition": g_action_recognition,
    "object_recognition": g_object_recognition,
    "counting": g_counting,
    "attribute_perception": g_attribute_perception,
    "ocr": g_ocr,
    "spatial": g_spatial,
    "hallucination": g_hallucination,
}

def _derive_rng(seed: int, *parts) -> random.Random:
    key = "|".join(str(p) for p in (seed, *parts))
    digest = hashlib.blake2b(key.encode(), digest_size=8).digest()
    return random.Random(int.from_bytes(digest, "big"))

def build_questions(
    video_keys: list[str], g: dict, target_n: int, seed: int = 0, tries: int | None = None
) -> list[dict]:
    apply_genome(g)
    W = _Q["adjacent_window"]
    tries = g["questioner"].get("tries", 3) if tries is None else tries
    active = [t for t in GENERATORS if type_on(t)]
    if not active:
        raise SystemExit("All question types are disabled in genome.questioner.type_enabled")
    pool: dict[str, list[dict]] = {t: [] for t in active}

    micro_cfg = g.get("questioner", {}).get("micro_questions") or {}
    micro_on = bool(micro_cfg.get("enabled", True))
    micro_per_macro = max(0, int(micro_cfg.get("max_per_macro", 2)))
    if micro_on and "temporal_reasoning" not in active:
        micro_on = False

    for vk in video_keys:
        mem = Memory(vk)
        if not g["questioner"]["use_asr"]:
            for m in mem.macros:
                m.pop("asr_text", None)
        for mid in mem.all_macro_ids():
            for qtype in active:
                for attempt in range(tries):
                    try:
                        qrnd = _derive_rng(seed, vk, mid, qtype, attempt)
                        for q in GENERATORS[qtype](mem, mid, g, qrnd, W):
                            pool[qtype].append(q)
                    except Exception as exc:
                        break
            if micro_on:
                for attempt in range(micro_per_macro):
                    try:
                        mrnd = _derive_rng(seed, vk, mid, "micro_event", attempt)
                        for q in g_micro_event(mem, mid, g, mrnd, W):
                            pool["temporal_reasoning"].append(q)
                    except Exception as exc:
                        break

    for t in pool:
        seen, uniq = set(), []
        for q in pool[t]:
            k = (q["question"], tuple(q["options"]))
            if k not in seen:
                seen.add(k)
                uniq.append(q)
        pool[t] = uniq

    disabled = [t for t in GENERATORS if not type_on(t)]
    if disabled:
        pass

    if not target_n:
        out = [q for v in pool.values() for q in v]
        _derive_rng(seed, "final_order").shuffle(out)
        return out

    total_w = sum(type_weight(t) for t in active) or 1.0
    out, deficit = [], 0
    quotas = {t: max(1, round(target_n * type_weight(t) / total_w)) for t in active}
    for t in active:
        _derive_rng(seed, "sample", t).shuffle(pool[t])
        take = pool[t][: quotas[t]]
        out.extend(take)
        deficit += quotas[t] - len(take)
    if deficit > 0:
        rest = [q for t in active for q in pool[t][quotas[t]:]]
        _derive_rng(seed, "deficit").shuffle(rest)
        out.extend(rest[:deficit])
    _derive_rng(seed, "final_order").shuffle(out)
    return out

def attach_route_fields(qs: list[dict], genome: dict | None = None) -> list[dict]:
    import video_routing as vr

    lo, hi = vr.routing_bounds(genome)
    vks = sorted({q.get("video_key") for q in qs if q.get("video_key")})
    path_of: dict[str, str] = {}
    for vk in vks:
        try:
            p = Memory(vk).video_path or ""
        except Exception:
            p = ""
        if p and Path(p).is_file():
            path_of[vk] = p
    vr.probe_missing(list(path_of.values()), progress=True)

    n_missing = 0
    for q in qs:
        vk = q.get("video_key") or ""
        p = path_of.get(vk, "")
        dur, src = vr.duration_for(p or None, None, allow_probe=False)
        q["video_path"] = p
        q["video_duration_sec"] = dur
        q["video_duration_source"] = src
        q["route"] = vr.route_for(dur, lo, hi)
        if dur is None:
            n_missing += 1
    if n_missing:
        pass
    return qs

def _print_micro_stats(qs: list[dict]) -> None:
    import statistics

    def _med(xs: list[float]) -> float:
        return statistics.median(xs) if xs else float("nan")

    micro_q = [q for q in qs if q.get("granularity") == "micro"]
    anchored = [q for q in qs if (q.get("anchor") or {}).get("micro_ids")]
    macro_w: list[float] = []
    micro_w: list[float] = []
    for q in qs:
        a = q.get("anchor") or {}
        tr, mtr = a.get("time_range"), a.get("micro_time_range")
        if tr and len(tr) == 2:
            macro_w.append(float(tr[1]) - float(tr[0]))
        if mtr and len(mtr) == 2:
            micro_w.append(float(mtr[1]) - float(mtr[0]))
    if micro_w:
        pass
    if micro_q:
        pass

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="*", default=None)
    ap.add_argument("--n", type=int, default=0, help="target question count (quota by Video-MME distribution), 0 = keep all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tries", type=int, default=None,
                    help="attempts per macro per type (default: genome.questioner.tries)")
    ap.add_argument("--allow-test-videos", action="store_true", help="smoke test only; output is not reportable")
    ap.add_argument("--rewrite-batch", type=int, default=None,
                    help="batch size for LLM rewriting; without it the genome's "
                         "questioner.llm_rewrite.batch_size is used (default 10)")
    ap.add_argument("--no-rewrite", action="store_true",
                    help="disable LLM rewriting for this run (the genome is not modified)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    g = load_genome()
    data_split = load_data_split()
    vks = args.videos or data_split["evolve_source"]["videos"] or available_videos()
    if not vks:
        raise SystemExit("No memory available; run build_memory.sh first")

    leakage_check(vks, args.allow_test_videos)

    qs = build_questions(vks, g, args.n, args.seed, args.tries)
    if not args.no_rewrite:
        qs = apply_llm_rewrite(qs, g, args.rewrite_batch)
    qs = assign_split(qs, g["split"]["evolve_ratio"], g["split"]["seed"])
    qs = attach_route_fields(qs, g)

    from collections import Counter

    c = Counter(q["type"] for q in qs)
    for t in GENERATORS:
        spec = TAXONOMY["types"][t]
        if not type_on(t):
            reason = "quota=0" if (spec.get("weight_gen") == 0) else "genome"
            continue
        gen = type_weight(t)
        tgt = target_weight(t)
        act = c.get(t, 0) / max(1, len(qs))
        flag = "  <-- under-generated on purpose" if gen + 1e-9 < tgt else ""

    gen_sum = sum(type_weight(t) for t in GENERATORS if type_on(t))
    tgt_covered = sum(target_weight(t) for t in GENERATORS if type_on(t))
    _print_micro_stats(qs)

    for label, counter, hint in (
        ("lexical shortcut", _REJECTED_LEX, "the correct option overlaps the question wording far more than the distractors do"),
        ("leaked answer position", _REJECTED_LEAK, "the interval stated in the question equals the anchor interval"),
    ):
        if counter:
            total = sum(counter.values())
            for t, k in sorted(counter.items(), key=lambda kv: -kv[1]):
                pass

    import question_rewrite as _qr
    _rw = _qr.format_stats()
    if _rw:
        n_rw = _qr.STATS["question_rewritten"] + _qr.STATS["distractors_rewritten"]
        n_rej = _qr.STATS["rejected_by_gate"]
        if n_rej and n_rej > max(3, 0.2 * max(1, n_rw)):
            pass

    p = save_questions(qs, args.out)
    if qs:
        for q in qs[:2]:
            for o in q["options"]:
                pass

if __name__ == "__main__":
    main()
