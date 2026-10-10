from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from pathlib import Path

from common import STORE, call_llm, load_cfg

STATS: dict[str, int] = {
    "asked": 0,
    "cache_hit": 0,
    "llm_calls": 0,
    "llm_failed": 0,
    "question_rewritten": 0,
    "distractors_rewritten": 0,
    "letter_stripped": 0,
    "rejected_by_gate": 0,
}
_stats_lock = threading.Lock()


def bump(key: str, n: int = 1) -> None:
    with _stats_lock:
        STATS[key] = STATS.get(key, 0) + n


def cfg_of(g: dict) -> dict:
    q = (g or {}).get("questioner") or {}
    c = dict(q.get("llm_rewrite") or {})
    c.setdefault("enabled", True)
    c.setdefault("rewrite_question", True)
    c.setdefault("rewrite_distractors", True)
    c.setdefault("batch_size", 10)
    c.setdefault("max_tokens", 2048)
    c.setdefault("timeout_s", 90)
    return c


def _cache_dir() -> Path:
    d = Path(os.environ.get("VEH_REWRITE_CACHE") or (STORE / "runs" / "rewrite_cache"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _key(payload: dict) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def _cache_get(key: str) -> dict | None:
    p = _cache_dir() / f"{key}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _cache_put(key: str, value: dict) -> None:
    p = _cache_dir() / f"{key}.json"
    try:
        tmp = p.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        tmp.replace(p)
    except Exception:
        pass


PROMPT_VERSION = "v1"

_TMPL = """You rewrite multiple-choice questions about videos. You never change what
the correct answer means.

You are given {n} multiple-choice items generated from a video memory graph.

For each item, do exactly the following.

{task_q}
{task_d}

## HARD RULES -- violating any of these makes the item unusable

1. NEVER change the meaning of `correct`. Return it untouched.
2. NEVER put a timestamp, time range, or phrase like "in the first minute" /
   "at the beginning" into the question. The question must not reveal WHERE in
   the video the answer is. This is the single most common failure.
3. Do NOT copy long phrases from `correct` into the question. If the question
   and the correct answer share most of their content words, the item is thrown
   away.
4. Distractors must describe things that DID NOT happen in the video.
5. Distractors must NOT be paraphrases, synonyms, or near-synonyms of `correct`
   or of each other. "opened the fridge" vs "pulled the fridge door open" is
   FORBIDDEN -- that makes two correct answers and silently poisons the dataset.
   Each distractor must be a genuinely DIFFERENT event or fact.
6. Keep every distractor close to `correct` in length and grammatical form, so
   the answer cannot be spotted by shape alone.
7. Keep the original language of the item.
8. The question must NOT contain the answer letter. Never append "B" or "(C)" or
   "Answer: D" to the stem. Output the stem only -- the letters are added later.

## Input

{items}

## Output

Return ONLY a JSON array, one object per item, same order, no prose:

[{{"i": 0, "question": "...", "distractors": ["...", "..."]}}]

{omit_note}"""

_TASK_Q = ("- Rewrite `question` so it reads naturally instead of like a filled "
           "template. Keep the meaning identical.")
_TASK_D = ("- Rewrite every entry of `distractors` so they are plausible but "
           "clearly wrong, following rules 4-6.")


def _build_prompt(items: list[dict], do_q: bool, do_d: bool) -> str:
    omit = []
    if not do_q:
        omit.append('Omit the "question" field; it must not be changed.')
    if not do_d:
        omit.append('Omit the "distractors" field; they must not be changed.')
    return _TMPL.format(
        n=len(items),
        task_q=_TASK_Q if do_q else "",
        task_d=_TASK_D if do_d else "",
        items=json.dumps(items, ensure_ascii=False, indent=2),
        omit_note="\n".join(omit),
    )


_TRAILING_LETTER = re.compile(
    r"""(?:\bAnswer\b\s*:?\s*)?
        [\s\-–—:,.]*
        [(\[]?\s*
        ([A-H])
        \s*[)\].]?\s*$
    """,
    re.I | re.X)


def strip_leaked_letter(question: str, n_opt: int = 8) -> str:
    if not question:
        return question
    m = _TRAILING_LETTER.search(question)
    if not m:
        return question
    idx = ord(m.group(1).upper()) - ord("A")
    if idx >= max(2, n_opt):
        return question
    cleaned = question[:m.start()].rstrip()
    return cleaned if len(cleaned) >= 12 else question


def _extract_json_array(text: str) -> list | None:
    if not text:
        return None
    t = text.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", t, re.S)
    if fence:
        t = fence.group(1).strip()
    i, j = t.find("["), t.rfind("]")
    if i < 0 or j <= i:
        return None
    try:
        out = json.loads(t[i:j + 1])
        return out if isinstance(out, list) else None
    except Exception:
        return None


def rewrite_batch(items: list[dict], g: dict, cfg: dict | None = None) -> list[dict]:
    if not items:
        return []
    rc = cfg_of(g)
    if not rc["enabled"] or not (rc["rewrite_question"] or rc["rewrite_distractors"]):
        return list(items)

    bump("asked", len(items))
    key = _key({
        "v": PROMPT_VERSION,
        "q": rc["rewrite_question"],
        "d": rc["rewrite_distractors"],
        "items": items,
    })
    cached = _cache_get(key)
    if cached is not None:
        bump("cache_hit", len(items))
        return _merge(items, cached.get("out") or [], rc)

    payload = [{"i": n, "question": it.get("question", ""),
                "correct": it.get("correct", ""),
                "distractors": list(it.get("distractors") or [])}
               for n, it in enumerate(items)]

    cfg = cfg or load_cfg()
    bump("llm_calls")
    try:
        text = call_llm(
            _build_prompt(payload, rc["rewrite_question"], rc["rewrite_distractors"]),
            cfg,
            max_tokens=int(rc["max_tokens"]),
            role="questioner_rewrite",
        )
    except Exception:
        bump("llm_failed")
        return list(items)

    parsed = _extract_json_array(text if isinstance(text, str) else "")
    if parsed is None:
        bump("llm_failed")
        return list(items)

    _cache_put(key, {"out": parsed})
    return _merge(items, parsed, rc)


def _merge(items: list[dict], parsed: list, rc: dict) -> list[dict]:
    by_i: dict[int, dict] = {}
    for row in parsed or []:
        if isinstance(row, dict) and isinstance(row.get("i"), int):
            by_i[row["i"]] = row

    out: list[dict] = []
    for n, it in enumerate(items):
        row = by_i.get(n)
        new = dict(it)
        if not row:
            out.append(new)
            continue

        if rc["rewrite_question"]:
            q = row.get("question")
            if isinstance(q, str) and q.strip():
                n_opt = len(it.get("distractors") or []) + 1
                cleaned = strip_leaked_letter(q.strip(), n_opt)
                if cleaned != q.strip():
                    bump("letter_stripped")
                new["question"] = cleaned
                bump("question_rewritten")

        if rc["rewrite_distractors"]:
            ds = row.get("distractors")
            if isinstance(ds, list):
                clean = [str(d).strip() for d in ds if str(d).strip()]
                if len(clean) >= len(it.get("distractors") or []):
                    new["distractors"] = clean
                    bump("distractors_rewritten")

        new["correct"] = it.get("correct", "")
        out.append(new)
    return out


def format_stats() -> str:
    s = STATS
    if not s.get("asked"):
        return ""
    extra = (f"; stripped the answer letter leaked at the end of the question {s['letter_stripped']}"
             if s.get("letter_stripped") else "")
    return (f"      [llm-rewrite] requested {s['asked']}; cache hits {s['cache_hit']}; "
            f"{s['llm_calls']} calls ({s['llm_failed']} failed); "
            f"rewrote questions {s['question_rewritten']}, distractors {s['distractors_rewritten']}"
            f"{extra}; rejected by the gate after rewriting {s['rejected_by_gate']} (fell back to the original)")
