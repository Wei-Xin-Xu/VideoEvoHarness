from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path

from common import STORE

VEH_DIR = Path(__file__).resolve().parent

def enabled() -> bool:
    return str(os.environ.get("VEH_RESUME", "")).strip().lower() in ("1", "on", "true", "yes")


_src_hash: str | None = None
_lock = threading.Lock()

_ANSWER_PATH_MODULES = frozenset({
    "solve.py",
    "core_arm.py",
    "common.py",
    "core_chunked_qa.py",
    "memory_tools.py",
    "state_bundle.py",
    "strategy_forge.py",
    "tool_forge.py",
    "video_routing.py",
    "benchmark_audit.py",
    "llm_provider.py",
})

_PROMPT_DIR = "prompts"
_PROMPT_SUFFIX = ".txt"


def solver_source_hash() -> str:
    global _src_hash
    with _lock:
        if _src_hash is not None:
            return _src_hash
        h = hashlib.sha256()
        for name in sorted(_ANSWER_PATH_MODULES):
            p = VEH_DIR / name
            if not p.exists():
                h.update(name.encode())
                h.update(b"<missing>")
                continue
            h.update(name.encode())
            h.update(p.read_bytes())
        pdir = VEH_DIR / _PROMPT_DIR
        if pdir.is_dir():
            for p in sorted(pdir.rglob(f"*{_PROMPT_SUFFIX}")):
                h.update(str(p.relative_to(VEH_DIR)).encode())
                h.update(p.read_bytes())
        else:
            h.update(b"<no-prompts-dir>")
        try:
            import llm_provider
            h.update(llm_provider.model_tag().encode())
        except Exception as exc:
            h.update(f"<model-tag-unavailable:{type(exc).__name__}>".encode())
        _src_hash = h.hexdigest()[:16]
        return _src_hash


def fingerprint(g: dict) -> str:
    canon = json.dumps(g, ensure_ascii=False, sort_keys=True, default=str)
    h = hashlib.sha256(canon.encode("utf-8")).hexdigest()[:16]
    return f"{h}_{solver_source_hash()}"


def _path(fp: str) -> Path:
    return STORE / "runs" / "cache" / f"{fp}.jsonl"


_mem: dict[str, dict[str, dict]] = {}


def _load(fp: str) -> dict[str, dict]:
    if fp in _mem:
        return _mem[fp]
    out: dict[str, dict] = {}
    p = _path(fp)
    if p.exists():
        with p.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rec = row.get("rec")
                if not isinstance(rec, dict):
                    continue
                out[f"{row.get('tag')}\x00{rec.get('qid')}"] = rec
    _mem[fp] = out
    return out


def split(fp: str, tag: str, questions: list[dict]) -> tuple[list[dict], list[dict]]:
    if not enabled():
        return [], list(questions)
    have = _load(fp)
    hit, miss = [], []
    for q in questions:
        r = have.get(f"{tag}\x00{q.get('qid')}")
        if r is None:
            miss.append(q)
        else:
            hit.append(r)
    return hit, miss


def put(fp: str, tag: str, rec: dict, label: "str | None" = None) -> None:
    if not enabled():
        return
    qid = rec.get("qid")
    if qid is None:
        return
    p = _path(fp)
    p.parent.mkdir(parents=True, exist_ok=True)
    row = {"tag": tag, "rec": rec}
    if label:
        row["label"] = label
    line = json.dumps(row, ensure_ascii=False, default=str)
    with _lock:
        with p.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        _mem.setdefault(fp, {})[f"{tag}\x00{qid}"] = rec


def stats(fp: str) -> dict:
    have = _load(fp) if enabled() else {}
    tags: dict[str, int] = {}
    for k in have:
        t = k.split("\x00", 1)[0]
        tags[t] = tags.get(t, 0) + 1
    return {"enabled": enabled(), "fingerprint": fp, "n": len(have),
            "by_tag": tags, "path": str(_path(fp))}
