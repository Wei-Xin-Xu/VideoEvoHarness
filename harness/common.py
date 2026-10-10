from __future__ import annotations

import json
import os
import random
import re
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

VEH_DIR = Path(__file__).resolve().parent
ROOT = VEH_DIR.parent
MEMORY_ROOT = Path(os.environ.get("VEH_MEMORY_ROOT", ROOT / "memory")).expanduser().resolve()
STORE = Path(os.environ.get("VEH_STORE_DIR", VEH_DIR / "store")).expanduser().resolve()
CONFIG_FILE = Path(os.environ.get("VEH_CONFIG_FILE", ROOT / "qwen-mm-config")).expanduser().resolve()
GENOME_FILE = Path(os.environ.get("VEH_GENOME_PATH", VEH_DIR / "genome.json")).expanduser().resolve()
DATA_SPLIT_FILE = Path(os.environ.get("VEH_DATA_SPLIT_PATH", VEH_DIR / "data_split.json")).expanduser().resolve()
PARQUET = Path(os.environ.get(
    "VEH_VIDEOMME_PARQUET", ROOT / "data/videomme/videomme/test-00000-of-00001.parquet"
)).expanduser().resolve()
TOKEN_LOG_FILE = Path(os.environ["VEH_TOKEN_LOG"]).expanduser().resolve() if os.environ.get("VEH_TOKEN_LOG") else None

for _d in (STORE, STORE / "runs", STORE / "genomes", VEH_DIR / "prompts"):
    _d.mkdir(parents=True, exist_ok=True)


def adaptation_context() -> dict:
    try:
        ckpt = int(os.environ.get("VEH_CHECKPOINT") or 0)
    except ValueError:
        ckpt = 0
    return {
        "arm": os.environ.get("VEH_ARM") or "evolving",
        "adaptation_mode": os.environ.get("VEH_ADAPTATION_MODE") or "frozen",
        "checkpoint": ckpt,
    }


def load_cfg() -> dict[str, str]:
    cfg: dict[str, str] = {}
    for line in CONFIG_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            cfg[k.strip()] = v.strip()
    for key in list(cfg):
        if os.environ.get(key) is not None:
            cfg[key] = os.environ[key]
    for key in ("QWEN_MM_CHAT_TIMEOUT", "QWEN_MM_CONNECT_TIMEOUT", "QWEN_MM_MAX_RETRIES",
                "QWEN_MM_READ_TIMEOUT_RETRIES", "QWEN_MM_FIRST_BYTE_TIMEOUT",
                "QWEN_MM_CALL_BUDGET", "LLM_PROVIDER", "QWEN_MM_API_VL_MODEL"):
        if os.environ.get(key) is not None:
            cfg[key] = os.environ[key]
    return cfg


def load_genome(path: str | Path | None = None) -> dict:
    return json.loads(Path(path or GENOME_FILE).read_text(encoding="utf-8"))


def save_genome(g: dict, path: str | Path | None = None) -> None:
    Path(path or GENOME_FILE).write_text(
        json.dumps(g, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def snapshot_genome(g: dict) -> Path:
    p = STORE / "genomes" / f"genome_v{g['version']}.json"
    p.write_text(json.dumps(g, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


VERSION_AXES = ("solver", "questioner")


def version_axes(g: dict) -> dict:
    sv, qv = g.get("solver_version"), g.get("questioner_version")
    return {
        "version": g.get("version", 0),
        "solver_version": sv if sv is not None else 0,
        "questioner_version": qv if qv is not None else 0,
        "changed_by": g.get("changed_by"),
        "axes_known": sv is not None and qv is not None,
        "axes_backfilled_at_version": g.get("axes_backfilled_at_version"),
    }


def bump_version(g: dict, who: str) -> dict:
    if who not in VERSION_AXES:
        raise ValueError(f"bump_version(who=) must be one of {VERSION_AXES}, received {who!r}")
    axes = version_axes(g)
    if not axes["axes_known"]:
        g["axes_backfilled_at_version"] = g.get("version", 0)
    g["parent_version"] = g.get("version", 0)
    g["version"] = g.get("version", 0) + 1
    g["solver_version"] = axes["solver_version"] + (who == "solver")
    g["questioner_version"] = axes["questioner_version"] + (who == "questioner")
    g["changed_by"] = who
    return g


CAPABILITY_MODES = ("llm", "rule", "both", "off")

DEFAULT_CAPABILITIES: dict[str, str] = {
    "hyperparam": "llm",
    "skill": "llm",
    "tool_policy": "llm",
    "strategy_code": "llm",
    "tool_code": "llm",
    "questioner": "rule",
}

CAPABILITY_PROPOSERS: dict[str, tuple[str, ...]] = {
    "hyperparam": ("llm", "rule"),
    "skill": ("llm", "rule"),
    "tool_policy": ("llm", "rule"),
    "strategy_code": ("llm",),
    "tool_code": ("llm",),
    "questioner": ("rule",),
}
CAPABILITY_KINDS = tuple(DEFAULT_CAPABILITIES)


def capability(g: dict, kind: str, overrides: dict | None = None) -> str:
    if kind not in DEFAULT_CAPABILITIES:
        raise ValueError(f"unknown candidate category {kind!r}; available {CAPABILITY_KINDS}")
    mode = ((overrides or {}).get(kind)
            or ((g.get("capabilities") or {}).get(kind))
            or DEFAULT_CAPABILITIES[kind])
    mode = str(mode).strip().lower()
    if mode not in CAPABILITY_MODES:
        raise ValueError(f"capabilities.{kind}={mode!r} is invalid; available {CAPABILITY_MODES}")
    have = CAPABILITY_PROPOSERS[kind]
    for src in ("llm", "rule"):
        if mode in (src, "both") and src not in have:
            raise ValueError(
                f"capabilities.{kind}={mode!r} requires a {src} proposer, but this category only has {have}. "
                f"That switch has no implementation, so enabling it silently proposes nothing -- rejected rather than faked.")
    return mode


def cap_allows(g: dict, kind: str, source: str, overrides: dict | None = None) -> bool:
    if source not in ("llm", "rule"):
        raise ValueError(f"source must be llm or rule, received {source!r}")
    return capability(g, kind, overrides) in (source, "both")


def resolve_capabilities(g: dict, overrides: dict | None = None) -> dict[str, str]:
    return {k: capability(g, k, overrides) for k in CAPABILITY_KINDS}


def parse_capability_overrides(items: list[str] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in items or []:
        if "=" not in raw:
            raise ValueError(f"--capability must be written kind=mode, received {raw!r}")
        k, v = raw.split("=", 1)
        k, v = k.strip().lower(), v.strip().lower()
        if k not in DEFAULT_CAPABILITIES:
            raise ValueError(f"unknown candidate category {k!r}; available {CAPABILITY_KINDS}")
        if v not in CAPABILITY_MODES:
            raise ValueError(f"{k} has invalid mode {v!r}; available {CAPABILITY_MODES}")
        out[k] = v
    return out


GATE_MODES = ("full", "acc_only", "record_only", "bypass")
NON_BYPASSABLE_GATES = ("ast_whitelist",)

DEFAULT_GATES: dict[str, dict] = {
    "paired_bootstrap": {"mode": "full", "alpha": 0.05, "n_boot": 2000, "metric": "correct"},
    "baseline_drift": {"mode": "full"},
    "frame_budget": {"mode": "full", "memory_mb": 6.0, "core_mb": 12.0},
    "macro_budget": {"mode": "full", "tol": 0.10},
    "strategy_gate5": {"mode": "full", "min_per_type": 6, "churn_max": 1.25},
    "tool_gate5": {"mode": "full", "min_pass": 0.75},
    "questioner_qscore": {"mode": "full", "min_delta": 0.0},
    "ast_whitelist": {"mode": "full"},
}
GATE_NAMES = tuple(DEFAULT_GATES)

_GATE_ENV = {
    ("paired_bootstrap", "metric"): ("VEH_GATE_METRIC", str),
    ("frame_budget", "memory_mb"): ("VEH_FRAME_BUDGET_MB", float),
    ("frame_budget", "core_mb"): ("VEH_CORE_FRAME_BUDGET_MB", float),
    ("macro_budget", "tol"): ("VEH_MACRO_BUDGET_TOL", float),
    ("strategy_gate5", "min_per_type"): ("VEH_STRATEGY_MIN_PER_TYPE", int),
    ("tool_gate5", "min_pass"): ("VEH_TOOL_GATE5_MIN_PASS", float),
}


def gate_cfg(g: dict, name: str) -> dict:
    if name not in DEFAULT_GATES:
        raise ValueError(f"unknown gate {name!r}; available {GATE_NAMES}")
    cfg = dict(DEFAULT_GATES[name])
    for (gname, key), (env, cast) in _GATE_ENV.items():
        if gname != name:
            continue
        raw = os.environ.get(env)
        if raw not in (None, ""):
            try:
                cfg[key] = cast(raw)
            except (TypeError, ValueError):
                raise ValueError(f"{env}={raw!r} cannot be parsed as {cast.__name__}")
    cfg.update({k: v for k, v in ((g.get("gates") or {}).get(name) or {}).items()})
    mode = str(cfg.get("mode", "full")).strip().lower()
    if mode not in GATE_MODES:
        raise ValueError(f"gates.{name}.mode={mode!r} is invalid; available {GATE_MODES}")
    if mode == "bypass" and name in NON_BYPASSABLE_GATES:
        raise ValueError(
            f"gates.{name} must not be bypassed: skipping it means executing arbitrary model-generated code in the container, "
            "which is not a looser criterion but no criterion at all")
    cfg["mode"] = mode
    return cfg


def gate_enforced(g: dict, name: str) -> bool:
    return gate_cfg(g, name)["mode"] == "full"


def gates_relaxed(g: dict) -> list[str]:
    return [n for n in GATE_NAMES if gate_cfg(g, n)["mode"] != "full"]


def parse_gate_overrides(items: list[str] | None) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for raw in items or []:
        if "=" not in raw:
            raise ValueError(f"--gate must be written name=mode or name.key=value, received {raw!r}")
        lhs, v = raw.split("=", 1)
        lhs, v = lhs.strip(), v.strip()
        name, _, key = lhs.partition(".")
        name = name.lower()
        if name not in DEFAULT_GATES:
            raise ValueError(f"unknown gate {name!r}; available {GATE_NAMES}")
        key = key or "mode"
        if key not in DEFAULT_GATES[name]:
            raise ValueError(f"{name} has no parameter {key!r}; available {tuple(DEFAULT_GATES[name])}")
        cur = DEFAULT_GATES[name][key]
        if key == "mode":
            val: object = v.lower()
        elif isinstance(cur, bool):
            val = v.lower() in ("1", "true", "yes", "on")
        elif isinstance(cur, int) and not isinstance(cur, bool):
            val = int(v)
        elif isinstance(cur, float):
            val = float(v)
        else:
            val = v
        out.setdefault(name, {})[key] = val
    return out


LLM_AUDIT: dict[str, dict[str, int]] = {}
LLM_COST: dict[str, dict[str, float]] = {}
_audit_lock = __import__("threading").Lock()


def _record_call(role: str, model: str) -> None:
    with _audit_lock:
        LLM_AUDIT.setdefault(role, {}).setdefault(model, 0)
        LLM_AUDIT[role][model] += 1


def _usage_int(usage: dict, key: str, *detail_paths: tuple[str, str]) -> int:
    value = usage.get(key)
    if value is not None:
        return int(value or 0)
    for parent, child in detail_paths:
        nested = usage.get(parent) or {}
        if nested.get(child) is not None:
            return int(nested.get(child) or 0)
    return 0


def _normalise_usage(usage: dict, prompt_chars: int) -> dict[str, int | bool]:
    prompt = _usage_int(usage, "prompt_tokens", ("input_tokens_details", "text_tokens"))
    completion = _usage_int(usage, "completion_tokens", ("output_tokens_details", "text_tokens"))
    total = _usage_int(usage, "total_tokens") or prompt + completion
    reasoning = _usage_int(
        usage, "reasoning_tokens",
        ("completion_tokens_details", "reasoning_tokens"),
        ("output_tokens_details", "reasoning_tokens"),
    )
    cached = _usage_int(
        usage, "cached_tokens",
        ("prompt_tokens_details", "cached_tokens"),
        ("input_tokens_details", "cached_tokens"),
    )
    image = _usage_int(
        usage, "image_tokens",
        ("prompt_tokens_details", "image_tokens"),
        ("input_tokens_details", "image_tokens"),
    )
    audio = _usage_int(
        usage, "audio_tokens",
        ("prompt_tokens_details", "audio_tokens"),
        ("input_tokens_details", "audio_tokens"),
    )
    estimated = not bool(prompt)
    if estimated:
        prompt = int(prompt_chars / 3.5)
        total = prompt + completion
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
        "reasoning_tokens": reasoning,
        "visible_completion_tokens": max(0, completion - reasoning),
        "cached_tokens": cached,
        "image_tokens": image,
        "audio_tokens": audio,
        "usage_estimated": estimated,
    }


def _append_token_event(event: dict[str, Any]) -> None:
    if TOKEN_LOG_FILE is None:
        return
    TOKEN_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
    fd = os.open(TOKEN_LOG_FILE, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        os.write(fd, payload)
    finally:
        os.close(fd)


def _record_cost(
    role: str,
    model: str,
    usage: dict,
    latency: float,
    prompt_chars: int,
    completion_chars: int,
    thinking: bool,
    status: str = "ok",
    error: str | None = None,
) -> None:
    u = _normalise_usage(usage, prompt_chars)
    if status != "ok" and not usage:
        u.update({"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
                  "reasoning_tokens": 0, "visible_completion_tokens": 0,
                  "cached_tokens": 0, "image_tokens": 0, "audio_tokens": 0,
                  "usage_estimated": False})
    with _audit_lock:
        d = LLM_COST.setdefault(
            role,
            {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
             "reasoning_tokens": 0, "cached_tokens": 0, "image_tokens": 0,
             "latency_sec": 0.0, "prompt_chars": 0},
        )
        d["calls"] += 1
        d["latency_sec"] += latency
        d["prompt_chars"] += prompt_chars
        for key in ("prompt_tokens", "completion_tokens", "reasoning_tokens", "cached_tokens", "image_tokens"):
            d[key] += int(u[key])
    _append_token_event({
        "event_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "stage": os.environ.get("VEH_STAGE", "unknown"),
        "variant": os.environ.get("VEH_VARIANT", "default"),
        "role": role,
        "model": model,
        "thinking_mode": "thinking" if thinking else "no_thinking",
        "status": status,
        "latency_sec": round(latency, 6),
        "prompt_chars": prompt_chars,
        "completion_chars": completion_chars,
        **u,
        "raw_usage": usage,
        **({"error": error} if error else {}),
    })


def llm_cost() -> dict:
    with _audit_lock:
        per_role = {r: dict(v) for r, v in LLM_COST.items()}
    total = {
        "calls": sum(v["calls"] for v in per_role.values()),
        "prompt_tokens": sum(v["prompt_tokens"] for v in per_role.values()),
        "completion_tokens": sum(v["completion_tokens"] for v in per_role.values()),
        "reasoning_tokens": sum(v.get("reasoning_tokens", 0) for v in per_role.values()),
        "cached_tokens": sum(v.get("cached_tokens", 0) for v in per_role.values()),
        "image_tokens": sum(v.get("image_tokens", 0) for v in per_role.values()),
        "latency_sec": round(sum(v["latency_sec"] for v in per_role.values()), 1),
    }
    total["total_tokens"] = total["prompt_tokens"] + total["completion_tokens"]
    return {"total": total, "by_role": per_role}


def reset_cost() -> None:
    with _audit_lock:
        LLM_COST.clear()


def llm_audit() -> dict:
    models = {m for per_role in LLM_AUDIT.values() for m in per_role}
    return {
        "single_model": len(models) <= 1,
        "distinct_models": sorted(models),
        "calls_by_role": {r: dict(v) for r, v in LLM_AUDIT.items()},
        "total_calls": sum(sum(v.values()) for v in LLM_AUDIT.values()),
    }


def reset_audit() -> None:
    with _audit_lock:
        LLM_AUDIT.clear()


class PayloadTooLargeError(RuntimeError):
    pass


class ContentFilteredError(RuntimeError):
    pass


class UpstreamTimeoutError(RuntimeError):
    pass


def call_llm(
    prompt: str,
    cfg: dict,
    max_tokens: int = 2048,
    temperature: float = 0.1,
    role: str = "unknown",
    model: str | None = None,
    frame_items: list[tuple[str, str]] | None = None,
    thinking: bool | None = None,
    frame_layout: str = "indexed",
) -> str:
    import llm_provider as _lp

    _prov = _lp.provider_name(cfg)
    mdl = model or _lp.model_of(cfg, _prov)
    if thinking is None:
        thinking = os.environ.get("VEH_THINKING", "0").strip().lower() in {"1", "true", "yes", "on", "thinking"}
    _record_call(role, mdl)
    content: list[dict] = []
    if frame_items:
        if frame_layout == "interleaved":
            for url, label in frame_items:
                content.append({"type": "text", "text": f"<{label}>"})
                content.append({"type": "image_url", "image_url": {"url": url}})
        else:
            ts = "\n".join(f"Frame {i + 1}: {label}" for i, (_, label) in enumerate(frame_items))
            prompt = f"Frame timestamps (in order):\n{ts}\n\n{prompt}"
            for url, _label in frame_items:
                content.append({"type": "image_url", "image_url": {"url": url}})
    content.append({"type": "text", "text": prompt})
    body = {
        "model": mdl,
        "messages": [{"role": "user", "content": content}],
        "max_completion_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
        "extra_body": {"enable_thinking": thinking},
    }
    body = _lp.adapt_body(body, cfg, _prov, thinking=bool(thinking))
    base = cfg["DASHSCOPE_BASE_URL"].rstrip("/")
    t0 = time.time()
    attempts = int(cfg.get("QWEN_MM_MAX_RETRIES", 6))
    read_attempts = int(cfg.get("QWEN_MM_READ_TIMEOUT_RETRIES", 2))
    gen_timeout = int(cfg.get("QWEN_MM_CHAT_TIMEOUT", 600))
    first_byte_timeout = int(cfg.get("QWEN_MM_FIRST_BYTE_TIMEOUT", min(gen_timeout, 120)))
    call_budget = float(cfg.get("QWEN_MM_CALL_BUDGET", 420))
    n_read_timeouts = 0
    last_exc: Exception | None = None
    for attempt in range(attempts):
        spent = time.time() - t0
        if spent > call_budget:
            _record_cost(role, mdl, {}, spent, len(prompt), 0, thinking,
                         "error", f"call budget {call_budget}s exceeded after {attempt} attempts")
            raise UpstreamTimeoutError(
                f"a single call exceeded the wall-clock budget {call_budget}s ({spent:.0f}s used over {attempt} attempts)")
        try:
            per_attempt = first_byte_timeout if attempt == 0 else gen_timeout
            read_to = max(5.0, min(float(per_attempt), call_budget - spent))
            r = requests.post(
                f"{base}/chat/completions",
                headers={"Authorization": _lp.auth_header(cfg["DASHSCOPE_API_KEY"], cfg, _prov),
                         "Content-Type": "application/json"},
                json=body,
                timeout=(int(cfg.get("QWEN_MM_CONNECT_TIMEOUT", 10)), read_to),
                proxies={"http": None, "https": None},
            )
            if r.status_code in (429, 500, 502, 503, 504) and attempt < attempts - 1:
                wait = min(2.0 * (2 ** attempt), 60.0) * (0.5 + random.random())
                ra = r.headers.get("Retry-After")
                if ra:
                    try:
                        wait = max(wait, float(ra))
                    except ValueError:
                        pass
                time.sleep(wait)
                continue
            if r.status_code in (400, 413) and frame_items:
                err_body = r.text or ""
                kind = _lp.classify_error(r.status_code, err_body, cfg, _prov)
                if kind == _lp.ERR_CONTENT:
                    raise ContentFilteredError(
                        f"content filter rejected the input frames: {err_body[:200]}")
                if kind == _lp.ERR_TOO_LARGE:
                    raise PayloadTooLargeError(
                        f"{len(frame_items)} frames exceed the model's per-message token limit: {err_body[:200]}")
            r.raise_for_status()
            data = r.json()
            message = data["choices"][0]["message"]
            result = message.get("content") or ""
            reasoning = message.get("reasoning_content") or ""
            usage = dict(data.get("usage") or {})
            if reasoning and not _normalise_usage(usage, len(prompt))["reasoning_tokens"]:
                usage["reasoning_tokens"] = int(len(reasoning) / 3.5)
            _record_cost(role, mdl, usage, time.time() - t0, len(prompt), len(result), thinking)
            return result
        except requests.Timeout as exc:
            last_exc = exc
            is_read = not isinstance(exc, requests.ConnectTimeout)
            if is_read:
                n_read_timeouts += 1
                if n_read_timeouts >= read_attempts:
                    waited = time.time() - t0
                    _record_cost(role, mdl, {}, waited, len(prompt), 0, thinking, "error",
                                 f"read timeout x{n_read_timeouts} "
                                 f"(first {first_byte_timeout}s, then {gen_timeout}s)")
                    raise UpstreamTimeoutError(
                        f"{n_read_timeouts} read timeouts (first-byte window {first_byte_timeout}s, "
                        f"{gen_timeout}s thereafter, {waited:.0f}s waited in total); giving up: {exc}") from exc
            if attempt < attempts - 1:
                time.sleep(min(2.0 * (2 ** attempt), 60.0) * (0.5 + random.random()))
                continue
            break
        except requests.RequestException as exc:
            last_exc = exc
            if attempt < attempts - 1:
                time.sleep(min(2.0 * (2 ** attempt), 60.0) * (0.5 + random.random()))
                continue
            break
        except Exception as exc:
            _record_cost(role, mdl, {}, time.time() - t0, len(prompt), 0, thinking, "error", str(exc)[:500])
            raise
    _record_cost(role, mdl, {}, time.time() - t0, len(prompt), 0, thinking, "error",
                 f"exhausted {attempts} attempts: {str(last_exc)[:400]}")
    raise RuntimeError(f"call_llm still failed after {attempts} retries: {last_exc}")


_EMBED_CACHE: dict[str, list[float]] = {}


def embed_texts(texts: list[str], cfg: dict | None = None) -> list[list[float]]:
    cfg = cfg or load_cfg()
    todo = [t for t in texts if t not in _EMBED_CACHE]
    if todo:
        base = cfg["DASHSCOPE_BASE_URL"].rstrip("/")
        from urllib.parse import urlsplit

        p = urlsplit(base)
        url = f"{p.scheme}://{p.netloc}/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding"
        for i in range(0, len(todo), 10):
            batch = todo[i : i + 10]
            r = requests.post(
                url,
                headers={"Authorization": f"Bearer {cfg['DASHSCOPE_API_KEY']}",
                         "Content-Type": "application/json"},
                json={"model": "qwen3-vl-embedding",
                      "input": {"contents": [{"text": t} for t in batch]},
                      "parameters": {"dimension": 2560}},
                timeout=120,
                proxies={"http": None, "https": None},
            )
            r.raise_for_status()
            embs = r.json()["output"]["embeddings"]
            for t, e in zip(batch, embs):
                _EMBED_CACHE[t] = e["embedding"] if isinstance(e, dict) else e
    return [_EMBED_CACHE[t] for t in texts]


def cosine(a: list[float], b: list[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return num / (na * nb) if na and nb else 0.0


def parse_choice(text: str, n_options: int = 4) -> str | None:
    if not text:
        return None
    letters = "".join(chr(ord("A") + i) for i in range(n_options))
    pats = [
        rf"Answer\s*:\s*\**\s*([{letters}])",
        rf"answer\s+is\s*\**\s*([{letters}])",
        rf"^\s*\**\s*([{letters}])\s*\**\s*$",
        rf"\b([{letters}])\.\s",
    ]
    for p in pats:
        m = re.search(p, text, re.M | re.I)
        if m:
            return m.group(1).upper()
    m = re.findall(rf"\b([{letters}])\b", text)
    return m[-1].upper() if m else None


def parse_anchor(text: str) -> list[float] | None:
    m = re.search(
        r"Evidence[^\n]*?(\d+(?::\d+)*(?:\.\d+)?)\s*(?:-|~|to|until)\s*(\d+(?::\d+)*(?:\.\d+)?)",
        text,
        re.I,
    )
    if not m:
        return None

    def to_sec(s: str) -> float:
        v = 0.0
        for p in (float(x) for x in s.split(":")):
            v = v * 60 + p
        return v

    try:
        return [to_sec(m.group(1)), to_sec(m.group(2))]
    except ValueError:
        return None


def active_memory_dir(video_key: str) -> Path:
    base = MEMORY_ROOT / video_key
    current = base / "CURRENT"
    if current.exists():
        version = current.read_text(encoding="utf-8").strip()
        candidate = (base / "versions" / version).resolve()
        versions = (base / "versions").resolve()
        if candidate.parent == versions and (candidate / "graph_memory.json").exists():
            return candidate
    return base


class Memory:

    def __init__(self, video_key: str):
        self.video_key = video_key
        self.dir = active_memory_dir(video_key)
        raw = json.loads((self.dir / "graph_memory.json").read_text(encoding="utf-8"))
        self.video_path: str = raw.get("video_path") or ""
        self.root: dict = raw.get("root") or {}
        self.super_events: list[dict] = raw.get("super_events") or []
        self.macros: list[dict] = raw.get("macro_events") or []
        self.macro_relations: list[dict] = raw.get("macro_relations") or []
        self._by_id = {m["macro_id"]: m for m in self.macros}

    def macro(self, macro_id: str) -> dict | None:
        return self._by_id.get(macro_id)

    def subgraph(self, macro_id: str) -> dict:
        m = self._by_id.get(macro_id) or {}
        return m.get("subgraph") or {}

    def visual_fields(self, macro_id: str) -> dict[str, list[dict]]:
        sg = self.subgraph(macro_id)
        return {
            "events": sg.get("micro_events") or [],
            "entities": sg.get("entities") or [],
            "ocr": sg.get("on_screen_texts") or [],
            "edges": sg.get("edges") or [],
        }

    def all_macro_ids(self) -> list[str]:
        return [m["macro_id"] for m in self.macros]

    def duration_sec(self) -> float | None:
        candidates = [
            self.root.get("duration_sec"), self.root.get("duration"),
            self.root.get("video_duration"),
        ]
        for value in candidates:
            try:
                if value is not None and float(value) > 0:
                    return float(value)
            except (TypeError, ValueError):
                pass
        ends: list[float] = []
        for macro in self.macros:
            tr = macro.get("time_range") or macro.get("timestamp") or []
            if isinstance(tr, (list, tuple)) and len(tr) >= 2:
                try:
                    ends.append(float(tr[1]))
                except (TypeError, ValueError):
                    pass
        return max(ends) if ends else None


def available_videos() -> list[str]:
    if not MEMORY_ROOT.is_dir():
        return []
    return sorted(
        d.name for d in MEMORY_ROOT.iterdir() if d.is_dir() and (d / "graph_memory.json").exists()
    )


_toolkit_cache: dict[str, Any] = {}


def load_toolkit(video_key: str):
    cap = ROOT / "Qwen-MM-Plugins" / "src" / "capabilities" / "video-memory"
    src = ROOT / "Qwen-MM-Plugins" / "src"
    for p in (cap, src):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    os.environ.setdefault("QWEN_MM_CONFIG", str(CONFIG_FILE))
    from qwen_mm_plugins_video_memory.loader import load_toolkit as _lt

    memory_dir = active_memory_dir(video_key)
    cache_key = f"{video_key}:{memory_dir}"
    if cache_key in _toolkit_cache:
        return _toolkit_cache[cache_key]
    tk = _lt(str(memory_dir))
    _toolkit_cache[cache_key] = tk
    return tk


def jload(s):
    return json.loads(s) if isinstance(s, str) else s


QUESTIONS = STORE / "questions.jsonl"


def save_questions(qs: list[dict], path: Path | None = None) -> Path:
    p = Path(path) if path else QUESTIONS
    with open(p, "w", encoding="utf-8") as f:
        for q in qs:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    return p


def load_questions(path: Path | None = None, split: str | None = None) -> list[dict]:
    p = Path(path) if path else QUESTIONS
    if not p.exists():
        return []
    qs = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    if split:
        qs = [q for q in qs if q.get("split") == split]
    return qs


def assign_split(qs: list[dict], evolve_ratio: float, seed: int) -> list[dict]:
    rnd = random.Random(seed)
    by_type: dict[str, list[dict]] = {}
    for q in qs:
        by_type.setdefault(q["type"], []).append(q)
    for _t, group in by_type.items():
        rnd.shuffle(group)
        cut = int(len(group) * evolve_ratio)
        for i, q in enumerate(group):
            q["split"] = "evolve" if i < cut else "heldout"
    return qs


def fmt_time(sec: float) -> str:
    sec = float(sec)
    return f"{int(sec // 60):02d}:{int(sec % 60):02d}"


def run_streaming(
    cmd: list[str],
    cwd: str | Path | None = None,
    env: dict | None = None,
    timeout: float | None = None,
    keep_tail: int = 60,
) -> tuple[int, list[str], list[str]]:
    import collections
    import os as _os
    import subprocess
    import threading as _threading

    env = dict(_os.environ) if env is None else dict(env)
    env.setdefault("PYTHONUNBUFFERED", "1")

    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1,
    )
    out_tail: collections.deque[str] = collections.deque(maxlen=keep_tail)
    err_tail: collections.deque[str] = collections.deque(maxlen=keep_tail)

    def _pump(stream, tail):
        try:
            for line in stream:
                tail.append(line.rstrip("\n"))
        except ValueError:
            pass
        finally:
            try:
                stream.close()
            except Exception:
                pass

    t_out = _threading.Thread(target=_pump, args=(proc.stdout, out_tail), daemon=True)
    t_err = _threading.Thread(target=_pump, args=(proc.stderr, err_tail), daemon=True)
    t_out.start()
    t_err.start()
    try:
        code = proc.wait(timeout=timeout) if timeout else proc.wait()
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        raise
    finally:
        t_out.join(timeout=5)
        t_err.join(timeout=5)
    return code, list(out_tail), list(err_tail)
