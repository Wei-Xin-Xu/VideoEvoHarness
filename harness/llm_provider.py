from __future__ import annotations

import copy
import os
import re


PROVIDERS: dict[str, dict] = {
    "qwen": {
        "model": "qwen3.7-plus",
        "tokens_key": "max_completion_tokens",
        "thinking_off": ("extra_body", "enable_thinking", False),
        "thinking_on": ("extra_body", "enable_thinking", True),
        "max_frames": 192,
        "auth_suffix": "",
        "err_content": ("DataInspectionFailed", "UpstreamContentFilter",
                        "inappropriate content"),
        "err_too_large": ("exceed max message tokens", "Request Entity Too Large"),
        "verified": True,
    },
    "gemini": {
        "model": "gemini-3.8-flash",
        "tokens_key": "max_tokens",
        "thinking_off": ("top", "reasoning_effort", "low"),
        "thinking_on": ("top", "reasoning_effort", "medium"),
        "max_frames": 192,
        "auth_suffix": "",
        "err_content": ("DataInspectionFailed", "UpstreamContentFilter",
                        "inappropriate content"),
        "err_too_large": ("exceed max message tokens", "Request Entity Too Large"),
        "verified": True,
    },
}

_GENERIC_TOO_LARGE = (
    "exceed max message tokens", "maximum context length", "context_length_exceeded",
    "too many images", "image count", "payload too large", "request entity too large",
    "string too long",
)
_GENERIC_CONTENT = (
    "content filter", "content_filter", "inappropriate", "safety",
    "blocked", "sensitive", "risk control",
)


class ProviderError(RuntimeError):
    pass


def provider_name(cfg: dict | None = None) -> str:
    name = (os.environ.get("LLM_PROVIDER")
            or (cfg or {}).get("LLM_PROVIDER")
            or "qwen").strip().lower()
    if name not in PROVIDERS:
        raise ProviderError(
            f"unknown provider {name!r}; available: {sorted(PROVIDERS)}. "
            "A typo must raise rather than silently fall back to the default, "
            "because that would run a whole round of experiments on the wrong model without warning.")
    return name


def spec(cfg: dict | None = None, name: str | None = None) -> dict:
    return PROVIDERS[name or provider_name(cfg)]


def model_of(cfg: dict | None = None, name: str | None = None) -> str:
    p = name or provider_name(cfg)
    env = (os.environ.get("QWEN_MM_API_VL_MODEL") or "").strip()
    if env:
        return env
    configured = ((cfg or {}).get("QWEN_MM_API_VL_MODEL") or "").strip()
    if configured:
        return configured
    return str(PROVIDERS[p].get("model") or "").strip()


def auth_header(api_key: str, cfg: dict | None = None, name: str | None = None) -> str:
    return f"Bearer {api_key}{spec(cfg, name)['auth_suffix']}"


def max_frames(cfg: dict | None = None, name: str | None = None) -> int:
    env = os.environ.get("VEH_CORE_MAX_FRAMES")
    if env:
        return int(env)
    return int(spec(cfg, name)["max_frames"])


def adapt_body(body: dict, cfg: dict | None = None, name: str | None = None,
               thinking: bool = False) -> dict:
    p = spec(cfg, name)
    out = copy.deepcopy(body)

    want = model_of(cfg, name)
    cur = (out.get("model") or "").strip()
    known = {p["model"] for p in PROVIDERS.values()}
    if not cur or cur in known:
        out["model"] = want

    tok = out.pop("max_completion_tokens", None)
    tok = out.pop("max_tokens", None) if tok is None else (out.pop("max_tokens", None) or tok)
    if tok is not None:
        out[p["tokens_key"]] = tok

    eb = dict(out.pop("extra_body", None) or {})
    eb.pop("enable_thinking", None)
    eb.pop("thinking", None)
    out.pop("reasoning_effort", None)

    where, key, val = p["thinking_on" if thinking else "thinking_off"]
    if where == "extra_body":
        eb[key] = val
    else:
        out[key] = val
    if eb:
        out["extra_body"] = eb

    return out


ERR_CONTENT = "content_filtered"
ERR_TOO_LARGE = "payload_too_large"
ERR_OTHER = "other"


def classify_error(status: int, text: str, cfg: dict | None = None,
                   name: str | None = None) -> str:
    if status == 413:
        return ERR_TOO_LARGE

    p = spec(cfg, name)
    low = (text or "").lower()

    for frag in p["err_content"]:
        if frag.lower() in low:
            return ERR_CONTENT
    for frag in p["err_too_large"]:
        if frag.lower() in low:
            return ERR_TOO_LARGE

    for frag in _GENERIC_CONTENT:
        if frag in low:
            return ERR_CONTENT
    for frag in _GENERIC_TOO_LARGE:
        if frag in low:
            return ERR_TOO_LARGE
    return ERR_OTHER


_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]+")


def slug(text: str) -> str:
    return _SLUG_RE.sub("-", (text or "").strip()).strip("-")


def model_tag(cfg: dict | None = None, name: str | None = None) -> str:
    p = name or provider_name(cfg)
    return f"{p}/{model_of(cfg, p)}"
