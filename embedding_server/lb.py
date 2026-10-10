from __future__ import annotations

import argparse
import asyncio
import os
import random

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

BACKENDS = [
    b.strip().rstrip("/")
    for b in os.environ.get(
        "LOCAL_EMBED_BACKENDS",
        "http://127.0.0.1:8902,http://127.0.0.1:8903,http://127.0.0.1:8904",
    ).split(",")
    if b.strip()
]
PORT = int(os.environ.get("LOCAL_EMBED_LB_PORT", "8901"))
HOST = os.environ.get("LOCAL_EMBED_HOST", "127.0.0.1")
EMBED_TIMEOUT = float(os.environ.get("LOCAL_EMBED_EMBED_TIMEOUT", "600"))
HEALTH_INTERVAL = float(os.environ.get("LOCAL_EMBED_HEALTH_INTERVAL", "5"))
MAX_EMBED_CHARS = int(os.environ.get("LOCAL_EMBED_MAX_EMBED_CHARS", "0"))

UPSTREAM_CHAT = os.environ.get("UPSTREAM_CHAT_BASE", "http://127.0.0.1:8900/v1")
CHAT_TIMEOUT = int(os.environ.get("LOCAL_EMBED_CHAT_TIMEOUT", "1800"))
CHAT_MAX_CONCURRENCY = int(os.environ.get("LOCAL_EMBED_CHAT_CONCURRENCY", "64"))
CHAT_RETRY_STATUS = {
    int(s) for s in os.environ.get("LOCAL_EMBED_RETRY_STATUS", "401,429,500,502,503,504").split(",") if s.strip()
}
CHAT_RETRY_ATTEMPTS = int(os.environ.get("LOCAL_EMBED_RETRY_ATTEMPTS", "3"))
CHAT_RETRY_BASE = float(os.environ.get("LOCAL_EMBED_RETRY_BASE", "1.5"))
CHAT_RETRY_BUDGET = float(os.environ.get("LOCAL_EMBED_RETRY_BUDGET", "90"))

EMBED_PATH = "/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding"


app = FastAPI(title="local-embedding-lb")

_ready: dict[str, bool] = {b: False for b in BACKENDS}
_inflight: dict[str, int] = {b: 0 for b in BACKENDS}
_done: dict[str, int] = {b: 0 for b in BACKENDS}
_failed: dict[str, int] = {b: 0 for b in BACKENDS}
_device: dict[str, str] = {}
_chat_stats = {"forwarded": 0, "retried": 0, "recovered": 0, "exhausted": 0}

_embed_client: httpx.AsyncClient | None = None
_chat_client: httpx.AsyncClient | None = None

_HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailer", "transfer-encoding", "upgrade", "content-length", "host",
}

@app.on_event("startup")
async def _startup() -> None:
    global _embed_client, _chat_client
    _embed_client = httpx.AsyncClient(
        timeout=httpx.Timeout(connect=10.0, read=EMBED_TIMEOUT, write=120.0, pool=EMBED_TIMEOUT),
        limits=httpx.Limits(max_connections=256, max_keepalive_connections=256),
        trust_env=False,
        follow_redirects=False,
    )
    _chat_client = httpx.AsyncClient(
        timeout=httpx.Timeout(connect=30.0, read=CHAT_TIMEOUT, write=120.0, pool=CHAT_TIMEOUT),
        limits=httpx.Limits(
            max_connections=CHAT_MAX_CONCURRENCY,
            max_keepalive_connections=CHAT_MAX_CONCURRENCY,
        ),
        trust_env=False,
        follow_redirects=False,
    )
    asyncio.create_task(_health_loop())

@app.on_event("shutdown")
async def _shutdown() -> None:
    for c in (_embed_client, _chat_client):
        if c is not None:
            await c.aclose()

async def _health_loop() -> None:
    while True:
        for b in BACKENDS:
            try:
                r = await _embed_client.get(f"{b}/health", timeout=5.0)
                d = r.json()
                was = _ready.get(b)
                _ready[b] = bool(d.get("model_loaded"))
                _device[b] = str(d.get("device"))
                if _ready[b] and not was:
                    pass
                if was and not _ready[b]:
                    pass
            except Exception:
                if _ready.get(b):
                    pass
                _ready[b] = False
        await asyncio.sleep(HEALTH_INTERVAL)

def _pick(exclude: set[str]) -> str | None:
    cands = [b for b in BACKENDS if _ready.get(b) and b not in exclude]
    if not cands:
        return None
    return min(cands, key=lambda b: (_inflight[b], _done[b]))

@app.post(EMBED_PATH)
async def multimodal_embedding(request: Request):
    body = await request.body()
    if MAX_EMBED_CHARS and len(body) > MAX_EMBED_CHARS:
        return JSONResponse(status_code=413, content={"code": "TooLarge", "message": f"body {len(body)} > {MAX_EMBED_CHARS}"})

    tried: set[str] = set()
    last = "no worker ready"
    for _ in range(len(BACKENDS)):
        b = _pick(tried)
        if b is None:
            break
        tried.add(b)
        _inflight[b] += 1
        try:
            r = await _embed_client.post(
                f"{b}{EMBED_PATH}",
                content=body,
                headers={"Content-Type": "application/json"},
            )
            _done[b] += 1
            if r.status_code < 500:
                return Response(content=r.content, status_code=r.status_code,
                               media_type=r.headers.get("content-type"))
            _failed[b] += 1
            last = f"{b} -> {r.status_code}: {r.text[:200]}"
        except Exception as exc:
            _failed[b] += 1
            last = f"{b} -> {exc!r}"
        finally:
            _inflight[b] -= 1

    return JSONResponse(status_code=503, content={"code": "NoBackendReady", "message": last})

async def _forward_chat(request: Request):
    if _chat_client is None:
        return JSONResponse(status_code=503, content={"error": {"message": "chat client not ready"}})

    body = await request.body()
    headers = {"Content-Type": request.headers.get("content-type", "application/json")}
    auth = request.headers.get("authorization")
    if auth:
        headers["Authorization"] = auth

    replayable = b'"stream": true' not in body and b'"stream":true' not in body
    slept = 0.0
    last: Response | JSONResponse | None = None
    for attempt in range(1, CHAT_RETRY_ATTEMPTS + 1):
        resp = None
        try:
            resp = await _chat_client.post(
                f"{UPSTREAM_CHAT.rstrip('/')}/chat/completions",
                headers=headers,
                content=body,
            )
            status, text = resp.status_code, resp.text
        except httpx.TimeoutException as exc:
            status, text = 504, f"upstream timeout: {exc}"
        except Exception as exc:
            status, text = 502, f"upstream failed: {exc}"

        if resp is not None:
            passthrough = {k: v for k, v in resp.headers.items() if k.lower() not in _HOP_BY_HOP}
            last = Response(content=resp.content, status_code=status,
                            headers=passthrough, media_type=resp.headers.get("content-type"))
        else:
            last = JSONResponse(status_code=status, content={"error": {"message": text}})

        if status < 400:
            _chat_stats["forwarded"] += 1
            if attempt > 1:
                _chat_stats["recovered"] += 1
            return last

        if not replayable or status not in CHAT_RETRY_STATUS or attempt == CHAT_RETRY_ATTEMPTS:
            break

        delay = CHAT_RETRY_BASE * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
        if resp is not None:
            try:
                delay = max(delay, float(resp.headers.get("retry-after", 0)))
            except (TypeError, ValueError):
                pass
        delay = min(delay, CHAT_RETRY_BUDGET - slept)
        if delay <= 0:
            break
        _chat_stats["retried"] += 1
        slept += delay
        await asyncio.sleep(delay)

    _chat_stats["exhausted"] += 1
    return last

@app.post("/standard/v1/chat/completions")
async def chat_standard(request: Request):
    return await _forward_chat(request)

@app.post("/v1/chat/completions")
async def chat_v1(request: Request):
    return await _forward_chat(request)

@app.get("/health")
async def health():
    n_ready = sum(1 for b in BACKENDS if _ready.get(b))
    devs = sorted({_device[b] for b in BACKENDS if _ready.get(b) and _device.get(b)})
    return {
        "status": "ok",
        "role": "lb",
        "model_loaded": n_ready > 0,
        "device": f"lb:pool({n_ready}/{len(BACKENDS)})" + (f" {','.join(devs)}" if devs else ""),
        "ready_backends": n_ready,
        "total_backends": len(BACKENDS),
        "backends": {
            b: {
                "ready": _ready.get(b, False),
                "device": _device.get(b),
                "inflight": _inflight.get(b, 0),
                "done": _done.get(b, 0),
                "failed": _failed.get(b, 0),
            }
            for b in BACKENDS
        },
        "chat_stats": dict(_chat_stats),
        "upstream_chat": UPSTREAM_CHAT,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=HOST)
    ap.add_argument("--port", type=int, default=PORT)
    args = ap.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")

if __name__ == "__main__":
    main()
