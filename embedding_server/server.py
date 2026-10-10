from __future__ import annotations

import argparse
import asyncio
import os
import random
import sys
import threading

import httpx
import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from starlette.concurrency import run_in_threadpool

MODEL_PATH = os.environ.get(
    "LOCAL_EMBED_MODEL_PATH",
    "./models/embedding-model",
)
HOST = os.environ.get("LOCAL_EMBED_HOST", "127.0.0.1")
PORT = int(os.environ.get("LOCAL_EMBED_PORT", "8901"))
DEVICE = os.environ.get("LOCAL_EMBED_DEVICE", "cuda:0")
MAX_LEN = int(os.environ.get("LOCAL_EMBED_MAX_LEN", "4096"))
UPSTREAM_CHAT = os.environ.get("UPSTREAM_CHAT_BASE", "http://127.0.0.1:8900/v1")
CHAT_TIMEOUT = int(os.environ.get("LOCAL_EMBED_CHAT_TIMEOUT", "1800"))
CHAT_MAX_CONCURRENCY = int(os.environ.get("LOCAL_EMBED_CHAT_CONCURRENCY", "64"))

CHAT_RETRY_STATUS = {
    int(s) for s in os.environ.get(
        "LOCAL_EMBED_RETRY_STATUS", "401,429,500,502,503,504"
    ).split(",") if s.strip()
}
CHAT_RETRY_ATTEMPTS = int(os.environ.get("LOCAL_EMBED_RETRY_ATTEMPTS", "3"))
CHAT_RETRY_BASE = float(os.environ.get("LOCAL_EMBED_RETRY_BASE", "1.5"))
CHAT_RETRY_BUDGET = float(os.environ.get("LOCAL_EMBED_RETRY_BUDGET", "90"))

_chat_stats = {"forwarded": 0, "retried": 0, "recovered": 0, "exhausted": 0}
_retry_by_status: dict[int, int] = {}


app = FastAPI(title="local-dashscope-embedding-shim")

_model = None
_tokenizer = None
_native_dim = None
_lock = threading.Lock()

_chat_client: httpx.AsyncClient | None = None
_HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailer", "transfer-encoding", "upgrade", "content-length", "host",
}

@app.on_event("startup")
async def _open_chat_client() -> None:
    global _chat_client
    _chat_client = httpx.AsyncClient(
        timeout=httpx.Timeout(connect=30.0, read=CHAT_TIMEOUT, write=120.0, pool=CHAT_TIMEOUT),
        limits=httpx.Limits(
            max_connections=CHAT_MAX_CONCURRENCY,
            max_keepalive_connections=CHAT_MAX_CONCURRENCY,
        ),
        trust_env=False,
        follow_redirects=False,
    )

@app.on_event("shutdown")
async def _close_chat_client() -> None:
    if _chat_client is not None:
        await _chat_client.aclose()

def _load_model():
    global _model, _tokenizer, _native_dim
    if _model is not None:
        return

    from transformers import AutoModel, AutoTokenizer

    if not os.path.isdir(MODEL_PATH):
        raise RuntimeError(
            f"\n{'=' * 66}\n"
            f"Model directory not found: {MODEL_PATH}\n\n"
            f"Place the Qwen3-Embedding-8B weights there, or set:\n"
            f"  export LOCAL_EMBED_MODEL_PATH=/path/to/your/model\n"
            f"{'=' * 66}"
        )

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, padding_side="left")
    _model = AutoModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map=None,
    ).to(DEVICE)
    _model.eval()
    _native_dim = int(_model.config.hidden_size)

def _last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    seq_lens = attention_mask.sum(dim=1) - 1
    bsz = last_hidden_states.shape[0]
    return last_hidden_states[torch.arange(bsz, device=last_hidden_states.device), seq_lens]

@torch.inference_mode()
def _embed(texts: list[str], dim: int | None) -> list[list[float]]:
    _load_model()
    batch = _tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt",
    ).to(DEVICE)

    out = _model(**batch)
    emb = _last_token_pool(out.last_hidden_state, batch["attention_mask"])
    emb = torch.nn.functional.normalize(emb, p=2, dim=1)

    if dim and dim < emb.shape[1]:
        emb = torch.nn.functional.normalize(emb[:, :dim], p=2, dim=1)

    return emb.float().cpu().tolist()

@app.post("/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding")
async def multimodal_embedding(request: Request):
    try:
        body = await request.json()
    except Exception as exc:
        return JSONResponse(status_code=400, content={"code": "InvalidJSON", "message": str(exc)})

    contents = (body.get("input") or {}).get("contents") or []
    texts: list[str] = []
    for item in contents:
        if isinstance(item, dict):
            if "text" not in item:
                return JSONResponse(
                    status_code=400,
                    content={
                        "code": "UnsupportedContent",
                        "message": f"this shim supports text input only, got: {list(item.keys())}",
                    },
                )
            texts.append(str(item["text"]))
        else:
            texts.append(str(item))

    if not texts:
        return JSONResponse(status_code=400, content={"code": "EmptyInput", "message": "input.contents is empty"})

    dim = (body.get("parameters") or {}).get("dimension")
    dim = int(dim) if dim else None

    try:
        def _run() -> list[list[float]]:
            with _lock:
                return _embed(texts, dim)

        vectors = await run_in_threadpool(_run)
    except Exception as exc:
        return JSONResponse(status_code=500, content={"code": "InternalError", "message": str(exc)})

    return {
        "output": {"embeddings": [{"index": i, "embedding": v} for i, v in enumerate(vectors)]},
        "usage": {"input_tokens": sum(len(t) for t in texts)},
        "request_id": "local-shim",
    }

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
            last = Response(
                content=resp.content,
                status_code=status,
                headers=passthrough,
                media_type=resp.headers.get("content-type"),
            )
        else:
            last = JSONResponse(status_code=status, content={"error": {"message": text}})

        if status < 400:
            _chat_stats["forwarded"] += 1
            if attempt > 1:
                _chat_stats["recovered"] += 1
            return last

        _retry_by_status[status] = _retry_by_status.get(status, 0) + 1

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
    return {
        "status": "ok",
        "model_path": MODEL_PATH,
        "model_loaded": _model is not None,
        "native_dim": _native_dim,
        "device": DEVICE,
        "upstream_chat": UPSTREAM_CHAT,
        "chat_stats": dict(_chat_stats),
        "retry_by_status": dict(_retry_by_status),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="only self-check model loading and output dimension; do not serve")
    ap.add_argument("--host", default=HOST)
    ap.add_argument("--port", type=int, default=PORT)
    args = ap.parse_args()

    if args.check:
        try:
            v = _embed(["dimension check"], 2560)
        except Exception as exc:
            sys.exit(1)
        return

    _load_model()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")

if __name__ == "__main__":
    main()
