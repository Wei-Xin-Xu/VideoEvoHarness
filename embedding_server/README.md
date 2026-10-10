# Local embedding service

`server.py` provides text embeddings for memory retrieval and forwards chat
completions to a separately supplied multimodal service. Run the examples from
the repository root after installing `requirements.txt`.

## Single server

Supply local **Qwen3-Embedding-8B** weights and a CUDA device with enough memory.
The implementation uses last-token pooling, normalization and optional dimension
truncation; it is not a generic sentence-transformers server. Build memory indexes
with the same checkpoint, pooling and dimension used at inference time.

Set `UPSTREAM_CHAT_BASE` to your actual chat API root, without `/chat/completions`.
The address below is an example, not a provided service.

```bash
export LOCAL_EMBED_MODEL_PATH="/path/to/Qwen3-Embedding-8B"
export LOCAL_EMBED_DEVICE="cuda:0"
export UPSTREAM_CHAT_BASE="https://api.example.com/v1"
python3 embedding_server/server.py --check
python3 embedding_server/server.py --host 127.0.0.1 --port 8901
```

`--check` tests embedding inference and exits with status 0 on success or 1 on
failure; it does not start a server or validate the upstream chat service. It is
silent in this release. The serving command stays in the foreground.

Set `DASHSCOPE_BASE_URL=http://127.0.0.1:8901/standard/v1` in your local
`qwen-mm-config`. Its `DASHSCOPE_API_KEY` is forwarded with chat requests.
Keep the service on loopback: it is a local adapter, not an authenticated public
endpoint. Model weights are not included.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding` | Text embeddings |
| POST | `/standard/v1/chat/completions` | Upstream chat forwarding |
| POST | `/v1/chat/completions` | Alternate chat path |
| GET | `/health` | Load status and process counters |

Embedding request body:

```json
{"model":"local-embedding","input":{"contents":[{"text":"A person enters a room."}]},"parameters":{"dimension":2560}}
```

The embeddings endpoint supports text only. `/health` describes this process;
its counters are cumulative and do not isolate individual experiments.

## Optional load balancing

For multiple embedding workers, start `server.py` separately on distinct ports
and devices. For example, use ports 8902 and 8903, then run:

```bash
export LOCAL_EMBED_BACKENDS="http://127.0.0.1:8902,http://127.0.0.1:8903"
python3 embedding_server/lb.py --host 127.0.0.1 --port 8901
```

Do not also run a standalone server on port 8901. Set `UPSTREAM_CHAT_BASE` in the
load balancer's environment too: `lb.py` distributes embedding requests but
forwards chat requests directly to the upstream service.
