# VaaniX

**Adaptive multilingual voice RAG for Indian languages** (MSMARCO-XI).

Voice or text → Sarvam STT (optional) → hybrid retrieval → grounded answer **or refusal**. The model is not allowed to use world knowledge.

Main entry: `run_query()` in `backend/orchestration/pipeline.py`. The API is a thin wrapper.

## Architecture

```
microphone / text
        │
        ▼
Sarvam STT (voice only)     POST /voice-query
        │
        ▼
validate → FAST|ACCURATE|DEEP routing (no LLM)
        │
        ▼
dense (multilingual-e5-small) + BM25  →  RRF hybrid
        │
        ▼
rerank policy (skip if hybrid is already confident)
        │
        ▼
evidence pack → LLM (JSON, evidence-only) → grounding check
        │
        ▼
structured answer / refusal  →  UI
```

Qdrant is local by default (`data/indexes/qdrant`). Query language is **not** forced to match document language.

## How RAG works

1. **Ingest** MSMARCO-XI parquet shards; synthesize `document_id = {query_id}:{passage_index}:{lang}`. Gold is `is_selected == 1` only.
2. **Chunk** with one of: structure, semantic, sliding, **adaptive** (short→structure, long→sliding, topic-shift→semantic).
3. **Retrieve** dense + BM25 in parallel, fuse with RRF (`k=60`).
4. **Route** FAST / ACCURATE / DEEP from query features (length, clauses, comparison, etc.).
5. **Rerank** a top-N list with a multilingual MiniLM cross-encoder **only when the policy says hybrid is ambiguous**. FAST never reranks by default.
6. **Generate** from packed evidence only. Weak retrieval or unsupported claims → refusal.

## Chunking strategies

| Name | Role |
|---|---|
| structure | Sentence-window chunks |
| semantic | Split on topic-shift (hashing n-grams, not E5) |
| sliding | Overlapping sentence windows |
| adaptive | Picks among the three from passage length / structure |

## Retrieval benchmark (112 gold queries, 3312 chunks)

Gold = `document_id` in `is_selected` passages. No fabricated labels. **No DEEP queries** in this MSMARCO-XI sample.

| System | R@5 | R@10 | MRR | warm p50 |
|---|---|---|---|---|
| Dense | 0.46 | 0.76 | 0.24 | 2.9 ms |
| BM25 | 0.80 | 0.89 | **0.53** | 3.2 ms |
| Hybrid | **0.81** | 0.88 | 0.39 | 6.1 ms |
| Always rerank | 0.75 | 0.88 | 0.38 | 91.5 ms |

Adaptive rerank (50% of queries): MRR **0.43**, R@5 0.78, p50 54 ms. Always-rerank does **not** improve retrieval here. BM25 MRR can beat hybrid; RRF is not automatically best.

Cold start (load embedder + reranker + index): **~24 s**. Warm hybrid probe: **~59 ms**.

## Guardrails

- Retrieval confidence: skip the LLM if hits are empty/weak.
- Evidence-only system prompt; `INSUFFICIENT_CONTEXT` when evidence is thin.
- Post-generation token-overlap / cited-id check; unsupported → one strict regen or refuse.
- Off-topic questions (e.g. “capital of Mars”) refuse rather than using general knowledge.

## Latency

Do **not** claim &lt;200 ms end-to-end when STT and a real LLM are in the loop. Those APIs dominate.

Measured **text** E2E on 20 gold queries after warmup (`python scripts/benchmark_e2e.py`, no LLM key, no STT):

| Stage | p50 | p70 | p100 |
|---|---|---|---|
| Retrieval | 6.5 ms | 7.2 ms | 13.7 ms |
| Rerank (when applied; 0 otherwise) | 23.8 ms | 69.2 ms | 162.1 ms |
| Generation (no API key → error/refuse path) | 0.02 ms | 0.02 ms | 0.03 ms |
| Total (warm) | 31.5 ms | 76.8 ms | 173.9 ms |

Cold start: **20.8 s** (embed 13.6 s, reranker 6.6 s). Warm probe: **29 ms**.

STT is only on `POST /voice-query` (`latency.stt_ms`). It was not measured here (no `SARVAM_API_KEY`). Real LLM `generation_ms` will be hundreds of ms to seconds once a key is set.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Set VAANIX_LLM_API_KEY or OPENAI_API_KEY
# Set SARVAM_API_KEY for voice
```

Index (if `data/indexes/qdrant` is empty):

```bash
python scripts/ingest.py --split validation --max-per-lang 15
python scripts/build_index.py --limit 0 --strategy adaptive
python scripts/build_retrieval_eval.py
```

Tests:

```bash
python -m pytest tests/ -q
```

## Run locally

```bash
# API (warmup on startup unless VAANIX_SKIP_WARMUP=true)
uvicorn backend.api.app:app --host 0.0.0.0 --port 8000

# UI
cd frontend && npm install && npm run dev
```

Open http://localhost:5173 (Vite proxies `/query`, `/voice-query`, `/health` to port 8000).

### Demo

1. Start the API, wait until `/health` returns `ok` (first boot can take ~25 s).
2. Start the frontend.
3. Type a query such as `कॉर्पोरेशन क्या है?` or `what is a corporation?`, or use **Record**.
4. You should see status `grounded` or `insufficient_context`, sources, and latency breakdown.

## API

- `GET /health`
- `POST /query` `{"query": "..."}`
- `POST /voice-query` multipart field `audio`

## Environment variables

See `.env.example`. Required for a full demo:

- `VAANIX_LLM_API_KEY` or `OPENAI_API_KEY`
- `SARVAM_API_KEY` (voice)
- Optional: `VAANIX_CORS_ORIGINS`, `VAANIX_SKIP_WARMUP`, Qdrant/embedding knobs

Never commit `.env`.

## Deploy (Render)

`render.yaml` + `Procfile`: `uvicorn backend.api.app:app --host 0.0.0.0 --port $PORT`

Set secrets in the Render dashboard. The local Qdrant index is **not** in git; attach a disk or rebuild the index on the instance. Frontend can be a second static site with `VITE_API_URL` pointing at the API, or `npm run build` in `frontend/` and serve `frontend/dist` from the same service (mounted if present).
