# VaaniX

**Adaptive Multilingual Voice RAG for Indian Languages**

This repository is built incrementally. Current stages: dataset inspection, ingestion, chunking, multilingual dense retrieval (Qdrant).

## Dataset

[ai4bharat/MSMARCO-XI](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI)

Hub layout is **parquet shards**, not per-language `load_dataset(..., "hi")` configs. Languages are discovered from filenames. There is **no native `document_id`**; we synthesize `{query_id}:{passage_index}:{lang}`. Gold relevance is `passages.is_selected`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Do not commit `.env`. `VAANIX_QDRANT_API_KEY` is only needed for a remote Qdrant server.

## Ingestion (dev subset)

```bash
python scripts/inspect_dataset.py --langs hi --split validation --samples 80
python scripts/ingest.py --langs hi --split validation --max-per-lang 10
python scripts/evaluate_chunking.py --max-passages 80
```

## Embedding model

**Chosen: [`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small)**

| Property | Value |
|---|---|
| Dimension | **384** (read from the model, not hard-coded) |
| Metric | Cosine (E5 vectors are L2-normalized) |
| Device | `auto` → CUDA / MPS / CPU |
| Query encoding | prefix `query: ` |
| Document encoding | prefix `passage: ` |

Why this model, not “whatever is popular”:

- It is a **retriever**, not an LLM. Contrastive query/passage training matches MSMARCO-style search.
- One multilingual space covering **English + Indic** languages in MSMARCO-XI, so a Hindi query can rank an English passage (and the reverse).
- 384-d and ~118M params: practical on CPU for a hackathon index; `e5-base` / BGE-M3 are stronger but heavier.
- The authors **require** different query vs document prefixes. VaaniX implements `embed_query` vs `embed_documents` accordingly.

The hashing n-gram embedder in chunking is **only** for semantic *chunk boundaries*. It is not the Qdrant encoder.

## Vector database

Local **Qdrant** at `data/indexes/qdrant` by default. Set `VAANIX_QDRANT_URL` (and optional `VAANIX_QDRANT_API_KEY`) for a server. There is no hard-coded remote URL.

Collection name: `vaanix_chunks` (configurable). Point IDs are UUID5 hashes of `chunk_id` so re-ingest is idempotent.

## Index + evaluate dense retrieval

```bash
python scripts/build_vector_index.py --mode rebuild --sample-size 200
python scripts/build_vector_index.py --mode upsert
python scripts/evaluate_dense_retrieval.py
```

Reports (measured on the current development subset — not a full-corpus claim):

- `data/reports/vector_index_report.json`
- `data/reports/dense_retrieval_report.json`

Latest development index (200 passages, adaptive chunks):

| | |
|---|---|
| Chunks indexed | 289 |
| Dimension | 384 |
| Device | MPS |
| Embedding time | 4.22 s |
| Throughput | 68.48 embeddings/s |
| Index wall time | 4.735 s |

Dense eval (10 Hindi validation queries, **4 with gold** `is_selected`; top-10):

| Setting | Recall@5 | Recall@10 | MRR | Qdrant search p50 |
|---|---|---|---|---|
| Indic query, all languages | 0.75 | 1.00 | 0.30 | 0.79 ms |
| English query, all languages | 0.50 | 1.00 | 0.25 | 0.63 ms |
| Indic query → English gold only | 0.00 | 0.25 | 0.04 | 0.44 ms |
| English query → Hindi gold only | 0.00 | 0.25 | 0.03 | 0.47 ms |

First-query embedding on MPS is ~100–300 ms (model warmup); later queries hit the embedding cache (~0.02 ms). Cross-lingual gold-only Recall is lower because the pool is tiny and gold is a specific parallel passage, not “any same-topic text”.

Evaluation uses **passage-level** gold (`is_selected` → `gold_document_ids`). A retrieved chunk is relevant if its `document_id` is in that set. There is no chunk-level label in MSMARCO-XI.

Default search is **cross-lingual** (no language filter). `language_filter="hi"` is optional.

## Tests

```bash
python -m pytest tests/ -v
```

Dense-retrieval tests use a fake embedder and **in-memory Qdrant**. They do not need a remote cluster or a downloaded E5 checkpoint.

## Not in this step

BM25, RRF, hybrid fusion, reranking, adaptive query routing, LLM generation, Sarvam STT, and the UI are later.
