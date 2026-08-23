"""Load retrieval models once and report cold vs warm latency."""

from __future__ import annotations

import time
from typing import Any

from backend.config import EmbeddingConfig, HybridConfig, RoutingConfig, VectorStoreConfig
from backend.retrieval.bm25 import BM25Retriever
from backend.retrieval.dense import DenseRetriever
from backend.retrieval.embeddings import create_embedder
from backend.retrieval.hybrid import HybridRetriever
from backend.retrieval.reranker import get_reranker
from backend.retrieval.vector_store import QdrantVectorStore


def warmup_runtime(
    *,
    embedder: Any | None = None,
    reranker: Any | None = None,
    store: Any | None = None,
    bm25: Any | None = None,
    hybrid: Any | None = None,
    include_reranker: bool = True,
) -> dict[str, Any]:
    """
    Load embedder, Qdrant, BM25, and reranker once.

    Returns measured cold_start (this call) and a warm_request probe if hybrid is available.
    """
    report: dict[str, Any] = {"components": {}}
    t0 = time.perf_counter()

    t = time.perf_counter()
    emb = embedder or create_embedder(EmbeddingConfig.from_env())
    if hasattr(emb, "warmup"):
        emb.warmup()
    report["components"]["embedding_ms"] = round((time.perf_counter() - t) * 1000, 3)

    t = time.perf_counter()
    vs = store or QdrantVectorStore(VectorStoreConfig.from_env())
    _ = vs.count() if hasattr(vs, "count") else None
    report["components"]["qdrant_ms"] = round((time.perf_counter() - t) * 1000, 3)

    t = time.perf_counter()
    hy_cfg = HybridConfig.from_env()
    if bm25 is None:
        if hy_cfg.bm25_index_path.exists():
            bm25 = BM25Retriever.load(hy_cfg.bm25_index_path, config=hy_cfg)
        else:
            bm25 = BM25Retriever.from_qdrant(vs, config=hy_cfg)
    report["components"]["bm25_ms"] = round((time.perf_counter() - t) * 1000, 3)

    rk = reranker
    if include_reranker:
        t = time.perf_counter()
        rk = rk or get_reranker(RoutingConfig.from_env())
        if hasattr(rk, "warmup"):
            rk.warmup()
        report["components"]["reranker_ms"] = round((time.perf_counter() - t) * 1000, 3)

    if hybrid is None:
        dense = DenseRetriever(embedder=emb, store=vs, default_top_k=hy_cfg.top_k)
        hybrid = HybridRetriever(dense, bm25, hy_cfg)

    report["cold_start_ms"] = round((time.perf_counter() - t0) * 1000, 3)

    t = time.perf_counter()
    hybrid.search_timed("warmup query", top_k=3)
    report["warm_request_ms"] = round((time.perf_counter() - t) * 1000, 3)
    report["embedder"] = emb
    report["reranker"] = rk
    report["store"] = vs
    report["bm25"] = bm25
    report["hybrid"] = hybrid
    return report
