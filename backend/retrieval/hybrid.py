"""
Hybrid retriever: dense + BM25 → RRF.

Default: no language filter (cross-lingual). Optional language_filter.
Dense and BM25 run independently; in parallel when HybridConfig.parallel is true.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Sequence

from backend.config import EmbeddingConfig, HybridConfig, VectorStoreConfig
from backend.retrieval.bm25 import BM25Retriever
from backend.retrieval.dense import DenseRetriever
from backend.retrieval.fusion import HybridHit, reciprocal_rank_fusion
from backend.retrieval.vector_store import QdrantVectorStore


class HybridRetriever:
    def __init__(
        self,
        dense: DenseRetriever,
        bm25: BM25Retriever,
        config: HybridConfig | None = None,
    ) -> None:
        self.dense = dense
        self.bm25 = bm25
        self.config = config or HybridConfig.from_env()

    @classmethod
    def from_store(
        cls,
        store: QdrantVectorStore | None = None,
        *,
        embedding_config: EmbeddingConfig | None = None,
        store_config: VectorStoreConfig | None = None,
        hybrid_config: HybridConfig | None = None,
        dense: DenseRetriever | None = None,
    ) -> HybridRetriever:
        cfg = hybrid_config or HybridConfig.from_env()
        vs = store or QdrantVectorStore(store_config)
        if cfg.bm25_index_path.exists():
            bm25 = BM25Retriever.load(cfg.bm25_index_path, config=cfg)
        else:
            bm25 = BM25Retriever.from_qdrant(vs, config=cfg)
            bm25.save(cfg.bm25_index_path)
        dens = dense or DenseRetriever(
            embedding_config=embedding_config,
            store=vs,
            store_config=store_config,
            default_top_k=cfg.top_k,
        )
        return cls(dens, bm25, cfg)

    def search(
        self,
        query: str,
        top_k: int | None = None,
        language_filter: str | Sequence[str] | None = None,
    ) -> list[HybridHit]:
        hits, _ = self.search_timed(query, top_k=top_k, language_filter=language_filter)
        return hits

    def search_timed(
        self,
        query: str,
        top_k: int | None = None,
        language_filter: str | Sequence[str] | None = None,
        candidate_k: int | None = None,
    ) -> tuple[list[HybridHit], dict[str, float]]:
        k_out = int(top_k or self.config.top_k)
        cand = max(k_out, int(candidate_k if candidate_k is not None else self.config.candidate_k))
        zero = {
            "dense_ms": 0.0,
            "bm25_ms": 0.0,
            "fusion_ms": 0.0,
            "total_ms": 0.0,
        }
        if not query or not str(query).strip():
            return [], zero

        import time

        t0 = time.perf_counter()
        q = str(query).strip()

        def _dense():
            return self.dense.search_timed(q, top_k=cand, language_filter=language_filter)

        def _bm25():
            return self.bm25.search_timed(q, top_k=cand, language_filter=language_filter)

        if self.config.parallel:
            with ThreadPoolExecutor(max_workers=2) as pool:
                f_d = pool.submit(_dense)
                f_b = pool.submit(_bm25)
                dense_hits, dense_lat = f_d.result()
                bm25_hits, bm25_lat = f_b.result()
        else:
            dense_hits, dense_lat = _dense()
            bm25_hits, bm25_lat = _bm25()

        t1 = time.perf_counter()
        for h in dense_hits:
            h.metadata["retriever"] = "dense"
        for h in bm25_hits:
            h.metadata["retriever"] = "bm25"
        fused = reciprocal_rank_fusion(
            [dense_hits, bm25_hits],
            k=self.config.rrf_k,
            top_k=k_out,
        )
        t2 = time.perf_counter()
        return fused, {
            "dense_ms": float(dense_lat.get("total_ms", 0.0)),
            "bm25_ms": float(bm25_lat.get("total_ms", 0.0)),
            "fusion_ms": round((t2 - t1) * 1000, 3),
            "total_ms": round((t2 - t0) * 1000, 3),
        }
