"""
Dense (semantic) retriever: query embedding → Qdrant top-K.

Default is cross-lingual (no language filter). Optional language_filter restricts payload.language.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from backend.config import EmbeddingConfig, VectorStoreConfig
from backend.retrieval.embeddings import TextEmbedder, create_embedder
from backend.retrieval.vector_store import QdrantVectorStore


@dataclass
class DenseHit:
    chunk_id: str
    document_id: str
    text: str
    language: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "text": self.text,
            "language": self.language,
            "score": self.score,
            "metadata": self.metadata,
        }


class DenseRetriever:
    """Reuse one embedder + one Qdrant client for the process lifetime."""

    def __init__(
        self,
        embedder: TextEmbedder | None = None,
        store: QdrantVectorStore | None = None,
        embedding_config: EmbeddingConfig | None = None,
        store_config: VectorStoreConfig | None = None,
        *,
        default_top_k: int | None = None,
    ) -> None:
        self.embedder = embedder or create_embedder(embedding_config)
        self.store = store or QdrantVectorStore(store_config)
        self.default_top_k = default_top_k or (store_config or VectorStoreConfig.from_env()).top_k

    def search(
        self,
        query: str,
        top_k: int | None = None,
        language_filter: str | Sequence[str] | None = None,
    ) -> list[DenseHit]:
        hits, _ = self.search_timed(query, top_k=top_k, language_filter=language_filter)
        return hits

    def search_timed(
        self,
        query: str,
        top_k: int | None = None,
        language_filter: str | Sequence[str] | None = None,
    ) -> tuple[list[DenseHit], dict[str, float]]:
        k = int(top_k or self.default_top_k)
        if not query or not str(query).strip():
            return [], {"query_embed_ms": 0.0, "search_ms": 0.0, "total_ms": 0.0}
        t0 = time.perf_counter()
        qvec = self.embedder.embed_query(str(query).strip())
        t1 = time.perf_counter()
        raw = self.store.search(qvec, top_k=k, language_filter=language_filter)
        t2 = time.perf_counter()
        return self._hits_from_raw(raw), {
            "query_embed_ms": round((t1 - t0) * 1000, 3),
            "search_ms": round((t2 - t1) * 1000, 3),
            "total_ms": round((t2 - t0) * 1000, 3),
        }

    @staticmethod
    def _hits_from_raw(raw: list[dict[str, Any]]) -> list[DenseHit]:
        hits: list[DenseHit] = []
        for row in raw:
            payload = row.get("payload") or {}
            hits.append(
                DenseHit(
                    chunk_id=str(payload.get("chunk_id") or ""),
                    document_id=str(payload.get("document_id") or ""),
                    text=str(payload.get("text") or ""),
                    language=str(payload.get("language") or ""),
                    score=float(row["score"]),
                    metadata={
                        "passage_id": payload.get("passage_id"),
                        "query_id": payload.get("query_id"),
                        "chunk_strategy": payload.get("chunk_strategy"),
                        "chunk_index": payload.get("chunk_index"),
                        "source": payload.get("source"),
                        "is_selected": payload.get("is_selected"),
                        "passage_source": payload.get("passage_source"),
                        "language_flores": payload.get("language_flores"),
                        "point_id": row.get("id"),
                    },
                )
            )
        return hits
