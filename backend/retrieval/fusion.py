"""Reciprocal Rank Fusion over ranked hit lists (same DenseHit shape)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from backend.retrieval.dense import DenseHit


@dataclass
class HybridHit:
    chunk_id: str
    document_id: str
    text: str
    language: str
    score: float  # current ranking score (RRF, or rerank final after reranking)
    dense_score: float | None = None
    bm25_score: float | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None
    final_score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "text": self.text,
            "language": self.language,
            "score": self.score,
            "dense_score": self.dense_score,
            "bm25_score": self.bm25_score,
            "rrf_score": self.rrf_score,
            "rerank_score": self.rerank_score,
            "final_score": self.final_score,
            "metadata": self.metadata,
        }


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[DenseHit]],
    *,
    k: int = 60,
    top_k: int = 10,
) -> list[HybridHit]:
    """
    RRF: score(d) = sum_i 1 / (k + rank_i(d))
    rank is 1-based. Lists may use different score scales; RRF ignores raw scores.
    """
    fused: dict[str, dict[str, Any]] = {}
    for hits in rankings:
        for rank, hit in enumerate(hits, start=1):
            cid = hit.chunk_id
            if not cid:
                continue
            slot = fused.setdefault(
                cid,
                {
                    "hit": hit,
                    "rrf": 0.0,
                    "dense_score": None,
                    "bm25_score": None,
                },
            )
            slot["rrf"] += 1.0 / (k + rank)
            retriever = (hit.metadata or {}).get("retriever")
            if retriever == "bm25":
                slot["bm25_score"] = hit.score
            else:
                slot["dense_score"] = hit.score
                if retriever is None:
                    # DenseHit from DenseRetriever has no retriever tag.
                    slot["dense_score"] = hit.score

    ordered = sorted(fused.values(), key=lambda x: x["rrf"], reverse=True)[:top_k]
    out: list[HybridHit] = []
    for item in ordered:
        h: DenseHit = item["hit"]
        meta = dict(h.metadata or {})
        meta["retriever"] = "hybrid"
        out.append(
            HybridHit(
                chunk_id=h.chunk_id,
                document_id=h.document_id,
                text=h.text,
                language=h.language,
                score=float(item["rrf"]),
                dense_score=item["dense_score"],
                bm25_score=item["bm25_score"],
                rrf_score=float(item["rrf"]),
                final_score=float(item["rrf"]),
                metadata=meta,
            )
        )
    return out
