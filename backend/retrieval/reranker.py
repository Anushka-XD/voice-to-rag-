"""
Rerank a small hybrid candidate list.

Preferred: multilingual MiniLM cross-encoder (mMARCO).
Fallback: lexical token overlap if the model cannot be loaded (offline / missing deps).
Never scores the full corpus — only the top-N hits passed in.
"""

from __future__ import annotations

import logging
from typing import Sequence

from backend.config import RoutingConfig
from backend.retrieval.bm25 import tokenize
from backend.retrieval.fusion import HybridHit

logger = logging.getLogger(__name__)

_MODEL = None
_MODEL_NAME: str | None = None
_BACKEND: str | None = None
_RERANKER_SINGLETON: Reranker | None = None
_CE_LOAD_COUNT = 0


def cross_encoder_load_count() -> int:
    return _CE_LOAD_COUNT


def reset_reranker_singleton() -> None:
    global _MODEL, _MODEL_NAME, _BACKEND, _RERANKER_SINGLETON, _CE_LOAD_COUNT
    _MODEL = None
    _MODEL_NAME = None
    _BACKEND = None
    _RERANKER_SINGLETON = None
    _CE_LOAD_COUNT = 0


def _copy_hit(h: HybridHit) -> HybridHit:
    return HybridHit(
        chunk_id=h.chunk_id,
        document_id=h.document_id,
        text=h.text,
        language=h.language,
        score=h.score,
        dense_score=h.dense_score,
        bm25_score=h.bm25_score,
        rrf_score=h.rrf_score if h.rrf_score is not None else h.score,
        rerank_score=h.rerank_score,
        final_score=h.final_score,
        metadata=dict(h.metadata or {}),
    )


def _lexical_scores(query: str, texts: Sequence[str]) -> list[float]:
    q = set(tokenize(query))
    out: list[float] = []
    for text in texts:
        d = set(tokenize(text or ""))
        if not q or not d:
            out.append(0.0)
        else:
            out.append(len(q & d) / len(q))
    return out


def _load_cross_encoder(config: RoutingConfig):
    global _MODEL, _MODEL_NAME, _BACKEND, _CE_LOAD_COUNT
    if _MODEL is not None and _MODEL_NAME == config.reranker_model and _BACKEND == "cross_encoder":
        return _MODEL
    from sentence_transformers import CrossEncoder

    from backend.retrieval.embeddings import resolve_device

    device = resolve_device(config.reranker_device)
    logger.info("Loading reranker %s on %s", config.reranker_model, device)
    _CE_LOAD_COUNT += 1
    _MODEL = CrossEncoder(config.reranker_model, device=device)
    _MODEL_NAME = config.reranker_model
    _BACKEND = "cross_encoder"
    return _MODEL


class Reranker:
    def __init__(self, config: RoutingConfig | None = None, *, backend: str | None = None) -> None:
        self.config = config or RoutingConfig.from_env()
        requested = (backend or self.config.reranker_backend or "auto").lower()
        self.backend = requested
        self._cross = None
        if requested == "lexical":
            self.backend = "lexical"
            return
        if requested in {"auto", "cross_encoder", "cross-encoder"}:
            try:
                self._cross = _load_cross_encoder(self.config)
                self.backend = "cross_encoder"
            except Exception as exc:  # model missing / offline
                logger.warning("Reranker model unavailable (%s); using lexical fallback", exc)
                if requested == "cross_encoder":
                    raise
                self.backend = "lexical"

    def rerank(
        self,
        query: str,
        hits: Sequence[HybridHit],
        *,
        top_n: int | None = None,
    ) -> list[HybridHit]:
        if not hits:
            return []
        n = int(top_n if top_n is not None else self.config.rerank_top_n)
        pool = [_copy_hit(h) for h in list(hits)[: max(n, 1)]]
        texts = [h.text or "" for h in pool]
        if self.backend == "cross_encoder" and self._cross is not None:
            pairs = [(query, t) for t in texts]
            raw = self._cross.predict(pairs, batch_size=self.config.reranker_batch_size)
            scores = [float(s) for s in raw]
        else:
            scores = _lexical_scores(query, texts)

        for hit, sc in zip(pool, scores):
            rrf = hit.rrf_score if hit.rrf_score is not None else hit.score
            hit.rrf_score = rrf
            hit.rerank_score = sc
            hit.final_score = sc
            hit.score = sc
            hit.metadata["retriever"] = "hybrid+rerank"
            hit.metadata["rrf_score"] = rrf
            hit.metadata["rerank_score"] = sc
            hit.metadata["final_score"] = sc
            hit.metadata["rerank_backend"] = self.backend
        pool.sort(key=lambda h: float(h.final_score or 0.0), reverse=True)
        return pool

    def warmup(self) -> None:
        dummy = HybridHit(
            chunk_id="warmup:0",
            document_id="warmup:0",
            text="warmup passage",
            language="en",
            score=0.01,
            rrf_score=0.01,
        )
        self.rerank("warmup", [dummy], top_n=1)


def get_reranker(config: RoutingConfig | None = None, *, singleton: bool = True) -> Reranker:
    global _RERANKER_SINGLETON
    cfg = config or RoutingConfig.from_env()
    if not singleton:
        return Reranker(cfg)
    if _RERANKER_SINGLETON is None:
        _RERANKER_SINGLETON = Reranker(cfg)
    return _RERANKER_SINGLETON
