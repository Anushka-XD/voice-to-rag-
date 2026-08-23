"""
BM25 over the same chunks stored in Qdrant.

Index is built once (from payloads or a pickle) and reused for every query.
Tokenization is Unicode-aware so Hindi/Indic whitespace-separated tokens work.
"""

from __future__ import annotations

import pickle
import re
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

from rank_bm25 import BM25Okapi

from backend.config import HybridConfig
from backend.retrieval.dense import DenseHit

# Letters/numbers in any script; splits punctuation without dropping Devanagari.
_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    if not text:
        return []
    return [t.casefold() for t in _TOKEN_RE.findall(text)]


def hit_from_payload(payload: dict[str, Any], score: float) -> DenseHit:
    return DenseHit(
        chunk_id=str(payload.get("chunk_id") or ""),
        document_id=str(payload.get("document_id") or ""),
        text=str(payload.get("text") or ""),
        language=str(payload.get("language") or ""),
        score=float(score),
        metadata={
            "passage_id": payload.get("passage_id"),
            "query_id": payload.get("query_id"),
            "chunk_strategy": payload.get("chunk_strategy"),
            "chunk_index": payload.get("chunk_index"),
            "source": payload.get("source"),
            "is_selected": payload.get("is_selected"),
            "passage_source": payload.get("passage_source"),
            "language_flores": payload.get("language_flores"),
            "retriever": "bm25",
        },
    )


class BM25Retriever:
    def __init__(
        self,
        payloads: list[dict[str, Any]],
        *,
        k1: float = 1.5,
        b: float = 0.75,
        default_top_k: int = 10,
    ) -> None:
        self.payloads = payloads
        self.k1 = k1
        self.b = b
        self.default_top_k = default_top_k
        tokenized = [tokenize(str(p.get("text") or "")) for p in payloads]
        # BM25Okapi requires a non-empty corpus; keep a dummy row if empty.
        self._bm25 = BM25Okapi(tokenized or [[""]], k1=k1, b=b)
        self._empty = len(payloads) == 0

    @classmethod
    def from_payloads(
        cls,
        payloads: Iterable[dict[str, Any]],
        config: HybridConfig | None = None,
    ) -> BM25Retriever:
        cfg = config or HybridConfig.from_env()
        docs = [p for p in payloads if str(p.get("text") or "").strip()]
        return cls(docs, k1=cfg.bm25_k1, b=cfg.bm25_b, default_top_k=cfg.top_k)

    @classmethod
    def from_qdrant(cls, store, config: HybridConfig | None = None) -> BM25Retriever:
        return cls.from_payloads(store.iter_payloads(), config=config)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(
            pickle.dumps(
                {"payloads": self.payloads, "k1": self.k1, "b": self.b},
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        )

    @classmethod
    def load(cls, path: Path, config: HybridConfig | None = None) -> BM25Retriever:
        data = pickle.loads(path.read_bytes())
        cfg = config or HybridConfig.from_env()
        return cls(
            data["payloads"],
            k1=float(data.get("k1", cfg.bm25_k1)),
            b=float(data.get("b", cfg.bm25_b)),
            default_top_k=cfg.top_k,
        )

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
        empty_lat = {"search_ms": 0.0, "total_ms": 0.0}
        if self._empty or not query or not str(query).strip():
            return [], empty_lat
        t0 = time.perf_counter()
        tokens = tokenize(str(query).strip())
        if not tokens:
            return [], empty_lat
        scores = self._bm25.get_scores(tokens)
        langs = _lang_set(language_filter)
        ranked = sorted(range(len(self.payloads)), key=lambda i: float(scores[i]), reverse=True)
        hits: list[DenseHit] = []
        for i in ranked:
            payload = self.payloads[i]
            if langs is not None and str(payload.get("language") or "") not in langs:
                continue
            sc = float(scores[i])
            if sc <= 0:
                continue
            hits.append(hit_from_payload(payload, sc))
            if len(hits) >= k:
                break
        ms = round((time.perf_counter() - t0) * 1000, 3)
        return hits, {"search_ms": ms, "total_ms": ms}


def _lang_set(language_filter: str | Sequence[str] | None) -> set[str] | None:
    if language_filter is None:
        return None
    if isinstance(language_filter, str):
        return {language_filter} if language_filter else None
    vals = {str(x) for x in language_filter if x}
    return vals or None
