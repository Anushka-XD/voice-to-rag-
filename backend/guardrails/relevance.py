"""Refuse before LLM when hybrid evidence is too weak."""

from __future__ import annotations

from typing import Sequence

from backend.config import GenerationConfig
from backend.retrieval.bm25 import tokenize
from backend.retrieval.fusion import HybridHit


REFUSAL_EN = (
    "I don't have enough information in the provided knowledge base to answer that reliably."
)
REFUSAL_HI = "ज्ञान आधार में पर्याप्त जानकारी नहीं है कि मैं इसका विश्वसनीय उत्तर दे सकूँ।"


def refusal_text(query_language: str) -> str:
    if query_language == "hi":
        return REFUSAL_HI
    return REFUSAL_EN


def query_evidence_overlap(query: str, hits: Sequence[HybridHit]) -> float:
    q = set(tokenize(query))
    if not q:
        return 0.0
    best = 0.0
    for h in hits:
        t = set(tokenize(h.text or ""))
        if not t:
            continue
        best = max(best, len(q & t) / len(q))
    return best


def retrieval_is_sufficient(
    hits: Sequence[HybridHit],
    query: str,
    config: GenerationConfig | None = None,
    *,
    query_language: str | None = None,
) -> tuple[bool, str]:
    cfg = config or GenerationConfig.from_env()
    if not hits:
        return False, "no_hits"
    top = max(h.score for h in hits)
    if top < cfg.min_rrf_score:
        return False, "low_score"
    hit_langs = {h.language for h in hits if h.language}
    if query_language and hit_langs and query_language not in hit_langs:
        return True, "ok_cross_lingual"
    overlap = query_evidence_overlap(query, hits)
    if overlap < cfg.min_query_overlap:
        return False, "low_lexical_overlap"
    return True, "ok"
