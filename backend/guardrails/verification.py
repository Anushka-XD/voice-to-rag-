"""Lightweight post-generation grounding check (token overlap, no extra LLM)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from backend.config import GenerationConfig
from backend.generation.context import EvidenceItem
from backend.retrieval.bm25 import tokenize

_STOP = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "of", "and", "or",
    "to", "in", "on", "as", "it", "its", "has", "have", "such", "for", "with",
    "that", "this", "at", "by", "from",
}

_SENT_RE = re.compile(r"(?<=[.!?।؟۔])\s+")


@dataclass
class ClaimVerdict:
    claim: str
    label: str  # supported | unsupported | uncertain
    overlap: float


@dataclass
class GroundingResult:
    verified: bool
    confidence: float
    verdicts: list[ClaimVerdict]

    @property
    def has_unsupported(self) -> bool:
        return any(v.label == "unsupported" for v in self.verdicts)


def split_claims(answer: str) -> list[str]:
    parts = [p.strip() for p in _SENT_RE.split(answer or "") if p.strip()]
    return parts or ([answer.strip()] if answer and answer.strip() else [])


def _overlap(claim: str, evidence_tokens: set[str]) -> float:
    c = {t for t in tokenize(claim) if t not in _STOP and not t.isdigit()}
    ev = {t for t in evidence_tokens if t not in _STOP and not t.isdigit()}
    if not c or not ev:
        return 0.0
    return len(c & ev) / len(c)


def verify_answer(
    answer: str,
    evidence: Sequence[EvidenceItem],
    config: GenerationConfig | None = None,
    *,
    query_language: str | None = None,
    cited_chunk_ids: set[str] | None = None,
) -> GroundingResult:
    cfg = config or GenerationConfig.from_env()
    ev_ids = {i.chunk_id for i in evidence}
    ev_langs = {i.language for i in evidence if i.language}

    # Cross-lingual: lexical overlap across scripts is not meaningful.
    if query_language and ev_langs and query_language not in ev_langs:
        cited = {c for c in (cited_chunk_ids or ev_ids) if c}
        ok = bool(cited) and cited <= ev_ids
        label = "supported" if ok else "unsupported"
        return GroundingResult(
            verified=ok,
            confidence=0.6 if ok else 0.0,
            verdicts=[ClaimVerdict(claim=answer, label=label, overlap=0.0)],
        )

    ev_tokens: set[str] = set()
    for item in evidence:
        ev_tokens.update(tokenize(item.text))
    claims = split_claims(answer)
    if not claims:
        return GroundingResult(verified=False, confidence=0.0, verdicts=[])

    verdicts: list[ClaimVerdict] = []
    for claim in claims:
        ov = _overlap(claim, ev_tokens)
        if ov >= cfg.verify_token_overlap:
            label = "supported"
        elif ov >= cfg.verify_token_overlap * 0.5:
            label = "uncertain"
        else:
            label = "unsupported"
        verdicts.append(ClaimVerdict(claim=claim, label=label, overlap=ov))

    unsupported = sum(1 for v in verdicts if v.label == "unsupported")
    uncertain = sum(1 for v in verdicts if v.label == "uncertain")
    supported = sum(1 for v in verdicts if v.label == "supported")
    n = len(verdicts)
    confidence = round((supported + 0.5 * uncertain) / n, 4)
    verified = unsupported == 0 and supported >= 1
    return GroundingResult(verified=verified, confidence=confidence, verdicts=verdicts)
