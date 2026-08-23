"""Pack hybrid hits into a compact evidence set for the LLM."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

from backend.config import GenerationConfig
from backend.retrieval.bm25 import tokenize
from backend.retrieval.fusion import HybridHit


@dataclass
class EvidenceItem:
    chunk_id: str
    document_id: str
    language: str
    score: float
    text: str
    chunk_strategy: str | None = None
    passage_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceSet:
    items: list[EvidenceItem] = field(default_factory=list)
    dropped_duplicates: int = 0
    truncated: bool = False

    def texts(self) -> str:
        return "\n\n".join(f"[{i.chunk_id}] ({i.language}) {i.text}" for i in self.items)

    def source_dicts(self) -> list[dict[str, Any]]:
        return [
            {"chunk_id": i.chunk_id, "language": i.language, "score": i.score, "document_id": i.document_id}
            for i in self.items
        ]


def _jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def select_evidence(
    hits: Sequence[HybridHit],
    config: GenerationConfig | None = None,
) -> EvidenceSet:
    cfg = config or GenerationConfig.from_env()
    ordered = sorted(hits, key=lambda h: h.score, reverse=True)
    kept: list[EvidenceItem] = []
    kept_tokens: list[list[str]] = []
    dropped = 0
    used_chars = 0
    truncated = False

    for hit in ordered:
        text = (hit.text or "").strip()
        if not text:
            continue
        toks = tokenize(text)
        if any(_jaccard(toks, prev) >= cfg.duplicate_jaccard for prev in kept_tokens):
            dropped += 1
            continue
        if len(kept) >= cfg.max_evidence_chunks:
            truncated = True
            break
        if used_chars + len(text) > cfg.max_context_chars and kept:
            truncated = True
            break
        meta = hit.metadata or {}
        kept.append(
            EvidenceItem(
                chunk_id=hit.chunk_id,
                document_id=hit.document_id,
                language=hit.language,
                score=float(hit.score),
                text=text,
                chunk_strategy=meta.get("chunk_strategy"),
                passage_id=meta.get("passage_id") or hit.document_id,
            )
        )
        kept_tokens.append(toks)
        used_chars += len(text)

    return EvidenceSet(items=kept, dropped_duplicates=dropped, truncated=truncated)
