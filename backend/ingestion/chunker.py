"""
Chunk MSMARCO-XI PassageRecord units. Does not concatenate unrelated rows.

Consumes the Step 2 IR (PassageRecord / JSONL passages_* files).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterator

from backend.config import ChunkConfig, ChunkStrategyName
from backend.ingestion.chunking_strategies import (
    HashingNgramEmbedder,
    SentenceEmbedder,
    SentenceSpan,
    choose_adaptive_strategy,
    join_sentences,
    spans_semantic,
    spans_sliding_window,
    spans_structure_aware,
    split_sentences,
)
from backend.ingestion.schemas import ChunkRecord, PassageRecord

logger = logging.getLogger(__name__)

STRATEGY_NAMES: tuple[ChunkStrategyName, ...] = ("structure", "semantic", "sliding", "adaptive")


def make_chunk_id(document_id: str, strategy: str, chunk_index: int) -> str:
    return f"{document_id}:{strategy}:{chunk_index}"


def passage_from_dict(row: dict[str, Any]) -> PassageRecord | None:
    """Rebuild a PassageRecord from ingest JSONL (passages_*.jsonl or nested example)."""
    if not isinstance(row, dict):
        return None
    text = row.get("text")
    if not isinstance(text, str) or not text.strip():
        return None
    try:
        query_id = int(row["query_id"])
        passage_index = int(row.get("passage_index", 0))
    except (KeyError, TypeError, ValueError):
        return None
    document_id = str(row.get("document_id") or f"{query_id}:{passage_index}:{row.get('language', 'unk')}")
    return PassageRecord(
        document_id=document_id,
        query_id=query_id,
        passage_index=passage_index,
        text=text,
        language=str(row.get("language") or ""),
        language_flores=str(row.get("language_flores") or ""),
        is_selected=bool(row.get("is_selected", False)),
        passage_source=str(row.get("passage_source") or ""),
        char_length=int(row.get("char_length") or len(text)),
    )


def iter_passages_jsonl(path: Path) -> Iterator[PassageRecord]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            rec = passage_from_dict(row)
            if rec:
                yield rec


class PassageChunker:
    def __init__(
        self,
        config: ChunkConfig | None = None,
        embedder: SentenceEmbedder | None = None,
    ) -> None:
        self.config = config or ChunkConfig()
        self.embedder = embedder or HashingNgramEmbedder(dim=self.config.semantic_embed_dim)

    def chunk_passage(
        self,
        passage: PassageRecord | dict[str, Any] | None,
        strategy: ChunkStrategyName | None = None,
    ) -> list[ChunkRecord]:
        rec = self._coerce_passage(passage)
        if rec is None:
            return []
        name = strategy or self.config.strategy
        sentences = split_sentences(rec.text)
        if not sentences:
            return []

        reason: str | None = None
        if name == "adaptive":
            chosen, reason = choose_adaptive_strategy(sentences, self.config, self.embedder)
            spans = self._spans_for(chosen, sentences)
            return self._materialize(rec, spans, sentences, chunk_strategy="adaptive", adaptive_reason=f"{chosen}:{reason}")
        spans = self._spans_for(name, sentences)
        return self._materialize(rec, spans, sentences, chunk_strategy=name, adaptive_reason=None)

    def chunk_passages(
        self,
        passages: Iterator[PassageRecord] | list[PassageRecord],
        strategy: ChunkStrategyName | None = None,
    ) -> Iterator[ChunkRecord]:
        for p in passages:
            yield from self.chunk_passage(p, strategy=strategy)

    def _spans_for(self, name: str, sentences: list[str]) -> list[SentenceSpan]:
        cfg = self.config
        if name == "sliding":
            return spans_sliding_window(sentences, cfg)
        if name == "semantic":
            return spans_semantic(sentences, cfg, self.embedder)
        return spans_structure_aware(sentences, cfg)

    def _materialize(
        self,
        rec: PassageRecord,
        spans: list[SentenceSpan],
        sentences: list[str],
        *,
        chunk_strategy: str,
        adaptive_reason: str | None,
    ) -> list[ChunkRecord]:
        chunks: list[ChunkRecord] = []
        for idx, span in enumerate(spans):
            text = join_sentences(span.slice(sentences))
            if not text:
                continue
            chunks.append(
                ChunkRecord(
                    chunk_id=make_chunk_id(rec.document_id, chunk_strategy, idx),
                    document_id=rec.document_id,
                    passage_id=rec.document_id,
                    query_id=rec.query_id,
                    language=rec.language,
                    language_flores=rec.language_flores,
                    chunk_strategy=chunk_strategy,
                    chunk_index=idx,
                    start_sentence=span.start,
                    end_sentence=span.end,
                    text=text,
                    source=self.config.source_name,
                    passage_index=rec.passage_index,
                    passage_source=rec.passage_source,
                    is_selected=rec.is_selected,
                    adaptive_reason=adaptive_reason,
                )
            )
        return chunks

    @staticmethod
    def _coerce_passage(passage: PassageRecord | dict[str, Any] | None) -> PassageRecord | None:
        if passage is None:
            return None
        if isinstance(passage, PassageRecord):
            if not passage.text or not str(passage.text).strip():
                return None
            return passage
        if isinstance(passage, dict):
            return passage_from_dict(passage)
        return None
