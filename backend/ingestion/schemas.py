"""Clean intermediate representation for MSMARCO-XI (chunking consumes this)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class TranslationMeta:
    """Preserved from raw `meta` (parquet may store numerics as int)."""

    model_name: str | None = None
    temperature: float | int | None = None
    max_tokens: int | None = None
    top_p: float | int | None = None
    frequency_penalty: float | int | None = None
    presence_penalty: float | int | None = None

    @classmethod
    def from_raw(cls, raw: Any) -> TranslationMeta:
        if not isinstance(raw, dict):
            return cls()
        return cls(
            model_name=raw.get("model_name"),
            temperature=raw.get("temperature"),
            max_tokens=raw.get("max_tokens"),
            top_p=raw.get("top_p"),
            frequency_penalty=raw.get("frequency_penalty"),
            presence_penalty=raw.get("presence_penalty"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PassageRecord:
    """
    One indexable passage unit.

    document_id is synthesized: '{query_id}:{passage_index}:{lang_tag}'
    because the dataset has no native document_id field.
    """

    document_id: str
    query_id: int
    passage_index: int
    text: str
    # Short code from shard filename discovery (e.g. hi) or 'en' for English passages.
    language: str
    # Exact FLORES-style tag from the row (eng_Latn / hin_Deva / …).
    language_flores: str
    is_selected: bool
    # "english" | "translated" — which list this text came from.
    passage_source: str
    char_length: int = 0

    def __post_init__(self) -> None:
        if not self.char_length:
            self.char_length = len(self.text)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CleanExample:
    """
    One cleaned MSMARCO-XI example.

    Preserves query↔passage↔selection relationships for retrieval evaluation.
    """

    query_id: int
    # Language code of the shard this row was loaded from (discovered, not invented).
    shard_lang: str
    source_lang: str
    target_lang: str
    query: str
    eng_query: str
    # None when empty/missing after cleaning — never fabricated.
    answer: str | None
    eng_answer: str | None
    query_type: str | None
    meta: TranslationMeta
    passages: list[PassageRecord] = field(default_factory=list)
    # Indices into passages that are gold (is_selected) — convenience for eval.
    gold_document_ids: list[str] = field(default_factory=list)
    split: str = "validation"
    raw_passage_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


@dataclass
class IngestStats:
    """Measured during an ingest run — never fabricated."""

    records_processed: int = 0
    records_emitted: int = 0
    records_skipped: int = 0
    empty_query_skipped: int = 0
    empty_answer_count: int = 0
    empty_eng_answer_count: int = 0
    passage_units_emitted: int = 0
    passages_deduped: int = 0
    examples_with_no_passages: int = 0
    language_distribution: dict[str, int] = field(default_factory=dict)
    flores_target_distribution: dict[str, int] = field(default_factory=dict)
    query_type_distribution: dict[str, int] = field(default_factory=dict)
    total_query_chars: int = 0
    total_passage_chars: int = 0
    processing_seconds: float = 0.0
    languages_requested: list[str] = field(default_factory=list)
    languages_loaded: list[str] = field(default_factory=list)
    split: str = ""
    dedupe_passages: bool = False
    max_examples_per_lang: int | None = None
    errors: list[str] = field(default_factory=list)

    def average_query_chars(self) -> float | None:
        if self.records_emitted == 0:
            return None
        return round(self.total_query_chars / self.records_emitted, 2)

    def average_passage_chars(self) -> float | None:
        if self.passage_units_emitted == 0:
            return None
        return round(self.total_passage_chars / self.passage_units_emitted, 2)

    def to_dict(self) -> dict[str, Any]:
        return {
            "records_processed": self.records_processed,
            "records_emitted": self.records_emitted,
            "records_skipped": self.records_skipped,
            "empty_query_skipped": self.empty_query_skipped,
            "empty_answer_count": self.empty_answer_count,
            "empty_eng_answer_count": self.empty_eng_answer_count,
            "passage_units_emitted": self.passage_units_emitted,
            "passages_deduped": self.passages_deduped,
            "examples_with_no_passages": self.examples_with_no_passages,
            "language_distribution": dict(self.language_distribution),
            "flores_target_distribution": dict(self.flores_target_distribution),
            "query_type_distribution": dict(self.query_type_distribution),
            "average_query_chars": self.average_query_chars(),
            "average_passage_chars": self.average_passage_chars(),
            "processing_seconds": round(self.processing_seconds, 3),
            "languages_requested": self.languages_requested,
            "languages_loaded": self.languages_loaded,
            "split": self.split,
            "dedupe_passages": self.dedupe_passages,
            "max_examples_per_lang": self.max_examples_per_lang,
            "errors": self.errors[:50],
        }
