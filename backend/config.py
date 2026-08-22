"""Central configuration for VaaniX (ingestion, chunking, dense retrieval)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REPORTS_DIR = DATA_DIR / "reports"
CLEAN_DIR = DATA_DIR / "clean"
INDEX_DIR = DATA_DIR / "indexes"
CACHE_DIR = DATA_DIR / "cache"
INSPECTION_REPORT = REPORTS_DIR / "msmarco_xi_inspection.json"

DATASET_ID = "ai4bharat/MSMARCO-XI"
HF_PREFIX = f"hf://datasets/{DATASET_ID}"


@dataclass
class IngestConfig:
    """Configurable MSMARCO-XI ingestion (dev subset → full corpus)."""

    split: Literal["train", "validation"] = "validation"
    # Empty → discover all languages available for the split from Hub/report.
    languages: list[str] = field(default_factory=list)
    max_examples_per_lang: int | None = 50
    streaming: bool = True

    include_english_passages: bool = True
    include_translated_passages: bool = True

    # Dedup is optional so we can benchmark with/without it.
    dedupe_passages: bool = False
    # "global" = across all langs in this run; "per_lang" = within shard only.
    dedupe_scope: Literal["global", "per_lang"] = "global"

    # Skip example only when both queries are empty after cleaning.
    skip_empty_queries: bool = True
    # Keep examples with empty Answer / Eng_Answer (null-safe); never invent text.
    drop_empty_answers: bool = False

    output_dir: Path = field(default_factory=lambda: CLEAN_DIR)
    write_jsonl: bool = True
    write_stats: bool = True

    inspection_report: Path = field(default_factory=lambda: INSPECTION_REPORT)
    dataset_id: str = DATASET_ID

    def resolve_output_dir(self) -> Path:
        return Path(self.output_dir)


ChunkStrategyName = Literal["structure", "semantic", "sliding", "adaptive"]


@dataclass
class ChunkConfig:
    """
    Central chunking knobs. Sizes are in characters unless named *_sentences.

    Defaults are tuned to MSMARCO-XI inspection: passages ~310 chars mean,
    ~89% multi-sentence, almost never multi-paragraph.
    """

    strategy: ChunkStrategyName = "adaptive"

    # Sentence-window / structure-aware
    target_chunk_chars: int = 420
    min_chunk_chars: int = 80
    max_chunk_chars: int = 900
    target_sentences: int = 3
    min_sentences: int = 1
    max_sentences: int = 6
    overlap_sentences: int = 1

    # Sliding window (sentence units)
    sliding_window_sentences: int = 4
    sliding_overlap_sentences: int = 2

    # Semantic boundaries (cosine of lightweight sentence embeddings)
    semantic_similarity_threshold: float = 0.42
    semantic_embed_dim: int = 256

    # Adaptive routing
    short_max_chars: int = 160
    short_max_sentences: int = 1
    long_min_chars: int = 520
    long_min_sentences: int = 6
    topic_shift_min_sentences: int = 3

    source_name: str = "MSMARCO-XI"
    development_sample_size: int = 80


def _env(name: str, default: str | None = None) -> str | None:
    val = os.environ.get(name)
    if val is None or val == "":
        return default
    return val


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    if raw is None:
        return default
    return int(raw)


@dataclass
class EmbeddingConfig:
    """
    Default model: intfloat/multilingual-e5-small

    Why: trained for retrieval, covers Indic + English in one space, 384-d,
    query/passage prefixes, practical on CPU. Not an LLM.
    E5 requires prefixes — see embed_query vs embed_documents.
    """

    model_name: str = "intfloat/multilingual-e5-small"
    device: str = "auto"
    batch_size: int = 32
    normalize: bool = True
    query_prefix: str = "query: "
    document_prefix: str = "passage: "
    cache_enabled: bool = True
    cache_dir: Path = field(default_factory=lambda: CACHE_DIR / "embeddings")

    @classmethod
    def from_env(cls) -> EmbeddingConfig:
        return cls(
            model_name=_env("VAANIX_EMBEDDING_MODEL", "intfloat/multilingual-e5-small") or "intfloat/multilingual-e5-small",
            device=_env("VAANIX_EMBEDDING_DEVICE", "auto") or "auto",
            batch_size=_env_int("VAANIX_EMBEDDING_BATCH_SIZE", 32),
            normalize=_env_bool("VAANIX_EMBEDDING_NORMALIZE", True),
            query_prefix=_env("VAANIX_EMBEDDING_QUERY_PREFIX", "query: ") or "query: ",
            document_prefix=_env("VAANIX_EMBEDDING_DOC_PREFIX", "passage: ") or "passage: ",
            cache_enabled=_env_bool("VAANIX_EMBEDDING_CACHE", True),
            cache_dir=Path(_env("VAANIX_EMBEDDING_CACHE_DIR", str(CACHE_DIR / "embeddings")) or str(CACHE_DIR / "embeddings")),
        )


@dataclass
class VectorStoreConfig:
    """Local Qdrant by default. Set VAANIX_QDRANT_URL for a server. Never hard-code a remote URL."""

    collection_name: str = "vaanix_chunks"
    distance: str = "Cosine"
    path: Path = field(default_factory=lambda: INDEX_DIR / "qdrant")
    url: str | None = None
    api_key: str | None = None
    prefer_grpc: bool = False
    top_k: int = 10
    index_sample_size: int | None = 200
    upsert_batch_size: int = 64

    @classmethod
    def from_env(cls) -> VectorStoreConfig:
        sample = _env("VAANIX_VECTOR_INDEX_SAMPLE_SIZE", "200")
        sample_n: int | None
        if sample is None or sample.lower() in {"none", "all", "0"}:
            sample_n = None if sample and sample.lower() in {"none", "all", "0"} else 200
        else:
            sample_n = int(sample)
            if sample_n <= 0:
                sample_n = None
        return cls(
            collection_name=_env("VAANIX_QDRANT_COLLECTION", "vaanix_chunks") or "vaanix_chunks",
            distance=_env("VAANIX_QDRANT_DISTANCE", "Cosine") or "Cosine",
            path=Path(_env("VAANIX_QDRANT_PATH", str(INDEX_DIR / "qdrant")) or str(INDEX_DIR / "qdrant")),
            url=_env("VAANIX_QDRANT_URL"),
            api_key=_env("VAANIX_QDRANT_API_KEY"),
            prefer_grpc=_env_bool("VAANIX_QDRANT_PREFER_GRPC", False),
            top_k=_env_int("VAANIX_VECTOR_TOP_K", 10),
            index_sample_size=sample_n,
            upsert_batch_size=_env_int("VAANIX_QDRANT_UPSERT_BATCH", 64),
        )
