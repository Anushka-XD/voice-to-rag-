"""Central configuration for VaaniX (ingestion, chunking, dense retrieval)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

ROOT = Path(__file__).resolve().parents[1]

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass
DATA_DIR = ROOT / "data"
REPORTS_DIR = DATA_DIR / "reports"
CLEAN_DIR = DATA_DIR / "clean"
EVAL_DIR = DATA_DIR / "eval"
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


def _env_float(name: str, default: float) -> float:
    raw = _env(name)
    if raw is None:
        return default
    return float(raw)


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


@dataclass
class HybridConfig:
    """RRF fusion of dense + BM25. k=60 is the standard Cormack constant."""

    rrf_k: int = 60
    candidate_k: int = 20
    top_k: int = 10
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    bm25_index_path: Path = field(default_factory=lambda: INDEX_DIR / "bm25.pkl")
    parallel: bool = True

    @classmethod
    def from_env(cls) -> HybridConfig:
        return cls(
            rrf_k=_env_int("VAANIX_RRF_K", 60),
            candidate_k=_env_int("VAANIX_HYBRID_CANDIDATE_K", 20),
            top_k=_env_int("VAANIX_VECTOR_TOP_K", 10),
            bm25_k1=float(_env("VAANIX_BM25_K1", "1.5") or 1.5),
            bm25_b=float(_env("VAANIX_BM25_B", "0.75") or 0.75),
            bm25_index_path=Path(_env("VAANIX_BM25_INDEX_PATH", str(INDEX_DIR / "bm25.pkl")) or str(INDEX_DIR / "bm25.pkl")),
            parallel=_env_bool("VAANIX_HYBRID_PARALLEL", True),
        )


@dataclass
class GenerationConfig:
    """Evidence packing, LLM, and guardrail thresholds."""

    max_evidence_chunks: int = 5
    max_context_chars: int = 3500
    duplicate_jaccard: float = 0.85
    min_rrf_score: float = 0.012
    min_query_overlap: float = 0.04
    verify_token_overlap: float = 0.12
    regenerate_on_fail: bool = True
    llm_model: str = "gpt-4o-mini"
    llm_base_url: str = "https://api.openai.com/v1"
    llm_timeout_s: float = 30.0
    llm_retries: int = 2
    llm_api_key: str | None = None

    @classmethod
    def from_env(cls) -> GenerationConfig:
        return cls(
            max_evidence_chunks=_env_int("VAANIX_MAX_EVIDENCE_CHUNKS", 5),
            max_context_chars=_env_int("VAANIX_MAX_CONTEXT_CHARS", 3500),
            duplicate_jaccard=_env_float("VAANIX_EVIDENCE_DUP_JACCARD", 0.85),
            min_rrf_score=_env_float("VAANIX_MIN_RRF_SCORE", 0.012),
            min_query_overlap=_env_float("VAANIX_MIN_QUERY_OVERLAP", 0.04),
            verify_token_overlap=_env_float("VAANIX_VERIFY_TOKEN_OVERLAP", 0.12),
            regenerate_on_fail=_env_bool("VAANIX_REGENERATE_ON_FAIL", True),
            llm_model=_env("VAANIX_LLM_MODEL", "gpt-4o-mini") or "gpt-4o-mini",
            llm_base_url=_env("VAANIX_LLM_BASE_URL", "https://api.openai.com/v1") or "https://api.openai.com/v1",
            llm_timeout_s=_env_float("VAANIX_LLM_TIMEOUT_S", 30.0),
            llm_retries=_env_int("VAANIX_LLM_RETRIES", 2),
            llm_api_key=_env("VAANIX_LLM_API_KEY") or _env("OPENAI_API_KEY"),
        )


RouteName = Literal["FAST", "ACCURATE", "DEEP"]


@dataclass
class RoutingConfig:
    """Deterministic FAST / ACCURATE / DEEP thresholds. No LLM classifier."""

    fast_max_tokens: int = 12
    fast_max_chars: int = 90
    fast_max_clauses: int = 1
    accurate_max_tokens: int = 28
    deep_min_tokens: int = 22
    deep_min_clauses: int = 3
    fast_top_k: int = 8
    accurate_top_k: int = 10
    deep_top_k: int = 10
    fast_candidate_k: int = 16
    accurate_candidate_k: int = 20
    deep_candidate_k: int = 40
    rerank_top_n: int = 12
    deep_verify_token_overlap: float = 0.18
    reranker_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    reranker_backend: str = "auto"
    reranker_batch_size: int = 16
    reranker_device: str = "auto"

    @classmethod
    def from_env(cls) -> RoutingConfig:
        return cls(
            fast_max_tokens=_env_int("VAANIX_ROUTE_FAST_MAX_TOKENS", 12),
            fast_max_chars=_env_int("VAANIX_ROUTE_FAST_MAX_CHARS", 90),
            fast_max_clauses=_env_int("VAANIX_ROUTE_FAST_MAX_CLAUSES", 1),
            accurate_max_tokens=_env_int("VAANIX_ROUTE_ACCURATE_MAX_TOKENS", 28),
            deep_min_tokens=_env_int("VAANIX_ROUTE_DEEP_MIN_TOKENS", 22),
            deep_min_clauses=_env_int("VAANIX_ROUTE_DEEP_MIN_CLAUSES", 3),
            fast_top_k=_env_int("VAANIX_ROUTE_FAST_TOP_K", 8),
            accurate_top_k=_env_int("VAANIX_ROUTE_ACCURATE_TOP_K", 10),
            deep_top_k=_env_int("VAANIX_ROUTE_DEEP_TOP_K", 10),
            fast_candidate_k=_env_int("VAANIX_ROUTE_FAST_CANDIDATE_K", 16),
            accurate_candidate_k=_env_int("VAANIX_ROUTE_ACCURATE_CANDIDATE_K", 20),
            deep_candidate_k=_env_int("VAANIX_ROUTE_DEEP_CANDIDATE_K", 40),
            rerank_top_n=_env_int("VAANIX_RERANK_TOP_N", 12),
            deep_verify_token_overlap=_env_float("VAANIX_DEEP_VERIFY_TOKEN_OVERLAP", 0.18),
            reranker_model=_env(
                "VAANIX_RERANKER_MODEL",
                "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
            )
            or "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
            reranker_backend=_env("VAANIX_RERANKER_BACKEND", "auto") or "auto",
            reranker_batch_size=_env_int("VAANIX_RERANKER_BATCH_SIZE", 16),
            reranker_device=_env("VAANIX_RERANKER_DEVICE", "auto") or "auto",
        )


@dataclass
class RerankPolicyConfig:
    """Skip the cross-encoder when hybrid RRF is already decisive."""

    min_top_rrf: float = 0.028
    min_score_gap: float = 0.006
    strong_rrf: float = 0.018
    max_strong_for_confident: int = 2
    deep_skip_min_rrf: float = 0.040
    allow_fast_rerank: bool = False

    @classmethod
    def from_env(cls) -> RerankPolicyConfig:
        return cls(
            min_top_rrf=_env_float("VAANIX_RERANK_MIN_TOP_RRF", 0.028),
            min_score_gap=_env_float("VAANIX_RERANK_MIN_SCORE_GAP", 0.006),
            strong_rrf=_env_float("VAANIX_RERANK_STRONG_RRF", 0.018),
            max_strong_for_confident=_env_int("VAANIX_RERANK_MAX_STRONG", 2),
            deep_skip_min_rrf=_env_float("VAANIX_RERANK_DEEP_SKIP_MIN_RRF", 0.040),
            allow_fast_rerank=_env_bool("VAANIX_RERANK_ALLOW_FAST", False),
        )


@dataclass
class AppConfig:
    cors_origins: str = "*"
    skip_warmup: bool = False
    sarvam_api_key: str | None = None
    sarvam_stt_url: str = "https://api.sarvam.ai/speech-to-text"
    sarvam_stt_model: str = "saaras:v3"
    sarvam_stt_mode: str = "transcribe"
    sarvam_timeout_s: float = 45.0

    @classmethod
    def from_env(cls) -> AppConfig:
        return cls(
            cors_origins=_env("VAANIX_CORS_ORIGINS", "*") or "*",
            skip_warmup=_env_bool("VAANIX_SKIP_WARMUP", False),
            sarvam_api_key=_env("SARVAM_API_KEY") or _env("VAANIX_SARVAM_API_KEY"),
            sarvam_stt_url=_env("VAANIX_SARVAM_STT_URL", "https://api.sarvam.ai/speech-to-text")
            or "https://api.sarvam.ai/speech-to-text",
            sarvam_stt_model=_env("VAANIX_SARVAM_STT_MODEL", "saaras:v3") or "saaras:v3",
            sarvam_stt_mode=_env("VAANIX_SARVAM_STT_MODE", "transcribe") or "transcribe",
            sarvam_timeout_s=_env_float("VAANIX_SARVAM_TIMEOUT_S", 45.0),
        )
