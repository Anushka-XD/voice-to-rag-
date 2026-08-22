"""
Multilingual embeddings for VaaniX dense retrieval.

Selected model: intfloat/multilingual-e5-small (384-d)

Why this model (vs popularity-only picks):
- Trained as a *retriever* (query/passage contrastive), not a generic LLM.
- Single multilingual space covering English and Indic languages in MSMARCO-XI
  (Hindi, Bengali, Tamil, Urdu, …) so Hindi queries can rank English passages.
- 384 dimensions: small enough for CPU + local Qdrant; larger E5/BGE-M3 are
  stronger but slower and heavier for a hackathon box.
- Official encoding uses prefixes:
    query encoding  -> "query: {text}"
    document encoding -> "passage: {text}"
  Using the same path for both would violate the model contract.

This module does NOT download a model in unit tests: inject any TextEmbedder.
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

from backend.config import EmbeddingConfig

logger = logging.getLogger(__name__)


class TextEmbedder(Protocol):
    model_name: str
    dimension: int

    def embed_text(self, text: str) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...
    def embed_documents(self, texts: Sequence[str]) -> np.ndarray: ...


def resolve_device(pref: str = "auto") -> str:
    if pref and pref != "auto":
        return pref
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def cache_key(model_name: str, prefixed_text: str) -> str:
    h = hashlib.blake2b(digest_size=16)
    h.update(model_name.encode("utf-8"))
    h.update(b"\0")
    h.update(prefixed_text.encode("utf-8"))
    return h.hexdigest()


class SqliteEmbeddingCache:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._conn = sqlite3.connect(str(path))
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS embeddings (k TEXT PRIMARY KEY, dim INTEGER NOT NULL, vec BLOB NOT NULL)"
        )
        self._conn.commit()

    def get(self, key: str, dim: int) -> np.ndarray | None:
        row = self._conn.execute("SELECT dim, vec FROM embeddings WHERE k = ?", (key,)).fetchone()
        if row is None:
            return None
        stored_dim, blob = row
        if int(stored_dim) != dim:
            return None
        arr = np.frombuffer(blob, dtype=np.float32)
        if arr.size != dim:
            return None
        return arr.copy()

    def put_many(self, items: list[tuple[str, np.ndarray]]) -> None:
        payload = [(k, int(v.size), v.astype(np.float32, copy=False).tobytes()) for k, v in items]
        self._conn.executemany(
            "INSERT OR REPLACE INTO embeddings (k, dim, vec) VALUES (?, ?, ?)",
            payload,
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()


class SentenceTransformerEmbedder:
    """
    sentence-transformers wrapper. Default: intfloat/multilingual-e5-small.

    embed_query / embed_documents apply E5 prefixes. embed_text uses the query prefix
    (online retrieval path).
    """

    def __init__(self, config: EmbeddingConfig | None = None, model: object | None = None) -> None:
        self.config = config or EmbeddingConfig.from_env()
        self.device = resolve_device(self.config.device)
        self.model_name = self.config.model_name
        self._model = model
        self._dimension: int | None = None
        self._cache: SqliteEmbeddingCache | None = None
        if self.config.cache_enabled:
            cache_file = Path(self.config.cache_dir) / f"{_safe_model_slug(self.model_name)}.sqlite"
            self._cache = SqliteEmbeddingCache(cache_file)

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading embedding model %s on %s", self.model_name, self.device)
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            get_dim = getattr(self.model, "get_embedding_dimension", None) or getattr(
                self.model, "get_sentence_embedding_dimension"
            )
            self._dimension = int(get_dim())
        return self._dimension

    def embed_text(self, text: str) -> np.ndarray:
        vecs = self.embed_query(text)
        return vecs

    def embed_query(self, text: str) -> np.ndarray:
        mat = self._encode([text], prefix=self.config.query_prefix)
        return mat[0]

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(list(texts), prefix=self.config.document_prefix)

    def _encode(self, texts: list[str], *, prefix: str) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        prefixed = [f"{prefix}{t}" if t is not None else prefix for t in texts]
        dim = self.dimension
        out = np.zeros((len(prefixed), dim), dtype=np.float32)
        missing_idx: list[int] = []
        missing_texts: list[str] = []
        missing_keys: list[str] = []

        for i, ptxt in enumerate(prefixed):
            key = cache_key(self.model_name, ptxt)
            cached = self._cache.get(key, dim) if self._cache else None
            if cached is not None:
                out[i] = cached
            else:
                missing_idx.append(i)
                missing_texts.append(ptxt)
                missing_keys.append(key)

        if missing_texts:
            encoded = self._run_model(missing_texts)
            to_store: list[tuple[str, np.ndarray]] = []
            for j, idx in enumerate(missing_idx):
                out[idx] = encoded[j]
                to_store.append((missing_keys[j], encoded[j]))
            if self._cache and to_store:
                self._cache.put_many(to_store)
        return out

    def _run_model(self, prefixed_texts: list[str]) -> np.ndarray:
        batch_size = max(1, self.config.batch_size)
        pieces: list[np.ndarray] = []
        for start in range(0, len(prefixed_texts), batch_size):
            batch = prefixed_texts[start : start + batch_size]
            vecs = self.model.encode(
                batch,
                batch_size=len(batch),
                convert_to_numpy=True,
                normalize_embeddings=self.config.normalize,
                show_progress_bar=False,
            )
            pieces.append(np.asarray(vecs, dtype=np.float32))
        return np.vstack(pieces) if pieces else np.zeros((0, self.dimension), dtype=np.float32)


def _safe_model_slug(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name)


def create_embedder(config: EmbeddingConfig | None = None) -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder(config or EmbeddingConfig.from_env())
