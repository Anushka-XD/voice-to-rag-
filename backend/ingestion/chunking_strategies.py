"""
Genuinely different chunking strategies over sentence sequences.

MSMARCO-XI passages are the base unit (inspection: ~89% multi-sentence,
~0% multi-paragraph). Chunkers never concatenate unrelated rows.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from backend.config import ChunkConfig

# Latin + Indic / Arabic danda-style terminators observed in MSMARCO-XI.
_SENTENCE_RE = re.compile(
    r"(?<=[.!?।؟۔؟!…])\s+"
)


def split_sentences(text: str) -> list[str]:
    """Split on sentence boundaries; never returns empty strings."""
    if text is None:
        return []
    if not isinstance(text, str):
        text = str(text)
    stripped = text.strip()
    if not stripped:
        return []
    parts = [p.strip() for p in _SENTENCE_RE.split(stripped) if p.strip()]
    return parts or [stripped]


def join_sentences(sentences: Sequence[str]) -> str:
    return " ".join(s.strip() for s in sentences if s and str(s).strip())


class SentenceEmbedder(Protocol):
    def encode(self, sentences: Sequence[str]) -> np.ndarray: ...


class HashingNgramEmbedder:
    """
    Lightweight multilingual sentence embeddings: hashed character n-grams.

    Fast enough for a full development subset; no LLM / no downloaded model.
    Sufficient for *neighbor* topic-shift detection, not for the later vector index.
    """

    def __init__(self, dim: int = 256, ngram_min: int = 3, ngram_max: int = 5) -> None:
        self.dim = dim
        self.ngram_min = ngram_min
        self.ngram_max = ngram_max

    def encode(self, sentences: Sequence[str]) -> np.ndarray:
        if not sentences:
            return np.zeros((0, self.dim), dtype=np.float32)
        mat = np.zeros((len(sentences), self.dim), dtype=np.float32)
        for i, sent in enumerate(sentences):
            mat[i] = self._vector(sent)
        return mat

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        compact = re.sub(r"\s+", "", text.casefold())
        if not compact:
            return vec
        for n in range(self.ngram_min, self.ngram_max + 1):
            if len(compact) < n:
                continue
            for j in range(len(compact) - n + 1):
                gram = compact[j : j + n]
                h = int(hashlib.blake2b(gram.encode("utf-8"), digest_size=8).hexdigest(), 16)
                vec[h % self.dim] += 1.0
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine; empty-safe."""
    if a.size == 0 or b.size == 0:
        return np.zeros((0,), dtype=np.float32)
    an = np.linalg.norm(a, axis=1, keepdims=True)
    bn = np.linalg.norm(b, axis=1, keepdims=True)
    an = np.where(an == 0, 1.0, an)
    bn = np.where(bn == 0, 1.0, bn)
    return np.sum((a / an) * (b / bn), axis=1).astype(np.float32)


def neighbor_similarities(embeddings: np.ndarray) -> list[float]:
    if embeddings.shape[0] < 2:
        return []
    sims = cosine_rows(embeddings[:-1], embeddings[1:])
    return [float(x) for x in sims]


@dataclass
class SentenceSpan:
    """Inclusive sentence indices into the source passage's sentence list."""

    start: int
    end: int  # inclusive

    def slice(self, sentences: Sequence[str]) -> list[str]:
        return list(sentences[self.start : self.end + 1])


def _clamp_span(start: int, end: int, n: int) -> SentenceSpan | None:
    if n <= 0:
        return None
    start = max(0, start)
    end = min(n - 1, end)
    if start > end:
        return None
    return SentenceSpan(start=start, end=end)


def spans_structure_aware(sentences: Sequence[str], cfg: ChunkConfig) -> list[SentenceSpan]:
    """
    Group consecutive sentences up to target size; optional sentence overlap.
    Never splits a sentence. One oversized sentence becomes its own chunk.
    """
    n = len(sentences)
    if n == 0:
        return []
    if n == 1:
        return [SentenceSpan(0, 0)]

    spans: list[SentenceSpan] = []
    i = 0
    overlap = max(0, min(cfg.overlap_sentences, cfg.target_sentences - 1))

    while i < n:
        start = i
        char_len = len(sentences[i])
        j = i
        while j + 1 < n:
            nxt = sentences[j + 1]
            next_len = char_len + 1 + len(nxt)
            next_count = (j + 1) - start + 1
            hit_sent_cap = next_count > cfg.target_sentences
            hit_char_cap = next_len > cfg.target_chunk_chars and (j - start + 1) >= cfg.min_sentences
            hit_max = next_len > cfg.max_chunk_chars or next_count > cfg.max_sentences
            if hit_sent_cap or hit_char_cap or hit_max:
                break
            j += 1
            char_len = next_len
        spans.append(SentenceSpan(start=start, end=j))
        if j >= n - 1:
            break
        nxt_start = j - overlap + 1 if overlap > 0 else j + 1
        i = max(nxt_start, start + 1)

    return spans


def spans_sliding_window(sentences: Sequence[str], cfg: ChunkConfig) -> list[SentenceSpan]:
    """Overlapping sentence windows, e.g. 1-4, 3-6, 5-8."""
    n = len(sentences)
    if n == 0:
        return []
    window = max(1, cfg.sliding_window_sentences)
    overlap = max(0, min(cfg.sliding_overlap_sentences, window - 1))
    stride = max(1, window - overlap)
    if n <= window:
        return [SentenceSpan(0, n - 1)]

    spans: list[SentenceSpan] = []
    start = 0
    while start < n:
        end = min(n - 1, start + window - 1)
        spans.append(SentenceSpan(start=start, end=end))
        if end >= n - 1:
            break
        start += stride
    return spans


def spans_semantic(
    sentences: Sequence[str],
    cfg: ChunkConfig,
    embedder: SentenceEmbedder,
) -> list[SentenceSpan]:
    """
    Cut between neighboring sentences whose cosine similarity falls below threshold,
    then merge undersized groups and cap oversized ones.
    """
    n = len(sentences)
    if n == 0:
        return []
    if n == 1:
        return [SentenceSpan(0, 0)]

    emb = embedder.encode(list(sentences))
    sims = neighbor_similarities(emb)
    boundaries = {0}
    for i, sim in enumerate(sims):
        if sim < cfg.semantic_similarity_threshold:
            boundaries.add(i + 1)
    boundaries.add(n)
    cuts = sorted(boundaries)

    raw: list[SentenceSpan] = []
    for a, b in zip(cuts, cuts[1:]):
        if a < b:
            raw.append(SentenceSpan(start=a, end=b - 1))

    # Merge tiny groups into the next (or previous) span.
    merged: list[SentenceSpan] = []
    for span in raw:
        text = join_sentences(span.slice(sentences))
        too_small = (
            span.end - span.start + 1 < cfg.min_sentences
            or len(text) < cfg.min_chunk_chars
        )
        if too_small and merged:
            prev = merged[-1]
            merged[-1] = SentenceSpan(start=prev.start, end=span.end)
        elif too_small and not merged:
            merged.append(span)
        else:
            merged.append(span)

    # Split groups that exceed max sentences / chars without breaking a sentence.
    final: list[SentenceSpan] = []
    for span in merged:
        piece = span.slice(sentences)
        if len(piece) <= cfg.max_sentences and len(join_sentences(piece)) <= cfg.max_chunk_chars:
            final.append(span)
            continue
        sub = spans_structure_aware(piece, cfg)
        for s in sub:
            shifted = _clamp_span(span.start + s.start, span.start + s.end, n)
            if shifted:
                final.append(shifted)
    return final or [SentenceSpan(0, n - 1)]


def choose_adaptive_strategy(
    sentences: Sequence[str],
    cfg: ChunkConfig,
    embedder: SentenceEmbedder,
) -> tuple[str, str]:
    """
    Deterministic router. Returns (strategy_name, reason).

    short → structure (keep whole)
    long  → sliding windows
    topic shift → semantic
    else → structure-aware
    """
    n = len(sentences)
    chars = sum(len(s) for s in sentences) + max(0, n - 1)
    if n == 0:
        return "structure", "empty"
    if n <= cfg.short_max_sentences or chars <= cfg.short_max_chars:
        return "structure", "short_passage"
    if n >= cfg.long_min_sentences or chars >= cfg.long_min_chars:
        return "sliding", "long_passage"
    if n >= cfg.topic_shift_min_sentences:
        emb = embedder.encode(list(sentences))
        sims = neighbor_similarities(emb)
        if sims:
            low = sum(1 for s in sims if s < cfg.semantic_similarity_threshold)
            if min(sims) < cfg.semantic_similarity_threshold and (low / len(sims)) >= 0.25:
                return "semantic", "topic_shift"
    return "structure", "typical_multi_sentence"
