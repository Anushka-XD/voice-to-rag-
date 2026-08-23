"""Warmup and process-level model load counting."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.retrieval.embeddings import (
    create_embedder,
    embedding_load_count,
    reset_embedder_singleton,
)
from backend.runtime.warmup import warmup_runtime


class DummyST:
    def __init__(self, *args, **kwargs):
        pass

    def get_sentence_embedding_dimension(self):
        return 8

    def encode(self, batch, **kwargs):
        return np.zeros((len(batch), 8), dtype=np.float32)


def test_warmup_does_not_require_real_models():
    class Emb:
        def __init__(self):
            self.n = 0

        def warmup(self):
            self.n += 1

    class Hyb:
        def search_timed(self, *a, **k):
            return [], {"total_ms": 1.0}

    class Store:
        def count(self):
            return 0

    class Bm:
        payloads = []

    class Rk:
        def warmup(self):
            pass

    emb = Emb()
    report = warmup_runtime(
        embedder=emb,
        reranker=Rk(),
        store=Store(),
        bm25=Bm(),
        hybrid=Hyb(),
        include_reranker=True,
    )
    assert emb.n == 1
    assert "cold_start_ms" in report
    assert "warm_request_ms" in report
    assert report["cold_start_ms"] >= 0
    assert report["warm_request_ms"] >= 0


def test_embedder_loads_once_across_queries(monkeypatch):
    pytest.importorskip("sentence_transformers")
    import backend.retrieval.embeddings as em

    monkeypatch.setattr("sentence_transformers.SentenceTransformer", DummyST)
    reset_embedder_singleton()
    a = create_embedder(singleton=True)
    a.config.cache_enabled = False
    a._cache = None
    a.warmup()
    loads = embedding_load_count()
    b = create_embedder(singleton=True)
    assert a is b
    b.embed_query("q1")
    b.embed_query("q2")
    assert embedding_load_count() == loads
    assert loads == 1
    reset_embedder_singleton()
