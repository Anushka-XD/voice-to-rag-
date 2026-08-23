"""BM25, RRF, and hybrid retrieval tests (in-memory Qdrant + fake dense embedder)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.config import HybridConfig, VectorStoreConfig
from backend.ingestion.chunker import make_chunk_id
from backend.ingestion.schemas import ChunkRecord
from backend.retrieval.bm25 import BM25Retriever, tokenize
from backend.retrieval.dense import DenseHit, DenseRetriever
from backend.retrieval.fusion import reciprocal_rank_fusion
from backend.retrieval.hybrid import HybridRetriever
from backend.retrieval.vector_store import QdrantVectorStore
from tests.test_retrieval import FakeEmbedder, _chunk


def _payloads(chunks: list[ChunkRecord]) -> list[dict]:
    return [
        {
            "chunk_id": c.chunk_id,
            "document_id": c.document_id,
            "passage_id": c.passage_id,
            "query_id": c.query_id,
            "language": c.language,
            "chunk_strategy": c.chunk_strategy,
            "chunk_index": c.chunk_index,
            "text": c.text,
            "source": c.source,
            "is_selected": c.is_selected,
            "passage_source": c.passage_source,
            "language_flores": c.language_flores,
        }
        for c in chunks
    ]


def _setup():
    chunks = [
        _chunk(
            "A corporation is a company recognized in law.",
            doc="1102432:0:en",
            lang="en",
            selected=True,
        ),
        _chunk(
            "निगम कानून में मान्यता प्राप्त कंपनी है।",
            doc="1102432:0:hi",
            lang="hi",
            selected=True,
        ),
        _chunk("Unrelated filler passage about weather and rain.", doc="1102432:1:en", lang="en", idx=1),
    ]
    store = QdrantVectorStore(VectorStoreConfig(collection_name="hyb"), in_memory=True)
    emb = FakeEmbedder()
    store.create_collection(emb.dimension, recreate=True)
    store.upsert(chunks, emb.embed_documents([c.text for c in chunks]))
    dense = DenseRetriever(embedder=emb, store=store, default_top_k=5)
    bm25 = BM25Retriever.from_payloads(_payloads(chunks), config=HybridConfig(top_k=5, parallel=False))
    hybrid = HybridRetriever(dense, bm25, HybridConfig(rrf_k=60, candidate_k=5, top_k=5, parallel=False))
    return dense, bm25, hybrid, chunks


def test_tokenize_multilingual():
    assert "corporation" in tokenize("A corporation is a company.")
    hi = tokenize("निगम कानून में मान्यता प्राप्त कंपनी है।")
    assert hi
    assert tokenize("") == []


def test_bm25_search_ranks_lexical_match():
    _, bm25, _, _ = _setup()
    hits = bm25.search("corporation", top_k=3)
    assert hits
    assert hits[0].document_id == "1102432:0:en"
    assert hits[0].chunk_id
    assert "corporation" in hits[0].text.lower()
    assert hits[0].metadata["chunk_strategy"] == "structure"
    assert hits[0].language == "en"


def test_bm25_empty_query():
    _, bm25, _, _ = _setup()
    assert bm25.search("") == []
    assert bm25.search("   ") == []


def test_dense_search_compatibility():
    dense, _, _, _ = _setup()
    hits = dense.search("what is a corporation?")
    assert hits
    h = hits[0]
    assert isinstance(h, DenseHit)
    assert h.chunk_id.startswith("1102432:")
    assert h.document_id
    assert h.text
    assert h.language in {"en", "hi"}
    assert isinstance(h.score, float)


def test_rrf_ranking_prefers_consensus():
    a = DenseHit("c1", "d1", "t1", "en", 0.9, {"retriever": "dense"})
    b = DenseHit("c2", "d2", "t2", "en", 0.8, {"retriever": "dense"})
    c = DenseHit("c1", "d1", "t1", "en", 5.0, {"retriever": "bm25"})
    d = DenseHit("c3", "d3", "t3", "en", 4.0, {"retriever": "bm25"})
    fused = reciprocal_rank_fusion([[a, b], [c, d]], k=60, top_k=3)
    assert fused[0].chunk_id == "c1"
    assert fused[0].dense_score == 0.9
    assert fused[0].bm25_score == 5.0
    assert fused[0].score > fused[1].score


def test_hybrid_search_and_metadata():
    _, _, hybrid, _ = _setup()
    hits = hybrid.search("corporation recognized in law", top_k=5)
    assert hits
    h = hits[0]
    assert h.chunk_id == make_chunk_id("1102432:0:en", "structure", 0)
    assert h.document_id == "1102432:0:en"
    assert h.language == "en"
    assert h.text
    assert h.metadata["source"] == "MSMARCO-XI"
    assert h.score > 0
    assert h.dense_score is not None or h.bm25_score is not None


def test_hybrid_empty_query():
    _, _, hybrid, _ = _setup()
    assert hybrid.search("") == []


def test_hybrid_language_filter():
    _, _, hybrid, _ = _setup()
    hi = hybrid.search("निगम", top_k=5, language_filter="hi")
    assert hi
    assert all(h.language == "hi" for h in hi)
