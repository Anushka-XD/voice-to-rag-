"""Dense retrieval tests: fake embedder + in-memory Qdrant (no remote service)."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.config import EmbeddingConfig, VectorStoreConfig
from backend.ingestion.chunker import PassageChunker, make_chunk_id
from backend.ingestion.schemas import ChunkRecord
from backend.retrieval.dense import DenseRetriever
from backend.retrieval.embeddings import cache_key
from backend.retrieval.vector_store import QdrantVectorStore, point_id_for_chunk


class FakeEmbedder:
    """Deterministic 8-d embedder. Parallel corporation/निगम texts share a cluster."""

    model_name = "fake-test-embedder"
    dimension = 8

    def embed_text(self, text: str) -> np.ndarray:
        return self.embed_query(text)

    def embed_query(self, text: str) -> np.ndarray:
        return self._vec(text)

    def embed_documents(self, texts):
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        return np.stack([self._vec(t) for t in texts])

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dimension, dtype=np.float32)
        blob = text or ""
        low = blob.casefold()
        if "corporation" in low or "निगम" in blob or "कॉर्पोरेशन" in blob:
            v[0] = 1.0
        elif "capital" in low or "राजधानी" in blob:
            v[1] = 1.0
        else:
            digest = hashlib.blake2b(blob.encode("utf-8"), digest_size=8).digest()
            for i, b in enumerate(digest):
                v[i] = b / 255.0
            v *= 0.2
        n = float(np.linalg.norm(v))
        return v / n if n else v


def _chunk(text: str, *, doc: str, lang: str, qid: int = 1102432, idx: int = 0, selected: bool = False) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=make_chunk_id(doc, "structure", 0),
        document_id=doc,
        passage_id=doc,
        query_id=qid,
        language=lang,
        language_flores="eng_Latn" if lang == "en" else "hin_Deva",
        chunk_strategy="structure",
        chunk_index=0,
        start_sentence=0,
        end_sentence=0,
        text=text,
        source="MSMARCO-XI",
        passage_index=idx,
        passage_source="english" if lang == "en" else "translated",
        is_selected=selected,
    )


def _retriever(chunks: list[ChunkRecord]) -> DenseRetriever:
    store = QdrantVectorStore(VectorStoreConfig(collection_name="test_chunks"), in_memory=True)
    embedder = FakeEmbedder()
    store.create_collection(embedder.dimension, recreate=True)
    vecs = embedder.embed_documents([c.text for c in chunks])
    store.upsert(chunks, vecs)
    return DenseRetriever(embedder=embedder, store=store, default_top_k=5)


def test_embed_single_query():
    v = FakeEmbedder().embed_query("कॉर्पोरेशन क्या है?")
    assert v.shape == (8,)
    assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-5


def test_embed_batch_and_empty():
    emb = FakeEmbedder()
    mat = emb.embed_documents(["a", "b"])
    assert mat.shape == (2, 8)
    empty = emb.embed_documents([])
    assert empty.shape == (0, 8)


def test_cache_key_is_deterministic():
    a = cache_key("intfloat/multilingual-e5-small", "query: hello")
    b = cache_key("intfloat/multilingual-e5-small", "query: hello")
    c = cache_key("intfloat/multilingual-e5-small", "passage: hello")
    assert a == b
    assert a != c


def test_deterministic_vector_ids():
    cid = "1102432:0:en:structure:0"
    assert point_id_for_chunk(cid) == point_id_for_chunk(cid)
    assert point_id_for_chunk(cid) != point_id_for_chunk("1102432:0:hi:structure:0")


def test_empty_query_returns_no_hits():
    r = _retriever([_chunk("hello world.", doc="1:0:en", lang="en")])
    assert r.search("") == []
    assert r.search("   ") == []


def test_malformed_chunk_skipped_by_chunker():
    chunker = PassageChunker()
    assert chunker.chunk_passage(None) == []
    assert chunker.chunk_passage({"text": "x"}) == []


def test_qdrant_create_upsert_search_and_metadata():
    chunk = _chunk(
        "A corporation is a company recognized in law.",
        doc="1102432:0:en",
        lang="en",
        selected=True,
    )
    r = _retriever([chunk])
    assert r.store.collection_exists()
    assert r.store.count() == 1
    hits = r.search("what is a corporation?")
    assert hits
    h = hits[0]
    assert h.chunk_id == chunk.chunk_id
    assert h.document_id == "1102432:0:en"
    assert h.language == "en"
    assert "corporation" in h.text.lower()
    assert h.metadata["source"] == "MSMARCO-XI"
    assert h.metadata["is_selected"] is True
    assert h.metadata["query_id"] == 1102432


def test_language_filter_and_no_filter():
    chunks = [
        _chunk("A corporation is a company recognized in law.", doc="1102432:0:en", lang="en", idx=0),
        _chunk("निगम कानून में मान्यता प्राप्त कंपनी है।", doc="1102432:0:hi", lang="hi", idx=0),
        _chunk("Unrelated weather passage about rain.", doc="1102432:1:en", lang="en", idx=1),
    ]
    r = _retriever(chunks)
    mixed = r.search("कॉर्पोरेशन क्या है?", top_k=5, language_filter=None)
    langs = {h.language for h in mixed}
    assert "en" in langs or "hi" in langs

    hi_only = r.search("कॉर्पोरेशन क्या है?", top_k=5, language_filter="hi")
    assert hi_only
    assert all(h.language == "hi" for h in hi_only)

    en_only = r.search("what is a corporation?", top_k=5, language_filter="en")
    assert en_only
    assert all(h.language == "en" for h in en_only)


def test_cross_lingual_hindi_query_retrieves_english_passage():
    """Uses MSMARCO-XI corporation pair (query_id 1102432) from inspection."""
    chunks = [
        _chunk(
            "A corporation is a company recognized in law.",
            doc="1102432:0:en",
            lang="en",
            selected=True,
        ),
        _chunk(
            "Today there is a growing community of Certified B Corps.",
            doc="1102432:1:en",
            lang="en",
            idx=1,
        ),
        _chunk(
            "मौसम के बारे में असंबंधित अंश।",
            doc="1102432:1:hi",
            lang="hi",
            idx=1,
        ),
    ]
    r = _retriever(chunks)
    hits = r.search("कॉर्पोरेशन क्या है?", top_k=3, language_filter=None)
    assert hits
    assert hits[0].document_id == "1102432:0:en"
    assert hits[0].language == "en"


def test_cross_lingual_english_query_retrieves_hindi_passage():
    chunks = [
        _chunk(
            "निगम कानून में मान्यता प्राप्त कंपनी है।",
            doc="1102432:0:hi",
            lang="hi",
            selected=True,
        ),
        _chunk("Unrelated filler passage about weather.", doc="1102432:2:en", lang="en", idx=2),
    ]
    r = _retriever(chunks)
    hits = r.search("what is a corporation?", top_k=3, language_filter=None)
    assert hits
    assert hits[0].document_id == "1102432:0:hi"
    assert hits[0].language == "hi"


def test_repeated_upsert_is_idempotent():
    chunk = _chunk("A corporation is a company recognized in law.", doc="1102432:0:en", lang="en")
    store = QdrantVectorStore(VectorStoreConfig(collection_name="idem"), in_memory=True)
    emb = FakeEmbedder()
    store.create_collection(emb.dimension, recreate=True)
    vecs = emb.embed_documents([chunk.text])
    store.upsert([chunk], vecs)
    store.upsert([chunk], vecs)
    assert store.count() == 1
    r = DenseRetriever(embedder=emb, store=store)
    assert r.search("corporation")[0].chunk_id == chunk.chunk_id


def test_rebuild_recreates_collection():
    store = QdrantVectorStore(VectorStoreConfig(collection_name="rebuild"), in_memory=True)
    emb = FakeEmbedder()
    store.create_collection(emb.dimension, recreate=True)
    a = _chunk("alpha", doc="1:0:en", lang="en")
    store.upsert([a], emb.embed_documents([a.text]))
    assert store.count() == 1
    store.create_collection(emb.dimension, recreate=True)
    assert store.count() == 0


def test_e5_wrapper_prefixes_and_cache(tmp_path):
    from backend.retrieval.embeddings import SentenceTransformerEmbedder

    class DummyST:
        def get_sentence_embedding_dimension(self):
            return 4

        def encode(self, batch, **kwargs):
            DummyST.seen = list(batch)
            return np.ones((len(batch), 4), dtype=np.float32)

    DummyST.seen = []
    cfg = EmbeddingConfig(
        model_name="dummy-e5",
        cache_enabled=True,
        cache_dir=tmp_path,
        batch_size=8,
        normalize=False,
        query_prefix="query: ",
        document_prefix="passage: ",
    )
    emb = SentenceTransformerEmbedder(cfg, model=DummyST())
    q = emb.embed_query("hello")
    assert q.shape == (4,)
    assert DummyST.seen == ["query: hello"]
    d = emb.embed_documents(["world"])
    assert d.shape == (1, 4)
    assert DummyST.seen == ["passage: world"]
    DummyST.seen = []
    _ = emb.embed_query("hello")
    assert DummyST.seen == []  # cache hit

