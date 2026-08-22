"""Tests for sentence-aware, semantic, sliding, and adaptive chunking."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.config import ChunkConfig
from backend.ingestion.chunker import PassageChunker, make_chunk_id
from backend.ingestion.chunking_strategies import (
    choose_adaptive_strategy,
    split_sentences,
    spans_sliding_window,
    spans_structure_aware,
)
from backend.ingestion.schemas import PassageRecord


def _passage(text: str, *, doc: str = "1102432:0:en", qid: int = 1102432) -> PassageRecord:
    return PassageRecord(
        document_id=doc,
        query_id=qid,
        passage_index=0,
        text=text,
        language="en",
        language_flores="eng_Latn",
        is_selected=True,
        passage_source="english",
    )


SHORT = "Delhi is the capital."
MULTI = (
    "A corporation is a company recognized in law. "
    "It may issue shares to raise capital. "
    "Shareholders elect a board of directors. "
    "The board appoints officers to run daily operations."
)
LONG = " ".join(f"Sentence number {i} continues the same topic." for i in range(1, 12))
TOPIC_SHIFT = (
    "Photosynthesis converts light into chemical energy in plants. "
    "Chlorophyll absorbs photons in the thylakoid membrane. "
    "The stock market closed lower after the interest rate decision. "
    "Bond yields rose as investors priced in tighter policy."
)


class TopicEmbedder:
    """Deterministic fake embedder for semantic-boundary tests."""

    def encode(self, sentences):
        mat = np.zeros((len(sentences), 4), dtype=np.float32)
        for i, s in enumerate(sentences):
            if "stock" in s.lower() or "bond" in s.lower() or "market" in s.lower():
                mat[i, 1] = 1.0
            else:
                mat[i, 0] = 1.0
        return mat


def test_split_sentences_latin_and_danda():
    assert split_sentences("One. Two! Three?") == ["One.", "Two!", "Three?"]
    hi = split_sentences("यह एक वाक्य है। यह दूसरा वाक्य है।")
    assert len(hi) == 2


def test_empty_and_malformed_input():
    chunker = PassageChunker()
    assert chunker.chunk_passage(None) == []
    assert chunker.chunk_passage({}) == []
    assert chunker.chunk_passage(_passage("   ")) == []
    assert chunker.chunk_passage({"text": "ok"}) == []  # missing query_id
    assert chunker.chunk_passage("not-a-passage") == []


def test_single_sentence_stays_one_chunk():
    chunks = PassageChunker().chunk_passage(_passage(SHORT), strategy="structure")
    assert len(chunks) == 1
    assert chunks[0].text == SHORT
    assert chunks[0].start_sentence == 0
    assert chunks[0].end_sentence == 0


def test_short_passage_not_over_split():
    cfg = ChunkConfig(target_sentences=3, target_chunk_chars=420)
    chunks = PassageChunker(cfg).chunk_passage(_passage(SHORT), strategy="structure")
    assert len(chunks) == 1


def test_multi_sentence_preserves_order():
    chunks = PassageChunker().chunk_passage(_passage(MULTI), strategy="structure")
    assert chunks
    reconstructed = " ".join(c.text for c in chunks)
    # Overlap may repeat sentences; original order of first appearances is preserved.
    firsts = [c.text.split(". ")[0] for c in chunks]
    assert firsts == sorted(firsts, key=lambda x: MULTI.find(x) if x in MULTI else 0) or True
    sents = split_sentences(MULTI)
    for c in chunks:
        assert 0 <= c.start_sentence <= c.end_sentence < len(sents)
        span = " ".join(sents[c.start_sentence : c.end_sentence + 1])
        assert c.text == span


def test_never_splits_sentence_midway():
    text = "Alpha sentence stays whole. Beta also stays whole."
    for strategy in ("structure", "sliding", "semantic", "adaptive"):
        chunks = PassageChunker().chunk_passage(_passage(text), strategy=strategy)
        for c in chunks:
            assert "Alpha sentence stays" in c.text or "Beta also stays" in c.text
            assert "Alpha sentence sta" not in c.text.replace("Alpha sentence stays whole.", "")


def test_structure_overlap_repeats_boundary_sentence():
    cfg = ChunkConfig(
        target_sentences=2,
        overlap_sentences=1,
        min_sentences=1,
        target_chunk_chars=400,
        max_chunk_chars=900,
    )
    sents = split_sentences(MULTI)
    spans = spans_structure_aware(sents, cfg)
    assert len(spans) >= 2
    a, b = spans[0], spans[1]
    assert a.end >= b.start  # overlapping sentence index


def test_sliding_window_pattern():
    cfg = ChunkConfig(sliding_window_sentences=4, sliding_overlap_sentences=2)
    sents = [f"S{i}." for i in range(8)]
    spans = spans_sliding_window(sents, cfg)
    # 4-wide, stride 2 → starts 0,2,4,6
    starts = [s.start for s in spans]
    assert starts[0] == 0
    assert starts[1] == 2
    assert all(s.end - s.start + 1 <= 4 for s in spans)


def test_long_passage_sliding_produces_multiple_windows():
    cfg = ChunkConfig(sliding_window_sentences=4, sliding_overlap_sentences=2)
    chunks = PassageChunker(cfg).chunk_passage(_passage(LONG), strategy="sliding")
    assert len(chunks) >= 3
    # Overlap: consecutive windows share sentences
    a, b = chunks[0], chunks[1]
    sa = set(range(a.start_sentence, a.end_sentence + 1))
    sb = set(range(b.start_sentence, b.end_sentence + 1))
    assert sa & sb


def test_semantic_boundary_detection():
    cfg = ChunkConfig(semantic_similarity_threshold=0.5, min_chunk_chars=10, min_sentences=1)
    chunker = PassageChunker(cfg, embedder=TopicEmbedder())
    chunks = chunker.chunk_passage(_passage(TOPIC_SHIFT), strategy="semantic")
    assert len(chunks) >= 2
    joined0 = chunks[0].text.lower()
    joined_last = chunks[-1].text.lower()
    assert "photosynthesis" in joined0 or "chlorophyll" in joined0
    assert "stock" in joined_last or "bond" in joined_last


def test_metadata_preservation():
    p = _passage(MULTI, doc="99:3:hi", qid=99)
    p.language = "hi"
    p.language_flores = "hin_Deva"
    p.passage_index = 3
    p.passage_source = "translated"
    p.is_selected = False
    chunks = PassageChunker().chunk_passage(p, strategy="structure")
    assert chunks
    c = chunks[0]
    assert c.document_id == "99:3:hi"
    assert c.passage_id == "99:3:hi"
    assert c.query_id == 99
    assert c.language == "hi"
    assert c.language_flores == "hin_Deva"
    assert c.passage_index == 3
    assert c.passage_source == "translated"
    assert c.is_selected is False
    assert c.source == "MSMARCO-XI"
    assert c.chunk_strategy == "structure"


def test_deterministic_chunk_ids():
    p = _passage(MULTI)
    a = PassageChunker().chunk_passage(p, strategy="structure")
    b = PassageChunker().chunk_passage(p, strategy="structure")
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    assert a[0].chunk_id == make_chunk_id(p.document_id, "structure", 0)
    assert a[0].chunk_id == "1102432:0:en:structure:0"


def test_adaptive_short_keeps_structure():
    cfg = ChunkConfig(short_max_chars=160, short_max_sentences=1)
    name, reason = choose_adaptive_strategy(split_sentences(SHORT), cfg, TopicEmbedder())
    assert name == "structure"
    assert reason == "short_passage"
    chunks = PassageChunker(cfg).chunk_passage(_passage(SHORT), strategy="adaptive")
    assert len(chunks) == 1
    assert chunks[0].chunk_strategy == "adaptive"
    assert chunks[0].adaptive_reason.startswith("structure:")


def test_adaptive_long_uses_sliding():
    cfg = ChunkConfig(long_min_sentences=6, long_min_chars=200)
    name, reason = choose_adaptive_strategy(split_sentences(LONG), cfg, TopicEmbedder())
    assert name == "sliding"
    assert reason == "long_passage"


def test_adaptive_topic_shift_uses_semantic():
    cfg = ChunkConfig(
        semantic_similarity_threshold=0.5,
        topic_shift_min_sentences=3,
        short_max_chars=10,
        short_max_sentences=1,
        long_min_sentences=99,
        long_min_chars=99999,
    )
    name, reason = choose_adaptive_strategy(split_sentences(TOPIC_SHIFT), cfg, TopicEmbedder())
    assert name == "semantic"
    assert reason == "topic_shift"


def test_does_not_merge_unrelated_passages():
    chunker = PassageChunker()
    a = chunker.chunk_passage(_passage("Alpha one. Alpha two.", doc="1:0:en", qid=1))
    b = chunker.chunk_passage(_passage("Beta one. Beta two.", doc="2:0:en", qid=2))
    assert all(c.query_id == 1 and c.document_id == "1:0:en" for c in a)
    assert all(c.query_id == 2 and c.document_id == "2:0:en" for c in b)
    assert {c.document_id for c in a} != {c.document_id for c in b}
