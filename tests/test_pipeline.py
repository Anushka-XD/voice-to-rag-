"""Text RAG pipeline tests — LLM is always mocked."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.config import GenerationConfig
from backend.generation.context import select_evidence
from backend.generation.llm import LLMError, parse_generation_json
from backend.orchestration.pipeline import VaaniXPipeline, run_query
from backend.query.language import detect_query_language
from backend.retrieval.fusion import HybridHit


def _hit(text: str, *, cid: str = "1102432:0:en:adaptive:0", lang: str = "en", score: float = 0.03) -> HybridHit:
    return HybridHit(
        chunk_id=cid,
        document_id="1102432:0:en",
        text=text,
        language=lang,
        score=score,
        metadata={"chunk_strategy": "adaptive", "passage_id": "1102432:0:en", "source": "MSMARCO-XI"},
    )


class FakeRetriever:
    def __init__(self, hits: list[HybridHit]):
        self.hits = hits

    def search_timed(self, query, top_k=None, language_filter=None, candidate_k=None):
        return list(self.hits), {"dense_ms": 0.4, "bm25_ms": 0.2, "fusion_ms": 0.1, "total_ms": 1.0}


class FakeReranker:
    def __init__(self):
        self.calls = 0

    def rerank(self, query, hits, *, top_n=None):
        self.calls += 1
        n = int(top_n or len(hits))
        out = list(hits)[:n]
        for i, h in enumerate(out):
            h.rerank_score = 1.0 - i * 0.01
            h.final_score = h.rerank_score
            h.rrf_score = h.rrf_score if h.rrf_score is not None else h.score
        return out


class FakeLLM:
    def __init__(self, payload=None, error=None, raw=None):
        self.payload = payload
        self.error = error
        self.raw = raw
        self.calls = 0

    def complete(self, *, system: str, user: str) -> str:
        self.calls += 1
        if self.error:
            raise self.error
        if self.raw is not None:
            return self.raw
        return json.dumps(self.payload)


CFG = GenerationConfig(
    min_rrf_score=0.01,
    min_query_overlap=0.04,
    verify_token_overlap=0.12,
    regenerate_on_fail=True,
    llm_retries=2,
    max_evidence_chunks=5,
    max_context_chars=3500,
    duplicate_jaccard=0.85,
)


def _pipe(hits, llm, reranker=None) -> VaaniXPipeline:
    rk = reranker if reranker is not None else FakeReranker()
    return VaaniXPipeline(retriever=FakeRetriever(hits), llm=llm, reranker=rk, config=CFG)


CORP = "A corporation is a company or group of people authorized to act as a single entity and recognized as such in law."


def test_successful_grounded_answer():
    llm = FakeLLM(
        {
            "answer": "A corporation is a company recognized as a single entity in law.",
            "status": "grounded",
            "sources": [{"chunk_id": "1102432:0:en:adaptive:0"}],
        }
    )
    out = _pipe([_hit(CORP)], llm).run_query("what is a corporation?")
    assert out["status"] == "grounded"
    assert "corporation" in out["answer"].lower()
    assert out["sources"][0]["chunk_id"] == "1102432:0:en:adaptive:0"
    assert out["sources"][0]["language"] == "en"
    assert out["grounding"]["verified"] is True
    assert out["retrieval_mode"] == "hybrid"


def test_insufficient_retrieval():
    out = _pipe([], FakeLLM({"answer": "nope", "status": "grounded", "sources": []})).run_query(
        "What is the capital of Mars?"
    )
    assert out["status"] == "insufficient_context"
    assert out["sources"] == []
    assert "enough information" in out["answer"].lower() or "विश्वसनीय" in out["answer"]


def test_unsupported_generated_answer():
    llm = FakeLLM(
        {
            "answer": "The capital of France is Paris and it has 12 million residents.",
            "status": "grounded",
            "sources": [{"chunk_id": "1102432:0:en:adaptive:0"}],
        }
    )
    out = _pipe([_hit(CORP)], llm).run_query("what is a corporation?")
    assert out["status"] == "insufficient_context"
    assert out["grounding"].get("reason") == "unsupported_claims" or out["status"] == "insufficient_context"


def test_empty_query():
    out = run_query("", pipeline=_pipe([_hit(CORP)], FakeLLM({"answer": "x", "status": "grounded", "sources": []})))
    assert out["status"] == "invalid_query"
    assert out["sources"] == []


def test_llm_timeout():
    llm = FakeLLM(error=LLMError("LLM timeout"))
    out = _pipe([_hit(CORP)], llm).run_query("what is a corporation?")
    assert out["status"] == "insufficient_context"
    assert out["grounding"].get("reason") == "llm_error"


def test_malformed_llm_response():
    llm = FakeLLM(raw="this is not json at all")
    out = _pipe([_hit(CORP)], llm).run_query("what is a corporation?")
    assert out["status"] == "insufficient_context"


def test_duplicate_evidence_removal():
    a = _hit(CORP, cid="a:en:0", score=0.04)
    b = _hit(CORP + " ", cid="b:en:0", score=0.03)
    ev = select_evidence([a, b], CFG)
    assert len(ev.items) == 1
    assert ev.dropped_duplicates >= 1


def test_multilingual_query():
    llm = FakeLLM(
        {
            "answer": "निगम एक कंपनी है जो कानून में एक इकाई के रूप में मान्य है।",
            "status": "grounded",
            "sources": [{"chunk_id": "1102432:0:en:adaptive:0"}],
        }
    )
    out = _pipe([_hit(CORP)], llm).run_query("कॉर्पोरेशन क्या है?")
    assert out["query_language"] == "hi"
    assert out["status"] == "grounded"
    assert detect_query_language("कॉर्पोरेशन क्या है?") == "hi"


def test_source_metadata_preservation():
    llm = FakeLLM(
        {
            "answer": "A corporation is a company recognized as a single entity in law.",
            "status": "grounded",
            "sources": [{"chunk_id": "1102432:0:en:adaptive:0"}],
        }
    )
    out = _pipe([_hit(CORP)], llm).run_query("what is a corporation?")
    src = out["sources"][0]
    assert "chunk_id" in src and "language" in src and "score" in src


def test_final_response_schema():
    llm = FakeLLM(
        {
            "answer": "A corporation is a company recognized as a single entity in law.",
            "status": "grounded",
            "sources": [{"chunk_id": "1102432:0:en:adaptive:0"}],
        }
    )
    out = _pipe([_hit(CORP)], llm).run_query("what is a corporation?")
    for key in (
        "query",
        "answer",
        "status",
        "sources",
        "retrieval_mode",
        "grounding",
        "latency",
        "route",
        "query_analysis",
        "reranked",
        "reranking",
    ):
        assert key in out
    assert "requested" in out["reranking"]
    assert "applied" in out["reranking"]
    for key in (
        "query_analysis_ms",
        "dense_ms",
        "bm25_ms",
        "fusion_ms",
        "rerank_ms",
        "evidence_ms",
        "generation_ms",
        "grounding_ms",
        "retrieval_ms",
        "verification_ms",
        "total_ms",
    ):
        assert key in out["latency"]
        assert isinstance(out["latency"][key], (int, float))


def test_parse_generation_json_rejects_empty():
    try:
        parse_generation_json("")
        assert False
    except LLMError:
        pass
