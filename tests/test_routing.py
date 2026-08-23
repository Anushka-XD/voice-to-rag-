"""Routing, reranker, and run_query integration tests — LLM always mocked."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.config import GenerationConfig, RoutingConfig
from backend.orchestration.pipeline import VaaniXPipeline
from backend.retrieval.fusion import HybridHit
from backend.retrieval.reranker import Reranker
from backend.routing.query_analyzer import analyze_query
from backend.routing.router import route_for
from tests.test_pipeline import CORP, FakeLLM, FakeReranker, FakeRetriever, _hit

CFG = GenerationConfig(min_rrf_score=0.01, min_query_overlap=0.04, verify_token_overlap=0.12)


SIMPLE = "what is a corporation?"
MODERATE = "How is a corporation formed under commercial law?"
COMPLEX = (
    "What are the differences between a corporation and an LLC, "
    "and which is better for a small business with foreign investors?"
)
HI_SIMPLE = "कॉर्पोरेशन क्या है?"


def test_simple_query_fast():
    a = analyze_query(SIMPLE)
    assert a.route == "FAST"
    assert a.complexity == "simple"
    assert route_for(a).rerank is False


def test_moderate_query_accurate():
    a = analyze_query(MODERATE)
    assert a.route == "ACCURATE"
    assert route_for(a).rerank is True
    assert route_for(a).strict_grounding is False


def test_complex_query_deep():
    a = analyze_query(COMPLEX)
    assert a.route == "DEEP"
    plan = route_for(a)
    assert plan.rerank is True
    assert plan.strict_grounding is True
    assert plan.candidate_k >= route_for(analyze_query(SIMPLE)).candidate_k


def test_multilingual_query_routes():
    a = analyze_query(HI_SIMPLE)
    assert a.language == "hi"
    assert a.route == "FAST"


def test_router_determinism():
    for q in (SIMPLE, MODERATE, COMPLEX, HI_SIMPLE):
        a1 = analyze_query(q)
        a2 = analyze_query(q)
        assert a1.to_dict() == a2.to_dict()
        assert route_for(a1) == route_for(a2)


def _hits():
    return [
        _hit(CORP, cid="good:0", score=0.02),
        _hit("Unrelated weather and rainfall statistics for Mars.", cid="bad:0", score=0.025),
        _hit("A corporation is recognized as a single legal entity.", cid="good:1", score=0.018),
    ]


def test_reranker_ordering_lexical():
    rk = Reranker(RoutingConfig(reranker_backend="lexical"))
    ranked = rk.rerank("what is a corporation?", _hits(), top_n=3)
    assert [h.chunk_id for h in ranked]
    assert ranked[0].chunk_id.startswith("good")
    assert ranked[0].rerank_score is not None
    assert ranked[0].final_score == ranked[0].rerank_score
    assert ranked[0].rrf_score is not None


def test_reranker_metadata_preservation():
    h = _hit(CORP, cid="keep:0", score=0.03)
    h.dense_score = 0.9
    h.bm25_score = 4.2
    h.metadata["passage_id"] = "1102432:0:en"
    ranked = Reranker(RoutingConfig(reranker_backend="lexical")).rerank("corporation", [h], top_n=4)
    assert ranked[0].chunk_id == "keep:0"
    assert ranked[0].dense_score == 0.9
    assert ranked[0].bm25_score == 4.2
    assert ranked[0].metadata["passage_id"] == "1102432:0:en"
    assert ranked[0].language == "en"


def _run(query: str, reranker: FakeReranker) -> dict:
    llm = FakeLLM(
        {
            "answer": "A corporation is a company recognized as a single entity in law.",
            "status": "grounded",
            "sources": [{"chunk_id": "1102432:0:en:adaptive:0"}],
        }
    )
    pipe = VaaniXPipeline(
        retriever=FakeRetriever([_hit(CORP)]),
        llm=llm,
        reranker=reranker,
        config=CFG,
    )
    return pipe.run_query(query)


def test_fast_does_not_invoke_reranker():
    rk = FakeReranker()
    out = _run(SIMPLE, rk)
    assert out["route"] == "FAST"
    assert out["reranked"] is False
    assert rk.calls == 0


def test_accurate_invokes_reranker():
    """ACCURATE still *requests* rerank on the route plan; application is policy-gated."""
    a = analyze_query(MODERATE)
    assert route_for(a).rerank is True


def test_deep_invokes_reranker():
    a = analyze_query(COMPLEX)
    assert route_for(a).rerank is True


def test_latency_fields_and_schema():
    out = _run(SIMPLE, FakeReranker())
    for key in (
        "query_analysis_ms",
        "dense_ms",
        "bm25_ms",
        "fusion_ms",
        "rerank_ms",
        "evidence_ms",
        "generation_ms",
        "grounding_ms",
        "total_ms",
    ):
        assert key in out["latency"]
        assert out["latency"][key] >= 0
    assert out["retrieval_mode"] == "hybrid"
    assert out["query_analysis"]["route"] == "FAST"
    assert "features" in out["query_analysis"]
    json.dumps(out)
