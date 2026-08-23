"""Adaptive rerank policy tests."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.config import GenerationConfig, RerankPolicyConfig
from backend.orchestration.pipeline import VaaniXPipeline
from backend.retrieval.fusion import HybridHit
from backend.retrieval.rerank_policy import decide_rerank
from backend.routing.query_analyzer import analyze_query
from backend.routing.router import route_for
from tests.test_pipeline import CORP, FakeLLM, FakeReranker, FakeRetriever, _hit
from tests.test_routing import COMPLEX, MODERATE, SIMPLE

CFG = GenerationConfig(min_rrf_score=0.01, min_query_overlap=0.04, verify_token_overlap=0.12)
POL = RerankPolicyConfig()


def _hh(score: float, cid: str = "c0") -> HybridHit:
    return HybridHit(
        chunk_id=cid,
        document_id=cid,
        text=CORP,
        language="en",
        score=score,
        rrf_score=score,
    )


def test_confident_retrieval_skips_reranking():
    plan = route_for(analyze_query(MODERATE))
    d = decide_rerank([_hh(0.04), _hh(0.01, "c1")], plan, POL)
    assert d.rerank_requested is True
    assert d.rerank_applied is False
    assert d.rerank_skipped_reason == "hybrid confidence was sufficient"


def test_ambiguous_retrieval_triggers_reranking():
    plan = route_for(analyze_query(MODERATE))
    d = decide_rerank([_hh(0.019), _hh(0.0185, "c1")], plan, POL)
    assert d.rerank_applied is True
    assert "ambiguous" in d.rerank_reason


def test_fast_skips_reranking_policy():
    plan = route_for(analyze_query(SIMPLE))
    d = decide_rerank([_hh(0.016), _hh(0.0159, "c1")], plan, POL)
    assert plan.rerank is False
    assert d.rerank_applied is False
    assert d.rerank_skipped_reason == "route_fast"


def test_accurate_can_skip_when_confident():
    plan = route_for(analyze_query(MODERATE))
    assert plan.rerank is True
    d = decide_rerank([_hh(0.035)], plan, POL)
    assert d.rerank_applied is False


def test_deep_can_request_reranking():
    plan = route_for(analyze_query(COMPLEX))
    assert plan.rerank is True
    d = decide_rerank([_hh(0.02), _hh(0.019, "c1")], plan, POL)
    assert d.rerank_applied is True
    assert d.rerank_requested is True


def test_deep_skips_when_uniquely_strong():
    plan = route_for(analyze_query(COMPLEX))
    d = decide_rerank([_hh(0.045), _hh(0.01, "c1")], plan, POL)
    assert d.rerank_requested is True
    assert d.rerank_applied is False


def _pipe(hits, rk):
    llm = FakeLLM(
        {
            "answer": "A corporation is a company recognized as a single entity in law.",
            "status": "grounded",
            "sources": [{"chunk_id": "1102432:0:en:adaptive:0"}],
        }
    )
    return VaaniXPipeline(
        retriever=FakeRetriever(hits),
        llm=llm,
        reranker=rk,
        config=CFG,
        rerank_policy=POL,
    )


def test_pipeline_fast_skips_reranker():
    rk = FakeReranker()
    out = _pipe([_hit(CORP, score=0.016)], rk).run_query(SIMPLE)
    assert out["route"] == "FAST"
    assert out["reranked"] is False
    assert rk.calls == 0
    assert out["reranking"]["applied"] is False


def test_pipeline_accurate_skips_when_confident():
    rk = FakeReranker()
    out = _pipe([_hit(CORP, score=0.04)], rk).run_query(MODERATE)
    assert out["route"] == "ACCURATE"
    assert rk.calls == 0
    assert out["reranking"]["applied"] is False
    assert "sufficient" in (out["reranking"].get("reason") or "")


def test_pipeline_accurate_reranks_when_ambiguous():
    rk = FakeReranker()
    hits = [_hit(CORP, cid="a", score=0.017), _hit(CORP, cid="b", score=0.0165)]
    out = _pipe(hits, rk).run_query(MODERATE)
    assert out["route"] == "ACCURATE"
    assert rk.calls == 1
    assert out["reranking"]["applied"] is True


def test_pipeline_deep_requests_rerank():
    rk = FakeReranker()
    hits = [_hit(CORP, cid="a", score=0.02), _hit(CORP, cid="b", score=0.019)]
    out = _pipe(hits, rk).run_query(COMPLEX)
    assert out["route"] == "DEEP"
    assert rk.calls == 1
    assert out["reranking"]["requested"] is True
    assert out["reranking"]["applied"] is True
