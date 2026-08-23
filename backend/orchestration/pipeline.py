"""
TEXT QUERY → analyze → route → hybrid → optional rerank → evidence → LLM → grounding.

Inject retriever/llm/reranker in tests. Production uses HybridRetriever + OpenAICompatClient.
"""

from __future__ import annotations

import copy
import time
from typing import Any, Callable

from backend.config import GenerationConfig, RerankPolicyConfig, RoutingConfig
from backend.generation.context import EvidenceSet, select_evidence
from backend.generation.llm import LLMError, generate_answer, GenerationResult
from backend.guardrails.relevance import refusal_text, retrieval_is_sufficient
from backend.guardrails.verification import verify_answer
from backend.query.language import detect_query_language
from backend.retrieval.rerank_policy import RerankDecision, decide_rerank
from backend.routing.query_analyzer import QueryAnalysis
from backend.routing.router import RoutePlan, route_for, select_route


def _empty_latency() -> dict[str, float]:
    return {
        "query_analysis_ms": 0.0,
        "dense_ms": 0.0,
        "bm25_ms": 0.0,
        "fusion_ms": 0.0,
        "rerank_ms": 0.0,
        "evidence_ms": 0.0,
        "generation_ms": 0.0,
        "grounding_ms": 0.0,
        "verification_ms": 0.0,
        "retrieval_ms": 0.0,
        "total_ms": 0.0,
    }


def _response(
    *,
    query: str,
    answer: str,
    status: str,
    sources: list[dict[str, Any]],
    query_language: str,
    grounding: dict[str, Any],
    latency: dict[str, float],
    route: str | None = None,
    query_analysis: dict[str, Any] | None = None,
    reranked: bool = False,
    reranking: dict[str, Any] | None = None,
    retrieval_mode: str = "hybrid",
) -> dict[str, Any]:
    return {
        "query": query,
        "answer": answer,
        "status": status,
        "sources": sources,
        "route": route,
        "query_analysis": query_analysis or {},
        "retrieval_mode": retrieval_mode,
        "reranked": reranked,
        "reranking": reranking or {
            "requested": False,
            "applied": False,
            "reason": "not_run",
            "rerank_requested": False,
            "rerank_applied": False,
            "rerank_reason": "not_run",
            "rerank_skipped_reason": None,
        },
        "query_language": query_language,
        "grounding": grounding,
        "latency": latency,
    }


class VaaniXPipeline:
    def __init__(
        self,
        *,
        retriever: Any | None = None,
        llm: Any | None = None,
        reranker: Any | None = None,
        config: GenerationConfig | None = None,
        routing_config: RoutingConfig | None = None,
        rerank_policy: RerankPolicyConfig | None = None,
        retriever_factory: Callable[[], Any] | None = None,
    ) -> None:
        self._retriever = retriever
        self._retriever_factory = retriever_factory
        self._reranker = reranker
        self._reranker_loaded = reranker is not None
        self.llm = llm
        self.config = config or GenerationConfig.from_env()
        self.routing_config = routing_config or RoutingConfig.from_env()
        self.rerank_policy = rerank_policy or RerankPolicyConfig.from_env()

    @property
    def retriever(self) -> Any:
        if self._retriever is None:
            if self._retriever_factory:
                self._retriever = self._retriever_factory()
            else:
                from backend.retrieval.hybrid import HybridRetriever

                self._retriever = HybridRetriever.from_store()
        return self._retriever

    @property
    def reranker(self) -> Any:
        if not self._reranker_loaded:
            from backend.retrieval.reranker import get_reranker

            self._reranker = get_reranker(self.routing_config)
            self._reranker_loaded = True
        return self._reranker

    def warmup(self) -> dict[str, Any]:
        from backend.runtime.warmup import warmup_runtime

        report = warmup_runtime(
            embedder=getattr(self.retriever, "dense", None) and getattr(self.retriever.dense, "embedder", None),
            reranker=self.reranker,
            store=getattr(getattr(self.retriever, "dense", None), "store", None),
            bm25=getattr(self.retriever, "bm25", None),
            hybrid=self.retriever,
            include_reranker=True,
        )
        return {k: v for k, v in report.items() if k not in {"embedder", "reranker", "store", "bm25", "hybrid"}}

    def run_query(self, query: str, *, top_k: int | None = None) -> dict[str, Any]:
        t0 = time.perf_counter()
        q = (query or "").strip()
        lang = detect_query_language(q) if q else "und"
        lat = _empty_latency()
        analysis: QueryAnalysis | None = None
        plan: RoutePlan | None = None
        reranked = False

        if not q:
            lat["total_ms"] = round((time.perf_counter() - t0) * 1000, 3)
            return _response(
                query=query or "",
                answer=refusal_text("en"),
                status="invalid_query",
                sources=[],
                query_language=lang,
                grounding={"verified": False, "confidence": 0.0},
                latency=lat,
                route=None,
                query_analysis={},
                reranked=False,
            )

        t_a = time.perf_counter()
        analysis = select_route(q, self.routing_config)
        plan = route_for(analysis, self.routing_config)
        lat["query_analysis_ms"] = round((time.perf_counter() - t_a) * 1000, 3)
        lang = analysis.language or lang
        analysis_dict = analysis.to_dict()

        k = int(top_k or plan.top_k)
        t_r = time.perf_counter()
        hits, rt = self.retriever.search_timed(
            q,
            top_k=k,
            language_filter=None,
            candidate_k=plan.candidate_k,
        )
        lat["dense_ms"] = round(float(rt.get("dense_ms", 0.0)), 3)
        lat["bm25_ms"] = round(float(rt.get("bm25_ms", 0.0)), 3)
        lat["fusion_ms"] = round(float(rt.get("fusion_ms", 0.0)), 3)
        lat["retrieval_ms"] = round((time.perf_counter() - t_r) * 1000, 3)

        decision: RerankDecision = decide_rerank(hits, plan, self.rerank_policy)
        rerank_info = decision.to_dict()
        if decision.rerank_applied:
            t_rk = time.perf_counter()
            hits = self.reranker.rerank(q, hits, top_n=plan.rerank_top_n or self.routing_config.rerank_top_n)
            lat["rerank_ms"] = round((time.perf_counter() - t_rk) * 1000, 3)
            reranked = True

        gen_cfg = self.config
        if plan.strict_grounding:
            gen_cfg = copy.copy(self.config)
            gen_cfg.verify_token_overlap = max(
                self.config.verify_token_overlap,
                self.routing_config.deep_verify_token_overlap,
            )
            gen_cfg.regenerate_on_fail = True

        ok, reason = retrieval_is_sufficient(hits, q, gen_cfg, query_language=lang)
        if not ok:
            lat["total_ms"] = round((time.perf_counter() - t0) * 1000, 3)
            return _response(
                query=q,
                answer=refusal_text(lang),
                status="insufficient_context",
                sources=[],
                query_language=lang,
                grounding={"verified": False, "confidence": 0.0, "reason": reason},
                latency=lat,
                route=plan.route,
                query_analysis=analysis_dict,
                reranked=reranked,
                reranking=rerank_info,
            )

        t_e = time.perf_counter()
        evidence = select_evidence(hits, gen_cfg)
        lat["evidence_ms"] = round((time.perf_counter() - t_e) * 1000, 3)
        if not evidence.items:
            lat["total_ms"] = round((time.perf_counter() - t0) * 1000, 3)
            return _response(
                query=q,
                answer=refusal_text(lang),
                status="insufficient_context",
                sources=[],
                query_language=lang,
                grounding={"verified": False, "confidence": 0.0, "reason": "no_evidence"},
                latency=lat,
                route=plan.route,
                query_analysis=analysis_dict,
                reranked=reranked,
                reranking=rerank_info,
            )

        meta = {
            "route": plan.route,
            "query_analysis": analysis_dict,
            "reranked": reranked,
            "reranking": rerank_info,
        }
        result = self._call_llm(q, lang, evidence, strict=plan.strict_grounding, lat=lat, t0=t0, **meta)
        if isinstance(result, dict):
            return result
        if result.status == "insufficient_context":
            lat["total_ms"] = round((time.perf_counter() - t0) * 1000, 3)
            return _response(
                query=q,
                answer=result.answer or refusal_text(lang),
                status="insufficient_context",
                sources=[],
                query_language=lang,
                grounding={"verified": False, "confidence": 0.0},
                latency=lat,
                route=plan.route,
                query_analysis=analysis_dict,
                reranked=reranked,
                reranking=rerank_info,
            )

        t_v = time.perf_counter()
        ground = verify_answer(
            result.answer,
            evidence.items,
            gen_cfg,
            query_language=lang,
            cited_chunk_ids={s.get("chunk_id") for s in result.sources if s.get("chunk_id")},
        )
        gms = round((time.perf_counter() - t_v) * 1000, 3)
        lat["grounding_ms"] = gms
        lat["verification_ms"] = gms

        if not ground.verified or ground.has_unsupported:
            if gen_cfg.regenerate_on_fail:
                retry = self._call_llm(q, lang, evidence, strict=True, lat=lat, t0=t0, **meta)
                if isinstance(retry, dict):
                    return retry
                result = retry
                t_v2 = time.perf_counter()
                ground = verify_answer(
                    result.answer,
                    evidence.items,
                    gen_cfg,
                    query_language=lang,
                    cited_chunk_ids={s.get("chunk_id") for s in result.sources if s.get("chunk_id")},
                )
                extra = round((time.perf_counter() - t_v2) * 1000, 3)
                lat["grounding_ms"] = round(lat["grounding_ms"] + extra, 3)
                lat["verification_ms"] = lat["grounding_ms"]
                if result.status == "insufficient_context":
                    lat["total_ms"] = round((time.perf_counter() - t0) * 1000, 3)
                    return _response(
                        query=q,
                        answer=result.answer or refusal_text(lang),
                        status="insufficient_context",
                        sources=[],
                        query_language=lang,
                        grounding={"verified": False, "confidence": ground.confidence},
                        latency=lat,
                        route=plan.route,
                        query_analysis=analysis_dict,
                        reranked=reranked,
                        reranking=rerank_info,
                    )
            if not ground.verified or ground.has_unsupported:
                lat["total_ms"] = round((time.perf_counter() - t0) * 1000, 3)
                return _response(
                    query=q,
                    answer=refusal_text(lang),
                    status="insufficient_context",
                    sources=[],
                    query_language=lang,
                    grounding={
                        "verified": False,
                        "confidence": ground.confidence,
                        "reason": "unsupported_claims",
                    },
                    latency=lat,
                    route=plan.route,
                    query_analysis=analysis_dict,
                    reranked=reranked,
                    reranking=rerank_info,
                )

        allowed_ids = {i.chunk_id for i in evidence.items}
        sources = [
            s for s in evidence.source_dicts()
            if s["chunk_id"] in allowed_ids
            and (
                not result.sources
                or s["chunk_id"] in {x.get("chunk_id") for x in result.sources}
            )
        ]
        if not sources:
            sources = evidence.source_dicts()

        lat["total_ms"] = round((time.perf_counter() - t0) * 1000, 3)
        return _response(
            query=q,
            answer=result.answer,
            status="grounded",
            sources=[{"chunk_id": s["chunk_id"], "language": s["language"], "score": s["score"]} for s in sources],
            query_language=lang,
            grounding={"verified": True, "confidence": ground.confidence},
            latency=lat,
            route=plan.route,
            query_analysis=analysis_dict,
            reranked=reranked,
            reranking=rerank_info,
        )

    def _call_llm(
        self,
        query: str,
        lang: str,
        evidence: EvidenceSet,
        *,
        strict: bool,
        lat: dict[str, float],
        t0: float,
        route: str | None = None,
        query_analysis: dict[str, Any] | None = None,
        reranked: bool = False,
        reranking: dict[str, Any] | None = None,
    ) -> GenerationResult | dict[str, Any]:
        t_g = time.perf_counter()
        try:
            result = generate_answer(
                query,
                lang,
                evidence,
                client=self.llm,
                config=self.config,
                strict=strict,
            )
        except LLMError:
            lat["generation_ms"] = round(lat.get("generation_ms", 0.0) + (time.perf_counter() - t_g) * 1000, 3)
            lat["total_ms"] = round((time.perf_counter() - t0) * 1000, 3)
            return _response(
                query=query,
                answer=refusal_text(lang),
                status="insufficient_context",
                sources=[],
                query_language=lang,
                grounding={"verified": False, "confidence": 0.0, "reason": "llm_error"},
                latency=lat,
                route=route,
                query_analysis=query_analysis,
                reranked=reranked,
                reranking=reranking,
            )
        lat["generation_ms"] = round(lat.get("generation_ms", 0.0) + (time.perf_counter() - t_g) * 1000, 3)
        return result


def run_query(query: str, *, pipeline: VaaniXPipeline | None = None, **kwargs: Any) -> dict[str, Any]:
    pipe = pipeline or VaaniXPipeline()
    return pipe.run_query(query, **kwargs)
