"""Map query analysis to retrieval knobs. Classification itself lives in query_analyzer."""

from __future__ import annotations

from dataclasses import dataclass

from backend.config import RoutingConfig
from backend.routing.query_analyzer import QueryAnalysis, analyze_query


@dataclass(frozen=True)
class RoutePlan:
    route: str
    top_k: int
    candidate_k: int
    rerank: bool
    rerank_top_n: int
    strict_grounding: bool


def select_route(query: str, config: RoutingConfig | None = None) -> QueryAnalysis:
    return analyze_query(query, config)


def route_for(analysis: QueryAnalysis, config: RoutingConfig | None = None) -> RoutePlan:
    cfg = config or RoutingConfig.from_env()
    route = analysis.route
    if route == "DEEP":
        return RoutePlan(
            route="DEEP",
            top_k=cfg.deep_top_k,
            candidate_k=cfg.deep_candidate_k,
            rerank=True,
            rerank_top_n=max(cfg.rerank_top_n, cfg.deep_top_k),
            strict_grounding=True,
        )
    if route == "ACCURATE":
        return RoutePlan(
            route="ACCURATE",
            top_k=cfg.accurate_top_k,
            candidate_k=cfg.accurate_candidate_k,
            rerank=True,
            rerank_top_n=cfg.rerank_top_n,
            strict_grounding=False,
        )
    return RoutePlan(
        route="FAST",
        top_k=cfg.fast_top_k,
        candidate_k=cfg.fast_candidate_k,
        rerank=False,
        rerank_top_n=0,
        strict_grounding=False,
    )
