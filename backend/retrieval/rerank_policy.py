"""Decide whether a hybrid candidate list is worth a cross-encoder pass."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from backend.config import RerankPolicyConfig
from backend.retrieval.fusion import HybridHit
from backend.routing.router import RoutePlan


@dataclass
class RerankDecision:
    rerank_requested: bool
    rerank_applied: bool
    rerank_reason: str
    rerank_skipped_reason: str | None = None
    top_rrf: float = 0.0
    score_gap: float = 0.0
    n_strong: int = 0

    def to_dict(self) -> dict:
        return {
            "requested": self.rerank_requested,
            "applied": self.rerank_applied,
            "reason": self.rerank_reason if self.rerank_applied else (self.rerank_skipped_reason or self.rerank_reason),
            "rerank_requested": self.rerank_requested,
            "rerank_applied": self.rerank_applied,
            "rerank_reason": self.rerank_reason,
            "rerank_skipped_reason": self.rerank_skipped_reason,
            "top_rrf": self.top_rrf,
            "score_gap": self.score_gap,
            "n_strong": self.n_strong,
        }


def _rrf(hit: HybridHit) -> float:
    if hit.rrf_score is not None:
        return float(hit.rrf_score)
    return float(hit.score or 0.0)


def hybrid_signals(hits: Sequence[HybridHit], config: RerankPolicyConfig) -> tuple[float, float, int]:
    if not hits:
        return 0.0, 0.0, 0
    scores = sorted((_rrf(h) for h in hits), reverse=True)
    top = scores[0]
    gap = (scores[0] - scores[1]) if len(scores) > 1 else top
    n_strong = sum(1 for s in scores if s >= config.strong_rrf)
    return top, gap, n_strong


def is_confident(top: float, gap: float, n_strong: int, config: RerankPolicyConfig) -> bool:
    if top < config.min_top_rrf:
        return False
    if gap < config.min_score_gap and n_strong > 1:
        return False
    if n_strong > config.max_strong_for_confident:
        return False
    return True


def decide_rerank(
    hits: Sequence[HybridHit],
    plan: RoutePlan,
    config: RerankPolicyConfig | None = None,
) -> RerankDecision:
    cfg = config or RerankPolicyConfig.from_env()
    top, gap, n_strong = hybrid_signals(hits, cfg)
    route_wants = bool(plan.rerank)
    confident = is_confident(top, gap, n_strong, cfg)

    if plan.route == "FAST" and not cfg.allow_fast_rerank:
        return RerankDecision(
            rerank_requested=False,
            rerank_applied=False,
            rerank_reason="FAST prefers hybrid only",
            rerank_skipped_reason="route_fast",
            top_rrf=top,
            score_gap=gap,
            n_strong=n_strong,
        )

    if plan.route == "DEEP":
        very_strong = top >= cfg.deep_skip_min_rrf and (gap >= cfg.min_score_gap or n_strong <= 1)
        if very_strong:
            return RerankDecision(
                rerank_requested=True,
                rerank_applied=False,
                rerank_reason="DEEP prefers rerank when hybrid is ambiguous",
                rerank_skipped_reason="hybrid confidence was sufficient",
                top_rrf=top,
                score_gap=gap,
                n_strong=n_strong,
            )
        return RerankDecision(
            rerank_requested=True,
            rerank_applied=True,
            rerank_reason="DEEP and hybrid ranking is not uniquely strong",
            top_rrf=top,
            score_gap=gap,
            n_strong=n_strong,
        )

    # ACCURATE (and FAST if allow_fast_rerank)
    if confident:
        return RerankDecision(
            rerank_requested=route_wants,
            rerank_applied=False,
            rerank_reason="route may request rerank",
            rerank_skipped_reason="hybrid confidence was sufficient",
            top_rrf=top,
            score_gap=gap,
            n_strong=n_strong,
        )
    if not route_wants and plan.route != "ACCURATE":
        return RerankDecision(
            rerank_requested=False,
            rerank_applied=False,
            rerank_reason="route does not request rerank",
            rerank_skipped_reason="route_does_not_request",
            top_rrf=top,
            score_gap=gap,
            n_strong=n_strong,
        )
    return RerankDecision(
        rerank_requested=True,
        rerank_applied=True,
        rerank_reason="hybrid ranking is ambiguous",
        top_rrf=top,
        score_gap=gap,
        n_strong=n_strong,
    )
