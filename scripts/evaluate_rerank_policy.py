#!/usr/bin/env python3
"""
Compare always-rerank vs never-rerank vs adaptive policy.

  python scripts/evaluate_rerank_policy.py --eval data/eval/retrieval_eval.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Literal

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.config import EVAL_DIR, REPORTS_DIR, RerankPolicyConfig, RoutingConfig
from backend.eval.dataset import load_jsonl
from backend.eval.metrics import latency_summary, mean_finite, mrr, recall_at_k
from backend.retrieval.rerank_policy import decide_rerank
from backend.routing.query_analyzer import analyze_query
from backend.routing.router import route_for
from backend.runtime.warmup import warmup_runtime

logger = logging.getLogger(__name__)

Strategy = Literal["always", "never", "adaptive"]


def _eval_strategy(
    name: Strategy,
    queries: list[dict[str, Any]],
    hybrid: Any,
    reranker: Any,
    policy: RerankPolicyConfig,
    top_k: int,
) -> dict[str, Any]:
    r5: list[float] = []
    r10: list[float] = []
    mrrs: list[float] = []
    lats: list[float] = []
    applied = 0
    rt_cfg = RoutingConfig.from_env()
    for item in queries:
        gold = set(item["gold_document_ids"])
        t0 = time.perf_counter()
        hits, _lat = hybrid.search_timed(item["query"], top_k=top_k)
        analysis = analyze_query(item["query"], rt_cfg)
        plan = route_for(analysis, rt_cfg)
        use = False
        if name == "always":
            use = True
        elif name == "never":
            use = False
        else:
            use = decide_rerank(hits, plan, policy).rerank_applied
        if use:
            hits = reranker.rerank(item["query"], hits, top_n=rt_cfg.rerank_top_n)
            applied += 1
        elapsed = round((time.perf_counter() - t0) * 1000, 3)
        ranked = [h.document_id for h in hits]
        r5.append(recall_at_k(ranked, gold, 5))
        r10.append(recall_at_k(ranked, gold, 10))
        mrrs.append(mrr(ranked, gold))
        lats.append(elapsed)
    n = len(queries) or 1
    return {
        "strategy": name,
        "n_queries": len(queries),
        "recall_at_5": mean_finite(r5),
        "recall_at_10": mean_finite(r10),
        "mrr": mean_finite(mrrs),
        "latency_ms": latency_summary(lats),
        "pct_reranked": round(100.0 * applied / n, 1),
        "n_reranked": applied,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate adaptive rerank policy")
    parser.add_argument("--eval", type=Path, default=EVAL_DIR / "retrieval_eval.jsonl")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--report", type=Path, default=REPORTS_DIR / "rerank_policy_evaluation.json")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    if not args.eval.exists():
        logger.error("Missing %s", args.eval)
        return 1
    queries = [r for r in load_jsonl(args.eval) if r.get("gold_document_ids")]
    warm = warmup_runtime(include_reranker=True)
    policy = RerankPolicyConfig.from_env()
    always = _eval_strategy("always", queries, warm["hybrid"], warm["reranker"], policy, args.top_k)
    never = _eval_strategy("never", queries, warm["hybrid"], warm["reranker"], policy, args.top_k)
    adaptive = _eval_strategy("adaptive", queries, warm["hybrid"], warm["reranker"], policy, args.top_k)
    report = {
        "n_queries": len(queries),
        "policy": {
            "min_top_rrf": policy.min_top_rrf,
            "min_score_gap": policy.min_score_gap,
            "strong_rrf": policy.strong_rrf,
            "deep_skip_min_rrf": policy.deep_skip_min_rrf,
            "allow_fast_rerank": policy.allow_fast_rerank,
        },
        "cold_start_ms": warm["cold_start_ms"],
        "always": always,
        "never": never,
        "adaptive": adaptive,
        "note": (
            "Quality/latency tradeoff. Do not assume rerank helps; compare MRR and p50. "
            "Latencies are warm (models loaded before the first scored query)."
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
