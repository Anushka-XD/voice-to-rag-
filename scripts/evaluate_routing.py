#!/usr/bin/env python3
"""
Evaluate FAST/ACCURATE/DEEP routing and hybrid vs hybrid+rerank on MSMARCO-XI gold.

  python scripts/evaluate_routing.py
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.config import CLEAN_DIR, EmbeddingConfig, HybridConfig, REPORTS_DIR, RoutingConfig, VectorStoreConfig
from backend.retrieval.bm25 import BM25Retriever
from backend.retrieval.dense import DenseRetriever
from backend.retrieval.embeddings import create_embedder
from backend.retrieval.hybrid import HybridRetriever
from backend.retrieval.reranker import Reranker
from backend.retrieval.vector_store import QdrantVectorStore
from backend.routing.query_analyzer import analyze_query
from backend.routing.router import route_for
from scripts.evaluate_retrieval import load_eval_queries, mean_finite, mrr, recall_at_k

logger = logging.getLogger(__name__)


def _lat_stats(vals: list[float]) -> dict[str, float | None]:
    if not vals:
        return {"p50": None, "mean": None}
    return {
        "p50": round(float(statistics.median(vals)), 3),
        "mean": round(float(statistics.mean(vals)), 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate query routing and reranking")
    parser.add_argument("--examples", type=Path, default=CLEAN_DIR / "examples_validation.jsonl")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--report", type=Path, default=REPORTS_DIR / "routing_evaluation.json")
    parser.add_argument("--max-queries", type=int, default=40)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")

    if not args.examples.exists():
        logger.error("Missing %s", args.examples)
        return 1

    vs_cfg = VectorStoreConfig.from_env()
    hy_cfg = HybridConfig.from_env()
    rt_cfg = RoutingConfig.from_env()
    store = QdrantVectorStore(vs_cfg)
    if not store.collection_exists() or store.count() == 0:
        logger.error("Empty Qdrant collection. Run scripts/build_vector_index.py first.")
        return 1

    embedder = create_embedder(EmbeddingConfig.from_env())
    dense = DenseRetriever(embedder=embedder, store=store, default_top_k=args.top_k)
    bm25 = BM25Retriever.from_qdrant(store, config=hy_cfg)
    hybrid = HybridRetriever(dense, bm25, hy_cfg)
    reranker = Reranker(rt_cfg)

    raw = load_eval_queries(args.examples)
    gold_rows = [r for r in raw if r.get("gold_document_ids")]
    queries = gold_rows[: args.max_queries] if args.max_queries else gold_rows

    routes: list[str] = []
    examples_by_route: dict[str, list[str]] = {"FAST": [], "ACCURATE": [], "DEEP": []}
    hybrid_r5: list[float] = []
    hybrid_r10: list[float] = []
    hybrid_mrr: list[float] = []
    rerank_r5: list[float] = []
    rerank_r10: list[float] = []
    rerank_mrr: list[float] = []
    lat_hybrid: list[float] = []
    lat_rerank: list[float] = []
    lat_by_route: dict[str, dict[str, list[float]]] = {
        r: {"analysis": [], "retrieval": [], "rerank": [], "total": []} for r in ("FAST", "ACCURATE", "DEEP")
    }
    per_query: list[dict[str, Any]] = []

    import time

    for item in queries:
        q = item["query"]
        gold = set(item["gold_document_ids"])
        t0 = time.perf_counter()
        analysis = analyze_query(q, rt_cfg)
        plan = route_for(analysis, rt_cfg)
        analysis_ms = round((time.perf_counter() - t0) * 1000, 3)
        routes.append(analysis.route)
        if len(examples_by_route[analysis.route]) < 3:
            examples_by_route[analysis.route].append(q)

        cand = plan.candidate_k
        hits, rt = hybrid.search_timed(q, top_k=args.top_k, language_filter=None, candidate_k=cand)
        ranked_h = [h.document_id for h in hits]
        hybrid_r5.append(recall_at_k(ranked_h, gold, 5))
        hybrid_r10.append(recall_at_k(ranked_h, gold, 10))
        hybrid_mrr.append(mrr(ranked_h, gold))
        retr_ms = float(rt.get("total_ms") or 0.0)
        lat_hybrid.append(retr_ms)

        t_rk = time.perf_counter()
        reranked = reranker.rerank(q, hits, top_n=rt_cfg.rerank_top_n)
        rerank_ms = round((time.perf_counter() - t_rk) * 1000, 3)
        ranked_r = [h.document_id for h in reranked]
        rerank_r5.append(recall_at_k(ranked_r, gold, 5))
        rerank_r10.append(recall_at_k(ranked_r, gold, 10))
        rerank_mrr.append(mrr(ranked_r, gold))
        lat_rerank.append(rerank_ms)

        routed_rerank_ms = rerank_ms if plan.rerank else 0.0
        total_ms = round(analysis_ms + retr_ms + routed_rerank_ms, 3)
        bucket = lat_by_route[analysis.route]
        bucket["analysis"].append(analysis_ms)
        bucket["retrieval"].append(retr_ms)
        bucket["rerank"].append(routed_rerank_ms)
        bucket["total"].append(total_ms)

        per_query.append(
            {
                "query_id": item.get("query_id"),
                "query": q,
                "route": analysis.route,
                "query_type": analysis.query_type,
                "language": analysis.language,
                "hybrid_recall_at_10": recall_at_k(ranked_h, gold, 10),
                "rerank_recall_at_10": recall_at_k(ranked_r, gold, 10),
                "hybrid_mrr": mrr(ranked_h, gold),
                "rerank_mrr": mrr(ranked_r, gold),
                "latency_ms": {
                    "query_analysis_ms": analysis_ms,
                    "dense_ms": rt.get("dense_ms"),
                    "bm25_ms": rt.get("bm25_ms"),
                    "fusion_ms": rt.get("fusion_ms"),
                    "rerank_ms": rerank_ms,
                    "total_ms": total_ms,
                },
            }
        )

    counts = Counter(routes)
    n = len(queries) or 1
    report = {
        "dataset": "ai4bharat/MSMARCO-XI",
        "n_queries": len(queries),
        "note": (
            "hybrid_plus_rerank always reranks for comparison. "
            "latency_by_route_ms follows production policy (FAST rerank_ms=0). "
            "This clean split has only 10 queries; none classified DEEP."
        ),
        "reranker_backend": reranker.backend,
        "reranker_model": rt_cfg.reranker_model if reranker.backend == "cross_encoder" else None,
        "route_distribution": {
            "FAST": counts.get("FAST", 0),
            "ACCURATE": counts.get("ACCURATE", 0),
            "DEEP": counts.get("DEEP", 0),
            "FAST_pct": round(100.0 * counts.get("FAST", 0) / n, 1),
            "ACCURATE_pct": round(100.0 * counts.get("ACCURATE", 0) / n, 1),
            "DEEP_pct": round(100.0 * counts.get("DEEP", 0) / n, 1),
        },
        "example_queries": examples_by_route,
        "hybrid": {
            "recall_at_5": mean_finite(hybrid_r5),
            "recall_at_10": mean_finite(hybrid_r10),
            "mrr": mean_finite(hybrid_mrr),
            "latency_ms": _lat_stats(lat_hybrid),
        },
        "hybrid_plus_rerank": {
            "recall_at_5": mean_finite(rerank_r5),
            "recall_at_10": mean_finite(rerank_r10),
            "mrr": mean_finite(rerank_mrr),
            "rerank_latency_ms": _lat_stats(lat_rerank),
        },
        "rerank_delta": {
            "recall_at_5": round((mean_finite(rerank_r5) or 0) - (mean_finite(hybrid_r5) or 0), 4),
            "recall_at_10": round((mean_finite(rerank_r10) or 0) - (mean_finite(hybrid_r10) or 0), 4),
            "mrr": round((mean_finite(rerank_mrr) or 0) - (mean_finite(hybrid_mrr) or 0), 4),
        },
        "latency_by_route_ms": {
            route: {
                "query_analysis": _lat_stats(vals["analysis"]),
                "retrieval": _lat_stats(vals["retrieval"]),
                "rerank": _lat_stats(vals["rerank"]),
                "total": _lat_stats(vals["total"]),
            }
            for route, vals in lat_by_route.items()
        },
        "queries": per_query,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = {k: v for k, v in report.items() if k != "queries"}
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
