#!/usr/bin/env python3
"""
Dense vs BM25 vs Hybrid vs Hybrid+rerank on a labeled eval JSONL.

  python scripts/evaluate_retrieval.py --eval data/eval/retrieval_eval.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.config import CLEAN_DIR, EVAL_DIR, REPORTS_DIR, RoutingConfig
from backend.eval.dataset import load_jsonl
from backend.eval.metrics import breakdown, metric_block, mrr, recall_at_k
from backend.retrieval.reranker import get_reranker
from backend.runtime.warmup import warmup_runtime

logger = logging.getLogger(__name__)


def _run_system(
    name: str,
    search: Callable[..., tuple[list[Any], dict[str, float]]],
    queries: list[dict[str, Any]],
    top_k: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for item in queries:
        gold = set(item["gold_document_ids"])
        t0 = time.perf_counter()
        hits, lat = search(item["query"], top_k=top_k)
        elapsed = float(lat.get("total_ms") or (time.perf_counter() - t0) * 1000)
        ranked = [h.document_id for h in hits]
        row = {
            "query_id": item.get("query_id"),
            "language": item.get("query_language") or "unknown",
            "route": item.get("expected_route") or "unknown",
            "complexity": item.get("complexity") or "unknown",
            "query_type": item.get("query_type") or "unknown",
            "r1": recall_at_k(ranked, gold, 1),
            "r5": recall_at_k(ranked, gold, 5),
            "r10": recall_at_k(ranked, gold, 10),
            "mrr": mrr(ranked, gold),
            "lat": elapsed,
        }
        rows.append(row)
    block = metric_block(
        [r["r1"] for r in rows],
        [r["r5"] for r in rows],
        [r["r10"] for r in rows],
        [r["mrr"] for r in rows],
        [r["lat"] for r in rows],
    )
    block["retriever"] = name
    block["by_language"] = breakdown(rows, "language")
    block["by_route"] = breakdown(rows, "route")
    block["by_complexity"] = breakdown(rows, "complexity")
    block["by_query_type"] = breakdown(rows, "query_type")
    return block


def main() -> int:
    parser = argparse.ArgumentParser(description="Retrieval benchmark")
    parser.add_argument("--eval", type=Path, default=EVAL_DIR / "retrieval_eval.jsonl")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--report", type=Path, default=REPORTS_DIR / "retrieval_benchmark.json")
    parser.add_argument("--skip-rerank", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")

    path = args.eval
    if not path.exists():
        fallback = CLEAN_DIR / "examples_validation.jsonl"
        if fallback.exists():
            logger.warning("Missing %s; falling back to %s", path, fallback)
            path = fallback
        else:
            logger.error("No eval file. Run scripts/build_retrieval_eval.py")
            return 1

    queries = [r for r in load_jsonl(path) if r.get("query") and r.get("gold_document_ids")]
    if not queries:
        logger.error("Eval file has no gold queries")
        return 1

    warm = warmup_runtime(include_reranker=not args.skip_rerank)
    hybrid = warm["hybrid"]
    dense = hybrid.dense
    bm25 = hybrid.bm25
    reranker = warm.get("reranker")
    rt_cfg = RoutingConfig.from_env()

    def hybrid_rerank(query: str, top_k: int = 10):
        hits, lat = hybrid.search_timed(query, top_k=top_k, language_filter=None)
        t = time.perf_counter()
        ranked = reranker.rerank(query, hits, top_n=rt_cfg.rerank_top_n)
        lat = dict(lat)
        lat["rerank_ms"] = round((time.perf_counter() - t) * 1000, 3)
        lat["total_ms"] = round(float(lat.get("total_ms") or 0) + lat["rerank_ms"], 3)
        return ranked, lat

    report: dict[str, Any] = {
        "dataset": "ai4bharat/MSMARCO-XI",
        "eval_path": str(path),
        "n_queries": len(queries),
        "relevance_definition": (
            "Passage-level gold: document_id in gold_document_ids from is_selected==1. "
            "No chunk-level labels."
        ),
        "collection_points": warm["store"].count(),
        "bm25_docs": len(bm25.payloads),
        "cold_start_ms": warm["cold_start_ms"],
        "warm_probe_ms": warm["warm_request_ms"],
        "warmup_components_ms": warm["components"],
        "dense": _run_system("dense", lambda q, top_k: dense.search_timed(q, top_k=top_k), queries, args.top_k),
        "bm25": _run_system("bm25", lambda q, top_k: bm25.search_timed(q, top_k=top_k), queries, args.top_k),
        "hybrid": _run_system("hybrid", lambda q, top_k: hybrid.search_timed(q, top_k=top_k), queries, args.top_k),
    }
    if not args.skip_rerank:
        report["hybrid_plus_rerank"] = _run_system("hybrid+rerank", hybrid_rerank, queries, args.top_k)
        h, r = report["hybrid"]["mrr"], report["hybrid_plus_rerank"]["mrr"]
        report["rerank_mrr_delta"] = None if h is None or r is None else round(r - h, 4)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    slim = {k: v for k, v in report.items()}
    print(json.dumps(slim, indent=2, ensure_ascii=False)[:15000])
    logger.info("Wrote %s", args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
