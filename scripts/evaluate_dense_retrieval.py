#!/usr/bin/env python3
"""
Evaluate dense retrieval on MSMARCO-XI gold labels (passages.is_selected).

Relevance is passage-level: a retrieved chunk is relevant iff its document_id
is in the example's gold_document_ids. Chunk-level labels do not exist.

  python scripts/evaluate_dense_retrieval.py
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.config import CLEAN_DIR, EmbeddingConfig, REPORTS_DIR, VectorStoreConfig
from backend.retrieval.dense import DenseRetriever
from backend.retrieval.embeddings import create_embedder
from backend.retrieval.vector_store import QdrantVectorStore

logger = logging.getLogger(__name__)


def recall_at_k(ranked_doc_ids: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return float("nan")
    hit = any(d in gold for d in ranked_doc_ids[:k])
    return 1.0 if hit else 0.0


def mrr(ranked_doc_ids: list[str], gold: set[str]) -> float:
    if not gold:
        return float("nan")
    for i, d in enumerate(ranked_doc_ids, start=1):
        if d in gold:
            return 1.0 / i
    return 0.0


def mean_finite(vals: list[float]) -> float | None:
    xs = [v for v in vals if v == v]  # drop NaN
    if not xs:
        return None
    return round(float(statistics.mean(xs)), 4)


def evaluate_queries(
    retriever: DenseRetriever,
    queries: list[dict[str, Any]],
    top_k: int,
    language_filter: str | None,
) -> dict[str, Any]:
    r1: list[float] = []
    r5: list[float] = []
    r10: list[float] = []
    mrrs: list[float] = []
    embed_ms: list[float] = []
    search_ms: list[float] = []
    total_ms: list[float] = []
    skipped_no_gold = 0
    n_used = 0

    for item in queries:
        gold: set[str] = set(item["gold_document_ids"])
        if language_filter:
            gold = {g for g in gold if g.endswith(f":{language_filter}")}
        if not gold:
            skipped_no_gold += 1
            continue
        hits, lat = retriever.search_timed(item["query"], top_k=top_k, language_filter=None)
        ranked = [h.document_id for h in hits]
        r1.append(recall_at_k(ranked, gold, 1))
        r5.append(recall_at_k(ranked, gold, 5))
        r10.append(recall_at_k(ranked, gold, 10))
        mrrs.append(mrr(ranked, gold))
        embed_ms.append(lat["query_embed_ms"])
        search_ms.append(lat["search_ms"])
        total_ms.append(lat["total_ms"])
        n_used += 1

    return {
        "n_queries_considered": len(queries),
        "n_queries_with_gold": n_used,
        "skipped_no_gold": skipped_no_gold,
        "recall_at_1": mean_finite(r1),
        "recall_at_5": mean_finite(r5),
        "recall_at_10": mean_finite(r10),
        "mrr": mean_finite(mrrs),
        "latency_ms": {
            "query_embed_p50": round(float(statistics.median(embed_ms)), 3) if embed_ms else None,
            "query_embed_mean": round(float(statistics.mean(embed_ms)), 3) if embed_ms else None,
            "search_p50": round(float(statistics.median(search_ms)), 3) if search_ms else None,
            "search_mean": round(float(statistics.mean(search_ms)), 3) if search_ms else None,
            "total_p50": round(float(statistics.median(total_ms)), 3) if total_ms else None,
            "total_mean": round(float(statistics.mean(total_ms)), 3) if total_ms else None,
        },
    }


def load_eval_queries(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            gold = row.get("gold_document_ids") or []
            out.append(
                {
                    "query_id": row.get("query_id"),
                    "query": row.get("query") or "",
                    "eng_query": row.get("eng_query") or "",
                    "gold_document_ids": list(gold),
                    "shard_lang": row.get("shard_lang"),
                }
            )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate VaaniX dense retrieval")
    parser.add_argument("--examples", type=Path, default=CLEAN_DIR / "examples_validation.jsonl")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--report", type=Path, default=REPORTS_DIR / "dense_retrieval_report.json")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.examples.exists():
        logger.error("Missing %s", args.examples)
        return 1

    emb_cfg = EmbeddingConfig.from_env()
    vs_cfg = VectorStoreConfig.from_env()
    embedder = create_embedder(emb_cfg)
    store = QdrantVectorStore(vs_cfg)
    if not store.collection_exists() or store.count() == 0:
        logger.error("Empty Qdrant collection. Run scripts/build_vector_index.py first.")
        return 1

    retriever = DenseRetriever(embedder=embedder, store=store, default_top_k=args.top_k)
    rows = load_eval_queries(args.examples)

    indic = [{"query": r["query"], "gold_document_ids": r["gold_document_ids"]} for r in rows if r["query"]]
    english = [{"query": r["eng_query"], "gold_document_ids": r["gold_document_ids"]} for r in rows if r["eng_query"]]
    # Cross-lingual: Indic query, gold restricted to English passages (*:en)
    xling = [
        {"query": r["query"], "gold_document_ids": [g for g in r["gold_document_ids"] if g.endswith(":en")]}
        for r in rows
        if r["query"]
    ]
    en_to_hi = [
        {"query": r["eng_query"], "gold_document_ids": [g for g in r["gold_document_ids"] if g.endswith(":hi")]}
        for r in rows
        if r["eng_query"]
    ]

    report = {
        "dataset": "ai4bharat/MSMARCO-XI",
        "relevance_definition": (
            "A chunk is relevant if payload.document_id is in gold_document_ids, "
            "which come from passages.is_selected==1 on the same query row. "
            "There is no independent chunk-level relevance label."
        ),
        "collection": store.stats(),
        "embedding_model": embedder.model_name,
        "embedding_dimension": embedder.dimension,
        "settings": {"top_k": args.top_k, "language_filter_default": None},
        "indic_query_multilingual_index": evaluate_queries(retriever, indic, args.top_k, None),
        "english_query_multilingual_index": evaluate_queries(retriever, english, args.top_k, None),
        "cross_lingual_indic_query_english_gold": evaluate_queries(retriever, xling, args.top_k, None),
        "cross_lingual_english_query_hindi_gold": evaluate_queries(retriever, en_to_hi, args.top_k, None),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
