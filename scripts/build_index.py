#!/usr/bin/env python3
"""
Repeatable development index.

  python scripts/build_index.py --limit 1500 --strategy adaptive

Does not ingest. Run scripts/ingest.py first if the clean JSONL is too small.
Rebuilds BM25 after Qdrant upsert.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.config import CLEAN_DIR, HybridConfig, REPORTS_DIR, VectorStoreConfig
from backend.retrieval.bm25 import BM25Retriever
from backend.retrieval.vector_store import QdrantVectorStore

logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a larger VaaniX development index")
    parser.add_argument("--from-jsonl", type=Path, default=CLEAN_DIR / "passages_validation.jsonl")
    parser.add_argument("--limit", type=int, default=1500, help="Max passages (0 = all in jsonl)")
    parser.add_argument("--strategy", default="adaptive")
    parser.add_argument("--collection", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--report", type=Path, default=REPORTS_DIR / "vector_index_report.json")
    args = parser.parse_args()

    argv = [
        str(ROOT / "scripts" / "build_vector_index.py"),
        "--mode",
        "rebuild",
        "--from-jsonl",
        str(args.from_jsonl),
        "--strategy",
        args.strategy,
        "--report",
        str(args.report),
    ]
    sample = None if args.limit == 0 else args.limit
    argv.extend(["--sample-size", "0" if sample is None else str(sample)])

    import os

    if args.collection:
        os.environ["VAANIX_QDRANT_COLLECTION"] = args.collection
    if args.batch_size:
        os.environ["VAANIX_EMBEDDING_BATCH_SIZE"] = str(args.batch_size)

    from scripts.build_vector_index import main as build_main

    # Re-parse via env + subprocess-equivalent: call build_vector_index main with sys.argv
    old = sys.argv
    sys.argv = argv
    try:
        rc = build_main()
    finally:
        sys.argv = old
    if rc != 0:
        return rc

    vs = QdrantVectorStore(VectorStoreConfig.from_env())
    hy = HybridConfig.from_env()
    bm25 = BM25Retriever.from_qdrant(vs, config=hy)
    bm25.save(hy.bm25_index_path)
    extra = {
        "bm25_docs": len(bm25.payloads),
        "bm25_index_path": str(hy.bm25_index_path),
        "limit_passages": sample,
        "note": "Indexed only the passages present in the source JSONL, capped by --limit.",
    }
    if args.report.exists():
        report = json.loads(args.report.read_text(encoding="utf-8"))
        report.update(extra)
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(extra, indent=2))
    logger.info("BM25 rebuilt at %s", hy.bm25_index_path)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    raise SystemExit(main())
