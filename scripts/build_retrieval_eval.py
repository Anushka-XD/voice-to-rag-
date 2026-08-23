#!/usr/bin/env python3
"""
Build a retrieval evaluation JSONL from cleaned MSMARCO-XI examples.

Gold labels are only passages with is_selected==1. Nothing is invented.

  python scripts/build_retrieval_eval.py
  python scripts/build_retrieval_eval.py --max-queries 120 --include-english
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.config import CLEAN_DIR, EVAL_DIR
from backend.eval.dataset import build_eval_records, document_ids_from_passages, load_jsonl

logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build MSMARCO-XI retrieval eval set")
    parser.add_argument("--examples", type=Path, default=CLEAN_DIR / "examples_validation.jsonl")
    parser.add_argument("--passages", type=Path, default=CLEAN_DIR / "passages_validation.jsonl")
    parser.add_argument("--out", type=Path, default=EVAL_DIR / "retrieval_eval.jsonl")
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--no-english", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")

    if not args.examples.exists():
        logger.error("Missing %s — ingest first (scripts/ingest.py).", args.examples)
        return 1

    indexed = document_ids_from_passages(args.passages) if args.passages.exists() else None
    examples = load_jsonl(args.examples)
    records, stats = build_eval_records(
        examples,
        include_english=not args.no_english,
        indexed_document_ids=indexed,
        max_queries=args.max_queries,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    langs = Counter(r["query_language"] for r in records)
    routes = Counter(r["expected_route"] for r in records)
    types = Counter(r.get("query_type") or "unknown" for r in records)
    summary = {
        "path": str(args.out),
        "n_queries": len(records),
        "language_distribution": dict(langs),
        "route_distribution": dict(routes),
        "query_type_distribution": dict(types),
        "indexed_document_ids": len(indexed) if indexed is not None else None,
        "build_stats": stats,
    }
    meta = args.out.with_suffix(".meta.json")
    meta.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
