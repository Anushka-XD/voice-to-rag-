#!/usr/bin/env python3
"""
Ingest MSMARCO-XI into a clean intermediate representation (no embeddings/indexing).

Examples:
  # Dev subset: Hindi validation, 25 examples
  python scripts/ingest.py --langs hi --split validation --max-per-lang 25

  # Multi-lang with passage dedupe
  python scripts/ingest.py --langs hi bn --max-per-lang 50 --dedupe

  # All discovered validation languages (still capped for safety)
  python scripts/ingest.py --split validation --max-per-lang 10
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

from backend.config import CLEAN_DIR, IngestConfig
from backend.ingestion.loader import MSMARCOXILoader
from backend.ingestion.shards import available_languages, discover_shards


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="VaaniX MSMARCO-XI ingestion")
    parser.add_argument(
        "--langs",
        nargs="*",
        default=None,
        help="Language codes to load (default: all discovered for the split)",
    )
    parser.add_argument(
        "--split",
        default="validation",
        choices=["train", "validation"],
    )
    parser.add_argument(
        "--max-per-lang",
        type=int,
        default=25,
        help="Max examples per language (omit with --no-limit for full shard)",
    )
    parser.add_argument(
        "--no-limit",
        action="store_true",
        help="Do not cap examples per language (loads entire shard — large)",
    )
    parser.add_argument(
        "--dedupe",
        action="store_true",
        help="Enable optional passage deduplication",
    )
    parser.add_argument(
        "--dedupe-scope",
        choices=["global", "per_lang"],
        default="global",
    )
    parser.add_argument(
        "--english-only",
        action="store_true",
        help="Index only English_passages",
    )
    parser.add_argument(
        "--translated-only",
        action="store_true",
        help="Index only Translated_passages",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CLEAN_DIR,
    )
    parser.add_argument(
        "--list-langs",
        action="store_true",
        help="Print discovered languages for the split and exit",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    setup_logging(args.verbose)

    shards = discover_shards(prefer_report=True)
    langs_available = available_languages(args.split, shards=shards)
    if args.list_langs:
        print(json.dumps({"split": args.split, "languages": langs_available}, indent=2))
        return 0

    include_en = not args.translated_only
    include_tr = not args.english_only
    if args.english_only and args.translated_only:
        logging.error("Cannot combine --english-only and --translated-only")
        return 2

    max_per = None if args.no_limit else args.max_per_lang
    cfg = IngestConfig(
        split=args.split,
        languages=list(args.langs) if args.langs else [],
        max_examples_per_lang=max_per,
        dedupe_passages=args.dedupe,
        dedupe_scope=args.dedupe_scope,
        include_english_passages=include_en,
        include_translated_passages=include_tr,
        output_dir=args.output_dir,
    )

    logging.info(
        "Starting ingest | split=%s langs=%s max_per_lang=%s dedupe=%s → %s",
        cfg.split,
        cfg.languages or f"(all discovered: {langs_available})",
        cfg.max_examples_per_lang,
        cfg.dedupe_passages,
        cfg.output_dir,
    )

    loader = MSMARCOXILoader(cfg)
    stats = loader.run_to_jsonl()
    print(json.dumps(stats.to_dict(), indent=2, ensure_ascii=False))
    return 0 if not stats.errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
