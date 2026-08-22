#!/usr/bin/env python3
"""
Compare chunking strategies on a development subset of cleaned MSMARCO-XI passages.

Measures real counts/lengths/latency. Does not claim a winner.

  python scripts/evaluate_chunking.py
  python scripts/evaluate_chunking.py --max-passages 80 --langs hi
  python scripts/evaluate_chunking.py --from-jsonl data/clean/passages_validation.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.config import CLEAN_DIR, ChunkConfig, IngestConfig, REPORTS_DIR
from backend.ingestion.chunker import PassageChunker, STRATEGY_NAMES, iter_passages_jsonl
from backend.ingestion.chunking_strategies import split_sentences
from backend.ingestion.loader import MSMARCOXILoader
from backend.ingestion.schemas import ChunkRecord, PassageRecord

logger = logging.getLogger(__name__)


def percentile(vals: list[int], p: float) -> float | None:
    if not vals:
        return None
    s = sorted(vals)
    idx = min(len(s) - 1, max(0, int(round((p / 100.0) * (len(s) - 1)))))
    return float(s[idx])


def consecutive_overlap_ratio(chunks: list[ChunkRecord]) -> float | None:
    """
    Mean Jaccard overlap of consecutive chunks from the same passage (sentence index sets).
    """
    by_doc: dict[str, list[ChunkRecord]] = {}
    for c in chunks:
        by_doc.setdefault(c.document_id, []).append(c)
    scores: list[float] = []
    for group in by_doc.values():
        group.sort(key=lambda x: x.chunk_index)
        for a, b in zip(group, group[1:]):
            sa = set(range(a.start_sentence, a.end_sentence + 1))
            sb = set(range(b.start_sentence, b.end_sentence + 1))
            union = sa | sb
            if not union:
                continue
            scores.append(len(sa & sb) / len(union))
    if not scores:
        return None
    return round(float(statistics.mean(scores)), 4)


def strategy_metrics(
    passages: list[PassageRecord],
    chunks: list[ChunkRecord],
    elapsed_s: float,
) -> dict[str, Any]:
    lengths = [c.char_length for c in chunks]
    sent_counts = [c.sentence_count for c in chunks]
    per_pass = Counter(c.document_id for c in chunks)
    chunks_per = list(per_pass.values()) if per_pass else []
    # Passages that produced zero chunks
    zero = sum(1 for p in passages if p.document_id not in per_pass)

    gold_passages = [p for p in passages if p.is_selected]
    gold_with_chunk = sum(1 for p in gold_passages if p.document_id in per_pass)

    sample_ids = [c.chunk_id for c in chunks[:3]]
    sample_texts = [
        {
            "chunk_id": c.chunk_id,
            "document_id": c.document_id,
            "query_id": c.query_id,
            "is_selected": c.is_selected,
            "start_sentence": c.start_sentence,
            "end_sentence": c.end_sentence,
            "adaptive_reason": c.adaptive_reason,
            "text_preview": c.text[:180],
        }
        for c in chunks[:3]
    ]

    reasons = Counter(c.adaptive_reason for c in chunks if c.adaptive_reason)

    return {
        "n_passages": len(passages),
        "n_chunks": len(chunks),
        "passages_with_zero_chunks": zero,
        "avg_chunks_per_passage": round(len(chunks) / len(passages), 4) if passages else None,
        "median_chunks_per_passage": (
            float(statistics.median(chunks_per)) if chunks_per else None
        ),
        "chunk_chars": {
            "avg": round(statistics.mean(lengths), 2) if lengths else None,
            "median": float(statistics.median(lengths)) if lengths else None,
            "min": min(lengths) if lengths else None,
            "max": max(lengths) if lengths else None,
            "p90": percentile(lengths, 90),
        },
        "chunk_sentences": {
            "avg": round(statistics.mean(sent_counts), 2) if sent_counts else None,
            "median": float(statistics.median(sent_counts)) if sent_counts else None,
            "min": min(sent_counts) if sent_counts else None,
            "max": max(sent_counts) if sent_counts else None,
        },
        "mean_consecutive_jaccard_overlap": consecutive_overlap_ratio(chunks),
        "gold_passages": len(gold_passages),
        "gold_passages_with_at_least_one_chunk": gold_with_chunk,
        "gold_boundary_preserved_pct": (
            round(100.0 * gold_with_chunk / len(gold_passages), 2) if gold_passages else None
        ),
        "adaptive_reason_counts": dict(reasons),
        "processing_seconds": round(elapsed_s, 3),
        "sample_chunk_ids": sample_ids,
        "sample_chunks": sample_texts,
    }


def load_passages(
    *,
    jsonl: Path | None,
    langs: list[str],
    split: str,
    max_passages: int,
) -> list[PassageRecord]:
    if jsonl and jsonl.exists():
        logger.info("Loading passages from %s", jsonl)
        out: list[PassageRecord] = []
        for rec in iter_passages_jsonl(jsonl):
            out.append(rec)
            if len(out) >= max_passages:
                break
        return out

    cfg = IngestConfig(
        split=split,  # type: ignore[arg-type]
        languages=langs,
        max_examples_per_lang=max(1, max_passages // max(1, len(langs) or 1)),
        include_english_passages=True,
        include_translated_passages=True,
    )
    logger.info("Streaming passages via loader langs=%s split=%s", langs, split)
    examples, _ = MSMARCOXILoader(cfg).run()
    passages: list[PassageRecord] = []
    for ex in examples:
        for p in ex.passages:
            passages.append(p)
            if len(passages) >= max_passages:
                return passages
    return passages


def passage_sentence_stats(passages: list[PassageRecord]) -> dict[str, Any]:
    n_sents = [len(split_sentences(p.text)) for p in passages]
    chars = [p.char_length for p in passages]
    return {
        "n": len(passages),
        "languages": dict(Counter(p.language for p in passages)),
        "selected_passages": sum(1 for p in passages if p.is_selected),
        "avg_passage_chars": round(statistics.mean(chars), 2) if chars else None,
        "avg_sentences_per_passage": round(statistics.mean(n_sents), 2) if n_sents else None,
        "single_sentence_passages": sum(1 for x in n_sents if x <= 1),
        "multi_sentence_passages": sum(1 for x in n_sents if x > 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate VaaniX chunking strategies")
    parser.add_argument("--from-jsonl", type=Path, default=CLEAN_DIR / "passages_validation.jsonl")
    parser.add_argument("--langs", nargs="*", default=["hi"])
    parser.add_argument("--split", default="validation", choices=["train", "validation"])
    parser.add_argument("--max-passages", type=int, default=None)
    parser.add_argument(
        "--report",
        type=Path,
        default=REPORTS_DIR / "chunking_comparison.json",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = ChunkConfig()
    max_passages = args.max_passages or cfg.development_sample_size
    jsonl = args.from_jsonl if args.from_jsonl.exists() else None
    passages = load_passages(
        jsonl=jsonl,
        langs=list(args.langs),
        split=args.split,
        max_passages=max_passages,
    )
    if not passages:
        logger.error("No passages loaded.")
        return 1

    chunker = PassageChunker(cfg)
    comparison: dict[str, Any] = {}
    for name in STRATEGY_NAMES:
        t0 = time.perf_counter()
        chunks = list(chunker.chunk_passages(passages, strategy=name))
        elapsed = time.perf_counter() - t0
        comparison[name] = strategy_metrics(passages, chunks, elapsed)
        logger.info(
            "%s | chunks=%s avg_len=%s time=%.3fs",
            name,
            comparison[name]["n_chunks"],
            comparison[name]["chunk_chars"]["avg"],
            elapsed,
        )

    report = {
        "dataset": "ai4bharat/MSMARCO-XI",
        "note": "Measured on a development subset. No strategy is declared winner here.",
        "input": {
            "jsonl": str(jsonl) if jsonl else None,
            "langs": args.langs,
            "split": args.split,
            "max_passages": max_passages,
            "passage_stats": passage_sentence_stats(passages),
        },
        "config": {
            "target_chunk_chars": cfg.target_chunk_chars,
            "min_chunk_chars": cfg.min_chunk_chars,
            "max_chunk_chars": cfg.max_chunk_chars,
            "target_sentences": cfg.target_sentences,
            "min_sentences": cfg.min_sentences,
            "max_sentences": cfg.max_sentences,
            "overlap_sentences": cfg.overlap_sentences,
            "sliding_window_sentences": cfg.sliding_window_sentences,
            "sliding_overlap_sentences": cfg.sliding_overlap_sentences,
            "semantic_similarity_threshold": cfg.semantic_similarity_threshold,
            "semantic_embed_dim": cfg.semantic_embed_dim,
            "short_max_chars": cfg.short_max_chars,
            "short_max_sentences": cfg.short_max_sentences,
            "long_min_chars": cfg.long_min_chars,
            "long_min_sentences": cfg.long_min_sentences,
            "topic_shift_min_sentences": cfg.topic_shift_min_sentences,
            "development_sample_size": cfg.development_sample_size,
            "source_name": cfg.source_name,
        },
        "strategies": comparison,
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    logger.info("Wrote %s", args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
