#!/usr/bin/env python3
"""
Build a Qdrant index from cleaned MSMARCO-XI passages.

  python scripts/build_vector_index.py --mode rebuild --sample-size 200
  python scripts/build_vector_index.py --mode upsert --from-jsonl data/clean/passages_validation.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.config import CLEAN_DIR, ChunkConfig, EmbeddingConfig, REPORTS_DIR, VectorStoreConfig
from backend.ingestion.chunker import PassageChunker, iter_passages_jsonl
from backend.ingestion.schemas import ChunkRecord
from backend.retrieval.embeddings import create_embedder
from backend.retrieval.vector_store import QdrantVectorStore

logger = logging.getLogger(__name__)


def batched(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def main() -> int:
    parser = argparse.ArgumentParser(description="Index VaaniX chunks into Qdrant")
    parser.add_argument("--mode", choices=["upsert", "rebuild"], default="rebuild")
    parser.add_argument("--from-jsonl", type=Path, default=CLEAN_DIR / "passages_validation.jsonl")
    parser.add_argument("--strategy", default="adaptive")
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--report", type=Path, default=REPORTS_DIR / "vector_index_report.json")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    emb_cfg = EmbeddingConfig.from_env()
    vs_cfg = VectorStoreConfig.from_env()
    sample = args.sample_size if args.sample_size is not None else vs_cfg.index_sample_size
    if sample == 0:
        sample = None

    if not args.from_jsonl.exists():
        logger.error("Missing %s — run scripts/ingest.py first.", args.from_jsonl)
        return 1

    embedder = create_embedder(emb_cfg)
    dim = embedder.dimension
    store = QdrantVectorStore(vs_cfg)
    store.create_collection(dim, recreate=(args.mode == "rebuild"))

    chunker = PassageChunker(ChunkConfig(strategy=args.strategy))  # type: ignore[arg-type]

    t_all = time.perf_counter()
    passages_n = 0
    skipped_empty = 0
    duplicate_chunks = 0
    seen_ids: set[str] = set()
    chunks_buffer: list[ChunkRecord] = []
    embeddings_generated = 0
    upserted = 0
    embed_seconds = 0.0

    def flush(buf: list[ChunkRecord]) -> None:
        nonlocal embeddings_generated, upserted, embed_seconds
        if not buf:
            return
        t0 = time.perf_counter()
        vecs = embedder.embed_documents([c.text for c in buf])
        embed_seconds += time.perf_counter() - t0
        embeddings_generated += len(buf)
        upserted += store.upsert(buf, vecs)
        buf.clear()

    for passage in iter_passages_jsonl(args.from_jsonl):
        if sample is not None and passages_n >= sample:
            break
        passages_n += 1
        produced = chunker.chunk_passage(passage, strategy=args.strategy)  # type: ignore[arg-type]
        if not produced:
            skipped_empty += 1
            continue
        for ch in produced:
            if ch.chunk_id in seen_ids:
                duplicate_chunks += 1
                continue
            seen_ids.add(ch.chunk_id)
            chunks_buffer.append(ch)
            if len(chunks_buffer) >= emb_cfg.batch_size:
                flush(chunks_buffer)

    flush(chunks_buffer)
    total_s = time.perf_counter() - t_all
    n_chunks = len(seen_ids)
    throughput = (embeddings_generated / embed_seconds) if embed_seconds > 0 else None

    report = {
        "mode": args.mode,
        "source_jsonl": str(args.from_jsonl),
        "chunk_strategy": args.strategy,
        "embedding_model": embedder.model_name,
        "embedding_dimension": dim,
        "embedding_device": embedder.device,
        "batch_size": emb_cfg.batch_size,
        "qdrant_collection": store.collection_name,
        "qdrant_distance": vs_cfg.distance,
        "qdrant_url": vs_cfg.url,
        "qdrant_path": str(vs_cfg.path) if not vs_cfg.url else None,
        "sample_size_cap": sample,
        "passages_processed": passages_n,
        "chunks_indexed": n_chunks,
        "embeddings_generated": embeddings_generated,
        "skipped_empty_passages": skipped_empty,
        "duplicate_chunks_skipped": duplicate_chunks,
        "points_upserted": upserted,
        "collection_points": store.count(),
        "embedding_seconds": round(embed_seconds, 3),
        "total_index_seconds": round(total_s, 3),
        "embeddings_per_second": round(throughput, 2) if throughput else None,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    logger.info("Wrote %s", args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
