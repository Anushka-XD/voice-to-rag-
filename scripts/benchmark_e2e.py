#!/usr/bin/env python3
"""
End-to-end latency over gold eval queries (warm vs optional cold).

Does not call Sarvam unless --stt-once is set and SARVAM_API_KEY is present.
LLM is called only if VAANIX_LLM_API_KEY / OPENAI_API_KEY is set; otherwise
generation_ms reflects the mocked or error path of the live pipeline.

  python scripts/benchmark_e2e.py --n 20
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

from backend.config import EVAL_DIR, REPORTS_DIR, GenerationConfig
from backend.eval.dataset import load_jsonl
from backend.eval.metrics import latency_summary
from backend.orchestration.pipeline import VaaniXPipeline
from backend.runtime.warmup import warmup_runtime

logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", type=Path, default=EVAL_DIR / "retrieval_eval.jsonl")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--report", type=Path, default=REPORTS_DIR / "e2e_latency.json")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")

    queries = [r["query"] for r in load_jsonl(args.eval) if r.get("query")][: args.n]
    if not queries:
        logger.error("No eval queries")
        return 1

    warm = warmup_runtime(include_reranker=True)
    pipe = VaaniXPipeline(
        retriever=warm["hybrid"],
        reranker=warm.get("reranker"),
        config=GenerationConfig.from_env(),
    )
    has_llm = bool(pipe.config.llm_api_key)

    rows = []
    for q in queries:
        out = pipe.run_query(q)
        lat = out.get("latency") or {}
        rows.append(
            {
                "query": q[:80],
                "status": out.get("status"),
                "route": out.get("route"),
                "reranked": out.get("reranked"),
                "latency": lat,
            }
        )

    def col(key: str) -> list[float]:
        return [float((r["latency"] or {}).get(key) or 0.0) for r in rows]

    report = {
        "n_queries": len(rows),
        "llm_configured": has_llm,
        "stt": "not measured (text-only bench; Sarvam is used on POST /voice-query)",
        "cold_start_ms": warm["cold_start_ms"],
        "warm_probe_ms": warm["warm_request_ms"],
        "warmup_components_ms": warm["components"],
        "retrieval_ms": latency_summary(col("retrieval_ms")),
        "rerank_ms": latency_summary(col("rerank_ms")),
        "generation_ms": latency_summary(col("generation_ms")),
        "total_ms": latency_summary(col("total_ms")),
        "note": (
            "Warm request latencies after process warmup. Cold-start is model load, not per query. "
            "generation_ms includes real LLM only if an API key is configured; otherwise it is the "
            "pipeline's refusal/error path. STT is not included in this text benchmark."
        ),
        "queries": rows,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    slim = {k: v for k, v in report.items() if k != "queries"}
    print(json.dumps(slim, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
