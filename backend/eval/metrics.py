"""Retrieval metrics. Gold is document_id ∈ is_selected ids — never inferred."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from typing import Any, Sequence


def recall_at_k(ranked_doc_ids: Sequence[str], gold: set[str], k: int) -> float:
    if not gold:
        return float("nan")
    return 1.0 if any(d in gold for d in list(ranked_doc_ids)[:k]) else 0.0


def mrr(ranked_doc_ids: Sequence[str], gold: set[str]) -> float:
    if not gold:
        return float("nan")
    for i, d in enumerate(ranked_doc_ids, start=1):
        if d in gold:
            return 1.0 / i
    return 0.0


def mean_finite(vals: Sequence[float]) -> float | None:
    xs = [v for v in vals if v == v]
    if not xs:
        return None
    return round(float(statistics.mean(xs)), 4)


def percentile(vals: Sequence[float], p: float) -> float | None:
    xs = sorted(v for v in vals if v == v)
    if not xs:
        return None
    if p >= 100:
        return round(float(xs[-1]), 3)
    if p <= 0:
        return round(float(xs[0]), 3)
    k = (len(xs) - 1) * (p / 100.0)
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return round(float(xs[lo]), 3)
    frac = k - lo
    return round(float(xs[lo] * (1 - frac) + xs[hi] * frac), 3)


def latency_summary(vals: Sequence[float]) -> dict[str, float | None]:
    xs = [float(v) for v in vals if v == v]
    if not xs:
        return {"mean": None, "p50": None, "p70": None, "p100": None}
    return {
        "mean": round(float(statistics.mean(xs)), 3),
        "p50": percentile(xs, 50),
        "p70": percentile(xs, 70),
        "p100": percentile(xs, 100),
    }


def metric_block(r1, r5, r10, mrrs, lats) -> dict[str, Any]:
    return {
        "n": len([x for x in r10 if x == x]),
        "recall_at_1": mean_finite(r1),
        "recall_at_5": mean_finite(r5),
        "recall_at_10": mean_finite(r10),
        "mrr": mean_finite(mrrs),
        "latency_ms": latency_summary(lats),
    }


def breakdown(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(key) or "unknown")].append(row)
    out: dict[str, Any] = {}
    for name, items in sorted(groups.items()):
        out[name] = metric_block(
            [i["r1"] for i in items],
            [i["r5"] for i in items],
            [i["r10"] for i in items],
            [i["mrr"] for i in items],
            [i["lat"] for i in items],
        )
    return out
