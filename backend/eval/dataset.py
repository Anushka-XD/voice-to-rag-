"""Build retrieval eval records from cleaned MSMARCO-XI examples (is_selected gold only)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from backend.query.language import detect_query_language
from backend.routing.query_analyzer import analyze_query


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def document_ids_from_passages(path: Path) -> set[str]:
    ids: set[str] = set()
    for row in load_jsonl(path):
        did = row.get("document_id")
        if did:
            ids.add(str(did))
    return ids


def _emit(
    *,
    query: str,
    query_id: Any,
    language: str,
    gold: list[str],
    query_type: str | None,
    source: str,
) -> dict[str, Any] | None:
    q = (query or "").strip()
    gold_ids = [g for g in gold if g]
    if not q or not gold_ids:
        return None
    analysis = analyze_query(q)
    return {
        "query_id": query_id,
        "query": q,
        "query_language": language or detect_query_language(q),
        "gold_document_ids": gold_ids,
        "query_type": query_type,
        "expected_route": analysis.route,
        "complexity": analysis.complexity,
        "query_source": source,
        "n_tokens": analysis.features.get("n_tokens"),
        "n_chars": analysis.features.get("n_chars"),
    }


def build_eval_records(
    examples: Iterable[dict[str, Any]],
    *,
    include_english: bool = True,
    indexed_document_ids: set[str] | None = None,
    max_queries: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    One eval item per genuine query string with is_selected gold.

    English queries reuse the same gold_document_ids (dataset labels, not invented).
    Gold ids missing from the index are dropped; the query is kept only if ≥1 gold remains.
    """
    stats = {
        "examples_seen": 0,
        "examples_without_gold": 0,
        "indic_emitted": 0,
        "english_emitted": 0,
        "dropped_no_indexed_gold": 0,
        "deep_in_dataset": 0,
        "note": None,
    }
    out: list[dict[str, Any]] = []
    for row in examples:
        stats["examples_seen"] += 1
        gold = list(row.get("gold_document_ids") or [])
        if indexed_document_ids is not None:
            gold = [g for g in gold if g in indexed_document_ids]
        if not gold:
            if row.get("gold_document_ids"):
                stats["dropped_no_indexed_gold"] += 1
            else:
                stats["examples_without_gold"] += 1
            continue
        qtype = row.get("query_type")
        qid = row.get("query_id")
        native = _emit(
            query=row.get("query") or "",
            query_id=qid,
            language=row.get("shard_lang") or detect_query_language(row.get("query") or ""),
            gold=gold,
            query_type=qtype,
            source="indic",
        )
        if native:
            out.append(native)
            stats["indic_emitted"] += 1
            if native["expected_route"] == "DEEP":
                stats["deep_in_dataset"] += 1
        if include_english:
            eng = (row.get("eng_query") or "").strip()
            native_q = (row.get("query") or "").strip()
            if eng and eng != native_q:
                rec = _emit(
                    query=eng,
                    query_id=qid,
                    language="en",
                    gold=gold,
                    query_type=qtype,
                    source="english",
                )
                if rec:
                    out.append(rec)
                    stats["english_emitted"] += 1
                    if rec["expected_route"] == "DEEP":
                        stats["deep_in_dataset"] += 1

    if max_queries is not None and len(out) > max_queries:
        out = _stratified_cap(out, max_queries)

    if stats["deep_in_dataset"] == 0:
        stats["note"] = (
            "No query in this gold set classified as DEEP. "
            "DEEP examples were not manufactured."
        )
    return out, stats


def _stratified_cap(rows: list[dict[str, Any]], cap: int) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in rows:
        key = (str(r.get("query_language")), str(r.get("expected_route")))
        buckets.setdefault(key, []).append(r)
    selected: list[dict[str, Any]] = []
    i = 0
    while len(selected) < cap:
        progressed = False
        for key in sorted(buckets):
            bucket = buckets[key]
            if i < len(bucket):
                selected.append(bucket[i])
                progressed = True
                if len(selected) >= cap:
                    break
        if not progressed:
            break
        i += 1
    return selected
