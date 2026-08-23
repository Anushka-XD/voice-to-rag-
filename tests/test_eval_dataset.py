"""Eval dataset builder and metrics — no fabricated gold."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.eval.dataset import build_eval_records
from backend.eval.metrics import latency_summary, mean_finite, mrr, recall_at_k


def test_eval_dataset_uses_only_is_selected_gold():
    examples = [
        {
            "query_id": 1,
            "shard_lang": "hi",
            "query": "कॉर्पोरेशन क्या है?",
            "eng_query": "what is a corporation?",
            "query_type": "DESCRIPTION",
            "gold_document_ids": ["1:0:en", "1:0:hi"],
        },
        {
            "query_id": 2,
            "shard_lang": "hi",
            "query": "मौसम कैसा है",
            "eng_query": "how is the weather",
            "query_type": "DESCRIPTION",
            "gold_document_ids": [],
        },
    ]
    recs, stats = build_eval_records(examples, include_english=True, indexed_document_ids={"1:0:en", "1:0:hi"})
    assert stats["examples_without_gold"] == 1
    assert len(recs) == 2  # indic + english from row 1 only
    for r in recs:
        assert r["gold_document_ids"] == ["1:0:en", "1:0:hi"]
        assert r["expected_route"] in {"FAST", "ACCURATE", "DEEP"}
        assert r["query_type"] == "DESCRIPTION"
    langs = {r["query_language"] for r in recs}
    assert "hi" in langs and "en" in langs


def test_eval_drops_gold_missing_from_index():
    examples = [
        {
            "query_id": 9,
            "shard_lang": "en",
            "query": "what is a corporation?",
            "eng_query": "what is a corporation?",
            "query_type": "ENTITY",
            "gold_document_ids": ["missing:0:en"],
        }
    ]
    recs, stats = build_eval_records(examples, indexed_document_ids={"other:1:en"})
    assert recs == []
    assert stats["dropped_no_indexed_gold"] == 1


def test_metrics_and_latency_summary():
    gold = {"a"}
    assert recall_at_k(["a", "b"], gold, 1) == 1.0
    assert recall_at_k(["b", "a"], gold, 1) == 0.0
    assert recall_at_k(["b", "a"], gold, 5) == 1.0
    assert mrr(["b", "a"], gold) == 0.5
    assert mean_finite([1.0, float("nan")]) == 1.0
    lat = latency_summary([10.0, 20.0, 30.0, 40.0])
    assert lat["p50"] is not None and lat["p70"] is not None and lat["p100"] == 40.0
    json.dumps(lat)
