"""Offline unit tests for MSMARCO-XI cleaner + loader transform (no Hub required)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.config import IngestConfig
from backend.ingestion.cleaner import (
    align_passage_lists,
    clean_answer,
    clean_eng_query,
    clean_query,
    passage_dedupe_key,
)
from backend.ingestion.loader import MSMARCOXILoader, make_document_id, transform_raw_example
from backend.ingestion.schemas import CleanExample


# Minimal raw row matching the verified schema from inspection.
SAMPLE_RAW = {
    "source_lang": "eng_Latn",
    "target_lang": "hin_Deva",
    "meta": {
        "frequency_penalty": 0,
        "max_tokens": 4096,
        "model_name": "ckpt-3epochs-sft-then-400k-kd",
        "presence_penalty": 0,
        "temperature": 0,
        "top_p": 1,
    },
    "Answer": "  निगम एक कंपनी है।  ",
    "query_id": 1102432,
    "query_type": "DESCRIPTION",
    "passages": {
        "English_passages": [
            "A corporation is a company recognized in law.",
            "Unrelated filler passage about weather.",
        ],
        "Translated_passages": [
            "निगम कानून में मान्यता प्राप्त कंपनी है।",
            "मौसम के बारे में असंबंधित अंश।",
        ],
        "is_selected": [1, 0],
    },
    "Eng_Query": ". what is a corporation?",
    "Eng_Answer": "A corporation is a company recognized as such in law.",
    "query": "  कॉर्पोरेशन क्या है?  ",
}


def test_clean_eng_query_strips_leading_punctuation():
    assert clean_eng_query(". what is a corporation?") == "what is a corporation?"
    assert clean_eng_query("...why did rachel carson write") == "why did rachel carson write"


def test_clean_query_strips_whitespace_keeps_indic():
    assert clean_query("  कॉर्पोरेशन क्या है?  ") == "कॉर्पोरेशन क्या है?"


def test_clean_answer_empty_becomes_none():
    assert clean_answer("") is None
    assert clean_answer("   ") is None
    assert clean_answer(None) is None
    assert clean_answer("valid") == "valid"


def test_align_passage_lists_preserves_selection():
    aligned = align_passage_lists(
        ["a", "b"],
        ["अ", "ब"],
        [1, 0],
    )
    assert len(aligned) == 2
    assert aligned[0] == (0, "a", "अ", True)
    assert aligned[1] == (1, "b", "ब", False)


def test_transform_raw_example_schema_and_relationships():
    example = transform_raw_example(SAMPLE_RAW, shard_lang="hi", split="validation")
    assert example is not None
    assert isinstance(example, CleanExample)
    assert example.query_id == 1102432
    assert example.shard_lang == "hi"
    assert example.source_lang == "eng_Latn"
    assert example.target_lang == "hin_Deva"
    assert example.query == "कॉर्पोरेशन क्या है?"
    assert example.eng_query == "what is a corporation?"
    assert example.answer is not None
    assert example.eng_answer is not None
    assert example.query_type == "DESCRIPTION"
    assert example.meta.model_name == "ckpt-3epochs-sft-then-400k-kd"
    # 2 indices × (en + hi) = 4 passage units
    assert len(example.passages) == 4
    assert make_document_id(1102432, 0, "en") in {
        p.document_id for p in example.passages
    }
    gold = [p for p in example.passages if p.is_selected]
    assert len(gold) == 2  # en + hi of index 0
    assert set(example.gold_document_ids) == {p.document_id for p in gold}


def test_transform_skips_empty_queries():
    raw = dict(SAMPLE_RAW)
    raw["query"] = "   "
    raw["Eng_Query"] = "..."
    # Eng_Query cleans to empty after punct strip
    out = transform_raw_example(raw, shard_lang="hi", split="validation")
    assert out is None


def test_transform_null_answers_safe():
    raw = dict(SAMPLE_RAW)
    raw["Answer"] = ""
    raw["Eng_Answer"] = None
    out = transform_raw_example(raw, shard_lang="hi", split="validation")
    assert out is not None
    assert out.answer is None
    assert out.eng_answer is None


def test_english_only_passages():
    out = transform_raw_example(
        SAMPLE_RAW,
        shard_lang="hi",
        split="validation",
        include_english=True,
        include_translated=False,
    )
    assert out is not None
    assert all(p.passage_source == "english" for p in out.passages)
    assert all(p.language == "en" for p in out.passages)


def test_dedupe_keys_normalize():
    assert passage_dedupe_key("  Hello World  ") == passage_dedupe_key("hello world")


def test_loader_resolves_languages_from_report():
    """Uses local inspection report — no network."""
    cfg = IngestConfig(split="validation", languages=["hi"], max_examples_per_lang=1)
    loader = MSMARCOXILoader(cfg)
    langs = loader.languages_for_run()
    assert langs == ["hi"]
    available = sorted(loader.shards["validation"].keys())
    # Must match discovered languages from inspection, not an invented list.
    assert "hi" in available
    assert "te" in available  # validation-only Telugu from inspection
    assert "teltrain" not in available


def test_loader_rejects_unknown_language():
    cfg = IngestConfig(split="validation", languages=["xx_fake"], max_examples_per_lang=1)
    loader = MSMARCOXILoader(cfg)
    with pytest.raises(KeyError):
        loader.languages_for_run()


def test_fixture_file_matches_verified_keys():
    fixture = Path(__file__).parent / "fixtures" / "sample_msmarco_xi_row.json"
    data = json.loads(fixture.read_text(encoding="utf-8"))
    expected = {
        "source_lang",
        "target_lang",
        "meta",
        "Answer",
        "query_id",
        "query_type",
        "passages",
        "Eng_Query",
        "Eng_Answer",
        "query",
    }
    assert set(data.keys()) == expected
    assert set(data["passages"].keys()) == {
        "English_passages",
        "Translated_passages",
        "is_selected",
    }
