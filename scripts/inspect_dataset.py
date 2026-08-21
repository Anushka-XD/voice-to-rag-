#!/usr/bin/env python3
"""
Inspect ai4bharat/MSMARCO-XI from real parquet shards (do not trust the README alone).

Verified Hub layout (Aug 2026):
  - HF configs: only `default` (per-language BuilderConfigs from the old loading
    script are gone after parquet conversion).
  - Files: train/{lang}train.parquet, validation/{lang}val.parquet
  - Languages live in separate shards; discover via Hub file listing + target_lang.

Usage:
  python scripts/inspect_dataset.py
  python scripts/inspect_dataset.py --langs hi bn ta --samples 100
  python scripts/inspect_dataset.py --discover-only
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATASET_ID = "ai4bharat/MSMARCO-XI"
HF_PREFIX = f"hf://datasets/{DATASET_ID}"

# Filename stem → ISO-ish short code used by VaaniX (derived from Hub filenames).
FILENAME_LANG_MAP = {
    "asm": "as",
    "ben": "bn",
    "guj": "gu",
    "hin": "hi",
    "kan": "kn",
    "mal": "ml",
    "mar": "mr",
    "nep": "ne",
    "ori": "or",
    "pan": "pa",
    "san": "sa",
    "tam": "ta",
    "tel": "te",
    "urd": "ur",
}

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?।؟۔؟!])\s+|(?<=\n)\s*")
PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n+")
TRAIN_RE = re.compile(r"^train/([a-z]+)train\.parquet$")
VAL_RE = re.compile(r"^validation/([a-z]+)val\.parquet$")


@dataclass
class FieldStats:
    present: int = 0
    missing_or_empty: int = 0
    types: Counter = field(default_factory=Counter)
    sample_values: list[Any] = field(default_factory=list)


@dataclass
class TextLengthStats:
    count: int = 0
    chars: list[int] = field(default_factory=list)
    words: list[int] = field(default_factory=list)
    sentences: list[int] = field(default_factory=list)
    paragraphs: list[int] = field(default_factory=list)

    def add(self, text: str) -> None:
        t = str(text)
        self.count += 1
        self.chars.append(len(t))
        self.words.append(len(t.split()))
        sents = [s for s in SENTENCE_SPLIT_RE.split(t) if s.strip()]
        paras = [p for p in PARAGRAPH_SPLIT_RE.split(t) if p.strip()]
        self.sentences.append(len(sents) if sents else (1 if t.strip() else 0))
        self.paragraphs.append(len(paras) if paras else (1 if t.strip() else 0))

    def summary(self) -> dict[str, Any]:
        def pct(vals: list[int]) -> dict[str, float]:
            if not vals:
                return {}
            s = sorted(vals)
            n = len(s)
            return {
                "min": float(s[0]),
                "p50": float(s[n // 2]),
                "p90": float(s[min(n - 1, int(n * 0.9))]),
                "p99": float(s[min(n - 1, int(n * 0.99))]),
                "max": float(s[-1]),
                "mean": round(float(statistics.mean(s)), 2),
            }

        return {
            "n": self.count,
            "chars": pct(self.chars),
            "words": pct(self.words),
            "sentences": pct(self.sentences),
            "paragraphs": pct(self.paragraphs),
            "multi_sentence_pct": (
                round(100.0 * sum(1 for x in self.sentences if x > 1) / self.count, 2)
                if self.count
                else 0.0
            ),
            "multi_paragraph_pct": (
                round(100.0 * sum(1 for x in self.paragraphs if x > 1) / self.count, 2)
                if self.count
                else 0.0
            ),
        }


def is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    if isinstance(value, (list, dict, tuple)) and len(value) == 0:
        return True
    return False


def summarize_value(value: Any, depth: int = 0) -> Any:
    if depth > 3:
        return type(value).__name__
    if isinstance(value, str):
        return value[:160] + ("…" if len(value) > 160 else "")
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        if not value:
            return []
        return {
            "list_len": len(value),
            "first": summarize_value(value[0], depth + 1),
            "elem_type": type(value[0]).__name__,
        }
    if isinstance(value, dict):
        return {k: summarize_value(v, depth + 1) for k, v in value.items()}
    return repr(value)[:120]


def flatten_keys(example: dict[str, Any], prefix: str = "") -> list[str]:
    keys: list[str] = []
    for k, v in example.items():
        path = f"{prefix}.{k}" if prefix else k
        keys.append(path)
        if isinstance(v, dict):
            keys.extend(flatten_keys(v, path))
    return keys


def discover_shards() -> dict[str, Any]:
    """List parquet shards and map filename stems → language codes."""
    from huggingface_hub import HfApi

    api = HfApi()
    files = api.list_repo_files(DATASET_ID, repo_type="dataset")
    train: dict[str, str] = {}
    validation: dict[str, str] = {}
    unknown: list[str] = []

    for path in files:
        m = TRAIN_RE.match(path)
        if m:
            stem = m.group(1)
            code = FILENAME_LANG_MAP.get(stem)
            if code:
                train[code] = path
            else:
                unknown.append(path)
            continue
        m = VAL_RE.match(path)
        if m:
            stem = m.group(1)
            code = FILENAME_LANG_MAP.get(stem)
            if code:
                validation[code] = path
            else:
                unknown.append(path)

    all_langs = sorted(set(train) | set(validation))
    return {
        "train_shards": train,
        "validation_shards": validation,
        "languages_from_filenames": all_langs,
        "train_only": sorted(set(train) - set(validation)),
        "validation_only": sorted(set(validation) - set(train)),
        "unknown_parquets": unknown,
        "readme_vs_hub_notes": [
            "README still documents load_dataset(..., 'hi') but Hub only exposes config 'default'.",
            "Prefer data_files=hf://…/{train|validation}/{stem}{train|val}.parquet.",
            "Gujarati file stem is 'guj' (not 'gu'); Odia stem is 'ori' (not 'or').",
            "Telugu appears in validation but may be missing from train (check validation_only).",
        ],
    }


def stream_shard(repo_path: str) -> Any:
    from datasets import load_dataset

    url = f"{HF_PREFIX}/{repo_path}"
    return load_dataset("parquet", data_files=url, split="train", streaming=True)


def iter_samples(ds: Any, n: int) -> Iterator[dict[str, Any]]:
    for i, row in enumerate(ds):
        if i >= n:
            break
        yield row


def inspect_shard(
    lang_code: str,
    split: str,
    repo_path: str,
    n_samples: int,
) -> dict[str, Any]:
    print(f"\n=== {lang_code} | {split} | {repo_path} | n={n_samples} ===")
    ds = stream_shard(repo_path)
    features_repr = str(getattr(ds, "features", None))

    field_stats: dict[str, FieldStats] = defaultdict(FieldStats)
    query_types: Counter = Counter()
    target_langs: Counter = Counter()
    source_langs: Counter = Counter()
    selected_counts: list[int] = []
    passage_counts: list[int] = []
    length_mismatch = 0
    passage_len_en = TextLengthStats()
    passage_len_tr = TextLengthStats()
    query_len = TextLengthStats()
    answer_len = TextLengthStats()
    eng_query_len = TextLengthStats()
    eng_answer_len = TextLengthStats()
    eng_passage_hashes: Counter = Counter()
    tr_passage_hashes: Counter = Counter()
    seen_query_ids: set[int] = set()
    duplicate_query_ids = 0
    query_ids: list[int] = []
    schema_keys: set[str] = set()
    sample_examples: list[dict[str, Any]] = []
    leading_punct_eng_query = 0
    empty_answer = 0
    empty_eng_answer = 0
    errors: list[str] = []
    rows = 0

    for i, example in enumerate(iter_samples(ds, n_samples)):
        rows += 1
        if not isinstance(example, dict):
            errors.append(f"row {i}: not a dict")
            continue

        schema_keys.update(flatten_keys(example))

        for key, value in example.items():
            fs = field_stats[key]
            fs.present += 1
            fs.types[type(value).__name__] += 1
            if is_empty(value):
                fs.missing_or_empty += 1
            elif len(fs.sample_values) < 2:
                fs.sample_values.append(summarize_value(value))

        passages = example.get("passages")
        if isinstance(passages, dict):
            for sub in ("is_selected", "English_passages", "Translated_passages"):
                path = f"passages.{sub}"
                fs = field_stats[path]
                fs.present += 1
                val = passages.get(sub)
                fs.types[type(val).__name__] += 1
                if is_empty(val):
                    fs.missing_or_empty += 1

            is_sel = list(passages.get("is_selected") or [])
            en_ps = list(passages.get("English_passages") or [])
            tr_ps = list(passages.get("Translated_passages") or [])
            selected_counts.append(sum(1 for x in is_sel if int(x) == 1))
            passage_counts.append(len(en_ps))
            if not (len(en_ps) == len(tr_ps) == len(is_sel)):
                length_mismatch += 1

            for p in en_ps:
                if isinstance(p, str) and p.strip():
                    passage_len_en.add(p)
                    eng_passage_hashes[hash(p.strip())] += 1
            for p in tr_ps:
                if isinstance(p, str) and p.strip():
                    passage_len_tr.add(p)
                    tr_passage_hashes[hash(p.strip())] += 1

        qid = example.get("query_id")
        if isinstance(qid, int):
            if qid in seen_query_ids:
                duplicate_query_ids += 1
            seen_query_ids.add(qid)
            query_ids.append(qid)

        if example.get("query_type"):
            query_types[str(example["query_type"])] += 1
        if example.get("target_lang"):
            target_langs[str(example["target_lang"])] += 1
        if example.get("source_lang"):
            source_langs[str(example["source_lang"])] += 1

        eng_q = example.get("Eng_Query")
        if isinstance(eng_q, str) and eng_q and eng_q[0] in ".?!,:;":
            leading_punct_eng_query += 1

        ans = example.get("Answer")
        if is_empty(ans):
            empty_answer += 1
        eng_ans = example.get("Eng_Answer")
        if is_empty(eng_ans):
            empty_eng_answer += 1

        for text, bucket in (
            (example.get("query"), query_len),
            (ans, answer_len),
            (eng_q, eng_query_len),
            (eng_ans, eng_answer_len),
        ):
            if isinstance(text, str) and text.strip():
                bucket.add(text)

        if len(sample_examples) < 2:
            sample_examples.append(
                {
                    "query_id": example.get("query_id"),
                    "query_type": example.get("query_type"),
                    "source_lang": example.get("source_lang"),
                    "target_lang": example.get("target_lang"),
                    "query": summarize_value(example.get("query")),
                    "Answer": summarize_value(example.get("Answer")),
                    "Eng_Query": summarize_value(example.get("Eng_Query")),
                    "Eng_Answer": summarize_value(example.get("Eng_Answer")),
                    "meta": summarize_value(example.get("meta")),
                    "passages": summarize_value(example.get("passages")),
                    "top_level_keys": sorted(example.keys()),
                }
            )

        if (i + 1) % 25 == 0:
            print(f"  … {i + 1}/{n_samples}")

    eng_dupes = sum(c - 1 for c in eng_passage_hashes.values() if c > 1)
    tr_dupes = sum(c - 1 for c in tr_passage_hashes.values() if c > 1)

    return {
        "lang_code": lang_code,
        "split": split,
        "repo_path": repo_path,
        "samples_inspected": rows,
        "features_repr": features_repr,
        "top_level_and_nested_keys": sorted(schema_keys),
        "field_presence": {
            k: {
                "present": v.present,
                "missing_or_empty": v.missing_or_empty,
                "types": dict(v.types),
                "samples": v.sample_values,
            }
            for k, v in sorted(field_stats.items())
        },
        "languages": {
            "source_lang_counts": dict(source_langs),
            "target_lang_counts": dict(target_langs),
        },
        "query_types": dict(query_types.most_common()),
        "query_ids": {
            "unique": len(seen_query_ids),
            "duplicate_in_sample": duplicate_query_ids,
            "min": min(query_ids) if query_ids else None,
            "max": max(query_ids) if query_ids else None,
            "note": (
                "query_id is the MS MARCO query id. There is NO document_id field; "
                "derive passage ids as f'{query_id}:{passage_index}:{lang}'."
            ),
        },
        "data_quality": {
            "passage_list_length_mismatches": length_mismatch,
            "empty_Answer": empty_answer,
            "empty_Eng_Answer": empty_eng_answer,
            "Eng_Query_leading_punctuation": leading_punct_eng_query,
            "Eng_Query_leading_punct_pct": (
                round(100.0 * leading_punct_eng_query / rows, 2) if rows else 0.0
            ),
        },
        "passages": {
            "count_per_example": {
                "min": min(passage_counts) if passage_counts else None,
                "max": max(passage_counts) if passage_counts else None,
                "mean": round(statistics.mean(passage_counts), 2) if passage_counts else None,
                "mode": Counter(passage_counts).most_common(1)[0][0] if passage_counts else None,
            },
            "selected_relevant_per_example": {
                "min": min(selected_counts) if selected_counts else None,
                "max": max(selected_counts) if selected_counts else None,
                "mean": round(statistics.mean(selected_counts), 3) if selected_counts else None,
                "rows_with_ge1_selected": sum(1 for c in selected_counts if c > 0),
                "rows_with_0_selected": sum(1 for c in selected_counts if c == 0),
                "rows_with_gt1_selected": sum(1 for c in selected_counts if c > 1),
            },
            "english_passage_lengths": passage_len_en.summary(),
            "translated_passage_lengths": passage_len_tr.summary(),
            "duplicate_passages_in_sample": {
                "english_extra_copies": eng_dupes,
                "translated_extra_copies": tr_dupes,
                "unique_english": len(eng_passage_hashes),
                "unique_translated": len(tr_passage_hashes),
            },
            "document_id": "ABSENT — synthesize from query_id + passage index + language tag",
        },
        "text_lengths": {
            "query": query_len.summary(),
            "Answer": answer_len.summary(),
            "Eng_Query": eng_query_len.summary(),
            "Eng_Answer": eng_answer_len.summary(),
        },
        "chunking_implications": {
            "base_unit": "Each entry in English_passages / Translated_passages",
            "multi_sentence_english_passages_pct": passage_len_en.summary().get(
                "multi_sentence_pct", 0
            ),
            "multi_paragraph_english_passages_pct": passage_len_en.summary().get(
                "multi_paragraph_pct", 0
            ),
            "recommendation": (
                "Index passage-level docs first. Structure-aware chunking further splits "
                "by sentence/paragraph when multi_sentence_pct is high; semantic and "
                "sliding-window chunkers operate on sentence sequences within a passage."
            ),
        },
        "evaluation_implications": {
            "gold_labels": "passages.is_selected[i] == 1 marks gold-relevant passage i",
            "cross_lingual": (
                "Same row has Eng_Query + English_passages and query + Translated_passages; "
                "supports Indic-query→English-evidence and reverse setups."
            ),
            "answer_fields": "Answer (Indic) and Eng_Answer (English)",
            "caveat": (
                "Each example is a ~10-passage candidate pool, not a global corpus. "
                "For corpus-level RAG, flatten unique passages across queries; for "
                "in-pool ranking metrics, rank within the 10 candidates."
            ),
        },
        "sample_examples": sample_examples,
        "errors_or_anomalies": errors[:20],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect ai4bharat/MSMARCO-XI structure")
    parser.add_argument(
        "--langs",
        nargs="+",
        default=["hi", "bn", "ta"],
        help="Short language codes (hi, bn, ta, …). Default: hi bn ta",
    )
    parser.add_argument(
        "--split",
        default="validation",
        choices=["train", "validation"],
        help="Prefer validation for faster inspection (~450MB/shard vs ~3.7GB)",
    )
    parser.add_argument("--samples", type=int, default=80)
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "data" / "reports" / "msmarco_xi_inspection.json",
    )
    parser.add_argument("--discover-only", action="store_true")
    args = parser.parse_args()

    print(f"Dataset: {DATASET_ID}")
    shards = discover_shards()
    print(f"Languages from filenames: {shards['languages_from_filenames']}")
    print(f"Train shards: {len(shards['train_shards'])}")
    print(f"Validation shards: {len(shards['validation_shards'])}")
    if shards["validation_only"]:
        print(f"Validation-only langs: {shards['validation_only']}")
    if shards["train_only"]:
        print(f"Train-only langs: {shards['train_only']}")

    # Also report HF config names (usually just 'default')
    hf_configs: list[str] = []
    try:
        from datasets import get_dataset_config_names, get_dataset_split_names

        hf_configs = get_dataset_config_names(DATASET_ID)
        print(f"HF configs: {hf_configs}")
        if hf_configs:
            print(f"Splits for {hf_configs[0]}: {get_dataset_split_names(DATASET_ID, hf_configs[0])}")
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] config discovery: {exc}", file=sys.stderr)

    if args.discover_only:
        out = {
            "dataset_id": DATASET_ID,
            "hf_configs": hf_configs,
            "shards": shards,
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(out, indent=2, ensure_ascii=False))
        print(f"Wrote {args.report}")
        return 0

    shard_map = shards["train_shards"] if args.split == "train" else shards["validation_shards"]
    per_lang: dict[str, Any] = {}
    for lang in args.langs:
        path = shard_map.get(lang)
        if not path:
            per_lang[lang] = {
                "error": f"No {args.split} shard for lang={lang}. Available: {sorted(shard_map)}",
                "lang_code": lang,
            }
            continue
        try:
            per_lang[lang] = inspect_shard(lang, args.split, path, args.samples)
        except Exception as exc:  # noqa: BLE001
            per_lang[lang] = {"error": str(exc), "lang_code": lang, "repo_path": path}

    key_sets = [
        set(r.get("top_level_and_nested_keys", []))
        for r in per_lang.values()
        if "top_level_and_nested_keys" in r
    ]
    common_keys = sorted(set.intersection(*key_sets)) if key_sets else []
    all_keys = sorted(set.union(*key_sets)) if key_sets else []

    # Aggregate query types / target langs across inspected langs
    all_query_types: Counter = Counter()
    all_target_langs: Counter = Counter()
    for r in per_lang.values():
        all_query_types.update(r.get("query_types", {}))
        all_target_langs.update(r.get("languages", {}).get("target_lang_counts", {}))

    final = {
        "dataset_id": DATASET_ID,
        "huggingface_url": f"https://huggingface.co/datasets/{DATASET_ID}",
        "inspection_method": (
            "Streamed real parquet shards via hf://datasets/... (not README assumptions)"
        ),
        "hf_configs": hf_configs,
        "shards": shards,
        "supported_languages_from_inspection": shards["languages_from_filenames"],
        "flores_target_langs_observed": dict(all_target_langs),
        "schema_consensus": {
            "common_keys_across_inspected_langs": common_keys,
            "all_keys_seen": all_keys,
            "canonical_top_level_fields": [
                "source_lang",
                "target_lang",
                "meta",
                "query",
                "Answer",
                "query_id",
                "query_type",
                "passages",
                "Eng_Query",
                "Eng_Answer",
            ],
            "passages_subfields": [
                "is_selected",
                "English_passages",
                "Translated_passages",
            ],
            "meta_subfields": [
                "model_name",
                "temperature",
                "max_tokens",
                "top_p",
                "frequency_penalty",
                "presence_penalty",
            ],
            "no_document_id_field": True,
            "answer_field_casing": "Answer (capital A) — not 'answers' or 'answer'",
            "parquet_type_notes": (
                "In parquet, meta numeric fields may be int64 (temperature/top_p stored as 0/1) "
                "even though the old builder script declared float32."
            ),
        },
        "query_types_observed": dict(all_query_types.most_common()),
        "cross_lingual": {
            "query_language_neq_document_language": True,
            "fields": {
                "indic_query": "query",
                "english_query": "Eng_Query",
                "indic_passages": "passages.Translated_passages",
                "english_passages": "passages.English_passages",
                "indic_answer": "Answer",
                "english_answer": "Eng_Answer",
            },
            "design": {
                "unified_index": "Embed en + Indic passages with one multilingual encoder",
                "per_language_bm25": "Separate lexical indexes per script/language",
                "answer_language": "User query language (detected), independent of evidence language",
            },
        },
        "per_language": per_lang,
        "ingestion_recommendations": {
            "load_pattern": (
                "load_dataset('parquet', data_files='hf://datasets/ai4bharat/MSMARCO-XI/"
                "validation/hinval.parquet', split='train', streaming=True)"
            ),
            "index_units": "Each passage string (English and/or translated) as a document",
            "doc_id_scheme": "{query_id}:{passage_idx}:{lang_tag}",
            "languages_to_index": shards["languages_from_filenames"] + ["en"],
            "gold_for_eval": "is_selected == 1 within the example's passage list",
            "cleaning": (
                "Strip leading punctuation from Eng_Query; drop empty Answer/Eng_Answer; "
                "dedupe identical passage text across queries when building a global corpus"
            ),
            "chunking": (
                "Start from passage-level units; apply sentence/semantic/sliding-window "
                "when passage length stats show multi-sentence content"
            ),
            "query_types_for_routing": (
                "DESCRIPTION / ENTITY / NUMERIC / LOCATION / PERSON etc. from query_type, "
                "plus length/complexity heuristics → FAST / ACCURATE / DEEP"
            ),
        },
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(final, indent=2, ensure_ascii=False))
    print(f"\nWrote report → {args.report}")

    print("\n======== VAANIX DATASET INSPECTION SUMMARY ========")
    print(f"Supported langs ({len(shards['languages_from_filenames'])}): "
          f"{', '.join(shards['languages_from_filenames'])}")
    print(f"Common keys: {', '.join(common_keys) if common_keys else '(see errors)'}")
    print(f"Query types: {dict(all_query_types)}")
    for lang, r in per_lang.items():
        if "error" in r:
            print(f"  [{lang}] ERROR: {r['error']}")
            continue
        qt = r.get("query_types", {})
        pc = r.get("passages", {}).get("count_per_example", {})
        sel = r.get("passages", {}).get("selected_relevant_per_example", {})
        dq = r.get("data_quality", {})
        print(
            f"  [{lang}] n={r.get('samples_inspected')} "
            f"target={r.get('languages', {}).get('target_lang_counts')} "
            f"passages/ex≈{pc.get('mean')} selected≥1={sel.get('rows_with_ge1_selected')} "
            f"empty_Answer={dq.get('empty_Answer')} "
            f"Eng_Query_punct%={dq.get('Eng_Query_leading_punct_pct')}"
        )
        print(f"         query_types={qt}")
        chunk = r.get("chunking_implications", {})
        print(
            f"         multi_sentence_passages%="
            f"{chunk.get('multi_sentence_english_passages_pct')} "
            f"multi_paragraph%={chunk.get('multi_paragraph_english_passages_pct')}"
        )
    print("===================================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
