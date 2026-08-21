"""
Discover MSMARCO-XI language shards from Hub or the local inspection report.

Languages are never invented: they come from parquet filenames on the Hub
(or the cached inspection report produced by scripts/inspect_dataset.py).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from backend.config import DATASET_ID, HF_PREFIX, INSPECTION_REPORT

logger = logging.getLogger(__name__)

# Stem → short code mapping derived from Hub filenames (asmtrain → as, etc.).
# This is a filename decoder, not a product language allow-list.
FILENAME_STEM_TO_CODE: dict[str, str] = {
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

TRAIN_RE = re.compile(r"^train/([a-z]+)train\.parquet$")
VAL_RE = re.compile(r"^validation/([a-z]+)val\.parquet$")


@dataclass(frozen=True)
class ShardInfo:
    lang_code: str
    split: Literal["train", "validation"]
    repo_path: str

    @property
    def hf_url(self) -> str:
        return f"{HF_PREFIX}/{self.repo_path}"


def _parse_parquets(files: list[str]) -> dict[str, dict[str, str]]:
    train: dict[str, str] = {}
    validation: dict[str, str] = {}
    for path in files:
        m = TRAIN_RE.match(path)
        if m:
            code = FILENAME_STEM_TO_CODE.get(m.group(1))
            if code:
                train[code] = path
            else:
                logger.warning("Unknown train stem in %s — not mapped", path)
            continue
        m = VAL_RE.match(path)
        if m:
            code = FILENAME_STEM_TO_CODE.get(m.group(1))
            if code:
                validation[code] = path
            else:
                logger.warning("Unknown validation stem in %s — not mapped", path)
    return {"train": train, "validation": validation}


def discover_shards_from_hub(dataset_id: str = DATASET_ID) -> dict[str, dict[str, str]]:
    from huggingface_hub import HfApi

    api = HfApi()
    files = api.list_repo_files(dataset_id, repo_type="dataset")
    parquets = [f for f in files if f.endswith(".parquet")]
    return _parse_parquets(parquets)


def discover_shards_from_report(report_path: Path | None = None) -> dict[str, dict[str, str]] | None:
    path = report_path or INSPECTION_REPORT
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    shards = data.get("shards") or {}
    train = shards.get("train_shards") or {}
    validation = shards.get("validation_shards") or {}
    if not train and not validation:
        return None
    return {"train": dict(train), "validation": dict(validation)}


def discover_shards(
    prefer_report: bool = True,
    report_path: Path | None = None,
    dataset_id: str = DATASET_ID,
) -> dict[str, dict[str, str]]:
    """
    Return {'train': {lang: repo_path}, 'validation': {...}}.

    Prefers the local inspection report (offline, reproducible), falls back to Hub.
    """
    if prefer_report:
        cached = discover_shards_from_report(report_path)
        if cached:
            logger.info(
                "Loaded shard map from inspection report (%s train, %s validation)",
                len(cached["train"]),
                len(cached["validation"]),
            )
            return cached
    logger.info("Discovering shards from Hugging Face Hub…")
    return discover_shards_from_hub(dataset_id)


def available_languages(
    split: Literal["train", "validation"],
    shards: dict[str, dict[str, str]] | None = None,
    **kwargs: Any,
) -> list[str]:
    mapping = shards or discover_shards(**kwargs)
    return sorted(mapping.get(split, {}).keys())


def resolve_shard(
    lang_code: str,
    split: Literal["train", "validation"],
    shards: dict[str, dict[str, str]] | None = None,
    **kwargs: Any,
) -> ShardInfo:
    mapping = shards or discover_shards(**kwargs)
    split_map = mapping.get(split, {})
    if lang_code not in split_map:
        available = sorted(split_map.keys())
        raise KeyError(
            f"No {split!r} shard for language {lang_code!r}. "
            f"Available (discovered): {available}"
        )
    return ShardInfo(lang_code=lang_code, split=split, repo_path=split_map[lang_code])


def flores_to_short(flores: str) -> str | None:
    """
    Best-effort FLORES → short code using the reverse of known inspection tags.
    Returns None if unknown — callers should keep the FLORES string.
    """
    # Prefix before underscore is the ISO-639-3-ish language id used by FLORES.
    if not flores or "_" not in flores:
        return None
    prefix = flores.split("_", 1)[0].lower()
    # Map common FLORES 3-letter codes seen in MSMARCO-XI to our shard codes.
    flores_prefix_to_code = {
        "eng": "en",
        "asm": "as",
        "ben": "bn",
        "guj": "gu",
        "hin": "hi",
        "kan": "kn",
        "mal": "ml",
        "mar": "mr",
        "npi": "ne",
        "nep": "ne",
        "ory": "or",
        "ori": "or",
        "pan": "pa",
        "san": "sa",
        "tam": "ta",
        "tel": "te",
        "urd": "ur",
    }
    return flores_prefix_to_code.get(prefix)
