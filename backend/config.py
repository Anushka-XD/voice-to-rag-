"""Central configuration for VaaniX (ingestion-focused for Step 2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REPORTS_DIR = DATA_DIR / "reports"
CLEAN_DIR = DATA_DIR / "clean"
INSPECTION_REPORT = REPORTS_DIR / "msmarco_xi_inspection.json"

DATASET_ID = "ai4bharat/MSMARCO-XI"
HF_PREFIX = f"hf://datasets/{DATASET_ID}"


@dataclass
class IngestConfig:
    """Configurable MSMARCO-XI ingestion (dev subset → full corpus)."""

    split: Literal["train", "validation"] = "validation"
    # Empty → discover all languages available for the split from Hub/report.
    languages: list[str] = field(default_factory=list)
    max_examples_per_lang: int | None = 50
    streaming: bool = True

    include_english_passages: bool = True
    include_translated_passages: bool = True

    # Dedup is optional so we can benchmark with/without it.
    dedupe_passages: bool = False
    # "global" = across all langs in this run; "per_lang" = within shard only.
    dedupe_scope: Literal["global", "per_lang"] = "global"

    # Skip example only when both queries are empty after cleaning.
    skip_empty_queries: bool = True
    # Keep examples with empty Answer / Eng_Answer (null-safe); never invent text.
    drop_empty_answers: bool = False

    output_dir: Path = field(default_factory=lambda: CLEAN_DIR)
    write_jsonl: bool = True
    write_stats: bool = True

    inspection_report: Path = field(default_factory=lambda: INSPECTION_REPORT)
    dataset_id: str = DATASET_ID

    def resolve_output_dir(self) -> Path:
        return Path(self.output_dir)
