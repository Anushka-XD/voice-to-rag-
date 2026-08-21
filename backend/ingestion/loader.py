"""
MSMARCO-XI dataset loader.

Streams language-specific parquet shards verified during inspection.
Does not embed or index — emits CleanExample records for the chunking stage.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Iterator

from backend.config import IngestConfig
from backend.ingestion.cleaner import (
    align_passage_lists,
    clean_answer,
    clean_eng_query,
    clean_query,
    passage_dedupe_key,
)
from backend.ingestion.schemas import (
    CleanExample,
    IngestStats,
    PassageRecord,
    TranslationMeta,
)
from backend.ingestion.shards import (
    available_languages,
    discover_shards,
    flores_to_short,
    resolve_shard,
)

logger = logging.getLogger(__name__)


def make_document_id(query_id: int, passage_index: int, lang_tag: str) -> str:
    return f"{query_id}:{passage_index}:{lang_tag}"


def stream_raw_rows(hf_url: str, max_examples: int | None) -> Iterator[dict[str, Any]]:
    """Stream raw parquet rows from a single Hub shard URL."""
    from datasets import load_dataset

    ds = load_dataset("parquet", data_files=hf_url, split="train", streaming=True)
    for i, row in enumerate(ds):
        if max_examples is not None and i >= max_examples:
            break
        if isinstance(row, dict):
            yield row


def transform_raw_example(
    raw: dict[str, Any],
    *,
    shard_lang: str,
    split: str,
    include_english: bool = True,
    include_translated: bool = True,
    skip_empty_queries: bool = True,
) -> CleanExample | None:
    """
    Map one raw MSMARCO-XI row → CleanExample.

    Returns None if the example should be skipped (e.g. empty queries).
    """
    query_id_raw = raw.get("query_id")
    try:
        query_id = int(query_id_raw)
    except (TypeError, ValueError):
        return None

    source_lang = str(raw.get("source_lang") or "")
    target_lang = str(raw.get("target_lang") or "")
    query = clean_query(raw.get("query"))
    eng_query = clean_eng_query(raw.get("Eng_Query"))

    if skip_empty_queries and not query and not eng_query:
        return None

    answer = clean_answer(raw.get("Answer"))
    eng_answer = clean_answer(raw.get("Eng_Answer"))
    query_type = raw.get("query_type")
    if query_type is not None:
        query_type = str(query_type).strip() or None

    meta = TranslationMeta.from_raw(raw.get("meta"))
    passages_obj = raw.get("passages") or {}
    if not isinstance(passages_obj, dict):
        passages_obj = {}

    aligned = align_passage_lists(
        passages_obj.get("English_passages"),
        passages_obj.get("Translated_passages"),
        passages_obj.get("is_selected"),
    )

    indic_short = flores_to_short(target_lang) or shard_lang
    en_short = flores_to_short(source_lang) or "en"

    passages: list[PassageRecord] = []
    for idx, en_text, tr_text, selected in aligned:
        if include_english and en_text:
            passages.append(
                PassageRecord(
                    document_id=make_document_id(query_id, idx, en_short),
                    query_id=query_id,
                    passage_index=idx,
                    text=en_text,
                    language=en_short,
                    language_flores=source_lang or "eng_Latn",
                    is_selected=selected,
                    passage_source="english",
                )
            )
        if include_translated and tr_text:
            passages.append(
                PassageRecord(
                    document_id=make_document_id(query_id, idx, indic_short),
                    query_id=query_id,
                    passage_index=idx,
                    text=tr_text,
                    language=indic_short,
                    language_flores=target_lang,
                    is_selected=selected,
                    passage_source="translated",
                )
            )

    gold_ids = [p.document_id for p in passages if p.is_selected]

    return CleanExample(
        query_id=query_id,
        shard_lang=shard_lang,
        source_lang=source_lang,
        target_lang=target_lang,
        query=query or "",
        eng_query=eng_query or "",
        answer=answer,
        eng_answer=eng_answer,
        query_type=query_type,
        meta=meta,
        passages=passages,
        gold_document_ids=gold_ids,
        split=split,
        raw_passage_count=len(aligned),
    )


class MSMARCOXILoader:
    """Shard-aware streaming loader + optional passage dedupe."""

    def __init__(self, config: IngestConfig | None = None) -> None:
        self.config = config or IngestConfig()
        self.shards = discover_shards(
            prefer_report=True,
            report_path=self.config.inspection_report,
            dataset_id=self.config.dataset_id,
        )

    def languages_for_run(self) -> list[str]:
        available = available_languages(self.config.split, shards=self.shards)
        if not self.config.languages:
            return available
        missing = [l for l in self.config.languages if l not in available]
        if missing:
            raise KeyError(
                f"Requested languages not available for split={self.config.split!r}: "
                f"{missing}. Discovered: {available}"
            )
        return list(self.config.languages)

    def iter_examples(self) -> Iterator[CleanExample]:
        """Yield cleaned examples (dedupe applied if configured)."""
        stats = IngestStats(
            split=self.config.split,
            dedupe_passages=self.config.dedupe_passages,
            max_examples_per_lang=self.config.max_examples_per_lang,
        )
        # iter_examples alone doesn't finalize stats; use run() for full stats.
        yield from self._iter_with_stats(stats)

    def _iter_with_stats(self, stats: IngestStats) -> Iterator[CleanExample]:
        langs = self.languages_for_run()
        stats.languages_requested = list(langs)
        seen_keys: set[str] = set()
        lang_seen: dict[str, set[str]] = {}

        for lang in langs:
            shard = resolve_shard(lang, self.config.split, shards=self.shards)
            logger.info("Streaming %s (%s)", shard.repo_path, shard.hf_url)
            try:
                raw_iter = stream_raw_rows(
                    shard.hf_url, self.config.max_examples_per_lang
                )
            except Exception as exc:  # noqa: BLE001
                msg = f"{lang}: failed to open shard: {exc}"
                logger.error(msg)
                stats.errors.append(msg)
                continue

            stats.languages_loaded.append(lang)
            lang_count = 0

            for raw in raw_iter:
                stats.records_processed += 1
                try:
                    example = transform_raw_example(
                        raw,
                        shard_lang=lang,
                        split=self.config.split,
                        include_english=self.config.include_english_passages,
                        include_translated=self.config.include_translated_passages,
                        skip_empty_queries=self.config.skip_empty_queries,
                    )
                except Exception as exc:  # noqa: BLE001
                    stats.records_skipped += 1
                    stats.errors.append(f"{lang} qid={raw.get('query_id')}: {exc}")
                    continue

                if example is None:
                    stats.records_skipped += 1
                    stats.empty_query_skipped += 1
                    continue

                if self.config.drop_empty_answers and example.answer is None and example.eng_answer is None:
                    stats.records_skipped += 1
                    continue

                if example.answer is None:
                    stats.empty_answer_count += 1
                if example.eng_answer is None:
                    stats.empty_eng_answer_count += 1

                # Optional passage deduplication
                if self.config.dedupe_passages:
                    kept: list[PassageRecord] = []
                    scope_keys = (
                        seen_keys
                        if self.config.dedupe_scope == "global"
                        else lang_seen.setdefault(lang, set())
                    )
                    for p in example.passages:
                        key = passage_dedupe_key(p.text)
                        # Scope key includes language so EN and HI of same meaning stay.
                        scoped = f"{p.language}:{key}"
                        if scoped in scope_keys:
                            stats.passages_deduped += 1
                            continue
                        scope_keys.add(scoped)
                        kept.append(p)
                    example.passages = kept
                    example.gold_document_ids = [
                        p.document_id for p in kept if p.is_selected
                    ]

                if not example.passages:
                    stats.examples_with_no_passages += 1

                stats.records_emitted += 1
                lang_count += 1
                stats.language_distribution[lang] = (
                    stats.language_distribution.get(lang, 0) + 1
                )
                if example.target_lang:
                    stats.flores_target_distribution[example.target_lang] = (
                        stats.flores_target_distribution.get(example.target_lang, 0) + 1
                    )
                if example.query_type:
                    stats.query_type_distribution[example.query_type] = (
                        stats.query_type_distribution.get(example.query_type, 0) + 1
                    )
                stats.total_query_chars += len(example.query) + len(example.eng_query)
                for p in example.passages:
                    stats.passage_units_emitted += 1
                    stats.total_passage_chars += p.char_length

                yield example

            logger.info("Finished lang=%s examples_emitted=%s", lang, lang_count)

    def run(self) -> tuple[list[CleanExample], IngestStats]:
        """
        Execute ingestion for the configured subset.

        Returns in-memory examples (appropriate for small max_examples) plus stats.
        For large runs prefer run_to_jsonl().
        """
        stats = IngestStats(
            split=self.config.split,
            dedupe_passages=self.config.dedupe_passages,
            max_examples_per_lang=self.config.max_examples_per_lang,
        )
        t0 = time.perf_counter()
        examples = list(self._iter_with_stats(stats))
        stats.processing_seconds = time.perf_counter() - t0
        self._log_stats(stats)
        return examples, stats

    def run_to_jsonl(self, output_dir: Path | None = None) -> IngestStats:
        """Stream clean examples to JSONL + stats JSON (chunking-ready IR)."""
        out = Path(output_dir or self.config.resolve_output_dir())
        out.mkdir(parents=True, exist_ok=True)

        examples_path = out / f"examples_{self.config.split}.jsonl"
        passages_path = out / f"passages_{self.config.split}.jsonl"
        stats_path = out / f"ingest_stats_{self.config.split}.json"

        stats = IngestStats(
            split=self.config.split,
            dedupe_passages=self.config.dedupe_passages,
            max_examples_per_lang=self.config.max_examples_per_lang,
        )
        t0 = time.perf_counter()

        with examples_path.open("w", encoding="utf-8") as ef, passages_path.open(
            "w", encoding="utf-8"
        ) as pf:
            for example in self._iter_with_stats(stats):
                ef.write(json.dumps(example.to_dict(), ensure_ascii=False) + "\n")
                for p in example.passages:
                    # Passage IR for chunking: text + metadata, no embeddings.
                    row = p.to_dict()
                    row["shard_lang"] = example.shard_lang
                    row["split"] = example.split
                    row["query_id"] = example.query_id
                    row["query_type"] = example.query_type
                    pf.write(json.dumps(row, ensure_ascii=False) + "\n")

        stats.processing_seconds = time.perf_counter() - t0
        if self.config.write_stats:
            stats_path.write_text(
                json.dumps(stats.to_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        self._log_stats(stats)
        logger.info("Wrote examples → %s", examples_path)
        logger.info("Wrote passages → %s", passages_path)
        if self.config.write_stats:
            logger.info("Wrote stats → %s", stats_path)
        return stats

    @staticmethod
    def _log_stats(stats: IngestStats) -> None:
        logger.info(
            "Ingest complete | processed=%s emitted=%s skipped=%s "
            "empty_queries=%s empty_Answer=%s empty_Eng_Answer=%s "
            "passages=%s deduped=%s avg_query_chars=%s avg_passage_chars=%s "
            "langs=%s time=%.3fs",
            stats.records_processed,
            stats.records_emitted,
            stats.records_skipped,
            stats.empty_query_skipped,
            stats.empty_answer_count,
            stats.empty_eng_answer_count,
            stats.passage_units_emitted,
            stats.passages_deduped,
            stats.average_query_chars(),
            stats.average_passage_chars(),
            dict(stats.language_distribution),
            stats.processing_seconds,
        )


def load_dev_subset(
    languages: list[str] | None = None,
    max_examples_per_lang: int = 20,
    split: str = "validation",
    dedupe_passages: bool = False,
) -> tuple[list[CleanExample], IngestStats]:
    """Convenience helper for local development subsets."""
    cfg = IngestConfig(
        split=split,  # type: ignore[arg-type]
        languages=list(languages or []),
        max_examples_per_lang=max_examples_per_lang,
        dedupe_passages=dedupe_passages,
    )
    return MSMARCOXILoader(cfg).run()
