"""
Text cleaning for verified MSMARCO-XI fields.

Decisions driven by data/reports/msmarco_xi_inspection.json:
- Strip leading/trailing whitespace on all text fields.
- Strip leading punctuation artifacts on Eng_Query (e.g. ". what is a corporation?").
- Empty/null Answer and Eng_Answer → None (do not invent).
- Preserve passage list alignment; drop empty passage strings but keep indices stable
  via explicit passage_index on kept units.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

# Leading junk seen on Eng_Query during inspection (punctuation / separators).
LEADING_PUNCT_RE = re.compile(r"^[\s\.\,\;\:\!\?\-\—\–\*\#\>\)\(]+")
# Collapse internal runs of whitespace (keep single spaces; keep newlines as space).
WHITESPACE_RE = re.compile(r"\s+")


def normalize_whitespace(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()


def strip_leading_punctuation(text: str) -> str:
    """Remove leading punctuation/separators identified in Eng_Query samples."""
    cleaned = LEADING_PUNCT_RE.sub("", text)
    return cleaned.strip()


def clean_text(
    value: Any,
    *,
    strip_leading_punct: bool = False,
    unicode_normalize: bool = True,
) -> str | None:
    """
    Normalize a text field.

    Returns None for missing/empty values after cleaning.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    if unicode_normalize:
        value = unicodedata.normalize("NFC", value)
    value = normalize_whitespace(value)
    if strip_leading_punct:
        value = strip_leading_punctuation(value)
        value = normalize_whitespace(value)
    if not value:
        return None
    return value


def clean_query(value: Any) -> str | None:
    return clean_text(value, strip_leading_punct=False)


def clean_eng_query(value: Any) -> str | None:
    """English queries may start with leftover punctuation from MS MARCO."""
    return clean_text(value, strip_leading_punct=True)


def clean_answer(value: Any) -> str | None:
    """Answers may be empty — return None, never fabricate."""
    return clean_text(value, strip_leading_punct=False)


def clean_passage(value: Any) -> str | None:
    return clean_text(value, strip_leading_punct=False)


def align_passage_lists(
    english: Any,
    translated: Any,
    is_selected: Any,
) -> list[tuple[int, str | None, str | None, bool]]:
    """
    Align the three passage lists by index.

    Returns list of (index, english_text|None, translated_text|None, is_selected).
    Truncates to the minimum length if lists disagree (inspection found 0 mismatches
    in sampled data, but we stay defensive).
    """
    en = list(english or [])
    tr = list(translated or [])
    sel = list(is_selected or [])
    n = min(len(en), len(tr), len(sel)) if (en and tr and sel) else max(len(en), len(tr), len(sel))
    # Prefer zip-shortest when all present; otherwise pad missing sides with None/0.
    if en and tr and sel:
        n = min(len(en), len(tr), len(sel))
    else:
        n = max(len(en), len(tr), len(sel))

    aligned: list[tuple[int, str | None, str | None, bool]] = []
    for i in range(n):
        e = clean_passage(en[i]) if i < len(en) else None
        t = clean_passage(tr[i]) if i < len(tr) else None
        s_raw = sel[i] if i < len(sel) else 0
        try:
            selected = int(s_raw) == 1
        except (TypeError, ValueError):
            selected = bool(s_raw)
        aligned.append((i, e, t, selected))
    return aligned


def passage_dedupe_key(text: str) -> str:
    """Normalized key for optional passage deduplication."""
    return normalize_whitespace(text).casefold()
