"""Detect query language from Unicode script — not a product language allow-list."""

from __future__ import annotations

import unicodedata


def detect_query_language(text: str) -> str:
    """
    Return a short tag from the dominant Unicode script in the query.

    Latin → en, Devanagari → hi, Bengali → bn, Tamil → ta, etc. when the
    script maps onto an MSMARCO-XI shard code; otherwise the script name.
    """
    counts: dict[str, int] = {}
    for ch in text or "":
        if not ch.isalpha():
            continue
        try:
            name = unicodedata.name(ch)
        except ValueError:
            continue
        script = name.split(" ")[0]
        counts[script] = counts.get(script, 0) + 1
    if not counts:
        return "und"
    script = max(counts, key=counts.get)  # type: ignore[arg-type]
    # Script → ISO only where Unicode script is unambiguous for this dataset.
    mapping = {
        "LATIN": "en",
        "DEVANAGARI": "hi",
        "BENGALI": "bn",
        "GUJARATI": "gu",
        "GURMUKHI": "pa",
        "KANNADA": "kn",
        "MALAYALAM": "ml",
        "ORIYA": "or",
        "TAMIL": "ta",
        "TELUGU": "te",
        "ARABIC": "ur",
    }
    return mapping.get(script, script.lower())
