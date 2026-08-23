"""Lightweight, deterministic query features for FAST / ACCURATE / DEEP routing."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from backend.config import RoutingConfig
from backend.query.language import detect_query_language
from backend.retrieval.bm25 import tokenize

_COMPARE_RE = re.compile(
    r"\b(compare|comparison|versus|vs\.?|differ|difference|differences|better|worse|trade-?off)\b",
    re.I,
)
_EXPLAIN_RE = re.compile(
    r"\b(why|how|explain|describe|because|mechanism|process|formed|works?)\b",
    re.I,
)
_FACTUAL_RE = re.compile(
    r"\b(what|who|when|where|which|capital|define|definition|meaning)\b",
    re.I,
)
_AMBIG_RE = re.compile(
    r"\b(or|either|maybe|possibly|unclear|depends|might|could)\b",
    re.I,
)
_MULTI_RE = re.compile(
    r"(\band then\b|\bfirst\b.+\bsecond\b|\balso\b.+\band\b|\d+\.\s|\band which\b|\band when\b)",
    re.I,
)
_CLAUSE_SPLIT = re.compile(r"[;?]|(?:\s+and\s+)|(?:\s+then\s+)|(?:\s*,\s+)")
_ENTITY_RE = re.compile(r"\b[A-Z][a-zA-Z]{2,}(?:\s+[A-Z][a-zA-Z]{2,})*\b")
_QUOTED_RE = re.compile(r"[\"“”']([^\"“”']+)[\"“”']")


@dataclass
class QueryAnalysis:
    query_type: str
    language: str
    complexity: str
    route: str
    features: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _n_clauses(text: str) -> int:
    parts = [p.strip() for p in _CLAUSE_SPLIT.split(text) if p.strip()]
    return max(1, len(parts)) if text.strip() else 0


def _n_questions(text: str) -> int:
    marks = text.count("?") + text.count("؟") + text.count("।")
    return max(1 if text.strip().endswith(("?", "؟")) else 0, marks)


def _n_entities(text: str) -> int:
    ents = set(_ENTITY_RE.findall(text or ""))
    ents.update(m.strip() for m in _QUOTED_RE.findall(text or "") if m.strip())
    return len(ents)


def classify_query_type(text: str, features: dict[str, Any]) -> str:
    if features["multi_part"] or features["n_questions"] >= 2:
        return "multi_part"
    if features["comparison"]:
        return "comparison"
    if features["explanation"]:
        return "explanation"
    if features["factual"]:
        return "factual"
    if text.strip().endswith(("?", "؟")):
        return "factual"
    return "other"


def analyze_query(query: str, config: RoutingConfig | None = None) -> QueryAnalysis:
    cfg = config or RoutingConfig.from_env()
    q = (query or "").strip()
    tokens = tokenize(q)
    n_tokens = len(tokens)
    n_chars = len(q)
    n_clauses = _n_clauses(q)
    n_questions = _n_questions(q)
    comparison = bool(_COMPARE_RE.search(q))
    explanation = bool(_EXPLAIN_RE.search(q))
    factual = bool(_FACTUAL_RE.search(q))
    ambiguity = bool(_AMBIG_RE.search(q)) and n_tokens >= 6
    multi_part = bool(_MULTI_RE.search(q)) or n_questions >= 2 or n_clauses >= cfg.deep_min_clauses
    language = detect_query_language(q) if q else "und"
    features = {
        "n_chars": n_chars,
        "n_tokens": n_tokens,
        "n_clauses": n_clauses,
        "n_questions": n_questions,
        "n_entities": _n_entities(q),
        "comparison": comparison,
        "explanation": explanation,
        "factual": factual,
        "ambiguity": ambiguity,
        "multi_part": multi_part,
        "language": language,
    }
    query_type = classify_query_type(q, features)
    features["query_type"] = query_type

    deep = bool(q) and (
        n_clauses >= cfg.deep_min_clauses
        or n_questions >= 2
        or multi_part and n_tokens >= 14
        or (comparison and explanation and n_clauses >= 2)
        or (n_tokens >= cfg.deep_min_tokens and (comparison or explanation or ambiguity))
        or (ambiguity and explanation)
    )
    accurate = (not deep) and bool(q) and (
        explanation
        or comparison
        or n_clauses >= 2
        or n_tokens > cfg.fast_max_tokens
        or n_chars > cfg.fast_max_chars
        or ambiguity
        or query_type in {"explanation", "comparison"}
    )
    if deep:
        route = "DEEP"
        complexity = "complex"
    elif accurate:
        route = "ACCURATE"
        complexity = "moderate"
    else:
        route = "FAST"
        complexity = "simple"

    return QueryAnalysis(
        query_type=query_type,
        language=language,
        complexity=complexity,
        route=route,
        features=features,
    )
