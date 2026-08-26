"""OpenAI-compatible chat LLM with JSON answers. Credentials from env only."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol

from backend.config import GenerationConfig
from backend.generation.context import EvidenceSet

SYSTEM_PROMPT = """You are VaaniX, a grounded RAG assistant for a multilingual knowledge base.

Rules:
- Answer ONLY using the provided evidence.
- Do not use outside knowledge.
- Do not invent facts.
- Answer in the same language as the user query.
- If the evidence is insufficient, set status to INSUFFICIENT_CONTEXT and a brief refusal.
- Keep the answer concise.
- Only list sources that appear in the evidence (chunk_id values you were given).

Return a single JSON object:
{"answer": string, "status": "grounded" | "INSUFFICIENT_CONTEXT", "sources": [{"chunk_id": string}]}
"""

STRICT_SYSTEM_PROMPT = (
    SYSTEM_PROMPT
    + "\nStricter pass: every sentence must be entailed by the evidence. "
    "If unsure, return INSUFFICIENT_CONTEXT."
)


class LLMError(Exception):
    pass


class LLMClient(Protocol):
    def complete(self, *, system: str, user: str) -> str: ...


@dataclass
class GenerationResult:
    answer: str
    status: str
    sources: list[dict[str, Any]]
    raw: str | None = None


def build_user_prompt(query: str, query_language: str, evidence: EvidenceSet) -> str:
    blocks = []
    for i, item in enumerate(evidence.items, start=1):
        blocks.append(
            f"[{i}] chunk_id={item.chunk_id} language={item.language} "
            f"document_id={item.document_id} score={item.score:.4f}\n{item.text}"
        )
    body = "\n\n".join(blocks) if blocks else "(no evidence)"
    return (
        f"Query language (detected from script, not a fixed list): {query_language}\n"
        f"User query:\n{query}\n\nEvidence:\n{body}\n"
    )


def parse_generation_json(raw: str) -> GenerationResult:
    if not raw or not str(raw).strip():
        raise LLMError("empty LLM response")
    text = str(raw).strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    blob = fence.group(1) if fence else text
    start, end = blob.find("{"), blob.rfind("}")
    if start < 0 or end < 0:
        raise LLMError("malformed LLM response: no JSON object")
    try:
        data = json.loads(blob[start : end + 1])
    except json.JSONDecodeError as exc:
        raise LLMError("malformed LLM response: invalid JSON") from exc
    if not isinstance(data, dict):
        raise LLMError("malformed LLM response: not an object")
    answer = data.get("answer")
    status = str(data.get("status") or "").strip()
    if not isinstance(answer, str) or not answer.strip():
        raise LLMError("malformed LLM response: missing answer")
    if status.upper() == "INSUFFICIENT_CONTEXT":
        status = "insufficient_context"
    elif status != "grounded":
        status = "grounded" if answer else "insufficient_context"
    sources = data.get("sources") or []
    if not isinstance(sources, list):
        sources = []
    clean_sources = []
    for s in sources:
        if isinstance(s, dict) and s.get("chunk_id"):
            clean_sources.append({"chunk_id": str(s["chunk_id"])})
        elif isinstance(s, str):
            clean_sources.append({"chunk_id": s})
    return GenerationResult(answer=answer.strip(), status=status, sources=clean_sources, raw=text)


class OpenAICompatClient:
    def __init__(self, config: GenerationConfig | None = None) -> None:
        self.config = config or GenerationConfig.from_env()

    def complete(self, *, system: str, user: str) -> str:
        key = self.config.llm_api_key
        if not key:
            raise LLMError("missing VAANIX_LLM_API_KEY / OPENAI_API_KEY")
        url = self.config.llm_base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": self.config.llm_model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key}",
            },
        )
        last_err: Exception | None = None
        for _ in range(max(1, self.config.llm_retries)):
            try:
                with urllib.request.urlopen(req, timeout=self.config.llm_timeout_s) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                choices = data.get("choices") or []
                if not choices:
                    raise LLMError("empty LLM response")
                content = choices[0].get("message", {}).get("content")
                if not content:
                    raise LLMError("empty LLM response")
                return str(content)
            except TimeoutError as exc:
                last_err = LLMError("LLM timeout")
                last_err.__cause__ = exc
            except urllib.error.HTTPError as exc:
                last_err = LLMError(f"LLM API failure: HTTP {exc.code}")
                last_err.__cause__ = exc
            except urllib.error.URLError as exc:
                reason = str(exc.reason) if getattr(exc, "reason", None) else str(exc)
                if "timed out" in reason.lower():
                    last_err = LLMError("LLM timeout")
                else:
                    last_err = LLMError(f"LLM API failure: {reason}")
                last_err.__cause__ = exc
            except LLMError as exc:
                last_err = exc
            except Exception as exc:  # noqa: BLE001
                last_err = LLMError(f"LLM API failure: {exc}")
        assert last_err is not None
        raise last_err


def generate_answer(
    query: str,
    query_language: str,
    evidence: EvidenceSet,
    *,
    client: LLMClient | None = None,
    config: GenerationConfig | None = None,
    strict: bool = False,
) -> GenerationResult:
    cfg = config or GenerationConfig.from_env()
    llm = client or OpenAICompatClient(cfg)
    system = STRICT_SYSTEM_PROMPT if strict else SYSTEM_PROMPT
    user = build_user_prompt(query, query_language, evidence)
    last: LLMError | None = None
    for _ in range(max(1, cfg.llm_retries)):
        try:
            raw = llm.complete(system=system, user=user)
            return parse_generation_json(raw)
        except LLMError as exc:
            last = exc
            if "timeout" in str(exc).lower() or "API failure" in str(exc):
                continue
            if "malformed" in str(exc).lower() or "empty" in str(exc).lower():
                continue
            raise
    raise last or LLMError("LLM failed")
