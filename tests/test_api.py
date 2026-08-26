"""API tests — pipeline and STT mocked; no live keys required."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["VAANIX_SKIP_WARMUP"] = "true"

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient

from backend.api.app import app, set_pipeline
from backend.config import AppConfig, GenerationConfig
from backend.orchestration.pipeline import VaaniXPipeline
from tests.test_pipeline import CORP, FakeLLM, FakeReranker, FakeRetriever, _hit


def _client():
    llm = FakeLLM(
        {
            "answer": "A corporation is a company recognized as a single entity in law.",
            "status": "grounded",
            "sources": [{"chunk_id": "1102432:0:en:adaptive:0"}],
        }
    )
    pipe = VaaniXPipeline(
        retriever=FakeRetriever([_hit(CORP)]),
        llm=llm,
        reranker=FakeReranker(),
        config=GenerationConfig(min_rrf_score=0.01, min_query_overlap=0.04),
    )
    set_pipeline(pipe)
    return TestClient(app)


def test_health():
    with _client() as c:
        r = c.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


def test_query_success():
    with _client() as c:
        r = c.post("/query", json={"query": "what is a corporation?"})
        assert r.status_code == 200
        body = r.json()
        assert body["status"] in {"grounded", "insufficient_context"}
        assert "latency" in body
        assert "sources" in body


def test_query_empty():
    with _client() as c:
        r = c.post("/query", json={"query": "  "})
        assert r.status_code == 400


def test_voice_query_stt_mocked(monkeypatch):
    from backend.stt.sarvam import STTResult
    import backend.api.app as api_mod

    def fake_stt(audio, **kwargs):
        return STTResult(transcript="what is a corporation?", language_code="en-IN", latency_ms=12.5)

    monkeypatch.setattr(api_mod, "transcribe_audio", fake_stt)
    with _client() as c:
        r = c.post("/voice-query", files={"audio": ("a.wav", b"RIFFXXXX", "audio/wav")})
        assert r.status_code == 200
        body = r.json()
        assert body["transcript"] == "what is a corporation?"
        assert body["latency"]["stt_ms"] == 12.5


def test_voice_query_stt_failure(monkeypatch):
    from backend.stt.sarvam import STTError
    import backend.api.app as api_mod

    def boom(*a, **k):
        raise STTError("missing SARVAM_API_KEY / VAANIX_SARVAM_API_KEY")

    monkeypatch.setattr(api_mod, "transcribe_audio", boom)
    with _client() as c:
        r = c.post("/voice-query", files={"audio": ("a.wav", b"xx", "audio/wav")})
        assert r.status_code == 502
        assert "SARVAM" in r.json()["detail"] or "missing" in r.json()["detail"].lower()
