"""Sarvam STT unit tests with mocked HTTP."""

from __future__ import annotations

import sys
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.config import AppConfig
from backend.stt.sarvam import STTError, transcribe_audio


def test_stt_missing_key():
    cfg = AppConfig(sarvam_api_key=None)
    with pytest.raises(STTError, match="missing"):
        transcribe_audio(b"abc", config=cfg)


def test_stt_success_mocked():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"transcript": "नमस्ते", "language_code": "hi-IN", "request_id": "r1"},
        )

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)
    cfg = AppConfig(sarvam_api_key="test-key", sarvam_stt_url="https://api.sarvam.ai/speech-to-text")
    out = transcribe_audio(b"audio-bytes", config=cfg, client=client)
    assert out.transcript == "नमस्ते"
    assert out.language_code == "hi-IN"
    assert out.latency_ms >= 0


def test_stt_http_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "unauthorized"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    cfg = AppConfig(sarvam_api_key="bad")
    with pytest.raises(STTError, match="HTTP 401"):
        transcribe_audio(b"x", config=cfg, client=client)
