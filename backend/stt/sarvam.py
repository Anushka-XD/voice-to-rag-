"""Sarvam Speech-to-Text (official REST: POST https://api.sarvam.ai/speech-to-text)."""

from __future__ import annotations

import time
from dataclasses import dataclass

import httpx

from backend.config import AppConfig


class STTError(Exception):
    pass


@dataclass
class STTResult:
    transcript: str
    language_code: str | None
    latency_ms: float
    request_id: str | None = None


def transcribe_audio(
    audio: bytes,
    *,
    filename: str = "audio.webm",
    content_type: str = "application/octet-stream",
    config: AppConfig | None = None,
    client: httpx.Client | None = None,
) -> STTResult:
    cfg = config or AppConfig.from_env()
    if not audio:
        raise STTError("empty audio")
    if not cfg.sarvam_api_key:
        raise STTError("missing SARVAM_API_KEY / VAANIX_SARVAM_API_KEY")

    files = {"file": (filename or "audio.webm", audio, content_type or "application/octet-stream")}
    data = {
        "model": cfg.sarvam_stt_model,
        "mode": cfg.sarvam_stt_mode,
        "language_code": "unknown",
    }
    headers = {"api-subscription-key": cfg.sarvam_api_key}
    t0 = time.perf_counter()
    own = client is None
    http = client or httpx.Client(timeout=cfg.sarvam_timeout_s)
    try:
        resp = http.post(cfg.sarvam_stt_url, headers=headers, files=files, data=data)
    except httpx.TimeoutException as exc:
        raise STTError("STT timeout") from exc
    except httpx.HTTPError as exc:
        raise STTError(f"STT API failure: {exc}") from exc
    finally:
        if own:
            http.close()
    ms = round((time.perf_counter() - t0) * 1000, 3)
    if resp.status_code >= 400:
        raise STTError(f"STT API failure: HTTP {resp.status_code}")
    try:
        payload = resp.json()
    except Exception as exc:
        raise STTError("STT malformed response") from exc
    transcript = (payload.get("transcript") or "").strip()
    if not transcript:
        raise STTError("STT empty transcript")
    return STTResult(
        transcript=transcript,
        language_code=payload.get("language_code"),
        latency_ms=ms,
        request_id=payload.get("request_id"),
    )
