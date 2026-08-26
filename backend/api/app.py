"""Minimal FastAPI wrapper around run_query() + Sarvam STT."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.config import AppConfig
from backend.orchestration.pipeline import VaaniXPipeline, run_query
from backend.stt.sarvam import STTError, transcribe_audio

logger = logging.getLogger(__name__)

_pipeline: VaaniXPipeline | None = None
_warmup: dict[str, Any] | None = None


def get_pipeline() -> VaaniXPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = VaaniXPipeline()
    return _pipeline


def set_pipeline(pipe: VaaniXPipeline | None) -> None:
    global _pipeline
    _pipeline = pipe


class QueryBody(BaseModel):
    query: str = Field(..., min_length=0)


def _cors_list(raw: str) -> list[str]:
    if raw.strip() == "*":
        return ["*"]
    return [p.strip() for p in raw.split(",") if p.strip()]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _warmup
    cfg = AppConfig.from_env()
    pipe = get_pipeline()
    if cfg.skip_warmup:
        logger.info("Skipping model warmup (VAANIX_SKIP_WARMUP)")
        _warmup = {"skipped": True}
    else:
        try:
            _warmup = pipe.warmup()
            logger.info("Warmup complete cold_start_ms=%s", _warmup.get("cold_start_ms"))
        except Exception:
            logger.exception("Warmup failed; requests may be cold")
            _warmup = {"failed": True}
    yield


app = FastAPI(title="VaaniX", version="0.1.0", lifespan=lifespan)
_app_cfg = AppConfig.from_env()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_list(_app_cfg.cors_origins),
    allow_credentials=_app_cfg.cors_origins.strip() != "*",
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "warmup": _warmup or {"pending": True},
    }


@app.post("/query")
def query(body: QueryBody) -> dict[str, Any]:
    q = (body.query or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="query is required")
    try:
        return run_query(q, pipeline=get_pipeline())
    except Exception as exc:
        logger.exception("query failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/voice-query")
async def voice_query(audio: UploadFile = File(...)) -> dict[str, Any]:
    raw = await audio.read()
    if not raw:
        raise HTTPException(status_code=400, detail="audio is required")
    try:
        stt = transcribe_audio(
            raw,
            filename=audio.filename or "audio.webm",
            content_type=audio.content_type or "application/octet-stream",
        )
    except STTError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    result = run_query(stt.transcript, pipeline=get_pipeline())
    lat = dict(result.get("latency") or {})
    lat["stt_ms"] = stt.latency_ms
    result["latency"] = lat
    result["transcript"] = stt.transcript
    result["stt_language"] = stt.language_code
    result["query"] = stt.transcript
    return result


from backend.config import ROOT as _ROOT
from fastapi.staticfiles import StaticFiles

_ui = _ROOT / "frontend" / "dist"
if _ui.is_dir():
    app.mount("/", StaticFiles(directory=str(_ui), html=True), name="ui")
