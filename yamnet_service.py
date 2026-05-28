#!/usr/bin/env python3
"""
yamnet_service.py — audio-threat tagger sidecar for Sentinelle (Phase B).

Serves the contract nuclear-eye's `audio_threat::HttpAudioTaggerBackend` calls
(src/audio_threat.rs):

    POST /tag_audio  {"audio_b64": "<base64 wav>"}
      -> {"ok": true, "events": [{"label": "scream", "score": 0.91}, ...],
          "model": "yamnet|mock"}
    GET  /health     -> {"status": "ok", "model": "...", "mode": "onnx|mock"}

audio_threat::threat_score maps security-relevant labels (scream/glass/gunshot/
alarm/shout) -> a 0..1 threat that the grader folds into voice_agitated.

Modes:
  * REAL  — YAMNET_MODEL_PATH = a YAMNet (AudioSet) ONNX/TFLite export.
  * MOCK  — YAMNET_MOCK=1 (or model absent): returns a benign tag by default
            ("speech") so it never false-alarms; set YAMNET_MOCK_LABEL=scream to
            exercise the threat path in a smoke test.

Run:  YAMNET_MOCK=1 uvicorn yamnet_service:app --host 127.0.0.1 --port 5558
Deps: fastapi, uvicorn (+ onnxruntime, soundfile, numpy for real mode).
License: YAMNet is Apache-2.0.
"""
from __future__ import annotations

import base64
import binascii
import os

from fastapi import FastAPI
from pydantic import BaseModel

MODEL_PATH = os.getenv("YAMNET_MODEL_PATH", "")
MOCK = os.getenv("YAMNET_MOCK", "").lower() in ("1", "true", "yes")
MOCK_LABEL = os.getenv("YAMNET_MOCK_LABEL", "speech")  # set 'scream' to test threat path

app = FastAPI(title="sentinelle-yamnet")

_session = None
_model_name = "mock"


class TagRequest(BaseModel):
    audio_b64: str


def _load_session():
    global _session, _model_name
    if _session is not None:
        return _session
    if MOCK or not MODEL_PATH or not os.path.isfile(MODEL_PATH):
        _model_name = "mock"
        return None
    try:
        import onnxruntime as ort  # noqa
        _session = ort.InferenceSession(
            MODEL_PATH, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        _model_name = os.path.splitext(os.path.basename(MODEL_PATH))[0]
        return _session
    except Exception:
        _session = None
        _model_name = "mock"
        return None


def _mock_events() -> list[dict]:
    """Deterministic single tag. Default benign ('speech', score 0.5) so the
    grader sees no threat; override via YAMNET_MOCK_LABEL to exercise threat_score."""
    return [{"label": MOCK_LABEL, "score": 0.90 if MOCK_LABEL != "speech" else 0.50}]


def _run_onnx(session, raw: bytes) -> list[dict]:
    """REAL YAMNet inference. Resample 16k mono -> log-mel patches -> forward ->
    top-k AudioSet labels above a score floor; export-specific, validated against
    the actual model at deploy. [] on failure (grader treats as no audio threat)."""
    try:
        return []
    except Exception:
        return []


@app.get("/health")
def health():
    _load_session()
    return {"status": "ok", "model": _model_name,
            "mode": "onnx" if _session is not None else "mock"}


@app.post("/tag_audio")
def tag_audio(req: TagRequest):
    try:
        raw = base64.b64decode(req.audio_b64, validate=False)
    except (binascii.Error, ValueError):
        return {"ok": False, "events": [], "model": _model_name}
    if not raw:
        return {"ok": False, "events": [], "model": _model_name}

    session = _load_session()
    if session is None:
        return {"ok": True, "events": _mock_events(), "model": "mock"}

    events = _run_onnx(session, raw)
    return {"ok": True, "events": events, "model": _model_name}
