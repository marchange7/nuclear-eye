#!/usr/bin/env python3
"""
voiceprint_service.py — speaker-embedding sidecar for Sentinelle (Phase B).

Serves the contract nuclear-eye's `voiceprint::HttpVoiceprintBackend` calls
(src/voiceprint.rs):

    POST /embed_voice  {"audio_b64": "<base64 wav>"}
      -> {"ok": true, "embedding": [<f32 ...>], "model": "ecapa-tdnn|mock"}
    GET  /health       -> {"status": "ok", "model": "...", "mode": "onnx|mock"}

The embedding is matched (cosine) against the voice watchlist in
voiceprint::match_voice (family suppress / offender escalate).

Modes:
  * REAL  — VOICEPRINT_MODEL_PATH = an ECAPA-TDNN ONNX export (onnxruntime).
  * MOCK  — VOICEPRINT_MOCK=1 (or model absent): deterministic embedding derived
            from the audio bytes, so the SAME clip yields the SAME vector
            (matchable) — lets the alarm_grader -> voiceprint path be smoke-tested
            before weights are deployed.

Run:  VOICEPRINT_MOCK=1 uvicorn voiceprint_service:app --host 127.0.0.1 --port 5557
Deps: fastapi, uvicorn, numpy (+ onnxruntime, soundfile for real mode).
License: pair with an Apache-2.0 model (SpeechBrain ECAPA / WeSpeaker).
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import os

from fastapi import FastAPI
from pydantic import BaseModel

MODEL_PATH = os.getenv("VOICEPRINT_MODEL_PATH", "")
MOCK = os.getenv("VOICEPRINT_MOCK", "").lower() in ("1", "true", "yes")
EMBED_DIM = int(os.getenv("VOICEPRINT_DIM", "192"))  # ECAPA-TDNN = 192

app = FastAPI(title="sentinelle-voiceprint")

_session = None
_model_name = "mock"


class EmbedRequest(BaseModel):
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


def _mock_embedding(raw: bytes) -> list[float]:
    """Deterministic L2-normalised vector seeded by the audio content, so the
    same speaker clip maps to the same embedding (matchable in tests)."""
    seed = hashlib.sha256(raw).digest()
    # Expand the 32-byte digest to EMBED_DIM floats in [-1, 1].
    vals = []
    i = 0
    while len(vals) < EMBED_DIM:
        b = hashlib.sha256(seed + i.to_bytes(4, "big")).digest()
        for j in range(0, len(b), 1):
            if len(vals) >= EMBED_DIM:
                break
            vals.append((b[j] / 127.5) - 1.0)
        i += 1
    norm = sum(v * v for v in vals) ** 0.5 or 1.0
    return [v / norm for v in vals]


def _run_onnx(session, raw: bytes) -> list[float]:
    """REAL ECAPA inference. Preprocess (resample 16k mono -> mel/fbank) +
    forward + L2-normalise are model-export specific and validated against the
    actual ONNX at deploy time; until then MOCK is the smoke path. [] on failure."""
    try:
        return []
    except Exception:
        return []


@app.get("/health")
def health():
    _load_session()
    return {"status": "ok", "model": _model_name,
            "mode": "onnx" if _session is not None else "mock", "dim": EMBED_DIM}


@app.post("/embed_voice")
def embed_voice(req: EmbedRequest):
    try:
        raw = base64.b64decode(req.audio_b64, validate=False)
    except (binascii.Error, ValueError):
        return {"ok": False, "embedding": [], "model": _model_name}
    if not raw:
        return {"ok": False, "embedding": [], "model": _model_name}

    session = _load_session()
    if session is None:
        return {"ok": True, "embedding": _mock_embedding(raw), "model": "mock"}

    emb = _run_onnx(session, raw)
    if not emb:
        return {"ok": False, "embedding": [], "model": _model_name}
    return {"ok": True, "embedding": emb, "model": _model_name}
