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
# Path to AudioSet display-name map (index,mid,display_name). Defaults next to model.
CLASS_MAP_PATH = os.getenv("YAMNET_CLASS_MAP", "")
SCORE_FLOOR = float(os.getenv("YAMNET_SCORE_FLOOR", "0.10"))
TOP_K = int(os.getenv("YAMNET_TOP_K", "5"))

app = FastAPI(title="sentinelle-yamnet")

_session = None
_model_name = "mock"
_labels: list[str] | None = None


def _load_labels() -> list[str]:
    """Lazily load the 521 AudioSet display names. Falls back to numeric ids."""
    global _labels
    if _labels is not None:
        return _labels
    import csv
    path = CLASS_MAP_PATH
    if not path and MODEL_PATH:
        path = os.path.join(os.path.dirname(MODEL_PATH), "yamnet_class_map.csv")
    names: list[str] = []
    try:
        with open(path, newline="") as f:
            rows = list(csv.reader(f))
        # header: index,mid,display_name
        for r in rows[1:]:
            names.append(r[2] if len(r) >= 3 else (r[-1] if r else ""))
    except Exception:
        names = []
    _labels = names
    return _labels


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


def _decode_audio_16k_mono(raw: bytes):
    """Decode WAV/FLAC/OGG bytes -> float32 mono @ 16 kHz in [-1, 1]."""
    import io
    import numpy as np
    import soundfile as sf
    data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != 16000:
        import librosa
        data = librosa.resample(data, orig_sr=sr, target_sr=16000)
    return np.ascontiguousarray(data, dtype=np.float32)


def _run_onnx(session, raw: bytes) -> list[dict]:
    """REAL YAMNet inference: WAV -> 16k mono waveform -> ONNX (mel frontend is
    in-graph) -> [frames, 521] scores -> max over frames per class -> top-k
    AudioSet display names above SCORE_FLOOR. The Rust audio_threat::threat_score
    substring-maps these names (Gunshot/Screaming/Glass/Alarm/...). [] on failure."""
    try:
        import numpy as np
        wav = _decode_audio_16k_mono(raw)
        # YAMNet needs >= ~1 frame (0.96 s = 15360 samples); pad short clips.
        if wav.size < 15360:
            wav = np.pad(wav, (0, 15360 - wav.size))
        labels = _load_labels()
        inp = session.get_inputs()[0].name
        out = session.run(None, {inp: wav})[0]  # output_0: [frames, 521]
        scores = np.asarray(out)
        if scores.ndim == 1:
            scores = scores[None, :]
        clip = scores.max(axis=0)  # max over time per class
        order = clip.argsort()[::-1][:TOP_K]
        events: list[dict] = []
        for i in order:
            s = float(clip[int(i)])
            if s < SCORE_FLOOR:
                break
            label = labels[int(i)] if int(i) < len(labels) else str(int(i))
            events.append({"label": label, "score": round(s, 4)})
        return events
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
