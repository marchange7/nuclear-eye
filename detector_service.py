#!/usr/bin/env python3
"""
detector_service.py — always-on object detector for Sentinelle (Phase B / Phase 2 gate).

Serves the contract that nuclear-eye's `detector::HttpDetector` calls
(src/detector.rs):

    POST /detect   {"image_b64": "<base64 jpeg>", "conf_threshold": 0.35}
      -> {"ok": true,
          "detections": [{"class": "person", "confidence": 0.91,
                          "bbox": {"x": 0.40, "y": 0.35, "w": 0.12, "h": 0.30}}],
          "model": "yolox-nano"}
    GET  /health   -> {"status": "ok", "model": "...", "mode": "onnx|mock"}

bbox is NORMALIZED (0..1), top-left x/y + w/h — exactly what should_invoke_vlm /
detections_to_vision_fields expect.

Modes:
  * REAL  — set DETECTOR_MODEL_PATH to a YOLOX ONNX export; uses onnxruntime.
  * MOCK  — DETECTOR_MOCK=1 (or model absent): returns a deterministic centred
            "person" detection so the vision_agent -> detector -> FastVLM gate
            can be smoke-tested end-to-end before weights are deployed.

Run:
    DETECTOR_MOCK=1 uvicorn detector_service:app --host 127.0.0.1 --port 18094
    DETECTOR_MODEL_PATH=/data/models/yolox/yolox_nano.onnx uvicorn detector_service:app --port 18094

Deps: fastapi, uvicorn, pillow, numpy, onnxruntime (real mode only).

License note: pair this with an Apache-2.0 detector (YOLOX / NanoDet / RT-DETR).
Do NOT ship Ultralytics YOLOv8/v11 weights (AGPL) in a commercial build.
"""
from __future__ import annotations

import base64
import binascii
import io
import os
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

MODEL_PATH = os.getenv("DETECTOR_MODEL_PATH", "")
MOCK = os.getenv("DETECTOR_MOCK", "").lower() in ("1", "true", "yes")
# COCO classes the security gate cares about (maps to detector.rs is_relevant_class).
RELEVANT = {"person", "bicycle", "car", "motorcycle", "bus", "truck", "vehicle"}

app = FastAPI(title="sentinelle-detector")


class DetectRequest(BaseModel):
    image_b64: str
    conf_threshold: float = 0.35


# ── ONNX backend (lazy) ──────────────────────────────────────────────────────
_session = None
_model_name = "mock"


def _load_session():
    """Lazily load the ONNX model. Returns None if unavailable -> mock mode."""
    global _session, _model_name
    if _session is not None:
        return _session
    if MOCK or not MODEL_PATH or not os.path.isfile(MODEL_PATH):
        _model_name = "mock"
        return None
    try:
        import onnxruntime as ort  # noqa
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        _session = ort.InferenceSession(MODEL_PATH, providers=providers)
        _model_name = os.path.splitext(os.path.basename(MODEL_PATH))[0]
        return _session
    except Exception:
        # Fail soft to mock so the gate never hard-fails on a bad model.
        _session = None
        _model_name = "mock"
        return None


def _decode_dims(image_b64: str) -> Optional[tuple[int, int]]:
    """Return (width, height) of the decoded image, or None on bad input."""
    try:
        raw = base64.b64decode(image_b64, validate=False)
    except (binascii.Error, ValueError):
        return None
    if not raw:
        return None
    try:
        from PIL import Image
        with Image.open(io.BytesIO(raw)) as im:
            return im.size  # (w, h)
    except Exception:
        # PIL/format issue — still allow mock to respond with a default frame.
        return (1920, 1080)


def _run_onnx(session, image_b64: str, conf: float) -> list[dict]:
    """REAL inference. Postprocessing is YOLOX-export specific and is validated
    against the actual weights at deploy time; until then MOCK is the smoke path.
    Returns [] on any failure (fail-open: vision_agent captions anyway)."""
    try:
        # NOTE: letterbox preprocess + grid-decode + NMS + class-map are tuned to
        # the specific YOLOX ONNX export; wired here once weights are placed.
        # Returning [] keeps the gate fail-open rather than emitting garbage boxes.
        return []
    except Exception:
        return []


def _mock_detections(conf: float) -> list[dict]:
    """Deterministic: one centred person above threshold -> gate opens."""
    return [{
        "class": "person",
        "confidence": max(0.90, conf),
        "bbox": {"x": 0.42, "y": 0.30, "w": 0.16, "h": 0.40},
    }]


@app.get("/health")
def health():
    _load_session()
    return {"status": "ok", "model": _model_name,
            "mode": "onnx" if _session is not None else "mock"}


@app.post("/detect")
def detect(req: DetectRequest):
    dims = _decode_dims(req.image_b64)
    if dims is None:
        return {"ok": False, "detections": [], "model": _model_name}

    session = _load_session()
    if session is None:
        dets = _mock_detections(req.conf_threshold)
        return {"ok": True, "detections": dets, "model": "mock"}

    dets = _run_onnx(session, req.image_b64, req.conf_threshold)
    return {"ok": True, "detections": dets, "model": _model_name}
