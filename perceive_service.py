#!/usr/bin/env python3
"""
nuclear-eye Perceptual Service — /v1/perceive
Unified multimodal perception endpoint.

Accepts: frame (base64 JPEG) + audio_chunk (base64 PCM 16kHz) + text_context
Calls in parallel:
  - FER (face emotion): nuclear-eye face_embedding_service :5555
  - SER (voice emotion): arianne-ser :8105
  - Pose (gesture):     nuclear-eye vision_agent health check (stub)
Returns: PerceptualState JSON

Port: 8091

Optional FER ONNX (local model): install extras from repo root:
  pip install -e ".[perceive-fer]"
Then set FER_MODEL_PATH (and optional FER_CLASSES_PATH) to your ONNX + labels JSON.
"""
import asyncio
import base64
import json
import os
import sys
import time
from typing import Optional

import aiohttp
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="nuclear-eye perceive", version="1.0.0")

# Service URLs (configurable)
FER_URL    = os.getenv("FER_URL",    "http://127.0.0.1:5555")
SER_URL    = os.getenv("SER_URL",    "http://127.0.0.1:8105")
VISION_URL = os.getenv("VISION_URL", "http://127.0.0.1:8090")

# Y6: Fail-closed wrapper guard
# Unset → default localhost (guard on). Explicit empty string → guard off (dev only).
WRAPPER_URL = (
    os.environ["NUCLEAR_WRAPPER_URL"].strip()
    if "NUCLEAR_WRAPPER_URL" in os.environ
    else "http://localhost:9090"
)
WRAPPER_POLL_INTERVAL_SECS = 60

RISK_ALERT_THRESHOLD = float(os.getenv("RISK_ALERT_THRESHOLD", "0.7"))

# Y6: Degraded mode flag — set True when wrapper goes away at runtime.
_perceive_degraded = False


async def check_wrapper_health():
    """Y6: Verify nuclear-wrapper is reachable before serving. Skip if WRAPPER_URL is empty."""
    if not WRAPPER_URL:
        print(
            "[perceive] CRITICAL: NUCLEAR_WRAPPER_URL is empty — wrapper guard DISABLED. "
            "Do not expose this endpoint on untrusted networks; set NUCLEAR_WRAPPER_URL in production.",
            flush=True,
        )
        return
    try:
        r = await httpx.AsyncClient().get(f"{WRAPPER_URL}/health", timeout=5.0)
        if r.status_code != 200:
            raise RuntimeError(f"wrapper returned HTTP {r.status_code}")
    except Exception as e:
        print(
            f"FATAL: nuclear-wrapper unreachable: {e}. Refusing to start unguarded.",
            file=sys.stderr,
        )
        sys.exit(1)


async def wrapper_monitor():
    """Y6: Periodic wrapper health check every 60s. Enters degraded mode if wrapper goes away."""
    global _perceive_degraded
    if not WRAPPER_URL:
        return  # dev mode — no wrapper guard
    while True:
        await asyncio.sleep(WRAPPER_POLL_INTERVAL_SECS)
        try:
            r = await httpx.AsyncClient().get(f"{WRAPPER_URL}/health", timeout=5.0)
            if r.status_code != 200:
                raise RuntimeError(f"wrapper returned HTTP {r.status_code}")
            if _perceive_degraded:
                import logging
                logging.getLogger("nuclear-eye-perceive").info(
                    "nuclear-wrapper: recovered — resuming normal operation"
                )
                _perceive_degraded = False
        except Exception as e:
            if not _perceive_degraded:
                import logging
                logging.getLogger("nuclear-eye-perceive").critical(
                    "CRITICAL: nuclear-wrapper lost at runtime: %s — entering degraded mode, "
                    "new perceive requests rejected", e
                )
            _perceive_degraded = True

EMOTION_AFFECT = {
    "happy":    ( 0.80,  0.60),
    "angry":    (-0.80,  0.80),
    "sad":      (-0.70, -0.50),
    "fear":     (-0.60,  0.70),
    "surprise": ( 0.20,  0.80),
    "disgust":  (-0.70,  0.30),
    "contempt": (-0.50,  0.20),
    "neutral":  ( 0.00,  0.00),
}

class PerceiveRequest(BaseModel):
    source_id: str = "default"
    frame_b64: Optional[str] = None      # base64 JPEG
    audio_b64: Optional[str] = None      # base64 PCM 16kHz mono
    sample_rate: int = 16000
    text_context: Optional[str] = None

class PerceiveResponse(BaseModel):
    timestamp: float
    source_id: str
    face: Optional[dict] = None
    voice: Optional[dict] = None
    gesture: Optional[dict] = None
    risk: Optional[dict] = None
    mood_summary: Optional[str] = None


def compute_risk(face, voice, gesture) -> Optional[dict]:
    """risk = 0.4*face_negative + 0.3*voice_agitated + 0.3*gesture_threat"""
    n = 0
    face_c = voice_c = gesture_c = 0.0

    if face:
        valence = face.get("valence", 0.0)
        conf    = face.get("confidence", 0.5)
        face_c  = ((-valence + 1.0) / 2.0) * conf
        n += 1

    if voice:
        valence = voice.get("valence", 0.0)
        arousal = voice.get("arousal", 0.0)
        conf    = voice.get("confidence", 0.5)
        voice_c = (((arousal + 1.0) / 2.0) * ((-valence + 1.0) / 2.0)) ** 0.5 * conf
        n += 1

    if gesture:
        intent_scores = {
            "attacking": 1.0, "approaching": 0.7, "loitering": 0.5,
            "fleeing": 0.4, "help_needed": 0.3, "normal": 0.0, "unknown": 0.2
        }
        conf      = gesture.get("confidence", 0.5)
        intent    = gesture.get("intent", "unknown")
        gesture_c = intent_scores.get(intent, 0.2) * conf
        n += 1

    if n < 2:
        return None

    score = 0.4 * face_c + 0.3 * voice_c + 0.3 * gesture_c
    return {
        "score": round(score, 4),
        "alert": score > RISK_ALERT_THRESHOLD,
        "components": {
            "face_contribution":    round(face_c * 0.4, 4),
            "voice_contribution":   round(voice_c * 0.3, 4),
            "gesture_contribution": round(gesture_c * 0.3, 4),
        }
    }


def mood_summary(face: Optional[dict]) -> Optional[str]:
    if not face:
        return None
    emotion = face.get("emotion", "neutral")
    valence = face.get("valence", 0.0)
    arousal = face.get("arousal", 0.0)
    v_word = "positive" if valence > 0.3 else "negative" if valence < -0.3 else "neutral"
    a_word = "energized" if arousal > 0.3 else "calm" if arousal < -0.3 else "moderate"
    return f"User appears {emotion} ({v_word}, {a_word})"


async def call_ser(session: aiohttp.ClientSession, audio_b64: str, sample_rate: int) -> Optional[dict]:
    try:
        async with session.post(
            f"{SER_URL}/v1/emotion/voice",
            json={"audio_b64": audio_b64, "sample_rate": sample_rate},
            timeout=aiohttp.ClientTimeout(total=2.0)
        ) as resp:
            if resp.status == 200:
                data = await resp.json()
                emotion = data.get("emotion", "neutral")
                valence, arousal = EMOTION_AFFECT.get(emotion, (0.0, 0.0))
                return {
                    "emotion": emotion,
                    "confidence": data.get("confidence", 0.5),
                    "valence": data.get("valence", valence),
                    "arousal": data.get("arousal", arousal),
                    "mock": data.get("mock", True),
                }
    except Exception:
        pass
    return None


_fer_session: Optional[object] = None  # onnxruntime.InferenceSession
_fer_classes: list = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# Russell circumplex: valence/arousal per FER class
_FER_AFFECT: dict = {
    "angry":   (-0.6,  0.8),
    "disgust": (-0.5,  0.2),
    "fear":    (-0.7,  0.7),
    "happy":   ( 0.8,  0.5),
    "neutral": ( 0.0,  0.0),
    "sad":     (-0.6, -0.4),
    "surprise":( 0.1,  0.8),
}


def _load_fer_model() -> bool:
    """Load EfficientViT FER ONNX model. Returns True on success."""
    global _fer_session, _fer_classes
    if _fer_session is not None:
        return True
    model_path = os.getenv("FER_MODEL_PATH", "/etc/nuclear/models/fer_efficientvit.onnx")
    classes_path = os.getenv("FER_CLASSES_PATH", "/etc/nuclear/models/fer_efficientvit_classes.json")
    if not os.path.exists(model_path):
        return False
    try:
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 2
        opts.intra_op_num_threads = 2
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        _fer_session = ort.InferenceSession(model_path, sess_options=opts, providers=providers)
        if os.path.exists(classes_path):
            with open(classes_path) as f:
                data = json.load(f)
                _fer_classes = data.get("classes", _fer_classes)
        print(f"[perceive] FER ONNX loaded: {model_path}  classes={_fer_classes}", flush=True)
        return True
    except Exception as e:
        print(f"[perceive] FER ONNX load failed: {e}", flush=True)
        return False


async def call_fer(session: aiohttp.ClientSession, frame_b64: str) -> Optional[dict]:
    """Run EfficientViT FER on the given base64 JPEG frame."""
    try:
        import numpy as np
        from PIL import Image
        import io as _io

        if not _load_fer_model():
            return None

        img_bytes = base64.b64decode(frame_b64)
        pil_img = Image.open(_io.BytesIO(img_bytes)).convert("RGB").resize((224, 224))
        arr = np.array(pil_img, dtype=np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std
        arr = arr.transpose(2, 0, 1)[np.newaxis]  # (1, 3, 224, 224)

        logits = _fer_session.run(None, {_fer_session.get_inputs()[0].name: arr})[0][0]
        probs = np.asarray(logits, dtype=np.float64).ravel()
        if probs.size == 0:
            return None
        # softmax (stable)
        exp_l = np.exp(probs - probs.max())
        p = exp_l / exp_l.sum()

        idx = int(p.argmax())
        n_cls = min(len(_fer_classes), p.size)
        # Model may output more logits than labels (or vice versa); never index past either.
        emotion = _fer_classes[idx] if idx < len(_fer_classes) else "neutral"
        confidence = float(p[idx]) if idx < p.size else 0.0
        valence, arousal = _FER_AFFECT.get(emotion, (0.0, 0.0))

        all_probs = {_fer_classes[i]: round(float(p[i]), 4) for i in range(n_cls)}

        return {
            "emotion": emotion,
            "confidence": round(confidence, 4),
            "valence": valence,
            "arousal": arousal,
            "source": "fer_onnx",
            "all_probs": all_probs,
        }
    except Exception as e:
        print(f"[perceive] FER inference error: {e}", flush=True)
        return None


@app.on_event("startup")
async def _startup_wrapper_guard():
    """Y6: Block startup if nuclear-wrapper is unreachable."""
    await check_wrapper_health()
    asyncio.get_event_loop().create_task(wrapper_monitor())


@app.post("/v1/perceive", response_model=PerceiveResponse)
async def perceive(req: PerceiveRequest):
    # Y6: Reject new perceive requests in degraded mode (wrapper gone at runtime).
    if _perceive_degraded:
        raise HTTPException(
            status_code=503,
            detail="nuclear-wrapper unreachable — perceive service in degraded mode (Y6 fail-closed)",
        )
    async with aiohttp.ClientSession() as session:
        has_face = req.frame_b64 is not None
        has_voice = req.audio_b64 is not None

        fer_task  = call_fer(session, req.frame_b64)  if has_face  else asyncio.sleep(0, result=None)
        ser_task  = call_ser(session, req.audio_b64, req.sample_rate) if has_voice else asyncio.sleep(0, result=None)

        face_result, voice_result = await asyncio.gather(fer_task, ser_task)

    risk  = compute_risk(face_result, voice_result, None)
    mood  = mood_summary(face_result)

    return PerceiveResponse(
        timestamp=time.time(),
        source_id=req.source_id,
        face=face_result,
        voice=voice_result,
        gesture=None,
        risk=risk,
        mood_summary=mood,
    )


@app.get("/health")
async def health():
    return {"status": "ok", "service": "nuclear-eye-perceive", "port": 8091}


@app.get("/v1/perceive/labels")
async def labels():
    return {
        "emotions": list(EMOTION_AFFECT.keys()),
        "gesture_intents": ["normal", "loitering", "approaching", "attacking", "fleeing", "help_needed", "unknown"],
    }


if __name__ == "__main__":
    port = int(os.getenv("PERCEIVE_PORT", "8091"))
    uvicorn.run(app, host="127.0.0.1", port=port)
