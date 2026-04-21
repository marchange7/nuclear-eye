#!/usr/bin/env python3
"""
nuclear-eye Perceptual Service — /v1/perceive
Unified multimodal perception endpoint.

Accepts: frame (base64 JPEG) + audio_chunk (base64 PCM 16kHz) + text_context
Calls in parallel:
  - FER (face emotion): EfficientViT ONNX local inference
  - SER (voice emotion): arianne-ser :8105
  - Gesture intent:      MediaPipe Hands (if installed) or OpenCV contour fallback
Returns: PerceptualState JSON

Port: 8091

Optional FER ONNX (local model): install extras from repo root:
  pip install -e ".[perceive-fer]"
Then set FER_MODEL_PATH (and optional FER_CLASSES_PATH) to your ONNX + labels JSON.

Optional gesture model (MediaPipe Hands):
  pip install -e ".[perceive-gesture]"

P4-7 — gesture_pose passthrough:
  Set GESTURE_POSE_PASSTHROUGH=true to include the raw Scout gesture label alongside
  the mapped security intent in the gesture dict.  The label comes from a Scout
  client that sets `gesture_pose` in the PerceiveRequest (see below).

  When GESTURE_POSE_PASSTHROUGH is false (default) only the mapped `intent` is
  included — backward-compatible with alarm_grader / compute_risk consumers.
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

# P4-7 — Scout gesture_pose → security intent mapping
from gesture_pose_mapping import build_gesture_dict, passthrough_enabled

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
    # P4-7: raw Scout gesture label forwarded from nuclear-scout GestureRecognizer.
    # When set, overrides MediaPipe/contour gesture detection and maps through
    # gesture_pose_mapping.py to the security intent taxonomy consumed by compute_risk.
    # Valid labels: standing|walking|running|crouching|raised_hands|
    #               thumbsUp|openPalm|pointUp|peace|fist|unknown
    gesture_pose: Optional[str] = None   # Scout GestureResult.gesture label
    gesture_pose_confidence: float = 0.5  # Scout GestureResult.confidence (0.0–1.0)

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
        # Keep in sync with /v1/perceive/labels "gesture_intents" and gesture_pose_mapping.py (P4-7).
        intent_scores = {
            "attacking": 1.0,
            "approaching": 0.7,
            "fast_approach": 0.85,  # mapped from Scout/body "running"
            "hands_raised": 0.88,  # mapped from "raised_hands"
            "loitering": 0.5,
            "fleeing": 0.4,
            "help_needed": 0.3,
            "normal": 0.0,
            "unknown": 0.2,
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


def _decode_frame_bgr(frame_b64: str):
    """Decode base64 JPEG to OpenCV BGR ndarray."""
    try:
        import cv2
        import numpy as np

        img_bytes = base64.b64decode(frame_b64)
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


def _classify_gesture_from_mediapipe(frame_bgr):
    """
    Estimate coarse intent from MediaPipe hand landmarks.
    Returns dict with intent/confidence/source or None.
    """
    try:
        import cv2
        import mediapipe as mp

        frame_h, frame_w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        with mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.45,
            min_tracking_confidence=0.45,
        ) as hands:
            res = hands.process(rgb)
        if not res.multi_hand_landmarks:
            return None

        best_intent = "normal"
        best_conf = 0.5
        for hand_landmarks in res.multi_hand_landmarks:
            xs = [lm.x for lm in hand_landmarks.landmark]
            ys = [lm.y for lm in hand_landmarks.landmark]
            if not xs or not ys:
                continue
            bbox_w = max(xs) - min(xs)
            bbox_h = max(ys) - min(ys)
            area_ratio = max(0.0, min(1.0, bbox_w * bbox_h))
            center_y = (max(ys) + min(ys)) / 2.0

            # Coarse intent heuristic for security triage.
            if area_ratio > 0.22:
                intent = "approaching"
                conf = min(0.95, 0.55 + area_ratio)
            elif area_ratio > 0.12 and center_y < 0.35:
                intent = "help_needed"
                conf = min(0.9, 0.5 + area_ratio)
            else:
                intent = "normal"
                conf = max(0.5, min(0.8, 0.45 + area_ratio))

            if conf > best_conf:
                best_intent = intent
                best_conf = conf

        return {
            "intent": best_intent,
            "confidence": round(float(best_conf), 4),
            "hands_visible": len(res.multi_hand_landmarks),
            "source": "mediapipe_hands",
            "frame_size": {"w": frame_w, "h": frame_h},
        }
    except Exception:
        return None


def _classify_gesture_from_contours(frame_bgr):
    """
    OpenCV-only fallback if MediaPipe isn't available.
    Uses skin-like mask + contour area for coarse intent.
    """
    try:
        import cv2
        import numpy as np

        h, w = frame_bgr.shape[:2]
        if h == 0 or w == 0:
            return None

        ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
        lower = np.array([0, 133, 77], dtype=np.uint8)
        upper = np.array([255, 173, 127], dtype=np.uint8)
        mask = cv2.inRange(ycrcb, lower, upper)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(largest))
        area_ratio = area / float(h * w)

        x, y, cw, ch = cv2.boundingRect(largest)
        center_y = (y + ch / 2.0) / float(h)

        if area_ratio > 0.20:
            intent = "approaching"
            conf = min(0.85, 0.5 + area_ratio)
        elif area_ratio > 0.10 and center_y < 0.38:
            intent = "help_needed"
            conf = min(0.78, 0.45 + area_ratio)
        elif area_ratio < 0.015:
            intent = "unknown"
            conf = 0.35
        else:
            intent = "normal"
            conf = max(0.45, min(0.7, 0.4 + area_ratio))

        return {
            "intent": intent,
            "confidence": round(float(conf), 4),
            "hands_visible": 1 if area_ratio > 0.03 else 0,
            "source": "opencv_contour_fallback",
            "area_ratio": round(float(area_ratio), 4),
            "bbox_norm": {
                "x": round(x / float(w), 4),
                "y": round(y / float(h), 4),
                "w": round(cw / float(w), 4),
                "h": round(ch / float(h), 4),
            },
        }
    except Exception:
        return None


async def call_gesture(frame_b64: str) -> Optional[dict]:
    """Gesture intent from frame. MediaPipe first, OpenCV fallback."""
    if not frame_b64:
        return None
    frame_bgr = _decode_frame_bgr(frame_b64)
    if frame_bgr is None:
        return None
    mp_result = _classify_gesture_from_mediapipe(frame_bgr)
    if mp_result is not None:
        return mp_result
    return _classify_gesture_from_contours(frame_bgr)


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

        # P4-7: When the caller (e.g. nuclear-scout pipeline) supplies a gesture_pose
        # label, bypass frame-based gesture detection and map directly through the
        # Scout → appliance taxonomy table.  This is the primary path when Scout is
        # the upstream sensor (lab/R&D mode with GESTURE_POSE_PASSTHROUGH=true) and
        # the secondary/override path for the appliance when a pre-classified label
        # arrives in the payload.
        if req.gesture_pose is not None:
            # Map Scout label → security intent; include raw label when env var set.
            gesture_result: Optional[dict] = build_gesture_dict(
                req.gesture_pose,
                req.gesture_pose_confidence,
                passthrough=passthrough_enabled(),
            )
            fer_task  = call_fer(session, req.frame_b64) if has_face else asyncio.sleep(0, result=None)
            ser_task  = call_ser(session, req.audio_b64, req.sample_rate) if has_voice else asyncio.sleep(0, result=None)
            face_result, voice_result = await asyncio.gather(fer_task, ser_task)
        else:
            fer_task  = call_fer(session, req.frame_b64) if has_face else asyncio.sleep(0, result=None)
            ser_task  = call_ser(session, req.audio_b64, req.sample_rate) if has_voice else asyncio.sleep(0, result=None)
            ges_task  = call_gesture(req.frame_b64) if has_face else asyncio.sleep(0, result=None)

            face_result, voice_result, gesture_result = await asyncio.gather(fer_task, ser_task, ges_task)

    risk  = compute_risk(face_result, voice_result, gesture_result)
    mood  = mood_summary(face_result)

    return PerceiveResponse(
        timestamp=time.time(),
        source_id=req.source_id,
        face=face_result,
        voice=voice_result,
        gesture=gesture_result,
        risk=risk,
        mood_summary=mood,
    )


@app.get("/health")
async def health():
    return {"status": "ok", "service": "nuclear-eye-perceive", "port": 8091}


@app.get("/v1/perceive/labels")
async def labels():
    from gesture_pose_mapping import (
        GESTURE_POSE_MAP,
        ALL_SCOUT_LABELS,
        APPLIANCE_ONLY_INTENTS,
    )
    return {
        "emotions": list(EMOTION_AFFECT.keys()),
        "gesture_intents": [
            "normal", "loitering", "approaching", "fast_approach",
            "hands_raised", "attacking", "fleeing", "help_needed", "unknown",
        ],
        "gesture_sources": [
            "mediapipe_hands",
            "opencv_contour_fallback",
            "scout_gesture_pose_mapping",  # P4-7
        ],
        # P4-7: Scout gesture label taxonomy
        "scout_gesture_pose_labels": sorted(ALL_SCOUT_LABELS),
        "appliance_only_intents": sorted(APPLIANCE_ONLY_INTENTS),
        "gesture_pose_map": {
            label: {
                "security_intent": entry.security_intent,
                "confidence_scale": entry.confidence_scale,
            }
            for label, entry in GESTURE_POSE_MAP.items()
        },
        "gesture_pose_passthrough": passthrough_enabled(),
    }


if __name__ == "__main__":
    port = int(os.getenv("PERCEIVE_PORT", "8091"))
    uvicorn.run(app, host="127.0.0.1", port=port)
