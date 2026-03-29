import asyncio
import base64
import collections
import io
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import cv2
import httpx
import numpy as np
import toml
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from PIL import Image
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
log = logging.getLogger("nuclear-camera")

CONFIG_PATH = os.environ.get("CAMERA_CONFIG", "cameras.toml")

MAX_BUFFER_SIZE = 200
MAX_BUFFER_ATTEMPTS = 10
FLUSH_INTERVAL_SECS = 30
HEALTH_STALE_SECS = 60
HEALTH_CHECK_INTERVAL_SECS = 30

# --- State ---
config: dict = {}
camera_states: dict[str, dict] = {}
_tasks: list[asyncio.Task] = []
# Offline buffer: deque of [payload, url, attempts] for failed /sensor/camera POSTs
_offline_buffer: collections.deque = collections.deque(maxlen=MAX_BUFFER_SIZE)

# --- FER (Facial Emotion Recognition) ---

_fer_model = None
_fer_backend = None  # "fer" | "deepface"

# FER universal emotion labels → AffectTriad (judgement, doubt, determination)
_FER_TO_TRIAD: dict[str, dict[str, float]] = {
    "angry":    {"judgement": 0.3, "doubt": 0.2, "determination": 0.8},
    "disgust":  {"judgement": 0.2, "doubt": 0.6, "determination": 0.5},
    "fear":     {"judgement": 0.4, "doubt": 0.9, "determination": 0.2},
    "happy":    {"judgement": 0.8, "doubt": 0.1, "determination": 0.6},
    "neutral":  {"judgement": 0.5, "doubt": 0.3, "determination": 0.5},
    "sad":      {"judgement": 0.4, "doubt": 0.7, "determination": 0.2},
    "surprise": {"judgement": 0.5, "doubt": 0.4, "determination": 0.6},
}


def _load_fer():
    global _fer_model, _fer_backend
    if _fer_model is not None:
        return _fer_model, _fer_backend

    # Primary: fer (lightweight FER2013 CNN, no GPU needed)
    try:
        from fer import FER
        detector = FER(mtcnn=False)
        _fer_model = detector
        _fer_backend = "fer"
        log.info("FER (fer2013) ready")
        return _fer_model, _fer_backend
    except Exception as e:
        log.warning("fer unavailable (%s), trying deepface…", e)

    # Fallback: deepface (heavier, more accurate)
    try:
        from deepface import DeepFace
        _fer_model = DeepFace
        _fer_backend = "deepface"
        log.info("DeepFace FER ready")
        return _fer_model, _fer_backend
    except Exception as e:
        raise RuntimeError(f"No FER backend available: {e}") from e


def _do_face_emotion(image_b64: str) -> dict:
    model, backend = _load_fer()
    raw = base64.b64decode(image_b64)
    img_array = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        # Try PIL fallback (handles non-JPEG inputs)
        buf = io.BytesIO(raw)
        pil_img = Image.open(buf).convert("RGB")
        img = np.array(pil_img)[:, :, ::-1]  # RGB→BGR

    if backend == "fer":
        results = model.detect_emotions(img)
        if not results:
            return {"dominant": "neutral", "emotions": {}, "triad": _FER_TO_TRIAD["neutral"], "face_detected": False}
        emotions_raw = results[0]["emotions"]
        dominant = max(emotions_raw, key=emotions_raw.get)
        total = sum(emotions_raw.values())
        emotions = {k: round(v / total, 4) for k, v in emotions_raw.items()} if total > 0 else emotions_raw

    else:  # deepface
        try:
            results = model.analyze(img, actions=["emotion"], enforce_detection=False, silent=True)
            result = results[0] if isinstance(results, list) else results
            emotions_raw = result.get("emotion", {})
            dominant = result.get("dominant_emotion", "neutral")
            total = sum(emotions_raw.values())
            emotions = {k: round(v / total, 4) for k, v in emotions_raw.items()} if total > 0 else {}
        except Exception:
            return {"dominant": "neutral", "emotions": {}, "triad": _FER_TO_TRIAD["neutral"], "face_detected": False}

    triad = _FER_TO_TRIAD.get(dominant.lower(), _FER_TO_TRIAD["neutral"])
    return {
        "dominant": dominant.lower(),
        "emotions": emotions,
        "triad": triad,
        "face_detected": True,
    }


class FaceEmotionRequest(BaseModel):
    image_b64: str


# --- Config loading with backup/restore ---

def load_config() -> dict:
    """Load cameras.toml with automatic backup/restore.

    Strategy:
      1. Try primary CONFIG_PATH.
      2. On failure, try CONFIG_PATH + '.bak'.
      3. Both failing → raise (fail loud, don't run blind).
      4. After successful primary load → write .bak so backup is always
         the last known-good copy.
    """
    bak_path = CONFIG_PATH + ".bak"
    used_backup = False

    try:
        with open(CONFIG_PATH) as f:
            raw = f.read()
    except OSError as primary_err:
        log.warning("config primary load failed (%s) — trying backup %s", primary_err, bak_path)
        try:
            with open(bak_path) as f:
                raw = f.read()
            used_backup = True
            log.warning("config: loaded from backup %s", bak_path)
        except OSError as bak_err:
            raise RuntimeError(
                f"config load failed — primary: {primary_err} | backup: {bak_err}"
            ) from bak_err

    try:
        cfg = toml.loads(raw)
    except Exception as parse_err:
        if not used_backup:
            log.warning("config primary parse failed (%s) — trying backup %s", parse_err, bak_path)
            try:
                with open(bak_path) as f:
                    bak_raw = f.read()
                cfg = toml.loads(bak_raw)
                used_backup = True
                log.warning("config: loaded from backup after parse failure")
            except Exception as bak_err:
                raise RuntimeError(
                    f"config parse failed — primary: {parse_err} | backup: {bak_err}"
                ) from bak_err
        else:
            raise

    # Refresh backup after a clean primary load
    if not used_backup:
        try:
            with open(bak_path, "w") as f:
                f.write(raw)
            log.debug("config: backup updated → %s", bak_path)
        except OSError as e:
            log.warning("config: failed to write backup %s: %s", bak_path, e)

    return cfg


# --- Frame grabbing ---

def _grab_rtsp_frame(url: str) -> Optional[bytes]:
    cap = cv2.VideoCapture(url)
    try:
        ok, frame = cap.read()
        if not ok or frame is None:
            return None
        _, buf = cv2.imencode(".jpg", frame)
        return buf.tobytes()
    finally:
        cap.release()


def _grab_device_frame(device_index: int) -> Optional[bytes]:
    """Grab a single frame from a local camera device (webcam, Continuity Camera, etc.)."""
    cap = cv2.VideoCapture(device_index)
    if not cap.isOpened():
        log.warning("device://%d — could not open camera", device_index)
        return None
    try:
        # Warm up: discard first frame (exposure/focus settle)
        cap.read()
        ok, frame = cap.read()
        if not ok or frame is None:
            return None
        _, buf = cv2.imencode(".jpg", frame)
        return buf.tobytes()
    finally:
        cap.release()


async def _grab_http_snapshot(url: str, client: httpx.AsyncClient) -> Optional[bytes]:
    resp = await client.get(url, timeout=10.0)
    resp.raise_for_status()
    return resp.content


async def grab_frame(cam: dict, client: httpx.AsyncClient) -> Optional[bytes]:
    url: str = cam["url"]
    if url.startswith("device://"):
        # Local device: device://0 = built-in, device://1 = Continuity Camera / USB
        try:
            device_index = int(url.removeprefix("device://"))
        except ValueError:
            log.error("Invalid device URL '%s' — expected device://<int>", url)
            return None
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _grab_device_frame, device_index)
    elif url.startswith("rtsp://"):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _grab_rtsp_frame, url)
    else:
        return await _grab_http_snapshot(url, client)


# --- Caption + forwarding ---

async def describe_frame(
    image_bytes: bytes,
    fastvlm_url: str,
    client: httpx.AsyncClient,
) -> Optional[str]:
    b64 = base64.b64encode(image_bytes).decode()
    payload = {
        "image_b64": b64,
        "prompt": "Describe this scene briefly for security monitoring.",
    }
    resp = await client.post(f"{fastvlm_url}/describe", json=payload, timeout=30.0)
    resp.raise_for_status()
    return resp.json().get("caption")


async def post_to_fortress(
    cam: dict,
    caption: str,
    fortress_url: str,
    client: httpx.AsyncClient,
    face_emotion: Optional[dict] = None,
) -> None:
    payload = {
        "source": "rtsp_camera",
        "camera_id": cam["id"],
        "camera_name": cam["name"],
        "event_id": str(uuid.uuid4()),
        "timestamp_ms": int(time.time() * 1000),
        "vlm_caption": caption,
        "alarm_hint": "person_detected" if "person" in caption.lower() else "motion_detected",
    }
    if face_emotion and face_emotion.get("face_detected"):
        payload["face_emotion"] = face_emotion["dominant"]
        payload["face_emotion_triad"] = face_emotion["triad"]
    resp = await client.post(f"{fortress_url}/v1/mesh/vision", json=payload, timeout=10.0)
    resp.raise_for_status()


async def post_to_house_security(
    cam: dict,
    caption: str,
    house_security_url: str,
    client: httpx.AsyncClient,
) -> None:
    payload = {
        "camera_id": cam["id"],
        "caption": caption,
        "timestamp_ms": int(time.time() * 1000),
    }
    url = f"{house_security_url}/sensor/camera"
    try:
        resp = await client.post(url, json=payload, timeout=10.0)
        resp.raise_for_status()
    except Exception:
        # Buffer for background retry
        _offline_buffer.append([payload, url, 0])
        raise


# --- Offline buffer flush ---

async def flush_offline_buffer(house_security_url: str) -> None:
    """Background task: retry buffered /sensor/camera events every 30s."""
    async with httpx.AsyncClient() as client:
        while True:
            await asyncio.sleep(FLUSH_INTERVAL_SECS)
            if not _offline_buffer:
                continue
            log.info("Flushing %d buffered events", len(_offline_buffer))
            pending = list(_offline_buffer)
            _offline_buffer.clear()
            for item in pending:
                payload, url, attempts = item
                try:
                    resp = await client.post(url, json=payload, timeout=10.0)
                    resp.raise_for_status()
                    log.info("Flushed buffered event camera_id=%s", payload.get("camera_id"))
                except Exception as exc:
                    attempts += 1
                    if attempts < MAX_BUFFER_ATTEMPTS:
                        _offline_buffer.append([payload, url, attempts])
                    else:
                        log.warning(
                            "Dropping buffered event camera_id=%s after %d attempts: %s",
                            payload.get("camera_id"), attempts, exc,
                        )


# --- Per-camera health monitor ---

async def camera_health_monitor() -> None:
    """Background task: log a warning for cameras that haven't produced a frame recently."""
    while True:
        await asyncio.sleep(HEALTH_CHECK_INTERVAL_SECS)
        now_ms = int(time.time() * 1000)
        for cam_id, state in camera_states.items():
            if not state.get("enabled"):
                continue
            last_seen = state.get("last_seen_ms")
            if last_seen is None:
                continue
            stale_ms = now_ms - last_seen
            if stale_ms > HEALTH_STALE_SECS * 1000:
                log.warning(
                    "[%s] camera stale — no frame for %.0fs (error: %s)",
                    cam_id, stale_ms / 1000, state.get("error"),
                )


# --- Per-camera loop ---

async def camera_loop(cam: dict, settings: dict) -> None:
    cam_id = cam["id"]
    interval = settings.get("grab_interval_secs", 2)
    fastvlm_url = settings["fastvlm_url"]
    fortress_url = settings["fortress_url"]
    house_security_url = settings["house_security_url"]

    camera_states[cam_id] = {
        "id": cam_id,
        "name": cam["name"],
        "url": cam["url"],
        "enabled": cam["enabled"],
        "last_caption": None,
        "last_seen_ms": None,
        "last_frame_bytes": None,
        "error": None,
        "consecutive_errors": 0,
        "health_ok": True,
    }

    async with httpx.AsyncClient() as client:
        while True:
            try:
                frame_bytes = await grab_frame(cam, client)
                if frame_bytes is None:
                    raise RuntimeError("Empty frame returned")

                caption = await describe_frame(frame_bytes, fastvlm_url, client)
                ts_ms = int(time.time() * 1000)
                camera_states[cam_id]["last_caption"] = caption
                camera_states[cam_id]["last_seen_ms"] = ts_ms
                camera_states[cam_id]["last_frame_bytes"] = frame_bytes
                camera_states[cam_id]["error"] = None
                camera_states[cam_id]["consecutive_errors"] = 0
                camera_states[cam_id]["health_ok"] = True
                log.info("[%s] %s", cam_id, caption)

                # FER: if a person is visible, detect facial emotion (non-blocking, best-effort)
                face_emotion_result: Optional[dict] = None
                if "person" in caption.lower() or "visage" in caption.lower() or "face" in caption.lower():
                    try:
                        loop = asyncio.get_event_loop()
                        frame_b64 = base64.b64encode(frame_bytes).decode()
                        face_emotion_result = await asyncio.wait_for(
                            loop.run_in_executor(None, _do_face_emotion, frame_b64),
                            timeout=3.0,
                        )
                        if face_emotion_result.get("face_detected"):
                            log.info("[%s] face_emotion=%s triad=%s", cam_id,
                                     face_emotion_result["dominant"], face_emotion_result["triad"])
                    except asyncio.TimeoutError:
                        log.debug("[%s] FER timeout, skipping", cam_id)
                    except RuntimeError:
                        pass  # FER backend not installed — skip silently
                    except Exception as fer_exc:
                        log.debug("[%s] FER error: %s", cam_id, fer_exc)

                # Fire-and-forget forwarding — don't let downstream failures stall the loop
                _fer_snapshot = face_emotion_result  # capture for closure

                async def _forward(fer=_fer_snapshot):
                    try:
                        await post_to_fortress(cam, caption, fortress_url, client, face_emotion=fer)
                    except Exception as exc:
                        log.warning("[%s] fortress post failed: %s", cam_id, exc)
                    try:
                        await post_to_house_security(cam, caption, house_security_url, client)
                    except Exception as exc:
                        log.warning("[%s] house-security post failed: %s", cam_id, exc)

                asyncio.create_task(_forward())

            except Exception as exc:
                n = camera_states[cam_id]["consecutive_errors"] + 1
                camera_states[cam_id]["consecutive_errors"] = n
                camera_states[cam_id]["error"] = str(exc)
                if n >= 5:
                    camera_states[cam_id]["health_ok"] = False
                    log.error("[%s] camera DOWN (%d consecutive errors): %s", cam_id, n, exc)
                else:
                    log.warning("[%s] frame error: %s — retrying in 10s", cam_id, exc)
                await asyncio.sleep(10)
                continue

            await asyncio.sleep(interval)


# --- Lifespan ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    global config
    config = load_config()
    settings = config.get("settings", {})
    cameras = config.get("cameras", [])

    house_security_url = settings.get("house_security_url", "http://127.0.0.1:8080")

    for cam in cameras:
        if cam.get("enabled", False):
            task = asyncio.create_task(camera_loop(cam, settings), name=f"cam-{cam['id']}")
            _tasks.append(task)
            log.info("Started camera loop: %s", cam["id"])
        else:
            camera_states[cam["id"]] = {
                "id": cam["id"],
                "name": cam["name"],
                "url": cam["url"],
                "enabled": False,
                "last_caption": None,
                "last_seen_ms": None,
                "last_frame_bytes": None,
                "error": None,
                "consecutive_errors": 0,
                "health_ok": True,
            }

    _tasks.append(asyncio.create_task(flush_offline_buffer(house_security_url), name="flush-buffer"))
    _tasks.append(asyncio.create_task(camera_health_monitor(), name="health-monitor"))
    log.info("Started offline buffer flush + camera health monitor")

    yield

    for t in _tasks:
        t.cancel()
    await asyncio.gather(*_tasks, return_exceptions=True)


app = FastAPI(title="Nuclear Camera", version="0.1.0", lifespan=lifespan)


# --- API ---

@app.get("/health")
def health():
    all_ok = all(s.get("health_ok", True) for s in camera_states.values() if s.get("enabled"))
    camera_views = []
    for state in camera_states.values():
        view = dict(state)
        view["has_frame"] = view.get("last_frame_bytes") is not None
        view.pop("last_frame_bytes", None)
        camera_views.append(view)
    return {
        "status": "ok" if all_ok else "degraded",
        "cameras": camera_views,
        "buffered_events": len(_offline_buffer),
    }


@app.get("/cameras")
def list_cameras():
    camera_views = []
    for state in camera_states.values():
        view = dict(state)
        view["has_frame"] = view.get("last_frame_bytes") is not None
        view.pop("last_frame_bytes", None)
        camera_views.append(view)
    return {"cameras": camera_views}


class SnapshotResponse(BaseModel):
    camera_id: str
    caption: str
    latency_ms: float


@app.post("/cameras/{cam_id}/snapshot", response_model=SnapshotResponse)
async def force_snapshot(cam_id: str):
    cfg = load_config()
    settings = cfg.get("settings", {})
    cameras_cfg = {c["id"]: c for c in cfg.get("cameras", [])}

    if cam_id not in cameras_cfg:
        raise HTTPException(status_code=404, detail=f"Camera '{cam_id}' not found")

    cam = cameras_cfg[cam_id]
    fastvlm_url = settings["fastvlm_url"]

    t0 = time.monotonic()
    async with httpx.AsyncClient() as client:
        frame_bytes = await grab_frame(cam, client)
        if frame_bytes is None:
            raise HTTPException(status_code=502, detail="Failed to grab frame")
        caption = await describe_frame(frame_bytes, fastvlm_url, client)

    latency_ms = (time.monotonic() - t0) * 1000

    if cam_id in camera_states:
        camera_states[cam_id]["last_caption"] = caption
        camera_states[cam_id]["last_seen_ms"] = int(time.time() * 1000)

    return SnapshotResponse(camera_id=cam_id, caption=caption, latency_ms=round(latency_ms, 1))


@app.get("/snapshot/{cam_id}")
async def snapshot_jpeg(cam_id: str):
    """Return latest cached JPEG frame for nuclear-watch camera grid."""
    state = camera_states.get(cam_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Camera '{cam_id}' not found")
    frame = state.get("last_frame_bytes")
    if frame is None:
        raise HTTPException(status_code=503, detail="No frame captured yet")
    return Response(content=frame, media_type="image/jpeg")


@app.post("/emotion/face")
async def face_emotion(req: FaceEmotionRequest):
    """Detect human facial emotion from a base64-encoded JPEG/PNG image.

    Returns:
      dominant: str — "angry" | "disgust" | "fear" | "happy" | "neutral" | "sad" | "surprise"
      emotions: dict — per-emotion confidence scores (normalized 0–1)
      triad: dict — mapped AffectTriad {judgement, doubt, determination}
      face_detected: bool
    """
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(None, _do_face_emotion, req.image_b64)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        log.exception("Face emotion detection failed")
        raise HTTPException(status_code=500, detail=str(e)) from e
    return result
