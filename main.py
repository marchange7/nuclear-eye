import asyncio
import base64
import io
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import cv2
import httpx
import toml
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
log = logging.getLogger("nuclear-camera")

CONFIG_PATH = os.environ.get("CAMERA_CONFIG", "cameras.toml")

# --- State ---
config: dict = {}
camera_states: dict[str, dict] = {}
_tasks: list[asyncio.Task] = []


# --- Config loading ---

def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return toml.load(f)


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


async def _grab_http_snapshot(url: str, client: httpx.AsyncClient) -> Optional[bytes]:
    resp = await client.get(url, timeout=10.0)
    resp.raise_for_status()
    return resp.content


async def grab_frame(cam: dict, client: httpx.AsyncClient) -> Optional[bytes]:
    url: str = cam["url"]
    if url.startswith("rtsp://"):
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
    resp = await client.post(f"{house_security_url}/sensor/camera", json=payload, timeout=10.0)
    resp.raise_for_status()


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
        "error": None,
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
                camera_states[cam_id]["error"] = None
                log.info("[%s] %s", cam_id, caption)

                # Fire-and-forget forwarding — don't let downstream failures stall the loop
                async def _forward():
                    try:
                        await post_to_fortress(cam, caption, fortress_url, client)
                    except Exception as exc:
                        log.warning("[%s] fortress post failed: %s", cam_id, exc)
                    try:
                        await post_to_house_security(cam, caption, house_security_url, client)
                    except Exception as exc:
                        log.warning("[%s] house-security post failed: %s", cam_id, exc)

                asyncio.create_task(_forward())

            except Exception as exc:
                log.warning("[%s] frame error: %s — retrying in 10s", cam_id, exc)
                camera_states[cam_id]["error"] = str(exc)
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
                "error": None,
            }

    yield

    for t in _tasks:
        t.cancel()
    await asyncio.gather(*_tasks, return_exceptions=True)


app = FastAPI(title="Nuclear Camera", version="0.1.0", lifespan=lifespan)


# --- API ---

@app.get("/health")
def health():
    return {
        "status": "ok",
        "cameras": list(camera_states.values()),
    }


@app.get("/cameras")
def list_cameras():
    return {"cameras": list(camera_states.values())}


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
