#!/usr/bin/env python3
"""
camera_server.py — M4 FaceTime camera snapshot server.

Serves JPEG snapshots from the built-in webcam to nuclear-eye vision_agent (b450).

Usage:
    python3 scripts/camera_server.py
    CAMERA_DEVICE=0 CAMERA_PORT=8085 python3 scripts/camera_server.py

Endpoints:
    GET /snapshot  → JPEG frame (Content-Type: image/jpeg)
    GET /health    → {"status": "ok", "camera": true, "device": 0}
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import threading
import time
from typing import Optional

import cv2
from aiohttp import web

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("camera-server")

CAMERA_DEVICE = int(os.getenv("CAMERA_DEVICE", "0"))
CAMERA_PORT   = int(os.getenv("CAMERA_PORT", "8085"))
JPEG_QUALITY  = int(os.getenv("JPEG_QUALITY", "85"))
FRAME_INTERVAL = float(os.getenv("FRAME_INTERVAL", "0.1"))  # 10 fps pre-capture


class CameraCapture:
    """Thread-safe continuous camera capture — always serves latest frame."""

    def __init__(self, device: int) -> None:
        self.device = device
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame: Optional[bytes] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_capture: float = 0.0

    def start(self) -> None:
        self._cap = cv2.VideoCapture(self.device)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera device {self.device}")
        # Set resolution — 1280x720 good balance for FastVLM
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        # Wait for first frame
        deadline = time.time() + 5.0
        while self._frame is None and time.time() < deadline:
            time.sleep(0.05)
        if self._frame is None:
            raise RuntimeError("Camera did not produce a frame within 5s")
        log.info("camera.ready device=%d", self.device)

    def _capture_loop(self) -> None:
        while self._running and self._cap is not None:
            ret, frame = self._cap.read()
            if not ret:
                log.warning("camera.read.failed — retrying")
                time.sleep(0.5)
                continue
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            with self._lock:
                self._frame = buf.tobytes()
                self._last_capture = time.time()
            time.sleep(FRAME_INTERVAL)

    def snapshot(self) -> Optional[bytes]:
        with self._lock:
            return self._frame

    def is_healthy(self) -> bool:
        if self._frame is None:
            return False
        return (time.time() - self._last_capture) < 5.0

    def stop(self) -> None:
        self._running = False
        if self._cap:
            self._cap.release()


_camera = CameraCapture(CAMERA_DEVICE)


async def handle_snapshot(request: web.Request) -> web.Response:
    frame = _camera.snapshot()
    if frame is None:
        return web.Response(status=503, text="Camera not ready")
    return web.Response(
        body=frame,
        content_type="image/jpeg",
        headers={"Cache-Control": "no-cache", "X-Camera-Device": str(CAMERA_DEVICE)},
    )


async def handle_health(request: web.Request) -> web.Response:
    healthy = _camera.is_healthy()
    return web.json_response({
        "status": "ok" if healthy else "degraded",
        "camera": healthy,
        "device": CAMERA_DEVICE,
        "last_capture_age_s": round(time.time() - _camera._last_capture, 2) if _camera._frame else None,
    }, status=200 if healthy else 503)


async def on_startup(app: web.Application) -> None:
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _camera.start)
    log.info("camera_server.started port=%d device=%d", CAMERA_PORT, CAMERA_DEVICE)


async def on_cleanup(app: web.Application) -> None:
    _camera.stop()
    log.info("camera_server.stopped")


def main() -> None:
    app = web.Application()
    app.router.add_get("/snapshot", handle_snapshot)
    app.router.add_get("/health", handle_health)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    web.run_app(app, host="0.0.0.0", port=CAMERA_PORT, access_log=None)


if __name__ == "__main__":
    main()
