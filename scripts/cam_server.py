#!/usr/bin/env python3
"""
cam_server.py — serves JPEG snapshots from Mac FaceTime / USB camera.

Usage:
    python3 scripts/cam_server.py
    python3 scripts/cam_server.py --port 8085 --device 0 --quality 80

Endpoints:
    GET /snapshot   — JPEG frame (for CAMERA_SNAPSHOT_URL)
    GET /health     — {"status":"ok","fps":...}
"""

import argparse
import base64
import io
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import cv2

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("cam_server")

# ── Shared frame buffer ───────────────────────────────────────────────────

_lock = threading.Lock()
_frame_jpg: bytes = b""
_last_ts: float = 0.0
_fps: float = 0.0


def capture_loop(device: int, quality: int, target_fps: int) -> None:
    global _frame_jpg, _last_ts, _fps
    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        log.error("Cannot open camera device %d", device)
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    interval = 1.0 / target_fps
    prev_time = time.time()
    log.info("Camera %d opened — serving at /%d fps", device, target_fps)

    while True:
        ret, frame = cap.read()
        if not ret:
            log.warning("Frame grab failed — retrying")
            time.sleep(0.5)
            continue

        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok:
            now = time.time()
            with _lock:
                _frame_jpg = buf.tobytes()
                _last_ts = now
                _fps = round(1.0 / max(now - prev_time, 0.001), 1)
            prev_time = now

        elapsed = time.time() - prev_time
        wait = interval - elapsed
        if wait > 0:
            time.sleep(wait)


# ── HTTP handler ──────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass  # silence access log

    def do_GET(self):
        if self.path in ("/snapshot", "/snapshot.jpg"):
            with _lock:
                jpg = _frame_jpg
            if not jpg:
                self.send_error(503, "No frame yet")
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(jpg)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(jpg)

        elif self.path == "/health":
            with _lock:
                age = round(time.time() - _last_ts, 2) if _last_ts else -1
                fps = _fps
            body = f'{{"status":"ok","fps":{fps},"frame_age_s":{age}}}\n'.encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        else:
            self.send_error(404)


# ── Entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port",    type=int, default=8085)
    parser.add_argument("--device",  type=int, default=0,   help="cv2 camera index")
    parser.add_argument("--quality", type=int, default=80,  help="JPEG quality 1-100")
    parser.add_argument("--fps",     type=int, default=5,   help="capture rate")
    args = parser.parse_args()

    t = threading.Thread(target=capture_loop, args=(args.device, args.quality, args.fps), daemon=True)
    t.start()

    # Wait for first frame
    for _ in range(20):
        with _lock:
            if _frame_jpg:
                break
        time.sleep(0.2)

    server = HTTPServer(("0.0.0.0", args.port), Handler)
    log.info("Snapshot server on http://0.0.0.0:%d/snapshot", args.port)
    log.info("Add to cameras.toml: url = \"http://192.168.2.22:%d/snapshot\"", args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Stopped")
