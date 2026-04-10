"""
face_embedding_service.py — ArcFace embedding sidecar for face_db (Track O4).

Serves 512-dim face embeddings via FastAPI at :5555.
Uses insightface (ArcFace) for biometric face recognition.

Endpoints:
  POST /embed         — base64 image → 512-dim float vector
  POST /compare       — two base64 images → cosine similarity score
  GET  /health        — liveness check

Run:
  pip install fastapi uvicorn insightface onnxruntime numpy pillow
  python face_embedding_service.py

GPU acceleration: set ONNX_PROVIDERS=CUDAExecutionProvider (auto-detected if
onnxruntime-gpu is installed and CUDA is available).
CPU is the default (Jetson / b450 without CUDA fallback).
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logger = logging.getLogger("face_embedding")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

# ── Model init ─────────────────────────────────────────────────────────────────

_MODEL: Optional[object] = None  # insightface FaceAnalysis


def _get_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    import insightface  # lazy — don't crash on import if unused

    providers_env = os.environ.get("ONNX_PROVIDERS", "")
    if providers_env:
        providers = [p.strip() for p in providers_env.split(",")]
    else:
        # Try CUDA, fall back to CPU silently
        try:
            import onnxruntime as ort

            available = ort.get_available_providers()
            providers = ["CUDAExecutionProvider"] if "CUDAExecutionProvider" in available else ["CPUExecutionProvider"]
        except ImportError:
            providers = ["CPUExecutionProvider"]

    logger.info("insightface providers: %s", providers)

    app_model = insightface.app.FaceAnalysis(
        name="buffalo_l",  # ArcFace R100, 512-dim
        providers=providers,
        allowed_modules=["detection", "recognition"],
    )
    # det_size must be divisible by 32; 640×640 works for most cameras
    app_model.prepare(ctx_id=0, det_size=(640, 640))
    _MODEL = app_model
    logger.info("ArcFace model ready")
    return _MODEL


# ── Request / response models ─────────────────────────────────────────────────


class EmbedRequest(BaseModel):
    """Single face embedding from a base64-encoded image."""

    image_b64: str
    """Base64-encoded JPEG or PNG (whole frame or pre-cropped face)."""

    max_faces: int = 1
    """Return embeddings for up to this many faces (sorted by detection score)."""


class EmbedResponse(BaseModel):
    ok: bool
    faces_detected: int
    embeddings: List[List[float]]
    """List of 512-dim vectors, one per detected face."""

    detection_scores: List[float]
    """Confidence scores from the face detector (same order as embeddings)."""


class CompareRequest(BaseModel):
    image_b64_a: str
    image_b64_b: str


class CompareResponse(BaseModel):
    ok: bool
    similarity: float
    """Cosine similarity in [−1, 1]. Values above ~0.28 indicate same person."""

    same_person: bool
    """True when similarity ≥ threshold (default 0.28 for ArcFace buffalo_l)."""

    threshold: float


# ── Helpers ───────────────────────────────────────────────────────────────────

SAME_PERSON_THRESHOLD = float(os.environ.get("ARCFACE_THRESHOLD", "0.28"))


def _decode_image(image_b64: str) -> np.ndarray:
    """Decode base64 image to BGR numpy array (OpenCV format)."""
    import cv2
    from PIL import Image

    try:
        raw = base64.b64decode(image_b64)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid base64: {exc}") from exc

    pil_img = Image.open(io.BytesIO(raw)).convert("RGB")
    # insightface expects BGR
    bgr = np.array(pil_img)[:, :, ::-1]
    return bgr


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b))


def _embed(image_b64: str, max_faces: int = 1) -> tuple[list[list[float]], list[float]]:
    """Return (embeddings, detection_scores) for up to max_faces in the image."""
    model = _get_model()
    img = _decode_image(image_b64)
    faces = model.get(img)

    if not faces:
        return [], []

    # Sort by detection score descending, keep top-N
    faces = sorted(faces, key=lambda f: f.det_score, reverse=True)[:max_faces]

    embeddings = [face.normed_embedding.tolist() for face in faces]
    scores = [float(face.det_score) for face in faces]
    return embeddings, scores


# ── GDPR purge ────────────────────────────────────────────────────────────────

GDPR_RETENTION_DAYS = int(os.environ.get("GDPR_FACE_RETENTION_DAYS", "30"))
_FACE_DB_DIR = os.environ.get("FACE_DB_DIR", "/app/face_db")
_DATABASE_URL = os.environ.get("DATABASE_URL", "")


async def purge_old_faces() -> int:
    """
    Delete face records not seen within GDPR_RETENTION_DAYS (default 30).

    Supports two storage backends:
    1. PostgreSQL (DATABASE_URL set): deletes rows from the faces table where
       last_seen_at < now() - interval.
    2. Filesystem fallback (FACE_DB_DIR): deletes subdirectories/files whose
       mtime is older than the retention window.

    Returns the count of deleted records/files.
    """
    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=GDPR_RETENTION_DAYS)
    deleted = 0

    if _DATABASE_URL:
        try:
            import asyncpg  # optional dep — only needed when DATABASE_URL is set

            conn = await asyncpg.connect(_DATABASE_URL)
            try:
                result = await conn.execute(
                    """
                    DELETE FROM faces
                    WHERE last_seen_at < $1
                    """,
                    cutoff,
                )
                # asyncpg returns "DELETE N" as a string
                deleted = int(result.split()[-1]) if result else 0
                logger.info(
                    "GDPR purge (postgres): deleted %d face records older than %d days",
                    deleted,
                    GDPR_RETENTION_DAYS,
                )
            finally:
                await conn.close()
        except Exception as exc:
            logger.error("GDPR purge (postgres) failed: %s", exc)
    else:
        # Filesystem fallback: treat each top-level entry in FACE_DB_DIR as one identity.
        # Each identity dir/file is purged if its mtime (last access) is past the cutoff.
        import os as _os

        cutoff_ts = cutoff.timestamp()
        try:
            for entry in _os.scandir(_FACE_DB_DIR):
                try:
                    mtime = entry.stat().st_mtime
                    if mtime < cutoff_ts:
                        if entry.is_dir(follow_symlinks=False):
                            import shutil
                            shutil.rmtree(entry.path, ignore_errors=True)
                        else:
                            _os.remove(entry.path)
                        deleted += 1
                        logger.debug("GDPR purge: removed %s (mtime %s)", entry.path, datetime.fromtimestamp(mtime))
                except Exception as exc:
                    logger.warning("GDPR purge: could not remove %s: %s", entry.path, exc)
            logger.info(
                "GDPR purge (filesystem): deleted %d face entries older than %d days",
                deleted,
                GDPR_RETENTION_DAYS,
            )
        except FileNotFoundError:
            logger.debug("GDPR purge: FACE_DB_DIR %s does not exist, skipping", _FACE_DB_DIR)
        except Exception as exc:
            logger.error("GDPR purge (filesystem) failed: %s", exc)

    return deleted


async def gdpr_purge_loop() -> None:
    """Background task: run purge_old_faces() every 24 hours."""
    while True:
        try:
            await purge_old_faces()
        except Exception as exc:
            logger.error("gdpr_purge_loop unexpected error: %s", exc)
        await asyncio.sleep(86400)  # 24 hours


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="face_embedding_service",
    description="ArcFace 512-dim face embeddings for nuclear-eye face_db (O4)",
    version="1.0.0",
)


@app.on_event("startup")
async def _startup():
    """Pre-warm the model and start GDPR purge background task."""
    try:
        _get_model()
    except Exception as exc:
        logger.warning("model pre-warm failed (will retry on first request): %s", exc)

    asyncio.create_task(gdpr_purge_loop())
    logger.info(
        "GDPR purge loop started (retention=%d days, backend=%s)",
        GDPR_RETENTION_DAYS,
        "postgres" if _DATABASE_URL else f"filesystem:{_FACE_DB_DIR}",
    )


@app.get("/health")
async def health():
    return {"ok": True, "service": "face_embedding", "model": "ArcFace-buffalo_l-512d"}


@app.post("/embed", response_model=EmbedResponse)
async def embed(req: EmbedRequest):
    """
    Extract ArcFace 512-dim embeddings from an image.

    Input: base64 JPEG/PNG (whole frame or pre-cropped face region).
    Output: list of 512-dim float vectors (one per face, sorted by confidence).
    """
    try:
        embeddings, scores = _embed(req.image_b64, req.max_faces)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("embed error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return EmbedResponse(
        ok=True,
        faces_detected=len(embeddings),
        embeddings=embeddings,
        detection_scores=scores,
    )


@app.post("/compare", response_model=CompareResponse)
async def compare(req: CompareRequest):
    """
    Compare two face images and return cosine similarity.

    Uses the highest-confidence face from each image.
    """
    try:
        emb_a, _ = _embed(req.image_b64_a, max_faces=1)
        emb_b, _ = _embed(req.image_b64_b, max_faces=1)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("compare error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if not emb_a or not emb_b:
        raise HTTPException(status_code=422, detail="no face detected in one or both images")

    sim = _cosine_similarity(np.array(emb_a[0]), np.array(emb_b[0]))
    return CompareResponse(
        ok=True,
        similarity=sim,
        same_person=sim >= SAME_PERSON_THRESHOLD,
        threshold=SAME_PERSON_THRESHOLD,
    )


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("FACE_EMBED_PORT", "5555"))
    host = os.environ.get("FACE_EMBED_HOST", "127.0.0.1")
    logger.info("face_embedding_service starting on %s:%d", host, port)
    uvicorn.run(app, host=host, port=port, log_level="info")
