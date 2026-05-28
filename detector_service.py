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
INPUT_SIZE = int(os.getenv("DETECTOR_INPUT_SIZE", "640"))  # yolox_s = 640; nano/tiny = 416
NMS_IOU = float(os.getenv("DETECTOR_NMS_IOU", "0.45"))
# COCO classes the security gate cares about (maps to detector.rs is_relevant_class).
RELEVANT = {"person", "bicycle", "car", "motorcycle", "bus", "truck", "vehicle"}

# COCO 80-class label order matching YOLOX class indices (Megvii export).
COCO_CLASSES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
)

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


def _letterbox(img_rgb, size: int):
    """Resize keeping aspect ratio, pad to (size,size) with 114 (YOLOX convention).
    Returns (chw_float32[1,3,size,size], ratio) where ratio scales model px->orig px."""
    import numpy as np
    h, w = img_rgb.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    from PIL import Image
    resized = np.asarray(
        Image.fromarray(img_rgb).resize((nw, nh), Image.BILINEAR), dtype=np.float32
    )
    canvas = np.full((size, size, 3), 114.0, dtype=np.float32)
    canvas[:nh, :nw, :] = resized
    # YOLOX expects BGR, CHW, float32, NO /255 normalization.
    bgr = canvas[:, :, ::-1]
    chw = bgr.transpose(2, 0, 1)[None, ...].copy()
    return chw, r


def _decode_yolox(output, size: int, conf: float):
    """Decode raw YOLOX output [1, N, 85] (grid+stride) -> (boxes_xyxy, scores, cls).
    boxes are in model-input pixel space (pre-letterbox-undo)."""
    import numpy as np
    pred = output[0]  # [N, 85]
    strides = (8, 16, 32)
    grids, expanded = [], []
    for s in strides:
        g = size // s
        xv, yv = np.meshgrid(np.arange(g), np.arange(g))
        grid = np.stack((xv, yv), 2).reshape(-1, 2)
        grids.append(grid)
        expanded.append(np.full((grid.shape[0], 1), s))
    grids = np.concatenate(grids, 0)
    strides_arr = np.concatenate(expanded, 0)
    xy = (pred[:, 0:2] + grids) * strides_arr
    wh = np.exp(pred[:, 2:4]) * strides_arr
    obj = pred[:, 4:5]
    cls = pred[:, 5:]
    cls_id = cls.argmax(1)
    cls_conf = cls[np.arange(cls.shape[0]), cls_id]
    scores = (obj[:, 0] * cls_conf)
    keep = scores >= conf
    if not keep.any():
        return np.empty((0, 4)), np.empty((0,)), np.empty((0,), dtype=int)
    xy, wh = xy[keep], wh[keep]
    boxes = np.concatenate([xy - wh / 2.0, xy + wh / 2.0], 1)  # xyxy
    return boxes, scores[keep], cls_id[keep]


def _nms(boxes, scores, iou_thr: float):
    import numpy as np
    if boxes.shape[0] == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thr]
    return keep


def _run_onnx(session, image_b64: str, conf: float) -> list[dict]:
    """REAL YOLOX inference: letterbox -> ONNX forward -> grid/stride decode ->
    per-class NMS -> un-letterbox -> bbox normalized to 0..1 (top-left x/y + w/h).
    Returns [] on any failure (fail-open: vision_agent captions anyway)."""
    try:
        import numpy as np
        from PIL import Image
        raw = base64.b64decode(image_b64, validate=False)
        with Image.open(io.BytesIO(raw)) as im:
            img_rgb = np.asarray(im.convert("RGB"))
        orig_h, orig_w = img_rgb.shape[:2]

        chw, ratio = _letterbox(img_rgb, INPUT_SIZE)
        inp_name = session.get_inputs()[0].name
        output = session.run(None, {inp_name: chw})[0]

        boxes, scores, cls_ids = _decode_yolox(output, INPUT_SIZE, conf)
        if boxes.shape[0] == 0:
            return []

        # Per-class NMS.
        dets: list[dict] = []
        for c in np.unique(cls_ids):
            m = cls_ids == c
            kept = _nms(boxes[m], scores[m], NMS_IOU)
            cb, cs = boxes[m][kept], scores[m][kept]
            label = COCO_CLASSES[int(c)] if int(c) < len(COCO_CLASSES) else str(int(c))
            for b, sc in zip(cb, cs):
                # Un-letterbox: model px / ratio -> orig px, clip, then normalize.
                x1 = float(np.clip(b[0] / ratio, 0, orig_w))
                y1 = float(np.clip(b[1] / ratio, 0, orig_h))
                x2 = float(np.clip(b[2] / ratio, 0, orig_w))
                y2 = float(np.clip(b[3] / ratio, 0, orig_h))
                dets.append({
                    "class": label,
                    "confidence": round(float(sc), 4),
                    "bbox": {
                        "x": round(x1 / orig_w, 5),
                        "y": round(y1 / orig_h, 5),
                        "w": round((x2 - x1) / orig_w, 5),
                        "h": round((y2 - y1) / orig_h, 5),
                    },
                })
        dets.sort(key=lambda d: d["confidence"], reverse=True)
        return dets
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
