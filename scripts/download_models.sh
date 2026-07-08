#!/usr/bin/env bash
# download_models.sh — fetch the Sentinelle detector weights.
#
# Pulls the official YOLOX-s ONNX export (Megvii-BaseDetection/YOLOX, Apache-2.0)
# into models/yolox/yolox_s.onnx — the always-on object gate model that
# detector_service.py (:18094) decodes (grid/stride + NMS, 640x640 input,
# output [1,8400,85]).
#
# Licensing: YOLOX is Apache-2.0 — commercial-safe. Do NOT substitute an
# Ultralytics YOLOv8/v11 export here (AGPL); the decode path is YOLOX-specific
# anyway. See os/117 (perception stack) + os/17 (model policy).
#
# Usage:
#   scripts/download_models.sh
#   DETECTOR_MODEL_PATH="$(pwd)/models/yolox/yolox_s.onnx" DETECTOR_MOCK=0 \
#     uvicorn detector_service:app --port 18094
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${ROOT_DIR}/models/yolox"
MODEL_PATH="${MODEL_DIR}/yolox_s.onnx"
# Pinned 0.1.1rc0 release asset (stable, versioned). Override via YOLOX_MODEL_URL.
MODEL_URL="${YOLOX_MODEL_URL:-https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx}"
MIN_BYTES=1000000  # real yolox_s.onnx is ~34MB; guard against truncated/placeholder files

mkdir -p "${MODEL_DIR}"

if [ -f "${MODEL_PATH}" ] && [ "$(wc -c < "${MODEL_PATH}")" -ge "${MIN_BYTES}" ]; then
  echo "model already present: ${MODEL_PATH} ($(wc -c < "${MODEL_PATH}") bytes)"
  exit 0
fi

echo "downloading YOLOX-s (Apache-2.0) -> ${MODEL_PATH}"
curl -fL --retry 3 -o "${MODEL_PATH}" "${MODEL_URL}"

bytes="$(wc -c < "${MODEL_PATH}")"
if [ "${bytes}" -lt "${MIN_BYTES}" ]; then
  echo "ERROR: downloaded file is only ${bytes} bytes — download failed or truncated" >&2
  rm -f "${MODEL_PATH}"
  exit 1
fi

echo "ok: ${MODEL_PATH} (${bytes} bytes)"
echo "set DETECTOR_MOCK=0 and DETECTOR_MODEL_PATH=${MODEL_PATH} to enable real inference"
