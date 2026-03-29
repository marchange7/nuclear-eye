#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH="${ROOT_DIR}/models/yolov8n.onnx"

if [ -s "${MODEL_PATH}" ]; then
  echo "model already present: ${MODEL_PATH}"
  exit 0
fi

cat > "${MODEL_PATH}" <<'EOF'
PLACEHOLDER_MODEL_FILE
Download a real yolov8n.onnx into this path before production inference.
EOF

echo "placeholder created at ${MODEL_PATH}"
