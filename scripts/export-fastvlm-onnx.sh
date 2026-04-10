#!/usr/bin/env bash
# export-fastvlm-onnx.sh
#
# Export the FastVLM vision encoder to ONNX opset 17 on Thunder H100.
# Produces: /data/models/fastvlm/fastvlm.onnx
#           /data/models/fastvlm/vocab.json      (if decoder tokenizer available)
#           /data/models/fastvlm/model_info.json
#
# Q7 — FastVLM local ONNX on-device (nuclear-eye)
#
# Prerequisites on Thunder:
#   CUDA 12.x, Python 3.11+, ~4GB disk for checkpoint + export
#
# Usage:
#   tnr connect 0          # open Thunder SSH tunnel
#   bash /data/git/nuclear-eye/scripts/export-fastvlm-onnx.sh
#
# After export, sync to b450:
#   rsync -avz /data/models/fastvlm/ b450:/data/models/fastvlm/
#
# Then on b450, set:
#   FASTVLM_ONNX_PATH=/data/models/fastvlm/fastvlm.onnx
# in /etc/nuclear/nuclear-eye.env and restart nuclear-eye.

set -euo pipefail

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_ID="${FASTVLM_MODEL:-apple/FastVLM}"          # HuggingFace model ID
# Fallback: if apple/FastVLM is not public, use a compatible open vision encoder
FALLBACK_MODEL_ID="openai/clip-vit-base-patch16"    # CLIP ViT-B/16 — same interface
OUT_DIR="/data/models/fastvlm"
OPSET=17
INPUT_SIZE=224

mkdir -p "${OUT_DIR}"

echo "=== FastVLM ONNX Export (Q7) ==="
echo "Target model : ${MODEL_ID}"
echo "Output dir   : ${OUT_DIR}"
echo "ONNX opset   : ${OPSET}"
echo ""

# ---------------------------------------------------------------------------
# 1. Install dependencies
# ---------------------------------------------------------------------------
echo "[1/5] Installing Python dependencies..."
pip install --quiet \
    transformers \
    torch \
    torchvision \
    onnx \
    onnxruntime-gpu \
    optimum[exporters] \
    pillow \
    numpy \
    huggingface_hub

# ---------------------------------------------------------------------------
# 2. Download model checkpoint
# ---------------------------------------------------------------------------
echo "[2/5] Downloading model checkpoint..."
python - << 'PYEOF'
import sys
from huggingface_hub import snapshot_download

model_id = "apple/FastVLM"
out_dir   = "/data/models/fastvlm/hf_checkpoint"

try:
    path = snapshot_download(repo_id=model_id, local_dir=out_dir, ignore_patterns=["*.msgpack", "*.h5"])
    print(f"Downloaded {model_id} -> {path}")
except Exception as e:
    print(f"apple/FastVLM not available ({e}), falling back to {model_id!r}")
    # Fallback: CLIP ViT-B/16 — same vision encoder architecture & interface
    fallback = "openai/clip-vit-base-patch16"
    path = snapshot_download(repo_id=fallback, local_dir=out_dir)
    print(f"Downloaded {fallback} -> {path}")
    with open("/tmp/fastvlm_fallback", "w") as f:
        f.write(fallback)
PYEOF

# ---------------------------------------------------------------------------
# 3. Export vision encoder to ONNX
# ---------------------------------------------------------------------------
echo "[3/5] Exporting vision encoder to ONNX opset ${OPSET}..."
python - << PYEOF
import json
import os
import sys
import torch
import numpy as np
from pathlib import Path

CHECKPOINT = "/data/models/fastvlm/hf_checkpoint"
OUT_DIR    = "/data/models/fastvlm"
OPSET      = ${OPSET}
INPUT_SIZE = ${INPUT_SIZE}
FALLBACK_FILE = "/tmp/fastvlm_fallback"

is_fallback = os.path.exists(FALLBACK_FILE)
model_id    = open(FALLBACK_FILE).read().strip() if is_fallback else "apple/FastVLM"

print(f"Loading model from {CHECKPOINT} (fallback={is_fallback})")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# ---- Try FastVLM native load first ----------------------------------------
session = None
try:
    from transformers import AutoModel, AutoProcessor
    processor = AutoProcessor.from_pretrained(CHECKPOINT)
    model     = AutoModel.from_pretrained(CHECKPOINT, torch_dtype=torch.float32)
    model     = model.vision_model if hasattr(model, "vision_model") else model
    model     = model.eval().to(device)
    print("Loaded as AutoModel (vision_model extracted)")

    # Save tokenizer vocab if available
    if hasattr(processor, "tokenizer"):
        vocab = processor.tokenizer.get_vocab()
        with open(f"{OUT_DIR}/vocab.json", "w") as f:
            json.dump(vocab, f)
        print(f"Saved vocab.json ({len(vocab)} tokens)")

except Exception as e:
    print(f"AutoModel load failed ({e}), using CLIPVisionModel")
    from transformers import CLIPVisionModel, CLIPProcessor
    processor = CLIPProcessor.from_pretrained(CHECKPOINT)
    model     = CLIPVisionModel.from_pretrained(CHECKPOINT, torch_dtype=torch.float32)
    model     = model.eval().to(device)
    print("Loaded CLIPVisionModel")

# ---- Export ---------------------------------------------------------------
onnx_path = f"{OUT_DIR}/fastvlm.onnx"
dummy     = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE, dtype=torch.float32).to(device)

with torch.no_grad():
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        opset_version=OPSET,
        input_names=["pixel_values"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "pixel_values":    {0: "batch"},
            "last_hidden_state": {0: "batch"},
        },
        do_constant_folding=True,
    )

print(f"Exported: {onnx_path}")

# ---- Save model info -------------------------------------------------------
import onnx
onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)
info = {
    "source_model": model_id,
    "opset": OPSET,
    "input_size": INPUT_SIZE,
    "input_name": "pixel_values",
    "output_name": "last_hidden_state",
    "onnx_ir_version": onnx_model.ir_version,
    "is_fallback": is_fallback,
}
with open(f"{OUT_DIR}/model_info.json", "w") as f:
    json.dump(info, f, indent=2)
print(f"Model info: {json.dumps(info, indent=2)}")
PYEOF

# ---------------------------------------------------------------------------
# 4. Verify with onnxruntime
# ---------------------------------------------------------------------------
echo "[4/5] Verifying ONNX model with onnxruntime..."
python - << PYEOF
import numpy as np
import onnxruntime as ort
import json

onnx_path  = "/data/models/fastvlm/fastvlm.onnx"
INPUT_SIZE = ${INPUT_SIZE}

sess = ort.InferenceSession(
    onnx_path,
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
print(f"Providers active: {sess.get_providers()}")

# Dummy inference
dummy  = np.random.randn(1, 3, INPUT_SIZE, INPUT_SIZE).astype(np.float32)
inputs = {sess.get_inputs()[0].name: dummy}
out    = sess.run(None, inputs)
print(f"Input  : {inputs[list(inputs)[0]].shape}")
print(f"Output : {out[0].shape}  dtype={out[0].dtype}")
print("ONNX verification: PASSED")
PYEOF

# ---------------------------------------------------------------------------
# 5. Summary + sync command
# ---------------------------------------------------------------------------
echo ""
echo "[5/5] Export complete."
echo ""
echo "Outputs:"
ls -lh "${OUT_DIR}"/*.onnx "${OUT_DIR}"/model_info.json 2>/dev/null || true
echo ""
echo "=== Sync to b450 ==="
echo "Run this from Thunder to copy the model to b450:"
echo ""
echo "  rsync -avz --progress /data/models/fastvlm/ crew@192.168.2.23:/data/models/fastvlm/"
echo ""
echo "Then on b450, verify:"
echo "  ls -lh /data/models/fastvlm/"
echo ""
echo "=== Activate in nuclear-eye ==="
echo "Add to /etc/nuclear/nuclear-eye.env (or systemd override):"
echo "  FASTVLM_ONNX_PATH=/data/models/fastvlm/fastvlm.onnx"
echo "  FASTVLM_DISABLED=false"
echo ""
echo "Restart service:"
echo "  sudo systemctl restart nuclear-eye"
echo ""
echo "=== Integration notes ==="
echo "nuclear-eye/src/fastvlm_onnx.py — on-device inference module"
echo "main.py describe_frame() — currently calls fastvlm_url HTTP service"
echo "To switch to local ONNX: import fastvlm from src.fastvlm_onnx"
echo "  and call: await fastvlm.load() at startup, then fastvlm.describe(frame_bytes)"
