"""FastVLM ONNX on-device inference. No cloud fallback.

Q7 — FastVLM local ONNX on-device (unblocked via Thunder H100 export).

Model path: /data/models/fastvlm/fastvlm.onnx
Input:  [1, 3, 224, 224] float32 (ImageNet-normalised RGB)
Output: decoded caption string via _decode_output()

Environment variables:
  FASTVLM_ONNX_PATH  — override model path (default: /data/models/fastvlm/fastvlm.onnx)
  FASTVLM_DISABLED   — set "true" to disable without error (default: false)
"""
import io
import logging
import os
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

FASTVLM_ONNX_PATH = os.getenv("FASTVLM_ONNX_PATH", "/data/models/fastvlm/fastvlm.onnx")
FASTVLM_DISABLED = os.getenv("FASTVLM_DISABLED", "false").lower() == "true"

# ImageNet normalisation constants (same as vision encoder pre-processing)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class FastVLMONNX:
    """On-device FastVLM vision encoder via ONNX Runtime.

    Usage:
        fastvlm = FastVLMONNX()
        await fastvlm.load()
        if fastvlm.available:
            caption = await fastvlm.describe(jpeg_bytes)
    """

    def __init__(self) -> None:
        self._session = None
        self._available = False

    async def load(self) -> None:
        """Load the ONNX model. Call once at startup."""
        if FASTVLM_DISABLED:
            logger.info("FastVLM ONNX disabled via FASTVLM_DISABLED=true")
            return

        model_path = Path(FASTVLM_ONNX_PATH)
        if not model_path.exists():
            logger.warning("FastVLM ONNX not found at %s — run export-fastvlm-onnx.sh on Thunder", model_path)
            return

        try:
            import onnxruntime as ort  # noqa: PLC0415
            opts = ort.SessionOptions()
            opts.log_severity_level = 3  # suppress verbose ONNX Runtime logs
            self._session = ort.InferenceSession(
                str(model_path),
                sess_options=opts,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            self._available = True
            logger.info("FastVLM ONNX loaded from %s (providers: %s)", model_path, self._session.get_providers())
        except Exception as exc:
            logger.error("FastVLM ONNX load failed: %s", exc)

    @property
    def available(self) -> bool:
        return self._available

    async def describe(self, image_bytes: bytes, prompt: str = "Describe this scene briefly for security monitoring.") -> str:
        """Run vision inference on raw image bytes (JPEG/PNG).

        Returns a caption string. Returns a safe fallback string on any failure —
        never raises so the camera loop is never stalled by vision errors.

        Args:
            image_bytes: Raw image bytes (JPEG or PNG).
            prompt:      Text prompt (reserved for future multimodal encoder; ignored by vision-only ONNX).

        Returns:
            Caption string, or a short error description starting with "FastVLM:".
        """
        if not self._available:
            return "FastVLM unavailable"

        try:
            from PIL import Image  # noqa: PLC0415

            img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((224, 224))
            arr = np.array(img, dtype=np.float32) / 255.0          # [H, W, 3]
            arr = (arr - _MEAN) / _STD                              # ImageNet normalise
            arr = arr.transpose(2, 0, 1)[np.newaxis, ...]          # [1, 3, H, W]

            inputs = {self._session.get_inputs()[0].name: arr}
            outputs = self._session.run(None, inputs)
            return self._decode_output(outputs)

        except Exception as exc:
            logger.error("FastVLM inference failed: %s", exc)
            return f"FastVLM: vision error ({exc})"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _decode_output(self, outputs: list) -> str:
        """Decode ONNX output tensors to a caption string.

        The vision-only ONNX export produces feature embeddings, not tokens.
        A full VLM export (with text decoder) produces token-ID arrays that
        must be decoded with the tokenizer vocab.

        For the on-device vision-encoder-only variant we return a structured
        description of the top activation region so the downstream pipeline
        (alarm_grader, La Rivière) receives something actionable.

        When the full VLM ONNX is available (opset 17 with decoder head),
        replace this method with proper token decoding.
        """
        if not outputs:
            return "FastVLM: no output"

        out = outputs[0]  # primary output tensor

        # Full VLM export: output is integer token IDs [batch, seq_len]
        if out.dtype in (np.int32, np.int64) and out.ndim == 2:
            return self._decode_tokens(out[0])

        # Vision-encoder-only export: output is float embeddings [batch, tokens, dim]
        # Summarise activation statistics as a structured caption for downstream use.
        if out.ndim == 3:
            # [1, num_patches, hidden_dim]
            patch_norms = np.linalg.norm(out[0], axis=-1)           # [num_patches]
            top_idx     = int(np.argmax(patch_norms))
            mean_act    = float(patch_norms.mean())
            peak_act    = float(patch_norms.max())
            n_patches   = patch_norms.shape[0]
            grid        = int(n_patches ** 0.5)
            row, col    = divmod(top_idx, max(grid, 1))
            region      = _patch_region(row, col, grid)
            return (
                f"[FastVLM vision] peak activation in {region} region "
                f"(patch {top_idx}/{n_patches}, mean={mean_act:.2f}, peak={peak_act:.2f})"
            )

        # 2-D output [batch, dim] — classification head or pooled features
        if out.ndim == 2:
            top_class = int(np.argmax(out[0]))
            confidence = float(np.max(out[0]))
            return f"[FastVLM] class={top_class} confidence={confidence:.3f}"

        return f"[FastVLM result: shape={out.shape} dtype={out.dtype}]"

    def _decode_tokens(self, token_ids: np.ndarray) -> str:
        """Decode integer token IDs to text.

        Attempts to load the tokenizer vocab from the model directory.
        Falls back to a hex-encoded placeholder if the vocab is not present.
        """
        vocab_path = Path(FASTVLM_ONNX_PATH).parent / "vocab.json"
        if vocab_path.exists():
            import json  # noqa: PLC0415
            with open(vocab_path) as f:
                vocab = json.load(f)
            id_to_token = {v: k for k, v in vocab.items()}
            tokens = [id_to_token.get(int(tid), f"<{tid}>") for tid in token_ids]
            text = "".join(tokens).replace("▁", " ").strip()
            return text if text else "[FastVLM: empty output]"

        # No vocab — return raw IDs as fallback
        ids_preview = token_ids[:16].tolist()
        return f"[FastVLM tokens (no vocab): {ids_preview}...]"


# ---------------------------------------------------------------------------
# Module-level singleton — import and call load() once at startup
# ---------------------------------------------------------------------------

fastvlm = FastVLMONNX()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _patch_region(row: int, col: int, grid: int) -> str:
    """Map (row, col) patch index to a human-readable spatial region."""
    if grid < 2:
        return "center"
    mid = grid / 2
    v = "top" if row < mid else "bottom"
    h = "left" if col < mid else "right"
    if abs(row - mid) < 0.5 and abs(col - mid) < 0.5:
        return "center"
    return f"{v}-{h}"
