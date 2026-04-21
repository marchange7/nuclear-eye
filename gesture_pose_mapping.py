"""
gesture_pose_mapping.py  —  P4-7 / A3

Scout gesture label → perceive_service security intent mapping.

Canonical gesture labels emitted by nuclear-scout GestureRecognizer.swift:
    "standing" | "walking" | "running" | "crouching" | "raised_hands" | "unknown"

Plus the discrete hand-pose subset from GestureRecognizer.swift:
    "thumbsUp" | "openPalm" | "pointUp" | "peace" | "fist" | "unknown"

Mapped to perceive_service intent taxonomy (compute_risk intent_scores), including
P4-7 extensions aligned with GET /v1/perceive/labels:
    "attacking" | "approaching" | "fast_approach" | "hands_raised" | "loitering" |
    "fleeing" | "help_needed" | "normal" | "unknown"

Confidence adjustments are *multiplicative* scalars applied to the raw Scout confidence.
They encode conservative security triage bias — err toward higher-threat intent when
the pose is ambiguous, to reduce false-negative alarm miss rate on the appliance.

os/37 §9 Appendix A is the governing source of truth for the mapping rationale.
This module is the *adopted* implementation; os/37 §9 was the normative draft.
"""

from dataclasses import dataclass
from typing import Optional


# ── Pose labels ──────────────────────────────────────────────────────────────

# Body-pose set (future ONNX / whole-body model on appliance)
BODY_POSE_LABELS: tuple[str, ...] = (
    "standing",
    "walking",
    "running",
    "crouching",
    "raised_hands",
    "unknown",
)

# Hand-pose subset from Scout GestureRecognizer.swift
HAND_POSE_LABELS: tuple[str, ...] = (
    "thumbsUp",
    "openPalm",
    "pointUp",
    "peace",
    "fist",
    "unknown",
)

ALL_SCOUT_LABELS: frozenset[str] = frozenset(BODY_POSE_LABELS + HAND_POSE_LABELS)


# ── Mapping entry ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GestureMappingEntry:
    """Maps one Scout gesture label to an appliance security intent."""

    gesture_pose: str
    """Raw Scout label (e.g. 'raised_hands')."""

    security_intent: str
    """Target intent string consumed by compute_risk / alarm_grader intent_scores."""

    confidence_scale: float
    """Multiplicative scalar applied to Scout's raw confidence (0.0–1.5).
    Values > 1.0 boost confidence; values < 1.0 dampen it.
    Conservative triage: ambiguous gestures carry a small upward bias."""

    rationale: str
    """One-line explanation for audit / review."""


# ── Canonical mapping table ───────────────────────────────────────────────────
# Aligns with os/37 §9 Appendix A (adopted mapping).

GESTURE_POSE_MAP: dict[str, GestureMappingEntry] = {
    # ── Body poses ───────────────────────────────────────────────────────────
    "standing": GestureMappingEntry(
        gesture_pose="standing",
        security_intent="normal",
        confidence_scale=1.0,
        rationale="Stationary, upright — default non-threat baseline.",
    ),
    "walking": GestureMappingEntry(
        gesture_pose="walking",
        security_intent="loitering",
        confidence_scale=0.9,
        rationale=(
            "Slow ambulation without clear direction — mild loitering signal. "
            "Scale down slightly; intent resolves with trajectory over time."
        ),
    ),
    "running": GestureMappingEntry(
        gesture_pose="running",
        security_intent="fast_approach",
        confidence_scale=1.1,
        rationale=(
            "Fast locomotion toward camera zone. "
            "Mapped to approaching with elevated confidence boost."
        ),
    ),
    "crouching": GestureMappingEntry(
        gesture_pose="crouching",
        security_intent="loitering",
        confidence_scale=1.0,
        rationale=(
            "Low-profile posture consistent with lingering / concealment. "
            "Not attacking without FER/audio corroboration."
        ),
    ),
    "raised_hands": GestureMappingEntry(
        gesture_pose="raised_hands",
        security_intent="hands_raised",
        confidence_scale=1.2,
        rationale=(
            "Both hands raised above shoulders — primary threat-indicator / "
            "compliance gesture. High-confidence boost; correlates with forced "
            "surrender or active threat display."
        ),
    ),
    # ── Hand poses (from GestureRecognizer.swift) ─────────────────────────────
    "thumbsUp": GestureMappingEntry(
        gesture_pose="thumbsUp",
        security_intent="normal",
        confidence_scale=1.0,
        rationale="Benign acknowledgement — 0.0 threat term in intent_scores.",
    ),
    "openPalm": GestureMappingEntry(
        gesture_pose="openPalm",
        security_intent="normal",
        confidence_scale=1.0,
        rationale="Visible empty hands — default non-threat.",
    ),
    "peace": GestureMappingEntry(
        gesture_pose="peace",
        security_intent="normal",
        confidence_scale=1.0,
        rationale="Social gesture — default non-threat.",
    ),
    "pointUp": GestureMappingEntry(
        gesture_pose="pointUp",
        security_intent="approaching",
        confidence_scale=0.9,
        rationale=(
            "Directed index-finger gesture toward camera / zone. "
            "0.7-weight approaching family; slightly dampened pending "
            "trajectory corroboration."
        ),
    ),
    "fist": GestureMappingEntry(
        gesture_pose="fist",
        security_intent="approaching",
        confidence_scale=0.85,
        rationale=(
            "Closed hand — elevated attention. Mapped to approaching (not attacking) "
            "without FER/audio corroboration. Tune to loitering (0.5) if "
            "false-positive rate is high in field tests."
        ),
    ),
    "unknown": GestureMappingEntry(
        gesture_pose="unknown",
        security_intent="unknown",
        confidence_scale=0.6,
        rationale=(
            "Unrecognised gesture — maps to unknown intent at 0.2 × confidence. "
            "Scale factor applied on top of existing 0.2 weight in compute_risk."
        ),
    ),
}

# Intent labels NOT covered by Scout discrete set — driven by MediaPipe/contour
# heuristics or future ONNX models on the appliance.  Do not invent Scout labels
# for these.
APPLIANCE_ONLY_INTENTS: frozenset[str] = frozenset(
    {"attacking", "fleeing", "help_needed"}
)


# ── Public API ────────────────────────────────────────────────────────────────

def map_gesture_pose(
    gesture_pose: str,
    raw_confidence: float,
) -> tuple[str, float]:
    """
    Map a Scout gesture label to (security_intent, adjusted_confidence).

    Parameters
    ----------
    gesture_pose:
        Raw label from Scout (e.g. 'raised_hands', 'fist').
    raw_confidence:
        Confidence from Scout (0.0–1.0).

    Returns
    -------
    (security_intent, adjusted_confidence)
        adjusted_confidence is clamped to [0.0, 1.0].
    """
    entry = GESTURE_POSE_MAP.get(gesture_pose)
    if entry is None:
        # Unknown label — conservative unknown fallback
        return ("unknown", min(1.0, max(0.0, raw_confidence * 0.5)))

    adjusted = min(1.0, max(0.0, raw_confidence * entry.confidence_scale))
    return (entry.security_intent, adjusted)


def build_gesture_dict(
    gesture_pose: str,
    raw_confidence: float,
    *,
    passthrough: bool = False,
) -> dict:
    """
    Build the gesture dict compatible with perceive_service PerceiveResponse.gesture.

    When passthrough=True (GESTURE_POSE_PASSTHROUGH=true env var) the raw
    gesture_pose label is included alongside the mapped intent so downstream
    consumers (alarm_grader, La Rivière, Fortress audit) can see the original
    Scout label.

    Parameters
    ----------
    gesture_pose:
        Raw Scout label.
    raw_confidence:
        Raw Scout confidence (0.0–1.0).
    passthrough:
        Include raw gesture_pose in the returned dict.
    """
    intent, adjusted_conf = map_gesture_pose(gesture_pose, raw_confidence)
    result: dict = {
        "intent": intent,
        "confidence": round(adjusted_conf, 4),
        "source": "scout_gesture_pose_mapping",
    }
    if passthrough:
        result["gesture_pose"] = gesture_pose
    return result


def passthrough_enabled() -> bool:
    """Return True when GESTURE_POSE_PASSTHROUGH env var is set to 'true' (case-insensitive)."""
    import os
    return os.getenv("GESTURE_POSE_PASSTHROUGH", "").strip().lower() == "true"
