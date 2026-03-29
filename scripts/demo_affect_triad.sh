#!/usr/bin/env bash
# Demo: AffectTriad computation via DecisionAgent
# Requires: decision_agent running on :8085
set -euo pipefail

BASE="${DECISION_AGENT_URL:-http://127.0.0.1:8085}"
BOLD='\033[1m' DIM='\033[2m' GREEN='\033[32m' YELLOW='\033[33m' RED='\033[31m' RESET='\033[0m'

header() { echo -e "\n${BOLD}── $1 ──${RESET}"; }
post()   { curl -sf -X POST "${BASE}/decide" -H 'Content-Type: application/json' -d "$1" | python3 -m json.tool 2>/dev/null || echo "(decision_agent unreachable at ${BASE})"; }

echo -e "${BOLD}AffectTriad Demo — DecisionAgent${RESET}"
echo "Target: ${BASE}"

# Health check
if ! curl -sf "${BASE}/health" >/dev/null 2>&1; then
    echo -e "${RED}decision_agent not reachable at ${BASE}${RESET}"
    echo "Start it with: HOUSE_SECURITY_CONFIG=config/security.toml cargo run --bin decision_agent"
    exit 1
fi

header "1. Calm scene — low risk, low stress, high confidence"
post '{
  "event": {
    "event_id": "demo-calm-001",
    "timestamp_ms": 1711497600000,
    "camera_id": "front-door",
    "behavior": "passby",
    "risk_score": 0.15,
    "stress_level": 0.10,
    "confidence": 0.95,
    "person_detected": true,
    "person_name": "known-resident",
    "hands_visible": 2,
    "object_held": null,
    "extra_tags": ["normal_motion"]
  }
}'

header "2. Suspicious — medium risk, elevated stress"
post '{
  "event": {
    "event_id": "demo-suspicious-002",
    "timestamp_ms": 1711497601000,
    "camera_id": "back-garden",
    "behavior": "loitering",
    "risk_score": 0.65,
    "stress_level": 0.55,
    "confidence": 0.80,
    "person_detected": true,
    "person_name": null,
    "hands_visible": 1,
    "object_held": "unknown_object",
    "extra_tags": ["repeat_pass", "attention_house"]
  }
}'

header "3. High threat — high risk, high stress, high confidence"
post '{
  "event": {
    "event_id": "demo-threat-003",
    "timestamp_ms": 1711497602000,
    "camera_id": "front-door",
    "behavior": "aggressive",
    "risk_score": 0.92,
    "stress_level": 0.88,
    "confidence": 0.90,
    "person_detected": true,
    "person_name": null,
    "hands_visible": 2,
    "object_held": "weapon-like",
    "extra_tags": ["rapid_gesture", "high_stress"]
  }
}'

header "4. Uncertain — low confidence, moderate risk"
post '{
  "event": {
    "event_id": "demo-uncertain-004",
    "timestamp_ms": 1711497603000,
    "camera_id": "side-window",
    "behavior": "unknown",
    "risk_score": 0.45,
    "stress_level": 0.30,
    "confidence": 0.25,
    "person_detected": false,
    "person_name": null,
    "hands_visible": 0,
    "object_held": null,
    "extra_tags": ["low_visibility", "night"]
  }
}'

header "5. Force relational mode on a risky event"
post '{
  "event": {
    "event_id": "demo-force-relational-005",
    "timestamp_ms": 1711497604000,
    "camera_id": "front-door",
    "behavior": "pacing",
    "risk_score": 0.70,
    "stress_level": 0.60,
    "confidence": 0.85,
    "person_detected": true,
    "person_name": "known-resident",
    "hands_visible": 2,
    "object_held": null,
    "extra_tags": []
  },
  "force_safety": false
}'

echo -e "\n${GREEN}Demo complete.${RESET}"
