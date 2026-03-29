#!/usr/bin/env bash
# Demo: Safety→Aurélie bridge at different alarm levels
# Requires: safety_aurelie_agent running on :8086
#           (Aurélie ChatAgent on :8090 optional — fallback replies used if absent)
set -euo pipefail

BASE="${SAFETY_AURELIE_URL:-http://127.0.0.1:8086}"
BOLD='\033[1m' GREEN='\033[32m' YELLOW='\033[33m' RED='\033[31m' RESET='\033[0m'

header() { echo -e "\n${BOLD}── $1 ──${RESET}"; }
post()   { curl -sf -X POST "${BASE}/alert" -H 'Content-Type: application/json' -d "$1" | python3 -m json.tool 2>/dev/null || echo "(safety_aurelie_agent unreachable at ${BASE})"; }

echo -e "${BOLD}Safety→Aurélie Demo${RESET}"
echo "Target: ${BASE}"

if ! curl -sf "${BASE}/health" >/dev/null 2>&1; then
    echo -e "${RED}safety_aurelie_agent not reachable at ${BASE}${RESET}"
    echo "Start it with: HOUSE_SECURITY_CONFIG=config/security.toml cargo run --bin safety_aurelie_agent"
    exit 1
fi

header "1. Low alarm — routine notification"
post '{
  "alarm_id": "alarm-demo-low-001",
  "timestamp_ms": 1711497600000,
  "level": "Low",
  "danger_score": 0.35,
  "risk_score": 0.30,
  "stress_level": 0.20,
  "person_detected": true,
  "person_name": "neighbor",
  "note": "camera=front level=low behavior=passby person=neighbor object=none"
}'

header "2. Medium alarm — elevated alert"
post '{
  "alarm_id": "alarm-demo-medium-002",
  "timestamp_ms": 1711497601000,
  "level": "Medium",
  "danger_score": 0.62,
  "risk_score": 0.65,
  "stress_level": 0.55,
  "person_detected": true,
  "person_name": null,
  "note": "camera=back level=medium behavior=loitering person=unknown object=bag"
}'

header "3. High alarm — critical threat"
post '{
  "alarm_id": "alarm-demo-high-003",
  "timestamp_ms": 1711497602000,
  "level": "High",
  "danger_score": 0.91,
  "risk_score": 0.90,
  "stress_level": 0.85,
  "person_detected": true,
  "person_name": null,
  "note": "camera=front level=high behavior=aggressive person=unknown object=weapon-like risk=0.90 stress=0.85"
}'

header "4. Low alarm — nobody detected (sensor noise)"
post '{
  "alarm_id": "alarm-demo-noise-004",
  "timestamp_ms": 1711497603000,
  "level": "Low",
  "danger_score": 0.28,
  "risk_score": 0.20,
  "stress_level": 0.10,
  "person_detected": false,
  "person_name": null,
  "note": "camera=side level=low behavior=motion_detected person=none object=none"
}'

echo -e "\n${GREEN}Demo complete.${RESET}"
echo -e "${DIM:-}(If Aurélie ChatAgent was not running, fallback responses were used.)${RESET}"
