#!/usr/bin/env bash
# Demo: Full pipeline walkthrough with curl
#
# This script does NOT start the services — it assumes they are already running.
# Use scripts/start_local.sh first, or run each binary manually.
#
# Flow: VisionEvent → SafetyAgent → AlarmGrader → (Telegram)
#                   → DecisionAgent (triad)
#                   → SafetyAurélieAgent (empathetic response)
set -euo pipefail

SAFETYAGENT="${SAFETYAGENT_URL:-http://127.0.0.1:8081}"
DECISION="${DECISION_AGENT_URL:-http://127.0.0.1:8085}"
AURELIE_BRIDGE="${SAFETY_AURELIE_URL:-http://127.0.0.1:8086}"
ALARM_GRADER="${ALARM_GRADER_URL:-http://127.0.0.1:8080}"
BOLD='\033[1m' DIM='\033[2m' GREEN='\033[32m' YELLOW='\033[33m' RED='\033[31m' CYAN='\033[36m' RESET='\033[0m'

header() { echo -e "\n${BOLD}${CYAN}═══ $1 ═══${RESET}"; }
step()   { echo -e "${BOLD}→ $1${RESET}"; }
warn()   { echo -e "${YELLOW}⚠ $1${RESET}"; }

echo -e "${BOLD}Full Pipeline Demo${RESET}"
echo "Services: alarm_grader=${ALARM_GRADER} safetyagent=${SAFETYAGENT} decision=${DECISION} aurelie_bridge=${AURELIE_BRIDGE}"
echo ""

# Pre-flight health checks
FAIL=0
for svc in "${ALARM_GRADER}/summary" "${SAFETYAGENT}/health" "${DECISION}/health" "${AURELIE_BRIDGE}/health"; do
    if curl -sf "$svc" >/dev/null 2>&1; then
        echo -e "  ${GREEN}✓${RESET} $svc"
    else
        echo -e "  ${RED}✗${RESET} $svc"
        FAIL=1
    fi
done
if [ "$FAIL" -eq 1 ]; then
    warn "Some services are down. Run scripts/start_local.sh first."
    warn "Continuing anyway — unreachable services will show errors."
    echo ""
fi

EVENT='{
  "event_id": "demo-pipeline-001",
  "timestamp_ms": 1711497600000,
  "camera_id": "front-door",
  "behavior": "loitering",
  "risk_score": 0.72,
  "stress_level": 0.65,
  "confidence": 0.80,
  "person_detected": true,
  "person_name": null,
  "hands_visible": 1,
  "object_held": "unknown_object",
  "extra_tags": ["repeat_pass", "attention_house"]
}'

# ── Step 1: DecisionAgent ──
header "Step 1: DecisionAgent — AffectTriad from VisionEvent"
step "POST ${DECISION}/decide"
DECIDE_RESP=$(curl -sf -X POST "${DECISION}/decide" -H 'Content-Type: application/json' \
    -d "{\"event\": ${EVENT}}" 2>/dev/null) && echo "$DECIDE_RESP" | python3 -m json.tool || warn "DecisionAgent unreachable"

# ── Step 2: SafetyAgent → AlarmGrader ──
header "Step 2: SafetyAgent → AlarmGrader — grade the event"
step "POST ${SAFETYAGENT}/evaluate"
SAFETY_RESP=$(curl -sf -X POST "${SAFETYAGENT}/evaluate" -H 'Content-Type: application/json' \
    -d "${EVENT}" 2>/dev/null) && echo "$SAFETY_RESP" | python3 -m json.tool || warn "SafetyAgent unreachable"

# ── Step 3: Check AlarmGrader summary ──
header "Step 3: AlarmGrader — current summary"
step "GET ${ALARM_GRADER}/summary"
curl -sf "${ALARM_GRADER}/summary" 2>/dev/null | python3 -m json.tool || warn "AlarmGrader unreachable"

# ── Step 4: SafetyAurélieAgent ──
header "Step 4: SafetyAurélieAgent — empathetic alarm response"
step "POST ${AURELIE_BRIDGE}/alert"

# Build an AlarmEvent from what we know
ALARM='{
  "alarm_id": "alarm-demo-pipeline-001",
  "timestamp_ms": 1711497600000,
  "level": "Medium",
  "danger_score": 0.67,
  "risk_score": 0.72,
  "stress_level": 0.65,
  "person_detected": true,
  "person_name": null,
  "note": "camera=front-door level=medium behavior=loitering person=unknown object=unknown_object risk=0.72 stress=0.65 conf=0.80"
}'
curl -sf -X POST "${AURELIE_BRIDGE}/alert" -H 'Content-Type: application/json' \
    -d "${ALARM}" 2>/dev/null | python3 -m json.tool || warn "SafetyAurélieAgent unreachable"

# ── Summary ──
header "Pipeline complete"
echo -e "${GREEN}VisionEvent → SafetyAgent → AlarmGrader → alarm graded${RESET}"
echo -e "${GREEN}VisionEvent → DecisionAgent → AffectTriad + DecisionAction${RESET}"
echo -e "${GREEN}AlarmEvent → SafetyAurélieAgent → Aurélie reply + Telegram${RESET}"
