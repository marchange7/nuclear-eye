#!/usr/bin/env bash
# Smoke test for canonical topology:
# camera payload -> /sensor/camera -> /ingest -> /summary (+ optional /ws probe)
set -euo pipefail

ALARM_GRADER="${ALARM_GRADER_URL:-http://127.0.0.1:8080}"
TIMEOUT="${TIMEOUT:-5}"

bold='\033[1m'
green='\033[32m'
yellow='\033[33m'
red='\033[31m'
reset='\033[0m'

need_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo -e "${red}missing required command:${reset} $1"
        exit 1
    fi
}

need_cmd curl

run_timeout() {
    local secs="$1"
    shift
    if command -v timeout >/dev/null 2>&1; then
        timeout "${secs}s" "$@"
    elif command -v gtimeout >/dev/null 2>&1; then
        gtimeout "${secs}s" "$@"
    else
        "$@"
    fi
}

to_ws_url() {
    local http_url="$1"
    if [[ "$http_url" == https://* ]]; then
        printf '%s' "wss://${http_url#https://}"
    else
        printf '%s' "ws://${http_url#http://}"
    fi
}

echo -e "${bold}nuclear-eye topology smoke test${reset}"
echo "ALARM_GRADER_URL=${ALARM_GRADER}"
echo ""

echo -n "[1/4] checking alarm_grader health... "
if curl -fsS --max-time "$TIMEOUT" "${ALARM_GRADER}/summary" >/dev/null; then
    echo -e "${green}ok${reset}"
else
    echo -e "${red}failed${reset}"
    exit 1
fi

echo -n "[2/4] posting camera caption to /sensor/camera... "
CAM_PAYLOAD="$(cat <<'JSON'
{
  "camera_id": "smoke-front-door",
  "caption": "Person loitering near the front door with uncertain movement",
  "timestamp_ms": 1711666666000
}
JSON
)"
CAM_RESP="$(curl -fsS --max-time "$TIMEOUT" \
  -H "Content-Type: application/json" \
  -d "${CAM_PAYLOAD}" \
  "${ALARM_GRADER}/sensor/camera")"
echo -e "${green}ok${reset}"
echo "    response: ${CAM_RESP}"

echo -n "[3/4] posting direct VisionEvent to /ingest... "
INGEST_PAYLOAD="$(cat <<'JSON'
{
  "event_id": "smoke-ingest-001",
  "timestamp_ms": 1711666667000,
  "camera_id": "smoke-front-door",
  "behavior": "loitering",
  "risk_score": 0.72,
  "stress_level": 0.61,
  "confidence": 0.78,
  "person_detected": true,
  "person_name": null,
  "hands_visible": 2,
  "object_held": null,
  "extra_tags": ["smoke-test"],
  "vlm_caption": "person lingering near entry"
}
JSON
)"
INGEST_RESP="$(curl -fsS --max-time "$TIMEOUT" \
  -H "Content-Type: application/json" \
  -d "${INGEST_PAYLOAD}" \
  "${ALARM_GRADER}/ingest")"
echo -e "${green}ok${reset}"
echo "    response: ${INGEST_RESP}"

echo -n "[4/4] checking /summary updated... "
SUMMARY="$(curl -fsS --max-time "$TIMEOUT" "${ALARM_GRADER}/summary")"
if command -v jq >/dev/null 2>&1; then
    LEVEL="$(printf '%s' "${SUMMARY}" | jq -r '.current_level')"
    SCORE="$(printf '%s' "${SUMMARY}" | jq -r '.danger_score')"
    COUNT="$(printf '%s' "${SUMMARY}" | jq '.last_n_alarms | length')"
    echo -e "${green}ok${reset}"
    echo "    current_level=${LEVEL} danger_score=${SCORE} last_n_alarms=${COUNT}"
else
    echo -e "${yellow}ok (jq not found, raw summary below)${reset}"
    echo "    ${SUMMARY}"
fi

if command -v websocat >/dev/null 2>&1; then
    echo -n "[ws] probing /ws endpoint with websocat... "
    WS_URL="$(to_ws_url "${ALARM_GRADER}")/ws"
    if run_timeout 3 websocat -n1 "${WS_URL}" >/dev/null 2>&1; then
        echo -e "${green}reachable${reset}"
    else
        echo -e "${yellow}reachable check inconclusive${reset}"
    fi
else
    echo -e "[ws] ${yellow}skipped${reset} (install websocat for live event probe)"
fi

echo ""
echo -e "${green}${bold}smoke test finished${reset}"
