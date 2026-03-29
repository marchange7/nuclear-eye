#!/usr/bin/env bash
# Health check for all house-security-ai services.
set -euo pipefail

BOLD='\033[1m' GREEN='\033[32m' RED='\033[31m' RESET='\033[0m'

SERVICES=(
    "alarm_grader:http://127.0.0.1:8080/summary"
    "safetyagent:http://127.0.0.1:8081/health"
    "face_db:http://127.0.0.1:8087/faces"
    "decision_agent:http://127.0.0.1:8085/health"
    "safety_aurelie:http://127.0.0.1:8086/health"
)

echo -e "${BOLD}house-security-ai — Health Check${RESET}"
echo ""

ALL_OK=0

for entry in "${SERVICES[@]}"; do
    name="${entry%%:*}"
    url="${entry#*:}"
    if curl -sf --max-time 3 "$url" >/dev/null 2>&1; then
        echo -e "  ${GREEN}✓${RESET} ${name}  ${url}"
    else
        echo -e "  ${RED}✗${RESET} ${name}  ${url}"
        ALL_OK=1
    fi
done

# Vision agent has no HTTP port — check PID file
echo ""
PID_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data/pids"
if [ -f "${PID_DIR}/vision_agent.pid" ]; then
    VA_PID=$(cat "${PID_DIR}/vision_agent.pid")
    if kill -0 "$VA_PID" 2>/dev/null; then
        echo -e "  ${GREEN}✓${RESET} vision_agent  PID ${VA_PID}"
    else
        echo -e "  ${RED}✗${RESET} vision_agent  PID ${VA_PID} (not running)"
        ALL_OK=1
    fi
else
    echo -e "  ${RED}✗${RESET} vision_agent  (no PID file)"
    ALL_OK=1
fi

echo ""
if [ "$ALL_OK" -eq 0 ]; then
    echo -e "${GREEN}${BOLD}All services healthy.${RESET}"
else
    echo -e "${RED}${BOLD}Some services are down.${RESET}"
    exit 1
fi
