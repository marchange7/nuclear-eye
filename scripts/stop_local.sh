#!/usr/bin/env bash
# Stop all locally-running nuclear-eye services.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_DIR="${ROOT_DIR}/data/pids"

BOLD='\033[1m' GREEN='\033[32m' DIM='\033[2m' RESET='\033[0m'

echo -e "${BOLD}Stopping nuclear-eye services...${RESET}"

STOPPED=0

if [ -d "$PID_DIR" ]; then
    for pidfile in "${PID_DIR}"/*.pid; do
        [ -f "$pidfile" ] || continue
        name="$(basename "$pidfile" .pid)"
        pid="$(cat "$pidfile")"
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
            echo -e "  ${GREEN}✓${RESET} ${name} (PID ${pid}) stopped"
            STOPPED=$((STOPPED + 1))
        else
            echo -e "  ${DIM}· ${name} (PID ${pid}) was not running${RESET}"
        fi
        rm -f "$pidfile"
    done
fi

if [ "$STOPPED" -eq 0 ]; then
    echo "  No running services found."
fi

echo -e "${BOLD}Done.${RESET}"
