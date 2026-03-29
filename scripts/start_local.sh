#!/usr/bin/env bash
# Start all nuclear-eye Rust services locally for development.
# Services start in dependency order with a brief pause between each.
# Logs go to data/logs/<service>.log (created if needed).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HOUSE_SECURITY_CONFIG="${ROOT_DIR}/config/security.local.toml"
export RUST_LOG="${RUST_LOG:-info}"
BUILD_PROFILE="${BUILD_PROFILE:-dev}"
LOG_DIR="${ROOT_DIR}/data/logs"
PID_DIR="${ROOT_DIR}/data/pids"

mkdir -p "$LOG_DIR" "$PID_DIR"

BOLD='\033[1m' GREEN='\033[32m' RED='\033[31m' RESET='\033[0m'

check_port_free() {
    local port="$1"
    local owner
    owner="$(lsof -nP -iTCP:"${port}" -sTCP:LISTEN 2>/dev/null | awk 'NR==2 {print $1 " (PID " $2 ")"}')"
    if [ -n "$owner" ]; then
        echo -e "  ${RED}✗${RESET} port ${port} already in use by ${owner}"
        echo "  Stop the conflicting process or run with different binds in config."
        exit 1
    fi
}

start_svc() {
    local name="$1" bin="$2" port="$3"
    shift 3
    local -a env_args=("$@")
    local -a cargo_args=(run --bin "$bin")
    if [ "$BUILD_PROFILE" = "release" ]; then
        cargo_args=(run --release --bin "$bin")
    fi

    if [ -f "${PID_DIR}/${name}.pid" ]; then
        local old_pid
        old_pid=$(cat "${PID_DIR}/${name}.pid")
        if kill -0 "$old_pid" 2>/dev/null; then
            echo -e "  ${GREEN}↻${RESET} ${name} already running (PID ${old_pid})"
            return 0
        fi
    fi

    if [ "${#env_args[@]}" -gt 0 ]; then
        env "${env_args[@]}" cargo "${cargo_args[@]}" >"${LOG_DIR}/${name}.log" 2>&1 &
    else
        cargo "${cargo_args[@]}" >"${LOG_DIR}/${name}.log" 2>&1 &
    fi
    local pid=$!
    echo "$pid" > "${PID_DIR}/${name}.pid"
    echo -e "  ${GREEN}✓${RESET} ${name} started (PID ${pid}, port ${port}, profile ${BUILD_PROFILE})"
}

wait_for() {
    local url="$1" name="$2" max=20 i=0
    while ! curl -sf "$url" >/dev/null 2>&1; do
        i=$((i + 1))
        if [ "$i" -ge "$max" ]; then
            echo -e "  ${RED}✗${RESET} ${name} did not become healthy within ${max}s"
            return 1
        fi
        sleep 1
    done
}

echo -e "${BOLD}Starting nuclear-eye local stack...${RESET}"
echo "Config: ${HOUSE_SECURITY_CONFIG}"
echo "Logs:   ${LOG_DIR}/"
echo "Profile: ${BUILD_PROFILE}"
echo ""

check_port_free 8080
check_port_free 8081
check_port_free 8085
check_port_free 8086
check_port_free 8087

# 1. Face DB (:8087)
start_svc "face_db" "face_db" "8087"
sleep 1

# 2. Alarm Grader (:8080)
start_svc "alarm_grader" "alarm_grader_agent" "8080"
sleep 1

# 3. SafetyAgent (:8081) — depends on alarm_grader
start_svc "safetyagent" "safetyagent" "8081"
sleep 1

# 4. Decision Agent (:8085)
start_svc "decision_agent" "decision_agent" "8085" "DECISION_AGENT_BIND=0.0.0.0:8085"
sleep 1

# 5. Safety Aurélie Agent (:8086) — depends on safetyagent (for alarm events)
start_svc "safety_aurelie" "safety_aurelie_agent" "8086" "SAFETY_AURELIE_BIND=0.0.0.0:8086"
sleep 1

# 6. Vision Agent (no port — sends events to alarm_grader)
start_svc "vision_agent" "vision_agent" "—"
sleep 1

echo ""
echo -e "${BOLD}Waiting for services to become healthy...${RESET}"
wait_for "http://127.0.0.1:8087/faces" "face_db"
wait_for "http://127.0.0.1:8080/summary" "alarm_grader"
wait_for "http://127.0.0.1:8081/health" "safetyagent"
wait_for "http://127.0.0.1:8085/health" "decision_agent"
wait_for "http://127.0.0.1:8086/health" "safety_aurelie"

echo ""
echo -e "${GREEN}${BOLD}All services started.${RESET}"
echo "Run scripts/health_check_all.sh to verify."
echo "Run scripts/stop_local.sh to stop all services."
