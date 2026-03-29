#!/usr/bin/env bash
# nuclear-eye systemd install script
# Run as: sudo bash deploy/install.sh
# Must be executed on b450 from /home/crew/git/nuclear-eye

set -euo pipefail

DEPLOY_DIR="$(cd "$(dirname "$0")" && pwd)"
SYSTEMD_DIR="/etc/systemd/system"
CONFIG_DIR="/home/crew/.config/nuclear-eye"
SERVICE_USER="crew"

# ── Build release binaries ────────────────────────────────────────────────
echo "→ Building release binaries…"
cargo build --release 2>&1 | tail -5

# ── Ensure config directory and env file ─────────────────────────────────
echo "→ Ensuring config directory at ${CONFIG_DIR}…"
install -d -o "${SERVICE_USER}" -g "${SERVICE_USER}" -m 750 "${CONFIG_DIR}"

if [[ ! -f "${CONFIG_DIR}/env" ]]; then
    echo "→ Creating stub env file (edit before starting services)…"
    cat > "${CONFIG_DIR}/env" <<'EOF'
# nuclear-eye environment — edit before enabling services
DATABASE_URL=postgres://crew:CHANGEME@127.0.0.1/nuclear_eye
HOUSE_JWT_TOKEN=CHANGEME
SIGNAL_TOKEN=
FORTRESS_BASE_URL=http://127.0.0.1:7710
IDENTITY_BASE_URL=http://127.0.0.1:7720
ACTUATOR_URL=http://127.0.0.1:8086
ALERT_LANG=fr
MQTT_HOST=127.0.0.1
MQTT_PORT=1883
# Uncomment to enable real camera mode:
# CAMERA_SNAPSHOT_URL=http://m4:8085/snapshot
# FASTVLM_URL=http://127.0.0.1:8091
EOF
    chown "${SERVICE_USER}:${SERVICE_USER}" "${CONFIG_DIR}/env"
    chmod 640 "${CONFIG_DIR}/env"
fi

# ── Install service files ─────────────────────────────────────────────────
SERVICES=(
    nuclear-eye-face-db.service
    nuclear-eye-alarm-grader.service
    nuclear-eye-iphone-sensor.service
    nuclear-eye-safety-aurelie.service
    nuclear-eye-safetyagent.service
    nuclear-eye-decision-agent.service
    nuclear-eye-actuator-agent.service
    nuclear-eye-vision-agent.service
    nuclear-eye-camera.service
    nuclear-eye-house-runtime.service
    nuclear-eye-house-sentinel.service
)

echo "→ Installing service files to ${SYSTEMD_DIR}…"
for svc in "${SERVICES[@]}"; do
    src="${DEPLOY_DIR}/${svc}"
    if [[ -f "${src}" ]]; then
        install -m 644 "${src}" "${SYSTEMD_DIR}/${svc}"
        echo "  ✓ ${svc}"
    else
        echo "  ✗ MISSING: ${src}" >&2
    fi
done

# ── Reload and enable ─────────────────────────────────────────────────────
echo "→ Reloading systemd daemon…"
systemctl daemon-reload

echo "→ Enabling long-running services…"
ENABLE=(
    nuclear-eye-face-db.service
    nuclear-eye-alarm-grader.service
    nuclear-eye-iphone-sensor.service
    nuclear-eye-safety-aurelie.service
    nuclear-eye-safetyagent.service
    nuclear-eye-decision-agent.service
    nuclear-eye-actuator-agent.service
    nuclear-eye-vision-agent.service
    nuclear-eye-camera.service
)
for svc in "${ENABLE[@]}"; do
    systemctl enable "${svc}"
done

echo ""
echo "Done. Review ${CONFIG_DIR}/env then:"
echo "  sudo systemctl start nuclear-eye-face-db nuclear-eye-alarm-grader"
echo "  sudo systemctl start nuclear-eye-{iphone-sensor,safety-aurelie,safetyagent,decision-agent,actuator-agent,vision-agent}"
echo "  sudo systemctl status nuclear-eye-alarm-grader"
