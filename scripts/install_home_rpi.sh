#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

sudo apt-get update
sudo apt-get install -y curl git sqlite3 ca-certificates

if ! command -v docker >/dev/null 2>&1; then
  curl -fsSL https://get.docker.com | sh
fi
sudo usermod -aG docker "${USER}"

mkdir -p "${PROJECT_DIR}/data" "${PROJECT_DIR}/models"
chmod +x "${PROJECT_DIR}/deploy_rpi.sh" "${PROJECT_DIR}/scripts/"*.sh || true

cat >/tmp/house-security-ai.service <<EOF
[Unit]
Description=house-security-ai docker compose stack
After=network-online.target docker.service
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=${PROJECT_DIR}
ExecStart=/usr/bin/docker compose up -d --build
ExecStop=/usr/bin/docker compose down
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
EOF

sudo mv /tmp/house-security-ai.service /etc/systemd/system/house-security-ai.service
sudo systemctl daemon-reload
sudo systemctl enable house-security-ai.service
sudo systemctl restart house-security-ai.service

echo "house-security-ai installed"
