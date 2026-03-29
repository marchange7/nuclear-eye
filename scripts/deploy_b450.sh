#!/usr/bin/env bash
# deploy_b450.sh — Deploy nuclear-eye from M4 to b450
#
# Run from M4: bash scripts/deploy_b450.sh [--start]
#
# Steps:
#   1. rsync source (no target/, no .git/) to b450:/data/git/nuclear-eye
#   2. SSH → sudo bash deploy/install.sh  (cargo build --release + systemd)
#   3. [--start] Start all services
#
# Prerequisites:
#   ssh alias b450 = crew@192.168.2.23 (in ~/.ssh/config)

set -euo pipefail

B450_HOST="${B450_HOST:-b450}"
REMOTE_DIR="${REMOTE_DIR:-/data/git/nuclear-eye}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
START_SERVICES="${1:-}"

echo "🚀 nuclear-eye → b450 deploy"
echo "   src  : $ROOT_DIR"
echo "   dst  : $B450_HOST:$REMOTE_DIR"
echo ""

# ── 1. Rsync source ───────────────────────────────────────────────────────────
echo "📤 Syncing source to $B450_HOST..."
rsync -az --delete \
  --exclude target/ \
  --exclude ".git/" \
  --exclude "*.log" \
  --exclude "data/" \
  --exclude "__pycache__/" \
  --exclude "*.pyc" \
  "$ROOT_DIR/" "$B450_HOST:$REMOTE_DIR/"
echo "   ✓ rsync done"

# ── 2. Install (build + systemd units) ───────────────────────────────────────
echo ""
echo "🔨 Running install on $B450_HOST (this builds all binaries)..."
ssh "$B450_HOST" "cd '$REMOTE_DIR' && sudo bash deploy/install.sh"
echo "   ✓ install done"

# ── 3. Start services (optional) ─────────────────────────────────────────────
if [[ "$START_SERVICES" == "--start" ]]; then
  echo ""
  echo "▶  Starting services on $B450_HOST..."
  ssh "$B450_HOST" "
    sudo systemctl start nuclear-eye-face-db nuclear-eye-alarm-grader
    sudo systemctl start nuclear-eye-iphone-sensor nuclear-eye-safety-aurelie \
      nuclear-eye-safetyagent nuclear-eye-decision-agent \
      nuclear-eye-actuator-agent nuclear-eye-vision-agent
  "
  echo "   ✓ services started"
  echo ""
  echo "📋 Status:"
  ssh "$B450_HOST" "sudo systemctl status nuclear-eye-alarm-grader --no-pager -l | head -20"
else
  echo ""
  echo "✅ Deploy complete. To start services:"
  echo "   bash scripts/deploy_b450.sh --start"
  echo "   — or on b450 directly:"
  echo "   sudo systemctl start nuclear-eye-face-db nuclear-eye-alarm-grader"
  echo "   sudo systemctl start nuclear-eye-{iphone-sensor,safety-aurelie,safetyagent,decision-agent,actuator-agent,vision-agent}"
fi
