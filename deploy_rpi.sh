#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export HOUSE_SECURITY_CONFIG="${ROOT_DIR}/config/security.toml"

echo "Building all binaries..."
cargo build --release \
    --bin alarm_grader_agent \
    --bin safetyagent \
    --bin face_db \
    --bin vision_agent \
    --bin decision_agent \
    --bin safety_aurelie_agent

"${ROOT_DIR}/scripts/deploy.sh"
"${ROOT_DIR}/scripts/health_check_deploy.sh"

echo "house-security-ai deployed (including decision_agent + safety_aurelie_agent)"
