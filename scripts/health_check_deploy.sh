#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${1:-http://127.0.0.1:8080}"
curl --fail --silent "${BASE_URL}/summary" >/dev/null
curl --fail --silent "http://127.0.0.1:8081/health" >/dev/null
echo "health-check ok"
