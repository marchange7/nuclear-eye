#!/usr/bin/env bash
# Source-of-truth validation script.
# Run this script on b450 to validate compile/test/smoke for nuclear-eye.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

bold='\033[1m'
green='\033[32m'
yellow='\033[33m'
red='\033[31m'
reset='\033[0m'

echo -e "${bold}nuclear-eye b450 source validation${reset}"
echo "host: $(hostname)"
echo "uname: $(uname -a)"
echo ""

if [[ "$(uname -s)" == "Darwin" ]]; then
    echo -e "${yellow}warning:${reset} this host is macOS, not b450 Linux source host."
    echo "Run this script again on b450 for source-of-truth validation."
    echo ""
fi

echo -n "[1/4] cargo check... "
cargo check --quiet
echo -e "${green}ok${reset}"

echo -n "[2/4] cargo test... "
cargo test --quiet
echo -e "${green}ok${reset}"

echo -n "[3/4] python syntax check (main.py)... "
python3 -m py_compile main.py
echo -e "${green}ok${reset}"

echo -n "[4/4] runtime smoke (/summary)... "
if curl -fsS --max-time 5 "http://127.0.0.1:8780/summary" >/dev/null 2>&1; then
    echo -e "${green}online${reset}"
    ./scripts/smoke_topology.sh
else
    echo -e "${yellow}skipped${reset} (alarm_grader not running on :8780)"
fi

echo ""
echo -e "${green}${bold}validation complete${reset}"
