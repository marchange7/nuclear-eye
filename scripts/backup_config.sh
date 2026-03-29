#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKUP_DIR="${ROOT_DIR}/data/backups"
STAMP="$(date +%Y%m%d-%H%M%S)"
mkdir -p "${BACKUP_DIR}"

tar -czf "${BACKUP_DIR}/house-security-ai-${STAMP}.tar.gz"   -C "${ROOT_DIR}" config data/face_db.sqlite

echo "${BACKUP_DIR}/house-security-ai-${STAMP}.tar.gz"
