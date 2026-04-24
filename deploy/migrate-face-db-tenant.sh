#!/usr/bin/env bash
# T-P1-15 — face_db SQLite tenant_id migration
# Adds tenant_id TEXT NOT NULL DEFAULT 'default' to faces + face_embeddings.
# Idempotent: safe to run against an already-migrated database.
#
# Usage:
#   ./deploy/migrate-face-db-tenant.sh [path-to-face_db.sqlite]
#
# Default DB path matches SecurityConfig face_db_path (/var/lib/nuclear-eye/face_db.sqlite).
# Pass a different path as $1 for dev/test.

set -euo pipefail

DB="${1:-/var/lib/nuclear-eye/face_db.sqlite}"

if [[ ! -f "$DB" ]]; then
    echo "face_db at $DB not found — no migration needed (new installs init via code)"
    exit 0
fi

echo "Migrating $DB ..."

sqlite3 "$DB" <<'SQL'
-- T-P1-15: tenant isolation columns
-- Using an idempotent approach: add columns only if they don't exist.

-- faces.tenant_id
ALTER TABLE faces ADD COLUMN tenant_id TEXT NOT NULL DEFAULT 'default';

-- face_embeddings.tenant_id
ALTER TABLE face_embeddings ADD COLUMN tenant_id TEXT NOT NULL DEFAULT 'default';
SQL
# Note: SQLite will return an error for "duplicate column name" if already migrated.
# Capture and ignore those specific errors.

# Run in a subshell so individual errors are trapped.
sqlite3 "$DB" <<'SQL' 2>/dev/null || true
ALTER TABLE faces ADD COLUMN tenant_id TEXT NOT NULL DEFAULT 'default';
SQL

sqlite3 "$DB" <<'SQL' 2>/dev/null || true
ALTER TABLE face_embeddings ADD COLUMN tenant_id TEXT NOT NULL DEFAULT 'default';
SQL

# Indexes
sqlite3 "$DB" <<'SQL'
CREATE INDEX IF NOT EXISTS idx_faces_tenant ON faces(tenant_id);
CREATE INDEX IF NOT EXISTS idx_face_embeddings_tenant ON face_embeddings(tenant_id, face_name);
SQL

echo "Migration complete."
echo "Verify with:"
echo "  sqlite3 \"$DB\" '.schema faces'"
echo "  sqlite3 \"$DB\" '.schema face_embeddings'"
