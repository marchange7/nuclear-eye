-- face_db_002_images_watchlist.sqlite.sql
-- SQLite variant of face_db_002 for the b450 / edge single-tenant runtime
-- (default DB: /var/lib/nuclear-eye/face_db.sqlite). The Postgres companion
-- (face_db_002_images_watchlist.sql) is the multi-tenant cloud version.
--
-- WHY A SEPARATE FILE: the edge runtime uses the rusqlite FaceStore backend
-- (see src/face_store.rs); SQLite has no pgcrypto and no row-level security, so
-- the Postgres migration cannot be applied here verbatim.
--
-- DIFFERENCES from the Postgres migration (semantics preserved where possible):
--   * ENCRYPTION: image_enc holds AES-256 ciphertext encrypted IN THE APP
--     (per-tenant DEK applied in Rust before insert) — SQLite has no pgcrypto.
--     Plaintext images MUST NEVER be written here.
--   * TENANT ISOLATION: enforced in application queries (WHERE tenant_id = ?),
--     since SQLite has no RLS. Single-tenant edge installs use 'default'.
--   * COMPLIANCE (intent unchanged): biometric special-category data
--     (GDPR Art.9 / BIPA). Retention via watchlist.expires_at. Enrol / remove /
--     status changes must be audited. NO automated offender action without a
--     human review step; threat_level is advisory.
--   * TIMES: epoch-milliseconds INTEGERs (matches VisionEvent.timestamp_ms).
--   * IDEMPOTENT: CREATE ... IF NOT EXISTS — safe to re-run.

-- One or more encrypted reference images per enrolled identity.
CREATE TABLE IF NOT EXISTS face_images (
    id          TEXT    PRIMARY KEY,
    tenant_id   TEXT    NOT NULL DEFAULT 'default',
    face_id     TEXT,                       -- logical ref to faces (external id)
    image_enc   BLOB    NOT NULL,           -- app-encrypted AES-256; never plaintext
    mime        TEXT    NOT NULL,           -- 'image/jpeg' | 'image/png'
    captured_at INTEGER NOT NULL            -- epoch ms
);
CREATE INDEX IF NOT EXISTS idx_face_images_tenant  ON face_images (tenant_id);
CREATE INDEX IF NOT EXISTS idx_face_images_face_id ON face_images (face_id);

-- Watchlist registry: who is family / watch / offender, and how severe.
CREATE TABLE IF NOT EXISTS watchlist (
    id           TEXT    PRIMARY KEY,
    tenant_id    TEXT    NOT NULL DEFAULT 'default',
    face_id      TEXT,
    label        TEXT    NOT NULL,
    status       TEXT    NOT NULL CHECK (status IN ('authorized', 'watch', 'offender')),
    threat_level INTEGER NOT NULL DEFAULT 0 CHECK (threat_level >= 0 AND threat_level <= 3),
    reason       TEXT,
    added_by     TEXT,
    added_at     INTEGER NOT NULL,          -- epoch ms
    expires_at   INTEGER                    -- NULL = no expiry; else epoch ms (retention)
);
CREATE INDEX IF NOT EXISTS idx_watchlist_tenant  ON watchlist (tenant_id);
CREATE INDEX IF NOT EXISTS idx_watchlist_face_id ON watchlist (face_id);
CREATE INDEX IF NOT EXISTS idx_watchlist_status  ON watchlist (status);
CREATE INDEX IF NOT EXISTS idx_watchlist_expires ON watchlist (expires_at);
