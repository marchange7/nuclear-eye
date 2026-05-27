-- face_db_002_images_watchlist.sql — Phase 4 "photo DB":
--   encrypted reference IMAGES + a watchlist registry for the
--   family / intruders biometric store.
--
-- Source:
--   * Sentinelle Phase 4 — store reference images (not just embeddings) plus a
--     watchlist registry that src/watchlist.rs matches against
--     (WatchStatus { Authorized, Watch, Offender }).
--   * Builds directly on face_db_001.sql (faces + face_embeddings) and reuses
--     its conventions byte-for-byte: pgcrypto column encryption, per-tenant DEK
--     via the app.face_db_key GUC, tenant_id + RLS (ENABLE + FORCE), the
--     kernel.tenant_check / kernel.default_row_tenant helpers.
--
-- ── COMPLIANCE POSTURE (read before touching this schema) ───────────────────
--   * The data here is BIOMETRIC SPECIAL-CATEGORY DATA: reference face imagery
--     and an identity watchlist. It is regulated under GDPR Art. 9 (special
--     categories) and, in the US, biometric-privacy statutes such as Illinois
--     BIPA. Collection requires a lawful basis / explicit consent for family
--     members; offenders are processed under a separate substantial-public-
--     interest / security basis.
--   * ENCRYPTED AT REST: face_images.image_enc holds pgcrypto AES-256
--     ciphertext only — plaintext image bytes NEVER touch disk. Encryption is
--     in-flight via the same per-tenant Data Encryption Key (DEK) pattern as
--     face_db.face_embeddings: the key is supplied per session through the
--     app.face_db_key GUC and is never persisted server-side.
--   * RETENTION: watchlist.expires_at drives data-minimisation. Rows past their
--     expiry must be purged by the retention job; NULL means "no auto-expiry"
--     (e.g. a permanent family enrolment) and is reviewed manually.
--   * AUDIT: every enrol / remove / status-change is expected to be written to
--     the audit train (see src/audit.rs). This migration provides the
--     schema-level isolation that audit + auth alone cannot.
--   * HUMAN-IN-THE-LOOP: an 'offender' classification is advisory. No automated
--     offender action (alarm escalation, lockout, notification to third
--     parties) may be taken without a human review step. threat_level is a
--     triage hint, not an actuator.
--
-- Idempotent: every statement uses IF NOT EXISTS / OR REPLACE / DROP-then-
-- CREATE for policies. Safe to re-apply.
-- Requires: face_db_001.sql (schema face_db, kernel.* helpers, pgcrypto).
--
-- NOTE on face_id: face_db.faces.id is bigserial (bigint) in face_db_001.sql.
-- Phase 4 keys these registry rows by a uuid `face_id` (stable external
-- identity handle) per spec; it is therefore a LOGICAL reference, not a hard
-- bigint FK, so this migration applies cleanly against the existing schema.

BEGIN;

-- ── Extensions / schema (idempotent, mirrors 001) ───────────────────────────
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE SCHEMA IF NOT EXISTS face_db;

-- ── face_images: encrypted reference image per identity ─────────────────────
--
-- One or more reference images per enrolled identity. The `image_enc` column
-- stores the raw image bytes (JPEG/PNG) AES-256-encrypted via
-- pgp_sym_encrypt_bytea(...)::bytea using the per-session app.face_db_key GUC
-- (same per-tenant DEK pattern as face_db.face_embeddings). Plaintext image
-- bytes are NEVER stored at rest — encryption happens in-flight on INSERT
-- through face_db.set_face_image below.
CREATE TABLE IF NOT EXISTS face_db.face_images (
    id          uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id   uuid        NOT NULL DEFAULT kernel.default_row_tenant(),
    face_id     uuid,
    image_enc   bytea       NOT NULL,
    mime        text        NOT NULL,
    captured_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_face_db_images_tenant
    ON face_db.face_images (tenant_id);
CREATE INDEX IF NOT EXISTS idx_face_db_images_face_id
    ON face_db.face_images (face_id);

COMMENT ON TABLE face_db.face_images IS
    'pgcrypto-encrypted reference face images (special-category biometric data, GDPR Art.9 / BIPA). image_enc bytes are AES-256 ciphertext via the per-session app.face_db_key GUC; plaintext never at rest. Encrypt/decrypt via face_db.set_face_image / face_db.get_face_image.';
COMMENT ON COLUMN face_db.face_images.image_enc IS
    'AES-256 ciphertext (pgp_sym_encrypt_bytea) of the raw image bytes. Per-tenant DEK from app.face_db_key; never plaintext at rest.';

-- ── watchlist: identity registry (what src/watchlist.rs matches against) ────
CREATE TABLE IF NOT EXISTS face_db.watchlist (
    id           uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id    uuid        NOT NULL DEFAULT kernel.default_row_tenant(),
    face_id      uuid,
    label        text        NOT NULL,
    status       text        NOT NULL CHECK (status IN ('authorized', 'watch', 'offender')),
    threat_level smallint    NOT NULL DEFAULT 0 CHECK (threat_level >= 0 AND threat_level <= 3),
    reason       text,
    added_by     text,
    added_at     timestamptz NOT NULL DEFAULT now(),
    expires_at   timestamptz
);

CREATE INDEX IF NOT EXISTS idx_face_db_watchlist_tenant
    ON face_db.watchlist (tenant_id);
CREATE INDEX IF NOT EXISTS idx_face_db_watchlist_face_id
    ON face_db.watchlist (face_id);
CREATE INDEX IF NOT EXISTS idx_face_db_watchlist_status
    ON face_db.watchlist (status);
CREATE INDEX IF NOT EXISTS idx_face_db_watchlist_expires
    ON face_db.watchlist (expires_at);

COMMENT ON TABLE face_db.watchlist IS
    'Per-tenant identity registry: status in (authorized|watch|offender) + threat_level 0..3. Retention via expires_at (NULL = no auto-expiry). offender is advisory only — human-in-the-loop required before any offender action.';

-- ── Helpers: in-flight encryption for reference images ──────────────────────
-- Mirrors face_db.set_embedding / face_db.get_embedding from 001 so plaintext
-- image bytes only ever exist inside a single SQL function call.

CREATE OR REPLACE FUNCTION face_db.set_face_image(
    p_face_id     uuid,
    p_raw         bytea,
    p_mime        text,
    p_captured_at timestamptz DEFAULT now()
) RETURNS uuid
LANGUAGE plpgsql AS $$
DECLARE
    v_key text := current_setting('app.face_db_key', true);
    v_id  uuid;
BEGIN
    IF v_key IS NULL OR v_key = '' THEN
        RAISE EXCEPTION 'face_db.set_face_image: app.face_db_key is not set on this session';
    END IF;
    INSERT INTO face_db.face_images(face_id, image_enc, mime, captured_at)
    VALUES (
        p_face_id,
        pgp_sym_encrypt_bytea(p_raw, v_key),
        p_mime,
        p_captured_at
    )
    RETURNING id INTO v_id;
    RETURN v_id;
END
$$;

CREATE OR REPLACE FUNCTION face_db.get_face_image(p_id uuid)
RETURNS bytea
LANGUAGE plpgsql STABLE AS $$
DECLARE
    v_key text := current_setting('app.face_db_key', true);
    v_enc bytea;
BEGIN
    IF v_key IS NULL OR v_key = '' THEN
        RAISE EXCEPTION 'face_db.get_face_image: app.face_db_key is not set on this session';
    END IF;
    SELECT image_enc INTO v_enc
    FROM face_db.face_images
    WHERE id = p_id;

    IF v_enc IS NULL THEN
        RETURN NULL;
    END IF;

    RETURN pgp_sym_decrypt_bytea(v_enc, v_key);
END
$$;

GRANT EXECUTE ON FUNCTION
    face_db.set_face_image(uuid, bytea, text, timestamptz),
    face_db.get_face_image(uuid)
TO PUBLIC;

-- ── RLS (ENABLE + FORCE + both policies, keyed on kernel.tenant_check) ──────

ALTER TABLE face_db.face_images ENABLE ROW LEVEL SECURITY;
ALTER TABLE face_db.face_images FORCE  ROW LEVEL SECURITY;
ALTER TABLE face_db.watchlist   ENABLE ROW LEVEL SECURITY;
ALTER TABLE face_db.watchlist   FORCE  ROW LEVEL SECURITY;

DROP POLICY IF EXISTS tenant_isolation        ON face_db.face_images;
DROP POLICY IF EXISTS tenant_isolation_insert ON face_db.face_images;
DROP POLICY IF EXISTS tenant_isolation        ON face_db.watchlist;
DROP POLICY IF EXISTS tenant_isolation_insert ON face_db.watchlist;

CREATE POLICY tenant_isolation ON face_db.face_images
    USING (kernel.tenant_check(tenant_id));
CREATE POLICY tenant_isolation_insert ON face_db.face_images
    FOR INSERT WITH CHECK (kernel.tenant_check(tenant_id));

CREATE POLICY tenant_isolation ON face_db.watchlist
    USING (kernel.tenant_check(tenant_id));
CREATE POLICY tenant_isolation_insert ON face_db.watchlist
    FOR INSERT WITH CHECK (kernel.tenant_check(tenant_id));

COMMIT;
