-- face_db_001.sql — biometric store on Postgres with pgcrypto column encryption,
-- tenant_id + RLS, and the `face_db` schema separated from public/riviere.
--
-- Source:
--   * os/56 P1-4 (Sentinelle deep rewire — face_db SQLite→Postgres)
--   * os/57 §3, §4.7  (multi-tenant kernel + Pass 1a/1b/1c rollout)
--   * os/55           (CRITICAL biometric encryption + HIGH face_db auth findings)
--
-- Closes:
--   * os/55 CRITICAL — biometric embeddings stored unencrypted in SQLite.
--   * os/55 HIGH     — face_db endpoints accept unauthenticated writes.
--                      (Auth is enforced in the binary; this migration provides the
--                       schema-level isolation that auth alone cannot.)
--
-- Idempotent: every statement uses IF NOT EXISTS / OR REPLACE / DO blocks.
-- Requires:  048_kernel_tenant_helpers.sql in fortress-pg train (defines
--            kernel.tenant_check / kernel.default_row_tenant).
--
-- Operator: see nuclear-eye/migrations/face_db_001.README.md for runbook.

BEGIN;

-- ── Extensions ───────────────────────────────────────────────────────────────
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- ── Schema ───────────────────────────────────────────────────────────────────
CREATE SCHEMA IF NOT EXISTS face_db;
COMMENT ON SCHEMA face_db IS
    'Biometric face store (ArcFace 512-dim embeddings). pgcrypto-encrypted, tenant-scoped via RLS.';

-- ── kernel.* helpers (re-declared idempotent for self-contained apply) ───────
-- Same definitions as nuclear-platform/fortress/db/migrations/048_kernel_tenant_helpers.sql.
-- Keep these byte-identical with that migration. CREATE OR REPLACE makes the
-- duplication safe.
CREATE SCHEMA IF NOT EXISTS kernel;

CREATE OR REPLACE FUNCTION kernel.legacy_default_tenant()
RETURNS uuid
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS
$$ SELECT '00000000-0000-0000-0000-000000000000'::uuid $$;

CREATE OR REPLACE FUNCTION kernel.require_tenant_header()
RETURNS boolean
LANGUAGE sql STABLE PARALLEL SAFE AS
$$
    SELECT current_setting('app.kernel_require_tenant_header', true) = '1'
$$;

CREATE OR REPLACE FUNCTION kernel.session_tenant()
RETURNS uuid
LANGUAGE plpgsql STABLE PARALLEL SAFE AS $$
DECLARE
    raw text := current_setting('app.tenant_id', true);
BEGIN
    IF raw IS NULL OR raw = '' THEN
        RETURN NULL;
    END IF;
    BEGIN
        RETURN raw::uuid;
    EXCEPTION WHEN invalid_text_representation THEN
        RETURN NULL;
    END;
END
$$;

CREATE OR REPLACE FUNCTION kernel.tenant_check(row_tenant uuid)
RETURNS boolean
LANGUAGE sql STABLE PARALLEL SAFE AS $$
    SELECT
        CASE
            WHEN kernel.require_tenant_header() THEN
                kernel.session_tenant() IS NOT NULL
                AND row_tenant = kernel.session_tenant()
            ELSE
                kernel.session_tenant() IS NULL
                OR kernel.session_tenant() = kernel.legacy_default_tenant()
                OR row_tenant = kernel.session_tenant()
        END
$$;

CREATE OR REPLACE FUNCTION kernel.default_row_tenant()
RETURNS uuid
LANGUAGE sql STABLE PARALLEL SAFE AS $$
    SELECT COALESCE(kernel.session_tenant(), kernel.legacy_default_tenant())
$$;

GRANT EXECUTE ON FUNCTION
    kernel.legacy_default_tenant(),
    kernel.require_tenant_header(),
    kernel.session_tenant(),
    kernel.tenant_check(uuid),
    kernel.default_row_tenant()
TO PUBLIC;

-- ── Tables ───────────────────────────────────────────────────────────────────

-- faces: identity metadata. NOT biometric — embeddings live in face_embeddings.
CREATE TABLE IF NOT EXISTS face_db.faces (
    id              bigserial PRIMARY KEY,
    tenant_id       uuid        NOT NULL DEFAULT kernel.default_row_tenant(),
    name            text        NOT NULL,
    embedding_hint  text        NOT NULL,
    authorized      boolean     NOT NULL DEFAULT false,
    created_at      timestamptz NOT NULL DEFAULT now(),
    last_matched_at timestamptz,
    UNIQUE (tenant_id, name)
);

CREATE INDEX IF NOT EXISTS idx_face_db_faces_tenant
    ON face_db.faces (tenant_id);
CREATE INDEX IF NOT EXISTS idx_face_db_faces_last_matched
    ON face_db.faces (last_matched_at);

COMMENT ON TABLE face_db.faces IS
    'Per-tenant face identity metadata. Biometric embeddings stored separately in face_db.face_embeddings.';

-- face_embeddings: pgcrypto-encrypted biometric vectors.
--
-- The `embedding_enc` column stores the 512-dim float32 vector serialized as a
-- little-endian byte string (2048 bytes raw) AES-256-encrypted via
-- `pgp_sym_encrypt(...)::bytea`. The encryption key is provided per-session
-- through the `app.face_db_key` GUC (set by the binary on connection check-out
-- and never persisted server-side).
--
-- Plaintext embeddings NEVER touch disk — encryption happens in-flight on
-- INSERT via a wrapper function (see fn_face_db_set_embedding below).
CREATE TABLE IF NOT EXISTS face_db.face_embeddings (
    id            bigserial   PRIMARY KEY,
    tenant_id     uuid        NOT NULL DEFAULT kernel.default_row_tenant(),
    face_id       bigint      NOT NULL REFERENCES face_db.faces(id) ON DELETE CASCADE,
    embedding_enc bytea       NOT NULL,
    dims          integer     NOT NULL DEFAULT 512,
    updated_at    timestamptz NOT NULL DEFAULT now(),
    UNIQUE (tenant_id, face_id)
);

CREATE INDEX IF NOT EXISTS idx_face_db_emb_tenant
    ON face_db.face_embeddings (tenant_id);
CREATE INDEX IF NOT EXISTS idx_face_db_emb_face_id
    ON face_db.face_embeddings (face_id);

COMMENT ON TABLE face_db.face_embeddings IS
    'pgcrypto-encrypted ArcFace embeddings. Encrypt/decrypt via face_db.set_embedding / face_db.get_embedding using the per-session app.face_db_key GUC.';

-- ── Helpers ──────────────────────────────────────────────────────────────────
-- Wrapper functions guarantee that plaintext embeddings only ever exist inside
-- a single SQL function call, never as a stored value. The binary calls
-- face_db.set_embedding(face_id, raw_bytea) — pgcrypto encrypts before INSERT.

CREATE OR REPLACE FUNCTION face_db.set_embedding(
    p_face_id bigint,
    p_raw     bytea,
    p_dims    integer DEFAULT 512
) RETURNS void
LANGUAGE plpgsql AS $$
DECLARE
    v_key text := current_setting('app.face_db_key', true);
BEGIN
    IF v_key IS NULL OR v_key = '' THEN
        RAISE EXCEPTION 'face_db.set_embedding: app.face_db_key is not set on this session';
    END IF;
    INSERT INTO face_db.face_embeddings(face_id, embedding_enc, dims)
    VALUES (
        p_face_id,
        pgp_sym_encrypt_bytea(p_raw, v_key),
        p_dims
    )
    ON CONFLICT (tenant_id, face_id) DO UPDATE
    SET embedding_enc = excluded.embedding_enc,
        dims          = excluded.dims,
        updated_at    = now();
END
$$;

CREATE OR REPLACE FUNCTION face_db.get_embedding(p_face_id bigint)
RETURNS bytea
LANGUAGE plpgsql STABLE AS $$
DECLARE
    v_key text := current_setting('app.face_db_key', true);
    v_enc bytea;
BEGIN
    IF v_key IS NULL OR v_key = '' THEN
        RAISE EXCEPTION 'face_db.get_embedding: app.face_db_key is not set on this session';
    END IF;
    SELECT embedding_enc INTO v_enc
    FROM face_db.face_embeddings
    WHERE face_id = p_face_id;

    IF v_enc IS NULL THEN
        RETURN NULL;
    END IF;

    RETURN pgp_sym_decrypt_bytea(v_enc, v_key);
END
$$;

GRANT EXECUTE ON FUNCTION
    face_db.set_embedding(bigint, bytea, integer),
    face_db.get_embedding(bigint)
TO PUBLIC;

-- ── RLS ──────────────────────────────────────────────────────────────────────

ALTER TABLE face_db.faces            ENABLE ROW LEVEL SECURITY;
ALTER TABLE face_db.faces            FORCE  ROW LEVEL SECURITY;
ALTER TABLE face_db.face_embeddings  ENABLE ROW LEVEL SECURITY;
ALTER TABLE face_db.face_embeddings  FORCE  ROW LEVEL SECURITY;

DROP POLICY IF EXISTS tenant_isolation        ON face_db.faces;
DROP POLICY IF EXISTS tenant_isolation_insert ON face_db.faces;
DROP POLICY IF EXISTS tenant_isolation        ON face_db.face_embeddings;
DROP POLICY IF EXISTS tenant_isolation_insert ON face_db.face_embeddings;

CREATE POLICY tenant_isolation ON face_db.faces
    USING (kernel.tenant_check(tenant_id));
CREATE POLICY tenant_isolation_insert ON face_db.faces
    FOR INSERT WITH CHECK (kernel.tenant_check(tenant_id));

CREATE POLICY tenant_isolation ON face_db.face_embeddings
    USING (kernel.tenant_check(tenant_id));
CREATE POLICY tenant_isolation_insert ON face_db.face_embeddings
    FOR INSERT WITH CHECK (kernel.tenant_check(tenant_id));

COMMIT;
