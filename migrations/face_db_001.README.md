# face_db_001.sql — operator runbook

Source: `os/56-sentinelle-deep-rewire-plan.md` P1-4, `os/57 §3 §4.7`, `os/55` (CRITICAL/HIGH).

## What it does

1. Creates the `face_db` schema (separate from `public.*` and `riviere.*`).
2. Re-asserts the `kernel.*` tenant-policy helpers (idempotent — see
   `nuclear-platform/fortress/db/migrations/048_kernel_tenant_helpers.sql`).
3. Defines `face_db.faces` (identity metadata) and `face_db.face_embeddings`
   (pgcrypto-encrypted biometric vectors).
4. Adds two SQL helpers, `face_db.set_embedding` / `face_db.get_embedding`,
   that encrypt/decrypt 512-dim float32 vectors using a per-session symmetric
   key supplied via the `app.face_db_key` GUC.
5. Enables and **forces** RLS on both tables; isolation policies route through
   `kernel.tenant_check(tenant_id)` so the same Pass-1a / 1b / 1c rollout flag
   that gates the kernel migrations also gates this one.

## Required prior migrations

- `nuclear-platform/fortress/db/migrations/048_kernel_tenant_helpers.sql`

The migration also re-declares the kernel helpers inline (`CREATE OR REPLACE`)
so it can be applied out of order against a fresh database; the inline copy
must stay byte-identical with `048`.

## Connection contract

The face_db binary must, **per transaction**:

```sql
SET LOCAL app.tenant_id = '<uuid-from-X-Tenant-Id>';
SET LOCAL app.face_db_key = '<32-byte-passphrase-from-FACE_DB_ENCRYPTION_KEY>';
-- only when running Pass 1c:
SET LOCAL app.kernel_require_tenant_header = '1';
```

`app.face_db_key` is intentionally a session-local GUC. Never assign it at the
role level (`ALTER ROLE … SET app.face_db_key = …`) — that persists the key in
`pg_db_role_setting` on disk, defeating the purpose of pgcrypto.

## Pass 1a / 1b / 1c rollout (`os/57 §4.7`)

| Pass | `KERNEL_REQUIRE_TENANT_HEADER` | Behaviour                                                                 |
| ---- | ------------------------------- | ------------------------------------------------------------------------- |
| 1a   | `0` (default)                   | Missing tenant header → reads/writes attributed to `kernel.legacy_default_tenant()`. Existing flows keep working. |
| 1b   | `0`                             | Same as 1a; this is the burn-in window where Sentinelle gateway starts forwarding `X-Tenant-Id`. |
| 1c   | `1`                             | Strict — connections without `SET LOCAL app.tenant_id` see no rows and cannot insert. Flip after `os/58 §2` and `os/58 §3` land. |

## psql smoke test

```sql
-- 0) Provision a key for this session and a tenant scope.
SET app.face_db_key = 'unit-test-key-please-change';
SET app.tenant_id   = '11111111-1111-1111-1111-111111111111';

-- 1) Insert identity metadata.
INSERT INTO face_db.faces(name, embedding_hint, authorized)
     VALUES ('alice', 'red coat', true)
RETURNING id;
-- → returns the face_id (call it $1 below).

-- 2) Encrypt + store a 512-dim embedding (raw bytes 2048 = 512 * 4).
SELECT face_db.set_embedding(:face_id, repeat('\x00', 2048)::bytea);

-- 3) Decrypt round-trip.
SELECT octet_length(face_db.get_embedding(:face_id));   -- → 2048

-- 4) Confirm RLS scope: switch tenant, the row disappears.
SET app.tenant_id = '22222222-2222-2222-2222-222222222222';
SELECT count(*) FROM face_db.faces WHERE name = 'alice';   -- → 0
```

## Rollback

Schema-only — drop with `DROP SCHEMA face_db CASCADE`. The `kernel.*`
helpers are owned by `048` and must NOT be dropped here.

## Pairs with

- `nuclear-eye/src/face_db_auth.rs` — bearer auth + tenant header extraction
  on the HTTP surface.
- `nuclear-eye/src/face_store.rs` (Pass 2 of P1-4) — Postgres backend behind
  the `face_db_pg` Cargo feature; calls `face_db.set_embedding` /
  `face_db.get_embedding` and runs every transaction with the GUCs above.
