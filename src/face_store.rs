//! Backend abstraction for the `face_db` binary.
//!
//! Sources:
//!   * `os/56-sentinelle-deep-rewire-plan.md` P1-4 (Pass 2)
//!   * `os/55-sentinelle-cross-repo-audit.md`     CRITICAL biometric encryption
//!   * `os/57-multitenant-kernel-architecture.md` §3 §4.7
//!
//! Two backends, runtime-selected from `FACE_DB_DATABASE_URL`:
//!
//! * [`FaceStore::Sqlite`]   — legacy on-disk store (rusqlite). Always
//!   compiled. Embeddings are stored unencrypted (matches pre-P1-4 behaviour);
//!   suitable only for single-tenant lab boxes.
//! * [`FaceStore::Postgres`] — pgcrypto-encrypted biometric store with
//!   tenant_id + RLS. Compiled only when the `face_db_pg` Cargo feature is
//!   enabled. Every transaction runs `SET LOCAL app.tenant_id` and
//!   `SET LOCAL app.face_db_key` before touching `face_db.*`.
//!
//! Backend selection rule (in `face_db.rs::main`):
//!   - `FACE_DB_DATABASE_URL` set + `face_db_pg` feature on ⇒ Postgres.
//!   - Otherwise ⇒ SQLite (the binary logs the chosen backend at startup).
//!
//! Closes:
//!   * `os/55` CRITICAL — biometric embeddings stored unencrypted in SQLite,
//!     once `FACE_DB_DATABASE_URL` is wired and the SQLite path is retired.

use anyhow::{Context, Result};
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Mutex;
use uuid::Uuid;

#[cfg(feature = "face_db_pg")]
use sqlx::postgres::PgPool;

// ── Cross-backend row types ─────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaceRecord {
    pub name: String,
    pub embedding_hint: String,
    pub authorized: bool,
}

#[derive(Debug, Clone)]
pub struct EmbeddingRow {
    pub name: String,
    pub embedding_hint: String,
    pub authorized: bool,
    /// Plaintext little-endian float32 bytes (2048 bytes for ArcFace 512-dim).
    pub embedding: Vec<u8>,
}

#[derive(Debug, Clone, Serialize)]
pub struct GdprRow {
    pub name: String,
    pub embedding_hint: String,
    pub authorized: bool,
    pub created_at: Option<i64>,
    pub last_matched_at: Option<i64>,
}

// ── Backend enum ────────────────────────────────────────────────────────────

#[derive(Clone)]
pub enum FaceStore {
    Sqlite(Arc<Mutex<Connection>>),
    #[cfg(feature = "face_db_pg")]
    Postgres {
        pool: PgPool,
        /// Per-session symmetric key for `pgp_sym_encrypt_bytea`. Held on the
        /// server only as a `SET LOCAL` GUC scoped to the open transaction.
        encryption_key: Arc<String>,
        /// `KERNEL_REQUIRE_TENANT_HEADER` value at boot (Pass 1c flag).
        require_tenant_header: bool,
    },
}

impl FaceStore {
    pub fn label(&self) -> &'static str {
        match self {
            FaceStore::Sqlite(_) => "sqlite",
            #[cfg(feature = "face_db_pg")]
            FaceStore::Postgres { .. } => "postgres+pgcrypto",
        }
    }

    // ── faces ───────────────────────────────────────────────────────────────

    pub async fn list_faces(&self, tenant: Uuid) -> Result<Vec<FaceRecord>> {
        match self {
            FaceStore::Sqlite(conn) => {
                let tenant_str = tenant.to_string();
                let conn = conn.lock().await;
                let mut stmt = conn
                    .prepare("SELECT name, embedding_hint, authorized FROM faces WHERE tenant_id = ?1 ORDER BY name")
                    .context("prepare list_faces (sqlite)")?;
                let rows = stmt
                    .query_map(params![tenant_str], |row| {
                        Ok(FaceRecord {
                            name: row.get(0)?,
                            embedding_hint: row.get(1)?,
                            authorized: row.get::<_, i64>(2)? != 0,
                        })
                    })
                    .context("query list_faces (sqlite)")?;
                Ok(rows.filter_map(|r| r.ok()).collect())
            }
            #[cfg(feature = "face_db_pg")]
            FaceStore::Postgres { pool, require_tenant_header, .. } => {
                let mut tx = pool.begin().await.context("begin tx (list_faces)")?;
                set_tenant(&mut tx, tenant, *require_tenant_header).await?;
                let rows = sqlx::query_as::<_, (String, String, bool)>(
                    "SELECT name, embedding_hint, authorized
                       FROM face_db.faces
                       ORDER BY name",
                )
                .fetch_all(&mut *tx)
                .await
                .context("select faces (postgres)")?;
                tx.commit().await.context("commit (list_faces)")?;
                Ok(rows
                    .into_iter()
                    .map(|(name, hint, authorized)| FaceRecord {
                        name,
                        embedding_hint: hint,
                        authorized,
                    })
                    .collect())
            }
        }
    }

    pub async fn find_face(&self, tenant: Uuid, name: &str) -> Result<Option<FaceRecord>> {
        match self {
            FaceStore::Sqlite(conn) => {
                let tenant_str = tenant.to_string();
                let conn = conn.lock().await;
                let mut stmt = conn
                    .prepare("SELECT name, embedding_hint, authorized FROM faces WHERE name = ?1 AND tenant_id = ?2")
                    .context("prepare find_face (sqlite)")?;
                let mut rows = stmt.query(params![name, tenant_str]).context("query find_face (sqlite)")?;
                if let Some(row) = rows.next().context("row iter find_face (sqlite)")? {
                    Ok(Some(FaceRecord {
                        name: row.get(0)?,
                        embedding_hint: row.get(1)?,
                        authorized: row.get::<_, i64>(2)? != 0,
                    }))
                } else {
                    Ok(None)
                }
            }
            #[cfg(feature = "face_db_pg")]
            FaceStore::Postgres { pool, require_tenant_header, .. } => {
                let mut tx = pool.begin().await.context("begin tx (find_face)")?;
                set_tenant(&mut tx, tenant, *require_tenant_header).await?;
                let row = sqlx::query_as::<_, (String, String, bool)>(
                    "SELECT name, embedding_hint, authorized
                       FROM face_db.faces
                       WHERE name = $1",
                )
                .bind(name)
                .fetch_optional(&mut *tx)
                .await
                .context("select find_face (postgres)")?;
                tx.commit().await.context("commit (find_face)")?;
                Ok(row.map(|(name, hint, authorized)| FaceRecord {
                    name,
                    embedding_hint: hint,
                    authorized,
                }))
            }
        }
    }

    /// Upsert face metadata. Returns true on insert/update.
    pub async fn upsert_face(&self, tenant: Uuid, rec: &FaceRecord) -> Result<bool> {
        match self {
            FaceStore::Sqlite(conn) => {
                let tenant_str = tenant.to_string();
                let conn = conn.lock().await;
                let now = unix_now();
                let n = conn
                    .execute(
                        "INSERT INTO faces(tenant_id, name, embedding_hint, authorized, created_at, last_matched_at)
                         VALUES(?1, ?2, ?3, ?4, ?5, ?5)
                         ON CONFLICT(tenant_id, name) DO UPDATE SET
                           embedding_hint   = excluded.embedding_hint,
                           authorized       = excluded.authorized,
                           last_matched_at  = ?5",
                        params![tenant_str, rec.name, rec.embedding_hint, i64::from(rec.authorized), now],
                    )
                    .context("upsert face (sqlite)")?;
                Ok(n > 0)
            }
            #[cfg(feature = "face_db_pg")]
            FaceStore::Postgres { pool, require_tenant_header, .. } => {
                let mut tx = pool.begin().await.context("begin tx (upsert_face)")?;
                set_tenant(&mut tx, tenant, *require_tenant_header).await?;
                let res = sqlx::query(
                    "INSERT INTO face_db.faces(tenant_id, name, embedding_hint, authorized, last_matched_at)
                     VALUES ($1, $2, $3, $4, now())
                     ON CONFLICT (tenant_id, name) DO UPDATE
                     SET embedding_hint  = EXCLUDED.embedding_hint,
                         authorized      = EXCLUDED.authorized,
                         last_matched_at = now()",
                )
                .bind(tenant)
                .bind(&rec.name)
                .bind(&rec.embedding_hint)
                .bind(rec.authorized)
                .execute(&mut *tx)
                .await
                .context("upsert face (postgres)")?;
                tx.commit().await.context("commit (upsert_face)")?;
                Ok(res.rows_affected() > 0)
            }
        }
    }

    /// Encrypt + store an embedding for an existing face. Returns true when
    /// the face row exists (and the embedding was written), false otherwise.
    pub async fn store_embedding(
        &self,
        tenant: Uuid,
        face_name: &str,
        blob: &[u8],
        dims: usize,
    ) -> Result<bool> {
        match self {
            FaceStore::Sqlite(conn) => {
                let tenant_str = tenant.to_string();
                let conn = conn.lock().await;
                let n = conn.execute(
                    "INSERT INTO face_embeddings(tenant_id, face_name, embedding, dims)
                     VALUES(?1, ?2, ?3, ?4)
                     ON CONFLICT(tenant_id, face_name) DO UPDATE
                     SET embedding = excluded.embedding,
                         dims = excluded.dims,
                         updated_at = strftime('%s','now')",
                    params![tenant_str, face_name, blob, dims as i64],
                );
                Ok(matches!(n, Ok(x) if x > 0))
            }
            #[cfg(feature = "face_db_pg")]
            FaceStore::Postgres { pool, encryption_key, require_tenant_header } => {
                let mut tx = pool.begin().await.context("begin tx (store_embedding)")?;
                set_tenant(&mut tx, tenant, *require_tenant_header).await?;
                set_encryption_key(&mut tx, encryption_key).await?;

                let row: Option<(i64,)> = sqlx::query_as(
                    "SELECT id FROM face_db.faces WHERE name = $1",
                )
                .bind(face_name)
                .fetch_optional(&mut *tx)
                .await
                .context("select face_id (postgres)")?;

                let face_id = match row {
                    Some((id,)) => id,
                    None => {
                        tx.rollback().await.ok();
                        return Ok(false);
                    }
                };

                sqlx::query("SELECT face_db.set_embedding($1, $2, $3)")
                    .bind(face_id)
                    .bind(blob)
                    .bind(dims as i32)
                    .execute(&mut *tx)
                    .await
                    .context("face_db.set_embedding (postgres)")?;

                tx.commit().await.context("commit (store_embedding)")?;
                Ok(true)
            }
        }
    }

    /// Load every embedding visible to `tenant`, decrypting along the way for
    /// the Postgres backend.
    pub async fn load_embeddings(&self, tenant: Uuid) -> Result<Vec<EmbeddingRow>> {
        match self {
            FaceStore::Sqlite(conn) => {
                let tenant_str = tenant.to_string();
                let conn = conn.lock().await;
                let mut stmt = conn
                    .prepare(
                        "SELECT fe.face_name, fe.embedding, f.embedding_hint, f.authorized
                           FROM face_embeddings fe
                           JOIN faces f ON f.name = fe.face_name AND f.tenant_id = fe.tenant_id
                           WHERE fe.tenant_id = ?1
                           ORDER BY fe.face_name",
                    )
                    .context("prepare load_embeddings (sqlite)")?;
                let rows = stmt
                    .query_map(params![tenant_str], |row| {
                        Ok(EmbeddingRow {
                            name: row.get(0)?,
                            embedding: row.get(1)?,
                            embedding_hint: row.get(2)?,
                            authorized: row.get::<_, i64>(3)? != 0,
                        })
                    })
                    .context("query load_embeddings (sqlite)")?;
                Ok(rows.filter_map(|r| r.ok()).collect())
            }
            #[cfg(feature = "face_db_pg")]
            FaceStore::Postgres { pool, encryption_key, require_tenant_header } => {
                let mut tx = pool.begin().await.context("begin tx (load_embeddings)")?;
                set_tenant(&mut tx, tenant, *require_tenant_header).await?;
                set_encryption_key(&mut tx, encryption_key).await?;

                let rows = sqlx::query_as::<_, (String, Vec<u8>, String, bool)>(
                    "SELECT f.name,
                            face_db.get_embedding(fe.face_id),
                            f.embedding_hint,
                            f.authorized
                       FROM face_db.face_embeddings fe
                       JOIN face_db.faces f ON f.id = fe.face_id
                       ORDER BY f.name",
                )
                .fetch_all(&mut *tx)
                .await
                .context("select+decrypt embeddings (postgres)")?;

                tx.commit().await.context("commit (load_embeddings)")?;
                Ok(rows
                    .into_iter()
                    .map(|(name, embedding, hint, authorized)| EmbeddingRow {
                        name,
                        embedding,
                        embedding_hint: hint,
                        authorized,
                    })
                    .collect())
            }
        }
    }

    pub async fn touch_last_matched(
        &self,
        tenant: Uuid,
        names: &[String],
        now_secs: i64,
    ) -> Result<()> {
        if names.is_empty() {
            return Ok(());
        }
        match self {
            FaceStore::Sqlite(conn) => {
                let tenant_str = tenant.to_string();
                let conn = conn.lock().await;
                for name in names {
                    let _ = conn.execute(
                        "UPDATE faces SET last_matched_at = ?1 WHERE name = ?2 AND tenant_id = ?3",
                        params![now_secs, name, tenant_str],
                    );
                }
                Ok(())
            }
            #[cfg(feature = "face_db_pg")]
            FaceStore::Postgres { pool, require_tenant_header, .. } => {
                let mut tx = pool.begin().await.context("begin tx (touch_last_matched)")?;
                set_tenant(&mut tx, tenant, *require_tenant_header).await?;
                for name in names {
                    sqlx::query(
                        "UPDATE face_db.faces SET last_matched_at = now() WHERE name = $1",
                    )
                    .bind(name)
                    .execute(&mut *tx)
                    .await
                    .context("update last_matched_at (postgres)")?;
                }
                tx.commit().await.context("commit (touch_last_matched)")?;
                Ok(())
            }
        }
    }

    /// Delete face rows whose `last_matched_at` (or `created_at` when never
    /// matched) is older than `cutoff_secs`. Returns rows deleted.
    pub async fn purge_stale(&self, tenant: Uuid, cutoff_secs: i64) -> Result<u64> {
        match self {
            FaceStore::Sqlite(conn) => {
                let tenant_str = tenant.to_string();
                let conn = conn.lock().await;
                let n = conn
                    .execute(
                        "DELETE FROM faces
                          WHERE tenant_id = ?2
                            AND ((last_matched_at IS NULL AND created_at < ?1)
                             OR last_matched_at < ?1)",
                        params![cutoff_secs, tenant_str],
                    )
                    .context("purge_stale (sqlite)")?;
                Ok(n as u64)
            }
            #[cfg(feature = "face_db_pg")]
            FaceStore::Postgres { pool, require_tenant_header, .. } => {
                let mut tx = pool.begin().await.context("begin tx (purge_stale)")?;
                set_tenant(&mut tx, tenant, *require_tenant_header).await?;
                let cutoff_ts = sqlx::types::time::OffsetDateTime::from_unix_timestamp(cutoff_secs)
                    .unwrap_or(sqlx::types::time::OffsetDateTime::UNIX_EPOCH);
                let res = sqlx::query(
                    "DELETE FROM face_db.faces
                      WHERE (last_matched_at IS NULL AND created_at < $1)
                         OR last_matched_at < $1",
                )
                .bind(cutoff_ts)
                .execute(&mut *tx)
                .await
                .context("purge_stale (postgres)")?;
                tx.commit().await.context("commit (purge_stale)")?;
                Ok(res.rows_affected())
            }
        }
    }

    pub async fn gdpr_export(&self, tenant: Uuid) -> Result<Vec<GdprRow>> {
        match self {
            FaceStore::Sqlite(conn) => {
                let tenant_str = tenant.to_string();
                let conn = conn.lock().await;
                let mut stmt = conn
                    .prepare(
                        "SELECT name, embedding_hint, authorized, created_at, last_matched_at
                           FROM faces WHERE tenant_id = ?1 ORDER BY name",
                    )
                    .context("prepare gdpr_export (sqlite)")?;
                let rows = stmt
                    .query_map(params![tenant_str], |row| {
                        Ok(GdprRow {
                            name: row.get(0)?,
                            embedding_hint: row.get(1)?,
                            authorized: row.get::<_, i64>(2)? != 0,
                            created_at: row.get::<_, Option<i64>>(3)?,
                            last_matched_at: row.get::<_, Option<i64>>(4)?,
                        })
                    })
                    .context("query gdpr_export (sqlite)")?;
                Ok(rows.filter_map(|r| r.ok()).collect())
            }
            #[cfg(feature = "face_db_pg")]
            FaceStore::Postgres { pool, require_tenant_header, .. } => {
                let mut tx = pool.begin().await.context("begin tx (gdpr_export)")?;
                set_tenant(&mut tx, tenant, *require_tenant_header).await?;
                let rows = sqlx::query_as::<_, (
                    String,
                    String,
                    bool,
                    Option<sqlx::types::time::OffsetDateTime>,
                    Option<sqlx::types::time::OffsetDateTime>,
                )>(
                    "SELECT name, embedding_hint, authorized, created_at, last_matched_at
                       FROM face_db.faces ORDER BY name",
                )
                .fetch_all(&mut *tx)
                .await
                .context("query gdpr_export (postgres)")?;
                tx.commit().await.context("commit (gdpr_export)")?;
                Ok(rows
                    .into_iter()
                    .map(|(name, hint, authorized, created, matched)| GdprRow {
                        name,
                        embedding_hint: hint,
                        authorized,
                        created_at: created.map(|t| t.unix_timestamp()),
                        last_matched_at: matched.map(|t| t.unix_timestamp()),
                    })
                    .collect())
            }
        }
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────────

fn unix_now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

#[cfg(feature = "face_db_pg")]
async fn set_tenant(
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    tenant: Uuid,
    require_tenant_header: bool,
) -> Result<()> {
    sqlx::query("SELECT set_config('app.tenant_id', $1, true)")
        .bind(tenant.to_string())
        .execute(&mut **tx)
        .await
        .context("SET LOCAL app.tenant_id")?;
    let flag = if require_tenant_header { "1" } else { "0" };
    sqlx::query("SELECT set_config('app.kernel_require_tenant_header', $1, true)")
        .bind(flag)
        .execute(&mut **tx)
        .await
        .context("SET LOCAL app.kernel_require_tenant_header")?;
    Ok(())
}

#[cfg(feature = "face_db_pg")]
async fn set_encryption_key(
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    key: &str,
) -> Result<()> {
    sqlx::query("SELECT set_config('app.face_db_key', $1, true)")
        .bind(key)
        .execute(&mut **tx)
        .await
        .context("SET LOCAL app.face_db_key")?;
    Ok(())
}

// ── Construction helpers ────────────────────────────────────────────────────

impl FaceStore {
    pub fn from_sqlite(conn: Connection) -> Self {
        FaceStore::Sqlite(Arc::new(Mutex::new(conn)))
    }

    /// Construct a Postgres-backed store. Pre-condition: `face_db_001.sql` has
    /// been applied against `database_url`. Pre-condition: `encryption_key`
    /// is at least 16 bytes (recommended 32+).
    #[cfg(feature = "face_db_pg")]
    pub async fn connect_postgres(
        database_url: &str,
        encryption_key: String,
        require_tenant_header: bool,
    ) -> Result<Self> {
        if encryption_key.len() < 16 {
            anyhow::bail!(
                "FACE_DB_ENCRYPTION_KEY must be at least 16 bytes (got {})",
                encryption_key.len()
            );
        }
        let pool = sqlx::postgres::PgPoolOptions::new()
            .max_connections(8)
            .acquire_timeout(std::time::Duration::from_secs(5))
            .connect(database_url)
            .await
            .context("connect to face_db postgres")?;
        Ok(FaceStore::Postgres {
            pool,
            encryption_key: Arc::new(encryption_key),
            require_tenant_header,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn label_reports_active_backend() {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE faces(name TEXT, embedding_hint TEXT, authorized INTEGER, tenant_id TEXT NOT NULL DEFAULT 'default');
             CREATE TABLE face_embeddings(face_name TEXT, embedding BLOB, dims INTEGER, tenant_id TEXT NOT NULL DEFAULT 'default');",
        )
        .unwrap();
        let s = FaceStore::from_sqlite(conn);
        assert_eq!(s.label(), "sqlite");
    }

    #[tokio::test]
    async fn sqlite_round_trip_upsert_and_list() {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE faces(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tenant_id TEXT NOT NULL DEFAULT 'default',
                name TEXT NOT NULL,
                embedding_hint TEXT NOT NULL,
                authorized INTEGER NOT NULL DEFAULT 0,
                created_at INTEGER NOT NULL DEFAULT (strftime('%s','now')),
                last_matched_at INTEGER,
                UNIQUE(tenant_id, name)
             );
             CREATE TABLE face_embeddings(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tenant_id TEXT NOT NULL DEFAULT 'default',
                face_name TEXT NOT NULL,
                embedding BLOB NOT NULL,
                dims INTEGER NOT NULL DEFAULT 512,
                updated_at INTEGER NOT NULL DEFAULT (strftime('%s','now')),
                UNIQUE(tenant_id, face_name)
             );",
        )
        .unwrap();
        let s = FaceStore::from_sqlite(conn);
        let tenant = Uuid::nil();

        assert!(s
            .upsert_face(
                tenant,
                &FaceRecord {
                    name: "alice".into(),
                    embedding_hint: "red coat".into(),
                    authorized: true,
                },
            )
            .await
            .unwrap());

        let listed = s.list_faces(tenant).await.unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].name, "alice");

        let found = s.find_face(tenant, "alice").await.unwrap();
        assert_eq!(found.unwrap().embedding_hint, "red coat");

        let blob = vec![0u8; 2048];
        assert!(s.store_embedding(tenant, "alice", &blob, 512).await.unwrap());

        let embeddings = s.load_embeddings(tenant).await.unwrap();
        assert_eq!(embeddings.len(), 1);
        assert_eq!(embeddings[0].embedding.len(), 2048);
    }
}
