// ArcFace-only identity. No text hint fallback.

use anyhow::{Context, Result};
use axum::{extract::{Path, State}, http::StatusCode, routing::{get, post}, Json, Router};
use nuclear_eye::{ensure_parent_dir, face_embedding, SecurityConfig};
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::info;

// ── Code truth note ───────────────────────────────────────────────────
//
// face_db uses ArcFace-only biometric identity (Track O4 / Z10):
//
// POST /faces/search-by-image — canonical identity lookup
//    Input: base64 image → ArcFace sidecar → 512-dim embedding →
//    cosine similarity against stored embeddings → ranked results.
//    Requires face_embedding_service.py running at ARCFACE_URL (:5555).
//    If sidecar unreachable or score below threshold → returns empty / "unknown".
//
// POST /faces/embed  — extract and store embedding from a base64 image.
//
// POST /faces/search — REMOVED. Text hint / SQL LIKE identity is gone.
//    Returns 410 Gone. Callers must migrate to /faces/search-by-image.
//
// ── Search types ──────────────────────────────────────────────────────

fn default_limit() -> usize { 5 }

/// Request for biometric image-based search (O4).
#[derive(Debug, Deserialize)]
struct SearchByImageRequest {
    /// Base64-encoded JPEG or PNG of the face to search for.
    image_b64: String,
    /// Max results (default 5).
    #[serde(default = "default_limit")]
    limit: usize,
    /// Minimum cosine similarity to include in results (default 0.20).
    #[serde(default = "default_min_similarity")]
    min_similarity: f32,
}

fn default_min_similarity() -> f32 { 0.20 }

/// Request to extract and optionally store an embedding (O4).
#[derive(Debug, Deserialize)]
struct EmbedRequest {
    /// Base64-encoded JPEG or PNG.
    image_b64: String,
    /// When set, store the embedding for this face name in the DB.
    face_name: Option<String>,
}

/// Result from biometric image search (O4).
#[derive(Debug, Serialize)]
struct BiometricSearchResult {
    name: String,
    embedding_hint: String,
    authorized: bool,
    /// Cosine similarity to the query embedding (0.0–1.0).
    similarity: f32,
    /// True when similarity ≥ 0.28 (ArcFace same-person threshold).
    likely_match: bool,
}

#[derive(Debug, Serialize)]
struct EmbedResponse {
    ok: bool,
    /// The 512-dim ArcFace embedding vector.
    embedding: Vec<f32>,
    /// Number of dimensions (always 512 for ArcFace buffalo_l).
    dims: usize,
    /// Name of the face if stored (echo of face_name input).
    stored_for: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct FaceRecord {
    name: String,
    embedding_hint: String,
    authorized: bool,
}

#[derive(Clone)]
struct AppState {
    conn: Arc<Mutex<Connection>>,
    http: reqwest::Client,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    // ── Nuclear wrapper — resilience sidecar ────────────────────────────
    match nuclear_wrapper::wrap!(
        node_id      = "face-db",
        pg_url       = std::env::var("DATABASE_URL").unwrap_or_default(),
        signal_token = std::env::var("SIGNAL_TOKEN").unwrap_or_default()
    ) {
        Ok(nw) => {
            tracing::info!("nuclear-wrapper: armed (tamper, health, discovery)");
            std::mem::forget(nw);
        }
        Err(e) => tracing::info!("nuclear-wrapper: start failed ({e}) — running unguarded"),
    }

    let cfg = SecurityConfig::load()?;
    if cfg.face_db.auto_create {
        ensure_parent_dir(&cfg.app.face_db_path)?;
    }
    let conn = Connection::open(&cfg.app.face_db_path)
        .with_context(|| format!("open db {}", cfg.app.face_db_path))?;
    init_db(&conn)?;

    let http = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()
        .context("build HTTP client")?;

    let state = AppState { conn: Arc::new(Mutex::new(conn)), http };
    let app = Router::new()
        // Q4: Lucky7 health probe
        .route("/health", get(|| async { axum::Json(serde_json::json!({"status":"ok","service":"face_db"})) }))
        .route("/faces", get(list_faces).post(add_face))
        // Z10: text-hint search removed — ArcFace-only identity. Use /faces/search-by-image.
        .route("/faces/search", post(search_faces_removed))
        .route("/faces/embed", post(embed_face))
        .route("/faces/search-by-image", post(search_by_image))
        .route("/faces/:name", get(find_face))
        // T9: GDPR retention endpoints
        .route("/faces/purge", post(purge_stale_faces))
        .route("/faces/gdpr-export", get(gdpr_export_faces))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(&cfg.app.bind_face_db).await?;
    info!("face_db listening on {}", cfg.app.bind_face_db);
    axum::serve(listener, app).await?;
    Ok(())
}

fn init_db(conn: &Connection) -> Result<()> {
    // Base schema
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            embedding_hint TEXT NOT NULL,
            authorized INTEGER NOT NULL DEFAULT 0
        );
        -- O4: stores ArcFace 512-dim embeddings as little-endian float32 BLOBs.
        -- One row per face (linked by name FK).  Updated via POST /faces/embed.
        CREATE TABLE IF NOT EXISTS face_embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            face_name TEXT NOT NULL UNIQUE REFERENCES faces(name) ON DELETE CASCADE,
            embedding BLOB NOT NULL,
            dims INTEGER NOT NULL DEFAULT 512,
            updated_at INTEGER NOT NULL DEFAULT (strftime('%s','now'))
        );",
    )?;

    // T9: GDPR retention migration — add created_at and last_matched_at columns
    // (idempotent: ADD COLUMN is a no-op if the column already exists in SQLite 3.37+,
    //  and we swallow the "duplicate column" error on older versions)
    for sql in [
        "ALTER TABLE faces ADD COLUMN created_at INTEGER NOT NULL DEFAULT (strftime('%s','now'))",
        "ALTER TABLE faces ADD COLUMN last_matched_at INTEGER",
    ] {
        match conn.execute_batch(sql) {
            Ok(_) => {}
            Err(e) if e.to_string().contains("duplicate column") => {}
            Err(e) => tracing::warn!(error = %e, "face_db migration warning (non-fatal)"),
        }
    }
    Ok(())
}

/// T9: Default face retention period in days (30 days; override with FACE_RETENTION_DAYS env var).
const DEFAULT_RETENTION_DAYS: i64 = 30;

fn face_retention_days() -> i64 {
    std::env::var("FACE_RETENTION_DAYS")
        .ok()
        .and_then(|v| v.parse::<i64>().ok())
        .unwrap_or(DEFAULT_RETENTION_DAYS)
}

async fn list_faces(State(state): State<AppState>) -> Json<Vec<FaceRecord>> {
    let conn = state.conn.lock().await;
    let mut stmt = conn.prepare("SELECT name, embedding_hint, authorized FROM faces ORDER BY name").unwrap();
    let rows = stmt
        .query_map([], |row| {
            Ok(FaceRecord {
                name: row.get(0)?,
                embedding_hint: row.get(1)?,
                authorized: row.get::<_, i64>(2)? != 0,
            })
        })
        .unwrap();

    let items = rows.filter_map(|r| r.ok()).collect::<Vec<_>>();
    Json(items)
}

async fn find_face(Path(name): Path<String>, State(state): State<AppState>) -> Json<serde_json::Value> {
    let conn = state.conn.lock().await;
    let mut stmt = conn
        .prepare("SELECT name, embedding_hint, authorized FROM faces WHERE name = ?1")
        .unwrap();
    let mut rows = stmt.query(params![name]).unwrap();
    if let Some(row) = rows.next().unwrap() {
        Json(serde_json::json!({
            "found": true,
            "name": row.get::<_, String>(0).unwrap(),
            "embedding_hint": row.get::<_, String>(1).unwrap(),
            "authorized": row.get::<_, i64>(2).unwrap() != 0
        }))
    } else {
        Json(serde_json::json!({"found": false}))
    }
}

async fn add_face(State(state): State<AppState>, Json(input): Json<FaceRecord>) -> Json<serde_json::Value> {
    let conn = state.conn.lock().await;
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;
    let changed = conn
        .execute(
            "INSERT INTO faces(name, embedding_hint, authorized, created_at, last_matched_at)
             VALUES(?1, ?2, ?3, ?4, ?4)
             ON CONFLICT(name) DO UPDATE SET
               embedding_hint   = excluded.embedding_hint,
               authorized       = excluded.authorized,
               last_matched_at  = ?4",
            params![input.name, input.embedding_hint, i64::from(input.authorized), now],
        )
        .unwrap_or(0);
    Json(serde_json::json!({"upserted": changed > 0}))
}

// ── T9: GDPR retention endpoints ────────────────────────────────────────────

/// T9: Purge faces that have not been matched in FACE_RETENTION_DAYS days.
/// Removes both the face record and any associated embedding (via ON DELETE CASCADE).
///
/// Safe to call from a cron job or Lucky7 maintenance window.
/// Returns the number of faces deleted.
async fn purge_stale_faces(State(state): State<AppState>) -> Json<serde_json::Value> {
    let retention = face_retention_days();
    let cutoff = (std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64)
        - retention * 86_400;

    let conn = state.conn.lock().await;
    // Delete faces where last_matched_at is NULL (never matched) and older than cutoff,
    // OR where last_matched_at < cutoff.
    let deleted = conn
        .execute(
            "DELETE FROM faces WHERE (last_matched_at IS NULL AND created_at < ?1) OR last_matched_at < ?1",
            params![cutoff],
        )
        .unwrap_or(0);

    tracing::info!(deleted, retention_days = retention, "T9: purged stale face records");
    Json(serde_json::json!({
        "deleted": deleted,
        "retention_days": retention,
    }))
}

/// T9: GDPR data export — returns all face records as JSON (NO biometric embeddings).
/// Embeddings are biometric PII and must never be exported to untrusted consumers.
/// Returns metadata only: name, hint, authorized, created_at, last_matched_at.
async fn gdpr_export_faces(State(state): State<AppState>) -> Json<serde_json::Value> {
    let conn = state.conn.lock().await;
    let mut stmt = match conn.prepare(
        "SELECT name, embedding_hint, authorized, created_at, last_matched_at FROM faces ORDER BY name",
    ) {
        Ok(s) => s,
        Err(e) => return Json(serde_json::json!({"error": e.to_string()})),
    };

    let rows: Vec<serde_json::Value> = match stmt.query_map([], |row| {
        Ok(serde_json::json!({
            "name":             row.get::<_, String>(0)?,
            "embedding_hint":   row.get::<_, String>(1)?,
            "authorized":       row.get::<_, i64>(2)? != 0,
            "created_at":       row.get::<_, Option<i64>>(3)?,
            "last_matched_at":  row.get::<_, Option<i64>>(4)?,
        }))
    }) {
        Ok(mapped) => mapped.filter_map(|r| r.ok()).collect(),
        Err(e) => {
            return Json(serde_json::json!({"error": e.to_string()}));
        }
    };

    let total = rows.len();
    Json(serde_json::json!({
        "faces": rows,
        "total": total,
        "note": "Biometric embeddings excluded. Names are identity hints only — not confirmed identities.",
        "retention_days": face_retention_days(),
    }))
}

/// Z10: POST /faces/search — tombstone handler.
///
/// Text-hint / SQL-LIKE identity search has been removed (Z10).
/// All face lookups must use biometric ArcFace matching: POST /faces/search-by-image.
/// If ArcFace score is below threshold → "unknown" is returned, never a text guess.
async fn search_faces_removed() -> (StatusCode, Json<serde_json::Value>) {
    tracing::warn!("POST /faces/search called — endpoint removed (Z10). Use /faces/search-by-image.");
    (
        StatusCode::GONE,
        Json(serde_json::json!({
            "error": "Text-hint face search removed (Z10). Use POST /faces/search-by-image with ArcFace biometric matching.",
            "migrate_to": "/faces/search-by-image",
        })),
    )
}

// ── O4: Biometric endpoints ───────────────────────────────────────────────────

/// POST /faces/embed — extract ArcFace embedding from a base64 image.
///
/// If `face_name` is provided and that name exists in the `faces` table, the
/// embedding is stored in `face_embeddings` for future biometric search.
/// The 512-dim vector is always returned in the response regardless.
async fn embed_face(
    State(state): State<AppState>,
    Json(req): Json<EmbedRequest>,
) -> Json<serde_json::Value> {
    match face_embedding::embed(&state.http, &req.image_b64).await {
        Err(e) => {
            tracing::warn!(error = %e, "face_embedding sidecar unreachable or no face detected");
            Json(serde_json::json!({
                "ok": false,
                "error": e.to_string(),
                "hint": "Is face_embedding_service.py running at FACE_EMBED_URL (default :5555)?",
            }))
        }
        Ok(embedding) => {
            let dims = embedding.len();
            let stored_for: Option<String> = if let Some(ref name) = req.face_name {
                let blob = face_embedding::embedding_to_bytes(&embedding);
                let conn = state.conn.lock().await;
                match conn.execute(
                    "INSERT INTO face_embeddings(face_name, embedding, dims)
                     VALUES(?1, ?2, ?3)
                     ON CONFLICT(face_name) DO UPDATE
                     SET embedding = excluded.embedding,
                         dims = excluded.dims,
                         updated_at = strftime('%s','now')",
                    params![name, blob, dims as i64],
                ) {
                    Ok(_) => {
                        tracing::info!(face = %name, dims, "embedding stored in face_embeddings");
                        Some(name.clone())
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, face = %name, "failed to store embedding (face may not exist in faces table)");
                        None
                    }
                }
            } else {
                None
            };

            let resp = EmbedResponse { ok: true, embedding, dims, stored_for };
            Json(serde_json::to_value(resp).unwrap())
        }
    }
}

/// POST /faces/search-by-image — rank all faces by cosine similarity to query image.
///
/// Calls the ArcFace sidecar to embed the query image, then computes cosine
/// similarity against every stored embedding in `face_embeddings`.
/// Falls back to an empty result (not an error) when the sidecar is down.
async fn search_by_image(
    State(state): State<AppState>,
    Json(req): Json<SearchByImageRequest>,
) -> Json<Vec<BiometricSearchResult>> {
    // Step 1: embed query image via sidecar
    let query_embedding = match face_embedding::embed(&state.http, &req.image_b64).await {
        Ok(e) => e,
        Err(err) => {
            tracing::warn!(error = %err, "search-by-image: embedding sidecar unavailable");
            return Json(vec![]);
        }
    };

    // Step 2: load all stored embeddings + face metadata (scoped so `Statement`/`Connection`
    // are dropped before any later `.await` — rusqlite types are not `Send`).
    let rows: Vec<(String, Vec<u8>, String, bool)> = {
        let conn = state.conn.lock().await;
        let mut stmt = match conn.prepare(
            "SELECT fe.face_name, fe.embedding, f.embedding_hint, f.authorized
             FROM face_embeddings fe
             JOIN faces f ON f.name = fe.face_name
             ORDER BY fe.face_name",
        ) {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!(error = %e, "search-by-image: DB query failed");
                return Json(vec![]);
            }
        };

        let collected: Vec<(String, Vec<u8>, String, bool)> = match stmt.query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, Vec<u8>>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, i64>(3)? != 0,
            ))
        }) {
            Ok(mapped) => mapped.filter_map(|r| r.ok()).collect(),
            Err(e) => {
                tracing::warn!(error = %e, "search-by-image: failed to iterate embeddings");
                vec![]
            }
        };
        collected
    };

    // Step 3: cosine similarity ranking
    let threshold = 0.28f32; // ArcFace same-person threshold
    let mut results: Vec<BiometricSearchResult> = rows
        .into_iter()
        .filter_map(|(name, blob, hint, authorized)| {
            let stored = face_embedding::bytes_to_embedding(&blob)?;
            let sim = face_embedding::cosine_similarity(&query_embedding, &stored);
            if sim < req.min_similarity {
                return None;
            }
            Some(BiometricSearchResult {
                name,
                embedding_hint: hint,
                authorized,
                similarity: sim,
                likely_match: sim >= threshold,
            })
        })
        .collect();

    results.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(req.limit);

    // T9: Update last_matched_at for faces above the likely-match threshold (biometric retention tracking).
    // Fire-and-forget: if the update fails, the search result is still returned.
    if results.iter().any(|r| r.likely_match) {
        let conn2 = state.conn.lock().await;
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;
        for r in results.iter().filter(|r| r.likely_match) {
            let _ = conn2.execute(
                "UPDATE faces SET last_matched_at = ?1 WHERE name = ?2",
                params![now, r.name],
            );
        }
    }

    Json(results)
}
