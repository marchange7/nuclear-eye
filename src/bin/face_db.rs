// ArcFace-only identity. No text hint fallback.
//
// os/56 P1-4 Pass 2 — backend dispatched through `nuclear_eye::face_store::FaceStore`:
//   * `FACE_DB_DATABASE_URL` set + binary built `--features face_db_pg`
//     ⇒ Postgres backend with pgcrypto column encryption + tenant_id + RLS.
//   * Otherwise ⇒ legacy SQLite backend on `cfg.app.face_db_path`.
//
// The HTTP surface is identical across backends; auth/tenant guard is wired in
// `nuclear_eye::face_db_auth` (Pass 1).

use anyhow::{Context, Result};
use axum::{
    extract::{Path, State},
    http::{HeaderMap, StatusCode},
    routing::{get, post},
    Json, Router,
};
use nuclear_eye::{
    ensure_parent_dir, face_db_auth, face_embedding,
    face_store::{FaceRecord, FaceStore},
    SecurityConfig,
};
use rusqlite::Connection;
use serde::Deserialize;
use serde::Serialize;
use std::sync::Arc;
use tracing::{info, warn};
use uuid::Uuid;

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

#[derive(Clone)]
struct AppState {
    store: FaceStore,
    http: reqwest::Client,
    /// `FACE_DB_TOKEN` at boot. `None` ⇒ open mode (legacy parity with
    /// `alarm_grader_agent`). Logged at startup.
    face_db_token: Option<Arc<String>>,
    /// `KERNEL_REQUIRE_TENANT_HEADER` at boot. `false` ⇒ Pass 1a/1b semantics
    /// (missing X-Tenant-Id resolves to `kernel.legacy_default_tenant()`).
    require_tenant_header: bool,
}

/// Apply the standard auth + tenant guard, returning the resolved
/// [`face_db_auth::AuthContext`] or a ready-to-emit error response.
fn guard(
    state: &AppState,
    headers: &HeaderMap,
) -> Result<face_db_auth::AuthContext, (StatusCode, Json<serde_json::Value>)> {
    face_db_auth::authenticate(
        headers,
        state.face_db_token.as_deref().map(String::as_str),
        state.require_tenant_header,
    )
    .map_err(face_db_auth::AuthError::into_response)
}

/// Convenience: log the resolved tenant on a structured field (`tenant_id`
/// rather than the raw UUID embedded in another span field) so operators can
/// aggregate face_db traffic per tenant during the Pass-1b burn-in window.
#[inline]
fn audit(action: &'static str, tenant: Uuid) {
    info!(action, tenant_id = %tenant, "face_db authenticated");
}

/// Convert anyhow::Error into the standard JSON 500 envelope.
fn internal(err: anyhow::Error) -> (StatusCode, Json<serde_json::Value>) {
    warn!(error = %err, "face_db internal error");
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(serde_json::json!({"error": err.to_string()})),
    )
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    // S-7: fail-closed wrapper probe
    nuclear_eye::wrapper_guard::check_wrapper("face-db").await?;

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

    // os/56 P1-4 — wire bearer auth + multi-tenant header guard.
    let face_db_token = face_db_auth::token_from_env().map(Arc::new);
    let require_tenant_header = face_db_auth::require_tenant_from_env();
    if face_db_token.is_none() {
        warn!(
            "FACE_DB_TOKEN is not set — POST /faces, /faces/embed, /faces/purge \
             accept unauthenticated requests; set the token in production"
        );
    }
    if !require_tenant_header {
        info!(
            "KERNEL_REQUIRE_TENANT_HEADER unset — Pass 1a/1b semantics (missing \
             X-Tenant-Id resolves to legacy default tenant)"
        );
    }

    // os/56 P1-4 Pass 2 — backend selection.
    let store = build_store(&cfg, require_tenant_header).await?;
    info!(backend = store.label(), "face_db backend selected");

    let http = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()
        .context("build HTTP client")?;

    let state = AppState { store, http, face_db_token, require_tenant_header };
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

/// Pick the active backend. Postgres requires both `FACE_DB_DATABASE_URL` *and*
/// the `face_db_pg` Cargo feature. When the URL is set but the feature is off,
/// we fall back to SQLite and log a loud WARN — operators usually want to know.
async fn build_store(cfg: &SecurityConfig, require_tenant_header: bool) -> Result<FaceStore> {
    let pg_url = std::env::var("FACE_DB_DATABASE_URL").ok().filter(|s| !s.is_empty());

    #[cfg(feature = "face_db_pg")]
    {
        if let Some(url) = pg_url {
            let key = std::env::var("FACE_DB_ENCRYPTION_KEY")
                .context("FACE_DB_ENCRYPTION_KEY required when FACE_DB_DATABASE_URL is set")?;
            return FaceStore::connect_postgres(&url, key, require_tenant_header).await;
        }
    }
    #[cfg(not(feature = "face_db_pg"))]
    {
        if pg_url.is_some() {
            warn!(
                "FACE_DB_DATABASE_URL is set but the binary was built without the \
                 `face_db_pg` Cargo feature — falling back to SQLite (biometric \
                 embeddings will NOT be encrypted at rest). Rebuild with \
                 `--features face_db_pg` to enable the Postgres backend."
            );
        }
        let _ = require_tenant_header; // unused without face_db_pg
    }

    if cfg.face_db.auto_create {
        ensure_parent_dir(&cfg.app.face_db_path)?;
    }
    let conn = Connection::open(&cfg.app.face_db_path)
        .with_context(|| format!("open db {}", cfg.app.face_db_path))?;
    init_sqlite(&conn)?;
    Ok(FaceStore::from_sqlite(conn))
}

// ── Handlers ────────────────────────────────────────────────────────────────

async fn list_faces(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<Vec<FaceRecord>>, (StatusCode, Json<serde_json::Value>)> {
    let ctx = guard(&state, &headers)?;
    audit("list_faces", ctx.tenant_id);
    state.store.list_faces(ctx.tenant_id).await.map(Json).map_err(internal)
}

async fn find_face(
    Path(name): Path<String>,
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let ctx = guard(&state, &headers)?;
    audit("find_face", ctx.tenant_id);
    let found = state.store.find_face(ctx.tenant_id, &name).await.map_err(internal)?;
    Ok(Json(match found {
        Some(rec) => serde_json::json!({
            "found": true,
            "name": rec.name,
            "embedding_hint": rec.embedding_hint,
            "authorized": rec.authorized
        }),
        None => serde_json::json!({"found": false}),
    }))
}

async fn add_face(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(input): Json<FaceRecord>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let ctx = guard(&state, &headers)?;
    audit("add_face", ctx.tenant_id);
    let upserted = state.store.upsert_face(ctx.tenant_id, &input).await.map_err(internal)?;
    Ok(Json(serde_json::json!({"upserted": upserted})))
}

// ── T9: GDPR retention endpoints ────────────────────────────────────────────

/// T9: Default face retention period in days (30 days; override with FACE_RETENTION_DAYS env var).
const DEFAULT_RETENTION_DAYS: i64 = 30;

fn face_retention_days() -> i64 {
    std::env::var("FACE_RETENTION_DAYS")
        .ok()
        .and_then(|v| v.parse::<i64>().ok())
        .unwrap_or(DEFAULT_RETENTION_DAYS)
}

/// T9: Purge faces that have not been matched in FACE_RETENTION_DAYS days.
async fn purge_stale_faces(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let ctx = guard(&state, &headers)?;
    audit("purge_stale_faces", ctx.tenant_id);
    let retention = face_retention_days();
    let cutoff = unix_now_secs() - retention * 86_400;
    let deleted = state.store.purge_stale(ctx.tenant_id, cutoff).await.map_err(internal)?;
    tracing::info!(deleted, retention_days = retention, "T9: purged stale face records");
    Ok(Json(serde_json::json!({
        "deleted": deleted,
        "retention_days": retention,
    })))
}

/// T9: GDPR data export — returns all face records as JSON (NO biometric embeddings).
async fn gdpr_export_faces(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let ctx = guard(&state, &headers)?;
    audit("gdpr_export_faces", ctx.tenant_id);
    let rows = state.store.gdpr_export(ctx.tenant_id).await.map_err(internal)?;
    let total = rows.len();
    Ok(Json(serde_json::json!({
        "faces": rows,
        "total": total,
        "note": "Biometric embeddings excluded. Names are identity hints only — not confirmed identities.",
        "retention_days": face_retention_days(),
    })))
}

/// Z10: POST /faces/search — tombstone handler.
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
async fn embed_face(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<EmbedRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let ctx = guard(&state, &headers)?;
    audit("embed_face", ctx.tenant_id);
    match face_embedding::embed(&state.http, &req.image_b64).await {
        Err(e) => {
            tracing::warn!(error = %e, "face_embedding sidecar unreachable or no face detected");
            Ok(Json(serde_json::json!({
                "ok": false,
                "error": e.to_string(),
                "hint": "Is face_embedding_service.py running at FACE_EMBED_URL (default :5555)?",
            })))
        }
        Ok(embedding) => {
            let dims = embedding.len();
            let stored_for: Option<String> = if let Some(ref name) = req.face_name {
                let blob = face_embedding::embedding_to_bytes(&embedding);
                match state.store.store_embedding(ctx.tenant_id, name, &blob, dims).await {
                    Ok(true) => {
                        tracing::info!(face = %name, dims, "embedding stored in face_embeddings");
                        Some(name.clone())
                    }
                    Ok(false) => {
                        tracing::warn!(face = %name, "store_embedding: face row missing — call POST /faces first");
                        None
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, face = %name, "failed to store embedding");
                        None
                    }
                }
            } else {
                None
            };

            let resp = EmbedResponse { ok: true, embedding, dims, stored_for };
            Ok(Json(serde_json::to_value(resp).unwrap()))
        }
    }
}

/// POST /faces/search-by-image — rank all faces by cosine similarity to query image.
async fn search_by_image(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<SearchByImageRequest>,
) -> Result<Json<Vec<BiometricSearchResult>>, (StatusCode, Json<serde_json::Value>)> {
    let ctx = guard(&state, &headers)?;
    audit("search_by_image", ctx.tenant_id);

    // Step 1: embed query image via sidecar
    let query_embedding = match face_embedding::embed(&state.http, &req.image_b64).await {
        Ok(e) => e,
        Err(err) => {
            tracing::warn!(error = %err, "search-by-image: embedding sidecar unavailable");
            return Ok(Json(vec![]));
        }
    };

    // Step 2: load all stored embeddings + face metadata via the active backend.
    let rows = match state.store.load_embeddings(ctx.tenant_id).await {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!(error = %e, "search-by-image: backend load failed");
            return Ok(Json(vec![]));
        }
    };

    // Step 3: cosine similarity ranking
    let threshold = 0.28f32; // ArcFace same-person threshold
    let mut results: Vec<BiometricSearchResult> = rows
        .into_iter()
        .filter_map(|row| {
            let stored = face_embedding::bytes_to_embedding(&row.embedding)?;
            let sim = face_embedding::cosine_similarity(&query_embedding, &stored);
            if sim < req.min_similarity {
                return None;
            }
            Some(BiometricSearchResult {
                name: row.name,
                embedding_hint: row.embedding_hint,
                authorized: row.authorized,
                similarity: sim,
                likely_match: sim >= threshold,
            })
        })
        .collect();

    results.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(req.limit);

    // T9: Update last_matched_at for faces above the likely-match threshold
    // (biometric retention tracking). Fire-and-forget: failures don't fail
    // the search response.
    let touched: Vec<String> = results
        .iter()
        .filter(|r| r.likely_match)
        .map(|r| r.name.clone())
        .collect();
    if !touched.is_empty() {
        if let Err(e) = state
            .store
            .touch_last_matched(ctx.tenant_id, &touched, unix_now_secs())
            .await
        {
            tracing::warn!(error = %e, "T9: touch_last_matched failed (non-fatal)");
        }
    }

    Ok(Json(results))
}

// ── SQLite bootstrap (legacy backend) ───────────────────────────────────────

fn init_sqlite(conn: &Connection) -> Result<()> {
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

fn unix_now_secs() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}
