use anyhow::{Context, Result};
use axum::{extract::{Path, State}, routing::{get, post}, Json, Router};
use nuclear_eye::{ensure_parent_dir, face_embedding, SecurityConfig};
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::info;

// ── Code truth note ───────────────────────────────────────────────────
//
// face_db now supports two search modes (Track O4):
//
// 1. TEXT SEARCH (backward compat) — POST /faces/search
//    Matches free-text query words against `embedding_hint` using SQL LIKE.
//    Used when no image is available (e.g., VLM caption lookup).
//
// 2. BIOMETRIC SEARCH (O4) — POST /faces/search-by-image
//    Input: base64 image → ArcFace sidecar → 512-dim embedding →
//    cosine similarity against stored embeddings → ranked results.
//    Requires face_embedding_service.py running at FACE_EMBED_URL (:5555).
//
// POST /faces/embed — extract and return embedding from a base64 image.
//
// ── Search types ──────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct SearchRequest {
    /// Free-text query matched against embedding_hint with SQL LIKE.
    /// Words are split and each must appear somewhere in embedding_hint.
    /// NOTE: This is text matching, NOT biometric recognition. See Track O4.
    query: String,
    /// Max results (default 5).
    #[serde(default = "default_limit")]
    limit: usize,
}

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

#[derive(Debug, Serialize)]
struct SearchResult {
    name: String,
    embedding_hint: String,
    authorized: bool,
    /// Rough match score: fraction of query words found in embedding_hint (0.0–1.0).
    score: f64,
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
        .route("/faces", get(list_faces).post(add_face))
        .route("/faces/search", post(search_faces))
        .route("/faces/embed", post(embed_face))
        .route("/faces/search-by-image", post(search_by_image))
        .route("/faces/:name", get(find_face))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(&cfg.app.bind_face_db).await?;
    info!("face_db listening on {}", cfg.app.bind_face_db);
    axum::serve(listener, app).await?;
    Ok(())
}

fn init_db(conn: &Connection) -> Result<()> {
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
    Ok(())
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
    let changed = conn
        .execute(
            "INSERT INTO faces(name, embedding_hint, authorized) VALUES(?1, ?2, ?3)
             ON CONFLICT(name) DO UPDATE SET embedding_hint = excluded.embedding_hint, authorized = excluded.authorized",
            params![input.name, input.embedding_hint, i64::from(input.authorized)],
        )
        .unwrap_or(0);
    Json(serde_json::json!({"upserted": changed > 0}))
}

/// Search faces by free-text query matched against `embedding_hint`.
///
/// Each word in `query` is checked against the lowercased embedding_hint.
/// Score = fraction of words that match (0.0–1.0).  Results above 0 are
/// returned sorted by score descending, up to `limit`.
///
/// Used by nuclear-scout to identify pedestrians via VLM captions.
async fn search_faces(
    State(state): State<AppState>,
    Json(req): Json<SearchRequest>,
) -> Json<Vec<SearchResult>> {
    let conn = state.conn.lock().await;
    let mut stmt = match conn.prepare(
        "SELECT name, embedding_hint, authorized FROM faces ORDER BY name",
    ) {
        Ok(s) => s,
        Err(_) => return Json(vec![]),
    };

    let words: Vec<String> = req.query
        .split_whitespace()
        .map(|w| w.to_ascii_lowercase())
        .filter(|w| w.len() >= 2)
        .collect();

    if words.is_empty() {
        return Json(vec![]);
    }

    let all: Vec<(String, String, bool)> = stmt
        .query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, i64>(2)? != 0,
            ))
        })
        .unwrap()
        .filter_map(|r| r.ok())
        .collect();

    let mut results: Vec<SearchResult> = all
        .into_iter()
        .filter_map(|(name, hint, authorized)| {
            let hint_lower = hint.to_ascii_lowercase();
            let matches = words.iter().filter(|w| hint_lower.contains(w.as_str())).count();
            if matches == 0 {
                return None;
            }
            let score = matches as f64 / words.len() as f64;
            Some(SearchResult { name, embedding_hint: hint, authorized, score })
        })
        .collect();

    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(req.limit);

    Json(results)
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

    // Step 2: load all stored embeddings + face metadata
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

    let rows: Vec<(String, Vec<u8>, String, bool)> = match stmt.query_map([], |row| {
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
    drop(stmt);
    drop(conn);

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

    Json(results)
}
