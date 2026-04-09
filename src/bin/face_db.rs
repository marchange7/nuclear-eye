use anyhow::{Context, Result};
use axum::{extract::{Path, State}, routing::{get, post}, Json, Router};
use nuclear_eye::{ensure_parent_dir, SecurityConfig};
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::info;

// ── Code truth note ───────────────────────────────────────────────────
//
// face_db is an IDENTITY HINT LOOKUP, not biometric face recognition.
// Search matches text words against the `embedding_hint` field using SQL LIKE.
// There is no image embedding, no cosine similarity, no neural face matching.
//
// Real biometric face recognition (ArcFace/FaceNet CoreML embeddings → cosine
// similarity) is Track O4 in Plan OS v4. Until then this is text lookup only.
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

#[derive(Debug, Serialize)]
struct SearchResult {
    name: String,
    embedding_hint: String,
    authorized: bool,
    /// Rough match score: fraction of query words found in embedding_hint (0.0–1.0).
    score: f64,
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

    let state = AppState { conn: Arc::new(Mutex::new(conn)) };
    let app = Router::new()
        .route("/faces", get(list_faces).post(add_face))
        .route("/faces/search", post(search_faces))
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
