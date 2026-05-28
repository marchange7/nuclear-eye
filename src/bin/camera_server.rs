// camera_server.rs — HTTP frame buffer for nuclear-eye / Sentinelle
//
// Captures frames from one OR many camera sources and serves the latest JPEG.
// vision_agent polls this endpoint at VISION_TICK_MS rate.
//
// Sources (per camera):
//   rtsp://...     RTSP stream via ffmpeg subprocess (one frame per tick)
//   http://...     HTTP snapshot endpoint (polled at CAMERA_FPS)
//
// Env vars:
//   CAMERA_CONFIGS      — JSON map {"<cam_id>":"<url>", ...} for MULTI-camera.
//                         Takes precedence over CAMERA_URL when set + valid.
//   CAMERA_URL          — single-camera source URL (fallback when no CAMERA_CONFIGS)
//   CAMERA_ID           — id for the single-camera fallback (default: "default")
//   CAMERA_FPS          — capture rate in frames/second (default: 1.0, max: 10)
//   BIND_HOST           — bind address (default: 127.0.0.1)
//   CAMERA_SERVER_PORT  — HTTP port (default: 8090)
//   CAMERA_FRESHNESS_SEC— /readyz staleness threshold (default: 10s)
//
// Routes:
//   GET /snapshot              — the default camera's latest JPEG (back-compat)
//   GET /snapshot/{cam_id}     — that camera's latest JPEG
//   GET /api/snapshot/{cam_id} — alias (matches the sentinelle camera-adapter contract)
//   GET /health                — 200 + per-camera frame ages
//   GET /readyz                — 503 unless ALL cameras have a fresh frame

use std::collections::BTreeMap;
use std::process::Stdio;
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::{
    body::Body,
    extract::{Path, State},
    http::{header, StatusCode},
    response::Response,
    routing::get,
    Json, Router,
};
use tokio::sync::RwLock;
use tracing::{info, warn};

type FrameStore = Arc<RwLock<Option<(Vec<u8>, Instant)>>>;

#[derive(Clone)]
struct AppState {
    cameras: Arc<BTreeMap<String, FrameStore>>,
    default_id: Option<String>,
}

/// Resolve the camera set + the default camera id (for bare `/snapshot`) from env
/// values. Pure so it can be unit-tested. `CAMERA_CONFIGS` (JSON id->url) wins;
/// otherwise the single `CAMERA_URL` under `cam_id` (default "default"). The
/// default id is `CAMERA_ID` when present in the map, else the sole camera, else
/// the first id in sorted order, else None.
fn resolve_cameras(
    configs: Option<String>,
    url: Option<String>,
    cam_id: Option<String>,
) -> (BTreeMap<String, String>, Option<String>) {
    let mut cams: BTreeMap<String, String> = BTreeMap::new();

    if let Some(cfg) = configs.filter(|s| !s.trim().is_empty()) {
        match serde_json::from_str::<BTreeMap<String, String>>(&cfg) {
            Ok(m) => cams = m.into_iter().filter(|(_, u)| !u.trim().is_empty()).collect(),
            Err(e) => warn!("CAMERA_CONFIGS parse failed ({e}) — falling back to CAMERA_URL"),
        }
    }
    if cams.is_empty() {
        if let Some(u) = url.filter(|s| !s.trim().is_empty()) {
            let id = cam_id.clone().filter(|s| !s.trim().is_empty()).unwrap_or_else(|| "default".into());
            cams.insert(id, u);
        }
    }

    let default_id = match cam_id.filter(|s| cams.contains_key(s)) {
        Some(id) => Some(id),
        None if cams.len() == 1 => cams.keys().next().cloned(),
        None => cams.keys().next().cloned(),
    };
    (cams, default_id)
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    // S-7: fail-closed wrapper probe.
    nuclear_eye::wrapper_guard::check_wrapper("camera-server").await?;

    let fps: f64 = std::env::var("CAMERA_FPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1.0_f64)
        .clamp(0.1, 10.0);
    let bind = format!(
        "{}:{}",
        std::env::var("BIND_HOST").unwrap_or_else(|_| "127.0.0.1".into()),
        std::env::var("CAMERA_SERVER_PORT").unwrap_or_else(|_| "8090".into()),
    );

    let (cams, default_id) = resolve_cameras(
        std::env::var("CAMERA_CONFIGS").ok(),
        std::env::var("CAMERA_URL").ok(),
        std::env::var("CAMERA_ID").ok(),
    );

    if cams.is_empty() {
        warn!("no cameras configured (CAMERA_CONFIGS / CAMERA_URL) — /snapshot returns 503");
    }

    // One frame store + capture loop per camera.
    let mut stores: BTreeMap<String, FrameStore> = BTreeMap::new();
    let interval = Duration::from_secs_f64(1.0 / fps);
    for (id, url) in &cams {
        let store: FrameStore = Arc::new(RwLock::new(None));
        stores.insert(id.clone(), store.clone());
        info!(cam_id = %id, url = %url, fps, "starting capture loop");
        let (u, s) = (url.clone(), store.clone());
        tokio::spawn(async move { capture_loop(u, interval, s).await; });
    }

    let state = AppState {
        cameras: Arc::new(stores),
        default_id,
    };

    let app = Router::new()
        .route("/snapshot", get(serve_default))
        .route("/snapshot/:cam_id", get(serve_cam_route))
        .route("/api/snapshot/:cam_id", get(serve_cam_route))
        .route("/health", get(health))
        .route("/readyz", get(readyz))
        .with_state(state);

    info!("camera_server listening on {bind} (cameras: {})", cams.len());
    let listener = tokio::net::TcpListener::bind(&bind).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

// ── Capture loop ──────────────────────────────────────────────────────────────

async fn capture_loop(url: String, interval: Duration, store: FrameStore) {
    loop {
        let maybe_frame = if url.starts_with("rtsp://") || url.starts_with("rtsps://") {
            capture_rtsp(&url).await
        } else {
            capture_http(&url).await
        };

        match maybe_frame {
            Some(bytes) => {
                *store.write().await = Some((bytes, Instant::now()));
            }
            None => {
                warn!("capture tick failed — keeping previous frame");
            }
        }

        tokio::time::sleep(interval).await;
    }
}

/// Grab one JPEG frame from an RTSP stream using ffmpeg.
///
/// Bounded so a dead/slow camera can NEVER hang the capture loop (the bug that
/// froze a feed for 7+ min when a camera dropped): `-rw_timeout` makes ffmpeg
/// itself fail fast on a stalled connection, a tokio `timeout` is the backstop,
/// and `kill_on_drop` reaps the child if the timeout fires. On any timeout/error
/// we return None → the loop keeps the previous frame and retries next tick, so
/// the feed self-heals the instant the camera comes back.
async fn capture_rtsp(url: &str) -> Option<Vec<u8>> {
    use tokio::time::{timeout, Duration};
    // RTSP socket I/O timeout in microseconds (8s) — ffmpeg aborts a stalled
    // connect/read. NB: the rtsp demuxer's option is `-timeout` (NOT `-rw_timeout`,
    // which this ffmpeg build rejects with "Option rw_timeout not found").
    let rw_timeout = std::env::var("CAMERA_FFMPEG_RW_TIMEOUT_US")
        .unwrap_or_else(|_| "8000000".into());
    let hard_timeout_s: u64 = std::env::var("CAMERA_FFMPEG_TIMEOUT_SEC")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(12);

    let child = tokio::process::Command::new("ffmpeg")
        .args([
            "-loglevel", "error",
            "-rtsp_transport", "tcp",
            "-timeout", &rw_timeout,
            "-i", url,
            "-vframes", "1",
            "-f", "image2pipe",
            "-vcodec", "mjpeg",
            "pipe:1",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .kill_on_drop(true)
        .spawn()
        .map_err(|e| warn!("ffmpeg spawn failed: {e}"))
        .ok()?;

    match timeout(Duration::from_secs(hard_timeout_s), child.wait_with_output()).await {
        Ok(Ok(output)) if output.status.success() && !output.stdout.is_empty() => {
            Some(output.stdout)
        }
        Ok(Ok(output)) => {
            warn!("ffmpeg exited with {:?} for {url}", output.status.code());
            None
        }
        Ok(Err(e)) => {
            warn!("ffmpeg io error for {url}: {e}");
            None
        }
        Err(_) => {
            // child dropped here → kill_on_drop reaps the hung ffmpeg.
            warn!("ffmpeg timed out after {hard_timeout_s}s ({url} unreachable) — killed");
            None
        }
    }
}

/// Fetch a JPEG snapshot from an HTTP/HTTPS endpoint.
async fn capture_http(url: &str) -> Option<Vec<u8>> {
    let resp = reqwest::get(url)
        .await
        .map_err(|e| warn!("HTTP capture error: {e}"))
        .ok()?;

    if !resp.status().is_success() {
        warn!("HTTP capture returned {}", resp.status());
        return None;
    }

    resp.bytes()
        .await
        .map(|b| b.to_vec())
        .map_err(|e| warn!("HTTP capture body read error: {e}"))
        .ok()
}

// ── HTTP handlers ─────────────────────────────────────────────────────────────

fn jpeg_response(bytes: &[u8], age_ms: u64) -> Response {
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "image/jpeg")
        .header(header::CACHE_CONTROL, "no-cache, no-store")
        // Frame freshness so consumers (vision_agent) can skip a wedged/stale feed.
        .header("X-Frame-Age-Ms", age_ms.to_string())
        .body(Body::from(bytes.to_vec()))
        .unwrap()
}

fn json_status(code: StatusCode, body: &str) -> Response {
    Response::builder()
        .status(code)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}

async fn serve_cam(state: &AppState, cam_id: &str) -> Response {
    let Some(store) = state.cameras.get(cam_id) else {
        return json_status(StatusCode::NOT_FOUND, r#"{"error":"unknown camera"}"#);
    };
    match store.read().await.as_ref() {
        Some((bytes, ts)) => jpeg_response(bytes, ts.elapsed().as_millis() as u64),
        None => json_status(StatusCode::SERVICE_UNAVAILABLE, r#"{"error":"no frame available"}"#),
    }
}

async fn serve_default(State(state): State<AppState>) -> Response {
    match &state.default_id {
        Some(id) => serve_cam(&state, id).await,
        None => json_status(StatusCode::SERVICE_UNAVAILABLE, r#"{"error":"no camera configured"}"#),
    }
}

async fn serve_cam_route(State(state): State<AppState>, Path(cam_id): Path<String>) -> Response {
    serve_cam(&state, &cam_id).await
}

async fn health(State(state): State<AppState>) -> (StatusCode, Json<serde_json::Value>) {
    let mut ages = serde_json::Map::new();
    let mut any_ready = false;
    for (id, store) in state.cameras.iter() {
        let age = store.read().await.as_ref().map(|(_, ts)| ts.elapsed().as_millis() as u64);
        any_ready = any_ready || age.is_some();
        ages.insert(id.clone(), serde_json::json!(age));
    }
    (
        StatusCode::OK,
        Json(serde_json::json!({
            "ok": true,
            "service": "camera_server",
            "cameras": state.cameras.len(),
            "frame_ready": any_ready,
            "camera_ages_ms": ages,
        })),
    )
}

/// Freshness-aware liveness: 503 unless EVERY configured camera has a frame newer
/// than CAMERA_FRESHNESS_SEC (default 10s). No cameras configured -> 503.
async fn readyz(State(state): State<AppState>) -> (StatusCode, Json<serde_json::Value>) {
    let freshness_ms: u64 = std::env::var("CAMERA_FRESHNESS_SEC")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .map(|s| (s * 1000.0) as u64)
        .unwrap_or(10_000);

    if state.cameras.is_empty() {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({ "ready": false, "reason": "no_cameras_configured" })),
        );
    }

    let mut stale = serde_json::Map::new();
    let mut all_ready = true;
    for (id, store) in state.cameras.iter() {
        let age = store.read().await.as_ref().map(|(_, ts)| ts.elapsed().as_millis() as u64);
        let ready = matches!(age, Some(a) if a <= freshness_ms);
        if !ready {
            all_ready = false;
            stale.insert(id.clone(), serde_json::json!(age));
        }
    }

    if all_ready {
        (StatusCode::OK, Json(serde_json::json!({ "ready": true, "cameras": state.cameras.len() })))
    } else {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({
                "ready": false, "reason": "frame_stale_or_missing",
                "threshold_ms": freshness_ms, "not_ready": stale,
            })),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::resolve_cameras;

    #[test]
    fn single_camera_url_fallback() {
        let (cams, def) = resolve_cameras(None, Some("rtsp://x/1".into()), None);
        assert_eq!(cams.len(), 1);
        assert_eq!(cams.get("default").map(String::as_str), Some("rtsp://x/1"));
        assert_eq!(def.as_deref(), Some("default"));
    }

    #[test]
    fn single_camera_with_explicit_id() {
        let (cams, def) = resolve_cameras(None, Some("rtsp://x/1".into()), Some("ssc-cabled-ch1".into()));
        assert_eq!(cams.get("ssc-cabled-ch1").map(String::as_str), Some("rtsp://x/1"));
        assert_eq!(def.as_deref(), Some("ssc-cabled-ch1"));
    }

    #[test]
    fn multi_camera_configs_win_over_url() {
        let cfg = r#"{"ch1":"rtsp://a/1","ch2":"rtsp://b/2"}"#;
        let (cams, def) = resolve_cameras(Some(cfg.into()), Some("rtsp://ignored/0".into()), None);
        assert_eq!(cams.len(), 2);
        assert_eq!(cams.get("ch2").map(String::as_str), Some("rtsp://b/2"));
        // no CAMERA_ID -> first sorted key
        assert_eq!(def.as_deref(), Some("ch1"));
    }

    #[test]
    fn multi_camera_default_honors_camera_id() {
        let cfg = r#"{"ch1":"rtsp://a/1","ch2":"rtsp://b/2"}"#;
        let (_cams, def) = resolve_cameras(Some(cfg.into()), None, Some("ch2".into()));
        assert_eq!(def.as_deref(), Some("ch2"));
    }

    #[test]
    fn bad_json_configs_falls_back_to_url() {
        let (cams, def) = resolve_cameras(Some("{not json".into()), Some("rtsp://x/1".into()), None);
        assert_eq!(cams.len(), 1);
        assert_eq!(def.as_deref(), Some("default"));
    }

    #[test]
    fn no_sources_is_empty() {
        let (cams, def) = resolve_cameras(None, None, None);
        assert!(cams.is_empty());
        assert!(def.is_none());
    }

    #[test]
    fn empty_url_values_filtered_from_configs() {
        let cfg = r#"{"ch1":"rtsp://a/1","ch2":""}"#;
        let (cams, _def) = resolve_cameras(Some(cfg.into()), None, None);
        assert_eq!(cams.len(), 1);
        assert!(cams.contains_key("ch1"));
    }
}
