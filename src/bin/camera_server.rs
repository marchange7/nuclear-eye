// camera_server.rs — HTTP frame buffer for nuclear-eye / Sentinelle
//
// Captures frames from a camera source and serves the latest JPEG at /snapshot.
// vision_agent polls this endpoint at VISION_TICK_MS rate.
//
// Sources (tried in order based on env vars set):
//   CAMERA_URL=rtsp://...     RTSP stream via ffmpeg subprocess (one frame per tick)
//   CAMERA_URL=http://...     HTTP snapshot endpoint (polled at CAMERA_FPS)
//
// Env vars:
//   CAMERA_URL          — camera source URL (RTSP or HTTP snapshot)
//   CAMERA_FPS          — capture rate in frames/second (default: 1.0, max: 10)
//   BIND_HOST           — bind address (default: 127.0.0.1)
//   CAMERA_SERVER_PORT  — HTTP port (default: 8090)

use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;

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

type FrameStore = Arc<RwLock<Option<Vec<u8>>>>;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    // S-7: fail-closed wrapper probe — exits(1) if NUCLEAR_WRAPPER_URL is set
    // but unreachable, or if WRAPPER_REQUIRED=1 and URL is absent.
    nuclear_eye::wrapper_guard::check_wrapper("camera-server").await?;

    let camera_url = std::env::var("CAMERA_URL").ok().filter(|s| !s.is_empty());
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

    let frame: FrameStore = Arc::new(RwLock::new(None));

    match &camera_url {
        Some(url) => {
            info!(url = %url, fps, "starting capture loop");
            let frame_clone = frame.clone();
            let url_clone = url.clone();
            let interval = Duration::from_secs_f64(1.0 / fps);
            tokio::spawn(async move {
                capture_loop(url_clone, interval, frame_clone).await;
            });
        }
        None => {
            warn!("CAMERA_URL not set — /snapshot returns 503 until a frame arrives");
        }
    }

    let app = Router::new()
        .route("/snapshot", get(serve_snapshot))
        .route("/snapshot/{cam_id}", get(serve_snapshot_cam))
        .route("/health", get(health))
        .with_state(frame);

    info!("camera_server listening on {bind}");
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
                *store.write().await = Some(bytes);
            }
            None => {
                warn!("capture tick failed — keeping previous frame");
            }
        }

        tokio::time::sleep(interval).await;
    }
}

/// Grab one JPEG frame from an RTSP stream using ffmpeg.
/// Requires `ffmpeg` on PATH (available in most Linux images via apt).
async fn capture_rtsp(url: &str) -> Option<Vec<u8>> {
    let output = tokio::process::Command::new("ffmpeg")
        .args([
            "-loglevel", "error",
            "-rtsp_transport", "tcp",
            "-i", url,
            "-vframes", "1",
            "-f", "image2pipe",
            "-vcodec", "mjpeg",
            "pipe:1",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .await
        .map_err(|e| warn!("ffmpeg spawn failed: {e}"))
        .ok()?;

    if output.status.success() && !output.stdout.is_empty() {
        Some(output.stdout)
    } else {
        warn!("ffmpeg exited with {:?} for {url}", output.status.code());
        None
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

async fn serve_snapshot(State(store): State<FrameStore>) -> Response {
    let guard = store.read().await;
    match guard.as_ref() {
        Some(bytes) => Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "image/jpeg")
            .header(header::CACHE_CONTROL, "no-cache, no-store")
            .body(Body::from(bytes.clone()))
            .unwrap(),
        None => Response::builder()
            .status(StatusCode::SERVICE_UNAVAILABLE)
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(r#"{"error":"no frame available"}"#))
            .unwrap(),
    }
}

/// Accept /snapshot/{cam_id} for compatibility with multi-camera callers.
/// Single-camera: cam_id is logged but all cameras share this one frame store.
async fn serve_snapshot_cam(
    State(store): State<FrameStore>,
    Path(cam_id): Path<String>,
) -> Response {
    tracing::debug!(cam_id = %cam_id, "snapshot requested for cam_id");
    serve_snapshot(State(store)).await
}

async fn health(State(store): State<FrameStore>) -> (StatusCode, Json<serde_json::Value>) {
    let frame_ready = store.read().await.is_some();
    (
        StatusCode::OK,
        Json(serde_json::json!({
            "ok": true,
            "service": "camera_server",
            "frame_ready": frame_ready,
        })),
    )
}
