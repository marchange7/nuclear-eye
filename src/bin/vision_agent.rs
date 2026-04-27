use anyhow::{Context, Result};
use axum::{
    extract::State,
    response::sse::{Event, KeepAlive, Sse},
    routing::get,
    Router,
};
use nuclear_eye::{caption_to_vision_event, now_ms, SecurityConfig, VisionEvent};
use nuclear_eye::memory::SecurityMemory;
use reqwest::Client;
use std::convert::Infallible;
use std::sync::{Arc, Mutex};
use tokio::sync::broadcast;
use tokio::time::{sleep, Duration};
use tokio_stream::wrappers::BroadcastStream;
use tokio_stream::StreamExt as _;
use tracing::{error, info, warn};
use uuid::Uuid;

const MAX_RETRIES: u32 = 3;
const MAX_BUFFER_ATTEMPTS: i32 = 10;
const FLUSH_INTERVAL_SECS: u64 = 30;

// ── RTSP / OpenCV frame capture ───────────────────────────────────────────────
//
// Q1: If CAMERA_URL is set, vision_agent captures frames directly from the RTSP
// stream and encodes them as JPEG for FastVLM, bypassing the Python snapshot server.
//
// Requires: the `opencv` or `gstreamer` feature (not enabled by default).
// On b450: install libopencv-dev and enable the `opencv` feature in Cargo.toml.
//
// If CAMERA_URL is not set, vision_agent falls back to:
//   1. CAMERA_SNAPSHOT_URL / SNAPSHOT_URL (HTTP JPEG endpoint) — real mode.
//   2. Synthetic events if VISION_ALLOW_SYNTHETIC=true — mock/dev mode.
//
// CAMERA_FPS controls how many frames per second are grabbed (default: 5).
// The vision tick rate (VISION_TICK_MS / vision.tick_ms) then paces VLM calls.

/// Grab one JPEG frame from an RTSP stream URL using OpenCV VideoCapture.
///
/// This function is only compiled when the `opencv` feature is enabled.
/// Without the feature it always returns None and logs a one-time warning.
#[cfg(feature = "opencv")]
fn grab_rtsp_frame(camera_url: &str) -> Option<Vec<u8>> {
    use opencv::{core, imgcodecs, videoio::{self, VideoCapture, CAP_ANY}};
    let mut cap = VideoCapture::from_file(camera_url, CAP_ANY).ok()?;
    if !cap.is_opened().unwrap_or(false) {
        warn!(url = %camera_url, "RTSP VideoCapture failed to open");
        return None;
    }
    let mut frame = core::Mat::default();
    cap.read(&mut frame).ok()?;
    if frame.empty() {
        return None;
    }
    let mut buf = core::Vector::<u8>::new();
    imgcodecs::imencode(".jpg", &frame, &mut buf, &core::Vector::new()).ok()?;
    Some(buf.to_vec())
}

#[cfg(not(feature = "opencv"))]
fn grab_rtsp_frame(_camera_url: &str) -> Option<Vec<u8>> {
    // opencv feature not enabled — RTSP capture unavailable at compile time.
    // Enable the `opencv` feature in Cargo.toml and install libopencv-dev.
    None
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    // S-7: fail-closed wrapper probe
    nuclear_eye::wrapper_guard::check_wrapper("vision-agent").await?;

    match nuclear_wrapper::wrap!(
        node_id      = "vision-agent",
        pg_url       = std::env::var("DATABASE_URL").unwrap_or_default(),
        signal_token = std::env::var("SIGNAL_TOKEN").unwrap_or_default()
    ) {
        Ok(nw) => { std::mem::forget(nw); }
        Err(e) => nuclear_eye::wrapper_guard::handle_wrap_failure("vision-agent", &e),
    }

    let cfg = SecurityConfig::load()?;
    let client = Client::new();
    let target_url = cfg.vision.target_url.clone();
    let tick = Duration::from_millis(cfg.vision.tick_ms);
    let camera_id = cfg.vision.default_camera_id.clone();
    let allow_synthetic = std::env::var("VISION_ALLOW_SYNTHETIC")
        .map(|v| matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "yes"))
        .unwrap_or(false);

    // Q1: RTSP camera support.
    // CAMERA_URL=rtsp://admin:pass@192.168.1.100:554/stream — direct RTSP capture.
    // Requires the `opencv` feature (libopencv-dev on b450).
    // CAMERA_FPS controls frame grab rate (default: 5). Actual VLM call rate is
    // governed by vision.tick_ms; CAMERA_FPS limits how often we grab from RTSP.
    let rtsp_url = std::env::var("CAMERA_URL").ok().filter(|s| !s.is_empty());
    let _camera_fps: u64 = std::env::var("CAMERA_FPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);

    let memory_path = {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
        format!("{home}/.nuclear-eye/memory.db")
    };
    std::fs::create_dir_all(std::path::Path::new(&memory_path).parent().unwrap()).ok();
    let memory = Arc::new(Mutex::new(
        SecurityMemory::open(&memory_path).context("failed to open memory db")?
    ));

    // Background: flush offline buffer every 30s
    let flush_client = client.clone();
    let flush_url = target_url.clone();
    let flush_mem = memory.clone();
    tokio::spawn(async move {
        loop {
            sleep(Duration::from_secs(FLUSH_INTERVAL_SECS)).await;
            flush_buffer(&flush_client, &flush_url, &flush_mem).await;
        }
    });

    // Q4: health sidecar for Lucky7 doctor probes + L2 /debug/frames SSE
    // Binds on VISION_HEALTH_PORT (default 8090).
    let (debug_frames_tx, _) = broadcast::channel::<String>(64);
    let debug_frames_tx = Arc::new(debug_frames_tx);
    {
        let health_port: u16 = std::env::var("VISION_HEALTH_PORT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(8090);
        let bind_host = std::env::var("BIND_HOST").unwrap_or_else(|_| "127.0.0.1".into());
        let addr = format!("{bind_host}:{health_port}");
        let debug_tx_route = debug_frames_tx.clone();
        let health_app = Router::new()
            .route("/health", get(|| async {
                axum::Json(serde_json::json!({"status":"ok","service":"vision_agent"}))
            }))
            .route("/debug/frames", get(move || {
                let rx = debug_tx_route.subscribe();
                let stream = BroadcastStream::new(rx).filter_map(|msg| match msg {
                    Ok(json) => Some(Ok::<Event, Infallible>(Event::default().data(json))),
                    Err(_)   => None,
                });
                async move { Sse::new(stream).keep_alive(KeepAlive::default()) }
            }));
        match tokio::net::TcpListener::bind(&addr).await {
            Ok(listener) => {
                info!("vision_agent sidecar listening on {addr} (health + /debug/frames)");
                tokio::spawn(async move {
                    if let Err(e) = axum::serve(listener, health_app).await {
                        warn!("vision_agent sidecar error: {e}");
                    }
                });
            }
            Err(e) => warn!("vision_agent: could not bind sidecar on {addr}: {e}"),
        }
    }

    // Env var > vision config > top-level fastvlm_url
    let snapshot_url = std::env::var("CAMERA_SNAPSHOT_URL").ok()
        .or_else(|| std::env::var("SNAPSHOT_URL").ok())
        .or_else(|| cfg.vision.snapshot_url.clone());
    let fastvlm_url = std::env::var("FASTVLM_URL").ok()
        .or_else(|| cfg.vision.fastvlm_url.clone())
        .or_else(|| cfg.fastvlm_url.clone());

    // Log startup mode
    if let Some(ref url) = rtsp_url {
        if cfg!(feature = "opencv") {
            info!("vision_agent.mode=rtsp camera_url={url} target_url={target_url} camera_id={camera_id}");
        } else {
            warn!(
                "vision_agent: CAMERA_URL={url} is set but the `opencv` feature is not compiled in; \
                 enable it in Cargo.toml and install libopencv-dev. Falling through to snapshot/synthetic."
            );
        }
    } else if let Some(ref snap) = snapshot_url {
        info!("vision_agent.mode=real snapshot_url={snap} target_url={target_url} camera_id={camera_id}");
    } else if allow_synthetic {
        warn!("vision_agent.mode=synthetic target_url={target_url} camera_id={camera_id} VISION_ALLOW_SYNTHETIC=true");
    } else {
        error!("vision_agent has no snapshot source configured (CAMERA_URL, CAMERA_SNAPSHOT_URL, or VISION_ALLOW_SYNTHETIC); stopping");
    }

    let mut index: u64 = 0;
    loop {
        index += 1;

        // Priority: RTSP (opencv) > HTTP snapshot > synthetic
        let event = if let (Some(ref url), Some(ref vlm_url)) = (&rtsp_url, &fastvlm_url) {
            // Q1: grab RTSP frame, encode as JPEG, describe via FastVLM
            match grab_rtsp_frame(url) {
                Some(jpeg) => {
                    if cfg!(feature = "opencv") {
                        describe_image(vlm_url, &jpeg).await
                            .map(|caption| caption_to_vision_event(&camera_id, &caption, now_ms()))
                    } else {
                        // Feature not compiled: warn once per cycle and fall through
                        warn!("CAMERA_URL set but opencv feature not enabled — install libopencv-dev and enable feature");
                        None
                    }
                }
                None => {
                    warn!(url = %url, "RTSP frame grab returned no data; skipping cycle");
                    None
                }
            }
        } else {
            match (&snapshot_url, &fastvlm_url) {
                (Some(snap_url), Some(vlm_url)) => {
                    capture_and_analyze(&client, snap_url, vlm_url, &camera_id)
                        .await
                }
                _ if allow_synthetic => Some(build_event(index, &camera_id)),
                _ => {
                    warn!("vision_agent degraded: missing snapshot_url or fastvlm_url; skipping cycle");
                    None
                }
            }
        };

        let Some(event) = event else {
            sleep(tick).await;
            continue;
        };

        let sent = send_with_retry(&client, &target_url, &event, MAX_RETRIES).await;

        if !sent {
            if let Ok(json) = serde_json::to_string(&event) {
                let mem = memory.lock().unwrap();
                match mem.buffer_event(&json, &target_url, now_ms()) {
                    Ok(_) => warn!(event_id = %event.event_id, "alarm_grader unreachable — buffered offline"),
                    Err(e) => error!("failed to buffer event: {e}"),
                }
            }
        }

        {
            let mem = memory.lock().unwrap();
            let _ = mem.record_vision(
                event.timestamp_ms, &event.behavior, event.risk_score,
                event.person_detected, event.person_name.as_deref(),
            );
        }

        // L2: broadcast to /debug/frames SSE
        if debug_frames_tx.receiver_count() > 0 {
            let debug_evt = serde_json::json!({
                "ts": event.timestamp_ms,
                "camera_id": event.camera_id,
                "behavior": event.behavior,
                "risk_score": event.risk_score,
                "stress_level": event.stress_level,
                "confidence": event.confidence,
                "person_detected": event.person_detected,
                "person_name": event.person_name,
                "sent": sent,
            });
            let _ = debug_frames_tx.send(debug_evt.to_string());
        }

        sleep(tick).await;
    }
}

async fn send_with_retry(client: &Client, url: &str, event: &VisionEvent, max_retries: u32) -> bool {
    let mut delay_ms = 500u64;
    for attempt in 1..=max_retries {
        match client.post(url).json(event).timeout(Duration::from_secs(5)).send().await {
            Ok(resp) if resp.status().is_success() => {
                info!(event_id = %event.event_id, attempt, "event sent");
                return true;
            }
            Ok(resp) => warn!(event_id = %event.event_id, attempt, status = %resp.status(), "non-success"),
            Err(e)   => warn!(event_id = %event.event_id, attempt, error = %e, "send failed"),
        }
        if attempt < max_retries {
            sleep(Duration::from_millis(delay_ms)).await;
            delay_ms *= 2;
        }
    }
    false
}

async fn flush_buffer(client: &Client, target_url: &str, memory: &Arc<Mutex<SecurityMemory>>) {
    let pending = {
        let mem = memory.lock().unwrap();
        mem.pending_events_for_target(target_url, 20).unwrap_or_default()
    };
    if pending.is_empty() { return; }
    info!("flushing {} buffered events", pending.len());

    for (id, json, target, _attempts) in pending {
        let event: VisionEvent = match serde_json::from_str(&json) {
            Ok(e) => e,
            Err(_) => { memory.lock().unwrap().delete_buffered_event(id).ok(); continue; }
        };
        match client.post(&target).json(&event).timeout(Duration::from_secs(5)).send().await {
            Ok(r) if r.status().is_success() => {
                info!(event_id = %event.event_id, "flushed buffered event");
                memory.lock().unwrap().delete_buffered_event(id).ok();
            }
            _ => {
                let mem = memory.lock().unwrap();
                mem.increment_buffer_attempts(id).ok();
                mem.prune_dead_events(MAX_BUFFER_ATTEMPTS).ok();
            }
        }
    }
}

fn build_event(index: u64, camera_id: &str) -> VisionEvent {
    let person_detected = !index.is_multiple_of(3);
    let known = index.is_multiple_of(5);
    let quad = index.is_multiple_of(4);
    VisionEvent {
        event_id: Uuid::new_v4().to_string(),
        timestamp_ms: now_ms(),
        camera_id: camera_id.to_string(),
        behavior: if quad { "loitering".into() } else { "passby".into() },
        risk_score: if quad { 0.78 } else { 0.34 },
        stress_level: if quad { 0.72 } else { 0.28 },
        confidence: if quad { 0.81 } else { 0.94 },
        person_detected,
        person_name: if known { Some("known-resident".into()) } else { None },
        hands_visible: if index.is_multiple_of(2) { 2 } else { 1 },
        object_held: if index.is_multiple_of(7) { Some("unknown_object".into()) } else { None },
        extra_tags: if quad {
            vec!["repeat_pass".into(), "attention_house".into(), "synthetic".into()]
        } else {
            vec!["normal_motion".into(), "synthetic".into()]
        },
        vlm_caption: None,
        depth_context: None,
        face_negative: None,
        voice_agitated: None,
        gesture_threat: None,
    }
}

/// Capture a JPEG snapshot from camera_server, describe via FastVLM, parse into VisionEvent.
async fn capture_and_analyze(
    client: &Client,
    snapshot_url: &str,
    fastvlm_url: &str,
    camera_id: &str,
) -> Option<VisionEvent> {
    let resp = client
        .get(snapshot_url)
        .timeout(Duration::from_secs(3))
        .send().await
        .map_err(|e| warn!("snapshot.get.failed: {e}"))
        .ok()?;

    if !resp.status().is_success() {
        warn!("snapshot.http.{}", resp.status());
        return None;
    }

    let image_bytes = resp.bytes().await
        .map_err(|e| warn!("snapshot.body.failed: {e}"))
        .ok()?
        .to_vec();

    let caption = describe_image(fastvlm_url, &image_bytes).await?;
    info!("vlm.caption: {caption}");
    Some(caption_to_vision_event(camera_id, &caption, now_ms()))
}

async fn describe_image(fastvlm_url: &str, image_bytes: &[u8]) -> Option<String> {
    use base64::Engine;
    let b64 = base64::engine::general_purpose::STANDARD.encode(image_bytes);
    let body = serde_json::json!({
        "image_b64": b64,
        "prompt": "Describe this security camera frame in one sentence. Note people, behavior, and any unusual activity."
    });
    let api_token = std::env::var("FASTVLM_API_TOKEN").unwrap_or_default();
    let token = api_token.trim();
    let client = reqwest::Client::new();
    let mut req = client
        .post(format!("{fastvlm_url}/describe"))
        .json(&body)
        .timeout(Duration::from_millis(800));
    if !token.is_empty() {
        req = req.bearer_auth(token);
    }
    let resp = req.send().await.ok()?;
    let json: serde_json::Value = resp.json().await.ok()?;
    json["caption"].as_str().map(|s| s.to_string())
}
