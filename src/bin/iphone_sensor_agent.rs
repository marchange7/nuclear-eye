use anyhow::{Context, Result};
use axum::{
    extract::State,
    http::{HeaderMap, StatusCode},
    response::sse::{Event, KeepAlive, Sse},
    routing::{get, post},
    Json, Router,
};
use nuclear_eye::{
    face_db_auth, now_ms, IPhoneSensorData, PedestrianSummary, SecurityConfig, VisionEvent,
};
use nuclear_eye::memory::SecurityMemory;
use reqwest::Client;
use serde::Serialize;
use std::convert::Infallible;
use std::sync::{Arc, Mutex};
use tokio::sync::broadcast;
use tokio::time::Duration;
use tokio_stream::wrappers::BroadcastStream;
use tokio_stream::StreamExt as _;
use tracing::{info, warn};
use uuid::Uuid;

fn hash_device_id(id: &str) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut h = DefaultHasher::new();
    id.hash(&mut h);
    format!("{:016x}", h.finish())
}

const MAX_RETRIES: u32 = 3;
const MAX_BUFFER_ATTEMPTS: i32 = 10;
const FLUSH_INTERVAL_SECS: u64 = 30;

/// Ring-buffer capacity for `/debug/depth` SSE broadcast.
const DEBUG_DEPTH_CHANNEL_CAP: usize = 64;

#[derive(Clone)]
struct AppState {
    client: Arc<Client>,
    alarm_grader_url: String,
    memory: Arc<Mutex<SecurityMemory>>,
    /// `IPHONE_SENSOR_TOKEN` at boot. `None` ⇒ open mode, parity with
    /// `alarm_grader_agent` / `face_db`. Logged at startup.
    ///
    /// Closes `os/55` HIGH at the HTTP layer.
    sensor_token: Option<Arc<String>>,
    /// `KERNEL_REQUIRE_TENANT_HEADER` at boot. `false` ⇒ Pass 1a/1b
    /// (missing `X-Tenant-Id` resolves to `kernel.legacy_default_tenant()`,
    /// `os/57 §4.7`).
    require_tenant_header: bool,
    /// SSE broadcast for `/debug/depth` — each frame serialised as JSON.
    debug_depth_tx: Arc<broadcast::Sender<String>>,
}

/// Standard auth + tenant guard. Reuses [`face_db_auth::authenticate`] with
/// an agent-specific bearer token env (`IPHONE_SENSOR_TOKEN`).
fn guard(
    state: &AppState,
    headers: &HeaderMap,
) -> Result<face_db_auth::AuthContext, (StatusCode, Json<serde_json::Value>)> {
    face_db_auth::authenticate(
        headers,
        state.sensor_token.as_deref().map(String::as_str),
        state.require_tenant_header,
    )
    .map_err(face_db_auth::AuthError::into_response)
}

fn sensor_token_from_env() -> Option<String> {
    std::env::var("IPHONE_SENSOR_TOKEN").ok().filter(|s| !s.is_empty())
}

#[derive(Debug, Serialize)]
struct IngestResponse {
    accepted: usize,
    skipped: usize,
    buffered: usize,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    // S-7: fail-closed wrapper probe
    nuclear_eye::wrapper_guard::check_wrapper("iphone-sensor-agent").await?;

    match nuclear_wrapper::wrap!(
        node_id      = "iphone-sensor-agent",
        pg_url       = std::env::var("DATABASE_URL").unwrap_or_default(),
        signal_token = std::env::var("SIGNAL_TOKEN").unwrap_or_default()
    ) {
        Ok(nw) => { std::mem::forget(nw); }
        Err(e) => tracing::info!("nuclear-wrapper: start failed ({e}) — running unguarded"),
    }

    let cfg = SecurityConfig::load()?;
    let client = Arc::new(Client::builder().build().context("failed to build HTTP client")?);

    let bind = std::env::var("IPHONE_SENSOR_BIND").unwrap_or_else(|_| {
        let host = std::env::var("BIND_HOST").unwrap_or_else(|_| "127.0.0.1".to_string());
        format!("{host}:8089")
    });
    let alarm_grader_url = cfg.app.alarm_grader_url.clone();

    let memory_path = {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
        format!("{home}/.nuclear-eye/memory.db")
    };
    std::fs::create_dir_all(std::path::Path::new(&memory_path).parent().unwrap()).ok();
    let memory = Arc::new(Mutex::new(
        SecurityMemory::open(&memory_path).context("failed to open memory db")?
    ));

    // os/56 P1-5 — wire bearer auth + multi-tenant header guard.
    // IPHONE_SENSOR_TOKEN unset ⇒ open mode (parity with face_db / alarm_grader).
    // KERNEL_REQUIRE_TENANT_HEADER=1 ⇒ Pass 1c strict (os/57 §4.7).
    let sensor_token = sensor_token_from_env().map(Arc::new);
    let require_tenant_header = face_db_auth::require_tenant_from_env();
    if sensor_token.is_none() {
        warn!(
            "IPHONE_SENSOR_TOKEN is not set — POST /sensor/iphone accepts \
             unauthenticated requests; set the token in production"
        );
    }
    if !require_tenant_header {
        info!(
            "KERNEL_REQUIRE_TENANT_HEADER unset — Pass 1a/1b semantics (missing \
             X-Tenant-Id resolves to legacy default tenant)"
        );
    }

    // Background: flush offline buffer every 30s
    let flush_client = client.clone();
    let flush_url = alarm_grader_url.clone();
    let flush_mem = memory.clone();
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(Duration::from_secs(FLUSH_INTERVAL_SECS)).await;
            flush_buffer(&flush_client, &flush_url, &flush_mem).await;
        }
    });

    let (debug_depth_tx, _) = broadcast::channel(DEBUG_DEPTH_CHANNEL_CAP);
    let debug_depth_tx = Arc::new(debug_depth_tx);

    let state = AppState {
        client,
        alarm_grader_url,
        memory,
        sensor_token,
        require_tenant_header,
        debug_depth_tx,
    };

    let app = Router::new()
        .route("/sensor/iphone", post(handle_iphone_sensor))
        .route("/debug/depth", get(debug_depth_sse))
        .route("/health", get(health))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(&bind).await?;
    info!(bind = %bind, "iphone_sensor_agent started");
    axum::serve(listener, app).await?;
    Ok(())
}

async fn handle_iphone_sensor(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(data): Json<IPhoneSensorData>,
) -> Result<(StatusCode, Json<IngestResponse>), (StatusCode, Json<serde_json::Value>)> {
    let ctx = guard(&state, &headers)?;
    info!(
        tenant_id = %ctx.tenant_id,
        device_id_hash = %hash_device_id(&data.device_id),
        pedestrians = data.pedestrians.len(),
        "nuclear-scout data received"
    );

    let events = iphone_to_vision_events(&data);
    let total = events.len();
    let mut accepted = 0usize;
    let mut buffered = 0usize;
    let ingest_url = format!("{}/ingest", state.alarm_grader_url);
    let tenant_str = ctx.tenant_id.to_string();

    for event in events {
        if send_with_retry(&state.client, &ingest_url, &event, &tenant_str, MAX_RETRIES).await {
            accepted += 1;
        } else if let Ok(json) = serde_json::to_string(&event) {
            // os/56 P1-5: buffered events currently lose tenant attribution
            // and are flushed under `kernel.legacy_default_tenant()`. The
            // canonical fix lives in P1-7 (kernel-pg outbox table carrying
            // tenant_id per row); the in-process SQLite buffer is intentionally
            // not extended here to keep the change surface small.
            state.memory.lock().unwrap().buffer_event(&json, &ingest_url, now_ms()).ok();
            buffered += 1;
        }
    }

    // L2: broadcast to /debug/depth SSE subscribers (fire-and-forget; lag tolerance = channel cap)
    let debug_event = serde_json::json!({
        "ts": now_ms(),
        "device_id_hash": hash_device_id(&data.device_id),
        "lidar_available": data.lidar_available,
        "tracking_quality": data.tracking_quality,
        "pedestrians": data.pedestrians.len(),
        "accepted": accepted,
        "buffered": buffered,
    });
    let _ = state.debug_depth_tx.send(debug_event.to_string());

    Ok((
        StatusCode::OK,
        Json(IngestResponse {
            accepted,
            skipped: total - accepted - buffered,
            buffered,
        }),
    ))
}

/// GET /health — nuclear-watch uses this to verify the scout ingest agent is alive.
async fn health() -> (StatusCode, Json<serde_json::Value>) {
    (StatusCode::OK, Json(serde_json::json!({ "ok": true, "service": "iphone_sensor_agent" })))
}

/// GET /debug/depth — L2 debug SSE stream: ARKit depth frames and pedestrian events.
///
/// Each event is a JSON line:
/// ```json
/// {"ts":…,"device_id_hash":"…","lidar_available":true,"tracking_quality":"normal","pedestrians":2,"accepted":2,"buffered":0}
/// ```
///
/// No auth — debug endpoints are local-only (BIND_HOST defaults to 127.0.0.1).
/// Use `ncli capture iphone-sensor-agent` to stream from the mesh.
async fn debug_depth_sse(
    State(state): State<AppState>,
) -> Sse<impl tokio_stream::Stream<Item = Result<Event, Infallible>>> {
    let rx = state.debug_depth_tx.subscribe();
    let stream = BroadcastStream::new(rx).filter_map(|msg| {
        match msg {
            Ok(json) => Some(Ok(Event::default().data(json))),
            Err(_) => None, // lagged; skip
        }
    });
    Sse::new(stream).keep_alive(KeepAlive::default())
}

async fn send_with_retry(
    client: &Client,
    url: &str,
    event: &VisionEvent,
    tenant_id: &str,
    max_retries: u32,
) -> bool {
    let mut delay_ms = 500u64;
    for attempt in 1..=max_retries {
        match client
            .post(url)
            .header("X-Tenant-Id", tenant_id)
            .json(event)
            .timeout(Duration::from_secs(5))
            .send()
            .await
        {
            Ok(r) if r.status().is_success() => { return true; }
            Ok(r) => warn!(status = %r.status(), attempt, "rejected"),
            Err(e) => warn!(error = %e, attempt, "send failed"),
        }
        if attempt < max_retries {
            tokio::time::sleep(Duration::from_millis(delay_ms)).await;
            delay_ms *= 2;
        }
    }
    false
}

async fn flush_buffer(client: &Client, alarm_url: &str, memory: &Arc<Mutex<SecurityMemory>>) {
    // os/56 P1-5: buffered events forward under the legacy default tenant —
    // see the P1-7 outbox follow-up for proper per-row tenant tracking.
    let tenant_str = face_db_auth::LEGACY_DEFAULT_TENANT.to_string();
    let ingest_url = format!("{alarm_url}/ingest");
    let pending = {
        memory
            .lock()
            .unwrap()
            .pending_events_for_target(&ingest_url, 20)
            .unwrap_or_default()
    };
    if pending.is_empty() { return; }
    info!("flushing {} buffered events", pending.len());
    for (id, json, target, _attempts) in pending {
        let event: VisionEvent = match serde_json::from_str(&json) {
            Ok(e) => e,
            Err(_) => { memory.lock().unwrap().delete_buffered_event(id).ok(); continue; }
        };
        match client
            .post(&target)
            .header("X-Tenant-Id", &tenant_str)
            .json(&event)
            .timeout(Duration::from_secs(5))
            .send()
            .await
        {
            Ok(r) if r.status().is_success() => { memory.lock().unwrap().delete_buffered_event(id).ok(); }
            _ => {
                let mem = memory.lock().unwrap();
                mem.increment_buffer_attempts(id).ok();
                mem.prune_dead_events(MAX_BUFFER_ATTEMPTS).ok();
            }
        }
    }
}

/// Compute stress from scene context rather than a simple risk proxy.
///
/// Inputs:
/// - `p`: the specific pedestrian being evaluated
/// - `all`: all pedestrians in this sensor frame (for crowding signal)
/// - `tracking_quality`: "normal" | "limited" | "not_available"
fn compute_stress(p: &PedestrianSummary, all: &[PedestrianSummary], tracking_quality: &str) -> f64 {
    let mut stress = 0.0f64;

    // Proximity: closer = higher stress (0.4 at 0 m, 0.0 at 5 m)
    if let Some(d) = p.distance_m {
        stress += (1.0 - (d / 5.0).min(1.0)) * 0.40;
    }

    // Collision imminence
    if let Some(eta) = p.collision_eta_s {
        stress += if eta < 2.0 { 0.40 } else if eta < 5.0 { 0.20 } else { 0.05 };
    }

    // Crowding: each additional pedestrian within 3 m adds stress (capped)
    let near_others = all.iter()
        .filter(|q| q.track_id != p.track_id && q.distance_m.map(|d| d < 3.0).unwrap_or(false))
        .count();
    stress += (near_others as f64 * 0.10).min(0.30);

    // Phone-distracted pedestrian is less predictable → harder to avoid
    if p.is_using_phone { stress += 0.10; }

    // Poor tracking → higher uncertainty → higher stress
    match tracking_quality {
        "limited"       => stress += 0.10,
        "not_available" => stress += 0.20,
        _ => {}
    }

    stress.clamp(0.0, 1.0)
}

fn iphone_to_vision_events(data: &IPhoneSensorData) -> Vec<VisionEvent> {
    data.pedestrians.iter()
        .filter(|p| p.distance_m.map(|d| d < 5.0).unwrap_or(false))
        .map(|p| {
            let risk = p.collision_eta_s.map(|eta| 1.0 - (eta / 10.0_f64).min(1.0)).unwrap_or(0.1);
            let stress = compute_stress(p, &data.pedestrians, &data.tracking_quality);
            let behavior = p.collision_eta_s.map(|eta| {
                if eta < 3.0 { "approaching_fast".to_string() } else { "nearby".to_string() }
            }).unwrap_or_else(|| "nearby".to_string());
            VisionEvent {
                event_id: Uuid::new_v4().to_string(),
                timestamp_ms: data.timestamp_ms,
                camera_id: format!("scout:{}", data.device_id),
                behavior,
                risk_score: risk,
                stress_level: stress,
                confidence: p.confidence,
                person_detected: true,
                person_name: p.identity.clone(),
                hands_visible: 0,
                object_held: if p.is_using_phone { Some("phone".to_string()) } else { None },
                extra_tags: {
                    let mut tags = vec!["nuclear-scout".to_string()];
                    if p.is_using_phone { tags.push("phone-distracted".to_string()); }
                    if !data.lidar_available { tags.push("no-lidar".to_string()); }
                    if let Some(ref model) = data.vlm_model {
                        tags.push(format!("vlm:{model}"));
                    }
                    tags
                },
                vlm_caption: data.vlm_caption.clone(),
                depth_context: None,
                face_negative: None,
                voice_agitated: None,
                gesture_threat: None,
            }
        })
        .collect()
}
