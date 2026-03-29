use anyhow::{Context, Result};
use axum::{extract::State, http::StatusCode, routing::post, Json, Router};
use nuclear_eye::{now_ms, IPhoneSensorData, PedestrianSummary, SecurityConfig, VisionEvent};
use nuclear_eye::memory::SecurityMemory;
use reqwest::Client;
use serde::Serialize;
use std::sync::{Arc, Mutex};
use tokio::time::Duration;
use tracing::{info, warn};
use uuid::Uuid;

const MAX_RETRIES: u32 = 3;
const MAX_BUFFER_ATTEMPTS: i32 = 10;
const FLUSH_INTERVAL_SECS: u64 = 30;

#[derive(Clone)]
struct AppState {
    client: Arc<Client>,
    alarm_grader_url: String,
    memory: Arc<Mutex<SecurityMemory>>,
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

    let bind = std::env::var("IPHONE_SENSOR_BIND")
        .unwrap_or_else(|_| "0.0.0.0:8089".to_string());
    let alarm_grader_url = cfg.app.alarm_grader_url.clone();

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
    let flush_url = alarm_grader_url.clone();
    let flush_mem = memory.clone();
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(Duration::from_secs(FLUSH_INTERVAL_SECS)).await;
            flush_buffer(&flush_client, &flush_url, &flush_mem).await;
        }
    });

    let state = AppState { client, alarm_grader_url, memory };

    let app = Router::new()
        .route("/sensor/iphone", post(handle_iphone_sensor))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(&bind).await?;
    info!(bind = %bind, "iphone_sensor_agent started");
    axum::serve(listener, app).await?;
    Ok(())
}

async fn handle_iphone_sensor(
    State(state): State<AppState>,
    Json(data): Json<IPhoneSensorData>,
) -> (StatusCode, Json<IngestResponse>) {
    info!(device_id = %data.device_id, pedestrians = data.pedestrians.len(), "nuclear-scout data received");

    let events = iphone_to_vision_events(&data);
    let total = events.len();
    let mut accepted = 0usize;
    let mut buffered = 0usize;
    let ingest_url = format!("{}/ingest", state.alarm_grader_url);

    for event in events {
        if send_with_retry(&state.client, &ingest_url, &event, MAX_RETRIES).await {
            accepted += 1;
        } else {
            if let Ok(json) = serde_json::to_string(&event) {
                state.memory.lock().unwrap().buffer_event(&json, &ingest_url, now_ms()).ok();
                buffered += 1;
            }
        }
    }

    (StatusCode::OK, Json(IngestResponse { accepted, skipped: total - accepted - buffered, buffered }))
}

async fn send_with_retry(client: &Client, url: &str, event: &VisionEvent, max_retries: u32) -> bool {
    let mut delay_ms = 500u64;
    for attempt in 1..=max_retries {
        match client.post(url).json(event).timeout(Duration::from_secs(5)).send().await {
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
        match client.post(&target).json(&event).timeout(Duration::from_secs(5)).send().await {
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
                    tags
                },
                vlm_caption: None,
            }
        })
        .collect()
}
