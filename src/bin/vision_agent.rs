use anyhow::{Context, Result};
use nuclear_eye::{caption_to_vision_event, now_ms, SecurityConfig, VisionEvent};
use nuclear_eye::memory::SecurityMemory;
use reqwest::Client;
use std::sync::{Arc, Mutex};
use tokio::time::{sleep, Duration};
use tracing::{error, info, warn};
use uuid::Uuid;

const MAX_RETRIES: u32 = 3;
const MAX_BUFFER_ATTEMPTS: i32 = 10;
const FLUSH_INTERVAL_SECS: u64 = 30;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    match nuclear_wrapper::wrap!(
        node_id      = "vision-agent",
        pg_url       = std::env::var("DATABASE_URL").unwrap_or_default(),
        signal_token = std::env::var("SIGNAL_TOKEN").unwrap_or_default()
    ) {
        Ok(nw) => { std::mem::forget(nw); }
        Err(e) => tracing::info!("nuclear-wrapper: start failed ({e}) — running unguarded"),
    }

    let cfg = SecurityConfig::load()?;
    let client = Client::new();
    let target_url = cfg.vision.target_url.clone();
    let tick = Duration::from_millis(cfg.vision.tick_ms);
    let camera_id = cfg.vision.default_camera_id.clone();
    let allow_synthetic = std::env::var("VISION_ALLOW_SYNTHETIC")
        .map(|v| matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "yes"))
        .unwrap_or(false);

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

    // Env var > vision config > top-level fastvlm_url
    let snapshot_url = std::env::var("CAMERA_SNAPSHOT_URL").ok()
        .or_else(|| std::env::var("SNAPSHOT_URL").ok())
        .or_else(|| cfg.vision.snapshot_url.clone());
    let fastvlm_url = std::env::var("FASTVLM_URL").ok()
        .or_else(|| cfg.vision.fastvlm_url.clone())
        .or_else(|| cfg.fastvlm_url.clone());

    if let Some(ref snap) = snapshot_url {
        info!("vision_agent.mode=real snapshot_url={snap} target_url={target_url} camera_id={camera_id}");
    } else if allow_synthetic {
        warn!("vision_agent.mode=synthetic target_url={target_url} camera_id={camera_id} VISION_ALLOW_SYNTHETIC=true");
    } else {
        error!("vision_agent has no snapshot source configured; synthetic mode is disabled");
    }

    let mut index: u64 = 0;
    loop {
        index += 1;
        let event = match (&snapshot_url, &fastvlm_url) {
            (Some(snap_url), Some(vlm_url)) => {
                capture_and_analyze(&client, snap_url, vlm_url, &camera_id)
                    .await
            }
            _ if allow_synthetic => Some(build_event(index, &camera_id)),
            _ => {
                warn!("vision_agent degraded: missing snapshot_url or fastvlm_url; skipping cycle");
                None
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
    let person_detected = index % 3 != 0;
    let known = index % 5 == 0;
    VisionEvent {
        event_id: Uuid::new_v4().to_string(),
        timestamp_ms: now_ms(),
        camera_id: camera_id.to_string(),
        behavior: if index % 4 == 0 { "loitering".into() } else { "passby".into() },
        risk_score: if index % 4 == 0 { 0.78 } else { 0.34 },
        stress_level: if index % 4 == 0 { 0.72 } else { 0.28 },
        confidence: if index % 4 == 0 { 0.81 } else { 0.94 },
        person_detected,
        person_name: if known { Some("known-resident".into()) } else { None },
        hands_visible: if index % 2 == 0 { 2 } else { 1 },
        object_held: if index % 7 == 0 { Some("unknown_object".into()) } else { None },
        extra_tags: if index % 4 == 0 {
            vec!["repeat_pass".into(), "attention_house".into(), "synthetic".into()]
        } else {
            vec!["normal_motion".into(), "synthetic".into()]
        },
        vlm_caption: None,
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
    let resp = reqwest::Client::new()
        .post(format!("{fastvlm_url}/describe"))
        .json(&body)
        .timeout(Duration::from_millis(800))
        .send().await.ok()?;
    let json: serde_json::Value = resp.json().await.ok()?;
    json["caption"].as_str().map(|s| s.to_string())
}
