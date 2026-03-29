use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use axum::{
    extract::{State, ws::{Message, WebSocket, WebSocketUpgrade}},
    routing::{get, post},
    Json, Router,
};
use nuclear_eye::{decide, AffectTriad, AlarmEvent, AlarmGrader, AlarmLevel, AlarmSummary, ConsulClient, SecurityConfig, VisionEvent};
use nuclear_eye::memory::SecurityMemory;
use serde::{Deserialize, Serialize};
use tokio::sync::{broadcast, Mutex};
use tracing::{info, warn};

const CONSUL_TIMEOUT_MS: u64 = 80;
const WATCH_CHANNEL_CAP: usize = 64;
const THREAT_KEYWORDS: &[&str] = &["person", "vehicle", "movement", "intrusion"];

/// Events broadcast to nuclear-watch over WebSocket.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum WatchEvent {
    Alarm {
        ts: u64,
        camera_id: String,
        level: String,
        score: f64,
        reason: String,
        caption: Option<String>,
    },
    Decision {
        ts: u64,
        camera_id: String,
        question: String,
        synthesis: String,
        confidence: f64,
    },
}

#[derive(Debug, Deserialize)]
struct CameraFrame {
    camera_id: String,
    caption: String,
    timestamp_ms: u64,
}

#[derive(Debug, Serialize)]
struct CameraFrameResponse {
    stored: bool,
    threat_detected: bool,
    forwarded: bool,
}

#[derive(Clone)]
struct AppState {
    grader: Arc<Mutex<AlarmGrader>>,
    consul: ConsulClient,
    fortress_url: String,
    fortress_enabled: bool,
    memory: Arc<Mutex<SecurityMemory>>,
    penny_url: Option<String>,
    watch_tx: broadcast::Sender<String>,
    alert_lang: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    // ── Nuclear wrapper — resilience sidecar ────────────────────────────
    match nuclear_wrapper::wrap!(
        node_id      = "alarm-grader-agent",
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
    let mut grader = AlarmGrader::new();
    grader.history_len = cfg.alarm.history_len;
    grader.hysteresis_window = cfg.alarm.hysteresis_window;
    grader.danger_thresholds = cfg.alarm.thresholds;

    let consul_url = std::env::var("CONSUL_URL")
        .unwrap_or_else(|_| "http://127.0.0.1:7710".to_string());
    let hc_consul_url = format!("{consul_url}/health");
    let consul = ConsulClient::new(consul_url, CONSUL_TIMEOUT_MS);

    let fortress_url = cfg.fortress_url();
    let fortress_enabled = cfg.fortress.mesh_enabled;
    let penny_url = std::env::var("PENNY_BRAIN_URL").ok();

    let memory_path = {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
        format!("{home}/.nuclear-eye/memory.db")
    };
    std::fs::create_dir_all(std::path::Path::new(&memory_path).parent().unwrap()).ok();
    let memory = SecurityMemory::open(&memory_path)
        .expect("failed to open security memory db");

    let (watch_tx, _) = broadcast::channel(WATCH_CHANNEL_CAP);
    let memory = Arc::new(Mutex::new(memory));

    let alert_lang = std::env::var("ALERT_LANG").unwrap_or_else(|_| "fr".to_string());

    let state = AppState {
        grader: Arc::new(Mutex::new(grader)),
        consul,
        fortress_url,
        fortress_enabled,
        memory: memory.clone(),
        penny_url,
        watch_tx,
        alert_lang,
    };

    let app = Router::new()
        .route("/ingest", post(ingest))
        .route("/sensor/camera", post(handle_camera_frame))
        .route("/summary", get(summary))
        .route("/ws", get(ws_handler))
        .with_state(state);

    // Background: health check every 30s
    let hc_mem = memory.clone();
    tokio::spawn(async move {
        let client = reqwest::Client::new();
        loop {
            tokio::time::sleep(Duration::from_secs(30)).await;
            let consul_ok = client.get(&hc_consul_url)
                .timeout(Duration::from_secs(2)).send().await
                .map(|r| r.status().is_success()).unwrap_or(false);
            let buffered = hc_mem.lock().await.buffered_count().unwrap_or(0);
            info!(consul_ok, buffered_events = buffered, "health_check");
        }
    });

    let listener = tokio::net::TcpListener::bind(&cfg.app.bind_alarm_grader).await?;
    info!("alarm_grader_agent listening on {}", cfg.app.bind_alarm_grader);
    axum::serve(listener, app).await?;
    Ok(())
}

async fn ingest(
    State(state): State<AppState>,
    Json(event): Json<VisionEvent>,
) -> Json<serde_json::Value> {
    process_event(state, event).await
}

async fn handle_camera_frame(
    State(state): State<AppState>,
    Json(frame): Json<CameraFrame>,
) -> Json<CameraFrameResponse> {
    let threat_detected = THREAT_KEYWORDS
        .iter()
        .any(|kw| frame.caption.to_lowercase().contains(kw));
    info!(camera_id = %frame.camera_id, threat_detected, "camera frame received");

    let mut forwarded = false;
    if threat_detected {
        let event = nuclear_eye::caption_to_vision_event(&frame.camera_id, &frame.caption, frame.timestamp_ms);
        let _ = process_event(state, event).await;
        forwarded = true;
    }

    Json(CameraFrameResponse {
        stored: true,
        threat_detected,
        forwarded,
    })
}

async fn process_event(
    state: AppState,
    event: VisionEvent,
) -> Json<serde_json::Value> {
    // Local grading is synchronous and fast — do it under the lock, then
    // release before any I/O so we never block other ingest calls.
    let mut alarm = {
        let mut grader = state.grader.lock().await;
        grader.grade_event(&event)
    };

    // For High alarms, fire a Consul deliberation and penny-brain in parallel
    // with the response path.  Consul gets up to CONSUL_TIMEOUT_MS; penny-brain
    // gets 500ms.  If neither replies in time, the local decision stands unchanged.
    let consul_note = if alarm.level == AlarmLevel::High {
        let question = format!(
            "House security HIGH alarm: behavior='{}', risk={:.2}, stress={:.2}, confidence={:.2}, person={:?}",
            event.behavior, event.risk_score, event.stress_level, event.confidence, event.person_name
        );

        // Fire penny-brain and consul in parallel
        let penny_future = async {
            if let Some(ref penny_url) = state.penny_url {
                query_penny(penny_url, &question).await
            } else {
                None
            }
        };
        let consul_handle = state.consul.query_async(&question);

        let (penny_result, consul_result) = tokio::join!(
            tokio::time::timeout(Duration::from_millis(500), penny_future),
            tokio::time::timeout(Duration::from_millis(CONSUL_TIMEOUT_MS), consul_handle),
        );

        let mut note = String::new();

        // Append consul result
        match consul_result {
            Ok(Ok(Some(cd))) => {
                info!(
                    decision = %cd.decision,
                    confidence = cd.confidence,
                    voices = cd.voices,
                    event_id = %event.event_id,
                    "consul enhanced high-alarm decision"
                );
                note.push_str(&format!(
                    " | consul={} conf={:.2} voices={}",
                    cd.decision, cd.confidence, cd.voices
                ));
                // Broadcast Consul decision to nuclear-watch
                if let Ok(json) = serde_json::to_string(&WatchEvent::Decision {
                    ts: event.timestamp_ms,
                    camera_id: event.camera_id.clone(),
                    question: format!("High alarm: {}", event.behavior),
                    synthesis: cd.decision.clone(),
                    confidence: cd.confidence,
                }) {
                    let _ = state.watch_tx.send(json);
                }
            }
            Ok(Ok(None)) => {}
            Ok(Err(e)) => {
                warn!(error = %e, "consul task panicked");
            }
            Err(_) => {
                tracing::debug!("consul did not respond within {CONSUL_TIMEOUT_MS} ms; using local grade");
            }
        }

        // Append penny-brain result if available
        if let Ok(Some(penny_note)) = penny_result {
            let short = if penny_note.len() > 120 {
                format!("{}…", &penny_note[..120])
            } else {
                penny_note
            };
            note.push_str(&format!(" | penny={short}"));
        }

        if note.is_empty() { None } else { Some(note) }
    } else {
        None
    };

    if let Some(extra) = &consul_note {
        alarm.note.push_str(extra);
    }

    // Broadcast alarm to nuclear-watch WebSocket clients
    if let Ok(json) = serde_json::to_string(&WatchEvent::Alarm {
        ts: alarm.timestamp_ms,
        camera_id: event.camera_id.clone(),
        level: alarm.level.to_string(),
        score: alarm.danger_score,
        reason: alarm.note.clone(),
        caption: alarm.vlm_caption.clone(),
    }) {
        let _ = state.watch_tx.send(json);
    }

    // Fire-and-forget publish to Fortress mesh
    if state.fortress_enabled {
        let triad = AffectTriad::from_alarm_event(&alarm);
        let decision = alarm.level.to_string();
        let fortress_url = state.fortress_url.clone();
        let alarm_clone = alarm.clone();
        tokio::spawn(async move {
            publish_to_mesh(&alarm_clone, &triad, &decision, &fortress_url).await;
        });
    }

    // Record alarm to SQLite long-term memory (fire-and-forget; never blocks the response)
    {
        let mem = state.memory.lock().await;
        let level_str = alarm.level.to_string();
        let note_str = if alarm.note.is_empty() { None } else { Some(alarm.note.as_str()) };
        if let Err(e) = mem.record_alarm(alarm.timestamp_ms, &level_str, alarm.danger_score, note_str, &level_str) {
            tracing::warn!("memory.record_alarm failed: {e}");
        }
    }

    // Synthesize voice alert for High alarms via nuclear-voice-client.
    let audio_b64 = if alarm.level == AlarmLevel::High {
        if let Some(vc) = nuclear_voice_client::VoiceClient::from_env() {
            let location = event.camera_id.replace('_', " ");
            let alert_text = match state.alert_lang.as_str() {
                "en" => format!("Security alert — danger level detected at {location}"),
                "de" => format!("Sicherheitsalarm — Gefahrenstufe erkannt bei {location}"),
                "es" => format!("Alerta de seguridad — nivel de peligro detectado en {location}"),
                _ => format!("Alerte sécurité — niveau danger détecté à {location}"),
            };
            vc.speak(&alert_text, Some("decisive"), Some(&state.alert_lang)).await
        } else {
            None
        }
    } else {
        None
    };

    // Fire-and-forget actuation (lights / buzzer / arm via MQTT)
    {
        let triad = AffectTriad::from_alarm_event(&alarm);
        let action_str = decide(&triad, alarm.level == AlarmLevel::High).to_string();
        let level_str = alarm.level.to_string();
        let cam_id = event.camera_id.clone();
        tokio::spawn(async move {
            if let Some(actuator_url) = std::env::var("ACTUATOR_URL").ok() {
                let client = reqwest::Client::builder()
                    .timeout(Duration::from_secs(2))
                    .build()
                    .expect("reqwest client");
                let payload = serde_json::json!({
                    "action": action_str,
                    "level": level_str,
                    "camera_id": cam_id,
                });
                if let Err(e) = client.post(format!("{actuator_url}/actuate"))
                    .json(&payload)
                    .send()
                    .await
                {
                    warn!(error = %e, "actuator_agent unreachable — physical output skipped");
                }
            }
        });
    }

    Json(serde_json::json!({
        "accepted": true,
        "event_id": event.event_id,
        "alarm": alarm,
        "audio_b64": audio_b64,
    }))
}

async fn summary(State(state): State<AppState>) -> Json<AlarmSummary> {
    let grader = state.grader.lock().await;
    Json(grader.summary())
}

async fn query_penny(penny_url: &str, question: &str) -> Option<String> {
    let client = reqwest::Client::new();
    let url = format!("{}/route", penny_url.trim_end_matches('/'));
    let payload = serde_json::json!({ "prompt": question, "history_tokens": 0 });
    let result = tokio::time::timeout(
        Duration::from_millis(500),
        client.post(&url).json(&payload).send(),
    )
    .await;
    match result {
        Ok(Ok(resp)) if resp.status().is_success() => {
            if let Ok(body) = resp.json::<serde_json::Value>().await {
                let text = body["response"].as_str().unwrap_or_default().trim().to_string();
                if !text.is_empty() {
                    tracing::debug!(
                        model = body["model_used"].as_str().unwrap_or("?"),
                        tier = body["tier"].as_u64().unwrap_or(0),
                        "penny-brain routed alarm assessment"
                    );
                    return Some(text);
                }
            }
            None
        }
        Ok(Ok(resp)) => {
            tracing::warn!(status = %resp.status(), "penny-brain non-success for alarm");
            None
        }
        Ok(Err(e)) => {
            tracing::warn!(error = %e, "penny-brain request failed");
            None
        }
        Err(_) => {
            tracing::warn!("penny-brain timed out (500ms)");
            None
        }
    }
}


// ── WebSocket — nuclear-watch LAN feed ──────────────────────────────────

async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
) -> impl axum::response::IntoResponse {
    let rx = state.watch_tx.subscribe();
    ws.on_upgrade(|socket| handle_watch_socket(socket, rx))
}

async fn handle_watch_socket(mut socket: WebSocket, mut rx: broadcast::Receiver<String>) {
    loop {
        match rx.recv().await {
            Ok(msg) => {
                if socket.send(Message::Text(msg)).await.is_err() {
                    break; // client disconnected
                }
            }
            Err(broadcast::error::RecvError::Closed) => break,
            Err(broadcast::error::RecvError::Lagged(n)) => {
                warn!("nuclear-watch ws lagged {n} messages");
            }
        }
    }
}

async fn publish_to_mesh(alarm: &AlarmEvent, triad: &AffectTriad, decision: &str, fortress_url: &str) {
    let client = reqwest::Client::new();
    let payload = serde_json::json!({
        "alarm": alarm,
        "triad": triad,
        "decision": decision,
        "context": alarm.vlm_caption.as_deref().unwrap_or_default(),
        "vision_source": "FastVLM-0.5B",
    });
    let result = client
        .post(format!("{}/v1/mesh/security", fortress_url))
        .json(&payload)
        .timeout(std::time::Duration::from_millis(500))
        .send()
        .await;
    match result {
        Ok(resp) => info!(status = %resp.status(), "published alarm to Fortress mesh"),
        Err(err) => warn!(%err, "Fortress publish failed (non-blocking)"),
    }
}
