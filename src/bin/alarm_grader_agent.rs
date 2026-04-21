use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use chrono::Utc;
use axum::{
    extract::{State, ws::{Message, WebSocket, WebSocketUpgrade}},
    routing::{get, post},
    Json, Router,
};
use nuclear_eye::{decide, riviere, AffectTriad, AlarmEvent, AlarmGrader, AlarmLevel, AlarmSummary, ConsulClient, SecurityConfig, VisionEvent};
use nuclear_eye::memory::SecurityMemory;
use nuclear_sdk::NuclearClient;
use serde::{Deserialize, Serialize};
use tokio::sync::{broadcast, Mutex};
use tracing::{info, warn};

const CONSUL_TIMEOUT_MS: u64 = 80;
/// Default Penny L1 timeout for High-alarm `query_penny` (override with `PENNY_GRADER_TIMEOUT_MS`).
const DEFAULT_PENNY_GRADER_TIMEOUT_MS: u64 = 800;
const WATCH_CHANNEL_CAP: usize = 64;
const THREAT_KEYWORDS: &[&str] = &["person", "vehicle", "movement", "intrusion"];

/// Events broadcast to nuclear-watch over WebSocket (O6 adds Pedestrian + Vision).
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum WatchEvent {
    /// Alarm level decision — existing event type.
    /// `degraded` is always serialized (`true`/`false`) for simple client parsers.
    /// For **High** alarms: `true` when Penny L1 did not apply (see `process_event` comment).
    /// For other levels: always `false`.
    Alarm {
        ts: u64,
        camera_id: String,
        level: String,
        score: f64,
        reason: String,
        caption: Option<String>,
        degraded: bool,
    },
    /// Consul deliberation result — existing event type.
    Decision {
        ts: u64,
        camera_id: String,
        question: String,
        synthesis: String,
        confidence: f64,
    },
    /// O6 / scout: pedestrian count + optional per-pedestrian detail from nuclear-scout.
    /// nuclear-watch decodes: source, distance_m, phone_flag, collision_eta_s.
    Pedestrian {
        ts: u64,
        camera_id: String,
        count: u32,
        positions: Vec<serde_json::Value>,
        /// Set when the event originates from nuclear-scout (camera_id starts with "scout:")
        #[serde(skip_serializing_if = "Option::is_none")]
        source: Option<String>,
        /// Closest pedestrian distance in metres (from scout sensor data)
        #[serde(skip_serializing_if = "Option::is_none")]
        distance_m: Option<f64>,
        /// True when the closest pedestrian is phone-distracted
        #[serde(skip_serializing_if = "Option::is_none")]
        phone_flag: Option<bool>,
        /// Time-to-collision in seconds for the closest pedestrian
        #[serde(skip_serializing_if = "Option::is_none")]
        collision_eta_s: Option<f64>,
    },
    /// O6: scene summary with detected objects from VLM caption.
    Vision {
        ts: u64,
        camera_id: String,
        scene: String,
        objects: Vec<String>,
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
    nk: NuclearClient,
    fortress_enabled: bool,
    memory: Arc<Mutex<SecurityMemory>>,
    watch_tx: broadcast::Sender<String>,
    alert_lang: String,
    /// Shared HTTP client for La Rivière domain event POSTs (O7).
    http: reqwest::Client,
    /// chain-comms base URL for SMS/Signal High-alarm notifications (optional).
    /// Set COMMS_URL=http://127.0.0.1:9100 to enable.
    comms_url: Option<String>,
    /// Bearer token for chain-comms API authentication.
    comms_api_token: Option<String>,
    /// Recipient phone/Signal number for High-alarm notifications (E.164).
    comms_alert_recipient: Option<String>,
    /// Optional bearer token for POST /feedback.
    /// Set ALARM_GRADER_FEEDBACK_TOKEN to require authentication on the feedback endpoint.
    /// If unset, the endpoint is open (internal-only — bind behind a gateway or VPN).
    feedback_token: Option<String>,
    /// `tokio::time::timeout` budget for Penny L1 on High alarms (`PENNY_GRADER_TIMEOUT_MS`, default 800).
    penny_grader_timeout_ms: u64,
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
    let consul = ConsulClient::new(consul_url, CONSUL_TIMEOUT_MS);

    let nk = NuclearClient::from_system()
        .expect("NuclearClient: check FORTRESS_URL / PENNY_BRAIN_URL env vars");

    let fortress_enabled = cfg.fortress.mesh_enabled;

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

    let http = reqwest::Client::builder()
        .timeout(std::time::Duration::from_millis(600))
        .build()
        .expect("build HTTP client");

    let comms_url = std::env::var("COMMS_URL").ok().filter(|s| !s.is_empty());
    let comms_api_token = std::env::var("COMMS_API_TOKEN").ok().filter(|s| !s.is_empty() && !s.starts_with("TODO"));
    let comms_alert_recipient = std::env::var("COMMS_ALERT_RECIPIENT").ok().filter(|s| s.starts_with('+'));
    let feedback_token = std::env::var("ALARM_GRADER_FEEDBACK_TOKEN").ok().filter(|s| !s.is_empty());
    if feedback_token.is_none() {
        warn!(
            "ALARM_GRADER_FEEDBACK_TOKEN is not set — POST /feedback accepts unauthenticated requests; set the token in production"
        );
    }

    let penny_grader_timeout_ms = std::env::var("PENNY_GRADER_TIMEOUT_MS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(DEFAULT_PENNY_GRADER_TIMEOUT_MS);

    let state = AppState {
        grader: Arc::new(Mutex::new(grader)),
        consul,
        nk: nk.clone(),
        fortress_enabled,
        memory: memory.clone(),
        watch_tx,
        alert_lang,
        http,
        comms_url,
        comms_api_token,
        comms_alert_recipient,
        feedback_token,
        penny_grader_timeout_ms,
    };

    let app = Router::new()
        .route("/ingest", post(ingest))
        .route("/sensor/camera", post(handle_camera_frame))
        .route("/feedback", post(handle_feedback))
        .route("/summary", get(summary))
        .route("/ws", get(ws_handler))
        .route("/health", get(alarm_health))
        .with_state(state);

    // Background: health check every 30s via SDK
    let nk_hc = nk.clone();
    let hc_mem = memory.clone();
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(Duration::from_secs(30)).await;
            let consul_ok = nk_hc.consul().health().await.is_ok();
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

    // JJ6: Depth-enhanced scoring — apply LiDAR adjustments before any fan-out.
    //
    // If depth context is present, `depth_adjust_score` may:
    //   • Suppress the alarm entirely (all blobs < 0.5m height → cat/pet)
    //   • Force Critical (fall_detected = true)
    //   • Amplify or attenuate based on interpersonal distance zone
    //
    // The adjusted score re-maps through map_danger_to_level so the alarm level
    // remains consistent with the grader's configured thresholds.
    if let Some(ref depth) = event.depth_context {
        let base = alarm.danger_score as f32;
        let (adjusted, suppress_reason) = depth_adjust_score(base, depth);
        if let Some(ref reason) = suppress_reason {
            tracing::info!(
                event_id = %event.event_id,
                camera_id = %event.camera_id,
                reason = %reason,
                "JJ6: alarm suppressed by depth context"
            );
            alarm.level = AlarmLevel::None;
            alarm.danger_score = 0.0;
            alarm.note.push_str(&format!(" | depth-suppressed: {reason}"));
        } else if (adjusted - base).abs() > 1e-4 {
            alarm.danger_score = adjusted as f64;
            // Re-derive level from adjusted score using grader thresholds.
            let thresholds = {
                let grader = state.grader.lock().await;
                grader.danger_thresholds
            };
            alarm.level = if adjusted >= thresholds[2] as f32 {
                AlarmLevel::High
            } else if adjusted >= thresholds[1] as f32 {
                AlarmLevel::Medium
            } else if adjusted >= thresholds[0] as f32 {
                AlarmLevel::Low
            } else {
                AlarmLevel::None
            };
            tracing::debug!(
                event_id = %event.event_id,
                base_score = base,
                adjusted_score = adjusted,
                level = %alarm.level,
                "JJ6: depth-adjusted danger score"
            );
        }
    }

    // JJ6-H: Sync the hysteresis window with the depth-adjusted outcome.
    // grade_event() pushed the pre-depth alarm into recent_events; overwrite
    // its level/score so future hysteresis decisions see the real result.
    if event.depth_context.is_some() {
        let mut grader = state.grader.lock().await;
        if let Some(last) = grader.recent_events.back_mut() {
            last.level = alarm.level.clone();
            last.danger_score = alarm.danger_score;
        }
    }

    // WebSocket `degraded` (High only): set when Penny L1 did not apply — see below.
    let mut watch_alarm_degraded = false;

    // For High alarms, fire Consul deliberation and penny-brain in parallel.
    // Consul gets up to CONSUL_TIMEOUT_MS; Penny L1 gets PENNY_GRADER_TIMEOUT_MS (default 800ms).
    // If neither replies in time, the local decision stands unchanged.
    let consul_note = if alarm.level == AlarmLevel::High {
        let question = format!(
            "House security HIGH alarm: behavior='{}', risk={:.2}, stress={:.2}, confidence={:.2}, person={:?}",
            event.behavior, event.risk_score, event.stress_level, event.confidence, event.person_name
        );

        let penny_future = {
            let nk = state.nk.clone();
            let q = question.clone();
            async move { query_penny(&nk, &q).await }
        };
        let consul_handle = state.consul.query_async(&question);

        let penny_timeout_ms = state.penny_grader_timeout_ms;
        let (penny_result, consul_result) = tokio::join!(
            tokio::time::timeout(Duration::from_millis(penny_timeout_ms), penny_future),
            tokio::time::timeout(Duration::from_millis(CONSUL_TIMEOUT_MS), consul_handle),
        );

        // Penny L1 "applied" only when we got non-empty text within the timeout.
        // `Err(_)` = timeout; `Ok(None)` = Penny error, empty trim, or inner failure (query_penny maps errors to None).
        // Product: `degraded` reflects Penny L1 only — Consul timeout alone does not set `degraded` if Penny succeeded.
        watch_alarm_degraded = !matches!(penny_result, Ok(Some(_)));

        let mut note = String::new();

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

    // ── Q5: Audit log — append verdict record before any fan-out ────────────────
    //
    // Synchronous append to AUDIT_LOG_PATH (default /var/log/nuclear-eye/audit.jsonl).
    // Spawned in a blocking task so we don't block the Tokio thread on file I/O.
    {
        let cam_id   = event.camera_id.clone();
        let behavior = event.behavior.clone();
        let verdict  = alarm.level.to_string();
        let conf     = alarm.danger_score as f32;
        let triad_a  = AffectTriad::from_alarm_event(&alarm);
        let action_s = decide(&triad_a, alarm.level == AlarmLevel::High).to_string();
        tokio::task::spawn_blocking(move || {
            nuclear_eye::audit::log_decision(&cam_id, &behavior, &verdict, conf, &action_s);
        });
    }

    // ── O7 / Q5: La Rivière FIRST (canonical source of truth) ────────────────
    //
    // Fan-out order: La Rivière → WebSocket → Fortress mesh → Telegram
    // La Rivière is the write-ahead log; WebSocket / mesh are derived views.

    // 1a. vision.person_detected (always, when person_detected = true)
    if event.person_detected {
        let http = state.http.clone();
        let cam_id = event.camera_id.clone();
        let ts = event.timestamp_ms;
        tokio::spawn(async move {
            riviere::emit_person_detected(&http, riviere::PersonDetectedPayload {
                camera_id: cam_id,
                count: 1,
                ts,
                positions: vec![],
            }).await;
        });
    }

    // 1b. vision.behavior_alert (always — captures every graded event)
    {
        let http = state.http.clone();
        let cam_id = event.camera_id.clone();
        let behavior = event.behavior.clone();
        let severity = alarm.level.to_string();
        let danger_score = alarm.danger_score;
        let ts = event.timestamp_ms;
        tokio::spawn(async move {
            riviere::emit_behavior_alert(&http, riviere::BehaviorAlertPayload {
                camera_id: cam_id,
                behavior,
                severity,
                danger_score,
                ts,
            }).await;
        });
    }

    // 1c. vision.scene_captured (when VLM caption available)
    if let Some(ref caption) = alarm.vlm_caption {
        let http = state.http.clone();
        let cam_id = event.camera_id.clone();
        let scene = caption.clone();
        let ts = event.timestamp_ms;
        // Extract objects from extra_tags for richer scene payload
        let objects: Vec<String> = event.extra_tags.iter()
            .filter(|t| t.as_str() != "vlm-derived")
            .cloned()
            .collect();
        tokio::spawn(async move {
            riviere::emit_scene_captured(&http, riviere::SceneCapturedPayload {
                camera_id: cam_id,
                scene,
                ts,
                objects,
            }).await;
        });
    }

    // 1d. JJ1: sentinelle.alarm domain event (continuous learning pipeline)
    //         JJ6: depth_context forwarded verbatim for learning pipeline correlation.
    {
        let http = state.http.clone();
        let alarm_id = alarm.alarm_id.clone();
        let cam_id = event.camera_id.clone();
        let level = alarm.level.to_string();
        let danger_score = alarm.danger_score;
        let risk_score = event.risk_score;
        let stress_level = event.stress_level;
        let confidence = event.confidence;
        let behavior = event.behavior.clone();
        let person_detected = event.person_detected;
        let person_name = event.person_name.clone();
        let ts = event.timestamp_ms;
        // JJ6: Serialize depth context to Value so it's preserved in the Rivière payload
        // without coupling riviere.rs to the DepthContext type.
        let depth_context_value = event
            .depth_context
            .as_ref()
            .and_then(|d| serde_json::to_value(d).ok());
        tokio::spawn(async move {
            riviere::emit_sentinelle_alarm(&http, riviere::SentinelleAlarmPayload {
                alarm_id,
                camera_id: cam_id,
                level,
                danger_score,
                risk_score,
                stress_level,
                confidence,
                behavior,
                person_detected,
                person_name,
                ts,
                depth_context: depth_context_value,
            }).await;
        });
    }

    // 1e-GG4. Push active alarm context to Fortress agent memory (Emile + Arianne)
    //         Medium or High → write active_alarm key; None or Low → clear it.
    {
        let http = state.http.clone();
        let level_str = alarm.level.to_string();
        let is_active = matches!(alarm.level, AlarmLevel::Medium | AlarmLevel::High);
        let risk_score = alarm.risk_score;
        let danger_score = alarm.danger_score;
        let cam_id = event.camera_id.clone();
        let behavior = event.behavior.clone();
        let decision = alarm.note.clone();
        let ts = alarm.timestamp_ms;
        tokio::spawn(async move {
            let fortress_url = std::env::var("FORTRESS_URL")
                .unwrap_or_else(|_| "http://127.0.0.1:7700".to_string());
            let api_token = std::env::var("FORTRESS_API_TOKEN").unwrap_or_default();
            let timestamp = chrono::Utc::now().to_rfc3339();
            let value = if is_active {
                serde_json::json!({
                    "level": level_str,
                    "risk_score": risk_score,
                    "danger_score": danger_score,
                    "camera_id": cam_id,
                    "behavior": behavior,
                    "decision": decision,
                    "ts": ts,
                })
            } else {
                serde_json::Value::Null
            };
            let payload = serde_json::json!({
                "key": "active_alarm",
                "value": value,
                "timestamp": timestamp,
            });
            let token = api_token.trim();
            for agent in &["arianne", "emile"] {
                let url = format!("{fortress_url}/v1/agents/{agent}/memory");
                let mut req = http
                    .post(&url)
                    .json(&payload)
                    .timeout(std::time::Duration::from_millis(500));
                if !token.is_empty() {
                    req = req.bearer_auth(token);
                }
                let result = req.send().await;
                match result {
                    Ok(r) if r.status().is_success() => {
                        tracing::debug!(agent = %agent, level = %level_str, "GG4: active_alarm memory pushed");
                    }
                    Ok(r) => {
                        tracing::warn!(agent = %agent, status = %r.status(), "GG4: active_alarm push non-success (non-blocking)");
                    }
                    Err(e) => {
                        tracing::warn!(agent = %agent, err = %e, "GG4: active_alarm push failed (non-blocking)");
                    }
                }
            }
        });
    }

    // 1f. Legacy La Rivière stream (reflection surface — backward compat)
    {
        let triad   = AffectTriad::from_alarm_event(&alarm);
        let action  = decide(&triad, alarm.level == AlarmLevel::High).to_string();
        let content = format!(
            "AlarmLevel::{} @ {} — {} — {} (J={:.2}, doubt={:.2}, det={:.2})",
            alarm.level, event.camera_id, event.behavior, action,
            triad.judgement, triad.doubt, triad.determination,
        );
        let nk = state.nk.clone();
        tokio::spawn(async move {
            nuclear_eye::riviere::post_event("nuclear-eye", "camera", &content, &nk).await;
        });
    }

    // ── 2. WebSocket broadcast to nuclear-watch (O6 + existing types) ─────────

    // 2a. Alarm event (existing)
    if let Ok(json) = serde_json::to_string(&WatchEvent::Alarm {
        ts: alarm.timestamp_ms,
        camera_id: event.camera_id.clone(),
        level: alarm.level.to_string(),
        score: alarm.danger_score,
        reason: alarm.note.clone(),
        caption: alarm.vlm_caption.clone(),
        degraded: watch_alarm_degraded,
    }) {
        let _ = state.watch_tx.send(json);
    }

    // 2b. Pedestrian event (O6 / scout) — emitted when at least one person detected.
    // For scout-origin events (camera_id = "scout:<device>") we surface per-pedestrian
    // detail fields that nuclear-watch decodes: source, distance_m, phone_flag, collision_eta_s.
    if event.person_detected {
        let is_scout = event.camera_id.starts_with("scout:");
        // Extract scout-specific fields from extra_tags and event metadata.
        // object_held == Some("phone") is the phone-distracted signal from iphone_sensor_agent.
        let phone_flag = if is_scout { Some(event.object_held.as_deref() == Some("phone")) } else { None };
        // distance_m and collision_eta_s are not stored in VisionEvent directly; they are
        // encoded in the behavior string and risk_score by iphone_to_vision_events(). We
        // surface what we have: None for fields without a canonical source in VisionEvent.
        if let Ok(json) = serde_json::to_string(&WatchEvent::Pedestrian {
            ts: event.timestamp_ms,
            camera_id: event.camera_id.clone(),
            count: 1,
            positions: vec![],
            source: if is_scout { Some(event.camera_id.clone()) } else { None },
            distance_m: None,
            phone_flag,
            collision_eta_s: None,
        }) {
            let _ = state.watch_tx.send(json);
        }
    }

    // 2c. Vision scene event (O6) — emitted when VLM caption is available
    if let Some(ref caption) = event.vlm_caption {
        let objects: Vec<String> = event.extra_tags.iter()
            .filter(|t| t.as_str() != "vlm-derived")
            .cloned()
            .collect();
        if let Ok(json) = serde_json::to_string(&WatchEvent::Vision {
            ts: event.timestamp_ms,
            camera_id: event.camera_id.clone(),
            scene: caption.clone(),
            objects,
        }) {
            let _ = state.watch_tx.send(json);
        }
    }

    // ── Q8: nuclear-chain dual-publish ────────────────────────────────────────────
    //
    // When CHAIN_ENABLED=true and NUCLEAR_CHAIN_URL is set, POST the alarm verdict
    // to nuclear-chain /v1/events in addition to the existing WebSocket broadcast.
    // Dual-publish during transition — existing WS path is always active.
    // Fire-and-forget: chain publish never delays ingest or blocks the caller.
    {
        let chain_enabled = std::env::var("CHAIN_ENABLED")
            .map(|v| v.trim().eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let chain_url = std::env::var("NUCLEAR_CHAIN_URL")
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());

        if chain_enabled {
            if let Some(chain_url) = chain_url {
                let http = state.http.clone();
                let cam_id = event.camera_id.clone();
                let level = alarm.level.to_string();
                let score = alarm.danger_score;
                let reason = alarm.note.clone();
                let caption = alarm.vlm_caption.clone();
                let ts = alarm.timestamp_ms;
                let chain_token = std::env::var("NUCLEAR_CHAIN_TOKEN")
                    .ok()
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty());

                tokio::spawn(async move {
                    let payload = serde_json::json!({
                        "type": "sentinelle.alarm.verdict",
                        "camera_id": cam_id,
                        "level": level,
                        "score": score,
                        "reason": reason,
                        "caption": caption,
                        "ts": ts,
                    });
                    let mut req = http
                        .post(format!("{chain_url}/v1/events"))
                        .header("X-Chain-Service", "alarm-grader-agent")
                        .header("X-Chain-Path", "/alerts")
                        .header("X-Chain-Target", "nuclear-watch")
                        .json(&payload)
                        .timeout(std::time::Duration::from_secs(2));
                    if let Some(ref token) = chain_token {
                        req = req.bearer_auth(token);
                    }
                    match req.send().await {
                        Ok(r) if r.status().is_success() => {
                            info!(camera_id = %cam_id, "Q8: alarm verdict published to nuclear-chain");
                        }
                        Ok(r) => {
                            warn!(status = %r.status(), camera_id = %cam_id, "Q8: nuclear-chain /v1/events non-success (non-blocking)");
                        }
                        Err(e) => {
                            warn!(error = %e, "Q8: nuclear-chain unreachable — chain publish skipped (non-blocking)");
                        }
                    }
                });
            } else {
                tracing::debug!("Q8: CHAIN_ENABLED=true but NUCLEAR_CHAIN_URL not set — chain publish skipped");
            }
        }
    }

    // ── 3. Fortress mesh publish ────────────────────────────────────────────────
    //
    // 3a. Q2: Enforced sentinelle.alarm.verdict stream event (always attempted).
    //     POSTs to FORTRESS_URL/v1/stream regardless of mesh_enabled flag so that
    //     La Rivière always receives the canonical verdict record.
    //     FORTRESS_URL and FORTRESS_API_TOKEN are read at event time (not cached at
    //     startup) so hot env-var changes in Docker / systemd take effect immediately.
    {
        let http = state.http.clone();
        let cam_id = event.camera_id.clone();
        let verdict = alarm.level.to_string();
        let confidence = alarm.danger_score as f32;
        let ts = chrono::Utc::now().to_rfc3339();
        tokio::spawn(async move {
            let fortress_url = std::env::var("FORTRESS_URL")
                .unwrap_or_else(|_| "http://127.0.0.1:7700".to_string());
            let api_token = std::env::var("FORTRESS_API_TOKEN")
                .unwrap_or_default();
            let payload = serde_json::json!({
                "type": "sentinelle.alarm.verdict",
                "camera_id": cam_id,
                "verdict": verdict,
                "confidence": confidence,
                "ts": ts,
            });
            let token = api_token.trim();
            let mut req = http
                .post(format!("{fortress_url}/v1/stream"))
                .json(&payload)
                .timeout(std::time::Duration::from_millis(500));
            if !token.is_empty() {
                req = req.bearer_auth(token);
            }
            let result = req.send().await;
            match result {
                Ok(r) if r.status().is_success() => {
                    tracing::debug!(camera_id = %cam_id, "sentinelle.alarm.verdict published to Fortress stream");
                }
                Ok(r) => {
                    warn!(status = %r.status(), camera_id = %cam_id, "Fortress /v1/stream non-success (non-blocking)");
                }
                Err(e) => {
                    warn!(error = %e, "Fortress /v1/stream unreachable — verdict not published (non-blocking)");
                }
            }
        });
    }

    // 3b. Legacy Fortress mesh publish (deep SecurityEvent shape).
    //     Retained for backward compat; can be removed once 3a covers all consumers.
    if state.fortress_enabled {
        let triad = AffectTriad::from_alarm_event(&alarm);
        let decision = alarm.level.to_string();
        let fortress_url = state.nk.config().fortress_url().to_string();
        let api_token = state.nk.config().fortress_token().unwrap_or("").to_string();
        let alarm_clone = alarm.clone();
        tokio::spawn(async move {
            publish_to_mesh(&alarm_clone, &triad, &decision, &fortress_url, &api_token).await;
        });
    }

    // ── 4. SQLite long-term memory (existing) ────────────────────────────────
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
            vc.speak_audio_only(&alert_text, Some("decisive"), Some(&state.alert_lang)).await
        } else {
            None
        }
    } else {
        None
    };

    // ── 5. chain-comms High-alarm notification (SMS and/or Signal) ──────────────
    // Fires only on High alarms when COMMS_URL + COMMS_ALERT_RECIPIENT are set.
    // Uses SMS if Twilio is configured on chain-comms, otherwise falls through to Signal.
    // Non-blocking: a slow/unreachable chain-comms never delays the ingest response.
    if alarm.level == AlarmLevel::High {
        if let (Some(comms_url), Some(recipient)) =
            (&state.comms_url, &state.comms_alert_recipient)
        {
            let http = state.http.clone();
            let comms_url = comms_url.clone();
            let recipient = recipient.clone();
            let api_token = state.comms_api_token.clone().unwrap_or_default();
            let location = event.camera_id.replace('_', " ");
            let score = alarm.danger_score;
            let note = alarm.note.clone();
            let alert_lang = state.alert_lang.clone();

            tokio::spawn(async move {
                let body = match alert_lang.as_str() {
                    "en" => format!(
                        "NUCLEAR ALERT — High danger at {location} (score={score:.2}). {note}"
                    ),
                    "de" => format!(
                        "NUCLEAR ALARM — Hohe Gefahr bei {location} (score={score:.2}). {note}"
                    ),
                    "es" => format!(
                        "NUCLEAR ALERTA — Peligro alto en {location} (score={score:.2}). {note}"
                    ),
                    _ => format!(
                        "NUCLEAR ALERTE — Danger élevé à {location} (score={score:.2}). {note}"
                    ),
                };

                // Try SMS first, then Signal. Both are fire-and-forget; we log but never panic.
                let comms_token = api_token.trim();
                let sms_payload = serde_json::json!({ "to": recipient, "body": body });
                let mut sms_req = http
                    .post(format!("{comms_url}/sms/send"))
                    .json(&sms_payload)
                    .timeout(std::time::Duration::from_secs(5));
                if !comms_token.is_empty() {
                    sms_req = sms_req.bearer_auth(comms_token);
                }
                let sms_result = sms_req.send().await;

                match sms_result {
                    Ok(r) if r.status().is_success() => {
                        info!(recipient = %recipient, "High-alarm SMS sent via chain-comms");
                    }
                    Ok(r) => {
                        // SMS unavailable (Twilio not configured) — try Signal fallback.
                        let status = r.status();
                        tracing::debug!(status = %status, "SMS unavailable, trying Signal");
                        let sig_payload = serde_json::json!({ "recipient": recipient, "message": body });
                        let mut sig_req = http
                            .post(format!("{comms_url}/signal/send"))
                            .json(&sig_payload)
                            .timeout(std::time::Duration::from_secs(5));
                        if !comms_token.is_empty() {
                            sig_req = sig_req.bearer_auth(comms_token);
                        }
                        let sig_result = sig_req.send().await;
                        match sig_result {
                            Ok(r) if r.status().is_success() => {
                                info!(recipient = %recipient, "High-alarm Signal message sent via chain-comms");
                            }
                            Ok(r) => warn!(status = %r.status(), "Signal send failed via chain-comms"),
                            Err(e) => warn!(error = %e, "chain-comms Signal send unreachable"),
                        }
                    }
                    Err(e) => warn!(error = %e, "chain-comms SMS send unreachable"),
                }
            });
        }
    }

    // Fire-and-forget actuation (lights / buzzer / arm via MQTT)
    {
        let triad = AffectTriad::from_alarm_event(&alarm);
        let action_str = decide(&triad, alarm.level == AlarmLevel::High).to_string();
        let level_str = alarm.level.to_string();
        let cam_id = event.camera_id.clone();
        tokio::spawn(async move {
            if let Ok(actuator_url) = std::env::var("ACTUATOR_URL") {
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

/// GET /health — nuclear-watch polls this to verify the alarm grader / WebSocket host is alive.
async fn alarm_health() -> (axum::http::StatusCode, Json<serde_json::Value>) {
    (axum::http::StatusCode::OK, Json(serde_json::json!({ "ok": true, "service": "alarm_grader_agent" })))
}

// ── JJ1: Operator feedback endpoint ─────────────────────────────────────────

/// Allowed `feedback` values (must match nuclear-watch `AlarmTimelineView` and any admin UI).
const FEEDBACK_ALLOWED: &[&str] = &["false_alarm", "confirmed", "escalate", "unclear"];

/// POST /feedback — operator annotation on an alarm decision.
///
/// Used by nuclear-watch or admin UI to mark alarms as false_alarm / confirmed.
/// Emits `sentinelle.feedback` to La Rivière for continuous learning:
/// the weekly harvest pipeline uses feedback to adjust alarm thresholds.
#[derive(Debug, Deserialize)]
struct FeedbackRequest {
    alarm_id: String,
    camera_id: String,
    /// "false_alarm" | "confirmed" | "escalate" | "unclear"
    feedback: String,
    #[serde(default)]
    operator: Option<String>,
    #[serde(default)]
    notes: Option<String>,
}

async fn handle_feedback(
    State(state): State<AppState>,
    headers: axum::http::HeaderMap,
    Json(req): Json<FeedbackRequest>,
) -> (axum::http::StatusCode, Json<serde_json::Value>) {
    // Bearer token guard — checked when ALARM_GRADER_FEEDBACK_TOKEN is configured.
    if let Some(ref expected) = state.feedback_token {
        let authorized = headers
            .get(axum::http::header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .map(|v| v == format!("Bearer {expected}"))
            .unwrap_or(false);
        if !authorized {
            warn!(alarm_id = %req.alarm_id, "feedback: unauthorized (missing or wrong token)");
            return (
                axum::http::StatusCode::UNAUTHORIZED,
                Json(serde_json::json!({ "error": "unauthorized" })),
            );
        }
    }

    if !FEEDBACK_ALLOWED.contains(&req.feedback.as_str()) {
        warn!(
            alarm_id = %req.alarm_id,
            feedback = %req.feedback,
            "feedback: invalid feedback value"
        );
        return (
            axum::http::StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": "invalid_feedback",
                "allowed": FEEDBACK_ALLOWED,
            })),
        );
    }

    let ts = Utc::now().timestamp_millis() as u64;
    let is_false = req.feedback == "false_alarm";

    // Record to local SQLite (existing false_alarm_log table)
    {
        let mem = state.memory.lock().await;
        if let Err(e) = mem.record_false_alarm(&req.alarm_id, is_false, req.notes.as_deref().unwrap_or("")) {
            warn!(error = %e, "feedback: record_false_alarm failed");
        }
    }

    // JJ1: Emit sentinelle.feedback to La Rivière (fire-and-forget)
    let http = state.http.clone();
    let alarm_id = req.alarm_id.clone();
    let camera_id = req.camera_id.clone();
    let feedback = req.feedback.clone();
    let operator = req.operator.clone();
    let notes = req.notes.clone();
    tokio::spawn(async move {
        riviere::emit_sentinelle_feedback(&http, riviere::SentinelleFeedbackPayload {
            alarm_id,
            camera_id,
            feedback,
            operator,
            notes,
            ts,
        }).await;
    });

    info!(alarm_id = %req.alarm_id, feedback = %req.feedback, "operator feedback recorded");
    (axum::http::StatusCode::OK, Json(serde_json::json!({ "ok": true, "alarm_id": req.alarm_id })))
}

/// Route a question through penny-brain via nuclear-sdk.
async fn query_penny(nk: &NuclearClient, question: &str) -> Option<String> {
    match nk.penny().route(question).await {
        Ok(resp) => {
            tracing::debug!(
                model = %resp.model_used,
                tier = resp.tier,
                "penny-brain routed alarm assessment"
            );
            let text = resp.response.trim().to_string();
            if text.is_empty() { None } else { Some(text) }
        }
        Err(e) => {
            tracing::warn!(error = %e, "penny-brain request failed");
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

// ── JJ6: Depth-enhanced alarm scoring ────────────────────────────────────────

/// Adjust a raw danger score using LiDAR depth context from nuclear-scout.
///
/// Returns `(adjusted_score, suppression_reason)`.
/// If `suppression_reason` is `Some`, the alarm should be suppressed (score = 0.0).
///
/// Rules applied in priority order:
/// 1. All blobs height < 0.5m → auto-suppress (cat/pet, not a person)
/// 2. Fall detected → always Critical (score = 1.0)
/// 3. Zone amplification: intimate (< 0.45m) +20%, projected (> 3.6m) –15%
/// 4. Single occupant in intimate zone → additional +10%
fn depth_adjust_score(
    base_score: f32,
    depth: &nuclear_eye::DepthContext,
) -> (f32, Option<String>) {
    let mut score = base_score;

    // Rule 1: All blobs height < 0.5m → cat/pet, auto-suppress.
    if let Some(ref blobs) = depth.blobs {
        if !blobs.is_empty() && blobs.iter().all(|b| b.height < 0.5) {
            return (
                0.0,
                Some("auto-suppressed: all blobs height < 0.5m (cat/pet)".into()),
            );
        }
    }

    // Rule 2: Fall detected → Critical regardless of zone.
    if depth.fall_detected == Some(true) {
        return (1.0, None);
    }

    // Rule 3: Zone-based amplitude adjustment.
    match depth.alert_zone.as_deref() {
        Some("intimate") => score = (score * 1.2).min(1.0),   // < 0.45m: amplify
        Some("projected") => score *= 0.85,                    // > 3.6m: attenuate
        _ => {}
    }

    // Rule 4: Single occupant in intimate zone adds extra urgency.
    if let (Some(count), Some("intimate")) =
        (depth.occupant_count, depth.alert_zone.as_deref())
    {
        if count == 1 {
            score = (score * 1.1).min(1.0);
        }
    }

    (score, None)
}

// ── Sentinelle perceptual risk scorer ────────────────────────────────────────

/// Multimodal risk signal fused from face, voice, and gesture perception.
///
/// Produced by [`compute_perceptual_risk`] when at least 2 modalities are present.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PerceptualRisk {
    pub score: f32,         // 0.0–1.0
    pub alert: bool,        // score > 0.7
    pub face_contrib: f32,
    pub voice_contrib: f32,
    pub gesture_contrib: f32,
}

/// Fuse face/voice/gesture signals into a Sentinelle risk score.
///
/// Inputs (all optional, 0.0 if absent):
///   face_negative:   (-valence + 1) / 2 × confidence  (from FER model)
///   voice_agitated:  sqrt(arousal+1/2 × neg_valence+1/2) × confidence
///   gesture_threat:  pre-scaled 0.0–1.0 from perceive (intent weights include
///   `fast_approach` / `hands_raised` for P4-7 Scout→appliance mapping; see `perceive_service` / `gesture_pose_mapping.py`).
///
/// Returns None if fewer than 2 modalities are present.
pub fn compute_perceptual_risk(
    face_negative: Option<f32>,
    voice_agitated: Option<f32>,
    gesture_threat: Option<f32>,
) -> Option<PerceptualRisk> {
    let mut n = 0u32;
    let fc = face_negative.inspect(|_| n += 1).unwrap_or(0.0);
    let vc = voice_agitated.inspect(|_| n += 1).unwrap_or(0.0);
    let gc = gesture_threat.inspect(|_| n += 1).unwrap_or(0.0);
    if n < 2 { return None; }
    let score = (0.4 * fc + 0.3 * vc + 0.3 * gc).clamp(0.0, 1.0);
    Some(PerceptualRisk {
        score,
        alert: score > 0.7,
        face_contrib:    (fc * 0.4 * 10000.0).round() / 10000.0,
        voice_contrib:   (vc * 0.3 * 10000.0).round() / 10000.0,
        gesture_contrib: (gc * 0.3 * 10000.0).round() / 10000.0,
    })
}

// TODO: replace with nk.fortress().ingest_security() once SecurityEvent type
//       alignment with the fortress mesh endpoint is confirmed.
async fn publish_to_mesh(alarm: &AlarmEvent, triad: &AffectTriad, decision: &str, fortress_url: &str, api_token: &str) {
    let client = reqwest::Client::new();
    let payload = serde_json::json!({
        "alarm": alarm,
        "triad": triad,
        "decision": decision,
        "context": alarm.vlm_caption.as_deref().unwrap_or_default(),
        "vision_source": "FastVLM-0.5B",
    });
    let token = api_token.trim();
    let mut req = client
        .post(format!("{}/v1/mesh/security", fortress_url))
        .json(&payload)
        .timeout(std::time::Duration::from_millis(500));
    if !token.is_empty() {
        req = req.bearer_auth(token);
    }
    let result = req.send().await;
    match result {
        Ok(resp) => info!(status = %resp.status(), "published alarm to Fortress mesh"),
        Err(err) => warn!(%err, "Fortress publish failed (non-blocking)"),
    }
}

#[cfg(test)]
mod risk_tests {
    use super::*;
    #[test]
    fn test_risk_alert_triggered() {
        // angry face + attacking gesture should exceed 0.7
        let risk = compute_perceptual_risk(Some(0.9), None, Some(1.0)).unwrap();
        assert!(risk.alert, "angry+attacking should trigger alert");
    }
    #[test]
    fn test_risk_normal_no_alert() {
        let risk = compute_perceptual_risk(Some(0.1), Some(0.1), Some(0.0)).unwrap();
        assert!(!risk.alert, "neutral face+calm voice should not alert");
    }
    #[test]
    fn test_risk_single_modality_returns_none() {
        let risk = compute_perceptual_risk(Some(0.9), None, None);
        assert!(risk.is_none(), "single modality should return None");
    }
}
