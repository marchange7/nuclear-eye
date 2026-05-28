use std::collections::VecDeque;
use std::convert::Infallible;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use chrono::Utc;
use axum::{
    body::Bytes,
    extract::{State, ws::{Message, WebSocket, WebSocketUpgrade}},
    http::StatusCode,
    response::sse::{Event, KeepAlive, Sse},
    routing::{get, post},
    Json, Router,
};
use tokio_stream::wrappers::BroadcastStream;
use tokio_stream::StreamExt as _;
use nuclear_eye::{decide, riviere, AffectTriad, AlarmEvent, AlarmGrader, AlarmLevel, AlarmSummary, ConsulClient, SecurityConfig, VisionEvent};
use nuclear_eye::{audio_threat, voiceprint};
use nuclear_eye::memory::SecurityMemory;
use nuclear_eye::alert::{
    AlertChainEnvelope, AlertConfidence, AlertDegraded, AlertEvent as CanonicalEvent,
    AlertEventSource, AlertEventType, AlertEvidence, AlertReason, AlertRecommendedAction,
    AlertSeverity, DegradedFlag, EvidenceKind, RecommendedActionPrimary, SentinelleAlert,
    SCHEMA_VERSION,
};
use std::collections::BTreeMap;
use nuclear_sdk::NuclearClient;
use serde::{Deserialize, Serialize};
use tokio::sync::{broadcast, Mutex};
use tracing::{info, warn};

// SEN-12: optional PG persistence (feature-gated on `alarm_pg`).
#[cfg(feature = "alarm_pg")]
mod alarm_pg;

const CONSUL_TIMEOUT_MS: u64 = 80;
/// Default AUDIT_LOG_PATH — matches audit.rs constant.
const DEFAULT_AUDIT_LOG_PATH: &str = "/var/log/nuclear-eye/audit.jsonl";
/// Maximum number of alarm IDs kept in the in-memory recent-alarm window (SEN-11).
const RECENT_ALARM_WINDOW: usize = 500;
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
    /// SEN-15 (D-full Phase 2, 2026-05-10) — canonical `SentinelleAlert`
    /// envelope per os/70 §4. Emitted alongside legacy `WatchEvent::Alarm`
    /// when `EMIT_CANONICAL_ALERTS=1`. The web inbox / sentinelle-web
    /// `live-alerts.ts:205` `isSentinelleAlert(record)` check picks this up
    /// and renders the full evidence/confidence/recommended-action panel
    /// (the AlertDetail modal we just landed in sentinelle-web).
    /// nuclear-watch / sentinelle-ios continue to parse the legacy `Alarm`
    /// variant until they migrate.
    /// `serde(tag="type")` flattens the `SentinelleAlert` fields next to
    /// `"type": "sentinelle.alert.canonical"`.
    #[serde(rename = "sentinelle.alert.canonical")]
    AlertCanonical(SentinelleAlert),
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
    /// L2: SSE broadcast for `/debug/alarms` — each graded alarm as JSON.
    debug_tx: Arc<broadcast::Sender<String>>,
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
    /// SEN-11: sliding window of recently generated alarm IDs (last RECENT_ALARM_WINDOW entries).
    /// POST /feedback validates alarm_id against this set to reject stale/forged IDs.
    recent_alarm_ids: Arc<Mutex<VecDeque<String>>>,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    // S-7: fail-closed wrapper probe
    nuclear_eye::wrapper_guard::check_wrapper("alarm-grader-agent").await?;

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
        Err(e) => nuclear_eye::wrapper_guard::handle_wrap_failure("alarm-grader-agent", &e),
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
    let (debug_tx_inner, _) = broadcast::channel::<String>(64);
    let debug_tx = Arc::new(debug_tx_inner);
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

    // SEN-9: validate AUDIT_LOG_PATH parent directory at startup so misconfiguration
    // surfaces immediately rather than silently dropping the first audit entry.
    {
        let audit_path = std::env::var("AUDIT_LOG_PATH")
            .unwrap_or_else(|_| DEFAULT_AUDIT_LOG_PATH.to_string());
        if let Some(parent) = std::path::Path::new(&audit_path).parent() {
            match std::fs::create_dir_all(parent) {
                Ok(()) => info!(path = %audit_path, "audit log directory ready"),
                Err(e) => warn!(
                    path = %audit_path,
                    error = %e,
                    "SEN-9: audit log directory cannot be created — audit writes will silently fail; \
                     set AUDIT_LOG_PATH to a writable path or fix directory permissions"
                ),
            }
        }
    }

    // SEN-12: eager pool init (fail-soft) so first alarm doesn't pay init latency.
    #[cfg(feature = "alarm_pg")]
    if std::env::var("SENTINELLE_PERSIST_ALARMS").unwrap_or_default().trim() == "1" {
        alarm_pg::init_pool().await;
    }

    let state = AppState {
        grader: Arc::new(Mutex::new(grader)),
        consul,
        nk: nk.clone(),
        fortress_enabled,
        memory: memory.clone(),
        watch_tx,
        debug_tx,
        alert_lang,
        http,
        comms_url,
        comms_api_token,
        comms_alert_recipient,
        feedback_token,
        penny_grader_timeout_ms,
        recent_alarm_ids: Arc::new(Mutex::new(VecDeque::with_capacity(RECENT_ALARM_WINDOW))),
    };

    let app = Router::new()
        .route("/ingest", post(ingest))
        .route("/sensor/camera", post(handle_camera_frame))
        .route("/feedback", post(handle_feedback))
        .route("/summary", get(summary))
        .route("/ws", get(ws_handler))
        .route("/debug/alarms", get(debug_alarms_sse))
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
    // Phase A: voice signals. An upstream audio tagger (YAMNet) + voiceprint
    // service feed audio-threat events + a captured voice embedding; fold them
    // into the event before grading. Empty/None here = no-op until that source
    // is wired (audio agent), matching the perceive bypass pattern.
    let event = apply_audio_threat(event, &[]);
    let event = apply_voiceprint(event, None, &[]);

    // Local grading is synchronous and fast — do it under the lock, then
    // release before any I/O so we never block other ingest calls.
    let mut alarm = {
        let mut grader = state.grader.lock().await;
        grader.grade_event(&event)
    };

    // SEN-11: register this alarm_id in the recent window so POST /feedback can
    // validate it. Evict oldest entry when the window is full.
    {
        let mut ids = state.recent_alarm_ids.lock().await;
        if ids.len() == RECENT_ALARM_WINDOW {
            ids.pop_front();
        }
        ids.push_back(alarm.alarm_id.clone());
    }

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

    // SEN-13: Perceptual risk fusion — face/voice/gesture multi-modal amplification.
    //
    // Applied after JJ6 depth adjustment so both scoring layers are visible.
    // When perceptual_risk fires (score > 0.7) AND the perceptual score exceeds
    // the current danger_score, the blended score re-derives the alarm level and
    // syncs the hysteresis window so future decisions reflect the true threat level.
    if let Some(risk) = compute_perceptual_risk(
        event.face_negative,
        event.voice_agitated,
        event.gesture_threat,
    ) {
        let pr_score = risk.score as f64;
        if risk.alert && pr_score > alarm.danger_score {
            let prev_level = alarm.level.clone();
            alarm.danger_score = pr_score;
            {
                let mut grader = state.grader.lock().await;
                alarm.level = grader.map_danger_to_level(pr_score);
                // Keep hysteresis window consistent with the fused result.
                if let Some(last) = grader.recent_events.back_mut() {
                    last.level = alarm.level.clone();
                    last.danger_score = alarm.danger_score;
                }
            }
            if alarm.level != prev_level {
                tracing::info!(
                    event_id = %event.event_id,
                    prev_level = %prev_level,
                    new_level = %alarm.level,
                    pr_score,
                    "SEN-13: perceptual risk fusion escalated alarm level"
                );
            }
            alarm.note.push_str(&format!(
                " [perceptual_risk={:.2} face={:.2} voice={:.2} gesture={:.2}]",
                risk.score, risk.face_contrib, risk.voice_contrib, risk.gesture_contrib,
            ));
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

        // L2: broadcast to /debug/alarms SSE (fire-and-forget)
        if state.debug_tx.receiver_count() > 0 {
            let debug_evt = serde_json::json!({
                "ts": nuclear_eye::now_ms(),
                "camera_id": cam_id,
                "behavior": behavior,
                "level": verdict,
                "score": conf,
                "triad": {
                    "judgement":     triad_a.judgement,
                    "doubt":         triad_a.doubt,
                    "determination": triad_a.determination,
                },
                "decision": action_s,
            });
            let _ = state.debug_tx.send(debug_evt.to_string());
        }

        let cam_id2   = cam_id.clone();
        let behavior2 = behavior.clone();
        let verdict2  = verdict.clone();
        let action_s2 = action_s.clone();
        tokio::task::spawn_blocking(move || {
            nuclear_eye::audit::log_decision(&cam_id2, &behavior2, &verdict2, conf, &action_s2);
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
                .unwrap_or_else(|_| "http://127.0.0.1:7700".to_string())
                .trim_end_matches('/')
                .to_string();
            let api_token = std::env::var("FORTRESS_API_TOKEN").unwrap_or_default();
            let service_token = std::env::var("NUCLEAR_SERVICE_TOKEN").unwrap_or_default();
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
            let nuclear_token = service_token.trim();
            // Fortress K-1: KERNEL_REQUIRE_TENANT_HEADER=1 rejects requests
            // missing X-Tenant-Id. Default to nil UUID for Pass 1a/1b soak.
            let tenant_id = std::env::var("FORTRESS_TENANT_ID").unwrap_or_default();
            let tenant_id = tenant_id.trim();
            let tenant_id = if tenant_id.is_empty() {
                "00000000-0000-0000-0000-000000000000"
            } else {
                tenant_id
            };
            for agent in &["arianne", "emile"] {
                let url = format!("{fortress_url}/v1/agents/{agent}/memory");
                let mut req = http
                    .post(&url)
                    .header("X-Tenant-Id", tenant_id)
                    .json(&payload)
                    .timeout(std::time::Duration::from_millis(500));
                if !token.is_empty() {
                    req = req.bearer_auth(token);
                }
                if !nuclear_token.is_empty() {
                    req = req.header("X-Nuclear-Token", nuclear_token);
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

    // 2a. Alarm broadcast — SEN-15 (D-full, 2026-05-10) introduces a flag
    // to switch between the legacy minimal `WatchEvent::Alarm` and the
    // canonical `SentinelleAlert` envelope per os/70 §4.
    //
    //   EMIT_CANONICAL_ALERTS unset / 0 → legacy only (status quo;
    //                                     nuclear-watch + sentinelle-ios
    //                                     parsers stay valid).
    //   EMIT_CANONICAL_ALERTS=1         → canonical only. The web inbox
    //                                     `live-alerts.ts:205`
    //                                     `isSentinelleAlert(record)` test
    //                                     picks it up and renders the full
    //                                     AlertDetail modal. nuclear-watch /
    //                                     iOS need to migrate before flipping
    //                                     this on in their environments.
    //   EMIT_CANONICAL_ALERTS=2         → both (DEV ONLY). Web shows duplicate
    //                                     rows because the two messages have
    //                                     different ids — useful for diffing.
    //
    // Phase 3: when canonical is emitted and SENTINELLE_GATEWAY_INGEST_URL
    // is set, the alert is also fire-and-forget POSTed to the gateway for
    // chain signing per os/74 (the gateway re-broadcasts with `chain.signed=true`).
    let canonical_mode = std::env::var("EMIT_CANONICAL_ALERTS")
        .ok()
        .map(|v| v.trim().to_ascii_lowercase())
        .unwrap_or_default();
    let emit_canonical = matches!(canonical_mode.as_str(), "1" | "true" | "2");
    let emit_legacy    = canonical_mode != "1" && canonical_mode != "true";

    if emit_legacy {
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
    }

    if emit_canonical && alarm.level != AlarmLevel::None {
        let chain_enabled_local = std::env::var("CHAIN_ENABLED")
            .map(|v| v.trim().eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let canonical = build_canonical_alert(
            &event,
            &alarm,
            consul_note.as_deref(),
            !watch_alarm_degraded,
            chain_enabled_local,
        );

        // L4: when SENTINELLE_GATEWAY_INGEST_URL is set, the gateway will
        // re-broadcast the chain-SIGNED canonical envelope to /ws/alerts
        // after Phase 3 publish. Skipping the local pre-sign emit here
        // halves WS traffic per alert and avoids the dedup-by-alert_id
        // race on the web side. When the env is unset (no gateway hop),
        // we keep emitting locally so /ws/alerts subscribers still see
        // the unsigned canonical (the only path they have).
        let gateway_will_rebroadcast = std::env::var("SENTINELLE_GATEWAY_INGEST_URL")
            .ok()
            .map(|s| !s.trim().is_empty())
            .unwrap_or(false);
        if !gateway_will_rebroadcast {
            if let Ok(json) = serde_json::to_string(&WatchEvent::AlertCanonical(canonical.clone())) {
                let _ = state.watch_tx.send(json);
            }
        }

        // Phase 3: fire-and-forget gateway ingest for chain signing.
        publish_canonical_to_gateway(state.http.clone(), canonical);
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
    // When CHAIN_ENABLED=true and NUCLEAR_CHAIN_COMMS_URL is set, POST the alarm verdict
    // to nuclear-chain /v1/events in addition to the existing WebSocket broadcast.
    // Dual-publish — existing WS path is always active.
    // Fire-and-forget: chain publish never delays ingest or blocks the caller.
    {
        let chain_enabled = std::env::var("CHAIN_ENABLED")
            .map(|v| v.trim().eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let chain_url = std::env::var("NUCLEAR_CHAIN_COMMS_URL")
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
                tracing::debug!("Q8: CHAIN_ENABLED=true but NUCLEAR_CHAIN_COMMS_URL not set — chain publish skipped");
            }
        }
    }

    // ── 3. Fortress mesh publish ────────────────────────────────────────────────
    //
    // 3a. Q2: Enforced sentinelle.alarm.verdict stream event (always attempted).
    //     POSTs to FORTRESS_URL/v1/stream/event regardless of mesh_enabled flag so that
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
                .unwrap_or_else(|_| "http://127.0.0.1:7700".to_string())
                .trim_end_matches('/')
                .to_string();
            let api_token = std::env::var("FORTRESS_API_TOKEN")
                .unwrap_or_default();
            let service_token = std::env::var("NUCLEAR_SERVICE_TOKEN").unwrap_or_default();
            let payload = serde_json::json!({
                "agent_id": "nuclear-eye",
                "surface": "sentinelle",
                "event_type": "internal",
                "content": "sentinelle.alarm.verdict",
                "context": {
                    "camera_id": cam_id,
                    "verdict": verdict,
                    "confidence": confidence,
                    "ts": ts,
                },
                "schema_version": 1,
            });
            let token = api_token.trim();
            let nuclear_token = service_token.trim();
            // Fortress K-1 tenant injection (see /v1/agents/*/memory site above).
            let tenant_id = std::env::var("FORTRESS_TENANT_ID").unwrap_or_default();
            let tenant_id = tenant_id.trim();
            let tenant_id = if tenant_id.is_empty() {
                "00000000-0000-0000-0000-000000000000"
            } else {
                tenant_id
            };
            let mut req = http
                .post(format!("{fortress_url}/v1/stream/event"))
                .header("X-Tenant-Id", tenant_id)
                .json(&payload)
                .timeout(std::time::Duration::from_millis(500));
            if !token.is_empty() {
                req = req.bearer_auth(token);
            }
            if !nuclear_token.is_empty() {
                req = req.header("X-Nuclear-Token", nuclear_token);
            }
            let result = req.send().await;
            match result {
                Ok(r) if r.status().is_success() => {
                    tracing::debug!(camera_id = %cam_id, "sentinelle.alarm.verdict published to Fortress stream");
                }
                Ok(r) => {
                    warn!(status = %r.status(), camera_id = %cam_id, "Fortress /v1/stream/event non-success (non-blocking)");
                }
                Err(e) => {
                    warn!(error = %e, "Fortress /v1/stream/event unreachable — verdict not published (non-blocking)");
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

    // ── SEN-12: PG persistence (fire-and-forget, opt-in via SENTINELLE_PERSIST_ALARMS=1) ──
    #[cfg(feature = "alarm_pg")]
    {
        let alarm_clone = alarm.clone();
        tokio::spawn(async move {
            if let Some(pool) = alarm_pg::get_pool().await {
                if let Err(e) = alarm_pg::insert_alarm(pool, &alarm_clone).await {
                    tracing::warn!(error = %e, "SEN-12: PG insert failed (non-blocking)");
                }
            }
        });
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

/// GET /debug/alarms — L2 SSE stream of every graded alarm with AffectTriad + decision.
///
/// Each event:
/// ```json
/// {"ts":…,"camera_id":"…","behavior":"…","level":"High","score":0.82,
///  "triad":{"judgement":0.54,"doubt":0.37,"determination":0.71},"decision":"escalate"}
/// ```
async fn debug_alarms_sse(
    State(state): State<AppState>,
) -> Sse<impl tokio_stream::Stream<Item = Result<Event, Infallible>>> {
    let rx = state.debug_tx.subscribe();
    let stream = BroadcastStream::new(rx).filter_map(|msg| match msg {
        Ok(json) => Some(Ok(Event::default().data(json))),
        Err(_)   => None,
    });
    Sse::new(stream).keep_alive(KeepAlive::default())
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
    body: Bytes,
) -> (StatusCode, Json<serde_json::Value>) {
    // Bearer token guard — runs BEFORE body deserialization so unauthenticated
    // requests always get 401, never 422 from missing required fields.
    if let Some(ref expected) = state.feedback_token {
        let authorized = headers
            .get(axum::http::header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .map(|v| v == format!("Bearer {expected}"))
            .unwrap_or(false);
        if !authorized {
            warn!("feedback: unauthorized (missing or wrong token)");
            return (
                StatusCode::UNAUTHORIZED,
                Json(serde_json::json!({ "error": "unauthorized" })),
            );
        }
    }
    let req: FeedbackRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => return (StatusCode::UNPROCESSABLE_ENTITY, Json(serde_json::json!({ "error": e.to_string() }))),
    };

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

    // SEN-11: reject alarm_ids that were not generated by this process instance.
    // Guards against stale replays and forged IDs. Alarms outside the last
    // RECENT_ALARM_WINDOW events are not tracked (process restart clears the
    // window); in those cases the caller should re-query the audit log.
    {
        let ids = state.recent_alarm_ids.lock().await;
        if !ids.contains(&req.alarm_id) {
            warn!(alarm_id = %req.alarm_id, "feedback: unknown alarm_id rejected");
            return (
                axum::http::StatusCode::UNPROCESSABLE_ENTITY,
                Json(serde_json::json!({
                    "error": "unknown_alarm_id",
                    "detail": "alarm_id not found in recent alarm window; \
                               it may be too old or fabricated",
                })),
            );
        }
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
///
/// Score is normalized by the sum of present-modality weights so that
/// 2-modality inputs are judged on the same 0.0–1.0 scale as 3-modality ones.
pub fn compute_perceptual_risk(
    face_negative: Option<f32>,
    voice_agitated: Option<f32>,
    gesture_threat: Option<f32>,
) -> Option<PerceptualRisk> {
    const W_FACE: f32    = 0.4;
    const W_VOICE: f32   = 0.3;
    const W_GESTURE: f32 = 0.3;

    let fc = face_negative.unwrap_or(f32::NAN);
    let vc = voice_agitated.unwrap_or(f32::NAN);
    let gc = gesture_threat.unwrap_or(f32::NAN);

    // Accumulate only present modalities.
    let mut weighted_sum = 0.0f32;
    let mut weight_total = 0.0f32;
    let mut n = 0u32;
    if face_negative.is_some()    { weighted_sum += W_FACE    * fc; weight_total += W_FACE;    n += 1; }
    if voice_agitated.is_some()   { weighted_sum += W_VOICE   * vc; weight_total += W_VOICE;   n += 1; }
    if gesture_threat.is_some()   { weighted_sum += W_GESTURE * gc; weight_total += W_GESTURE; n += 1; }

    if n < 2 { return None; }

    // Normalize so absent modalities do not suppress the score.
    let score = (weighted_sum / weight_total).clamp(0.0, 1.0);
    let fc = if face_negative.is_some()  { fc } else { 0.0 };
    let vc = if voice_agitated.is_some() { vc } else { 0.0 };
    let gc = if gesture_threat.is_some() { gc } else { 0.0 };
    Some(PerceptualRisk {
        score,
        alert: score > 0.7,
        face_contrib:    (fc * W_FACE    * 10000.0).round() / 10000.0,
        voice_contrib:   (vc * W_VOICE   * 10000.0).round() / 10000.0,
        gesture_contrib: (gc * W_GESTURE * 10000.0).round() / 10000.0,
    })
}

// SEN-9:  DONE — audit log directory validated at startup (main).
// SEN-11: DONE — alarm_id validated against recent_alarm_ids window in handle_feedback.
// SEN-13: DONE — perceptual_risk fusion wired into process_event after JJ6 depth adjustment;
//                VisionEvent carries face_negative / voice_agitated / gesture_threat fields.
// TODO(SEN-6): wrapper fail-closed — alarm_grader_agent runs unguarded when nuclear-wrapper
//              is unreachable; fail-closed mode required for production — see os/PLAN2.md §8
// SEN-12 closed: optional PG persist via SENTINELLE_PERSIST_ALARMS=1 + DATABASE_URL.
//               See src/bin/alarm_pg.rs. acknowledged_at / acknowledged_by deferred to SEN-12.2.
// TODO: replace with nk.fortress().ingest_security() once SecurityEvent type
//       alignment with the fortress mesh endpoint is confirmed.
async fn publish_to_mesh(alarm: &AlarmEvent, triad: &AffectTriad, decision: &str, fortress_url: &str, api_token: &str) {
    let client = reqwest::Client::new();
    let fortress_url = fortress_url.trim_end_matches('/');
    let payload = serde_json::json!({
        "alarm": alarm,
        "triad": triad,
        "decision": decision,
        "context": alarm.vlm_caption.as_deref().unwrap_or_default(),
        "vision_source": "FastVLM-0.5B",
    });
    let token = api_token.trim();
    let service_token = std::env::var("NUCLEAR_SERVICE_TOKEN").unwrap_or_default();
    let nuclear_token = service_token.trim();
    // Fortress K-1: KERNEL_REQUIRE_TENANT_HEADER=1 rejects requests missing
    // X-Tenant-Id. Default to nil UUID so Pass 1a/1b soak passes; operator
    // can set FORTRESS_TENANT_ID to a real tenant when Pass 1c lands.
    let tenant_id = std::env::var("FORTRESS_TENANT_ID")
        .unwrap_or_default();
    let tenant_id = tenant_id.trim();
    let tenant_id = if tenant_id.is_empty() {
        "00000000-0000-0000-0000-000000000000"
    } else {
        tenant_id
    };
    let mut req = client
        .post(format!("{fortress_url}/v1/mesh/security"))
        .header("X-Tenant-Id", tenant_id)
        .json(&payload)
        .timeout(std::time::Duration::from_millis(500));
    if !token.is_empty() {
        req = req.bearer_auth(token);
    }
    if !nuclear_token.is_empty() {
        req = req.header("X-Nuclear-Token", nuclear_token);
    }
    let result = req.send().await;
    match result {
        Ok(resp) => info!(status = %resp.status(), "published alarm to Fortress mesh"),
        Err(err) => warn!(%err, "Fortress publish failed (non-blocking)"),
    }
}

// ── SEN-15 D-full Phase 2: canonical SentinelleAlert builder ─────────────────
//
// Builds a `SentinelleAlert` per os/70 §4 from grader-side state. Pre-sign:
// `chain.signed = false`, hashes null. Phase 3 (`publish_canonical_to_gateway`)
// POSTs this to the gateway `/api/alerts/ingest` for chain signing if the
// SENTINELLE_GATEWAY_INGEST_URL env is set. Otherwise the unsigned envelope
// rides the WebSocket directly — web inbox renders fine but `chain.signed`
// stays false (visible as ⚠ unsigned in the AlertDetail modal).

/// Map our internal `AlarmLevel` plus danger score to canonical
/// `AlertSeverity` per os/70. The only non-trivial cut is High vs Critical
/// — score >= 0.85 within High level escalates to critical, which gates
/// `requires_biometric_auth` and `dispatch_emergency` in the recommended
/// action.
fn map_severity(level: &AlarmLevel, score: f64) -> AlertSeverity {
    match level {
        AlarmLevel::None => AlertSeverity::Info,
        AlarmLevel::Low => AlertSeverity::Low,
        AlarmLevel::Medium => AlertSeverity::Medium,
        AlarmLevel::High if score >= 0.85 => AlertSeverity::Critical,
        AlarmLevel::High => AlertSeverity::High,
    }
}

/// Map a vlm-derived behavior token + caption keywords onto an
/// `AlertEventType` from the os/70 controlled vocabulary.
///
/// The current 9-value enum doesn't cover behavioral detections (weapons,
/// fighting, loitering); these collapse to `MotionInRestrictedZone` with
/// the original behavior preserved as the alert `subtype` for downstream
/// fidelity. When the schema grows a `behavioral_threat` event type, this
/// mapping is the place to update.
fn map_event_type(behavior: &str, caption: Option<&str>) -> AlertEventType {
    let lc_caption = caption.unwrap_or("").to_lowercase();
    if lc_caption.contains("intruder")
        || lc_caption.contains("breaking")
        || lc_caption.contains("forced")
    {
        return AlertEventType::PerimeterBreach;
    }
    if lc_caption.contains("smoke") || lc_caption.contains("fire") {
        return AlertEventType::FireOrSmoke;
    }
    if lc_caption.contains("glass") && lc_caption.contains("break") {
        return AlertEventType::GlassBreak;
    }
    match behavior {
        "no_activity" | "vehicle_present" | "passby" => AlertEventType::MotionInRestrictedZone,
        // weapon_detected / fighting / running / loitering / approaching → motion+subtype
        _ => AlertEventType::MotionInRestrictedZone,
    }
}

/// Map severity + behavior to recommended action (os/70 §4.6).
fn map_recommended_action(
    severity: AlertSeverity,
    person_known: bool,
) -> AlertRecommendedAction {
    let primary = match severity {
        AlertSeverity::Critical => RecommendedActionPrimary::DispatchEmergency,
        AlertSeverity::High if person_known => RecommendedActionPrimary::NotifyOperator,
        AlertSeverity::High => RecommendedActionPrimary::ArmEscalation,
        AlertSeverity::Medium => RecommendedActionPrimary::NotifyUser,
        AlertSeverity::Low => RecommendedActionPrimary::SilentLog,
        AlertSeverity::Info => RecommendedActionPrimary::Defer,
    };
    AlertRecommendedAction {
        primary,
        secondary: vec![],
        requires_biometric_auth: Some(matches!(primary, RecommendedActionPrimary::DispatchEmergency)),
        requires_operator_ack: Some(matches!(
            severity,
            AlertSeverity::High | AlertSeverity::Critical
        )),
        // `reversible_until` would mark a soft-cancel window; left None
        // until the policy layer (os/70 §5 stage 5) decides per-event.
        reversible_until: None,
    }
}

/// Build a canonical `SentinelleAlert` from the grader's local context
/// post-grade. Tenant id is read from `NUCLEAR_TENANT_ID` (falls back to
/// the well-known dev UUID) — gateway re-stamps anyway on ingest.
#[allow(clippy::too_many_arguments)]
fn build_canonical_alert(
    event: &VisionEvent,
    alarm: &AlarmEvent,
    consul_note: Option<&str>,
    penny_applied: bool,
    chain_enabled: bool,
) -> SentinelleAlert {
    let severity = map_severity(&alarm.level, alarm.danger_score);
    let event_type = map_event_type(&event.behavior, event.vlm_caption.as_deref());

    // vision_agent stamps perimeter + watchlist results into extra_tags:
    //   "zone:<name>"            → canonical source.zone_id (spatial context)
    //   "watchlist:<status>:..." → IdentityMatch evidence extra.watch_status
    //                              (authorized=family / watch / offender=escalate)
    let zone_id: Option<String> = event
        .extra_tags
        .iter()
        .find_map(|t| t.strip_prefix("zone:").map(|z| z.to_string()));
    let watch_status: Option<String> = event
        .extra_tags
        .iter()
        .find_map(|t| t.strip_prefix("watchlist:").and_then(|r| r.split(':').next().map(str::to_string)));

    // Confidence components: vision (always) + face/voice/gesture when
    // perceive_service populated them (Phase 1).
    let mut components: BTreeMap<String, f32> = BTreeMap::new();
    components.insert("vision".to_string(), event.confidence as f32);
    if let Some(v) = event.face_negative   { components.insert("face".to_string(),    v); }
    if let Some(v) = event.voice_agitated  { components.insert("voice".to_string(),   v); }
    if let Some(v) = event.gesture_threat  { components.insert("gesture".to_string(), v); }

    let mut method_parts: Vec<&str> = vec!["caption_to_vision_event"];
    if event.face_negative.is_some()
        || event.voice_agitated.is_some()
        || event.gesture_threat.is_some()
    {
        method_parts.push("perceive_service");
    }
    if penny_applied { method_parts.push("penny_l1"); }
    if consul_note.is_some() { method_parts.push("consul"); }
    let method = method_parts.join("+");

    // Evidence list: vision_inference always; identity_match when face_db
    // returned a person_name on the upstream event (face_db ArcFace search
    // result rides on `VisionEvent.person_name` from vision_agent / iphone_sensor_agent);
    // crew_verdict when Consul synthesized.
    let mut evidence: Vec<AlertEvidence> = vec![AlertEvidence {
        kind: EvidenceKind::VisionInference,
        model: Some("nuclear-eye/vision_agent+fastvlm".to_string()),
        label: Some(event.behavior.clone()),
        confidence: alarm.danger_score as f32,
        frame_refs: vec![],
        redactions: vec![],
        extra: BTreeMap::new(),
    }];

    if let Some(name) = event.person_name.as_ref() {
        // N4 (D-full follow-up): face_db ArcFace match surfaced via the
        // upstream VisionEvent.person_name. We don't have the raw match
        // score plumbed through yet — use `event.confidence` (the vision
        // pipeline's own self-report) as a proxy. When face_db match
        // scores are propagated end-to-end, swap to that field and add
        // `extra: { match_id, threshold }` per os/70 evidence schema.
        let mut extra = BTreeMap::new();
        extra.insert(
            "match_via".into(),
            serde_json::Value::String("vision_event.person_name".to_string()),
        );
        // Watchlist taxonomy (family-suppress / offender-escalate) so the
        // operator UI can color the recognized-identity chip by status.
        if let Some(ws) = watch_status.as_ref() {
            extra.insert("watch_status".into(), serde_json::Value::String(ws.clone()));
        }
        // Label encodes status when known ("offender:alice") so SDK consumers
        // that only decode `label` (iOS, per the SDK's minimal AlertEvidence)
        // still get the watchlist status; web reads the structured extra.
        let id_label = match watch_status.as_ref() {
            Some(ws) => format!("{ws}:{name}"),
            None => name.clone(),
        };
        evidence.push(AlertEvidence {
            kind: EvidenceKind::IdentityMatch,
            model: Some("face_db/arcface".to_string()),
            label: Some(id_label),
            confidence: event.confidence as f32,
            frame_refs: vec![],
            redactions: vec![],
            extra,
        });
    }

    if let Some(note) = consul_note {
        let mut extra = BTreeMap::new();
        extra.insert("synthesis".into(), serde_json::Value::String(note.to_string()));
        evidence.push(AlertEvidence {
            kind: EvidenceKind::CrewVerdict,
            model: Some("nuclear-consul".to_string()),
            label: Some("consul deliberation".to_string()),
            // We don't surface a numeric confidence from the consul note
            // string; downstream renderers degrade to "—" when zero.
            confidence: 0.0,
            frame_refs: vec![],
            redactions: vec![],
            extra,
        });
    }

    // Degraded flags. Penny didn't apply → kernel_unreachable. Score
    // window low → low_confidence. When chain is disabled the chain
    // envelope rides as signed=false and we surface chain_unavailable
    // so the UI's degraded badge stays honest.
    let mut flags: Vec<DegradedFlag> = vec![];
    if !penny_applied && alarm.level == AlarmLevel::High {
        flags.push(DegradedFlag::KernelUnreachable);
    }
    if alarm.danger_score > 0.0 && alarm.danger_score < 0.35 {
        flags.push(DegradedFlag::LowConfidence);
    }
    if !chain_enabled {
        flags.push(DegradedFlag::ChainUnavailable);
    }
    // Perception degraded: vision_agent populated event.confidence, but
    // perceive_service didn't return any modality. Caller can't always
    // tell unset-because-disabled vs unset-because-failed; we only flag
    // it when PERCEIVE_URL is set on this host but all modalities are None.
    let perceive_configured = std::env::var("PERCEIVE_URL")
        .ok()
        .map(|s| !s.trim().is_empty())
        .unwrap_or(false);
    if perceive_configured
        && event.face_negative.is_none()
        && event.voice_agitated.is_none()
        && event.gesture_threat.is_none()
    {
        flags.push(DegradedFlag::PerceptionDegraded);
    }

    // Reason: clip summary to schema cap (280); full note rides in
    // detail_markdown so AlertDetail can render the long form.
    let summary_full = if alarm.note.is_empty() {
        format!("{} on {}", event.behavior, event.camera_id)
    } else {
        alarm.note.clone()
    };
    let summary: String = summary_full.chars().take(280).collect();

    let person_known = event.person_name.is_some();

    let tenant_id = std::env::var("NUCLEAR_TENANT_ID")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| "00000000-0000-0000-0000-000000000000".to_string());

    let observed_at = chrono::DateTime::<Utc>::from_timestamp_millis(event.timestamp_ms as i64)
        .unwrap_or_else(Utc::now);

    SentinelleAlert {
        schema_version: SCHEMA_VERSION,
        alert_id: alarm.alarm_id.clone(),
        tenant_id,
        product: "sentinelle".to_string(),
        issued_at: Utc::now(),
        issued_by: "nuclear-eye/alarm_grader_agent".to_string(),
        event: CanonicalEvent {
            kind: event_type,
            subtype: Some(event.behavior.clone()),
            severity,
            source: AlertEventSource {
                camera_id: Some(event.camera_id.clone()),
                sensor_id: None,
                zone_id: zone_id.clone(),
            },
            observed_at,
            duration_ms: None,
        },
        confidence: AlertConfidence {
            overall: alarm.danger_score as f32,
            components,
            method,
        },
        evidence,
        reason: AlertReason {
            summary,
            detail_markdown: event.vlm_caption.clone().or_else(|| Some(alarm.note.clone())),
        },
        degraded: AlertDegraded {
            any: !flags.is_empty(),
            flags,
        },
        recommended_action: map_recommended_action(severity, person_known),
        chain: AlertChainEnvelope {
            // Pre-sign: gateway `/api/alerts/ingest` replaces this on
            // signing. WS-direct path keeps signed=false.
            signed: false,
            chain_hash: None,
            prev_alert_hash: None,
            signer: None,
        },
    }
}

/// Phase 3: fire-and-forget POST to the gateway's `/api/alerts/ingest`
/// for chain signing (os/74) and re-broadcast with the signed envelope.
/// Gated on `SENTINELLE_GATEWAY_INGEST_URL` + `SENTINELLE_GATEWAY_KEY`.
/// Failure is logged at WARN; the WS broadcast already carries the
/// (unsigned) canonical envelope so consumers never go blind.
fn publish_canonical_to_gateway(
    http: reqwest::Client,
    alert: SentinelleAlert,
) {
    let url = match std::env::var("SENTINELLE_GATEWAY_INGEST_URL")
        .ok()
        .and_then(|s| Some(s.trim().to_string()).filter(|v| !v.is_empty()))
    {
        Some(u) => u,
        None => return,
    };
    let key = std::env::var("SENTINELLE_GATEWAY_KEY").unwrap_or_default();
    // Y5 (T-P1-18): inter-service token. Without these headers the gateway
    // logs "Y5: inter-service call missing or invalid service token" once
    // per request; with FORTRESS_REQUIRE_SERVICE_TOKEN=true it hard-401s.
    // The grader reuses the same NUCLEAR_SERVICE_TOKEN it already passes
    // to fortress / kernel.
    let service_token = std::env::var("NUCLEAR_SERVICE_TOKEN").unwrap_or_default();
    let alert_id = alert.alert_id.clone();
    tokio::spawn(async move {
        let mut req = http
            .post(&url)
            .header("X-Sentinelle-Key", key.trim())
            .header("X-Nuclear-Service", "alarm_grader_agent")
            .timeout(Duration::from_millis(800))
            .json(&alert);
        let token = service_token.trim();
        if !token.is_empty() {
            req = req.header("X-Nuclear-Token", token);
        }
        let res = req.send().await;
        match res {
            Ok(r) if r.status().is_success() => {
                info!(alert_id, status = %r.status(), "canonical alert chain-signed via gateway");
            }
            Ok(r) => {
                warn!(alert_id, status = %r.status(), "gateway ingest non-2xx");
            }
            Err(e) => {
                warn!(alert_id, error = %e, "gateway ingest failed (non-blocking)");
            }
        }
    });
}

#[cfg(test)]
mod risk_tests {
    use super::*;
    #[test]
    fn test_risk_alert_triggered() {
        // angry face + attacking gesture should exceed 0.7 after normalization
        let risk = compute_perceptual_risk(Some(0.9), None, Some(1.0)).unwrap();
        assert!(risk.alert, "angry+attacking should trigger alert");
        // score = (0.4*0.9 + 0.3*1.0) / 0.7 ≈ 0.943
        assert!(risk.score > 0.7, "score should be above threshold: {}", risk.score);
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
    #[test]
    fn test_risk_all_three_modalities_not_suppressed() {
        // Full 3-modality path: normalized weight sum = 1.0, no suppression
        let risk = compute_perceptual_risk(Some(0.8), Some(0.8), Some(0.8)).unwrap();
        // score = (0.4*0.8 + 0.3*0.8 + 0.3*0.8) / 1.0 = 0.8
        assert!((risk.score - 0.8).abs() < 0.001, "3-modality score should be 0.8, got {}", risk.score);
        assert!(risk.alert, "all high modalities should trigger alert");
    }
    #[test]
    fn test_recent_alarm_window_eviction() {
        // Ensure the VecDeque window evicts oldest entry when full.
        let mut ids: VecDeque<String> = VecDeque::with_capacity(RECENT_ALARM_WINDOW);
        for i in 0..RECENT_ALARM_WINDOW {
            if ids.len() == RECENT_ALARM_WINDOW { ids.pop_front(); }
            ids.push_back(format!("alarm-{i}"));
        }
        assert_eq!(ids.len(), RECENT_ALARM_WINDOW);
        // Insert one more — alarm-0 should be evicted
        ids.pop_front();
        ids.push_back("alarm-overflow".to_string());
        assert!(!ids.contains(&"alarm-0".to_string()), "oldest alarm should be evicted");
        assert!(ids.contains(&"alarm-overflow".to_string()), "new alarm should be present");
    }
}

// ── Phase A: voice signal helpers ──────────────────────────────────────────

/// Fold YAMNet-style audio-threat events into the event's `voice_agitated`
/// signal (max with any existing value), so scream/glass/gunshot raise the
/// perceptual risk the grader already reads.
fn apply_audio_threat(mut event: VisionEvent, audio: &[audio_threat::AudioEvent]) -> VisionEvent {
    if let Some(va) = audio_threat::to_voice_agitated(audio_threat::threat_score(audio)) {
        event.voice_agitated = Some(event.voice_agitated.map_or(va, |cur| cur.max(va)));
    }
    event
}

/// Match a captured voice embedding against the voice watchlist. On a hit, set
/// `person_name` (if empty) and adjust `risk_score` by the hit's risk delta
/// (Family suppresses, Watch/Offender escalate).
fn apply_voiceprint(
    mut event: VisionEvent,
    voice_emb: Option<&[f32]>,
    watchlist: &[voiceprint::VoiceprintEntry],
) -> VisionEvent {
    if let Some(emb) = voice_emb {
        if let Some(hit) =
            voiceprint::match_voice(emb, watchlist, voiceprint::DEFAULT_VOICE_MATCH_THRESHOLD)
        {
            if event.person_name.as_deref().map_or(true, |s| s.is_empty()) {
                event.person_name = Some(hit.label.clone());
            }
            event.risk_score = (event.risk_score + hit.risk_delta() as f64).clamp(0.0, 1.0);
        }
    }
    event
}

#[cfg(test)]
mod voice_wiring_tests {
    use super::*;
    use nuclear_eye::audio_threat::AudioEvent;
    use nuclear_eye::voiceprint::{VoiceStatus, VoiceprintEntry};

    fn vev() -> VisionEvent {
        VisionEvent {
            event_id: "e1".into(),
            timestamp_ms: 1,
            camera_id: "cam1".into(),
            behavior: "passby".into(),
            risk_score: 0.2,
            stress_level: 0.0,
            confidence: 0.9,
            person_detected: false,
            person_name: None,
            hands_visible: 0,
            object_held: None,
            extra_tags: vec![],
            vlm_caption: None,
            depth_context: None,
            face_negative: None,
            voice_agitated: None,
            gesture_threat: None,
        }
    }

    fn vp(id: &str, status: VoiceStatus, threat: u8, emb: Vec<f32>) -> VoiceprintEntry {
        VoiceprintEntry { id: id.into(), label: id.into(), status, threat_level: threat, embedding: emb }
    }

    #[test]
    fn scream_raises_voice_agitated() {
        let ev = apply_audio_threat(vev(), &[AudioEvent { label: "scream".into(), score: 0.9 }]);
        assert!(ev.voice_agitated.unwrap_or(0.0) > 0.0);
    }

    #[test]
    fn benign_audio_no_false_alarm() {
        let ev = apply_audio_threat(vev(), &[AudioEvent { label: "music".into(), score: 0.9 }]);
        assert!(ev.voice_agitated.unwrap_or(0.0) <= 0.0);
    }

    #[test]
    fn no_audio_unchanged() {
        let ev = apply_audio_threat(vev(), &[]);
        assert!(ev.voice_agitated.unwrap_or(0.0) <= 0.0);
    }

    #[test]
    fn offender_voiceprint_raises_risk_and_names() {
        let emb = vec![1.0, 0.0, 0.0, 0.0];
        let wl = vec![vp("intruder", VoiceStatus::Offender, 80, emb.clone())];
        let ev = apply_voiceprint(vev(), Some(&emb), &wl);
        assert_eq!(ev.person_name.as_deref(), Some("intruder"));
        assert!(ev.risk_score > 0.2);
    }

    #[test]
    fn family_voiceprint_suppresses_risk() {
        let emb = vec![0.0, 1.0, 0.0, 0.0];
        let wl = vec![vp("mum", VoiceStatus::Family, 0, emb.clone())];
        let ev = apply_voiceprint(vev(), Some(&emb), &wl);
        assert!(ev.risk_score < 0.2);
    }

    #[test]
    fn no_embedding_unchanged() {
        let ev = apply_voiceprint(vev(), None, &[]);
        assert!((ev.risk_score - 0.2).abs() < 1e-9 && ev.person_name.is_none());
    }
}
