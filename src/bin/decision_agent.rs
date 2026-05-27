use anyhow::Result;
use axum::{
    extract::State,
    http::StatusCode,
    response::sse::{Event, KeepAlive, Sse},
    routing::{get, post},
    Json, Router,
};
use std::convert::Infallible;
use tokio::sync::broadcast;
use tokio_stream::wrappers::BroadcastStream;
use tokio_stream::StreamExt as _;
use nuclear_eye::{
    decide, AffectTriad, ConsulClient, DecisionAction, SecurityConfig, VisionEvent,
};
use nuclear_eye::behavior::{apply_repetition, RepetitionTracker};
use nuclear_eye::memory::SecurityMemory;
use nuclear_sdk::NuclearClient;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::signal;
use tracing::{info, instrument, warn};

// ── State ──────────────────────────────────────────────────────────────

const CONSUL_TIMEOUT_MS: u64 = 5_000;
const HEALTH_INTERVAL_SECS: u64 = 30;

#[derive(Clone)]
struct AppState {
    safety_risk_threshold: f64,
    consul: ConsulClient,
    memory: Arc<Mutex<SecurityMemory>>,
    nk: NuclearClient,
    /// L2: SSE broadcast for `/debug/decisions`.
    debug_tx: Arc<broadcast::Sender<String>>,
    /// Phase A: repeat-sighting tracker — repeated detections of the same
    /// person/behavior within the window escalate risk toward Alarm.
    repetition: Arc<Mutex<RepetitionTracker>>,
}

// ── Request / Response types ───────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct DecideRequest {
    event: VisionEvent,
    /// Override safety-critical detection (default: derived from risk_score).
    #[serde(default)]
    force_safety: Option<bool>,
}

#[derive(Debug, Serialize)]
struct DecisionResponse {
    event_id: String,
    triad: AffectTriad,
    action: String,
    is_safety_critical: bool,
    dominant_dimension: &'static str,
    consul_synthesis: Option<String>,
    consul_confidence: Option<f64>,
}

#[derive(Debug, Serialize)]
struct HealthResponse {
    status: &'static str,
    consul_ok: bool,
    consul_backend: String,
    decisions_logged: u64,
    buffered_events: u32,
}

#[derive(Debug, Serialize)]
struct ErrorBody {
    error: String,
}

// ── Main ───────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    // S-7: fail-closed wrapper probe
    nuclear_eye::wrapper_guard::check_wrapper("decision-agent").await?;

    // ── Nuclear wrapper — resilience sidecar ────────────────────────────
    match nuclear_wrapper::wrap!(
        node_id      = "decision-agent",
        pg_url       = std::env::var("DATABASE_URL").unwrap_or_default(),
        signal_token = std::env::var("SIGNAL_TOKEN").unwrap_or_default()
    ) {
        Ok(nw) => {
            info!("nuclear-wrapper: armed (tamper, health, discovery)");
            std::mem::forget(nw);
        }
        Err(e) => nuclear_eye::wrapper_guard::handle_wrap_failure("decision-agent", &e),
    }

    let cfg = SecurityConfig::load()?;
    let bind = std::env::var("DECISION_AGENT_BIND")
        .unwrap_or_else(|_| cfg.decision.bind.clone());

    let consul_url = std::env::var("CONSUL_URL")
        .unwrap_or_else(|_| "http://127.0.0.1:7710".to_string());
    let consul = ConsulClient::new(consul_url, CONSUL_TIMEOUT_MS);

    let nk = NuclearClient::from_system()
        .expect("NuclearClient: check FORTRESS_URL env var");

    let memory_path = {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
        format!("{home}/.nuclear-eye/memory.db")
    };
    std::fs::create_dir_all(std::path::Path::new(&memory_path).parent().unwrap())?;
    let memory = Arc::new(Mutex::new(SecurityMemory::open(&memory_path)?));

    let (debug_tx_inner, _) = broadcast::channel::<String>(64);
    let debug_tx = Arc::new(debug_tx_inner);

    let state = AppState {
        safety_risk_threshold: cfg.decision.safety_risk_threshold,
        consul,
        memory: memory.clone(),
        nk: nk.clone(),
        debug_tx,
        // 5-minute window, escalate after the 3rd sighting of the same key.
        repetition: Arc::new(Mutex::new(RepetitionTracker::new(300_000, 3))),
    };

    // ── Background health check via SDK ──────────────────────────────────
    let nk_hc = nk.clone();
    let hc_mem = memory.clone();
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(Duration::from_secs(HEALTH_INTERVAL_SECS)).await;
            let consul_ok = nk_hc.consul().health().await.is_ok();
            let buffered = hc_mem.lock().ok()
                .and_then(|m| m.buffered_count().ok())
                .unwrap_or(0);
            info!(consul_ok, buffered_events = buffered, "decision_agent health_check");
        }
    });

    let app = Router::new()
        .route("/decide", post(handle_decide))
        .route("/debug/decisions", get(debug_decisions_sse))
        .route("/health", get(health))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(&bind).await?;
    info!(bind = %bind, "decision_agent started");

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    info!("decision_agent shut down cleanly");
    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => info!("received Ctrl+C"),
        _ = terminate => info!("received SIGTERM"),
    }
}

// ── Handlers ───────────────────────────────────────────────────────────

async fn health(State(state): State<AppState>) -> Json<HealthResponse> {
    let consul_ok = state.nk.consul().health().await.is_ok();
    let consul_backend = std::env::var("CONSUL_BACKEND")
        .unwrap_or_else(|_| "local".to_string());

    let (decisions_logged, buffered_events): (u64, u32) = state
        .memory
        .lock()
        .map(|m| {
            let decisions = m.decision_count().unwrap_or(0);
            let buffered = m.buffered_count().unwrap_or(0);
            (decisions, buffered)
        })
        .unwrap_or((0, 0));

    let status = if consul_ok || consul_backend == "cloud" { "ok" } else { "degraded" };

    Json(HealthResponse {
        status,
        consul_ok,
        consul_backend,
        decisions_logged,
        buffered_events,
    })
}

#[instrument(skip_all, fields(event_id))]
async fn handle_decide(
    State(state): State<AppState>,
    payload: Result<Json<DecideRequest>, axum::extract::rejection::JsonRejection>,
) -> Result<Json<DecisionResponse>, (StatusCode, Json<ErrorBody>)> {
    let Json(mut req) = payload.map_err(|err| {
        warn!(%err, "bad request");
        (
            StatusCode::BAD_REQUEST,
            Json(ErrorBody { error: err.to_string() }),
        )
    })?;

    tracing::Span::current().record("event_id", req.event.event_id.as_str());

    // Phase A: repetition escalation — repeated sightings of the same person /
    // behavior within the window raise risk BEFORE the triad + safety-critical
    // check, so a returning intruder trends toward Alarm.
    {
        let key = repetition_key(&req.event);
        let mut tracker = state.repetition.lock().unwrap();
        tracker.observe(&key, req.event.timestamp_ms);
        let boost = tracker.repetition_boost(&key, req.event.timestamp_ms);
        if boost > 0.0 {
            req.event.risk_score = apply_repetition(req.event.risk_score, boost);
            info!(key, boost, risk = req.event.risk_score, "repetition escalation applied");
        }
    }

    let triad = AffectTriad::from_vision_event(&req.event);
    let is_safety_critical = req
        .force_safety
        .unwrap_or(req.event.risk_score > state.safety_risk_threshold);
    let action = decide(&triad, is_safety_critical);
    let dominant = triad.dominant();

    // Escalate to Consul on Alarm actions
    let (consul_synthesis, consul_confidence) = if action == DecisionAction::Alarm {
        let question = format!(
            "Security alarm: camera={} behavior={} risk={:.2} stress={:.2} person={:?}",
            req.event.camera_id, req.event.behavior,
            req.event.risk_score, req.event.stress_level, req.event.person_name
        );
        match state.consul.query_async(&question).await {
            Ok(Some(cd)) => {
                info!(synthesis = %cd.decision, confidence = cd.confidence, "consul escalation");
                (Some(cd.decision), Some(cd.confidence))
            }
            Ok(None) => { warn!("consul returned no decision"); (None, None) }
            Err(e) => { warn!(error = %e, "consul unreachable — local decision stands"); (None, None) }
        }
    } else {
        (None, None)
    };

    // Log decision to SQLite
    let ts_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;
    if let Ok(mem) = state.memory.lock() {
        if let Err(e) = mem.record_decision(
            ts_ms,
            &req.event.event_id,
            &req.event.camera_id,
            &action.to_string(),
            is_safety_critical,
            dominant,
            consul_synthesis.as_deref(),
            consul_confidence,
        ) {
            warn!("failed to log decision to SQLite: {e}");
        }
    }

    // Fire-and-forget: feed decision to La Rivière via SDK
    {
        let content = format!(
            "Decide::{} @ {} — {} (J={:.2}, doubt={:.2}, det={:.2}, safety={is_safety_critical})",
            action, req.event.camera_id, req.event.behavior,
            triad.judgement, triad.doubt, triad.determination,
        );
        let nk = state.nk.clone();
        tokio::spawn(async move {
            nuclear_eye::riviere::post_event("nuclear-eye", "camera", &content, &nk).await;
        });
    }

    info!(
        %triad, %action,
        safety = is_safety_critical,
        dominant,
        consul = consul_synthesis.is_some(),
        "decision computed"
    );

    // L2: broadcast to /debug/decisions SSE
    if state.debug_tx.receiver_count() > 0 {
        let debug_evt = serde_json::json!({
            "ts": ts_ms,
            "event_id": req.event.event_id,
            "camera_id": req.event.camera_id,
            "behavior": req.event.behavior,
            "risk_score": req.event.risk_score,
            "triad": {
                "judgement":     triad.judgement,
                "doubt":         triad.doubt,
                "determination": triad.determination,
            },
            "action": action.to_string(),
            "is_safety_critical": is_safety_critical,
            "dominant": dominant,
            "consul": consul_synthesis,
        });
        let _ = state.debug_tx.send(debug_evt.to_string());
    }

    Ok(Json(DecisionResponse {
        event_id: req.event.event_id,
        triad,
        action: action.to_string(),
        is_safety_critical,
        dominant_dimension: dominant,
        consul_synthesis,
        consul_confidence,
    }))
}

/// GET /debug/decisions — L2 SSE stream of each decision with AffectTriad + action.
async fn debug_decisions_sse(
    State(state): State<AppState>,
) -> Sse<impl tokio_stream::Stream<Item = Result<Event, Infallible>>> {
    let rx = state.debug_tx.subscribe();
    let stream = BroadcastStream::new(rx).filter_map(|msg| match msg {
        Ok(json) => Some(Ok(Event::default().data(json))),
        Err(_)   => None,
    });
    Sse::new(stream).keep_alive(KeepAlive::default())
}

/// Build the repetition-tracking key for an event: prefer a recognized person,
/// then the behavior tag (per camera), else the camera id.
fn repetition_key(ev: &VisionEvent) -> String {
    match ev.person_name.as_deref().filter(|s| !s.is_empty()) {
        Some(name) => format!("person:{name}"),
        None if !ev.behavior.is_empty() => format!("behavior:{}@{}", ev.behavior, ev.camera_id),
        None => format!("cam:{}", ev.camera_id),
    }
}

#[cfg(test)]
mod wiring_tests {
    use super::*;

    fn ev(person: Option<&str>, behavior: &str, cam: &str, ts: u64) -> VisionEvent {
        VisionEvent {
            event_id: format!("e-{ts}"),
            timestamp_ms: ts,
            camera_id: cam.to_string(),
            behavior: behavior.to_string(),
            risk_score: 0.2,
            stress_level: 0.0,
            confidence: 0.9,
            person_detected: person.is_some(),
            person_name: person.map(|s| s.to_string()),
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

    #[test]
    fn key_prefers_person_name() {
        assert_eq!(repetition_key(&ev(Some("intruder-A"), "loitering", "cam1", 1)), "person:intruder-A");
    }

    #[test]
    fn key_falls_back_to_behavior() {
        assert_eq!(repetition_key(&ev(None, "loitering", "cam1", 1)), "behavior:loitering@cam1");
    }

    #[test]
    fn key_falls_back_to_camera() {
        assert_eq!(repetition_key(&ev(None, "", "cam1", 1)), "cam:cam1");
    }

    #[test]
    fn repeated_sightings_escalate_risk() {
        let mut t = RepetitionTracker::new(300_000, 3);
        let key = "person:intruder-A";
        t.observe(key, 1000);
        t.observe(key, 2000);
        t.observe(key, 3000);
        let boost = t.repetition_boost(key, 3000);
        assert!(boost > 0.0, "3rd sighting within window should boost");
        let risk = apply_repetition(0.2, boost);
        assert!(risk > 0.2 && risk <= 1.0);
    }

    #[test]
    fn single_sighting_no_escalation() {
        let mut t = RepetitionTracker::new(300_000, 3);
        t.observe("person:x", 1000);
        assert_eq!(t.repetition_boost("person:x", 1000), 0.0);
    }
}
