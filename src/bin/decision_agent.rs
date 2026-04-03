use anyhow::Result;
use axum::{
    extract::State,
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use nuclear_eye::{
    decide, AffectTriad, ConsulClient, DecisionAction, SecurityConfig, VisionEvent,
};
use nuclear_eye::memory::SecurityMemory;
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
    consul_url: String,
    fortress_url: String,
    fortress_api_token: String,
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
        Err(e) => info!("nuclear-wrapper: start failed ({e}) — running unguarded"),
    }

    let cfg = SecurityConfig::load()?;
    let bind = std::env::var("DECISION_AGENT_BIND")
        .unwrap_or_else(|_| cfg.decision.bind.clone());

    let consul_url = std::env::var("CONSUL_URL")
        .unwrap_or_else(|_| "http://127.0.0.1:7710".to_string());
    let consul = ConsulClient::new(consul_url.clone(), CONSUL_TIMEOUT_MS);

    let memory_path = {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
        format!("{home}/.nuclear-eye/memory.db")
    };
    std::fs::create_dir_all(std::path::Path::new(&memory_path).parent().unwrap())?;
    let memory = Arc::new(Mutex::new(SecurityMemory::open(&memory_path)?));

    let fortress_url       = std::env::var("FORTRESS_URL").unwrap_or_else(|_| cfg.fortress_url());
    let fortress_api_token = std::env::var("FORTRESS_API_TOKEN").unwrap_or_default();

    let state = AppState {
        safety_risk_threshold: cfg.decision.safety_risk_threshold,
        consul,
        memory: memory.clone(),
        consul_url: consul_url.clone(),
        fortress_url,
        fortress_api_token,
    };

    // ── Background health check ──────────────────────────────────────────
    let hc_consul_url = format!("{consul_url}/health");
    let hc_mem = memory.clone();
    tokio::spawn(async move {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(2))
            .build()
            .expect("reqwest client");
        loop {
            tokio::time::sleep(Duration::from_secs(HEALTH_INTERVAL_SECS)).await;
            let consul_ok = client.get(&hc_consul_url).send().await
                .map(|r| r.status().is_success())
                .unwrap_or(false);
            let buffered = hc_mem.lock().ok()
                .and_then(|m| m.buffered_count().ok())
                .unwrap_or(0);
            info!(consul_ok, buffered_events = buffered, "decision_agent health_check");
        }
    });

    let app = Router::new()
        .route("/decide", post(handle_decide))
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
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(1))
        .build()
        .expect("reqwest client");

    let consul_ok = client
        .get(format!("{}/health", state.consul_url))
        .send()
        .await
        .map(|r| r.status().is_success())
        .unwrap_or(false);

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
    let Json(req) = payload.map_err(|err| {
        warn!(%err, "bad request");
        (
            StatusCode::BAD_REQUEST,
            Json(ErrorBody { error: err.to_string() }),
        )
    })?;

    tracing::Span::current().record("event_id", &req.event.event_id.as_str());

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

    // Fire-and-forget: feed decision to La Rivière (dual-write, SQLite is the fallback)
    {
        let content = format!(
            "Decide::{} @ {} — {} (J={:.2}, doubt={:.2}, det={:.2}, safety={is_safety_critical})",
            action, req.event.camera_id, req.event.behavior,
            triad.judgement, triad.doubt, triad.determination,
        );
        let url   = state.fortress_url.clone();
        let token = state.fortress_api_token.clone();
        tokio::spawn(async move {
            nuclear_eye::riviere::post_event("nuclear-eye", "camera", &content, &url, &token).await;
        });
    }

    info!(
        %triad, %action,
        safety = is_safety_critical,
        dominant,
        consul = consul_synthesis.is_some(),
        "decision computed"
    );

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
