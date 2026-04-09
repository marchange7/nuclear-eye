use anyhow::{Context, Result};
use axum::{
    extract::State,
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use nuclear_eye::{
    decide, AffectTriad, AlarmEvent, DecisionAction, SecurityConfig, TelegramNotifier,
};
use nuclear_sdk::NuclearClient;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tokio::{signal, time::Duration};
use tracing::{error, info, instrument, warn};

// ── State ──────────────────────────────────────────────────────────────

#[derive(Clone)]
struct AppState {
    client: Client,
    /// Aurelia POST /api/safety endpoint URL.
    aurelia_safety_url: String,
    telegram: Option<TelegramNotifier>,
    telegram_on_alarm: bool,
    nk: NuclearClient,
    fortress_enabled: bool,
}

// ── Aurelia /api/safety protocol ──────────────────────────────────────

/// Structured alarm payload for Aurelia's `/api/safety` endpoint.
/// Aurelia builds the empathetic prompt internally — no freeform text encoding.
#[derive(Debug, Serialize)]
struct SafetyAlarmRequest {
    /// "low" | "medium" | "high"
    severity: String,
    zone: String,
    danger_score: f64,
    message: String,
    person_name: Option<String>,
    stress_level: Option<f64>,
}

impl SafetyAlarmRequest {
    fn from_alarm(alarm: &AlarmEvent) -> Self {
        Self {
            severity: alarm.level.to_string().to_lowercase(),
            // AlarmEvent has no zone field yet — derive from VLM caption or default
            zone: alarm
                .vlm_caption
                .as_deref()
                .and_then(|c| {
                    // "zone:front-door ..." style hint in caption
                    c.split_whitespace()
                        .find(|w| w.starts_with("zone:"))
                        .map(|w| w.trim_start_matches("zone:").to_string())
                })
                .unwrap_or_else(|| "perimeter".into()),
            danger_score: alarm.danger_score,
            message: alarm.note.clone(),
            person_name: alarm.person_name.clone(),
            stress_level: Some(alarm.stress_level),
        }
    }
}

#[derive(Debug, Deserialize)]
struct SafetyAlarmResponse {
    reply: String,
    action: String,
    #[allow(dead_code)]
    level: String,
    #[allow(dead_code)]
    conversation_id: uuid::Uuid,
}

// ── Response types ─────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
struct AlertResponse {
    aurelia_reply: String,
    triad: AffectTriad,
    action: String,
    alarm_level: String,
    telegram_sent: bool,
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
        node_id      = "safety-aurelie-agent",
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
    let timeout = Duration::from_secs(cfg.aurelie_bridge.request_timeout_secs);

    let client = Client::builder()
        .timeout(timeout)
        .build()
        .context("failed to build HTTP client")?;

    let telegram = TelegramNotifier::from_config(&cfg.telegram, &client)
        .context("failed to initialise Telegram notifier")
        .unwrap_or_else(|e| {
            warn!("Telegram disabled: {e:#}");
            None
        });

    let bind = std::env::var("SAFETY_AURELIE_BIND")
        .unwrap_or_else(|_| cfg.aurelie_bridge.bind.clone());

    // AURELIE_CHAT_URL kept for backward-compat; new canonical name is AURELIA_SAFETY_URL
    let aurelia_safety_url = std::env::var("AURELIA_SAFETY_URL")
        .or_else(|_| std::env::var("AURELIE_CHAT_URL"))
        .unwrap_or_else(|_| cfg.aurelie_bridge.aurelie_chat_url.clone());

    let fortress_enabled = cfg.fortress.mesh_enabled;
    let nk = NuclearClient::from_system()
        .expect("NuclearClient: check FORTRESS_URL env var");

    let state = AppState {
        client,
        aurelia_safety_url,
        telegram,
        telegram_on_alarm: cfg.aurelie_bridge.telegram_on_alarm,
        nk,
        fortress_enabled,
    };

    let app = Router::new()
        .route("/alert", post(handle_alert))
        .route("/health", get(health))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(&bind).await?;
    info!(bind = %bind, "safety_aurelie_agent started (→ Aurelia /api/safety)");

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    info!("safety_aurelie_agent shut down cleanly");
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

async fn health() -> &'static str {
    "ok"
}

#[instrument(skip_all, fields(alarm_id))]
async fn handle_alert(
    State(state): State<AppState>,
    payload: Result<Json<AlarmEvent>, axum::extract::rejection::JsonRejection>,
) -> Result<Json<AlertResponse>, (StatusCode, Json<ErrorBody>)> {
    let Json(alarm) = payload.map_err(|err| {
        warn!(%err, "bad request");
        (
            StatusCode::BAD_REQUEST,
            Json(ErrorBody {
                error: err.to_string(),
            }),
        )
    })?;

    tracing::Span::current().record("alarm_id", &alarm.alarm_id.as_str());

    let triad = AffectTriad::from_alarm_event(&alarm);
    let action = decide(&triad, true);

    info!(
        level = %alarm.level,
        %triad,
        %action,
        "processing alarm → Aurelia /api/safety"
    );

    // Call Aurelia's /api/safety with one retry on network failure
    let (aurelia_reply, aurelia_action) = call_aurelia_with_retry(&state, &alarm).await;

    // Optionally soften Telegram tone based on user's relational mood
    let user_mood = if state.fortress_enabled {
        get_user_mood(&state.nk).await
    } else {
        None
    };

    // Telegram notification (fire-and-forget, errors logged)
    let telegram_sent = maybe_send_telegram(
        &state,
        &alarm,
        &triad,
        &action,
        &aurelia_reply,
        user_mood.as_deref(),
    )
    .await;

    info!(
        reply_len = aurelia_reply.len(),
        telegram_sent,
        "alert processed"
    );

    Ok(Json(AlertResponse {
        aurelia_reply,
        triad,
        // Prefer Aurelia's action decision; fall back to local decide()
        action: aurelia_action.unwrap_or_else(|| action.to_string()),
        alarm_level: alarm.level.to_string(),
        telegram_sent,
    }))
}

// ── Aurelia HTTP call with retry ───────────────────────────────────────

async fn call_aurelia_with_retry(state: &AppState, alarm: &AlarmEvent) -> (String, Option<String>) {
    match call_aurelia(state, alarm).await {
        Ok(resp) => (resp.reply, Some(resp.action)),
        Err(first_err) => {
            warn!(%first_err, "first Aurelia /api/safety call failed, retrying");
            tokio::time::sleep(Duration::from_millis(500)).await;
            match call_aurelia(state, alarm).await {
                Ok(resp) => (resp.reply, Some(resp.action)),
                Err(retry_err) => {
                    error!(%retry_err, "Aurelia unreachable after retry");
                    (fallback_response(alarm, &decide(&AffectTriad::from_alarm_event(alarm), true)), None)
                }
            }
        }
    }
}

async fn call_aurelia(state: &AppState, alarm: &AlarmEvent) -> Result<SafetyAlarmResponse> {
    let payload = SafetyAlarmRequest::from_alarm(alarm);

    let resp = state
        .client
        .post(&state.aurelia_safety_url)
        .json(&payload)
        .send()
        .await
        .context("HTTP request to Aurelia /api/safety failed")?;

    let status = resp.status();
    if !status.is_success() {
        anyhow::bail!("Aurelia /api/safety returned HTTP {status}");
    }

    resp.json::<SafetyAlarmResponse>()
        .await
        .context("failed to parse Aurelia SafetyAlarmResponse")
}

fn fallback_response(alarm: &AlarmEvent, action: &DecisionAction) -> String {
    match action {
        DecisionAction::Alarm => format!(
            "⚠️ Alerte niveau {}. Situation détectée : {}. Restez vigilant.",
            alarm.level, alarm.note,
        ),
        DecisionAction::Pause => {
            "La situation est incertaine. Prenez un moment pour observer.".into()
        }
        _ => format!(
            "Alerte {} enregistrée. Aucune action immédiate requise.",
            alarm.level,
        ),
    }
}

// ── Telegram ───────────────────────────────────────────────────────────

async fn maybe_send_telegram(
    state: &AppState,
    alarm: &AlarmEvent,
    triad: &AffectTriad,
    action: &DecisionAction,
    aurelia_reply: &str,
    user_mood: Option<&str>,
) -> bool {
    if !state.telegram_on_alarm {
        return false;
    }

    let Some(ref tg) = state.telegram else {
        return false;
    };

    let stressed = user_mood
        .map(|m| m.contains("stress") || m.contains("frustrated"))
        .unwrap_or(false);

    let message = if stressed {
        format!(
            "🏠 Mise à jour sécurité\n\
             Niveau: {} (risque {:.2})\n\
             Situation: {}\n\
             Aurelia: {aurelia_reply}",
            alarm.level, alarm.danger_score, alarm.note,
        )
    } else {
        format!(
            "🏠 Safety→Aurelia\n\
             Alarm: {} (danger {:.2})\n\
             Triad: {triad} → {action}\n\
             Aurelia: {aurelia_reply}",
            alarm.level, alarm.danger_score,
        )
    };

    match tg.send(&message).await {
        Ok(sent) => sent,
        Err(err) => {
            warn!(%err, "telegram notification failed");
            false
        }
    }
}

async fn get_user_mood(nk: &NuclearClient) -> Option<String> {
    let json = nk.fortress().get_relational("andrzej").await.ok()?;
    json["mood"].as_str().map(|s| s.to_string())
}
