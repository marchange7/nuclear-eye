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
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tokio::{signal, time::Duration};
use tracing::{error, info, instrument, warn};

// ── State ──────────────────────────────────────────────────────────────

#[derive(Clone)]
struct AppState {
    client: Client,
    aurelie_chat_url: String,
    telegram: Option<TelegramNotifier>,
    telegram_on_alarm: bool,
    fortress_url: String,
    fortress_enabled: bool,
}

// ── Aurélie chat protocol ──────────────────────────────────────────────

#[derive(Debug, Serialize)]
struct AurelieChatPayload {
    user_id: String,
    text: String,
    context: AurelieMultiModalContext,
}

/// Minimal subset of Aurélie's MultiModalContext — enough to let her
/// generate an empathetic response from an alarm event.
#[derive(Debug, Serialize)]
struct AurelieMultiModalContext {
    user_id: String,
    timestamp_ms: u64,
    behavior: String,
    stress_level: f32,
    voice_emotion: String,
    voice_energy: f32,
    intimacy_level: f32,
    trust_score: f32,
    hand_gesture: Option<String>,
    object_held: Option<String>,
    recent_moods: Vec<String>,
}

impl AurelieMultiModalContext {
    fn from_alarm(alarm: &AlarmEvent) -> Self {
        let triad = AffectTriad::from_alarm_event(alarm);
        Self {
            user_id: alarm
                .person_name
                .clone()
                .unwrap_or_else(|| "unknown".into()),
            timestamp_ms: alarm.timestamp_ms,
            behavior: alarm.note.clone(),
            stress_level: alarm.stress_level as f32,
            voice_emotion: triad.dominant().into(),
            voice_energy: 0.7,
            intimacy_level: 0.3,
            trust_score: 0.5,
            hand_gesture: None,
            object_held: None,
            recent_moods: vec![format!("alarm-{}", alarm.level)],
        }
    }
}

#[derive(Debug, Deserialize)]
struct AurelieChatResponse {
    reply: String,
}

// ── Response types ─────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
struct AlertResponse {
    aurelie_reply: String,
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

    let aurelie_chat_url = std::env::var("AURELIE_CHAT_URL")
        .unwrap_or_else(|_| cfg.aurelie_bridge.aurelie_chat_url.clone());

    let fortress_url = cfg.fortress_url();
    let fortress_enabled = cfg.fortress.mesh_enabled;

    let state = AppState {
        client,
        aurelie_chat_url,
        telegram,
        telegram_on_alarm: cfg.aurelie_bridge.telegram_on_alarm,
        fortress_url,
        fortress_enabled,
    };

    let app = Router::new()
        .route("/alert", post(handle_alert))
        .route("/health", get(health))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(&bind).await?;
    info!(bind = %bind, "safety_aurelie_agent started");

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
        "processing alarm"
    );

    // Call Aurélie with one retry on network failure
    let aurelie_reply = call_aurelie_with_retry(&state, &alarm).await;

    // Optionally soften Telegram tone based on user's relational mood
    let user_mood = if state.fortress_enabled {
        get_user_mood(&state.fortress_url).await
    } else {
        None
    };

    // Telegram notification (fire-and-forget, errors logged)
    let telegram_sent = maybe_send_telegram(&state, &alarm, &triad, &action, &aurelie_reply, user_mood.as_deref()).await;

    info!(
        reply_len = aurelie_reply.len(),
        telegram_sent,
        "alert processed"
    );

    Ok(Json(AlertResponse {
        aurelie_reply,
        triad,
        action: action.to_string(),
        alarm_level: alarm.level.to_string(),
        telegram_sent,
    }))
}

// ── Aurélie HTTP call with retry ───────────────────────────────────────

async fn call_aurelie_with_retry(state: &AppState, alarm: &AlarmEvent) -> String {
    match call_aurelie(state, alarm).await {
        Ok(reply) => reply,
        Err(first_err) => {
            warn!(%first_err, "first Aurélie call failed, retrying");
            tokio::time::sleep(Duration::from_millis(500)).await;
            match call_aurelie(state, alarm).await {
                Ok(reply) => reply,
                Err(retry_err) => {
                    error!(%retry_err, "Aurélie unreachable after retry");
                    fallback_response(alarm, &decide(&AffectTriad::from_alarm_event(alarm), true))
                }
            }
        }
    }
}

async fn call_aurelie(state: &AppState, alarm: &AlarmEvent) -> Result<String> {
    let ctx = AurelieMultiModalContext::from_alarm(alarm);
    let alarm_text = format!(
        "[ALERTE SÉCURITÉ – niveau {}] Danger {:.2}, stress {:.2}. {}",
        alarm.level, alarm.danger_score, alarm.stress_level, alarm.note,
    );

    let payload = AurelieChatPayload {
        user_id: ctx.user_id.clone(),
        text: alarm_text,
        context: ctx,
    };

    let resp = state
        .client
        .post(&state.aurelie_chat_url)
        .json(&payload)
        .send()
        .await
        .context("HTTP request to Aurélie failed")?;

    let status = resp.status();
    if !status.is_success() {
        anyhow::bail!("Aurélie returned HTTP {status}");
    }

    let body: AurelieChatResponse = resp
        .json()
        .await
        .context("failed to parse Aurélie response")?;

    Ok(body.reply)
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
    aurelie_reply: &str,
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
             Aurélie: {aurelie_reply}",
            alarm.level, alarm.danger_score, alarm.note,
        )
    } else {
        format!(
            "🏠 Safety→Aurélie\n\
             Alarm: {} (danger {:.2})\n\
             Triad: {triad} → {action}\n\
             Aurélie: {aurelie_reply}",
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

async fn get_user_mood(fortress_url: &str) -> Option<String> {
    let client = reqwest::Client::new();
    let resp = client
        .get(format!("{}/v1/mesh/relational/andrzej", fortress_url))
        .timeout(std::time::Duration::from_millis(200))
        .send()
        .await
        .ok()?;
    let json: serde_json::Value = resp.json().await.ok()?;
    json["mood"].as_str().map(|s| s.to_string())
}
