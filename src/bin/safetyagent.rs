use anyhow::{Context, Result};
use axum::{extract::State, routing::{get, post}, Json, Router};
use nuclear_eye::{level_from_string, AlarmLevel, SecurityConfig, VisionEvent};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{error, info};

#[derive(Clone)]
struct AppState {
    cfg: Arc<SecurityConfig>,
    client: Client,
}

#[derive(Debug, Serialize, Deserialize)]
struct TelegramTest {
    text: Option<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    // S-7: fail-closed wrapper probe
    nuclear_eye::wrapper_guard::check_wrapper("safetyagent").await?;

    // ── Nuclear wrapper — resilience sidecar ────────────────────────────
    match nuclear_wrapper::wrap!(
        node_id      = "safetyagent",
        pg_url       = std::env::var("DATABASE_URL").unwrap_or_default(),
        signal_token = std::env::var("SIGNAL_TOKEN").unwrap_or_default()
    ) {
        Ok(nw) => {
            tracing::info!("nuclear-wrapper: armed (tamper, health, discovery)");
            std::mem::forget(nw);
        }
        Err(e) => tracing::info!("nuclear-wrapper: start failed ({e}) — running unguarded"),
    }

    let cfg = Arc::new(SecurityConfig::load()?);
    let client = Client::new();

    let app = Router::new()
        .route("/evaluate", post(evaluate))
        .route("/telegram/test", post(test_message))
        .route("/health", get(health))
        .with_state(AppState { cfg, client });

    let bind = app_state_config();
    let listener = tokio::net::TcpListener::bind(&bind).await?;
    info!("safetyagent listening on {bind}");
    axum::serve(listener, app).await?;
    Ok(())
}

fn app_state_config() -> String {
    std::env::var("HOUSE_SECURITY_CONFIG")
        .ok()
        .and_then(|_| SecurityConfig::load().ok())
        .map(|c| c.app.bind_safetyagent)
        .unwrap_or_else(|| {
            let host = std::env::var("BIND_HOST").unwrap_or_else(|_| "127.0.0.1".to_string());
            format!("{host}:8081")
        })
}

async fn health() -> &'static str {
    "ok"
}

async fn evaluate(
    State(state): State<AppState>,
    Json(event): Json<VisionEvent>,
) -> Json<serde_json::Value> {
    let response = state
        .client
        .post(format!("{}/ingest", state.cfg.app.alarm_grader_url.trim_end_matches('/')))
        .json(&event)
        .send()
        .await;

    let Ok(resp) = response else {
        error!("alarm grader unavailable");
        return Json(serde_json::json!({"accepted": false, "error": "alarm grader unavailable"}));
    };

    let body: serde_json::Value = match resp.json().await {
        Ok(v) => v,
        Err(err) => return Json(serde_json::json!({"accepted": false, "error": err.to_string()})),
    };

    let level = body
        .get("alarm")
        .and_then(|a| a.get("level"))
        .and_then(|v| v.as_str())
        .map(level_from_string)
        .unwrap_or(AlarmLevel::None);

    let min_level = level_from_string(&state.cfg.alarm.telegram_min_level);
    let sent = should_notify(&level, &min_level)
        && send_telegram_if_enabled(&state, &body.to_string()).await.unwrap_or(false);

    Json(serde_json::json!({"accepted": true, "upstream": body, "telegram_sent": sent}))
}

async fn test_message(
    State(state): State<AppState>,
    Json(input): Json<TelegramTest>,
) -> Json<serde_json::Value> {
    let text = input.text.unwrap_or_else(|| "house-security-ai test message".to_string());
    let sent = send_telegram_if_enabled(&state, &text).await.unwrap_or(false);
    Json(serde_json::json!({"sent": sent}))
}

fn should_notify(current: &AlarmLevel, min: &AlarmLevel) -> bool {
    let to_i = |level: &AlarmLevel| match level {
        AlarmLevel::None => 0,
        AlarmLevel::Low => 1,
        AlarmLevel::Medium => 2,
        AlarmLevel::High => 3,
    };
    to_i(current) >= to_i(min)
}

async fn send_telegram_if_enabled(state: &AppState, message: &str) -> Result<bool> {
    if !state.cfg.telegram.enabled {
        return Ok(false);
    }

    let bot_token = std::env::var(&state.cfg.telegram.bot_token_env)
        .with_context(|| format!("missing env {}", state.cfg.telegram.bot_token_env))?;
    let chat_id = std::env::var(&state.cfg.telegram.chat_id_env)
        .with_context(|| format!("missing env {}", state.cfg.telegram.chat_id_env))?;

    let endpoint = format!("https://api.telegram.org/bot{bot_token}/sendMessage");
    let payload = serde_json::json!({
        "chat_id": chat_id,
        "text": message,
        "disable_web_page_preview": true
    });

    let resp = state.client.post(endpoint).json(&payload).send().await?;
    info!("telegram status={}", resp.status());
    Ok(resp.status().is_success())
}
