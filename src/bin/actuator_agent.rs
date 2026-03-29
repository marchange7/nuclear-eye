/// actuator_agent — translates DecisionAction / AlarmLevel into MQTT commands
/// for physical outputs (lights, buzzers, servo arms).
///
/// Env vars:
///   ACTUATOR_BIND         bind address          (default: 0.0.0.0:8086)
///   ACTUATOR_MQTT_HOST    MQTT broker host       (default: 127.0.0.1)
///   ACTUATOR_MQTT_PORT    MQTT broker port       (default: 1883)
///   ACTUATOR_MQTT_PREFIX  topic prefix           (default: nuclear/actuator)
///
/// MQTT topics published:
///   {prefix}/light        — "off" | "green" | "amber" | "blue" | "red"
///   {prefix}/buzzer       — "off" | "on"
///   {prefix}/arm          — JSON {"action":"...", "level":"...", "camera_id":"..."}
///
/// REST endpoint:
///   POST /actuate         — body: ActuateRequest
///   GET  /health

use anyhow::Result;
use axum::{
    extract::State,
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use rumqttc::{AsyncClient, MqttOptions, QoS};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;
use tokio::{signal, time::timeout};
use tracing::{info, warn};

// ── MQTT payload helpers ────────────────────────────────────────────────

/// Map DecisionAction string → light colour + buzzer state.
fn action_to_outputs(action: &str, level: &str) -> (&'static str, &'static str) {
    match action {
        "Alarm" => ("red", "on"),
        "Challenge" => ("amber", "off"),
        "Support" => ("blue", "off"),
        "Reassure" => ("green", "off"),
        "Pause" => ("amber", "off"),
        _ => match level {
            "high" => ("red", "on"),
            "medium" => ("amber", "off"),
            "low" => ("green", "off"),
            _ => ("off", "off"),
        },
    }
}

// ── State ───────────────────────────────────────────────────────────────

#[derive(Clone)]
struct AppState {
    mqtt: Arc<AsyncClient>,
    prefix: String,
}

// ── Request / Response ──────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct ActuateRequest {
    /// DecisionAction string: "None" | "Reassure" | "Challenge" | "Support" | "Alarm" | "Pause"
    action: String,
    /// AlarmLevel string: "none" | "low" | "medium" | "high"
    #[serde(default)]
    level: String,
    /// Source camera (informational, forwarded to arm topic)
    #[serde(default)]
    camera_id: String,
}

#[derive(Debug, Serialize)]
struct ActuateResponse {
    light: &'static str,
    buzzer: &'static str,
    published: bool,
}

#[derive(Debug, Serialize)]
struct HealthResponse {
    status: &'static str,
    mqtt_prefix: String,
}

// ── Handlers ────────────────────────────────────────────────────────────

async fn actuate(
    State(state): State<AppState>,
    Json(req): Json<ActuateRequest>,
) -> Result<Json<ActuateResponse>, (StatusCode, String)> {
    let (light, buzzer) = action_to_outputs(&req.action, &req.level);

    let arm_payload = serde_json::json!({
        "action": req.action,
        "level": req.level,
        "camera_id": req.camera_id,
    })
    .to_string();

    let light_topic = format!("{}/light", state.prefix);
    let buzzer_topic = format!("{}/buzzer", state.prefix);
    let arm_topic = format!("{}/arm", state.prefix);

    let mqtt_timeout = Duration::from_millis(500);
    let r1 = timeout(mqtt_timeout, state.mqtt.publish(&light_topic,  QoS::AtLeastOnce, false, light)).await;
    let r2 = timeout(mqtt_timeout, state.mqtt.publish(&buzzer_topic, QoS::AtLeastOnce, false, buzzer)).await;
    let r3 = timeout(mqtt_timeout, state.mqtt.publish(&arm_topic,    QoS::AtLeastOnce, false, arm_payload)).await;

    let published = r1.map(|r| r.is_ok()).unwrap_or(false)
        && r2.map(|r| r.is_ok()).unwrap_or(false)
        && r3.map(|r| r.is_ok()).unwrap_or(false);
    if !published {
        warn!(action = %req.action, level = %req.level, "one or more MQTT publishes failed");
    }

    info!(action = %req.action, level = %req.level, light, buzzer, published, "actuate");

    Ok(Json(ActuateResponse { light, buzzer, published }))
}

async fn health(State(state): State<AppState>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok",
        mqtt_prefix: state.prefix.clone(),
    })
}

// ── Graceful shutdown ───────────────────────────────────────────────────

async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c().await.expect("failed to install Ctrl+C handler");
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

// ── Main ────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let bind = std::env::var("ACTUATOR_BIND")
        .unwrap_or_else(|_| "0.0.0.0:8086".to_string());
    let mqtt_host = std::env::var("ACTUATOR_MQTT_HOST")
        .unwrap_or_else(|_| "127.0.0.1".to_string());
    let mqtt_port: u16 = std::env::var("ACTUATOR_MQTT_PORT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1883);
    let prefix = std::env::var("ACTUATOR_MQTT_PREFIX")
        .unwrap_or_else(|_| "nuclear/actuator".to_string());

    // Build MQTT client — keep-alive 30s, reconnect handled by rumqttc event loop
    let mut mqttopts = MqttOptions::new("nuclear-actuator-agent", &mqtt_host, mqtt_port);
    mqttopts.set_keep_alive(Duration::from_secs(30));

    let (mqtt_client, mut event_loop) = AsyncClient::new(mqttopts, 64);

    // Spawn MQTT event loop — must be running for publishes to flush
    tokio::spawn(async move {
        loop {
            match event_loop.poll().await {
                Ok(_) => {}
                Err(e) => {
                    warn!("MQTT event loop error: {e} — reconnecting in 5s");
                    tokio::time::sleep(Duration::from_secs(5)).await;
                }
            }
        }
    });

    info!(mqtt_host = %mqtt_host, mqtt_port, prefix = %prefix, "MQTT ready");

    let state = AppState {
        mqtt: Arc::new(mqtt_client),
        prefix,
    };

    let app = Router::new()
        .route("/actuate", post(actuate))
        .route("/health", get(health))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(&bind).await?;
    info!(bind = %bind, "actuator_agent started");

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    info!("actuator_agent shut down cleanly");
    Ok(())
}
