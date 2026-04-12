// La Rivière bridge — fire-and-forget event append via nuclear-sdk.
//
// Two emission paths:
//
// 1. LEGACY: `post_event()` — sends a `StreamEvent` (reflection surface) via the
//    nuclear-sdk. Used by alarm_grader_agent for the existing behavioural stream.
//
// 2. DOMAIN EVENTS (O7 / Q5): `post_domain_event()` — POST to Fortress
//    /v1/events with the canonical `riviere.domain_events` schema:
//
//      { event_type, source_domain, target_domain, session_id,
//        payload, priority, status, site_id }
//
//    Supported types (matching riviere.domain_event_types):
//      vision.person_detected   source=vision, target=null (broadcast)
//      vision.behavior_alert    source=vision, target=null (broadcast)
//      vision.scene_captured    source=vision, target=null (broadcast)
//      vision.face_identified   source=vision, target=text
//
// All functions are fire-and-forget (called inside tokio::spawn).
// Errors are logged as WARN but never propagate — La Rivière is additive,
// not a hard dependency for alarm delivery.

use nuclear_sdk::{NuclearClient, types::stream::StreamEvent};
use serde::Serialize;
use std::time::Duration;

// ── Fortress endpoint (O7) ────────────────────────────────────────────────────

const DEFAULT_FORTRESS_URL: &str = "http://localhost:7700";
const DOMAIN_EVENT_TIMEOUT_MS: u64 = 500;

fn fortress_url() -> String {
    std::env::var("FORTRESS_URL").unwrap_or_else(|_| DEFAULT_FORTRESS_URL.to_string())
}

// ── Domain event payload types (Q5) ──────────────────────────────────────────

/// vision.person_detected payload.
#[derive(Debug, Serialize)]
pub struct PersonDetectedPayload {
    pub camera_id: String,
    pub count: u32,
    /// Unix timestamp (ms).
    pub ts: u64,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub positions: Vec<serde_json::Value>,
}

/// vision.behavior_alert payload.
#[derive(Debug, Serialize)]
pub struct BehaviorAlertPayload {
    pub camera_id: String,
    pub behavior: String,
    pub severity: String,
    pub danger_score: f64,
    pub ts: u64,
}

/// vision.scene_captured payload.
#[derive(Debug, Serialize)]
pub struct SceneCapturedPayload {
    pub camera_id: String,
    pub scene: String,
    pub ts: u64,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub objects: Vec<String>,
}

/// vision.face_identified payload.
#[derive(Debug, Serialize)]
pub struct FaceIdentifiedPayload {
    pub camera_id: String,
    pub name: String,
    pub authorized: bool,
    pub similarity: f32,
    pub ts: u64,
}

// ── Wire type for POST /v1/events ─────────────────────────────────────────────

#[derive(Debug, Serialize)]
struct DomainEventRequest {
    event_type: String,
    source_domain: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    target_domain: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    session_id: Option<String>,
    payload: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    priority: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    site_id: Option<String>,
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Legacy: Post a single reflection event to La Rivière via the nuclear-sdk stream.
///
/// Silently swallows errors — wrap in `tokio::spawn` so it never blocks
/// the hot path.  The NuclearClient is cloned cheaply (arc-backed pool).
pub async fn post_event(
    agent_id: &str,
    surface: &str,
    content: &str,
    nk: &NuclearClient,
) {
    let mut event = StreamEvent::new(agent_id, content);
    event.surface = surface.to_string();
    event.event_type = "reflection".into();
    match nk.fortress().append_stream_event(&event).await {
        Ok(()) => tracing::debug!(agent = agent_id, "La Rivière: appended"),
        Err(e) => tracing::warn!(error = %e, "La Rivière: unreachable (SQLite active)"),
    }
}

/// O7 / Q5: Post a domain event to Fortress /v1/events (La Rivière canonical schema).
///
/// `event_type`    — e.g. "vision.person_detected"
/// `source_domain` — always "vision" for nuclear-eye
/// `target_domain` — None for broadcasts, Some("text") for directed messages
/// `payload`       — serialized event payload
///
/// Errors are logged at WARN; never panics.
pub async fn post_domain_event(
    client: &reqwest::Client,
    event_type: &str,
    source_domain: &str,
    target_domain: Option<&str>,
    payload: serde_json::Value,
) {
    let url = format!("{}/v1/events", fortress_url());
    let body = DomainEventRequest {
        event_type: event_type.to_string(),
        source_domain: source_domain.to_string(),
        target_domain: target_domain.map(str::to_string),
        session_id: None,
        payload,
        priority: None,
        site_id: std::env::var("SITE_ID").ok(),
    };

    let result = client
        .post(&url)
        .json(&body)
        .timeout(Duration::from_millis(DOMAIN_EVENT_TIMEOUT_MS))
        .send()
        .await;

    match result {
        Ok(resp) if resp.status().is_success() => {
            tracing::debug!(event_type, "La Rivière domain event accepted");
        }
        Ok(resp) => {
            // 404 / 501 expected while /v1/events is still being built in fortress
            tracing::debug!(
                status = %resp.status(),
                event_type,
                "La Rivière /v1/events not yet available (non-blocking)"
            );
        }
        Err(e) => {
            tracing::warn!(error = %e, event_type, "La Rivière domain event POST failed (non-blocking)");
        }
    }
}

// ── Typed helpers — one per event type (Q5) ───────────────────────────────────

/// Emit `vision.person_detected` to La Rivière.
pub async fn emit_person_detected(
    client: &reqwest::Client,
    payload: PersonDetectedPayload,
) {
    let value = match serde_json::to_value(&payload) {
        Ok(v) => v,
        Err(e) => { tracing::warn!(error = %e, "emit_person_detected: serialize failed"); return; }
    };
    post_domain_event(client, "vision.person_detected", "vision", None, value).await;
}

/// Emit `vision.behavior_alert` to La Rivière.
pub async fn emit_behavior_alert(
    client: &reqwest::Client,
    payload: BehaviorAlertPayload,
) {
    let value = match serde_json::to_value(&payload) {
        Ok(v) => v,
        Err(e) => { tracing::warn!(error = %e, "emit_behavior_alert: serialize failed"); return; }
    };
    post_domain_event(client, "vision.behavior_alert", "vision", None, value).await;
}

/// Emit `vision.scene_captured` to La Rivière.
pub async fn emit_scene_captured(
    client: &reqwest::Client,
    payload: SceneCapturedPayload,
) {
    let value = match serde_json::to_value(&payload) {
        Ok(v) => v,
        Err(e) => { tracing::warn!(error = %e, "emit_scene_captured: serialize failed"); return; }
    };
    post_domain_event(client, "vision.scene_captured", "vision", None, value).await;
}

/// Emit `vision.face_identified` to La Rivière (target=text, directed event).
pub async fn emit_face_identified(
    client: &reqwest::Client,
    payload: FaceIdentifiedPayload,
) {
    let value = match serde_json::to_value(&payload) {
        Ok(v) => v,
        Err(e) => { tracing::warn!(error = %e, "emit_face_identified: serialize failed"); return; }
    };
    post_domain_event(client, "vision.face_identified", "vision", Some("text"), value).await;
}

// ── JJ1: Sentinelle domain events (continuous learning pipeline) ─────────────

/// sentinelle.alarm — alarm decision for continuous learning.
#[derive(Debug, Serialize)]
pub struct SentinelleAlarmPayload {
    pub alarm_id: String,
    pub camera_id: String,
    pub level: String,
    pub danger_score: f64,
    pub risk_score: f64,
    pub stress_level: f64,
    pub confidence: f64,
    pub behavior: String,
    pub person_detected: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub person_name: Option<String>,
    pub ts: u64,
    /// JJ6: Raw depth context forwarded from nuclear-scout (preserved verbatim
    /// so the learning pipeline can correlate depth features with alarm outcomes).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub depth_context: Option<serde_json::Value>,
}

/// sentinelle.feedback — operator annotation on alarm decision.
#[derive(Debug, Serialize)]
pub struct SentinelleFeedbackPayload {
    pub alarm_id: String,
    pub camera_id: String,
    pub feedback: String,  // "false_alarm", "confirmed", "escalate"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operator: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub notes: Option<String>,
    pub ts: u64,
}

/// sentinelle.face — face identity match for learning pipeline.
#[derive(Debug, Serialize)]
pub struct SentinelleFacePayload {
    pub camera_id: String,
    pub name: String,
    pub authorized: bool,
    pub similarity: f32,
    pub ts: u64,
}

/// JJ1: Emit `sentinelle.alarm` to La Rivière.
pub async fn emit_sentinelle_alarm(
    client: &reqwest::Client,
    payload: SentinelleAlarmPayload,
) {
    let value = match serde_json::to_value(&payload) {
        Ok(v) => v,
        Err(e) => { tracing::warn!(error = %e, "emit_sentinelle_alarm: serialize failed"); return; }
    };
    post_domain_event(client, "sentinelle.alarm", "sentinelle", None, value).await;
}

/// JJ1: Emit `sentinelle.feedback` to La Rivière.
pub async fn emit_sentinelle_feedback(
    client: &reqwest::Client,
    payload: SentinelleFeedbackPayload,
) {
    let value = match serde_json::to_value(&payload) {
        Ok(v) => v,
        Err(e) => { tracing::warn!(error = %e, "emit_sentinelle_feedback: serialize failed"); return; }
    };
    post_domain_event(client, "sentinelle.feedback", "sentinelle", None, value).await;
}

/// JJ1: Emit `sentinelle.face` to La Rivière.
pub async fn emit_sentinelle_face(
    client: &reqwest::Client,
    payload: SentinelleFacePayload,
) {
    let value = match serde_json::to_value(&payload) {
        Ok(v) => v,
        Err(e) => { tracing::warn!(error = %e, "emit_sentinelle_face: serialize failed"); return; }
    };
    post_domain_event(client, "sentinelle.face", "sentinelle", None, value).await;
}
