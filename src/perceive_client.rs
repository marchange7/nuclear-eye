//! perceive_client — HTTP client for `perceive_service` `/v1/perceive` (port 8091).
//!
//! Closes os/05 §"Multimodal Perception Stack" — Track W. Until this module
//! lands, `vision_agent` and `iphone_sensor_agent` hardcoded
//! `face_negative/voice_agitated/gesture_threat = None`, so the
//! `compute_perceptual_risk` fusion in `alarm_grader_agent` was wired but
//! starved (see HANDOFF entry 2026-05-10 D-full).
//!
//! ## Contract
//!
//! - `PERCEIVE_URL` env (e.g. `http://127.0.0.1:8091`). When unset the client
//!   stays silent and `perceive_frame()` returns `Ok(None)` — callers must
//!   tolerate the bypass.
//! - Default timeout 250ms (configurable via `PERCEIVE_TIMEOUT_MS`).
//!   Tighter than vision_agent's 800ms VLM budget so a slow perceive_service
//!   never holds the camera→grader pipeline.
//! - `request_id`-style logging via `source_id` field (we pass `camera_id`).
//!
//! ## Fields produced
//!
//! `PerceiveOutput.face_negative / voice_agitated / gesture_threat` are the
//! 0..1 inputs that `alarm_grader_agent::compute_perceptual_risk` consumes.
//! The mapping mirrors the `compute_risk()` sub-formulas in
//! `perceive_service.py`:
//!
//!   - face_negative  = ((-valence + 1.0) / 2.0) * confidence
//!   - voice_agitated = sqrt(((arousal+1)/2) * ((-valence+1)/2)) * confidence
//!   - gesture_threat = intent_score(intent) * confidence
//!
//! `PerceiveOutput.components` carries the per-signal raw scores so the
//! canonical `SentinelleAlert.confidence.components` map can be populated
//! upstream without re-deriving from raw face/voice/gesture dicts.
//!
//! ## Boundary (os/69 / os/05)
//!
//! Sentinelle perceptual emotions are about INTRUDER THREAT, not affect.
//! `perceive_service` MUST never ingest Arianne affect payloads, and this
//! client MUST never forward `X-Arianne-*` headers. We send only
//! `X-Tenant-Id` (when present) and the raw frame/audio bytes.

use std::collections::BTreeMap;
use std::time::Duration;

use base64::Engine;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tracing::{debug, warn};

const DEFAULT_TIMEOUT_MS: u64 = 250;

/// Outcome of a single `/v1/perceive` call. All `Option<f32>` fields are
/// 0..1; `None` means "this modality wasn't returned" (perceive_service
/// emits `face: null` etc. when the input wasn't present or the model
/// declined to score).
#[derive(Debug, Clone, Default)]
pub struct PerceiveOutput {
    /// `compute_perceptual_risk` `face` input.
    pub face_negative: Option<f32>,
    /// `compute_perceptual_risk` `voice` input.
    pub voice_agitated: Option<f32>,
    /// `compute_perceptual_risk` `gesture` input.
    pub gesture_threat: Option<f32>,
    /// `risk.score` from the service when ≥2 modalities present.
    pub fused_risk: Option<f32>,
    /// True when the service classified the fused risk as alert-worthy
    /// (`risk.alert == true`). Pre-fusion; the grader still applies
    /// hysteresis + Penny.
    pub fused_alert: bool,
    /// Per-component contributions (already weighted by 0.4/0.3/0.3).
    /// Suitable for `SentinelleAlert.confidence.components`.
    pub components: BTreeMap<String, f32>,
    /// Free-text mood string from the service (e.g. "negative-low-arousal").
    pub mood_summary: Option<String>,
    /// Method tag for `SentinelleAlert.confidence.method` traceability.
    /// Format: `"perceive_service+<modalities>"` e.g.
    /// `"perceive_service+face+gesture"`.
    pub method: String,
}

impl PerceiveOutput {
    /// `true` when at least one modality scored. Useful for
    /// `degraded.flags = [PerceptionDegraded]` decisions: if false AND
    /// the service returned a 200, the frame had no person / no audio
    /// — that's expected, not degraded. If the service errored, the
    /// caller gets `Err(_)` and decides how to flag.
    pub fn any_modality(&self) -> bool {
        self.face_negative.is_some()
            || self.voice_agitated.is_some()
            || self.gesture_threat.is_some()
    }
}

#[derive(Debug, Serialize)]
struct PerceiveRequest<'a> {
    source_id: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    frame_b64: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    audio_b64: Option<String>,
    sample_rate: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    text_context: Option<&'a str>,
    /// P4-7 Scout passthrough. None when not driven by Scout.
    #[serde(skip_serializing_if = "Option::is_none")]
    gesture_pose: Option<&'a str>,
    gesture_pose_confidence: f32,
}

#[derive(Debug, Deserialize)]
struct PerceiveResponse {
    #[allow(dead_code)]
    timestamp: f64,
    #[allow(dead_code)]
    source_id: String,
    face: Option<serde_json::Value>,
    voice: Option<serde_json::Value>,
    gesture: Option<serde_json::Value>,
    risk: Option<RiskBlock>,
    mood_summary: Option<String>,
}

#[derive(Debug, Deserialize)]
struct RiskBlock {
    score: f32,
    alert: bool,
    #[serde(default)]
    components: BTreeMap<String, f32>,
}

/// Read `PERCEIVE_URL` from env. Returns `None` when unset/empty so callers
/// can short-circuit before allocating the JPEG base64.
pub fn perceive_url_from_env() -> Option<String> {
    std::env::var("PERCEIVE_URL")
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
}

fn timeout_from_env() -> Duration {
    let ms = std::env::var("PERCEIVE_TIMEOUT_MS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(DEFAULT_TIMEOUT_MS);
    Duration::from_millis(ms)
}

/// Call `/v1/perceive` with a JPEG frame and (optional) PCM audio.
///
/// Returns:
/// - `Ok(Some(out))` on a 2xx with parseable body.
/// - `Ok(None)` when `PERCEIVE_URL` is unset (bypass mode — caller keeps
///   `face_negative/voice_agitated/gesture_threat = None`).
/// - `Err(_)` on network / decode failure. Caller decides whether to flag
///   `degraded.flags = [PerceptionDegraded]`.
pub async fn perceive_frame(
    client: &Client,
    perceive_url: &str,
    source_id: &str,
    frame_jpeg: Option<&[u8]>,
    audio_pcm: Option<&[u8]>,
    sample_rate: u32,
    text_context: Option<&str>,
) -> anyhow::Result<PerceiveOutput> {
    let frame_b64 = frame_jpeg
        .map(|b| base64::engine::general_purpose::STANDARD.encode(b));
    let audio_b64 = audio_pcm
        .map(|b| base64::engine::general_purpose::STANDARD.encode(b));

    let body = PerceiveRequest {
        source_id,
        frame_b64,
        audio_b64,
        sample_rate,
        text_context,
        gesture_pose: None,
        gesture_pose_confidence: 0.5,
    };

    let url = format!("{}/v1/perceive", perceive_url.trim_end_matches('/'));

    let resp = client
        .post(&url)
        .json(&body)
        .timeout(timeout_from_env())
        .send()
        .await?;

    if !resp.status().is_success() {
        anyhow::bail!("perceive {} returned {}", url, resp.status());
    }

    let parsed: PerceiveResponse = resp.json().await?;
    Ok(map_response(parsed))
}

/// Convenience wrapper for the camera-only path: no audio, no text context.
/// `vision_agent` and `alarm_grader_agent` use this.
pub async fn perceive_frame_only(
    client: &Client,
    perceive_url: &str,
    source_id: &str,
    frame_jpeg: &[u8],
) -> anyhow::Result<PerceiveOutput> {
    perceive_frame(client, perceive_url, source_id, Some(frame_jpeg), None, 16000, None).await
}

fn map_response(r: PerceiveResponse) -> PerceiveOutput {
    // Mirror the sub-formulas in perceive_service.py::compute_risk(). We
    // re-derive the per-modality scores here so the grader sees the same
    // 0..1 values that the service used internally — without needing to
    // round-trip the raw face/voice/gesture dicts.
    let face_negative = r.face.as_ref().map(|f| {
        let valence = f.get("valence").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
        let conf = f.get("confidence").and_then(|v| v.as_f64()).unwrap_or(0.5) as f32;
        (((-valence + 1.0) / 2.0) * conf).clamp(0.0, 1.0)
    });

    let voice_agitated = r.voice.as_ref().map(|v| {
        let valence = v.get("valence").and_then(|x| x.as_f64()).unwrap_or(0.0) as f32;
        let arousal = v.get("arousal").and_then(|x| x.as_f64()).unwrap_or(0.0) as f32;
        let conf = v.get("confidence").and_then(|x| x.as_f64()).unwrap_or(0.5) as f32;
        let combined = ((arousal + 1.0) / 2.0) * ((-valence + 1.0) / 2.0);
        (combined.sqrt() * conf).clamp(0.0, 1.0)
    });

    let gesture_threat = r.gesture.as_ref().map(|g| {
        // Mirror perceive_service intent_scores. Keep in sync there.
        let intent = g.get("intent").and_then(|x| x.as_str()).unwrap_or("unknown");
        let conf = g.get("confidence").and_then(|x| x.as_f64()).unwrap_or(0.5) as f32;
        let intent_score: f32 = match intent {
            "attacking" => 1.0,
            "approaching" => 0.7,
            "fast_approach" => 0.85,
            "hands_raised" => 0.88,
            "loitering" => 0.5,
            "fleeing" => 0.4,
            "help_needed" => 0.3,
            "normal" => 0.0,
            _ => 0.2,
        };
        (intent_score * conf).clamp(0.0, 1.0)
    });

    let mut method_parts: Vec<&str> = Vec::with_capacity(3);
    if face_negative.is_some()  { method_parts.push("face"); }
    if voice_agitated.is_some() { method_parts.push("voice"); }
    if gesture_threat.is_some() { method_parts.push("gesture"); }
    let method = if method_parts.is_empty() {
        "perceive_service+empty".to_string()
    } else {
        format!("perceive_service+{}", method_parts.join("+"))
    };

    let (fused_risk, fused_alert, mut components) = match r.risk {
        Some(rb) => (Some(rb.score), rb.alert, rb.components),
        None => (None, false, BTreeMap::new()),
    };

    // Surface raw per-modality scores under canonical keys so they can map
    // 1:1 onto SentinelleAlert.confidence.components (face/voice/gesture).
    if let Some(v) = face_negative  { components.insert("face".to_string(),    v); }
    if let Some(v) = voice_agitated { components.insert("voice".to_string(),   v); }
    if let Some(v) = gesture_threat { components.insert("gesture".to_string(), v); }

    let out = PerceiveOutput {
        face_negative,
        voice_agitated,
        gesture_threat,
        fused_risk,
        fused_alert,
        components,
        mood_summary: r.mood_summary,
        method,
    };

    debug!(
        face = ?out.face_negative,
        voice = ?out.voice_agitated,
        gesture = ?out.gesture_threat,
        risk = ?out.fused_risk,
        alert = out.fused_alert,
        method = %out.method,
        "perceive_client: response mapped"
    );
    out
}

/// Best-effort wrapper: returns a `PerceiveOutput` (possibly empty) and
/// never errors. Logs a warning on failure. Use when you'd rather degrade
/// silently than break the camera pipeline. Returns `None` only when
/// `PERCEIVE_URL` is unset.
pub async fn perceive_frame_silent(
    client: &Client,
    source_id: &str,
    frame_jpeg: &[u8],
) -> Option<PerceiveOutput> {
    let url = perceive_url_from_env()?;
    match perceive_frame_only(client, &url, source_id, frame_jpeg).await {
        Ok(out) => Some(out),
        Err(e) => {
            warn!(error = %e, source_id, "perceive_client: call failed; degrading to None");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_response_face_only() {
        let r: PerceiveResponse = serde_json::from_value(serde_json::json!({
            "timestamp": 1.0,
            "source_id": "cam-1",
            "face": { "valence": -0.6, "confidence": 0.9 },
            "voice": null,
            "gesture": null,
            "risk": null,
            "mood_summary": null,
        })).unwrap();
        let out = map_response(r);
        assert!(out.face_negative.is_some());
        // (-(-0.6) + 1) / 2 * 0.9 = 1.6/2 * 0.9 = 0.72
        assert!((out.face_negative.unwrap() - 0.72).abs() < 0.01);
        assert!(out.voice_agitated.is_none());
        assert!(out.gesture_threat.is_none());
        assert_eq!(out.method, "perceive_service+face");
    }

    #[test]
    fn map_response_attacking_gesture() {
        let r: PerceiveResponse = serde_json::from_value(serde_json::json!({
            "timestamp": 1.0,
            "source_id": "cam-1",
            "face": null,
            "voice": null,
            "gesture": { "intent": "attacking", "confidence": 1.0 },
            "risk": null,
            "mood_summary": null,
        })).unwrap();
        let out = map_response(r);
        assert_eq!(out.gesture_threat, Some(1.0));
        assert_eq!(out.method, "perceive_service+gesture");
    }

    #[test]
    fn map_response_propagates_fused_risk() {
        let r: PerceiveResponse = serde_json::from_value(serde_json::json!({
            "timestamp": 1.0,
            "source_id": "cam-1",
            "face": { "valence": -0.8, "confidence": 0.9 },
            "voice": null,
            "gesture": { "intent": "approaching", "confidence": 0.8 },
            "risk": { "score": 0.74, "alert": true, "components": {"face_contribution": 0.36} },
            "mood_summary": "negative-high-arousal",
        })).unwrap();
        let out = map_response(r);
        assert_eq!(out.fused_risk, Some(0.74));
        assert!(out.fused_alert);
        assert_eq!(out.mood_summary.as_deref(), Some("negative-high-arousal"));
        assert!(out.components.contains_key("face"));
        assert!(out.components.contains_key("gesture"));
        assert!(out.components.contains_key("face_contribution"));
        assert!(out.any_modality());
    }

    #[test]
    fn perceive_url_from_env_empty_returns_none() {
        // Clear and assert
        // Note: not using #[serial] here, but env interactions in tests are
        // a known footgun. Subsequent tests don't depend on PERCEIVE_URL.
        std::env::remove_var("PERCEIVE_URL");
        assert!(perceive_url_from_env().is_none());
        std::env::set_var("PERCEIVE_URL", "");
        assert!(perceive_url_from_env().is_none());
        std::env::set_var("PERCEIVE_URL", "http://127.0.0.1:8091");
        assert_eq!(perceive_url_from_env().as_deref(), Some("http://127.0.0.1:8091"));
        std::env::remove_var("PERCEIVE_URL");
    }
}
