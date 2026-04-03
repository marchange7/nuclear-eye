// La Rivière bridge — fire-and-forget event append.
//
// All nuclear-eye agents call `post_event()` after each decision.
// SQLite in SecurityMemory is the durable fallback; this is additive.

use std::time::Duration;

/// Post a single event to La Rivière (POST /v1/stream/event on fortress).
///
/// Silently swallows errors — the caller must not await this; wrap in
/// `tokio::spawn` so it never blocks the hot path.
pub async fn post_event(
    agent_id: &str,
    surface: &str,
    content: &str,
    fortress_url: &str,
    api_token: &str,
) {
    if fortress_url.is_empty() {
        return;
    }
    let client = match reqwest::Client::builder()
        .timeout(Duration::from_secs(2))
        .build()
    {
        Ok(c) => c,
        Err(e) => {
            tracing::warn!(error = %e, "riviere: failed to build client");
            return;
        }
    };
    let payload = serde_json::json!({
        "agent_id":   agent_id,
        "surface":    surface,
        "event_type": "reflection",
        "content":    content,
    });
    let result = client
        .post(format!("{}/v1/stream/event", fortress_url.trim_end_matches('/')))
        .bearer_auth(api_token)
        .json(&payload)
        .send()
        .await;
    match result {
        Ok(resp) if resp.status().is_success() => {
            tracing::debug!(agent = agent_id, "La Rivière: appended");
        }
        Ok(resp) => {
            tracing::warn!(status = %resp.status(), "La Rivière: non-success (SQLite active)");
        }
        Err(e) => {
            tracing::warn!(error = %e, "La Rivière: unreachable (SQLite active)");
        }
    }
}
