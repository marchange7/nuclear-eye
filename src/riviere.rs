// La Rivière bridge — fire-and-forget event append via nuclear-sdk.
//
// All nuclear-eye agents call `post_event()` after each decision.
// SQLite in SecurityMemory is the durable fallback; this is additive.

use nuclear_sdk::{NuclearClient, types::stream::StreamEvent};

/// Post a single event to La Rivière (POST /v1/stream/event on fortress).
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
