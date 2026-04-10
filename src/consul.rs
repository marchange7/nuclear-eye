/// Async, non-blocking Consul client for alarm-grader decisions.
///
/// Local-only backend — POST to nuclear-consul via nuclear-sdk.
/// Timeout: 80 ms (inside the 100 ms alarm-grading SLA).
///
/// If nuclear-consul is unreachable the caller always gets `None` and the
/// local alarm grade stands unchanged. No cloud fallback — sovereignty first.
use nuclear_sdk::{NuclearClient, types::routing::ConsulQuery};
use std::time::Duration;
use tokio::task::JoinHandle;

// ── Public types ──────────────────────────────────────────────────────

/// A distilled Consul deliberation result.
#[derive(Debug, Clone)]
pub struct ConsulDecision {
    /// Consul's textual verdict ("approve", "deny", "escalate", "monitor", …).
    pub decision: String,
    /// Fraction of voices that returned `ok = true` (0.0 – 1.0).
    pub confidence: f64,
    /// How many voices participated.
    pub voices: usize,
}

// ── Client ────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct ConsulClient {
    pub timeout_ms: u64,
    nk: NuclearClient,
}

impl ConsulClient {
    /// Create a client targeting `consul_url` with the given timeout.
    ///
    /// Only the local nuclear-consul backend is supported. If nuclear-consul
    /// is unreachable, queries return `None` — no cloud fallback is attempted.
    pub fn new(consul_url: String, timeout_ms: u64) -> Self {
        tracing::info!("consul: local backend → {consul_url}");
        Self {
            timeout_ms,
            nk: build_consul_nk(&consul_url),
        }
    }

    /// Fire a Consul query in a background task and return its handle.
    ///
    /// Call with `tokio::time::timeout(…, handle).await` to merge the result
    /// if it arrives in time, or drop the handle for fire-and-forget.
    pub fn query_async(&self, question: &str) -> JoinHandle<Option<ConsulDecision>> {
        let timeout_ms = self.timeout_ms;
        let question = question.to_string();
        let nk = self.nk.clone();

        tokio::spawn(async move {
            query_local(nk, question, timeout_ms).await
        })
    }
}

impl Default for ConsulClient {
    fn default() -> Self {
        Self::new("http://127.0.0.1:7710".to_string(), 80)
    }
}

/// Build a NuclearClient for the consul endpoint.
fn build_consul_nk(consul_url: &str) -> NuclearClient {
    let mut config = nuclear_sdk::NuclearConfig::from_system()
        .expect("NuclearConfig for consul backend");
    if !consul_url.is_empty() {
        if let Ok(url) = consul_url.parse() {
            config.consul_url = url;
        }
    }
    NuclearClient::from_config(config)
        .expect("NuclearClient for consul backend")
}

// ── Local backend (nuclear-consul via SDK) ────────────────────────────

async fn query_local(
    nk: NuclearClient,
    question: String,
    timeout_ms: u64,
) -> Option<ConsulDecision> {
    let q = ConsulQuery {
        query: question,
        caller: Some("nuclear-eye".to_string()),
        context: Some("alarm grader high-severity security assessment".to_string()),
        require_security: true,
        require_ethics: false,
        max_output_tokens: Some(150),
    };

    let result = tokio::time::timeout(
        Duration::from_millis(timeout_ms),
        nk.consul().query(&q),
    )
    .await;

    match result {
        Ok(Ok(cd)) => {
            let total = cd.consulted_voices.len();
            let ok_votes = cd.consulted_voices.iter().filter(|v| v.ok).count();
            let confidence = if total > 0 { ok_votes as f64 / total as f64 } else { 0.5 };
            Some(ConsulDecision {
                decision: cd.decision,
                confidence,
                voices: total,
            })
        }
        Ok(Err(e)) => {
            tracing::error!("nuclear-consul unreachable, failing closed — no cloud fallback: {e}");
            None
        }
        Err(_) => {
            tracing::error!(
                "nuclear-consul unreachable, failing closed — no cloud fallback: timed out (>{timeout_ms} ms)"
            );
            None
        }
    }
}
