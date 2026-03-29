/// Async, non-blocking Consul client for alarm-grader decisions.
///
/// Supports two backends, selected at runtime via `CONSUL_BACKEND`:
///
/// - `local`  (default) — POST to nuclear-platform fortress `/consul/query`
///            Timeout: 80 ms (inside the 100 ms alarm-grading SLA).
///
/// - `cloud`  — POST to Anthropic Claude API (claude-haiku-4-5-20251001)
///            Requires `ANTHROPIC_API_KEY` env var.
///            ~$0.002 / deliberation. Used for Tier 1 customers without
///            a local fortress.
///
/// Consul unreachability / API errors are never fatal — the caller always
/// gets `None` and the local alarm grade stands unchanged.
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tokio::task::JoinHandle;

// ── Local backend types ───────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct VoiceVerdict {
    ok: bool,
}

#[derive(Debug, Deserialize)]
struct RawConsulResponse {
    decision: String,
    consulted_voices: Vec<VoiceVerdict>,
}

#[derive(Debug, Serialize)]
struct ConsulQueryRequest {
    query: String,
    caller: Option<String>,
    context: Option<String>,
    require_security: Option<bool>,
    max_output_tokens: Option<u32>,
}

// ── Cloud backend types ───────────────────────────────────────────────

#[derive(Debug, Serialize)]
struct ClaudeMessage {
    role: String,
    content: String,
}

#[derive(Debug, Serialize)]
struct ClaudeRequest {
    model: String,
    max_tokens: u32,
    messages: Vec<ClaudeMessage>,
}

#[derive(Debug, Deserialize)]
struct ClaudeContentBlock {
    #[serde(rename = "type")]
    block_type: String,
    text: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ClaudeResponse {
    content: Vec<ClaudeContentBlock>,
}

/// Shape we expect Claude to return as JSON inside its text block.
#[derive(Debug, Deserialize)]
struct ClaudeVerdict {
    decision: String,
    confidence: f64,
}

// ── Public types ──────────────────────────────────────────────────────

/// A distilled Consul deliberation result.
#[derive(Debug, Clone)]
pub struct ConsulDecision {
    /// Consul's textual verdict ("approve", "deny", "escalate", "monitor", …).
    pub decision: String,
    /// Fraction of voices that returned `ok = true`, or Claude's self-reported
    /// confidence (0.0 – 1.0).
    pub confidence: f64,
    /// How many voices / API calls participated.
    pub voices: usize,
}

// ── Backend enum ──────────────────────────────────────────────────────

#[derive(Clone)]
enum Backend {
    Local { url: String },
    Cloud { api_key: String },
}

// ── Client ────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct ConsulClient {
    pub timeout_ms: u64,
    backend: Backend,
    client: Client,
}

impl ConsulClient {
    /// Create a client from `consul_url` and `timeout_ms`.
    ///
    /// Reads `CONSUL_BACKEND` at construction time:
    ///   `local`  → calls `consul_url`/consul/query  (default)
    ///   `cloud`  → calls Anthropic API, ignores `consul_url`
    pub fn new(consul_url: String, timeout_ms: u64) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_millis(timeout_ms + 500))
            .build()
            .expect("reqwest client for ConsulClient");

        let backend_env = std::env::var("CONSUL_BACKEND")
            .unwrap_or_else(|_| "local".to_string());

        let backend = if backend_env.trim().eq_ignore_ascii_case("cloud") {
            let api_key = std::env::var("ANTHROPIC_API_KEY")
                .unwrap_or_default();
            if api_key.is_empty() {
                tracing::warn!("CONSUL_BACKEND=cloud but ANTHROPIC_API_KEY is unset — falling back to local");
                Backend::Local { url: consul_url }
            } else {
                tracing::info!("consul: cloud backend (Claude API)");
                Backend::Cloud { api_key }
            }
        } else {
            tracing::info!("consul: local backend → {consul_url}");
            Backend::Local { url: consul_url }
        };

        Self { timeout_ms, backend, client }
    }

    /// Fire a Consul query in a background task and return its handle.
    ///
    /// Call with `tokio::time::timeout(…, handle).await` to merge the result
    /// if it arrives in time, or drop the handle for fire-and-forget.
    pub fn query_async(&self, question: &str) -> JoinHandle<Option<ConsulDecision>> {
        let client = self.client.clone();
        let timeout_ms = self.timeout_ms;
        let question = question.to_string();
        let backend = self.backend.clone();

        tokio::spawn(async move {
            match backend {
                Backend::Local { url } => {
                    query_local(client, url, question, timeout_ms).await
                }
                Backend::Cloud { api_key } => {
                    query_cloud(client, api_key, question, timeout_ms).await
                }
            }
        })
    }
}

impl Default for ConsulClient {
    fn default() -> Self {
        Self::new("http://127.0.0.1:7710".to_string(), 80)
    }
}

// ── Local backend ─────────────────────────────────────────────────────

async fn query_local(
    client: Client,
    url: String,
    question: String,
    timeout_ms: u64,
) -> Option<ConsulDecision> {
    let endpoint = format!("{url}/consul/query");
    let req = ConsulQueryRequest {
        query: question,
        caller: Some("nuclear-eye".to_string()),
        context: Some("alarm grader high-severity security assessment".to_string()),
        require_security: Some(true),
        max_output_tokens: Some(150),
    };

    let result = tokio::time::timeout(
        Duration::from_millis(timeout_ms),
        client.post(&endpoint).json(&req).send(),
    )
    .await;

    match result {
        Ok(Ok(resp)) if resp.status().is_success() => {
            match resp.json::<RawConsulResponse>().await {
                Ok(r) => {
                    let total = r.consulted_voices.len();
                    let ok_votes = r.consulted_voices.iter().filter(|v| v.ok).count();
                    let confidence = if total > 0 { ok_votes as f64 / total as f64 } else { 0.5 };
                    Some(ConsulDecision { decision: r.decision, confidence, voices: total })
                }
                Err(e) => { tracing::debug!("consul local: parse error: {e}"); None }
            }
        }
        Ok(Ok(resp)) => { tracing::debug!("consul local: status {}", resp.status()); None }
        Ok(Err(e)) => { tracing::debug!("consul local: request error: {e}"); None }
        Err(_) => { tracing::debug!("consul local: timed out (>{timeout_ms} ms)"); None }
    }
}

// ── Cloud backend ─────────────────────────────────────────────────────

const CLAUDE_API_URL: &str = "https://api.anthropic.com/v1/messages";
const CLAUDE_MODEL: &str = "claude-haiku-4-5-20251001";
const CLOUD_TIMEOUT_MS: u64 = 5_000;

async fn query_cloud(
    client: Client,
    api_key: String,
    question: String,
    _timeout_ms: u64,
) -> Option<ConsulDecision> {
    let prompt = format!(
        "You are a home security AI consultant. A high-severity alarm was triggered.\n\
         Assess the situation and respond ONLY with a JSON object on a single line:\n\
         {{\"decision\": \"<escalate|monitor|dismiss>\", \"confidence\": <0.0-1.0>}}\n\n\
         Alarm context: {question}"
    );

    let req = ClaudeRequest {
        model: CLAUDE_MODEL.to_string(),
        max_tokens: 120,
        messages: vec![ClaudeMessage { role: "user".to_string(), content: prompt }],
    };

    let result = tokio::time::timeout(
        Duration::from_millis(CLOUD_TIMEOUT_MS),
        client
            .post(CLAUDE_API_URL)
            .header("x-api-key", &api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&req)
            .send(),
    )
    .await;

    match result {
        Ok(Ok(resp)) if resp.status().is_success() => {
            match resp.json::<ClaudeResponse>().await {
                Ok(r) => {
                    let text = r.content.into_iter()
                        .find(|b| b.block_type == "text")
                        .and_then(|b| b.text)?;

                    // Try to parse the JSON verdict Claude was asked to produce.
                    // If it added prose around it, find the first `{…}` substring.
                    let verdict: ClaudeVerdict = if let Ok(v) = serde_json::from_str(&text) {
                        v
                    } else if let Some(start) = text.find('{') {
                        if let Some(end) = text[start..].find('}') {
                            let slice = &text[start..=start + end];
                            serde_json::from_str(slice).ok()?
                        } else { return None; }
                    } else {
                        return None;
                    };

                    tracing::debug!(
                        "consul cloud: decision={} confidence={:.2}",
                        verdict.decision, verdict.confidence
                    );
                    Some(ConsulDecision {
                        decision: verdict.decision,
                        confidence: verdict.confidence.clamp(0.0, 1.0),
                        voices: 1,
                    })
                }
                Err(e) => { tracing::debug!("consul cloud: parse error: {e}"); None }
            }
        }
        Ok(Ok(resp)) => { tracing::debug!("consul cloud: API status {}", resp.status()); None }
        Ok(Err(e)) => { tracing::debug!("consul cloud: request error: {e}"); None }
        Err(_) => { tracing::debug!("consul cloud: timed out (>{CLOUD_TIMEOUT_MS} ms)"); None }
    }
}
