//! S-7 — Fail-closed wrapper probe for nuclear-eye binaries.
//!
//! When `NUCLEAR_WRAPPER_URL` is set, this module verifies the wrapper is
//! reachable at startup. If unreachable, the process exits(1).
//!
//! When `WRAPPER_REQUIRED=1` is set (and `NUCLEAR_WRAPPER_URL` is absent),
//! the process also exits(1) — ensures no binary runs unguarded in hardened
//! deploys even if the URL was not wired in the compose manifest.
//!
//! Environment:
//!   NUCLEAR_WRAPPER_URL  — HTTP base URL of the nuclear-wrapper sidecar
//!                          (e.g. http://nuclear-wrapper:9090 or http://127.0.0.1:9090)
//!   WRAPPER_REQUIRED     — set to "1" or "true" to fail even when URL is absent;
//!                          defaults to required when NUCLEAR_ENV=production
//!   WRAPPER_PROBE_RETRIES — number of probe attempts (default: 3)
//!   WRAPPER_PROBE_TIMEOUT_MS — per-attempt timeout ms (default: 3000)

use anyhow::{bail, Result};
use std::time::Duration;
use tracing::{info, warn, error};

fn trim_env(key: &str) -> Option<String> {
    let v = std::env::var(key).unwrap_or_default();
    let t = v.trim().to_string();
    if t.is_empty() { None } else { Some(t) }
}

fn parse_bool_env(key: &str) -> bool {
    matches!(
        std::env::var(key).unwrap_or_default().trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes"
    )
}

fn wrapper_required_by_default() -> bool {
    parse_bool_env("WRAPPER_REQUIRED")
        || matches!(
            std::env::var("NUCLEAR_ENV").unwrap_or_default().trim().to_ascii_lowercase().as_str(),
            "prod" | "production"
        )
}

/// Run at binary startup — blocks synchronously until the probe resolves or fails.
///
/// Call this from within a `tokio::main` context.
pub async fn check_wrapper(binary_name: &str) -> Result<()> {
    let wrapper_url = trim_env("NUCLEAR_WRAPPER_URL");
    let wrapper_required = wrapper_required_by_default();

    let retries: u32 = std::env::var("WRAPPER_PROBE_RETRIES")
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(3);
    let timeout_ms: u64 = std::env::var("WRAPPER_PROBE_TIMEOUT_MS")
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(3000);

    match &wrapper_url {
        None => {
            if wrapper_required {
                error!(
                    binary = binary_name,
                    "FATAL: WRAPPER_REQUIRED=1 but NUCLEAR_WRAPPER_URL is not set — \
                     refusing to start unguarded."
                );
                bail!("wrapper_required_but_url_missing");
            }
            // URL absent, not required — running unguarded is acceptable.
            return Ok(());
        }
        Some(url) => {
            let health_url = format!("{}/health", url.trim_end_matches('/'));
            let client = reqwest::Client::builder()
                .timeout(Duration::from_millis(timeout_ms))
                .build()
                .unwrap_or_default();

            let backoff_ms: &[u64] = &[500, 1000, 2000, 4000, 8000];

            for attempt in 1..=retries {
                match client.get(&health_url).send().await {
                    Ok(resp) if resp.status().is_success() => {
                        info!(
                            binary = binary_name,
                            url = %health_url,
                            attempt,
                            "nuclear-wrapper: healthy"
                        );
                        return Ok(());
                    }
                    Ok(resp) => {
                        warn!(
                            binary = binary_name,
                            url = %health_url,
                            status = resp.status().as_u16(),
                            attempt,
                            "nuclear-wrapper: probe non-2xx"
                        );
                    }
                    Err(err) => {
                        warn!(
                            binary = binary_name,
                            url = %health_url,
                            err = %err,
                            attempt,
                            next_retry_ms = if attempt < retries {
                                backoff_ms.get((attempt - 1) as usize).copied().unwrap_or(8000)
                            } else { 0 },
                            "nuclear-wrapper: probe failed"
                        );
                    }
                }

                if attempt < retries {
                    let delay = backoff_ms.get((attempt - 1) as usize).copied().unwrap_or(8000);
                    tokio::time::sleep(Duration::from_millis(delay)).await;
                }
            }

            error!(
                binary = binary_name,
                url = %health_url,
                attempts = retries,
                "FATAL: nuclear-wrapper unreachable after retries — refusing to start unguarded."
            );
            bail!("wrapper_unreachable_after_retries");
        }
    }
}
