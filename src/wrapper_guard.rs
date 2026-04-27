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

/// True when this build must refuse to run unguarded — operator
/// `WRAPPER_REQUIRED=1` OR `NUCLEAR_ENV ∈ {prod, production}`. Same gate
/// `check_wrapper` consults; exposed so each binary's in-process
/// `nuclear_wrapper::wrap!()` fallback can defer to it instead of silently
/// downgrading to "running unguarded" (per code-review 2026-04-26 P1 #9).
pub fn wrapper_required_by_default() -> bool {
    parse_bool_env("WRAPPER_REQUIRED")
        || matches!(
            std::env::var("NUCLEAR_ENV").unwrap_or_default().trim().to_ascii_lowercase().as_str(),
            "prod" | "production"
        )
}

/// Handle an in-process `nuclear_wrapper::wrap!()` startup failure.
///
/// In production (gate above) we log `error!` + exit(1) — security-product
/// binaries must NOT run with tamper/health/discovery sidecars disabled. In
/// dev we log `warn!` (was `info!`) so the operator notices.
///
/// Each `bin/*.rs` calls this from its `Err(e)` arm of the `match wrap!()`
/// block. Centralising the policy means raising the bar in one place
/// raises it for every binary.
pub fn handle_wrap_failure<E: std::fmt::Display>(binary_name: &str, err: &E) {
    if wrapper_required_by_default() {
        error!(
            binary = binary_name,
            err = %err,
            "FATAL: nuclear-wrapper start failed in WRAPPER_REQUIRED / NUCLEAR_ENV=production — \
             refusing to run unguarded"
        );
        std::process::exit(1);
    } else {
        // Was tracing::info!; raised to warn! so it's visible at default log level.
        warn!(
            binary = binary_name,
            err = %err,
            "nuclear-wrapper: start failed — running unguarded (dev mode)"
        );
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Tests in this module mutate process env; serialize so they don't race.
    /// Poison-tolerant per the project's elsewhere-pattern.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn with_env<F: FnOnce()>(vars: &[(&str, Option<&str>)], f: F) {
        let _g = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let prior: Vec<(String, Option<String>)> = vars
            .iter()
            .map(|(k, _)| (k.to_string(), std::env::var(k).ok()))
            .collect();
        for (k, v) in vars {
            match v {
                Some(s) => std::env::set_var(k, s),
                None => std::env::remove_var(k),
            }
        }
        f();
        for (k, v) in prior {
            match v {
                Some(s) => std::env::set_var(&k, s),
                None => std::env::remove_var(&k),
            }
        }
    }

    #[test]
    fn wrapper_required_when_explicit_flag() {
        with_env(
            &[("WRAPPER_REQUIRED", Some("1")), ("NUCLEAR_ENV", None)],
            || {
                assert!(wrapper_required_by_default());
            },
        );
        with_env(
            &[("WRAPPER_REQUIRED", Some("true")), ("NUCLEAR_ENV", None)],
            || {
                assert!(wrapper_required_by_default());
            },
        );
    }

    #[test]
    fn wrapper_required_when_nuclear_env_production() {
        for v in ["prod", "production", "PRODUCTION", "Prod"] {
            with_env(
                &[("WRAPPER_REQUIRED", None), ("NUCLEAR_ENV", Some(v))],
                || {
                    assert!(
                        wrapper_required_by_default(),
                        "NUCLEAR_ENV={v} should require wrapper"
                    );
                },
            );
        }
    }

    #[test]
    fn wrapper_optional_in_dev_default() {
        with_env(
            &[
                ("WRAPPER_REQUIRED", None),
                ("NUCLEAR_ENV", None),
            ],
            || {
                assert!(!wrapper_required_by_default());
            },
        );
        // Explicitly dev-named values must NOT trigger required.
        for v in ["dev", "development", "staging", "test", ""] {
            with_env(
                &[("WRAPPER_REQUIRED", None), ("NUCLEAR_ENV", Some(v))],
                || {
                    assert!(
                        !wrapper_required_by_default(),
                        "NUCLEAR_ENV={v} should NOT require wrapper"
                    );
                },
            );
        }
    }
}
