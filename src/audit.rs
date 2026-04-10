/// Q5 — Audit log trail (GDPR-ready, append-only JSONL).
///
/// Appends one JSON line per alarm verdict to AUDIT_LOG_PATH.
/// Default path: /var/log/nuclear-eye/audit.jsonl
///
/// GDPR note:
///   - No PII is stored in this file. `camera_id` is a location label (e.g.
///     "front-door"), not a person identifier. `verdict` and `action` are
///     operational outcomes. Retain for no more than 30 days.
///   - Auto-rotation: if the log file is older than 30 days it is renamed to
///     `audit.jsonl.old` before the new entry is written. The old file should
///     be securely deleted by a separate scheduled task (cron / Lucky7).
///   - voice_refs, face embeddings, and person names are never written here.
use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::Path,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use tracing::warn;

/// Default path for the audit log. Override with `AUDIT_LOG_PATH` env var.
const DEFAULT_AUDIT_LOG_PATH: &str = "/var/log/nuclear-eye/audit.jsonl";

/// Maximum age of the audit log file before rotation (30 days).
const MAX_AGE_DAYS: u64 = 30;

/// Append one structured decision record to the audit log.
///
/// - `camera_id`   — location label (no PII)
/// - `alarm_type`  — behavior / alarm type string (e.g. "loitering")
/// - `verdict`     — alarm level: "none" | "low" | "medium" | "high"
/// - `confidence`  — danger score [0.0, 1.0]
/// - `action`      — decided action (e.g. "Alarm", "Reassure", "Pause")
///
/// This function is synchronous and intentionally cheap (single file append).
/// Call it from a `tokio::task::spawn_blocking` block if you need async context.
pub fn log_decision(
    camera_id: &str,
    alarm_type: &str,
    verdict: &str,
    confidence: f32,
    action: &str,
) {
    let path = std::env::var("AUDIT_LOG_PATH")
        .unwrap_or_else(|_| DEFAULT_AUDIT_LOG_PATH.to_string());

    let path = Path::new(&path);

    // Ensure parent directory exists.
    if let Some(parent) = path.parent() {
        if let Err(e) = fs::create_dir_all(parent) {
            warn!(error = %e, "audit: failed to create log directory {:?}", parent);
            return;
        }
    }

    // GDPR rotation: rename to .old if file is older than MAX_AGE_DAYS.
    maybe_rotate(path);

    // Build ISO 8601 timestamp without chrono to keep this module dependency-free.
    let ts = iso8601_now();

    let line = format!(
        "{{\"ts\":\"{ts}\",\"camera_id\":\"{camera_id}\",\"alarm_type\":\"{alarm_type}\",\
         \"verdict\":\"{verdict}\",\"confidence\":{confidence:.4},\"action\":\"{action}\"}}\n",
    );

    match OpenOptions::new().create(true).append(true).open(path) {
        Ok(mut f) => {
            if let Err(e) = f.write_all(line.as_bytes()) {
                warn!(error = %e, "audit: write failed");
            }
        }
        Err(e) => {
            warn!(error = %e, path = %path.display(), "audit: open failed");
        }
    }
}

/// Rotate the audit log if it is older than MAX_AGE_DAYS.
///
/// The old file is renamed to `<path>.old`. Any previous `.old` is overwritten.
/// A missing or inaccessible file is silently skipped.
fn maybe_rotate(path: &Path) {
    let metadata = match fs::metadata(path) {
        Ok(m) => m,
        Err(_) => return, // file doesn't exist yet — nothing to rotate
    };

    let age = metadata
        .modified()
        .ok()
        .and_then(|mtime| SystemTime::now().duration_since(mtime).ok())
        .unwrap_or(Duration::ZERO);

    if age > Duration::from_secs(MAX_AGE_DAYS * 24 * 3600) {
        let old_path = path.with_extension("jsonl.old");
        if let Err(e) = fs::rename(path, &old_path) {
            warn!(
                error = %e,
                "audit: rotation failed — could not rename {:?} to {:?}",
                path, old_path,
            );
        } else {
            tracing::info!(
                "audit: rotated {:?} → {:?} (file older than {} days)",
                path, old_path, MAX_AGE_DAYS,
            );
        }
    }
}

/// Produce an ISO 8601 / RFC 3339 UTC timestamp without pulling in `chrono`.
///
/// Format: `2026-04-09T12:34:56Z`
fn iso8601_now() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::ZERO)
        .as_secs();

    // Julian-day based decomposition (Gregorian calendar, proleptic for years ≥ 1970).
    let days = secs / 86400;
    let time_of_day = secs % 86400;
    let h = time_of_day / 3600;
    let m = (time_of_day % 3600) / 60;
    let s = time_of_day % 60;

    // Convert Julian Day Number (JDN) for 1970-01-01 = 2440588.
    let jdn = days + 2440588;
    let f = jdn + 1401 + (((4 * jdn + 274277) / 146097) * 3) / 4 - 38;
    let e = 4 * f + 3;
    let g = (e % 1461) / 4;
    let h_cal = 5 * g + 2;

    let day   = (h_cal % 153) / 5 + 1;
    let month = (h_cal / 153 + 2) % 12 + 1;
    let year  = e / 1461 - 4716 + (14 - month) / 12;

    format!("{year:04}-{month:02}-{day:02}T{h:02}:{m:02}:{s:02}Z")
}

#[cfg(test)]
mod tests {
    use super::iso8601_now;

    #[test]
    fn iso8601_format_valid() {
        let ts = iso8601_now();
        // Basic format check: "YYYY-MM-DDTHH:MM:SSZ"
        assert_eq!(ts.len(), 20, "unexpected length: {ts}");
        assert!(ts.ends_with('Z'));
        assert!(ts.contains('T'));
    }
}
