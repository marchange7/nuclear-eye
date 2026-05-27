//! Phase 4 "photo DB" — enrollment + watchlist registry (pure-Rust logic).
//!
//! This module owns the application-side contract for the encrypted reference-
//! image / watchlist store defined in
//! `migrations/face_db_002_images_watchlist.sql`. It mirrors the
//! `WatchStatus { Authorized, Watch, Offender }` taxonomy that
//! `src/watchlist.rs` matches against — this module persists what that matcher
//! evaluates.
//!
//! Compliance posture (see the migration header for the full statement):
//!   * Reference images + the watchlist are BIOMETRIC SPECIAL-CATEGORY DATA
//!     (GDPR Art. 9 / BIPA). Images are encrypted at rest with a per-tenant DEK.
//!   * `expires_at` drives retention / data-minimisation.
//!   * An `offender` classification is ADVISORY: human-in-the-loop is required
//!     before any offender action. `threat_level` is a triage hint only.
//!
//! DB access is behind the [`WatchlistStore`] trait so the validation / struct
//! logic here is unit-testable without a live Postgres.

use std::fmt;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Watchlist classification.
///
/// Kept byte-aligned with `src/watchlist.rs` and with the
/// `status IN ('authorized','watch','offender')` CHECK in
/// `face_db.watchlist`. The lowercase wire string is the source of truth shared
/// with the SQL layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum WatchStatus {
    /// Trusted identity (family member). No alarm escalation.
    Authorized,
    /// Flagged for elevated attention but not a confirmed threat.
    Watch,
    /// Known intruder / threat. Advisory only — human-in-the-loop required
    /// before any offender action.
    Offender,
}

impl WatchStatus {
    /// Wire / SQL representation. Must match the watchlist CHECK constraint.
    pub fn as_str(self) -> &'static str {
        match self {
            WatchStatus::Authorized => "authorized",
            WatchStatus::Watch => "watch",
            WatchStatus::Offender => "offender",
        }
    }

    /// Parse from the SQL/wire string. Case-insensitive on input.
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "authorized" => Some(WatchStatus::Authorized),
            "watch" => Some(WatchStatus::Watch),
            "offender" => Some(WatchStatus::Offender),
            _ => None,
        }
    }
}

impl fmt::Display for WatchStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Maximum permitted `threat_level` (mirrors the `0..3` CHECK in SQL).
pub const MAX_THREAT_LEVEL: u8 = 3;

/// Allowed reference-image MIME types.
pub const ALLOWED_MIMES: [&str; 2] = ["image/jpeg", "image/png"];

/// A request to enrol an identity into the watchlist with a reference image.
///
/// The raw image bytes themselves are passed separately to the [`WatchlistStore`]
/// so they can be encrypted in-flight (per-tenant DEK) and never logged; this
/// struct carries only the validated metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnrollRequest {
    /// Human-readable label for the identity (e.g. "Alice", "unknown-intruder-3").
    pub label: String,
    /// Watchlist classification.
    pub status: WatchStatus,
    /// Triage hint, 0..=3. Advisory only.
    pub threat_level: u8,
    /// MIME type of the reference image (image/jpeg | image/png).
    pub image_mime: String,
    /// Optional free-text reason (why this identity is on the watchlist).
    #[serde(default)]
    pub reason: Option<String>,
    /// Operator / system that added the entry (for audit).
    #[serde(default)]
    pub added_by: Option<String>,
    /// Optional retention deadline. `None` = no auto-expiry (manual review).
    #[serde(default)]
    pub expires_at: Option<DateTime<Utc>>,
}

/// Validation failures for [`validate_enroll`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EnrollmentError {
    /// `label` was empty or whitespace-only.
    EmptyLabel,
    /// `threat_level` exceeded [`MAX_THREAT_LEVEL`].
    ThreatLevelTooHigh(u8),
    /// `image_mime` was not in [`ALLOWED_MIMES`].
    UnsupportedMime(String),
    /// `expires_at` was set but already in the past.
    ExpiryInPast,
}

impl fmt::Display for EnrollmentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EnrollmentError::EmptyLabel => write!(f, "label must be non-empty"),
            EnrollmentError::ThreatLevelTooHigh(n) => {
                write!(f, "threat_level {n} exceeds max {MAX_THREAT_LEVEL}")
            }
            EnrollmentError::UnsupportedMime(m) => {
                write!(f, "unsupported image mime '{m}' (allowed: {ALLOWED_MIMES:?})")
            }
            EnrollmentError::ExpiryInPast => write!(f, "expires_at is in the past"),
        }
    }
}

impl std::error::Error for EnrollmentError {}

/// Validate an [`EnrollRequest`] before it touches the store.
///
/// Checks, in order:
///   * `label` is non-empty (after trimming),
///   * `threat_level <= MAX_THREAT_LEVEL`,
///   * `image_mime` is one of [`ALLOWED_MIMES`] (case-insensitive),
///   * if `expires_at` is set, it is strictly in the future (relative to now).
///
/// `status` is type-checked by [`WatchStatus`] at the boundary, so an invalid
/// status string can never reach here.
pub fn validate_enroll(req: &EnrollRequest) -> Result<(), EnrollmentError> {
    if req.label.trim().is_empty() {
        return Err(EnrollmentError::EmptyLabel);
    }
    if req.threat_level > MAX_THREAT_LEVEL {
        return Err(EnrollmentError::ThreatLevelTooHigh(req.threat_level));
    }
    let mime = req.image_mime.trim().to_ascii_lowercase();
    if !ALLOWED_MIMES.contains(&mime.as_str()) {
        return Err(EnrollmentError::UnsupportedMime(req.image_mime.clone()));
    }
    if let Some(exp) = req.expires_at {
        if exp <= Utc::now() {
            return Err(EnrollmentError::ExpiryInPast);
        }
    }
    Ok(())
}

/// True when `expires_at` is set and is at or before `now`.
///
/// A `None` expiry means "no auto-expiry" and is therefore never expired.
pub fn is_expired(now: DateTime<Utc>, expires_at: Option<DateTime<Utc>>) -> bool {
    match expires_at {
        Some(exp) => exp <= now,
        None => false,
    }
}

/// A persisted watchlist entry, as returned by the store.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WatchlistEntry {
    pub id: uuid::Uuid,
    pub face_id: Option<uuid::Uuid>,
    pub label: String,
    pub status: WatchStatus,
    pub threat_level: u8,
    pub reason: Option<String>,
    pub added_by: Option<String>,
    pub added_at: DateTime<Utc>,
    pub expires_at: Option<DateTime<Utc>>,
}

/// DB contract for the encrypted photo DB / watchlist registry.
///
/// Implementations (e.g. a Postgres-backed store gated behind the `face_db_pg`
/// feature) own connection handling, per-tenant `app.tenant_id` /
/// `app.face_db_key` GUC setup, in-flight image encryption, and audit writes.
/// This trait exists so the pure logic in this module can be tested without a
/// live database.
pub trait WatchlistStore {
    type Error;

    /// Enrol an identity: persist the watchlist row and encrypt + store the
    /// reference image (`image_bytes`) in-flight. Returns the new entry id.
    fn enroll(
        &self,
        req: &EnrollRequest,
        face_id: Option<uuid::Uuid>,
        image_bytes: &[u8],
    ) -> Result<uuid::Uuid, Self::Error>;

    /// List all watchlist entries for the current tenant.
    fn list(&self) -> Result<Vec<WatchlistEntry>, Self::Error>;

    /// Remove a watchlist entry (and its reference images) by id.
    fn remove(&self, id: uuid::Uuid) -> Result<(), Self::Error>;

    /// Find watchlist entries by the (logical) face_id handle.
    fn find_by_face(&self, face_id: uuid::Uuid) -> Result<Vec<WatchlistEntry>, Self::Error>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;

    fn valid_request() -> EnrollRequest {
        EnrollRequest {
            label: "Alice".to_string(),
            status: WatchStatus::Authorized,
            threat_level: 0,
            image_mime: "image/jpeg".to_string(),
            reason: None,
            added_by: Some("operator".to_string()),
            expires_at: None,
        }
    }

    #[test]
    fn validate_accepts_valid_request() {
        assert_eq!(validate_enroll(&valid_request()), Ok(()));
    }

    #[test]
    fn validate_accepts_png_and_future_expiry() {
        let mut req = valid_request();
        req.status = WatchStatus::Offender;
        req.threat_level = 3;
        req.image_mime = "image/png".to_string();
        req.expires_at = Some(Utc::now() + Duration::days(30));
        assert_eq!(validate_enroll(&req), Ok(()));
    }

    #[test]
    fn validate_accepts_mixed_case_mime() {
        let mut req = valid_request();
        req.image_mime = "IMAGE/JPEG".to_string();
        assert_eq!(validate_enroll(&req), Ok(()));
    }

    #[test]
    fn validate_rejects_empty_label() {
        let mut req = valid_request();
        req.label = "   ".to_string();
        assert_eq!(validate_enroll(&req), Err(EnrollmentError::EmptyLabel));
    }

    #[test]
    fn validate_rejects_threat_level_above_max() {
        let mut req = valid_request();
        req.threat_level = 4;
        assert_eq!(
            validate_enroll(&req),
            Err(EnrollmentError::ThreatLevelTooHigh(4))
        );
    }

    #[test]
    fn validate_rejects_unsupported_mime() {
        let mut req = valid_request();
        req.image_mime = "image/gif".to_string();
        assert_eq!(
            validate_enroll(&req),
            Err(EnrollmentError::UnsupportedMime("image/gif".to_string()))
        );
    }

    #[test]
    fn validate_rejects_past_expiry() {
        let mut req = valid_request();
        req.expires_at = Some(Utc::now() - Duration::days(1));
        assert_eq!(validate_enroll(&req), Err(EnrollmentError::ExpiryInPast));
    }

    #[test]
    fn is_expired_handles_none_past_future() {
        let now = Utc::now();
        // None never expires.
        assert!(!is_expired(now, None));
        // Past expiry → expired.
        assert!(is_expired(now, Some(now - Duration::seconds(1))));
        // Exactly now → expired (boundary is inclusive).
        assert!(is_expired(now, Some(now)));
        // Future expiry → not expired.
        assert!(!is_expired(now, Some(now + Duration::seconds(1))));
    }

    #[test]
    fn watch_status_roundtrips_with_sql_strings() {
        for s in [WatchStatus::Authorized, WatchStatus::Watch, WatchStatus::Offender] {
            assert_eq!(WatchStatus::parse(s.as_str()), Some(s));
        }
        assert_eq!(WatchStatus::parse("OFFENDER"), Some(WatchStatus::Offender));
        assert_eq!(WatchStatus::parse("bogus"), None);
    }
}
