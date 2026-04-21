//! Bearer-token + multi-tenant header guard for the `face_db` HTTP surface.
//!
//! Sources:
//!   * `os/56-sentinelle-deep-rewire-plan.md` P1-4
//!   * `os/55-sentinelle-cross-repo-audit.md`     CRITICAL biometric encryption,
//!                                                HIGH face_db unauthenticated
//!   * `os/57-multitenant-kernel-architecture.md` §4.7  Pass 1a/1b/1c rollout
//!
//! Closes `os/55` HIGH at the HTTP layer:
//!   * Mutating endpoints (`POST /faces`, `POST /faces/embed`,
//!     `POST /faces/purge`, `GET /faces/gdpr-export`) require a bearer token
//!     when `FACE_DB_TOKEN` is set in the environment. When unset, the binary
//!     logs a one-shot WARN at startup and the endpoints stay open — matching
//!     the pre-existing pattern in `alarm_grader_agent` so deployments without
//!     the env var keep working during rollout.
//!
//! Multi-tenancy semantics mirror the kernel doctrine in `os/57 §4.7`:
//!   * `X-Tenant-Id` header is parsed as a UUID.
//!   * When `KERNEL_REQUIRE_TENANT_HEADER=1` (Pass 1c), a missing/malformed
//!     header is rejected with `401`.
//!   * Otherwise (Pass 1a / 1b — default), a missing/malformed header silently
//!     resolves to `kernel.legacy_default_tenant()` and the request proceeds
//!     so legacy single-tenant deployments don't break overnight.

use axum::http::{header, HeaderMap, StatusCode};
use axum::Json;
use serde_json::json;
use subtle::ConstantTimeEq;
use uuid::Uuid;

/// Legacy default tenant for Pass 1a / 1b — must stay byte-identical with
/// `kernel.legacy_default_tenant()` (`00000000-0000-0000-0000-000000000000`).
pub const LEGACY_DEFAULT_TENANT: Uuid = Uuid::nil();

/// Result of a successful authentication check.
#[derive(Debug, Clone, Copy)]
pub struct AuthContext {
    /// Resolved tenant for this request.
    ///
    /// In Pass 1a/1b this is `LEGACY_DEFAULT_TENANT` whenever the caller
    /// omits the `X-Tenant-Id` header. In Pass 1c a missing header is
    /// rejected before this struct is constructed.
    pub tenant_id: Uuid,
}

/// Failure modes for the auth guard.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuthError {
    /// `FACE_DB_TOKEN` is configured but the request omitted or mismatched it.
    Unauthorized,
    /// Pass-1c-only: `X-Tenant-Id` header is required but was missing.
    TenantHeaderRequired,
    /// `X-Tenant-Id` header is present but isn't a valid UUID.
    MalformedTenantHeader,
}

impl AuthError {
    /// Convert to an `axum`-compatible `(StatusCode, Json)` tuple matching
    /// the existing error envelope used by `alarm_grader_agent`.
    pub fn into_response(self) -> (StatusCode, Json<serde_json::Value>) {
        match self {
            AuthError::Unauthorized => (
                StatusCode::UNAUTHORIZED,
                Json(json!({
                    "error": "unauthorized",
                    "hint":  "set Authorization: Bearer <FACE_DB_TOKEN>",
                })),
            ),
            AuthError::TenantHeaderRequired => (
                StatusCode::UNAUTHORIZED,
                Json(json!({
                    "error": "tenant_header_required",
                    "hint":  "set X-Tenant-Id: <uuid> (KERNEL_REQUIRE_TENANT_HEADER=1)",
                })),
            ),
            AuthError::MalformedTenantHeader => (
                StatusCode::BAD_REQUEST,
                Json(json!({
                    "error": "malformed_tenant_header",
                    "hint":  "X-Tenant-Id must be a valid UUID",
                })),
            ),
        }
    }
}

/// Authenticate an incoming request.
///
/// * `expected_token` — the value of `FACE_DB_TOKEN` at boot, `None` when
///   unset (open mode — matches `alarm_grader_agent` convention).
/// * `require_tenant` — the value of `KERNEL_REQUIRE_TENANT_HEADER` at boot
///   (`true` ⇔ Pass 1c).
pub fn authenticate(
    headers: &HeaderMap,
    expected_token: Option<&str>,
    require_tenant: bool,
) -> Result<AuthContext, AuthError> {
    if let Some(expected) = expected_token {
        let supplied = headers
            .get(header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.strip_prefix("Bearer "))
            .unwrap_or("");
        let supplied_b = supplied.as_bytes();
        let expected_b = expected.as_bytes();
        let lengths_match = (supplied_b.len() == expected_b.len()) as u8;
        let bytes_match = if supplied_b.len() == expected_b.len() {
            supplied_b.ct_eq(expected_b).unwrap_u8()
        } else {
            0
        };
        if lengths_match & bytes_match == 0 {
            return Err(AuthError::Unauthorized);
        }
    }

    let tenant_id = match headers.get("x-tenant-id").and_then(|v| v.to_str().ok()) {
        Some(raw) => match Uuid::parse_str(raw) {
            Ok(uuid) => uuid,
            Err(_) => return Err(AuthError::MalformedTenantHeader),
        },
        None if require_tenant => return Err(AuthError::TenantHeaderRequired),
        None => LEGACY_DEFAULT_TENANT,
    };

    Ok(AuthContext { tenant_id })
}

/// Read `FACE_DB_TOKEN` from the environment, treating an empty string as
/// "unset" (matches `alarm_grader_agent`).
pub fn token_from_env() -> Option<String> {
    std::env::var("FACE_DB_TOKEN").ok().filter(|s| !s.is_empty())
}

/// Read `KERNEL_REQUIRE_TENANT_HEADER` from the environment.
pub fn require_tenant_from_env() -> bool {
    std::env::var("KERNEL_REQUIRE_TENANT_HEADER")
        .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE"))
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::{HeaderName, HeaderValue};

    fn hm(pairs: &[(&str, &str)]) -> HeaderMap {
        let mut h = HeaderMap::new();
        for (k, v) in pairs {
            let name = HeaderName::from_bytes(k.as_bytes()).unwrap();
            h.insert(name, HeaderValue::from_str(v).unwrap());
        }
        h
    }

    #[test]
    fn open_mode_passes_without_token() {
        let ctx = authenticate(&hm(&[]), None, false).expect("open mode allows missing token");
        assert_eq!(ctx.tenant_id, LEGACY_DEFAULT_TENANT);
    }

    #[test]
    fn token_required_when_configured() {
        let err = authenticate(&hm(&[]), Some("secret"), false).unwrap_err();
        assert_eq!(err, AuthError::Unauthorized);
    }

    #[test]
    fn token_match_succeeds() {
        let h = hm(&[("authorization", "Bearer secret")]);
        let ctx = authenticate(&h, Some("secret"), false).unwrap();
        assert_eq!(ctx.tenant_id, LEGACY_DEFAULT_TENANT);
    }

    #[test]
    fn token_mismatch_fails() {
        let h = hm(&[("authorization", "Bearer wrong")]);
        let err = authenticate(&h, Some("secret"), false).unwrap_err();
        assert_eq!(err, AuthError::Unauthorized);
    }

    #[test]
    fn token_length_mismatch_fails_constant_time() {
        let h = hm(&[("authorization", "Bearer short")]);
        let err = authenticate(&h, Some("muchlongerexpectedtoken"), false).unwrap_err();
        assert_eq!(err, AuthError::Unauthorized);
    }

    #[test]
    fn tenant_header_parsed_when_present() {
        let uuid = "11111111-1111-1111-1111-111111111111";
        let h = hm(&[("x-tenant-id", uuid)]);
        let ctx = authenticate(&h, None, false).unwrap();
        assert_eq!(ctx.tenant_id, Uuid::parse_str(uuid).unwrap());
    }

    #[test]
    fn malformed_tenant_header_rejected() {
        let h = hm(&[("x-tenant-id", "not-a-uuid")]);
        let err = authenticate(&h, None, false).unwrap_err();
        assert_eq!(err, AuthError::MalformedTenantHeader);
    }

    #[test]
    fn pass_1c_requires_tenant_header() {
        let err = authenticate(&hm(&[]), None, true).unwrap_err();
        assert_eq!(err, AuthError::TenantHeaderRequired);
    }

    #[test]
    fn pass_1c_accepts_valid_tenant() {
        let uuid = "22222222-2222-2222-2222-222222222222";
        let h = hm(&[("x-tenant-id", uuid)]);
        let ctx = authenticate(&h, None, true).unwrap();
        assert_eq!(ctx.tenant_id, Uuid::parse_str(uuid).unwrap());
    }

    #[test]
    fn pass_1a_falls_back_to_legacy_default() {
        let ctx = authenticate(&hm(&[]), None, false).unwrap();
        assert_eq!(ctx.tenant_id, LEGACY_DEFAULT_TENANT);
        assert_eq!(ctx.tenant_id, Uuid::nil());
    }

    #[test]
    fn auth_error_response_status_codes() {
        let (st, _) = AuthError::Unauthorized.into_response();
        assert_eq!(st, StatusCode::UNAUTHORIZED);
        let (st, _) = AuthError::TenantHeaderRequired.into_response();
        assert_eq!(st, StatusCode::UNAUTHORIZED);
        let (st, _) = AuthError::MalformedTenantHeader.into_response();
        assert_eq!(st, StatusCode::BAD_REQUEST);
    }
}
