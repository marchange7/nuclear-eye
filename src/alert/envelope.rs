// AUTO-GENERATED FROM nuclear-sdk/schemas/sentinelle-alert.v1.json
// schema-hash: e3a4f792910d448c313690a9a6574dd454978ff8831ec9f7c2ae28af0cd559da
// contract:    os/70-sentinelle-judgment-contract.md  4
// regenerate:  cd nuclear-sdk && bash scripts/gen-schema-types.sh
//
// DO NOT HAND-EDIT. Modify schemas/sentinelle-alert.v1.json (with a major bump)
// or regenerate via the script above. The schema-hash header is checked by
// nuclear-sdk/scripts/verify-generated-types.sh — drift fails CI.

//! Canonical Rust envelope mirror of `sentinelle-alert.v1.json`.
//!
//! This is the *single* place inside `nuclear-eye` where the canonical
//! `SentinelleAlert` envelope shape is declared. `alarm_grader_agent` and any
//! other binary that emits or consumes a graded alert constructs / deserialises
//! `SentinelleAlert` directly — never an ad-hoc `serde_json::Value`. That's
//! the whole point of `os/70 7`: every consumer binds to one type generated
//! from one schema.
//!
//! Today this module is hand-written. When `nuclear-sdk/scripts/gen-schema-types.sh`
//! grows a real `typify` step, this file becomes generator output; the
//! schema-hash header keeps the contract identical either way.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// SHA-256 of `nuclear-sdk/schemas/sentinelle-alert.v1.json` at the time this
/// file was generated. The drift check in
/// `nuclear-sdk/scripts/verify-generated-types.sh` compares this constant to
/// the live schema hash.
pub const SCHEMA_HASH: &str =
    "e3a4f792910d448c313690a9a6574dd454978ff8831ec9f7c2ae28af0cd559da";

/// Schema major version — `1` for v1.json. Bumped only when the JSON Schema
/// file is renamed to `.v2.json` etc.
pub const SCHEMA_VERSION: u32 = 1;

// ─── Enums (controlled vocab — extending requires a major bump) ─────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AlertSeverity {
    Info,
    Low,
    Medium,
    High,
    Critical,
}

impl AlertSeverity {
    /// Numeric ranking, useful for ordering in feeds / Postgres ORDER BY.
    pub const fn rank(self) -> u8 {
        match self {
            Self::Info => 0,
            Self::Low => 1,
            Self::Medium => 2,
            Self::High => 3,
            Self::Critical => 4,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AlertEventType {
    PerimeterBreach,
    IdentityUnrecognized,
    MotionInRestrictedZone,
    CameraTamper,
    GlassBreak,
    FireOrSmoke,
    PanicTriggered,
    SystemDegraded,
    /// Synthetic event for drills; never triggers emergency dispatch.
    TestEvent,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceKind {
    VisionInference,
    IdentityMatch,
    SensorReading,
    PerimeterState,
    CrewVerdict,
    OperatorOverride,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DegradedFlag {
    CameraOffline,
    PerceptionDegraded,
    IdentityDbUnavailable,
    KernelUnreachable,
    ChainUnavailable,
    Lucky7Unavailable,
    WrapperUnavailable,
    LowConfidence,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecommendedActionPrimary {
    NotifyUser,
    NotifyOperator,
    SilentLog,
    ArmEscalation,
    DispatchEmergency,
    RequestOperatorAck,
    Defer,
}

// ─── Sub-objects ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AlertEventSource {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub camera_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sensor_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub zone_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AlertEvent {
    #[serde(rename = "type")]
    pub kind: AlertEventType,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subtype: Option<String>,
    pub severity: AlertSeverity,
    pub source: AlertEventSource,
    pub observed_at: DateTime<Utc>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration_ms: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AlertConfidence {
    /// 0.0..=1.0
    pub overall: f32,
    /// Per-component scores; component names are open vocabulary
    /// (e.g. `vision`, `identity_match`, `perimeter_state`,
    /// `temporal_correlation`).
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub components: BTreeMap<String, f32>,
    pub method: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AlertEvidence {
    pub kind: EvidenceKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    /// 0.0..=1.0
    pub confidence: f32,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub frame_refs: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub redactions: Vec<String>,
    /// Schema allows additionalProperties on evidence items so kind-specific
    /// fields (`identity_match.match_id`, `temporal_correlation.related_alert_ids`,
    /// etc.) can ride along. This bag holds them without losing data through
    /// the round-trip.
    #[serde(flatten)]
    pub extra: BTreeMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AlertReason {
    /// Schema enforces `<= 280` chars; callers should clip on the way in.
    pub summary: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail_markdown: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AlertDegraded {
    pub any: bool,
    pub flags: Vec<DegradedFlag>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AlertRecommendedAction {
    pub primary: RecommendedActionPrimary,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub secondary: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub requires_biometric_auth: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub requires_operator_ack: Option<bool>,
    /// When `Some`, surfaces MUST render a countdown timer + cancel control.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reversible_until: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AlertChainEnvelope {
    /// Mirrors `chain-envelope.v1.json`. The signing layer fills `chain_hash`
    /// + `signer` + `prev_alert_hash`; a freshly-graded alert pre-sign carries
    /// `signed=false` and null hashes.
    pub signed: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chain_hash: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prev_alert_hash: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub signer: Option<String>,
}

// ─── Top-level envelope ─────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SentinelleAlert {
    /// Always `1` for this file. A `SentinelleAlertV2` lands as a sibling
    /// type when the schema bumps.
    pub schema_version: u32,
    /// ULID, monotonic per tenant.
    pub alert_id: String,
    /// UUID.
    pub tenant_id: String,
    /// Always `"sentinelle"`.
    pub product: String,
    pub issued_at: DateTime<Utc>,
    pub issued_by: String,
    pub event: AlertEvent,
    pub confidence: AlertConfidence,
    pub evidence: Vec<AlertEvidence>,
    pub reason: AlertReason,
    pub degraded: AlertDegraded,
    pub recommended_action: AlertRecommendedAction,
    pub chain: AlertChainEnvelope,
}

impl SentinelleAlert {
    /// Returns `true` when the surface MUST gate the recommended action
    /// behind biometric auth (os/70 6 rendering rule).
    pub fn requires_biometric_before_action(&self) -> bool {
        matches!(
            self.recommended_action.primary,
            RecommendedActionPrimary::DispatchEmergency
        ) || self.recommended_action.requires_biometric_auth == Some(true)
    }

    /// User-visible degraded flags (empty when `degraded.any == false`).
    pub fn visible_degraded_flags(&self) -> &[DegradedFlag] {
        if self.degraded.any {
            &self.degraded.flags
        } else {
            &[]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> SentinelleAlert {
        SentinelleAlert {
            schema_version: 1,
            alert_id: "01HZX".to_string() + &"A".repeat(21),
            tenant_id: "00000000-0000-4000-8000-000000000001".into(),
            product: "sentinelle".into(),
            issued_at: "2026-05-02T12:34:56.789Z".parse().unwrap(),
            issued_by: "alarm-grader@b450".into(),
            event: AlertEvent {
                kind: AlertEventType::PerimeterBreach,
                subtype: Some("rear_door_forced".into()),
                severity: AlertSeverity::High,
                source: AlertEventSource {
                    camera_id: Some("cam-rear".into()),
                    sensor_id: None,
                    zone_id: Some("zone-back-yard".into()),
                },
                observed_at: "2026-05-02T12:34:55.210Z".parse().unwrap(),
                duration_ms: Some(1820),
            },
            confidence: AlertConfidence {
                overall: 0.87,
                components: BTreeMap::from([
                    ("vision".to_string(), 0.91),
                    ("identity_match".to_string(), 0.62),
                ]),
                method: "alarm-grader-v2.3".into(),
            },
            evidence: vec![AlertEvidence {
                kind: EvidenceKind::VisionInference,
                model: Some("fastvlm-v1".into()),
                label: Some("human_forced_entry".into()),
                confidence: 0.91,
                frame_refs: vec![],
                redactions: vec![],
                extra: BTreeMap::new(),
            }],
            reason: AlertReason {
                summary: "rear door forced while armed".into(),
                detail_markdown: None,
            },
            degraded: AlertDegraded {
                any: false,
                flags: vec![],
            },
            recommended_action: AlertRecommendedAction {
                primary: RecommendedActionPrimary::DispatchEmergency,
                secondary: vec!["silent_panic".into()],
                requires_biometric_auth: Some(true),
                requires_operator_ack: Some(false),
                reversible_until: Some("2026-05-02T12:35:16.789Z".parse().unwrap()),
            },
            chain: AlertChainEnvelope {
                signed: true,
                chain_hash: Some("sha256:abc".into()),
                prev_alert_hash: None,
                signer: Some("alarm-grader-b450".into()),
            },
        }
    }

    #[test]
    fn round_trips_canonical_json() {
        let alert = fixture();
        let json = serde_json::to_string(&alert).expect("serialise");
        let back: SentinelleAlert = serde_json::from_str(&json).expect("deserialise");
        assert_eq!(alert, back);
    }

    #[test]
    fn snake_case_on_the_wire() {
        let alert = fixture();
        let v: serde_json::Value = serde_json::to_value(&alert).unwrap();
        // Schema field names are snake_case; canonical JSON MUST match.
        for required in &[
            "schema_version",
            "alert_id",
            "tenant_id",
            "product",
            "issued_at",
            "issued_by",
            "event",
            "confidence",
            "evidence",
            "reason",
            "degraded",
            "recommended_action",
            "chain",
        ] {
            assert!(
                v.get(*required).is_some(),
                "required field missing on wire: {required}"
            );
        }
        assert_eq!(v["product"], "sentinelle");
        assert_eq!(v["schema_version"], 1);
    }

    #[test]
    fn dispatch_emergency_requires_biometric() {
        let alert = fixture();
        assert!(alert.requires_biometric_before_action());
    }

    #[test]
    fn degraded_off_means_no_visible_flags() {
        let alert = fixture();
        assert!(alert.visible_degraded_flags().is_empty());
    }

    #[test]
    fn unknown_evidence_extra_round_trips() {
        // Schema declares `additionalProperties: true` on evidence items so
        // identity_match.match_id and similar kind-specific fields ride along
        // in `extra`.
        let mut alert = fixture();
        alert.evidence[0].kind = EvidenceKind::IdentityMatch;
        alert.evidence[0].extra.insert(
            "match_id".into(),
            serde_json::Value::String("face-42".into()),
        );
        let json = serde_json::to_string(&alert).unwrap();
        assert!(
            json.contains("\"match_id\":\"face-42\""),
            "evidence.extra must serialise inline; got: {json}"
        );
        let back: SentinelleAlert = serde_json::from_str(&json).unwrap();
        assert_eq!(
            back.evidence[0].extra.get("match_id"),
            Some(&serde_json::Value::String("face-42".into()))
        );
    }

    #[test]
    fn schema_hash_constant_is_present() {
        // Drift is enforced by the verify-generated-types.sh script in
        // nuclear-sdk; this test just makes sure the constant is exported and
        // non-empty so a typo caught at compile-time, not at audit-time.
        assert_eq!(SCHEMA_HASH.len(), 64);
        assert_eq!(SCHEMA_VERSION, 1);
    }
}

