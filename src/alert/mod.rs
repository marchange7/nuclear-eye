//! Canonical Sentinelle alert envelope (os/70-sentinelle-judgment-contract.md).
//!
//! Re-exported as `nuclear_eye::alert::SentinelleAlert` so other binaries can
//! `use nuclear_eye::alert::SentinelleAlert` directly.

pub mod envelope;

pub use envelope::{
    AlertChainEnvelope, AlertConfidence, AlertDegraded, AlertEvent, AlertEventSource,
    AlertEventType, AlertEvidence, AlertReason, AlertRecommendedAction, AlertSeverity,
    DegradedFlag, EvidenceKind, RecommendedActionPrimary, SentinelleAlert, SCHEMA_HASH,
    SCHEMA_VERSION,
};

