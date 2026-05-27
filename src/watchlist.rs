//! Offender / known-person watchlist matching.
//!
//! Matches a probe face embedding (512-dim ArcFace) against enrolled watchlist
//! entries via cosine similarity (`face_embedding::cosine_similarity`). This is
//! pure logic — no I/O, no models — so it unit-tests deterministically. The
//! encrypted storage (face_db, per-tenant DEK) and enrollment API live
//! elsewhere; this module is the matching + risk-contribution core that feeds
//! `VisionEvent` → the L1/L2 decision path.
//!
//! Status semantics:
//!   * `Authorized` — resident/staff: a match *suppresses* alarms.
//!   * `Watch`      — person of interest: flag, no auto-action.
//!   * `Offender`   — banned/known offender: raise threat, escalate.
//!
//! Safety: a match is *advisory* — it boosts `risk_score`; it does NOT directly
//! trigger an actuator. Human-in-the-loop confirmation gates any offender action
//! (false-positive / wrongful-ID liability — see compliance notes).

use crate::face_embedding::cosine_similarity;
use serde::{Deserialize, Serialize};

/// Default ArcFace match threshold (cosine). Mirrors `face_embedding` / os/17.
pub const DEFAULT_MATCH_THRESHOLD: f32 = 0.28;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum WatchStatus {
    /// Known-authorized (resident, staff) — a match suppresses alarms.
    Authorized,
    /// Person of interest — flag, no auto-action.
    Watch,
    /// Banned / known offender — high threat, escalate.
    Offender,
}

/// An enrolled watchlist entry: an identity + its reference embedding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WatchlistEntry {
    pub id: String,
    pub label: String,
    pub status: WatchStatus,
    /// 0..=3 (only meaningful for `Watch` / `Offender`).
    pub threat_level: u8,
    /// 512-dim ArcFace embedding (decrypted in memory only).
    pub embedding: Vec<f32>,
}

/// The result of matching a probe against the watchlist.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WatchlistHit {
    pub id: String,
    pub label: String,
    pub status: WatchStatus,
    pub threat_level: u8,
    /// Cosine similarity of the winning match, in [-1, 1].
    pub similarity: f32,
}

impl WatchlistHit {
    /// Contribution to `VisionEvent.risk_score` (caller clamps to [0,1]).
    /// Authorized matches *reduce* risk (suppress); Watch/Offender raise it,
    /// scaled by threat level.
    pub fn risk_delta(&self) -> f64 {
        match self.status {
            WatchStatus::Authorized => -0.5,
            WatchStatus::Watch => 0.30,
            WatchStatus::Offender => 0.30 + 0.20 * f64::from(self.threat_level.min(3)),
        }
    }

    /// True when the hit should be surfaced as a security flag (not a suppress).
    pub fn is_flag(&self) -> bool {
        !matches!(self.status, WatchStatus::Authorized)
    }
}

/// Match a probe embedding against the watchlist.
///
/// Returns the best entry whose cosine similarity ≥ `threshold`. "Best" = highest
/// similarity, tie-broken toward higher `threat_level` (so a borderline offender
/// outranks an equally-similar authorized entry). Entries whose embedding
/// dimension differs from the probe are skipped (defensive). `None` = no match.
pub fn match_face(probe: &[f32], entries: &[WatchlistEntry], threshold: f32) -> Option<WatchlistHit> {
    entries
        .iter()
        .filter(|e| e.embedding.len() == probe.len() && !probe.is_empty())
        .filter_map(|e| {
            let sim = cosine_similarity(probe, &e.embedding);
            (sim >= threshold).then_some((e, sim))
        })
        .max_by(|(ea, sa), (eb, sb)| {
            sa.partial_cmp(sb)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(ea.threat_level.cmp(&eb.threat_level))
        })
        .map(|(e, sim)| WatchlistHit {
            id: e.id.clone(),
            label: e.label.clone(),
            status: e.status,
            threat_level: e.threat_level,
            similarity: sim,
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(id: &str, status: WatchStatus, threat: u8, emb: Vec<f32>) -> WatchlistEntry {
        WatchlistEntry { id: id.into(), label: id.into(), status, threat_level: threat, embedding: emb }
    }

    #[test]
    fn exact_match_offender_hits() {
        let v = vec![1.0, 0.0, 0.0, 0.0];
        let wl = vec![entry("o1", WatchStatus::Offender, 3, v.clone())];
        let hit = match_face(&v, &wl, DEFAULT_MATCH_THRESHOLD).expect("should match");
        assert_eq!(hit.id, "o1");
        assert_eq!(hit.status, WatchStatus::Offender);
        assert!((hit.similarity - 1.0).abs() < 1e-5);
        assert!(hit.is_flag());
    }

    #[test]
    fn orthogonal_probe_no_match() {
        let wl = vec![entry("o1", WatchStatus::Offender, 3, vec![1.0, 0.0, 0.0, 0.0])];
        let probe = vec![0.0, 1.0, 0.0, 0.0]; // cosine 0 < 0.28
        assert!(match_face(&probe, &wl, DEFAULT_MATCH_THRESHOLD).is_none());
    }

    #[test]
    fn authorized_match_suppresses_risk() {
        let v = vec![0.0, 1.0, 0.0, 0.0];
        let wl = vec![entry("staff", WatchStatus::Authorized, 0, v.clone())];
        let hit = match_face(&v, &wl, DEFAULT_MATCH_THRESHOLD).unwrap();
        assert!(!hit.is_flag());
        assert!(hit.risk_delta() < 0.0, "authorized must reduce risk");
    }

    #[test]
    fn offender_threat_scales_risk() {
        let mk = |t| WatchlistHit { id: "x".into(), label: "x".into(), status: WatchStatus::Offender, threat_level: t, similarity: 0.9 };
        assert!(mk(3).risk_delta() > mk(1).risk_delta());
    }

    #[test]
    fn picks_highest_similarity() {
        let probe = vec![1.0, 0.0, 0.0, 0.0];
        let wl = vec![
            entry("near", WatchStatus::Watch, 1, vec![0.9, 0.1, 0.0, 0.0]),
            entry("exact", WatchStatus::Watch, 1, vec![1.0, 0.0, 0.0, 0.0]),
        ];
        let hit = match_face(&probe, &wl, DEFAULT_MATCH_THRESHOLD).unwrap();
        assert_eq!(hit.id, "exact");
    }

    #[test]
    fn dimension_mismatch_skipped() {
        let probe = vec![1.0, 0.0, 0.0, 0.0];
        let wl = vec![entry("bad", WatchStatus::Offender, 3, vec![1.0, 0.0])]; // wrong dim
        assert!(match_face(&probe, &wl, DEFAULT_MATCH_THRESHOLD).is_none());
    }

    #[test]
    fn empty_probe_no_match() {
        let wl = vec![entry("o1", WatchStatus::Offender, 3, vec![])];
        assert!(match_face(&[], &wl, DEFAULT_MATCH_THRESHOLD).is_none());
    }
}
