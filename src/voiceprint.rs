//! voiceprint — pure speaker-identification logic for the voice watchlist.
//!
//! Phase 5 (a): match a probe voice embedding against a curated voice
//! watchlist so Sentinelle can tell **family** from **intruders** by voice
//! alone. This module is deliberately PURE: no live models, no GPU, no
//! network. The embedding extraction (ECAPA-TDNN) lives in a Python sidecar
//! behind [`VoiceprintBackend`]; this file only does the maths and the
//! risk policy so it is fully unit-testable.
//!
//! ## Matching
//!
//! Voiceprints are speaker embeddings (ECAPA-TDNN style, ~192 or 256 dims).
//! Same-speaker verification uses cosine similarity, reusing
//! [`crate::face_embedding::cosine_similarity`]. The ECAPA same-speaker
//! decision threshold is typically low (~0.25 on the cosine scale, far below
//! ArcFace's 0.28 for faces because the embedding geometry differs), exposed
//! as [`DEFAULT_VOICE_MATCH_THRESHOLD`].
//!
//! ## Risk policy
//!
//! A matched voiceprint contributes a signed risk delta:
//!   - **Family** voices *suppress* threat (negative delta) — a recognised
//!     household member talking should calm the system, not alarm it.
//!   - **Watch** / **Offender** voices *raise* threat, scaled by the stored
//!     `threat_level` and the match `similarity`.

use serde::{Deserialize, Serialize};

use crate::face_embedding::cosine_similarity;

/// ECAPA-TDNN same-speaker cosine threshold. Below this the probe is treated
/// as a different speaker (no hit). Tunable per deployment; conservative
/// default chosen to favour recall (catch intruders) over precision.
pub const DEFAULT_VOICE_MATCH_THRESHOLD: f32 = 0.25;

/// Classification of a known voice in the watchlist.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VoiceStatus {
    /// Household member — recognising this voice should *suppress* threat.
    Family,
    /// Person of interest — recognising this voice *raises* threat mildly.
    Watch,
    /// Known offender — recognising this voice *raises* threat strongly.
    Offender,
}

impl std::fmt::Display for VoiceStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VoiceStatus::Family => write!(f, "family"),
            VoiceStatus::Watch => write!(f, "watch"),
            VoiceStatus::Offender => write!(f, "offender"),
        }
    }
}

/// A single enrolled voiceprint in the watchlist.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceprintEntry {
    /// Stable identifier (e.g. UUID or slug).
    pub id: String,
    /// Human-readable label (e.g. "mum", "intruder-2026-05").
    pub label: String,
    /// Family / Watch / Offender classification.
    pub status: VoiceStatus,
    /// Stored severity 0..=100; scales the risk delta for Watch/Offender.
    pub threat_level: u8,
    /// ECAPA-TDNN speaker embedding (~192 or 256 dims).
    pub embedding: Vec<f32>,
}

/// A successful match against the watchlist.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VoiceprintHit {
    pub id: String,
    pub label: String,
    pub status: VoiceStatus,
    pub threat_level: u8,
    /// Cosine similarity in [-1, 1] of the winning match.
    pub similarity: f32,
}

impl VoiceprintHit {
    /// Signed risk contribution of this match, in roughly [-1.0, 1.0].
    ///
    /// - **Family**: negative (suppresses). Magnitude grows with how
    ///   confident the match is, so a strong family match calms more.
    /// - **Watch / Offender**: positive (raises), scaled by the stored
    ///   `threat_level` (0..=100 → 0.0..=1.0) and the match `similarity`.
    ///
    /// `similarity` is clamped to [0, 1] for scaling (negative cosines never
    /// reach here in practice — they fail the threshold first).
    pub fn risk_delta(&self) -> f32 {
        let sim = self.similarity.clamp(0.0, 1.0);
        let level = (self.threat_level as f32 / 100.0).clamp(0.0, 1.0);
        match self.status {
            // Recognised family member: suppress. Stronger match → stronger
            // suppression (up to -0.5).
            VoiceStatus::Family => -0.5 * sim,
            // Watcher: mild raise.
            VoiceStatus::Watch => 0.4 * level * sim,
            // Known offender: strong raise.
            VoiceStatus::Offender => level * sim,
        }
    }
}

/// Match a probe embedding against the watchlist, returning the best hit at
/// or above `threshold`, or `None`.
///
/// Reuses [`cosine_similarity`]. Entries whose embedding dimensionality does
/// not match the probe are skipped (a 192-dim probe never matches a 256-dim
/// enrolment — comparing them is meaningless, so they are ignored rather than
/// silently scored 0).
pub fn match_voice(
    probe: &[f32],
    entries: &[VoiceprintEntry],
    threshold: f32,
) -> Option<VoiceprintHit> {
    if probe.is_empty() {
        return None;
    }

    let mut best: Option<VoiceprintHit> = None;
    for entry in entries {
        // Skip dimension-mismatched entries: cross-dim cosine is meaningless.
        if entry.embedding.len() != probe.len() {
            continue;
        }
        let sim = cosine_similarity(probe, &entry.embedding);
        if sim < threshold {
            continue;
        }
        let is_better = best.as_ref().map(|b| sim > b.similarity).unwrap_or(true);
        if is_better {
            best = Some(VoiceprintHit {
                id: entry.id.clone(),
                label: entry.label.clone(),
                status: entry.status,
                threat_level: entry.threat_level,
                similarity: sim,
            });
        }
    }
    best
}

// ── Service contract ────────────────────────────────────────────────────
//
// Network I/O is kept OUT of this module's unit tests. The trait describes
// the HTTP contract with the ECAPA-TDNN sidecar; the stub impl below is the
// only place that touches the wire, and it is async (never exercised by the
// pure tests).

/// Contract for the voiceprint extraction sidecar.
///
/// The sidecar accepts base64 audio (16 kHz mono PCM/WAV) and returns a
/// single speaker embedding. Configured via the `VOICEPRINT_URL` env var.
#[allow(async_fn_in_trait)]
pub trait VoiceprintBackend {
    /// Extract a speaker embedding from base64-encoded audio.
    async fn embed_voice(&self, audio_b64: &str) -> anyhow::Result<Vec<f32>>;
}

/// Default URL of the ECAPA-TDNN voiceprint sidecar.
pub const DEFAULT_VOICEPRINT_URL: &str = "http://127.0.0.1:5557";

/// Resolve the voiceprint sidecar URL from the `VOICEPRINT_URL` env var,
/// falling back to [`DEFAULT_VOICEPRINT_URL`].
pub fn voiceprint_url() -> String {
    std::env::var("VOICEPRINT_URL").unwrap_or_else(|_| DEFAULT_VOICEPRINT_URL.to_string())
}

/// HTTP implementation of [`VoiceprintBackend`]. Talks to the Python sidecar
/// at [`voiceprint_url`]. Constructed with a shared `reqwest::Client`.
#[derive(Clone)]
pub struct HttpVoiceprintBackend {
    client: reqwest::Client,
    base_url: String,
}

impl HttpVoiceprintBackend {
    pub fn new(client: reqwest::Client) -> Self {
        Self { client, base_url: voiceprint_url() }
    }

    pub fn with_url(client: reqwest::Client, base_url: impl Into<String>) -> Self {
        Self { client, base_url: base_url.into() }
    }
}

#[derive(Debug, Serialize)]
struct VoiceEmbedRequest<'a> {
    audio_b64: &'a str,
}

#[derive(Debug, Deserialize)]
struct VoiceEmbedResponse {
    ok: bool,
    embedding: Vec<f32>,
}

impl VoiceprintBackend for HttpVoiceprintBackend {
    async fn embed_voice(&self, audio_b64: &str) -> anyhow::Result<Vec<f32>> {
        let url = format!("{}/embed_voice", self.base_url);
        let resp = self
            .client
            .post(&url)
            .json(&VoiceEmbedRequest { audio_b64 })
            .send()
            .await?
            .error_for_status()?
            .json::<VoiceEmbedResponse>()
            .await?;
        if !resp.ok || resp.embedding.is_empty() {
            anyhow::bail!("voiceprint sidecar returned no embedding");
        }
        Ok(resp.embedding)
    }
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build an entry with a deterministic embedding for matching tests.
    fn entry(id: &str, status: VoiceStatus, threat: u8, emb: Vec<f32>) -> VoiceprintEntry {
        VoiceprintEntry {
            id: id.into(),
            label: id.into(),
            status,
            threat_level: threat,
            embedding: emb,
        }
    }

    #[test]
    fn same_speaker_matches() {
        // Probe nearly identical to an enrolled family voice → hit.
        let probe = vec![0.9, 0.1, 0.4, 0.2];
        let entries = vec![entry(
            "mum",
            VoiceStatus::Family,
            0,
            vec![0.91, 0.09, 0.41, 0.21],
        )];
        let hit = match_voice(&probe, &entries, DEFAULT_VOICE_MATCH_THRESHOLD)
            .expect("near-identical embedding should match");
        assert_eq!(hit.id, "mum");
        assert!(hit.similarity > 0.99, "similarity too low: {}", hit.similarity);
    }

    #[test]
    fn different_speaker_misses() {
        // Orthogonal embedding → cosine ~0 → below threshold → no hit.
        let probe = vec![1.0, 0.0, 0.0, 0.0];
        let entries = vec![entry(
            "stranger",
            VoiceStatus::Watch,
            50,
            vec![0.0, 1.0, 0.0, 0.0],
        )];
        assert!(match_voice(&probe, &entries, DEFAULT_VOICE_MATCH_THRESHOLD).is_none());
    }

    #[test]
    fn best_match_wins_among_several() {
        let probe = vec![1.0, 0.0, 0.0, 0.0];
        let entries = vec![
            entry("weak", VoiceStatus::Watch, 50, vec![0.6, 0.8, 0.0, 0.0]), // sim 0.6
            entry("strong", VoiceStatus::Offender, 90, vec![0.99, 0.14, 0.0, 0.0]), // sim ~0.99
        ];
        let hit = match_voice(&probe, &entries, DEFAULT_VOICE_MATCH_THRESHOLD).unwrap();
        assert_eq!(hit.id, "strong");
    }

    #[test]
    fn dimension_mismatch_skipped() {
        // 256-dim enrolment vs 4-dim probe must be ignored, not scored 0.
        let probe = vec![1.0, 0.0, 0.0, 0.0];
        let entries = vec![
            entry("wrong-dim", VoiceStatus::Offender, 100, vec![1.0; 256]),
            entry("right-dim", VoiceStatus::Family, 0, vec![1.0, 0.01, 0.0, 0.0]),
        ];
        let hit = match_voice(&probe, &entries, DEFAULT_VOICE_MATCH_THRESHOLD).unwrap();
        assert_eq!(hit.id, "right-dim", "mismatched-dim entry should be skipped");
    }

    #[test]
    fn empty_probe_no_hit() {
        let entries = vec![entry("x", VoiceStatus::Family, 0, vec![1.0, 0.0])];
        assert!(match_voice(&[], &entries, DEFAULT_VOICE_MATCH_THRESHOLD).is_none());
    }

    #[test]
    fn family_suppresses_risk() {
        let hit = VoiceprintHit {
            id: "dad".into(),
            label: "dad".into(),
            status: VoiceStatus::Family,
            threat_level: 0,
            similarity: 0.95,
        };
        assert!(hit.risk_delta() < 0.0, "family must suppress: {}", hit.risk_delta());
    }

    #[test]
    fn offender_raises_more_than_watch() {
        let watch = VoiceprintHit {
            id: "w".into(),
            label: "w".into(),
            status: VoiceStatus::Watch,
            threat_level: 80,
            similarity: 0.9,
        };
        let offender = VoiceprintHit {
            id: "o".into(),
            label: "o".into(),
            status: VoiceStatus::Offender,
            threat_level: 80,
            similarity: 0.9,
        };
        assert!(watch.risk_delta() > 0.0);
        assert!(offender.risk_delta() > watch.risk_delta());
    }

    #[test]
    fn risk_delta_scales_with_threat_level() {
        let low = VoiceprintHit {
            id: "a".into(),
            label: "a".into(),
            status: VoiceStatus::Offender,
            threat_level: 10,
            similarity: 1.0,
        };
        let high = VoiceprintHit {
            id: "b".into(),
            label: "b".into(),
            status: VoiceStatus::Offender,
            threat_level: 100,
            similarity: 1.0,
        };
        assert!(high.risk_delta() > low.risk_delta());
    }

    #[test]
    fn status_display() {
        assert_eq!(VoiceStatus::Family.to_string(), "family");
        assert_eq!(VoiceStatus::Watch.to_string(), "watch");
        assert_eq!(VoiceStatus::Offender.to_string(), "offender");
    }

    #[test]
    fn threshold_boundary_respected() {
        // Construct a pair with cosine just under a high custom threshold.
        let probe = vec![1.0, 0.0];
        let entries = vec![entry("t", VoiceStatus::Watch, 50, vec![0.8, 0.6])]; // cos = 0.8
        assert!(match_voice(&probe, &entries, 0.9).is_none(), "0.8 < 0.9");
        assert!(match_voice(&probe, &entries, 0.7).is_some(), "0.8 >= 0.7");
    }
}
