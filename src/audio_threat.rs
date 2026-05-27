//! audio_threat — pure audio-event → threat-score logic (Phase 5 (b)).
//!
//! Maps audio-tagging events (YAMNet-style labels with per-label scores) to a
//! single 0..1 threat score, and adapts that to
//! [`crate::VisionEvent::voice_agitated`]. Like [`crate::voiceprint`], this
//! module is PURE: no live models, no GPU. The audio tagger runs in a Python
//! sidecar behind [`AudioTaggerBackend`]; network I/O is kept out of the unit
//! tests.
//!
//! ## Scoring
//!
//! Each security-relevant label carries a weight. The threat score is the
//! **max-weighted** contribution across the present events
//! (`weight * score`), so one loud gunshot dominates rather than averaging
//! out against benign background tags. Benign tags (speech, music, silence)
//! carry zero weight and never raise the score. An empty event list scores
//! exactly 0.

use serde::{Deserialize, Serialize};

/// A single audio-tagging detection (YAMNet-style).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioEvent {
    /// Tag label, e.g. "scream", "glass", "gunshot", "speech", "music".
    pub label: String,
    /// Model confidence for this label, expected in [0, 1].
    pub score: f32,
}

/// Weight for a security-relevant audio label. Returns `0.0` for benign or
/// unknown labels. Matching is case-insensitive and substring-based so
/// raw YAMNet display names ("Glass", "Breaking", "Gunshot, gunfire",
/// "Screaming") map onto our security categories.
fn label_weight(label: &str) -> f32 {
    let l = label.to_ascii_lowercase();
    // Most severe first; substring match against YAMNet-style display names.
    if l.contains("gunshot") || l.contains("gunfire") || l.contains("machine gun") {
        1.0
    } else if l.contains("scream") || l.contains("screaming") {
        0.95
    } else if l.contains("glass") || l.contains("shatter") || l.contains("breaking") {
        0.85
    } else if l.contains("alarm") || l.contains("siren") || l.contains("smoke detector") {
        0.7
    } else if l.contains("shout") || l.contains("yell") {
        0.6
    } else {
        // Benign: speech, music, silence, footsteps, etc.
        0.0
    }
}

/// Compute a 0..1 threat score from audio-tagging events.
///
/// Uses the **max** of `weight * score.clamp(0,1)` across all events, so the
/// single most threatening sound dominates. Empty input → `0.0`.
pub fn threat_score(events: &[AudioEvent]) -> f32 {
    events
        .iter()
        .map(|e| label_weight(&e.label) * e.score.clamp(0.0, 1.0))
        .fold(0.0_f32, f32::max)
        .clamp(0.0, 1.0)
}

/// Adapt an audio threat score to [`crate::VisionEvent::voice_agitated`].
///
/// Returns `None` when the threat is negligible (≤ epsilon) so that callers
/// leave `voice_agitated` unset rather than feeding a spurious zero into the
/// perceptual-risk fusion. Otherwise returns the score unchanged (already
/// in 0..1).
pub fn to_voice_agitated(threat: f32) -> Option<f32> {
    if threat <= 1e-6 {
        None
    } else {
        Some(threat.clamp(0.0, 1.0))
    }
}

// ── Service contract ────────────────────────────────────────────────────

/// Contract for the YAMNet-style audio-tagging sidecar.
///
/// The sidecar accepts base64 audio (16 kHz mono) and returns scored tags.
/// Configured via the `YAMNET_URL` env var.
#[allow(async_fn_in_trait)]
pub trait AudioTaggerBackend {
    /// Tag base64-encoded audio, returning scored [`AudioEvent`]s.
    async fn tag_audio(&self, audio_b64: &str) -> anyhow::Result<Vec<AudioEvent>>;
}

/// Default URL of the YAMNet audio-tagging sidecar.
pub const DEFAULT_YAMNET_URL: &str = "http://127.0.0.1:5558";

/// Resolve the audio-tagger URL from the `YAMNET_URL` env var, falling back
/// to [`DEFAULT_YAMNET_URL`].
pub fn yamnet_url() -> String {
    std::env::var("YAMNET_URL").unwrap_or_else(|_| DEFAULT_YAMNET_URL.to_string())
}

/// HTTP implementation of [`AudioTaggerBackend`]. Talks to the Python sidecar
/// at [`yamnet_url`].
#[derive(Clone)]
pub struct HttpAudioTaggerBackend {
    client: reqwest::Client,
    base_url: String,
}

impl HttpAudioTaggerBackend {
    pub fn new(client: reqwest::Client) -> Self {
        Self { client, base_url: yamnet_url() }
    }

    pub fn with_url(client: reqwest::Client, base_url: impl Into<String>) -> Self {
        Self { client, base_url: base_url.into() }
    }
}

#[derive(Debug, Serialize)]
struct TagRequest<'a> {
    audio_b64: &'a str,
}

#[derive(Debug, Deserialize)]
struct TagResponse {
    ok: bool,
    events: Vec<AudioEvent>,
}

impl AudioTaggerBackend for HttpAudioTaggerBackend {
    async fn tag_audio(&self, audio_b64: &str) -> anyhow::Result<Vec<AudioEvent>> {
        let url = format!("{}/tag_audio", self.base_url);
        let resp = self
            .client
            .post(&url)
            .json(&TagRequest { audio_b64 })
            .send()
            .await?
            .error_for_status()?
            .json::<TagResponse>()
            .await?;
        if !resp.ok {
            anyhow::bail!("audio tagger sidecar reported failure");
        }
        Ok(resp.events)
    }
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn ev(label: &str, score: f32) -> AudioEvent {
        AudioEvent { label: label.into(), score }
    }

    #[test]
    fn empty_is_zero() {
        assert_eq!(threat_score(&[]), 0.0);
    }

    #[test]
    fn scream_is_high() {
        let s = threat_score(&[ev("Screaming", 0.9)]);
        assert!(s > 0.8, "scream should score high: {s}");
    }

    #[test]
    fn glass_break_is_high() {
        let s = threat_score(&[ev("Glass", 0.95), ev("Breaking", 0.8)]);
        assert!(s > 0.7, "glass break should score high: {s}");
    }

    #[test]
    fn gunshot_is_highest() {
        let s = threat_score(&[ev("Gunshot, gunfire", 1.0)]);
        assert!(s > 0.95, "gunshot should be near-max: {s}");
    }

    #[test]
    fn benign_is_low() {
        let speech = threat_score(&[ev("Speech", 0.99)]);
        let music = threat_score(&[ev("Music", 0.99)]);
        assert_eq!(speech, 0.0, "speech is benign");
        assert_eq!(music, 0.0, "music is benign");
    }

    #[test]
    fn benign_mixed_with_threat_takes_max() {
        // Loud benign speech must not dilute a real gunshot.
        let s = threat_score(&[ev("Speech", 1.0), ev("Music", 1.0), ev("Gunshot", 0.9)]);
        assert!(s > 0.85, "max-weighting should let gunshot dominate: {s}");
    }

    #[test]
    fn low_confidence_scales_down() {
        let loud = threat_score(&[ev("scream", 0.9)]);
        let faint = threat_score(&[ev("scream", 0.1)]);
        assert!(faint < loud, "confidence should scale threat");
        assert!(faint > 0.0);
    }

    #[test]
    fn score_clamped_to_unit() {
        // Out-of-range score is clamped before weighting.
        let s = threat_score(&[ev("gunshot", 5.0)]);
        assert!(s <= 1.0, "must clamp to 1.0: {s}");
    }

    #[test]
    fn to_voice_agitated_none_for_benign() {
        assert_eq!(to_voice_agitated(threat_score(&[ev("Speech", 1.0)])), None);
        assert_eq!(to_voice_agitated(0.0), None);
    }

    #[test]
    fn to_voice_agitated_some_for_threat() {
        let threat = threat_score(&[ev("scream", 0.9)]);
        let agitated = to_voice_agitated(threat).expect("scream should yield Some");
        assert!((agitated - threat).abs() < 1e-6);
    }

    #[test]
    fn alarm_and_shout_weighted_below_gunshot() {
        let alarm = threat_score(&[ev("Alarm", 1.0)]);
        let shout = threat_score(&[ev("Shouting", 1.0)]);
        let gun = threat_score(&[ev("Gunshot", 1.0)]);
        assert!(alarm > 0.0 && alarm < gun);
        assert!(shout > 0.0 && shout < alarm);
    }
}
