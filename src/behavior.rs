//! SEN-13 Phase 3: human-behavior signal fusion + repetition-of-actions detection.
//!
//! Two concerns live here, both pure and unit-testable (no models, no GPU):
//!
//!  1. [`fuse_risk`] — collapse the multi-modal affect/threat signals attached to
//!     a `VisionEvent` (face emotion, gesture intent, voice agitation) plus the
//!     watchlist identity delta into a single clamped `[0,1]` risk score.
//!
//!  2. [`RepetitionTracker`] — detect that the *same* person or behavior keeps
//!     recurring inside a sliding time window. A lone sighting is noise; the same
//!     intruder seen three times in a few minutes is a pattern, and the threat
//!     score should escalate accordingly. [`apply_repetition`] folds that boost
//!     back into a risk score.

use std::collections::HashMap;

// ── Signal fusion ───────────────────────────────────────────────────────

/// Weight applied to the pre-scaled gesture-threat signal (`fast_approach` /
/// `hands_raised`). Gestures are the strongest *behavioral* predictor of an
/// imminent physical threat, so this carries the most weight of the affect signals.
const W_GESTURE: f64 = 0.35;

/// Weight applied to the watchlist identity delta. A known-bad identity match is
/// the strongest *identity* signal and weighs as heavily as gesture intent.
const W_WATCHLIST: f64 = 0.35;

/// Weight applied to the FER face-negative signal. A frightened/angry face is a
/// moderate corroborating signal — informative but easily produced by benign
/// expressions, so it weighs less than gesture/identity.
const W_FACE: f64 = 0.15;

/// Weight applied to the voice-agitation signal. Shouting/agitation corroborates
/// a threat but is environment-sensitive (TV, music, distance), so it is moderate.
const W_VOICE: f64 = 0.15;

/// Fraction of the *headroom* above the base risk that the combined affect/identity
/// signals can fill. The fused score never drops below `base_risk`: signals only
/// add evidence, they never exonerate. With all signals saturated and a non-zero
/// base, the score reaches 1.0.
///
/// Concretely: `fused = base + (1 - base) * weighted_signal_sum`, clamped to `[0,1]`.
/// Because `W_GESTURE + W_WATCHLIST + W_FACE + W_VOICE == 1.0`, a fully saturated
/// signal set drives `weighted_signal_sum` to 1.0 and the fused score to exactly 1.0.
///
/// Combine human-behavior + watchlist signals into a single risk score in `[0,1]`.
///
/// `base_risk` is the perception-pipeline's own risk estimate. The optional affect
/// signals (`face_negative`, `gesture_threat`, `voice_agitated`) are each clamped
/// to `[0,1]`; `None` contributes zero. `watchlist_delta` is the additive risk from
/// an identity match (e.g. `WatchlistHit::risk_delta()`), also clamped to `[0,1]`.
///
/// Weighting (documented constants above):
///   - gesture_threat and watchlist_delta weigh most (0.35 each),
///   - face_negative and voice_agitated are moderate (0.15 each).
///
/// The signals fill the headroom above `base_risk`, so the result is monotonic in
/// every signal, never below `base_risk`, and clamped to `[0,1]`.
pub fn fuse_risk(
    base_risk: f64,
    face_negative: Option<f32>,
    gesture_threat: Option<f32>,
    voice_agitated: Option<f32>,
    watchlist_delta: f64,
) -> f64 {
    let base = base_risk.clamp(0.0, 1.0);

    let face = face_negative.unwrap_or(0.0).clamp(0.0, 1.0) as f64;
    let gesture = gesture_threat.unwrap_or(0.0).clamp(0.0, 1.0) as f64;
    let voice = voice_agitated.unwrap_or(0.0).clamp(0.0, 1.0) as f64;
    let watchlist = watchlist_delta.clamp(0.0, 1.0);

    let weighted = W_GESTURE * gesture
        + W_WATCHLIST * watchlist
        + W_FACE * face
        + W_VOICE * voice;
    let weighted = weighted.clamp(0.0, 1.0);

    // Signals fill the headroom above base; they never lower the base estimate.
    (base + (1.0 - base) * weighted).clamp(0.0, 1.0)
}

// ── Repetition-of-actions detection ─────────────────────────────────────

/// Tracks recurring observations keyed by an arbitrary string (person name,
/// behavior label, camera+behavior pair, …) inside a sliding time window.
///
/// Old timestamps are pruned lazily on every `observe`/query so memory stays
/// bounded to roughly the events seen within `window_ms` per key.
#[derive(Debug, Clone)]
pub struct RepetitionTracker {
    /// Sliding window length in milliseconds.
    window_ms: u64,
    /// Minimum repeat count (inclusive) within the window before a boost applies.
    threshold: usize,
    /// Per-key sorted-ascending list of observation timestamps (ms).
    events: HashMap<String, Vec<u64>>,
}

/// Per-extra-repeat risk add-on once the threshold is reached. Each sighting at or
/// beyond `threshold` adds this much, so repetition escalates linearly.
const BOOST_STEP: f64 = 0.15;

/// Hard cap on the repetition boost so a flapping signal cannot dominate the score.
const BOOST_CAP: f64 = 0.6;

impl RepetitionTracker {
    /// Create a tracker with the given sliding `window_ms` and repeat `threshold`.
    /// `threshold` is floored at 1.
    pub fn new(window_ms: u64, threshold: usize) -> Self {
        Self {
            window_ms,
            threshold: threshold.max(1),
            events: HashMap::new(),
        }
    }

    fn window_start(&self, now_ms: u64) -> u64 {
        now_ms.saturating_sub(self.window_ms)
    }

    /// Drop timestamps for `key` that fall before the window starting at `now_ms`.
    /// Removes the key entirely once it has no live observations.
    fn prune_key(&mut self, key: &str, now_ms: u64) {
        let start = self.window_start(now_ms);
        if let Some(ts) = self.events.get_mut(key) {
            ts.retain(|&t| t >= start);
            if ts.is_empty() {
                self.events.remove(key);
            }
        }
    }

    /// Record an observation of `key` at `timestamp_ms`. Prunes that key's stale
    /// entries relative to the new timestamp so the window stays current.
    pub fn observe(&mut self, key: &str, timestamp_ms: u64) {
        let entry = self.events.entry(key.to_string()).or_default();
        entry.push(timestamp_ms);
        // Keep ascending order even if events arrive slightly out of order.
        entry.sort_unstable();
        self.prune_key(key, timestamp_ms);
    }

    /// Count observations of `key` within `[now_ms - window_ms, now_ms]`.
    /// Timestamps strictly in the future relative to `now_ms` are ignored.
    pub fn repeat_count(&self, key: &str, now_ms: u64) -> usize {
        let start = self.window_start(now_ms);
        self.events
            .get(key)
            .map(|ts| ts.iter().filter(|&&t| t >= start && t <= now_ms).count())
            .unwrap_or(0)
    }

    /// Escalating risk add-on for `key` once repeats reach the threshold.
    ///
    /// Below `threshold` the boost is `0.0`. At and beyond it, each sighting from
    /// the `threshold`-th onward adds [`BOOST_STEP`], capped at [`BOOST_CAP`].
    ///
    /// Example with `threshold == 3`, `BOOST_STEP == 0.15`:
    ///   count 1,2 → 0.0; count 3 → 0.15; count 4 → 0.30; …
    pub fn repetition_boost(&self, key: &str, now_ms: u64) -> f64 {
        let count = self.repeat_count(key, now_ms);
        if count < self.threshold {
            return 0.0;
        }
        let over = (count - self.threshold + 1) as f64;
        (over * BOOST_STEP).min(BOOST_CAP)
    }

    /// Prune stale entries across every key relative to `now_ms`.
    pub fn prune_all(&mut self, now_ms: u64) {
        let keys: Vec<String> = self.events.keys().cloned().collect();
        for key in keys {
            self.prune_key(&key, now_ms);
        }
    }
}

/// Add a repetition boost to a risk score, clamped to `[0,1]`.
pub fn apply_repetition(risk: f64, boost: f64) -> f64 {
    (risk.clamp(0.0, 1.0) + boost.max(0.0)).clamp(0.0, 1.0)
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // --- fuse_risk ---

    #[test]
    fn fuse_no_signals_equals_base() {
        let r = fuse_risk(0.4, None, None, None, 0.0);
        assert!((r - 0.4).abs() < 1e-9, "no signals should preserve base, got {r}");
    }

    #[test]
    fn fuse_zero_base_zero_signals_is_zero() {
        assert_eq!(fuse_risk(0.0, None, None, None, 0.0), 0.0);
    }

    #[test]
    fn fuse_high_gesture_raises_risk() {
        let base = fuse_risk(0.3, None, None, None, 0.0);
        let with_gesture = fuse_risk(0.3, None, Some(0.9), None, 0.0);
        assert!(with_gesture > base, "gesture should raise risk: {with_gesture} vs {base}");
    }

    #[test]
    fn fuse_high_watchlist_raises_risk() {
        let base = fuse_risk(0.3, None, None, None, 0.0);
        let with_wl = fuse_risk(0.3, None, None, None, 0.9);
        assert!(with_wl > base, "watchlist should raise risk: {with_wl} vs {base}");
    }

    #[test]
    fn fuse_gesture_and_watchlist_weigh_more_than_face() {
        // Same magnitude signal, applied to gesture vs face — gesture must win.
        let gesture = fuse_risk(0.2, None, Some(0.8), None, 0.0);
        let face = fuse_risk(0.2, Some(0.8), None, None, 0.0);
        assert!(gesture > face, "gesture weighs more than face: {gesture} vs {face}");

        let watchlist = fuse_risk(0.2, None, None, None, 0.8);
        let voice = fuse_risk(0.2, None, None, Some(0.8), 0.0);
        assert!(watchlist > voice, "watchlist weighs more than voice: {watchlist} vs {voice}");
    }

    #[test]
    fn fuse_clamps_at_one() {
        let r = fuse_risk(0.9, Some(1.0), Some(1.0), Some(1.0), 1.0);
        assert!(r <= 1.0, "must not exceed 1.0: {r}");
        assert!((r - 1.0).abs() < 1e-9, "saturated signals from non-zero base → 1.0, got {r}");
    }

    #[test]
    fn fuse_never_below_base() {
        // Even with all-zero signals and a high base, never drop below base.
        let r = fuse_risk(0.7, Some(0.0), Some(0.0), Some(0.0), 0.0);
        assert!(r >= 0.7 - 1e-9, "must not fall below base: {r}");
    }

    #[test]
    fn fuse_clamps_out_of_range_inputs() {
        // Out-of-range base and signals are clamped, not panicking.
        let r = fuse_risk(2.0, Some(5.0), Some(-1.0), Some(2.0), 9.0);
        assert!((0.0..=1.0).contains(&r), "result in range: {r}");
    }

    // --- RepetitionTracker ---

    #[test]
    fn single_event_no_boost() {
        let mut t = RepetitionTracker::new(60_000, 3);
        t.observe("intruder", 1_000);
        assert_eq!(t.repeat_count("intruder", 1_000), 1);
        assert_eq!(t.repetition_boost("intruder", 1_000), 0.0);
    }

    #[test]
    fn below_threshold_no_boost() {
        let mut t = RepetitionTracker::new(60_000, 3);
        t.observe("bob", 1_000);
        t.observe("bob", 2_000);
        assert_eq!(t.repeat_count("bob", 2_000), 2);
        assert_eq!(t.repetition_boost("bob", 2_000), 0.0);
    }

    #[test]
    fn at_threshold_within_window_boosts() {
        let mut t = RepetitionTracker::new(60_000, 3);
        t.observe("intruder", 1_000);
        t.observe("intruder", 2_000);
        t.observe("intruder", 3_000);
        assert_eq!(t.repeat_count("intruder", 3_000), 3);
        let boost = t.repetition_boost("intruder", 3_000);
        assert!(boost > 0.0, "threshold reached → boost: {boost}");
        assert!((boost - BOOST_STEP).abs() < 1e-9, "first boost step: {boost}");
    }

    #[test]
    fn boost_escalates_with_more_repeats() {
        let mut t = RepetitionTracker::new(60_000, 3);
        for ts in [1_000u64, 2_000, 3_000, 4_000, 5_000] {
            t.observe("intruder", ts);
        }
        // count 5, threshold 3 → over = 3 → 3 * 0.15 = 0.45
        let boost = t.repetition_boost("intruder", 5_000);
        assert!(boost > BOOST_STEP, "more repeats escalate: {boost}");
        assert!((boost - 0.45).abs() < 1e-9, "expected 0.45, got {boost}");
    }

    #[test]
    fn boost_capped() {
        let mut t = RepetitionTracker::new(1_000_000, 2);
        for i in 0..50u64 {
            t.observe("spammer", i * 100);
        }
        let boost = t.repetition_boost("spammer", 50 * 100);
        assert!(boost <= BOOST_CAP + 1e-9, "boost must be capped: {boost}");
        assert!((boost - BOOST_CAP).abs() < 1e-9, "should hit cap: {boost}");
    }

    #[test]
    fn events_outside_window_pruned_and_not_counted() {
        let mut t = RepetitionTracker::new(10_000, 3);
        // Three sightings, but spread so they never coexist in a 10s window.
        t.observe("ghost", 1_000);
        t.observe("ghost", 20_000);
        t.observe("ghost", 40_000);
        // At t=40_000 only the last sighting is within [30_000, 40_000].
        assert_eq!(t.repeat_count("ghost", 40_000), 1);
        assert_eq!(t.repetition_boost("ghost", 40_000), 0.0);
    }

    #[test]
    fn observe_prunes_stale_entries() {
        let mut t = RepetitionTracker::new(5_000, 3);
        t.observe("x", 1_000);
        t.observe("x", 2_000);
        // This observe is far in the future; prune drops the two stale ones.
        t.observe("x", 100_000);
        assert_eq!(t.repeat_count("x", 100_000), 1);
    }

    #[test]
    fn three_in_window_then_window_slides_off() {
        let mut t = RepetitionTracker::new(10_000, 3);
        t.observe("intruder", 1_000);
        t.observe("intruder", 2_000);
        t.observe("intruder", 3_000);
        assert!(t.repetition_boost("intruder", 3_000) > 0.0);
        // Much later, the window has slid past all three sightings.
        assert_eq!(t.repeat_count("intruder", 30_000), 0);
        assert_eq!(t.repetition_boost("intruder", 30_000), 0.0);
    }

    #[test]
    fn different_keys_independent() {
        let mut t = RepetitionTracker::new(60_000, 3);
        for ts in [1_000u64, 2_000, 3_000] {
            t.observe("alice", ts);
        }
        t.observe("bob", 3_000);
        assert!(t.repetition_boost("alice", 3_000) > 0.0, "alice repeats");
        assert_eq!(t.repetition_boost("bob", 3_000), 0.0, "bob does not");
        assert_eq!(t.repeat_count("alice", 3_000), 3);
        assert_eq!(t.repeat_count("bob", 3_000), 1);
    }

    #[test]
    fn unseen_key_is_zero() {
        let t = RepetitionTracker::new(60_000, 3);
        assert_eq!(t.repeat_count("nobody", 1_000), 0);
        assert_eq!(t.repetition_boost("nobody", 1_000), 0.0);
    }

    #[test]
    fn out_of_order_observations_handled() {
        let mut t = RepetitionTracker::new(60_000, 3);
        t.observe("z", 3_000);
        t.observe("z", 1_000);
        t.observe("z", 2_000);
        assert_eq!(t.repeat_count("z", 3_000), 3);
        assert!(t.repetition_boost("z", 3_000) > 0.0);
    }

    #[test]
    fn prune_all_removes_stale_keys() {
        let mut t = RepetitionTracker::new(5_000, 1);
        t.observe("a", 1_000);
        t.observe("b", 2_000);
        t.prune_all(100_000);
        assert_eq!(t.repeat_count("a", 100_000), 0);
        assert_eq!(t.repeat_count("b", 100_000), 0);
    }

    #[test]
    fn future_timestamps_excluded_from_count() {
        let mut t = RepetitionTracker::new(60_000, 1);
        t.observe("f", 10_000);
        // Querying at an earlier "now" must not count the future sighting.
        assert_eq!(t.repeat_count("f", 5_000), 0);
    }

    #[test]
    fn threshold_floored_at_one() {
        let mut t = RepetitionTracker::new(60_000, 0);
        t.observe("k", 1_000);
        // threshold 0 is floored to 1, so a single sighting already boosts.
        assert!(t.repetition_boost("k", 1_000) > 0.0);
    }

    // --- apply_repetition ---

    #[test]
    fn apply_repetition_adds_and_clamps() {
        assert!((apply_repetition(0.5, 0.2) - 0.7).abs() < 1e-9);
        assert_eq!(apply_repetition(0.9, 0.5), 1.0, "clamps at 1.0");
        assert_eq!(apply_repetition(0.5, 0.0), 0.5, "zero boost is identity");
    }

    #[test]
    fn apply_repetition_negative_boost_ignored() {
        assert_eq!(apply_repetition(0.5, -0.3), 0.5, "negative boost cannot lower risk");
    }

    #[test]
    fn fuse_then_repetition_pipeline() {
        // End-to-end: fuse signals, then escalate on repetition.
        let mut t = RepetitionTracker::new(60_000, 3);
        for ts in [1_000u64, 2_000, 3_000] {
            t.observe("intruder", ts);
        }
        let fused = fuse_risk(0.3, Some(0.6), Some(0.7), Some(0.5), 0.5);
        let boost = t.repetition_boost("intruder", 3_000);
        let final_risk = apply_repetition(fused, boost);
        assert!(final_risk > fused, "repetition escalates fused risk");
        assert!((0.0..=1.0).contains(&final_risk));
    }
}
