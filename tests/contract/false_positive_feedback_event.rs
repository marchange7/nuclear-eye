// CONTRACT TEST — AUD-112-P2-1
// Reference: os/75-three-product-feedback-loops.md §4 + §6 + §8
//            os/112-cross-product-reality-audit-2026-05-12.md row AUD-112-P2-1
//
// Invariant:
//   When an operator presses "this was nothing" on nuclear-watch within
//   the alert's `reversible_until` window, a SentinelleFalsePositiveFeedback
//   envelope MUST be emitted and MUST be chain-signed (os/74 §4 MAY class).
//   The envelope shape mirrors SentinelleAlert plus `correction: "false_positive"`
//   and `original_alert_id`. No biometric payload (frames, audio, face templates)
//   may appear in the envelope. os/75 §4 names this the highest-value training
//   signal: directly labels a Sentinelle error.
//
// When implemented, this test should:
//   1. Construct a SentinelleAlert fixture with recommended_action.reversible_until
//      set to Utc::now() + 30s.
//   2. Call the (not yet wired) `cancel_alert(alert_id, tenant_id)` path in
//      nuclear-eye/src/alarm_grader or nuclear-watch surface layer.
//   3. Assert a SentinelleFalsePositiveFeedback value is returned / emitted.
//   4. Assert feedback.chain.signed == true.
//   5. Assert feedback.original_alert_id == alert.alert_id.
//   6. Assert feedback.correction == "false_positive".
//   7. Assert no field named frame_data / audio_pcm / face_template appears in
//      the serialised JSON of the feedback envelope.
//   8. Repeat the cancel call with reversible_until already elapsed -> assert
//      the function returns Err (window closed).
//
// TODO(AUD-112-P2-1): implement SentinelleFalsePositiveFeedback in
//   nuclear-sdk/crates/nuclear-sdk/src/types/ (new file or extend telemetry.rs)
//   and wire the cancel path in nuclear-eye/src/alert/ before removing the
//   #[ignore] annotation below.

use nuclear_eye::alert::envelope::{AlertChainEnvelope, SentinelleAlert};

// ── Runnable today: structural invariants on the existing alert type ──────────

/// The chain envelope on SentinelleAlert carries a `signed` bool and optional
/// chain_hash / signer fields. A future feedback envelope will mirror this shape.
/// This test confirms the existing type compiles and the signed field is accessible,
/// so the feedback variant can reuse the same pattern without a breaking change.
#[test]
fn sentinelle_alert_chain_envelope_has_signed_field() {
    let chain = AlertChainEnvelope {
        signed: false,
        chain_hash: None,
        prev_alert_hash: None,
        signer: None,
    };
    assert!(!chain.signed, "unsigned envelope must have signed == false");
}

/// reversible_until is the trigger window for the false-positive feedback path.
/// Confirm the field is present on AlertRecommendedAction so the feedback event
/// emitter can read it without a schema change.
#[test]
fn recommended_action_has_reversible_until() {
    use nuclear_eye::alert::envelope::AlertRecommendedAction;
    use nuclear_eye::alert::envelope::RecommendedActionPrimary;
    let action = AlertRecommendedAction {
        primary: RecommendedActionPrimary::RequestOperatorAck,
        secondary: vec![],
        requires_biometric_auth: None,
        requires_operator_ack: Some(true),
        reversible_until: None,
    };
    // Field exists and is None when not set; the feedback emitter checks Some(ts).
    assert!(action.reversible_until.is_none());
}

// ── Pending: full feedback event emission and chain-signing enforcement ───────

#[test]
#[ignore = "TODO(AUD-112-P2-1): SentinelleFalsePositiveFeedback not yet implemented; \
            wire cancel_alert() path in nuclear-eye/src/alert/ and add the event type \
            to nuclear-sdk before enabling"]
fn cancel_within_window_emits_chain_signed_false_positive_feedback() {
    // See step-by-step in the module header comment above.
    todo!()
}

#[test]
#[ignore = "TODO(AUD-112-P2-1): window-expired rejection path not yet implemented"]
fn cancel_after_reversible_until_returns_error() {
    todo!()
}

#[test]
#[ignore = "TODO(AUD-112-P2-1): no raw biometric in feedback envelope"]
fn false_positive_feedback_envelope_contains_no_raw_biometric_fields() {
    // Serialize the feedback envelope and assert the JSON string contains none of:
    // "frame_data", "audio_pcm", "face_template", "voice_print", "lidar_points"
    todo!()
}
