#[derive(Debug, Clone)]
pub struct SceneEvidence {
    pub person_detected: bool,
    pub repeated_passes: u32,
    pub hands_visible: u8,
    pub object_held: Option<String>,
    pub confidence: f64,
}

#[derive(Debug, Clone)]
pub struct SceneVerdict {
    pub suspicious: bool,
    pub score: f64,
    pub reason: String,
}

pub fn verify_scene(e: &SceneEvidence) -> SceneVerdict {
    let mut score = 0.0;
    if e.person_detected {
        score += 0.25;
    }
    if e.repeated_passes >= 2 {
        score += 0.35;
    }
    if e.hands_visible > 0 {
        score += 0.10;
    }
    if e.object_held.is_some() {
        score += 0.20;
    }
    score += (1.0 - e.confidence.clamp(0.0, 1.0)) * 0.10;

    let suspicious = score >= 0.50;
    let reason = if suspicious {
        "scene flagged for additional alarm grading"
    } else {
        "scene below suspicion threshold"
    };

    SceneVerdict {
        suspicious,
        score: score.clamp(0.0, 1.0),
        reason: reason.to_string(),
    }
}
