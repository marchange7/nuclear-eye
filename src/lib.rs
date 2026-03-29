pub mod consul;
pub mod guard;
pub mod house;
pub mod memory;
pub mod runtime;
pub mod types;

pub use consul::{ConsulClient, ConsulDecision};
pub use guard::HouseGuard;
pub use house::House;
pub use runtime::HouseRuntime;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::fs;
use std::path::{Path, PathBuf};

// ── iPhone sensor types ───────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IPhoneSensorData {
    pub device_id: String,
    pub timestamp_ms: u64,
    pub pedestrians: Vec<PedestrianSummary>,
    pub lidar_available: bool,
    /// "normal" | "limited" | "not_available"
    pub tracking_quality: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PedestrianSummary {
    pub track_id: String,
    pub distance_m: Option<f64>,
    pub speed_mps: Option<f64>,
    pub collision_eta_s: Option<f64>,
    pub is_using_phone: bool,
    pub confidence: f64,
    pub identity: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionEvent {
    pub event_id: String,
    pub timestamp_ms: u64,
    pub camera_id: String,
    pub behavior: String,
    pub risk_score: f64,
    pub stress_level: f64,
    pub confidence: f64,
    pub person_detected: bool,
    pub person_name: Option<String>,
    pub hands_visible: u8,
    pub object_held: Option<String>,
    pub extra_tags: Vec<String>,
    #[serde(default)]
    pub vlm_caption: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum AlarmLevel {
    None,
    Low,
    Medium,
    High,
}

impl std::fmt::Display for AlarmLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AlarmLevel::None => write!(f, "none"),
            AlarmLevel::Low => write!(f, "low"),
            AlarmLevel::Medium => write!(f, "medium"),
            AlarmLevel::High => write!(f, "high"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlarmEvent {
    pub alarm_id: String,
    pub timestamp_ms: u64,
    pub level: AlarmLevel,
    pub danger_score: f64,
    pub risk_score: f64,
    pub stress_level: f64,
    pub person_detected: bool,
    pub person_name: Option<String>,
    pub note: String,
    #[serde(default)]
    pub vlm_caption: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AlarmSummary {
    pub current_level: AlarmLevel,
    pub danger_score: f64,
    pub last_n_alarms: Vec<AlarmLevel>,
}

#[derive(Debug, Clone)]
pub struct AlarmGrader {
    pub history_len: usize,
    pub recent_events: VecDeque<AlarmEvent>,
    pub hysteresis_window: usize,
    pub danger_thresholds: [f64; 3],
}

impl AlarmGrader {
    pub fn new() -> Self {
        Self {
            history_len: 20,
            recent_events: VecDeque::new(),
            hysteresis_window: 5,
            danger_thresholds: [0.3, 0.5, 0.8],
        }
    }

    pub fn compute_danger_score(&self, risk: f64, stress: f64, confidence: f64) -> f64 {
        let normalized_confidence = confidence.clamp(0.0, 1.0);
        let normalized_risk = risk.clamp(0.0, 1.0);
        let normalized_stress = stress.clamp(0.0, 1.0);
        ((normalized_risk + normalized_stress + (1.0 - normalized_confidence)) / 1.8).clamp(0.0, 1.0)
    }

    pub fn map_danger_to_level(&self, danger: f64) -> AlarmLevel {
        let [t0, t1, t2] = self.danger_thresholds;
        if danger < t0 {
            AlarmLevel::None
        } else if danger < t1 {
            AlarmLevel::Low
        } else if danger < t2 {
            AlarmLevel::Medium
        } else {
            AlarmLevel::High
        }
    }

    fn severity(level: &AlarmLevel) -> u8 {
        match level {
            AlarmLevel::None => 0,
            AlarmLevel::Low => 1,
            AlarmLevel::Medium => 2,
            AlarmLevel::High => 3,
        }
    }

    fn high_streak(&self) -> usize {
        self.recent_events
            .iter()
            .rev()
            .take(self.hysteresis_window)
            .filter(|e| matches!(e.level, AlarmLevel::High))
            .count()
    }

    fn medium_or_higher_streak(&self) -> usize {
        self.recent_events
            .iter()
            .rev()
            .take(self.hysteresis_window)
            .filter(|e| matches!(e.level, AlarmLevel::Medium | AlarmLevel::High))
            .count()
    }

    pub fn apply_hysteresis(&self, proposed: AlarmLevel) -> AlarmLevel {
        let Some(last) = self.recent_events.back().map(|e| e.level.clone()) else {
            return proposed;
        };

        let cur = Self::severity(&last);
        let next = Self::severity(&proposed);

        if next <= cur {
            if cur == 3 && self.high_streak() >= 2 {
                return AlarmLevel::High;
            }
            return proposed;
        }

        match (&last, &proposed) {
            (AlarmLevel::Low, AlarmLevel::Medium) if self.medium_or_higher_streak() < 2 => AlarmLevel::Low,
            (AlarmLevel::Medium, AlarmLevel::High) if self.high_streak() < 2 => AlarmLevel::Medium,
            _ => proposed,
        }
    }

    pub fn grade_event(&mut self, event: &VisionEvent) -> AlarmEvent {
        let raw = self.compute_danger_score(event.risk_score, event.stress_level, event.confidence);
        let proposed = self.map_danger_to_level(raw);
        let level = self.apply_hysteresis(proposed);
        let alarm = AlarmEvent {
            alarm_id: format!("alarm-{}", event.event_id),
            timestamp_ms: event.timestamp_ms,
            level: level.clone(),
            danger_score: raw,
            risk_score: event.risk_score,
            stress_level: event.stress_level,
            person_detected: event.person_detected,
            person_name: event.person_name.clone(),
            note: build_alarm_note(event, &level),
            vlm_caption: event.vlm_caption.clone(),
        };

        self.recent_events.push_back(alarm.clone());
        if self.recent_events.len() > self.history_len {
            self.recent_events.pop_front();
        }
        alarm
    }

    pub fn summary(&self) -> AlarmSummary {
        let current_level = self.recent_events.back().map(|e| e.level.clone()).unwrap_or(AlarmLevel::None);
        let danger_score = self.recent_events.back().map(|e| e.danger_score).unwrap_or(0.0);
        let last_n_alarms = self.recent_events.iter().rev().take(10).map(|e| e.level.clone()).collect();
        AlarmSummary { current_level, danger_score, last_n_alarms }
    }
}

pub fn build_alarm_note(event: &VisionEvent, level: &AlarmLevel) -> String {
    let object = event.object_held.clone().unwrap_or_else(|| "none".to_string());
    let person = event.person_name.clone().unwrap_or_else(|| "unknown".to_string());
    format!(
        "camera={} level={} behavior={} person={} object={} risk={:.2} stress={:.2} conf={:.2}",
        event.camera_id, level, event.behavior, person, object, event.risk_score, event.stress_level, event.confidence
    )
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct DecisionConfig {
    pub bind: String,
    /// VisionEvent.risk_score above this triggers safety-critical mode.
    pub safety_risk_threshold: f64,
}

impl Default for DecisionConfig {
    fn default() -> Self {
        Self {
            bind: "0.0.0.0:8085".into(),
            safety_risk_threshold: 0.5,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct AurelieBridgeConfig {
    pub bind: String,
    pub aurelie_chat_url: String,
    pub request_timeout_secs: u64,
    /// Also send Telegram notification when forwarding alarms.
    pub telegram_on_alarm: bool,
}

impl Default for AurelieBridgeConfig {
    fn default() -> Self {
        Self {
            bind: "0.0.0.0:8086".into(),
            aurelie_chat_url: "http://127.0.0.1:8090/chat".into(),
            request_timeout_secs: 30,
            telegram_on_alarm: true,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct FortressConfig {
    pub url: String,
    pub mesh_enabled: bool,
    pub publish_timeout_ms: u64,
}

impl Default for FortressConfig {
    fn default() -> Self {
        Self {
            url: "http://localhost:7700".into(),
            mesh_enabled: true,
            publish_timeout_ms: 500,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct SecurityConfig {
    pub app: AppConfig,
    pub alarm: AlarmConfig,
    pub telegram: TelegramConfig,
    pub face_db: FaceDbConfig,
    pub vision: VisionConfig,
    #[serde(default)]
    pub decision: DecisionConfig,
    #[serde(default)]
    pub aurelie_bridge: AurelieBridgeConfig,
    #[serde(default)]
    pub fortress: FortressConfig,
    #[serde(skip)]
    pub fastvlm_url: Option<String>,
}

impl SecurityConfig {
    /// Effective Fortress URL: `FORTRESS_URL` env var overrides config.
    pub fn fortress_url(&self) -> String {
        std::env::var("FORTRESS_URL").unwrap_or_else(|_| self.fortress.url.clone())
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct AppConfig {
    pub bind_alarm_grader: String,
    pub bind_safetyagent: String,
    pub bind_face_db: String,
    pub alarm_grader_url: String,
    pub face_db_path: String,
    pub scene_mode: String,
    pub model_path: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AlarmConfig {
    pub history_len: usize,
    pub hysteresis_window: usize,
    pub thresholds: [f64; 3],
    pub telegram_min_level: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TelegramConfig {
    pub enabled: bool,
    pub bot_token_env: String,
    pub chat_id_env: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct FaceDbConfig {
    pub auto_create: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct VisionConfig {
    pub tick_ms: u64,
    pub default_camera_id: String,
    pub target_url: String,
    /// HTTP URL of a JPEG snapshot endpoint (e.g. nuclear-eye/main.py `/snapshot/{cam_id}`).
    /// When set, vision_agent fetches real frames instead of generating synthetic events.
    /// Overridden by `CAMERA_SNAPSHOT_URL` env var.
    #[serde(default)]
    pub snapshot_url: Option<String>,
    /// FastVLM describe endpoint. Overridden by `FASTVLM_URL` env var.
    #[serde(default)]
    pub fastvlm_url: Option<String>,
}

impl SecurityConfig {
    /// Load config with automatic backup/restore.
    ///
    /// Strategy:
    ///   1. Try primary path (`HOUSE_SECURITY_CONFIG` or `config/security.toml`).
    ///   2. On failure, try `<path>.bak`.
    ///   3. On both failing, return an error (never silently run with unknown defaults).
    ///   4. After a successful primary-path load, write `<path>.bak` so the backup
    ///      is always the last known-good primary.
    pub fn load() -> Result<Self> {
        let path = std::env::var("HOUSE_SECURITY_CONFIG")
            .unwrap_or_else(|_| "config/security.toml".to_string());
        let bak_path = format!("{path}.bak");

        let (raw, used_backup) = match fs::read_to_string(&path) {
            Ok(s) => (s, false),
            Err(primary_err) => {
                tracing::warn!(
                    "config primary load failed ({primary_err}) — trying backup {bak_path}"
                );
                match fs::read_to_string(&bak_path) {
                    Ok(s) => {
                        tracing::warn!("config: loaded from backup {bak_path}");
                        (s, true)
                    }
                    Err(bak_err) => {
                        anyhow::bail!(
                            "config load failed — primary: {primary_err} | backup: {bak_err}"
                        );
                    }
                }
            }
        };

        let mut cfg: Self = toml::from_str(&raw)
            .with_context(|| format!("failed to parse config (backup={used_backup})"))?;
        cfg.fastvlm_url = std::env::var("FASTVLM_URL").ok();

        // Refresh backup only after a clean primary-path load.
        if !used_backup {
            if let Err(e) = fs::write(&bak_path, &raw) {
                tracing::warn!("config: failed to write backup {bak_path}: {e}");
            } else {
                tracing::debug!("config: backup updated → {bak_path}");
            }
        }

        Ok(cfg)
    }
}

pub fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

pub fn ensure_parent_dir(path: impl AsRef<Path>) -> Result<()> {
    let p: PathBuf = path.as_ref().to_path_buf();
    if let Some(parent) = p.parent() {
        fs::create_dir_all(parent).with_context(|| format!("failed to create {}", parent.display()))?;
    }
    Ok(())
}

pub fn level_from_string(input: &str) -> AlarmLevel {
    match input.trim().to_ascii_lowercase().as_str() {
        "high" => AlarmLevel::High,
        "medium" => AlarmLevel::Medium,
        "low" => AlarmLevel::Low,
        _ => AlarmLevel::None,
    }
}

/// Numeric severity for AlarmLevel (useful for comparisons outside AlarmGrader).
pub fn alarm_severity(level: &AlarmLevel) -> u8 {
    match level {
        AlarmLevel::None => 0,
        AlarmLevel::Low => 1,
        AlarmLevel::Medium => 2,
        AlarmLevel::High => 3,
    }
}

// ── Telegram helper ────────────────────────────────────────────────────

/// Reusable Telegram notification client.
/// Constructed once from config; shared across handlers via Arc.
#[derive(Clone)]
pub struct TelegramNotifier {
    client: reqwest::Client,
    bot_token: String,
    chat_id: String,
}

impl TelegramNotifier {
    /// Build from config.  Returns `Ok(None)` when telegram is disabled.
    pub fn from_config(
        cfg: &TelegramConfig,
        client: &reqwest::Client,
    ) -> Result<Option<Self>> {
        if !cfg.enabled {
            return Ok(None);
        }
        let bot_token = std::env::var(&cfg.bot_token_env)
            .with_context(|| format!("env {} not set", cfg.bot_token_env))?;
        let chat_id = std::env::var(&cfg.chat_id_env)
            .with_context(|| format!("env {} not set", cfg.chat_id_env))?;
        Ok(Some(Self {
            client: client.clone(),
            bot_token,
            chat_id,
        }))
    }

    /// Send a text message.  Returns whether the Telegram API accepted it.
    pub async fn send(&self, message: &str) -> Result<bool> {
        let url = format!(
            "https://api.telegram.org/bot{}/sendMessage",
            self.bot_token
        );
        let payload = serde_json::json!({
            "chat_id": self.chat_id,
            "text": message,
            "disable_web_page_preview": true,
        });
        let resp = self
            .client
            .post(&url)
            .json(&payload)
            .send()
            .await
            .context("telegram HTTP request failed")?;
        let status = resp.status();
        if !status.is_success() {
            tracing::warn!(%status, "telegram API rejected message");
        }
        Ok(status.is_success())
    }
}

// ── AffectTriad: jugement / doute / détermination ──────────────────────
//
// Three affective dimensions that govern AI decision-making, derived from
// the brainstorm's "triptyque affectif" design.
//
// Judgement  (jugement)      — clarity of situational understanding
// Doubt     (doute)          — degree of uncertainty / re-evaluation
// Determination (détermination) — readiness to decide and act
//
// The three values are always clamped to [0.0, 1.0].

#[derive(Clone, Debug, Serialize, Deserialize, Default, PartialEq)]
pub struct AffectTriad {
    pub judgement: f64,
    pub doubt: f64,
    pub determination: f64,
}

impl AffectTriad {
    pub fn new(judgement: f64, doubt: f64, determination: f64) -> Self {
        Self {
            judgement: judgement.clamp(0.0, 1.0),
            doubt: doubt.clamp(0.0, 1.0),
            determination: determination.clamp(0.0, 1.0),
        }
    }

    /// Derive triad from raw stress and confidence signals.
    ///
    /// - High stress  → impaired judgement
    /// - Low confidence → elevated doubt
    /// - High confidence → strong determination
    pub fn from_stress_confidence(stress: f64, confidence: f64) -> Self {
        let s = stress.clamp(0.0, 1.0);
        let c = confidence.clamp(0.0, 1.0);
        Self::new(1.0 - s, 1.0 - c, c)
    }

    /// Derive triad from a full VisionEvent, incorporating risk as a third axis.
    ///
    /// Weights:
    ///   J = 0.7 × (1 − stress) + 0.3 × (1 − risk)
    ///   D = 0.6 × (1 − confidence) + 0.4 × risk
    ///   Dt = 0.5 × confidence + 0.3 × risk + 0.2 × stress
    ///
    /// Rationale: risk amplifies doubt and urgency (determination) while
    /// dampening calm judgement.
    pub fn from_vision_event(event: &VisionEvent) -> Self {
        let c = event.confidence.clamp(0.0, 1.0);
        let s = event.stress_level.clamp(0.0, 1.0);
        let r = event.risk_score.clamp(0.0, 1.0);

        Self::new(
            (1.0 - s) * 0.7 + (1.0 - r) * 0.3,
            (1.0 - c) * 0.6 + r * 0.4,
            c * 0.5 + r * 0.3 + s * 0.2,
        )
    }

    /// Derive triad from an AlarmEvent (post-grading).
    ///
    /// Uses danger_score as inverse confidence and alarm severity to boost
    /// determination.
    pub fn from_alarm_event(event: &AlarmEvent) -> Self {
        let s = event.stress_level.clamp(0.0, 1.0);
        let d = event.danger_score.clamp(0.0, 1.0);
        let severity = alarm_severity(&event.level) as f64 / 3.0; // 0.0–1.0

        Self::new(
            (1.0 - s) * 0.6 + (1.0 - d) * 0.4,
            d * 0.5 + (1.0 - severity) * 0.2 + s * 0.3,
            severity * 0.5 + d * 0.3 + s * 0.2,
        )
    }

    /// Returns which dimension is currently dominant.
    pub fn dominant(&self) -> &'static str {
        if self.judgement >= self.doubt && self.judgement >= self.determination {
            "judgement"
        } else if self.doubt >= self.determination {
            "doubt"
        } else {
            "determination"
        }
    }
}

impl std::fmt::Display for AffectTriad {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "J={:.2} D={:.2} Dt={:.2}",
            self.judgement, self.doubt, self.determination
        )
    }
}

// ── DecisionAction ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DecisionAction {
    /// No specific action required.
    None,
    /// Soothe / calm down — the system is confused and uncertain.
    Reassure,
    /// Probe deeper — the system understands but has significant doubt.
    Challenge,
    /// Affirm / encourage — clear understanding, low doubt.
    Support,
    /// Trigger alarm — safety-critical with high determination.
    Alarm,
    /// Suggest a pause — safety-critical but too uncertain to act.
    Pause,
}

impl DecisionAction {
    pub fn is_actionable(self) -> bool {
        !matches!(self, Self::None)
    }
}

impl std::fmt::Display for DecisionAction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::None => write!(f, "none"),
            Self::Reassure => write!(f, "reassure"),
            Self::Challenge => write!(f, "challenge"),
            Self::Support => write!(f, "support"),
            Self::Alarm => write!(f, "alarm"),
            Self::Pause => write!(f, "pause"),
        }
    }
}

/// Map an AffectTriad to a DecisionAction.
///
/// **Safety-critical mode** (is_safety_critical = true):
///
/// | Condition              | Action |
/// |------------------------|--------|
/// | Dt > 0.7 ∧ D < 0.3    | Alarm  |
/// | Dt < 0.4 ∧ D > 0.6    | Pause  |
/// | otherwise              | None   |
///
/// **Relational mode** (is_safety_critical = false):
///
/// | Condition              | Action    |
/// |------------------------|-----------|
/// | J > 0.7 ∧ D > 0.5     | Challenge |
/// | J > 0.6 ∧ D < 0.4     | Support   |
/// | D > 0.7 ∧ J < 0.4     | Reassure  |
/// | Dt > 0.7 ∧ J > 0.5    | Support   |
/// | otherwise              | None      |
pub fn decide(triad: &AffectTriad, is_safety_critical: bool) -> DecisionAction {
    let j = triad.judgement;
    let d = triad.doubt;
    let dt = triad.determination;

    if is_safety_critical {
        if dt > 0.7 && d < 0.3 {
            return DecisionAction::Alarm;
        }
        if dt < 0.4 && d > 0.6 {
            return DecisionAction::Pause;
        }
        return DecisionAction::None;
    }

    // Relational / chat mode
    if j > 0.7 && d > 0.5 {
        DecisionAction::Challenge
    } else if j > 0.6 && d < 0.4 {
        DecisionAction::Support
    } else if d > 0.7 && j < 0.4 {
        DecisionAction::Reassure
    } else if dt > 0.7 && j > 0.5 {
        DecisionAction::Support
    } else {
        DecisionAction::None
    }
}

// ── Caption → VisionEvent ──────────────────────────────────────────────

/// Build a `VisionEvent` from a VLM caption string.
///
/// Extracts risk, stress, behavior, and object signals from free-form text so
/// that both `vision_agent` (RTSP path) and `alarm_grader_agent` (camera-frame
/// path) can share a single, well-tested signal-extraction routine.
pub fn caption_to_vision_event(camera_id: &str, caption: &str, timestamp_ms: u64) -> VisionEvent {
    let lc = caption.to_lowercase();

    let person_detected = lc.contains("person") || lc.contains("people")
        || lc.contains(" man ") || lc.contains("woman") || lc.contains("child")
        || lc.contains("individual");

    // Behavior keyword matching — most specific wins
    let behavior = if lc.contains("weapon") || lc.contains("gun") || lc.contains("knife") || lc.contains("firearm") {
        "weapon_detected"
    } else if lc.contains("fight") || lc.contains("struggle") || lc.contains("altercation") || lc.contains("attack") {
        "fighting"
    } else if lc.contains("running") || lc.contains("fleeing") || lc.contains("sprinting") || lc.contains("rushing") {
        "running"
    } else if lc.contains("loitering") || lc.contains("suspicious") || lc.contains("lurking") {
        "loitering"
    } else if lc.contains("approaching") || lc.contains("walking toward") {
        "approaching"
    } else if lc.contains("vehicle") || lc.contains("car") || lc.contains("truck") {
        "vehicle_present"
    } else if person_detected {
        "passby"
    } else {
        "no_activity"
    };

    // Risk score: accumulate threat signals
    let risk_score: f64 = {
        let mut r = 0.2f64;
        if person_detected { r += 0.25; }
        if lc.contains("running") || lc.contains("rushing") || lc.contains("sprinting") { r += 0.25; }
        if lc.contains("weapon") || lc.contains("gun") || lc.contains("knife") || lc.contains("firearm") { r += 0.55; }
        if lc.contains("fight") || lc.contains("struggle") || lc.contains("attack") { r += 0.40; }
        if lc.contains("loitering") || lc.contains("suspicious") || lc.contains("lurking") { r += 0.30; }
        if lc.contains("breaking") || lc.contains("forced") || lc.contains("intruder") { r += 0.45; }
        if lc.contains("calm") || lc.contains("standing") || lc.contains("walking normally") { r -= 0.10; }
        r.clamp(0.0, 1.0)
    };

    // Stress: urgency + crowding signals
    let stress_level: f64 = {
        let mut s = 0.15f64;
        let person_count = lc.matches("person").count()
            + lc.matches("people").count()
            + lc.matches(" man ").count()
            + lc.matches("woman").count();
        s += (person_count as f64) * 0.12;
        if lc.contains("running") || lc.contains("fast") || lc.contains("rushing") { s += 0.30; }
        if lc.contains("crowd") || lc.contains("group") || lc.contains("several") { s += 0.25; }
        if lc.contains("argument") || lc.contains("shouting") || lc.contains("aggressive") { s += 0.35; }
        if lc.contains("dark") || lc.contains("night") || lc.contains("low light") { s += 0.10; }
        s.clamp(0.0, 1.0)
    };

    // Confidence: longer, specific captions are more reliable
    let confidence: f64 = if lc.contains("unclear") || lc.contains("obscured") || lc.contains("partial") {
        0.48
    } else if caption.len() > 60 {
        0.83
    } else if caption.len() > 25 {
        0.66
    } else {
        0.50
    };

    // Object detection
    let object_held = if lc.contains("gun") || lc.contains("weapon") || lc.contains("firearm") {
        Some("weapon".to_string())
    } else if lc.contains("phone") || lc.contains("smartphone") {
        Some("phone".to_string())
    } else if lc.contains("bag") || lc.contains("backpack") || lc.contains("suitcase") {
        Some("bag".to_string())
    } else if lc.contains("knife") {
        Some("knife".to_string())
    } else {
        None
    };

    let hands_visible: u8 = if lc.contains("hand") || lc.contains("holding") { 1 } else { 0 };

    // Extra tags
    let mut extra_tags = vec!["vlm-derived".to_string()];
    if object_held.as_deref() == Some("weapon") || object_held.as_deref() == Some("knife") {
        extra_tags.push("threat-object".to_string());
    }
    if lc.contains("night") || lc.contains("dark") { extra_tags.push("low-light".to_string()); }
    if lc.contains("multiple") || lc.contains("group") || lc.contains("crowd") {
        extra_tags.push("multiple-persons".to_string());
    }

    VisionEvent {
        event_id: uuid::Uuid::new_v4().to_string(),
        timestamp_ms,
        camera_id: camera_id.to_string(),
        behavior: behavior.to_string(),
        risk_score,
        stress_level,
        confidence,
        person_detected,
        person_name: None,
        hands_visible,
        object_held,
        extra_tags,
        vlm_caption: Some(caption.to_string()),
    }
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn triad_clamps_to_unit_range() {
        let t = AffectTriad::new(-0.5, 1.5, 0.5);
        assert_eq!(t.judgement, 0.0);
        assert_eq!(t.doubt, 1.0);
        assert_eq!(t.determination, 0.5);
    }

    #[test]
    fn triad_from_stress_confidence_boundaries() {
        let calm_certain = AffectTriad::from_stress_confidence(0.0, 1.0);
        assert_eq!(calm_certain.judgement, 1.0);
        assert_eq!(calm_certain.doubt, 0.0);
        assert_eq!(calm_certain.determination, 1.0);

        let stressed_uncertain = AffectTriad::from_stress_confidence(1.0, 0.0);
        assert_eq!(stressed_uncertain.judgement, 0.0);
        assert_eq!(stressed_uncertain.doubt, 1.0);
        assert_eq!(stressed_uncertain.determination, 0.0);
    }

    #[test]
    fn triad_from_vision_event_high_risk() {
        let event = VisionEvent {
            event_id: "test".into(),
            timestamp_ms: 0,
            camera_id: "c0".into(),
            behavior: "loitering".into(),
            risk_score: 0.9,
            stress_level: 0.8,
            confidence: 0.85,
            person_detected: true,
            person_name: None,
            hands_visible: 2,
            object_held: None,
            extra_tags: vec![],
            vlm_caption: None,
        };
        let t = AffectTriad::from_vision_event(&event);
        // High risk + high stress → low judgement
        assert!(t.judgement < 0.3, "judgement should be low: {}", t.judgement);
        // High confidence but high risk → moderate-high doubt
        assert!(t.doubt > 0.3, "doubt should be elevated: {}", t.doubt);
        // High confidence + risk + stress → high determination
        assert!(t.determination > 0.6, "determination should be high: {}", t.determination);
    }

    #[test]
    fn triad_from_alarm_event() {
        let alarm = AlarmEvent {
            alarm_id: "a1".into(),
            timestamp_ms: 0,
            level: AlarmLevel::High,
            danger_score: 0.85,
            risk_score: 0.9,
            stress_level: 0.7,
            person_detected: true,
            person_name: None,
            vlm_caption: None,
            note: "test".into(),
        };
        let t = AffectTriad::from_alarm_event(&alarm);
        assert!(t.determination > 0.5, "high alarm → high determination");
        assert!(t.doubt > 0.3, "high danger → doubt");
    }

    #[test]
    fn decide_safety_alarm() {
        // High determination, low doubt → Alarm
        let t = AffectTriad::new(0.5, 0.2, 0.8);
        assert_eq!(decide(&t, true), DecisionAction::Alarm);
    }

    #[test]
    fn decide_safety_pause() {
        // Low determination, high doubt → Pause
        let t = AffectTriad::new(0.5, 0.8, 0.3);
        assert_eq!(decide(&t, true), DecisionAction::Pause);
    }

    #[test]
    fn decide_safety_none() {
        // Middle ground → None
        let t = AffectTriad::new(0.5, 0.5, 0.5);
        assert_eq!(decide(&t, true), DecisionAction::None);
    }

    #[test]
    fn decide_relational_challenge() {
        let t = AffectTriad::new(0.8, 0.6, 0.5);
        assert_eq!(decide(&t, false), DecisionAction::Challenge);
    }

    #[test]
    fn decide_relational_support() {
        let t = AffectTriad::new(0.7, 0.3, 0.5);
        assert_eq!(decide(&t, false), DecisionAction::Support);
    }

    #[test]
    fn decide_relational_reassure() {
        let t = AffectTriad::new(0.3, 0.8, 0.5);
        assert_eq!(decide(&t, false), DecisionAction::Reassure);
    }

    #[test]
    fn decide_relational_none() {
        let t = AffectTriad::new(0.5, 0.5, 0.5);
        assert_eq!(decide(&t, false), DecisionAction::None);
    }

    #[test]
    fn dominant_dimension() {
        assert_eq!(AffectTriad::new(0.9, 0.1, 0.1).dominant(), "judgement");
        assert_eq!(AffectTriad::new(0.1, 0.9, 0.1).dominant(), "doubt");
        assert_eq!(AffectTriad::new(0.1, 0.1, 0.9).dominant(), "determination");
    }

    #[test]
    fn decision_action_display() {
        assert_eq!(DecisionAction::Alarm.to_string(), "alarm");
        assert_eq!(DecisionAction::None.to_string(), "none");
    }

    #[test]
    fn decision_action_is_actionable() {
        assert!(!DecisionAction::None.is_actionable());
        assert!(DecisionAction::Alarm.is_actionable());
        assert!(DecisionAction::Support.is_actionable());
    }

    #[test]
    fn alarm_severity_ordering() {
        assert!(alarm_severity(&AlarmLevel::High) > alarm_severity(&AlarmLevel::Medium));
        assert!(alarm_severity(&AlarmLevel::Medium) > alarm_severity(&AlarmLevel::Low));
        assert!(alarm_severity(&AlarmLevel::Low) > alarm_severity(&AlarmLevel::None));
    }

    // ── AlarmGrader / Hysteresis tests ─────────────────────────────────

    fn make_event(risk: f64, stress: f64, confidence: f64) -> VisionEvent {
        VisionEvent {
            event_id: "test".into(),
            timestamp_ms: 0,
            camera_id: "c0".into(),
            behavior: "test".into(),
            risk_score: risk,
            stress_level: stress,
            confidence,
            person_detected: true,
            person_name: None,
            hands_visible: 2,
            object_held: None,
            extra_tags: vec![],
            vlm_caption: None,
        }
    }

    #[test]
    fn grader_danger_score_normalized() {
        let g = AlarmGrader::new();
        // All zeros → low danger
        assert!((g.compute_danger_score(0.0, 0.0, 1.0) - 0.0).abs() < 0.01);
        // All max threat
        let max = g.compute_danger_score(1.0, 1.0, 0.0);
        assert!((max - 1.0).abs() < 0.01, "max danger should be ~1.0, got {max}");
    }

    #[test]
    fn grader_level_mapping() {
        let g = AlarmGrader::new(); // thresholds: [0.3, 0.5, 0.8]
        assert_eq!(g.map_danger_to_level(0.0), AlarmLevel::None);
        assert_eq!(g.map_danger_to_level(0.29), AlarmLevel::None);
        assert_eq!(g.map_danger_to_level(0.35), AlarmLevel::Low);
        assert_eq!(g.map_danger_to_level(0.55), AlarmLevel::Medium);
        assert_eq!(g.map_danger_to_level(0.85), AlarmLevel::High);
    }

    #[test]
    fn grader_first_event_no_hysteresis() {
        let mut g = AlarmGrader::new();
        let event = make_event(0.9, 0.9, 0.1); // high danger
        let alarm = g.grade_event(&event);
        assert_eq!(alarm.level, AlarmLevel::High);
    }

    #[test]
    fn hysteresis_prevents_immediate_escalation_to_high() {
        let mut g = AlarmGrader::new();

        // First event: should produce Medium (danger ~0.6)
        // danger = (risk + stress + (1 - confidence)) / 1.8
        // (0.4 + 0.3 + 0.38) / 1.8 = 0.6 → Medium
        let e1 = make_event(0.4, 0.3, 0.62);
        let a1 = g.grade_event(&e1);
        assert_eq!(
            a1.level,
            AlarmLevel::Medium,
            "first event should be Medium, danger={:.2}",
            a1.danger_score
        );

        // Second event: try to jump to High — hysteresis should hold at Medium
        // (high_streak < 2 needed to escalate from Medium to High)
        let e2 = make_event(0.95, 0.95, 0.1);
        let a2 = g.grade_event(&e2);
        assert_eq!(
            a2.level,
            AlarmLevel::Medium,
            "hysteresis should prevent immediate escalation to High"
        );
    }

    #[test]
    fn hysteresis_sustained_high_eventually_escalates() {
        let mut g = AlarmGrader::new();

        // Send enough high-danger events to exceed hysteresis window
        for _ in 0..5 {
            let e = make_event(0.95, 0.95, 0.1);
            g.grade_event(&e);
        }

        let final_alarm = g.grade_event(&make_event(0.95, 0.95, 0.1));
        assert_eq!(
            final_alarm.level,
            AlarmLevel::High,
            "sustained high should escalate"
        );
    }

    #[test]
    fn grader_summary_reflects_last_event() {
        let mut g = AlarmGrader::new();
        let e = make_event(0.6, 0.6, 0.4);
        g.grade_event(&e);
        let summary = g.summary();
        assert_ne!(summary.danger_score, 0.0);
        assert!(!summary.last_n_alarms.is_empty());
    }

    #[test]
    fn grader_history_capped() {
        let mut g = AlarmGrader::new();
        g.history_len = 5;
        for _i in 0..10 {
            g.grade_event(&make_event(0.5, 0.5, 0.5));
        }
        assert_eq!(g.recent_events.len(), 5);
    }

    // ── AffectTriad edge-case tests ────────────────────────────────────

    #[test]
    fn triad_from_stress_confidence_midpoint() {
        let t = AffectTriad::from_stress_confidence(0.5, 0.5);
        assert_eq!(t.judgement, 0.5);
        assert_eq!(t.doubt, 0.5);
        assert_eq!(t.determination, 0.5);
    }

    #[test]
    fn triad_from_stress_confidence_clamps_out_of_range() {
        let t = AffectTriad::from_stress_confidence(-0.5, 1.5);
        // Inputs are clamped inside new()
        assert!(t.judgement >= 0.0 && t.judgement <= 1.0);
        assert!(t.doubt >= 0.0 && t.doubt <= 1.0);
        assert!(t.determination >= 0.0 && t.determination <= 1.0);
    }

    #[test]
    fn triad_from_vision_event_calm_scene() {
        let event = make_event(0.1, 0.1, 0.95);
        let t = AffectTriad::from_vision_event(&event);
        assert!(t.judgement > 0.8, "calm scene → high judgement: {}", t.judgement);
        assert!(t.doubt < 0.15, "high confidence → low doubt: {}", t.doubt);
    }

    #[test]
    fn triad_display_format() {
        let t = AffectTriad::new(0.123, 0.456, 0.789);
        let s = format!("{t}");
        assert!(s.contains("J=0.12"), "display should show J: {s}");
        assert!(s.contains("D=0.46"), "display should show D: {s}");
        assert!(s.contains("Dt=0.79"), "display should show Dt: {s}");
    }

    #[test]
    fn decide_support_via_high_determination() {
        // Dt > 0.7 and J > 0.5, but doesn't hit challenge/support branches first
        let t = AffectTriad::new(0.55, 0.45, 0.8);
        assert_eq!(decide(&t, false), DecisionAction::Support);
    }
}
