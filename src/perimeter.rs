//! perimeter — operator-defined security zones + spatial risk reasoning.
//!
//! This is the piece that turns Sentinelle from a system that thinks in
//! **events** ("a person was detected") into one that thinks **spatially**
//! ("a person is in the RESTRICTED east-gate zone while the perimeter is
//! ARMED"). It is the durable thesis from the April Vision note
//! (`os/Notes/Sentinelle Vision.md`) and os/117 gap #1.
//!
//! It layers *security semantics* on top of the pure geometric
//! [`crate::detector::Zone`] / [`crate::detector::point_in_polygon`]:
//!
//!   * [`ZoneKind`] — what a zone *means* (perimeter / entry / restricted /
//!     interior / safe) and how much risk a presence there contributes.
//!   * [`SecurityZone`] — a named, kinded, individually-armable polygon.
//!   * [`PerimeterMap`] — all zones for one camera + a global arming state,
//!     loadable from an operator-authored JSON config.
//!   * [`PerimeterAssessment`] — the per-frame verdict: spatial risk delta +
//!     zone tags, fed into the same risk fusion the rest of the pipeline uses.
//!
//! PURE: no I/O in the logic. Config loading is a thin `std::fs` wrapper around
//! the pure [`PerimeterMap::from_config_json`] parser so tests stay hermetic.

use serde::{Deserialize, Serialize};

use crate::detector::{point_in_polygon, Detection};

fn default_true() -> bool {
    true
}

/// Env var pointing at the operator perimeter config (JSON). Unset → every
/// camera falls back to a single armed full-frame perimeter (status quo: any
/// relevant detection fires the VLM, modest spatial risk).
pub const PERIMETER_CONFIG_ENV: &str = "SENTINELLE_PERIMETER_CONFIG";

/// Security meaning of a zone. Drives the spatial risk contribution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ZoneKind {
    /// Outer boundary of the property. Presence here matters mostly when armed.
    Perimeter,
    /// Door / gate / driveway — expected transit, logged, mild escalation armed.
    Entry,
    /// Nobody should ever be here (vault, plant room, pool, roof). Always escalates.
    Restricted,
    /// Normal occupied interior — low weight, only notable when armed (night/away).
    Interior,
    /// Known-safe area (own porch when disarmed). Suppresses risk.
    Safe,
}

impl ZoneKind {
    /// Spatial risk delta contributed when a relevant detection
    /// (person/vehicle) sits inside a zone of this kind, given arming state.
    ///
    /// Roughly `[-0.15, 0.60]`. Restricted dominates regardless of arming;
    /// Safe is negative (suppresses). These are *deltas* folded into the same
    /// `behavior::fuse_risk` the watchlist/behaviour signals use.
    pub fn risk_delta(self, armed: bool) -> f32 {
        match self {
            ZoneKind::Restricted => 0.60,
            ZoneKind::Perimeter => if armed { 0.45 } else { 0.10 },
            ZoneKind::Entry => if armed { 0.30 } else { 0.05 },
            ZoneKind::Interior => if armed { 0.25 } else { 0.0 },
            ZoneKind::Safe => -0.15,
        }
    }

    fn tag(self) -> &'static str {
        match self {
            ZoneKind::Perimeter => "perimeter",
            ZoneKind::Entry => "entry",
            ZoneKind::Restricted => "restricted",
            ZoneKind::Interior => "interior",
            ZoneKind::Safe => "safe",
        }
    }
}

/// An operator-defined security zone (one polygon, in normalized 0..1 image
/// coordinates) for a single camera.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SecurityZone {
    pub name: String,
    pub kind: ZoneKind,
    /// Per-zone arm override. A zone is *effective-armed* only when both the
    /// map and the zone are armed (a Restricted zone still escalates either way
    /// via its base weight). Defaults to armed.
    #[serde(default = "default_true")]
    pub armed: bool,
    /// Ordered ring of `(x, y)` vertices; closing edge is implicit.
    pub polygon: Vec<(f32, f32)>,
}

impl SecurityZone {
    pub fn new(name: impl Into<String>, kind: ZoneKind, polygon: Vec<(f32, f32)>) -> Self {
        Self { name: name.into(), kind, armed: true, polygon }
    }

    /// True when the detection's bbox centroid lies inside this zone.
    pub fn contains_detection(&self, det: &Detection) -> bool {
        let (cx, cy) = det.bbox.centroid();
        point_in_polygon(cx, cy, &self.polygon)
    }
}

/// All security zones for one camera plus the global arming state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PerimeterMap {
    pub camera_id: String,
    /// Global arming state for this camera (operator armed the system).
    #[serde(default = "default_true")]
    pub armed: bool,
    #[serde(default)]
    pub zones: Vec<SecurityZone>,
}

/// The per-frame spatial verdict.
#[derive(Debug, Clone, PartialEq)]
pub struct PerimeterAssessment {
    /// Max spatial risk delta across all (relevant detection, matched zone)
    /// pairs. `0.0` when nothing relevant landed in any zone. May be negative
    /// when the only matches are Safe zones (suppression).
    pub risk_delta: f32,
    /// Stable tags for the event, e.g. `["zone:east-gate", "zone-kind:restricted"]`.
    pub zone_tags: Vec<String>,
    /// True when at least one relevant detection sits inside at least one zone.
    pub in_zone: bool,
}

impl PerimeterMap {
    /// Default map: one armed full-frame perimeter. Preserves the prior
    /// "any relevant detection fires the VLM" behaviour and gives a mild
    /// armed-perimeter risk, so unconfigured deployments still behave sanely.
    pub fn full_frame(camera_id: impl Into<String>) -> Self {
        Self {
            camera_id: camera_id.into(),
            armed: true,
            zones: vec![SecurityZone::new(
                "full-frame",
                ZoneKind::Perimeter,
                vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
            )],
        }
    }

    /// Geometric zones for the FastVLM gate ([`crate::detector::should_invoke_vlm`]).
    /// Falls back to a full-frame zone when no zones are configured so the gate
    /// never silently closes on a misconfigured map.
    pub fn gate_zones(&self) -> Vec<crate::detector::Zone> {
        if self.zones.is_empty() {
            return PerimeterMap::full_frame(self.camera_id.clone()).gate_zones();
        }
        self.zones
            .iter()
            .map(|z| crate::detector::Zone::new(z.name.clone(), z.polygon.clone()))
            .collect()
    }

    /// Zones (by reference) whose polygon contains the detection's centroid.
    pub fn zones_for<'a>(&'a self, det: &Detection) -> Vec<&'a SecurityZone> {
        self.zones.iter().filter(|z| z.contains_detection(det)).collect()
    }

    /// Evaluate a frame's detections into a [`PerimeterAssessment`].
    ///
    /// Only relevant detections (`person`/`vehicle`, via
    /// [`Detection::is_relevant_class`]) contribute. The spatial risk is the
    /// **max** delta over every (relevant detection × matched zone) pair, so the
    /// single most dangerous placement dominates (a person in a Restricted zone
    /// is not averaged away by another standing in a Safe zone).
    pub fn evaluate(&self, detections: &[Detection]) -> PerimeterAssessment {
        let mut deltas: Vec<f32> = Vec::new();
        let mut tags: Vec<String> = Vec::new();
        let mut in_zone = false;

        for det in detections.iter().filter(|d| d.is_relevant_class()) {
            for z in self.zones_for(det) {
                in_zone = true;
                let eff_armed = self.armed && z.armed;
                deltas.push(z.kind.risk_delta(eff_armed));
                let zt = format!("zone:{}", z.name);
                if !tags.contains(&zt) {
                    tags.push(zt);
                }
                let kt = format!("zone-kind:{}", z.kind.tag());
                if !tags.contains(&kt) {
                    tags.push(kt);
                }
            }
        }

        let risk_delta = deltas
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let risk_delta = if deltas.is_empty() { 0.0 } else { risk_delta.clamp(-0.20, 0.70) };

        PerimeterAssessment { risk_delta, zone_tags: tags, in_zone }
    }

    /// Parse an operator perimeter config (pure; no fs).
    ///
    /// Config shape — a JSON object keyed by `camera_id`:
    /// ```json
    /// {
    ///   "ssc-cabled-ch1": {
    ///     "armed": true,
    ///     "zones": [
    ///       {"name":"east-gate","kind":"entry","polygon":[[0.0,0.5],[0.3,0.5],[0.3,1.0],[0.0,1.0]]},
    ///       {"name":"pool","kind":"restricted","polygon":[[0.6,0.6],[1.0,0.6],[1.0,1.0],[0.6,1.0]]}
    ///     ]
    ///   }
    /// }
    /// ```
    /// Returns the map for `camera_id`, or `None` if the camera is absent or
    /// the JSON is malformed (callers fall back to [`PerimeterMap::full_frame`]).
    pub fn from_config_json(json: &str, camera_id: &str) -> Option<PerimeterMap> {
        #[derive(Deserialize)]
        struct Entry {
            #[serde(default = "default_true")]
            armed: bool,
            #[serde(default)]
            zones: Vec<SecurityZone>,
        }
        let parsed: std::collections::HashMap<String, Entry> = serde_json::from_str(json).ok()?;
        let e = parsed.get(camera_id)?;
        Some(PerimeterMap {
            camera_id: camera_id.to_string(),
            armed: e.armed,
            zones: e.zones.clone(),
        })
    }

    /// Load the perimeter map for `camera_id` from the `SENTINELLE_PERIMETER_CONFIG`
    /// file. Falls back to [`PerimeterMap::full_frame`] when the env var is unset,
    /// the file is unreadable, or the camera has no configured zones.
    pub fn load_for(camera_id: &str) -> PerimeterMap {
        let path = match std::env::var(PERIMETER_CONFIG_ENV) {
            Ok(p) if !p.trim().is_empty() => p,
            _ => return PerimeterMap::full_frame(camera_id),
        };
        let json = match std::fs::read_to_string(&path) {
            Ok(s) => s,
            Err(_) => return PerimeterMap::full_frame(camera_id),
        };
        PerimeterMap::from_config_json(&json, camera_id)
            .filter(|m| !m.zones.is_empty())
            .unwrap_or_else(|| PerimeterMap::full_frame(camera_id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::detector::{BBox, Detection};

    fn person_at(cx: f32, cy: f32) -> Detection {
        // bbox whose centroid is (cx, cy)
        Detection::new("person", 0.9, BBox::new(cx - 0.02, cy - 0.02, 0.04, 0.04))
    }

    fn square(x0: f32, y0: f32, x1: f32, y1: f32) -> Vec<(f32, f32)> {
        vec![(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    }

    #[test]
    fn risk_delta_ordering_armed() {
        assert!(ZoneKind::Restricted.risk_delta(true) > ZoneKind::Perimeter.risk_delta(true));
        assert!(ZoneKind::Perimeter.risk_delta(true) > ZoneKind::Entry.risk_delta(true));
        assert!(ZoneKind::Entry.risk_delta(true) > ZoneKind::Interior.risk_delta(true));
        assert!(ZoneKind::Safe.risk_delta(true) < 0.0);
    }

    #[test]
    fn arming_raises_perimeter_and_entry() {
        assert!(ZoneKind::Perimeter.risk_delta(true) > ZoneKind::Perimeter.risk_delta(false));
        assert!(ZoneKind::Entry.risk_delta(true) > ZoneKind::Entry.risk_delta(false));
        // Restricted ignores arming.
        assert_eq!(ZoneKind::Restricted.risk_delta(true), ZoneKind::Restricted.risk_delta(false));
    }

    #[test]
    fn full_frame_gate_and_evaluate() {
        let m = PerimeterMap::full_frame("cam1");
        // gate covers the whole frame
        assert_eq!(m.gate_zones().len(), 1);
        let a = m.evaluate(&[person_at(0.5, 0.5)]);
        assert!(a.in_zone);
        assert!((a.risk_delta - ZoneKind::Perimeter.risk_delta(true)).abs() < 1e-6);
        assert!(a.zone_tags.iter().any(|t| t == "zone-kind:perimeter"));
    }

    #[test]
    fn restricted_dominates_safe() {
        let m = PerimeterMap {
            camera_id: "cam1".into(),
            armed: true,
            zones: vec![
                SecurityZone::new("pool", ZoneKind::Restricted, square(0.4, 0.4, 0.9, 0.9)),
                SecurityZone::new("porch", ZoneKind::Safe, square(0.0, 0.0, 0.6, 0.6)),
            ],
        };
        // (0.5,0.5) is inside BOTH pool and porch → restricted wins.
        let a = m.evaluate(&[person_at(0.5, 0.5)]);
        assert!((a.risk_delta - 0.60).abs() < 1e-6);
        assert!(a.zone_tags.iter().any(|t| t == "zone:pool"));
        assert!(a.zone_tags.iter().any(|t| t == "zone:porch"));
    }

    #[test]
    fn safe_only_suppresses() {
        let m = PerimeterMap {
            camera_id: "cam1".into(),
            armed: true,
            zones: vec![SecurityZone::new("porch", ZoneKind::Safe, square(0.0, 0.0, 0.6, 0.6))],
        };
        let a = m.evaluate(&[person_at(0.2, 0.2)]);
        assert!(a.in_zone);
        assert!(a.risk_delta < 0.0);
    }

    #[test]
    fn detection_outside_all_zones_is_neutral() {
        let m = PerimeterMap {
            camera_id: "cam1".into(),
            armed: true,
            zones: vec![SecurityZone::new("gate", ZoneKind::Entry, square(0.0, 0.0, 0.2, 0.2))],
        };
        let a = m.evaluate(&[person_at(0.9, 0.9)]);
        assert!(!a.in_zone);
        assert_eq!(a.risk_delta, 0.0);
        assert!(a.zone_tags.is_empty());
    }

    #[test]
    fn irrelevant_class_ignored() {
        let m = PerimeterMap::full_frame("cam1");
        let cat = Detection::new("cat", 0.99, BBox::new(0.48, 0.48, 0.04, 0.04));
        let a = m.evaluate(&[cat]);
        assert!(!a.in_zone);
        assert_eq!(a.risk_delta, 0.0);
    }

    #[test]
    fn per_zone_disarm_lowers_delta() {
        let armed = SecurityZone::new("front", ZoneKind::Perimeter, square(0.0, 0.0, 1.0, 1.0));
        let mut disarmed = armed.clone();
        disarmed.armed = false;
        let m_armed = PerimeterMap { camera_id: "c".into(), armed: true, zones: vec![armed] };
        let m_dis = PerimeterMap { camera_id: "c".into(), armed: true, zones: vec![disarmed] };
        let a1 = m_armed.evaluate(&[person_at(0.5, 0.5)]);
        let a2 = m_dis.evaluate(&[person_at(0.5, 0.5)]);
        assert!(a1.risk_delta > a2.risk_delta);
    }

    #[test]
    fn config_json_parses_named_camera() {
        let json = r#"{
            "ssc-cabled-ch1": {
                "armed": true,
                "zones": [
                    {"name":"east-gate","kind":"entry","polygon":[[0.0,0.5],[0.3,0.5],[0.3,1.0],[0.0,1.0]]},
                    {"name":"pool","kind":"restricted","armed":true,"polygon":[[0.6,0.6],[1.0,0.6],[1.0,1.0],[0.6,1.0]]}
                ]
            }
        }"#;
        let m = PerimeterMap::from_config_json(json, "ssc-cabled-ch1").unwrap();
        assert_eq!(m.zones.len(), 2);
        assert_eq!(m.zones[0].kind, ZoneKind::Entry);
        assert_eq!(m.zones[1].kind, ZoneKind::Restricted);
        // a person at (0.8,0.8) is in the pool (restricted)
        let a = m.evaluate(&[person_at(0.8, 0.8)]);
        assert!((a.risk_delta - 0.60).abs() < 1e-6);
        // absent camera → None
        assert!(PerimeterMap::from_config_json(json, "nope").is_none());
        // malformed → None
        assert!(PerimeterMap::from_config_json("{not json", "x").is_none());
    }
}
