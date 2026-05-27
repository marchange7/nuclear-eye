// detector.rs — Phase 2 "always-on detector → FastVLM-on-hit" gating layer.
//
// Architecture:
//   A lightweight object detector (YOLOX / NanoDet class) runs on EVERY frame.
//   The heavy FastVLM scene captioner is invoked ONLY when the detector produces
//   a relevant hit — a `person` / `vehicle` whose centroid falls inside a watched
//   zone, above a confidence threshold. This module is the pure-logic gate plus
//   the detector service HTTP contract.
//
// Everything in this file is unit-testable without a live model or GPU:
//   - Geometry (point-in-polygon, tripwire segment crossing) is pure std.
//   - `should_invoke_vlm` / `detections_to_vision_fields` are pure functions.
//   - `HttpDetector` is a thin POST/parse stub; model I/O stays out of the tests.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// Default detector service URL. Overridden by the `DETECTOR_URL` env var.
const DEFAULT_DETECTOR_URL: &str = "http://127.0.0.1:8094/detect";

/// Default confidence threshold below which detections do not gate the VLM.
pub const DEFAULT_CONF_THRESHOLD: f32 = 0.35;

/// Detector read timeout (ms). The detector is the per-frame fast path, so this
/// is deliberately tight — a slow detector must not stall the pipeline.
const DETECT_TIMEOUT_MS: u64 = 1500;

// ── Geometry ────────────────────────────────────────────────────────────────

/// Axis-aligned bounding box in normalized image coordinates (0..1).
///
/// `(x, y)` is the top-left corner; `w` / `h` are width / height.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BBox {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

impl BBox {
    pub fn new(x: f32, y: f32, w: f32, h: f32) -> Self {
        Self { x, y, w, h }
    }

    /// Centroid of the box in normalized coordinates.
    pub fn centroid(&self) -> (f32, f32) {
        (self.x + self.w / 2.0, self.y + self.h / 2.0)
    }
}

/// A single object detection from the always-on detector.
///
/// `class` is the detector label (e.g. "person", "vehicle", "car", "dog").
/// `confidence` is in 0..1. `bbox` is normalized.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Detection {
    pub class: String,
    pub confidence: f32,
    pub bbox: BBox,
}

impl Detection {
    pub fn new(class: impl Into<String>, confidence: f32, bbox: BBox) -> Self {
        Self {
            class: class.into(),
            confidence,
            bbox,
        }
    }

    /// True when the detection class is one of the VLM-relevant classes
    /// (person / vehicle family). Case-insensitive; `car`, `truck`, `bus`,
    /// `motorcycle`, `motorbike` all count as a vehicle.
    pub fn is_relevant_class(&self) -> bool {
        let c = self.class.trim().to_ascii_lowercase();
        matches!(
            c.as_str(),
            "person"
                | "people"
                | "pedestrian"
                | "vehicle"
                | "car"
                | "truck"
                | "bus"
                | "motorcycle"
                | "motorbike"
                | "van"
        )
    }

    /// True when this detection class belongs to the vehicle family.
    pub fn is_vehicle(&self) -> bool {
        let c = self.class.trim().to_ascii_lowercase();
        matches!(
            c.as_str(),
            "vehicle" | "car" | "truck" | "bus" | "motorcycle" | "motorbike" | "van"
        )
    }

    /// True when this detection class is a person.
    pub fn is_person(&self) -> bool {
        let c = self.class.trim().to_ascii_lowercase();
        matches!(c.as_str(), "person" | "people" | "pedestrian")
    }
}

/// A named polygonal zone in normalized image coordinates.
///
/// The polygon is an ordered ring of `(x, y)` vertices; the closing edge from
/// the last vertex back to the first is implicit.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Zone {
    pub name: String,
    pub polygon: Vec<(f32, f32)>,
}

impl Zone {
    pub fn new(name: impl Into<String>, polygon: Vec<(f32, f32)>) -> Self {
        Self {
            name: name.into(),
            polygon,
        }
    }

    /// Point-in-polygon test (ray casting / even-odd rule).
    ///
    /// Returns `true` when `(px, py)` is strictly inside the polygon OR lies on
    /// an edge. Degenerate polygons (< 3 vertices) always return `false`.
    pub fn contains(&self, px: f32, py: f32) -> bool {
        point_in_polygon(px, py, &self.polygon)
    }
}

/// Ray-casting point-in-polygon with explicit on-edge handling.
///
/// On-edge points are treated as contained (inclusive), which matches operator
/// intent for security zones — a person standing exactly on a zone boundary
/// should still count as inside.
pub fn point_in_polygon(px: f32, py: f32, polygon: &[(f32, f32)]) -> bool {
    let n = polygon.len();
    if n < 3 {
        return false;
    }

    // Inclusive edge: any point lying on a polygon edge counts as inside.
    for i in 0..n {
        let a = polygon[i];
        let b = polygon[(i + 1) % n];
        if point_on_segment(px, py, a, b) {
            return true;
        }
    }

    // Standard even-odd ray cast to the right (+x).
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let (xi, yi) = polygon[i];
        let (xj, yj) = polygon[j];
        let intersects = ((yi > py) != (yj > py))
            && (px < (xj - xi) * (py - yi) / (yj - yi) + xi);
        if intersects {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// True when point `p` lies on segment `a`–`b` (within a small epsilon).
fn point_on_segment(px: f32, py: f32, a: (f32, f32), b: (f32, f32)) -> bool {
    const EPS: f32 = 1e-6;
    let cross = (b.0 - a.0) * (py - a.1) - (b.1 - a.1) * (px - a.0);
    if cross.abs() > EPS {
        return false; // not collinear
    }
    // Within the bounding box of the segment?
    let within_x = px >= a.0.min(b.0) - EPS && px <= a.0.max(b.0) + EPS;
    let within_y = py >= a.1.min(b.1) - EPS && py <= a.1.max(b.1) + EPS;
    within_x && within_y
}

/// A named tripwire — a directed line segment from `a` to `b` in normalized
/// coordinates. A "trip" is registered when a tracked object's path between two
/// consecutive positions crosses the wire.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Tripwire {
    pub name: String,
    pub a: (f32, f32),
    pub b: (f32, f32),
}

impl Tripwire {
    pub fn new(name: impl Into<String>, a: (f32, f32), b: (f32, f32)) -> Self {
        Self {
            name: name.into(),
            a,
            b,
        }
    }

    /// True when the movement segment `prev`→`curr` crosses this tripwire.
    ///
    /// Uses an orientation-based proper-segment-intersection test. A path that
    /// runs parallel to (or merely alongside) the wire does NOT count as a
    /// crossing.
    pub fn crossed(&self, prev: (f32, f32), curr: (f32, f32)) -> bool {
        segments_intersect(self.a, self.b, prev, curr)
    }
}

/// Orientation of the ordered triple (p, q, r):
///   > 0 counter-clockwise, < 0 clockwise, ≈ 0 collinear.
fn orientation(p: (f32, f32), q: (f32, f32), r: (f32, f32)) -> f32 {
    (q.1 - p.1) * (r.0 - q.0) - (q.0 - p.0) * (r.1 - q.1)
}

/// Proper segment-intersection test for segments p1p2 and p3p4.
///
/// Returns `true` for a genuine crossing, including the collinear-overlap case
/// where one endpoint lies on the other segment. Returns `false` for parallel,
/// disjoint, or merely touching-at-distance segments.
fn segments_intersect(p1: (f32, f32), p2: (f32, f32), p3: (f32, f32), p4: (f32, f32)) -> bool {
    const EPS: f32 = 1e-9;
    let d1 = orientation(p3, p4, p1);
    let d2 = orientation(p3, p4, p2);
    let d3 = orientation(p1, p2, p3);
    let d4 = orientation(p1, p2, p4);

    // General case: the two segments straddle each other.
    if ((d1 > EPS && d2 < -EPS) || (d1 < -EPS && d2 > EPS))
        && ((d3 > EPS && d4 < -EPS) || (d3 < -EPS && d4 > EPS))
    {
        return true;
    }

    // Collinear / touching special cases.
    if d1.abs() <= EPS && point_on_segment(p1.0, p1.1, p3, p4) {
        return true;
    }
    if d2.abs() <= EPS && point_on_segment(p2.0, p2.1, p3, p4) {
        return true;
    }
    if d3.abs() <= EPS && point_on_segment(p3.0, p3.1, p1, p2) {
        return true;
    }
    if d4.abs() <= EPS && point_on_segment(p4.0, p4.1, p1, p2) {
        return true;
    }

    false
}

// ── VLM gate ──────────────────────────────────────────────────────────────

/// The FastVLM gate.
///
/// Returns `true` when at least one relevant detection (`person` / `vehicle`)
/// with confidence ≥ `conf_threshold` has its bbox centroid inside any of the
/// given zones. This is the single decision that decides whether to spend the
/// heavy FastVLM scene-caption call on a frame.
///
/// When `zones` is empty there is nothing to guard, so the gate stays closed
/// (returns `false`) — callers that want "any relevant detection fires the VLM"
/// should pass a full-frame zone covering 0..1.
pub fn should_invoke_vlm(detections: &[Detection], zones: &[Zone], conf_threshold: f32) -> bool {
    if zones.is_empty() {
        return false;
    }
    detections.iter().any(|d| {
        if d.confidence < conf_threshold || !d.is_relevant_class() {
            return false;
        }
        let (cx, cy) = d.bbox.centroid();
        zones.iter().any(|z| z.contains(cx, cy))
    })
}

/// Convenience wrapper using the default confidence threshold (0.35).
pub fn should_invoke_vlm_default(detections: &[Detection], zones: &[Zone]) -> bool {
    should_invoke_vlm(detections, zones, DEFAULT_CONF_THRESHOLD)
}

/// Fields derived from detections + zone hits, suitable for populating a
/// `VisionEvent` before (or instead of) a FastVLM caption.
///
/// `base_risk` is a coarse pre-VLM risk floor:
///   - person in a zone:  0.45
///   - vehicle in a zone: 0.40
///   - relevant detection present but outside every zone: 0.20
///   - nothing relevant: 0.05
/// The maximum across all detections wins. A zone hit by a person yields the
/// "loitering" behavior tag; a vehicle zone hit yields "vehicle_present";
/// otherwise "passby" (relevant but out-of-zone) or "no_activity".
pub fn detections_to_vision_fields(
    detections: &[Detection],
    zones: &[Zone],
    conf_threshold: f32,
) -> (bool, String, f64) {
    let mut person_detected = false;
    let mut person_in_zone = false;
    let mut vehicle_in_zone = false;
    let mut relevant_present = false;
    let mut base_risk: f64 = 0.05;

    for d in detections {
        if d.confidence < conf_threshold {
            continue;
        }
        if d.is_person() {
            person_detected = true;
        }
        if !d.is_relevant_class() {
            continue;
        }
        relevant_present = true;
        let (cx, cy) = d.bbox.centroid();
        let in_zone = zones.iter().any(|z| z.contains(cx, cy));

        if in_zone && d.is_person() {
            person_in_zone = true;
            base_risk = base_risk.max(0.45);
        } else if in_zone && d.is_vehicle() {
            vehicle_in_zone = true;
            base_risk = base_risk.max(0.40);
        } else {
            // relevant but out of zone
            base_risk = base_risk.max(0.20);
        }
    }

    let behavior = if person_in_zone {
        "loitering"
    } else if vehicle_in_zone {
        "vehicle_present"
    } else if relevant_present {
        "passby"
    } else {
        "no_activity"
    }
    .to_string();

    (person_detected, behavior, base_risk)
}

// ── Detector service contract ───────────────────────────────────────────────

/// Effective detector URL: `DETECTOR_URL` env var overrides the default.
pub fn detector_url() -> String {
    std::env::var("DETECTOR_URL").unwrap_or_else(|_| DEFAULT_DETECTOR_URL.to_string())
}

/// Wire request: a base64 JPEG plus the confidence floor the detector should
/// apply server-side.
#[derive(Debug, Serialize)]
struct DetectRequest<'a> {
    image_b64: &'a str,
    conf_threshold: f32,
}

/// Wire response from the detector service.
#[derive(Debug, Deserialize)]
pub struct DetectResponse {
    pub ok: bool,
    #[serde(default)]
    pub detections: Vec<Detection>,
    /// Detector model identifier (e.g. "yolox-nano", "nanodet-plus").
    #[serde(default)]
    pub model: Option<String>,
}

/// Contract for an always-on object detector backend.
///
/// Implementors POST a JPEG frame and return the detections found. This trait
/// exists so the gate logic can be tested against an in-memory fake while the
/// production path talks to the HTTP service.
#[allow(async_fn_in_trait)]
pub trait DetectorBackend {
    /// Run detection on a JPEG-encoded frame.
    async fn detect(&self, jpeg: &[u8]) -> Result<Vec<Detection>>;
}

/// HTTP detector talking to the detector service at [`detector_url`].
///
/// This is a thin POST/parse stub: it base64-encodes the JPEG, POSTs the
/// `DetectRequest`, and parses the `DetectResponse`. No model is loaded in this
/// process, and this path is intentionally excluded from the unit tests.
#[derive(Clone)]
pub struct HttpDetector {
    client: reqwest::Client,
    url: String,
    conf_threshold: f32,
}

impl HttpDetector {
    /// Build from a shared client, reading the URL from `DETECTOR_URL`.
    pub fn new(client: reqwest::Client) -> Self {
        Self {
            client,
            url: detector_url(),
            conf_threshold: DEFAULT_CONF_THRESHOLD,
        }
    }

    /// Override the endpoint URL.
    pub fn with_url(mut self, url: impl Into<String>) -> Self {
        self.url = url.into();
        self
    }

    /// Override the server-side confidence threshold.
    pub fn with_conf_threshold(mut self, conf: f32) -> Self {
        self.conf_threshold = conf;
        self
    }
}

impl DetectorBackend for HttpDetector {
    async fn detect(&self, jpeg: &[u8]) -> Result<Vec<Detection>> {
        use base64::Engine as _;
        let image_b64 = base64::engine::general_purpose::STANDARD.encode(jpeg);
        let body = DetectRequest {
            image_b64: &image_b64,
            conf_threshold: self.conf_threshold,
        };

        let resp = self
            .client
            .post(&self.url)
            .json(&body)
            .timeout(std::time::Duration::from_millis(DETECT_TIMEOUT_MS))
            .send()
            .await
            .context("detector HTTP request failed")?
            .error_for_status()
            .context("detector returned error status")?
            .json::<DetectResponse>()
            .await
            .context("failed to parse detector response")?;

        if !resp.ok {
            anyhow::bail!("detector reported not-ok");
        }
        Ok(resp.detections)
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_square() -> Vec<(f32, f32)> {
        vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    }

    fn det(class: &str, conf: f32, cx: f32, cy: f32) -> Detection {
        // Build a small box centered at (cx, cy).
        let w = 0.1;
        let h = 0.1;
        Detection::new(class, conf, BBox::new(cx - w / 2.0, cy - h / 2.0, w, h))
    }

    // ── BBox ──────────────────────────────────────────────────────────────

    #[test]
    fn bbox_centroid() {
        let b = BBox::new(0.2, 0.4, 0.4, 0.2);
        let (cx, cy) = b.centroid();
        assert!((cx - 0.4).abs() < 1e-6);
        assert!((cy - 0.5).abs() < 1e-6);
    }

    // ── point-in-polygon ────────────────────────────────────────────────────

    #[test]
    fn pip_inside() {
        assert!(point_in_polygon(0.5, 0.5, &unit_square()));
    }

    #[test]
    fn pip_outside() {
        assert!(!point_in_polygon(1.5, 0.5, &unit_square()));
        assert!(!point_in_polygon(-0.1, 0.5, &unit_square()));
        assert!(!point_in_polygon(0.5, 1.5, &unit_square()));
    }

    #[test]
    fn pip_on_edge_is_inside() {
        // Point exactly on the bottom edge.
        assert!(point_in_polygon(0.5, 0.0, &unit_square()));
        // Point exactly on a vertex.
        assert!(point_in_polygon(0.0, 0.0, &unit_square()));
        // Point on the right edge.
        assert!(point_in_polygon(1.0, 0.5, &unit_square()));
    }

    #[test]
    fn pip_degenerate_polygon_is_false() {
        assert!(!point_in_polygon(0.5, 0.5, &[(0.0, 0.0), (1.0, 1.0)]));
        assert!(!point_in_polygon(0.5, 0.5, &[]));
    }

    #[test]
    fn pip_concave_polygon() {
        // An arrow / chevron concave shape.
        let poly = vec![
            (0.0, 0.0),
            (1.0, 0.5),
            (0.0, 1.0),
            (0.3, 0.5),
        ];
        // Point inside the body of the chevron (right of the notch tip).
        assert!(point_in_polygon(0.6, 0.5, &poly));
        // Point in the concave notch (left of the tip at x=0.3) is outside.
        assert!(!point_in_polygon(0.15, 0.5, &poly));
    }

    #[test]
    fn zone_contains_delegates() {
        let z = Zone::new("front_door", unit_square());
        assert!(z.contains(0.5, 0.5));
        assert!(!z.contains(2.0, 2.0));
    }

    // ── tripwire crossing ─────────────────────────────────────────────────

    #[test]
    fn tripwire_crosses() {
        // Vertical wire at x=0.5; horizontal path from left to right crosses it.
        let wire = Tripwire::new("gate", (0.5, 0.0), (0.5, 1.0));
        assert!(wire.crossed((0.2, 0.5), (0.8, 0.5)));
    }

    #[test]
    fn tripwire_parallel_does_not_cross() {
        // Vertical wire at x=0.5; a parallel vertical path never crosses it.
        let wire = Tripwire::new("gate", (0.5, 0.0), (0.5, 1.0));
        assert!(!wire.crossed((0.2, 0.1), (0.2, 0.9)));
    }

    #[test]
    fn tripwire_path_stops_short_no_cross() {
        // Horizontal path that ends before reaching the wire at x=0.5.
        let wire = Tripwire::new("gate", (0.5, 0.0), (0.5, 1.0));
        assert!(!wire.crossed((0.1, 0.5), (0.4, 0.5)));
    }

    #[test]
    fn tripwire_collinear_overlap_counts() {
        // Path running along the same line as the wire and overlapping it.
        let wire = Tripwire::new("gate", (0.0, 0.5), (1.0, 0.5));
        assert!(wire.crossed((0.5, 0.5), (1.5, 0.5)));
    }

    #[test]
    fn tripwire_endpoint_touch_counts() {
        // Path that ends exactly on the wire.
        let wire = Tripwire::new("gate", (0.5, 0.0), (0.5, 1.0));
        assert!(wire.crossed((0.2, 0.5), (0.5, 0.5)));
    }

    // ── VLM gate ────────────────────────────────────────────────────────────

    #[test]
    fn gate_person_in_zone_fires() {
        let zones = vec![Zone::new("z", unit_square())];
        let dets = vec![det("person", 0.9, 0.5, 0.5)];
        assert!(should_invoke_vlm(&dets, &zones, DEFAULT_CONF_THRESHOLD));
        assert!(should_invoke_vlm_default(&dets, &zones));
    }

    #[test]
    fn gate_person_out_of_zone_does_not_fire() {
        let zones = vec![Zone::new("z", unit_square())];
        // Centroid at (2.0, 2.0) is outside the unit square.
        let dets = vec![det("person", 0.95, 2.0, 2.0)];
        assert!(!should_invoke_vlm(&dets, &zones, DEFAULT_CONF_THRESHOLD));
    }

    #[test]
    fn gate_low_confidence_does_not_fire() {
        let zones = vec![Zone::new("z", unit_square())];
        let dets = vec![det("person", 0.20, 0.5, 0.5)]; // below 0.35
        assert!(!should_invoke_vlm(&dets, &zones, DEFAULT_CONF_THRESHOLD));
    }

    #[test]
    fn gate_vehicle_in_zone_fires() {
        let zones = vec![Zone::new("driveway", unit_square())];
        let dets = vec![det("car", 0.7, 0.5, 0.5)];
        assert!(should_invoke_vlm(&dets, &zones, DEFAULT_CONF_THRESHOLD));
    }

    #[test]
    fn gate_irrelevant_class_does_not_fire() {
        let zones = vec![Zone::new("z", unit_square())];
        // A dog in the zone is not a VLM-relevant class.
        let dets = vec![det("dog", 0.99, 0.5, 0.5)];
        assert!(!should_invoke_vlm(&dets, &zones, DEFAULT_CONF_THRESHOLD));
    }

    #[test]
    fn gate_empty_zones_stays_closed() {
        let dets = vec![det("person", 0.99, 0.5, 0.5)];
        assert!(!should_invoke_vlm(&dets, &[], DEFAULT_CONF_THRESHOLD));
    }

    #[test]
    fn gate_confidence_at_threshold_fires() {
        let zones = vec![Zone::new("z", unit_square())];
        let dets = vec![det("person", DEFAULT_CONF_THRESHOLD, 0.5, 0.5)];
        // conf == threshold is inclusive (>=).
        assert!(should_invoke_vlm(&dets, &zones, DEFAULT_CONF_THRESHOLD));
    }

    // ── detections_to_vision_fields ─────────────────────────────────────────

    #[test]
    fn fields_person_in_zone() {
        let zones = vec![Zone::new("z", unit_square())];
        let dets = vec![det("person", 0.9, 0.5, 0.5)];
        let (person, behavior, risk) = detections_to_vision_fields(&dets, &zones, DEFAULT_CONF_THRESHOLD);
        assert!(person);
        assert_eq!(behavior, "loitering");
        assert!((risk - 0.45).abs() < 1e-9, "risk={risk}");
    }

    #[test]
    fn fields_vehicle_in_zone() {
        let zones = vec![Zone::new("driveway", unit_square())];
        let dets = vec![det("truck", 0.8, 0.5, 0.5)];
        let (person, behavior, risk) = detections_to_vision_fields(&dets, &zones, DEFAULT_CONF_THRESHOLD);
        assert!(!person);
        assert_eq!(behavior, "vehicle_present");
        assert!((risk - 0.40).abs() < 1e-9, "risk={risk}");
    }

    #[test]
    fn fields_person_out_of_zone_is_passby() {
        let zones = vec![Zone::new("z", unit_square())];
        let dets = vec![det("person", 0.9, 2.0, 2.0)];
        let (person, behavior, risk) = detections_to_vision_fields(&dets, &zones, DEFAULT_CONF_THRESHOLD);
        assert!(person, "person still detected even out of zone");
        assert_eq!(behavior, "passby");
        assert!((risk - 0.20).abs() < 1e-9, "risk={risk}");
    }

    #[test]
    fn fields_nothing_relevant() {
        let zones = vec![Zone::new("z", unit_square())];
        let dets = vec![det("dog", 0.9, 0.5, 0.5)];
        let (person, behavior, risk) = detections_to_vision_fields(&dets, &zones, DEFAULT_CONF_THRESHOLD);
        assert!(!person);
        assert_eq!(behavior, "no_activity");
        assert!((risk - 0.05).abs() < 1e-9, "risk={risk}");
    }

    #[test]
    fn fields_low_conf_ignored() {
        let zones = vec![Zone::new("z", unit_square())];
        let dets = vec![det("person", 0.10, 0.5, 0.5)];
        let (person, behavior, risk) = detections_to_vision_fields(&dets, &zones, DEFAULT_CONF_THRESHOLD);
        assert!(!person, "below-threshold person is ignored");
        assert_eq!(behavior, "no_activity");
        assert!((risk - 0.05).abs() < 1e-9);
    }

    #[test]
    fn fields_person_zone_dominates_vehicle() {
        // Both a person and a vehicle in the zone; person risk (0.45) wins.
        let zones = vec![Zone::new("z", unit_square())];
        let dets = vec![det("car", 0.9, 0.4, 0.4), det("person", 0.9, 0.6, 0.6)];
        let (person, behavior, risk) = detections_to_vision_fields(&dets, &zones, DEFAULT_CONF_THRESHOLD);
        assert!(person);
        assert_eq!(behavior, "loitering");
        assert!((risk - 0.45).abs() < 1e-9, "risk={risk}");
    }

    // ── class helpers ───────────────────────────────────────────────────────

    #[test]
    fn class_helpers() {
        assert!(Detection::new("Person", 0.5, BBox::new(0.0, 0.0, 0.1, 0.1)).is_person());
        assert!(Detection::new("CAR", 0.5, BBox::new(0.0, 0.0, 0.1, 0.1)).is_vehicle());
        assert!(Detection::new("motorbike", 0.5, BBox::new(0.0, 0.0, 0.1, 0.1)).is_relevant_class());
        assert!(!Detection::new("dog", 0.5, BBox::new(0.0, 0.0, 0.1, 0.1)).is_relevant_class());
    }

    #[test]
    fn detector_url_default() {
        // Don't mutate process env in tests; just assert the default is returned
        // when DETECTOR_URL is unset is environment-dependent, so only check the
        // constant wiring.
        assert_eq!(DEFAULT_DETECTOR_URL, "http://127.0.0.1:8094/detect");
    }

    #[test]
    fn detection_roundtrips_json() {
        let d = det("person", 0.77, 0.5, 0.5);
        let s = serde_json::to_string(&d).unwrap();
        let back: Detection = serde_json::from_str(&s).unwrap();
        assert_eq!(d, back);
    }

    #[test]
    fn detect_response_parses() {
        let raw = r#"{"ok":true,"detections":[{"class":"person","confidence":0.9,"bbox":{"x":0.1,"y":0.1,"w":0.2,"h":0.3}}],"model":"yolox-nano"}"#;
        let resp: DetectResponse = serde_json::from_str(raw).unwrap();
        assert!(resp.ok);
        assert_eq!(resp.detections.len(), 1);
        assert_eq!(resp.detections[0].class, "person");
        assert_eq!(resp.model.as_deref(), Some("yolox-nano"));
    }
}
