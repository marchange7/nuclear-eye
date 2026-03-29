//! Integration test: start DecisionAgent on an ephemeral port, send real
//! HTTP requests, and verify the response contract.
//!
//! This test does NOT require an external config file — it uses a minimal
//! in-process axum router identical to the decision_agent binary.

use axum::{
    extract::State,
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use nuclear_eye::{decide, AffectTriad, VisionEvent};
use serde::{Deserialize, Serialize};

// ── Duplicated types (mirrors the binary — a shared crate would be cleaner) ──

#[derive(Clone)]
struct TestAppState {
    safety_risk_threshold: f64,
}

#[derive(Debug, Deserialize)]
struct DecideRequest {
    event: VisionEvent,
    #[serde(default)]
    force_safety: Option<bool>,
}

#[derive(Debug, Deserialize, Serialize)]
struct DecisionResponse {
    event_id: String,
    triad: AffectTriad,
    action: String,
    is_safety_critical: bool,
    dominant_dimension: String,
}

#[derive(Debug, Deserialize, Serialize)]
struct ErrorBody {
    error: String,
}

fn test_router() -> Router {
    let state = TestAppState {
        safety_risk_threshold: 0.5,
    };
    Router::new()
        .route("/decide", post(handle_decide))
        .route("/health", get(health))
        .with_state(state)
}

async fn health() -> &'static str {
    "ok"
}

async fn handle_decide(
    State(state): State<TestAppState>,
    payload: Result<Json<DecideRequest>, axum::extract::rejection::JsonRejection>,
) -> Result<Json<DecisionResponse>, (StatusCode, Json<ErrorBody>)> {
    let Json(req) = payload.map_err(|err| {
        (
            StatusCode::BAD_REQUEST,
            Json(ErrorBody {
                error: err.to_string(),
            }),
        )
    })?;

    let triad = AffectTriad::from_vision_event(&req.event);
    let is_safety_critical = req
        .force_safety
        .unwrap_or(req.event.risk_score > state.safety_risk_threshold);
    let action = decide(&triad, is_safety_critical);
    let dominant = triad.dominant().to_string();

    Ok(Json(DecisionResponse {
        event_id: req.event.event_id,
        triad,
        action: action.to_string(),
        is_safety_critical,
        dominant_dimension: dominant,
    }))
}

// ── Tests ──────────────────────────────────────────────────────────────

#[tokio::test]
async fn health_endpoint_returns_ok() {
    let app = test_router();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    let client = reqwest::Client::new();
    let resp = client
        .get(format!("http://{addr}/health"))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    assert_eq!(resp.text().await.unwrap(), "ok");
}

#[tokio::test]
async fn decide_endpoint_returns_valid_response() {
    let app = test_router();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    let client = reqwest::Client::new();
    let body = serde_json::json!({
        "event": {
            "event_id": "integ-test-001",
            "timestamp_ms": 1711497600000_u64,
            "camera_id": "test-cam",
            "behavior": "loitering",
            "risk_score": 0.72,
            "stress_level": 0.65,
            "confidence": 0.80,
            "person_detected": true,
            "person_name": null,
            "hands_visible": 2,
            "object_held": null,
            "extra_tags": [],
            "vlm_caption": null
        }
    });

    let resp = client
        .post(format!("http://{addr}/decide"))
        .json(&body)
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);

    let result: DecisionResponse = resp.json().await.unwrap();
    assert_eq!(result.event_id, "integ-test-001");
    assert!(result.is_safety_critical, "risk 0.72 > threshold 0.5");
    assert!(result.triad.judgement >= 0.0 && result.triad.judgement <= 1.0);
    assert!(result.triad.doubt >= 0.0 && result.triad.doubt <= 1.0);
    assert!(result.triad.determination >= 0.0 && result.triad.determination <= 1.0);
    assert!(
        ["none", "alarm", "pause"].contains(&result.action.as_str()),
        "safety-critical action must be none/alarm/pause, got: {}",
        result.action
    );
    assert!(
        ["judgement", "doubt", "determination"].contains(&result.dominant_dimension.as_str()),
        "unexpected dominant: {}",
        result.dominant_dimension
    );
}

#[tokio::test]
async fn decide_endpoint_force_relational() {
    let app = test_router();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    let client = reqwest::Client::new();
    let body = serde_json::json!({
        "event": {
            "event_id": "integ-test-002",
            "timestamp_ms": 1711497600000_u64,
            "camera_id": "test-cam",
            "behavior": "loitering",
            "risk_score": 0.72,
            "stress_level": 0.65,
            "confidence": 0.80,
            "person_detected": true,
            "person_name": null,
            "hands_visible": 2,
            "object_held": null,
            "extra_tags": [],
            "vlm_caption": null
        },
        "force_safety": false
    });

    let resp = client
        .post(format!("http://{addr}/decide"))
        .json(&body)
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let result: DecisionResponse = resp.json().await.unwrap();
    assert!(!result.is_safety_critical, "force_safety=false should override");
}

#[tokio::test]
async fn decide_endpoint_bad_json_returns_400() {
    let app = test_router();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("http://{addr}/decide"))
        .header("Content-Type", "application/json")
        .body(r#"{"not_event": true}"#)
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 400);
    let err: ErrorBody = resp.json().await.unwrap();
    assert!(!err.error.is_empty());
}

#[tokio::test]
async fn decide_calm_event_is_not_safety_critical() {
    let app = test_router();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    let client = reqwest::Client::new();
    let body = serde_json::json!({
        "event": {
            "event_id": "integ-test-003",
            "timestamp_ms": 0,
            "camera_id": "c0",
            "behavior": "passby",
            "risk_score": 0.15,
            "stress_level": 0.10,
            "confidence": 0.95,
            "person_detected": true,
            "person_name": "known-resident",
            "hands_visible": 2,
            "object_held": null,
            "extra_tags": [],
            "vlm_caption": null
        }
    });

    let resp = client
        .post(format!("http://{addr}/decide"))
        .json(&body)
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let result: DecisionResponse = resp.json().await.unwrap();
    assert!(!result.is_safety_critical, "risk 0.15 < 0.5");
    // Calm scene: high judgement, low doubt → likely Support
    assert!(result.triad.judgement > 0.7);
    assert!(result.triad.doubt < 0.2);
}
