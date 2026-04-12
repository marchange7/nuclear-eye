// face_embedding.rs — HTTP client for the ArcFace embedding sidecar (Track O4).
//
// face_db calls this module to:
//   1. Obtain a 512-dim embedding from a base64 image  (`embed`)
//   2. Compare two embeddings via cosine similarity    (`cosine_similarity`)
//
// The Python sidecar (face_embedding_service.py) must be running at
// ARCFACE_URL (default http://127.0.0.1:5555, env: FACE_EMBED_URL) for biometric matching.
// If the sidecar is unreachable, all functions return `Err`.
// face_db returns "unknown" — no text-hint fallback, no silent degradation (Z10).

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};

/// Default URL of the face_embedding_service.py sidecar.
/// Configurable via ARCFACE_URL env (canonical) or FACE_EMBED_URL (legacy alias).
const DEFAULT_EMBED_URL: &str = "http://127.0.0.1:5555";

/// Read timeout for embedding calls (ms). Embedding is CPU-bound ~50-150ms.
const EMBED_TIMEOUT_MS: u64 = 2000;

fn embed_url() -> String {
    std::env::var("ARCFACE_URL")
        .or_else(|_| std::env::var("FACE_EMBED_URL"))
        .unwrap_or_else(|_| DEFAULT_EMBED_URL.to_string())
}

// ── Wire types ────────────────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
struct EmbedRequest<'a> {
    image_b64: &'a str,
    max_faces: u32,
}

#[derive(Debug, Deserialize)]
pub struct EmbedResponse {
    pub ok: bool,
    pub faces_detected: usize,
    /// 512-dim vectors, one per face (sorted by detection confidence).
    pub embeddings: Vec<Vec<f32>>,
    pub detection_scores: Vec<f32>,
}

#[derive(Debug, Serialize)]
struct CompareRequest<'a> {
    image_b64_a: &'a str,
    image_b64_b: &'a str,
}

#[derive(Debug, Deserialize)]
pub struct CompareResponse {
    pub ok: bool,
    /// Cosine similarity in [−1, 1].  Above ~0.28 = same person.
    pub similarity: f32,
    pub same_person: bool,
    pub threshold: f32,
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Extract ArcFace embeddings from a base64-encoded JPEG/PNG image.
///
/// Returns the first (highest-confidence) 512-dim vector on success.
/// Returns `Err` when the sidecar is unreachable or detects no faces.
pub async fn embed(client: &reqwest::Client, image_b64: &str) -> Result<Vec<f32>> {
    let url = format!("{}/embed", embed_url());
    let body = EmbedRequest { image_b64, max_faces: 1 };

    let resp = client
        .post(&url)
        .json(&body)
        .timeout(std::time::Duration::from_millis(EMBED_TIMEOUT_MS))
        .send()
        .await?
        .error_for_status()?
        .json::<EmbedResponse>()
        .await?;

    if !resp.ok || resp.embeddings.is_empty() {
        bail!("no face detected in image (faces_detected={})", resp.faces_detected);
    }

    Ok(resp.embeddings.into_iter().next().unwrap())
}

/// Compare two base64 images and return cosine similarity + same_person flag.
pub async fn compare_images(
    client: &reqwest::Client,
    image_b64_a: &str,
    image_b64_b: &str,
) -> Result<CompareResponse> {
    let url = format!("{}/compare", embed_url());
    let body = CompareRequest { image_b64_a, image_b64_b };

    let resp = client
        .post(&url)
        .json(&body)
        .timeout(std::time::Duration::from_millis(EMBED_TIMEOUT_MS * 2))
        .send()
        .await?
        .error_for_status()?
        .json::<CompareResponse>()
        .await?;

    Ok(resp)
}

/// Cosine similarity between two 512-dim embedding vectors.
///
/// Returns a value in [−1, 1].  Typical ArcFace threshold: 0.28.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a < 1e-8 || norm_b < 1e-8 {
        return 0.0;
    }
    (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
}

// ── Serialization helpers ─────────────────────────────────────────────────────

/// Serialize a 512-dim f32 vector to little-endian bytes for BLOB storage.
pub fn embedding_to_bytes(embedding: &[f32]) -> Vec<u8> {
    embedding.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Deserialize a BLOB back to a 512-dim f32 vector.
///
/// Returns `None` if the byte slice length is not a multiple of 4.
pub fn bytes_to_embedding(bytes: &[u8]) -> Option<Vec<f32>> {
    if !bytes.len().is_multiple_of(4) {
        return None;
    }
    Some(
        bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),
    )
}
