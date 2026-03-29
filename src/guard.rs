use nuclear_consul::{
    clients::house_consul_client::{HouseConsulClient, HouseConsulRequest},
    clients::identity_client::{IdentityClient, IdentityScoreRequest},
    identity::types::{FaceMatchInput, TokenMatchInput, VoiceMatchInput},
    runtime::house_guard::{
        HouseProtectedActionRequest, HouseProtectedActionResult, HouseRuntimeGuard,
        HouseRuntimeGuardConfig,
    },
};
use nuclear_consul::runtime::types::SensitiveActionKind;
use anyhow::Result;

#[derive(Clone)]
pub struct HouseGuard {
    pub guard: HouseRuntimeGuard,
}

impl HouseGuard {
    pub fn new(
        identity_base_url: String,
        consul_base_url: String,
        jwt_token: String,
    ) -> Result<Self> {
        let identity_client =
            IdentityClient::new(identity_base_url, 3000, Some(jwt_token.clone()))?;
        let consul_client =
            HouseConsulClient::new(consul_base_url, 5000, Some(jwt_token))?;
        let guard = HouseRuntimeGuard::new(
            identity_client,
            consul_client,
            HouseRuntimeGuardConfig::default(),
        );
        Ok(Self { guard })
    }

    pub async fn protect(&self, req: HouseProtectedActionRequest) -> HouseProtectedActionResult {
        self.guard.evaluate(req).await
    }

    pub async fn simple_decision(
        &self,
        action: SensitiveActionKind,
        query: &str,
        context: Option<&str>,
        identity_data: Option<(&str, f32, f32, bool, f32, f32, bool)>,
    ) -> String {
        let mut id_req = IdentityScoreRequest {
            expected_identity_id: "andrzej".to_string(),
            face: None,
            voice: None,
            token: Some(TokenMatchInput {
                entity: "house".to_string(),
                subject: "house-runtime".to_string(),
                session_id: "sess-001".to_string(),
                valid: true,
                scope_ok: true,
            }),
            weights: None,
        };

        if let Some((fid, fc, fl, fok, vc, vq, vok)) = identity_data {
            id_req.face = Some(FaceMatchInput {
                enrolled_identity_id: fid.to_string(),
                confidence: fc,
                liveness_score: fl,
                camera_ok: fok,
            });
            id_req.voice = Some(VoiceMatchInput {
                enrolled_identity_id: fid.to_string(),
                confidence: vc,
                sample_quality: vq,
                microphone_ok: vok,
            });
        }

        let guard_req = HouseProtectedActionRequest {
            action_name: format!("{:?}", action),
            kind: action,
            context: context.map(|s| s.to_string()),
            local_confidence: 0.75,
            ambiguity_score: 0.22,
            identity_request: Some(id_req),
            consul_request: HouseConsulRequest {
                query: query.to_string(),
                context: context.map(|s| s.to_string()),
                require_ethics: true,
                require_security: true,
                max_output_tokens: 220,
            },
        };

        let result = self.guard.evaluate(guard_req).await;
        result.decision
    }
}
