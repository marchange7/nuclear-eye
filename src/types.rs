// Re-exports of nuclear-consul types used throughout house-runtime.
// Import from here to avoid depending on nuclear-consul directly in application code.

pub use nuclear_consul::clients::house_consul_client::{
    DecisionSource, HouseConsulClient, HouseConsulRequest, HouseDecision, QueryMode,
};
pub use nuclear_consul::clients::identity_client::{
    IdentityClient, IdentityScoreRequest,
};
pub use nuclear_consul::identity::types::{
    FaceMatchInput, IdentityFusionResult, IdentityWeights, TokenMatchInput, VoiceMatchInput,
};
pub use nuclear_consul::runtime::house_guard::{
    HouseActionRoute, HouseProtectedActionRequest, HouseProtectedActionResult,
    HouseRuntimeGuard, HouseRuntimeGuardConfig,
};
pub use nuclear_consul::runtime::types::{IdentityGateResult, SensitiveActionKind};
