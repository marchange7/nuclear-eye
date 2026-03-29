use crate::guard::HouseGuard;
use crate::house::House;
use anyhow::Result;
use nuclear_consul::runtime::types::SensitiveActionKind;

pub struct HouseRuntime {
    pub house: House,
    pub guard: HouseGuard,
}

impl HouseRuntime {
    pub fn new(
        fortress_base_url: &str,
        identity_base_url: &str,
        jwt_token: &str,
    ) -> Result<Self> {
        let house = House::new(fortress_base_url.to_string(), jwt_token.to_string())?;
        let guard = HouseGuard::new(
            identity_base_url.to_string(),
            fortress_base_url.to_string(),
            jwt_token.to_string(),
        )?;
        Ok(Self { house, guard })
    }

    /// Quick deliberation — no identity check
    pub async fn deliberate(&self, query: &str, context: Option<&str>) -> String {
        let resp = self.house.deliberate(query, context).await;
        resp.decision
    }

    /// Protected action — identity verification + consul deliberation
    pub async fn protected_action(
        &self,
        action: SensitiveActionKind,
        query: &str,
        context: Option<&str>,
        identity_data: Option<(&str, f32, f32, bool, f32, f32, bool)>,
    ) -> String {
        self.guard
            .simple_decision(action, query, context, identity_data)
            .await
    }
}
