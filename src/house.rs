use nuclear_consul::clients::house_consul_client::{
    HouseConsulClient, HouseConsulRequest, HouseDecision,
};
use anyhow::Result;

#[derive(Clone)]
pub struct House {
    pub client: HouseConsulClient,
}

impl House {
    pub fn new(base_url: String, bearer_token: String) -> Result<Self> {
        let client = HouseConsulClient::new(base_url, 5000, Some(bearer_token))?;
        Ok(Self { client })
    }

    pub async fn deliberate(
        &self,
        query: &str,
        context: Option<&str>,
    ) -> HouseDecision {
        let req = HouseConsulRequest {
            query: query.to_string(),
            context: context.map(|s| s.to_string()),
            require_ethics: true,
            require_security: true,
            max_output_tokens: 220,
        };
        self.client.decide(req).await
    }
}
