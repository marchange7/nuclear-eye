use anyhow::Result;
use clap::Parser;
use nuclear_eye::types::{
    HouseConsulRequest, HouseProtectedActionRequest, HouseRuntimeGuard, HouseRuntimeGuardConfig,
    SensitiveActionKind,
};
use nuclear_consul::clients::house_consul_client::HouseConsulClient;
use nuclear_consul::clients::identity_client::IdentityClient;

#[derive(Parser, Debug)]
#[command(name = "house-sentinel", about = "House sentinel guard — evaluate protected actions via nuclear-consul")]
struct Args {
    #[arg(long, env = "IDENTITY_BASE_URL", default_value = "http://127.0.0.1:7720")]
    identity_url: String,

    #[arg(long, env = "FORTRESS_BASE_URL", default_value = "http://127.0.0.1:7710")]
    consul_url: String,

    #[arg(long, env = "HOUSE_JWT_TOKEN")]
    jwt_token: String,

    #[arg(long, help = "Human-readable action label (e.g. motion-alert)")]
    action: String,

    #[arg(long, help = "Optional context string for deliberation")]
    context: Option<String>,

    #[arg(long, default_value = "alarm_escalation", help = "Action kind: observe_only | alarm_escalation | unlock_operation | …")]
    kind: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    // S-7: fail-closed wrapper probe
    nuclear_eye::wrapper_guard::check_wrapper("house-sentinel").await?;

    // ── Nuclear wrapper — resilience sidecar ────────────────────────────
    match nuclear_wrapper::wrap!(
        node_id      = "house-sentinel",
        pg_url       = std::env::var("DATABASE_URL").unwrap_or_default(),
        signal_token = std::env::var("SIGNAL_TOKEN").unwrap_or_default()
    ) {
        Ok(nw) => {
            tracing::info!("nuclear-wrapper: armed (tamper, health, discovery)");
            std::mem::forget(nw);
        }
        Err(e) => tracing::info!("nuclear-wrapper: start failed ({e}) — running unguarded"),
    }

    let args = Args::parse();

    let kind = parse_kind(&args.kind);

    let identity = IdentityClient::new(&args.identity_url, 5000, Some(args.jwt_token.clone()))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let consul = HouseConsulClient::new(&args.consul_url, 10000, Some(args.jwt_token))
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    let guard = HouseRuntimeGuard::new(identity, consul, HouseRuntimeGuardConfig::default());

    let query = format!(
        "Action '{}' demandée. House doit-elle autoriser?",
        args.action
    );

    let req = HouseProtectedActionRequest {
        action_name: args.action.clone(),
        kind,
        context: args.context.clone(),
        local_confidence: 0.92,
        ambiguity_score: 0.15,
        identity_request: None, // Multimodal input injected by upstream caller
        consul_request: HouseConsulRequest {
            query,
            context: args.context,
            require_ethics: true,
            require_security: true,
            max_output_tokens: 220,
        },
    };

    let result = guard.evaluate(req).await;

    println!("route:             {:?}", result.route);
    println!("allow_action:      {}", result.allow_action);
    println!("should_notify:     {}", result.should_notify);
    println!("should_escalate:   {}", result.should_escalate_human);
    println!("decision:          {}", result.decision);
    println!("action:            {}", result.action);
    println!("reason:            {}", result.reason);
    println!("identity.verified: {} (score={:.2})", result.identity.verified, result.identity.score);

    Ok(())
}

fn parse_kind(s: &str) -> SensitiveActionKind {
    match s {
        "observe_only" => SensitiveActionKind::ObserveOnly,
        "unlock_operation" => SensitiveActionKind::UnlockOperation,
        "voice_enrollment" => SensitiveActionKind::VoiceEnrollment,
        "face_enrollment" => SensitiveActionKind::FaceEnrollment,
        "external_message" => SensitiveActionKind::ExternalMessage,
        "relationship_mutation" => SensitiveActionKind::RelationshipMutation,
        "memory_export" => SensitiveActionKind::MemoryExport,
        "preference_overwrite" => SensitiveActionKind::PreferenceOverwrite,
        "account_link" => SensitiveActionKind::AccountLink,
        _ => SensitiveActionKind::AlarmEscalation,
    }
}
