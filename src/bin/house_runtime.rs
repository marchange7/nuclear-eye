use anyhow::Result;
use nuclear_eye::runtime::HouseRuntime;
use nuclear_consul::runtime::types::SensitiveActionKind;
use std::env;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    // S-7: fail-closed wrapper probe
    nuclear_eye::wrapper_guard::check_wrapper("house-runtime").await?;

    // ── Nuclear wrapper — resilience sidecar ────────────────────────────
    match nuclear_wrapper::wrap!(
        node_id      = "house-runtime",
        pg_url       = std::env::var("DATABASE_URL").unwrap_or_default(),
        signal_token = std::env::var("SIGNAL_TOKEN").unwrap_or_default()
    ) {
        Ok(nw) => {
            tracing::info!("nuclear-wrapper: armed (tamper, health, discovery)");
            std::mem::forget(nw);
        }
        Err(e) => tracing::info!("nuclear-wrapper: start failed ({e}) — running unguarded"),
    }

    let fortress_base_url =
        env::var("FORTRESS_BASE_URL").unwrap_or_else(|_| "http://127.0.0.1:7710".to_string());
    let identity_base_url =
        env::var("IDENTITY_BASE_URL").unwrap_or_else(|_| "http://127.0.0.1:7720".to_string());
    let jwt_token = env::var("HOUSE_JWT_TOKEN").expect("HOUSE_JWT_TOKEN required");

    let rt = HouseRuntime::new(&fortress_base_url, &identity_base_url, &jwt_token)?;

    let motion_query = "Motion détecté arrière (confiance: 0.61). House doit-elle escalader?";
    let context =
        Some("zone:rear_entry signal:motion timestamp:2026-03-27T21:38 confidence:0.61");

    println!("🛡️  House Runtime — Motion Alert");
    println!("Query: {motion_query}");

    // 1. Simple deliberation (no identity)
    let simple = rt.deliberate(motion_query, context).await;
    println!("\n📋 Simple House: {simple}");

    // 2. Protected with strong identity
    let strong = rt
        .protected_action(
            SensitiveActionKind::AlarmEscalation,
            motion_query,
            context,
            Some(("andrzej", 0.91, 0.95, true, 0.82, 0.90, true)),
        )
        .await;
    println!("\n🔒 Protected (strong ID): {strong}");

    // 3. Weak identity → conservative fallback
    let weak = rt
        .protected_action(
            SensitiveActionKind::AlarmEscalation,
            motion_query,
            context,
            Some(("unknown", 0.42, 0.30, true, 0.0, 0.0, false)),
        )
        .await;
    println!("\n⚠️  Protected (weak ID): {weak}");

    println!("\n✅ House Runtime healthy");
    Ok(())
}
