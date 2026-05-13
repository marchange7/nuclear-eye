// SEN-12: optional PG persistence for alarm verdicts.
//
// Opt-in via env vars:
//   SENTINELLE_PERSIST_ALARMS=1  -- enable; default off
//   DATABASE_URL=postgresql://...  -- Postgres DSN
//   FORTRESS_TENANT_ID=<uuid>      -- reused from Fortress mesh publish
//
// Design: fire-and-forget tokio::spawn; never blocks or panics the alarm path.
// Pool init is lazy (first call) and fail-soft (None on any error).
// RLS: BEGIN + SET LOCAL app.tenant_id + INSERT + COMMIT per alarm.

use sqlx::postgres::PgPoolOptions;
use sqlx::PgPool;
use std::time::Duration;
use tokio::sync::OnceCell;
use tracing::{info, warn};
use uuid::Uuid;

use nuclear_eye::{AlarmEvent, AlarmLevel};
use nuclear_eye::alert::AlertSeverity;

static PG_POOL: OnceCell<Option<PgPool>> = OnceCell::const_new();

const INSERT_SQL: &str = r#"
    INSERT INTO sentinelle_alarms
        (tenant_id, camera_id, level, score, reason, severity, triggered_at, created_by)
    VALUES
        ($1, $2, $3, $4, $5, $6, $7, $8)
    RETURNING id
"#;

/// Initialise the pool once at startup (called from main when
/// SENTINELLE_PERSIST_ALARMS=1). Logs the result; never panics.
pub async fn init_pool() -> Option<&'static PgPool> {
    get_pool().await
}

/// Lazy singleton: returns Some(&pool) on success, None if disabled/failed.
pub async fn get_pool() -> Option<&'static PgPool> {
    PG_POOL
        .get_or_init(|| async {
            let enabled = std::env::var("SENTINELLE_PERSIST_ALARMS")
                .map(|v| v.trim() == "1" || v.trim().eq_ignore_ascii_case("true"))
                .unwrap_or(false);
            if !enabled {
                return None;
            }

            let url = match std::env::var("DATABASE_URL") {
                Ok(u) if !u.trim().is_empty() => u,
                _ => {
                    warn!("SEN-12: SENTINELLE_PERSIST_ALARMS=1 but DATABASE_URL not set; persistence disabled");
                    return None;
                }
            };

            match PgPoolOptions::new()
                .max_connections(4)
                .acquire_timeout(Duration::from_millis(300))
                .connect(&url)
                .await
            {
                Ok(pool) => {
                    info!("SEN-12: PG pool initialised for alarm persistence");
                    Some(pool)
                }
                Err(e) => {
                    warn!(error = %e, "SEN-12: PG pool init failed; alarm persistence disabled (non-blocking)");
                    None
                }
            }
        })
        .await
        .as_ref()
}

/// Insert one alarm verdict row. Called inside a fire-and-forget tokio::spawn.
/// Uses BEGIN + SET LOCAL app.tenant_id + INSERT + COMMIT for RLS compliance.
pub async fn insert_alarm(pool: &PgPool, alarm: &AlarmEvent) -> anyhow::Result<Uuid> {
    // Resolve tenant_id (nil UUID if FORTRESS_TENANT_ID unset, matching
    // existing Fortress mesh publish pattern).
    let tenant_id_str = std::env::var("FORTRESS_TENANT_ID").unwrap_or_default();
    let tenant_id_str = tenant_id_str.trim();
    let tenant_id_str = if tenant_id_str.is_empty() {
        "00000000-0000-0000-0000-000000000000"
    } else {
        tenant_id_str
    };
    let tenant_id: Uuid = tenant_id_str.parse().unwrap_or_else(|_| {
        warn!("SEN-12: FORTRESS_TENANT_ID is not a valid UUID, using nil UUID");
        Uuid::nil()
    });

    let severity = map_severity_str(&alarm.level, alarm.danger_score);
    let level_str = alarm.level.to_string();
    let reason = if alarm.note.is_empty() {
        alarm.vlm_caption.clone().unwrap_or_default()
    } else {
        alarm.note.clone()
    };

    // timestamp_ms -> time::OffsetDateTime (sqlx uses the `time` crate feature, not chrono)
    let triggered_at = sqlx::types::time::OffsetDateTime::from_unix_timestamp_nanos(
        (alarm.timestamp_ms as i128) * 1_000_000,
    )
    .unwrap_or_else(|_| sqlx::types::time::OffsetDateTime::now_utc());

    let mut tx = pool.begin().await?;

    // RLS: set app.tenant_id for this transaction so kernel.tenant_check passes.
    sqlx::query("SET LOCAL app.tenant_id = $1")
        .bind(tenant_id_str)
        .execute(&mut *tx)
        .await?;

    // TODO(SEN-12.1): resolve camera_id UUID from sentinelle_cameras WHERE
    //   pairing_string = alarm.camera_id. Requires one SELECT per alarm on first
    //   encounter; consider an in-process LRU cache (16 entries) to avoid per-alarm
    //   roundtrips under chain-storm conditions.
    let camera_id: Option<Uuid> = None;

    let row: (Uuid,) = sqlx::query_as(INSERT_SQL)
        .bind(tenant_id)
        .bind(camera_id)
        .bind(&level_str)
        .bind(alarm.danger_score)
        .bind(&reason)
        .bind(severity)
        .bind(triggered_at)
        .bind("alarm_grader_agent")
        .fetch_one(&mut *tx)
        .await?;

    tx.commit().await?;

    Ok(row.0)
}

/// Map AlarmLevel + danger_score to the CHECK-constrained severity column.
/// Must be exhaustive: CHECK (severity IN ('low','medium','high','critical')).
fn map_severity_str(level: &AlarmLevel, score: f64) -> &'static str {
    match map_alert_severity(level, score) {
        AlertSeverity::Info | AlertSeverity::Low => "low",
        AlertSeverity::Medium => "medium",
        AlertSeverity::High => "high",
        AlertSeverity::Critical => "critical",
    }
}

/// Mirrors alarm_grader_agent::map_severity so the mapping stays consistent.
fn map_alert_severity(level: &AlarmLevel, score: f64) -> AlertSeverity {
    match level {
        AlarmLevel::None => AlertSeverity::Info,
        AlarmLevel::Low => AlertSeverity::Low,
        AlarmLevel::Medium => AlertSeverity::Medium,
        AlarmLevel::High if score >= 0.85 => AlertSeverity::Critical,
        AlarmLevel::High => AlertSeverity::High,
    }
}
