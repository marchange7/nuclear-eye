use rusqlite::{Connection, Result, params};

pub struct SecurityMemory {
    conn: Connection,
}

impl SecurityMemory {
    pub fn open(path: &str) -> Result<Self> {
        let conn = Connection::open(path)?;
        conn.execute_batch("
            CREATE TABLE IF NOT EXISTS alarm_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp_ms INTEGER NOT NULL,
                level TEXT NOT NULL,
                danger_score REAL,
                note TEXT,
                decision TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS vision_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp_ms INTEGER NOT NULL,
                behavior TEXT,
                risk_score REAL,
                person_detected INTEGER,
                person_name TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS false_alarm_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                alarm_id TEXT,
                confirmed_false INTEGER DEFAULT 0,
                note TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS decision_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp_ms INTEGER NOT NULL,
                event_id TEXT NOT NULL,
                camera_id TEXT NOT NULL,
                action TEXT NOT NULL,
                is_safety_critical INTEGER NOT NULL,
                dominant_dimension TEXT NOT NULL,
                consul_synthesis TEXT,
                consul_confidence REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            -- Offline buffer: VisionEvents that failed to reach alarm_grader_agent
            CREATE TABLE IF NOT EXISTS event_buffer (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_json TEXT NOT NULL,
                target_url TEXT NOT NULL,
                queued_at INTEGER NOT NULL,
                attempts INTEGER DEFAULT 0
            );
        ")?;
        Ok(Self { conn })
    }

    pub fn record_alarm(&self, timestamp_ms: u64, level: &str, danger_score: f64, note: Option<&str>, decision: &str) -> Result<()> {
        self.conn.execute(
            "INSERT INTO alarm_history (timestamp_ms, level, danger_score, note, decision) VALUES (?1, ?2, ?3, ?4, ?5)",
            params![timestamp_ms as i64, level, danger_score, note, decision],
        )?;
        Ok(())
    }

    pub fn record_vision(&self, timestamp_ms: u64, behavior: &str, risk_score: f64, person_detected: bool, person_name: Option<&str>) -> Result<()> {
        self.conn.execute(
            "INSERT INTO vision_history (timestamp_ms, behavior, risk_score, person_detected, person_name) VALUES (?1, ?2, ?3, ?4, ?5)",
            params![timestamp_ms as i64, behavior, risk_score, person_detected as i32, person_name],
        )?;
        Ok(())
    }

    pub fn record_decision(
        &self,
        timestamp_ms: u64,
        event_id: &str,
        camera_id: &str,
        action: &str,
        is_safety_critical: bool,
        dominant_dimension: &str,
        consul_synthesis: Option<&str>,
        consul_confidence: Option<f64>,
    ) -> Result<()> {
        self.conn.execute(
            "INSERT INTO decision_log (timestamp_ms, event_id, camera_id, action, is_safety_critical, dominant_dimension, consul_synthesis, consul_confidence)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                timestamp_ms as i64,
                event_id,
                camera_id,
                action,
                is_safety_critical as i32,
                dominant_dimension,
                consul_synthesis,
                consul_confidence,
            ],
        )?;
        Ok(())
    }

    pub fn recent_alarms(&self, limit: u32) -> Result<Vec<(String, f64, String)>> {
        let mut stmt = self.conn.prepare(
            "SELECT level, danger_score, decision FROM alarm_history ORDER BY timestamp_ms DESC LIMIT ?1"
        )?;
        let rows = stmt.query_map(params![limit], |row| {
            Ok((row.get(0)?, row.get(1)?, row.get(2)?))
        })?.collect::<Result<Vec<_>>>()?;
        Ok(rows)
    }

    pub fn false_alarm_count_last_hour(&self) -> Result<u32> {
        self.conn.query_row(
            "SELECT COUNT(*) FROM alarm_history WHERE level != 'none' AND created_at > datetime('now', '-1 hour')",
            [],
            |r| r.get::<_, u32>(0),
        )
    }

    pub fn decision_count(&self) -> Result<u64> {
        self.conn.query_row("SELECT COUNT(*) FROM decision_log", [], |r| r.get(0))
    }

    // ── Offline event buffer ────────────────────────────────────────────

    pub fn buffer_event(&self, event_json: &str, target_url: &str, now_ms: u64) -> Result<()> {
        self.conn.execute(
            "INSERT INTO event_buffer (event_json, target_url, queued_at) VALUES (?1, ?2, ?3)",
            params![event_json, target_url, now_ms as i64],
        )?;
        Ok(())
    }

    /// Return up to `limit` oldest buffered events: (id, json, url, attempts).
    pub fn pending_events(&self, limit: u32) -> Result<Vec<(i64, String, String, i32)>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, event_json, target_url, attempts FROM event_buffer ORDER BY queued_at ASC LIMIT ?1"
        )?;
        let rows = stmt.query_map(params![limit], |row| {
            Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?))
        })?.collect::<Result<Vec<_>>>()?;
        Ok(rows)
    }

    /// Return up to `limit` oldest buffered events for one target URL.
    pub fn pending_events_for_target(
        &self,
        target_url: &str,
        limit: u32,
    ) -> Result<Vec<(i64, String, String, i32)>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, event_json, target_url, attempts
             FROM event_buffer
             WHERE target_url = ?1
             ORDER BY queued_at ASC
             LIMIT ?2",
        )?;
        let rows = stmt.query_map(params![target_url, limit], |row| {
            Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?))
        })?.collect::<Result<Vec<_>>>()?;
        Ok(rows)
    }

    pub fn delete_buffered_event(&self, id: i64) -> Result<()> {
        self.conn.execute("DELETE FROM event_buffer WHERE id = ?1", params![id])?;
        Ok(())
    }

    pub fn increment_buffer_attempts(&self, id: i64) -> Result<()> {
        self.conn.execute(
            "UPDATE event_buffer SET attempts = attempts + 1 WHERE id = ?1",
            params![id],
        )?;
        Ok(())
    }

    /// Drop events that have failed more than `max_attempts` times.
    pub fn prune_dead_events(&self, max_attempts: i32) -> Result<usize> {
        let n = self.conn.execute(
            "DELETE FROM event_buffer WHERE attempts >= ?1",
            params![max_attempts],
        )?;
        Ok(n)
    }

    pub fn buffered_count(&self) -> Result<u32> {
        self.conn.query_row("SELECT COUNT(*) FROM event_buffer", [], |r| r.get(0))
    }
}
