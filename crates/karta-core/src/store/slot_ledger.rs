use std::sync::{Arc, Mutex};

use chrono::{DateTime, Utc};
use rusqlite::{Connection, Row};

use crate::error::{KartaError, Result};

/// Append-only ledger of mutable-slot values (entity_key/predicate/value)
/// with deterministic supersession over time.
pub struct SlotLedger {
    conn: Arc<Mutex<Connection>>,
}

/// A single row read from the `slot_ledger` table.
#[derive(Debug, Clone, PartialEq)]
pub struct LedgerRow {
    pub id: i64,
    pub entity_key: String,
    pub predicate: String,
    pub value: String,
    pub value_norm: String,
    pub seq: u64,
    pub valid_from: Option<DateTime<Utc>>,
    pub mention_time: DateTime<Utc>,
    pub valid_to: Option<DateTime<Utc>>,
    pub superseded_by: Option<i64>,
    pub conflict_group: Option<i64>,
    pub note_id: String,
    pub source_span: String,
}

/// Fields required to open a new ledger row.
pub struct NewLedgerRow {
    pub entity_key: String,
    pub predicate: String,
    pub value: String,
    pub value_norm: String,
    pub seq: u64,
    pub valid_from: Option<DateTime<Utc>>,
    pub mention_time: DateTime<Utc>,
    pub note_id: String,
    pub source_span: String,
}

fn parse_rfc3339(s: &str) -> Result<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(s)
        .map(|d| d.with_timezone(&Utc))
        .map_err(|e| KartaError::VectorStore(format!("invalid rfc3339 timestamp: {e}")))
}

fn parse_rfc3339_opt(s: Option<String>) -> Result<Option<DateTime<Utc>>> {
    match s {
        Some(s) => Ok(Some(parse_rfc3339(&s)?)),
        None => Ok(None),
    }
}

fn row_to_ledger_row(row: &Row) -> rusqlite::Result<(
    i64,
    String,
    String,
    String,
    String,
    i64,
    Option<String>,
    String,
    Option<String>,
    Option<i64>,
    Option<i64>,
    String,
    String,
)> {
    Ok((
        row.get("id")?,
        row.get("entity_key")?,
        row.get("predicate")?,
        row.get("value")?,
        row.get("value_norm")?,
        row.get("seq")?,
        row.get("valid_from")?,
        row.get("mention_time")?,
        row.get("valid_to")?,
        row.get("superseded_by")?,
        row.get("conflict_group")?,
        row.get("note_id")?,
        row.get("source_span")?,
    ))
}

impl SlotLedger {
    pub fn new(conn: Arc<Mutex<Connection>>) -> Result<Self> {
        let ledger = SlotLedger { conn };
        ledger.init_schema()?;
        Ok(ledger)
    }

    fn init_schema(&self) -> Result<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| KartaError::VectorStore(e.to_string()))?;
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS slot_ledger (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_key    TEXT NOT NULL,
                predicate     TEXT NOT NULL,
                value         TEXT NOT NULL,
                value_norm    TEXT NOT NULL,
                seq           INTEGER NOT NULL,
                valid_from    TEXT,
                mention_time  TEXT NOT NULL,
                valid_to      TEXT,
                superseded_by INTEGER,
                conflict_group INTEGER,
                note_id       TEXT NOT NULL,
                source_span   TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_ledger_current
                ON slot_ledger(entity_key, predicate, valid_to);
            CREATE INDEX IF NOT EXISTS idx_ledger_history
                ON slot_ledger(entity_key, predicate, seq);",
        )
        .map_err(|e| KartaError::VectorStore(e.to_string()))?;
        Ok(())
    }

    pub fn insert_open(&self, row: &NewLedgerRow) -> Result<i64> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| KartaError::VectorStore(e.to_string()))?;
        let valid_from = row.valid_from.map(|d| d.to_rfc3339());
        let mention_time = row.mention_time.to_rfc3339();
        let id: i64 = conn
            .query_row(
                "INSERT INTO slot_ledger
                    (entity_key, predicate, value, value_norm, seq, valid_from, mention_time, note_id, source_span)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
                 RETURNING id",
                rusqlite::params![
                    row.entity_key,
                    row.predicate,
                    row.value,
                    row.value_norm,
                    row.seq as i64,
                    valid_from,
                    mention_time,
                    row.note_id,
                    row.source_span,
                ],
                |r| r.get(0),
            )
            .map_err(|e| KartaError::VectorStore(e.to_string()))?;
        Ok(id)
    }

    pub fn current(&self, entity_key: &str, predicate: &str) -> Result<Vec<LedgerRow>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| KartaError::VectorStore(e.to_string()))?;
        let mut stmt = conn
            .prepare(
                "SELECT id, entity_key, predicate, value, value_norm, seq, valid_from,
                        mention_time, valid_to, superseded_by, conflict_group, note_id, source_span
                 FROM slot_ledger
                 WHERE entity_key = ?1 AND predicate = ?2 AND valid_to IS NULL
                 ORDER BY seq",
            )
            .map_err(|e| KartaError::VectorStore(e.to_string()))?;
        let rows = stmt
            .query_map(rusqlite::params![entity_key, predicate], row_to_ledger_row)
            .map_err(|e| KartaError::VectorStore(e.to_string()))?;
        let mut out = Vec::new();
        for r in rows {
            let tuple = r.map_err(|e| KartaError::VectorStore(e.to_string()))?;
            out.push(tuple_to_row(tuple)?);
        }
        Ok(out)
    }

    pub fn history(&self, entity_key: &str, predicate: &str) -> Result<Vec<LedgerRow>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| KartaError::VectorStore(e.to_string()))?;
        let mut stmt = conn
            .prepare(
                "SELECT id, entity_key, predicate, value, value_norm, seq, valid_from,
                        mention_time, valid_to, superseded_by, conflict_group, note_id, source_span
                 FROM slot_ledger
                 WHERE entity_key = ?1 AND predicate = ?2
                 ORDER BY seq",
            )
            .map_err(|e| KartaError::VectorStore(e.to_string()))?;
        let rows = stmt
            .query_map(rusqlite::params![entity_key, predicate], row_to_ledger_row)
            .map_err(|e| KartaError::VectorStore(e.to_string()))?;
        let mut out = Vec::new();
        for r in rows {
            let tuple = r.map_err(|e| KartaError::VectorStore(e.to_string()))?;
            out.push(tuple_to_row(tuple)?);
        }
        Ok(out)
    }

    pub fn close(&self, id: i64, valid_to: DateTime<Utc>, superseded_by: i64) -> Result<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| KartaError::VectorStore(e.to_string()))?;
        conn.execute(
            "UPDATE slot_ledger SET valid_to = ?2, superseded_by = ?3 WHERE id = ?1",
            rusqlite::params![id, valid_to.to_rfc3339(), superseded_by],
        )
        .map_err(|e| KartaError::VectorStore(e.to_string()))?;
        Ok(())
    }

    pub fn set_conflict_group(&self, ids: &[i64], group: i64) -> Result<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| KartaError::VectorStore(e.to_string()))?;
        for id in ids {
            conn.execute(
                "UPDATE slot_ledger SET conflict_group = ?2 WHERE id = ?1",
                rusqlite::params![id, group],
            )
            .map_err(|e| KartaError::VectorStore(e.to_string()))?;
        }
        Ok(())
    }
}

#[allow(clippy::type_complexity)]
fn tuple_to_row(
    tuple: (
        i64,
        String,
        String,
        String,
        String,
        i64,
        Option<String>,
        String,
        Option<String>,
        Option<i64>,
        Option<i64>,
        String,
        String,
    ),
) -> Result<LedgerRow> {
    let (
        id,
        entity_key,
        predicate,
        value,
        value_norm,
        seq,
        valid_from,
        mention_time,
        valid_to,
        superseded_by,
        conflict_group,
        note_id,
        source_span,
    ) = tuple;
    Ok(LedgerRow {
        id,
        entity_key,
        predicate,
        value,
        value_norm,
        seq: seq as u64,
        valid_from: parse_rfc3339_opt(valid_from)?,
        mention_time: parse_rfc3339(&mention_time)?,
        valid_to: parse_rfc3339_opt(valid_to)?,
        superseded_by,
        conflict_group,
        note_id,
        source_span,
    })
}
