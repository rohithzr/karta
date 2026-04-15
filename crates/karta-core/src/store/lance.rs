use std::sync::Arc;

use arrow_array::{
    Float32Array, RecordBatch, RecordBatchIterator, StringArray,
    RecordBatchReader,
};
use arrow_schema::{DataType, Field, Schema};
use async_trait::async_trait;
use futures::TryStreamExt;
use lancedb::{
    Connection,
    connect,
    query::{ExecutableQuery, QueryBase},
    Table as LanceTable,
};
use tokio::sync::RwLock;

use crate::error::{KartaError, Result};
use crate::note::{MemoryNote, NoteStatus, Provenance};

const TABLE_NAME: &str = "notes";
const FACTS_TABLE_NAME: &str = "atomic_facts";
const EMBEDDING_DIM: usize = 1536; // text-embedding-3-small default

pub struct LanceVectorStore {
    conn: Connection,
    table: RwLock<Option<LanceTable>>,
    facts_table: RwLock<Option<LanceTable>>,
}

impl LanceVectorStore {
    pub async fn new(uri: &str) -> Result<Self> {
        // Only create local directories for local paths
        if !uri.starts_with("gs://") && !uri.starts_with("s3://") && !uri.starts_with("az://") {
            std::fs::create_dir_all(uri)
                .map_err(|e| KartaError::VectorStore(e.to_string()))?;
        }
        let conn = connect(uri)
            .execute()
            .await
            .map_err(|e| KartaError::VectorStore(e.to_string()))?;

        let store = Self {
            conn,
            table: RwLock::new(None),
            facts_table: RwLock::new(None),
        };
        store.ensure_table().await?;
        store.ensure_facts_table().await?;
        Ok(store)
    }

    fn schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("content", DataType::Utf8, false),
            Field::new("context", DataType::Utf8, false),
            Field::new("keywords_json", DataType::Utf8, false),
            Field::new("tags_json", DataType::Utf8, false),
            Field::new("provenance_json", DataType::Utf8, false),
            Field::new("confidence", DataType::Float32, false),
            Field::new("created_at", DataType::Utf8, false),
            Field::new("updated_at", DataType::Utf8, false),
            Field::new("status_json", DataType::Utf8, false),
            Field::new("last_accessed_at", DataType::Utf8, false),
            Field::new("turn_index", DataType::Utf8, true),
            Field::new("source_timestamp", DataType::Utf8, true),
            Field::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    EMBEDDING_DIM as i32,
                ),
                false,
            ),
        ]))
    }

    fn make_reader(
        batches: Vec<RecordBatch>,
        schema: Arc<Schema>,
    ) -> Box<dyn RecordBatchReader + Send> {
        Box::new(RecordBatchIterator::new(
            batches.into_iter().map(Ok),
            schema,
        ))
    }

    async fn ensure_table(&self) -> Result<()> {
        let mut table_lock = self.table.write().await;
        if table_lock.is_some() {
            return Ok(());
        }

        let names = self
            .conn
            .table_names()
            .execute()
            .await
            .map_err(|e| KartaError::VectorStore(e.to_string()))?;

        let table = if names.contains(&TABLE_NAME.to_string()) {
            self.conn
                .open_table(TABLE_NAME)
                .execute()
                .await
                .map_err(|e| KartaError::VectorStore(e.to_string()))?
        } else {
            let schema = Self::schema();
            let empty_batch = RecordBatch::new_empty(schema.clone());
            let reader = Self::make_reader(vec![empty_batch], schema);
            self.conn
                .create_table(TABLE_NAME, reader)
                .execute()
                .await
                .map_err(|e| KartaError::VectorStore(e.to_string()))?
        };

        *table_lock = Some(table);
        Ok(())
    }

    // --- Atomic Facts table ---

    fn facts_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("content", DataType::Utf8, false),
            Field::new("source_note_id", DataType::Utf8, false),
            Field::new("ordinal", DataType::Utf8, false),
            Field::new("subject", DataType::Utf8, true),
            Field::new("created_at", DataType::Utf8, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    EMBEDDING_DIM as i32,
                ),
                false,
            ),
        ]))
    }

    async fn ensure_facts_table(&self) -> Result<()> {
        let mut table_lock = self.facts_table.write().await;
        if table_lock.is_some() {
            return Ok(());
        }

        let names = self.conn.table_names().execute().await
            .map_err(|e| KartaError::VectorStore(e.to_string()))?;

        let table = if names.contains(&FACTS_TABLE_NAME.to_string()) {
            self.conn.open_table(FACTS_TABLE_NAME).execute().await
                .map_err(|e| KartaError::VectorStore(e.to_string()))?
        } else {
            let schema = Self::facts_schema();
            let empty_batch = RecordBatch::new_empty(schema.clone());
            let reader = Self::make_reader(vec![empty_batch], schema);
            self.conn.create_table(FACTS_TABLE_NAME, reader).execute().await
                .map_err(|e| KartaError::VectorStore(e.to_string()))?
        };

        *table_lock = Some(table);
        Ok(())
    }

    fn fact_to_batch(fact: &crate::note::AtomicFact) -> Result<RecordBatch> {
        let embedding = if fact.embedding.len() == EMBEDDING_DIM {
            fact.embedding.clone()
        } else {
            vec![0.0f32; EMBEDDING_DIM]
        };

        let vector_array = arrow_array::FixedSizeListArray::try_new(
            Arc::new(Field::new("item", DataType::Float32, true)),
            EMBEDDING_DIM as i32,
            Arc::new(Float32Array::from(embedding)),
            None,
        ).map_err(|e| KartaError::VectorStore(e.to_string()))?;

        let created_at = fact.created_at.to_rfc3339();
        let ordinal_str = fact.ordinal.to_string();
        let subject_str = fact.subject.clone().unwrap_or_default();

        RecordBatch::try_new(
            Self::facts_schema(),
            vec![
                Arc::new(StringArray::from(vec![fact.id.as_str()])),
                Arc::new(StringArray::from(vec![fact.content.as_str()])),
                Arc::new(StringArray::from(vec![fact.source_note_id.as_str()])),
                Arc::new(StringArray::from(vec![ordinal_str.as_str()])),
                Arc::new(StringArray::from(vec![subject_str.as_str()])),
                Arc::new(StringArray::from(vec![created_at.as_str()])),
                Arc::new(vector_array),
            ],
        ).map_err(|e| KartaError::VectorStore(e.to_string()))
    }

    fn batch_to_facts(batch: &RecordBatch) -> Result<Vec<crate::note::AtomicFact>> {
        let ids = batch.column(0).as_any().downcast_ref::<StringArray>().unwrap();
        let contents = batch.column(1).as_any().downcast_ref::<StringArray>().unwrap();
        let source_ids = batch.column(2).as_any().downcast_ref::<StringArray>().unwrap();
        let ordinals = batch.column(3).as_any().downcast_ref::<StringArray>().unwrap();
        let subjects = batch.column(4).as_any().downcast_ref::<StringArray>().unwrap();
        let created_ats = batch.column(5).as_any().downcast_ref::<StringArray>().unwrap();
        let vector_col = batch.column(6).as_any()
            .downcast_ref::<arrow_array::FixedSizeListArray>().unwrap();

        let mut facts = Vec::with_capacity(batch.num_rows());
        for i in 0..batch.num_rows() {
            let embedding = vector_col.value(i).as_any()
                .downcast_ref::<Float32Array>().unwrap().values().to_vec();
            let subject_val = subjects.value(i);

            facts.push(crate::note::AtomicFact {
                id: ids.value(i).to_string(),
                content: contents.value(i).to_string(),
                source_note_id: source_ids.value(i).to_string(),
                ordinal: ordinals.value(i).parse().unwrap_or(0),
                subject: if subject_val.is_empty() { None } else { Some(subject_val.to_string()) },
                embedding,
                created_at: chrono::DateTime::parse_from_rfc3339(created_ats.value(i))
                    .unwrap_or_default().with_timezone(&chrono::Utc),
            });
        }
        Ok(facts)
    }

    async fn get_facts_table(&self) -> Result<LanceTable> {
        let lock = self.facts_table.read().await;
        lock.clone()
            .ok_or_else(|| KartaError::VectorStore("Facts table not initialized".into()))
    }

    fn note_to_batch(note: &MemoryNote) -> Result<RecordBatch> {
        let embedding = if note.embedding.len() == EMBEDDING_DIM {
            note.embedding.clone()
        } else {
            vec![0.0f32; EMBEDDING_DIM]
        };

        let vector_array = arrow_array::FixedSizeListArray::try_new(
            Arc::new(Field::new("item", DataType::Float32, true)),
            EMBEDDING_DIM as i32,
            Arc::new(Float32Array::from(embedding)),
            None,
        )
        .map_err(|e| KartaError::VectorStore(e.to_string()))?;

        // Serialize JSON fields to owned Strings first
        let keywords_json = serde_json::to_string(&note.keywords)?;
        let tags_json = serde_json::to_string(&note.tags)?;
        let provenance_json = serde_json::to_string(&note.provenance)?;
        let status_json = serde_json::to_string(&note.status)?;
        let created_at = note.created_at.to_rfc3339();
        let updated_at = note.updated_at.to_rfc3339();
        let last_accessed_at = note.last_accessed_at.to_rfc3339();
        let turn_index_str = note.turn_index.map(|t| t.to_string()).unwrap_or_default();
        let source_timestamp_str = note.source_timestamp.map(|t| t.to_rfc3339()).unwrap_or_default();

        let batch = RecordBatch::try_new(
            Self::schema(),
            vec![
                Arc::new(StringArray::from(vec![note.id.as_str()])),
                Arc::new(StringArray::from(vec![note.content.as_str()])),
                Arc::new(StringArray::from(vec![note.context.as_str()])),
                Arc::new(StringArray::from(vec![keywords_json.as_str()])),
                Arc::new(StringArray::from(vec![tags_json.as_str()])),
                Arc::new(StringArray::from(vec![provenance_json.as_str()])),
                Arc::new(Float32Array::from(vec![note.confidence])),
                Arc::new(StringArray::from(vec![created_at.as_str()])),
                Arc::new(StringArray::from(vec![updated_at.as_str()])),
                Arc::new(StringArray::from(vec![status_json.as_str()])),
                Arc::new(StringArray::from(vec![last_accessed_at.as_str()])),
                Arc::new(StringArray::from(vec![turn_index_str.as_str()])),
                Arc::new(StringArray::from(vec![source_timestamp_str.as_str()])),
                Arc::new(vector_array),
            ],
        )
        .map_err(|e| KartaError::VectorStore(e.to_string()))?;

        Ok(batch)
    }

    fn batch_to_notes(batch: &RecordBatch) -> Result<Vec<MemoryNote>> {
        let ids = batch.column(0).as_any().downcast_ref::<StringArray>().unwrap();
        let contents = batch.column(1).as_any().downcast_ref::<StringArray>().unwrap();
        let contexts = batch.column(2).as_any().downcast_ref::<StringArray>().unwrap();
        let keywords_jsons = batch.column(3).as_any().downcast_ref::<StringArray>().unwrap();
        let tags_jsons = batch.column(4).as_any().downcast_ref::<StringArray>().unwrap();
        let provenance_jsons = batch.column(5).as_any().downcast_ref::<StringArray>().unwrap();
        let confidences = batch.column(6).as_any().downcast_ref::<Float32Array>().unwrap();
        let created_ats = batch.column(7).as_any().downcast_ref::<StringArray>().unwrap();
        let updated_ats = batch.column(8).as_any().downcast_ref::<StringArray>().unwrap();
        let status_jsons = batch.column(9).as_any().downcast_ref::<StringArray>().unwrap();
        let last_accessed_ats = batch.column(10).as_any().downcast_ref::<StringArray>().unwrap();
        let turn_index_strs = batch.column(11).as_any().downcast_ref::<StringArray>().unwrap();
        let source_timestamp_strs = batch.column(12).as_any().downcast_ref::<StringArray>().unwrap();

        let vector_col = batch
            .column(13)
            .as_any()
            .downcast_ref::<arrow_array::FixedSizeListArray>()
            .unwrap();

        let mut notes = Vec::with_capacity(batch.num_rows());

        for i in 0..batch.num_rows() {
            let embedding_array = vector_col
                .value(i)
                .as_any()
                .downcast_ref::<Float32Array>()
                .unwrap()
                .values()
                .to_vec();

            let keywords: Vec<String> =
                serde_json::from_str(keywords_jsons.value(i)).unwrap_or_default();
            let tags: Vec<String> =
                serde_json::from_str(tags_jsons.value(i)).unwrap_or_default();
            let provenance: Provenance =
                serde_json::from_str(provenance_jsons.value(i))
                    .unwrap_or(Provenance::Observed);
            let status: NoteStatus =
                serde_json::from_str(status_jsons.value(i))
                    .unwrap_or_default();

            notes.push(MemoryNote {
                id: ids.value(i).to_string(),
                content: contents.value(i).to_string(),
                context: contexts.value(i).to_string(),
                keywords,
                tags,
                links: Vec::new(),
                embedding: embedding_array,
                created_at: chrono::DateTime::parse_from_rfc3339(created_ats.value(i))
                    .unwrap_or_default()
                    .with_timezone(&chrono::Utc),
                updated_at: chrono::DateTime::parse_from_rfc3339(updated_ats.value(i))
                    .unwrap_or_default()
                    .with_timezone(&chrono::Utc),
                evolution_history: Vec::new(),
                provenance,
                confidence: confidences.value(i),
                status,
                last_accessed_at: chrono::DateTime::parse_from_rfc3339(last_accessed_ats.value(i))
                    .unwrap_or_default()
                    .with_timezone(&chrono::Utc),
                turn_index: {
                    let s = turn_index_strs.value(i);
                    if s.is_empty() { None } else { s.parse().ok() }
                },
                source_timestamp: {
                    let s = source_timestamp_strs.value(i);
                    if s.is_empty() {
                        None
                    } else {
                        chrono::DateTime::parse_from_rfc3339(s)
                            .ok()
                            .map(|d| d.with_timezone(&chrono::Utc))
                    }
                },
            });
        }

        Ok(notes)
    }

    async fn get_table(&self) -> Result<LanceTable> {
        let lock = self.table.read().await;
        lock.clone()
            .ok_or_else(|| KartaError::VectorStore("Table not initialized".into()))
    }

    async fn collect_batches(
        stream: impl futures::Stream<Item = std::result::Result<RecordBatch, lancedb::Error>> + Unpin,
    ) -> Result<Vec<RecordBatch>> {
        stream
            .try_collect()
            .await
            .map_err(|e| KartaError::VectorStore(e.to_string()))
    }
}

#[async_trait]
impl crate::store::VectorStore for LanceVectorStore {
    async fn upsert(&self, note: &MemoryNote) -> Result<()> {
        let table = self.get_table().await?;

        // Delete existing row if present (upsert semantics)
        let _ = table
            .delete(&format!("id = '{}'", note.id))
            .await;

        let batch = Self::note_to_batch(note)?;
        let schema = Self::schema();
        let reader = Self::make_reader(vec![batch], schema);
        table
            .add(reader)
            .execute()
            .await
            .map_err(|e| KartaError::VectorStore(e.to_string()))?;

        Ok(())
    }

    async fn find_similar(
        &self,
        embedding: &[f32],
        top_k: usize,
        exclude_ids: &[&str],
    ) -> Result<Vec<(MemoryNote, f32)>> {
        let table = self.get_table().await?;

        let query = table
            .vector_search(embedding)
            .map_err(|e| KartaError::VectorStore(e.to_string()))?
            .limit(top_k + exclude_ids.len());

        let results = query
            .execute()
            .await
            .map_err(|e| KartaError::VectorStore(e.to_string()))?;

        let batches = Self::collect_batches(results).await?;

        let mut scored_notes = Vec::new();
        for batch in &batches {
            let notes = Self::batch_to_notes(batch)?;
            // LanceDB adds a _distance column
            let distance_col = batch
                .column_by_name("_distance")
                .and_then(|c| c.as_any().downcast_ref::<Float32Array>());

            for (i, note) in notes.into_iter().enumerate() {
                if exclude_ids.contains(&note.id.as_str()) {
                    continue;
                }
                // LanceDB returns L2 distance; convert to similarity score
                let distance = distance_col.map(|d| d.value(i)).unwrap_or(0.0);
                let score = 1.0 / (1.0 + distance);
                scored_notes.push((note, score));
            }
        }

        scored_notes.truncate(top_k);
        Ok(scored_notes)
    }

    async fn get(&self, id: &str) -> Result<Option<MemoryNote>> {
        let table = self.get_table().await?;

        let results = table
            .query()
            .only_if(format!("id = '{}'", id))
            .execute()
            .await
            .map_err(|e| KartaError::VectorStore(e.to_string()))?;

        let batches = Self::collect_batches(results).await?;

        for batch in &batches {
            let notes = Self::batch_to_notes(batch)?;
            if let Some(note) = notes.into_iter().next() {
                return Ok(Some(note));
            }
        }
        Ok(None)
    }

    async fn get_many(&self, ids: &[&str]) -> Result<Vec<MemoryNote>> {
        let mut notes = Vec::new();
        for id in ids {
            if let Some(note) = self.get(id).await? {
                notes.push(note);
            }
        }
        Ok(notes)
    }

    async fn get_all(&self) -> Result<Vec<MemoryNote>> {
        let table = self.get_table().await?;

        let results = table
            .query()
            .execute()
            .await
            .map_err(|e| KartaError::VectorStore(e.to_string()))?;

        let batches = Self::collect_batches(results).await?;

        let mut all_notes = Vec::new();
        for batch in &batches {
            all_notes.extend(Self::batch_to_notes(batch)?);
        }
        Ok(all_notes)
    }

    async fn delete(&self, id: &str) -> Result<()> {
        let table = self.get_table().await?;
        table
            .delete(&format!("id = '{}'", id))
            .await
            .map_err(|e| KartaError::VectorStore(e.to_string()))?;
        Ok(())
    }

    async fn count(&self) -> Result<usize> {
        let table = self.get_table().await?;
        let count = table
            .count_rows(None)
            .await
            .map_err(|e| KartaError::VectorStore(e.to_string()))?;
        Ok(count)
    }

    // --- Atomic Facts ---

    async fn upsert_fact(&self, fact: &crate::note::AtomicFact) -> Result<()> {
        let table = self.get_facts_table().await?;
        let _ = table.delete(&format!("id = '{}'", fact.id)).await;
        let batch = Self::fact_to_batch(fact)?;
        let schema = Self::facts_schema();
        let reader = Self::make_reader(vec![batch], schema);
        table.add(reader).execute().await
            .map_err(|e| KartaError::VectorStore(e.to_string()))?;
        Ok(())
    }

    async fn find_similar_facts(
        &self,
        embedding: &[f32],
        top_k: usize,
        exclude_source_note_ids: &[&str],
    ) -> Result<Vec<(crate::note::AtomicFact, f32)>> {
        let table = self.get_facts_table().await?;
        let query = table.vector_search(embedding)
            .map_err(|e| KartaError::VectorStore(e.to_string()))?
            .limit(top_k + exclude_source_note_ids.len() * 5);
        let results = query.execute().await
            .map_err(|e| KartaError::VectorStore(e.to_string()))?;
        let batches = Self::collect_batches(results).await?;

        let mut scored = Vec::new();
        for batch in &batches {
            let facts = Self::batch_to_facts(batch)?;
            let distance_col = batch.column_by_name("_distance")
                .and_then(|c| c.as_any().downcast_ref::<Float32Array>());
            for (i, fact) in facts.into_iter().enumerate() {
                if exclude_source_note_ids.contains(&fact.source_note_id.as_str()) {
                    continue;
                }
                let distance = distance_col.map(|d| d.value(i)).unwrap_or(0.0);
                let score = 1.0 / (1.0 + distance);
                scored.push((fact, score));
            }
        }
        scored.truncate(top_k);
        Ok(scored)
    }

    async fn get_facts_for_note(&self, note_id: &str) -> Result<Vec<crate::note::AtomicFact>> {
        let table = self.get_facts_table().await?;
        let results = table.query()
            .only_if(format!("source_note_id = '{}'", note_id))
            .execute().await
            .map_err(|e| KartaError::VectorStore(e.to_string()))?;
        let batches = Self::collect_batches(results).await?;
        let mut facts = Vec::new();
        for batch in &batches {
            facts.extend(Self::batch_to_facts(batch)?);
        }
        facts.sort_by_key(|f| f.ordinal);
        Ok(facts)
    }
}
