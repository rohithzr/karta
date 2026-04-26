use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// A safe action that a rule can perform. Constrained to prompt/retrieval
/// modifications only — rules cannot mutate storage or execute arbitrary code.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum RuleAction {
    AppendSystemPrompt { text: String },
    RewriteQuery { prefix: String },
    FilterByTags { tags: Vec<String> },
    BoostKeywords { keywords: Vec<String>, boost: f32 },
    LimitTopK { top_k: usize },
    WarnContradiction { message: String },
}

/// Condition that triggers a rule.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum RuleCondition {
    QueryContains { keywords: Vec<String> },
    SessionMatches { session_id: String },
    ContradictionForEntity { entity: String },
    RetrievedTagPresent { tag: String },
    Always,
}

/// A procedural rule that modifies retrieval/answer behavior when its condition matches.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProceduralRule {
    pub id: String,
    pub name: String,
    pub description: String,
    pub condition: RuleCondition,
    pub actions: Vec<RuleAction>,
    pub enabled: bool,
    pub protected: bool,
    pub source_note_id: Option<String>,
    pub fire_count: u64,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl ProceduralRule {
    pub fn new(
        name: String,
        description: String,
        condition: RuleCondition,
        actions: Vec<RuleAction>,
    ) -> Self {
        let now = Utc::now();
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            name,
            description,
            condition,
            actions,
            enabled: true,
            protected: false,
            source_note_id: None,
            fire_count: 0,
            created_at: now,
            updated_at: now,
        }
    }

    pub fn with_source(mut self, note_id: &str) -> Self {
        self.source_note_id = Some(note_id.to_string());
        self.protected = true;
        self
    }
}

/// Context passed to the rule engine for evaluation.
#[derive(Debug, Clone, Default)]
pub struct RuleContext {
    pub query: String,
    pub session_id: Option<String>,
    pub retrieved_tags: Vec<String>,
    pub contradiction_entities: Vec<String>,
}

/// Result of rule evaluation.
#[derive(Debug, Clone, Serialize, Default)]
pub struct RuleEvaluation {
    pub fired_rules: Vec<FiredRule>,
}

#[derive(Debug, Clone, Serialize)]
pub struct FiredRule {
    pub rule_id: String,
    pub rule_name: String,
    pub actions: Vec<RuleAction>,
}
