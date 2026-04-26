use std::sync::Arc;

use serde::Serialize;

use crate::error::Result;
use crate::rules::{FiredRule, ProceduralRule, RuleCondition, RuleContext, RuleEvaluation};
use crate::store::GraphStore;

/// Engine that evaluates procedural rules against retrieval context.
pub struct RuleEngine {
    graph_store: Arc<dyn GraphStore>,
}

impl RuleEngine {
    pub fn new(graph_store: Arc<dyn GraphStore>) -> Self {
        Self { graph_store }
    }

    /// Evaluate all enabled rules against the given context.
    pub async fn evaluate(&self, ctx: &RuleContext) -> Result<RuleEvaluation> {
        let rules = self.list_rules().await?;
        let mut fired_rules = Vec::new();

        for rule in &rules {
            if !rule.enabled {
                continue;
            }

            if Self::matches(&rule.condition, ctx) {
                fired_rules.push(FiredRule {
                    rule_id: rule.id.clone(),
                    rule_name: rule.name.clone(),
                    actions: rule.actions.clone(),
                });
                let _ = self.graph_store.increment_rule_fire_count(&rule.id).await;
            }
        }

        Ok(RuleEvaluation { fired_rules })
    }

    pub async fn add_rule(&self, rule: ProceduralRule) -> Result<()> {
        self.graph_store.upsert_procedural_rule(&rule).await
    }

    pub async fn disable_rule(&self, rule_id: &str) -> Result<()> {
        self.graph_store.disable_procedural_rule(rule_id).await
    }

    pub async fn list_rules(&self) -> Result<Vec<ProceduralRule>> {
        self.graph_store.list_procedural_rules().await
    }

    fn matches(condition: &RuleCondition, ctx: &RuleContext) -> bool {
        match condition {
            RuleCondition::QueryContains { keywords } => {
                let query_lower = ctx.query.to_lowercase();
                keywords
                    .iter()
                    .any(|k| query_lower.contains(&k.to_lowercase()))
            }
            RuleCondition::SessionMatches { session_id } => {
                ctx.session_id.as_ref().is_some_and(|s| s == session_id)
            }
            RuleCondition::ContradictionForEntity { entity } => {
                ctx.contradiction_entities.contains(entity)
            }
            RuleCondition::RetrievedTagPresent { tag } => ctx.retrieved_tags.contains(tag),
            RuleCondition::Always => true,
        }
    }
}

/// Trace output showing which rules fired during a query.
#[derive(Debug, Clone, Serialize, Default)]
pub struct RuleTrace {
    pub fired_rules: Vec<FiredRuleTrace>,
}

#[derive(Debug, Clone, Serialize)]
pub struct FiredRuleTrace {
    pub rule_id: String,
    pub rule_name: String,
}
