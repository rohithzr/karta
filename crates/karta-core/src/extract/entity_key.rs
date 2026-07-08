//! Write-time entity-key canonicalization: normalization + alias resolution.
//!
//! Normalizes free-text entity mentions into a stable key (lowercase, trimmed,
//! whitespace-collapsed, stopwords dropped, last token lightly singularized),
//! and resolves near-duplicate mentions to a single canonical form via exact
//! match or cosine similarity over optional embeddings.

const STOPWORDS: &[&str] = &["the", "a", "an", "my", "our", "your"];

/// Normalize an entity mention into a stable comparison key.
pub fn normalize_entity(text: &str) -> String {
    let mut tokens: Vec<String> = text
        .trim()
        .split_whitespace()
        .map(|t| t.to_lowercase())
        .filter(|t| !STOPWORDS.contains(&t.as_str()))
        .collect();

    if let Some(last) = tokens.last_mut() {
        if let Some(stripped) = last.strip_suffix('s') {
            if !stripped.is_empty() {
                *last = stripped.to_string();
            }
        }
    }

    tokens.join(" ")
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

/// A simple in-memory table of canonical entity keys and their embeddings,
/// used to resolve near-duplicate mentions to a single canonical form.
#[derive(Debug, Clone, Default)]
pub struct EntityAliases {
    canon: Vec<(String, Vec<f32>)>,
}

impl EntityAliases {
    pub fn new() -> Self {
        Self { canon: Vec::new() }
    }

    /// Resolve a normalized entity string to a canonical form.
    ///
    /// Exact match on the normalized string wins first; otherwise, if an
    /// embedding is provided, the best cosine match at or above `threshold`
    /// is returned. Otherwise, `norm` is inserted as a new canonical entry
    /// and returned.
    pub fn resolve(&mut self, norm: &str, embed: Option<&[f32]>, threshold: f32) -> String {
        if let Some((existing, _)) = self.canon.iter().find(|(c, _)| c == norm) {
            return existing.clone();
        }

        if let Some(vec) = embed {
            let mut best: Option<(f32, &str)> = None;
            for (c, v) in &self.canon {
                let sim = cosine_similarity(vec, v);
                if best.map(|(b, _)| sim > b).unwrap_or(true) {
                    best = Some((sim, c.as_str()));
                }
            }
            if let Some((sim, c)) = best {
                if sim >= threshold {
                    return c.to_string();
                }
            }
        }

        let stored_vec = embed.map(|v| v.to_vec()).unwrap_or_default();
        self.canon.push((norm.to_string(), stored_vec));
        norm.to_string()
    }

    /// Look up a canonical entity key by exact normalized-string match,
    /// without inserting a new entry when absent.
    pub fn lookup_exact(&self, norm: &str) -> Option<String> {
        self.canon
            .iter()
            .find(|(c, _)| c == norm)
            .map(|(c, _)| c.clone())
    }

    /// All known canonical entity keys.
    pub fn keys(&self) -> Vec<String> {
        self.canon.iter().map(|(c, _)| c.clone()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn normalize_strips_articles_and_singularizes() {
        assert_eq!(normalize_entity("The Project"), "project");
        assert_eq!(normalize_entity("my  API  keys"), "api key");
    }
    #[test]
    fn resolve_exact_norm_collapses_without_embeddings() {
        let mut a = EntityAliases::new();
        assert_eq!(a.resolve("first sprint", None, 0.9), "first sprint");
        assert_eq!(a.resolve("first sprint", None, 0.9), "first sprint");
    }
    #[test]
    fn resolve_keeps_distinct_entities_apart() {
        let mut a = EntityAliases::new();
        // orthogonal embeddings, high threshold → no over-merge
        assert_eq!(a.resolve("dashboard api", Some(&[1.0, 0.0]), 0.9), "dashboard api");
        assert_eq!(a.resolve("grocery budget", Some(&[0.0, 1.0]), 0.9), "grocery budget");
    }
    #[test]
    fn lookup_exact_finds_without_inserting() {
        let mut a = EntityAliases::new();
        assert_eq!(a.lookup_exact("first sprint"), None);
        a.resolve("first sprint", None, 0.9);
        assert_eq!(a.lookup_exact("first sprint"), Some("first sprint".to_string()));
        // still absent, still not inserted
        assert_eq!(a.lookup_exact("second sprint"), None);
        assert_eq!(a.keys(), vec!["first sprint".to_string()]);
    }
}
