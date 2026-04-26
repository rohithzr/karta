use serde::{Deserialize, Serialize};

/// Result of running an extractor on content.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ExtractionResult {
    pub facts: Vec<ExtractedFact>,
    pub edges: Vec<ExtractedEdge>,
    pub metadata: Vec<(String, String)>,
}

/// A fact extracted deterministically from structured content.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedFact {
    pub key: String,
    pub value: String,
    pub confidence: f32,
}

/// An edge extracted from structured content (e.g., dependency relationships).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedEdge {
    pub from: String,
    pub to: String,
    pub edge_type: String,
}

/// Trait for deterministic extractors that run before LLM extraction.
pub trait Extractor: Send + Sync {
    /// Returns the name of this extractor (e.g., "markdown", "json", "cargo_toml").
    fn name(&self) -> &str;

    /// Returns true if this extractor can handle the given content type.
    fn can_extract(&self, content_type: &str, content: &str) -> bool;

    /// Extract facts, edges, and metadata from content.
    fn extract(&self, content_type: &str, content: &str) -> ExtractionResult;
}

/// Registry of deterministic extractors.
pub struct ExtractorRegistry {
    extractors: Vec<Box<dyn Extractor>>,
}

impl ExtractorRegistry {
    pub fn new() -> Self {
        let mut registry = Self {
            extractors: Vec::new(),
        };
        registry.register_default_extractors();
        registry
    }

    pub fn register(&mut self, extractor: Box<dyn Extractor>) {
        self.extractors.push(extractor);
    }

    pub fn run(&self, content_type: &str, content: &str) -> ExtractionResult {
        let mut combined = ExtractionResult::default();
        for extractor in &self.extractors {
            if extractor.can_extract(content_type, content) {
                let result = extractor.extract(content_type, content);
                combined.facts.extend(result.facts);
                combined.edges.extend(result.edges);
                combined.metadata.extend(result.metadata);
            }
        }
        combined
    }

    fn register_default_extractors(&mut self) {
        self.register(Box::new(MarkdownExtractor));
        self.register(Box::new(JsonExtractor));
        self.register(Box::new(YamlExtractor));
        self.register(Box::new(CargoTomlExtractor));
    }
}

impl Default for ExtractorRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Extractor for Markdown content.
pub struct MarkdownExtractor;

impl Extractor for MarkdownExtractor {
    fn name(&self) -> &str {
        "markdown"
    }

    fn can_extract(&self, content_type: &str, _content: &str) -> bool {
        content_type == "markdown" || content_type == "md"
    }

    fn extract(&self, _content_type: &str, content: &str) -> ExtractionResult {
        let mut result = ExtractionResult::default();

        for line in content.lines() {
            if line.starts_with('#') {
                let level = line.chars().take_while(|c| *c == '#').count();
                let text = line
                    .chars()
                    .skip(level)
                    .skip_while(|c| c.is_whitespace())
                    .collect::<String>();
                result.facts.push(ExtractedFact {
                    key: format!("heading_level_{}", level),
                    value: text,
                    confidence: 1.0,
                });
            }

            if line.starts_with("```") {
                let lang = line.strip_prefix("```").unwrap_or("").trim();
                if !lang.is_empty() {
                    result
                        .metadata
                        .push(("code_fence_language".into(), lang.into()));
                }
            }

            if let (true, Some(link_text)) = (
                line.starts_with('[') && line.contains("]("),
                line.split_once("]("),
            ) {
                result.facts.push(ExtractedFact {
                    key: "link".into(),
                    value: format!(
                        "{} -> {}",
                        link_text.0.trim_start_matches('[').trim_end_matches(']'),
                        link_text.1.trim_end_matches(')')
                    ),
                    confidence: 1.0,
                });
            }
        }

        result
    }
}

/// Extractor for JSON content.
pub struct JsonExtractor;

impl Extractor for JsonExtractor {
    fn name(&self) -> &str {
        "json"
    }

    fn can_extract(&self, content_type: &str, content: &str) -> bool {
        (content_type == "json") && content.trim_start().starts_with('{')
    }

    fn extract(&self, _content_type: &str, content: &str) -> ExtractionResult {
        let mut result = ExtractionResult::default();

        if let Ok(value) = serde_json::from_str::<serde_json::Value>(content) {
            extract_json_paths(&value, "", &mut result.facts);
        }

        result
    }
}

fn extract_json_paths(value: &serde_json::Value, path: &str, facts: &mut Vec<ExtractedFact>) {
    match value {
        serde_json::Value::Object(map) => {
            for (key, val) in map {
                let new_path = if path.is_empty() {
                    key.clone()
                } else {
                    format!("{}.{}", path, key)
                };
                extract_json_paths(val, &new_path, facts);
            }
        }
        serde_json::Value::Array(arr) => {
            for (i, val) in arr.iter().enumerate() {
                let new_path = format!("{}[{}]", path, i);
                extract_json_paths(val, &new_path, facts);
            }
        }
        serde_json::Value::String(s) => {
            facts.push(ExtractedFact {
                key: path.to_string(),
                value: s.clone(),
                confidence: 1.0,
            });
        }
        serde_json::Value::Number(n) => {
            facts.push(ExtractedFact {
                key: path.to_string(),
                value: n.to_string(),
                confidence: 1.0,
            });
        }
        serde_json::Value::Bool(b) => {
            facts.push(ExtractedFact {
                key: path.to_string(),
                value: b.to_string(),
                confidence: 1.0,
            });
        }
        _ => {}
    }
}

/// Extractor for YAML content.
pub struct YamlExtractor;

impl Extractor for YamlExtractor {
    fn name(&self) -> &str {
        "yaml"
    }

    fn can_extract(&self, content_type: &str, content: &str) -> bool {
        (content_type == "yaml" || content_type == "yml") && content.contains(':')
    }

    fn extract(&self, _content_type: &str, content: &str) -> ExtractionResult {
        let mut result = ExtractionResult::default();

        for line in content.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            if let Some((key, value)) = trimmed.split_once(':') {
                let value = value.trim().trim_matches('"').trim_matches('\'');
                if !value.is_empty() {
                    result.facts.push(ExtractedFact {
                        key: key.trim().to_string(),
                        value: value.to_string(),
                        confidence: 0.9,
                    });
                }
            }
        }

        result
    }
}

/// Extractor for Cargo.toml files.
pub struct CargoTomlExtractor;

impl Extractor for CargoTomlExtractor {
    fn name(&self) -> &str {
        "cargo_toml"
    }

    fn can_extract(&self, content_type: &str, content: &str) -> bool {
        (content_type == "toml" || content_type == "cargo_toml")
            && (content.contains("[package]") || content.contains("[workspace]"))
    }

    fn extract(&self, _content_type: &str, content: &str) -> ExtractionResult {
        let mut result = ExtractionResult::default();
        let mut current_section = String::new();

        for line in content.lines() {
            let trimmed = line.trim();

            if trimmed.starts_with('[') && trimmed.ends_with(']') {
                current_section = trimmed.trim_matches('[').trim_matches(']').to_string();
                continue;
            }

            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }

            if let Some((key, value)) = trimmed.split_once('=') {
                let key = key.trim();
                let value = value.trim().trim_matches('"');

                if current_section == "package" {
                    result.facts.push(ExtractedFact {
                        key: format!("package.{}", key),
                        value: value.to_string(),
                        confidence: 1.0,
                    });
                } else if current_section.starts_with("dependencies") {
                    result.edges.push(ExtractedEdge {
                        from: "this_crate".into(),
                        to: key.to_string(),
                        edge_type: "depends_on".into(),
                    });
                } else if current_section == "workspace" && key == "members" {
                    result
                        .metadata
                        .push(("workspace_members".into(), value.to_string()));
                }
            }
        }

        result
    }
}
