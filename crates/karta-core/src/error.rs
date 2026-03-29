use thiserror::Error;

#[derive(Error, Debug)]
pub enum KartaError {
    #[error("Vector store error: {0}")]
    VectorStore(String),

    #[error("Graph store error: {0}")]
    GraphStore(String),

    #[error("LLM provider error: {0}")]
    Llm(String),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Note not found: {0}")]
    NoteNotFound(String),
}

pub type Result<T> = std::result::Result<T, KartaError>;
