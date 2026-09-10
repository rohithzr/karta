use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde_json::json;

pub type Result<T> = std::result::Result<T, ServerError>;

#[derive(Debug, thiserror::Error)]
pub enum ServerError {
    #[error("Database error: {0}")]
    Db(#[from] rusqlite::Error),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Invalid request: {0}")]
    BadRequest(String),

    #[error("Unauthorized: {0}")]
    Unauthorized(String),

    #[error("Not found: {0}")]
    #[allow(dead_code)]
    NotFound(String),

    #[error("OAuth error: {error}")]
    OAuth {
        error: String,
        error_description: String,
        status: StatusCode,
    },

    #[error("Upstream IdP error: {0}")]
    IdpError(String),

    #[error("Internal error: {0}")]
    Internal(String),
}

impl ServerError {
    pub fn oauth(error: &str, description: &str, status: StatusCode) -> Self {
        Self::OAuth {
            error: error.to_string(),
            error_description: description.to_string(),
            status,
        }
    }
}

impl IntoResponse for ServerError {
    fn into_response(self) -> Response {
        let (status, body) = match &self {
            ServerError::Db(e) => {
                tracing::error!("Database error: {e}");
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    json!({"error": "server_error", "error_description": "Internal server error"}),
                )
            }
            ServerError::Config(msg) => {
                tracing::error!("Config error: {msg}");
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    json!({"error": "server_error", "error_description": "Server misconfiguration"}),
                )
            }
            ServerError::BadRequest(msg) => {
                tracing::warn!("Bad request: {msg}");
                (
                    StatusCode::BAD_REQUEST,
                    json!({"error": "invalid_request", "error_description": msg}),
                )
            }
            ServerError::Unauthorized(msg) => {
                tracing::warn!("Unauthorized: {msg}");
                (
                    StatusCode::UNAUTHORIZED,
                    json!({"error": "invalid_token", "error_description": msg}),
                )
            }
            ServerError::NotFound(msg) => (
                StatusCode::NOT_FOUND,
                json!({"error": "not_found", "error_description": msg}),
            ),
            ServerError::OAuth {
                error,
                error_description,
                status,
            } => {
                tracing::warn!(error = %error, description = %error_description, "OAuth error");
                (
                    *status,
                    json!({"error": error, "error_description": error_description}),
                )
            }
            ServerError::IdpError(msg) => {
                tracing::error!("IdP error: {msg}");
                (
                    StatusCode::BAD_GATEWAY,
                    json!({"error": "server_error", "error_description": "Upstream identity provider error"}),
                )
            }
            ServerError::Internal(msg) => {
                tracing::error!("Internal error: {msg}");
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    json!({"error": "server_error", "error_description": "Internal server error"}),
                )
            }
        };

        (status, axum::Json(body)).into_response()
    }
}
