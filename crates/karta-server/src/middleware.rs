use axum::extract::FromRequestParts;
use axum::http::request::Parts;
use sha2::{Digest, Sha256};

use crate::error::ServerError;
use crate::state::AppState;

/// Authenticated user extracted from Bearer token.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct AuthenticatedUser {
    pub user_id: String,
    pub scope: Option<String>,
}

impl FromRequestParts<AppState> for AuthenticatedUser {
    type Rejection = ServerError;

    async fn from_request_parts(
        parts: &mut Parts,
        state: &AppState,
    ) -> Result<Self, Self::Rejection> {
        let auth_header_value = parts
            .headers
            .get("authorization")
            .ok_or_else(|| ServerError::Unauthorized("Missing Authorization header".to_string()))?;

        let auth_header = auth_header_value.to_str().map_err(|_| {
            ServerError::Unauthorized(
                "Authorization header contains invalid characters".to_string(),
            )
        })?;

        let token = auth_header
            .strip_prefix("Bearer ")
            .ok_or_else(|| ServerError::Unauthorized("Invalid Authorization scheme".to_string()))?;

        let token_hash = hash_token(token);

        let (user_id, scope) = state
            .db
            .validate_access_token(&token_hash)?
            .ok_or_else(|| ServerError::Unauthorized("Invalid or expired token".to_string()))?;

        Ok(AuthenticatedUser { user_id, scope })
    }
}

/// Tower middleware for MCP route protection.
pub async fn validate_token_middleware(
    axum::extract::State(state): axum::extract::State<AppState>,
    request: axum::http::Request<axum::body::Body>,
    next: axum::middleware::Next,
) -> axum::response::Response {
    use axum::response::IntoResponse;

    let auth_header = request
        .headers()
        .get("authorization")
        .and_then(|v| v.to_str().ok());

    let token = match auth_header {
        Some(h) => match h.strip_prefix("Bearer ") {
            Some(t) => t,
            None => return axum::http::StatusCode::UNAUTHORIZED.into_response(),
        },
        None => return axum::http::StatusCode::UNAUTHORIZED.into_response(),
    };

    let token_hash = hash_token(token);
    match state.db.validate_access_token(&token_hash) {
        Ok(Some(_)) => next.run(request).await,
        _ => axum::http::StatusCode::UNAUTHORIZED.into_response(),
    }
}

/// Hash a token with SHA-256 and return hex string.
pub fn hash_token(token: &str) -> String {
    let digest = Sha256::digest(token.as_bytes());
    let mut hex = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write;
        write!(hex, "{byte:02x}").unwrap();
    }
    hex
}
