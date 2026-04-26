use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::db::OAuthClient;
use crate::error::{Result, ServerError};
use crate::state::AppState;

#[derive(Debug, Deserialize)]
pub struct RegisterRequest {
    pub redirect_uris: Vec<String>,
    pub client_name: Option<String>,
    pub token_endpoint_auth_method: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct RegisterResponse {
    pub client_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub client_secret: Option<String>,
    pub redirect_uris: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub client_name: Option<String>,
    pub token_endpoint_auth_method: String,
}

/// `POST /oauth/register` — Dynamic Client Registration (RFC 7591).
pub async fn register_client(
    State(state): State<AppState>,
    headers: axum::http::HeaderMap,
    Json(req): Json<RegisterRequest>,
) -> Result<(StatusCode, Json<RegisterResponse>)> {
    // Check registration token if configured
    if let Some(expected_token) = &state.config.registration_token {
        let provided = headers
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.strip_prefix("Bearer "))
            .ok_or_else(|| {
                ServerError::Unauthorized(
                    "Registration requires Authorization: Bearer <token>".to_string(),
                )
            })?;
        if provided != expected_token {
            return Err(ServerError::Unauthorized(
                "Invalid registration token".to_string(),
            ));
        }
    }

    if req.redirect_uris.is_empty() {
        return Err(ServerError::BadRequest(
            "redirect_uris must not be empty".to_string(),
        ));
    }

    // Validate all redirect URIs are valid URLs with appropriate schemes
    for uri in &req.redirect_uris {
        let parsed = url::Url::parse(uri)
            .map_err(|_| ServerError::BadRequest(format!("Invalid redirect_uri: {uri}")))?;
        let is_loopback = parsed
            .host_str()
            .is_some_and(|h| h == "localhost" || h == "127.0.0.1");
        if parsed.scheme() != "https" && !(parsed.scheme() == "http" && is_loopback) {
            return Err(ServerError::BadRequest(
                "redirect_uri must use HTTPS (HTTP allowed only for localhost)".to_string(),
            ));
        }
    }

    let auth_method = req.token_endpoint_auth_method.as_deref().unwrap_or("none");

    if auth_method != "none" && auth_method != "client_secret_post" {
        return Err(ServerError::BadRequest(format!(
            "Unsupported token_endpoint_auth_method: {auth_method}. Supported: none, client_secret_post"
        )));
    }

    let client_id = uuid::Uuid::new_v4().to_string();

    let (client_secret, client_secret_hash) = if auth_method == "client_secret_post" {
        let secret = generate_random_string();
        let hash = sha256_hex(&secret);
        (Some(secret), Some(hash))
    } else {
        (None, None)
    };

    let client = OAuthClient {
        client_id: client_id.clone(),
        client_secret_hash,
        redirect_uris: req.redirect_uris.clone(),
        client_name: req.client_name.clone(),
    };

    state.db.insert_client(&client)?;

    tracing::info!(client_id = %client_id, "Registered new OAuth client");

    Ok((
        StatusCode::CREATED,
        Json(RegisterResponse {
            client_id,
            client_secret,
            redirect_uris: req.redirect_uris,
            client_name: req.client_name,
            token_endpoint_auth_method: auth_method.to_string(),
        }),
    ))
}

fn generate_random_string() -> String {
    use base64::Engine;
    let bytes: [u8; 32] = rand::random();
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes)
}

fn sha256_hex(input: &str) -> String {
    let digest = Sha256::digest(input.as_bytes());
    let mut hex = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write;
        write!(hex, "{byte:02x}").unwrap();
    }
    hex
}
