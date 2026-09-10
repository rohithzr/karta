use axum::extract::State;
use axum::Json;
use serde_json::{Value, json};

use crate::state::AppState;

/// `GET /.well-known/oauth-authorization-server`
/// Returns OAuth 2.0 Authorization Server Metadata per RFC 8414.
pub async fn oauth_metadata(State(state): State<AppState>) -> Json<Value> {
    let base = state.config.base_url.trim_end_matches('/');
    Json(json!({
        "issuer": base,
        "authorization_endpoint": format!("{base}/oauth/authorize"),
        "token_endpoint": format!("{base}/oauth/token"),
        "registration_endpoint": format!("{base}/oauth/register"),
        "response_types_supported": ["code"],
        "grant_types_supported": ["authorization_code", "refresh_token"],
        "code_challenge_methods_supported": ["S256"],
        "token_endpoint_auth_methods_supported": ["client_secret_post", "none"],
        "scopes_supported": ["read", "write", "admin"]
    }))
}
