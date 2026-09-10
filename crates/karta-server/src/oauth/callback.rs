use axum::extract::{Query, State};
use axum::response::{IntoResponse, Redirect, Response};
use oauth2::{AuthorizationCode, PkceCodeVerifier};
use oauth2::TokenResponse as _;  // for access_token() on GitHub response
use openidconnect::TokenResponse as _; // for id_token() on Google response
use serde::Deserialize;

use crate::db::AuthCode;
use crate::error::{Result, ServerError};
use crate::state::AppState;

#[derive(Debug, Deserialize)]
pub struct CallbackParams {
    pub code: Option<String>,
    pub state: Option<String>,
    pub error: Option<String>,
    pub error_description: Option<String>,
}

/// `GET /auth/google/callback` — Google OIDC callback.
pub async fn google_callback(
    State(state): State<AppState>,
    Query(params): Query<CallbackParams>,
) -> Result<Response> {
    if let Some(err) = &params.error {
        let desc = params.error_description.as_deref().unwrap_or("Unknown error");
        return Err(ServerError::IdpError(format!("Google OAuth error: {err}: {desc}")));
    }

    let idp_code = params
        .code
        .as_deref()
        .ok_or_else(|| ServerError::BadRequest("Missing code parameter".to_string()))?;
    let idp_state = params
        .state
        .as_deref()
        .ok_or_else(|| ServerError::BadRequest("Missing state parameter".to_string()))?;

    // Look up pending auth request by IdP CSRF token
    let pending = state
        .db
        .consume_pending_auth(idp_state)?
        .ok_or_else(|| ServerError::BadRequest("Unknown or expired state".to_string()))?;

    if pending.provider != "google" {
        return Err(ServerError::BadRequest("State does not match Google provider".to_string()));
    }

    // Check expiry
    let expires = chrono::NaiveDateTime::parse_from_str(&pending.expires_at, "%Y-%m-%dT%H:%M:%SZ")
        .map_err(|e| ServerError::Internal(format!("Bad expiry: {e}")))?;
    if expires.and_utc() < chrono::Utc::now() {
        return Err(ServerError::BadRequest("Authorization request expired".to_string()));
    }

    // Exchange code for tokens at Google
    let pkce_verifier = PkceCodeVerifier::new(pending.idp_pkce_verifier.clone());

    let token_response = state
        .google_client
        .exchange_code(openidconnect::AuthorizationCode::new(idp_code.to_string()))
        .map_err(|e| ServerError::IdpError(format!("Google token exchange config error: {e}")))?
        .set_pkce_verifier(pkce_verifier)
        .request_async(&state.http_client)
        .await
        .map_err(|e| ServerError::IdpError(format!("Google token exchange failed: {e}")))?;

    // Validate ID token
    let id_token = token_response
        .id_token()
        .ok_or_else(|| ServerError::IdpError("Google did not return an ID token".to_string()))?;

    let nonce = pending
        .idp_nonce
        .as_ref()
        .map(|n| openidconnect::Nonce::new(n.clone()))
        .ok_or_else(|| ServerError::Internal("Missing nonce for Google OIDC".to_string()))?;

    let verifier = state.google_client.id_token_verifier();
    let claims = id_token
        .claims(&verifier, &nonce)
        .map_err(|e| ServerError::IdpError(format!("Google ID token validation failed: {e}")))?;

    let sub = claims.subject().to_string();
    let email: Option<String> = claims.email().map(|e| e.to_string());
    let name: Option<String> = claims
        .given_name()
        .and_then(|names| names.get(None))
        .map(|n| n.to_string())
        .or_else(|| {
            claims
                .name()
                .and_then(|names| names.get(None))
                .map(|n| n.to_string())
        });

    // Upsert user
    let user = state.db.upsert_user("google", &sub, email.as_deref(), name.as_deref())?;

    // Generate authorization code for the downstream client
    let auth_code = generate_auth_code(&state, &pending, &user.id)?;

    // Redirect to client's redirect_uri
    let redirect_url = build_client_redirect(&pending.redirect_uri, &auth_code, &pending.original_state);
    tracing::info!(user_id = %user.id, provider = "google", "User authenticated");

    Ok(Redirect::temporary(&redirect_url).into_response())
}

/// `GET /auth/github/callback` — GitHub OAuth callback.
pub async fn github_callback(
    State(state): State<AppState>,
    Query(params): Query<CallbackParams>,
) -> Result<Response> {
    if let Some(err) = &params.error {
        let desc = params.error_description.as_deref().unwrap_or("Unknown error");
        return Err(ServerError::IdpError(format!("GitHub OAuth error: {err}: {desc}")));
    }

    let idp_code = params
        .code
        .as_deref()
        .ok_or_else(|| ServerError::BadRequest("Missing code parameter".to_string()))?;
    let idp_state = params
        .state
        .as_deref()
        .ok_or_else(|| ServerError::BadRequest("Missing state parameter".to_string()))?;

    // Look up pending auth request
    let pending = state
        .db
        .consume_pending_auth(idp_state)?
        .ok_or_else(|| ServerError::BadRequest("Unknown or expired state".to_string()))?;

    if pending.provider != "github" {
        return Err(ServerError::BadRequest("State does not match GitHub provider".to_string()));
    }

    // Check expiry
    let expires = chrono::NaiveDateTime::parse_from_str(&pending.expires_at, "%Y-%m-%dT%H:%M:%SZ")
        .map_err(|e| ServerError::Internal(format!("Bad expiry: {e}")))?;
    if expires.and_utc() < chrono::Utc::now() {
        return Err(ServerError::BadRequest("Authorization request expired".to_string()));
    }

    // Exchange code for tokens at GitHub
    let pkce_verifier = PkceCodeVerifier::new(pending.idp_pkce_verifier.clone());

    let token_response = state
        .github_client
        .exchange_code(AuthorizationCode::new(idp_code.to_string()))
        .set_pkce_verifier(pkce_verifier)
        .request_async(&state.http_client)
        .await
        .map_err(|e| ServerError::IdpError(format!("GitHub token exchange failed: {e}")))?;

    let access_token = token_response.access_token().secret().clone();

    // Fetch user info from GitHub API with timeout
    let resp = state
        .http_client
        .get("https://api.github.com/user")
        .header("Authorization", format!("Bearer {access_token}"))
        .header("User-Agent", "karta-server")
        .header("Accept", "application/json")
        .timeout(std::time::Duration::from_secs(10))
        .send()
        .await
        .map_err(|e| ServerError::IdpError(format!("GitHub user API failed: {e}")))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Err(ServerError::IdpError(format!(
            "GitHub user API returned {status}: {body}"
        )));
    }

    let github_user: GitHubUser = resp
        .json()
        .await
        .map_err(|e| ServerError::IdpError(format!("GitHub user API parse failed: {e}")))?;

    let sub = github_user.id.to_string();
    let email = github_user.email;
    let name = github_user.name.or(github_user.login);

    // Upsert user
    let user = state.db.upsert_user("github", &sub, email.as_deref(), name.as_deref())?;

    // Generate authorization code
    let auth_code = generate_auth_code(&state, &pending, &user.id)?;

    // Redirect to client's redirect_uri
    let redirect_url = build_client_redirect(&pending.redirect_uri, &auth_code, &pending.original_state);
    tracing::info!(user_id = %user.id, provider = "github", "User authenticated");

    Ok(Redirect::temporary(&redirect_url).into_response())
}

#[derive(Debug, Deserialize)]
struct GitHubUser {
    id: u64,
    login: Option<String>,
    name: Option<String>,
    email: Option<String>,
}

fn generate_auth_code(
    state: &AppState,
    pending: &crate::db::PendingAuth,
    user_id: &str,
) -> Result<String> {
    use base64::Engine;
    let code_bytes: [u8; 32] = rand::random();
    let code = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(code_bytes);

    let expires_at = (chrono::Utc::now() + chrono::Duration::minutes(10))
        .format("%Y-%m-%dT%H:%M:%SZ")
        .to_string();

    let auth_code = AuthCode {
        code: code.clone(),
        client_id: pending.client_id.clone(),
        user_id: user_id.to_string(),
        redirect_uri: pending.redirect_uri.clone(),
        code_challenge: pending.code_challenge.clone(),
        code_challenge_method: pending.code_challenge_method.clone(),
        scope: pending.scope.clone(),
        expires_at,
    };
    state.db.insert_auth_code(&auth_code)?;

    Ok(code)
}

fn build_client_redirect(redirect_uri: &str, code: &str, state: &str) -> String {
    format!(
        "{redirect_uri}?code={}&state={}",
        urlencoding::encode(code),
        urlencoding::encode(state),
    )
}
