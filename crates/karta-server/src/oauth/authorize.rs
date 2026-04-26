use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::{Html, IntoResponse, Redirect, Response};
use oauth2::PkceCodeChallenge;
use serde::Deserialize;

use crate::db::PendingAuth;
use crate::error::{Result, ServerError};
use crate::state::AppState;

const VALID_SCOPES: &[&str] = &["read", "write", "admin"];

#[derive(Debug, Deserialize)]
pub struct AuthorizeParams {
    pub response_type: Option<String>,
    pub client_id: Option<String>,
    pub redirect_uri: Option<String>,
    pub code_challenge: Option<String>,
    pub code_challenge_method: Option<String>,
    pub state: Option<String>,
    pub scope: Option<String>,
    pub provider: Option<String>,
}

/// `GET /oauth/authorize` — Authorization endpoint.
pub async fn authorize(
    State(state): State<AppState>,
    Query(params): Query<AuthorizeParams>,
) -> Result<Response> {
    // Validate required parameters
    let response_type = params
        .response_type
        .as_deref()
        .ok_or_else(|| ServerError::BadRequest("response_type is required".to_string()))?;
    if response_type != "code" {
        return Err(ServerError::oauth(
            "unsupported_response_type",
            "Only response_type=code is supported",
            StatusCode::BAD_REQUEST,
        ));
    }

    let client_id = params
        .client_id
        .as_deref()
        .ok_or_else(|| ServerError::BadRequest("client_id is required".to_string()))?;
    let redirect_uri = params
        .redirect_uri
        .as_deref()
        .ok_or_else(|| ServerError::BadRequest("redirect_uri is required".to_string()))?;
    let code_challenge = params
        .code_challenge
        .as_deref()
        .ok_or_else(|| ServerError::BadRequest("code_challenge is required (PKCE)".to_string()))?;
    let code_challenge_method = params.code_challenge_method.as_deref().unwrap_or("S256");
    let client_state = params
        .state
        .as_deref()
        .ok_or_else(|| ServerError::BadRequest("state is required".to_string()))?;

    if code_challenge_method != "S256" {
        return Err(ServerError::oauth(
            "invalid_request",
            "Only code_challenge_method=S256 is supported",
            StatusCode::BAD_REQUEST,
        ));
    }

    // Validate scope if provided
    if let Some(scope) = &params.scope {
        for s in scope.split_whitespace() {
            if !VALID_SCOPES.contains(&s) {
                return Err(ServerError::oauth(
                    "invalid_scope",
                    &format!("Unknown scope: {s}"),
                    StatusCode::BAD_REQUEST,
                ));
            }
        }
    }

    // Validate client exists and redirect_uri matches
    let client = state.db.get_client(client_id)?.ok_or_else(|| {
        ServerError::oauth(
            "invalid_client",
            "Unknown client_id",
            StatusCode::BAD_REQUEST,
        )
    })?;

    if !client.redirect_uris.iter().any(|u| u == redirect_uri) {
        return Err(ServerError::oauth(
            "invalid_request",
            "redirect_uri does not match any registered URI",
            StatusCode::BAD_REQUEST,
        ));
    }

    // If no provider selected, show the login page
    let provider = match &params.provider {
        Some(p) => p.as_str(),
        None => {
            return Ok(render_login_page(&state.config.base_url, &params).into_response());
        }
    };

    if provider != "google" && provider != "github" {
        return Err(ServerError::BadRequest(format!(
            "Unknown provider: {provider}"
        )));
    }

    let idp_csrf = generate_random_string();
    let expires_at = (chrono::Utc::now() + chrono::Duration::minutes(10))
        .format("%Y-%m-%dT%H:%M:%SZ")
        .to_string();

    // Build upstream IdP authorization URL
    let idp_redirect_url = match provider {
        "google" => {
            let (pkce_challenge, pkce_verifier) = PkceCodeChallenge::new_random_sha256();
            let nonce = openidconnect::Nonce::new_random();
            let nonce_secret = nonce.secret().clone();

            let pending = PendingAuth {
                state_token: uuid::Uuid::new_v4().to_string(),
                client_id: client_id.to_string(),
                redirect_uri: redirect_uri.to_string(),
                code_challenge: code_challenge.to_string(),
                code_challenge_method: code_challenge_method.to_string(),
                scope: params.scope.clone(),
                original_state: client_state.to_string(),
                idp_csrf: idp_csrf.clone(),
                idp_nonce: Some(nonce_secret),
                idp_pkce_verifier: pkce_verifier.secret().to_string(),
                provider: "google".to_string(),
                expires_at,
            };
            state.db.insert_pending_auth(&pending)?;

            let (auth_url, _csrf_token, _nonce) = state
                .google_client
                .authorize_url(
                    openidconnect::core::CoreAuthenticationFlow::AuthorizationCode,
                    move || openidconnect::CsrfToken::new(idp_csrf.clone()),
                    move || nonce,
                )
                .add_scope(openidconnect::Scope::new("email".to_string()))
                .add_scope(openidconnect::Scope::new("profile".to_string()))
                .set_pkce_challenge(pkce_challenge)
                .url();

            auth_url.to_string()
        }
        "github" => {
            let (pkce_challenge, pkce_verifier) = PkceCodeChallenge::new_random_sha256();

            let pending = PendingAuth {
                state_token: uuid::Uuid::new_v4().to_string(),
                client_id: client_id.to_string(),
                redirect_uri: redirect_uri.to_string(),
                code_challenge: code_challenge.to_string(),
                code_challenge_method: code_challenge_method.to_string(),
                scope: params.scope.clone(),
                original_state: client_state.to_string(),
                idp_csrf: idp_csrf.clone(),
                idp_nonce: None,
                idp_pkce_verifier: pkce_verifier.secret().to_string(),
                provider: "github".to_string(),
                expires_at,
            };
            state.db.insert_pending_auth(&pending)?;

            let (auth_url, _csrf_token) = state
                .github_client
                .authorize_url(|| oauth2::CsrfToken::new(idp_csrf.clone()))
                .add_scope(oauth2::Scope::new("read:user".to_string()))
                .add_scope(oauth2::Scope::new("user:email".to_string()))
                .set_pkce_challenge(pkce_challenge)
                .url();

            auth_url.to_string()
        }
        _ => unreachable!(),
    };

    Ok(Redirect::temporary(&idp_redirect_url).into_response())
}

fn render_login_page(base_url: &str, params: &AuthorizeParams) -> Html<String> {
    let mut qs = Vec::new();
    if let Some(v) = &params.response_type {
        qs.push(format!("response_type={}", urlencoding::encode(v)));
    }
    if let Some(v) = &params.client_id {
        qs.push(format!("client_id={}", urlencoding::encode(v)));
    }
    if let Some(v) = &params.redirect_uri {
        qs.push(format!("redirect_uri={}", urlencoding::encode(v)));
    }
    if let Some(v) = &params.code_challenge {
        qs.push(format!("code_challenge={}", urlencoding::encode(v)));
    }
    if let Some(v) = &params.code_challenge_method {
        qs.push(format!("code_challenge_method={}", urlencoding::encode(v)));
    }
    if let Some(v) = &params.state {
        qs.push(format!("state={}", urlencoding::encode(v)));
    }
    if let Some(v) = &params.scope {
        qs.push(format!("scope={}", urlencoding::encode(v)));
    }

    let query_string = qs.join("&");
    let google_url = format!("{base_url}/oauth/authorize?{query_string}&provider=google");
    let github_url = format!("{base_url}/oauth/authorize?{query_string}&provider=github");

    Html(format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Karta - Sign In</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            margin: 0;
            background: #f5f5f5;
        }}
        .card {{
            background: white;
            border-radius: 12px;
            padding: 2rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            text-align: center;
            max-width: 360px;
            width: 100%;
        }}
        h1 {{
            margin: 0 0 0.5rem;
            font-size: 1.5rem;
        }}
        p {{
            color: #666;
            margin: 0 0 1.5rem;
        }}
        a.btn {{
            display: block;
            padding: 0.75rem 1rem;
            margin: 0.5rem 0;
            border-radius: 8px;
            text-decoration: none;
            font-weight: 500;
            font-size: 1rem;
        }}
        .btn-google {{
            background: #4285f4;
            color: white;
        }}
        .btn-github {{
            background: #24292e;
            color: white;
        }}
        .btn:hover {{
            opacity: 0.9;
        }}
    </style>
</head>
<body>
    <div class="card">
        <h1>Sign in to Karta</h1>
        <p>Choose your identity provider</p>
        <a class="btn btn-google" href="{google_url}">Sign in with Google</a>
        <a class="btn btn-github" href="{github_url}">Sign in with GitHub</a>
    </div>
</body>
</html>"#
    ))
}

fn generate_random_string() -> String {
    use base64::Engine;
    let bytes: [u8; 32] = rand::random();
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes)
}
