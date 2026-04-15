use axum_extra::extract::cookie::Key;
use oauth2::basic::BasicClient;
use oauth2::{AuthUrl, ClientId, ClientSecret, EndpointMaybeSet, EndpointNotSet, EndpointSet, RedirectUrl, TokenUrl};
use openidconnect::core::CoreClient;
use openidconnect::{IssuerUrl, ClientId as OidcClientId, ClientSecret as OidcClientSecret};

use crate::config::ServerConfig;
use crate::db::AuthDb;
use crate::error::{Result, ServerError};

/// The Google OIDC client type after discovery (auth URL set, token URL maybe set).
pub type GoogleClient = CoreClient<
    EndpointSet,
    EndpointNotSet,
    EndpointNotSet,
    EndpointNotSet,
    EndpointMaybeSet,
    EndpointMaybeSet,
>;

/// The GitHub OAuth2 client type with auth + token URLs set.
pub type GithubClient = BasicClient<
    EndpointSet,
    EndpointNotSet,
    EndpointNotSet,
    EndpointNotSet,
    EndpointSet,
>;

#[derive(Clone)]
#[allow(dead_code)]
pub struct AppState {
    pub db: AuthDb,
    pub google_client: GoogleClient,
    pub github_client: GithubClient,
    pub config: ServerConfig,
    pub http_client: reqwest::Client,
    pub cookie_key: Key,
}

impl AppState {
    pub async fn new(config: ServerConfig, db: AuthDb) -> Result<Self> {
        let http_client = reqwest::ClientBuilder::new()
            .redirect(reqwest::redirect::Policy::none())
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .map_err(|e| ServerError::Internal(format!("Failed to build HTTP client: {e}")))?;

        let google_client = build_google_client(&config, &http_client).await?;
        let github_client = build_github_client(&config)?;
        let cookie_key = Key::from(&config.cookie_secret);

        Ok(Self {
            db,
            google_client,
            github_client,
            config,
            http_client,
            cookie_key,
        })
    }
}

async fn build_google_client(
    config: &ServerConfig,
    http_client: &reqwest::Client,
) -> Result<GoogleClient> {
    let issuer = IssuerUrl::new("https://accounts.google.com".to_string())
        .map_err(|e| ServerError::Config(format!("Invalid Google issuer URL: {e}")))?;

    let provider_metadata = openidconnect::core::CoreProviderMetadata::discover_async(
        issuer,
        http_client,
    )
    .await
    .map_err(|e| ServerError::Config(format!("Failed to discover Google OIDC: {e}")))?;

    let redirect_url = format!("{}/auth/google/callback", config.base_url);

    let client = CoreClient::from_provider_metadata(
        provider_metadata,
        OidcClientId::new(config.google_client_id.clone()),
        Some(OidcClientSecret::new(config.google_client_secret.clone())),
    )
    .set_redirect_uri(
        openidconnect::RedirectUrl::new(redirect_url)
            .map_err(|e| ServerError::Config(format!("Invalid Google redirect URL: {e}")))?,
    );

    Ok(client)
}

fn build_github_client(config: &ServerConfig) -> Result<GithubClient> {
    let redirect_url = format!("{}/auth/github/callback", config.base_url);

    let client = BasicClient::new(ClientId::new(config.github_client_id.clone()))
        .set_client_secret(ClientSecret::new(config.github_client_secret.clone()))
        .set_auth_uri(
            AuthUrl::new("https://github.com/login/oauth/authorize".to_string())
                .map_err(|e| ServerError::Config(format!("Invalid GitHub auth URL: {e}")))?,
        )
        .set_token_uri(
            TokenUrl::new("https://github.com/login/oauth/access_token".to_string())
                .map_err(|e| ServerError::Config(format!("Invalid GitHub token URL: {e}")))?,
        )
        .set_redirect_uri(
            RedirectUrl::new(redirect_url)
                .map_err(|e| ServerError::Config(format!("Invalid GitHub redirect URL: {e}")))?,
        );

    Ok(client)
}
