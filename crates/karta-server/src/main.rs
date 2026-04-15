mod config;
mod db;
mod error;
mod mcp;
mod middleware;
mod oauth;
mod routes;
mod state;

use std::sync::Arc;

use axum::Router;
use axum::http::{Method, HeaderValue, header};
use axum::routing::{get, post};
use tower_http::cors::{CorsLayer, AllowOrigin, Any as CorsAny};
use tower_http::trace::TraceLayer;
use rmcp::transport::streamable_http_server::{
    StreamableHttpService, StreamableHttpServerConfig,
    session::local::LocalSessionManager,
};

use config::ServerConfig;
use db::AuthDb;
use karta_core::Karta;
use state::AppState;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "karta_server=info,tower_http=info".into()),
        )
        .init();

    let config = ServerConfig::from_env()?;
    tracing::info!(host = %config.host, port = %config.port, "Starting karta-server");

    // Initialize auth database
    let db = AuthDb::new(&config.db_path)?;

    // Background cleanup task
    let cleanup_db = db.clone();
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(600));
        loop {
            interval.tick().await;
            match cleanup_db.cleanup_expired() {
                Ok(n) => {
                    if n > 0 {
                        tracing::info!(cleaned = n, "Expired token cleanup");
                    }
                }
                Err(e) => tracing::error!("Token cleanup failed: {e}"),
            }
        }
    });

    // Build OAuth application state
    let app_state = AppState::new(config.clone(), db).await?;

    // Initialize karta-core
    let karta = {
        let mut karta_config = karta_core::config::KartaConfig::default();
        if let Ok(lance_uri) = std::env::var("KARTA_LANCE_URI") {
            karta_config.storage.lance_uri = Some(lance_uri);
        }
        match Karta::with_defaults(karta_config).await {
            Ok(k) => {
                tracing::info!("karta-core initialized");
                Some(Arc::new(k))
            }
            Err(e) => {
                tracing::warn!("karta-core not available (MCP tools disabled): {e}");
                None
            }
        }
    };

    // Build CORS for OAuth endpoints
    let origins: Vec<HeaderValue> = config
        .allowed_origins
        .iter()
        .filter_map(|o| o.parse::<HeaderValue>().ok())
        .collect();
    let oauth_cors = CorsLayer::new()
        .allow_origin(AllowOrigin::list(origins))
        .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
        .allow_headers([header::AUTHORIZATION, header::CONTENT_TYPE, header::ACCEPT]);

    // MCP CORS needs to be more permissive (Claude.ai sends various headers)
    let mcp_cors = CorsLayer::new()
        .allow_origin(CorsAny)
        .allow_methods(CorsAny)
        .allow_headers(CorsAny);

    // OAuth routes (unprotected)
    let oauth_routes = Router::new()
        .route(
            "/.well-known/oauth-authorization-server",
            get(oauth::discovery::oauth_metadata),
        )
        .route("/oauth/register", post(oauth::register::register_client))
        .route("/oauth/authorize", get(oauth::authorize::authorize))
        .route("/oauth/token", post(oauth::token::token))
        .route(
            "/auth/google/callback",
            get(oauth::callback::google_callback),
        )
        .route(
            "/auth/github/callback",
            get(oauth::callback::github_callback),
        )
        .layer(oauth_cors)
        .with_state(app_state.clone());

    // Static icon endpoint
    let icon_route = Router::new().route("/icon.svg", get(routes::icon_svg));

    // Health check (protected)
    let health_route = Router::new()
        .route("/api/health", get(routes::health))
        .with_state(app_state.clone());

    // Build final router
    let mut app = Router::new()
        .merge(oauth_routes)
        .merge(icon_route)
        .merge(health_route);

    // Add MCP service if karta-core is available
    if let Some(karta_ref) = karta {
        // Extract hostname from base URL for DNS rebinding protection
        let allowed_host = url::Url::parse(&config.base_url)
            .ok()
            .and_then(|u| u.host_str().map(String::from))
            .unwrap_or_else(|| "localhost".to_string());

        let mcp_config = StreamableHttpServerConfig::default()
            .with_allowed_hosts([
                "localhost".to_string(),
                "127.0.0.1".to_string(),
                allowed_host,
            ]);

        let base_url = config.base_url.clone();
        let mcp_service: StreamableHttpService<mcp::KartaService, LocalSessionManager> =
            StreamableHttpService::new(
                move || Ok(mcp::KartaService::new(karta_ref.clone(), base_url.clone())),
                LocalSessionManager::default().into(),
                mcp_config,
            );

        // Protected MCP routes (require Bearer token)
        // In axum, last .layer() is outermost (runs first on request).
        // CORS must be outermost so OPTIONS preflight isn't blocked by auth.
        let protected_mcp = Router::new()
            .nest_service("/mcp", mcp_service)
            .layer(axum::middleware::from_fn_with_state(
                app_state.clone(),
                middleware::validate_token_middleware,
            ))
            .layer(mcp_cors);

        app = app.merge(protected_mcp);
        tracing::info!("MCP endpoint enabled at /mcp");
    }

    app = app.layer(TraceLayer::new_for_http());

    let addr = format!("{}:{}", config.host, config.port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    tracing::info!(addr = %addr, "Listening");
    axum::serve(listener, app).await?;

    Ok(())
}
