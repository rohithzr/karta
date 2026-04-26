use crate::error::{Result, ServerError};

#[derive(Clone)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub base_url: String,
    pub cookie_secret: Vec<u8>,
    pub google_client_id: String,
    pub google_client_secret: String,
    pub github_client_id: String,
    pub github_client_secret: String,
    pub db_path: String,
    pub registration_token: Option<String>,
    pub allowed_origins: Vec<String>,
}

impl std::fmt::Debug for ServerConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ServerConfig")
            .field("host", &self.host)
            .field("port", &self.port)
            .field("base_url", &self.base_url)
            .field("cookie_secret", &"[REDACTED]")
            .field("google_client_id", &self.google_client_id)
            .field("google_client_secret", &"[REDACTED]")
            .field("github_client_id", &self.github_client_id)
            .field("github_client_secret", &"[REDACTED]")
            .field("db_path", &self.db_path)
            .field(
                "registration_token",
                &self.registration_token.as_ref().map(|_| "[REDACTED]"),
            )
            .field("allowed_origins", &self.allowed_origins)
            .finish()
    }
}

impl ServerConfig {
    pub fn from_env() -> Result<Self> {
        let host = std::env::var("KARTA_HOST").unwrap_or_else(|_| "0.0.0.0".to_string());
        let port = std::env::var("KARTA_PORT")
            .unwrap_or_else(|_| "3000".to_string())
            .parse::<u16>()
            .map_err(|e| ServerError::Config(format!("Invalid KARTA_PORT: {e}")))?;
        let base_url = required_env("KARTA_BASE_URL")?;
        let cookie_secret_b64 = required_env("KARTA_COOKIE_SECRET")?;
        let cookie_secret = base64::Engine::decode(
            &base64::engine::general_purpose::URL_SAFE_NO_PAD,
            &cookie_secret_b64,
        )
        .or_else(|_| {
            base64::Engine::decode(
                &base64::engine::general_purpose::STANDARD,
                &cookie_secret_b64,
            )
        })
        .map_err(|e| ServerError::Config(format!("Invalid KARTA_COOKIE_SECRET base64: {e}")))?;
        if cookie_secret.len() < 64 {
            return Err(ServerError::Config(
                "KARTA_COOKIE_SECRET must decode to at least 64 bytes (88 base64 chars)"
                    .to_string(),
            ));
        }

        let google_client_id = required_env("GOOGLE_CLIENT_ID")?;
        let google_client_secret = required_env("GOOGLE_CLIENT_SECRET")?;
        let github_client_id = required_env("GITHUB_CLIENT_ID")?;
        let github_client_secret = required_env("GITHUB_CLIENT_SECRET")?;

        let db_path =
            std::env::var("KARTA_AUTH_DB_PATH").unwrap_or_else(|_| "karta-auth.db".to_string());

        let registration_token = std::env::var("KARTA_REGISTRATION_TOKEN").ok();

        let allowed_origins = std::env::var("KARTA_ALLOWED_ORIGINS")
            .map(|v| {
                v.split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect()
            })
            .unwrap_or_else(|_| vec![base_url.clone()]);

        Ok(Self {
            host,
            port,
            base_url,
            cookie_secret,
            google_client_id,
            google_client_secret,
            github_client_id,
            github_client_secret,
            db_path,
            registration_token,
            allowed_origins,
        })
    }
}

fn required_env(name: &str) -> Result<String> {
    std::env::var(name).map_err(|_| ServerError::Config(format!("{name} is required")))
}
