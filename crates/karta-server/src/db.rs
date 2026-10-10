use rusqlite::{Connection, params};
use std::sync::Mutex;

use crate::error::{Result, ServerError};

/// OAuth client registered via dynamic client registration.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OAuthClient {
    pub client_id: String,
    pub client_secret_hash: Option<String>,
    pub redirect_uris: Vec<String>,
    pub client_name: Option<String>,
}

/// User created on first login via an upstream IdP.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct User {
    pub id: String,
    pub provider: String,
    pub provider_sub: String,
    pub email: Option<String>,
    pub display_name: Option<String>,
}

/// Short-lived authorization code.
#[derive(Debug, Clone)]
pub struct AuthCode {
    pub code: String,
    pub client_id: String,
    pub user_id: String,
    pub redirect_uri: String,
    pub code_challenge: String,
    pub code_challenge_method: String,
    pub scope: Option<String>,
    pub expires_at: String,
}

/// Pending authorization request stored while user logs in via IdP.
#[derive(Debug, Clone)]
pub struct PendingAuth {
    pub state_token: String,
    pub client_id: String,
    pub redirect_uri: String,
    pub code_challenge: String,
    pub code_challenge_method: String,
    pub scope: Option<String>,
    pub original_state: String,
    pub idp_csrf: String,
    pub idp_nonce: Option<String>,
    pub idp_pkce_verifier: String,
    pub provider: String,
    pub expires_at: String,
}

/// SQLite-backed auth database (matches karta-core's Mutex<Connection> pattern).
#[derive(Clone)]
pub struct AuthDb {
    conn: std::sync::Arc<Mutex<Connection>>,
}

impl AuthDb {
    pub fn new(path: &str) -> Result<Self> {
        let conn = Connection::open(path)
            .map_err(|e| ServerError::Internal(format!("Failed to open auth db: {e}")))?;

        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON;")
            .map_err(|e| ServerError::Internal(format!("Failed to set pragmas: {e}")))?;

        let db = Self {
            conn: std::sync::Arc::new(Mutex::new(conn)),
        };
        db.init()?;
        Ok(db)
    }

    fn lock(&self) -> Result<std::sync::MutexGuard<'_, Connection>> {
        self.conn
            .lock()
            .map_err(|e| ServerError::Internal(format!("DB mutex poisoned: {e}")))
    }

    fn init(&self) -> Result<()> {
        let conn = self.lock()?;
        conn.execute_batch(
            "
            CREATE TABLE IF NOT EXISTS oauth_clients (
                client_id TEXT PRIMARY KEY,
                client_secret_hash TEXT,
                redirect_uris TEXT NOT NULL,
                client_name TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                provider TEXT NOT NULL,
                provider_sub TEXT NOT NULL,
                email TEXT,
                display_name TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                UNIQUE(provider, provider_sub)
            );

            CREATE TABLE IF NOT EXISTS auth_codes (
                code TEXT PRIMARY KEY,
                client_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                redirect_uri TEXT NOT NULL,
                code_challenge TEXT NOT NULL,
                code_challenge_method TEXT NOT NULL DEFAULT 'S256',
                scope TEXT,
                expires_at TEXT NOT NULL,
                used INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS access_tokens (
                token_hash TEXT PRIMARY KEY,
                client_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                scope TEXT,
                expires_at TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS refresh_tokens (
                token_hash TEXT PRIMARY KEY,
                client_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                scope TEXT,
                expires_at TEXT NOT NULL,
                revoked INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS pending_auth_requests (
                state_token TEXT PRIMARY KEY,
                client_id TEXT NOT NULL,
                redirect_uri TEXT NOT NULL,
                code_challenge TEXT NOT NULL,
                code_challenge_method TEXT NOT NULL,
                scope TEXT,
                original_state TEXT NOT NULL,
                idp_csrf TEXT NOT NULL UNIQUE,
                idp_nonce TEXT,
                idp_pkce_verifier TEXT NOT NULL,
                provider TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            ",
        )?;
        Ok(())
    }

    // ── Clients ──────────────────────────────────────────────

    pub fn insert_client(&self, client: &OAuthClient) -> Result<()> {
        let conn = self.lock()?;
        let uris_json = serde_json::to_string(&client.redirect_uris)
            .map_err(|e| ServerError::Internal(e.to_string()))?;
        conn.execute(
            "INSERT INTO oauth_clients (client_id, client_secret_hash, redirect_uris, client_name)
             VALUES (?1, ?2, ?3, ?4)",
            params![
                client.client_id,
                client.client_secret_hash,
                uris_json,
                client.client_name,
            ],
        )?;
        Ok(())
    }

    pub fn get_client(&self, client_id: &str) -> Result<Option<OAuthClient>> {
        let conn = self.lock()?;
        let mut stmt = conn.prepare(
            "SELECT client_id, client_secret_hash, redirect_uris, client_name
             FROM oauth_clients WHERE client_id = ?1",
        )?;
        let mut rows = stmt.query(params![client_id])?;
        match rows.next()? {
            Some(row) => {
                let uris_json: String = row.get(2)?;
                let redirect_uris: Vec<String> = serde_json::from_str(&uris_json)
                    .map_err(|e| ServerError::Internal(e.to_string()))?;
                Ok(Some(OAuthClient {
                    client_id: row.get(0)?,
                    client_secret_hash: row.get(1)?,
                    redirect_uris,
                    client_name: row.get(3)?,
                }))
            }
            None => Ok(None),
        }
    }

    // ── Users ────────────────────────────────────────────────

    pub fn upsert_user(
        &self,
        provider: &str,
        provider_sub: &str,
        email: Option<&str>,
        display_name: Option<&str>,
    ) -> Result<User> {
        let conn = self.lock()?;
        // Try to find existing user
        let mut stmt = conn.prepare(
            "SELECT id, provider, provider_sub, email, display_name
             FROM users WHERE provider = ?1 AND provider_sub = ?2",
        )?;
        let existing: Option<User> = stmt
            .query(params![provider, provider_sub])?
            .next()?
            .map(|row| -> rusqlite::Result<User> {
                Ok(User {
                    id: row.get(0)?,
                    provider: row.get(1)?,
                    provider_sub: row.get(2)?,
                    email: row.get(3)?,
                    display_name: row.get(4)?,
                })
            })
            .transpose()?;

        if let Some(mut user) = existing {
            // Update email/name if they changed
            conn.execute(
                "UPDATE users SET email = COALESCE(?1, email), display_name = COALESCE(?2, display_name)
                 WHERE id = ?3",
                params![email, display_name, user.id],
            )?;
            user.email = email.map(String::from).or(user.email);
            user.display_name = display_name.map(String::from).or(user.display_name);
            Ok(user)
        } else {
            let id = uuid::Uuid::new_v4().to_string();
            conn.execute(
                "INSERT INTO users (id, provider, provider_sub, email, display_name)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                params![id, provider, provider_sub, email, display_name],
            )?;
            Ok(User {
                id,
                provider: provider.to_string(),
                provider_sub: provider_sub.to_string(),
                email: email.map(String::from),
                display_name: display_name.map(String::from),
            })
        }
    }

    // ── Auth codes ───────────────────────────────────────────

    pub fn insert_auth_code(&self, code: &AuthCode) -> Result<()> {
        let conn = self.lock()?;
        let code_hash = crate::middleware::hash_token(&code.code);
        conn.execute(
            "INSERT INTO auth_codes (code, client_id, user_id, redirect_uri, code_challenge, code_challenge_method, scope, expires_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                code_hash,
                code.client_id,
                code.user_id,
                code.redirect_uri,
                code.code_challenge,
                code.code_challenge_method,
                code.scope,
                code.expires_at,
            ],
        )?;
        Ok(())
    }

    /// Consume an auth code atomically (one-time use). Returns None if not found, already used, or expired.
    pub fn consume_auth_code(&self, code: &str) -> Result<Option<AuthCode>> {
        let conn = self.lock()?;
        let now = chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string();
        let code_hash = crate::middleware::hash_token(code);

        // Atomically mark the code as used, checking used=0 and expiry in the WHERE clause
        conn.execute(
            "UPDATE auth_codes SET used = 1 WHERE code = ?1 AND used = 0 AND expires_at >= ?2",
            params![code_hash, now],
        )?;

        if conn.changes() == 0 {
            return Ok(None);
        }

        // Read back the consumed row
        let mut stmt = conn.prepare(
            "SELECT code, client_id, user_id, redirect_uri, code_challenge, code_challenge_method, scope, expires_at
             FROM auth_codes WHERE code = ?1",
        )?;
        let result: Option<AuthCode> = stmt
            .query(params![code_hash])?
            .next()?
            .map(|row| -> rusqlite::Result<AuthCode> {
                Ok(AuthCode {
                    code: row.get(0)?,
                    client_id: row.get(1)?,
                    user_id: row.get(2)?,
                    redirect_uri: row.get(3)?,
                    code_challenge: row.get(4)?,
                    code_challenge_method: row.get(5)?,
                    scope: row.get(6)?,
                    expires_at: row.get(7)?,
                })
            })
            .transpose()?;

        Ok(result)
    }

    // ── Access tokens ────────────────────────────────────────

    pub fn insert_access_token(
        &self,
        token_hash: &str,
        client_id: &str,
        user_id: &str,
        scope: Option<&str>,
        expires_at: &str,
    ) -> Result<()> {
        let conn = self.lock()?;
        conn.execute(
            "INSERT INTO access_tokens (token_hash, client_id, user_id, scope, expires_at)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![token_hash, client_id, user_id, scope, expires_at],
        )?;
        Ok(())
    }

    /// Validate an access token by its hash. Returns (user_id, scope) if valid.
    pub fn validate_access_token(&self, token_hash: &str) -> Result<Option<(String, Option<String>)>> {
        let conn = self.lock()?;
        let mut stmt = conn.prepare(
            "SELECT user_id, scope, expires_at FROM access_tokens WHERE token_hash = ?1",
        )?;
        let result = stmt
            .query(params![token_hash])?
            .next()?
            .map(|row| -> rusqlite::Result<(String, Option<String>, String)> {
                Ok((row.get(0)?, row.get(1)?, row.get(2)?))
            })
            .transpose()?;

        match result {
            Some((user_id, scope, expires_at)) => {
                let expires = chrono::NaiveDateTime::parse_from_str(&expires_at, "%Y-%m-%dT%H:%M:%SZ")
                    .map_err(|e| ServerError::Internal(format!("Bad expiry timestamp: {e}")))?;
                let expires_utc = expires.and_utc();
                if expires_utc < chrono::Utc::now() {
                    Ok(None)
                } else {
                    Ok(Some((user_id, scope)))
                }
            }
            None => Ok(None),
        }
    }

    // ── Refresh tokens ───────────────────────────────────────

    pub fn insert_refresh_token(
        &self,
        token_hash: &str,
        client_id: &str,
        user_id: &str,
        scope: Option<&str>,
        expires_at: &str,
    ) -> Result<()> {
        let conn = self.lock()?;
        conn.execute(
            "INSERT INTO refresh_tokens (token_hash, client_id, user_id, scope, expires_at)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![token_hash, client_id, user_id, scope, expires_at],
        )?;
        Ok(())
    }

    /// Consume a refresh token (rotation). Atomically revokes the old one, returns (client_id, user_id, scope).
    /// Implements breach detection: if a revoked token is reused, all tokens for that client/user are revoked.
    pub fn consume_refresh_token(
        &self,
        token_hash: &str,
    ) -> Result<Option<(String, String, Option<String>)>> {
        let conn = self.lock()?;
        let now = chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string();

        // Atomically revoke the token, checking revoked=0 and expiry in the WHERE clause
        conn.execute(
            "UPDATE refresh_tokens SET revoked = 1 WHERE token_hash = ?1 AND revoked = 0 AND expires_at >= ?2",
            params![token_hash, now],
        )?;

        if conn.changes() == 0 {
            // Check if token exists but was already revoked (breach detection)
            let already_revoked: bool = conn.prepare(
                "SELECT 1 FROM refresh_tokens WHERE token_hash = ?1 AND revoked = 1",
            )?.exists(params![token_hash])?;

            if already_revoked {
                // Revoke all tokens for this client+user (token theft detected)
                let (client_id, user_id): (String, String) = {
                    let mut s = conn.prepare(
                        "SELECT client_id, user_id FROM refresh_tokens WHERE token_hash = ?1",
                    )?;
                    s.query_row(params![token_hash], |r| Ok((r.get(0)?, r.get(1)?)))?
                };
                conn.execute(
                    "UPDATE refresh_tokens SET revoked = 1 WHERE client_id = ?1 AND user_id = ?2",
                    params![client_id, user_id],
                )?;
                conn.execute(
                    "DELETE FROM access_tokens WHERE client_id = ?1 AND user_id = ?2",
                    params![client_id, user_id],
                )?;
                tracing::warn!(
                    client_id = %client_id,
                    user_id = %user_id,
                    "Refresh token reuse detected - revoked all tokens for client/user"
                );
            }
            return Ok(None);
        }

        // Read back the consumed row
        let mut stmt = conn.prepare(
            "SELECT client_id, user_id, scope FROM refresh_tokens WHERE token_hash = ?1",
        )?;
        let result = stmt
            .query(params![token_hash])?
            .next()?
            .map(|row| -> rusqlite::Result<(String, String, Option<String>)> {
                Ok((row.get(0)?, row.get(1)?, row.get(2)?))
            })
            .transpose()?;

        Ok(result)
    }

    // ── Pending auth requests ────────────────────────────────

    pub fn insert_pending_auth(&self, pending: &PendingAuth) -> Result<()> {
        let conn = self.lock()?;
        conn.execute(
            "INSERT INTO pending_auth_requests
             (state_token, client_id, redirect_uri, code_challenge, code_challenge_method,
              scope, original_state, idp_csrf, idp_nonce, idp_pkce_verifier, provider, expires_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
            params![
                pending.state_token,
                pending.client_id,
                pending.redirect_uri,
                pending.code_challenge,
                pending.code_challenge_method,
                pending.scope,
                pending.original_state,
                pending.idp_csrf,
                pending.idp_nonce,
                pending.idp_pkce_verifier,
                pending.provider,
                pending.expires_at,
            ],
        )?;
        Ok(())
    }

    /// Consume a pending auth request by the IdP CSRF token.
    /// Safety: the Mutex serializes all access within this process, preventing TOCTOU races.
    pub fn consume_pending_auth(&self, idp_csrf: &str) -> Result<Option<PendingAuth>> {
        let conn = self.lock()?;
        let mut stmt = conn.prepare(
            "SELECT state_token, client_id, redirect_uri, code_challenge, code_challenge_method,
                    scope, original_state, idp_csrf, idp_nonce, idp_pkce_verifier, provider, expires_at
             FROM pending_auth_requests WHERE idp_csrf = ?1",
        )?;
        let result: Option<PendingAuth> = stmt
            .query(params![idp_csrf])?
            .next()?
            .map(|row| -> rusqlite::Result<PendingAuth> {
                Ok(PendingAuth {
                    state_token: row.get(0)?,
                    client_id: row.get(1)?,
                    redirect_uri: row.get(2)?,
                    code_challenge: row.get(3)?,
                    code_challenge_method: row.get(4)?,
                    scope: row.get(5)?,
                    original_state: row.get(6)?,
                    idp_csrf: row.get(7)?,
                    idp_nonce: row.get(8)?,
                    idp_pkce_verifier: row.get(9)?,
                    provider: row.get(10)?,
                    expires_at: row.get(11)?,
                })
            })
            .transpose()?;

        if result.is_some() {
            conn.execute(
                "DELETE FROM pending_auth_requests WHERE idp_csrf = ?1",
                params![idp_csrf],
            )?;
            if conn.changes() == 0 {
                // Someone else consumed it between SELECT and DELETE
                return Ok(None);
            }
        }
        Ok(result)
    }

    // ── Cleanup ──────────────────────────────────────────────

    pub fn cleanup_expired(&self) -> Result<u64> {
        let conn = self.lock()?;
        let now = chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string();
        let mut total = 0u64;
        total += conn.execute(
            "DELETE FROM auth_codes WHERE expires_at < ?1 OR used = 1",
            params![now],
        )? as u64;
        total += conn.execute(
            "DELETE FROM access_tokens WHERE expires_at < ?1",
            params![now],
        )? as u64;
        total += conn.execute(
            "DELETE FROM refresh_tokens WHERE expires_at < ?1 OR revoked = 1",
            params![now],
        )? as u64;
        total += conn.execute(
            "DELETE FROM pending_auth_requests WHERE expires_at < ?1",
            params![now],
        )? as u64;
        Ok(total)
    }
}
