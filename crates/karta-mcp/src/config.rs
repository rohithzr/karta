//! Environment-only configuration for the karta-mcp wrapper.
//!
//! This module reads `std::env` only. It never loads `.env` files or any other
//! configuration source. Model/endpoint variables are left in the environment
//! for `karta_core::Karta::with_defaults` to read.

use std::env;
use std::path::Path;

pub const ENV_STORE_DIR: &str = "KARTA_STORE_DIR";
pub const ENV_CAPTURE_PORT: &str = "KARTA_CAPTURE_PORT";
pub const ENV_PRECOMPACT: &str = "KARTA_PRECOMPACT";

const DEFAULT_CAPTURE_PORT: u16 = 3137;

/// Wrapper configuration produced from the process environment.
#[derive(Debug, Clone)]
pub struct Config {
    /// `karta_core` configuration with `storage.data_dir` set from the env.
    pub core: karta_core::config::KartaConfig,
    /// Localhost HTTP capture endpoint port.
    pub capture_port: u16,
    /// Whether `PreCompact` capture events should trigger consolidation.
    pub precompact: bool,
}

impl Config {
    /// Build a `Config` from the current process environment.
    ///
    /// This performs no network or database I/O. It only reads environment
    /// variables and resolves paths.
    pub fn from_env() -> anyhow::Result<Self> {
        let data_dir = data_dir_from_env()?;
        let capture_port = parse_capture_port(env::var(ENV_CAPTURE_PORT).ok())?;
        let precompact = parse_precompact(env::var(ENV_PRECOMPACT).ok());

        let mut core = karta_core::config::KartaConfig::default();
        core.storage.data_dir = data_dir;

        Ok(Self {
            core,
            capture_port,
            precompact,
        })
    }

    /// Return the configured storage directory.
    pub fn store_dir(&self) -> &str {
        &self.core.storage.data_dir
    }
}

fn data_dir_from_env() -> anyhow::Result<String> {
    let raw = match env::var(ENV_STORE_DIR) {
        Ok(v) => expand_home(&v),
        Err(_) => default_store_dir()?,
    };
    normalize_data_dir(&raw)
}

fn default_store_dir() -> anyhow::Result<String> {
    let home =
        env::var("HOME").map_err(|_| anyhow::anyhow!("HOME environment variable is not set"))?;
    let path = Path::new(&home).join(".karta").join("store");
    Ok(path.to_string_lossy().into_owned())
}

fn expand_home(path: &str) -> String {
    if let Some(rest) = path.strip_prefix("~/")
        && let Ok(home) = env::var("HOME")
    {
        return Path::new(&home).join(rest).to_string_lossy().into_owned();
    }
    path.to_string()
}

fn normalize_data_dir(dir: &str) -> anyhow::Result<String> {
    let path = Path::new(dir);
    let abs = if path.is_absolute() {
        path.to_path_buf()
    } else {
        let cwd = env::current_dir()
            .map_err(|e| anyhow::anyhow!("cannot determine current directory: {e}"))?;
        cwd.join(path)
    };
    Ok(abs.to_string_lossy().into_owned())
}

fn parse_capture_port(raw: Option<String>) -> anyhow::Result<u16> {
    match raw {
        None => Ok(DEFAULT_CAPTURE_PORT),
        Some(s) => {
            let value = s.trim().parse::<u16>().map_err(|_| {
                anyhow::anyhow!(
                    "invalid {ENV_CAPTURE_PORT} value {s:?}: must be a valid port number (1-65535)"
                )
            })?;
            if value == 0 {
                anyhow::bail!(
                    "invalid {ENV_CAPTURE_PORT} value {s:?}: port must be greater than 0"
                );
            }
            Ok(value)
        }
    }
}

fn parse_precompact(raw: Option<String>) -> bool {
    raw.as_deref() == Some("1")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    /// Run `f` while holding a lock and with a set of env vars applied.
    /// Values of `None` remove the variable; `Some` sets it. Original values
    /// are restored afterwards.
    fn with_env<F>(vars: &[(&str, Option<&str>)], f: F)
    where
        F: FnOnce(),
    {
        let _guard = ENV_LOCK.lock().unwrap();
        let mut saved: Vec<(&str, Option<String>)> = Vec::new();
        for (key, value) in vars {
            saved.push((*key, env::var(key).ok()));
            match value {
                Some(v) => unsafe { env::set_var(key, v) },
                None => unsafe { env::remove_var(key) },
            }
        }
        f();
        for (key, value) in saved {
            match value {
                Some(v) => unsafe { env::set_var(key, v) },
                None => unsafe { env::remove_var(key) },
            }
        }
    }

    #[test]
    fn defaults_are_correct() {
        with_env(
            &[
                (ENV_STORE_DIR, None),
                (ENV_CAPTURE_PORT, None),
                (ENV_PRECOMPACT, None),
            ],
            || {
                let home = env::var("HOME").expect("HOME should be set");
                let cfg = Config::from_env().unwrap();
                assert_eq!(cfg.store_dir(), format!("{home}/.karta/store"));
                assert_eq!(cfg.capture_port, 3137);
                assert!(!cfg.precompact);
            },
        );
    }

    #[test]
    fn store_dir_override_is_honored() {
        with_env(
            &[
                (ENV_STORE_DIR, Some("/tmp/karta-test-store")),
                (ENV_CAPTURE_PORT, None),
                (ENV_PRECOMPACT, None),
            ],
            || {
                let cfg = Config::from_env().unwrap();
                assert_eq!(cfg.store_dir(), "/tmp/karta-test-store");
            },
        );
    }

    #[test]
    fn store_dir_tilde_expansion() {
        with_env(
            &[
                (ENV_STORE_DIR, Some("~/custom-karta-store")),
                (ENV_CAPTURE_PORT, None),
                (ENV_PRECOMPACT, None),
            ],
            || {
                let home = env::var("HOME").unwrap();
                let cfg = Config::from_env().unwrap();
                assert_eq!(cfg.store_dir(), format!("{home}/custom-karta-store"));
            },
        );
    }

    #[test]
    fn capture_port_override_is_honored() {
        with_env(
            &[
                (ENV_STORE_DIR, None),
                (ENV_CAPTURE_PORT, Some("4147")),
                (ENV_PRECOMPACT, None),
            ],
            || {
                let cfg = Config::from_env().unwrap();
                assert_eq!(cfg.capture_port, 4147);
            },
        );
    }

    #[test]
    fn invalid_capture_port_fails_fast() {
        for bad in ["abc", "0", "70000", "-1", "3.14"] {
            with_env(
                &[
                    (ENV_STORE_DIR, None),
                    (ENV_CAPTURE_PORT, Some(bad)),
                    (ENV_PRECOMPACT, None),
                ],
                || {
                    let err = Config::from_env().unwrap_err();
                    let msg = err.to_string();
                    assert!(
                        msg.contains(ENV_CAPTURE_PORT),
                        "error should mention {ENV_CAPTURE_PORT}: {msg}"
                    );
                },
            );
        }
    }

    #[test]
    fn precompact_defaults_off() {
        with_env(
            &[
                (ENV_STORE_DIR, None),
                (ENV_CAPTURE_PORT, None),
                (ENV_PRECOMPACT, None),
            ],
            || {
                let cfg = Config::from_env().unwrap();
                assert!(!cfg.precompact);
            },
        );
    }

    #[test]
    fn precompact_only_opt_in_with_one() {
        for (value, expected) in [
            ("1", true),
            ("0", false),
            ("true", false),
            ("yes", false),
            ("", false),
        ] {
            with_env(
                &[
                    (ENV_STORE_DIR, None),
                    (ENV_CAPTURE_PORT, None),
                    (ENV_PRECOMPACT, Some(value)),
                ],
                || {
                    let cfg = Config::from_env().unwrap();
                    assert_eq!(cfg.precompact, expected, "KARTA_PRECOMPACT={value}");
                },
            );
        }
    }

    #[test]
    fn env_precedence_over_defaults() {
        with_env(
            &[
                (ENV_STORE_DIR, Some("/tmp/karta-precedence")),
                (ENV_CAPTURE_PORT, Some("4242")),
                (ENV_PRECOMPACT, Some("1")),
            ],
            || {
                let cfg = Config::from_env().unwrap();
                assert_eq!(cfg.store_dir(), "/tmp/karta-precedence");
                assert_eq!(cfg.capture_port, 4242);
                assert!(cfg.precompact);
            },
        );
    }

    #[test]
    fn ignores_dotenv_file_in_working_directory() {
        use std::fs;

        with_env(
            &[
                (ENV_STORE_DIR, None),
                (ENV_CAPTURE_PORT, None),
                (ENV_PRECOMPACT, None),
            ],
            || {
                let tmp = tempfile::tempdir().unwrap();
                let dotenv_path = tmp.path().join(".env");
                fs::write(&dotenv_path, "KARTA_STORE_DIR=/from-dotenv\n").unwrap();

                let original_cwd = env::current_dir().unwrap();
                env::set_current_dir(tmp.path()).unwrap();
                let home = env::var("HOME").unwrap();

                let cfg = Config::from_env().unwrap();
                // The wrapper must not read the .env file, so it falls back
                // to the default store dir derived from HOME.
                assert_eq!(cfg.store_dir(), format!("{home}/.karta/store"));

                env::set_current_dir(original_cwd).unwrap();
            },
        );
    }
}
