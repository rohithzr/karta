//! Library for the `karta-mcp` binary.
//!
//! All modules are public so the binary and integration tests can construct
//! the individual components (e.g., the axum capture router) directly with
//! injected `Karta` and `CaptureQueue` instances.

pub mod capture;
pub mod config;
pub mod karta_handle;
pub mod queue;
pub mod server;
pub mod session;
pub mod tools;
pub mod transcript;
