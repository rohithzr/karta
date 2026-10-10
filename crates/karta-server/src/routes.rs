use axum::Json;
use axum::http::{StatusCode, header};
use axum::response::IntoResponse;
use serde_json::{Value, json};

use crate::error::Result;
use crate::middleware::AuthenticatedUser;

const LOGO_SVG: &str = include_str!("logo.svg");

/// `GET /api/health` — Protected health check.
pub async fn health(user: AuthenticatedUser) -> Result<Json<Value>> {
    Ok(Json(json!({
        "status": "ok",
        "user_id": user.user_id,
    })))
}

/// `GET /icon.svg` — Serve the SVG logo.
pub async fn icon_svg() -> impl IntoResponse {
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "image/svg+xml")],
        LOGO_SVG,
    )
}
