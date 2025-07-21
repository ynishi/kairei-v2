use crate::AppState;
use axum::Router;

pub mod base_model;

/// API v1 routes
pub fn routes() -> Router<AppState> {
    Router::new().nest("/models", base_model::routes())
    // Add other v1 routes here
    // .nest("/loras", loras::routes())
}
