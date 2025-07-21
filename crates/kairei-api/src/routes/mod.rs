use crate::AppState;
use axum::Router;

pub mod health;
pub mod v1;

/// Merge all routes
pub fn routes() -> Router<AppState> {
    Router::new()
        .merge(health::routes())
        .nest("/api/v1", v1::routes())
}
