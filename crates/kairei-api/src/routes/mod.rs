use crate::AppState;
use axum::Router;

pub mod health;

/// Merge all routes
pub fn routes() -> Router<AppState> {
    Router::new().merge(health::routes())
    // Add other routes here
    // .merge(loras::routes())
    // .merge(models::routes())
}
