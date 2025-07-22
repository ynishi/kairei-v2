use crate::AppState;
use axum::Router;

pub mod base_model;
pub mod chat;
pub mod lora;
pub mod trainer;
pub mod training_data;

/// API v1 routes
pub fn routes() -> Router<AppState> {
    Router::new()
        .nest("/models", base_model::routes())
        .nest("/loras", lora::routes())
        .nest("/training-data", training_data::routes())
        .nest("/train", trainer::routes())
        .nest("/chat", chat::routes())
}
