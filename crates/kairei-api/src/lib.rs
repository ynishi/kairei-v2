use axum::Router;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;
use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;

pub mod config;
pub mod routes;

pub use config::ApiConfig;

use kairei::base_model::{BaseModelRepository, BaseModelService};
use std::sync::Arc;

/// Application state
#[derive(Clone)]
pub struct AppState {
    pub base_model_service: BaseModelService,
}

impl AppState {
    /// Create a new AppState with the given repository
    pub fn new(base_model_repository: Arc<dyn BaseModelRepository>) -> Self {
        Self {
            base_model_service: BaseModelService::new(base_model_repository),
        }
    }
}

#[derive(OpenApi)]
#[openapi(
    paths(
        crate::routes::health::health_check,
        crate::routes::v1::base_model::list_models,
        crate::routes::v1::base_model::get_model,
        crate::routes::v1::base_model::create_model,
        crate::routes::v1::base_model::update_model,
        crate::routes::v1::base_model::delete_model,
    ),
    components(
        schemas(
            crate::routes::health::HealthResponse,
            crate::routes::v1::base_model::ModelDto,
            crate::routes::v1::base_model::ModelStatus,
            crate::routes::v1::base_model::ListModelsResponse,
            crate::routes::v1::base_model::CreateModelRequest,
            crate::routes::v1::base_model::UpdateModelRequest,
            crate::routes::v1::base_model::BaseModelMetadataDto,
        )
    ),
    tags(
        (name = "health", description = "Health check endpoints"),
        (name = "models", description = "Model management endpoints")
    )
)]
pub struct ApiDoc;

/// Build API application
pub fn build_app(state: AppState) -> Router {
    Router::new()
        .merge(routes::routes())
        .merge(SwaggerUi::new("/swagger-ui").url("/api-docs/openapi.json", ApiDoc::openapi()))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}
