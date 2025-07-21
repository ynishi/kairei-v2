use axum::Router;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;
use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;

pub mod config;
pub mod middleware;
pub mod routes;

pub use config::ApiConfig;

use crate::config::AuthConfig;
use kairei::base_model::{BaseModelRepository, BaseModelService, ModelDownloader};
use kairei::storage::Storage;
use std::sync::Arc;

/// Application state
#[derive(Clone)]
pub struct AppState {
    pub base_model_service: BaseModelService,
    pub auth_config: AuthConfig,
}

impl AppState {
    /// Create a new AppState with the given repository, storage and downloader
    pub fn new(
        base_model_repository: Arc<dyn BaseModelRepository>,
        storage: Arc<dyn Storage>,
        downloader: Arc<dyn ModelDownloader>,
        auth_config: AuthConfig,
    ) -> Self {
        Self {
            base_model_service: BaseModelService::new(base_model_repository, storage, downloader),
            auth_config,
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
        crate::routes::v1::base_model::download_model,
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
    // Create the main API routes
    let api_routes = routes::routes();

    // Build the complete app with Swagger UI
    let app = Router::new()
        .merge(SwaggerUi::new("/swagger-ui").url("/api-docs/openapi.json", ApiDoc::openapi()))
        .merge(api_routes);

    // Apply authentication middleware if enabled
    let app = if state.auth_config.enabled {
        app.layer(axum::middleware::from_fn_with_state(
            state.clone(),
            middleware::auth_middleware,
        ))
    } else {
        app
    };

    app.layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}
