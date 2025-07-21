use axum::Router;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;
use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;

pub mod config;
pub mod routes;

pub use config::ApiConfig;

/// Application state
#[derive(Clone, Default)]
pub struct AppState {
    // Add future fields here
    // pub db_pool: DatabasePool,
    // pub model_manager: ModelManager,
    // pub config: AppConfig,
}

#[derive(OpenApi)]
#[openapi(
    paths(
        crate::routes::health::health_check,
    ),
    components(
        schemas(crate::routes::health::HealthResponse)
    ),
    tags(
        (name = "health", description = "Health check endpoints")
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
