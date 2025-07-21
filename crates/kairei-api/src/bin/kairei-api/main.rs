use clap::Parser;
use kairei::base_model::InMemoryBaseModelRepository;
use kairei_api::{ApiConfig, AppState, build_app};
use std::path::PathBuf;
use std::sync::Arc;
use tracing::{info, warn};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

/// Kairei API Server
#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Host address to bind to
    #[arg(short = 'H', long, env = "KAIREI_HOST", default_value = "127.0.0.1")]
    host: String,

    /// Port to listen on
    #[arg(short, long, env = "KAIREI_PORT", default_value_t = 3000)]
    port: u16,

    /// Log level (error, warn, info, debug, trace)
    #[arg(short, long, env = "KAIREI_LOG_LEVEL", default_value = "info")]
    log_level: String,

    /// Config file path
    #[arg(short, long, env = "KAIREI_CONFIG")]
    config_file: Option<PathBuf>,

    /// Disable Swagger UI
    #[arg(long, env = "KAIREI_DISABLE_SWAGGER", default_value_t = false)]
    disable_swagger: bool,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Parse CLI arguments
    let cli = Cli::parse();

    // Initialize logging
    let filter = format!("kairei_api={},tower_http=debug", cli.log_level);
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| filter.into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    // Load configuration
    let mut config = if let Some(config_path) = &cli.config_file {
        match ApiConfig::load_from_file(config_path) {
            Ok(cfg) => {
                info!("Configuration loaded from: {}", config_path.display());
                cfg
            }
            Err(e) => {
                warn!(
                    "Failed to load config file: {}. Using default configuration.",
                    e
                );
                ApiConfig::default()
            }
        }
    } else {
        ApiConfig::default()
    };

    // Override with CLI options
    config.host = cli.host;
    config.port = cli.port;
    config.log_level = cli.log_level;
    config.enable_swagger = !cli.disable_swagger;

    let addr = config.socket_addr()?;

    // Initialize repository and application state
    let base_model_repository = Arc::new(InMemoryBaseModelRepository::new());
    let state = AppState::new(base_model_repository, config.auth.clone());

    // Build application
    let app = build_app(state);

    // Start server
    info!("Starting server on: {}", addr);
    if config.enable_swagger {
        info!("Swagger UI: http://{}/swagger-ui", addr);
    }
    info!(
        "Authentication: {}",
        if config.auth.enabled {
            "ENABLED"
        } else {
            "DISABLED"
        }
    );
    if config.auth.enabled {
        debug!(
            "Auth config: domain={:?}, audience={:?}",
            config.auth.auth0_domain, config.auth.auth0_audience
        );
    }

    let listener = tokio::net::TcpListener::bind(addr).await?;

    axum::serve(listener, app)
        .await
        .map_err(|e| anyhow::anyhow!("Server error: {}", e))
}
