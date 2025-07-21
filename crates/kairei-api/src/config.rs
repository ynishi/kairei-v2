use serde::{Deserialize, Serialize};
use std::net::SocketAddr;

/// API configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ApiConfig {
    /// Host address
    pub host: String,

    /// Port number
    pub port: u16,

    /// CORS configuration
    pub cors: CorsConfig,

    /// Log level
    pub log_level: String,

    /// Enable Swagger UI
    pub enable_swagger: bool,
}

impl Default for ApiConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 3000,
            cors: CorsConfig::default(),
            log_level: "info".to_string(),
            enable_swagger: true,
        }
    }
}

impl ApiConfig {
    /// Convert to SocketAddr
    pub fn socket_addr(&self) -> anyhow::Result<SocketAddr> {
        let addr = format!("{}:{}", self.host, self.port);
        addr.parse()
            .map_err(|e| anyhow::anyhow!("Invalid address {}: {}", addr, e))
    }

    /// Load from configuration file
    pub fn load_from_file(path: &std::path::Path) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("Failed to read config file: {}", e))?;

        serde_json::from_str(&content)
            .map_err(|e| anyhow::anyhow!("Failed to parse config file: {}", e))
    }
}

/// CORS configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct CorsConfig {
    /// Allowed origins
    pub allowed_origins: Vec<String>,

    /// Allow any origin
    pub allow_any_origin: bool,
}

impl Default for CorsConfig {
    fn default() -> Self {
        Self {
            allowed_origins: vec![],
            allow_any_origin: true, // Default to true for development
        }
    }
}
