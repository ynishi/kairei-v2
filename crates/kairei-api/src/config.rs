use kairei::config::{AuthConfig as KaireiAuthConfig, KaireiConfig};
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

    /// Authentication configuration
    pub auth: AuthConfig,
}

impl Default for ApiConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 3000,
            cors: CorsConfig::default(),
            log_level: "info".to_string(),
            enable_swagger: true,
            auth: AuthConfig::default(),
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

    /// Convert to KaireiConfig
    pub fn to_kairei_config(&self) -> KaireiConfig {
        KaireiConfig {
            log_level: self.log_level.clone(),
            auth: KaireiAuthConfig {
                enabled: self.auth.enabled,
                auth0_domain: self.auth.auth0_domain.clone(),
                auth0_audience: self.auth.auth0_audience.clone(),
                issuer: self.auth.issuer.clone(),
                algorithms: self.auth.algorithms.clone(),
                required_scopes: self.auth.required_scopes.clone(),
            },
            ..Default::default()
        }
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

/// Authentication configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AuthConfig {
    /// Enable authentication
    pub enabled: bool,

    /// Auth0 domain (e.g., "your-tenant.auth0.com")
    pub auth0_domain: Option<String>,

    /// Auth0 audience (API identifier)
    pub auth0_audience: Option<String>,

    /// JWT issuer
    pub issuer: Option<String>,

    /// JWT algorithms (default: ["RS256"])
    pub algorithms: Vec<String>,

    /// Required scopes for API access
    pub required_scopes: Vec<String>,
}

impl Default for AuthConfig {
    fn default() -> Self {
        Self {
            enabled: true, // Enabled by default for security
            auth0_domain: None,
            auth0_audience: None,
            issuer: None,
            algorithms: vec!["RS256".to_string()],
            required_scopes: vec![],
        }
    }
}
