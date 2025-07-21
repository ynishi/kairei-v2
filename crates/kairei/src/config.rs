use serde::{Deserialize, Serialize};

/// Kairei configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct KaireiConfig {
    /// Base directory for all storage
    pub base_dir: String,

    /// Models directory
    pub models_dir: String,

    /// LoRAs directory
    pub loras_dir: String,

    /// Datasets directory
    pub datasets_dir: String,

    /// Base models directory
    pub base_models_dir: String,

    /// Log level
    pub log_level: String,

    /// Authentication configuration
    pub auth: AuthConfig,
}

impl Default for KaireiConfig {
    fn default() -> Self {
        Self {
            base_dir: ".".to_string(),
            models_dir: "models".to_string(),
            loras_dir: "loras".to_string(),
            datasets_dir: "lora_datasets".to_string(),
            base_models_dir: "base_models".to_string(),
            log_level: "info".to_string(),
            auth: AuthConfig::default(),
        }
    }
}

impl KaireiConfig {
    /// Load from configuration file
    pub fn load_from_file(path: &std::path::Path) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("Failed to read config file: {}", e))?;

        serde_json::from_str(&content)
            .map_err(|e| anyhow::anyhow!("Failed to parse config file: {}", e))
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
            enabled: false, // Disabled by default for CLI usage
            auth0_domain: None,
            auth0_audience: None,
            issuer: None,
            algorithms: vec!["RS256".to_string()],
            required_scopes: vec![],
        }
    }
}

// Conversion from ApiConfig to KaireiConfig for kairei-api integration
#[cfg(feature = "api-config")]
impl From<crate::api::config::ApiConfig> for KaireiConfig {
    fn from(api_config: crate::api::config::ApiConfig) -> Self {
        Self {
            auth: api_config.auth,
            ..Default::default()
        }
    }
}
