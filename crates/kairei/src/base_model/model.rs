use serde::{Deserialize, Serialize};
use std::fmt;

/// NewType pattern for Request ID
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BaseModelId(String);

impl BaseModelId {
    /// Create a new ModelId
    pub fn new() -> Self {
        Self(uuid::Uuid::new_v4().to_string())
    }

    /// Create from existing string
    pub fn from_string(s: impl Into<String>) -> Self {
        Self(s.into())
    }

    /// Get the inner string
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for BaseModelId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for BaseModelId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BaseModel {
    pub id: BaseModelId,
    pub name: String,
    pub description: Option<String>,
    pub repo_id: Option<String>, // Hugging Face repo ID
    pub filename: Option<String>,
    pub size_mb: Option<u64>,
    pub metadata: Option<BaseModelMetadata>,
}

/// Model metadata representing a downloadable model
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BaseModelMetadata {
    pub repo_id: String,
    pub name: String, // name of the Hugging Face model
    pub description: Option<String>,
    pub downloaded_at: Option<String>,
    // Model specific params
    pub parameters: Option<String>,   // e.g., "1.1B", "7B", etc.
    pub architecture: Option<String>, // e.g., "llama2c", "llama3", etc.
    pub quantization: Option<String>, // e.g., "f16", "int8", etc.
}
