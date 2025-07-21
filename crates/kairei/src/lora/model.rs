use crate::base_model::BaseModelId;
use serde::{Deserialize, Serialize};
use std::fmt;
use strum_macros::{Display, EnumString};

/// NewType pattern for LoRA ID
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct LoraId(String);

impl LoraId {
    /// Create a new LoraId
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

impl Default for LoraId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for LoraId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// LoRA status
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Display, EnumString)]
#[strum(serialize_all = "lowercase")]
pub enum LoraStatus {
    /// LoRA is available and ready to use
    Available,
    /// LoRA is currently being trained
    Training,
    /// LoRA file is missing
    Missing,
    /// LoRA has errors
    Error,
}

/// LoRA model metadata
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct LoraMetadata {
    // LoRA specific params
    pub rank: Option<usize>,
    pub alpha: Option<f64>,

    // Training information
    pub training_info: Option<TrainingInfo>,

    // Lineage tracking
    pub parent_lora_id: Option<LoraId>,

    // Version tracking
    pub version: Option<String>,
    pub training_framework: Option<String>,
}

/// Training information for a LoRA
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TrainingInfo {
    pub training_data: String,
    pub training_data_hash: Option<String>,
    pub epochs: u32,
    pub batch_size: u32,
    pub learning_rate: f32,
    pub final_loss: Option<f32>,
    pub training_duration: Option<String>,
}

/// LoRA model representation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Lora {
    // Core fields (frequently used for search/display)
    pub id: LoraId,
    pub name: String,
    pub description: Option<String>,
    pub base_model_id: Option<BaseModelId>,
    pub created_at: String,
    pub status: LoraStatus,

    // File information (dynamic)
    pub file_path: Option<String>,
    pub size_bytes: Option<u64>,

    // Additional details
    pub metadata: LoraMetadata,
}

impl Lora {
    /// Create a new LoRA
    pub fn new(
        name: String,
        description: Option<String>,
        base_model_id: Option<BaseModelId>,
        metadata: LoraMetadata,
    ) -> Self {
        Self {
            id: LoraId::new(),
            name,
            description,
            base_model_id,
            created_at: chrono::Utc::now().to_rfc3339(),
            status: LoraStatus::Available,
            file_path: None,
            size_bytes: None,
            metadata,
        }
    }
}
