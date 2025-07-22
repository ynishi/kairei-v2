use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Training data ID
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TrainingDataId(Uuid);

impl TrainingDataId {
    /// Create a new training data ID
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    /// Create from string
    pub fn from_string(s: String) -> Self {
        Self(Uuid::parse_str(&s).unwrap_or_else(|_| Uuid::new_v4()))
    }

    /// Convert to string
    pub fn to_string(&self) -> String {
        self.0.to_string()
    }
}

impl Default for TrainingDataId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for TrainingDataId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Data type enumeration
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DataType {
    Text,
    Json,
    Csv,
    Parquet,
    Image,
    Audio,
    Video,
    Binary,
    Custom(String),
}

impl std::fmt::Display for DataType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DataType::Text => write!(f, "text"),
            DataType::Json => write!(f, "json"),
            DataType::Csv => write!(f, "csv"),
            DataType::Parquet => write!(f, "parquet"),
            DataType::Image => write!(f, "image"),
            DataType::Audio => write!(f, "audio"),
            DataType::Video => write!(f, "video"),
            DataType::Binary => write!(f, "binary"),
            DataType::Custom(s) => write!(f, "custom:{}", s),
        }
    }
}

/// Data format details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataFormat {
    /// MIME type
    pub mime_type: Option<String>,
    /// File extension
    pub extension: Option<String>,
    /// Encoding (e.g., utf-8)
    pub encoding: Option<String>,
    /// Compression type if any
    pub compression: Option<String>,
}

impl Default for DataFormat {
    fn default() -> Self {
        Self {
            mime_type: None,
            extension: None,
            encoding: Some("utf-8".to_string()),
            compression: None,
        }
    }
}

/// Training data metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
#[derive(Default)]
pub struct TrainingDataMetadata {
    /// Data source (e.g., "user_upload", "generated", "scraped")
    pub source: Option<String>,
    /// Tags for categorization
    pub tags: Vec<String>,
    /// Number of samples/records
    pub sample_count: Option<usize>,
    /// Data statistics
    pub statistics: Option<serde_json::Value>,
    /// Custom properties
    pub properties: Option<serde_json::Value>,
}


/// Training data entity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingData {
    /// Unique identifier
    pub id: TrainingDataId,
    /// Name of the training data
    pub name: String,
    /// Description
    pub description: Option<String>,
    /// Data type
    pub data_type: DataType,
    /// Data format details
    pub format: DataFormat,
    /// File path in storage
    pub file_path: Option<String>,
    /// File size in bytes
    pub size_bytes: Option<u64>,
    /// SHA-256 hash of the data
    pub hash: Option<String>,
    /// Associated LoRA IDs that use this data
    pub lora_ids: Vec<String>,
    /// Metadata
    pub metadata: TrainingDataMetadata,
    /// Creation timestamp
    pub created_at: String,
    /// Last update timestamp
    pub updated_at: String,
    /// Whether the data is archived
    pub archived: bool,
}

impl TrainingData {
    /// Create a new training data instance
    pub fn new(
        name: String,
        description: Option<String>,
        data_type: DataType,
        format: DataFormat,
        metadata: TrainingDataMetadata,
    ) -> Self {
        let now = chrono::Utc::now().to_rfc3339();
        Self {
            id: TrainingDataId::new(),
            name,
            description,
            data_type,
            format,
            file_path: None,
            size_bytes: None,
            hash: None,
            lora_ids: Vec::new(),
            metadata,
            created_at: now.clone(),
            updated_at: now,
            archived: false,
        }
    }
}
