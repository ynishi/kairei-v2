use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use super::{StorageError, StorageResult};

/// Storage configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StorageConfig {
    Local { base_path: String },
    S3 { bucket: String, prefix: String },
    GCS { bucket: String, prefix: String },
}

/// Storage trait for abstract file operations
#[async_trait]
pub trait Storage: Send + Sync {
    /// Check if a path exists
    async fn exists(&self, path: &str) -> StorageResult<bool>;

    /// Read file contents
    async fn read(&self, path: &str) -> StorageResult<Vec<u8>>;

    /// Write file contents
    async fn write(&self, path: &str, content: &[u8]) -> StorageResult<()>;

    /// Delete a file
    async fn delete(&self, path: &str) -> StorageResult<()>;

    /// List files in a directory
    async fn list(&self, prefix: &str) -> StorageResult<Vec<String>>;

    /// Create directory
    async fn create_dir(&self, path: &str) -> StorageResult<()>;

    /// Copy file from source to destination
    async fn copy(&self, source: &str, destination: &str) -> StorageResult<()>;

    /// Move file from source to destination
    async fn rename(&self, source: &str, destination: &str) -> StorageResult<()>;

    /// Get file metadata (size, modified time, etc.)
    async fn metadata(&self, path: &str) -> StorageResult<FileMetadata>;

    /// Download from URL to storage
    async fn download_from_url(&self, _url: &str, _destination: &str) -> StorageResult<()> {
        Err(StorageError::NotSupported(
            "URL download not implemented".to_string(),
        ))
    }
}

/// File metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileMetadata {
    pub size: u64,
    pub modified: Option<chrono::DateTime<chrono::Utc>>,
    pub is_dir: bool,
}
