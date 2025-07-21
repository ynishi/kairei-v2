use thiserror::Error;

pub type StorageResult<T> = Result<T, StorageError>;

#[derive(Error, Debug)]
pub enum StorageError {
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Path not found: {0}")]
    NotFound(String),

    #[error("Invalid path: {0}")]
    InvalidPath(String),

    #[error("Permission denied: {0}")]
    PermissionDenied(String),

    #[error("Storage configuration error: {0}")]
    ConfigError(String),

    #[error("Operation not supported: {0}")]
    NotSupported(String),

    #[error("Storage error: {0}")]
    Other(String),
}
