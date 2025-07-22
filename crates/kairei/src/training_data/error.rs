use thiserror::Error;

/// Training data related errors
#[derive(Error, Debug)]
pub enum TrainingDataError {
    #[error("Training data not found: {0}")]
    NotFound(String),

    #[error("Training data already exists: {0}")]
    AlreadyExists(String),

    #[error("Invalid data format: {0}")]
    InvalidFormat(String),

    #[error("Storage error: {0}")]
    StorageError(#[from] crate::storage::StorageError),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("Invalid file type: {0}")]
    InvalidFileType(String),

    #[error("File too large: {0} bytes (max: {1} bytes)")]
    FileTooLarge(u64, u64),

    #[error("Other error: {0}")]
    Other(String),
}

/// Result type for training data operations
pub type Result<T> = std::result::Result<T, TrainingDataError>;
