use thiserror::Error;

#[derive(Error, Debug)]
pub enum LoraError {
    #[error("LoRA not found: {0}")]
    NotFound(String),

    #[error("LoRA already exists: {0}")]
    AlreadyExists(String),

    #[error("Invalid LoRA metadata: {0}")]
    InvalidMetadata(String),

    #[error("Storage error: {0}")]
    StorageError(#[from] crate::storage::StorageError),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("Database error: {0}")]
    DatabaseError(String),
}

pub type Result<T> = std::result::Result<T, LoraError>;
