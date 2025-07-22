//! Trainer error types

use std::fmt;

#[derive(Debug)]
pub enum TrainerError {
    /// Training process failed
    TrainingFailed(String),

    /// Invalid configuration
    InvalidConfig(String),

    /// File system error
    FileSystem(String),

    /// Poetry not found
    PoetryNotFound,

    /// Training script not found
    ScriptNotFound(String),

    /// Other error
    Other(String),
}

impl fmt::Display for TrainerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TrainingFailed(msg) => write!(f, "Training failed: {}", msg),
            Self::InvalidConfig(msg) => write!(f, "Invalid configuration: {}", msg),
            Self::FileSystem(msg) => write!(f, "File system error: {}", msg),
            Self::PoetryNotFound => {
                write!(f, "Poetry not found. Please install Poetry to run training")
            }
            Self::ScriptNotFound(path) => write!(f, "Training script not found at: {}", path),
            Self::Other(msg) => write!(f, "Trainer error: {}", msg),
        }
    }
}

impl std::error::Error for TrainerError {}

/// Result type for trainer operations
pub type Result<T> = std::result::Result<T, TrainerError>;
