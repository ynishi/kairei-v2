//! Error types for kairei-core

use thiserror::Error;

#[derive(Error, Debug)]
pub enum CoreError {
    #[error("Processing error: {0}")]
    Processing(String),

    #[error("Invalid request: {0}")]
    InvalidRequest(String),

    #[error("Candle error: {0}")]
    Candle(String),
}

pub type Result<T> = std::result::Result<T, CoreError>;
