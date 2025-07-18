//! Error types for kairei-candle

use thiserror::Error;

#[derive(Error, Debug)]
pub enum CandleError {
    #[error("Candle error: {0}")]
    Candle(#[from] candle_core::Error),

    #[error("Model not loaded")]
    ModelNotLoaded,

    #[error("Generation error: {0}")]
    Generation(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}
