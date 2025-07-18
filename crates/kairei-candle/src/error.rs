//! Error types for kairei-candle

use thiserror::Error;

#[derive(Error, Debug)]
pub enum CandleError {
    #[error("Candle error: {0}")]
    Candle(#[from] candle_core::Error),

    #[error("Tokenizer error: {0}")]
    Tokenizer(#[from] tokenizers::Error),

    #[error("Model not loaded")]
    ModelNotLoaded,

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Other error: {0}")]
    Other(String),
}

// Convert CandleError to CoreError
impl From<CandleError> for kairei_core::CoreError {
    fn from(err: CandleError) -> Self {
        kairei_core::CoreError::Backend(err.to_string())
    }
}
