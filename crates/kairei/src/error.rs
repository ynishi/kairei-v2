//! Error types for kairei crate

use thiserror::Error;

#[derive(Error, Debug)]
pub enum KaireiError {
    #[error("Core error: {0}")]
    Core(#[from] kairei_core::CoreError),

    #[error("Build error: {0}")]
    Build(String),

    #[cfg(feature = "candle")]
    #[error("Candle error: {0}")]
    Candle(#[from] kairei_candle::error::CandleError),
}

pub type Result<T> = std::result::Result<T, KaireiError>;
