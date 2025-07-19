//! Error types for kairei-cli

use thiserror::Error;

#[derive(Error, Debug)]
pub enum CliError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Kairei error: {0}")]
    Kairei(#[from] kairei::KaireiError),

    #[error("Core error: {0}")]
    Candle(#[from] kairei_candle::error::CandleError),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Readline error: {0}")]
    Readline(#[from] rustyline::error::ReadlineError),
}
