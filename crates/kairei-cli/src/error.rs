//! Error types for kairei-cli

use thiserror::Error;

#[derive(Error, Debug)]
pub enum CliError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("Core error: {0}")]
    Core(String),
    
    #[error("Invalid input: {0}")]
    InvalidInput(String),
}