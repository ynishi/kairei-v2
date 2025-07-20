//! Kairei CLI library

pub mod commands;
pub mod error;

/// Model type for CLI
#[derive(Debug, Clone, Copy, clap::ValueEnum)]
pub enum ModelType {
    /// Llama2-C model (smaller, faster)
    Llama2c,
    /// Full Llama2 model with LoRA support
    Llama2,
    /// Llama3 model with LoRA support
    Llama3,
}

impl std::fmt::Display for ModelType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelType::Llama2c => write!(f, "llama2c"),
            ModelType::Llama2 => write!(f, "llama2"),
            ModelType::Llama3 => write!(f, "llama3"),
        }
    }
}
