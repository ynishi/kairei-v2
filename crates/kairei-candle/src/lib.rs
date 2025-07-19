//! Candle integration for Kairei-v2 AgentCulture Framework

pub mod error;
mod llama2c_processor;
mod model;
mod models;

pub use error::CandleError;
pub use llama2c_processor::{Llama2CProcessor, Llama2CProcessorBuilder};
pub use model::{CandleModel, ModelConfig};

/// Initialize the candle backend
pub fn init() {
    println!("kairei-candle initialized");
}
