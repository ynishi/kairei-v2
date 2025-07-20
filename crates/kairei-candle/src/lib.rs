//! Candle integration for Kairei-v2 AgentCulture Framework

pub mod error;
mod llama2_lora;
mod llama2c_processor;
mod llama3_lora;
mod model;
pub mod models;

pub use error::CandleError;
pub use llama2_lora::{Config as Llama2Config, Llama2LoraProcessor};
pub use llama2c_processor::{Llama2CProcessor, Llama2CProcessorBuilder};
pub use llama3_lora::{Config as Llama3Config, Llama3LoraProcessor};
pub use model::{CandleModel, ModelConfig};

/// Initialize the candle backend
pub fn init() {
    println!("kairei-candle initialized");
}
