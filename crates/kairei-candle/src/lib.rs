//! Candle integration for Kairei-v2 AgentCulture Framework

pub mod error;
mod llama2c_processor;
mod llama2_lora_processor;
mod llama2_manual_lora;
mod llama2_simple_lora_processor;
mod model;
pub mod models;

pub use error::CandleError;
pub use llama2c_processor::{Llama2CProcessor, Llama2CProcessorBuilder};
pub use llama2_lora_processor::Llama2LoraProcessor;
pub use llama2_manual_lora::{ManualLlama2LoraProcessor, Config as Llama2Config};
pub use llama2_simple_lora_processor::SimpleLlama2LoraProcessor;
pub use model::{CandleModel, ModelConfig};

/// Initialize the candle backend
pub fn init() {
    println!("kairei-candle initialized");
}
