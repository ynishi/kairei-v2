//! Candle integration for Kairei-v2 AgentCulture Framework

mod error;
mod llama2c;
mod model;
mod processor;
mod train;

pub use error::CandleError;
pub use llama2c::Llama2CProcessor;
pub use model::{CandleModel, ModelConfig};
pub use processor::{CandleProcessor, CandleProcessorBuilder};
pub use train::{TrainingConfig, train_lora};

/// Initialize the candle backend
pub fn init() {
    println!("kairei-candle initialized");
}
