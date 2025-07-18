//! Candle integration for Kairei-v2 AgentCulture Framework

mod error;
mod model;

pub use error::CandleError;
pub use model::{CandleModel, ModelConfig};

/// Initialize the candle backend
pub fn init() {
    println!("kairei-candle initialized");
}
