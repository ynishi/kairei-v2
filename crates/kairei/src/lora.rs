//! LoRA training functionality wrapper for Kairei

#[cfg(feature = "candle")]
use kairei_candle::{TrainingConfig as CandleTrainingConfig, train_lora as candle_train_lora};
use kairei_core::Result;

/// Training configuration for LoRA
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    pub culture_name: String,
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub lora_rank: usize,
    pub lora_alpha: f64,
    pub lora_dropout: Option<f32>,
}

/// Train a LoRA model using the candle backend
#[cfg(feature = "candle")]
pub fn train_lora(config: TrainingConfig) -> Result<()> {
    let candle_config = CandleTrainingConfig {
        culture_name: config.culture_name,
        epochs: config.epochs,
        batch_size: config.batch_size,
        learning_rate: config.learning_rate,
        lora_rank: config.lora_rank,
        lora_alpha: config.lora_alpha,
        lora_dropout: config.lora_dropout,
    };

    candle_train_lora(candle_config)
        .map_err(|e| kairei_core::CoreError::Processing(format!("Training failed: {}", e)))
}

/// Train a LoRA model without candle backend (placeholder)
#[cfg(not(feature = "candle"))]
pub fn train_lora(_config: TrainingConfig) -> Result<()> {
    Err(kairei_core::CoreError::Processing(
        "LoRA training requires the 'candle' feature to be enabled".to_string(),
    ))
}
