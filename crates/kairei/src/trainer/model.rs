//! Training data models

use serde::{Deserialize, Serialize};

/// Parameters for training a LoRA model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainParams {
    /// Training data path
    pub train_data: String,

    /// Number of epochs
    pub epochs: u32,

    /// LoRA rank (r)
    pub lora_r: u32,

    /// LoRA alpha
    pub lora_alpha: u32,

    /// Output directory (optional, will be generated if not provided)
    pub output_dir: Option<String>,

    /// Model name or HuggingFace ID
    pub model: String,

    /// Batch size
    pub batch_size: u32,

    /// Learning rate
    pub learning_rate: f32,

    /// Run test after training
    pub test: bool,
}

/// Result of a training run
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingResult {
    /// Output directory where the model was saved
    pub output_dir: String,

    /// Status of the training
    pub status: TrainingStatus,

    /// Final loss value (if available)
    pub final_loss: Option<f32>,

    /// Total training duration
    pub duration: Option<String>,

    /// Any error message
    pub error: Option<String>,

    /// Model checkpoints created
    pub checkpoints: Vec<String>,
}

/// Status of a training job
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TrainingStatus {
    /// Training is queued
    Queued,

    /// Training is in progress
    Running,

    /// Training completed successfully
    Completed,

    /// Training failed
    Failed,

    /// Training was cancelled
    Cancelled,
}
