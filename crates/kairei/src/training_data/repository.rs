use async_trait::async_trait;

use super::{Result, TrainingData, TrainingDataId};

/// Repository trait for training data
#[async_trait]
pub trait TrainingDataRepository: Send + Sync {
    /// Create a new training data entry
    async fn create(&self, data: TrainingData) -> Result<TrainingData>;

    /// Get training data by ID
    async fn get(&self, id: &TrainingDataId) -> Result<Option<TrainingData>>;

    /// Get training data by name
    async fn get_by_name(&self, name: &str) -> Result<Option<TrainingData>>;

    /// List all training data
    async fn list(&self) -> Result<Vec<TrainingData>>;

    /// List training data by LoRA ID
    async fn list_by_lora(&self, lora_id: &str) -> Result<Vec<TrainingData>>;

    /// Update training data
    async fn update(&self, data: TrainingData) -> Result<TrainingData>;

    /// Delete training data
    async fn delete(&self, id: &TrainingDataId) -> Result<()>;

    /// Check if training data exists
    async fn exists(&self, id: &TrainingDataId) -> Result<bool>;
}
