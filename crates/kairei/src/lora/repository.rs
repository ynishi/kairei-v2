use super::{Lora, LoraId, Result};
use async_trait::async_trait;

/// Repository trait for LoRA models
#[async_trait]
pub trait LoraRepository: Send + Sync {
    /// Create a new LoRA
    async fn create(&self, lora: Lora) -> Result<Lora>;

    /// Get a LoRA by ID
    async fn get(&self, id: &LoraId) -> Result<Option<Lora>>;

    /// Get a LoRA by name
    async fn get_by_name(&self, name: &str) -> Result<Option<Lora>>;

    /// List all LoRAs
    async fn list(&self) -> Result<Vec<Lora>>;

    /// Update a LoRA
    async fn update(&self, lora: Lora) -> Result<Lora>;

    /// Delete a LoRA
    async fn delete(&self, id: &LoraId) -> Result<()>;
}
