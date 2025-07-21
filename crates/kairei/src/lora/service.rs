use super::{Lora, LoraId, LoraMetadata, LoraRepository, Result};
use crate::base_model::BaseModelId;
use crate::storage::Storage;
use std::sync::Arc;

/// Service for managing LoRA models
#[derive(Clone)]
pub struct LoraService {
    repository: Arc<dyn LoraRepository>,
    storage: Arc<dyn Storage>,
}

impl LoraService {
    /// Create a new LoraService
    pub fn new(repository: Arc<dyn LoraRepository>, storage: Arc<dyn Storage>) -> Self {
        Self {
            repository,
            storage,
        }
    }

    /// Create a new LoRA
    pub async fn create(
        &self,
        name: String,
        description: Option<String>,
        base_model_id: Option<BaseModelId>,
        metadata: LoraMetadata,
    ) -> Result<Lora> {
        let lora = Lora::new(name, description, base_model_id, metadata);
        self.repository.create(lora).await
    }

    /// Get a LoRA by ID
    pub async fn get(&self, id: &LoraId) -> Result<Option<Lora>> {
        self.repository.get(id).await
    }

    /// Get a LoRA by name
    pub async fn get_by_name(&self, name: &str) -> Result<Option<Lora>> {
        self.repository.get_by_name(name).await
    }

    /// List all LoRAs
    pub async fn list(&self) -> Result<Vec<Lora>> {
        self.repository.list().await
    }

    /// Update a LoRA
    pub async fn update(&self, lora: Lora) -> Result<Lora> {
        self.repository.update(lora).await
    }

    /// Delete a LoRA
    pub async fn delete(&self, id: &LoraId) -> Result<()> {
        self.repository.delete(id).await
    }
}
