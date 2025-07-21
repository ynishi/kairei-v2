use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

use super::{Lora, LoraError, LoraId, LoraRepository, Result};

/// In-memory implementation of LoraRepository
#[derive(Clone)]
pub struct InMemoryLoraRepository {
    loras: Arc<RwLock<HashMap<LoraId, Lora>>>,
}

impl InMemoryLoraRepository {
    /// Create a new in-memory LoRA repository
    pub fn new() -> Self {
        Self {
            loras: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

impl Default for InMemoryLoraRepository {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl LoraRepository for InMemoryLoraRepository {
    async fn create(&self, lora: Lora) -> Result<Lora> {
        let mut loras = self.loras.write().await;

        // Check if name already exists
        for existing_lora in loras.values() {
            if existing_lora.name == lora.name {
                return Err(LoraError::AlreadyExists(lora.name));
            }
        }

        let id = lora.id.clone();
        loras.insert(id, lora.clone());
        Ok(lora)
    }

    async fn get(&self, id: &LoraId) -> Result<Option<Lora>> {
        let loras = self.loras.read().await;
        Ok(loras.get(id).cloned())
    }

    async fn get_by_name(&self, name: &str) -> Result<Option<Lora>> {
        let loras = self.loras.read().await;
        Ok(loras.values().find(|l| l.name == name).cloned())
    }

    async fn list(&self) -> Result<Vec<Lora>> {
        let loras = self.loras.read().await;
        Ok(loras.values().cloned().collect())
    }

    async fn update(&self, lora: Lora) -> Result<Lora> {
        let mut loras = self.loras.write().await;

        if !loras.contains_key(&lora.id) {
            return Err(LoraError::NotFound(lora.id.to_string()));
        }

        // Check if name conflict with other loras
        for (existing_id, existing_lora) in loras.iter() {
            if existing_lora.name == lora.name && existing_id != &lora.id {
                return Err(LoraError::AlreadyExists(lora.name));
            }
        }

        loras.insert(lora.id.clone(), lora.clone());
        Ok(lora)
    }

    async fn delete(&self, id: &LoraId) -> Result<()> {
        let mut loras = self.loras.write().await;

        if loras.remove(id).is_none() {
            return Err(LoraError::NotFound(id.to_string()));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lora::{LoraMetadata, LoraStatus};

    fn create_test_metadata() -> LoraMetadata {
        LoraMetadata {
            name: "test-lora".to_string(),
            base_model_id: None,
            description: Some("Test LoRA".to_string()),
            created_at: chrono::Utc::now().to_rfc3339(),
            rank: Some(16),
            alpha: Some(32.0),
            training_info: None,
            parent_lora_id: None,
            version: Some("1.0.0".to_string()),
            training_framework: None,
        }
    }

    #[tokio::test]
    async fn test_create_and_get() {
        let repo = InMemoryLoraRepository::new();
        let metadata = create_test_metadata();
        let lora = Lora::new("test-lora".to_string(), metadata);
        let id = lora.id.clone();

        // Create
        let created = repo.create(lora).await.unwrap();
        assert_eq!(created.name, "test-lora");

        // Get by ID
        let retrieved = repo.get(&id).await.unwrap().unwrap();
        assert_eq!(retrieved.id, id);
        assert_eq!(retrieved.name, "test-lora");

        // Get by name
        let by_name = repo.get_by_name("test-lora").await.unwrap().unwrap();
        assert_eq!(by_name.id, id);
    }

    #[tokio::test]
    async fn test_duplicate_name_error() {
        let repo = InMemoryLoraRepository::new();
        let metadata = create_test_metadata();
        let lora1 = Lora::new("test-lora".to_string(), metadata.clone());
        let lora2 = Lora::new("test-lora".to_string(), metadata);

        repo.create(lora1).await.unwrap();
        let result = repo.create(lora2).await;

        assert!(matches!(result, Err(LoraError::AlreadyExists(_))));
    }

    #[tokio::test]
    async fn test_update() {
        let repo = InMemoryLoraRepository::new();
        let metadata = create_test_metadata();
        let mut lora = Lora::new("test-lora".to_string(), metadata);
        let id = lora.id.clone();

        repo.create(lora.clone()).await.unwrap();

        // Update status
        lora.status = LoraStatus::Training;
        let updated = repo.update(lora).await.unwrap();
        assert_eq!(updated.status, LoraStatus::Training);

        // Verify update persisted
        let retrieved = repo.get(&id).await.unwrap().unwrap();
        assert_eq!(retrieved.status, LoraStatus::Training);
    }

    #[tokio::test]
    async fn test_delete() {
        let repo = InMemoryLoraRepository::new();
        let metadata = create_test_metadata();
        let lora = Lora::new("test-lora".to_string(), metadata);
        let id = lora.id.clone();

        repo.create(lora).await.unwrap();
        repo.delete(&id).await.unwrap();

        let result = repo.get(&id).await.unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_list() {
        let repo = InMemoryLoraRepository::new();
        let metadata = create_test_metadata();

        let lora1 = Lora::new("lora1".to_string(), metadata.clone());
        let lora2 = Lora::new("lora2".to_string(), metadata);

        repo.create(lora1).await.unwrap();
        repo.create(lora2).await.unwrap();

        let list = repo.list().await.unwrap();
        assert_eq!(list.len(), 2);
    }
}
