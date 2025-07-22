use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

use super::{Result, TrainingData, TrainingDataError, TrainingDataId, TrainingDataRepository};

/// In-memory implementation of TrainingDataRepository
#[derive(Debug, Clone)]
pub struct InMemoryTrainingDataRepository {
    data: Arc<RwLock<HashMap<TrainingDataId, TrainingData>>>,
}

impl InMemoryTrainingDataRepository {
    /// Create a new in-memory repository
    pub fn new() -> Self {
        Self {
            data: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

impl Default for InMemoryTrainingDataRepository {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl TrainingDataRepository for InMemoryTrainingDataRepository {
    async fn create(&self, data: TrainingData) -> Result<TrainingData> {
        let mut storage = self.data.write().await;

        // Check if already exists
        if storage.contains_key(&data.id) {
            return Err(TrainingDataError::AlreadyExists(data.id.to_string()));
        }

        // Check for duplicate name
        for existing in storage.values() {
            if existing.name == data.name {
                return Err(TrainingDataError::AlreadyExists(data.name.clone()));
            }
        }

        storage.insert(data.id.clone(), data.clone());
        Ok(data)
    }

    async fn get(&self, id: &TrainingDataId) -> Result<Option<TrainingData>> {
        let storage = self.data.read().await;
        Ok(storage.get(id).cloned())
    }

    async fn get_by_name(&self, name: &str) -> Result<Option<TrainingData>> {
        let storage = self.data.read().await;
        Ok(storage.values().find(|d| d.name == name).cloned())
    }

    async fn list(&self) -> Result<Vec<TrainingData>> {
        let storage = self.data.read().await;
        let mut data_list: Vec<TrainingData> = storage.values().cloned().collect();
        data_list.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        Ok(data_list)
    }

    async fn list_by_lora(&self, lora_id: &str) -> Result<Vec<TrainingData>> {
        let storage = self.data.read().await;
        let mut data_list: Vec<TrainingData> = storage
            .values()
            .filter(|d| d.lora_ids.contains(&lora_id.to_string()))
            .cloned()
            .collect();
        data_list.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        Ok(data_list)
    }

    async fn update(&self, data: TrainingData) -> Result<TrainingData> {
        let mut storage = self.data.write().await;

        if !storage.contains_key(&data.id) {
            return Err(TrainingDataError::NotFound(data.id.to_string()));
        }

        storage.insert(data.id.clone(), data.clone());
        Ok(data)
    }

    async fn delete(&self, id: &TrainingDataId) -> Result<()> {
        let mut storage = self.data.write().await;

        if storage.remove(id).is_none() {
            return Err(TrainingDataError::NotFound(id.to_string()));
        }

        Ok(())
    }

    async fn exists(&self, id: &TrainingDataId) -> Result<bool> {
        let storage = self.data.read().await;
        Ok(storage.contains_key(id))
    }
}
