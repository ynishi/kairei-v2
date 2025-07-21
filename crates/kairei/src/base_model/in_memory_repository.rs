use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::RwLock;

use super::{BaseModel, BaseModelError, BaseModelId, BaseModelRepository, BaseModelResult};

/// In-memory implementation of BaseModelRepository for testing and development
pub struct InMemoryBaseModelRepository {
    models: RwLock<HashMap<String, BaseModel>>,
}

impl InMemoryBaseModelRepository {
    pub fn new() -> Self {
        Self {
            models: RwLock::new(HashMap::new()),
        }
    }

    /// Initialize with some default models for testing
    pub fn with_defaults() -> Self {
        // Add default models here if needed
        Self::new()
    }
}

impl Default for InMemoryBaseModelRepository {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl BaseModelRepository for InMemoryBaseModelRepository {
    async fn list(&self) -> BaseModelResult<Vec<BaseModel>> {
        let models = self.models.read().map_err(|e| {
            BaseModelError::RepositoryError(format!("Failed to acquire read lock: {}", e))
        })?;
        Ok(models.values().cloned().collect())
    }

    async fn get(&self, id: &BaseModelId) -> BaseModelResult<BaseModel> {
        let models = self.models.read().map_err(|e| {
            BaseModelError::RepositoryError(format!("Failed to acquire read lock: {}", e))
        })?;
        models
            .get(id.as_str())
            .cloned()
            .ok_or_else(|| BaseModelError::NotFound(id.to_string()))
    }

    async fn get_by_name(&self, name: &str) -> BaseModelResult<BaseModel> {
        let models = self.models.read().map_err(|e| {
            BaseModelError::RepositoryError(format!("Failed to acquire read lock: {}", e))
        })?;
        models
            .values()
            .find(|model| model.name == name)
            .cloned()
            .ok_or_else(|| {
                BaseModelError::NotFound(format!("Model with name '{}' not found", name))
            })
    }

    async fn create(&self, model: BaseModel) -> BaseModelResult<BaseModel> {
        let mut models = self.models.write().map_err(|e| {
            BaseModelError::RepositoryError(format!("Failed to acquire write lock: {}", e))
        })?;

        if models.contains_key(model.id.as_str()) {
            return Err(BaseModelError::AlreadyExists(model.id.to_string()));
        }

        models.insert(model.id.to_string(), model.clone());
        Ok(model)
    }

    async fn update(&self, model: BaseModel) -> BaseModelResult<BaseModel> {
        let mut models = self.models.write().map_err(|e| {
            BaseModelError::RepositoryError(format!("Failed to acquire write lock: {}", e))
        })?;

        if !models.contains_key(model.id.as_str()) {
            return Err(BaseModelError::NotFound(model.id.to_string()));
        }

        models.insert(model.id.to_string(), model.clone());
        Ok(model)
    }

    async fn delete(&self, id: &BaseModelId) -> BaseModelResult<()> {
        let mut models = self.models.write().map_err(|e| {
            BaseModelError::RepositoryError(format!("Failed to acquire write lock: {}", e))
        })?;

        models
            .remove(id.as_str())
            .ok_or_else(|| BaseModelError::NotFound(id.to_string()))?;
        Ok(())
    }

    async fn exists(&self, id: &BaseModelId) -> BaseModelResult<bool> {
        let models = self.models.read().map_err(|e| {
            BaseModelError::RepositoryError(format!("Failed to acquire read lock: {}", e))
        })?;
        Ok(models.contains_key(id.as_str()))
    }

    async fn exists_by_name(&self, name: &str) -> BaseModelResult<bool> {
        let models = self.models.read().map_err(|e| {
            BaseModelError::RepositoryError(format!("Failed to acquire read lock: {}", e))
        })?;
        Ok(models.values().any(|model| model.name == name))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_create_and_get() {
        let repo = InMemoryBaseModelRepository::new();

        let model = BaseModel {
            id: BaseModelId::new(),
            name: "test-model".to_string(),
            description: Some("Test model".to_string()),
            repo_id: Some("test/repo".to_string()),
            filename: Some("model.bin".to_string()),
            size_mb: Some(100),
            metadata: None,
        };

        let created = repo.create(model.clone()).await.unwrap();
        assert_eq!(created.id, model.id);

        let retrieved = repo.get(&model.id).await.unwrap();
        assert_eq!(retrieved.name, model.name);
    }

    #[tokio::test]
    async fn test_get_by_name() {
        let repo = InMemoryBaseModelRepository::new();

        let model = BaseModel {
            id: BaseModelId::new(),
            name: "unique-model".to_string(),
            description: Some("Unique model".to_string()),
            repo_id: Some("test/repo".to_string()),
            filename: Some("model.bin".to_string()),
            size_mb: Some(100),
            metadata: None,
        };

        repo.create(model.clone()).await.unwrap();

        let retrieved = repo.get_by_name("unique-model").await.unwrap();
        assert_eq!(retrieved.id, model.id);
    }

    #[tokio::test]
    async fn test_update() {
        let repo = InMemoryBaseModelRepository::new();

        let mut model = BaseModel {
            id: BaseModelId::new(),
            name: "original-name".to_string(),
            description: Some("Original description".to_string()),
            repo_id: Some("test/repo".to_string()),
            filename: Some("model.bin".to_string()),
            size_mb: Some(100),
            metadata: None,
        };

        repo.create(model.clone()).await.unwrap();

        model.name = "updated-name".to_string();
        model.description = Some("Updated description".to_string());

        let updated = repo.update(model.clone()).await.unwrap();
        assert_eq!(updated.name, "updated-name");
        assert_eq!(updated.description, Some("Updated description".to_string()));
    }

    #[tokio::test]
    async fn test_delete() {
        let repo = InMemoryBaseModelRepository::new();

        let model = BaseModel {
            id: BaseModelId::new(),
            name: "to-delete".to_string(),
            description: Some("Will be deleted".to_string()),
            repo_id: Some("test/repo".to_string()),
            filename: Some("model.bin".to_string()),
            size_mb: Some(100),
            metadata: None,
        };

        repo.create(model.clone()).await.unwrap();
        assert!(repo.exists(&model.id).await.unwrap());

        repo.delete(&model.id).await.unwrap();
        assert!(!repo.exists(&model.id).await.unwrap());
    }
}
