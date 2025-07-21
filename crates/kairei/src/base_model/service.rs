use std::sync::Arc;

use super::{
    BaseModel, BaseModelError, BaseModelId, BaseModelMetadata, BaseModelRepository, BaseModelResult,
};

/// Service layer for BaseModel operations
#[derive(Clone)]
pub struct BaseModelService {
    repository: Arc<dyn BaseModelRepository>,
}

impl BaseModelService {
    /// Create a new BaseModelService
    pub fn new(repository: Arc<dyn BaseModelRepository>) -> Self {
        Self { repository }
    }

    /// List all available base models
    pub async fn list_models(&self) -> BaseModelResult<Vec<BaseModel>> {
        self.repository.list().await
    }

    /// Get a specific model by ID
    pub async fn get_model(&self, id: &BaseModelId) -> BaseModelResult<BaseModel> {
        self.repository.get(id).await
    }

    /// Get a specific model by name
    pub async fn get_model_by_name(&self, name: &str) -> BaseModelResult<BaseModel> {
        self.repository.get_by_name(name).await
    }

    /// Register a new base model
    pub async fn register_model(
        &self,
        name: String,
        description: String,
        repo_id: String,
        filename: String,
        size_mb: u64,
        metadata: Option<BaseModelMetadata>,
    ) -> BaseModelResult<BaseModel> {
        // Validate input
        if name.is_empty() {
            return Err(BaseModelError::InvalidData(
                "Model name cannot be empty".to_string(),
            ));
        }

        if repo_id.is_empty() {
            return Err(BaseModelError::InvalidData(
                "Repository ID cannot be empty".to_string(),
            ));
        }

        // Check if model with same name already exists
        if self.repository.exists_by_name(&name).await? {
            return Err(BaseModelError::AlreadyExists(name));
        }

        let model = BaseModel {
            id: BaseModelId::new(),
            name,
            description,
            repo_id,
            filename,
            size_mb,
            metadata,
        };

        self.repository.create(model).await
    }

    /// Update an existing model
    pub async fn update_model(
        &self,
        id: &BaseModelId,
        name: Option<String>,
        description: Option<String>,
        metadata: Option<BaseModelMetadata>,
    ) -> BaseModelResult<BaseModel> {
        let mut model = self.repository.get(id).await?;

        if let Some(name) = name {
            if name.is_empty() {
                return Err(BaseModelError::InvalidData(
                    "Model name cannot be empty".to_string(),
                ));
            }
            // Check if another model with the same name exists
            if let Ok(existing) = self.repository.get_by_name(&name).await {
                if existing.id != *id {
                    return Err(BaseModelError::AlreadyExists(name));
                }
            }
            model.name = name;
        }

        if let Some(description) = description {
            model.description = description;
        }

        if let Some(metadata) = metadata {
            model.metadata = Some(metadata);
        }

        self.repository.update(model).await
    }

    /// Delete a model
    pub async fn delete_model(&self, id: &BaseModelId) -> BaseModelResult<()> {
        // Check if model exists
        self.repository.get(id).await?;
        self.repository.delete(id).await
    }

    /// Check if a model is downloaded
    pub async fn is_model_downloaded(&self, id: &BaseModelId) -> BaseModelResult<bool> {
        let model = self.repository.get(id).await?;
        Ok(model
            .metadata
            .as_ref()
            .and_then(|m| m.downloaded_at.as_ref())
            .is_some())
    }
}
