use std::sync::Arc;

use super::{
    BaseModel, BaseModelError, BaseModelId, BaseModelMetadata, BaseModelRepository,
    BaseModelResult, ModelDownloader,
};
use crate::storage::Storage;

/// Service layer for BaseModel operations
#[derive(Clone)]
pub struct BaseModelService {
    repository: Arc<dyn BaseModelRepository>,
    storage: Arc<dyn Storage>,
    downloader: Arc<dyn ModelDownloader>,
}

impl BaseModelService {
    /// Create a new BaseModelService
    pub fn new(
        repository: Arc<dyn BaseModelRepository>,
        storage: Arc<dyn Storage>,
        downloader: Arc<dyn ModelDownloader>,
    ) -> Self {
        Self {
            repository,
            storage,
            downloader,
        }
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
        description: Option<String>,
        repo_id: Option<String>,
        filename: Option<String>,
        size_mb: Option<u64>,
        metadata: Option<BaseModelMetadata>,
    ) -> BaseModelResult<BaseModel> {
        // Validate input
        if name.is_empty() {
            return Err(BaseModelError::InvalidData(
                "Model name cannot be empty".to_string(),
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
            model.description = Some(description);
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

    /// Download a model from Hugging Face
    pub async fn download_model(&self, id: &BaseModelId, force: bool) -> BaseModelResult<()> {
        let model = self.repository.get(id).await?;

        // Check if already downloaded
        if !force && self.is_model_downloaded(id).await? {
            return Ok(());
        }

        // Validate model has repo_id and filename
        let repo_id = model
            .repo_id
            .as_ref()
            .ok_or_else(|| BaseModelError::InvalidData("Model has no repository ID".to_string()))?;

        let filename = model
            .filename
            .as_ref()
            .ok_or_else(|| BaseModelError::InvalidData("Model has no filename".to_string()))?;

        // Check if file exists in storage
        if !force
            && self
                .storage
                .exists(filename)
                .await
                .map_err(|e| BaseModelError::InvalidData(format!("Storage error: {}", e)))?
        {
            // File exists, just update metadata
            let mut updated_model = model.clone();
            if let Some(ref mut metadata) = updated_model.metadata {
                metadata.downloaded_at = Some(chrono::Utc::now().to_rfc3339());
            } else {
                updated_model.metadata = Some(BaseModelMetadata {
                    repo_id: repo_id.clone(),
                    name: updated_model.name.clone(),
                    description: updated_model.description.clone(),
                    downloaded_at: Some(chrono::Utc::now().to_rfc3339()),
                    parameters: None,
                    architecture: None,
                    quantization: None,
                });
            }
            self.repository.update(updated_model).await?;
            return Ok(());
        }

        // Download the model file
        self.downloader
            .download(repo_id, filename, self.storage.as_ref(), force)
            .await?;

        // Update metadata to mark as downloaded
        let mut updated_model = model.clone();
        if let Some(ref mut metadata) = updated_model.metadata {
            metadata.downloaded_at = Some(chrono::Utc::now().to_rfc3339());
        } else {
            updated_model.metadata = Some(BaseModelMetadata {
                repo_id: repo_id.clone(),
                name: updated_model.name.clone(),
                description: updated_model.description.clone(),
                downloaded_at: Some(chrono::Utc::now().to_rfc3339()),
                parameters: None,
                architecture: None,
                quantization: None,
            });
        }

        self.repository.update(updated_model).await?;
        Ok(())
    }

    /// Download a custom model from Hugging Face
    pub async fn download_custom_model(
        &self,
        name: String,
        repo_id: String,
        force: bool,
    ) -> BaseModelResult<BaseModel> {
        // Check if model already exists
        if !force {
            if let Ok(existing) = self.repository.get_by_name(&name).await {
                if self.is_model_downloaded(&existing.id).await? {
                    return Ok(existing);
                }
            }
        }

        // Common model file patterns
        let patterns = vec![
            "*.safetensors",
            "*.bin",
            "*.gguf",
            "tokenizer.json",
            "tokenizer_config.json",
            "tokenizer.model",
            "config.json",
        ];

        // Download model files
        let downloaded_files = self
            .downloader
            .download_files(&repo_id, &patterns, self.storage.as_ref(), force)
            .await?;

        // Find the main model file (largest one typically)
        let main_file = downloaded_files
            .iter()
            .find(|f| f.ends_with(".safetensors") || f.ends_with(".bin") || f.ends_with(".gguf"))
            .cloned()
            .or_else(|| downloaded_files.first().cloned());

        // Register the model
        let model = self
            .register_model(
                name.clone(),
                Some(format!("Custom model from {}", repo_id)),
                Some(repo_id.clone()),
                main_file,
                None, // Size could be calculated from files
                Some(BaseModelMetadata {
                    repo_id: repo_id.clone(),
                    name: name.clone(),
                    description: Some(format!("Downloaded from HuggingFace: {}", repo_id)),
                    downloaded_at: Some(chrono::Utc::now().to_rfc3339()),
                    parameters: None,
                    architecture: None,
                    quantization: None,
                }),
            )
            .await?;

        Ok(model)
    }

    /// List downloaded models
    pub async fn list_downloaded_models(&self) -> BaseModelResult<Vec<BaseModel>> {
        // Get all models from repository
        let all_models = self.repository.list().await?;

        // Filter only downloaded ones
        let mut downloaded_models = Vec::new();
        for model in all_models {
            if self.is_model_downloaded(&model.id).await? {
                downloaded_models.push(model);
            }
        }

        Ok(downloaded_models)
    }
}
