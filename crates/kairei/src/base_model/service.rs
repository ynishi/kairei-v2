use std::sync::Arc;

use super::{
    BaseModel, BaseModelError, BaseModelId, BaseModelMetadata, BaseModelRepository,
    BaseModelResult, ModelDownloader,
};
use crate::config::KaireiConfig;
use crate::storage::Storage;

/// Service layer for BaseModel operations
#[derive(Clone)]
pub struct BaseModelService {
    repository: Arc<dyn BaseModelRepository>,
    storage: Arc<dyn Storage>,
    downloader: Arc<dyn ModelDownloader>,
    config: KaireiConfig,
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
            config: KaireiConfig::default(),
        }
    }

    /// Create a new BaseModelService with config
    pub fn with_config(
        repository: Arc<dyn BaseModelRepository>,
        storage: Arc<dyn Storage>,
        downloader: Arc<dyn ModelDownloader>,
        config: KaireiConfig,
    ) -> Self {
        Self {
            repository,
            storage,
            downloader,
            config,
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

        // Check if file exists in storage with new structure
        let model_dir = format!("{}/{}", self.config.models_dir, model.name);
        let file_path = format!("{}/{}", model_dir, filename);
        let meta_path = format!("{}/meta.toml", model_dir);

        if !force
            && self
                .storage
                .exists(&file_path)
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

        // Download the model file with new directory structure
        self.downloader
            .download_to_model_dir(&model.name, repo_id, filename, self.storage.as_ref(), force)
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

        // Create meta.toml content using toml serialization
        if let Some(ref metadata) = updated_model.metadata {
            let meta_toml = toml::to_string_pretty(metadata).map_err(|e| {
                BaseModelError::InvalidData(format!("Failed to serialize metadata: {}", e))
            })?;

            // Save meta.toml
            self.storage
                .write(&meta_path, meta_toml.as_bytes())
                .await
                .map_err(|e| {
                    BaseModelError::InvalidData(format!("Failed to write meta.toml: {}", e))
                })?;
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

        // Download model files with new directory structure
        let downloaded_files = self
            .downloader
            .download_files_to_model_dir(&name, &repo_id, &patterns, self.storage.as_ref(), force)
            .await?;

        // Find the main model file (largest one typically)
        let main_file = downloaded_files
            .iter()
            .find(|f| f.ends_with(".safetensors") || f.ends_with(".bin") || f.ends_with(".gguf"))
            .cloned()
            .or_else(|| downloaded_files.first().cloned())
            .and_then(|path| {
                // Extract just the filename from the full path
                path.split('/').last().map(|s| s.to_string())
            });

        // Try to read config.json to get model information
        let mut architecture = None;
        let mut parameters = None;

        let config_path = format!("{}/{}/config.json", self.config.models_dir, name);
        if let Ok(config_content) = self.storage.read(&config_path).await {
            if let Ok(config_str) = String::from_utf8(config_content) {
                if let Ok(config) = serde_json::from_str::<serde_json::Value>(&config_str) {
                    // Extract architecture info
                    if let Some(arch_value) = config.get("architectures") {
                        if let Some(arch_array) = arch_value.as_array() {
                            if let Some(first_arch) = arch_array.first() {
                                architecture = first_arch.as_str().map(|s| s.to_string());
                            }
                        }
                    }

                    // Try to estimate model size from parameters
                    if let Some(hidden_size) = config.get("hidden_size").and_then(|v| v.as_u64()) {
                        if let Some(num_layers) =
                            config.get("num_hidden_layers").and_then(|v| v.as_u64())
                        {
                            let approx_params = (hidden_size * num_layers * 4) / 1_000_000; // Very rough estimate
                            if approx_params < 1000 {
                                parameters = Some(format!("~{}M", approx_params));
                            } else {
                                parameters =
                                    Some(format!("~{:.1}B", approx_params as f64 / 1000.0));
                            }
                        }
                    }
                }
            }
        }

        // Create metadata
        let metadata = BaseModelMetadata {
            repo_id: repo_id.clone(),
            name: name.clone(),
            description: Some(format!("Downloaded from HuggingFace: {}", repo_id)),
            downloaded_at: Some(chrono::Utc::now().to_rfc3339()),
            parameters,
            architecture,
            quantization: None,
        };

        // Create meta.toml content using toml serialization
        let meta_toml = toml::to_string_pretty(&metadata).map_err(|e| {
            BaseModelError::InvalidData(format!("Failed to serialize metadata: {}", e))
        })?;

        // Save meta.toml
        let meta_path = format!("{}/{}/meta.toml", self.config.models_dir, name);
        self.storage
            .write(&meta_path, meta_toml.as_bytes())
            .await
            .map_err(|e| {
                BaseModelError::InvalidData(format!("Failed to write meta.toml: {}", e))
            })?;

        // Register the model
        let model = self
            .register_model(
                name.clone(),
                Some(format!("Custom model from {}", repo_id)),
                Some(repo_id.clone()),
                main_file,
                None, // Size could be calculated from files
                Some(metadata),
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

    /// Scan storage for models and register them in the repository
    pub async fn scan_and_register_models(&self) -> BaseModelResult<Vec<BaseModel>> {
        let mut registered_models = Vec::new();

        // List all directories in models/
        let models_dir = &self.config.models_dir;
        let entries = self.storage.list(models_dir).await.map_err(|e| {
            BaseModelError::InvalidData(format!("Failed to list models directory: {}", e))
        })?;

        for entry in entries {
            // Skip if not a directory (only process directories one level deep)
            if !entry.ends_with('/') {
                continue;
            }

            // Extract model name from path (remove "models/" prefix and trailing "/")
            let model_name = entry
                .strip_prefix(&format!("{}/", models_dir))
                .unwrap_or(&entry)
                .trim_end_matches('/')
                .to_string();

            // Skip if model already exists
            if self.repository.exists_by_name(&model_name).await? {
                println!("⏭️  Model '{}' already registered, skipping...", model_name);
                continue;
            }

            // Check for meta.toml
            let meta_path = format!("{}/{}/meta.toml", models_dir, model_name);
            if !self
                .storage
                .exists(&meta_path)
                .await
                .map_err(|e| BaseModelError::InvalidData(format!("Storage error: {}", e)))?
            {
                println!("⚠️  No meta.toml found for '{}', skipping...", model_name);
                continue;
            }

            // Read and parse meta.toml
            let meta_content = self.storage.read(&meta_path).await.map_err(|e| {
                BaseModelError::InvalidData(format!("Failed to read meta.toml: {}", e))
            })?;
            let meta_str = String::from_utf8(meta_content).map_err(|e| {
                BaseModelError::InvalidData(format!("Invalid UTF-8 in meta.toml: {}", e))
            })?;
            let metadata: BaseModelMetadata = toml::from_str(&meta_str).map_err(|e| {
                BaseModelError::InvalidData(format!("Failed to parse meta.toml: {}", e))
            })?;

            // Find the main model file
            let model_dir_path = format!("{}/{}", models_dir, model_name);
            let model_files = self.storage.list(&model_dir_path).await.map_err(|e| {
                BaseModelError::InvalidData(format!("Failed to list model directory: {}", e))
            })?;

            let main_file = model_files
                .iter()
                .find(|f| {
                    f.ends_with(".safetensors") || f.ends_with(".bin") || f.ends_with(".gguf")
                })
                .and_then(|f| f.split('/').last())
                .map(|s| s.to_string());

            // Register the model
            let model = self
                .register_model(
                    model_name.clone(),
                    metadata.description.clone(),
                    Some(metadata.repo_id.clone()),
                    main_file,
                    None, // size_mb can be calculated later if needed
                    Some(metadata),
                )
                .await?;

            println!("✅ Registered model: {}", model_name);
            registered_models.push(model);
        }

        Ok(registered_models)
    }
}
