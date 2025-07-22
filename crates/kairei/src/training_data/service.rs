use super::{
    DataFormat, DataType, Result, TrainingData, TrainingDataError, TrainingDataId,
    TrainingDataMetadata, TrainingDataRepository,
};
use crate::config::KaireiConfig;
use crate::storage::Storage;
use std::sync::Arc;

/// Service for managing training data
#[derive(Clone)]
pub struct TrainingDataService {
    repository: Arc<dyn TrainingDataRepository>,
    storage: Arc<dyn Storage>,
    config: KaireiConfig,
}

impl TrainingDataService {
    /// Create a new TrainingDataService
    pub fn new(repository: Arc<dyn TrainingDataRepository>, storage: Arc<dyn Storage>) -> Self {
        Self {
            repository,
            storage,
            config: KaireiConfig::default(),
        }
    }

    /// Create a new TrainingDataService with config
    pub fn with_config(
        repository: Arc<dyn TrainingDataRepository>,
        storage: Arc<dyn Storage>,
        config: KaireiConfig,
    ) -> Self {
        Self {
            repository,
            storage,
            config,
        }
    }

    /// Create a new training data entry
    pub async fn create(
        &self,
        name: String,
        description: Option<String>,
        data_type: DataType,
        format: DataFormat,
        metadata: TrainingDataMetadata,
    ) -> Result<TrainingData> {
        // Check if training data with the same name already exists
        if let Some(_) = self.repository.get_by_name(&name).await? {
            return Err(TrainingDataError::AlreadyExists(name));
        }

        // Create training data directory
        let data_dir = format!("{}/{}", self.config.datasets_dir, name);
        self.storage.create_dir(&data_dir).await?;

        // Create training data entity
        let data = TrainingData::new(name, description, data_type, format, metadata);

        // Save metadata to meta.toml
        let meta_path = format!("{}/meta.toml", data_dir);
        let meta_toml = toml::to_string_pretty(&data).map_err(|e| {
            TrainingDataError::SerializationError(format!(
                "Failed to serialize training data: {}",
                e
            ))
        })?;

        self.storage.write(&meta_path, meta_toml.as_bytes()).await?;

        // Create in repository
        self.repository.create(data).await
    }

    /// Upload data file
    pub async fn upload_file(
        &self,
        id: &TrainingDataId,
        file_name: &str,
        content: &[u8],
    ) -> Result<TrainingData> {
        // Get existing training data
        let mut data = self
            .repository
            .get(id)
            .await?
            .ok_or_else(|| TrainingDataError::NotFound(id.to_string()))?;

        // Check file size limit (e.g., 1GB)
        const MAX_FILE_SIZE: u64 = 1024 * 1024 * 1024; // 1GB
        let file_size = content.len() as u64;
        if file_size > MAX_FILE_SIZE {
            return Err(TrainingDataError::FileTooLarge(file_size, MAX_FILE_SIZE));
        }

        // Save file to storage
        let data_dir = format!("{}/{}", self.config.datasets_dir, data.name);
        let file_path = format!("{}/{}", data_dir, file_name);
        self.storage.write(&file_path, content).await?;

        // Calculate hash
        let hash = sha256::digest(content);

        // Update training data entity
        data.file_path = Some(file_path);
        data.size_bytes = Some(file_size);
        data.hash = Some(hash);
        data.updated_at = chrono::Utc::now().to_rfc3339();

        // Update metadata
        let meta_path = format!("{}/meta.toml", data_dir);
        let meta_toml = toml::to_string_pretty(&data).map_err(|e| {
            TrainingDataError::SerializationError(format!(
                "Failed to serialize training data: {}",
                e
            ))
        })?;
        self.storage.write(&meta_path, meta_toml.as_bytes()).await?;

        // Update in repository
        self.repository.update(data).await
    }

    /// Get training data by ID
    pub async fn get(&self, id: &TrainingDataId) -> Result<Option<TrainingData>> {
        self.repository.get(id).await
    }

    /// Get training data by name
    pub async fn get_by_name(&self, name: &str) -> Result<Option<TrainingData>> {
        self.repository.get_by_name(name).await
    }

    /// List all training data
    pub async fn list(&self) -> Result<Vec<TrainingData>> {
        self.repository.list().await
    }

    /// List training data by LoRA ID
    pub async fn list_by_lora(&self, lora_id: &str) -> Result<Vec<TrainingData>> {
        self.repository.list_by_lora(lora_id).await
    }

    /// Update training data
    pub async fn update(&self, mut data: TrainingData) -> Result<TrainingData> {
        // Update timestamp
        data.updated_at = chrono::Utc::now().to_rfc3339();

        // Update in repository first
        let updated_data = self.repository.update(data).await?;

        // Update meta.toml
        let data_dir = format!("{}/{}", self.config.datasets_dir, updated_data.name);
        let meta_path = format!("{}/meta.toml", data_dir);

        let meta_toml = toml::to_string_pretty(&updated_data).map_err(|e| {
            TrainingDataError::SerializationError(format!(
                "Failed to serialize training data: {}",
                e
            ))
        })?;

        self.storage.write(&meta_path, meta_toml.as_bytes()).await?;

        Ok(updated_data)
    }

    /// Delete training data
    pub async fn delete(&self, id: &TrainingDataId) -> Result<()> {
        // Get training data to find directory
        let data = self
            .repository
            .get(id)
            .await?
            .ok_or_else(|| TrainingDataError::NotFound(id.to_string()))?;

        // Delete from repository first
        self.repository.delete(id).await?;

        // Delete directory from storage
        let data_dir = format!("{}/{}", self.config.datasets_dir, data.name);
        self.storage.delete_dir(&data_dir).await?;

        Ok(())
    }

    /// Associate training data with a LoRA
    pub async fn add_lora_association(
        &self,
        id: &TrainingDataId,
        lora_id: &str,
    ) -> Result<TrainingData> {
        let mut data = self
            .repository
            .get(id)
            .await?
            .ok_or_else(|| TrainingDataError::NotFound(id.to_string()))?;

        if !data.lora_ids.contains(&lora_id.to_string()) {
            data.lora_ids.push(lora_id.to_string());
            self.update(data).await
        } else {
            Ok(data)
        }
    }

    /// Remove LoRA association from training data
    pub async fn remove_lora_association(
        &self,
        id: &TrainingDataId,
        lora_id: &str,
    ) -> Result<TrainingData> {
        let mut data = self
            .repository
            .get(id)
            .await?
            .ok_or_else(|| TrainingDataError::NotFound(id.to_string()))?;

        data.lora_ids.retain(|id| id != lora_id);
        self.update(data).await
    }
}
