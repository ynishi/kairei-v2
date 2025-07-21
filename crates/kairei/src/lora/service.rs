use super::{Lora, LoraError, LoraId, LoraMetadata, LoraRepository, Result};
use crate::base_model::BaseModelId;
use crate::config::KaireiConfig;
use crate::storage::Storage;
use std::sync::Arc;

/// Service for managing LoRA models
#[derive(Clone)]
pub struct LoraService {
    repository: Arc<dyn LoraRepository>,
    storage: Arc<dyn Storage>,
    config: KaireiConfig,
}

impl LoraService {
    /// Create a new LoraService
    pub fn new(repository: Arc<dyn LoraRepository>, storage: Arc<dyn Storage>) -> Self {
        Self {
            repository,
            storage,
            config: KaireiConfig::default(),
        }
    }

    /// Create a new LoraService with config
    pub fn with_config(
        repository: Arc<dyn LoraRepository>,
        storage: Arc<dyn Storage>,
        config: KaireiConfig,
    ) -> Self {
        Self {
            repository,
            storage,
            config,
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
        // Create LoRA directory
        let lora_dir = format!("{}/{}", self.config.loras_dir, name);
        self.storage.create_dir(&lora_dir).await?;

        // Create LoRA entity
        let lora = Lora::new(name, description, base_model_id, metadata);

        // Save the entire Lora entity to meta.toml
        let meta_path = format!("{}/meta.toml", lora_dir);
        let meta_toml = toml::to_string_pretty(&lora)
            .map_err(|e| LoraError::InvalidMetadata(format!("Failed to serialize LoRA: {}", e)))?;

        self.storage.write(&meta_path, meta_toml.as_bytes()).await?;

        // Create in repository
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
    pub async fn update(&self, mut lora: Lora) -> Result<Lora> {
        // Update timestamp
        lora.updated_at = chrono::Utc::now().to_rfc3339();

        // Update in repository first
        let updated_lora = self.repository.update(lora).await?;

        // Update meta.toml
        let lora_dir = format!("{}/{}", self.config.loras_dir, updated_lora.name);
        let meta_path = format!("{}/meta.toml", lora_dir);

        let meta_toml = toml::to_string_pretty(&updated_lora)
            .map_err(|e| LoraError::InvalidMetadata(format!("Failed to serialize LoRA: {}", e)))?;

        self.storage.write(&meta_path, meta_toml.as_bytes()).await?;

        Ok(updated_lora)
    }

    /// Delete a LoRA completely (removes from DB and deletes files)
    pub async fn delete(&self, id: &LoraId) -> Result<()> {
        // Get LoRA info before deleting from repository
        let lora = self
            .repository
            .get(id)
            .await?
            .ok_or_else(|| LoraError::NotFound(id.to_string()))?;

        // Delete from repository
        self.repository.delete(id).await?;

        // Delete the entire LoRA directory
        let lora_dir = format!("{}/{}", self.config.loras_dir, lora.name);
        if self.storage.exists(&lora_dir).await? {
            self.storage.delete(&lora_dir).await?;
        }

        Ok(())
    }

    /// Archive a LoRA (marks as archived but keeps in DB and files)
    pub async fn archive(&self, id: &LoraId) -> Result<()> {
        // Get LoRA
        let mut lora = self
            .repository
            .get(id)
            .await?
            .ok_or_else(|| LoraError::NotFound(id.to_string()))?;

        // Mark as archived and update timestamp
        lora.archived = true;
        lora.updated_at = chrono::Utc::now().to_rfc3339();

        // Update through the regular update method
        self.update(lora).await?;

        Ok(())
    }

    /// Ensure LoRA directories exist
    pub async fn ensure_directories(&self) -> Result<()> {
        let lora_dir = &self.config.loras_dir;
        self.storage.create_dir(lora_dir).await?;
        Ok(())
    }

    /// Create a new LoRA with source file
    pub async fn create_with_source(
        &self,
        source_path: &str,
        name: String,
        description: Option<String>,
        base_model_id: Option<BaseModelId>,
        metadata: LoraMetadata,
    ) -> Result<Lora> {
        // Create LoRA directory
        let lora_dir = format!("{}/{}", self.config.loras_dir, name);

        // Check if LoRA already exists
        if self.storage.exists(&lora_dir).await? {
            return Err(LoraError::AlreadyExists(name));
        }

        // Create directory
        self.storage.create_dir(&lora_dir).await?;

        // Copy source file to adapter.safetensors
        let adapter_path = format!("{}/adapter.safetensors", lora_dir);

        // Read source file
        let source_content = self.storage.read(source_path).await.inspect_err(|e| {
            // Clean up directory on error
            let _ = self.storage.delete(&lora_dir);
        })?;

        // Write adapter file
        self.storage
            .write(&adapter_path, &source_content)
            .await
            .inspect_err(|e| {
                // Clean up directory on error
                let _ = self.storage.delete(&lora_dir);
            })?;

        // Get file size
        let file_metadata = self.storage.metadata(&adapter_path).await?;

        // Create LoRA entity with file information
        let mut lora = Lora::new(name, description, base_model_id, metadata);
        lora.file_path = Some(adapter_path);
        lora.size_bytes = Some(file_metadata.size);

        // Save the entire Lora entity to meta.toml
        let meta_path = format!("{}/meta.toml", lora_dir);
        let meta_toml = toml::to_string_pretty(&lora).map_err(|e| {
            // Clean up directory on error
            let _ = self.storage.delete(&lora_dir);
            LoraError::InvalidMetadata(format!("Failed to serialize LoRA: {}", e))
        })?;

        self.storage
            .write(&meta_path, meta_toml.as_bytes())
            .await
            .inspect_err(|e| {
                // Clean up directory on error
                let _ = self.storage.delete(&lora_dir);
            })?;

        // Create in repository
        self.repository.create(lora).await.inspect_err(|e| {
            // Clean up directory on error
            let _ = self.storage.delete(&lora_dir);
        })
    }

    /// Scan directory and register existing LoRAs
    pub async fn scan_and_register(&self) -> Result<Vec<Lora>> {
        self.ensure_directories().await?;

        let lora_base_dir = &self.config.loras_dir;
        let entries = self.storage.list(lora_base_dir).await?;

        let mut registered = Vec::new();

        for entry in entries {
            // Skip if not a directory
            let entry_path = format!("{}/{}", lora_base_dir, entry);
            let entry_metadata = self.storage.metadata(&entry_path).await?;
            if !entry_metadata.is_dir {
                continue;
            }

            // Extract name from directory
            let name = entry.clone();

            // Skip if already registered
            if let Some(_) = self.repository.get_by_name(&name).await? {
                continue;
            }

            // Check for meta.toml
            let meta_path = format!("{}/meta.toml", entry_path);
            if !self.storage.exists(&meta_path).await? {
                continue;
            }

            // Read Lora entity from meta.toml
            let meta_content = self.storage.read(&meta_path).await?;
            let mut lora: Lora =
                toml::from_str(&String::from_utf8_lossy(&meta_content)).map_err(|e| {
                    LoraError::InvalidMetadata(format!(
                        "Failed to parse meta.toml for {}: {}",
                        name, e
                    ))
                })?;

            // Check for adapter file and update file_path and size_bytes
            let adapter_path = format!("{}/adapter.safetensors", entry_path);
            if self.storage.exists(&adapter_path).await? {
                let file_metadata = self.storage.metadata(&adapter_path).await?;
                lora.file_path = Some(adapter_path);
                lora.size_bytes = Some(file_metadata.size);
            }

            registered.push(self.repository.create(lora).await?);
        }

        Ok(registered)
    }
}
