use async_trait::async_trait;
use hf_hub::api::tokio::{Api, ApiBuilder};

use super::{BaseModelError, BaseModelResult};
use crate::storage::Storage;

/// Model downloader trait
#[async_trait]
pub trait ModelDownloader: Send + Sync {
    /// Download a model from repository
    async fn download(
        &self,
        repo_id: &str,
        filename: &str,
        storage: &dyn Storage,
        force: bool,
    ) -> BaseModelResult<String>;

    /// Download multiple files from repository
    async fn download_files(
        &self,
        repo_id: &str,
        patterns: &[&str],
        storage: &dyn Storage,
        force: bool,
    ) -> BaseModelResult<Vec<String>>;
}

/// HuggingFace model downloader
pub struct HuggingFaceDownloader {
    api_token: Option<String>,
}

impl HuggingFaceDownloader {
    /// Create a new HuggingFace downloader
    pub fn new(api_token: Option<String>) -> Self {
        Self { api_token }
    }

    /// Create HuggingFace API client
    fn create_api(&self) -> BaseModelResult<Api> {
        let mut builder = ApiBuilder::new();

        if let Some(token) = &self.api_token {
            builder = builder.with_token(Some(token.clone()));
        }

        builder
            .build()
            .map_err(|e| BaseModelError::DownloadError(format!("Failed to create HF API: {}", e)))
    }
}

#[async_trait]
impl ModelDownloader for HuggingFaceDownloader {
    async fn download(
        &self,
        repo_id: &str,
        filename: &str,
        storage: &dyn Storage,
        force: bool,
    ) -> BaseModelResult<String> {
        // Check if file already exists
        if !force
            && storage
                .exists(filename)
                .await
                .map_err(|e| BaseModelError::DownloadError(format!("Storage error: {}", e)))?
        {
            return Ok(filename.to_string());
        }

        // Create API client
        let api = self.create_api()?;
        let repo = api.model(repo_id.to_string());

        // Download file to local temp path
        let downloaded_path = repo
            .download(filename)
            .await
            .map_err(|e| BaseModelError::DownloadError(format!("Download failed: {}", e)))?;

        // Read the downloaded file
        let content = tokio::fs::read(&downloaded_path).await.map_err(|e| {
            BaseModelError::DownloadError(format!("Failed to read downloaded file: {}", e))
        })?;

        // Write to storage
        storage.write(filename, &content).await.map_err(|e| {
            BaseModelError::DownloadError(format!("Failed to write to storage: {}", e))
        })?;

        Ok(filename.to_string())
    }

    async fn download_files(
        &self,
        repo_id: &str,
        patterns: &[&str],
        storage: &dyn Storage,
        force: bool,
    ) -> BaseModelResult<Vec<String>> {
        let api = self.create_api()?;
        let repo = api.model(repo_id.to_string());

        // Get repository info
        let info = repo.info().await.map_err(|e| {
            BaseModelError::DownloadError(format!("Failed to get repo info: {}", e))
        })?;

        let mut downloaded_files = Vec::new();

        // Filter files by patterns
        for sibling in &info.siblings {
            let filename = &sibling.rfilename;

            // Check if file matches any pattern
            let should_download = patterns.iter().any(|pattern| {
                if pattern.starts_with("*.") {
                    filename.ends_with(&pattern[1..])
                } else {
                    filename == pattern
                }
            });

            if should_download {
                // Check if file already exists
                if !force
                    && storage.exists(filename).await.map_err(|e| {
                        BaseModelError::DownloadError(format!("Storage error: {}", e))
                    })?
                {
                    downloaded_files.push(filename.clone());
                    continue;
                }

                // Download file
                let downloaded_path = repo.download(filename).await.map_err(|e| {
                    BaseModelError::DownloadError(format!("Failed to download {}: {}", filename, e))
                })?;

                // Read the downloaded file
                let content = tokio::fs::read(&downloaded_path).await.map_err(|e| {
                    BaseModelError::DownloadError(format!("Failed to read downloaded file: {}", e))
                })?;

                // Write to storage
                storage.write(filename, &content).await.map_err(|e| {
                    BaseModelError::DownloadError(format!("Failed to write to storage: {}", e))
                })?;

                downloaded_files.push(filename.clone());
            }
        }

        if downloaded_files.is_empty() {
            return Err(BaseModelError::DownloadError(
                "No files found matching the patterns".to_string(),
            ));
        }

        Ok(downloaded_files)
    }
}
