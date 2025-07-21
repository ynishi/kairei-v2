use async_trait::async_trait;
use std::path::{Path, PathBuf};
use tokio::fs;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

use super::traits::FileMetadata;
use super::{Storage, StorageResult};
use crate::config::KaireiConfig;

/// Local filesystem storage implementation
#[derive(Debug, Clone)]
pub struct LocalStorage {
    base_path: PathBuf,
}

impl LocalStorage {
    /// Create a new LocalStorage instance
    pub fn new(base_path: impl AsRef<Path>) -> Self {
        Self {
            base_path: base_path.as_ref().to_path_buf(),
        }
    }

    /// Create LocalStorage from KaireiConfig
    pub fn from_config(config: &KaireiConfig) -> Self {
        Self::new(&config.base_dir)
    }

    /// Get full path by joining base path with relative path
    fn full_path(&self, path: &str) -> PathBuf {
        self.base_path.join(path)
    }
}

#[async_trait]
impl Storage for LocalStorage {
    async fn exists(&self, path: &str) -> StorageResult<bool> {
        let full_path = self.full_path(path);
        Ok(full_path.exists())
    }

    async fn read(&self, path: &str) -> StorageResult<Vec<u8>> {
        let full_path = self.full_path(path);
        let mut file = fs::File::open(&full_path).await?;
        let mut content = Vec::new();
        file.read_to_end(&mut content).await?;
        Ok(content)
    }

    async fn write(&self, path: &str, content: &[u8]) -> StorageResult<()> {
        let full_path = self.full_path(path);

        // Create parent directory if it doesn't exist
        if let Some(parent) = full_path.parent() {
            fs::create_dir_all(parent).await?;
        }

        let mut file = fs::File::create(&full_path).await?;
        file.write_all(content).await?;
        file.sync_all().await?;
        Ok(())
    }

    async fn delete(&self, path: &str) -> StorageResult<()> {
        let full_path = self.full_path(path);
        if full_path.is_dir() {
            fs::remove_dir_all(&full_path).await?;
        } else {
            fs::remove_file(&full_path).await?;
        }
        Ok(())
    }

    async fn list(&self, prefix: &str) -> StorageResult<Vec<String>> {
        let full_path = self.full_path(prefix);

        if !full_path.exists() {
            return Ok(Vec::new());
        }

        let mut entries = Vec::new();
        let mut read_dir = fs::read_dir(&full_path).await?;

        while let Some(entry) = read_dir.next_entry().await? {
            if let Ok(relative) = entry.path().strip_prefix(&self.base_path) {
                let mut path_str = relative.to_string_lossy().to_string();

                // Add trailing slash for directories
                if entry.path().is_dir() {
                    path_str.push('/');
                }

                entries.push(path_str);
            }
        }

        Ok(entries)
    }

    async fn create_dir(&self, path: &str) -> StorageResult<()> {
        let full_path = self.full_path(path);
        fs::create_dir_all(&full_path).await?;
        Ok(())
    }

    async fn copy(&self, source: &str, destination: &str) -> StorageResult<()> {
        let source_path = self.full_path(source);
        let dest_path = self.full_path(destination);

        // Create parent directory if it doesn't exist
        if let Some(parent) = dest_path.parent() {
            fs::create_dir_all(parent).await?;
        }

        fs::copy(&source_path, &dest_path).await?;
        Ok(())
    }

    async fn rename(&self, source: &str, destination: &str) -> StorageResult<()> {
        let source_path = self.full_path(source);
        let dest_path = self.full_path(destination);

        // Create parent directory if it doesn't exist
        if let Some(parent) = dest_path.parent() {
            fs::create_dir_all(parent).await?;
        }

        fs::rename(&source_path, &dest_path).await?;
        Ok(())
    }

    async fn metadata(&self, path: &str) -> StorageResult<FileMetadata> {
        let full_path = self.full_path(path);
        let metadata = fs::metadata(&full_path).await?;

        let modified = metadata.modified().ok().and_then(|time| {
            time.duration_since(std::time::UNIX_EPOCH)
                .ok()
                .and_then(|duration| chrono::DateTime::from_timestamp(duration.as_secs() as i64, 0))
        });

        Ok(FileMetadata {
            size: metadata.len(),
            modified,
            is_dir: metadata.is_dir(),
        })
    }
}
