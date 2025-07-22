//! Trainer service implementation

use super::{Result, TrainParams, TrainerError, TrainingResult, TrainingStatus};
use crate::config::KaireiConfig;
use crate::storage::Storage;
use chrono::Local;
use std::fs;
use std::path::Path;
use std::process::Command;
use std::sync::Arc;

/// Service for training LoRA models
#[derive(Clone)]
pub struct TrainerService {
    storage: Arc<dyn Storage>,
    config: KaireiConfig,
}

impl TrainerService {
    /// Create a new TrainerService
    pub fn new(storage: Arc<dyn Storage>) -> Self {
        Self {
            storage,
            config: KaireiConfig::default(),
        }
    }

    /// Create a new TrainerService with config
    pub fn with_config(storage: Arc<dyn Storage>, config: KaireiConfig) -> Self {
        Self { storage, config }
    }

    /// Train a LoRA model
    pub async fn train_lora(&self, params: TrainParams) -> Result<TrainingResult> {
        // Generate output directory with timestamp if not specified
        let output_dir = params.output_dir.clone().unwrap_or_else(|| {
            let timestamp = Local::now().format("%Y%m%d_%H%M%S");
            let model_name = params
                .model
                .split('/')
                .last()
                .unwrap_or(&params.model)
                .replace('.', "_");
            format!("training/lora_output/{}_{}", model_name, timestamp)
        });

        // Create output directory if it doesn't exist
        fs::create_dir_all(&output_dir).map_err(|e| {
            TrainerError::FileSystem(format!("Failed to create output directory: {}", e))
        })?;

        // Check if training script exists
        let script_path = "training/scripts/train_lora.py";
        if !Path::new(script_path).exists() {
            return Err(TrainerError::ScriptNotFound(format!(
                "{}. Make sure you're in the project root directory.",
                script_path
            )));
        }

        // Check if poetry is available
        let poetry_check = Command::new("poetry").arg("--version").output();
        if poetry_check.is_err() {
            return Err(TrainerError::PoetryNotFound);
        }

        // Resolve model name
        let resolved_model = self.resolve_model_name(&params.model).await?;

        // Adjust paths for poetry execution
        let adjusted_train_data = self.adjust_training_path(&params.train_data);
        let adjusted_output_dir = self.adjust_training_path(&output_dir);

        // Build and run the command
        let status = self.run_training_command(
            &resolved_model,
            &adjusted_train_data,
            &adjusted_output_dir,
            &params,
        )?;

        if !status.success() {
            return Ok(TrainingResult {
                output_dir,
                status: TrainingStatus::Failed,
                final_loss: None,
                duration: None,
                error: Some("Training process failed".to_string()),
                checkpoints: vec![],
            });
        }

        // TODO: Parse training logs to extract final_loss and checkpoints
        Ok(TrainingResult {
            output_dir,
            status: TrainingStatus::Completed,
            final_loss: None,
            duration: None,
            error: None,
            checkpoints: vec![],
        })
    }

    /// Resolve model name (check if it's a registered model)
    async fn resolve_model_name(&self, model: &str) -> Result<String> {
        if model.contains('/') {
            // Already looks like a HuggingFace ID
            return Ok(model.to_string());
        }

        // Check if it's a registered model
        let model_path = format!("models/{}", model);
        if self.storage.exists(&model_path).await.unwrap_or(false) {
            // Try to extract the repo_id from meta.toml
            let meta_path = format!("{}/meta.toml", model_path);
            if let Ok(meta_content) = self.storage.read(&meta_path).await {
                if let Ok(meta_str) = String::from_utf8(meta_content) {
                    if let Ok(metadata) = toml::from_str::<toml::Value>(&meta_str) {
                        if let Some(repo_id) = metadata.get("repo_id").and_then(|v| v.as_str()) {
                            return Ok(repo_id.to_string());
                        }
                    }
                }
            }
        }

        // Fallback to original model string
        Ok(model.to_string())
    }

    /// Adjust path for poetry execution
    fn adjust_training_path(&self, path: &str) -> String {
        if path.starts_with("training/") {
            path.strip_prefix("training/").unwrap().to_string()
        } else {
            path.to_string()
        }
    }

    /// Run the training command
    fn run_training_command(
        &self,
        model: &str,
        train_data: &str,
        output_dir: &str,
        params: &TrainParams,
    ) -> Result<std::process::ExitStatus> {
        let mut cmd = Command::new("poetry");
        cmd.current_dir("training")
            .arg("run")
            .arg("python")
            .arg("scripts/train_lora.py")
            .arg("--train-data")
            .arg(train_data)
            .arg("--epochs")
            .arg(params.epochs.to_string())
            .arg("--lora-r")
            .arg(params.lora_r.to_string())
            .arg("--lora-alpha")
            .arg(params.lora_alpha.to_string())
            .arg("--output-dir")
            .arg(output_dir)
            .arg("--model")
            .arg(model)
            .arg("--batch-size")
            .arg(params.batch_size.to_string())
            .arg("--learning-rate")
            .arg(params.learning_rate.to_string());

        if params.test {
            cmd.arg("--test");
        }

        cmd.status()
            .map_err(|e| TrainerError::TrainingFailed(format!("Failed to run training: {}", e)))
    }
}
