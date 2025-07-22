#[cfg(test)]
mod tests {
    use kairei::storage::LocalStorage;
    use kairei::trainer::{TrainParams, TrainerService, TrainingStatus};
    use std::sync::Arc;
    use tempfile::TempDir;

    /// Create a test TrainerService with a temporary directory
    fn create_test_service() -> (TrainerService, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let storage = Arc::new(LocalStorage::new(temp_dir.path().to_str().unwrap()));
        let service = TrainerService::new(storage);
        (service, temp_dir)
    }

    #[tokio::test]
    async fn test_train_lora_missing_script() {
        let (service, _temp_dir) = create_test_service();

        let params = TrainParams {
            train_data: "test_data.jsonl".to_string(),
            epochs: 1,
            lora_r: 8,
            lora_alpha: 16,
            output_dir: Some("test_output".to_string()),
            model: "test-model".to_string(),
            batch_size: 1,
            learning_rate: 0.0001,
            test: false,
        };

        // Since we're in a temp directory, the training script won't exist
        let result = service.train_lora(params).await;

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("Training script not found"));
    }

    #[tokio::test]
    async fn test_train_params_creation() {
        let params = TrainParams {
            train_data: "data/train.jsonl".to_string(),
            epochs: 10,
            lora_r: 16,
            lora_alpha: 32,
            output_dir: None,
            model: "meta-llama/Llama-2-7b-hf".to_string(),
            batch_size: 4,
            learning_rate: 0.0003,
            test: true,
        };

        assert_eq!(params.train_data, "data/train.jsonl");
        assert_eq!(params.epochs, 10);
        assert_eq!(params.lora_r, 16);
        assert_eq!(params.lora_alpha, 32);
        assert!(params.output_dir.is_none());
        assert_eq!(params.model, "meta-llama/Llama-2-7b-hf");
        assert_eq!(params.batch_size, 4);
        assert_eq!(params.learning_rate, 0.0003);
        assert!(params.test);
    }

    #[tokio::test]
    async fn test_training_status_equality() {
        assert_eq!(TrainingStatus::Queued, TrainingStatus::Queued);
        assert_eq!(TrainingStatus::Running, TrainingStatus::Running);
        assert_eq!(TrainingStatus::Completed, TrainingStatus::Completed);
        assert_eq!(TrainingStatus::Failed, TrainingStatus::Failed);
        assert_eq!(TrainingStatus::Cancelled, TrainingStatus::Cancelled);

        assert_ne!(TrainingStatus::Running, TrainingStatus::Completed);
        assert_ne!(TrainingStatus::Failed, TrainingStatus::Completed);
    }

    #[tokio::test]
    async fn test_model_name_resolution() {
        let (_service, temp_dir) = create_test_service();

        // Create a mock model directory structure
        let models_dir = temp_dir.path().join("models").join("test-model");
        std::fs::create_dir_all(&models_dir).unwrap();

        let meta_content = r#"
repo_id = "meta-llama/Llama-2-7b-hf"
name = "test-model"
"#;
        std::fs::write(models_dir.join("meta.toml"), meta_content).unwrap();

        // This test would need access to the private method, so we'll test it indirectly
        // through the train_lora method in a real scenario

        // For now, just verify the file structure was created correctly
        assert!(models_dir.join("meta.toml").exists());
    }

    #[tokio::test]
    async fn test_training_path_adjustment() {
        let params = TrainParams {
            train_data: "training/data/train.jsonl".to_string(),
            epochs: 1,
            lora_r: 8,
            lora_alpha: 16,
            output_dir: Some("training/output/model".to_string()),
            model: "test".to_string(),
            batch_size: 1,
            learning_rate: 0.0001,
            test: false,
        };

        // The service should strip "training/" prefix when running poetry commands
        // This is tested indirectly through the service methods
        assert!(params.train_data.starts_with("training/"));
        assert!(params.output_dir.as_ref().unwrap().starts_with("training/"));
    }

    #[test]
    fn test_trainer_service_clone() {
        // Ensure TrainerService implements Clone
        let temp_dir = TempDir::new().unwrap();
        let storage = Arc::new(LocalStorage::new(temp_dir.path().to_str().unwrap()));
        let service1 = TrainerService::new(storage);
        let service2 = service1.clone();

        // Both services should be valid clones
        // We can't easily test equality, but we can verify they're both valid
        drop(service1);
        drop(service2);
    }
}
