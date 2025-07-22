//! Train command implementation

use crate::error::CliError;
use kairei::storage::LocalStorage;
use kairei::trainer::{TrainParams, TrainerService, TrainingStatus};
use std::sync::Arc;

/// Run the training script with specified parameters
pub async fn run_train(
    train_data: String,
    epochs: u32,
    lora_r: u32,
    lora_alpha: u32,
    output_dir: Option<String>,
    model: String,
    batch_size: u32,
    learning_rate: f32,
    test: bool,
) -> Result<(), CliError> {
    println!("🎯 Starting LoRA training...\n");

    // Create storage and trainer service
    let storage = Arc::new(LocalStorage::new("."));
    let trainer = TrainerService::new(storage);

    // Create training parameters
    let params = TrainParams {
        train_data,
        epochs,
        lora_r,
        lora_alpha,
        output_dir,
        model,
        batch_size,
        learning_rate,
        test,
    };

    // Print training configuration
    println!("📝 Training configuration:");
    println!("  Model: {}", params.model);
    println!("  Training data: {}", params.train_data);
    println!("  Epochs: {}", params.epochs);
    println!("  LoRA rank: {}", params.lora_r);
    println!("  LoRA alpha: {}", params.lora_alpha);
    println!("  Batch size: {}", params.batch_size);
    println!("  Learning rate: {}", params.learning_rate);
    println!(
        "  Output: {}",
        params
            .output_dir
            .as_ref()
            .unwrap_or(&"auto-generated".to_string())
    );
    println!("  Test after training: {}\n", params.test);

    // Run the training
    println!("⚡ Running training script...\n");

    match trainer.train_lora(params).await {
        Ok(result) => {
            if result.status == TrainingStatus::Completed {
                println!("\n✅ Training completed successfully!");
                println!("📁 Model saved to: {}", result.output_dir);

                // If the training was successful, suggest converting to candle format
                println!("\n💡 To use the trained model with Kairei:");
                println!("   1. Convert to candle format:");
                println!(
                    "      kairei lora convert {} --output my_lora.safetensors",
                    result.output_dir
                );
                println!("   2. Add to registry:");
                println!("      kairei lora add my_lora.safetensors --name my_culture");
                println!("   3. Use in chat:");
                println!("      kairei chat --lora my_culture");

                Ok(())
            } else {
                let error_msg = result.error.unwrap_or_else(|| "Unknown error".to_string());
                Err(CliError::InvalidInput(format!(
                    "Training failed: {}",
                    error_msg
                )))
            }
        }
        Err(e) => Err(CliError::InvalidInput(format!("Training error: {}", e))),
    }
}
