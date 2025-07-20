//! Train command implementation

use crate::error::CliError;
use std::path::Path;
use std::process::Command;

/// Run the training script with specified parameters
pub async fn run_train(
    train_data: String,
    epochs: u32,
    lora_r: u32,
    lora_alpha: u32,
    output_dir: String,
    model: String,
    batch_size: u32,
    learning_rate: f32,
    test: bool,
) -> Result<(), CliError> {
    println!("🎯 Starting LoRA training...\n");

    // Check if training script exists
    let script_path = "training/scripts/train_lora.py";
    if !Path::new(script_path).exists() {
        return Err(CliError::InvalidInput(format!(
            "Training script not found at {}. Make sure you're in the project root directory.",
            script_path
        )));
    }

    // Check if poetry is available
    let poetry_check = Command::new("poetry").arg("--version").output();

    if poetry_check.is_err() {
        return Err(CliError::InvalidInput(
            "Poetry not found. Please install Poetry to run training.".to_string(),
        ));
    }

    // Resolve model name (check if it's a registered model)
    let resolved_model = if !model.contains('/') {
        // Might be a registered model name, check if it exists
        let model_path = Path::new("models").join(&model);
        if model_path.exists() {
            // Check for meta.toml to get the HuggingFace ID
            let meta_path = model_path.join("meta.toml");
            if meta_path.exists() {
                // Try to extract the repo_id from meta.toml
                if let Ok(meta_content) = std::fs::read_to_string(&meta_path) {
                    if let Ok(metadata) = toml::from_str::<toml::Value>(&meta_content) {
                        if let Some(repo_id) = metadata.get("repo_id").and_then(|v| v.as_str()) {
                            println!("📦 Resolved '{}' to HuggingFace model: {}", model, repo_id);
                            repo_id.to_string()
                        } else {
                            // Fallback: use the original model string
                            model.clone()
                        }
                    } else {
                        model.clone()
                    }
                } else {
                    model.clone()
                }
            } else {
                model.clone()
            }
        } else {
            // Not a registered model, assume it's a HuggingFace ID
            model.clone()
        }
    } else {
        // Already looks like a HuggingFace ID
        model.clone()
    };

    // Adjust train_data path if it starts with "training/"
    let adjusted_train_data = if train_data.starts_with("training/") {
        train_data.strip_prefix("training/").unwrap().to_string()
    } else {
        train_data.clone()
    };

    // Build the command
    let mut cmd = Command::new("poetry");
    cmd.current_dir("training")
        .arg("run")
        .arg("python")
        .arg("scripts/train_lora.py")
        .arg("--train-data")
        .arg(&adjusted_train_data)
        .arg("--epochs")
        .arg(epochs.to_string())
        .arg("--lora-r")
        .arg(lora_r.to_string())
        .arg("--lora-alpha")
        .arg(lora_alpha.to_string())
        .arg("--output-dir")
        .arg(&output_dir)
        .arg("--model")
        .arg(&resolved_model)
        .arg("--batch-size")
        .arg(batch_size.to_string())
        .arg("--learning-rate")
        .arg(learning_rate.to_string());

    if test {
        cmd.arg("--test");
    }

    println!("📝 Training configuration:");
    println!("  Model: {}", resolved_model);
    println!("  Training data: {}", train_data);
    println!("  Epochs: {}", epochs);
    println!("  LoRA rank: {}", lora_r);
    println!("  LoRA alpha: {}", lora_alpha);
    println!("  Batch size: {}", batch_size);
    println!("  Learning rate: {}", learning_rate);
    println!("  Output: {}", output_dir);
    println!("  Test after training: {}\n", test);

    // Run the training
    println!("⚡ Running training script...\n");

    let status = cmd
        .status()
        .map_err(|e| CliError::InvalidInput(format!("Failed to run training: {}", e)))?;

    if !status.success() {
        return Err(CliError::InvalidInput("Training failed".to_string()));
    }

    println!("\n✅ Training completed successfully!");
    println!("📁 Model saved to: {}", output_dir);

    // If the training was successful, suggest converting to candle format
    println!("\n💡 To use the trained model with Kairei:");
    println!("   1. Convert to candle format:");
    println!(
        "      kairei lora convert {} --output my_lora.safetensors",
        output_dir
    );
    println!("   2. Add to registry:");
    println!("      kairei lora add my_lora.safetensors --name my_culture");
    println!("   3. Use in chat:");
    println!("      kairei chat --lora my_culture");

    Ok(())
}
