use crate::error::CliError;
use serde::Deserialize;
use std::fs;
use std::path::Path;

#[derive(Debug, Deserialize)]
struct LoraConfig {
    culture: CultureConfig,
    model: ModelConfig,
    training: TrainingConfig,
    lora: LoraSpecificConfig,
}

#[derive(Debug, Deserialize)]
struct CultureConfig {
    name: String,
    description: String,
}

#[derive(Debug, Deserialize)]
struct ModelConfig {
    base_model: String,
    model_path: String,
}

#[derive(Debug, Deserialize)]
struct TrainingConfig {
    epochs: usize,
    batch_size: usize,
    learning_rate: f64,
}

#[derive(Debug, Deserialize)]
struct LoraSpecificConfig {
    rank: usize,
    alpha: f64,
    dropout: f32,
    target_modules: Vec<String>,
}

pub async fn setup_lora() -> Result<(), CliError> {
    println!("🔧 Setting up LoRA development environment...");

    // Define directories to create
    let directories = vec!["loras", "lora_datasets", "base_models"];

    // Create each directory
    for dir in &directories {
        let path = Path::new(dir);
        if path.exists() {
            println!("✅ {} already exists", dir);
        } else {
            fs::create_dir_all(path).map_err(|e| {
                CliError::InvalidInput(format!("Failed to create directory {}: {}", dir, e))
            })?;
            println!("📁 Created {}/", dir);
        }
    }

    println!("\n✅ LoRA development environment setup complete!");
    println!("   You can now run: kairei lora:new <culture_name>");

    Ok(())
}

pub async fn lora_new(culture_name: &str) -> Result<(), CliError> {
    println!("✨ Creating new LoRA culture: {}", culture_name);

    // Check if loras directory exists
    let loras_dir = Path::new("loras");
    if !loras_dir.exists() {
        return Err(CliError::InvalidInput(
            "LoRA environment not initialized. Please run: kairei setup lora".to_string(),
        ));
    }

    // Create culture directory
    let culture_dir = loras_dir.join(culture_name);
    if culture_dir.exists() {
        return Err(CliError::InvalidInput(format!(
            "Culture '{}' already exists",
            culture_name
        )));
    }

    fs::create_dir_all(&culture_dir).map_err(|e| {
        CliError::InvalidInput(format!("Failed to create culture directory: {}", e))
    })?;
    println!("📁 Created loras/{}/", culture_name);

    // Create dataset directory
    let dataset_dir = culture_dir.join("dataset");
    fs::create_dir_all(&dataset_dir).map_err(|e| {
        CliError::InvalidInput(format!("Failed to create dataset directory: {}", e))
    })?;
    println!("📁 Created loras/{}/dataset/", culture_name);

    // Create config.toml
    let config_path = culture_dir.join("config.toml");
    let config_content = format!(
        r#"# LoRA Configuration for {}
[culture]
name = "{}"
description = "LoRA culture for {}"

[model]
base_model = "stories15M"  # Available: stories15M, stories42M, stories110M
model_path = "./base_models/stories15M.bin"

[training]
epochs = 5
batch_size = 4
learning_rate = 1e-4

[lora]
rank = 8
alpha = 16
dropout = 0.1
target_modules = ["q_proj", "v_proj"]  # Which layers to apply LoRA to
"#,
        culture_name, culture_name, culture_name
    );

    fs::write(&config_path, config_content)
        .map_err(|e| CliError::InvalidInput(format!("Failed to create config.toml: {}", e)))?;
    println!("📝 Created loras/{}/config.toml", culture_name);

    println!("\n✅ Culture '{}' created successfully!", culture_name);
    println!("\nNext steps:");
    println!("1. Add training data to: loras/{}/dataset/", culture_name);
    println!("2. Edit configuration: loras/{}/config.toml", culture_name);
    println!("3. Train the model: kairei lora:train {}", culture_name);

    Ok(())
}

pub async fn lora_list() -> Result<(), CliError> {
    println!("📋 Listing all LoRA cultures...\n");

    let loras_dir = Path::new("loras");
    if !loras_dir.exists() {
        println!("❌ LoRA environment not initialized. Please run: kairei setup lora");
        return Ok(());
    }

    let mut cultures = Vec::new();

    // Read all directories in loras/
    let entries = fs::read_dir(loras_dir)
        .map_err(|e| CliError::InvalidInput(format!("Failed to read loras directory: {}", e)))?;

    for entry in entries {
        let entry = entry.map_err(|e| {
            CliError::InvalidInput(format!("Failed to read directory entry: {}", e))
        })?;

        let path = entry.path();
        if path.is_dir() {
            if let Some(culture_name) = path.file_name() {
                if let Some(culture_name_str) = culture_name.to_str() {
                    // Check if config.toml exists (to verify it's a valid culture)
                    let config_path = path.join("config.toml");
                    let has_config = config_path.exists();

                    // Check if dataset directory exists
                    let dataset_path = path.join("dataset");
                    let has_dataset = dataset_path.exists();

                    // Count dataset files if directory exists
                    let dataset_count = if has_dataset {
                        fs::read_dir(&dataset_path)
                            .map(|entries| entries.count())
                            .unwrap_or(0)
                    } else {
                        0
                    };

                    cultures.push((culture_name_str.to_string(), has_config, dataset_count));
                }
            }
        }
    }

    if cultures.is_empty() {
        println!("No LoRA cultures found. Create one with: kairei lora:new <culture_name>");
    } else {
        println!("Found {} culture(s):", cultures.len());
        println!("─────────────────────────────────────────");

        for (name, has_config, dataset_count) in cultures {
            println!("📂 {}", name);
            if has_config {
                println!("   ✅ config.toml");
            } else {
                println!("   ❌ config.toml (missing)");
            }
            println!("   📊 {} dataset file(s)", dataset_count);
            println!();
        }
    }

    Ok(())
}

pub async fn lora_train(culture_name: &str) -> Result<(), CliError> {
    println!("🚀 Training LoRA culture: {}", culture_name);

    // Check if culture exists
    let culture_dir = Path::new("loras").join(culture_name);
    if !culture_dir.exists() {
        return Err(CliError::InvalidInput(format!(
            "Culture '{}' does not exist. Create it with: kairei lora new {}",
            culture_name, culture_name
        )));
    }

    // Read config.toml
    let config_path = culture_dir.join("config.toml");
    let config_content = fs::read_to_string(&config_path)
        .map_err(|e| CliError::InvalidInput(format!("Failed to read config.toml: {}", e)))?;

    let lora_config: LoraConfig = toml::from_str(&config_content)
        .map_err(|e| CliError::InvalidInput(format!("Failed to parse config.toml: {}", e)))?;

    println!("📋 Loaded configuration:");
    println!("   Culture: {}", lora_config.culture.name);
    println!("   Model: {}", lora_config.model.base_model);
    println!("   Epochs: {}", lora_config.training.epochs);
    println!("   Batch size: {}", lora_config.training.batch_size);
    println!("   Learning rate: {}", lora_config.training.learning_rate);
    println!("   LoRA rank: {}", lora_config.lora.rank);
    println!("   LoRA alpha: {}", lora_config.lora.alpha);

    // Create training config from loaded config
    let training_config = kairei::TrainingConfig {
        culture_name: lora_config.culture.name,
        epochs: lora_config.training.epochs,
        batch_size: lora_config.training.batch_size,
        learning_rate: lora_config.training.learning_rate,
        lora_rank: lora_config.lora.rank,
        lora_alpha: lora_config.lora.alpha,
        lora_dropout: Some(lora_config.lora.dropout),
    };

    // Call the training function through kairei
    kairei::train_lora(training_config)
        .map_err(|e| CliError::InvalidInput(format!("Training failed: {}", e)))?;

    println!("✅ Training completed!");
    Ok(())
}
