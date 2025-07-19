use crate::error::CliError;
use std::fs;
use std::path::Path;

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
