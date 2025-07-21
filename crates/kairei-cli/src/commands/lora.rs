use crate::error::CliError;
use kairei::lora::LoraService;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

// LoRA metadata stored as meta.toml in each LoRA directory
#[derive(Debug, Serialize, Deserialize)]
struct LoraMetadata {
    name: String,
    base_model: Option<String>,
    description: Option<String>,
    created_at: Option<String>,
    // LoRA specific params
    rank: Option<usize>,
    alpha: Option<f64>,
    // Training information
    training_data: Option<String>,
    training_data_hash: Option<String>,
    epochs: Option<u32>,
    batch_size: Option<u32>,
    learning_rate: Option<f32>,
    // Lineage tracking
    parent_lora: Option<String>,
    source_model: Option<String>,
    // Performance metrics
    final_loss: Option<f32>,
    training_duration: Option<String>,
    // Version tracking
    version: Option<String>,
    training_framework: Option<String>,
}

pub async fn setup_lora(service: &LoraService) -> Result<(), CliError> {
    println!("🔧 Setting up LoRA development environment...");

    // Ensure LoRA directories exist through service
    service
        .ensure_directories()
        .await
        .map_err(|e| CliError::InvalidInput(format!("Failed to create directories: {}", e)))?;

    // Also create additional directories that might not be managed by the service
    let additional_dirs = vec!["lora_datasets", "base_models"];
    for dir in &additional_dirs {
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

pub async fn lora_list(service: &LoraService) -> Result<(), CliError> {
    println!("📋 Listing all registered LoRA models...\n");

    // Get all LoRAs from service
    let loras = service
        .list()
        .await
        .map_err(|e| CliError::InvalidInput(format!("Failed to list LoRAs: {}", e)))?;

    if loras.is_empty() {
        println!("No LoRA models found. Add one with: kairei lora add <source>");
    } else {
        println!("Found {} LoRA model(s):", loras.len());
        println!("─────────────────────────────────────────");

        for lora in loras {
            println!("📦 {}", lora.name);
            if let Some(base_model_id) = &lora.base_model_id {
                println!("   🧠 Base: {}", base_model_id);
            }
            if let Some(desc) = &lora.description {
                println!("   📝 {}", desc);
            }
            if lora.file_path.is_none() {
                println!("   ⚠️  adapter.safetensors missing!");
            }
            println!();
        }
    }

    Ok(())
}

// Add a LoRA model to the registry
pub async fn lora_add(
    service: &LoraService,
    source: &str,
    name: Option<String>,
    base_model: Option<String>,
    description: Option<String>,
) -> Result<(), CliError> {
    println!("📥 Adding LoRA model from: {}", source);

    // Ensure loras directory exists
    let loras_dir = Path::new("loras");
    if !loras_dir.exists() {
        return Err(CliError::InvalidInput(
            "LoRA environment not initialized. Please run: kairei setup lora".to_string(),
        ));
    }

    // Determine the LoRA name
    let lora_name = if let Some(n) = name {
        n
    } else {
        // Extract name from source filename
        Path::new(source)
            .file_stem()
            .and_then(|s| s.to_str())
            .map(|s| s.to_string())
            .ok_or_else(|| {
                CliError::InvalidInput("Cannot determine LoRA name from source".to_string())
            })?
    };

    // Create directory for this LoRA
    let lora_dir = loras_dir.join(&lora_name);
    if lora_dir.exists() {
        return Err(CliError::InvalidInput(format!(
            "LoRA '{}' already exists",
            lora_name
        )));
    }
    fs::create_dir_all(&lora_dir)
        .map_err(|e| CliError::InvalidInput(format!("Failed to create LoRA directory: {}", e)))?;

    // Copy the LoRA file
    let source_path = Path::new(source);
    if !source_path.exists() {
        // Clean up directory on error
        let _ = fs::remove_dir_all(&lora_dir);
        return Err(CliError::InvalidInput(format!(
            "Source file does not exist: {}",
            source
        )));
    }

    let dest_path = lora_dir.join("adapter.safetensors");
    fs::copy(source_path, &dest_path).map_err(|e| {
        // Clean up directory on error
        let _ = fs::remove_dir_all(&lora_dir);
        CliError::InvalidInput(format!("Failed to copy LoRA file: {}", e))
    })?;

    // Create metadata
    let metadata = LoraMetadata {
        name: lora_name.clone(),
        base_model,
        description,
        created_at: Some(chrono::Utc::now().to_rfc3339()),
        rank: None,
        alpha: None,
        // Training information
        training_data: None,
        training_data_hash: None,
        epochs: None,
        batch_size: None,
        learning_rate: None,
        // Lineage tracking
        parent_lora: None,
        source_model: None,
        // Performance metrics
        final_loss: None,
        training_duration: None,
        // Version tracking
        version: Some("1.0.0".to_string()),
        training_framework: None,
    };

    // Write metadata
    let meta_path = lora_dir.join("meta.toml");
    let meta_content = toml::to_string_pretty(&metadata)
        .map_err(|e| CliError::InvalidInput(format!("Failed to serialize metadata: {}", e)))?;
    fs::write(&meta_path, meta_content).map_err(|e| {
        // Clean up directory on error
        let _ = fs::remove_dir_all(&lora_dir);
        CliError::InvalidInput(format!("Failed to write metadata: {}", e))
    })?;

    println!("✅ LoRA '{}' added successfully!", lora_name);
    println!("   📁 Location: {}", lora_dir.display());

    Ok(())
}

// Show information about a specific LoRA
pub async fn lora_show(service: &LoraService, name: &str) -> Result<(), CliError> {
    // Get LoRA from service by name
    let lora = service
        .get_by_name(name)
        .await
        .map_err(|e| CliError::InvalidInput(format!("Failed to get LoRA: {}", e)))?
        .ok_or_else(|| CliError::InvalidInput(format!("LoRA '{}' not found", name)))?;

    // Display as TOML
    println!("📦 LoRA: {}", lora.name);
    println!("─────────────────────────────────────────");

    let lora_toml = toml::to_string_pretty(&lora)
        .map_err(|e| CliError::InvalidInput(format!("Failed to serialize LoRA: {}", e)))?;

    println!("{}", lora_toml);

    Ok(())
}

// Remove a LoRA from the registry
pub async fn lora_remove(
    service: &LoraService,
    name: &str,
    keep_file: bool,
) -> Result<(), CliError> {
    // Get LoRA by name first
    let lora = service
        .get_by_name(name)
        .await
        .map_err(|e| CliError::InvalidInput(format!("Failed to get LoRA: {}", e)))?
        .ok_or_else(|| CliError::InvalidInput(format!("LoRA '{}' not found", name)))?;

    if keep_file {
        println!("📁 Archiving LoRA '{}' (keeping files)...", name);

        // Archive the LoRA
        service
            .archive(&lora.id)
            .await
            .map_err(|e| CliError::InvalidInput(format!("Failed to archive LoRA: {}", e)))?;

        println!("✅ LoRA '{}' archived (files kept)", name);
    } else {
        println!("🗑️  Removing LoRA '{}' and all its files...", name);

        // Delete the LoRA completely
        service
            .delete(&lora.id)
            .await
            .map_err(|e| CliError::InvalidInput(format!("Failed to delete LoRA: {}", e)))?;

        println!("✅ LoRA '{}' removed completely", name);
    }

    Ok(())
}

pub async fn convert_peft_to_candle_lora(
    peft_dir: String,
    output_path: String,
    prefix: Option<String>,
) -> Result<(), CliError> {
    use candle_core::Device;
    use candle_lora::convert_peft_dir_to_candle_lora_typed;

    println!("🔄 Converting PEFT to candle-lora format...");

    let device = Device::Cpu;
    // Note: prefix is ignored in the typed conversion as it automatically determines prefixes
    if prefix.is_some() {
        println!(
            "⚠️  Note: prefix parameter is ignored in typed conversion (auto-determined by layer type)"
        );
    }

    // Use the new typed conversion function with dummy embeddings enabled
    convert_peft_dir_to_candle_lora_typed(&peft_dir, &output_path, &device, true)
        .map_err(|e| CliError::InvalidInput(format!("Conversion failed: {}", e)))?;

    Ok(())
}

// Update LoRA metadata with training information
pub async fn update_lora_training_info(
    lora_name: &str,
    training_data: &str,
    epochs: u32,
    batch_size: u32,
    learning_rate: f32,
    final_loss: Option<f32>,
    training_duration: Option<String>,
    source_model: &str,
) -> Result<(), CliError> {
    let lora_dir = Path::new("loras").join(lora_name);
    let meta_path = lora_dir.join("meta.toml");

    if !meta_path.exists() {
        return Err(CliError::InvalidInput(format!(
            "Metadata not found for LoRA '{}'",
            lora_name
        )));
    }

    // Read existing metadata
    let meta_content = fs::read_to_string(&meta_path)
        .map_err(|e| CliError::InvalidInput(format!("Failed to read metadata: {}", e)))?;
    let mut metadata: LoraMetadata = toml::from_str(&meta_content)
        .map_err(|e| CliError::InvalidInput(format!("Failed to parse metadata: {}", e)))?;

    // Calculate training data hash
    let training_data_hash = if Path::new(training_data).exists() {
        let data_content = fs::read_to_string(training_data).ok();
        data_content.map(|content| {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            let mut hasher = DefaultHasher::new();
            content.hash(&mut hasher);
            format!("{:x}", hasher.finish())
        })
    } else {
        None
    };

    // Update metadata
    metadata.training_data = Some(training_data.to_string());
    metadata.training_data_hash = training_data_hash;
    metadata.epochs = Some(epochs);
    metadata.batch_size = Some(batch_size);
    metadata.learning_rate = Some(learning_rate);
    metadata.final_loss = final_loss;
    metadata.training_duration = training_duration;
    metadata.source_model = Some(source_model.to_string());
    metadata.training_framework = Some("transformers/peft".to_string());

    // Write updated metadata
    let meta_content = toml::to_string_pretty(&metadata)
        .map_err(|e| CliError::InvalidInput(format!("Failed to serialize metadata: {}", e)))?;
    fs::write(&meta_path, meta_content)
        .map_err(|e| CliError::InvalidInput(format!("Failed to write metadata: {}", e)))?;

    println!("📝 Updated training info for LoRA '{}'", lora_name);

    Ok(())
}
