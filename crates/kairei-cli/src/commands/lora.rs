use crate::error::CliError;
use kairei::base_model::BaseModelId;
use kairei::lora::{LoraMetadata, LoraService};
use std::path::Path;

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

    // Ensure directories are set up
    service
        .ensure_directories()
        .await
        .map_err(|e| CliError::InvalidInput(format!("Failed to ensure directories: {}", e)))?;

    // Check if source file exists
    let source_path = Path::new(source);
    if !source_path.exists() {
        return Err(CliError::InvalidInput(format!(
            "Source file does not exist: {}",
            source
        )));
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

    // Create metadata
    let metadata = LoraMetadata {
        rank: None,
        alpha: None,
        training_info: None,
        parent_lora_id: None,
        version: Some("1.0.0".to_string()),
        training_framework: None,
    };

    // Use the service to create LoRA with source
    let lora = service
        .create_with_source(
            source,
            lora_name.clone(),
            description,
            base_model.map(BaseModelId::from_string),
            metadata,
        )
        .await
        .map_err(|e| CliError::InvalidInput(format!("Failed to create LoRA: {}", e)))?;

    println!("✅ LoRA '{}' added successfully!", lora_name);
    if let Some(file_path) = &lora.file_path {
        println!("   📁 Location: {}", file_path);
    }

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
