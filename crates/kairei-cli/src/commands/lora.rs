use crate::error::CliError;
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

pub async fn lora_list() -> Result<(), CliError> {
    println!("📋 Listing all registered LoRA models...\n");

    let loras_dir = Path::new("loras");
    if !loras_dir.exists() {
        println!("❌ LoRA environment not initialized. Please run: kairei setup lora");
        return Ok(());
    }

    let mut lora_list = Vec::new();

    // Read all directories in loras/
    let entries = fs::read_dir(loras_dir)
        .map_err(|e| CliError::InvalidInput(format!("Failed to read loras directory: {}", e)))?;

    for entry in entries {
        let entry = entry.map_err(|e| {
            CliError::InvalidInput(format!("Failed to read directory entry: {}", e))
        })?;

        let path = entry.path();
        if path.is_dir() {
            // Try to read meta.toml
            let meta_path = path.join("meta.toml");
            if meta_path.exists() {
                match fs::read_to_string(&meta_path) {
                    Ok(content) => {
                        match toml::from_str::<LoraMetadata>(&content) {
                            Ok(metadata) => {
                                // Check if the actual LoRA file exists
                                let lora_file = path.join("adapter.safetensors");
                                let file_exists = lora_file.exists();

                                lora_list.push((metadata, file_exists));
                            }
                            Err(_) => {
                                // Invalid metadata, skip
                                continue;
                            }
                        }
                    }
                    Err(_) => {
                        // Can't read metadata, skip
                        continue;
                    }
                }
            }
        }
    }

    if lora_list.is_empty() {
        println!("No LoRA models found. Add one with: kairei lora add <source>");
    } else {
        println!("Found {} LoRA model(s):", lora_list.len());
        println!("─────────────────────────────────────────");

        for (metadata, file_exists) in lora_list {
            println!("📦 {}", metadata.name);
            if let Some(base_model) = &metadata.base_model {
                println!("   🧠 Base: {}", base_model);
            }
            if let Some(desc) = &metadata.description {
                println!("   📝 {}", desc);
            }
            if !file_exists {
                println!("   ⚠️  adapter.safetensors missing!");
            }
            println!();
        }
    }

    Ok(())
}

// Add a LoRA model to the registry
pub async fn lora_add(
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
pub async fn lora_show(name: &str) -> Result<(), CliError> {
    let lora_dir = Path::new("loras").join(name);
    if !lora_dir.exists() {
        return Err(CliError::InvalidInput(format!("LoRA '{}' not found", name)));
    }

    // Read metadata
    let meta_path = lora_dir.join("meta.toml");
    if !meta_path.exists() {
        return Err(CliError::InvalidInput(format!(
            "Metadata not found for LoRA '{}'",
            name
        )));
    }

    let meta_content = fs::read_to_string(&meta_path)
        .map_err(|e| CliError::InvalidInput(format!("Failed to read metadata: {}", e)))?;
    let metadata: LoraMetadata = toml::from_str(&meta_content)
        .map_err(|e| CliError::InvalidInput(format!("Failed to parse metadata: {}", e)))?;

    // Check if LoRA file exists
    let lora_file = lora_dir.join("adapter.safetensors");
    let file_exists = lora_file.exists();
    let file_size = if file_exists {
        fs::metadata(&lora_file).ok().map(|m| m.len())
    } else {
        None
    };

    // Display information
    println!("📦 LoRA: {}", metadata.name);
    println!("─────────────────────────────────────────");

    if let Some(base_model) = &metadata.base_model {
        println!("🧠 Base Model: {}", base_model);
    }

    if let Some(desc) = &metadata.description {
        println!("📝 Description: {}", desc);
    }

    if let Some(created) = &metadata.created_at {
        println!("📅 Created: {}", created);
    }

    if let Some(rank) = metadata.rank {
        println!("🔢 Rank: {}", rank);
    }

    if let Some(alpha) = metadata.alpha {
        println!("🔢 Alpha: {}", alpha);
    }

    println!("📁 Location: {}", lora_dir.display());

    if file_exists {
        if let Some(size) = file_size {
            let size_mb = size as f64 / 1_048_576.0;
            println!("💾 File Size: {:.2} MB", size_mb);
        }
    } else {
        println!("⚠️  LoRA file (adapter.safetensors) is missing!");
    }

    Ok(())
}

// Remove a LoRA from the registry
pub async fn lora_remove(name: &str, keep_file: bool) -> Result<(), CliError> {
    let lora_dir = Path::new("loras").join(name);
    if !lora_dir.exists() {
        return Err(CliError::InvalidInput(format!("LoRA '{}' not found", name)));
    }

    if keep_file {
        println!("📁 Keeping the LoRA files...");

        // Just remove the metadata to "unregister" it
        let meta_path = lora_dir.join("meta.toml");
        if meta_path.exists() {
            fs::remove_file(&meta_path)
                .map_err(|e| CliError::InvalidInput(format!("Failed to remove metadata: {}", e)))?;
        }

        println!(
            "✅ LoRA '{}' unregistered (files kept in {})",
            name,
            lora_dir.display()
        );
    } else {
        println!("🗑️  Removing LoRA '{}' and all its files...", name);

        // Remove the entire directory
        fs::remove_dir_all(&lora_dir).map_err(|e| {
            CliError::InvalidInput(format!("Failed to remove LoRA directory: {}", e))
        })?;

        println!("✅ LoRA '{}' removed completely", name);
    }

    Ok(())
}
