//! Setup command handler

use crate::error::CliError;
use chrono::Utc;
use hf_hub::api::tokio::Api;
use kairei::base_model::{
    BaseModelMetadata, BaseModelService, HuggingFaceDownloader, InMemoryBaseModelRepository,
};
use kairei::storage::LocalStorage;
use std::fs;
use std::path::Path;
use std::sync::Arc;

pub async fn run_setup(
    list: bool,
    models: bool,
    model: Option<String>,
    force: bool,
    name: Option<String>,
    repo_id: Option<String>,
) -> Result<(), CliError> {
    // Initialize the base model service with local storage and downloader
    let repository = Arc::new(InMemoryBaseModelRepository::new());
    let storage = Arc::new(LocalStorage::new("models"));
    let downloader = Arc::new(HuggingFaceDownloader::new(None)); // No API token for now
    let service = BaseModelService::new(repository, storage, downloader);
    if list {
        list_models(&service).await?;
        return Ok(());
    }

    if models {
        list_downloaded_models(&service).await?;
        return Ok(());
    }

    // Create necessary directories
    let _models_dir = Path::new("models");
    let directories = vec!["models", "loras", "lora_datasets", "base_models"];
    for dir in &directories {
        let path = Path::new(dir);
        if !path.exists() {
            println!("📁 Creating {} directory...", dir);
            fs::create_dir_all(path).map_err(|e| {
                CliError::InvalidInput(format!("Failed to create {} directory: {}", dir, e))
            })?;
        }
    }

    // Check if this is a custom model download
    if let (Some(custom_name), Some(custom_repo)) = (name.clone(), repo_id.clone()) {
        // Download custom HuggingFace model
        println!(
            "\n🤖 Downloading custom model: {} from {}",
            custom_name, custom_repo
        );

        let model = service
            .download_custom_model(custom_name.clone(), custom_repo.clone(), force)
            .await
            .map_err(|e| CliError::InvalidInput(e.to_string()))?;

        println!("✅ Downloaded {} successfully!", custom_name);
        println!("\n🎉 Setup complete! You can now use your custom model.");
        return Ok(());
    }

    // If no specific name/repo provided and no specific model requested, show help
    if name.is_none() && repo_id.is_none() && model.is_none() {
        println!("🎯 No model specified. Use one of the following:");
        println!("  kairei setup --name <name> --repo-id <repo>  # Download custom model");
        println!("  kairei setup --list                           # List downloaded models");
        println!("  kairei setup --models                         # Show downloaded models");
        return Ok(());
    }

    // Download specific model if model name is provided
    if let Some(_model_name) = model {
        return Err(CliError::InvalidInput(
            "Model download by name is not supported. Please use --name and --repo-id".to_string(),
        ));
    }

    Ok(())
}

async fn list_models(service: &BaseModelService) -> Result<(), CliError> {
    println!("📦 Downloaded models:");
    println!("===================");
    println!();

    let models = service
        .list_models()
        .await
        .map_err(|e| CliError::InvalidInput(e.to_string()))?;

    if models.is_empty() {
        println!("  No models registered yet.");
        println!();
        println!("To download a model:");
        println!("  kairei setup --name <name> --repo-id <huggingface-repo>");
    } else {
        for model in models {
            println!(
                "  {} - {}",
                model.name,
                model.description.as_deref().unwrap_or("No description")
            );
            if let Some(repo_id) = &model.repo_id {
                println!("       Repo: {}", repo_id);
            }
            if let Some(filename) = &model.filename {
                println!(
                    "       File: {} (~{}MB)",
                    filename,
                    model.size_mb.unwrap_or(0)
                );
            }
            if service
                .is_model_downloaded(&model.id)
                .await
                .map_err(|e| CliError::InvalidInput(e.to_string()))?
            {
                println!("       Status: ✅ Downloaded");
            } else {
                println!("       Status: ❌ Not downloaded");
            }
            println!();
        }
    }

    Ok(())
}

async fn list_downloaded_models(service: &BaseModelService) -> Result<(), CliError> {
    let models_dir = Path::new("models");

    if !models_dir.exists() {
        println!("📁 No models directory found. Run 'kairei setup' first to download models.");
        return Ok(());
    }

    println!("📦 Downloaded models:");
    println!("===================");
    println!();

    let downloaded_models = service
        .list_downloaded_models()
        .await
        .map_err(|e| CliError::InvalidInput(e.to_string()))?;

    if downloaded_models.is_empty() {
        println!("  No models found. Run 'kairei setup' to download models.");
    } else {
        for model in downloaded_models {
            println!(
                "  {} - {}",
                model.name,
                model.description.as_deref().unwrap_or("No description")
            );
            if let Some(repo_id) = &model.repo_id {
                println!("       Repo: {}", repo_id);
            }
            if let Some(metadata) = &model.metadata {
                if let Some(params) = &metadata.parameters {
                    println!("       Size: {}", params);
                }
                if let Some(arch) = &metadata.architecture {
                    println!("       Architecture: {}", arch);
                }
                if let Some(downloaded_at) = &metadata.downloaded_at {
                    println!("       Downloaded: {}", downloaded_at);
                }
            }
        }
    }

    // Also check filesystem for legacy models
    let mut found_models = false;

    // Check for llama2c style models (*.bin files)
    if let Ok(entries) = fs::read_dir(models_dir) {
        for entry in entries {
            if let Ok(entry) = entry {
                let path = entry.path();

                // Check for .bin files (llama2c models)
                if path.extension().and_then(|s| s.to_str()) == Some("bin") {
                    if let Some(filename) = path.file_name().and_then(|s| s.to_str()) {
                        println!("  {} (llama2c model)", filename);
                        found_models = true;
                    }
                }

                // Check for directories with meta.toml
                if path.is_dir() {
                    let meta_path = path.join("meta.toml");

                    if meta_path.exists() {
                        // Try to read metadata
                        match fs::read_to_string(&meta_path) {
                            Ok(content) => {
                                match toml::from_str::<BaseModelMetadata>(&content) {
                                    Ok(metadata) => {
                                        println!(
                                            "  {} - {}",
                                            metadata.name,
                                            metadata
                                                .description
                                                .as_deref()
                                                .unwrap_or("No description")
                                        );
                                        println!("       Repo: {}", metadata.repo_id);
                                        if let Some(params) = &metadata.parameters {
                                            println!("       Size: {}", params);
                                        }
                                        if let Some(arch) = &metadata.architecture {
                                            println!("       Architecture: {}", arch);
                                        }
                                        if let Some(downloaded_at) = &metadata.downloaded_at {
                                            println!("       Downloaded: {}", downloaded_at);
                                        }
                                    }
                                    Err(_) => {
                                        // Fallback to directory name if metadata is invalid
                                        if let Some(dirname) =
                                            path.file_name().and_then(|s| s.to_str())
                                        {
                                            println!("  {} (metadata error)", dirname);
                                        }
                                    }
                                }
                            }
                            Err(_) => {
                                // Fallback to directory name if can't read metadata
                                if let Some(dirname) = path.file_name().and_then(|s| s.to_str()) {
                                    println!("  {} (no metadata)", dirname);
                                }
                            }
                        }
                        found_models = true;
                    } else {
                        // Check if it's a model directory without meta.toml
                        let has_model_files =
                            ["model.safetensors", "pytorch_model.bin", "config.json"]
                                .iter()
                                .any(|file| path.join(file).exists());

                        if has_model_files {
                            if let Some(dirname) = path.file_name().and_then(|s| s.to_str()) {
                                println!("  {} (no metadata)", dirname);
                                found_models = true;
                            }
                        }
                    }
                }
            }
        }
    }

    // Check for tokenizer
    let tokenizer_path = models_dir.join("tokenizer.json");
    if tokenizer_path.exists() {
        println!("  tokenizer.json (LLaMA tokenizer)");
        found_models = true;
    }

    if !found_models {
        println!("  No models found. Run 'kairei setup' to download models.");
    }

    println!();
    println!("To download more models:");
    println!("  kairei setup --list                              # See available models");
    println!("  kairei setup <name> <repo_id>                    # Download from HuggingFace");
    println!();
    println!("To train with a model:");
    println!("  kairei train -m <model_name> -d <data.json>      # Use model name from above");

    Ok(())
}

async fn download_model_from_hf(
    repo_id: &str,
    filename: &str,
    target_path: &Path,
) -> Result<(), CliError> {
    // Create parent directory if needed
    if let Some(parent) = target_path.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| CliError::InvalidInput(format!("Failed to create directory: {}", e)))?;
    }

    // Use hf-hub to download the file
    let api = Api::new().map_err(|e| {
        CliError::InvalidInput(format!("Failed to create HuggingFace API client: {}", e))
    })?;

    let repo = api.model(repo_id.to_string());

    println!("Connecting to HuggingFace Hub...");

    let downloaded_path = repo
        .download(filename)
        .await
        .map_err(|e| CliError::InvalidInput(format!("Download failed: {}", e)))?;

    // Copy to target location
    fs::copy(&downloaded_path, target_path)
        .map_err(|e| CliError::InvalidInput(format!("Failed to copy file: {}", e)))?;

    Ok(())
}

async fn download_custom_model(
    _name: &str,
    repo_id: &str,
    target_dir: &Path,
) -> Result<(), CliError> {
    // Create target directory
    fs::create_dir_all(target_dir)
        .map_err(|e| CliError::InvalidInput(format!("Failed to create directory: {}", e)))?;

    // Use hf-hub to download model files
    let api = Api::new().map_err(|e| {
        CliError::InvalidInput(format!("Failed to create HuggingFace API client: {}", e))
    })?;

    let repo = api.model(repo_id.to_string());

    println!("📥 Downloading model files from {}...", repo_id);

    // List and download important model files
    let model_patterns = [
        "*.safetensors",
        "*.bin",
        "*.gguf",
        "tokenizer.json",
        "tokenizer_config.json",
        "tokenizer.model",
        "config.json",
    ];

    let info = repo
        .info()
        .await
        .map_err(|e| CliError::InvalidInput(format!("Failed to get repo info: {}", e)))?;

    let mut downloaded_files = 0;

    for sibling in &info.siblings {
        let filename = &sibling.rfilename;

        // Check if file matches our patterns
        let should_download = model_patterns.iter().any(|pattern| {
            if pattern.starts_with("*.") {
                filename.ends_with(&pattern[1..])
            } else {
                filename == pattern
            }
        });

        if should_download {
            println!("  Downloading {}...", filename);

            let downloaded_path = repo.download(filename).await.map_err(|e| {
                CliError::InvalidInput(format!("Failed to download {}: {}", filename, e))
            })?;

            let target_path = target_dir.join(filename);

            // Create parent directories if needed
            if let Some(parent) = target_path.parent() {
                fs::create_dir_all(parent).map_err(|e| {
                    CliError::InvalidInput(format!("Failed to create directory: {}", e))
                })?;
            }

            fs::copy(&downloaded_path, &target_path)
                .map_err(|e| CliError::InvalidInput(format!("Failed to copy file: {}", e)))?;

            downloaded_files += 1;
        }
    }

    if downloaded_files == 0 {
        return Err(CliError::InvalidInput(
            "No model files found in the repository".to_string(),
        ));
    }

    println!("✅ Downloaded {} files successfully!", downloaded_files);

    // Try to read config.json to get model information
    let mut architecture = None;
    let mut parameters = None;

    let config_path = target_dir.join("config.json");
    if config_path.exists() {
        if let Ok(config_content) = fs::read_to_string(&config_path) {
            if let Ok(config) = serde_json::from_str::<serde_json::Value>(&config_content) {
                // Extract architecture info
                if let Some(arch_value) = config.get("architectures") {
                    if let Some(arch_array) = arch_value.as_array() {
                        if let Some(first_arch) = arch_array.first() {
                            architecture = first_arch.as_str().map(|s| s.to_string());
                        }
                    }
                }

                // Try to estimate model size from parameters
                if let Some(hidden_size) = config.get("hidden_size").and_then(|v| v.as_u64()) {
                    if let Some(num_layers) =
                        config.get("num_hidden_layers").and_then(|v| v.as_u64())
                    {
                        let approx_params = (hidden_size * num_layers * 4) / 1_000_000; // Very rough estimate
                        if approx_params < 1000 {
                            parameters = Some(format!("~{}M", approx_params));
                        } else {
                            parameters = Some(format!("~{:.1}B", approx_params as f64 / 1000.0));
                        }
                    }
                }
            }
        }
    }

    // Create metadata file
    let metadata = BaseModelMetadata {
        name: _name.to_string(),
        repo_id: repo_id.to_string(),
        description: Some(format!("Downloaded from HuggingFace: {}", repo_id)),
        downloaded_at: Some(Utc::now().to_rfc3339()),
        parameters,
        architecture,
        quantization: None,
    };

    let meta_path = target_dir.join("meta.toml");
    let toml_string = toml::to_string_pretty(&metadata)
        .map_err(|e| CliError::InvalidInput(format!("Failed to serialize metadata: {}", e)))?;

    fs::write(&meta_path, toml_string)
        .map_err(|e| CliError::InvalidInput(format!("Failed to write metadata file: {}", e)))?;

    println!("📝 Created metadata file: meta.toml");

    Ok(())
}
