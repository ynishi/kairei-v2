//! Setup command handler

use crate::error::CliError;
use chrono::Utc;
use hf_hub::api::tokio::Api;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

/// Model metadata stored as meta.toml in each model directory
#[derive(Debug, Serialize, Deserialize)]
struct ModelMetadata {
    name: String,
    repo_id: String,
    model_type: Option<String>,
    description: Option<String>,
    downloaded_at: String,
    // Model specific params
    parameters: Option<String>,   // e.g., "1.1B", "7B", etc.
    architecture: Option<String>, // e.g., "llama", "mistral", etc.
    quantization: Option<String>, // e.g., "f16", "int8", etc.
}

/// Model information
struct Model {
    name: &'static str,
    filename: &'static str,
    description: &'static str,
    repo_id: &'static str,
    size_mb: u64,
}

impl Model {
    const MODELS: &'static [Model] = &[
        Model {
            name: "stories15M",
            filename: "stories15M.bin",
            description: "15M parameter tiny story model",
            repo_id: "karpathy/tinyllamas",
            size_mb: 58,
        },
        Model {
            name: "stories42M",
            filename: "stories42M.bin",
            description: "42M parameter story model",
            repo_id: "karpathy/tinyllamas",
            size_mb: 161,
        },
        Model {
            name: "stories110M",
            filename: "stories110M.bin",
            description: "110M parameter story model",
            repo_id: "karpathy/tinyllamas",
            size_mb: 420,
        },
        Model {
            name: "tokenizer",
            filename: "tokenizer.json",
            description: "LLaMA tokenizer",
            repo_id: "hf-internal-testing/llama-tokenizer",
            size_mb: 2,
        },
        Model {
            name: "tinyllama",
            filename: "tinyllama/model.safetensors",
            description: "TinyLlama 1.1B Chat model",
            repo_id: "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            size_mb: 2200,
        },
    ];

    fn find(name: &str) -> Option<&'static Model> {
        Self::MODELS.iter().find(|m| m.name == name)
    }
}

pub async fn run_setup(
    list: bool,
    model: Option<String>,
    force: bool,
    name: Option<String>,
    repo_id: Option<String>,
) -> Result<(), CliError> {
    if list {
        list_models();
        return Ok(());
    }

    // Create necessary directories
    let models_dir = Path::new("models");
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
    if let (Some(custom_name), Some(custom_repo)) = (name, repo_id) {
        // Download custom HuggingFace model
        println!(
            "\n🤖 Downloading custom model: {} from {}",
            custom_name, custom_repo
        );

        let model_dir = Path::new("models").join(&custom_name);
        if model_dir.exists() && !force {
            println!(
                "✅ {} already exists (use --force to re-download)",
                custom_name
            );
            return Ok(());
        }

        download_custom_model(&custom_name, &custom_repo, &model_dir).await?;

        println!("✅ Downloaded {} successfully!", custom_name);
        println!("\n🎉 Setup complete! You can now use your custom model.");
        return Ok(());
    }

    // Determine which models to download
    let models_to_download = if let Some(model_name) = model {
        // Download specific model
        if let Some(model) = Model::find(&model_name) {
            vec![model]
        } else {
            return Err(CliError::InvalidInput(format!(
                "Unknown model: {}",
                model_name
            )));
        }
    } else {
        // Default: download tokenizer and smallest model
        vec![
            Model::find("tokenizer").unwrap(),
            Model::find("stories15M").unwrap(),
        ]
    };

    // Download models
    for model in models_to_download {
        let target_path = models_dir.join(model.filename);

        if target_path.exists() && !force {
            println!(
                "✅ {} already exists (use --force to re-download)",
                model.filename
            );
            continue;
        }

        println!(
            "\n🌐 Downloading {} ({}, ~{}MB)...",
            model.name, model.description, model.size_mb
        );

        download_model(model, &target_path).await?;

        println!("✅ Downloaded {} successfully!", model.filename);
    }

    println!("\n🎉 Setup complete! You can now run:");
    println!("   kairei chat --candle -m \"Hello, world!\"");

    Ok(())
}

fn list_models() {
    println!("📦 Available models:");
    println!("==================");
    println!();

    for model in Model::MODELS {
        println!("  {} - {}", model.name, model.description);
        println!("       File: {} (~{}MB)", model.filename, model.size_mb);
        println!();
    }

    println!("Usage examples:");
    println!("  kairei setup                    # Download default models");
    println!("  kairei setup --model stories42M # Download specific model");
    println!("  kairei setup --list            # List available models");
}

async fn download_model(model: &Model, target_path: &Path) -> Result<(), CliError> {
    // Create parent directory if needed
    if let Some(parent) = target_path.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| CliError::InvalidInput(format!("Failed to create directory: {}", e)))?;
    }

    // Use hf-hub to download the file
    let api = Api::new().map_err(|e| {
        CliError::InvalidInput(format!("Failed to create HuggingFace API client: {}", e))
    })?;

    let repo = api.model(model.repo_id.to_string());

    println!("Connecting to HuggingFace Hub...");

    let downloaded_path = repo
        .download(model.filename)
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
    let model_patterns = ["*.safetensors",
        "*.bin",
        "*.gguf",
        "tokenizer.json",
        "tokenizer_config.json",
        "tokenizer.model",
        "config.json"];

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
    let mut model_type = None;
    let mut architecture = None;
    let mut parameters = None;

    let config_path = target_dir.join("config.json");
    if config_path.exists() {
        if let Ok(config_content) = fs::read_to_string(&config_path) {
            if let Ok(config) = serde_json::from_str::<serde_json::Value>(&config_content) {
                // Extract model type
                if let Some(model_type_value) = config.get("model_type") {
                    model_type = model_type_value.as_str().map(|s| s.to_string());
                }

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
    let metadata = ModelMetadata {
        name: _name.to_string(),
        repo_id: repo_id.to_string(),
        model_type,
        description: Some(format!("Downloaded from HuggingFace: {}", repo_id)),
        downloaded_at: Utc::now().to_rfc3339(),
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
