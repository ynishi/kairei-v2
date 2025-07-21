//! Setup command handler

use crate::error::CliError;
use kairei::base_model::{BaseModelService, HuggingFaceDownloader, InMemoryBaseModelRepository};
use kairei::config::KaireiConfig;
use kairei::storage::LocalStorage;
use std::fs;
use std::path::Path;
use std::sync::Arc;

pub async fn run_setup(
    list: bool,
    force: bool,
    scan: bool,
    name: Option<String>,
    repo_id: Option<String>,
) -> Result<(), CliError> {
    // Create config and initialize services
    let config = KaireiConfig::default();
    let repository = Arc::new(InMemoryBaseModelRepository::new());
    let storage = Arc::new(LocalStorage::from_config(&config));
    let downloader = Arc::new(HuggingFaceDownloader::new(None)); // No API token for now
    let service =
        BaseModelService::with_config(repository.clone(), storage, downloader, config.clone());

    if scan {
        // Scan models directory and register found models
        println!("🔍 Scanning models directory for existing models...");
        let registered_models = service
            .scan_and_register_models()
            .await
            .map_err(|e| CliError::InvalidInput(e.to_string()))?;

        println!();
        if registered_models.is_empty() {
            println!("No new models found to register.");
        } else {
            println!(
                "✅ Successfully registered {} models:",
                registered_models.len()
            );
            for model in &registered_models {
                println!("  - {}", model.name);
            }
        }
        return Ok(());
    }

    if list {
        list_models(&service).await?;
        return Ok(());
    }

    // Create necessary directories from config
    let directories = vec![
        &config.models_dir,
        &config.loras_dir,
        &config.datasets_dir,
        &config.base_models_dir,
    ];
    for dir in &directories {
        let path = Path::new(&config.base_dir).join(dir);
        if !path.exists() {
            println!("📁 Creating {} directory...", dir);
            fs::create_dir_all(&path).map_err(|e| {
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

    // If no specific name/repo provided, show help
    if name.is_none() && repo_id.is_none() {
        println!("🎯 No model specified. Use one of the following:");
        println!("  kairei setup <name> <repo-id>    # Download custom model from HuggingFace");
        println!("  kairei setup --list              # List all registered models");
        println!(
            "  kairei setup --scan              # Scan models directory and register found models"
        );
        return Ok(());
    }

    Ok(())
}

async fn list_models(service: &BaseModelService) -> Result<(), CliError> {
    println!("📦 Registered models:");
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
        println!("  kairei setup <name> <huggingface-repo>");
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

#[deprecated(
    since = "0.1.0",
    note = "Legacy model scanning will be reimplemented when Repository persistence is added"
)]
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

    // Legacy model scanning is temporarily disabled
    // Will be reimplemented when Repository persistence is added

    println!();
    println!("To download more models:");
    println!("  kairei setup --list                              # See available models");
    println!("  kairei setup <name> <repo_id>                    # Download from HuggingFace");
    println!();
    println!("To train with a model:");
    println!("  kairei train -m <model_name> -d <data.json>      # Use model name from above");

    Ok(())
}
