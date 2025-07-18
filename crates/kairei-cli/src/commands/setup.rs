//! Setup command handler

use crate::error::CliError;
use std::fs;
use std::path::Path;
use tokio::process::Command;

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
    ];

    fn find(name: &str) -> Option<&'static Model> {
        Self::MODELS.iter().find(|m| m.name == name)
    }
}

pub async fn run_setup(list: bool, model: Option<String>, force: bool) -> Result<(), CliError> {
    if list {
        list_models();
        return Ok(());
    }

    // Create models directory if it doesn't exist
    let models_dir = Path::new("models");
    if !models_dir.exists() {
        println!("📁 Creating models directory...");
        fs::create_dir_all(models_dir).map_err(|e| {
            CliError::InvalidInput(format!("Failed to create models directory: {}", e))
        })?;
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
    // Check if Python and required packages are available
    check_python_deps().await?;

    // Use Python script to download with progress
    let python_script = format!(
        r#"
import sys
import os
from huggingface_hub import hf_hub_download

try:
    print("Connecting to HuggingFace Hub...")
    file_path = hf_hub_download(
        repo_id='{}',
        filename='{}',
        cache_dir='./temp/hf_cache'
    )
    
    # Copy to target location
    import shutil
    os.makedirs(os.path.dirname('{}'), exist_ok=True)
    shutil.copy2(file_path, '{}')
    
except Exception as e:
    print(f"Error: {{e}}", file=sys.stderr)
    sys.exit(1)
"#,
        model.repo_id,
        model.filename,
        target_path.display(),
        target_path.display()
    );

    let output = Command::new("python3")
        .arg("-c")
        .arg(&python_script)
        .output()
        .await
        .map_err(|e| CliError::InvalidInput(format!("Failed to run Python: {}", e)))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(CliError::InvalidInput(format!(
            "Download failed: {}",
            stderr
        )));
    }

    Ok(())
}

async fn check_python_deps() -> Result<(), CliError> {
    // Check if Python is available
    let python_check = Command::new("python3").arg("--version").output().await;

    if python_check.is_err() {
        return Err(CliError::InvalidInput(
            "Python 3 is required for downloading models. Please install Python 3.".to_string(),
        ));
    }

    // Check if huggingface_hub is installed
    let hf_check = Command::new("python3")
        .arg("-c")
        .arg("import huggingface_hub")
        .output()
        .await
        .map_err(|e| CliError::InvalidInput(format!("Failed to check Python packages: {}", e)))?;

    if !hf_check.status.success() {
        println!("⚠️  huggingface_hub is not installed. Installing...");

        let install_output = Command::new("pip3")
            .arg("install")
            .arg("huggingface_hub")
            .output()
            .await
            .map_err(|e| {
                CliError::InvalidInput(format!("Failed to install huggingface_hub: {}", e))
            })?;

        if !install_output.status.success() {
            return Err(CliError::InvalidInput(
                "Failed to install huggingface_hub. Please run: pip3 install huggingface_hub"
                    .to_string(),
            ));
        }

        println!("✅ huggingface_hub installed successfully!");
    }

    Ok(())
}
