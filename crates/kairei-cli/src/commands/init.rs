//! Init command handler

use crate::error::CliError;
use std::fs;
use std::path::Path;

pub async fn run_init() -> Result<(), CliError> {
    println!("🚀 Initializing Kairei project...");

    // Create project structure
    let dirs = vec!["models", "data", "logs", "examples"];

    for dir in dirs {
        let path = Path::new(dir);
        if !path.exists() {
            println!("📁 Creating {} directory...", dir);
            fs::create_dir_all(path)
                .map_err(|e| CliError::InvalidInput(format!("Failed to create {}: {}", dir, e)))?;
        } else {
            println!("✅ {} directory already exists", dir);
        }
    }

    // Create a sample .env file if it doesn't exist
    let env_path = Path::new(".env");
    if !env_path.exists() {
        println!("📝 Creating .env file...");
        let env_content = r#"# Kairei Configuration
# Add your environment variables here
# KAIREI_LOG_LEVEL=info
"#;
        fs::write(env_path, env_content)
            .map_err(|e| CliError::InvalidInput(format!("Failed to create .env: {}", e)))?;
    }

    println!("\n✨ Kairei project initialized successfully!");
    println!("\nNext steps:");
    println!("  1. Run 'kairei setup' to download models");
    println!("  2. Run 'kairei chat --candle' to start chatting");

    Ok(())
}
