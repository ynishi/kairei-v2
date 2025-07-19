//! Chat command handler

use crate::error::CliError;
use kairei::prelude::*;
use rustyline::DefaultEditor;
use std::path::{Path, PathBuf};

/// Resolve LoRA model path from name
fn resolve_lora_path(lora_name: &str) -> Result<PathBuf, CliError> {
    let lora_dir = Path::new("loras").join(lora_name);
    let lora_file = lora_dir.join("adapter.safetensors");

    if !lora_file.exists() {
        return Err(CliError::InvalidInput(format!(
            "LoRA model '{}' not found. Use 'kairei lora list' to see available models.",
            lora_name
        )));
    }

    Ok(lora_file)
}

pub async fn run_chat(
    initial_message: Option<String>,
    once: bool,
    use_candle: bool,
    lora_models: Vec<String>,
    base_model: Option<String>,
) -> Result<(), CliError> {
    // Only show header in interactive mode
    if !once {
        println!("🤖 Kairei Chat - Type 'exit' to quit");
        println!("================================");
    }

    // Resolve LoRA paths
    let lora_paths: Vec<PathBuf> = if !lora_models.is_empty() {
        println!("📚 Resolving LoRA models...");
        let mut paths = Vec::new();
        for lora_name in &lora_models {
            match resolve_lora_path(lora_name) {
                Ok(path) => {
                    println!("   ✅ Found: {} -> {}", lora_name, path.display());
                    paths.push(path);
                }
                Err(e) => {
                    return Err(e);
                }
            }
        }
        if let Some(ref model) = base_model {
            println!("🧠 Base model: {}", model);
        }
        paths
    } else {
        Vec::new()
    };

    // Build the application
    let app = if use_candle {
        println!("🔥 Initializing Candle backend (LLaMA2-C)...");

        // Create processor with LoRA if specified
        let processor = if !lora_paths.is_empty() {
            println!("🔧 Applying LoRA adapters...");
            // TODO: Create processor with LoRA support
            // For now, we'll create the standard processor and warn
            println!("⚠️  LoRA support is not yet implemented in Candle backend");
            kairei::Llama2CProcessor::new_tiny().map_err(|e| CliError::Core(e.to_string()))?
        } else {
            kairei::Llama2CProcessor::new_tiny().map_err(|e| CliError::Core(e.to_string()))?
        };

        println!("✅ LLaMA2-C backend ready!");

        KaireiApp::builder("kairei-chat")
            .llm_mode()
            .processor(processor)
            .build()
            .map_err(CliError::Core)?
    } else {
        // Use echo processor as fallback
        KaireiApp::builder("kairei-chat")
            .llm_mode()
            .processor(kairei::app::processors::EchoProcessor)
            .build()
            .map_err(CliError::Core)?
    };

    // If initial message provided, process it
    if let Some(msg) = initial_message {
        if !once {
            println!("You: {}", msg);
        }
        let request = Request::simple(msg);
        let response = app
            .process(request)
            .await
            .map_err(|e| CliError::Core(e.to_string()))?;

        // In once mode, just print the response without "Bot:" prefix
        if once {
            println!("{}", response.message);
            return Ok(());
        } else {
            println!("Bot: {}", response.message);
        }
    } else if once {
        // In once mode, we need a message
        return Err(CliError::InvalidInput(
            "Message is required in once mode".to_string(),
        ));
    }

    // Interactive loop (only if not in once mode)
    let mut rl = DefaultEditor::new()?;

    loop {
        let readline = rl.readline("🤖 > ");
        match readline {
            Ok(line) => {
                let input = line.trim();

                if input == "exit" || input == "quit" {
                    println!("Goodbye! 👋");
                    break;
                }

                if input.is_empty() {
                    continue;
                }

                // Add to history
                rl.add_history_entry(&line).ok();

                // Process through app
                let request = Request::simple(input);
                let response = app
                    .process(request)
                    .await
                    .map_err(|e| CliError::Core(e.to_string()))?;
                println!("Bot: {}", response.message);
            }
            Err(rustyline::error::ReadlineError::Interrupted) => {
                println!("\nCtrl-C received. Use 'exit' to quit.");
                continue;
            }
            Err(rustyline::error::ReadlineError::Eof) => {
                println!("\nGoodbye! 👋");
                break;
            }
            Err(err) => {
                return Err(err.into());
            }
        }
    }

    Ok(())
}
