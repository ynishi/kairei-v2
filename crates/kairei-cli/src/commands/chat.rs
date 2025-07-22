//! Chat command handler

use crate::ModelType;
use crate::error::CliError;
use kairei::base_model::BaseModelService;
use kairei::lora::LoraService;
use kairei::prelude::*;
use rustyline::DefaultEditor;
use std::path::PathBuf;

pub async fn run_chat(
    lora_service: &LoraService,
    base_model_service: &BaseModelService,
    initial_message: Option<String>,
    once: bool,
    use_candle: bool,
    lora_models: Vec<String>,
    base_model: Option<String>,
    model_type: ModelType,
    tokenizer_path: Option<String>,
    max_tokens: usize,
) -> Result<(), CliError> {
    // Only show header in interactive mode
    if !once {
        println!("🤖 Kairei Chat - Type 'exit' to quit");
        println!("================================");
    }

    // Resolve base model information if specified
    let base_model_info = if let Some(ref model_name) = base_model {
        match base_model_service.get_model_by_name(model_name).await {
            Ok(model) => {
                println!("🧠 Base model: {} ({})", model.name, model.id);
                Some(model)
            }
            Err(e) => {
                // If not found in service, treat it as a direct path for backward compatibility
                println!(
                    "⚠️  Base model '{}' not found in registry, using as direct path",
                    model_name
                );
                println!("   Error: {}", e);
                None
            }
        }
    } else {
        None
    };

    // Resolve LoRA paths using LoraService
    let lora_paths: Vec<PathBuf> = if !lora_models.is_empty() {
        println!("📚 Resolving LoRA models...");
        let mut paths = Vec::new();
        for lora_name in &lora_models {
            // Use LoraService to get LoRA by name
            match lora_service.get_by_name(lora_name).await {
                Ok(Some(lora)) => {
                    if let Some(file_path) = &lora.file_path {
                        let path = PathBuf::from(file_path);
                        println!("   ✅ Found: {} -> {}", lora_name, path.display());
                        paths.push(path);
                    } else {
                        return Err(CliError::InvalidInput(format!(
                            "LoRA '{}' exists but has no file path",
                            lora_name
                        )));
                    }
                }
                Ok(None) => {
                    return Err(CliError::InvalidInput(format!(
                        "LoRA model '{}' not found. Use 'kairei lora list' to see available models.",
                        lora_name
                    )));
                }
                Err(e) => {
                    return Err(CliError::InvalidInput(format!(
                        "Failed to get LoRA '{}': {}",
                        lora_name, e
                    )));
                }
            }
        }
        paths
    } else {
        Vec::new()
    };

    // Build the application
    let app = if use_candle {
        match model_type {
            ModelType::Llama2 => {
                println!("🚀 Initializing Candle backend (Llama2 with LoRA)...");

                // Resolve model path using BaseModelService or fallback
                let (model_path, model_dir) = if let Some(ref model_info) = base_model_info {
                    // Use model from BaseModelService
                    let model_dir = format!("models/{}", model_info.name);
                    let model_path = if let Some(ref filename) = model_info.filename {
                        format!("{}/{}", model_dir, filename)
                    } else {
                        // Try to find a model file in the directory
                        format!("{}/model.safetensors", model_dir)
                    };
                    (model_path, model_dir)
                } else if let Some(ref model_name) = base_model {
                    // Use as direct path for backward compatibility
                    let model_path = model_name.clone();
                    let model_dir = std::path::Path::new(&model_path)
                        .parent()
                        .map(|p| p.to_string_lossy().to_string())
                        .unwrap_or_else(|| "models".to_string());
                    (model_path, model_dir)
                } else {
                    // Default model path
                    (
                        "models/llama2-7b.safetensors".to_string(),
                        "models".to_string(),
                    )
                };

                // Tokenizer path with auto-detection
                let tokenizer_path = if let Some(path) = tokenizer_path.as_ref() {
                    path.clone()
                } else {
                    // Check for common tokenizer names in model directory
                    let possible_paths = vec![
                        format!("{}/tokenizer.json", model_dir),
                        format!("{}/tokenizer.model", model_dir),
                        format!("{}/tokenizer_config.json", model_dir),
                        "models/tokenizer.json".to_string(),
                    ];

                    let mut found_path = None;
                    for path in &possible_paths {
                        if std::path::Path::new(path).exists() {
                            println!("🔍 Auto-detected tokenizer: {}", path);
                            found_path = Some(path.clone());
                            break;
                        }
                    }

                    found_path.unwrap_or_else(|| {
                        println!("⚠️  No tokenizer auto-detected, using default");
                        "models/tokenizer.json".to_string()
                    })
                };

                // Check if files exist
                if !std::path::Path::new(&model_path).exists() {
                    return Err(CliError::InvalidInput(format!(
                        "Model file not found: {}. Please download a Llama2 model first.",
                        model_path
                    )));
                }

                if !std::path::Path::new(&tokenizer_path).exists() {
                    return Err(CliError::InvalidInput(format!(
                        "Tokenizer file not found: {}. Please download the tokenizer first or specify with --tokenizer.",
                        tokenizer_path
                    )));
                }

                // Get LoRA path if specified
                let lora_path = if !lora_paths.is_empty() {
                    Some(lora_paths[0].to_str().unwrap())
                } else {
                    None
                };

                println!("📄 Model: {}", model_path);
                println!("📖 Tokenizer: {}", tokenizer_path);
                if let Some(lora) = lora_path {
                    println!("🎯 LoRA: {}", lora);
                }

                // Initialize Llama2LoraProcessor
                use candle_core::{DType, Device};
                // use candle_lora_transformers::llama::Config;
                use kairei::Llama2Config as Config;

                // Load config from model directory if available
                let config = if model_path.contains("tinyllama") {
                    // TinyLlama config
                    Config {
                        dim: 2048,
                        hidden_dim: 5632,
                        vocab_size: 32000,
                        n_layers: 22,
                        n_heads: 32,
                        n_kv_heads: 4,
                        seq_len: 2048,
                        norm_eps: 1e-5,
                    }
                } else {
                    // Default to 7B config
                    Config {
                        dim: 4096,
                        hidden_dim: 11008,
                        vocab_size: 32000,
                        n_layers: 32,
                        n_heads: 32,
                        n_kv_heads: 32,
                        seq_len: 2048,
                        norm_eps: 1e-5,
                    }
                };
                let device = Device::Cpu;
                let dtype = DType::F16;

                let processor = kairei::Llama2LoraProcessor::init(
                    &model_path,
                    &tokenizer_path,
                    lora_path,
                    device,
                    dtype,
                    config,
                    max_tokens,
                )?;

                println!("✅ Llama2 with LoRA backend ready!");

                KaireiApp::builder("kairei-chat")
                    .llm_mode()
                    .processor(processor)
                    .build()?
            }
            ModelType::Llama3 => {
                println!("🚀 Initializing Candle backend (Llama3 with LoRA)...");

                // Resolve model path using BaseModelService or fallback
                let (model_path, model_dir) = if let Some(ref model_info) = base_model_info {
                    // Use model from BaseModelService
                    let model_dir = format!("models/{}", model_info.name);
                    let model_path = if let Some(ref filename) = model_info.filename {
                        format!("{}/{}", model_dir, filename)
                    } else {
                        // Try to find a model file in the directory
                        format!("{}/model.safetensors", model_dir)
                    };
                    (model_path, model_dir)
                } else if let Some(ref model_name) = base_model {
                    // Use as direct path for backward compatibility
                    let model_path = model_name.clone();
                    let model_dir = std::path::Path::new(&model_path)
                        .parent()
                        .map(|p| p.to_string_lossy().to_string())
                        .unwrap_or_else(|| "models".to_string());
                    (model_path, model_dir)
                } else {
                    // Default model path
                    (
                        "models/llama3-8b.safetensors".to_string(),
                        "models".to_string(),
                    )
                };

                // Tokenizer path with auto-detection
                let tokenizer_path = if let Some(path) = tokenizer_path.as_ref() {
                    path.clone()
                } else {
                    // Check for common tokenizer names in model directory
                    let possible_paths = vec![
                        format!("{}/tokenizer.json", model_dir),
                        format!("{}/tokenizer.model", model_dir),
                        format!("{}/tokenizer_config.json", model_dir),
                        "models/tokenizer.json".to_string(),
                    ];

                    let mut found_path = None;
                    for path in &possible_paths {
                        if std::path::Path::new(path).exists() {
                            println!("🔍 Auto-detected tokenizer: {}", path);
                            found_path = Some(path.clone());
                            break;
                        }
                    }

                    found_path.unwrap_or_else(|| {
                        println!("⚠️  No tokenizer auto-detected, using default");
                        "models/tokenizer.json".to_string()
                    })
                };

                // Check if files exist
                if !std::path::Path::new(&model_path).exists() {
                    return Err(CliError::InvalidInput(format!(
                        "Model file not found: {}. Please download a Llama3 model first.",
                        model_path
                    )));
                }

                if !std::path::Path::new(&tokenizer_path).exists() {
                    return Err(CliError::InvalidInput(format!(
                        "Tokenizer file not found: {}. Please download the tokenizer first or specify with --tokenizer.",
                        tokenizer_path
                    )));
                }

                // Get LoRA path if specified
                let lora_path = if !lora_paths.is_empty() {
                    Some(lora_paths[0].to_str().unwrap())
                } else {
                    None
                };

                println!("📄 Model: {}", model_path);
                println!("📖 Tokenizer: {}", tokenizer_path);
                if let Some(lora) = lora_path {
                    println!("🎯 LoRA: {}", lora);
                }

                // Initialize Llama3LoraProcessor
                use candle_core::{DType, Device};
                use kairei::Llama3Config as Config;

                // Load config from model directory if available
                let config =
                    if model_path.contains("llama3_1b") || model_path.contains("Llama-3.2-1B") {
                        // Llama3.2 1B config
                        Config {
                            hidden_size: 2048,
                            intermediate_size: 8192,
                            vocab_size: 128256,
                            num_hidden_layers: 16,
                            num_attention_heads: 32,
                            num_key_value_heads: 8,
                            max_position_embeddings: 131072,
                            rope_theta: 500000.0,
                            rms_norm_eps: 1e-5,
                            rope_scaling: None,
                            tie_word_embeddings: true,
                            use_flash_attn: false,
                        }
                    } else {
                        // Default to 8B config
                        Config {
                            hidden_size: 4096,
                            intermediate_size: 14336,
                            vocab_size: 128256,
                            num_hidden_layers: 32,
                            num_attention_heads: 32,
                            num_key_value_heads: 8,
                            max_position_embeddings: 131072,
                            rope_theta: 500000.0,
                            rms_norm_eps: 1e-5,
                            rope_scaling: None,
                            tie_word_embeddings: true,
                            use_flash_attn: false,
                        }
                    };
                let device = Device::Cpu;
                let dtype = DType::F16;

                let processor = kairei::Llama3LoraProcessor::new(
                    &model_path,
                    &tokenizer_path,
                    lora_path,
                    device,
                    dtype,
                    config,
                    max_tokens,
                )?;

                println!("✅ Llama3 with LoRA backend ready!");

                KaireiApp::builder("kairei-chat")
                    .llm_mode()
                    .processor(processor)
                    .build()?
            }
            ModelType::Llama2c => {
                println!("🔥 Initializing Candle backend (LLaMA2-C)...");

                // Create processor builder
                let mut builder = kairei::Llama2CProcessorBuilder::new();

                // Resolve model path using BaseModelService or fallback
                let model_path = if let Some(ref model_info) = base_model_info {
                    // Use model from BaseModelService
                    let model_dir = format!("models/{}", model_info.name);
                    if let Some(ref filename) = model_info.filename {
                        format!("{}/{}", model_dir, filename)
                    } else {
                        // Try to find a model file in the directory
                        format!("{}/model.bin", model_dir)
                    }
                } else if let Some(ref model_name) = base_model {
                    // Use as direct path for backward compatibility
                    model_name.clone()
                } else {
                    // Default model path
                    "models/stories15M.bin".to_string()
                };

                // Add model file if it exists
                if std::path::Path::new(&model_path).exists() {
                    println!("📄 Found model file: {}", model_path);
                    builder = builder.with_model_file(&model_path);
                } else {
                    println!(
                        "⚠️  Model file not found: {}, using default config",
                        model_path
                    );
                }

                // Add LoRA adapters if any
                if !lora_paths.is_empty() {
                    println!("🔧 Adding {} LoRA adapter(s)...", lora_paths.len());
                    for (idx, path) in lora_paths.iter().enumerate() {
                        println!("   [{}] {}", idx + 1, path.display());
                    }
                    builder = builder.with_loras(lora_paths);
                }

                // Build the processor
                println!("🏗️  Building processor...");
                let processor = builder.build()?;
                println!("✅ LLaMA2-C backend ready!");

                KaireiApp::builder("kairei-chat")
                    .llm_mode()
                    .processor(processor)
                    .build()?
            }
        }
    } else {
        // Use echo processor as fallback
        KaireiApp::builder("kairei-chat")
            .llm_mode()
            .processor(kairei::app::processors::EchoProcessor)
            .build()?
    };

    // If initial message provided, process it
    if let Some(msg) = initial_message {
        if !once {
            println!("You: {}", msg);
        }
        let request = Request::simple(msg);
        let response = app.process(request).await?;

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
                let response = app.process(request).await?;
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
