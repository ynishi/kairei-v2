//! Training functionality for Kairei-candle

use crate::CandleError;
use candle_core::{DType, Device, Tensor};
use candle_lora::LoraConfig;
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap, loss};
use candle_transformers::models::llama2_c::{self, Cache, Config, Llama};
use candle_transformers::models::llama2_c_weights::TransformerWeights;
use std::path::Path;

/// Training configuration for LoRA
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    pub culture_name: String,
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub lora_rank: usize,
    pub lora_alpha: f64,
    pub lora_dropout: Option<f32>,
}

/// Load training data (placeholder for now)
fn load_training_data(culture_name: &str) -> Result<Vec<String>, CandleError> {
    println!("📚 Loading training data for culture: {}", culture_name);

    // TODO: Load actual training data from files or database
    // For now, return some dummy data
    Ok(vec![
        "Hello from Kairei culture!".to_string(),
        "This is a test training sample.".to_string(),
        "We are learning with LoRA!".to_string(),
    ])
}

/// Create LoRA model configuration
fn create_lora_config(config: &TrainingConfig) -> LoraConfig {
    LoraConfig::new(config.lora_rank, config.lora_alpha, config.lora_dropout)
}

/// Initialize tokenizer
fn init_tokenizer() -> Result<tokenizers::Tokenizer, CandleError> {
    println!("🔤 Initializing tokenizer...");

    // Try to load tokenizer from default path
    let tokenizer_path = "models/tokenizer.json";

    if Path::new(tokenizer_path).exists() {
        tokenizers::Tokenizer::from_file(tokenizer_path)
            .map_err(|e| CandleError::Other(format!("Failed to load tokenizer: {}", e)))
    } else {
        // Create a simple tokenizer for testing
        println!("⚠️  Tokenizer file not found, using default test tokenizer");

        // For now, we'll create a simple tokenizer
        // In production, this should load a proper tokenizer
        let _vocab_size = 32000;

        // This is a placeholder - in real implementation,
        // we should load a proper tokenizer model
        Err(CandleError::Other(
            "Tokenizer initialization not fully implemented yet".to_string(),
        ))
    }
}

/// Initialize model with LoRA
fn init_model_with_lora(
    _lora_config: &LoraConfig,
    device: &Device,
) -> Result<(Llama, VarMap, Cache), CandleError> {
    println!("🤖 Initializing model with LoRA...");

    // Create tiny config for testing
    let config = Config {
        dim: 288,          // Hidden size
        hidden_dim: 768,   // FFN hidden size
        n_layers: 6,       // Number of layers
        n_heads: 6,        // Number of attention heads
        n_kv_heads: 6,     // Number of key-value heads
        vocab_size: 32000, // Vocabulary size
        seq_len: 256,      // Maximum sequence length
        norm_eps: 1e-5,    // RMS normalization epsilon
    };

    // Create VarMap for LoRA parameters
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);

    // Try to load base model weights if available
    let model_path = "models/stories15M.bin";
    if Path::new(model_path).exists() {
        println!("📦 Loading base model from: {}", model_path);

        // Load weights from file
        let mut file = std::fs::File::open(model_path)
            .map_err(|e| CandleError::Other(format!("Failed to open model file: {}", e)))?;

        // Read config from file
        let file_config = Config::from_reader(&mut file)
            .map_err(|e| CandleError::Other(format!("Failed to read config: {}", e)))?;

        // Load transformer weights
        let weights = TransformerWeights::from_reader(&mut file, &file_config, device)
            .map_err(|e| CandleError::Other(format!("Failed to load weights: {}", e)))?;

        // Create cache (disable KV cache for training)
        let cache = llama2_c::Cache::new(false, &file_config, vb.pp("rot"))?;

        // Create VarBuilder from weights
        let weight_vb = weights.var_builder(&file_config, device)?;

        // Load model with weights
        let model = Llama::load(weight_vb, file_config)?;

        println!("✅ Base model loaded successfully!");
        Ok((model, varmap, cache))
    } else {
        println!("⚠️  Base model not found, creating random initialized model");

        // Create cache (disable KV cache for training)
        let cache = llama2_c::Cache::new(false, &config, vb.pp("rot"))?;

        // Create model with random weights
        let model = Llama::load(vb, config)?;

        println!("✅ Model initialized with random weights");
        Ok((model, varmap, cache))
    }
}

/// Train a LoRA model
pub fn train_lora(config: TrainingConfig) -> Result<(), CandleError> {
    println!("🚀 HELLO from kairei-candle train_lora!");
    println!("📋 Training config:");
    println!("   Culture: {}", config.culture_name);
    println!("   Epochs: {}", config.epochs);
    println!("   Batch size: {}", config.batch_size);
    println!("   Learning rate: {}", config.learning_rate);
    println!("   LoRA rank: {}", config.lora_rank);
    println!("   LoRA alpha: {}", config.lora_alpha);

    // Load training data
    let training_data = load_training_data(&config.culture_name)?;
    println!("📊 Loaded {} training samples", training_data.len());

    // Create LoRA configuration
    let lora_config = create_lora_config(&config);
    println!(
        "⚙️  Created LoRA config with rank={}, alpha={}",
        config.lora_rank, config.lora_alpha
    );

    // Initialize device
    let device = Device::Cpu;
    println!("🖥️  Using device: CPU");

    // Initialize tokenizer
    match init_tokenizer() {
        Ok(tokenizer) => {
            println!("✅ Tokenizer initialized successfully");
        }
        Err(e) => {
            println!("⚠️  Tokenizer initialization failed: {}", e);
            println!("   Continuing with training setup...");
        }
    }

    // Initialize model with LoRA
    let (model, varmap, mut cache) = init_model_with_lora(&lora_config, &device)?;

    // Set up optimizer
    let params = ParamsAdamW {
        lr: config.learning_rate,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        weight_decay: 0.01,
    };
    let mut optimizer = AdamW::new(varmap.all_vars(), params)?;
    println!(
        "📈 Optimizer initialized with learning rate: {}",
        config.learning_rate
    );

    // Get tokenizer for training
    let tokenizer = match init_tokenizer() {
        Ok(t) => t,
        Err(_) => {
            println!("⚠️  Skipping actual training due to tokenizer initialization failure");
            return Ok(());
        }
    };

    // Training loop
    println!("🏃 Starting training loop...");
    for epoch in 0..config.epochs {
        println!("\n📊 Epoch {}/{}", epoch + 1, config.epochs);

        let mut total_loss = 0.0;
        let mut batch_count = 0;

        // Process each training sample
        for (idx, text) in training_data.iter().enumerate() {
            // Tokenize the text
            let tokens = tokenizer
                .encode(text.as_str(), true)
                .map_err(|e| CandleError::Other(format!("Tokenization failed: {}", e)))?
                .get_ids()
                .to_vec();

            if tokens.len() < 2 {
                continue; // Skip very short sequences
            }

            // Create input and target tensors
            let input_tokens = &tokens[..tokens.len() - 1];
            let target_tokens = &tokens[1..];

            let input = Tensor::new(input_tokens, &device)?.unsqueeze(0)?;
            let target = Tensor::new(target_tokens, &device)?.unsqueeze(0)?;

            // Debug info
            println!("\n   Debug - Input shape: {:?}", input.shape());
            println!("   Debug - Target shape: {:?}", target.shape());

            // Forward pass
            let logits = model.forward(&input, 0, &mut cache)?;

            // Calculate loss
            let loss = loss::cross_entropy(&logits.flatten_to(1)?, &target.flatten_to(1)?)?;

            // Backward pass and optimizer step
            optimizer.backward_step(&loss)?;

            // Track loss
            let loss_value = loss.to_vec0::<f32>()? as f64;
            total_loss += loss_value;
            batch_count += 1;

            // Progress update
            let progress = (idx + 1) as f32 / training_data.len() as f32;
            let bar_width = 20;
            let filled = (progress * bar_width as f32) as usize;
            let empty = bar_width - filled;
            print!(
                "\r   Progress: [{}{}] {:.1}% | Loss: {:.4}",
                "█".repeat(filled),
                " ".repeat(empty),
                progress * 100.0,
                loss_value
            );
            std::io::Write::flush(&mut std::io::stdout())
                .map_err(|e| CandleError::Other(format!("Flush failed: {}", e)))?;
        }

        // Print epoch summary
        let avg_loss = if batch_count > 0 {
            total_loss / batch_count as f64
        } else {
            0.0
        };
        println!("\n   Average loss: {:.4}", avg_loss);

        // Save checkpoint every epoch
        if epoch % 1 == 0 {
            let checkpoint_name = format!("checkpoint_epoch_{}.safetensors", epoch + 1);
            varmap.save(&checkpoint_name)?;
            println!("   💾 Saved checkpoint: {}", checkpoint_name);
        }
    }

    // Save final model
    let final_model_path = format!("lora_{}_final.safetensors", config.culture_name);
    varmap.save(&final_model_path)?;
    println!(
        "\n🎉 Training completed! Final model saved to: {}",
        final_model_path
    );

    Ok(())
}
