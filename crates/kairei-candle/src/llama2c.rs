//! Simple LLaMA2-C based processor

use async_trait::async_trait;
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::llama2_c::{self, Config};
use candle_transformers::models::llama2_c_weights::TransformerWeights;
use kairei_core::{Processor, ProcessorMetadata, Request, Response, Result as CoreResult};
use std::sync::RwLock;
use tokenizers::Tokenizer;

use crate::CandleError;

/// LLaMA2-C based processor
pub struct Llama2CProcessor {
    model: llama2_c::Llama,
    #[allow(dead_code)]
    config: llama2_c::Config,
    cache: RwLock<llama2_c::Cache>, // RwLock for thread-safe interior mutability
    device: Device,
    tokenizer: Tokenizer,
    logits_processor: RwLock<LogitsProcessor>,
    #[allow(dead_code)]
    seed: u64,
    #[allow(dead_code)]
    temperature: f64,
    #[allow(dead_code)]
    top_p: Option<f64>,
}

impl Llama2CProcessor {
    /// Create a new Llama2CProcessor with tiny config
    pub fn new_tiny() -> Result<Self, CandleError> {
        // Try to load from default model file first
        let default_model_path = "models/stories15M.bin";
        if std::path::Path::new(default_model_path).exists() {
            println!("Loading model from: {}", default_model_path);
            Self::new_from_file(default_model_path)
        } else {
            // Fall back to zero-initialized model
            println!("Model file not found, using zero-initialized weights");
            let tokenizer = Self::load_tokenizer(None)?;
            Self::new_with_tokenizer(tokenizer)
        }
    }

    /// Create a new Llama2CProcessor with model weights from file
    pub fn new_from_file(model_path: &str) -> Result<Self, CandleError> {
        let tokenizer = Self::load_tokenizer(None)?;
        let device = Device::Cpu;

        // Load model from file
        let mut file = std::fs::File::open(model_path)
            .map_err(|e| CandleError::Other(format!("Failed to open model file: {}", e)))?;

        // Read config from the model file
        let config = Config::from_reader(&mut file)
            .map_err(|e| CandleError::Other(format!("Failed to read config: {}", e)))?;

        println!(
            "Loaded model config: dim={}, n_layers={}, n_heads={}",
            config.dim, config.n_layers, config.n_heads
        );

        // Load weights
        let weights = TransformerWeights::from_reader(&mut file, &config, &device)
            .map_err(|e| CandleError::Other(format!("Failed to load weights: {}", e)))?;

        // Create VarBuilder from weights
        let vb = weights
            .var_builder(&config, &device)
            .map_err(|e| CandleError::Other(format!("Failed to create var builder: {}", e)))?;

        // Create cache
        let cache = llama2_c::Cache::new(true, &config, vb.pp("rot"))?;

        // Load model
        let model = llama2_c::Llama::load(vb, config.clone())?;

        // Initialize generation parameters
        let seed = 299792458;
        let temperature = 0.8;
        let top_p = Some(0.9);
        let logits_processor = LogitsProcessor::new(seed, Some(temperature), top_p);

        Ok(Self {
            model,
            config,
            cache: RwLock::new(cache),
            device,
            tokenizer,
            logits_processor: RwLock::new(logits_processor),
            seed,
            temperature,
            top_p,
        })
    }

    /// Create a new Llama2CProcessor with custom tokenizer path
    pub fn new_with_tokenizer_path(tokenizer_path: Option<&str>) -> Result<Self, CandleError> {
        let tokenizer = Self::load_tokenizer(tokenizer_path)?;
        Self::new_with_tokenizer(tokenizer)
    }

    /// Create with a specific tokenizer
    pub fn new_with_tokenizer(tokenizer: Tokenizer) -> Result<Self, CandleError> {
        // Use tiny config for testing
        let config = llama2_c::Config::tiny_15m();
        let device = Device::Cpu;

        // Create a simple VarBuilder with zeros for testing
        let vb = VarBuilder::zeros(DType::F32, &device);

        // Create cache
        let cache = llama2_c::Cache::new(true, &config, vb.pp("rot"))?;

        // Load model
        let model = llama2_c::Llama::load(vb, config.clone())?;

        // Initialize generation parameters
        let seed = 299792458; // Speed of light for fun
        let temperature = 0.8;
        let top_p = Some(0.9);
        let logits_processor = LogitsProcessor::new(seed, Some(temperature), top_p);

        Ok(Self {
            model,
            config,
            cache: RwLock::new(cache),
            device,
            tokenizer,
            logits_processor: RwLock::new(logits_processor),
            seed,
            temperature,
            top_p,
        })
    }

    /// Load tokenizer from local file or HuggingFace
    fn load_tokenizer(tokenizer_path: Option<&str>) -> Result<Tokenizer, CandleError> {
        use std::path::PathBuf;

        // 1. Try provided path
        if let Some(path) = tokenizer_path {
            println!("Loading tokenizer from provided path: {}", path);
            return Tokenizer::from_file(path).map_err(CandleError::Tokenizer);
        }

        // 2. Try default local path
        let default_path = PathBuf::from("models/tokenizer.json");
        if default_path.exists() {
            println!("Loading tokenizer from default path: {:?}", default_path);
            return Tokenizer::from_file(&default_path).map_err(CandleError::Tokenizer);
        }

        // 3. Download from HuggingFace as fallback
        use hf_hub::api::sync::Api;
        println!("Local tokenizer not found, downloading from HuggingFace...");

        let api = Api::new().map_err(|e| CandleError::Other(e.to_string()))?;
        let repo = api.model("hf-internal-testing/llama-tokenizer".to_string());
        let downloaded_path = repo
            .get("tokenizer.json")
            .map_err(|e| CandleError::Other(e.to_string()))?;

        println!("Loading tokenizer from: {:?}", downloaded_path);
        Tokenizer::from_file(downloaded_path).map_err(CandleError::Tokenizer)
    }
}

#[async_trait]
impl Processor for Llama2CProcessor {
    async fn process(&self, request: Request) -> CoreResult<Response> {
        // Tokenize the input
        let encoding = self
            .tokenizer
            .encode(request.message.as_str(), false)
            .map_err(|e| CandleError::Other(e.to_string()))?;

        let mut tokens = encoding.get_ids().to_vec();
        if tokens.is_empty() {
            return Ok(Response::simple(
                request.id,
                "No tokens generated".to_string(),
            ));
        }

        // Generate tokens
        let max_new_tokens = 50; // Generate up to 50 new tokens
        let mut generated_tokens = Vec::new();
        let mut index_pos = 0;

        for _ in 0..max_new_tokens {
            // Prepare input
            let context_size = if index_pos > 0 { 1 } else { tokens.len() };
            let ctxt_len;
            let logits = {
                let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
                ctxt_len = ctxt.len();
                let input = Tensor::new(ctxt, &self.device)
                    .map_err(CandleError::from)?
                    .unsqueeze(0)
                    .map_err(CandleError::from)?;

                // Forward pass
                let logits = self
                    .model
                    .forward(&input, index_pos, &mut self.cache.write().unwrap())
                    .map_err(CandleError::from)?;

                // Get last token logits
                logits
                    .i((0, logits.dim(1).map_err(CandleError::from)? - 1))
                    .map_err(CandleError::from)?
            };

            // Sample next token
            let next_token = self
                .logits_processor
                .write()
                .unwrap()
                .sample(&logits)
                .map_err(CandleError::from)?;

            tokens.push(next_token);
            generated_tokens.push(next_token);
            index_pos += ctxt_len;

            // Simple stop condition (if we have a stop token)
            if next_token == 0 {
                // Assuming 0 is EOS
                break;
            }
        }

        // Decode generated tokens
        let generated_text = self
            .tokenizer
            .decode(&generated_tokens, false)
            .map_err(|e| CandleError::Other(e.to_string()))?;

        let message = format!(
            "Input: '{}' (tokens: {:?}) -> Generated: '{}' (tokens: {:?})",
            request.message,
            encoding.get_ids(),
            generated_text,
            generated_tokens
        );

        Ok(Response::simple(request.id, message))
    }

    fn metadata(&self) -> ProcessorMetadata {
        ProcessorMetadata {
            name: Some("LLaMA2-C Processor".to_string()),
            description: Some("Tiny LLaMA2 model processor".to_string()),
            version: Some("0.1.0".to_string()),
            capabilities: vec!["text-generation".to_string(), "transformer".to_string()],
        }
    }
}
