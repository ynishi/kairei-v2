//! Simple LLaMA2-C based processor

use crate::models::llama2c::{Cache, Llama, TransformerWeights};
use crate::models::lora::{LoraConfig, LoraManager};
use async_trait::async_trait;
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::llama2_c::Config;
use kairei_core::{Processor, ProcessorMetadata, Request, Response, Result as CoreResult};
use std::path::PathBuf;
use std::sync::RwLock;
use tokenizers::Tokenizer;

use crate::CandleError;

/// Model source variants
#[derive(Clone)]
pub enum ModelSource {
    /// Load from file with path
    File(PathBuf),
    /// Use pre-defined config with zero weights
    Config(Config),
    /// Download from HuggingFace
    HuggingFace { repo: String, file: String },
}

/// Tokenizer source variants
#[allow(clippy::large_enum_variant)]
#[derive(Clone)]
pub enum TokenizerSource {
    /// Load from file
    File(PathBuf),
    /// Use pre-loaded tokenizer
    Instance(Tokenizer),
    /// Download from HuggingFace
    HuggingFace { repo: String, file: String },
    /// Use default (will try local then HF)
    Default,
}

/// Generation configuration
#[derive(Clone)]
pub struct GenerationConfig {
    pub seed: u64,
    pub temperature: f64,
    pub top_p: Option<f64>,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            seed: 299792458,
            temperature: 0.8,
            top_p: Some(0.9),
        }
    }
}

/// Builder for creating Llama2CProcessor with flexible configuration
#[derive(Clone)]
pub struct Llama2CProcessorBuilder {
    model_source: ModelSource,
    tokenizer_source: TokenizerSource,
    device: Device,
    lora_paths: Vec<PathBuf>,
    generation_config: GenerationConfig,
}

impl Default for Llama2CProcessorBuilder {
    fn default() -> Self {
        Self {
            model_source: ModelSource::Config(Config::tiny_15m()),
            tokenizer_source: TokenizerSource::Default,
            device: Device::Cpu,
            lora_paths: Vec::new(),
            generation_config: GenerationConfig::default(),
        }
    }
}

impl Llama2CProcessorBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self::default()
    }

    /// Set model from file
    pub fn with_model_file(mut self, path: impl Into<PathBuf>) -> Self {
        self.model_source = ModelSource::File(path.into());
        self
    }

    /// Set model from config (zero weights)
    pub fn with_model_config(mut self, config: Config) -> Self {
        self.model_source = ModelSource::Config(config);
        self
    }

    /// Set model from HuggingFace
    pub fn with_model_from_hf(mut self, repo: impl Into<String>, file: impl Into<String>) -> Self {
        self.model_source = ModelSource::HuggingFace {
            repo: repo.into(),
            file: file.into(),
        };
        self
    }

    /// Set tokenizer from file
    pub fn with_tokenizer_file(mut self, path: impl Into<PathBuf>) -> Self {
        self.tokenizer_source = TokenizerSource::File(path.into());
        self
    }

    /// Set tokenizer instance
    pub fn with_tokenizer(mut self, tokenizer: Tokenizer) -> Self {
        self.tokenizer_source = TokenizerSource::Instance(tokenizer);
        self
    }

    /// Set tokenizer from HuggingFace
    pub fn with_tokenizer_from_hf(
        mut self,
        repo: impl Into<String>,
        file: impl Into<String>,
    ) -> Self {
        self.tokenizer_source = TokenizerSource::HuggingFace {
            repo: repo.into(),
            file: file.into(),
        };
        self
    }

    /// Add LoRA adapter
    pub fn with_lora(mut self, path: impl Into<PathBuf>) -> Self {
        self.lora_paths.push(path.into());
        self
    }

    /// Add multiple LoRA adapters
    pub fn with_loras(mut self, paths: impl IntoIterator<Item = impl Into<PathBuf>>) -> Self {
        self.lora_paths.extend(paths.into_iter().map(|p| p.into()));
        self
    }

    /// Set device
    pub fn with_device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }

    /// Set generation config
    pub fn with_generation_config(mut self, config: GenerationConfig) -> Self {
        self.generation_config = config;
        self
    }

    /// Set seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.generation_config.seed = seed;
        self
    }

    /// Set temperature
    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.generation_config.temperature = temperature;
        self
    }

    /// Set top_p
    pub fn with_top_p(mut self, top_p: Option<f64>) -> Self {
        self.generation_config.top_p = top_p;
        self
    }

    /// Build the processor
    pub fn build(self) -> Result<Llama2CProcessor, CandleError> {
        // 1. Load tokenizer
        let tokenizer = self.load_tokenizer()?;

        // 2. Load model and config
        let (model, config, cache) = self.load_model()?;

        // 3. Create logits processor
        let logits_processor = LogitsProcessor::new(
            self.generation_config.seed,
            Some(self.generation_config.temperature),
            self.generation_config.top_p,
        );

        // 4. Create base processor
        let mut processor = Llama2CProcessor {
            model,
            config,
            cache: RwLock::new(cache),
            device: self.device.clone(),
            tokenizer,
            logits_processor: RwLock::new(logits_processor),
            seed: self.generation_config.seed,
            temperature: self.generation_config.temperature,
            top_p: self.generation_config.top_p,
        };

        // 5. Apply LoRA if specified
        if !self.lora_paths.is_empty() {
            self.apply_lora(&mut processor)?;
        }

        Ok(processor)
    }

    /// Load tokenizer based on source
    fn load_tokenizer(&self) -> Result<Tokenizer, CandleError> {
        match &self.tokenizer_source {
            TokenizerSource::File(path) => {
                println!("Loading tokenizer from: {:?}", path);
                Tokenizer::from_file(path).map_err(CandleError::Tokenizer)
            }
            TokenizerSource::Instance(tokenizer) => Ok(tokenizer.clone()),
            TokenizerSource::HuggingFace { repo, file } => {
                use hf_hub::api::sync::Api;
                println!("Downloading tokenizer from HF: {}/{}", repo, file);
                let api = Api::new().map_err(|e| CandleError::Other(e.to_string()))?;
                let hf_repo = api.model(repo.clone());
                let path = hf_repo
                    .get(file)
                    .map_err(|e| CandleError::Other(e.to_string()))?;
                Tokenizer::from_file(path).map_err(CandleError::Tokenizer)
            }
            TokenizerSource::Default => Llama2CProcessor::load_tokenizer(None),
        }
    }

    /// Load model based on source
    fn load_model(&self) -> Result<(Llama, Config, Cache), CandleError> {
        match &self.model_source {
            ModelSource::File(path) => {
                let mut file = std::fs::File::open(path)
                    .map_err(|e| CandleError::Other(format!("Failed to open model file: {}", e)))?;

                let config = Config::from_reader(&mut file)
                    .map_err(|e| CandleError::Other(format!("Failed to read config: {}", e)))?;

                let weights = TransformerWeights::from_reader(&mut file, &config, &self.device)
                    .map_err(|e| CandleError::Other(format!("Failed to load weights: {}", e)))?;

                let vb = weights.var_builder(&config, &self.device).map_err(|e| {
                    CandleError::Other(format!("Failed to create var builder: {}", e))
                })?;

                let cache = Cache::new(true, &config, vb.pp("rot"))?;
                let model = Llama::load(vb, config.clone())?;

                Ok((model, config, cache))
            }
            ModelSource::Config(config) => {
                let vb = VarBuilder::zeros(DType::F32, &self.device);
                let cache = Cache::new(true, config, vb.pp("rot"))?;
                let model = Llama::load(vb, config.clone())?;
                Ok((model, config.clone(), cache))
            }
            ModelSource::HuggingFace { repo, file } => {
                use hf_hub::api::sync::Api;
                println!("Downloading model from HF: {}/{}", repo, file);
                let api = Api::new().map_err(|e| CandleError::Other(e.to_string()))?;
                let hf_repo = api.model(repo.clone());
                let path = hf_repo
                    .get(file)
                    .map_err(|e| CandleError::Other(e.to_string()))?;

                // Then load from downloaded file
                self.clone().with_model_file(path).load_model()
            }
        }
    }

    /// Apply LoRA adapters
    fn apply_lora(&self, processor: &mut Llama2CProcessor) -> Result<(), CandleError> {
        println!("📎 Applying {} LoRA adapter(s)...", self.lora_paths.len());

        let lora_config = LoraConfig::llama_default();
        let mut lora_manager = LoraManager::new(lora_config);

        for (idx, lora_path) in self.lora_paths.iter().enumerate() {
            println!(
                "   [{}] Loading LoRA from: {}",
                idx + 1,
                lora_path.display()
            );

            // Check if this is a directory or a file
            if lora_path.is_dir() {
                // Directory - check for PEFT format
                let adapter_config = lora_path.join("adapter_config.json");
                let adapter_weights = lora_path
                    .join("adapter_model.safetensors")
                    .exists()
                    .then_some(lora_path.join("adapter_model.safetensors"))
                    .or_else(|| {
                        lora_path
                            .join("adapter.safetensors")
                            .exists()
                            .then_some(lora_path.join("adapter.safetensors"))
                    });

                if adapter_config.exists() || adapter_weights.is_some() {
                    println!("   📚 Detected PEFT directory format, using PEFT directory loader");
                    lora_manager
                        .load_from_peft_dir(lora_path.to_str().unwrap(), &processor.device)?;
                } else {
                    return Err(CandleError::Candle(candle_core::Error::Msg(format!(
                        "Directory {} does not contain PEFT LoRA files (adapter_config.json or adapter*.safetensors)",
                        lora_path.display()
                    ))));
                }
            } else {
                // File - check if this is a PEFT format file by looking at tensor names
                let is_peft = {
                    let tensors = candle_core::safetensors::load(
                        lora_path.to_str().unwrap(),
                        &processor.device,
                    )?;
                    tensors.keys().any(|k| k.starts_with("base_model.model"))
                };

                if is_peft {
                    println!("   📚 Detected PEFT format, using PEFT loader");
                    lora_manager.load_from_peft(lora_path.to_str().unwrap(), &processor.device)?;
                } else {
                    lora_manager
                        .load_from_safetensors(lora_path.to_str().unwrap(), &processor.device)?;
                }
            }
        }

        processor.model.apply_lora(&lora_manager)?;
        println!("   🎯 LoRA adapters applied!");

        Ok(())
    }
}

/// LLaMA2-C based processor
pub struct Llama2CProcessor {
    model: Llama,
    #[allow(dead_code)]
    config: Config,
    cache: RwLock<Cache>, // RwLock for thread-safe interior mutability
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
        eprintln!("\n🚀 Llama2CProcessor.process called!");
        eprintln!("  Input message: {}", request.message);

        // Tokenize the input
        let encoding = self
            .tokenizer
            .encode(request.message.as_str(), false)
            .map_err(|e| CandleError::Other(e.to_string()))?;

        let mut tokens = encoding.get_ids().to_vec();
        eprintln!(
            "  Tokenized to {} tokens: {:?}",
            tokens.len(),
            &tokens[..tokens.len().min(10)]
        );

        if tokens.is_empty() {
            return Ok(Response::simple(
                request.id,
                "No tokens generated".to_string(),
            ));
        }

        // Generate tokens
        let max_new_tokens = 50; // Generate up to 50 new tokens
        println!("  Generating up to {} new tokens...", max_new_tokens);
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

        let message = format!("Generated: '{}'", generated_text);

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
