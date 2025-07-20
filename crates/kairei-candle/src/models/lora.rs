//! LoRA (Low-Rank Adaptation) implementation for Candle models
//!
//! This module provides generic LoRA functionality that can be used with any Candle-based model.
//! It includes PEFT (Parameter-Efficient Fine-Tuning) format compatibility for seamless
//! integration with HuggingFace PEFT adapters.

use async_trait::async_trait;
use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::Linear;
use kairei_core::{LoRATunerConfig, Tuner, TunerMetadata, TuningContext, TuningResult};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

/// LoRA adapter configuration
#[derive(Debug, Clone)]
pub struct LoraConfig {
    /// Rank of the LoRA adaptation
    pub r: usize,
    /// Scaling factor (alpha / r)
    pub alpha: f64,
    /// Dropout rate (not implemented yet)
    pub dropout: f64,
    /// Target modules to apply LoRA
    pub target_modules: Vec<String>,
}

impl Default for LoraConfig {
    fn default() -> Self {
        Self {
            r: 16,
            alpha: 16.0,
            dropout: 0.0,
            // Empty by default - should be configured per model type
            target_modules: vec![],
        }
    }
}

impl LoraConfig {
    /// Create a configuration with common Llama-style target modules
    pub fn llama_default() -> Self {
        Self {
            r: 16,
            alpha: 16.0,
            dropout: 0.0,
            target_modules: vec![
                "q_proj".to_string(),
                "v_proj".to_string(),
                "k_proj".to_string(),
                "o_proj".to_string(),
                "gate_proj".to_string(),
                "up_proj".to_string(),
                "down_proj".to_string(),
            ],
        }
    }

    /// Create a configuration with basic attention-only target modules
    pub fn attention_only() -> Self {
        Self {
            r: 16,
            alpha: 16.0,
            dropout: 0.0,
            target_modules: vec![
                "q_proj".to_string(),
                "v_proj".to_string(),
                "k_proj".to_string(),
                "o_proj".to_string(),
            ],
        }
    }
}

/// LoRA adapter layer
#[derive(Debug, Clone)]
pub struct LoraLinear {
    /// Base linear layer
    pub base: Linear,
    /// LoRA A matrix (down projection)
    lora_a: Option<Tensor>,
    /// LoRA B matrix (up projection)
    lora_b: Option<Tensor>,
    /// Scaling factor
    scale: f64,
}

impl LoraLinear {
    /// Create a new LoRA linear layer
    pub fn new(base: Linear, lora_a: Option<Tensor>, lora_b: Option<Tensor>, scale: f64) -> Self {
        Self {
            base,
            lora_a,
            lora_b,
            scale,
        }
    }

    /// Create from a base linear layer without LoRA
    pub fn from_linear(base: Linear) -> Self {
        Self {
            base,
            lora_a: None,
            lora_b: None,
            scale: 1.0,
        }
    }

    /// Add LoRA adapter to the layer
    pub fn add_lora(
        &mut self,
        rank: usize,
        alpha: f64,
        device: &Device,
        dtype: DType,
    ) -> Result<()> {
        let (out_features, in_features) = self.base.weight().dims2()?;

        // Initialize LoRA matrices
        // A: (in_features, rank) - initialized with normal distribution
        // B: (rank, out_features) - initialized with zeros
        let lora_a = Tensor::randn(0.0, 0.02, (in_features, rank), device)?.to_dtype(dtype)?;
        let lora_b = Tensor::zeros((rank, out_features), dtype, device)?;

        self.lora_a = Some(lora_a);
        self.lora_b = Some(lora_b);
        self.scale = alpha / rank as f64;

        Ok(())
    }

    /// Load LoRA weights from tensors
    pub fn load_lora(&mut self, lora_a: Tensor, lora_b: Tensor, alpha: f64, rank: usize) {
        self.lora_a = Some(lora_a);
        self.lora_b = Some(lora_b);
        self.scale = alpha / rank as f64;
    }
}

impl Module for LoraLinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Base forward pass
        let mut output = self.base.forward(x)?;

        // Add LoRA if present
        if let (Some(lora_a), Some(lora_b)) = (&self.lora_a, &self.lora_b) {
            // x @ A @ B * scale
            let lora_out = x.matmul(lora_a)?.matmul(lora_b)?.affine(self.scale, 0.0)?;
            output = (output + lora_out)?;
        }

        Ok(output)
    }
}

/// PEFT adapter_config.json structure
#[derive(Debug, Deserialize)]
pub struct PeftConfig {
    pub r: usize,
    pub lora_alpha: f64,
    #[serde(default)]
    pub lora_dropout: f64,
    pub target_modules: Vec<String>,
    pub peft_type: String,
    #[serde(default)]
    pub base_model_name_or_path: String,
}

/// Manager for LoRA adapters in a model
pub struct LoraManager {
    /// LoRA configuration
    pub config: LoraConfig,
    /// Loaded LoRA weights by name
    pub lora_weights: HashMap<String, (Tensor, Tensor)>,
}

impl LoraManager {
    pub fn new(config: LoraConfig) -> Self {
        Self {
            config,
            lora_weights: HashMap::new(),
        }
    }

    /// Load LoRA weights from a safetensors file
    pub fn load_from_safetensors(&mut self, path: &str, device: &Device) -> Result<()> {
        println!("🎯 Loading LoRA weights from: {}", path);

        // Load the safetensors file
        let tensors = candle_core::safetensors::load(path, device)?;

        // Extract LoRA weights
        for (name, tensor) in tensors.iter() {
            if name.contains(".lora_A.") {
                let base_name = name.replace(".lora_A.weight", "");
                if let Some(lora_b_tensor) = tensors.get(&format!("{}.lora_B.weight", base_name)) {
                    println!("  Found LoRA pair for: {}", base_name);
                    self.lora_weights
                        .insert(base_name.clone(), (tensor.clone(), lora_b_tensor.clone()));
                }
            }
        }

        println!("  Loaded {} LoRA adapter pairs", self.lora_weights.len());
        Ok(())
    }

    /// Load LoRA weights from a PEFT format safetensors file
    pub fn load_from_peft(&mut self, path: &str, device: &Device) -> Result<()> {
        println!("🎯 Loading PEFT LoRA weights from: {}", path);

        // Load the safetensors file
        let tensors = candle_core::safetensors::load(path, device)?;

        // PEFT uses hierarchical naming like:
        // base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight
        // We need to extract the meaningful parts for matching

        let mut peft_pairs = HashMap::new();

        // First pass: collect all PEFT LoRA pairs
        for (name, tensor) in tensors.iter() {
            if name.contains(".lora_A.weight") {
                let base_name = name.replace(".lora_A.weight", "");
                let b_name = format!("{}.lora_B.weight", base_name);

                if let Some(lora_b_tensor) = tensors.get(&b_name) {
                    // Extract the meaningful parts for our model structure
                    // We'll create multiple possible keys for flexible matching
                    let parts: Vec<&str> = base_name.split('.').collect();

                    // Find the layer number and module name
                    let mut layer_idx = None;
                    let mut module_type = None; // self_attn or mlp
                    let mut module_name = None; // q_proj, k_proj, etc.

                    for (i, part) in parts.iter().enumerate() {
                        if *part == "layers" && i + 1 < parts.len() {
                            if let Ok(idx) = parts[i + 1].parse::<usize>() {
                                layer_idx = Some(idx);
                            }
                        }
                        // Track module type (common patterns: self_attn, mlp, attention, feed_forward)
                        if matches!(
                            *part,
                            "self_attn" | "mlp" | "attention" | "feed_forward" | "attn" | "ffn"
                        ) {
                            module_type = Some(*part);
                            if i + 1 < parts.len() {
                                module_name = Some(parts[i + 1]);
                            }
                        }
                    }

                    // Check for common special modules (language model heads, embeddings, etc.)
                    let special_modules = [
                        "lm_head",
                        "embed_tokens",
                        "embed_in",
                        "embed_out",
                        "head",
                        "classifier",
                    ];
                    let mut found_special = false;
                    for special in &special_modules {
                        if base_name.contains(special) {
                            peft_pairs.insert(
                                special.to_string(),
                                (tensor.clone(), lora_b_tensor.clone()),
                            );
                            found_special = true;
                            break;
                        }
                    }
                    if found_special {
                        continue;
                    }

                    // Create multiple key formats for flexible matching
                    if let (Some(layer), Some(module)) = (layer_idx, module_name) {
                        // Format 1: layers.{layer}.{module} (e.g., layers.0.q_proj)
                        let key1 = format!("layers.{}.{}", layer, module);
                        peft_pairs.insert(key1.clone(), (tensor.clone(), lora_b_tensor.clone()));

                        // Format 2: Include module type for more specific matching
                        if let Some(mod_type) = module_type {
                            let key2 = format!("layers.{}.{}.{}", layer, mod_type, module);
                            peft_pairs.insert(key2, (tensor.clone(), lora_b_tensor.clone()));
                        }

                        // Format 3: Just the module name for broad matching
                        peft_pairs
                            .insert(module.to_string(), (tensor.clone(), lora_b_tensor.clone()));
                    }
                }
            }
        }

        // Clear existing weights and store new ones
        self.lora_weights.clear();
        for (key, (lora_a, lora_b)) in peft_pairs {
            println!("  Found PEFT LoRA pair for: {}", key);
            self.lora_weights.insert(key, (lora_a, lora_b));
        }

        println!(
            "  Loaded {} PEFT LoRA adapter pairs",
            self.lora_weights.len()
        );
        Ok(())
    }

    /// Load LoRA configuration from PEFT adapter_config.json
    pub fn load_peft_config(&mut self, config_path: &str) -> Result<()> {
        let config_str = std::fs::read_to_string(config_path)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to read config: {}", e)))?;

        let peft_config: PeftConfig = serde_json::from_str(&config_str)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to parse config: {}", e)))?;

        // Update our config from PEFT config
        self.config.r = peft_config.r;
        self.config.alpha = peft_config.lora_alpha;
        self.config.dropout = peft_config.lora_dropout;
        self.config.target_modules = peft_config.target_modules;

        println!(
            "📋 Loaded PEFT config: r={}, alpha={}",
            self.config.r, self.config.alpha
        );
        Ok(())
    }

    /// Load LoRA weights from a PEFT directory (with adapter_config.json)
    pub fn load_from_peft_dir(&mut self, dir_path: &str, device: &Device) -> Result<()> {
        // First load config if available
        let config_path = Path::new(dir_path).join("adapter_config.json");
        if config_path.exists() {
            self.load_peft_config(config_path.to_str().unwrap())?;
        }

        // Then load weights - try both naming conventions
        let adapter_path = Path::new(dir_path).join("adapter_model.safetensors");
        let adapter_path_alt = Path::new(dir_path).join("adapter.safetensors");

        if adapter_path.exists() {
            self.load_from_peft(adapter_path.to_str().unwrap(), device)
        } else if adapter_path_alt.exists() {
            self.load_from_peft(adapter_path_alt.to_str().unwrap(), device)
        } else {
            Err(candle_core::Error::Msg(
                "No adapter weights found (tried adapter_model.safetensors and adapter.safetensors)".to_string()
            ))
        }
    }

    /// Apply LoRA weights to a linear layer if it's a target module
    pub fn apply_to_linear(&self, linear: Linear, module_name: &str) -> LoraLinear {
        // Check if this module should have LoRA
        let should_apply = self
            .config
            .target_modules
            .iter()
            .any(|target| module_name.contains(target));

        if should_apply {
            // Try multiple matching strategies for flexible compatibility
            // Strategy 1: Exact match
            if let Some((lora_a, lora_b)) = self.lora_weights.get(module_name) {
                println!("  ✨ Applying LoRA to: {} (exact match)", module_name);
                let mut lora_linear = LoraLinear::from_linear(linear);
                lora_linear.load_lora(
                    lora_a.clone(),
                    lora_b.clone(),
                    self.config.alpha,
                    self.config.r,
                );
                return lora_linear;
            }

            // Strategy 2: Contains match (original behavior)
            for (lora_name, (lora_a, lora_b)) in &self.lora_weights {
                if module_name.contains(lora_name) || lora_name.contains(module_name) {
                    println!(
                        "  ✨ Applying LoRA to: {} (matched with: {})",
                        module_name, lora_name
                    );
                    let mut lora_linear = LoraLinear::from_linear(linear);
                    lora_linear.load_lora(
                        lora_a.clone(),
                        lora_b.clone(),
                        self.config.alpha,
                        self.config.r,
                    );
                    return lora_linear;
                }
            }

            // Strategy 3: Extract just the module suffix and try to match
            let parts: Vec<&str> = module_name.split('.').collect();
            if let Some(suffix) = parts.last() {
                if let Some((lora_a, lora_b)) = self.lora_weights.get(*suffix) {
                    println!(
                        "  ✨ Applying LoRA to: {} (suffix match: {})",
                        module_name, suffix
                    );
                    let mut lora_linear = LoraLinear::from_linear(linear);
                    lora_linear.load_lora(
                        lora_a.clone(),
                        lora_b.clone(),
                        self.config.alpha,
                        self.config.r,
                    );
                    return lora_linear;
                }
            }
        }

        // No LoRA for this module
        LoraLinear::from_linear(linear)
    }
}

/// Tuner implementation for LoRA
pub struct CandleLoRATuner {
    /// LoRA manager
    manager: LoraManager,
    /// Device for computations
    device: Device,
    /// Data type
    dtype: DType,
}

impl CandleLoRATuner {
    /// Create a new LoRA tuner
    pub fn new(config: LoraConfig, device: Device, dtype: DType) -> Self {
        Self {
            manager: LoraManager::new(config),
            device,
            dtype,
        }
    }

    /// Create from a LoRATunerConfig (from kairei-core)
    pub fn from_core_config(config: &LoRATunerConfig, device: Device, dtype: DType) -> Self {
        let lora_config = LoraConfig {
            r: config.rank,
            alpha: config.alpha as f64,
            dropout: config.dropout.unwrap_or(0.0) as f64,
            target_modules: config.target_modules.clone(),
        };
        Self::new(lora_config, device, dtype)
    }

    /// Get the underlying LoRA manager
    pub fn manager(&self) -> &LoraManager {
        &self.manager
    }

    /// Get mutable reference to the LoRA manager
    pub fn manager_mut(&mut self) -> &mut LoraManager {
        &mut self.manager
    }
}

#[async_trait]
impl Tuner for CandleLoRATuner {
    async fn apply(&self, _context: TuningContext) -> kairei_core::Result<TuningResult> {
        // For now, LoRA application is done through the LoraLinear layers
        // This method could be used for fine-tuning in the future
        Ok(TuningResult {
            success: true,
            loss: None,
            metrics: HashMap::new(),
            messages: vec!["LoRA weights applied through model layers".to_string()],
        })
    }

    async fn save(&self, path: &str) -> kairei_core::Result<()> {
        // Convert LoRA weights to safetensors format
        let mut tensors = HashMap::new();

        for (name, (lora_a, lora_b)) in &self.manager.lora_weights {
            tensors.insert(format!("{}.lora_A.weight", name), lora_a.clone());
            tensors.insert(format!("{}.lora_B.weight", name), lora_b.clone());
        }

        // Save using candle's safetensors
        candle_core::safetensors::save(&tensors, path).map_err(|e| {
            kairei_core::CoreError::Processing(format!("Failed to save LoRA weights: {}", e))
        })?;

        Ok(())
    }

    async fn load(&self, path: &str) -> kairei_core::Result<()> {
        // This would need a mutable reference to actually load weights
        // For now, use manager's load methods directly
        Err(kairei_core::CoreError::Processing(
            "Use manager_mut().load_from_safetensors() for loading".to_string(),
        ))
    }

    fn metadata(&self) -> TunerMetadata {
        TunerMetadata {
            name: Some("CandleLoRATuner".to_string()),
            tuning_type: Some("LoRA".to_string()),
            description: Some("Low-Rank Adaptation for Candle models".to_string()),
            version: Some("0.1.0".to_string()),
            supported_models: vec![
                "llama".to_string(),
                "llama2".to_string(),
                "llama3".to_string(),
            ],
            capabilities: vec![
                "peft_compatible".to_string(),
                "safetensors".to_string(),
                "flexible_matching".to_string(),
            ],
        }
    }
}
