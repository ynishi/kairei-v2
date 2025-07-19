//! LoRA (Low-Rank Adaptation) implementation for Llama2c

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::Linear;
use std::collections::HashMap;

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
}

/// LoRA adapter layer
#[derive(Debug, Clone)]
pub struct LoraLinear {
    /// Base linear layer
    base: Linear,
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

    /// Apply LoRA weights to a linear layer if it's a target module
    pub fn apply_to_linear(&self, linear: Linear, module_name: &str) -> LoraLinear {
        // Check if this module should have LoRA
        let should_apply = self
            .config
            .target_modules
            .iter()
            .any(|target| module_name.contains(target));

        if should_apply {
            // Look for LoRA weights for this module
            for (lora_name, (lora_a, lora_b)) in &self.lora_weights {
                if module_name.contains(lora_name) {
                    println!("  ✨ Applying LoRA to: {}", module_name);
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
