//! Manual LoRA implementation for Llama2 (without macro magic)
//! 
//! Based on llama2c.rs from candle-lora-transformers

use async_trait::async_trait;
use candle_core::{Device, DType, Tensor, Result as CandleResult, D, IndexOp};
use candle_nn::Module;
use candle_nn::VarBuilder;
use candle_lora::{LoraLinear, LoraConfig, LinearLayerLike, LoraLinearConfig};
use candle_nn::{Linear, linear_no_bias as linear};
use std::sync::Arc;
use kairei_core::{Processor, ProcessorMetadata, Request, Response, Result as CoreResult};
use std::collections::HashMap;
use tokenizers::Tokenizer;

use crate::CandleError;

/// Cache for KV and positional embeddings
#[derive(Debug, Clone)]
pub struct Cache {
    masks: HashMap<usize, Tensor>,
    pub use_kv_cache: bool,
    pub kvs: Vec<Option<(Tensor, Tensor)>>,
    pub cos: Tensor,
    pub sin: Tensor,
    device: Device,
}

impl Cache {
    pub fn new(config: &Config, device: &Device, dtype: DType) -> Result<Self, CandleError> {
        let n_elem = config.dim / config.n_heads;
        let theta: Vec<_> = (0..n_elem)
            .step_by(2)
            .map(|i| 1f32 / 10000f32.powf(i as f32 / n_elem as f32))
            .collect();
        let theta = Tensor::new(theta.as_slice(), device)?;
        let idx_theta = Tensor::arange(0, config.seq_len as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((config.seq_len, 1))?
            .matmul(&theta.reshape((1, theta.elem_count()))?)?;
        let cos = idx_theta.cos()?;
        let sin = idx_theta.sin()?;
        let cos = cos.reshape((config.seq_len, config.head_size() / 2, 1))?;
        let sin = sin.reshape((config.seq_len, config.head_size() / 2, 1))?;
        
        Ok(Self {
            masks: HashMap::new(),
            use_kv_cache: true,
            kvs: vec![None; config.n_layers],
            cos,
            sin,
            device: device.clone(),
        })
    }
    
    pub fn mask(&mut self, t: usize) -> Result<Tensor, CandleError> {
        if let Some(mask) = self.masks.get(&t) {
            Ok(mask.clone())
        } else {
            let mask: Vec<_> = (0..t)
                .flat_map(|i| (0..t).map(move |j| u8::from(j > i)))
                .collect();
            let mask = Tensor::from_slice(&mask, (t, t), &self.device)?;
            self.masks.insert(t, mask.clone());
            Ok(mask)
        }
    }
}

/// Configuration for Llama2
#[derive(Debug, Clone)]
pub struct Config {
    pub dim: usize,
    pub hidden_dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub vocab_size: usize,
    pub seq_len: usize,
    pub norm_eps: f64,
}

impl Config {
    pub fn head_size(&self) -> usize {
        self.dim / self.n_heads
    }
}

/// SiLU activation function
fn silu(xs: &Tensor) -> CandleResult<Tensor> {
    xs / (xs.neg()?.exp()? + 1.0)?
}

/// Masked fill operation
fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: f32) -> CandleResult<Tensor> {
    let shape = mask.shape();
    let on_true = Tensor::new(on_true, on_false.device())?.broadcast_as(shape.dims())?;
    let m = mask.to_dtype(DType::U8)?;
    m.where_cond(&on_true, on_false)
}

/// Linear layer that can be either normal or LoRA
type DynLinear = Box<dyn LinearLayerLike + Send + Sync>;

// Simple LoRA layer implementation is commented out for now
// as we're using a different approach

/// Causal self-attention with LoRA support
struct CausalSelfAttention {
    q_proj: DynLinear,
    k_proj: DynLinear,
    v_proj: DynLinear,
    o_proj: DynLinear,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
}

impl CausalSelfAttention {
    
    fn new(
        q_proj: DynLinear,
        k_proj: DynLinear,
        v_proj: DynLinear,
        o_proj: DynLinear,
        n_head: usize,
        n_kv_head: usize,
        head_dim: usize,
    ) -> Self {
        Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            n_head,
            n_kv_head,
            head_dim,
        }
    }

    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self, CandleError> {
        let dim = cfg.dim;
        let q_proj = Box::new(linear(dim, dim, vb.pp("q_proj"))?) as DynLinear;
        let k_proj = Box::new(linear(dim, dim, vb.pp("k_proj"))?) as DynLinear;
        let v_proj = Box::new(linear(dim, dim, vb.pp("v_proj"))?) as DynLinear;
        let o_proj = Box::new(linear(dim, dim, vb.pp("o_proj"))?) as DynLinear;
        
        Ok(Self::new(
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            cfg.n_heads,
            cfg.n_kv_heads,
            cfg.head_size(),
        ))
    }

    fn forward(
        &self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        cache: &mut Cache,
    ) -> Result<Tensor, CandleError> {
        let (b_sz, seq_len, n_embd) = x.dims3()?;
        
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;
        
        let q = q.reshape((b_sz, seq_len, self.n_head, self.head_dim))?;
        let k = k.reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?;
        let mut v = v.reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?;
        
        // Apply rotary embeddings
        let q = self.apply_rotary_emb(&q, index_pos, cache)?;
        let mut k = self.apply_rotary_emb(&k, index_pos, cache)?;
        
        // KV cache
        if cache.use_kv_cache {
            if let Some((k_cache, v_cache)) = &cache.kvs[block_idx] {
                k = Tensor::cat(&[k_cache, &k], 1)?;
                v = Tensor::cat(&[v_cache, &v], 1)?;
            }
            cache.kvs[block_idx] = Some((k.clone(), v.clone()));
        }
        
        // Repeat KV heads if necessary
        let k = self.repeat_kv(k)?;
        let v = self.repeat_kv(v)?;
        
        // Attention
        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;
        
        let scale = 1f64 / (self.head_dim as f64).sqrt();
        let attn_weights = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        
        // Apply mask
        let mask = cache.mask(seq_len)?.broadcast_as(attn_weights.shape())?;
        let attn_weights = masked_fill(&attn_weights, &mask, f32::NEG_INFINITY)?;
        
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&v)?;
        
        let attn_output = attn_output.transpose(1, 2)?.reshape((b_sz, seq_len, n_embd))?;
        self.o_proj.forward(&attn_output).map_err(CandleError::from)
    }
    
    fn apply_rotary_emb(&self, x: &Tensor, index_pos: usize, cache: &Cache) -> Result<Tensor, CandleError> {
        let (b_sz, seq_len, n_head, head_dim) = x.dims4()?;
        let cos = cache.cos.i(index_pos..index_pos + seq_len)?;
        let sin = cache.sin.i(index_pos..index_pos + seq_len)?;
        let cos = cos.unsqueeze(1)?;
        let sin = sin.unsqueeze(1)?;
        let cos = cos.broadcast_as((seq_len, n_head, head_dim / 2, 1))?;
        let sin = sin.broadcast_as((seq_len, n_head, head_dim / 2, 1))?;
        let x1 = x.narrow(D::Minus1, 0, head_dim / 2)?;
        let x2 = x.narrow(D::Minus1, head_dim / 2, head_dim / 2)?;
        let rotate_x = Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)?;
        let rope = (x.broadcast_mul(&cos)? + rotate_x.broadcast_mul(&sin)?)?;
        Ok(rope)
    }
    
    fn repeat_kv(&self, x: Tensor) -> Result<Tensor, CandleError> {
        if self.n_kv_head == self.n_head {
            Ok(x)
        } else {
            let (b_sz, seq_len, n_kv_head, head_dim) = x.dims4()?;
            let n_rep = self.n_head / self.n_kv_head;
            let x = x.unsqueeze(3)?
                .expand((b_sz, seq_len, n_kv_head, n_rep, head_dim))?
                .reshape((b_sz, seq_len, n_kv_head * n_rep, head_dim))?;
            Ok(x)
        }
    }
}

/// MLP layer with LoRA support
struct Mlp {
    gate_proj: DynLinear,
    up_proj: DynLinear,
    down_proj: DynLinear,
}

impl Mlp {
    fn new(gate_proj: DynLinear, up_proj: DynLinear, down_proj: DynLinear) -> Self {
        Self {
            gate_proj,
            up_proj,
            down_proj,
        }
    }

    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self, CandleError> {
        let h_size = cfg.dim;
        let i_size = cfg.hidden_dim;
        let gate_proj = Box::new(linear(h_size, i_size, vb.pp("gate_proj"))?) as DynLinear;
        let up_proj = Box::new(linear(h_size, i_size, vb.pp("up_proj"))?) as DynLinear;
        let down_proj = Box::new(linear(i_size, h_size, vb.pp("down_proj"))?) as DynLinear;
        Ok(Self::new(gate_proj, up_proj, down_proj))
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor, CandleError> {
        let x = (silu(&self.gate_proj.forward(x)?)? * self.up_proj.forward(x)?)?;
        self.down_proj.forward(&x).map_err(CandleError::from)
    }
}

/// Transformer block
struct Block {
    rms_1: candle_nn::RmsNorm,
    attn: CausalSelfAttention,
    rms_2: candle_nn::RmsNorm,
    mlp: Mlp,
}

impl Block {
    fn new(
        rms_1: candle_nn::RmsNorm,
        attn: CausalSelfAttention,
        rms_2: candle_nn::RmsNorm,
        mlp: Mlp,
    ) -> Self {
        Self {
            rms_1,
            attn,
            rms_2,
            mlp,
        }
    }

    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self, CandleError> {
        use candle_nn::rms_norm;
        
        let rms_1 = rms_norm(cfg.dim, cfg.norm_eps, vb.pp("input_layernorm"))?;
        let rms_2 = rms_norm(cfg.dim, cfg.norm_eps, vb.pp("post_attention_layernorm"))?;
        let attn = CausalSelfAttention::load(vb.pp("self_attn"), cfg)?;
        let mlp = Mlp::load(vb.pp("mlp"), cfg)?;
        
        Ok(Self::new(rms_1, attn, rms_2, mlp))
    }

    fn forward(
        &self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        cache: &mut Cache,
    ) -> Result<Tensor, CandleError> {
        let residual = x;
        let x = self.rms_1.forward(x)?;
        let x = (self.attn.forward(&x, index_pos, block_idx, cache)? + residual)?;
        let residual = &x;
        let x = (self.mlp.forward(&self.rms_2.forward(&x)?)? + residual)?;
        Ok(x)
    }
}

/// Main Llama2 model with LoRA support
pub struct Llama2WithLora {
    wte: candle_nn::Embedding,
    blocks: Vec<Block>,
    ln_f: candle_nn::RmsNorm,
    lm_head: DynLinear,
    cache: Cache,
    lora_weights: HashMap<String, (Tensor, Tensor, f64)>, // name -> (A, B, scale)
}

impl Llama2WithLora {
    pub fn load(vb: VarBuilder, cfg: &Config, device: &Device, dtype: DType) -> Result<Self, CandleError> {
        use candle_nn::{embedding, rms_norm};
        
        println!("🔨 Loading Llama2 model...");
        
        // Load embeddings
        let wte = embedding(cfg.vocab_size, cfg.dim, vb.pp("model.embed_tokens"))?;
        
        // Load blocks
        let mut blocks = Vec::with_capacity(cfg.n_layers);
        for idx in 0..cfg.n_layers {
            println!("  Loading block {}/{}", idx + 1, cfg.n_layers);
            let block = Block::load(vb.pp(&format!("model.layers.{}", idx)), cfg)?;
            blocks.push(block);
        }
        
        // Load final norm and head
        let ln_f = rms_norm(cfg.dim, cfg.norm_eps, vb.pp("model.norm"))?;
        let lm_head = Box::new(linear(cfg.dim, cfg.vocab_size, vb.pp("lm_head"))?) as DynLinear;
        
        // Initialize cache
        let cache = Cache::new(cfg, device, dtype)?;
        
        println!("✅ Model loaded successfully");
        
        Ok(Self {
            wte,
            blocks,
            ln_f,
            lm_head,
            cache,
            lora_weights: HashMap::new(),
        })
    }

    pub fn apply_lora_weights(&mut self, lora_weights: HashMap<String, (Tensor, Tensor)>, lora_config: &LoraConfig) -> Result<(), CandleError> {
        println!("🎯 Storing LoRA weights for {} layers", lora_weights.len());
        
        for (name, (lora_a, lora_b)) in lora_weights {
            println!("  Processing LoRA weights for: {}", name);
            
            // Get rank from lora_a dimensions
            let rank = lora_a.dim(1)?;
            // Use the same alpha value as in the config creation (32.0)
            let alpha = 32.0;
            let scale = alpha / rank as f64;
            
            // Store the weights with scale
            self.lora_weights.insert(name.clone(), (lora_a, lora_b, scale));
            println!("    ✓ Stored with rank={}, scale={:.4}", rank, scale);
        }
        
        println!("✅ LoRA weights stored successfully");
        Ok(())
    }

    pub fn forward(&mut self, input_ids: &Tensor, index_pos: usize) -> Result<Tensor, CandleError> {
        let (_b_sz, seq_len) = input_ids.dims2()?;
        
        // Embedding
        let mut x = self.wte.forward(input_ids)?;
        
        // Pass through all blocks
        for (block_idx, block) in self.blocks.iter().enumerate() {
            x = block.forward(&x, index_pos, block_idx, &mut self.cache)?;
        }
        
        // Final norm and head
        let x = self.ln_f.forward(&x)?;
        let logits = self.lm_head.forward(&x)?;
        
        Ok(logits)
    }
}

/// Processor implementation
pub struct ManualLlama2LoraProcessor {
    model: Llama2WithLora,
    tokenizer: Tokenizer,
    device: Device,
    max_tokens: usize,
}

impl ManualLlama2LoraProcessor {
    pub fn new(
        model_path: &str,
        tokenizer_path: &str,
        lora_path: Option<&str>,
        device: Device,
        dtype: DType,
        config: Config,
        max_tokens: usize,
    ) -> Result<Self, CandleError> {
        println!("🚀 Initializing Manual Llama2 LoRA Processor...");
        
        // Load tokenizer
        println!("📖 Loading tokenizer from: {}", tokenizer_path);
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| CandleError::Other(format!("Failed to load tokenizer: {}", e)))?;
        
        // Load model weights
        println!("🏗️ Loading model weights from: {}", model_path);
        let tensors = candle_core::safetensors::load(model_path, &device)?;
        let vb = VarBuilder::from_tensors(tensors, dtype, &device);
        
        // Load model
        let mut model = Llama2WithLora::load(vb, &config, &device, dtype)?;
        
        // Apply LoRA weights if provided
        if let Some(lora_path) = lora_path {
            println!("🎯 Loading LoRA weights from: {}", lora_path);
            let lora_tensors = candle_core::safetensors::load(lora_path, &device)?;
            
            let mut lora_weights = HashMap::new();
            let lora_config = LoraConfig::new(16, 32.0, Some(0.0));
            
            // Extract LoRA A/B pairs
            for (name, tensor) in lora_tensors.iter() {
                if name.contains(".lora_a.") {
                    let base_name = name.replace(".lora_a.weight", "");
                    let b_name = name.replace(".lora_a.", ".lora_b.");
                    
                    if let Some(b_tensor) = lora_tensors.get(&b_name) {
                        println!("   Found LoRA pair: {}", base_name);
                        lora_weights.insert(base_name, (tensor.clone(), b_tensor.clone()));
                    }
                }
            }
            
            println!("✅ Loaded {} LoRA weight pairs", lora_weights.len());
            model.apply_lora_weights(lora_weights, &lora_config)?;
        }
        
        Ok(Self {
            model,
            tokenizer,
            device,
            max_tokens,
        })
    }
}

#[async_trait]
impl Processor for ManualLlama2LoraProcessor {
    async fn process(&self, request: Request) -> CoreResult<Response> {
        println!("🚀 ManualLlama2LoraProcessor.process called!");
        println!("  Input message: {}", request.message);
        
        // Simple test to verify LoRA is loaded
        if self.model.lora_weights.is_empty() {
            println!("  ⚠️  No LoRA weights loaded");
        } else {
            println!("  ✅ LoRA weights loaded for {} layers", self.model.lora_weights.len());
            
            // Print loaded LoRA layers
            for (name, (a, b, scale)) in &self.model.lora_weights {
                println!("    - {} (rank={}, scale={:.4})", name, a.dim(1).unwrap_or(0), scale);
            }
        }
        
        // For now, return a simple response showing LoRA status
        let response_text = if self.model.lora_weights.is_empty() {
            "Manual LoRA processor initialized (no LoRA weights loaded)".to_string()
        } else {
            format!("Manual LoRA processor initialized with {} LoRA layers!", self.model.lora_weights.len())
        };
        
        Ok(Response::simple(request.id, response_text))
    }

    fn metadata(&self) -> ProcessorMetadata {
        ProcessorMetadata {
            name: Some("Manual Llama2 LoRA Processor".to_string()),
            description: Some("Llama2 with manual LoRA implementation".to_string()),
            version: Some("0.1.0".to_string()),
            capabilities: vec!["text-generation".to_string(), "lora".to_string()],
        }
    }
}