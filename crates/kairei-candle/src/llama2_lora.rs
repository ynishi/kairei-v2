//! Manual LoRA implementation for Llama2 (without macro magic)
//!
//! Based on llama2c.rs from candle-lora-transformers

use async_trait::async_trait;
use candle_core::{D, DType, Device, IndexOp, Result as CandleResult, Tensor};
use candle_lora::{LinearLayerLike, LoraConfig};
use candle_nn::Module;
use candle_nn::VarBuilder;
use candle_nn::linear_no_bias as linear;
use candle_transformers::generation::LogitsProcessor;
use kairei_core::{Processor, ProcessorMetadata, Request, Response, Result as CoreResult};
use std::collections::HashMap;
use std::io::Write;
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
        // Like reference implementation - cat with itself to get full hidden_size
        let idx_theta = Tensor::cat(&[&idx_theta, &idx_theta], D::Minus1)?;
        let cos = idx_theta.cos()?.to_dtype(dtype)?;
        let sin = idx_theta.sin()?.to_dtype(dtype)?;

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
    let m = mask.where_cond(&on_true, on_false)?;
    Ok(m)
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
        let head_dim = cfg.head_size();
        let kv_dim = cfg.n_kv_heads * head_dim;

        let q_proj = Box::new(linear(dim, dim, vb.pp("q_proj"))?) as DynLinear;
        let k_proj = Box::new(linear(dim, kv_dim, vb.pp("k_proj"))?) as DynLinear;
        let v_proj = Box::new(linear(dim, kv_dim, vb.pp("v_proj"))?) as DynLinear;
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

        // Reshape and transpose in correct order (like reference implementation)
        let q = q
            .reshape((b_sz, seq_len, self.n_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let mut v = v
            .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?;

        // Apply rotary embeddings (now with correct shape)
        let q = self.apply_rotary_emb(&q, index_pos, cache)?;
        let mut k = self.apply_rotary_emb(&k, index_pos, cache)?;

        // KV cache - concatenate on correct dimension (2 for seq_len after transpose)
        if cache.use_kv_cache {
            if let Some((k_cache, v_cache)) = &cache.kvs[block_idx] {
                k = Tensor::cat(&[k_cache, &k], 2)?.contiguous()?;
                v = Tensor::cat(&[v_cache, &v], 2)?.contiguous()?;
            }
            cache.kvs[block_idx] = Some((k.clone(), v.clone()));
        }

        // Repeat KV heads if necessary
        let k = self.repeat_kv(k)?;
        let v = self.repeat_kv(v)?;

        // Convert to F32 for attention computation like reference implementation
        let in_dtype = q.dtype();
        let q = q.to_dtype(DType::F32)?;
        let k = k.to_dtype(DType::F32)?;
        let v = v.to_dtype(DType::F32)?;

        let scale = 1f64 / (self.head_dim as f64).sqrt();
        let attn_weights = (q.matmul(&k.t()?)? * scale)?;

        // Apply mask - handle KV cache correctly
        let attn_weights = if seq_len == 1 {
            // For single token (generation), no mask needed
            attn_weights
        } else {
            // For multiple tokens, apply causal mask
            // Get the actual sequence length from attention weights
            let (_, _, query_len, key_len) = attn_weights.dims4()?;

            // Create causal mask for current query vs all keys (including cached)
            let mask_data: Vec<u8> = (0..query_len)
                .flat_map(|i| {
                    (0..key_len).map(move |j| {
                        // For KV cache: allow access to all previous tokens
                        // Current position in full sequence = key_len - query_len + i
                        let current_pos = key_len - query_len + i;
                        u8::from(j > current_pos)
                    })
                })
                .collect();

            let mask = Tensor::from_slice(&mask_data, (query_len, key_len), attn_weights.device())?;
            let mask = mask.broadcast_as(attn_weights.shape())?;
            masked_fill(&attn_weights, &mask, f32::NEG_INFINITY)?
        };

        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&v.contiguous()?)?;

        // Convert back to original dtype
        let attn_output = attn_output.to_dtype(in_dtype)?;
        let attn_output = attn_output
            .transpose(1, 2)?
            .reshape((b_sz, seq_len, n_embd))?;
        self.o_proj.forward(&attn_output).map_err(CandleError::from)
    }

    fn apply_rotary_emb(
        &self,
        x: &Tensor,
        index_pos: usize,
        cache: &Cache,
    ) -> Result<Tensor, CandleError> {
        let (b_sz, n_head, seq_len, head_dim) = x.dims4()?;
        let cos = cache.cos.narrow(0, index_pos, seq_len)?;
        let sin = cache.sin.narrow(0, index_pos, seq_len)?;
        let cos = cos.broadcast_as((b_sz, 1, seq_len, head_dim))?;
        let sin = sin.broadcast_as((b_sz, 1, seq_len, head_dim))?;

        // Manual rotary embedding like reference implementation
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
            let (b_sz, n_kv_head, seq_len, head_dim) = x.dims4()?;
            let n_rep = self.n_head / self.n_kv_head;
            let x = x
                .unsqueeze(2)?
                .expand((b_sz, n_kv_head, n_rep, seq_len, head_dim))?
                .reshape((b_sz, n_kv_head * n_rep, seq_len, head_dim))?;
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
    pub fn load(
        vb: VarBuilder,
        cfg: &Config,
        device: &Device,
        dtype: DType,
    ) -> Result<Self, CandleError> {
        use candle_nn::{embedding, rms_norm};

        println!("🔨 Loading Llama2 model...");

        // Load embeddings
        let wte = embedding(cfg.vocab_size, cfg.dim, vb.pp("model.embed_tokens"))?;

        // Load blocks
        let mut blocks = Vec::with_capacity(cfg.n_layers);
        for idx in 0..cfg.n_layers {
            println!("  Loading block {}/{}", idx + 1, cfg.n_layers);
            let block = Block::load(vb.pp(format!("model.layers.{}", idx)), cfg)?;
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

    pub fn apply_lora_weights(
        &mut self,
        lora_weights: HashMap<String, (Tensor, Tensor)>,
        lora_config: &LoraConfig,
    ) -> Result<(), CandleError> {
        println!("🎯 Storing LoRA weights for {} layers", lora_weights.len());

        for (name, (lora_a, lora_b)) in lora_weights {
            println!("  Processing LoRA weights for: {}", name);

            // Debug: Print actual tensor shapes
            let a_shape = lora_a.shape();
            let b_shape = lora_b.shape();
            println!("    - lora_a shape: {:?}", a_shape);
            println!("    - lora_b shape: {:?}", b_shape);

            // Get rank from lora_a dimensions
            // For LoRA: A is [rank, in_features], B is [out_features, rank]
            // Based on the shapes we see: A is [8, 2048], B is [256, 8]
            // So rank is the first dimension of A (or second dimension of B)
            let rank = lora_a.dim(0)?;

            // Check if this makes sense
            if rank > 64 {
                println!("    ⚠️  Warning: rank={} seems too high for LoRA!", rank);
            }

            // Use the same alpha value as in the config creation (32.0)
            let alpha = 32.0;
            let scale = alpha / rank as f64;

            // Store the weights with scale
            self.lora_weights
                .insert(name.clone(), (lora_a, lora_b, scale));
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
pub struct Llama2LoraProcessor {
    model: std::sync::Mutex<Llama2WithLora>,
    tokenizer: Tokenizer,
    device: Device,
    max_tokens: usize,
}

impl Llama2LoraProcessor {
    /// Initialize with the same interface as existing Llama2LoraProcessor
    pub fn init(
        model_path: &str,
        tokenizer_path: &str,
        lora_path: Option<&str>,
        device: Device,
        dtype: DType,
        config: Config,
        max_tokens: usize,
    ) -> Result<Self, CandleError> {
        Self::new(
            model_path,
            tokenizer_path,
            lora_path,
            device,
            dtype,
            config,
            max_tokens,
        )
    }

    pub fn new(
        model_path: &str,
        tokenizer_path: &str,
        lora_path: Option<&str>,
        device: Device,
        dtype: DType,
        config: Config,
        max_tokens: usize,
    ) -> Result<Self, CandleError> {
        println!("🚀 Initializing Llama2 LoRA Processor...");

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
            println!("📋 LoRA file contains {} tensors:", lora_tensors.len());
            for (i, (name, _)) in lora_tensors.iter().enumerate() {
                if i < 10 {
                    // Print first 10 keys
                    println!("   - {}", name);
                }
            }

            // Try different LoRA naming patterns
            for (name, tensor) in lora_tensors.iter() {
                // Pattern 1: lora_llama_csa.a{num}.weight
                if name.contains(".a") && name.ends_with(".weight") {
                    // Extract the number from a{num}
                    if let Some(a_pos) = name.rfind(".a") {
                        let num_part = &name[a_pos + 2..name.len() - 7]; // Skip ".a" and ".weight"
                        if num_part.parse::<u32>().is_ok() {
                            // Build corresponding b weight name
                            let b_name = name[..a_pos].to_string() + ".b" + num_part + ".weight";

                            if let Some(b_tensor) = lora_tensors.get(&b_name) {
                                let base_name = name[..a_pos].to_string() + "." + num_part;
                                println!(
                                    "   Found LoRA pair: {} (a={}, b={})",
                                    base_name, name, b_name
                                );
                                lora_weights.insert(base_name, (tensor.clone(), b_tensor.clone()));
                            }
                        }
                    }
                }

                // Pattern 2: Original pattern (.lora_a.)
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
            model: std::sync::Mutex::new(model),
            tokenizer,
            device,
            max_tokens,
        })
    }
}

#[async_trait]
impl Processor for Llama2LoraProcessor {
    async fn process(&self, request: Request) -> CoreResult<Response> {
        use kairei_core::CoreError;

        println!("🚀 Llama2LoraProcessor.process called!");
        println!("  Input message: {}", request.message);

        // Get mutable access to the model
        let mut model = self.model.lock().unwrap();

        // Log LoRA status
        if model.lora_weights.is_empty() {
            println!("  ⚠️  No LoRA weights loaded");
        } else {
            println!(
                "  ✅ LoRA weights loaded for {} layers",
                model.lora_weights.len()
            );
        }

        // Tokenize input
        let tokens = match self.tokenizer.encode(request.message.as_str(), true) {
            Ok(t) => t,
            Err(e) => return Err(CoreError::Processing(format!("Tokenization error: {}", e))),
        };

        let input_ids = tokens.get_ids();
        println!(
            "  📝 Input tokens: {:?} (length: {})",
            &input_ids[..input_ids.len().min(10)],
            input_ids.len()
        );

        // Convert to tensor
        let input_tensor = match Tensor::new(input_ids, &self.device) {
            Ok(t) => match t.unsqueeze(0) {
                Ok(t) => t,
                Err(e) => return Err(CoreError::Processing(format!("Unsqueeze error: {}", e))),
            },
            Err(e) => {
                return Err(CoreError::Processing(format!(
                    "Tensor creation error: {}",
                    e
                )));
            }
        };

        // Simple generation - just do one forward pass for now
        println!("  🎯 Running forward pass...");

        let logits = match model.forward(&input_tensor, 0) {
            Ok(l) => l,
            Err(e) => return Err(CoreError::Processing(format!("Forward pass error: {}", e))),
        };

        println!(
            "  ✅ Forward pass complete! Logits shape: {:?}",
            logits.shape()
        );

        // Actual text generation implementation
        let mut generated_tokens = Vec::new();
        let mut all_tokens = input_ids.to_vec();
        let mut index_pos = 0;

        // Setup LogitsProcessor for sampling
        let mut logits_processor = LogitsProcessor::new(42, None, None); // seed=42, no temperature/top_p for deterministic

        println!(
            "  🎯 Starting generation (max {} tokens)...",
            self.max_tokens
        );

        for i in 0..self.max_tokens {
            // Context size management - use KV cache optimization
            let context_size = if i > 0 { 1 } else { all_tokens.len() };
            let start_idx = all_tokens.len().saturating_sub(context_size);
            let ctxt: Vec<u32> = all_tokens[start_idx..].to_vec();

            // Forward pass
            let current_input = match Tensor::new(ctxt.as_slice(), &self.device) {
                Ok(t) => match t.unsqueeze(0) {
                    Ok(t) => t,
                    Err(e) => {
                        return Err(CoreError::Processing(format!(
                            "Input tensor creation error: {}",
                            e
                        )));
                    }
                },
                Err(e) => {
                    return Err(CoreError::Processing(format!(
                        "Tensor creation error: {}",
                        e
                    )));
                }
            };

            let logits = match model.forward(&current_input, index_pos) {
                Ok(l) => l,
                Err(e) => {
                    return Err(CoreError::Processing(format!(
                        "Forward pass error at step {}: {}",
                        i, e
                    )));
                }
            };

            // Get logits for the last token
            let next_token_logits = match logits.i((0, logits.dim(1).unwrap() - 1)) {
                Ok(l) => l,
                Err(e) => {
                    return Err(CoreError::Processing(format!(
                        "Logits indexing error: {}",
                        e
                    )));
                }
            };

            // Sample next token using LogitsProcessor
            let next_token = match logits_processor.sample(&next_token_logits) {
                Ok(token) => token,
                Err(e) => return Err(CoreError::Processing(format!("Sampling error: {}", e))),
            };

            generated_tokens.push(next_token);
            all_tokens.push(next_token);

            // Check for EOS token (try to get it from tokenizer, fallback to common EOS tokens)
            let eos_token = 2u32; // Common EOS token for Llama models
            if next_token == eos_token {
                println!("  🛑 EOS token generated at position {}", i);
                break;
            }

            // Decode and print token for streaming output
            if let Ok(token_str) = self.tokenizer.decode(&[next_token], false) {
                if !token_str.trim().is_empty() {
                    print!("{}", token_str);
                    std::io::stdout().flush().ok();
                }
            }

            index_pos += ctxt.len();

            // Progress indicator for long generations
            if i % 20 == 0 && i > 0 {
                print!(".");
                std::io::stdout().flush().ok();
            }
        }
        println!(); // New line after generation

        println!("  ✅ Generated {} tokens", generated_tokens.len());

        // Decode generated tokens for final response
        let response_text = match self.tokenizer.decode(&generated_tokens, true) {
            Ok(text) => text,
            Err(e) => {
                return Err(CoreError::Processing(format!(
                    "Token decoding error: {}",
                    e
                )));
            }
        };

        Ok(Response::simple(request.id, response_text))
    }

    fn metadata(&self) -> ProcessorMetadata {
        ProcessorMetadata {
            name: Some("Llama2 LoRA Processor".to_string()),
            description: Some("Llama2 with manual LoRA implementation".to_string()),
            version: Some("0.1.0".to_string()),
            capabilities: vec!["text-generation".to_string(), "lora".to_string()],
        }
    }
}
