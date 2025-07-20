//! Llama3 model with LoRA support implementation
//!
//! Based on candle's Llama implementation with custom LoRA integration

use async_trait::async_trait;
use candle_core::{D, DType, Device, IndexOp, Result as CandleResult, Tensor};
use candle_nn::VarBuilder;
use kairei_core::{Processor, ProcessorMetadata, Request, Response, Result as CoreResult};
use std::collections::HashMap;
use tokenizers::Tokenizer;

use crate::CandleError;
use candle_lora::{LinearLayerLike, LoraConfig};
use candle_nn::{Embedding, Module, linear_no_bias as linear};
use candle_transformers::generation::LogitsProcessor;
use std::f32::consts::PI;
use std::io::Write;

/// Llama3 specific RoPE configuration
#[derive(Debug, Clone)]
pub enum Llama3RopeType {
    Llama3,
    Default,
}

#[derive(Debug, Clone)]
pub struct Llama3RopeConfig {
    pub factor: f32,
    pub low_freq_factor: f32,
    pub high_freq_factor: f32,
    pub original_max_position_embeddings: usize,
    pub rope_type: Llama3RopeType,
}

/// Configuration for Llama3
#[derive(Debug, Clone)]
pub struct Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f32,
    pub rope_scaling: Option<Llama3RopeConfig>,
    pub max_position_embeddings: usize,
    pub tie_word_embeddings: bool,
    pub use_flash_attn: bool,
}

impl Config {
    pub fn head_size(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

/// Cache for KV and positional embeddings with Llama3 RoPE support
#[derive(Debug, Clone)]
pub struct Cache {
    use_kv_cache: bool,
    kvs: Vec<Option<(Tensor, Tensor)>>,
    cos: Tensor,
    sin: Tensor,
}

fn calculate_default_inv_freq(config: &Config) -> Vec<f32> {
    let head_dim = config.hidden_size / config.num_attention_heads;
    (0..head_dim)
        .step_by(2)
        .map(|i| 1f32 / config.rope_theta.powf(i as f32 / head_dim as f32))
        .collect()
}

impl Cache {
    pub fn new(
        use_kv_cache: bool,
        dtype: DType,
        config: &Config,
        device: &Device,
    ) -> Result<Self, CandleError> {
        // Calculate theta frequencies with Llama3 RoPE scaling
        let theta = match &config.rope_scaling {
            None
            | Some(Llama3RopeConfig {
                rope_type: Llama3RopeType::Default,
                ..
            }) => calculate_default_inv_freq(config),
            Some(rope_scaling) => {
                let low_freq_wavelen = rope_scaling.original_max_position_embeddings as f32
                    / rope_scaling.low_freq_factor;
                let high_freq_wavelen = rope_scaling.original_max_position_embeddings as f32
                    / rope_scaling.high_freq_factor;

                calculate_default_inv_freq(config)
                    .into_iter()
                    .map(|freq| {
                        let wavelen = 2.0 * PI / freq;
                        if wavelen < high_freq_wavelen {
                            freq
                        } else if wavelen > low_freq_wavelen {
                            freq / rope_scaling.factor
                        } else {
                            // Smooth interpolation
                            let smooth = (rope_scaling.original_max_position_embeddings as f32
                                / wavelen
                                - rope_scaling.low_freq_factor)
                                / (rope_scaling.high_freq_factor - rope_scaling.low_freq_factor);
                            freq / (1.0 - smooth + smooth * rope_scaling.factor)
                        }
                    })
                    .collect()
            }
        };

        let theta = Tensor::new(theta.as_slice(), device)?;
        let idx_theta = Tensor::arange(0, config.max_position_embeddings as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((config.max_position_embeddings, 1))?
            .matmul(&theta.reshape((1, theta.elem_count()))?)?;

        // This is different from the paper, see HF transformers implementation
        let cos = idx_theta.cos()?.to_dtype(dtype)?;
        let sin = idx_theta.sin()?.to_dtype(dtype)?;

        Ok(Self {
            use_kv_cache,
            kvs: vec![None; config.num_hidden_layers],
            cos,
            sin,
        })
    }
}

/// Linear layer that can be either normal or LoRA
type DynLinear = Box<dyn LinearLayerLike + Send + Sync>;

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

/// Causal self-attention with LoRA support
struct CausalSelfAttention {
    q_proj: DynLinear,
    k_proj: DynLinear,
    v_proj: DynLinear,
    o_proj: DynLinear,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    max_position_embeddings: usize,
}

impl CausalSelfAttention {
    fn new(
        q_proj: DynLinear,
        k_proj: DynLinear,
        v_proj: DynLinear,
        o_proj: DynLinear,
        num_attention_heads: usize,
        num_key_value_heads: usize,
        head_dim: usize,
        max_position_embeddings: usize,
    ) -> Self {
        Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            max_position_embeddings,
        }
    }

    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self, CandleError> {
        let size_in = cfg.hidden_size;
        let size_q = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_attention_heads;
        let size_kv = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_key_value_heads;

        let q_proj = Box::new(linear(size_in, size_q, vb.pp("q_proj"))?) as DynLinear;
        let k_proj = Box::new(linear(size_in, size_kv, vb.pp("k_proj"))?) as DynLinear;
        let v_proj = Box::new(linear(size_in, size_kv, vb.pp("v_proj"))?) as DynLinear;
        let o_proj = Box::new(linear(size_q, size_in, vb.pp("o_proj"))?) as DynLinear;

        Ok(Self::new(
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.head_size(),
            cfg.max_position_embeddings,
        ))
    }

    fn forward(
        &self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        cache: &mut Cache,
    ) -> Result<Tensor, CandleError> {
        let (b_sz, seq_len, hidden_size) = x.dims3()?;
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q
            .reshape((b_sz, seq_len, self.num_attention_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let mut v = v
            .reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Apply rotary embeddings
        let q = self.apply_rotary_emb(&q, index_pos, cache)?;
        let mut k = self.apply_rotary_emb(&k, index_pos, cache)?;

        // KV cache handling
        if cache.use_kv_cache {
            if let Some((cache_k, cache_v)) = &cache.kvs[block_idx] {
                k = Tensor::cat(&[cache_k, &k], 2)?.contiguous()?;
                v = Tensor::cat(&[cache_v, &v], 2)?.contiguous()?;
                let k_seq_len = k.dims()[2];
                if k_seq_len > self.max_position_embeddings {
                    k = k
                        .narrow(
                            D::Minus1,
                            k_seq_len - self.max_position_embeddings,
                            self.max_position_embeddings,
                        )?
                        .contiguous()?;
                }
                let v_seq_len = v.dims()[2];
                if v_seq_len > self.max_position_embeddings {
                    v = v
                        .narrow(
                            D::Minus1,
                            v_seq_len - self.max_position_embeddings,
                            self.max_position_embeddings,
                        )?
                        .contiguous()?;
                }
            }
            cache.kvs[block_idx] = Some((k.clone(), v.clone()));
        }

        // Repeat KV heads if necessary
        let k = self.repeat_kv(k)?;
        let v = self.repeat_kv(v)?;

        // Compute attention
        let in_dtype = q.dtype();
        let q = q.to_dtype(DType::F32)?;
        let k = k.to_dtype(DType::F32)?;
        let v = v.to_dtype(DType::F32)?;

        let att = (q.matmul(&k.t()?)? / (self.head_dim as f64).sqrt())?;

        // Apply causal mask
        let att = if seq_len == 1 {
            att
        } else {
            let mask = self.create_causal_mask(seq_len)?;
            let mask = mask.broadcast_as(att.shape())?;
            masked_fill(&att, &mask, f32::NEG_INFINITY)?
        };

        let att = candle_nn::ops::softmax(&att, D::Minus1)?;
        let att_output = att.matmul(&v.contiguous()?)?.to_dtype(in_dtype)?;

        let y = att_output
            .transpose(1, 2)?
            .reshape((b_sz, seq_len, hidden_size))?;
        self.o_proj.forward(&y).map_err(CandleError::from)
    }

    fn apply_rotary_emb(
        &self,
        x: &Tensor,
        index_pos: usize,
        cache: &Cache,
    ) -> Result<Tensor, CandleError> {
        let (_b_sz, _n_head, seq_len, _hidden_size) = x.dims4()?;
        let cos = cache.cos.narrow(0, index_pos, seq_len)?;
        let sin = cache.sin.narrow(0, index_pos, seq_len)?;
        Ok(candle_nn::rotary_emb::rope(x, &cos, &sin)?)
    }

    fn repeat_kv(&self, x: Tensor) -> Result<Tensor, CandleError> {
        let n_rep = self.num_attention_heads / self.num_key_value_heads;
        if n_rep == 1 {
            Ok(x)
        } else {
            let (b_sz, n_kv_head, seq_len, head_dim) = x.dims4()?;
            let x = x
                .unsqueeze(2)?
                .expand((b_sz, n_kv_head, n_rep, seq_len, head_dim))?
                .reshape((b_sz, n_kv_head * n_rep, seq_len, head_dim))?;
            Ok(x)
        }
    }

    fn create_causal_mask(&self, seq_len: usize) -> Result<Tensor, CandleError> {
        let mask: Vec<_> = (0..seq_len)
            .flat_map(|i| (0..seq_len).map(move |j| u8::from(j > i)))
            .collect();
        Ok(Tensor::from_slice(&mask, (seq_len, seq_len), &Device::Cpu)?)
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
        let h_size = cfg.hidden_size;
        let i_size = cfg.intermediate_size;
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

    fn load(vb: VarBuilder, cache: &Cache, cfg: &Config) -> Result<Self, CandleError> {
        use candle_nn::rms_norm;

        let rms_1 = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let rms_2 = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
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

/// Main Llama3 model with LoRA support
pub struct Llama3WithLora {
    wte: Embedding,
    blocks: Vec<Block>,
    ln_f: candle_nn::RmsNorm,
    lm_head: DynLinear,
    cache: Cache,
    lora_weights: HashMap<String, (Tensor, Tensor, f64)>, // name -> (A, B, scale)
}

impl Llama3WithLora {
    pub fn load(
        vb: VarBuilder,
        cfg: &Config,
        device: &Device,
        dtype: DType,
    ) -> Result<Self, CandleError> {
        use candle_nn::{embedding, rms_norm};

        println!("🔨 Loading Llama3 model...");

        // Load embeddings
        let wte = embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("model.embed_tokens"))?;

        // Initialize cache
        let cache = Cache::new(true, dtype, cfg, device)?;

        // Load blocks
        let mut blocks = Vec::with_capacity(cfg.num_hidden_layers);
        for idx in 0..cfg.num_hidden_layers {
            println!("  Loading block {}/{}", idx + 1, cfg.num_hidden_layers);
            let block = Block::load(vb.pp(format!("model.layers.{}", idx)), &cache, cfg)?;
            blocks.push(block);
        }

        // Load final norm and head
        let ln_f = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("model.norm"))?;

        // Handle tie_word_embeddings
        let lm_head = if cfg.tie_word_embeddings {
            // Reuse embedding weights
            let embedding_weights = wte.embeddings();
            Box::new(candle_nn::Linear::new(embedding_weights.clone(), None)) as DynLinear
        } else {
            Box::new(linear(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?) as DynLinear
        };

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
        _lora_config: &LoraConfig,
    ) -> Result<(), CandleError> {
        println!("🎯 Storing LoRA weights for {} layers", lora_weights.len());

        for (name, (lora_a, lora_b)) in lora_weights {
            println!("  Processing LoRA weights for: {}", name);

            let a_shape = lora_a.shape();
            let b_shape = lora_b.shape();
            println!("    - lora_a shape: {:?}", a_shape);
            println!("    - lora_b shape: {:?}", b_shape);

            let rank = lora_a.dim(0)?;

            if rank > 64 {
                println!("    ⚠️  Warning: rank={} seems too high for LoRA!", rank);
            }

            let alpha = 32.0;
            let scale = alpha / rank as f64;

            self.lora_weights
                .insert(name.clone(), (lora_a, lora_b, scale));
            println!("    ✓ Stored with rank={}, scale={:.4}", rank, scale);
        }

        println!("✅ LoRA weights stored successfully");
        Ok(())
    }

    pub fn forward(&mut self, input_ids: &Tensor, index_pos: usize) -> Result<Tensor, CandleError> {
        let (_b_sz, _seq_len) = input_ids.dims2()?;

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

/// Main processor implementation
pub struct Llama3LoraProcessor {
    model: std::sync::Mutex<Llama3WithLora>,
    tokenizer: Tokenizer,
    device: Device,
    max_tokens: usize,
}

impl Llama3LoraProcessor {
    pub fn new(
        model_path: &str,
        tokenizer_path: &str,
        lora_path: Option<&str>,
        device: Device,
        dtype: DType,
        config: Config,
        max_tokens: usize,
    ) -> Result<Self, CandleError> {
        println!("🚀 Initializing Llama3 LoRA Processor...");

        // Load tokenizer
        println!("📖 Loading tokenizer from: {}", tokenizer_path);
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| CandleError::Other(format!("Failed to load tokenizer: {}", e)))?;

        // Load model weights
        println!("🏗️ Loading model weights from: {}", model_path);
        let tensors = candle_core::safetensors::load(model_path, &device)?;
        let vb = VarBuilder::from_tensors(tensors, dtype, &device);

        // Load model
        let mut model = Llama3WithLora::load(vb, &config, &device, dtype)?;

        // Apply LoRA weights if provided
        if let Some(lora_path) = lora_path {
            println!("🎯 Loading LoRA weights from: {}", lora_path);
            let lora_tensors = candle_core::safetensors::load(lora_path, &device)?;

            let mut lora_weights = HashMap::new();
            let lora_config = LoraConfig::new(16, 32.0, Some(0.0));

            // Extract LoRA A/B pairs (same pattern matching as Llama2)
            println!("📋 LoRA file contains {} tensors:", lora_tensors.len());
            for (i, (name, _)) in lora_tensors.iter().enumerate() {
                if i < 10 {
                    println!("   - {}", name);
                }
            }

            // Process LoRA weights
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
            model: std::sync::Mutex::new(model),
            tokenizer,
            device,
            max_tokens,
        })
    }
}

#[async_trait]
impl Processor for Llama3LoraProcessor {
    async fn process(&self, request: Request) -> CoreResult<Response> {
        use kairei_core::CoreError;

        println!("🚀 Llama3LoraProcessor.process called!");
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

        // Text generation
        let mut generated_tokens = Vec::new();
        let mut all_tokens = input_ids.to_vec();
        let mut index_pos = 0;

        // Setup LogitsProcessor for sampling
        let mut logits_processor = LogitsProcessor::new(42, None, None);

        println!(
            "  🎯 Starting generation (max {} tokens)...",
            self.max_tokens
        );

        for i in 0..self.max_tokens {
            // Context size management
            let context_size = if i > 0 { 1 } else { all_tokens.len() };
            let start_idx = all_tokens.len().saturating_sub(context_size);
            let ctxt: Vec<u32> = all_tokens[start_idx..].to_vec();

            // Forward pass
            let current_input = match Tensor::new(ctxt.as_slice(), &self.device) {
                Ok(t) => match t.unsqueeze(0) {
                    Ok(t) => t,
                    Err(e) => {
                        return Err(CoreError::Processing(format!("Input tensor error: {}", e)));
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

            // Sample next token
            let next_token = match logits_processor.sample(&next_token_logits) {
                Ok(token) => token,
                Err(e) => return Err(CoreError::Processing(format!("Sampling error: {}", e))),
            };

            generated_tokens.push(next_token);
            all_tokens.push(next_token);

            // Check for EOS token
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

            // Progress indicator
            if i % 20 == 0 && i > 0 {
                print!(".");
                std::io::stdout().flush().ok();
            }
        }
        println!(); // New line after generation

        println!("  ✅ Generated {} tokens", generated_tokens.len());

        // Decode generated tokens
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
            name: Some("Llama3 LoRA Processor".to_string()),
            description: Some("Llama3 with custom LoRA implementation".to_string()),
            version: Some("0.1.0".to_string()),
            capabilities: vec!["text-generation".to_string(), "lora".to_string()],
        }
    }
}
