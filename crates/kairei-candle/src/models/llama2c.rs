//! Llama2 inference implementation.
//!
//! See ["LLaMA 2: Open Foundation and Fine-Tuned Chat Models"](https://arxiv.org/abs/2307.09288)
//!
//! - ⚡ [Interactive Wasm Example](https://huggingface.co/spaces/lmz/candle-llama2)
//! - 💻 llama2.c [GH Link](https://github.com/karpathy/llama2.c)
//!

// Re-export TransformerWeights and Config from candle-transformers for compatibility
pub use candle_transformers::models::llama2_c::Config;
pub use candle_transformers::models::llama2_c_weights::TransformerWeights;

use candle_core::{D, DType, Device, IndexOp, Result, Tensor};
use candle_nn::linear_no_bias as linear;
use candle_nn::{Embedding, Module, RmsNorm, VarBuilder, embedding, rms_norm};
use std::collections::HashMap;

use crate::models::lora::LoraLinear;

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
    pub fn new(use_kv_cache: bool, cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let n_elem = cfg.dim / cfg.n_heads;
        let theta: Vec<_> = (0..n_elem)
            .step_by(2)
            .map(|i| 1f32 / 10000f32.powf(i as f32 / n_elem as f32))
            .collect();
        let theta = Tensor::new(theta.as_slice(), vb.device())?;
        let idx_theta = Tensor::arange(0, cfg.seq_len as u32, vb.device())?
            .to_dtype(DType::F32)?
            .reshape((cfg.seq_len, 1))?
            .matmul(&theta.reshape((1, theta.elem_count()))?)?;
        let precomputed_cos = idx_theta.cos()?;
        let precomputed_sin = idx_theta.sin()?;

        let freq_cis_real = vb
            .get((cfg.seq_len, cfg.head_size() / 2), "freq_cis_real")
            .unwrap_or(precomputed_cos);
        let freq_cis_imag = vb
            .get((cfg.seq_len, cfg.head_size() / 2), "freq_cis_imag")
            .unwrap_or(precomputed_sin);
        let cos = freq_cis_real.reshape((cfg.seq_len, cfg.head_size() / 2, 1))?;
        let sin = freq_cis_imag.reshape((cfg.seq_len, cfg.head_size() / 2, 1))?;
        Ok(Self {
            masks: HashMap::new(),
            use_kv_cache,
            kvs: vec![None; cfg.n_layers],
            cos,
            sin,
            device: vb.device().clone(),
        })
    }

    pub fn mask(&mut self, t: usize) -> Result<Tensor> {
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

fn silu(xs: &Tensor) -> Result<Tensor> {
    xs / (xs.neg()?.exp()? + 1.0)?
}

#[derive(Debug, Clone)]
struct CausalSelfAttention {
    q_proj: LoraLinear,
    k_proj: LoraLinear,
    v_proj: LoraLinear,
    o_proj: LoraLinear,
    n_head: usize,
    n_key_value_head: usize,
    head_dim: usize,
}

impl CausalSelfAttention {
    fn apply_rotary_emb(&self, x: &Tensor, index_pos: usize, cache: &Cache) -> Result<Tensor> {
        let (b_sz, seq_len, h, n_embd) = x.dims4()?;
        let cos = cache.cos.i(index_pos..index_pos + seq_len)?;
        let sin = cache.sin.i(index_pos..index_pos + seq_len)?;
        let cos = cos.unsqueeze(1)?;
        let sin = sin.unsqueeze(1)?;
        let cos = cos.broadcast_as((b_sz, seq_len, 1, n_embd / 2, 1))?;
        let sin = sin.broadcast_as((b_sz, seq_len, 1, n_embd / 2, 1))?;
        let x = x.reshape((b_sz, seq_len, h, n_embd / 2, 2))?;
        let x0 = x.narrow(D::Minus1, 0, 1)?;
        let x1 = x.narrow(D::Minus1, 1, 1)?;
        let dst0 = (x0.broadcast_mul(&cos)? - x1.broadcast_mul(&sin)?)?;
        let dst1 = (x0.broadcast_mul(&sin)? + x1.broadcast_mul(&cos)?)?;
        let rope = Tensor::cat(&[&dst0, &dst1], D::Minus1)?.reshape((b_sz, seq_len, h, n_embd))?;
        Ok(rope)
    }

    fn forward(
        &self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        cache: &mut Cache,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, n_embd) = x.dims3()?;
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q.reshape((b_sz, seq_len, self.n_head, self.head_dim))?;
        let k = k.reshape((b_sz, seq_len, self.n_key_value_head, self.head_dim))?;
        let mut v = v.reshape((b_sz, seq_len, self.n_key_value_head, self.head_dim))?;

        let q = self.apply_rotary_emb(&q, index_pos, cache)?;
        let mut k = self.apply_rotary_emb(&k, index_pos, cache)?;

        if cache.use_kv_cache {
            if let Some((cache_k, cache_v)) = &cache.kvs[block_idx] {
                k = Tensor::cat(&[cache_k, &k], 1)?.contiguous()?;
                v = Tensor::cat(&[cache_v, &v], 1)?.contiguous()?;
            }
            cache.kvs[block_idx] = Some((k.clone(), v.clone()))
        }

        let k = self.repeat_kv(k)?;
        let v = self.repeat_kv(v)?;

        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;

        let att = (q.matmul(&k.t()?)? / (self.head_dim as f64).sqrt())?;
        let att = if seq_len <= 1 {
            att
        } else {
            let mask = cache.mask(seq_len)?.broadcast_as(att.shape())?;
            masked_fill(&att, &mask, f32::NEG_INFINITY)?
        };
        let att = candle_nn::ops::softmax(&att, D::Minus1)?;
        // Convert to contiguous as matmul doesn't support strided vs for now.
        let y = att.matmul(&v.contiguous()?)?;
        let y = y.transpose(1, 2)?.reshape(&[b_sz, seq_len, n_embd])?;
        let y = self.o_proj.forward(&y)?;
        Ok(y)
    }

    fn repeat_kv(&self, x: Tensor) -> Result<Tensor> {
        let n_rep = self.n_head / self.n_key_value_head;
        if n_rep == 1 {
            Ok(x)
        } else {
            let (b_sz, seq_len, n_kv_head, head_dim) = x.dims4()?;
            let x = x
                .unsqueeze(3)?
                .expand((b_sz, seq_len, n_kv_head, n_rep, head_dim))?
                .reshape((b_sz, seq_len, n_kv_head * n_rep, head_dim))?;
            Ok(x)
        }
    }

    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let size_in = cfg.dim;
        let size_q = (cfg.dim / cfg.n_heads) * cfg.n_heads;
        let size_kv = (cfg.dim / cfg.n_heads) * cfg.n_kv_heads;
        let q_proj = LoraLinear::from_linear(linear(size_in, size_q, vb.pp("q_proj"))?);
        let k_proj = LoraLinear::from_linear(linear(size_in, size_kv, vb.pp("k_proj"))?);
        let v_proj = LoraLinear::from_linear(linear(size_in, size_kv, vb.pp("v_proj"))?);
        let o_proj = LoraLinear::from_linear(linear(size_q, size_in, vb.pp("o_proj"))?);
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            n_head: cfg.n_heads,
            n_key_value_head: cfg.n_kv_heads,
            head_dim: cfg.dim / cfg.n_heads,
        })
    }
}

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: f32) -> Result<Tensor> {
    let shape = mask.shape();
    let on_true = Tensor::new(on_true, on_false.device())?.broadcast_as(shape.dims())?;
    let m = mask.where_cond(&on_true, on_false)?;
    Ok(m)
}

#[derive(Debug, Clone)]
struct Mlp {
    c_fc1: LoraLinear,
    c_fc2: LoraLinear,
    c_proj: LoraLinear,
}

impl Mlp {
    fn new(c_fc1: LoraLinear, c_fc2: LoraLinear, c_proj: LoraLinear) -> Self {
        Self {
            c_fc1,
            c_fc2,
            c_proj,
        }
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = (silu(&self.c_fc1.forward(x)?)? * self.c_fc2.forward(x)?)?;
        self.c_proj.forward(&x)
    }

    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let h_size = cfg.dim;
        let i_size = cfg.hidden_dim;
        let c_fc1 = LoraLinear::from_linear(linear(h_size, i_size, vb.pp("gate_proj"))?);
        let c_fc2 = LoraLinear::from_linear(linear(h_size, i_size, vb.pp("up_proj"))?);
        let c_proj = LoraLinear::from_linear(linear(i_size, h_size, vb.pp("down_proj"))?);
        Ok(Self::new(c_fc1, c_fc2, c_proj))
    }
}

#[derive(Debug, Clone)]
struct Block {
    rms_1: RmsNorm,
    attn: CausalSelfAttention,
    rms_2: RmsNorm,
    mlp: Mlp,
}

impl Block {
    fn new(rms_1: RmsNorm, attn: CausalSelfAttention, rms_2: RmsNorm, mlp: Mlp) -> Self {
        Self {
            rms_1,
            attn,
            rms_2,
            mlp,
        }
    }

    fn forward(
        &self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        cache: &mut Cache,
    ) -> Result<Tensor> {
        let residual = x;
        let x = self.rms_1.forward(x)?;
        let x = (self.attn.forward(&x, index_pos, block_idx, cache)? + residual)?;
        let residual = &x;
        let x = (self.mlp.forward(&self.rms_2.forward(&x)?)? + residual)?;
        Ok(x)
    }

    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let attn = CausalSelfAttention::load(vb.pp("self_attn"), cfg)?;
        let mlp = Mlp::load(vb.pp("mlp"), cfg)?;
        let input_layernorm = rms_norm(cfg.dim, cfg.norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm =
            rms_norm(cfg.dim, cfg.norm_eps, vb.pp("post_attention_layernorm"))?;
        Ok(Self::new(
            input_layernorm,
            attn,
            post_attention_layernorm,
            mlp,
        ))
    }
}

#[derive(Debug, Clone)]
pub struct Llama {
    wte: Embedding,
    blocks: Vec<Block>,
    ln_f: RmsNorm,
    lm_head: LoraLinear,
    #[allow(dead_code)]
    pub config: Config,
}

impl Llama {
    pub fn forward(&self, x: &Tensor, index_pos: usize, cache: &mut Cache) -> Result<Tensor> {
        println!("🔥 Llama.forward called! index_pos: {}", index_pos);
        let (_b_sz, _seq_len) = x.dims2()?;
        println!("  Input shape: {:?}", x.shape());

        let mut x = self.wte.forward(x)?;
        println!("  After embedding: {:?}", x.shape());

        for (block_idx, block) in self.blocks.iter().enumerate() {
            x = block.forward(&x, index_pos, block_idx, cache)?;
            if block_idx == 0 {
                println!("  After first block: {:?}", x.shape());
            }
        }

        let x = self.ln_f.forward(&x)?;
        println!("  After final norm: {:?}", x.shape());

        let logits = self.lm_head.forward(&x)?;
        println!("  Output logits shape: {:?}", logits.shape());

        logits.to_dtype(DType::F32)
    }

    pub fn load(vb: VarBuilder, cfg: Config) -> Result<Self> {
        let wte = embedding(cfg.vocab_size, cfg.dim, vb.pp("model.embed_tokens"))?;
        let lm_head = LoraLinear::from_linear(linear(cfg.dim, cfg.vocab_size, vb.pp("lm_head"))?);
        let ln_f = rms_norm(cfg.dim, cfg.norm_eps, vb.pp("model.norm"))?;
        let blocks: Vec<_> = (0..cfg.n_layers)
            .map(|i| Block::load(vb.pp(format!("model.layers.{i}")), &cfg).unwrap())
            .collect();
        Ok(Self {
            wte,
            blocks,
            ln_f,
            lm_head,
            config: cfg,
        })
    }

    pub fn apply_lora(&mut self, lora_manager: &crate::models::lora::LoraManager) -> Result<()> {
        println!("🚀 Applying LoRA to Llama model...");

        // Apply LoRA to lm_head
        self.lm_head = lora_manager.apply_to_linear(self.lm_head.base.clone(), "lm_head");

        // Apply LoRA to each block
        for (block_idx, block) in self.blocks.iter_mut().enumerate() {
            println!("  📦 Processing block {}", block_idx);

            // Apply LoRA to attention layers
            let attn = &mut block.attn;
            attn.q_proj = lora_manager.apply_to_linear(
                attn.q_proj.base.clone(),
                &format!("model.layers.{}.self_attn.q_proj", block_idx),
            );
            attn.k_proj = lora_manager.apply_to_linear(
                attn.k_proj.base.clone(),
                &format!("model.layers.{}.self_attn.k_proj", block_idx),
            );
            attn.v_proj = lora_manager.apply_to_linear(
                attn.v_proj.base.clone(),
                &format!("model.layers.{}.self_attn.v_proj", block_idx),
            );
            attn.o_proj = lora_manager.apply_to_linear(
                attn.o_proj.base.clone(),
                &format!("model.layers.{}.self_attn.o_proj", block_idx),
            );

            // Apply LoRA to MLP layers
            let mlp = &mut block.mlp;
            mlp.c_fc1 = lora_manager.apply_to_linear(
                mlp.c_fc1.base.clone(),
                &format!("model.layers.{}.mlp.gate_proj", block_idx),
            );
            mlp.c_fc2 = lora_manager.apply_to_linear(
                mlp.c_fc2.base.clone(),
                &format!("model.layers.{}.mlp.up_proj", block_idx),
            );
            mlp.c_proj = lora_manager.apply_to_linear(
                mlp.c_proj.base.clone(),
                &format!("model.layers.{}.mlp.down_proj", block_idx),
            );
        }

        println!("✅ LoRA application complete!");
        Ok(())
    }
}
