//! Gemma 4 dense (`gemma4` architecture) forward pass on candle CPU tensors,
//! loaded from a GGUF file. The burn GPU engine (`gpu`, `provider-local-gpu`)
//! reproduces it on a device and shares the host-side pieces here
//! ([`Gemma4Params`], [`RowTable`], RoPE tables, masks); [`Model`] is what
//! the engine holds.
//!
//! Every dimension comes from the GGUF metadata and tensor shapes (layer
//! count, widths, head counts and dimensions, per-layer FFN width, vocabulary,
//! per-layer-embedding width): the loader holds no checkpoint constants, so
//! the E2B and E4B checkpoints load through the same code, and an unsupported
//! one (MoE, per-layer KV head counts, a tensor dtype candle cannot read) is
//! rejected with an error naming the cause. The numbers quoted below are
//! those of the E2B checkpoint, the reference.
//!
//! The architecture differs from earlier Gemma generations in these ways, all
//! implemented in [`Gemma4::forward`]:
//!
//! - RMSNorm multiplies by the weight directly (no `1 + w`), attention scale
//!   is 1.0, and there is no attention soft-cap; the value projection gets a
//!   weightless RMSNorm.
//! - Layers follow `sliding_window_pattern`: sliding-window (E2B: window 512,
//!   head dim 256, RoPE base 1e4) or full attention (head dim 512, RoPE base
//!   1e6 with the `rope_freqs.weight` per-pair factors that leave 192 of 256
//!   pairs unrotated). `head_count_kv` KV heads serve the `head_count` query
//!   heads in equal groups (grouped-query attention): E2B has one KV head for
//!   8 query heads, E4B two.
//! - The last `shared_kv_layers` layers own no K/V projections: they attend
//!   to the cache of the last earlier layer of their own attention type.
//! - Per-layer embeddings: every token also selects a row of
//!   `per_layer_token_embd` (E2B: 8960 wide), split into one
//!   `embedding_length_per_layer_input`-wide slice per layer and gated into
//!   that layer's residual stream.
//! - Final logits are soft-capped: `30 * tanh(x / 30)`.
//!
//! # Memory
//!
//! - Weights stay quantized (any dtype candle's quantized matmul reads) and
//!   are consumed by candle's quantized matmul; norm vectors and
//!   `per_layer_model_proj` (BF16 in the published files) are held as f32.
//! - The token embedding (E2B Q4_0 file: Q8_0, 427 MB) is held once and shared
//!   with the tied LM head.
//! - `per_layer_token_embd` (E2B Q4_0 file: Q4_0, 1.32 GB) is never loaded:
//!   each token's row (5,040 bytes there) is read from the GGUF file with a
//!   seek and dequantized by [`dequant_row`], so a forward pass costs one
//!   small disk read (usually an OS page-cache hit) per input token.
//! - The KV cache is preallocated for `context_length` positions: only the
//!   layers that own K/V store them, `context_length * 2 * 4 *
//!   sum(n_kv_head * head_dim)` bytes in f32, the sum running over those
//!   layers with `head_dim` of that layer's attention type. For E2B that is
//!   15 layers (12 sliding with head dim 256, 3 full with head dim 512, one KV
//!   head): `context_length * 36,864` bytes (302 MB at 8192). Sliding layers
//!   keep every position, not a ring, so a sliding layer reads a
//!   `window`-long slice at the end of its cache.
//! - RoPE cos/sin tables are built once for `context_length` positions.
//! - A prefill call processes at most [`prefill_chunk`] tokens, which bounds
//!   the transient `n_head * chunk * context_length` f32 score matrix of a
//!   full-attention layer to `n_head * 8 MB` (64 MB at 8 query heads).

use std::collections::HashMap;
use std::f64::consts::FRAC_1_SQRT_2;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::Arc;

use candle_core::quantized::{GgmlDType, QMatMul, QTensor, gguf_file};
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::ops::{rms_norm, softmax_last_dim};
use candle_nn::rotary_emb::rope;

use crate::error::CucaError;
use crate::types::ProviderEndpoint;

#[cfg(feature = "provider-local-gpu")]
pub(crate) mod gpu;

/// GGUF tensor names: the one table of the Gemma 4 naming scheme. Per-layer
/// names are `blk.{layer}.{part}.weight`.
mod name {
    pub(super) const TOKEN_EMBD: &str = "token_embd.weight";
    pub(super) const PER_LAYER_TOKEN_EMBD: &str = "per_layer_token_embd.weight";
    pub(super) const PER_LAYER_MODEL_PROJ: &str = "per_layer_model_proj.weight";
    pub(super) const PER_LAYER_PROJ_NORM: &str = "per_layer_proj_norm.weight";
    pub(super) const OUTPUT_NORM: &str = "output_norm.weight";
    pub(super) const ROPE_FREQS: &str = "rope_freqs.weight";

    pub(super) const ATTN_NORM: &str = "attn_norm";
    pub(super) const ATTN_Q: &str = "attn_q";
    pub(super) const ATTN_Q_NORM: &str = "attn_q_norm";
    pub(super) const ATTN_K: &str = "attn_k";
    pub(super) const ATTN_K_NORM: &str = "attn_k_norm";
    pub(super) const ATTN_V: &str = "attn_v";
    pub(super) const ATTN_OUTPUT: &str = "attn_output";
    pub(super) const POST_ATTENTION_NORM: &str = "post_attention_norm";
    pub(super) const FFN_NORM: &str = "ffn_norm";
    pub(super) const FFN_GATE: &str = "ffn_gate";
    pub(super) const FFN_UP: &str = "ffn_up";
    pub(super) const FFN_DOWN: &str = "ffn_down";
    pub(super) const POST_FFW_NORM: &str = "post_ffw_norm";
    pub(super) const INP_GATE: &str = "inp_gate";
    pub(super) const PROJ: &str = "proj";
    pub(super) const POST_NORM: &str = "post_norm";
    pub(super) const LAYER_OUTPUT_SCALE: &str = "layer_output_scale";

    /// Full tensor name of `part` in block `layer`.
    pub(super) fn blk(layer: usize, part: &str) -> String {
        format!("blk.{layer}.{part}.weight")
    }
}

fn err(message: impl Into<String>) -> CucaError {
    CucaError::provider(ProviderEndpoint::Local, message)
}

fn candle_err(e: candle_core::Error) -> CucaError {
    err(e.to_string())
}

fn meta_err(key: &str, e: impl std::fmt::Display) -> CucaError {
    err(format!("GGUF metadata `{key}`: {e}"))
}

fn md_get<'a>(
    md: &'a HashMap<String, gguf_file::Value>,
    key: &str,
) -> Result<&'a gguf_file::Value, CucaError> {
    md.get(key).ok_or_else(|| meta_err(key, "missing"))
}

fn md_usize(md: &HashMap<String, gguf_file::Value>, key: &str) -> Result<usize, CucaError> {
    let v = md_get(md, key)?.to_u32().map_err(|e| meta_err(key, e))?;
    usize::try_from(v).map_err(|e| meta_err(key, e))
}

fn md_f32(md: &HashMap<String, gguf_file::Value>, key: &str) -> Result<f32, CucaError> {
    md_get(md, key)?.to_f32().map_err(|e| meta_err(key, e))
}

/// The KV head count: `head_count_kv` is a scalar in the published files; an
/// array is accepted when every layer agrees.
fn md_kv_heads(md: &HashMap<String, gguf_file::Value>) -> Result<usize, CucaError> {
    const KEY: &str = "gemma4.attention.head_count_kv";
    let value = md_get(md, KEY)?;
    let Ok(items) = value.to_vec() else {
        return md_usize(md, KEY);
    };
    let counts = items
        .iter()
        .map(|v| v.to_u32().map_err(|e| meta_err(KEY, e)))
        .collect::<Result<Vec<u32>, CucaError>>()?;
    match counts.split_first() {
        Some((&first, rest)) if rest.iter().all(|&c| c == first) => {
            usize::try_from(first).map_err(|e| meta_err(KEY, e))
        }
        _ => Err(err(
            "GGUF metadata `gemma4.attention.head_count_kv`: per-layer KV head counts are not supported",
        )),
    }
}

/// Reject a value length that differs from the key length: the cache and the
/// attention code use one head dimension for both.
fn require_equal_value_length(
    md: &HashMap<String, gguf_file::Value>,
    key: &str,
    key_length: usize,
) -> Result<(), CucaError> {
    match md.get(key) {
        Some(_) if md_usize(md, key)? != key_length => Err(err(format!(
            "GGUF metadata `{key}`: differs from the key length {key_length}; only equal key and value lengths are supported"
        ))),
        _ => Ok(()),
    }
}

/// Hyperparameters read from the `gemma4.*` GGUF metadata.
#[derive(Debug, Clone)]
pub(crate) struct Gemma4Params {
    /// Transformer blocks.
    pub(crate) n_layer: usize,
    /// Residual stream width.
    pub(crate) n_embd: usize,
    /// Query heads, served by `n_kv_head` KV heads in equal groups.
    pub(crate) n_head: usize,
    /// KV heads per K/V-owning layer (grouped-query attention).
    pub(crate) n_kv_head: usize,
    /// Head dimension of full-attention layers.
    pub(crate) head_dim_full: usize,
    /// Head dimension of sliding-window layers.
    pub(crate) head_dim_swa: usize,
    /// Sliding-window length in tokens (the query itself included).
    pub(crate) window: usize,
    /// Per layer: `true` for sliding-window attention.
    pub(crate) is_sliding: Vec<bool>,
    /// Layers `0..n_layer_kv` own K/V projections and a cache.
    pub(crate) n_layer_kv: usize,
    /// Per layer: the layer whose cache it reads (itself when it owns K/V).
    pub(crate) kv_source: Vec<usize>,
    /// Width of one layer's per-layer-embedding slice.
    pub(crate) ple_dim: usize,
    /// RoPE base of full-attention layers.
    pub(crate) rope_base: f64,
    /// RoPE base of sliding-window layers.
    pub(crate) rope_base_swa: f64,
    /// RMSNorm epsilon.
    pub(crate) eps: f64,
    /// Final logit soft-cap.
    pub(crate) softcap: f64,
    /// Maximum context length the checkpoint was trained for.
    pub(crate) model_ctx: usize,
}

impl Gemma4Params {
    /// Read and validate the hyperparameters; rejects checkpoints this
    /// implementation does not cover (MoE, per-layer KV head counts,
    /// differing key and value lengths).
    pub(crate) fn from_metadata(md: &HashMap<String, gguf_file::Value>) -> Result<Self, CucaError> {
        if let Some(experts) = md.get("gemma4.expert_count")
            && !matches!(experts.to_u32(), Ok(0))
        {
            return Err(err("Gemma 4 MoE checkpoints are not supported"));
        }

        let n_layer = md_usize(md, "gemma4.block_count")?;
        let model_ctx = md_usize(md, "gemma4.context_length")?;
        let n_embd = md_usize(md, "gemma4.embedding_length")?;
        let n_head = md_usize(md, "gemma4.attention.head_count")?;
        let n_kv_head = md_kv_heads(md)?;
        let head_dim_full = md_usize(md, "gemma4.attention.key_length")?;
        let head_dim_swa = md_usize(md, "gemma4.attention.key_length_swa")?;
        let window = md_usize(md, "gemma4.attention.sliding_window")?;
        let shared_kv = md_usize(md, "gemma4.attention.shared_kv_layers")?;
        let ple_dim = md_usize(md, "gemma4.embedding_length_per_layer_input")?;
        let rope_base = f64::from(md_f32(md, "gemma4.rope.freq_base")?);
        let rope_base_swa = f64::from(md_f32(md, "gemma4.rope.freq_base_swa")?);
        let eps = f64::from(md_f32(md, "gemma4.attention.layer_norm_rms_epsilon")?);
        let softcap = f64::from(md_f32(md, "gemma4.final_logit_softcapping")?);

        const PATTERN: &str = "gemma4.attention.sliding_window_pattern";
        let is_sliding = md_get(md, PATTERN)?
            .to_vec()
            .map_err(|e| meta_err(PATTERN, e))?
            .iter()
            .map(|v| v.to_bool().map_err(|e| meta_err(PATTERN, e)))
            .collect::<Result<Vec<bool>, CucaError>>()?;

        if n_kv_head == 0 || !n_head.is_multiple_of(n_kv_head) {
            return Err(err(format!(
                "GGUF metadata `gemma4.attention.head_count_kv`: {n_kv_head} KV heads do not divide {n_head} query heads"
            )));
        }
        require_equal_value_length(md, "gemma4.attention.value_length", head_dim_full)?;
        require_equal_value_length(md, "gemma4.attention.value_length_swa", head_dim_swa)?;
        if n_layer == 0 || n_embd == 0 || n_head == 0 || ple_dim == 0 || model_ctx == 0 {
            return Err(err(
                "GGUF metadata: layer, embedding, head, per-layer-input and context sizes must be non-zero",
            ));
        }
        if head_dim_full == 0
            || head_dim_swa == 0
            || !head_dim_full.is_multiple_of(2)
            || !head_dim_swa.is_multiple_of(2)
        {
            return Err(err(
                "GGUF metadata: attention key lengths must be even and non-zero",
            ));
        }
        if window == 0 {
            return Err(err("GGUF metadata: sliding window must be non-zero"));
        }
        if is_sliding.len() != n_layer {
            return Err(meta_err(
                PATTERN,
                format!("expected {n_layer} entries, found {}", is_sliding.len()),
            ));
        }
        if !softcap.is_finite() || softcap <= 0.0 {
            return Err(err("GGUF metadata: final logit soft-cap must be positive"));
        }
        let n_layer_kv = n_layer
            .checked_sub(shared_kv)
            .filter(|&n| n > 0)
            .ok_or_else(|| {
                meta_err(
                    "gemma4.attention.shared_kv_layers",
                    format!("{shared_kv} shared layers leave no K/V owner among {n_layer} layers"),
                )
            })?;
        let kv_source = kv_sources(&is_sliding, n_layer_kv)?;

        Ok(Self {
            n_layer,
            n_embd,
            n_head,
            n_kv_head,
            head_dim_full,
            head_dim_swa,
            window,
            is_sliding,
            n_layer_kv,
            kv_source,
            ple_dim,
            rope_base,
            rope_base_swa,
            eps,
            softcap,
            model_ctx,
        })
    }
}

/// For each layer, the layer whose K/V cache it reads: itself below
/// `n_layer_kv`, otherwise the last layer below `n_layer_kv` with the same
/// attention type (sliding or full).
pub(crate) fn kv_sources(is_sliding: &[bool], n_layer_kv: usize) -> Result<Vec<usize>, CucaError> {
    if n_layer_kv == 0 || n_layer_kv > is_sliding.len() {
        return Err(err(format!(
            "{n_layer_kv} K/V-owning layers do not fit {} layers",
            is_sliding.len()
        )));
    }
    (0..is_sliding.len())
        .map(|il| {
            if il < n_layer_kv {
                return Ok(il);
            }
            (0..n_layer_kv)
                .rev()
                .find(|&j| is_sliding[j] == is_sliding[il])
                .ok_or_else(|| {
                    let kind = if is_sliding[il] { "sliding" } else { "full" };
                    err(format!(
                        "layer {il} shares K/V but no earlier {kind}-attention layer owns any"
                    ))
                })
        })
        .collect()
}

/// Additive attention mask for `t` queries at positions `start..start + t`
/// against the cached keys `lo..hi`, row-major `t x (hi - lo)`.
///
/// An entry is `0.0` when key position `p` satisfies `p <= query` and, for a
/// sliding layer (`window = Some(w)`), `query - p < w`; `NEG_INFINITY`
/// elsewhere.
pub(crate) fn attention_mask(
    start: usize,
    t: usize,
    lo: usize,
    hi: usize,
    window: Option<usize>,
) -> Vec<f32> {
    let l = hi - lo;
    let mut mask = vec![f32::NEG_INFINITY; t * l];
    for (i, row) in mask.chunks_exact_mut(l.max(1)).enumerate().take(t) {
        let query = start + i;
        let first = match window {
            Some(w) => lo.max((query + 1).saturating_sub(w)),
            None => lo,
        };
        let last = query.min(hi.saturating_sub(1));
        for p in first..=last {
            row[p - lo] = 0.0;
        }
    }
    mask
}

/// Most tokens one prefill call processes: bounds the transient score matrix
/// (`n_head * chunk * context_length` f32 per full-attention layer).
pub(crate) fn prefill_chunk(context_length: usize) -> usize {
    (2_097_152 / context_length.max(1)).clamp(16, 256)
}

/// Dequantize one row of a tensor into `out`, multiplying every element by
/// `scale`.
///
/// F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1 and Q8_0 are decoded in place, with
/// no allocation. Every other dtype candle reads (the K-quants and Q8_1) goes
/// through candle's own dequantizer, which allocates per call.
///
/// `bytes` holds the row's blocks and must be exactly `out.len() /
/// block_size` blocks of `type_size` bytes (for example 18 bytes per 32
/// elements for Q4_0, 34 for Q8_0, 144 per 256 for Q4_K).
pub(crate) fn dequant_row(
    dtype: GgmlDType,
    bytes: &[u8],
    scale: f32,
    out: &mut [f32],
) -> Result<(), CucaError> {
    let block = dtype.block_size();
    if !out.len().is_multiple_of(block) || bytes.len() != out.len() / block * dtype.type_size() {
        return Err(err(format!(
            "row of {} bytes does not hold {} {dtype:?} elements",
            bytes.len(),
            out.len()
        )));
    }
    let f16_at = |b: &[u8], at: usize| {
        half::f16::from_bits(u16::from_le_bytes([b[at], b[at + 1]])).to_f32() * scale
    };
    match dtype {
        GgmlDType::F32 => {
            let (words, _) = bytes.as_chunks::<4>();
            for (o, w) in out.iter_mut().zip(words) {
                *o = f32::from_le_bytes(*w) * scale;
            }
        }
        GgmlDType::F16 => {
            let (halves, _) = bytes.as_chunks::<2>();
            for (o, h) in out.iter_mut().zip(halves) {
                *o = half::f16::from_bits(u16::from_le_bytes(*h)).to_f32() * scale;
            }
        }
        GgmlDType::BF16 => {
            let (halves, _) = bytes.as_chunks::<2>();
            for (o, h) in out.iter_mut().zip(halves) {
                *o = half::bf16::from_bits(u16::from_le_bytes(*h)).to_f32() * scale;
            }
        }
        GgmlDType::Q4_0 | GgmlDType::Q4_1 | GgmlDType::Q5_0 | GgmlDType::Q5_1 | GgmlDType::Q8_0 => {
            let (out_blocks, _) = out.as_chunks_mut::<32>();
            for (b, o) in bytes.chunks_exact(dtype.type_size()).zip(out_blocks) {
                let d = f16_at(b, 0);
                match dtype {
                    // d, 16 bytes of nibbles: low nibbles are elements 0..16,
                    // high nibbles 16..32, each minus 8.
                    GgmlDType::Q4_0 => {
                        for (j, &q) in b[2..18].iter().enumerate() {
                            o[j] = f32::from(i16::from(q & 0x0F) - 8) * d;
                            o[j + 16] = f32::from(i16::from(q >> 4) - 8) * d;
                        }
                    }
                    // d, m, nibbles: element = nibble * d + m.
                    GgmlDType::Q4_1 => {
                        let m = f16_at(b, 2);
                        for (j, &q) in b[4..20].iter().enumerate() {
                            o[j] = f32::from(q & 0x0F) * d + m;
                            o[j + 16] = f32::from(q >> 4) * d + m;
                        }
                    }
                    // d, 32 high bits, nibbles: a 5-bit quant minus 16. High
                    // bit j of `qh` extends low-nibble element j, bit j + 16
                    // extends high-nibble element j + 16.
                    GgmlDType::Q5_0 => {
                        let qh = u32::from_le_bytes([b[2], b[3], b[4], b[5]]);
                        for (j, &q) in b[6..22].iter().enumerate() {
                            let (lo, hi) = q5_pair(q, qh, j);
                            o[j] = f32::from(lo - 16) * d;
                            o[j + 16] = f32::from(hi - 16) * d;
                        }
                    }
                    // d, m, 32 high bits, nibbles: element = 5-bit quant * d + m.
                    GgmlDType::Q5_1 => {
                        let m = f16_at(b, 2);
                        let qh = u32::from_le_bytes([b[4], b[5], b[6], b[7]]);
                        for (j, &q) in b[8..24].iter().enumerate() {
                            let (lo, hi) = q5_pair(q, qh, j);
                            o[j] = f32::from(lo) * d + m;
                            o[j + 16] = f32::from(hi) * d + m;
                        }
                    }
                    // d, 32 signed bytes.
                    _ => {
                        for (o, &q) in o.iter_mut().zip(&b[2..34]) {
                            *o = f32::from(q.cast_signed()) * d;
                        }
                    }
                }
            }
        }
        other => {
            let info = gguf_file::TensorInfo {
                ggml_dtype: other,
                shape: out.len().into(),
                offset: 0,
            };
            let values = info
                .read(&mut std::io::Cursor::new(bytes), 0, &Device::Cpu)
                .and_then(|t| t.dequantize(&Device::Cpu))
                .and_then(|t| t.to_vec1::<f32>())
                .map_err(candle_err)?;
            for (o, v) in out.iter_mut().zip(values) {
                *o = v * scale;
            }
        }
    }
    Ok(())
}

/// The 5-bit quants (low nibble plus its high bit) of elements `j` and
/// `j + 16` of a Q5_0 or Q5_1 block.
fn q5_pair(q: u8, qh: u32, j: usize) -> (i16, i16) {
    let lo = (q & 0x0F) | ((((qh >> j) << 4) & 0x10) as u8);
    let hi = (q >> 4) | (((qh >> (j + 12)) & 0x10) as u8);
    (i16::from(lo), i16::from(hi))
}

/// RoPE inverse frequencies for `head_dim / 2` pairs: `base^(-2i/d)` divided
/// by the optional per-pair factor. A factor of `1e30` rotates by an angle
/// that rounds to exactly zero, so such a pair passes through unchanged.
fn rope_inv_freq(head_dim: usize, base: f64, factors: Option<&[f32]>) -> Vec<f64> {
    (0..head_dim / 2)
        .map(|i| {
            let freq = base.powf(-(2.0 * i as f64) / head_dim as f64);
            match factors {
                Some(ff) => freq / f64::from(ff[i]),
                None => freq,
            }
        })
        .collect()
}

/// Row-major `positions x inv_freq.len()` cosine and sine tables.
fn rope_cos_sin(inv_freq: &[f64], positions: usize) -> (Vec<f32>, Vec<f32>) {
    let mut cos = Vec::with_capacity(positions * inv_freq.len());
    let mut sin = Vec::with_capacity(positions * inv_freq.len());
    for pos in 0..positions {
        for &freq in inv_freq {
            let angle = pos as f64 * freq;
            cos.push(angle.cos() as f32);
            sin.push(angle.sin() as f32);
        }
    }
    (cos, sin)
}

/// RoPE cos/sin tensors of shape `(positions, head_dim / 2)`.
struct RopeTable {
    cos: Tensor,
    sin: Tensor,
}

impl RopeTable {
    fn new(
        head_dim: usize,
        base: f64,
        factors: Option<&[f32]>,
        positions: usize,
    ) -> Result<Self, CucaError> {
        let inv_freq = rope_inv_freq(head_dim, base, factors);
        let (cos, sin) = rope_cos_sin(&inv_freq, positions);
        let shape = (positions, head_dim / 2);
        Ok(Self {
            cos: Tensor::from_vec(cos, shape, &Device::Cpu).map_err(candle_err)?,
            sin: Tensor::from_vec(sin, shape, &Device::Cpu).map_err(candle_err)?,
        })
    }
}

/// The token embedding table (also the tied LM head), held quantized.
struct TokenEmbedding {
    tensor: Arc<QTensor>,
    dtype: GgmlDType,
    rows: usize,
    row_len: usize,
    row_bytes: usize,
}

impl TokenEmbedding {
    /// Embedding rows of `tokens`, concatenated and multiplied by `scale`.
    fn rows(&self, tokens: &[u32], scale: f32) -> Result<Vec<f32>, CucaError> {
        let data = self.tensor.data().map_err(candle_err)?;
        let mut out = vec![0f32; tokens.len() * self.row_len];
        for (&id, dst) in tokens.iter().zip(out.chunks_exact_mut(self.row_len)) {
            let id = id as usize;
            if id >= self.rows {
                return Err(err(format!(
                    "token id {id} outside the embedding table of {} rows",
                    self.rows
                )));
            }
            let row = &data[id * self.row_bytes..(id + 1) * self.row_bytes];
            dequant_row(self.dtype, row, scale, dst)?;
        }
        Ok(out)
    }
}

/// A 2-D tensor read row by row from the GGUF file instead of being held in
/// memory: the per-layer embedding table on every device, and the token
/// embedding lookups of the GPU engine.
pub(crate) struct RowTable {
    /// GGUF tensor name, for error messages.
    tensor: &'static str,
    /// What the rows are, for error messages ("per-layer embedding table").
    label: &'static str,
    file: File,
    base: u64,
    dtype: GgmlDType,
    rows: usize,
    row_len: usize,
    row_bytes: usize,
    buf: Vec<u8>,
}

impl RowTable {
    /// Index the `[rows, row_len]` tensor `tensor` of the GGUF file at
    /// `path`; no row is read yet.
    pub(crate) fn open(
        path: &Path,
        content: &gguf_file::Content,
        tensor: &'static str,
        label: &'static str,
        row_len: usize,
    ) -> Result<Self, CucaError> {
        let info = tensor_info(content, tensor)?;
        let dims = info.shape.dims();
        if dims.len() != 2 || dims[1] != row_len {
            return Err(err(format!(
                "tensor `{tensor}`: expected shape [rows, {row_len}], found {dims:?}"
            )));
        }
        let dtype = info.ggml_dtype;
        if !row_len.is_multiple_of(dtype.block_size()) {
            return Err(err(format!(
                "tensor `{tensor}`: row length {row_len} is not a multiple of the {dtype:?} block size {}",
                dtype.block_size()
            )));
        }
        let row_bytes = row_len / dtype.block_size() * dtype.type_size();
        let base = content.tensor_data_offset + info.offset;
        let file = File::open(path)?;
        let end = base + (dims[0] * row_bytes) as u64;
        if file.metadata()?.len() < end {
            return Err(err(format!(
                "tensor `{tensor}` extends past the end of the file: truncated GGUF"
            )));
        }
        Ok(Self {
            tensor,
            label,
            file,
            base,
            dtype,
            rows: dims[0],
            row_len,
            row_bytes,
            buf: vec![0u8; row_bytes],
        })
    }

    /// Rows of `tokens`, concatenated and multiplied by `scale`.
    pub(crate) fn rows(&mut self, tokens: &[u32], scale: f32) -> Result<Vec<f32>, CucaError> {
        let tensor = self.tensor;
        let io_err = |e: std::io::Error| CucaError::Io {
            message: format!("reading `{tensor}` rows: {e}"),
        };
        let mut out = vec![0f32; tokens.len() * self.row_len];
        for (&id, dst) in tokens.iter().zip(out.chunks_exact_mut(self.row_len)) {
            let id = id as usize;
            if id >= self.rows {
                return Err(err(format!(
                    "token id {id} outside the {} of {} rows",
                    self.label, self.rows
                )));
            }
            self.file
                .seek(SeekFrom::Start(self.base + (id * self.row_bytes) as u64))
                .map_err(io_err)?;
            self.file.read_exact(&mut self.buf).map_err(io_err)?;
            dequant_row(self.dtype, &self.buf, scale, dst)?;
        }
        Ok(out)
    }
}

/// The index entry of `tensor_name`, without reading its data.
fn tensor_info<'a>(
    content: &'a gguf_file::Content,
    tensor_name: &str,
) -> Result<&'a gguf_file::TensorInfo, CucaError> {
    content.tensor_infos.get(tensor_name).ok_or_else(|| {
        err(format!(
            "tensor `{tensor_name}`: missing from the GGUF file"
        ))
    })
}

/// Weights of one transformer block.
struct LayerWeights {
    attn_norm: Tensor,
    wq: QMatMul,
    q_norm: Tensor,
    /// K/V projections exist only on layers below `n_layer_kv`.
    kv: Option<KvWeights>,
    wo: QMatMul,
    post_attn_norm: Tensor,
    ffn_norm: Tensor,
    gate: QMatMul,
    up: QMatMul,
    down: QMatMul,
    post_ffw_norm: Tensor,
    inp_gate: QMatMul,
    proj: QMatMul,
    post_norm: Tensor,
    out_scale: f64,
}

struct KvWeights {
    wk: QMatMul,
    k_norm: Tensor,
    wv: QMatMul,
}

/// K/V cache of the layers that own K/V, plus the shared filled length.
struct KvCache {
    layers: Vec<Option<(Tensor, Tensor)>>,
    len: usize,
}

/// Reads tensors from the GGUF file, checking each against its expected shape.
struct Loader<'a> {
    content: &'a gguf_file::Content,
    reader: BufReader<File>,
    /// Hold matmul weights as dequantized f32 instead of quantized (test
    /// reference only: candle's quantized matmul rounds activations to Q8).
    dequantize: bool,
}

impl Loader<'_> {
    fn qtensor(&mut self, tensor_name: &str, shape: &[usize]) -> Result<QTensor, CucaError> {
        let qt = self
            .content
            .tensor(&mut self.reader, tensor_name, &Device::Cpu)
            .map_err(|e| err(format!("tensor `{tensor_name}`: {e}")))?;
        if qt.shape().dims() != shape {
            return Err(err(format!(
                "tensor `{tensor_name}`: expected shape {shape:?}, found {:?}",
                qt.shape().dims()
            )));
        }
        Ok(qt)
    }

    fn matmul(&mut self, tensor_name: &str, shape: &[usize]) -> Result<QMatMul, CucaError> {
        let qt = Arc::new(self.qtensor(tensor_name, shape)?);
        self.to_matmul(qt, tensor_name)
    }

    fn to_matmul(&self, qt: Arc<QTensor>, tensor_name: &str) -> Result<QMatMul, CucaError> {
        let matmul = if self.dequantize {
            qt.dequantize(&Device::Cpu).map(QMatMul::Tensor)
        } else {
            QMatMul::from_arc(qt)
        };
        matmul.map_err(|e| err(format!("tensor `{tensor_name}`: {e}")))
    }

    fn vector(&mut self, tensor_name: &str, len: usize) -> Result<Tensor, CucaError> {
        self.qtensor(tensor_name, &[len])?
            .dequantize(&Device::Cpu)
            .map_err(|e| err(format!("tensor `{tensor_name}`: {e}")))
    }

    fn scalar(&mut self, tensor_name: &str) -> Result<f64, CucaError> {
        let values = self
            .vector(tensor_name, 1)?
            .to_vec1::<f32>()
            .map_err(|e| err(format!("tensor `{tensor_name}`: {e}")))?;
        Ok(f64::from(values[0]))
    }

    /// Shape of a tensor from the file's index, without reading its data.
    fn shape(&self, tensor_name: &str) -> Result<&[usize], CucaError> {
        tensor_info(self.content, tensor_name).map(|info| info.shape.dims())
    }
}

/// A loaded Gemma 4 model with its K/V cache; see the module docs.
pub(crate) struct Gemma4 {
    params: Gemma4Params,
    context_length: usize,
    embed: TokenEmbedding,
    ple: RowTable,
    ple_proj: QMatMul,
    ple_proj_norm: Tensor,
    layers: Vec<LayerWeights>,
    output_norm: Tensor,
    lm_head: QMatMul,
    rope_full: RopeTable,
    rope_swa: RopeTable,
    /// Weights of the weightless value-norm, one per head dimension.
    ones_full: Tensor,
    ones_swa: Tensor,
    kv: KvCache,
}

impl Gemma4 {
    /// Load the model from the GGUF file at `path`, whose index `content`
    /// was read already, with a K/V cache for `context_length` positions.
    ///
    /// # Errors
    ///
    /// [`CucaError::Config`] when `context_length` is zero or above the
    /// checkpoint maximum; [`CucaError::Provider`] for unsupported or
    /// malformed checkpoints; [`CucaError::Io`] when the file cannot be read.
    pub(crate) fn load(
        content: &gguf_file::Content,
        path: &Path,
        context_length: usize,
    ) -> Result<Self, CucaError> {
        Self::load_with(content, path, context_length, false)
    }

    /// [`Gemma4::load`] with every matmul weight dequantized to f32: the
    /// exact-arithmetic reference the GPU engine is compared against.
    #[cfg(all(test, feature = "provider-local-gpu"))]
    pub(crate) fn load_dequantized(
        content: &gguf_file::Content,
        path: &Path,
        context_length: usize,
    ) -> Result<Self, CucaError> {
        Self::load_with(content, path, context_length, true)
    }

    fn load_with(
        content: &gguf_file::Content,
        path: &Path,
        context_length: usize,
        dequantize: bool,
    ) -> Result<Self, CucaError> {
        let params = Gemma4Params::from_metadata(&content.metadata)?;
        if context_length == 0 || context_length > params.model_ctx {
            return Err(CucaError::Config(format!(
                "context_length {context_length} exceeds the model maximum {}",
                params.model_ctx
            )));
        }
        let mut loader = Loader {
            content,
            reader: BufReader::new(File::open(path)?),
            dequantize,
        };
        let p = &params;

        // Token embedding: shared by the lookup and the tied LM head.
        let vocab = *loader
            .shape(name::TOKEN_EMBD)?
            .first()
            .ok_or_else(|| err(format!("tensor `{}`: scalar", name::TOKEN_EMBD)))?;
        let embd = loader.qtensor(name::TOKEN_EMBD, &[vocab, p.n_embd])?;
        let embd_dtype = embd.dtype();
        if !p.n_embd.is_multiple_of(embd_dtype.block_size()) {
            return Err(err(format!(
                "tensor `{}`: row length {} is not a multiple of the {embd_dtype:?} block size {}",
                name::TOKEN_EMBD,
                p.n_embd,
                embd_dtype.block_size()
            )));
        }
        let embd = Arc::new(embd);
        let embed = TokenEmbedding {
            row_bytes: p.n_embd / embd_dtype.block_size() * embd_dtype.type_size(),
            tensor: Arc::clone(&embd),
            dtype: embd_dtype,
            rows: vocab,
            row_len: p.n_embd,
        };
        let lm_head = loader.to_matmul(embd, name::TOKEN_EMBD)?;

        // Per-layer embeddings: index only; rows are read on demand.
        let ple_row_len = p.n_layer * p.ple_dim;
        let ple = RowTable::open(
            path,
            content,
            name::PER_LAYER_TOKEN_EMBD,
            "per-layer embedding table",
            ple_row_len,
        )?;
        let ple_proj = loader.matmul(name::PER_LAYER_MODEL_PROJ, &[ple_row_len, p.n_embd])?;
        let ple_proj_norm = loader.vector(name::PER_LAYER_PROJ_NORM, p.ple_dim)?;
        let output_norm = loader.vector(name::OUTPUT_NORM, p.n_embd)?;

        let factors = if content.tensor_infos.contains_key(name::ROPE_FREQS) {
            let ff = loader
                .vector(name::ROPE_FREQS, p.head_dim_full / 2)?
                .to_vec1::<f32>()
                .map_err(candle_err)?;
            if ff.iter().any(|f| !f.is_finite() || *f <= 0.0) {
                return Err(err(format!(
                    "tensor `{}`: factors must be finite and positive",
                    name::ROPE_FREQS
                )));
            }
            Some(ff)
        } else {
            None
        };
        let rope_full = RopeTable::new(
            p.head_dim_full,
            p.rope_base,
            factors.as_deref(),
            context_length,
        )?;
        let rope_swa = RopeTable::new(p.head_dim_swa, p.rope_base_swa, None, context_length)?;

        let mut layers = Vec::with_capacity(p.n_layer);
        let mut kv_layers = Vec::with_capacity(p.n_layer);
        for il in 0..p.n_layer {
            let d = if p.is_sliding[il] {
                p.head_dim_swa
            } else {
                p.head_dim_full
            };
            let q_width = p.n_head * d;
            let kv_width = p.n_kv_head * d;
            let gate_name = name::blk(il, name::FFN_GATE);
            let ff = *loader
                .shape(&gate_name)?
                .first()
                .ok_or_else(|| err(format!("tensor `{gate_name}`: scalar")))?;
            let blk = |part: &str| name::blk(il, part);

            let kv = if il < p.n_layer_kv {
                kv_layers.push(Some((
                    Tensor::zeros(
                        (1, p.n_kv_head, context_length, d),
                        DType::F32,
                        &Device::Cpu,
                    )
                    .map_err(candle_err)?,
                    Tensor::zeros(
                        (1, p.n_kv_head, context_length, d),
                        DType::F32,
                        &Device::Cpu,
                    )
                    .map_err(candle_err)?,
                )));
                Some(KvWeights {
                    wk: loader.matmul(&blk(name::ATTN_K), &[kv_width, p.n_embd])?,
                    k_norm: loader.vector(&blk(name::ATTN_K_NORM), d)?,
                    wv: loader.matmul(&blk(name::ATTN_V), &[kv_width, p.n_embd])?,
                })
            } else {
                kv_layers.push(None);
                None
            };
            layers.push(LayerWeights {
                attn_norm: loader.vector(&blk(name::ATTN_NORM), p.n_embd)?,
                wq: loader.matmul(&blk(name::ATTN_Q), &[q_width, p.n_embd])?,
                q_norm: loader.vector(&blk(name::ATTN_Q_NORM), d)?,
                kv,
                wo: loader.matmul(&blk(name::ATTN_OUTPUT), &[p.n_embd, q_width])?,
                post_attn_norm: loader.vector(&blk(name::POST_ATTENTION_NORM), p.n_embd)?,
                ffn_norm: loader.vector(&blk(name::FFN_NORM), p.n_embd)?,
                gate: loader.matmul(&gate_name, &[ff, p.n_embd])?,
                up: loader.matmul(&blk(name::FFN_UP), &[ff, p.n_embd])?,
                down: loader.matmul(&blk(name::FFN_DOWN), &[p.n_embd, ff])?,
                post_ffw_norm: loader.vector(&blk(name::POST_FFW_NORM), p.n_embd)?,
                inp_gate: loader.matmul(&blk(name::INP_GATE), &[p.ple_dim, p.n_embd])?,
                proj: loader.matmul(&blk(name::PROJ), &[p.n_embd, p.ple_dim])?,
                post_norm: loader.vector(&blk(name::POST_NORM), p.n_embd)?,
                out_scale: loader.scalar(&blk(name::LAYER_OUTPUT_SCALE))?,
            });
        }

        let ones_full =
            Tensor::ones(p.head_dim_full, DType::F32, &Device::Cpu).map_err(candle_err)?;
        let ones_swa =
            Tensor::ones(p.head_dim_swa, DType::F32, &Device::Cpu).map_err(candle_err)?;
        Ok(Self {
            params,
            context_length,
            embed,
            ple,
            ple_proj,
            ple_proj_norm,
            layers,
            output_norm,
            lm_head,
            rope_full,
            rope_swa,
            ones_full,
            ones_swa,
            kv: KvCache {
                layers: kv_layers,
                len: 0,
            },
        })
    }

    /// Run `tokens` through the model at positions `start_pos..`, appending
    /// their K/V to the cache. Returns the soft-capped next-token logits of
    /// the last token when `want_logits`, otherwise `None` (prefill chunks
    /// that precede the last skip the LM head).
    ///
    /// `start_pos` must equal the cache length; the cache grows to
    /// `start_pos + tokens.len()`, which must not exceed the context length.
    pub(crate) fn forward(
        &mut self,
        tokens: &[u32],
        start_pos: usize,
        want_logits: bool,
    ) -> Result<Option<Vec<f32>>, CucaError> {
        if tokens.is_empty() {
            return Err(err("forward requires at least one token"));
        }
        if start_pos != self.kv.len {
            return Err(err(format!(
                "forward start {start_pos} does not match the cache length {}",
                self.kv.len
            )));
        }
        let hi = start_pos + tokens.len();
        if hi > self.context_length {
            return Err(err(format!(
                "{hi} positions exceed the context length {}",
                self.context_length
            )));
        }
        let embeddings = self
            .embed
            .rows(tokens, (self.params.n_embd as f32).sqrt())?;
        let per_layer = self.ple.rows(tokens, (self.params.ple_dim as f32).sqrt())?;
        let logits = self
            .forward_layers(embeddings, per_layer, tokens.len(), start_pos, want_logits)
            .map_err(candle_err)?;
        self.kv.len = hi;
        Ok(logits)
    }

    /// Drop every cached position from `len` on; the next `forward` must
    /// start at `len`. Positions at or past it are overwritten, not cleared.
    pub(crate) fn truncate_cache(&mut self, len: usize) {
        self.kv.len = len.min(self.kv.len);
    }

    /// Number of cached positions (test-only: the engine tracks the ids).
    #[cfg(test)]
    pub(crate) fn cache_len(&self) -> usize {
        self.kv.len
    }

    fn forward_layers(
        &self,
        embeddings: Vec<f32>,
        per_layer: Vec<f32>,
        t: usize,
        start: usize,
        want_logits: bool,
    ) -> candle_core::Result<Option<Vec<f32>>> {
        let p = &self.params;
        let dev = &Device::Cpu;
        let eps = p.eps as f32;
        let mut x = Tensor::from_vec(embeddings, (1, t, p.n_embd), dev)?;
        let per_layer_tokens = Tensor::from_vec(per_layer, (1, t, p.n_layer, p.ple_dim), dev)?;
        let per_layer_ctx = self
            .ple_proj
            .forward(&x)?
            .affine(1.0 / (p.n_embd as f64).sqrt(), 0.0)?
            .reshape((1, t, p.n_layer, p.ple_dim))?;
        let per_layer_ctx = rms(&per_layer_ctx, &self.ple_proj_norm, eps)?;
        let ple = (&per_layer_ctx + &per_layer_tokens)?.affine(FRAC_1_SQRT_2, 0.0)?;

        for il in 0..p.n_layer {
            x = self.layer(il, &x, &ple, start, t)?;
        }

        if !want_logits {
            return Ok(None);
        }
        let last = x.narrow(1, t - 1, 1)?.contiguous()?;
        let logits = self.lm_head.forward(&rms(&last, &self.output_norm, eps)?)?;
        let logits = logits
            .affine(1.0 / p.softcap, 0.0)?
            .tanh()?
            .affine(p.softcap, 0.0)?;
        Ok(Some(logits.flatten_all()?.to_vec1::<f32>()?))
    }

    /// One transformer block over `t` positions starting at `start`.
    fn layer(
        &self,
        il: usize,
        x: &Tensor,
        ple: &Tensor,
        start: usize,
        t: usize,
    ) -> candle_core::Result<Tensor> {
        let p = &self.params;
        let lw = &self.layers[il];
        let eps = p.eps as f32;
        let sliding = p.is_sliding[il];
        let (d, rope_table, ones) = if sliding {
            (p.head_dim_swa, &self.rope_swa, &self.ones_swa)
        } else {
            (p.head_dim_full, &self.rope_full, &self.ones_full)
        };
        let hi = start + t;
        let cos = rope_table.cos.narrow(0, start, t)?;
        let sin = rope_table.sin.narrow(0, start, t)?;

        let h = rms(x, &lw.attn_norm, eps)?;
        let q = lw.wq.forward(&h)?.reshape((1, t, p.n_head, d))?;
        let q = rms(&q, &lw.q_norm, eps)?.transpose(1, 2)?.contiguous()?;
        let q = rope(&q, &cos, &sin)?;

        if let Some(kv) = &lw.kv {
            let k = kv.wk.forward(&h)?.reshape((1, t, p.n_kv_head, d))?;
            let k = rms(&k, &kv.k_norm, eps)?.transpose(1, 2)?.contiguous()?;
            let k = rope(&k, &cos, &sin)?;
            let v = kv.wv.forward(&h)?.reshape((1, t, p.n_kv_head, d))?;
            let v = rms(&v, ones, eps)?.transpose(1, 2)?.contiguous()?;
            let (cache_k, cache_v) = self.cache(il)?;
            cache_k.slice_set(&k, 2, start)?;
            cache_v.slice_set(&v, 2, start)?;
        }

        let (cache_k, cache_v) = self.cache(p.kv_source[il])?;
        let lo = if sliding {
            start.saturating_sub(p.window - 1)
        } else {
            0
        };
        let l = hi - lo;
        let k = cache_k.narrow(2, lo, l)?;
        let v = cache_v.narrow(2, lo, l)?;
        let mask = if t > 1 {
            let mask = attention_mask(start, t, lo, hi, sliding.then_some(p.window));
            Some(Tensor::from_vec(mask, (t, l), &Device::Cpu)?)
        } else {
            None
        };
        let attn = attend(&q, &k, &v, mask.as_ref())?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((1, t, p.n_head * d))?;
        let attn = lw.wo.forward(&attn)?;
        let x = (x + rms(&attn, &lw.post_attn_norm, eps)?)?;

        let f = rms(&x, &lw.ffn_norm, eps)?;
        let gated = (lw.gate.forward(&f)?.gelu()? * lw.up.forward(&f)?)?;
        let f = lw.down.forward(&gated)?;
        let x = (&x + rms(&f, &lw.post_ffw_norm, eps)?)?;

        let slice = ple
            .narrow(2, il, 1)?
            .contiguous()?
            .reshape((1, t, p.ple_dim))?;
        let g = (lw.inp_gate.forward(&x)?.gelu()? * slice)?;
        let x = (&x + rms(&lw.proj.forward(&g)?, &lw.post_norm, eps)?)?;
        x.affine(lw.out_scale, 0.0)
    }

    /// The K and V cache tensors of the K/V-owning layer `il`.
    fn cache(&self, il: usize) -> candle_core::Result<&(Tensor, Tensor)> {
        self.kv
            .layers
            .get(il)
            .and_then(Option::as_ref)
            .ok_or_else(|| candle_core::Error::msg(format!("layer {il} owns no K/V cache")))
    }
}

/// The loaded model on the device `LocalModelConfig::device` selected: candle
/// on the CPU, or burn on a GPU. Both expose the same two calls the engine
/// makes.
// One `Model` exists per engine, so the size gap between the variants costs
// nothing worth a box.
#[allow(clippy::large_enum_variant)]
pub(crate) enum Model {
    Cpu(Gemma4),
    #[cfg(feature = "provider-local-gpu")]
    Gpu(gpu::GpuGemma4),
}

impl Model {
    /// See [`Gemma4::forward`].
    pub(crate) fn forward(
        &mut self,
        tokens: &[u32],
        start_pos: usize,
        want_logits: bool,
    ) -> Result<Option<Vec<f32>>, CucaError> {
        match self {
            Model::Cpu(m) => m.forward(tokens, start_pos, want_logits),
            #[cfg(feature = "provider-local-gpu")]
            Model::Gpu(m) => m.forward(tokens, start_pos, want_logits),
        }
    }

    /// See [`Gemma4::truncate_cache`].
    pub(crate) fn truncate_cache(&mut self, len: usize) {
        match self {
            Model::Cpu(m) => m.truncate_cache(len),
            #[cfg(feature = "provider-local-gpu")]
            Model::Gpu(m) => m.truncate_cache(len),
        }
    }

    /// Number of cached positions (test-only).
    #[cfg(test)]
    pub(crate) fn cache_len(&self) -> usize {
        match self {
            Model::Cpu(m) => m.cache_len(),
            #[cfg(feature = "provider-local-gpu")]
            Model::Gpu(m) => m.cache_len(),
        }
    }
}

/// RMSNorm with a learned weight (no `1 + w`).
fn rms(x: &Tensor, weight: &Tensor, eps: f32) -> candle_core::Result<Tensor> {
    rms_norm(&x.contiguous()?, weight, eps)
}

/// Grouped-query attention without a score scale: `q` is `(1, n_head, t, d)`,
/// `k` and `v` are `(1, n_kv, l, d)` and `mask` the additive `t x l` mask
/// (`None` for a single query, which sees every key). Query head `h` reads KV
/// head `h / (n_head / n_kv)`. The result is `(1, n_head, t, d)`.
///
/// The `n_head / n_kv` query heads of one KV head are stacked into the rows of
/// one matmul against that head's keys, so K and V are never repeated.
fn attend(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
) -> candle_core::Result<Tensor> {
    let (_, n_head, t, d) = q.dims4()?;
    let (_, n_kv, l, _) = k.dims4()?;
    let rows = n_head / n_kv * t;
    let scores = q.reshape((1, n_kv, rows, d))?.matmul(&k.transpose(2, 3)?)?;
    let scores = match mask {
        Some(mask) => scores
            .reshape((1, n_head, t, l))?
            .broadcast_add(mask)?
            .reshape((1, n_kv, rows, l))?,
        None => scores,
    };
    softmax_last_dim(&scores.contiguous()?)?
        .matmul(v)?
        .reshape((1, n_head, t, d))
}

#[cfg(test)]
pub(crate) mod tiny {
    //! A tiny random-weight Gemma 4 written to a temporary GGUF file:
    //! small enough to run in a unit test, structured like the real model
    //! (shared K/V layers, sliding and full layers with a window shorter
    //! than the prompt, partial RoPE factors, per-layer embeddings).
    use std::sync::atomic::{AtomicU32, Ordering};

    use candle_core::quantized::gguf_file::Value;

    use super::*;

    const N_LAYER: usize = 6;
    const N_LAYER_KV: usize = 3;
    const N_EMBD: usize = 64;
    const N_HEAD: usize = 2;
    const HEAD_FULL: usize = 64;
    const HEAD_SWA: usize = 32;
    const WINDOW: usize = 4;
    const PLE: usize = 32;
    pub(crate) const VOCAB: usize = 16;
    const CONTEXT: usize = 32;
    const SLIDING: [bool; N_LAYER] = [true, true, false, true, true, false];

    /// Deterministic pseudo-random values in `[-scale, scale]`.
    struct Lcg(u64);

    impl Lcg {
        fn values(&mut self, n: usize, scale: f32) -> Vec<f32> {
            (0..n)
                .map(|_| {
                    self.0 = self
                        .0
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    let unit = (self.0 >> 40) as f32 / (1u64 << 24) as f32;
                    (unit * 2.0 - 1.0) * scale
                })
                .collect()
        }
    }

    pub(crate) struct TinyModel {
        pub(crate) path: std::path::PathBuf,
    }

    impl Drop for TinyModel {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.path);
        }
    }

    fn quantized(rng: &mut Lcg, dims: &[usize], dtype: GgmlDType, scale: f32) -> QTensor {
        let n: usize = dims.iter().product();
        let t = Tensor::from_vec(rng.values(n, scale), dims, &Device::Cpu).unwrap();
        QTensor::quantize(&t, dtype).unwrap()
    }

    fn norm(rng: &mut Lcg, len: usize) -> QTensor {
        let values: Vec<f32> = rng.values(len, 0.25).iter().map(|v| 1.0 + v).collect();
        let t = Tensor::from_vec(values, len, &Device::Cpu).unwrap();
        QTensor::quantize(&t, GgmlDType::F32).unwrap()
    }

    /// Head counts and embedding dtypes of a tiny model.
    #[derive(Clone, Copy)]
    pub(crate) struct TinyOptions {
        pub(crate) n_head: usize,
        pub(crate) n_kv_head: usize,
        /// Dtype of `token_embd.weight` (rows of `N_EMBD` = 64).
        pub(crate) embd_dtype: GgmlDType,
        /// Dtype of `per_layer_token_embd.weight` (rows of 192).
        pub(crate) ple_dtype: GgmlDType,
    }

    impl Default for TinyOptions {
        fn default() -> Self {
            Self {
                n_head: N_HEAD,
                n_kv_head: 1,
                embd_dtype: GgmlDType::Q8_0,
                ple_dtype: GgmlDType::Q4_0,
            }
        }
    }

    pub(crate) fn write_tiny_model() -> TinyModel {
        write_tiny_model_with(TinyOptions::default())
    }

    pub(crate) fn write_tiny_model_with(o: TinyOptions) -> TinyModel {
        static COUNTER: AtomicU32 = AtomicU32::new(0);
        let path = std::env::temp_dir().join(format!(
            "cuca-gemma4-tiny-{}-{}.gguf",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        let mut rng = Lcg(7);
        let mut tensors: Vec<(String, QTensor)> = Vec::new();
        let mut add = |n: &str, t: QTensor| tensors.push((n.to_string(), t));

        add(
            "token_embd.weight",
            quantized(&mut rng, &[VOCAB, N_EMBD], o.embd_dtype, 0.5),
        );
        add(
            "per_layer_token_embd.weight",
            quantized(&mut rng, &[VOCAB, N_LAYER * PLE], o.ple_dtype, 0.5),
        );
        add(
            "per_layer_model_proj.weight",
            quantized(&mut rng, &[N_LAYER * PLE, N_EMBD], GgmlDType::BF16, 0.2),
        );
        add("per_layer_proj_norm.weight", norm(&mut rng, PLE));
        add("output_norm.weight", norm(&mut rng, N_EMBD));
        let mut factors = vec![1.0f32; HEAD_FULL / 4];
        factors.resize(HEAD_FULL / 2, 1e30);
        add(
            "rope_freqs.weight",
            QTensor::quantize(
                &Tensor::from_vec(factors, HEAD_FULL / 2, &Device::Cpu).unwrap(),
                GgmlDType::F32,
            )
            .unwrap(),
        );
        for (il, &sliding) in SLIDING.iter().enumerate() {
            let d = if sliding { HEAD_SWA } else { HEAD_FULL };
            let ff = if il < N_LAYER_KV { 64 } else { 128 };
            let b = |part: &str| format!("blk.{il}.{part}.weight");
            add(&b("attn_norm"), norm(&mut rng, N_EMBD));
            add(
                &b("attn_q"),
                quantized(&mut rng, &[o.n_head * d, N_EMBD], GgmlDType::Q4_0, 0.3),
            );
            add(&b("attn_q_norm"), norm(&mut rng, d));
            if il < N_LAYER_KV {
                add(
                    &b("attn_k"),
                    quantized(&mut rng, &[o.n_kv_head * d, N_EMBD], GgmlDType::Q4_0, 0.3),
                );
                add(&b("attn_k_norm"), norm(&mut rng, d));
                add(
                    &b("attn_v"),
                    quantized(&mut rng, &[o.n_kv_head * d, N_EMBD], GgmlDType::Q4_0, 0.3),
                );
            }
            add(
                &b("attn_output"),
                quantized(&mut rng, &[N_EMBD, o.n_head * d], GgmlDType::Q4_0, 0.3),
            );
            add(&b("post_attention_norm"), norm(&mut rng, N_EMBD));
            add(&b("ffn_norm"), norm(&mut rng, N_EMBD));
            add(
                &b("ffn_gate"),
                quantized(&mut rng, &[ff, N_EMBD], GgmlDType::Q4_0, 0.3),
            );
            add(
                &b("ffn_up"),
                quantized(&mut rng, &[ff, N_EMBD], GgmlDType::Q4_0, 0.3),
            );
            add(
                &b("ffn_down"),
                quantized(&mut rng, &[N_EMBD, ff], GgmlDType::Q4_0, 0.3),
            );
            add(&b("post_ffw_norm"), norm(&mut rng, N_EMBD));
            add(
                &b("inp_gate"),
                quantized(&mut rng, &[PLE, N_EMBD], GgmlDType::Q4_0, 0.3),
            );
            add(
                &b("proj"),
                quantized(&mut rng, &[N_EMBD, PLE], GgmlDType::Q4_0, 0.3),
            );
            add(&b("post_norm"), norm(&mut rng, N_EMBD));
            add(
                &b("layer_output_scale"),
                QTensor::quantize(
                    &Tensor::from_vec(vec![0.9f32], 1, &Device::Cpu).unwrap(),
                    GgmlDType::F32,
                )
                .unwrap(),
            );
        }

        let pattern = Value::Array(SLIDING.iter().copied().map(Value::Bool).collect());
        let meta: Vec<(&str, Value)> = vec![
            ("general.architecture", Value::String("gemma4".into())),
            ("gemma4.block_count", Value::U32(N_LAYER as u32)),
            ("gemma4.context_length", Value::U32(CONTEXT as u32)),
            ("gemma4.embedding_length", Value::U32(N_EMBD as u32)),
            ("gemma4.attention.head_count", Value::U32(o.n_head as u32)),
            (
                "gemma4.attention.head_count_kv",
                Value::U32(o.n_kv_head as u32),
            ),
            ("gemma4.attention.key_length", Value::U32(HEAD_FULL as u32)),
            (
                "gemma4.attention.key_length_swa",
                Value::U32(HEAD_SWA as u32),
            ),
            ("gemma4.attention.sliding_window", Value::U32(WINDOW as u32)),
            ("gemma4.attention.sliding_window_pattern", pattern),
            (
                "gemma4.attention.shared_kv_layers",
                Value::U32((N_LAYER - N_LAYER_KV) as u32),
            ),
            (
                "gemma4.embedding_length_per_layer_input",
                Value::U32(PLE as u32),
            ),
            ("gemma4.rope.freq_base", Value::F32(1e6)),
            ("gemma4.rope.freq_base_swa", Value::F32(1e4)),
            ("gemma4.attention.layer_norm_rms_epsilon", Value::F32(1e-6)),
            ("gemma4.final_logit_softcapping", Value::F32(30.0)),
        ];
        let meta_refs: Vec<(&str, &Value)> = meta.iter().map(|(k, v)| (*k, v)).collect();
        let tensor_refs: Vec<(&str, &QTensor)> =
            tensors.iter().map(|(n, t)| (n.as_str(), t)).collect();
        let mut file = File::create(&path).unwrap();
        gguf_file::write(&mut file, &meta_refs, &tensor_refs).unwrap();
        TinyModel { path }
    }

    pub(crate) fn load(model: &TinyModel, context_length: usize) -> Gemma4 {
        let mut reader = BufReader::new(File::open(&model.path).unwrap());
        let content = gguf_file::Content::read(&mut reader).unwrap();
        Gemma4::load(&content, &model.path, context_length).unwrap()
    }

    pub(crate) const PROMPT: [u32; 11] = [3, 5, 7, 1, 9, 2, 4, 6, 8, 10, 12];

    pub(crate) fn assert_close(a: &[f32], b: &[f32]) {
        assert_eq!(a.len(), VOCAB);
        assert_eq!(b.len(), VOCAB);
        let scale = a.iter().fold(1.0f32, |m, v| m.max(v.abs()));
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            assert!(x.is_finite() && y.is_finite(), "logit {i} not finite");
            assert!(
                (x - y).abs() <= 2e-3 * scale,
                "logit {i}: {x} vs {y} (scale {scale})"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Gemma 4 E2B sliding pattern: every fifth layer (4, 9, ...) is full.
    fn e2b_pattern() -> Vec<bool> {
        (0..35).map(|il| il % 5 != 4).collect()
    }

    #[test]
    fn kv_sources_follow_the_e2b_pattern() {
        let sources = kv_sources(&e2b_pattern(), 15).unwrap();
        assert_eq!(sources.len(), 35);
        for (il, &src) in sources.iter().enumerate().take(15) {
            assert_eq!(src, il);
        }
        let pattern = e2b_pattern();
        for il in 15..35 {
            let expected = if pattern[il] { 13 } else { 14 };
            assert_eq!(sources[il], expected, "layer {il}");
        }
    }

    #[test]
    fn kv_sources_reject_a_type_with_no_owner() {
        // Layers 1.. are full but the only K/V owner (layer 0) is sliding.
        let err = kv_sources(&[true, false, false], 1).unwrap_err();
        assert!(err.to_string().contains("no earlier full-attention layer"));
    }

    #[test]
    fn attention_mask_is_causal() {
        // Two queries at positions 2 and 3 over keys 0..4.
        let mask = attention_mask(2, 2, 0, 4, None);
        let ninf = f32::NEG_INFINITY;
        assert_eq!(mask, vec![0.0, 0.0, 0.0, ninf, 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn attention_mask_window_boundary() {
        // Query 600, window 512: key 88 is 512 behind (masked), key 89 is 511
        // behind (visible); key 600 is the query itself.
        let mask = attention_mask(600, 1, 0, 601, Some(512));
        assert_eq!(mask[87], f32::NEG_INFINITY);
        assert_eq!(mask[88], f32::NEG_INFINITY);
        assert_eq!(mask[89], 0.0);
        assert_eq!(mask[600], 0.0);
        assert_eq!(mask.iter().filter(|&&m| m == 0.0).count(), 512);
    }

    #[test]
    fn attention_mask_window_with_offset_slice() {
        // lo = start - (W - 1) as the layer computes it: the first query sees
        // exactly keys lo..=start, later queries drop the oldest keys.
        let (start, t, w) = (10, 3, 4);
        let lo = start - (w - 1);
        let mask = attention_mask(start, t, lo, start + t, Some(w));
        let l = start + t - lo;
        let visible = |row: usize| -> Vec<usize> {
            (0..l)
                .filter(|&j| mask[row * l + j] == 0.0)
                .map(|j| j + lo)
                .collect()
        };
        assert_eq!(visible(0), vec![7, 8, 9, 10]);
        assert_eq!(visible(1), vec![8, 9, 10, 11]);
        assert_eq!(visible(2), vec![9, 10, 11, 12]);
    }

    #[test]
    fn prefill_chunk_bounds() {
        assert_eq!(prefill_chunk(8192), 256);
        assert_eq!(prefill_chunk(32768), 64);
        assert_eq!(prefill_chunk(131_072), 16);
        assert_eq!(prefill_chunk(1), 256);
    }

    fn f16_bytes(v: f32) -> [u8; 2] {
        half::f16::from_f32(v).to_bits().to_le_bytes()
    }

    #[test]
    fn dequant_row_q4_0() {
        // One block, scale 0.5: low nibbles are elements 0..16, high nibbles
        // elements 16..32, each minus 8.
        let mut block = Vec::from(f16_bytes(0.5));
        for j in 0..16u8 {
            block.push(j | ((15 - j) << 4));
        }
        let mut out = [0f32; 32];
        dequant_row(GgmlDType::Q4_0, &block, 2.0, &mut out).unwrap();
        for j in 0..16 {
            assert_eq!(out[j], (j as f32 - 8.0) * 0.5 * 2.0, "low {j}");
            assert_eq!(out[j + 16], ((15 - j) as f32 - 8.0) * 0.5 * 2.0, "high {j}");
        }
    }

    #[test]
    fn dequant_row_q8_0() {
        let mut block = Vec::from(f16_bytes(0.25));
        let quants: Vec<i8> = (0..32).map(|j| (j * 8 - 128) as i8).collect();
        block.extend(quants.iter().map(|&q| q as u8));
        // Two blocks to cover the block stride.
        block.extend_from_within(..);
        let mut out = [0f32; 64];
        dequant_row(GgmlDType::Q8_0, &block, 4.0, &mut out).unwrap();
        for (i, &o) in out.iter().enumerate() {
            assert_eq!(o, f32::from(quants[i % 32]) * 0.25 * 4.0, "element {i}");
        }
    }

    #[test]
    fn dequant_row_rejects_bad_input() {
        let mut out = [0f32; 32];
        assert!(dequant_row(GgmlDType::Q4_0, &[0u8; 17], 1.0, &mut out).is_err());
        assert!(dequant_row(GgmlDType::F32, &[0u8; 127], 1.0, &mut out).is_err());
        // 32 elements are not a whole 256-element K-quant block.
        assert!(dequant_row(GgmlDType::Q4K, &[0u8; 144], 1.0, &mut out).is_err());
    }

    /// Nibble bytes `j | (15 - j) << 4` of a 32-element legacy block.
    fn nibbles() -> Vec<u8> {
        (0..16u8).map(|j| j | ((15 - j) << 4)).collect()
    }

    #[test]
    fn dequant_row_q4_1() {
        // Element = nibble * d + m, scaled; low nibbles first, then high.
        let mut block = Vec::from(f16_bytes(0.5));
        block.extend(f16_bytes(-1.0));
        block.extend(nibbles());
        let mut out = [0f32; 32];
        dequant_row(GgmlDType::Q4_1, &block, 2.0, &mut out).unwrap();
        for j in 0..16 {
            assert_eq!(out[j], (j as f32 * 0.5 - 1.0) * 2.0, "low {j}");
            assert_eq!(out[j + 16], ((15 - j) as f32 * 0.5 - 1.0) * 2.0, "high {j}");
        }
    }

    /// The 5-bit quant of element `i` of a Q5 block: its nibble (low nibbles
    /// are elements 0..16, high nibbles 16..32) plus bit `i` of `qh` as bit 4.
    fn q5_quant(qh: u32, i: usize) -> f32 {
        let q = nibbles()[i % 16];
        let nibble = if i < 16 { q & 0x0F } else { q >> 4 };
        f32::from(nibble | ((((qh >> i) & 1) as u8) << 4))
    }

    #[test]
    fn dequant_row_q5_0() {
        let qh = 0xA5C3_0F1Eu32;
        let mut block = Vec::from(f16_bytes(0.25));
        block.extend(qh.to_le_bytes());
        block.extend(nibbles());
        let mut out = [0f32; 32];
        dequant_row(GgmlDType::Q5_0, &block, 2.0, &mut out).unwrap();
        for (i, &o) in out.iter().enumerate() {
            assert_eq!(o, (q5_quant(qh, i) - 16.0) * 0.25 * 2.0, "element {i}");
        }
    }

    #[test]
    fn dequant_row_q5_1() {
        let qh = 0x5A3C_F0E1u32;
        let mut block = Vec::from(f16_bytes(0.25));
        block.extend(f16_bytes(3.0));
        block.extend(qh.to_le_bytes());
        block.extend(nibbles());
        // Two blocks to cover the block stride.
        block.extend_from_within(..);
        let mut out = [0f32; 64];
        dequant_row(GgmlDType::Q5_1, &block, 2.0, &mut out).unwrap();
        for (i, &o) in out.iter().enumerate() {
            assert_eq!(o, (q5_quant(qh, i % 32) * 0.25 + 3.0) * 2.0, "element {i}");
        }
    }

    #[test]
    fn dequant_row_float_dtypes() {
        let values = [1.5f32, -2.0, 0.25, 8.0];
        let mut out = [0f32; 4];

        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        dequant_row(GgmlDType::F32, &bytes, 2.0, &mut out).unwrap();
        assert_eq!(out, [3.0, -4.0, 0.5, 16.0]);

        let bytes: Vec<u8> = values.iter().flat_map(|&v| f16_bytes(v)).collect();
        dequant_row(GgmlDType::F16, &bytes, 2.0, &mut out).unwrap();
        assert_eq!(out, [3.0, -4.0, 0.5, 16.0]);

        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|&v| half::bf16::from_f32(v).to_bits().to_le_bytes())
            .collect();
        dequant_row(GgmlDType::BF16, &bytes, 2.0, &mut out).unwrap();
        assert_eq!(out, [3.0, -4.0, 0.5, 16.0]);
    }

    /// Every dtype candle can quantize to must dequantize a row exactly as
    /// candle's own dequantizer does: the legacy and float paths are decoded
    /// here, the K-quants are delegated.
    #[test]
    fn dequant_row_matches_candle_for_every_dtype() {
        let row: Vec<f32> = (0..512)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.07)
            .collect();
        let tensor = Tensor::from_vec(row, 512, &Device::Cpu).unwrap();
        for dtype in [
            GgmlDType::F32,
            GgmlDType::F16,
            GgmlDType::BF16,
            GgmlDType::Q4_0,
            GgmlDType::Q4_1,
            GgmlDType::Q5_0,
            GgmlDType::Q5_1,
            GgmlDType::Q8_0,
            GgmlDType::Q2K,
            GgmlDType::Q3K,
            GgmlDType::Q4K,
            GgmlDType::Q5K,
            GgmlDType::Q6K,
        ] {
            let qt = QTensor::quantize(&tensor, dtype).unwrap();
            let expected = qt
                .dequantize(&Device::Cpu)
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let mut out = vec![0f32; 512];
            dequant_row(dtype, &qt.data().unwrap(), 1.0, &mut out).unwrap();
            assert_eq!(out, expected, "{dtype:?}");
            dequant_row(dtype, &qt.data().unwrap(), 4.0, &mut out).unwrap();
            for (o, e) in out.iter().zip(&expected) {
                assert_eq!(*o, e * 4.0, "{dtype:?}");
            }
        }
    }

    #[test]
    fn rope_unit_factors_leave_pairs_unrotated() {
        // 64 pairs at 1.0, 192 at 1e30 (the E2B full-attention layout).
        let mut factors = vec![1.0f32; 64];
        factors.extend(std::iter::repeat_n(1e30f32, 192));
        let inv_freq = rope_inv_freq(512, 1e6, Some(&factors));
        assert_eq!(inv_freq.len(), 256);
        let (cos, sin) = rope_cos_sin(&inv_freq, 131_072);
        let last_row = 131_071 * 256;
        for i in 64..256 {
            assert_eq!(cos[last_row + i], 1.0, "pair {i}");
            assert!(sin[last_row + i].abs() < 1e-12, "pair {i}");
        }
        // The rotated pairs still rotate.
        assert!(cos[last_row] < 0.999 || sin[last_row].abs() > 1e-3);
        // Position 0 never rotates.
        assert!(cos[..256].iter().all(|&c| c == 1.0));
        assert!(sin[..256].iter().all(|&s| s == 0.0));
    }

    #[test]
    fn rope_inv_freq_without_factors_matches_the_definition() {
        let inv = rope_inv_freq(8, 10_000.0, None);
        assert_eq!(inv.len(), 4);
        for (i, f) in inv.iter().enumerate() {
            let expected = 10_000f64.powf(-(2.0 * i as f64) / 8.0);
            assert!((f - expected).abs() < 1e-15);
        }
    }

    /// candle must accept the narrowed, transposed K/V views `layer` feeds to
    /// matmul without a copy.
    #[test]
    fn attention_matmul_accepts_narrowed_cache_views() {
        let dev = Device::Cpu;
        let values: Vec<f32> = (0..16 * 8).map(|i| i as f32 * 0.01).collect();
        let cache = Tensor::from_vec(values.clone(), (1, 1, 16, 8), &dev).unwrap();
        let k = cache.narrow(2, 3, 5).unwrap();
        let q = Tensor::ones((1, 1, 6, 8), DType::F32, &dev).unwrap();
        let scores = q.matmul(&k.transpose(2, 3).unwrap()).unwrap();
        assert_eq!(scores.dims(), &[1, 1, 6, 5]);
        let got = scores.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let expected_row: Vec<f32> = (3..8)
            .map(|pos| values[pos * 8..(pos + 1) * 8].iter().sum())
            .collect();
        for row in got.chunks(5) {
            for (a, b) in row.iter().zip(&expected_row) {
                assert!((a - b).abs() < 1e-4, "{a} vs {b}");
            }
        }
        let a = softmax_last_dim(&scores.contiguous().unwrap())
            .unwrap()
            .matmul(&cache.narrow(2, 3, 5).unwrap())
            .unwrap();
        assert_eq!(a.dims(), &[1, 1, 6, 8]);
    }

    /// `attend` on narrowed cache views against a per-head scalar reference:
    /// query head `h` must read KV head `h / (n_head / n_kv)`.
    #[test]
    fn grouped_attention_matches_a_per_head_reference() {
        let dev = Device::Cpu;
        let (n_head, n_kv, ctx, d, lo, l) = (4usize, 2usize, 16usize, 8usize, 3usize, 5usize);
        let gen_values = |n: usize, salt: usize| -> Vec<f32> {
            (0..n)
                .map(|i| (((i * 31 + salt * 17) % 23) as f32 - 11.0) * 0.1)
                .collect()
        };
        let kc = gen_values(n_kv * ctx * d, 1);
        let vc = gen_values(n_kv * ctx * d, 2);
        let cache_k = Tensor::from_vec(kc.clone(), (1, n_kv, ctx, d), &dev).unwrap();
        let cache_v = Tensor::from_vec(vc.clone(), (1, n_kv, ctx, d), &dev).unwrap();
        let k = cache_k.narrow(2, lo, l).unwrap();
        let v = cache_v.narrow(2, lo, l).unwrap();

        for t in [3usize, 1] {
            let qv = gen_values(n_head * t * d, 3 + t);
            let q = Tensor::from_vec(qv.clone(), (1, n_head, t, d), &dev).unwrap();
            let hi = lo + l;
            let mask_values = attention_mask(hi - t, t, lo, hi, None);
            let mask =
                (t > 1).then(|| Tensor::from_vec(mask_values.clone(), (t, l), &dev).unwrap());
            let got = attend(&q, &k, &v, mask.as_ref())
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            assert_eq!(got.len(), n_head * t * d);

            for h in 0..n_head {
                let kv_head = h / (n_head / n_kv);
                for i in 0..t {
                    let scores: Vec<f32> = (0..l)
                        .map(|j| {
                            let key = &kc[(kv_head * ctx + lo + j) * d..][..d];
                            let dot: f32 = qv[(h * t + i) * d..][..d]
                                .iter()
                                .zip(key)
                                .map(|(a, b)| a * b)
                                .sum();
                            dot + mask_values[i * l + j]
                        })
                        .collect();
                    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let weights: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
                    let total: f32 = weights.iter().sum();
                    for x in 0..d {
                        let expected: f32 = (0..l)
                            .map(|j| weights[j] / total * vc[(kv_head * ctx + lo + j) * d + x])
                            .sum();
                        let actual = got[(h * t + i) * d + x];
                        assert!(
                            (actual - expected).abs() < 1e-4,
                            "t {t} head {h} query {i} dim {x}: {actual} vs {expected}"
                        );
                    }
                }
            }
        }
    }

    fn md(entries: &[(&str, gguf_file::Value)]) -> HashMap<String, gguf_file::Value> {
        entries
            .iter()
            .map(|(k, v)| ((*k).to_string(), v.clone()))
            .collect()
    }

    fn e2b_metadata() -> HashMap<String, gguf_file::Value> {
        use gguf_file::Value::{Array, Bool, F32, U32};
        md(&[
            ("gemma4.block_count", U32(35)),
            ("gemma4.context_length", U32(131_072)),
            ("gemma4.embedding_length", U32(1536)),
            ("gemma4.attention.head_count", U32(8)),
            ("gemma4.attention.head_count_kv", U32(1)),
            ("gemma4.attention.key_length", U32(512)),
            ("gemma4.attention.key_length_swa", U32(256)),
            ("gemma4.attention.sliding_window", U32(512)),
            (
                "gemma4.attention.sliding_window_pattern",
                Array(e2b_pattern().into_iter().map(Bool).collect()),
            ),
            ("gemma4.attention.shared_kv_layers", U32(20)),
            ("gemma4.embedding_length_per_layer_input", U32(256)),
            ("gemma4.rope.freq_base", F32(1e6)),
            ("gemma4.rope.freq_base_swa", F32(1e4)),
            ("gemma4.attention.layer_norm_rms_epsilon", F32(1e-6)),
            ("gemma4.final_logit_softcapping", F32(30.0)),
        ])
    }

    #[test]
    fn params_read_the_e2b_metadata() {
        let p = Gemma4Params::from_metadata(&e2b_metadata()).unwrap();
        assert_eq!(p.n_layer, 35);
        assert_eq!(p.n_layer_kv, 15);
        assert_eq!(p.n_embd, 1536);
        assert_eq!(p.n_kv_head, 1);
        assert_eq!((p.head_dim_full, p.head_dim_swa), (512, 256));
        assert_eq!((p.window, p.ple_dim, p.model_ctx), (512, 256, 131_072));
        assert_eq!(p.kv_source[15], 13);
        assert_eq!(p.kv_source[19], 14);
        assert_eq!(p.softcap, 30.0);
    }

    #[test]
    fn params_report_the_missing_and_mistyped_key() {
        let mut meta = e2b_metadata();
        meta.remove("gemma4.block_count");
        let msg = Gemma4Params::from_metadata(&meta).unwrap_err().to_string();
        assert!(msg.contains("GGUF metadata `gemma4.block_count`"), "{msg}");

        let mut meta = e2b_metadata();
        meta.insert(
            "gemma4.rope.freq_base".into(),
            gguf_file::Value::U32(1_000_000),
        );
        let msg = Gemma4Params::from_metadata(&meta).unwrap_err().to_string();
        assert!(
            msg.contains("GGUF metadata `gemma4.rope.freq_base`"),
            "{msg}"
        );
    }

    #[test]
    fn params_read_the_gqa_metadata() {
        let mut meta = e2b_metadata();
        meta.insert(
            "gemma4.attention.head_count_kv".into(),
            gguf_file::Value::U32(2),
        );
        assert_eq!(Gemma4Params::from_metadata(&meta).unwrap().n_kv_head, 2);

        // A uniform per-layer array is accepted as one count.
        meta.insert(
            "gemma4.attention.head_count_kv".into(),
            gguf_file::Value::Array(vec![gguf_file::Value::U32(4); 35]),
        );
        assert_eq!(Gemma4Params::from_metadata(&meta).unwrap().n_kv_head, 4);
    }

    #[test]
    fn params_reject_unsupported_checkpoints() {
        use gguf_file::Value::{Array, U32};
        let reject = |key: &str, value: gguf_file::Value, needle: &str| {
            let mut meta = e2b_metadata();
            meta.insert(key.into(), value);
            let msg = Gemma4Params::from_metadata(&meta).unwrap_err().to_string();
            assert!(msg.contains(needle), "{msg}");
        };
        reject(
            "gemma4.attention.head_count_kv",
            U32(3),
            "3 KV heads do not divide 8 query heads",
        );
        reject(
            "gemma4.attention.head_count_kv",
            U32(0),
            "0 KV heads do not divide",
        );
        reject(
            "gemma4.attention.head_count_kv",
            Array(vec![U32(1), U32(2)]),
            "per-layer KV head counts",
        );
        reject(
            "gemma4.attention.value_length",
            U32(256),
            "differs from the key length 512",
        );
        reject(
            "gemma4.attention.value_length_swa",
            U32(128),
            "differs from the key length 256",
        );
        reject("gemma4.expert_count", U32(8), "MoE");

        let mut meta = e2b_metadata();
        meta.insert("gemma4.expert_count".into(), U32(0));
        meta.insert("gemma4.attention.value_length".into(), U32(512));
        meta.insert("gemma4.attention.value_length_swa".into(), U32(256));
        assert!(Gemma4Params::from_metadata(&meta).is_ok());
    }

    use super::tiny::{
        PROMPT, TinyOptions, assert_close, load, write_tiny_model, write_tiny_model_with,
    };

    #[test]
    fn tiny_model_loads_and_yields_finite_varying_logits() {
        let model = write_tiny_model();
        let mut g = load(&model, 32);
        let logits = g.forward(&PROMPT, 0, true).unwrap().unwrap();
        assert_eq!(logits.len(), 16);
        assert!(logits.iter().all(|l| l.is_finite() && l.abs() <= 30.0));
        let (min, max) = logits
            .iter()
            .fold((f32::MAX, f32::MIN), |(lo, hi), &l| (lo.min(l), hi.max(l)));
        assert!(max - min > 1e-3, "logits are constant");
        assert!(
            g.forward(&PROMPT[..2], 0, false).is_err(),
            "start must equal cache length"
        );
    }

    #[test]
    fn chunked_prefill_matches_single_shot() {
        let model = write_tiny_model();
        let mut whole = load(&model, 32);
        let expected = whole.forward(&PROMPT, 0, true).unwrap().unwrap();

        // Chunks cross the 4-token sliding window and mix lengths 1, 3, 4, 3.
        let mut chunked = load(&model, 32);
        assert!(chunked.forward(&PROMPT[..1], 0, false).unwrap().is_none());
        assert!(chunked.forward(&PROMPT[1..4], 1, false).unwrap().is_none());
        assert!(chunked.forward(&PROMPT[4..8], 4, false).unwrap().is_none());
        let got = chunked.forward(&PROMPT[8..], 8, true).unwrap().unwrap();
        assert_close(&expected, &got);
    }

    #[test]
    fn decode_step_matches_prefill() {
        let model = write_tiny_model();
        let mut whole = load(&model, 32);
        let expected = whole.forward(&PROMPT, 0, true).unwrap().unwrap();

        let mut stepped = load(&model, 32);
        let n = PROMPT.len();
        stepped.forward(&PROMPT[..n - 1], 0, false).unwrap();
        let got = stepped
            .forward(&PROMPT[n - 1..], n - 1, true)
            .unwrap()
            .unwrap();
        assert_close(&expected, &got);
    }

    #[test]
    fn truncate_cache_replays_identically() {
        let model = write_tiny_model();
        let mut g = load(&model, 32);
        let first = g.forward(&PROMPT, 0, true).unwrap().unwrap();
        g.truncate_cache(5);
        assert!(
            g.forward(&PROMPT[6..], 6, true).is_err(),
            "gap after truncate"
        );
        let replay = g.forward(&PROMPT[5..], 5, true).unwrap().unwrap();
        assert_close(&first, &replay);
        // A different continuation after truncation must change the logits.
        g.truncate_cache(5);
        let other = g
            .forward(&[11, 13, 15, 14, 13, 12], 5, true)
            .unwrap()
            .unwrap();
        let diff = first
            .iter()
            .zip(&other)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(diff > 1e-3, "different suffix produced the same logits");
    }

    #[test]
    fn forward_rejects_overflow_and_unknown_tokens() {
        let model = write_tiny_model();
        let mut g = load(&model, 8);
        assert!(
            g.forward(&PROMPT, 0, true).is_err(),
            "11 tokens exceed context 8"
        );
        assert!(g.forward(&[], 0, true).is_err());
        assert!(
            g.forward(&[16], 0, true).is_err(),
            "token 16 is outside the vocabulary"
        );
        assert!(g.forward(&PROMPT[..8], 0, true).is_ok());
    }

    #[test]
    fn load_rejects_context_length_above_the_model_maximum() {
        let model = write_tiny_model();
        let mut reader = BufReader::new(File::open(&model.path).unwrap());
        let content = gguf_file::Content::read(&mut reader).unwrap();
        let err = Gemma4::load(&content, &model.path, 33).err().unwrap();
        assert!(matches!(&err, CucaError::Config(m) if m.contains("exceeds the model maximum 32")));
        assert!(Gemma4::load(&content, &model.path, 0).is_err());
    }

    /// Per-position KV bytes by the general formula: every K/V-owning layer
    /// stores `n_kv_head * head_dim` K and V values in f32.
    fn kv_bytes_per_position(p: &Gemma4Params) -> usize {
        (0..p.n_layer_kv)
            .map(|il| {
                let d = if p.is_sliding[il] {
                    p.head_dim_swa
                } else {
                    p.head_dim_full
                };
                p.n_kv_head * d * 2 * 4
            })
            .sum()
    }

    #[test]
    fn kv_cache_bytes_follow_the_general_formula() {
        // E2B: 12 sliding (256) and 3 full (512) owners, one KV head.
        let e2b = Gemma4Params::from_metadata(&e2b_metadata()).unwrap();
        assert_eq!(kv_bytes_per_position(&e2b), 36_864);
        assert_eq!(kv_bytes_per_position(&e2b) * 8192, 301_989_888);

        // E4B: 42 layers (every sixth full), 18 shared, two KV heads: 20
        // sliding and 4 full owners.
        let mut meta = e2b_metadata();
        let pattern = (0..42)
            .map(|il| gguf_file::Value::Bool(il % 6 != 5))
            .collect();
        for (key, value) in [
            ("gemma4.block_count", gguf_file::Value::U32(42)),
            ("gemma4.embedding_length", gguf_file::Value::U32(2560)),
            ("gemma4.attention.head_count_kv", gguf_file::Value::U32(2)),
            (
                "gemma4.attention.shared_kv_layers",
                gguf_file::Value::U32(18),
            ),
            (
                "gemma4.attention.sliding_window_pattern",
                gguf_file::Value::Array(pattern),
            ),
        ] {
            meta.insert(key.into(), value);
        }
        let e4b = Gemma4Params::from_metadata(&meta).unwrap();
        assert_eq!((e4b.n_layer_kv, e4b.n_kv_head), (24, 2));
        assert_eq!(kv_bytes_per_position(&e4b), 114_688);
    }

    /// Grouped-query attention (4 query heads over 2 KV heads) must keep the
    /// forward pass consistent across chunking and single-token decode, like
    /// the single-KV-head model.
    #[test]
    fn gqa_tiny_model_is_chunk_and_decode_consistent() {
        let model = write_tiny_model_with(TinyOptions {
            n_head: 4,
            n_kv_head: 2,
            ..TinyOptions::default()
        });
        let mut whole = load(&model, 32);
        let expected = whole.forward(&PROMPT, 0, true).unwrap().unwrap();
        assert!(expected.iter().all(|l| l.is_finite()));

        let mut chunked = load(&model, 32);
        chunked.forward(&PROMPT[..3], 0, false).unwrap();
        chunked.forward(&PROMPT[3..8], 3, false).unwrap();
        let got = chunked.forward(&PROMPT[8..], 8, true).unwrap().unwrap();
        assert_close(&expected, &got);

        let mut stepped = load(&model, 32);
        let n = PROMPT.len();
        stepped.forward(&PROMPT[..n - 1], 0, false).unwrap();
        let got = stepped
            .forward(&PROMPT[n - 1..], n - 1, true)
            .unwrap()
            .unwrap();
        assert_close(&expected, &got);
    }

    /// Both embedding tables load and look up rows in every legacy and float
    /// dtype, not just the published Q4_0 / Q8_0 layout.
    #[test]
    fn tiny_models_load_with_other_embedding_dtypes() {
        for (embd_dtype, ple_dtype) in [
            (GgmlDType::Q4_1, GgmlDType::Q5_0),
            (GgmlDType::Q5_1, GgmlDType::Q8_0),
            (GgmlDType::Q5_0, GgmlDType::Q4_1),
            (GgmlDType::F16, GgmlDType::Q5_1),
            (GgmlDType::F32, GgmlDType::F16),
            (GgmlDType::BF16, GgmlDType::BF16),
        ] {
            let model = write_tiny_model_with(TinyOptions {
                embd_dtype,
                ple_dtype,
                ..TinyOptions::default()
            });
            let mut g = load(&model, 32);
            let logits = g.forward(&PROMPT, 0, true).unwrap().unwrap();
            assert!(
                logits.iter().all(|l| l.is_finite() && l.abs() <= 30.0),
                "{embd_dtype:?} / {ple_dtype:?}"
            );
        }
    }
}
