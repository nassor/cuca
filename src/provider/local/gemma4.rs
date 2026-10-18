//! Gemma 4 E2B forward pass on candle CPU tensors, loaded from a GGUF file.
//!
//! The architecture differs from earlier Gemma generations in these ways, all
//! implemented in [`Gemma4::forward`]:
//!
//! - RMSNorm multiplies by the weight directly (no `1 + w`), attention scale
//!   is 1.0, and there is no attention soft-cap; the value projection gets a
//!   weightless RMSNorm.
//! - Layers alternate sliding-window (window 512, head dim 256, RoPE base
//!   1e4) and full attention (head dim 512, RoPE base 1e6 with the
//!   `rope_freqs.weight` per-pair factors that leave 192 of 256 pairs
//!   unrotated). One KV head serves all 8 query heads.
//! - The last `shared_kv_layers` layers own no K/V projections: they attend
//!   to the cache of the last earlier layer of their own attention type.
//! - Per-layer embeddings: every token also selects a 8960-wide row of
//!   `per_layer_token_embd`, split into one 256-wide slice per layer and
//!   gated into that layer's residual stream.
//! - Final logits are soft-capped: `30 * tanh(x / 30)`.
//!
//! # Memory
//!
//! - Weights stay quantized (Q4_0 / Q8_0) and are consumed by candle's
//!   quantized matmul; norm vectors and `per_layer_model_proj` (BF16) are
//!   held as f32.
//! - The token embedding (Q8_0, 427 MB) is held once and shared with the tied
//!   LM head.
//! - `per_layer_token_embd` (Q4_0, 1.32 GB) is never loaded: each token's row
//!   (5,040 bytes) is read from the GGUF file with a seek and dequantized, so
//!   a forward pass costs one small disk read (usually an OS page-cache hit)
//!   per input token.
//! - The KV cache is preallocated for `context_length` positions: only the
//!   15 layers that own K/V store them (12 sliding with head dim 256, 3 full
//!   with head dim 512), `context_length * 36,864` bytes in f32 (302 MB at
//!   8192). Sliding layers keep every position, not a ring, so a sliding
//!   layer reads a `window`-long slice at the end of its cache.
//! - RoPE cos/sin tables are built once for `context_length` positions.
//! - A prefill call processes at most [`prefill_chunk`] tokens, which bounds
//!   the transient `n_head * chunk * context_length` f32 score matrix of a
//!   full-attention layer to about 64 MB.

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

/// Hyperparameters read from the `gemma4.*` GGUF metadata.
#[derive(Debug, Clone)]
pub(crate) struct Gemma4Params {
    /// Transformer blocks.
    pub(crate) n_layer: usize,
    /// Residual stream width.
    pub(crate) n_embd: usize,
    /// Query heads (one KV head serves them all).
    pub(crate) n_head: usize,
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
    /// implementation does not cover (MoE, more than one KV head).
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
        let n_head_kv = md_usize(md, "gemma4.attention.head_count_kv")?;
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

        if n_head_kv != 1 {
            return Err(err("only single-KV-head Gemma 4 checkpoints are supported"));
        }
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

/// Dequantize one row of a Q4_0 or Q8_0 tensor into `out`, multiplying every
/// element by `scale`.
///
/// `bytes` holds the row's blocks and must be exactly
/// `out.len() / 32` blocks long (18 bytes per Q4_0 block, 34 per Q8_0 block).
pub(crate) fn dequant_row(
    dtype: GgmlDType,
    bytes: &[u8],
    scale: f32,
    out: &mut [f32],
) -> Result<(), CucaError> {
    let (block_bytes, quants) = match dtype {
        GgmlDType::Q4_0 => (18, 16),
        GgmlDType::Q8_0 => (34, 32),
        other => {
            return Err(err(format!(
                "row dequantization supports Q4_0 and Q8_0 only, got {other:?}"
            )));
        }
    };
    if !out.len().is_multiple_of(32) || bytes.len() != out.len() / 32 * block_bytes {
        return Err(err(format!(
            "row of {} bytes does not hold {} {dtype:?} elements",
            bytes.len(),
            out.len()
        )));
    }
    let (out_blocks, _) = out.as_chunks_mut::<32>();
    for (block, o) in bytes.chunks_exact(block_bytes).zip(out_blocks) {
        let d = half::f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32() * scale;
        let qs = &block[2..2 + quants];
        if dtype == GgmlDType::Q4_0 {
            for (j, &q) in qs.iter().enumerate() {
                o[j] = f32::from(i16::from(q & 0x0F) - 8) * d;
                o[j + 16] = f32::from(i16::from(q >> 4) - 8) * d;
            }
        } else {
            for (o, &q) in o.iter_mut().zip(qs) {
                *o = f32::from(q.cast_signed()) * d;
            }
        }
    }
    Ok(())
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

/// The per-layer embedding table, read row by row from the GGUF file instead
/// of being held in memory.
struct PleTable {
    file: File,
    base: u64,
    dtype: GgmlDType,
    rows: usize,
    row_len: usize,
    row_bytes: usize,
    buf: Vec<u8>,
}

impl PleTable {
    /// Rows of `tokens`, concatenated and multiplied by `scale`.
    fn rows(&mut self, tokens: &[u32], scale: f32) -> Result<Vec<f32>, CucaError> {
        let io_err = |e: std::io::Error| CucaError::Io {
            message: format!("reading per-layer embeddings: {e}"),
        };
        let mut out = vec![0f32; tokens.len() * self.row_len];
        for (&id, dst) in tokens.iter().zip(out.chunks_exact_mut(self.row_len)) {
            let id = id as usize;
            if id >= self.rows {
                return Err(err(format!(
                    "token id {id} outside the per-layer embedding table of {} rows",
                    self.rows
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
        let qt = self.qtensor(tensor_name, shape)?;
        QMatMul::from_arc(Arc::new(qt)).map_err(|e| err(format!("tensor `{tensor_name}`: {e}")))
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
        self.content
            .tensor_infos
            .get(tensor_name)
            .map(|info| info.shape.dims())
            .ok_or_else(|| {
                err(format!(
                    "tensor `{tensor_name}`: missing from the GGUF file"
                ))
            })
    }
}

/// A loaded Gemma 4 model with its K/V cache; see the module docs.
pub(crate) struct Gemma4 {
    params: Gemma4Params,
    context_length: usize,
    embed: TokenEmbedding,
    ple: PleTable,
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
        };
        let p = &params;

        // Token embedding: shared by the lookup and the tied LM head.
        let vocab = *loader
            .shape(name::TOKEN_EMBD)?
            .first()
            .ok_or_else(|| err(format!("tensor `{}`: scalar", name::TOKEN_EMBD)))?;
        let embd = loader.qtensor(name::TOKEN_EMBD, &[vocab, p.n_embd])?;
        let embd_dtype = embd.dtype();
        if !matches!(embd_dtype, GgmlDType::Q8_0 | GgmlDType::Q4_0) {
            return Err(err(format!(
                "tensor `{}`: dtype {embd_dtype:?} is not supported (Q8_0 or Q4_0 expected)",
                name::TOKEN_EMBD
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
        let lm_head = QMatMul::from_arc(embd)
            .map_err(|e| err(format!("tensor `{}`: {e}", name::TOKEN_EMBD)))?;

        // Per-layer embeddings: index only; rows are read on demand.
        let ple_row_len = p.n_layer * p.ple_dim;
        let ple_info = content
            .tensor_infos
            .get(name::PER_LAYER_TOKEN_EMBD)
            .ok_or_else(|| {
                err(format!(
                    "tensor `{}`: missing from the GGUF file",
                    name::PER_LAYER_TOKEN_EMBD
                ))
            })?;
        let ple_dims = ple_info.shape.dims();
        if ple_dims.len() != 2 || ple_dims[1] != ple_row_len {
            return Err(err(format!(
                "tensor `{}`: expected shape [rows, {ple_row_len}], found {ple_dims:?}",
                name::PER_LAYER_TOKEN_EMBD
            )));
        }
        let ple_dtype = ple_info.ggml_dtype;
        if !matches!(ple_dtype, GgmlDType::Q4_0 | GgmlDType::Q8_0)
            || !ple_row_len.is_multiple_of(ple_dtype.block_size())
        {
            return Err(err(format!(
                "tensor `{}`: dtype {ple_dtype:?} is not supported (Q4_0 or Q8_0 expected)",
                name::PER_LAYER_TOKEN_EMBD
            )));
        }
        let ple_row_bytes = ple_row_len / ple_dtype.block_size() * ple_dtype.type_size();
        let ple_base = content.tensor_data_offset + ple_info.offset;
        let ple_file = File::open(path)?;
        let ple_end = ple_base + (ple_dims[0] * ple_row_bytes) as u64;
        if ple_file.metadata()?.len() < ple_end {
            return Err(err(format!(
                "tensor `{}` extends past the end of the file: truncated GGUF",
                name::PER_LAYER_TOKEN_EMBD
            )));
        }
        let ple = PleTable {
            file: ple_file,
            base: ple_base,
            dtype: ple_dtype,
            rows: ple_dims[0],
            row_len: ple_row_len,
            row_bytes: ple_row_bytes,
            buf: vec![0u8; ple_row_bytes],
        };
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
            let gate_name = name::blk(il, name::FFN_GATE);
            let ff = *loader
                .shape(&gate_name)?
                .first()
                .ok_or_else(|| err(format!("tensor `{gate_name}`: scalar")))?;
            let blk = |part: &str| name::blk(il, part);

            let kv = if il < p.n_layer_kv {
                kv_layers.push(Some((
                    Tensor::zeros((1, 1, context_length, d), DType::F32, &Device::Cpu)
                        .map_err(candle_err)?,
                    Tensor::zeros((1, 1, context_length, d), DType::F32, &Device::Cpu)
                        .map_err(candle_err)?,
                )));
                Some(KvWeights {
                    wk: loader.matmul(&blk(name::ATTN_K), &[d, p.n_embd])?,
                    k_norm: loader.vector(&blk(name::ATTN_K_NORM), d)?,
                    wv: loader.matmul(&blk(name::ATTN_V), &[d, p.n_embd])?,
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
            let k = kv.wk.forward(&h)?.reshape((1, t, 1, d))?;
            let k = rms(&k, &kv.k_norm, eps)?.transpose(1, 2)?.contiguous()?;
            let k = rope(&k, &cos, &sin)?;
            let v = kv.wv.forward(&h)?.reshape((1, t, 1, d))?;
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
        let scores = q
            .reshape((1, 1, p.n_head * t, d))?
            .matmul(&k.transpose(2, 3)?)?;
        let scores = if t > 1 {
            let mask = attention_mask(start, t, lo, hi, sliding.then_some(p.window));
            let mask = Tensor::from_vec(mask, (t, l), &Device::Cpu)?;
            scores
                .reshape((1, p.n_head, t, l))?
                .broadcast_add(&mask)?
                .reshape((1, 1, p.n_head * t, l))?
        } else {
            scores
        };
        let attn = softmax_last_dim(&scores.contiguous()?)?.matmul(&v)?;
        let attn = attn
            .reshape((1, p.n_head, t, d))?
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

/// RMSNorm with a learned weight (no `1 + w`).
fn rms(x: &Tensor, weight: &Tensor, eps: f32) -> candle_core::Result<Tensor> {
    rms_norm(&x.contiguous()?, weight, eps)
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

    pub(crate) fn write_tiny_model() -> TinyModel {
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
            quantized(&mut rng, &[VOCAB, N_EMBD], GgmlDType::Q8_0, 0.5),
        );
        add(
            "per_layer_token_embd.weight",
            quantized(&mut rng, &[VOCAB, N_LAYER * PLE], GgmlDType::Q4_0, 0.5),
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
                quantized(&mut rng, &[N_HEAD * d, N_EMBD], GgmlDType::Q4_0, 0.3),
            );
            add(&b("attn_q_norm"), norm(&mut rng, d));
            if il < N_LAYER_KV {
                add(
                    &b("attn_k"),
                    quantized(&mut rng, &[d, N_EMBD], GgmlDType::Q4_0, 0.3),
                );
                add(&b("attn_k_norm"), norm(&mut rng, d));
                add(
                    &b("attn_v"),
                    quantized(&mut rng, &[d, N_EMBD], GgmlDType::Q4_0, 0.3),
                );
            }
            add(
                &b("attn_output"),
                quantized(&mut rng, &[N_EMBD, N_HEAD * d], GgmlDType::Q4_0, 0.3),
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
            ("gemma4.attention.head_count", Value::U32(N_HEAD as u32)),
            ("gemma4.attention.head_count_kv", Value::U32(1)),
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
        assert!(dequant_row(GgmlDType::F32, &[0u8; 128], 1.0, &mut out).is_err());
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
    fn params_reject_unsupported_checkpoints() {
        let mut meta = e2b_metadata();
        meta.insert(
            "gemma4.attention.head_count_kv".into(),
            gguf_file::Value::U32(2),
        );
        let msg = Gemma4Params::from_metadata(&meta).unwrap_err().to_string();
        assert!(msg.contains("only single-KV-head"), "{msg}");

        let mut meta = e2b_metadata();
        meta.insert("gemma4.expert_count".into(), gguf_file::Value::U32(8));
        let msg = Gemma4Params::from_metadata(&meta).unwrap_err().to_string();
        assert!(msg.contains("MoE"), "{msg}");

        let mut meta = e2b_metadata();
        meta.insert("gemma4.expert_count".into(), gguf_file::Value::U32(0));
        assert!(Gemma4Params::from_metadata(&meta).is_ok());
    }

    use super::tiny::{PROMPT, assert_close, load, write_tiny_model};

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
}
