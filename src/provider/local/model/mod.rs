//! The generic transformer forward pass behind `provider-local`: any GGUF
//! architecture in the [`arch`] table runs through the same code, on the CPU
//! (candle, [`cpu`]) or on a GPU (burn/CubeCL, `gpu`, `provider-local-gpu`).
//!
//! Every dimension comes from the GGUF metadata and tensor shapes
//! ([`params::Hparams`], [`layout::LayerLayout`]); the loader holds no
//! checkpoint constants, and an unsupported checkpoint (unknown architecture,
//! per-layer value length different from the key length, a tensor dtype
//! candle cannot read) is rejected with an error naming the cause.
//!
//! # What the forward pass covers
//!
//! Following llama.cpp's `llm_build_*` graphs, selected per architecture by
//! [`arch::Arch::behavior`] and per layer by [`params::LayerSpec`]:
//!
//! - grouped-query attention with an optional Q/K/V/O bias, fused `attn_qkv`,
//!   per-head QK-norm, a per-layer score scale, an optional tanh soft-cap on
//!   the scores, and causal or sliding-window masks (per-layer pattern);
//! - RoPE ([`rope`]): adjacent-pair (`NORM`) or half-split (`NEOX`) pairing,
//!   partial rotary dimensions, a base per layer type, linear and YaRN
//!   scaling, per-pair factors (`rope_freqs`, LongRoPE short/long with the
//!   attention factor) and NoPE layers;
//! - RMSNorm (the weight multiplies directly: GGUF files bake Gemma's `+1`
//!   in), optional post-attention and post-FFN norms, embedding scale,
//!   residual scale, logit scale and final logit soft-cap;
//! - SwiGLU / GeGLU FFNs with separate or fused gate and up projections;
//! - top-k mixture of experts (router softmax, optional renormalisation);
//! - tied or separate output head;
//! - Gemma 4 specifics: shared K/V layers, per-layer embeddings, weightless
//!   value norm, per-layer output scale;
//! - Qwen 3.5 specifics ([`arch::Arch::Qwen35`]): a hybrid stack of gated
//!   delta-net layers ([`delta`]: a causal depthwise convolution and a
//!   recurrent delta-rule matrix state per value head, no K/V cache) and, in
//!   every `full_attention_interval`-th layer, gated full attention
//!   (`attn_q` packs `[query | gate]` per head, the output is multiplied by
//!   `sigmoid(gate)`) with partial NEOX RoPE over `rope.dimension_count`
//!   dimensions (text positions make the multimodal sections irrelevant); the
//!   FFN norm is the tensor named `post_attention_norm`; the trailing
//!   `nextn_predict_layers` blocks are a multi-token-prediction head, not
//!   trunk layers: the trunk loaders ignore them and their tensors stay in
//!   the file for the head ([`layout::name`] `nextn.*`), which
//!   [`mtp::Drafter::InTrunk`] drives (module docs of [`cpu::NextnHead`]). The
//!   final-norm hidden rows it reads, one per position, come from
//!   `forward_ext(.., Hidden::PostNormAll, ..)`. The GPU engine does not run
//!   it: loading it on a GPU is an error.
//!
//! # Cache rollback
//!
//! K/V rows are positional: dropping a position just lowers the cache
//! length. A delta-net layer's state summarises every token before the
//! current end, cannot be truncated and cannot be undone, so
//! [`Model::truncate_cache`] returns the number of positions a model really
//! retained, and a hybrid model retains only positions it holds a state for:
//!
//! - the current end (the live state);
//! - the **checkpoint** ([`Model::checkpoint`], one slot, replaced on every
//!   call; the engine takes it while prefilling, see below);
//! - the **verify record**: after a `Want::All` call over `2..=17` rows
//!   ([`cpu::MAX_VERIFY_ROWS`], one drafter verification round: the pending
//!   token and up to 16 drafts) every position inside the call (`start ..
//!   start + rows`, the position before its first row included), until the
//!   next forward call. Its state is rebuilt, not stored per row: the model
//!   keeps one copy of the state before the call and the per-row inputs of
//!   the recurrence, and replays the first rows of the call on rollback,
//!   bit for bit as the original run computed them. That costs one state
//!   copy per verification round instead of one per row.
//!
//! Any other requested length falls back to the largest of those positions
//! not above it, `0` (recompute everything, from the zero state) if there is
//! none. A checkpoint above the retained length is dropped: its positions are
//! about to be overwritten. A forward call that fails half-way empties the
//! cache (a partly advanced state cannot be repaired). A prefill in any
//! chunking, a token-by-token run and a restore-and-replay produce the same
//! logits and state bit for bit.
//!
//! The engine consumes the returned length: it keeps `cached` to the retained
//! prefix, prefills from there and reports that as the reused prompt tokens.
//! While prefilling a prompt it checkpoints the state after the last
//! end-of-turn token (the end of the last complete message: the generation
//! prompt that follows is what chat templates rewrite on the next turn; a
//! prompt without one is checkpointed at its end), so the next turn of a
//! conversation restarts from there and the output equals a cold run's.
//!
//! # Memory
//!
//! - Weights stay quantized (any dtype candle's quantized matmul reads) and
//!   are consumed by candle's quantized matmul; norm vectors, biases and
//!   float-typed matmul weights (including `per_layer_model_proj`, BF16 in
//!   the published Gemma 4 files) are held as f32. Resident weights are about
//!   the file size.
//! - Expert tensors are split once at load into one quantized 2-D weight per
//!   expert, so a mixture-of-experts model also costs its file size, and the
//!   transient memory of a forward pass is independent of the expert count:
//!   tokens are grouped by expert, each expert runs on its own tokens
//!   only, and the FFN transient is `tokens * n_embd` f32.
//! - The token embedding is held once (quantized) and shared with the tied
//!   LM head. `per_layer_token_embd` (Gemma 4, 1.32 GB in E2B Q4_0) is never
//!   loaded: each token's row is read from the GGUF file with a seek and
//!   dequantized by [`quant::dequant_row`].
//! - The KV cache is preallocated for `context_length` positions: only the
//!   layers that own K/V store them, [`params::Hparams::kv_bytes_per_position`]
//!   `= 2 * 4 * sum(n_kv_head * head_dim)` bytes per position in f32, the sum
//!   running over the K/V-owning layers with that layer's own head count and
//!   head dimension. Sliding layers keep every position, not a ring, so a
//!   sliding layer reads a `window`-long slice at the end of its cache.
//! - A hybrid model's recurrent state is independent of the context length:
//!   [`params::Hparams::recurrent_state_bytes`] is 4 bytes times the sum, over
//!   the delta-net layers, of `(conv_kernel - 1) * conv_dim + n_v_heads *
//!   head_k_dim * head_v_dim` (Qwen3.5-0.8B: 18 layers of 1.12 MB, 20.2 MB).
//!   The cache keeps that state live, plus the checkpoint (a second copy,
//!   allocated at the first checkpoint) and, once a model verifies drafts, the
//!   verify record: one more copy and `4 * (conv_dim + 2 * n_v_heads)` bytes
//!   per verified row per layer (at most 17 rows: 445 KB per row over the 18
//!   layers of the 0.8B model, 7.6 MB for 17). In all that is about three
//!   times the recurrent state (61 MB for the 0.8B model).
//! - The K/V cache of a hybrid model covers its attention layers only:
//!   `kv_bytes_per_position` counts just the K/V-owning layers (24,576 bytes
//!   per position for the 0.8B model: 6 attention layers with 2 KV heads of
//!   256).
//! - The multi-token-prediction head of a Qwen 3.5 `*-MTP` file, when the
//!   engine drafts with it, adds its block's quantized weights (21.7 MB for
//!   the 0.8B model) and a K/V cache of one attention layer
//!   (`2 * 4 * n_kv_head * head_dim` bytes per position, 4 KB for the 0.8B
//!   model: 33.5 MB at 8192 positions); see [`cpu::NextnHead`].
//! - RoPE cos/sin tables are built once for `context_length` positions, one
//!   pair per distinct rotary embedding.
//! - A prefill call processes at most [`prefill_chunk`] tokens, which bounds
//!   the transient `n_head * chunk * context_length` f32 score matrix of a
//!   full-attention layer to `n_head * 8 MB`.

pub(crate) mod arch;
pub(crate) mod cpu;
pub(crate) mod delta;
#[cfg(feature = "provider-local-gpu")]
pub(crate) mod gpu;
pub(crate) mod layout;
pub(crate) mod moe;
pub(crate) mod mtp;
pub(crate) mod params;
pub(crate) mod qmm;
pub(crate) mod quant;
pub(crate) mod rope;
#[cfg(test)]
pub(crate) mod tiny;

use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use candle_core::Device;
use candle_core::quantized::gguf_file;

use self::layout::name;
use self::params::Hparams;
use self::rope::{CosSin, Factors};
use super::LocalDevice;
use crate::error::CucaError;
use crate::types::ProviderEndpoint;

pub(crate) fn err(message: impl Into<String>) -> CucaError {
    CucaError::provider(ProviderEndpoint::Local, message)
}

pub(crate) fn candle_err(e: candle_core::Error) -> CucaError {
    err(e.to_string())
}

pub(crate) fn meta_err(key: &str, e: impl std::fmt::Display) -> CucaError {
    err(format!("GGUF metadata `{key}`: {e}"))
}

/// Which logits a forward pass returns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Want {
    /// None: the pass only fills the K/V cache (prefill chunks that precede
    /// the last skip the LM head).
    None,
    /// The next-token logits of the last token: `vocab` values.
    Last,
    /// The logits after every token: `tokens * vocab` values, row-major.
    /// Speculative verification reads one row per drafted position.
    All,
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

/// The per-pair factors of `spec`, read from the tensor its [`Factors`]
/// names, validated against the rotated pair count.
fn rope_factors(
    content: &gguf_file::Content,
    path: &Path,
    hp: &Hparams,
    spec: &rope::RopeSpec,
    context_length: usize,
) -> Result<Option<Vec<f32>>, CucaError> {
    let tensor = match spec.factors {
        Factors::None => return Ok(None),
        Factors::Freqs => name::ROPE_FREQS,
        Factors::LongShort if context_length > hp.n_ctx_orig => name::ROPE_FACTORS_LONG,
        Factors::LongShort => name::ROPE_FACTORS_SHORT,
    };
    let mut reader = BufReader::new(File::open(path)?);
    let values = content
        .tensor(&mut reader, tensor, &Device::Cpu)
        .and_then(|q| q.dequantize(&Device::Cpu))
        .and_then(|t| t.flatten_all())
        .and_then(|t| t.to_vec1::<f32>())
        .map_err(|e| err(format!("tensor `{tensor}`: {e}")))?;
    if values.len() != spec.n_dims / 2 {
        return Err(err(format!(
            "tensor `{tensor}`: {} factors for {} rotated pairs",
            values.len(),
            spec.n_dims / 2
        )));
    }
    if values.iter().any(|f| !f.is_finite() || *f <= 0.0) {
        return Err(err(format!(
            "tensor `{tensor}`: factors must be finite and positive"
        )));
    }
    Ok(Some(values))
}

/// Cosine and sine tables for every distinct rotary embedding of `hp`, for
/// `context_length` positions.
pub(crate) fn build_rope_tables(
    content: &gguf_file::Content,
    path: &Path,
    hp: &Hparams,
    context_length: usize,
) -> Result<Vec<CosSin>, CucaError> {
    hp.ropes
        .iter()
        .map(|spec| {
            let factors = rope_factors(content, path, hp, spec, context_length)?;
            Ok(spec.tables(factors.as_deref(), context_length))
        })
        .collect()
}

/// Reject a context length of zero or above the checkpoint maximum.
pub(crate) fn check_context(hp: &Hparams, context_length: usize) -> Result<(), CucaError> {
    if context_length == 0 || context_length > hp.n_ctx_train {
        return Err(CucaError::Config(format!(
            "context_length {context_length} exceeds the model maximum {}",
            hp.n_ctx_train
        )));
    }
    Ok(())
}

/// Read the hyperparameters of the file whose index is `content`.
pub(crate) fn read_hparams(content: &gguf_file::Content) -> Result<Hparams, CucaError> {
    Hparams::from_metadata(&content.metadata, &|n| content.tensor_infos.contains_key(n))
}

/// The loaded model on the device `LocalModelConfig::device` selected: candle
/// on the CPU, or burn on a GPU. Both expose the same calls the engine makes.
// One `Model` exists per engine, so the size gap between the variants costs
// nothing worth a box.
#[allow(clippy::large_enum_variant)]
pub(crate) enum Model {
    Cpu(cpu::Transformer),
    #[cfg(feature = "provider-local-gpu")]
    Gpu(gpu::GpuTransformer),
}

impl Model {
    /// Load the model from the GGUF file at `path`, whose index `content` was
    /// read already, onto `device`, with a K/V cache for `context_length`
    /// positions.
    ///
    /// # Errors
    ///
    /// [`CucaError::Config`] when `context_length` is zero or above the
    /// checkpoint maximum or `device` is compiled out; [`CucaError::Provider`]
    /// for unsupported or malformed checkpoints; [`CucaError::Io`] when the
    /// file cannot be read.
    pub(crate) fn load(
        content: &gguf_file::Content,
        path: &Path,
        context_length: usize,
        device: LocalDevice,
    ) -> Result<Self, CucaError> {
        match device {
            LocalDevice::Cpu => Ok(Model::Cpu(cpu::Transformer::load(
                content,
                path,
                context_length,
            )?)),
            #[cfg(feature = "provider-local-gpu")]
            device => Ok(Model::Gpu(gpu::GpuTransformer::load(
                content,
                path,
                context_length,
                device,
            )?)),
            #[cfg(not(feature = "provider-local-gpu"))]
            device => Err(super::device_unavailable(device).unwrap_or_else(|| {
                CucaError::Config(format!("device {device:?} is not available"))
            })),
        }
    }

    /// Run `tokens` through the model at positions `start_pos..`, appending
    /// their K/V to the cache; returns the logits `want` selects and the
    /// final-norm hidden rows `hidden` selects (one per logits row).
    ///
    /// `start_pos` must equal the cache length; the cache grows to
    /// `start_pos + tokens.len()`, which must not exceed the context length.
    /// With `exact`, every row equals what a single-token step computes bit
    /// for bit, which speculative verification needs: guaranteed on the CPU;
    /// on a GPU the rows agree only up to float rounding.
    ///
    /// # Errors
    ///
    /// [`CucaError::Provider`] for a rejected call or a failed forward pass.
    pub(crate) fn forward_ext(
        &mut self,
        tokens: &[u32],
        start_pos: usize,
        want: Want,
        hidden: cpu::Hidden,
        exact: bool,
    ) -> Result<cpu::Forward, CucaError> {
        match self {
            Model::Cpu(m) => m.forward_ext(tokens, start_pos, want, hidden, exact),
            #[cfg(feature = "provider-local-gpu")]
            Model::Gpu(m) => m.forward_ext(tokens, start_pos, want, hidden, exact),
        }
    }

    /// The architecture.
    pub(crate) fn arch(&self) -> arch::Arch {
        match self {
            Model::Cpu(m) => m.arch(),
            #[cfg(feature = "provider-local-gpu")]
            Model::Gpu(m) => m.arch(),
        }
    }

    /// Drop every cached position from `len` on and return how many positions
    /// the cache retains: at most `len` and at most the current length. The
    /// next `forward` must start at the returned length.
    ///
    /// An attention-only model retains exactly `min(len, current length)`. A
    /// hybrid model (recurrent layers) retains only positions it holds a
    /// state for: the current end, the [`Model::checkpoint`] and the rows of
    /// the last multi-row `Want::All` call; otherwise it falls back to the
    /// largest of those not above `len` (`0`: recompute from scratch). See
    /// [`cpu::Transformer::truncate_cache`] and `Cache rollback` in the
    /// module docs.
    pub(crate) fn truncate_cache(&mut self, len: usize) -> usize {
        match self {
            Model::Cpu(m) => m.truncate_cache(len),
            #[cfg(feature = "provider-local-gpu")]
            Model::Gpu(m) => m.truncate_cache(len),
        }
    }

    /// Keep the recurrent state at the current end of the cache as the one
    /// checkpoint [`Model::truncate_cache`] can return to, replacing the
    /// previous one. A no-op for models that can truncate anywhere
    /// (attention-only, and every GPU model).
    pub(crate) fn checkpoint(&mut self) {
        match self {
            Model::Cpu(m) => m.checkpoint(),
            #[cfg(feature = "provider-local-gpu")]
            Model::Gpu(_) => {}
        }
    }

    /// Whether the model keeps a recurrent state next to its K/V cache (gated
    /// delta-net layers): its cache can then return only to the positions
    /// [`Model::truncate_cache`] lists.
    pub(crate) fn is_hybrid(&self) -> bool {
        match self {
            Model::Cpu(m) => m.is_hybrid(),
            #[cfg(feature = "provider-local-gpu")]
            Model::Gpu(_) => false,
        }
    }

    /// Most rows of one `Want::All` forward call after which every
    /// intermediate length can still be returned to with
    /// [`Model::truncate_cache`]: unbounded unless the model is hybrid.
    pub(crate) fn rollback_rows(&self) -> usize {
        match self {
            Model::Cpu(m) => m.rollback_rows(),
            #[cfg(feature = "provider-local-gpu")]
            Model::Gpu(_) => usize::MAX,
        }
    }

    /// Number of cached positions (test-only: the engine tracks the ids).
    #[cfg(test)]
    pub(crate) fn cache_len(&self) -> usize {
        match self {
            Model::Cpu(m) => m.cache_len(),
            #[cfg(feature = "provider-local-gpu")]
            Model::Gpu(m) => m.cache_len(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attention_mask_is_causal() {
        // Two queries at positions 2 and 3 over keys 0..4.
        let mask = attention_mask(2, 2, 0, 4, None);
        let ninf = f32::NEG_INFINITY;
        assert_eq!(mask, [0.0, 0.0, 0.0, ninf, 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn attention_mask_window_boundary() {
        // Query 600, window 512: key 88 is 512 behind (masked), key 89 is 511
        // behind (visible); key 600 is the query itself.
        let mask = attention_mask(600, 1, 0, 601, Some(512));
        assert_eq!(mask[88], f32::NEG_INFINITY);
        assert_eq!(mask[89], 0.0);
        assert_eq!(mask[600], 0.0);
    }

    #[test]
    fn attention_mask_window_with_offset_slice() {
        // lo = start - (W - 1) as the layer computes it: the first query sees
        // exactly keys lo..=start, later queries drop the oldest keys.
        let (start, t, w) = (10, 3, 4);
        let lo = start - (w - 1);
        let hi = start + t;
        let l = hi - lo;
        let mask = attention_mask(start, t, lo, hi, Some(w));
        for i in 0..t {
            for j in 0..l {
                let (q, p) = (start + i, lo + j);
                let visible = p <= q && q - p < w;
                assert_eq!(mask[i * l + j] == 0.0, visible, "q {q} p {p}");
            }
        }
    }

    #[test]
    fn prefill_chunk_bounds() {
        assert_eq!(prefill_chunk(8192), 256);
        assert_eq!(prefill_chunk(32768), 64);
        assert_eq!(prefill_chunk(131_072), 16);
        assert_eq!(prefill_chunk(1), 256);
    }

    /// A GPU device refuses a hybrid model with a clear error before it
    /// opens any device (so no GPU is needed to see it).
    #[cfg(feature = "provider-local-gpu")]
    #[test]
    fn a_gpu_rejects_recurrent_layers_with_a_clear_error() {
        use super::tiny::{TinyQwen35, write_qwen35};
        let model = write_qwen35(&TinyQwen35::default());
        let mut reader = BufReader::new(File::open(&model.path).unwrap());
        let content = gguf_file::Content::read(&mut reader).unwrap();
        let e = Model::load(&content, &model.path, 32, LocalDevice::Cuda(0))
            .err()
            .unwrap();
        assert!(matches!(e, CucaError::Provider { .. }), "{e:?}");
        let msg = e.to_string();
        assert!(
            msg.contains("recurrent (gated delta-net) layers are not implemented on GPU devices")
                && msg.contains("LocalDevice::Cpu")
                && msg.contains("qwen35"),
            "{msg}"
        );
    }
}
