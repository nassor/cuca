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
//!   it: `DrafterSource::InTrunk` on a GPU is a configuration error.
//!
//! # Cache rollback
//!
//! K/V rows are positional: dropping a position just lowers the cache
//! length. A delta-net layer's state summarises every token before the
//! current end, cannot be truncated and cannot be undone, so
//! [`Model::rewind`] returns the number of positions a model really
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
//! The GPU engine follows the same rules for the live state and the
//! checkpoint (its states are immutable device tensors, so a checkpoint is a
//! set of handles and costs nothing) but keeps no verify record
//! ([`Model::rollback_rows`] is 1 for a hybrid GPU model): on a GPU the
//! batched rows equal single steps only up to float rounding, and no drafter
//! can pair with a hybrid model there.
//!
//! The engine consumes the returned length: it truncates its sequence's block
//! tables to the retained prefix, prefills from there and reports that as the
//! reused prompt tokens. An attention-only model keeps no state besides K/V,
//! so its requests look their prompt up in the prefix store instead
//! ([`paged::KvPools::lookup`]); a hybrid model keeps its last sequence. While
//! prefilling a prompt a hybrid model checkpoints the state after the last
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
//!   only, and the FFN transient is `tokens * n_embd` f32. With
//!   `ExpertPlacement::Tiered` the experts are not loaded: [`tier`] reads
//!   them from the file on demand into bounded per-layer caches (RAM, and
//!   device memory on a GPU), so expert memory is the configured budgets.
//! - The token embedding is held once (quantized) and shared with the tied
//!   LM head. `per_layer_token_embd` (Gemma 4, 1.32 GB in E2B Q4_0) is never
//!   loaded: each token's row is read from the GGUF file with a seek and
//!   dequantized by [`quant::dequant_row`].
//! - The K/V cache is paged and preallocated ([`paged`]): only the layers
//!   that own K/V store them, in pools of blocks of 16 positions, `2 * 4 * 16
//!   * n_kv_head * head_dim` bytes per block in f32 per K/V-owning layer, with
//!   that layer's own head count and head dimension. Layers with the same
//!   window share block ids (a group). A full-attention group holds twice
//!   `context_length` positions, so a finished request's blocks stay cached
//!   while the next runs; a sliding-window group `ceil((window - 1 + rows) /
//!   16) + 2` blocks per running sequence plus up to `context_length`
//!   positions of cached prefixes ([`paged::KvLayout::new`],
//!   [`paged::KvLayout::pool_bytes`] sums it all). A sliding-window block is
//!   released once every position in it is before the window of the next
//!   query; the rollback of a draft round (at most 16 rows past the accepted
//!   length) never needs a released block. Prefix hits shorten by whole
//!   blocks until every group still holds the blocks its window needs.
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
//!   `kv_cache_bytes` counts just the K/V-owning layers (24,576 bytes per
//!   position for the 0.8B model: 6 attention layers with 2 KV heads of
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
#[cfg(feature = "provider-local-gpu")]
pub(crate) mod moe_kernel;
pub(crate) mod mtp;
pub(crate) mod paged;
#[cfg(feature = "provider-local-gpu")]
pub(crate) mod paged_kernel;
pub(crate) mod params;
pub(crate) mod qmm;
pub(crate) mod quant;
pub(crate) mod rope;
pub(crate) mod tier;
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
use super::{ExpertPlacement, LocalDevice};
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
        if first <= last {
            row[first - lo..=last - lo].fill(0.0);
        }
    }
    mask
}

/// Most tokens one prefill call processes: bounds the transient score matrix
/// (`n_head * chunk * context_length` f32 per full-attention layer).
pub(crate) fn prefill_chunk(context_length: usize) -> usize {
    (2_097_152 / context_length.max(1)).clamp(16, 256)
}

/// Most rows one forward call writes: the largest [`prefill_chunk`], and the
/// widest verification call ([`cpu::MAX_VERIFY_ROWS`]). The sliding-window
/// block pools are sized for it ([`paged::KvLayout::new`]).
pub(crate) fn max_call_rows(context_length: usize) -> usize {
    prefill_chunk(context_length).max(cpu::MAX_VERIFY_ROWS)
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
    /// checkpoint maximum, `device` is compiled out, or the `experts`
    /// budgets do not fit the model; [`CucaError::Provider`] for unsupported
    /// or malformed checkpoints; [`CucaError::Io`] when the file cannot be
    /// read.
    pub(crate) fn load(
        content: &gguf_file::Content,
        path: &Path,
        context_length: usize,
        device: LocalDevice,
        experts: &ExpertPlacement,
    ) -> Result<Self, CucaError> {
        match device {
            LocalDevice::Cpu => Ok(Model::Cpu(cpu::Transformer::load_placed(
                content,
                path,
                context_length,
                experts,
            )?)),
            #[cfg(feature = "provider-local-gpu")]
            device => Ok(Model::Gpu(gpu::GpuTransformer::load_placed(
                content,
                path,
                context_length,
                device,
                experts,
            )?)),
            #[cfg(not(feature = "provider-local-gpu"))]
            device => Err(super::device_unavailable(device).unwrap_or_else(|| {
                CucaError::Config(format!("device {device:?} is not available"))
            })),
        }
    }

    /// Counters and heat of the expert tiers; `None` unless the model loaded
    /// with `ExpertPlacement::Tiered`.
    pub(crate) fn expert_tiers(&self) -> Option<std::sync::Arc<tier::TierShared>> {
        match self {
            Model::Cpu(m) => m.expert_tiers(),
            #[cfg(feature = "provider-local-gpu")]
            Model::Gpu(m) => m.expert_tiers(),
        }
    }

    /// Run `tokens` through the model at positions `start_pos..`, storing
    /// their K/V in the blocks of `tables` (one block table per K/V group of
    /// [`Model::kv_layout`], holding every block the call writes); returns
    /// the logits `want` selects and the final-norm hidden rows `hidden`
    /// selects (one per logits row).
    ///
    /// `start_pos + tokens.len()` must not exceed the context length. With
    /// `exact`, every row equals what a single-token step computes bit for
    /// bit, which speculative verification needs: guaranteed on the CPU; on
    /// a GPU the rows agree only up to float rounding.
    ///
    /// # Errors
    ///
    /// [`CucaError::Provider`] for a rejected call or a failed forward pass.
    pub(crate) fn forward_ext(
        &mut self,
        tokens: &[u32],
        start_pos: usize,
        tables: &[paged::BlockTable],
        want: Want,
        hidden: cpu::Hidden,
        exact: bool,
    ) -> Result<cpu::Forward, CucaError> {
        match self {
            Model::Cpu(m) => m.forward_seq(tokens, start_pos, tables, want, hidden, exact),
            #[cfg(feature = "provider-local-gpu")]
            Model::Gpu(m) => m.forward_seq(tokens, start_pos, tables, want, hidden, exact),
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

    /// Take the model state back to at most `len` positions and return the
    /// position it now sits at; the caller truncates its sequence's block
    /// tables to it. An attention-only model keeps no state besides the K/V
    /// blocks and returns `len`. A hybrid model (recurrent layers) retains
    /// only positions it holds a state for: the current end, the
    /// [`Model::checkpoint`] and the rows of the last multi-row `Want::All`
    /// call; otherwise it falls back to the largest of those not above `len`
    /// (`0`: recompute from scratch). See [`cpu::Transformer::rewind`] and
    /// `Cache rollback` in the module docs.
    pub(crate) fn rewind(&mut self, len: usize) -> usize {
        match self {
            Model::Cpu(m) => m.rewind(len),
            #[cfg(feature = "provider-local-gpu")]
            Model::Gpu(m) => m.rewind(len),
        }
    }

    /// Keep the recurrent state at the current end of the cache as the one
    /// checkpoint [`Model::rewind`] can return to, replacing the previous
    /// one. A no-op for models that can truncate anywhere (attention-only).
    /// On a GPU the checkpoint is a set of handles to the current immutable
    /// state tensors, so it costs nothing.
    pub(crate) fn checkpoint(&mut self) {
        match self {
            Model::Cpu(m) => m.checkpoint(),
            #[cfg(feature = "provider-local-gpu")]
            Model::Gpu(m) => m.checkpoint(),
        }
    }

    /// Whether the model keeps a recurrent state next to its K/V cache (gated
    /// delta-net layers): its state can then return only to the positions
    /// [`Model::rewind`] lists.
    pub(crate) fn is_hybrid(&self) -> bool {
        match self {
            Model::Cpu(m) => m.is_hybrid(),
            #[cfg(feature = "provider-local-gpu")]
            Model::Gpu(m) => m.is_hybrid(),
        }
    }

    /// Most rows of one `Want::All` forward call after which every
    /// intermediate length can still be returned to with
    /// [`Model::rewind`]: unbounded unless the model is hybrid. A hybrid GPU
    /// model keeps no verify record (1: only the end of a call).
    pub(crate) fn rollback_rows(&self) -> usize {
        match self {
            Model::Cpu(m) => m.rollback_rows(),
            #[cfg(feature = "provider-local-gpu")]
            Model::Gpu(m) => m.rollback_rows(),
        }
    }

    /// The K/V block pools' layout: the groups, and the blocks each holds.
    pub(crate) fn kv_layout(&self) -> &paged::KvLayout {
        match self {
            Model::Cpu(m) => m.kv_layout(),
            #[cfg(feature = "provider-local-gpu")]
            Model::Gpu(m) => m.kv_layout(),
        }
    }

    /// Bytes of K/V data the block pools hold.
    pub(crate) fn kv_pool_bytes(&self) -> u64 {
        match self {
            Model::Cpu(m) => m.kv_pool_bytes(),
            #[cfg(feature = "provider-local-gpu")]
            Model::Gpu(m) => m.kv_pool_bytes(),
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
}
