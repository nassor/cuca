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
//!   per-head QK-norm, a per-layer score scale, and causal or
//!   sliding-window masks (per-layer pattern);
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
//! retained, and a hybrid model retains only positions it holds a state for
//! (each running sequence owns a state slot, [`StateSlot`]):
//!
//! - the slot's current end (the live state);
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
//! Any other requested length falls back to `0` (recompute everything, from
//! the zero state). A forward call that fails half-way empties the slots of
//! its sequences (a partly advanced state cannot be repaired). A prefill in
//! any chunking, a token-by-token run and a restore-and-replay produce the
//! same logits and state bit for bit.
//!
//! A **snapshot** ([`Model::snapshot`], [`Model::restore`]) is a copy of one
//! slot's state, which the engine's prefix store keeps next to the K/V
//! blocks of the same prefix. The GPU engine follows the same rules for the
//! live state and the snapshots (its states are immutable device tensors, so
//! a snapshot is a set of handles and costs nothing) but keeps no verify
//! record ([`Model::rollback_rows`] is 1 for a hybrid GPU model): on a GPU the
//! batched rows equal single steps only up to float rounding, and no drafter
//! can pair with a hybrid model there.
//!
//! The engine consumes the returned length: it truncates its sequence's block
//! tables to the retained prefix, prefills from there and reports that as the
//! reused prompt tokens. Every model looks a prompt up in the prefix store
//! ([`paged::KvPools::lookup`]); a hybrid model also needs the snapshot of
//! the position, so it resumes from the nearest block boundary it has one
//! for. While prefilling a prompt the engine snapshots a hybrid model's state
//! at the block boundary below the last end-of-turn token (the end of the
//! last complete message: the generation prompt that follows is what chat
//! templates rewrite on the next turn; a prompt without one is snapshotted
//! below its end), so the next turn of a conversation restarts from there and
//! the output equals a cold run's.
//!
//! # Batching
//!
//! [`Model::forward_batch`] runs the rows of several sequences ([`Batch`],
//! one [`SeqRows`] per sequence) in one pass: embeddings, projections, FFN and
//! the LM head see the concatenation of all rows (each weight is read once
//! per step), while attention, the K/V block writes and a hybrid model's
//! recurrent scan stay per sequence, against that sequence's block tables and
//! state slot. On the CPU, with `exact`, a sequence's rows are bit-identical
//! to running it alone.
//!
//! # Memory
//!
//! - Weights stay quantized (any dtype candle's quantized matmul reads) and
//!   are consumed by the row-batched kernel of [`qmm`] (Q4_0, Q4_1, Q5_0,
//!   Q5_1 and Q8_0) or by candle's quantized matmul (the K-quants Q2_K to
//!   Q6_K and any other dtype, one activation row at a time); norm vectors,
//!   biases and float-typed matmul
//!   weights (including `per_layer_model_proj`, BF16 in the published Gemma 4
//!   files) are held as f32. Resident weights are about the file size.
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
//!   that own K/V store them, in pools of blocks of 16 positions, 8 bytes
//!   times 16 times `n_kv_head` times `head_dim` per block (f32 keys and
//!   values) per K/V-owning layer, with that layer's own head count and head
//!   dimension. Layers with the same window share block ids (a group). A
//!   full-attention group holds `kv_tokens` positions (twice
//!   `context_length` by default) shared by every running sequence and the
//!   cached prefixes, so a finished request's blocks stay cached while the
//!   next runs; a sliding-window group holds `ceil((window - 1 + rows) / 16) +
//!   2` blocks per running sequence plus up to `kv_tokens - context_length`
//!   positions of cached prefixes ([`paged::KvLayout::new`];
//!   [`paged::KvLayout::pool_bytes`] sums it all). A sliding-window block is
//!   released once every position in it is before the window of the next
//!   query; the rollback of a draft round (at most 16 rows past the accepted
//!   length) never needs a released block. Prefix hits shorten by whole
//!   blocks until every group still holds the blocks its window needs.
//! - A hybrid model's recurrent state is independent of the context length:
//!   `Hparams::recurrent_state_bytes` is 4 bytes times the sum, over
//!   the delta-net layers, of `(conv_kernel - 1) * conv_dim + n_v_heads *
//!   head_k_dim * head_v_dim` (Qwen3.5-0.8B: 18 layers of 1.12 MB, 20.2 MB).
//!   Every state slot (one per sequence that can run, `max_sequences`) keeps
//!   that state live and, once a model verifies drafts, the verify record: one
//!   more copy and `4 * (conv_dim + 2 * n_v_heads)` bytes per verified row per
//!   layer (at most 17 rows: 445 KB per row over the 18 layers of the 0.8B
//!   model, 7.6 MB for 17). The prefix store keeps one more copy per snapshot
//!   (at most `max_sequences + 4`).
//! - The K/V cache of a hybrid model covers its attention layers only:
//!   `kv_cache_bytes` counts just the K/V-owning layers (24,576 bytes per
//!   position for the 0.8B model: 6 attention layers with 2 KV heads of
//!   256).
//! - The K/V cache of a latent-attention model (Mistral Small 4) is one row of
//!   `kv_lora + rot` floats per position and layer (`4 * 16 * (kv_lora +
//!   rot)` bytes per block and layer, from [`params::LayerSpec::kv_floats_per_position`]):
//!   1,280 bytes per position and layer, 46,080 over its 36 layers. The value
//!   is the first `kv_lora` columns of the key row, so the pool holds no values.
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
mod pool;
pub(crate) mod qmm;
pub(crate) mod quant;
pub(crate) mod rope;
pub(crate) mod split;
pub(crate) mod tier;
#[cfg(test)]
pub(crate) mod tiny;
#[cfg(feature = "provider-local-gpu")]
pub(crate) mod tp;

use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use candle_core::Device;
use candle_core::quantized::gguf_file;

use self::layout::name;
use self::params::Hparams;
use self::rope::{CosSin, Factors};
use super::{ExpertPlacement, ExpertTiers, LocalDevice};
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

/// Which recurrent-state slot of a hybrid model a sequence's rows advance:
/// each running sequence of a hybrid model owns one (`0..max_sequences`).
/// An attention-only model has no state and ignores it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct StateSlot(pub(crate) usize);

/// The rows one sequence contributes to a forward step.
pub(crate) struct SeqRows<'a> {
    /// The tokens, at positions `start_pos..`.
    pub(crate) tokens: &'a [u32],
    pub(crate) start_pos: usize,
    /// The sequence's block tables, one per K/V group of the model, holding
    /// every block the rows write.
    pub(crate) tables: &'a [paged::BlockTable],
    /// The recurrent state the rows advance (hybrid models).
    pub(crate) state: StateSlot,
    /// Which logits the step returns for this sequence.
    pub(crate) want: Want,
    /// Which final-norm hidden rows it returns next to them.
    pub(crate) hidden: cpu::Hidden,
}

/// One forward step over the rows of several sequences, see
/// [`Model::forward_batch`].
pub(crate) struct Batch<'a> {
    pub(crate) rows: Vec<SeqRows<'a>>,
    /// Every row equals, bit for bit, what a single-token step computes
    /// (guaranteed on the CPU; on a GPU the rows agree only up to float
    /// rounding).
    pub(crate) exact: bool,
}

impl<'a> Batch<'a> {
    /// A step over one sequence in state slot 0 (the unit tests).
    #[cfg(test)]
    pub(crate) fn single(
        tokens: &'a [u32],
        start_pos: usize,
        tables: &'a [paged::BlockTable],
        want: Want,
        hidden: cpu::Hidden,
        exact: bool,
    ) -> Self {
        Self {
            rows: vec![SeqRows {
                tokens,
                start_pos,
                tables,
                state: StateSlot(0),
                want,
                hidden,
            }],
            exact,
        }
    }
}

/// The recurrent state of a hybrid model at one position (a copy of every
/// delta-net layer's state), kept by the prefix store next to the K/V blocks
/// of the same prefix. On a GPU the state tensors are immutable, so a
/// snapshot is a set of handles.
pub(crate) enum RecurrentSnapshot {
    Cpu(Vec<delta::DeltaState>),
    #[cfg(feature = "provider-local-gpu")]
    Gpu(gpu::GpuSnapshot),
    /// Both halves of a model with layers offloaded to the GPU.
    #[cfg(feature = "provider-local-gpu")]
    Split(split::SplitSnapshot),
}

impl RecurrentSnapshot {
    /// Bytes the snapshot holds.
    pub(crate) fn bytes(&self) -> usize {
        match self {
            RecurrentSnapshot::Cpu(layers) => layers.iter().map(delta::DeltaState::bytes).sum(),
            #[cfg(feature = "provider-local-gpu")]
            RecurrentSnapshot::Gpu(snapshot) => snapshot.bytes(),
            #[cfg(feature = "provider-local-gpu")]
            RecurrentSnapshot::Split(snapshot) => snapshot.bytes(),
        }
    }
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
///
/// # Errors
///
/// [`CucaError::Provider`] for a DFlash drafter file (it needs a target, see
/// [`mtp::Drafter`]) or whatever [`Hparams::from_metadata`] rejects.
pub(crate) fn read_hparams(content: &gguf_file::Content) -> Result<Hparams, CucaError> {
    if content
        .metadata
        .get("general.architecture")
        .and_then(|v| v.to_string().ok())
        .is_some_and(|a| a == cpu::dflash::ARCH)
    {
        return Err(err(format!(
            "`{}` is a block-diffusion drafter, not a standalone model: pass it as the `drafter` of the model it was trained for",
            cpu::dflash::ARCH
        )));
    }
    Hparams::from_metadata(&content.metadata, &|n| content.tensor_infos.contains_key(n))
}

/// The placement of a model that runs entirely on the CPU, from one set for
/// a GPU device: the same tiers without the device tier.
fn cpu_experts(experts: &ExpertPlacement) -> ExpertPlacement {
    match experts {
        // The CPU keeps every expert in RAM already.
        ExpertPlacement::Resident | ExpertPlacement::Host { .. } => ExpertPlacement::Resident,
        ExpertPlacement::Tiered(tiers) => ExpertPlacement::Tiered(ExpertTiers {
            device_bytes: 0,
            ..tiers.clone()
        }),
    }
}

/// The loaded model on the device `LocalModelConfig::device` selected: candle
/// on the CPU, burn on a GPU, or both with some layers offloaded to the GPU
/// ([`split`]). All expose the same calls the engine makes.
// One `Model` exists per engine, so the size gap between the variants costs
// nothing worth a box.
#[allow(clippy::large_enum_variant)]
pub(crate) enum Model {
    Cpu(cpu::Transformer),
    #[cfg(feature = "provider-local-gpu")]
    Gpu(gpu::GpuTransformer),
    #[cfg(feature = "provider-local-gpu")]
    Split(split::SplitModel),
}

/// Run `$call` with `$m` bound to the engine of whichever device the model
/// is on.
macro_rules! on_model {
    ($model:expr, $m:ident => $call:expr) => {
        match $model {
            Model::Cpu($m) => $call,
            #[cfg(feature = "provider-local-gpu")]
            Model::Gpu($m) => $call,
            #[cfg(feature = "provider-local-gpu")]
            Model::Split($m) => $call,
        }
    };
}

impl Model {
    /// Load the model from the GGUF file at `path`, whose index `content` was
    /// read already, onto `device`, with a K/V cache for `context_length`
    /// positions; `shards` are the devices after `device` that share the
    /// model (tensor parallelism, GPU only: empty for one device); `sizing`
    /// says how many sequences the K/V pools and the recurrent-state slots
    /// serve.
    ///
    /// # Errors
    ///
    /// [`CucaError::Config`] when `context_length` is zero or above the
    /// checkpoint maximum, `device` is compiled out, or the `experts`
    /// budgets do not fit the model; [`CucaError::Provider`] for unsupported
    /// or malformed checkpoints; [`CucaError::Io`] when the file cannot be
    /// read.
    #[expect(
        clippy::too_many_arguments,
        reason = "the file, the devices, the placements and the sizing are independent inputs"
    )]
    pub(crate) fn load(
        content: &gguf_file::Content,
        path: &Path,
        context_length: usize,
        device: LocalDevice,
        shards: &[LocalDevice],
        experts: &ExpertPlacement,
        sizing: &paged::Sizing,
        gpu_layers: Option<usize>,
    ) -> Result<Self, CucaError> {
        if let Some(n) = gpu_layers {
            return Self::load_offloaded(
                content,
                path,
                context_length,
                device,
                shards,
                experts,
                sizing,
                n,
            );
        }
        match device {
            LocalDevice::Cpu => Ok(Model::Cpu(cpu::Transformer::load_placed(
                content,
                path,
                context_length,
                experts,
                sizing,
            )?)),
            #[cfg(feature = "provider-local-gpu")]
            device => Ok(Model::Gpu(gpu::GpuTransformer::load_placed(
                content,
                path,
                context_length,
                device,
                shards,
                experts,
                sizing,
            )?)),
            #[cfg(not(feature = "provider-local-gpu"))]
            device => {
                // Shards are GPU devices: `validate` rejects them without one.
                let _ = shards;
                Err(super::device_unavailable(device).unwrap_or_else(|| {
                    CucaError::Config(format!("device {device:?} is not available"))
                }))
            }
        }
    }

    /// [`Model::load`] with `gpu_layers` layers on `device` and the rest on
    /// the CPU (see [`split`]).
    ///
    /// # Errors
    ///
    /// [`CucaError::Config`] for more layers than the model has, `shards`
    /// (tensor parallelism cannot be combined with layer offload), a CPU
    /// `device`, or a GPU compiled out; otherwise as [`Model::load`].
    #[expect(
        clippy::too_many_arguments,
        reason = "the file, the devices, the placements and the sizing are independent inputs"
    )]
    fn load_offloaded(
        content: &gguf_file::Content,
        path: &Path,
        context_length: usize,
        device: LocalDevice,
        shards: &[LocalDevice],
        experts: &ExpertPlacement,
        sizing: &paged::Sizing,
        gpu_layers: usize,
    ) -> Result<Self, CucaError> {
        if !shards.is_empty() {
            return Err(CucaError::Config(
                "gpu_layers cannot be combined with tensor parallelism (`shards`)".into(),
            ));
        }
        if device == LocalDevice::Cpu {
            return Err(CucaError::Config(
                "gpu_layers needs a GPU `device`, got cpu".into(),
            ));
        }
        let hp = read_hparams(content)?;
        let placement = split::plan(&hp, gpu_layers)?;
        match placement {
            // Nothing on the GPU: a CPU model, no device opens.
            split::Placement::Cpu => Self::load(
                content,
                path,
                context_length,
                LocalDevice::Cpu,
                &[],
                &cpu_experts(experts),
                sizing,
                None,
            ),
            split::Placement::Gpu => Self::load(
                content,
                path,
                context_length,
                device,
                &[],
                experts,
                sizing,
                None,
            ),
            #[cfg(feature = "provider-local-gpu")]
            split::Placement::Split { first } => Ok(Model::Split(split::SplitModel::load(
                content,
                path,
                context_length,
                device,
                experts,
                sizing,
                first,
            )?)),
            #[cfg(not(feature = "provider-local-gpu"))]
            split::Placement::Split { .. } => Err(super::device_unavailable(device)
                .unwrap_or_else(|| {
                    CucaError::Config(format!("device {device:?} is not available"))
                })),
        }
    }

    /// Layers on the GPU and layers in all when layers are offloaded
    /// ([`split`]); `None` for a model on one device.
    pub(crate) fn layer_split(&self) -> Option<(usize, usize)> {
        match self {
            Model::Cpu(_) => None,
            #[cfg(feature = "provider-local-gpu")]
            Model::Gpu(_) => None,
            #[cfg(feature = "provider-local-gpu")]
            Model::Split(m) => Some((m.gpu_layers(), m.n_layer())),
        }
    }

    /// Counters and heat of the expert tiers; `None` unless the model loaded
    /// with `ExpertPlacement::Tiered`.
    pub(crate) fn expert_tiers(&self) -> Option<std::sync::Arc<tier::TierShared>> {
        on_model!(self, m => m.expert_tiers())
    }

    /// Run the rows of several sequences through the model in one pass (a
    /// forward step): each sequence's `tokens` at positions `start_pos..`,
    /// their K/V stored in the blocks of its `tables` (one block table per
    /// K/V group of [`Model::kv_layout`], holding every block the rows
    /// write). Returns, per sequence, the logits `want` selects and the
    /// final-norm hidden rows `hidden` selects (one per logits row), plus a
    /// [`RecurrentSnapshot`] when asked. The dense layers read each weight
    /// once for all rows of the step; a sequence's rows do not depend on the
    /// other sequences in the step (bit for bit on the CPU with `exact`, up to
    /// float rounding on a GPU).
    ///
    /// Each sequence's `start_pos + tokens.len()` must not exceed the context
    /// length. With `exact`, every row equals what a single-token step
    /// computes bit for bit, which speculative verification needs:
    /// guaranteed on the CPU; on a GPU the rows agree only up to float
    /// rounding.
    ///
    /// # Errors
    ///
    /// [`CucaError::Provider`] for a rejected call or a failed forward pass.
    pub(crate) fn forward_batch(
        &mut self,
        batch: &Batch<'_>,
    ) -> Result<Vec<cpu::Forward>, CucaError> {
        on_model!(self, m => m.forward_batch(batch))
    }

    /// The architecture.
    pub(crate) fn arch(&self) -> arch::Arch {
        on_model!(self, m => m.arch())
    }

    /// Take the recurrent state of `slot` back to at most `len` positions and
    /// return the position it now sits at; the caller truncates its
    /// sequence's block tables to it. An attention-only model keeps no state
    /// besides the K/V blocks and returns `len`. A hybrid model (recurrent
    /// layers) retains only positions it holds a state for: the slot's current
    /// end and the rows of its last multi-row `Want::All` call; otherwise it
    /// falls back to `0` (the zero state: recompute from scratch, or
    /// [`Model::restore`] a snapshot). See [`cpu::Transformer::rewind`] and
    /// `Cache rollback` in the module docs.
    pub(crate) fn rewind(&mut self, slot: StateSlot, len: usize) -> usize {
        on_model!(self, m => m.rewind(slot.0, len))
    }

    /// A copy of the recurrent state of `slot` (after the rows of its last
    /// forward call): the engine's prefix store keeps it next to the K/V
    /// blocks of the same prefix. `None` for an attention-only model.
    pub(crate) fn snapshot(&self, slot: StateSlot) -> Option<RecurrentSnapshot> {
        match self {
            Model::Cpu(m) => m.snapshot(slot.0).map(RecurrentSnapshot::Cpu),
            #[cfg(feature = "provider-local-gpu")]
            Model::Gpu(m) => m.snapshot(slot.0).map(RecurrentSnapshot::Gpu),
            #[cfg(feature = "provider-local-gpu")]
            Model::Split(m) => m.snapshot(slot.0).map(RecurrentSnapshot::Split),
        }
    }

    /// Make `snapshot` (taken by [`Model::snapshot`] on a model of the same
    /// device) the recurrent state of `slot`, now at
    /// position `len`; `None` is the zero state at position 0. A no-op for
    /// attention-only models.
    ///
    /// # Errors
    ///
    /// [`CucaError::Provider`] for a snapshot that does not belong to this
    /// model.
    pub(crate) fn restore(
        &mut self,
        slot: StateSlot,
        snapshot: Option<&RecurrentSnapshot>,
        len: usize,
    ) -> Result<(), CucaError> {
        if !self.is_hybrid() {
            return Ok(());
        }
        match (self, snapshot) {
            (Model::Cpu(m), None) => {
                m.reset_slot(slot.0);
                Ok(())
            }
            (Model::Cpu(m), Some(RecurrentSnapshot::Cpu(layers))) => m.restore(slot.0, layers, len),
            #[cfg(feature = "provider-local-gpu")]
            (Model::Gpu(m), None) => {
                m.reset_slot(slot.0);
                Ok(())
            }
            #[cfg(feature = "provider-local-gpu")]
            (Model::Gpu(m), Some(RecurrentSnapshot::Gpu(snapshot))) => {
                m.restore(slot.0, snapshot, len)
            }
            #[cfg(feature = "provider-local-gpu")]
            (Model::Split(m), None) => {
                m.reset_slot(slot.0);
                Ok(())
            }
            #[cfg(feature = "provider-local-gpu")]
            (Model::Split(m), Some(RecurrentSnapshot::Split(snapshot))) => {
                m.restore(slot.0, snapshot, len)
            }
            #[cfg(feature = "provider-local-gpu")]
            _ => Err(err("the snapshot comes from another device")),
        }
    }

    /// Whether the model keeps a recurrent state next to its K/V cache (gated
    /// delta-net layers): its state can then return only to the positions
    /// [`Model::rewind`] lists.
    pub(crate) fn is_hybrid(&self) -> bool {
        on_model!(self, m => m.is_hybrid())
    }

    /// Most rows of one `Want::All` forward call after which every
    /// intermediate length can still be returned to with
    /// [`Model::rewind`]: unbounded unless the model is hybrid. A hybrid GPU
    /// model keeps no verify record (1: only the end of a call).
    pub(crate) fn rollback_rows(&self) -> usize {
        on_model!(self, m => m.rollback_rows())
    }

    /// The K/V block pools' layout: the groups, and the blocks each holds.
    pub(crate) fn kv_layout(&self) -> &paged::KvLayout {
        on_model!(self, m => m.kv_layout())
    }

    /// Bytes of K/V data the block pools hold.
    pub(crate) fn kv_pool_bytes(&self) -> u64 {
        on_model!(self, m => m.kv_pool_bytes())
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
