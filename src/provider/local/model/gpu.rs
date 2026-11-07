//! The generic transformer forward pass on a GPU through burn (CubeCL),
//! loaded from the same GGUF file as the CPU engine.
//!
//! [`GpuTransformer`] mirrors [`cpu::Transformer`](super::cpu::Transformer)
//! operation for operation: the same hyperparameters ([`Hparams`]) and
//! per-layer tensor layouts ([`LayerLayout`]), the same RoPE tables
//! ([`build_rope_tables`]), masks ([`attention_mask`]), prefill chunking,
//! K/V sharing and MoE routing ([`mk::route`]). Every architecture of the
//! [`arch`](super::arch) table runs through it: separate or fused
//! `attn_qkv`, Q/K/V/O biases, QK-norm, post-norms, sliding windows,
//! adjacent-pair (`NORM`) and half-split (`NEOX`) RoPE with partial rotary
//! dimensions and NoPE layers, a score scale,
//! dense gated FFNs (fused or separate gate/up, SiLU or GELU) and top-k
//! mixtures of experts, embedding / residual / logit scales, a final logit
//! soft-cap, tied or separate output head, and the Gemma 4 extras (shared
//! K/V layers, per-layer embeddings, weightless value norm, per-layer output
//! scale). GELU is the tanh approximation, as in candle.
//!
//! # Weights
//!
//! - Q4_0, Q4_1 and Q8_0 matmul weights are repacked once at load into the
//!   `u32` words of the dense kernels of [`mk`] ([`mk::DenseLayout`]: block
//!   values then f16 scales, and minimums for Q4_1, in block-major order so
//!   that the threads of a launch, one per output row, read consecutive
//!   blocks) and stay quantized on the device, one copy per weight. Up to
//!   [`GEMV_ROWS`] input rows (decode, batched decode, verification) the
//!   matrix-vector kernel of [`mk::dense_gemv`] dots the blocks with the
//!   input in f32; more rows (prefill) dequantize slices of the weight to f32
//!   ([`mk::dense_dequant`]) and use burn's float matmul. burn's own
//!   quantized matmul is not used by any weight (see [`linear`]).
//! - F32, F16 and BF16 matmul weights are dequantized on the host row by row
//!   ([`dequant_row`]) and uploaded as f32 for burn's float matmul, exactly.
//!   They cost 4 bytes per element on the device whatever their size in the
//!   file (an F16 or BF16 weight doubles).
//! - Every other quantized dtype (Q5_0, Q5_1, Q8_1 and the K-quants Q2_K
//!   to Q6_K) is dequantized on the host row by row and requantized to Q8_0
//!   ([`quantize_q8_0`]), then packed like a native Q8_0 weight, so it costs
//!   about 1.06 bytes per element on the device instead of 4: a 3.8
//!   B-parameter Q4_K_M file needs about 4 GB rather than 15 GB. The price is
//!   a rounding error: each element moves by at most half a Q8_0 step, that
//!   is `block_max / 254` (plus the f16 rounding of the scale, below 0.1% of
//!   the block maximum). Q8_0 resolves a block with 255 levels where Q4_K,
//!   Q5_K and Q6_K resolve theirs with about 15, 31 and 63, so this extra
//!   error is below the one the file's own quantization already carries.
//!   Logits therefore differ from a host-dequantized reference by far more
//!   than float rounding (about 1e-2 of their magnitude in the tiny-model
//!   tests), so a K-quant model decodes the CPU engine's tokens except where
//!   two candidates are that close.
//! - Norm vectors, biases and scalars are f32 whatever their stored dtype.
//!
//! # Memory
//!
//! - Device memory holds the matmul weights as above. The token embedding is
//!   uploaded only when it doubles as the LM head (no `output.weight`), once,
//!   as the head weight. A weight over 256 MiB on the device is split into row
//!   chunks (a large tied head becomes several `[rows_i, n_embd]` weights);
//!   [`linear`] multiplies by each chunk and concatenates the outputs, which
//!   is exact, so no single weight buffer exceeds [`MAX_WEIGHT_CHUNK_BYTES`]
//!   (and no buffer at all exceeds [`MAX_PART_BYTES`]).
//! - Embedding lookups never touch the device copy: the host reads each
//!   token's row of `token_embd` (and, for Gemma 4, of `per_layer_token_embd`)
//!   from the GGUF file ([`RowTable`]) and uploads the dequantized rows. The
//!   per-layer table is never uploaded.
//! - Resident experts (the default) are read, converted to the MoE kernels'
//!   words ([`mk`], not burn's quantized layout) and uploaded as one pool per
//!   layer, expert `e` in slot `e`: the host buffer is one layer's pool and the
//!   device memory is the file's bytes for Q4_0, Q4_1 and Q8_0 experts.
//!   `ExpertPlacement::Tiered` reads no expert at load: see "Tiered experts"
//!   below.
//! - The K/V cache lives on the device as a paged cache (see the
//!   [`paged`](super::paged) module docs): per K/V-owning layer one pool of
//!   `k` and `v` tensors `[n_blocks, n_kv_head, 16, head_dim]` (f32, zeroed
//!   once at load, `n_blocks` from [`KvLayout::sized`] for the load's
//!   [`Sizing`]), and each sequence's block tables, one per group of layers
//!   with one window, map a position to its block and row. A pass writes
//!   its rows block by block with `slice_assign` into the pool, after taking
//!   the layer's tensors out of the cache so the write is in place, and
//!   attention views are dropped before the next write, so no write copies a
//!   pool. One query row (a decode step, every drafter layer) reads the
//!   blocks in place with the CubeCL kernel of [`paged_kernel`] (an online
//!   softmax over the visible keys), and a step whose sequences are all
//!   single rows does it for all of them in one launch per layer; several
//!   rows (prefill, verification) gather the visible blocks of the table
//!   into one contiguous view and run burn's matmul attention with the mask.
//!   The engine owns the allocation of blocks and the tables; this module
//!   only checks that every block a pass touches is present.
//! - A hybrid model keeps one recurrent state per sequence slot
//!   (`Sizing::max_sequences` of them): per delta-net layer a conv history
//!   and the delta-rule matrix, immutable device tensors, so
//!   [`GpuTransformer::snapshot`] and [`GpuTransformer::restore`] only move
//!   handles.
//! - RoPE cos/sin tables for `context_length` positions, one pair per distinct
//!   rotary embedding, live on the device.
//! - During load the host holds the read buffer of the whole tensor (its file
//!   size), the converted copy of one chunk (f32 or Q8_0, at most 1 GiB of
//!   device bytes) and the repacked copy of that chunk; per forward pass it
//!   holds the embedding rows, the attention masks and, per MoE layer, the
//!   router logits (`tokens * n_expert` f32).
//! - Dropping a [`GpuTransformer`] or [`GpuDrafter`] waits for the device
//!   and returns the freed pages to the driver (`Device::memory_cleanup`);
//!   burn otherwise keeps freed pages in its pools, which starves a model
//!   loaded later in the same process.
//!
//! # Batched steps
//!
//! [`GpuTransformer::forward_batch`] runs the rows of several sequences in
//! one pass. Everything dense (embeddings, projections, norms, FFN or MoE,
//! the LM head over the rows the sequences want) sees the concatenated rows,
//! so each weight is read once per step; RoPE positions, K/V block writes,
//! sliding windows and attention are per sequence, and a delta-net layer
//! runs its convolution and recurrence per sequence over its row range from
//! the sequence's own slot. A sequence's rows equal what a pass over that
//! sequence alone computes up to float rounding (the matmul kernels differ
//! by row count).
//!
//! # MoE
//!
//! A MoE layer is the router matmul (a burn operation), then the kernels of
//! [`mk`]: softmax and top-`n_used` of the logits on the device, `act(gate .
//! x) * (up . x)` for every (token, expert) pair, the routing-weighted down
//! projections into a scratch, and a sum over each token's experts in expert
//! order. The kernels read the weights from the layer's pool, a `u32` tensor
//! of expert slots, indexed by the router's own output, so a resident layer
//! needs no host readback and a whole token of resident MoE layers queues
//! without a wait. Routing weights are applied inside the kernel. The shared
//! expert and everything outside the experts stay burn operations.
//!
//! The kernels exist because burn's quantized matmuls cannot gather a
//! data-dependent subset of experts (`q_select` dequantizes) and because a
//! per-expert matmul triple is host-bound by the per-operation launch cost.
//! The dense weights use the same layout and dot-row arithmetic through
//! [`mk::dense_gemv`].
//!
//! # Tiered experts
//!
//! With `ExpertPlacement::Tiered` everything above stays resident except the
//! experts, which stay in the GGUF file and reach the device on demand (the
//! tiers, the heat and the eviction policy are described in the
//! [`tier`](super::tier) module docs). Placement only decides speed: an
//! expert's words are built by the same conversion ([`mk::pack_expert`]) as a
//! resident one, and the kernels' result does not depend on how a pass groups
//! its experts, so tiered placements agree bit for bit. A resident load never
//! reads the router output back and a tiered one does at every layer, which
//! splits burn's fused elementwise kernels at other places: they agree to
//! float rounding.
//!
//! - Device tier: per MoE layer a pool of `device_slots` slots
//!   ([`GpuExpertBank::Tiered`]), with `device_slots = min(n_expert,
//!   device_bytes / per_expert)` where `per_expert` is the size of one
//!   expert of every MoE layer in the kernels' layout (the file's bytes for
//!   Q4_0, Q4_1 and Q8_0). A budget below `n_used` experts per layer fails
//!   the load with `CucaError::Config` naming the minimum. The pool is
//!   allocated at load.
//! - RAM tier (`ram_bytes > 0`): the packed host copies of experts
//!   ([`PackedExpert`]: the words of one expert, shared with the uploads). They
//!   are converted on the loader threads, never on the engine thread.
//!   `ram_bytes == 0` disables the tier.
//! - Disk: a pool of loader threads reads the three matrices of a missing
//!   expert with positional reads and converts them.
//!
//! Per MoE layer pass the kernels' routing is read back (ids and weights, a
//! few hundred floats; the one host wait of a tiered layer) and gives the
//! ascending experts to compute. The device hits are counted first; the
//! experts missing on the device walk through the lower tiers in the same
//! ascending order (reads in flight count against the RAM slots). Experts are
//! computed in groups of at most `device_slots`: the pool's hits first, then
//! misses written into a free slot or the slot of the table's victim (the
//! resident expert outside the group with the lowest `(heat, last use)`) by
//! a one-slot kernel on the engine thread (burn's device streams are per
//! thread). A single-token pass is one group. Every group writes its
//! (token, expert) results into one scratch that the last kernel sums in
//! expert order.
//!
//! A warm start (`ExpertTiers::warm_from`) writes the hottest experts of
//! each layer's device tier at load and waits for the device after every
//! layer, so at most one layer's uploads are queued at a time.
//!
//! # Host experts
//!
//! With `ExpertPlacement::Host` the experts of the hosted MoE layers are read
//! into RAM at load ([`cpu::HostExperts`](super::cpu::HostExperts), the CPU
//! engine's resident experts) and [`GpuExpertBank::Host`] runs them: the
//! router and [`mk::route`] run on the device, one readback brings the layer's
//! normed rows and the routing to the host, the CPU computes every selected
//! expert over its own tokens and sums them in ascending expert order, and one
//! upload returns `[t, n_embd]`. Resident layers beyond `layers` keep their
//! experts on the device.
//!
//! # Latent attention
//!
//! A `mistral4` layer ([`Weights::mla`]) caches one `[latent | rotary key]`
//! row per position in a pool `[blocks, 1, 16, kv_lora + rot]` with no value
//! pool, absorbs `k_b` into the query with one batched matmul, runs the paged
//! decode kernel (key width 320, value width 256, the pool passed as both) or
//! the gather path, and applies `v_b` to the attended latent.
//!
//! # Layer offload
//!
//! [`GpuTransformer::load_back`] loads the last layers and the head
//! ([`Back`]); [`GpuTransformer::forward_back`] takes the residual stream the
//! CPU layers produced and the rows of the K/V sources it holds
//! ([`MirrorRows`](super::split::MirrorRows)); see [`split`](super::split).
//!
//! # Speculative drafting
//!
//! [`GpuTransformer::forward_ext`] returns the final-norm hidden rows next to
//! the logits, and [`GpuDrafter`] drafts tokens from a Gemma 4
//! multi-token-prediction drafter exactly as the CPU `mtp::Drafter` does: the
//! drafter's layers attend to the target's K/V cache, which they reach through
//! clones of its tensor handles taken for the length of one `draft` call (a
//! handle held across a target forward would turn its in-place cache writes
//! into whole-cache copies).
//! The GPU does not keep a batched pass bit-equal to single-token steps
//! (`exact` is ignored), so a speculative greedy run can differ from plain
//! greedy where two logits are within float rounding.
//!
//! # Tensor parallelism
//!
//! [`GpuTransformer::load_placed`] takes `shards`, the devices after its
//! `device`: one model split across `world = 1 + shards.len()` devices, each
//! layer's attention and FFN sliced across them with two all-reduces per
//! layer. A device may repeat (`vulkan:0+vulkan:0`: two shards on one GPU are
//! still split and reduced, which tests the path without a second GPU but
//! saves nothing). The split itself (which heads, rows and columns a shard
//! owns, the rules a model must meet, the all-reduce) is described in
//! [`tp`]; the rules are checked before any weight is read and fail with
//! `CucaError::Config("tensor parallelism: <rule> (<numbers>)")`. Hybrid
//! (delta-net) models, latent attention, a drafter and tiered or host experts
//! are not supported with shards. What this module does for it:
//!
//! - Weights per shard: one [`Weights`] per device, built with the
//!   shard's own hyperparameters (its head counts) from sliced reads
//!   ([`WeightReader::matrix_cut`]: a shard reads only its rows from the
//!   file and copies its columns' bytes out of whole rows; a column cut that
//!   splits a block of the file's dtype, as a K-quant `ffn_down` can, is cut
//!   from the dequantized rows and held as Q8_0). Only shard 0 holds
//!   the LM head and the final norm; norms, per-head Q/K norms, the router,
//!   the shared expert's gate, the per-layer-embedding weights, the layer
//!   output scales and the RoPE tables are replicated, and so is the
//!   `attn_output` bias, added once after the reduction. Q/K/V biases are
//!   cut with their rows.
//! - Experts: every shard holds every expert, each cut to `ff / world`
//!   units (gate and up rows, down columns, before [`mk::pack_expert`]
//!   converts them), and the shared expert is cut the same way. The router
//!   is replicated, so the kernels keep indexing experts by id, and the
//!   weighted sum over experts is linear in the down projection, so the
//!   shards' partial outputs add up exactly.
//! - K/V pools: one set per shard with the shard's own KV heads and the
//!   same blocks and block tables ([`GpuTransformer::kv_pool_bytes`] sums
//!   them: a KV head that is replicated, because the model has fewer KV
//!   heads than shards, is stored on every shard that serves it).
//! - A layer: every shard runs [`Weights::mixer`] (pre-norm, the
//!   projections of its heads, RoPE, K/V writes, paged attention and the
//!   output projection's matmul, no bias), the partial `[t, n_embd]`
//!   outputs are summed ([`tp::Collective::all_reduce_sum`]), every shard
//!   runs [`Weights::attn_tail`] (bias, post-norm, residual add) and
//!   [`Weights::ffn_half`] (pre-norm, its slice of the FFN or MoE), a second
//!   reduction, and [`Weights::ffn_tail`] (post-norm, residual add, the
//!   per-layer-embedding block, the output scale). On one device
//!   [`Weights::layer`] runs the same functions in the same order and
//!   nothing is reduced. The residual stream is replicated: the
//!   embedding rows go to every shard, and the final norm and the LM head
//!   run on shard 0's copy.
//! - The sum is host-staged (every part read back, added in f32 in shard
//!   order, uploaded to every device) except for distinct CUDA devices on
//!   Linux, which use NCCL through burn's `all_reduce`. Two sharded loads
//!   agree bit for bit on the host path; a sharded and a single-device run
//!   agree to float rounding (the reduction order differs). The host path
//!   costs two device-to-host round trips per layer: it is for correctness
//!   and for GPUs without a collective, not for speed.
//! - Memory: each device holds about `1 / world` of the attention,
//!   FFN and expert weights plus the replicated tensors (and shard 0 the
//!   LM head), its own K/V pools and, during a forward pass, the whole
//!   `[t, n_embd]` residual stream. The host reads each shard's tensors in
//!   turn with the buffer of one whole tensor (a row cut reads only its
//!   rows), so the load's host footprint does not grow with `world`.
//!
//! # Why burn's quantized matmul is not used
//!
//! On burn 0.22 a quantized weight on the right of a matmul loses the Vulkan
//! device in some autotune candidates, a single-row product with the weight
//! on the left returns wrong values, the `vecmat` autotune entry is shared
//! by every shape of a power-of-two bucket (a candidate tuned for one shape
//! fails to launch for another), and the autotune key holds the shapes, the
//! element types and the strides but not the quantization scheme, so a
//! candidate tuned for a Q8_0 weight is replayed on a Q4_0 one. The own kernels
//! ([`mk::dense_gemv`], [`mk::dense_dequant`]) have none of these limits.
//! Only the f32 weights (the float dtypes) go through burn's matmul, and
//! they follow the workarounds of [`linear`].

#[cfg(not(target_endian = "little"))]
compile_error!(
    "provider-local-gpu copies GGUF little-endian f16 scales into burn's native-endian layout"
);

use std::collections::HashMap;
use std::f64::consts::FRAC_1_SQRT_2;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::ops::Range;
use std::path::Path;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use burn::tensor::{AllocationProperty, Bytes, DType, Device, Int, Tensor, TensorData, activation};
use candle_core::quantized::{GgmlDType, gguf_file};

use super::arch::{Act, Arch};
use super::cpu::{Forward, Hidden, HostExperts};
use super::layout::{Ffn, LayerLayout, Qkv, ffn_norm_part, name, tensor_info};
use super::moe::Routing;
use super::moe_kernel::{self as mk, MoeCall, SlotLayout};
#[cfg(test)]
use super::paged::Solo;
use super::paged::{BLOCK_SIZE, BlockTable, KvLayout, NO_BLOCK, Sizing};
use super::paged_kernel::{self, PagedCall};
use super::params::{DeltaSpec, Hparams, LayerSpec, MlaSpec, MoeSpec};
use super::quant::{RowTable, dequant_row};
use super::rope::{RopeKind, RopeSpec};
use super::split::MirrorRows;
use super::tier::{
    BuildFn, ExpertSlice, ExpertStore, LayerExperts, SlotTable, TierShared, add, sub, tier_slots,
};
use super::tp::{self, Collective, Cut, Tp};
use super::{Batch, Want, attention_mask, build_rope_tables, check_context, err, read_hparams};
use crate::error::CucaError;
use crate::provider::local::{
    ExpertHeat, ExpertPlacement, ExpertTiers, LocalDevice, device_unavailable,
};

/// The burn device for `device`, opened now so that a missing adapter or
/// driver is an error here rather than a panic at the first tensor.
pub(super) fn open_device(device: LocalDevice) -> Result<Device, CucaError> {
    match device {
        #[cfg(feature = "provider-local-vulkan")]
        LocalDevice::Vulkan(adapter) => open_wgpu(
            device,
            adapter,
            burn::tensor::wgpu::WgpuBackend::Vulkan,
            Device::vulkan,
        ),
        #[cfg(feature = "provider-local-metal")]
        LocalDevice::Metal(adapter) => open_wgpu(
            device,
            adapter,
            burn::tensor::wgpu::WgpuBackend::Metal,
            Device::metal,
        ),
        #[cfg(feature = "provider-local-cuda")]
        LocalDevice::Cuda(index) => {
            require_index(device, burn::tensor::DeviceType::Cuda, index)?;
            Ok(Device::cuda(index))
        }
        #[cfg(feature = "provider-local-rocm")]
        LocalDevice::Rocm(index) => {
            require_index(device, burn::tensor::DeviceType::Rocm, index)?;
            Ok(Device::rocm(index))
        }
        other => Err(device_unavailable(other).unwrap_or_else(|| {
            CucaError::Config(format!("device {other:?} does not run the GPU engine"))
        })),
    }
}

/// Initialize a wgpu-backed device. A second model on the same adapter finds
/// its runtime registered already and shares it through the lazy
/// constructor `shared`.
#[cfg(any(feature = "provider-local-vulkan", feature = "provider-local-metal"))]
fn open_wgpu(
    device: LocalDevice,
    adapter: crate::provider::local::GpuAdapter,
    api: burn::tensor::wgpu::WgpuBackend,
    shared: fn(burn::tensor::DeviceKind) -> Device,
) -> Result<Device, CucaError> {
    use burn::tensor::DeviceKind;
    use burn::tensor::wgpu::WgpuInitError;

    use crate::provider::local::GpuAdapter;

    let kind = || match adapter {
        GpuAdapter::Default => DeviceKind::DefaultDevice,
        GpuAdapter::Discrete(n) => DeviceKind::DiscreteGpu(n),
        GpuAdapter::Integrated(n) => DeviceKind::IntegratedGpu(n),
    };
    match Device::wgpu_options()
        .device_kind(kind())
        .graphics_api(api)
        .init()
    {
        Ok(opened) => Ok(opened),
        Err(WgpuInitError::Registration { .. }) => Ok(shared(kind())),
        Err(e) => Err(err(format!("device {device:?}: {e}"))),
    }
}

/// Fail unless the backend reports a device at `index`.
#[cfg(any(feature = "provider-local-cuda", feature = "provider-local-rocm"))]
fn require_index(
    device: LocalDevice,
    kind: burn::tensor::DeviceType,
    index: usize,
) -> Result<(), CucaError> {
    let found = Device::enumerate(kind).into_vec().len();
    if index >= found {
        return Err(err(format!(
            "device {device:?}: {found} such device(s) found (missing driver or GPU?)"
        )));
    }
    Ok(())
}

/// Largest device buffer one part of a weight may occupy. wgpu bounds one
/// storage buffer by `min(max_storage_buffer_binding_size, max_buffer_size)`
/// of the adapter, which burn does not expose, so a conservative 1 GiB is
/// assumed everywhere.
const MAX_PART_BYTES: usize = 1 << 30;

/// Device bytes of one row chunk of a dense weight. burn's default memory
/// pool carves its allocations from pages as large as the largest allocation
/// it has served, so one 1 GiB weight chunk makes every page 1 GiB and a
/// nearly full card cannot place the next one. Chunks of 256 MiB keep the
/// pages that size.
const MAX_WEIGHT_CHUNK_BYTES: usize = 256 << 20;

/// Tokens of a delta-net scan between two waits for the device (see
/// [`DeltaWeights::forward`]).
const SCAN_DRAIN: usize = 16;

/// A `[rows, cols]` matmul weight on the device as one or more row chunks
/// (one unless the weight would exceed [`MAX_WEIGHT_CHUNK_BYTES`], as a large
/// tied LM head or FFN matrix does); [`linear`] concatenates the chunks' outputs, which is exact.
struct Weight {
    parts: Vec<Part>,
}

/// A weight held in the dense kernels' own layout on the device
/// ([`mk::DenseLayout`]): one `u32` pool.
#[derive(Clone)]
struct Packed {
    pool: Tensor<1, Int>,
    layout: mk::DenseLayout,
}

/// One chunk of a [`Weight`].
#[derive(Clone)]
enum Part {
    /// f32, row-major `[rows, cols]` (the float dtypes).
    Float(Tensor<2>),
    /// Q4_0, Q4_1 or Q8_0 blocks.
    Packed(Packed),
}

impl Part {
    /// Rows of the chunk.
    fn rows(&self) -> usize {
        match self {
            Part::Float(t) => t.dims()[0],
            Part::Packed(p) => p.layout.rows,
        }
    }
}

/// Quantize `values` (a whole number of 32-element blocks) to the GGUF Q8_0
/// layout, appended to `out`: per block an f16 scale `amax / 127` and 32
/// signed bytes. The values are rounded against the f16-rounded scale, so a
/// dequantized element is within half a scale step of the input.
pub(super) fn quantize_q8_0(values: &[f32], out: &mut Vec<u8>) {
    let (blocks, _) = values.as_chunks::<32>();
    for block in blocks {
        let amax = block.iter().fold(0f32, |m, v| m.max(v.abs()));
        let scale = half::f16::from_f32(amax / 127.0);
        let step = scale.to_f32();
        let inverse = if step > 0.0 { 1.0 / step } else { 0.0 };
        out.extend_from_slice(&scale.to_bits().to_le_bytes());
        out.extend(
            block
                .iter()
                .map(|v| (v * inverse).round().clamp(-127.0, 127.0) as i8 as u8),
        );
    }
}

/// The `[rows, cols]` weight whose GGUF bytes are `raw`, on the device, with
/// parts of at most [`MAX_WEIGHT_CHUNK_BYTES`]; see [`upload_matrix_with`].
#[cfg(test)]
fn upload_matrix(
    tensor_name: &str,
    dtype: GgmlDType,
    raw: &[u8],
    rows: usize,
    cols: usize,
    device: &Device,
) -> Result<Weight, CucaError> {
    upload_matrix_with(
        tensor_name,
        dtype,
        raw,
        rows,
        cols,
        device,
        MAX_WEIGHT_CHUNK_BYTES,
    )
}

/// The bytes of the columns `cols` of every row of `raw` (rows `row_len`
/// wide, GGUF dtype `dtype`) appended to `out` as `held` ([`tp::column_dtype`]
/// says which): the sub-slice of every row when `held` is `dtype` and the
/// cut lies on block bounds, else the rows dequantized, cut and quantized to
/// Q8_0.
///
/// # Errors
///
/// [`CucaError::Provider`] for rows that are not whole blocks of `dtype`, a
/// cut past the row, a `held` other than `dtype` or Q8_0, or a Q8_0 cut
/// that is not whole 32-element blocks.
fn append_columns(
    dtype: GgmlDType,
    raw: &[u8],
    row_len: usize,
    cols: &Range<usize>,
    held: GgmlDType,
    out: &mut Vec<u8>,
) -> Result<(), CucaError> {
    if held == dtype
        && let Some(bytes) = tp::copy_columns(dtype, raw, row_len, cols)
    {
        out.extend_from_slice(&bytes);
        return Ok(());
    }
    let row_bytes = row_len / dtype.block_size() * dtype.type_size();
    if row_bytes == 0
        || !row_len.is_multiple_of(dtype.block_size())
        || !raw.len().is_multiple_of(row_bytes)
        || cols.end > row_len
        || held != GgmlDType::Q8_0
        || !cols.len().is_multiple_of(32)
    {
        return Err(err(format!(
            "{} bytes of {dtype:?} do not hold whole rows of {row_len} cut at columns {cols:?} as {held:?}",
            raw.len()
        )));
    }
    let mut values = vec![0f32; row_len];
    for row in raw.chunks_exact(row_bytes) {
        dequant_row(dtype, row, 1.0, &mut values)?;
        quantize_q8_0(&values[cols.clone()], out);
    }
    Ok(())
}

/// How a `[rows, cols]` weight of GGUF dtype `dtype` reaches the device: the
/// layout it takes there and the row chunks it is split into.
struct PackPlan<'a> {
    tensor_name: &'a str,
    dtype: GgmlDType,
    rows: usize,
    cols: usize,
    /// Quantized in the kernels' layout; `false` for f32 (the float dtypes).
    quantized: bool,
    /// File bytes per row.
    row_bytes: usize,
    /// Rows of every chunk but the last, which may be shorter.
    chunk: usize,
}

impl<'a> PackPlan<'a> {
    /// Plan the `[rows, cols]` weight `tensor_name` whose file bytes are
    /// `raw_len` long, in chunks of at most `limit` device bytes.
    fn new(
        tensor_name: &'a str,
        dtype: GgmlDType,
        raw_len: usize,
        rows: usize,
        cols: usize,
        limit: usize,
    ) -> Result<Self, CucaError> {
        let block = dtype.block_size();
        if !cols.is_multiple_of(block) {
            return Err(err(format!(
                "tensor `{tensor_name}`: row length {cols} is not a multiple of the {dtype:?} block size {block}"
            )));
        }
        let row_bytes = cols / block * dtype.type_size();
        if raw_len != rows * row_bytes {
            return Err(err(format!(
                "tensor `{tensor_name}`: {raw_len} bytes do not hold a [{rows}, {cols}] {dtype:?} tensor"
            )));
        }
        let quantized = !matches!(dtype, GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16);
        let device_row_bytes = if quantized {
            if !cols.is_multiple_of(mk::BLOCK) {
                return Err(err(format!(
                    "tensor `{tensor_name}`: row length {cols} is not a multiple of the {}-element block",
                    mk::BLOCK
                )));
            }
            mk::DenseLayout::new(dtype, 1, cols)
                .map_err(|e| err(format!("tensor `{tensor_name}`: {e}")))?
                .bytes()
        } else {
            cols * 4
        };
        let chunk = (limit / device_row_bytes.max(1)).clamp(1, rows.max(1));
        Ok(Self {
            tensor_name,
            dtype,
            rows,
            cols,
            quantized,
            row_bytes,
            chunk,
        })
    }

    /// The row ranges of the chunks.
    fn chunks(&self) -> impl Iterator<Item = Range<usize>> + use<> {
        let (rows, chunk) = (self.rows, self.chunk);
        (0..rows)
            .step_by(chunk)
            .map(move |start| start..(start + chunk).min(rows))
    }

    /// Convert the rows `rows` of the weight, whose file bytes are `raw`, and
    /// upload them as one part.
    fn upload(&self, raw: &[u8], rows: Range<usize>, device: &Device) -> Result<Part, CucaError> {
        let context = |e: CucaError| err(format!("tensor `{}`: {e}", self.tensor_name));
        let (dtype, cols, row_bytes) = (self.dtype, self.cols, self.row_bytes);
        let n = rows.len();
        let src = &raw[rows.start * row_bytes..rows.end * row_bytes];
        if self.quantized {
            let layout = mk::DenseLayout::new(dtype, n, cols).map_err(context)?;
            let words = mk::pack_dense(&layout, dtype, src).map_err(context)?;
            // The words travel as the bits of an i32 tensor.
            let words: Vec<i32> = words.into_iter().map(|w| w as i32).collect();
            let len = words.len();
            Ok(Part::Packed(Packed {
                pool: Tensor::from_data(TensorData::new(words, [len]), device),
                layout,
            }))
        } else {
            let mut floats = vec![0f32; n * cols];
            for (row, dst) in src
                .chunks_exact(row_bytes.max(1))
                .zip(floats.chunks_exact_mut(cols.max(1)))
            {
                dequant_row(dtype, row, 1.0, dst).map_err(context)?;
            }
            Ok(Part::Float(Tensor::from_data(
                TensorData::new(floats, [n, cols]),
                device,
            )))
        }
    }
}

/// The `[rows, cols]` weight whose GGUF bytes are `raw`, on the device, in
/// row chunks of at most `limit` device bytes:
///
/// - Q4_0, Q4_1 and Q8_0 are repacked into the dense kernels' block-major
///   words ([`mk::pack_dense`]) and stay quantized;
/// - F32, F16 and BF16 are dequantized on the host row by row to f32
///   (exact);
/// - every other quantized dtype (Q5_0, Q5_1, Q8_1, the K-quants) is
///   dequantized row by row and requantized to Q8_0 ([`quantize_q8_0`]), at
///   a rounding error of at most half a Q8_0 step per element.
///
/// Chunks are converted and uploaded one at a time (the host holds one
/// converted chunk, not the whole weight).
fn upload_matrix_with(
    tensor_name: &str,
    dtype: GgmlDType,
    raw: &[u8],
    rows: usize,
    cols: usize,
    device: &Device,
    limit: usize,
) -> Result<Weight, CucaError> {
    let plan = PackPlan::new(tensor_name, dtype, raw.len(), rows, cols, limit)?;
    let parts = plan
        .chunks()
        .map(|range| plan.upload(raw, range, device))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Weight { parts })
}

/// Reads tensors from the GGUF file into one reused buffer and uploads them.
struct WeightReader<'a> {
    content: &'a gguf_file::Content,
    file: File,
    buf: Vec<u8>,
    device: &'a Device,
    /// `ExpertPlacement::Host`: which layers keep their experts in host RAM.
    host: Option<HostPlan>,
}

impl<'a> WeightReader<'a> {
    /// A reader of the file at `path` (index `content`) that uploads to
    /// `device`.
    fn open(
        content: &'a gguf_file::Content,
        path: &Path,
        device: &'a Device,
        host: Option<HostPlan>,
    ) -> Result<Self, CucaError> {
        Ok(Self {
            content,
            file: File::open(path)?,
            buf: Vec::new(),
            device,
            host,
        })
    }
}

/// The MoE layers `lead..lead + n` keep their experts in host RAM
/// (`ExpertPlacement::Host`; `n` is unbounded for all of them).
#[derive(Clone)]
struct HostPlan {
    lead: usize,
    n: usize,
    path: std::path::PathBuf,
    hp: Hparams,
    /// Hold the hosted experts as f32 (the exact reference of the tests).
    dequantize: bool,
}

impl WeightReader<'_> {
    /// Read `len` bytes at `offset` bytes into the data of `tensor_name` into
    /// `buf`.
    fn read_at(&mut self, tensor_name: &str, offset: usize, len: usize) -> Result<(), CucaError> {
        let info = tensor_info(self.content, tensor_name)?;
        self.buf.resize(len, 0);
        self.file.seek(SeekFrom::Start(
            self.content.tensor_data_offset + info.offset + offset as u64,
        ))?;
        self.file
            .read_exact(&mut self.buf)
            .map_err(|e| err(format!("tensor `{tensor_name}`: {e} (truncated GGUF?)")))
    }

    /// Read `tensor_name` into `buf` after checking its shape; returns its
    /// dtype.
    fn read(&mut self, tensor_name: &str, shape: &[usize]) -> Result<GgmlDType, CucaError> {
        let info = tensor_info(self.content, tensor_name)?;
        let dims = info.shape.dims();
        if dims != shape {
            return Err(err(format!(
                "tensor `{tensor_name}`: expected shape {shape:?}, found {dims:?}"
            )));
        }
        let dtype = info.ggml_dtype;
        let elements: usize = shape.iter().product();
        if !elements.is_multiple_of(dtype.block_size()) {
            return Err(err(format!(
                "tensor `{tensor_name}`: {elements} elements are not whole {dtype:?} blocks"
            )));
        }
        self.read_at(
            tensor_name,
            0,
            elements / dtype.block_size() * dtype.type_size(),
        )?;
        Ok(dtype)
    }

    /// A `[rows, cols]` matmul weight on the device; see [`upload_matrix`].
    fn matrix(&mut self, tensor_name: &str, rows: usize, cols: usize) -> Result<Weight, CucaError> {
        let dtype = self.read(tensor_name, &[rows, cols])?;
        upload_matrix_with(
            tensor_name,
            dtype,
            &self.buf,
            rows,
            cols,
            self.device,
            MAX_WEIGHT_CHUNK_BYTES,
        )
    }

    /// The bytes the part `cut` keeps of the `[rows, cols]` weight
    /// `tensor_name`, and the dtype they are in: the cut's row ranges are
    /// read from the file at their byte offsets (a shard reads only its
    /// rows), its columns are a per-row byte sub-slice of the file's dtype
    /// when the bounds lie on block boundaries and otherwise cut from the
    /// dequantized rows ([`tp::column_dtype`]). Touches no device.
    ///
    /// # Errors
    ///
    /// [`CucaError::Provider`] for a tensor that is missing, of another
    /// shape, truncated or cut outside its matrix.
    fn cut_bytes(
        &mut self,
        tensor_name: &str,
        (rows, cols): (usize, usize),
        cut: &Cut,
    ) -> Result<(GgmlDType, Vec<u8>), CucaError> {
        let info = tensor_info(self.content, tensor_name)?;
        let dims = info.shape.dims();
        if dims != [rows, cols] {
            return Err(err(format!(
                "tensor `{tensor_name}`: expected shape {:?}, found {dims:?}",
                [rows, cols]
            )));
        }
        let dtype = info.ggml_dtype;
        let block = dtype.block_size();
        if !cols.is_multiple_of(block) {
            return Err(err(format!(
                "tensor `{tensor_name}`: row length {cols} is not a multiple of the {dtype:?} block size {block}"
            )));
        }
        if cut.cols.end > cols || cut.rows.iter().any(|r| r.end > rows) {
            return Err(err(format!(
                "tensor `{tensor_name}`: the cut {cut:?} lies outside the [{rows}, {cols}] matrix"
            )));
        }
        let row_bytes = cols / block * dtype.type_size();
        if cut.cols == (0..cols)
            && let [r] = cut.rows.as_slice()
        {
            self.read_at(tensor_name, r.start * row_bytes, r.len() * row_bytes)?;
            return Ok((dtype, std::mem::take(&mut self.buf)));
        }
        let held = tp::column_dtype(dtype, cols, &cut.cols);
        let mut out = Vec::new();
        for r in &cut.rows {
            self.read_at(tensor_name, r.start * row_bytes, r.len() * row_bytes)?;
            append_columns(dtype, &self.buf, cols, &cut.cols, held, &mut out)
                .map_err(|e| err(format!("tensor `{tensor_name}`: {e}")))?;
        }
        Ok((held, out))
    }

    /// The part `cut` keeps of the `[rows, cols]` matmul weight `tensor_name`
    /// on the device (a tensor-parallel shard's slice, see [`tp`] and
    /// [`WeightReader::cut_bytes`]), uploaded under the rules of
    /// [`upload_matrix_with`]. The whole matrix is [`WeightReader::matrix`].
    fn matrix_cut(
        &mut self,
        tensor_name: &str,
        (rows, cols): (usize, usize),
        cut: &Cut,
    ) -> Result<Weight, CucaError> {
        if cut.is_whole(rows, cols) {
            return self.matrix(tensor_name, rows, cols);
        }
        let (held, bytes) = self.cut_bytes(tensor_name, (rows, cols), cut)?;
        upload_matrix_with(
            tensor_name,
            held,
            &bytes,
            cut.n_rows(),
            cut.cols.len(),
            self.device,
            MAX_WEIGHT_CHUNK_BYTES,
        )
    }

    /// The elements `rows` (ranges, one after the other) of the `[len]`
    /// vector `tensor_name` as a device tensor: the bias rows that follow a
    /// cut weight's rows.
    fn vector_cut(
        &mut self,
        tensor_name: &str,
        len: usize,
        rows: &[Range<usize>],
    ) -> Result<Tensor<1>, CucaError> {
        if let [r] = rows
            && *r == (0..len)
        {
            return self.vector(tensor_name, len);
        }
        let values = self.host_floats(tensor_name, &[len])?;
        let mut kept: Vec<f32> = Vec::new();
        for r in rows {
            let part = values.get(r.clone()).ok_or_else(|| {
                err(format!(
                    "tensor `{tensor_name}`: rows {r:?} lie outside its {len} elements"
                ))
            })?;
            kept.extend_from_slice(part);
        }
        let n = kept.len();
        Ok(Tensor::from_data(TensorData::new(kept, [n]), self.device))
    }

    /// A matrix weight, cut to `cut`, plus, when `bias` is set, its `[rows]`
    /// bias cut to the same rows.
    fn proj(
        &mut self,
        il: usize,
        part: &str,
        shape: [usize; 2],
        bias: bool,
        cut: &Cut,
    ) -> Result<Proj, CucaError> {
        Ok(Proj {
            w: self.matrix_cut(&name::blk(il, part), (shape[0], shape[1]), cut)?,
            bias: if bias {
                Some(self.vector_cut(&name::blk_bias(il, part), shape[0], &cut.rows)?)
            } else {
                None
            },
        })
    }

    /// The rows `rows` of the matrix of expert `e` of the stacked expert
    /// tensor `tensor_name` (`slice` describes it).
    fn expert_rows(
        &mut self,
        tensor_name: &str,
        slice: &ExpertSlice,
        e: usize,
        rows: &Range<usize>,
    ) -> Result<Vec<u8>, CucaError> {
        let row_bytes = slice.cols / slice.dtype.block_size() * slice.dtype.type_size();
        // The up half of a fused gate/up tensor starts inside the expert's
        // record, which `slice` knows as an absolute file offset.
        let tensor_start =
            self.content.tensor_data_offset + tensor_info(self.content, tensor_name)?.offset;
        let within = (slice.offset(e) - tensor_start) as usize;
        self.read_at(
            tensor_name,
            within + rows.start * row_bytes,
            rows.len() * row_bytes,
        )?;
        Ok(std::mem::take(&mut self.buf))
    }

    /// The experts of MoE layer `il` (FFN width `ff`; `fused` when gate and
    /// up share one `ffn_gate_up_exps` tensor): every expert read,
    /// converted and uploaded now, or, with `tiers`, left in the file for
    /// the tiers to load into an empty pool. Shard `tp` of a split model
    /// holds every expert cut to its `ff / world` units (gate and up rows,
    /// down columns).
    #[allow(clippy::too_many_arguments)]
    fn expert_bank(
        &mut self,
        il: usize,
        n_expert: usize,
        ff: usize,
        n_embd: usize,
        fused: bool,
        tiers: Option<&TierPlan>,
        tp: Tp,
    ) -> Result<GpuExpertBank, CucaError> {
        let (layer, layout) = layer_experts(self.content, il, n_expert, ff, n_embd, fused)?;
        if let Some(h) = &self.host
            && il >= h.lead
            && il - h.lead < h.n
        {
            return Ok(GpuExpertBank::Host {
                experts: HostExperts::load(self.content, &h.path, &h.hp, il, h.dequantize)?,
                layout,
            });
        }
        let no_list = Tensor::<1, Int>::zeros([2], self.device);
        if let Some(plan) = tiers {
            // One buffer holds at most MAX_PART_BYTES.
            let device_slots = plan
                .device_slots
                .min(MAX_PART_BYTES / layout.bytes().max(1))
                .max(1);
            let pool = Tensor::<1, Int>::zeros([device_slots * layout.words], self.device);
            return Ok(GpuExpertBank::Tiered(TieredBank {
                il,
                layout,
                device_slots,
                store: Arc::clone(&plan.store),
                state: Mutex::new(TierState {
                    pool: Some(pool),
                    table: SlotTable::new(device_slots),
                    free: (0..device_slots).rev().collect(),
                }),
            }));
        }
        // The units of the FFN width this shard computes, and the dtype of
        // the down matrix once its columns are cut.
        let part = tp.part(ff);
        let whole = part == (0..ff);
        let down_dtype = tp::column_dtype(layer.down.dtype, ff, &part);
        let dtypes = [layer.gate.dtype, layer.up.dtype, down_dtype];
        let layout = SlotLayout::new(n_embd, part.len(), dtypes)?;
        if n_expert * layout.bytes() > MAX_PART_BYTES {
            return Err(CucaError::Config(format!(
                "layer {il}: {n_expert} experts of {} bytes do not fit one device buffer of {MAX_PART_BYTES} bytes; use tiered experts",
                layout.bytes()
            )));
        }
        // The same conversion the tiers apply to an expert they load, so a
        // resident and a tiered expert are the same words.
        let blk = |tensor: &str| name::blk(il, tensor);
        let (gate_name, up_name) = if fused {
            (blk(name::FFN_GATE_UP_EXPS), blk(name::FFN_GATE_UP_EXPS))
        } else {
            (blk(name::FFN_GATE_EXPS), blk(name::FFN_UP_EXPS))
        };
        let mut words: Vec<i32> = Vec::with_capacity(n_expert * layout.words);
        for e in 0..n_expert {
            let gate = self.expert_rows(&gate_name, &layer.gate, e, &part)?;
            let up = self.expert_rows(&up_name, &layer.up, e, &part)?;
            let down = self.expert_rows(&blk(name::FFN_DOWN_EXPS), &layer.down, e, &(0..n_embd))?;
            let down = if whole {
                down
            } else {
                let mut cut = Vec::new();
                append_columns(layer.down.dtype, &down, ff, &part, down_dtype, &mut cut)
                    .map_err(|e| err(format!("layer {il} experts: {e}")))?;
                cut
            };
            let packed = mk::pack_expert(&layout, dtypes, &gate, &up, &down)?;
            words.extend(packed.iter().map(|w| *w as i32));
        }
        let n = words.len();
        let pool = Tensor::<1, Int>::from_data(TensorData::new(words, [n]), self.device);
        Ok(GpuExpertBank::Resident {
            pool,
            no_list,
            layout,
        })
    }

    /// The values of the (float or quantized) tensor `tensor_name`, whose
    /// shape must be `shape`, as row-major f32.
    fn host_floats(&mut self, tensor_name: &str, shape: &[usize]) -> Result<Vec<f32>, CucaError> {
        let dtype = self.read(tensor_name, shape)?;
        let mut out = vec![0f32; shape.iter().product()];
        dequant_row(dtype, &self.buf, 1.0, &mut out)
            .map_err(|e| err(format!("tensor `{tensor_name}`: {e}")))?;
        Ok(out)
    }

    /// A 1-D f32 tensor on the device.
    fn vector(&mut self, tensor_name: &str, len: usize) -> Result<Tensor<1>, CucaError> {
        let values = self.host_floats(tensor_name, &[len])?;
        Ok(Tensor::from_data(
            TensorData::new(values, [len]),
            self.device,
        ))
    }

    /// `vector` of `tensor_name` when `present`.
    fn opt_vector(
        &mut self,
        present: bool,
        tensor_name: &str,
        len: usize,
    ) -> Result<Option<Tensor<1>>, CucaError> {
        present.then(|| self.vector(tensor_name, len)).transpose()
    }

    /// A one-element tensor as a host scalar.
    fn scalar(&mut self, tensor_name: &str) -> Result<f64, CucaError> {
        let values = self.host_floats(tensor_name, &[1])?;
        values
            .first()
            .map(|&v| f64::from(v))
            .ok_or_else(|| err(format!("tensor `{tensor_name}`: empty")))
    }
}

/// A matrix weight `[out, in]` with its optional bias `[out]`.
struct Proj {
    w: Weight,
    bias: Option<Tensor<1>>,
}

impl Proj {
    /// `x [t, in]` through the weight, plus the bias: `[t, out]`.
    fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        let y = linear(x, &self.w);
        match &self.bias {
            Some(b) => y + b.clone().unsqueeze::<2>(),
            None => y,
        }
    }
}

/// Query, key and value projections of one layer.
// One value per layer, built at load and only read afterwards: the size gap
// between the variants costs nothing worth an indirection in every forward
// pass.
#[allow(clippy::large_enum_variant)]
enum QkvWeights {
    /// One matmul whose output is `[q | k | v]`.
    Fused(Proj),
    /// Separate projections; K/V exist only on layers that own K/V.
    Split { q: Proj, kv: Option<(Proj, Proj)> },
}

/// The key and value projections `[t, nkv * d]` of a layer that owns K/V.
type KvProj = (Tensor<2>, Tensor<2>);

/// Attention weights of one layer.
struct AttnWeights {
    qkv: QkvWeights,
    q_norm: Option<Tensor<1>>,
    k_norm: Option<Tensor<1>>,
    /// Whether the layer applies the weightless value norm.
    v_norm: bool,
    /// `attn_gate`: the sigmoid gate of the attention output, computed from
    /// the normed layer input ([`Behavior::attn_gate_tensor`](super::arch::Behavior)).
    /// Its rows follow the shard's heads like the query's.
    gate: Option<Proj>,
    wo: Proj,
}

/// Gate and up projections of a dense FFN.
enum GateUp {
    Split {
        gate: Weight,
        up: Weight,
    },
    /// `up` yields `[gate | up]`, `ff` wide each.
    Fused {
        up: Weight,
        ff: usize,
    },
}

// One value per layer, built at load and only read afterwards.
#[allow(clippy::large_enum_variant)]
enum FfnWeights {
    Dense {
        gate_up: GateUp,
        down: Weight,
    },
    Moe {
        spec: MoeSpec,
        router: Weight,
        experts: GpuExpertBank,
        shared: Option<GpuShared>,
    },
}

/// The shared expert of a Qwen 3.5 MoE layer: a dense gated FFN every token
/// runs, scaled per token by `sigmoid(gate_inp . x)`.
struct GpuShared {
    gate: Weight,
    up: Weight,
    down: Weight,
    /// `ffn_gate_inp_shexp` as `[1, n_embd]`; `None` for an ungated shared
    /// expert (Mistral Small 4), whose output is added as it is.
    gate_inp: Option<Tensor<2>>,
}

/// Gated delta-net weights of one layer (see [`super::delta`] for the
/// algorithm; this is its device form).
struct DeltaWeights {
    spec: DeltaSpec,
    /// `attn_qkv`: the convolution inputs `[q | k | v]`.
    qkv: Weight,
    /// `attn_gate`: the output gate `z`.
    gate: Weight,
    /// `ssm_beta` rows then `ssm_alpha` rows, held as one f32 weight
    /// `[2 * n_v_heads, n_embd]`: two tiny matmuls (a narrow `vecmat` shape
    /// that burn runs by dequantizing the whole weight) become one.
    beta_alpha: Weight,
    /// `ssm_out`.
    out: Weight,
    /// The convolution taps, `[conv_kernel, conv_dim]`.
    conv: Tensor<2>,
    /// `ssm_norm`, `[head_v_dim]`.
    norm: Tensor<1>,
    /// `ssm_a`, `[1, n_v_heads]`.
    a: Tensor<2>,
    /// `ssm_dt.bias`, `[1, n_v_heads]`.
    dt_bias: Tensor<2>,
}

/// The recurrent state of one delta-net layer on the device. Device tensors
/// are immutable values, so a pass builds the next state and a snapshot is
/// just another handle to an earlier one.
#[derive(Clone)]
struct DeltaState {
    /// The last `conv_kernel - 1` convolution inputs, `[conv_kernel - 1,
    /// conv_dim]`.
    conv: Tensor<2>,
    /// One `[head_k_dim, head_v_dim]` matrix per value head, as `[n_v_heads
    /// / n_k_heads, n_k_heads, head_k_dim, head_v_dim]`: value head
    /// `r * n_k_heads + j` is `[r, j]` and reads key head `j`.
    s: Tensor<4>,
}

impl DeltaState {
    fn zeros(spec: &DeltaSpec, device: &Device) -> Self {
        Self {
            conv: Tensor::zeros([spec.conv_kernel - 1, spec.conv_dim()], device),
            s: Tensor::zeros(
                [
                    spec.n_v_heads / spec.n_k_heads,
                    spec.n_k_heads,
                    spec.head_k_dim,
                    spec.head_v_dim,
                ],
                device,
            ),
        }
    }
}

/// Multi-head latent attention weights of one layer (Mistral Small 4, see
/// [`Weights::mla`]). The per-head up-projections are f32 on the device,
/// already transposed into the right operands of the two batched products
/// the layer runs around the attention.
struct MlaWeights {
    q_a: Weight,
    q_a_norm: Tensor<1>,
    q_b: Weight,
    kv_a: Weight,
    kv_a_norm: Tensor<1>,
    /// `attn_k_b` as `[n_head, nope, kv_lora]`: `q_nope [n_head, t, nope] .
    /// k_b` is the query absorbed into the latent space.
    k_b: Tensor<3>,
    /// `attn_v_b` as `[n_head, kv_lora, v_head]`: the attended latent
    /// `[n_head, t, kv_lora] . v_b` is the head's value.
    v_b: Tensor<3>,
    wo: Proj,
}

/// What mixes the tokens of one layer.
// One value per layer, built at load and only read afterwards.
#[allow(clippy::large_enum_variant)]
enum Mixer {
    Attention(AttnWeights),
    Delta(Box<DeltaWeights>),
    Mla(Box<MlaWeights>),
}

/// One routed expert converted on the host to the words of the MoE kernels
/// ([`mk::pack_expert`]): the item of the RAM tier, built on the loader
/// threads. The bytes are shared, so an upload of an expert the RAM tier
/// keeps copies nothing.
struct PackedExpert {
    words: bytes::Bytes,
}

impl PackedExpert {
    fn new(words: &[u32]) -> Self {
        Self {
            words: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        }
    }

    /// The words as a device tensor.
    fn upload(&self, device: &Device) -> Tensor<1, Int> {
        let n = self.words.len() / 4;
        Tensor::from_data(
            TensorData::from_bytes(
                Bytes::from_shared(self.words.clone(), AllocationProperty::Other),
                [n],
                DType::I32,
            ),
            device,
        )
    }
}

/// Lock `mutex`, ignoring poisoning: the tables stay consistent between
/// operations, so a panic elsewhere does not invalidate them.
fn locked<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The routing of a layer from the router output `[ids | weights]` read back
/// to the host (`tokens * n_used` of each, see [`mk::route`]): per expert,
/// the tokens it computes and their weights.
fn routing_of(route: &[f32], tokens: usize, n_used: usize, n_expert: usize) -> Routing {
    let (ids, weights) = route
        .split_at_checked(tokens * n_used)
        .unwrap_or((route, &[]));
    let mut per_expert: Vec<Vec<(u32, f32)>> = vec![Vec::new(); n_expert];
    for (pair, (&id, &weight)) in ids.iter().zip(weights).enumerate() {
        if let Some(picks) = per_expert.get_mut(id as usize) {
            picks.push(((pair / n_used) as u32, weight));
        }
    }
    Routing { per_expert }
}

/// The experts of one MoE layer, as slots of one `u32` tensor (the *pool*)
/// that the MoE kernels index.
enum GpuExpertBank {
    /// Every expert on the device, expert `e` in slot `e`.
    Resident {
        pool: Tensor<1, Int>,
        /// An unused pair list (the kernels take the pairs from the router
        /// output).
        no_list: Tensor<1, Int>,
        layout: SlotLayout,
    },
    /// The experts stay in the file; the pool holds at most `device_slots`
    /// of them, the rest come through the `store`'s tiers.
    Tiered(TieredBank),
    /// The experts are in host RAM and the CPU computes them
    /// (`ExpertPlacement::Host`); `layout` only sizes the kernels' calls.
    Host {
        experts: HostExperts,
        layout: SlotLayout,
    },
}

struct TieredBank {
    il: usize,
    layout: SlotLayout,
    device_slots: usize,
    store: Arc<ExpertStore<PackedExpert>>,
    state: Mutex<TierState>,
}

/// The device tier of one layer: which expert sits in which slot.
struct TierState {
    /// `None` only while a slot write owns it.
    pool: Option<Tensor<1, Int>>,
    /// Expert -> slot.
    table: SlotTable<usize>,
    /// Slots no expert holds.
    free: Vec<usize>,
}

/// The tiers of a tiered load: the expert store (RAM tier and disk) and the
/// device tier's slots per layer.
struct TierPlan {
    store: Arc<ExpertStore<PackedExpert>>,
    device_slots: usize,
}

impl TierPlan {
    /// Seed the heat from `heat` and load the hottest experts: the device
    /// tier of every layer first, then the RAM tier with the next ones.
    fn warm(&self, weights: &Weights, heat: &ExpertHeat) -> Result<(), CucaError> {
        self.store.shared().seed(heat);
        for layer in &weights.layers {
            if let FfnWeights::Moe { experts, .. } = &layer.ffn {
                experts.warm(heat, &weights.device)?;
            }
        }
        self.store
            .warm(heat, self.device_slots, self.store.ram_slots())
    }
}

impl GpuExpertBank {
    fn layout(&self) -> &SlotLayout {
        match self {
            Self::Resident { layout, .. } | Self::Host { layout, .. } => layout,
            Self::Tiered(bank) => &bank.layout,
        }
    }

    /// The experts' output `[t, n_embd]` for the normed input `f` and the
    /// router output `route` (see [`mk::route`]) plus the output of `shared`;
    /// `call(pairs, use_list)` describes a kernel call.
    ///
    /// A resident bank queues the kernels and returns: the router output
    /// never leaves the device. A tiered bank reads it back to find the
    /// experts the pool lacks. A host bank reads the layer's normed rows and
    /// routing back in one buffer, queues `shared` while the CPU computes
    /// the experts and uploads their sum in one transfer. (burn submits the
    /// queued work lazily, and no `Device::flush` forces a submit before the
    /// CPU experts: it slows decode on Vulkan.)
    fn run(
        &self,
        f: Tensor<2>,
        route: Tensor<1>,
        call: &dyn Fn(usize, bool) -> MoeCall,
        device: &Device,
        shared: impl FnOnce() -> Option<Tensor<2>>,
    ) -> Result<Tensor<2>, CucaError> {
        let c = call(0, false);
        let (t, k, n_embd) = (c.tokens, c.n_used, c.slot.down.rows);
        let (routed, extra) = match self {
            Self::Resident { pool, no_list, .. } => {
                let c = call(t * k, false);
                let hidden = mk::up(f, route.clone(), no_list.clone(), pool.clone(), &c);
                let scratch = Tensor::<2>::empty([t * k, n_embd], device);
                let scratch = mk::down(hidden, route, no_list.clone(), pool.clone(), scratch, &c);
                (mk::sum(scratch, &c), shared())
            }
            Self::Tiered(bank) => (bank.run(f, route, call, device)?, shared()),
            Self::Host { experts, .. } => {
                let rows = t * n_embd;
                let mut host = read_f32(
                    Tensor::cat(vec![f.reshape([rows]), route], 0),
                    "the rows of host experts",
                )?;
                let routing = routing_of(&host.split_off(rows), t, k, c.n_expert);
                let extra = shared();
                let act = if c.act == 0 { Act::Silu } else { Act::Gelu };
                let out = experts.forward(&host, t, n_embd, &routing, act)?;
                (
                    Tensor::from_data(TensorData::new(out, [t, n_embd]), device),
                    extra,
                )
            }
        };
        Ok(match extra {
            Some(extra) => routed + extra,
            None => routed,
        })
    }

    /// Fill the pool of a tiered bank with the hottest experts of `heat`
    /// (at most its slots), read through the loader pool and uploaded here,
    /// then wait for the uploads: a warm start queues one layer's uploads at
    /// a time instead of the whole device tier at once. A resident bank has
    /// nothing to load.
    fn warm(&self, heat: &ExpertHeat, device: &Device) -> Result<(), CucaError> {
        let Self::Tiered(bank) = self else {
            return Ok(());
        };
        let ids: Vec<u32> = heat
            .ranked(bank.il)
            .into_iter()
            .take(bank.device_slots)
            .collect();
        let built = bank.store.fetch(bank.il, &ids)?;
        {
            let mut state = locked(&bank.state);
            for (e, packed) in ids.into_iter().zip(built) {
                let slot = bank.take_slot(&mut state, &[], &mut None)?;
                bank.write(&mut state, slot, &packed, device);
                state.table.insert(e, Arc::new(slot));
                add(
                    &bank.store.shared().device_resident_bytes,
                    bank.layout.bytes() as u64,
                );
            }
        }
        device
            .sync()
            .map_err(|e| err(format!("layer {}: uploading warm experts: {e}", bank.il)))
    }
}

impl TieredBank {
    /// A slot for a new expert: a free one, else the one of the table's
    /// victim (never an expert in `keep`, the ascending experts the current
    /// pass computes), which leaves the table.
    fn take_slot(
        &self,
        state: &mut TierState,
        keep: &[u32],
        heat: &mut Option<Vec<u32>>,
    ) -> Result<usize, CucaError> {
        if let Some(slot) = state.free.pop() {
            return Ok(slot);
        }
        let shared = self.store.shared();
        let heat = heat.get_or_insert_with(|| shared.heat_snapshot(self.il));
        match state.table.victim(heat, keep) {
            Some(victim) if keep.binary_search(&victim).is_err() => {
                let slot = state
                    .table
                    .remove(victim)
                    .ok_or_else(|| err("the device tier lost its victim"))?;
                sub(&shared.device_resident_bytes, self.layout.bytes() as u64);
                Ok(*slot)
            }
            _ => Err(err(format!(
                "layer {}: no device slot free for a new expert",
                self.il
            ))),
        }
    }

    /// Write `packed` into `slot` of the pool.
    fn write(&self, state: &mut TierState, slot: usize, packed: &PackedExpert, device: &Device) {
        if let Some(pool) = state.pool.take() {
            state.pool = Some(mk::write_slot(
                pool,
                packed.upload(device),
                slot * self.layout.words,
            ));
        }
    }

    /// The experts' output of one pass; see [`GpuExpertBank::run`].
    ///
    /// The pass computes the experts the router picked in groups of at most
    /// `device_slots` experts (one group unless a long prompt needs more
    /// experts than the pool holds): the pool's hits first, then misses
    /// loaded in ascending order into free slots or the victims' slots. The
    /// kernels write every (token, expert) result into one scratch and sum it
    /// at the end in expert order, so the grouping does not change a bit.
    fn run(
        &self,
        f: Tensor<2>,
        route: Tensor<1>,
        call: &dyn Fn(usize, bool) -> MoeCall,
        device: &Device,
    ) -> Result<Tensor<2>, CucaError> {
        let c = call(0, false);
        let (t, k, n_embd, n_expert) = (c.tokens, c.n_used, c.slot.down.rows, c.n_expert);
        let route_host = read_f32(route.clone(), "routing")?;
        let routing = routing_of(&route_host, t, k, n_expert);
        let expert_of: Vec<usize> = route_host.iter().take(t * k).map(|&e| e as usize).collect();
        let shared = self.store.shared();
        shared.record(self.il, &routing, t);
        let needed: Vec<u32> = routing
            .per_expert
            .iter()
            .enumerate()
            .filter(|(_, picks)| !picks.is_empty())
            .map(|(e, _)| e as u32)
            .collect();
        add(&shared.lookups, needed.len() as u64);

        let mut state = locked(&self.state);
        let missing: Vec<u32> = needed
            .iter()
            .copied()
            .filter(|&e| !state.table.contains(e))
            .collect();
        add(&shared.device_hits, (needed.len() - missing.len()) as u64);
        let mut lower = self.store.pass(self.il, &missing);
        let mut heat: Option<Vec<u32>> = None;
        let mut pending = needed;
        let mut scratch = Tensor::<2>::empty([t * k, n_embd], device);
        while !pending.is_empty() {
            let mut group: Vec<u32> = pending
                .iter()
                .copied()
                .filter(|&e| state.table.contains(e))
                .collect();
            let room = self.device_slots - group.len();
            let new: Vec<u32> = pending
                .iter()
                .copied()
                .filter(|&e| !state.table.contains(e))
                .take(room)
                .collect();
            group.extend(&new);
            group.sort_unstable();
            for e in new {
                let packed = lower.next(e)?;
                let slot = self.take_slot(&mut state, &group, &mut heat)?;
                self.write(&mut state, slot, &packed, device);
                state.table.insert(e, Arc::new(slot));
                add(&shared.device_resident_bytes, self.layout.bytes() as u64);
            }
            // The pairs of the group, with their slots.
            let mut slots = HashMap::with_capacity(group.len());
            for &e in &group {
                if let Some(slot) = state.table.get(e) {
                    slots.insert(e as usize, *slot);
                }
            }
            let mut list: Vec<i32> = Vec::new();
            for (pair, e) in expert_of.iter().enumerate() {
                if let Some(&slot) = slots.get(e) {
                    list.push(pair as i32);
                    list.push(slot as i32);
                }
            }
            let pairs = list.len() / 2;
            let list = Tensor::<1, Int>::from_data(TensorData::new(list, [2 * pairs]), device);
            let pool = state
                .pool
                .clone()
                .ok_or_else(|| err("the expert pool is being written"))?;
            let c = call(pairs, true);
            let hidden = mk::up(f.clone(), route.clone(), list.clone(), pool.clone(), &c);
            scratch = mk::down(hidden, route.clone(), list, pool, scratch, &c);
            pending.retain(|e| group.binary_search(e).is_err());
        }
        Ok(mk::sum(scratch, &call(t * k, false)))
    }
}

/// Per-layer-embedding block of one layer (Gemma 4).
struct PleLayer {
    inp_gate: Weight,
    proj: Weight,
    post_norm: Tensor<1>,
}

/// Weights of one transformer block.
struct LayerWeights {
    /// The pre-attention norm; `None` when the architecture has no pre-norms.
    attn_norm: Option<Tensor<1>>,
    mixer: Mixer,
    post_attn_norm: Option<Tensor<1>>,
    ffn_norm: Option<Tensor<1>>,
    ffn: FfnWeights,
    post_ffw_norm: Option<Tensor<1>>,
    ple: Option<PleLayer>,
    out_scale: Option<f64>,
}

/// Model-level per-layer-embedding weights (Gemma 4).
struct PleWeights {
    proj: Weight,
    proj_norm: Tensor<1>,
}

/// RoPE cos/sin tables of shape `[positions, n_dims / 2]` on the device.
struct RopeTable {
    cos: Tensor<2>,
    sin: Tensor<2>,
}

/// One K/V-owning layer's pool on the device: `[n_blocks, n_kv_head,
/// BLOCK_SIZE, head_dim]` keys and values, block `b` holding the positions of
/// the sequences' tables that name it (see [`paged`](super::paged)).
struct KvPool {
    k: Tensor<4>,
    /// `None` for a latent-attention layer, whose value is the leading
    /// columns of its key row: the pool holds one row per position.
    v: Option<Tensor<4>>,
}

/// The pool of every layer (`None` for layers that share another layer's
/// pool and layers without attention).
type KvLayers = Vec<Option<KvPool>>;

impl KvPool {
    /// The pool with new rows written: each of `writes` puts rows `base +
    /// w.off ..` of `k` (and of `v`, for a pool that holds values), both
    /// `[1, n_kv, rows, width]`, into rows `w.r0..w.r1` of block `w.block`.
    /// It takes the pool by value so that the assignments own the only
    /// handle to its tensors and write in place.
    fn write(
        mut self,
        writes: &[BlockWrite],
        base: usize,
        k: &Tensor<4>,
        v: Option<&Tensor<4>>,
    ) -> Self {
        let [_, n_kv, _, width] = self.k.dims();
        let mut values = self.v.take();
        for w in writes {
            let at = [w.block..w.block + 1, 0..n_kv, w.r0..w.r1, 0..width];
            let (from, rows) = (base + w.off, w.r1 - w.r0);
            self.k = self
                .k
                .slice_assign(at.clone(), k.clone().narrow(2, from, rows));
            values = match (values, v) {
                (Some(pool), Some(v)) => {
                    Some(pool.slice_assign(at, v.clone().narrow(2, from, rows)))
                }
                (pool, _) => pool,
            };
        }
        self.v = values;
        self
    }
}

/// Take the pool of layer `il` out of `kv`; put it back after [`KvPool::write`].
fn take_pool(kv: &mut KvLayers, il: usize) -> Result<KvPool, CucaError> {
    kv.get_mut(il)
        .and_then(Option::take)
        .ok_or_else(|| err(format!("layer {il} owns no K/V cache")))
}

/// The RoPE rows of the new positions: `[1, 1, t, n_dims / 2]`.
struct RopeRows {
    cos: Tensor<4>,
    sin: Tensor<4>,
}

/// Rows `r0..r1` of one block a pass writes, the new rows `off..off + (r1 -
/// r0)` of the call.
struct BlockWrite {
    block: usize,
    r0: usize,
    r1: usize,
    off: usize,
}

/// How a pass reads the visible positions of one window group.
enum BlockRead {
    /// A single query row: the decode kernel reads the blocks in place.
    /// `table` is `[1, n]` ids of the blocks from the one holding the first
    /// visible position, `range` the visible `lo`, `hi` relative to it.
    Decode {
        table: Box<Tensor<2, Int>>,
        range: Box<Tensor<1, Int>>,
    },
    /// Several rows: `idx` the same ids gathered into one contiguous view,
    /// of which rows `skip..skip + len` are visible.
    Gather {
        idx: Box<Tensor<1, Int>>,
        skip: usize,
        len: usize,
    },
    /// A single query row of a decode step whose sequences all read in one
    /// launch (see [`StackedDecode`]): the block ids of `Decode` and its
    /// `range`, still on the host.
    Rows { ids: Vec<i32>, range: [i32; 2] },
}

/// What a pass does with the blocks of one window group, shared by its
/// layers.
struct GroupBlocks {
    writes: Vec<BlockWrite>,
    read: BlockRead,
}

/// The positions and blocks one pass touches in one window group.
struct BlockRows<'a> {
    /// The sequence's block table of the group.
    table: &'a [u32],
    /// Blocks of the group's pools.
    n_blocks: usize,
    /// First new position and number of new rows.
    start: usize,
    t: usize,
    /// Visible positions `lo..hi`.
    lo: usize,
    hi: usize,
    /// Whether the pass writes its rows (a drafter's does not).
    write: bool,
    /// Keep a single row's blocks as host ids ([`BlockRead::Rows`]) for a
    /// stacked decode launch instead of uploading them.
    stack: bool,
}

impl BlockRows<'_> {
    /// The block writes and the read of the pass.
    ///
    /// # Errors
    ///
    /// [`CucaError::Provider`] when the table does not reach the visible
    /// positions or holds no block (a released or out-of-pool id) for one of
    /// them.
    fn view(&self, device: &Device) -> Result<GroupBlocks, CucaError> {
        let Self {
            table,
            n_blocks,
            start,
            t,
            lo,
            hi,
            write,
            stack,
        } = *self;
        let first = lo / BLOCK_SIZE;
        let last = (hi - 1) / BLOCK_SIZE;
        let ids = table.get(first..=last).ok_or_else(|| {
            err(format!(
                "the block table holds {} blocks but positions {lo}..{hi} need block {last}",
                table.len()
            ))
        })?;
        if let Some(at) = ids
            .iter()
            .position(|&b| b == NO_BLOCK || b as usize >= n_blocks)
        {
            return Err(err(format!(
                "position {} of the block table has no block (entry {}, pool of {n_blocks})",
                (first + at) * BLOCK_SIZE,
                ids[at]
            )));
        }
        let mut writes = Vec::new();
        if write {
            for bi in start / BLOCK_SIZE..=(start + t - 1) / BLOCK_SIZE {
                let from = start.max(bi * BLOCK_SIZE);
                let to = (start + t).min((bi + 1) * BLOCK_SIZE);
                writes.push(BlockWrite {
                    block: ids[bi - first] as usize,
                    r0: from - bi * BLOCK_SIZE,
                    r1: to - bi * BLOCK_SIZE,
                    off: from - start,
                });
            }
        }
        // Block ids are below the pool size, which fits an i32.
        let ids: Vec<i32> = ids.iter().map(|&b| b as i32).collect();
        let n = ids.len();
        let skip = lo - first * BLOCK_SIZE;
        let read = if t == 1 && stack {
            BlockRead::Rows {
                ids,
                range: [skip as i32, (hi - first * BLOCK_SIZE) as i32],
            }
        } else if t == 1 {
            let range = vec![skip as i32, (hi - first * BLOCK_SIZE) as i32];
            BlockRead::Decode {
                table: Box::new(Tensor::from_data(TensorData::new(ids, [1, n]), device)),
                range: Box::new(Tensor::from_data(TensorData::new(range, [2]), device)),
            }
        } else {
            BlockRead::Gather {
                idx: Box::new(Tensor::from_data(TensorData::new(ids, [n]), device)),
                skip,
                len: hi - lo,
            }
        };
        Ok(GroupBlocks { writes, read })
    }
}

/// The visible range, mask and blocks of the layers with one window.
struct MaskInput {
    window: Option<usize>,
    /// `[1, 1, t, hi - lo]`; `None` for a single query, which sees every key.
    mask: Option<Tensor<4>>,
    /// `None` when no K/V group has this window (a layer without
    /// attention).
    blocks: Option<GroupBlocks>,
}

/// Where a pass finds its blocks.
enum PassKv<'a> {
    /// The target's own pass: the sequence's tables, one per group of the
    /// layout.
    Trunk {
        layout: &'a KvLayout,
        tables: &'a [BlockTable],
    },
    /// A drafter's: the target's pools, read-only.
    Drafter(&'a GpuExtKv),
}

/// One sequence of a pass as [`Weights::pass`] takes it: `t` rows at
/// positions `start..`, advancing recurrent-state slot `slot`.
struct PassSeq<'a> {
    start: usize,
    t: usize,
    slot: usize,
    kv: PassKv<'a>,
}

/// What a pass knows about one of its sequences.
struct SeqPass {
    /// First row among the pass's rows, and the row count.
    off: usize,
    t: usize,
    /// The recurrent-state slot (hybrid models).
    slot: usize,
    /// One entry per distinct layer window.
    masks: Vec<MaskInput>,
}

impl SeqPass {
    /// The mask and the block view of the layers of `window`.
    ///
    /// # Errors
    ///
    /// [`CucaError::Provider`] when no K/V group has this window.
    fn visible(
        &self,
        il: usize,
        window: Option<usize>,
    ) -> Result<(&MaskInput, &GroupBlocks), CucaError> {
        let visible = self
            .masks
            .iter()
            .find(|m| m.window == window)
            .ok_or_else(|| err(format!("layer {il}: no attention mask for its window")))?;
        let blocks = visible.blocks.as_ref().ok_or_else(|| {
            err(format!(
                "layer {il}: the sequence has no block table for its window"
            ))
        })?;
        Ok((visible, blocks))
    }
}

/// The blocks of every sequence of a decode step for one window, ready for
/// one launch of the decode kernel: `table` `[sequences, width]` block ids
/// (a row per sequence, padded) and `range` `[sequences * 2]` visible `lo`,
/// `hi`.
struct StackedDecode {
    window: Option<usize>,
    table: Tensor<2, Int>,
    range: Tensor<1, Int>,
}

/// What one forward pass shares between its layers.
struct Pass {
    /// New rows, all sequences'.
    t: usize,
    /// Per-layer inputs of the Gemma 4 per-layer embeddings,
    /// `[t, n_layer, ple_dim]`.
    ple: Option<Tensor<3>>,
    /// One entry per [`Hparams::ropes`] entry, the rows of every sequence
    /// one after the other.
    ropes: Vec<RopeRows>,
    /// Per-row query scale, when a row is past the scaling length.
    q_scale: Option<Tensor<4>>,
    /// The sequences, in row order.
    seqs: Vec<SeqPass>,
    /// Empty unless every sequence is one query row (and there are several):
    /// then one entry per window with K/V blocks, replacing the sequences'
    /// own reads.
    stacked: Vec<StackedDecode>,
}

/// The final norm and the LM head, held by the first shard only.
struct Head {
    norm: Tensor<1>,
    lm_head: Weight,
}

/// Everything read-only during a forward pass: the weights of one device (one
/// shard of a tensor-parallel model, see [`tp`]; the whole model on one
/// device), with the hyperparameters as that shard sees them (its own head
/// counts).
struct Weights {
    hp: Hparams,
    device: Device,
    ple: Option<PleWeights>,
    /// The layers `first..hp.n_layer`: a model offloaded in part to the CPU
    /// (see [`split`](super::split)) holds the last layers only.
    layers: Vec<LayerWeights>,
    /// Index of the first layer in `layers` (0 unless layers are offloaded).
    first: usize,
    /// `None` on every shard but the first.
    head: Option<Head>,
    ropes: Vec<RopeTable>,
}

/// One sequence's recurrent state: every delta-net layer's state (`None` for
/// attention layers) and the position it sits at (where the sequence's next
/// forward call must start). Device tensors are immutable values, so a pass
/// builds the next state and a [`GpuSnapshot`] is just more handles to the
/// current one.
struct DeltaSlot {
    len: usize,
    layers: Vec<Option<DeltaState>>,
}

/// The recurrent state of a hybrid model at one position: a handle to every
/// delta-net layer's state tensors (`None` for attention layers), taken by
/// [`GpuTransformer::snapshot`] and put back by [`GpuTransformer::restore`].
/// Free to take (the tensors are immutable) and sized by the state alone.
pub(crate) struct GpuSnapshot {
    layers: Vec<Option<DeltaState>>,
}

impl GpuSnapshot {
    /// Bytes of f32 state the snapshot keeps alive on the device.
    pub(crate) fn bytes(&self) -> usize {
        self.layers
            .iter()
            .flatten()
            .map(|d| (d.conv.shape().num_elements() + d.s.shape().num_elements()) * 4)
            .sum()
    }
}

/// A loaded model on a GPU with its K/V cache; see the module docs.
pub(crate) struct GpuTransformer {
    /// The weights of every shard, shard 0 (the `device` of the load, which
    /// also holds the LM head) first; never empty.
    weights: Vec<Weights>,
    /// The sum of the shards' partial layer outputs.
    collective: Collective,
    context_length: usize,
    vocab: usize,
    /// Behind a lock because a drafter reads embedding rows through a shared
    /// reference to the target while the table seeks in its file.
    embed: Mutex<RowTable>,
    ple_table: Option<RowTable>,
    /// The K/V pool of every K/V-owning layer, per shard.
    kv: Vec<KvLayers>,
    /// The groups of the pools and their sizes.
    layout: KvLayout,
    /// The implicit sequence of the unit tests.
    #[cfg(test)]
    solo: Solo,
    /// The recurrent state of every sequence slot of a hybrid model
    /// (`Sizing::max_sequences` of them; none for an attention-only model).
    slots: Vec<DeltaSlot>,
    /// The zero state of every delta-net layer, shared by every empty slot.
    zero: Vec<Option<DeltaState>>,
    /// The model has delta-net layers.
    hybrid: bool,
    /// Counters and heat of the expert tiers, with `ExpertPlacement::Tiered`.
    tiers: Option<Arc<TierShared>>,
}

/// Wait for `device`'s queued work, then hand the pages no live tensor uses
/// back to the driver. burn keeps freed pages in its pools for reuse, so a
/// dropped model's device memory stays reserved by the process unless it is
/// returned here.
fn return_device_memory(device: &Device) {
    // Best effort: a lost device fails the wait, and dropping must not fail.
    let _ = device.sync();
    device.memory_cleanup();
}

impl Drop for GpuTransformer {
    fn drop(&mut self) {
        // The communication server first: it holds the devices' streams.
        self.collective.close();
        for kv in &mut self.kv {
            kv.clear();
        }
        self.slots.clear();
        self.zero.clear();
        for weights in &mut self.weights {
            weights.release();
        }
        // Every distinct device once: shards may share one.
        let mut returned: Vec<&Device> = Vec::with_capacity(self.weights.len());
        for weights in &self.weights {
            if !returned.contains(&&weights.device) {
                return_device_memory(&weights.device);
                returned.push(&weights.device);
            }
        }
    }
}

impl Weights {
    /// Drop the device tensors (all but the final norm vector), so that
    /// [`return_device_memory`] finds their pages free.
    fn release(&mut self) {
        self.layers.clear();
        self.ple = None;
        if let Some(head) = &mut self.head {
            head.lm_head.parts.clear();
        }
        self.ropes.clear();
    }

    /// The final norm and the LM head.
    ///
    /// # Errors
    ///
    /// [`CucaError::Provider`] on a shard that holds none.
    fn head_parts(&self) -> Result<&Head, CucaError> {
        self.head
            .as_ref()
            .ok_or_else(|| err("this tensor-parallel shard holds no LM head"))
    }

    /// Read the weights of the model `hp` describes (as it is in the file)
    /// that shard `tp` holds through `reader` (onto the reader's device):
    /// the layers (each cut to the shard's slice, see [`tp`]), on the first
    /// shard the final norm and the LM head (`output.weight`, else the token
    /// embedding), the per-layer-embedding projection and the RoPE tables.
    /// The result's `hp` is the shard's (its own head counts). The K/V pools
    /// and the recurrent states are the caller's ([`alloc_pools`]): a
    /// drafter has none, it reads the target's.
    /// With `tiers`, the experts of the MoE layers stay in the file (their
    /// banks hold the tiers) instead of being read.
    fn load(
        reader: &mut WeightReader<'_>,
        path: &Path,
        hp: &Hparams,
        tp: Tp,
        context_length: usize,
        tiers: Option<&TierPlan>,
        first: usize,
    ) -> Result<Self, CucaError> {
        let content = reader.content;
        let device = reader.device.clone();
        let n_embd = hp.n_embd;
        let head = if tp.shard == 0 {
            let vocab = *tensor_info(content, name::TOKEN_EMBD)?
                .shape
                .dims()
                .first()
                .ok_or_else(|| err(format!("tensor `{}`: scalar", name::TOKEN_EMBD)))?;
            let head = if content.tensor_infos.contains_key(name::OUTPUT) {
                name::OUTPUT
            } else {
                name::TOKEN_EMBD
            };
            Some(Head {
                lm_head: reader.matrix(head, vocab, n_embd)?,
                norm: reader.vector(name::OUTPUT_NORM, n_embd)?,
            })
        } else {
            None
        };
        let ple = match hp.ple_dim.filter(|_| first == 0) {
            Some(dim) => {
                let row_len = hp.n_layer * dim;
                Some(PleWeights {
                    proj: reader.matrix(name::PER_LAYER_MODEL_PROJ, row_len, n_embd)?,
                    proj_norm: reader.vector(name::PER_LAYER_PROJ_NORM, dim)?,
                })
            }
            None => None,
        };

        let ropes = build_rope_tables(content, path, hp, context_length)?
            .into_iter()
            .map(|t| {
                let shape = [context_length, t.half];
                RopeTable {
                    cos: Tensor::from_data(TensorData::new(t.cos, shape), &device),
                    sin: Tensor::from_data(TensorData::new(t.sin, shape), &device),
                }
            })
            .collect();

        let mut layers = Vec::with_capacity(hp.n_layer - first);
        for il in first..hp.n_layer {
            let spec = &hp.layers[il];
            let layout = LayerLayout::read(hp, il, spec, content)?;
            layers.push(load_layer(reader, hp, tp, il, spec, &layout, tiers)?);
        }
        Ok(Self {
            hp: tp.local_hparams(hp),
            device,
            ple,
            layers,
            first,
            head,
            ropes,
        })
    }
}

/// The K/V pool of every K/V-owning layer of `hp`, zeroed on `device`, `[n_blocks,
/// n_kv_head, BLOCK_SIZE, head_dim]` with the block count of the layer's group
/// in `layout`; `None` for the layers that share another layer's pool or have
/// no attention.
///
/// # Errors
///
/// [`CucaError::Provider`] when an owning layer belongs to no group of
/// `layout`.
fn alloc_pools(
    hp: &Hparams,
    layout: &KvLayout,
    device: &Device,
    first: usize,
    mirror: &[usize],
) -> Result<KvLayers, CucaError> {
    let mut kv: KvLayers = Vec::with_capacity(hp.n_layer);
    for (il, spec) in hp.layers.iter().enumerate() {
        // A layer before `first` runs elsewhere; only the pools of those
        // whose K/V a later layer reads exist here (see [`Back::mirror`]).
        if (il < first && !mirror.contains(&il)) || !spec.owns_kv(il) {
            kv.push(None);
            continue;
        }
        let blocks = layout.layer_group[il]
            .map(|g| layout.blocks[g])
            .ok_or_else(|| err(format!("layer {il} owns K/V but belongs to no block group")))?;
        let shape = [blocks, spec.n_kv_head, BLOCK_SIZE, spec.head_dim];
        kv.push(Some(KvPool {
            k: Tensor::zeros(shape, device),
            v: spec.mla.is_none().then(|| Tensor::zeros(shape, device)),
        }));
    }
    Ok(kv)
}

/// Where every MoE layer's experts live in the file and what one occupies in
/// the RAM and device tiers (its kernel words, see [`SlotLayout`]).
///
/// # Errors
///
/// The loader's errors for missing or malformed expert tensors.
fn tier_layout(
    content: &gguf_file::Content,
    hp: &Hparams,
    spec: MoeSpec,
    first: usize,
) -> Result<Vec<LayerExperts>, CucaError> {
    let mut layers = Vec::new();
    for (il, layer_spec) in hp.layers.iter().enumerate().skip(first) {
        let Ffn::Moe { ff, fused, .. } = LayerLayout::read(hp, il, layer_spec, content)?.ffn else {
            continue;
        };
        layers.push(layer_experts(content, il, spec.n_expert, ff, hp.n_embd, fused)?.0);
    }
    Ok(layers)
}

/// Where the `n_expert` experts of MoE layer `il` (FFN width `ff`, `n_embd`
/// inputs) live in the file, and the slot layout of one on the device;
/// `fused` when the gate and up matrices share one `ffn_gate_up_exps` tensor
/// (per expert, the gate rows then the up rows).
///
/// # Errors
///
/// The loader's errors for missing or malformed expert tensors,
/// [`CucaError::Config`] for rows that are not whole blocks.
fn layer_experts(
    content: &gguf_file::Content,
    il: usize,
    n_expert: usize,
    ff: usize,
    n_embd: usize,
    fused: bool,
) -> Result<(LayerExperts, SlotLayout), CucaError> {
    let slice = |part: &str, rows, cols| {
        ExpertSlice::read(content, &name::blk(il, part), n_expert, rows, cols)
    };
    let (gate, up) = if fused {
        ExpertSlice::read_fused(
            content,
            &name::blk(il, name::FFN_GATE_UP_EXPS),
            n_expert,
            ff,
            n_embd,
        )?
    } else {
        (
            slice(name::FFN_GATE_EXPS, ff, n_embd)?,
            slice(name::FFN_UP_EXPS, ff, n_embd)?,
        )
    };
    let down = slice(name::FFN_DOWN_EXPS, n_embd, ff)?;
    let layout = SlotLayout::new(n_embd, ff, [gate.dtype, up.dtype, down.dtype])?;
    let layer = LayerExperts {
        il,
        gate,
        up,
        down,
        stored_bytes: layout.bytes(),
    };
    Ok((layer, layout))
}

/// Convert one expert of `layer` from its three matrices' file bytes to the
/// kernels' words, as the resident loader converts it.
fn build_packed_expert(
    layer: &LayerExperts,
    gate: &[u8],
    up: &[u8],
    down: &[u8],
) -> Result<PackedExpert, CucaError> {
    let dtypes = [layer.gate.dtype, layer.up.dtype, layer.down.dtype];
    let layout = SlotLayout::new(layer.gate.cols, layer.gate.rows, dtypes)?;
    let words = mk::pack_expert(&layout, dtypes, gate, up, down)?;
    debug_assert_eq!(words.len(), layout.words);
    Ok(PackedExpert::new(&words))
}

/// The tiers of a tiered model (`ExpertPlacement::Tiered`): the experts of
/// every MoE layer from `first` on located in the file, the device tier's
/// slots and the RAM tier behind it sized by `tiers`; `shared` takes the
/// counters of another store of the same model (the CPU half of an offloaded
/// one).
///
/// # Errors
///
/// [`CucaError::Config`] for a model without experts, a device or RAM budget
/// below `n_used` experts per layer or heat of another shape; the loader's
/// errors for unreadable expert tensors.
fn open_tiers(
    content: &gguf_file::Content,
    path: &Path,
    hp: &Hparams,
    tiers: &ExpertTiers,
    first: usize,
    shared: Option<Arc<TierShared>>,
) -> Result<TierPlan, CucaError> {
    let spec = hp.moe.ok_or_else(|| {
        CucaError::Config(format!(
            "expert tiers need a mixture-of-experts model; `{}` has no experts",
            hp.arch.name()
        ))
    })?;
    let layers = tier_layout(content, hp, spec, first)?;
    let per_expert: usize = layers.iter().map(|l| l.stored_bytes).sum();
    let device_slots = tier_slots(
        "device",
        tiers.device_bytes,
        per_expert,
        spec.n_expert,
        spec.n_used,
    )?;
    let shared = shared.unwrap_or_else(|| Arc::new(TierShared::new(hp.n_layer, spec.n_expert)));
    if let Some(heat) = &tiers.warm_from {
        shared.check(heat)?;
    }
    let build: Arc<BuildFn<PackedExpert>> = Arc::new(build_packed_expert);
    let store = ExpertStore::open(
        path,
        hp.n_layer,
        layers,
        spec.n_expert,
        spec.n_used,
        tiers.ram_bytes,
        true,
        tiers.loader_threads,
        build,
        shared,
    )?;
    Ok(TierPlan {
        store: Arc::new(store),
        device_slots,
    })
}

/// The last layers of a model whose first layers run on the CPU (layer
/// offload, see [`split`](super::split)).
pub(crate) struct Back {
    /// The first layer this device holds, `1..n_layer`.
    pub(crate) first: usize,
    /// The expert-tier counters and heat shared with the CPU half's expert
    /// store, so that one set of statistics covers the whole model.
    pub(crate) shared: Option<Arc<TierShared>>,
    /// The K/V-owning layers before `first` whose K/V a layer from `first`
    /// on reads (Gemma 4's shared layers): the CPU computes them and hands
    /// each pass's new rows over ([`MirrorRows`]), written into a pool this
    /// device holds for them under the same block tables.
    pub(crate) mirror: Vec<usize>,
}

impl GpuTransformer {
    /// [`GpuTransformer::load_placed`] with every expert resident, one
    /// device and room for one sequence (tests).
    #[cfg(test)]
    pub(crate) fn load(
        content: &gguf_file::Content,
        path: &Path,
        context_length: usize,
        device: LocalDevice,
    ) -> Result<Self, CucaError> {
        Self::load_placed(
            content,
            path,
            context_length,
            device,
            &[],
            &ExpertPlacement::Resident,
            &Sizing::single(context_length),
        )
    }

    /// Load the model from the GGUF file at `path`, whose index `content`
    /// was read already, onto `device` and, for tensor parallelism (see
    /// [`tp`]), the `shards` after it: with K/V pools and recurrent-state
    /// slots for `sizing` (`context_length` positions per sequence), its
    /// experts placed as `experts` says (`Tiered`: left in the file, see the
    /// module docs). With no `shards` the model runs on `device` alone. A
    /// device may appear several times (`device` and a shard, or two
    /// shards): the model is still split and reduced across them, which
    /// saves no time but exercises the same code.
    ///
    /// # Errors
    ///
    /// [`CucaError::Config`] when `context_length` is zero or above the
    /// checkpoint maximum, a device is not a GPU compiled into this build,
    /// the `experts` budgets do not fit the model (no experts, a device or
    /// RAM tier below `n_used` experts per layer, heat of another shape) or
    /// the model cannot be split across `1 + shards.len()` devices (checked
    /// before any weight is read, see [`tp::validate`]);
    /// [`CucaError::Provider`] for unsupported or malformed checkpoints and
    /// a device that fails to open; [`CucaError::Io`] when the file cannot
    /// be read.
    pub(crate) fn load_placed(
        content: &gguf_file::Content,
        path: &Path,
        context_length: usize,
        device: LocalDevice,
        shards: &[LocalDevice],
        experts: &ExpertPlacement,
        sizing: &Sizing,
    ) -> Result<Self, CucaError> {
        Self::load_part(
            content,
            path,
            context_length,
            device,
            shards,
            experts,
            sizing,
            None,
            false,
        )
    }

    /// Load the last layers of the model (`back.first..`, with the final
    /// norm and the LM head) onto `device` for layer offload (see
    /// [`split`](super::split)): the layers before them run elsewhere and
    /// hand their output to [`GpuTransformer::forward_back`]. The K/V pools,
    /// recurrent states and expert tiers cover those layers only.
    ///
    /// # Errors
    ///
    /// As [`GpuTransformer::load_placed`].
    pub(crate) fn load_back(
        content: &gguf_file::Content,
        path: &Path,
        context_length: usize,
        device: LocalDevice,
        experts: &ExpertPlacement,
        sizing: &Sizing,
        back: &Back,
    ) -> Result<Self, CucaError> {
        Self::load_part(
            content,
            path,
            context_length,
            device,
            &[],
            experts,
            sizing,
            Some(back),
            false,
        )
    }

    /// [`GpuTransformer::load_placed`] with the hosted experts held as f32
    /// (tests): candle's quantized matmul rounds activations to Q8_0, which
    /// the device kernels do not, so the exact reference of the host path.
    #[cfg(test)]
    pub(crate) fn load_host_exact(
        content: &gguf_file::Content,
        path: &Path,
        context_length: usize,
        device: LocalDevice,
        experts: &ExpertPlacement,
        sizing: &Sizing,
    ) -> Result<Self, CucaError> {
        Self::load_part(
            content,
            path,
            context_length,
            device,
            &[],
            experts,
            sizing,
            None,
            true,
        )
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "the file, the placement and the part of the model are independent inputs"
    )]
    fn load_part(
        content: &gguf_file::Content,
        path: &Path,
        context_length: usize,
        device: LocalDevice,
        shards: &[LocalDevice],
        experts: &ExpertPlacement,
        sizing: &Sizing,
        back: Option<&Back>,
        host_exact: bool,
    ) -> Result<Self, CucaError> {
        let hp = read_hparams(content)?;
        if hp.arch.is_drafter() {
            return Err(err(format!(
                "`{}` is a multi-token-prediction drafter, not a standalone model: pass it as the `drafter` of the model it was trained for",
                hp.arch.name()
            )));
        }
        let hybrid = hp.behavior.hybrid_delta_net;
        check_context(&hp, context_length)?;
        let world = 1 + shards.len();
        tp::validate(content, &hp, world, experts)?;
        let locals: Vec<LocalDevice> = std::iter::once(device)
            .chain(shards.iter().copied())
            .collect();
        let devices = locals
            .iter()
            .map(|d| open_device(*d))
            .collect::<Result<Vec<_>, _>>()?;

        // Every shard cuts equal shapes, so one plan serves all.
        let first = back.map_or(0, |b| b.first);
        let host = match experts {
            ExpertPlacement::Host { layers } => {
                if hp.moe.is_none() {
                    return Err(CucaError::Config(format!(
                        "host experts need a mixture-of-experts model; `{}` has no experts",
                        hp.arch.name()
                    )));
                }
                Some(HostPlan {
                    lead: hp.leading_dense,
                    n: layers.unwrap_or(usize::MAX),
                    path: path.to_path_buf(),
                    hp: hp.clone(),
                    dequantize: host_exact,
                })
            }
            _ => None,
        };
        let tiers = match experts {
            ExpertPlacement::Resident | ExpertPlacement::Host { .. } => None,
            ExpertPlacement::Tiered(t) => Some((
                t,
                open_tiers(
                    content,
                    path,
                    &hp,
                    t,
                    first,
                    back.and_then(|b| b.shared.clone()),
                )?,
            )),
        };

        let embed = RowTable::open(
            path,
            content,
            name::TOKEN_EMBD,
            "embedding table",
            hp.n_embd,
        )?;
        let vocab = embed.rows;
        let ple_table = match hp.ple_dim {
            Some(dim) => Some(RowTable::open(
                path,
                content,
                name::PER_LAYER_TOKEN_EMBD,
                "per-layer embedding table",
                hp.n_layer * dim,
            )?),
            None => None,
        };

        let plan = tiers.as_ref().map(|(_, plan)| plan);
        let mut weights: Vec<Weights> = Vec::with_capacity(world);
        for (shard, device) in devices.iter().enumerate() {
            let mut reader = WeightReader::open(content, path, device, host.clone())?;
            weights.push(Weights::load(
                &mut reader,
                path,
                &hp,
                Tp { world, shard },
                context_length,
                plan,
                first,
            )?);
        }
        if let Some((tiers, plan)) = &tiers
            && let Some(heat) = &tiers.warm_from
        {
            plan.warm(&weights[0], heat)?;
        }

        // The pool layout depends on the layers' windows and K/V sources,
        // which every shard shares; each shard's pools hold its own heads.
        let layout = KvLayout::sized(&weights[0].hp, context_length, sizing);
        let kv = weights
            .iter()
            .map(|w| {
                alloc_pools(
                    &w.hp,
                    &layout,
                    &w.device,
                    first,
                    back.map_or(&[][..], |b| &b.mirror),
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        #[cfg(test)]
        let solo = Solo::new(&layout);
        // One zero state per delta-net layer, shared by every empty slot
        // (the states are immutable); a hybrid model is never split.
        let zero: Vec<Option<DeltaState>> = weights[0]
            .hp
            .layers
            .iter()
            .enumerate()
            .map(|(il, spec)| {
                spec.delta()
                    .filter(|_| il >= first)
                    .map(|d| DeltaState::zeros(d, &weights[0].device))
            })
            .collect();
        let slots = if hybrid {
            (0..sizing.max_sequences)
                .map(|_| DeltaSlot {
                    len: 0,
                    layers: zero.clone(),
                })
                .collect()
        } else {
            Vec::new()
        };
        Ok(Self {
            weights,
            collective: Collective::new(&locals, &devices),
            context_length,
            vocab,
            embed: Mutex::new(embed),
            ple_table,
            kv,
            layout,
            #[cfg(test)]
            solo,
            slots,
            zero,
            hybrid,
            tiers: tiers.map(|(_, plan)| Arc::clone(plan.store.shared())),
        })
    }

    /// The devices the model is split across (1 when it is not).
    pub(crate) fn world(&self) -> usize {
        self.weights.len()
    }

    /// Counters and heat of the expert tiers; `None` unless loaded with
    /// `ExpertPlacement::Tiered`.
    pub(crate) fn expert_tiers(&self) -> Option<Arc<TierShared>> {
        self.tiers.clone()
    }

    /// Run the rows of several sequences through the model in one pass (a
    /// forward step): `batch.rows[i].tokens` at positions `start_pos..`,
    /// their K/V written into the blocks of its `tables` (one table per group
    /// of [`GpuTransformer::kv_layout`], every block the call writes already
    /// allocated) and the positions before them read from there. Returns, per
    /// sequence, the logits `want` selects and the final-norm hidden rows
    /// `hidden` selects.
    ///
    /// Every dense operation (embedding, projections, FFN or MoE, norms, the
    /// LM head) runs once over the concatenated rows of all sequences, so
    /// each weight is read once per step; attention reads each sequence's own
    /// blocks (one launch of the decode kernel for all sequences when every
    /// sequence is a single row, else one launch group per sequence) and the
    /// recurrent layers advance each sequence's own state slot.
    ///
    /// A hybrid model's recurrent state sits at one position per state slot
    /// (the end of the slot's last call): `start_pos` must equal it, and a
    /// slot appears at most once per step.
    ///
    /// `batch.exact` is ignored: a batched pass on the GPU equals
    /// single-token steps only up to float rounding (the matmul kernels
    /// differ by shape), so a speculative greedy run can pick a different
    /// token than plain decoding where two logits are within that rounding.
    ///
    /// # Errors
    ///
    /// [`CucaError::Provider`] for an empty batch or a sequence without
    /// tokens, a hybrid model's `start_pos` other than its slot's position, a
    /// state slot out of range or used twice, more positions than the context
    /// length, a table that lacks a block the pass reads or writes or a
    /// failed device operation (the slots of every sequence of the step are
    /// then reset); [`CucaError::Io`] when an embedding row cannot be read.
    pub(crate) fn forward_batch(&mut self, batch: &Batch<'_>) -> Result<Vec<Forward>, CucaError> {
        let (tokens, segs) = self.plan_batch(batch)?;
        let hp = &self.weights[0].hp;
        let mut embeddings = self
            .embed
            .get_mut()
            .map_err(|_| err("the embedding table lock is poisoned"))?
            .rows(&tokens, hp.embd_scale)?;
        hp.norm_embeddings(&mut embeddings);
        let per_layer = match (&mut self.ple_table, hp.ple_dim) {
            (Some(table), Some(dim)) => Some(table.rows(&tokens, (dim as f32).sqrt())?),
            _ => None,
        };
        let outs = self.forward_layers(embeddings, per_layer, &segs);
        self.settle(&segs, outs)
    }

    /// [`GpuTransformer::forward_batch`] for the last layers of an offloaded
    /// model ([`GpuTransformer::load_back`]): `hidden` is the residual
    /// stream `[rows, n_embd]` the layers before them produced on the CPU
    /// and `per_layer` Gemma 4's per-layer inputs `[rows, n_layer,
    /// ple_dim]`; the hidden state crosses to the device once per call.
    ///
    /// # Errors
    ///
    /// As [`GpuTransformer::forward_batch`]; [`CucaError::Provider`] when
    /// `hidden` does not hold one row per token of the batch.
    pub(crate) fn forward_back(
        &mut self,
        batch: &Batch<'_>,
        hidden: Vec<f32>,
        per_layer: Option<Vec<f32>>,
        mirror: &[MirrorRows],
    ) -> Result<Vec<Forward>, CucaError> {
        let (tokens, segs) = self.plan_batch(batch)?;
        let n_embd = self.weights[0].hp.n_embd;
        if hidden.len() != tokens.len() * n_embd {
            return Err(err(format!(
                "the hidden state holds {} values for {} rows of {n_embd}",
                hidden.len(),
                tokens.len()
            )));
        }
        let outs = self.back_layers(hidden, per_layer, mirror, &segs);
        self.settle(&segs, outs)
    }

    /// Check the batch and describe its sequences: the concatenated tokens
    /// and one [`Seg`] per sequence.
    ///
    /// # Errors
    ///
    /// [`CucaError::Provider`] for an empty batch or a sequence without
    /// tokens, more positions than the context length, a hybrid model's
    /// `start_pos` other than its slot's position or a state slot out of
    /// range or used twice.
    fn plan_batch<'a>(&self, batch: &'a Batch<'_>) -> Result<(Vec<u32>, Vec<Seg<'a>>), CucaError> {
        if batch.rows.is_empty() {
            return Err(err("a forward step requires at least one sequence"));
        }
        let mut tokens: Vec<u32> = Vec::new();
        let mut segs = Vec::with_capacity(batch.rows.len());
        for (i, r) in batch.rows.iter().enumerate() {
            if r.tokens.is_empty() {
                return Err(err("forward requires at least one token"));
            }
            let hi = r.start_pos + r.tokens.len();
            if hi > self.context_length {
                return Err(err(format!(
                    "{hi} positions exceed the context length {}",
                    self.context_length
                )));
            }
            if self.hybrid {
                let Some(state) = self.slots.get(r.state.0) else {
                    return Err(err(format!(
                        "state slot {} is out of range (the model has {})",
                        r.state.0,
                        self.slots.len()
                    )));
                };
                if r.start_pos != state.len {
                    return Err(err(format!(
                        "forward start {} does not match the recurrent state's position {}",
                        r.start_pos, state.len
                    )));
                }
                if batch.rows[..i].iter().any(|o| o.state == r.state) {
                    return Err(err(format!(
                        "state slot {} appears twice in one step",
                        r.state.0
                    )));
                }
            }
            segs.push(Seg {
                start: r.start_pos,
                offset: tokens.len(),
                t: r.tokens.len(),
                tables: r.tables,
                slot: r.state.0,
                want: r.want,
                hidden: r.hidden,
            });
            tokens.extend_from_slice(r.tokens);
        }
        Ok((tokens, segs))
    }

    /// The end of a forward step: a failed pass empties the recurrent state
    /// of every sequence of the step (a half-advanced state cannot be
    /// repaired; the CPU engine does the same), a finished one moves it to
    /// the end of the sequence's rows.
    fn settle(
        &mut self,
        segs: &[Seg<'_>],
        outs: Result<Vec<Forward>, CucaError>,
    ) -> Result<Vec<Forward>, CucaError> {
        let outs = match outs {
            Ok(outs) => outs,
            Err(e) => {
                for seg in segs {
                    self.reset_slot(seg.slot);
                }
                return Err(e);
            }
        };
        if self.hybrid {
            for seg in segs {
                if let Some(state) = self.slots.get_mut(seg.slot) {
                    state.len = seg.start + seg.t;
                }
            }
        }
        Ok(outs)
    }

    /// The layers this load holds over the residual stream `hidden` of the
    /// sequences `segs` describe, then the final norm and the LM head.
    fn back_layers(
        &mut self,
        hidden: Vec<f32>,
        per_layer: Option<Vec<f32>>,
        mirror: &[MirrorRows],
        segs: &[Seg<'_>],
    ) -> Result<Vec<Forward>, CucaError> {
        let ([weights], [kv]) = (self.weights.as_slice(), self.kv.as_mut_slice()) else {
            return Err(err(
                "layer offload cannot be combined with tensor parallelism",
            ));
        };
        let hp = &weights.hp;
        let t = hidden.len() / hp.n_embd;
        let x = Tensor::<2>::from_data(TensorData::new(hidden, [t, hp.n_embd]), &weights.device);
        let ple = match (per_layer, hp.ple_dim) {
            (Some(rows), Some(dim)) => Some(Tensor::<3>::from_data(
                TensorData::new(rows, [t, hp.n_layer, dim]),
                &weights.device,
            )),
            _ => None,
        };
        weights.run_layers(x, ple, mirror, kv, &self.layout, &mut self.slots, segs)
    }

    /// The layers over the concatenated rows `embeddings` of the sequences
    /// `segs` describe, then the final norm and the LM head over the rows
    /// they want ([`Weights::forward_layers`] on one device).
    ///
    /// A split model runs every layer on every shard: each shard's attention
    /// half, an all-reduce of the partial outputs, each shard's residual
    /// tail, each shard's FFN half, a second all-reduce and each shard's
    /// residual tail. Every shard holds the whole residual stream, so the
    /// final norm and the LM head run on shard 0's copy.
    ///
    /// # Errors
    ///
    /// As [`Weights::forward_layers`]; [`CucaError::Provider`] for a failed
    /// all-reduce.
    fn forward_layers(
        &mut self,
        embeddings: Vec<f32>,
        per_layer: Option<Vec<f32>>,
        segs: &[Seg<'_>],
    ) -> Result<Vec<Forward>, CucaError> {
        if let ([weights], [kv]) = (self.weights.as_slice(), self.kv.as_mut_slice()) {
            return weights.forward_layers(
                kv,
                &self.layout,
                &mut self.slots,
                embeddings,
                per_layer,
                segs,
            );
        }
        let (mut embeddings, mut per_layer) = (embeddings, per_layer);
        let n = self.weights.len();
        let mut xs: Vec<Tensor<2>> = Vec::with_capacity(n);
        let mut ples: Vec<Option<Tensor<3>>> = Vec::with_capacity(n);
        for (s, weights) in self.weights.iter().enumerate() {
            let (e, p) = if s + 1 == n {
                (std::mem::take(&mut embeddings), per_layer.take())
            } else {
                (embeddings.clone(), per_layer.clone())
            };
            let (x, ple) = weights.embed_input(e, p);
            xs.push(x);
            ples.push(ple);
        }
        let seqs = trunk_seqs(&self.layout, segs);
        let passes = self
            .weights
            .iter()
            .zip(ples)
            .map(|(weights, ple)| weights.pass(ple, &seqs))
            .collect::<Result<Vec<_>, _>>()?;
        for il in 0..self.weights[0].hp.n_layer {
            // Attention: each shard's heads, then the sum of their outputs.
            let mut parts = Vec::with_capacity(n);
            for (((weights, x), pass), kv) in self
                .weights
                .iter()
                .zip(&xs)
                .zip(&passes)
                .zip(self.kv.iter_mut())
            {
                parts.push(weights.mixer(il, x, pass, kv, &mut [], None)?);
            }
            let sums = self.collective.all_reduce_sum(parts)?;
            xs = self
                .weights
                .iter()
                .zip(xs)
                .zip(sums)
                .map(|((weights, x), sum)| weights.attn_tail(il, x, sum))
                .collect();
            // The FFN: each shard's slice of it, then the sum.
            let mut parts = Vec::with_capacity(n);
            for (weights, x) in self.weights.iter().zip(&xs) {
                parts.push(weights.ffn_half(il, x)?);
            }
            let sums = self.collective.all_reduce_sum(parts)?;
            xs = self
                .weights
                .iter()
                .zip(xs)
                .zip(sums)
                .zip(&passes)
                .map(|(((weights, x), sum), pass)| weights.ffn_tail(il, x, sum, pass))
                .collect();
        }
        let Some(x) = xs.into_iter().next() else {
            return Err(err("a model without devices"));
        };
        self.weights[0].finish(x, segs)
    }

    /// Vocabulary size.
    pub(crate) fn vocab(&self) -> usize {
        self.vocab
    }

    /// Width of the hidden state (and of an embedding row).
    pub(crate) fn n_embd(&self) -> usize {
        self.weights[0].hp.n_embd
    }

    /// The architecture.
    pub(crate) fn arch(&self) -> Arch {
        self.weights[0].hp.arch
    }

    /// The `n_embd`-wide embedding row of `token`, scaled as the first layer
    /// sees it: a drafter's input half next to the target's hidden state.
    ///
    /// # Errors
    ///
    /// [`CucaError::Io`] when the row cannot be read from the file.
    pub(crate) fn embedding_row(&self, token: u32) -> Result<Vec<f32>, CucaError> {
        self.embed
            .lock()
            .map_err(|_| err("the embedding table lock is poisoned"))?
            .rows(&[token], self.weights[0].hp.embd_scale)
    }

    /// The target's paged K/V a drafter reads, for the sequence whose block
    /// tables are `tables`: the pools of the last sliding-window layer's
    /// source and of the last full-attention layer's source (the layers
    /// Gemma 4's shared-K/V layers read), `len` positions of them. The
    /// handles are clones: hold them only for the length of one drafting
    /// call, since a K/V write while another handle to a pool exists copies
    /// the whole pool instead of writing in place.
    ///
    /// # Errors
    ///
    /// [`CucaError::Provider`] when the model has no sliding or no
    /// full-attention K/V-owning layer to share, or `tables` has no table
    /// for its group.
    fn ext_kv(&self, tables: &[BlockTable], len: usize) -> Result<GpuExtKv, CucaError> {
        let source = |sliding: bool| -> Result<ExtGroup, CucaError> {
            let layer = self.weights[0]
                .hp
                .layers
                .iter()
                .rev()
                .find(|l| l.window.is_some() == sliding)
                .ok_or_else(|| {
                    err(format!(
                        "the drafter needs a target with {} attention layers",
                        if sliding { "sliding-window" } else { "full" }
                    ))
                })?;
            let pool = self.kv[0]
                .get(layer.kv_source)
                .and_then(Option::as_ref)
                .ok_or_else(|| err(format!("layer {} owns no K/V cache", layer.kv_source)))?;
            let table = self.layout.layer_group[layer.kv_source]
                .and_then(|g| tables.get(g))
                .ok_or_else(|| err("the sequence has no block table for the drafter's layer"))?;
            Ok(ExtGroup {
                k: pool.k.clone(),
                v: pool.v.clone().ok_or_else(|| {
                    err("the drafter reads a target with a latent K/V cache, which has no value pool")
                })?,
                table: table.clone(),
            })
        };
        Ok(GpuExtKv {
            sliding: source(true)?,
            full: source(false)?,
            len,
        })
    }

    /// The block pools' layout: groups and blocks per group.
    pub(crate) fn kv_layout(&self) -> &KvLayout {
        &self.layout
    }

    /// Bytes of K/V data the pools of every shard hold: with KV-head
    /// replication a shard keeps whole heads that other shards keep too, so
    /// a split model holds more than the same model on one device.
    pub(crate) fn kv_pool_bytes(&self) -> u64 {
        self.weights
            .iter()
            .map(|w| self.layout.pool_bytes(&w.hp))
            .sum()
    }

    /// Take the recurrent state of `slot` back to at most `len` positions
    /// and return the position it now sits at: the next forward call of a
    /// hybrid model for the slot must start there. K/V rows are positional
    /// and live in the sequence's blocks, so an attention-only model has no
    /// state to move and returns `len`; the caller truncates its block tables
    /// to what it gets back. A hybrid model's state summarises everything
    /// before the slot's current end and cannot be undone: it retains only
    /// that end (`len >=` the slot's position returns the position) and the
    /// zero state (`len == 0`); any other length resets the slot and returns
    /// `0`, so the caller recomputes from the start or
    /// [`GpuTransformer::restore`]s a snapshot. Unlike the CPU engine it keeps
    /// no verify record (no drafter can pair with a hybrid model on a GPU).
    pub(crate) fn rewind(&mut self, slot: usize, len: usize) -> usize {
        if !self.hybrid {
            return len;
        }
        let Some(state) = self.slots.get(slot) else {
            return 0;
        };
        if len != 0 && len >= state.len {
            return state.len;
        }
        self.reset_slot(slot);
        0
    }

    /// The zero state at position 0 for `slot` (a sequence start, or after a
    /// failed pass). A no-op for an attention-only model and a slot out of
    /// range. Free: the zero tensors are shared handles.
    pub(crate) fn reset_slot(&mut self, slot: usize) {
        if let Some(state) = self.slots.get_mut(slot) {
            state.layers.clone_from(&self.zero);
            state.len = 0;
        }
    }

    /// Handles to the recurrent state of every delta-net layer for `slot`
    /// (after its last forward call): `None` for an attention-only model or
    /// a slot out of range. Free: the states are immutable device tensors.
    pub(crate) fn snapshot(&self, slot: usize) -> Option<GpuSnapshot> {
        self.slots.get(slot).map(|state| GpuSnapshot {
            layers: state.layers.clone(),
        })
    }

    /// Make `snapshot` (taken from this model) the state of `slot`, which
    /// then sits at position `len`.
    ///
    /// # Errors
    ///
    /// [`CucaError::Provider`] for a slot out of range (every slot of an
    /// attention-only model) or a snapshot of another model (a different
    /// layer count or mix of recurrent layers).
    pub(crate) fn restore(
        &mut self,
        slot: usize,
        snapshot: &GpuSnapshot,
        len: usize,
    ) -> Result<(), CucaError> {
        let fits = snapshot.layers.len() == self.zero.len()
            && snapshot
                .layers
                .iter()
                .zip(&self.zero)
                .all(|(a, b)| a.is_some() == b.is_some());
        let state = self
            .slots
            .get_mut(slot)
            .filter(|_| fits)
            .ok_or_else(|| err("the snapshot does not fit this model's recurrent state"))?;
        state.layers.clone_from(&snapshot.layers);
        state.len = len;
        Ok(())
    }

    /// Whether the model has recurrent layers.
    pub(crate) fn is_hybrid(&self) -> bool {
        self.hybrid
    }

    /// Most rows of one forward call whose every intermediate position can
    /// be returned to: unbounded for an attention-only model, 1 for a hybrid
    /// one (no verify record is kept).
    pub(crate) fn rollback_rows(&self) -> usize {
        if self.hybrid { 1 } else { usize::MAX }
    }
}

/// The single-sequence calls the unit tests use: one implicit sequence over
/// the model's own pools ([`Solo`]). The engine passes its sequences' block
/// tables explicitly.
#[cfg(test)]
impl GpuTransformer {
    /// [`GpuTransformer::forward_ext`] logits only, not exact.
    pub(crate) fn forward(
        &mut self,
        tokens: &[u32],
        start_pos: usize,
        want: Want,
    ) -> Result<Option<Vec<f32>>, CucaError> {
        Ok(self
            .forward_ext(tokens, start_pos, want, Hidden::None, false)?
            .logits)
    }

    /// [`GpuTransformer::forward_batch`] over the implicit sequence, in
    /// state slot 0: its blocks are allocated here; `start_pos` must equal
    /// its length.
    pub(crate) fn forward_ext(
        &mut self,
        tokens: &[u32],
        start_pos: usize,
        want: Want,
        hidden: Hidden,
        exact: bool,
    ) -> Result<Forward, CucaError> {
        let seq = self.solo.begin(start_pos, tokens.len())?;
        let batch = Batch::single(tokens, start_pos, &seq.tables, want, hidden, exact);
        let out = self.forward_batch(&batch).and_then(|mut outs| {
            if outs.len() == 1 {
                Ok(outs.swap_remove(0))
            } else {
                Err(err("a single-sequence step returned several results"))
            }
        });
        self.solo
            .finish(seq, out.is_ok().then_some(start_pos + tokens.len()));
        out
    }

    /// Drop the positions from `len` on of the implicit sequence and return
    /// how many it retains (see [`GpuTransformer::rewind`] and
    /// [`KvPools::truncate`](super::paged::KvPools::truncate)).
    pub(crate) fn truncate_cache(&mut self, len: usize) -> usize {
        let state = self.rewind(0, len).min(self.solo.len());
        let kept = self.solo.truncate(state);
        if kept != state {
            self.rewind(0, kept);
        }
        kept
    }

    /// Positions of the implicit sequence.
    pub(crate) fn cache_len(&self) -> usize {
        self.solo.len()
    }

    /// The implicit sequence's block tables, for a drafter's `draft`.
    pub(crate) fn solo_tables(&self) -> &[BlockTable] {
        self.solo.tables()
    }
}

/// The pools and block table of one window group of the target a drafter
/// reads.
struct ExtGroup {
    k: Tensor<4>,
    v: Tensor<4>,
    table: BlockTable,
}

/// The target's paged K/V a drafter attends to; see
/// [`GpuTransformer::ext_kv`].
struct GpuExtKv {
    sliding: ExtGroup,
    full: ExtGroup,
    /// Positions of the target's cache the drafter reads.
    len: usize,
}

impl Weight {
    /// Rows of the whole weight (the output width of [`linear`]).
    fn rows(&self) -> usize {
        self.parts.iter().map(Part::rows).sum()
    }
}

/// A Gemma 4 multi-token-prediction drafter (`gemma4-assistant`) on the GPU,
/// mirroring the CPU [`Drafter`](super::mtp::Drafter): a four-layer,
/// query-only decoder whose layers attend to the K/V cache of the target
/// ([`GpuTransformer`]) it drafts for, so it owns no cache and adds none.
///
/// Its weights follow the rules of the module docs; the F16 token embedding
/// that serves as its LM head (`262144 x 256` for the E2B drafter) is held as
/// f32, 268 MB.
pub(crate) struct GpuDrafter {
    net: Weights,
    /// `nextn.pre_projection`: `2 * n_embd_out` to the drafter width.
    pre: Weight,
    /// `nextn.post_projection`: the drafter width to `n_embd_out`.
    post: Weight,
    context_length: usize,
}

impl Drop for GpuDrafter {
    fn drop(&mut self) {
        self.pre.parts.clear();
        self.post.parts.clear();
        self.net.release();
        return_device_memory(&self.net.device);
    }
}

impl GpuDrafter {
    /// Load the drafter at `path` (index `content`) onto the device of
    /// `target`, with RoPE tables for `context_length` positions.
    ///
    /// # Errors
    ///
    /// [`CucaError::Provider`] when `target` is not a Gemma 4 model, the file
    /// is not a drafter, the hidden widths, vocabularies or K/V geometry
    /// differ, or a weight cannot be read.
    pub(crate) fn open(
        content: &gguf_file::Content,
        path: &Path,
        target: &GpuTransformer,
        context_length: usize,
    ) -> Result<Self, CucaError> {
        if target.world() > 1 {
            return Err(CucaError::Config(format!(
                "tensor parallelism: a drafter cannot pair with a model split across {} devices",
                target.world()
            )));
        }
        if target.arch() != Arch::Gemma4 {
            return Err(err(format!(
                "the `gemma4-assistant` drafter pairs with a `gemma4` model, not `{}`",
                target.arch().name()
            )));
        }
        let hp = read_hparams(content)?;
        if !hp.arch.is_drafter() {
            return Err(err(format!(
                "`{}` is not a drafter architecture",
                hp.arch.name()
            )));
        }
        check_context(&hp, context_length)?;
        if hp.n_embd_out != target.n_embd() {
            return Err(err(format!(
                "drafter hidden width {} differs from the target's {}: it was trained for another model size",
                hp.n_embd_out,
                target.n_embd()
            )));
        }
        let target_layers = &target.weights[0].hp.layers;
        for (il, spec) in hp.layers.iter().enumerate() {
            let sliding = spec.window.is_some();
            let source = target_layers
                .iter()
                .rev()
                .find(|l| l.window.is_some() == sliding)
                .and_then(|l| target_layers.get(l.kv_source));
            if !source.is_some_and(|s| s.n_kv_head == spec.n_kv_head && s.head_dim == spec.head_dim)
            {
                return Err(err(format!(
                    "drafter layer {il} ({} attention) expects {} K/V heads of width {}, which the target's cache does not have",
                    if sliding { "sliding-window" } else { "full" },
                    spec.n_kv_head,
                    spec.head_dim
                )));
            }
        }
        let (n_embd, n_embd_out) = (hp.n_embd, hp.n_embd_out);
        let mut reader = WeightReader::open(content, path, &target.weights[0].device, None)?;
        let pre = reader.matrix("nextn.pre_projection.weight", n_embd, 2 * n_embd_out)?;
        let post = reader.matrix("nextn.post_projection.weight", n_embd_out, n_embd)?;
        let net = Weights::load(&mut reader, path, &hp, Tp::SOLO, context_length, None, 0)?;
        let vocab = net.head_parts()?.lm_head.rows();
        if vocab != target.vocab() {
            return Err(err(format!(
                "drafter vocabulary {vocab} differs from the target's {}",
                target.vocab()
            )));
        }
        Ok(Self {
            net,
            pre,
            post,
            context_length,
        })
    }

    /// Draft at most `k` tokens after the pending token `first` of the
    /// sequence whose block tables (one per K/V group of the target) are
    /// `tables`, as the CPU drafter.
    ///
    /// `hidden` is the target's final-norm hidden state at its last cached
    /// position (`Hidden::PostNorm` of its last forward pass), `len` the
    /// number of cached positions (`first` sits at position `len` and is not
    /// cached). Each step's logits go to `pick`, which chooses the token or
    /// ends the chain with `None`; the returned drafts are the tokens `pick`
    /// accepted. Every step uses the position `len`.
    ///
    /// # Errors
    ///
    /// [`CucaError::Provider`] when `hidden` has the wrong width, `len` is
    /// zero, beyond the context length or the blocks `tables` holds, a
    /// block it reads is released, or a device operation fails.
    #[expect(
        clippy::too_many_arguments,
        reason = "one drafting call's inputs; a struct would only rename them"
    )]
    pub(crate) fn draft(
        &self,
        target: &GpuTransformer,
        tables: &[BlockTable],
        hidden: &[f32],
        first: u32,
        len: usize,
        k: usize,
        pick: &mut dyn FnMut(&mut [f32]) -> Option<u32>,
    ) -> Result<Vec<u32>, CucaError> {
        let hp = &self.net.hp;
        if hidden.len() != hp.n_embd_out {
            return Err(err(format!(
                "drafter hidden state has {} floats, expected {}",
                hidden.len(),
                hp.n_embd_out
            )));
        }
        if len == 0 || len >= self.context_length {
            return Err(err(format!(
                "cannot draft at position {len}: the drafter's context is {}",
                self.context_length
            )));
        }
        // Handles to the target's pools live for this call only.
        let ext = target.ext_kv(tables, len)?;
        let pass = self.net.pass(
            None,
            &[PassSeq {
                start: len,
                t: 1,
                slot: 0,
                kv: PassKv::Drafter(&ext),
            }],
        )?;
        let mut no_cache: KvLayers = Vec::new();
        let head = self.net.head_parts()?;
        let mut drafts = Vec::with_capacity(k);
        let mut hidden = hidden.to_vec();
        let mut token = first;
        for _ in 0..k {
            let mut input = target.embedding_row(token)?;
            input.extend_from_slice(&hidden);
            let width = input.len();
            let x = Tensor::<2>::from_data(TensorData::new(input, [1, width]), &self.net.device);
            let mut x = linear(x, &self.pre);
            for il in 0..hp.n_layer {
                x = self
                    .net
                    .layer(il, x, &pass, &mut no_cache, &mut [], Some(&ext))?;
            }
            let normed = rms(x, Some(&head.norm), hp.eps);
            let next = linear(normed.clone(), &self.post);
            let mut logits = read_f32(self.net.head(normed)?, "drafter logits")?;
            let Some(next_token) = pick(&mut logits) else {
                break;
            };
            hidden = read_f32(next, "drafter hidden state")?;
            token = next_token;
            drafts.push(token);
        }
        Ok(drafts)
    }
}

/// Read the attention weights of layer `il` (a gated attention packs
/// `[query | gate]` per head into its query projection, twice the width),
/// cut to the heads shard `tp` computes. `hp` and `spec` describe the layer
/// as it is in the file.
fn load_attention(
    reader: &mut WeightReader<'_>,
    hp: &Hparams,
    tp: Tp,
    il: usize,
    spec: &LayerSpec,
    layout: &LayerLayout,
) -> Result<AttnWeights, CucaError> {
    let n_embd = hp.n_embd;
    let (nh, nkv, d) = (spec.n_head, spec.n_kv_head, spec.head_dim);
    let q_width = nh * d;
    let q_rows = if hp.behavior.gated_attention {
        2 * q_width
    } else {
        q_width
    };
    let kv_width = nkv * d;
    let blk = |part: &str| name::blk(il, part);
    let owns = spec.owns_kv(il);
    let proj = |reader: &mut WeightReader<'_>, part: &str, shape: [usize; 2], bias: bool| {
        let cut = tp::part_cut(hp, layout, il, part, (shape[0], shape[1]), tp);
        reader.proj(il, part, shape, bias, &cut)
    };

    let qkv = match layout.qkv {
        Qkv::Fused => {
            if !owns {
                return Err(err(format!(
                    "layer {il}: a fused attn_qkv layer cannot share another layer's K/V"
                )));
            }
            QkvWeights::Fused(proj(
                reader,
                name::ATTN_QKV,
                [q_width + 2 * kv_width, n_embd],
                layout.biases.q,
            )?)
        }
        Qkv::Split => QkvWeights::Split {
            q: proj(reader, name::ATTN_Q, [q_rows, n_embd], layout.biases.q)?,
            kv: if owns {
                Some((
                    proj(reader, name::ATTN_K, [kv_width, n_embd], layout.biases.k)?,
                    proj(reader, name::ATTN_V, [kv_width, n_embd], layout.biases.v)?,
                ))
            } else {
                None
            },
        },
    };
    Ok(AttnWeights {
        qkv,
        q_norm: reader.opt_vector(
            layout.q_norm,
            &blk(name::ATTN_Q_NORM),
            if hp.behavior.qk_norm_full { q_width } else { d },
        )?,
        k_norm: reader.opt_vector(
            layout.k_norm,
            &blk(name::ATTN_K_NORM),
            if hp.behavior.qk_norm_full {
                kv_width
            } else {
                d
            },
        )?,
        v_norm: hp.behavior.gemma4_extras && owns,
        gate: if layout.attn_gate {
            Some(proj(reader, name::ATTN_GATE, [q_width, n_embd], false)?)
        } else {
            None
        },
        // Its columns follow the shard's heads; its bias (whole) is added
        // once after the shards' outputs are summed.
        wo: proj(
            reader,
            name::ATTN_OUTPUT,
            [n_embd, q_width],
            layout.biases.o,
        )?,
    })
}

/// Transpose each of the `heads` row-major `[rows, cols]` matrices of
/// `values` to `[cols, rows]`.
fn transpose_heads(values: &[f32], heads: usize, rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0f32; values.len()];
    for h in 0..heads {
        for r in 0..rows {
            for c in 0..cols {
                out[(h * cols + c) * rows + r] = values[(h * rows + r) * cols + c];
            }
        }
    }
    out
}

/// Read the latent-attention weights of layer `il`: the low-rank query, the
/// latent projection and the per-head up-projections (dequantized on the
/// host, held as f32: two small batched products use them).
fn load_mla(
    reader: &mut WeightReader<'_>,
    hp: &Hparams,
    il: usize,
    spec: &LayerSpec,
    mla: &MlaSpec,
) -> Result<MlaWeights, CucaError> {
    let n_embd = hp.n_embd;
    let nh = spec.n_head;
    let blk = |part: &str| name::blk(il, part);
    // Stored `[n_head, kv_lora, nope]` and `[n_head, v_head, kv_lora]`.
    let k_b = reader.host_floats(&blk(name::ATTN_K_B), &[nh, mla.kv_lora, mla.nope])?;
    let v_b = reader.host_floats(&blk(name::ATTN_V_B), &[nh, mla.v_head, mla.kv_lora])?;
    let k_b = transpose_heads(&k_b, nh, mla.kv_lora, mla.nope);
    let v_b = transpose_heads(&v_b, nh, mla.v_head, mla.kv_lora);
    Ok(MlaWeights {
        q_a: reader.matrix(&blk(name::ATTN_Q_A), mla.q_lora, n_embd)?,
        q_a_norm: reader.vector(&blk(name::ATTN_Q_A_NORM), mla.q_lora)?,
        q_b: reader.matrix(&blk(name::ATTN_Q_B), nh * mla.q_head(), mla.q_lora)?,
        kv_a: reader.matrix(&blk(name::ATTN_KV_A_MQA), mla.latent_width(), n_embd)?,
        kv_a_norm: reader.vector(&blk(name::ATTN_KV_A_NORM), mla.kv_lora)?,
        k_b: Tensor::from_data(
            TensorData::new(k_b, [nh, mla.nope, mla.kv_lora]),
            reader.device,
        ),
        v_b: Tensor::from_data(
            TensorData::new(v_b, [nh, mla.kv_lora, mla.v_head]),
            reader.device,
        ),
        wo: Proj {
            w: reader.matrix(&blk(name::ATTN_OUTPUT), n_embd, nh * mla.v_head)?,
            bias: None,
        },
    })
}

/// Read the gated delta-net weights of layer `il`.
fn load_delta(
    reader: &mut WeightReader<'_>,
    hp: &Hparams,
    il: usize,
    spec: &DeltaSpec,
) -> Result<DeltaWeights, CucaError> {
    let n_embd = hp.n_embd;
    let (cd, taps, nv) = (spec.conv_dim(), spec.conv_kernel, spec.n_v_heads);
    let blk = |part: &str| name::blk(il, part);
    // Stored `[conv_dim][conv_kernel]`; the taps go to the device as
    // `[conv_kernel, conv_dim]` so one tap is a contiguous row.
    let stored = reader.host_floats(&blk(name::SSM_CONV1D), &[cd, taps])?;
    let mut conv = vec![0f32; taps * cd];
    for c in 0..cd {
        for i in 0..taps {
            conv[i * cd + c] = stored[c * taps + i];
        }
    }
    let mut beta_alpha = reader.host_floats(&blk(name::SSM_BETA), &[nv, n_embd])?;
    beta_alpha.extend(reader.host_floats(&blk(name::SSM_ALPHA), &[nv, n_embd])?);
    let beta_alpha = Weight {
        parts: vec![Part::Float(Tensor::from_data(
            TensorData::new(beta_alpha, [2 * nv, n_embd]),
            reader.device,
        ))],
    };
    let mut head_row = |tensor: String| -> Result<Tensor<2>, CucaError> {
        let values = reader.host_floats(&tensor, &[nv])?;
        Ok(Tensor::from_data(
            TensorData::new(values, [1, nv]),
            reader.device,
        ))
    };
    let a = head_row(name::blk_bare(il, name::SSM_A))?;
    let dt_bias = head_row(name::blk_bias(il, name::SSM_DT))?;
    Ok(DeltaWeights {
        spec: *spec,
        qkv: reader.matrix(&blk(name::ATTN_QKV), cd, n_embd)?,
        gate: reader.matrix(&blk(name::ATTN_GATE), spec.value_dim(), n_embd)?,
        beta_alpha,
        out: reader.matrix(&blk(name::SSM_OUT), n_embd, spec.value_dim())?,
        conv: Tensor::from_data(TensorData::new(conv, [taps, cd]), reader.device),
        norm: reader.vector(&blk(name::SSM_NORM), spec.head_v_dim)?,
        a,
        dt_bias,
    })
}

/// Read every weight of layer `il` that shard `tp` holds; with `tiers`, the
/// layer's experts stay in the file. `hp` and `spec` describe the layer as
/// it is in the file.
fn load_layer(
    reader: &mut WeightReader<'_>,
    hp: &Hparams,
    tp: Tp,
    il: usize,
    spec: &LayerSpec,
    layout: &LayerLayout,
    tiers: Option<&TierPlan>,
) -> Result<LayerWeights, CucaError> {
    let n_embd = hp.n_embd;
    let blk = |part: &str| name::blk(il, part);
    let matrix = |reader: &mut WeightReader<'_>, part: &str, rows: usize, cols: usize| {
        let cut = tp::part_cut(hp, layout, il, part, (rows, cols), tp);
        reader.matrix_cut(&blk(part), (rows, cols), &cut)
    };

    let mixer = match (spec.delta(), spec.mla) {
        (Some(delta), _) => Mixer::Delta(Box::new(load_delta(reader, hp, il, delta)?)),
        (None, Some(mla)) => Mixer::Mla(Box::new(load_mla(reader, hp, il, spec, &mla)?)),
        (None, None) => Mixer::Attention(load_attention(reader, hp, tp, il, spec, layout)?),
    };

    let ffn = match (layout.ffn, hp.moe) {
        (Ffn::Dense { ff, fused }, _) => FfnWeights::Dense {
            gate_up: if fused {
                GateUp::Fused {
                    up: matrix(reader, name::FFN_UP, 2 * ff, n_embd)?,
                    ff: ff / tp.world,
                }
            } else {
                GateUp::Split {
                    gate: matrix(reader, name::FFN_GATE, ff, n_embd)?,
                    up: matrix(reader, name::FFN_UP, ff, n_embd)?,
                }
            },
            down: matrix(reader, name::FFN_DOWN, n_embd, ff)?,
        },
        (
            Ffn::Moe {
                ff,
                shared_ff,
                fused,
            },
            Some(spec),
        ) => FfnWeights::Moe {
            spec,
            router: reader.matrix(&blk(name::FFN_GATE_INP), spec.n_expert, n_embd)?,
            experts: reader.expert_bank(il, spec.n_expert, ff, n_embd, fused, tiers, tp)?,
            shared: match shared_ff {
                Some(sff) => {
                    let gate_inp = if hp.behavior.shared_expert_ungated {
                        None
                    } else {
                        let values =
                            reader.host_floats(&blk(name::FFN_GATE_INP_SHEXP), &[n_embd])?;
                        Some(Tensor::from_data(
                            TensorData::new(values, [1, n_embd]),
                            reader.device,
                        ))
                    };
                    Some(GpuShared {
                        gate: matrix(reader, name::FFN_GATE_SHEXP, sff, n_embd)?,
                        up: matrix(reader, name::FFN_UP_SHEXP, sff, n_embd)?,
                        down: matrix(reader, name::FFN_DOWN_SHEXP, n_embd, sff)?,
                        gate_inp,
                    })
                }
                None => None,
            },
        },
        (Ffn::Moe { .. }, None) => return Err(err("expert tensors without an expert count")),
    };

    let (ple, out_scale) = if hp.behavior.gemma4_extras {
        let ple = match hp.ple_dim {
            Some(dim) => Some(PleLayer {
                inp_gate: reader.matrix(&blk(name::INP_GATE), dim, n_embd)?,
                proj: reader.matrix(&blk(name::PROJ), n_embd, dim)?,
                post_norm: reader.vector(&blk(name::POST_NORM), n_embd)?,
            }),
            None => None,
        };
        (ple, Some(reader.scalar(&blk(name::LAYER_OUTPUT_SCALE))?))
    } else {
        (None, None)
    };

    Ok(LayerWeights {
        attn_norm: reader.opt_vector(hp.behavior.pre_norms, &blk(name::ATTN_NORM), n_embd)?,
        mixer,
        post_attn_norm: reader.opt_vector(
            layout.post_attn_norm,
            &blk(name::POST_ATTENTION_NORM),
            n_embd,
        )?,
        ffn_norm: reader.opt_vector(
            hp.behavior.pre_norms,
            &blk(ffn_norm_part(hp.behavior.ffn_norm_is_post_attention)),
            n_embd,
        )?,
        ffn,
        post_ffw_norm: reader.opt_vector(
            layout.post_ffw_norm,
            &blk(name::POST_FFW_NORM),
            n_embd,
        )?,
        ple,
        out_scale,
    })
}

/// One sequence of a forward step, as [`Weights::forward_layers`] hands it
/// to the K/V and recurrent-state code.
struct Seg<'a> {
    /// Position of its first row.
    start: usize,
    /// First row among the step's concatenated rows, and the row count.
    offset: usize,
    t: usize,
    tables: &'a [BlockTable],
    /// The recurrent-state slot (hybrid models).
    slot: usize,
    want: Want,
    hidden: Hidden,
}

/// Host-side row ranges, coalescing a range that continues the last one.
pub(super) fn push_rows(ranges: &mut Vec<Range<usize>>, rows: Range<usize>) {
    match ranges.last_mut() {
        Some(last) if last.end == rows.start => last.end = rows.end,
        _ => ranges.push(rows),
    }
}

/// The rows `ranges` (ascending, not empty) of `x`, one after the other.
fn take_rows(x: Tensor<2>, ranges: &[Range<usize>]) -> Tensor<2> {
    let [total, _] = x.dims();
    match ranges {
        [rows] => row_range(x, rows, total),
        _ => Tensor::cat(
            ranges
                .iter()
                .map(|r| x.clone().narrow(0, r.start, r.len()))
                .collect(),
            0,
        ),
    }
}

/// The pass inputs of the sequences `segs`, over the target's own blocks.
fn trunk_seqs<'a>(layout: &'a KvLayout, segs: &[Seg<'a>]) -> Vec<PassSeq<'a>> {
    segs.iter()
        .map(|s| PassSeq {
            start: s.start,
            t: s.t,
            slot: s.slot,
            kv: PassKv::Trunk {
                layout,
                tables: s.tables,
            },
        })
        .collect()
}

impl Weights {
    /// The residual stream `[t, n_embd]` the layers start from, for the
    /// concatenated rows `embeddings`, and the Gemma 4 per-layer inputs
    /// `(ctx + tok) / sqrt(2)`: the projected embeddings merged with the
    /// token-identity rows `per_layer`.
    fn embed_input(
        &self,
        embeddings: Vec<f32>,
        per_layer: Option<Vec<f32>>,
    ) -> (Tensor<2>, Option<Tensor<3>>) {
        let hp = &self.hp;
        let dev = &self.device;
        let eps = hp.eps;
        let t = embeddings.len() / hp.n_embd;
        let x = Tensor::<2>::from_data(TensorData::new(embeddings, [t, hp.n_embd]), dev);
        let ple = match (&self.ple, hp.ple_dim, per_layer) {
            (Some(ple), Some(dim), Some(tokens)) => {
                let tokens =
                    Tensor::<3>::from_data(TensorData::new(tokens, [t, hp.n_layer, dim]), dev);
                let ctx = linear(x.clone(), &ple.proj)
                    .mul_scalar(1.0 / (hp.n_embd as f64).sqrt())
                    .reshape([t, hp.n_layer, dim]);
                let ctx = rms(ctx, Some(&ple.proj_norm), eps);
                Some((ctx + tokens).mul_scalar(FRAC_1_SQRT_2))
            }
            _ => None,
        };
        (x, ple)
    }

    /// The layers over the concatenated rows `embeddings` of the sequences
    /// `segs` describe, then the final norm and the LM head over the rows
    /// the sequences want (the whole model on this one device).
    ///
    /// # Errors
    ///
    /// [`CucaError::Provider`] when a block table lacks a block the pass
    /// reads or writes, a slot has no state or a device readback fails.
    fn forward_layers(
        &self,
        kv: &mut KvLayers,
        layout: &KvLayout,
        slots: &mut [DeltaSlot],
        embeddings: Vec<f32>,
        per_layer: Option<Vec<f32>>,
        segs: &[Seg<'_>],
    ) -> Result<Vec<Forward>, CucaError> {
        let (x, ple) = self.embed_input(embeddings, per_layer);
        self.run_layers(x, ple, &[], kv, layout, slots, segs)
    }

    /// Write the rows the CPU cached for the K/V sources it holds into the
    /// mirror pools of this device ([`Back::mirror`]), through the blocks
    /// each sequence's pass already resolved.
    ///
    /// # Errors
    ///
    /// [`CucaError::Provider`] for rows of a layer without a mirror pool or
    /// of the wrong size, or a sequence without blocks for the layer.
    fn write_mirror(
        &self,
        pass: &Pass,
        kv: &mut KvLayers,
        mirror: &[MirrorRows],
    ) -> Result<(), CucaError> {
        // Everything that can fail is checked before any pool is taken out of
        // `kv`, so that a rejected call leaves every pool in place.
        let mut plan = Vec::with_capacity(mirror.len());
        for m in mirror {
            let spec = self
                .hp
                .layers
                .get(m.layer)
                .ok_or_else(|| err(format!("layer {} does not exist", m.layer)))?;
            let (nkv, d) = (spec.n_kv_head, spec.head_dim);
            if m.seqs.len() != pass.seqs.len() {
                return Err(err(format!(
                    "layer {}: rows for {} sequences in a step of {}",
                    m.layer,
                    m.seqs.len(),
                    pass.seqs.len()
                )));
            }
            if kv.get(m.layer).is_none_or(Option::is_none) {
                return Err(err(format!("layer {} has no mirrored K/V pool", m.layer)));
            }
            let mut writes = Vec::with_capacity(pass.seqs.len());
            for (seq, (k, v)) in pass.seqs.iter().zip(&m.seqs) {
                let want = nkv * seq.t * d;
                if k.len() != want || v.len() != want {
                    return Err(err(format!(
                        "layer {}: {} key and {} value values for {want} expected",
                        m.layer,
                        k.len(),
                        v.len()
                    )));
                }
                writes.push(&seq.visible(m.layer, spec.window)?.1.writes);
            }
            plan.push((m, nkv, d, writes));
        }
        for (m, nkv, d, writes) in plan {
            let Some(mut pool) = kv[m.layer].take() else {
                return Err(err(format!("layer {} has no mirrored K/V pool", m.layer)));
            };
            for ((seq, (k, v)), writes) in pass.seqs.iter().zip(&m.seqs).zip(writes) {
                let upload = |rows: &[f32]| {
                    Tensor::<4>::from_data(
                        TensorData::new(rows.to_vec(), [1, nkv, seq.t, d]),
                        &self.device,
                    )
                };
                pool = pool.write(writes, 0, &upload(k), Some(&upload(v)));
            }
            kv[m.layer] = Some(pool);
        }
        Ok(())
    }

    /// The layers this device holds over the residual stream `x` of the
    /// sequences `segs` describe (`ple` Gemma 4's per-layer inputs), then
    /// the final norm and the LM head over the rows they want.
    ///
    /// # Errors
    ///
    /// As [`Weights::forward_layers`].
    #[expect(
        clippy::too_many_arguments,
        reason = "the residual stream, the mirrored K/V and the pass's state are independent inputs"
    )]
    fn run_layers(
        &self,
        mut x: Tensor<2>,
        ple: Option<Tensor<3>>,
        mirror: &[MirrorRows],
        kv: &mut KvLayers,
        layout: &KvLayout,
        slots: &mut [DeltaSlot],
        segs: &[Seg<'_>],
    ) -> Result<Vec<Forward>, CucaError> {
        let seqs = trunk_seqs(layout, segs);
        let pass = self.pass(ple, &seqs)?;
        self.write_mirror(&pass, kv, mirror)?;
        for il in self.first..self.hp.n_layer {
            x = self.layer(il, x, &pass, kv, slots, None)?;
        }
        self.finish(x, segs)
    }

    /// The final norm and the LM head over the rows of `x` (one row per
    /// token of the step) the sequences `segs` want.
    ///
    /// The final norm runs over every row of a sequence that wants all
    /// logits or every position's hidden row, else over its last row only
    /// (it is row-independent). The selected rows of all sequences share
    /// one norm and one LM head matmul, so the head weight is read once.
    fn finish(&self, x: Tensor<2>, segs: &[Seg<'_>]) -> Result<Vec<Forward>, CucaError> {
        let n_embd = self.hp.n_embd;
        let mut outs: Vec<Forward> = segs.iter().map(|_| Forward::default()).collect();
        let mut ranges: Vec<Range<usize>> = Vec::new();
        // Per sequence: first row among the normed rows, row count, whether
        // those are all of its rows.
        let mut plan = Vec::with_capacity(segs.len());
        let mut at = 0;
        for seg in segs {
            let all_hidden = seg.hidden == Hidden::PostNormAll;
            if seg.want == Want::None && !all_hidden {
                plan.push(None);
                continue;
            }
            let all_rows = all_hidden || seg.want == Want::All;
            let (from, n) = if all_rows {
                (seg.offset, seg.t)
            } else {
                (seg.offset + seg.t - 1, 1)
            };
            push_rows(&mut ranges, from..from + n);
            plan.push(Some((at, n, all_rows)));
            at += n;
        }
        if ranges.is_empty() {
            return Ok(outs);
        }
        let head = self.head_parts()?;
        let normed = rms(take_rows(x, &ranges), Some(&head.norm), self.hp.eps);

        // The logits rows among the normed rows, per sequence: first row
        // among the head's rows and row count.
        let mut logit_ranges: Vec<Range<usize>> = Vec::new();
        let mut spans = Vec::with_capacity(segs.len());
        let mut lat = 0;
        for (seg, p) in segs.iter().zip(&plan) {
            let Some((at, n, all_rows)) = *p else {
                spans.push(None);
                continue;
            };
            let (from, rows) = match seg.want {
                Want::None => {
                    spans.push(None);
                    continue;
                }
                Want::Last if all_rows => (at + n - 1, 1),
                Want::Last | Want::All => (at, n),
            };
            push_rows(&mut logit_ranges, from..from + rows);
            spans.push(Some((lat, rows)));
            lat += rows;
        }
        let wants_hidden = segs
            .iter()
            .zip(&plan)
            .any(|(seg, p)| p.is_some() && seg.hidden != Hidden::None);
        // The head is queued before the hidden-state readback so the device
        // keeps working through it instead of idling at that sync point.
        let logits = if logit_ranges.is_empty() {
            None
        } else {
            Some(self.head(take_rows(normed.clone(), &logit_ranges))?)
        };
        if wants_hidden {
            let flat = read_f32(normed, "hidden states")?;
            if let ([seg], [Some(_)]) = (segs, plan.as_slice()) {
                if seg.hidden != Hidden::None {
                    outs[0].hidden = Some(flat);
                }
            } else {
                for ((out, seg), p) in outs.iter_mut().zip(segs).zip(&plan) {
                    if let (Some((at, n, _)), true) = (*p, seg.hidden != Hidden::None) {
                        out.hidden = Some(flat[at * n_embd..(at + n) * n_embd].to_vec());
                    }
                }
            }
        }
        if let Some(logits) = logits {
            let flat = read_f32(logits, "logits")?;
            if let [Some(_)] = spans.as_slice() {
                outs[0].logits = Some(flat);
            } else {
                let vocab = flat.len() / lat;
                for (out, span) in outs.iter_mut().zip(&spans) {
                    if let Some((at, n)) = *span {
                        out.logits = Some(flat[at * vocab..(at + n) * vocab].to_vec());
                    }
                }
            }
        }
        Ok(outs)
    }

    /// LM head over final-norm output rows: matmul, logit scale, soft-cap.
    ///
    /// # Errors
    ///
    /// [`CucaError::Provider`] on a shard that holds no LM head.
    fn head(&self, normed: Tensor<2>) -> Result<Tensor<2>, CucaError> {
        let hp = &self.hp;
        let mut logits = linear(normed, &self.head_parts()?.lm_head);
        if hp.logit_scale != 1.0 {
            logits = logits.div_scalar(f64::from(hp.logit_scale));
        }
        if hp.logit_mult != 1.0 {
            logits = logits.mul_scalar(f64::from(hp.logit_mult));
        }
        if let Some(cap) = hp.logit_softcap {
            logits = logits.div_scalar(cap).tanh().mul_scalar(cap);
        }
        Ok(logits)
    }

    /// The RoPE rows, visible ranges, masks and block views shared by the
    /// layers of one forward pass over the sequences `seqs` (their rows one
    /// after the other), reading the blocks each names.
    ///
    /// When every sequence is a single query row (a decode step), the
    /// blocks the layers read are stacked into one launch of the decode
    /// kernel ([`Pass::stacked`]).
    ///
    /// # Errors
    ///
    /// [`CucaError::Provider`] when a block table lacks a block the pass
    /// reads or writes.
    fn pass(&self, ple: Option<Tensor<3>>, seqs: &[PassSeq<'_>]) -> Result<Pass, CucaError> {
        let total: usize = seqs.iter().map(|s| s.t).sum();
        // The position of every row, when the sequences' rows are not one
        // contiguous range of the tables.
        let positions = (seqs.len() > 1).then(|| {
            let at: Vec<i32> = seqs
                .iter()
                .flat_map(|s| s.start..s.start + s.t)
                // Positions are below the context length, which fits an i32.
                .map(|p| p as i32)
                .collect();
            Tensor::<1, Int>::from_data(TensorData::new(at, [total]), &self.device)
        });
        let ropes = self
            .ropes
            .iter()
            .map(|table| {
                let [_, half] = table.cos.dims();
                let rows = |m: &Tensor<2>| {
                    let picked = match (&positions, seqs.first()) {
                        (Some(at), _) => m.clone().select(0, at.clone()),
                        (None, Some(only)) => m.clone().narrow(0, only.start, only.t),
                        (None, None) => m.clone(),
                    };
                    picked.reshape([1, 1, total, half])
                };
                RopeRows {
                    cos: rows(&table.cos),
                    sin: rows(&table.sin),
                }
            })
            .collect();

        // The Llama 4 query scaling of each row (`mistral3`), `[1, 1, t, 1]`;
        // none while every position is below its length.
        let q_scale = self.hp.attn_temp.and_then(|temp| {
            let scales: Vec<f32> = seqs
                .iter()
                .flat_map(|s| s.start..s.start + s.t)
                .map(|p| temp.at(p))
                .collect();
            scales.iter().any(|&s| s != 1.0).then(|| {
                Tensor::<4>::from_data(TensorData::new(scales, [1, 1, total, 1]), &self.device)
            })
        });
        let stack = seqs.len() > 1
            && seqs
                .iter()
                .all(|s| s.t == 1 && matches!(s.kv, PassKv::Trunk { .. }));
        let mut passes = Vec::with_capacity(seqs.len());
        let mut off = 0;
        for seq in seqs {
            passes.push(SeqPass {
                off,
                t: seq.t,
                slot: seq.slot,
                masks: self.masks(seq, stack)?,
            });
            off += seq.t;
        }
        let stacked = if stack {
            self.stack_decode(&passes)?
        } else {
            Vec::new()
        };
        Ok(Pass {
            t: total,
            ple,
            ropes,
            q_scale,
            seqs: passes,
            stacked,
        })
    }

    /// The visible range, mask and block view of every distinct layer
    /// window for one sequence of a pass; `stack` keeps a single row's
    /// blocks as host ids for [`Weights::stack_decode`].
    fn masks(&self, seq: &PassSeq<'_>, stack: bool) -> Result<Vec<MaskInput>, CucaError> {
        let (start, t) = (seq.start, seq.t);
        let mut masks: Vec<MaskInput> = Vec::new();
        for spec in &self.hp.layers {
            if masks.iter().any(|m| m.window == spec.window) {
                continue;
            }
            // A drafter's query is not cached: it sees the `len` cached
            // positions, the sliding layers the window before it.
            let (lo, hi) = match &seq.kv {
                PassKv::Trunk { .. } => (
                    spec.window.map_or(0, |w| start.saturating_sub(w - 1)),
                    start + t,
                ),
                PassKv::Drafter(ext) => (
                    spec.window.map_or(0, |w| {
                        start.saturating_sub(w - 1).min(ext.len.saturating_sub(1))
                    }),
                    ext.len,
                ),
            };
            let mask = (t > 1).then(|| {
                let values = attention_mask(start, t, lo, hi, spec.window);
                Tensor::from_data(TensorData::new(values, [1, 1, t, hi - lo]), &self.device)
            });
            let blocks = match &seq.kv {
                PassKv::Trunk { layout, tables } => {
                    match layout.groups.iter().position(|g| g.window == spec.window) {
                        Some(g) => {
                            let table = tables.get(g).ok_or_else(|| {
                                err(format!("the sequence has no block table for group {g}"))
                            })?;
                            let rows = BlockRows {
                                table,
                                n_blocks: layout.blocks[g],
                                start,
                                t,
                                lo,
                                hi,
                                write: true,
                                stack,
                            };
                            Some(rows.view(&self.device)?)
                        }
                        None => None,
                    }
                }
                PassKv::Drafter(ext) => {
                    let group = if spec.window.is_some() {
                        &ext.sliding
                    } else {
                        &ext.full
                    };
                    let rows = BlockRows {
                        table: &group.table,
                        n_blocks: group.k.dims()[0],
                        start,
                        t,
                        lo,
                        hi,
                        write: false,
                        stack: false,
                    };
                    Some(rows.view(&self.device)?)
                }
            };
            masks.push(MaskInput {
                window: spec.window,
                mask,
                blocks,
            });
        }
        Ok(masks)
    }

    /// The blocks of a decode step's sequences, one row each, as the tables
    /// and ranges of one launch of the decode kernel per window: a table
    /// row per sequence, padded with block 0 (never read: a range stays
    /// inside the sequence's own blocks).
    fn stack_decode(&self, seqs: &[SeqPass]) -> Result<Vec<StackedDecode>, CucaError> {
        let Some(first) = seqs.first() else {
            return Ok(Vec::new());
        };
        let mut stacked = Vec::new();
        for (mi, m) in first.masks.iter().enumerate() {
            if m.blocks.is_none() {
                continue;
            }
            let mut rows = Vec::with_capacity(seqs.len());
            for seq in seqs {
                match seq
                    .masks
                    .get(mi)
                    .and_then(|m| m.blocks.as_ref())
                    .map(|b| &b.read)
                {
                    Some(BlockRead::Rows { ids, range }) => rows.push((ids, range)),
                    _ => return Err(err("a decode step's sequences disagree on their blocks")),
                }
            }
            let width = rows.iter().map(|(ids, _)| ids.len()).max().unwrap_or(0);
            let mut table: Vec<i32> = Vec::with_capacity(rows.len() * width);
            let mut range: Vec<i32> = Vec::with_capacity(rows.len() * 2);
            for (i, (ids, r)) in rows.iter().enumerate() {
                table.extend_from_slice(ids);
                table.resize((i + 1) * width, 0);
                range.extend_from_slice(*r);
            }
            let n = rows.len();
            stacked.push(StackedDecode {
                window: m.window,
                table: Tensor::from_data(TensorData::new(table, [n, width]), &self.device),
                range: Tensor::from_data(TensorData::new(range, [n * 2]), &self.device),
            });
        }
        Ok(stacked)
    }

    /// One transformer block over the `pass.t` rows of the step's
    /// sequences (the whole block on this device). The projections, norms
    /// and FFN run over all rows (each weight is read once); the K/V writes,
    /// the attention and the recurrent layers run per sequence.
    fn layer(
        &self,
        il: usize,
        x: Tensor<2>,
        pass: &Pass,
        kv: &mut KvLayers,
        slots: &mut [DeltaSlot],
        ext: Option<&GpuExtKv>,
    ) -> Result<Tensor<2>, CucaError> {
        let attn = self.mixer(il, &x, pass, kv, slots, ext)?;
        let x = self.attn_tail(il, x, attn);
        let f = self.ffn_half(il, &x)?;
        Ok(self.ffn_tail(il, x, f, pass))
    }

    /// The first half of layer `il` over the residual stream `x`: the
    /// pre-attention norm, then the attention (or recurrent) mixer, up to
    /// and including the output projection's matmul and no further. On a
    /// split model it is this shard's heads' share of the projection, a
    /// partial sum of the layer's output.
    fn mixer(
        &self,
        il: usize,
        x: &Tensor<2>,
        pass: &Pass,
        kv: &mut KvLayers,
        slots: &mut [DeltaSlot],
        ext: Option<&GpuExtKv>,
    ) -> Result<Tensor<2>, CucaError> {
        let lw = &self.layers[il - self.first];
        let h = match &lw.attn_norm {
            Some(w) => rms(x.clone(), Some(w), self.hp.eps),
            None => x.clone(),
        };
        match &lw.mixer {
            Mixer::Mla(mw) => self.mla(il, h, mw, pass, kv),
            Mixer::Delta(dw) => self.delta(il, h, dw, pass, slots),
            Mixer::Attention(aw) => self.attention(il, h, aw, pass, kv, ext),
        }
    }

    /// The gated delta-net mixer of layer `il` over the normed rows `h`: each
    /// sequence's convolution and recurrence run from its slot's state, which
    /// then moves to the end of its rows.
    fn delta(
        &self,
        il: usize,
        h: Tensor<2>,
        dw: &DeltaWeights,
        pass: &Pass,
        slots: &mut [DeltaSlot],
    ) -> Result<Tensor<2>, CucaError> {
        let mut states = Vec::with_capacity(pass.seqs.len());
        for seq in &pass.seqs {
            let state = slots
                .get(seq.slot)
                .and_then(|s| s.layers.get(il))
                .and_then(Option::as_ref)
                .ok_or_else(|| {
                    err(format!(
                        "layer {il} has no recurrent state in slot {}",
                        seq.slot
                    ))
                })?;
            states.push((seq.off..seq.off + seq.t, state));
        }
        let (y, next) = dw.forward(h, &states, self.hp.eps);
        for (seq, state) in pass.seqs.iter().zip(next) {
            if let Some(s) = slots.get_mut(seq.slot).and_then(|s| s.layers.get_mut(il)) {
                *s = Some(state);
            }
        }
        Ok(y)
    }

    /// The attention mixer of layer `il` over the normed rows `h`: the
    /// projections, norms and RoPE, the K/V writes of a layer that owns K/V,
    /// the paged attention over the layer's source pools ([`attend_pass`]),
    /// the output gate and the output projection.
    fn attention(
        &self,
        il: usize,
        h: Tensor<2>,
        aw: &AttnWeights,
        pass: &Pass,
        kv: &mut KvLayers,
        ext: Option<&GpuExtKv>,
    ) -> Result<Tensor<2>, CucaError> {
        let hp = &self.hp;
        let spec = &hp.layers[il];
        let eps = hp.eps;
        let (nh, nkv, d) = (spec.n_head, spec.n_kv_head, spec.head_dim);
        let t = pass.t;
        let rotate = |m: Tensor<4>| match spec.rope {
            Some(r) => rope(m, &pass.ropes[r], &hp.ropes[r]),
            None => m,
        };

        // A gate projection of its own reads the same normed input.
        let own_gate = aw.gate.as_ref().map(|g| g.forward(h.clone()));
        let (q, kv_proj) = aw.project(h, nh, nkv, d);
        let full_qk = hp.behavior.qk_norm_full;
        let q = match &aw.q_norm {
            Some(w) if full_qk => rms(q, Some(w), eps),
            _ => q,
        };
        // A gated attention packs `[query | gate]` per head into the query
        // projection (each head's `2 * d` values contiguous).
        let (q, gate) = if hp.behavior.gated_attention {
            let packed = q.reshape([t, nh, 2 * d]);
            (
                packed.clone().narrow(2, 0, d).reshape([1, t, nh, d]),
                Some(packed.narrow(2, d, d).reshape([t, nh * d])),
            )
        } else {
            (q.reshape([1, t, nh, d]), None)
        };
        let gate = own_gate.or(gate);
        let q = match &aw.q_norm {
            Some(w) if !full_qk => rms(q, Some(w), eps),
            _ => q,
        };
        let q = rotate(q.swap_dims(1, 2));
        let q = match (&pass.q_scale, spec.rope) {
            (Some(s), Some(_)) => q * s.clone(),
            _ => q,
        };
        // Every sequence must have its window's blocks, also on the layers
        // that only read.
        for seq in &pass.seqs {
            seq.visible(il, spec.window)?;
        }

        if let Some((k, v)) = kv_proj {
            let k = match &aw.k_norm {
                Some(w) if full_qk => rms(k, Some(w), eps),
                _ => k,
            };
            let k = k.reshape([1, t, nkv, d]);
            let k = match &aw.k_norm {
                Some(w) if !full_qk => rms(k, Some(w), eps),
                _ => k,
            };
            let k = rotate(k.swap_dims(1, 2));
            let v = v.reshape([1, t, nkv, d]);
            let v = if aw.v_norm { rms(v, None, eps) } else { v };
            let v = v.swap_dims(1, 2);
            let mut pool = take_pool(kv, il)?;
            for seq in &pass.seqs {
                let (_, blocks) = seq.visible(il, spec.window)?;
                pool = pool.write(&blocks.writes, seq.off, &k, Some(&v));
            }
            kv[il] = Some(pool);
        }

        let src = spec.kv_source;
        let (pool_k, pool_v) = if src == usize::MAX {
            // A drafter layer reads the target's pools.
            let ext = ext.ok_or_else(|| {
                err(format!(
                    "layer {il}: a drafter layer needs the target's K/V"
                ))
            })?;
            let group = if spec.window.is_some() {
                &ext.sliding
            } else {
                &ext.full
            };
            (&group.k, &group.v)
        } else {
            let pool = kv
                .get(src)
                .and_then(Option::as_ref)
                .ok_or_else(|| err(format!("layer {src} owns no K/V cache")))?;
            let v = pool.v.as_ref().ok_or_else(|| {
                err(format!(
                    "layer {src} has a latent K/V cache, which holds no value pool"
                ))
            })?;
            (&pool.k, v)
        };

        let call = PagedCall {
            scale: spec.scale,
            v_width: None,
        };
        let attn =
            attend_pass(pass, il, spec.window, &q, pool_k, pool_v, &call)?.reshape([t, nh * d]);
        let attn = match gate {
            Some(gate) => attn * activation::sigmoid(gate),
            None => attn,
        };
        // The output projection's product only: its bias is added after the
        // shards' products are summed ([`Weights::attn_tail`]).
        Ok(linear(attn, &aw.wo.w))
    }

    /// The latent attention of layer `il` (Mistral Small 4) over the normed
    /// rows `h [t, n_embd]` of the pass, up to and including the output
    /// projection, as the CPU `mla_in` computes it.
    ///
    /// The query is low rank (`q_b(norm(q_a(h)))`, `n_head` heads of `nope +
    /// rot` values). `kv_a(h)` yields a latent `c` (`kv_lora` values, normed)
    /// and a rotary key `k_rot` (`rot` values, rotated), both shared by every
    /// head, and the layer's cache is the single row `[c | k_rot]` per
    /// position in the paged pool `[n_blocks, 1, 16, kv_lora + rot]` (written
    /// block by block as a normal layer's keys are). The key up-projection
    /// is absorbed into the query, `q_abs = q_nope . k_b` (one batched
    /// product), so the score of head `h` is `[q_abs | q_rot] . [c | k_rot]`
    /// against the shared row, and the attended latent, the softmax-weighted
    /// sum of the rows' first `kv_lora` columns, goes through `v_b` per head
    /// afterwards: no per-head key or value exists. A single query row
    /// reads the pool in place ([`paged_kernel::decode`] with the pool as
    /// both keys and values, values being the row's prefix); several rows
    /// gather the visible rows and use [`attend`]. Each query row is scaled
    /// by the Llama 4 factor of its position ([`Pass::q_scale`]).
    fn mla(
        &self,
        il: usize,
        h: Tensor<2>,
        mw: &MlaWeights,
        pass: &Pass,
        kv: &mut KvLayers,
    ) -> Result<Tensor<2>, CucaError> {
        let hp = &self.hp;
        let spec = &hp.layers[il];
        let mla = spec
            .mla
            .ok_or_else(|| err(format!("layer {il} is not a latent attention layer")))?;
        let rope_at = spec.rope.ok_or_else(|| {
            err(format!(
                "layer {il}: latent attention needs rotary embeddings"
            ))
        })?;
        let eps = hp.eps;
        let t = pass.t;
        let (nh, nope, rot, kvl) = (spec.n_head, mla.nope, mla.rot, mla.kv_lora);
        let rotate = |m: Tensor<4>| rope(m, &pass.ropes[rope_at], &hp.ropes[rope_at]);

        // The query rows `[nope | rot]` per head: the nope part absorbed
        // through `k_b`, the rot part rotated, scaled by position.
        let cq = rms(linear(h.clone(), &mw.q_a), Some(&mw.q_a_norm), eps);
        let q = linear(cq, &mw.q_b).reshape([t, nh, mla.q_head()]);
        let q_nope = q.clone().narrow(2, 0, nope).swap_dims(0, 1);
        let q_rot = rotate(
            q.narrow(2, nope, rot)
                .reshape([1, t, nh, rot])
                .swap_dims(1, 2),
        );
        let q_abs = q_nope.matmul(mw.k_b.clone()).reshape([1, nh, t, kvl]);
        let q_full = Tensor::cat(vec![q_abs, q_rot], 3);
        let q_full = match &pass.q_scale {
            Some(s) => q_full * s.clone(),
            None => q_full,
        };

        // The cached row `[c | k_rot]` of every new position.
        let latent = linear(h, &mw.kv_a);
        let c = rms(latent.clone().narrow(1, 0, kvl), Some(&mw.kv_a_norm), eps);
        let k_rot = rotate(latent.narrow(1, kvl, rot).reshape([1, 1, t, rot]));
        let row = Tensor::cat(vec![c.reshape([1, 1, t, kvl]), k_rot], 3);
        let mut pool = take_pool(kv, il)?;
        for seq in &pass.seqs {
            let (_, blocks) = seq.visible(il, spec.window)?;
            pool = pool.write(&blocks.writes, seq.off, &row, None);
        }
        kv[il] = Some(pool);
        let pool_k = kv
            .get(il)
            .and_then(Option::as_ref)
            .map(|p| &p.k)
            .ok_or_else(|| err(format!("layer {il} owns no K/V cache")))?;

        // The attended latent `[n_head, t, kv_lora]`.
        let call = PagedCall {
            scale: spec.scale,
            v_width: Some(kvl),
        };
        let attended =
            attend_pass(pass, il, spec.window, &q_full, pool_k, pool_k, &call)?.swap_dims(0, 1);
        let out = attended
            .matmul(mw.v_b.clone())
            .swap_dims(0, 1)
            .reshape([t, nh * mla.v_head]);
        Ok(linear(out, &mw.wo.w))
    }

    /// `branch` scaled by the residual scale (the branch itself when 1).
    fn scaled(&self, branch: Tensor<2>) -> Tensor<2> {
        if self.hp.residual_scale == 1.0 {
            branch
        } else {
            branch.mul_scalar(f64::from(self.hp.residual_scale))
        }
    }

    /// What follows the mixer in layer `il`: the output projection's bias,
    /// the post-attention norm and the residual add. `attn` is
    /// [`Weights::mixer`]'s output (on a split model, summed over the
    /// shards: the bias belongs to the sum, so it is added once, here).
    fn attn_tail(&self, il: usize, x: Tensor<2>, mut attn: Tensor<2>) -> Tensor<2> {
        let lw = &self.layers[il - self.first];
        if let Mixer::Attention(aw) = &lw.mixer
            && let Some(b) = &aw.wo.bias
        {
            attn = attn + b.clone().unsqueeze::<2>();
        }
        if let Some(w) = &lw.post_attn_norm {
            attn = rms(attn, Some(w), self.hp.post_eps);
        }
        x + self.scaled(attn)
    }

    /// The FFN of layer `il` over the residual stream `x`: the pre-FFN norm
    /// and the dense FFN or MoE (on a split model, this shard's slice of it:
    /// a partial sum of the layer's output).
    ///
    /// # Errors
    ///
    /// [`CucaError::Provider`] when an expert tier fails to load an expert.
    fn ffn_half(&self, il: usize, x: &Tensor<2>) -> Result<Tensor<2>, CucaError> {
        let lw = &self.layers[il - self.first];
        let f = match &lw.ffn_norm {
            Some(w) => rms(x.clone(), Some(w), self.hp.eps),
            None => x.clone(),
        };
        lw.ffn.forward(f, self.hp.act(), &self.device)
    }

    /// What follows the FFN in layer `il`: the post-FFN norm, the residual
    /// add of `f` ([`Weights::ffn_half`]'s output, summed over the shards of
    /// a split model), the Gemma 4 per-layer-embedding block and the layer's
    /// output scale.
    fn ffn_tail(&self, il: usize, x: Tensor<2>, mut f: Tensor<2>, pass: &Pass) -> Tensor<2> {
        let hp = &self.hp;
        let lw = &self.layers[il - self.first];
        let eps = hp.eps;
        if let Some(w) = &lw.post_ffw_norm {
            f = rms(f, Some(w), hp.post_eps);
        }
        let mut x = x + self.scaled(f);

        if let (Some(layer), Some(ple), Some(dim)) = (&lw.ple, &pass.ple, hp.ple_dim) {
            let slice = ple.clone().narrow(1, il, 1).reshape([pass.t, dim]);
            let g = gelu(linear(x.clone(), &layer.inp_gate)) * slice;
            x = x + rms(linear(g, &layer.proj), Some(&layer.post_norm), eps);
        }
        match lw.out_scale {
            Some(s) => x.mul_scalar(s),
            None => x,
        }
    }
}

impl AttnWeights {
    /// Query `[t, nh * d]` and, on K/V-owning layers, key and value
    /// `[t, nkv * d]` projections of `h`.
    fn project(
        &self,
        h: Tensor<2>,
        nh: usize,
        nkv: usize,
        d: usize,
    ) -> (Tensor<2>, Option<KvProj>) {
        match &self.qkv {
            QkvWeights::Fused(w) => {
                let y = w.forward(h);
                let (q, k) = (nh * d, nkv * d);
                (
                    y.clone().narrow(1, 0, q),
                    Some((y.clone().narrow(1, q, k), y.narrow(1, q + k, k))),
                )
            }
            QkvWeights::Split { q, kv } => {
                let query = q.forward(h.clone());
                let kv = kv
                    .as_ref()
                    .map(|(k, v)| (k.forward(h.clone()), v.forward(h)));
                (query, kv)
            }
        }
    }
}

impl FfnWeights {
    /// The FFN over `f [t, n_embd]`.
    fn forward(&self, f: Tensor<2>, act: Act, device: &Device) -> Result<Tensor<2>, CucaError> {
        match self {
            FfnWeights::Dense { gate_up, down } => {
                let (gate, up) = match gate_up {
                    GateUp::Split { gate, up } => (linear(f.clone(), gate), linear(f, up)),
                    GateUp::Fused { up, ff } => {
                        let y = linear(f, up);
                        (y.clone().narrow(1, 0, *ff), y.narrow(1, *ff, *ff))
                    }
                };
                Ok(linear(activate(gate, act) * up, down))
            }
            FfnWeights::Moe {
                spec,
                router,
                experts,
                shared,
            } => {
                let [t, _] = f.dims();
                let act_code = match act {
                    Act::Silu => 0,
                    Act::Gelu => 1,
                };
                // Routing and the experts run in the MoE kernels (see
                // [`mk`]); a resident bank never reads the router output
                // back.
                let layout = *experts.layout();
                let call = |pairs: usize, use_list: bool| MoeCall {
                    tokens: t,
                    n_used: spec.n_used,
                    n_expert: spec.n_expert,
                    pairs,
                    use_list,
                    act: act_code,
                    scale: spec.scale,
                    slot: layout,
                };
                let logits = linear(f.clone(), router);
                let route = mk::route(logits, &call(t * spec.n_used, false), spec.norm_weights);
                // The shared expert, scaled per token by
                // `sigmoid(gate_inp . x)`: the dot product is an
                // elementwise product and a row sum, not a matmul.
                let shared_out = |sh: &GpuShared, f: Tensor<2>| {
                    let hidden =
                        activate(linear(f.clone(), &sh.gate), act) * linear(f.clone(), &sh.up);
                    let y = linear(hidden, &sh.down);
                    match &sh.gate_inp {
                        Some(gate_inp) => {
                            y * activation::sigmoid((f * gate_inp.clone()).sum_dim(1))
                        }
                        // Ungated: every token takes the whole output.
                        None => y,
                    }
                };
                experts.run(f.clone(), route, &call, device, || {
                    shared.as_ref().map(|sh| shared_out(sh, f))
                })
            }
        }
    }
}

/// `x` scaled to unit length per head (the last dimension), with the
/// reference's epsilon inside the root.
fn l2_normalise(x: Tensor<3>) -> Tensor<3> {
    let inv = x
        .clone()
        .square()
        .sum_dim(2)
        .add_scalar(1e-6)
        .sqrt()
        .recip();
    x * inv
}

/// Rows `rows` of the `total` rows of `m` (the tensor itself when they are
/// all of them: no operation to queue).
fn row_range<const D: usize>(m: Tensor<D>, rows: &Range<usize>, total: usize) -> Tensor<D> {
    if rows.len() == total {
        m
    } else {
        m.narrow(0, rows.start, rows.len())
    }
}

/// Row `ti` of the `t` rows of `m` (the tensor itself for a single row: no
/// operation to queue).
fn token_row<const D: usize>(m: Tensor<D>, ti: usize, t: usize) -> Tensor<D> {
    if t == 1 { m } else { m.narrow(0, ti, 1) }
}

impl DeltaWeights {
    /// The layer's output `[T, n_embd]` for the normed rows `h [T, n_embd]`
    /// of the step's sequences, and the state of each sequence after its
    /// rows. `seqs` holds, per sequence, its row range in `h` and its state
    /// before them (see [`super::delta`]: the same recurrence, run per token
    /// on the device). The projections, the gate and the output matmul run
    /// over all rows at once (each weight is read once per step); the
    /// convolution and the recurrence run per sequence.
    ///
    /// Decode is host-bound (every queued operation costs the host time),
    /// so a single token runs few, wide operations: the convolution
    /// is one product and sum over the window, the key and query heads are
    /// normalised together, and the value heads that share a key head
    /// broadcast it instead of repeating it.
    fn forward(
        &self,
        h: Tensor<2>,
        seqs: &[(Range<usize>, &DeltaState)],
        eps: f64,
    ) -> (Tensor<2>, Vec<DeltaState>) {
        let d = &self.spec;
        let [total, _] = h.dims();
        let (nv, dv, vd) = (d.n_v_heads, d.head_v_dim, d.value_dim());

        let mixed = linear(h.clone(), &self.qkv);
        let z = linear(h.clone(), &self.gate);
        let ba = linear(h, &self.beta_alpha);
        let beta = activation::sigmoid(ba.clone().narrow(1, 0, nv));
        // exp(ssm_a * softplus(alpha + dt_bias)), softplus linear above 20.
        let pre = ba.narrow(1, nv, nv) + self.dt_bias.clone();
        let soft = pre.clone().exp().log1p();
        let soft = soft.mask_where(pre.clone().greater_elem(20.0), pre);
        let decay = (soft * self.a.clone()).exp();

        let mut outs = Vec::with_capacity(seqs.len());
        let mut states = Vec::with_capacity(seqs.len());
        for (rows, state) in seqs {
            let (o, next) = self.scan(
                row_range(mixed.clone(), rows, total),
                row_range(beta.clone(), rows, total),
                row_range(decay.clone(), rows, total),
                state,
            );
            outs.push(o);
            states.push(next);
        }
        let o = if outs.len() == 1 {
            outs.swap_remove(0)
        } else {
            Tensor::cat(outs, 0)
        };

        // Gated RMSNorm per head: rms(o) * weight * silu(z).
        let z = z.reshape([total, nv, dv]);
        let y = rms(o, Some(&self.norm), eps) * activation::silu(z);
        let out = linear(y.reshape([total, vd]), &self.out);
        (out, states)
    }

    /// The convolution and the delta rule over the `t` rows of one sequence:
    /// `mixed [t, conv_dim]` (the projected `[q | k | v]`), `beta` and
    /// `decay` `[t, n_v_heads]`, from the sequence's `state`. Returns the
    /// outputs `[t, n_v_heads, head_v_dim]` and the state after the rows.
    fn scan(
        &self,
        mixed: Tensor<2>,
        beta: Tensor<2>,
        decay: Tensor<2>,
        state: &DeltaState,
    ) -> (Tensor<3>, DeltaState) {
        let d = &self.spec;
        let [t, _] = mixed.dims();
        let device = mixed.device();
        let (nk, nv, dk, dv) = (d.n_k_heads, d.n_v_heads, d.head_k_dim, d.head_v_dim);
        let (kd, vd, taps) = (d.key_dim(), d.value_dim(), d.conv_kernel);
        let rep = nv / nk;

        // Causal depthwise convolution over `history ++ mixed`, then silu.
        let padded = Tensor::cat(vec![state.conv.clone(), mixed], 0);
        let conv = if t == 1 {
            // The window is the whole padded history.
            (padded.clone() * self.conv.clone()).sum_dim(0)
        } else {
            let tap = |i: usize| padded.clone().narrow(0, i, t) * self.conv.clone().narrow(0, i, 1);
            let mut conv = tap(0);
            for i in 1..taps {
                conv = conv + tap(i);
            }
            conv
        };
        let x = activation::silu(conv);
        let next_conv = padded.narrow(0, t, taps - 1);

        // Queries and keys are adjacent in `x`: one normalisation of `2 * nk`
        // heads.
        let qk = l2_normalise(x.clone().narrow(1, 0, 2 * kd).reshape([t, 2 * nk, dk]));
        let q = qk
            .clone()
            .narrow(1, 0, nk)
            .mul_scalar(1.0 / (dk as f64).sqrt());
        let k = qk.narrow(1, nk, nk);
        let v = x.narrow(1, 2 * kd, vd).reshape([t, nv, dv]);

        // The delta rule, one token at a time: decay the state, read what
        // it predicts for `k`, write the correction, read the output. The
        // state is `[rep, nk, dk, dv]`; value head `r * nk + j` reads key
        // head `j`, which broadcasts over the leading axis.
        let mut s = state.s.clone();
        let mut rows = Vec::with_capacity(t);
        for ti in 0..t {
            let kt = token_row(k.clone(), ti, t).reshape([1, nk, dk, 1]);
            let qt = token_row(q.clone(), ti, t).reshape([1, nk, dk, 1]);
            let vt = token_row(v.clone(), ti, t).reshape([rep, nk, 1, dv]);
            let bt = token_row(beta.clone(), ti, t).reshape([rep, nk, 1, 1]);
            let gt = token_row(decay.clone(), ti, t).reshape([rep, nk, 1, 1]);
            s = s * gt;
            let kv_mem = (s.clone() * kt.clone()).sum_dim(2);
            let delta = (vt - kv_mem) * bt;
            s = s + kt * delta;
            rows.push((s.clone() * qt).sum_dim(2).reshape([1, nv, dv]));
            // Each token queues a few state-sized temporaries; without a
            // periodic drain a long prefill chunk holds hundreds of MB of
            // them at once. Best effort: a failed wait surfaces at the next
            // readback.
            if (ti + 1) % SCAN_DRAIN == 0 && ti + 1 < t {
                let _ = device.sync();
            }
        }
        let o = if t == 1 {
            rows.swap_remove(0)
        } else {
            Tensor::cat(rows, 0)
        };
        (o, DeltaState { conv: next_conv, s })
    }
}

/// Read a device tensor back to the host as f32 values, row-major.
pub(super) fn read_f32<const D: usize>(
    tensor: Tensor<D>,
    what: &str,
) -> Result<Vec<f32>, CucaError> {
    let data = tensor
        .try_into_data()
        .map_err(|e| err(format!("reading {what} from the device: {e}")))?;
    data.try_into_vec::<f32>()
        .map_err(|e| err(format!("reading {what} from the device: {e:?}")))
}

/// The widest GPU subgroup ("plane") burn's matmul kernels may be built for:
/// wgpu reports up to 64 lanes on AMD, 32 elsewhere.
const MAX_PLANE: usize = 64;

/// Whether burn's `vecmat` kernel can launch `x · wᵀ` for a single row `x`
/// and the f32 `[out, in]` weight part `w`: it needs `in` to be a multiple of
/// the plane width and `out` of the plane width, the widest plane
/// ([`MAX_PLANE`]) assumed.
fn vecmat_fits(w: &Tensor<2>) -> bool {
    let [out, inn] = w.dims();
    out.is_multiple_of(MAX_PLANE) && inn.is_multiple_of(MAX_PLANE)
}

/// Input rows up to which a quantized weight runs the matrix-vector kernel
/// ([`mk::dense_gemv`]): a decode step, the rows of a batched decode step or
/// of a verification call, the tail of a prompt. More rows (a 256-row
/// prefill chunk) expand the weight to f32 in slices and use burn's matmul.
const GEMV_ROWS: usize = 64;

/// f32 bytes of one dequantized slice of a quantized weight in a prefill.
const DEQUANT_SLICE_BYTES: usize = 64 << 20;

/// `x [t, in]` times the `[out, in]` weight `w`, transposed: `[t, out]`.
/// A weight in several row chunks runs one product per chunk and
/// concatenates the outputs.
///
/// - A quantized chunk (Q4_0, Q4_1, Q8_0 in the layout of
///   [`mk::DenseLayout`]) with at most [`GEMV_ROWS`] input rows runs the
///   matrix-vector kernel of this crate (f32 accumulation straight from the
///   blocks, one thread per row and token). More rows dequantize slices of
///   the weight to f32 ([`mk::dense_dequant`]) and use burn's float matmul.
///   burn's own quantized matmul is not used (see the module docs).
/// - An f32 chunk follows burn 0.22's rules on Vulkan: `t > 1` runs
///   `(w · xᵀ)ᵀ`; `t == 1` runs `x · wᵀ` ([`vecmat_fits`]) or, for other
///   shapes, a product of the token duplicated into two rows (the `vecmat`
///   autotune entry is shared by every shape of its power-of-two bucket).
fn linear(x: Tensor<2>, w: &Weight) -> Tensor<2> {
    match w.parts.as_slice() {
        [part] => linear_part(x, part),
        parts => Tensor::cat(
            parts
                .iter()
                .map(|part| linear_part(x.clone(), part))
                .collect(),
            1,
        ),
    }
}

/// [`linear`] for one chunk of a weight.
fn linear_part(x: Tensor<2>, part: &Part) -> Tensor<2> {
    let t = x.dims()[0];
    match part {
        Part::Float(w) => {
            if t > 1 {
                w.clone().matmul(x.transpose()).transpose()
            } else if vecmat_fits(w) {
                x.matmul(w.clone().transpose())
            } else {
                let pair = Tensor::cat(vec![x.clone(), x], 0);
                w.clone()
                    .matmul(pair.transpose())
                    .transpose()
                    .narrow(0, 0, 1)
            }
        }
        Part::Packed(p) => {
            let layout = p.layout;
            let call = |row0, count| mk::DenseCall {
                layout,
                tokens: t,
                row0,
                count,
            };
            if t <= GEMV_ROWS {
                return mk::dense_gemv(x, p.pool.clone(), &call(0, 0));
            }
            let like = Tensor::<1>::zeros([1], &x.device());
            let slice = (DEQUANT_SLICE_BYTES / (layout.cols * 4)).clamp(1, layout.rows);
            let mut outs = Vec::with_capacity(layout.rows.div_ceil(slice));
            for row0 in (0..layout.rows).step_by(slice) {
                let count = slice.min(layout.rows - row0);
                let w = mk::dense_dequant(like.clone(), p.pool.clone(), &call(row0, count));
                outs.push(w.matmul(x.clone().transpose()).transpose());
            }
            if outs.len() == 1 {
                outs.swap_remove(0)
            } else {
                Tensor::cat(outs, 1)
            }
        }
    }
}

/// RMSNorm over the last dimension, multiplied by `weight` (no `1 + w`);
/// `None` is the weightless norm.
fn rms<const D: usize>(x: Tensor<D>, weight: Option<&Tensor<1>>, eps: f64) -> Tensor<D> {
    let inv = x
        .clone()
        .square()
        .mean_dim(D - 1)
        .add_scalar(eps)
        .sqrt()
        .recip();
    let y = x * inv;
    match weight {
        Some(w) => y * w.clone().unsqueeze::<D>(),
        None => y,
    }
}

/// The gate activation.
fn activate<const D: usize>(x: Tensor<D>, act: Act) -> Tensor<D> {
    match act {
        Act::Silu => activation::silu(x),
        Act::Gelu => gelu(x),
    }
}

/// GELU, tanh approximation (candle's `Tensor::gelu`). The cube is written
/// as `x * x * x`: a GPU `pow` with a negative base is undefined.
fn gelu<const D: usize>(x: Tensor<D>) -> Tensor<D> {
    const SQRT_2_OVER_PI: f64 = 0.797_884_560_802_865_4;
    let cube = x.clone() * x.clone() * x.clone();
    let inner = (x.clone() + cube.mul_scalar(0.044_715)).mul_scalar(SQRT_2_OVER_PI);
    (x * inner.tanh().add_scalar(1.0)).mul_scalar(0.5)
}

/// Rotate the first `spec.n_dims` values of every head of `x`
/// `[1, heads, t, d]` under `spec.kind`; the remaining `d - n_dims` values
/// pass through. `rows` hold `n_dims / 2` cos/sin entries per position.
///
/// - `NEOX` pairs element `i` with `i + n_dims / 2`.
/// - `NORM` pairs `2i` with `2i + 1`: the head is viewed as `[.., n_dims / 2,
///   2]` and the even and odd columns are rotated and concatenated back.
fn rope(x: Tensor<4>, rows: &RopeRows, spec: &RopeSpec) -> Tensor<4> {
    let [_, heads, t, d] = x.dims();
    let n = spec.n_dims;
    let half = n / 2;
    let (head, rest) = if n == d {
        (x, None)
    } else {
        (x.clone().narrow(3, 0, n), Some(x.narrow(3, n, d - n)))
    };
    let rotated = match spec.kind {
        RopeKind::Neox => {
            let x1 = head.clone().narrow(3, 0, half);
            let x2 = head.narrow(3, half, half);
            Tensor::cat(
                vec![
                    x1.clone() * rows.cos.clone() - x2.clone() * rows.sin.clone(),
                    x1 * rows.sin.clone() + x2 * rows.cos.clone(),
                ],
                3,
            )
        }
        RopeKind::Norm => {
            let pairs = head.reshape([1, heads, t, half, 2]);
            let even = pairs.clone().narrow(4, 0, 1);
            let odd = pairs.narrow(4, 1, 1);
            let cos = rows.cos.clone().reshape([1, 1, t, half, 1]);
            let sin = rows.sin.clone().reshape([1, 1, t, half, 1]);
            Tensor::cat(
                vec![
                    even.clone() * cos.clone() - odd.clone() * sin.clone(),
                    even * sin + odd * cos,
                ],
                4,
            )
            .reshape([1, heads, t, n])
        }
    };
    match rest {
        Some(rest) => Tensor::cat(vec![rotated, rest], 3),
        None => rotated,
    }
}

/// Rows `skip..skip + len` of the blocks `idx` of `pool` `[n_blocks, n_kv,
/// BLOCK_SIZE, d]`, laid end to end as one `[1, n_kv, len, d]` tensor.
fn gather_view(pool: &Tensor<4>, idx: &Tensor<1, Int>, skip: usize, len: usize) -> Tensor<4> {
    let [n_blocks] = idx.dims();
    let [_, n_kv, _, d] = pool.dims();
    pool.clone()
        .select(0, idx.clone())
        .swap_dims(0, 1)
        .reshape([1, n_kv, n_blocks * BLOCK_SIZE, d])
        .narrow(2, skip, len)
}

/// Attention of the pass's query rows `q [1, n_head, t, key_width]` over the
/// K/V blocks of the layers with window `window`: `[t, n_head, value_width]`,
/// the rows of the sequences one after the other.
///
/// A decode step of several sequences is one launch of the decode kernel
/// (`pass.stacked`); else each sequence reads its own blocks: one query row in
/// place with the decode kernel, several rows as the visible blocks gathered
/// into one contiguous `[1, n_kv, len, d]` view and [`attend`]ed. `call.v_width`
/// selects latent attention, where the values are the leading columns of the
/// key rows (the caller passes the key pool as `pool_v` too).
///
/// # Errors
///
/// [`CucaError::Provider`] for a sequence without blocks for `window`.
fn attend_pass(
    pass: &Pass,
    il: usize,
    window: Option<usize>,
    q: &Tensor<4>,
    pool_k: &Tensor<4>,
    pool_v: &Tensor<4>,
    call: &PagedCall,
) -> Result<Tensor<3>, CucaError> {
    let [_, nh, t, key_width] = q.dims();
    let value_width = call.v_width.unwrap_or(key_width);
    if let Some(st) = pass.stacked.iter().find(|s| s.window == window) {
        return Ok(paged_kernel::decode(
            q.clone().swap_dims(1, 2).reshape([t, nh, key_width]),
            pool_k.clone(),
            pool_v.clone(),
            st.table.clone(),
            st.range.clone(),
            call,
        ));
    }
    let mut outs = Vec::with_capacity(pass.seqs.len());
    for seq in &pass.seqs {
        let (visible, blocks) = seq.visible(il, window)?;
        let qs = if pass.seqs.len() == 1 {
            q.clone()
        } else {
            q.clone().narrow(2, seq.off, seq.t)
        };
        outs.push(match &blocks.read {
            BlockRead::Decode { table, range } => paged_kernel::decode(
                qs.reshape([1, nh, key_width]),
                pool_k.clone(),
                pool_v.clone(),
                Tensor::clone(table),
                Tensor::clone(range),
                call,
            ),
            BlockRead::Gather { idx, skip, len } => {
                let keys = gather_view(pool_k, idx, *skip, *len);
                let values = match call.v_width {
                    Some(width) => keys.clone().narrow(3, 0, width),
                    None => gather_view(pool_v, idx, *skip, *len),
                };
                attend(qs, keys, values, visible.mask.clone(), call.scale)
                    .swap_dims(1, 2)
                    .reshape([seq.t, nh, value_width])
            }
            BlockRead::Rows { .. } => {
                return Err(err(format!(
                    "layer {il}: the step's blocks were stacked for one launch"
                )));
            }
        });
    }
    Ok(if outs.len() == 1 {
        outs.swap_remove(0)
    } else {
        Tensor::cat(outs, 0)
    })
}

/// Grouped-query attention, as the CPU `attend`: `q` is `[1, n_head, t, d]`,
/// `k` and `v` are `[1, n_kv, l, d]`, `mask` the additive `[1, 1, t, l]`
/// mask. The scores are scaled by `scale` before the mask. The `n_head / n_kv`
/// query heads of one KV head are stacked into the rows of one matmul.
/// Returns `[1, n_head, t, d]`.
fn attend(
    q: Tensor<4>,
    k: Tensor<4>,
    v: Tensor<4>,
    mask: Option<Tensor<4>>,
    scale: f32,
) -> Tensor<4> {
    let [_, n_head, t, d] = q.dims();
    let [_, n_kv, l, _] = k.dims();
    let [_, _, _, dv] = v.dims();
    let rows = n_head / n_kv * t;
    let mut scores = q.reshape([1, n_kv, rows, d]).matmul(k.transpose());
    if scale != 1.0 {
        scores = scores.mul_scalar(f64::from(scale));
    }
    let scores = match mask {
        Some(mask) => (scores.reshape([1, n_head, t, l]) + mask).reshape([1, n_kv, rows, l]),
        None => scores,
    };
    activation::softmax(scores, 3)
        .matmul(v)
        .reshape([1, n_head, t, dv])
}

#[cfg(test)]
mod tests {
    use std::io::BufReader;

    use super::super::cpu::Transformer;
    use super::super::paged::{KvPools, SeqBlocks};
    use super::super::split::SplitModel;
    use super::super::tiny::{
        Lcg, PROMPT, TinyMistral4, TinyModel, TinyMoe, TinyOptions, TinyQwen35, TinySpec, VOCAB,
        write_mistral4, write_qwen35, write_spec, write_tiny_model_with,
    };
    use super::super::{SeqRows, StateSlot};
    use super::*;

    fn f16_bytes(v: f32) -> [u8; 2] {
        half::f16::from_f32(v).to_bits().to_le_bytes()
    }

    /// The GPU devices under test, from `CUCA_LOCAL_DEVICE`: one device or a
    /// `+`-separated list (`vulkan:0+vulkan:1`); empty (skip) when unset or
    /// when the first is `cpu`.
    fn test_devices() -> Vec<LocalDevice> {
        let Ok(list) = std::env::var("CUCA_LOCAL_DEVICE") else {
            return Vec::new();
        };
        let devices: Vec<LocalDevice> = list
            .split('+')
            .map(|d| d.trim().parse().expect("CUCA_LOCAL_DEVICE"))
            .collect();
        if devices.first().is_none_or(|d| *d == LocalDevice::Cpu) {
            Vec::new()
        } else {
            devices
        }
    }

    /// The GPU device under test, the first of [`test_devices`]; `None`
    /// (skip) when unset or `cpu`.
    fn test_device() -> Option<LocalDevice> {
        let device = test_devices().first().copied();
        if device.is_none() {
            println!("SKIP: set CUCA_LOCAL_DEVICE to a GPU device");
        }
        device
    }

    fn content_of(model: &TinyModel) -> gguf_file::Content {
        gguf_file::Content::read(&mut BufReader::new(File::open(&model.path).unwrap())).unwrap()
    }

    fn load_both(model: &TinyModel, device: LocalDevice) -> (Transformer, GpuTransformer) {
        let content = content_of(model);
        (
            Transformer::load_dequantized(&content, &model.path, 32).unwrap(),
            GpuTransformer::load(&content, &model.path, 32, device).unwrap(),
        )
    }

    /// The absolute and relative tolerance of the GPU-versus-CPU compares
    /// on engines that agree to float rounding plus the Q8_0 requantization
    /// some weight types go through.
    const LOOSE: (f32, f32) = (1e-3, 1e-2);

    fn assert_matches_cpu(cpu: &[f32], gpu: &[f32], (abs, rel): (f32, f32), what: &str) {
        assert_eq!(cpu.len(), gpu.len(), "{what}");
        for (i, (c, g)) in cpu.iter().zip(gpu).enumerate() {
            assert!(
                (g - c).abs() <= abs + rel * c.abs(),
                "{what}: logit {i}: gpu {g} vs cpu {c}"
            );
        }
    }

    /// Run the same forward call on both engines and compare every returned
    /// logit.
    fn compare(
        cpu: &mut Transformer,
        gpu: &mut GpuTransformer,
        tokens: &[u32],
        start: usize,
        want: Want,
        what: &str,
    ) {
        compare_within(cpu, gpu, tokens, start, want, LOOSE, what);
    }

    /// [`compare`] to the tolerance `tol` (absolute, relative).
    fn compare_within(
        cpu: &mut Transformer,
        gpu: &mut GpuTransformer,
        tokens: &[u32],
        start: usize,
        want: Want,
        tol: (f32, f32),
        what: &str,
    ) {
        let c = cpu.forward(tokens, start, want).unwrap().unwrap();
        let g = gpu.forward(tokens, start, want).unwrap().unwrap();
        assert_matches_cpu(&c, &g, tol, what);
    }

    /// The GPU engine must reproduce the CPU engine on one tiny model:
    /// single-shot prefill, decode, every-position logits, chunked prefill
    /// and truncate-and-replay. The reference holds its weights dequantized:
    /// candle's quantized matmul rounds the activations to Q8_0, which the
    /// GPU kernels (f32 activations) do not, and over several random-weight
    /// layers that alone moves logits by several percent.
    fn check_model(model: &TinyModel, device: LocalDevice, label: &str) {
        check_model_within(model, device, label, LOOSE);
    }

    /// [`check_model`] to the tolerance `tol` (absolute, relative).
    fn check_model_within(model: &TinyModel, device: LocalDevice, label: &str, tol: (f32, f32)) {
        let n = PROMPT.len();

        // (a) single-shot prefill, then (c) one decode step.
        let (mut cpu, mut gpu) = load_both(model, device);
        compare_within(
            &mut cpu,
            &mut gpu,
            &PROMPT,
            0,
            Want::Last,
            tol,
            &format!("{label} prefill"),
        );
        compare_within(
            &mut cpu,
            &mut gpu,
            &[7],
            n,
            Want::Last,
            tol,
            &format!("{label} decode"),
        );

        // (e) every-position logits, with a prefix already cached.
        let (mut cpu, mut gpu) = load_both(model, device);
        assert!(gpu.forward(&PROMPT[..4], 0, Want::None).unwrap().is_none());
        assert!(cpu.forward(&PROMPT[..4], 0, Want::None).unwrap().is_none());
        compare_within(
            &mut cpu,
            &mut gpu,
            &PROMPT[4..],
            4,
            Want::All,
            tol,
            &format!("{label} all"),
        );

        // (b) chunked prefill in chunks of 3, crossing every window.
        let (mut cpu, mut gpu) = load_both(model, device);
        let mut start = 0;
        while start + 3 < n {
            let chunk = &PROMPT[start..start + 3];
            assert!(gpu.forward(chunk, start, Want::None).unwrap().is_none());
            cpu.forward(chunk, start, Want::None).unwrap();
            start += 3;
        }
        compare_within(
            &mut cpu,
            &mut gpu,
            &PROMPT[start..],
            start,
            Want::Last,
            tol,
            &format!("{label} chunked"),
        );
        assert_eq!(gpu.cache_len(), n);

        // (d) truncate and replay a different suffix.
        cpu.truncate_cache(5);
        gpu.truncate_cache(5);
        assert!(
            gpu.forward(&PROMPT[6..], 6, Want::Last).is_err(),
            "{label}: gap after truncate"
        );
        compare_within(
            &mut cpu,
            &mut gpu,
            &[11, 13, 15, 14, 13, 12],
            5,
            Want::Last,
            tol,
            &format!("{label} replay"),
        );
    }

    #[test]
    fn gpu_matches_cpu_on_the_tiny_gemma4() {
        let Some(device) = test_device() else {
            return;
        };
        for (label, options) in [
            ("gemma4 mqa", TinyOptions::default()),
            (
                "gemma4 gqa",
                TinyOptions {
                    n_head: 4,
                    n_kv_head: 2,
                    ..TinyOptions::default()
                },
            ),
        ] {
            check_model(&write_tiny_model_with(options), device, label);
        }
    }

    #[test]
    fn gpu_matches_cpu_on_every_architecture() {
        let Some(device) = test_device() else {
            return;
        };
        for spec in TinySpec::all() {
            check_model(&write_spec(&spec), device, spec.arch);
        }
        // Latent attention (and fused gate/up experts, an ungated shared
        // expert, a routing scale, YaRN and per-position query scaling)
        // against the dequantized CPU reference at 1e-4.
        for (label, spec) in mistral4_variants() {
            check_model_within(&write_mistral4(&spec), device, label, (1e-4, 1e-4));
        }
        for (label, spec) in [("qwen35", tiny_qwen35()), ("qwen35moe", tiny_qwen35moe())] {
            check_hybrid_model(&write_qwen35(&spec), device, label);
        }
    }

    /// The tiny `mistral4` models of the GPU tests: fused and split gate/up
    /// experts, a dense leading layer, float dense weights.
    fn mistral4_variants() -> Vec<(&'static str, TinyMistral4)> {
        vec![
            ("mistral4 fused", TinyMistral4::default()),
            (
                "mistral4 split",
                TinyMistral4 {
                    fused: false,
                    ..TinyMistral4::default()
                },
            ),
            (
                "mistral4 dense first layer",
                TinyMistral4 {
                    dense_layers: 1,
                    ..TinyMistral4::default()
                },
            ),
            (
                "mistral4 F32",
                TinyMistral4 {
                    dtype: GgmlDType::F32,
                    ..TinyMistral4::default()
                },
            ),
        ]
    }

    fn tiny_qwen35() -> TinyQwen35 {
        TinyQwen35::default()
    }

    fn tiny_qwen35moe() -> TinyQwen35 {
        TinyQwen35 {
            moe: Some(TinyMoe {
                n_expert: 8,
                n_used: 2,
                ff: 64,
                shared_ff: 32,
            }),
            ..TinyQwen35::default()
        }
    }

    /// A hybrid model (delta-net + gated attention, with a shared-expert MoE
    /// FFN for `qwen35moe`): the GPU engine reproduces the CPU engine
    /// through prefill, decode, chunked prefill and the rule of `rewind` (the
    /// slot's live state is kept, any length inside the cache falls back to
    /// the zero state), and a snapshot restored into the slot continues as
    /// the uninterrupted run did.
    fn check_hybrid_model(model: &TinyModel, device: LocalDevice, label: &str) {
        let n = PROMPT.len();
        let (mut cpu, mut gpu) = load_both(model, device);
        assert!(gpu.is_hybrid() && cpu.is_hybrid(), "{label}");
        compare(&mut cpu, &mut gpu, &PROMPT, 0, Want::Last, label);
        compare(&mut cpu, &mut gpu, &[7], n, Want::Last, label);
        compare(&mut cpu, &mut gpu, &[9, 2], n + 1, Want::All, label);

        // Chunked prefill, with a snapshot of the state after the first
        // chunks (handles to the slot's immutable state tensors).
        let (mut cpu, mut gpu) = load_both(model, device);
        for (start, chunk) in [(0, &PROMPT[..3]), (3, &PROMPT[3..6])] {
            assert!(gpu.forward(chunk, start, Want::None).unwrap().is_none());
            cpu.forward(chunk, start, Want::None).unwrap();
        }
        let snapshot = gpu.snapshot(0).unwrap();
        assert!(snapshot.bytes() > 0, "{label}: the snapshot holds state");
        assert!(gpu.snapshot(1).is_none(), "{label}: one slot");
        let continued = gpu.forward(&PROMPT[6..], 6, Want::Last).unwrap().unwrap();
        let reference = cpu.forward(&PROMPT[6..], 6, Want::Last).unwrap().unwrap();
        assert_matches_cpu(&reference, &continued, LOOSE, &format!("{label} chunked"));
        assert_eq!(gpu.cache_len(), n);

        // The K/V blocks go back to position 6 (nothing there is named) and
        // the snapshot becomes the slot's state: the continuation is the
        // uninterrupted run's.
        assert_eq!(gpu.solo.truncate(6), 6, "{label}");
        gpu.restore(0, &snapshot, 6).unwrap();
        assert!(
            gpu.restore(1, &snapshot, 6).is_err(),
            "{label}: slot out of range"
        );
        let again = gpu.forward(&PROMPT[6..], 6, Want::Last).unwrap().unwrap();
        assert_close_logits(&[again], &[continued], &format!("{label} restored"));
        assert_eq!(gpu.cache_len(), n);
        assert!(
            gpu.forward(&PROMPT[8..], 8, Want::Last).is_err(),
            "{label}: gap after the restored run"
        );

        // The end of the cache is retained as is; any length inside it has
        // no state (the GPU keeps no checkpoint and no verify record), so
        // both engines fall back to the zero state.
        assert_eq!(gpu.truncate_cache(99), cpu.truncate_cache(99), "{label}");
        assert_eq!(gpu.cache_len(), n, "{label}: the end of the cache");
        assert_eq!(gpu.truncate_cache(9), cpu.truncate_cache(9), "{label}");
        assert_eq!(gpu.cache_len(), 0, "{label}: no state at 9");
        compare(
            &mut cpu,
            &mut gpu,
            &PROMPT,
            0,
            Want::Last,
            &format!("{label} from the zero state"),
        );
        assert_eq!(gpu.truncate_cache(0), 0);
    }

    /// One sequence's rows of a step of [`check_batched`].
    #[derive(Clone, Copy)]
    struct Step {
        seq: usize,
        n: usize,
        want: Want,
        hidden: Hidden,
    }

    fn step(seq: usize, n: usize, want: Want, hidden: Hidden) -> Step {
        Step {
            seq,
            n,
            want,
            hidden,
        }
    }

    /// `len` tokens of test sequence `seq`: different streams per sequence.
    fn stream(seq: usize, len: usize) -> Vec<u32> {
        (0..len)
            .map(|i| ((i * 7 + seq * 5 + 1) % VOCAB) as u32)
            .collect()
    }

    /// The logits and hidden rows of two runs of one sequence's call agree to
    /// `tol` of their scale (and are present in both or neither). Batched and
    /// solo steps run different matmul shapes (and, in a mixture of experts,
    /// group different rows per expert), so they agree up to float rounding
    /// of the vector's scale.
    fn assert_forward_within(got: &Forward, want: &Forward, tol: f32, what: &str) {
        for (name, g, w) in [
            ("logits", &got.logits, &want.logits),
            ("hidden", &got.hidden, &want.hidden),
        ] {
            match (g, w) {
                (Some(g), Some(w)) => assert_rows_within(g, w, tol, &format!("{what} {name}")),
                (None, None) => {}
                _ => panic!("{what}: {name} is present in only one of the runs"),
            }
        }
    }

    /// Three sequences run as batched steps (prefill together, decode
    /// together, one absent from a step, one prefilling a chunk next to
    /// decoding ones, `Want::All` and every-position hidden rows) equal each
    /// sequence run alone through the solo API of another GPU load. A hybrid
    /// model also forks sequence 1 from a snapshot of its state in a spare
    /// slot and the fork's continuation equals the original's.
    fn check_batched(model: &TinyModel, device: LocalDevice, label: &str) {
        const CTX: usize = 32;
        let content = content_of(model);
        // Three sequences and a spare state slot for the fork.
        let sizing = Sizing {
            max_sequences: 4,
            kv_tokens: 2 * CTX,
            max_rows: 16,
        };
        let mut batched = GpuTransformer::load_placed(
            &content,
            &model.path,
            CTX,
            device,
            &[],
            &ExpertPlacement::Resident,
            &sizing,
        )
        .unwrap();
        let mut solo = GpuTransformer::load(&content, &model.path, CTX, device).unwrap();
        let hybrid = batched.is_hybrid();
        assert_eq!(solo.is_hybrid(), hybrid, "{label}");

        let (last, all, none) = (Want::Last, Want::All, Want::None);
        let (plain, post, post_all) = (Hidden::None, Hidden::PostNorm, Hidden::PostNormAll);
        // 14, 17 and 10 tokens per sequence in total.
        let steps = [
            vec![
                step(0, 5, last, plain),
                step(1, 9, last, post),
                step(2, 3, last, plain),
            ],
            vec![
                step(0, 1, last, post),
                step(1, 1, last, plain),
                step(2, 1, last, post),
            ],
            vec![
                step(0, 1, last, plain),
                step(1, 1, last, plain),
                step(2, 1, last, plain),
            ],
            // Sequence 0 sits this step out.
            vec![step(1, 1, last, plain), step(2, 1, last, plain)],
            // Sequence 0 prefills a chunk next to two decoding ones.
            vec![
                step(0, 4, last, plain),
                step(1, 1, last, post),
                step(2, 1, last, plain),
            ],
            // Hidden rows only, every logits row, one logits row.
            vec![
                step(0, 2, none, post_all),
                step(1, 3, all, post),
                step(2, 1, last, plain),
            ],
            vec![
                step(0, 1, last, plain),
                step(1, 1, last, plain),
                step(2, 2, last, post_all),
            ],
        ];
        let tokens: Vec<Vec<u32>> = (0..3).map(|s| stream(s, 20)).collect();

        let mut pools = KvPools::new(batched.kv_layout());
        let mut seqs: Vec<SeqBlocks> = (0..3).map(|_| pools.seq()).collect();
        let mut at = [0usize; 3];
        let mut got: Vec<Vec<Forward>> = (0..3).map(|_| Vec::new()).collect();
        let mut snapshot = None;
        for (si, rows_of_step) in steps.iter().enumerate() {
            for st in rows_of_step {
                pools.reserve(&mut seqs[st.seq], at[st.seq] + st.n).unwrap();
            }
            let rows: Vec<SeqRows<'_>> = rows_of_step
                .iter()
                .map(|st| SeqRows {
                    tokens: &tokens[st.seq][at[st.seq]..at[st.seq] + st.n],
                    start_pos: at[st.seq],
                    tables: &seqs[st.seq].tables,
                    state: StateSlot(st.seq),
                    want: st.want,
                    hidden: st.hidden,
                })
                .collect();
            let outs = batched
                .forward_batch(&Batch { rows, exact: false })
                .unwrap();
            assert_eq!(outs.len(), rows_of_step.len(), "{label} step {si}");
            for (st, out) in rows_of_step.iter().zip(outs) {
                got[st.seq].push(out);
                at[st.seq] += st.n;
            }
            if si == 2 {
                // Sequence 1 sits at position 11.
                snapshot = batched.snapshot(1);
            }
        }
        assert_eq!(at, [14, 17, 10], "{label}");
        assert_eq!(
            snapshot.is_some(),
            hybrid,
            "{label}: a snapshot of a hybrid"
        );

        // Each sequence alone.
        for s in 0..3 {
            assert_eq!(solo.truncate_cache(0), 0, "{label}");
            let (mut pos, mut k) = (0, 0);
            for st in steps.iter().flatten().filter(|st| st.seq == s) {
                let alone = solo
                    .forward_ext(&tokens[s][pos..pos + st.n], pos, st.want, st.hidden, false)
                    .unwrap();
                assert_forward_within(
                    &got[s][k],
                    &alone,
                    1e-3,
                    &format!("{label} seq {s} call {k}"),
                );
                pos += st.n;
                k += 1;
            }
        }

        let bad = |slot: usize, start_pos: usize, tables: &'static [Vec<u32>]| SeqRows {
            tokens: &[1],
            start_pos,
            tables,
            state: StateSlot(slot),
            want: Want::Last,
            hidden: Hidden::None,
        };
        assert!(
            batched
                .forward_batch(&Batch {
                    rows: Vec::new(),
                    exact: false
                })
                .is_err(),
            "{label}: empty batch"
        );
        // A table without blocks fails the step (and empties its slot).
        let failed = batched.forward_batch(&Batch {
            rows: vec![bad(3, 0, &[])],
            exact: false,
        });
        assert!(failed.is_err(), "{label}: no block table");

        if !hybrid {
            assert!(batched.snapshot(0).is_none(), "{label}");
            assert_eq!(batched.rewind(0, 7), 7, "{label}: K/V only");
            return;
        }
        let snapshot = snapshot.unwrap();
        assert!(snapshot.bytes() > 0, "{label}");
        // Slot 3 continues sequence 1 from the snapshot of its state at
        // position 11, over sequence 1's blocks (the rows from position 11
        // are written again with the same values).
        let mut pos = 11;
        batched.restore(3, &snapshot, pos).unwrap();
        for (k, st) in (3..).zip(steps[3..].iter().flatten().filter(|st| st.seq == 1)) {
            let rows = vec![SeqRows {
                tokens: &tokens[1][pos..pos + st.n],
                start_pos: pos,
                tables: &seqs[1].tables,
                state: StateSlot(3),
                want: st.want,
                hidden: st.hidden,
            }];
            let mut outs = batched
                .forward_batch(&Batch { rows, exact: false })
                .unwrap();
            assert_forward_within(
                &outs.swap_remove(0),
                &got[1][k],
                1e-3,
                &format!("{label} fork {k}"),
            );
            pos += st.n;
        }
        assert_eq!(pos, 17, "{label}");

        // The rule of `rewind`: the end is retained, anything inside the
        // cache falls back to the zero state, a missing slot has none.
        assert_eq!(batched.rewind(3, 99), 17, "{label}");
        assert_eq!(batched.rewind(3, 5), 0, "{label}");
        assert_eq!(batched.rewind(3, 99), 0, "{label}: reset");
        assert_eq!(batched.rewind(9, 4), 0, "{label}: no such slot");
        assert!(batched.restore(9, &snapshot, 4).is_err(), "{label}");
        assert_eq!(batched.rewind(1, 99), 17, "{label}: slot 1 untouched");

        // A failed step resets the slot it names.
        batched.restore(3, &snapshot, 11).unwrap();
        assert!(
            batched
                .forward_batch(&Batch {
                    rows: vec![bad(3, 11, &[])],
                    exact: false,
                })
                .is_err(),
            "{label}"
        );
        assert_eq!(batched.rewind(3, 99), 0, "{label}: emptied by the failure");

        // The same slot twice in a step and a start other than the slot's
        // position are rejected before anything runs.
        let twice = Batch {
            rows: vec![
                SeqRows {
                    tokens: &tokens[1][17..18],
                    start_pos: 17,
                    tables: &seqs[1].tables,
                    state: StateSlot(1),
                    want: Want::Last,
                    hidden: Hidden::None,
                },
                SeqRows {
                    tokens: &tokens[1][17..18],
                    start_pos: 17,
                    tables: &seqs[1].tables,
                    state: StateSlot(1),
                    want: Want::Last,
                    hidden: Hidden::None,
                },
            ],
            exact: false,
        };
        assert!(
            batched.forward_batch(&twice).is_err(),
            "{label}: slot twice"
        );
        let off = Batch {
            rows: vec![SeqRows {
                tokens: &tokens[0][..1],
                start_pos: 3,
                tables: &seqs[0].tables,
                state: StateSlot(0),
                want: Want::Last,
                hidden: Hidden::None,
            }],
            exact: false,
        };
        assert!(batched.forward_batch(&off).is_err(), "{label}: wrong start");
        assert_eq!(batched.rewind(0, 99), 14, "{label}: slot 0 untouched");
    }

    #[test]
    fn gpu_batched_matches_gpu_solo_on_every_architecture() {
        let Some(device) = test_device() else {
            return;
        };
        check_batched(
            &write_tiny_model_with(TinyOptions::default()),
            device,
            "gemma4",
        );
        for spec in TinySpec::all() {
            check_batched(&write_spec(&spec), device, spec.arch);
        }
        for (label, spec) in [("qwen35", tiny_qwen35()), ("qwen35moe", tiny_qwen35moe())] {
            check_batched(&write_qwen35(&spec), device, label);
        }
        for (label, spec) in mistral4_variants() {
            check_batched(&write_mistral4(&spec), device, label);
        }
    }

    /// Device bytes of one expert summed over the MoE layers of `model`, as
    /// the loader counts them.
    fn expert_device_bytes(model: &TinyModel) -> usize {
        let content = content_of(model);
        let hp = read_hparams(&content).unwrap();
        let layers = tier_layout(&content, &hp, hp.moe.unwrap(), 0).unwrap();
        layers.iter().map(|l| l.stored_bytes).sum()
    }

    fn load_placed_gpu(
        model: &TinyModel,
        device: LocalDevice,
        experts: &ExpertPlacement,
    ) -> Result<GpuTransformer, CucaError> {
        let content = content_of(model);
        GpuTransformer::load_placed(
            &content,
            &model.path,
            32,
            device,
            &[],
            experts,
            &Sizing::single(32),
        )
    }

    /// A 24-token prompt in chunks of 16 then 8 greedy steps: the logits of
    /// every call.
    fn greedy_run(m: &mut GpuTransformer) -> Vec<Vec<f32>> {
        let prompt: Vec<u32> = (0..24u32).map(|i| (i * 5 + 3) % VOCAB as u32).collect();
        let mut out: Vec<Vec<f32>> = Vec::new();
        let mut at = 0;
        for chunk in prompt.chunks(16) {
            out.push(m.forward(chunk, at, Want::Last).unwrap().unwrap());
            at += chunk.len();
        }
        for _ in 0..8 {
            let last = out.last().unwrap();
            let next = (0..last.len())
                .max_by(|&a, &b| last[a].total_cmp(&last[b]))
                .unwrap() as u32;
            out.push(m.forward(&[next], at, Want::Last).unwrap().unwrap());
            at += 1;
        }
        out
    }

    /// Logits of two runs that differ only in where the experts live: the
    /// resident run never reads the router output back, a tiered run does
    /// at every layer, which splits burn's fused elementwise kernels at
    /// different places, so they agree to float rounding (tiered placements
    /// among themselves agree bit for bit). The bound is `3e-4` relative to
    /// `1 + |logit|`: the tiny hybrid MoE model has one sensitive step (the
    /// second of the greedy run) where the order of float accumulation, not
    /// the routing, moves the logits by far more than at the other steps.
    fn assert_close_logits(got: &[Vec<f32>], want: &[Vec<f32>], what: &str) {
        assert_eq!(got.len(), want.len(), "{what}");
        for (step, (g, w)) in got.iter().zip(want).enumerate() {
            assert_eq!(g.len(), w.len(), "{what}");
            for (i, (a, b)) in g.iter().zip(w).enumerate() {
                assert!(
                    (a - b).abs() <= 3e-4 * (1.0 + b.abs()),
                    "{what}: step {step} logit {i}: {a} vs {b}"
                );
            }
            let top = |v: &[f32]| {
                v.iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .map(|p| p.0)
            };
            assert_eq!(top(g), top(w), "{what}: step {step} argmax");
        }
    }

    /// Placement only decides speed: with the smallest device budget (two of
    /// eight experts per layer, so nearly every routing evicts), with and
    /// without a RAM tier, and warm-started from the heat of a run, the
    /// logits equal the resident GPU load's to float rounding, for every MoE
    /// architecture (same kernels, same packed words; the kernels' results
    /// do not depend on how a pass groups its experts). All loads share one
    /// process, so burn's autotune picks are shared.
    #[test]
    fn gpu_tiered_experts_match_gpu_resident() {
        let Some(device) = test_device() else {
            return;
        };
        let qwen35moe = write_qwen35(&tiny_qwen35moe());
        let models: Vec<(&str, TinyModel)> = [TinySpec::qwen3moe(), TinySpec::olmoe()]
            .into_iter()
            .map(|spec| {
                let arch = spec.arch;
                let model = write_spec(&TinySpec {
                    moe: Some((8, 2)),
                    n_layer: 4,
                    ..spec
                });
                (arch, model)
            })
            .chain([
                ("qwen35moe", qwen35moe),
                (
                    "mistral4",
                    write_mistral4(&TinyMistral4 {
                        layers: 4,
                        ..TinyMistral4::default()
                    }),
                ),
                (
                    "mistral4 dense first layer",
                    write_mistral4(&TinyMistral4 {
                        layers: 4,
                        dense_layers: 1,
                        ..TinyMistral4::default()
                    }),
                ),
            ])
            .collect();
        for (arch, model) in &models {
            let min = 2 * expert_device_bytes(model) as u64;
            let want = greedy_run(
                &mut load_placed_gpu(model, device, &ExpertPlacement::Resident).unwrap(),
            );
            let tiers = |ram_bytes, warm_from| {
                ExpertPlacement::Tiered(ExpertTiers {
                    ram_bytes,
                    device_bytes: min,
                    loader_threads: 2,
                    warm_from,
                })
            };
            let mut heat = None;
            for ram_bytes in [0, min] {
                let mut tiered = load_placed_gpu(model, device, &tiers(ram_bytes, None)).unwrap();
                let got = greedy_run(&mut tiered);
                assert_close_logits(&got, &want, &format!("{arch} ram={ram_bytes}"));
                let stats = tiered.expert_tiers().unwrap().stats();
                assert!(stats.disk_reads > 0, "{arch}: {stats:?}");
                assert!(stats.device_resident_bytes <= min, "{arch}: {stats:?}");
                assert!(stats.ram_resident_bytes <= ram_bytes, "{arch}: {stats:?}");
                assert_eq!(
                    stats.lookups,
                    stats.device_hits + stats.ram_hits + stats.disk_reads,
                    "{arch}: {stats:?}"
                );
                heat = Some(tiered.expert_tiers().unwrap().heat());
            }
            // The warm start fills the device tier at load and changes no
            // logit.
            let mut warm = load_placed_gpu(model, device, &tiers(min, heat)).unwrap();
            let loaded = warm.expert_tiers().unwrap().stats();
            assert!(loaded.device_resident_bytes > 0, "{arch}: {loaded:?}");
            assert!(loaded.device_resident_bytes <= min, "{arch}: {loaded:?}");
            let got = greedy_run(&mut warm);
            assert_close_logits(&got, &want, &format!("{arch} warm"));
            let stats = warm.expert_tiers().unwrap().stats();
            assert!(stats.device_resident_bytes <= min, "{arch}: {stats:?}");
            assert!(stats.device_hits > 0, "{arch}: {stats:?}");

            // A budget below `n_used` experts per layer names the minimum.
            let below = ExpertPlacement::Tiered(ExpertTiers {
                device_bytes: min - 1,
                loader_threads: 2,
                ..ExpertTiers::default()
            });
            let e = load_placed_gpu(model, device, &below).err().unwrap();
            assert!(
                matches!(&e, CucaError::Config(m) if m.contains(&format!("at least 2 ({min} bytes)"))),
                "{arch}: {e:?}"
            );
        }
        let dense = write_spec(&TinySpec::llama());
        let placement = ExpertPlacement::Tiered(ExpertTiers {
            device_bytes: 1 << 30,
            loader_threads: 2,
            ..ExpertTiers::default()
        });
        let e = load_placed_gpu(&dense, device, &placement).err().unwrap();
        assert!(
            matches!(&e, CucaError::Config(m) if m.contains("`llama` has no experts")),
            "{e:?}"
        );
    }

    /// Host experts (`ExpertPlacement::Host`: the CPU computes the routed
    /// experts from RAM, the GPU everything else) give the resident GPU
    /// logits to 1e-4 for every MoE architecture, hosting every MoE layer or
    /// only the first one (the others stay on the device), and a model
    /// without experts is rejected.
    #[test]
    fn gpu_host_experts_match_gpu_resident() {
        let Some(device) = test_device() else {
            return;
        };
        let mut models: Vec<(String, TinyModel)> = [TinySpec::qwen3moe(), TinySpec::olmoe()]
            .into_iter()
            .map(|spec| {
                let arch = spec.arch.to_string();
                let model = write_spec(&TinySpec {
                    moe: Some((8, 2)),
                    n_layer: 4,
                    ..spec
                });
                (arch, model)
            })
            .collect();
        models.push(("qwen35moe".into(), write_qwen35(&tiny_qwen35moe())));
        for (label, spec) in mistral4_variants() {
            models.push((label.to_string(), write_mistral4(&spec)));
        }
        // The CPU half holds the hosted experts as f32 (see
        // [`GpuTransformer::load_host_exact`]).
        for (label, model) in &models {
            let want = greedy_run(
                &mut load_placed_gpu(model, device, &ExpertPlacement::Resident).unwrap(),
            );
            let content = content_of(model);
            for layers in [None, Some(1)] {
                let mut host = GpuTransformer::load_host_exact(
                    &content,
                    &model.path,
                    32,
                    device,
                    &ExpertPlacement::Host { layers },
                    &Sizing::single(32),
                )
                .unwrap();
                let got = greedy_run(&mut host);
                assert_close_logits(&got, &want, &format!("{label} host {layers:?}"));
            }
        }
        let dense = write_spec(&TinySpec::llama());
        let e = load_placed_gpu(&dense, device, &ExpertPlacement::Host { layers: None })
            .err()
            .unwrap();
        assert!(
            matches!(&e, CucaError::Config(m) if m.contains("host experts need a mixture-of-experts model")),
            "{e:?}"
        );
    }

    /// Past a sliding window (the tiny Gemma 3 at 1400 positions, window 4,
    /// whose blocks behind the window are released) the blocks it holds, the
    /// gathered reads and the decode kernel agree with the CPU engine's, also
    /// after a rollback inside the live window.
    #[test]
    fn gpu_matches_cpu_on_a_bounded_sliding_cache() {
        let Some(device) = test_device() else {
            return;
        };
        let model = write_spec(&TinySpec {
            context: 1400,
            ..TinySpec::gemma3()
        });
        let content = content_of(&model);
        let mut cpu = Transformer::load_dequantized(&content, &model.path, 1400).unwrap();
        let mut gpu = GpuTransformer::load(&content, &model.path, 1400, device).unwrap();
        let tokens: Vec<u32> = (0..1350u32).map(|i| (i * 7 + i / 5) % 16).collect();
        for (i, chunk) in tokens.chunks(256).enumerate() {
            let at = i * 256;
            compare(
                &mut cpu,
                &mut gpu,
                chunk,
                at,
                Want::Last,
                &format!("chunk at {at}"),
            );
        }
        let sliding = gpu
            .kv_layout()
            .groups
            .iter()
            .position(|g| g.window.is_some())
            .unwrap();
        let live = gpu.solo_tables()[sliding]
            .iter()
            .filter(|&&b| b != NO_BLOCK)
            .count();
        assert!(
            live <= 4,
            "the sliding group released its old blocks: {live}"
        );
        for (j, token) in [3u32, 9, 1, 14, 6].into_iter().enumerate() {
            let at = 1350 + j;
            compare(
                &mut cpu,
                &mut gpu,
                &[token],
                at,
                Want::Last,
                &format!("step at {at}"),
            );
        }
        // A rollback inside the rows a verification round keeps.
        assert_eq!(gpu.truncate_cache(1350), 1350);
        assert_eq!(cpu.truncate_cache(1350), 1350);
        compare(
            &mut cpu,
            &mut gpu,
            &[4, 8],
            1350,
            Want::All,
            "after rollback",
        );
        assert_eq!(
            gpu.truncate_cache(100),
            0,
            "the window before 100 was released: emptied"
        );
        assert_eq!(cpu.truncate_cache(100), 0);
        compare(&mut cpu, &mut gpu, &tokens[..40], 0, Want::Last, "refilled");
    }

    /// One decode-kernel call: the query row `q` (`n_head * d` values) over
    /// the positions `lo..hi` of the sequence whose block table is `table`,
    /// launched the way a pass launches it (the table from the first block
    /// the range touches, the range relative to it).
    fn decode_row(
        device: &Device,
        pools: (&Tensor<4>, &Tensor<4>),
        table: &[u32],
        (lo, hi): (usize, usize),
        q: &[f32],
        (nh, d): (usize, usize),
        call: &PagedCall,
    ) -> Vec<f32> {
        let first = lo / BLOCK_SIZE;
        let last = (hi - 1) / BLOCK_SIZE;
        let ids: Vec<i32> = table[first..=last].iter().map(|&b| b as i32).collect();
        let n = ids.len();
        let table = Tensor::<2, Int>::from_data(TensorData::new(ids, [1, n]), device);
        let base = first * BLOCK_SIZE;
        let range = vec![(lo - base) as i32, (hi - base) as i32];
        let range = Tensor::<1, Int>::from_data(TensorData::new(range, [2]), device);
        let q = Tensor::<3>::from_data(TensorData::new(q.to_vec(), [1, nh, d]), device);
        let out = paged_kernel::decode(q, pools.0.clone(), pools.1.clone(), table, range, call);
        read_f32(out, "decode").unwrap()
    }

    /// The decode kernel reads scrambled blocks like the CPU's paged
    /// attention does: lengths around block boundaries, with and without a
    /// soft-cap and a window floor, grouped query heads.
    #[test]
    fn gpu_paged_decode_kernel_matches_cpu() {
        use super::super::cpu::paged_attn::{PoolShape, attend_paged_row};

        let Some(device) = test_device() else {
            return;
        };
        let device = open_device(device).unwrap();
        let mut rand = Lcg(7);
        let n_blocks = 12;
        // Blocks of the sequence, in no particular order.
        let scrambled: [u32; 6] = [7, 2, 11, 0, 9, 4];
        for (nh, nkv, d) in [(4usize, 2usize, 16usize), (6, 2, 8), (2, 2, 32)] {
            let pool_len = n_blocks * nkv * BLOCK_SIZE * d;
            let host_k = rand.values(pool_len, 1.0);
            let host_v = rand.values(pool_len, 1.0);
            let shape = [n_blocks, nkv, BLOCK_SIZE, d];
            let pool_k = Tensor::<4>::from_data(TensorData::new(host_k.clone(), shape), &device);
            let pool_v = Tensor::<4>::from_data(TensorData::new(host_v.clone(), shape), &device);
            let scale = 1.0 / (d as f32).sqrt();
            for hi in [1usize, 2, 15, 16, 17, 31, 32, 33, 70] {
                // The whole prefix, a short window and (past one block)
                // a floor at a block boundary.
                let mut los = vec![0, hi.saturating_sub(5)];
                if hi > BLOCK_SIZE + 1 {
                    los.push(BLOCK_SIZE);
                }
                los.dedup();
                for lo in los {
                    // Blocks before the range may be released.
                    let mut table = scrambled.to_vec();
                    table[..lo / BLOCK_SIZE].fill(NO_BLOCK);
                    let q = rand.values(nh * d, 4.0);
                    let want = attend_paged_row(
                        &q,
                        &table,
                        lo,
                        hi,
                        &host_k,
                        &host_v,
                        PoolShape {
                            n_head: nh,
                            n_kv_head: nkv,
                            head_dim: d,
                        },
                        scale,
                    )
                    .unwrap();
                    let call = PagedCall {
                        scale,
                        v_width: None,
                    };
                    let got = decode_row(
                        &device,
                        (&pool_k, &pool_v),
                        &table,
                        (lo, hi),
                        &q,
                        (nh, d),
                        &call,
                    );
                    assert_eq!(got.len(), want.len());
                    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                        assert!(
                            (g - w).abs() <= 1e-4,
                            "heads {nh}/{nkv} d {d} {lo}..{hi}: value {i}: gpu {g} vs cpu {w}"
                        );
                    }
                }
            }
        }
    }

    /// The latent decode kernel (one shared `[latent | rotary]` row per
    /// position, the value its first `kv_lora` columns, every query head
    /// reading it) equals the CPU's latent attention over scrambled blocks,
    /// at lengths around block boundaries, with and without a range floor,
    /// for the key/value widths of the tiny model and of Mistral Small 4
    /// (320 and 256 with 32 heads).
    #[test]
    fn gpu_paged_latent_decode_kernel_matches_cpu() {
        use super::super::cpu::paged_attn::{PoolShape, attend_latent_rows};

        let Some(device) = test_device() else {
            return;
        };
        let device = open_device(device).unwrap();
        let mut rand = Lcg(13);
        let n_blocks = 12;
        let scrambled: [u32; 6] = [7, 2, 11, 0, 9, 4];
        for (nh, width, v_width) in [(4usize, 24usize, 16usize), (32, 320, 256)] {
            let pool_len = n_blocks * BLOCK_SIZE * width;
            let host = rand.values(pool_len, 1.0);
            let shape = [n_blocks, 1, BLOCK_SIZE, width];
            let pool = Tensor::<4>::from_data(TensorData::new(host.clone(), shape), &device);
            let scale = 1.0 / (width as f32).sqrt();
            for hi in [1usize, 2, 15, 16, 17, 33, 70] {
                let mut los = vec![0];
                if hi > BLOCK_SIZE + 1 {
                    los.push(BLOCK_SIZE);
                }
                for lo in los {
                    let mut table = scrambled.to_vec();
                    table[..lo / BLOCK_SIZE].fill(NO_BLOCK);
                    let q = rand.values(nh * width, 4.0);
                    let want = attend_latent_rows(
                        &q,
                        1,
                        |_| (lo, hi),
                        &table,
                        &host,
                        PoolShape {
                            n_head: nh,
                            n_kv_head: 1,
                            head_dim: width,
                        },
                        v_width,
                        scale,
                    )
                    .unwrap();
                    let call = PagedCall {
                        scale,
                        v_width: Some(v_width),
                    };
                    let got = decode_row(
                        &device,
                        (&pool, &pool),
                        &table,
                        (lo, hi),
                        &q,
                        (nh, width),
                        &call,
                    );
                    assert_eq!(got.len(), nh * v_width);
                    assert_eq!(want.len(), nh * v_width);
                    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                        assert!(
                            (g - w).abs() <= 1e-4,
                            "heads {nh} width {width}/{v_width} {lo}..{hi}: value {i}: gpu {g} vs cpu {w}"
                        );
                    }
                }
            }
        }
    }

    /// The gather path of a prefill call (the visible blocks as one tensor,
    /// burn's matmul attention with the mask) equals the decode kernel row by
    /// row, each row over its own visible range.
    #[test]
    fn gpu_prefill_gather_matches_decode_rows() {
        let Some(device) = test_device() else {
            return;
        };
        let device = open_device(device).unwrap();
        let mut rand = Lcg(11);
        let (nh, nkv, d, n_blocks) = (4usize, 2usize, 16usize, 12usize);
        let pool_len = n_blocks * nkv * BLOCK_SIZE * d;
        let shape = [n_blocks, nkv, BLOCK_SIZE, d];
        let host_k = rand.values(pool_len, 1.0);
        let host_v = rand.values(pool_len, 1.0);
        let pool_k = Tensor::<4>::from_data(TensorData::new(host_k, shape), &device);
        let pool_v = Tensor::<4>::from_data(TensorData::new(host_v, shape), &device);
        let table: Vec<u32> = vec![7, 2, 11, 0, 9, 4];
        let scale = 1.0 / (d as f32).sqrt();
        // Twenty rows from the middle of the second block into the third.
        let (start, t) = (21usize, 20usize);
        let hi = start + t;
        let q = rand.values(nh * t * d, 4.0);
        for window in [None, Some(8usize)] {
            let lo = window.map_or(0, |w| start.saturating_sub(w - 1));
            let rows = BlockRows {
                table: &table,
                n_blocks,
                start,
                t,
                lo,
                hi,
                write: false,
                stack: false,
            };
            let BlockRead::Gather { idx, skip, len } = rows.view(&device).unwrap().read else {
                panic!("several rows gather");
            };
            let mask = Tensor::<4>::from_data(
                TensorData::new(attention_mask(start, t, lo, hi, window), [1, 1, t, hi - lo]),
                &device,
            );
            let qt = Tensor::<4>::from_data(TensorData::new(q.clone(), [1, nh, t, d]), &device);
            let out = attend(
                qt,
                gather_view(&pool_k, &idx, skip, len),
                gather_view(&pool_v, &idx, skip, len),
                Some(mask),
                scale,
            );
            let gathered = read_f32(out, "gathered attention").unwrap();
            let call = PagedCall {
                scale,
                v_width: None,
            };
            for r in 0..t {
                let pos = start + r;
                let row_lo = window.map_or(0, |w| lo.max((pos + 1).saturating_sub(w)));
                let q_row: Vec<f32> = (0..nh)
                    .flat_map(|h| q[(h * t + r) * d..][..d].iter().copied())
                    .collect();
                let row = decode_row(
                    &device,
                    (&pool_k, &pool_v),
                    &table,
                    (row_lo, pos + 1),
                    &q_row,
                    (nh, d),
                    &call,
                );
                for h in 0..nh {
                    for i in 0..d {
                        let (g, k) = (gathered[(h * t + r) * d + i], row[h * d + i]);
                        assert!(
                            (g - k).abs() <= 1e-4,
                            "window {window:?} row {r} head {h}[{i}]: gather {g} vs kernel {k}"
                        );
                    }
                }
            }
        }
    }

    /// A model whose matmul weights are float or Q8_0 matches the dequantized
    /// CPU reference as closely as a Q4_0 one: no requantization happens.
    /// A model in Q4_1, Q5_0 or Q5_1, which the engine requantizes to Q8_0,
    /// is within the Q8_0 rounding of it: at most 8% of the largest logit
    /// (a random-weight model amplifies the rounding).
    #[test]
    fn gpu_matches_cpu_on_other_weight_dtypes() {
        let Some(device) = test_device() else {
            return;
        };
        for dtype in [
            GgmlDType::F16,
            GgmlDType::F32,
            GgmlDType::Q8_0,
            GgmlDType::Q4_1,
            GgmlDType::Q5_0,
            GgmlDType::Q5_1,
        ] {
            let requantized = !matches!(dtype, GgmlDType::F16 | GgmlDType::F32 | GgmlDType::Q8_0);
            let spec = TinySpec {
                weight_dtype: dtype,
                embd_dtype: GgmlDType::Q5_0,
                ..TinySpec::llama()
            };
            let model = write_spec(&spec);
            let (mut cpu, mut gpu) = load_both(&model, device);
            for (tokens, start, what) in [
                (&PROMPT[..], 0, "prefill"),
                (&[7][..], PROMPT.len(), "decode"),
            ] {
                let c = cpu.forward(tokens, start, Want::Last).unwrap().unwrap();
                let g = gpu.forward(tokens, start, Want::Last).unwrap().unwrap();
                if requantized {
                    let scale = c.iter().fold(1f32, |m, v| m.max(v.abs()));
                    let worst = c
                        .iter()
                        .zip(&g)
                        .fold(0f32, |m, (a, b)| m.max((a - b).abs()))
                        / scale;
                    println!("{dtype:?} {what}: max error {worst:.4} of the largest logit");
                    assert!(worst <= 0.08, "{dtype:?} {what}: {worst}");
                } else {
                    assert_matches_cpu(&c, &g, LOOSE, &format!("{dtype:?} {what}"));
                }
            }
        }
    }

    /// `linear` against a host matmul of the dequantized weight, over shapes
    /// that stress burn's Vulkan matmul and `vecmat` paths. The order replays cached
    /// autotune entries across shapes of one power-of-two bucket (`out = 96`
    /// then `128`) and mixes one-row inputs with weights the `vecmat` kernel
    /// can launch (`512 x 1024`) and cannot (`96 x 64`), and small odd row
    /// counts.
    #[test]
    fn linear_matches_host_across_shapes() {
        let Some(device) = test_device() else {
            return;
        };
        let device = open_device(device).unwrap();
        let shapes = [
            (96, 64),
            (128, 64),
            (64, 96),
            (64, 128),
            (4, 64),
            (16, 64),
            (512, 1024),
            (1024, 512),
            (768, 256),
            (256, 768),
        ];
        for (out, inn) in shapes {
            for dtype in [GgmlDType::Q4_0, GgmlDType::Q8_0, GgmlDType::F32] {
                let mut raw = Vec::new();
                if dtype == GgmlDType::F32 {
                    for i in 0..out * inn {
                        raw.extend((((i * 13 + 1) % 17) as f32 * 0.02 - 0.15).to_le_bytes());
                    }
                } else {
                    for b in 0..out * inn / 32 {
                        raw.extend(f16_bytes(0.05 + (b % 7) as f32 * 0.01));
                        raw.extend(
                            (0..dtype.type_size() - 2).map(|j| ((b * 31 + j * 17 + 5) % 251) as u8),
                        );
                    }
                }
                let w = upload_matrix("w", dtype, &raw, out, inn, &device).unwrap();
                let mut host = vec![0f32; out * inn];
                let row_bytes = inn / dtype.block_size() * dtype.type_size();
                for (src, dst) in raw.chunks_exact(row_bytes).zip(host.chunks_exact_mut(inn)) {
                    dequant_row(dtype, src, 1.0, dst).unwrap();
                }
                for t in [1usize, 2, 3, 1, 7, 17, 40] {
                    let x: Vec<f32> = (0..t * inn)
                        .map(|i| ((i * 7 + 3) % 13) as f32 * 0.1 - 0.6)
                        .collect();
                    let xt = Tensor::<2>::from_data(TensorData::new(x.clone(), [t, inn]), &device);
                    let got = read_f32(linear(xt, &w), "linear").unwrap();
                    for r in 0..t {
                        for o in 0..out {
                            let e: f32 = (0..inn).map(|i| host[o * inn + i] * x[r * inn + i]).sum();
                            let g = got[r * out + o];
                            assert!(
                                (g - e).abs() <= 1e-3 * (1.0 + e.abs()),
                                "{dtype:?} [{out}, {inn}] t={t}: row {r} col {o}: gpu {g} vs host {e}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// The `[rows, cols]` values of `w` as the device multiplies by them:
    /// `x · wᵀ` for the identity `x`, transposed back on the host.
    fn read_weight(w: &Weight, rows: usize, cols: usize, device: &Device) -> Vec<f32> {
        let eye: Vec<f32> = (0..cols * cols)
            .map(|i| if i / cols == i % cols { 1.0 } else { 0.0 })
            .collect();
        let x = Tensor::<2>::from_data(TensorData::new(eye, [cols, cols]), device);
        let wt = read_f32(linear(x, w), "w").unwrap();
        let mut out = vec![0f32; rows * cols];
        for c in 0..cols {
            for r in 0..rows {
                out[r * cols + c] = wt[c * rows + r];
            }
        }
        out
    }

    /// Deterministic `[rows, cols]` values in `[-1, 1]` with a different
    /// magnitude per 32-element block.
    fn test_values(rows: usize, cols: usize) -> Vec<f32> {
        (0..rows * cols)
            .map(|i| {
                let block = (i / 32) % 5 + 1;
                (((i * 37 + i / 7) % 101) as f32 - 50.0) * 0.02 * block as f32 * 0.2
            })
            .collect()
    }

    /// The float dtypes (F32, F16, BF16) upload exactly: no requantization.
    #[test]
    fn float_dtypes_upload_exactly() {
        let Some(device) = test_device() else {
            return;
        };
        let device = open_device(device).unwrap();
        let (rows, cols) = (6usize, 64usize);
        let values = test_values(rows, cols);
        for dtype in [GgmlDType::F32, GgmlDType::F16, GgmlDType::BF16] {
            let (raw, expected): (Vec<u8>, Vec<f32>) = match dtype {
                GgmlDType::F32 => (
                    values.iter().flat_map(|v| v.to_le_bytes()).collect(),
                    values.clone(),
                ),
                GgmlDType::F16 => (
                    values
                        .iter()
                        .flat_map(|&v| half::f16::from_f32(v).to_bits().to_le_bytes())
                        .collect(),
                    values
                        .iter()
                        .map(|&v| half::f16::from_f32(v).to_f32())
                        .collect(),
                ),
                _ => (
                    values
                        .iter()
                        .flat_map(|&v| half::bf16::from_f32(v).to_bits().to_le_bytes())
                        .collect(),
                    values
                        .iter()
                        .map(|&v| half::bf16::from_f32(v).to_f32())
                        .collect(),
                ),
            };
            let w = upload_matrix("w", dtype, &raw, rows, cols, &device).unwrap();
            assert!(matches!(w.parts[0], Part::Float(_)), "{dtype:?}");
            let got = read_weight(&w, rows, cols, &device);
            for (i, (g, e)) in got.iter().zip(&expected).enumerate() {
                assert!((g - e).abs() <= 1e-6, "{dtype:?} element {i}: {g} vs {e}");
            }
        }
    }

    /// Q8_0 quantization is within half a step of the input, per block, plus
    /// the f16 rounding of the scale.
    #[test]
    fn quantize_q8_0_stays_within_half_a_step() {
        let values = test_values(4, 128);
        let mut bytes = Vec::new();
        quantize_q8_0(&values, &mut bytes);
        assert_eq!(bytes.len(), values.len() / 32 * 34);
        let mut back = vec![0f32; values.len()];
        let row_bytes = 34 * 4;
        for (src, dst) in bytes
            .chunks_exact(row_bytes)
            .zip(back.chunks_exact_mut(values.len() / 4))
        {
            dequant_row(GgmlDType::Q8_0, src, 1.0, dst).unwrap();
        }
        for (block, (x, y)) in values.chunks(32).zip(back.chunks(32)).enumerate() {
            let amax = x.iter().fold(0f32, |m, v| m.max(v.abs()));
            let bound = amax * (0.5 / 127.0 + 1.0 / 1024.0);
            for (a, b) in x.iter().zip(y) {
                assert!((a - b).abs() <= bound, "block {block}: {a} vs {b}");
            }
        }
        // An all-zero block stays zero.
        let mut zeros = Vec::new();
        quantize_q8_0(&[0.0; 32], &mut zeros);
        let mut back = [1f32; 32];
        dequant_row(GgmlDType::Q8_0, &zeros, 1.0, &mut back).unwrap();
        assert!(back.iter().all(|v| *v == 0.0));
    }

    /// Every dtype the engine requantizes to Q8_0 (the K-quants too, whose
    /// blocks are 256 wide) lands on the device as a Q8_0 weight within half
    /// a Q8_0 step, per 32-element block, of candle's own dequantization.
    #[test]
    fn requantized_upload_matches_host_dequantization() {
        use candle_core::quantized::QTensor;

        let Some(device) = test_device() else {
            return;
        };
        let device = open_device(device).unwrap();
        let (rows, cols) = (6usize, 512usize);
        let source = candle_core::Tensor::from_vec(
            test_values(rows, cols),
            (rows, cols),
            &candle_core::Device::Cpu,
        )
        .unwrap();
        for dtype in [
            GgmlDType::Q4_1,
            GgmlDType::Q5_0,
            GgmlDType::Q5_1,
            GgmlDType::Q4K,
            GgmlDType::Q5K,
            GgmlDType::Q6K,
        ] {
            let qt = QTensor::quantize(&source, dtype).unwrap();
            let expected = qt
                .dequantize(&candle_core::Device::Cpu)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let w = upload_matrix("k", dtype, &qt.data().unwrap(), rows, cols, &device).unwrap();
            let want = if dtype == GgmlDType::Q4_1 {
                mk::ExpertFormat::Q4_1
            } else {
                mk::ExpertFormat::Q8_0
            };
            assert!(
                matches!(&w.parts[0], Part::Packed(p) if p.layout.format == want),
                "{dtype:?} is held as {want:?}"
            );
            let got = read_weight(&w, rows, cols, &device);
            assert_eq!(got.len(), expected.len(), "{dtype:?}");
            for (b, (g, e)) in got.chunks(32).zip(expected.chunks(32)).enumerate() {
                let amax = e.iter().fold(0f32, |m, v| m.max(v.abs()));
                let bound = amax * (0.5 / 127.0 + 1.0 / 1024.0) + 1e-6;
                for (x, y) in g.iter().zip(e) {
                    assert!((x - y).abs() <= bound, "{dtype:?} block {b}: {x} vs {y}");
                }
            }
        }
    }

    /// The dense kernels against a host product of candle's dequantization,
    /// for every dtype a weight can be stored as (Q4_0, Q4_1, Q8_0 as they
    /// are, a K-quant requantized to Q8_0) at odd row counts (not a multiple
    /// of a thread group or of 512), Muse's 256-wide K/V projection, a real
    /// 6144-wide model dimension and a 320-wide one, with one to forty input
    /// rows: the matrix-vector kernel up to [`GEMV_ROWS`] and the
    /// dequantizing prefill path beyond it. The bound is the requantization
    /// error of a Q8_0 step per element plus f32 accumulation noise.
    #[test]
    fn dense_kernels_match_the_host_reference() {
        use candle_core::quantized::QTensor;

        let Some(device) = test_device() else {
            return;
        };
        let device = open_device(device).unwrap();
        let cases: [(GgmlDType, &[(usize, usize)]); 4] = [
            (
                GgmlDType::Q4_0,
                &[(1, 32), (97, 64), (130, 96), (256, 6144), (1030, 320)],
            ),
            (
                GgmlDType::Q4_1,
                &[(1, 32), (97, 64), (130, 96), (256, 6144), (1030, 320)],
            ),
            (
                GgmlDType::Q8_0,
                &[(1, 32), (97, 64), (130, 96), (256, 6144), (1030, 320)],
            ),
            (GgmlDType::Q6K, &[(97, 256), (600, 512), (256, 6144)]),
        ];
        for (dtype, shapes) in cases {
            for &(rows, cols) in shapes {
                let source = candle_core::Tensor::from_vec(
                    test_values(rows, cols),
                    (rows, cols),
                    &candle_core::Device::Cpu,
                )
                .unwrap();
                let qt = QTensor::quantize(&source, dtype).unwrap();
                let host = qt
                    .dequantize(&candle_core::Device::Cpu)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap();
                let w =
                    upload_matrix("w", dtype, &qt.data().unwrap(), rows, cols, &device).unwrap();
                assert!(matches!(w.parts[0], Part::Packed(_)), "{dtype:?}");
                for t in [1usize, 2, 16, 17, 40] {
                    let x: Vec<f32> = (0..t * cols)
                        .map(|i| ((i * 7 + 3) % 13) as f32 * 0.1 - 0.6)
                        .collect();
                    let xt = Tensor::<2>::from_data(TensorData::new(x.clone(), [t, cols]), &device);
                    let got = read_f32(linear(xt, &w), "linear").unwrap();
                    assert_eq!(got.len(), t * rows, "{dtype:?} [{rows}, {cols}] t={t}");
                    for r in 0..t {
                        for o in 0..rows {
                            let (mut e, mut bound) = (0f32, 1e-3f32);
                            for (hw, xv) in host[o * cols..(o + 1) * cols]
                                .chunks(32)
                                .zip(x[r * cols..(r + 1) * cols].chunks(32))
                            {
                                let amax = hw.iter().fold(0f32, |m, v| m.max(v.abs()));
                                for (a, c) in hw.iter().zip(xv) {
                                    e += a * c;
                                    bound += c.abs() * amax * (0.5 / 127.0 + 1.0 / 1024.0);
                                }
                            }
                            bound += 1e-3 * e.abs();
                            let g = got[r * rows + o];
                            assert!(
                                (g - e).abs() <= bound,
                                "{dtype:?} [{rows}, {cols}] t={t}: row {r} col {o}: gpu {g} vs host {e} (bound {bound})"
                            );
                        }
                    }
                }
            }
        }
    }

    /// A weight split into row chunks multiplies like the whole weight, for
    /// every storage class and for a last chunk that is narrow.
    #[test]
    fn chunked_weight_matches_whole_weight() {
        use candle_core::quantized::QTensor;

        let Some(device) = test_device() else {
            return;
        };
        let device = open_device(device).unwrap();
        let (rows, cols) = (200usize, 256usize);
        let source = candle_core::Tensor::from_vec(
            test_values(rows, cols),
            (rows, cols),
            &candle_core::Device::Cpu,
        )
        .unwrap();
        for dtype in [
            GgmlDType::Q4_0,
            GgmlDType::Q8_0,
            GgmlDType::F32,
            GgmlDType::Q6K,
        ] {
            let raw = QTensor::quantize(&source, dtype)
                .unwrap()
                .data()
                .unwrap()
                .to_vec();
            let whole = upload_matrix("w", dtype, &raw, rows, cols, &device).unwrap();
            assert_eq!(whole.parts.len(), 1, "{dtype:?}");
            // A limit of 64 rows of the stored layout: 64, 64, 64 and 8.
            let row_bytes = match dtype {
                GgmlDType::Q4_0 => cols / 2 + cols / 16,
                GgmlDType::F32 => cols * 4,
                _ => cols + cols / 16,
            };
            let limit = 64 * row_bytes;
            let split = upload_matrix_with("w", dtype, &raw, rows, cols, &device, limit).unwrap();
            let sizes: Vec<usize> = split.parts.iter().map(Part::rows).collect();
            assert_eq!(sizes, [64, 64, 64, 8], "{dtype:?}");
            for t in [1usize, 3] {
                let x: Vec<f32> = (0..t * cols)
                    .map(|i| ((i * 7 + 3) % 13) as f32 * 0.1 - 0.6)
                    .collect();
                let xt =
                    |d: &Device| Tensor::<2>::from_data(TensorData::new(x.clone(), [t, cols]), d);
                let a = read_f32(linear(xt(&device), &whole), "whole").unwrap();
                let b = read_f32(linear(xt(&device), &split), "split").unwrap();
                assert_eq!(a.len(), t * rows);
                assert_eq!(b.len(), t * rows);
                for (i, (p, q)) in a.iter().zip(&b).enumerate() {
                    assert!(
                        (p - q).abs() <= 1e-4 * (1.0 + p.abs()),
                        "{dtype:?} t={t} element {i}: whole {p} vs split {q}"
                    );
                }
            }
        }
    }

    /// The MoE kernels against a host evaluation of the same experts, on tiny
    /// shapes: identity slots (the resident form) and permuted slots in two
    /// passes through a pair list (the tiered form, which must agree bit for
    /// bit).
    #[test]
    fn moe_kernels_match_host_reference() {
        use super::super::moe::{Routing, route as host_route};
        use super::super::moe_kernel::{self as mk, MoeCall, SlotLayout};
        use candle_core::quantized::QTensor;

        let Some(device) = test_device() else {
            return;
        };
        let device = open_device(device).unwrap();
        let (n_embd, ff, n_expert, n_used) = (64usize, 64usize, 6usize, 2usize);
        let mut state = 12345u64;
        let mut rand = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f32 / (1u64 << 31) as f32) - 0.5
        };
        let formats = [
            [GgmlDType::Q4_0, GgmlDType::Q4_0, GgmlDType::Q8_0],
            [GgmlDType::Q4_1, GgmlDType::Q4_1, GgmlDType::Q4_1],
            [GgmlDType::Q8_0, GgmlDType::Q8_0, GgmlDType::F32],
            [GgmlDType::Q5_0, GgmlDType::F16, GgmlDType::Q4_0],
        ];
        for dtypes in formats {
            let layout = SlotLayout::new(n_embd, ff, dtypes).unwrap();
            // Raw file bytes and host dequantization of every matrix.
            let make =
                |dtype: GgmlDType, rows: usize, cols: usize, rand: &mut dyn FnMut() -> f32| {
                    let values: Vec<f32> = (0..rows * cols).map(|_| rand()).collect();
                    let raw: Vec<u8> = match dtype {
                        GgmlDType::F32 => values.iter().flat_map(|v| v.to_le_bytes()).collect(),
                        GgmlDType::F16 => values
                            .iter()
                            .flat_map(|v| half::f16::from_f32(*v).to_le_bytes())
                            .collect(),
                        _ => {
                            let t = candle_core::Tensor::from_vec(
                                values,
                                (rows, cols),
                                &candle_core::Device::Cpu,
                            )
                            .unwrap();
                            QTensor::quantize(&t, dtype)
                                .unwrap()
                                .data()
                                .unwrap()
                                .to_vec()
                        }
                    };
                    let row_bytes = cols / dtype.block_size() * dtype.type_size();
                    let mut host = vec![0f32; rows * cols];
                    for (r, row) in raw.chunks_exact(row_bytes).enumerate() {
                        dequant_row(dtype, row, 1.0, &mut host[r * cols..(r + 1) * cols]).unwrap();
                    }
                    (raw, host)
                };
            let mut experts = Vec::new();
            let mut words: Vec<Vec<u32>> = Vec::new();
            for _ in 0..n_expert {
                let g = make(dtypes[0], ff, n_embd, &mut rand);
                let u = make(dtypes[1], ff, n_embd, &mut rand);
                let d = make(dtypes[2], n_embd, ff, &mut rand);
                words.push(
                    mk::pack_expert(&layout, dtypes, &g.0, &u.0, &d.0)
                        .unwrap()
                        .to_vec(),
                );
                experts.push((g.1, u.1, d.1));
            }
            let upload = |slots: &[usize]| -> Tensor<1, Int> {
                let mut all: Vec<i32> = Vec::new();
                for &e in slots {
                    all.extend(words[e].iter().map(|w| *w as i32));
                }
                let n = all.len();
                Tensor::<1, Int>::from_data(TensorData::new(all, [n]), &device)
            };
            for (norm, scale) in [(true, 1.0f32), (false, 1.0), (true, 1.5), (false, 1.5)] {
                for t in [1usize, 3, 5] {
                    let logits: Vec<f32> = (0..t * n_expert).map(|_| rand() * 6.0).collect();
                    let x: Vec<f32> = (0..t * n_embd).map(|_| rand()).collect();
                    let spec = MoeSpec {
                        n_expert,
                        n_used,
                        norm_weights: norm,
                        scale,
                    };
                    let routing: Routing = host_route(&logits, t, &spec);
                    // Host reference.
                    let mut want = vec![0f32; t * n_embd];
                    for (e, picks) in routing.per_expert.iter().enumerate() {
                        for &(tok, w) in picks {
                            let xr = &x[tok as usize * n_embd..(tok as usize + 1) * n_embd];
                            let dot =
                                |m: &[f32], rows: usize, cols: usize, v: &[f32]| -> Vec<f32> {
                                    (0..rows)
                                        .map(|r| (0..cols).map(|c| m[r * cols + c] * v[c]).sum())
                                        .collect()
                                };
                            let g = dot(&experts[e].0, ff, n_embd, xr);
                            let u = dot(&experts[e].1, ff, n_embd, xr);
                            let h: Vec<f32> = g
                                .iter()
                                .zip(&u)
                                .map(|(g, u)| g / (1.0 + (-g).exp()) * u)
                                .collect();
                            let y = dot(&experts[e].2, n_embd, ff, &h);
                            for (o, y) in want[tok as usize * n_embd..(tok as usize + 1) * n_embd]
                                .iter_mut()
                                .zip(&y)
                            {
                                *o += w * y;
                            }
                        }
                    }
                    let call = |pairs: usize, use_list: bool| MoeCall {
                        tokens: t,
                        n_used,
                        n_expert,
                        pairs,
                        use_list,
                        act: 0,
                        scale: spec.scale,
                        slot: layout,
                    };
                    let xt =
                        || Tensor::<2>::from_data(TensorData::new(x.clone(), [t, n_embd]), &device);
                    let logits_t = Tensor::<2>::from_data(
                        TensorData::new(logits.clone(), [t, n_expert]),
                        &device,
                    );
                    let dummy = Tensor::<1, Int>::zeros([2], &device);
                    let route = mk::route(logits_t, &call(t * n_used, false), norm);
                    let route_host = read_f32(route.clone(), "route").unwrap();
                    // Routing: every token's experts and weights.
                    for tok in 0..t {
                        let mut got: Vec<(usize, f32)> = (0..n_used)
                            .map(|j| {
                                (
                                    route_host[tok * n_used + j] as usize,
                                    route_host[t * n_used + tok * n_used + j],
                                )
                            })
                            .collect();
                        got.sort_by_key(|p| p.0);
                        let mut exp: Vec<(usize, f32)> = routing
                            .per_expert
                            .iter()
                            .enumerate()
                            .flat_map(|(e, p)| {
                                p.iter()
                                    .filter(|q| q.0 as usize == tok)
                                    .map(move |q| (e, q.1))
                            })
                            .collect();
                        exp.sort_by_key(|p| p.0);
                        assert_eq!(got.len(), exp.len());
                        for (g, e) in got.iter().zip(&exp) {
                            assert_eq!(g.0, e.0, "{dtypes:?} t={t} token {tok} experts");
                            assert!((g.1 - e.1).abs() <= 1e-5, "weight {} vs {}", g.1, e.1);
                        }
                    }
                    // Resident form.
                    let pool = upload(&(0..n_expert).collect::<Vec<_>>());
                    let c = call(t * n_used, false);
                    let hidden = mk::up(xt(), route.clone(), dummy.clone(), pool.clone(), &c);
                    let scratch = Tensor::<2>::zeros([t * n_used, n_embd], &device);
                    let scratch = mk::down(hidden, route.clone(), dummy.clone(), pool, scratch, &c);
                    let resident = read_f32(mk::sum(scratch, &c), "out").unwrap();
                    // A dtype the kernels hold requantized to Q8_0 (Q5_0 here) is
                    // off by the requantization step.
                    let tol = if dtypes[0] == GgmlDType::Q5_0 {
                        4e-2
                    } else {
                        2e-3
                    };
                    for (i, (g, e)) in resident.iter().zip(&want).enumerate() {
                        assert!(
                            (g - e).abs() <= tol * (1.0 + e.abs()),
                            "{dtypes:?} norm={norm} t={t} element {i}: gpu {g} vs host {e}"
                        );
                    }
                    // Tiered form: permuted slots, two passes over a pair list.
                    let perm: Vec<usize> = (0..n_expert).rev().collect();
                    let pool = upload(&perm);
                    let slot_of = |e: usize| perm.iter().position(|&p| p == e).unwrap();
                    let mut scratch = Tensor::<2>::zeros([t * n_used, n_embd], &device);
                    for half in 0..2usize {
                        let mut list: Vec<i32> = Vec::new();
                        for (pair, expert) in route_host[..t * n_used].iter().enumerate() {
                            if pair % 2 == half {
                                list.push(pair as i32);
                                list.push(slot_of(*expert as usize) as i32);
                            }
                        }
                        let p = list.len() / 2;
                        if p == 0 {
                            continue;
                        }
                        let list =
                            Tensor::<1, Int>::from_data(TensorData::new(list, [2 * p]), &device);
                        let c = call(p, true);
                        let hidden = mk::up(xt(), route.clone(), list.clone(), pool.clone(), &c);
                        scratch = mk::down(hidden, route.clone(), list, pool.clone(), scratch, &c);
                    }
                    let tiered =
                        read_f32(mk::sum(scratch, &call(t * n_used, false)), "out").unwrap();
                    assert_eq!(
                        tiered, resident,
                        "{dtypes:?} norm={norm} t={t}: passes differ"
                    );
                }
            }
        }
    }

    /// The hidden rows `forward_ext` returns (the final-norm output the head
    /// reads) match the CPU's, for the last row and for every row.
    #[test]
    fn forward_ext_hidden_rows_match_cpu() {
        let Some(device) = test_device() else {
            return;
        };
        let model = write_tiny_model_with(TinyOptions::default());
        let (mut cpu, mut gpu) = load_both(&model, device);
        let n = gpu.n_embd();
        for (tokens, start, want, rows) in [
            (&PROMPT[..4], 0, Want::Last, 1),
            (&PROMPT[4..9], 4, Want::All, 5),
            (&PROMPT[9..], 9, Want::Last, 1),
        ] {
            let c = cpu
                .forward_ext(tokens, start, want, Hidden::PostNorm, false)
                .unwrap();
            let g = gpu
                .forward_ext(tokens, start, want, Hidden::PostNorm, false)
                .unwrap();
            let (ch, gh) = (c.hidden.unwrap(), g.hidden.unwrap());
            assert_eq!(gh.len(), rows * n);
            assert_matches_cpu(&ch, &gh, LOOSE, "hidden rows");
            assert_matches_cpu(&c.logits.unwrap(), &g.logits.unwrap(), LOOSE, "logits");
        }
        let none = gpu
            .forward_ext(&[7], PROMPT.len(), Want::None, Hidden::PostNorm, false)
            .unwrap();
        assert!(none.logits.is_none() && none.hidden.is_none());
        assert_eq!(gpu.vocab(), cpu.vocab());
        assert_eq!(gpu.arch(), cpu.arch());
        assert_eq!(gpu.n_embd(), cpu.n_embd());
        let (c, g) = (cpu.embedding_row(5).unwrap(), gpu.embedding_row(5).unwrap());
        assert_eq!(
            c, g,
            "embedding rows are read from the same file the same way"
        );
    }

    /// The GPU drafter's logits match the CPU drafter's step by step, at
    /// cache lengths shorter and longer than the sliding window, with the
    /// chosen tokens fixed so the two chains cannot diverge. The CPU drafter
    /// holds its weights as Q8_0 and multiplies them by Q8_0-rounded
    /// activations, so the comparison is within a few percent of the largest
    /// logit; a drafter at the wrong position is measurably further away.
    #[test]
    fn gpu_drafter_matches_cpu_drafter() {
        use super::super::mtp::Drafter;
        use super::super::tiny::write_tiny_drafter;

        let Some(device) = test_device() else {
            return;
        };
        let model = write_tiny_model_with(TinyOptions::default());
        let drafter = write_tiny_drafter();
        let content = content_of(&drafter);
        let schedule = [5u32, 9, 2, 7];
        let k = schedule.len();

        // A `pick` that records every step's logits and answers from
        // `schedule`.
        fn recorder<'a>(
            picked: &'a mut Vec<Vec<f32>>,
            schedule: &'a [u32],
        ) -> impl FnMut(&mut [f32]) -> Option<u32> + 'a {
            move |logits| {
                picked.push(logits.to_vec());
                Some(schedule.get(picked.len() - 1).copied().unwrap_or(0))
            }
        }

        // The largest logit error of the GPU drafter at `at` against the CPU
        // drafter at `len`, relative to the largest logit.
        let worst = |c: &[Vec<f32>], g: &[Vec<f32>]| -> f32 {
            let scale = c.iter().flatten().fold(1f32, |m, v| m.max(v.abs()));
            c.iter()
                .zip(g)
                .flat_map(|(a, b)| a.iter().zip(b))
                .fold(0f32, |m, (x, y)| m.max((x - y).abs()))
                / scale
        };
        for len in [3usize, PROMPT.len()] {
            let (mut cpu, mut gpu) = load_both(&model, device);
            let c = cpu
                .forward_ext(&PROMPT[..len], 0, Want::Last, Hidden::PostNorm, false)
                .unwrap();
            let g = gpu
                .forward_ext(&PROMPT[..len], 0, Want::Last, Hidden::PostNorm, false)
                .unwrap();
            let (c_hidden, g_hidden) = (c.hidden.unwrap(), g.hidden.unwrap());
            let c_tables = cpu.solo_tables().to_vec();
            let cpu = super::super::Model::Cpu(cpu);
            let cpu_drafter = Drafter::open(&content, &drafter.path, &cpu, 32).unwrap();
            let gpu_drafter = GpuDrafter::open(&content, &drafter.path, &gpu, 32).unwrap();

            let mut c_logits = Vec::new();
            let c_tokens = cpu_drafter
                .draft(
                    &cpu,
                    &c_tables,
                    &c_hidden,
                    6,
                    len,
                    k,
                    &mut recorder(&mut c_logits, &schedule),
                )
                .unwrap();
            let mut g_logits = Vec::new();
            let g_tokens = gpu_drafter
                .draft(
                    &gpu,
                    gpu.solo_tables(),
                    &g_hidden,
                    6,
                    len,
                    k,
                    &mut recorder(&mut g_logits, &schedule),
                )
                .unwrap();
            assert_eq!(c_tokens, schedule);
            assert_eq!(g_tokens, schedule);
            assert_eq!(g_logits.len(), k);
            let right = worst(&c_logits, &g_logits);
            println!("drafter at len {len}: max error {right:.4} of the largest logit");
            assert!(right <= 0.06, "len {len}: {right}");

            // The same chain one position earlier is a different function.
            let mut early = Vec::new();
            gpu_drafter
                .draft(
                    &gpu,
                    gpu.solo_tables(),
                    &g_hidden,
                    6,
                    len - 1,
                    k,
                    &mut recorder(&mut early, &schedule),
                )
                .unwrap();
            let wrong = worst(&c_logits, &early);
            println!("drafter at len {}: max error {wrong:.4}", len - 1);
            assert!(wrong > 2.0 * right, "len {len}: {wrong} vs {right}");
        }
    }

    /// A drafter must pair with a Gemma 4 target of its own width and
    /// refuses a model that is not a drafter, a drafter used as a model and
    /// a position the target has not cached.
    #[test]
    fn gpu_drafter_validates_its_inputs() {
        use super::super::tiny::write_tiny_drafter;

        let Some(device) = test_device() else {
            return;
        };
        let model = write_tiny_model_with(TinyOptions::default());
        let drafter = write_tiny_drafter();
        let (_, mut gpu) = load_both(&model, device);
        let (model_content, drafter_content) = (content_of(&model), content_of(&drafter));

        let e = GpuDrafter::open(&model_content, &model.path, &gpu, 32)
            .err()
            .unwrap();
        assert!(e.to_string().contains("not a drafter"), "{e}");
        let e = GpuTransformer::load(&drafter_content, &drafter.path, 32, device)
            .err()
            .unwrap();
        assert!(e.to_string().contains("drafter"), "{e}");

        let net = GpuDrafter::open(&drafter_content, &drafter.path, &gpu, 32).unwrap();
        let mut pick = |_: &mut [f32]| Some(0u32);
        let hidden = vec![0.1f32; gpu.n_embd()];
        gpu.forward(&PROMPT[..5], 0, Want::None).unwrap();
        let tables = gpu.solo_tables();
        // The table holds one block: position 17 needs a second.
        assert!(
            net.draft(&gpu, tables, &hidden, 1, 17, 2, &mut pick)
                .is_err(),
            "beyond the table"
        );
        assert!(
            net.draft(&gpu, tables, &hidden, 1, 0, 2, &mut pick)
                .is_err(),
            "empty cache"
        );
        assert!(
            net.draft(&gpu, &[], &hidden, 1, 5, 2, &mut pick).is_err(),
            "no block tables"
        );
        assert!(
            net.draft(&gpu, tables, &hidden[1..], 1, 5, 2, &mut pick)
                .is_err(),
            "hidden width"
        );
        assert_eq!(
            net.draft(&gpu, tables, &hidden, 1, 5, 2, &mut pick)
                .unwrap()
                .len(),
            2
        );
    }

    /// A Qwen3-30B-A3B-shaped MoE layer (128 experts, top 8, Q4_0 gate and
    /// up `[768, 2048]`, Q8_0 down `[2048, 768]`) over a pass whose experts
    /// each get one row, after longer and single-token passes have tuned the
    /// kernels: every row equals the same row's single-token pass. Gathering
    /// the one row and scaling it by a `[1, 1]` device tensor loses the
    /// device on this sequence.
    #[test]
    fn gpu_moe_one_row_experts_match_single_tokens() {
        let Some(device) = test_device() else {
            return;
        };
        let device = open_device(device).unwrap();
        let raw_of = |dtype: GgmlDType, out: usize, inn: usize, seed: usize| {
            let mut raw = Vec::new();
            if dtype == GgmlDType::F32 {
                for i in 0..out * inn {
                    let v = (((i + seed) * 13 + 1) % 17) as f32 * 0.002 - 0.015;
                    raw.extend(v.to_le_bytes());
                }
            } else {
                for b in 0..out * inn / 32 {
                    raw.extend(f16_bytes(0.005 + ((b + seed) % 7) as f32 * 0.001));
                    raw.extend(
                        (0..dtype.type_size() - 2)
                            .map(|j| (((b + seed) * 31 + j * 17 + 5) % 251) as u8),
                    );
                }
            }
            raw
        };
        let upload = |dtype, out, inn, seed| {
            upload_matrix(
                "w",
                dtype,
                &raw_of(dtype, out, inn, seed),
                out,
                inn,
                &device,
            )
            .unwrap()
        };
        let (n_embd, ff, n_expert) = (2048usize, 768usize, 128usize);
        let dtypes = [GgmlDType::Q4_0, GgmlDType::Q4_0, GgmlDType::Q8_0];
        let layout = SlotLayout::new(n_embd, ff, dtypes).unwrap();
        let sets: Vec<Vec<i32>> = (0..8)
            .map(|s| {
                let words = mk::pack_expert(
                    &layout,
                    dtypes,
                    &raw_of(dtypes[0], ff, n_embd, s),
                    &raw_of(dtypes[1], ff, n_embd, s + 50),
                    &raw_of(dtypes[2], n_embd, ff, s + 90),
                )
                .unwrap();
                words.iter().map(|w| *w as i32).collect()
            })
            .collect();
        let spec = MoeSpec {
            n_expert,
            n_used: 8,
            norm_weights: true,
            scale: 1.0,
        };
        let router = upload(GgmlDType::F32, n_expert, n_embd, 3);
        let router_parts = router.parts.clone();
        let all: Vec<i32> = (0..n_expert)
            .flat_map(|e| sets[e % sets.len()].iter().copied())
            .collect();
        let n = all.len();
        let layer = FfnWeights::Moe {
            spec,
            router,
            experts: GpuExpertBank::Resident {
                pool: Tensor::<1, Int>::from_data(TensorData::new(all, [n]), &device),
                no_list: Tensor::<1, Int>::zeros([2], &device),
                layout,
            },
            shared: None,
        };
        let input = |t: usize, step: usize| -> Vec<f32> {
            (0..t * n_embd)
                .map(|i| (((i + step * 11) * 7 + 3) % 13) as f32 * 0.1 - 0.6)
                .collect()
        };
        let run = |x: &[f32], t: usize| -> Vec<f32> {
            let x = Tensor::<2>::from_data(TensorData::new(x.to_vec(), [t, n_embd]), &device);
            read_f32(layer.forward(x, Act::Silu, &device).unwrap(), "ffn").unwrap()
        };
        // A longer pass and two single-token passes tune the kernels first.
        for (step, t) in [(0, 30), (1, 1), (2, 1)] {
            assert!(run(&input(t, step), t).iter().all(|v| v.is_finite()));
        }
        let x = input(4, 3);
        let xt = Tensor::<2>::from_data(TensorData::new(x.clone(), [4, n_embd]), &device);
        let logits = read_f32(
            linear(
                xt,
                &Weight {
                    parts: router_parts,
                },
            ),
            "router",
        )
        .unwrap();
        assert!(
            super::super::moe::route(&logits, 4, &spec)
                .per_expert
                .iter()
                .any(|p| p.len() == 1),
            "the pass must route single rows to experts"
        );
        let rows = run(&x, 4);
        for (r, (got, x)) in rows
            .chunks_exact(n_embd)
            .zip(x.chunks_exact(n_embd))
            .enumerate()
        {
            let want = run(x, 1);
            for (g, w) in got.iter().zip(&want) {
                assert!(
                    (g - w).abs() <= 1e-4 * (1.0 + w.abs()),
                    "row {r}: {g} vs {w}"
                );
            }
        }
    }

    // ------------------------------------------------------------------
    // Tensor parallelism (see `tp`)
    // ------------------------------------------------------------------

    /// A `rows x cols` matrix of test values encoded as `dtype` (Q8_0, F32 or
    /// F16): its GGUF bytes and the values they decode to.
    fn encoded_matrix(dtype: GgmlDType, rows: usize, cols: usize) -> (Vec<u8>, Vec<f32>) {
        let values = test_values(rows, cols);
        let mut raw = Vec::new();
        match dtype {
            GgmlDType::Q8_0 => quantize_q8_0(&values, &mut raw),
            GgmlDType::F32 => raw.extend(values.iter().flat_map(|v| v.to_le_bytes())),
            GgmlDType::F16 => raw.extend(values.iter().flat_map(|v| f16_bytes(*v))),
            other => panic!("{other:?}"),
        }
        let decoded = decode_rows(dtype, &raw, cols);
        (raw, decoded)
    }

    /// The values of `raw`, rows of `width` elements stored as `dtype`.
    fn decode_rows(dtype: GgmlDType, raw: &[u8], width: usize) -> Vec<f32> {
        let row_bytes = width / dtype.block_size() * dtype.type_size();
        let mut out = vec![0f32; raw.len() / row_bytes * width];
        for (src, dst) in raw.chunks_exact(row_bytes).zip(out.chunks_exact_mut(width)) {
            dequant_row(dtype, src, 1.0, dst).unwrap();
        }
        out
    }

    /// The columns `cols` of every row of the row-major `values`.
    fn host_columns(values: &[f32], width: usize, cols: &Range<usize>) -> Vec<f32> {
        values
            .chunks(width)
            .flat_map(|row| row[cols.clone()].iter().copied())
            .collect()
    }

    /// A column cut at block bounds is a byte copy of the rows' sub-slices and
    /// decodes to the same columns of the whole matrix (Q8_0, F32, F16), a
    /// float cut may start anywhere, and a cut inside a Q8_0 block is not a
    /// byte copy. Host only.
    #[test]
    fn column_cuts_match_the_host_reference() {
        let (rows, width) = (5usize, 128usize);
        for dtype in [GgmlDType::Q8_0, GgmlDType::F32, GgmlDType::F16] {
            let (raw, whole) = encoded_matrix(dtype, rows, width);
            for cols in [0..128, 0..64, 64..128, 32..96, 96..128, 0..32] {
                let held = tp::column_dtype(dtype, width, &cols);
                assert_eq!(held, dtype, "{dtype:?} {cols:?} is a byte copy");
                let mut out = Vec::new();
                append_columns(dtype, &raw, width, &cols, held, &mut out).unwrap();
                assert_eq!(
                    decode_rows(held, &out, cols.len()),
                    host_columns(&whole, width, &cols),
                    "{dtype:?} {cols:?}"
                );
            }
        }
        for dtype in [GgmlDType::F32, GgmlDType::F16] {
            let (raw, whole) = encoded_matrix(dtype, rows, width);
            let cols = 5..77;
            assert_eq!(tp::column_dtype(dtype, width, &cols), dtype);
            let mut out = Vec::new();
            append_columns(dtype, &raw, width, &cols, dtype, &mut out).unwrap();
            assert_eq!(
                decode_rows(dtype, &out, cols.len()),
                host_columns(&whole, width, &cols)
            );
        }
        // Inside a Q8_0 block the cut is requantized (never a byte copy).
        let (raw, whole) = encoded_matrix(GgmlDType::Q8_0, rows, width);
        let cols = 16..80;
        assert!(tp::copy_columns(GgmlDType::Q8_0, &raw, width, &cols).is_none());
        let mut out = Vec::new();
        append_columns(
            GgmlDType::Q8_0,
            &raw,
            width,
            &cols,
            GgmlDType::Q8_0,
            &mut out,
        )
        .unwrap();
        let kept = host_columns(&whole, width, &cols);
        for (g, w) in decode_rows(GgmlDType::Q8_0, &out, cols.len())
            .chunks(32)
            .zip(kept.chunks(32))
        {
            let amax = w.iter().fold(0f32, |m, v| m.max(v.abs()));
            let bound = amax * (0.5 / 127.0 + 1.0 / 1024.0) + 1e-6;
            assert!(g.iter().zip(w).all(|(a, b)| (a - b).abs() <= bound));
        }
        // A cut that runs past the row, or bytes that are not whole rows.
        assert!(
            append_columns(
                GgmlDType::Q8_0,
                &raw,
                width,
                &(64..160),
                GgmlDType::Q8_0,
                &mut out
            )
            .is_err()
        );
        assert!(
            append_columns(
                GgmlDType::Q8_0,
                &raw[..35],
                width,
                &(0..64),
                GgmlDType::Q8_0,
                &mut out
            )
            .is_err()
        );
    }

    /// A column cut that splits a 256-element K-quant block dequantizes the
    /// rows and cuts the values, as Q8_0; a cut on block bounds stays a byte
    /// copy. Host only.
    #[test]
    fn column_cuts_inside_a_k_quant_block_requantize() {
        use candle_core::quantized::QTensor;

        let (rows, width) = (3usize, 512usize);
        let source = candle_core::Tensor::from_vec(
            test_values(rows, width),
            (rows, width),
            &candle_core::Device::Cpu,
        )
        .unwrap();
        let qt = QTensor::quantize(&source, GgmlDType::Q6K).unwrap();
        let raw = qt.data().unwrap();
        let whole = decode_rows(GgmlDType::Q6K, &raw, width);

        let aligned = 256..512;
        assert_eq!(
            tp::column_dtype(GgmlDType::Q6K, width, &aligned),
            GgmlDType::Q6K
        );
        let mut copy = Vec::new();
        append_columns(
            GgmlDType::Q6K,
            &raw,
            width,
            &aligned,
            GgmlDType::Q6K,
            &mut copy,
        )
        .unwrap();
        assert_eq!(
            decode_rows(GgmlDType::Q6K, &copy, aligned.len()),
            host_columns(&whole, width, &aligned)
        );

        let cols = 32..288;
        assert_eq!(
            tp::column_dtype(GgmlDType::Q6K, width, &cols),
            GgmlDType::Q8_0
        );
        let kept = host_columns(&whole, width, &cols);
        let mut q8 = Vec::new();
        append_columns(GgmlDType::Q6K, &raw, width, &cols, GgmlDType::Q8_0, &mut q8).unwrap();
        let mut want = Vec::new();
        quantize_q8_0(&kept, &mut want);
        assert_eq!(q8, want);
    }

    /// The GPU devices under test, as `test_devices` plus a repeat of the
    /// first when the list names one only (two shards on one device are
    /// still split and reduced).
    fn two_test_devices() -> Option<Vec<LocalDevice>> {
        let mut devices = test_devices();
        let first = *devices.first()?;
        if devices.len() < 2 {
            devices.push(first);
        }
        Some(devices)
    }

    /// `got` agrees with `want` to `tol` of the vector's scale
    /// (`max(1, max |want|)`).
    fn assert_rows_within(got: &[f32], want: &[f32], tol: f32, what: &str) {
        assert_eq!(got.len(), want.len(), "{what}");
        let scale = want.iter().fold(1.0f32, |m, v| m.max(v.abs()));
        for (i, (a, b)) in got.iter().zip(want).enumerate() {
            assert!(
                (a - b).abs() <= tol * scale,
                "{what}: value {i}: {a} vs {b} (scale {scale}, tolerance {tol})"
            );
        }
    }

    /// A model the step runners drive: the GPU engine or an offloaded one.
    trait Stepped {
        fn layout(&self) -> &KvLayout;
        fn forward(&mut self, batch: &Batch<'_>) -> Vec<Forward>;
    }

    impl Stepped for GpuTransformer {
        fn layout(&self) -> &KvLayout {
            self.kv_layout()
        }

        fn forward(&mut self, batch: &Batch<'_>) -> Vec<Forward> {
            self.forward_batch(batch).unwrap()
        }
    }

    impl Stepped for SplitModel {
        fn layout(&self) -> &KvLayout {
            self.kv_layout()
        }

        fn forward(&mut self, batch: &Batch<'_>) -> Vec<Forward> {
            self.forward_batch(batch).unwrap()
        }
    }

    /// Run `steps` (the rows of each step, over `tokens` per sequence)
    /// through the batched API of `m` over fresh block pools: the output of
    /// every sequence's call, in order.
    fn run_steps(m: &mut impl Stepped, steps: &[Vec<Step>], tokens: &[Vec<u32>]) -> Vec<Forward> {
        let mut pools = KvPools::new(m.layout());
        let mut seqs: Vec<SeqBlocks> = tokens.iter().map(|_| pools.seq()).collect();
        let mut at = vec![0usize; tokens.len()];
        let mut out = Vec::new();
        for rows_of_step in steps {
            for st in rows_of_step {
                pools.reserve(&mut seqs[st.seq], at[st.seq] + st.n).unwrap();
            }
            let rows: Vec<SeqRows<'_>> = rows_of_step
                .iter()
                .map(|st| SeqRows {
                    tokens: &tokens[st.seq][at[st.seq]..at[st.seq] + st.n],
                    start_pos: at[st.seq],
                    tables: &seqs[st.seq].tables,
                    state: StateSlot(st.seq),
                    want: st.want,
                    hidden: st.hidden,
                })
                .collect();
            let outs = m.forward(&Batch { rows, exact: false });
            assert_eq!(outs.len(), rows_of_step.len());
            for (st, o) in rows_of_step.iter().zip(outs) {
                out.push(o);
                at[st.seq] += st.n;
            }
        }
        out
    }

    /// Layer offload: every split point of `model` (`n` layers on the GPU,
    /// the rest on the CPU, held dequantized so that the two halves compute
    /// the same values) gives the logits and hidden rows of the GPU-only
    /// load to 1e-4 of their scale, through prefill, chunked prefill,
    /// single-row decode, stacked decode, every-position logits and hidden
    /// rows, two sequences at once.
    fn check_offload(model: &TinyModel, device: LocalDevice, label: &str) {
        const CTX: usize = 32;
        let content = content_of(model);
        let sizing = Sizing {
            max_sequences: 2,
            kv_tokens: 2 * CTX,
            max_rows: 16,
        };
        let steps = sharded_steps();
        let tokens: Vec<Vec<u32>> = (0..2).map(|s| stream(s, 24)).collect();
        let mut reference = GpuTransformer::load_placed(
            &content,
            &model.path,
            CTX,
            device,
            &[],
            &ExpertPlacement::Resident,
            &sizing,
        )
        .unwrap();
        let want = run_steps(&mut reference, &steps, &tokens);
        let n_layer = read_hparams(&content).unwrap().n_layer;
        assert!(n_layer >= 2, "{label}: a model with several layers");
        for gpu_layers in 1..n_layer {
            let first = n_layer - gpu_layers;
            let mut split = SplitModel::load_with(
                &content,
                &model.path,
                CTX,
                device,
                &ExpertPlacement::Resident,
                &sizing,
                first,
                true,
            )
            .unwrap();
            assert_eq!(split.gpu_layers(), gpu_layers, "{label}");
            let got = run_steps(&mut split, &steps, &tokens);
            assert_eq!(got.len(), want.len(), "{label}");
            for (call, (g, w)) in got.iter().zip(&want).enumerate() {
                let at =
                    format!("{label}: {gpu_layers} of {n_layer} layers on the GPU, call {call}");
                assert_forward_within(g, w, 1e-4, &at);
            }
        }
    }

    /// The steps of a sharded run: one sequence alone (a prefill, a chunk
    /// with every logits row and every hidden row, decode steps), a second
    /// prefilling alone, both decoding together (the stacked decode launch),
    /// a prefill chunk next to a decoding sequence, hidden rows only, one
    /// logits row.
    fn sharded_steps() -> Vec<Vec<Step>> {
        let (last, all, none) = (Want::Last, Want::All, Want::None);
        let (plain, post, post_all) = (Hidden::None, Hidden::PostNorm, Hidden::PostNormAll);
        vec![
            vec![step(0, 5, last, plain)],
            vec![step(0, 3, all, post_all)],
            vec![step(0, 1, last, post)],
            vec![step(0, 1, last, plain)],
            vec![step(1, 9, last, plain)],
            vec![step(0, 1, last, post), step(1, 1, last, plain)],
            vec![step(0, 1, last, plain), step(1, 1, last, plain)],
            vec![step(0, 4, none, post_all), step(1, 1, last, plain)],
            vec![step(0, 1, last, plain), step(1, 2, all, post)],
        ]
    }

    /// Layer offload gives the single-device logits at every split point of
    /// every architecture of the table: the dense, windowed, MoE, Gemma 4
    /// (its shared-K/V layers read a cache the CPU half computes, at the
    /// split points that cross it), delta-net and latent-attention models.
    #[test]
    fn gpu_offload_matches_single_device_at_every_split() {
        let Some(device) = test_device() else {
            return;
        };
        for spec in TinySpec::all() {
            check_offload(&write_spec(&spec), device, spec.arch);
        }
        check_offload(
            &write_tiny_model_with(TinyOptions::default()),
            device,
            "gemma4",
        );
        for (label, spec) in [("qwen35", tiny_qwen35()), ("qwen35moe", tiny_qwen35moe())] {
            check_offload(&write_qwen35(&spec), device, label);
        }
        for (label, spec) in mistral4_variants() {
            check_offload(&write_mistral4(&spec), device, label);
        }
    }

    /// A mirror write the GPU half rejects (rows of the wrong size) fails the
    /// call and leaves every pool in place: the next call, with the right
    /// rows, serves and equals the single-device logits.
    #[test]
    fn gpu_offload_serves_again_after_a_rejected_mirror_write() {
        use super::super::cpu::Front;
        use super::super::split::mirrored;

        let Some(device) = test_device() else {
            return;
        };
        const CTX: usize = 32;
        let model = write_tiny_model_with(TinyOptions::default());
        let content = content_of(&model);
        let hp = read_hparams(&content).unwrap();
        let first = (1..hp.n_layer)
            .find(|&f| !mirrored(&hp, f).is_empty())
            .expect("a split crossing a shared cache");
        let mirror = mirrored(&hp, first);
        let sizing = Sizing::single(CTX);
        let mut front = Transformer::load_front(
            &content,
            &model.path,
            CTX,
            &ExpertPlacement::Resident,
            &sizing,
            &Front {
                layers: first,
                shared: None,
                export: mirror.clone(),
            },
            true,
        )
        .unwrap();
        let mut back = GpuTransformer::load_back(
            &content,
            &model.path,
            CTX,
            device,
            &ExpertPlacement::Resident,
            &sizing,
            &Back {
                first,
                shared: None,
                mirror,
            },
        )
        .unwrap();
        let mut reference = GpuTransformer::load(&content, &model.path, CTX, device).unwrap();
        let want = reference
            .forward(&PROMPT[..6], 0, Want::Last)
            .unwrap()
            .unwrap();

        let mut pools = KvPools::new(back.kv_layout());
        let mut seq = pools.seq();
        pools.reserve(&mut seq, 6).unwrap();
        let batch = Batch {
            rows: vec![SeqRows {
                tokens: &PROMPT[..6],
                start_pos: 0,
                tables: &seq.tables,
                state: StateSlot(0),
                want: Want::Last,
                hidden: Hidden::None,
            }],
            exact: false,
        };
        let mut out = front.forward_front(&batch).unwrap();
        out.kv[0].seqs[0].0.pop();
        let failed = back.forward_back(&batch, out.hidden, out.per_layer, &out.kv);
        assert!(failed.is_err(), "rows of the wrong size are rejected");
        let out = front.forward_front(&batch).unwrap();
        let mut served = back
            .forward_back(&batch, out.hidden, out.per_layer, &out.kv)
            .unwrap();
        let got = served.swap_remove(0).logits.unwrap();
        assert_rows_within(&got, &want, 1e-4, "after a rejected mirror write");
    }

    /// `model` split across `devices[0]` and `devices[1]` equals the same
    /// model on `devices[0]` alone (to `tol` of the logits' scale: the
    /// shards sum two half-width partial products where the single device's
    /// matmul sums the whole inner dimension, so only the reduction order
    /// differs) and two sharded loads agree bit for bit (the host
    /// reduction adds in a fixed shard order). Also: the shards' pools are at
    /// least the single device's, and the split model reports its world.
    fn check_sharded(model: &TinyModel, devices: &[LocalDevice], tol: f32, label: &str) {
        const CTX: usize = 32;
        let content = content_of(model);
        let sizing = Sizing {
            max_sequences: 2,
            kv_tokens: 2 * CTX,
            max_rows: 16,
        };
        let load = |shards: &[LocalDevice]| {
            GpuTransformer::load_placed(
                &content,
                &model.path,
                CTX,
                devices[0],
                shards,
                &ExpertPlacement::Resident,
                &sizing,
            )
            .unwrap()
        };
        let steps = sharded_steps();
        let tokens: Vec<Vec<u32>> = (0..2).map(|s| stream(s, 20)).collect();

        let mut single = load(&[]);
        assert_eq!(single.world(), 1, "{label}");
        let want = run_steps(&mut single, &steps, &tokens);
        let single_bytes = single.kv_pool_bytes();
        let layout = single.kv_layout().clone();
        drop(single);

        let mut split = load(&devices[1..2]);
        assert_eq!(split.world(), 2, "{label}");
        assert_eq!(split.kv_layout(), &layout, "{label}: same block layout");
        assert!(
            split.kv_pool_bytes() >= single_bytes,
            "{label}: the shards' pools hold at least the single device's"
        );
        let got = run_steps(&mut split, &steps, &tokens);
        drop(split);
        assert_eq!(got.len(), want.len(), "{label}");
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert_forward_within(g, w, tol, &format!("{label} call {i}"));
        }

        // Determinism: the first sharded load tuned burn's kernels, so the
        // next two run the same kernels on the same data.
        let a = run_steps(&mut load(&devices[1..2]), &steps, &tokens);
        let b = run_steps(&mut load(&devices[1..2]), &steps, &tokens);
        for (i, (x, y)) in a.iter().zip(&b).enumerate() {
            assert!(
                x.logits == y.logits && x.hidden == y.hidden,
                "{label}: call {i} differs between two sharded loads"
            );
        }
    }

    /// A model split across two devices equals the model on one: the tiny
    /// Gemma 4 (per-layer embeddings, shared K/V, sliding windows; its
    /// single KV head is replicated on both shards, the GQA variant divides
    /// its two) and every `TinySpec` the split rules allow (OLMo 2 and OLMoE
    /// have full-width Q/K norms, which a shard cannot compute alone). The
    /// dense specs' FFN width 96 is widened to 128, since a shard's FFN
    /// slice must be a multiple of 32 (a property of the tiny dimensions,
    /// not of the rule); the MoE specs' experts (64 wide) split as they are.
    /// With one device in `CUCA_LOCAL_DEVICE`, both shards live on it.
    #[test]
    fn gpu_sharded_matches_single_device() {
        let Some(devices) = two_test_devices() else {
            println!("SKIP: set CUCA_LOCAL_DEVICE to a GPU device (`a+b` for two)");
            return;
        };
        for (label, options) in [
            ("gemma4 mqa", TinyOptions::default()),
            (
                "gemma4 gqa",
                TinyOptions {
                    n_head: 4,
                    n_kv_head: 2,
                    ..TinyOptions::default()
                },
            ),
        ] {
            check_sharded(&write_tiny_model_with(options), &devices, 1e-4, label);
        }
        let mut ran = Vec::new();
        for spec in TinySpec::all() {
            let arch = spec.arch;
            if spec.qk_norm_full {
                println!("SKIP {arch}: tensor parallelism rejects a full-width Q/K norm");
                continue;
            }
            let moe = spec.moe.is_some();
            let ff = if moe {
                spec.ff
            } else {
                spec.ff.div_ceil(64) * 64
            };
            let model = write_spec(&TinySpec { ff, ..spec });
            // Routing near-ties and the weighted expert sums add rounding
            // steps the dense models do not have.
            check_sharded(&model, &devices, if moe { 1e-3 } else { 1e-4 }, arch);
            ran.push(arch);
        }
        for must in ["llama", "qwen2", "qwen3", "phi3", "gemma3", "qwen3moe"] {
            assert!(ran.contains(&must), "{must} did not run: {ran:?}");
        }
    }

    /// Load a split of `model` that must fail with a tensor-parallelism
    /// configuration error naming `rule`. The rules are checked before any
    /// device is opened or weight read, so a missing GPU does not matter.
    fn assert_split_rejected(
        model: &TinyModel,
        shards: usize,
        experts: &ExpertPlacement,
        rule: &str,
    ) {
        use crate::provider::local::GpuAdapter;

        let device = test_devices()
            .first()
            .copied()
            .unwrap_or(LocalDevice::Vulkan(GpuAdapter::Default));
        let content = content_of(model);
        let e = GpuTransformer::load_placed(
            &content,
            &model.path,
            32,
            device,
            &vec![device; shards],
            experts,
            &Sizing::single(32),
        )
        .err()
        .unwrap_or_else(|| panic!("a split with {shards} shards must be rejected ({rule})"));
        match &e {
            CucaError::Config(message) => assert!(
                message.starts_with("tensor parallelism: ") && message.contains(rule),
                "{message}"
            ),
            other => panic!("{rule}: expected a Config error, got {other:?}"),
        }
    }

    /// Each rule of the split fails the load with a `CucaError::Config` that
    /// names it (and the numbers it failed on).
    #[test]
    fn gpu_sharded_rejects_unsplittable_heads() {
        let resident = ExpertPlacement::Resident;
        let llama = TinySpec {
            ff: 128,
            ..TinySpec::llama()
        };
        // Heads that do not divide.
        // (F32 weights: three heads of 16 are 48 columns, not a Q4_0 block
        // multiple, which the tiny writer cannot quantize.)
        let heads = write_spec(&TinySpec {
            n_head: 3,
            n_kv_head: 1,
            weight_dtype: GgmlDType::F32,
            ..llama.clone()
        });
        assert_split_rejected(&heads, 1, &resident, "attention heads do not divide");
        // Three shards of four heads.
        assert_split_rejected(&write_spec(&llama), 2, &resident, "4 heads, 3 shards");
        // KV heads that neither divide nor replicate.
        let kv = write_spec(&TinySpec {
            n_head: 6,
            n_kv_head: 3,
            ..llama.clone()
        });
        assert_split_rejected(&kv, 1, &resident, "key/value heads");
        // A shard's query width off the 32-element blocks (1 head of 16).
        let narrow = write_spec(&TinySpec {
            n_head: 2,
            n_kv_head: 2,
            ..llama.clone()
        });
        assert_split_rejected(&narrow, 1, &resident, "query width");
        // The tiny llama's FFN width 96 leaves 48 per shard.
        assert_split_rejected(
            &write_spec(&TinySpec::llama()),
            1,
            &resident,
            "dense FFN width",
        );
        // Full-width Q/K norms.
        assert_split_rejected(
            &write_spec(&TinySpec::olmo2()),
            1,
            &resident,
            "full-width Q/K norm",
        );
        // Recurrent layers.
        assert_split_rejected(&write_qwen35(&tiny_qwen35()), 1, &resident, "delta-net");
        // Latent attention.
        assert_split_rejected(
            &write_mistral4(&TinyMistral4::default()),
            1,
            &resident,
            "multi-head latent attention",
        );
        // Tiered experts.
        let tiered = ExpertPlacement::Tiered(ExpertTiers {
            ram_bytes: 0,
            device_bytes: 1 << 20,
            loader_threads: 1,
            warm_from: None,
        });
        assert_split_rejected(
            &write_spec(&TinySpec::qwen3moe()),
            1,
            &tiered,
            "tiered experts",
        );
    }

    /// The sum over shards is added in shard order on the host and comes
    /// back on every shard's device (with NCCL, in any order).
    #[test]
    fn gpu_all_reduce_adds_in_shard_order() {
        let Some(mut locals) = two_test_devices() else {
            println!("SKIP: set CUCA_LOCAL_DEVICE to a GPU device");
            return;
        };
        let first = locals[0];
        locals.resize(3, first);
        let devices: Vec<Device> = locals.iter().map(|d| open_device(*d).unwrap()).collect();
        let collective = Collective::new(&locals, &devices);
        // (2^24 + 1) + 1 = 2^24 in f32, 2^24 + (1 + 1) = 2^24 + 2.
        let parts: Vec<Tensor<2>> = [16_777_216.0f32, 1.0, 1.0]
            .iter()
            .zip(&devices)
            .map(|(v, device)| Tensor::from_data(TensorData::new(vec![*v; 6], [2, 3]), device))
            .collect();
        let sums = collective.all_reduce_sum(parts).unwrap();
        assert_eq!(sums.len(), 3);
        for (s, (sum, device)) in sums.into_iter().zip(&devices).enumerate() {
            assert!(
                sum.device() == *device,
                "shard {s}: the sum is on its device"
            );
            let values = read_f32(sum, "the sum").unwrap();
            assert_eq!(values.len(), 6);
            for v in values {
                if collective.uses_nccl() {
                    assert!(v == 16_777_216.0 || v == 16_777_218.0, "{v}");
                } else {
                    assert_eq!(v, 16_777_216.0, "shard {s}");
                }
            }
        }
    }

    /// What a shard reads of every per-layer matrix of a tiny Phi 4 mini (fused
    /// QKV and gate-up, several row ranges per cut), Llama (separate
    /// projections) and Gemma 3 (one KV head, replicated) equals the rows
    /// and columns of the whole tensor read the plain way, at world 2 for
    /// both shards. The cut reads touch the file and the host only: the
    /// reader's device is never used.
    #[test]
    fn cut_reads_match_the_whole_tensor() {
        for spec in [
            TinySpec {
                ff: 128,
                ..TinySpec::phi3()
            },
            TinySpec {
                ff: 128,
                ..TinySpec::llama()
            },
            TinySpec {
                ff: 128,
                ..TinySpec::gemma3()
            },
        ] {
            let model = write_spec(&spec);
            let content = content_of(&model);
            let hp = read_hparams(&content).unwrap();
            let device = Device::default();
            let mut reader = WeightReader::open(&content, &model.path, &device, None).unwrap();
            let matrices: Vec<(String, usize, usize)> = content
                .tensor_infos
                .iter()
                .filter_map(|(name, info)| match info.shape.dims() {
                    [rows, cols] => Some((name.clone(), *rows, *cols)),
                    _ => None,
                })
                .collect();
            let mut cut_weights = 0;
            for shard in 0..2 {
                let tp = Tp { world: 2, shard };
                for (tensor, rows, cols) in &matrices {
                    let (rows, cols) = (*rows, *cols);
                    let Some((il, part)) = tp::block_part(tensor) else {
                        continue;
                    };
                    let layout = LayerLayout::read(&hp, il, &hp.layers[il], &content).unwrap();
                    let cut = tp::part_cut(&hp, &layout, il, part, (rows, cols), tp);
                    let dtype = reader.read(tensor, &[rows, cols]).unwrap();
                    let whole = decode_rows(dtype, &reader.buf, cols);
                    let (held, bytes) = reader.cut_bytes(tensor, (rows, cols), &cut).unwrap();
                    assert_eq!(held, dtype, "{} {tensor}", spec.arch);
                    let want: Vec<f32> = cut
                        .rows
                        .iter()
                        .flat_map(|r| r.clone())
                        .flat_map(|row| {
                            whole[row * cols..(row + 1) * cols][cut.cols.clone()].to_vec()
                        })
                        .collect();
                    assert_eq!(want.len(), cut.n_rows() * cut.cols.len());
                    assert_eq!(
                        decode_rows(held, &bytes, cut.cols.len()),
                        want,
                        "{} shard {shard} {tensor} {cut:?}",
                        spec.arch
                    );
                    if !cut.is_whole(rows, cols) {
                        cut_weights += 1;
                    }
                }
            }
            assert!(cut_weights > 0, "{}: nothing was cut", spec.arch);
            // A cut outside the matrix is an error, not a panic.
            let outside = Cut::rows_only(0..1000, 64);
            assert!(
                reader
                    .cut_bytes("blk.0.attn_output.weight", (64, 128), &outside)
                    .is_err()
            );
        }
    }
}
