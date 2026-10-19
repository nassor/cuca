//! The generic transformer forward pass on a GPU through burn (CubeCL),
//! loaded from the same GGUF file as the CPU engine.
//!
//! [`GpuTransformer`] mirrors [`cpu::Transformer`](super::cpu::Transformer)
//! operation for operation: the same hyperparameters ([`Hparams`]) and
//! per-layer tensor layouts ([`LayerLayout`]), the same RoPE tables
//! ([`build_rope_tables`]), masks ([`attention_mask`]), prefill chunking,
//! K/V sharing and MoE routing ([`route`]). Every architecture of the
//! [`arch`](super::arch) table runs through it: separate or fused
//! `attn_qkv`, Q/K/V/O biases, QK-norm, post-norms, sliding windows,
//! adjacent-pair (`NORM`) and half-split (`NEOX`) RoPE with partial rotary
//! dimensions and NoPE layers, a score scale and optional tanh soft-cap,
//! dense gated FFNs (fused or separate gate/up, SiLU or GELU) and top-k
//! mixtures of experts, embedding / residual / logit scales, a final logit
//! soft-cap, tied or separate output head, and the Gemma 4 extras (shared
//! K/V layers, per-layer embeddings, weightless value norm, per-layer output
//! scale). GELU is the tanh approximation, as in candle.
//!
//! # Weights
//!
//! - Q4_0 and Q8_0 matmul weights are repacked once at load into burn's
//!   block-quantized layout (`Q4S` / `Q8S` values packed into `u32` words, one
//!   f16 scale per 32-element block, see [`pack_blocks`]) and stay quantized
//!   on the device: burn's matmul reads them directly and dequantizes inside
//!   the kernel.
//! - F32, F16 and BF16 matmul weights are dequantized on the host row by row
//!   ([`dequant_row`]) and uploaded as f32 for burn's float matmul, exactly.
//!   They cost 4 bytes per element on the device whatever their size in the
//!   file (an F16 or BF16 weight doubles).
//! - **Every other dtype** (Q4_1, Q5_0, Q5_1, Q8_1 and the K-quants Q2_K to
//!   Q6_K) is dequantized on the host row by row and **requantized to Q8_0**
//!   ([`quantize_q8_0`]), then repacked like a native Q8_0 weight, so it costs
//!   about 1.06 bytes per element on the device instead of 4: a 3.8
//!   B-parameter Q4_K_M file needs about 4 GB rather than 15 GB. The price is
//!   a rounding error: each element moves by at most half a Q8_0 step, that
//!   is `block_max / 254` (plus the f16 rounding of the scale, below 0.1% of
//!   the block maximum). Q8_0 resolves a block with 255 levels where Q4_K,
//!   Q5_K and Q6_K resolve theirs with about 15, 31 and 63, so this extra
//!   error is below the one the file's own quantization already carries.
//!   Logits still differ from a host-dequantized reference by far more than
//!   float rounding (about 1e-2 of their magnitude in the tiny-model tests,
//!   rather than 1e-5), so a K-quant model decodes the CPU engine's tokens
//!   except where two candidates are that close.
//! - Norm vectors, biases and scalars are f32 whatever their stored dtype.
//! - **Exception**: a weight that could share an autotune key with a weight of
//!   the other scheme is held as f32 whatever its dtype (exact), see the
//!   known bugs below and [`plan_dense`].
//!
//! # Memory
//!
//! - Device memory holds the matmul weights as above. The token embedding is
//!   uploaded only when it doubles as the LM head (no `output.weight`), once,
//!   as the head weight. A weight over 1 GiB on the device is split into row
//!   chunks (a large tied head becomes several `[rows_i, n_embd]` weights);
//!   [`linear`] multiplies by each chunk and concatenates the outputs, which
//!   is exact, so no single device buffer exceeds [`MAX_PART_BYTES`].
//! - Embedding lookups never touch the device copy: the host reads each
//!   token's row of `token_embd` (and, for Gemma 4, of `per_layer_token_embd`)
//!   from the GGUF file ([`RowTable`]) and uploads the dequantized rows. The
//!   per-layer table is never uploaded.
//! - Expert tensors are read and uploaded one expert at a time (a `[n_expert,
//!   rows, cols]` tensor becomes `n_expert` 2-D device weights), so the host
//!   buffer is one expert and the device memory is the dtype rule above.
//! - The K/V cache lives on the device, preallocated for `context_length`
//!   positions as on the CPU ([`Hparams::kv_bytes_per_position`] bytes per
//!   position). Writes are in place: a layer takes its cache tensors out of
//!   the cache, assigns the new positions, and puts them back, and attention
//!   views are dropped before the next write, so no write copies the cache.
//! - RoPE cos/sin tables for `context_length` positions, one pair per distinct
//!   rotary embedding, live on the device.
//! - During load the host holds the read buffer of the whole tensor (its file
//!   size), the converted copy of one chunk (f32 or Q8_0, at most 1 GiB of
//!   device bytes) and the repacked copy of that chunk; per forward pass it
//!   holds the embedding rows, the attention masks and, per MoE layer, the
//!   router logits (`tokens * n_expert` f32).
//!
//! # MoE
//!
//! The router logits are read back to the host once per MoE layer (a device
//! synchronization point), [`route`] picks the experts and their weights, then
//! each expert runs on the rows of its own tokens: gathered with `select`,
//! scaled by their routing weights, and accumulated with `select_assign(Add)`.
//! A single token skips the gather and scatter.
//!
//! Routing weights are applied in two ways. A pass over several rows scales
//! the gathered hidden rows by a device tensor of routing weights before the
//! down projection (the projection is linear, so the result is the same):
//! multiplying the projection's output by that tensor made a fused kernel lose
//! the Vulkan device on the tiny Granite MoE test, and any host readback
//! inside the expert chain avoided the loss; the mechanism was not
//! established. A single token multiplies the expert's output by the routing
//! weight as a host scalar (`mul_scalar`); that form ran through the Granite
//! MoE live suite on Vulkan, every decode step included, without device loss.
//!
//! # Speculative drafting
//!
//! [`GpuTransformer::forward_ext`] returns the final-norm hidden rows next to
//! the logits, and [`GpuDrafter`] drafts tokens from a Gemma 4
//! multi-token-prediction drafter exactly as the CPU `mtp::Drafter` does: the
//! drafter's layers attend to the target's K/V cache, which they reach through
//! clones of its tensor handles taken for the length of one `draft` call (a
//! handle held across a target forward would turn its in-place cache writes
//! into whole-cache copies). The GPU does not keep a batched pass bit-equal
//! to single-token steps (`exact` is ignored), so a speculative greedy run can
//! differ from plain greedy where two logits are within float rounding; in
//! the measured runs the outputs were identical.
//!
//! # Known burn 0.22 Vulkan bugs
//!
//! The first three are worked around in [`linear`], whose docs give the details and the tests
//! that pin them: a quantized weight on the right of a matmul loses the device
//! in some autotune candidates; a single-row product through a quantized
//! weight either returns wrong values (weight on the left) or fails to launch
//! for shapes the `vecmat` kernel does not support once another shape of the
//! same size bucket tuned it. A single-token (decode) product uses `vecmat`
//! only for weights it supports and otherwise dequantizes the weight, so such
//! shapes (an input width that is not a multiple of 64, or an output width
//! that is not a multiple of 512 for Q4_0, 256 for Q8_0 or 64 for f32, with
//! the widest plane of 64 lanes assumed) decode several times slower than the
//! rest.
//!
//! The other one lives in the weights, not in `linear`: burn's matmul
//! autotune key holds the shapes (rounded up to a power of two), the element
//! types and the strides, but **not the quantization scheme**. A kernel tuned
//! for a Q8_0 weight (`num_quants` 4) is replayed on a Q4_0 weight (8) of any
//! shape in the same bucket and fails at launch ("quantized view float vector
//! size 4 must be a positive multiple of num_quants 8", or "Shared memory
//! budget exceeded" when the replayed tile needs more than the 32 KiB the
//! device allows); the other order happens to work, so which model fails
//! depends on which weight a forward pass tunes first. Files that mix the
//! schemes (a Q4_0 file with a few Q4_1 layers and a Q6_K head, as Llama 3.2
//! and Phi 3.5 are) hit it on the first pass that tunes a Q8 weight before a
//! Q4 one of the same bucket (any prefill length, or a decode step). The
//! autotune cache lives per process and device, so the plan covers the
//! weights of one model and, for a drafter, those of its target. The
//! loader therefore plans the weights ([`plan_dense`]): where a Q4 and a Q8
//! weight could share a key (both dimensions within a factor of 2.5) the one
//! of the scheme with fewer bytes is held as f32 instead, which has its own
//! key. That is exact (an f32 copy of the file's dequantized values, no
//! requantization) and costs 4 bytes per element for those few weights; a
//! head far from every other weight (Gemma 4's 262144-row Q8_0 one) and a
//! file of one scheme are untouched.

#[cfg(not(target_endian = "little"))]
compile_error!(
    "provider-local-gpu copies GGUF little-endian f16 scales into burn's native-endian layout"
);

use std::collections::HashSet;
use std::f64::consts::FRAC_1_SQRT_2;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::ops::Range;
use std::path::Path;
use std::sync::Mutex;

use burn::tensor::quantization::{QuantScheme, QuantStore, QuantValue, ScaleDtype};
use burn::tensor::{DType, Device, IndexingUpdateOp, Int, Tensor, TensorData, activation};
use candle_core::quantized::{GgmlDType, gguf_file};

use super::arch::{Act, Arch};
use super::cpu::{Forward, Hidden};
use super::layout::{Ffn, LayerLayout, Qkv, name, tensor_info};
use super::moe::route;
use super::params::{Hparams, LayerSpec, MoeSpec};
use super::quant::{RowTable, dequant_row};
use super::rope::{RopeKind, RopeSpec};
use super::{Want, attention_mask, build_rope_tables, check_context, err, read_hparams};
use crate::error::CucaError;
use crate::provider::local::{LocalDevice, device_unavailable};

/// burn's quantization scheme equivalent to a GGUF block dtype, `None` for
/// every dtype the GPU engine does not keep quantized.
fn quant_scheme(dtype: GgmlDType) -> Option<QuantScheme> {
    let value = match dtype {
        GgmlDType::Q4_0 => QuantValue::Q4S,
        GgmlDType::Q8_0 => QuantValue::Q8S,
        _ => return None,
    };
    Some(
        QuantScheme::default()
            .with_value(value)
            .per_block([32], ScaleDtype::F16)
            .with_store(QuantStore::PackedU32(0)),
    )
}

/// Repack the GGUF bytes of a `[rows, cols]` Q4_0 or Q8_0 tensor into the
/// byte layout of a burn quantized tensor with [`quant_scheme`]: every
/// block's values, packed into little-endian `u32` words, followed by every
/// block's f16 scale, both in row-major block order.
///
/// - Q4_0 element `e = nibble - 8` is stored as its 4-bit two's complement
///   (`nibble ^ 8`); element `i` of each run of 8 sits at bits `4 * (i % 8)`
///   of word `i / 8`. GGUF holds elements `0..16` in the low nibbles of a
///   block's 16 bytes and `16..32` in the high nibbles.
/// - Q8_0 values are signed bytes already in burn's order (element `i` at
///   bits `8 * (i % 4)` of word `i / 4`) and are copied verbatim.
/// - Scales are copied verbatim: GGUF stores them little-endian and burn
///   native-endian, equal on the little-endian hosts this module builds for.
fn pack_blocks(
    dtype: GgmlDType,
    raw: &[u8],
    rows: usize,
    cols: usize,
) -> Result<Vec<u8>, CucaError> {
    if quant_scheme(dtype).is_none() {
        return Err(err(format!("{dtype:?} has no burn block-quantized layout")));
    }
    let block_bytes = dtype.type_size();
    if !cols.is_multiple_of(32) || raw.len() != rows * cols / 32 * block_bytes {
        return Err(err(format!(
            "{} bytes do not hold a [{rows}, {cols}] {dtype:?} tensor",
            raw.len()
        )));
    }
    let blocks = rows * cols / 32;
    // Values: 16 bytes per Q4 block, 32 per Q8 block; scales: 2 per block.
    let value_bytes = block_bytes - 2;
    let mut out = Vec::with_capacity(blocks * block_bytes);
    for block in raw.chunks_exact(block_bytes) {
        let q = &block[2..];
        if dtype == GgmlDType::Q4_0 {
            for word in 0..4 {
                let mut packed = 0u32;
                for i in 0..8 {
                    let element = 8 * word + i;
                    let byte = q[element % 16];
                    let nibble = if element < 16 { byte & 0x0F } else { byte >> 4 };
                    packed |= u32::from(nibble ^ 8) << (4 * i);
                }
                out.extend_from_slice(&packed.to_le_bytes());
            }
        } else {
            out.extend_from_slice(&q[..value_bytes]);
        }
    }
    for block in raw.chunks_exact(block_bytes) {
        out.extend_from_slice(&block[..2]);
    }
    Ok(out)
}

/// The burn device for `device`, opened now so that a missing adapter or
/// driver is an error here rather than a panic at the first tensor.
fn open_device(device: LocalDevice) -> Result<Device, CucaError> {
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
/// of the adapter, which burn does not expose; the development GPU refused a
/// 2.46 GB buffer, so a conservative 1 GiB is assumed everywhere.
const MAX_PART_BYTES: usize = 1 << 30;

/// A `[rows, cols]` matmul weight on the device as one or more row chunks
/// (one unless the weight would exceed [`MAX_PART_BYTES`], as a large tied
/// LM head does); [`linear`] concatenates the chunks' outputs, which is exact.
struct Weight {
    parts: Vec<Tensor<2>>,
}

/// Quantize `values` (a whole number of 32-element blocks) to the GGUF Q8_0
/// layout, appended to `out`: per block an f16 scale `amax / 127` and 32
/// signed bytes. The values are rounded against the f16-rounded scale, so a
/// dequantized element is within half a scale step of the input.
fn quantize_q8_0(values: &[f32], out: &mut Vec<u8>) {
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
/// parts of at most [`MAX_PART_BYTES`]; see [`upload_matrix_with`].
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
        MAX_PART_BYTES,
        false,
    )
}

/// The layout a weight of GGUF dtype `dtype` takes on the device under the
/// rules of [`upload_matrix_with`]: `None` for f32.
fn stored_layout(dtype: GgmlDType) -> Option<GgmlDType> {
    match dtype {
        GgmlDType::Q4_0 | GgmlDType::Q8_0 => Some(dtype),
        GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16 => None,
        _ => Some(GgmlDType::Q8_0),
    }
}

/// Device bytes of one `cols`-wide row stored as `stored`.
fn stored_row_bytes(stored: Option<GgmlDType>, cols: usize) -> usize {
    match stored {
        Some(GgmlDType::Q4_0) => cols / 2 + cols / 16,
        Some(_) => cols + cols / 16,
        None => cols * 4,
    }
    .max(1)
}

/// The two block-quantization schemes the engine keeps on the device. burn's
/// matmul autotune key omits the scheme, so a kernel tuned on one
/// replays on the other for any shape in the same bucket, and a kernel tuned
/// for Q8 (`num_quants` 4) fails on Q4 (`num_quants` 8): see [`plan_dense`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SchemeClass {
    Q4,
    Q8,
}

/// The scheme a weight of GGUF dtype `dtype` is kept in, `None` for the
/// float dtypes (held as f32).
fn scheme_class(dtype: GgmlDType) -> Option<SchemeClass> {
    match stored_layout(dtype) {
        Some(GgmlDType::Q4_0) => Some(SchemeClass::Q4),
        Some(_) => Some(SchemeClass::Q8),
        None => None,
    }
}

/// One quantized matmul weight as [`plan_dense`] sees it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct QuantShape {
    rows: usize,
    cols: usize,
    class: SchemeClass,
}

impl QuantShape {
    /// Rows of the largest chunk the weight is split into.
    fn part_rows(&self) -> usize {
        (MAX_PART_BYTES / stored_row_bytes(Some(self.stored()), self.cols))
            .clamp(1, self.rows.max(1))
    }

    /// The GGUF layout the weight is kept in on the device.
    fn stored(&self) -> GgmlDType {
        match self.class {
            SchemeClass::Q4 => GgmlDType::Q4_0,
            SchemeClass::Q8 => GgmlDType::Q8_0,
        }
    }

    /// Bytes on the device.
    fn bytes(&self) -> usize {
        self.rows * stored_row_bytes(Some(self.stored()), self.cols)
    }

    /// Whether burn's matmul autotune may give the two weights one key. The
    /// key holds the dimensions rounded up to a power of the anchor base
    /// (2 by default, 2.5 at the coarsest autotune level), so two dimensions
    /// within a factor of 2.5 may share a bucket and weights whose rows and
    /// columns both are cannot be told apart. The anchor has no clamp, so
    /// the bucket of a dimension is decided by its ratio alone.
    fn may_share_key(&self, other: &QuantShape) -> bool {
        let close = |x: usize, y: usize| x.max(y) * 2 < x.min(y) * 5;
        close(self.part_rows(), other.part_rows()) && close(self.cols, other.cols)
    }
}

/// The matmul weights of the model in `content` that are held quantized, by
/// name, in name order: every 2-D tensor and every expert of a 3-D tensor
/// except the per-layer embedding table (read row by row) and a token
/// embedding that is not the LM head.
fn quant_shapes(content: &gguf_file::Content) -> Vec<(String, QuantShape)> {
    let tied = !content.tensor_infos.contains_key(name::OUTPUT);
    let mut out: Vec<(String, QuantShape)> = content
        .tensor_infos
        .iter()
        .filter(|(tensor, _)| {
            tensor.as_str() != name::PER_LAYER_TOKEN_EMBD
                && !(tensor.as_str() == name::TOKEN_EMBD && !tied)
        })
        .filter_map(|(tensor, info)| {
            let class = scheme_class(info.ggml_dtype)?;
            let (rows, cols) = match info.shape.dims() {
                [rows, cols] | [_, rows, cols] => (*rows, *cols),
                _ => return None,
            };
            Some((tensor.clone(), QuantShape { rows, cols, class }))
        })
        .collect();
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

/// The weights of `own` to hold as f32 instead of quantized so that no Q4 and
/// Q8 weight can share an autotune key (the scheme is not part of it).
///
/// For every pair of weights of different schemes that [may share a
/// key](QuantShape::may_share_key) one is converted: the weight of the
/// scheme with fewer device bytes among `own` and `peers` (a model's own
/// weights and the already loaded weights of the model it pairs with), or
/// the `own` weight when the other belongs to `peers`, which cannot change.
/// Dense weights are exact (an f32 copy of the dequantized file values), so
/// the plan costs memory, not accuracy, and a model of one scheme, or whose
/// schemes differ in shape, converts nothing.
fn plan_dense(own: &[(String, QuantShape)], peers: &[QuantShape]) -> HashSet<String> {
    let total = |class| {
        own.iter()
            .map(|(_, s)| s)
            .chain(peers)
            .filter(|s| s.class == class)
            .map(QuantShape::bytes)
            .sum::<usize>()
    };
    let minority = if total(SchemeClass::Q4) < total(SchemeClass::Q8) {
        SchemeClass::Q4
    } else {
        SchemeClass::Q8
    };
    let mut dense = HashSet::new();
    for (tensor, weight) in own {
        let collides = |other: &QuantShape, fixed: bool| {
            other.class != weight.class
                && weight.may_share_key(other)
                && (fixed || weight.class == minority)
        };
        if peers.iter().any(|p| collides(p, true)) || own.iter().any(|(_, o)| collides(o, false)) {
            dense.insert(tensor.clone());
        }
    }
    dense
}

/// The `[rows, cols]` weight whose GGUF bytes are `raw`, on the device, in
/// row chunks of at most `limit` device bytes:
///
/// - Q4_0 and Q8_0 are repacked and stay quantized ([`pack_blocks`]);
/// - F32, F16 and BF16 are dequantized on the host row by row to f32
///   (exact);
/// - every other dtype (Q4_1, Q5_0, Q5_1, Q8_1, the K-quants) is dequantized
///   row by row and requantized to Q8_0 ([`quantize_q8_0`]), so it costs
///   about its file size or less instead of four bytes per element, at the
///   price of a rounding error of at most half a Q8_0 step per element;
/// - with `dense`, any dtype is dequantized to f32 (exact), whatever the
///   rules above say: [`plan_dense`] picks the weights that must not share
///   an autotune key with a weight of the other quantization scheme.
#[expect(
    clippy::too_many_arguments,
    reason = "tensor identity, shape, destination and the two upload policies are all independent"
)]
fn upload_matrix_with(
    tensor_name: &str,
    dtype: GgmlDType,
    raw: &[u8],
    rows: usize,
    cols: usize,
    device: &Device,
    limit: usize,
    dense: bool,
) -> Result<Weight, CucaError> {
    let context = |e: CucaError| err(format!("tensor `{tensor_name}`: {e}"));
    let block = dtype.block_size();
    if !cols.is_multiple_of(block) {
        return Err(err(format!(
            "tensor `{tensor_name}`: row length {cols} is not a multiple of the {dtype:?} block size {block}"
        )));
    }
    let row_bytes = cols / block * dtype.type_size();
    if raw.len() != rows * row_bytes {
        return Err(err(format!(
            "tensor `{tensor_name}`: {} bytes do not hold a [{rows}, {cols}] {dtype:?} tensor",
            raw.len()
        )));
    }
    // The layout the weight takes on the device, and its bytes per row.
    let stored = if dense { None } else { stored_layout(dtype) };
    if stored == Some(GgmlDType::Q8_0) && !cols.is_multiple_of(32) {
        return Err(err(format!(
            "tensor `{tensor_name}`: row length {cols} is not a multiple of the 32-element Q8_0 block"
        )));
    }
    let mut chunk = (limit / stored_row_bytes(stored, cols)).clamp(1, rows.max(1));
    if chunk < rows && chunk >= 64 {
        // Keep chunk widths multiples of the plane width, which the
        // single-row kernel needs ([`vecmat_fits`]).
        chunk -= chunk % 64;
    }

    let mut parts = Vec::with_capacity(rows.div_ceil(chunk));
    let mut values = vec![0f32; cols];
    let mut start = 0;
    while start < rows {
        let end = (start + chunk).min(rows);
        let n = end - start;
        let src = &raw[start * row_bytes..end * row_bytes];
        let data = match (stored, quant_scheme(stored.unwrap_or(dtype))) {
            (Some(layout), Some(scheme)) if layout == dtype => TensorData::from_bytes_vec(
                pack_blocks(dtype, src, n, cols).map_err(context)?,
                [n, cols],
                DType::QFloat(scheme),
            ),
            (Some(layout), Some(scheme)) => {
                let mut q8 = Vec::with_capacity(n * (cols / 32) * 34);
                for row in src.chunks_exact(row_bytes.max(1)) {
                    dequant_row(dtype, row, 1.0, &mut values).map_err(context)?;
                    quantize_q8_0(&values, &mut q8);
                }
                TensorData::from_bytes_vec(
                    pack_blocks(layout, &q8, n, cols).map_err(context)?,
                    [n, cols],
                    DType::QFloat(scheme),
                )
            }
            _ => {
                let mut floats = vec![0f32; n * cols];
                for (row, dst) in src
                    .chunks_exact(row_bytes.max(1))
                    .zip(floats.chunks_exact_mut(cols.max(1)))
                {
                    dequant_row(dtype, row, 1.0, dst).map_err(context)?;
                }
                TensorData::new(floats, [n, cols])
            }
        };
        parts.push(Tensor::from_data(data, device));
        start = end;
    }
    Ok(Weight { parts })
}

/// Reads tensors from the GGUF file into one reused buffer and uploads them.
struct WeightReader<'a> {
    content: &'a gguf_file::Content,
    file: File,
    buf: Vec<u8>,
    device: &'a Device,
    /// Tensors held as f32 whatever their dtype; see [`plan_dense`].
    dense: HashSet<String>,
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
            MAX_PART_BYTES,
            self.dense.contains(tensor_name),
        )
    }

    /// A matrix weight plus, when `bias` is set, its `[rows]` bias.
    fn proj(
        &mut self,
        il: usize,
        part: &str,
        shape: [usize; 2],
        bias: bool,
    ) -> Result<Proj, CucaError> {
        Ok(Proj {
            w: self.matrix(&name::blk(il, part), shape[0], shape[1])?,
            bias: if bias {
                Some(self.vector(&name::blk_bias(il, part), shape[0])?)
            } else {
                None
            },
        })
    }

    /// The `[n_expert, rows, cols]` expert tensor `tensor_name`, one device
    /// weight per expert, read one expert at a time.
    fn experts(
        &mut self,
        tensor_name: &str,
        n_expert: usize,
        rows: usize,
        cols: usize,
    ) -> Result<Vec<Weight>, CucaError> {
        let info = tensor_info(self.content, tensor_name)?;
        if info.shape.dims() != [n_expert, rows, cols] {
            return Err(err(format!(
                "tensor `{tensor_name}`: expected shape {:?}, found {:?}",
                [n_expert, rows, cols],
                info.shape.dims()
            )));
        }
        let dtype = info.ggml_dtype;
        if !cols.is_multiple_of(dtype.block_size()) {
            return Err(err(format!(
                "tensor `{tensor_name}`: row length {cols} is not a multiple of the {dtype:?} block size {}",
                dtype.block_size()
            )));
        }
        let expert_bytes = rows * (cols / dtype.block_size() * dtype.type_size());
        let mut out = Vec::with_capacity(n_expert);
        for e in 0..n_expert {
            self.read_at(tensor_name, e * expert_bytes, expert_bytes)?;
            out.push(upload_matrix_with(
                tensor_name,
                dtype,
                &self.buf,
                rows,
                cols,
                self.device,
                MAX_PART_BYTES,
                self.dense.contains(tensor_name),
            )?);
        }
        Ok(out)
    }

    /// A 1-D tensor dequantized on the host.
    fn host_vector(&mut self, tensor_name: &str, len: usize) -> Result<Vec<f32>, CucaError> {
        let dtype = self.read(tensor_name, &[len])?;
        let mut out = vec![0f32; len];
        dequant_row(dtype, &self.buf, 1.0, &mut out)
            .map_err(|e| err(format!("tensor `{tensor_name}`: {e}")))?;
        Ok(out)
    }

    /// A 1-D f32 tensor on the device.
    fn vector(&mut self, tensor_name: &str, len: usize) -> Result<Tensor<1>, CucaError> {
        let values = self.host_vector(tensor_name, len)?;
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
        let values = self.host_vector(tensor_name, 1)?;
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

enum FfnWeights {
    Dense {
        gate_up: GateUp,
        down: Weight,
    },
    Moe {
        spec: MoeSpec,
        router: Weight,
        gate: Vec<Weight>,
        up: Vec<Weight>,
        down: Vec<Weight>,
    },
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
    attn: AttnWeights,
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

/// K and V cache tensors, `[1, n_kv_head, context_length, head_dim]`, of
/// every layer (`None` for layers that share another layer's cache).
type KvLayers = Vec<Option<(Tensor<4>, Tensor<4>)>>;

/// The RoPE rows of the new positions: `[1, 1, t, n_dims / 2]`.
struct RopeRows {
    cos: Tensor<4>,
    sin: Tensor<4>,
}

/// The visible range and mask of the layers with one window.
struct MaskInput {
    window: Option<usize>,
    /// First visible cached position.
    lo: usize,
    /// `[1, 1, t, hi - lo]`; `None` for a single query, which sees every key.
    mask: Option<Tensor<4>>,
}

/// What one forward pass shares between its layers.
struct Pass {
    start: usize,
    /// New tokens.
    t: usize,
    /// Per-layer inputs of the Gemma 4 per-layer embeddings,
    /// `[t, n_layer, ple_dim]`.
    ple: Option<Tensor<3>>,
    /// One entry per [`Hparams::ropes`] entry.
    ropes: Vec<RopeRows>,
    /// One entry per distinct layer window.
    masks: Vec<MaskInput>,
}

/// Everything read-only during a forward pass.
struct Weights {
    hp: Hparams,
    device: Device,
    ple: Option<PleWeights>,
    layers: Vec<LayerWeights>,
    output_norm: Tensor<1>,
    lm_head: Weight,
    ropes: Vec<RopeTable>,
}

/// A loaded model on a GPU with its K/V cache; see the module docs.
pub(crate) struct GpuTransformer {
    weights: Weights,
    context_length: usize,
    vocab: usize,
    /// Behind a lock because a drafter reads embedding rows through a shared
    /// reference to the target while the table seeks in its file.
    embed: Mutex<RowTable>,
    ple_table: Option<RowTable>,
    kv: KvLayers,
    kv_len: usize,
    /// The weights held quantized, for a drafter's [`plan_dense`].
    peers: Vec<QuantShape>,
}

impl Weights {
    /// Read every weight of the model `hp` describes through `reader` (onto
    /// the reader's device) and allocate its K/V cache: the layers, the final
    /// norm, the LM head (`output.weight`, else the token embedding), the
    /// per-layer-embedding projection and the RoPE tables. A drafter's cache
    /// is empty: it reads the target's.
    fn load(
        reader: &mut WeightReader<'_>,
        path: &Path,
        hp: Hparams,
        context_length: usize,
    ) -> Result<(Self, KvLayers), CucaError> {
        let content = reader.content;
        let device = reader.device.clone();
        let n_embd = hp.n_embd;
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
        let lm_head = reader.matrix(head, vocab, n_embd)?;
        let output_norm = reader.vector(name::OUTPUT_NORM, n_embd)?;
        let ple = match hp.ple_dim {
            Some(dim) => {
                let row_len = hp.n_layer * dim;
                Some(PleWeights {
                    proj: reader.matrix(name::PER_LAYER_MODEL_PROJ, row_len, n_embd)?,
                    proj_norm: reader.vector(name::PER_LAYER_PROJ_NORM, dim)?,
                })
            }
            None => None,
        };

        let ropes = build_rope_tables(content, path, &hp, context_length)?
            .into_iter()
            .map(|t| {
                let shape = [context_length, t.half];
                RopeTable {
                    cos: Tensor::from_data(TensorData::new(t.cos, shape), &device),
                    sin: Tensor::from_data(TensorData::new(t.sin, shape), &device),
                }
            })
            .collect();

        let mut layers = Vec::with_capacity(hp.n_layer);
        let mut kv: KvLayers = Vec::with_capacity(hp.n_layer);
        for il in 0..hp.n_layer {
            let spec = &hp.layers[il];
            let layout = LayerLayout::read(&hp, il, spec, content)?;
            if spec.owns_kv(il) {
                let shape = [1, spec.n_kv_head, context_length, spec.head_dim];
                kv.push(Some((
                    Tensor::zeros(shape, &device),
                    Tensor::zeros(shape, &device),
                )));
            } else {
                kv.push(None);
            }
            layers.push(load_layer(reader, &hp, il, spec, &layout)?);
        }
        Ok((
            Self {
                hp,
                device,
                ple,
                layers,
                output_norm,
                lm_head,
                ropes,
            },
            kv,
        ))
    }
}

impl GpuTransformer {
    /// Load the model from the GGUF file at `path`, whose index `content`
    /// was read already, onto `device` with a K/V cache for
    /// `context_length` positions.
    ///
    /// # Errors
    ///
    /// [`CucaError::Config`] when `context_length` is zero or above the
    /// checkpoint maximum or `device` is not a GPU compiled into this build;
    /// [`CucaError::Provider`] for unsupported or malformed checkpoints and a
    /// device that fails to open; [`CucaError::Io`] when the file cannot be
    /// read.
    pub(crate) fn load(
        content: &gguf_file::Content,
        path: &Path,
        context_length: usize,
        device: LocalDevice,
    ) -> Result<Self, CucaError> {
        let hp = read_hparams(content)?;
        if hp.arch.is_drafter() {
            return Err(err(format!(
                "`{}` is a multi-token-prediction drafter, not a standalone model: pass it as the `drafter` of a Gemma 4 model",
                hp.arch.name()
            )));
        }
        if hp.behavior.hybrid_delta_net {
            return Err(err(format!(
                "`{}`: recurrent (gated delta-net) layers are not implemented on GPU devices; use LocalDevice::Cpu",
                hp.arch.name()
            )));
        }
        check_context(&hp, context_length)?;
        let device = open_device(device)?;

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

        let shapes = quant_shapes(content);
        let dense = plan_dense(&shapes, &[]);
        let peers = shapes
            .iter()
            .filter(|(tensor, _)| !dense.contains(tensor))
            .map(|(_, shape)| *shape)
            .collect();
        let mut reader = WeightReader {
            content,
            file: File::open(path)?,
            buf: Vec::new(),
            device: &device,
            dense,
        };
        let (weights, kv) = Weights::load(&mut reader, path, hp, context_length)?;
        drop(reader);

        Ok(Self {
            weights,
            context_length,
            vocab,
            embed: Mutex::new(embed),
            ple_table,
            kv,
            kv_len: 0,
            peers,
        })
    }

    /// Run `tokens` through the model at positions `start_pos..`, appending
    /// their K/V to the cache; returns the logits `want` selects. Same
    /// contract and errors as the CPU engine's `forward`.
    #[cfg(test)]
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

    /// [`GpuTransformer::forward`] that also returns the hidden-state rows
    /// `hidden` selects (one per row of logits, `n_embd` floats each).
    ///
    /// `exact` is ignored: a batched pass on the GPU equals single-token
    /// steps only up to float rounding (the matmul kernels differ by shape),
    /// so a speculative greedy run can pick a different token than plain
    /// decoding where two logits are within that rounding.
    ///
    /// # Errors
    ///
    /// [`CucaError::Provider`] when `hidden` is neither `None` nor
    /// `PostNorm`, `tokens` is empty, `start_pos` is not the cache length,
    /// the pass exceeds the context length or a device operation fails;
    /// [`CucaError::Io`] when an embedding row cannot be read.
    pub(crate) fn forward_ext(
        &mut self,
        tokens: &[u32],
        start_pos: usize,
        want: Want,
        hidden: Hidden,
        _exact: bool,
    ) -> Result<Forward, CucaError> {
        if !matches!(hidden, Hidden::None | Hidden::PostNorm) {
            return Err(err(
                "hidden rows beyond the logits rows are not supported on the GPU engine",
            ));
        }
        if tokens.is_empty() {
            return Err(err("forward requires at least one token"));
        }
        if start_pos != self.kv_len {
            return Err(err(format!(
                "forward start {start_pos} does not match the cache length {}",
                self.kv_len
            )));
        }
        let hi = start_pos + tokens.len();
        if hi > self.context_length {
            return Err(err(format!(
                "{hi} positions exceed the context length {}",
                self.context_length
            )));
        }
        let hp = &self.weights.hp;
        let embeddings = self
            .embed
            .get_mut()
            .map_err(|_| err("the embedding table lock is poisoned"))?
            .rows(tokens, hp.embd_scale)?;
        let per_layer = match (&mut self.ple_table, hp.ple_dim) {
            (Some(table), Some(dim)) => Some(table.rows(tokens, (dim as f32).sqrt())?),
            _ => None,
        };
        let out = self.weights.forward_layers(
            &mut self.kv,
            embeddings,
            per_layer,
            tokens.len(),
            start_pos,
            want,
            hidden,
        )?;
        self.kv_len = hi;
        Ok(out)
    }

    /// Vocabulary size.
    pub(crate) fn vocab(&self) -> usize {
        self.vocab
    }

    /// Width of the hidden state (and of an embedding row).
    pub(crate) fn n_embd(&self) -> usize {
        self.weights.hp.n_embd
    }

    /// The architecture.
    pub(crate) fn arch(&self) -> Arch {
        self.weights.hp.arch
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
            .rows(&[token], self.weights.hp.embd_scale)
    }

    /// The K/V cache tensors a drafter reads, `len` positions of them: the
    /// cache of the last sliding-window layer's source and of the last
    /// full-attention layer's source (the layers Gemma 4's shared-K/V layers
    /// read). The handles are clones: hold them only for the length of one
    /// drafting call, since a K/V write while another handle to the cache
    /// exists copies the whole cache instead of writing in place.
    fn ext_kv(&self, len: usize) -> Result<GpuExtKv, CucaError> {
        let source = |sliding: bool| -> Result<(Tensor<4>, Tensor<4>), CucaError> {
            let layer = self
                .weights
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
            self.kv
                .get(layer.kv_source)
                .and_then(Option::as_ref)
                .cloned()
                .ok_or_else(|| err(format!("layer {} owns no K/V cache", layer.kv_source)))
        };
        Ok(GpuExtKv {
            sliding: source(true)?,
            full: source(false)?,
            len,
        })
    }

    /// Drop every cached position from `len` on and return how many remain
    /// (`min(len, current length)`: K/V rows are positional); the next
    /// `forward` must start there. Positions at or past it are overwritten,
    /// not cleared.
    pub(crate) fn truncate_cache(&mut self, len: usize) -> usize {
        self.kv_len = len.min(self.kv_len);
        self.kv_len
    }

    /// Number of cached positions (test-only).
    #[cfg(test)]
    pub(crate) fn cache_len(&self) -> usize {
        self.kv_len
    }
}

/// The target's K/V tensors a drafter attends to; see
/// [`GpuTransformer::ext_kv`].
struct GpuExtKv {
    sliding: (Tensor<4>, Tensor<4>),
    full: (Tensor<4>, Tensor<4>),
    /// Positions of the target cache the drafter may read.
    len: usize,
}

impl Weight {
    /// Rows of the whole weight (the output width of [`linear`]).
    fn rows(&self) -> usize {
        self.parts.iter().map(|p| p.dims()[0]).sum()
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
        let target_layers = &target.weights.hp.layers;
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
        let shapes = quant_shapes(content);
        let dense = plan_dense(&shapes, &target.peers);
        let mut reader = WeightReader {
            content,
            file: File::open(path)?,
            buf: Vec::new(),
            device: &target.weights.device,
            dense,
        };
        let pre = reader.matrix("nextn.pre_projection.weight", n_embd, 2 * n_embd_out)?;
        let post = reader.matrix("nextn.post_projection.weight", n_embd_out, n_embd)?;
        let (net, _) = Weights::load(&mut reader, path, hp, context_length)?;
        if net.lm_head.rows() != target.vocab() {
            return Err(err(format!(
                "drafter vocabulary {} differs from the target's {}",
                net.lm_head.rows(),
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

    /// Draft `k` tokens after the pending token `first`, as the CPU drafter.
    ///
    /// `hidden` is the target's final-norm hidden state at its last cached
    /// position (`Hidden::PostNorm` of its last forward pass), `len` the
    /// number of cached positions (`first` sits at position `len` and is not
    /// cached). Each step's logits go to `pick`, which chooses (and may
    /// record) the drafted token; every step uses the position `len`.
    ///
    /// # Errors
    ///
    /// [`CucaError::Provider`] when `hidden` has the wrong width, `len` is
    /// zero, beyond the target's cache or the context length, or a device
    /// operation fails.
    pub(crate) fn draft(
        &self,
        target: &GpuTransformer,
        hidden: &[f32],
        first: u32,
        len: usize,
        k: usize,
        pick: &mut dyn FnMut(&mut [f32]) -> u32,
    ) -> Result<Vec<u32>, CucaError> {
        let hp = &self.net.hp;
        if hidden.len() != hp.n_embd_out {
            return Err(err(format!(
                "drafter hidden state has {} floats, expected {}",
                hidden.len(),
                hp.n_embd_out
            )));
        }
        if len == 0 || len > target.kv_len || len >= self.context_length {
            return Err(err(format!(
                "cannot draft at position {len}: the target caches {} positions and the drafter's context is {}",
                target.kv_len, self.context_length
            )));
        }
        // Handles to the target's cache live for this call only.
        let ext = target.ext_kv(len)?;
        let pass = self.net.pass(None, len, 1);
        let mut no_cache: KvLayers = Vec::new();
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
                x = self.net.layer(il, x, &pass, &mut no_cache, Some(&ext))?;
            }
            let normed = rms(x, Some(&self.net.output_norm), hp.eps);
            let next = linear(normed.clone(), &self.post);
            let mut logits = read_f32(self.net.head(normed), "drafter logits")?;
            hidden = read_f32(next, "drafter hidden state")?;
            token = pick(&mut logits);
            drafts.push(token);
        }
        Ok(drafts)
    }
}

/// Read every weight of layer `il`.
fn load_layer(
    reader: &mut WeightReader<'_>,
    hp: &Hparams,
    il: usize,
    spec: &LayerSpec,
    layout: &LayerLayout,
) -> Result<LayerWeights, CucaError> {
    let n_embd = hp.n_embd;
    let (nh, nkv, d) = (spec.n_head, spec.n_kv_head, spec.head_dim);
    let q_width = nh * d;
    let kv_width = nkv * d;
    let blk = |part: &str| name::blk(il, part);
    let owns = spec.owns_kv(il);

    let qkv = match layout.qkv {
        Qkv::Fused => {
            if !owns {
                return Err(err(format!(
                    "layer {il}: a fused attn_qkv layer cannot share another layer's K/V"
                )));
            }
            QkvWeights::Fused(reader.proj(
                il,
                name::ATTN_QKV,
                [q_width + 2 * kv_width, n_embd],
                layout.biases.q,
            )?)
        }
        Qkv::Split => QkvWeights::Split {
            q: reader.proj(il, name::ATTN_Q, [q_width, n_embd], layout.biases.q)?,
            kv: if owns {
                Some((
                    reader.proj(il, name::ATTN_K, [kv_width, n_embd], layout.biases.k)?,
                    reader.proj(il, name::ATTN_V, [kv_width, n_embd], layout.biases.v)?,
                ))
            } else {
                None
            },
        },
    };
    let attn = AttnWeights {
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
        wo: reader.proj(il, name::ATTN_OUTPUT, [n_embd, q_width], layout.biases.o)?,
    };

    let ffn = match (layout.ffn, hp.moe) {
        (Ffn::Dense { ff, fused }, _) => FfnWeights::Dense {
            gate_up: if fused {
                GateUp::Fused {
                    up: reader.matrix(&blk(name::FFN_UP), 2 * ff, n_embd)?,
                    ff,
                }
            } else {
                GateUp::Split {
                    gate: reader.matrix(&blk(name::FFN_GATE), ff, n_embd)?,
                    up: reader.matrix(&blk(name::FFN_UP), ff, n_embd)?,
                }
            },
            down: reader.matrix(&blk(name::FFN_DOWN), n_embd, ff)?,
        },
        (Ffn::Moe { ff }, Some(spec)) => FfnWeights::Moe {
            spec,
            router: reader.matrix(&blk(name::FFN_GATE_INP), spec.n_expert, n_embd)?,
            gate: reader.experts(&blk(name::FFN_GATE_EXPS), spec.n_expert, ff, n_embd)?,
            up: reader.experts(&blk(name::FFN_UP_EXPS), spec.n_expert, ff, n_embd)?,
            down: reader.experts(&blk(name::FFN_DOWN_EXPS), spec.n_expert, n_embd, ff)?,
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
        attn,
        post_attn_norm: reader.opt_vector(
            layout.post_attn_norm,
            &blk(name::POST_ATTENTION_NORM),
            n_embd,
        )?,
        ffn_norm: reader.opt_vector(hp.behavior.pre_norms, &blk(name::FFN_NORM), n_embd)?,
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

impl Weights {
    #[expect(
        clippy::too_many_arguments,
        reason = "one pass's inputs, each used once; a struct would only rename them"
    )]
    fn forward_layers(
        &self,
        kv: &mut KvLayers,
        embeddings: Vec<f32>,
        per_layer: Option<Vec<f32>>,
        t: usize,
        start: usize,
        want: Want,
        hidden: Hidden,
    ) -> Result<Forward, CucaError> {
        let hp = &self.hp;
        let dev = &self.device;
        let eps = hp.eps;
        let mut x = Tensor::<2>::from_data(TensorData::new(embeddings, [t, hp.n_embd]), dev);

        // Gemma 4 per-layer inputs: projected embeddings merged with the
        // token-identity rows, `(ctx + tok) / sqrt(2)`.
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

        let pass = self.pass(ple, start, t);
        for il in 0..hp.n_layer {
            x = self.layer(il, x, &pass, kv, None)?;
        }

        let rows = match want {
            Want::None => {
                return Ok(Forward {
                    logits: None,
                    hidden: None,
                });
            }
            Want::Last => x.narrow(0, t - 1, 1),
            Want::All => x,
        };
        let normed = rms(rows, Some(&self.output_norm), eps);
        // The head is queued before the hidden-state readback so the device
        // keeps working through it instead of idling at that sync point.
        let (hidden, logits) = if hidden == Hidden::PostNorm {
            let logits = self.head(normed.clone());
            (Some(read_f32(normed, "hidden states")?), logits)
        } else {
            (None, self.head(normed))
        };
        let logits = read_f32(logits, "logits")?;
        Ok(Forward {
            logits: Some(logits),
            hidden,
        })
    }

    /// LM head over final-norm output rows: matmul, logit scale, soft-cap.
    fn head(&self, normed: Tensor<2>) -> Tensor<2> {
        let hp = &self.hp;
        let mut logits = linear(normed, &self.lm_head);
        if hp.logit_scale != 1.0 {
            logits = logits.div_scalar(f64::from(hp.logit_scale));
        }
        if let Some(cap) = hp.logit_softcap {
            logits = logits.div_scalar(cap).tanh().mul_scalar(cap);
        }
        logits
    }

    /// The RoPE rows, visible ranges and masks shared by the layers of one
    /// forward pass of `t` queries at `start..`.
    fn pass(&self, ple: Option<Tensor<3>>, start: usize, t: usize) -> Pass {
        let hp = &self.hp;
        let ropes = self
            .ropes
            .iter()
            .map(|table| {
                let [_, half] = table.cos.dims();
                let rows = |m: &Tensor<2>| m.clone().narrow(0, start, t).reshape([1, 1, t, half]);
                RopeRows {
                    cos: rows(&table.cos),
                    sin: rows(&table.sin),
                }
            })
            .collect();

        let hi = start + t;
        let mut masks: Vec<MaskInput> = Vec::new();
        for spec in &hp.layers {
            if masks.iter().any(|m| m.window == spec.window) {
                continue;
            }
            let lo = match spec.window {
                Some(w) => start.saturating_sub(w - 1),
                None => 0,
            };
            let mask = (t > 1).then(|| {
                let values = attention_mask(start, t, lo, hi, spec.window);
                Tensor::from_data(TensorData::new(values, [1, 1, t, hi - lo]), &self.device)
            });
            masks.push(MaskInput {
                window: spec.window,
                lo,
                mask,
            });
        }
        Pass {
            start,
            t,
            ple,
            ropes,
            masks,
        }
    }

    /// One transformer block over `pass.t` positions starting at `pass.start`.
    fn layer(
        &self,
        il: usize,
        x: Tensor<2>,
        pass: &Pass,
        kv: &mut KvLayers,
        ext: Option<&GpuExtKv>,
    ) -> Result<Tensor<2>, CucaError> {
        let hp = &self.hp;
        let spec = &hp.layers[il];
        let lw = &self.layers[il];
        let eps = hp.eps;
        let (nh, nkv, d) = (spec.n_head, spec.n_kv_head, spec.head_dim);
        let (start, t) = (pass.start, pass.t);
        let hi = start + t;
        let scaled = |branch: Tensor<2>| {
            if hp.residual_scale == 1.0 {
                branch
            } else {
                branch.mul_scalar(f64::from(hp.residual_scale))
            }
        };
        let rotate = |m: Tensor<4>| match spec.rope {
            Some(r) => rope(m, &pass.ropes[r], &hp.ropes[r]),
            None => m,
        };

        let h = match &lw.attn_norm {
            Some(w) => rms(x.clone(), Some(w), eps),
            None => x.clone(),
        };
        let (q, kv_proj) = lw.attn.project(h, nh, nkv, d);
        let full_qk = hp.behavior.qk_norm_full;
        let q = match &lw.attn.q_norm {
            Some(w) if full_qk => rms(q, Some(w), eps),
            _ => q,
        };
        let q = q.reshape([1, t, nh, d]);
        let q = match &lw.attn.q_norm {
            Some(w) if !full_qk => rms(q, Some(w), eps),
            _ => q,
        };
        let q = rotate(q.swap_dims(1, 2));

        if let Some((k, v)) = kv_proj {
            let k = match &lw.attn.k_norm {
                Some(w) if full_qk => rms(k, Some(w), eps),
                _ => k,
            };
            let k = k.reshape([1, t, nkv, d]);
            let k = match &lw.attn.k_norm {
                Some(w) if !full_qk => rms(k, Some(w), eps),
                _ => k,
            };
            let k = rotate(k.swap_dims(1, 2));
            let v = v.reshape([1, t, nkv, d]);
            let v = if lw.attn.v_norm { rms(v, None, eps) } else { v };
            let v = v.swap_dims(1, 2);
            // Take the cache out so the assignment owns the only handle and
            // writes in place.
            let (cache_k, cache_v) = kv
                .get_mut(il)
                .and_then(Option::take)
                .ok_or_else(|| err(format!("layer {il} owns no K/V cache")))?;
            let at = cache_ranges(nkv, start..hi, d);
            kv[il] = Some((
                cache_k.slice_assign(at.clone(), k),
                cache_v.slice_assign(at, v),
            ));
        }

        let src = spec.kv_source;
        let attn = if src == usize::MAX {
            // A drafter layer: attend to the first `ext.len` positions of the
            // target's cache (the sliding layers to the window before the
            // query), with no mask since the drafter has a single query.
            let ext = ext.ok_or_else(|| {
                err(format!(
                    "layer {il}: a drafter layer needs the target's K/V"
                ))
            })?;
            let (cache_k, cache_v) = if spec.window.is_some() {
                &ext.sliding
            } else {
                &ext.full
            };
            let lo = match spec.window {
                Some(w) => start.saturating_sub(w - 1).min(ext.len.saturating_sub(1)),
                None => 0,
            };
            let at = cache_ranges(nkv, lo..ext.len, d);
            let k = cache_k.clone().slice(at.clone());
            let v = cache_v.clone().slice(at);
            attend(q, k, v, None, spec.scale, hp.attn_softcap)
        } else {
            let visible = pass
                .masks
                .iter()
                .find(|m| m.window == spec.window)
                .ok_or_else(|| err(format!("layer {il}: no attention mask for its window")))?;
            let (cache_k, cache_v) = kv
                .get(src)
                .and_then(Option::as_ref)
                .ok_or_else(|| err(format!("layer {src} owns no K/V cache")))?;
            let src_spec = hp
                .layers
                .get(src)
                .ok_or_else(|| err(format!("layer {il}: K/V source layer {src} out of range")))?;
            let at = cache_ranges(src_spec.n_kv_head, visible.lo..hi, src_spec.head_dim);
            let k = cache_k.clone().slice(at.clone());
            let v = cache_v.clone().slice(at);
            attend(q, k, v, visible.mask.clone(), spec.scale, hp.attn_softcap)
        }
        .swap_dims(1, 2)
        .reshape([t, nh * d]);
        let mut attn = lw.attn.wo.forward(attn);
        if let Some(w) = &lw.post_attn_norm {
            attn = rms(attn, Some(w), eps);
        }
        let x = x + scaled(attn);

        let f = match &lw.ffn_norm {
            Some(w) => rms(x.clone(), Some(w), eps),
            None => x.clone(),
        };
        let mut f = lw.ffn.forward(f, hp.act(), &self.device)?;
        if let Some(w) = &lw.post_ffw_norm {
            f = rms(f, Some(w), eps);
        }
        let mut x = x + scaled(f);

        if let (Some(layer), Some(ple), Some(dim)) = (&lw.ple, &pass.ple, hp.ple_dim) {
            let slice = ple.clone().narrow(1, il, 1).reshape([t, dim]);
            let g = gelu(linear(x.clone(), &layer.inp_gate)) * slice;
            x = x + rms(linear(g, &layer.proj), Some(&layer.post_norm), eps);
        }
        Ok(match lw.out_scale {
            Some(s) => x.mul_scalar(s),
            None => x,
        })
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
                gate,
                up,
                down,
            } => {
                let [t, n_embd] = f.dims();
                let logits = read_f32(linear(f.clone(), router), "router logits")?;
                let routing = route(&logits, t, spec);
                // One expert on `x`; `weights` (one per row, multi-row passes
                // only) scale the rows of the hidden state, which commutes
                // with the linear down projection. Multiplying the down
                // projection's output by that device tensor made a fused
                // kernel lose the Vulkan device (burn 0.22, tiny Granite MoE
                // test), so the scale stays upstream of the last matmul. The
                // single-token branch below multiplies the output by a host
                // scalar instead, a form the live suite ran without loss.
                let expert = |e: usize,
                              x: Tensor<2>,
                              weights: Option<Tensor<2>>|
                 -> Result<Tensor<2>, CucaError> {
                    let (gate, up, down) = gate
                        .get(e)
                        .zip(up.get(e))
                        .zip(down.get(e))
                        .map(|((g, u), d)| (g, u, d))
                        .ok_or_else(|| err(format!("expert {e} out of range")))?;
                    let hidden = activate(linear(x.clone(), gate), act) * linear(x, up);
                    let hidden = match weights {
                        Some(w) => hidden * w,
                        None => hidden,
                    };
                    Ok(linear(hidden, down))
                };
                let mut out = Tensor::<2>::zeros([t, n_embd], device);
                for (e, picks) in routing.per_expert.iter().enumerate() {
                    let Some(&(_, weight)) = picks.first() else {
                        continue;
                    };
                    if t == 1 {
                        // The one token reaches the expert whole: no gather
                        // and no scatter.
                        let y = expert(e, f.clone(), None)?;
                        out = out + y.mul_scalar(f64::from(weight));
                        continue;
                    }
                    let ids: Vec<i32> = picks.iter().map(|p| p.0 as i32).collect();
                    let weights: Vec<f32> = picks.iter().map(|p| p.1).collect();
                    let m = ids.len();
                    let ids = Tensor::<1, Int>::from_ints(TensorData::new(ids, [m]), device);
                    let weights = Tensor::<2>::from_data(TensorData::new(weights, [m, 1]), device);
                    let y = expert(e, f.clone().select(0, ids.clone()), Some(weights))?;
                    out = out.select_assign(0, ids, y, IndexingUpdateOp::Add);
                }
                Ok(out)
            }
        }
    }
}

/// The cache region of positions `positions` of every KV head.
fn cache_ranges(n_kv_head: usize, positions: Range<usize>, d: usize) -> [Range<usize>; 4] {
    [0..1, 0..n_kv_head, positions, 0..d]
}

/// Read a device tensor back to the host as f32 values, row-major.
fn read_f32<const D: usize>(tensor: Tensor<D>, what: &str) -> Result<Vec<f32>, CucaError> {
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
/// and the `[out, in]` weight part `w`: it needs `in` to be a multiple of the
/// plane width and `out` a multiple of the plane width times the values
/// packed per `u32` word of the weight (8 for Q4_0, 4 for Q8_0, 1 for f32).
/// The conditions are checked against [`MAX_PLANE`], the worst case.
fn vecmat_fits(w: &Tensor<2>) -> bool {
    let [out, inn] = w.dims();
    let per_word = match w.dtype() {
        DType::QFloat(scheme) if Some(scheme) == quant_scheme(GgmlDType::Q4_0) => 8,
        DType::QFloat(_) => 4,
        _ => 1,
    };
    out.is_multiple_of(MAX_PLANE * per_word) && inn.is_multiple_of(MAX_PLANE)
}

/// `x [t, in]` times the `[out, in]` weight `w`, transposed: `[t, out]`. A
/// quantized `w` is read by the matmul kernel directly (dequantized in
/// registers); the transposes only swap strides. A weight in several row
/// chunks runs one product per chunk and concatenates the outputs.
///
/// The operand order depends on `t` and on the shape of each chunk, working
/// around burn 0.22 kernel bugs seen on Vulkan (RX 9070 XT), all pinned by
/// `linear_matches_host_across_shapes`, `gpu_matches_cpu_on_every_architecture`
/// and the live suite:
///
/// - `t > 1` (prefill): `(w · xᵀ)ᵀ`, the weight on the left in its stored
///   layout. With the stride-swapped quantized weight on the right, some
///   autotune candidates lose the device.
/// - `t == 1` (decode) and [`vecmat_fits`]: `x · wᵀ`, which selects the
///   `vecmat` kernel that reads the quantized weight once, in registers.
///   `(w · xᵀ)ᵀ` with one column selects a matrix-vector kernel that returns
///   wrong values for a quantized left operand.
/// - `t == 1` for any other shape: `vecmat`'s autotune entry is shared by
///   every shape of the same power-of-two bucket, so a candidate tuned for
///   `out = 96` is replayed for `out = 128` and fails with "could not find
///   supported vector size". The token is duplicated into two rows, the
///   product runs as for a prefill, and the second row is dropped. Burn
///   dequantizes the whole weight to f32 for that product, so such a shape
///   decodes several times slower than a `vecmat` one.
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
fn linear_part(x: Tensor<2>, w: &Tensor<2>) -> Tensor<2> {
    if x.dims()[0] > 1 {
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

/// Grouped-query attention, as the CPU `attend`: `q` is `[1, n_head, t, d]`,
/// `k` and `v` are `[1, n_kv, l, d]`, `mask` the additive `[1, 1, t, l]`
/// mask. The scores are scaled by `scale` and, with `softcap`, squashed as
/// `cap * tanh(score / cap)` before the mask. The `n_head / n_kv` query heads
/// of one KV head are stacked into the rows of one matmul. Returns `[1,
/// n_head, t, d]`.
fn attend(
    q: Tensor<4>,
    k: Tensor<4>,
    v: Tensor<4>,
    mask: Option<Tensor<4>>,
    scale: f32,
    softcap: Option<f64>,
) -> Tensor<4> {
    let [_, n_head, t, d] = q.dims();
    let [_, n_kv, l, _] = k.dims();
    let rows = n_head / n_kv * t;
    let mut scores = q.reshape([1, n_kv, rows, d]).matmul(k.transpose());
    if scale != 1.0 {
        scores = scores.mul_scalar(f64::from(scale));
    }
    if let Some(cap) = softcap {
        scores = scores.div_scalar(cap).tanh().mul_scalar(cap);
    }
    let scores = match mask {
        Some(mask) => (scores.reshape([1, n_head, t, l]) + mask).reshape([1, n_kv, rows, l]),
        None => scores,
    };
    activation::softmax(scores, 3)
        .matmul(v)
        .reshape([1, n_head, t, d])
}

#[cfg(test)]
mod tests {
    use std::io::BufReader;

    use super::super::cpu::Transformer;
    use super::super::tiny::{
        PROMPT, TinyModel, TinyOptions, TinySpec, write_spec, write_tiny_model_with,
    };
    use super::*;

    fn f16_bytes(v: f32) -> [u8; 2] {
        half::f16::from_f32(v).to_bits().to_le_bytes()
    }

    /// The `index`-th signed element of a packed values region, decoded by
    /// the documented bit layout: `bits`-wide fields, `32 / bits` per
    /// little-endian `u32`, lowest field first, two's complement.
    fn packed_element(values: &[u8], bits: u32, index: usize) -> i32 {
        let per_word = (32 / bits) as usize;
        let at = index / per_word * 4;
        let word = u32::from_le_bytes(values[at..at + 4].try_into().unwrap());
        let field = (word >> (bits * (index % per_word) as u32)) & ((1 << bits) - 1);
        // Sign-extend the field.
        ((field << (32 - bits)) as i32) >> (32 - bits)
    }

    /// Two Q4_0 blocks (one row of 64) whose nibbles cover 0..16 in both
    /// halves, so every value -8..=7 appears.
    fn q4_0_row() -> Vec<u8> {
        let mut raw = Vec::new();
        for scale in [0.5f32, -0.25] {
            raw.extend(f16_bytes(scale));
            raw.extend((0..16u8).map(|j| j | ((15 - j) << 4)));
        }
        raw
    }

    #[test]
    fn pack_q4_0_matches_dequant_row() {
        let raw = q4_0_row();
        let packed = pack_blocks(GgmlDType::Q4_0, &raw, 1, 64).unwrap();
        // 64 nibbles in 8 words, then 2 scales.
        assert_eq!(packed.len(), 32 + 4);
        let mut expected = [0f32; 64];
        dequant_row(GgmlDType::Q4_0, &raw, 1.0, &mut expected).unwrap();
        for (i, &e) in expected.iter().enumerate() {
            let scale = if i < 32 { 0.5 } else { -0.25 };
            assert_eq!(
                packed_element(&packed, 4, i) as f32,
                e / scale,
                "element {i}"
            );
        }
        assert_eq!(&packed[32..34], &raw[0..2]);
        assert_eq!(&packed[34..36], &raw[18..20]);
    }

    #[test]
    fn pack_q8_0_matches_dequant_row() {
        // Two rows of one block each.
        let mut raw = Vec::new();
        for (scale, salt) in [(0.25f32, 3i32), (2.0, 11)] {
            raw.extend(f16_bytes(scale));
            raw.extend((0..32).map(|j| ((j * 8 + salt) % 256 - 128) as i8 as u8));
        }
        let packed = pack_blocks(GgmlDType::Q8_0, &raw, 2, 32).unwrap();
        assert_eq!(packed.len(), 64 + 4);
        let mut expected = [0f32; 64];
        dequant_row(GgmlDType::Q8_0, &raw, 1.0, &mut expected).unwrap();
        for (i, &e) in expected.iter().enumerate() {
            let scale = if i < 32 { 0.25 } else { 2.0 };
            assert_eq!(
                packed_element(&packed, 8, i) as f32,
                e / scale,
                "element {i}"
            );
        }
        assert_eq!(&packed[64..66], &raw[0..2]);
        assert_eq!(&packed[66..68], &raw[34..36]);
    }

    #[test]
    fn only_q4_0_and_q8_0_stay_quantized() {
        assert!(quant_scheme(GgmlDType::Q4_0).is_some());
        assert!(quant_scheme(GgmlDType::Q8_0).is_some());
        for dtype in [
            GgmlDType::F32,
            GgmlDType::F16,
            GgmlDType::BF16,
            GgmlDType::Q4_1,
            GgmlDType::Q5_0,
            GgmlDType::Q5_1,
            GgmlDType::Q4K,
            GgmlDType::Q6K,
        ] {
            assert!(quant_scheme(dtype).is_none(), "{dtype:?}");
            assert!(pack_blocks(dtype, &[], 0, 32).is_err(), "{dtype:?}");
        }
        // A byte count that is not the tensor's block count.
        assert!(pack_blocks(GgmlDType::Q4_0, &q4_0_row()[..35], 1, 64).is_err());
    }

    /// The GPU device under test, from `CUCA_LOCAL_DEVICE`; `None` (skip)
    /// when unset or `cpu`.
    fn test_device() -> Option<LocalDevice> {
        let device: LocalDevice = std::env::var("CUCA_LOCAL_DEVICE")
            .ok()?
            .parse()
            .expect("CUCA_LOCAL_DEVICE");
        (device != LocalDevice::Cpu).then_some(device)
    }

    fn load_both(model: &TinyModel, device: LocalDevice) -> (Transformer, GpuTransformer) {
        let mut reader = BufReader::new(File::open(&model.path).unwrap());
        let content = gguf_file::Content::read(&mut reader).unwrap();
        (
            Transformer::load_dequantized(&content, &model.path, 32).unwrap(),
            GpuTransformer::load(&content, &model.path, 32, device).unwrap(),
        )
    }

    fn assert_matches_cpu(cpu: &[f32], gpu: &[f32], what: &str) {
        assert_eq!(cpu.len(), gpu.len(), "{what}");
        for (i, (c, g)) in cpu.iter().zip(gpu).enumerate() {
            assert!(
                (g - c).abs() <= 1e-3 + 1e-2 * c.abs(),
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
        let c = cpu.forward(tokens, start, want).unwrap().unwrap();
        let g = gpu.forward(tokens, start, want).unwrap().unwrap();
        assert_matches_cpu(&c, &g, what);
    }

    /// The GPU engine must reproduce the CPU engine on one tiny model:
    /// single-shot prefill, decode, every-position logits, chunked prefill
    /// and truncate-and-replay. The reference holds its weights dequantized:
    /// candle's quantized matmul rounds the activations to Q8_0, which the
    /// GPU kernels (f32 activations) do not, and over several random-weight
    /// layers that alone moves logits by several percent.
    fn check_model(model: &TinyModel, device: LocalDevice, label: &str) {
        let n = PROMPT.len();

        // (a) single-shot prefill, then (c) one decode step.
        let (mut cpu, mut gpu) = load_both(model, device);
        compare(
            &mut cpu,
            &mut gpu,
            &PROMPT,
            0,
            Want::Last,
            &format!("{label} prefill"),
        );
        compare(
            &mut cpu,
            &mut gpu,
            &[7],
            n,
            Want::Last,
            &format!("{label} decode"),
        );

        // (e) every-position logits, with a prefix already cached.
        let (mut cpu, mut gpu) = load_both(model, device);
        assert!(gpu.forward(&PROMPT[..4], 0, Want::None).unwrap().is_none());
        assert!(cpu.forward(&PROMPT[..4], 0, Want::None).unwrap().is_none());
        compare(
            &mut cpu,
            &mut gpu,
            &PROMPT[4..],
            4,
            Want::All,
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
        compare(
            &mut cpu,
            &mut gpu,
            &PROMPT[start..],
            start,
            Want::Last,
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
        compare(
            &mut cpu,
            &mut gpu,
            &[11, 13, 15, 14, 13, 12],
            5,
            Want::Last,
            &format!("{label} replay"),
        );
    }

    #[test]
    fn gpu_matches_cpu_on_the_tiny_gemma4() {
        let Some(device) = test_device() else {
            println!("SKIP: set CUCA_LOCAL_DEVICE to a GPU device");
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
            println!("SKIP: set CUCA_LOCAL_DEVICE to a GPU device");
            return;
        };
        for spec in TinySpec::all() {
            check_model(&write_spec(&spec), device, spec.arch);
        }
    }

    /// A model whose matmul weights are float or Q8_0 matches the dequantized
    /// CPU reference as closely as a Q4_0 one: no requantization happens.
    /// A model in Q4_1, Q5_0 or Q5_1, which the engine requantizes to Q8_0,
    /// is within the Q8_0 rounding of it: at most 8% of the largest logit
    /// (measured up to 4.5%: a random-weight model amplifies the rounding).
    #[test]
    fn gpu_matches_cpu_on_other_weight_dtypes() {
        let Some(device) = test_device() else {
            println!("SKIP: set CUCA_LOCAL_DEVICE to a GPU device");
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
                    assert_matches_cpu(&c, &g, &format!("{dtype:?} {what}"));
                }
            }
        }
    }

    /// `linear` against a host matmul of the dequantized weight, over the
    /// shapes that exposed burn 0.22 Vulkan bugs. The order replays cached
    /// autotune entries across shapes of one power-of-two bucket (`out = 96`
    /// then `128`) and mixes one-row inputs with weights the `vecmat` kernel
    /// can launch (`512 x 1024`) and cannot (`96 x 64`), and small odd row
    /// counts.
    #[test]
    fn linear_matches_host_across_shapes() {
        let Some(device) = test_device() else {
            println!("SKIP: set CUCA_LOCAL_DEVICE to a GPU device");
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
                for t in [1usize, 2, 3, 1, 7] {
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
            println!("SKIP: set CUCA_LOCAL_DEVICE to a GPU device");
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
            assert_eq!(w.parts[0].dtype(), DType::F32, "{dtype:?}");
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
            println!("SKIP: set CUCA_LOCAL_DEVICE to a GPU device");
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
            assert_eq!(
                w.parts[0].dtype(),
                DType::QFloat(quant_scheme(GgmlDType::Q8_0).unwrap()),
                "{dtype:?} is held as Q8_0"
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

    /// A weight split into row chunks multiplies like the whole weight, for
    /// every storage class and for a last chunk that is narrow.
    #[test]
    fn chunked_weight_matches_whole_weight() {
        use candle_core::quantized::QTensor;

        let Some(device) = test_device() else {
            println!("SKIP: set CUCA_LOCAL_DEVICE to a GPU device");
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
            let split =
                upload_matrix_with("w", dtype, &raw, rows, cols, &device, limit, false).unwrap();
            let sizes: Vec<usize> = split.parts.iter().map(|p| p.dims()[0]).collect();
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

    fn shape(rows: usize, cols: usize, class: SchemeClass) -> QuantShape {
        QuantShape { rows, cols, class }
    }

    fn named(shapes: &[(&str, QuantShape)]) -> Vec<(String, QuantShape)> {
        shapes.iter().map(|(n, s)| (n.to_string(), *s)).collect()
    }

    /// After the plan, no Q4 and Q8 weight that remain quantized may share an
    /// autotune key.
    fn assert_plan_separates(own: &[(String, QuantShape)], peers: &[QuantShape]) {
        let dense = plan_dense(own, peers);
        let kept: Vec<&QuantShape> = own
            .iter()
            .filter(|(n, _)| !dense.contains(n))
            .map(|(_, s)| s)
            .collect();
        for a in &kept {
            for b in kept.iter().copied().chain(peers) {
                assert!(
                    a.class == b.class || !a.may_share_key(b),
                    "{a:?} and {b:?} may share a key"
                );
            }
        }
    }

    #[test]
    fn plan_dense_converts_only_the_minority_scheme_near_the_other() {
        use SchemeClass::{Q4, Q8};
        // One scheme: nothing to do, however similar the shapes.
        let q4 = named(&[("a", shape(512, 2048, Q4)), ("b", shape(2048, 8192, Q4))]);
        assert!(plan_dense(&q4, &[]).is_empty());
        let q8 = named(&[("a", shape(512, 2048, Q8)), ("b", shape(512, 2048, Q8))]);
        assert!(plan_dense(&q8, &[]).is_empty());

        // Q4 majority: a Q8 weight is converted when a Q4 weight is near (a
        // Q4_1 attention layer of a Q4_0 file, a Q6_K head of a similar
        // width) and kept when none is (a narrow projection).
        let mut mixed = named(&[
            ("q4.up", shape(16384, 3072, Q4)),
            ("q4.down", shape(3072, 8192, Q4)),
            ("q8.layer", shape(3072, 8192, Q8)),
            ("q8.head", shape(32064, 3072, Q8)),
            ("q8.far", shape(1024, 3072, Q8)),
        ]);
        // The other layers of the Q4_0 file.
        mixed.extend((0..30).map(|i| (format!("q4.layer{i}"), shape(16384, 3072, Q4))));
        let dense = plan_dense(&mixed, &[]);
        let mut names: Vec<&str> = dense.iter().map(String::as_str).collect();
        names.sort();
        assert_eq!(names, ["q8.head", "q8.layer"]);
        assert_plan_separates(&mixed, &[]);

        // Q8 majority: the Q4 weight near a Q8 one is converted instead.
        let flipped = named(&[
            ("q8.a", shape(4096, 4096, Q8)),
            ("q8.b", shape(4096, 4096, Q8)),
            ("q4.a", shape(4096, 2048, Q4)),
            ("q4.far", shape(64, 64, Q4)),
        ]);
        let dense = plan_dense(&flipped, &[]);
        assert_eq!(dense.len(), 1);
        assert!(dense.contains("q4.a"));

        // The factor-2.5 boundary of the anchor bucketing.
        let edge = |rows| named(&[("q4", shape(1000, 1000, Q4)), ("q8", shape(rows, 1000, Q8))]);
        assert_eq!(plan_dense(&edge(2400), &[]).len(), 1, "2.4x is near");
        assert!(plan_dense(&edge(2600), &[]).is_empty(), "2.6x is far");

        // A drafter's own weights give way to the loaded target's whatever
        // their size.
        let peers = [shape(2048, 1536, Q8)];
        let own = named(&[("big", shape(4096, 1536, Q4)), ("far", shape(64, 64, Q4))]);
        assert_eq!(plan_dense(&own, &peers).len(), 1);
        assert!(plan_dense(&own, &peers).contains("big"));
        assert_plan_separates(&own, &peers);
    }

    /// The tensors the plan sees in a real file layout: a Q4_0 model with a
    /// Q8_0 output head close to its projections.
    #[test]
    fn plan_dense_reads_the_shapes_of_a_gguf_file() {
        let spec = TinySpec {
            embd_dtype: GgmlDType::Q8_0,
            ..TinySpec::llama()
        };
        let model = write_spec(&spec);
        let mut reader = BufReader::new(File::open(&model.path).unwrap());
        let content = gguf_file::Content::read(&mut reader).unwrap();
        let shapes = quant_shapes(&content);
        let names: Vec<&str> = shapes.iter().map(|(n, _)| n.as_str()).collect();
        // The token embedding is a lookup table here (the head is a
        // separate `output.weight`), so it does not count.
        assert!(!names.contains(&"token_embd.weight"), "{names:?}");
        assert!(names.contains(&"output.weight"), "{names:?}");
        assert!(names.contains(&"blk.0.ffn_up.weight"), "{names:?}");
        // Q8_0 head [16, 64] vs Q4_0 K/V projection [32, 64].
        let dense = plan_dense(&shapes, &[]);
        assert_eq!(dense.iter().collect::<Vec<_>>(), ["output.weight"]);
        assert_plan_separates(&shapes, &[]);
    }

    /// The bytes of a synthetic `[out, inn]` weight of `dtype`.
    fn synthetic_weight(dtype: GgmlDType, out: usize, inn: usize) -> Vec<u8> {
        let mut raw = Vec::with_capacity(out * inn / dtype.block_size() * dtype.type_size());
        for b in 0..out * inn / 32 {
            raw.extend(f16_bytes(0.05 + (b % 7) as f32 * 0.01));
            raw.extend((0..dtype.type_size() - 2).map(|j| ((b * 31 + j * 17 + 5) % 251) as u8));
        }
        raw
    }

    /// burn 0.22's matmul autotune key omits the quantization scheme, so a
    /// kernel tuned for a Q8_0 weight is replayed on a Q4_0 weight of the same
    /// shape bucket and fails ("quantized view float vector size 4 must be a
    /// positive multiple of num_quants 8", or a shared-memory budget). The
    /// weights below are the shape classes of Llama 3.2 and Phi 3.5 files
    /// (Q4_0 with a few Q4_1 layers and a Q6_K head) plus small ones with
    /// the same residues, uploaded as the loader does (the plan applied),
    /// Q8 weights first, then multiplied for prefill lengths of every
    /// residue class and for decode in one process. The autotune cache is
    /// per process and device, so every test of this module shares it: a
    /// Q8 weight tuned here can reach a Q4 weight of the same bucket in
    /// another test, which is why the plan is applied to every family.
    #[test]
    fn mixed_scheme_weights_run_in_every_order() {
        use GgmlDType::{Q4_0, Q8_0};

        let Some(device) = test_device() else {
            println!("SKIP: set CUCA_LOCAL_DEVICE to a GPU device");
            return;
        };
        let device = open_device(device).unwrap();
        type Family = (&'static str, Vec<(usize, usize, GgmlDType)>);
        let families: [Family; 3] = [
            (
                "phi3.5",
                vec![
                    (3072, 8192, Q8_0),
                    (32064, 3072, Q8_0),
                    (9216, 3072, Q4_0),
                    (3072, 3072, Q4_0),
                    (16384, 3072, Q4_0),
                    (3072, 8192, Q4_0),
                ],
            ),
            (
                "llama3.2",
                vec![
                    (512, 2048, Q8_0),
                    (2048, 8192, Q8_0),
                    (2048, 2048, Q4_0),
                    (512, 2048, Q4_0),
                    (8192, 2048, Q4_0),
                    (2048, 8192, Q4_0),
                ],
            ),
            (
                "small",
                vec![
                    (96, 64, Q8_0),
                    (16, 64, Q8_0),
                    (64, 128, Q8_0),
                    (96, 64, Q4_0),
                    (64, 96, Q4_0),
                    (32, 64, Q4_0),
                    (64, 64, Q4_0),
                    (128, 64, Q4_0),
                ],
            ),
        ];
        let ts = [1usize, 2, 3, 5, 7, 8, 9, 17, 31, 64, 80, 101, 256, 1];
        for (family, weights) in &families {
            let shapes: Vec<(String, QuantShape)> = weights
                .iter()
                .enumerate()
                .map(|(i, &(rows, cols, d))| {
                    let class = scheme_class(d).unwrap();
                    (format!("{family}.{i}"), QuantShape { rows, cols, class })
                })
                .collect();
            let dense = plan_dense(&shapes, &[]);
            assert!(!dense.is_empty(), "{family}: the plan converts something");
            let uploaded: Vec<(Weight, Vec<u8>)> = shapes
                .iter()
                .zip(weights)
                .map(|((tensor, _), &(rows, cols, d))| {
                    let raw = synthetic_weight(d, rows, cols);
                    let w = upload_matrix_with(
                        tensor,
                        d,
                        &raw,
                        rows,
                        cols,
                        &device,
                        MAX_PART_BYTES,
                        dense.contains(tensor),
                    )
                    .unwrap();
                    (w, raw)
                })
                .collect();
            for &t in &ts {
                for (i, ((w, raw), &(rows, cols, d))) in uploaded.iter().zip(weights).enumerate() {
                    let x: Vec<f32> = (0..t * cols)
                        .map(|j| ((j * 7 + 3) % 13) as f32 * 0.1 - 0.6)
                        .collect();
                    let xt = Tensor::<2>::from_data(TensorData::new(x.clone(), [t, cols]), &device);
                    let got = read_f32(linear(xt, w), "linear")
                        .unwrap_or_else(|e| panic!("{family} weight {i} {rows}x{cols} t={t}: {e}"));
                    let row_bytes = cols / d.block_size() * d.type_size();
                    let mut host = vec![0f32; cols];
                    for o in (0..rows).step_by(rows / 7 + 1) {
                        dequant_row(d, &raw[o * row_bytes..(o + 1) * row_bytes], 1.0, &mut host)
                            .unwrap();
                        for r in 0..t.min(3) {
                            let e: f32 = (0..cols).map(|j| host[j] * x[r * cols + j]).sum();
                            let g = got[r * rows + o];
                            assert!(
                                (g - e).abs() <= 2e-3 * (1.0 + e.abs()),
                                "{family} weight {i} {d:?} {rows}x{cols} t={t}: row {r} col {o}: gpu {g} vs host {e}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// The hidden rows `forward_ext` returns (the final-norm output the head
    /// reads) match the CPU's, for the last row and for every row.
    #[test]
    fn forward_ext_hidden_rows_match_cpu() {
        let Some(device) = test_device() else {
            println!("SKIP: set CUCA_LOCAL_DEVICE to a GPU device");
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
            assert_matches_cpu(&ch, &gh, "hidden rows");
            assert_matches_cpu(&c.logits.unwrap(), &g.logits.unwrap(), "logits");
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
            println!("SKIP: set CUCA_LOCAL_DEVICE to a GPU device");
            return;
        };
        let model = write_tiny_model_with(TinyOptions::default());
        let drafter = write_tiny_drafter();
        let mut reader = BufReader::new(File::open(&drafter.path).unwrap());
        let content = gguf_file::Content::read(&mut reader).unwrap();
        let schedule = [5u32, 9, 2, 7];
        let k = schedule.len();

        // A `pick` that records every step's logits and answers from
        // `schedule`.
        fn recorder<'a>(
            picked: &'a mut Vec<Vec<f32>>,
            schedule: &'a [u32],
        ) -> impl FnMut(&mut [f32]) -> u32 + 'a {
            move |logits| {
                picked.push(logits.to_vec());
                schedule.get(picked.len() - 1).copied().unwrap_or(0)
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
            let cpu = super::super::Model::Cpu(cpu);
            let cpu_drafter = Drafter::open(&content, &drafter.path, &cpu, 32).unwrap();
            let gpu_drafter = GpuDrafter::open(&content, &drafter.path, &gpu, 32).unwrap();

            let mut c_logits = Vec::new();
            let c_tokens = cpu_drafter
                .draft(
                    &cpu,
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
            println!("SKIP: set CUCA_LOCAL_DEVICE to a GPU device");
            return;
        };
        let model = write_tiny_model_with(TinyOptions::default());
        let drafter = write_tiny_drafter();
        let read = |m: &TinyModel| {
            let mut reader = BufReader::new(File::open(&m.path).unwrap());
            gguf_file::Content::read(&mut reader).unwrap()
        };
        let (_, mut gpu) = load_both(&model, device);
        let (model_content, drafter_content) = (read(&model), read(&drafter));

        let e = GpuDrafter::open(&model_content, &model.path, &gpu, 32)
            .err()
            .unwrap();
        assert!(e.to_string().contains("not a drafter"), "{e}");
        let e = GpuTransformer::load(&drafter_content, &drafter.path, 32, device)
            .err()
            .unwrap();
        assert!(e.to_string().contains("drafter"), "{e}");

        let net = GpuDrafter::open(&drafter_content, &drafter.path, &gpu, 32).unwrap();
        let mut pick = |_: &mut [f32]| 0u32;
        let hidden = vec![0.1f32; gpu.n_embd()];
        gpu.forward(&PROMPT[..5], 0, Want::None).unwrap();
        assert!(
            net.draft(&gpu, &hidden, 1, 6, 2, &mut pick).is_err(),
            "beyond the cache"
        );
        assert!(
            net.draft(&gpu, &hidden, 1, 0, 2, &mut pick).is_err(),
            "empty cache"
        );
        assert!(
            net.draft(&gpu, &hidden[1..], 1, 5, 2, &mut pick).is_err(),
            "hidden width"
        );
        assert_eq!(
            net.draft(&gpu, &hidden, 1, 5, 2, &mut pick).unwrap().len(),
            2
        );
    }
}
