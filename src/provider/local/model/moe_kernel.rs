//! Fused mixture-of-experts kernels for the GPU engine (CubeCL).
//!
//! burn's quantized matmuls cannot gather the weights of a data-dependent
//! subset of experts (`q_select` dequantizes), so a MoE layer ran one
//! matmul triple per selected expert, and its routing needed a host readback
//! per layer. These kernels read the experts' weights straight from a
//! per-layer *pool* of expert slots held as one `u32` tensor, indexed by the
//! router's own output:
//!
//! 1. [`MoeKernels::moe_route`]: softmax and top-`k` of the router logits per
//!    token, as `[ids (tokens * k) | weights (tokens * k)]` in one f32 tensor;
//! 2. [`MoeKernels::moe_up`]: `act(gate . x) * (up . x)` for every (token,
//!    expert) pair;
//! 3. [`MoeKernels::moe_down`]: the down projection, scaled by the routing
//!    weight, into a per-pair scratch;
//! 4. [`MoeKernels::moe_sum`]: the scratch summed over the `k` experts of a
//!    token in expert order;
//! 5. [`MoeKernels::moe_write_slot`]: one expert's packed words into a slot.
//!
//! The kernels are the only code here that is not plain tensor algebra. They
//! are written in safe Rust (`#[cube(launch)]` with the checked `launch`),
//! reach the backend through burn's `backend_extension` (the generated
//! `Fusion` glue registers them as opaque operations) and read the pool in
//! this module's own layout ([`ExpertFormat`], [`pack_matrix`]), not burn's
//! quantized layout, so burn's autotune keys do not apply to them.
//!
//! The sum over experts is a separate kernel with a fixed order so that the
//! result of a layer does not depend on how its experts were grouped into
//! passes (a tiered model computes the experts of a long prompt in several
//! passes): tiered placements agree bit for bit with each other. Against a
//! resident load they agree to float rounding only: the resident layer never
//! reads the router output back, a tiered one does, which splits burn's fused
//! elementwise kernels around the layer at different places.

// The `#[cube]` kernels are Rust the CubeCL macro translates: it needs runtime
// values built from runtime values (`tid - tid` is a runtime zero where a
// literal would be evaluated by rustc), plain `x = x / 2` and nested `if`s.
#![allow(
    clippy::eq_op,
    clippy::assign_op_pattern,
    clippy::collapsible_if,
    clippy::too_many_arguments
)]

use std::sync::Arc;

use burn::backend::{TensorMetadata, backend_extension, tensor::FloatTensor, tensor::IntTensor};
use burn::cubecl::prelude::*;
use burn::cubecl::{CubeDim, calculate_cube_count_elemwise};
use burn::tensor::{DType, Shape};
use burn_cubecl::kernel::into_contiguous;
use burn_cubecl::ops::numeric::empty_device_dtype;
use burn_cubecl::{CubeBackend, tensor::CubeTensor};
use candle_core::quantized::GgmlDType;

use super::quant::dequant_row;
use crate::error::CucaError;

/// Elements per quantization block (every format here).
pub(crate) const BLOCK: usize = 32;

/// A format the kernels read a matrix in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ExpertFormat {
    /// 4-bit values `(q - 8) * d`, an f16 scale per block.
    Q4_0,
    /// 4-bit values `q * d + m`, f16 scale and minimum per block.
    Q4_1,
    /// 8-bit values `q * d`, an f16 scale per block.
    Q8_0,
    /// f32 values.
    F32,
}

impl ExpertFormat {
    /// The format of a matrix held as GGUF dtype `dtype`: Q4_0, Q4_1, Q8_0
    /// and the float dtypes as they are, every other dtype requantized to
    /// Q8_0.
    pub(crate) fn of(dtype: GgmlDType) -> Self {
        match dtype {
            GgmlDType::Q4_0 => Self::Q4_0,
            GgmlDType::Q4_1 => Self::Q4_1,
            GgmlDType::Q8_0 => Self::Q8_0,
            GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16 => Self::F32,
            _ => Self::Q8_0,
        }
    }

    /// The code the kernels branch on.
    fn code(self) -> u32 {
        match self {
            Self::Q4_0 => 0,
            Self::Q4_1 => 1,
            Self::Q8_0 => 2,
            Self::F32 => 3,
        }
    }

    /// `u32` words of values per block.
    fn value_words(self) -> usize {
        match self {
            Self::Q4_0 | Self::Q4_1 => 4,
            Self::Q8_0 => 8,
            Self::F32 => BLOCK,
        }
    }

    fn has_scale(self) -> bool {
        self != Self::F32
    }

    fn has_min(self) -> bool {
        self == Self::Q4_1
    }
}

/// Where a `[rows, cols]` matrix of one expert sits in a slot of the pool.
#[derive(Clone, Copy, Debug)]
pub(crate) struct MatrixLayout {
    pub(crate) format: ExpertFormat,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    /// First word of the matrix in the slot.
    pub(crate) offset: usize,
}

impl MatrixLayout {
    fn blocks(&self) -> usize {
        self.rows * self.cols / BLOCK
    }

    /// Words of the f16 scales (or minimums): two per word.
    fn half_words(&self) -> usize {
        self.blocks().div_ceil(2)
    }

    fn scale_offset(&self) -> usize {
        self.offset + self.blocks() * self.format.value_words()
    }

    fn min_offset(&self) -> usize {
        self.scale_offset() + self.half_words()
    }

    /// Words the matrix occupies.
    pub(crate) fn words(&self) -> usize {
        self.blocks() * self.format.value_words()
            + if self.format.has_scale() {
                self.half_words()
            } else {
                0
            }
            + if self.format.has_min() {
                self.half_words()
            } else {
                0
            }
    }
}

/// The three matrices of one expert and the words of a slot.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SlotLayout {
    pub(crate) gate: MatrixLayout,
    pub(crate) up: MatrixLayout,
    pub(crate) down: MatrixLayout,
    pub(crate) words: usize,
}

impl SlotLayout {
    /// The layout of an expert of FFN width `ff` over `n_embd` inputs whose
    /// matrices are held as `dtypes` (gate, up, down).
    ///
    /// # Errors
    ///
    /// [`CucaError::Config`] when a row is not whole 32-element blocks.
    pub(crate) fn new(n_embd: usize, ff: usize, dtypes: [GgmlDType; 3]) -> Result<Self, CucaError> {
        if !n_embd.is_multiple_of(BLOCK) || !ff.is_multiple_of(BLOCK) {
            return Err(CucaError::Config(format!(
                "expert matrices of {n_embd} inputs and {ff} features are not whole {BLOCK}-element blocks"
            )));
        }
        let mut offset = 0;
        let mut next = |format, rows, cols| {
            let m = MatrixLayout {
                format,
                rows,
                cols,
                offset,
            };
            offset += m.words();
            m
        };
        let gate = next(ExpertFormat::of(dtypes[0]), ff, n_embd);
        let up = next(ExpertFormat::of(dtypes[1]), ff, n_embd);
        let down = next(ExpertFormat::of(dtypes[2]), n_embd, ff);
        Ok(Self {
            gate,
            up,
            down,
            words: offset,
        })
    }

    /// Device bytes of one slot.
    pub(crate) fn bytes(&self) -> usize {
        self.words * 4
    }
}

/// f16 bits of `x`'s block scale read from the GGUF block bytes.
fn le16(block: &[u8]) -> u16 {
    u16::from_le_bytes([block[0], block[1]])
}

fn put_half(words: &mut [u32], index: usize, bits: u16) {
    words[index / 2] |= u32::from(bits) << (16 * (index % 2));
}

/// Append the kernel words of the `[rows, cols]` matrix of GGUF dtype `dtype`
/// whose file bytes are `raw` to `out`: the values of every block, then the
/// f16 scales and, for Q4_1, the f16 minimums, two per word.
///
/// # Errors
///
/// [`CucaError::Provider`] for a length that does not match the shape, or a
/// dtype the dequantizer cannot read.
pub(crate) fn pack_matrix(
    layout: &MatrixLayout,
    dtype: GgmlDType,
    raw: &[u8],
    out: &mut Vec<u32>,
) -> Result<(), CucaError> {
    let err = |m: String| super::err(format!("expert matrix: {m}"));
    let blocks = layout.blocks();
    let vw = layout.format.value_words();
    let start = out.len();
    out.resize(start + layout.words(), 0);
    let (values, rest) = out[start..].split_at_mut(blocks * vw);
    let (scales, mins) = rest.split_at_mut(if layout.format.has_scale() {
        layout.half_words()
    } else {
        0
    });
    match layout.format {
        ExpertFormat::Q4_0 | ExpertFormat::Q4_1 => {
            let block_bytes = dtype.type_size();
            if dtype != GgmlDType::Q4_0 && dtype != GgmlDType::Q4_1
                || raw.len() != blocks * block_bytes
            {
                return Err(err(format!(
                    "{} bytes of {dtype:?} for {blocks} blocks",
                    raw.len()
                )));
            }
            let head = if layout.format.has_min() { 4 } else { 2 };
            for (b, block) in raw.chunks_exact(block_bytes).enumerate() {
                put_half(scales, b, le16(block));
                if layout.format.has_min() {
                    put_half(mins, b, le16(&block[2..]));
                }
                let qs = &block[head..];
                for w in 0..4 {
                    let mut word = 0u32;
                    for i in 0..8 {
                        let e = 8 * w + i;
                        let byte = qs[e % 16];
                        let q = if e < 16 { byte & 0x0F } else { byte >> 4 };
                        word |= u32::from(q) << (4 * i);
                    }
                    values[b * 4 + w] = word;
                }
            }
        }
        ExpertFormat::Q8_0 => {
            // The file's own Q8_0 blocks, or any other quantized dtype
            // dequantized and requantized to them.
            let q8: Vec<u8>;
            let source: &[u8] = if dtype == GgmlDType::Q8_0 {
                raw
            } else {
                let row_bytes = layout.cols / dtype.block_size() * dtype.type_size();
                if row_bytes * layout.rows != raw.len() {
                    return Err(err(format!("{} bytes of {dtype:?}", raw.len())));
                }
                let mut buf = vec![0f32; layout.cols];
                let mut bytes = Vec::with_capacity(blocks * 34);
                for row in raw.chunks_exact(row_bytes) {
                    dequant_row(dtype, row, 1.0, &mut buf).map_err(|e| err(e.to_string()))?;
                    super::gpu::quantize_q8_0(&buf, &mut bytes);
                }
                q8 = bytes;
                &q8
            };
            if source.len() != blocks * 34 {
                return Err(err(format!(
                    "{} bytes of Q8_0 for {blocks} blocks",
                    source.len()
                )));
            }
            for (b, block) in source.as_chunks::<34>().0.iter().enumerate() {
                put_half(scales, b, le16(block));
                for w in 0..8 {
                    let at = 2 + 4 * w;
                    values[b * 8 + w] = u32::from_le_bytes([
                        block[at],
                        block[at + 1],
                        block[at + 2],
                        block[at + 3],
                    ]);
                }
            }
        }
        ExpertFormat::F32 => {
            let row_bytes = layout.cols / dtype.block_size() * dtype.type_size();
            if row_bytes * layout.rows != raw.len() {
                return Err(err(format!("{} bytes of {dtype:?}", raw.len())));
            }
            let mut buf = vec![0f32; layout.cols];
            for (r, row) in raw.chunks_exact(row_bytes).enumerate() {
                dequant_row(dtype, row, 1.0, &mut buf).map_err(|e| err(e.to_string()))?;
                for (c, v) in buf.iter().enumerate() {
                    values[r * layout.cols + c] = v.to_bits();
                }
            }
        }
    }
    let _ = mins;
    Ok(())
}

/// The words of one expert (gate, up, down in this order) from its three
/// matrices' file bytes, shared so that a RAM-tier hit uploads without a copy.
///
/// # Errors
///
/// As [`pack_matrix`].
pub(crate) fn pack_expert(
    layout: &SlotLayout,
    dtypes: [GgmlDType; 3],
    gate: &[u8],
    up: &[u8],
    down: &[u8],
) -> Result<Arc<[u32]>, CucaError> {
    let mut words = Vec::with_capacity(layout.words);
    pack_matrix(&layout.gate, dtypes[0], gate, &mut words)?;
    pack_matrix(&layout.up, dtypes[1], up, &mut words)?;
    pack_matrix(&layout.down, dtypes[2], down, &mut words)?;
    Ok(words.into())
}

/// What one kernel call is about: shapes, formats and offsets.
#[derive(Clone, Debug)]
pub(crate) struct MoeCall {
    /// Tokens of the pass.
    pub(crate) tokens: usize,
    pub(crate) n_used: usize,
    pub(crate) n_expert: usize,
    /// Pairs (token, expert) this call computes: `tokens * n_used`, or the
    /// length of the pair list.
    pub(crate) pairs: usize,
    /// Whether the pairs and their slots come from a list (tiered passes)
    /// instead of the route tensor (identity slots, resident).
    pub(crate) use_list: bool,
    /// 0 silu, 1 gelu.
    pub(crate) act: u32,
    /// Multiplier on the routing weights after the softmax and the optional
    /// renormalisation (`expert_weights_scale`, 1 when the file has none).
    pub(crate) scale: f32,
    pub(crate) slot: SlotLayout,
}

/// Router output `[ids | weights]` shape of `tokens` tokens.
fn route_len(call: &MoeCall) -> usize {
    call.tokens * call.n_used * 2
}

// ---------------------------------------------------------------- kernels

/// An f16 (as its 16 bits) to f32. Inf and NaN are not expected in weights.
#[cube]
fn half_to_f32(h: u32) -> f32 {
    let sign = (h & 0x8000u32) << 16u32;
    let exp = (h >> 10u32) & 0x1Fu32;
    let man = h & 0x3FFu32;
    let mut out = f32::cast_from(man) * 5.960_464_5e-8f32;
    if sign != 0u32 {
        out = -out;
    }
    if exp != 0u32 {
        let bits = sign | ((exp + 112u32) << 23u32) | (man << 13u32);
        out = f32::reinterpret(bits);
    }
    out
}

/// The `index`th f16 of the run of f16 pairs starting at word `base`.
#[cube]
fn half_at(pool: &Tensor<u32>, base: usize, index: usize) -> f32 {
    let word = pool[base + index / 2usize];
    let shift = u32::cast_from(index % 2usize) * 16u32;
    half_to_f32((word >> shift) & 0xFFFFu32)
}

/// Dot product of row `row` of the matrix at `matrix` (words from the start
/// of the pool) with `x[x_base..]`.
#[cube]
#[allow(clippy::too_many_arguments)]
fn dot_row(
    pool: &Tensor<u32>,
    matrix: usize,
    scales: usize,
    mins: usize,
    format: u32,
    row: usize,
    blocks_per_row: usize,
    x: &Tensor<f32>,
    x_base: usize,
) -> f32 {
    let mut acc = 0.0f32;
    for b in 0..blocks_per_row {
        let block = row * blocks_per_row + b;
        let xo = x_base + b * 32usize;
        if format == 3u32 {
            let wb = matrix + block * 32usize;
            let mut s = 0.0f32;
            for i in 0..32usize {
                s += f32::reinterpret(pool[wb + i]) * x[xo + i];
            }
            acc += s;
        } else if format == 2u32 {
            let wb = matrix + block * 8usize;
            let mut s = 0.0f32;
            for w in 0..8usize {
                let word = pool[wb + w];
                for i in 0..4usize {
                    let byte = (word >> (u32::cast_from(i) * 8u32)) & 0xFFu32;
                    let q = i32::cast_from(byte & 0x7Fu32) - i32::cast_from(byte & 0x80u32);
                    s += f32::cast_from(q) * x[xo + w * 4usize + i];
                }
            }
            acc += half_at(pool, scales, block) * s;
        } else {
            let wb = matrix + block * 4usize;
            let mut s = 0.0f32;
            let mut sx = 0.0f32;
            for w in 0..4usize {
                let word = pool[wb + w];
                for i in 0..8usize {
                    let q = (word >> (u32::cast_from(i) * 4u32)) & 0xFu32;
                    let xv = x[xo + w * 8usize + i];
                    s += f32::cast_from(q) * xv;
                    sx += xv;
                }
            }
            if format == 0u32 {
                acc += half_at(pool, scales, block) * (s - 8.0f32 * sx);
            } else {
                acc += half_at(pool, scales, block) * s + half_at(pool, mins, block) * sx;
            }
        }
    }
    acc
}

/// Softmax and top-`n_used` per token: one cube of `dim` threads (a power of
/// two, at least `n_expert`) per token, thread `e` holding expert `e`. The
/// probabilities are summed in expert order by one thread and the selection
/// repeats an argmax tree (lower expert first on ties) `n_used` times, so the
/// result is the host's `route` up to the `exp` implementation.
#[cube(launch)]
fn route_kernel(
    logits: &Tensor<f32>,
    out: &mut Tensor<f32>,
    tokens: u32,
    norm: u32,
    scale: f32,
    #[comptime] n_expert: usize,
    #[comptime] n_used: usize,
    #[comptime] dim: usize,
    #[comptime] steps: usize,
) {
    let tok = CUBE_POS_X as usize;
    let tid = UNIT_POS_X as usize;
    let base = tok * n_expert;
    let mut red = Shared::<[f32]>::new_slice(dim);
    let mut probs = Shared::<[f32]>::new_slice(dim);
    let mut who = Shared::<[u32]>::new_slice(dim);

    let mut mine = logits[base];
    if tid < n_expert {
        mine = logits[base + tid];
    }
    red[tid] = mine;
    sync_cube();
    let mut s = (tid - tid) + dim / 2usize;
    for _ in 0..steps {
        if tid < s {
            if red[tid + s] > red[tid] {
                red[tid] = red[tid + s];
            }
        }
        sync_cube();
        s = s / 2usize;
    }
    let top = red[0];
    sync_cube();

    // Idle threads and taken experts hold values below every probability.
    let mut p = (top - top) - 1.0f32;
    if tid < n_expert {
        p = (mine - top).exp();
    }
    probs[tid] = p;
    sync_cube();
    if tid == 0usize {
        let mut total = 0.0f32;
        for e in 0..n_expert {
            total += probs[e];
        }
        red[0] = total;
    }
    sync_cube();
    let total = red[0];
    sync_cube();

    let weights_at = usize::cast_from(tokens) * n_used;
    let mut kept = 0.0f32;
    for i in 0..n_used {
        red[tid] = probs[tid];
        who[tid] = u32::cast_from(tid);
        sync_cube();
        let mut s = (tid - tid) + dim / 2usize;
        for _ in 0..steps {
            if tid < s {
                let a = red[tid];
                let b = red[tid + s];
                let ia = who[tid];
                let ib = who[tid + s];
                if b > a || (b == a && ib < ia) {
                    red[tid] = b;
                    who[tid] = ib;
                }
            }
            sync_cube();
            s = s / 2usize;
        }
        let best = usize::cast_from(who[0]);
        let best_p = red[0];
        sync_cube();
        if tid == 0usize {
            out[tok * n_used + i] = f32::cast_from(best);
            out[weights_at + tok * n_used + i] = best_p;
            kept += best_p;
            probs[best] = (top - top) - 2.0f32;
        }
        sync_cube();
    }
    if tid == 0usize {
        let denom = if norm != 0u32 { kept } else { total };
        for i in 0..n_used {
            out[weights_at + tok * n_used + i] = out[weights_at + tok * n_used + i] / denom * scale;
        }
    }
}

/// `hidden[pair] = act(gate . x) * (up . x)`: one thread per (row, pair).
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn up_kernel(
    x: &Tensor<f32>,
    route: &Tensor<f32>,
    list: &Tensor<u32>,
    pool: &Tensor<u32>,
    hidden: &mut Tensor<f32>,
    use_list: u32,
    n_used: u32,
    n_embd: u32,
    ff: u32,
    act: u32,
    slot_words: u32,
    gate_fmt: u32,
    gate_at: u32,
    gate_scales: u32,
    gate_mins: u32,
    up_fmt: u32,
    up_at: u32,
    up_scales: u32,
    up_mins: u32,
) {
    let row = CUBE_POS_X as usize * CUBE_DIM_X as usize + UNIT_POS_X as usize;
    let pi = CUBE_POS_Y as usize;
    if row < usize::cast_from(ff) {
        let mut pair = pi;
        let mut slot = usize::cast_from(route[pi]);
        if use_list != 0u32 {
            pair = usize::cast_from(list[2usize * pi]);
            slot = usize::cast_from(list[2usize * pi + 1usize]);
        }
        let token = pair / usize::cast_from(n_used);
        let blocks = usize::cast_from(n_embd) / 32usize;
        let base = slot * usize::cast_from(slot_words);
        let x_base = token * usize::cast_from(n_embd);
        let g = dot_row(
            pool,
            base + usize::cast_from(gate_at),
            base + usize::cast_from(gate_scales),
            base + usize::cast_from(gate_mins),
            gate_fmt,
            row,
            blocks,
            x,
            x_base,
        );
        let u = dot_row(
            pool,
            base + usize::cast_from(up_at),
            base + usize::cast_from(up_scales),
            base + usize::cast_from(up_mins),
            up_fmt,
            row,
            blocks,
            x,
            x_base,
        );
        let mut a = g / (1.0f32 + (-g).exp());
        if act != 0u32 {
            let z = 0.797_884_6f32 * (g + 0.044_715f32 * g * g * g);
            let e = (-2.0f32 * f32::abs(z)).exp();
            let t = (1.0f32 - e) / (1.0f32 + e);
            let th = if z < 0.0f32 { -t } else { t };
            a = 0.5f32 * g * (1.0f32 + th);
        }
        hidden[pi * usize::cast_from(ff) + row] = a * u;
    }
}

/// `scratch[pair] = weight * (down . hidden[pair])`: one thread per (row,
/// pair).
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn down_kernel(
    hidden: &Tensor<f32>,
    route: &Tensor<f32>,
    list: &Tensor<u32>,
    pool: &Tensor<u32>,
    scratch: &mut Tensor<f32>,
    use_list: u32,
    tokens: u32,
    n_used: u32,
    n_embd: u32,
    ff: u32,
    slot_words: u32,
    fmt: u32,
    at: u32,
    scales: u32,
    mins: u32,
) {
    let row = CUBE_POS_X as usize * CUBE_DIM_X as usize + UNIT_POS_X as usize;
    let pi = CUBE_POS_Y as usize;
    if row < usize::cast_from(n_embd) {
        let mut pair = pi;
        let mut slot = usize::cast_from(route[pi]);
        if use_list != 0u32 {
            pair = usize::cast_from(list[2usize * pi]);
            slot = usize::cast_from(list[2usize * pi + 1usize]);
        }
        let weight = route[usize::cast_from(tokens) * usize::cast_from(n_used) + pair];
        let base = slot * usize::cast_from(slot_words);
        let d = dot_row(
            pool,
            base + usize::cast_from(at),
            base + usize::cast_from(scales),
            base + usize::cast_from(mins),
            fmt,
            row,
            usize::cast_from(ff) / 32usize,
            hidden,
            pi * usize::cast_from(ff),
        );
        scratch[pair * usize::cast_from(n_embd) + row] = weight * d;
    }
}

/// `out[token] = sum over experts of scratch[token, expert]`, in expert
/// order: one thread per output element.
#[cube(launch)]
fn sum_kernel(scratch: &Tensor<f32>, out: &mut Tensor<f32>, total: u32, n_used: u32, n_embd: u32) {
    let i = ABSOLUTE_POS;
    if i < usize::cast_from(total) {
        let token = i / usize::cast_from(n_embd);
        let row = i % usize::cast_from(n_embd);
        let mut acc = 0.0f32;
        for j in 0..usize::cast_from(n_used) {
            acc += scratch[(token * usize::cast_from(n_used) + j) * usize::cast_from(n_embd) + row];
        }
        out[i] = acc;
    }
}

/// `pool[offset + i] = data[i]`.
#[cube(launch)]
fn write_kernel(pool: &mut Tensor<u32>, data: &Tensor<u32>, offset: u32, words: u32) {
    let i = ABSOLUTE_POS;
    if i < usize::cast_from(words) {
        pool[usize::cast_from(offset) + i] = data[i];
    }
}

// --------------------------------------------------------------- extension

/// Backend operations of the MoE kernels (see the module docs).
#[backend_extension(Cube, Fusion: cfg(feature = "provider-local-gpu"))]
pub trait MoeKernels: burn::backend::Backend {
    /// Router logits `[tokens, n_expert]` to `[ids | weights]`, `tokens *
    /// n_used` each.
    #[fusion(dtype = logits, shape = Shape::new([call.tokens * call.n_used * 2]))]
    fn moe_route(logits: FloatTensor<Self>, call: MoeCall, norm: bool) -> FloatTensor<Self>;

    /// `act(gate . x) * (up . x)` of every pair: `[pairs, ff]`.
    #[fusion(dtype = x, shape = Shape::new([call.pairs, call.slot.gate.rows]))]
    fn moe_up(
        x: FloatTensor<Self>,
        route: FloatTensor<Self>,
        list: IntTensor<Self>,
        pool: IntTensor<Self>,
        call: MoeCall,
    ) -> FloatTensor<Self>;

    /// The scaled down projections of the pairs, written into `scratch`
    /// `[tokens * n_used, n_embd]`, which is returned.
    #[fusion(dtype = hidden, shape = scratch)]
    fn moe_down(
        hidden: FloatTensor<Self>,
        route: FloatTensor<Self>,
        list: IntTensor<Self>,
        pool: IntTensor<Self>,
        scratch: FloatTensor<Self>,
        call: MoeCall,
    ) -> FloatTensor<Self>;

    /// `scratch` summed over each token's experts: `[tokens, n_embd]`.
    #[fusion(dtype = scratch, shape = Shape::new([call.tokens, call.slot.down.rows]))]
    fn moe_sum(scratch: FloatTensor<Self>, call: MoeCall) -> FloatTensor<Self>;

    /// `pool` with `data` written at word `offset`.
    #[fusion(dtype = pool, shape = pool)]
    fn moe_write_slot(
        pool: IntTensor<Self>,
        data: IntTensor<Self>,
        offset: usize,
    ) -> IntTensor<Self>;
}

fn f32_tensor(like: &CubeTensor, shape: Shape) -> CubeTensor {
    empty_device_dtype(like.client.clone(), like.device.clone(), shape, DType::F32)
}

impl MoeKernels for CubeBackend {
    fn moe_route(logits: CubeTensor, call: MoeCall, norm: bool) -> CubeTensor {
        // The kernels index the buffers linearly: a transposed view (the
        // router matmul's output for several tokens) is made row-major.
        let logits = into_contiguous(logits);
        let out = f32_tensor(&logits, Shape::new([route_len(&call)]));
        let client = logits.client.clone();
        let dim = call.n_expert.next_power_of_two().max(2);
        route_kernel::launch(
            &client,
            CubeCount::Static(call.tokens as u32, 1, 1),
            CubeDim::new_1d(dim as u32),
            logits.into_tensor_arg(),
            out.clone().into_tensor_arg(),
            call.tokens as u32,
            u32::from(norm),
            call.scale,
            call.n_expert,
            call.n_used,
            dim,
            dim.trailing_zeros() as usize,
        );
        out
    }

    fn moe_up(
        x: CubeTensor,
        route: CubeTensor,
        list: CubeTensor,
        pool: CubeTensor,
        call: MoeCall,
    ) -> CubeTensor {
        let (g, u) = (&call.slot.gate, &call.slot.up);
        let x = into_contiguous(x);
        let out = f32_tensor(&x, Shape::new([call.pairs, g.rows]));
        let client = x.client.clone();
        let rows = g.rows.div_ceil(64) as u32;
        up_kernel::launch(
            &client,
            CubeCount::Static(rows, call.pairs as u32, 1),
            CubeDim::new_1d(64),
            x.into_tensor_arg(),
            route.into_tensor_arg(),
            list.into_tensor_arg(),
            pool.into_tensor_arg(),
            out.clone().into_tensor_arg(),
            u32::from(call.use_list),
            call.n_used as u32,
            g.cols as u32,
            g.rows as u32,
            call.act,
            call.slot.words as u32,
            g.format.code(),
            g.offset as u32,
            g.scale_offset() as u32,
            g.min_offset() as u32,
            u.format.code(),
            u.offset as u32,
            u.scale_offset() as u32,
            u.min_offset() as u32,
        );
        out
    }

    fn moe_down(
        hidden: CubeTensor,
        route: CubeTensor,
        list: CubeTensor,
        pool: CubeTensor,
        scratch: CubeTensor,
        call: MoeCall,
    ) -> CubeTensor {
        let d = &call.slot.down;
        let client = hidden.client.clone();
        let rows = d.rows.div_ceil(64) as u32;
        down_kernel::launch(
            &client,
            CubeCount::Static(rows, call.pairs as u32, 1),
            CubeDim::new_1d(64),
            hidden.into_tensor_arg(),
            route.into_tensor_arg(),
            list.into_tensor_arg(),
            pool.into_tensor_arg(),
            scratch.clone().into_tensor_arg(),
            u32::from(call.use_list),
            call.tokens as u32,
            call.n_used as u32,
            d.rows as u32,
            d.cols as u32,
            call.slot.words as u32,
            d.format.code(),
            d.offset as u32,
            d.scale_offset() as u32,
            d.min_offset() as u32,
        );
        scratch
    }

    fn moe_sum(scratch: CubeTensor, call: MoeCall) -> CubeTensor {
        let n_embd = call.slot.down.rows;
        let total = call.tokens * n_embd;
        let out = f32_tensor(&scratch, Shape::new([call.tokens, n_embd]));
        let client = scratch.client.clone();
        let dim = CubeDim::new(&client, total);
        let count = calculate_cube_count_elemwise(&client, total, dim);
        sum_kernel::launch(
            &client,
            count,
            dim,
            scratch.into_tensor_arg(),
            out.clone().into_tensor_arg(),
            total as u32,
            call.n_used as u32,
            n_embd as u32,
        );
        out
    }

    fn moe_write_slot(pool: CubeTensor, data: CubeTensor, offset: usize) -> CubeTensor {
        // In place when nothing else holds the pool, else into a copy.
        let pool = if pool.can_mut() { pool } else { pool.copy() };
        let words = data.shape().num_elements();
        let client = pool.client.clone();
        let dim = CubeDim::new(&client, words);
        let count = calculate_cube_count_elemwise(&client, words, dim);
        write_kernel::launch(
            &client,
            count,
            dim,
            pool.clone().into_tensor_arg(),
            data.into_tensor_arg(),
            offset as u32,
            words as u32,
        );
        pool
    }
}

// ---------------------------------------------------------- tensor wrappers

use burn::backend::Dispatch;
use burn::tensor::{Int, Tensor as BTensor};

/// [`MoeKernels::moe_route`] on tensors.
pub(crate) fn route(logits: BTensor<2>, call: &MoeCall, norm: bool) -> BTensor<1> {
    BTensor::from_dispatch(<Dispatch as MoeKernels>::moe_route(
        logits.into_dispatch(),
        call.clone(),
        norm,
    ))
}

/// [`MoeKernels::moe_up`] on tensors.
pub(crate) fn up(
    x: BTensor<2>,
    route: BTensor<1>,
    list: BTensor<1, Int>,
    pool: BTensor<1, Int>,
    call: &MoeCall,
) -> BTensor<2> {
    BTensor::from_dispatch(<Dispatch as MoeKernels>::moe_up(
        x.into_dispatch(),
        route.into_dispatch(),
        list.into_dispatch(),
        pool.into_dispatch(),
        call.clone(),
    ))
}

/// [`MoeKernels::moe_down`] on tensors.
pub(crate) fn down(
    hidden: BTensor<2>,
    route: BTensor<1>,
    list: BTensor<1, Int>,
    pool: BTensor<1, Int>,
    scratch: BTensor<2>,
    call: &MoeCall,
) -> BTensor<2> {
    BTensor::from_dispatch(<Dispatch as MoeKernels>::moe_down(
        hidden.into_dispatch(),
        route.into_dispatch(),
        list.into_dispatch(),
        pool.into_dispatch(),
        scratch.into_dispatch(),
        call.clone(),
    ))
}

/// [`MoeKernels::moe_sum`] on tensors.
pub(crate) fn sum(scratch: BTensor<2>, call: &MoeCall) -> BTensor<2> {
    BTensor::from_dispatch(<Dispatch as MoeKernels>::moe_sum(
        scratch.into_dispatch(),
        call.clone(),
    ))
}

/// [`MoeKernels::moe_write_slot`] on tensors.
pub(crate) fn write_slot(
    pool: BTensor<1, Int>,
    data: BTensor<1, Int>,
    offset: usize,
) -> BTensor<1, Int> {
    BTensor::from_dispatch(<Dispatch as MoeKernels>::moe_write_slot(
        pool.into_dispatch(),
        data.into_dispatch(),
        offset,
    ))
}
