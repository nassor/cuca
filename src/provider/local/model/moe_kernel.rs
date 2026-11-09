//! Fused mixture-of-experts kernels for the GPU engine (CubeCL).
//!
//! burn's quantized matmuls cannot gather the weights of a data-dependent
//! subset of experts (`q_select` dequantizes), and a matmul triple per
//! selected expert would be host-bound and need the routing on the host.
//! These kernels read the experts' weights straight from a
//! per-layer *pool* of expert slots held as one `u32` tensor, indexed by the
//! router's own output:
//!
//! 1. [`MoeKernels::moe_route`]: the expert scores of the router logits
//!    (softmax, or `sqrt(softplus)`, see [`Scoring`]; [`MoeKernels::moe_route_biased`]
//!    adds a selection-only bias) and the top-`k` per token, as `[ids (tokens
//!    * k) | weights (tokens * k)]` in one f32 tensor;
//! 2. [`MoeKernels::moe_up`]: `act(gate . x) * (up . x)` for every (token,
//!    expert) pair, with an optional SwiGLU clamp;
//! 3. [`MoeKernels::moe_down`]: the down projection, scaled by the routing
//!    weight, into a per-pair scratch;
//! 4. [`MoeKernels::moe_sum`]: the scratch summed over the `k` experts of a
//!    token in expert order;
//! 5. [`MoeKernels::moe_write_slot`]: one expert's packed words into a slot.
//!
//! The dense weights (attention projections, dense FFNs, the LM head) use the
//! same block formats and dot-row arithmetic through two more operations,
//! with a block-major word order ([`DenseLayout`]) so that the threads of a
//! launch, one per output row, read consecutive blocks:
//!
//! - [`MoeKernels::dense_gemv`]: `x . w` for a few input rows (decode,
//!   batched decode, verification);
//! - [`MoeKernels::dense_dequant`]: a slice of rows of the weight as f32,
//!   which a prefill multiplies with burn's float matmul.
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
//!
//! MXFP4 matrices (OCP microscaling FP4: 32 E2M1 values and one E8M0 scale
//! byte per 17 file bytes) are read as the file stores them, with no
//! requantization: the 16 nibble bytes of a block are 4 `u32` words (byte `j`
//! holds element `j` in its low nibble and element `j + 16` in its high one)
//! and the scale bytes follow the values, four per word. A nibble selects one
//! of the doubled E2M1 values `{0, 1, 2, 3, 4, 6, 8, 12}` (sign in bit 3) and
//! the block scale is `2^(e - 128)` (ggml's half scale of the E8M0 byte `e`).
//! The dot products multiply them with the f32 activations; the CPU engine
//! rounds the activations to Q8_0 first, so the engines agree loosely.

// The `#[cube]` kernels are Rust the CubeCL macro translates: it needs runtime
// values built from runtime values (`tid - tid` is a runtime zero where a
// literal would be evaluated by rustc), plain `x = x / 2` and nested `if`s.
#![allow(
    clippy::eq_op,
    clippy::assign_op_pattern,
    clippy::collapsible_if,
    clippy::manual_div_ceil,
    clippy::too_many_arguments
)]

use std::sync::Arc;

use burn::backend::{TensorMetadata, backend_extension, tensor::FloatTensor, tensor::IntTensor};
use burn::cubecl::prelude::*;
use burn::cubecl::{CubeDim, calculate_cube_count_elemwise};
use burn::tensor::{DType, Shape};
use burn_cubecl::kernel::into_contiguous;
use burn_cubecl::ops::numeric::empty_device_contiguous_dtype;
use burn_cubecl::{CubeBackend, tensor::CubeTensor};
use candle_core::quantized::GgmlDType;

use super::quant::dequant_row;
use crate::error::CucaError;
use crate::provider::local::gguf::GgufType;

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
    /// OCP MXFP4: 4-bit E2M1 values (the doubled table of ggml's
    /// `kvalues_mxfp4`) times a power-of-two E8M0 scale byte per block, as the
    /// file stores them.
    Mxfp4,
}

impl ExpertFormat {
    /// The format of a matrix held as `dtype`: Q4_0, Q4_1, Q8_0, MXFP4 and the
    /// float dtypes as they are, every other dtype requantized to Q8_0.
    pub(crate) fn of(dtype: impl Into<GgufType>) -> Self {
        match dtype.into() {
            GgufType::Ggml(GgmlDType::Q4_0) => Self::Q4_0,
            GgufType::Ggml(GgmlDType::Q4_1) => Self::Q4_1,
            GgufType::Ggml(GgmlDType::Q8_0) => Self::Q8_0,
            GgufType::Ggml(GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16) => Self::F32,
            GgufType::Mxfp4 => Self::Mxfp4,
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
            Self::Mxfp4 => 4,
        }
    }

    /// `u32` words of values per block.
    fn value_words(self) -> usize {
        match self {
            Self::Q4_0 | Self::Q4_1 | Self::Mxfp4 => 4,
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

    /// Bits of one block scale (and minimum) in the scale words: an f16, or
    /// MXFP4's E8M0 byte.
    fn scale_bits(self) -> u32 {
        if self == Self::Mxfp4 { 8 } else { 16 }
    }
}

/// Words of `count` scales of `bits` bits each, packed from the low bits.
fn packed_words(count: usize, bits: u32) -> usize {
    count.div_ceil((32 / bits) as usize)
}

/// OR `value` (of `bits` bits) in as scale number `index` of `words`.
fn put_scale(words: &mut [u32], index: usize, value: u32, bits: u32) {
    let per = (32 / bits) as usize;
    words[index / per] |= value << (bits * (index % per) as u32);
}

/// Scale number `index` of `words`.
fn scale_at(words: &[u32], index: usize, bits: u32) -> u32 {
    let per = (32 / bits) as usize;
    (words[index / per] >> (bits * (index % per) as u32)) & (u32::MAX >> (32 - bits))
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

    /// Words of the scales (or minimums): two f16 per word, or four E8M0
    /// bytes per word.
    fn scale_words(&self) -> usize {
        packed_words(self.blocks(), self.format.scale_bits())
    }

    fn scale_offset(&self) -> usize {
        self.offset + self.blocks() * self.format.value_words()
    }

    fn min_offset(&self) -> usize {
        self.scale_offset() + self.scale_words()
    }

    /// Words the matrix occupies.
    pub(crate) fn words(&self) -> usize {
        let scales = usize::from(self.format.has_scale()) + usize::from(self.format.has_min());
        self.blocks() * self.format.value_words() + scales * self.scale_words()
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
    pub(crate) fn new<T: Into<GgufType>>(
        n_embd: usize,
        ff: usize,
        dtypes: [T; 3],
    ) -> Result<Self, CucaError> {
        if !n_embd.is_multiple_of(BLOCK) || !ff.is_multiple_of(BLOCK) {
            return Err(CucaError::Config(format!(
                "expert matrices of {n_embd} inputs and {ff} features are not whole {BLOCK}-element blocks"
            )));
        }
        let [gate_type, up_type, down_type] = dtypes;
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
        let gate = next(ExpertFormat::of(gate_type), ff, n_embd);
        let up = next(ExpertFormat::of(up_type), ff, n_embd);
        let down = next(ExpertFormat::of(down_type), n_embd, ff);
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

/// The little-endian f16 bits at the start of `block` (a GGUF block's scale).
fn le16(block: &[u8]) -> u32 {
    u32::from(u16::from_le_bytes([block[0], block[1]]))
}

/// Append the kernel words of the `[rows, cols]` matrix of type `dtype`
/// whose file bytes are `raw` to `out`: the values of every block, then the
/// scales (f16, two per word; MXFP4's E8M0 bytes, four per word) and, for
/// Q4_1, the f16 minimums, two per word.
///
/// # Errors
///
/// [`CucaError::Provider`] for a length that does not match the shape, or a
/// dtype the dequantizer cannot read.
pub(crate) fn pack_matrix(
    layout: &MatrixLayout,
    dtype: impl Into<GgufType>,
    raw: &[u8],
    out: &mut Vec<u32>,
) -> Result<(), CucaError> {
    let dtype = dtype.into();
    let err = |m: String| super::err(format!("expert matrix: {m}"));
    let blocks = layout.blocks();
    let vw = layout.format.value_words();
    let start = out.len();
    out.resize(start + layout.words(), 0);
    let (values, rest) = out[start..].split_at_mut(blocks * vw);
    let (scales, mins) = rest.split_at_mut(if layout.format.has_scale() {
        layout.scale_words()
    } else {
        0
    });
    // Bytes per row of a type that needs the dequantizer, which must hold
    // the rows of the matrix exactly.
    let row_bytes = || match dtype.byte_len(layout.cols) {
        Some(n) if n * layout.rows == raw.len() => Ok(n),
        _ => Err(err(format!("{} bytes of {}", raw.len(), dtype.name()))),
    };
    match layout.format {
        ExpertFormat::Q4_0 | ExpertFormat::Q4_1 => {
            let block_bytes = dtype.type_size().unwrap_or(0);
            if dtype != GgufType::Ggml(GgmlDType::Q4_0) && dtype != GgufType::Ggml(GgmlDType::Q4_1)
                || raw.len() != blocks * block_bytes
            {
                return Err(err(format!(
                    "{} bytes of {} for {blocks} blocks",
                    raw.len(),
                    dtype.name()
                )));
            }
            let head = if layout.format.has_min() { 4 } else { 2 };
            for (b, block) in raw.chunks_exact(block_bytes).enumerate() {
                put_scale(scales, b, le16(block), 16);
                if layout.format.has_min() {
                    put_scale(mins, b, le16(&block[2..]), 16);
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
            let source: &[u8] = if dtype == GgufType::Ggml(GgmlDType::Q8_0) {
                raw
            } else {
                let row_bytes = row_bytes()?;
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
                put_scale(scales, b, le16(block), 16);
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
            let row_bytes = row_bytes()?;
            let mut buf = vec![0f32; layout.cols];
            for (r, row) in raw.chunks_exact(row_bytes).enumerate() {
                dequant_row(dtype, row, 1.0, &mut buf).map_err(|e| err(e.to_string()))?;
                for (c, v) in buf.iter().enumerate() {
                    values[r * layout.cols + c] = v.to_bits();
                }
            }
        }
        ExpertFormat::Mxfp4 => {
            // The file's blocks as they are: the scale byte, then 16 bytes of
            // nibbles (byte `j`: element `j` low, element `j + 16` high).
            if dtype != GgufType::Mxfp4 || raw.len() != blocks * 17 {
                return Err(err(format!(
                    "{} bytes of {} for {blocks} blocks",
                    raw.len(),
                    dtype.name()
                )));
            }
            for (b, block) in raw.as_chunks::<17>().0.iter().enumerate() {
                put_scale(scales, b, u32::from(block[0]), 8);
                for w in 0..4 {
                    let at = 1 + 4 * w;
                    values[b * 4 + w] = u32::from_le_bytes([
                        block[at],
                        block[at + 1],
                        block[at + 2],
                        block[at + 3],
                    ]);
                }
            }
        }
    }
    Ok(())
}

/// The words of one expert (gate, up, down in this order) from its three
/// matrices' file bytes, shared so that a RAM-tier hit uploads without a copy.
///
/// # Errors
///
/// As [`pack_matrix`].
pub(crate) fn pack_expert<T: Into<GgufType> + Copy>(
    layout: &SlotLayout,
    dtypes: [T; 3],
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

/// A dense `[rows, cols]` weight (an attention projection, a dense FFN
/// matrix, the LM head) in the layout of the dense kernels: like a slot's
/// matrix, but in block-major order, so that the threads of a launch, one
/// per row, read consecutive blocks. The block `b` of row `r` is block `b *
/// rows + r`; the values of all blocks come first (4 words per Q4 or MXFP4
/// block, 8 per Q8 block), then the scales (f16, two per word; MXFP4's E8M0
/// bytes, four per word) and, for Q4_1, the f16 minimums, two per word, in the
/// same block order.
#[derive(Clone, Copy, Debug)]
pub(crate) struct DenseLayout {
    pub(crate) format: ExpertFormat,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
}

impl DenseLayout {
    /// The layout of a `[rows, cols]` weight held as `dtype`: Q4_0, Q4_1,
    /// Q8_0 and MXFP4 as they are, every other quantized dtype requantized
    /// to Q8_0.
    ///
    /// # Errors
    ///
    /// [`CucaError::Config`] for a float dtype (floats stay f32 tensors) or
    /// a row that is not whole 32-element blocks.
    pub(crate) fn new(
        dtype: impl Into<GgufType>,
        rows: usize,
        cols: usize,
    ) -> Result<Self, CucaError> {
        let dtype = dtype.into();
        let format = ExpertFormat::of(dtype);
        if format == ExpertFormat::F32 || !cols.is_multiple_of(BLOCK) {
            return Err(CucaError::Config(format!(
                "a dense {} weight of rows of {cols} values has no block layout",
                dtype.name()
            )));
        }
        Ok(Self { format, rows, cols })
    }

    pub(crate) fn blocks_per_row(&self) -> usize {
        self.cols / BLOCK
    }

    fn blocks(&self) -> usize {
        self.rows * self.blocks_per_row()
    }

    fn scale_words(&self) -> usize {
        packed_words(self.blocks(), self.format.scale_bits())
    }

    fn scale_offset(&self) -> usize {
        self.blocks() * self.format.value_words()
    }

    fn min_offset(&self) -> usize {
        self.scale_offset() + self.scale_words()
    }

    /// Words of the weight.
    pub(crate) fn words(&self) -> usize {
        self.min_offset()
            + if self.format.has_min() {
                self.scale_words()
            } else {
                0
            }
    }

    /// Device bytes of the weight.
    pub(crate) fn bytes(&self) -> usize {
        self.words() * 4
    }
}

/// The words of the `[rows, cols]` weight of type `dtype` whose file bytes
/// are `raw`, in the order of [`DenseLayout`].
///
/// # Errors
///
/// As [`pack_matrix`].
pub(crate) fn pack_dense(
    layout: &DenseLayout,
    dtype: impl Into<GgufType>,
    raw: &[u8],
) -> Result<Vec<u32>, CucaError> {
    let matrix = MatrixLayout {
        format: layout.format,
        rows: layout.rows,
        cols: layout.cols,
        offset: 0,
    };
    let mut row_major = Vec::with_capacity(matrix.words());
    pack_matrix(&matrix, dtype, raw, &mut row_major)?;
    let (rows, bpr, vw) = (
        layout.rows,
        layout.blocks_per_row(),
        layout.format.value_words(),
    );
    let bits = layout.format.scale_bits();
    let mut out = vec![0u32; layout.words()];
    let (src_values, src_rest) = row_major.split_at(layout.blocks() * vw);
    let (dst_values, dst_rest) = out.split_at_mut(layout.blocks() * vw);
    let scale_words = layout.scale_words();
    let (dst_scales, dst_mins) = dst_rest.split_at_mut(scale_words);
    for r in 0..rows {
        for b in 0..bpr {
            let (from, to) = (r * bpr + b, b * rows + r);
            dst_values[to * vw..(to + 1) * vw]
                .copy_from_slice(&src_values[from * vw..(from + 1) * vw]);
            put_scale(dst_scales, to, scale_at(src_rest, from, bits), bits);
            if layout.format.has_min() {
                put_scale(
                    dst_mins,
                    to,
                    scale_at(&src_rest[scale_words..], from, 16),
                    16,
                );
            }
        }
    }
    Ok(out)
}

/// One call of a dense kernel: the weight, the rows of the input and (for
/// the dequantizing kernel) the rows of the weight it expands.
#[derive(Clone, Debug)]
pub(crate) struct DenseCall {
    pub(crate) layout: DenseLayout,
    /// Input rows (tokens) of the matrix-vector kernel.
    pub(crate) tokens: usize,
    /// First weight row and row count the dequantizing kernel expands.
    pub(crate) row0: usize,
    pub(crate) count: usize,
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
    /// Multiplier on the routing weights after the scoring and the optional
    /// renormalisation (`expert_weights_scale`, 1 when the file has none).
    pub(crate) scale: f32,
    pub(crate) slot: SlotLayout,
}

/// How the router logits become expert scores.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Scoring {
    /// `exp(l - max l)`, the softmax numerator. The kept scores are divided
    /// by their sum (`norm`) or by the sum over all experts.
    Softmax,
    /// `sqrt(softplus(l))` with torch's threshold (`l` above 20 stays `l`),
    /// every expert scored on its own. The kept scores are divided by their
    /// sum plus `1e-20` (`norm`) or used as they are.
    SqrtSoftplus,
}

impl Scoring {
    /// The code the router kernel branches on.
    fn code(self) -> u32 {
        match self {
            Self::Softmax => 0,
            Self::SqrtSoftplus => 1,
        }
    }
}

/// What a router call does beyond the shapes of its [`MoeCall`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RouteSpec {
    pub(crate) scoring: Scoring,
    /// Renormalise the kept scores (see [`Scoring`]).
    pub(crate) norm: bool,
}

/// Threads of a router cube at most: a layer with more experts has each
/// thread loop over several.
const ROUTE_DIM: usize = 256;

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

/// Byte `index` of the bytes (four per word) that start at word `base`.
#[cube]
fn byte_at(pool: &Tensor<u32>, base: usize, index: usize) -> u32 {
    let word = pool[base + index / 4usize];
    (word >> (u32::cast_from(index % 4usize) * 8u32)) & 0xFFu32
}

/// `s * 2^(e - 128)`, `e` the E8M0 scale byte of an MXFP4 block (ggml's half
/// scale, as `quant::e8m0_half`, so a finite `s` stays finite up to `e` =
/// 255). The two lowest exponents (`2^-128`, `2^-127`) are f32 denormals,
/// which a GPU may flush when they are operands: they are applied as two
/// normal factors instead (`2^-64`, then `2^-64` or `2^-63`).
#[cube]
fn mxfp4_scaled(s: f32, e: u32) -> f32 {
    let mut head = s;
    let mut bits = (e - 1u32) << 23u32;
    if e < 2u32 {
        head = s * f32::reinterpret(0x1F80_0000u32);
        bits = 0x1F80_0000u32 + (e << 23u32);
    }
    head * f32::reinterpret(bits)
}

/// The doubled E2M1 value of the MXFP4 nibble `n`: bit 3 is the sign, the
/// magnitudes `0..8` are `{0, 1, 2, 3, 4, 6, 8, 12}`, held as the nibbles of
/// one constant.
#[cube]
fn mxfp4_value(n: u32) -> f32 {
    let magnitude = f32::cast_from((0xC864_3210u32 >> ((n & 7u32) * 4u32)) & 0xFu32);
    let mut value = magnitude;
    if (n & 8u32) != 0u32 {
        value = -magnitude;
    }
    value
}

/// `2^(e - 128) * (block . x[xo..xo + 32])` of the MXFP4 block whose 4 value
/// words start at word `wb` and whose scale byte is `e`.
#[cube]
fn mxfp4_block_dot(pool: &Tensor<u32>, wb: usize, e: u32, x: &Tensor<f32>, xo: usize) -> f32 {
    let mut s = 0.0f32;
    for w in 0..4usize {
        let word = pool[wb + w];
        for k in 0..4usize {
            let byte = (word >> (u32::cast_from(k) * 8u32)) & 0xFFu32;
            let j = w * 4usize + k;
            s += mxfp4_value(byte & 0xFu32) * x[xo + j]
                + mxfp4_value(byte >> 4u32) * x[xo + j + 16usize];
        }
    }
    mxfp4_scaled(s, e)
}

/// Partial dot product of row `row` of the matrix at `matrix` (words from the
/// start of the pool) with `x[x_base..]` over the blocks `first`, `first +
/// step`, ... (`count` of them).
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
    first: usize,
    step: usize,
    count: usize,
    x: &Tensor<f32>,
    x_base: usize,
) -> f32 {
    let mut acc = 0.0f32;
    for i in 0..count {
        let b = first + i * step;
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
        } else if format == 4u32 {
            let e = byte_at(pool, scales, block);
            acc += mxfp4_block_dot(pool, matrix + block * 4usize, e, x, xo);
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

/// The score of the router logit `l` of a row whose largest logit is `top`:
/// `exp(l - top)` (the softmax numerator) for `scoring == 0`,
/// `sqrt(softplus(l))` for `scoring == 1`. The softplus is torch's: `l` above
/// 20, else `ln(1 + e^l)`, as `ln(u) * y / (u - 1)` with `u = 1 + y` (Goldberg's
/// `log1p`) so that the score of a very negative logit keeps its digits.
#[cube]
fn route_score(l: f32, top: f32, scoring: u32) -> f32 {
    let mut score = (l - top).exp();
    if scoring == 1u32 {
        let mut softplus = l;
        if l <= 20.0f32 {
            let y = l.exp();
            let u = 1.0f32 + y;
            softplus = y;
            if u != 1.0f32 {
                softplus = u.ln() * y / (u - 1.0f32);
            }
        }
        score = softplus.sqrt();
    }
    score
}

/// Scores and top-`n_used` per token: one cube of `dim` threads (a power of
/// two, at most [`ROUTE_DIM`]) per token, thread `t` holding the experts `t`,
/// `t + dim`, ... (`per` of them, so any number of experts fits). The softmax
/// divisor is summed in expert order by one thread and the selection repeats
/// an argmax tree over the thread-local bests (the key is the score plus the
/// bias; lower expert first on ties, which is a total order, so the result does
/// not depend on the tree) `n_used` times: the result is the host's
/// `route_with` up to the `exp`, `ln` and `sqrt` implementations.
#[cube(launch)]
fn route_kernel(
    logits: &Tensor<f32>,
    bias: &Tensor<f32>,
    out: &mut Tensor<f32>,
    tokens: u32,
    norm: u32,
    scoring: u32,
    has_bias: u32,
    scale: f32,
    #[comptime] n_expert: usize,
    #[comptime] n_used: usize,
    #[comptime] dim: usize,
    #[comptime] per: usize,
    #[comptime] slots: usize,
    #[comptime] steps: usize,
) {
    let tok = CUBE_POS_X as usize;
    let tid = UNIT_POS_X as usize;
    let base = tok * n_expert;
    let mut red = Shared::<[f32]>::new_slice(dim);
    let mut who = Shared::<[u32]>::new_slice(dim);
    let mut keys = Shared::<[f32]>::new_slice(slots);

    // The largest logit of the row: each thread's experts, then a tree.
    let mut mine = logits[base];
    for j in 0..per {
        let e = tid + j * dim;
        if e < n_expert {
            if logits[base + e] > mine {
                mine = logits[base + e];
            }
        }
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

    // The selection key of every expert; the padding past the last expert and
    // the experts already taken hold a value below every key.
    let lowest = f32::cast_from(tid - tid) - 3.0e38f32;
    for j in 0..per {
        let e = tid + j * dim;
        let mut key = lowest;
        if e < n_expert {
            key = route_score(logits[base + e], top, scoring);
            if has_bias != 0u32 {
                key = key + bias[e];
            }
        }
        keys[e] = key;
    }
    sync_cube();

    // The softmax divisor: the sum of `exp(l - top)` over the experts in
    // order (the keys when there is no bias).
    let mut total = top - top;
    if tid == 0usize {
        if scoring == 0u32 {
            for e in 0..n_expert {
                if has_bias != 0u32 {
                    total += (logits[base + e] - top).exp();
                } else {
                    total += keys[e];
                }
            }
        }
    }

    let weights_at = usize::cast_from(tokens) * n_used;
    let mut kept = top - top;
    for i in 0..n_used {
        let mut best_key = keys[tid];
        let mut best_at = u32::cast_from(tid);
        for j in 1..per {
            let e = tid + j * dim;
            if keys[e] > best_key {
                best_key = keys[e];
                best_at = u32::cast_from(e);
            }
        }
        red[tid] = best_key;
        who[tid] = best_at;
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
        sync_cube();
        if tid == 0usize {
            let raw = route_score(logits[base + best], top, scoring);
            out[tok * n_used + i] = f32::cast_from(best);
            out[weights_at + tok * n_used + i] = raw;
            kept += raw;
            keys[best] = lowest;
        }
        sync_cube();
    }
    if tid == 0usize {
        let mut denom = total;
        if scoring == 1u32 {
            denom = (top - top) + 1.0f32;
        }
        if norm != 0u32 {
            denom = kept;
            if scoring == 1u32 {
                denom = kept + 1.0e-20f32;
            }
        }
        for i in 0..n_used {
            out[weights_at + tok * n_used + i] = out[weights_at + tok * n_used + i] / denom * scale;
        }
    }
}

/// Threads of a cube of [`up_kernel`] and [`down_kernel`].
const EXPERT_GROUP: usize = 64;

/// Threads per output row of the expert kernels for a matrix of
/// `blocks_per_row` blocks per row: [`EXPERT_LANES`] or fewer for a short
/// row. A function of the layout alone, so that a row's result does not
/// depend on how many rows share the launch.
fn expert_lanes(blocks_per_row: usize) -> usize {
    let mut lanes = 1;
    while lanes < EXPERT_LANES && lanes * 2 <= blocks_per_row {
        lanes *= 2;
    }
    lanes
}

/// Lanes of an output row of the expert kernels (a power of two dividing
/// [`EXPERT_GROUP`]): the blocks of a row are split over the lanes, which
/// makes the weight reads of a cube coalesced (the lanes of a row read
/// consecutive blocks) and gives a launch of a few pairs, a decode step,
/// enough threads to fill the device.
const EXPERT_LANES: usize = 8;

/// `hidden[pair] = act(gate . x) * (up . x)`. A cube of [`EXPERT_GROUP`]
/// threads serves `EXPERT_GROUP / lanes` rows of one pair, `lanes` threads per
/// row: thread `tid` is lane `tid % lanes` of row `tid / lanes`. Lane `l`
/// takes the blocks `l`, `l + lanes`, ... in f32 and lane 0 adds the lanes'
/// partial sums in lane order.
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
    clamp: f32,
    lanes: u32,
    #[comptime] group: usize,
) {
    let tid = UNIT_POS_X as usize;
    let lanes = usize::cast_from(lanes);
    let lane = tid % lanes;
    let row = CUBE_POS_X as usize * (group / lanes) + tid / lanes;
    let pi = CUBE_POS_Y as usize;
    let mut gate_part = Shared::<[f32]>::new_slice(group);
    let mut up_part = Shared::<[f32]>::new_slice(group);
    let mut g_acc = f32::cast_from(tid - tid);
    let mut u_acc = f32::cast_from(tid - tid);
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
        let mut count = 0usize;
        if lane < blocks {
            count = (blocks - lane + lanes - 1usize) / lanes;
        }
        g_acc = dot_row(
            pool,
            base + usize::cast_from(gate_at),
            base + usize::cast_from(gate_scales),
            base + usize::cast_from(gate_mins),
            gate_fmt,
            row,
            blocks,
            lane,
            lanes,
            count,
            x,
            x_base,
        );
        u_acc = dot_row(
            pool,
            base + usize::cast_from(up_at),
            base + usize::cast_from(up_scales),
            base + usize::cast_from(up_mins),
            up_fmt,
            row,
            blocks,
            lane,
            lanes,
            count,
            x,
            x_base,
        );
    }
    gate_part[tid] = g_acc;
    up_part[tid] = u_acc;
    sync_cube();
    if lane == 0usize && row < usize::cast_from(ff) {
        let mut g = gate_part[tid];
        let mut u = up_part[tid];
        for l in 1..lanes {
            g += gate_part[tid + l];
            u += up_part[tid + l];
        }
        // The SwiGLU clamp (`Glu::apply`): the gate from above, the up branch
        // to `[-clamp, clamp]`, before the activation; 0 is none.
        if clamp > 0.0f32 {
            if g > clamp {
                g = clamp;
            }
            if u > clamp {
                u = clamp;
            }
            if u < -clamp {
                u = -clamp;
            }
        }
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

/// `scratch[pair] = weight * (down . hidden[pair])`, the lanes of a row
/// as in [`up_kernel`].
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
    lanes: u32,
    #[comptime] group: usize,
) {
    let tid = UNIT_POS_X as usize;
    let lanes = usize::cast_from(lanes);
    let lane = tid % lanes;
    let row = CUBE_POS_X as usize * (group / lanes) + tid / lanes;
    let pi = CUBE_POS_Y as usize;
    let mut part = Shared::<[f32]>::new_slice(group);
    let mut acc = f32::cast_from(tid - tid);
    let mut pair = pi;
    let mut slot = usize::cast_from(route[pi]);
    if use_list != 0u32 {
        pair = usize::cast_from(list[2usize * pi]);
        slot = usize::cast_from(list[2usize * pi + 1usize]);
    }
    if row < usize::cast_from(n_embd) {
        let base = slot * usize::cast_from(slot_words);
        let blocks = usize::cast_from(ff) / 32usize;
        let mut count = 0usize;
        if lane < blocks {
            count = (blocks - lane + lanes - 1usize) / lanes;
        }
        acc = dot_row(
            pool,
            base + usize::cast_from(at),
            base + usize::cast_from(scales),
            base + usize::cast_from(mins),
            fmt,
            row,
            blocks,
            lane,
            lanes,
            count,
            hidden,
            pi * usize::cast_from(ff),
        );
    }
    part[tid] = acc;
    sync_cube();
    if lane == 0usize && row < usize::cast_from(n_embd) {
        let mut d = part[tid];
        for l in 1..lanes {
            d += part[tid + l];
        }
        let weight = route[usize::cast_from(tokens) * usize::cast_from(n_used) + pair];
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

/// Threads of a cube of [`dense_gemv_kernel`].
const GEMV_GROUP: usize = 64;

/// Threads per row of [`dense_gemv_kernel`] for a weight of `rows` rows of
/// `blocks_per_row` blocks and `tokens` input rows: a power of two up to
/// [`GEMV_GROUP`] and the blocks per row, growing while the launch has fewer
/// than 256 cubes, so that a weight with few rows still fills the device.
fn gemv_lanes(rows: usize, blocks_per_row: usize, tokens: usize) -> usize {
    let mut lanes = 1;
    while lanes < GEMV_GROUP
        && lanes * 2 <= blocks_per_row
        && rows.div_ceil(GEMV_GROUP / lanes) * tokens < 256
    {
        lanes *= 2;
    }
    lanes
}

/// Partial dot product of row `row` of the dense weight (block-major, see
/// [`DenseLayout`]) with `x[x_base..]` over the blocks `first`, `first +
/// step`, ... (`count` of them).
#[cube]
#[allow(clippy::too_many_arguments)]
fn dense_dot(
    pool: &Tensor<u32>,
    row: usize,
    rows: usize,
    first: usize,
    step: usize,
    count: usize,
    format: u32,
    scales: usize,
    mins: usize,
    x: &Tensor<f32>,
    x_base: usize,
) -> f32 {
    let mut acc = 0.0f32;
    for i in 0..count {
        let b = first + i * step;
        let block = b * rows + row;
        let xo = x_base + b * 32usize;
        if format == 4u32 {
            let e = byte_at(pool, scales, block);
            acc += mxfp4_block_dot(pool, block * 4usize, e, x, xo);
        } else if format == 2u32 {
            let wb = block * 8usize;
            let mut s = 0.0f32;
            for w in 0..8usize {
                let word = pool[wb + w];
                for k in 0..4usize {
                    let byte = (word >> (u32::cast_from(k) * 8u32)) & 0xFFu32;
                    let q = i32::cast_from(byte & 0x7Fu32) - i32::cast_from(byte & 0x80u32);
                    s += f32::cast_from(q) * x[xo + w * 4usize + k];
                }
            }
            acc += half_at(pool, scales, block) * s;
        } else {
            let wb = block * 4usize;
            let mut s = 0.0f32;
            let mut sx = 0.0f32;
            for w in 0..4usize {
                let word = pool[wb + w];
                for k in 0..8usize {
                    let q = (word >> (u32::cast_from(k) * 4u32)) & 0xFu32;
                    let xv = x[xo + w * 8usize + k];
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

/// `out[token, row] = weight[row] . x[token]`. A cube of `group` threads
/// serves `group / lanes` rows, `lanes` threads per row: thread `tid` is
/// lane `tid / (group / lanes)` of row `tid % (group / lanes)`, so the
/// threads of a wavefront read the consecutive blocks of consecutive rows.
/// Lane `l` takes the blocks `l`, `l + lanes`, ... in f32; lane 0 adds the
/// lanes' partial sums in lane order. Short weights (few rows) take more
/// lanes to fill the device.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn dense_gemv_kernel(
    x: &Tensor<f32>,
    pool: &Tensor<u32>,
    out: &mut Tensor<f32>,
    rows: u32,
    blocks_per_row: u32,
    format: u32,
    scales: u32,
    mins: u32,
    lanes: u32,
    #[comptime] group: usize,
) {
    let tid = UNIT_POS_X as usize;
    let lanes = usize::cast_from(lanes);
    let per_group = group / lanes;
    let local = tid % per_group;
    let lane = tid / per_group;
    let rows = usize::cast_from(rows);
    let row = CUBE_POS_X as usize * per_group + local;
    let token = CUBE_POS_Y as usize;
    let bpr = usize::cast_from(blocks_per_row);
    let mut partial = Shared::<[f32]>::new_slice(group);
    let mut acc = f32::cast_from(tid - tid);
    if row < rows {
        let count = (bpr - lane + lanes - 1usize) / lanes;
        acc = dense_dot(
            pool,
            row,
            rows,
            lane,
            lanes,
            count,
            format,
            usize::cast_from(scales),
            usize::cast_from(mins),
            x,
            token * bpr * 32usize,
        );
    }
    partial[tid] = acc;
    sync_cube();
    if lane == 0usize && row < rows {
        let mut total = partial[local];
        for l in 1..lanes {
            total += partial[l * per_group + local];
        }
        out[token * rows + row] = total;
    }
}

/// `out[r, c]` = element `c` of row `row0 + r` of the weight, for `count`
/// rows: one thread per (row, block).
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn dense_dequant_kernel(
    pool: &Tensor<u32>,
    out: &mut Tensor<f32>,
    rows: u32,
    row0: u32,
    count: u32,
    blocks_per_row: u32,
    format: u32,
    scales: u32,
    mins: u32,
) {
    let i = ABSOLUTE_POS;
    let bpr = usize::cast_from(blocks_per_row);
    if i < usize::cast_from(count) * bpr {
        let r = i / bpr;
        let b = i % bpr;
        let block = b * usize::cast_from(rows) + usize::cast_from(row0) + r;
        let dst = i * 32usize;
        if format == 4u32 {
            let e = byte_at(pool, usize::cast_from(scales), block);
            for w in 0..4usize {
                let word = pool[block * 4usize + w];
                for k in 0..4usize {
                    let byte = (word >> (u32::cast_from(k) * 8u32)) & 0xFFu32;
                    let j = w * 4usize + k;
                    out[dst + j] = mxfp4_scaled(mxfp4_value(byte & 0xFu32), e);
                    out[dst + j + 16usize] = mxfp4_scaled(mxfp4_value(byte >> 4u32), e);
                }
            }
        } else if format == 2u32 {
            let d = half_at(pool, usize::cast_from(scales), block);
            for w in 0..8usize {
                let word = pool[block * 8usize + w];
                for k in 0..4usize {
                    let byte = (word >> (u32::cast_from(k) * 8u32)) & 0xFFu32;
                    let q = i32::cast_from(byte & 0x7Fu32) - i32::cast_from(byte & 0x80u32);
                    out[dst + w * 4usize + k] = d * f32::cast_from(q);
                }
            }
        } else {
            let d = half_at(pool, usize::cast_from(scales), block);
            let mut m = 0.0f32;
            if format == 1u32 {
                m = half_at(pool, usize::cast_from(mins), block);
            }
            for w in 0..4usize {
                let word = pool[block * 4usize + w];
                for k in 0..8usize {
                    let q = f32::cast_from((word >> (u32::cast_from(k) * 4u32)) & 0xFu32);
                    if format == 0u32 {
                        out[dst + w * 8usize + k] = d * (q - 8.0f32);
                    } else {
                        out[dst + w * 8usize + k] = d * q + m;
                    }
                }
            }
        }
    }
}

// --------------------------------------------------------------- extension

/// Backend operations of the MoE kernels (see the module docs).
#[backend_extension(Cube, Fusion: cfg(feature = "provider-local-gpu"))]
pub trait MoeKernels: burn::backend::Backend {
    /// Router logits `[tokens, n_expert]` to `[ids | weights]`, `tokens *
    /// n_used` each.
    #[fusion(dtype = logits, shape = Shape::new([call.tokens * call.n_used * 2]))]
    fn moe_route(logits: FloatTensor<Self>, call: MoeCall, spec: RouteSpec) -> FloatTensor<Self>;

    /// [`Self::moe_route`] with a per-expert selection bias `[n_expert]`
    /// added to the scores to pick the experts; the weights are the scores
    /// themselves.
    #[fusion(dtype = logits, shape = Shape::new([call.tokens * call.n_used * 2]))]
    fn moe_route_biased(
        logits: FloatTensor<Self>,
        bias: FloatTensor<Self>,
        call: MoeCall,
        spec: RouteSpec,
    ) -> FloatTensor<Self>;

    /// `act(gate . x) * (up . x)` of every pair: `[pairs, ff]`, with the
    /// SwiGLU clamp `clamp` (0: none).
    #[fusion(dtype = x, shape = Shape::new([call.pairs, call.slot.gate.rows]))]
    fn moe_up(
        x: FloatTensor<Self>,
        route: FloatTensor<Self>,
        list: IntTensor<Self>,
        pool: IntTensor<Self>,
        call: MoeCall,
        clamp: f32,
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

    /// `x [tokens, cols]` through the dense weight in `pool`: `[tokens,
    /// rows]`.
    #[fusion(dtype = x, shape = Shape::new([call.tokens, call.layout.rows]))]
    fn dense_gemv(
        x: FloatTensor<Self>,
        pool: IntTensor<Self>,
        call: DenseCall,
    ) -> FloatTensor<Self>;

    /// Rows `call.row0..call.row0 + call.count` of the dense weight in
    /// `pool` as f32 `[count, cols]` (`like` only carries the output's
    /// dtype).
    #[fusion(dtype = like, shape = Shape::new([call.count, call.layout.cols]))]
    fn dense_dequant(
        like: FloatTensor<Self>,
        pool: IntTensor<Self>,
        call: DenseCall,
    ) -> FloatTensor<Self>;
}

fn f32_tensor(like: &CubeTensor, shape: Shape) -> CubeTensor {
    empty_device_contiguous_dtype(like.client.clone(), like.device.clone(), shape, DType::F32)
}

/// The router kernel over `logits`; `bias` is read only when it is given.
fn route_launch(
    logits: CubeTensor,
    bias: Option<CubeTensor>,
    call: &MoeCall,
    spec: RouteSpec,
) -> CubeTensor {
    // The kernels index the buffers linearly: a transposed view (the
    // router matmul's output for several tokens) is made row-major.
    let logits = into_contiguous(logits);
    let has_bias = bias.is_some();
    // Without a bias the kernel never reads the buffer it is bound to.
    let bias = bias.map_or_else(|| f32_tensor(&logits, Shape::new([1])), into_contiguous);
    let out = f32_tensor(&logits, Shape::new([route_len(call)]));
    let client = logits.client.clone();
    let dim = call.n_expert.next_power_of_two().clamp(2, ROUTE_DIM);
    let per = call.n_expert.div_ceil(dim);
    route_kernel::launch(
        &client,
        CubeCount::Static(call.tokens as u32, 1, 1),
        CubeDim::new_1d(dim as u32),
        logits.into_tensor_arg(),
        bias.into_tensor_arg(),
        out.clone().into_tensor_arg(),
        call.tokens as u32,
        u32::from(spec.norm),
        spec.scoring.code(),
        u32::from(has_bias),
        call.scale,
        call.n_expert,
        call.n_used,
        dim,
        per,
        dim * per,
        dim.trailing_zeros() as usize,
    );
    out
}

impl MoeKernels for CubeBackend {
    fn moe_route(logits: CubeTensor, call: MoeCall, spec: RouteSpec) -> CubeTensor {
        route_launch(logits, None, &call, spec)
    }

    fn moe_route_biased(
        logits: CubeTensor,
        bias: CubeTensor,
        call: MoeCall,
        spec: RouteSpec,
    ) -> CubeTensor {
        route_launch(logits, Some(bias), &call, spec)
    }

    fn moe_up(
        x: CubeTensor,
        route: CubeTensor,
        list: CubeTensor,
        pool: CubeTensor,
        call: MoeCall,
        clamp: f32,
    ) -> CubeTensor {
        let (g, u) = (&call.slot.gate, &call.slot.up);
        let x = into_contiguous(x);
        let out = f32_tensor(&x, Shape::new([call.pairs, g.rows]));
        let client = x.client.clone();
        let lanes = expert_lanes(g.cols / BLOCK);
        let rows = g.rows.div_ceil(EXPERT_GROUP / lanes) as u32;
        up_kernel::launch(
            &client,
            CubeCount::Static(rows, call.pairs as u32, 1),
            CubeDim::new_1d(EXPERT_GROUP as u32),
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
            clamp,
            lanes as u32,
            EXPERT_GROUP,
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
        let lanes = expert_lanes(d.cols / BLOCK);
        let rows = d.rows.div_ceil(EXPERT_GROUP / lanes) as u32;
        down_kernel::launch(
            &client,
            CubeCount::Static(rows, call.pairs as u32, 1),
            CubeDim::new_1d(EXPERT_GROUP as u32),
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
            lanes as u32,
            EXPERT_GROUP,
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

    fn dense_gemv(x: CubeTensor, pool: CubeTensor, call: DenseCall) -> CubeTensor {
        let x = into_contiguous(x);
        let l = &call.layout;
        let out = f32_tensor(&x, Shape::new([call.tokens, l.rows]));
        let lanes = gemv_lanes(l.rows, l.blocks_per_row(), call.tokens);
        let per_group = GEMV_GROUP / lanes;
        let client = x.client.clone();
        dense_gemv_kernel::launch(
            &client,
            CubeCount::Static(l.rows.div_ceil(per_group) as u32, call.tokens as u32, 1),
            CubeDim::new_1d(GEMV_GROUP as u32),
            x.into_tensor_arg(),
            pool.into_tensor_arg(),
            out.clone().into_tensor_arg(),
            l.rows as u32,
            l.blocks_per_row() as u32,
            l.format.code(),
            l.scale_offset() as u32,
            l.min_offset() as u32,
            lanes as u32,
            GEMV_GROUP,
        );
        out
    }

    fn dense_dequant(like: CubeTensor, pool: CubeTensor, call: DenseCall) -> CubeTensor {
        let l = &call.layout;
        let out = f32_tensor(&like, Shape::new([call.count, l.cols]));
        let client = like.client.clone();
        let threads = call.count * l.blocks_per_row();
        let dim = CubeDim::new(&client, threads);
        let count = calculate_cube_count_elemwise(&client, threads, dim);
        dense_dequant_kernel::launch(
            &client,
            count,
            dim,
            pool.into_tensor_arg(),
            out.clone().into_tensor_arg(),
            l.rows as u32,
            call.row0 as u32,
            call.count as u32,
            l.blocks_per_row() as u32,
            l.format.code(),
            l.scale_offset() as u32,
            l.min_offset() as u32,
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

/// [`MoeKernels::moe_route`] on tensors: the softmax scores, no bias.
pub(crate) fn route(logits: BTensor<2>, call: &MoeCall, norm: bool) -> BTensor<1> {
    route_scored(logits, None, call, Scoring::Softmax, norm)
}

/// [`MoeKernels::moe_route`] on tensors with the scoring function `scoring`
/// and an optional selection bias `[n_expert]` (added to the scores to pick
/// the experts, not to the weights).
pub(crate) fn route_scored(
    logits: BTensor<2>,
    bias: Option<BTensor<1>>,
    call: &MoeCall,
    scoring: Scoring,
    norm: bool,
) -> BTensor<1> {
    let spec = RouteSpec { scoring, norm };
    BTensor::from_dispatch(match bias {
        None => <Dispatch as MoeKernels>::moe_route(logits.into_dispatch(), call.clone(), spec),
        Some(bias) => <Dispatch as MoeKernels>::moe_route_biased(
            logits.into_dispatch(),
            bias.into_dispatch(),
            call.clone(),
            spec,
        ),
    })
}

/// [`MoeKernels::moe_up`] on tensors, without the SwiGLU clamp.
#[cfg(test)]
pub(crate) fn up(
    x: BTensor<2>,
    route: BTensor<1>,
    list: BTensor<1, Int>,
    pool: BTensor<1, Int>,
    call: &MoeCall,
) -> BTensor<2> {
    up_clamped(x, route, list, pool, call, 0.0)
}

/// [`MoeKernels::moe_up`] on tensors with the SwiGLU clamp `clamp` (0: none).
pub(crate) fn up_clamped(
    x: BTensor<2>,
    route: BTensor<1>,
    list: BTensor<1, Int>,
    pool: BTensor<1, Int>,
    call: &MoeCall,
    clamp: f32,
) -> BTensor<2> {
    BTensor::from_dispatch(<Dispatch as MoeKernels>::moe_up(
        x.into_dispatch(),
        route.into_dispatch(),
        list.into_dispatch(),
        pool.into_dispatch(),
        call.clone(),
        clamp,
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

/// [`MoeKernels::dense_gemv`] on tensors.
pub(crate) fn dense_gemv(x: BTensor<2>, pool: BTensor<1, Int>, call: &DenseCall) -> BTensor<2> {
    BTensor::from_dispatch(<Dispatch as MoeKernels>::dense_gemv(
        x.into_dispatch(),
        pool.into_dispatch(),
        call.clone(),
    ))
}

/// [`MoeKernels::dense_dequant`] on tensors; `like` is any f32 tensor.
pub(crate) fn dense_dequant(
    like: BTensor<1>,
    pool: BTensor<1, Int>,
    call: &DenseCall,
) -> BTensor<2> {
    BTensor::from_dispatch(<Dispatch as MoeKernels>::dense_dequant(
        like.into_dispatch(),
        pool.into_dispatch(),
        call.clone(),
    ))
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use burn::tensor::{Device, TensorData};

    use super::*;
    use crate::provider::local::model::csa::Gating;
    use crate::provider::local::model::gpu::{kernel_test_device, quantize_q8_0, read_f32};
    use crate::provider::local::model::moe::route_with;
    use crate::provider::local::model::params::MoeSpec;
    use crate::provider::local::model::quant::e8m0_half;
    use crate::provider::local::model::tiny::mxfp4_bytes;

    const MXFP4: GgufType = GgufType::Mxfp4;
    const Q8_0: GgufType = GgufType::Ggml(GgmlDType::Q8_0);

    /// A small deterministic generator (the same sequence on every run).
    struct Rng(u64);

    impl Rng {
        fn bits(&mut self) -> u32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 33) as u32
        }

        /// Uniform in `[-0.5, 0.5)`.
        fn unit(&mut self) -> f32 {
            self.bits() as f32 / (1u64 << 31) as f32 - 0.5
        }

        fn below(&mut self, n: usize) -> usize {
            self.bits() as usize % n
        }
    }

    /// `f(0), .., f(n - 1)` computed on every core, in order.
    fn par_collect<T: Send>(n: usize, f: impl Fn(usize) -> T + Sync) -> Vec<T> {
        let threads = std::thread::available_parallelism()
            .map_or(1, |p| p.get())
            .min(n.max(1));
        let chunk = n.div_ceil(threads);
        std::thread::scope(|s| {
            let f = &f;
            let handles: Vec<_> = (0..threads)
                .map(|i| {
                    s.spawn(move || {
                        (i * chunk..((i + 1) * chunk).min(n))
                            .map(f)
                            .collect::<Vec<T>>()
                    })
                })
                .collect();
            handles
                .into_iter()
                .flat_map(|h| h.join().unwrap())
                .collect()
        })
    }

    /// A `[rows, cols]` matrix as the file bytes of a type.
    struct Raw {
        dtype: GgufType,
        bytes: Vec<u8>,
    }

    impl Raw {
        /// Random values in `[-amp / 2, amp / 2)` quantized to `dtype` (MXFP4
        /// by the tiny model's quantizer, Q8_0 by the engine's).
        fn random(dtype: GgufType, seed: u64, rows: usize, cols: usize, amp: f32) -> Self {
            let mut rng = Rng(seed);
            let values: Vec<f32> = (0..rows * cols).map(|_| rng.unit() * amp).collect();
            let bytes = if dtype == MXFP4 {
                mxfp4_bytes(&values)
            } else {
                let mut bytes = Vec::new();
                quantize_q8_0(&values, &mut bytes);
                bytes
            };
            Self { dtype, bytes }
        }

        fn row_bytes(&self, cols: usize) -> usize {
            self.dtype.byte_len(cols).unwrap()
        }

        /// Row `r`, dequantized.
        fn row(&self, r: usize, cols: usize) -> Vec<f32> {
            let n = self.row_bytes(cols);
            let mut out = vec![0f32; cols];
            dequant_row(self.dtype, &self.bytes[r * n..(r + 1) * n], 1.0, &mut out).unwrap();
            out
        }

        /// The whole matrix, dequantized.
        fn dequantized(&self, rows: usize, cols: usize) -> Vec<f32> {
            (0..rows).flat_map(|r| self.row(r, cols)).collect()
        }

        /// `self . x` of every row, accumulated in f64.
        fn mat_vec(&self, rows: usize, cols: usize, x: &[f32]) -> Vec<f32> {
            par_collect(rows, |r| dot64(&self.row(r, cols), x))
        }
    }

    fn dot64(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(a, b)| f64::from(*a) * f64::from(*b))
            .sum::<f64>() as f32
    }

    /// An expert of the three matrices as file bytes.
    struct Expert {
        gate: Raw,
        up: Raw,
        down: Raw,
    }

    impl Expert {
        fn dtypes(&self) -> [GgufType; 3] {
            [self.gate.dtype, self.up.dtype, self.down.dtype]
        }

        /// `down . (swiglu(gate . x, up . x))` on the host.
        fn forward(&self, x: &[f32], ff: usize, act: u32, clamp: f32) -> Vec<f32> {
            let n_embd = x.len();
            let g = self.gate.mat_vec(ff, n_embd, x);
            let u = self.up.mat_vec(ff, n_embd, x);
            let h: Vec<f32> = g
                .iter()
                .zip(&u)
                .map(|(g, u)| swiglu(*g, *u, act, clamp))
                .collect();
            self.down.mat_vec(n_embd, ff, &h)
        }
    }

    /// `Glu::apply` of the CPU engine (the activations as the kernel's).
    fn swiglu(g: f32, u: f32, act: u32, clamp: f32) -> f32 {
        let (g, u) = if clamp > 0.0 {
            (g.min(clamp), u.clamp(-clamp, clamp))
        } else {
            (g, u)
        };
        let g = f64::from(g);
        let a = if act == 0 {
            g / (1.0 + (-g).exp())
        } else {
            0.5 * g * (1.0 + (0.797_884_6 * (g + 0.044_715 * g * g * g)).tanh())
        };
        (a * f64::from(u)) as f32
    }

    /// `slots` experts of the given types, the weights in `[-amp / 2, amp / 2)`.
    fn experts(
        dtypes: [GgufType; 3],
        n_embd: usize,
        ff: usize,
        slots: usize,
        seed: u64,
        amp: f32,
    ) -> Vec<Expert> {
        let mut matrices = par_collect(slots * 3, |i| {
            let (rows, cols) = if i % 3 == 2 {
                (n_embd, ff)
            } else {
                (ff, n_embd)
            };
            Raw::random(dtypes[i % 3], seed + 7919 * i as u64, rows, cols, amp)
        })
        .into_iter();
        (0..slots)
            .map(|_| Expert {
                gate: matrices.next().unwrap(),
                up: matrices.next().unwrap(),
                down: matrices.next().unwrap(),
            })
            .collect()
    }

    fn upload_pool(device: &Device, layout: &SlotLayout, experts: &[Expert]) -> BTensor<1, Int> {
        let mut words: Vec<i32> = Vec::with_capacity(experts.len() * layout.words);
        for e in experts {
            let packed = pack_expert(
                layout,
                e.dtypes(),
                &e.gate.bytes,
                &e.up.bytes,
                &e.down.bytes,
            )
            .unwrap();
            words.extend(packed.iter().map(|w| *w as i32));
        }
        let n = words.len();
        BTensor::<1, Int>::from_data(TensorData::new(words, [n]), device)
    }

    /// Rows `[t, n]` of `[-1, 1)` whose row `i` does not depend on `t`.
    fn random_x(t: usize, n: usize) -> Vec<f32> {
        (0..t)
            .flat_map(|tok| {
                let mut rng = Rng(2000 + tok as u64);
                (0..n).map(move |_| rng.unit() * 2.0).collect::<Vec<_>>()
            })
            .collect()
    }

    /// `[ids | weights]` of `t` tokens of `n_used` distinct slots out of
    /// `slots`, the weights positive and summing to one per token; the row of a
    /// token does not depend on `t`.
    fn random_route(t: usize, n_used: usize, slots: usize) -> Vec<f32> {
        let mut out = vec![0f32; t * n_used * 2];
        for tok in 0..t {
            let mut rng = Rng(1000 + tok as u64);
            let mut order: Vec<usize> = (0..slots).collect();
            let mut weights = Vec::with_capacity(n_used);
            for j in 0..n_used {
                let pick = j + rng.below(slots - j);
                order.swap(j, pick);
                out[tok * n_used + j] = order[j] as f32;
                weights.push(rng.unit() + 0.6);
            }
            let sum: f32 = weights.iter().sum();
            for (j, w) in weights.iter().enumerate() {
                out[t * n_used + tok * n_used + j] = w / sum;
            }
        }
        out
    }

    /// The up, down and sum kernels over a resident pool (identity slots).
    fn run_experts(
        device: &Device,
        layout: &SlotLayout,
        pool: &BTensor<1, Int>,
        x: &[f32],
        route: &[f32],
        (t, n_used, slots): (usize, usize, usize),
        act: u32,
        clamp: f32,
    ) -> Vec<f32> {
        let n_embd = layout.down.rows;
        let call = MoeCall {
            tokens: t,
            n_used,
            n_expert: slots,
            pairs: t * n_used,
            use_list: false,
            act,
            scale: 1.0,
            slot: *layout,
        };
        let x = BTensor::<2>::from_data(TensorData::new(x.to_vec(), [t, n_embd]), device);
        let route = BTensor::<1>::from_data(TensorData::new(route.to_vec(), [route.len()]), device);
        let none = BTensor::<1, Int>::zeros([2], device);
        let hidden = up_clamped(x, route.clone(), none.clone(), pool.clone(), &call, clamp);
        let scratch = BTensor::<2>::zeros([t * n_used, n_embd], device);
        let scratch = down(hidden, route, none, pool.clone(), scratch, &call);
        read_f32(sum(scratch, &call), "experts").unwrap()
    }

    /// Milliseconds per layer of `reps` layers of up, down and sum queued
    /// back to back over one pool (one read back at the end), the way a
    /// resident decode step runs them: the kernels' time without a host
    /// round trip per layer.
    fn queued_experts_ms(
        device: &Device,
        layout: &SlotLayout,
        pool: &BTensor<1, Int>,
        (t, n_used, slots): (usize, usize, usize),
        reps: usize,
    ) -> f64 {
        let n_embd = layout.down.rows;
        let call = MoeCall {
            tokens: t,
            n_used,
            n_expert: slots,
            pairs: t * n_used,
            use_list: false,
            act: 0,
            scale: 1.0,
            slot: *layout,
        };
        let x = BTensor::<2>::from_data(TensorData::new(random_x(t, n_embd), [t, n_embd]), device);
        let route = random_route(t, n_used, slots);
        let route = BTensor::<1>::from_data(TensorData::new(route.clone(), [route.len()]), device);
        let none = BTensor::<1, Int>::zeros([2], device);
        let run = |x: BTensor<2>| {
            let hidden = up_clamped(x, route.clone(), none.clone(), pool.clone(), &call, 0.0);
            let scratch = BTensor::<2>::zeros([t * n_used, n_embd], device);
            let scratch = down(
                hidden,
                route.clone(),
                none.clone(),
                pool.clone(),
                scratch,
                &call,
            );
            sum(scratch, &call)
        };
        read_f32(run(x.clone()), "warm").unwrap();
        let started = Instant::now();
        let mut y = x.clone();
        for _ in 0..reps {
            y = run(y);
        }
        read_f32(y, "queued").unwrap();
        started.elapsed().as_secs_f64() * 1e3 / reps as f64
    }

    /// `|got - want| <= tol * (|want| + 1% of the largest |want|)` everywhere.
    fn assert_close(got: &[f32], want: &[f32], tol: f32, what: &str) {
        assert_eq!(got.len(), want.len(), "{what}: length");
        let max = want.iter().fold(0f32, |m, v| m.max(v.abs()));
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!(
                (g - w).abs() <= tol * (w.abs() + 0.01 * max),
                "{what}: element {i}: gpu {g} vs host {w} (largest {max})"
            );
        }
    }

    /// The host expectation for the tokens `check` of a pass: `sum_j w_j *
    /// expert_j(x)`.
    fn host_moe(
        experts: &[Expert],
        x: &[f32],
        route: &[f32],
        (t, n_used, ff): (usize, usize, usize),
        (act, clamp): (u32, f32),
        tok: usize,
    ) -> Vec<f32> {
        let n_embd = x.len() / t;
        let mut out = vec![0f64; n_embd];
        for j in 0..n_used {
            let e = route[tok * n_used + j] as usize;
            let w = f64::from(route[t * n_used + tok * n_used + j]);
            let y = experts[e].forward(&x[tok * n_embd..(tok + 1) * n_embd], ff, act, clamp);
            for (o, y) in out.iter_mut().zip(&y) {
                *o += w * f64::from(*y);
            }
        }
        out.into_iter().map(|v| v as f32).collect()
    }

    // ------------------------------------------------------------ packing

    /// The 32 values of block `block` read back from kernel words the way the
    /// kernels read them: values at word `block * 4`, the scale byte `block`
    /// of the byte run at word `scales`.
    fn unpack_block(words: &[u32], scales: usize, block: usize) -> Vec<f32> {
        const TABLE: [f32; 16] = [
            0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 0.0, -1.0, -2.0, -3.0, -4.0, -6.0, -8.0, -12.0,
        ];
        let d = e8m0_half(scale_at(&words[scales..], block, 8) as u8);
        let mut out = vec![0f32; 32];
        for j in 0..16 {
            let byte = (words[block * 4 + j / 4] >> (8 * (j % 4))) & 0xFF;
            out[j] = TABLE[(byte & 0xF) as usize] * d;
            out[j + 16] = TABLE[(byte >> 4) as usize] * d;
        }
        out
    }

    #[test]
    fn mxfp4_words_unpack_to_dequant_row_in_both_layouts() {
        let (rows, cols) = (5usize, 96usize);
        let raw = Raw::random(MXFP4, 7, rows, cols, 2.0);
        let want = raw.dequantized(rows, cols);
        let bpr = cols / BLOCK;
        let blocks = rows * bpr;

        // A slot's matrix: row-major blocks, then the scale bytes.
        let m = MatrixLayout {
            format: ExpertFormat::Mxfp4,
            rows,
            cols,
            offset: 0,
        };
        let mut words = Vec::new();
        pack_matrix(&m, MXFP4, &raw.bytes, &mut words).unwrap();
        assert_eq!(words.len(), m.words());
        assert_eq!(m.words(), blocks * 4 + blocks.div_ceil(4));
        assert_eq!(m.scale_offset(), blocks * 4);
        for block in 0..blocks {
            let got = unpack_block(&words, m.scale_offset(), block);
            assert_eq!(
                got,
                want[block * BLOCK..(block + 1) * BLOCK],
                "block {block}"
            );
        }

        // The dense layout: block `b` of row `r` is block `b * rows + r`; 15
        // blocks leave the last scale word partly empty.
        let d = DenseLayout::new(MXFP4, rows, cols).unwrap();
        assert_eq!(d.words(), blocks * 4 + blocks.div_ceil(4));
        let words = pack_dense(&d, MXFP4, &raw.bytes).unwrap();
        assert_eq!(words.len(), d.words());
        for r in 0..rows {
            for b in 0..bpr {
                let got = unpack_block(&words, d.scale_offset(), b * rows + r);
                let at = r * cols + b * BLOCK;
                assert_eq!(got, want[at..at + BLOCK], "row {r} block {b}");
            }
        }

        // MXFP4 is read as MXFP4 only.
        assert_eq!(ExpertFormat::of(MXFP4), ExpertFormat::Mxfp4);
        assert_eq!(ExpertFormat::of(GgmlDType::Q4_0), ExpertFormat::Q4_0);
        let mut sink = Vec::new();
        assert!(pack_matrix(&m, Q8_0, &raw.bytes, &mut sink).is_err());
        assert!(pack_matrix(&m, MXFP4, &raw.bytes[17..], &mut sink).is_err());
        let q4 = MatrixLayout {
            format: ExpertFormat::Q4_0,
            ..m
        };
        assert!(pack_matrix(&q4, MXFP4, &raw.bytes, &mut sink).is_err());
    }

    #[test]
    fn mxfp4_slot_bytes_at_the_real_shape() {
        // 3 matrices of 2304 x 5120 values: 17 bytes per 32 values.
        let layout = SlotLayout::new(5120, 2304, [MXFP4; 3]).unwrap();
        assert_eq!(layout.bytes(), 3 * 2304 * 5120 / 32 * 17);
        assert_eq!(layout.bytes(), 18_800_640);
    }

    // ------------------------------------------------------------ experts

    #[test]
    fn mxfp4_experts_match_the_host_at_tiny_shapes() {
        let Some(device) = kernel_test_device() else {
            return;
        };
        // Square, wide and narrow experts, and a slot that mixes formats.
        let configs = [
            (64usize, 64usize, [MXFP4; 3]),
            (96, 160, [MXFP4; 3]),
            (32, 96, [MXFP4; 3]),
            (64, 64, [MXFP4, MXFP4, Q8_0]),
            (64, 96, [Q8_0, MXFP4, MXFP4]),
            // 13 and 9 blocks per row: the blocks do not divide over the lanes.
            (416, 288, [MXFP4; 3]),
            (416, 288, [Q8_0; 3]),
        ];
        let (slots, n_used) = (6usize, 2usize);
        for (n_embd, ff, dtypes) in configs {
            let layout = SlotLayout::new(n_embd, ff, dtypes).unwrap();
            let all = experts(dtypes, n_embd, ff, slots, 11, 1.0);
            let pool = upload_pool(&device, &layout, &all);
            for act in [0u32, 1] {
                for t in [1usize, 3, 5] {
                    let x = random_x(t, n_embd);
                    let route = random_route(t, n_used, slots);
                    let got = run_experts(
                        &device,
                        &layout,
                        &pool,
                        &x,
                        &route,
                        (t, n_used, slots),
                        act,
                        0.0,
                    );
                    for tok in 0..t {
                        let want = host_moe(&all, &x, &route, (t, n_used, ff), (act, 0.0), tok);
                        assert_close(
                            &got[tok * n_embd..(tok + 1) * n_embd],
                            &want,
                            1e-4,
                            &format!("{dtypes:?} {n_embd}x{ff} act={act} t={t} token {tok}"),
                        );
                    }
                }
            }
        }
    }

    /// The lanes of a row follow the row's blocks, never the launch.
    #[test]
    fn expert_lanes_depend_on_the_layout_alone() {
        for (blocks, want) in [(1, 1), (2, 2), (3, 2), (7, 4), (8, 8), (72, 8), (160, 8)] {
            assert_eq!(expert_lanes(blocks), want, "{blocks} blocks");
        }
        assert!(EXPERT_GROUP.is_multiple_of(EXPERT_LANES));
    }

    #[test]
    fn swiglu_clamp_matches_the_host() {
        let Some(device) = kernel_test_device() else {
            return;
        };
        let (n_embd, ff, slots, n_used, t) = (64usize, 64usize, 6usize, 3usize, 4usize);
        let layout = SlotLayout::new(n_embd, ff, [MXFP4; 3]).unwrap();
        // Large weights, so that the gate and up outputs cross the limit.
        let all = experts([MXFP4; 3], n_embd, ff, slots, 31, 3.0);
        let pool = upload_pool(&device, &layout, &all);
        let x = random_x(t, n_embd);
        let route = random_route(t, n_used, slots);
        for act in [0u32, 1] {
            let run = |clamp: f32| {
                run_experts(
                    &device,
                    &layout,
                    &pool,
                    &x,
                    &route,
                    (t, n_used, slots),
                    act,
                    clamp,
                )
            };
            let plain = run(0.0);
            for clamp in [0.5f32, 2.0, 10.0] {
                let got = run(clamp);
                for tok in 0..t {
                    let want = host_moe(&all, &x, &route, (t, n_used, ff), (act, clamp), tok);
                    assert_close(
                        &got[tok * n_embd..(tok + 1) * n_embd],
                        &want,
                        1e-4,
                        &format!("act={act} clamp={clamp} token {tok}"),
                    );
                }
                if clamp < 5.0 {
                    let max = plain.iter().fold(0f32, |m, v| m.max(v.abs()));
                    let moved = got
                        .iter()
                        .zip(&plain)
                        .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
                    assert!(moved > 1e-2 * max, "clamp {clamp} changed nothing");
                }
            }
            // Zero is no clamp.
            assert_eq!(run(0.0), plain);
        }
    }

    #[test]
    fn mxfp4_experts_match_the_host_at_the_real_shape() {
        let Some(device) = kernel_test_device() else {
            return;
        };
        let (n_embd, ff, slots, n_used) = (5120usize, 2304usize, 8usize, 6usize);
        let layout = SlotLayout::new(n_embd, ff, [MXFP4; 3]).unwrap();
        let started = Instant::now();
        let all = experts([MXFP4; 3], n_embd, ff, slots, 5, 0.1);
        let pool = upload_pool(&device, &layout, &all);
        println!(
            "real-shape pool of {slots} slots ({} bytes each) built and uploaded in {:.1} s",
            layout.bytes(),
            started.elapsed().as_secs_f64()
        );
        let mut first: Option<Vec<f32>> = None;
        for (t, check) in [
            (1usize, vec![0usize]),
            (4, vec![0, 3]),
            (33, vec![0, 17, 32]),
        ] {
            let x = random_x(t, n_embd);
            let route = random_route(t, n_used, slots);
            let run = || {
                run_experts(
                    &device,
                    &layout,
                    &pool,
                    &x,
                    &route,
                    (t, n_used, slots),
                    0,
                    0.0,
                )
            };
            println!(
                "real-shape queued, t={t}: {:.3} ms per layer",
                queued_experts_ms(&device, &layout, &pool, (t, n_used, slots), 20)
            );
            let _warm = run();
            // Median of 15 runs, each including the read back.
            let mut times = Vec::new();
            let mut got = Vec::new();
            for _ in 0..15 {
                let started = Instant::now();
                got = run();
                times.push(started.elapsed().as_secs_f64() * 1e3);
            }
            times.sort_by(f64::total_cmp);
            println!(
                "real-shape up+down+sum, t={t} ({} pairs): median {:.2} ms, best {:.2} ms",
                t * n_used,
                times[times.len() / 2],
                times[0]
            );
            for tok in check {
                let want = host_moe(&all, &x, &route, (t, n_used, ff), (0, 0.0), tok);
                assert_close(
                    &got[tok * n_embd..(tok + 1) * n_embd],
                    &want,
                    1e-4,
                    &format!("real shape t={t} token {tok}"),
                );
            }
            // A row's result does not depend on the other rows of the launch.
            let token0 = got[..n_embd].to_vec();
            match &first {
                None => first = Some(token0),
                Some(f) => assert_eq!(&token0, f, "token 0 at t={t}"),
            }
        }
    }

    // ------------------------------------------------------------- dense

    fn upload_dense(device: &Device, layout: &DenseLayout, raw: &[u8]) -> BTensor<1, Int> {
        let words: Vec<i32> = pack_dense(layout, MXFP4, raw)
            .unwrap()
            .into_iter()
            .map(|w| w as i32)
            .collect();
        let n = words.len();
        BTensor::<1, Int>::from_data(TensorData::new(words, [n]), device)
    }

    /// `dense_gemv` of `ts` input rows and `dense_dequant` of the row ranges
    /// `slices` against the host, on the rows `sample` (all of them when
    /// empty). Returns the milliseconds of the last gemv.
    fn check_dense(
        device: &Device,
        raw: &Raw,
        (rows, cols): (usize, usize),
        ts: &[usize],
        sample: &[usize],
        slices: &[(usize, usize)],
    ) -> f64 {
        let layout = DenseLayout::new(MXFP4, rows, cols).unwrap();
        let pool = upload_dense(device, &layout, &raw.bytes);
        let all: Vec<usize> = (0..rows).collect();
        let sample = if sample.is_empty() { &all } else { sample };
        let host_rows: Vec<Vec<f32>> = sample.iter().map(|&r| raw.row(r, cols)).collect();
        let mut ms = 0.0;
        for &t in ts {
            let x = random_x(t, cols);
            let xt = BTensor::<2>::from_data(TensorData::new(x.clone(), [t, cols]), device);
            let call = DenseCall {
                layout,
                tokens: t,
                row0: 0,
                count: rows,
            };
            let run = || read_f32(dense_gemv(xt.clone(), pool.clone(), &call), "gemv").unwrap();
            let _warm = run();
            let started = Instant::now();
            let got = run();
            ms = started.elapsed().as_secs_f64() * 1e3;
            for tok in 0..t {
                let want: Vec<f32> = host_rows
                    .iter()
                    .map(|row| dot64(row, &x[tok * cols..(tok + 1) * cols]))
                    .collect();
                let have: Vec<f32> = sample.iter().map(|&r| got[tok * rows + r]).collect();
                assert_close(&have, &want, 1e-4, &format!("gemv {rows}x{cols} t={t}"));
            }
        }
        let like = BTensor::<1>::zeros([1], device);
        for &(row0, count) in slices {
            let call = DenseCall {
                layout,
                tokens: 1,
                row0,
                count,
            };
            let got =
                read_f32(dense_dequant(like.clone(), pool.clone(), &call), "dequant").unwrap();
            assert_eq!(got.len(), count * cols);
            for r in 0..count {
                let want = raw.row(row0 + r, cols);
                for (c, (g, w)) in got[r * cols..(r + 1) * cols].iter().zip(&want).enumerate() {
                    // A denormal result may be flushed to zero.
                    assert!(
                        g == w || (w.abs() < f32::MIN_POSITIVE && g.abs() <= w.abs()),
                        "dequant {rows}x{cols} row {} column {c}: gpu {g} vs host {w}",
                        row0 + r
                    );
                }
            }
        }
        ms
    }

    #[test]
    fn mxfp4_dense_matches_the_host() {
        let Some(device) = kernel_test_device() else {
            return;
        };
        // Odd block counts (the last scale word is partly empty), one block
        // per row, and the query projection of the real model.
        for (rows, cols, ts) in [
            (7usize, 96usize, vec![1usize, 3]),
            (70, 64, vec![1, 5]),
            (33, 32, vec![1, 2]),
            (1280, 5120, vec![1, 4, 33]),
        ] {
            let raw = Raw::random(MXFP4, 3, rows, cols, 2.0);
            check_dense(
                &device,
                &raw,
                (rows, cols),
                &ts,
                &[],
                &[(0, rows), (rows / 3, rows / 2)],
            );
        }
    }

    #[test]
    fn mxfp4_dense_dequant_handles_every_scale_byte() {
        let Some(device) = kernel_test_device() else {
            return;
        };
        // Scale bytes from the denormal exponents (0, 1) to the overflowing
        // ones, as `quant::e8m0_half` reads them.
        let (rows, cols) = (13usize, 96usize);
        let mut raw = Raw::random(MXFP4, 9, rows, cols, 2.0);
        let scales = [0u8, 1, 2, 3, 100, 126, 127, 128, 200, 250, 254, 255, 130];
        for (b, block) in raw.bytes.as_chunks_mut::<17>().0.iter_mut().enumerate() {
            block[0] = scales[b % scales.len()];
        }
        let layout = DenseLayout::new(MXFP4, rows, cols).unwrap();
        let pool = upload_dense(&device, &layout, &raw.bytes);
        let call = DenseCall {
            layout,
            tokens: 1,
            row0: 0,
            count: rows,
        };
        let got = read_f32(
            dense_dequant(BTensor::<1>::zeros([1], &device), pool, &call),
            "dequant",
        )
        .unwrap();
        let want = raw.dequantized(rows, cols);
        let (mut flushed, mut exact) = (0, 0);
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            let block = i / BLOCK;
            if g == w {
                exact += 1;
            } else if w.abs() < f32::MIN_POSITIVE && g.abs() <= w.abs() {
                flushed += 1;
            } else {
                panic!(
                    "element {i} (scale byte {}): gpu {g} vs host {w}",
                    raw.bytes[block * 17]
                );
            }
        }
        println!("every-scale dequant: {exact} exact, {flushed} flushed denormals");
    }

    /// A weight the size of the model's LM head (129280 x 5120, 351 MB),
    /// random blocks.
    #[test]
    fn mxfp4_dense_matches_the_host_at_the_lm_head_shape() {
        let Some(device) = kernel_test_device() else {
            return;
        };
        let (rows, cols) = (129_280usize, 5120usize);
        let mut rng = Rng(77);
        let mut bytes = vec![0u8; rows * cols / BLOCK * 17];
        for block in bytes.as_chunks_mut::<17>().0 {
            block[0] = 118 + rng.below(8) as u8;
            for pair in block[1..].as_chunks_mut::<4>().0 {
                pair.copy_from_slice(&rng.bits().to_le_bytes());
            }
        }
        let raw = Raw {
            dtype: MXFP4,
            bytes,
        };
        let sample = [0usize, 1, 63, 64, 12_345, 65_535, 65_536, 100_001, rows - 1];
        let ms = check_dense(
            &device,
            &raw,
            (rows, cols),
            &[1, 4],
            &sample,
            &[(0, 70), (rows - 280, 280)],
        );
        println!("lm-head-shape dense_gemv t=4: {ms:.2} ms");
    }

    // ------------------------------------------------------------ router

    /// Logits with many exact ties, values beyond softplus's threshold of 20
    /// and very negative ones.
    const LOGITS: [f32; 13] = [
        -40.0, -3.0, -1.0, 0.0, 0.5, 1.0, 2.0, 2.0, 5.0, 19.0, 21.0, 25.0, 30.0,
    ];
    const BIASES: [f32; 4] = [-0.25, 0.0, 0.125, 0.5];

    /// Whether the (biased) scores of a row separate the `k`th from the
    /// `k + 1`th expert by more than float noise (or tie exactly), so that the
    /// host and the device must agree on the picks.
    fn picks_are_stable(row: &[f32], bias: Option<&[f32]>, scoring: Scoring, k: usize) -> bool {
        let max = row.iter().fold(f32::MIN, |m, v| m.max(*v));
        let mut keys: Vec<f64> = row
            .iter()
            .map(|&l| {
                let l = f64::from(l);
                match scoring {
                    Scoring::Softmax => (l - f64::from(max)).exp(),
                    Scoring::SqrtSoftplus => (if l > 20.0 { l } else { l.exp().ln_1p() }).sqrt(),
                }
            })
            .collect();
        if let Some(bias) = bias {
            for (key, b) in keys.iter_mut().zip(bias) {
                *key += f64::from(*b);
            }
        }
        keys.sort_by(|a, b| b.total_cmp(a));
        k >= keys.len() || {
            let gap = keys[k - 1] - keys[k];
            gap == 0.0 || gap > 1e-4 * keys[k - 1].abs()
        }
    }

    #[test]
    fn the_router_matches_route_with() {
        let Some(device) = kernel_test_device() else {
            return;
        };
        let slot = SlotLayout::new(32, 32, [MXFP4; 3]).unwrap();
        // (kernel scoring, host gating, renormalise, selection bias).
        let modes = [
            (Scoring::Softmax, Gating::Softmax, true, false),
            (Scoring::Softmax, Gating::Softmax, false, false),
            (Scoring::Softmax, Gating::Softmax, true, true),
            (Scoring::SqrtSoftplus, Gating::SqrtSoftplus, true, false),
            (Scoring::SqrtSoftplus, Gating::SqrtSoftplus, false, false),
            (Scoring::SqrtSoftplus, Gating::SqrtSoftplus, true, true),
            (Scoring::SqrtSoftplus, Gating::SqrtSoftplus, false, true),
        ];
        let (mut compared, mut skipped) = ([0usize; 7], [0usize; 7]);
        // The real router has 384 experts: two rounds of the widest cube (256
        // threads), the second one partly empty. 512 fills both, 100 is not a
        // power of two.
        for n_expert in [8usize, 64, 100, 384, 512] {
            let mut rng = Rng(n_expert as u64);
            let logits: Vec<f32> = (0..33 * n_expert)
                .map(|_| LOGITS[rng.below(LOGITS.len())])
                .collect();
            let bias: Vec<f32> = (0..n_expert)
                .map(|_| BIASES[rng.below(BIASES.len())])
                .collect();
            for n_used in [3usize, 6, 8] {
                for (mode, (scoring, gating, norm, with_bias)) in modes.into_iter().enumerate() {
                    let bias = with_bias.then_some(bias.as_slice());
                    let scale = if norm { 1.0 } else { 1.5 };
                    // The picks of the 33-token pass, bit for bit, which the
                    // shorter passes must repeat.
                    let mut whole: Vec<(Vec<u32>, Vec<u32>)> = Vec::new();
                    for t in [33usize, 5, 1] {
                        let call = MoeCall {
                            tokens: t,
                            n_used,
                            n_expert,
                            pairs: t * n_used,
                            use_list: false,
                            act: 0,
                            scale,
                            slot,
                        };
                        let rows = &logits[..t * n_expert];
                        let logits_t = BTensor::<2>::from_data(
                            TensorData::new(rows.to_vec(), [t, n_expert]),
                            &device,
                        );
                        let bias_t = bias.map(|b| {
                            BTensor::<1>::from_data(
                                TensorData::new(b.to_vec(), [n_expert]),
                                &device,
                            )
                        });
                        let got = read_f32(
                            route_scored(logits_t, bias_t, &call, scoring, norm),
                            "route",
                        )
                        .unwrap();
                        let spec = MoeSpec {
                            n_expert,
                            n_used,
                            norm_weights: norm,
                            scale,
                        };
                        let host = route_with(rows, t, &spec, gating, bias);
                        for tok in 0..t {
                            let at =
                                |base: usize| &got[base + tok * n_used..base + (tok + 1) * n_used];
                            let (ids, weights) = (at(0), at(t * n_used));
                            let bits =
                                |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
                            // A token's picks do not depend on the other rows.
                            if t == 33 {
                                whole.push((bits(ids), bits(weights)));
                            } else {
                                assert_eq!(
                                    (bits(ids), bits(weights)),
                                    whole[tok],
                                    "{scoring:?} norm={norm} bias={with_bias} {n_expert}x{n_used}: token {tok} at t={t} differs from t=33"
                                );
                            }
                            let row = &rows[tok * n_expert..(tok + 1) * n_expert];
                            if !picks_are_stable(row, bias, scoring, n_used) {
                                skipped[mode] += 1;
                                continue;
                            }
                            compared[mode] += 1;
                            let mut got: Vec<(usize, f32)> = ids
                                .iter()
                                .map(|&i| i as usize)
                                .zip(weights.iter().copied())
                                .collect();
                            got.sort_by_key(|p| p.0);
                            let want: Vec<(usize, f32)> = host
                                .per_expert
                                .iter()
                                .enumerate()
                                .flat_map(|(e, p)| {
                                    p.iter()
                                        .filter(move |q| q.0 as usize == tok)
                                        .map(move |q| (e, q.1))
                                })
                                .collect();
                            let label = format!(
                                "{scoring:?} norm={norm} bias={with_bias} {n_expert} experts top {n_used} t={t} token {tok}"
                            );
                            assert_eq!(
                                got.iter().map(|p| p.0).collect::<Vec<_>>(),
                                want.iter().map(|p| p.0).collect::<Vec<_>>(),
                                "{label}: experts"
                            );
                            for (g, w) in got.iter().zip(&want) {
                                assert!(
                                    (g.1 - w.1).abs() <= 1e-4 * w.1.abs() + 1e-6,
                                    "{label}: expert {} weight {} vs {}",
                                    g.0,
                                    g.1,
                                    w.1
                                );
                            }
                        }
                    }
                }
            }
        }
        println!(
            "router: tokens compared with route_with per mode {compared:?}, skipped as near ties {skipped:?}"
        );
        // The modes without a bias have no ties but the exact ones.
        assert!(
            skipped[0] + skipped[1] + skipped[3] + skipped[4] == 0,
            "unbiased routes skipped"
        );
        assert!(
            compared.iter().sum::<usize>() > 10 * skipped.iter().sum::<usize>(),
            "too many tokens skipped"
        );
    }
}
