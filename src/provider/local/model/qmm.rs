//! Row-batched quantized matmul for every quantized matrix dtype the engine
//! reads: the legacy blocks (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0), the K-quants
//! (Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K), the I-quants (IQ1_S, IQ1_M, IQ2_XXS,
//! IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS), the ternary TQ1_0 and
//! TQ2_0, MXFP4 and NVFP4. No quantized matrix runs on candle's kernel.
//!
//! candle's quantized matmul loops over the activation rows and streams the
//! whole weight matrix once per row, so verifying `k + 1` drafted tokens in one
//! forward pass costs close to `k + 1` decode steps. [`QmmWeight::forward_rows`]
//! reads every weight block once and dots it against all `m` activation rows,
//! so a batch of `m` rows costs about one row of memory traffic plus `m` times
//! the (much cheaper) integer arithmetic.
//!
//! # Arithmetic
//!
//! The activations are quantised to Q8_0 blocks of 32 values (`d = max|x| /
//! 127` rounded to f16, `q = round(x * (1 / d))`), exactly as candle's vec-dot
//! partners do, plus, for a format with minima, a sum term the minimum
//! multiplies:
//!
//! - Q4_1, Q5_1: ggml's Q8_1 term `s = f16(f16(sum q) * d)` of the 32 quants;
//! - Q2_K, Q4_K, Q5_K: the two half sums `d * sum q` of the first and the
//!   last 16 quants, each the exact integer sum times `d` rounded once to f32
//!   (ggml's `Q8_K` keeps the same sixteen-value sums).
//!
//! A weight row is a sequence of *entries*, each the quants of 32 consecutive
//! values as `i16`, one f32 scale and, with minima, two f32 minima (one per
//! 16 values) that the format defines from its block ([`Format::entries`]).
//! The integer sub-block scales of a K-quant or I-quant are folded into the
//! quants (`sc * q`, at most 4096 in magnitude), so the entry's scale is the
//! block's own factor; NVFP4's four 16-value scales cannot be folded, so each
//! 32 values make two entries whose quants are zero in the other half.
//!
//! The value for activation row `j` and weight row `r` is a fixed function of
//! that row pair alone: sixteen `f32` lanes, each adding, entry by entry in
//! row order, its share of the exact integer dot of the entry's quants with
//! the activation block's quants times the product of the two f32 scales (how
//! the integer part is computed is free), reduced by a fixed pairwise tree,
//! plus a scalar sum of minimum terms added after the reduction, block by
//! block (Q4_1, Q5_1: the minimum times `s`; Q2_K, Q4_K, Q5_K: the first
//! minimum times the first half sum, then the second times the second, the
//! minima being the negated `dmin * mn`). No cross-row term exists and tile
//! shape never changes the order, so a row's result is bit-identical whether
//! it is computed alone or in a batch of any size, which speculative-decoding
//! verification needs. It differs from candle's and ggml's own results only
//! in float rounding order and, for the K-quants and I-quants, in the
//! activation format (Q8_0 blocks of 32 instead of Q8_K blocks of 256).
//!
//! Work is split in chunks of weight rows over candle's persistent barrier pool
//! (the one its own quantized matmul used, so the two never oversubscribe the
//! cores), with dynamic chunk claiming; which worker computes a chunk never
//! changes its bits.
//!
//! One dtype is served by exactly one kernel: a row's bits must not depend on
//! the batch size, so the dispatch cannot pick a faster kernel per `m`.

use std::sync::{Arc, Mutex, PoisonError};

use candle_core::utils::barrier_pool;
use half::f16;
use wide::{f32x4, f32x16, i8x16, i8x32, i16x16, i16x32, u8x32, u32x8, u64x4};

use super::pool::for_each;
use super::quant::block::{MAX_BLOCK, Unpacked};
use super::quant::iq_tables::{
    IQ1S_GRID, IQ2S_GRID, IQ2XS_GRID, IQ2XXS_GRID, IQ3S_GRID, IQ3XXS_GRID, KSIGNS_IQ2XS,
    KVALUES_FP4, KVALUES_IQ4NL,
};
use super::quant::{iquants as iq, kquants as kq};
use crate::provider::local::gguf::GgufType;

/// Elements per quantisation block of the served formats (Q4_0, Q4_1, Q5_0,
/// Q5_1, Q8_0), and of their Q8_0/Q8_1 activations.
const QK: usize = 32;
/// Lanes: a block's two halves of sixteen elements are summed lane-wise.
const LANES: usize = QK / 2;
/// Weight rows decoded together.
const NR: usize = 4;
/// Activation rows per register tile.
const MR: usize = 4;
/// Below this many weight elements the matmul runs on the calling thread.
const SERIAL_BELOW: usize = 1 << 17;

/// A format without minima.
const MIN_NONE: u8 = 0;
/// One minimum per 32 values (Q4_1, Q5_1), multiplied by Q8_1's sum term.
const MIN_Q8_1: u8 = 1;
/// Two minima per 32 values, one per 16 (Q2_K, Q4_K, Q5_K), multiplied by the
/// exact half sums of the activation block.
const MIN_HALVES: u8 = 2;
/// Most entries one block decodes to (a 256-value block of 32-value entries).
const MAX_ENTRIES: usize = 8;

/// A block's 32 quants as two halves of sixteen: lane `l` of the first half is
/// element `l`, lane `l` of the second is element `l + 16`. The definition of
/// a format's quants ([`Legacy::halves`]).
type Halves = [[i16; LANES]; 2];

/// A block's 32 quants as `i16` lanes, element `l` in lane `2 * l` and element
/// `l + 16` in lane `2 * l + 1`: one multiply-add of two such vectors
/// (`vpmaddwd`) yields the sixteen lane sums the module docs define, with no
/// further shuffle. A product of a weight quant (`-128..=127`) and an
/// activation quant (`-127..=127`) fits an `i16` and the sum of two fits an
/// `i32` exactly.
type Quants = i16x32;

/// [`Halves`] in the layout of [`Quants`].
#[inline(always)]
fn interleave(h: &Halves) -> Quants {
    let mut lanes = [0i16; QK];
    for l in 0..LANES {
        lanes[2 * l] = h[0][l];
        lanes[2 * l + 1] = h[1][l];
    }
    Quants::new(lanes)
}

/// The 16 nibble bytes of a block as 32 bytes `[lo0, hi0, lo1, hi1, ...]`:
/// byte `i` holds element `i` in its low nibble and element `i + 16` in its
/// high one, which is the pair [`Quants`] puts side by side.
#[inline(always)]
fn nibble_pairs(nibbles: &[u8]) -> i8x32 {
    let bytes: [i8; LANES] = std::array::from_fn(|i| nibbles[i] as i8);
    let wide = i16x16::from_i8x16(i8x16::new(bytes)) & i16x16::splat(0xFF);
    let pairs = (wide & i16x16::splat(0x0F)) | ((wide & i16x16::splat(0xF0)) << 4u32);
    i8x32::from_ne_bytes(pairs.to_ne_bytes())
}

/// One entry of a weight row: 32 values' quants, f32 scale and minima. The
/// definition of a format ([`Format::entries`]).
#[derive(Clone, Copy, Default)]
struct Entry {
    halves: Halves,
    scale: f32,
    /// `[min of the first 16 values, min of the last 16]`; the element is
    /// `scale * quant + min` (Q4_1, Q5_1: only `mins[0]`, for all 32).
    mins: [f32; 2],
}

/// Where [`Format::decode`] writes one block's entries.
struct Dst<'a> {
    quants: &'a mut [Quants],
    scales: &'a mut [f32],
    /// One pair per 32 values of the block; empty for a format without minima.
    mins: &'a mut [[f32; 2]],
}

/// A weight format: blocks of bytes that decode to 32-value entries.
trait Format {
    type Block;
    /// Which minima the blocks carry (`MIN_*`).
    const MIN: u8 = MIN_NONE;
    /// Entries per 32 values: 2 when the two halves of 32 values have
    /// scales that cannot be folded (NVFP4), else 1. A format with 2 has no
    /// minima.
    const SUB: usize = 1;
    /// Values per block.
    const VALUES: usize = QK;
    /// Entries per block.
    const ENTRIES: usize = Self::VALUES / QK * Self::SUB;
    /// The blocks of one row (`row` is a whole number of them).
    fn blocks(row: &[u8]) -> &[Self::Block];
    /// The block's entries, scalar: the format's definition. `out` has
    /// [`Format::ENTRIES`] elements.
    fn entries(block: &Self::Block, out: &mut [Entry]);
    /// [`Format::entries`] as vectors, into `dst`. A format that overrides it
    /// with vector code writes the same values (`decode_equals_halves`).
    #[inline(always)]
    fn decode(block: &Self::Block, dst: &mut Dst) {
        let mut e = [Entry::default(); MAX_ENTRIES];
        let e = &mut e[..Self::ENTRIES];
        Self::entries(block, e);
        for ((q, s), e) in dst.quants.iter_mut().zip(&mut *dst.scales).zip(&*e) {
            *q = interleave(&e.halves);
            *s = e.scale;
        }
        for (m, e) in dst.mins.iter_mut().zip(e.iter().step_by(Self::SUB)) {
            *m = e.mins;
        }
    }
}

/// A format of 32-value blocks with one scale and at most one minimum.
trait Legacy {
    type Block;
    /// Blocks carry a minimum (`element = scale * quant + min`), paired with
    /// Q8_1 activations whose sum term it multiplies.
    const HAS_MIN: bool = false;
    /// The blocks of one row (`row` is a whole number of them).
    fn blocks(row: &[u8]) -> &[Self::Block];
    fn scale(block: &Self::Block) -> f32;
    fn min(_block: &Self::Block) -> f32 {
        0.0
    }
    /// The block's quants, scalar: the format's definition.
    fn halves(block: &Self::Block) -> Halves;
    /// [`Legacy::halves`] as [`Quants`]. A format that overrides it with
    /// vector code returns the same lanes (`decode_equals_halves`).
    #[inline(always)]
    fn decode(block: &Self::Block) -> Quants {
        interleave(&Self::halves(block))
    }
}

impl<L: Legacy> Format for L {
    type Block = L::Block;
    const MIN: u8 = if L::HAS_MIN { MIN_Q8_1 } else { MIN_NONE };

    #[inline(always)]
    fn blocks(row: &[u8]) -> &[Self::Block] {
        L::blocks(row)
    }

    #[inline(always)]
    fn entries(block: &Self::Block, out: &mut [Entry]) {
        out[0] = Entry {
            halves: L::halves(block),
            scale: L::scale(block),
            mins: [L::min(block), 0.0],
        };
    }

    #[inline(always)]
    fn decode(block: &Self::Block, dst: &mut Dst) {
        dst.quants[0] = L::decode(block);
        dst.scales[0] = L::scale(block);
        if L::HAS_MIN {
            dst.mins[0] = [L::min(block), 0.0];
        }
    }
}

/// The f16 at byte `at` of a block, widened.
#[inline(always)]
fn f16_le(block: &[u8], at: usize) -> f32 {
    f16::from_le_bytes([block[at], block[at + 1]]).to_f32()
}

/// `d: f16`, then 16 bytes of nibbles: element `i` is the low nibble of byte
/// `i` and element `i + 16` its high nibble, each minus 8.
struct Q4_0;

impl Legacy for Q4_0 {
    type Block = [u8; 18];

    #[inline(always)]
    fn blocks(row: &[u8]) -> &[Self::Block] {
        row.as_chunks().0
    }

    #[inline(always)]
    fn scale(block: &Self::Block) -> f32 {
        f16_le(block, 0)
    }

    #[inline(always)]
    fn halves(block: &Self::Block) -> Halves {
        let mut w = [[0i16; LANES]; 2];
        for i in 0..LANES {
            let b = block[2 + i];
            w[0][i] = i16::from(b & 0x0F) - 8;
            w[1][i] = i16::from(b >> 4) - 8;
        }
        w
    }

    #[inline(always)]
    fn decode(block: &Self::Block) -> Quants {
        Quants::from_i8x32(nibble_pairs(&block[2..18]) - i8x32::splat(8))
    }
}

/// `d: f16`, `m: f16`, then 16 bytes of nibbles: element = nibble `* d + m`.
struct Q4_1;

impl Legacy for Q4_1 {
    type Block = [u8; 20];
    const HAS_MIN: bool = true;

    #[inline(always)]
    fn blocks(row: &[u8]) -> &[Self::Block] {
        row.as_chunks().0
    }

    #[inline(always)]
    fn scale(block: &Self::Block) -> f32 {
        f16_le(block, 0)
    }

    #[inline(always)]
    fn min(block: &Self::Block) -> f32 {
        f16_le(block, 2)
    }

    #[inline(always)]
    fn halves(block: &Self::Block) -> Halves {
        let mut w = [[0i16; LANES]; 2];
        for i in 0..LANES {
            let b = block[4 + i];
            w[0][i] = i16::from(b & 0x0F);
            w[1][i] = i16::from(b >> 4);
        }
        w
    }

    #[inline(always)]
    fn decode(block: &Self::Block) -> Quants {
        Quants::from_i8x32(nibble_pairs(&block[4..20]))
    }
}

/// The 5-bit quants of elements `i` and `i + 16` of a Q5_0 or Q5_1 block: the
/// low and high nibble of `nibbles[i]` extended by bits `i` and `i + 16` of
/// the 32 high bits `qh`.
#[inline(always)]
fn q5_halves(nibbles: &[u8], qh: u32) -> Halves {
    let mut w = [[0i16; LANES]; 2];
    for i in 0..LANES {
        let b = nibbles[i];
        w[0][i] = i16::from(b & 0x0F) | i16::from((qh >> i) as u8 & 1) << 4;
        w[1][i] = i16::from(b >> 4) | i16::from((qh >> (i + 16)) as u8 & 1) << 4;
    }
    w
}

/// `d: f16`, 32 high bits, then 16 bytes of nibbles: element = (5-bit quant -
/// 16) `* d`.
struct Q5_0;

impl Legacy for Q5_0 {
    type Block = [u8; 22];

    #[inline(always)]
    fn blocks(row: &[u8]) -> &[Self::Block] {
        row.as_chunks().0
    }

    #[inline(always)]
    fn scale(block: &Self::Block) -> f32 {
        f16_le(block, 0)
    }

    #[inline(always)]
    fn halves(block: &Self::Block) -> Halves {
        let qh = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);
        let mut w = q5_halves(&block[6..22], qh);
        for half in &mut w {
            for q in half {
                *q -= 16;
            }
        }
        w
    }
}

/// `d: f16`, `m: f16`, 32 high bits, then 16 bytes of nibbles: element =
/// 5-bit quant `* d + m`.
struct Q5_1;

impl Legacy for Q5_1 {
    type Block = [u8; 24];
    const HAS_MIN: bool = true;

    #[inline(always)]
    fn blocks(row: &[u8]) -> &[Self::Block] {
        row.as_chunks().0
    }

    #[inline(always)]
    fn scale(block: &Self::Block) -> f32 {
        f16_le(block, 0)
    }

    #[inline(always)]
    fn min(block: &Self::Block) -> f32 {
        f16_le(block, 2)
    }

    #[inline(always)]
    fn halves(block: &Self::Block) -> Halves {
        let qh = u32::from_le_bytes([block[4], block[5], block[6], block[7]]);
        q5_halves(&block[8..24], qh)
    }
}

/// Byte `2 * l` of the interleaved layout is element `l`, byte `2 * l + 1`
/// element `l + 16`: the shuffle that turns a block's 32 bytes in element order
/// into [`Quants`]' order.
const INTERLEAVE: u8x32 = {
    let mut at = [0u8; QK];
    let mut l = 0;
    while l < LANES {
        at[2 * l] = l as u8;
        at[2 * l + 1] = (l + LANES) as u8;
        l += 1;
    }
    u8x32::new(at)
};

/// `d: f16`, then 32 signed bytes.
struct Q8_0;

impl Legacy for Q8_0 {
    type Block = [u8; 34];

    #[inline(always)]
    fn blocks(row: &[u8]) -> &[Self::Block] {
        row.as_chunks().0
    }

    #[inline(always)]
    fn scale(block: &Self::Block) -> f32 {
        f16_le(block, 0)
    }

    #[inline(always)]
    fn halves(block: &Self::Block) -> Halves {
        let mut w = [[0i16; LANES]; 2];
        for i in 0..LANES {
            w[0][i] = i16::from(block[2 + i] as i8);
            w[1][i] = i16::from(block[2 + LANES + i] as i8);
        }
        w
    }

    #[inline(always)]
    fn decode(block: &Self::Block) -> Quants {
        let bytes: [i8; QK] = std::array::from_fn(|i| block[2 + i] as i8);
        Quants::from_i8x32(i8x32::new(bytes).shuffle(INTERLEAVE))
    }
}

/// A 16-entry code table as a byte table, once in each 128-bit half of a
/// vector (a byte shuffle looks up within its half).
const fn byte_table(values: &[i8; 16]) -> i8x32 {
    let mut t = [0i8; QK];
    let mut i = 0;
    while i < QK {
        t[i] = values[i % 16];
        i += 1;
    }
    i8x32::new(t)
}

/// The doubled E2M1 values an MXFP4 nibble selects (ggml's `kvalues_mxfp4`).
const MXFP4_TABLE: i8x32 = byte_table(&KVALUES_FP4);

/// IQ4_NL's 16-entry code table (ggml's `kvalues_iq4nl`).
const IQ4NL_TABLE: i8x32 = byte_table(&KVALUES_IQ4NL);

/// OCP MXFP4: one E8M0 scale byte, then 16 bytes of nibbles (element `i` is
/// the low nibble of byte `i`, element `i + 16` its high nibble) that index
/// the doubled E2M1 table; the block scale is `2^(e - 128)` (the table is
/// doubled, so ggml's scale is the half of the E8M0 value).
struct Mxfp4;

impl Legacy for Mxfp4 {
    type Block = [u8; 17];

    #[inline(always)]
    fn blocks(row: &[u8]) -> &[Self::Block] {
        row.as_chunks().0
    }

    #[inline(always)]
    fn scale(block: &Self::Block) -> f32 {
        super::quant::e8m0_half(block[0])
    }

    #[inline(always)]
    fn halves(block: &Self::Block) -> Halves {
        let mut w = [[0i16; LANES]; 2];
        for i in 0..LANES {
            let b = block[1 + i];
            w[0][i] = i16::from(KVALUES_FP4[usize::from(b & 0x0F)]);
            w[1][i] = i16::from(KVALUES_FP4[usize::from(b >> 4)]);
        }
        w
    }

    #[inline(always)]
    fn decode(block: &Self::Block) -> Quants {
        Quants::from_i8x32(MXFP4_TABLE.swizzle_half_relaxed(nibble_pairs(&block[1..17])))
    }
}

/// `d: f16`, then 16 bytes of nibbles (element `i` is the low nibble of byte
/// `i`, element `i + 16` its high nibble) that index ggml's `kvalues_iq4nl`.
struct Iq4Nl;

impl Legacy for Iq4Nl {
    type Block = [u8; 18];

    #[inline(always)]
    fn blocks(row: &[u8]) -> &[Self::Block] {
        row.as_chunks().0
    }

    #[inline(always)]
    fn scale(block: &Self::Block) -> f32 {
        f16_le(block, 0)
    }

    #[inline(always)]
    fn halves(block: &Self::Block) -> Halves {
        let mut w = [[0i16; LANES]; 2];
        for i in 0..LANES {
            let b = block[2 + i];
            w[0][i] = i16::from(KVALUES_IQ4NL[usize::from(b & 0x0F)]);
            w[1][i] = i16::from(KVALUES_IQ4NL[usize::from(b >> 4)]);
        }
        w
    }

    #[inline(always)]
    fn decode(block: &Self::Block) -> Quants {
        Quants::from_i8x32(IQ4NL_TABLE.swizzle_half_relaxed(nibble_pairs(&block[2..18])))
    }
}

/// The entries of an unpacked block ([`Unpacked`]): the folded quants
/// `sc * q` of each 32 values, the scale of the sub-block they lie in, and
/// with `mins` the negated `dmin * mn` of the sub-blocks of each 16 values.
/// With `sub == 2` every 32 values are two entries holding 16 values each.
fn unpacked_entries(u: &Unpacked, mins: bool, sub: usize, out: &mut [Entry]) {
    let mut folded = [0i16; MAX_BLOCK];
    u.fold_into(&mut folded);
    let group = u.group.max(1);
    for (i, e) in out.iter_mut().enumerate() {
        let at = QK * (i / sub);
        let sixteen = |from: usize| -> [i16; LANES] { std::array::from_fn(|l| folded[from + l]) };
        if sub == 1 {
            e.halves = [sixteen(at), sixteen(at + LANES)];
            e.scale = u.d[at / group];
            if mins {
                e.mins = [-u.min_f32(at / group), -u.min_f32((at + LANES) / group)];
            }
        } else {
            let half = i % sub;
            let from = at + LANES * half;
            e.halves = [[0; LANES]; 2];
            e.halves[half] = sixteen(from);
            e.scale = u.d[from / group];
        }
    }
}

/// A format of 256-value (NVFP4: 64-value) blocks that [`Unpacked`]
/// describes: `$unpack` is the foundation's unpacker and the entries are
/// derived from it ([`unpacked_entries`]), with an optional vector `decode`.
macro_rules! superblock {
    (
        $(#[$meta:meta])*
        $name:ident, $bytes:expr, $unpack:path, min: $min:expr, values: $values:expr, sub: $sub:expr
        $(, decode: $fast:path)?
    ) => {
        $(#[$meta])*
        struct $name;

        impl Format for $name {
            type Block = [u8; $bytes];
            const MIN: u8 = $min;
            const SUB: usize = $sub;
            const VALUES: usize = $values;

            #[inline(always)]
            fn blocks(row: &[u8]) -> &[Self::Block] {
                row.as_chunks().0
            }

            fn entries(block: &Self::Block, out: &mut [Entry]) {
                let mut u = Unpacked::default();
                $unpack(block, &mut u);
                unpacked_entries(&u, $min != MIN_NONE, $sub, out);
            }

            $(
                #[inline(always)]
                fn decode(block: &Self::Block, dst: &mut Dst) {
                    $fast(block, dst)
                }
            )?
        }
    };
}

superblock!(
    /// Q2_K: 16 sub-blocks of 16 values with a 4-bit scale and a 4-bit min.
    Q2K, kq::Q2_K_BYTES, kq::unpack_q2_k, min: MIN_HALVES, values: 256, sub: 1, decode: decode_q2k
);
superblock!(
    /// Q3_K: 16 sub-blocks of 16 values with a 6-bit scale.
    Q3K, kq::Q3_K_BYTES, kq::unpack_q3_k, min: MIN_NONE, values: 256, sub: 1, decode: decode_q3k
);
superblock!(
    /// Q4_K: 8 sub-blocks of 32 values with a 6-bit scale and a 6-bit min.
    Q4K, kq::Q4_K_BYTES, kq::unpack_q4_k, min: MIN_HALVES, values: 256, sub: 1, decode: decode_q4k
);
superblock!(
    /// Q5_K: Q4_K with a fifth bit.
    Q5K, kq::Q5_K_BYTES, kq::unpack_q5_k, min: MIN_HALVES, values: 256, sub: 1, decode: decode_q5k
);
superblock!(
    /// Q6_K: 16 sub-blocks of 16 values with an 8-bit scale.
    Q6K, kq::Q6_K_BYTES, kq::unpack_q6_k, min: MIN_NONE, values: 256, sub: 1, decode: decode_q6k
);
superblock!(
    /// Q8_K: an f32 scale and 256 signed bytes.
    Q8K, kq::Q8_K_BYTES, kq::unpack_q8_k, min: MIN_NONE, values: 256, sub: 1
);
superblock!(
    /// IQ2_XXS: 8 sub-blocks of 32 values from a grid, with a sign set and a
    /// 4-bit scale each.
    Iq2Xxs, iq::IQ2_XXS_BYTES, iq::unpack_iq2_xxs, min: MIN_NONE, values: 256, sub: 1, decode: decode_iq2xxs
);
superblock!(
    /// IQ2_XS: 16 sub-blocks of 16 values from a larger grid.
    Iq2Xs, iq::IQ2_XS_BYTES, iq::unpack_iq2_xs, min: MIN_NONE, values: 256, sub: 1, decode: decode_iq2xs
);
superblock!(
    /// IQ2_S: IQ2_XS with explicit sign bytes.
    Iq2S, iq::IQ2_S_BYTES, iq::unpack_iq2_s, min: MIN_NONE, values: 256, sub: 1, decode: decode_iq2s
);
superblock!(
    /// IQ3_XXS: 8 sub-blocks of 32 values from a grid of 4.
    Iq3Xxs, iq::IQ3_XXS_BYTES, iq::unpack_iq3_xxs, min: MIN_NONE, values: 256, sub: 1, decode: decode_iq3xxs
);
superblock!(
    /// IQ3_S: IQ3_XXS with explicit sign bytes.
    Iq3S, iq::IQ3_S_BYTES, iq::unpack_iq3_s, min: MIN_NONE, values: 256, sub: 1, decode: decode_iq3s
);
superblock!(
    /// IQ1_S: ternary grid points with a delta, 8 sub-blocks of 32 values.
    Iq1S, iq::IQ1_S_BYTES, iq::unpack_iq1_s, min: MIN_NONE, values: 256, sub: 1, decode: decode_iq1s
);
superblock!(
    /// IQ1_M: IQ1_S with a delta per 16 values.
    Iq1M, iq::IQ1_M_BYTES, iq::unpack_iq1_m, min: MIN_NONE, values: 256, sub: 1, decode: decode_iq1m
);
superblock!(
    /// IQ4_XS: 8 sub-blocks of 32 values through the IQ4_NL table.
    Iq4Xs, iq::IQ4_XS_BYTES, iq::unpack_iq4_xs, min: MIN_NONE, values: 256, sub: 1, decode: decode_iq4xs
);
superblock!(
    /// TQ1_0: 256 ternary values, five per byte.
    Tq1_0, iq::TQ1_0_BYTES, iq::unpack_tq1_0, min: MIN_NONE, values: 256, sub: 1, decode: decode_tq1_0
);
superblock!(
    /// TQ2_0: 256 ternary values, two bits each.
    Tq2_0, iq::TQ2_0_BYTES, iq::unpack_tq2_0, min: MIN_NONE, values: 256, sub: 1, decode: decode_tq2_0
);
superblock!(
    /// NVFP4: 64 E2M1 values as four sub-blocks of 16 with a UE4M3 scale each.
    Nvfp4, iq::NVFP4_BYTES, iq::unpack_nvfp4, min: MIN_NONE, values: 64, sub: 2
);

/// 16 bytes as 16 lanes holding `0..=255`.
#[inline(always)]
fn bytes16(bytes: &[u8]) -> i16x16 {
    i16x16::from_i8x16(i8x16::new(std::array::from_fn(|i| bytes[i] as i8))) & i16x16::splat(0xFF)
}

/// The quants of an entry whose elements `l` and `l + 16` are lane `l` of `a`
/// and of `b` (each `0..=127`): [`Quants`]' layout is element `l` in lane `2 *
/// l` and element `l + 16` in lane `2 * l + 1`.
#[inline(always)]
fn pair_lanes(a: i16x16, b: i16x16) -> Quants {
    Quants::from_i8x32(i8x32::from_ne_bytes((a | (b << 8u32)).to_ne_bytes()))
}

/// The scales of an entry with one per 16 values: `lo` for the elements `0..16`
/// (even lanes of [`Quants`]), `hi` for the elements `16..32` (odd lanes).
#[inline(always)]
fn pair_scale(lo: i16, hi: i16) -> Quants {
    Quants::new(std::array::from_fn(|i| if i % 2 == 0 { lo } else { hi }))
}

/// The 6-bit scale and min of sub-block `j` (0 to 7) out of the 12 packed
/// bytes of a Q4_K or Q5_K block (ggml's `get_scale_min_k4`).
#[inline(always)]
fn scale_min_k4(j: usize, q: &[u8]) -> (i16, i16) {
    let (sc, mn) = if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    };
    (i16::from(sc), i16::from(mn))
}

/// The sixteen 6-bit scales of a Q3_K block, minus 32, out of its 12 packed
/// bytes.
#[inline(always)]
fn q3_scales(sc: &[u8]) -> [i16; 16] {
    std::array::from_fn(|i| {
        let low = if i < 8 { sc[i] & 0x0F } else { sc[i - 8] >> 4 };
        let high = (sc[8 + i % 4] >> (2 * (i / 4))) & 3;
        i16::from(low | (high << 4)) - 32
    })
}

/// [`Format::decode`] of Q2_K: entry `e` holds sub-blocks `2e` and `2e + 1`,
/// whose 2-bit quants lie at shift `2 * (e % 4)` of the 32 bytes `32 * (e /
/// 4)` on.
#[inline(always)]
fn decode_q2k(b: &[u8; kq::Q2_K_BYTES], dst: &mut Dst) {
    let (scales, rest) = b.split_at(16);
    let (qs, rest) = rest.split_at(64);
    let (d, dmin) = (f16_le(rest, 0), f16_le(rest, 2));
    let three = i16x16::splat(3);
    for e in 0..8 {
        let q = &qs[32 * (e / 4)..32 * (e / 4) + 32];
        let shift = 2 * (e % 4) as u32;
        let a = (bytes16(&q[..16]) >> shift) & three;
        let c = (bytes16(&q[16..]) >> shift) & three;
        let (s0, s1) = (scales[2 * e], scales[2 * e + 1]);
        dst.quants[e] = pair_lanes(a, c) * pair_scale(i16::from(s0 & 0x0F), i16::from(s1 & 0x0F));
        dst.scales[e] = d;
        dst.mins[e] = [
            -(dmin * f32::from(i16::from(s0 >> 4))),
            -(dmin * f32::from(i16::from(s1 >> 4))),
        ];
    }
}

/// [`Format::decode`] of Q3_K: as Q2_K, the quant minus 4 where the matching
/// bit of `hmask` is clear.
#[inline(always)]
fn decode_q3k(b: &[u8; kq::Q3_K_BYTES], dst: &mut Dst) {
    let (hmask, rest) = b.split_at(32);
    let (qs, rest) = rest.split_at(64);
    let (packed, rest) = rest.split_at(12);
    let d = f16_le(rest, 0);
    let scales = q3_scales(packed);
    let (ha, hb) = (bytes16(&hmask[..16]), bytes16(&hmask[16..]));
    let (three, one) = (i16x16::splat(3), i16x16::splat(1));
    for e in 0..8 {
        let q = &qs[32 * (e / 4)..32 * (e / 4) + 32];
        let shift = 2 * (e % 4) as u32;
        let a = (bytes16(&q[..16]) >> shift) & three;
        let c = (bytes16(&q[16..]) >> shift) & three;
        let bit = e as u32;
        let (ba, bb) = (((ha >> bit) & one) << 2u32, ((hb >> bit) & one) << 2u32);
        let quants = pair_lanes(a | ba, c | bb) - Quants::splat(4);
        dst.quants[e] = quants * pair_scale(scales[2 * e], scales[2 * e + 1]);
        dst.scales[e] = d;
    }
}

/// [`Format::decode`] of Q4_K: entry `2j` is the low nibbles and entry `2j +
/// 1` the high nibbles of the 32 bytes `32 * j` on.
#[inline(always)]
fn decode_q4k(b: &[u8; kq::Q4_K_BYTES], dst: &mut Dst) {
    let (d, dmin) = (f16_le(b, 0), f16_le(b, 2));
    let nibble = i16x16::splat(0x0F);
    for (j, q) in b[16..].as_chunks::<32>().0.iter().enumerate() {
        let (a, c) = (bytes16(&q[..16]), bytes16(&q[16..]));
        for t in 0..2 {
            let e = 2 * j + t;
            let (sc, mn) = scale_min_k4(e, &b[4..16]);
            let (a, c) = if t == 0 {
                (a & nibble, c & nibble)
            } else {
                ((a >> 4u32) & nibble, (c >> 4u32) & nibble)
            };
            dst.quants[e] = pair_lanes(a, c) * Quants::splat(sc);
            dst.scales[e] = d;
            let m = -(dmin * f32::from(mn));
            dst.mins[e] = [m, m];
        }
    }
}

/// [`Format::decode`] of Q5_K: Q4_K with bit `e` of the 32 high-bit bytes as
/// the fifth bit of entry `e`.
#[inline(always)]
fn decode_q5k(b: &[u8; kq::Q5_K_BYTES], dst: &mut Dst) {
    let (d, dmin) = (f16_le(b, 0), f16_le(b, 2));
    let (qh, qs) = b[16..].split_at(32);
    let (ha, hb) = (bytes16(&qh[..16]), bytes16(&qh[16..]));
    let (nibble, one) = (i16x16::splat(0x0F), i16x16::splat(1));
    for (j, q) in qs.as_chunks::<32>().0.iter().enumerate() {
        let (a, c) = (bytes16(&q[..16]), bytes16(&q[16..]));
        for t in 0..2 {
            let e = 2 * j + t;
            let (sc, mn) = scale_min_k4(e, &b[4..16]);
            let (a, c) = if t == 0 {
                (a & nibble, c & nibble)
            } else {
                ((a >> 4u32) & nibble, (c >> 4u32) & nibble)
            };
            let bit = e as u32;
            let (ba, bb) = (((ha >> bit) & one) << 4u32, ((hb >> bit) & one) << 4u32);
            dst.quants[e] = pair_lanes(a | ba, c | bb) * Quants::splat(sc);
            dst.scales[e] = d;
            let m = -(dmin * f32::from(mn));
            dst.mins[e] = [m, m];
        }
    }
}

/// [`Format::decode`] of Q6_K: entry `4h + e` holds the `e`-th of the four
/// 32-value runs a byte of half `h` carries (low then high nibble of the first
/// and the second 32 `ql` bytes, plus two bits of `qh`), minus 32.
#[inline(always)]
fn decode_q6k(b: &[u8; kq::Q6_K_BYTES], dst: &mut Dst) {
    let (ql, rest) = b.split_at(128);
    let (qh, rest) = rest.split_at(64);
    let (sc, rest) = rest.split_at(16);
    let d = f16_le(rest, 0);
    let (nibble, three) = (i16x16::splat(0x0F), i16x16::splat(3));
    for h in 0..2 {
        let ql = &ql[64 * h..64 * h + 64];
        let qh = &qh[32 * h..32 * h + 32];
        let (ha, hb) = (bytes16(&qh[..16]), bytes16(&qh[16..]));
        for e in 0..4 {
            let src = &ql[32 * (e % 2)..32 * (e % 2) + 32];
            let (a, c) = (bytes16(&src[..16]), bytes16(&src[16..]));
            let (a, c) = if e < 2 {
                (a & nibble, c & nibble)
            } else {
                ((a >> 4u32) & nibble, (c >> 4u32) & nibble)
            };
            let shift = 2 * e as u32;
            let (ba, bb) = (
                ((ha >> shift) & three) << 4u32,
                ((hb >> shift) & three) << 4u32,
            );
            let quants = pair_lanes(a | ba, c | bb) - Quants::splat(32);
            let s = pair_scale(
                i16::from(sc[8 * h + 2 * e].cast_signed()),
                i16::from(sc[8 * h + 2 * e + 1].cast_signed()),
            );
            dst.quants[4 * h + e] = quants * s;
            dst.scales[4 * h + e] = d;
        }
    }
}

/// [`Format::decode`] of IQ4_XS: entry `ib` is 16 bytes of nibbles through the
/// IQ4_NL table times its 6-bit scale minus 32.
#[inline(always)]
fn decode_iq4xs(b: &[u8; iq::IQ4_XS_BYTES], dst: &mut Dst) {
    let d = f16_le(b, 0);
    let high = u16::from_le_bytes([b[2], b[3]]);
    for ib in 0..8 {
        let low = (b[4 + ib / 2] >> (4 * (ib % 2))) & 0x0F;
        let ls = i16::from(low | ((((high >> (2 * ib)) & 3) as u8) << 4)) - 32;
        let table = IQ4NL_TABLE.swizzle_half_relaxed(nibble_pairs(&b[8 + 16 * ib..24 + 16 * ib]));
        dst.quants[ib] = Quants::from_i8x32(table) * Quants::splat(ls);
        dst.scales[ib] = d;
    }
}

/// Byte `j` of entry `s` is `0xFF` where bit `j` of `s` is set: the mask that
/// negates the grid values a sign byte flags.
const SIGN_BYTES: [u64; 256] = {
    let mut t = [0u64; 256];
    let mut s = 0;
    while s < 256 {
        let mut j = 0;
        while j < 8 {
            if (s >> j) & 1 != 0 {
                t[s] |= 0xFF << (8 * j);
            }
            j += 1;
        }
        s += 1;
    }
    t
};

/// Four 8-byte groups (elements `8 * g` on) as 32 bytes in the group order 0,
/// 2, 1, 3, so that each 128-bit half holds the groups [`element_order`] pairs.
#[inline(always)]
fn join_groups(groups: [u64; 4]) -> i8x32 {
    let [g0, g1, g2, g3] = groups;
    i8x32::from_ne_bytes(u64x4::new([g0, g2, g1, g3]).to_ne_bytes())
}

/// Byte `i` of a half of [`join_groups`]' output that goes to byte `i` of the
/// same half in [`Quants`]' layout (element `l` in byte `2 * l`, element `l +
/// 16` in byte `2 * l + 1`): a byte shuffle within each 128-bit half.
const PAIR_BYTES: i8x32 = {
    let mut at = [0i8; QK];
    let mut i = 0;
    while i < QK {
        at[i] = (i % 16 / 2 + 8 * (i % 2)) as i8;
        i += 1;
    }
    i8x32::new(at)
};

/// 32 values of [`join_groups`] as [`Quants`].
#[inline(always)]
fn element_order(values: i8x32) -> Quants {
    Quants::from_i8x32(values.swizzle_half_relaxed(PAIR_BYTES))
}

/// [`SIGN_BYTES`] of ggml's `ksigns_iq2xs` (a 7-bit index to its sign byte):
/// the mask of IQ2_XXS, IQ2_XS and IQ3_XXS in one lookup.
const KSIGN_MASKS: [u64; 128] = {
    let mut t = [0u64; 128];
    let mut i = 0;
    while i < 128 {
        t[i] = SIGN_BYTES[KSIGNS_IQ2XS[i] as usize];
        i += 1;
    }
    t
};

/// A grid of 32 signed values ([`join_groups`] order), negated where `mask`
/// bytes are `0xFF`, as [`Quants`] (unscaled).
#[inline(always)]
fn signed(grid: i8x32, mask: i8x32) -> Quants {
    element_order((grid ^ mask) - mask)
}

/// Eight grid words of 4 values as 32 bytes in [`join_groups`] order.
#[inline(always)]
fn join_words(w: [u32; 8]) -> i8x32 {
    i8x32::from_ne_bytes(u32x8::new([w[0], w[1], w[4], w[5], w[2], w[3], w[6], w[7]]).to_ne_bytes())
}

/// Byte `k` of [`join_groups`]' output belongs to the group whose sign byte is
/// byte `SIGN_OF[k]` of the word that holds the four.
const SIGN_OF: i8x32 = {
    let mut at = [0i8; QK];
    let mut k = 0;
    while k < QK {
        at[k] = [0, 2, 1, 3][k / 8];
        k += 1;
    }
    i8x32::new(at)
};

/// Bit `k % 8` as a byte: the sign bit of value `k` within its group.
const SIGN_BIT: i8x32 = {
    let mut at = [0i8; QK];
    let mut k = 0;
    while k < QK {
        at[k] = (1u8 << (k % 8)) as i8;
        k += 1;
    }
    i8x32::new(at)
};

/// The mask ([`signed`]) of four sign bytes (little-endian in `signs`), one per
/// group of 8 values, bit `j` flagging value `j`.
#[inline(always)]
fn sign_masks(signs: u32) -> i8x32 {
    let bytes = i8x32::from_ne_bytes(u32x8::splat(signs).to_ne_bytes());
    (bytes.swizzle_half_relaxed(SIGN_OF) & SIGN_BIT).simd_eq(SIGN_BIT)
}

/// The scale of an entry with one per 16 values from a 4-bit pair `sc`: `2 *
/// nibble + 1` for each half (IQ2_XS, IQ2_S).
#[inline(always)]
fn nibble_pair_scale(sc: u8) -> Quants {
    pair_scale(i16::from(2 * (sc & 0x0F) + 1), i16::from(2 * (sc >> 4) + 1))
}

/// [`Format::decode`] of IQ2_XXS.
#[inline(always)]
fn decode_iq2xxs(b: &[u8; iq::IQ2_XXS_BYTES], dst: &mut Dst) {
    let d = f16_le(b, 0) * 0.125;
    for ib in 0..8 {
        let qs = &b[2 + 8 * ib..10 + 8 * ib];
        let aux = u32::from_le_bytes([qs[4], qs[5], qs[6], qs[7]]);
        let groups = std::array::from_fn(|l| IQ2XXS_GRID[usize::from(qs[l])]);
        let masks = std::array::from_fn(|l| KSIGN_MASKS[((aux >> (7 * l)) & 127) as usize]);
        let sc = 2 * (aux >> 28) as i16 + 1;
        dst.quants[ib] = signed(join_groups(groups), join_groups(masks)) * Quants::splat(sc);
        dst.scales[ib] = d;
    }
}

/// [`Format::decode`] of IQ2_XS.
#[inline(always)]
fn decode_iq2xs(b: &[u8; iq::IQ2_XS_BYTES], dst: &mut Dst) {
    let d = f16_le(b, 0) * 0.125;
    let (qs, scales) = b[2..].split_at(64);
    for ib in 0..8 {
        let v: [usize; 4] = std::array::from_fn(|l| {
            usize::from(u16::from_le_bytes([
                qs[8 * ib + 2 * l],
                qs[8 * ib + 2 * l + 1],
            ]))
        });
        let groups = v.map(|v| IQ2XS_GRID[v & 511]);
        let masks = v.map(|v| KSIGN_MASKS[v >> 9]);
        dst.quants[ib] =
            signed(join_groups(groups), join_groups(masks)) * nibble_pair_scale(scales[ib]);
        dst.scales[ib] = d;
    }
}

/// [`Format::decode`] of IQ2_S.
#[inline(always)]
fn decode_iq2s(b: &[u8; iq::IQ2_S_BYTES], dst: &mut Dst) {
    let d = f16_le(b, 0) * 0.125;
    let (qs, rest) = b[2..].split_at(32);
    let (signs, rest) = rest.split_at(32);
    let (qh, scales) = rest.split_at(8);
    for ib in 0..8 {
        let high = usize::from(qh[ib]);
        let groups = std::array::from_fn(|l| {
            IQ2S_GRID[usize::from(qs[4 * ib + l]) | ((high << (8 - 2 * l)) & 0x300)]
        });
        let signs = u32::from_le_bytes(std::array::from_fn(|l| signs[4 * ib + l]));
        dst.quants[ib] =
            signed(join_groups(groups), sign_masks(signs)) * nibble_pair_scale(scales[ib]);
        dst.scales[ib] = d;
    }
}

/// [`Format::decode`] of IQ3_XXS.
#[inline(always)]
fn decode_iq3xxs(b: &[u8; iq::IQ3_XXS_BYTES], dst: &mut Dst) {
    let d = f16_le(b, 0) * 0.25;
    let (qs, sas) = b[2..].split_at(64);
    for ib in 0..8 {
        let q = &qs[8 * ib..8 * ib + 8];
        let aux = u32::from_le_bytes([
            sas[4 * ib],
            sas[4 * ib + 1],
            sas[4 * ib + 2],
            sas[4 * ib + 3],
        ]);
        let words = std::array::from_fn(|i| IQ3XXS_GRID[usize::from(q[i])]);
        let masks = std::array::from_fn(|l| KSIGN_MASKS[((aux >> (7 * l)) & 127) as usize]);
        let sc = 2 * (aux >> 28) as i16 + 1;
        dst.quants[ib] = signed(join_words(words), join_groups(masks)) * Quants::splat(sc);
        dst.scales[ib] = d;
    }
}

/// [`Format::decode`] of IQ3_S.
#[inline(always)]
fn decode_iq3s(b: &[u8; iq::IQ3_S_BYTES], dst: &mut Dst) {
    let d = f16_le(b, 0);
    let (qs, rest) = b[2..].split_at(64);
    let (qh, rest) = rest.split_at(8);
    let (signs, scales) = rest.split_at(32);
    for ib in 0..8 {
        let q = &qs[8 * ib..8 * ib + 8];
        let high = usize::from(qh[ib]);
        let words =
            std::array::from_fn(|i| IQ3S_GRID[usize::from(q[i]) | ((high << (8 - i)) & 256)]);
        let signs = u32::from_le_bytes(std::array::from_fn(|l| signs[4 * ib + l]));
        let sc = 1 + 2 * i16::from((scales[ib / 2] >> (4 * (ib % 2))) & 0x0F);
        dst.quants[ib] = signed(join_words(words), sign_masks(signs)) * Quants::splat(sc);
        dst.scales[ib] = d;
    }
}

/// [`Format::decode`] of IQ1_S: the grid values times 8 plus or minus 1, times
/// the 3-bit scale `2 * s + 1`.
#[inline(always)]
fn decode_iq1s(b: &[u8; iq::IQ1_S_BYTES], dst: &mut Dst) {
    let d = f16_le(b, 0) * 0.125;
    let (qs, qh) = b[2..].split_at(32);
    for ib in 0..8 {
        let high = u16::from_le_bytes([qh[2 * ib], qh[2 * ib + 1]]);
        let groups = std::array::from_fn(|l| {
            IQ1S_GRID[usize::from(qs[4 * ib + l]) | (usize::from((high >> (3 * l)) & 7) << 8)]
        });
        let sc = 2 * ((high >> 12) & 7) as i16 + 1;
        let delta = if high & 0x8000 != 0 { -sc } else { sc };
        dst.quants[ib] =
            element_order(join_groups(groups)) * Quants::splat(8 * sc) + Quants::splat(delta);
        dst.scales[ib] = d;
    }
}

/// [`Format::decode`] of IQ1_M: IQ1_S with a delta sign per group of 8 and a
/// scale per 16 values.
#[inline(always)]
fn decode_iq1m(b: &[u8; iq::IQ1_M_BYTES], dst: &mut Dst) {
    let (qs, rest) = b.split_at(32);
    let (qh, scales) = rest.split_at(16);
    let words: [u16; 4] =
        std::array::from_fn(|i| u16::from_le_bytes([scales[2 * i], scales[2 * i + 1]]));
    let bits = (words[0] >> 12)
        | ((words[1] >> 8) & 0x00F0)
        | ((words[2] >> 4) & 0x0F00)
        | (words[3] & 0xF000);
    let d = f16::from_bits(bits).to_f32() * 0.125;
    for ib in 0..8 {
        let word = words[ib / 2];
        let (lo, hi) = (
            2 * ((word >> (6 * (ib % 2))) & 7) as i16 + 1,
            2 * ((word >> (6 * (ib % 2) + 3)) & 7) as i16 + 1,
        );
        let group = |l: usize| {
            let high = qh[2 * ib + l / 2];
            let even = l.is_multiple_of(2);
            let shift = if even { 8 } else { 4 };
            let idx = usize::from(qs[4 * ib + l]) | ((usize::from(high) << shift) & 0x700);
            let negative = high & if even { 0x08 } else { 0x80 } != 0;
            (
                IQ1S_GRID[idx],
                if negative {
                    u64::MAX
                } else {
                    0x0101_0101_0101_0101
                },
            )
        };
        let picked: [(u64, u64); 4] = std::array::from_fn(group);
        let grid = element_order(join_groups(picked.map(|p| p.0)));
        let deltas = element_order(join_groups(picked.map(|p| p.1)));
        dst.quants[ib] = (grid * Quants::splat(8) + deltas) * pair_scale(lo, hi);
        dst.scales[ib] = d;
    }
}

/// `3^n`, the multiplier of digit `n` of a TQ1_0 byte.
const POW3: [i16; 6] = [1, 3, 9, 27, 81, 243];

/// The ternary digit `n` (0 to 2, before the minus 1) of the TQ1_0 bytes `q`
/// (`0..=255` per lane): `((q * 3^n mod 256) * 3) >> 8`.
#[inline(always)]
fn trit(q: i16x16, n: usize) -> i16x16 {
    (((q * i16x16::splat(POW3[n])) & i16x16::splat(0xFF)) * i16x16::splat(3)) >> 8u32
}

/// [`Format::decode`] of TQ1_0: entries 0 to 4 are the five digits of the
/// first 32 bytes, entries 5 and 6 pairs of digits of the next 16, entry 7 the
/// fifth digit of those and the four digits of the last 4 bytes.
#[inline(always)]
fn decode_tq1_0(b: &[u8; iq::TQ1_0_BYTES], dst: &mut Dst) {
    let d = f16_le(b, 52);
    let one = Quants::splat(1);
    let (a, c) = (bytes16(&b[..16]), bytes16(&b[16..32]));
    for (n, q) in dst.quants.iter_mut().enumerate().take(5) {
        *q = pair_lanes(trit(a, n), trit(c, n)) - one;
    }
    let x = bytes16(&b[32..48]);
    for k in 0..2 {
        dst.quants[5 + k] = pair_lanes(trit(x, 2 * k), trit(x, 2 * k + 1)) - one;
    }
    let tail = i16x16::new(std::array::from_fn(|l| {
        let q = (i32::from(b[48 + l % 4]) * i32::from(POW3[l / 4])) & 0xFF;
        ((q * 3) >> 8) as i16
    }));
    dst.quants[7] = pair_lanes(trit(x, 4), tail) - one;
    dst.scales[..8].fill(d);
}

/// [`Format::decode`] of TQ2_0: entry `4c + l` is bits `2 * l` of the 32 bytes
/// `32 * c` on, minus 1.
#[inline(always)]
fn decode_tq2_0(b: &[u8; iq::TQ2_0_BYTES], dst: &mut Dst) {
    let d = f16_le(b, 64);
    let three = i16x16::splat(3);
    for c in 0..2 {
        let q = &b[32 * c..32 * c + 32];
        let (a, e) = (bytes16(&q[..16]), bytes16(&q[16..]));
        for l in 0..4 {
            let shift = 2 * l as u32;
            dst.quants[4 * c + l] =
                pair_lanes((a >> shift) & three, (e >> shift) & three) - Quants::splat(1);
            dst.scales[4 * c + l] = d;
        }
    }
}

/// Activations of one call, quantised to Q8_0 (plus the sum terms of
/// [`Format::MIN`]) and laid out block-major so the blocks of consecutive rows
/// at one block index are contiguous.
struct Acts {
    /// Blocks per row.
    nb: usize,
    /// Rows.
    m: usize,
    /// `[nb][m]` quants in [`Quants`]' layout.
    qs: Vec<Quants>,
    /// `[nb][m]` f16-rounded scales.
    scales: Vec<f32>,
    /// `[nb][m]` Q8_1 sum terms `f16(f16(sum of the quants) * d)`; empty
    /// unless the format is [`MIN_Q8_1`].
    sums: Vec<f32>,
    /// `[nb][m]` half sums `[d * sum of the first 16 quants, d * sum of the
    /// last 16]`; empty unless the format is [`MIN_HALVES`].
    halves: Vec<[f32; 2]>,
}

/// `f32::round` (nearest, ties away from zero) in a form that vectorises:
/// `t - trunc(t)` is exact, so the tie test is too.
#[inline(always)]
fn round_away(t: f32) -> f32 {
    let whole = t.trunc();
    if (t - whole).abs() >= 0.5 {
        whole + 1.0f32.copysign(t)
    } else {
        whole
    }
}

/// One activation block quantised.
struct Quantised {
    quants: Quants,
    /// The f16-rounded scale.
    scale: f32,
    /// The Q8_1 sum term, for [`MIN_Q8_1`].
    sum: f32,
    /// The half sums, for [`MIN_HALVES`].
    halves: [f32; 2],
}

/// One activation block quantised: its quants in [`Quants`]' layout, its
/// f16-rounded scale and the sum terms `min` asks for.
#[inline(always)]
fn quantise_block(src: &[f32; QK], min: u8) -> Quantised {
    // `max` ignores NaN and is associative, so eight running maxima give the
    // sequential fold's value.
    let mut maxima = [0f32; 8];
    for lane in src.as_chunks::<8>().0 {
        for (m8, &v) in maxima.iter_mut().zip(lane) {
            *m8 = m8.max(v.abs());
        }
    }
    let amax = maxima.iter().fold(0f32, |a, &v| a.max(v));
    let d = amax / 127.0;
    let id = if d != 0.0 { 1.0 / d } else { 0.0 };
    let d16 = f16::from_f32(d);
    let mut lanes = [0i16; QK];
    for l in 0..LANES {
        lanes[2 * l] = i16::from(round_away(src[l] * id) as i8);
        lanes[2 * l + 1] = i16::from(round_away(src[l + LANES] * id) as i8);
    }
    let mut sum = 0.0;
    let mut halves = [0.0; 2];
    if min == MIN_Q8_1 {
        let total: i32 = lanes.iter().map(|&q| i32::from(q)).sum();
        sum = (f16::from_f32(total as f32) * d16).to_f32();
    } else if min == MIN_HALVES {
        let (mut lo, mut hi) = (0i32, 0i32);
        for l in 0..LANES {
            lo += i32::from(lanes[2 * l]);
            hi += i32::from(lanes[2 * l + 1]);
        }
        let d = d16.to_f32();
        halves = [lo as f32 * d, hi as f32 * d];
    }
    Quantised {
        quants: Quants::new(lanes),
        scale: d16.to_f32(),
        sum,
        halves,
    }
}

/// One job of [`Acts::new`]: a run of blocks' quants, scales and sum terms.
type Part<'a> = (
    &'a mut [Quants],
    &'a mut [f32],
    &'a mut [f32],
    &'a mut [[f32; 2]],
);

impl Acts {
    /// Quantise `x` (`m` rows of `k` floats, `k` a multiple of [`QK`]) the way
    /// candle's `BlockQ8_0::from_float` does, with `BlockQ8_1::from_float`'s
    /// sum term for [`MIN_Q8_1`] or the half sums for [`MIN_HALVES`].
    ///
    /// A large call splits the blocks over the pool: each block is quantised
    /// from its own 32 values alone, so the split changes no bit.
    fn new(x: &[f32], m: usize, k: usize, min: u8) -> Self {
        let nb = k / QK;
        let mut qs = vec![Quants::default(); nb * m];
        let mut scales = vec![0f32; nb * m];
        let mut sums = vec![0f32; if min == MIN_Q8_1 { nb * m } else { 0 }];
        let mut halves = vec![[0f32; 2]; if min == MIN_HALVES { nb * m } else { 0 }];
        let pooled = m * k >= SERIAL_BELOW && barrier_pool().n_workers() > 0;
        let threads = barrier_pool().n_workers() + 1;
        let per = if pooled {
            nb.div_ceil(threads * 4).max(1)
        } else {
            nb.max(1)
        };
        let width = (per * m).max(1);
        let mut sum_chunks = sums.chunks_mut(width);
        let mut half_chunks = halves.chunks_mut(width);
        let jobs: Vec<Mutex<Part<'_>>> = qs
            .chunks_mut(width)
            .zip(scales.chunks_mut(width))
            .map(|(q, s)| {
                let su: &mut [f32] = if min == MIN_Q8_1 {
                    sum_chunks.next().unwrap_or_default()
                } else {
                    &mut []
                };
                let hs: &mut [[f32; 2]] = if min == MIN_HALVES {
                    half_chunks.next().unwrap_or_default()
                } else {
                    &mut []
                };
                Mutex::new((q, s, su, hs))
            })
            .collect();
        for_each(
            jobs.len(),
            pooled && jobs.len() > 1,
            || (),
            |(), i| {
                let mut part = jobs[i].lock().unwrap_or_else(PoisonError::into_inner);
                let (q, s, su, hs) = &mut *part;
                for bi in 0..q.len() / m {
                    let b = i * per + bi;
                    for j in 0..m {
                        let at = j * k + b * QK;
                        let Ok(src) = <&[f32; QK]>::try_from(&x[at..at + QK]) else {
                            continue;
                        };
                        let block = quantise_block(src, min);
                        q[bi * m + j] = block.quants;
                        s[bi * m + j] = block.scale;
                        if min == MIN_Q8_1 {
                            su[bi * m + j] = block.sum;
                        } else if min == MIN_HALVES {
                            hs[bi * m + j] = block.halves;
                        }
                    }
                }
            },
        );
        drop(jobs);
        Self {
            nb,
            m,
            qs,
            scales,
            sums,
            halves,
        }
    }
}

/// Pairwise tree over the lanes, in a fixed order.
#[inline(always)]
fn hsum(a: &[f32; LANES]) -> f32 {
    let mut h = [0f32; LANES / 2];
    for i in 0..LANES / 2 {
        h[i] = a[i] + a[i + LANES / 2];
    }
    ((h[0] + h[4]) + (h[1] + h[5])) + ((h[2] + h[6]) + (h[3] + h[7]))
}

/// Up to [`NR`] weight rows with every block decoded: the entries' quants and
/// f32 scales and the minima of the rows by `nb` blocks of 32 values, kept
/// apart from the weights so that one decode serves every activation tile.
struct Decoded {
    nb: usize,
    /// `[row][block][sub]`.
    quants: Vec<Quants>,
    scales: Vec<f32>,
    /// `[row][block]`; empty for a format without minima.
    mins: Vec<[f32; 2]>,
}

impl Decoded {
    fn new<F: Format>(nb: usize) -> Self {
        Self {
            nb,
            quants: vec![Quants::default(); NR * nb * F::SUB],
            scales: vec![0f32; NR * nb * F::SUB],
            mins: vec![[0f32; 2]; if F::MIN != MIN_NONE { NR * nb } else { 0 }],
        }
    }

    /// Decode `rows` (at most [`NR`], each `nb` 32-value blocks) into rows `0..`.
    fn load<F: Format>(&mut self, rows: &[&[F::Block]]) {
        let nb = self.nb;
        let per = F::VALUES / QK;
        for (r, row) in rows.iter().enumerate() {
            for (i, block) in row.iter().enumerate() {
                let first = r * nb + i * per;
                let entries = first * F::SUB..(first + per) * F::SUB;
                let mut dst = Dst {
                    quants: &mut self.quants[entries.clone()],
                    scales: &mut self.scales[entries],
                    mins: if F::MIN != MIN_NONE {
                        &mut self.mins[first..first + per]
                    } else {
                        &mut []
                    },
                };
                F::decode(block, &mut dst);
            }
        }
    }
}

/// `d * sc[jj]` for the `M` activation rows of a block: the f32 products the
/// accumulation scales by, four at a time when a tile is full.
#[inline(always)]
fn scale_products<const M: usize>(d: f32, sc: &[f32]) -> [f32; M] {
    let mut s = [0f32; M];
    if M == MR {
        let four = f32x4::splat(d) * f32x4::new([sc[0], sc[1], sc[2], sc[3]]);
        s.copy_from_slice(&four.to_array());
    } else {
        for (p, &x) in s.iter_mut().zip(sc) {
            *p = d * x;
        }
    }
    s
}

/// `R <= 4` decoded weight rows against activation rows `j0..j0 + M` (`M <=
/// 4`): `out[i][jj]` is the dot of weight row `i` with activation row `j0 +
/// jj`, defined as sixteen `f32` lanes that each add, entry by entry in block
/// order, the exact integer sum of the entry's two elements of that lane times
/// the product of the two scales, reduced by [`hsum`]. The accumulators stay
/// in registers across the blocks.
///
/// A format with minima (`MIN`) adds `min * s` per block to a scalar sum in
/// block order after the lanes are reduced: `s` is the activation block's
/// Q8_1 sum term ([`MIN_Q8_1`]), or the two minima times the two half sums,
/// first half first ([`MIN_HALVES`]).
#[inline(always)]
fn tile<const MIN: u8, const SUB: usize, const R: usize, const M: usize>(
    dec: &Decoded,
    act: &Acts,
    j0: usize,
) -> [[f32; M]; R] {
    let nb = act.nb;
    let mut acc = [[f32x16::ZERO; M]; R];
    let mut extra = [[0f32; M]; R];
    let rows: [&[Quants]; R] =
        std::array::from_fn(|r| &dec.quants[r * nb * SUB..(r + 1) * nb * SUB]);
    let scales: [&[f32]; R] =
        std::array::from_fn(|r| &dec.scales[r * nb * SUB..(r + 1) * nb * SUB]);
    for b in 0..nb {
        let at = b * act.m + j0;
        let xs = &act.qs[at..at + M];
        let sc = &act.scales[at..at + M];
        for r in 0..R {
            for t in 0..SUB {
                let w = rows[r][b * SUB + t];
                let s = scale_products::<M>(scales[r][b * SUB + t], sc);
                for jj in 0..M {
                    let p = f32x16::from_i32x16(w.dot(xs[jj]));
                    acc[r][jj] += p * f32x16::splat(s[jj]);
                }
            }
        }
        if MIN == MIN_Q8_1 {
            let sums = &act.sums[at..at + M];
            for (r, row) in extra.iter_mut().enumerate() {
                let [mn, _] = dec.mins[r * nb + b];
                for (e, &s) in row.iter_mut().zip(sums) {
                    *e += mn * s;
                }
            }
        } else if MIN == MIN_HALVES {
            let halves = &act.halves[at..at + M];
            for (r, row) in extra.iter_mut().enumerate() {
                let [lo, hi] = dec.mins[r * nb + b];
                for (e, h) in row.iter_mut().zip(halves) {
                    *e += lo * h[0];
                    *e += hi * h[1];
                }
            }
        }
    }
    let mut out = [[0f32; M]; R];
    for r in 0..R {
        for jj in 0..M {
            let lanes = acc[r][jj].to_array();
            out[r][jj] = if MIN == MIN_NONE {
                hsum(&lanes)
            } else {
                hsum(&lanes) + extra[r][jj]
            };
        }
    }
    out
}

/// Store `tile[i][jj]` (weight row `r0 + i`, activation row `j0 + jj`) into the
/// chunk's `res[j * rows + r]` scratch.
#[inline(always)]
fn put<const R: usize, const M: usize>(
    tile: [[f32; M]; R],
    res: &mut [f32],
    rows: usize,
    r0: usize,
    j0: usize,
) {
    for (i, tile_row) in tile.iter().enumerate() {
        for (jj, v) in tile_row.iter().enumerate() {
            res[(j0 + jj) * rows + r0 + i] = *v;
        }
    }
}

/// `R` decoded weight rows against every activation row: full [`MR`]-row
/// tiles, then one tile for the remaining 1..=3 rows. One instance per shape
/// (`MIN`, `SUB`, `R`), shared by every format of it.
#[inline(never)]
fn tiles<const MIN: u8, const SUB: usize, const R: usize>(
    dec: &Decoded,
    act: &Acts,
    res: &mut [f32],
    rows: usize,
    r: usize,
) {
    let m = act.m;
    let mut j0 = 0;
    while m - j0 >= MR {
        put(tile::<MIN, SUB, R, MR>(dec, act, j0), res, rows, r, j0);
        j0 += MR;
    }
    match m - j0 {
        0 => {}
        1 => put(tile::<MIN, SUB, R, 1>(dec, act, j0), res, rows, r, j0),
        2 => put(tile::<MIN, SUB, R, 2>(dec, act, j0), res, rows, r, j0),
        _ => put(tile::<MIN, SUB, R, 3>(dec, act, j0), res, rows, r, j0),
    }
}

/// [`tiles`] of the shape a format has (`min`: `MIN_*`, `sub`: [`Format::SUB`]).
#[inline(always)]
fn shaped_tiles<const R: usize>(
    (min, sub): (u8, usize),
    dec: &Decoded,
    act: &Acts,
    res: &mut [f32],
    rows: usize,
    r: usize,
) {
    match (min, sub) {
        (MIN_Q8_1, _) => tiles::<MIN_Q8_1, 1, R>(dec, act, res, rows, r),
        (MIN_HALVES, _) => tiles::<MIN_HALVES, 1, R>(dec, act, res, rows, r),
        (_, 2) => tiles::<MIN_NONE, 2, R>(dec, act, res, rows, r),
        _ => tiles::<MIN_NONE, 1, R>(dec, act, res, rows, r),
    }
}

/// Bytes of a weight row of `k` values.
fn row_bytes<F: Format>(k: usize) -> usize {
    k / F::VALUES * size_of::<F::Block>()
}

/// Weight rows `w` (a whole number of rows) against every activation row;
/// `res[j * rows + r]` receives the dot of weight row `r` with activation row
/// `j`. A group of [`NR`] rows is decoded once and dotted against all the
/// activation tiles.
fn chunk<F: Format>(w: &[u8], act: &Acts, dec: &mut Decoded, res: &mut [f32]) {
    let row_bytes = row_bytes::<F>(act.nb * QK);
    let rows = w.len() / row_bytes;
    let row = |r: usize| F::blocks(&w[r * row_bytes..(r + 1) * row_bytes]);
    let shape = (F::MIN, F::SUB);
    let mut r = 0;
    while r + NR <= rows {
        let rs: [&[F::Block]; NR] = std::array::from_fn(|i| row(r + i));
        dec.load::<F>(&rs);
        shaped_tiles::<NR>(shape, dec, act, res, rows, r);
        r += NR;
    }
    while r < rows {
        dec.load::<F>(&[row(r)]);
        shaped_tiles::<1>(shape, dec, act, res, rows, r);
        r += 1;
    }
}

/// The weight formats the kernel serves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
    Iq2Xxs,
    Iq2Xs,
    Iq2S,
    Iq3Xxs,
    Iq3S,
    Iq1S,
    Iq1M,
    Iq4Nl,
    Iq4Xs,
    Tq1_0,
    Tq2_0,
    Mxfp4,
    Nvfp4,
}

impl Kind {
    /// The format of a GGUF type, `None` for a float or unknown type.
    fn of(ty: GgufType) -> Option<Self> {
        use candle_core::quantized::GgmlDType as G;
        Some(match ty {
            GgufType::Ggml(G::Q4_0) => Self::Q4_0,
            GgufType::Ggml(G::Q4_1) => Self::Q4_1,
            GgufType::Ggml(G::Q5_0) => Self::Q5_0,
            GgufType::Ggml(G::Q5_1) => Self::Q5_1,
            GgufType::Ggml(G::Q8_0) => Self::Q8_0,
            GgufType::Ggml(G::Q2K) => Self::Q2K,
            GgufType::Ggml(G::Q3K) => Self::Q3K,
            GgufType::Ggml(G::Q4K) => Self::Q4K,
            GgufType::Ggml(G::Q5K) => Self::Q5K,
            GgufType::Ggml(G::Q6K) => Self::Q6K,
            GgufType::Ggml(G::Q8K) => Self::Q8K,
            GgufType::Iq2Xxs => Self::Iq2Xxs,
            GgufType::Iq2Xs => Self::Iq2Xs,
            GgufType::Iq2S => Self::Iq2S,
            GgufType::Iq3Xxs => Self::Iq3Xxs,
            GgufType::Iq3S => Self::Iq3S,
            GgufType::Iq1S => Self::Iq1S,
            GgufType::Iq1M => Self::Iq1M,
            GgufType::Iq4Nl => Self::Iq4Nl,
            GgufType::Iq4Xs => Self::Iq4Xs,
            GgufType::Tq1_0 => Self::Tq1_0,
            GgufType::Tq2_0 => Self::Tq2_0,
            GgufType::Mxfp4 => Self::Mxfp4,
            GgufType::Nvfp4 => Self::Nvfp4,
            _ => return None,
        })
    }
}

/// Run `$body` with the type alias `$f` naming the [`Format`] of `$kind`.
macro_rules! with_format {
    ($kind:expr, $f:ident => $body:expr) => {
        match $kind {
            Kind::Q4_0 => {
                type $f = Q4_0;
                $body
            }
            Kind::Q4_1 => {
                type $f = Q4_1;
                $body
            }
            Kind::Q5_0 => {
                type $f = Q5_0;
                $body
            }
            Kind::Q5_1 => {
                type $f = Q5_1;
                $body
            }
            Kind::Q8_0 => {
                type $f = Q8_0;
                $body
            }
            Kind::Q2K => {
                type $f = Q2K;
                $body
            }
            Kind::Q3K => {
                type $f = Q3K;
                $body
            }
            Kind::Q4K => {
                type $f = Q4K;
                $body
            }
            Kind::Q5K => {
                type $f = Q5K;
                $body
            }
            Kind::Q6K => {
                type $f = Q6K;
                $body
            }
            Kind::Q8K => {
                type $f = Q8K;
                $body
            }
            Kind::Iq2Xxs => {
                type $f = Iq2Xxs;
                $body
            }
            Kind::Iq2Xs => {
                type $f = Iq2Xs;
                $body
            }
            Kind::Iq2S => {
                type $f = Iq2S;
                $body
            }
            Kind::Iq3Xxs => {
                type $f = Iq3Xxs;
                $body
            }
            Kind::Iq3S => {
                type $f = Iq3S;
                $body
            }
            Kind::Iq1S => {
                type $f = Iq1S;
                $body
            }
            Kind::Iq1M => {
                type $f = Iq1M;
                $body
            }
            Kind::Iq4Nl => {
                type $f = Iq4Nl;
                $body
            }
            Kind::Iq4Xs => {
                type $f = Iq4Xs;
                $body
            }
            Kind::Tq1_0 => {
                type $f = Tq1_0;
                $body
            }
            Kind::Tq2_0 => {
                type $f = Tq2_0;
                $body
            }
            Kind::Mxfp4 => {
                type $f = Mxfp4;
                $body
            }
            Kind::Nvfp4 => {
                type $f = Nvfp4;
                $body
            }
        }
    };
}

/// A quantized `[n, k]` weight (`y = x * w^T`) computed by the row-batched
/// kernel, over the blocks as the GGUF file stores them (`k` a whole number
/// of blocks per row). Reads the blocks in place.
pub(crate) struct QmmWeight {
    data: Arc<[u8]>,
    kind: Kind,
    n: usize,
    k: usize,
}

impl QmmWeight {
    /// The `[n, k]` weight of GGUF type `dtype` over its file bytes, or `None`
    /// for a type the kernel does not serve (the float types), a row that is
    /// not whole blocks, or a wrong byte count.
    pub(crate) fn from_raw(
        dtype: impl Into<GgufType>,
        data: Arc<[u8]>,
        n: usize,
        k: usize,
    ) -> Option<Self> {
        let dtype = dtype.into();
        let kind = Kind::of(dtype)?;
        let row = dtype.byte_len(k)?;
        (n > 0 && row.checked_mul(n) == Some(data.len())).then_some(Self { data, kind, n, k })
    }

    /// Output width (weight rows).
    pub(crate) fn n(&self) -> usize {
        self.n
    }

    /// Input width (weight columns).
    pub(crate) fn k(&self) -> usize {
        self.k
    }

    /// `x` is `m` rows of `k` floats; returns `m` rows of `n` floats.
    ///
    /// Row `j` of the result depends on row `j` of `x` only, bit for bit,
    /// whatever `m` is. The engine goes through [`QmmWeight::forward_rows`].
    #[cfg(test)]
    pub(crate) fn forward(&self, x: &[f32], m: usize) -> candle_core::Result<Vec<f32>> {
        self.forward_rows(x, m, self.n)
    }

    /// The product of `x` with the first `rows` weight rows only
    /// (`rows <= n`): `m` rows of `rows` floats, each equal to the first
    /// `rows` outputs of the full product, bit for bit.
    pub(crate) fn forward_rows(
        &self,
        x: &[f32],
        m: usize,
        rows: usize,
    ) -> candle_core::Result<Vec<f32>> {
        if x.len() != m * self.k {
            candle_core::bail!(
                "qmm: activations hold {} floats, expected {m} rows of {}",
                x.len(),
                self.k
            );
        }
        if rows == 0 || rows > self.n {
            candle_core::bail!("qmm: {rows} weight rows of {}", self.n);
        }
        let mut out = vec![0f32; m * rows];
        if m == 0 {
            return Ok(out);
        }
        let row_bytes = self.data.len() / self.n;
        let weights = self.data.get(..rows * row_bytes).unwrap_or(&self.data[..]);
        let (k, o) = (self.k, &mut out);
        with_format!(self.kind, F => run::<F>(weights, x, m, k, rows, o))?;
        Ok(out)
    }
}

/// Quantise the `m` activation rows for the format `F`, split the `[m, n]`
/// output into `rows_per_chunk`-wide column chunks per activation row and
/// compute them over the pool; each worker claims chunks dynamically,
/// computes one into a private scratch (with its own decode buffers) and
/// copies it out.
fn run<F: Format>(
    data: &[u8],
    x: &[f32],
    m: usize,
    k: usize,
    n: usize,
    out: &mut [f32],
) -> candle_core::Result<()> {
    let act = Acts::new(x, m, k, F::MIN);
    let act = &act;
    let row_bytes = row_bytes::<F>(k);
    if data.len() != n * row_bytes {
        candle_core::bail!(
            "qmm: weight holds {} bytes, expected {n} rows of {row_bytes}",
            data.len()
        );
    }
    let threads = barrier_pool().n_workers() + 1;
    let serial = n * k < SERIAL_BELOW || threads == 1;
    let rows_per_chunk = if serial {
        n
    } else {
        (n / (threads * 4)).div_ceil(NR).max(1) * NR
    };
    let n_chunks = n.div_ceil(rows_per_chunk);
    let dst: Vec<Mutex<&mut [f32]>> = out
        .chunks_mut(n)
        .flat_map(|row| row.chunks_mut(rows_per_chunk))
        .map(Mutex::new)
        .collect();

    for_each(
        n_chunks,
        !serial,
        || (vec![0f32; m * rows_per_chunk], Decoded::new::<F>(act.nb)),
        |(res, dec), c| {
            let r0 = c * rows_per_chunk;
            let rows = rows_per_chunk.min(n - r0);
            chunk::<F>(
                &data[r0 * row_bytes..(r0 + rows) * row_bytes],
                act,
                dec,
                &mut res[..m * rows],
            );
            for (j, row) in dst.chunks_exact(n_chunks).enumerate() {
                row[c]
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .copy_from_slice(&res[j * rows..(j + 1) * rows]);
            }
        },
    );
    Ok(())
}

#[cfg(test)]
mod tests;
