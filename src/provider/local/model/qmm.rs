//! Row-batched quantized matmul for Q4_0, Q4_1, Q5_0, Q5_1, Q8_0 and MXFP4
//! weights.
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
//! The activations are quantised the way ggml pairs them with the weight
//! dtype, exactly as candle's vec-dot partners do:
//!
//! - Q4_0, Q5_0, Q8_0, MXFP4: Q8_0 (`d = max|x| / 127` rounded to f16,
//!   `q = round(x * (1 / d))`, 32-element blocks);
//! - Q4_1, Q5_1: Q8_1, the same quants plus the sum term
//!   `s = f16(f16(sum q) * d)`, which multiplies the weight block's minimum.
//!
//! The products are the integer dot of one weight block with one activation
//! block, scaled by the product of the two block scales.
//!
//! The value for activation row `j` and weight row `r` is a fixed function of
//! that row pair alone: sixteen `f32` lanes, each adding its share of the
//! block's exact integer dot times the block scale in block order (how the
//! integer part is computed is free), reduced by a fixed pairwise tree, plus
//! for Q4_1 and Q5_1 a scalar `min * s` per block added after the reduction.
//! No cross-row term exists and tile shape never changes the order, so a row's
//! result is bit-identical whether it is computed alone or in a batch of any
//! size, which speculative-decoding verification needs. It differs from
//! candle's own result only in float rounding order.
//!
//! Work is split in chunks of weight rows over candle's persistent barrier pool
//! (the one its own quantized matmul uses, so the two never oversubscribe the
//! cores), with dynamic chunk claiming; which worker computes a chunk never
//! changes its bits.
//!
//! # K-quants stay on candle
//!
//! Q2_K to Q6_K weights are not served here: [`QmmWeight::new`] returns `None`
//! for them and candle's per-row kernel multiplies them (a row's result still
//! depends on that row alone, whatever the batch). A row's bits must not depend
//! on the batch size, so the dispatch cannot pick a faster kernel per `m`: one
//! dtype is served by exactly one kernel.

use std::sync::{Arc, Mutex, PoisonError};

use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::utils::barrier_pool;
use half::f16;
use wide::{f32x4, f32x16, i8x16, i8x32, i16x16, i16x32, u8x32};

use super::pool::for_each;

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

/// A 32-element weight block: its bytes, f32 scale, optional minimum and quants.
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

/// The doubled E2M1 values an MXFP4 nibble selects (ggml's `kvalues_mxfp4`).
const KVALUES_MXFP4: [i16; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

/// [`KVALUES_MXFP4`] as a byte table, once in each 128-bit half of a vector
/// (a byte shuffle looks up within its half).
const MXFP4_TABLE: i8x32 = {
    let mut t = [0i8; QK];
    let mut i = 0;
    while i < QK {
        t[i] = KVALUES_MXFP4[i % 16] as i8;
        i += 1;
    }
    i8x32::new(t)
};

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
            w[0][i] = KVALUES_MXFP4[usize::from(b & 0x0F)];
            w[1][i] = KVALUES_MXFP4[usize::from(b >> 4)];
        }
        w
    }

    #[inline(always)]
    fn decode(block: &Self::Block) -> Quants {
        Quants::from_i8x32(MXFP4_TABLE.swizzle_half_relaxed(nibble_pairs(&block[1..17])))
    }
}

/// Activations of one call, quantised to Q8_0 (or Q8_1, which adds the sum
/// term) and laid out block-major so the blocks of consecutive rows at one
/// block index are contiguous.
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
    /// for plain Q8_0.
    sums: Vec<f32>,
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

/// One activation block quantised: its quants in [`Quants`]' layout, its
/// f16-rounded scale and, when `with_sums`, the Q8_1 sum term.
#[inline(always)]
fn quantise_block(src: &[f32; QK], with_sums: bool) -> (Quants, f32, f32) {
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
    let sum = if with_sums {
        let sum: i32 = lanes.iter().map(|&q| i32::from(q)).sum();
        (f16::from_f32(sum as f32) * d16).to_f32()
    } else {
        0.0
    };
    (Quants::new(lanes), d16.to_f32(), sum)
}

/// One job of [`Acts::new`]: a run of blocks' quants, scales and sum terms.
type Part<'a> = (&'a mut [Quants], &'a mut [f32], &'a mut [f32]);

impl Acts {
    /// Quantise `x` (`m` rows of `k` floats, `k` a multiple of [`QK`]) the way
    /// candle's `BlockQ8_0::from_float` does, or `BlockQ8_1::from_float` when
    /// `with_sums`.
    ///
    /// A large call splits the blocks over the pool: each block is quantised
    /// from its own 32 values alone, so the split changes no bit.
    fn new(x: &[f32], m: usize, k: usize, with_sums: bool) -> Self {
        let nb = k / QK;
        let mut qs = vec![Quants::default(); nb * m];
        let mut scales = vec![0f32; nb * m];
        let mut sums = vec![0f32; if with_sums { nb * m } else { 0 }];
        let pooled = m * k >= SERIAL_BELOW && barrier_pool().n_workers() > 0;
        let threads = barrier_pool().n_workers() + 1;
        let per = if pooled {
            nb.div_ceil(threads * 4).max(1)
        } else {
            nb.max(1)
        };
        let width = (per * m).max(1);
        let mut sum_chunks = sums.chunks_mut(width);
        let jobs: Vec<Mutex<Part<'_>>> = qs
            .chunks_mut(width)
            .zip(scales.chunks_mut(width))
            .map(|(q, s)| {
                let su: &mut [f32] = if with_sums {
                    sum_chunks.next().unwrap_or_default()
                } else {
                    &mut []
                };
                Mutex::new((q, s, su))
            })
            .collect();
        for_each(
            jobs.len(),
            pooled && jobs.len() > 1,
            || (),
            |(), i| {
                let mut part = jobs[i].lock().unwrap_or_else(PoisonError::into_inner);
                let (q, s, su) = &mut *part;
                for bi in 0..q.len() / m {
                    let b = i * per + bi;
                    for j in 0..m {
                        let at = j * k + b * QK;
                        let Ok(src) = <&[f32; QK]>::try_from(&x[at..at + QK]) else {
                            continue;
                        };
                        let (quants, scale, sum) = quantise_block(src, with_sums);
                        q[bi * m + j] = quants;
                        s[bi * m + j] = scale;
                        if with_sums {
                            su[bi * m + j] = sum;
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

/// Up to [`NR`] weight rows with every block decoded: the quants, f32 scales
/// and minima of rows by `nb` blocks, kept apart from the weights so that one
/// decode serves every activation tile.
struct Decoded {
    nb: usize,
    /// `[row][block]`.
    quants: Vec<Quants>,
    scales: Vec<f32>,
    /// Empty for a format without minima.
    mins: Vec<f32>,
}

impl Decoded {
    fn new<L: Legacy>(nb: usize) -> Self {
        Self {
            nb,
            quants: vec![Quants::default(); NR * nb],
            scales: vec![0f32; NR * nb],
            mins: vec![0f32; if L::HAS_MIN { NR * nb } else { 0 }],
        }
    }

    /// Decode `rows` (at most [`NR`], each `nb` blocks) into rows `0..`.
    fn load<L: Legacy>(&mut self, rows: &[&[L::Block]]) {
        let nb = self.nb;
        for (r, row) in rows.iter().enumerate() {
            for (b, block) in row.iter().enumerate() {
                self.quants[r * nb + b] = L::decode(block);
                self.scales[r * nb + b] = L::scale(block);
                if L::HAS_MIN {
                    self.mins[r * nb + b] = L::min(block);
                }
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
/// jj`, defined as sixteen `f32` lanes that each add, in block order, the
/// exact integer sum of the block's two elements of that lane times the
/// product of the two block scales, reduced by [`hsum`]. The accumulators stay
/// in registers across the blocks.
///
/// A format with a block minimum (`L::HAS_MIN`) adds `min * s` per block to a
/// scalar sum in block order, where `s` is the activation block's Q8_1 sum
/// term, after the lanes are reduced.
#[inline(always)]
fn tile<L: Legacy, const R: usize, const M: usize>(
    dec: &Decoded,
    act: &Acts,
    j0: usize,
) -> [[f32; M]; R] {
    let nb = act.nb;
    let mut acc = [[f32x16::ZERO; M]; R];
    let mut extra = [[0f32; M]; R];
    let rows: [&[Quants]; R] = std::array::from_fn(|r| &dec.quants[r * nb..(r + 1) * nb]);
    let scales: [&[f32]; R] = std::array::from_fn(|r| &dec.scales[r * nb..(r + 1) * nb]);
    for b in 0..nb {
        let at = b * act.m + j0;
        let xs = &act.qs[at..at + M];
        let sc = &act.scales[at..at + M];
        for r in 0..R {
            let w = rows[r][b];
            let s = scale_products::<M>(scales[r][b], sc);
            for jj in 0..M {
                let p = f32x16::from_i32x16(w.dot(xs[jj]));
                acc[r][jj] += p * f32x16::splat(s[jj]);
            }
        }
        if L::HAS_MIN {
            let sums = &act.sums[at..at + M];
            for (r, row) in extra.iter_mut().enumerate() {
                let mn = dec.mins[r * nb + b];
                for (e, &s) in row.iter_mut().zip(sums) {
                    *e += mn * s;
                }
            }
        }
    }
    let mut out = [[0f32; M]; R];
    for r in 0..R {
        for jj in 0..M {
            let lanes = acc[r][jj].to_array();
            out[r][jj] = if L::HAS_MIN {
                hsum(&lanes) + extra[r][jj]
            } else {
                hsum(&lanes)
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
/// tiles, then one tile for the remaining 1..=3 rows.
#[inline(always)]
fn tiles<L: Legacy, const R: usize>(
    dec: &Decoded,
    act: &Acts,
    res: &mut [f32],
    rows: usize,
    r: usize,
) {
    let m = act.m;
    let mut j0 = 0;
    while m - j0 >= MR {
        put(tile::<L, R, MR>(dec, act, j0), res, rows, r, j0);
        j0 += MR;
    }
    match m - j0 {
        0 => {}
        1 => put(tile::<L, R, 1>(dec, act, j0), res, rows, r, j0),
        2 => put(tile::<L, R, 2>(dec, act, j0), res, rows, r, j0),
        _ => put(tile::<L, R, 3>(dec, act, j0), res, rows, r, j0),
    }
}

/// Weight rows `w` (a whole number of rows) against every activation row;
/// `res[j * rows + r]` receives the dot of weight row `r` with activation row
/// `j`. A group of [`NR`] rows is decoded once and dotted against all the
/// activation tiles.
fn chunk<L: Legacy>(w: &[u8], act: &Acts, dec: &mut Decoded, res: &mut [f32]) {
    let row_bytes = act.nb * size_of::<L::Block>();
    let rows = w.len() / row_bytes;
    let row = |r: usize| L::blocks(&w[r * row_bytes..(r + 1) * row_bytes]);
    let mut r = 0;
    while r + NR <= rows {
        let rs: [&[L::Block]; NR] = std::array::from_fn(|i| row(r + i));
        dec.load::<L>(&rs);
        tiles::<L, NR>(dec, act, res, rows, r);
        r += NR;
    }
    while r < rows {
        dec.load::<L>(&[row(r)]);
        tiles::<L, 1>(dec, act, res, rows, r);
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
    Mxfp4,
}

impl Kind {
    fn of(dtype: GgmlDType) -> Option<Self> {
        Some(match dtype {
            GgmlDType::Q4_0 => Self::Q4_0,
            GgmlDType::Q4_1 => Self::Q4_1,
            GgmlDType::Q5_0 => Self::Q5_0,
            GgmlDType::Q5_1 => Self::Q5_1,
            GgmlDType::Q8_0 => Self::Q8_0,
            _ => return None,
        })
    }
}

/// Where a weight's blocks live: candle's tensor, or bytes candle has no type
/// for (MXFP4).
enum Blocks {
    Tensor(Arc<QTensor>),
    Raw(Arc<[u8]>),
}

/// A quantized `[n, k]` weight (`y = x * w^T`) computed by the row-batched
/// kernel: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0 or MXFP4. Reads the blocks in place.
pub(crate) struct QmmWeight {
    blocks: Blocks,
    kind: Kind,
    n: usize,
    k: usize,
}

impl QmmWeight {
    /// The kernel's view of `qt`, or `None` when it is not a 2-D tensor of a
    /// served dtype with rows a whole number of blocks long (candle's path
    /// serves it).
    pub(crate) fn new(qt: &Arc<QTensor>) -> Option<Self> {
        let kind = Kind::of(qt.dtype())?;
        let [n, k] = *qt.shape().dims() else {
            return None;
        };
        (k % QK == 0 && n > 0).then(|| Self {
            blocks: Blocks::Tensor(Arc::clone(qt)),
            kind,
            n,
            k,
        })
    }

    /// An MXFP4 `[n, k]` weight over its file bytes (`k / 32` blocks of 17
    /// bytes per row); `None` for a row that is not whole blocks or a wrong
    /// byte count.
    pub(crate) fn mxfp4(bytes: Arc<[u8]>, n: usize, k: usize) -> Option<Self> {
        (k.is_multiple_of(QK) && n > 0 && bytes.len() == n * (k / QK) * size_of::<[u8; 17]>()).then(
            || Self {
                blocks: Blocks::Raw(bytes),
                kind: Kind::Mxfp4,
                n,
                k,
            },
        )
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
        let data = match &self.blocks {
            Blocks::Tensor(qt) => qt.data()?,
            Blocks::Raw(bytes) => std::borrow::Cow::Borrowed(&bytes[..]),
        };
        let row_bytes = data.len() / self.n;
        let weights = data.get(..rows * row_bytes).unwrap_or(&data[..]);
        let (k, o) = (self.k, &mut out);
        match self.kind {
            Kind::Q4_0 => run::<Q4_0>(weights, x, m, k, rows, o)?,
            Kind::Q4_1 => run::<Q4_1>(weights, x, m, k, rows, o)?,
            Kind::Q5_0 => run::<Q5_0>(weights, x, m, k, rows, o)?,
            Kind::Q5_1 => run::<Q5_1>(weights, x, m, k, rows, o)?,
            Kind::Q8_0 => run::<Q8_0>(weights, x, m, k, rows, o)?,
            Kind::Mxfp4 => run::<Mxfp4>(weights, x, m, k, rows, o)?,
        }
        Ok(out)
    }
}

/// Quantise the `m` activation rows for the format `L`, split the `[m, n]`
/// output into `rows_per_chunk`-wide column chunks per activation row and
/// compute them over the pool; each worker claims chunks dynamically,
/// computes one into a private scratch (with its own decode buffers) and
/// copies it out.
fn run<L: Legacy>(
    data: &[u8],
    x: &[f32],
    m: usize,
    k: usize,
    n: usize,
    out: &mut [f32],
) -> candle_core::Result<()> {
    let act = Acts::new(x, m, k, L::HAS_MIN);
    let act = &act;
    let row_bytes = act.nb * size_of::<L::Block>();
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
        || (vec![0f32; m * rows_per_chunk], Decoded::new::<L>(act.nb)),
        |(res, dec), c| {
            let r0 = c * rows_per_chunk;
            let rows = rows_per_chunk.min(n - r0);
            chunk::<L>(
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
mod tests {
    use super::*;
    use candle_core::quantized::QMatMul;
    use candle_core::{Device, Module, Tensor};

    /// Every dtype the kernel serves.
    const DTYPES: [GgmlDType; 5] = [
        GgmlDType::Q4_0,
        GgmlDType::Q4_1,
        GgmlDType::Q5_0,
        GgmlDType::Q5_1,
        GgmlDType::Q8_0,
    ];

    /// Deterministic pseudo-random floats in `[-1, 1)` scaled by `scale`.
    pub(super) fn noise(len: usize, seed: u64, scale: f32) -> Vec<f32> {
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..len)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 40) as f32 / (1u64 << 23) as f32 - 1.0) * scale
            })
            .collect()
    }

    pub(super) fn weight(n: usize, k: usize, dtype: GgmlDType, seed: u64) -> Arc<QTensor> {
        let t = Tensor::from_vec(noise(n * k, seed, 0.5), (n, k), &Device::Cpu).unwrap();
        Arc::new(QTensor::quantize(&t, dtype).unwrap())
    }

    /// `(n, k, m)` shapes whose rows are whole blocks: tails in both tilings,
    /// and 600 rows exceeding `SERIAL_BELOW` so the pooled path runs.
    fn shapes() -> [(usize, usize, usize); 4] {
        [(10, 2 * QK, 7), (3, QK, 1), (600, 256, 5), (64, 3 * QK, 17)]
    }

    /// Element `e` of block `b` of activation row `j`, from the kernel's
    /// [`Quants`] layout.
    fn act_quant(act: &Acts, b: usize, j: usize, e: usize) -> i16 {
        let lane = if e < LANES {
            2 * e
        } else {
            2 * (e - LANES) + 1
        };
        act.qs[b * act.m + j].to_array()[lane]
    }

    /// The definition in the module docs, scalar and in the straightforward
    /// order: the product of weight row `r` and activation row `j` is sixteen
    /// `f32` lanes, each adding its two elements' exact integer sum times the
    /// product of the block scales block by block, reduced by [`hsum`].
    fn scalar_definition<L: Legacy>(data: &[u8], act: &Acts, n: usize) -> Vec<f32> {
        let row_bytes = act.nb * size_of::<L::Block>();
        let mut out = vec![0f32; act.m * n];
        for j in 0..act.m {
            for r in 0..n {
                let mut lanes = [0f32; LANES];
                let mut extra = 0f32;
                for (b, block) in L::blocks(&data[r * row_bytes..(r + 1) * row_bytes])
                    .iter()
                    .enumerate()
                {
                    let w = L::halves(block);
                    let s = L::scale(block) * act.scales[b * act.m + j];
                    for (l, lane) in lanes.iter_mut().enumerate() {
                        let p = i32::from(w[0][l]) * i32::from(act_quant(act, b, j, l))
                            + i32::from(w[1][l]) * i32::from(act_quant(act, b, j, l + LANES));
                        *lane += p as f32 * s;
                    }
                    if L::HAS_MIN {
                        extra += L::min(block) * act.sums[b * act.m + j];
                    }
                }
                out[j * n + r] = if L::HAS_MIN {
                    hsum(&lanes) + extra
                } else {
                    hsum(&lanes)
                };
            }
        }
        out
    }

    /// [`scalar_definition`] for the format `q` holds.
    fn scalar_forward(q: &QmmWeight, x: &[f32], m: usize) -> Vec<f32> {
        let data = match &q.blocks {
            Blocks::Tensor(qt) => qt.data().unwrap().into_owned(),
            Blocks::Raw(bytes) => bytes.to_vec(),
        };
        let act = |has_min| Acts::new(x, m, q.k, has_min);
        match q.kind {
            Kind::Q4_0 => scalar_definition::<Q4_0>(&data, &act(false), q.n),
            Kind::Q4_1 => scalar_definition::<Q4_1>(&data, &act(true), q.n),
            Kind::Q5_0 => scalar_definition::<Q5_0>(&data, &act(false), q.n),
            Kind::Q5_1 => scalar_definition::<Q5_1>(&data, &act(true), q.n),
            Kind::Q8_0 => scalar_definition::<Q8_0>(&data, &act(false), q.n),
            Kind::Mxfp4 => scalar_definition::<Mxfp4>(&data, &act(false), q.n),
        }
    }

    #[test]
    fn the_kernel_equals_the_scalar_definition_bit_for_bit() {
        for dtype in DTYPES {
            for (n, k, m) in [
                (10, 2 * QK, 7),
                (3, 3 * QK, 1),
                (130, 512, 41),
                (64, 3 * QK, 17),
                (37, 4 * QK, 6),
                (9, 8 * QK, 2),
            ] {
                let w = weight(n, k, dtype, 11);
                let x = awkward_acts(m, k);
                let q = QmmWeight::new(&w).unwrap();
                let got = q.forward(&x, m).unwrap();
                let want = scalar_forward(&q, &x, m);
                for (i, (g, r)) in got.iter().zip(&want).enumerate() {
                    assert_eq!(
                        g.to_bits(),
                        r.to_bits(),
                        "{dtype:?} {n}x{k} m={m} output {i}: {g} vs {r}"
                    );
                }
            }
        }
    }

    /// Every format's vector decode gives the lanes of its scalar definition,
    /// on random blocks.
    #[test]
    fn decode_equals_halves() {
        fn check<L: Legacy>(name: &str) {
            let bytes: Vec<u8> = noise(4096 * size_of::<L::Block>(), 7, 127.0)
                .iter()
                .map(|&v| (v * 2.0) as i32 as u8)
                .collect();
            for (i, block) in L::blocks(&bytes).iter().enumerate() {
                assert_eq!(
                    L::decode(block).to_array(),
                    interleave(&L::halves(block)).to_array(),
                    "{name} block {i}"
                );
            }
        }
        check::<Q4_0>("Q4_0");
        check::<Q4_1>("Q4_1");
        check::<Q5_0>("Q5_0");
        check::<Q5_1>("Q5_1");
        check::<Q8_0>("Q8_0");
        check::<Mxfp4>("Mxfp4");
    }

    /// The activations as the kernel sees them: quantised and dequantised.
    fn rounded_acts(x: &[f32], m: usize, k: usize) -> Vec<f64> {
        let mut xd = vec![0f64; m * k];
        let act = Acts::new(x, m, k, false);
        for j in 0..m {
            for b in 0..act.nb {
                for e in 0..QK {
                    xd[j * k + b * QK + e] =
                        f64::from(act_quant(&act, b, j, e)) * f64::from(act.scales[b * act.m + j]);
                }
            }
        }
        xd
    }

    /// The dequantised weights, row-major.
    pub(super) fn dequantised(w: &QTensor) -> Vec<f32> {
        w.dequantize(&Device::Cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
    }

    /// Reference in f64: dequantised weights times dequantised (rounded)
    /// activations.
    fn reference(w: &QTensor, x: &[f32], m: usize) -> Vec<f64> {
        let [n, k] = *w.shape().dims() else {
            unreachable!()
        };
        let wd = dequantised(w);
        let xd = rounded_acts(x, m, k);
        let mut out = vec![0f64; m * n];
        for j in 0..m {
            for r in 0..n {
                out[j * n + r] = (0..k)
                    .map(|e| f64::from(wd[r * k + e]) * xd[j * k + e])
                    .sum();
            }
        }
        out
    }

    /// Per output, the most the Q8_1 sum term's f16 rounding can move a Q4_1 or
    /// Q5_1 result: `|min| * |s|` per block, at 2^-10 (two f16 roundings of the
    /// sum term `s`). Zero for every other dtype, whose arithmetic is exact
    /// integer.
    fn sum_term_slack(w: &QTensor, x: &[f32], m: usize) -> Vec<f64> {
        let [n, k] = *w.shape().dims() else {
            unreachable!()
        };
        let mut slack = vec![0f64; m * n];
        if !matches!(w.dtype(), GgmlDType::Q4_1 | GgmlDType::Q5_1) {
            return slack;
        }
        let act = Acts::new(x, m, k, true);
        let data = w.data().unwrap();
        let block = w.dtype().type_size();
        for j in 0..m {
            for r in 0..n {
                slack[j * n + r] = (0..act.nb)
                    .map(|b| {
                        let at = (r * act.nb + b) * block;
                        let min = f64::from(f16_le(&data[at..at + block], 2));
                        min.abs() * f64::from(act.sums[b * act.m + j]).abs() * 1.0e-3
                    })
                    .sum();
            }
        }
        slack
    }

    #[test]
    fn matches_the_f64_reference_and_candle() {
        for dtype in DTYPES {
            for (n, k, m) in shapes() {
                let w = weight(n, k, dtype, 11);
                let x = noise(m * k, 5, 1.0);
                let q = QmmWeight::new(&w).unwrap();
                let got = q.forward(&x, m).unwrap();
                let want = reference(&w, &x, m);
                let slack = sum_term_slack(&w, &x, m);
                let scale = want.iter().fold(0f64, |a, v| a.max(v.abs()));
                for ((g, r), s) in got.iter().zip(&want).zip(&slack) {
                    assert!(
                        (f64::from(*g) - r).abs() <= 1e-4 * scale + s,
                        "{dtype:?} {n}x{k} m={m}: {g} vs {r}"
                    );
                }
                // candle's Q4_1, Q5_0 and Q5_1 dots assert an even block count.
                if (k / QK) % 2 == 1
                    && matches!(dtype, GgmlDType::Q4_1 | GgmlDType::Q5_0 | GgmlDType::Q5_1)
                {
                    continue;
                }
                let xt = Tensor::from_vec(x, (m, k), &Device::Cpu).unwrap();
                let candle = QMatMul::from_arc(Arc::clone(&w))
                    .unwrap()
                    .forward(&xt)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap();
                for (g, c) in got.iter().zip(&candle) {
                    assert!(
                        (f64::from(*g) - f64::from(*c)).abs() <= 1e-4 * scale,
                        "{dtype:?} {n}x{k} m={m}: {g} vs candle {c}"
                    );
                }
            }
        }
    }

    /// Against the dequantised weights and the true (unrounded) activations the
    /// error is the activation rounding alone: at most half a quant step
    /// (plus the f16 rounding of the scale, 2^-11 of 127 steps) per element,
    /// `0.6 * d_block * sum |w|` per block.
    #[test]
    fn the_error_is_bounded_by_the_activation_rounding() {
        for dtype in DTYPES {
            let b = QK;
            for (n, k, m) in [(12, 2 * b, 5), (20, 512, 3)] {
                let w = weight(n, k, dtype, 23);
                let mut x = noise(m * k, 8, 1.0);
                // An outlier per row makes the quant step coarse.
                for j in 0..m {
                    x[j * k + 7] = 6.0 * (j as f32 + 1.0);
                }
                let got = QmmWeight::new(&w).unwrap().forward(&x, m).unwrap();
                let wd = dequantised(&w);
                let slack = sum_term_slack(&w, &x, m);
                for j in 0..m {
                    for r in 0..n {
                        let mut exact = 0f64;
                        let mut bound = slack[j * n + r];
                        for blk in 0..k / b {
                            let xs = &x[j * k + blk * b..j * k + (blk + 1) * b];
                            let ws = &wd[r * k + blk * b..r * k + (blk + 1) * b];
                            let amax = xs.iter().fold(0f32, |a, v| a.max(v.abs()));
                            let abs_w: f64 = ws.iter().map(|v| f64::from(v.abs())).sum();
                            bound += 0.6 * f64::from(amax) / 127.0 * abs_w;
                            exact += xs
                                .iter()
                                .zip(ws)
                                .map(|(&a, &c)| f64::from(a) * f64::from(c))
                                .sum::<f64>();
                        }
                        let g = f64::from(got[j * n + r]);
                        assert!(
                            (g - exact).abs() <= bound + 1e-5,
                            "{dtype:?} {n}x{k} row {j} col {r}: {g} vs {exact} (bound {bound})"
                        );
                    }
                }
            }
        }
    }

    /// A prefix of the weight rows computes exactly the first outputs of the
    /// whole product, whatever the tiling.
    #[test]
    fn a_weight_row_prefix_equals_the_start_of_the_product() {
        for dtype in DTYPES {
            let b = QK;
            for (n, k, m, rows) in [(10, 2 * b, 3, 7), (600, 256, 2, 450), (64, 3 * b, 5, 1)] {
                let w = weight(n, k, dtype, 17);
                let x = noise(m * k, 9, 1.0);
                let q = QmmWeight::new(&w).unwrap();
                let whole = q.forward(&x, m).unwrap();
                let part = q.forward_rows(&x, m, rows).unwrap();
                assert_eq!(part.len(), m * rows);
                for j in 0..m {
                    assert_eq!(
                        part[j * rows..(j + 1) * rows],
                        whole[j * n..j * n + rows],
                        "{dtype:?} {n}x{k} m={m} rows={rows} row {j}"
                    );
                }
                assert!(q.forward_rows(&x, m, 0).is_err());
                assert!(q.forward_rows(&x, m, n + 1).is_err());
            }
        }
    }

    #[test]
    fn round_away_equals_f32_round() {
        let mut values = noise(4096, 2, 300.0);
        values.extend((-260..260).map(|i| i as f32 * 0.5));
        values.extend([
            0.0,
            -0.0,
            0.499_999_97,
            -0.499_999_97,
            0.500_000_06,
            8_388_608.0,
            -16_777_216.0,
            f32::NAN,
        ]);
        for t in values {
            assert_eq!(
                round_away(t).to_bits(),
                t.round().to_bits(),
                "round({t}) = {}",
                t.round()
            );
        }
    }

    /// Activations with a zero block, outliers, and a block whose largest
    /// magnitude appears twice (first negative).
    fn awkward_acts(m: usize, k: usize) -> Vec<f32> {
        let mut x = noise(m * k, 4, 3.0);
        x[..QK].fill(0.0);
        x[QK + 3] = 127.0;
        x[QK + 4] = 0.5;
        x[QK + 5] = -1.5;
        x[2 * QK + 10] = -9.0;
        x[2 * QK + 20] = 9.0;
        x
    }

    #[test]
    fn activations_quantise_like_candles_q8_0_and_q8_1() {
        let (m, k) = (3, 512);
        let x = awkward_acts(m, k);
        let t = Tensor::from_vec(x.clone(), (m, k), &Device::Cpu).unwrap();
        for (dtype, with_sums, size, quants_at) in [
            (GgmlDType::Q8_0, false, 34, 2),
            (GgmlDType::Q8_1, true, 36, 4),
        ] {
            let q = QTensor::quantize(&t, dtype).unwrap();
            let bytes = q.data().unwrap();
            let act = Acts::new(&x, m, k, with_sums);
            for j in 0..m {
                for b in 0..k / QK {
                    let block = &bytes[(j * (k / QK) + b) * size..][..size];
                    assert_eq!(
                        act.scales[b * act.m + j],
                        f16_le(block, 0),
                        "{dtype:?} scale of row {j} block {b}"
                    );
                    if with_sums {
                        assert_eq!(
                            act.sums[b * act.m + j],
                            f16_le(block, 2),
                            "{dtype:?} sum term of row {j} block {b}"
                        );
                    }
                    for e in 0..QK {
                        assert_eq!(
                            act_quant(&act, b, j, e),
                            i16::from(block[quants_at + e] as i8),
                            "{dtype:?} quant {e} of row {j} block {b}"
                        );
                    }
                }
            }
        }
    }

    /// Row exactness at the weight shapes of a real 100B-class MoE model
    /// (n not a multiple of the chunk size: the pooled path has tail chunks).
    #[test]
    fn real_shapes_are_row_exact() {
        for dtype in DTYPES {
            for (n, k) in [
                (4104usize, 4096usize),
                (2056, 2048),
                (4096, 2048),
                (1030, 4096),
            ] {
                assert_eq!(k % QK, 0);
                let w = weight(n, k, dtype, 3);
                let q = QmmWeight::new(&w).unwrap();
                let x = noise(9 * k, 9, 2.0);
                let alone: Vec<Vec<f32>> = (0..9)
                    .map(|j| q.forward(&x[j * k..(j + 1) * k], 1).unwrap())
                    .collect();
                for m in [2, 3, 4, 5, 6, 7, 8, 9] {
                    let batch = q.forward(&x[..m * k], m).unwrap();
                    for j in 0..m {
                        assert_eq!(
                            batch[j * n..(j + 1) * n],
                            alone[j][..],
                            "{dtype:?} {n}x{k}: row {j} of a batch of {m}"
                        );
                    }
                }
            }
        }
    }

    /// The determinism contract: a row's bits do not depend on the batch.
    #[test]
    fn a_row_is_bit_identical_alone_and_in_any_batch() {
        for dtype in DTYPES {
            let b = QK;
            for (n, k) in [(10, 2 * b), (512, 512)] {
                let w = weight(n, k, dtype, 3);
                let q = QmmWeight::new(&w).unwrap();
                let x = noise(17 * k, 9, 2.0);
                let alone: Vec<Vec<f32>> = (0..17)
                    .map(|j| q.forward(&x[j * k..(j + 1) * k], 1).unwrap())
                    .collect();
                for m in [2, 3, 4, 5, 8, 16, 17] {
                    let batch = q.forward(&x[..m * k], m).unwrap();
                    for j in 0..m {
                        assert_eq!(
                            batch[j * n..(j + 1) * n],
                            alone[j][..],
                            "{dtype:?} {n}x{k}: row {j} of a batch of {m}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn zero_activations_give_exact_zeros() {
        for dtype in DTYPES {
            let b = QK;
            let w = weight(9, 2 * b, dtype, 6);
            let q = QmmWeight::new(&w).unwrap();
            let out = q.forward(&vec![0f32; 4 * 2 * b], 4).unwrap();
            assert!(out.iter().all(|v| *v == 0.0), "{dtype:?}: {out:?}");
        }
    }

    #[test]
    fn other_dtypes_and_shapes_are_left_to_candle() {
        for dtype in DTYPES {
            let w = weight(4, 2 * QK, dtype, 1);
            assert!(QmmWeight::new(&w).is_some(), "{dtype:?}");
        }
        // The K-quants stay on candle's per-row kernel (see the module docs).
        for dtype in [
            GgmlDType::Q2K,
            GgmlDType::Q3K,
            GgmlDType::Q4K,
            GgmlDType::Q5K,
            GgmlDType::Q6K,
            GgmlDType::F16,
            GgmlDType::F32,
            GgmlDType::Q8_1,
        ] {
            let w = weight(4, 256, dtype, 1);
            assert!(QmmWeight::new(&w).is_none(), "{dtype:?}");
        }
        let t = Tensor::zeros((2, 4, 64), candle_core::DType::F32, &Device::Cpu).unwrap();
        let three_d = Arc::new(QTensor::quantize(&t, GgmlDType::Q8_0).unwrap());
        assert!(QmmWeight::new(&three_d).is_none());
    }

    // ---- MXFP4 ----------------------------------------------------------

    use crate::provider::local::gguf::GgufType;
    use crate::provider::local::model::quant::dequant_row;
    use crate::provider::local::model::tiny::mxfp4_bytes;

    fn mxfp4_weight(n: usize, k: usize, seed: u64) -> (Arc<[u8]>, Vec<f32>) {
        let bytes = mxfp4_bytes(&noise(n * k, seed, 3.0));
        let row_bytes = k / QK * 17;
        let mut values = vec![0f32; n * k];
        for (src, dst) in bytes
            .chunks_exact(row_bytes)
            .zip(values.chunks_exact_mut(k))
        {
            dequant_row(GgufType::Mxfp4, src, 1.0, dst).unwrap();
        }
        (Arc::from(bytes), values)
    }

    #[test]
    fn mxfp4_matches_the_f64_reference() {
        for (n, k, m) in shapes() {
            let (bytes, wd) = mxfp4_weight(n, k, 11);
            let x = noise(m * k, 5, 1.0);
            let got = QmmWeight::mxfp4(bytes, n, k)
                .unwrap()
                .forward(&x, m)
                .unwrap();
            let xd = rounded_acts(&x, m, k);
            let mut scale = 0f64;
            let mut want = vec![0f64; m * n];
            for j in 0..m {
                for r in 0..n {
                    want[j * n + r] = (0..k)
                        .map(|e| f64::from(wd[r * k + e]) * xd[j * k + e])
                        .sum();
                    scale = scale.max(want[j * n + r].abs());
                }
            }
            for (g, w) in got.iter().zip(&want) {
                assert!(
                    (f64::from(*g) - w).abs() <= 1e-4 * scale,
                    "{n}x{k} m={m}: {g} vs {w}"
                );
            }
        }
    }

    #[test]
    fn mxfp4_rows_are_independent_of_the_batch_at_real_shapes() {
        // The expert matrices of the real model: gate/up (2304 x 5120) and
        // down (5120 x 2304).
        for (n, k) in [(2304, 5120), (5120, 2304)] {
            let (bytes, _) = mxfp4_weight(n, k, 3);
            let q = QmmWeight::mxfp4(bytes, n, k).unwrap();
            let x = noise(17 * k, 9, 1.0);
            let batch = q.forward(&x, 17).unwrap();
            for j in [0usize, 5, 16] {
                let alone = q.forward(&x[j * k..(j + 1) * k], 1).unwrap();
                assert_eq!(alone, batch[j * n..(j + 1) * n], "{n}x{k} row {j}");
            }
            let part = q.forward(&x[..3 * k], 3).unwrap();
            assert_eq!(part, batch[..3 * n]);
            let rows = q.forward_rows(&x, 17, n / 3).unwrap();
            for j in 0..17 {
                assert_eq!(
                    rows[j * (n / 3)..(j + 1) * (n / 3)],
                    batch[j * n..j * n + n / 3]
                );
            }
        }
    }

    #[test]
    fn mxfp4_rejects_a_wrong_byte_count() {
        assert!(QmmWeight::mxfp4(Arc::from(vec![0u8; 16]), 1, 32).is_none());
        assert!(QmmWeight::mxfp4(Arc::from(vec![0u8; 17]), 1, 32).is_some());
        assert!(QmmWeight::mxfp4(Arc::from(vec![0u8; 17]), 1, 48).is_none());
    }
}
