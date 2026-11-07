//! Row-batched quantized matmul for Q4_0, Q4_1, Q5_0, Q5_1 and Q8_0 weights.
//!
//! candle's quantized matmul loops over the activation rows and streams the
//! whole weight matrix once per row, so verifying `k + 1` drafted tokens in one
//! forward pass costs close to `k + 1` decode steps. [`QmmWeight::forward_rows`]
//! reads every weight block once and dots it against all `m` activation rows,
//! so a batch of `m` rows costs about one row of memory traffic plus `m` times
//! the (much cheaper) integer arithmetic. Measured on a Ryzen 9 9950X3D2, the
//! kernel is 2-4x faster than candle's for Q4_1, Q5_0 and Q5_1 from `m = 4`
//! rows on.
//!
//! # Arithmetic
//!
//! The activations are quantised the way ggml pairs them with the weight
//! dtype, exactly as candle's vec-dot partners do:
//!
//! - Q4_0, Q5_0, Q8_0: Q8_0 (`d = max|x| / 127` rounded to f16,
//!   `q = round(x / d)`, 32-element blocks);
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
//! depends on that row alone, whatever the batch). Measured on the same CPU,
//! candle's K-quant prefill was already at parity with Q4_0's (phi4-mini
//! `q4_k_m` against `q4_0`, long-prompt perf test; the 3D V-cache L3 keeps the
//! per-row weight re-reads cheap), and a row-batched safe-Rust K-quant kernel
//! written for this module was 2-3x slower than candle's at `m = 1` (the Muse
//! Glimmer FFN shapes: 2-3.2 ms against 0.9-1.6 ms). The dispatch cannot pick
//! the faster kernel per `m` either: a row's bits must not depend on the batch
//! size, so one dtype is served by one kernel.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::utils::barrier_pool;
use half::f16;

/// Elements per quantisation block of the served formats (Q4_0, Q4_1, Q5_0,
/// Q5_1, Q8_0), and of their Q8_0/Q8_1 activations.
const QK: usize = 32;
/// Lanes: a block's two halves of sixteen elements are summed lane-wise.
const LANES: usize = QK / 2;
/// Weight rows decoded together.
const NR: usize = 4;
/// Activation rows per full register tile.
const MR: usize = 4;
/// Activation rows of the tile that serves a remainder of one to three rows.
const MIN_TILE: usize = 3;
/// Below this many weight elements the matmul runs on the calling thread.
const SERIAL_BELOW: usize = 1 << 17;

/// A block's 32 quants as i16 lanes: elements `0..16`, then `16..32`.
///
/// Every product of a weight (`-128..=127`, or `0..=31` for the 5-bit
/// formats) and an activation quant (`-127..=127`) fits an `i16`, and so does
/// the sum of two, so the halves are combined in `i16` without overflow.
type Halves = [[i16; LANES]; 2];

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
    fn halves(block: &Self::Block) -> Halves;
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
}

/// Activations of one call, quantised to Q8_0 (or Q8_1, which adds the sum
/// term) and laid out block-major so the blocks of consecutive rows at one
/// block index are contiguous. At least [`MIN_TILE`] rows are stored (zero
/// padding beyond `m`): see [`chunk`].
struct Acts {
    /// Blocks per row.
    nb: usize,
    /// Rows.
    m: usize,
    /// Rows stored per block index: `m.max(MIN_TILE)`.
    stride: usize,
    /// `[nb][stride]` quants.
    qs: Vec<Halves>,
    /// `[nb][stride]` f16-rounded scales.
    scales: Vec<f32>,
    /// `[nb][stride]` Q8_1 sum terms `f16(f16(sum of the quants) * d)`; empty
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

impl Acts {
    /// Quantise `x` (`m` rows of `k` floats, `k` a multiple of [`QK`]) the way
    /// candle's `BlockQ8_0::from_float` does, or `BlockQ8_1::from_float` when
    /// `with_sums`.
    fn new(x: &[f32], m: usize, k: usize, with_sums: bool) -> Self {
        let nb = k / QK;
        let stride = m.max(MIN_TILE);
        let mut qs = vec![[[0i16; LANES]; 2]; nb * stride];
        let mut scales = vec![0f32; nb * stride];
        let mut sums = vec![0f32; if with_sums { nb * stride } else { 0 }];
        for (j, row) in x.chunks_exact(k).enumerate().take(m) {
            for (b, src) in row.as_chunks::<QK>().0.iter().enumerate() {
                // `max` ignores NaN and is associative, so eight running
                // maxima give the sequential fold's value.
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
                scales[b * stride + j] = d16.to_f32();
                let dst = &mut qs[b * stride + j];
                for (half, vals) in dst.iter_mut().zip(src.as_chunks::<LANES>().0) {
                    for (q, &v) in half.iter_mut().zip(vals) {
                        *q = i16::from(round_away(v * id) as i8);
                    }
                }
                if with_sums {
                    let sum: i32 = dst.iter().flatten().map(|&q| i32::from(q)).sum();
                    sums[b * stride + j] = (f16::from_f32(sum as f32) * d16).to_f32();
                }
            }
        }
        Self {
            nb,
            m,
            stride,
            qs,
            scales,
            sums,
        }
    }
}

/// Exact integer dot of one weight block with one activation block, as
/// sixteen lane sums.
#[inline(always)]
fn dot_block(w: &Halves, x: &Halves) -> [i32; LANES] {
    let mut p = [0i32; LANES];
    for (l, pl) in p.iter_mut().enumerate() {
        *pl = i32::from(
            w[0][l]
                .wrapping_mul(x[0][l])
                .wrapping_add(w[1][l].wrapping_mul(x[1][l])),
        );
    }
    p
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

/// Repeat `$body` with `$i` bound to 0, 1, 2 and 3: a full unroll, so indices
/// into small arrays are constants and the arrays live in registers.
macro_rules! unroll4 {
    ($i:ident => $body:expr) => {{
        {
            let $i = 0usize;
            $body
        }
        {
            let $i = 1usize;
            $body
        }
        {
            let $i = 2usize;
            $body
        }
        {
            let $i = 3usize;
            $body
        }
    }};
}

/// `R <= 4` weight rows against activation rows `j0..j0 + M` (`M <= 4`) of a
/// 32-element-block format: `out[i][jj]` is the dot of weight row `i` with
/// activation row `j0 + jj`. Unrolled so the accumulators stay in registers.
///
/// A format with a block minimum (`L::HAS_MIN`) adds `min * s` per block to a
/// scalar sum in block order, where `s` is the activation block's Q8_1 sum
/// term, after the lanes are reduced.
#[inline(always)]
#[allow(clippy::needless_range_loop)] // indices are unrolled constants
fn legacy_tile<L: Legacy, const R: usize, const M: usize>(
    rows: &[&[L::Block]; R],
    act: &Acts,
    j0: usize,
) -> [[f32; M]; R] {
    let mut acc = [[[0f32; LANES]; MR]; NR];
    let mut extra = [[0f32; MR]; NR];
    // Every row holds `act.nb` blocks: `run` validates the weight's byte length
    // against it before any tile runs.
    debug_assert!(rows.iter().all(|row| row.len() == act.nb));
    for b in 0..act.nb {
        let xs = &act.qs[b * act.stride + j0..b * act.stride + j0 + M];
        let sc = &act.scales[b * act.stride + j0..b * act.stride + j0 + M];
        let sums: &[f32] = if L::HAS_MIN {
            &act.sums[b * act.stride + j0..b * act.stride + j0 + M]
        } else {
            &[]
        };
        let mut d = [0f32; NR];
        let mut mn = [0f32; NR];
        let mut w = [[[0i16; LANES]; 2]; NR];
        unroll4!(r => if r < R {
            d[r] = L::scale(&rows[r][b]);
            if L::HAS_MIN {
                mn[r] = L::min(&rows[r][b]);
            }
            w[r] = L::halves(&rows[r][b]);
        });
        unroll4!(jj => if jj < M {
            let dx = sc[jj];
            unroll4!(r => if r < R {
                let p = dot_block(&w[r], &xs[jj]);
                let s = d[r] * dx;
                for l in 0..LANES {
                    acc[r][jj][l] += p[l] as f32 * s;
                }
                if L::HAS_MIN {
                    extra[r][jj] += mn[r] * sums[jj];
                }
            });
        });
    }
    let mut out = [[0f32; M]; R];
    for r in 0..R {
        for jj in 0..M {
            out[r][jj] = if L::HAS_MIN {
                hsum(&acc[r][jj]) + extra[r][jj]
            } else {
                hsum(&acc[r][jj])
            };
        }
    }
    out
}

/// Store `tile[i][jj]` (weight row `r0 + i`, activation row `j0 + jj`) into the
/// chunk's `res[j * rows + r]` scratch, skipping padding rows (`>= m`).
#[inline(always)]
fn put<const R: usize, const M: usize>(
    tile: [[f32; M]; R],
    res: &mut [f32],
    m: usize,
    rows: usize,
    r0: usize,
    j0: usize,
) {
    for (i, tile_row) in tile.iter().enumerate() {
        for (jj, v) in tile_row.iter().enumerate() {
            if j0 + jj < m {
                res[(j0 + jj) * rows + r0 + i] = *v;
            }
        }
    }
}

/// `R` weight rows against every activation row: full [`MR`]-row tiles, then
/// one [`MIN_TILE`]-row tile for the remaining 1..=3 rows, placed to end at row
/// `m` (overlapping rows are recomputed identically) or, when `m` is below
/// [`MIN_TILE`], starting at row 0 over the zero padding.
///
/// Tiles of one and two rows compile to much slower code than three and four
/// (measured: a 6144x1536 Q4_0 matrix takes 1.9 ms and 3.4 ms on one thread
/// for one and two rows, 0.7 ms for three), so a lone row is cheaper as a
/// three-row tile with two dummy rows.
#[inline(always)]
fn rows_of<L: Legacy, const R: usize>(
    rs: &[&[L::Block]; R],
    act: &Acts,
    res: &mut [f32],
    rows: usize,
    r: usize,
) {
    let m = act.m;
    let mut j0 = 0;
    while m - j0 >= MR {
        put(legacy_tile::<L, R, MR>(rs, act, j0), res, m, rows, r, j0);
        j0 += MR;
    }
    if j0 < m {
        let j1 = m.saturating_sub(MIN_TILE);
        put(
            legacy_tile::<L, R, MIN_TILE>(rs, act, j1),
            res,
            m,
            rows,
            r,
            j1,
        );
    }
}

/// Weight rows `w` (a whole number of rows) against every activation row;
/// `res[j * rows + r]` receives the dot of weight row `r` with activation row
/// `j`.
fn chunk<L: Legacy>(w: &[u8], act: &Acts, res: &mut [f32]) {
    let row_bytes = act.nb * size_of::<L::Block>();
    let rows = w.len() / row_bytes;
    let row = |r: usize| L::blocks(&w[r * row_bytes..(r + 1) * row_bytes]);
    let mut r = 0;
    while r + NR <= rows {
        let rs: [&[L::Block]; NR] = std::array::from_fn(|i| row(r + i));
        rows_of::<L, NR>(&rs, act, res, rows, r);
        r += NR;
    }
    while r < rows {
        rows_of::<L, 1>(&[row(r)], act, res, rows, r);
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

/// A quantized `[n, k]` weight (`y = x * w^T`) computed by the row-batched
/// kernel: Q4_0, Q4_1, Q5_0, Q5_1 or Q8_0. Holds
/// the candle tensor and reads its bytes in place.
pub(crate) struct QmmWeight {
    qt: Arc<QTensor>,
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
            qt: Arc::clone(qt),
            kind,
            n,
            k,
        })
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
        let data = self.qt.data()?;
        let row_bytes = data.len() / self.n;
        let weights = data.get(..rows * row_bytes).unwrap_or(&data[..]);
        let (k, o) = (self.k, &mut out);
        match self.kind {
            Kind::Q4_0 => run::<Q4_0>(weights, x, m, k, rows, o)?,
            Kind::Q4_1 => run::<Q4_1>(weights, x, m, k, rows, o)?,
            Kind::Q5_0 => run::<Q5_0>(weights, x, m, k, rows, o)?,
            Kind::Q5_1 => run::<Q5_1>(weights, x, m, k, rows, o)?,
            Kind::Q8_0 => run::<Q8_0>(weights, x, m, k, rows, o)?,
        }
        Ok(out)
    }
}

/// Quantise the `m` activation rows for the format `L`, split the `[m, n]`
/// output into `rows_per_chunk`-wide column chunks per activation row and
/// compute them over the pool; each worker claims chunks dynamically,
/// computes one into a private scratch and copies it out.
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
    let pool = barrier_pool();
    let threads = pool.n_workers() + 1;
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

    let next = AtomicUsize::new(0);
    let work = |_: usize| {
        let mut res = vec![0f32; m * rows_per_chunk];
        loop {
            let c = next.fetch_add(1, Ordering::Relaxed);
            if c >= n_chunks {
                break;
            }
            let r0 = c * rows_per_chunk;
            let rows = rows_per_chunk.min(n - r0);
            chunk::<L>(
                &data[r0 * row_bytes..(r0 + rows) * row_bytes],
                act,
                &mut res[..m * rows],
            );
            for (j, row) in dst.chunks_exact(n_chunks).enumerate() {
                row[c]
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .copy_from_slice(&res[j * rows..(j + 1) * rows]);
            }
        }
    };
    if serial {
        work(0);
    } else {
        pool.execute(work);
    }
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

    /// The activations as the kernel sees them: quantised and dequantised.
    fn rounded_acts(x: &[f32], m: usize, k: usize) -> Vec<f64> {
        let mut xd = vec![0f64; m * k];
        let act = Acts::new(x, m, k, false);
        for j in 0..m {
            for b in 0..act.nb {
                for e in 0..QK {
                    xd[j * k + b * QK + e] =
                        f64::from(act.qs[b * act.stride + j][e / LANES][e % LANES])
                            * f64::from(act.scales[b * act.stride + j]);
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
                        min.abs() * f64::from(act.sums[b * act.stride + j]).abs() * 1.0e-3
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
                        act.scales[b * act.stride + j],
                        f16_le(block, 0),
                        "{dtype:?} scale of row {j} block {b}"
                    );
                    if with_sums {
                        assert_eq!(
                            act.sums[b * act.stride + j],
                            f16_le(block, 2),
                            "{dtype:?} sum term of row {j} block {b}"
                        );
                    }
                    for e in 0..QK {
                        assert_eq!(
                            act.qs[b * act.stride + j][e / LANES][e % LANES],
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
}
