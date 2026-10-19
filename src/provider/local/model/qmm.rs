//! Row-batched quantized matmul for Q4_0 and Q8_0 weights.
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
//! The activations are quantised to Q8_0 exactly as candle's Q8_0 vec-dot
//! partner does (`d = max|x| / 127` rounded to f16, `q = round(x / d)`); the
//! products are the integer dot of one 32-element weight block with one
//! activation block, scaled by the product of the two f32-widened f16 scales.
//!
//! The value for activation row `j` and weight row `r` is a fixed function of
//! that row pair alone: sixteen `f32` lanes, lane `l` adding
//! `(w[l] * x[l] + w[l + 16] * x[l + 16]) * scale` per block in block order
//! (the integer part is exact, so how it is computed is free), reduced by a
//! fixed pairwise tree. No cross-row term exists and tile shape never changes
//! the order, so a row's result is bit-identical whether it is computed alone
//! or in a batch of any size, which speculative-decoding verification needs.
//! It differs from candle's own result only in float rounding order.
//!
//! Work is split in chunks of weight rows over candle's persistent barrier pool
//! (the one its own quantized matmul uses, so the two never oversubscribe the
//! cores), with dynamic chunk claiming.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::utils::barrier_pool;
use half::f16;

/// Elements per quantisation block, shared by Q4_0 and Q8_0.
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
/// Every product of a weight (`-128..=127`) and an activation quant
/// (`-127..=127`) fits an `i16`, and so does the sum of two, so the halves are
/// combined in `i16` without overflow.
type Halves = [[i16; LANES]; 2];

/// A weight block format: its bytes, f32 scale and quants.
trait Format {
    type Block;
    /// The blocks of one row (`row` is a whole number of them).
    fn blocks(row: &[u8]) -> &[Self::Block];
    fn scale(block: &Self::Block) -> f32;
    fn halves(block: &Self::Block) -> Halves;
}

/// `d: f16`, then 16 bytes of nibbles: element `i` is the low nibble of byte
/// `i` and element `i + 16` its high nibble, each minus 8.
struct Q4_0;

impl Format for Q4_0 {
    type Block = [u8; 18];

    #[inline(always)]
    fn blocks(row: &[u8]) -> &[Self::Block] {
        row.as_chunks().0
    }

    #[inline(always)]
    fn scale(block: &Self::Block) -> f32 {
        f16::from_le_bytes([block[0], block[1]]).to_f32()
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

/// `d: f16`, then 32 signed bytes.
struct Q8_0;

impl Format for Q8_0 {
    type Block = [u8; 34];

    #[inline(always)]
    fn blocks(row: &[u8]) -> &[Self::Block] {
        row.as_chunks().0
    }

    #[inline(always)]
    fn scale(block: &Self::Block) -> f32 {
        f16::from_le_bytes([block[0], block[1]]).to_f32()
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

/// Activations of one call, quantised to Q8_0 and laid out block-major so the
/// blocks of consecutive rows at one block index are contiguous. At least
/// [`MIN_TILE`] rows are stored (zero padding beyond `m`): see [`chunk`].
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
    /// candle's `BlockQ8_0::from_float` does.
    fn new(x: &[f32], m: usize, k: usize) -> Self {
        let nb = k / QK;
        let stride = m.max(MIN_TILE);
        let mut qs = vec![[[0i16; LANES]; 2]; nb * stride];
        let mut scales = vec![0f32; nb * stride];
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
                scales[b * stride + j] = f16::from_f32(d).to_f32();
                let dst = &mut qs[b * stride + j];
                for (half, vals) in dst.iter_mut().zip(src.as_chunks::<LANES>().0) {
                    for (q, &v) in half.iter_mut().zip(vals) {
                        *q = i16::from(round_away(v * id) as i8);
                    }
                }
            }
        }
        Self {
            nb,
            m,
            stride,
            qs,
            scales,
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

/// `R <= 4` weight rows against activation rows `j0..j0 + M` (`M <= 4`):
/// `out[i][jj]` is the dot of weight row `i` with activation row `j0 + jj`.
/// Unrolled so the accumulators stay in registers.
#[inline(always)]
#[allow(clippy::needless_range_loop)] // indices are unrolled constants
fn tile<F: Format, const R: usize, const M: usize>(
    rows: &[&[F::Block]; R],
    act: &Acts,
    j0: usize,
) -> [[f32; M]; R] {
    let mut acc = [[[0f32; LANES]; MR]; NR];
    // Every row holds `act.nb` blocks: `run` validates the weight's byte length
    // against it before any tile runs.
    debug_assert!(rows.iter().all(|row| row.len() == act.nb));
    for b in 0..act.nb {
        let xs = &act.qs[b * act.stride + j0..b * act.stride + j0 + M];
        let sc = &act.scales[b * act.stride + j0..b * act.stride + j0 + M];
        let mut d = [0f32; NR];
        let mut w = [[[0i16; LANES]; 2]; NR];
        unroll4!(r => if r < R {
            d[r] = F::scale(&rows[r][b]);
            w[r] = F::halves(&rows[r][b]);
        });
        unroll4!(jj => if jj < M {
            let dx = sc[jj];
            unroll4!(r => if r < R {
                let p = dot_block(&w[r], &xs[jj]);
                let s = d[r] * dx;
                for l in 0..LANES {
                    acc[r][jj][l] += p[l] as f32 * s;
                }
            });
        });
    }
    let mut out = [[0f32; M]; R];
    for r in 0..R {
        for jj in 0..M {
            out[r][jj] = hsum(&acc[r][jj]);
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
fn rows_of<F: Format, const R: usize>(
    rs: &[&[F::Block]; R],
    act: &Acts,
    res: &mut [f32],
    rows: usize,
    r: usize,
) {
    let mut j0 = 0;
    while act.m - j0 >= MR {
        put(tile::<F, R, MR>(rs, act, j0), res, act.m, rows, r, j0);
        j0 += MR;
    }
    if j0 < act.m {
        let j1 = act.m.saturating_sub(MIN_TILE);
        put(tile::<F, R, MIN_TILE>(rs, act, j1), res, act.m, rows, r, j1);
    }
}

/// Weight rows `w` (a whole number of rows) against every activation row;
/// `res[j * rows + r]` receives the dot of weight row `r` with activation row
/// `j`.
fn chunk<F: Format>(w: &[u8], act: &Acts, res: &mut [f32]) {
    let row_bytes = act.nb * size_of::<F::Block>();
    let rows = w.len() / row_bytes;
    let row = |r: usize| F::blocks(&w[r * row_bytes..(r + 1) * row_bytes]);
    let mut r = 0;
    while r + NR <= rows {
        let rs: [&[F::Block]; NR] = std::array::from_fn(|i| row(r + i));
        rows_of::<F, NR>(&rs, act, res, rows, r);
        r += NR;
    }
    while r < rows {
        rows_of::<F, 1>(&[row(r)], act, res, rows, r);
        r += 1;
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Q4_0,
    Q8_0,
}

/// A Q4_0 or Q8_0 `[n, k]` weight (`y = x * w^T`) computed by the row-batched
/// kernel. Holds the candle tensor and reads its bytes in place.
pub(crate) struct QmmWeight {
    qt: Arc<QTensor>,
    kind: Kind,
    n: usize,
    k: usize,
}

impl QmmWeight {
    /// The kernel's view of `qt`, or `None` when it is not a 2-D Q4_0 or Q8_0
    /// tensor with rows a multiple of [`QK`] long (candle's path serves it).
    pub(crate) fn new(qt: &Arc<QTensor>) -> Option<Self> {
        let kind = match qt.dtype() {
            GgmlDType::Q4_0 => Kind::Q4_0,
            GgmlDType::Q8_0 => Kind::Q8_0,
            _ => return None,
        };
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
        let k = self.k;
        let mut out = vec![0f32; m * rows];
        if m == 0 {
            return Ok(out);
        }
        let data = self.qt.data()?;
        let row_bytes = data.len() / self.n;
        let weights = data.get(..rows * row_bytes).unwrap_or(&data[..]);
        let act = Acts::new(x, m, k);
        match self.kind {
            Kind::Q4_0 => run::<Q4_0>(weights, &act, rows, &mut out)?,
            Kind::Q8_0 => run::<Q8_0>(weights, &act, rows, &mut out)?,
        }
        Ok(out)
    }
}

/// Split the `[m, n]` output into `rows_per_chunk`-wide column chunks per
/// activation row and compute them over the pool; each worker claims chunks
/// dynamically, computes one into a private scratch and copies it out.
fn run<F: Format>(data: &[u8], act: &Acts, n: usize, out: &mut [f32]) -> candle_core::Result<()> {
    let row_bytes = act.nb * size_of::<F::Block>();
    if data.len() != n * row_bytes {
        candle_core::bail!(
            "qmm: weight holds {} bytes, expected {n} rows of {row_bytes}",
            data.len()
        );
    }
    let pool = barrier_pool();
    let threads = pool.n_workers() + 1;
    let serial = n * act.nb * QK < SERIAL_BELOW || threads == 1;
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
        let mut res = vec![0f32; act.m * rows_per_chunk];
        loop {
            let c = next.fetch_add(1, Ordering::Relaxed);
            if c >= n_chunks {
                break;
            }
            let r0 = c * rows_per_chunk;
            let rows = rows_per_chunk.min(n - r0);
            chunk::<F>(
                &data[r0 * row_bytes..(r0 + rows) * row_bytes],
                act,
                &mut res[..act.m * rows],
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

    /// Deterministic pseudo-random floats in `[-1, 1)` scaled by `scale`.
    fn noise(len: usize, seed: u64, scale: f32) -> Vec<f32> {
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

    fn weight(n: usize, k: usize, dtype: GgmlDType, seed: u64) -> Arc<QTensor> {
        let t = Tensor::from_vec(noise(n * k, seed, 0.5), (n, k), &Device::Cpu).unwrap();
        Arc::new(QTensor::quantize(&t, dtype).unwrap())
    }

    /// Reference in f64: dequantised weights times dequantised (Q8_0-rounded)
    /// activations.
    fn reference(w: &QTensor, x: &[f32], m: usize) -> Vec<f64> {
        let [n, k] = *w.shape().dims() else {
            unreachable!()
        };
        let wd = w
            .dequantize(&Device::Cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let act = Acts::new(x, m, k);
        let mut xd = vec![0f64; m * k];
        for j in 0..m {
            for b in 0..act.nb {
                for e in 0..QK {
                    xd[j * k + b * QK + e] =
                        f64::from(act.qs[b * act.stride + j][e / LANES][e % LANES])
                            * f64::from(act.scales[b * act.stride + j]);
                }
            }
        }
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

    #[test]
    fn matches_the_f64_reference_and_candle() {
        for dtype in [GgmlDType::Q4_0, GgmlDType::Q8_0] {
            // n = 10 and m = 7 leave tails in both tilings; 600 rows exceed
            // SERIAL_BELOW so the pooled path runs too.
            for (n, k, m) in [(10, 64, 7), (3, 32, 1), (600, 256, 5), (64, 96, 17)] {
                let w = weight(n, k, dtype, 11);
                let x = noise(m * k, 5, 1.0);
                let q = QmmWeight::new(&w).unwrap();
                let got = q.forward(&x, m).unwrap();
                let want = reference(&w, &x, m);
                let scale = want.iter().fold(0f64, |a, v| a.max(v.abs()));
                for (g, r) in got.iter().zip(&want) {
                    assert!(
                        (f64::from(*g) - r).abs() <= 1e-4 * scale,
                        "{dtype:?} {n}x{k} m={m}: {g} vs {r}"
                    );
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

    /// A prefix of the weight rows computes exactly the first outputs of the
    /// whole product, whatever the tiling.
    #[test]
    fn a_weight_row_prefix_equals_the_start_of_the_product() {
        for dtype in [GgmlDType::Q4_0, GgmlDType::Q8_0] {
            for (n, k, m, rows) in [(10, 64, 3, 7), (600, 256, 2, 450), (64, 96, 5, 1)] {
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

    #[test]
    fn activations_quantise_like_candles_q8_0() {
        let (m, k) = (3, 96);
        let mut x = noise(m * k, 4, 3.0);
        x[..QK].fill(0.0);
        x[QK + 3] = 127.0;
        x[QK + 4] = 0.5;
        x[QK + 5] = -1.5;
        let t = Tensor::from_vec(x.clone(), (m, k), &Device::Cpu).unwrap();
        let q = QTensor::quantize(&t, GgmlDType::Q8_0).unwrap();
        let bytes = q.data().unwrap();
        let act = Acts::new(&x, m, k);
        for j in 0..m {
            for b in 0..k / QK {
                let block = &bytes[(j * (k / QK) + b) * 34..][..34];
                let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
                assert_eq!(
                    act.scales[b * act.stride + j],
                    d,
                    "scale of row {j} block {b}"
                );
                for e in 0..QK {
                    assert_eq!(
                        act.qs[b * act.stride + j][e / LANES][e % LANES],
                        i16::from(block[2 + e] as i8),
                        "quant {e} of row {j} block {b}"
                    );
                }
            }
        }
    }

    #[test]
    fn a_row_is_bit_identical_alone_and_in_any_batch() {
        for dtype in [GgmlDType::Q4_0, GgmlDType::Q8_0] {
            for (n, k) in [(10, 64), (512, 256)] {
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
    fn other_dtypes_and_shapes_are_left_to_candle() {
        let q40 = weight(4, 64, GgmlDType::Q4_0, 1);
        assert!(QmmWeight::new(&q40).is_some());
        let q80 = weight(4, 64, GgmlDType::Q8_0, 1);
        assert!(QmmWeight::new(&q80).is_some());
        let q4k = weight(4, 256, GgmlDType::Q4K, 1);
        assert!(QmmWeight::new(&q4k).is_none());
        let t = Tensor::zeros((2, 4, 64), candle_core::DType::F32, &Device::Cpu).unwrap();
        let three_d = Arc::new(QTensor::quantize(&t, GgmlDType::Q8_0).unwrap());
        assert!(QmmWeight::new(&three_d).is_none());
    }
}
