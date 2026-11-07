//! The K-quant formats (Q2_K, Q3_K, Q4_K, Q5_K, Q6_K) of the row-batched
//! matmul, and their Q8_K activations. See the parent module for the
//! arithmetic.
//!
//! A K-quant block holds 256 elements in sixteen groups of sixteen, each with
//! its own scale (Q4_K and Q5_K share one scale and minimum per two groups).
//! Decoding a weight block expands it to `i16` elements already multiplied by
//! their group scale (`quant * scale`, at most `4096` in magnitude), so the
//! block's dot with a Q8_K activation block is one exact integer dot
//! (`4096 * 128 * 256` fits an `i32`). The minimum of Q2_K, Q4_K and Q5_K is
//! `dmin * sum(min_g * dy * bsum_g)`, a sixteen-term float dot per block
//! against the activation's per-group quant sums.

use super::{Format, LANES, MIN_TILE, MR, NR, QK_K, Quantised, f16_le, hsum, round_away};

/// Groups per block.
const GROUPS: usize = QK_K / 16;

/// Q8_K activations of one call, laid out block-major like the Q8_0 ones: at
/// least [`MIN_TILE`] rows are stored, zero padding beyond `m`.
pub(super) struct ActsK {
    /// Blocks per row.
    pub(super) nb: usize,
    /// Rows.
    pub(super) m: usize,
    /// Rows stored per block index: `m.max(MIN_TILE)`.
    pub(super) stride: usize,
    /// `[nb][stride]` quants.
    pub(super) qs: Vec<[i16; QK_K]>,
    /// `[nb][stride]` sums of each group of sixteen quants.
    pub(super) bsums: Vec<[i32; GROUPS]>,
    /// `[nb][stride]` block scales (`d`, f32).
    pub(super) scales: Vec<f32>,
}

impl ActsK {
    /// Quantise `x` (`m` rows of `k` floats, `k` a multiple of [`QK_K`]) the
    /// way candle's `BlockQ8K::from_float` does.
    pub(super) fn new(x: &[f32], m: usize, k: usize) -> Self {
        let nb = k / QK_K;
        let stride = m.max(MIN_TILE);
        let mut qs = vec![[0i16; QK_K]; nb * stride];
        let mut bsums = vec![[0i32; GROUPS]; nb * stride];
        let mut scales = vec![0f32; nb * stride];
        for (j, row) in x.chunks_exact(k).enumerate().take(m) {
            for (b, src) in row.as_chunks::<QK_K>().0.iter().enumerate() {
                let i = b * stride + j;
                // `max` ignores NaN and is associative, so sixteen running
                // maxima give the sequential scan's value.
                let mut maxima = [0f32; 16];
                for lane in src.as_chunks::<16>().0 {
                    for (m16, &v) in maxima.iter_mut().zip(lane) {
                        *m16 = m16.max(v.abs());
                    }
                }
                let amax = maxima.iter().fold(0f32, |a, &v| a.max(v));
                if amax == 0.0 {
                    continue;
                }
                // The first element of the largest magnitude, signed.
                let max = src.iter().copied().find(|v| v.abs() == amax);
                let iscale = -127.0 / max.unwrap_or(amax);
                for (q, &v) in qs[i].iter_mut().zip(src) {
                    *q = round_away(iscale * v).clamp(-128.0, 127.0) as i16;
                }
                for (sum, group) in bsums[i].iter_mut().zip(qs[i].as_chunks::<16>().0) {
                    *sum = group.iter().map(|&q| i32::from(q)).sum();
                }
                scales[i] = 1.0 / iscale;
            }
        }
        Self {
            nb,
            m,
            stride,
            qs,
            bsums,
            scales,
        }
    }
}

impl Quantised for ActsK {
    fn rows(&self) -> usize {
        self.m
    }

    fn blocks(&self) -> usize {
        self.nb
    }
}

/// One decoded weight block.
#[derive(Clone)]
pub(super) struct Wide {
    d: f32,
    /// Elements times their group scale.
    ws: [i16; QK_K],
    /// `dmin` times the minimum of each group (all zero for Q3_K and Q6_K).
    mins: [f32; GROUPS],
}

impl Wide {
    const ZERO: Self = Self {
        d: 0.0,
        ws: [0; QK_K],
        mins: [0.0; GROUPS],
    };
}

/// A K-quant block layout.
pub(super) trait KQuant {
    type Block;
    /// The format has group minimums.
    const HAS_MIN: bool;
    /// The blocks of one row (`row` is a whole number of them).
    fn blocks(row: &[u8]) -> &[Self::Block];
    /// Expand `block` into `out`.
    fn decode(block: &Self::Block, out: &mut Wide);
}

/// Exact integer dot of a decoded weight block with a Q8_K activation block,
/// as sixteen lane sums. The pairwise form is what `vpmaddwd` computes.
#[inline(always)]
fn dot_block(w: &[i16; QK_K], x: &[i16; QK_K]) -> [i32; LANES] {
    let mut p = [0i32; LANES];
    for (wc, xc) in w.as_chunks::<32>().0.iter().zip(x.as_chunks::<32>().0) {
        for (l, pl) in p.iter_mut().enumerate() {
            *pl += i32::from(wc[2 * l]) * i32::from(xc[2 * l])
                + i32::from(wc[2 * l + 1]) * i32::from(xc[2 * l + 1]);
        }
    }
    p
}

/// Decode `R` weight rows into `panel`, laid out `[block][NR]`: the decode is
/// shared by every activation tile instead of repeated for each.
fn load<K: KQuant, const R: usize>(rows: &[&[K::Block]; R], panel: &mut Vec<Wide>) {
    let nb = rows.first().map_or(0, |row| row.len());
    panel.resize(nb * NR, Wide::ZERO);
    for (b, blocks) in panel.as_chunks_mut::<NR>().0.iter_mut().enumerate() {
        for (row, out) in rows.iter().zip(blocks) {
            K::decode(&row[b], out);
        }
    }
}

/// `R <= 4` decoded weight rows against activation rows `j0..j0 + M`
/// (`M <= 4`): the K-quant counterpart of the parent's `legacy_tile`.
///
/// Per block, lane `l` adds the exact integer dot's share times `d * dy`, and
/// for the formats with minimums, `dmin * min * (dy * bsum)` (float products of
/// two exact factors) is subtracted from lane `g`, the group.
#[inline(always)]
#[allow(clippy::needless_range_loop)] // indices are unrolled constants
fn tile<K: KQuant, const R: usize, const M: usize>(
    panel: &[Wide],
    act: &ActsK,
    j0: usize,
) -> [[f32; M]; R] {
    let mut acc = [[[0f32; LANES]; MR]; NR];
    debug_assert_eq!(panel.len(), act.nb * NR);
    for (b, dec) in panel.as_chunks::<NR>().0.iter().enumerate() {
        let at = b * act.stride + j0;
        let xs = &act.qs[at..at + M];
        let bs = &act.bsums[at..at + M];
        let sc = &act.scales[at..at + M];
        unroll4!(jj => if jj < M {
            let mut v = [0f32; LANES];
            if K::HAS_MIN {
                for l in 0..LANES {
                    v[l] = bs[jj][l] as f32 * sc[jj];
                }
            }
            unroll4!(r => if r < R {
                let p = dot_block(&dec[r].ws, &xs[jj]);
                let s = dec[r].d * sc[jj];
                for l in 0..LANES {
                    acc[r][jj][l] += p[l] as f32 * s;
                }
                if K::HAS_MIN {
                    for l in 0..LANES {
                        acc[r][jj][l] -= dec[r].mins[l] * v[l];
                    }
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

macro_rules! k_format {
    ($($t:ident),*) => {$(
        impl Format for $t {
            type Block = <$t as KQuant>::Block;
            type Act = ActsK;
            type Panel = Vec<Wide>;
            const LONE_ROW_TILE: bool = true;

            #[inline(always)]
            fn blocks(row: &[u8]) -> &[Self::Block] {
                <$t as KQuant>::blocks(row)
            }

            fn quantise(x: &[f32], m: usize, k: usize) -> ActsK {
                ActsK::new(x, m, k)
            }

            fn load<const R: usize>(rows: &[&[Self::Block]; R], panel: &mut Vec<Wide>) {
                load::<$t, R>(rows, panel);
            }

            #[inline(always)]
            fn tile<const R: usize, const M: usize>(
                _rows: &[&[Self::Block]; R],
                panel: &Vec<Wide>,
                act: &ActsK,
                j0: usize,
            ) -> [[f32; M]; R] {
                tile::<$t, R, M>(panel, act, j0)
            }
        }
    )*};
}

k_format!(Q2K, Q3K, Q4K, Q5K, Q6K);

/// The six-bit scales and minimums of Q4_K and Q5_K: eight each, packed in
/// twelve bytes (ggml's `get_scale_min_k4`).
fn scales_mins_k4(packed: &[u8]) -> ([i16; 8], [i16; 8]) {
    let mut sc = [0i16; 8];
    let mut mn = [0i16; 8];
    for j in 0..8 {
        if j < 4 {
            sc[j] = i16::from(packed[j] & 63);
            mn[j] = i16::from(packed[j + 4] & 63);
        } else {
            sc[j] = i16::from((packed[j + 4] & 0x0F) | ((packed[j - 4] >> 6) << 4));
            mn[j] = i16::from((packed[j + 4] >> 4) | ((packed[j] >> 6) << 4));
        }
    }
    (sc, mn)
}

/// Store Q4_K and Q5_K's eight group-pair minimums scaled by `dmin`, once for
/// each group of the pair.
fn store_mins(dmin: f32, mins: &[i16; 8], out: &mut [f32; GROUPS]) {
    for (pair, &m) in out.as_chunks_mut::<2>().0.iter_mut().zip(mins) {
        pair.fill(dmin * f32::from(m));
    }
}

/// Sixteen 2-bit quants `(src >> SHIFT) & 3` times `scale`. The shift is a
/// constant so the loop is one vector shift, mask and multiply.
#[inline(always)]
fn two_bits<const SHIFT: u32>(dst: &mut [i16; 16], src: &[u8; 16], scale: i16) {
    for (d, &s) in dst.iter_mut().zip(src) {
        *d = ((i16::from(s) >> SHIFT) & 3) * scale;
    }
}

/// Sixteen Q3_K quants: the 2 bits `(src >> SHIFT) & 3` minus 4 where the
/// `hmask` byte lacks `BIT`, times `scale`.
#[inline(always)]
fn three_bits<const SHIFT: u32, const BIT: u8>(
    dst: &mut [i16; 16],
    src: &[u8; 16],
    hmask: &[u8; 16],
    scale: i16,
) {
    for ((d, &s), &hm) in dst.iter_mut().zip(src).zip(hmask) {
        let low = (i16::from(s) >> SHIFT) & 3;
        let high = i16::from(u8::from(hm & BIT == 0)) * 4;
        *d = (low - high) * scale;
    }
}

/// `scales: [u8; 16]` (low nibble scale, high nibble minimum), `qs: [u8; 64]`
/// of 2-bit quants, `d: f16`, `dmin: f16`.
pub(super) struct Q2K;

impl KQuant for Q2K {
    type Block = [u8; 84];
    const HAS_MIN: bool = true;

    #[inline(always)]
    fn blocks(row: &[u8]) -> &[Self::Block] {
        row.as_chunks().0
    }

    #[inline(always)]
    fn decode(block: &Self::Block, out: &mut Wide) {
        let (sc, rest) = block.split_at(16);
        let (qs, tail) = rest.split_at(64);
        out.d = f16_le(tail, 0);
        let dmin = f16_le(tail, 2);
        for (m, &s) in out.mins.iter_mut().zip(sc) {
            *m = dmin * f32::from(s >> 4);
        }
        // Half `h` reads `qs[32h..32h + 32]` four times, 2 bits at a time.
        let ws = out.ws.as_chunks_mut::<16>().0;
        let src = qs.as_chunks::<16>().0;
        macro_rules! pass {
            ($h:literal, $p:literal) => {
                for half in 0..2 {
                    two_bits::<{ 2 * $p }>(
                        &mut ws[8 * $h + 2 * $p + half],
                        &src[2 * $h + half],
                        i16::from(sc[8 * $h + 2 * $p + half] & 15),
                    );
                }
            };
        }
        pass!(0, 0);
        pass!(0, 1);
        pass!(0, 2);
        pass!(0, 3);
        pass!(1, 0);
        pass!(1, 1);
        pass!(1, 2);
        pass!(1, 3);
    }
}

/// `hmask: [u8; 32]`, `qs: [u8; 64]` (low 2 bits), twelve bytes of packed
/// six-bit scales, `d: f16`. The quant is the 2 bits minus 4 when its `hmask`
/// bit is clear; the scale is the six bits minus 32.
pub(super) struct Q3K;

impl KQuant for Q3K {
    type Block = [u8; 110];
    const HAS_MIN: bool = false;

    #[inline(always)]
    fn blocks(row: &[u8]) -> &[Self::Block] {
        row.as_chunks().0
    }

    #[inline(always)]
    fn decode(block: &Self::Block, out: &mut Wide) {
        const KMASK1: u32 = 0x0303_0303;
        const KMASK2: u32 = 0x0f0f_0f0f;
        let (hmask, rest) = block.split_at(32);
        let (qs, rest) = rest.split_at(64);
        let (packed, tail) = rest.split_at(12);
        out.d = f16_le(tail, 0);
        let word =
            |i: usize| u32::from_le_bytes([packed[i], packed[i + 1], packed[i + 2], packed[i + 3]]);
        let (a0, a1, tmp) = (word(0), word(4), word(8));
        let unpacked = [
            (a0 & KMASK2) | ((tmp & KMASK1) << 4),
            (a1 & KMASK2) | (((tmp >> 2) & KMASK1) << 4),
            ((a0 >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4),
            ((a1 >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4),
        ];
        let mut scales = [0i16; GROUPS];
        for (chunk, w) in scales.as_chunks_mut::<4>().0.iter_mut().zip(unpacked) {
            for (s, byte) in chunk.iter_mut().zip(w.to_le_bytes()) {
                *s = i16::from(byte) - 32;
            }
        }
        // Half `h` reads `qs[32h..32h + 32]` four times (shift `2p`), each pass
        // owning bit `4h + p` of `hmask`.
        let ws = out.ws.as_chunks_mut::<16>().0;
        let src = qs.as_chunks::<16>().0;
        let hm = hmask.as_chunks::<16>().0;
        macro_rules! pass {
            ($h:literal, $p:literal) => {
                for half in 0..2 {
                    three_bits::<{ 2 * $p }, { 1u8 << (4 * $h + $p) }>(
                        &mut ws[8 * $h + 2 * $p + half],
                        &src[2 * $h + half],
                        &hm[half],
                        scales[8 * $h + 2 * $p + half],
                    );
                }
            };
        }
        pass!(0, 0);
        pass!(0, 1);
        pass!(0, 2);
        pass!(0, 3);
        pass!(1, 0);
        pass!(1, 1);
        pass!(1, 2);
        pass!(1, 3);
    }
}

/// `d: f16`, `dmin: f16`, twelve bytes of packed scales and minimums,
/// `qs: [u8; 128]` of nibbles: for each 64 elements, the low nibbles of 32
/// bytes then their high nibbles.
pub(super) struct Q4K;

impl KQuant for Q4K {
    type Block = [u8; 144];
    const HAS_MIN: bool = true;

    #[inline(always)]
    fn blocks(row: &[u8]) -> &[Self::Block] {
        row.as_chunks().0
    }

    #[inline(always)]
    fn decode(block: &Self::Block, out: &mut Wide) {
        out.d = f16_le(block, 0);
        let (sc, mn) = scales_mins_k4(&block[4..16]);
        store_mins(f16_le(block, 2), &mn, &mut out.mins);
        let qs = &block[16..144];
        for c in 0..4 {
            let src = &qs[32 * c..32 * c + 32];
            let (lo, hi) = out.ws[64 * c..64 * c + 64].split_at_mut(32);
            for l in 0..32 {
                lo[l] = (i16::from(src[l]) & 0x0F) * sc[2 * c];
                hi[l] = (i16::from(src[l]) >> 4) * sc[2 * c + 1];
            }
        }
    }
}

/// Q4_K's layout with `qh: [u8; 32]` (the fifth bit of each element: bit `2c`
/// for the low nibbles of chunk `c`, bit `2c + 1` for the high ones) between
/// the scales and the nibbles.
pub(super) struct Q5K;

impl KQuant for Q5K {
    type Block = [u8; 176];
    const HAS_MIN: bool = true;

    #[inline(always)]
    fn blocks(row: &[u8]) -> &[Self::Block] {
        row.as_chunks().0
    }

    #[inline(always)]
    fn decode(block: &Self::Block, out: &mut Wide) {
        out.d = f16_le(block, 0);
        let (sc, mn) = scales_mins_k4(&block[4..16]);
        store_mins(f16_le(block, 2), &mn, &mut out.mins);
        let qh = &block[16..48];
        let qs = &block[48..176];
        for c in 0..4 {
            let src = &qs[32 * c..32 * c + 32];
            let (lo, hi) = out.ws[64 * c..64 * c + 64].split_at_mut(32);
            let (bit_lo, bit_hi) = (1u8 << (2 * c), 2u8 << (2 * c));
            for l in 0..32 {
                let fifth_lo = if qh[l] & bit_lo != 0 { 16 } else { 0 };
                let fifth_hi = if qh[l] & bit_hi != 0 { 16 } else { 0 };
                lo[l] = ((i16::from(src[l]) & 0x0F) + fifth_lo) * sc[2 * c];
                hi[l] = ((i16::from(src[l]) >> 4) + fifth_hi) * sc[2 * c + 1];
            }
        }
    }
}

/// `ql: [u8; 128]` (low 4 bits), `qh: [u8; 64]` (high 2 bits), `scales:
/// [i8; 16]`, `d: f16`; the quant is the six bits minus 32.
pub(super) struct Q6K;

impl KQuant for Q6K {
    type Block = [u8; 210];
    const HAS_MIN: bool = false;

    #[inline(always)]
    fn blocks(row: &[u8]) -> &[Self::Block] {
        row.as_chunks().0
    }

    #[inline(always)]
    fn decode(block: &Self::Block, out: &mut Wide) {
        out.d = f16_le(block, 208);
        let scales = &block[192..208];
        let sc = |g: usize| i16::from(scales[g].cast_signed());
        for h in 0..2 {
            let ql = &block[64 * h..64 * h + 64];
            let qh = &block[128 + 32 * h..128 + 32 * h + 32];
            let dst = &mut out.ws[128 * h..128 * h + 128];
            // Elements `l`, `l + 32`, `l + 64` and `l + 96` of the half share
            // `qh[l]`, and every sixteen of them one scale.
            for half in 0..2 {
                let g = 8 * h + half;
                let (s1, s2, s3, s4) = (sc(g), sc(g + 2), sc(g + 4), sc(g + 6));
                for i in 0..16 {
                    let l = 16 * half + i;
                    let high = i16::from(qh[l]);
                    let (lo0, lo1) = (i16::from(ql[l]), i16::from(ql[l + 32]));
                    let q1 = (lo0 & 0x0F) | ((high & 3) << 4);
                    let q2 = (lo1 & 0x0F) | (((high >> 2) & 3) << 4);
                    let q3 = (lo0 >> 4) | (((high >> 4) & 3) << 4);
                    let q4 = (lo1 >> 4) | (((high >> 6) & 3) << 4);
                    dst[l] = (q1 - 32) * s1;
                    dst[l + 32] = (q2 - 32) * s2;
                    dst[l + 64] = (q3 - 32) * s3;
                    dst[l + 96] = (q4 - 32) * s4;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use candle_core::quantized::GgmlDType;

    use super::super::tests::{dequantised, weight};
    use super::*;

    /// Every decoded element, `d * ws - dmin * min`, equals candle's
    /// dequantisation of the block.
    fn decodes_like_candle<K: KQuant>(dtype: GgmlDType) {
        let (n, k) = (7, 2 * QK_K);
        let w = weight(n, k, dtype, 31);
        let data = w.data().unwrap();
        let want = dequantised(&w);
        let row_bytes = data.len() / n;
        let mut dec = Wide::ZERO;
        for r in 0..n {
            for (b, block) in K::blocks(&data[r * row_bytes..(r + 1) * row_bytes])
                .iter()
                .enumerate()
            {
                K::decode(block, &mut dec);
                for e in 0..QK_K {
                    let term = dec.d * f32::from(dec.ws[e]);
                    let min = if K::HAS_MIN { dec.mins[e / 16] } else { 0.0 };
                    let got = term - min;
                    let expected = want[r * k + b * QK_K + e];
                    assert!(
                        (got - expected).abs() <= 1e-6 * (term.abs() + min.abs()) + 1e-9,
                        "{dtype:?} row {r} block {b} element {e}: {got} vs {expected}"
                    );
                }
            }
        }
    }

    #[test]
    fn q2_k_decodes_like_candle() {
        decodes_like_candle::<Q2K>(GgmlDType::Q2K);
    }

    #[test]
    fn q3_k_decodes_like_candle() {
        decodes_like_candle::<Q3K>(GgmlDType::Q3K);
    }

    #[test]
    fn q4_k_decodes_like_candle() {
        decodes_like_candle::<Q4K>(GgmlDType::Q4K);
    }

    #[test]
    fn q5_k_decodes_like_candle() {
        decodes_like_candle::<Q5K>(GgmlDType::Q5K);
    }

    #[test]
    fn q6_k_decodes_like_candle() {
        decodes_like_candle::<Q6K>(GgmlDType::Q6K);
    }
}
