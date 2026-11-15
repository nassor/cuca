//! Ports of ggml's reference quantizers (`quantize_row_q5_0_ref`,
//! `quantize_row_q5_1_ref`, `quantize_row_q4_K_ref`, `quantize_row_q5_K_ref`
//! and `quantize_row_q6_K_ref` in `ggml-quants.c`), operation for operation in
//! `f32`, so the blocks are the bytes `llama-quantize` writes without an
//! importance matrix.
//!
//! candle's `QTensor::quantize` is not used for the K-quants: its Q4_K and
//! Q5_K quantizers search each sub-block's scale with `make_qkx1_quants`,
//! while ggml's reference code uses `make_qkx2_quants` (weighted by the
//! sub-block's magnitude, with a wider search), and its Q6_K uses an older
//! `make_qx_quants` (three refinement rounds, where ggml scans 18 nearby
//! scales), so candle's blocks differ from ggml's on real weights (a test
//! pins this). `tensors::encode` calls these through [`quantize_row`].
//!
//! Every function takes whole blocks of values and the bytes of exactly those
//! blocks. A block's layout is ggml's `block_*` struct, little-endian:
//!
//! | type | values | bytes | layout |
//! | --- | --- | --- | --- |
//! | Q5_0 | 32 | 22 | `d` f16, `qh` u32, `qs[16]` |
//! | Q5_1 | 32 | 24 | `d` f16, `m` f16, `qh` u32, `qs[16]` |
//! | Q4_K | 256 | 144 | `d` f16, `dmin` f16, `scales[12]`, `qs[128]` |
//! | Q5_K | 256 | 176 | `d` f16, `dmin` f16, `scales[12]`, `qh[32]`, `qs[128]` |
//! | Q6_K | 256 | 210 | `ql[128]`, `qh[64]`, `scales[16]` i8, `d` f16 |

use half::f16;

/// Values in a K-quant super-block.
pub(super) const QK_K: usize = 256;

/// ggml's `GROUP_MAX_EPS`: a group whose largest magnitude is below this is
/// all zero.
const GROUP_MAX_EPS: f32 = 1e-15;

/// ggml's `nearest_int`: round to nearest, ties to even, through the
/// magic-number trick (`|v|` at most 4194303).
fn nearest_int(v: f32) -> i32 {
    let biased = v + 12_582_912.0;
    (biased.to_bits() as i32 & 0x007f_ffff) - 0x0040_0000
}

fn half(x: f32) -> u16 {
    f16::from_f32(x).to_bits()
}

fn unhalf(bits: u16) -> f32 {
    f16::from_bits(bits).to_f32()
}

/// ggml's `make_qx_quants` with `rmse_type = 1` and no weights: the scale of
/// `x.len()` values quantized to `-nmax..nmax` (stored as `nmax + level`),
/// searched over 18 nearby inverse scales. The levels go to `l`.
fn make_qx_quants(nmax: i32, x: &[f32], l: &mut [i8]) -> f32 {
    let (mut max, mut amax) = (0f32, 0f32);
    for &v in x {
        let ax = v.abs();
        if ax > amax {
            amax = ax;
            max = v;
        }
    }
    if amax < GROUP_MAX_EPS {
        l.fill(0);
        return 0.0;
    }
    let nmaxf = nmax as f32;
    let level = |iscale: f32, v: f32| nearest_int(iscale * v).clamp(-nmax, nmax - 1);
    let mut iscale = -nmaxf / max;
    let (mut sumlx, mut suml2) = (0f32, 0f32);
    for (q, &v) in l.iter_mut().zip(x) {
        let lv = level(iscale, v);
        *q = (lv + nmax) as i8;
        let w = v * v;
        sumlx += w * v * lv as f32;
        suml2 += w * lv as f32 * lv as f32;
    }
    let mut scale = if suml2 != 0.0 { sumlx / suml2 } else { 0.0 };
    let mut best = scale * sumlx;
    for is in -9..=9i32 {
        if is == 0 {
            continue;
        }
        iscale = -(nmaxf + 0.1f32 * is as f32) / max;
        sumlx = 0.0;
        suml2 = 0.0;
        for &v in x {
            let lv = level(iscale, v);
            let w = v * v;
            sumlx += w * v * lv as f32;
            suml2 += w * lv as f32 * lv as f32;
        }
        if suml2 > 0.0 && sumlx * sumlx > best * suml2 {
            for (q, &v) in l.iter_mut().zip(x) {
                *q = (nmax + level(iscale, v)) as i8;
            }
            scale = sumlx / suml2;
            best = scale * sumlx;
        }
    }
    scale
}

/// ggml's `make_qkx2_quants` with `use_mad = false` and `rdelta = 0.1` (the
/// value both callers pass): the scale and minimum of 32 values quantized to
/// `0..=nmax`, searched over `nstep + 1` inverse scales starting at
/// `nmax + rmin`, each step 0.1 wider. Returns `(scale, -min)`; the levels go
/// to `l`, `laux` is scratch.
fn make_qkx2_quants(
    nmax: i32,
    x: &[f32; 32],
    weights: &[f32; 32],
    l: &mut [u8],
    laux: &mut [u8; 32],
    (rmin, nstep): (f32, i32),
) -> (f32, f32) {
    let mut min = x[0];
    let mut max = x[0];
    let mut sum_w = weights[0];
    let mut sum_x = sum_w * x[0];
    for i in 1..32 {
        if x[i] < min {
            min = x[i];
        }
        if x[i] > max {
            max = x[i];
        }
        let w = weights[i];
        sum_w += w;
        sum_x += w * x[i];
    }
    if min > 0.0 {
        min = 0.0;
    }
    if max == min {
        l.fill(0);
        return (0.0, -min);
    }
    let nmaxf = nmax as f32;
    let mut iscale = nmaxf / (max - min);
    let mut scale = 1.0 / iscale;
    let mut best_error = 0f32;
    for i in 0..32 {
        let lv = nearest_int(iscale * (x[i] - min)).clamp(0, nmax);
        l[i] = lv as u8;
        let diff = scale * lv as f32 + min - x[i];
        best_error += weights[i] * (diff * diff);
    }
    if nstep < 1 {
        return (scale, -min);
    }
    for is in 0..=nstep {
        iscale = (rmin + 0.1f32 * is as f32 + nmaxf) / (max - min);
        let (mut sum_l, mut sum_l2, mut sum_xl) = (0f32, 0f32, 0f32);
        for i in 0..32 {
            let lv = nearest_int(iscale * (x[i] - min)).clamp(0, nmax);
            laux[i] = lv as u8;
            let w = weights[i];
            let lf = lv as f32;
            sum_l += w * lf;
            sum_l2 += w * lf * lf;
            sum_xl += w * lf * x[i];
        }
        let d = sum_w * sum_l2 - sum_l * sum_l;
        if d > 0.0 {
            let mut this_scale = (sum_w * sum_xl - sum_x * sum_l) / d;
            let mut this_min = (sum_l2 * sum_x - sum_l * sum_xl) / d;
            if this_min > 0.0 {
                this_min = 0.0;
                this_scale = sum_xl / sum_l2;
            }
            let mut cur_error = 0f32;
            for i in 0..32 {
                let diff = this_scale * laux[i] as f32 + this_min - x[i];
                cur_error += weights[i] * (diff * diff);
            }
            if cur_error < best_error {
                l.copy_from_slice(&laux[..]);
                best_error = cur_error;
                scale = this_scale;
                min = this_min;
            }
        }
    }
    (scale, -min)
}

/// ggml's `get_scale_min_k4`: sub-block `j`'s 6-bit scale and minimum out of
/// the 12 packed bytes.
fn scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

/// The part Q4_K and Q5_K share: per 32-value sub-block, scale and minimum
/// (`make_qkx2_quants` weighted by `av_x + |x|`), the 6-bit packing of both
/// into `scales`, the two f16 super-scales, and the final levels `0..=nmax`
/// of every value against the packed (rounded) scales. Returns `(d, dmin)` as
/// f16 bits.
fn quantize_k4(
    x: &[f32; QK_K],
    nmax: i32,
    search: (f32, i32),
    scales_out: &mut [u8],
    levels: &mut [u8; QK_K],
) -> (u16, u16) {
    let mut laux = [0u8; 32];
    let mut weights = [0f32; 32];
    let mut mins = [0f32; QK_K / 32];
    let mut scales = [0f32; QK_K / 32];
    let (mut max_scale, mut max_min) = (0f32, 0f32);
    for (j, (xs, ls)) in x
        .as_chunks::<32>()
        .0
        .iter()
        .zip(levels.as_chunks_mut::<32>().0.iter_mut())
        .enumerate()
    {
        let mut sum_x2 = 0f32;
        for &v in xs {
            sum_x2 += v * v;
        }
        let av_x = (sum_x2 / 32.0).sqrt();
        for (w, &v) in weights.iter_mut().zip(xs) {
            *w = av_x + v.abs();
        }
        (scales[j], mins[j]) = make_qkx2_quants(nmax, xs, &weights, ls, &mut laux, search);
        if scales[j] > max_scale {
            max_scale = scales[j];
        }
        if mins[j] > max_min {
            max_min = mins[j];
        }
    }
    let inv_scale = if max_scale > 0.0 {
        63.0 / max_scale
    } else {
        0.0
    };
    let inv_min = if max_min > 0.0 { 63.0 / max_min } else { 0.0 };
    scales_out.fill(0);
    for j in 0..QK_K / 32 {
        let ls = (nearest_int(inv_scale * scales[j]) as u8).min(63);
        let lm = (nearest_int(inv_min * mins[j]) as u8).min(63);
        if j < 4 {
            scales_out[j] = ls;
            scales_out[j + 4] = lm;
        } else {
            scales_out[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
            scales_out[j - 4] |= (ls >> 4) << 6;
            scales_out[j] |= (lm >> 4) << 6;
        }
    }
    let d = half(max_scale / 63.0);
    let dmin = half(max_min / 63.0);
    for j in 0..QK_K / 32 {
        let (sc, m) = scale_min_k4(j, scales_out);
        let dd = unhalf(d) * sc as f32;
        if dd == 0.0 {
            continue;
        }
        let dm = unhalf(dmin) * m as f32;
        for ii in 0..32 {
            let l = nearest_int((x[32 * j + ii] + dm) / dd);
            levels[32 * j + ii] = l.clamp(0, nmax) as u8;
        }
    }
    (d, dmin)
}

/// `quantize_row_q4_K_ref` for one block of 256 values into 144 bytes.
fn q4_k_block(x: &[f32; QK_K], out: &mut [u8; 144]) {
    let mut levels = [0u8; QK_K];
    let (head, rest) = out.split_at_mut(4);
    let (scales, qs) = rest.split_at_mut(12);
    let (d, dmin) = quantize_k4(x, 15, (-1.0, 20), scales, &mut levels);
    head[..2].copy_from_slice(&d.to_le_bytes());
    head[2..].copy_from_slice(&dmin.to_le_bytes());
    for (q, l) in qs
        .as_chunks_mut::<32>()
        .0
        .iter_mut()
        .zip(levels.as_chunks::<64>().0)
    {
        for (i, b) in q.iter_mut().enumerate() {
            *b = l[i] | (l[i + 32] << 4);
        }
    }
}

/// `quantize_row_q5_K_ref` for one block of 256 values into 176 bytes.
fn q5_k_block(x: &[f32; QK_K], out: &mut [u8; 176]) {
    let mut levels = [0u8; QK_K];
    let (head, rest) = out.split_at_mut(4);
    let (scales, rest) = rest.split_at_mut(12);
    let (qh, qs) = rest.split_at_mut(32);
    let (d, dmin) = quantize_k4(x, 31, (-0.5, 15), scales, &mut levels);
    head[..2].copy_from_slice(&d.to_le_bytes());
    head[2..].copy_from_slice(&dmin.to_le_bytes());
    qh.fill(0);
    let (mut m1, mut m2) = (1u8, 2u8);
    for (ql, l) in qs
        .as_chunks_mut::<32>()
        .0
        .iter_mut()
        .zip(levels.as_chunks::<64>().0)
    {
        for j in 0..32 {
            let mut l1 = l[j];
            if l1 > 15 {
                l1 -= 16;
                qh[j] |= m1;
            }
            let mut l2 = l[j + 32];
            if l2 > 15 {
                l2 -= 16;
                qh[j] |= m2;
            }
            ql[j] = l1 | (l2 << 4);
        }
        m1 <<= 2;
        m2 <<= 2;
    }
}

/// `quantize_row_q6_K_ref` for one block of 256 values into 210 bytes.
fn q6_k_block(x: &[f32; QK_K], out: &mut [u8; 210]) {
    let mut levels = [0i8; QK_K];
    let mut scales = [0f32; QK_K / 16];
    let (mut max_scale, mut max_abs_scale) = (0f32, 0f32);
    let xs_chunks = x.as_chunks::<16>().0.iter();
    for (ib, (xs, ls)) in xs_chunks
        .zip(levels.as_chunks_mut::<16>().0.iter_mut())
        .enumerate()
    {
        let scale = make_qx_quants(32, xs, ls);
        scales[ib] = scale;
        if scale.abs() > max_abs_scale {
            max_abs_scale = scale.abs();
            max_scale = scale;
        }
    }
    out.fill(0);
    if max_abs_scale < GROUP_MAX_EPS {
        return;
    }
    let (ql, rest) = out.split_at_mut(128);
    let (qh, rest) = rest.split_at_mut(64);
    let (sc_out, d_out) = rest.split_at_mut(16);
    let iscale = -128.0 / max_scale;
    let d = half(1.0 / iscale);
    d_out.copy_from_slice(&d.to_le_bytes());
    let mut sc = [0i8; QK_K / 16];
    for ib in 0..QK_K / 16 {
        sc[ib] = nearest_int(iscale * scales[ib]).min(127) as i8;
        sc_out[ib] = sc[ib] as u8;
    }
    for j in 0..QK_K / 16 {
        let dd = unhalf(d) * sc[j] as f32;
        if dd == 0.0 {
            continue;
        }
        for ii in 0..16 {
            let l = nearest_int(x[16 * j + ii] / dd).clamp(-32, 31);
            levels[16 * j + ii] = (l + 32) as i8;
        }
    }
    let ql_chunks = ql.as_chunks_mut::<64>().0.iter_mut();
    let qh_chunks = qh.as_chunks_mut::<32>().0.iter_mut();
    for ((ql, qh), l) in ql_chunks.zip(qh_chunks).zip(levels.as_chunks::<128>().0) {
        for i in 0..32 {
            let (q1, q2, q3, q4) = (
                l[i] as u8,
                l[i + 32] as u8,
                l[i + 64] as u8,
                l[i + 96] as u8,
            );
            ql[i] = (q1 & 0xF) | ((q3 & 0xF) << 4);
            ql[i + 32] = (q2 & 0xF) | ((q4 & 0xF) << 4);
            qh[i] = (q1 >> 4) | ((q2 >> 4) << 2) | ((q3 >> 4) << 4) | ((q4 >> 4) << 6);
        }
    }
}

/// `quantize_row_q5_0_ref` for one block of 32 values into 22 bytes.
fn q5_0_block(x: &[f32; 32], out: &mut [u8; 22]) {
    let (mut amax, mut max) = (0f32, 0f32);
    for &v in x {
        if amax < v.abs() {
            amax = v.abs();
            max = v;
        }
    }
    let d = max / -16.0;
    let id = if d != 0.0 { 1.0 / d } else { 0.0 };
    out[..2].copy_from_slice(&half(d).to_le_bytes());
    let mut qh = 0u32;
    for j in 0..16 {
        // C: `MIN(31, (int8_t)(x0 + 16.5f))` compares as `int`, then narrows.
        let xi0 = i32::min(31, (x[j] * id + 16.5) as i8 as i32) as u8;
        let xi1 = i32::min(31, (x[j + 16] * id + 16.5) as i8 as i32) as u8;
        out[6 + j] = (xi0 & 0x0F) | ((xi1 & 0x0F) << 4);
        qh |= u32::from((xi0 & 0x10) >> 4) << j;
        qh |= u32::from((xi1 & 0x10) >> 4) << (j + 16);
    }
    out[2..6].copy_from_slice(&qh.to_le_bytes());
}

/// `quantize_row_q5_1_ref` for one block of 32 values into 24 bytes.
fn q5_1_block(x: &[f32; 32], out: &mut [u8; 24]) {
    let (mut min, mut max) = (f32::MAX, -f32::MAX);
    for &v in x {
        if v < min {
            min = v;
        }
        if v > max {
            max = v;
        }
    }
    let d = (max - min) / 31.0;
    let id = if d != 0.0 { 1.0 / d } else { 0.0 };
    out[..2].copy_from_slice(&half(d).to_le_bytes());
    out[2..4].copy_from_slice(&half(min).to_le_bytes());
    let mut qh = 0u32;
    for j in 0..16 {
        let xi0 = ((x[j] - min) * id + 0.5) as u8;
        let xi1 = ((x[j + 16] - min) * id + 0.5) as u8;
        out[8 + j] = (xi0 & 0x0F) | ((xi1 & 0x0F) << 4);
        qh |= u32::from((xi0 & 0x10) >> 4) << j;
        qh |= u32::from((xi1 & 0x10) >> 4) << (j + 16);
    }
    out[4..8].copy_from_slice(&qh.to_le_bytes());
}

/// The reference quantizers this module ports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Kind {
    Q5_0,
    Q5_1,
    Q4K,
    Q5K,
    Q6K,
}

/// Quantize whole blocks of `src` into `dst`, which holds exactly their bytes
/// (a multiple of 32 values for Q5_0 and Q5_1, of 256 for the K-quants).
pub(super) fn quantize_row(kind: Kind, src: &[f32], dst: &mut [u8]) {
    match kind {
        Kind::Q5_0 => {
            for (d, x) in dst
                .as_chunks_mut::<22>()
                .0
                .iter_mut()
                .zip(src.as_chunks::<32>().0)
            {
                q5_0_block(x, d);
            }
        }
        Kind::Q5_1 => {
            for (d, x) in dst
                .as_chunks_mut::<24>()
                .0
                .iter_mut()
                .zip(src.as_chunks::<32>().0)
            {
                q5_1_block(x, d);
            }
        }
        Kind::Q4K => {
            for (d, x) in dst
                .as_chunks_mut::<144>()
                .0
                .iter_mut()
                .zip(src.as_chunks::<QK_K>().0)
            {
                q4_k_block(x, d);
            }
        }
        Kind::Q5K => {
            for (d, x) in dst
                .as_chunks_mut::<176>()
                .0
                .iter_mut()
                .zip(src.as_chunks::<QK_K>().0)
            {
                q5_k_block(x, d);
            }
        }
        Kind::Q6K => {
            for (d, x) in dst
                .as_chunks_mut::<210>()
                .0
                .iter_mut()
                .zip(src.as_chunks::<QK_K>().0)
            {
                q6_k_block(x, d);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 16 blocks of 256 values (zero, constant, positive-only, heavy-tailed,
    /// an outlier, mixed-scale sub-blocks, below `GROUP_MAX_EPS`, ...).
    const INPUT: &[u8] = include_bytes!("../testdata/kquant/input.f32");

    fn input() -> Vec<f32> {
        INPUT
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect()
    }

    /// Each type's bytes against what ggml's `quantize_row_<type>_ref` wrote
    /// (`testdata/kquant/make_fixture.py`, llama.cpp commit 6d3568e), block by
    /// block so a failure names the block.
    #[test]
    fn blocks_equal_llama_cpps_reference_quantizers() {
        let x = input();
        let cases: [(Kind, &[u8], usize, usize); 5] = [
            (
                Kind::Q5_0,
                include_bytes!("../testdata/kquant/q5_0.bin"),
                32,
                22,
            ),
            (
                Kind::Q5_1,
                include_bytes!("../testdata/kquant/q5_1.bin"),
                32,
                24,
            ),
            (
                Kind::Q4K,
                include_bytes!("../testdata/kquant/q4_K.bin"),
                QK_K,
                144,
            ),
            (
                Kind::Q5K,
                include_bytes!("../testdata/kquant/q5_K.bin"),
                QK_K,
                176,
            ),
            (
                Kind::Q6K,
                include_bytes!("../testdata/kquant/q6_K.bin"),
                QK_K,
                210,
            ),
        ];
        for (kind, want, block, bytes) in cases {
            let mut got = vec![0u8; x.len() / block * bytes];
            quantize_row(kind, &x, &mut got);
            assert_eq!(got.len(), want.len(), "{kind:?} size");
            for (i, (g, w)) in got.chunks(bytes).zip(want.chunks(bytes)).enumerate() {
                assert!(g == w, "{kind:?} block {i}: {g:02x?} != {w:02x?}");
            }
        }
    }

    #[test]
    fn nearest_int_rounds_half_to_even() {
        let got: Vec<i32> = [0.5, 1.5, 2.5, -0.5, -1.5, 2.4999, -2.51, 4_194_303.0]
            .into_iter()
            .map(nearest_int)
            .collect();
        assert_eq!(got, [0, 2, 2, 0, -2, 2, -3, 4_194_303]);
    }

    /// candle 0.11.1's `QTensor::quantize` does not write ggml's bytes for Q4_K,
    /// Q5_K or Q6_K on this input (its Q4_K and Q5_K search each sub-block with
    /// `make_qkx1_quants`, ggml with `make_qkx2_quants`; its Q6_K scale search
    /// is an older `make_qx_quants` with three refinement rounds where ggml
    /// scans 18 nearby scales, and it divides by zero on an all-zero block),
    /// which is why the converter ports ggml's reference code instead of
    /// calling candle. The exactly pinned candle version makes this a fact
    /// about that version.
    #[test]
    fn candle_does_not_write_ggmls_bytes_for_the_k_quants() {
        use candle_core::quantized::{GgmlDType, QTensor};
        use candle_core::{Device, Tensor};

        let x = input();
        for (kind, dtype, bytes) in [
            (Kind::Q4K, GgmlDType::Q4K, 144),
            (Kind::Q5K, GgmlDType::Q5K, 176),
            (Kind::Q6K, GgmlDType::Q6K, 210),
        ] {
            let mut ours = vec![0u8; x.len() / QK_K * bytes];
            quantize_row(kind, &x, &mut ours);
            let t = Tensor::from_vec(x.clone(), (1, x.len()), &Device::Cpu).unwrap();
            let q = QTensor::quantize(&t, dtype).unwrap();
            let theirs = q.data().unwrap();
            let differing = ours
                .chunks(bytes)
                .zip(theirs.chunks(bytes))
                .filter(|(a, b)| a != b)
                .count();
            assert!(differing > 0, "{kind:?}: candle now equals ggml");
        }
    }
}
