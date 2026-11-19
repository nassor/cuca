use super::*;
use crate::provider::local::model::quant::dequant_row;
use crate::provider::local::model::tiny::{fix_scales, mxfp4_bytes, random_blocks};
use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::{Device, Module, Tensor};

/// Every type the kernel serves: the legacy blocks, the K-quants, the
/// I-quants, the ternary types, MXFP4 and NVFP4.
fn types() -> Vec<GgufType> {
    use GgmlDType as G;
    [
        G::Q4_0,
        G::Q4_1,
        G::Q5_0,
        G::Q5_1,
        G::Q8_0,
        G::Q2K,
        G::Q3K,
        G::Q4K,
        G::Q5K,
        G::Q6K,
        G::Q8K,
    ]
    .into_iter()
    .map(GgufType::Ggml)
    .chain([
        GgufType::Iq2Xxs,
        GgufType::Iq2Xs,
        GgufType::Iq2S,
        GgufType::Iq3Xxs,
        GgufType::Iq3S,
        GgufType::Iq1S,
        GgufType::Iq1M,
        GgufType::Iq4Nl,
        GgufType::Iq4Xs,
        GgufType::Tq1_0,
        GgufType::Tq2_0,
        GgufType::Mxfp4,
        GgufType::Nvfp4,
    ])
    .collect()
}

/// The types of [`types`] that bring minima (their sums are tested apart).
fn has_min(ty: GgufType) -> bool {
    Kind::of(ty).is_some_and(|k| with_format!(k, F => F::MIN != MIN_NONE))
}

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

/// `n` rows of `k` values of type `ty` as random but valid blocks (scales of
/// 0.004 to 0.012).
fn random_bytes(ty: GgufType, n: usize, k: usize, seed: u64) -> Vec<u8> {
    random_blocks(ty, n, k, seed, 0.004)
}

/// `n` rows of `k` noise values quantized to `ty`: through candle where it
/// implements the type, from the E2M1 grid for MXFP4, else random blocks.
fn weight_bytes(ty: GgufType, n: usize, k: usize, seed: u64) -> Vec<u8> {
    match ty {
        GgufType::Ggml(dtype) => {
            let t = Tensor::from_vec(noise(n * k, seed, 0.5), (n, k), &Device::Cpu).unwrap();
            QTensor::quantize(&t, dtype)
                .unwrap()
                .data()
                .unwrap()
                .to_vec()
        }
        GgufType::Mxfp4 => mxfp4_bytes(&noise(n * k, seed, 3.0)),
        _ => random_bytes(ty, n, k, seed),
    }
}

/// The dequantised weights of `bytes`, row-major (the foundation's decoders,
/// which are ggml's).
fn dequantised(ty: GgufType, bytes: &[u8], n: usize, k: usize) -> Vec<f32> {
    let row = ty.byte_len(k).unwrap();
    let mut out = vec![0f32; n * k];
    for (src, dst) in bytes.chunks_exact(row).zip(out.chunks_exact_mut(k)) {
        dequant_row(ty, src, 1.0, dst).unwrap();
    }
    out
}

/// A weight under test: its bytes, kernel view and dequantised values.
struct Case {
    ty: GgufType,
    bytes: Vec<u8>,
    q: QmmWeight,
    n: usize,
    k: usize,
}

impl Case {
    fn of(ty: GgufType, bytes: Vec<u8>, n: usize, k: usize) -> Self {
        let q = QmmWeight::from_raw(ty, Arc::from(&bytes[..]), n, k).unwrap();
        Self { ty, bytes, q, n, k }
    }

    fn new(ty: GgufType, n: usize, k: usize, seed: u64) -> Self {
        Self::of(ty, weight_bytes(ty, n, k, seed), n, k)
    }

    fn random(ty: GgufType, n: usize, k: usize, seed: u64) -> Self {
        Self::of(ty, random_bytes(ty, n, k, seed), n, k)
    }

    fn values(&self) -> Vec<f32> {
        dequantised(self.ty, &self.bytes, self.n, self.k)
    }
}

/// `(n, k, m)` shapes whose rows are whole blocks: tails in both tilings, and
/// 600 rows exceeding `SERIAL_BELOW` so the pooled path runs. Rows of 256
/// values hold whole blocks of every type; the others add 32-value blocks.
fn shapes(ty: GgufType) -> Vec<(usize, usize, usize)> {
    let mut s = vec![(10, 512, 7), (3, 256, 1), (600, 256, 5), (64, 768, 17)];
    if ty.block_size() == Some(QK) {
        s.extend([(10, 2 * QK, 7), (3, QK, 1), (64, 3 * QK, 17)]);
    }
    s
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
/// product of the scales entry by entry, reduced by [`hsum`], plus the sum of
/// the minima terms.
fn scalar_definition<F: Format>(data: &[u8], act: &Acts, n: usize) -> Vec<f32> {
    let row = row_bytes::<F>(act.nb * QK);
    let per = F::VALUES / QK;
    let mut out = vec![0f32; act.m * n];
    for j in 0..act.m {
        for r in 0..n {
            let mut lanes = [0f32; LANES];
            let mut extra = 0f32;
            for (i, block) in F::blocks(&data[r * row..(r + 1) * row]).iter().enumerate() {
                let mut e = [Entry::default(); MAX_ENTRIES];
                F::entries(block, &mut e[..F::ENTRIES]);
                for (t, en) in e[..F::ENTRIES].iter().enumerate() {
                    let b = i * per + t / F::SUB;
                    let s = en.scale * act.scales[b * act.m + j];
                    for (l, lane) in lanes.iter_mut().enumerate() {
                        let p = i32::from(en.halves[0][l]) * i32::from(act_quant(act, b, j, l))
                            + i32::from(en.halves[1][l])
                                * i32::from(act_quant(act, b, j, l + LANES));
                        *lane += p as f32 * s;
                    }
                    if t % F::SUB == F::SUB - 1 {
                        if F::MIN == MIN_Q8_1 {
                            extra += en.mins[0] * act.sums[b * act.m + j];
                        } else if F::MIN == MIN_HALVES {
                            let h = act.halves[b * act.m + j];
                            extra += en.mins[0] * h[0];
                            extra += en.mins[1] * h[1];
                        }
                    }
                }
            }
            out[j * n + r] = if F::MIN == MIN_NONE {
                hsum(&lanes)
            } else {
                hsum(&lanes) + extra
            };
        }
    }
    out
}

/// [`scalar_definition`] for the format `q` holds.
fn scalar_forward(q: &QmmWeight, x: &[f32], m: usize) -> Vec<f32> {
    with_format!(q.kind, F => scalar_definition::<F>(&q.data, &Acts::new(x, m, q.k, F::MIN), q.n))
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
fn the_kernel_equals_the_scalar_definition_bit_for_bit() {
    for ty in types() {
        for (n, k, m) in [(10, 512, 7), (3, 256, 1), (130, 512, 41), (9, 1024, 2)] {
            let c = Case::new(ty, n, k, 11);
            let x = awkward_acts(m, k);
            let got = c.q.forward(&x, m).unwrap();
            let want = scalar_forward(&c.q, &x, m);
            for (i, (g, r)) in got.iter().zip(&want).enumerate() {
                assert_eq!(
                    g.to_bits(),
                    r.to_bits(),
                    "{} {n}x{k} m={m} output {i}: {g} vs {r}",
                    ty.name()
                );
            }
        }
        if ty.block_size() == Some(QK) {
            for (n, k, m) in [(10, 2 * QK, 7), (3, 3 * QK, 1), (37, 4 * QK, 6)] {
                let c = Case::new(ty, n, k, 11);
                let x = awkward_acts(m, k);
                assert_eq!(
                    c.q.forward(&x, m).unwrap(),
                    scalar_forward(&c.q, &x, m),
                    "{} {n}x{k} m={m}",
                    ty.name()
                );
            }
        }
    }
}

/// [`QmmWeight::forward_indexed`] equals selecting those columns from the full
/// [`QmmWeight::forward`] bit for bit, across every format and index order
/// (including duplicates).
#[test]
fn forward_indexed_equals_selected_columns_of_forward_bit_for_bit() {
    let rows = [9u32, 0, 5, 5, 12];
    for ty in types() {
        let k = 512;
        let c = Case::new(ty, 13, k, 17);
        for m in [1usize, 3] {
            let x = awkward_acts(m, k);
            let full = c.q.forward(&x, m).unwrap();
            let got = c.q.forward_indexed(&x, m, &rows).unwrap();
            assert_eq!(got.len(), m * rows.len());
            for j in 0..m {
                for (p, &r) in rows.iter().enumerate() {
                    let g = got[j * rows.len() + p];
                    let w = full[j * c.q.n() + r as usize];
                    assert_eq!(
                        g.to_bits(),
                        w.to_bits(),
                        "{} m={m} j={j} row={r}: {g} vs {w}",
                        ty.name()
                    );
                }
            }
        }
        assert!(
            c.q.forward_indexed(&awkward_acts(1, k), 1, &[])
                .unwrap()
                .is_empty()
        );
        assert!(c.q.forward_indexed(&awkward_acts(1, k), 1, &[13]).is_err());
    }
}

/// Every format's vector decode gives the entries of its scalar definition
/// (quants, scales and minima, bit for bit), on random blocks.
#[test]
fn decode_equals_halves() {
    for ty in types() {
        let kind = Kind::of(ty).unwrap();
        with_format!(kind, F => check_decode::<F>(&ty.name(), ty, 64));
    }
}

/// The grid I-quants and the ternary types decode through table lookups and
/// bit fields: thousands of random blocks reach every grid index, sign byte,
/// high-bit combination and ternary byte, and still equal the scalar entries.
#[test]
fn grid_and_ternary_decodes_equal_the_entries_over_whole_tables() {
    for ty in [
        GgufType::Iq2Xxs,
        GgufType::Iq2Xs,
        GgufType::Iq2S,
        GgufType::Iq3Xxs,
        GgufType::Iq3S,
        GgufType::Iq1S,
        GgufType::Iq1M,
        GgufType::Tq1_0,
        GgufType::Tq2_0,
    ] {
        let kind = Kind::of(ty).unwrap();
        with_format!(kind, F => check_decode::<F>(&ty.name(), ty, 4096));
    }
}

/// [`Format::decode`] equals [`Format::entries`] on `rows` rows of four random
/// blocks.
fn check_decode<F: Format>(name: &str, ty: GgufType, rows: usize) {
    let bytes = random_bytes(ty, rows, F::VALUES * 4, 7);
    for (i, block) in F::blocks(&bytes).iter().enumerate() {
        let mut e = [Entry::default(); MAX_ENTRIES];
        F::entries(block, &mut e[..F::ENTRIES]);
        let mut quants = [Quants::default(); MAX_ENTRIES];
        let mut scales = [0f32; MAX_ENTRIES];
        let mut mins = [[0f32; 2]; MAX_ENTRIES];
        let per = F::VALUES / QK;
        F::decode(
            block,
            &mut Dst {
                quants: &mut quants[..F::ENTRIES],
                scales: &mut scales[..F::ENTRIES],
                mins: if F::MIN != MIN_NONE {
                    &mut mins[..per]
                } else {
                    &mut []
                },
            },
        );
        for (t, en) in e[..F::ENTRIES].iter().enumerate() {
            assert_eq!(
                quants[t].to_array(),
                interleave(&en.halves).to_array(),
                "{name} block {i} entry {t}: quants"
            );
            assert_eq!(
                scales[t].to_bits(),
                en.scale.to_bits(),
                "{name} block {i} entry {t}: scale"
            );
            if F::MIN != MIN_NONE {
                let want = if F::MIN == MIN_Q8_1 {
                    [en.mins[0], 0.0]
                } else {
                    en.mins
                };
                let got = mins[t];
                assert_eq!(
                    [got[0].to_bits(), got[1].to_bits()],
                    [want[0].to_bits(), want[1].to_bits()],
                    "{name} block {i} entry {t}: minima"
                );
            }
        }
    }
}

/// The entries are the foundation's decode: `scale * quant + min` per value
/// equals `dequant_row`, for every format and random blocks.
#[test]
fn the_entries_reproduce_the_dequantised_weights() {
    fn check<F: Format>(name: &str, ty: GgufType) {
        let k = F::VALUES * 3;
        let bytes = random_bytes(ty, 5, k, 21);
        let want = dequantised(ty, &bytes, 5, k);
        for (r, row) in bytes.chunks_exact(ty.byte_len(k).unwrap()).enumerate() {
            for (i, block) in F::blocks(row).iter().enumerate() {
                let mut e = [Entry::default(); MAX_ENTRIES];
                F::entries(block, &mut e[..F::ENTRIES]);
                for (t, en) in e[..F::ENTRIES].iter().enumerate() {
                    let v0 = r * k + i * F::VALUES + QK * (t / F::SUB);
                    for l in 0..LANES {
                        for (h, off) in [(0, 0), (1, LANES)] {
                            let min = if F::MIN == MIN_Q8_1 {
                                en.mins[0]
                            } else if F::MIN == MIN_HALVES {
                                en.mins[h]
                            } else {
                                0.0
                            };
                            let got = en.scale * f32::from(en.halves[h][l]) + min;
                            let expect = want[v0 + off + l];
                            // NVFP4 entries hold one 16-value group each: the
                            // other half is zero and belongs to the sibling.
                            if F::SUB == 2 && en.halves[h].iter().all(|&q| q == 0) {
                                continue;
                            }
                            assert!(
                                (got - expect).abs() <= 1e-6 * expect.abs().max(1e-3),
                                "{name} row {r} block {i} entry {t} value {}: {got} vs {expect}",
                                off + l
                            );
                        }
                    }
                }
            }
        }
    }
    for ty in types() {
        let kind = Kind::of(ty).unwrap();
        with_format!(kind, F => check::<F>(&ty.name(), ty));
    }
}

/// The activations as the kernel sees them: quantised and dequantised.
fn rounded_acts(x: &[f32], m: usize, k: usize) -> Vec<f64> {
    let mut xd = vec![0f64; m * k];
    let act = Acts::new(x, m, k, MIN_NONE);
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

/// Reference in f64: dequantised weights times dequantised (rounded)
/// activations.
fn reference(c: &Case, x: &[f32], m: usize) -> Vec<f64> {
    let wd = c.values();
    let xd = rounded_acts(x, m, c.k);
    let mut out = vec![0f64; m * c.n];
    for j in 0..m {
        for r in 0..c.n {
            out[j * c.n + r] = (0..c.k)
                .map(|e| f64::from(wd[r * c.k + e]) * xd[j * c.k + e])
                .sum();
        }
    }
    out
}

/// Per output, the most the Q8_1 sum term's f16 rounding can move a Q4_1 or
/// Q5_1 result: `|min| * |s|` per block, at 2^-10 (two f16 roundings of the
/// sum term `s`). Zero for every other type, whose sums are exact.
fn sum_term_slack(c: &Case, x: &[f32], m: usize) -> Vec<f64> {
    let mut slack = vec![0f64; m * c.n];
    if !matches!(c.ty, GgufType::Ggml(GgmlDType::Q4_1 | GgmlDType::Q5_1)) {
        return slack;
    }
    let act = Acts::new(x, m, c.k, MIN_Q8_1);
    let block = c.ty.type_size().unwrap();
    for j in 0..m {
        for r in 0..c.n {
            slack[j * c.n + r] = (0..act.nb)
                .map(|b| {
                    let at = (r * act.nb + b) * block;
                    let min = f64::from(f16_le(&c.bytes[at..at + block], 2));
                    min.abs() * f64::from(act.sums[b * act.m + j]).abs() * 1.0e-3
                })
                .sum();
        }
    }
    slack
}

/// The most the rounding of the activations to blocks of `block` values can
/// move a dot of the weight row `w` with `x`: at most half a quant step (plus
/// the f16 rounding of the scale, 2^-11 of 127 steps) per element,
/// `0.6 * amax / 127 * sum |w|` per block.
fn rounding_bound(w: &[f32], x: &[f32], block: usize) -> f64 {
    x.chunks(block)
        .zip(w.chunks(block))
        .map(|(xs, ws)| {
            let amax = xs.iter().fold(0f32, |a, v| a.max(v.abs()));
            let abs_w: f64 = ws.iter().map(|v| f64::from(v.abs())).sum();
            0.6 * f64::from(amax) / 127.0 * abs_w
        })
        .sum()
}

#[test]
fn matches_the_f64_reference_and_candle() {
    for ty in types() {
        for (n, k, m) in shapes(ty) {
            let c = Case::new(ty, n, k, 11);
            let x = noise(m * k, 5, 1.0);
            let got = c.q.forward(&x, m).unwrap();
            let want = reference(&c, &x, m);
            let slack = sum_term_slack(&c, &x, m);
            let scale = want.iter().fold(0f64, |a, v| a.max(v.abs()));
            for ((g, r), s) in got.iter().zip(&want).zip(&slack) {
                assert!(
                    (f64::from(*g) - r).abs() <= 1e-4 * scale + s,
                    "{} {n}x{k} m={m}: {g} vs {r}",
                    ty.name()
                );
            }
            let GgufType::Ggml(dtype) = ty else { continue };
            // candle's Q4_1, Q5_0 and Q5_1 dots assert an even block count,
            // its Q8_K matmul does not exist.
            if (k / QK) % 2 == 1
                && matches!(dtype, GgmlDType::Q4_1 | GgmlDType::Q5_0 | GgmlDType::Q5_1)
            {
                continue;
            }
            if matches!(dtype, GgmlDType::Q8K) {
                continue;
            }
            // candle pairs the K-quants with Q8_K activations of 256 values
            // and rounds the others' its own way: each kernel is within its
            // activation rounding of the exact product, so they agree within
            // the sum of the two bounds. The f64 reference above is the exact
            // check; this catches a layout or scale mistake against an
            // independent kernel.
            let candle_block = if ty.block_size() == Some(256) {
                256
            } else {
                QK
            };
            let w = Arc::new(
                crate::provider::local::gguf::qtensor_from_bytes(
                    dtype,
                    &[n, k],
                    &c.bytes,
                    &Device::Cpu,
                )
                .unwrap(),
            );
            let xt = Tensor::from_vec(x.clone(), (m, k), &Device::Cpu).unwrap();
            let candle = QMatMul::from_arc(w)
                .unwrap()
                .forward(&xt)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let wd = c.values();
            for j in 0..m {
                for r in 0..n {
                    let (xs, ws) = (&x[j * k..(j + 1) * k], &wd[r * k..(r + 1) * k]);
                    let tol = rounding_bound(ws, xs, QK)
                        + rounding_bound(ws, xs, candle_block)
                        + slack[j * n + r]
                        + 1e-4 * scale;
                    let (g, cd) = (got[j * n + r], candle[j * n + r]);
                    assert!(
                        (f64::from(g) - f64::from(cd)).abs() <= tol,
                        "{} {n}x{k} m={m}: {g} vs candle {cd} (tolerance {tol})",
                        ty.name()
                    );
                }
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
    for ty in types() {
        for (n, k, m) in [(12, 512, 5), (20, 256, 3)] {
            let c = Case::new(ty, n, k, 23);
            let mut x = noise(m * k, 8, 1.0);
            // An outlier per row makes the quant step coarse.
            for j in 0..m {
                x[j * k + 7] = 6.0 * (j as f32 + 1.0);
            }
            let got = c.q.forward(&x, m).unwrap();
            let wd = c.values();
            let slack = sum_term_slack(&c, &x, m);
            for j in 0..m {
                for r in 0..n {
                    let (xs, ws) = (&x[j * k..(j + 1) * k], &wd[r * k..(r + 1) * k]);
                    let exact: f64 = xs
                        .iter()
                        .zip(ws)
                        .map(|(&a, &w)| f64::from(a) * f64::from(w))
                        .sum();
                    let bound = slack[j * n + r] + rounding_bound(ws, xs, QK);
                    let g = f64::from(got[j * n + r]);
                    assert!(
                        (g - exact).abs() <= bound + 1e-5 * exact.abs().max(1.0),
                        "{} {n}x{k} row {j} col {r}: {g} vs {exact} (bound {bound})",
                        ty.name()
                    );
                }
            }
        }
    }
}

/// The largest and smallest possible quants and scales: the integer parts
/// stay exact (a folded quant is at most 4096 in magnitude).
#[test]
fn extreme_blocks_match_the_f64_reference() {
    for ty in types() {
        for fill in [0x00u8, 0xFF, 0x80, 0x7F, 0xA5] {
            let (n, k, m) = (6, 512, 3);
            let row = ty.byte_len(k).unwrap();
            let mut bytes = vec![fill; n * row];
            for (i, block) in bytes.chunks_exact_mut(ty.type_size().unwrap()).enumerate() {
                // Only the scale fields change: the quant bytes stay extreme.
                fix_scales(ty, block, 40 + i as u32 % 5, 0.004);
            }
            let c = Case::of(ty, bytes, n, k);
            let x = noise(m * k, 9, 4.0);
            let got = c.q.forward(&x, m).unwrap();
            let want = reference(&c, &x, m);
            let slack = sum_term_slack(&c, &x, m);
            let scale = want.iter().fold(0f64, |a, v| a.max(v.abs()));
            for ((g, r), s) in got.iter().zip(&want).zip(&slack) {
                assert!(
                    (f64::from(*g) - r).abs() <= 1e-4 * scale.max(1e-3) + s,
                    "{} fill {fill:#x}: {g} vs {r}",
                    ty.name()
                );
            }
        }
    }
}

/// A prefix of the weight rows computes exactly the first outputs of the
/// whole product, whatever the tiling.
#[test]
fn a_weight_row_prefix_equals_the_start_of_the_product() {
    for ty in types() {
        for (n, k, m, rows) in [(10, 512, 3, 7), (600, 256, 2, 450), (64, 768, 5, 1)] {
            let c = Case::new(ty, n, k, 17);
            let x = noise(m * k, 9, 1.0);
            let whole = c.q.forward(&x, m).unwrap();
            let part = c.q.forward_rows(&x, m, rows).unwrap();
            assert_eq!(part.len(), m * rows);
            for j in 0..m {
                assert_eq!(
                    part[j * rows..(j + 1) * rows],
                    whole[j * n..j * n + rows],
                    "{} {n}x{k} m={m} rows={rows} row {j}",
                    ty.name()
                );
            }
            assert!(c.q.forward_rows(&x, m, 0).is_err());
            assert!(c.q.forward_rows(&x, m, n + 1).is_err());
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
fn activations_quantise_like_candles_q8_0_and_q8_1() {
    let (m, k) = (3, 512);
    let x = awkward_acts(m, k);
    let t = Tensor::from_vec(x.clone(), (m, k), &Device::Cpu).unwrap();
    for (dtype, min, size, quants_at) in [
        (GgmlDType::Q8_0, MIN_NONE, 34, 2),
        (GgmlDType::Q8_1, MIN_Q8_1, 36, 4),
    ] {
        let q = QTensor::quantize(&t, dtype).unwrap();
        let bytes = q.data().unwrap();
        let act = Acts::new(&x, m, k, min);
        for j in 0..m {
            for b in 0..k / QK {
                let block = &bytes[(j * (k / QK) + b) * size..][..size];
                assert_eq!(
                    act.scales[b * act.m + j],
                    f16_le(block, 0),
                    "{dtype:?} scale of row {j} block {b}"
                );
                if min == MIN_Q8_1 {
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

/// The half sums are the exact integer sums of the first and the last 16
/// quants times the block scale.
#[test]
fn the_half_sums_are_exact_integer_sums_times_the_scale() {
    let (m, k) = (3, 512);
    let x = awkward_acts(m, k);
    let act = Acts::new(&x, m, k, MIN_HALVES);
    for j in 0..m {
        for b in 0..k / QK {
            let d = act.scales[b * act.m + j];
            let sum = |r: std::ops::Range<usize>| -> i32 {
                r.map(|e| i32::from(act_quant(&act, b, j, e))).sum()
            };
            assert_eq!(
                act.halves[b * act.m + j],
                [sum(0..LANES) as f32 * d, sum(LANES..QK) as f32 * d],
                "row {j} block {b}"
            );
        }
    }
}

/// Row exactness at weight shapes of real models (n not a multiple of the
/// chunk size: the pooled path has tail chunks): the batch's row `j` equals
/// the row alone, for every type.
#[test]
fn real_shapes_are_row_exact() {
    for ty in types() {
        for (n, k) in [
            (1030usize, 2048usize),
            (3072, 1024),
            (1024, 3072),
            (2056, 1024),
        ] {
            let c = Case::random(ty, n, k, 3);
            let x = noise(9 * k, 9, 2.0);
            let alone: Vec<Vec<f32>> = (0..9)
                .map(|j| c.q.forward(&x[j * k..(j + 1) * k], 1).unwrap())
                .collect();
            for m in [2, 3, 4, 5, 9] {
                let batch = c.q.forward(&x[..m * k], m).unwrap();
                for j in 0..m {
                    assert_eq!(
                        batch[j * n..(j + 1) * n],
                        alone[j][..],
                        "{} {n}x{k}: row {j} of a batch of {m}",
                        ty.name()
                    );
                }
            }
        }
    }
}

/// The determinism contract: a row's bits do not depend on the batch.
#[test]
fn a_row_is_bit_identical_alone_and_in_any_batch() {
    for ty in types() {
        for (n, k) in [(10, 512), (512, 512)] {
            let c = Case::new(ty, n, k, 3);
            let x = noise(17 * k, 9, 2.0);
            let alone: Vec<Vec<f32>> = (0..17)
                .map(|j| c.q.forward(&x[j * k..(j + 1) * k], 1).unwrap())
                .collect();
            for m in [2, 3, 4, 5, 8, 16, 17] {
                let batch = c.q.forward(&x[..m * k], m).unwrap();
                for j in 0..m {
                    assert_eq!(
                        batch[j * n..(j + 1) * n],
                        alone[j][..],
                        "{} {n}x{k}: row {j} of a batch of {m}",
                        ty.name()
                    );
                }
            }
        }
    }
}

#[test]
fn zero_activations_give_exact_zeros() {
    for ty in types() {
        let c = Case::new(ty, 9, 512, 6);
        let out = c.q.forward(&vec![0f32; 4 * 512], 4).unwrap();
        assert!(out.iter().all(|v| *v == 0.0), "{}: {out:?}", ty.name());
    }
}

#[test]
fn from_raw_serves_quantized_types_and_rejects_the_rest() {
    for ty in types() {
        let row = ty.byte_len(512).unwrap();
        assert!(
            QmmWeight::from_raw(ty, Arc::from(vec![0u8; 4 * row]), 4, 512).is_some(),
            "{}",
            ty.name()
        );
        // A wrong byte count.
        assert!(QmmWeight::from_raw(ty, Arc::from(vec![0u8; 4 * row - 1]), 4, 512).is_none());
        assert!(QmmWeight::from_raw(ty, Arc::from(vec![0u8; 4 * row]), 3, 512).is_none());
        // A row that is not whole blocks.
        assert!(QmmWeight::from_raw(ty, Arc::from(vec![0u8; 4 * row]), 4, 512 + 16).is_none());
        assert!(QmmWeight::from_raw(ty, Arc::from(Vec::new()), 0, 512).is_none());
    }
    // The float types and ids the engine cannot run have no kernel.
    for ty in [
        GgufType::Ggml(GgmlDType::F32),
        GgufType::Ggml(GgmlDType::F16),
        GgufType::Ggml(GgmlDType::BF16),
        GgufType::I8,
        GgufType::Unknown(99),
    ] {
        assert!(
            QmmWeight::from_raw(ty, Arc::from(vec![0u8; 4096]), 4, 256).is_none(),
            "{}",
            ty.name()
        );
    }
}

/// A Q4_K matrix computed through the kernel agrees with candle's per-row
/// kernel on real quantized data (the activation rounding differs).
#[test]
fn k_quant_rows_track_candles_kernel_at_a_real_shape() {
    for dtype in [
        GgmlDType::Q2K,
        GgmlDType::Q3K,
        GgmlDType::Q4K,
        GgmlDType::Q5K,
        GgmlDType::Q6K,
    ] {
        let (n, k, m) = (256, 1024, 3);
        let c = Case::new(GgufType::Ggml(dtype), n, k, 5);
        let x = noise(m * k, 6, 1.0);
        let got = c.q.forward(&x, m).unwrap();
        let w = Arc::new(
            crate::provider::local::gguf::qtensor_from_bytes(
                dtype,
                &[n, k],
                &c.bytes,
                &Device::Cpu,
            )
            .unwrap(),
        );
        let xt = Tensor::from_vec(x, (m, k), &Device::Cpu).unwrap();
        let candle = QMatMul::from_arc(w)
            .unwrap()
            .forward(&xt)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let scale = candle.iter().fold(0f32, |a, v| a.max(v.abs()));
        for (g, cd) in got.iter().zip(&candle) {
            assert!(
                (g - cd).abs() <= 3e-2 * scale,
                "{dtype:?}: {g} vs candle {cd}"
            );
        }
    }
}

#[test]
fn types_with_minima_are_the_ones_the_activations_carry_sums_for() {
    for ty in types() {
        let min = has_min(ty);
        assert_eq!(
            min,
            matches!(
                ty,
                GgufType::Ggml(
                    GgmlDType::Q4_1
                        | GgmlDType::Q5_1
                        | GgmlDType::Q2K
                        | GgmlDType::Q4K
                        | GgmlDType::Q5K
                )
            ),
            "{}",
            ty.name()
        );
    }
}
