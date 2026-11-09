//! Hyper-connection arithmetic: the residual stream is `S` copies of the
//! hidden state per token (row-major `[S][D]`), and every sub-block (attention,
//! FFN) reads a weighted sum of the copies and writes back through
//! coefficients predicted from the stream itself.
//!
//! For a stream `x` of one token, `mixes = (hc_fn . x) / rms(x)` where the rms
//! covers all `S * D` values. `mixes` holds, in order, `S` pre weights, `S`
//! post weights and an `S x S` matrix:
//!
//! - `pre[s] = sigmoid(mixes * scale[0] + base) + eps`
//! - `post[s] = 2 * sigmoid(mixes * scale[1] + base)`
//! - `comb`: a row softmax of `mixes * scale[2] + base`, `+ eps`, made nearly
//!   doubly stochastic by `sinkhorn_iters - 1` alternations of row and column
//!   normalisations after one column normalisation.
//!
//! The sub-block reads `sum_s pre[s] x[s]` and its output `y` becomes
//! `new[j] = post[j] y + sum_a comb[a][j] x[a]`.
//!
//! Everything here is per token and in `f32` with a fixed summation order, so
//! a token's result never depends on the batch it travels in.

use super::csa::HyperSpec;

/// Most streams supported (`hyper_connection.count`).
pub(crate) const MAX_STREAMS: usize = 8;

/// The learned parameters of one hyper-connection site.
#[derive(Debug, Clone)]
pub(crate) struct HcWeights {
    /// `[(2 + S) * S][S * D]` row-major.
    pub(crate) mix: Vec<f32>,
    /// `(2 + S) * S` offsets.
    pub(crate) base: Vec<f32>,
    /// Scales of the pre, post and comb coefficients.
    pub(crate) scale: [f32; 3],
}

/// Floats of one token's coefficients for `streams` streams: `pre[S]`,
/// `post[S]`, then `comb[a * S + j]` (the weight of source stream `a` in new
/// stream `j`).
pub(crate) fn coeff_len(streams: usize) -> usize {
    2 * streams + streams * streams
}

/// The `pre` weights of the coefficients `c` of one token.
pub(crate) fn pre(c: &[f32], streams: usize) -> &[f32] {
    &c[..streams]
}

/// `a . b` with eight running partial sums, combined in a fixed order.
pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0.0f32; 8];
    let ((a8, ar), (b8, br)) = (a.as_chunks::<8>(), b.as_chunks::<8>());
    for (x, y) in a8.iter().zip(b8) {
        for i in 0..8 {
            acc[i] += x[i] * y[i];
        }
    }
    let mut tail = 0.0;
    for (x, y) in ar.iter().zip(br) {
        tail += x * y;
    }
    ((acc[0] + acc[4]) + (acc[1] + acc[5])) + ((acc[2] + acc[6]) + (acc[3] + acc[7])) + tail
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// The pre / post / comb coefficients of the stream `x` (`S * D` values),
/// written to `out` ([`coeff_len`] floats). `norm_eps` is the model's
/// RMS-norm epsilon.
pub(crate) fn coefficients(
    x: &[f32],
    w: &HcWeights,
    spec: &HyperSpec,
    norm_eps: f32,
    out: &mut [f32],
) {
    let s = spec.streams;
    let width = x.len();
    let rsqrt = 1.0 / (dot(x, x) / width as f32 + norm_eps).sqrt();

    let mut mixes = [0.0f32; (2 + MAX_STREAMS) * MAX_STREAMS];
    for (m, row) in mixes
        .iter_mut()
        .zip(w.mix.chunks_exact(width))
        .take((2 + s) * s)
    {
        *m = dot(row, x) * rsqrt;
    }

    let (pre, rest) = out.split_at_mut(s);
    let (post, comb) = rest.split_at_mut(s);
    for i in 0..s {
        pre[i] = sigmoid(mixes[i] * w.scale[0] + w.base[i]) + spec.eps;
        post[i] = 2.0 * sigmoid(mixes[s + i] * w.scale[1] + w.base[s + i]);
    }
    for (i, c) in comb.iter_mut().enumerate() {
        *c = mixes[2 * s + i] * w.scale[2] + w.base[2 * s + i];
    }
    // Row softmax (the row is the source stream), then `+ eps`.
    for row in comb.chunks_exact_mut(s) {
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut total = 0.0;
        for v in row.iter_mut() {
            *v = (*v - max).exp();
            total += *v;
        }
        for v in row.iter_mut() {
            *v = *v / total + spec.eps;
        }
    }
    normalise_columns(comb, s, spec.eps);
    for _ in 1..spec.sinkhorn_iters {
        for row in comb.chunks_exact_mut(s) {
            let total: f32 = row.iter().sum();
            for v in row.iter_mut() {
                *v /= total + spec.eps;
            }
        }
        normalise_columns(comb, s, spec.eps);
    }
}

/// Divide every column of the `s x s` matrix by its sum plus `eps`.
fn normalise_columns(comb: &mut [f32], s: usize, eps: f32) {
    for j in 0..s {
        let mut total = 0.0;
        for a in 0..s {
            total += comb[a * s + j];
        }
        for a in 0..s {
            comb[a * s + j] /= total + eps;
        }
    }
}

/// `out[d] = sum_s pre[s] * x[s][d]`: the sub-block input of one token.
pub(crate) fn collapse(x: &[f32], pre: &[f32], out: &mut [f32]) {
    let d = out.len();
    for (i, o) in out.iter_mut().enumerate() {
        let mut acc = 0.0;
        for (s, &p) in pre.iter().enumerate().take(x.len() / d) {
            acc += p * x[s * d + i];
        }
        *o = acc;
    }
}

/// The stream after a sub-block: `new[j] = post[j] * y + sum_a comb[a][j] *
/// x[a]`, written to `out` (`S * D` values; `x` is the stream before the
/// sub-block, `y` its `D`-wide output).
pub(crate) fn expand(y: &[f32], x: &[f32], post: &[f32], comb: &[f32], out: &mut [f32]) {
    let d = y.len();
    let s = x.len() / d;
    for j in 0..s {
        let dst = &mut out[j * d..(j + 1) * d];
        for (i, o) in dst.iter_mut().enumerate() {
            let mut acc = 0.0;
            for a in 0..s {
                acc += comb[a * s + j] * x[a * d + i];
            }
            *o = post[j] * y[i] + acc;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(streams: usize) -> HyperSpec {
        HyperSpec {
            streams,
            sinkhorn_iters: 20,
            eps: 1e-6,
        }
    }

    #[test]
    fn dot_matches_the_plain_sum() {
        let a: Vec<f32> = (0..37).map(|i| (i as f32 * 0.37).sin()).collect();
        let b: Vec<f32> = (0..37).map(|i| (i as f32 * 0.11).cos()).collect();
        let expected: f64 = a
            .iter()
            .zip(&b)
            .map(|(&x, &y)| f64::from(x) * f64::from(y))
            .sum();
        assert!((f64::from(dot(&a, &b)) - expected).abs() < 1e-5);
    }

    #[test]
    fn comb_is_doubly_stochastic_after_sinkhorn() {
        let s = 4;
        let d = 6;
        let x: Vec<f32> = (0..s * d)
            .map(|i| ((i * 7 % 13) as f32 - 6.0) * 0.3)
            .collect();
        let mix: Vec<f32> = (0..(2 + s) * s * s * d)
            .map(|i| ((i * 5 % 17) as f32 - 8.0) * 0.05)
            .collect();
        let base: Vec<f32> = (0..(2 + s) * s).map(|i| (i as f32 - 10.0) * 0.1).collect();
        let w = HcWeights {
            mix,
            base,
            scale: [0.5, 0.7, 0.9],
        };
        let mut c = vec![0.0; coeff_len(s)];
        coefficients(&x, &w, &spec(s), 1e-6, &mut c);
        let comb = &c[2 * s..];
        for a in 0..s {
            let row: f32 = (0..s).map(|j| comb[a * s + j]).sum();
            assert!((row - 1.0).abs() < 1e-3, "row {a}: {row}");
        }
        for j in 0..s {
            let col: f32 = (0..s).map(|a| comb[a * s + j]).sum();
            assert!((col - 1.0).abs() < 1e-3, "col {j}: {col}");
        }
        for i in 0..s {
            assert!(c[i] > 0.0 && c[i] < 1.0 + 1e-5);
            assert!(c[s + i] > 0.0 && c[s + i] < 2.0);
        }
    }

    #[test]
    fn collapse_and_expand_follow_their_definitions() {
        let (s, d) = (3, 2);
        let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut y = [0.0; 2];
        collapse(&x, &[0.5, 0.25, 0.25], &mut y);
        assert_eq!(y, [0.5 + 0.75 + 1.25, 1.0 + 1.0 + 1.5]);
        let post = [1.0, 2.0, 3.0];
        // Identity comb: every stream keeps itself.
        let mut comb = [0.0; 9];
        for a in 0..s {
            comb[a * s + a] = 1.0;
        }
        let mut out = [0.0; 6];
        expand(&[10.0, 20.0], &x, &post, &comb, &mut out);
        assert_eq!(out, [11.0, 22.0, 23.0, 44.0, 35.0, 66.0]);
        let _ = d;
    }

    // Goldens from the torch port of the reference (`hc_mixes` +
    // `hc_split_sinkhorn`, `.omp/ds-research/gen_l0_goldens.py`).
    const HC_PRE: [f32; 4] = [0.20649546, 0.48299024, 0.7820599, 0.46195415];
    const HC_POST: [f32; 4] = [1.4079318, 0.6789722, 1.3780661, 0.6111839];
    const HC_COMB: [f32; 16] = [
        0.20469923,
        0.04556825,
        0.32619628,
        0.4235353,
        0.1339557,
        0.45954883,
        0.14964555,
        0.2568489,
        0.15905865,
        0.40244317,
        0.17323723,
        0.26525995,
        0.5022854,
        0.092438735,
        0.3509199,
        0.05435493,
    ];

    #[test]
    fn coefficients_match_the_torch_port() {
        let (s, d) = (4usize, 8usize);
        let x: Vec<f32> = (0..s * d)
            .map(|i| ((((i * 37) % 61) as i64 - 30) as f64 / 17.0) as f32)
            .collect();
        let mix: Vec<f32> = (0..(2 + s) * s * s * d)
            .map(|i| ((((i * 37) % 101) as i64 - 50) as f64 / 400.0) as f32)
            .collect();
        let base: Vec<f32> = (0..(2 + s) * s)
            .map(|i| ((((i * 13) % 29) as i64 - 14) as f64 / 10.0) as f32)
            .collect();
        let w = HcWeights {
            mix,
            base,
            scale: [0.8, 0.6, 0.9],
        };
        let mut c = vec![0.0; coeff_len(s)];
        coefficients(&x, &w, &spec(s), 1e-6, &mut c);
        for (got, want) in c.iter().zip(HC_PRE.iter().chain(&HC_POST).chain(&HC_COMB)) {
            assert!((got - want).abs() < 2e-6, "{got} vs {want}");
        }
    }
}
