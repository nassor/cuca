//! Hyper-connection kernels of the GPU engine (CubeCL): the device twins of
//! [`hc`](super::hc) and of the engram gate of the CPU engine's
//! `engram_layer`.
//!
//! The residual stream of a token is `S` copies of the hidden state
//! (`[S][D]`, `S * D` floats), a batch of `t` tokens a `[t, S * D]` tensor.
//! Every kernel works per token with a fixed reduction order, so a token's
//! result never depends on the batch it travels in:
//!
//! - [`hc_coefficients`]: the `pre | post | comb` coefficients of one token
//!   (`2 S + S^2` floats, the layout of [`hc::coeff_len`](super::hc::coeff_len)).
//!   One cube of [`HC_UNITS`] units per token: the units stride over the `S *
//!   D` values (unit `u` takes `u, u + HC_UNITS, ...`), each accumulating the
//!   sum of squares and the `(2 + S) * S` dot products with the rows of the
//!   mix matrix in registers, a shared-memory tree adds the partial sums (the
//!   same tree for every token), and unit 0 finishes the sigmoids, the row
//!   softmax and the Sinkhorn passes on `S * S` values. A burn matmul with so
//!   few rows hits known burn/Vulkan problems, and the products would be
//!   separate launches anyway. The cube reads the whole mix matrix (2 MB at
//!   the real shape), which one compute unit of the GPU pulls at about 30
//!   GB/s whatever the arithmetic, so a call costs about 60 us at the real
//!   shape however many tokens it holds (up to the tokens one GPU's compute
//!   units can run at once);
//! - [`hc_collapse`] / [`stream_mean`]: the `D`-wide sub-block input `sum_s
//!   w[s] x[s]` (the learned `pre` weights, or `1 / S` each);
//! - [`hc_expand`]: the stream after a sub-block, `new[j] = post[j] y +
//!   sum_a comb[a][j] x[a]`;
//! - [`stream_repeat`]: `S` copies of a `D`-wide row (a token's embedding
//!   becomes its stream);
//! - [`engram_gate`]: the engram layer's per-stream gate of a fetched value
//!   (`h += sigmoid(g) value`, see the function).
//!
//! The arithmetic follows the host functions operation for operation (the
//! same summation order in the per-element kernels), and the reductions use
//! a tree instead of the host's eight running sums, so results agree to float
//! rounding (about `1e-6` relative), not bit for bit.
//!
//! The kernels are written in safe Rust (`#[cube(launch)]` with the checked
//! `launch`) and reach the backend through burn's `backend_extension`, like
//! [`moe_kernel`](super::moe_kernel).
//!
//! # Limits
//!
//! - at most 65535 tokens per call (one cube per token for the per-token
//!   kernels), and a `[t, S * D]` tensor must stay below the 1 GiB of a wgpu
//!   storage buffer (`t * S * D * 4` bytes);
//! - `S` at most [`MAX_KERNEL_STREAMS`] (4, the real model's count): the
//!   coefficient kernel's tree holds `(2 + S) * S + 1` sums per unit, which is
//!   25.6 KB of the 32 KB of shared memory at `S = 4`.

// The `#[cube]` kernels are Rust the CubeCL macro translates: it needs runtime
// values built from runtime values (`eps - eps` is a runtime zero where a
// literal would be evaluated by rustc), plain `x = x / 2` and nested `if`s.
#![allow(
    clippy::eq_op,
    clippy::assign_op_pattern,
    clippy::collapsible_if,
    clippy::manual_div_ceil,
    clippy::too_many_arguments
)]

use burn::backend::{backend_extension, tensor::FloatTensor};
use burn::cubecl::prelude::*;
use burn::cubecl::{CubeDim, calculate_cube_count_elemwise};
use burn::tensor::{DType, Shape};
use burn_cubecl::kernel::into_contiguous;
use burn_cubecl::ops::numeric::empty_device_contiguous_dtype;
use burn_cubecl::{CubeBackend, tensor::CubeTensor};

use super::hc::coeff_len;

/// Units of the cube of a per-token reduction kernel (a power of two). Fixed,
/// so that the reduction tree, and with it every result, is the same whatever
/// the batch.
pub(crate) const HC_UNITS: usize = 256;

/// Most streams the kernels take: the shared-memory budget of the coefficient
/// kernel (see the module docs).
const MAX_KERNEL_STREAMS: usize = 4;

/// Elements of the stream a unit loads before using any of them in
/// [`hc_coefficients`].
const PREFETCH: usize = 4;

/// Tokens one call may hold: wgpu's limit of cubes along one dimension.
const MAX_TOKENS: usize = 65535;

/// The constants of one hyper-connection site.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct HcShape {
    /// Copies of the hidden state per token (`S`).
    pub(crate) streams: usize,
    /// Width of one copy (`D`).
    pub(crate) dim: usize,
    /// Sinkhorn normalisation passes of the stream-mixing matrix.
    pub(crate) sinkhorn_iters: usize,
    /// The epsilon of the coefficients (`hyper_connection.epsilon`).
    pub(crate) hc_eps: f32,
    /// The model's RMS-norm epsilon (the stream's and the engram's).
    pub(crate) norm_eps: f32,
}

/// What a backend operation takes: the tokens of the call and the site's
/// constants.
#[derive(Clone, Debug)]
pub(crate) struct HcCall {
    pub(crate) tokens: usize,
    pub(crate) shape: HcShape,
}

// ---------------------------------------------------------------- kernels

#[cube]
fn sigmoid(x: f32) -> f32 {
    1.0f32 / (1.0f32 + (-x).exp())
}

/// Divide every column of the `s x s` matrix by its sum plus `eps`
/// (`hc::normalise_columns`). `zero` is a runtime zero.
#[cube]
fn normalise_columns(comb: &mut Array<f32>, eps: f32, zero: f32, #[comptime] s: usize) {
    #[unroll]
    for j in 0..s {
        let mut total = zero;
        #[unroll]
        for a in 0..s {
            total += comb[a * s + j];
        }
        #[unroll]
        for a in 0..s {
            comb[a * s + j] = comb[a * s + j] / (total + eps);
        }
    }
}

/// The `pre | post | comb` coefficients of the streams `x` `[tokens, width]`:
/// cube `t` handles token `t`. The `nm + 1` partial sums of a unit (the
/// stream's sum of squares, then the `nm = (2 + s) * s` dot products with the
/// rows of `mix` `[nm, width]`) are added by a shared-memory tree; unit 0
/// then follows `hc::coefficients`. `base` is `[nm]`, `scale` `[3]`, `out`
/// `[tokens, cl]`.
#[cube(launch)]
fn hc_coefficients_kernel(
    x: &Tensor<f32>,
    mix: &Tensor<f32>,
    base: &Tensor<f32>,
    scale: &Tensor<f32>,
    out: &mut Tensor<f32>,
    width: u32,
    sinkhorn: u32,
    hc_eps: f32,
    norm_eps: f32,
    #[comptime] s: usize,
    #[comptime] nm: usize,
    #[comptime] nm1: usize,
    #[comptime] cl: usize,
    #[comptime] units: usize,
    #[comptime] steps: usize,
    #[comptime] red_len: usize,
    #[comptime] ahead: usize,
    #[comptime] ahead_mix: usize,
) {
    let tok = CUBE_POS_X as usize;
    let tid = UNIT_POS_X as usize;
    let w = usize::cast_from(width);
    let at = tok * w;
    let zero = norm_eps - norm_eps;
    let mut red = Shared::<[f32]>::new_slice(red_len);

    let mut acc = Array::<f32>::new(nm1);
    #[unroll]
    for m in 0..nm1 {
        acc[m] = zero;
    }
    // The unit's elements `tid, tid + units, ...`, `ahead` of them loaded
    // before any is used (the loop is memory-latency bound: one cube reads
    // the whole mix matrix), the products added in element order. A missing
    // element (past the end) is zero and adds nothing.
    let iters = (w + units - 1usize) / units;
    let groups = (iters + ahead - 1usize) / ahead;
    let mut vs = Array::<f32>::new(ahead);
    let mut ms = Array::<f32>::new(ahead_mix);
    for g in 0..groups {
        #[unroll]
        for p in 0..ahead {
            let i = tid + (g * ahead + p) * units;
            vs[p] = zero;
            #[unroll]
            for m in 0..nm {
                ms[p * nm + m] = zero;
            }
            if i < w {
                vs[p] = x[at + i];
                #[unroll]
                for m in 0..nm {
                    ms[p * nm + m] = mix[m * w + i];
                }
            }
        }
        #[unroll]
        for p in 0..ahead {
            let v = vs[p];
            acc[0] = acc[0] + v * v;
            #[unroll]
            for m in 0..nm {
                acc[m + 1usize] = acc[m + 1usize] + v * ms[p * nm + m];
            }
        }
    }
    #[unroll]
    for m in 0..nm1 {
        red[m * units + tid] = acc[m];
    }
    sync_cube();
    let mut st = (tid - tid) + units / 2usize;
    for _ in 0..steps {
        if tid < st {
            #[unroll]
            for m in 0..nm1 {
                red[m * units + tid] = red[m * units + tid] + red[m * units + tid + st];
            }
        }
        sync_cube();
        st = st / 2usize;
    }

    if tid == 0usize {
        let inv = 1.0f32 / (red[0] / f32::cast_from(w) + norm_eps).sqrt();
        let o = tok * cl;
        #[unroll]
        for i in 0..s {
            let pre = red[(1usize + i) * units] * inv;
            let post = red[(1usize + s + i) * units] * inv;
            out[o + i] = sigmoid(pre * scale[0] + base[i]) + hc_eps;
            out[o + s + i] = 2.0f32 * sigmoid(post * scale[1] + base[s + i]);
        }
        let mut comb = Array::<f32>::new(s * s);
        #[unroll]
        for i in 0..s * s {
            comb[i] =
                red[(1usize + 2usize * s + i) * units] * inv * scale[2] + base[2usize * s + i];
        }
        // Row softmax (the row is the source stream), then `+ eps`.
        #[unroll]
        for a in 0..s {
            let mut top = comb[a * s];
            #[unroll]
            for j in 1..s {
                if comb[a * s + j] > top {
                    top = comb[a * s + j];
                }
            }
            let mut total = zero;
            #[unroll]
            for j in 0..s {
                let e = (comb[a * s + j] - top).exp();
                comb[a * s + j] = e;
                total += e;
            }
            #[unroll]
            for j in 0..s {
                comb[a * s + j] = comb[a * s + j] / total + hc_eps;
            }
        }
        normalise_columns(&mut comb, hc_eps, zero, s);
        for _ in 1..usize::cast_from(sinkhorn) {
            #[unroll]
            for a in 0..s {
                let mut total = zero;
                #[unroll]
                for j in 0..s {
                    total += comb[a * s + j];
                }
                #[unroll]
                for j in 0..s {
                    comb[a * s + j] = comb[a * s + j] / (total + hc_eps);
                }
            }
            normalise_columns(&mut comb, hc_eps, zero, s);
        }
        #[unroll]
        for i in 0..s * s {
            out[o + 2usize * s + i] = comb[i];
        }
    }
}

/// `out[token, i] = sum_a w[token, a] * x[token, a, i]` with `w` the first
/// `s` columns of the coefficient rows (`cl` wide) and the sum in stream
/// order (`hc::collapse`): one unit per output element.
#[cube(launch)]
fn hc_collapse_kernel(
    x: &Tensor<f32>,
    coeffs: &Tensor<f32>,
    out: &mut Tensor<f32>,
    total: u32,
    d: u32,
    #[comptime] s: usize,
    #[comptime] cl: usize,
) {
    let job = ABSOLUTE_POS;
    if job < usize::cast_from(total) {
        let dd = usize::cast_from(d);
        let tok = job / dd;
        let i = job % dd;
        let mut acc = f32::cast_from(job - job);
        #[unroll]
        for a in 0..s {
            acc += coeffs[tok * cl + a] * x[(tok * s + a) * dd + i];
        }
        out[job] = acc;
    }
}

/// `out[token, i] = sum_a x[token, a, i] * inv` (`hc::collapse` with equal
/// weights): the mean of the copies, one unit per output element.
#[cube(launch)]
fn stream_mean_kernel(
    x: &Tensor<f32>,
    out: &mut Tensor<f32>,
    total: u32,
    d: u32,
    inv: f32,
    #[comptime] s: usize,
) {
    let job = ABSOLUTE_POS;
    if job < usize::cast_from(total) {
        let dd = usize::cast_from(d);
        let tok = job / dd;
        let i = job % dd;
        let mut acc = f32::cast_from(job - job);
        #[unroll]
        for a in 0..s {
            acc += inv * x[(tok * s + a) * dd + i];
        }
        out[job] = acc;
    }
}

/// The stream after a sub-block (`hc::expand`): `new[j, i] = post[j] y[i] +
/// sum_a comb[a][j] x[a, i]`, one unit per (token, `i`) writing the `s`
/// copies. `coeffs` rows are `cl` wide (`pre | post | comb`).
#[cube(launch)]
fn hc_expand_kernel(
    y: &Tensor<f32>,
    x: &Tensor<f32>,
    coeffs: &Tensor<f32>,
    out: &mut Tensor<f32>,
    total: u32,
    d: u32,
    #[comptime] s: usize,
    #[comptime] cl: usize,
) {
    let job = ABSOLUTE_POS;
    if job < usize::cast_from(total) {
        let dd = usize::cast_from(d);
        let tok = job / dd;
        let i = job % dd;
        let yv = y[job];
        let mut xs = Array::<f32>::new(s);
        #[unroll]
        for a in 0..s {
            xs[a] = x[(tok * s + a) * dd + i];
        }
        let c = tok * cl;
        #[unroll]
        for j in 0..s {
            let mut acc = f32::cast_from(job - job);
            #[unroll]
            for a in 0..s {
                acc += coeffs[c + 2usize * s + a * s + j] * xs[a];
            }
            out[(tok * s + j) * dd + i] = coeffs[c + s + j] * yv + acc;
        }
    }
}

/// `s` copies of every `d`-wide row: one unit per input element.
#[cube(launch)]
fn stream_repeat_kernel(
    x: &Tensor<f32>,
    out: &mut Tensor<f32>,
    total: u32,
    d: u32,
    #[comptime] s: usize,
) {
    let job = ABSOLUTE_POS;
    if job < usize::cast_from(total) {
        let dd = usize::cast_from(d);
        let tok = job / dd;
        let i = job % dd;
        let v = x[job];
        #[unroll]
        for a in 0..s {
            out[(tok * s + a) * dd + i] = v;
        }
    }
}

/// The engram gate: cube `t` handles token `t`. Per stream `c` with the
/// token's copy `h`, key `k` (the first `s * d` columns of the `kv` row) and
/// per-stream weights `w` (`gate`): `rstd = rsqrt(mean(h^2) + eps) *
/// rsqrt(mean(k^2) + eps)`, `x = sum(h w k) * rstd * scale`, `g =
/// sigmoid(copysign(sqrt(max(|x|, 1e-6)), x))` and `h += g * value` (the
/// last `d` columns of the `kv` row). The three sums of every stream are
/// added by the shared-memory tree of the coefficient kernel.
#[cube(launch)]
fn engram_gate_kernel(
    h: &Tensor<f32>,
    kv: &Tensor<f32>,
    gate: &Tensor<f32>,
    out: &mut Tensor<f32>,
    d: u32,
    eps: f32,
    scale: f32,
    #[comptime] s: usize,
    #[comptime] n3: usize,
    #[comptime] units: usize,
    #[comptime] steps: usize,
    #[comptime] red_len: usize,
) {
    let tok = CUBE_POS_X as usize;
    let tid = UNIT_POS_X as usize;
    let dd = usize::cast_from(d);
    let h_at = tok * s * dd;
    let key_at = tok * (s + 1usize) * dd;
    let val_at = key_at + s * dd;
    let zero = eps - eps;
    let mut red = Shared::<[f32]>::new_slice(red_len);

    let mut acc = Array::<f32>::new(n3);
    #[unroll]
    for m in 0..n3 {
        acc[m] = zero;
    }
    let iters = (dd + units - 1usize) / units;
    for k in 0..iters {
        let i = tid + k * units;
        if i < dd {
            #[unroll]
            for c in 0..s {
                let hv = h[h_at + c * dd + i];
                let kv_c = kv[key_at + c * dd + i];
                acc[3usize * c] = acc[3usize * c] + hv * hv;
                acc[3usize * c + 1usize] = acc[3usize * c + 1usize] + kv_c * kv_c;
                acc[3usize * c + 2usize] = acc[3usize * c + 2usize] + hv * gate[c * dd + i] * kv_c;
            }
        }
    }
    #[unroll]
    for m in 0..n3 {
        red[m * units + tid] = acc[m];
    }
    sync_cube();
    let mut st = (tid - tid) + units / 2usize;
    for _ in 0..steps {
        if tid < st {
            #[unroll]
            for m in 0..n3 {
                red[m * units + tid] = red[m * units + tid] + red[m * units + tid + st];
            }
        }
        sync_cube();
        st = st / 2usize;
    }

    // Every unit derives the `s` gates from the totals.
    let mut g = Array::<f32>::new(s);
    #[unroll]
    for c in 0..s {
        let hh = red[3usize * c * units];
        let kk = red[(3usize * c + 1usize) * units];
        let hwk = red[(3usize * c + 2usize) * units];
        let rstd = (1.0f32 / (hh / f32::cast_from(dd) + eps).sqrt())
            * (1.0f32 / (kk / f32::cast_from(dd) + eps).sqrt());
        let x = hwk * rstd * scale;
        let mut mag = f32::abs(x);
        if mag < 1.0e-6f32 {
            mag = (mag - mag) + 1.0e-6f32;
        }
        let mut signed = mag.sqrt();
        if (u32::reinterpret(x) & 0x8000_0000u32) != 0u32 {
            signed = -signed;
        }
        g[c] = sigmoid(signed);
    }
    for k in 0..iters {
        let i = tid + k * units;
        if i < dd {
            let value = kv[val_at + i];
            #[unroll]
            for c in 0..s {
                out[h_at + c * dd + i] = h[h_at + c * dd + i] + g[c] * value;
            }
        }
    }
}

// --------------------------------------------------------------- extension

/// Backend operations of the hyper-connection stream (see the module docs).
#[backend_extension(Cube, Fusion: cfg(feature = "provider-local-gpu"))]
pub trait HcKernels: burn::backend::Backend {
    /// Coefficients `[tokens, 2 S + S^2]` of the streams `stream` `[tokens,
    /// S D]` with the mix matrix `mix` `[(2 + S) S, S D]`, the offsets `base`
    /// `[(2 + S) S]` and the scales `scale` `[3]`.
    #[fusion(dtype = stream, shape = Shape::new([
        call.tokens,
        2 * call.shape.streams + call.shape.streams * call.shape.streams,
    ]))]
    fn hc_coefficients(
        stream: FloatTensor<Self>,
        mix: FloatTensor<Self>,
        base: FloatTensor<Self>,
        scale: FloatTensor<Self>,
        call: HcCall,
    ) -> FloatTensor<Self>;

    /// The streams `[tokens, S D]` collapsed by the first `S` columns of
    /// `coeffs` `[tokens, 2 S + S^2]`: `[tokens, D]`.
    #[fusion(dtype = stream, shape = Shape::new([call.tokens, call.shape.dim]))]
    fn hc_collapse(
        stream: FloatTensor<Self>,
        coeffs: FloatTensor<Self>,
        call: HcCall,
    ) -> FloatTensor<Self>;

    /// The mean of the `S` copies of every stream: `[tokens, D]`.
    #[fusion(dtype = stream, shape = Shape::new([call.tokens, call.shape.dim]))]
    #[cfg_attr(not(test), allow(dead_code))]
    fn stream_mean(stream: FloatTensor<Self>, call: HcCall) -> FloatTensor<Self>;

    /// The stream after a sub-block: `y` `[tokens, D]` is the block's output,
    /// `stream` its input stream `[tokens, S D]`, `coeffs` the coefficients
    /// of the block's site: `[tokens, S D]`.
    #[fusion(dtype = stream, shape = Shape::new([
        call.tokens,
        call.shape.streams * call.shape.dim,
    ]))]
    fn hc_expand(
        y: FloatTensor<Self>,
        stream: FloatTensor<Self>,
        coeffs: FloatTensor<Self>,
        call: HcCall,
    ) -> FloatTensor<Self>;

    /// `S` copies of every row of `x` `[tokens, D]`: `[tokens, S D]`.
    #[fusion(dtype = x, shape = Shape::new([
        call.tokens,
        call.shape.streams * call.shape.dim,
    ]))]
    fn stream_repeat(x: FloatTensor<Self>, call: HcCall) -> FloatTensor<Self>;

    /// The engram gate of the streams `[tokens, S D]` with the fetched
    /// `kv` `[tokens, (S + 1) D]` (a key per stream, then one value) and the
    /// per-stream weights `gate` `[S D]`: `[tokens, S D]`.
    #[fusion(dtype = stream, shape = Shape::new([
        call.tokens,
        call.shape.streams * call.shape.dim,
    ]))]
    fn engram_gate(
        stream: FloatTensor<Self>,
        kv: FloatTensor<Self>,
        gate: FloatTensor<Self>,
        call: HcCall,
    ) -> FloatTensor<Self>;
}

fn f32_tensor(like: &CubeTensor, shape: Shape) -> CubeTensor {
    empty_device_contiguous_dtype(like.client.clone(), like.device.clone(), shape, DType::F32)
}

/// Cubes of a per-token reduction kernel: one per token.
fn per_token(tokens: usize) -> CubeCount {
    CubeCount::Static(tokens as u32, 1, 1)
}

impl HcKernels for CubeBackend {
    fn hc_coefficients(
        stream: CubeTensor,
        mix: CubeTensor,
        base: CubeTensor,
        scale: CubeTensor,
        call: HcCall,
    ) -> CubeTensor {
        // The kernels index the buffers linearly.
        let stream = into_contiguous(stream);
        let mix = into_contiguous(mix);
        let base = into_contiguous(base);
        let scale = into_contiguous(scale);
        let s = call.shape.streams;
        let cl = coeff_len(s);
        let nm = (2 + s) * s;
        let out = f32_tensor(&stream, Shape::new([call.tokens, cl]));
        let client = stream.client.clone();
        hc_coefficients_kernel::launch(
            &client,
            per_token(call.tokens),
            CubeDim::new_1d(HC_UNITS as u32),
            stream.into_tensor_arg(),
            mix.into_tensor_arg(),
            base.into_tensor_arg(),
            scale.into_tensor_arg(),
            out.clone().into_tensor_arg(),
            (s * call.shape.dim) as u32,
            call.shape.sinkhorn_iters as u32,
            call.shape.hc_eps,
            call.shape.norm_eps,
            s,
            nm,
            nm + 1,
            cl,
            HC_UNITS,
            HC_UNITS.trailing_zeros() as usize,
            (nm + 1) * HC_UNITS,
            PREFETCH,
            PREFETCH * nm,
        );
        out
    }

    fn hc_collapse(stream: CubeTensor, coeffs: CubeTensor, call: HcCall) -> CubeTensor {
        let stream = into_contiguous(stream);
        let coeffs = into_contiguous(coeffs);
        let s = call.shape.streams;
        let d = call.shape.dim;
        let total = call.tokens * d;
        let out = f32_tensor(&stream, Shape::new([call.tokens, d]));
        let client = stream.client.clone();
        let dim = CubeDim::new(&client, total);
        let count = calculate_cube_count_elemwise(&client, total, dim);
        hc_collapse_kernel::launch(
            &client,
            count,
            dim,
            stream.into_tensor_arg(),
            coeffs.into_tensor_arg(),
            out.clone().into_tensor_arg(),
            total as u32,
            d as u32,
            s,
            coeff_len(s),
        );
        out
    }

    fn stream_mean(stream: CubeTensor, call: HcCall) -> CubeTensor {
        let stream = into_contiguous(stream);
        let s = call.shape.streams;
        let d = call.shape.dim;
        let total = call.tokens * d;
        let out = f32_tensor(&stream, Shape::new([call.tokens, d]));
        let client = stream.client.clone();
        let dim = CubeDim::new(&client, total);
        let count = calculate_cube_count_elemwise(&client, total, dim);
        stream_mean_kernel::launch(
            &client,
            count,
            dim,
            stream.into_tensor_arg(),
            out.clone().into_tensor_arg(),
            total as u32,
            d as u32,
            1.0 / s as f32,
            s,
        );
        out
    }

    fn hc_expand(
        y: CubeTensor,
        stream: CubeTensor,
        coeffs: CubeTensor,
        call: HcCall,
    ) -> CubeTensor {
        let y = into_contiguous(y);
        let stream = into_contiguous(stream);
        let coeffs = into_contiguous(coeffs);
        let s = call.shape.streams;
        let d = call.shape.dim;
        let total = call.tokens * d;
        let out = f32_tensor(&stream, Shape::new([call.tokens, s * d]));
        let client = stream.client.clone();
        let dim = CubeDim::new(&client, total);
        let count = calculate_cube_count_elemwise(&client, total, dim);
        hc_expand_kernel::launch(
            &client,
            count,
            dim,
            y.into_tensor_arg(),
            stream.into_tensor_arg(),
            coeffs.into_tensor_arg(),
            out.clone().into_tensor_arg(),
            total as u32,
            d as u32,
            s,
            coeff_len(s),
        );
        out
    }

    fn stream_repeat(x: CubeTensor, call: HcCall) -> CubeTensor {
        let x = into_contiguous(x);
        let s = call.shape.streams;
        let d = call.shape.dim;
        let total = call.tokens * d;
        let out = f32_tensor(&x, Shape::new([call.tokens, s * d]));
        let client = x.client.clone();
        let dim = CubeDim::new(&client, total);
        let count = calculate_cube_count_elemwise(&client, total, dim);
        stream_repeat_kernel::launch(
            &client,
            count,
            dim,
            x.into_tensor_arg(),
            out.clone().into_tensor_arg(),
            total as u32,
            d as u32,
            s,
        );
        out
    }

    fn engram_gate(
        stream: CubeTensor,
        kv: CubeTensor,
        gate: CubeTensor,
        call: HcCall,
    ) -> CubeTensor {
        let stream = into_contiguous(stream);
        let kv = into_contiguous(kv);
        let gate = into_contiguous(gate);
        let s = call.shape.streams;
        let d = call.shape.dim;
        let out = f32_tensor(&stream, Shape::new([call.tokens, s * d]));
        let client = stream.client.clone();
        engram_gate_kernel::launch(
            &client,
            per_token(call.tokens),
            CubeDim::new_1d(HC_UNITS as u32),
            stream.into_tensor_arg(),
            kv.into_tensor_arg(),
            gate.into_tensor_arg(),
            out.clone().into_tensor_arg(),
            d as u32,
            call.shape.norm_eps,
            (d as f32).powf(-0.5),
            s,
            3 * s,
            HC_UNITS,
            HC_UNITS.trailing_zeros() as usize,
            3 * s * HC_UNITS,
        );
        out
    }
}

// ---------------------------------------------------------- tensor wrappers

use burn::backend::Dispatch;
use burn::tensor::Tensor as BTensor;

/// The call of `tokens` tokens; checks the stream layout and the launch
/// limits.
fn call_of(stream: &BTensor<2>, shape: &HcShape) -> HcCall {
    let [tokens, width] = stream.dims();
    assert_eq!(
        width,
        shape.streams * shape.dim,
        "a stream row holds streams * dim values"
    );
    assert!(
        (1..=MAX_KERNEL_STREAMS).contains(&shape.streams),
        "{} streams (at most {MAX_KERNEL_STREAMS})",
        shape.streams
    );
    assert!(
        tokens <= MAX_TOKENS,
        "{tokens} tokens in one call (at most {MAX_TOKENS})"
    );
    HcCall {
        tokens,
        shape: *shape,
    }
}

/// [`HcKernels::hc_coefficients`] on tensors: `stream` is `[t, S * D]`, `mix`
/// `[(2 + S) * S, S * D]` (row `r` dots with the stream), `base` `[(2 + S) *
/// S]` and `scale` `[3]` (pre, post, comb); the result is `[t, 2 S + S^2]`,
/// `pre[S] | post[S] | comb[a * S + j]` per token.
pub(crate) fn hc_coefficients(
    stream: BTensor<2>,
    mix: BTensor<2>,
    base: BTensor<1>,
    scale: BTensor<1>,
    shape: &HcShape,
) -> BTensor<2> {
    let call = call_of(&stream, shape);
    let nm = (2 + shape.streams) * shape.streams;
    assert_eq!(mix.dims(), [nm, shape.streams * shape.dim], "mix");
    assert_eq!(base.dims(), [nm], "base");
    assert_eq!(scale.dims(), [3], "scale");
    BTensor::from_dispatch(<Dispatch as HcKernels>::hc_coefficients(
        stream.into_dispatch(),
        mix.into_dispatch(),
        base.into_dispatch(),
        scale.into_dispatch(),
        call,
    ))
}

/// [`HcKernels::hc_collapse`] on tensors: `stream` `[t, S * D]`, `coeffs` `[t,
/// 2 S + S^2]` (the `pre` weights are its first `S` columns); `[t, D]`.
pub(crate) fn hc_collapse(stream: BTensor<2>, coeffs: BTensor<2>, shape: &HcShape) -> BTensor<2> {
    let call = call_of(&stream, shape);
    assert_eq!(coeffs.dims(), [call.tokens, coeff_len(shape.streams)]);
    BTensor::from_dispatch(<Dispatch as HcKernels>::hc_collapse(
        stream.into_dispatch(),
        coeffs.into_dispatch(),
        call,
    ))
}

/// [`HcKernels::stream_mean`] on tensors: `[t, S * D]` to `[t, D]`.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn stream_mean(stream: BTensor<2>, shape: &HcShape) -> BTensor<2> {
    let call = call_of(&stream, shape);
    BTensor::from_dispatch(<Dispatch as HcKernels>::stream_mean(
        stream.into_dispatch(),
        call,
    ))
}

/// [`HcKernels::hc_expand`] on tensors: the block output `y` `[t, D]`, its
/// input `stream` `[t, S * D]` and the site's `coeffs` `[t, 2 S + S^2]` (the
/// `post` weights and `comb` are columns `S..`); `[t, S * D]`.
pub(crate) fn hc_expand(
    y: BTensor<2>,
    stream: BTensor<2>,
    coeffs: BTensor<2>,
    shape: &HcShape,
) -> BTensor<2> {
    let call = call_of(&stream, shape);
    assert_eq!(y.dims(), [call.tokens, shape.dim], "y");
    assert_eq!(coeffs.dims(), [call.tokens, coeff_len(shape.streams)]);
    BTensor::from_dispatch(<Dispatch as HcKernels>::hc_expand(
        y.into_dispatch(),
        stream.into_dispatch(),
        coeffs.into_dispatch(),
        call,
    ))
}

/// [`HcKernels::stream_repeat`] on tensors: `x` `[t, D]` to `[t, S * D]`.
pub(crate) fn stream_repeat(x: BTensor<2>, shape: &HcShape) -> BTensor<2> {
    let [tokens, dim] = x.dims();
    assert_eq!(dim, shape.dim, "x holds one copy per token");
    assert!(
        tokens <= MAX_TOKENS && (1..=MAX_KERNEL_STREAMS).contains(&shape.streams),
        "{tokens} tokens, {} streams",
        shape.streams
    );
    BTensor::from_dispatch(<Dispatch as HcKernels>::stream_repeat(
        x.into_dispatch(),
        HcCall {
            tokens,
            shape: *shape,
        },
    ))
}

/// [`HcKernels::engram_gate`] on tensors: `stream` is `[t, S * D]`, `kv`
/// `[t, (S + 1) * D]` (the key of every stream, `[S][D]`, then the value
/// `[D]`) and `gate` `[S * D]` (`[S][D]`); the result is `[t, S * D]`.
/// `shape.norm_eps` is the epsilon of both RMS norms.
pub(crate) fn engram_gate(
    stream: BTensor<2>,
    kv: BTensor<2>,
    gate: BTensor<1>,
    shape: &HcShape,
) -> BTensor<2> {
    let call = call_of(&stream, shape);
    assert_eq!(
        kv.dims(),
        [call.tokens, (shape.streams + 1) * shape.dim],
        "kv"
    );
    assert_eq!(gate.dims(), [shape.streams * shape.dim], "gate");
    BTensor::from_dispatch(<Dispatch as HcKernels>::engram_gate(
        stream.into_dispatch(),
        kv.into_dispatch(),
        gate.into_dispatch(),
        call,
    ))
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use burn::tensor::{Device, TensorData};

    use super::super::csa::HyperSpec;
    use super::super::gpu::{kernel_test_device, read_f32};
    use super::super::hc::{self, HcWeights};
    use super::*;

    /// A small deterministic generator (xorshift64*).
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }

        /// A float in `[-1, 1)`.
        fn signed(&mut self) -> f32 {
            ((self.next() >> 40) as f32 / (1u64 << 23) as f32) - 1.0
        }

        fn signed_vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
            (0..n).map(|_| self.signed() * scale).collect()
        }
    }

    /// One hyper-connection site: its constants and its host weights.
    struct Site {
        shape: HcShape,
        w: HcWeights,
    }

    impl Site {
        /// Weights whose mixes are of order one, offsets and scales like a
        /// trained site's.
        fn new(rng: &mut Rng, streams: usize, dim: usize, sinkhorn_iters: usize) -> Self {
            let width = streams * dim;
            let nm = (2 + streams) * streams;
            let a = 2.6 / (width as f32).sqrt();
            Site {
                shape: HcShape {
                    streams,
                    dim,
                    sinkhorn_iters,
                    hc_eps: 1e-6,
                    norm_eps: 1e-6,
                },
                w: HcWeights {
                    mix: rng.signed_vec(nm * width, a),
                    base: rng.signed_vec(nm, 1.0),
                    scale: [0.8, 0.6, 0.9],
                },
            }
        }

        fn spec(&self) -> HyperSpec {
            HyperSpec {
                streams: self.shape.streams,
                sinkhorn_iters: self.shape.sinkhorn_iters,
                eps: self.shape.hc_eps,
            }
        }

        fn width(&self) -> usize {
            self.shape.streams * self.shape.dim
        }

        fn cl(&self) -> usize {
            coeff_len(self.shape.streams)
        }
    }

    /// Streams of `t` tokens whose scale differs per token.
    fn streams(rng: &mut Rng, t: usize, width: usize) -> Vec<f32> {
        (0..t)
            .flat_map(|r| {
                let scale = 0.5 + r as f32 * 0.37;
                rng.signed_vec(width, scale)
            })
            .collect()
    }

    fn up2(device: &Device, data: &[f32], rows: usize) -> BTensor<2> {
        BTensor::<2>::from_data(
            TensorData::new(data.to_vec(), [rows, data.len() / rows]),
            device,
        )
    }

    fn up1(device: &Device, data: &[f32]) -> BTensor<1> {
        BTensor::<1>::from_data(TensorData::new(data.to_vec(), [data.len()]), device)
    }

    fn down(t: BTensor<2>) -> Vec<f32> {
        read_f32(t, "hc kernel").unwrap()
    }

    /// The largest `|got - want| / (|want| + 1e-3)`: relative where values
    /// are not tiny. Asserts `rel * |want| + abs`.
    fn assert_close(what: &str, got: &[f32], want: &[f32], rel: f32, abs: f32) -> f32 {
        assert_eq!(got.len(), want.len(), "{what}");
        let mut worst = 0f32;
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!(
                (g - w).abs() <= rel * w.abs() + abs,
                "{what}: element {i}: gpu {g:e} vs host {w:e}"
            );
            worst = worst.max((g - w).abs() / (w.abs() + 1e-3));
        }
        worst
    }

    /// The largest magnitude of the listed inputs' row `r`, for each of `rows`
    /// rows: `(values, row width)` pairs.
    fn row_scales(rows: usize, inputs: &[(&[f32], usize)]) -> Vec<f32> {
        (0..rows)
            .map(|r| {
                inputs
                    .iter()
                    .flat_map(|(v, w)| v[r * w..(r + 1) * w].iter())
                    .fold(0f32, |m, x| m.max(x.abs()))
            })
            .collect()
    }

    /// Asserts `|got - want| <= 1e-5 |want| + 16 eps scale` per element, with
    /// `scale` the magnitude of the element's row of inputs: a sum of terms
    /// that cancel is only as accurate as its largest term. Returns the
    /// worst error in units of `eps * scale`.
    fn assert_close_scaled(
        what: &str,
        got: &[f32],
        want: &[f32],
        row: usize,
        scales: &[f32],
    ) -> f32 {
        assert_eq!(got.len(), want.len(), "{what}");
        let mut worst = 0f32;
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            let unit = f32::EPSILON * scales[i / row];
            assert!(
                (g - w).abs() <= 1e-5 * w.abs() + 16.0 * unit,
                "{what}: element {i}: gpu {g:e} vs host {w:e} (scale {})",
                scales[i / row]
            );
            worst = worst.max((g - w).abs() / unit);
        }
        worst
    }

    fn assert_same_bits(what: &str, got: &[f32], want: &[f32]) {
        assert_eq!(got.len(), want.len(), "{what}");
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!(
                g.to_bits() == w.to_bits(),
                "{what}: element {i}: {g:e} vs {w:e}"
            );
        }
    }

    fn host_coefficients(site: &Site, x: &[f32]) -> Vec<f32> {
        let cl = site.cl();
        let mut out = vec![0f32; x.len() / site.width() * cl];
        for (row, c) in x.chunks_exact(site.width()).zip(out.chunks_exact_mut(cl)) {
            hc::coefficients(row, &site.w, &site.spec(), site.shape.norm_eps, c);
        }
        out
    }

    fn host_collapse(site: &Site, x: &[f32], coeffs: &[f32]) -> Vec<f32> {
        let (s, d, cl) = (site.shape.streams, site.shape.dim, site.cl());
        let mut out = vec![0f32; x.len() / site.width() * d];
        for ((row, c), o) in x
            .chunks_exact(site.width())
            .zip(coeffs.chunks_exact(cl))
            .zip(out.chunks_exact_mut(d))
        {
            hc::collapse(row, hc::pre(c, s), o);
        }
        out
    }

    fn host_mean(site: &Site, x: &[f32]) -> Vec<f32> {
        let (s, d) = (site.shape.streams, site.shape.dim);
        let mut out = vec![0f32; x.len() / site.width() * d];
        for (row, o) in x.chunks_exact(site.width()).zip(out.chunks_exact_mut(d)) {
            hc::collapse(row, &vec![1.0 / s as f32; s], o);
        }
        out
    }

    fn host_expand(site: &Site, y: &[f32], x: &[f32], coeffs: &[f32]) -> Vec<f32> {
        let (s, d, cl) = (site.shape.streams, site.shape.dim, site.cl());
        let mut out = vec![0f32; x.len()];
        for (r, o) in out.chunks_exact_mut(site.width()).enumerate() {
            let c = &coeffs[r * cl..(r + 1) * cl];
            hc::expand(
                &y[r * d..(r + 1) * d],
                &x[r * site.width()..(r + 1) * site.width()],
                &c[s..2 * s],
                &c[2 * s..],
                o,
            );
        }
        out
    }

    /// The inner loop of the CPU engine's `engram_layer`, verbatim.
    fn host_engram(site: &Site, h: &[f32], kv: &[f32], gate: &[f32]) -> Vec<f32> {
        let (s, d) = (site.shape.streams, site.shape.dim);
        let eps = site.shape.norm_eps;
        let width = (s + 1) * d;
        let scale = (d as f32).powf(-0.5);
        let mut out = h.to_vec();
        for (r, row) in out.chunks_exact_mut(s * d).enumerate() {
            let key = &kv[r * width..r * width + s * d];
            let value = &kv[r * width + s * d..(r + 1) * width];
            for (copy, (k, w)) in row
                .chunks_exact_mut(d)
                .zip(key.chunks_exact(d).zip(gate.chunks_exact(d)))
            {
                let rstd = (1.0 / (hc::dot(copy, copy) / d as f32 + eps).sqrt())
                    * (1.0 / (hc::dot(k, k) / d as f32 + eps).sqrt());
                let mut acc = 0.0;
                for i in 0..d {
                    acc += copy[i] * w[i] * k[i];
                }
                let x = acc * rstd * scale;
                let gate = 1.0 / (1.0 + (-x.abs().max(1e-6).sqrt().copysign(x)).exp());
                for (h, &v) in copy.iter_mut().zip(value) {
                    *h += gate * v;
                }
            }
        }
        out
    }

    /// (streams, dim, sinkhorn passes, tokens) the kernels are checked on:
    /// odd sizes, other stream counts, a width below the cube's units, the
    /// tiny model and the real model.
    const CASES: [(usize, usize, usize, &[usize]); 6] = [
        (2, 16, 3, &[1, 2]),
        (3, 40, 1, &[1, 4]),
        (4, 5, 20, &[3]),
        (4, 64, 20, &[1, 3, 9]),
        (4, 300, 20, &[2]),
        (4, 5120, 20, &[1, 3, 33]),
    ];

    /// The coefficients agree with `hc::coefficients` to 1e-5 relative.
    #[test]
    fn coefficients_match_the_host() {
        let Some(device) = kernel_test_device() else {
            return;
        };
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        for (s, d, iters, ts) in CASES {
            let site = Site::new(&mut rng, s, d, iters);
            let mix = up2(&device, &site.w.mix, (2 + s) * s);
            let base = up1(&device, &site.w.base);
            let scale = up1(&device, &site.w.scale);
            for &t in ts {
                let x = streams(&mut rng, t, site.width());
                let began = Instant::now();
                let got = down(hc_coefficients(
                    up2(&device, &x, t),
                    mix.clone(),
                    base.clone(),
                    scale.clone(),
                    &site.shape,
                ));
                let took = began.elapsed();
                let want = host_coefficients(&site, &x);
                let worst = assert_close(
                    &format!("coefficients S={s} D={d} t={t}"),
                    &got,
                    &want,
                    1e-5,
                    1e-7,
                );
                println!(
                    "coefficients S={s} D={d} t={t}: worst error {worst:.2e} ({took:?} with upload and readback)"
                );
            }
        }
    }

    /// Collapse, expand, the stream mean and repeat agree with the host
    /// functions, fed the host's coefficients.
    #[test]
    fn collapse_expand_mean_and_repeat_match_the_host() {
        let Some(device) = kernel_test_device() else {
            return;
        };
        let mut rng = Rng(0xD1B5_4A32_D192_ED03);
        for (s, d, iters, ts) in CASES {
            let site = Site::new(&mut rng, s, d, iters);
            for &t in ts {
                let x = streams(&mut rng, t, site.width());
                let y = rng.signed_vec(t * d, 2.0);
                let coeffs = host_coefficients(&site, &x);
                let what = format!("S={s} D={d} t={t}");

                let got = down(hc_collapse(
                    up2(&device, &x, t),
                    up2(&device, &coeffs, t),
                    &site.shape,
                ));
                let worst = assert_close_scaled(
                    &format!("collapse {what}"),
                    &got,
                    &host_collapse(&site, &x, &coeffs),
                    d,
                    &row_scales(t, &[(&x, site.width())]),
                );
                println!("collapse {what}: worst error {worst:.1} eps*scale");

                let got = down(hc_expand(
                    up2(&device, &y, t),
                    up2(&device, &x, t),
                    up2(&device, &coeffs, t),
                    &site.shape,
                ));
                let worst = assert_close_scaled(
                    &format!("expand {what}"),
                    &got,
                    &host_expand(&site, &y, &x, &coeffs),
                    site.width(),
                    &row_scales(t, &[(&x, site.width()), (&y, d)]),
                );
                println!("expand {what}: worst error {worst:.1} eps*scale");

                let got = down(stream_mean(up2(&device, &x, t), &site.shape));
                let worst = assert_close_scaled(
                    &format!("mean {what}"),
                    &got,
                    &host_mean(&site, &x),
                    d,
                    &row_scales(t, &[(&x, site.width())]),
                );
                println!("mean {what}: worst error {worst:.1} eps*scale");

                let got = down(stream_repeat(up2(&device, &y, t), &site.shape));
                let want: Vec<f32> = y
                    .chunks_exact(d)
                    .flat_map(|row| (0..s).flat_map(move |_| row.iter().copied()))
                    .collect();
                assert_same_bits(&format!("repeat {what}"), &got, &want);
            }
        }
    }

    /// The engram gate agrees with the CPU engine's inner loop, for both
    /// epsilons the model could use, and for a zero stream (the `1e-6` floor
    /// of the gate argument).
    #[test]
    fn engram_gate_matches_the_host() {
        let Some(device) = kernel_test_device() else {
            return;
        };
        let mut rng = Rng(0x1234_5678_9ABC_DEF1);
        for (s, d, iters, ts) in CASES {
            for eps in [1e-6f32, 1e-20] {
                let mut site = Site::new(&mut rng, s, d, iters);
                site.shape.norm_eps = eps;
                let gate = rng.signed_vec(s * d, 1.0);
                let gate_t = up1(&device, &gate);
                for &t in ts {
                    let mut x = streams(&mut rng, t, site.width());
                    // A zero token (when there are several).
                    if t > 1 {
                        x[site.width()..2 * site.width()].fill(0.0);
                    }
                    let kv: Vec<f32> = (0..t)
                        .flat_map(|r| rng.signed_vec((s + 1) * d, 0.5 + r as f32 * 0.2))
                        .collect();
                    let got = down(engram_gate(
                        up2(&device, &x, t),
                        up2(&device, &kv, t),
                        gate_t.clone(),
                        &site.shape,
                    ));
                    let want = host_engram(&site, &x, &kv, &gate);
                    let scales = row_scales(t, &[(&x, site.width()), (&kv, (s + 1) * d)]);
                    let worst = assert_close_scaled(
                        &format!("engram S={s} D={d} t={t} eps={eps:e}"),
                        &got,
                        &want,
                        site.width(),
                        &scales,
                    );
                    println!(
                        "engram S={s} D={d} t={t} eps={eps:e}: worst error {worst:.1} eps*scale"
                    );
                }
            }
        }
    }

    /// A token alone equals the same token inside a batch, bit for bit, at
    /// the real model's shape and at tiny ones.
    #[test]
    fn tokens_do_not_depend_on_the_batch() {
        let Some(device) = kernel_test_device() else {
            return;
        };
        let mut rng = Rng(0x0F1E_2D3C_4B5A_6978);
        for (s, d, iters, t) in [(4usize, 64usize, 20usize, 9usize), (4, 5120, 20, 33)] {
            let site = Site::new(&mut rng, s, d, iters);
            let (width, cl) = (site.width(), site.cl());
            let mix = up2(&device, &site.w.mix, (2 + s) * s);
            let base = up1(&device, &site.w.base);
            let scale = up1(&device, &site.w.scale);
            let gate = rng.signed_vec(width, 1.0);
            let x = streams(&mut rng, t, width);
            let y = rng.signed_vec(t * d, 2.0);
            let kv: Vec<f32> = (0..t)
                .flat_map(|r| rng.signed_vec((s + 1) * d, 0.5 + r as f32 * 0.2))
                .collect();

            // The batch, once: each stage reads the one before.
            let coeffs_t = hc_coefficients(
                up2(&device, &x, t),
                mix.clone(),
                base.clone(),
                scale.clone(),
                &site.shape,
            );
            let coeffs = down(coeffs_t);
            let collapsed = down(hc_collapse(
                up2(&device, &x, t),
                up2(&device, &coeffs, t),
                &site.shape,
            ));
            let expanded = down(hc_expand(
                up2(&device, &y, t),
                up2(&device, &x, t),
                up2(&device, &coeffs, t),
                &site.shape,
            ));
            let mean = down(stream_mean(up2(&device, &x, t), &site.shape));
            let repeated = down(stream_repeat(up2(&device, &y, t), &site.shape));
            let gated = down(engram_gate(
                up2(&device, &x, t),
                up2(&device, &kv, t),
                up1(&device, &gate),
                &site.shape,
            ));

            for r in [0, t / 2, t - 1] {
                let row = |v: &[f32], w: usize| v[r * w..(r + 1) * w].to_vec();
                let xr = row(&x, width);
                let alone = down(hc_coefficients(
                    up2(&device, &xr, 1),
                    mix.clone(),
                    base.clone(),
                    scale.clone(),
                    &site.shape,
                ));
                assert_same_bits(&format!("coefficients row {r}"), &alone, &row(&coeffs, cl));
                let alone = down(hc_collapse(
                    up2(&device, &xr, 1),
                    up2(&device, &row(&coeffs, cl), 1),
                    &site.shape,
                ));
                assert_same_bits(&format!("collapse row {r}"), &alone, &row(&collapsed, d));
                let alone = down(hc_expand(
                    up2(&device, &row(&y, d), 1),
                    up2(&device, &xr, 1),
                    up2(&device, &row(&coeffs, cl), 1),
                    &site.shape,
                ));
                assert_same_bits(&format!("expand row {r}"), &alone, &row(&expanded, width));
                let alone = down(stream_mean(up2(&device, &xr, 1), &site.shape));
                assert_same_bits(&format!("mean row {r}"), &alone, &row(&mean, d));
                let alone = down(stream_repeat(up2(&device, &row(&y, d), 1), &site.shape));
                assert_same_bits(&format!("repeat row {r}"), &alone, &row(&repeated, width));
                let alone = down(engram_gate(
                    up2(&device, &xr, 1),
                    up2(&device, &row(&kv, (s + 1) * d), 1),
                    up1(&device, &gate),
                    &site.shape,
                ));
                assert_same_bits(&format!("engram row {r}"), &alone, &row(&gated, width));
            }
        }
    }

    /// The same `[rows, cols]` data as a contiguous tensor, as the transpose
    /// of a `[cols, rows]` one (swapped strides) and as a window of a taller
    /// tensor (an offset): the kernels must read all three the same way.
    fn forms(device: &Device, data: &[f32], rows: usize) -> [BTensor<2>; 3] {
        let cols = data.len() / rows;
        let transposed: Vec<f32> = (0..cols)
            .flat_map(|c| (0..rows).map(move |r| data[r * cols + c]))
            .collect();
        let mut taller = vec![f32::NAN; 2 * cols];
        taller.extend_from_slice(data);
        taller.extend(std::iter::repeat_n(f32::NAN, cols));
        [
            up2(device, data, rows),
            BTensor::<2>::from_data(TensorData::new(transposed, [cols, rows]), device).transpose(),
            up2(device, &taller, rows + 3).narrow(0, 2, rows),
        ]
    }

    /// Inputs that are views (the transpose of a matmul output, a window of
    /// a longer tensor) give the results of contiguous inputs, bit for bit.
    #[test]
    fn strided_inputs_are_read_like_contiguous_ones() {
        let Some(device) = kernel_test_device() else {
            return;
        };
        let mut rng = Rng(0xABCD_EF01_2345_6789);
        let (s, d, t) = (4usize, 64usize, 5usize);
        let site = Site::new(&mut rng, s, d, 20);
        let width = site.width();
        let mix = up2(&device, &site.w.mix, (2 + s) * s);
        let base = up1(&device, &site.w.base);
        let scale = up1(&device, &site.w.scale);
        let gate = up1(&device, &rng.signed_vec(width, 1.0));
        let x = streams(&mut rng, t, width);
        let y = rng.signed_vec(t * d, 2.0);
        let kv = rng.signed_vec(t * (s + 1) * d, 0.5);
        let coeffs = host_coefficients(&site, &x);
        let (xs, ys, kvs, cs) = (
            forms(&device, &x, t),
            forms(&device, &y, t),
            forms(&device, &kv, t),
            forms(&device, &coeffs, t),
        );
        let want_coeffs = down(hc_coefficients(
            xs[0].clone(),
            mix.clone(),
            base.clone(),
            scale.clone(),
            &site.shape,
        ));
        let want_collapse = down(hc_collapse(xs[0].clone(), cs[0].clone(), &site.shape));
        let want_expand = down(hc_expand(
            ys[0].clone(),
            xs[0].clone(),
            cs[0].clone(),
            &site.shape,
        ));
        let want_mean = down(stream_mean(xs[0].clone(), &site.shape));
        let want_repeat = down(stream_repeat(ys[0].clone(), &site.shape));
        let want_gate = down(engram_gate(
            xs[0].clone(),
            kvs[0].clone(),
            gate.clone(),
            &site.shape,
        ));
        for f in 1..3 {
            let got = down(hc_coefficients(
                xs[f].clone(),
                mix.clone(),
                base.clone(),
                scale.clone(),
                &site.shape,
            ));
            assert_same_bits(&format!("coefficients form {f}"), &got, &want_coeffs);
            let got = down(hc_collapse(xs[f].clone(), cs[f].clone(), &site.shape));
            assert_same_bits(&format!("collapse form {f}"), &got, &want_collapse);
            let got = down(hc_expand(
                ys[f].clone(),
                xs[f].clone(),
                cs[f].clone(),
                &site.shape,
            ));
            assert_same_bits(&format!("expand form {f}"), &got, &want_expand);
            let got = down(stream_mean(xs[f].clone(), &site.shape));
            assert_same_bits(&format!("mean form {f}"), &got, &want_mean);
            let got = down(stream_repeat(ys[f].clone(), &site.shape));
            assert_same_bits(&format!("repeat form {f}"), &got, &want_repeat);
            let got = down(engram_gate(
                xs[f].clone(),
                kvs[f].clone(),
                gate.clone(),
                &site.shape,
            ));
            assert_same_bits(&format!("engram form {f}"), &got, &want_gate);
        }
    }

    /// A named launch of one kernel on prepared inputs.
    type Op = (&'static str, Box<dyn Fn() -> BTensor<2>>);

    /// Time per launch at the real shape (`S = 4`, `D = 5120`), for the
    /// integrator's budget: one launch followed by its readback, and the
    /// pipelined cost of launches queued back to back.
    #[test]
    fn real_shape_timings() {
        let Some(device) = kernel_test_device() else {
            return;
        };
        let mut rng = Rng(0x5EED_5EED_5EED_5EED);
        let site = Site::new(&mut rng, 4, 5120, 20);
        let (s, d, width) = (4usize, 5120usize, site.width());
        let mix = up2(&device, &site.w.mix, (2 + s) * s);
        let base = up1(&device, &site.w.base);
        let scale = up1(&device, &site.w.scale);
        let gate = up1(&device, &rng.signed_vec(width, 1.0));
        for t in [1usize, 33] {
            let x = up2(&device, &streams(&mut rng, t, width), t);
            let y = up2(&device, &rng.signed_vec(t * d, 2.0), t);
            let kv = up2(&device, &rng.signed_vec(t * (s + 1) * d, 0.5), t);
            let coeffs = hc_coefficients(
                x.clone(),
                mix.clone(),
                base.clone(),
                scale.clone(),
                &site.shape,
            );
            let _ = down(coeffs.clone());
            let ops: [Op; 6] = [
                (
                    "coefficients",
                    Box::new({
                        let (x, mix, base, scale, shape) = (
                            x.clone(),
                            mix.clone(),
                            base.clone(),
                            scale.clone(),
                            site.shape,
                        );
                        move || {
                            hc_coefficients(
                                x.clone(),
                                mix.clone(),
                                base.clone(),
                                scale.clone(),
                                &shape,
                            )
                        }
                    }),
                ),
                (
                    "collapse",
                    Box::new({
                        let (x, c, shape) = (x.clone(), coeffs.clone(), site.shape);
                        move || hc_collapse(x.clone(), c.clone(), &shape)
                    }),
                ),
                (
                    "expand",
                    Box::new({
                        let (y, x, c, shape) = (y.clone(), x.clone(), coeffs.clone(), site.shape);
                        move || hc_expand(y.clone(), x.clone(), c.clone(), &shape)
                    }),
                ),
                (
                    "stream_mean",
                    Box::new({
                        let (x, shape) = (x.clone(), site.shape);
                        move || stream_mean(x.clone(), &shape)
                    }),
                ),
                (
                    "stream_repeat",
                    Box::new({
                        let (y, shape) = (y.clone(), site.shape);
                        move || stream_repeat(y.clone(), &shape)
                    }),
                ),
                (
                    "engram_gate",
                    Box::new({
                        let (x, kv, gate, shape) =
                            (x.clone(), kv.clone(), gate.clone(), site.shape);
                        move || engram_gate(x.clone(), kv.clone(), gate.clone(), &shape)
                    }),
                ),
            ];
            for (name, op) in &ops {
                let _ = down(op());
                let began = Instant::now();
                let _ = down(op());
                let single = began.elapsed();
                const N: u32 = 50;
                let began = Instant::now();
                let outs: Vec<BTensor<2>> = (0..N).map(|_| op()).collect();
                let last = outs.into_iter().last().unwrap();
                let _ = down(last);
                let piped = began.elapsed() / N;
                println!(
                    "t={t:2} {name:13}: one launch + readback {single:?}, pipelined {piped:?} per launch"
                );
            }
        }
    }
}
