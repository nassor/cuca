//! Block quantize-dequantize kernels of the GPU engine (CubeCL): the device
//! twins of [`kvquant`](super::kvquant).
//!
//! A model trained with quantized activations expects the attention path to
//! carry the rounding error of the training run's grids. These kernels do on
//! the device what the host loop does, returning a new `[.., cols]` tensor:
//! every run of `block` consecutive values along the last dimension gets a
//! scale from its largest magnitude, is rounded to the nearest point of the
//! grid (ties to even) and comes back as `f32`:
//!
//! - [`qdq_fp8_ue8m0`]: FP8 `e4m3` values, a power-of-two scale (the smallest
//!   at or above `amax / 448`);
//! - [`qdq_fp4_e8m0`]: FP4 `e2m1` values, a power-of-two scale (the smallest
//!   at or above `amax / 6`);
//! - [`qdq_fp4_e4m3`]: FP4 `e2m1` values, an `e4m3` scale (`amax / 6` rounded
//!   to the E4M3 grid).
//!
//! One unit handles one block (two passes over its values: the maximum, then
//! the rounding), so a block's result never depends on the rest of the
//! launch. The power-of-two scale is built from the exponent bits and every
//! value is divided by a power of two, so [`qdq_fp8_ue8m0`] and
//! [`qdq_fp4_e8m0`] are bit-exact against the host functions. The rounding
//! onto the grids avoids the float rounding mode (floor and exact
//! comparisons only). The `e4m3` scale of [`qdq_fp4_e4m3`] is not a power of
//! two, so its divisions (`amax / 6`, `value / scale`) rely on the device's
//! division: a rare one-step difference from the host is possible there and
//! only there.
//!
//! The kernels are written in safe Rust (`#[cube(launch)]` with the checked
//! `launch`) and reach the backend through burn's `backend_extension`, like
//! the [`moe_kernel`](super::moe_kernel) ones. Denormal results (values below
//! `2^-126` after scaling) are outside the contract: a device may flush them.

// The `#[cube]` kernels are Rust the CubeCL macro translates: it needs runtime
// values built from runtime values, plain `x = x + 1`, nested `if`s and limits
// written as `if`s.
#![allow(
    clippy::eq_op,
    clippy::assign_op_pattern,
    clippy::collapsible_if,
    clippy::manual_clamp,
    clippy::manual_div_ceil,
    clippy::too_many_arguments
)]

use burn::backend::{TensorMetadata, backend_extension, tensor::FloatTensor};
use burn::cubecl::prelude::*;
use burn::cubecl::{CubeDim, calculate_cube_count_elemwise};
use burn::tensor::DType;
use burn_cubecl::kernel::into_contiguous;
use burn_cubecl::ops::numeric::empty_device_dtype;
use burn_cubecl::{CubeBackend, tensor::CubeTensor};

/// `1 / 448` rounded to `f32`: the FP8 scale divides `amax` by the largest
/// `e4m3` value this way.
const FP8_INV_MAX: f32 = (1.0 / 448.0) as f32;
/// The smallest `amax` of the FP8 variant.
const FP8_MIN_AMAX: f32 = 1e-4;
/// `1 / 6` rounded to `f32` (the power-of-two FP4 scale).
const FP4_INV_MAX: f32 = (1.0 / 6.0) as f32;
/// The smallest `amax` of the power-of-two FP4 variant.
const FP4_E8M0_MIN_AMAX: f32 = 6.0 * f32::MIN_POSITIVE;
/// The smallest `amax` of the `e4m3`-scale FP4 variant.
const FP4_E4M3_MIN_AMAX: f32 = 6.0 / 512.0;

/// The grids a call quantizes to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum QdqKind {
    /// FP8 `e4m3` values, power-of-two scale.
    Fp8Ue8m0,
    /// FP4 `e2m1` values, power-of-two scale.
    Fp4E8m0,
    /// FP4 `e2m1` values, `e4m3` scale.
    Fp4E4m3,
}

/// What one launch is about.
#[derive(Clone, Debug)]
pub(crate) struct QdqCall {
    pub(crate) kind: QdqKind,
    /// Values per scale.
    pub(crate) block: usize,
}

/// `2^ceil(log2 v)` for a positive `v`, from the exponent bits
/// (`kvquant::pow2_ceil`): the exponent field plus one when the mantissa is
/// not zero, clamped to the normal range.
#[cube]
fn pow2_ceil(v: f32) -> f32 {
    let bits = u32::reinterpret(v);
    let mut e = (bits >> 23u32) & 0xFFu32;
    if (bits & 0x7F_FFFFu32) != 0u32 {
        e = e + 1u32;
    }
    if e < 1u32 {
        e = 1u32;
    }
    if e > 254u32 {
        e = 254u32;
    }
    f32::reinterpret(e << 23u32)
}

/// Round `q >= 0` (below `2^23`) to an integer, ties to even, with floor and
/// exact differences only.
#[cube]
fn round_ties_even(q: f32) -> f32 {
    let n = q.floor();
    let d = q - n;
    let half = (n * 0.5f32).floor();
    let odd = n - half - half;
    let mut r = n;
    if d > 0.5f32 || (d == 0.5f32 && odd > 0.5f32) {
        r = n + 1.0f32;
    }
    r
}

/// `x` rounded to the nearest `e4m3fn` value (ties to even, subnormals
/// included), sign kept (`kvquant::round_e4m3`).
#[cube]
fn round_e4m3(x: f32) -> f32 {
    let bits = u32::reinterpret(x);
    let sign = bits & 0x8000_0000u32;
    let mag_bits = bits & 0x7FFF_FFFFu32;
    let a = f32::reinterpret(mag_bits);
    // The biased exponent, at least that of the smallest normal (2^-6): the
    // spacing is 2^(e - 3).
    let mut e = mag_bits >> 23u32;
    if e < 121u32 {
        e = 121u32;
    }
    let quantum = f32::reinterpret((e - 3u32) << 23u32);
    let r = round_ties_even(a / quantum) * quantum;
    f32::reinterpret(u32::reinterpret(r) | sign)
}

/// `q` (`|q| <= 6`) rounded to the nearest `e2m1` value (`0 .5 1 1.5 2 3 4
/// 6`), ties to even, with the sign of `v` (`kvquant::round_e2m1`: the
/// boundary between codes `k` and `k + 1` belongs to the even code, the upper
/// one for odd `k`).
#[cube]
fn round_e2m1(q: f32, v: f32) -> f32 {
    let m = f32::abs(q);
    let mut r = m - m;
    if m > 0.25f32 {
        r = 0.5f32;
    }
    if m >= 0.75f32 {
        r = 1.0f32;
    }
    if m > 1.25f32 {
        r = 1.5f32;
    }
    if m >= 1.75f32 {
        r = 2.0f32;
    }
    if m > 2.5f32 {
        r = 3.0f32;
    }
    if m >= 3.5f32 {
        r = 4.0f32;
    }
    if m > 5.0f32 {
        r = 6.0f32;
    }
    let sign = u32::reinterpret(v) & 0x8000_0000u32;
    f32::reinterpret(u32::reinterpret(r) | sign)
}

/// `q` limited to `[-lim, lim]`.
#[cube]
fn clamp_to(q: f32, lim: f32) -> f32 {
    let mut r = q;
    if q > lim {
        r = lim;
    }
    if q < -lim {
        r = -lim;
    }
    r
}

/// The largest magnitude of the `block` values from `at`, at least `lowest`.
#[cube]
fn block_amax(x: &Tensor<f32>, at: usize, lowest: f32, #[comptime] block: usize) -> f32 {
    let mut m = lowest - lowest;
    for i in 0..block {
        let a = f32::abs(x[at + i]);
        if a > m {
            m = a;
        }
    }
    if lowest > m {
        m = lowest;
    }
    m
}

/// FP8 `e4m3` values, power-of-two scale: one unit per block.
#[cube(launch)]
fn qdq_fp8_kernel(
    x: &Tensor<f32>,
    out: &mut Tensor<f32>,
    blocks: u32,
    lowest: f32,
    inv_max: f32,
    #[comptime] block: usize,
) {
    let b = ABSOLUTE_POS;
    if b < usize::cast_from(blocks) {
        let at = b * block;
        let amax = block_amax(x, at, lowest, block);
        let s = pow2_ceil(amax * inv_max);
        for i in 0..block {
            let q = clamp_to(x[at + i] / s, 448.0f32);
            out[at + i] = round_e4m3(q) * s;
        }
    }
}

/// FP4 `e2m1` values, power-of-two scale: one unit per block.
#[cube(launch)]
fn qdq_fp4_e8m0_kernel(
    x: &Tensor<f32>,
    out: &mut Tensor<f32>,
    blocks: u32,
    lowest: f32,
    inv_max: f32,
    #[comptime] block: usize,
) {
    let b = ABSOLUTE_POS;
    if b < usize::cast_from(blocks) {
        let at = b * block;
        let amax = block_amax(x, at, lowest, block);
        let s = pow2_ceil(amax * inv_max);
        for i in 0..block {
            let v = x[at + i];
            out[at + i] = round_e2m1(clamp_to(v / s, 6.0f32), v) * s;
        }
    }
}

/// FP4 `e2m1` values, `e4m3` scale: one unit per block. `six` is passed as a
/// runtime value so that `amax / six` stays a division.
#[cube(launch)]
fn qdq_fp4_e4m3_kernel(
    x: &Tensor<f32>,
    out: &mut Tensor<f32>,
    blocks: u32,
    lowest: f32,
    six: f32,
    #[comptime] block: usize,
) {
    let b = ABSOLUTE_POS;
    if b < usize::cast_from(blocks) {
        let at = b * block;
        let amax = block_amax(x, at, lowest, block);
        let s = round_e4m3(amax / six);
        for i in 0..block {
            let v = x[at + i];
            out[at + i] = round_e2m1(clamp_to(v / s, six), v) * s;
        }
    }
}

// --------------------------------------------------------------- extension

/// Backend operations of the block quantizers (see the module docs).
#[backend_extension(Cube, Fusion: cfg(feature = "provider-local-gpu"))]
pub trait QdqKernels: burn::backend::Backend {
    /// `x` rounded to the grid of `call.kind` per block of `call.block`
    /// consecutive values; the result has the shape of `x`.
    #[fusion(dtype = x, shape = x)]
    fn block_qdq(x: FloatTensor<Self>, call: QdqCall) -> FloatTensor<Self>;
}

impl QdqKernels for CubeBackend {
    fn block_qdq(x: CubeTensor, call: QdqCall) -> CubeTensor {
        // The kernels index the buffer linearly.
        let x = into_contiguous(x);
        let total = x.shape().num_elements();
        let out = empty_device_dtype(x.client.clone(), x.device.clone(), x.shape(), DType::F32);
        let blocks = total / call.block;
        if blocks == 0 {
            return out;
        }
        let client = x.client.clone();
        let dim = CubeDim::new(&client, blocks);
        let count = calculate_cube_count_elemwise(&client, blocks, dim);
        match call.kind {
            QdqKind::Fp8Ue8m0 => qdq_fp8_kernel::launch(
                &client,
                count,
                dim,
                x.into_tensor_arg(),
                out.clone().into_tensor_arg(),
                blocks as u32,
                FP8_MIN_AMAX,
                FP8_INV_MAX,
                call.block,
            ),
            QdqKind::Fp4E8m0 => qdq_fp4_e8m0_kernel::launch(
                &client,
                count,
                dim,
                x.into_tensor_arg(),
                out.clone().into_tensor_arg(),
                blocks as u32,
                FP4_E8M0_MIN_AMAX,
                FP4_INV_MAX,
                call.block,
            ),
            QdqKind::Fp4E4m3 => qdq_fp4_e4m3_kernel::launch(
                &client,
                count,
                dim,
                x.into_tensor_arg(),
                out.clone().into_tensor_arg(),
                blocks as u32,
                FP4_E4M3_MIN_AMAX,
                6.0,
                call.block,
            ),
        }
        out
    }
}

// ---------------------------------------------------------- tensor wrappers

use burn::backend::Dispatch;
use burn::tensor::Tensor as BTensor;

/// [`QdqKernels::block_qdq`] on a tensor; panics when the last dimension is
/// not a multiple of `block` (the host functions chunk exactly too).
fn block_qdq<const D: usize>(x: BTensor<D>, kind: QdqKind, block: usize) -> BTensor<D> {
    let cols = x.dims()[D - 1];
    assert!(
        block > 0 && cols.is_multiple_of(block),
        "the last dimension {cols} is not a multiple of the block {block}"
    );
    BTensor::from_dispatch(<Dispatch as QdqKernels>::block_qdq(
        x.into_dispatch(),
        QdqCall { kind, block },
    ))
}

/// `kvquant::qdq_fp8_ue8m0` on the last dimension of `x`.
pub(crate) fn qdq_fp8_ue8m0<const D: usize>(x: BTensor<D>, block: usize) -> BTensor<D> {
    block_qdq(x, QdqKind::Fp8Ue8m0, block)
}

/// `kvquant::qdq_fp4_e8m0` on the last dimension of `x`.
pub(crate) fn qdq_fp4_e8m0<const D: usize>(x: BTensor<D>, block: usize) -> BTensor<D> {
    block_qdq(x, QdqKind::Fp4E8m0, block)
}

/// `kvquant::qdq_fp4_e4m3` on the last dimension of `x`.
pub(crate) fn qdq_fp4_e4m3<const D: usize>(x: BTensor<D>, block: usize) -> BTensor<D> {
    block_qdq(x, QdqKind::Fp4E4m3, block)
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use burn::tensor::{Device, TensorData};

    use super::super::gpu::{kernel_test_device, read_f32};
    use super::super::kvquant;
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

        fn below(&mut self, n: usize) -> usize {
            (self.next() >> 33) as usize % n
        }
    }

    /// The training-kernel golden inputs of `kvquant`'s tests.
    fn golden_input(mul: f64) -> Vec<f32> {
        (0..160)
            .map(|i: i64| {
                ((((i * 7919) % 2003) - 1001) as f64 / 97.0 * (1 + (i / 32) % 5) as f64 * mul)
                    as f32
            })
            .collect()
    }

    /// Blocks of `block` values with very different magnitudes: ordinary
    /// ones, all-zero, tiny (below every minimum amax), huge, with signed
    /// zeros sprinkled in.
    fn random_input(rng: &mut Rng, blocks: usize, block: usize) -> Vec<f32> {
        let mut x = Vec::with_capacity(blocks * block);
        for _ in 0..blocks {
            let kind = rng.below(16);
            let scale = match kind {
                0 => 0.0,
                1 => 1e-6,
                2 => 3e5,
                _ => 2f32.powi(rng.below(17) as i32 - 8),
            };
            for _ in 0..block {
                let v = rng.signed() * scale;
                x.push(match rng.below(40) {
                    0 => 0.0,
                    1 => -0.0,
                    _ => v,
                });
            }
        }
        x
    }

    /// The next float above `v` (`v > 0`) and the one below.
    fn up(v: f32) -> f32 {
        f32::from_bits(v.to_bits() + 1)
    }

    fn down(v: f32) -> f32 {
        f32::from_bits(v.to_bits() - 1)
    }

    /// Blocks of the FP4 grid's decision points, exactly and one float to
    /// either side, scaled by a power of two chosen so that the first value
    /// of the block (`5 * 2^k`, itself a point) fixes the scale at `2^k`.
    fn fp4_tie_input(rng: &mut Rng, blocks: usize, block: usize) -> Vec<f32> {
        const POINTS: [f32; 7] = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0];
        let mut x = Vec::with_capacity(blocks * block);
        for _ in 0..blocks {
            let k = rng.below(21) as i32 - 10;
            let s = 2f32.powi(k);
            x.push(5.0 * s);
            for _ in 1..block {
                let p = POINTS[rng.below(POINTS.len())] * s;
                let v = match rng.below(4) {
                    0 => up(p),
                    1 => down(p),
                    _ => p,
                };
                x.push(if rng.below(2) == 0 { v } else { -v });
            }
        }
        x
    }

    /// Blocks of `e4m3` midpoints (`(n + 1/2)` steps of every binade),
    /// scaled so that the first value (`300 * 2^k`) fixes the scale at `2^k`.
    fn fp8_tie_input(rng: &mut Rng, blocks: usize, block: usize) -> Vec<f32> {
        let mut x = Vec::with_capacity(blocks * block);
        for _ in 0..blocks {
            let k = rng.below(21) as i32 - 10;
            let s = 2f32.powi(k);
            x.push(300.0 * s);
            for _ in 1..block {
                let e = rng.below(15) as i32 - 7;
                let (n, quantum) = if e == -7 {
                    (rng.below(8), 2f32.powi(-9))
                } else {
                    (8 + rng.below(8), 2f32.powi(e - 3))
                };
                let v = (n as f32 + 0.5) * quantum * s;
                let v = match rng.below(5) {
                    0 => up(v),
                    1 => down(v),
                    _ => v,
                };
                x.push(if rng.below(2) == 0 { v } else { -v });
            }
        }
        x
    }

    fn host(kind: QdqKind, block: usize, x: &[f32]) -> Vec<f32> {
        let mut v = x.to_vec();
        match kind {
            QdqKind::Fp8Ue8m0 => kvquant::qdq_fp8_ue8m0(&mut v, block),
            QdqKind::Fp4E8m0 => kvquant::qdq_fp4_e8m0(&mut v, block),
            QdqKind::Fp4E4m3 => kvquant::qdq_fp4_e4m3(&mut v, block),
        }
        v
    }

    fn device_qdq(
        device: &Device,
        kind: QdqKind,
        block: usize,
        rows: usize,
        x: &[f32],
    ) -> Vec<f32> {
        let cols = x.len() / rows;
        let t = BTensor::<2>::from_data(TensorData::new(x.to_vec(), [rows, cols]), device);
        let y = match kind {
            QdqKind::Fp8Ue8m0 => qdq_fp8_ue8m0(t, block),
            QdqKind::Fp4E8m0 => qdq_fp4_e8m0(t, block),
            QdqKind::Fp4E4m3 => qdq_fp4_e4m3(t, block),
        };
        read_f32(y, "qdq").unwrap()
    }

    fn assert_bits(what: &str, x: &[f32], got: &[f32], want: &[f32]) {
        assert_eq!(got.len(), want.len(), "{what}");
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!(
                g.to_bits() == w.to_bits(),
                "{what}: element {i} (input {:e}): gpu {g:e} vs host {w:e}",
                x[i]
            );
        }
    }

    /// The power-of-two variants are bit-exact against `kvquant`: the
    /// training goldens, random blocks of every magnitude and blocks made of
    /// the grids' decision points (the tie rule and its neighbours).
    #[test]
    fn power_of_two_scales_are_bit_exact() {
        let Some(device) = kernel_test_device() else {
            return;
        };
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        let cases: [(QdqKind, usize); 2] = [(QdqKind::Fp8Ue8m0, 32), (QdqKind::Fp4E8m0, 32)];
        for (kind, block) in cases {
            let mul = if kind == QdqKind::Fp8Ue8m0 { 1.0 } else { 0.31 };
            let golden = golden_input(mul);
            let got = device_qdq(&device, kind, block, 1, &golden);
            assert_bits(
                &format!("{kind:?} golden"),
                &golden,
                &got,
                &host(kind, block, &golden),
            );
            // The same values as 5 rows of one block.
            let got = device_qdq(&device, kind, block, 5, &golden);
            assert_bits(
                &format!("{kind:?} golden rows"),
                &golden,
                &got,
                &host(kind, block, &golden),
            );
            // Other block sizes the model may use.
            for b in [16usize, 8] {
                let got = device_qdq(&device, kind, b, 5, &golden);
                assert_bits(
                    &format!("{kind:?} golden block {b}"),
                    &golden,
                    &got,
                    &host(kind, b, &golden),
                );
            }
            for (name, x) in [
                ("random", random_input(&mut rng, 4096, block)),
                (
                    "ties",
                    if kind == QdqKind::Fp8Ue8m0 {
                        fp8_tie_input(&mut rng, 2048, block)
                    } else {
                        fp4_tie_input(&mut rng, 2048, block)
                    },
                ),
            ] {
                let rows = 8;
                let got = device_qdq(&device, kind, block, rows, &x);
                assert_bits(
                    &format!("{kind:?} {name}"),
                    &x,
                    &got,
                    &host(kind, block, &x),
                );
                // Idempotence.
                let again = device_qdq(&device, kind, block, rows, &got);
                assert_bits(&format!("{kind:?} {name} idempotent"), &got, &again, &got);
            }
        }
    }

    /// The `e4m3`-scale variant divides by a scale that is not a power of
    /// two: the device's division may differ from the host's by a step in
    /// rare elements. The mismatch rate stays below 0.1%, the golden is
    /// matched, and the result is idempotent.
    #[test]
    fn e4m3_scale_matches_up_to_rare_division_steps() {
        let Some(device) = kernel_test_device() else {
            return;
        };
        let mut rng = Rng(0xD1B5_4A32_D192_ED03);
        let golden = golden_input(0.043);
        let got = device_qdq(&device, QdqKind::Fp4E4m3, 16, 1, &golden);
        let want = host(QdqKind::Fp4E4m3, 16, &golden);
        let off = got
            .iter()
            .zip(&want)
            .filter(|(g, w)| g.to_bits() != w.to_bits())
            .count();
        assert!(off <= 1, "golden: {off} of {} differ", golden.len());
        for block in [16usize, 32] {
            let x = random_input(&mut rng, 8192, block);
            let got = device_qdq(&device, QdqKind::Fp4E4m3, block, 8, &x);
            let want = host(QdqKind::Fp4E4m3, block, &x);
            let off = got
                .iter()
                .zip(&want)
                .filter(|(g, w)| g.to_bits() != w.to_bits())
                .count();
            println!("e4m3 scale, block {block}: {off} of {} differ", x.len());
            assert!(
                off * 1000 < x.len(),
                "block {block}: {off} of {} elements differ",
                x.len()
            );
            let again = device_qdq(&device, QdqKind::Fp4E4m3, block, 8, &got);
            assert_bits("e4m3 idempotent", &got, &again, &got);
        }
    }

    /// A row alone equals the same row in a batch, bit for bit, on the
    /// shapes of the real model (attention output `[t, 64 * 512]`, indexer
    /// queries `[t, 32 * 128]`, compressed rows `[t, 512]`); and a 3-D
    /// tensor quantizes along its last dimension.
    #[test]
    fn rows_do_not_depend_on_the_batch() {
        let Some(device) = kernel_test_device() else {
            return;
        };
        let mut rng = Rng(0x1234_5678_9ABC_DEF1);
        let cases: [(QdqKind, usize, usize); 3] = [
            (QdqKind::Fp8Ue8m0, 32, 64 * 512),
            (QdqKind::Fp4E8m0, 32, 32 * 128),
            (QdqKind::Fp4E4m3, 16, 512),
        ];
        for (kind, block, cols) in cases {
            let t = 33;
            let x = random_input(&mut rng, t * cols / block, block);
            let began = Instant::now();
            let all = device_qdq(&device, kind, block, t, &x);
            let batch = began.elapsed();
            let began = Instant::now();
            let first = device_qdq(&device, kind, block, 1, &x[..cols]);
            let single = began.elapsed();
            println!(
                "{kind:?} block {block} [{t}, {cols}]: batch {batch:?} (with upload), one row {single:?}"
            );
            assert_bits(&format!("{kind:?} row 0"), &x[..cols], &first, &all[..cols]);
            for r in [t / 2, t - 1] {
                let one = device_qdq(&device, kind, block, 1, &x[r * cols..(r + 1) * cols]);
                assert_bits(
                    &format!("{kind:?} row {r}"),
                    &x[r * cols..(r + 1) * cols],
                    &one,
                    &all[r * cols..(r + 1) * cols],
                );
            }
            if kind != QdqKind::Fp4E4m3 {
                assert_bits(
                    &format!("{kind:?} vs host"),
                    &x,
                    &all,
                    &host(kind, block, &x),
                );
            }
        }
        let x = random_input(&mut rng, 2 * 3 * 64 / 32, 32);
        let t = BTensor::<3>::from_data(TensorData::new(x.clone(), [2, 3, 64]), &device);
        let got = read_f32(qdq_fp8_ue8m0(t, 32), "qdq 3-d").unwrap();
        assert_bits("3-d", &x, &got, &host(QdqKind::Fp8Ue8m0, 32, &x));
    }
}
