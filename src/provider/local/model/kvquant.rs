//! Simulated quantization of the attention path of a model trained with
//! quantized activations: values are rounded to the grid the training run
//! used and returned as `f32`, so the cached rows, the indexer's vectors and
//! the attention output carry exactly the error the model learned to live
//! with.
//!
//! Every function works in place on consecutive blocks of the slice (a
//! block's scale is derived from its largest magnitude) and rounds to nearest
//! with ties to even on the target grid. The scale formulas follow the
//! training kernels: a power of two (`ue8m0` / `e8m0`) is the smallest one at
//! or above `amax / grid_max`.

/// `2^ceil(log2 v)` for a positive normal `v`.
fn pow2_ceil(v: f32) -> f32 {
    let bits = v.to_bits();
    let exp = ((bits >> 23) & 0xff) as i32;
    if exp == 0 {
        // Subnormal: the smallest normal power of two is above it.
        return f32::MIN_POSITIVE;
    }
    let k = exp - 127 + i32::from(bits & 0x7f_ffff != 0);
    f32::from_bits(((k + 127).clamp(1, 254) as u32) << 23)
}

/// Round to the nearest `e4m3fn` value (ties to even, subnormals included);
/// the magnitude must not exceed 448.
fn round_e4m3(x: f32) -> f32 {
    let a = x.abs();
    if a == 0.0 {
        return x;
    }
    // Exponent of the value, at least that of the smallest normal (-6): the
    // spacing is 2^(e - 3).
    let e = (((a.to_bits() >> 23) & 0xff) as i32 - 127).max(-6);
    let quantum = f32::from_bits(((e - 3 + 127) as u32) << 23);
    ((a / quantum).round_ties_even() * quantum).copysign(x)
}

/// Round `|q| <= 6` to the nearest `e2m1` value (`0 .5 1 1.5 2 3 4 6`), ties
/// to even, sign kept.
fn round_e2m1(q: f32) -> f32 {
    const VALUES: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    // Boundary `k` lies between codes `k` and `k + 1`; a tie goes to the even
    // code, which is the upper one for odd `k`.
    const BOUNDS: [f32; 7] = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0];
    let mag = q.abs();
    let mut code = 0;
    for (k, &b) in BOUNDS.iter().enumerate() {
        if mag > b || (k % 2 == 1 && mag == b) {
            code += 1;
        }
    }
    VALUES[code].copysign(q)
}

/// FP8 `e4m3` values with a power-of-two scale per block of `block` values
/// (the activations of the attention output and the sliding-window keys).
pub(crate) fn qdq_fp8_ue8m0(x: &mut [f32], block: usize) {
    const INV_MAX: f32 = (1.0 / 448.0) as f32;
    for b in x.chunks_exact_mut(block) {
        let amax = b.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-4);
        let s = pow2_ceil(amax * INV_MAX);
        for v in b {
            *v = round_e4m3((*v / s).clamp(-448.0, 448.0)) * s;
        }
    }
}

/// FP4 `e2m1` values with a power-of-two scale per block of `block` values
/// (the indexer's queries and keys).
pub(crate) fn qdq_fp4_e8m0(x: &mut [f32], block: usize) {
    const MIN_AMAX: f32 = 6.0 * f32::MIN_POSITIVE;
    const INV_MAX: f32 = (1.0 / 6.0) as f32;
    for b in x.chunks_exact_mut(block) {
        let amax = b.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(MIN_AMAX);
        let s = pow2_ceil(amax * INV_MAX);
        for v in b {
            *v = round_e2m1((*v / s).clamp(-6.0, 6.0)) * s;
        }
    }
}

/// FP4 `e2m1` values with an `e4m3` scale per block of `block` values (the
/// compressed rows).
pub(crate) fn qdq_fp4_e4m3(x: &mut [f32], block: usize) {
    const MIN_AMAX: f32 = 6.0 / 512.0;
    for b in x.chunks_exact_mut(block) {
        let amax = b.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(MIN_AMAX);
        let s = round_e4m3(amax / 6.0);
        for v in b {
            *v = round_e2m1((*v / s).clamp(-6.0, 6.0)) * s;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pow2_ceil_rounds_up_to_a_power_of_two() {
        assert_eq!(pow2_ceil(1.0), 1.0);
        assert_eq!(pow2_ceil(1.0000001), 2.0);
        assert_eq!(pow2_ceil(0.3), 0.5);
        assert_eq!(pow2_ceil(0.25), 0.25);
        assert_eq!(pow2_ceil(1000.0), 1024.0);
    }

    #[test]
    fn e4m3_rounding_hits_the_grid() {
        // Spacing 0.125 in [1, 2): 1.0625 is the tie between 1.0 and 1.125,
        // which goes to the even mantissa.
        assert_eq!(round_e4m3(1.0625), 1.0);
        assert_eq!(round_e4m3(1.1875), 1.25);
        assert_eq!(round_e4m3(448.0), 448.0);
        assert_eq!(round_e4m3(447.0), 448.0);
        // Subnormals step by 2^-9.
        assert_eq!(round_e4m3(0.003), 2.0 * 2.0f32.powi(-9));
        assert_eq!(round_e4m3(0.0), 0.0);
        assert_eq!(round_e4m3(-2.6), -2.5);
    }

    #[test]
    fn e2m1_rounding_ties_to_even() {
        for (x, want) in [
            (0.25, 0.0),
            (0.2501, 0.5),
            (0.75, 1.0),
            (1.25, 1.0),
            (1.75, 2.0),
            (2.5, 2.0),
            (3.5, 4.0),
            (5.0, 4.0),
            (5.01, 6.0),
            (-0.75, -1.0),
        ] {
            assert_eq!(round_e2m1(x), want, "{x}");
        }
    }

    #[test]
    fn quantized_values_are_idempotent() {
        let src: Vec<f32> = (0..64)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.173)
            .collect();
        let mut a = src.clone();
        qdq_fp8_ue8m0(&mut a, 32);
        let mut again = a.clone();
        qdq_fp8_ue8m0(&mut again, 32);
        assert_eq!(a, again);
        let mut b = src.clone();
        qdq_fp4_e8m0(&mut b, 32);
        let mut again = b.clone();
        qdq_fp4_e8m0(&mut again, 32);
        assert_eq!(b, again);
        let mut c = src;
        qdq_fp4_e4m3(&mut c, 16);
        // Four-bit values: at most 15 distinct magnitudes per block scale.
        for blk in c.chunks(16) {
            let mut mags: Vec<u32> = blk.iter().map(|v| v.abs().to_bits()).collect();
            mags.sort_unstable();
            mags.dedup();
            assert!(mags.len() <= 8, "{mags:?}");
        }
    }

    // Goldens from the torch port of the training kernels (`kernel_torch.py`,
    // `.omp/ds-research/gen_l0_goldens.py`): the same inputs, bit for bit.
    fn golden_input(mul: f64) -> Vec<f32> {
        (0..160)
            .map(|i: i64| {
                ((((i * 7919) % 2003) - 1001) as f64 / 97.0 * (1 + (i / 32) % 5) as f64 * mul)
                    as f32
            })
            .collect()
    }

    const FP8_BITS: [u32; 160] = [
        0xc1200000, 0x41100000, 0x41000000, 0x40f00000, 0x40d00000, 0x40b00000, 0x40900000,
        0x40600000, 0x40300000, 0x3fe00000, 0x3f400000, 0xbe600000, 0xbf900000, 0xc0100000,
        0xc0400000, 0xc0800000, 0xc0a00000, 0xc0c00000, 0xc0e00000, 0xc1000000, 0xc1100000,
        0xc1200000, 0x41200000, 0x41100000, 0x41000000, 0x40e00000, 0x40c00000, 0x40a00000,
        0x40800000, 0x40500000, 0x40100000, 0x3fa00000, 0x3f200000, 0xbfb00000, 0xc0500000,
        0xc0a00000, 0xc0e00000, 0xc1100000, 0xc1300000, 0xc1500000, 0xc1700000, 0xc1800000,
        0xc1900000, 0xc1a00000, 0x41900000, 0x41800000, 0x41700000, 0x41500000, 0x41300000,
        0x41100000, 0x40f00000, 0x40b00000, 0x40600000, 0x3fd00000, 0xbe900000, 0xc0100000,
        0xc0800000, 0xc0c00000, 0xc1000000, 0xc1200000, 0xc1400000, 0xc1600000, 0xc1800000,
        0xc1900000, 0xc1f00000, 0x41f00000, 0x41e00000, 0x41c00000, 0x41b00000, 0x41900000,
        0x41700000, 0x41500000, 0x41200000, 0x40e00000, 0x40800000, 0x3f900000, 0xbfe00000,
        0xc0900000, 0xc0f00000, 0xc1200000, 0xc1500000, 0xc1800000, 0xc1a00000, 0xc1b00000,
        0xc1c00000, 0xc1e00000, 0xc1f00000, 0x41e00000, 0x41d00000, 0x41b00000, 0x41a00000,
        0x41900000, 0x41600000, 0x41300000, 0x41000000, 0x40b00000, 0x40600000, 0xbe900000,
        0xc0800000, 0xc1000000, 0xc1400000, 0xc1800000, 0xc1a00000, 0xc1c00000, 0xc1e00000,
        0xc1f00000, 0xc2100000, 0xc2200000, 0x42200000, 0x42100000, 0x42000000, 0x41e00000,
        0x41c00000, 0x41a00000, 0x41900000, 0x41500000, 0x41100000, 0x40b00000, 0x3fe00000,
        0xc0000000, 0xc0c00000, 0xc1200000, 0xc1600000, 0xc1900000, 0xc1b00000, 0xc1d00000,
        0xc1e00000, 0xc2000000, 0xc2300000, 0xc2500000, 0x42400000, 0x42300000, 0x42200000,
        0x42000000, 0x41e00000, 0x41c00000, 0x41a00000, 0x41600000, 0x41200000, 0x40a00000,
        0x00000000, 0xc0a00000, 0xc1200000, 0xc1600000, 0xc1a00000, 0xc1c00000, 0xc1e00000,
        0xc2000000, 0xc2200000, 0xc2300000, 0xc2400000, 0x42500000, 0x42300000, 0x42200000,
        0x42100000, 0x42000000, 0x41d00000, 0x41b00000, 0x41800000, 0x41400000,
    ];
    const FP4_E8M0_BITS: [u32; 160] = [
        0xc0400000, 0x40400000, 0x40400000, 0x40000000, 0x40000000, 0x3fc00000, 0x3fc00000,
        0x3f800000, 0x3f800000, 0x3f000000, 0x00000000, 0x80000000, 0xbf000000, 0xbf000000,
        0xbf800000, 0xbfc00000, 0xbfc00000, 0xc0000000, 0xc0000000, 0xc0000000, 0xc0400000,
        0xc0400000, 0x40400000, 0x40400000, 0x40000000, 0x40000000, 0x40000000, 0x3fc00000,
        0x3fc00000, 0x3f800000, 0x3f000000, 0x3f000000, 0x00000000, 0x80000000, 0xbf800000,
        0xc0000000, 0xc0000000, 0xc0400000, 0xc0400000, 0xc0800000, 0xc0800000, 0xc0c00000,
        0xc0c00000, 0xc0c00000, 0x40c00000, 0x40c00000, 0x40800000, 0x40800000, 0x40400000,
        0x40400000, 0x40000000, 0x40000000, 0x3f800000, 0x3f800000, 0x80000000, 0xbf800000,
        0xbf800000, 0xc0000000, 0xc0000000, 0xc0400000, 0xc0800000, 0xc0800000, 0xc0800000,
        0xc0c00000, 0xc1000000, 0x41000000, 0x41000000, 0x41000000, 0x40c00000, 0x40c00000,
        0x40800000, 0x40800000, 0x40400000, 0x40000000, 0x3f800000, 0x00000000, 0xbf800000,
        0xbf800000, 0xc0000000, 0xc0400000, 0xc0800000, 0xc0c00000, 0xc0c00000, 0xc0c00000,
        0xc1000000, 0xc1000000, 0xc1000000, 0x41000000, 0x41000000, 0x41000000, 0x40c00000,
        0x40c00000, 0x40800000, 0x40400000, 0x40400000, 0x40000000, 0x40000000, 0x80000000,
        0xc0000000, 0xc0000000, 0xc0800000, 0xc0800000, 0xc0c00000, 0xc1000000, 0xc1000000,
        0xc1000000, 0xc1400000, 0xc1400000, 0x41400000, 0x41400000, 0x41400000, 0x41000000,
        0x41000000, 0x40c00000, 0x40c00000, 0x40800000, 0x40000000, 0x40000000, 0x00000000,
        0x80000000, 0xc0000000, 0xc0800000, 0xc0800000, 0xc0c00000, 0xc0c00000, 0xc1000000,
        0xc1000000, 0xc1400000, 0xc1800000, 0xc1800000, 0x41800000, 0x41400000, 0x41400000,
        0x41400000, 0x41000000, 0x41000000, 0x40c00000, 0x40800000, 0x40000000, 0x40000000,
        0x00000000, 0xc0000000, 0xc0000000, 0xc0800000, 0xc0c00000, 0xc1000000, 0xc1000000,
        0xc1400000, 0xc1400000, 0xc1400000, 0xc1800000, 0x41800000, 0x41800000, 0x41400000,
        0x41400000, 0x41000000, 0x41000000, 0x40c00000, 0x40c00000, 0x40800000,
    ];
    const FP4_E4M3_BITS: [u32; 160] = [
        0xbed80000, 0x3ed80000, 0x3ed80000, 0x3e900000, 0x3e900000, 0x3e580000, 0x3e580000,
        0x3e100000, 0x3dd80000, 0x3d900000, 0x3d100000, 0x80000000, 0xbd100000, 0xbdd80000,
        0xbe100000, 0xbe100000, 0xbe580000, 0xbe900000, 0xbe900000, 0xbe900000, 0xbed80000,
        0xbed80000, 0x3ed80000, 0x3ed80000, 0x3e900000, 0x3e900000, 0x3e900000, 0x3e580000,
        0x3e580000, 0x3e100000, 0x3dd80000, 0x3d900000, 0x00000000, 0xbd900000, 0xbe100000,
        0xbe580000, 0xbe900000, 0xbed80000, 0xbed80000, 0xbf100000, 0xbf100000, 0xbf580000,
        0xbf580000, 0xbf580000, 0x3f580000, 0x3f580000, 0x3f100000, 0x3f100000, 0x3f000000,
        0x3ec00000, 0x3ec00000, 0x3e800000, 0x3e000000, 0x3d800000, 0x80000000, 0xbe000000,
        0xbe400000, 0xbe800000, 0xbec00000, 0xbec00000, 0xbf000000, 0xbf000000, 0xbf400000,
        0xbf400000, 0xbfa80000, 0x3fa80000, 0x3fa80000, 0x3f600000, 0x3f600000, 0x3f600000,
        0x3f280000, 0x3ee00000, 0x3ee00000, 0x3ea80000, 0x3e600000, 0x00000000, 0xbde00000,
        0xbe600000, 0xbea80000, 0xbee00000, 0xbf280000, 0xbf280000, 0xbf600000, 0xbf600000,
        0xbf600000, 0xbfa80000, 0xbfa80000, 0x3fa80000, 0x3fa80000, 0x3f600000, 0x3f600000,
        0x3f280000, 0x3f280000, 0x3ee00000, 0x3ea80000, 0x3e600000, 0x3e100000, 0x80000000,
        0xbe100000, 0xbe900000, 0xbf100000, 0xbf100000, 0xbf580000, 0xbf900000, 0xbf900000,
        0xbf900000, 0xbfd80000, 0xbfd80000, 0x3fd80000, 0x3fd80000, 0x3f900000, 0x3f900000,
        0x3f700000, 0x3f700000, 0x3f340000, 0x3ef00000, 0x3eb40000, 0x3e700000, 0x3df00000,
        0xbdf00000, 0xbe700000, 0xbef00000, 0xbef00000, 0xbf340000, 0xbf700000, 0xbf700000,
        0xbfb40000, 0xbfb40000, 0xc0100000, 0xc0100000, 0x40100000, 0x3fc00000, 0x3fc00000,
        0x3fc00000, 0x3f900000, 0x3f900000, 0x3f400000, 0x3f100000, 0x3ec00000, 0x3e400000,
        0x00000000, 0xbe400000, 0xbec00000, 0xbf100000, 0xbf400000, 0xbf900000, 0xbf900000,
        0xbfc00000, 0xbfc00000, 0xbfc00000, 0xc0100000, 0x40100000, 0x40100000, 0x3fc00000,
        0x3fc00000, 0x3fc00000, 0x3f900000, 0x3f400000, 0x3f400000, 0x3f100000,
    ];

    fn bits(x: &[f32]) -> Vec<u32> {
        x.iter().map(|v| v.to_bits()).collect()
    }

    #[test]
    fn the_quantizers_match_the_torch_port_bit_for_bit() {
        let mut x = golden_input(1.0);
        qdq_fp8_ue8m0(&mut x, 32);
        assert_eq!(bits(&x), FP8_BITS);
        let mut x = golden_input(0.31);
        qdq_fp4_e8m0(&mut x, 32);
        assert_eq!(bits(&x), FP4_E8M0_BITS);
        let mut x = golden_input(0.043);
        qdq_fp4_e4m3(&mut x, 16);
        assert_eq!(bits(&x), FP4_E4M3_BITS);
    }
}
