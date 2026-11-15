//! The K-quant, I-quant, ternary and NVFP4 weights of the GPU kernels, held
//! in the file's own blocks.
//!
//! A weight of one of these dtypes is not requantized: every block of the
//! file (256 values; NVFP4: 64) keeps its bytes, padded with zero bytes to
//! whole `u32` words (at most 3 bytes per block: 0.09 bit per value at most,
//! under 0.2 % of a Q4_K block). The words are stored *word-major*: word `w`
//! of block `blk` of a matrix of `nblk` blocks is word `w * nblk + blk` of
//! the matrix, so that the threads of a launch (consecutive rows of a dense
//! weight, consecutive blocks of a row of an expert) read consecutive words.
//! [`pack_native`] writes this order from the file bytes; the block order is
//! row-major for an expert's matrix and block-major (`b * rows + row`) for a
//! dense weight, as for the legacy formats ([`super::DenseLayout`]).
//!
//! The device decodes a block with one function per type (`q4k_block`, ...)
//! that follows the decoders of `model/quant/` (ggml's `dequantize_row_*`)
//! operation for operation, so the values are those of
//! [`super::dequant_row`]: each element is the same f32 expression of the
//! same integers. The function either adds `value * x[xo + e]` to a dot
//! product or stores the value (`emit`), which lets the matrix-vector
//! kernels and the prefill's dequantizing kernel share one decoder per type.
//!
//! The types that look values up in a table (IQ4_NL, IQ4_XS, the grid
//! types, NVFP4's scales) read it from the *tables* buffer ([`table_words`]):
//! the grids and sign table of `model/quant/iq_tables.rs` as `u32` words,
//! one device tensor per model load and device, not per layer or weight.

use burn::cubecl::prelude::*;

use super::{half_to_f32, mxfp4_value};
use crate::error::CucaError;
use crate::provider::local::model::quant::iq_tables as tab;
use crate::provider::local::model::quant::iquants::UE4M3_HALF;

/// Codes of the formats, the kernels' comptime format argument.
pub(super) mod code {
    pub(crate) const Q4_0: u32 = 0;
    pub(crate) const Q4_1: u32 = 1;
    pub(crate) const Q8_0: u32 = 2;
    pub(crate) const F32: u32 = 3;
    pub(crate) const MXFP4: u32 = 4;
    pub(crate) const IQ4_NL: u32 = 5;
    pub(crate) const Q2_K: u32 = 6;
    pub(crate) const Q3_K: u32 = 7;
    pub(crate) const Q4_K: u32 = 8;
    pub(crate) const Q5_K: u32 = 9;
    pub(crate) const Q6_K: u32 = 10;
    pub(crate) const IQ4_XS: u32 = 11;
    pub(crate) const IQ2_XXS: u32 = 12;
    pub(crate) const IQ2_XS: u32 = 13;
    pub(crate) const IQ2_S: u32 = 14;
    pub(crate) const IQ3_XXS: u32 = 15;
    pub(crate) const IQ3_S: u32 = 16;
    pub(crate) const IQ1_S: u32 = 17;
    pub(crate) const IQ1_M: u32 = 18;
    pub(crate) const TQ1_0: u32 = 19;
    pub(crate) const TQ2_0: u32 = 20;
    pub(crate) const NVFP4: u32 = 21;
    pub(crate) const Q8_K: u32 = 22;
    pub(crate) const Q5_0: u32 = 23;
    pub(crate) const Q5_1: u32 = 24;
}

/// Values per block of the format with code `format`: 32 for the legacy
/// formats, MXFP4 and IQ4_NL, 64 for NVFP4, 256 for the super-block types.
pub(super) const fn block_elems_of(format: u32) -> usize {
    if format <= code::IQ4_NL || format == code::Q5_0 || format == code::Q5_1 {
        32
    } else if format == code::NVFP4 {
        64
    } else {
        256
    }
}

/// Whether the format with code `format` is held in the word-major layout of
/// this module.
pub(super) const fn is_native_code(format: u32) -> bool {
    format > code::IQ4_NL
}

/// File bytes of one block of the native format with code `format`, 0 for the
/// others.
pub(super) const fn file_block_bytes(format: u32) -> usize {
    match format {
        code::Q2_K => 84,
        code::Q3_K => 110,
        code::Q4_K => 144,
        code::Q5_K => 176,
        code::Q6_K => 210,
        code::IQ4_XS => 136,
        code::IQ2_XXS => 66,
        code::IQ2_XS => 74,
        code::IQ2_S => 82,
        code::IQ3_XXS => 98,
        code::IQ3_S => 110,
        code::IQ1_S => 50,
        code::IQ1_M => 56,
        code::TQ1_0 => 54,
        code::TQ2_0 => 66,
        code::NVFP4 => 36,
        code::Q8_K => 292,
        code::Q5_0 => 22,
        code::Q5_1 => 24,
        _ => 0,
    }
}

/// Words of one block of the native format with code `format` (the file
/// bytes padded to whole words).
pub(super) const fn block_words_of(format: u32) -> usize {
    file_block_bytes(format).div_ceil(4)
}

// Where each table starts in the tables buffer, in words.
/// `ksigns_iq2xs`: 128 bytes.
const T_SIGNS: usize = 0;
/// `kvalues_iq4nl`: 16 signed bytes.
const T_KV4: usize = T_SIGNS + 32;
/// The IQ2_XXS grid: 256 entries of 8 bytes.
const T_G2XXS: usize = T_KV4 + 4;
/// The IQ2_XS grid: 512 entries of 8 bytes.
const T_G2XS: usize = T_G2XXS + 256 * 2;
/// The IQ2_S grid: 1024 entries of 8 bytes.
const T_G2S: usize = T_G2XS + 512 * 2;
/// The IQ3_XXS grid: 256 entries of 4 bytes.
const T_G3XXS: usize = T_G2S + 1024 * 2;
/// The IQ3_S grid: 512 entries of 4 bytes.
const T_G3S: usize = T_G3XXS + 256;
/// The IQ1 grid: 2048 entries of 8 signed bytes.
const T_G1S: usize = T_G3S + 512;
/// UE4M3 scale bytes as f32 (the half scales of NVFP4): 256 entries.
const T_UE4M3: usize = T_G1S + 2048 * 2;
/// Words of the tables buffer.
const TABLE_WORDS: usize = T_UE4M3 + 256;

/// The words of `bytes`, four to a little-endian word, zero padded.
fn byte_words(bytes: impl IntoIterator<Item = u8>) -> Vec<u32> {
    let bytes: Vec<u8> = bytes.into_iter().collect();
    bytes
        .chunks(4)
        .map(|c| {
            let mut le = [0u8; 4];
            le[..c.len()].copy_from_slice(c);
            u32::from_le_bytes(le)
        })
        .collect()
}

/// The tables buffer: every lookup table the native formats read as `u32`
/// words (bytes four to a word, 64-bit grid entries as two words, the UE4M3
/// scales as f32 bits). One per device.
pub(crate) fn table_words() -> Vec<u32> {
    let mut words = Vec::with_capacity(TABLE_WORDS);
    let mut section = |at: usize, part: Vec<u32>| {
        debug_assert_eq!(words.len(), at);
        words.extend(part);
    };
    let grid64 = |g: &[u64]| -> Vec<u32> {
        g.iter()
            .flat_map(|v| [*v as u32, (*v >> 32) as u32])
            .collect()
    };
    section(T_SIGNS, byte_words(tab::KSIGNS_IQ2XS));
    section(T_KV4, byte_words(tab::KVALUES_IQ4NL.map(i8::cast_unsigned)));
    section(T_G2XXS, grid64(&tab::IQ2XXS_GRID));
    section(T_G2XS, grid64(&tab::IQ2XS_GRID));
    section(T_G2S, grid64(&tab::IQ2S_GRID));
    section(T_G3XXS, tab::IQ3XXS_GRID.to_vec());
    section(T_G3S, tab::IQ3S_GRID.to_vec());
    section(T_G1S, grid64(&tab::IQ1S_GRID));
    section(T_UE4M3, UE4M3_HALF.iter().map(|v| v.to_bits()).collect());
    words
}

/// Write the word-major words of a `rows x cols` matrix of the native format
/// `format` whose file blocks are `raw` into `out` (`blocks * block_words`
/// long): block `idx` of the file (row `idx / bpr`, block `idx % bpr` of the
/// row) goes to block `idx` (`dense == false`) or `b * rows + r` (the dense
/// kernels' block-major order).
///
/// # Errors
///
/// [`CucaError::Provider`] for `raw` that is not whole blocks of the matrix.
pub(super) fn pack_native(
    format: u32,
    rows: usize,
    cols: usize,
    raw: &[u8],
    dense: bool,
    out: &mut [u32],
) -> Result<(), CucaError> {
    let (bytes, words, elems) = (
        file_block_bytes(format),
        block_words_of(format),
        block_elems_of(format),
    );
    let bpr = cols / elems;
    let nblk = rows * bpr;
    if bytes == 0 || raw.len() != nblk * bytes || out.len() != nblk * words {
        return Err(super::super::err(format!(
            "native matrix: {} bytes for {nblk} blocks of {bytes}",
            raw.len()
        )));
    }
    for (idx, block) in raw.chunks_exact(bytes).enumerate() {
        let (r, b) = (idx / bpr, idx % bpr);
        let blk = if dense { b * rows + r } else { idx };
        for (w, chunk) in block.chunks(4).enumerate() {
            let mut le = [0u8; 4];
            le[..chunk.len()].copy_from_slice(chunk);
            out[w * nblk + blk] = u32::from_le_bytes(le);
        }
    }
    Ok(())
}

// ------------------------------------------------------------ device side

/// Byte `k` (0 to 3) of `word`.
#[cube]
pub(super) fn byte_of(word: u32, k: usize) -> u32 {
    (word >> (u32::cast_from(k) * 8u32)) & 0xFFu32
}

/// The signed byte `v` (0 to 255) as an f32.
#[cube]
fn sext8(v: u32) -> f32 {
    f32::cast_from(i32::cast_from(v & 0x7Fu32) - i32::cast_from(v & 0x80u32))
}

/// The scale (low byte) and minimum (second byte) of sub-block `s` (0 to 7)
/// of a Q4_K or Q5_K block whose 12 scale bytes are the words `w0`, `w1`,
/// `w2`: ggml's `get_scale_min_k4`.
#[cube]
fn k4_scale_min(w0: u32, w1: u32, w2: u32, s: usize) -> u32 {
    let sh = u32::cast_from(s % 4usize) * 8u32;
    let mut sc = (w0 >> sh) & 63u32;
    let mut m = (w1 >> sh) & 63u32;
    if s >= 4usize {
        let a = (w2 >> sh) & 0xFFu32;
        sc = (a & 0xFu32) | ((((w0 >> sh) & 0xFFu32) >> 6u32) << 4u32);
        m = (a >> 4u32) | ((((w1 >> sh) & 0xFFu32) >> 6u32) << 4u32);
    }
    sc | (m << 8u32)
}

/// Q2_K block `i0 + w * nblk`: 16 scale/min bytes, 64 bytes of 2-bit quants,
/// `d`, `dmin`.
#[cube]
#[allow(clippy::too_many_arguments)]
fn q2k_block(
    pool: &Tensor<u32>,
    i0: usize,
    nblk: usize,
    x: &Tensor<f32>,
    xo: usize,
    out: &mut Tensor<f32>,
    oo: usize,
    #[comptime] emit: bool,
) -> f32 {
    let mut acc = 0.0f32;
    let dm = pool[i0 + 20usize * nblk];
    let d = half_to_f32(dm & 0xFFFFu32);
    let dmin = half_to_f32(dm >> 16u32);
    for n in 0..2usize {
        for j in 0..4usize {
            for half in 0..2usize {
                let si = 8usize * n + 2usize * j + half;
                let sb = byte_of(pool[i0 + (si / 4usize) * nblk], si % 4usize);
                let dl = d * f32::cast_from(sb & 0xFu32);
                let ml = dmin * f32::cast_from(sb >> 4u32);
                for wd in 0..4usize {
                    let q = pool[i0 + (4usize + 8usize * n + 4usize * half + wd) * nblk];
                    for k in 0..4usize {
                        let v = dl
                            * f32::cast_from((byte_of(q, k) >> (2u32 * u32::cast_from(j))) & 3u32)
                            - ml;
                        let e = 128usize * n + 32usize * j + 16usize * half + 4usize * wd + k;
                        if emit {
                            out[oo + e] = v;
                        } else {
                            acc += v * x[xo + e];
                        }
                    }
                }
            }
        }
    }
    acc
}

/// Q3_K block: 32 bytes of high bits, 64 bytes of 2-bit quants, 12 bytes of
/// 6-bit scales, `d`.
#[cube]
#[allow(clippy::too_many_arguments)]
fn q3k_block(
    pool: &Tensor<u32>,
    i0: usize,
    nblk: usize,
    x: &Tensor<f32>,
    xo: usize,
    out: &mut Tensor<f32>,
    oo: usize,
    #[comptime] emit: bool,
) -> f32 {
    let mut acc = 0.0f32;
    let d = half_to_f32(pool[i0 + 27usize * nblk] & 0xFFFFu32);
    let w0 = pool[i0 + 24usize * nblk];
    let w1 = pool[i0 + 25usize * nblk];
    let tmp = pool[i0 + 26usize * nblk];
    let a0 = (w0 & 0x0F0F_0F0Fu32) | ((tmp & 0x0303_0303u32) << 4u32);
    let a1 = (w1 & 0x0F0F_0F0Fu32) | (((tmp >> 2u32) & 0x0303_0303u32) << 4u32);
    let a2 = ((w0 >> 4u32) & 0x0F0F_0F0Fu32) | (((tmp >> 4u32) & 0x0303_0303u32) << 4u32);
    let a3 = ((w1 >> 4u32) & 0x0F0F_0F0Fu32) | (((tmp >> 6u32) & 0x0303_0303u32) << 4u32);
    for n in 0..2usize {
        for j in 0..4usize {
            for half in 0..2usize {
                let si = 8usize * n + 2usize * j + half;
                let mut aw = a3;
                if si < 4usize {
                    aw = a0;
                } else if si < 8usize {
                    aw = a1;
                } else if si < 12usize {
                    aw = a2;
                }
                let dl = d * (f32::cast_from(byte_of(aw, si % 4usize)) - 32.0f32);
                for wdh in 0..4usize {
                    let wd = 4usize * half + wdh;
                    let q = pool[i0 + (8usize + 8usize * n + wd) * nblk];
                    let h = pool[i0 + wd * nblk];
                    for k in 0..4usize {
                        let q2 = (byte_of(q, k) >> (2u32 * u32::cast_from(j))) & 3u32;
                        let hm = (byte_of(h, k) >> u32::cast_from(4usize * n + j)) & 1u32;
                        let v = dl * (f32::cast_from(q2) - 4.0f32 * f32::cast_from(1u32 - hm));
                        let e = 128usize * n + 32usize * j + 4usize * wd + k;
                        if emit {
                            out[oo + e] = v;
                        } else {
                            acc += v * x[xo + e];
                        }
                    }
                }
            }
        }
    }
    acc
}

/// Q4_K block: `d`, `dmin`, 12 bytes of 6-bit scales and minimums, 128 bytes
/// of nibbles.
#[cube]
#[allow(clippy::too_many_arguments)]
fn q4k_block(
    pool: &Tensor<u32>,
    i0: usize,
    nblk: usize,
    x: &Tensor<f32>,
    xo: usize,
    out: &mut Tensor<f32>,
    oo: usize,
    #[comptime] emit: bool,
) -> f32 {
    let mut acc = 0.0f32;
    let dm = pool[i0];
    let d = half_to_f32(dm & 0xFFFFu32);
    let dmin = half_to_f32(dm >> 16u32);
    let w0 = pool[i0 + nblk];
    let w1 = pool[i0 + 2usize * nblk];
    let w2 = pool[i0 + 3usize * nblk];
    for j in 0..4usize {
        let s1 = k4_scale_min(w0, w1, w2, 2usize * j);
        let s2 = k4_scale_min(w0, w1, w2, 2usize * j + 1usize);
        let d1 = d * f32::cast_from(s1 & 0xFFu32);
        let m1 = dmin * f32::cast_from(s1 >> 8u32);
        let d2 = d * f32::cast_from(s2 & 0xFFu32);
        let m2 = dmin * f32::cast_from(s2 >> 8u32);
        for wd in 0..8usize {
            let q = pool[i0 + (4usize + 8usize * j + wd) * nblk];
            for k in 0..4usize {
                let b = byte_of(q, k);
                let lo = d1 * f32::cast_from(b & 0xFu32) - m1;
                let hi = d2 * f32::cast_from(b >> 4u32) - m2;
                let e = 64usize * j + 4usize * wd + k;
                if emit {
                    out[oo + e] = lo;
                    out[oo + e + 32usize] = hi;
                } else {
                    acc += lo * x[xo + e] + hi * x[xo + e + 32usize];
                }
            }
        }
    }
    acc
}

/// Q5_K block: `d`, `dmin`, 12 bytes of scales, 32 bytes of fifth bits,
/// 128 bytes of nibbles.
#[cube]
#[allow(clippy::too_many_arguments)]
fn q5k_block(
    pool: &Tensor<u32>,
    i0: usize,
    nblk: usize,
    x: &Tensor<f32>,
    xo: usize,
    out: &mut Tensor<f32>,
    oo: usize,
    #[comptime] emit: bool,
) -> f32 {
    let mut acc = 0.0f32;
    let dm = pool[i0];
    let d = half_to_f32(dm & 0xFFFFu32);
    let dmin = half_to_f32(dm >> 16u32);
    let w0 = pool[i0 + nblk];
    let w1 = pool[i0 + 2usize * nblk];
    let w2 = pool[i0 + 3usize * nblk];
    for j in 0..4usize {
        let s1 = k4_scale_min(w0, w1, w2, 2usize * j);
        let s2 = k4_scale_min(w0, w1, w2, 2usize * j + 1usize);
        let d1 = d * f32::cast_from(s1 & 0xFFu32);
        let m1 = dmin * f32::cast_from(s1 >> 8u32);
        let d2 = d * f32::cast_from(s2 & 0xFFu32);
        let m2 = dmin * f32::cast_from(s2 >> 8u32);
        let u1 = 2u32 * u32::cast_from(j);
        for wd in 0..8usize {
            let q = pool[i0 + (12usize + 8usize * j + wd) * nblk];
            let h = pool[i0 + (4usize + wd) * nblk];
            for k in 0..4usize {
                let b = byte_of(q, k);
                let hb = byte_of(h, k);
                let lo = d1 * f32::cast_from((b & 0xFu32) + (((hb >> u1) & 1u32) << 4u32)) - m1;
                let hi =
                    d2 * f32::cast_from((b >> 4u32) + (((hb >> (u1 + 1u32)) & 1u32) << 4u32)) - m2;
                let e = 64usize * j + 4usize * wd + k;
                if emit {
                    out[oo + e] = lo;
                    out[oo + e + 32usize] = hi;
                } else {
                    acc += lo * x[xo + e] + hi * x[xo + e + 32usize];
                }
            }
        }
    }
    acc
}

/// Q6_K block: 128 bytes of low nibbles, 64 bytes of high bit pairs, 16
/// signed scales, `d`.
#[cube]
#[allow(clippy::too_many_arguments)]
fn q6k_block(
    pool: &Tensor<u32>,
    i0: usize,
    nblk: usize,
    x: &Tensor<f32>,
    xo: usize,
    out: &mut Tensor<f32>,
    oo: usize,
    #[comptime] emit: bool,
) -> f32 {
    let mut acc = 0.0f32;
    let d = half_to_f32(pool[i0 + 52usize * nblk] & 0xFFFFu32);
    for n in 0..2usize {
        let s0 = pool[i0 + (48usize + 2usize * n) * nblk];
        let s1 = pool[i0 + (49usize + 2usize * n) * nblk];
        for lw in 0..8usize {
            let is = lw / 4usize;
            let c1 = d * sext8(byte_of(s0, is));
            let c2 = d * sext8(byte_of(s0, is + 2usize));
            let c3 = d * sext8(byte_of(s1, is));
            let c4 = d * sext8(byte_of(s1, is + 2usize));
            let a = pool[i0 + (16usize * n + lw) * nblk];
            let b = pool[i0 + (16usize * n + 8usize + lw) * nblk];
            let h = pool[i0 + (32usize + 8usize * n + lw) * nblk];
            for k in 0..4usize {
                let ab = byte_of(a, k);
                let bb = byte_of(b, k);
                let hb = byte_of(h, k);
                let q1 = (ab & 0xFu32) | ((hb & 3u32) << 4u32);
                let q2 = (bb & 0xFu32) | (((hb >> 2u32) & 3u32) << 4u32);
                let q3 = (ab >> 4u32) | (((hb >> 4u32) & 3u32) << 4u32);
                let q4 = (bb >> 4u32) | ((hb >> 6u32) << 4u32);
                let v1 = c1 * (f32::cast_from(q1) - 32.0f32);
                let v2 = c2 * (f32::cast_from(q2) - 32.0f32);
                let v3 = c3 * (f32::cast_from(q3) - 32.0f32);
                let v4 = c4 * (f32::cast_from(q4) - 32.0f32);
                let e = 128usize * n + 4usize * lw + k;
                if emit {
                    out[oo + e] = v1;
                    out[oo + e + 32usize] = v2;
                    out[oo + e + 64usize] = v3;
                    out[oo + e + 96usize] = v4;
                } else {
                    acc += v1 * x[xo + e]
                        + v2 * x[xo + e + 32usize]
                        + v3 * x[xo + e + 64usize]
                        + v4 * x[xo + e + 96usize];
                }
            }
        }
    }
    acc
}

/// Byte `off` of block `i0 + w * nblk` of a native matrix.
#[cube]
fn rbyte(pool: &Tensor<u32>, i0: usize, nblk: usize, off: usize) -> u32 {
    (pool[i0 + (off / 4usize) * nblk] >> (u32::cast_from(off % 4usize) * 8u32)) & 0xFFu32
}

/// The little-endian `u16` at the even byte offset `off` of a block.
#[cube]
fn rhalf(pool: &Tensor<u32>, i0: usize, nblk: usize, off: usize) -> u32 {
    (pool[i0 + (off / 4usize) * nblk] >> (u32::cast_from(off % 4usize) * 8u32)) & 0xFFFFu32
}

/// The little-endian `u32` at the even byte offset `off` of a block (it may
/// straddle two words).
#[cube]
fn rword(pool: &Tensor<u32>, i0: usize, nblk: usize, off: usize) -> u32 {
    rhalf(pool, i0, nblk, off) | (rhalf(pool, i0, nblk, off + 2usize) << 16u32)
}

/// Byte `i` of the table that starts at word `base` of the tables buffer.
#[cube]
fn tbyte(tables: &Tensor<u32>, base: usize, i: usize) -> u32 {
    (tables[base + i / 4usize] >> (u32::cast_from(i % 4usize) * 8u32)) & 0xFFu32
}

/// Entry `n` (0 to 15) of `kvalues_iq4nl`.
#[cube]
pub(super) fn kv4(tables: &Tensor<u32>, n: u32) -> f32 {
    sext8(tbyte(tables, comptime![T_KV4], usize::cast_from(n)))
}

/// `v`, negated where bit `j` of `signs` is set.
#[cube]
fn signed(v: f32, signs: u32, j: usize) -> f32 {
    let mut r = v;
    if ((signs >> u32::cast_from(j)) & 1u32) != 0u32 {
        r = -v;
    }
    r
}

/// `v * x[xo + e]`, or (`emit`) `v` stored at `out[oo + e]` and 0.
#[cube]
#[allow(clippy::too_many_arguments)]
fn consume(
    v: f32,
    x: &Tensor<f32>,
    xo: usize,
    out: &mut Tensor<f32>,
    oo: usize,
    e: usize,
    #[comptime] emit: bool,
) -> f32 {
    let mut r = f32::cast_from(e - e);
    if comptime![emit] {
        out[oo + e] = v;
    } else {
        r = v * x[xo + e];
    }
    r
}

/// `3^n`.
#[cube]
fn pow3(n: usize) -> u32 {
    let mut p = u32::cast_from(n - n) + 1u32;
    for _ in 0..n {
        p = p * 3u32;
    }
    p
}

/// Q5_0 (22 bytes) and Q5_1 (24 bytes) block of 32 values: `d` (and for Q5_1
/// `m`) as f16, a `u32` of the fifth bits, 16 bytes of nibbles (byte `j`:
/// element `j` low, element `j + 16` high).
#[cube]
#[allow(clippy::too_many_arguments)]
fn q5_block(
    pool: &Tensor<u32>,
    i0: usize,
    nblk: usize,
    x: &Tensor<f32>,
    xo: usize,
    out: &mut Tensor<f32>,
    oo: usize,
    #[comptime] emit: bool,
    #[comptime] with_min: bool,
) -> f32 {
    let mut acc = 0.0f32;
    let d = half_to_f32(rhalf(pool, i0, nblk, 0usize));
    let mut m = d - d;
    let mut at = 2usize;
    if comptime![with_min] {
        m = half_to_f32(rhalf(pool, i0, nblk, 2usize));
        at = 4usize;
    }
    let qh = rword(pool, i0, nblk, at);
    for j in 0..16usize {
        let q = rbyte(pool, i0, nblk, at + 4usize + j);
        let ju = u32::cast_from(j);
        let xh0 = ((qh >> ju) << 4u32) & 0x10u32;
        let xh1 = (qh >> (ju + 12u32)) & 0x10u32;
        let lo = (q & 0xFu32) | xh0;
        let hi = (q >> 4u32) | xh1;
        let mut v0 = (f32::cast_from(lo) - 16.0f32) * d;
        let mut v1 = (f32::cast_from(hi) - 16.0f32) * d;
        if comptime![with_min] {
            v0 = f32::cast_from(lo) * d + m;
            v1 = f32::cast_from(hi) * d + m;
        }
        acc += consume(v0, x, xo, out, oo, j, emit);
        acc += consume(v1, x, xo, out, oo, j + 16usize, emit);
    }
    acc
}

/// Q8_K block (292 bytes): an f32 `d`, 256 signed bytes, 16 `i16` block sums
/// the kernels do not read.
#[cube]
#[allow(clippy::too_many_arguments)]
fn q8k_block(
    pool: &Tensor<u32>,
    i0: usize,
    nblk: usize,
    x: &Tensor<f32>,
    xo: usize,
    out: &mut Tensor<f32>,
    oo: usize,
    #[comptime] emit: bool,
) -> f32 {
    let mut acc = 0.0f32;
    let d = f32::reinterpret(pool[i0]);
    for w in 0..64usize {
        let word = pool[i0 + (1usize + w) * nblk];
        for k in 0..4usize {
            let v = d * sext8(byte_of(word, k));
            acc += consume(v, x, xo, out, oo, 4usize * w + k, emit);
        }
    }
    acc
}

/// IQ2_XXS block (66 bytes): `d`, then per 32 values a grid index byte per 8
/// values and a `u32` of four 7-bit sign indices and a 4-bit scale.
#[cube]
#[allow(clippy::too_many_arguments)]
fn iq2xxs_block(
    pool: &Tensor<u32>,
    tables: &Tensor<u32>,
    i0: usize,
    nblk: usize,
    x: &Tensor<f32>,
    xo: usize,
    out: &mut Tensor<f32>,
    oo: usize,
    #[comptime] emit: bool,
) -> f32 {
    let mut acc = 0.0f32;
    let d = half_to_f32(rhalf(pool, i0, nblk, 0usize));
    for ib in 0..8usize {
        let aux = rword(pool, i0, nblk, 6usize + 8usize * ib);
        let db = d * (0.5f32 + f32::cast_from(aux >> 28u32)) * 0.25f32;
        for l in 0..4usize {
            let idx = usize::cast_from(rbyte(pool, i0, nblk, 2usize + 8usize * ib + l));
            let signs = tbyte(
                tables,
                comptime![T_SIGNS],
                usize::cast_from((aux >> (7u32 * u32::cast_from(l))) & 127u32),
            );
            for j in 0..8usize {
                let g = f32::cast_from(tbyte(tables, comptime![T_G2XXS], idx * 8usize + j));
                let e = 32usize * ib + 8usize * l + j;
                acc += consume(signed(db * g, signs, j), x, xo, out, oo, e, emit);
            }
        }
    }
    acc
}

/// IQ2_XS block (74 bytes): `d`, 32 `u16` (9-bit grid index, 7-bit sign
/// index), 8 scale bytes (a nibble per 16 values).
#[cube]
#[allow(clippy::too_many_arguments)]
fn iq2xs_block(
    pool: &Tensor<u32>,
    tables: &Tensor<u32>,
    i0: usize,
    nblk: usize,
    x: &Tensor<f32>,
    xo: usize,
    out: &mut Tensor<f32>,
    oo: usize,
    #[comptime] emit: bool,
) -> f32 {
    let mut acc = 0.0f32;
    let d = half_to_f32(rhalf(pool, i0, nblk, 0usize));
    for ib in 0..8usize {
        let sc = rbyte(pool, i0, nblk, 66usize + ib);
        let db0 = d * (0.5f32 + f32::cast_from(sc & 0xFu32)) * 0.25f32;
        let db1 = d * (0.5f32 + f32::cast_from(sc >> 4u32)) * 0.25f32;
        for l in 0..4usize {
            let v = rhalf(pool, i0, nblk, 2usize + 8usize * ib + 2usize * l);
            let idx = usize::cast_from(v & 511u32);
            let signs = tbyte(tables, comptime![T_SIGNS], usize::cast_from(v >> 9u32));
            let mut db = db0;
            if l >= 2usize {
                db = db1;
            }
            for j in 0..8usize {
                let g = f32::cast_from(tbyte(tables, comptime![T_G2XS], idx * 8usize + j));
                let e = 32usize * ib + 8usize * l + j;
                acc += consume(signed(db * g, signs, j), x, xo, out, oo, e, emit);
            }
        }
    }
    acc
}

/// IQ2_S block (82 bytes): `d`, 32 low grid index bytes, 32 sign bytes, 8
/// bytes of high index bits, 8 scale bytes.
#[cube]
#[allow(clippy::too_many_arguments)]
fn iq2s_block(
    pool: &Tensor<u32>,
    tables: &Tensor<u32>,
    i0: usize,
    nblk: usize,
    x: &Tensor<f32>,
    xo: usize,
    out: &mut Tensor<f32>,
    oo: usize,
    #[comptime] emit: bool,
) -> f32 {
    let mut acc = 0.0f32;
    let d = half_to_f32(rhalf(pool, i0, nblk, 0usize));
    for ib in 0..8usize {
        let qh = rbyte(pool, i0, nblk, 66usize + ib);
        let sc = rbyte(pool, i0, nblk, 74usize + ib);
        let db0 = d * (0.5f32 + f32::cast_from(sc & 0xFu32)) * 0.25f32;
        let db1 = d * (0.5f32 + f32::cast_from(sc >> 4u32)) * 0.25f32;
        for l in 0..4usize {
            let lo = rbyte(pool, i0, nblk, 2usize + 4usize * ib + l);
            let idx = usize::cast_from(lo | ((qh << (8u32 - 2u32 * u32::cast_from(l))) & 0x300u32));
            let signs = rbyte(pool, i0, nblk, 34usize + 4usize * ib + l);
            let mut db = db0;
            if l >= 2usize {
                db = db1;
            }
            for j in 0..8usize {
                let g = f32::cast_from(tbyte(tables, comptime![T_G2S], idx * 8usize + j));
                let e = 32usize * ib + 8usize * l + j;
                acc += consume(signed(db * g, signs, j), x, xo, out, oo, e, emit);
            }
        }
    }
    acc
}

/// IQ3_XXS block (98 bytes): `d`, 64 grid index bytes (4 values each), then
/// per 32 values a `u32` of four 7-bit sign indices and a 4-bit scale.
#[cube]
#[allow(clippy::too_many_arguments)]
fn iq3xxs_block(
    pool: &Tensor<u32>,
    tables: &Tensor<u32>,
    i0: usize,
    nblk: usize,
    x: &Tensor<f32>,
    xo: usize,
    out: &mut Tensor<f32>,
    oo: usize,
    #[comptime] emit: bool,
) -> f32 {
    let mut acc = 0.0f32;
    let d = half_to_f32(rhalf(pool, i0, nblk, 0usize));
    for ib in 0..8usize {
        let aux = rword(pool, i0, nblk, 66usize + 4usize * ib);
        let db = d * (0.5f32 + f32::cast_from(aux >> 28u32)) * 0.5f32;
        for l in 0..4usize {
            let signs = tbyte(
                tables,
                comptime![T_SIGNS],
                usize::cast_from((aux >> (7u32 * u32::cast_from(l))) & 127u32),
            );
            let i1 = usize::cast_from(rbyte(pool, i0, nblk, 2usize + 8usize * ib + 2usize * l));
            let i2 = usize::cast_from(rbyte(pool, i0, nblk, 3usize + 8usize * ib + 2usize * l));
            for j in 0..4usize {
                let g1 = f32::cast_from(tbyte(tables, comptime![T_G3XXS], i1 * 4usize + j));
                let g2 = f32::cast_from(tbyte(tables, comptime![T_G3XXS], i2 * 4usize + j));
                let e = 32usize * ib + 8usize * l + j;
                acc += consume(signed(db * g1, signs, j), x, xo, out, oo, e, emit);
                acc += consume(
                    signed(db * g2, signs, j + 4usize),
                    x,
                    xo,
                    out,
                    oo,
                    e + 4usize,
                    emit,
                );
            }
        }
    }
    acc
}

/// IQ3_S block (110 bytes): `d`, 64 low grid index bytes, 8 bytes of high
/// index bits, 32 sign bytes, 4 scale bytes (a nibble per 32 values).
#[cube]
#[allow(clippy::too_many_arguments)]
fn iq3s_block(
    pool: &Tensor<u32>,
    tables: &Tensor<u32>,
    i0: usize,
    nblk: usize,
    x: &Tensor<f32>,
    xo: usize,
    out: &mut Tensor<f32>,
    oo: usize,
    #[comptime] emit: bool,
) -> f32 {
    let mut acc = 0.0f32;
    let d = half_to_f32(rhalf(pool, i0, nblk, 0usize));
    for ib in 0..8usize {
        let qh = rbyte(pool, i0, nblk, 66usize + ib);
        let s = (rbyte(pool, i0, nblk, 106usize + ib / 2usize)
            >> (4u32 * u32::cast_from(ib % 2usize)))
            & 0xFu32;
        let db = d * f32::cast_from(1u32 + 2u32 * s);
        for l in 0..4usize {
            let lu = u32::cast_from(l);
            let i1 = usize::cast_from(
                rbyte(pool, i0, nblk, 2usize + 8usize * ib + 2usize * l)
                    | ((qh << (8u32 - 2u32 * lu)) & 256u32),
            );
            let i2 = usize::cast_from(
                rbyte(pool, i0, nblk, 3usize + 8usize * ib + 2usize * l)
                    | ((qh << (7u32 - 2u32 * lu)) & 256u32),
            );
            let signs = rbyte(pool, i0, nblk, 74usize + 4usize * ib + l);
            for j in 0..4usize {
                let g1 = f32::cast_from(tbyte(tables, comptime![T_G3S], i1 * 4usize + j));
                let g2 = f32::cast_from(tbyte(tables, comptime![T_G3S], i2 * 4usize + j));
                let e = 32usize * ib + 8usize * l + j;
                acc += consume(signed(db * g1, signs, j), x, xo, out, oo, e, emit);
                acc += consume(
                    signed(db * g2, signs, j + 4usize),
                    x,
                    xo,
                    out,
                    oo,
                    e + 4usize,
                    emit,
                );
            }
        }
    }
    acc
}

/// IQ1_S block (50 bytes): `d`, 32 low grid index bytes, 8 `u16` (3 index
/// bits per 8 values, a 3-bit scale, a delta sign).
#[cube]
#[allow(clippy::too_many_arguments)]
fn iq1s_block(
    pool: &Tensor<u32>,
    tables: &Tensor<u32>,
    i0: usize,
    nblk: usize,
    x: &Tensor<f32>,
    xo: usize,
    out: &mut Tensor<f32>,
    oo: usize,
    #[comptime] emit: bool,
) -> f32 {
    let mut acc = 0.0f32;
    let d = half_to_f32(rhalf(pool, i0, nblk, 0usize));
    for ib in 0..8usize {
        let qh = rhalf(pool, i0, nblk, 34usize + 2usize * ib);
        let dl = d * f32::cast_from(2u32 * ((qh >> 12u32) & 7u32) + 1u32);
        let delta = 0.125f32 - 0.25f32 * f32::cast_from((qh >> 15u32) & 1u32);
        for l in 0..4usize {
            let lo = rbyte(pool, i0, nblk, 2usize + 4usize * ib + l);
            let idx = usize::cast_from(lo | (((qh >> (3u32 * u32::cast_from(l))) & 7u32) << 8u32));
            for j in 0..8usize {
                let g = sext8(tbyte(tables, comptime![T_G1S], idx * 8usize + j));
                let e = 32usize * ib + 8usize * l + j;
                acc += consume(dl * (g + delta), x, xo, out, oo, e, emit);
            }
        }
    }
    acc
}

/// IQ1_M block (56 bytes): 32 low grid index bytes, 16 bytes of (3 index
/// bits, a delta sign) per 8 values, 4 `u16` of 3-bit scales whose top
/// nibbles hold the f16 `d`.
#[cube]
#[allow(clippy::too_many_arguments)]
fn iq1m_block(
    pool: &Tensor<u32>,
    tables: &Tensor<u32>,
    i0: usize,
    nblk: usize,
    x: &Tensor<f32>,
    xo: usize,
    out: &mut Tensor<f32>,
    oo: usize,
    #[comptime] emit: bool,
) -> f32 {
    let mut acc = 0.0f32;
    let s0 = rhalf(pool, i0, nblk, 48usize);
    let s1 = rhalf(pool, i0, nblk, 50usize);
    let s2 = rhalf(pool, i0, nblk, 52usize);
    let s3 = rhalf(pool, i0, nblk, 54usize);
    let bits =
        (s0 >> 12u32) | ((s1 >> 8u32) & 0x00F0u32) | ((s2 >> 4u32) & 0x0F00u32) | (s3 & 0xF000u32);
    let d = half_to_f32(bits);
    for ib in 0..8usize {
        let sw = rhalf(pool, i0, nblk, 48usize + 2usize * (ib / 2usize));
        let sh = 6u32 * u32::cast_from(ib % 2usize);
        let dl0 = d * f32::cast_from(2u32 * ((sw >> sh) & 7u32) + 1u32);
        let dl1 = d * f32::cast_from(2u32 * ((sw >> (sh + 3u32)) & 7u32) + 1u32);
        for l in 0..4usize {
            let qhb = rbyte(pool, i0, nblk, 32usize + 2usize * ib + l / 2usize);
            let odd = u32::cast_from(l % 2usize);
            let shift = 8u32 - 4u32 * odd;
            let lo = rbyte(pool, i0, nblk, 4usize * ib + l);
            let idx = usize::cast_from(lo | ((qhb << shift) & 0x700u32));
            let delta = 0.125f32 - 0.25f32 * f32::cast_from((qhb >> (3u32 + 4u32 * odd)) & 1u32);
            let mut dl = dl0;
            if l >= 2usize {
                dl = dl1;
            }
            for j in 0..8usize {
                let g = sext8(tbyte(tables, comptime![T_G1S], idx * 8usize + j));
                let e = 32usize * ib + 8usize * l + j;
                acc += consume(dl * (g + delta), x, xo, out, oo, e, emit);
            }
        }
    }
    acc
}

/// IQ4_XS block (136 bytes): `d`, the high scale bits (`u16`), 4 bytes of
/// low scale bits, 128 bytes of nibbles through `kvalues_iq4nl`.
#[cube]
#[allow(clippy::too_many_arguments)]
fn iq4xs_block(
    pool: &Tensor<u32>,
    tables: &Tensor<u32>,
    i0: usize,
    nblk: usize,
    x: &Tensor<f32>,
    xo: usize,
    out: &mut Tensor<f32>,
    oo: usize,
    #[comptime] emit: bool,
) -> f32 {
    let mut acc = 0.0f32;
    let d = half_to_f32(rhalf(pool, i0, nblk, 0usize));
    let high = rhalf(pool, i0, nblk, 2usize);
    for ib in 0..8usize {
        let low = (rbyte(pool, i0, nblk, 4usize + ib / 2usize)
            >> (4u32 * u32::cast_from(ib % 2usize)))
            & 0xFu32;
        let hi2 = (high >> (2u32 * u32::cast_from(ib))) & 3u32;
        let ls = low | (hi2 << 4u32);
        let dl = d * (f32::cast_from(ls) - 32.0f32);
        for l in 0..16usize {
            let q = rbyte(pool, i0, nblk, 8usize + 16usize * ib + l);
            acc += consume(
                dl * kv4(tables, q & 0xFu32),
                x,
                xo,
                out,
                oo,
                32usize * ib + l,
                emit,
            );
            acc += consume(
                dl * kv4(tables, q >> 4u32),
                x,
                xo,
                out,
                oo,
                32usize * ib + 16usize + l,
                emit,
            );
        }
    }
    acc
}

/// One run of TQ1_0 digits: byte `i` (of `len`) at byte offset `first` holds
/// the digits `n` (0 to `powers`) of elements `at + n * len + i`.
#[cube]
#[allow(clippy::too_many_arguments)]
fn tq1_run(
    pool: &Tensor<u32>,
    i0: usize,
    nblk: usize,
    first: usize,
    len: usize,
    powers: usize,
    at: usize,
    d: f32,
    x: &Tensor<f32>,
    xo: usize,
    out: &mut Tensor<f32>,
    oo: usize,
    #[comptime] emit: bool,
) -> f32 {
    let mut acc = 0.0f32;
    for n in 0..powers {
        let p = pow3(n);
        for i in 0..len {
            let q = rbyte(pool, i0, nblk, first + i);
            let t = (((q * p) & 0xFFu32) * 3u32) >> 8u32;
            let v = f32::cast_from(i32::cast_from(t) - 1i32) * d;
            acc += consume(v, x, xo, out, oo, at + n * len + i, emit);
        }
    }
    acc
}

/// TQ1_0 block (54 bytes): 48 + 4 bytes of base-3 digits (five and four
/// values per byte), `d` last.
#[cube]
#[allow(clippy::too_many_arguments)]
fn tq10_block(
    pool: &Tensor<u32>,
    i0: usize,
    nblk: usize,
    x: &Tensor<f32>,
    xo: usize,
    out: &mut Tensor<f32>,
    oo: usize,
    #[comptime] emit: bool,
) -> f32 {
    let d = half_to_f32(rhalf(pool, i0, nblk, 52usize));
    let mut acc = tq1_run(
        pool, i0, nblk, 0usize, 32usize, 5usize, 0usize, d, x, xo, out, oo, emit,
    );
    acc += tq1_run(
        pool, i0, nblk, 32usize, 16usize, 5usize, 160usize, d, x, xo, out, oo, emit,
    );
    acc += tq1_run(
        pool, i0, nblk, 48usize, 4usize, 4usize, 240usize, d, x, xo, out, oo, emit,
    );
    acc
}

/// TQ2_0 block (66 bytes): 64 bytes of 2-bit values, `d` last.
#[cube]
#[allow(clippy::too_many_arguments)]
fn tq20_block(
    pool: &Tensor<u32>,
    i0: usize,
    nblk: usize,
    x: &Tensor<f32>,
    xo: usize,
    out: &mut Tensor<f32>,
    oo: usize,
    #[comptime] emit: bool,
) -> f32 {
    let mut acc = 0.0f32;
    let d = half_to_f32(rhalf(pool, i0, nblk, 64usize));
    for n in 0..2usize {
        for l in 0..4usize {
            for i in 0..32usize {
                let b = rbyte(pool, i0, nblk, 32usize * n + i);
                let q = (b >> (2u32 * u32::cast_from(l))) & 3u32;
                let v = f32::cast_from(i32::cast_from(q) - 1i32) * d;
                acc += consume(v, x, xo, out, oo, 128usize * n + 32usize * l + i, emit);
            }
        }
    }
    acc
}

/// NVFP4 block (36 bytes, 64 values): four UE4M3 scale bytes, then per 16
/// values 8 bytes of nibbles (low nibbles the first 8, high the last 8).
#[cube]
#[allow(clippy::too_many_arguments)]
fn nvfp4_block(
    pool: &Tensor<u32>,
    tables: &Tensor<u32>,
    i0: usize,
    nblk: usize,
    x: &Tensor<f32>,
    xo: usize,
    out: &mut Tensor<f32>,
    oo: usize,
    #[comptime] emit: bool,
) -> f32 {
    let mut acc = 0.0f32;
    for g in 0..4usize {
        let s = usize::cast_from(rbyte(pool, i0, nblk, g));
        let d = f32::reinterpret(tables[comptime![T_UE4M3] + s]);
        for i in 0..8usize {
            let q = rbyte(pool, i0, nblk, 4usize + 8usize * g + i);
            acc += consume(
                mxfp4_value(q & 0xFu32) * d,
                x,
                xo,
                out,
                oo,
                16usize * g + i,
                emit,
            );
            acc += consume(
                mxfp4_value(q >> 4u32) * d,
                x,
                xo,
                out,
                oo,
                16usize * g + 8usize + i,
                emit,
            );
        }
    }
    acc
}

/// The dot product of block `i0 + w * nblk` of a native-format matrix (word
/// `w` of the block at `i0 + w * nblk` of `pool`) with `x[xo..]`, or (`emit`)
/// its values stored at `out[oo..]`, which returns 0.
#[cube]
#[allow(clippy::too_many_arguments)]
pub(super) fn native_block(
    #[comptime] format: u32,
    pool: &Tensor<u32>,
    tables: &Tensor<u32>,
    i0: usize,
    nblk: usize,
    x: &Tensor<f32>,
    xo: usize,
    out: &mut Tensor<f32>,
    oo: usize,
    #[comptime] emit: bool,
) -> f32 {
    if comptime![format == code::Q2_K] {
        q2k_block(pool, i0, nblk, x, xo, out, oo, emit)
    } else if comptime![format == code::Q3_K] {
        q3k_block(pool, i0, nblk, x, xo, out, oo, emit)
    } else if comptime![format == code::Q4_K] {
        q4k_block(pool, i0, nblk, x, xo, out, oo, emit)
    } else if comptime![format == code::Q5_K] {
        q5k_block(pool, i0, nblk, x, xo, out, oo, emit)
    } else if comptime![format == code::Q6_K] {
        q6k_block(pool, i0, nblk, x, xo, out, oo, emit)
    } else if comptime![format == code::IQ4_XS] {
        iq4xs_block(pool, tables, i0, nblk, x, xo, out, oo, emit)
    } else if comptime![format == code::IQ2_XXS] {
        iq2xxs_block(pool, tables, i0, nblk, x, xo, out, oo, emit)
    } else if comptime![format == code::IQ2_XS] {
        iq2xs_block(pool, tables, i0, nblk, x, xo, out, oo, emit)
    } else if comptime![format == code::IQ2_S] {
        iq2s_block(pool, tables, i0, nblk, x, xo, out, oo, emit)
    } else if comptime![format == code::IQ3_XXS] {
        iq3xxs_block(pool, tables, i0, nblk, x, xo, out, oo, emit)
    } else if comptime![format == code::IQ3_S] {
        iq3s_block(pool, tables, i0, nblk, x, xo, out, oo, emit)
    } else if comptime![format == code::IQ1_S] {
        iq1s_block(pool, tables, i0, nblk, x, xo, out, oo, emit)
    } else if comptime![format == code::IQ1_M] {
        iq1m_block(pool, tables, i0, nblk, x, xo, out, oo, emit)
    } else if comptime![format == code::TQ1_0] {
        tq10_block(pool, i0, nblk, x, xo, out, oo, emit)
    } else if comptime![format == code::TQ2_0] {
        tq20_block(pool, i0, nblk, x, xo, out, oo, emit)
    } else if comptime![format == code::Q5_0] {
        q5_block(pool, i0, nblk, x, xo, out, oo, emit, false)
    } else if comptime![format == code::Q5_1] {
        q5_block(pool, i0, nblk, x, xo, out, oo, emit, true)
    } else if comptime![format == code::Q8_K] {
        q8k_block(pool, i0, nblk, x, xo, out, oo, emit)
    } else {
        nvfp4_block(pool, tables, i0, nblk, x, xo, out, oo, emit)
    }
}
