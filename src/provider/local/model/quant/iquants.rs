//! The I-quants (IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL,
//! IQ4_XS), the ternary types TQ1_0 and TQ2_0, MXFP4 and NVFP4: in-place f32
//! decoders and integer unpackers for one block.
//!
//! Ported operation for operation from llama.cpp's `dequantize_row_*`
//! (`ggml/src/ggml-quants.c`, commit named in `iq_tables.rs`), so every f32
//! result is the one llama.cpp and gguf-py compute. A block is 256 values
//! except IQ4_NL and MXFP4 (32) and NVFP4 (64). Layouts (`ggml-common.h`), in
//! bytes, `d` an f16 unless noted:
//!
//! - IQ2_XXS, 66: `d`, then per 32 values two `u32`: four grid indices (one
//!   byte each) and 4 x 7 sign indices plus a 4-bit scale in the top bits.
//! - IQ2_XS, 74: `d`, 32 `u16` (9-bit grid index, 7-bit sign index), 8 scale
//!   bytes (a nibble per 16 values).
//! - IQ2_S, 82: `d`, 32 bytes of low grid index bits, 32 sign bytes, 8 bytes of
//!   high grid index bits (2 per 8 values), 8 scale bytes.
//! - IQ3_XXS, 98: `d`, 64 grid index bytes (4 values each), then per 32 values
//!   a `u32` of 4 x 7 sign indices and a 4-bit scale in the top bits.
//! - IQ3_S, 110: `d`, 64 low grid index bytes, 8 bytes of high index bits, 32
//!   sign bytes, 4 scale bytes (a nibble per 32 values).
//! - IQ1_S, 50: `d`, 32 low grid index bytes, 8 `u16` (3 index bits per 8
//!   values, a 3-bit scale, a delta sign).
//! - IQ1_M, 56: 32 low grid index bytes, 16 bytes of (3 index bits, a delta
//!   sign) per 8 values, 4 `u16` of 3-bit scales whose top nibbles hold the
//!   f16 `d`.
//! - IQ4_NL, 18: `d`, 16 bytes of nibbles through a 16-entry table.
//! - IQ4_XS, 136: `d`, `u16` high scale bits, 4 bytes of low scale bits, 128
//!   bytes of nibbles through the IQ4_NL table.
//! - TQ1_0, 54: 48 bytes and 4 bytes of base-3 digits (five and four values
//!   per byte), `d` last.
//! - TQ2_0, 66: 64 bytes of 2-bit values, `d` last.
//! - MXFP4, 17: one E8M0 scale byte, 16 bytes of nibbles.
//! - NVFP4, 36: four UE4M3 scale bytes (one per 16 values), 32 bytes of nibbles.

use super::block::{Unpacked, f16_at, u16_at, u32_at};
use super::e8m0_half;
use super::iq_tables::{
    IQ1S_GRID, IQ2S_GRID, IQ2XS_GRID, IQ2XXS_GRID, IQ3S_GRID, IQ3XXS_GRID, KMASK_IQ2XS,
    KSIGNS_IQ2XS, KVALUES_FP4, KVALUES_IQ4NL,
};

/// Bytes of one IQ2_XXS block.
pub(crate) const IQ2_XXS_BYTES: usize = 66;
/// Bytes of one IQ2_XS block.
pub(crate) const IQ2_XS_BYTES: usize = 74;
/// Bytes of one IQ2_S block.
pub(crate) const IQ2_S_BYTES: usize = 82;
/// Bytes of one IQ3_XXS block.
pub(crate) const IQ3_XXS_BYTES: usize = 98;
/// Bytes of one IQ3_S block.
pub(crate) const IQ3_S_BYTES: usize = 110;
/// Bytes of one IQ1_S block.
pub(crate) const IQ1_S_BYTES: usize = 50;
/// Bytes of one IQ1_M block.
pub(crate) const IQ1_M_BYTES: usize = 56;
/// Bytes of one IQ4_NL block (32 values).
pub(crate) const IQ4_NL_BYTES: usize = 18;
/// Bytes of one IQ4_XS block.
pub(crate) const IQ4_XS_BYTES: usize = 136;
/// Bytes of one TQ1_0 block.
pub(crate) const TQ1_0_BYTES: usize = 54;
/// Bytes of one TQ2_0 block.
pub(crate) const TQ2_0_BYTES: usize = 66;
/// Bytes of one MXFP4 block (32 values).
pub(crate) const MXFP4_BYTES: usize = 17;
/// Bytes of one NVFP4 block (64 values).
pub(crate) const NVFP4_BYTES: usize = 36;
/// Values of one IQ4_NL or MXFP4 block.
pub(crate) const QK32: usize = 32;
/// Values of one NVFP4 block.
pub(crate) const QK_NVFP4: usize = 64;
/// Values of every other block of this file.
pub(crate) const QK_K: usize = 256;

/// llama.cpp's `IQ1S_DELTA` and `IQ1M_DELTA`.
const IQ1_DELTA: f32 = 0.125;

/// The f32 value of UE4M3 scale byte `x` times 0.5 (`ggml_ue4m3_to_fp32`):
/// zero and 0x7F decode to 0 and the sign bit is ignored, as llama.cpp does.
const fn ue4m3_half(x: u8) -> f32 {
    if x == 0 || x == 0x7F {
        return 0.0;
    }
    let exp = ((x >> 3) & 0x0F) as i32;
    let man = (x & 7) as f32;
    let raw = if exp == 0 {
        man * pow2(-9)
    } else {
        (1.0 + man / 8.0) * pow2(exp - 7)
    };
    raw * 0.5
}

/// `2^e` for `e` in the normal f32 range.
const fn pow2(e: i32) -> f32 {
    f32::from_bits(((e + 127) as u32) << 23)
}

/// [`ue4m3_half`] of every byte: NVFP4's per-16-values scale.
pub(crate) const UE4M3_HALF: [f32; 256] = {
    let mut t = [0.0; 256];
    let mut i = 0;
    while i < 256 {
        t[i] = ue4m3_half(i as u8);
        i += 1;
    }
    t
};

/// 8 (or 4) grid values `grid` (byte `j` is value `j`), times `db`, negated
/// where bit `first + j` of `signs` is set.
#[inline]
fn put<const N: usize>(y: &mut [f32], grid: [u8; N], signs: u8, first: usize, db: f32) {
    for (j, (y, g)) in y.iter_mut().zip(grid).enumerate() {
        let v = db * f32::from(g);
        *y = if signs & KMASK_IQ2XS[first + j] != 0 {
            -v
        } else {
            v
        };
    }
}

/// The grid values `grid` as signed integers, negated where bit `first + j` of
/// `signs` is set.
#[inline]
fn put_int<const N: usize>(y: &mut [i8], grid: [u8; N], signs: u8, first: usize) {
    for (j, (y, g)) in y.iter_mut().zip(grid).enumerate() {
        let v = g.cast_signed();
        *y = if signs & KMASK_IQ2XS[first + j] != 0 {
            -v
        } else {
            v
        };
    }
}

/// The 8 IQ1 grid values of `grid` (as `i8`) plus `delta`, times `dl`.
#[inline]
fn put_iq1(y: &mut [f32], grid: u64, delta: f32, dl: f32) {
    for (y, g) in y.iter_mut().zip(grid.to_le_bytes()) {
        *y = dl * (f32::from(g.cast_signed()) + delta);
    }
}

/// The IQ1 delta of a block's sign bit, as llama.cpp writes it.
#[inline]
fn iq1_delta(negative: bool) -> f32 {
    if negative { -IQ1_DELTA } else { IQ1_DELTA }
}

/// The 8 IQ1 grid values of `grid` times 8 plus or minus 1 (`delta` times 8):
/// the exact integer form of `grid + delta`.
#[inline]
fn put_iq1_int(y: &mut [i8], grid: u64, negative: bool) {
    let delta = if negative { -1 } else { 1 };
    for (y, g) in y.iter_mut().zip(grid.to_le_bytes()) {
        *y = g.cast_signed() * 8 + delta;
    }
}

/// Decode one IQ2_XXS block.
pub(crate) fn dequant_iq2_xxs(b: &[u8; IQ2_XXS_BYTES], out: &mut [f32; QK_K]) {
    let d = f16_at(b, 0);
    for (qs, y) in b[2..]
        .as_chunks::<8>()
        .0
        .iter()
        .zip(out.as_chunks_mut::<32>().0.iter_mut())
    {
        let aux = u32_at(qs, 4);
        let db = d * (0.5 + f32::from((aux >> 28) as u8)) * 0.25;
        for (l, y) in y.as_chunks_mut::<8>().0.iter_mut().enumerate() {
            let grid = IQ2XXS_GRID[usize::from(qs[l])].to_le_bytes();
            let signs = KSIGNS_IQ2XS[((aux >> (7 * l)) & 127) as usize];
            put(y, grid, signs, 0, db);
        }
    }
}

/// Unpack one IQ2_XXS block: 8 sub-blocks of 32 values, raw quants the signed
/// grid values (+-8, +-25, +-43), `sc` odd 1 to 31, `d / 8`.
pub(crate) fn unpack_iq2_xxs(b: &[u8; IQ2_XXS_BYTES], out: &mut Unpacked) {
    out.reset(QK_K, 32);
    let d = f16_at(b, 0) * 0.125;
    for (ib, (qs, y)) in b[2..]
        .as_chunks::<8>()
        .0
        .iter()
        .zip(out.q.as_chunks_mut::<32>().0.iter_mut())
        .enumerate()
    {
        let aux = u32_at(qs, 4);
        out.d[ib] = d;
        out.sc[ib] = 2 * i16::from((aux >> 28) as u8) + 1;
        for (l, y) in y.as_chunks_mut::<8>().0.iter_mut().enumerate() {
            let grid = IQ2XXS_GRID[usize::from(qs[l])].to_le_bytes();
            put_int(y, grid, KSIGNS_IQ2XS[((aux >> (7 * l)) & 127) as usize], 0);
        }
    }
}

/// Decode one IQ2_XS block.
pub(crate) fn dequant_iq2_xs(b: &[u8; IQ2_XS_BYTES], out: &mut [f32; QK_K]) {
    let d = f16_at(b, 0);
    let (qs, scales) = b[2..].split_at(64);
    for ((qs, &sc), y) in qs
        .as_chunks::<8>()
        .0
        .iter()
        .zip(scales)
        .zip(out.as_chunks_mut::<32>().0.iter_mut())
    {
        let db = [
            d * (0.5 + f32::from(sc & 0x0F)) * 0.25,
            d * (0.5 + f32::from(sc >> 4)) * 0.25,
        ];
        for (l, y) in y.as_chunks_mut::<8>().0.iter_mut().enumerate() {
            let v = usize::from(u16_at(qs, 2 * l));
            let grid = IQ2XS_GRID[v & 511].to_le_bytes();
            put(y, grid, KSIGNS_IQ2XS[v >> 9], 0, db[l / 2]);
        }
    }
}

/// Unpack one IQ2_XS block: 16 sub-blocks of 16 values, raw quants the signed
/// grid values (+-8, +-25, +-43), `sc` odd 1 to 31, `d / 8`.
pub(crate) fn unpack_iq2_xs(b: &[u8; IQ2_XS_BYTES], out: &mut Unpacked) {
    out.reset(QK_K, 16);
    let d = f16_at(b, 0) * 0.125;
    out.d = [d; 16];
    let (qs, scales) = b[2..].split_at(64);
    for (ib, ((qs, &sc), y)) in qs
        .as_chunks::<8>()
        .0
        .iter()
        .zip(scales)
        .zip(out.q.as_chunks_mut::<32>().0.iter_mut())
        .enumerate()
    {
        out.sc[2 * ib] = 2 * i16::from(sc & 0x0F) + 1;
        out.sc[2 * ib + 1] = 2 * i16::from(sc >> 4) + 1;
        for (l, y) in y.as_chunks_mut::<8>().0.iter_mut().enumerate() {
            let v = usize::from(u16_at(qs, 2 * l));
            put_int(
                y,
                IQ2XS_GRID[v & 511].to_le_bytes(),
                KSIGNS_IQ2XS[v >> 9],
                0,
            );
        }
    }
}

/// The IQ2_S grid index of 8-value group `l` (0 to 3) of a 32-value sub-block
/// with index bytes `qs` (4 per sub-block) and high bits `qh`.
#[inline]
fn iq2_s_index(qs: &[u8], qh: u8, l: usize) -> usize {
    usize::from(qs[l]) | ((usize::from(qh) << (8 - 2 * l)) & 0x300)
}

/// Decode one IQ2_S block.
pub(crate) fn dequant_iq2_s(b: &[u8; IQ2_S_BYTES], out: &mut [f32; QK_K]) {
    let d = f16_at(b, 0);
    let (qs, rest) = b[2..].split_at(32);
    let (signs, rest) = rest.split_at(32);
    let (qh, scales) = rest.split_at(8);
    for ((((qs, signs), &qh), &sc), y) in qs
        .as_chunks::<4>()
        .0
        .iter()
        .zip(signs.as_chunks::<4>().0.iter())
        .zip(qh)
        .zip(scales)
        .zip(out.as_chunks_mut::<32>().0.iter_mut())
    {
        let db = [
            d * (0.5 + f32::from(sc & 0x0F)) * 0.25,
            d * (0.5 + f32::from(sc >> 4)) * 0.25,
        ];
        for (l, y) in y.as_chunks_mut::<8>().0.iter_mut().enumerate() {
            let grid = IQ2S_GRID[iq2_s_index(qs, qh, l)].to_le_bytes();
            put(y, grid, signs[l], 0, db[l / 2]);
        }
    }
}

/// Unpack one IQ2_S block: 16 sub-blocks of 16 values, raw quants the signed
/// grid values (+-8, +-25, +-43), `sc` odd 1 to 31, `d / 8`.
pub(crate) fn unpack_iq2_s(b: &[u8; IQ2_S_BYTES], out: &mut Unpacked) {
    out.reset(QK_K, 16);
    out.d = [f16_at(b, 0) * 0.125; 16];
    let (qs, rest) = b[2..].split_at(32);
    let (signs, rest) = rest.split_at(32);
    let (qh, scales) = rest.split_at(8);
    for (ib, ((((qs, signs), &qh), &sc), y)) in qs
        .as_chunks::<4>()
        .0
        .iter()
        .zip(signs.as_chunks::<4>().0.iter())
        .zip(qh)
        .zip(scales)
        .zip(out.q.as_chunks_mut::<32>().0.iter_mut())
        .enumerate()
    {
        out.sc[2 * ib] = 2 * i16::from(sc & 0x0F) + 1;
        out.sc[2 * ib + 1] = 2 * i16::from(sc >> 4) + 1;
        for (l, y) in y.as_chunks_mut::<8>().0.iter_mut().enumerate() {
            put_int(
                y,
                IQ2S_GRID[iq2_s_index(qs, qh, l)].to_le_bytes(),
                signs[l],
                0,
            );
        }
    }
}

/// Decode one IQ3_XXS block.
pub(crate) fn dequant_iq3_xxs(b: &[u8; IQ3_XXS_BYTES], out: &mut [f32; QK_K]) {
    let d = f16_at(b, 0);
    let (qs, sas) = b[2..].split_at(64);
    for ((qs, sas), y) in qs
        .as_chunks::<8>()
        .0
        .iter()
        .zip(sas.as_chunks::<4>().0.iter())
        .zip(out.as_chunks_mut::<32>().0.iter_mut())
    {
        let aux = u32_at(sas, 0);
        let db = d * (0.5 + f32::from((aux >> 28) as u8)) * 0.5;
        for (l, y) in y.as_chunks_mut::<8>().0.iter_mut().enumerate() {
            let signs = KSIGNS_IQ2XS[((aux >> (7 * l)) & 127) as usize];
            let (lo, hi) = y.split_at_mut(4);
            put(
                lo,
                IQ3XXS_GRID[usize::from(qs[2 * l])].to_le_bytes(),
                signs,
                0,
                db,
            );
            put(
                hi,
                IQ3XXS_GRID[usize::from(qs[2 * l + 1])].to_le_bytes(),
                signs,
                4,
                db,
            );
        }
    }
}

/// Unpack one IQ3_XXS block: 8 sub-blocks of 32 values, raw quants the signed
/// grid values (+-4 to +-62), `sc` odd 1 to 31, `d / 4`.
pub(crate) fn unpack_iq3_xxs(b: &[u8; IQ3_XXS_BYTES], out: &mut Unpacked) {
    out.reset(QK_K, 32);
    let d = f16_at(b, 0) * 0.25;
    let (qs, sas) = b[2..].split_at(64);
    for (ib, ((qs, sas), y)) in qs
        .as_chunks::<8>()
        .0
        .iter()
        .zip(sas.as_chunks::<4>().0.iter())
        .zip(out.q.as_chunks_mut::<32>().0.iter_mut())
        .enumerate()
    {
        let aux = u32_at(sas, 0);
        out.d[ib] = d;
        out.sc[ib] = 2 * i16::from((aux >> 28) as u8) + 1;
        for (l, y) in y.as_chunks_mut::<8>().0.iter_mut().enumerate() {
            let signs = KSIGNS_IQ2XS[((aux >> (7 * l)) & 127) as usize];
            let (lo, hi) = y.split_at_mut(4);
            put_int(
                lo,
                IQ3XXS_GRID[usize::from(qs[2 * l])].to_le_bytes(),
                signs,
                0,
            );
            put_int(
                hi,
                IQ3XXS_GRID[usize::from(qs[2 * l + 1])].to_le_bytes(),
                signs,
                4,
            );
        }
    }
}

/// The two IQ3_S grid indices of 8-value group `l` (0 to 3) of a 32-value
/// sub-block with high bits `qh` and index bytes `qs` (8 per sub-block).
#[inline]
fn iq3_s_indices(qs: &[u8], qh: u8, l: usize) -> (usize, usize) {
    let qh = usize::from(qh);
    (
        usize::from(qs[2 * l]) | ((qh << (8 - 2 * l)) & 256),
        usize::from(qs[2 * l + 1]) | ((qh << (7 - 2 * l)) & 256),
    )
}

/// Decode one IQ3_S block.
pub(crate) fn dequant_iq3_s(b: &[u8; IQ3_S_BYTES], out: &mut [f32; QK_K]) {
    let d = f16_at(b, 0);
    let (qs, rest) = b[2..].split_at(64);
    let (qh, rest) = rest.split_at(8);
    let (signs, scales) = rest.split_at(32);
    for ((((qs, &qh), signs), y), db) in qs
        .as_chunks::<8>()
        .0
        .iter()
        .zip(qh)
        .zip(signs.as_chunks::<4>().0.iter())
        .zip(out.as_chunks_mut::<32>().0.iter_mut())
        .zip(
            scales
                .iter()
                .flat_map(|&s| [s & 0x0F, s >> 4])
                .map(|s| d * f32::from(1 + 2 * s)),
        )
    {
        for (l, y) in y.as_chunks_mut::<8>().0.iter_mut().enumerate() {
            let (i1, i2) = iq3_s_indices(qs, qh, l);
            let (lo, hi) = y.split_at_mut(4);
            put(lo, IQ3S_GRID[i1].to_le_bytes(), signs[l], 0, db);
            put(hi, IQ3S_GRID[i2].to_le_bytes(), signs[l], 4, db);
        }
    }
}

/// Unpack one IQ3_S block: 8 sub-blocks of 32 values, raw quants the signed
/// grid values (+-1 to +-15, odd), `sc` odd 1 to 31, `d`.
pub(crate) fn unpack_iq3_s(b: &[u8; IQ3_S_BYTES], out: &mut Unpacked) {
    out.reset(QK_K, 32);
    out.d = [f16_at(b, 0); 16];
    let (qs, rest) = b[2..].split_at(64);
    let (qh, rest) = rest.split_at(8);
    let (signs, scales) = rest.split_at(32);
    for (ib, (((qs, &qh), signs), y)) in qs
        .as_chunks::<8>()
        .0
        .iter()
        .zip(qh)
        .zip(signs.as_chunks::<4>().0.iter())
        .zip(out.q.as_chunks_mut::<32>().0.iter_mut())
        .enumerate()
    {
        let sc = scales[ib / 2] >> (4 * (ib % 2)) & 0x0F;
        out.sc[ib] = 1 + 2 * i16::from(sc);
        for (l, y) in y.as_chunks_mut::<8>().0.iter_mut().enumerate() {
            let (i1, i2) = iq3_s_indices(qs, qh, l);
            let (lo, hi) = y.split_at_mut(4);
            put_int(lo, IQ3S_GRID[i1].to_le_bytes(), signs[l], 0);
            put_int(hi, IQ3S_GRID[i2].to_le_bytes(), signs[l], 4);
        }
    }
}

/// Decode one IQ1_S block.
pub(crate) fn dequant_iq1_s(b: &[u8; IQ1_S_BYTES], out: &mut [f32; QK_K]) {
    let d = f16_at(b, 0);
    let (qs, qh) = b[2..].split_at(32);
    for (ib, (qs, y)) in qs
        .as_chunks::<4>()
        .0
        .iter()
        .zip(out.as_chunks_mut::<32>().0.iter_mut())
        .enumerate()
    {
        let qh = u16_at(qh, 2 * ib);
        let dl = d * f32::from(2 * ((qh >> 12) & 7) + 1);
        let delta = iq1_delta(qh & 0x8000 != 0);
        for (l, y) in y.as_chunks_mut::<8>().0.iter_mut().enumerate() {
            let grid = IQ1S_GRID[usize::from(qs[l]) | (usize::from((qh >> (3 * l)) & 7) << 8)];
            put_iq1(y, grid, delta, dl);
        }
    }
}

/// Unpack one IQ1_S block: 8 sub-blocks of 32 values, raw quants `8 * grid +- 1`
/// (+-1, +-7, +-9), `sc` odd 1 to 15, `d / 8`.
pub(crate) fn unpack_iq1_s(b: &[u8; IQ1_S_BYTES], out: &mut Unpacked) {
    out.reset(QK_K, 32);
    let d = f16_at(b, 0) * 0.125;
    let (qs, qh) = b[2..].split_at(32);
    for (ib, (qs, y)) in qs
        .as_chunks::<4>()
        .0
        .iter()
        .zip(out.q.as_chunks_mut::<32>().0.iter_mut())
        .enumerate()
    {
        let qh = u16_at(qh, 2 * ib);
        out.d[ib] = d;
        out.sc[ib] = 2 * i16::from((qh >> 12) as u8 & 7) + 1;
        for (l, y) in y.as_chunks_mut::<8>().0.iter_mut().enumerate() {
            let grid = IQ1S_GRID[usize::from(qs[l]) | (usize::from((qh >> (3 * l)) & 7) << 8)];
            put_iq1_int(y, grid, qh & 0x8000 != 0);
        }
    }
}

/// The f16 `d` of an IQ1_M block, gathered from the top nibbles of its four
/// 16-bit scale words (`sc`).
#[inline]
fn iq1_m_d(sc: [u16; 4]) -> f32 {
    let bits = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0) | ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000);
    half::f16::from_bits(bits).to_f32()
}

/// The four 16-bit scale words of an IQ1_M block.
#[inline]
fn iq1_m_words(scales: &[u8]) -> [u16; 4] {
    [
        u16_at(scales, 0),
        u16_at(scales, 2),
        u16_at(scales, 4),
        u16_at(scales, 6),
    ]
}

/// Decode one IQ1_M block.
pub(crate) fn dequant_iq1_m(b: &[u8; IQ1_M_BYTES], out: &mut [f32; QK_K]) {
    let (qs, rest) = b.split_at(32);
    let (qh, scales) = rest.split_at(16);
    let sc = iq1_m_words(scales);
    let d = iq1_m_d(sc);
    for (ib, ((qs, qh), y)) in qs
        .as_chunks::<4>()
        .0
        .iter()
        .zip(qh.as_chunks::<2>().0.iter())
        .zip(out.as_chunks_mut::<32>().0.iter_mut())
        .enumerate()
    {
        let word = sc[ib / 2];
        let dl = [
            d * f32::from(2 * ((word >> (6 * (ib % 2))) & 7) + 1),
            d * f32::from(2 * ((word >> (6 * (ib % 2) + 3)) & 7) + 1),
        ];
        for (l, y) in y.as_chunks_mut::<8>().0.iter_mut().enumerate() {
            let (qh, shift) = (qh[l / 2], if l % 2 == 0 { 8 } else { 4 });
            let idx = usize::from(qs[l]) | ((usize::from(qh) << shift) & 0x700);
            let delta = iq1_delta(qh & if l % 2 == 0 { 0x08 } else { 0x80 } != 0);
            put_iq1(y, IQ1S_GRID[idx], delta, dl[l / 2]);
        }
    }
}

/// Unpack one IQ1_M block: 16 sub-blocks of 16 values, raw quants `8 * grid +-
/// 1` (+-1, +-7, +-9), `sc` odd 1 to 15, `d / 8`.
pub(crate) fn unpack_iq1_m(b: &[u8; IQ1_M_BYTES], out: &mut Unpacked) {
    let (qs, rest) = b.split_at(32);
    let (qh, scales) = rest.split_at(16);
    let sc = iq1_m_words(scales);
    out.reset(QK_K, 16);
    out.d = [iq1_m_d(sc) * 0.125; 16];
    for (ib, ((qs, qh), y)) in qs
        .as_chunks::<4>()
        .0
        .iter()
        .zip(qh.as_chunks::<2>().0.iter())
        .zip(out.q.as_chunks_mut::<32>().0.iter_mut())
        .enumerate()
    {
        let word = sc[ib / 2];
        out.sc[2 * ib] = 2 * i16::from((word >> (6 * (ib % 2))) as u8 & 7) + 1;
        out.sc[2 * ib + 1] = 2 * i16::from((word >> (6 * (ib % 2) + 3)) as u8 & 7) + 1;
        for (l, y) in y.as_chunks_mut::<8>().0.iter_mut().enumerate() {
            let (qh, shift) = (qh[l / 2], if l % 2 == 0 { 8 } else { 4 });
            let idx = usize::from(qs[l]) | ((usize::from(qh) << shift) & 0x700);
            put_iq1_int(
                y,
                IQ1S_GRID[idx],
                qh & if l % 2 == 0 { 0x08 } else { 0x80 } != 0,
            );
        }
    }
}

/// Unpack one IQ4_NL block: 32 values in one sub-block, raw quants the table
/// values (-127 to 113), `sc` 1, `d`.
pub(crate) fn unpack_iq4_nl(b: &[u8; IQ4_NL_BYTES], out: &mut Unpacked) {
    out.reset(QK32, QK32);
    out.d[0] = f16_at(b, 0);
    let (lo, hi) = out.q[..QK32].split_at_mut(16);
    for ((lo, hi), &q) in lo.iter_mut().zip(hi).zip(&b[2..]) {
        *lo = KVALUES_IQ4NL[usize::from(q & 0x0F)];
        *hi = KVALUES_IQ4NL[usize::from(q >> 4)];
    }
}

/// The scale `ls` (0 to 63, before the offset of 32) of 32-value sub-block
/// `ib` of an IQ4_XS block.
#[inline]
fn iq4_xs_scale(b: &[u8; IQ4_XS_BYTES], ib: usize) -> i16 {
    let low = (b[4 + ib / 2] >> (4 * (ib % 2))) & 0x0F;
    let high = ((u16_at(b, 2) >> (2 * ib)) & 3) as u8;
    i16::from(low | (high << 4))
}

/// Decode one IQ4_XS block.
pub(crate) fn dequant_iq4_xs(b: &[u8; IQ4_XS_BYTES], out: &mut [f32; QK_K]) {
    let d = f16_at(b, 0);
    for (ib, (qs, y)) in b[8..]
        .as_chunks::<16>()
        .0
        .iter()
        .zip(out.as_chunks_mut::<32>().0.iter_mut())
        .enumerate()
    {
        let dl = d * f32::from(iq4_xs_scale(b, ib) - 32);
        let (lo, hi) = y.split_at_mut(16);
        for ((lo, hi), &q) in lo.iter_mut().zip(hi).zip(qs) {
            *lo = dl * f32::from(KVALUES_IQ4NL[usize::from(q & 0x0F)]);
            *hi = dl * f32::from(KVALUES_IQ4NL[usize::from(q >> 4)]);
        }
    }
}

/// Unpack one IQ4_XS block: 8 sub-blocks of 32 values, raw quants the table
/// values (-127 to 113), `sc` -32 to 31, `d`.
pub(crate) fn unpack_iq4_xs(b: &[u8; IQ4_XS_BYTES], out: &mut Unpacked) {
    out.reset(QK_K, 32);
    out.d = [f16_at(b, 0); 16];
    for (ib, (qs, y)) in b[8..]
        .as_chunks::<16>()
        .0
        .iter()
        .zip(out.q.as_chunks_mut::<32>().0.iter_mut())
        .enumerate()
    {
        out.sc[ib] = iq4_xs_scale(b, ib) - 32;
        let (lo, hi) = y.split_at_mut(16);
        for ((lo, hi), &q) in lo.iter_mut().zip(hi).zip(qs) {
            *lo = KVALUES_IQ4NL[usize::from(q & 0x0F)];
            *hi = KVALUES_IQ4NL[usize::from(q >> 4)];
        }
    }
}

/// `3^n` for the base-3 digits of TQ1_0.
const POW3: [u8; 6] = [1, 3, 9, 27, 81, 243];

/// The ternary value (-1, 0, 1) of digit `n` of TQ1_0 byte `q`.
#[inline]
fn tq1_trit(q: u8, n: usize) -> i8 {
    ((u16::from(q.wrapping_mul(POW3[n])) * 3) >> 8) as i8 - 1
}

/// Every digit of TQ1_0 bytes `src` (`powers` digits each, digit-major) into
/// `y`.
fn tq1_run(src: &[u8], powers: usize, y: &mut [f32], d: f32) {
    for (n, y) in y.chunks_exact_mut(src.len()).take(powers).enumerate() {
        for (y, &q) in y.iter_mut().zip(src) {
            *y = f32::from(tq1_trit(q, n)) * d;
        }
    }
}

/// Decode one TQ1_0 block.
pub(crate) fn dequant_tq1_0(b: &[u8; TQ1_0_BYTES], out: &mut [f32; QK_K]) {
    let d = f16_at(b, 52);
    let (y0, rest) = out.split_at_mut(160);
    let (y1, y2) = rest.split_at_mut(80);
    tq1_run(&b[..32], 5, y0, d);
    tq1_run(&b[32..48], 5, y1, d);
    tq1_run(&b[48..52], 4, y2, d);
}

/// Unpack one TQ1_0 block: one sub-block of 256 values, raw quants -1 to 1,
/// `sc` 1, `d`.
pub(crate) fn unpack_tq1_0(b: &[u8; TQ1_0_BYTES], out: &mut Unpacked) {
    out.reset(QK_K, QK_K);
    out.d[0] = f16_at(b, 52);
    let (y0, rest) = out.q.split_at_mut(160);
    let (y1, y2) = rest.split_at_mut(80);
    for (src, powers, y) in [
        (&b[..32], 5, y0),
        (&b[32..48], 5, y1),
        (&b[48..52], 4, &mut y2[..16]),
    ] {
        for (n, y) in y.chunks_exact_mut(src.len()).take(powers).enumerate() {
            for (y, &q) in y.iter_mut().zip(src) {
                *y = tq1_trit(q, n);
            }
        }
    }
}

/// Decode one TQ2_0 block.
pub(crate) fn dequant_tq2_0(b: &[u8; TQ2_0_BYTES], out: &mut [f32; QK_K]) {
    let d = f16_at(b, 64);
    for (qs, y) in b[..64]
        .as_chunks::<32>()
        .0
        .iter()
        .zip(out.as_chunks_mut::<128>().0.iter_mut())
    {
        for (l, y) in y.as_chunks_mut::<32>().0.iter_mut().enumerate() {
            for (y, &q) in y.iter_mut().zip(qs) {
                *y = f32::from(((q >> (2 * l)) & 3).cast_signed() - 1) * d;
            }
        }
    }
}

/// Unpack one TQ2_0 block: one sub-block of 256 values, raw quants -1 to 2
/// (the 2-bit value minus 1; llama.cpp's quantizer never writes 3), `sc` 1,
/// `d`.
pub(crate) fn unpack_tq2_0(b: &[u8; TQ2_0_BYTES], out: &mut Unpacked) {
    out.reset(QK_K, QK_K);
    out.d[0] = f16_at(b, 64);
    for (qs, y) in b[..64]
        .as_chunks::<32>()
        .0
        .iter()
        .zip(out.q.as_chunks_mut::<128>().0.iter_mut())
    {
        for (l, y) in y.as_chunks_mut::<32>().0.iter_mut().enumerate() {
            for (y, &q) in y.iter_mut().zip(qs) {
                *y = ((q >> (2 * l)) & 3).cast_signed() - 1;
            }
        }
    }
}

/// Unpack one MXFP4 block (32 values): one sub-block, raw quants the doubled
/// E2M1 values (+-12 at most), `sc` 1, `d` the E8M0 byte as a half scale
/// ([`e8m0_half`]). The f32 decoder is [`super::dequant_row`].
pub(crate) fn unpack_mxfp4(b: &[u8; MXFP4_BYTES], out: &mut Unpacked) {
    out.reset(QK32, QK32);
    out.d[0] = e8m0_half(b[0]);
    let (lo, hi) = out.q[..QK32].split_at_mut(16);
    for ((lo, hi), &q) in lo.iter_mut().zip(hi).zip(&b[1..]) {
        *lo = KVALUES_FP4[usize::from(q & 0x0F)];
        *hi = KVALUES_FP4[usize::from(q >> 4)];
    }
}

/// Decode one NVFP4 block (64 values).
pub(crate) fn dequant_nvfp4(b: &[u8; NVFP4_BYTES], out: &mut [f32; QK_NVFP4]) {
    let (scales, qs) = b.split_at(4);
    for ((&s, qs), y) in scales
        .iter()
        .zip(qs.as_chunks::<8>().0.iter())
        .zip(out.as_chunks_mut::<16>().0.iter_mut())
    {
        let d = UE4M3_HALF[usize::from(s)];
        let (lo, hi) = y.split_at_mut(8);
        for ((lo, hi), &q) in lo.iter_mut().zip(hi).zip(qs) {
            *lo = f32::from(KVALUES_FP4[usize::from(q & 0x0F)]) * d;
            *hi = f32::from(KVALUES_FP4[usize::from(q >> 4)]) * d;
        }
    }
}

/// Unpack one NVFP4 block: 4 sub-blocks of 16 values, raw quants the doubled
/// E2M1 values (+-12 at most), `sc` 1, `d[g]` the sub-block's UE4M3 scale
/// times 0.5 (its own f32 per 16 values).
pub(crate) fn unpack_nvfp4(b: &[u8; NVFP4_BYTES], out: &mut Unpacked) {
    out.reset(QK_NVFP4, 16);
    let (scales, qs) = b.split_at(4);
    for (g, ((&s, qs), y)) in scales
        .iter()
        .zip(qs.as_chunks::<8>().0.iter())
        .zip(out.q.as_chunks_mut::<16>().0.iter_mut())
        .enumerate()
    {
        out.d[g] = UE4M3_HALF[usize::from(s)];
        let (lo, hi) = y.split_at_mut(8);
        for ((lo, hi), &q) in lo.iter_mut().zip(hi).zip(qs) {
            *lo = KVALUES_FP4[usize::from(q & 0x0F)];
            *hi = KVALUES_FP4[usize::from(q >> 4)];
        }
    }
}
