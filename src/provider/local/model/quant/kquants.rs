//! The K-quants Q2_K, Q3_K, Q4_K, Q5_K, Q6_K and Q8_K: in-place f32 decoders
//! and integer unpackers for one 256-value super-block.
//!
//! Ported operation for operation from llama.cpp's `dequantize_row_q*_K`
//! (`ggml/src/ggml-quants.c`, commit named in `iq_tables.rs`), so every f32
//! result is the one llama.cpp and gguf-py compute. Block layouts
//! (`ggml-common.h`), in bytes:
//!
//! - Q2_K, 84: 16 scale bytes (low nibble scale, high nibble min, per 16
//!   values), 64 bytes of 2-bit quants, f16 `d`, f16 `dmin`.
//! - Q3_K, 110: 32 bytes of high bits, 64 bytes of 2-bit quants, 12 bytes of
//!   sixteen 6-bit scales, f16 `d`.
//! - Q4_K, 144: f16 `d`, f16 `dmin`, 12 bytes of eight 6-bit scales and mins,
//!   128 bytes of nibbles.
//! - Q5_K, 176: Q4_K's header, 32 bytes of high bits, 128 bytes of nibbles.
//! - Q6_K, 210: 128 bytes of low nibbles, 64 bytes of high 2-bit pairs, sixteen
//!   `i8` scales, f16 `d`.
//! - Q8_K, 292: f32 `d`, 256 `i8` quants, sixteen `i16` sums of 16 quants (not
//!   read here).

use super::block::{Unpacked, f16_at, u32_at};

/// Bytes of one Q2_K block.
pub(crate) const Q2_K_BYTES: usize = 84;
/// Bytes of one Q3_K block.
pub(crate) const Q3_K_BYTES: usize = 110;
/// Bytes of one Q4_K block.
pub(crate) const Q4_K_BYTES: usize = 144;
/// Bytes of one Q5_K block.
pub(crate) const Q5_K_BYTES: usize = 176;
/// Bytes of one Q6_K block.
pub(crate) const Q6_K_BYTES: usize = 210;
/// Bytes of one Q8_K block.
pub(crate) const Q8_K_BYTES: usize = 292;
/// Values of every K-quant block.
pub(crate) const QK_K: usize = 256;

/// The 6-bit scale and min of sub-block `j` (0 to 7) out of Q4_K's and Q5_K's
/// 12 packed bytes (`get_scale_min_k4`).
#[inline]
fn scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

/// The sixteen 6-bit scales of Q3_K, minus 32, out of its 12 packed bytes:
/// scale `i` is the low nibble of byte `i % 8` (high nibble for `i >= 8`)
/// plus bits `2 * (i / 4)` and above of byte `8 + i % 4` as bits 4 and 5.
#[inline]
fn q3_scales(sc: &[u8]) -> [i8; 16] {
    let mut out = [0i8; 16];
    for (i, o) in out.iter_mut().enumerate() {
        let low = if i < 8 { sc[i] & 0x0F } else { sc[i - 8] >> 4 };
        let high = (sc[8 + i % 4] >> (2 * (i / 4))) & 3;
        *o = (low | (high << 4)).cast_signed() - 32;
    }
    out
}

/// Decode one Q2_K block.
pub(crate) fn dequant_q2_k(b: &[u8; Q2_K_BYTES], out: &mut [f32; QK_K]) {
    let (scales, rest) = b.split_at(16);
    let (qs, rest) = rest.split_at(64);
    let d = f16_at(rest, 0);
    let min = f16_at(rest, 2);
    for (is, (sc, y)) in scales
        .iter()
        .zip(out.as_chunks_mut::<16>().0.iter_mut())
        .enumerate()
    {
        let dl = d * f32::from(sc & 0x0F);
        let ml = min * f32::from(sc >> 4);
        let (q, shift) = q2_source(qs, is);
        for (y, &q) in y.iter_mut().zip(q) {
            *y = dl * f32::from((q >> shift) & 3) - ml;
        }
    }
}

/// The 16 bytes of a Q2_K or Q3_K quant area that sub-block `is` reads, and
/// the shift of its 2-bit quants: 128 values per 32 bytes, four shifts of two
/// sub-blocks each.
#[inline]
fn q2_source(qs: &[u8], is: usize) -> (&[u8], usize) {
    let at = is / 8 * 32 + is % 2 * 16;
    (&qs[at..at + 16], is / 2 % 4 * 2)
}

/// Unpack one Q2_K block: 16 sub-blocks of 16 values, raw quants 0 to 3, `sc`
/// 0 to 15, `mn` 0 to 15, `d`, `dmin`.
pub(crate) fn unpack_q2_k(b: &[u8; Q2_K_BYTES], out: &mut Unpacked) {
    let (scales, rest) = b.split_at(16);
    let (qs, rest) = rest.split_at(64);
    out.reset(QK_K, 16);
    out.dmin = f16_at(rest, 2);
    out.d = [f16_at(rest, 0); 16];
    for (is, (&sc, y)) in scales
        .iter()
        .zip(out.q.as_chunks_mut::<16>().0.iter_mut())
        .enumerate()
    {
        out.sc[is] = i16::from(sc & 0x0F);
        out.mn[is] = i16::from(sc >> 4);
        let (q, shift) = q2_source(qs, is);
        for (y, &q) in y.iter_mut().zip(q) {
            *y = ((q >> shift) & 3).cast_signed();
        }
    }
}

/// The 16 quants of Q3_K sub-block `is`: each is its 2 low bits minus 4 where
/// bit `is / 2` of the matching `hmask` byte (bytes 0 to 15 for even `is`, 16
/// to 31 for odd) is clear.
#[inline]
fn q3_quants<'a>(qs: &'a [u8], hmask: &'a [u8], is: usize) -> impl Iterator<Item = i8> + 'a {
    let (q, shift) = q2_source(qs, is);
    let h = is % 2 * 16;
    let m = 1u8 << (is / 2);
    q.iter()
        .zip(&hmask[h..h + 16])
        .map(move |(&q, &hm)| ((q >> shift) & 3).cast_signed() - if hm & m != 0 { 0 } else { 4 })
}

/// Decode one Q3_K block.
pub(crate) fn dequant_q3_k(b: &[u8; Q3_K_BYTES], out: &mut [f32; QK_K]) {
    let (hmask, rest) = b.split_at(32);
    let (qs, rest) = rest.split_at(64);
    let (packed, rest) = rest.split_at(12);
    let d_all = f16_at(rest, 0);
    let scales = q3_scales(packed);
    for (is, (&sc, y)) in scales
        .iter()
        .zip(out.as_chunks_mut::<16>().0.iter_mut())
        .enumerate()
    {
        let dl = d_all * f32::from(sc);
        for (y, v) in y.iter_mut().zip(q3_quants(qs, hmask, is)) {
            *y = dl * f32::from(v);
        }
    }
}

/// Unpack one Q3_K block: 16 sub-blocks of 16 values, raw quants -4 to 3 (the
/// 2 low bits minus 4 where the high bit is clear), `sc` -32 to 31, `d`.
pub(crate) fn unpack_q3_k(b: &[u8; Q3_K_BYTES], out: &mut Unpacked) {
    let (hmask, rest) = b.split_at(32);
    let (qs, rest) = rest.split_at(64);
    let (packed, rest) = rest.split_at(12);
    out.reset(QK_K, 16);
    out.d = [f16_at(rest, 0); 16];
    for (is, (sc, y)) in q3_scales(packed)
        .into_iter()
        .zip(out.q.as_chunks_mut::<16>().0.iter_mut())
        .enumerate()
    {
        out.sc[is] = i16::from(sc);
        for (y, v) in y.iter_mut().zip(q3_quants(qs, hmask, is)) {
            *y = v;
        }
    }
}

/// Decode one Q4_K block.
pub(crate) fn dequant_q4_k(b: &[u8; Q4_K_BYTES], out: &mut [f32; QK_K]) {
    let d = f16_at(b, 0);
    let min = f16_at(b, 2);
    let scales = &b[4..16];
    for (j, (q, y)) in b[16..]
        .as_chunks::<32>()
        .0
        .iter()
        .zip(out.as_chunks_mut::<64>().0.iter_mut())
        .enumerate()
    {
        let (sc, m) = scale_min_k4(2 * j, scales);
        let (d1, m1) = (d * f32::from(sc), min * f32::from(m));
        let (sc, m) = scale_min_k4(2 * j + 1, scales);
        let (d2, m2) = (d * f32::from(sc), min * f32::from(m));
        let (lo, hi) = y.split_at_mut(32);
        for ((lo, hi), &q) in lo.iter_mut().zip(hi).zip(q) {
            *lo = d1 * f32::from(q & 0x0F) - m1;
            *hi = d2 * f32::from(q >> 4) - m2;
        }
    }
}

/// Unpack one Q4_K block: 8 sub-blocks of 32 values, raw quants 0 to 15, `sc`
/// 0 to 63, `mn` 0 to 63, `d`, `dmin`.
pub(crate) fn unpack_q4_k(b: &[u8; Q4_K_BYTES], out: &mut Unpacked) {
    out.reset(QK_K, 32);
    out.d = [f16_at(b, 0); 16];
    out.dmin = f16_at(b, 2);
    let scales = &b[4..16];
    for (j, (q, y)) in b[16..]
        .as_chunks::<32>()
        .0
        .iter()
        .zip(out.q.as_chunks_mut::<64>().0.iter_mut())
        .enumerate()
    {
        for k in 0..2 {
            let (sc, m) = scale_min_k4(2 * j + k, scales);
            out.sc[2 * j + k] = i16::from(sc);
            out.mn[2 * j + k] = i16::from(m);
        }
        let (lo, hi) = y.split_at_mut(32);
        for ((lo, hi), &q) in lo.iter_mut().zip(hi).zip(q) {
            *lo = (q & 0x0F).cast_signed();
            *hi = (q >> 4).cast_signed();
        }
    }
}

/// Decode one Q5_K block.
pub(crate) fn dequant_q5_k(b: &[u8; Q5_K_BYTES], out: &mut [f32; QK_K]) {
    let d = f16_at(b, 0);
    let min = f16_at(b, 2);
    let scales = &b[4..16];
    let (qh, qs) = b[16..].split_at(32);
    let (mut u1, mut u2) = (1u8, 2u8);
    for (j, (ql, y)) in qs
        .as_chunks::<32>()
        .0
        .iter()
        .zip(out.as_chunks_mut::<64>().0.iter_mut())
        .enumerate()
    {
        let (sc, m) = scale_min_k4(2 * j, scales);
        let (d1, m1) = (d * f32::from(sc), min * f32::from(m));
        let (sc, m) = scale_min_k4(2 * j + 1, scales);
        let (d2, m2) = (d * f32::from(sc), min * f32::from(m));
        let (lo, hi) = y.split_at_mut(32);
        for (((lo, hi), &ql), &qh) in lo.iter_mut().zip(hi).zip(ql).zip(qh) {
            *lo = d1 * f32::from((ql & 0x0F) + if qh & u1 != 0 { 16 } else { 0 }) - m1;
            *hi = d2 * f32::from((ql >> 4) + if qh & u2 != 0 { 16 } else { 0 }) - m2;
        }
        u1 <<= 2;
        u2 <<= 2;
    }
}

/// Unpack one Q5_K block: 8 sub-blocks of 32 values, raw quants 0 to 31, `sc`
/// 0 to 63, `mn` 0 to 63, `d`, `dmin`.
pub(crate) fn unpack_q5_k(b: &[u8; Q5_K_BYTES], out: &mut Unpacked) {
    out.reset(QK_K, 32);
    out.d = [f16_at(b, 0); 16];
    out.dmin = f16_at(b, 2);
    let scales = &b[4..16];
    let (qh, qs) = b[16..].split_at(32);
    let (mut u1, mut u2) = (1u8, 2u8);
    for (j, (ql, y)) in qs
        .as_chunks::<32>()
        .0
        .iter()
        .zip(out.q.as_chunks_mut::<64>().0.iter_mut())
        .enumerate()
    {
        for k in 0..2 {
            let (sc, m) = scale_min_k4(2 * j + k, scales);
            out.sc[2 * j + k] = i16::from(sc);
            out.mn[2 * j + k] = i16::from(m);
        }
        let (lo, hi) = y.split_at_mut(32);
        for (((lo, hi), &ql), &qh) in lo.iter_mut().zip(hi).zip(ql).zip(qh) {
            *lo = ((ql & 0x0F) + if qh & u1 != 0 { 16 } else { 0 }).cast_signed();
            *hi = ((ql >> 4) + if qh & u2 != 0 { 16 } else { 0 }).cast_signed();
        }
        u1 <<= 2;
        u2 <<= 2;
    }
}

/// The four 6-bit quants minus 32 of byte `l` of a Q6_K half block: elements
/// `l`, `l + 32`, `l + 64` and `l + 96` of the 128 values it covers.
#[inline]
fn q6_quants(ql: &[u8], qh: u8, l: usize) -> [i8; 4] {
    [
        ((ql[l] & 0x0F) | ((qh & 3) << 4)).cast_signed() - 32,
        ((ql[l + 32] & 0x0F) | (((qh >> 2) & 3) << 4)).cast_signed() - 32,
        ((ql[l] >> 4) | (((qh >> 4) & 3) << 4)).cast_signed() - 32,
        ((ql[l + 32] >> 4) | (((qh >> 6) & 3) << 4)).cast_signed() - 32,
    ]
}

/// Decode one Q6_K block.
pub(crate) fn dequant_q6_k(b: &[u8; Q6_K_BYTES], out: &mut [f32; QK_K]) {
    let (ql, rest) = b.split_at(128);
    let (qh, rest) = rest.split_at(64);
    let (sc, rest) = rest.split_at(16);
    let d = f16_at(rest, 0);
    for (((ql, qh), sc), y) in ql
        .as_chunks::<64>()
        .0
        .iter()
        .zip(qh.as_chunks::<32>().0.iter())
        .zip(sc.as_chunks::<8>().0.iter())
        .zip(out.as_chunks_mut::<128>().0.iter_mut())
    {
        for (l, &qh) in qh.iter().enumerate() {
            let is = l / 16;
            let [q1, q2, q3, q4] = q6_quants(ql, qh, l);
            y[l] = d * f32::from(sc[is].cast_signed()) * f32::from(q1);
            y[l + 32] = d * f32::from(sc[is + 2].cast_signed()) * f32::from(q2);
            y[l + 64] = d * f32::from(sc[is + 4].cast_signed()) * f32::from(q3);
            y[l + 96] = d * f32::from(sc[is + 6].cast_signed()) * f32::from(q4);
        }
    }
}

/// Unpack one Q6_K block: 16 sub-blocks of 16 values, raw quants -32 to 31,
/// `sc` -128 to 127, `d`.
pub(crate) fn unpack_q6_k(b: &[u8; Q6_K_BYTES], out: &mut Unpacked) {
    let (ql, rest) = b.split_at(128);
    let (qh, rest) = rest.split_at(64);
    let (sc, rest) = rest.split_at(16);
    out.reset(QK_K, 16);
    out.d = [f16_at(rest, 0); 16];
    for (o, &s) in out.sc.iter_mut().zip(sc) {
        *o = i16::from(s.cast_signed());
    }
    for ((ql, qh), y) in ql
        .as_chunks::<64>()
        .0
        .iter()
        .zip(qh.as_chunks::<32>().0.iter())
        .zip(out.q.as_chunks_mut::<128>().0.iter_mut())
    {
        for (l, &qh) in qh.iter().enumerate() {
            let [q1, q2, q3, q4] = q6_quants(ql, qh, l);
            y[l] = q1;
            y[l + 32] = q2;
            y[l + 64] = q3;
            y[l + 96] = q4;
        }
    }
}

/// Decode one Q8_K block.
pub(crate) fn dequant_q8_k(b: &[u8; Q8_K_BYTES], out: &mut [f32; QK_K]) {
    let d = f32::from_bits(u32_at(b, 0));
    for (y, &q) in out.iter_mut().zip(&b[4..4 + QK_K]) {
        *y = d * f32::from(q.cast_signed());
    }
}

/// Unpack one Q8_K block: a single sub-block of 256 values, raw quants -128 to
/// 127, `sc` 1, `d` (an f32 in the block).
pub(crate) fn unpack_q8_k(b: &[u8; Q8_K_BYTES], out: &mut Unpacked) {
    out.reset(QK_K, QK_K);
    out.d[0] = f32::from_bits(u32_at(b, 0));
    for (o, &q) in out.q.iter_mut().zip(&b[4..4 + QK_K]) {
        *o = q.cast_signed();
    }
}
