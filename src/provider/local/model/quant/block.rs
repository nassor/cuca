//! The integer view of one quantized block: [`Unpacked`].
//!
//! The f32 decoders of `kquants.rs` and `iquants.rs` reproduce llama.cpp's
//! `dequantize_row_*` bit for bit. A row-batched integer kernel wants the same
//! block as small integers instead: it multiplies `i16` weight quants against
//! `i8` activation quants and applies the f32 scales once per sub-block. Every
//! block type has an `unpack_*` function that fills an [`Unpacked`], and
//! [`Unpacked::fold_into`] folds the integer sub-block scales into `i16` quants.
//!
//! # Contract
//!
//! With `g = i / group`, element `i` of a block is, as a real number,
//!
//! ```text
//! d[g] * (sc[g] * q[i]) - dmin * mn[g]
//! ```
//!
//! `q` are the raw integer quants (signed, grid and sign tables already
//! applied), `sc` the integer sub-block scale (1 where the type has none),
//! `mn` the integer sub-block min (0 where the type has none), `d[g]` the f32
//! factor of the product `sc[g] * q[i]` and `dmin` the f32 factor of `mn[g]`.
//! The block's own power-of-two constants are inside `d` (IQ2 types: `d / 8`,
//! IQ3_XXS: `d / 4`, IQ1 types: `d / 8` with the delta folded into `q` as
//! `8 * grid +- 1`), so no further constant applies. Every product the f32
//! decoder rounds is exact in f32 except the last, so evaluating the formula
//! with `fold_into` and one f32 multiply gives the decoder's value (a zero may
//! differ in sign).
//!
//! [`Unpacked::fold_into`] returns `sc[g] * q[i]` as `i16`. The largest
//! magnitude over every possible block, which a kernel can rely on when it
//! accumulates `i16 * i8` products in `i32` (a 256-value block sums to at most
//! `256 * 128 * bound`, below 2^31 for every row of the table):
//!
//! | type    | values | group | raw `q`                  | `sc`           | `d[g]`            | min term           | max abs folded |
//! | ------- | ------ | ----- | ------------------------ | -------------- | ----------------- | ------------------ | -------------- |
//! | Q2_K    | 256    | 16    | 0 to 3                   | 0 to 15        | `d`               | `dmin * mn`, 0..15 | 45             |
//! | Q3_K    | 256    | 16    | -4 to 3                  | -32 to 31      | `d`               | none               | 128            |
//! | Q4_K    | 256    | 32    | 0 to 15                  | 0 to 63        | `d`               | `dmin * mn`, 0..63 | 945            |
//! | Q5_K    | 256    | 32    | 0 to 31                  | 0 to 63        | `d`               | `dmin * mn`, 0..63 | 1953           |
//! | Q6_K    | 256    | 16    | -32 to 31                | -128 to 127    | `d`               | none               | 4096           |
//! | Q8_K    | 256    | 256   | -128 to 127              | 1              | `d` (f32)         | none               | 128            |
//! | IQ2_XXS | 256    | 32    | +-8, +-25, +-43          | odd, 1 to 31   | `d / 8`           | none               | 1333           |
//! | IQ2_XS  | 256    | 16    | +-8, +-25, +-43          | odd, 1 to 31   | `d / 8`           | none               | 1333           |
//! | IQ2_S   | 256    | 16    | +-8, +-25, +-43          | odd, 1 to 31   | `d / 8`           | none               | 1333           |
//! | IQ3_XXS | 256    | 32    | +-4 to +-62 (table)      | odd, 1 to 31   | `d / 4`           | none               | 1922           |
//! | IQ3_S   | 256    | 32    | +-1 to +-15 (odd)        | odd, 1 to 31   | `d`               | none               | 465            |
//! | IQ1_S   | 256    | 32    | +-1, +-7, +-9            | odd, 1 to 15   | `d / 8`           | none               | 135            |
//! | IQ1_M   | 256    | 16    | +-1, +-7, +-9            | odd, 1 to 15   | `d / 8`           | none               | 135            |
//! | IQ4_XS  | 256    | 32    | IQ4_NL table, +-127 max  | -32 to 31      | `d`               | none               | 4064           |
//! | IQ4_NL  | 32     | 32    | IQ4_NL table, +-127 max  | 1              | `d`               | none               | 127            |
//! | TQ1_0   | 256    | 256   | -1 to 1                  | 1              | `d`               | none               | 1              |
//! | TQ2_0   | 256    | 256   | -1 to 2                  | 1              | `d`               | none               | 2              |
//! | MXFP4   | 32     | 32    | doubled E2M1, +-12 max   | 1              | E8M0 half scale   | none               | 12             |
//! | NVFP4   | 64     | 16    | doubled E2M1, +-12 max   | 1              | UE4M3 half scale  | none               | 12             |
//!
//! The `min term` of Q2_K, Q4_K and Q5_K is subtracted from the group's dot
//! product as `dmin * mn[g] * sum(x)` over the group's activation values.
//! `tests` sweeps every field range and asserts the table.

/// The most values in one block of any type (a K-quant or I-quant super-block).
pub(crate) const MAX_BLOCK: usize = 256;
/// The most sub-blocks in one block (Q2_K, Q3_K, Q6_K, IQ2_XS, IQ2_S and IQ1_M
/// have 16 of 16 values).
pub(crate) const MAX_GROUPS: usize = 16;

/// One block as integers; see the [module documentation](self) for the
/// meaning of every field. Only the first `len` values of `q` and the first
/// `len / group` entries of `sc`, `mn` and `d` are meaningful.
#[derive(Debug, Clone)]
pub(crate) struct Unpacked {
    /// Values in the block: 256, except 64 for NVFP4 and 32 for IQ4_NL and
    /// MXFP4.
    pub len: usize,
    /// Values per sub-block, a divisor of `len`.
    pub group: usize,
    /// Raw integer quants.
    pub q: [i8; MAX_BLOCK],
    /// Integer scale of each sub-block.
    pub sc: [i16; MAX_GROUPS],
    /// Integer min of each sub-block.
    pub mn: [i16; MAX_GROUPS],
    /// f32 factor of `sc[g] * q[i]` in each sub-block.
    pub d: [f32; MAX_GROUPS],
    /// f32 factor of every `mn[g]`.
    pub dmin: f32,
}

impl Default for Unpacked {
    fn default() -> Self {
        Self {
            len: 0,
            group: 1,
            q: [0; MAX_BLOCK],
            sc: [1; MAX_GROUPS],
            mn: [0; MAX_GROUPS],
            d: [0.0; MAX_GROUPS],
            dmin: 0.0,
        }
    }
}

// `fold_into`, `reconstruct` and the f32 accessors are what the kernel phase
// calls; until then only the tests do.
#[allow(dead_code)]
impl Unpacked {
    /// Reset to a block of `len` values in sub-blocks of `group` values: unit
    /// scales, no min. The quants are left for the caller to overwrite.
    pub(crate) fn reset(&mut self, len: usize, group: usize) {
        self.len = len;
        self.group = group;
        self.sc = [1; MAX_GROUPS];
        self.mn = [0; MAX_GROUPS];
        self.d = [0.0; MAX_GROUPS];
        self.dmin = 0.0;
    }

    /// Sub-blocks in the block.
    pub(crate) fn groups(&self) -> usize {
        self.len / self.group.max(1)
    }

    /// `sc[g] * q[i]` for every value, the integer sub-block scales folded into
    /// the quants. The first `len` entries of `out` are written; the largest
    /// magnitude is the table's last column.
    pub(crate) fn fold_into(&self, out: &mut [i16; MAX_BLOCK]) {
        let len = self.len.min(MAX_BLOCK);
        let group = self.group.max(1);
        for ((&sc, q), o) in self
            .sc
            .iter()
            .zip(self.q[..len].chunks_exact(group))
            .zip(out[..len].chunks_exact_mut(group))
        {
            for (o, &q) in o.iter_mut().zip(q) {
                *o = sc * i16::from(q);
            }
        }
    }

    /// The f32 factor `d[g] * sc[g]` that, times the raw quant, is the value
    /// without its min.
    pub(crate) fn scale_f32(&self, g: usize) -> f32 {
        self.d[g] * f32::from(self.sc[g])
    }

    /// The f32 min `dmin * mn[g]` of sub-block `g`.
    pub(crate) fn min_f32(&self, g: usize) -> f32 {
        self.dmin * f32::from(self.mn[g])
    }

    /// The block in f32 from the folded form, `d[g] * folded - dmin * mn[g]`:
    /// llama.cpp's value, except for the sign of a zero.
    pub(crate) fn reconstruct(&self, out: &mut [f32; MAX_BLOCK]) {
        let mut folded = [0i16; MAX_BLOCK];
        self.fold_into(&mut folded);
        let len = self.len.min(MAX_BLOCK);
        let group = self.group.max(1);
        for (g, (f, o)) in folded[..len]
            .chunks_exact(group)
            .zip(out[..len].chunks_exact_mut(group))
            .enumerate()
        {
            let (d, min) = (self.d[g], self.min_f32(g));
            for (o, &f) in o.iter_mut().zip(f) {
                *o = d * f32::from(f) - min;
            }
        }
    }
}

/// The f16 at byte `at` of a block, widened exactly.
#[inline]
pub(crate) fn f16_at(b: &[u8], at: usize) -> f32 {
    half::f16::from_bits(u16::from_le_bytes([b[at], b[at + 1]])).to_f32()
}

/// The little-endian `u16` at byte `at` of a block.
#[inline]
pub(crate) fn u16_at(b: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([b[at], b[at + 1]])
}

/// The little-endian `u32` at byte `at` of a block.
#[inline]
pub(crate) fn u32_at(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]])
}
