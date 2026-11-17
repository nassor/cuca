//! The little-endian frame codec for K/V transfer and layer-stage calls
//! between cuca nodes.
//!
//! Every integer and float is little-endian. A slice is a `u64` element count
//! (elements, not bytes) followed by its elements: 4-byte values for `u32`
//! and `f32` slices. A [`Reader`] borrows the frame and allocates only the
//! vectors it returns; a short read or a count whose byte size overflows is
//! an error, never a panic or an oversized allocation.

use super::CarryDtype;
use super::model::err as local_err;
use crate::error::CucaError;

/// The version of the frame layout, bumped on any incompatible change. It
/// goes into the model fingerprint a stage worker checks at `open` and a K/V
/// header carries, so nodes of different layouts refuse each other at once.
/// 2: layer-stage calls carry a call number, and a forward call carries the
/// restore, snapshot and Engram prefetch that were calls of their own.
/// 3: `open` names the number type of the carry, and every carry starts with
/// the tag of the type it is written in.
pub(crate) const WIRE_VERSION: u32 = 3;

impl CarryDtype {
    /// The byte a frame names the type with.
    pub(crate) fn tag(self) -> u8 {
        match self {
            Self::F32 => 0,
            Self::Bf16 => 1,
            Self::F16 => 2,
        }
    }

    /// The type a frame's tag names.
    ///
    /// # Errors
    ///
    /// [`CucaError::Provider`] for a tag no type has.
    pub(crate) fn from_tag(tag: u8) -> Result<Self, CucaError> {
        match tag {
            0 => Ok(Self::F32),
            1 => Ok(Self::Bf16),
            2 => Ok(Self::F16),
            other => Err(local_err(format!(
                "unknown carry type tag {other} in frame"
            ))),
        }
    }

    /// The name the configuration gives it.
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::Bf16 => "bf16",
            Self::F16 => "f16",
        }
    }
}

/// Appends values to a frame.
pub(crate) struct Writer {
    buf: Vec<u8>,
}

impl Writer {
    pub(crate) fn new() -> Self {
        Self { buf: Vec::new() }
    }

    pub(crate) fn with_capacity(n: usize) -> Self {
        Self {
            buf: Vec::with_capacity(n),
        }
    }

    pub(crate) fn put_u8(&mut self, v: u8) {
        self.buf.push(v);
    }

    pub(crate) fn put_u32(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    pub(crate) fn put_u64(&mut self, v: u64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    pub(crate) fn put_u32s(&mut self, v: &[u32]) {
        self.put_u64(v.len() as u64);
        self.buf.reserve(v.len() * 4);
        for x in v {
            self.buf.extend_from_slice(&x.to_le_bytes());
        }
    }

    pub(crate) fn put_f32s(&mut self, v: &[f32]) {
        self.put_u64(v.len() as u64);
        self.buf.reserve(v.len() * 4);
        for x in v {
            self.buf.extend_from_slice(&x.to_le_bytes());
        }
    }

    /// A slice of floats as `dtype`: the count, then 4-byte values for `F32`
    /// (bit-exact) or 2-byte values rounded to nearest even for `Bf16` and
    /// `F16`.
    ///
    /// # Errors
    ///
    /// [`CucaError::Provider`] for a finite value an `F16` cannot hold
    /// (beyond 65504): it would travel as infinity.
    pub(crate) fn put_floats(&mut self, v: &[f32], dtype: CarryDtype) -> Result<(), CucaError> {
        match dtype {
            CarryDtype::F32 => self.put_f32s(v),
            CarryDtype::Bf16 => {
                self.put_u64(v.len() as u64);
                self.buf.reserve(v.len() * 2);
                for x in v {
                    self.buf
                        .extend_from_slice(&half::bf16::from_f32(*x).to_le_bytes());
                }
            }
            CarryDtype::F16 => {
                self.put_u64(v.len() as u64);
                self.buf.reserve(v.len() * 2);
                for x in v {
                    let h = half::f16::from_f32(*x);
                    if x.is_finite() && !h.is_finite() {
                        return Err(local_err(format!(
                            "the value {x} does not fit an f16 carry (largest 65504): \
                             use `carry=\"bf16\"` or `\"f32\"`"
                        )));
                    }
                    self.buf.extend_from_slice(&h.to_le_bytes());
                }
            }
        }
        Ok(())
    }

    pub(crate) fn into_bytes(self) -> Vec<u8> {
        self.buf
    }
}

/// Reads values from a frame.
pub(crate) struct Reader<'a> {
    buf: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    pub(crate) fn new(buf: &'a [u8]) -> Self {
        Self { buf, at: 0 }
    }

    /// The bytes not read yet.
    pub(crate) fn remaining(&self) -> usize {
        self.buf.len() - self.at
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.remaining() == 0
    }

    fn truncated(&self, want: usize) -> CucaError {
        local_err(format!(
            "truncated frame: {want} bytes wanted at offset {} of {}",
            self.at,
            self.buf.len()
        ))
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], CucaError> {
        if n > self.remaining() {
            return Err(self.truncated(n));
        }
        let out = &self.buf[self.at..self.at + n];
        self.at += n;
        Ok(out)
    }

    fn take_array<const N: usize>(&mut self) -> Result<[u8; N], CucaError> {
        let mut out = [0u8; N];
        out.copy_from_slice(self.take(N)?);
        Ok(out)
    }

    /// A count of `size`-byte elements and the bytes they span.
    fn take_counted(&mut self, size: usize) -> Result<&'a [u8], CucaError> {
        let count = self.get_u64()?;
        let bytes = usize::try_from(count)
            .ok()
            .and_then(|n| n.checked_mul(size))
            .ok_or_else(|| self.truncated(usize::MAX))?;
        self.take(bytes)
    }

    pub(crate) fn get_u8(&mut self) -> Result<u8, CucaError> {
        Ok(self.take_array::<1>()?[0])
    }

    pub(crate) fn get_u32(&mut self) -> Result<u32, CucaError> {
        Ok(u32::from_le_bytes(self.take_array()?))
    }

    pub(crate) fn get_u64(&mut self) -> Result<u64, CucaError> {
        Ok(u64::from_le_bytes(self.take_array()?))
    }

    pub(crate) fn get_u32s(&mut self) -> Result<Vec<u32>, CucaError> {
        Ok(self
            .take_counted(4)?
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_le_bytes(*c))
            .collect())
    }

    pub(crate) fn get_f32s(&mut self) -> Result<Vec<f32>, CucaError> {
        Ok(self
            .take_counted(4)?
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect())
    }

    /// A slice of floats written as `dtype` ([`Writer::put_floats`]), widened
    /// to `f32`.
    pub(crate) fn get_floats(&mut self, dtype: CarryDtype) -> Result<Vec<f32>, CucaError> {
        match dtype {
            CarryDtype::F32 => self.get_f32s(),
            CarryDtype::Bf16 => Ok(self
                .take_counted(2)?
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| half::bf16::from_le_bytes(*c).to_f32())
                .collect()),
            CarryDtype::F16 => Ok(self
                .take_counted(2)?
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| half::f16::from_le_bytes(*c).to_f32())
                .collect()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_type_roundtrips() {
        let mut w = Writer::with_capacity(64);
        w.put_u8(7);
        w.put_u32(0xDEAD_BEEF);
        w.put_u64(u64::MAX - 1);
        w.put_u32s(&[1, 2, u32::MAX]);
        w.put_u32s(&[]);
        w.put_f32s(&[1.5, -0.0, f32::INFINITY]);
        w.put_f32s(&[]);
        let bytes = w.into_bytes();

        let mut r = Reader::new(&bytes);
        assert_eq!(r.get_u8().unwrap(), 7);
        assert_eq!(r.get_u32().unwrap(), 0xDEAD_BEEF);
        assert_eq!(r.get_u64().unwrap(), u64::MAX - 1);
        assert_eq!(r.get_u32s().unwrap(), vec![1, 2, u32::MAX]);
        assert!(r.get_u32s().unwrap().is_empty());
        let f = r.get_f32s().unwrap();
        assert_eq!(f[0], 1.5);
        assert!(f[1].is_sign_negative() && f[1] == 0.0);
        assert_eq!(f[2], f32::INFINITY);
        assert!(r.get_f32s().unwrap().is_empty());
        assert!(r.is_empty());
        assert_eq!(r.remaining(), 0);
    }

    #[test]
    fn integers_are_little_endian() {
        let mut w = Writer::new();
        w.put_u32(1);
        w.put_u64(2);
        assert_eq!(w.into_bytes(), [1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn a_short_read_reports_the_offset() {
        let mut r = Reader::new(&[1, 2, 3, 4, 5]);
        r.get_u32().unwrap();
        let e = r.get_u32().unwrap_err().to_string();
        assert!(
            e.contains("truncated frame: 4 bytes wanted at offset 4 of 5"),
            "{e}"
        );
    }

    #[test]
    fn a_huge_count_fails_without_allocating() {
        for count in [u64::MAX, u64::MAX / 4 + 1, 1 << 40] {
            let mut w = Writer::new();
            w.put_u64(count);
            let bytes = w.into_bytes();
            assert!(
                Reader::new(&bytes)
                    .get_u32s()
                    .unwrap_err()
                    .to_string()
                    .contains("truncated frame"),
            );
            assert!(Reader::new(&bytes).get_f32s().is_err());
        }
        for dtype in [CarryDtype::Bf16, CarryDtype::F16] {
            for count in [u64::MAX / 2 + 1, 1 << 40] {
                let mut w = Writer::new();
                w.put_u64(count);
                assert!(
                    Reader::new(&w.into_bytes()).get_floats(dtype).is_err(),
                    "{dtype:?} {count}"
                );
            }
        }
    }

    fn floats_through(values: &[f32], dtype: CarryDtype) -> (Vec<f32>, usize) {
        let mut w = Writer::new();
        w.put_floats(values, dtype).unwrap();
        let bytes = w.into_bytes();
        let n = bytes.len();
        let mut r = Reader::new(&bytes);
        let got = r.get_floats(dtype).unwrap();
        assert!(r.is_empty());
        (got, n)
    }

    #[test]
    fn f32_floats_are_exact_and_the_others_are_half_the_bytes_and_bounded() {
        let values: Vec<f32> = (0..2000)
            .map(|i| ((i as f32) * 0.37).sin() * 40.0 + 0.001 * i as f32)
            .collect();
        let (exact, f32_bytes) = floats_through(&values, CarryDtype::F32);
        assert!(
            exact
                .iter()
                .zip(&values)
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );
        for (dtype, bits) in [(CarryDtype::Bf16, 8), (CarryDtype::F16, 11)] {
            let (got, bytes) = floats_through(&values, dtype);
            assert_eq!(bytes - 8, (f32_bytes - 8) / 2, "{dtype:?}: two bytes each");
            for (g, v) in got.iter().zip(&values) {
                // Rounding to `bits` significant bits: relative error at
                // most 2^-bits (half an ulp), plus the f16 subnormal step
                // (2^-25) near zero.
                let bound = v.abs() * 2f32.powi(-bits) + 3.0e-8;
                assert!((g - v).abs() <= bound, "{dtype:?}: {v} came back {g}");
            }
        }
    }

    #[test]
    fn non_finite_values_survive_and_an_f16_overflow_is_an_error() {
        for dtype in [CarryDtype::Bf16, CarryDtype::F16] {
            let (got, _) = floats_through(&[f32::INFINITY, f32::NEG_INFINITY, -0.0], dtype);
            assert_eq!(got[0], f32::INFINITY);
            assert_eq!(got[1], f32::NEG_INFINITY);
            assert!(got[2] == 0.0 && got[2].is_sign_negative());
            let (got, _) = floats_through(&[f32::NAN], dtype);
            assert!(got[0].is_nan());
        }
        // 65504 is the largest f16; bf16 reaches the range of f32.
        assert_eq!(floats_through(&[65504.0], CarryDtype::F16).0, [65504.0]);
        assert!(floats_through(&[3.0e38], CarryDtype::Bf16).0[0].is_finite());
        let mut w = Writer::new();
        let e = w
            .put_floats(&[1.0, 70000.0], CarryDtype::F16)
            .unwrap_err()
            .to_string();
        assert!(e.contains("70000") && e.contains("f16"), "{e}");
    }

    #[test]
    fn a_carry_tag_names_its_type_and_no_other() {
        for dtype in [CarryDtype::F32, CarryDtype::Bf16, CarryDtype::F16] {
            assert_eq!(CarryDtype::from_tag(dtype.tag()).unwrap(), dtype);
        }
        assert!(CarryDtype::from_tag(3).is_err());
    }
}
