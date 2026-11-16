//! The little-endian frame codec for K/V transfer and layer-stage calls
//! between cuca nodes.
//!
//! Every integer and float is little-endian. A slice is a `u64` element count
//! (elements, not bytes) followed by its elements: 4-byte values for `u32`
//! and `f32` slices. A [`Reader`] borrows the frame and allocates only the
//! vectors it returns; a short read or a count whose byte size overflows is
//! an error, never a panic or an oversized allocation.

use super::model::err as local_err;
use crate::error::CucaError;

/// The version of the frame layout, bumped on any incompatible change.
pub(crate) const WIRE_VERSION: u32 = 1;

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
    }
}
