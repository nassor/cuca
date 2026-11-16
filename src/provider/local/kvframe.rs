//! The frames of a K/V transfer between two nodes, on top of the codec of
//! [`wire`](super::wire).
//!
//! A frame is a `u8` kind, a `u64` payload length and the payload. A transfer
//! is, in order: one header, rows frames of at most `chunk` tokens, the
//! recurrent-state snapshot (a hybrid model only) and an end frame.
//!
//! - header (1): `WIRE_VERSION u32`, `fingerprint u64`, `from u64`, `to u64`,
//!   `chunk u64`, `snapshot u8`;
//! - rows (2): `a u64`, `b u64` (the positions `a..b`), `count u32`, then per
//!   entry `layer u32`, `compressed u8`, `k` and `v` as `f32` slices (see
//!   [`transfer`](super::model::transfer) for the entries and their layout);
//! - snapshot (3): `count u32`, then per delta-net layer `conv` and `s` as
//!   `f32` slices;
//! - end (4): empty.

use super::model::err;
use super::model::transfer::{DeltaRows, PoolRows};
use super::wire::{Reader, Writer};
use crate::error::CucaError;

pub(crate) const HEADER: u8 = 1;
pub(crate) const ROWS: u8 = 2;
pub(crate) const SNAPSHOT: u8 = 3;
pub(crate) const END: u8 = 4;

/// The bytes before a frame's payload: the kind and the payload length.
pub(crate) const ENVELOPE: usize = 1 + 8;

/// The first frame of a transfer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Header {
    pub(crate) version: u32,
    pub(crate) fingerprint: u64,
    pub(crate) from: usize,
    pub(crate) to: usize,
    pub(crate) chunk: usize,
    /// A recurrent-state snapshot frame follows the rows.
    pub(crate) snapshot: bool,
}

/// `payload` in a frame of `kind`.
fn frame(kind: u8, payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(ENVELOPE + payload.len());
    out.push(kind);
    out.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    out.extend_from_slice(payload);
    out
}

/// The kind and payload of a whole frame.
///
/// # Errors
///
/// [`CucaError::Provider`] for a frame shorter than its envelope or whose
/// length field is not the payload's.
pub(crate) fn parse(frame: &[u8]) -> Result<(u8, &[u8]), CucaError> {
    let (Some(&kind), Some(len)) = (frame.first(), frame.get(1..ENVELOPE)) else {
        return Err(err("a K/V frame is shorter than its header"));
    };
    let len = u64::from_le_bytes(
        len.try_into()
            .map_err(|_| err("a K/V frame is truncated"))?,
    );
    let payload = &frame[ENVELOPE..];
    if payload.len() as u64 != len {
        return Err(err(format!(
            "a K/V frame announces {len} payload bytes and holds {}",
            payload.len()
        )));
    }
    Ok((kind, payload))
}

pub(crate) fn header_frame(h: &Header) -> Vec<u8> {
    let mut w = Writer::new();
    w.put_u32(h.version);
    w.put_u64(h.fingerprint);
    w.put_u64(h.from as u64);
    w.put_u64(h.to as u64);
    w.put_u64(h.chunk as u64);
    w.put_u8(u8::from(h.snapshot));
    frame(HEADER, &w.into_bytes())
}

pub(crate) fn decode_header(payload: &[u8]) -> Result<Header, CucaError> {
    let mut r = Reader::new(payload);
    let version = r.get_u32()?;
    let fingerprint = r.get_u64()?;
    let (from, to, chunk) = (r.get_u64()?, r.get_u64()?, r.get_u64()?);
    let snapshot = r.get_u8()? != 0;
    let size = |v: u64| usize::try_from(v).map_err(|_| err("a K/V header position is too large"));
    Ok(Header {
        version,
        fingerprint,
        from: size(from)?,
        to: size(to)?,
        chunk: size(chunk)?,
        snapshot,
    })
}

pub(crate) fn rows_frame(a: usize, b: usize, rows: &[PoolRows]) -> Vec<u8> {
    let floats: usize = rows.iter().map(|r| r.k.len() + r.v.len()).sum();
    let mut w = Writer::with_capacity(32 + rows.len() * 32 + 4 * floats);
    w.put_u64(a as u64);
    w.put_u64(b as u64);
    w.put_u32(u32::try_from(rows.len()).unwrap_or(u32::MAX));
    for row in rows {
        w.put_u32(u32::try_from(row.layer).unwrap_or(u32::MAX));
        w.put_u8(u8::from(row.compressed));
        w.put_f32s(&row.k);
        w.put_f32s(&row.v);
    }
    frame(ROWS, &w.into_bytes())
}

/// The positions and entries of a rows frame's payload.
pub(crate) fn decode_rows(payload: &[u8]) -> Result<(usize, usize, Vec<PoolRows>), CucaError> {
    let mut r = Reader::new(payload);
    let size = |v: u64| usize::try_from(v).map_err(|_| err("a K/V row position is too large"));
    let (a, b) = (size(r.get_u64()?)?, size(r.get_u64()?)?);
    let count = r.get_u32()?;
    let mut rows = Vec::new();
    for _ in 0..count {
        let layer = r.get_u32()? as usize;
        let compressed = r.get_u8()? != 0;
        let k = r.get_f32s()?;
        let v = r.get_f32s()?;
        rows.push(PoolRows {
            layer,
            compressed,
            k,
            v,
        });
    }
    if !r.is_empty() {
        return Err(err("a K/V rows frame has bytes after its last entry"));
    }
    Ok((a, b, rows))
}

pub(crate) fn snapshot_frame(layers: &[DeltaRows]) -> Vec<u8> {
    let floats: usize = layers.iter().map(|l| l.conv.len() + l.s.len()).sum();
    let mut w = Writer::with_capacity(16 + layers.len() * 16 + 4 * floats);
    w.put_u32(u32::try_from(layers.len()).unwrap_or(u32::MAX));
    for layer in layers {
        w.put_f32s(&layer.conv);
        w.put_f32s(&layer.s);
    }
    frame(SNAPSHOT, &w.into_bytes())
}

pub(crate) fn decode_snapshot(payload: &[u8]) -> Result<Vec<DeltaRows>, CucaError> {
    let mut r = Reader::new(payload);
    let count = r.get_u32()?;
    let mut layers = Vec::new();
    for _ in 0..count {
        let conv = r.get_f32s()?;
        let s = r.get_f32s()?;
        layers.push(DeltaRows { conv, s });
    }
    if !r.is_empty() {
        return Err(err("a K/V snapshot frame has bytes after its last layer"));
    }
    Ok(layers)
}

pub(crate) fn end_frame() -> Vec<u8> {
    frame(END, &[])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_frame_roundtrips() {
        let header = Header {
            version: 1,
            fingerprint: 0xfeed_beef,
            from: 32,
            to: 96,
            chunk: 256,
            snapshot: true,
        };
        let bytes = header_frame(&header);
        let (kind, payload) = parse(&bytes).unwrap();
        assert_eq!(kind, HEADER);
        assert_eq!(decode_header(payload).unwrap(), header);

        let rows = vec![
            PoolRows {
                layer: 3,
                compressed: false,
                k: vec![1.0, 2.0, 3.0],
                v: vec![4.0, 5.0, 6.0],
            },
            PoolRows {
                layer: 4,
                compressed: true,
                k: vec![7.0],
                v: Vec::new(),
            },
        ];
        let bytes = rows_frame(32, 48, &rows);
        let (kind, payload) = parse(&bytes).unwrap();
        assert_eq!(kind, ROWS);
        assert_eq!(decode_rows(payload).unwrap(), (32, 48, rows));

        let layers = vec![DeltaRows {
            conv: vec![1.0, 2.0],
            s: vec![3.0],
        }];
        let bytes = snapshot_frame(&layers);
        let (kind, payload) = parse(&bytes).unwrap();
        assert_eq!(kind, SNAPSHOT);
        assert_eq!(decode_snapshot(payload).unwrap(), layers);

        let bytes = end_frame();
        assert_eq!(parse(&bytes).unwrap(), (END, &[][..]));
    }

    #[test]
    fn a_malformed_frame_is_an_error() {
        assert!(parse(&[]).is_err());
        assert!(parse(&[2, 1, 0]).is_err());
        // A length field that does not match the payload.
        let mut bytes = end_frame();
        bytes.push(7);
        assert!(parse(&bytes).is_err());
        // A rows frame cut short, or with bytes left over.
        let good = rows_frame(0, 16, &[]);
        let (_, payload) = parse(&good).unwrap();
        assert!(decode_rows(&payload[..payload.len() - 1]).is_err());
        let mut long = payload.to_vec();
        long.push(0);
        assert!(decode_rows(&long).is_err());
    }
}
