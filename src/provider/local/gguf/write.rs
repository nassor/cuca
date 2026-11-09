//! A small in-memory GGUF writer for test fixtures (the converter has its own
//! streaming writer, `convert::gguf_writer`). Lays a file out like candle's
//! writer did: version 2, 32-byte alignment.

use std::io::{Seek, Write};

use candle_core::Result;
use candle_core::quantized::QTensor;

use super::{GgufType, Value, write_string};

/// A tensor whose bytes are already in the file's type.
pub(crate) struct RawTensor<'a> {
    pub name: &'a str,
    /// Dimensions, slowest first.
    pub dims: &'a [usize],
    pub dtype: GgufType,
    pub data: &'a [u8],
}

/// Write `metadata` and `tensors` (candle tensors) as a GGUF file.
pub(crate) fn write<W: Seek + Write>(
    w: &mut W,
    metadata: &[(&str, &Value)],
    tensors: &[(&str, &QTensor)],
) -> Result<()> {
    let data: Vec<_> = tensors
        .iter()
        .map(|(_, t)| t.data().map(|d| d.into_owned()))
        .collect::<Result<_>>()?;
    let raw: Vec<RawTensor> = tensors
        .iter()
        .zip(&data)
        .map(|((name, t), data)| RawTensor {
            name,
            dims: t.shape().dims(),
            dtype: GgufType::Ggml(t.dtype()),
            data,
        })
        .collect();
    write_raw(w, metadata, &raw)
}

/// Write `metadata` and `tensors` whose bytes are given.
pub(crate) fn write_raw<W: Seek + Write>(
    w: &mut W,
    metadata: &[(&str, &Value)],
    tensors: &[RawTensor],
) -> Result<()> {
    w.write_all(b"GGUF")?;
    w.write_all(&2u32.to_le_bytes())?;
    w.write_all(&(tensors.len() as u64).to_le_bytes())?;
    w.write_all(&(metadata.len() as u64).to_le_bytes())?;
    for (name, value) in metadata {
        write_string(w, name)?;
        w.write_all(&value.value_type().id().to_le_bytes())?;
        value.write(w)?;
    }
    let mut offset = 0usize;
    for t in tensors {
        write_string(w, t.name)?;
        w.write_all(&(t.dims.len() as u32).to_le_bytes())?;
        for &d in t.dims.iter().rev() {
            w.write_all(&(d as u64).to_le_bytes())?;
        }
        w.write_all(&t.dtype.id().to_le_bytes())?;
        w.write_all(&(offset as u64).to_le_bytes())?;
        offset += t.data.len() + (31 - (31 + t.data.len()) % 32);
    }
    let pos = w.stream_position()? as usize;
    w.write_all(&vec![0u8; 31 - (31 + pos) % 32])?;
    for t in tensors {
        w.write_all(t.data)?;
        w.write_all(&vec![0u8; 31 - (31 + t.data.len()) % 32])?;
    }
    Ok(())
}
