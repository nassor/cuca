//! A streaming GGUF v3 writer.
//!
//! The converter knows every tensor's name, shape and storage type before it
//! reads any weight (from the safetensors headers and the name mapping), so
//! the whole header - magic, counts, the key/value section and the tensor
//! infos - is written first by [`GgufWriter::start`], and the tensor data
//! follows one tensor at a time through [`GgufWriter::write`]: peak memory is
//! one chunk of one tensor, never the model. Data is aligned to
//! `general.alignment` (32) between tensors and before the first one, exactly
//! as `candle_core::quantized::gguf_file::write` lays a file out, so
//! `gguf_file::Content::read` loads either.

use std::collections::VecDeque;
use std::io::Write;

use candle_core::quantized::gguf_file::Value;

use crate::error::CucaError;

/// `GGUF` in little-endian ASCII.
const MAGIC: u32 = 0x4655_4747;
/// The format version written.
const VERSION: u32 = 3;
/// Tensor data alignment, the value of `general.alignment`.
pub(crate) const ALIGNMENT: u64 = 32;
/// Dimensions a tensor may have in GGUF.
const MAX_DIMS: usize = 4;

/// How a tensor's elements are stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Dtype {
    F32,
    F16,
    Q4_0,
    Q8_0,
}

impl Dtype {
    /// The GGML type id stored in the tensor info.
    fn ggml_id(self) -> u32 {
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
            Self::Q4_0 => 2,
            Self::Q8_0 => 8,
        }
    }

    /// Elements per storage block.
    pub(crate) fn block_size(self) -> usize {
        match self {
            Self::F32 | Self::F16 => 1,
            Self::Q4_0 | Self::Q8_0 => 32,
        }
    }

    /// Bytes per storage block.
    fn type_size(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
            Self::Q4_0 => 18,
            Self::Q8_0 => 34,
        }
    }

    /// Bytes of a row of `cols` elements.
    pub(crate) fn row_bytes(self, cols: usize) -> usize {
        cols / self.block_size() * self.type_size()
    }
}

/// One tensor to be written: its name, its shape in row-major order (GGUF
/// stores the dimensions reversed) and its storage type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TensorSpec {
    pub(crate) name: String,
    pub(crate) dims: Vec<usize>,
    pub(crate) dtype: Dtype,
}

impl TensorSpec {
    /// Bytes the tensor's data occupies.
    pub(crate) fn byte_len(&self) -> usize {
        let cols = self.dims.last().copied().unwrap_or(1);
        let rows: usize = self.dims.iter().rev().skip(1).product();
        rows * self.dtype.row_bytes(cols)
    }
}

fn io(e: std::io::Error) -> CucaError {
    CucaError::from(e)
}

fn write_str<W: Write>(w: &mut W, s: &str) -> Result<(), CucaError> {
    w.write_all(&(s.len() as u64).to_le_bytes()).map_err(io)?;
    w.write_all(s.as_bytes()).map_err(io)
}

/// GGUF value type ids.
fn value_type(v: &Value) -> u32 {
    match v {
        Value::U8(_) => 0,
        Value::I8(_) => 1,
        Value::U16(_) => 2,
        Value::I16(_) => 3,
        Value::U32(_) => 4,
        Value::I32(_) => 5,
        Value::F32(_) => 6,
        Value::Bool(_) => 7,
        Value::String(_) => 8,
        Value::Array(_) => 9,
        Value::U64(_) => 10,
        Value::I64(_) => 11,
        Value::F64(_) => 12,
    }
}

fn write_value<W: Write>(w: &mut W, v: &Value) -> Result<(), CucaError> {
    match v {
        Value::U8(x) => w.write_all(&[*x]),
        Value::I8(x) => w.write_all(&x.to_le_bytes()),
        Value::U16(x) => w.write_all(&x.to_le_bytes()),
        Value::I16(x) => w.write_all(&x.to_le_bytes()),
        Value::U32(x) => w.write_all(&x.to_le_bytes()),
        Value::I32(x) => w.write_all(&x.to_le_bytes()),
        Value::U64(x) => w.write_all(&x.to_le_bytes()),
        Value::I64(x) => w.write_all(&x.to_le_bytes()),
        Value::F32(x) => w.write_all(&x.to_le_bytes()),
        Value::F64(x) => w.write_all(&x.to_le_bytes()),
        Value::Bool(x) => w.write_all(&[u8::from(*x)]),
        Value::String(s) => return write_str(w, s),
        Value::Array(items) => {
            let ty = items.first().map_or(4, value_type);
            if items.iter().any(|i| value_type(i) != ty) {
                return Err(CucaError::Config(
                    "GGUF array values must all have one type".into(),
                ));
            }
            w.write_all(&ty.to_le_bytes()).map_err(io)?;
            w.write_all(&(items.len() as u64).to_le_bytes())
                .map_err(io)?;
            for item in items {
                write_value(w, item)?;
            }
            Ok(())
        }
    }
    .map_err(io)
}

/// Streams a GGUF file: header first, then tensor data in the order the
/// [`TensorSpec`]s were given.
pub(crate) struct GgufWriter<W: Write> {
    out: W,
    /// Bytes written so far.
    pos: u64,
    /// Tensors still to be written, with the bytes each still needs.
    pending: VecDeque<(String, usize)>,
    /// Bytes still needed by the tensor being written.
    left: usize,
    /// Padding bytes owed after the current tensor.
    pad: usize,
}

impl<W: Write> GgufWriter<W> {
    /// Write the header, the metadata and the tensor infos.
    ///
    /// `general.alignment` is appended to `metadata`. The tensor data follows
    /// with [`GgufWriter::write`], tensor by tensor in `tensors` order.
    pub(crate) fn start(
        mut out: W,
        metadata: &[(String, Value)],
        tensors: &[TensorSpec],
    ) -> Result<Self, CucaError> {
        let mut pos = 0u64;
        let alignment = Value::U32(ALIGNMENT as u32);
        let has_alignment = metadata.iter().any(|(k, _)| k == "general.alignment");
        let n_kv = metadata.len() + usize::from(!has_alignment);
        put(&mut out, &mut pos, &MAGIC.to_le_bytes())?;
        put(&mut out, &mut pos, &VERSION.to_le_bytes())?;
        put(&mut out, &mut pos, &(tensors.len() as u64).to_le_bytes())?;
        put(&mut out, &mut pos, &(n_kv as u64).to_le_bytes())?;

        // Keys and tensor infos are small; serialising into a scratch buffer
        // keeps the position bookkeeping in one place.
        let mut scratch: Vec<u8> = Vec::new();
        let extra = (!has_alignment).then_some(("general.alignment", &alignment));
        for (key, value) in metadata.iter().map(|(k, v)| (k.as_str(), v)).chain(extra) {
            scratch.clear();
            write_str(&mut scratch, key)?;
            scratch.extend_from_slice(&value_type(value).to_le_bytes());
            write_value(&mut scratch, value)?;
            put(&mut out, &mut pos, &scratch)?;
        }
        let mut offset = 0u64;
        for t in tensors {
            if t.dims.is_empty() || t.dims.len() > MAX_DIMS {
                return Err(CucaError::Config(format!(
                    "tensor `{}` has {} dimensions; GGUF stores 1 to {MAX_DIMS}",
                    t.name,
                    t.dims.len()
                )));
            }
            let cols = t.dims.last().copied().unwrap_or(0);
            if !cols.is_multiple_of(t.dtype.block_size()) {
                return Err(CucaError::Config(format!(
                    "tensor `{}`: row length {cols} is not a multiple of the {:?} block size",
                    t.name, t.dtype
                )));
            }
            scratch.clear();
            write_str(&mut scratch, &t.name)?;
            scratch.extend_from_slice(&(t.dims.len() as u32).to_le_bytes());
            for &d in t.dims.iter().rev() {
                scratch.extend_from_slice(&(d as u64).to_le_bytes());
            }
            scratch.extend_from_slice(&t.dtype.ggml_id().to_le_bytes());
            scratch.extend_from_slice(&offset.to_le_bytes());
            put(&mut out, &mut pos, &scratch)?;
            offset += pad_to(t.byte_len() as u64);
        }
        let padding = (pad_to(pos) - pos) as usize;
        put(&mut out, &mut pos, &vec![0u8; padding])?;

        let pending: VecDeque<(String, usize)> = tensors
            .iter()
            .map(|t| (t.name.clone(), t.byte_len()))
            .collect();
        let mut writer = Self {
            out,
            pos,
            pending,
            left: 0,
            pad: 0,
        };
        writer.advance()?;
        Ok(writer)
    }

    /// Begin the next tensor (skipping the ones with no data).
    fn advance(&mut self) -> Result<(), CucaError> {
        while self.left == 0 {
            if self.pad > 0 {
                self.out.write_all(&vec![0u8; self.pad]).map_err(io)?;
                self.pos += self.pad as u64;
                self.pad = 0;
            }
            let Some((_, len)) = self.pending.pop_front() else {
                return Ok(());
            };
            self.left = len;
            self.pad = (pad_to(len as u64) - len as u64) as usize;
            if len == 0 {
                continue;
            }
        }
        Ok(())
    }

    /// Append bytes to the current tensor; the writer moves to the next
    /// tensor by itself once the current one is complete. Bytes past the end
    /// of the last tensor are an error.
    pub(crate) fn write(&mut self, mut bytes: &[u8]) -> Result<(), CucaError> {
        while !bytes.is_empty() {
            if self.left == 0 {
                return Err(CucaError::Config(
                    "converter wrote more tensor data than it declared".into(),
                ));
            }
            let n = bytes.len().min(self.left);
            self.out.write_all(&bytes[..n]).map_err(io)?;
            self.pos += n as u64;
            self.left -= n;
            bytes = &bytes[n..];
            if self.left == 0 {
                self.advance()?;
            }
        }
        Ok(())
    }

    /// Flush and return the sink; every declared tensor must be complete.
    pub(crate) fn finish(mut self) -> Result<W, CucaError> {
        if self.left != 0 || !self.pending.is_empty() {
            let name = self
                .pending
                .front()
                .map_or("the last tensor", |(n, _)| n.as_str());
            return Err(CucaError::Config(format!(
                "converter ended before writing the data of `{name}`"
            )));
        }
        self.out.flush().map_err(io)?;
        Ok(self.out)
    }
}

/// Write `bytes` and count them.
fn put<W: Write>(out: &mut W, pos: &mut u64, bytes: &[u8]) -> Result<(), CucaError> {
    out.write_all(bytes).map_err(io)?;
    *pos += bytes.len() as u64;
    Ok(())
}

/// `n` rounded up to the alignment.
fn pad_to(n: u64) -> u64 {
    n.div_ceil(ALIGNMENT) * ALIGNMENT
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use candle_core::quantized::gguf_file::{self, Content};
    use candle_core::quantized::{GgmlDType, QTensor};
    use candle_core::{Device, Tensor};

    use super::*;
    use crate::provider::local::model::tiny::Lcg;

    fn qtensor(rng: &mut Lcg, dims: &[usize], dtype: GgmlDType) -> QTensor {
        let n: usize = dims.iter().product();
        let t = Tensor::from_vec(rng.values(n, 1.0), dims.to_vec(), &Device::Cpu).unwrap();
        QTensor::quantize(&t, dtype).unwrap()
    }

    /// A model-shaped file written by `GgufWriter` reads back, through
    /// `gguf_file::Content::read`, with the metadata, shapes, types and bytes
    /// `gguf_file::write` stores for the same tensors.
    #[test]
    fn matches_candles_writer() {
        let mut rng = Lcg(5);
        let tensors: Vec<(String, QTensor)> = vec![
            (
                "token_embd.weight".into(),
                qtensor(&mut rng, &[16, 64], GgmlDType::Q8_0),
            ),
            (
                "blk.0.attn_norm.weight".into(),
                qtensor(&mut rng, &[64], GgmlDType::F32),
            ),
            (
                "blk.0.attn_q.weight".into(),
                qtensor(&mut rng, &[32, 64], GgmlDType::Q4_0),
            ),
            (
                "blk.0.ffn_gate_exps.weight".into(),
                qtensor(&mut rng, &[3, 8, 64], GgmlDType::F16),
            ),
            (
                "blk.0.odd.weight".into(),
                qtensor(&mut rng, &[5, 7], GgmlDType::F32),
            ),
        ];
        let metadata: Vec<(String, Value)> = vec![
            ("general.architecture".into(), Value::String("llama".into())),
            ("llama.block_count".into(), Value::U32(1)),
            ("llama.rope.freq_base".into(), Value::F32(10000.0)),
            ("llama.flag".into(), Value::Bool(true)),
            (
                "tokenizer.ggml.tokens".into(),
                Value::Array(vec![Value::String("a".into()), Value::String("b".into())]),
            ),
            (
                "tokenizer.ggml.token_type".into(),
                Value::Array(vec![Value::I32(1), Value::I32(3)]),
            ),
        ];

        let mut reference = Cursor::new(Vec::new());
        let meta_ref: Vec<(&str, &Value)> = metadata.iter().map(|(k, v)| (k.as_str(), v)).collect();
        let tensor_ref: Vec<(&str, &QTensor)> =
            tensors.iter().map(|(n, t)| (n.as_str(), t)).collect();
        gguf_file::write(&mut reference, &meta_ref, &tensor_ref).unwrap();

        let specs: Vec<TensorSpec> = tensors
            .iter()
            .map(|(n, t)| TensorSpec {
                name: n.clone(),
                dims: t.shape().dims().to_vec(),
                dtype: match t.dtype() {
                    GgmlDType::F32 => Dtype::F32,
                    GgmlDType::F16 => Dtype::F16,
                    GgmlDType::Q4_0 => Dtype::Q4_0,
                    GgmlDType::Q8_0 => Dtype::Q8_0,
                    other => panic!("unexpected {other:?}"),
                },
            })
            .collect();
        let mut writer = GgufWriter::start(Cursor::new(Vec::new()), &metadata, &specs).unwrap();
        for (_, t) in &tensors {
            let data = t.data().unwrap();
            // Two chunks per tensor exercise the chunked path.
            let mid = data.len() / 2;
            writer.write(&data[..mid]).unwrap();
            writer.write(&data[mid..]).unwrap();
        }
        let written = writer.finish().unwrap().into_inner();

        let mut a = Cursor::new(reference.into_inner());
        let mut b = Cursor::new(written);
        let ca = Content::read(&mut a).unwrap();
        let cb = Content::read(&mut b).unwrap();
        assert_eq!(cb.tensor_infos.len(), ca.tensor_infos.len());
        for (key, va) in &ca.metadata {
            assert_eq!(
                format!("{va:?}"),
                format!("{:?}", cb.metadata[key]),
                "{key}"
            );
        }
        assert_eq!(cb.metadata["general.alignment"].to_u32().unwrap(), 32);
        for (name, qt) in &tensors {
            let ia = &ca.tensor_infos[name];
            let ib = &cb.tensor_infos[name];
            assert_eq!(ia.shape.dims(), ib.shape.dims(), "{name}");
            assert_eq!(ia.ggml_dtype, ib.ggml_dtype, "{name}");
            let ta = ca.tensor(&mut a, name, &Device::Cpu).unwrap();
            let tb = cb.tensor(&mut b, name, &Device::Cpu).unwrap();
            assert_eq!(ta.data().unwrap(), tb.data().unwrap(), "{name}");
            assert_eq!(qt.data().unwrap(), tb.data().unwrap(), "{name}");
        }
    }

    #[test]
    fn rejects_short_and_long_data() {
        let spec = TensorSpec {
            name: "t".into(),
            dims: vec![2, 3],
            dtype: Dtype::F32,
        };
        assert_eq!(spec.byte_len(), 24);
        let mut w = GgufWriter::start(Vec::new(), &[], std::slice::from_ref(&spec)).unwrap();
        w.write(&[0u8; 10]).unwrap();
        assert!(w.finish().is_err());
        let mut w = GgufWriter::start(Vec::new(), &[], &[spec]).unwrap();
        w.write(&[0u8; 24]).unwrap();
        assert!(w.write(&[0u8; 1]).is_err());
    }

    #[test]
    fn rejects_rows_that_do_not_fill_blocks() {
        let spec = TensorSpec {
            name: "t".into(),
            dims: vec![2, 33],
            dtype: Dtype::Q8_0,
        };
        assert!(GgufWriter::start(Vec::new(), &[], &[spec]).is_err());
    }
}
