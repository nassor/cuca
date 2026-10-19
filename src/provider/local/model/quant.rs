//! Quantized-row access shared by the CPU and GPU engines: row dequantization
//! ([`dequant_row`]), the token-embedding table held quantized
//! ([`TokenEmbedding`]) and tensors read row by row from the GGUF file
//! ([`RowTable`]).

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::Arc;

use candle_core::Device;
use candle_core::quantized::{GgmlDType, QTensor, gguf_file};

use super::layout::tensor_info;
use super::{candle_err, err};
use crate::error::CucaError;

/// Dequantize one row of a tensor into `out`, multiplying every element by
/// `scale`.
///
/// F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1 and Q8_0 are decoded in place, with
/// no allocation. Every other dtype candle reads (the K-quants and Q8_1) goes
/// through candle's own dequantizer, which allocates per call.
///
/// `bytes` holds the row's blocks and must be exactly `out.len() /
/// block_size` blocks of `type_size` bytes (for example 18 bytes per 32
/// elements for Q4_0, 34 for Q8_0, 144 per 256 for Q4_K).
pub(crate) fn dequant_row(
    dtype: GgmlDType,
    bytes: &[u8],
    scale: f32,
    out: &mut [f32],
) -> Result<(), CucaError> {
    let block = dtype.block_size();
    if !out.len().is_multiple_of(block) || bytes.len() != out.len() / block * dtype.type_size() {
        return Err(err(format!(
            "row of {} bytes does not hold {} {dtype:?} elements",
            bytes.len(),
            out.len()
        )));
    }
    let f16_at = |b: &[u8], at: usize| {
        half::f16::from_bits(u16::from_le_bytes([b[at], b[at + 1]])).to_f32() * scale
    };
    match dtype {
        GgmlDType::F32 => {
            let (words, _) = bytes.as_chunks::<4>();
            for (o, w) in out.iter_mut().zip(words) {
                *o = f32::from_le_bytes(*w) * scale;
            }
        }
        GgmlDType::F16 => {
            let (halves, _) = bytes.as_chunks::<2>();
            for (o, h) in out.iter_mut().zip(halves) {
                *o = half::f16::from_bits(u16::from_le_bytes(*h)).to_f32() * scale;
            }
        }
        GgmlDType::BF16 => {
            let (halves, _) = bytes.as_chunks::<2>();
            for (o, h) in out.iter_mut().zip(halves) {
                *o = half::bf16::from_bits(u16::from_le_bytes(*h)).to_f32() * scale;
            }
        }
        GgmlDType::Q4_0 | GgmlDType::Q4_1 | GgmlDType::Q5_0 | GgmlDType::Q5_1 | GgmlDType::Q8_0 => {
            let (out_blocks, _) = out.as_chunks_mut::<32>();
            for (b, o) in bytes.chunks_exact(dtype.type_size()).zip(out_blocks) {
                let d = f16_at(b, 0);
                match dtype {
                    // d, 16 bytes of nibbles: low nibbles are elements 0..16,
                    // high nibbles 16..32, each minus 8.
                    GgmlDType::Q4_0 => {
                        let (lo, hi) = o.split_at_mut(16);
                        for ((lo, hi), &q) in lo.iter_mut().zip(hi).zip(&b[2..18]) {
                            *lo = f32::from(i16::from(q & 0x0F) - 8) * d;
                            *hi = f32::from(i16::from(q >> 4) - 8) * d;
                        }
                    }
                    // d, m, nibbles: element = nibble * d + m.
                    GgmlDType::Q4_1 => {
                        let m = f16_at(b, 2);
                        let (lo, hi) = o.split_at_mut(16);
                        for ((lo, hi), &q) in lo.iter_mut().zip(hi).zip(&b[4..20]) {
                            *lo = f32::from(q & 0x0F) * d + m;
                            *hi = f32::from(q >> 4) * d + m;
                        }
                    }
                    // d, 32 high bits, nibbles: a 5-bit quant minus 16. High
                    // bit j of `qh` extends low-nibble element j, bit j + 16
                    // extends high-nibble element j + 16.
                    GgmlDType::Q5_0 => {
                        let qh = u32::from_le_bytes([b[2], b[3], b[4], b[5]]);
                        let (lo, hi) = o.split_at_mut(16);
                        for (j, ((o_lo, o_hi), &q)) in
                            lo.iter_mut().zip(hi).zip(&b[6..22]).enumerate()
                        {
                            let (q_lo, q_hi) = q5_pair(q, qh, j);
                            *o_lo = f32::from(q_lo - 16) * d;
                            *o_hi = f32::from(q_hi - 16) * d;
                        }
                    }
                    // d, m, 32 high bits, nibbles: element = 5-bit quant * d + m.
                    GgmlDType::Q5_1 => {
                        let m = f16_at(b, 2);
                        let qh = u32::from_le_bytes([b[4], b[5], b[6], b[7]]);
                        let (lo, hi) = o.split_at_mut(16);
                        for (j, ((o_lo, o_hi), &q)) in
                            lo.iter_mut().zip(hi).zip(&b[8..24]).enumerate()
                        {
                            let (q_lo, q_hi) = q5_pair(q, qh, j);
                            *o_lo = f32::from(q_lo) * d + m;
                            *o_hi = f32::from(q_hi) * d + m;
                        }
                    }
                    // d, 32 signed bytes.
                    _ => {
                        for (o, &q) in o.iter_mut().zip(&b[2..34]) {
                            *o = f32::from(q.cast_signed()) * d;
                        }
                    }
                }
            }
        }
        other => {
            let info = gguf_file::TensorInfo {
                ggml_dtype: other,
                shape: out.len().into(),
                offset: 0,
            };
            let values = info
                .read(&mut std::io::Cursor::new(bytes), 0, &Device::Cpu)
                .and_then(|t| t.dequantize(&Device::Cpu))
                .and_then(|t| t.to_vec1::<f32>())
                .map_err(candle_err)?;
            for (o, v) in out.iter_mut().zip(values) {
                *o = v * scale;
            }
        }
    }
    Ok(())
}

/// The 5-bit quants (low nibble plus its high bit) of elements `j` and
/// `j + 16` of a Q5_0 or Q5_1 block.
fn q5_pair(q: u8, qh: u32, j: usize) -> (i16, i16) {
    let lo = (q & 0x0F) | ((((qh >> j) << 4) & 0x10) as u8);
    let hi = (q >> 4) | (((qh >> (j + 12)) & 0x10) as u8);
    (i16::from(lo), i16::from(hi))
}

/// The token embedding table (also the tied LM head), held quantized.
pub(crate) struct TokenEmbedding {
    tensor: Arc<QTensor>,
    dtype: GgmlDType,
    rows: usize,
    row_len: usize,
    row_bytes: usize,
}

impl TokenEmbedding {
    /// Wrap the `[rows, row_len]` table `tensor`.
    pub(crate) fn new(tensor: Arc<QTensor>) -> Result<Self, CucaError> {
        let &[rows, row_len] = tensor.shape().dims() else {
            return Err(err("the token embedding is not a matrix"));
        };
        let dtype = tensor.dtype();
        if !row_len.is_multiple_of(dtype.block_size()) {
            return Err(err(format!(
                "token embedding: row length {row_len} is not a multiple of the {dtype:?} block size {}",
                dtype.block_size()
            )));
        }
        Ok(Self {
            row_bytes: row_len / dtype.block_size() * dtype.type_size(),
            tensor,
            dtype,
            rows,
            row_len,
        })
    }

    /// Embedding rows of `tokens`, concatenated and multiplied by `scale`.
    pub(crate) fn rows(&self, tokens: &[u32], scale: f32) -> Result<Vec<f32>, CucaError> {
        let data = self.tensor.data().map_err(candle_err)?;
        let mut out = vec![0f32; tokens.len() * self.row_len];
        for (&id, dst) in tokens.iter().zip(out.chunks_exact_mut(self.row_len)) {
            let id = id as usize;
            if id >= self.rows {
                return Err(err(format!(
                    "token id {id} outside the embedding table of {} rows",
                    self.rows
                )));
            }
            let row = data
                .get(id * self.row_bytes..(id + 1) * self.row_bytes)
                .ok_or_else(|| {
                    err(format!(
                        "token embedding row {id} lies outside the tensor data"
                    ))
                })?;
            dequant_row(self.dtype, row, scale, dst)?;
        }
        Ok(out)
    }
}

/// A 2-D tensor read row by row from the GGUF file instead of being held in
/// memory: the per-layer embedding table on every device, and the token
/// embedding lookups of the GPU engine.
pub(crate) struct RowTable {
    /// GGUF tensor name, for error messages.
    tensor: &'static str,
    /// What the rows are, for error messages ("per-layer embedding table").
    label: &'static str,
    file: File,
    base: u64,
    dtype: GgmlDType,
    pub(crate) rows: usize,
    pub(crate) row_len: usize,
    row_bytes: usize,
    buf: Vec<u8>,
}

impl RowTable {
    /// Index the `[rows, row_len]` tensor `tensor` of the GGUF file at
    /// `path`; no row is read yet.
    pub(crate) fn open(
        path: &Path,
        content: &gguf_file::Content,
        tensor: &'static str,
        label: &'static str,
        row_len: usize,
    ) -> Result<Self, CucaError> {
        let info = tensor_info(content, tensor)?;
        let dims = info.shape.dims();
        if dims.len() != 2 || dims[1] != row_len {
            return Err(err(format!(
                "tensor `{tensor}`: expected shape [rows, {row_len}], found {dims:?}"
            )));
        }
        let dtype = info.ggml_dtype;
        if !row_len.is_multiple_of(dtype.block_size()) {
            return Err(err(format!(
                "tensor `{tensor}`: row length {row_len} is not a multiple of the {dtype:?} block size {}",
                dtype.block_size()
            )));
        }
        let row_bytes = row_len / dtype.block_size() * dtype.type_size();
        let base = content.tensor_data_offset + info.offset;
        let file = File::open(path)?;
        let end = base + (dims[0] * row_bytes) as u64;
        if file.metadata()?.len() < end {
            return Err(err(format!(
                "tensor `{tensor}` extends past the end of the file: truncated GGUF"
            )));
        }
        Ok(Self {
            tensor,
            label,
            file,
            base,
            dtype,
            rows: dims[0],
            row_len,
            row_bytes,
            buf: vec![0u8; row_bytes],
        })
    }

    /// Rows of `tokens`, concatenated and multiplied by `scale`.
    pub(crate) fn rows(&mut self, tokens: &[u32], scale: f32) -> Result<Vec<f32>, CucaError> {
        let tensor = self.tensor;
        let io_err = |e: std::io::Error| CucaError::Io {
            message: format!("reading `{tensor}` rows: {e}"),
        };
        let mut out = vec![0f32; tokens.len() * self.row_len];
        for (&id, dst) in tokens.iter().zip(out.chunks_exact_mut(self.row_len)) {
            let id = id as usize;
            if id >= self.rows {
                return Err(err(format!(
                    "token id {id} outside the {} of {} rows",
                    self.label, self.rows
                )));
            }
            self.file
                .seek(SeekFrom::Start(self.base + (id * self.row_bytes) as u64))
                .map_err(io_err)?;
            self.file.read_exact(&mut self.buf).map_err(io_err)?;
            dequant_row(self.dtype, &self.buf, scale, dst)?;
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use candle_core::Tensor;

    use super::*;

    fn f16_bytes(v: f32) -> [u8; 2] {
        half::f16::from_f32(v).to_bits().to_le_bytes()
    }

    #[test]
    fn dequant_row_q4_0() {
        // One block, scale 0.5: low nibbles are elements 0..16, high nibbles
        // elements 16..32, each minus 8.
        let mut block = Vec::from(f16_bytes(0.5));
        for j in 0..16u8 {
            block.push(j | ((15 - j) << 4));
        }
        let mut out = [0f32; 32];
        dequant_row(GgmlDType::Q4_0, &block, 2.0, &mut out).unwrap();
        for j in 0..16 {
            assert_eq!(out[j], (j as f32 - 8.0) * 0.5 * 2.0, "low {j}");
            assert_eq!(out[j + 16], ((15 - j) as f32 - 8.0) * 0.5 * 2.0, "high {j}");
        }
    }

    #[test]
    fn dequant_row_q8_0() {
        let mut block = Vec::from(f16_bytes(0.25));
        let quants: Vec<i8> = (0..32).map(|j| (j * 8 - 128) as i8).collect();
        block.extend(quants.iter().map(|&q| q as u8));
        // Two blocks to cover the block stride.
        block.extend_from_within(..);
        let mut out = [0f32; 64];
        dequant_row(GgmlDType::Q8_0, &block, 4.0, &mut out).unwrap();
        for (i, &o) in out.iter().enumerate() {
            assert_eq!(o, f32::from(quants[i % 32]) * 0.25 * 4.0, "element {i}");
        }
    }

    #[test]
    fn dequant_row_rejects_bad_input() {
        let mut out = [0f32; 32];
        assert!(dequant_row(GgmlDType::Q4_0, &[0u8; 17], 1.0, &mut out).is_err());
        assert!(dequant_row(GgmlDType::F32, &[0u8; 127], 1.0, &mut out).is_err());
        // 32 elements are not a whole 256-element K-quant block.
        assert!(dequant_row(GgmlDType::Q4K, &[0u8; 144], 1.0, &mut out).is_err());
    }

    /// Nibble bytes `j | (15 - j) << 4` of a 32-element legacy block.
    fn nibbles() -> Vec<u8> {
        (0..16u8).map(|j| j | ((15 - j) << 4)).collect()
    }

    #[test]
    fn dequant_row_q4_1() {
        // Element = nibble * d + m, scaled; low nibbles first, then high.
        let mut block = Vec::from(f16_bytes(0.5));
        block.extend(f16_bytes(-1.0));
        block.extend(nibbles());
        let mut out = [0f32; 32];
        dequant_row(GgmlDType::Q4_1, &block, 2.0, &mut out).unwrap();
        for j in 0..16 {
            assert_eq!(out[j], (j as f32 * 0.5 - 1.0) * 2.0, "low {j}");
            assert_eq!(out[j + 16], ((15 - j) as f32 * 0.5 - 1.0) * 2.0, "high {j}");
        }
    }

    /// The 5-bit quant of element `i` of a Q5 block: its nibble (low nibbles
    /// are elements 0..16, high nibbles 16..32) plus bit `i` of `qh` as bit 4.
    fn q5_quant(qh: u32, i: usize) -> f32 {
        let q = nibbles()[i % 16];
        let nibble = if i < 16 { q & 0x0F } else { q >> 4 };
        f32::from(nibble | ((((qh >> i) & 1) as u8) << 4))
    }

    #[test]
    fn dequant_row_q5_0() {
        let qh = 0xA5C3_0F1Eu32;
        let mut block = Vec::from(f16_bytes(0.25));
        block.extend(qh.to_le_bytes());
        block.extend(nibbles());
        let mut out = [0f32; 32];
        dequant_row(GgmlDType::Q5_0, &block, 2.0, &mut out).unwrap();
        for (i, &o) in out.iter().enumerate() {
            assert_eq!(o, (q5_quant(qh, i) - 16.0) * 0.25 * 2.0, "element {i}");
        }
    }

    #[test]
    fn dequant_row_q5_1() {
        let qh = 0x5A3C_F0E1u32;
        let mut block = Vec::from(f16_bytes(0.25));
        block.extend(f16_bytes(3.0));
        block.extend(qh.to_le_bytes());
        block.extend(nibbles());
        // Two blocks to cover the block stride.
        block.extend_from_within(..);
        let mut out = [0f32; 64];
        dequant_row(GgmlDType::Q5_1, &block, 2.0, &mut out).unwrap();
        for (i, &o) in out.iter().enumerate() {
            assert_eq!(o, (q5_quant(qh, i % 32) * 0.25 + 3.0) * 2.0, "element {i}");
        }
    }

    #[test]
    fn dequant_row_float_dtypes() {
        let values = [1.5f32, -2.0, 0.25, 8.0];
        let mut out = [0f32; 4];

        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        dequant_row(GgmlDType::F32, &bytes, 2.0, &mut out).unwrap();
        assert_eq!(out, [3.0, -4.0, 0.5, 16.0]);

        let bytes: Vec<u8> = values.iter().flat_map(|&v| f16_bytes(v)).collect();
        dequant_row(GgmlDType::F16, &bytes, 2.0, &mut out).unwrap();
        assert_eq!(out, [3.0, -4.0, 0.5, 16.0]);

        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|&v| half::bf16::from_f32(v).to_bits().to_le_bytes())
            .collect();
        dequant_row(GgmlDType::BF16, &bytes, 2.0, &mut out).unwrap();
        assert_eq!(out, [3.0, -4.0, 0.5, 16.0]);
    }

    /// Every dtype candle can quantize to must dequantize a row exactly as
    /// candle's own dequantizer does: the legacy and float paths are decoded
    /// here, the K-quants are delegated.
    #[test]
    fn dequant_row_matches_candle_for_every_dtype() {
        let row: Vec<f32> = (0..512)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.07)
            .collect();
        let tensor = Tensor::from_vec(row, 512, &Device::Cpu).unwrap();
        for dtype in [
            GgmlDType::F32,
            GgmlDType::F16,
            GgmlDType::BF16,
            GgmlDType::Q4_0,
            GgmlDType::Q4_1,
            GgmlDType::Q5_0,
            GgmlDType::Q5_1,
            GgmlDType::Q8_0,
            GgmlDType::Q2K,
            GgmlDType::Q3K,
            GgmlDType::Q4K,
            GgmlDType::Q5K,
            GgmlDType::Q6K,
        ] {
            let qt = QTensor::quantize(&tensor, dtype).unwrap();
            let expected = qt
                .dequantize(&Device::Cpu)
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let mut out = vec![0f32; 512];
            dequant_row(dtype, &qt.data().unwrap(), 1.0, &mut out).unwrap();
            assert_eq!(out, expected, "{dtype:?}");
            dequant_row(dtype, &qt.data().unwrap(), 4.0, &mut out).unwrap();
            for (o, e) in out.iter().zip(&expected) {
                assert_eq!(*o, e * 4.0, "{dtype:?}");
            }
        }
    }
}
