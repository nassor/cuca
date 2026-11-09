//! Streaming a planned tensor's values into the GGUF writer.
//!
//! Nothing here holds a whole tensor except the small ones: a matrix is read in
//! blocks of whole rows (a multiple of 32 so each block carries its own scale
//! rows), the routed experts one expert at a time and the Engram table in
//! chunks of rows. A reader thread fills one buffer (I/O and the exact
//! dequantization to `f32`) while this thread encodes and writes the previous
//! one; two buffers of at most `BLOCK_ELEMS` (32 MiB of `f32`) circulate, so the
//! peak is bounded whatever the tensor.

use std::sync::mpsc;

use super::super::gguf_writer::Dtype;
use super::super::safetensors::{BLOCK, E4M3, Shards, e8m0};
use super::super::tensors::encode;
use super::plan::{Item, Source, Table};
use crate::error::CucaError;
use crate::provider::local::convert::cfg_err;

/// Elements decoded and encoded at once (32 MiB of `f32`).
const BLOCK_ELEMS: usize = 8 << 20;

/// The 16 values of an E2M1 nibble, `[0, .5, 1, 1.5, 2, 3, 4, 6]` and their
/// negatives (`inference/convert.py`'s `FP4_TABLE`; code 8 is -0).
const FP4: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// Run `read(i, buffer)` for `i` in `0..n` on a reader thread and `consume` the
/// filled buffers in order here; buffers are recycled (two circulate).
pub(super) fn pipelined<B: Default + Send>(
    n: usize,
    read: impl Fn(usize, &mut B) -> Result<(), CucaError> + Sync,
    mut consume: impl FnMut(&B) -> Result<(), CucaError>,
) -> Result<(), CucaError> {
    if n == 0 {
        return Ok(());
    }
    let (full_tx, full_rx) = mpsc::sync_channel::<Result<B, CucaError>>(1);
    let (free_tx, free_rx) = mpsc::channel::<B>();
    for _ in 0..2 {
        let _ = free_tx.send(B::default());
    }
    std::thread::scope(|scope| {
        let read = &read;
        std::thread::Builder::new().spawn_scoped(scope, move || {
            for i in 0..n {
                let Ok(mut buffer) = free_rx.recv() else {
                    return;
                };
                let filled = read(i, &mut buffer).map(|()| buffer);
                let failed = filled.is_err();
                if full_tx.send(filled).is_err() || failed {
                    return;
                }
            }
        })?;
        let mut result = Ok(());
        for filled in &full_rx {
            match filled {
                Ok(buffer) => {
                    if let Err(e) = consume(&buffer) {
                        result = Err(e);
                        break;
                    }
                    let _ = free_tx.send(buffer);
                }
                Err(e) => {
                    result = Err(e);
                    break;
                }
            }
        }
        // Closing both ends frees a reader waiting for a buffer or for room.
        drop(full_rx);
        drop(free_tx);
        result
    })
}

/// Stream `item`'s data, encoded as its dtype, to `sink` in whole-row chunks.
///
/// # Errors
///
/// A tensor the shards lack or hold in another shape than planned.
pub(crate) fn write_item(
    item: &Item,
    shards: &Shards,
    sink: &mut dyn FnMut(&[u8]) -> Result<(), CucaError>,
) -> Result<(), CucaError> {
    write_blocks(item, shards, BLOCK_ELEMS, sink)
}

/// [`write_item`] with blocks of about `block_elems` elements.
fn write_blocks(
    item: &Item,
    shards: &Shards,
    block_elems: usize,
    sink: &mut dyn FnMut(&[u8]) -> Result<(), CucaError>,
) -> Result<(), CucaError> {
    match &item.source {
        Source::Matrix { tensor } => matrix(shards, tensor, item.dtype, block_elems, sink),
        Source::Product { a, b } => product(shards, a, b, item.dtype, sink),
        Source::Experts { prefix, proj, n } => experts(shards, prefix, proj, *n, item, sink),
        Source::Engram { tables, dim } => {
            engram(shards, tables, *dim, item.dtype, block_elems, sink)
        }
    }
}

/// A float or FP8 matrix (or vector) in blocks of whole rows.
fn matrix(
    shards: &Shards,
    tensor: &str,
    dtype: Dtype,
    block_elems: usize,
    sink: &mut dyn FnMut(&[u8]) -> Result<(), CucaError>,
) -> Result<(), CucaError> {
    let meta = shards
        .tensor(tensor)
        .ok_or_else(|| cfg_err(format!("checkpoint has no tensor `{tensor}`")))?;
    let (rows, row_len) = (meta.rows(), meta.row_len().max(1));
    // A multiple of the scale grid's 32 rows, so a block reads whole scale rows.
    let mut block_rows = (block_elems / row_len).max(1);
    if block_rows >= BLOCK {
        block_rows -= block_rows % BLOCK;
    }
    let blocks = rows.div_ceil(block_rows);
    let mut bytes = Vec::new();
    pipelined(
        blocks,
        |i, values: &mut Vec<f32>| {
            let first = i * block_rows;
            shards.block_scaled_rows(tensor, first, block_rows.min(rows - first), values)
        },
        |values| {
            encode(dtype, values, &mut bytes)?;
            sink(&bytes)
        },
    )
}

/// `a * b`, elementwise, as `f32` (Engram's `q_weight * k_weight`: bf16 times
/// bf16 is exact in `f32`).
fn product(
    shards: &Shards,
    a: &str,
    b: &str,
    dtype: Dtype,
    sink: &mut dyn FnMut(&[u8]) -> Result<(), CucaError>,
) -> Result<(), CucaError> {
    let read = |name: &str| -> Result<Vec<f32>, CucaError> {
        let rows = shards
            .tensor(name)
            .ok_or_else(|| cfg_err(format!("checkpoint has no tensor `{name}`")))?
            .rows();
        let mut out = Vec::new();
        shards.read_rows(name, 0, rows, &mut out)?;
        Ok(out)
    };
    let (mut x, y) = (read(a)?, read(b)?);
    if x.len() != y.len() {
        return Err(cfg_err(format!("`{a}` and `{b}` differ in size")));
    }
    for (x, y) in x.iter_mut().zip(&y) {
        *x *= *y;
    }
    let mut bytes = Vec::new();
    encode(dtype, &x, &mut bytes)?;
    sink(&bytes)
}

/// The scratch of one expert: its packed weights and scales, and their
/// dequantization when the tensor is stored as floats.
#[derive(Default)]
struct ExpertBuf {
    weight: Vec<u8>,
    scale: Vec<u8>,
    values: Vec<f32>,
}

/// Repack one expert's rows (`[rows, cols / 2]` bytes, low nibble = even
/// element, high nibble = odd; `[rows, cols / 32]` E8M0 scales) into MXFP4
/// blocks: per 32 elements the scale byte, then 16 bytes whose low nibble is
/// element `j` and whose high nibble is element `j + 16`. A pure byte shuffle:
/// the nibble codes and the exponent are the same numbers (ggml's MXFP4 doubles
/// the values and halves the scale).
fn repack_mxfp4(weight: &[u8], scale: &[u8], cols: usize, out: &mut Vec<u8>) {
    let (packed, blocks) = (cols / 2, cols / 32);
    out.clear();
    out.reserve(scale.len() * 17);
    for (row, scales) in weight.chunks_exact(packed).zip(scale.chunks_exact(blocks)) {
        for (block, &exponent) in row.as_chunks::<16>().0.iter().zip(scales) {
            out.push(exponent);
            // Elements `j` and `j + 16` of the block sit in bytes `j / 2` and
            // `8 + j / 2`.
            out.extend((0..16).map(|j| {
                let shift = 4 * (j & 1);
                let lo = (block[j / 2] >> shift) & 0xF;
                let hi = (block[8 + j / 2] >> shift) & 0xF;
                lo | (hi << 4)
            }));
        }
    }
}

/// An expert's FP4 values times their block scales, exactly, as `f32`.
fn dequantize_fp4(weight: &[u8], scale: &[u8], cols: usize, out: &mut Vec<f32>) {
    let (packed, blocks) = (cols / 2, cols / 32);
    out.clear();
    out.reserve(weight.len() * 2);
    for (row, scales) in weight.chunks_exact(packed).zip(scale.chunks_exact(blocks)) {
        for (block, &exponent) in row.as_chunks::<16>().0.iter().zip(scales) {
            let s = e8m0(exponent);
            for &byte in block {
                out.push(FP4[usize::from(byte & 0xF)] * s);
                out.push(FP4[usize::from(byte >> 4)] * s);
            }
        }
    }
}

/// The stack of one projection of every expert of a block, expert by expert.
fn experts(
    shards: &Shards,
    prefix: &str,
    proj: &str,
    n: usize,
    item: &Item,
    sink: &mut dyn FnMut(&[u8]) -> Result<(), CucaError>,
) -> Result<(), CucaError> {
    let (rows, cols) = (item.dims[1], item.dims[2]);
    let dtype = item.dtype;
    let mut bytes = Vec::new();
    pipelined(
        n,
        |e, buf: &mut ExpertBuf| {
            let weight = format!("{prefix}.ffn.experts.{e}.{proj}.weight");
            let scale = format!("{prefix}.ffn.experts.{e}.{proj}.scale");
            shards.read_byte_rows(&weight, 0, rows, &mut buf.weight)?;
            shards.read_byte_rows(&scale, 0, rows, &mut buf.scale)?;
            if dtype != Dtype::Mxfp4 {
                dequantize_fp4(&buf.weight, &buf.scale, cols, &mut buf.values);
            }
            Ok(())
        },
        |buf| {
            if dtype == Dtype::Mxfp4 {
                repack_mxfp4(&buf.weight, &buf.scale, cols, &mut bytes);
            } else {
                encode(dtype, &buf.values, &mut bytes)?;
            }
            sink(&bytes)
        },
    )
}

/// The scratch of one chunk of an Engram table.
#[derive(Default)]
struct RowBuf {
    weight: Vec<u8>,
    scale: Vec<u8>,
    values: Vec<f32>,
}

/// The tables concatenated, in chunks of whole rows: each element is its e4m3
/// value times the `2^(e - 127)` of its 32-column group.
fn engram(
    shards: &Shards,
    tables: &[Table],
    dim: usize,
    dtype: Dtype,
    block_elems: usize,
    sink: &mut dyn FnMut(&[u8]) -> Result<(), CucaError>,
) -> Result<(), CucaError> {
    let chunk_rows = (block_elems / dim.max(1)).max(1);
    // (table, first row, rows) of every chunk, in file order.
    let chunks: Vec<(usize, usize, usize)> = tables
        .iter()
        .enumerate()
        .flat_map(|(t, table)| {
            (0..table.rows.div_ceil(chunk_rows)).map(move |c| {
                let first = c * chunk_rows;
                (t, first, chunk_rows.min(table.rows - first))
            })
        })
        .collect();
    let groups = dim / BLOCK;
    let mut bytes = Vec::new();
    pipelined(
        chunks.len(),
        |i, buf: &mut RowBuf| {
            let (t, first, rows) = chunks[i];
            let table = &tables[t];
            shards.read_byte_rows(&table.weight, first, rows, &mut buf.weight)?;
            shards.read_byte_rows(&table.scale, first, rows, &mut buf.scale)?;
            buf.values.clear();
            buf.values.reserve(buf.weight.len());
            for (row, scales) in buf
                .weight
                .chunks_exact(dim)
                .zip(buf.scale.chunks_exact(groups))
            {
                for (group, &exponent) in row.as_chunks::<BLOCK>().0.iter().zip(scales) {
                    let s = e8m0(exponent);
                    buf.values
                        .extend(group.iter().map(|b| E4M3[usize::from(*b)] * s));
                }
            }
            Ok(())
        },
        |buf| {
            encode(dtype, &buf.values, &mut bytes)?;
            sink(&bytes)
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One 32-element block's nibbles `codes[0..32]` packed the checkpoint's way
    /// (low nibble = even element).
    fn pack(codes: &[u8]) -> Vec<u8> {
        codes.chunks(2).map(|p| p[0] | (p[1] << 4)).collect()
    }

    #[test]
    fn mxfp4_repack_pairs_element_j_with_j_plus_16() {
        let codes: Vec<u8> = (0..32).map(|i| (i * 5 + 3) % 16).collect();
        let mut out = Vec::new();
        repack_mxfp4(&pack(&codes), &[131], 32, &mut out);
        assert_eq!(out.len(), 17);
        assert_eq!(out[0], 131, "the scale byte is copied");
        for j in 0..16 {
            assert_eq!(out[1 + j] & 0xF, codes[j], "low nibble of byte {j}");
            assert_eq!(out[1 + j] >> 4, codes[j + 16], "high nibble of byte {j}");
        }
    }

    #[test]
    fn mxfp4_repack_keeps_rows_and_blocks_in_order() {
        // 2 rows of 64 columns: 4 blocks.
        let codes: Vec<u8> = (0..128).map(|i| (i * 7 + i / 5) as u8 % 16).collect();
        let scales = [120u8, 121, 122, 123];
        let mut out = Vec::new();
        repack_mxfp4(&pack(&codes), &scales, 64, &mut out);
        assert_eq!(out.len(), 4 * 17);
        for (block, &scale) in scales.iter().enumerate() {
            let at = block * 17;
            assert_eq!(out[at], scale);
            for j in 0..16 {
                assert_eq!(out[at + 1 + j] & 0xF, codes[block * 32 + j]);
                assert_eq!(out[at + 1 + j] >> 4, codes[block * 32 + j + 16]);
            }
        }
    }

    #[test]
    fn fp4_dequantization_follows_the_checkpoints_table_and_scale() {
        let codes: Vec<u8> = (0..32).map(|i| i as u8 % 16).collect();
        let mut out = Vec::new();
        dequantize_fp4(&pack(&codes), &[128], 32, &mut out);
        for (i, &v) in out.iter().enumerate() {
            assert_eq!(v, FP4[usize::from(codes[i])] * 2.0, "element {i}");
        }
        assert_eq!(
            out[8].to_bits(),
            (-0.0f32).to_bits(),
            "code 8 is negative zero"
        );
    }

    #[test]
    fn the_repack_and_the_dequantization_agree_with_ggml_mxfp4() {
        // ggml: value = kvalue(doubled grid) * 2^(e - 128) per nibble, with
        // element j in the low nibble and j + 16 in the high one.
        const KVALUES: [f32; 16] = [
            0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, -0.0, -1.0, -2.0, -3.0, -4.0, -6.0, -8.0,
            -12.0,
        ];
        let codes: Vec<u8> = (0..32).map(|i| (i * 11 + 1) as u8 % 16).collect();
        let (mut blocks, mut floats) = (Vec::new(), Vec::new());
        repack_mxfp4(&pack(&codes), &[126], 32, &mut blocks);
        dequantize_fp4(&pack(&codes), &[126], 32, &mut floats);
        let half_scale = e8m0(blocks[0]) / 2.0;
        let mut ggml = [0.0f32; 32];
        for j in 0..16 {
            ggml[j] = KVALUES[usize::from(blocks[1 + j] & 0xF)] * half_scale;
            ggml[j + 16] = KVALUES[usize::from(blocks[1 + j] >> 4)] * half_scale;
        }
        assert_eq!(floats, ggml);
    }

    #[test]
    fn the_pipeline_delivers_blocks_in_order_and_stops_on_an_error() {
        let mut seen = Vec::new();
        pipelined(
            5,
            |i, buf: &mut Vec<usize>| {
                buf.clear();
                buf.push(i * 10);
                Ok(())
            },
            |buf| {
                seen.push(buf[0]);
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(seen, [0, 10, 20, 30, 40]);

        let failed = pipelined(
            5,
            |i, _: &mut Vec<usize>| {
                if i == 2 {
                    Err(cfg_err("read failed"))
                } else {
                    Ok(())
                }
            },
            |_| Ok(()),
        );
        assert!(failed.unwrap_err().to_string().contains("read failed"));

        let mut count = 0;
        let stopped = pipelined(
            100,
            |_, _: &mut Vec<usize>| Ok(()),
            |_| {
                count += 1;
                if count == 3 {
                    Err(cfg_err("write failed"))
                } else {
                    Ok(())
                }
            },
        );
        assert!(stopped.unwrap_err().to_string().contains("write failed"));
        assert_eq!(count, 3);
    }

    use std::path::PathBuf;

    use super::super::plan::Part;
    use crate::provider::local::convert::safetensors::tests::{temp_dir, write_file};

    /// Deterministic pseudo-random bytes.
    fn bytes(n: usize, seed: u64) -> Vec<u8> {
        let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..n)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                (x >> 24) as u8
            })
            .collect()
    }

    /// Bytes of finite e4m3 values (the NaN codes `0x7F` and `0xFF` replaced).
    fn fp8_bytes(n: usize, seed: u64) -> Vec<u8> {
        bytes(n, seed)
            .into_iter()
            .map(|b| if b & 0x7F == 0x7F { b & 0x80 | 0x38 } else { b })
            .collect()
    }

    /// `[rows, cols]` e4m3 bytes times the `2^(e - 127)` of their 32x32 block,
    /// by the definition (f64 arithmetic, no shortcut).
    fn dequantized(weight: &[u8], grid: &[u8], rows: usize, cols: usize) -> Vec<f32> {
        let grid_cols = cols.div_ceil(32);
        (0..rows * cols)
            .map(|i| {
                let (r, c) = (i / cols, i % cols);
                let e = grid[(r / 32) * grid_cols + c / 32];
                (f64::from(E4M3[usize::from(weight[i])]) * 2f64.powi(i32::from(e) - 127)) as f32
            })
            .collect()
    }

    fn shards(name: &str, tensors: &[(&str, &str, &[usize], Vec<u8>)]) -> (Shards, PathBuf) {
        let dir = temp_dir(name);
        let file = dir.join("model.safetensors");
        write_file(&file, tensors);
        (Shards::open(&[file]).unwrap(), dir)
    }

    fn collect(item: &Item, shards: &Shards, block_elems: usize) -> Result<Vec<u8>, CucaError> {
        let mut out = Vec::new();
        write_blocks(item, shards, block_elems, &mut |b| {
            out.extend_from_slice(b);
            Ok(())
        })?;
        Ok(out)
    }

    fn item(name: &str, dims: &[usize], dtype: Dtype, source: Source) -> Item {
        Item {
            name: name.into(),
            dims: dims.to_vec(),
            dtype,
            source,
            part: Part::Backbone,
        }
    }

    #[test]
    fn fp8_weights_take_the_exponent_of_their_32x32_block_even_with_partial_blocks() {
        // 70 rows (two full grid rows and a 6-row one) by 70 columns (two full
        // grid columns and a 6-column one).
        let (rows, cols) = (70, 70);
        let weight = fp8_bytes(rows * cols, 1);
        let grid: Vec<u8> = (0..9).map(|i| 120 + 2 * i as u8).collect();
        let (shards, dir) = shards(
            "fp8grid",
            &[
                ("m.weight", "F8_E4M3", &[rows, cols], weight.clone()),
                ("m.scale", "F8_E8M0", &[3, 3], grid.clone()),
            ],
        );
        let want = dequantized(&weight, &grid, rows, cols);
        // Whatever the blocking (single rows, odd row counts, all at once), every
        // row sees its own scale row.
        for block_rows in [1, 7, 32, 33, 70] {
            let mut got = Vec::new();
            let mut first = 0;
            while first < rows {
                let n = block_rows.min(rows - first);
                let mut part = Vec::new();
                shards
                    .block_scaled_rows("m.weight", first, n, &mut part)
                    .unwrap();
                got.extend(part);
                first += n;
            }
            assert_eq!(got, want, "blocks of {block_rows} rows");
        }
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn an_fp8_weight_without_a_matching_scale_grid_is_refused() {
        let weight = fp8_bytes(64 * 64, 2);
        let (no_scale, d1) = shards(
            "fp8none",
            &[("m.weight", "F8_E4M3", &[64, 64], weight.clone())],
        );
        let e = no_scale
            .block_scaled_rows("m.weight", 0, 64, &mut Vec::new())
            .unwrap_err()
            .to_string();
        assert!(e.contains("m.scale"), "{e}");
        let (wrong, d2) = shards(
            "fp8wrong",
            &[
                ("m.weight", "F8_E4M3", &[64, 64], weight),
                ("m.scale", "F8_E8M0", &[1, 2], vec![127; 2]),
            ],
        );
        let e = wrong
            .block_scaled_rows("m.weight", 0, 64, &mut Vec::new())
            .unwrap_err()
            .to_string();
        assert!(e.contains("[2, 2]") && e.contains("32x32"), "{e}");
        for dir in [d1, d2] {
            let _ = std::fs::remove_dir_all(dir);
        }
    }

    #[test]
    fn a_streamed_matrix_equals_one_shot_encoding_at_any_blocking() {
        let (rows, cols) = (70, 96);
        let weight = fp8_bytes(rows * cols, 3);
        let grid: Vec<u8> = (0..9).map(|i| 124 + i as u8 % 5).collect();
        let (shards, dir) = shards(
            "matrix",
            &[
                ("m.weight", "F8_E4M3", &[rows, cols], weight.clone()),
                ("m.scale", "F8_E8M0", &[3, 3], grid.clone()),
            ],
        );
        let values = dequantized(&weight, &grid, rows, cols);
        for dtype in [Dtype::Q8_0, Dtype::F16, Dtype::F32] {
            let mut want = Vec::new();
            encode(dtype, &values, &mut want).unwrap();
            let it = item(
                "m",
                &[rows, cols],
                dtype,
                Source::Matrix {
                    tensor: "m.weight".into(),
                },
            );
            for block_elems in [1, cols * 5, cols * 33, 1 << 20] {
                assert_eq!(
                    collect(&it, &shards, block_elems).unwrap(),
                    want,
                    "{dtype:?} {block_elems}"
                );
            }
        }
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn float_matrices_and_vectors_stream_as_they_are() {
        let values: Vec<f32> = (0..24).map(|i| i as f32 * 0.25 - 3.0).collect();
        let bf16: Vec<u8> = values
            .iter()
            .flat_map(|v| half::bf16::from_f32(*v).to_le_bytes())
            .collect();
        let (shards, dir) = shards(
            "float",
            &[
                ("w.weight", "BF16", &[4, 6], bf16.clone()),
                ("n.weight", "BF16", &[24], bf16),
            ],
        );
        for (tensor, dims) in [("w.weight", vec![4, 6]), ("n.weight", vec![24])] {
            let it = item(
                "x",
                &dims,
                Dtype::F32,
                Source::Matrix {
                    tensor: tensor.into(),
                },
            );
            let got = collect(&it, &shards, 5).unwrap();
            let want: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            assert_eq!(got, want, "{tensor}");
        }
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn the_engram_gate_is_the_elementwise_product_in_f32() {
        let q: Vec<f32> = (0..8).map(|i| 1.0 + i as f32 / 8.0).collect();
        let k: Vec<f32> = (0..8).map(|i| 3.0 - i as f32 / 4.0).collect();
        let to_bf16 = |v: &[f32]| -> Vec<u8> {
            v.iter()
                .flat_map(|x| half::bf16::from_f32(*x).to_le_bytes())
                .collect()
        };
        let (shards, dir) = shards(
            "gate",
            &[
                ("q", "BF16", &[2, 4], to_bf16(&q)),
                ("k", "BF16", &[2, 4], to_bf16(&k)),
            ],
        );
        let it = item(
            "g",
            &[2, 4],
            Dtype::F32,
            Source::Product {
                a: "q".into(),
                b: "k".into(),
            },
        );
        let got = collect(&it, &shards, 100).unwrap();
        let want: Vec<u8> = q
            .iter()
            .zip(&k)
            .flat_map(|(a, b)| {
                (half::bf16::from_f32(*a).to_f32() * half::bf16::from_f32(*b).to_f32())
                    .to_le_bytes()
            })
            .collect();
        assert_eq!(got, want);
        let _ = std::fs::remove_dir_all(dir);
    }

    /// Three experts of `[rows, cols]` FP4 weights with their scales.
    fn expert_shards(name: &str, n: usize, rows: usize, cols: usize) -> (Shards, PathBuf) {
        let mut owned: Vec<(String, &str, Vec<usize>, Vec<u8>)> = Vec::new();
        for e in 0..n {
            owned.push((
                format!("b.ffn.experts.{e}.w1.weight"),
                "I8",
                vec![rows, cols / 2],
                bytes(rows * cols / 2, 10 + e as u64),
            ));
            owned.push((
                format!("b.ffn.experts.{e}.w1.scale"),
                "F8_E8M0",
                vec![rows, cols / 32],
                bytes(rows * cols / 32, 50 + e as u64)
                    .iter()
                    .map(|b| 120 + b % 12)
                    .collect(),
            ));
        }
        let refs: Vec<(&str, &str, &[usize], Vec<u8>)> = owned
            .iter()
            .map(|(n, d, s, b)| (n.as_str(), *d, s.as_slice(), b.clone()))
            .collect();
        shards(name, &refs)
    }

    #[test]
    fn expert_stacks_repack_to_mxfp4_and_dequantize_to_the_same_values() {
        let (n, rows, cols) = (3, 5, 64);
        let (shards, dir) = expert_shards("experts", n, rows, cols);
        let source = Source::Experts {
            prefix: "b".into(),
            proj: "w1",
            n,
        };
        let mx = collect(
            &item("e", &[n, rows, cols], Dtype::Mxfp4, source.clone()),
            &shards,
            1,
        )
        .unwrap();
        assert_eq!(mx.len(), n * rows * cols / 32 * 17);
        let f16 = collect(&item("e", &[n, rows, cols], Dtype::F16, source), &shards, 1).unwrap();
        // Dequantize the MXFP4 bytes the way ggml does and compare with the
        // exactly-dequantized half floats of the same experts.
        const KVALUES: [f32; 16] = [
            0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, -0.0, -1.0, -2.0, -3.0, -4.0, -6.0, -8.0,
            -12.0,
        ];
        let mut from_blocks = Vec::new();
        for block in mx.as_chunks::<17>().0.iter() {
            let half_scale = e8m0(block[0]) / 2.0;
            let mut values = [0.0f32; 32];
            for j in 0..16 {
                values[j] = KVALUES[usize::from(block[1 + j] & 0xF)] * half_scale;
                values[j + 16] = KVALUES[usize::from(block[1 + j] >> 4)] * half_scale;
            }
            from_blocks.extend(values);
        }
        let from_f16: Vec<f32> = f16
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32())
            .collect();
        assert_eq!(from_f16, from_blocks);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn a_missing_or_misshapen_expert_names_the_tensor() {
        let (shards, dir) = expert_shards("experts-missing", 2, 4, 64);
        let source = Source::Experts {
            prefix: "b".into(),
            proj: "w1",
            n: 3,
        };
        let e = collect(&item("e", &[3, 4, 64], Dtype::Mxfp4, source), &shards, 1)
            .unwrap_err()
            .to_string();
        assert!(e.contains("b.ffn.experts.2.w1.weight"), "{e}");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn engram_tables_concatenate_in_chunks_that_cross_table_boundaries() {
        let dim = 64;
        let (rows_a, rows_b) = (5, 7);
        let mut tensors: Vec<(String, &str, Vec<usize>, Vec<u8>)> = Vec::new();
        let mut want = Vec::new();
        for (t, rows) in [rows_a, rows_b].into_iter().enumerate() {
            let weight = fp8_bytes(rows * dim, 70 + t as u64);
            let scale: Vec<u8> = bytes(rows * 2, 80 + t as u64)
                .iter()
                .map(|b| 122 + b % 8)
                .collect();
            for r in 0..rows {
                for c in 0..dim {
                    let e = scale[r * 2 + c / 32];
                    want.push(
                        E4M3[usize::from(weight[r * dim + c])] * 2f32.powi(i32::from(e) - 127),
                    );
                }
            }
            tensors.push((format!("t{t}.weight"), "F8_E4M3", vec![rows, dim], weight));
            tensors.push((format!("t{t}.scale"), "F8_E8M0", vec![rows, 2], scale));
        }
        let refs: Vec<(&str, &str, &[usize], Vec<u8>)> = tensors
            .iter()
            .map(|(n, d, s, b)| (n.as_str(), *d, s.as_slice(), b.clone()))
            .collect();
        let (shards, dir) = shards("engram", &refs);
        let tables = || -> Vec<Table> {
            (0..2)
                .map(|t| Table {
                    weight: format!("t{t}.weight"),
                    scale: format!("t{t}.scale"),
                    rows: [rows_a, rows_b][t],
                })
                .collect()
        };
        for dtype in [Dtype::Bf16, Dtype::Q8_0, Dtype::Q4_0] {
            let source = Source::Engram {
                tables: tables(),
                dim,
            };
            let it = item("p", &[rows_a + rows_b, dim], dtype, source);
            let mut one_shot = Vec::new();
            encode(dtype, &want, &mut one_shot).unwrap();
            // Chunks of 1, 3 and 100 rows (more than a table).
            for chunk_rows in [1, 3, 100] {
                assert_eq!(
                    collect(&it, &shards, chunk_rows * dim).unwrap(),
                    one_shot,
                    "{dtype:?} in chunks of {chunk_rows} rows"
                );
            }
        }
        let _ = std::fs::remove_dir_all(dir);
    }
}
