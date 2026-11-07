//! Tensor names, layouts and values: the HF checkpoint's tensors mapped to the
//! GGUF the engine reads.
//!
//! [`plan`] walks the checkpoint's tensor names (headers only, no data) and
//! produces one [`Plan`] per GGUF tensor: its name, shape, where its values
//! come from (row ranges of checkpoint tensors, or generated values) and the
//! transforms the pinned llama.cpp converter applies (`modify_tensors`):
//! Q/K row permutation, the Gemma and Qwen 3.5 norm shifts, delta-net
//! parameter rewrites, expert stacking and splitting, multi-token-prediction
//! block renaming, the split of the latent attention's `kv_b_proj` into
//! `k_b` and `v_b`. [`Plan::write`] then streams one plan's values into the
//! writer, quantizing as it goes.
//!
//! GGUF stores `[out, in]` row-major exactly as safetensors does, so nothing
//! is ever transposed.

use std::collections::HashSet;
use std::num::NonZero;
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::sync::{Mutex, PoisonError};

use super::arch::ConvArch;
use super::cfg_err;
use super::config::{ExtraTensor, Hp};
use super::gguf_writer::{Dtype, TensorSpec};
use super::safetensors::{Shards, StDtype, TensorMeta};
use crate::error::CucaError;
use crate::provider::local::WeightQuant;
use crate::provider::local::model::layout::name as gguf;

/// Elements read and converted at once when a tensor needs no whole-tensor
/// transform (32 MiB of `f32`).
const BLOCK_ELEMS: usize = 8 << 20;

/// Elements one encoding thread takes at a time (a multiple of every block
/// size).
const ENCODE_CHUNK: usize = 1 << 16;

/// A transform of a tensor viewed as `[rows, cols]`.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Op {
    /// Add a constant (the Gemma and Qwen 3.5 norm shifts).
    Add(f32),
    /// `-exp(x)` (delta-net `A_log` to `ssm_a`).
    NegExp,
    /// `LlamaModel.permute`: reorder the rows of a Q or K projection (or
    /// bias) from HF's half-split rotary layout to adjacent pairs.
    PermuteQk { n_head: usize },
    /// `_reorder_v_heads`: value heads grouped by key head to tiled order,
    /// `nk` key heads of `nvk` value heads of `hd` rows (or columns), over the
    /// rows from `offset` (or all columns).
    ReorderV {
        cols: bool,
        offset: usize,
        nk: usize,
        nvk: usize,
        hd: usize,
    },
    /// `k_b.transpose(1, 2)`: the rows are `batch` stacked matrices, each
    /// transposed in place.
    TransposeBatches { batch: usize },
}

impl Op {
    /// Whether the op acts on every element independently, so a tensor can
    /// be converted in row blocks.
    fn elementwise(&self) -> bool {
        matches!(self, Op::Add(_) | Op::NegExp)
    }
}

/// Rows `row_start..row_start + rows` of a checkpoint tensor viewed as
/// `[shape[0], rest...]` flattened to two dimensions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Slice {
    pub(crate) tensor: String,
    pub(crate) row_start: usize,
    pub(crate) rows: usize,
}

/// Where a GGUF tensor's values come from.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Body {
    /// The rows of these slices, one after the other.
    Slices(Vec<Slice>),
    /// Generated values (rope factors).
    Generated(Vec<f32>),
}

/// One GGUF tensor to write.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Plan {
    pub(crate) name: String,
    /// Row-major shape of the result.
    pub(crate) dims: Vec<usize>,
    pub(crate) body: Body,
    pub(crate) ops: Vec<Op>,
}

/// Whether the engine reads a tensor as a weight matrix that may be
/// quantized (the converter's rule: a 2-D or larger `.weight` that is not a
/// norm, the router, or a convolution kernel).
fn is_matmul(name: &str, n_dims: usize) -> bool {
    n_dims >= 2
        && name.ends_with(".weight")
        && !name.ends_with("_norm.weight")
        && !name.ends_with(".ffn_gate_inp.weight")
        && !name.ends_with(".ffn_gate_inp_shexp.weight")
        && !name.ends_with(".ssm_conv1d.weight")
}

impl Plan {
    /// The storage type of this tensor under `quant`.
    pub(crate) fn dtype(&self, quant: WeightQuant) -> Dtype {
        let cols = self.dims.last().copied().unwrap_or(1);
        if !is_matmul(&self.name, self.dims.len()) {
            return Dtype::F32;
        }
        // llama.cpp's Q4_0 mixes keep the embedding table and the output head
        // (the tied head too) above 4 bits: the published Gemma 4 E2B Q4_0
        // has `token_embd` at Q8_0 and `per_layer_model_proj` at half
        // precision, everything else at Q4_0. `q4_0` follows it.
        let dtype = match (quant, self.name.as_str()) {
            (WeightQuant::F16, _) => Dtype::F16,
            (WeightQuant::Q8_0, _) => Dtype::Q8_0,
            (WeightQuant::Q4_0, "token_embd.weight" | "output.weight") => Dtype::Q8_0,
            (WeightQuant::Q4_0, "per_layer_model_proj.weight") => Dtype::F16,
            (WeightQuant::Q4_0, _) => Dtype::Q4_0,
        };
        if cols.is_multiple_of(dtype.block_size()) {
            dtype
        } else {
            Dtype::F32
        }
    }

    pub(crate) fn spec(&self, quant: WeightQuant) -> TensorSpec {
        TensorSpec {
            name: self.name.clone(),
            dims: self.dims.clone(),
            dtype: self.dtype(quant),
        }
    }

    /// Stream the tensor's data, encoded as `dtype`, to `sink` in whole-row
    /// chunks.
    pub(crate) fn write(
        &self,
        shards: &Shards,
        dtype: Dtype,
        sink: &mut dyn FnMut(&[u8]) -> Result<(), CucaError>,
    ) -> Result<(), CucaError> {
        let mut bytes = Vec::new();
        // Reused by every block: the values, and the rearranging ops' output.
        let mut buf = Vec::new();
        let mut tmp = Vec::new();
        match &self.body {
            Body::Generated(values) => {
                encode(dtype, values, &mut bytes)?;
                sink(&bytes)
            }
            Body::Slices(slices) if self.ops.iter().all(Op::elementwise) => {
                // A reader thread fills one buffer while this thread encodes
                // and writes the other.
                let (full_tx, full_rx) = mpsc::sync_channel(1);
                let (free_tx, free_rx) = mpsc::channel();
                for _ in 0..2 {
                    let _ = free_tx.send(Vec::new());
                }
                std::thread::scope(|scope| {
                    let result: Result<(), CucaError> = (|| {
                        std::thread::Builder::new().spawn_scoped(scope, move || {
                            read_blocks(shards, slices, &free_rx, &full_tx);
                        })?;
                        for block in &full_rx {
                            let Block {
                                mut values,
                                rows,
                                row_len,
                            } = block?;
                            for op in &self.ops {
                                apply(op, &mut values, &mut tmp, rows, row_len)?;
                            }
                            encode(dtype, &values, &mut bytes)?;
                            sink(&bytes)?;
                            let _ = free_tx.send(values);
                        }
                        Ok(())
                    })();
                    // Closing both ends frees a reader still waiting for a
                    // buffer or for room to hand over a block.
                    drop(full_rx);
                    drop(free_tx);
                    result
                })
            }
            Body::Slices(slices) => {
                let mut rows = 0;
                let mut row_len = 0;
                for s in slices {
                    shards.append_rows(&s.tensor, s.row_start, s.rows, &mut buf)?;
                    rows += s.rows;
                    row_len = shards.tensor(&s.tensor).map_or(0, |m| m.row_len());
                }
                for op in &self.ops {
                    apply(op, &mut buf, &mut tmp, rows, row_len)?;
                }
                encode(dtype, &buf, &mut bytes)?;
                sink(&bytes)
            }
        }
    }
}

/// A block of whole rows read from the checkpoint.
struct Block {
    values: Vec<f32>,
    rows: usize,
    row_len: usize,
}

/// Read `slices` in blocks of about [`BLOCK_ELEMS`] elements into the buffers
/// that arrive on `free`, handing each filled block (or the first error) to
/// `full`; returns when everything is read or either channel is closed.
fn read_blocks(
    shards: &Shards,
    slices: &[Slice],
    free: &Receiver<Vec<f32>>,
    full: &SyncSender<Result<Block, CucaError>>,
) {
    for s in slices {
        let Some(meta) = shards.tensor(&s.tensor) else {
            let _ = full.send(Err(cfg_err(format!(
                "checkpoint has no tensor `{}`",
                s.tensor
            ))));
            return;
        };
        let row_len = meta.row_len().max(1);
        let block_rows = (BLOCK_ELEMS / row_len).max(1);
        let mut done = 0;
        while done < s.rows {
            let rows = block_rows.min(s.rows - done);
            let Ok(mut values) = free.recv() else { return };
            let read = shards.read_rows(&s.tensor, s.row_start + done, rows, &mut values);
            let failed = read.is_err();
            let block = read.map(|()| Block {
                values,
                rows,
                row_len,
            });
            if full.send(block).is_err() || failed {
                return;
            }
            done += rows;
        }
    }
}

/// Apply `op` to `data` viewed as `[rows, cols]`; the ops that rearrange
/// build the result in `scratch` and swap it in.
fn apply(
    op: &Op,
    data: &mut Vec<f32>,
    scratch: &mut Vec<f32>,
    rows: usize,
    cols: usize,
) -> Result<(), CucaError> {
    match *op {
        Op::Add(c) => data.iter_mut().for_each(|x| *x += c),
        Op::NegExp => data.iter_mut().for_each(|x| *x = -x.exp()),
        Op::PermuteQk { n_head } => {
            if n_head == 0 || !rows.is_multiple_of(2 * n_head) {
                return Err(cfg_err(format!(
                    "cannot permute {rows} rows into {n_head} heads of rotary pairs"
                )));
            }
            let per_head = rows / n_head;
            let half = per_head / 2;
            scratch.clear();
            scratch.resize(data.len(), 0.0);
            for h in 0..n_head {
                for i in 0..half {
                    for j in 0..2 {
                        let from = h * per_head + j * half + i;
                        let to = h * per_head + i * 2 + j;
                        scratch[to * cols..(to + 1) * cols]
                            .copy_from_slice(&data[from * cols..(from + 1) * cols]);
                    }
                }
            }
            std::mem::swap(data, scratch);
        }
        Op::ReorderV {
            cols: on_cols,
            offset,
            nk,
            nvk,
            hd,
        } => {
            let span = nk * nvk * hd;
            scratch.clear();
            scratch.extend_from_slice(data);
            if on_cols {
                if cols != span {
                    return Err(cfg_err(format!(
                        "cannot reorder {cols} columns of {span} value channels"
                    )));
                }
                for r in 0..rows {
                    let row = &data[r * cols..(r + 1) * cols];
                    let dst = &mut scratch[r * cols..(r + 1) * cols];
                    reorder(row, dst, nk, nvk, hd);
                }
            } else {
                if rows != offset + span {
                    return Err(cfg_err(format!(
                        "cannot reorder {rows} rows as {offset} plain rows and {span} value rows"
                    )));
                }
                reorder(
                    &data[offset * cols..],
                    &mut scratch[offset * cols..],
                    nk,
                    nvk,
                    hd * cols,
                );
            }
            std::mem::swap(data, scratch);
        }
        Op::TransposeBatches { batch } => {
            if batch == 0 || !rows.is_multiple_of(batch) {
                return Err(cfg_err(format!(
                    "cannot transpose {rows} rows as {batch} matrices"
                )));
            }
            let r = rows / batch;
            scratch.clear();
            scratch.resize(data.len(), 0.0);
            for b in 0..batch {
                let src = &data[b * r * cols..(b + 1) * r * cols];
                let dst = &mut scratch[b * r * cols..(b + 1) * r * cols];
                for i in 0..r {
                    for j in 0..cols {
                        dst[j * r + i] = src[i * cols + j];
                    }
                }
            }
            std::mem::swap(data, scratch);
        }
    }
    Ok(())
}

/// `[nk, nvk, unit]` to `[nvk, nk, unit]`.
fn reorder(src: &[f32], dst: &mut [f32], nk: usize, nvk: usize, unit: usize) {
    for k in 0..nk {
        for j in 0..nvk {
            let from = (k * nvk + j) * unit;
            let to = (j * nk + k) * unit;
            dst[to..to + unit].copy_from_slice(&src[from..from + unit]);
        }
    }
}

/// Encode `data` as `dtype` into `out` (cleared first).
///
/// Every storage block depends on its own elements only, so the data is cut
/// into [`ENCODE_CHUNK`]-element pieces that the calling thread and, for a
/// large tensor, one helper thread per spare core encode in parallel: the
/// bytes are those of a single-threaded pass.
fn encode(dtype: Dtype, data: &[f32], out: &mut Vec<u8>) -> Result<(), CucaError> {
    let block = dtype.block_size();
    if !data.len().is_multiple_of(block) {
        return Err(cfg_err(format!(
            "{} values do not fill whole blocks of {block}",
            data.len()
        )));
    }
    out.clear();
    out.resize(data.len() / block * dtype.type_size(), 0);
    let piece = ENCODE_CHUNK / block * dtype.type_size();
    let work = Mutex::new(data.chunks(ENCODE_CHUNK).zip(out.chunks_mut(piece)));
    let drain = || {
        loop {
            let next = work.lock().unwrap_or_else(PoisonError::into_inner).next();
            let Some((src, dst)) = next else { return };
            encode_piece(dtype, src, dst);
        }
    };
    let helpers = if data.len() < 4 * ENCODE_CHUNK {
        0
    } else {
        std::thread::available_parallelism().map_or(1, NonZero::get) - 1
    };
    std::thread::scope(|scope| {
        for _ in 0..helpers {
            if std::thread::Builder::new()
                .spawn_scoped(scope, drain)
                .is_err()
            {
                break;
            }
        }
        drain();
    });
    Ok(())
}

/// Encode whole blocks of `src` into `dst`, which holds exactly their bytes.
///
/// The quantizers are llama.cpp's `quantize_row_q4_0` / `quantize_row_q8_0`
/// reference code (candle's `from_float` for the two types), operation for
/// operation, so the bytes are those of `ggml`.
fn encode_piece(dtype: Dtype, src: &[f32], dst: &mut [u8]) {
    match dtype {
        Dtype::F32 => {
            for (d, x) in dst.as_chunks_mut::<4>().0.iter_mut().zip(src) {
                *d = x.to_le_bytes();
            }
        }
        Dtype::F16 => {
            for (d, x) in dst.as_chunks_mut::<2>().0.iter_mut().zip(src) {
                *d = half::f16::from_f32(*x).to_le_bytes();
            }
        }
        Dtype::Q8_0 => {
            let blocks = dst.as_chunks_mut::<34>().0.iter_mut();
            for (d, xs) in blocks.zip(src.as_chunks::<32>().0) {
                let amax = xs.iter().fold(0f32, |m, x| m.max(x.abs()));
                let scale = amax / 127.0;
                let inv = if scale != 0.0 { 1.0 / scale } else { 0.0 };
                let (head, qs) = d.split_at_mut(2);
                head.copy_from_slice(&half::f16::from_f32(scale).to_le_bytes());
                for (q, x) in qs.iter_mut().zip(xs) {
                    *q = f32::round(x * inv) as i8 as u8;
                }
            }
        }
        Dtype::Q4_0 => {
            let blocks = dst.as_chunks_mut::<18>().0.iter_mut();
            for (d, xs) in blocks.zip(src.as_chunks::<32>().0) {
                let (mut amax, mut max) = (0f32, 0f32);
                for &x in xs {
                    if amax < x.abs() {
                        amax = x.abs();
                        max = x;
                    }
                }
                let scale = max / -8.0;
                let inv = if scale != 0.0 { 1.0 / scale } else { 0.0 };
                let (head, qs) = d.split_at_mut(2);
                head.copy_from_slice(&half::f16::from_f32(scale).to_le_bytes());
                let (lo, hi) = xs.split_at(16);
                for ((q, a), b) in qs.iter_mut().zip(lo).zip(hi) {
                    let qa = u8::min(15, (a * inv + 8.5) as u8);
                    let qb = u8::min(15, (b * inv + 8.5) as u8);
                    *q = qa | (qb << 4);
                }
            }
        }
    }
}

/// Whether the engine knows a GGUF tensor name (`layout::name`).
fn is_known(name: &str) -> bool {
    const GLOBAL: [&str; 12] = [
        gguf::TOKEN_EMBD,
        gguf::OUTPUT,
        gguf::OUTPUT_NORM,
        gguf::ROPE_FREQS,
        gguf::ROPE_FACTORS_LONG,
        gguf::ROPE_FACTORS_SHORT,
        gguf::PER_LAYER_TOKEN_EMBD,
        gguf::PER_LAYER_MODEL_PROJ,
        gguf::PER_LAYER_PROJ_NORM,
        "nextn.pre_projection.weight",
        "nextn.post_projection.weight",
        gguf::EAGLE_EH_PROJ,
    ];
    if GLOBAL.contains(&name) {
        return true;
    }
    let Some(rest) = name.strip_prefix("blk.") else {
        return false;
    };
    let Some((idx, part)) = rest.split_once('.') else {
        return false;
    };
    if idx.parse::<usize>().is_err() {
        return false;
    }
    const WEIGHTS: &[&str] = &[
        gguf::ATTN_NORM,
        gguf::ATTN_Q,
        gguf::ATTN_Q_NORM,
        gguf::ATTN_K,
        gguf::ATTN_K_NORM,
        gguf::ATTN_V,
        gguf::ATTN_QKV,
        gguf::ATTN_OUTPUT,
        gguf::ATTN_Q_A,
        gguf::ATTN_Q_A_NORM,
        gguf::ATTN_Q_B,
        gguf::ATTN_KV_A_MQA,
        gguf::ATTN_KV_A_NORM,
        gguf::ATTN_K_B,
        gguf::ATTN_V_B,
        gguf::POST_ATTENTION_NORM,
        gguf::FFN_NORM,
        gguf::FFN_GATE,
        gguf::FFN_UP,
        gguf::FFN_DOWN,
        gguf::FFN_GATE_INP,
        gguf::FFN_GATE_EXPS,
        gguf::FFN_UP_EXPS,
        gguf::FFN_DOWN_EXPS,
        gguf::FFN_GATE_UP_EXPS,
        gguf::FFN_GATE_SHEXP,
        gguf::FFN_UP_SHEXP,
        gguf::FFN_DOWN_SHEXP,
        gguf::FFN_GATE_INP_SHEXP,
        gguf::POST_FFW_NORM,
        gguf::INP_GATE,
        gguf::PROJ,
        gguf::POST_NORM,
        gguf::LAYER_OUTPUT_SCALE,
        gguf::ATTN_GATE,
        gguf::SSM_ALPHA,
        gguf::SSM_BETA,
        gguf::SSM_CONV1D,
        gguf::SSM_NORM,
        gguf::SSM_OUT,
        gguf::NEXTN_EH_PROJ,
        gguf::NEXTN_ENORM,
        gguf::NEXTN_HNORM,
        gguf::NEXTN_SHARED_HEAD_NORM,
    ];
    if let Some(p) = part.strip_suffix(".weight") {
        return WEIGHTS.contains(&p);
    }
    if let Some(p) = part.strip_suffix(".bias") {
        return [
            gguf::ATTN_Q,
            gguf::ATTN_K,
            gguf::ATTN_V,
            gguf::ATTN_QKV,
            gguf::ATTN_OUTPUT,
            gguf::SSM_DT,
        ]
        .contains(&p);
    }
    part == gguf::SSM_A
}

/// Tensors that exist in a checkpoint but are not part of the language model.
fn skipped(name: &str) -> bool {
    let stripped = name.strip_prefix("model.").unwrap_or(name);
    [
        "vit.",
        "vpm.",
        "siglip2.",
        "conformer.",
        "connector.",
        "merger.",
        "resampler.",
        "sound_encoder.",
        "sound_projection.",
        "speech_embeddings.",
        "mlp",
    ]
    .iter()
    .any(|p| stripped.starts_with(p))
        || [
            "visual.",
            "vision.",
            "audio.",
            "talker.",
            "vision_",
            "audio_",
            "token2wav.",
            "code2wav.",
            "projector.",
            "pre_mm_projector_norm",
            "image_newline",
            "view_seperator",
            "patch_embed",
            "patch_embedding",
            "patch_merger.",
            "patch_merge_mlp.",
        ]
        .iter()
        .any(|p| stripped.contains(p))
        || name.ends_with("rotary_emb.inv_freq")
        || name.ends_with(".attention.masked_bias")
        || name.ends_with(".attention.bias")
}

/// A tensor that exists only to dequantize an FP8 one: its `_scale_inv`
/// (applied while reading the weight, see [`Shards::read_rows`]), or an
/// activation scale (unused: the converter writes weights only).
fn is_fp8_scale(name: &str, shards: &Shards) -> bool {
    if name.ends_with(".activation_scale") || name.ends_with("_activation_scale") {
        return true;
    }
    // Mistral format: `<module>.qscale_act` and `<module>.qscale_weight`
    // next to an FP8 `<module>.weight`.
    if let Some(module) = name
        .strip_suffix(".qscale_act")
        .or_else(|| name.strip_suffix(".qscale_weight"))
    {
        return shards
            .tensor(&format!("{module}.weight"))
            .is_some_and(|w| w.dtype == StDtype::F8E4m3);
    }
    name.strip_suffix("_scale_inv")
        .and_then(|weight| shards.tensor(weight))
        .is_some_and(|w| w.dtype == StDtype::F8E4m3)
}

/// The checkpoint name with the language-model wrapper removed
/// (`model.language_model.` and `language_model.model.` become `model.`).
fn canonical(name: &str) -> String {
    for (from, to) in [
        ("model.language_model.", "model."),
        ("language_model.model.", "model."),
        ("language_model.lm_head.", "lm_head."),
    ] {
        if let Some(rest) = name.strip_prefix(from) {
            return format!("{to}{rest}");
        }
    }
    name.to_string()
}

struct Planner<'a> {
    arch: ConvArch,
    hp: &'a Hp,
    shards: &'a Shards,
    plans: Vec<Plan>,
    /// Layers whose experts are already stacked.
    stacked: HashSet<usize>,
    n_vocab: Option<usize>,
    n_layer: usize,
    tied: bool,
}

/// `model.layers.{i}.rest` split into `(i, rest)`.
fn layer_of(name: &str) -> Option<(usize, &str)> {
    let rest = name.strip_prefix("model.layers.")?;
    let (idx, rest) = rest.split_once('.')?;
    Some((idx.parse().ok()?, rest))
}

impl Planner<'_> {
    fn meta(&self, hf: &str) -> Result<&TensorMeta, CucaError> {
        self.shards
            .tensor(hf)
            .ok_or_else(|| cfg_err(format!("checkpoint has no tensor `{hf}`")))
    }

    fn shape(&self, hf: &str) -> Result<Vec<usize>, CucaError> {
        Ok(self.meta(hf)?.shape.clone())
    }

    fn rows_of(&self, hf: &str) -> Result<usize, CucaError> {
        Ok(self.meta(hf)?.rows())
    }

    /// Add a plan, which must carry a name the engine reads.
    fn push(
        &mut self,
        name: String,
        dims: Vec<usize>,
        body: Body,
        ops: Vec<Op>,
    ) -> Result<(), CucaError> {
        if !is_known(&name) {
            return Err(cfg_err(format!(
                "internal: `{name}` is not a tensor name the engine reads"
            )));
        }
        self.plans.push(Plan {
            name,
            dims,
            body,
            ops,
        });
        Ok(())
    }

    /// The whole tensor `hf` as GGUF tensor `name`.
    fn direct(&mut self, hf: &str, name: String, ops: Vec<Op>) -> Result<(), CucaError> {
        let dims = self.shape(hf)?;
        self.direct_dims(hf, name, dims, ops)
    }

    fn direct_dims(
        &mut self,
        hf: &str,
        name: String,
        dims: Vec<usize>,
        ops: Vec<Op>,
    ) -> Result<(), CucaError> {
        let rows = self.rows_of(hf)?;
        let slice = Slice {
            tensor: hf.to_string(),
            row_start: 0,
            rows,
        };
        self.push(name, dims, Body::Slices(vec![slice]), ops)
    }

    /// Rows `start..start + len` of `hf` as GGUF tensor `name`.
    fn rows(
        &mut self,
        hf: &str,
        start: usize,
        len: usize,
        name: String,
        ops: Vec<Op>,
    ) -> Result<(), CucaError> {
        let meta = self.meta(hf)?;
        let have = meta.rows();
        let mut dims = meta.shape.clone();
        if start + len > have || dims.len() != 2 {
            return Err(cfg_err(format!(
                "tensor `{hf}` of shape {dims:?} has {have} rows, need {}",
                start + len
            )));
        }
        dims[0] = len;
        let slice = Slice {
            tensor: hf.to_string(),
            row_start: start,
            rows: len,
        };
        self.push(name, dims, Body::Slices(vec![slice]), ops)
    }

    /// One slice per expert, stacked into `[E, rows, cols]`.
    fn stack(&mut self, slices: Vec<Slice>, name: String) -> Result<(), CucaError> {
        let first = slices
            .first()
            .ok_or_else(|| cfg_err("experts without tensors"))?;
        let mut dims = vec![slices.len(), first.rows];
        dims.extend(self.meta(&first.tensor)?.shape.iter().skip(1));
        self.push(name, dims, Body::Slices(slices), Vec::new())
    }

    fn n_kv_head(&self) -> Result<usize, CucaError> {
        Ok(self
            .hp
            .uint(&["n_kv_heads", "num_key_value_heads"])
            .unwrap_or(self.hp.n_head()?))
    }

    /// The norm shift of a tensor: `norm_shift` of Gemma 3, and
    /// Qwen 3.5's `+1` on every norm but the delta-net's own.
    fn norm_ops(&self, hf: &str) -> Vec<Op> {
        let shift = if self.arch.is_qwen35() {
            if hf.ends_with("norm.weight") && !hf.ends_with("linear_attn.norm.weight") {
                1.0
            } else {
                0.0
            }
        } else if self.arch.shifts_layernorms() {
            if hf.ends_with("layernorm.weight") {
                1.0
            } else {
                0.0
            }
        } else if hf.ends_with("norm.weight") {
            self.arch.norm_shift()
        } else {
            0.0
        };
        if shift == 0.0 {
            Vec::new()
        } else {
            vec![Op::Add(shift)]
        }
    }

    fn plan_one(&mut self, hf: &str) -> Result<(), CucaError> {
        if self.arch == ConvArch::Mistral4Eagle {
            return self.plan_eagle(hf);
        }
        let canon = canonical(hf);
        let arch = self.arch;
        // Multi-token-prediction head tensors of Qwen 3.5 (`mtp.*`) take the
        // names of blocks after the trunk.
        let canon = if arch.is_qwen35() {
            mtp_name(&canon, self.n_layer).unwrap_or(canon)
        } else {
            canon
        };
        match canon.as_str() {
            "model.embed_tokens.weight" => {
                let have = self.rows_of(hf)?;
                let rows = self.n_vocab.map_or(have, |n| n.min(have));
                return self.rows(hf, 0, rows, gguf::TOKEN_EMBD.into(), Vec::new());
            }
            "model.norm.weight" => {
                let ops = self.norm_ops(&canon);
                return self.direct(hf, gguf::OUTPUT_NORM.into(), ops);
            }
            "lm_head.weight" => {
                if self.tied || arch.is_gemma() {
                    return Ok(());
                }
                return self.direct(hf, gguf::OUTPUT.into(), Vec::new());
            }
            "model.embed_tokens_per_layer.weight" if arch == ConvArch::Gemma4 => {
                return self.direct(hf, gguf::PER_LAYER_TOKEN_EMBD.into(), Vec::new());
            }
            "model.per_layer_model_projection.weight" if arch == ConvArch::Gemma4 => {
                return self.direct(hf, gguf::PER_LAYER_MODEL_PROJ.into(), Vec::new());
            }
            "model.per_layer_projection_norm.weight" if arch == ConvArch::Gemma4 => {
                return self.direct(hf, gguf::PER_LAYER_PROJ_NORM.into(), Vec::new());
            }
            "pre_projection.weight" if arch == ConvArch::Gemma4Assistant => {
                return self.direct(hf, "nextn.pre_projection.weight".into(), Vec::new());
            }
            "post_projection.weight" if arch == ConvArch::Gemma4Assistant => {
                return self.direct(hf, "nextn.post_projection.weight".into(), Vec::new());
            }
            _ => {}
        }
        if arch == ConvArch::Gemma4Assistant && canon.starts_with("masked_embedding.") {
            return Ok(());
        }
        let Some((il, rest)) = layer_of(&canon) else {
            return Err(self.unmapped(hf));
        };
        self.plan_layer(hf, il, rest)
    }

    /// A tensor of the Mistral-format EAGLE drafter: the HF names of
    /// `convert_mistral4_weight_to_hf.py` read backwards (`attention_norm` is
    /// `input_layernorm`, `wkv_a_with_mqa` is `kv_a_proj_with_mqa`, `w1`/`w3`/`w2`
    /// are the gate, up and down projections), the latent attention exactly as
    /// the full model's, plus the fusion projection `eagle_linear` and the
    /// final `norm`. The drafter has no embedding and no LM head.
    fn plan_eagle(&mut self, hf: &str) -> Result<(), CucaError> {
        match hf {
            "eagle_linear.weight" => {
                return self.direct(hf, gguf::EAGLE_EH_PROJ.into(), Vec::new());
            }
            "norm.weight" => return self.direct(hf, gguf::OUTPUT_NORM.into(), Vec::new()),
            _ => {}
        }
        let layer = hf
            .strip_prefix("layers.")
            .and_then(|r| r.split_once('.'))
            .and_then(|(idx, rest)| Some((idx.parse::<usize>().ok()?, rest)));
        let Some((il, rest)) = layer else {
            return Err(self.unmapped(hf));
        };
        let part = match rest {
            "attention_norm.weight" => gguf::ATTN_NORM,
            "ffn_norm.weight" => gguf::FFN_NORM,
            "attention.wq_a.weight" => gguf::ATTN_Q_A,
            "attention.q_a_norm.weight" => gguf::ATTN_Q_A_NORM,
            "attention.wq_b.weight" => gguf::ATTN_Q_B,
            "attention.wkv_a_with_mqa.weight" => gguf::ATTN_KV_A_MQA,
            "attention.kv_a_norm.weight" => gguf::ATTN_KV_A_NORM,
            "attention.wo.weight" => gguf::ATTN_OUTPUT,
            "feed_forward.w1.weight" => gguf::FFN_GATE,
            "feed_forward.w3.weight" => gguf::FFN_UP,
            "feed_forward.w2.weight" => gguf::FFN_DOWN,
            "attention.wkv_b.weight" => return self.plan_kv_b(hf, il),
            _ => return Err(self.unmapped(hf)),
        };
        self.direct(hf, gguf::blk(il, part), Vec::new())
    }

    fn unmapped(&self, hf: &str) -> CucaError {
        cfg_err(format!(
            "tensor `{hf}` of `{}` has no GGUF name",
            self.arch.gguf_name()
        ))
    }

    fn plan_layer(&mut self, hf: &str, il: usize, rest: &str) -> Result<(), CucaError> {
        let arch = self.arch;
        let blk = |part: &str| gguf::blk(il, part);
        let bias = |part: &str| gguf::blk_bias(il, part);
        let norm = self.norm_ops(rest);

        // Muse Glimmer's Q/K norms are scale-less in the checkpoint; the GGUF
        // carries constant weights that fold in `qk_scale_factor` (Q) and 1 (K).
        if arch == ConvArch::MuseGlimmer && rest == "self_attn.q_proj.weight" {
            self.plan_muse_qk_norms(il)?;
        }
        if arch == ConvArch::Mistral4 {
            return self.plan_mla(hf, il, rest);
        }

        // Q/K projections: Llama-family rows are permuted.
        if arch.permutes_qk() {
            let permute = |n_head: usize| vec![Op::PermuteQk { n_head }];
            match rest {
                "self_attn.q_proj.weight" => {
                    let ops = permute(self.hp.n_head()?);
                    return self.direct(hf, blk(gguf::ATTN_Q), ops);
                }
                "self_attn.q_proj.bias" => {
                    let ops = permute(self.hp.n_head()?);
                    return self.direct(hf, bias(gguf::ATTN_Q), ops);
                }
                "self_attn.k_proj.weight" => {
                    let ops = permute(self.n_kv_head()?);
                    return self.direct(hf, blk(gguf::ATTN_K), ops);
                }
                "self_attn.k_proj.bias" => {
                    let ops = permute(self.n_kv_head()?);
                    return self.direct(hf, bias(gguf::ATTN_K), ops);
                }
                _ => {}
            }
        }

        if arch.is_qwen35()
            && let Some(sub) = rest.strip_prefix("linear_attn.")
        {
            return self.plan_delta(hf, il, sub);
        }
        if arch.is_qwen35() {
            // MTP head tensors renamed by `mtp_name`.
            let nextn = match rest {
                "eh_proj.weight" => Some(gguf::NEXTN_EH_PROJ),
                "enorm.weight" => Some(gguf::NEXTN_ENORM),
                "hnorm.weight" => Some(gguf::NEXTN_HNORM),
                "shared_head.norm.weight" => Some(gguf::NEXTN_SHARED_HEAD_NORM),
                _ => None,
            };
            if let Some(part) = nextn {
                return self.direct(hf, blk(part), norm);
            }
        }
        if arch.is_moe() && self.plan_moe(hf, il, rest)? {
            return Ok(());
        }

        // Fused projections kept fused (Phi 3) or split (Ernie 4.5).
        match (arch, rest) {
            (ConvArch::Phi3, "self_attn.qkv_proj.weight") => {
                return self.direct(hf, blk(gguf::ATTN_QKV), Vec::new());
            }
            (ConvArch::Phi3, "mlp.gate_up_proj.weight") => {
                return self.direct(hf, blk(gguf::FFN_UP), Vec::new());
            }
            (ConvArch::Ernie4_5, "self_attn.qkv_proj.weight") => {
                let hd = self.hp.head_dim()?;
                let (q, k) = (self.hp.n_head()? * hd, self.n_kv_head()? * hd);
                self.rows(hf, 0, q, blk(gguf::ATTN_Q), Vec::new())?;
                self.rows(hf, q, k, blk(gguf::ATTN_K), Vec::new())?;
                return self.rows(hf, q + k, k, blk(gguf::ATTN_V), Vec::new());
            }
            (ConvArch::Ernie4_5, "mlp.up_gate_proj.weight") => {
                let half = self.rows_of(hf)? / 2;
                self.rows(hf, 0, half, blk(gguf::FFN_GATE), Vec::new())?;
                return self.rows(hf, half, half, blk(gguf::FFN_UP), Vec::new());
            }
            _ => {}
        }

        let part: Option<(&str, bool)> = match rest {
            "self_attn.q_proj.weight" => Some((gguf::ATTN_Q, false)),
            "self_attn.k_proj.weight" => Some((gguf::ATTN_K, false)),
            "self_attn.v_proj.weight" => Some((gguf::ATTN_V, false)),
            "self_attn.o_proj.weight" => Some((gguf::ATTN_OUTPUT, false)),
            "self_attn.q_proj.bias" => Some((gguf::ATTN_Q, true)),
            "self_attn.k_proj.bias" => Some((gguf::ATTN_K, true)),
            "self_attn.v_proj.bias" => Some((gguf::ATTN_V, true)),
            "self_attn.o_proj.bias" => Some((gguf::ATTN_OUTPUT, true)),
            "self_attn.q_norm.weight" => Some((gguf::ATTN_Q_NORM, false)),
            "self_attn.k_norm.weight" => Some((gguf::ATTN_K_NORM, false)),
            "input_layernorm.weight" => Some((gguf::ATTN_NORM, false)),
            "mlp.gate_proj.weight" => Some((gguf::FFN_GATE, false)),
            "mlp.up_proj.weight" => Some((gguf::FFN_UP, false)),
            "mlp.down_proj.weight" => Some((gguf::FFN_DOWN, false)),
            "post_attention_layernorm.weight" => Some(match arch {
                ConvArch::Gemma3
                | ConvArch::Gemma4
                | ConvArch::Gemma4Assistant
                | ConvArch::MuseGlimmer
                | ConvArch::Olmo2
                | ConvArch::Qwen35
                | ConvArch::Qwen35Moe => (gguf::POST_ATTENTION_NORM, false),
                _ => (gguf::FFN_NORM, false),
            }),
            "pre_feedforward_layernorm.weight"
                if arch.is_gemma() || arch == ConvArch::MuseGlimmer =>
            {
                Some((gguf::FFN_NORM, false))
            }
            "post_feedforward_layernorm.weight"
                if arch.is_gemma() || arch == ConvArch::Olmo2 || arch == ConvArch::MuseGlimmer =>
            {
                Some((gguf::POST_FFW_NORM, false))
            }
            "self_attn.gate_proj.weight" if arch == ConvArch::MuseGlimmer => {
                Some((gguf::ATTN_GATE, false))
            }
            "per_layer_input_gate.weight" if arch == ConvArch::Gemma4 => {
                Some((gguf::INP_GATE, false))
            }
            "per_layer_projection.weight" if arch == ConvArch::Gemma4 => Some((gguf::PROJ, false)),
            "post_per_layer_input_norm.weight" if arch == ConvArch::Gemma4 => {
                Some((gguf::POST_NORM, false))
            }
            "layer_scalar" if arch.is_gemma() => Some((gguf::LAYER_OUTPUT_SCALE, false)),
            _ => None,
        };
        match part {
            Some((p, true)) => self.direct(hf, bias(p), Vec::new()),
            Some((p, false)) => self.direct(hf, blk(p), norm),
            None => Err(self.unmapped(hf)),
        }
    }

    /// `MuseGlimmerModel.modify_tensors`: the constant `attn_q_norm`
    /// (`qk_scale_factor`) and `attn_k_norm` (ones) weights of layer `il`,
    /// one `head_dim` vector each.
    fn plan_muse_qk_norms(&mut self, il: usize) -> Result<(), CucaError> {
        let head_dim = self.hp.head_dim()?;
        let scale =
            self.hp
                .float(&["qk_scale_factor"])
                .ok_or_else(|| cfg_err("config.json has no qk_scale_factor"))? as f32;
        for (part, value) in [(gguf::ATTN_Q_NORM, scale), (gguf::ATTN_K_NORM, 1.0)] {
            let body = Body::Generated(vec![value; head_dim]);
            self.push(gguf::blk(il, part), vec![head_dim], body, Vec::new())?;
        }
        Ok(())
    }

    /// A layer of the latent-attention mixture-of-experts family
    /// (`DeepseekV2Model.modify_tensors` under `Mistral4Model`): the
    /// low-rank attention projections, `kv_b_proj` split into the per-head
    /// `k_b` (transposed) and `v_b`, a router, shared experts, and expert
    /// stacks that the checkpoint already holds as `[E, 2 * ff, hidden]`
    /// (gate and up fused, kept fused) and `[E, hidden, ff]`.
    fn plan_mla(&mut self, hf: &str, il: usize, rest: &str) -> Result<(), CucaError> {
        let blk = |part: &str| gguf::blk(il, part);
        let part = match rest {
            "self_attn.q_a_proj.weight" => gguf::ATTN_Q_A,
            "self_attn.q_a_layernorm.weight" => gguf::ATTN_Q_A_NORM,
            "self_attn.q_b_proj.weight" => gguf::ATTN_Q_B,
            "self_attn.kv_a_proj_with_mqa.weight" => gguf::ATTN_KV_A_MQA,
            "self_attn.kv_a_layernorm.weight" => gguf::ATTN_KV_A_NORM,
            "self_attn.o_proj.weight" => gguf::ATTN_OUTPUT,
            "input_layernorm.weight" => gguf::ATTN_NORM,
            "post_attention_layernorm.weight" => gguf::FFN_NORM,
            "mlp.gate_proj.weight" => gguf::FFN_GATE,
            "mlp.up_proj.weight" => gguf::FFN_UP,
            "mlp.down_proj.weight" => gguf::FFN_DOWN,
            "mlp.gate.weight" => gguf::FFN_GATE_INP,
            "mlp.shared_experts.gate_proj.weight" => gguf::FFN_GATE_SHEXP,
            "mlp.shared_experts.up_proj.weight" => gguf::FFN_UP_SHEXP,
            "mlp.shared_experts.down_proj.weight" => gguf::FFN_DOWN_SHEXP,
            // The experts are bare parameters, not modules: no `.weight`.
            "mlp.experts.gate_up_proj" => gguf::FFN_GATE_UP_EXPS,
            "mlp.experts.down_proj" => gguf::FFN_DOWN_EXPS,
            "self_attn.kv_b_proj.weight" => return self.plan_kv_b(hf, il),
            _ => return Err(self.unmapped(hf)),
        };
        self.direct(hf, blk(part), Vec::new())
    }

    /// `kv_b_proj` `[heads * (nope + v), kv_lora]`, viewed as
    /// `[heads, nope + v, kv_lora]`, split along the second dimension into
    /// `k_b` `[heads, kv_lora, nope]` (each head transposed) and `v_b`
    /// `[heads, v, kv_lora]`.
    fn plan_kv_b(&mut self, hf: &str, il: usize) -> Result<(), CucaError> {
        // The checkpoint's head count: the converter replaces it by 1 in the
        // metadata only after the tensors are processed.
        let heads = self.hp.req_uint(&["num_key_value_heads"])?;
        let nope = self.hp.req_uint(&["qk_nope_head_dim"])?;
        let v = self.hp.req_uint(&["v_head_dim"])?;
        let shape = self.shape(hf)?;
        if shape.len() != 2 || shape[0] != heads * (nope + v) {
            return Err(cfg_err(format!(
                "`{hf}` has shape {shape:?}, expected {} rows of {heads} heads of {nope} + {v}",
                heads * (nope + v)
            )));
        }
        let lora = shape[1];
        let head_rows = |start: usize, len: usize| -> Vec<Slice> {
            (0..heads)
                .map(|h| Slice {
                    tensor: hf.to_string(),
                    row_start: h * (nope + v) + start,
                    rows: len,
                })
                .collect()
        };
        self.push(
            gguf::blk(il, gguf::ATTN_K_B),
            vec![heads, lora, nope],
            Body::Slices(head_rows(0, nope)),
            vec![Op::TransposeBatches { batch: heads }],
        )?;
        self.push(
            gguf::blk(il, gguf::ATTN_V_B),
            vec![heads, v, lora],
            Body::Slices(head_rows(nope, v)),
            Vec::new(),
        )
    }

    /// Gated delta-net parameters of a Qwen 3.5 layer (`modify_tensors` of
    /// `Qwen3NextModel` and `_LinearAttentionVReorderBase`).
    fn plan_delta(&mut self, hf: &str, il: usize, sub: &str) -> Result<(), CucaError> {
        let hp = self.hp;
        let nk = hp.req_uint(&["linear_num_key_heads"])?;
        let nv = hp.req_uint(&["linear_num_value_heads"])?;
        let hk = hp.req_uint(&["linear_key_head_dim"])?;
        let hv = hp.req_uint(&["linear_value_head_dim"])?;
        if nk == 0 || !nv.is_multiple_of(nk) {
            return Err(cfg_err(format!(
                "{nv} value heads are not a multiple of {nk} key heads"
            )));
        }
        let nvk = nv / nk;
        let tiled = nk != nv;
        let reorder = |cols: bool, offset: usize, hd: usize| -> Vec<Op> {
            tiled
                .then_some(Op::ReorderV {
                    cols,
                    offset,
                    nk,
                    nvk,
                    hd,
                })
                .into_iter()
                .collect()
        };
        let rows = |offset: usize, hd: usize| reorder(false, offset, hd);
        let blk = |part: &str| gguf::blk(il, part);
        match sub {
            "A_log" => {
                let mut ops = rows(0, 1);
                ops.push(Op::NegExp);
                self.direct(hf, gguf::blk_bare(il, gguf::SSM_A), ops)
            }
            "dt_bias" => self.direct(hf, gguf::blk_bias(il, gguf::SSM_DT), rows(0, 1)),
            "conv1d.weight" => {
                let shape = self.shape(hf)?;
                let dims = vec![shape[0], shape[1..].iter().product()];
                self.direct_dims(hf, blk(gguf::SSM_CONV1D), dims, rows(2 * hk * nk, hv))
            }
            "in_proj_qkv.weight" => self.direct(hf, blk(gguf::ATTN_QKV), rows(2 * hk * nk, hv)),
            "in_proj_z.weight" => self.direct(hf, blk(gguf::ATTN_GATE), rows(0, hv)),
            "in_proj_a.weight" => self.direct(hf, blk(gguf::SSM_ALPHA), rows(0, 1)),
            "in_proj_b.weight" => self.direct(hf, blk(gguf::SSM_BETA), rows(0, 1)),
            "out_proj.weight" => self.direct(hf, blk(gguf::SSM_OUT), reorder(true, 0, hv)),
            "norm.weight" => self.direct(hf, blk(gguf::SSM_NORM), Vec::new()),
            _ => Err(self.unmapped(hf)),
        }
    }

    /// Router, experts and shared expert of an MoE layer; whether the tensor
    /// was one of them.
    fn plan_moe(&mut self, hf: &str, il: usize, rest: &str) -> Result<bool, CucaError> {
        let arch = self.arch;
        let n_expert =
            self.hp
                .req_uint(&["num_local_experts", "num_experts", "n_routed_experts"])?;
        if rest == "mlp.gate.weight" {
            self.direct(hf, gguf::blk(il, gguf::FFN_GATE_INP), Vec::new())?;
            return Ok(true);
        }
        if arch == ConvArch::Qwen35Moe {
            let shared = match rest {
                "mlp.shared_expert.gate_proj.weight" => Some(gguf::FFN_GATE_SHEXP),
                "mlp.shared_expert.up_proj.weight" => Some(gguf::FFN_UP_SHEXP),
                "mlp.shared_expert.down_proj.weight" => Some(gguf::FFN_DOWN_SHEXP),
                "mlp.shared_expert_gate.weight" => Some(gguf::FFN_GATE_INP_SHEXP),
                _ => None,
            };
            if let Some(part) = shared {
                self.direct(hf, gguf::blk(il, part), Vec::new())?;
                return Ok(true);
            }
        }
        // Fused expert tensors: [E, 2 * ff, hidden] and [E, hidden, ff].
        let fused =
            |s: &str| s == "mlp.experts.gate_up_proj" || s == "mlp.experts.gate_up_proj.weight";
        let down = |s: &str| s == "mlp.experts.down_proj" || s == "mlp.experts.down_proj.weight";
        if fused(rest) {
            let shape = self.shape(hf)?;
            if shape.len() != 3 || shape[0] != n_expert || !shape[1].is_multiple_of(2) {
                return Err(cfg_err(format!(
                    "unexpected expert tensor shape {shape:?} for `{hf}`"
                )));
            }
            let ff = shape[1] / 2;
            for (part, off) in [(gguf::FFN_GATE_EXPS, 0), (gguf::FFN_UP_EXPS, ff)] {
                let slices = (0..n_expert)
                    .map(|e| Slice {
                        tensor: hf.to_string(),
                        row_start: e * shape[1] + off,
                        rows: ff,
                    })
                    .collect();
                let name = gguf::blk(il, part);
                let dims = vec![n_expert, ff, shape[2]];
                self.push(name, dims, Body::Slices(slices), Vec::new())?;
            }
            return Ok(true);
        }
        if down(rest) {
            self.direct(hf, gguf::blk(il, gguf::FFN_DOWN_EXPS), Vec::new())?;
            return Ok(true);
        }
        // Per-expert tensors: `mlp.experts.{e}.{gate,up,down}_proj.weight`.
        if let Some(tail) = rest.strip_prefix("mlp.experts.")
            && let Some((idx, w)) = tail.split_once('.')
            && idx.parse::<usize>().is_ok()
            && matches!(
                w,
                "gate_proj.weight" | "up_proj.weight" | "down_proj.weight"
            )
        {
            if !self.stacked.insert(il) {
                return Ok(true);
            }
            let prefix = hf
                .rsplit_once("mlp.experts.")
                .map(|(p, _)| p.to_string())
                .ok_or_else(|| self.unmapped(hf))?;
            for (w, part) in [
                ("gate_proj", gguf::FFN_GATE_EXPS),
                ("up_proj", gguf::FFN_UP_EXPS),
                ("down_proj", gguf::FFN_DOWN_EXPS),
            ] {
                let mut slices = Vec::with_capacity(n_expert);
                for e in 0..n_expert {
                    let name = format!("{prefix}mlp.experts.{e}.{w}.weight");
                    let rows = self.rows_of(&name)?;
                    slices.push(Slice {
                        tensor: name,
                        row_start: 0,
                        rows,
                    });
                }
                self.stack(slices, gguf::blk(il, part))?;
            }
            return Ok(true);
        }
        Ok(false)
    }
}

/// The converter's remapping of Qwen 3.5 `mtp.*` tensors to blocks after the
/// trunk: `mtp.layers.{i}.x` to `model.layers.{n + i}.x`, `mtp.fc` to
/// `eh_proj`, the two pre-fc norms to `enorm` and `hnorm`, and `mtp.norm` to
/// `shared_head.norm`, all in block `n`.
fn mtp_name(name: &str, n_layer: usize) -> Option<String> {
    let name = name.strip_prefix("model.").unwrap_or(name);
    let rest = name.strip_prefix("mtp.")?;
    if let Some(r) = rest.strip_prefix("layers.") {
        let (idx, tail) = r.split_once('.')?;
        let i: usize = idx.parse().ok()?;
        return Some(format!("model.layers.{}.{tail}", n_layer + i));
    }
    let (what, suffix) = rest.split_once('.')?;
    let part = match what {
        "fc" => "eh_proj",
        "pre_fc_norm_embedding" => "enorm",
        "pre_fc_norm_hidden" => "hnorm",
        "norm" => "shared_head.norm",
        _ => return None,
    };
    Some(format!("model.layers.{n_layer}.{part}.{suffix}"))
}

/// Number of `mtp.layers.*` blocks in the checkpoint.
pub(crate) fn mtp_blocks(shards: &Shards) -> usize {
    shards
        .names()
        .iter()
        .filter_map(|n| {
            let n = n.strip_prefix("model.").unwrap_or(n);
            let r = n.strip_prefix("mtp.layers.")?;
            r.split_once('.')?.0.parse::<usize>().ok()
        })
        .max()
        .map_or(0, |m| m + 1)
}

/// Plan every tensor of the checkpoint.
///
/// `n_vocab` trims the token embedding to the vocabulary's real size (Gemma
/// 3 and 4); `extra` are the generated tensors, written first. An HF tensor
/// with no GGUF name is an error, except those the converter skips by design
/// (vision and audio towers, rotary inverse frequencies, a tied `lm_head`).
pub(crate) fn plan(
    arch: ConvArch,
    hp: &Hp,
    shards: &Shards,
    extra: Vec<ExtraTensor>,
    n_vocab: Option<usize>,
) -> Result<Vec<Plan>, CucaError> {
    let mut planner = Planner {
        arch,
        hp,
        shards,
        plans: Vec::new(),
        stacked: HashSet::new(),
        n_vocab,
        n_layer: hp.n_layer()?,
        tied: hp
            .get("tie_word_embeddings")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false),
    };
    for e in extra {
        let dims = vec![e.data.len()];
        planner.push(
            e.name.to_string(),
            dims,
            Body::Generated(e.data),
            Vec::new(),
        )?;
    }
    for hf in shards.names() {
        if skipped(hf) || is_fp8_scale(hf, shards) {
            continue;
        }
        planner.plan_one(hf)?;
    }
    let mut seen = HashSet::new();
    for p in &planner.plans {
        if !seen.insert(p.name.as_str()) {
            return Err(cfg_err(format!(
                "two checkpoint tensors map to `{}`",
                p.name
            )));
        }
    }
    Ok(planner.plans)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::local::convert::safetensors::tests::{f32_bytes, temp_dir, write_file};

    /// A checkpoint with these tensors (values `0..n`), opened as shards.
    fn checkpoint(tag: &str, tensors: &[(&str, &[usize])]) -> (Shards, std::path::PathBuf) {
        let dir = temp_dir(tag);
        let data: Vec<(&str, &str, &[usize], Vec<u8>)> = tensors
            .iter()
            .map(|(n, shape)| {
                let count: usize = shape.iter().product();
                let vals: Vec<f32> = (0..count).map(|i| i as f32).collect();
                (*n, "F32", *shape, f32_bytes(&vals))
            })
            .collect();
        let path = dir.join("m.safetensors");
        write_file(&path, &data);
        (Shards::open(&[path]).unwrap(), dir)
    }

    fn hp(v: serde_json::Value) -> Hp {
        Hp::new(&v).unwrap()
    }

    fn values(shards: &Shards, p: &Plan) -> Vec<f32> {
        let mut out = Vec::new();
        p.write(shards, Dtype::F32, &mut |b| {
            out.extend(b.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)));
            Ok(())
        })
        .unwrap();
        out
    }

    fn by_name<'a>(plans: &'a [Plan], name: &str) -> &'a Plan {
        plans.iter().find(|p| p.name == name).unwrap_or_else(|| {
            panic!(
                "no `{name}` among {:?}",
                plans.iter().map(|p| &p.name).collect::<Vec<_>>()
            )
        })
    }

    #[test]
    fn llama_permutes_q_and_k_rows() {
        let (s, dir) = checkpoint(
            "t-llama",
            &[
                ("model.embed_tokens.weight", &[4, 2]),
                ("model.norm.weight", &[2]),
                ("model.layers.0.input_layernorm.weight", &[2]),
                ("model.layers.0.post_attention_layernorm.weight", &[2]),
                ("model.layers.0.self_attn.q_proj.weight", &[8, 2]),
                ("model.layers.0.self_attn.k_proj.weight", &[4, 2]),
                ("model.layers.0.self_attn.v_proj.weight", &[4, 2]),
                ("model.layers.0.self_attn.o_proj.weight", &[2, 8]),
                ("model.layers.0.mlp.gate_proj.weight", &[3, 2]),
                ("model.layers.0.mlp.up_proj.weight", &[3, 2]),
                ("model.layers.0.mlp.down_proj.weight", &[2, 3]),
                ("model.visual.patch_embed.weight", &[2, 2]),
            ],
        );
        let h = hp(serde_json::json!({
            "num_hidden_layers": 1, "hidden_size": 2, "num_attention_heads": 2, "num_key_value_heads": 1
        }));
        let plans = plan(ConvArch::Llama, &h, &s, Vec::new(), None).unwrap();
        let names: Vec<&str> = plans.iter().map(|p| p.name.as_str()).collect();
        for want in [
            "token_embd.weight",
            "output_norm.weight",
            "blk.0.attn_norm.weight",
            "blk.0.ffn_norm.weight",
            "blk.0.attn_q.weight",
            "blk.0.attn_k.weight",
            "blk.0.attn_v.weight",
            "blk.0.attn_output.weight",
            "blk.0.ffn_gate.weight",
            "blk.0.ffn_up.weight",
            "blk.0.ffn_down.weight",
        ] {
            assert!(names.contains(&want), "{want} in {names:?}");
        }
        assert_eq!(plans.len(), 11, "vision tensors are skipped");
        // 2 heads of 4 rows: per head, rows [0, 1, 2, 3] -> [0, 2, 1, 3].
        let q = values(&s, by_name(&plans, "blk.0.attn_q.weight"));
        let row = |i: usize| [2.0 * i as f32, 2.0 * i as f32 + 1.0];
        let want: Vec<f32> = [0, 2, 1, 3, 4, 6, 5, 7].into_iter().flat_map(row).collect();
        assert_eq!(q, want);
        // K has one KV head: its 4 rows are permuted as one head.
        let k = values(&s, by_name(&plans, "blk.0.attn_k.weight"));
        let want: Vec<f32> = [0, 2, 1, 3].into_iter().flat_map(row).collect();
        assert_eq!(k, want);
        let v = values(&s, by_name(&plans, "blk.0.attn_v.weight"));
        assert_eq!(v, (0..8).map(|i| i as f32).collect::<Vec<_>>());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn unmapped_tensors_are_an_error_naming_them() {
        let (s, dir) = checkpoint(
            "t-unmapped",
            &[("model.layers.0.self_attn.rotary.weight", &[2])],
        );
        let h = hp(serde_json::json!({"num_hidden_layers": 1}));
        let err = plan(ConvArch::Llama, &h, &s, Vec::new(), None)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("`model.layers.0.self_attn.rotary.weight`") && err.contains("`llama`"),
            "{err}"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn gemma3_shifts_every_norm_and_renames_the_feed_forward_norms() {
        let (s, dir) = checkpoint(
            "t-gemma3",
            &[
                ("model.embed_tokens.weight", &[4, 2]),
                ("lm_head.weight", &[4, 2]),
                ("model.norm.weight", &[2]),
                ("model.layers.0.input_layernorm.weight", &[2]),
                ("model.layers.0.post_attention_layernorm.weight", &[2]),
                ("model.layers.0.pre_feedforward_layernorm.weight", &[2]),
                ("model.layers.0.post_feedforward_layernorm.weight", &[2]),
            ],
        );
        let h = hp(serde_json::json!({"num_hidden_layers": 1}));
        let plans = plan(ConvArch::Gemma3, &h, &s, Vec::new(), None).unwrap();
        assert_eq!(plans.len(), 6, "lm_head is not exported");
        assert_eq!(
            values(&s, by_name(&plans, "blk.0.ffn_norm.weight")),
            [1.0, 2.0]
        );
        assert_eq!(
            values(&s, by_name(&plans, "blk.0.post_attention_norm.weight")),
            [1.0, 2.0]
        );
        assert_eq!(
            values(&s, by_name(&plans, "blk.0.post_ffw_norm.weight")),
            [1.0, 2.0]
        );
        assert_eq!(
            values(&s, by_name(&plans, "output_norm.weight")),
            [1.0, 2.0]
        );
        // The embedding is not a norm.
        assert_eq!(values(&s, by_name(&plans, "token_embd.weight"))[0], 0.0);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn muse_glimmer_shifts_layer_norms_synthesizes_qk_norms_and_skips_vision() {
        let (s, dir) = checkpoint(
            "t-muse",
            &[
                ("model.language_model.embed_tokens.weight", &[4, 2]),
                ("lm_head.weight", &[4, 2]),
                ("model.language_model.norm.weight", &[2]),
                ("model.language_model.layers.0.input_layernorm.weight", &[2]),
                (
                    "model.language_model.layers.0.post_attention_layernorm.weight",
                    &[2],
                ),
                (
                    "model.language_model.layers.0.pre_feedforward_layernorm.weight",
                    &[2],
                ),
                (
                    "model.language_model.layers.0.post_feedforward_layernorm.weight",
                    &[2],
                ),
                (
                    "model.language_model.layers.0.self_attn.q_proj.weight",
                    &[8, 2],
                ),
                (
                    "model.language_model.layers.0.self_attn.k_proj.weight",
                    &[4, 2],
                ),
                (
                    "model.language_model.layers.0.self_attn.v_proj.weight",
                    &[4, 2],
                ),
                (
                    "model.language_model.layers.0.self_attn.gate_proj.weight",
                    &[8, 2],
                ),
                (
                    "model.language_model.layers.0.self_attn.o_proj.weight",
                    &[2, 8],
                ),
                (
                    "model.language_model.layers.0.mlp.gate_proj.weight",
                    &[3, 2],
                ),
                ("model.language_model.layers.0.mlp.up_proj.weight", &[3, 2]),
                (
                    "model.language_model.layers.0.mlp.down_proj.weight",
                    &[2, 3],
                ),
                ("model.vision_tower.ln_post.weight", &[2]),
                ("model.vision_adapter.fc1.weight", &[2, 2]),
                ("model.vision_projection.weight", &[2, 2]),
            ],
        );
        let h = hp(serde_json::json!({
            "text_config": {"num_hidden_layers": 1, "hidden_size": 2, "num_attention_heads": 2,
                "num_key_value_heads": 1, "head_dim": 4, "qk_scale_factor": 3.87}
        }));
        let plans = plan(ConvArch::MuseGlimmer, &h, &s, Vec::new(), None).unwrap();
        // Embedding, output norm and head, four layer norms, Q/K/V/gate/O,
        // three FFN matrices and the two synthesized Q/K norms: no vision tensor.
        assert_eq!(plans.len(), 17, "{plans:?}");
        for norm in [
            "attn_norm",
            "post_attention_norm",
            "ffn_norm",
            "post_ffw_norm",
        ] {
            assert_eq!(
                values(&s, by_name(&plans, &format!("blk.0.{norm}.weight"))),
                [1.0, 2.0],
                "{norm}"
            );
        }
        // The final norm is a plain multiplier.
        assert_eq!(
            values(&s, by_name(&plans, "output_norm.weight")),
            [0.0, 1.0]
        );
        assert_eq!(
            values(&s, by_name(&plans, "blk.0.attn_q_norm.weight")),
            [3.87; 4]
        );
        assert_eq!(
            values(&s, by_name(&plans, "blk.0.attn_k_norm.weight")),
            [1.0; 4]
        );
        // Q rows are permuted per head (2 heads of 4 rows), the gate is not.
        let q = values(&s, by_name(&plans, "blk.0.attn_q.weight"));
        let row = |i: usize| [2.0 * i as f32, 2.0 * i as f32 + 1.0];
        let want: Vec<f32> = [0, 2, 1, 3, 4, 6, 5, 7].into_iter().flat_map(row).collect();
        assert_eq!(q, want);
        let gate = values(&s, by_name(&plans, "blk.0.attn_gate.weight"));
        assert_eq!(gate, (0..16).map(|i| i as f32).collect::<Vec<_>>());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn gemma4_maps_per_layer_tensors_without_a_norm_shift() {
        let (s, dir) = checkpoint(
            "t-gemma4",
            &[
                ("model.language_model.embed_tokens.weight", &[4, 2]),
                (
                    "model.language_model.embed_tokens_per_layer.weight",
                    &[4, 6],
                ),
                (
                    "model.language_model.per_layer_model_projection.weight",
                    &[6, 2],
                ),
                (
                    "model.language_model.per_layer_projection_norm.weight",
                    &[3],
                ),
                ("model.language_model.layers.0.layer_scalar", &[1]),
                (
                    "model.language_model.layers.0.per_layer_input_gate.weight",
                    &[3, 2],
                ),
                (
                    "model.language_model.layers.0.per_layer_projection.weight",
                    &[2, 3],
                ),
                (
                    "model.language_model.layers.0.post_per_layer_input_norm.weight",
                    &[2],
                ),
                ("model.language_model.layers.0.input_layernorm.weight", &[2]),
                ("model.vision_tower.encoder.weight", &[2]),
            ],
        );
        let h = hp(serde_json::json!({"num_hidden_layers": 1}));
        let plans = plan(ConvArch::Gemma4, &h, &s, Vec::new(), None).unwrap();
        for want in [
            "token_embd.weight",
            "per_layer_token_embd.weight",
            "per_layer_model_proj.weight",
            "per_layer_proj_norm.weight",
            "blk.0.layer_output_scale.weight",
            "blk.0.inp_gate.weight",
            "blk.0.proj.weight",
            "blk.0.post_norm.weight",
            "blk.0.attn_norm.weight",
        ] {
            by_name(&plans, want);
        }
        assert_eq!(
            values(&s, by_name(&plans, "blk.0.attn_norm.weight")),
            [0.0, 1.0]
        );
        assert_eq!(
            by_name(&plans, "blk.0.layer_output_scale.weight").dtype(WeightQuant::Q8_0),
            Dtype::F32
        );
        assert_eq!(
            by_name(&plans, "per_layer_proj_norm.weight").dtype(WeightQuant::Q8_0),
            Dtype::F32
        );
        // [4, 6]: 6 is not a multiple of 32, so it stays f32.
        assert_eq!(
            by_name(&plans, "per_layer_token_embd.weight").dtype(WeightQuant::Q4_0),
            Dtype::F32
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn phi3_keeps_fused_projections() {
        let (s, dir) = checkpoint(
            "t-phi3",
            &[
                ("model.layers.0.self_attn.qkv_proj.weight", &[12, 4]),
                ("model.layers.0.mlp.gate_up_proj.weight", &[6, 4]),
                ("model.layers.0.post_attention_layernorm.weight", &[4]),
            ],
        );
        let h = hp(serde_json::json!({"num_hidden_layers": 1}));
        let plans = plan(ConvArch::Phi3, &h, &s, Vec::new(), None).unwrap();
        by_name(&plans, "blk.0.attn_qkv.weight");
        by_name(&plans, "blk.0.ffn_up.weight");
        by_name(&plans, "blk.0.ffn_norm.weight");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn per_expert_tensors_stack_into_one() {
        let mut names: Vec<(String, Vec<usize>)> = Vec::new();
        for e in 0..3 {
            for (w, shape) in [
                ("gate_proj", vec![2, 4]),
                ("up_proj", vec![2, 4]),
                ("down_proj", vec![4, 2]),
            ] {
                names.push((format!("model.layers.0.mlp.experts.{e}.{w}.weight"), shape));
            }
        }
        names.push(("model.layers.0.mlp.gate.weight".into(), vec![3, 4]));
        let refs: Vec<(&str, &[usize])> = names
            .iter()
            .map(|(n, s)| (n.as_str(), s.as_slice()))
            .collect();
        let (s, dir) = checkpoint("t-experts", &refs);
        let h = hp(serde_json::json!({"num_hidden_layers": 1, "num_experts": 3}));
        let plans = plan(ConvArch::Olmoe, &h, &s, Vec::new(), None).unwrap();
        assert_eq!(plans.len(), 4);
        let up = by_name(&plans, "blk.0.ffn_up_exps.weight");
        assert_eq!(up.dims, [3, 2, 4]);
        assert_eq!(values(&s, up).len(), 24);
        assert_eq!(
            by_name(&plans, "blk.0.ffn_down_exps.weight").dims,
            [3, 4, 2]
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn qwen35_rewrites_delta_net_parameters_and_the_mtp_block() {
        let (s, dir) = checkpoint(
            "t-q35",
            &[
                ("model.language_model.embed_tokens.weight", &[4, 2]),
                ("model.language_model.norm.weight", &[2]),
                ("model.language_model.layers.0.input_layernorm.weight", &[2]),
                (
                    "model.language_model.layers.0.post_attention_layernorm.weight",
                    &[2],
                ),
                ("model.language_model.layers.0.linear_attn.A_log", &[4]),
                ("model.language_model.layers.0.linear_attn.dt_bias", &[4]),
                (
                    "model.language_model.layers.0.linear_attn.conv1d.weight",
                    &[8, 1, 3],
                ),
                (
                    "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
                    &[8, 2],
                ),
                (
                    "model.language_model.layers.0.linear_attn.in_proj_z.weight",
                    &[4, 2],
                ),
                (
                    "model.language_model.layers.0.linear_attn.in_proj_a.weight",
                    &[4, 2],
                ),
                (
                    "model.language_model.layers.0.linear_attn.in_proj_b.weight",
                    &[4, 2],
                ),
                (
                    "model.language_model.layers.0.linear_attn.norm.weight",
                    &[2],
                ),
                (
                    "model.language_model.layers.0.linear_attn.out_proj.weight",
                    &[2, 4],
                ),
                ("mtp.fc.weight", &[2, 4]),
                ("mtp.pre_fc_norm_embedding.weight", &[2]),
                ("mtp.pre_fc_norm_hidden.weight", &[2]),
                ("mtp.norm.weight", &[2]),
                ("mtp.layers.0.input_layernorm.weight", &[2]),
                ("mtp.layers.0.self_attn.q_proj.weight", &[4, 2]),
                ("model.visual.blocks.0.attn.proj.weight", &[2, 2]),
            ],
        );
        let h = hp(serde_json::json!({
            "num_hidden_layers": 1, "mtp_num_hidden_layers": 1,
            "linear_num_key_heads": 2, "linear_num_value_heads": 4,
            "linear_key_head_dim": 1, "linear_value_head_dim": 1
        }));
        assert_eq!(mtp_blocks(&s), 1);
        let plans = plan(ConvArch::Qwen35, &h, &s, Vec::new(), None).unwrap();
        // A_log -> -exp, with the 4 value heads (2 key heads x 2) reordered
        // from grouped [g0v0, g0v1, g1v0, g1v1] to tiled [g0v0, g1v0, g0v1, g1v1].
        let a = values(&s, by_name(&plans, "blk.0.ssm_a"));
        let want: Vec<f32> = [0.0f32, 2.0, 1.0, 3.0].iter().map(|x| -x.exp()).collect();
        assert_eq!(a, want);
        by_name(&plans, "blk.0.ssm_dt.bias");
        let conv = by_name(&plans, "blk.0.ssm_conv1d.weight");
        assert_eq!(conv.dims, [8, 3]);
        assert_eq!(conv.dtype(WeightQuant::Q8_0), Dtype::F32);
        // The delta-net's own norm is not shifted; every other norm is.
        assert_eq!(
            values(&s, by_name(&plans, "blk.0.ssm_norm.weight")),
            [0.0, 1.0]
        );
        assert_eq!(
            values(&s, by_name(&plans, "blk.0.post_attention_norm.weight")),
            [1.0, 2.0]
        );
        assert_eq!(
            values(&s, by_name(&plans, "output_norm.weight")),
            [1.0, 2.0]
        );
        // MTP block: block 1 holds the head tensors and the block's own layer.
        for want in [
            "blk.1.nextn.eh_proj.weight",
            "blk.1.nextn.enorm.weight",
            "blk.1.nextn.hnorm.weight",
            "blk.1.nextn.shared_head_norm.weight",
            "blk.1.attn_norm.weight",
            "blk.1.attn_q.weight",
        ] {
            by_name(&plans, want);
        }
        assert_eq!(
            values(&s, by_name(&plans, "blk.1.nextn.enorm.weight")),
            [1.0, 2.0]
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn engine_names_are_recognised() {
        assert!(is_known("blk.3.ssm_a"));
        assert!(is_known("blk.3.ssm_dt.bias"));
        assert!(is_known("blk.12.ffn_down_exps.weight"));
        assert!(is_known("nextn.pre_projection.weight"));
        assert!(!is_known("blk.3.mystery.weight"));
        assert!(!is_known("blk.x.attn_q.weight"));
    }

    #[test]
    fn quantized_blocks_match_whole_tensor_quantization() {
        // 3 rows of 64: quantizing row blocks equals quantizing the matrix.
        let rows = 3;
        let cols = 64;
        let (s, dir) = checkpoint("t-quant", &[("model.embed_tokens.weight", &[rows, cols])]);
        let h = hp(serde_json::json!({"num_hidden_layers": 1}));
        let plans = plan(ConvArch::Qwen2, &h, &s, Vec::new(), None).unwrap();
        let p = &plans[0];
        assert_eq!(p.dtype(WeightQuant::Q8_0), Dtype::Q8_0);
        let mut got = Vec::new();
        p.write(&s, Dtype::Q8_0, &mut |b| {
            got.extend_from_slice(b);
            Ok(())
        })
        .unwrap();
        assert_eq!(
            got.len(),
            TensorSpec {
                name: "x".into(),
                dims: p.dims.clone(),
                dtype: Dtype::Q8_0
            }
            .byte_len()
        );
        let all: Vec<f32> = (0..rows * cols).map(|i| i as f32).collect();
        let mut want = Vec::new();
        encode(Dtype::Q8_0, &all, &mut want).unwrap();
        assert_eq!(got, want);
        let _ = std::fs::remove_dir_all(dir);
    }

    /// The encoders (threaded, written without candle tensors) produce the
    /// bytes of candle's quantizers, which are `ggml`'s, on values that
    /// exercise ties, zero blocks, large and non-finite inputs.
    #[test]
    fn encoders_match_candle() {
        use candle_core::quantized::{GgmlDType, QTensor};
        use candle_core::{Device, Tensor};

        // Past the threading threshold, with a ragged last piece.
        let n = 6 * ENCODE_CHUNK + 96;
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut data: Vec<f32> = (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                ((state >> 40) as f32 / (1u64 << 23) as f32 - 1.0) * 3.0
            })
            .collect();
        data[..32].fill(0.0);
        data[32..64].fill(-2.5);
        data[64] = 1e30;
        data[100] = -f32::MAX;
        data[200] = f32::NAN;
        for (dtype, ggml) in [
            (Dtype::Q8_0, GgmlDType::Q8_0),
            (Dtype::Q4_0, GgmlDType::Q4_0),
        ] {
            let mut got = Vec::new();
            encode(dtype, &data, &mut got).unwrap();
            let t = Tensor::from_vec(data.clone(), (1, n), &Device::Cpu).unwrap();
            let q = QTensor::quantize(&t, ggml).unwrap();
            let want = q.data().unwrap();
            assert!(got == want.as_ref(), "{dtype:?} differs from candle");
        }
    }

    #[test]
    fn mistral4_layers_use_the_latent_attention_layout() {
        // 2 heads of nope 2, rope 1, v 3; kv_lora 4, q_lora 5, hidden 6;
        // 3 experts of width 2.
        let layer = "language_model.model.layers.0.";
        let t = |n: &str| format!("{layer}{n}");
        let tensors: Vec<(String, Vec<usize>)> = vec![
            (
                "language_model.model.embed_tokens.weight".into(),
                vec![8, 6],
            ),
            ("language_model.model.norm.weight".into(), vec![6]),
            ("language_model.lm_head.weight".into(), vec![8, 6]),
            ("vision_tower.ln_pre.weight".into(), vec![2]),
            ("multi_modal_projector.norm.weight".into(), vec![2]),
            (t("input_layernorm.weight"), vec![6]),
            (t("post_attention_layernorm.weight"), vec![6]),
            (t("self_attn.q_a_proj.weight"), vec![5, 6]),
            (t("self_attn.q_a_proj.activation_scale"), vec![]),
            (t("self_attn.q_a_layernorm.weight"), vec![5]),
            (t("self_attn.q_b_proj.weight"), vec![6, 5]),
            (t("self_attn.kv_a_proj_with_mqa.weight"), vec![5, 6]),
            (t("self_attn.kv_a_layernorm.weight"), vec![4]),
            (t("self_attn.kv_b_proj.weight"), vec![10, 4]),
            (t("self_attn.o_proj.weight"), vec![6, 6]),
            (t("mlp.gate.weight"), vec![3, 6]),
            (t("mlp.experts.gate_up_proj"), vec![3, 4, 6]),
            (t("mlp.experts.gate_up_proj_activation_scale"), vec![3]),
            (t("mlp.experts.down_proj"), vec![3, 6, 2]),
            (t("mlp.shared_experts.gate_proj.weight"), vec![2, 6]),
            (t("mlp.shared_experts.up_proj.weight"), vec![2, 6]),
            (t("mlp.shared_experts.down_proj.weight"), vec![6, 2]),
        ];
        let refs: Vec<(&str, &[usize])> = tensors
            .iter()
            .map(|(n, s)| (n.as_str(), s.as_slice()))
            .collect();
        let (s, dir) = checkpoint("t-mistral4", &refs);
        let h = hp(serde_json::json!({
            "num_hidden_layers": 1, "hidden_size": 6, "num_attention_heads": 2,
            "num_key_value_heads": 2, "qk_nope_head_dim": 2, "v_head_dim": 3,
            "n_routed_experts": 3
        }));
        let plans = plan(ConvArch::Mistral4, &h, &s, Vec::new(), None).unwrap();
        let mut names: Vec<&str> = plans.iter().map(|p| p.name.as_str()).collect();
        names.sort_unstable();
        assert_eq!(
            names,
            [
                "blk.0.attn_k_b.weight",
                "blk.0.attn_kv_a_mqa.weight",
                "blk.0.attn_kv_a_norm.weight",
                "blk.0.attn_norm.weight",
                "blk.0.attn_output.weight",
                "blk.0.attn_q_a.weight",
                "blk.0.attn_q_a_norm.weight",
                "blk.0.attn_q_b.weight",
                "blk.0.attn_v_b.weight",
                "blk.0.ffn_down_exps.weight",
                "blk.0.ffn_down_shexp.weight",
                "blk.0.ffn_gate_inp.weight",
                "blk.0.ffn_gate_shexp.weight",
                "blk.0.ffn_gate_up_exps.weight",
                "blk.0.ffn_norm.weight",
                "blk.0.ffn_up_shexp.weight",
                "output.weight",
                "output_norm.weight",
                "token_embd.weight",
            ]
        );
        // The experts keep the checkpoint's stacked, fused layout.
        assert_eq!(
            by_name(&plans, "blk.0.ffn_gate_up_exps.weight").dims,
            [3, 4, 6]
        );
        // `kv_b_proj` is `[2 heads, 2 + 3, 4]`: k_b is each head's first two
        // rows transposed, v_b the other three rows.
        let k_b = by_name(&plans, "blk.0.attn_k_b.weight");
        assert_eq!(k_b.dims, [2, 4, 2]);
        assert_eq!(
            values(&s, k_b),
            [
                0., 4., 1., 5., 2., 6., 3., 7., 20., 24., 21., 25., 22., 26., 23., 27.
            ]
        );
        let v_b = by_name(&plans, "blk.0.attn_v_b.weight");
        assert_eq!(v_b.dims, [2, 3, 4]);
        let want: Vec<f32> = (8..20).chain(28..40).map(|i| i as f32).collect();
        assert_eq!(values(&s, v_b), want);
        // Quantized, k_b and v_b stay weights; the router and norms stay f32.
        assert_eq!(k_b.dtype(WeightQuant::F16), Dtype::F16);
        assert_eq!(
            by_name(&plans, "blk.0.ffn_gate_inp.weight").dtype(WeightQuant::F16),
            Dtype::F32
        );
        let _ = std::fs::remove_dir_all(dir);
    }
}
