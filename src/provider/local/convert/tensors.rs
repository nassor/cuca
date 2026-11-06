//! Tensor names, layouts and values: the HF checkpoint's tensors mapped to the
//! GGUF the engine reads.
//!
//! [`plan`] walks the checkpoint's tensor names (headers only, no data) and
//! produces one [`Plan`] per GGUF tensor: its name, shape, where its values
//! come from (row ranges of checkpoint tensors, or generated values) and the
//! transforms the pinned llama.cpp converter applies (`modify_tensors`):
//! Q/K row permutation, the Gemma and Qwen 3.5 norm shifts, delta-net
//! parameter rewrites, expert stacking and splitting, multi-token-prediction
//! block renaming. [`Plan::write`] then streams one plan's values into the
//! writer, quantizing as it goes.
//!
//! GGUF stores `[out, in]` row-major exactly as safetensors does, so nothing
//! is ever transposed.

use std::collections::HashSet;

use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{Device, Tensor};

use super::arch::ConvArch;
use super::config::{ExtraTensor, Hp};
use super::gguf_writer::{Dtype, TensorSpec};
use super::safetensors::Shards;
use crate::error::CucaError;
use crate::provider::local::WeightQuant;
use crate::provider::local::model::layout::name as gguf;

fn cfg_err(msg: impl Into<String>) -> CucaError {
    CucaError::Config(msg.into())
}

/// Elements read and converted at once when a tensor needs no whole-tensor
/// transform (32 MiB of `f32`).
const BLOCK_ELEMS: usize = 8 << 20;

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
        let cols = self.dims.last().copied().unwrap_or(1);
        let mut bytes = Vec::new();
        match &self.body {
            Body::Generated(values) => {
                encode(dtype, values.clone(), cols, &mut bytes)?;
                sink(&bytes)
            }
            Body::Slices(slices) if self.ops.iter().all(Op::elementwise) => {
                let mut buf = Vec::new();
                for s in slices {
                    let row_len = shards
                        .tensor(&s.tensor)
                        .map(|m| m.row_len())
                        .ok_or_else(|| cfg_err(format!("checkpoint has no tensor `{}`", s.tensor)))?
                        .max(1);
                    let block = (BLOCK_ELEMS / row_len).max(1);
                    let mut done = 0;
                    while done < s.rows {
                        let n = block.min(s.rows - done);
                        shards.read_rows(&s.tensor, s.row_start + done, n, &mut buf)?;
                        for op in &self.ops {
                            apply(op, &mut buf, n, row_len)?;
                        }
                        encode(dtype, std::mem::take(&mut buf), cols, &mut bytes)?;
                        sink(&bytes)?;
                        done += n;
                    }
                }
                Ok(())
            }
            Body::Slices(slices) => {
                let mut all: Vec<f32> = Vec::new();
                let mut part = Vec::new();
                let mut rows = 0;
                let mut row_len = 0;
                for s in slices {
                    shards.read_rows(&s.tensor, s.row_start, s.rows, &mut part)?;
                    all.extend_from_slice(&part);
                    rows += s.rows;
                    row_len = shards.tensor(&s.tensor).map_or(0, |m| m.row_len());
                }
                for op in &self.ops {
                    apply(op, &mut all, rows, row_len)?;
                }
                encode(dtype, all, cols, &mut bytes)?;
                sink(&bytes)
            }
        }
    }
}

/// Apply `op` to `data` viewed as `[rows, cols]`.
fn apply(op: &Op, data: &mut Vec<f32>, rows: usize, cols: usize) -> Result<(), CucaError> {
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
            let mut out = vec![0.0f32; data.len()];
            for h in 0..n_head {
                for i in 0..half {
                    for j in 0..2 {
                        let from = h * per_head + j * half + i;
                        let to = h * per_head + i * 2 + j;
                        out[to * cols..(to + 1) * cols]
                            .copy_from_slice(&data[from * cols..(from + 1) * cols]);
                    }
                }
            }
            *data = out;
        }
        Op::ReorderV {
            cols: on_cols,
            offset,
            nk,
            nvk,
            hd,
        } => {
            let span = nk * nvk * hd;
            let mut out = data.clone();
            if on_cols {
                if cols != span {
                    return Err(cfg_err(format!(
                        "cannot reorder {cols} columns of {span} value channels"
                    )));
                }
                for r in 0..rows {
                    let row = &data[r * cols..(r + 1) * cols];
                    let dst = &mut out[r * cols..(r + 1) * cols];
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
                    &mut out[offset * cols..],
                    nk,
                    nvk,
                    hd * cols,
                );
            }
            *data = out;
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

/// Encode whole rows of `data` as `dtype` into `out` (cleared first).
fn encode(dtype: Dtype, data: Vec<f32>, cols: usize, out: &mut Vec<u8>) -> Result<(), CucaError> {
    out.clear();
    match dtype {
        Dtype::F32 => out.extend(data.iter().flat_map(|x| x.to_le_bytes())),
        Dtype::F16 => out.extend(
            data.iter()
                .flat_map(|x| half::f16::from_f32(*x).to_le_bytes()),
        ),
        Dtype::Q8_0 | Dtype::Q4_0 => {
            let rows = data.len() / cols.max(1);
            let g = if dtype == Dtype::Q8_0 {
                GgmlDType::Q8_0
            } else {
                GgmlDType::Q4_0
            };
            let t = Tensor::from_vec(data, (rows, cols), &Device::Cpu)
                .map_err(|e| cfg_err(e.to_string()))?;
            let q = QTensor::quantize(&t, g).map_err(|e| cfg_err(e.to_string()))?;
            out.extend_from_slice(&q.data().map_err(|e| cfg_err(e.to_string()))?);
        }
    }
    Ok(())
}

/// Whether the engine knows a GGUF tensor name (`layout::name`).
pub(crate) fn is_known(name: &str) -> bool {
    const GLOBAL: [&str; 11] = [
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
        gguf::POST_ATTENTION_NORM,
        gguf::FFN_NORM,
        gguf::FFN_GATE,
        gguf::FFN_UP,
        gguf::FFN_DOWN,
        gguf::FFN_GATE_INP,
        gguf::FFN_GATE_EXPS,
        gguf::FFN_UP_EXPS,
        gguf::FFN_DOWN_EXPS,
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
    fn shape(&self, hf: &str) -> Result<Vec<usize>, CucaError> {
        self.shards
            .tensor(hf)
            .map(|m| m.shape.clone())
            .ok_or_else(|| cfg_err(format!("checkpoint has no tensor `{hf}`")))
    }

    fn rows_of(&self, hf: &str) -> Result<usize, CucaError> {
        self.shards
            .tensor(hf)
            .map(|m| m.rows())
            .ok_or_else(|| cfg_err(format!("checkpoint has no tensor `{hf}`")))
    }

    fn check(&self, name: &str) -> Result<(), CucaError> {
        if is_known(name) {
            Ok(())
        } else {
            Err(cfg_err(format!(
                "internal: `{name}` is not a tensor name the engine reads"
            )))
        }
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
        self.check(&name)?;
        let rows = self.rows_of(hf)?;
        self.plans.push(Plan {
            name,
            dims,
            body: Body::Slices(vec![Slice {
                tensor: hf.to_string(),
                row_start: 0,
                rows,
            }]),
            ops,
        });
        Ok(())
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
        self.check(&name)?;
        let mut dims = self.shape(hf)?;
        let have = self.rows_of(hf)?;
        if start + len > have || dims.len() != 2 {
            return Err(cfg_err(format!(
                "tensor `{hf}` of shape {dims:?} has {have} rows, need {}",
                start + len
            )));
        }
        dims[0] = len;
        self.plans.push(Plan {
            name,
            dims,
            body: Body::Slices(vec![Slice {
                tensor: hf.to_string(),
                row_start: start,
                rows: len,
            }]),
            ops,
        });
        Ok(())
    }

    /// One slice per expert, stacked into `[E, rows, cols]`.
    fn stack(&mut self, slices: Vec<Slice>, name: String) -> Result<(), CucaError> {
        self.check(&name)?;
        let first = slices
            .first()
            .ok_or_else(|| cfg_err("experts without tensors"))?;
        let shape = self.shape(&first.tensor)?;
        let mut dims = vec![slices.len(), first.rows];
        dims.extend_from_slice(&shape[1..]);
        self.plans.push(Plan {
            name,
            dims,
            body: Body::Slices(slices),
            ops: Vec::new(),
        });
        Ok(())
    }

    fn n_head(&self) -> Result<usize, CucaError> {
        self.hp.n_head()
    }

    fn n_kv_head(&self) -> Result<usize, CucaError> {
        Ok(self
            .hp
            .uint(&["n_kv_heads", "num_key_value_heads"])
            .unwrap_or(self.n_head()?))
    }

    /// The norm shift of a tensor: `norm_shift` of Gemma 2 and 3, and
    /// Qwen 3.5's `+1` on every norm but the delta-net's own.
    fn norm_ops(&self, hf: &str) -> Vec<Op> {
        let shift = if self.arch.is_qwen35() {
            if hf.ends_with("norm.weight") && !hf.ends_with("linear_attn.norm.weight") {
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
        let canon = canonical(hf);
        let arch = self.arch;
        // Multi-token-prediction head tensors of Qwen 3.5 (`mtp.*`) take the
        // names of blocks after the trunk.
        let canon = if arch.is_qwen35() {
            match mtp_name(&canon, self.n_layer) {
                Some(n) => n,
                None => canon,
            }
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

        // Q/K projections: Llama-family rows are permuted.
        if arch.permutes_qk() {
            let permute = |n_head: usize| vec![Op::PermuteQk { n_head }];
            match rest {
                "self_attn.q_proj.weight" => {
                    let ops = permute(self.n_head()?);
                    return self.direct(hf, blk(gguf::ATTN_Q), ops);
                }
                "self_attn.q_proj.bias" => {
                    let ops = permute(self.n_head()?);
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
        if arch == ConvArch::Qwen35 || arch == ConvArch::Qwen35Moe {
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
        if arch.is_moe()
            && let Some(done) = self.plan_moe(hf, il, rest)?
        {
            return Ok(done);
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
                let (q, k) = (self.n_head()? * hd, self.n_kv_head()? * hd);
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
                ConvArch::Gemma2
                | ConvArch::Gemma3
                | ConvArch::Gemma4
                | ConvArch::Gemma4Assistant
                | ConvArch::Olmo2
                | ConvArch::Qwen35
                | ConvArch::Qwen35Moe => (gguf::POST_ATTENTION_NORM, false),
                _ => (gguf::FFN_NORM, false),
            }),
            "pre_feedforward_layernorm.weight" if arch.is_gemma() => Some((gguf::FFN_NORM, false)),
            "post_feedforward_layernorm.weight" if arch.is_gemma() || arch == ConvArch::Olmo2 => {
                Some((gguf::POST_FFW_NORM, false))
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
        let rows = |offset: usize, hd: usize| {
            tiled.then_some(Op::ReorderV {
                cols: false,
                offset,
                nk,
                nvk,
                hd,
            })
        };
        let blk = |part: &str| gguf::blk(il, part);
        match sub {
            "A_log" => {
                let mut ops: Vec<Op> = rows(0, 1).into_iter().collect();
                ops.push(Op::NegExp);
                self.direct(hf, gguf::blk_bare(il, gguf::SSM_A), ops)
            }
            "dt_bias" => {
                let ops = rows(0, 1).into_iter().collect();
                self.direct(hf, gguf::blk_bias(il, gguf::SSM_DT), ops)
            }
            "conv1d.weight" => {
                let shape = self.shape(hf)?;
                let dims = vec![shape[0], shape[1..].iter().product()];
                let ops = rows(2 * hk * nk, hv).into_iter().collect();
                self.direct_dims(hf, blk(gguf::SSM_CONV1D), dims, ops)
            }
            "in_proj_qkv.weight" => {
                let ops = rows(2 * hk * nk, hv).into_iter().collect();
                self.direct(hf, blk(gguf::ATTN_QKV), ops)
            }
            "in_proj_z.weight" => {
                let ops = rows(0, hv).into_iter().collect();
                self.direct(hf, blk(gguf::ATTN_GATE), ops)
            }
            "in_proj_a.weight" => {
                let ops = rows(0, 1).into_iter().collect();
                self.direct(hf, blk(gguf::SSM_ALPHA), ops)
            }
            "in_proj_b.weight" => {
                let ops = rows(0, 1).into_iter().collect();
                self.direct(hf, blk(gguf::SSM_BETA), ops)
            }
            "out_proj.weight" => {
                let ops = tiled
                    .then_some(Op::ReorderV {
                        cols: true,
                        offset: 0,
                        nk,
                        nvk,
                        hd: hv,
                    })
                    .into_iter()
                    .collect();
                self.direct(hf, blk(gguf::SSM_OUT), ops)
            }
            "norm.weight" => self.direct(hf, blk(gguf::SSM_NORM), Vec::new()),
            _ => Err(self.unmapped(hf)),
        }
    }

    /// Router, experts and shared expert of an MoE layer; `Some(())` when the
    /// tensor was one of them.
    fn plan_moe(&mut self, hf: &str, il: usize, rest: &str) -> Result<Option<()>, CucaError> {
        let arch = self.arch;
        let n_expert =
            self.hp
                .req_uint(&["num_local_experts", "num_experts", "n_routed_experts"])?;
        let router = matches!(
            rest,
            "mlp.gate.weight" | "block_sparse_moe.router.layer.weight"
        );
        if router {
            self.direct(hf, gguf::blk(il, gguf::FFN_GATE_INP), Vec::new())?;
            return Ok(Some(()));
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
                return Ok(Some(()));
            }
        }
        // Fused expert tensors: [E, 2 * ff, hidden] and [E, hidden, ff].
        let fused = |s: &str| {
            s == "mlp.experts.gate_up_proj"
                || s == "mlp.experts.gate_up_proj.weight"
                || s == "block_sparse_moe.input_linear.weight"
        };
        let down = |s: &str| {
            s == "mlp.experts.down_proj"
                || s == "mlp.experts.down_proj.weight"
                || s == "block_sparse_moe.output_linear.weight"
        };
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
                self.stack_with_dims(slices, gguf::blk(il, part), vec![n_expert, ff, shape[2]])?;
            }
            return Ok(Some(()));
        }
        if down(rest) {
            self.direct(hf, gguf::blk(il, gguf::FFN_DOWN_EXPS), Vec::new())?;
            return Ok(Some(()));
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
                return Ok(Some(()));
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
            return Ok(Some(()));
        }
        Ok(None)
    }

    fn stack_with_dims(
        &mut self,
        slices: Vec<Slice>,
        name: String,
        dims: Vec<usize>,
    ) -> Result<(), CucaError> {
        self.check(&name)?;
        self.plans.push(Plan {
            name,
            dims,
            body: Body::Slices(slices),
            ops: Vec::new(),
        });
        Ok(())
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
        planner.check(e.name)?;
        planner.plans.push(Plan {
            name: e.name.to_string(),
            dims: vec![e.data.len()],
            body: Body::Generated(e.data),
            ops: Vec::new(),
        });
    }
    for hf in shards.names() {
        if skipped(hf) {
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
    fn gemma2_shifts_every_norm_and_renames_the_feed_forward_norms() {
        let (s, dir) = checkpoint(
            "t-gemma2",
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
        let plans = plan(ConvArch::Gemma2, &h, &s, Vec::new(), None).unwrap();
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
    fn granite_moe_splits_input_linear_and_stacks_nothing_else() {
        let (s, dir) = checkpoint(
            "t-gmoe",
            &[
                (
                    "model.layers.0.block_sparse_moe.router.layer.weight",
                    &[3, 4],
                ),
                (
                    "model.layers.0.block_sparse_moe.input_linear.weight",
                    &[3, 10, 4],
                ),
                (
                    "model.layers.0.block_sparse_moe.output_linear.weight",
                    &[3, 4, 5],
                ),
            ],
        );
        let h = hp(serde_json::json!({"num_hidden_layers": 1, "num_local_experts": 3}));
        let plans = plan(ConvArch::GraniteMoe, &h, &s, Vec::new(), None).unwrap();
        let gate = by_name(&plans, "blk.0.ffn_gate_exps.weight");
        let up = by_name(&plans, "blk.0.ffn_up_exps.weight");
        assert_eq!(gate.dims, [3, 5, 4]);
        assert_eq!(up.dims, [3, 5, 4]);
        let g = values(&s, gate);
        let u = values(&s, up);
        // Expert 1: rows 10..20 of the flat [30, 4] tensor; gate = first 5 rows.
        assert_eq!(g[20], 40.0);
        assert_eq!(u[20], 60.0);
        assert_eq!(
            by_name(&plans, "blk.0.ffn_down_exps.weight").dims,
            [3, 4, 5]
        );
        assert_eq!(by_name(&plans, "blk.0.ffn_gate_inp.weight").dims, [3, 4]);
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
        encode(Dtype::Q8_0, all, cols, &mut want).unwrap();
        assert_eq!(got, want);
        let _ = std::fs::remove_dir_all(dir);
    }
}
