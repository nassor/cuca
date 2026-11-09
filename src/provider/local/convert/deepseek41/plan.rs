//! The GGUF tensors of a DeepSeek-V4.1 conversion: names, shapes, storage types
//! and where each one's values come from (llama.cpp's `deepseek41` and `dflash`
//! layouts).
//!
//! The plan is made from tensor names, dtypes and shapes only (an [`Index`]),
//! so it can be made over the real checkpoint's headers, or over a listing of
//! them, without reading a weight.
//!
//! | checkpoint | GGUF | stored as |
//! |---|---|---|
//! | `embed.weight`, `head.weight` | `token_embd.weight`, `output.weight` | matrix |
//! | `norm.weight` | `output_norm.weight` | `f32` |
//! | FP8 `attn.{wq_a,wq_b,wkv,wo_a,wo_b}`, shared experts, `indexer.wq_b`, `engram.wkv` | `blk.N.*` | matrix |
//! | `compressor.{wkv,wgate}`, `indexer.{wk,weights_proj}` | `blk.N.*` | `f16` (they feed a softmax-pooled latent) |
//! | norms, `hc_*`, `attn_sink`, router, `gate.bias` | `blk.N.*` | `f32` |
//! | `engram.{q,k}_weight` | `blk.N.engram_gate.weight` (`q * k`) | `f32` |
//! | `ffn.experts.E.w{1,3,2}` (FP4) | `ffn_{gate,up,down}_exps`, stacked | MXFP4 (`f16` at `quant=f16`) |
//! | `engram.embed` of every Engram layer | `per_layer_token_embd.weight`, rows concatenated | [`EngramQuant`] |
//!
//! A matrix is stored at the [`WeightQuant`] precision, except that the token
//! embedding and the output head (and the drafter's markov tables) stay at
//! Q8_0 under Q4_0. `ffn.gate.bias_vl`, the vision tower, the aligner and the
//! image tokens are not converted.

use std::collections::HashSet;

use super::super::cfg_err;
use super::super::gguf_writer::{Dtype, TensorSpec};
use super::super::safetensors::{BLOCK, Shards, StDtype};
use super::hparams::Config;
use crate::error::CucaError;
use crate::provider::local::{EngramQuant, WeightQuant};

/// A source of tensor names, dtypes and shapes: the checkpoint's shards, or a
/// listing of their headers.
pub(crate) trait Index {
    /// The dtype and shape of `name`.
    fn tensor(&self, name: &str) -> Option<(StDtype, Vec<usize>)>;
    /// Every tensor name.
    fn names(&self) -> Vec<String>;
}

impl Index for Shards {
    fn tensor(&self, name: &str) -> Option<(StDtype, Vec<usize>)> {
        Shards::tensor(self, name).map(|m| (m.dtype, m.shape.clone()))
    }

    fn names(&self) -> Vec<String> {
        Shards::names(self).into_iter().map(String::from).collect()
    }
}

/// Which file of the conversion a tensor goes to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Part {
    /// Everything but the Engram table.
    Backbone,
    /// `per_layer_token_embd.weight`, written to a file of its own.
    Engram,
}

/// One Engram table: its rows and the scales beside them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Table {
    pub(crate) weight: String,
    pub(crate) scale: String,
    pub(crate) rows: usize,
}

/// Where a GGUF tensor's values come from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Source {
    /// A float or block-scaled FP8 checkpoint matrix (or vector).
    Matrix { tensor: String },
    /// The elementwise product of two float tensors (`engram_gate`).
    Product { a: String, b: String },
    /// The stack of every expert's `proj` (`w1`, `w2` or `w3`) of a block,
    /// `{prefix}.ffn.experts.E.{proj}.{weight,scale}` for `E` in `0..n`.
    Experts {
        prefix: String,
        proj: &'static str,
        n: usize,
    },
    /// The tables of the Engram layers, rows concatenated.
    Engram { tables: Vec<Table>, dim: usize },
}

/// One GGUF tensor to write.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Item {
    pub(crate) name: String,
    /// Row-major shape of the result.
    pub(crate) dims: Vec<usize>,
    pub(crate) dtype: Dtype,
    pub(crate) source: Source,
    pub(crate) part: Part,
}

impl Item {
    pub(crate) fn spec(&self) -> TensorSpec {
        TensorSpec {
            name: self.name.clone(),
            dims: self.dims.clone(),
            dtype: self.dtype,
        }
    }

    /// Bytes of the tensor's data.
    pub(crate) fn bytes(&self) -> u64 {
        self.spec().byte_len() as u64
    }
}

/// The tensors of one conversion, in the order they are written.
#[derive(Debug, Clone, Default)]
pub(crate) struct Plan {
    pub(crate) items: Vec<Item>,
}

impl Plan {
    /// The tensors that go to `part`.
    pub(crate) fn part(&self, part: Part) -> Vec<&Item> {
        self.items.iter().filter(|i| i.part == part).collect()
    }

    /// Bytes of the data of the tensors of `part`.
    pub(crate) fn bytes(&self, part: Part) -> u64 {
        self.part(part).iter().map(|i| i.bytes()).sum()
    }
}

/// Which model a plan is for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Role {
    /// The backbone: `deepseek41`.
    Main,
    /// The DSpark drafter stages: `dflash`.
    Drafter,
}

/// How a transformer-block tensor is stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    /// At the conversion's matrix precision.
    Matrix,
    /// Half precision whatever the precision.
    Half,
    /// `f32`.
    Float,
}

/// Block tensors, `(checkpoint suffix, GGUF suffix, kind)`. A block holds
/// the ones its role needs (the compressor and the indexer only where the layer
/// owns one).
const BLOCK_TENSORS: &[(&str, &str, Kind)] = &[
    ("attn_norm.weight", "attn_norm.weight", Kind::Float),
    ("ffn_norm.weight", "ffn_norm.weight", Kind::Float),
    ("attn.attn_sink", "attn_sinks.weight", Kind::Float),
    ("attn.wq_a.weight", "attn_q_a.weight", Kind::Matrix),
    ("attn.q_norm.weight", "attn_q_a_norm.weight", Kind::Float),
    ("attn.wq_b.weight", "attn_q_b.weight", Kind::Matrix),
    ("attn.wkv.weight", "attn_kv.weight", Kind::Matrix),
    ("attn.kv_norm.weight", "attn_kv_a_norm.weight", Kind::Float),
    ("attn.wo_a.weight", "attn_output_a.weight", Kind::Matrix),
    ("attn.wo_b.weight", "attn_output_b.weight", Kind::Matrix),
    ("hc_attn_fn", "hc_attn_fn.weight", Kind::Float),
    ("hc_attn_base", "hc_attn_base.weight", Kind::Float),
    ("hc_attn_scale", "hc_attn_scale.weight", Kind::Float),
    ("hc_ffn_fn", "hc_ffn_fn.weight", Kind::Float),
    ("hc_ffn_base", "hc_ffn_base.weight", Kind::Float),
    ("hc_ffn_scale", "hc_ffn_scale.weight", Kind::Float),
    (
        "attn.compressor.wkv.weight",
        "attn_compressor_kv.weight",
        Kind::Half,
    ),
    (
        "attn.compressor.wgate.weight",
        "attn_compressor_gate.weight",
        Kind::Half,
    ),
    (
        "attn.compressor.norm.weight",
        "attn_compressor_norm.weight",
        Kind::Float,
    ),
    (
        "attn.indexer.wk.weight",
        "indexer.attn_k.weight",
        Kind::Half,
    ),
    (
        "attn.indexer.k_norm.weight",
        "indexer.k_norm.weight",
        Kind::Float,
    ),
    (
        "attn.indexer.wq_b.weight",
        "indexer.attn_q_b.weight",
        Kind::Matrix,
    ),
    (
        "attn.indexer.weights_proj.weight",
        "indexer.proj.weight",
        Kind::Half,
    ),
    ("ffn.gate.weight", "ffn_gate_inp.weight", Kind::Float),
    ("ffn.gate.bias", "exp_probs_b.bias", Kind::Float),
    (
        "ffn.shared_experts.w1.weight",
        "ffn_gate_shexp.weight",
        Kind::Matrix,
    ),
    (
        "ffn.shared_experts.w3.weight",
        "ffn_up_shexp.weight",
        Kind::Matrix,
    ),
    (
        "ffn.shared_experts.w2.weight",
        "ffn_down_shexp.weight",
        Kind::Matrix,
    ),
    ("engram.wkv.weight", "engram_kv.weight", Kind::Matrix),
];

/// Block tensors the plan reads nothing of: `ffn.gate.bias_vl` routes image
/// tokens only (llama.cpp marks it not required).
const IGNORED_SUFFIXES: &[&str] = &["ffn.gate.bias_vl"];

/// The stacked expert projections: `(checkpoint, GGUF suffix)`.
const EXPERT_STACKS: [(&str, &str); 3] = [
    ("w1", "ffn_gate_exps.weight"),
    ("w3", "ffn_up_exps.weight"),
    ("w2", "ffn_down_exps.weight"),
];

/// The storage type of a matrix at `quant`: the token embedding, the output head
/// and the markov tables stay at Q8_0 under Q4_0; a row that does not fill whole
/// blocks falls back to `f32`.
fn matrix_dtype(quant: WeightQuant, gguf_name: &str, cols: usize) -> Dtype {
    let dtype = match (quant, gguf_name) {
        (WeightQuant::F16, _) => Dtype::F16,
        (WeightQuant::Q8_0, _) => Dtype::Q8_0,
        (
            WeightQuant::Q4_0,
            "token_embd.weight" | "output.weight" | "markov_w1.weight" | "markov_w2.weight",
        ) => Dtype::Q8_0,
        (WeightQuant::Q4_0, _) => Dtype::Q4_0,
    };
    if cols.is_multiple_of(dtype.block_size()) {
        dtype
    } else {
        Dtype::F32
    }
}

fn engram_dtype(quant: EngramQuant) -> Dtype {
    match quant {
        EngramQuant::Q8_0 => Dtype::Q8_0,
        EngramQuant::Q4_0 => Dtype::Q4_0,
        EngramQuant::Bf16 => Dtype::Bf16,
    }
}

/// Whether `dtype` is a float the matrix reader decodes.
fn is_float(dtype: StDtype) -> bool {
    matches!(
        dtype,
        StDtype::F32 | StDtype::F16 | StDtype::Bf16 | StDtype::F8E4m3
    )
}

/// Planning state: the checkpoint's index, the tensors consumed so far and the
/// items made.
struct Planner<'a> {
    index: &'a dyn Index,
    cfg: &'a Config,
    quant: WeightQuant,
    items: Vec<Item>,
    used: HashSet<String>,
}

impl Planner<'_> {
    fn find(&mut self, name: &str) -> Result<(StDtype, Vec<usize>), CucaError> {
        let found = self
            .index
            .tensor(name)
            .ok_or_else(|| cfg_err(format!("checkpoint has no tensor `{name}`")))?;
        self.used.insert(name.to_string());
        Ok(found)
    }

    /// A matrix or vector, FP8 with its scale grid or float.
    fn matrix(
        &mut self,
        tensor: &str,
        name: String,
        kind: Kind,
        part: Part,
    ) -> Result<(), CucaError> {
        let (dtype, mut dims) = self.find(tensor)?;
        if !is_float(dtype) {
            return Err(cfg_err(format!(
                "tensor `{tensor}` has dtype {}, which the converter does not read as a matrix",
                dtype.name()
            )));
        }
        if dtype == StDtype::F8E4m3 {
            let scale = tensor
                .strip_suffix(".weight")
                .map(|m| format!("{m}.scale"))
                .ok_or_else(|| cfg_err(format!("FP8 tensor `{tensor}` is not a `.weight`")))?;
            let (scale_dtype, scale_dims) = self.find(&scale)?;
            let want = [dims[0].div_ceil(BLOCK), dims[1].div_ceil(BLOCK)];
            if dims.len() != 2 || scale_dtype != StDtype::F8E8m0 || scale_dims != want {
                return Err(cfg_err(format!(
                    "`{scale}` is {} {scale_dims:?}, expected F8_E8M0 {want:?}: one exponent per {BLOCK}x{BLOCK} block of `{tensor}` {dims:?}",
                    scale_dtype.name()
                )));
            }
        }
        // ggml-org's drafter stores the confidence head `(1, n)` as a vector.
        if name == "conf_proj.weight" && dims.len() == 2 && dims[0] == 1 {
            dims.remove(0);
        }
        let cols = dims.last().copied().unwrap_or(1);
        let out = match kind {
            Kind::Float => Dtype::F32,
            Kind::Half => Dtype::F16,
            Kind::Matrix if dims.len() < 2 => Dtype::F32,
            Kind::Matrix => matrix_dtype(self.quant, &name, cols),
        };
        self.items.push(Item {
            name,
            dims,
            dtype: out,
            source: Source::Matrix {
                tensor: tensor.to_string(),
            },
            part,
        });
        Ok(())
    }

    /// Whether the checkpoint has `name`.
    fn has(&self, name: &str) -> bool {
        self.index.tensor(name).is_some()
    }

    /// The three stacks of a block's routed experts.
    fn experts(&mut self, prefix: &str, gguf_prefix: &str, n: usize) -> Result<(), CucaError> {
        for (proj, suffix) in EXPERT_STACKS {
            let mut shape: Option<(Vec<usize>, Vec<usize>)> = None;
            for e in 0..n {
                let weight = format!("{prefix}.ffn.experts.{e}.{proj}.weight");
                let scale = format!("{prefix}.ffn.experts.{e}.{proj}.scale");
                let (wd, wdims) = self.find(&weight)?;
                let (sd, sdims) = self.find(&scale)?;
                if wd != StDtype::I8 || sd != StDtype::F8E8m0 {
                    return Err(cfg_err(format!(
                        "expert `{weight}` is {} with {} scales, expected packed FP4 (I8) with F8_E8M0",
                        wd.name(),
                        sd.name()
                    )));
                }
                match &shape {
                    None => shape = Some((wdims, sdims)),
                    Some((w, s)) if *w == wdims && *s == sdims => {}
                    Some(_) => {
                        return Err(cfg_err(format!(
                            "expert `{weight}` has another shape than expert 0"
                        )));
                    }
                }
            }
            let Some((wdims, sdims)) = shape else {
                return Err(cfg_err(format!("`{prefix}` has no routed experts")));
            };
            let (rows, cols) = (wdims[0], wdims[1] * 2);
            if wdims.len() != 2 || sdims != [rows, cols / 32] || !cols.is_multiple_of(32) {
                return Err(cfg_err(format!(
                    "experts `{prefix}.ffn.experts.E.{proj}`: weight {wdims:?} and scale {sdims:?} are not packed FP4 with one scale per 32 columns"
                )));
            }
            let dtype = match self.quant {
                WeightQuant::F16 => Dtype::F16,
                _ => Dtype::Mxfp4,
            };
            self.items.push(Item {
                name: format!("{gguf_prefix}.{suffix}"),
                dims: vec![n, rows, cols],
                dtype,
                source: Source::Experts {
                    prefix: prefix.to_string(),
                    proj,
                    n,
                },
                part: Part::Backbone,
            });
        }
        Ok(())
    }

    /// The tensors of one transformer block named `prefix` in the checkpoint
    /// (`layers.N` or `mtp.N`) and `blk.M` in the GGUF.
    fn block(
        &mut self,
        prefix: &str,
        gguf: &str,
        n_expert: usize,
        engram: bool,
    ) -> Result<(), CucaError> {
        for &(suffix, gguf_suffix, kind) in BLOCK_TENSORS {
            if suffix.starts_with("engram.") {
                continue;
            }
            let tensor = format!("{prefix}.{suffix}");
            if !self.has(&tensor) {
                // The compressor and the indexer exist on the layers that own
                // them; everything else a block needs.
                if suffix.starts_with("attn.compressor.") || suffix.starts_with("attn.indexer.") {
                    continue;
                }
            }
            self.matrix(
                &tensor,
                format!("{gguf}.{gguf_suffix}"),
                kind,
                Part::Backbone,
            )?;
        }
        self.experts(prefix, gguf, n_expert)?;
        if engram && self.has(&format!("{prefix}.engram.wkv.weight")) {
            self.matrix(
                &format!("{prefix}.engram.wkv.weight"),
                format!("{gguf}.engram_kv.weight"),
                Kind::Matrix,
                Part::Backbone,
            )?;
            let (q, k) = (
                format!("{prefix}.engram.q_weight"),
                format!("{prefix}.engram.k_weight"),
            );
            let (qd, qdims) = self.find(&q)?;
            let (kd, kdims) = self.find(&k)?;
            if !is_float(qd) || !is_float(kd) || qdims != kdims {
                return Err(cfg_err(format!(
                    "`{q}` {qdims:?} and `{k}` {kdims:?} cannot be multiplied elementwise"
                )));
            }
            self.items.push(Item {
                name: format!("{gguf}.engram_gate.weight"),
                dims: qdims,
                dtype: Dtype::F32,
                source: Source::Product { a: q, b: k },
                part: Part::Backbone,
            });
        }
        Ok(())
    }

    /// The tensors left under `prefix.` that no rule consumed.
    fn leftovers(&self, prefix: &str) -> Vec<String> {
        let dotted = format!("{prefix}.");
        let mut left: Vec<String> = self
            .index
            .names()
            .into_iter()
            .filter(|n| n.starts_with(&dotted) && !self.used.contains(n))
            .filter(|n| {
                let rest = &n[dotted.len()..];
                !IGNORED_SUFFIXES.contains(&rest)
            })
            .collect();
        left.sort_unstable();
        left
    }
}

/// Plan the conversion of the checkpoint `index` describes for `role`.
///
/// # Errors
///
/// [`CucaError::Config`] naming a missing tensor, a tensor of a type or shape the
/// converter does not read, or a checkpoint tensor no rule accounts for.
pub(super) fn plan(
    index: &dyn Index,
    cfg: &Config,
    quant: WeightQuant,
    engram_quant: EngramQuant,
    role: Role,
) -> Result<Plan, CucaError> {
    let mut p = Planner {
        index,
        cfg,
        quant,
        items: Vec::new(),
        used: HashSet::new(),
    };
    match role {
        Role::Main => main(&mut p, engram_quant)?,
        Role::Drafter => drafter(&mut p)?,
    }
    Ok(Plan { items: p.items })
}

fn main(p: &mut Planner<'_>, engram_quant: EngramQuant) -> Result<(), CucaError> {
    let cfg = p.cfg;
    p.matrix(
        "embed.weight",
        "token_embd.weight".into(),
        Kind::Matrix,
        Part::Backbone,
    )?;
    for il in 0..cfg.n_layer {
        let engram = cfg.engram.as_ref().is_some_and(|e| e.layers.contains(&il));
        p.block(
            &format!("layers.{il}"),
            &format!("blk.{il}"),
            cfg.n_expert,
            engram,
        )?;
        if engram {
            // The tables are read into the model-wide `per_layer_token_embd`
            // below; they count as consumed here.
            for suffix in ["weight", "scale"] {
                p.find(&format!("layers.{il}.engram.embed.{suffix}"))?;
            }
        }
        let left = p.leftovers(&format!("layers.{il}"));
        if let Some(name) = left.first() {
            return Err(cfg_err(format!(
                "checkpoint tensor `{name}` has no GGUF counterpart ({} such tensors in block {il})",
                left.len()
            )));
        }
    }
    p.matrix(
        "norm.weight",
        "output_norm.weight".into(),
        Kind::Float,
        Part::Backbone,
    )?;
    p.matrix(
        "head.weight",
        "output.weight".into(),
        Kind::Matrix,
        Part::Backbone,
    )?;
    if let Some(e) = &cfg.engram {
        let mut tables = Vec::new();
        for (&layer, &rows) in e.layers.iter().zip(&e.rows) {
            let (weight, scale) = (
                format!("layers.{layer}.engram.embed.weight"),
                format!("layers.{layer}.engram.embed.scale"),
            );
            let (wd, wdims) = p.find(&weight)?;
            let (sd, sdims) = p.find(&scale)?;
            let rows = rows as usize;
            if wd != StDtype::F8E4m3
                || sd != StDtype::F8E8m0
                || wdims != [rows, e.dim]
                || sdims != [rows, e.dim / 32]
            {
                return Err(cfg_err(format!(
                    "`{weight}` is {} {wdims:?} with {} scales {sdims:?}, expected F8_E4M3 [{rows}, {}] with one F8_E8M0 scale per 32 columns",
                    wd.name(),
                    sd.name(),
                    e.dim
                )));
            }
            tables.push(Table {
                weight,
                scale,
                rows,
            });
        }
        let total: usize = tables.iter().map(|t| t.rows).sum();
        p.items.push(Item {
            name: "per_layer_token_embd.weight".into(),
            dims: vec![total, e.dim],
            dtype: engram_dtype(engram_quant),
            source: Source::Engram { tables, dim: e.dim },
            part: Part::Engram,
        });
    }
    Ok(())
}

/// The drafter's roots, `(checkpoint suffix after `mtp.N.`, GGUF name, kind,
/// stage)`: `Some(n)` says which stage holds it (`main_*` on the first, `norm`
/// and the heads on the last), `None` any.
const DRAFTER_ROOTS: &[(&str, &str, Kind, Stage)] = &[
    ("main_proj.weight", "fc.weight", Kind::Matrix, Stage::First),
    (
        "main_norm.weight",
        "enc.output_norm.weight",
        Kind::Float,
        Stage::First,
    ),
    (
        "norm.weight",
        "output_norm.weight",
        Kind::Float,
        Stage::Last,
    ),
    (
        "markov_head.embed.weight",
        "markov_w1.weight",
        Kind::Matrix,
        Stage::Last,
    ),
    (
        "markov_head.head.weight",
        "markov_w2.weight",
        Kind::Matrix,
        Stage::Last,
    ),
    (
        "confidence_head.proj.weight",
        "conf_proj.weight",
        Kind::Float,
        Stage::Last,
    ),
];

#[derive(Debug, Clone, Copy)]
enum Stage {
    First,
    Last,
}

fn drafter(p: &mut Planner<'_>) -> Result<(), CucaError> {
    let cfg = p.cfg;
    let Some(d) = &cfg.dspark else {
        return Err(cfg_err(
            "the checkpoint has no DSpark drafter (num_nextn_predict_layers and dspark_* in config.json)",
        ));
    };
    let n = cfg.n_mtp;
    for stage in 0..n {
        p.block(
            &format!("mtp.{stage}"),
            &format!("blk.{stage}"),
            d.n_expert,
            false,
        )?;
        for &(suffix, gguf, kind, at) in DRAFTER_ROOTS {
            let wanted = match at {
                Stage::First => stage == 0,
                Stage::Last => stage + 1 == n,
            };
            let tensor = format!("mtp.{stage}.{suffix}");
            if wanted || p.has(&tensor) {
                p.matrix(&tensor, gguf.into(), kind, Part::Backbone)?;
            }
        }
        let left = p.leftovers(&format!("mtp.{stage}"));
        if let Some(name) = left.first() {
            return Err(cfg_err(format!(
                "checkpoint tensor `{name}` has no GGUF counterpart ({} such tensors in stage {stage})",
                left.len()
            )));
        }
    }
    // The drafter's file holds no token embedding or output head: it shares
    // the target's.
    Ok(())
}

#[cfg(test)]
pub(super) mod tests {
    use std::collections::BTreeMap;
    use std::path::{Path, PathBuf};

    use super::*;
    use crate::provider::local::convert::config::Hp;

    /// An in-memory listing of a checkpoint's headers.
    #[derive(Default)]
    pub(in crate::provider::local::convert) struct Listing(
        pub(in crate::provider::local::convert) BTreeMap<String, (StDtype, Vec<usize>)>,
    );

    impl Index for Listing {
        fn tensor(&self, name: &str) -> Option<(StDtype, Vec<usize>)> {
            self.0.get(name).cloned()
        }

        fn names(&self) -> Vec<String> {
            self.0.keys().cloned().collect()
        }
    }

    fn st_dtype(name: &str) -> StDtype {
        match name {
            "F32" => StDtype::F32,
            "F16" => StDtype::F16,
            "BF16" => StDtype::Bf16,
            "F8_E4M3" => StDtype::F8E4m3,
            "F8_E8M0" => StDtype::F8E8m0,
            "I8" => StDtype::I8,
            other => panic!("unexpected dtype {other}"),
        }
    }

    fn research(file: &str) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join(".omp")
            .join(file)
    }

    #[test]
    fn a_matrix_falls_back_to_f32_when_rows_do_not_fill_blocks() {
        assert_eq!(matrix_dtype(WeightQuant::Q8_0, "x", 64), Dtype::Q8_0);
        assert_eq!(matrix_dtype(WeightQuant::Q8_0, "x", 48), Dtype::F32);
        assert_eq!(matrix_dtype(WeightQuant::F16, "x", 48), Dtype::F16);
        assert_eq!(matrix_dtype(WeightQuant::Q4_0, "x", 64), Dtype::Q4_0);
        assert_eq!(
            matrix_dtype(WeightQuant::Q4_0, "token_embd.weight", 64),
            Dtype::Q8_0
        );
        assert_eq!(
            matrix_dtype(WeightQuant::Q4_0, "markov_w2.weight", 256),
            Dtype::Q8_0
        );
    }

    /// The plan over the real V4.1-Flash-1017 headers (`tensors.json`, read by
    /// HTTP range from the 48 shards) and config: the tensor table and the
    /// part sizes. Without the research files the test prints `SKIP:`.
    #[test]
    fn the_plan_of_the_real_checkpoint_matches_the_design_table_and_sizes() {
        let (listing, config) = (
            research("ds-research/tensors.json"),
            research("local-models/ref/deepseek-v41/config.json"),
        );
        if !listing.is_file() || !config.is_file() {
            crate::provider::local::tokenizer::golden::skip("DeepSeek-V4.1 research files in .omp");
            return;
        }
        let json: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(listing).unwrap()).unwrap();
        let index = Listing(
            json.as_object()
                .unwrap()
                .iter()
                .map(|(name, v)| {
                    let shape = v[1]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|d| d.as_u64().unwrap() as usize);
                    (
                        name.clone(),
                        (st_dtype(v[0].as_str().unwrap()), shape.collect()),
                    )
                })
                .collect(),
        );
        let hp = Hp::new(&serde_json::from_str(&std::fs::read_to_string(config).unwrap()).unwrap())
            .unwrap();
        let cfg = Config::read(&hp).unwrap();

        let gb = |bytes: u64| bytes as f64 / 1e9;
        let within = |got: f64, want: f64| (got - want).abs() / want < 0.01;
        let main = plan(
            &index,
            &cfg,
            WeightQuant::Q8_0,
            EngramQuant::Q8_0,
            Role::Main,
        )
        .unwrap();
        let backbone = main.bytes(Part::Backbone);
        let table = main.bytes(Part::Engram);
        let experts: u64 = main
            .items
            .iter()
            .filter(|i| i.dtype == Dtype::Mxfp4)
            .map(Item::bytes)
            .sum();
        eprintln!(
            "PLAN|{} tensors|backbone {:.2} GB|experts {:.2} GB|engram q8_0 {:.2} GB",
            main.items.len(),
            gb(backbone),
            gb(experts),
            gb(table)
        );
        // 40 x (3 experts stacks) of MXFP4 and every other tensor of section 4.2.
        assert_eq!(
            main.items
                .iter()
                .filter(|i| i.dtype == Dtype::Mxfp4)
                .count(),
            120
        );
        assert!(within(gb(experts), 288.78), "experts {}", gb(experts));
        assert!(within(gb(backbone), 297.94), "backbone {}", gb(backbone));
        assert!(within(gb(table), 208.9), "engram {}", gb(table));
        // llama.cpp's 1,043 tensors, less the 40 `exp_probs_b_vl` it adds.
        assert_eq!(main.items.len(), 1043 - 40);
        let by_name: BTreeMap<&str, &Item> =
            main.items.iter().map(|i| (i.name.as_str(), i)).collect();
        let dims = |name: &str| by_name[name].dims.clone();
        assert_eq!(dims("token_embd.weight"), [129_280, 5120]);
        assert_eq!(dims("blk.0.attn_output_a.weight"), [8192, 4096]);
        assert_eq!(dims("blk.7.ffn_gate_exps.weight"), [384, 2304, 5120]);
        assert_eq!(dims("blk.7.ffn_down_exps.weight"), [384, 5120, 2304]);
        assert_eq!(dims("blk.1.engram_gate.weight"), [4, 5120]);
        assert_eq!(dims("blk.14.engram_kv.weight"), [25_600, 6144]);
        assert_eq!(dims("per_layer_token_embd.weight"), [768_022_850, 256]);
        assert_eq!(dims("blk.5.hc_attn_fn.weight"), [24, 20_480]);
        assert_eq!(by_name["blk.5.hc_attn_fn.weight"].dtype, Dtype::F32);
        assert_eq!(by_name["blk.2.attn_compressor_kv.weight"].dtype, Dtype::F16);
        assert!(!by_name.contains_key("blk.3.attn_compressor_kv.weight"));
        assert!(by_name.contains_key("blk.8.attn_compressor_gate.weight"));
        assert!(!by_name.contains_key("blk.20.attn_compressor_gate.weight"));
        assert!(by_name.contains_key("blk.36.indexer.attn_q_b.weight"));
        assert!(!by_name.contains_key("blk.36.indexer.attn_k.weight"));
        assert_eq!(by_name["blk.0.ffn_gate_inp.weight"].dtype, Dtype::F32);
        assert!(by_name.keys().all(|n| !n.contains("bias_vl")));

        // Other precisions: the table follows `engram_quant`, the experts stay MXFP4.
        let small = plan(
            &index,
            &cfg,
            WeightQuant::Q4_0,
            EngramQuant::Q4_0,
            Role::Main,
        )
        .unwrap();
        assert!(
            within(gb(small.bytes(Part::Engram)), 110.6),
            "{}",
            gb(small.bytes(Part::Engram))
        );
        let bf16 = plan(
            &index,
            &cfg,
            WeightQuant::Q8_0,
            EngramQuant::Bf16,
            Role::Main,
        )
        .unwrap();
        assert!(within(gb(bf16.bytes(Part::Engram)), 393.2));

        let drafter = plan(
            &index,
            &cfg,
            WeightQuant::Q8_0,
            EngramQuant::Q8_0,
            Role::Drafter,
        )
        .unwrap();
        let drafter_bytes = drafter.bytes(Part::Backbone);
        eprintln!(
            "PLAN|drafter {} tensors|{:.2} GB",
            drafter.items.len(),
            gb(drafter_bytes)
        );
        assert!(within(gb(drafter_bytes), 7.9), "{}", gb(drafter_bytes));
        assert_eq!(drafter.items.len(), 78);
        let names: Vec<&str> = drafter.items.iter().map(|i| i.name.as_str()).collect();
        for root in [
            "fc.weight",
            "enc.output_norm.weight",
            "output_norm.weight",
            "markov_w1.weight",
            "markov_w2.weight",
            "conf_proj.weight",
        ] {
            assert!(names.contains(&root), "{root}");
        }
    }
}
