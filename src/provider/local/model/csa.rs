//! Specifications of the mechanisms a latent multi-query stack with
//! hyper-connections adds to the generic transformer, and the readers that
//! turn `{arch}.*` metadata into them:
//!
//! - hyper-connections ([`HyperSpec`]): the residual stream is `streams`
//!   parallel copies of the hidden state, mixed by coefficients predicted from
//!   the stream itself;
//! - compressed sparse attention ([`CsaSpec`], [`CsaGlobals`]): every layer
//!   attends to a sliding window of its own keys and, when its compression
//!   ratio is not 0, to up to `top_k` rows of a compressed cache chosen by an
//!   indexer; compressed rows and the indexer's picks are shared between
//!   layers (the *source* layers compute them, the layers after a source read
//!   them);
//! - per-layer hashed table lookups ([`EngramSpec`]);
//! - the mixture-of-experts router of such a model ([`MoeExt`]): the scoring
//!   function, a selection-only bias and the SwiGLU clamps.
//!
//! The numbers come from llama.cpp's merged GGUF layout (see the converter,
//! `convert::deepseek41`); the validations follow its loader.

use super::arch::Arch;
use super::params::{Keys, Metadata, value_usize};
use super::rope::{RopeKind, RopeSpec, Yarn};
use super::{err, meta_err};
use crate::error::CucaError;
use crate::provider::local::gguf::Value;

/// Hyper-connections: `streams` copies of the hidden state per token.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct HyperSpec {
    pub(crate) streams: usize,
    /// Sinkhorn normalisation passes of the stream-mixing matrix.
    pub(crate) sinkhorn_iters: usize,
    /// The epsilon of the coefficients (`hyper_connection.epsilon`).
    pub(crate) eps: f32,
}

/// What a layer does with the candidate pool the candidate source layer
/// publishes (hierarchical indexing).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Candidates {
    /// The layer does not take part (it scores every row, or reuses picks).
    None,
    /// The layer scores every row and publishes the candidate blocks.
    Source,
    /// The layer scores only the rows of the published candidate blocks.
    Consumer,
}

/// The compressed-attention role of one layer with a non-zero ratio.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CsaSpec {
    /// Tokens per compressed row (1 or 2).
    pub(crate) ratio: usize,
    /// The layer whose compressed cache this layer reads (itself when it
    /// owns one).
    pub(crate) kv_source: usize,
    /// The layer whose top-k picks this layer attends with (itself when it
    /// scores).
    pub(crate) index_source: usize,
    /// The layer computes the compressed rows (and the index keys) it reads.
    pub(crate) owns_kv: bool,
    /// The layer scores rows with its own indexer queries.
    pub(crate) scores: bool,
    pub(crate) candidates: Candidates,
    /// `f32` values of one compressed row of the layer's cache: the latent
    /// followed by the index key (owners only; 0 for the readers).
    pub(crate) row_width: usize,
}

/// The indexer shared by the layers of a model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct IndexerSpec {
    pub(crate) heads: usize,
    pub(crate) dim: usize,
    pub(crate) top_k: usize,
    /// `(source layer, blocks kept, rows per block)` of the hierarchical
    /// candidate pool.
    pub(crate) candidates: Option<(usize, usize, usize)>,
}

/// What the attention layers of the model share.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CsaGlobals {
    pub(crate) q_lora: usize,
    /// Groups of the block-diagonal output projection and the rank each one
    /// produces.
    pub(crate) out_groups: usize,
    pub(crate) out_rank: usize,
    pub(crate) indexer: IndexerSpec,
}

/// How the router scores the experts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Gating {
    /// llama.cpp `LLM_GATING_SOFTMAX` (1).
    Softmax,
    /// `LLM_GATING_SIGMOID` (2).
    Sigmoid,
    /// `LLM_GATING_SQRTSOFTPLUS` (4): `sqrt(softplus(logit))`.
    SqrtSoftplus,
}

/// The router of a mixture-of-experts model beyond a plain softmax top-k.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct MoeExt {
    pub(crate) gating: Gating,
    /// Per-layer SwiGLU clamp of the routed experts (`swiglu_clamp_exp`): the
    /// up branch is clamped to `[-c, c]`, the gate branch from above.
    pub(crate) clamp: Vec<f32>,
    /// The same for the shared expert (`swiglu_clamp_shexp`).
    pub(crate) clamp_shared: Vec<f32>,
}

/// Hashed per-layer table lookups: for each token, `heads` rows of every
/// n-gram order `2..=ngram` of the token and its predecessors are fetched from
/// a very large table and gate a residual add.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct EngramSpec {
    /// The layers that run the lookup before their block, ascending.
    pub(crate) layers: Vec<usize>,
    /// The longest n-gram (`ple.ngram_size`).
    pub(crate) ngram: usize,
    /// Hash heads per n-gram order.
    pub(crate) heads: usize,
    /// Width of one table row.
    pub(crate) row_dim: usize,
    /// Hash multipliers, `ngram` per engram layer.
    pub(crate) multipliers: Vec<u64>,
    /// Bucket count of every (layer, order, head).
    pub(crate) head_sizes: Vec<u64>,
    /// First table row of every (layer, order, head), absolute over the one
    /// table that holds all the layers' rows.
    pub(crate) head_offsets: Vec<u64>,
    pub(crate) pad_id: u32,
    /// Compressed token id of every vocabulary entry.
    pub(crate) token_map: Vec<u32>,
}

impl EngramSpec {
    /// Table rows each token reads per engram layer.
    pub(crate) fn cols(&self) -> usize {
        (self.ngram - 1) * self.heads
    }

    /// Position of `layer` among the engram layers.
    pub(crate) fn index_of(&self, layer: usize) -> Option<usize> {
        self.layers.iter().position(|&l| l == layer)
    }
}

/// Every unsigned integer of an array-valued key.
fn usize_array(k: &Keys<'_>, arch: Arch, key: &str) -> Result<Option<Vec<usize>>, CucaError> {
    let Some(v) = k.value(key) else {
        return Ok(None);
    };
    let full = format!("{}.{key}", arch.name());
    let Value::Array(items) = v else {
        return Err(meta_err(&full, "not an array"));
    };
    items
        .iter()
        .map(|v| value_usize(v).ok_or_else(|| meta_err(&full, "not an unsigned integer")))
        .collect::<Result<Vec<_>, _>>()
        .map(Some)
}

/// An array of 64-bit unsigned integers.
fn u64_array(md: &Metadata, key: &str) -> Result<Option<Vec<u64>>, CucaError> {
    let Some(v) = md.get(key) else {
        return Ok(None);
    };
    let Value::Array(items) = v else {
        return Err(meta_err(key, "not an array"));
    };
    items
        .iter()
        .map(|v| match v {
            Value::U64(n) => Ok(*n),
            other => value_usize(other)
                .map(|n| n as u64)
                .ok_or_else(|| meta_err(key, "not an unsigned integer")),
        })
        .collect::<Result<Vec<_>, _>>()
        .map(Some)
}

/// An array of floats.
fn f32_array(md: &Metadata, key: &str) -> Result<Option<Vec<f32>>, CucaError> {
    let Some(v) = md.get(key) else {
        return Ok(None);
    };
    let Value::Array(items) = v else {
        return Err(meta_err(key, "not an array"));
    };
    items
        .iter()
        .map(|v| match v {
            Value::F32(x) => Ok(*x),
            Value::F64(x) => Ok(*x as f32),
            _ => Err(meta_err(key, "not a float")),
        })
        .collect::<Result<Vec<_>, _>>()
        .map(Some)
}

/// `hyper_connection.*`.
pub(crate) fn read_hyper(k: &Keys<'_>, arch: Arch) -> Result<HyperSpec, CucaError> {
    let streams = k.usize("hyper_connection.count")?;
    if !(2..=8).contains(&streams) {
        return Err(meta_err(
            &format!("{}.hyper_connection.count", arch.name()),
            format!("{streams} streams are not supported (2 to 8)"),
        ));
    }
    Ok(HyperSpec {
        streams,
        sinkhorn_iters: k.usize("hyper_connection.sinkhorn_iterations")?.max(1),
        eps: k.f64("hyper_connection.epsilon")? as f32,
    })
}

/// The compressed-sparse-attention layer specs and what the layers share:
/// the specs of the `n_layer` layers, the rotary embeddings they use (pushed
/// to `ropes`) and the shared geometry.
///
/// Layers with ratio 0 attend to their sliding window only and rotate with
/// `rope.freq_base`; the others rotate with `attention.compress_rope_freq_base`
/// (YaRN-scaled), because a compressed row stands for `ratio` tokens.
pub(crate) fn read_csa(
    k: &Keys<'_>,
    arch: Arch,
    n_layer: usize,
    n_head: &[usize],
    ropes: &mut Vec<RopeSpec>,
) -> Result<(Vec<super::params::LayerSpec>, CsaGlobals), CucaError> {
    let key = |s: &str| format!("{}.{s}", arch.name());
    let head_dim = k.usize("attention.key_length")?;
    let rot = k.usize("rope.dimension_count")?;
    if head_dim == 0 || rot == 0 || rot > head_dim || !rot.is_multiple_of(2) {
        return Err(err(format!(
            "GGUF metadata: rope dimension count {rot} must be even and within the key length {head_dim}"
        )));
    }
    let window = k.usize("attention.sliding_window")?;
    if window == 0 {
        return Err(meta_err(
            &key("attention.sliding_window"),
            "must be non-zero",
        ));
    }
    let ratios = usize_array(k, arch, "attention.compress_ratios")?
        .ok_or_else(|| meta_err(&key("attention.compress_ratios"), "missing"))?;
    if ratios.len() < n_layer {
        return Err(meta_err(
            &key("attention.compress_ratios"),
            format!("{} entries for {n_layer} layers", ratios.len()),
        ));
    }
    let kv_sources = usize_array(k, arch, "attention.kv_source_layers")?.ok_or_else(|| {
        meta_err(
            &key("attention.kv_source_layers"),
            "missing (a GGUF of an older layout of this model)",
        )
    })?;
    let index_sources = usize_array(k, arch, "attention.indexer.source_layers")?
        .ok_or_else(|| meta_err(&key("attention.indexer.source_layers"), "missing"))?;
    let heads = k.usize("attention.indexer.head_count")?;
    let dim = k.usize("attention.indexer.key_length")?;
    let top_k = k.usize("attention.indexer.top_k")?;
    if heads == 0 || dim == 0 || top_k == 0 || dim < rot || !rot.is_multiple_of(2) {
        return Err(err(
            "GGUF metadata: the indexer needs non-zero heads, key length and top-k",
        ));
    }
    // `candidate_source_layer` is omitted when negative.
    let candidate_source = match k.value("attention.indexer.candidate_source_layer") {
        None => None,
        Some(Value::I32(n)) => usize::try_from(*n).ok(),
        Some(v) => value_usize(v),
    };
    let candidates = match candidate_source {
        None => None,
        Some(src) => {
            let blocks = k.usize("attention.indexer.candidate_top_k_blocks")?;
            let size = k.usize("attention.indexer.candidate_block_size")?;
            if blocks == 0 || size == 0 {
                return Err(err(
                    "GGUF metadata: candidate blocks and block size must be non-zero",
                ));
            }
            if !index_sources.contains(&src) {
                return Err(meta_err(
                    &key("attention.indexer.candidate_source_layer"),
                    format!("layer {src} is not an indexer source layer"),
                ));
            }
            Some((src, blocks, size))
        }
    };
    let indexer = IndexerSpec {
        heads,
        dim,
        top_k,
        candidates,
    };
    let globals = CsaGlobals {
        q_lora: k.usize("attention.q_lora_rank")?,
        out_groups: k.usize("attention.output_group_count")?,
        out_rank: k.usize("attention.output_lora_rank")?,
        indexer,
    };
    if globals.q_lora == 0 || globals.out_groups == 0 || globals.out_rank == 0 {
        return Err(err(
            "GGUF metadata: q_lora_rank and the output projection ranks must be non-zero",
        ));
    }

    // Rotary embeddings: plain for the window-only layers, YaRN-scaled at the
    // compress base for the others. The reference applies no magnitude
    // correction (`softmax_scale = head_dim^-0.5` alone), so the table's own
    // YaRN magnitude is cancelled.
    let n_ctx_train = k.usize("context_length")?;
    let n_ctx_orig = k
        .usize_opt("rope.scaling.original_context_length")?
        .unwrap_or(n_ctx_train);
    let base = k.f64_opt("rope.freq_base")?.unwrap_or(10_000.0);
    let compress_base = k
        .f64_opt("attention.compress_rope_freq_base")?
        .unwrap_or(base);
    let factor = k.f64_opt("rope.scaling.factor")?.unwrap_or(0.0);
    let compress_rope = if k.str_opt("rope.scaling.type") == Some("yarn") && factor > 0.0 {
        RopeSpec {
            kind: RopeKind::Norm,
            n_dims: rot,
            base: compress_base,
            freq_scale: 1.0 / factor,
            yarn: Some(Yarn {
                ext_factor: 1.0,
                beta_fast: k.f64_opt("rope.scaling.yarn_beta_fast")?.unwrap_or(32.0),
                beta_slow: k.f64_opt("rope.scaling.yarn_beta_slow")?.unwrap_or(1.0),
                n_ctx_orig,
            }),
            attn_factor: 1.0 / (1.0 + 0.1 * factor.ln()),
            factors: super::rope::Factors::None,
        }
    } else {
        RopeSpec::plain(RopeKind::Norm, rot, compress_base)
    };
    let window_rope = RopeSpec::plain(RopeKind::Norm, rot, base);
    let mut rope_index = |spec: RopeSpec| match ropes.iter().position(|r| *r == spec) {
        Some(i) => i,
        None => {
            ropes.push(spec);
            ropes.len() - 1
        }
    };

    let latest = |sources: &[usize], il: usize| sources.iter().rev().find(|&&s| s <= il).copied();
    let mut layers = Vec::with_capacity(n_layer);
    for il in 0..n_layer {
        let ratio = ratios[il];
        let csa = if ratio == 0 {
            None
        } else {
            if ratio > 2 {
                return Err(meta_err(
                    &key("attention.compress_ratios"),
                    format!("layer {il}: ratio {ratio} is not supported (1 or 2)"),
                ));
            }
            let kv_source = latest(&kv_sources, il).ok_or_else(|| {
                err(format!(
                    "layer {il} reads compressed rows but no earlier layer computes them"
                ))
            })?;
            let index_source = latest(&index_sources, il).ok_or_else(|| {
                err(format!(
                    "layer {il} reads indexer picks but no earlier layer scores"
                ))
            })?;
            if ratios[kv_source] != ratio || ratios[index_source] != ratio {
                return Err(err(format!(
                    "layer {il} has ratio {ratio} but its cache source {kv_source} or index source {index_source} has another"
                )));
            }
            let owns_kv = kv_source == il;
            let scores = index_source == il;
            Some(CsaSpec {
                ratio,
                kv_source,
                index_source,
                owns_kv,
                scores,
                candidates: match candidates {
                    Some((src, _, _)) if src == il => Candidates::Source,
                    Some((src, _, _)) if scores && src < il => Candidates::Consumer,
                    _ => Candidates::None,
                },
                row_width: if owns_kv { head_dim + dim } else { 0 },
            })
        };
        let rope = rope_index(if ratio == 0 {
            window_rope
        } else {
            compress_rope
        });
        layers.push(super::params::LayerSpec {
            kind: super::params::LayerKind::Attention,
            n_head: n_head[il],
            n_kv_head: 1,
            head_dim,
            window: Some(window),
            rope: Some(rope),
            kv_source: il,
            scale: (1.0 / (head_dim as f64).sqrt()) as f32,
            mla: None,
            kv_is_value: true,
            csa,
        });
    }
    Ok((layers, globals))
}

/// `ple.*` and `engram.*`: the hashed table lookups.
pub(crate) fn read_engram(
    md: &Metadata,
    arch: Arch,
    k: &Keys<'_>,
    n_layer: usize,
) -> Result<EngramSpec, CucaError> {
    let layers = usize_array(k, arch, "ple.layers")?
        .ok_or_else(|| meta_err(&format!("{}.ple.layers", arch.name()), "missing"))?;
    let ngram = k.usize("ple.ngram_size")?;
    let heads = k.usize("ple.heads_per_ngram")?;
    let row_dim = k.usize("embedding_length_per_layer_input")?;
    if layers.is_empty()
        || !(2..=super::engram::MAX_NGRAM).contains(&ngram)
        || heads == 0
        || row_dim == 0
    {
        return Err(err(
            "GGUF metadata: the engram needs layers, an n-gram size of at least 2, heads and a row width",
        ));
    }
    if layers.iter().any(|&l| l >= n_layer) || layers.windows(2).any(|w| w[0] >= w[1]) {
        return Err(meta_err(
            &format!("{}.ple.layers", arch.name()),
            "layers must ascend and exist",
        ));
    }
    let name = |s: &str| format!("{}.{s}", arch.name());
    let multipliers = u64_array(md, &name("ple.layer_multipliers"))?
        .ok_or_else(|| meta_err(&name("ple.layer_multipliers"), "missing"))?;
    let head_offsets = u64_array(md, &name("ple.head_offsets"))?
        .ok_or_else(|| meta_err(&name("ple.head_offsets"), "missing"))?;
    let head_sizes = u64_array(md, &name("ple.head_vocab_sizes"))?
        .ok_or_else(|| meta_err(&name("ple.head_vocab_sizes"), "missing"))?;
    let per_layer = (ngram - 1) * heads;
    if multipliers.len() != layers.len() * ngram {
        return Err(meta_err(
            &name("ple.layer_multipliers"),
            format!(
                "{} values, expected {}",
                multipliers.len(),
                layers.len() * ngram
            ),
        ));
    }
    for (key, v) in [
        ("ple.head_offsets", &head_offsets),
        ("ple.head_vocab_sizes", &head_sizes),
    ] {
        if v.len() != layers.len() * per_layer {
            return Err(meta_err(
                &name(key),
                format!("{} values, expected {}", v.len(), layers.len() * per_layer),
            ));
        }
    }
    if head_sizes.contains(&0) {
        return Err(meta_err(
            &name("ple.head_vocab_sizes"),
            "a bucket count of 0",
        ));
    }
    let token_map = match md.get(&name("engram.token_map")) {
        Some(Value::Array(items)) => items
            .iter()
            .map(|v| {
                value_usize(v)
                    .and_then(|n| u32::try_from(n).ok())
                    .ok_or_else(|| meta_err(&name("engram.token_map"), "not a token id"))
            })
            .collect::<Result<Vec<_>, _>>()?,
        _ => return Err(meta_err(&name("engram.token_map"), "missing")),
    };
    let pad_id = k.usize("engram.pad_token_id")?;
    let pad_id = u32::try_from(pad_id)
        .ok()
        .filter(|&p| (p as usize) < token_map.len())
        .ok_or_else(|| meta_err(&name("engram.pad_token_id"), "outside the token map"))?;
    Ok(EngramSpec {
        layers,
        ngram,
        heads,
        row_dim,
        multipliers,
        head_sizes,
        head_offsets,
        pad_id,
        token_map,
    })
}

/// `expert_gating_func` and the SwiGLU clamps.
pub(crate) fn read_moe_ext(
    md: &Metadata,
    arch: Arch,
    k: &Keys<'_>,
    n_layer: usize,
) -> Result<MoeExt, CucaError> {
    let gating = match k.usize_opt("expert_gating_func")? {
        None | Some(1) => Gating::Softmax,
        Some(2) => Gating::Sigmoid,
        Some(4) => Gating::SqrtSoftplus,
        Some(other) => {
            return Err(meta_err(
                &format!("{}.expert_gating_func", arch.name()),
                format!("gating function {other} is not supported"),
            ));
        }
    };
    let per_layer = |key: &str| -> Result<Vec<f32>, CucaError> {
        let full = format!("{}.{key}", arch.name());
        match f32_array(md, &full)? {
            Some(v) if v.len() == n_layer => Ok(v),
            Some(v) => Err(meta_err(
                &full,
                format!("{} entries for {n_layer} layers", v.len()),
            )),
            None => Ok(vec![0.0; n_layer]),
        }
    };
    Ok(MoeExt {
        gating,
        clamp: per_layer("swiglu_clamp_exp")?,
        clamp_shared: per_layer("swiglu_clamp_shexp")?,
    })
}
