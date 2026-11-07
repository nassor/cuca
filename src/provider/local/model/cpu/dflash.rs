//! The DFlash block-diffusion drafter (`dflash` GGUF files; Muse Glimmer's
//! `Muse-Glimmer-30B-assistant`, Qwen's and others').
//!
//! A DFlash drafter proposes a whole block of tokens in **one** forward pass
//! instead of chaining one token per pass. It owns no embedding table and no
//! output head: it reads the target's, and it never sees the target's K/V
//! cache. What it reads from the target is its **hidden features**, the
//! residual stream after a few chosen decoder layers, of every position the
//! target has accepted ([`Hidden::Layers`](super::Hidden)).
//!
//! # One round
//!
//! `len` positions are cached in the target and the pending token `first`
//! (sampled from the target's last logits, not cached yet) sits at position
//! `len`. The drafter runs a block of `1 + k` rows (`k` drafts wanted, at
//! most `block_size - 1`; the block of Muse Glimmer's is 16 rows) at
//! positions `len..len + 1 + k`: row 0 is `first`, rows `1..=k` are the mask
//! token. Their inputs are the target's *raw*
//! token embeddings (the lookup, without the target's embedding norm). Every
//! layer is a pre-norm block, `x += attn(norm(x))`, `x += ffn(norm(x))`, with
//! grouped-query attention, per-head weighted Q/K RMS norms and NEOX RoPE
//! (the scores scale by `1 / sqrt(head_dim)`, nothing else), whose keys and
//! values are
//!
//! - the **context**, one K/V entry per accepted position and layer:
//!   `K = rope(k_norm(k_proj(c)))`, `V = v_proj(c)` for the context row
//!   `c = RMSNorm_enc(fc(concat(features)))` (`fc` maps the `n_features`
//!   concatenated `n_embd` feature rows to the drafter width), and
//! - the **block rows** themselves, projected from the normed block input
//!   like any attention layer.
//!
//! The block rows attend to each other in both directions (bidirectional)
//! and to the context causally, each layer restricted to the sliding window
//! when it has one: a query at position `q` reads a key at position `p` only
//! when `|q - p| <= window` (transformers'
//! `create_bidirectional_sliding_window_mask`; the checkpoint's reference).
//! After the last layer, the final norm and the **target's** LM head (with
//! its logit multiplier and soft-cap, so the drafter's distribution lives on
//! the target's scale) turn rows `1..=k` into the logits of the `k` draft
//! tokens, the token after `first` first: one pass, `k` tokens.
//!
//! # Context cache
//!
//! The context K/V of a position depends only on that position's feature row,
//! so the drafter caches it as the rows arrive ([`DflashNet::observe`]) and a
//! round only projects the block. Every layer here is a sliding-window layer
//! (a full-attention layer is supported too), so the cache is a ring of the
//! last `window` positions per layer: `5 layers x 2 x 8 heads x 128 x 4 bytes
//! x 2048` = 82 MB for the Muse drafter whatever the context length. A
//! position older than the window is overwritten and never read again by a
//! forward round, since the next block starts at or after the last one.
//!
//! The ring holds the contiguous positions `[lo, hi)`:
//!
//! - observing rows at `pos == hi` appends them; at `pos < hi` the rows it
//!   replaces (`pos..hi`) are dropped first (a rejected draft, a replayed
//!   prompt tail); at `pos > hi` the held range is dropped (nothing joins it)
//!   and a new one starts at `pos`;
//! - [`DflashNet::truncate`]`(shared)` keeps the positions below `shared` (a
//!   position's row depends only on the tokens up to it, so the first
//!   `shared` rows stay valid when a new prompt shares `shared` tokens with
//!   the cached ids), [`DflashNet::reset`] drops everything;
//! - a draft needs `hi == len`: without the context up to the pending token
//!   the drafter proposes nothing (the round runs plain).
//!
//! A context shorter than the window should hold (after a reset, or a prefix
//! whose older rows were gone) only costs proposal quality: the drafter only
//! proposes, and verification keeps the output equal to plain decoding.
//!
//! # Target layers
//!
//! `dflash.target_layers` follows llama.cpp: entry `i` names the layer whose
//! *input* feeds the drafter (the output of layer `i - 1`), so the HF
//! `target_layer_ids` `[1, 13, 25, 37, 49]` of the Muse checkpoint are stored
//! as `[2, 14, 26, 38, 50]`. transformers' `target_layer_ids` are *outputs*
//! of decoder layer `i` (`output_hidden_states=[ids]` returns the hook of
//! layer `i`; checked against the real `generate(.., speculation_type=
//! "dflash")` path on a tiny model: the features handed to the assistant
//! equal `hidden_states[i + 1]` of the full-form output, and differ from
//! `hidden_states[i]`). The loader subtracts one and requires the layers to
//! ascend, because the target returns them in ascending order.
//!
//! # Cost
//!
//! One round is one drafter pass over `1 + k` rows whatever the number of
//! drafts kept (at most `k`), plus the LM head over `k` rows: a cost the
//! engine's cost model prices per round, not per draft (see `engine`).
//!
//! # Shorter blocks
//!
//! The block is `1 + k` rows, not always `block_size`: llama.cpp's `dflash`
//! graph builds blocks of "anchor + drafted positions" (bounded by the
//! training block size) too, and the DFlash paper (arXiv 2602.06036, Table
//! 8: an 8-layer drafter trained on blocks of 16, Math500) measures an
//! average accepted length of 6.33 at inference blocks of 16 and 5.09 at
//! blocks of 8. The pass costs `(1 + k) / block_size` of a full block on the
//! compute-bound CPU. The transformers reference runs the full block; the
//! oracle test compares both (a reference block of 16 rows and a shorter
//! one).

use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use candle_core::quantized::gguf_file::{self, Value};
use candle_core::{Device, Tensor};

use super::super::arch::Act;
use super::super::layout::name;
use super::super::params::{Keys, Metadata, value_usize};
use super::super::rope::{RopeKind, RopeSpec};
use super::super::{candle_err, err, meta_err};
use super::{FfnWeights, GateUp, LayerSet, Loader, QMatMul, Transformer, apply_rope, attend, rms};
use crate::error::CucaError;

/// `general.architecture` of a DFlash drafter file (llama.cpp's name).
pub(crate) const ARCH: &str = "dflash";

/// `fc.weight`: the projection of the concatenated target features.
const FC: &str = "fc.weight";
/// `enc.output_norm.weight`: the norm applied to the projected features.
const ENC_NORM: &str = "enc.output_norm.weight";

/// What the `dflash.*` metadata says.
#[derive(Debug, Clone, PartialEq)]
struct Geometry {
    n_embd: usize,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    /// Rotated leading dimensions of a head.
    rope_dims: usize,
    n_ff: usize,
    eps: f32,
    rope_base: f64,
    /// Rows of one block: the pending token and `block_size - 1` masks.
    block_size: usize,
    mask_token: u32,
    /// The target layers whose outputs feed the context, ascending.
    features: Vec<usize>,
    /// The sliding window of every layer (`None`: full attention).
    windows: Vec<Option<usize>>,
}

/// A non-negative integer array under `key`.
fn array_usize(md: &Metadata, key: &str) -> Result<Option<Vec<usize>>, CucaError> {
    let Some(value) = md.get(key) else {
        return Ok(None);
    };
    let Value::Array(items) = value else {
        return Err(meta_err(key, "not an array"));
    };
    items
        .iter()
        .map(|v| value_usize(v).ok_or_else(|| meta_err(key, "not an unsigned integer")))
        .collect::<Result<Vec<_>, _>>()
        .map(Some)
}

/// Whether `md` describes a plain DFlash drafter this engine runs (the
/// converter checks the metadata it writes with it).
///
/// # Errors
///
/// [`CucaError::Provider`] naming the key or the inconsistency.
pub(crate) fn check_metadata(md: &Metadata) -> Result<(), CucaError> {
    read_geometry(md).map(|_| ())
}

fn read_geometry(md: &Metadata) -> Result<Geometry, CucaError> {
    let k = Keys::new(md, ARCH);
    for key in [
        "conv_kernel_size",
        "conv_group_size",
        "selector_rank",
        "selector_top_k",
    ] {
        if k.usize_opt(key)?.is_some() {
            return Err(err(format!(
                "`{ARCH}.{key}` marks a DFlash variant with convolution or selector heads, which this engine does not run (it runs the plain block drafter)"
            )));
        }
    }
    let n_layer = k.usize("block_count")?;
    let n_embd = k.usize("embedding_length")?;
    let n_head = k.usize("attention.head_count")?;
    let n_kv_head = k.usize_opt("attention.head_count_kv")?.unwrap_or(n_head);
    let head_dim = k
        .usize_opt("attention.key_length")?
        .unwrap_or(n_embd / n_head.max(1));
    if k.usize_opt("attention.value_length")?
        .is_some_and(|v| v != head_dim)
    {
        return Err(err(
            "a DFlash drafter needs equal key and value lengths".to_string()
        ));
    }
    let rope_dims = k.usize_opt("rope.dimension_count")?.unwrap_or(head_dim);
    let block_size = k.usize("block_size")?;
    if n_layer == 0
        || n_head == 0
        || n_kv_head == 0
        || !n_head.is_multiple_of(n_kv_head)
        || head_dim == 0
        || rope_dims == 0
        || rope_dims > head_dim
        || !rope_dims.is_multiple_of(2)
        || block_size < 2
    {
        return Err(err(format!(
            "`{ARCH}` metadata: inconsistent geometry ({n_layer} layers, {n_head} heads over {n_kv_head} KV heads of {head_dim}, {rope_dims} rotated, blocks of {block_size})"
        )));
    }

    let target_key = format!("{ARCH}.target_layers");
    let inputs = array_usize(md, &target_key)?
        .ok_or_else(|| meta_err(&target_key, "missing: the drafter needs its target layers"))?;
    // llama.cpp's entries are the layers whose *input* feeds the drafter.
    let features = inputs
        .iter()
        .map(|&i| {
            i.checked_sub(1)
                .ok_or_else(|| meta_err(&target_key, "layer 0 (the embeddings) is not a feature"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if features.is_empty() || features.windows(2).any(|w| w[0] >= w[1]) {
        return Err(meta_err(
            &target_key,
            "the target layers must be a non-empty ascending list",
        ));
    }

    let window = k.usize_opt("attention.sliding_window")?;
    let windows = match window {
        None => vec![None; n_layer],
        Some(0) => {
            return Err(meta_err(
                &format!("{ARCH}.attention.sliding_window"),
                "a window of 0 tokens",
            ));
        }
        Some(w) => {
            let key = format!("{ARCH}.attention.sliding_window_pattern");
            match md.get(&key) {
                None => vec![Some(w); n_layer],
                Some(Value::Array(items)) if items.len() == n_layer => items
                    .iter()
                    .map(|v| match v {
                        Value::Bool(sliding) => Ok(sliding.then_some(w)),
                        _ => Err(meta_err(&key, "not an array of booleans")),
                    })
                    .collect::<Result<Vec<_>, _>>()?,
                Some(_) => {
                    return Err(meta_err(
                        &key,
                        format!("expected {n_layer} booleans, one per layer"),
                    ));
                }
            }
        }
    };

    let mask_key = "tokenizer.ggml.mask_token_id";
    let mask_token = md
        .get(mask_key)
        .and_then(value_usize)
        .and_then(|id| u32::try_from(id).ok())
        .ok_or_else(|| meta_err(mask_key, "missing: the drafter's mask token"))?;
    Ok(Geometry {
        n_embd,
        n_head,
        n_kv_head,
        head_dim,
        rope_dims,
        n_ff: k.usize("feed_forward_length")?,
        eps: k.f64("attention.layer_norm_rms_epsilon")? as f32,
        rope_base: k.f64("rope.freq_base")?,
        block_size,
        mask_token,
        features,
        windows,
    })
}

/// One drafter layer.
struct DflashLayer {
    attn_norm: Tensor,
    q: QMatMul,
    k: QMatMul,
    v: QMatMul,
    o: QMatMul,
    q_norm: Tensor,
    k_norm: Tensor,
    ffn_norm: Tensor,
    ffn: FfnWeights,
    window: Option<usize>,
}

/// A DFlash drafter and its context cache; see the [module docs](self).
pub(crate) struct DflashNet {
    geo: Geometry,
    /// The target layers, as the set a forward pass captures.
    layers_set: LayerSet,
    /// Floats of one feature row: `features.len() * target n_embd`.
    feature_width: usize,
    fc: QMatMul,
    enc_norm: Tensor,
    layers: Vec<DflashLayer>,
    out_norm: Tensor,
    /// RoPE cos/sin `(context_length + block_size, rope_dims / 2)`.
    cos: Tensor,
    sin: Tensor,
    /// Slots of the context ring: position `p` lives in slot `p % cap`.
    cap: usize,
    /// The held positions are `lo..hi`.
    lo: usize,
    hi: usize,
    /// Per layer, K and V as `[n_kv_head][cap][head_dim]`.
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
}

impl DflashNet {
    /// Load the drafter at `path` (index `content`) for `target`, whose
    /// positions go up to `context_length`.
    ///
    /// # Errors
    ///
    /// [`CucaError::Provider`] for a file that is not a plain DFlash drafter
    /// (missing keys or tensors, shapes that do not fit), a drafter whose
    /// width differs from the target's or that reads a layer the target does
    /// not have, a mask token outside the target's vocabulary or a target
    /// without a token embedding.
    pub(crate) fn open(
        content: &gguf_file::Content,
        path: &Path,
        target: &Transformer,
        context_length: usize,
    ) -> Result<Self, CucaError> {
        Self::open_with(content, path, target, context_length, false)
    }

    /// [`DflashNet::open`] with every matmul weight dequantized to f32 (the
    /// exact-arithmetic reference the transformers oracle is compared to).
    ///
    /// # Errors
    ///
    /// As [`DflashNet::open`].
    #[cfg(test)]
    pub(crate) fn open_dequantized(
        content: &gguf_file::Content,
        path: &Path,
        target: &Transformer,
        context_length: usize,
    ) -> Result<Self, CucaError> {
        Self::open_with(content, path, target, context_length, true)
    }

    /// [`DflashNet::open`], with every matmul weight dequantized to f32 when
    /// `dequantize` is set (the test reference).
    fn open_with(
        content: &gguf_file::Content,
        path: &Path,
        target: &Transformer,
        context_length: usize,
        dequantize: bool,
    ) -> Result<Self, CucaError> {
        let geo = read_geometry(&content.metadata)?;
        let t = &target.hp;
        if geo.n_embd != t.n_embd {
            return Err(err(format!(
                "the DFlash drafter's width {} differs from the target's {}: it was trained for another model size",
                geo.n_embd, t.n_embd
            )));
        }
        if let Some(&last) = geo.features.last()
            && last >= t.n_layer
        {
            return Err(err(format!(
                "the DFlash drafter reads the output of layer {last}, but the target has {} layers",
                t.n_layer
            )));
        }
        if geo.mask_token as usize >= target.vocab {
            return Err(err(format!(
                "the DFlash mask token {} is outside the target's vocabulary of {}",
                geo.mask_token, target.vocab
            )));
        }
        if target.embed.is_none() {
            return Err(err(
                "the DFlash drafter needs a target with a token embedding",
            ));
        }
        let layers_set = LayerSet::new(geo.features.iter().copied())
            .ok_or_else(|| err("the DFlash drafter reads a layer beyond the 128th"))?;

        let mut loader = Loader {
            content,
            reader: BufReader::new(File::open(path)?),
            dequantize,
            // A drafter only proposes: float weights are held as Q8_0.
            requantize: true,
        };
        let feature_width = geo.features.len() * t.n_embd;
        let n = geo.n_embd;
        let fc = loader.matmul(FC, &[n, feature_width])?;
        let enc_norm = loader.vector(ENC_NORM, n)?;
        let out_norm = loader.vector(name::OUTPUT_NORM, n)?;
        let (qd, kd) = (geo.n_head * geo.head_dim, geo.n_kv_head * geo.head_dim);
        let mut layers = Vec::with_capacity(geo.windows.len());
        for (il, &window) in geo.windows.iter().enumerate() {
            let w = |part: &str| name::blk(il, part);
            layers.push(DflashLayer {
                attn_norm: loader.vector(&w(name::ATTN_NORM), n)?,
                q: loader.matmul(&w(name::ATTN_Q), &[qd, n])?,
                k: loader.matmul(&w(name::ATTN_K), &[kd, n])?,
                v: loader.matmul(&w(name::ATTN_V), &[kd, n])?,
                o: loader.matmul(&w(name::ATTN_OUTPUT), &[n, qd])?,
                q_norm: loader.vector(&w(name::ATTN_Q_NORM), geo.head_dim)?,
                k_norm: loader.vector(&w(name::ATTN_K_NORM), geo.head_dim)?,
                ffn_norm: loader.vector(&w(name::FFN_NORM), n)?,
                ffn: FfnWeights::Dense {
                    gate_up: GateUp::Split {
                        gate: loader.matmul(&w(name::FFN_GATE), &[geo.n_ff, n])?,
                        up: loader.matmul(&w(name::FFN_UP), &[geo.n_ff, n])?,
                    },
                    down: loader.matmul(&w(name::FFN_DOWN), &[n, geo.n_ff])?,
                },
                window,
            });
        }

        let table = RopeSpec::plain(RopeKind::Neox, geo.rope_dims, geo.rope_base)
            .tables(None, context_length + geo.block_size);
        let shape = (context_length + geo.block_size, table.half);
        let dev = &Device::Cpu;
        let cos = Tensor::from_vec(table.cos, shape, dev).map_err(candle_err)?;
        let sin = Tensor::from_vec(table.sin, shape, dev).map_err(candle_err)?;

        // The ring holds what the widest layer reads: a window, or the whole
        // context for a full-attention layer.
        let cap = geo
            .windows
            .iter()
            .map(|w| w.unwrap_or(context_length))
            .max()
            .unwrap_or(1)
            .min(context_length)
            .max(1);
        let ring = geo.n_kv_head * cap * geo.head_dim;
        let n_layer = layers.len();
        Ok(Self {
            geo,
            layers_set,
            feature_width,
            fc,
            enc_norm,
            layers,
            out_norm,
            cos,
            sin,
            cap,
            lo: 0,
            hi: 0,
            k: vec![vec![0.0; ring]; n_layer],
            v: vec![vec![0.0; ring]; n_layer],
        })
    }

    /// The target layers whose outputs the drafter reads, as the set to ask
    /// a forward pass for ([`Hidden::Layers`](super::Hidden)).
    pub(crate) fn features(&self) -> LayerSet {
        self.layers_set
    }

    /// Floats of one feature row: the target layers' outputs one after the
    /// other.
    pub(crate) fn feature_width(&self) -> usize {
        self.feature_width
    }

    /// The positions the context cache holds, as `lo..hi`.
    #[cfg(test)]
    pub(crate) fn held(&self) -> (usize, usize) {
        (self.lo, self.hi)
    }

    /// Forget every position.
    pub(crate) fn reset(&mut self) {
        self.lo = 0;
        self.hi = 0;
    }

    /// Keep the positions below `shared` (they depend on the first `shared`
    /// tokens only) and forget the rest.
    pub(crate) fn truncate(&mut self, shared: usize) {
        self.hi = self.hi.min(shared);
        self.lo = self.lo.min(self.hi);
    }

    /// Rotate the rows `m` `(1, heads, t, head_dim)` held at positions
    /// `start..start + t`.
    fn rotate(&self, m: &Tensor, start: usize) -> candle_core::Result<Tensor> {
        let t = m.dim(2)?;
        apply_rope(
            m,
            &self.cos.narrow(0, start, t)?,
            &self.sin.narrow(0, start, t)?,
            RopeKind::Neox,
            self.geo.rope_dims,
        )
    }

    /// Cache the context rows of the positions `pos..`: `feats` holds one
    /// feature row per position (the target layers' outputs one after the
    /// other, [`Hidden::Layers`](super::Hidden)).
    ///
    /// # Errors
    ///
    /// [`CucaError::Provider`] for a length that is not a whole number of
    /// feature rows, positions beyond the context or a failed projection.
    pub(crate) fn observe(&mut self, pos: usize, feats: &[f32]) -> Result<(), CucaError> {
        let width = self.feature_width;
        if feats.is_empty() || !feats.len().is_multiple_of(width) {
            return Err(err(format!(
                "DFlash observe: {} floats are not whole rows of {width}",
                feats.len()
            )));
        }
        let rows = feats.len() / width;
        if pos + rows + self.geo.block_size > self.cos.dim(0).map_err(candle_err)? {
            return Err(err(format!(
                "DFlash observe: positions up to {} exceed the context",
                pos + rows
            )));
        }
        // Rows that would leave the ring at once are not projected.
        let skip = rows.saturating_sub(self.cap);
        let first = pos + skip;
        let kept = &feats[skip * width..];
        let per_layer = self.context_kv(kept, first).map_err(candle_err)?;
        // Only now does the held range change: a failed projection leaves it
        // as it was.
        if pos < self.hi {
            self.hi = pos;
            self.lo = self.lo.min(pos);
        } else if pos > self.hi || self.hi == self.lo {
            self.lo = pos;
            self.hi = pos;
        }
        let d = self.geo.head_dim;
        let m = rows - skip;
        for (il, (ks, vs)) in per_layer.iter().enumerate() {
            for h in 0..self.geo.n_kv_head {
                for r in 0..m {
                    let slot = (first + r) % self.cap;
                    let dst = (h * self.cap + slot) * d;
                    let src = (h * m + r) * d;
                    self.k[il][dst..dst + d].copy_from_slice(&ks[src..src + d]);
                    self.v[il][dst..dst + d].copy_from_slice(&vs[src..src + d]);
                }
            }
        }
        self.hi = pos + rows;
        self.lo = self.lo.max(self.hi.saturating_sub(self.cap));
        Ok(())
    }

    /// Per layer, the K and V rows `[n_kv_head][m][head_dim]` of the context
    /// positions `first..` whose feature rows are `feats`.
    fn context_kv(
        &self,
        feats: &[f32],
        first: usize,
    ) -> candle_core::Result<Vec<(Vec<f32>, Vec<f32>)>> {
        let g = &self.geo;
        let m = feats.len() / self.feature_width;
        let x = Tensor::from_slice(feats, (1, m, self.feature_width), &Device::Cpu)?;
        let ctx = rms(&self.fc.forward(&x)?, &self.enc_norm, g.eps)?;
        self.layers
            .iter()
            .map(|lw| {
                let k =
                    lw.k.forward(&ctx)?
                        .reshape((1, m, g.n_kv_head, g.head_dim))?;
                let k = rms(&k, &lw.k_norm, g.eps)?;
                let k = self.rotate(&k.transpose(1, 2)?.contiguous()?, first)?;
                let v =
                    lw.v.forward(&ctx)?
                        .reshape((1, m, g.n_kv_head, g.head_dim))?
                        .transpose(1, 2)?
                        .contiguous()?;
                Ok((
                    k.flatten_all()?.to_vec1::<f32>()?,
                    v.flatten_all()?.to_vec1::<f32>()?,
                ))
            })
            .collect()
    }

    /// Propose at most `k` tokens after the pending token `first` of a
    /// sequence with `len` cached positions: one pass over a block, then
    /// each draft row's logits go to `pick`, which chooses the token or ends
    /// the chain with `None`. The returned drafts are the tokens `pick`
    /// accepted. Nothing is cached: the block's own K/V is dropped.
    ///
    /// Empty when the context cache does not hold every position below `len`
    /// (nothing to condition the block on) or `k` is zero.
    ///
    /// # Errors
    ///
    /// [`CucaError::Provider`] for a failed pass.
    pub(crate) fn draft(
        &self,
        target: &Transformer,
        first: u32,
        len: usize,
        k: usize,
        pick: &mut dyn FnMut(&mut [f32]) -> Option<u32>,
    ) -> Result<Vec<u32>, CucaError> {
        let k = k.min(self.geo.block_size - 1);
        if k == 0 || len == 0 || self.hi != len || self.lo >= self.hi {
            return Ok(Vec::new());
        }
        let rows = k + 1;
        let mut ids = vec![self.geo.mask_token; rows];
        ids[0] = first;
        let embeddings = target
            .embed
            .as_ref()
            .ok_or_else(|| err("the DFlash drafter needs a target with a token embedding"))?
            .rows(&ids, target.hp.embd_scale)?;
        let mut logits = self
            .block_logits(target, embeddings, len, k)
            .map_err(candle_err)?;
        let vocab = logits.len() / k;
        let mut drafts = Vec::with_capacity(k);
        for row in logits.chunks_exact_mut(vocab) {
            let Some(token) = pick(row) else { break };
            drafts.push(token);
        }
        Ok(drafts)
    }

    /// The logits of block rows `1..=k` (`k * vocab` floats) for the block
    /// whose input rows are `embeddings` at positions `len..`.
    fn block_logits(
        &self,
        target: &Transformer,
        embeddings: Vec<f32>,
        len: usize,
        k: usize,
    ) -> candle_core::Result<Vec<f32>> {
        let g = &self.geo;
        let mut x = Tensor::from_vec(embeddings, (1, k + 1, g.n_embd), &Device::Cpu)?;
        for (il, lw) in self.layers.iter().enumerate() {
            x = self.block_layer(lw, il, &x, len)?;
        }
        let normed = rms(&x, &self.out_norm, g.eps)?;
        target
            .head(&normed.narrow(1, 1, k)?)?
            .flatten_all()?
            .to_vec1::<f32>()
    }

    /// One layer over the block rows `x` `(1, rows, n_embd)` at positions
    /// `len..`.
    fn block_layer(
        &self,
        lw: &DflashLayer,
        il: usize,
        x: &Tensor,
        len: usize,
    ) -> candle_core::Result<Tensor> {
        let g = &self.geo;
        let b = x.dim(1)?;
        let (nh, nkv, d) = (g.n_head, g.n_kv_head, g.head_dim);
        let h = rms(x, &lw.attn_norm, g.eps)?;
        let q = rms(
            &lw.q.forward(&h)?.reshape((1, b, nh, d))?,
            &lw.q_norm,
            g.eps,
        )?;
        let q = self.rotate(&q.transpose(1, 2)?.contiguous()?, len)?;
        let k_blk = rms(
            &lw.k.forward(&h)?.reshape((1, b, nkv, d))?,
            &lw.k_norm,
            g.eps,
        )?;
        let k_blk = self.rotate(&k_blk.transpose(1, 2)?.contiguous()?, len)?;
        let v_blk =
            lw.v.forward(&h)?
                .reshape((1, b, nkv, d))?
                .transpose(1, 2)?
                .contiguous()?;

        // The context a row can reach: from the window of the first row (it
        // reaches furthest back) to the pending token.
        let from = match lw.window {
            Some(w) => self.lo.max(len.saturating_sub(w)),
            None => self.lo,
        };
        let n_ctx = len - from;
        let (keys, values) = if n_ctx == 0 {
            (k_blk, v_blk)
        } else {
            let (ck, cv) = self.gather(il, from, len);
            let dev = &Device::Cpu;
            (
                Tensor::cat(&[Tensor::from_vec(ck, (1, nkv, n_ctx, d), dev)?, k_blk], 2)?,
                Tensor::cat(&[Tensor::from_vec(cv, (1, nkv, n_ctx, d), dev)?, v_blk], 2)?,
            )
        };
        let mask = match lw.window {
            None => None,
            Some(w) => {
                let l = n_ctx + b;
                let mut mask = vec![0f32; b * l];
                for i in 0..b {
                    for c in 0..n_ctx {
                        if len + i - (from + c) > w {
                            mask[i * l + c] = f32::NEG_INFINITY;
                        }
                    }
                    for j in 0..b {
                        if i.abs_diff(j) > w {
                            mask[i * l + n_ctx + j] = f32::NEG_INFINITY;
                        }
                    }
                }
                Some(Tensor::from_vec(mask, (b, l), &Device::Cpu)?)
            }
        };
        let attn = attend(&q, &keys, &values, mask.as_ref(), 1.0 / (d as f32).sqrt())?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((1, b, nh * d))?;
        let x = (x + lw.o.forward(&attn)?)?;
        let f = lw.ffn.forward(&rms(&x, &lw.ffn_norm, g.eps)?, Act::Silu)?;
        &x + f
    }

    /// The K and V rows `[n_kv_head][to - from][head_dim]` of the cached
    /// positions `from..to` of layer `il`.
    fn gather(&self, il: usize, from: usize, to: usize) -> (Vec<f32>, Vec<f32>) {
        let (nkv, d) = (self.geo.n_kv_head, self.geo.head_dim);
        let n = to - from;
        let mut ks = Vec::with_capacity(nkv * n * d);
        let mut vs = Vec::with_capacity(nkv * n * d);
        for h in 0..nkv {
            let base = h * self.cap * d;
            let slot = from % self.cap;
            // The range wraps around the end of the ring at most once.
            let head = n.min(self.cap - slot);
            for (src, dst) in [(&self.k[il], &mut ks), (&self.v[il], &mut vs)] {
                dst.extend_from_slice(&src[base + slot * d..base + (slot + head) * d]);
                dst.extend_from_slice(&src[base..base + (n - head) * d]);
            }
        }
        (ks, vs)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use candle_core::Device;

    use super::super::Hidden;
    use super::*;
    use crate::provider::local::model::Want;
    use crate::provider::local::model::tiny::{
        DFLASH_HEAD_DIM, DFLASH_HEADS, DFLASH_KV_HEADS, DFLASH_MASK, Lcg, N_EMBD, PROMPT,
        TinyDflash, TinyModel, TinySpec, load, write_spec, write_tiny_dflash,
    };

    const CTX: usize = 64;

    /// A tiny Muse Glimmer target (four layers, vocabulary 16).
    fn muse() -> TinyModel {
        write_spec(&TinySpec {
            context: CTX,
            ..TinySpec::muse_glimmer()
        })
    }

    fn open_net(
        target: &Transformer,
        drafter: &TinyModel,
        ctx: usize,
        dequantize: bool,
    ) -> DflashNet {
        let content =
            gguf_file::Content::read(&mut BufReader::new(File::open(&drafter.path).unwrap()))
                .unwrap();
        DflashNet::open_with(&content, &drafter.path, target, ctx, dequantize).unwrap()
    }

    /// `n` random feature rows.
    fn features(net: &DflashNet, n: usize, seed: u64) -> Vec<f32> {
        Lcg(seed).values(n * net.feature_width(), 1.5)
    }

    /// The logits rows of the block drafted after `len` positions (`k` of
    /// them).
    fn draft_logits(
        net: &DflashNet,
        target: &Transformer,
        len: usize,
        first: u32,
        k: usize,
    ) -> Vec<Vec<f32>> {
        let mut rows = Vec::new();
        net.draft(target, first, len, k, &mut |l| {
            rows.push(l.to_vec());
            Some(0)
        })
        .unwrap();
        rows
    }

    // ---- Hidden::Layers --------------------------------------------------

    /// `Hidden::Layers` returns the residual stream after each layer of the
    /// set, ascending, for every row, without changing the logits, and
    /// chunked calls return the same rows.
    #[test]
    fn layers_capture_returns_the_residual_stream_of_every_row() {
        let model = muse();
        let tokens = &PROMPT[..];
        let n = tokens.len();
        let set = LayerSet::new([3, 1]).unwrap();
        assert_eq!(set.len(), 2);
        assert!(set.contains(1) && set.contains(3) && !set.contains(2));
        assert!(LayerSet::new([128]).is_none());

        let want = load(&model, CTX)
            .forward_ext(tokens, 0, Want::All, Hidden::None, true)
            .unwrap();
        let got = load(&model, CTX)
            .forward_ext(tokens, 0, Want::All, Hidden::Layers(set), true)
            .unwrap();
        assert_eq!(got.logits, want.logits, "the logits keep their bits");
        let rows = got.hidden.unwrap();
        assert_eq!(rows.len(), n * 2 * N_EMBD);

        // Each layer alone, with no LM head: its rows sit in its slot of
        // every row.
        for (slot, il) in [1usize, 3].into_iter().enumerate() {
            let alone = load(&model, CTX)
                .forward_ext(
                    tokens,
                    0,
                    Want::None,
                    Hidden::Layers(LayerSet::new([il]).unwrap()),
                    true,
                )
                .unwrap();
            assert!(alone.logits.is_none());
            let alone = alone.hidden.unwrap();
            for r in 0..n {
                assert_eq!(
                    rows[r * 2 * N_EMBD + slot * N_EMBD..][..N_EMBD],
                    alone[r * N_EMBD..][..N_EMBD],
                    "layer {il} row {r}"
                );
            }
        }

        // The last layer's output through the final norm is the post-norm
        // hidden row the LM head reads.
        let mut model_last = load(&model, CTX);
        let last = model_last
            .forward_ext(
                tokens,
                0,
                Want::None,
                Hidden::Layers(LayerSet::new([3]).unwrap()),
                true,
            )
            .unwrap()
            .hidden
            .unwrap();
        let mut post = load(&model, CTX);
        let normed = post
            .forward_ext(tokens, 0, Want::None, Hidden::PostNormAll, true)
            .unwrap()
            .hidden
            .unwrap();
        let t = Tensor::from_vec(last, (1, n, N_EMBD), &Device::Cpu).unwrap();
        let expect = rms(&t, &post.output_norm, post.hp.eps as f32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(expect, normed);

        // Chunks (a prefill, a decode, a verification round) give the same
        // rows.
        let mut chunked = load(&model, CTX);
        let (mut all, mut at) = (Vec::new(), 0);
        for len in [3usize, 1, 5, 2] {
            let step = chunked
                .forward_ext(
                    &tokens[at..at + len],
                    at,
                    Want::Last,
                    Hidden::Layers(set),
                    true,
                )
                .unwrap();
            all.extend(step.hidden.unwrap());
            at += len;
        }
        assert_eq!(all, rows);
    }

    // ---- the plain-loop reference -----------------------------------------

    /// Every tensor of a drafter file as f32.
    fn read_weights(drafter: &TinyModel) -> HashMap<String, Vec<f32>> {
        let mut reader = BufReader::new(File::open(&drafter.path).unwrap());
        let content = gguf_file::Content::read(&mut reader).unwrap();
        let names: Vec<String> = content.tensor_infos.keys().cloned().collect();
        names
            .into_iter()
            .map(|n| {
                let t = content
                    .tensor(&mut reader, &n, &Device::Cpu)
                    .unwrap()
                    .dequantize(&Device::Cpu)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap();
                (n, t)
            })
            .collect()
    }

    /// One drafting round computed with plain loops from the file's weights:
    /// no candle op, no cache, every context row projected on the spot.
    struct Reference {
        w: HashMap<String, Vec<f32>>,
        geo: Geometry,
    }

    fn rms_ref(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
        let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
        let s = 1.0 / (ms + eps).sqrt();
        x.iter().zip(w).map(|(v, w)| v * s * w).collect()
    }

    impl Reference {
        fn new(drafter: &TinyModel) -> Self {
            let content =
                gguf_file::Content::read(&mut BufReader::new(File::open(&drafter.path).unwrap()))
                    .unwrap();
            Self {
                w: read_weights(drafter),
                geo: read_geometry(&content.metadata).unwrap(),
            }
        }

        fn matvec(&self, name: &str, x: &[f32]) -> Vec<f32> {
            self.w[name]
                .chunks_exact(x.len())
                .map(|row| row.iter().zip(x).map(|(a, b)| a * b).sum())
                .collect()
        }

        /// Per-head weighted RMS norm.
        fn head_norm(&self, name: &str, x: &[f32]) -> Vec<f32> {
            x.chunks(self.geo.head_dim)
                .flat_map(|h| rms_ref(h, &self.w[name], self.geo.eps))
                .collect()
        }

        /// NEOX RoPE of every head of `x` at `pos`.
        fn rope(&self, x: &mut [f32], pos: usize) {
            let d = self.geo.head_dim;
            for head in x.chunks_mut(d) {
                for i in 0..d / 2 {
                    let theta = pos as f64 * self.geo.rope_base.powf(-((2 * i) as f64) / d as f64);
                    let (c, s) = (theta.cos() as f32, theta.sin() as f32);
                    let (a, b) = (head[i], head[i + d / 2]);
                    head[i] = a * c - b * s;
                    head[i + d / 2] = a * s + b * c;
                }
            }
        }

        /// The logits rows `1..=k` of the block after `feats.len()` context
        /// positions.
        fn round(
            &self,
            target: &Transformer,
            feats: &[Vec<f32>],
            first: u32,
            k: usize,
        ) -> Vec<f32> {
            let g = &self.geo;
            let (len, b, d) = (feats.len(), k + 1, g.head_dim);
            let (nh, nkv) = (g.n_head, g.n_kv_head);
            let blk = |il: usize, part: &str| format!("blk.{il}.{part}.weight");

            // Context rows and their K/V per layer.
            let ctx: Vec<Vec<f32>> = feats
                .iter()
                .map(|f| rms_ref(&self.matvec("fc.weight", f), &self.w[ENC_NORM], g.eps))
                .collect();
            let mut ck = vec![Vec::new(); g.windows.len()];
            let mut cv = vec![Vec::new(); g.windows.len()];
            for il in 0..g.windows.len() {
                for (p, c) in ctx.iter().enumerate() {
                    let k_proj = self.matvec(&blk(il, "attn_k"), c);
                    let mut kp = self.head_norm(&blk(il, "attn_k_norm"), &k_proj);
                    self.rope(&mut kp, p);
                    ck[il].push(kp);
                    cv[il].push(self.matvec(&blk(il, "attn_v"), c));
                }
            }

            // The block.
            let mut ids = vec![g.mask_token; b];
            ids[0] = first;
            let mut x: Vec<Vec<f32>> = ids
                .iter()
                .map(|&t| target.embedding_row(t).unwrap())
                .collect();
            for (il, &window) in g.windows.iter().enumerate() {
                let h: Vec<Vec<f32>> = x
                    .iter()
                    .map(|r| rms_ref(r, &self.w[&blk(il, "attn_norm")], g.eps))
                    .collect();
                let mut q = Vec::new();
                let (mut kb, mut vb) = (Vec::new(), Vec::new());
                for (i, hi) in h.iter().enumerate() {
                    let mut qi = self.head_norm(
                        &blk(il, "attn_q_norm"),
                        &self.matvec(&blk(il, "attn_q"), hi),
                    );
                    self.rope(&mut qi, len + i);
                    q.push(qi);
                    let mut ki = self.head_norm(
                        &blk(il, "attn_k_norm"),
                        &self.matvec(&blk(il, "attn_k"), hi),
                    );
                    self.rope(&mut ki, len + i);
                    kb.push(ki);
                    vb.push(self.matvec(&blk(il, "attn_v"), hi));
                }
                let allowed = |qp: usize, kp: usize| window.is_none_or(|w| qp.abs_diff(kp) <= w);
                for i in 0..b {
                    let mut attn = vec![0f32; nh * d];
                    for head in 0..nh {
                        let kv = head / (nh / nkv);
                        let qv = &q[i][head * d..(head + 1) * d];
                        let mut scores = Vec::new();
                        let mut values: Vec<&[f32]> = Vec::new();
                        for p in 0..len {
                            if allowed(len + i, p) {
                                let kr = &ck[il][p][kv * d..(kv + 1) * d];
                                scores.push(qv.iter().zip(kr).map(|(a, b)| a * b).sum::<f32>());
                                values.push(&cv[il][p][kv * d..(kv + 1) * d]);
                            }
                        }
                        for j in 0..b {
                            if allowed(len + i, len + j) {
                                let kr = &kb[j][kv * d..(kv + 1) * d];
                                scores.push(qv.iter().zip(kr).map(|(a, b)| a * b).sum::<f32>());
                                values.push(&vb[j][kv * d..(kv + 1) * d]);
                            }
                        }
                        let scale = 1.0 / (d as f32).sqrt();
                        let top = scores.iter().fold(f32::MIN, |m, s| m.max(s * scale));
                        let exps: Vec<f32> =
                            scores.iter().map(|s| (s * scale - top).exp()).collect();
                        let sum: f32 = exps.iter().sum();
                        for (e, v) in exps.iter().zip(&values) {
                            for (o, x) in attn[head * d..(head + 1) * d].iter_mut().zip(*v) {
                                *o += e / sum * x;
                            }
                        }
                    }
                    let o = self.matvec(&blk(il, "attn_output"), &attn);
                    for (xi, oi) in x[i].iter_mut().zip(&o) {
                        *xi += oi;
                    }
                    let f = rms_ref(&x[i], &self.w[&blk(il, "ffn_norm")], g.eps);
                    let gate = self.matvec(&blk(il, "ffn_gate"), &f);
                    let up = self.matvec(&blk(il, "ffn_up"), &f);
                    let act: Vec<f32> = gate
                        .iter()
                        .zip(&up)
                        .map(|(g, u)| g / (1.0 + (-g).exp()) * u)
                        .collect();
                    let down = self.matvec(&blk(il, "ffn_down"), &act);
                    for (xi, di) in x[i].iter_mut().zip(&down) {
                        *xi += di;
                    }
                }
            }
            let normed: Vec<f32> = x[1..=k]
                .iter()
                .flat_map(|r| rms_ref(r, &self.w["output_norm.weight"], g.eps))
                .collect();
            target
                .head(&Tensor::from_vec(normed, (1, k, g.n_embd), &Device::Cpu).unwrap())
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        }
    }

    /// One block after `len` random context rows equals the plain-loop
    /// reference, for sliding windows shorter than, equal to and longer than
    /// the context, a window shorter than the block, full attention, and
    /// every `k`.
    #[test]
    fn a_block_matches_the_plain_loop_reference() {
        let target_model = muse();
        let target = load(&target_model, CTX);
        let cases = [
            // (window, block, len)
            (Some(6), 8, 20),
            (Some(6), 8, 4),
            (Some(6), 4, 30),
            (Some(40), 8, 20),
            (Some(3), 8, 20),
            (None, 8, 20),
            (Some(8), 16, 25),
        ];
        let target_ids = vec![0usize, 2];
        for (window, block, len) in cases {
            let spec = TinyDflash {
                window,
                block,
                target_ids: target_ids.clone(),
                dtype: candle_core::quantized::GgmlDType::F32,
                ..TinyDflash::default()
            };
            let drafter = write_tiny_dflash(&spec);
            let mut net = open_net(&target, &drafter, CTX, true);
            let reference = Reference::new(&drafter);
            let w = net.feature_width();
            let feats = features(&net, len, 7 + len as u64);
            // Observed in two calls: a ring that wraps and a replay.
            let cut = len / 2;
            net.observe(0, &feats[..cut * w]).unwrap();
            net.observe(cut, &feats[cut * w..]).unwrap();
            let rows: Vec<Vec<f32>> = feats.chunks(w).map(<[f32]>::to_vec).collect();
            for k in [1, block - 1] {
                let want = reference.round(&target, &rows, 5, k);
                let got: Vec<f32> = draft_logits(&net, &target, len, 5, k).concat();
                assert_eq!(got.len(), want.len());
                let scale = want.iter().fold(1f32, |m, v| m.max(v.abs()));
                let worst = got
                    .iter()
                    .zip(&want)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                assert!(
                    worst <= 2e-3 * scale,
                    "window {window:?} block {block} len {len} k {k}: max |dlogit| {worst} (scale {scale})"
                );
            }
        }
    }

    /// A sliding window changes the block's logits where it should: a
    /// context longer than the window makes the first row's view differ from
    /// full attention, a context inside it does not (the block is shorter
    /// than the window, so its own rows see each other either way).
    #[test]
    fn the_window_restricts_only_what_it_reaches() {
        let target = load(&muse(), CTX);
        let run = |window: Option<usize>, len: usize| {
            let drafter = write_tiny_dflash(&TinyDflash {
                window,
                block: 4,
                ..TinyDflash::default()
            });
            let mut net = open_net(&target, &drafter, CTX, false);
            let feats = features(&net, len, 3);
            net.observe(0, &feats).unwrap();
            draft_logits(&net, &target, len, 5, 3)
        };
        assert_eq!(run(Some(6), 3), run(None, 3), "inside the window");
        assert_ne!(run(Some(6), 20), run(None, 20), "past the window");
        assert_ne!(
            run(Some(6), 20),
            run(Some(7), 20),
            "one position of the window is felt"
        );
    }

    // ---- the context cache -------------------------------------------------

    /// Chunked observation equals one call, bit for bit; replays and
    /// truncation drop exactly what they supersede; a draft needs every
    /// position up to the pending token.
    #[test]
    fn the_context_cache_follows_the_trunk() {
        let target = load(&muse(), CTX);
        let drafter = write_tiny_dflash(&TinyDflash::default());
        let mut whole = open_net(&target, &drafter, CTX, false);
        let w = whole.feature_width();
        let feats = features(&whole, 30, 5);
        whole.observe(0, &feats).unwrap();
        // The window of 6 holds the last six positions.
        assert_eq!(whole.held(), (24, 30));
        let reference = draft_logits(&whole, &target, 30, 7, 7);
        assert_eq!(reference.len(), 7);

        let mut net = open_net(&target, &drafter, CTX, false);
        for (at, n) in [(0usize, 7usize), (7, 1), (8, 9), (17, 13)] {
            net.observe(at, &feats[at * w..(at + n) * w]).unwrap();
        }
        assert_eq!(net.held(), (24, 30));
        assert_eq!(draft_logits(&net, &target, 30, 7, 7), reference);

        // A new prompt shares 20 tokens: the rows below 20 stay, the rest
        // are replayed.
        net.truncate(20);
        assert_eq!(net.held(), (20, 20), "the window's rows were all above 20");
        net.observe(20, &feats[20 * w..]).unwrap();
        assert_eq!(net.held(), (24, 30));
        assert_eq!(draft_logits(&net, &target, 30, 7, 7), reference);

        // A replay that starts inside the held range replaces its tail.
        net.observe(27, &feats[27 * w..]).unwrap();
        assert_eq!(net.held(), (24, 30));
        assert_eq!(draft_logits(&net, &target, 30, 7, 7), reference);
        net.truncate(27);
        assert_eq!(net.held(), (24, 27));
        net.observe(27, &feats[27 * w..]).unwrap();
        assert_eq!(draft_logits(&net, &target, 30, 7, 7), reference);

        // Without the rows up to the pending token there is nothing to draft
        // from.
        let none = |net: &DflashNet, len: usize, k: usize| {
            net.draft(&target, 7, len, k, &mut |_| Some(0)).unwrap()
        };
        assert!(none(&net, 29, 3).is_empty(), "one row short");
        assert!(none(&net, 31, 3).is_empty(), "one row ahead");
        assert!(none(&net, 30, 0).is_empty(), "no drafts asked for");
        assert_eq!(none(&net, 30, 3).len(), 3);
        // A block yields at most `block_size - 1` drafts.
        assert_eq!(none(&net, 30, 100).len(), 7);
        // `pick` ends the chain.
        let mut calls = 0;
        let drafts = net
            .draft(&target, 7, 30, 5, &mut |_| {
                calls += 1;
                (calls < 3).then_some(4)
            })
            .unwrap();
        assert_eq!((drafts, calls), (vec![4, 4], 3));

        // A gap: the rows after it start a new range.
        net.observe(40, &feats[..2 * w]).unwrap();
        assert_eq!(net.held(), (40, 42));
        assert!(none(&net, 30, 3).is_empty());
        assert_eq!(none(&net, 42, 3).len(), 3);

        // A malformed call leaves the range alone; reset forgets it.
        assert!(net.observe(42, &feats[..w - 1]).is_err());
        assert!(
            net.observe(CTX - 1, &feats[..2 * w]).is_err(),
            "beyond the context"
        );
        assert_eq!(net.held(), (40, 42));
        net.reset();
        assert_eq!(net.held(), (0, 0));
        assert!(none(&net, 0, 3).is_empty());
    }

    /// A drafter whose weights are held as f16 or Q8_0 loads and drafts like
    /// the quantized ones, and a draft is deterministic.
    #[test]
    fn drafting_is_deterministic_for_every_weight_dtype() {
        use candle_core::quantized::GgmlDType;
        let target = load(&muse(), CTX);
        for dtype in [
            GgmlDType::F32,
            GgmlDType::F16,
            GgmlDType::Q8_0,
            GgmlDType::Q4_0,
        ] {
            let drafter = write_tiny_dflash(&TinyDflash {
                dtype,
                ..TinyDflash::default()
            });
            let mut net = open_net(&target, &drafter, CTX, false);
            let feats = features(&net, 12, 9);
            net.observe(0, &feats).unwrap();
            let a = draft_logits(&net, &target, 12, 3, 5);
            assert_eq!(a, draft_logits(&net, &target, 12, 3, 5), "{dtype:?}");
            assert!(a.iter().flatten().all(|v| v.is_finite()), "{dtype:?}");
        }
    }

    // ---- loading -----------------------------------------------------------

    fn open_err(target: &Transformer, spec: &TinyDflash) -> String {
        let drafter = write_tiny_dflash(spec);
        let content =
            gguf_file::Content::read(&mut BufReader::new(File::open(&drafter.path).unwrap()))
                .unwrap();
        DflashNet::open(&content, &drafter.path, target, CTX)
            .err()
            .expect("must be refused")
            .to_string()
    }

    #[test]
    fn a_drafter_that_does_not_fit_its_target_is_refused() {
        let target = load(&muse(), CTX);
        let e = open_err(
            &target,
            &TinyDflash {
                target_ids: vec![0, 9],
                ..TinyDflash::default()
            },
        );
        assert!(e.contains("layer 9") && e.contains("4 layers"), "{e}");
        let e = open_err(
            &target,
            &TinyDflash {
                target_ids: vec![2, 1],
                ..TinyDflash::default()
            },
        );
        assert!(e.contains("ascending"), "{e}");
        let e = open_err(
            &target,
            &TinyDflash {
                block: 1,
                ..TinyDflash::default()
            },
        );
        assert!(e.contains("inconsistent geometry"), "{e}");
        let e = open_err(
            &target,
            &TinyDflash {
                meta: vec![("dflash.selector_rank".into(), Value::U32(2))],
                ..TinyDflash::default()
            },
        );
        assert!(e.contains("selector_rank") && e.contains("variant"), "{e}");
        let e = open_err(
            &target,
            &TinyDflash {
                meta: vec![("tokenizer.ggml.mask_token_id".into(), Value::U32(99))],
                ..TinyDflash::default()
            },
        );
        assert!(e.contains("mask token 99"), "{e}");
    }

    /// The tiny drafter's metadata is what the converter checks.
    #[test]
    fn the_geometry_follows_the_metadata() {
        let drafter = write_tiny_dflash(&TinyDflash::default());
        let content =
            gguf_file::Content::read(&mut BufReader::new(File::open(&drafter.path).unwrap()))
                .unwrap();
        let g = read_geometry(&content.metadata).unwrap();
        assert_eq!(
            (g.n_embd, g.n_head, g.n_kv_head, g.head_dim, g.block_size),
            (N_EMBD, DFLASH_HEADS, DFLASH_KV_HEADS, DFLASH_HEAD_DIM, 8)
        );
        // The file holds llama.cpp's `inputs of layer i + 1`; the loader reads
        // the outputs of layers 0 and 2.
        assert_eq!(g.features, [0, 2]);
        assert_eq!(g.windows, [Some(6), Some(6)]);
        assert_eq!(g.mask_token, DFLASH_MASK);
        assert!(check_metadata(&content.metadata).is_ok());
    }
}
