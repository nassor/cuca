//! The DSpark block drafter of DeepSeek-V4.1 (a `dflash` GGUF with
//! `dflash.hyper_connection.count`).
//!
//! DSpark drafts a block of `block_size` (5) tokens in one pass from the
//! target's hidden features. Its stages are blocks of the target's own kind:
//! hyper-connection streams, latent attention with per-head sinks and a
//! grouped low-rank output, and a `sqrtsoftplus`-routed mixture of experts
//! with a SwiGLU clamp, but every attention is window-only (no compressed rows,
//! no indexer) and the block rows attend to each other in both directions.
//! It owns no embedding table and no head: the block reads the target's, and
//! a small Markov head adds a logit bias that depends on the previous draft.
//!
//! # One round
//!
//! `len` positions are cached in the target and the pending token `first`
//! (sampled from the target's last logits) sits at position `len`. The target
//! hands over, for every position it computed, the mean over its streams of
//! the input of the layers `dflash.target_layers` ([`Hidden::Layers`]), which
//! `main_x = norm(fc * concat(features))` turns into one row per position.
//! Every stage keeps the key (`kv_norm(wkv(main_x))`, rotated, FP8-simulated)
//! of the last `sliding_window` positions in a ring ([`DsparkNet::observe`]).
//! The block rows are `[first, noise, noise, noise, noise]` embedded with the
//! target's table, repeated over the streams, at positions `len..len + 5`.
//! Each stage attends, per block row, to every ring row and to all block rows,
//! then runs its MoE; the stages share one mixing chain (`pre_mix` starts at
//! the identity and is carried over). After the last stage the block collapses
//! (`hc_pre`), goes through `output_norm` and the target's head for the logits
//! of 5 rows. Row `i` predicts the token after block row `i`:
//! `logits_i += markov_w2 * markov_w1[out_i]` with `out_0 = first` and
//! `out_{i+1}` the token chosen from the biased logits, so the draft chain
//! runs through the caller's `pick`. `conf_proj * [hidden_i ; markov_w1[out_i]]`
//! is the drafter's own estimate for row `i`'s acceptance.
//!
//! The block always runs its 5 rows (a shorter block would change the
//! bidirectional attention); `k` drafts are the first `k` rows' tokens.
//!
//! # Context ring
//!
//! As the DFlash drafter's: position `p` of a stage lives in slot `p % window`,
//! the held positions are `lo..hi`, observing at `pos < hi` drops what it
//! replaces, [`DsparkNet::truncate`] keeps the positions below `shared`, and a
//! draft needs `hi == len`. A context shorter than it should be costs proposal
//! quality only: verification keeps the output equal to plain decoding.
//!
//! # Memory
//!
//! The stages' experts (128 per stage, MXFP4, 7.2 GB in the real file) are
//! resident: they are read once when the drafter opens, and the file is
//! separate from the target's, so no tier is involved. The rings are
//! `stages * window * head_dim` floats.

use std::sync::Arc;

use crate::provider::local::gguf::{self, Value};

use super::super::arch::Arch;
use super::super::csa::{Gating, HyperSpec, read_hyper};
use super::super::hc::{self, HcWeights, coeff_len, coefficients, collapse, dot, expand};
use super::super::kvquant;
use super::super::layout::{name, tensor_info};
use super::super::params::{Keys, Metadata, MoeSpec, value_usize};
use super::super::pool::{par_chunks, par_chunks_with};
use super::super::quant::TokenEmbedding;
use super::super::rope::{RopeKind, RopeSpec};
use super::super::{candle_err, err, meta_err};
use super::hyper::{Rope, load_hc, mm, qdq_disabled, rms_row, row_groups};
use super::{
    CpuExpert, FfnWeights, LayerSet, Loader, QMatMul, RouterExt, SharedExpert, Transformer,
};
use crate::error::CucaError;
use candle_core::{Device, Tensor};

/// `general.architecture` of the file: the DFlash name, told apart by
/// [`is_dspark`].
const PREFIX: &str = super::dflash::ARCH;

/// Whether `md` describes a DSpark drafter (a `dflash` file with
/// hyper-connection streams) rather than a plain DFlash one.
pub(crate) fn is_dspark(md: &Metadata) -> bool {
    md.contains_key(&format!("{PREFIX}.hyper_connection.count"))
}

/// What the `dflash.*` metadata says.
#[derive(Debug, Clone)]
struct Geometry {
    n_embd: usize,
    n_head: usize,
    head_dim: usize,
    rope_dims: usize,
    rope_base: f64,
    window: usize,
    q_lora: usize,
    out_groups: usize,
    out_rank: usize,
    shared_ff: usize,
    expert_ff: usize,
    n_expert: usize,
    n_used: usize,
    norm_weights: bool,
    expert_scale: f32,
    gating: Gating,
    clamp: Vec<f32>,
    clamp_shared: Vec<f32>,
    eps: f32,
    n_stage: usize,
    block_size: usize,
    mask_token: u32,
    features: Vec<usize>,
}

fn array(md: &Metadata, key: &str) -> Result<Option<Vec<Value>>, CucaError> {
    match md.get(key) {
        None => Ok(None),
        Some(Value::Array(items)) => Ok(Some(items.clone())),
        Some(_) => Err(meta_err(key, "not an array")),
    }
}

fn usizes(md: &Metadata, key: &str) -> Result<Option<Vec<usize>>, CucaError> {
    array(md, key)?
        .map(|items| {
            items
                .iter()
                .map(|v| value_usize(v).ok_or_else(|| meta_err(key, "not an unsigned integer")))
                .collect()
        })
        .transpose()
}

fn floats(md: &Metadata, key: &str, n: usize) -> Result<Vec<f32>, CucaError> {
    let Some(items) = array(md, key)? else {
        return Ok(vec![0.0; n]);
    };
    let v = items
        .iter()
        .map(|v| match v {
            Value::F32(x) => Ok(*x),
            Value::F64(x) => Ok(*x as f32),
            _ => Err(meta_err(key, "not a float")),
        })
        .collect::<Result<Vec<_>, _>>()?;
    if v.len() != n {
        return Err(meta_err(key, format!("{} entries for {n} stages", v.len())));
    }
    Ok(v)
}

fn read_geometry(md: &Metadata) -> Result<(Geometry, HyperSpec), CucaError> {
    let k = Keys::new(md, PREFIX);
    let n_stage = k.usize("block_count")?;
    let n_embd = k.usize("embedding_length")?;
    let n_head = k.usize("attention.head_count")?;
    let head_dim = k.usize("attention.key_length")?;
    if k.usize_opt("attention.head_count_kv")?
        .is_some_and(|n| n != 1)
        || k.usize_opt("attention.value_length")?
            .is_some_and(|n| n != head_dim)
    {
        return Err(err(
            "a DSpark drafter needs one K/V head and equal key and value lengths".to_string(),
        ));
    }
    let rope_dims = k.usize("rope.dimension_count")?;
    let window = k.usize("attention.sliding_window")?;
    let block_size = k.usize("block_size")?;
    let ratios = usizes(md, &format!("{PREFIX}.attention.compress_ratios"))?;
    if ratios.is_some_and(|r| r.iter().any(|&x| x != 0)) {
        return Err(err(
            "a DSpark drafter attends to a window only: its compress ratios must be 0".to_string(),
        ));
    }
    if n_stage == 0
        || n_embd == 0
        || n_head == 0
        || head_dim == 0
        || window == 0
        || block_size < 2
        || rope_dims == 0
        || rope_dims > head_dim
        || !rope_dims.is_multiple_of(2)
    {
        return Err(err(format!(
            "`{PREFIX}` metadata: inconsistent DSpark geometry ({n_stage} stages, {n_head} heads of {head_dim}, window {window}, blocks of {block_size})"
        )));
    }
    let target_key = format!("{PREFIX}.target_layers");
    // llama.cpp lists the layers whose *input* feeds the drafter; a
    // hyper-connection target hands those inputs over as they are.
    let features = usizes(md, &target_key)?
        .ok_or_else(|| meta_err(&target_key, "missing: the drafter needs its target layers"))?;
    if features.is_empty() || features.windows(2).any(|w| w[0] >= w[1]) {
        return Err(meta_err(
            &target_key,
            "the target layers must be a non-empty ascending list",
        ));
    }
    let mask_key = "tokenizer.ggml.mask_token_id";
    let mask_token = md
        .get(mask_key)
        .and_then(value_usize)
        .and_then(|id| u32::try_from(id).ok())
        .ok_or_else(|| meta_err(mask_key, "missing: the drafter's noise token"))?;
    let shared_ff = k.usize("feed_forward_length")?;
    let gating = match k.usize_opt("expert_gating_func")? {
        None | Some(1) => Gating::Softmax,
        Some(2) => Gating::Sigmoid,
        Some(4) => Gating::SqrtSoftplus,
        Some(other) => {
            return Err(meta_err(
                &format!("{PREFIX}.expert_gating_func"),
                format!("gating function {other} is not supported"),
            ));
        }
    };
    let geo = Geometry {
        n_embd,
        n_head,
        head_dim,
        rope_dims,
        rope_base: k.f64_opt("rope.freq_base")?.unwrap_or(10_000.0),
        window,
        q_lora: k.usize("attention.q_lora_rank")?,
        out_groups: k.usize("attention.output_group_count")?,
        out_rank: k.usize("attention.output_lora_rank")?,
        shared_ff,
        expert_ff: k
            .usize_opt("expert_feed_forward_length")?
            .unwrap_or(shared_ff),
        n_expert: k.usize("expert_count")?,
        n_used: k.usize("expert_used_count")?,
        norm_weights: k.bool_opt("expert_weights_norm")?.unwrap_or(true),
        expert_scale: k.f64_opt("expert_weights_scale")?.unwrap_or(1.0) as f32,
        gating,
        clamp: floats(md, &format!("{PREFIX}.swiglu_clamp_exp"), n_stage)?,
        clamp_shared: floats(md, &format!("{PREFIX}.swiglu_clamp_shexp"), n_stage)?,
        eps: k.f64("attention.layer_norm_rms_epsilon")? as f32,
        n_stage,
        block_size,
        mask_token,
        features,
    };
    if geo.q_lora == 0
        || geo.out_groups == 0
        || geo.out_rank == 0
        || !(n_head * head_dim).is_multiple_of(geo.out_groups)
        || geo.n_expert == 0
        || geo.n_used == 0
        || geo.n_used > geo.n_expert
    {
        return Err(err(format!(
            "`{PREFIX}` metadata: inconsistent DSpark projections or experts"
        )));
    }
    let spec = read_hyper(&k, Arch::Deepseek41)?;
    Ok((geo, spec))
}

/// One stage: a hyper-connection block over a window-only attention.
struct Stage {
    hc_attn: HcWeights,
    hc_ffn: HcWeights,
    attn_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    q_a: QMatMul,
    q_a_norm: Vec<f32>,
    q_b: QMatMul,
    kv: QMatMul,
    kv_norm: Vec<f32>,
    /// One `[rank, heads * head_dim / groups]` projection per output group.
    out_a: Vec<QMatMul>,
    out_b: QMatMul,
    sinks: Vec<f32>,
    ffn: FfnWeights,
}

/// A DSpark drafter and its context rings; see the [module docs](self).
pub(crate) struct DsparkNet {
    geo: Geometry,
    spec: HyperSpec,
    set: LayerSet,
    /// Floats of one feature row: the target layers' rows one after the other.
    feature_width: usize,
    fc: QMatMul,
    enc_norm: Vec<f32>,
    out_norm: Vec<f32>,
    stages: Vec<Stage>,
    markov_in: TokenEmbedding,
    markov_out: QMatMul,
    markov_rank: usize,
    /// `[n_embd + markov_rank]`: the confidence head.
    conf: Vec<f32>,
    rope: Rope,
    qdq: bool,
    /// Slots of each stage's ring.
    cap: usize,
    lo: usize,
    hi: usize,
    /// Per stage, the keys `[cap][head_dim]`.
    ring: Vec<Vec<f32>>,
}

impl DsparkNet {
    /// Load the drafter in `content` for `target`, whose positions go up to
    /// `context_length`.
    ///
    /// # Errors
    ///
    /// [`CucaError::Provider`] for a file that is not a DSpark drafter
    /// (missing keys or tensors, shapes that do not fit), a target that is not
    /// a hyper-connection model of the same width and stream count, a target
    /// layer the target does not have, or a noise token outside its
    /// vocabulary.
    pub(crate) fn open(
        content: &gguf::Content,
        target: &Transformer,
        context_length: usize,
    ) -> Result<Self, CucaError> {
        let (geo, spec) = read_geometry(&content.metadata)?;
        let t = &target.hp;
        match t.hyper {
            Some(h) if h.streams == spec.streams => {}
            _ => {
                return Err(err(format!(
                    "the DSpark drafter pairs with a target of {} hyper-connection streams",
                    spec.streams
                )));
            }
        }
        if geo.n_embd != t.n_embd {
            return Err(err(format!(
                "the DSpark drafter's width {} differs from the target's {}: it was trained for another model size",
                geo.n_embd, t.n_embd
            )));
        }
        if let Some(&last) = geo.features.last()
            && last >= t.n_layer
        {
            return Err(err(format!(
                "the DSpark drafter reads layer {last}, but the target has {} layers",
                t.n_layer
            )));
        }
        if geo.mask_token as usize >= target.vocab {
            return Err(err(format!(
                "the DSpark noise token {} is outside the target's vocabulary of {}",
                geo.mask_token, target.vocab
            )));
        }
        if target.embed.is_none() {
            return Err(err(
                "the DSpark drafter needs a target with a token embedding",
            ));
        }
        let set = LayerSet::new(geo.features.iter().copied())
            .ok_or_else(|| err("the DSpark drafter reads a layer beyond the 128th"))?;

        let mut loader = Loader {
            content,
            reader: content.reader(),
            dequantize: false,
            requantize: false,
        };
        let d = geo.n_embd;
        let feature_width = geo.features.len() * t.n_embd;
        let fc = loader.matmul("fc.weight", &[d, feature_width])?;
        let enc_norm = loader.floats("enc.output_norm.weight", &[d])?;
        let out_norm = loader.floats(name::OUTPUT_NORM, &[d])?;
        let markov_info = tensor_info(content, "markov_w1.weight")?;
        let &[vocab, markov_rank] = markov_info.shape.dims() else {
            return Err(err("tensor `markov_w1.weight` is not a matrix"));
        };
        if vocab != target.vocab {
            return Err(err(format!(
                "the DSpark Markov head covers {vocab} tokens, the target's vocabulary has {}",
                target.vocab
            )));
        }
        let markov_in = TokenEmbedding::new(Arc::new(
            loader.qtensor("markov_w1.weight", &[vocab, markov_rank])?,
        ))?;
        let markov_out = loader.matmul("markov_w2.weight", &[vocab, markov_rank])?;
        let conf = loader.floats("conf_proj.weight", &[d + markov_rank])?;

        let mut stages = Vec::with_capacity(geo.n_stage);
        for il in 0..geo.n_stage {
            stages.push(load_stage(&mut loader, &geo, &spec, il)?);
        }

        let positions = context_length + geo.block_size;
        let table =
            RopeSpec::plain(RopeKind::Norm, geo.rope_dims, geo.rope_base).tables(None, positions);
        let cap = geo.window.min(context_length).max(1);
        Ok(Self {
            ring: vec![vec![0.0; cap * geo.head_dim]; geo.n_stage],
            rope: Rope {
                half: table.half,
                cos: table.cos,
                sin: table.sin,
            },
            qdq: !qdq_disabled(),
            geo,
            spec,
            set,
            feature_width,
            fc,
            enc_norm,
            out_norm,
            stages,
            markov_in,
            markov_out,
            markov_rank,
            conf,
            cap,
            lo: 0,
            hi: 0,
        })
    }

    /// The target layers whose inputs the drafter reads, as the set to ask a
    /// forward pass for ([`Hidden::Layers`](super::Hidden)).
    pub(crate) fn features(&self) -> LayerSet {
        self.set
    }

    /// Floats of one feature row.
    pub(crate) fn feature_width(&self) -> usize {
        self.feature_width
    }

    /// The positions the rings hold, as `lo..hi`.
    #[cfg(test)]
    pub(crate) fn held(&self) -> (usize, usize) {
        (self.lo, self.hi)
    }

    /// Keep the positions below `shared` and forget the rest.
    pub(crate) fn truncate(&mut self, shared: usize) {
        self.hi = self.hi.min(shared);
        self.lo = self.lo.min(self.hi);
    }

    /// Take the feature rows `feats` of the positions `pos..` into the rings.
    ///
    /// # Errors
    ///
    /// [`CucaError::Provider`] for a length that is not whole feature rows,
    /// positions beyond the context or a failed projection.
    pub(crate) fn observe(&mut self, pos: usize, feats: &[f32]) -> Result<(), CucaError> {
        let width = self.feature_width;
        if feats.is_empty() || !feats.len().is_multiple_of(width) {
            return Err(err(format!(
                "DSpark observe: {} floats are not whole rows of {width}",
                feats.len()
            )));
        }
        let rows = feats.len() / width;
        if (pos + rows + self.geo.block_size) * self.rope.half > self.rope.cos.len() {
            return Err(err(format!(
                "DSpark observe: positions up to {} exceed the context",
                pos + rows
            )));
        }
        // Rows that would leave the ring at once are not projected.
        let skip = rows.saturating_sub(self.cap);
        let first = pos + skip;
        let m = rows - skip;
        let (d, hd) = (self.geo.n_embd, self.geo.head_dim);
        let main = self.main_rows(&feats[skip * width..]).map_err(candle_err)?;
        let mut keys = Vec::with_capacity(self.stages.len());
        for st in &self.stages {
            let mut kv = mm(&st.kv, &main, d).map_err(candle_err)?;
            let mut row = vec![0f32; hd];
            for (r, k) in kv.chunks_exact_mut(hd).enumerate() {
                rms_row(k, &st.kv_norm, self.geo.eps, &mut row);
                self.rope.tail(&mut row, first + r, false);
                if self.qdq {
                    kvquant::qdq_fp8_ue8m0(&mut row, 32);
                }
                k.copy_from_slice(&row);
            }
            keys.push(kv);
        }
        // Only now does the held range change.
        if pos < self.hi {
            self.hi = pos;
            self.lo = self.lo.min(pos);
        } else if pos > self.hi || self.hi == self.lo {
            self.lo = pos;
            self.hi = pos;
        }
        for (ring, kv) in self.ring.iter_mut().zip(&keys) {
            for (r, k) in kv.chunks_exact(hd).enumerate().take(m) {
                let slot = (first + r) % self.cap;
                ring[slot * hd..(slot + 1) * hd].copy_from_slice(k);
            }
        }
        self.hi = pos + rows;
        self.lo = self.lo.max(self.hi.saturating_sub(self.cap));
        Ok(())
    }

    /// `main_x = norm(fc * features)` of every feature row.
    fn main_rows(&self, feats: &[f32]) -> candle_core::Result<Vec<f32>> {
        let d = self.geo.n_embd;
        let mut x = mm(&self.fc, feats, self.feature_width)?;
        let mut row = vec![0f32; d];
        for r in x.chunks_exact_mut(d) {
            rms_row(r, &self.enc_norm, self.geo.eps, &mut row);
            r.copy_from_slice(&row);
        }
        Ok(x)
    }

    /// Propose at most `k` tokens after the pending token `first` of a
    /// sequence with `len` cached positions: one block pass, then each draft
    /// row's biased logits go to `pick`, which chooses the token or ends the
    /// chain with `None`. Nothing is cached: the block's own keys are dropped.
    ///
    /// Empty when the rings do not hold every position below `len` (nothing
    /// to condition the block on) or `k` is zero.
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
        self.draft_scored(target, first, len, k, pick)
            .map(|(drafts, _)| drafts)
    }

    /// [`DsparkNet::draft`] with the drafter's confidence for each row it
    /// produced a draft for (the raw logit of the acceptance estimate).
    ///
    /// # Errors
    ///
    /// As [`DsparkNet::draft`].
    pub(crate) fn draft_scored(
        &self,
        target: &Transformer,
        first: u32,
        len: usize,
        k: usize,
        pick: &mut dyn FnMut(&mut [f32]) -> Option<u32>,
    ) -> Result<(Vec<u32>, Vec<f32>), CucaError> {
        let k = k.min(self.geo.block_size);
        if k == 0 || len == 0 || self.hi != len || self.lo >= self.hi {
            return Ok((Vec::new(), Vec::new()));
        }
        let (n, d, s) = (self.geo.block_size, self.geo.n_embd, self.spec.streams);
        let mut ids = vec![self.geo.mask_token; n];
        ids[0] = first;
        let embeddings = target.embedding_rows(&ids)?;
        let mut stream = vec![0f32; n * s * d];
        for (row, e) in stream
            .chunks_exact_mut(s * d)
            .zip(embeddings.chunks_exact(d))
        {
            for copy in row.chunks_exact_mut(d) {
                copy.copy_from_slice(e);
            }
        }
        let mut pre_mix = vec![0f32; n * s];
        for row in pre_mix.chunks_exact_mut(s) {
            row[0] = 1.0;
        }
        for il in 0..self.stages.len() {
            self.stage(il, &mut stream, &mut pre_mix, len)
                .map_err(candle_err)?;
        }
        // The block collapsed by the last FFN's pre weights, then the
        // target's head over every row.
        let mut hidden = vec![0f32; n * d];
        for (r, out) in hidden.chunks_exact_mut(d).enumerate() {
            collapse(
                &stream[r * s * d..(r + 1) * s * d],
                &pre_mix[r * s..(r + 1) * s],
                out,
            );
        }
        let mut normed = vec![0f32; n * d];
        for (x, out) in hidden.chunks_exact(d).zip(normed.chunks_exact_mut(d)) {
            rms_row(x, &self.out_norm, self.geo.eps, out);
        }
        let normed = Tensor::from_vec(normed, (1, n, d), &Device::Cpu).map_err(candle_err)?;
        let mut logits = target
            .head(&normed)
            .and_then(|l| l.flatten_all()?.to_vec1::<f32>())
            .map_err(candle_err)?;
        let vocab = logits.len() / n;

        let mut drafts = Vec::with_capacity(k);
        let mut confidence = Vec::with_capacity(k);
        let mut previous = first;
        for (i, row) in logits.chunks_exact_mut(vocab).enumerate().take(k) {
            let embed = self.markov_in.rows(&[previous], 1.0)?;
            let bias = mm(&self.markov_out, &embed, self.markov_rank).map_err(candle_err)?;
            for (l, b) in row.iter_mut().zip(&bias) {
                *l += b;
            }
            confidence.push(
                dot(&self.conf[..d], &hidden[i * d..(i + 1) * d]) + dot(&self.conf[d..], &embed),
            );
            let Some(token) = pick(row) else {
                confidence.pop();
                break;
            };
            drafts.push(token);
            previous = token;
        }
        Ok((drafts, confidence))
    }

    /// One stage over the block rows of `stream` (`[rows][streams][n_embd]`):
    /// attention then FFN, each between its hyper-connection collapse and
    /// expansion; `pre_mix` is what the previous FFN predicted.
    fn stage(
        &self,
        il: usize,
        stream: &mut Vec<f32>,
        pre_mix: &mut [f32],
        len: usize,
    ) -> candle_core::Result<()> {
        let st = &self.stages[il];
        let (s, d) = (self.spec.streams, self.geo.n_embd);
        let t = stream.len() / (s * d);
        let eps = self.geo.eps;
        let cl = coeff_len(s);

        let mut attn_c = vec![0f32; t * cl];
        par_chunks(&mut attn_c, cl, |r, c| {
            coefficients(
                &stream[r * s * d..(r + 1) * s * d],
                &st.hc_attn,
                &self.spec,
                eps,
                c,
            );
        });
        let mut h = vec![0f32; t * d];
        par_chunks(&mut h, d, |r, out| {
            let mut x = vec![0f32; d];
            collapse(
                &stream[r * s * d..(r + 1) * s * d],
                &pre_mix[r * s..(r + 1) * s],
                &mut x,
            );
            rms_row(&x, &st.attn_norm, eps, out);
        });
        let y = self.attention(il, &h, len)?;
        let mut next = vec![0f32; t * s * d];
        par_chunks(&mut next, s * d, |r, out| {
            let c = &attn_c[r * cl..(r + 1) * cl];
            expand(
                &y[r * d..(r + 1) * d],
                &stream[r * s * d..(r + 1) * s * d],
                &c[s..2 * s],
                &c[2 * s..],
                out,
            );
        });

        let mut ffn_c = vec![0f32; t * cl];
        par_chunks(&mut ffn_c, cl, |r, c| {
            coefficients(
                &next[r * s * d..(r + 1) * s * d],
                &st.hc_ffn,
                &self.spec,
                eps,
                c,
            );
        });
        let mut f_in = vec![0f32; t * d];
        par_chunks(&mut f_in, d, |r, out| {
            let mut x = vec![0f32; d];
            collapse(
                &next[r * s * d..(r + 1) * s * d],
                hc::pre(&attn_c[r * cl..(r + 1) * cl], s),
                &mut x,
            );
            rms_row(&x, &st.ffn_norm, eps, out);
        });
        let f = st
            .ffn
            .forward(
                &Tensor::from_vec(f_in, (1, t, d), &Device::Cpu)?,
                super::super::arch::Act::Silu,
            )?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let mut out = vec![0f32; t * s * d];
        par_chunks(&mut out, s * d, |r, o| {
            let c = &ffn_c[r * cl..(r + 1) * cl];
            expand(
                &f[r * d..(r + 1) * d],
                &next[r * s * d..(r + 1) * s * d],
                &c[s..2 * s],
                &c[2 * s..],
                o,
            );
        });
        for (dst, c) in pre_mix.chunks_exact_mut(s).zip(ffn_c.chunks_exact(cl)) {
            dst.copy_from_slice(&c[..s]);
        }
        *stream = out;
        Ok(())
    }

    /// The attention of stage `il` over the normed block rows `h`
    /// `(rows, n_embd)` at positions `len..`: every row reads the ring (the
    /// `window` positions before `len`) and all block rows.
    fn attention(&self, il: usize, h: &[f32], len: usize) -> candle_core::Result<Vec<f32>> {
        let st = &self.stages[il];
        let g = &self.geo;
        let (d, nh, hd) = (g.n_embd, g.n_head, g.head_dim);
        let t = h.len() / d;
        let eps = g.eps;
        let rope = &self.rope;

        let qa = mm(&st.q_a, h, d)?;
        let mut qn = vec![0f32; t * g.q_lora];
        par_chunks(&mut qn, g.q_lora, |r, out| {
            rms_row(
                &qa[r * g.q_lora..(r + 1) * g.q_lora],
                &st.q_a_norm,
                eps,
                out,
            );
        });
        let mut q = mm(&st.q_b, &qn, g.q_lora)?;
        par_chunks(&mut q, nh * hd, |r, row| {
            for head in row.chunks_exact_mut(hd) {
                rope.tail(head, len + r, false);
            }
        });
        let mut kv = mm(&st.kv, h, d)?;
        par_chunks(&mut kv, hd, |r, row| {
            let mut v = vec![0f32; hd];
            rms_row(row, &st.kv_norm, eps, &mut v);
            rope.tail(&mut v, len + r, false);
            if self.qdq {
                kvquant::qdq_fp8_ue8m0(&mut v, 32);
            }
            row.copy_from_slice(&v);
        });

        let ring = &self.ring[il];
        let (lo, hi, cap) = (self.lo, self.hi, self.cap);
        let scale = (1.0 / (hd as f64).sqrt()) as f32;
        let mut o = vec![0f32; t * nh * hd];
        par_chunks_with(&mut o, hd, Vec::<f32>::new, |scores, idx, out| {
            let (r, head) = (idx / nh, idx % nh);
            let qv = &q[(r * nh + head) * hd..(r * nh + head + 1) * hd];
            scores.clear();
            for p in lo..hi {
                let slot = p % cap;
                scores.push(dot(qv, &ring[slot * hd..(slot + 1) * hd]) * scale);
            }
            for key in kv.chunks_exact(hd) {
                scores.push(dot(qv, key) * scale);
            }
            let m = scores
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max)
                .max(-1e30);
            let mut denom = 0.0;
            for sc in scores.iter_mut() {
                *sc = (*sc - m).exp();
                denom += *sc;
            }
            denom += (st.sinks[head] - m).exp();
            out.fill(0.0);
            let mut at = 0;
            for p in lo..hi {
                let slot = p % cap;
                let w = scores[at];
                for (o, &v) in out.iter_mut().zip(&ring[slot * hd..(slot + 1) * hd]) {
                    *o += w * v;
                }
                at += 1;
            }
            for key in kv.chunks_exact(hd) {
                let w = scores[at];
                for (o, &v) in out.iter_mut().zip(key) {
                    *o += w * v;
                }
                at += 1;
            }
            for v in out.iter_mut() {
                *v /= denom;
            }
        });

        // Undo the query's rotation, simulate the output's quantization and
        // project through the grouped low-rank output.
        par_chunks(&mut o, nh * hd, |r, row| {
            for head in row.chunks_exact_mut(hd) {
                rope.tail(head, len + r, true);
            }
            if self.qdq {
                kvquant::qdq_fp8_ue8m0(row, 32);
            }
        });
        let width = nh * hd / g.out_groups;
        let mut mid = vec![0f32; t * g.out_groups * g.out_rank];
        for (grp, w) in st.out_a.iter().enumerate() {
            let mut cols = vec![0f32; t * width];
            for (dst, row) in cols.chunks_exact_mut(width).zip(o.chunks_exact(nh * hd)) {
                dst.copy_from_slice(&row[grp * width..(grp + 1) * width]);
            }
            let part = mm(w, &cols, width)?;
            for (dst, src) in mid
                .chunks_exact_mut(g.out_groups * g.out_rank)
                .zip(part.chunks_exact(g.out_rank))
            {
                dst[grp * g.out_rank..(grp + 1) * g.out_rank].copy_from_slice(src);
            }
        }
        mm(&st.out_b, &mid, g.out_groups * g.out_rank)
    }
}

/// The weights of stage `il`.
fn load_stage(
    loader: &mut Loader<'_>,
    g: &Geometry,
    spec: &HyperSpec,
    il: usize,
) -> Result<Stage, CucaError> {
    let blk = |part: &str| name::blk(il, part);
    let d = g.n_embd;
    let (nh, hd) = (g.n_head, g.head_dim);
    let ffn = FfnWeights::Moe {
        spec: MoeSpec {
            n_expert: g.n_expert,
            n_used: g.n_used,
            norm_weights: g.norm_weights,
            scale: g.expert_scale,
        },
        ext: Some(Box::new(RouterExt {
            gating: g.gating,
            bias: Some(loader.floats(&name::blk_bias(il, name::EXP_PROBS_B), &[g.n_expert])?),
            clamp: g.clamp[il],
            clamp_shared: g.clamp_shared[il],
        })),
        router: loader.matmul(&blk(name::FFN_GATE_INP), &[g.n_expert, d])?,
        experts: loader.expert_bank(il, g.n_expert, g.expert_ff, d, false, None)?,
        shared: Some(SharedExpert {
            ffn: CpuExpert {
                gate: loader.matmul(&blk(name::FFN_GATE_SHEXP), &[g.shared_ff, d])?,
                up: loader.matmul(&blk(name::FFN_UP_SHEXP), &[g.shared_ff, d])?,
                down: loader.matmul(&blk(name::FFN_DOWN_SHEXP), &[d, g.shared_ff])?,
            },
            gate_inp: None,
        }),
    };
    Ok(Stage {
        hc_attn: load_hc(
            loader,
            d,
            il,
            spec,
            name::HC_ATTN_FN,
            name::HC_ATTN_BASE,
            name::HC_ATTN_SCALE,
        )?,
        hc_ffn: load_hc(
            loader,
            d,
            il,
            spec,
            name::HC_FFN_FN,
            name::HC_FFN_BASE,
            name::HC_FFN_SCALE,
        )?,
        attn_norm: loader.floats(&blk(name::ATTN_NORM), &[d])?,
        ffn_norm: loader.floats(&blk(name::FFN_NORM), &[d])?,
        q_a: loader.matmul(&blk(name::ATTN_Q_A), &[g.q_lora, d])?,
        q_a_norm: loader.floats(&blk(name::ATTN_Q_A_NORM), &[g.q_lora])?,
        q_b: loader.matmul(&blk(name::ATTN_Q_B), &[nh * hd, g.q_lora])?,
        kv: loader.matmul(&blk(name::ATTN_KV), &[hd, d])?,
        kv_norm: loader.floats(&blk(name::ATTN_KV_A_NORM), &[hd])?,
        out_a: row_groups(
            loader,
            &blk(name::ATTN_OUTPUT_A),
            g.out_groups * g.out_rank,
            nh * hd / g.out_groups,
            g.out_groups,
        )?,
        out_b: loader.matmul(&blk(name::ATTN_OUTPUT_B), &[d, g.out_groups * g.out_rank])?,
        sinks: loader.floats(&blk(name::ATTN_SINKS), &[nh])?,
        ffn,
    })
}
