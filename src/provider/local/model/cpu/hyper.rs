//! Hyper-connections with compressed sparse attention on the CPU (DeepSeek
//! V4.1).
//!
//! The residual stream of a token is `S` copies of the hidden state
//! ([`hc`]); each layer's attention and FFN read a weighted collapse of them
//! and write back through predicted coefficients. The attention is a latent
//! multi-query attention: one `head_dim` vector per position is both key and
//! value (its last `rope.dimension_count` values rotate), queries come from a
//! low-rank projection, and the output of the heads goes back through the
//! inverse rotation and a block-diagonal low-rank projection.
//!
//! A query attends to
//!
//! - the last `sliding_window` positions of its layer's own vectors (the
//!   *window* group of the paged cache, one K row per position), and
//! - for a layer with a compression ratio `r`, up to `top_k` *compressed*
//!   rows: row `j` pools the tokens `j*r..(j+1)*r` (a learned softmax gate for
//!   `r = 2`, a plain projection for `r = 1`) and becomes visible to a query
//!   at position `p` once `(j + 1) * r - 1 <= p`. An indexer scores the visible
//!   rows (`sum_h relu(q_h . k_j) * w_h` with FP4-simulated queries and keys)
//!   and the best `top_k` are attended, in row order; a hierarchical candidate
//!   pool narrows later scoring layers to the blocks the candidate source
//!   layer liked best.
//!
//! The compressed rows (with their index keys) live in the *compressed* groups
//! of the paged cache, one block per `16 * r` tokens. The only state that is
//! not in the cache is the compressor's half-finished pair: a call that ends
//! at an odd position leaves its last row's pooling inputs in the sequence
//! slot's [`CompState`], which also keeps the last call's rows so that a
//! speculative rollback can restore it.
//!
//! Every step is row-independent with a fixed summation order: a token's
//! result does not depend on its batch, on how its sequence was chunked or on
//! the thread that computed it.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, PoisonError};

use candle_core::{Device, Tensor};

use super::super::csa::{Candidates, CsaGlobals, CsaSpec, EngramSpec, HyperSpec};
use super::super::engram::{self, RowSource};
use super::super::engram_io::EngramCounters;
use super::super::hc::{self, HcWeights, coeff_len, coefficients, collapse, dot, expand};
use super::super::kvquant;
use super::super::layout::name;
use super::super::paged::{BLOCK_SIZE, NO_BLOCK};
use super::super::params::{Hparams, LayerSpec};
use super::super::pool::{for_each, par_chunks, par_chunks_with};
use super::super::rope::CosSin;
use super::{
    Hidden, KvLayer, LayerWeights, Loader, MAX_VERIFY_ROWS, Mixer, QMatMul, Seg, Transformer,
    TrunkOut,
};
use crate::error::CucaError;

fn err(e: impl std::fmt::Display) -> CucaError {
    super::super::err(e.to_string())
}

fn msg(s: impl std::fmt::Display) -> candle_core::Error {
    candle_core::Error::msg(s.to_string())
}

// ---- weights ------------------------------------------------------------

/// The attention weights of one layer.
pub(super) struct CsaWeights {
    q_a: QMatMul,
    q_a_norm: Vec<f32>,
    q_b: QMatMul,
    kv: QMatMul,
    kv_norm: Vec<f32>,
    /// One `[rank, heads * head_dim / groups]` projection per output group.
    out_a: Vec<QMatMul>,
    out_b: QMatMul,
    sinks: Vec<f32>,
    /// The cache owner's compressor.
    compressor: Option<Compressor>,
    /// A scoring layer's indexer.
    indexer: Option<IndexerWeights>,
}

struct Compressor {
    kv: QMatMul,
    /// The pooling gate (`r = 2`); a ratio-1 layer pools nothing.
    gate: Option<QMatMul>,
    norm: Vec<f32>,
    /// The projection of the latent to the index key and its norm.
    index_key: QMatMul,
    index_norm: Vec<f32>,
}

struct IndexerWeights {
    q_b: QMatMul,
    /// `[heads, n_embd]`: the per-head weight of a query.
    proj: QMatMul,
}

/// The hyper-connection sites of one layer.
pub(super) struct HcLayer {
    attn: HcWeights,
    ffn: HcWeights,
    attn_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    /// The layer's table lookup, on the engram layers.
    engram: Option<EngramLayer>,
}

/// The weights of one engram layer: the projection of the fetched rows to a
/// key per stream and one value, and the product of the query and key
/// weights.
struct EngramLayer {
    /// Position among the engram layers.
    ordinal: usize,
    kv: QMatMul,
    /// `[streams][n_embd]`.
    gate: Vec<f32>,
}

pub(super) fn load_hc(
    loader: &mut Loader<'_>,
    n_embd: usize,
    il: usize,
    spec: &HyperSpec,
    fn_part: &str,
    base_part: &str,
    scale_part: &str,
) -> Result<HcWeights, CucaError> {
    let mix = (2 + spec.streams) * spec.streams;
    let scale = loader.floats(&name::blk(il, scale_part), &[3])?;
    Ok(HcWeights {
        mix: loader.floats(&name::blk(il, fn_part), &[mix, spec.streams * n_embd])?,
        base: loader.floats(&name::blk(il, base_part), &[mix])?,
        scale: [scale[0], scale[1], scale[2]],
    })
}

/// The hyper-connection weights of layer `il`.
pub(super) fn load_hc_layer(
    loader: &mut Loader<'_>,
    hp: &Hparams,
    il: usize,
) -> Result<HcLayer, CucaError> {
    let spec = hp
        .hyper
        .as_ref()
        .ok_or_else(|| err("hyper-connections without a stream count"))?;
    Ok(HcLayer {
        attn: load_hc(
            loader,
            hp.n_embd,
            il,
            spec,
            name::HC_ATTN_FN,
            name::HC_ATTN_BASE,
            name::HC_ATTN_SCALE,
        )?,
        ffn: load_hc(
            loader,
            hp.n_embd,
            il,
            spec,
            name::HC_FFN_FN,
            name::HC_FFN_BASE,
            name::HC_FFN_SCALE,
        )?,
        attn_norm: loader.floats(&name::blk(il, name::ATTN_NORM), &[hp.n_embd])?,
        ffn_norm: loader.floats(&name::blk(il, name::FFN_NORM), &[hp.n_embd])?,
        engram: match hp
            .engram
            .as_ref()
            .and_then(|e| e.index_of(il).map(|o| (e, o)))
        {
            Some((e, ordinal)) => Some(EngramLayer {
                ordinal,
                kv: loader.matmul(
                    &name::blk(il, name::ENGRAM_KV),
                    &[(spec.streams + 1) * hp.n_embd, e.cols() * e.row_dim],
                )?,
                gate: loader.floats(
                    &name::blk(il, name::ENGRAM_GATE),
                    &[spec.streams, hp.n_embd],
                )?,
            }),
            None => None,
        },
    })
}

/// The `[rows, cols]` matrix `tensor_name` as `groups` matmul weights of
/// `rows / groups` consecutive rows each.
pub(super) fn row_groups(
    loader: &mut Loader<'_>,
    tensor_name: &str,
    rows: usize,
    cols: usize,
    groups: usize,
) -> Result<Vec<QMatMul>, CucaError> {
    let qt = loader.qtensor(tensor_name, &[rows, cols])?;
    let per = rows / groups;
    let wrap = |e: candle_core::Error| err(format!("tensor `{tensor_name}`: {e}"));
    if loader.dequantize {
        let t = qt.dequantize(&Device::Cpu).map_err(wrap)?;
        return (0..groups)
            .map(|g| {
                t.narrow(0, g * per, per)
                    .and_then(|p| p.contiguous())
                    .map(QMatMul::tensor)
                    .map_err(wrap)
            })
            .collect();
    }
    let dtype = qt.dtype();
    let row_bytes = cols / dtype.block_size() * dtype.type_size();
    let data = qt.data().map_err(wrap)?;
    (0..groups)
        .map(|g| {
            let bytes = &data[g * per * row_bytes..(g + 1) * per * row_bytes];
            super::super::super::gguf::qtensor_from_bytes(dtype, &[per, cols], bytes, &Device::Cpu)
                .map(std::sync::Arc::new)
                .map_err(wrap)
                .and_then(|q| loader.to_matmul(q, tensor_name))
        })
        .collect()
}

/// The attention weights of layer `il`.
pub(super) fn load_csa(
    loader: &mut Loader<'_>,
    hp: &Hparams,
    il: usize,
    spec: &LayerSpec,
) -> Result<CsaWeights, CucaError> {
    let g = hp
        .csa
        .as_ref()
        .ok_or_else(|| err("compressed attention without its geometry"))?;
    let (d, nh, hd) = (hp.n_embd, spec.n_head, spec.head_dim);
    let blk = |part: &str| name::blk(il, part);
    if !(nh * hd).is_multiple_of(g.out_groups) || !(nh * hd / g.out_groups).is_multiple_of(32) {
        return Err(err(format!(
            "{nh} heads of {hd} do not split into {} output groups of a multiple of 32",
            g.out_groups
        )));
    }
    let ix = &g.indexer;
    let compressor = match spec.csa.filter(|c| c.owns_kv) {
        Some(c) => Some(Compressor {
            kv: loader.matmul(&blk(name::ATTN_COMPRESSOR_KV), &[hd, d])?,
            gate: if c.ratio > 1 {
                Some(loader.matmul(&blk(name::ATTN_COMPRESSOR_GATE), &[hd, d])?)
            } else {
                None
            },
            norm: loader.floats(&blk(name::ATTN_COMPRESSOR_NORM), &[hd])?,
            index_key: loader.matmul(&blk(name::INDEXER_ATTN_K), &[ix.dim, hd])?,
            index_norm: loader.floats(&blk(name::INDEXER_K_NORM), &[ix.dim])?,
        }),
        None => None,
    };
    let indexer = match spec.csa.filter(|c| c.scores) {
        Some(_) => Some(IndexerWeights {
            q_b: loader.matmul(&blk(name::INDEXER_ATTN_Q_B), &[ix.heads * ix.dim, g.q_lora])?,
            proj: loader.matmul(&blk(name::INDEXER_PROJ), &[ix.heads, d])?,
        }),
        None => None,
    };
    Ok(CsaWeights {
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
        compressor,
        indexer,
    })
}

// ---- per-model state ------------------------------------------------------

/// Rotary tables of one distinct rotation.
pub(super) struct Rope {
    pub(super) half: usize,
    pub(super) cos: Vec<f32>,
    pub(super) sin: Vec<f32>,
}

/// The compressor state of one sequence slot: the rows of the last committed
/// forward call (positions `end - n..end`, per pairing layer `(kv | score)`
/// of `head_dim` floats each), kept so that the next call can finish a
/// half-pooled pair and a rollback inside the call can restore one.
#[derive(Default)]
struct CompState {
    /// The position the state sits at.
    end: usize,
    log: Vec<Vec<f32>>,
    /// What the running pass is building; it becomes `log` at its commit.
    staged: Vec<Vec<f32>>,
    staged_end: usize,
}

/// What the hyper-connection path keeps next to the layers.
pub(super) struct HyperModel {
    spec: HyperSpec,
    ropes: Vec<Rope>,
    /// The cache simulates the quantization the model trained with.
    qdq: bool,
    /// The compressed-row pool of each layer that owns one.
    pub(super) ckv: Vec<Option<std::sync::Arc<KvLayer>>>,
    /// The ordinal of each ratio-2 cache owner among them.
    owners: Vec<Option<usize>>,
    n_owners: usize,
    states: Mutex<Vec<CompState>>,
    /// `head_dim` of the pairing layers' rows.
    row_floats: usize,
    /// The engram table's rows.
    engram: Option<std::sync::Arc<dyn RowSource>>,
}

impl HyperModel {
    pub(super) fn new(
        hp: &Hparams,
        tables: &[CosSin],
        ckv: Vec<Option<std::sync::Arc<KvLayer>>>,
        slots: usize,
        engram: Option<std::sync::Arc<dyn RowSource>>,
    ) -> Option<Self> {
        let spec = hp.hyper?;
        let mut n_owners = 0;
        let owners = hp
            .layers
            .iter()
            .map(|l| {
                l.csa.filter(|c| c.owns_kv && c.ratio == 2).map(|_| {
                    n_owners += 1;
                    n_owners - 1
                })
            })
            .collect();
        Some(Self {
            spec,
            ropes: tables
                .iter()
                .map(|t| Rope {
                    half: t.half,
                    cos: t.cos.clone(),
                    sin: t.sin.clone(),
                })
                .collect(),
            qdq: hp.behavior.kv_qdq && !qdq_disabled(),
            ckv,
            owners,
            n_owners,
            states: Mutex::new((0..slots.max(1)).map(|_| CompState::default()).collect()),
            engram,
            row_floats: hp
                .layers
                .iter()
                .find(|l| l.csa.is_some())
                .map_or(0, |l| l.head_dim),
        })
    }

    fn states(&self) -> std::sync::MutexGuard<'_, Vec<CompState>> {
        self.states.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Forget the compressor state of `slot`.
    pub(super) fn reset(&self, slot: usize) {
        if let Some(s) = self.states().get_mut(slot) {
            *s = CompState::default();
        }
    }

    /// Make the state the running pass staged the state of its sequences.
    pub(super) fn commit(&self, slots: impl IntoIterator<Item = usize>) {
        let mut states = self.states();
        for slot in slots {
            if let Some(s) = states.get_mut(slot) {
                s.log = std::mem::take(&mut s.staged);
                s.end = s.staged_end;
            }
        }
    }

    /// Return `slot` to position `len` (at most the end of its last call) when
    /// its state allows; the position it sits at afterwards, 0 when it cannot
    /// (the caller recomputes from scratch).
    pub(super) fn rewind(&self, slot: usize, len: usize) -> usize {
        let mut states = self.states();
        let Some(s) = states.get_mut(slot) else {
            return 0;
        };
        if len >= s.end {
            return s.end;
        }
        let n = s
            .log
            .first()
            .map_or(0, |l| l.len() / (2 * self.row_floats.max(1)));
        let first = s.end - n.min(s.end);
        if len % 2 == 1 && (len <= first || len > s.end) {
            *s = CompState::default();
            return 0;
        }
        // Keep the rows before `len`.
        let keep = len.saturating_sub(first).min(n);
        for l in &mut s.log {
            l.truncate(keep * 2 * self.row_floats);
        }
        s.end = len;
        len
    }
}

// ---- small row helpers -----------------------------------------------------

/// `x` through the weight `w`: `rows` is `t` rows of `k` floats.
pub(super) fn mm(w: &QMatMul, rows: &[f32], k: usize) -> candle_core::Result<Vec<f32>> {
    let t = rows.len() / k;
    let x = Tensor::from_slice(rows, (1, t, k), &Device::Cpu)?;
    w.forward(&x)?.flatten_all()?.to_vec1::<f32>()
}

/// `out = w * (x / rms(x))`.
pub(super) fn rms_row(x: &[f32], w: &[f32], eps: f32, out: &mut [f32]) {
    let inv = 1.0 / (dot(x, x) / x.len() as f32 + eps).sqrt();
    for ((o, &x), &w) in out.iter_mut().zip(x).zip(w) {
        *o = w * (x * inv);
    }
}

/// Rotate adjacent pairs of `x` (the rotated tail of a vector) by the angles
/// of one position: `(a, b) -> (a cos - b sin, a sin + b cos)`, or the inverse
/// rotation.
fn rotate(x: &mut [f32], cos: &[f32], sin: &[f32], inverse: bool) {
    for (i, pair) in x.as_chunks_mut::<2>().0.iter_mut().enumerate() {
        let (a, b) = (pair[0], pair[1]);
        let (c, s) = (cos[i], if inverse { -sin[i] } else { sin[i] });
        pair[0] = a * c - b * s;
        pair[1] = a * s + b * c;
    }
}

impl Rope {
    fn at(&self, pos: usize) -> (&[f32], &[f32]) {
        (
            &self.cos[pos * self.half..(pos + 1) * self.half],
            &self.sin[pos * self.half..(pos + 1) * self.half],
        )
    }

    /// Rotate the last `2 * half` values of `v` at position `pos`.
    pub(super) fn tail(&self, v: &mut [f32], pos: usize, inverse: bool) {
        let (cos, sin) = self.at(pos);
        let n = v.len();
        rotate(&mut v[n - 2 * self.half..], cos, sin, inverse);
    }
}

/// The row of position/row `i` of a paged pool of rows of `width` floats.
fn pool_row(table: &[u32], i: usize, width: usize) -> Option<usize> {
    let b = *table.get(i / BLOCK_SIZE)?;
    (b != NO_BLOCK).then(|| (b as usize * BLOCK_SIZE + i % BLOCK_SIZE) * width)
}

/// Where each row of the step sits.
struct RowMap {
    seg: Vec<u32>,
    pos: Vec<u32>,
}

/// What a scoring layer hands to the layers that reuse its picks.
#[derive(Default)]
struct Shared {
    /// Per row, the compressed rows to attend, ascending.
    picks: Vec<u32>,
    pick_off: Vec<usize>,
    /// Per row, the blocks the candidate source kept, ascending.
    cand: Vec<u32>,
    cand_off: Vec<usize>,
}

// ---- the pass -----------------------------------------------------------

impl Transformer {
    /// The hyper-connection layers over the concatenated rows `embeddings` of
    /// the sequences `segs` describe: the stream after the last layer
    /// collapsed to one `(1, t, n_embd)` hidden state, or, on a [`Front`]
    /// (no LM head), the stream `(1, t, S * n_embd)` and the pre weights
    /// `(1, t, S)` its last layer's FFN predicted.
    pub(super) fn hyper_trunk(
        &self,
        embeddings: Vec<f32>,
        segs: &[Seg<'_>],
    ) -> candle_core::Result<TrunkOut> {
        let hp = &self.hp;
        let hy = self
            .hyper
            .as_ref()
            .ok_or_else(|| msg("hyper-connection pass without hyper-connections"))?;
        let (s, d) = (hy.spec.streams, hp.n_embd);
        let t = embeddings.len() / d;
        let mut stream = vec![0f32; t * s * d];
        for (row, e) in stream
            .chunks_exact_mut(s * d)
            .zip(embeddings.chunks_exact(d))
        {
            for copy in row.chunks_exact_mut(d) {
                copy.copy_from_slice(e);
            }
        }
        let mut pre_mix = vec![0f32; t * s];
        for row in pre_mix.chunks_exact_mut(s) {
            row[0] = 1.0;
        }
        let mut rows = RowMap {
            seg: Vec::with_capacity(t),
            pos: Vec::with_capacity(t),
        };
        for (i, seg) in segs.iter().enumerate() {
            for p in 0..seg.t {
                rows.seg.push(i as u32);
                rows.pos.push((seg.start + p) as u32);
            }
        }
        {
            let mut states = hy.states();
            let mut seen = Vec::with_capacity(segs.len());
            for seg in segs {
                let slot = states
                    .get_mut(seg.slot)
                    .ok_or_else(|| msg(format!("state slot {} is out of range", seg.slot)))?;
                if seen.contains(&seg.slot) {
                    return Err(msg(format!(
                        "state slot {} appears twice in one step",
                        seg.slot
                    )));
                }
                seen.push(seg.slot);
                slot.staged = vec![Vec::new(); hy.n_owners];
                slot.staged_end = seg.start + seg.t;
            }
        }
        let mut shared = Shared::default();
        // Every engram layer's rows depend on token ids only: start reading
        // them all now, so the reads of the later layers overlap the compute
        // of the earlier ones.
        let mut engram_ids: Vec<Vec<u64>> = Vec::new();
        if let (Some(es), Some(source)) = (hp.engram.as_ref(), hy.engram.as_deref()) {
            for ordinal in 0..es.layers.len() {
                let ids = self.engram_row_ids(es, ordinal, t, segs)?;
                source.prefetch(&ids);
                engram_ids.push(ids);
            }
        }
        let mut captured: Vec<Vec<Vec<f32>>> = vec![Vec::new(); segs.len()];
        for il in 0..self.layers.len() {
            if let (Some(hc), Some(es)) = (self.layers[il].hc.as_deref(), hp.engram.as_ref())
                && let Some(eng) = &hc.engram
            {
                self.engram_layer(eng, es, &engram_ids[eng.ordinal], &mut stream)?;
            }
            for (seg, out) in segs.iter().zip(&mut captured) {
                if let Hidden::Layers(set) = seg.hidden
                    && set.contains(il)
                {
                    let mut mean = vec![0f32; seg.t * d];
                    for (m, row) in mean.chunks_exact_mut(d).zip(
                        stream[seg.offset * s * d..(seg.offset + seg.t) * s * d]
                            .chunks_exact(s * d),
                    ) {
                        collapse(row, &vec![1.0 / s as f32; s], m);
                    }
                    out.push(mean);
                }
            }
            self.hyper_layer(il, &mut stream, &mut pre_mix, &mut shared, &rows, segs)?;
        }
        // A front (the first layers of an offloaded model) hands the stream
        // and the pre weights its last FFN predicted to the layers after it.
        if self.lm_head.is_none() {
            return Ok((
                Tensor::from_vec(stream, (1, t, s * d), &Device::Cpu)?,
                Some(Tensor::from_vec(pre_mix, (1, t, s), &Device::Cpu)?),
                captured,
            ));
        }
        // The head reads the stream collapsed by the last FFN's pre weights.
        let mut x = vec![0f32; t * d];
        par_chunks(&mut x, d, |r, out| {
            collapse(
                &stream[r * s * d..(r + 1) * s * d],
                &pre_mix[r * s..(r + 1) * s],
                out,
            );
        });
        Ok((
            Tensor::from_vec(x, (1, t, d), &Device::Cpu)?,
            None,
            captured,
        ))
    }

    /// Add the hashed table lookup of an engram layer to every stream: the
    /// rows the n-grams ending at each token hash to, projected to one key
    /// per stream and a value; a stream takes the value scaled by a gate that
    /// measures how well the stream matches its key.
    fn engram_layer(
        &self,
        eng: &EngramLayer,
        es: &EngramSpec,
        ids: &[u64],
        stream: &mut [f32],
    ) -> candle_core::Result<()> {
        let hp = &self.hp;
        let hy = self
            .hyper
            .as_ref()
            .ok_or_else(|| msg("no hyper-connections"))?;
        let source = hy
            .engram
            .as_deref()
            .ok_or_else(|| msg("an engram layer without a table"))?;
        let (s, d) = (hy.spec.streams, hp.n_embd);
        let t = stream.len() / (s * d);
        let cols = es.cols();
        let mut rows = vec![0f32; t * cols * es.row_dim];
        source.rows(ids, &mut rows).map_err(msg)?;
        let kv = mm(&eng.kv, &rows, cols * es.row_dim)?;
        let eps = hp.eps as f32;
        let width = (s + 1) * d;
        let scale = (d as f32).powf(-0.5);
        par_chunks(stream, s * d, |r, row| {
            let key = &kv[r * width..r * width + s * d];
            let value = &kv[r * width + s * d..(r + 1) * width];
            for (copy, (k, w)) in row
                .chunks_exact_mut(d)
                .zip(key.chunks_exact(d).zip(eng.gate.chunks_exact(d)))
            {
                let rstd = (1.0 / (dot(copy, copy) / d as f32 + eps).sqrt())
                    * (1.0 / (dot(k, k) / d as f32 + eps).sqrt());
                let mut acc = 0.0;
                for i in 0..d {
                    acc += copy[i] * w[i] * k[i];
                }
                let x = acc * rstd * scale;
                let gate = 1.0 / (1.0 + (-x.abs().max(1e-6).sqrt().copysign(x)).exp());
                for (h, &v) in copy.iter_mut().zip(value) {
                    *h += gate * v;
                }
            }
        });
        Ok(())
    }

    /// The table rows (absolute indices, `cols` per row) the engram layer with
    /// ordinal `ordinal` reads for the rows of `segs`. They depend on token ids
    /// only, so a pass knows them before it computes anything.
    fn engram_row_ids(
        &self,
        es: &EngramSpec,
        ordinal: usize,
        t: usize,
        segs: &[Seg<'_>],
    ) -> candle_core::Result<Vec<u64>> {
        let cols = es.cols();
        let mut ids = vec![0u64; t * cols];
        for seg in segs {
            if seg.start > 0 && seg.history.len() != seg.start {
                return Err(msg(format!(
                    "the engram needs the {} tokens before position {} (got {})",
                    seg.start,
                    seg.start,
                    seg.history.len()
                )));
            }
            let at = seg.offset * cols;
            engram::sequence_row_ids(
                es,
                ordinal,
                seg.tokens,
                seg.history,
                &mut ids[at..at + seg.t * cols],
            )
            .map_err(msg)?;
        }
        Ok(ids)
    }

    /// Start reading, in the background, the table rows a later pass over
    /// `tokens` (the next ones of a sequence, which follow `history`) will
    /// look up, so that they are in the cache by then. A hint: a model without
    /// an engram table, or a bad id, does nothing.
    pub(crate) fn engram_prefetch(&self, history: &[u32], tokens: &[u32]) {
        let (Some(hy), Some(es)) = (self.hyper.as_ref(), self.hp.engram.as_ref()) else {
            return;
        };
        let Some(source) = hy.engram.as_deref() else {
            return;
        };
        let cols = es.cols();
        let mut ids = vec![0u64; tokens.len() * cols];
        for ordinal in 0..es.layers.len() {
            if engram::sequence_row_ids(es, ordinal, tokens, history, &mut ids).is_ok() {
                source.prefetch(&ids);
            }
        }
    }

    /// The counters of the engram table, when the model has one.
    pub(crate) fn engram_counters(&self) -> Option<std::sync::Arc<EngramCounters>> {
        self.hyper.as_ref()?.engram.as_deref()?.counters()
    }

    /// One layer: attention then FFN, each between its hyper-connection
    /// collapse and expansion. `pre_mix` is what the previous layer's FFN
    /// predicted for the attention to read.
    fn hyper_layer(
        &self,
        il: usize,
        stream: &mut Vec<f32>,
        pre_mix: &mut [f32],
        shared: &mut Shared,
        rows: &RowMap,
        segs: &[Seg<'_>],
    ) -> candle_core::Result<()> {
        let hp = &self.hp;
        let hy = self
            .hyper
            .as_ref()
            .ok_or_else(|| msg("no hyper-connections"))?;
        let lw: &LayerWeights = &self.layers[il];
        let (Some(hc), Mixer::Csa(cw)) = (lw.hc.as_deref(), &lw.mixer) else {
            return Err(msg(format!("layer {il} has no hyper-connection weights")));
        };
        let (s, d) = (hy.spec.streams, hp.n_embd);
        let t = stream.len() / (s * d);
        let eps = hp.eps as f32;
        let cl = coeff_len(s);

        // Attention.
        let mut attn_c = vec![0f32; t * cl];
        par_chunks(&mut attn_c, cl, |r, c| {
            coefficients(
                &stream[r * s * d..(r + 1) * s * d],
                &hc.attn,
                &hy.spec,
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
            rms_row(&x, &hc.attn_norm, eps, out);
        });
        let y = self.csa_attention(il, cw, &h, rows, segs, shared)?;
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

        // Feed-forward.
        let mut ffn_c = vec![0f32; t * cl];
        par_chunks(&mut ffn_c, cl, |r, c| {
            coefficients(&next[r * s * d..(r + 1) * s * d], &hc.ffn, &hy.spec, eps, c);
        });
        let mut f_in = vec![0f32; t * d];
        par_chunks(&mut f_in, d, |r, out| {
            let mut x = vec![0f32; d];
            collapse(
                &next[r * s * d..(r + 1) * s * d],
                hc::pre(&attn_c[r * cl..(r + 1) * cl], s),
                &mut x,
            );
            rms_row(&x, &hc.ffn_norm, eps, out);
        });
        let f = lw
            .ffn
            .forward(&Tensor::from_vec(f_in, (1, t, d), &Device::Cpu)?, hp.act())?
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

    /// The attention of layer `il` over the normed rows `h` `(t, n_embd)`:
    /// `(t, n_embd)`.
    fn csa_attention(
        &self,
        il: usize,
        cw: &CsaWeights,
        h: &[f32],
        rows: &RowMap,
        segs: &[Seg<'_>],
        shared: &mut Shared,
    ) -> candle_core::Result<Vec<f32>> {
        let hp = &self.hp;
        let hy = self
            .hyper
            .as_ref()
            .ok_or_else(|| msg("no hyper-connections"))?;
        let g: &CsaGlobals = hp
            .csa
            .as_ref()
            .ok_or_else(|| msg("no attention geometry"))?;
        let spec = &hp.layers[il];
        let (d, nh, hd) = (hp.n_embd, spec.n_head, spec.head_dim);
        let t = h.len() / d;
        let eps = hp.eps as f32;
        let rope = &hy.ropes[spec
            .rope
            .ok_or_else(|| msg("layer without a rotary table"))?];
        let window = spec.window.ok_or_else(|| msg("layer without a window"))?;
        let qdq = hy.qdq;
        let pos_of = |r: usize| rows.pos[r] as usize;

        // Queries and the window's key/value row of every token.
        let qa = mm(&cw.q_a, h, d)?;
        let mut qn = vec![0f32; t * g.q_lora];
        par_chunks(&mut qn, g.q_lora, |r, out| {
            rms_row(
                &qa[r * g.q_lora..(r + 1) * g.q_lora],
                &cw.q_a_norm,
                eps,
                out,
            );
        });
        let mut q = mm(&cw.q_b, &qn, g.q_lora)?;
        par_chunks(&mut q, nh * hd, |r, row| {
            for head in row.chunks_exact_mut(hd) {
                rope.tail(head, pos_of(r), false);
            }
        });
        let mut kv = mm(&cw.kv, h, d)?;
        par_chunks(&mut kv, hd, |r, row| {
            let mut v = vec![0f32; hd];
            rms_row(row, &cw.kv_norm, eps, &mut v);
            rope.tail(&mut v, pos_of(r), false);
            if qdq {
                kvquant::qdq_fp8_ue8m0(&mut v, 32);
            }
            row.copy_from_slice(&v);
        });

        // The window rows of this call, then the compressed rows.
        let wlayer = self.cache(il)?;
        let wgroup = self.layout.layer_group[il].ok_or_else(|| msg("layer without a K/V group"))?;
        {
            let mut data = wlayer.lock();
            for (r, row) in kv.chunks_exact(hd).enumerate() {
                let seg = &segs[rows.seg[r] as usize];
                let off = pool_row(&seg.tables[wgroup], pos_of(r), hd)
                    .ok_or_else(|| msg("a position of the call has no K/V block"))?;
                data.k[off..off + hd].copy_from_slice(row);
            }
        }
        if let (Some(csa), Some(comp)) = (spec.csa, &cw.compressor) {
            self.compress(il, &csa, comp, h, rows, segs)?;
        }

        // Which compressed rows each query attends to.
        if let Some(csa) = spec.csa {
            if let Some(ix) = &cw.indexer {
                self.index(il, &csa, ix, h, &qn, rows, segs, shared)?;
            } else if shared.pick_off.len() != t + 1 {
                return Err(msg(format!(
                    "layer {il} reuses the indexer's picks but no layer computed them"
                )));
            }
        }

        // Attention over the window and the picks.
        let scale = spec.scale;
        let ratio_group = self.layout.csa_group[il];
        let source_width = spec
            .csa
            .and_then(|c| hp.layers[c.kv_source].csa)
            .map_or(0, |c| c.row_width);
        let clayer = spec
            .csa
            .and_then(|c| hy.ckv[c.kv_source].as_ref().map(|l| (c, l)));
        let wdata = wlayer.lock();
        let cdata = clayer.as_ref().map(|(_, l)| l.lock());
        let missing = AtomicBool::new(false);
        let mut o = vec![0f32; t * nh * hd];
        par_chunks_with(&mut o, hd, Vec::<f32>::new, |scores, idx, out| {
            let (r, head) = (idx / nh, idx % nh);
            let seg = &segs[rows.seg[r] as usize];
            let p = pos_of(r);
            let lo = (p + 1).saturating_sub(window);
            let qv = &q[(r * nh + head) * hd..(r * nh + head + 1) * hd];
            let picks: &[u32] = match &clayer {
                Some(_) => &shared.picks[shared.pick_off[r]..shared.pick_off[r + 1]],
                None => &[],
            };
            scores.clear();
            let wtable = &seg.tables[wgroup];
            for pos in lo..=p {
                let Some(off) = pool_row(wtable, pos, hd) else {
                    missing.store(true, Ordering::Relaxed);
                    return;
                };
                scores.push(dot(qv, &wdata.k[off..off + hd]) * scale);
            }
            if let (Some(_), Some(cdata), Some(grp)) = (&clayer, &cdata, ratio_group) {
                let ctable = &seg.tables[grp];
                for &j in picks {
                    let Some(off) = pool_row(ctable, j as usize, source_width) else {
                        missing.store(true, Ordering::Relaxed);
                        return;
                    };
                    scores.push(dot(qv, &cdata.k[off..off + hd]) * scale);
                }
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
            denom += (cw.sinks[head] - m).exp();
            out.fill(0.0);
            let mut at = 0;
            for pos in lo..=p {
                if let Some(off) = pool_row(wtable, pos, hd) {
                    let w = scores[at];
                    for (o, &v) in out.iter_mut().zip(&wdata.k[off..off + hd]) {
                        *o += w * v;
                    }
                }
                at += 1;
            }
            if let (Some(_), Some(cdata), Some(grp)) = (&clayer, &cdata, ratio_group) {
                let ctable = &seg.tables[grp];
                for &j in picks {
                    if let Some(off) = pool_row(ctable, j as usize, source_width) {
                        let w = scores[at];
                        for (o, &v) in out.iter_mut().zip(&cdata.k[off..off + hd]) {
                            *o += w * v;
                        }
                    }
                    at += 1;
                }
            }
            for v in out.iter_mut() {
                *v /= denom;
            }
        });
        drop((wdata, cdata));
        if missing.load(Ordering::Relaxed) {
            return Err(msg("attention reads a position without a K/V block"));
        }

        // Undo the query's rotation, simulate the output's quantization and
        // project through the grouped low-rank output.
        par_chunks(&mut o, nh * hd, |r, row| {
            for head in row.chunks_exact_mut(hd) {
                rope.tail(head, pos_of(r), true);
            }
            if qdq {
                kvquant::qdq_fp8_ue8m0(row, 32);
            }
        });
        let width = nh * hd / g.out_groups;
        let mut mid = vec![0f32; t * g.out_groups * g.out_rank];
        for (grp, w) in cw.out_a.iter().enumerate() {
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
        mm(&cw.out_b, &mid, g.out_groups * g.out_rank)
    }

    /// The compressed rows the tokens of this call complete: pool them,
    /// derive the index keys, simulate the quantization and write the rows
    /// into the sequences' compressed blocks.
    fn compress(
        &self,
        il: usize,
        csa: &CsaSpec,
        comp: &Compressor,
        h: &[f32],
        rows: &RowMap,
        segs: &[Seg<'_>],
    ) -> candle_core::Result<()> {
        let hp = &self.hp;
        let hy = self
            .hyper
            .as_ref()
            .ok_or_else(|| msg("no hyper-connections"))?;
        let spec = &hp.layers[il];
        let (d, hd) = (hp.n_embd, spec.head_dim);
        let ix = &hp
            .csa
            .as_ref()
            .ok_or_else(|| msg("no attention geometry"))?
            .indexer;
        let eps = hp.eps as f32;
        let rope = &hy.ropes[spec
            .rope
            .ok_or_else(|| msg("layer without a rotary table"))?];
        let ratio = csa.ratio;

        let kv = mm(&comp.kv, h, d)?;
        let gate = match &comp.gate {
            Some(g) => Some(mm(g, h, d)?),
            None => None,
        };
        // The latents the call completes: (sequence, row index, latent).
        let mut done: Vec<(usize, usize, Vec<f32>)> = Vec::new();
        let owner = hy.owners[il];
        let mut states = hy.states();
        for (si, seg) in segs.iter().enumerate() {
            let slot = states
                .get_mut(seg.slot)
                .ok_or_else(|| msg(format!("state slot {} is out of range", seg.slot)))?;
            let range = seg.offset..seg.offset + seg.t;
            if let (Some(k), Some(gate)) = (owner, &gate) {
                // The pair a previous call left half-pooled.
                let pending: Option<(&[f32], &[f32])> = if seg.start % 2 == 1 {
                    let log = slot.log.get(k).filter(|l| !l.is_empty());
                    match (log, slot.end == seg.start) {
                        (Some(l), true) => {
                            let last = &l[l.len() - 2 * hd..];
                            Some((&last[..hd], &last[hd..]))
                        }
                        _ => {
                            return Err(msg(format!(
                                "the compressor state of slot {} is not at position {}",
                                seg.slot, seg.start
                            )));
                        }
                    }
                } else {
                    None
                };
                for i in 0..seg.t {
                    let p = seg.start + i;
                    if p % 2 == 0 {
                        continue;
                    }
                    let r = seg.offset + i;
                    let (kv1, s1) = (&kv[r * hd..(r + 1) * hd], &gate[r * hd..(r + 1) * hd]);
                    let (kv0, s0) = if i >= 1 {
                        (&kv[(r - 1) * hd..r * hd], &gate[(r - 1) * hd..r * hd])
                    } else {
                        pending.ok_or_else(|| msg("a pair without its first row"))?
                    };
                    let mut pooled = vec![0f32; hd];
                    for c in 0..hd {
                        let m = s0[c].max(s1[c]);
                        let (e0, e1) = ((s0[c] - m).exp(), (s1[c] - m).exp());
                        let total = e0 + e1;
                        pooled[c] = kv0[c] * (e0 / total) + kv1[c] * (e1 / total);
                    }
                    let mut latent = vec![0f32; hd];
                    rms_row(&pooled, &comp.norm, eps, &mut latent);
                    done.push((si, p / 2, latent));
                }
                // The last rows of the call, for the next call and rollbacks.
                let n = seg.t.min(MAX_VERIFY_ROWS + 1);
                let mut log = Vec::with_capacity(n * 2 * hd);
                for r in range.end - n..range.end {
                    log.extend_from_slice(&kv[r * hd..(r + 1) * hd]);
                    log.extend_from_slice(&gate[r * hd..(r + 1) * hd]);
                }
                slot.staged[k] = log;
            } else {
                // Ratio 1: every token is a row.
                for r in range.clone() {
                    let mut latent = vec![0f32; hd];
                    rms_row(&kv[r * hd..(r + 1) * hd], &comp.norm, eps, &mut latent);
                    done.push((si, rows.pos[r] as usize, latent));
                }
            }
        }
        drop(states);
        if done.is_empty() {
            return Ok(());
        }

        // Index keys from the latents before their rotation.
        let m = done.len();
        let mut flat = vec![0f32; m * hd];
        for (dst, (_, _, latent)) in flat.chunks_exact_mut(hd).zip(&done) {
            dst.copy_from_slice(latent);
        }
        let ik = mm(&comp.index_key, &flat, hd)?;
        let width = csa.row_width;
        let mut out = vec![0f32; m * width];
        let qdq = hy.qdq;
        par_chunks(&mut out, width, |i, row| {
            let (_, j, latent) = &done[i];
            let (lat, key) = row.split_at_mut(hd);
            rms_row(
                &ik[i * ix.dim..(i + 1) * ix.dim],
                &comp.index_norm,
                eps,
                key,
            );
            rope.tail(key, j * ratio, false);
            lat.copy_from_slice(latent);
            rope.tail(lat, j * ratio, false);
            if qdq {
                kvquant::qdq_fp4_e8m0(key, 32);
                kvquant::qdq_fp4_e4m3(lat, 16);
            }
        });
        let layer = hy.ckv[il]
            .as_ref()
            .ok_or_else(|| msg("a compressing layer without a compressed pool"))?;
        let group = self.layout.csa_group[il].ok_or_else(|| msg("no compressed group"))?;
        let mut data = layer.lock();
        for (i, (si, j, _)) in done.iter().enumerate() {
            let off = pool_row(&segs[*si].tables[group], *j, width)
                .ok_or_else(|| msg("a compressed row has no block"))?;
            data.k[off..off + width].copy_from_slice(&out[i * width..(i + 1) * width]);
        }
        Ok(())
    }

    /// Score the compressed rows each query can see and keep the best.
    #[allow(clippy::too_many_arguments)]
    fn index(
        &self,
        il: usize,
        csa: &CsaSpec,
        ix: &IndexerWeights,
        h: &[f32],
        qn: &[f32],
        rows: &RowMap,
        segs: &[Seg<'_>],
        shared: &mut Shared,
    ) -> candle_core::Result<()> {
        let hp = &self.hp;
        let hy = self
            .hyper
            .as_ref()
            .ok_or_else(|| msg("no hyper-connections"))?;
        let g = hp
            .csa
            .as_ref()
            .ok_or_else(|| msg("no attention geometry"))?;
        let spec = &hp.layers[il];
        let (d, hd) = (hp.n_embd, spec.head_dim);
        let idx = &g.indexer;
        let t = h.len() / d;
        let rope = &hy.ropes[spec
            .rope
            .ok_or_else(|| msg("layer without a rotary table"))?];
        let ratio = csa.ratio;
        let qdq = hy.qdq;

        let mut iq = mm(&ix.q_b, qn, g.q_lora)?;
        par_chunks(&mut iq, idx.heads * idx.dim, |r, row| {
            for head in row.chunks_exact_mut(idx.dim) {
                rope.tail(head, rows.pos[r] as usize, false);
            }
            if qdq {
                kvquant::qdq_fp4_e8m0(row, 32);
            }
        });
        let mut wts = mm(&ix.proj, h, d)?;
        let k = ((idx.dim as f32).powf(-0.5)) * (idx.heads as f32).powf(-0.5);
        for w in &mut wts {
            *w *= k;
        }

        let clayer = hy.ckv[csa.kv_source]
            .as_ref()
            .ok_or_else(|| msg("an indexer without compressed rows"))?;
        let group = self.layout.csa_group[il].ok_or_else(|| msg("no compressed group"))?;
        let cdata = clayer.lock();
        let width = hp.layers[csa.kv_source].csa.map_or(0, |c| c.row_width);
        let (cand_blocks, cand_size) = idx.candidates.map_or((0, 1), |(_, b, s)| (b, s));
        let consumer = csa.candidates == Candidates::Consumer;
        if consumer && shared.cand_off.len() != t + 1 {
            return Err(msg(format!(
                "layer {il} narrows to the candidate blocks but no layer published them"
            )));
        }
        let source = csa.candidates == Candidates::Source;
        let (chunk_rows, budget) = index_split();
        let picks: Vec<Mutex<(Vec<u32>, Vec<u32>)>> = (0..t)
            .map(|_| Mutex::new((Vec::new(), Vec::new())))
            .collect();
        let failed = AtomicBool::new(false);
        let vis_of = |r: usize| (rows.pos[r] as usize + 1) / ratio;
        let cands_of = |r: usize| -> &[u32] {
            if consumer {
                &shared.cand[shared.cand_off[r]..shared.cand_off[r + 1]]
            } else {
                &[]
            }
        };
        // The score of visible row `j` of token `r`: the weighted sum of the
        // heads' positive dot products, in head order.
        let score_of = |r: usize, j: usize| -> Option<f32> {
            let seg = &segs[rows.seg[r] as usize];
            let off = pool_row(&seg.tables[group], j, width)?;
            let key = &cdata.k[off + hd..off + hd + idx.dim];
            let q = &iq[r * idx.heads * idx.dim..(r + 1) * idx.heads * idx.dim];
            let w = &wts[r * idx.heads..(r + 1) * idx.heads];
            let mut score = 0.0;
            for (head, &wh) in w.iter().enumerate() {
                let x = dot(&q[head * idx.dim..(head + 1) * idx.dim], key);
                if x > 0.0 {
                    score += x * wh;
                }
            }
            Some(score)
        };
        // Scores of the rows `j0..j1` of token `r` into `out[j - j0]`: all of
        // them, or (a candidate-narrowed layer) those in the candidate blocks.
        let score_chunk = |r: usize, j0: usize, j1: usize, out: &mut [f32]| {
            let mut one = |j: usize| match score_of(r, j) {
                Some(score) => out[j - j0] = score,
                None => failed.store(true, Ordering::Relaxed),
            };
            if consumer {
                let cands = cands_of(r);
                let first = cands.partition_point(|&b| (b as usize + 1) * cand_size <= j0);
                for &b in &cands[first..] {
                    let lo = (b as usize * cand_size).max(j0);
                    if lo >= j1 {
                        break;
                    }
                    (lo..((b as usize + 1) * cand_size).min(j1)).for_each(&mut one);
                }
            } else {
                (j0..j1).for_each(one);
            }
        };

        // Scoring is the work that grows with the context (every visible row
        // against every head), so it is split into chunks of rows over the
        // pool whatever the number of tokens: a decode token has one row to
        // work on, a prefill chunk thousands. Every score is computed alone,
        // in head order, so the split changes no bit. The tokens go in
        // groups whose score buffers stay within `budget`.
        let mut first = 0;
        while first < t {
            let (mut end, mut chunks) = (first, 0);
            while end < t {
                let need = vis_of(end).div_ceil(chunk_rows);
                if end > first && (chunks + need) * chunk_rows > budget {
                    break;
                }
                chunks += need;
                end += 1;
            }
            // `starts[g]`: the first chunk of the group's token `g`.
            let mut starts = Vec::with_capacity(end - first + 1);
            let mut at = 0;
            for r in first..end {
                starts.push(at);
                at += vis_of(r).div_ceil(chunk_rows);
            }
            starts.push(at);
            let mut scores = vec![0f32; at * chunk_rows];
            par_chunks(&mut scores, chunk_rows, |c, out| {
                let g = starts.partition_point(|&s| s <= c) - 1;
                let r = first + g;
                let j0 = (c - starts[g]) * chunk_rows;
                let j1 = (j0 + chunk_rows).min(vis_of(r));
                score_chunk(r, j0, j1, &mut out[..j1 - j0]);
            });
            if failed.load(Ordering::Relaxed) {
                break;
            }
            for_each(
                end - first,
                end - first > 1,
                Vec::<(f32, u32)>::new,
                |order, g| {
                    let r = first + g;
                    let vis = vis_of(r);
                    let scores = &scores[starts[g] * chunk_rows..][..vis];
                    let own_cand = cands_of(r);
                    // `scores` holds a value only where a row was scored.
                    let mut cand = Vec::new();
                    if source && vis > 0 {
                        // Level one: the best blocks, the one holding the
                        // newest row always among them.
                        let bs = cand_size;
                        let nb = vis.div_ceil(bs);
                        let last = (vis - 1) / bs;
                        let mut blocks: Vec<(f32, u32)> = (0..nb)
                            .map(|b| {
                                let best = if b == last {
                                    f32::INFINITY
                                } else {
                                    scores[b * bs..((b + 1) * bs).min(vis)]
                                        .iter()
                                        .copied()
                                        .fold(f32::NEG_INFINITY, f32::max)
                                };
                                (best, b as u32)
                            })
                            .collect();
                        let n = cand_blocks.min(nb);
                        let by = |a: &(f32, u32), b: &(f32, u32)| {
                            b.0.total_cmp(&a.0).then(a.1.cmp(&b.1))
                        };
                        if 0 < n && n < nb {
                            blocks.select_nth_unstable_by(n - 1, by);
                        }
                        blocks.truncate(n);
                        cand = blocks
                            .into_iter()
                            .filter(|b| b.0 > f32::NEG_INFINITY)
                            .map(|b| b.1)
                            .collect();
                        cand.sort_unstable();
                    }
                    // Level two: the `top_k` best scored rows, in row order.
                    order.clear();
                    let mut keep = |j: usize| {
                        if scores[j] > f32::NEG_INFINITY {
                            order.push((scores[j], j as u32));
                        }
                    };
                    if consumer {
                        for &b in own_cand {
                            (b as usize * cand_size..((b as usize + 1) * cand_size).min(vis))
                                .for_each(&mut keep);
                        }
                    } else {
                        (0..vis).for_each(keep);
                    }
                    let by =
                        |a: &(f32, u32), b: &(f32, u32)| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1));
                    let k = idx.top_k.min(order.len());
                    if 0 < k && k < order.len() {
                        order.select_nth_unstable_by(k - 1, by);
                    }
                    order.truncate(k);
                    let mut chosen: Vec<u32> = order.iter().map(|o| o.1).collect();
                    chosen.sort_unstable();
                    *picks[r].lock().unwrap_or_else(PoisonError::into_inner) = (chosen, cand);
                },
            );
            first = end;
        }
        drop(cdata);
        if failed.load(Ordering::Relaxed) {
            return Err(msg("the indexer reads a compressed row without a block"));
        }
        let mut next = Shared::default();
        next.pick_off.push(0);
        next.cand_off.push(0);
        for m in picks {
            let (chosen, cand) = m.into_inner().unwrap_or_else(PoisonError::into_inner);
            next.picks.extend(chosen);
            next.pick_off.push(next.picks.len());
            next.cand.extend(cand);
            next.cand_off.push(next.cand.len());
        }
        // The consumers of this layer's candidates are later layers: keep the
        // candidates of the source across layers that do not publish new ones.
        if !source {
            next.cand = std::mem::take(&mut shared.cand);
            next.cand_off = std::mem::take(&mut shared.cand_off);
        }
        *shared = next;
        Ok(())
    }
}

/// Rows of the indexer's scores one pool job computes.
const INDEX_CHUNK: usize = 64;

/// The most score floats the indexer holds at once (16 MiB): tokens are
/// scored in groups that fit it, a single token whose visible rows exceed it
/// alone in its group.
const INDEX_SCORE_BUDGET: usize = 1 << 22;

/// `(rows per pool job, floats per group)` of the indexer's scoring.
fn index_split() -> (usize, usize) {
    #[cfg(test)]
    if let Some(split) = INDEX_SPLIT.get() {
        return split;
    }
    (INDEX_CHUNK, INDEX_SCORE_BUDGET)
}

/// Run `f` with the indexer scoring in jobs of `chunk` rows and groups of at
/// most `budget` floats on this thread (the split must not change a bit).
#[cfg(test)]
pub(super) fn with_index_split<R>(chunk: usize, budget: usize, f: impl FnOnce() -> R) -> R {
    let before = INDEX_SPLIT.replace(Some((chunk, budget)));
    let r = f();
    INDEX_SPLIT.set(before);
    r
}

#[cfg(test)]
thread_local! {
    static INDEX_SPLIT: std::cell::Cell<Option<(usize, usize)>> = const { std::cell::Cell::new(None) };
}

#[cfg(test)]
thread_local! {
    static QDQ_OFF: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Whether models loaded on this thread simulate no quantization (the
/// unquantized reference of the oracle tests).
#[cfg(test)]
pub(super) fn qdq_disabled() -> bool {
    QDQ_OFF.get()
}

#[cfg(not(test))]
pub(super) fn qdq_disabled() -> bool {
    false
}

/// Run `f` with models loaded on this thread simulating no quantization.
#[cfg(test)]
pub(super) fn without_qdq<R>(f: impl FnOnce() -> R) -> R {
    let before = QDQ_OFF.replace(true);
    let r = f();
    QDQ_OFF.set(before);
    r
}

#[cfg(test)]
mod rope_tests {
    use super::Rope;

    #[test]
    fn an_inverse_rotation_undoes_a_rotation() {
        let (rot, half) = (8usize, 4usize);
        let angle = |p: usize, i: usize| p as f32 * 0.1 * (i + 1) as f32;
        let rope = Rope {
            half,
            cos: (0..64 * half)
                .map(|k| angle(k / half, k % half).cos())
                .collect(),
            sin: (0..64 * half)
                .map(|k| angle(k / half, k % half).sin())
                .collect(),
        };
        let mut v: Vec<f32> = (0..32).map(|i| (i as f32 * 0.37).sin()).collect();
        let before = v.clone();
        rope.tail(&mut v, 13, false);
        assert_ne!(v, before);
        assert_eq!(v[..32 - rot], before[..32 - rot], "only the tail rotates");
        rope.tail(&mut v, 13, true);
        for (a, b) in v.iter().zip(&before) {
            assert!((a - b).abs() < 1e-6);
        }
    }
}
