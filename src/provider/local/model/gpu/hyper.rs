//! Hyper-connections with compressed sparse attention on the GPU (DeepSeek
//! V4.1). [`cpu::hyper`](super::super::cpu) is the specification of the
//! arithmetic; this module runs it on the device.
//!
//! The residual stream is one `[t, S * D]` device tensor. Per layer:
//!
//! - [`hc_kernel`] computes the mixing coefficients of a stream (one cube per
//!   token), the collapse before a sub-block and the expansion after it;
//! - the attention projections are the engine's [`linear`] over the
//!   quantized weights; [`csa_kernel`] rotates the tail of every head, writes
//!   the new window rows and the compressed rows into the paged pools and
//!   attends over the window and the picked compressed rows with the sink;
//!   [`topk_kernel`] scores the indexer and picks the rows;
//! - the MoE is the engine's ([`FfnWeights::forward`]) with the router of
//!   the model (`sqrtsoftplus`, selection bias) and the SwiGLU clamp.
//!
//! Pools. The window pool of a layer is one `[blocks * 16, head_dim]` tensor
//! (a position's key row is also its value) and the compressed pool of a
//! layer that owns compressed rows one `[blocks * 16, head_dim + index_dim]`
//! tensor; a sequence's block tables of the layout's groups (window groups
//! with one row per position, compressed groups with one row per `ratio`
//! positions) say where a row lives. Everything position-dependent a pass
//! needs (row to sequence, positions, destination rows, the pairs a ratio-2
//! compressor completes) is computed on the host from the tables once per
//! pass and shared by the layers.
//!
//! The compressor's half-finished pair is kept per sequence slot as the last
//! rows of the previous call (`[kv | score]` per owner layer), staged during a
//! pass and committed when it succeeds, exactly as the CPU engine does, so a
//! rollback within the last call restores it.

use burn::tensor::{Device, Int, Tensor, TensorData};

use super::super::cpu::Forward;
use super::super::csa::{Candidates, CsaGlobals, CsaSpec, EngramSpec, HyperSpec};
use super::super::engram::{self, RowSource};
use super::super::engram_io::EngramCounters;
use super::super::paged::{BLOCK_SIZE, KvLayout, NO_BLOCK};
use super::super::params::{Hparams, LayerSpec};
use super::super::{csa_kernel, hc_kernel, qdq_kernel, topk_kernel};
use super::{Cut, LayerWeights, Mixer, Seg, Weight, WeightReader, Weights, err, linear, name, rms};
use crate::error::CucaError;

// ---- weights --------------------------------------------------------------

/// The attention weights of one layer.
pub(super) struct GCsa {
    q_a: Weight,
    q_a_norm: Tensor<1>,
    q_b: Weight,
    kv: Weight,
    kv_norm: Tensor<1>,
    /// One `[rank, heads * head_dim / groups]` projection per output group.
    out_a: Vec<Weight>,
    out_b: Weight,
    sinks: Tensor<1>,
    compressor: Option<GCompressor>,
    indexer: Option<GIndexer>,
}

struct GCompressor {
    kv: Weight,
    gate: Option<Weight>,
    norm: Tensor<1>,
    index_key: Weight,
    index_norm: Tensor<1>,
}

struct GIndexer {
    q_b: Weight,
    proj: Weight,
}

/// One hyper-connection site: the mixing weights, offsets and scales.
struct GSite {
    mix: Tensor<2>,
    base: Tensor<1>,
    scale: Tensor<1>,
}

/// The weights of one engram layer.
struct GEngram {
    /// Position among the engram layers.
    ordinal: usize,
    kv: Weight,
    /// `[streams * n_embd]`.
    gate: Tensor<1>,
}

/// The hyper-connection weights of one layer.
pub(super) struct GHc {
    attn: GSite,
    ffn: GSite,
    attn_norm: Tensor<1>,
    ffn_norm: Tensor<1>,
    engram: Option<GEngram>,
}

/// What the model-level hyper-connection path holds besides the layers.
pub(super) struct HyperWeights {
    pub(super) spec: HyperSpec,
    /// The cache simulates the quantization the model trained with.
    qdq: bool,
    /// The engram table's rows.
    engram: Option<std::sync::Arc<dyn RowSource>>,
    /// The ordinal of each ratio-2 compressed-row owner among them.
    owners: Vec<Option<usize>>,
    pub(super) n_owners: usize,
}

impl HyperWeights {
    pub(super) fn new(
        hp: &Hparams,
        spec: HyperSpec,
        engram: Option<std::sync::Arc<dyn RowSource>>,
    ) -> Self {
        let mut n = 0;
        let owners = hp
            .layers
            .iter()
            .map(|l| {
                l.csa.filter(|c| c.owns_kv && c.ratio == 2).map(|_| {
                    n += 1;
                    n - 1
                })
            })
            .collect();
        Self {
            spec,
            qdq: hp.behavior.kv_qdq && !qdq_disabled(),
            engram,
            owners,
            n_owners: n,
        }
    }
}

fn site(
    reader: &mut WeightReader<'_>,
    hp: &Hparams,
    il: usize,
    spec: &HyperSpec,
    parts: [&str; 3],
) -> Result<GSite, CucaError> {
    let mix = (2 + spec.streams) * spec.streams;
    let width = spec.streams * hp.n_embd;
    let values = reader.host_floats(&name::blk(il, parts[0]), &[mix, width])?;
    Ok(GSite {
        mix: Tensor::from_data(TensorData::new(values, [mix, width]), reader.device),
        base: reader.vector(&name::blk(il, parts[1]), mix)?,
        scale: reader.vector(&name::blk(il, parts[2]), 3)?,
    })
}

/// The hyper-connection weights of layer `il`.
pub(super) fn load_hc(
    reader: &mut WeightReader<'_>,
    hp: &Hparams,
    il: usize,
) -> Result<GHc, CucaError> {
    let spec = hp
        .hyper
        .as_ref()
        .ok_or_else(|| err("hyper-connections without a stream count"))?;
    let engram = match hp
        .engram
        .as_ref()
        .and_then(|e| e.index_of(il).map(|o| (e, o)))
    {
        Some((e, ordinal)) => Some(GEngram {
            ordinal,
            kv: reader.matrix(
                &name::blk(il, name::ENGRAM_KV),
                (spec.streams + 1) * hp.n_embd,
                e.cols() * e.row_dim,
            )?,
            gate: {
                let values = reader.host_floats(
                    &name::blk(il, name::ENGRAM_GATE),
                    &[spec.streams, hp.n_embd],
                )?;
                Tensor::from_data(
                    TensorData::new(values, [spec.streams * hp.n_embd]),
                    reader.device,
                )
            },
        }),
        None => None,
    };
    Ok(GHc {
        attn: site(
            reader,
            hp,
            il,
            spec,
            [name::HC_ATTN_FN, name::HC_ATTN_BASE, name::HC_ATTN_SCALE],
        )?,
        ffn: site(
            reader,
            hp,
            il,
            spec,
            [name::HC_FFN_FN, name::HC_FFN_BASE, name::HC_FFN_SCALE],
        )?,
        attn_norm: reader.vector(&name::blk(il, name::ATTN_NORM), hp.n_embd)?,
        ffn_norm: reader.vector(&name::blk(il, name::FFN_NORM), hp.n_embd)?,
        engram,
    })
}

/// The attention weights of layer `il`.
pub(super) fn load_csa(
    reader: &mut WeightReader<'_>,
    hp: &Hparams,
    il: usize,
    spec: &LayerSpec,
) -> Result<GCsa, CucaError> {
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
        Some(c) => Some(GCompressor {
            kv: reader.matrix(&blk(name::ATTN_COMPRESSOR_KV), hd, d)?,
            gate: if c.ratio > 1 {
                Some(reader.matrix(&blk(name::ATTN_COMPRESSOR_GATE), hd, d)?)
            } else {
                None
            },
            norm: reader.vector(&blk(name::ATTN_COMPRESSOR_NORM), hd)?,
            index_key: reader.matrix(&blk(name::INDEXER_ATTN_K), ix.dim, hd)?,
            index_norm: reader.vector(&blk(name::INDEXER_K_NORM), ix.dim)?,
        }),
        None => None,
    };
    let indexer = match spec.csa.filter(|c| c.scores) {
        Some(_) => Some(GIndexer {
            q_b: reader.matrix(&blk(name::INDEXER_ATTN_Q_B), ix.heads * ix.dim, g.q_lora)?,
            proj: reader.matrix(&blk(name::INDEXER_PROJ), ix.heads, d)?,
        }),
        None => None,
    };
    let rows = g.out_groups * g.out_rank;
    let per = g.out_rank;
    let cols = nh * hd / g.out_groups;
    let out_a = (0..g.out_groups)
        .map(|grp| {
            reader.matrix_cut(
                &blk(name::ATTN_OUTPUT_A),
                (rows, cols),
                &Cut::of_rows(std::iter::once(grp * per..(grp + 1) * per), 0..cols),
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(GCsa {
        q_a: reader.matrix(&blk(name::ATTN_Q_A), g.q_lora, d)?,
        q_a_norm: reader.vector(&blk(name::ATTN_Q_A_NORM), g.q_lora)?,
        q_b: reader.matrix(&blk(name::ATTN_Q_B), nh * hd, g.q_lora)?,
        kv: reader.matrix(&blk(name::ATTN_KV), hd, d)?,
        kv_norm: reader.vector(&blk(name::ATTN_KV_A_NORM), hd)?,
        out_a,
        out_b: reader.matrix(&blk(name::ATTN_OUTPUT_B), d, rows)?,
        sinks: reader.vector(&blk(name::ATTN_SINKS), nh)?,
        compressor,
        indexer,
    })
}

// ---- state ----------------------------------------------------------------

/// The compressor state of one sequence slot: the rows `[kv | score]` of the
/// last committed call, per ratio-2 owner layer (`log`), the position the
/// state sits at and what the running pass stages.
#[derive(Default)]
struct CompSlot {
    end: usize,
    log: Vec<Option<Tensor<2>>>,
    staged: Vec<Option<Tensor<2>>>,
    staged_end: usize,
}

/// The pools and the per-slot state of a hyper-connection model.
pub(super) struct HyperState {
    /// The window pool of every layer: `[blocks * 16, head_dim]`.
    window: Vec<Option<Tensor<2>>>,
    /// The compressed pool of every layer that owns one:
    /// `[blocks * 16, head_dim + index_dim]`.
    compressed: Vec<Option<Tensor<2>>>,
    slots: Vec<CompSlot>,
}

impl HyperState {
    /// Zeroed pools on `device` for the groups of `layout`.
    pub(super) fn new(
        hp: &Hparams,
        layout: &KvLayout,
        device: &Device,
        slots: usize,
        first: usize,
    ) -> Result<Self, CucaError> {
        let mut window = Vec::with_capacity(hp.n_layer);
        let mut compressed = Vec::with_capacity(hp.n_layer);
        let limit = super::MAX_PART_BYTES;
        for (il, spec) in hp.layers.iter().enumerate() {
            // A layer before `first` runs elsewhere.
            window.push(match layout.layer_group[il].filter(|_| spec.owns_kv(il) && il >= first) {
                Some(g) => {
                    let rows = layout.blocks[g] * BLOCK_SIZE;
                    if rows * spec.head_dim * 4 > limit {
                        return Err(CucaError::Config(format!(
                            "layer {il}: a window pool of {rows} rows of {} floats does not fit one device buffer; lower `kv_tokens` or the batching limits",
                            spec.head_dim
                        )));
                    }
                    Some(Tensor::zeros([rows, spec.head_dim], device))
                }
                None => None,
            });
            compressed.push(
                match (
                    spec.csa.filter(|c| c.owns_kv && il >= first),
                    layout.csa_group[il],
                ) {
                    (Some(c), Some(g)) => {
                        let rows = layout.blocks[g] * BLOCK_SIZE;
                        if rows * c.row_width * 4 > limit {
                            return Err(CucaError::Config(format!(
                                "layer {il}: a compressed pool of {rows} rows of {} floats does not fit one device buffer ({limit} bytes); lower `kv_tokens`",
                                c.row_width
                            )));
                        }
                        Some(Tensor::zeros([rows, c.row_width], device))
                    }
                    _ => None,
                },
            );
        }
        Ok(Self {
            window,
            compressed,
            slots: (0..slots.max(1)).map(|_| CompSlot::default()).collect(),
        })
    }

    /// Forget the compressor state of `slot`.
    pub(super) fn reset(&mut self, slot: usize) {
        if let Some(s) = self.slots.get_mut(slot) {
            *s = CompSlot::default();
        }
    }

    /// Return `slot` to position `len` (at most the end of its last call)
    /// when its state allows; the position it sits at afterwards, 0 when it
    /// cannot.
    pub(super) fn rewind(&mut self, slot: usize, len: usize) -> usize {
        let Some(s) = self.slots.get_mut(slot) else {
            return 0;
        };
        if len >= s.end {
            return s.end;
        }
        let n = s
            .log
            .iter()
            .flatten()
            .map(|l| l.dims()[0])
            .next()
            .unwrap_or(0);
        let first = s.end - n.min(s.end);
        if len % 2 == 1 && (len <= first || len > s.end) {
            *s = CompSlot::default();
            return 0;
        }
        let keep = len.saturating_sub(first).min(n);
        if keep == 0 {
            s.log.clear();
        }
        for l in s.log.iter_mut().flatten() {
            *l = l.clone().narrow(0, 0, keep);
        }
        s.end = len;
        len
    }

    fn commit(&mut self, slots: impl IntoIterator<Item = usize>) {
        for slot in slots {
            if let Some(s) = self.slots.get_mut(slot) {
                s.log = std::mem::take(&mut s.staged);
                s.end = s.staged_end;
            }
        }
    }
}

// ---- the pass ---------------------------------------------------------------

/// The pairs a compressor completes in a pass and where their rows go.
struct CmpPlan {
    ratio: usize,
    tables: Tensor<2, Int>,
    /// Ratio 2: indices into `[pending rows ; the call's rows]` of the two
    /// tokens of each completed pair.
    left: Tensor<1, Int>,
    right: Tensor<1, Int>,
    /// Pool row of each completed row.
    dest: Tensor<1, Int>,
    /// RoPE position (`j * ratio`) of each completed row.
    rope_pos: Tensor<1, Int>,
    /// Ratio 2: the sequences whose first token completes a pair begun by the
    /// previous call, as `(sequence, slot)`; their pending rows come first.
    pending: Vec<(usize, usize)>,
    /// Rows completed (0 when no pair ends in this call).
    m: usize,
}

/// What the layers of one pass share.
struct HyperPass {
    /// Rows of the pass.
    rows: usize,
    /// The engram row ids of each engram layer, by ordinal.
    engram: Vec<Option<Vec<u64>>>,
    row_seq: Tensor<1, Int>,
    row_pos: Tensor<1, Int>,
    /// The window group's tables and each row's destination in the pool.
    win_tables: Tensor<2, Int>,
    win_dest: Tensor<1, Int>,
    /// One plan per compressed group of the layout (index = group).
    cmp: Vec<Option<CmpPlan>>,
    /// `(first row, rows, slot)` per sequence.
    seqs: Vec<(usize, usize, usize)>,
    /// The largest position of the pass's rows.
    max_pos: usize,
    /// The compressed group each layer reads (`KvLayout::csa_group`).
    csa_group: Vec<Option<usize>>,
}

/// The picks the indexer layers hand to the layers after them.
#[derive(Default)]
struct Shared {
    picks: Option<Tensor<2, Int>>,
    cand: Option<Tensor<2, Int>>,
}

fn int1(values: Vec<i32>, device: &Device) -> Tensor<1, Int> {
    let n = values.len();
    Tensor::from_data(TensorData::new(values, [n]), device)
}

fn int2(values: Vec<i32>, shape: [usize; 2], device: &Device) -> Tensor<2, Int> {
    Tensor::from_data(TensorData::new(values, shape), device)
}

/// The block tables of `segs` for group `g`, padded with -1 to the longest:
/// `[sequences, width]`.
fn table_tensor(segs: &[Seg<'_>], g: usize, device: &Device) -> Result<Tensor<2, Int>, CucaError> {
    let width = segs
        .iter()
        .map(|s| s.tables.get(g).map_or(0, Vec::len))
        .max()
        .unwrap_or(0)
        .max(1);
    let mut values = vec![-1i32; segs.len() * width];
    for (i, s) in segs.iter().enumerate() {
        let table = s
            .tables
            .get(g)
            .ok_or_else(|| err("the sequence has no block table for a group of the layout"))?;
        for (j, &b) in table.iter().enumerate() {
            if b != NO_BLOCK {
                values[i * width + j] = b as i32;
            }
        }
    }
    Ok(int2(values, [segs.len(), width], device))
}

/// The row scatter writes every destination once per launch (two writes to
/// one row race): fail when a plan repeats one.
fn distinct(dest: &[i32], what: &str) -> Result<(), CucaError> {
    let mut sorted = dest.to_vec();
    sorted.sort_unstable();
    if sorted.windows(2).any(|w| w[0] == w[1]) {
        return Err(err(format!(
            "two {what} rows of one pass map to the same pool row"
        )));
    }
    Ok(())
}

/// The pool row of row `i` of a sequence, through its table.
fn pool_row(table: &[u32], i: usize) -> Result<i32, CucaError> {
    match table.get(i / BLOCK_SIZE) {
        Some(&b) if b != NO_BLOCK => Ok((b as usize * BLOCK_SIZE + i % BLOCK_SIZE) as i32),
        _ => Err(err(format!("row {i} has no block in the sequence's table"))),
    }
}

impl Weights {
    fn hyper(&self) -> Result<&HyperWeights, CucaError> {
        self.hyper
            .as_ref()
            .ok_or_else(|| err("a hyper-connection pass without hyper-connections"))
    }

    /// The pass inputs of the sequences `segs`: positions, tables, window
    /// destinations and the compressor plans.
    fn hyper_pass(
        &self,
        state: &HyperState,
        layout: &KvLayout,
        segs: &[Seg<'_>],
    ) -> Result<HyperPass, CucaError> {
        let hp = &self.hp;
        let dev = &self.device;
        let t: usize = segs.iter().map(|s| s.t).sum();
        let mut row_seq = Vec::with_capacity(t);
        let mut row_pos = Vec::with_capacity(t);
        let mut seen = Vec::with_capacity(segs.len());
        for (i, s) in segs.iter().enumerate() {
            if seen.contains(&s.slot) {
                return Err(err(format!(
                    "state slot {} appears twice in one step",
                    s.slot
                )));
            }
            seen.push(s.slot);
            if s.slot >= state.slots.len() {
                return Err(err(format!("state slot {} is out of range", s.slot)));
            }
            for p in 0..s.t {
                row_seq.push(i as i32);
                row_pos.push((s.start + p) as i32);
            }
        }
        let wgroup = hp
            .layers
            .iter()
            .enumerate()
            .find_map(|(il, l)| layout.layer_group[il].filter(|_| l.owns_kv(il)))
            .ok_or_else(|| err("the model has no window group"))?;
        let mut win_dest = Vec::with_capacity(t);
        for s in segs {
            let table = s
                .tables
                .get(wgroup)
                .ok_or_else(|| err("the sequence has no window block table"))?;
            for p in 0..s.t {
                win_dest.push(pool_row(table, s.start + p)?);
            }
            // The attention kernel skips a window row without a block where
            // the CPU engine fails: every block the window of the call's
            // rows reaches must be in the table.
            if s.t > 0 {
                let window = layout.groups[wgroup].window.unwrap_or(usize::MAX);
                let first = s.start.saturating_sub(window.saturating_sub(1));
                for block in first / BLOCK_SIZE..=(s.start + s.t - 1) / BLOCK_SIZE {
                    pool_row(table, block * BLOCK_SIZE)?;
                }
            }
        }
        distinct(&win_dest, "window")?;
        let mut cmp: Vec<Option<CmpPlan>> = (0..layout.groups.len()).map(|_| None).collect();
        // The groups the layers this device holds read (a layer before
        // `first` runs elsewhere, with its own state).
        let used: Vec<usize> = layout.csa_group[self.first..]
            .iter()
            .flatten()
            .copied()
            .collect();
        for (g, group) in layout.groups.iter().enumerate() {
            if !group.compressed || !used.contains(&g) {
                continue;
            }
            cmp[g] = Some(self.cmp_plan(state, segs, g, group.stride)?);
        }
        Ok(HyperPass {
            row_seq: int1(row_seq, dev),
            row_pos: int1(row_pos, dev),
            win_tables: table_tensor(segs, wgroup, dev)?,
            win_dest: int1(win_dest, dev),
            cmp,
            seqs: segs.iter().map(|s| (s.offset, s.t, s.slot)).collect(),
            max_pos: segs.iter().map(|s| s.start + s.t - 1).max().unwrap_or(0),
            csa_group: layout.csa_group.clone(),
            engram: self.engram_ids(segs)?,
            rows: t,
        })
    }

    /// The rows the compressors of group `g` (rows of `ratio` tokens)
    /// complete in this pass.
    fn cmp_plan(
        &self,
        state: &HyperState,
        segs: &[Seg<'_>],
        g: usize,
        ratio: usize,
    ) -> Result<CmpPlan, CucaError> {
        let dev = &self.device;
        let mut pending: Vec<(usize, usize)> = Vec::new();
        if ratio == 2 {
            for (i, s) in segs.iter().enumerate() {
                if s.start % 2 == 1 {
                    let slot = &state.slots[s.slot];
                    if slot.end != s.start || slot.log.iter().flatten().next().is_none() {
                        return Err(err(format!(
                            "the compressor state of slot {} is not at position {}",
                            s.slot, s.start
                        )));
                    }
                    pending.push((i, s.slot));
                }
            }
        }
        let (mut left, mut right, mut dest, mut rope_pos) = (vec![], vec![], vec![], vec![]);
        let mut next_pending = 0;
        for s in segs {
            let table = s
                .tables
                .get(g)
                .ok_or_else(|| err("the sequence has no compressed block table"))?;
            let pend = (ratio == 2 && s.start % 2 == 1).then(|| {
                next_pending += 1;
                next_pending - 1
            });
            for i in 0..s.t {
                let p = s.start + i;
                if ratio == 2 {
                    if p % 2 == 0 {
                        continue;
                    }
                    let r = pending.len() + s.offset + i;
                    left.push(if i >= 1 {
                        (r - 1) as i32
                    } else {
                        pend.unwrap_or(0)
                    });
                    right.push(r as i32);
                }
                let j = p / ratio;
                dest.push(pool_row(table, j)?);
                rope_pos.push((j * ratio) as i32);
            }
        }
        distinct(&dest, "compressed")?;
        let m = dest.len();
        Ok(CmpPlan {
            ratio,
            tables: table_tensor(segs, g, dev)?,
            left: int1(left, dev),
            right: int1(right, dev),
            dest: int1(dest, dev),
            rope_pos: int1(rope_pos, dev),
            pending,
            m,
        })
    }

    /// The layers over the concatenated rows `embeddings` of `segs`, then
    /// the final norm and the LM head over the rows the sequences want.
    pub(super) fn run_hyper(
        &self,
        state: &mut HyperState,
        layout: &KvLayout,
        embeddings: Vec<f32>,
        segs: &[Seg<'_>],
    ) -> Result<Vec<Forward>, CucaError> {
        let hy = self.hyper()?;
        let (s, d) = (hy.spec.streams, self.hp.n_embd);
        let t = embeddings.len() / d;
        let x = Tensor::<2>::from_data(TensorData::new(embeddings, [t, d]), &self.device);
        let stream = hc_kernel::stream_repeat(x, &hc_shape(&self.hp, hy));
        // The first layer reads stream 0 alone.
        let cl = super::super::hc::coeff_len(s);
        let mut first = vec![0f32; t * cl];
        for row in first.chunks_exact_mut(cl) {
            row[0] = 1.0;
        }
        let pre = Tensor::<2>::from_data(TensorData::new(first, [t, cl]), &self.device);
        self.run_hyper_from(state, layout, stream, pre, segs)
    }

    /// [`Weights::run_hyper`] for the layers after the ones that run
    /// elsewhere (layer offload): `stream` `[rows, S * D]` and the pre weights
    /// `pre` `[rows, S]` the layers before them produced.
    pub(super) fn run_hyper_offloaded(
        &self,
        state: &mut HyperState,
        layout: &KvLayout,
        stream: Vec<f32>,
        pre: Vec<f32>,
        segs: &[Seg<'_>],
    ) -> Result<Vec<Forward>, CucaError> {
        let hy = self.hyper()?;
        let s = hy.spec.streams;
        let t = pre.len() / s;
        if stream.len() != t * s * self.hp.n_embd {
            return Err(err(format!(
                "the stream holds {} values for {t} rows of {} streams of {}",
                stream.len(),
                s,
                self.hp.n_embd
            )));
        }
        let cl = super::super::hc::coeff_len(s);
        let mut coeffs = vec![0f32; t * cl];
        for (row, p) in coeffs.chunks_exact_mut(cl).zip(pre.chunks_exact(s)) {
            row[..s].copy_from_slice(p);
        }
        let stream = Tensor::<2>::from_data(
            TensorData::new(stream, [t, s * self.hp.n_embd]),
            &self.device,
        );
        let pre = Tensor::<2>::from_data(TensorData::new(coeffs, [t, cl]), &self.device);
        self.run_hyper_from(state, layout, stream, pre, segs)
    }

    fn run_hyper_from(
        &self,
        state: &mut HyperState,
        layout: &KvLayout,
        mut stream: Tensor<2>,
        mut pre: Tensor<2>,
        segs: &[Seg<'_>],
    ) -> Result<Vec<Forward>, CucaError> {
        let hp = &self.hp;
        let hy = self.hyper()?;
        let pass = self.hyper_pass(state, layout, segs)?;
        for seg in segs {
            let slot = &mut state.slots[seg.slot];
            slot.staged = vec![None; hy.n_owners];
            slot.staged_end = seg.start + seg.t;
        }
        let mut shared = Shared::default();
        for il in self.first..hp.n_layer {
            let (next, coeffs) = self.hyper_layer(il, state, &pass, stream, pre, &mut shared)?;
            stream = next;
            pre = coeffs;
        }
        let x = hc_kernel::hc_collapse(stream, pre, &hc_shape(hp, hy));
        let outs = self.finish(x, segs)?;
        state.commit(segs.iter().map(|s| s.slot));
        Ok(outs)
    }

    /// One layer: attention then FFN between their hyper-connection collapse
    /// and expansion. Returns the new stream and the FFN's coefficients, the
    /// pre weights of which the next layer's attention reads.
    fn hyper_layer(
        &self,
        il: usize,
        state: &mut HyperState,
        pass: &HyperPass,
        mut stream: Tensor<2>,
        pre: Tensor<2>,
        shared: &mut Shared,
    ) -> Result<(Tensor<2>, Tensor<2>), CucaError> {
        let hp = &self.hp;
        let hy = self.hyper()?;
        let lw: &LayerWeights = &self.layers[il - self.first];
        let (Some(hc), Mixer::Csa(cw)) = (lw.hc.as_deref(), &lw.mixer) else {
            return Err(err(format!("layer {il} has no hyper-connection weights")));
        };
        let shape = hc_shape(hp, hy);
        let eps = hp.eps;
        if let (Some(eng), Some(es)) = (&hc.engram, hp.engram.as_ref()) {
            let ids = pass.engram[eng.ordinal]
                .as_deref()
                .ok_or_else(|| err("an engram layer without row ids"))?;
            let t = pass.rows;
            stream = self.engram_layer(eng, es, stream, ids, t)?;
        }

        let attn_c = hc_kernel::hc_coefficients(
            stream.clone(),
            hc.attn.mix.clone(),
            hc.attn.base.clone(),
            hc.attn.scale.clone(),
            &shape,
        );
        let x = hc_kernel::hc_collapse(stream.clone(), pre, &shape);
        let h = rms(x, Some(&hc.attn_norm), eps);
        let y = self.csa_attention(il, cw, h, pass, state, shared)?;
        let next = hc_kernel::hc_expand(y, stream, attn_c.clone(), &shape);

        let ffn_c = hc_kernel::hc_coefficients(
            next.clone(),
            hc.ffn.mix.clone(),
            hc.ffn.base.clone(),
            hc.ffn.scale.clone(),
            &shape,
        );
        let x = hc_kernel::hc_collapse(next.clone(), attn_c, &shape);
        let f = rms(x, Some(&hc.ffn_norm), eps);
        let f = lw.ffn.forward(f, hp.act(), &self.device)?;
        Ok((hc_kernel::hc_expand(f, next, ffn_c.clone(), &shape), ffn_c))
    }

    /// The row ids of every engram layer this device holds for the rows of
    /// `segs`, by ordinal (`None` for a layer it does not hold). They depend
    /// on token ids only, so a pass computes them all at its entry and starts
    /// the reads, and the reads of the later layers overlap the compute of the
    /// earlier ones.
    fn engram_ids(&self, segs: &[Seg<'_>]) -> Result<Vec<Option<Vec<u64>>>, CucaError> {
        let (Some(es), Some(source)) = (self.hp.engram.as_ref(), self.hyper()?.engram.as_deref())
        else {
            return Ok(Vec::new());
        };
        let t: usize = segs.iter().map(|s| s.t).sum();
        let cols = es.cols();
        let mut by_ordinal: Vec<Option<Vec<u64>>> = vec![None; es.layers.len()];
        for eng in self
            .layers
            .iter()
            .filter_map(|l| l.hc.as_deref()?.engram.as_ref())
        {
            let mut ids = vec![0u64; t * cols];
            for seg in segs {
                if seg.start > 0 && seg.history.len() != seg.start {
                    return Err(err(format!(
                        "the engram needs the {} tokens before position {} (got {})",
                        seg.start,
                        seg.start,
                        seg.history.len()
                    )));
                }
                let at = seg.offset * cols;
                engram::sequence_row_ids(
                    es,
                    eng.ordinal,
                    seg.tokens,
                    seg.history,
                    &mut ids[at..at + seg.t * cols],
                )?;
            }
            source.prefetch(&ids);
            by_ordinal[eng.ordinal] = Some(ids);
        }
        Ok(by_ordinal)
    }

    /// Start reading, in the background, the table rows a later pass over
    /// `tokens` (the next ones of a sequence, which follow `history`) will
    /// look up. A hint: a model without an engram table, or a bad id, does
    /// nothing.
    pub(super) fn engram_prefetch(&self, history: &[u32], tokens: &[u32]) {
        let (Some(es), Some(hy)) = (self.hp.engram.as_ref(), self.hyper.as_ref()) else {
            return;
        };
        let Some(source) = hy.engram.as_deref() else {
            return;
        };
        let mut ids = vec![0u64; tokens.len() * es.cols()];
        for eng in self
            .layers
            .iter()
            .filter_map(|l| l.hc.as_deref()?.engram.as_ref())
        {
            if engram::sequence_row_ids(es, eng.ordinal, tokens, history, &mut ids).is_ok() {
                source.prefetch(&ids);
            }
        }
    }

    /// The counters of the engram table, when this device holds one.
    pub(super) fn engram_counters(&self) -> Option<std::sync::Arc<EngramCounters>> {
        self.hyper.as_ref()?.engram.as_deref()?.counters()
    }

    /// Add the hashed table lookup of an engram layer to every stream.
    fn engram_layer(
        &self,
        eng: &GEngram,
        es: &EngramSpec,
        stream: Tensor<2>,
        ids: &[u64],
        t: usize,
    ) -> Result<Tensor<2>, CucaError> {
        let hp = &self.hp;
        let hy = self.hyper()?;
        let source = hy
            .engram
            .as_deref()
            .ok_or_else(|| err("an engram layer without a table"))?;
        let cols = es.cols();
        let mut rows = vec![0f32; t * cols * es.row_dim];
        source.rows(ids, &mut rows)?;
        let rows =
            Tensor::<2>::from_data(TensorData::new(rows, [t, cols * es.row_dim]), &self.device);
        let kv = linear(rows, &eng.kv);
        Ok(hc_kernel::engram_gate(
            stream,
            kv,
            eng.gate.clone(),
            &hc_shape(hp, hy),
        ))
    }

    /// The attention of layer `il` over the normed rows `h` `[t, n_embd]`:
    /// `[t, n_embd]`.
    fn csa_attention(
        &self,
        il: usize,
        cw: &GCsa,
        h: Tensor<2>,
        pass: &HyperPass,
        state: &mut HyperState,
        shared: &mut Shared,
    ) -> Result<Tensor<2>, CucaError> {
        let hp = &self.hp;
        let hy = self.hyper()?;
        let g: &CsaGlobals = hp
            .csa
            .as_ref()
            .ok_or_else(|| err("no attention geometry"))?;
        let spec = &hp.layers[il];
        let (nh, hd) = (spec.n_head, spec.head_dim);
        let eps = hp.eps;
        let rope = &self.ropes[spec
            .rope
            .ok_or_else(|| err("layer without a rotary table"))?];
        let rot = 2 * rope.cos.dims()[1];
        let qdq = hy.qdq;

        // Queries and the window key/value row of every token.
        let qn = rms(linear(h.clone(), &cw.q_a), Some(&cw.q_a_norm), eps);
        let q = linear(qn.clone(), &cw.q_b);
        let q = csa_kernel::rope_tail(
            q,
            &rope.cos,
            &rope.sin,
            pass.row_pos.clone(),
            hd,
            rot,
            false,
        );
        let kv = rms(linear(h.clone(), &cw.kv), Some(&cw.kv_norm), eps);
        let mut kv = csa_kernel::rope_tail(
            kv,
            &rope.cos,
            &rope.sin,
            pass.row_pos.clone(),
            hd,
            rot,
            false,
        );
        if qdq {
            kv = qdq_kernel::qdq_fp8_ue8m0(kv, 32);
        }
        let window = state.window[il]
            .take()
            .ok_or_else(|| err(format!("layer {il} has no window pool")))?;
        let window = csa_kernel::scatter_rows(window, kv, pass.win_dest.clone());
        state.window[il] = Some(window);

        if let (Some(csa), Some(comp)) = (spec.csa, &cw.compressor) {
            self.compress(il, &csa, comp, h.clone(), pass, state)?;
        }
        if let Some(csa) = spec.csa {
            if let Some(ix) = &cw.indexer {
                self.index(il, &csa, ix, h, qn, pass, state, shared)?;
            } else if shared.picks.is_none() {
                return Err(err(format!(
                    "layer {il} reuses the indexer's picks but no layer computed them"
                )));
            }
        }

        let cmp = match spec.csa {
            Some(csa) => {
                let group = pass_group(pass, il)?;
                let pool = state.compressed[csa.kv_source]
                    .as_ref()
                    .ok_or_else(|| err("compressed rows without a pool"))?;
                let plan = pass.cmp[group]
                    .as_ref()
                    .ok_or_else(|| err("no compressed plan"))?;
                let picks = shared
                    .picks
                    .clone()
                    .ok_or_else(|| err("no picks for an attention layer with compressed rows"))?;
                Some(csa_kernel::CmpRead {
                    pool: pool.clone(),
                    tables: plan.tables.clone(),
                    picks,
                })
            }
            None => None,
        };
        let o = csa_kernel::csa_attention(
            q,
            state.window[il]
                .as_ref()
                .ok_or_else(|| err("no window pool"))?,
            &pass.win_tables,
            cmp.as_ref(),
            pass.row_seq.clone(),
            pass.row_pos.clone(),
            &cw.sinks,
            &csa_kernel::CsaCall {
                heads: nh,
                head_dim: hd,
                window: spec.window.unwrap_or(usize::MAX),
                scale: spec.scale,
            },
        );

        // Undo the query's rotation, simulate the output's quantization and
        // project through the grouped low-rank output.
        let mut o =
            csa_kernel::rope_tail(o, &rope.cos, &rope.sin, pass.row_pos.clone(), hd, rot, true);
        if qdq {
            o = qdq_kernel::qdq_fp8_ue8m0(o, 32);
        }
        let width = nh * hd / g.out_groups;
        let parts: Vec<Tensor<2>> = cw
            .out_a
            .iter()
            .enumerate()
            .map(|(grp, w)| linear(o.clone().narrow(1, grp * width, width), w))
            .collect();
        let mid = if parts.len() == 1 {
            parts.into_iter().next().unwrap_or_else(|| unreachable!())
        } else {
            Tensor::cat(parts, 1)
        };
        Ok(linear(mid, &cw.out_b))
    }

    /// The compressed rows the tokens of this call complete: pool them,
    /// derive the index keys, simulate the quantization and write them into
    /// the sequences' compressed blocks.
    fn compress(
        &self,
        il: usize,
        csa: &CsaSpec,
        comp: &GCompressor,
        h: Tensor<2>,
        pass: &HyperPass,
        state: &mut HyperState,
    ) -> Result<(), CucaError> {
        let hp = &self.hp;
        let hy = self.hyper()?;
        let spec = &hp.layers[il];
        let (hd, eps) = (spec.head_dim, hp.eps);
        let ix = &hp
            .csa
            .as_ref()
            .ok_or_else(|| err("no attention geometry"))?
            .indexer;
        let rope = &self.ropes[spec
            .rope
            .ok_or_else(|| err("layer without a rotary table"))?];
        let rot = 2 * rope.cos.dims()[1];
        let group = pass_group(pass, il)?;
        let plan = pass.cmp[group]
            .as_ref()
            .ok_or_else(|| err("no compressed plan"))?;
        let kv = linear(h.clone(), &comp.kv);
        let latent = if let (Some(gate), Some(k)) = (&comp.gate, hy.owners[il]) {
            let score = linear(h, gate);
            // The rows of the call, for the next call and rollbacks.
            let both = Tensor::cat(vec![kv.clone(), score.clone()], 1);
            for &(first, rows, slot) in &pass.seqs {
                let n = rows.min(super::super::cpu::MAX_VERIFY_ROWS + 1);
                state.slots[slot].staged[k] = Some(both.clone().narrow(0, first + rows - n, n));
            }
            if plan.m == 0 {
                return Ok(());
            }
            // The pending rows (of the sequences that begin on an odd
            // position) precede the call's rows.
            let (kv_ext, score_ext) = if plan.pending.is_empty() {
                (kv, score)
            } else {
                let rows = |slot: usize, at: usize| {
                    state.slots[slot]
                        .log
                        .get(k)
                        .and_then(Option::as_ref)
                        .map(|l| {
                            let n = l.dims()[0];
                            l.clone().narrow(0, n - 1, 1).narrow(1, at, hd)
                        })
                };
                let mut pk = Vec::new();
                let mut ps = Vec::new();
                for &(_, slot) in &plan.pending {
                    pk.push(
                        rows(slot, 0).ok_or_else(|| err("a pending compressor row is missing"))?,
                    );
                    ps.push(
                        rows(slot, hd).ok_or_else(|| err("a pending compressor row is missing"))?,
                    );
                }
                pk.push(kv);
                ps.push(score);
                (Tensor::cat(pk, 0), Tensor::cat(ps, 0))
            };
            let pooled =
                csa_kernel::pair_pool(kv_ext, score_ext, plan.left.clone(), plan.right.clone());
            rms(pooled, Some(&comp.norm), eps)
        } else {
            if plan.m == 0 {
                return Ok(());
            }
            rms(kv, Some(&comp.norm), eps)
        };

        let key = rms(
            linear(latent.clone(), &comp.index_key),
            Some(&comp.index_norm),
            eps,
        );
        let mut key = csa_kernel::rope_tail(
            key,
            &rope.cos,
            &rope.sin,
            plan.rope_pos.clone(),
            ix.dim,
            rot,
            false,
        );
        let mut lat = csa_kernel::rope_tail(
            latent,
            &rope.cos,
            &rope.sin,
            plan.rope_pos.clone(),
            hd,
            rot,
            false,
        );
        if hy.qdq {
            key = qdq_kernel::qdq_fp4_e8m0(key, 32);
            lat = qdq_kernel::qdq_fp4_e4m3(lat, 16);
        }
        let rows = Tensor::cat(vec![lat, key], 1);
        let pool = state.compressed[il]
            .take()
            .ok_or_else(|| err("a compressing layer without a compressed pool"))?;
        state.compressed[il] = Some(csa_kernel::scatter_rows(pool, rows, plan.dest.clone()));
        let _ = csa;
        Ok(())
    }

    /// Score the compressed rows each query can see and keep the best.
    #[expect(clippy::too_many_arguments, reason = "the pass's independent inputs")]
    fn index(
        &self,
        il: usize,
        csa: &CsaSpec,
        ix: &GIndexer,
        h: Tensor<2>,
        qn: Tensor<2>,
        pass: &HyperPass,
        state: &HyperState,
        shared: &mut Shared,
    ) -> Result<(), CucaError> {
        let hp = &self.hp;
        let hy = self.hyper()?;
        let g = hp
            .csa
            .as_ref()
            .ok_or_else(|| err("no attention geometry"))?;
        let spec = &hp.layers[il];
        let idx = &g.indexer;
        let rope = &self.ropes[spec
            .rope
            .ok_or_else(|| err("layer without a rotary table"))?];
        let rot = 2 * rope.cos.dims()[1];
        let mut iq = linear(qn, &ix.q_b);
        iq = csa_kernel::rope_tail(
            iq,
            &rope.cos,
            &rope.sin,
            pass.row_pos.clone(),
            idx.dim,
            rot,
            false,
        );
        if hy.qdq {
            iq = qdq_kernel::qdq_fp4_e8m0(iq, 32);
        }
        let k = (idx.dim as f64).powf(-0.5) * (idx.heads as f64).powf(-0.5);
        let wts = linear(h, &ix.proj).mul_scalar(k);
        let group = pass_group(pass, il)?;
        let plan = pass.cmp[group]
            .as_ref()
            .ok_or_else(|| err("no compressed plan"))?;
        let source = hp.layers[csa.kv_source]
            .csa
            .ok_or_else(|| err("a cache source without its spec"))?;
        let pool = state.compressed[csa.kv_source]
            .as_ref()
            .ok_or_else(|| err("an indexer without compressed rows"))?;
        let (cand_blocks, cand_size) = idx.candidates.map_or((0, 1), |(_, b, s)| (b, s));
        let cand_in = match csa.candidates {
            Candidates::Consumer => Some(shared.cand.clone().ok_or_else(|| {
                err(format!(
                    "layer {il} narrows to candidate blocks but no layer published them"
                ))
            })?),
            _ => None,
        };
        let out = topk_kernel::index_topk(
            &topk_kernel::IndexCall {
                heads: idx.heads,
                dim: idx.dim,
                key_col: hp.layers[il].head_dim,
                row_width: source.row_width,
                ratio: plan.ratio,
                top_k: idx.top_k,
                cand: csa.candidates,
                cand_blocks,
                cand_size,
                max_visible: pass.max_pos / plan.ratio + 1,
            },
            iq,
            wts,
            pool,
            &plan.tables,
            pass.row_seq.clone(),
            pass.row_pos.clone(),
            cand_in.as_ref(),
        );
        shared.picks = Some(out.picks);
        if csa.candidates == Candidates::Source {
            shared.cand = out.cand;
        }
        Ok(())
    }
}

/// The compressed group of layer `il`: the pass's plans are indexed by it.
fn pass_group(pass: &HyperPass, il: usize) -> Result<usize, CucaError> {
    pass.csa_group
        .get(il)
        .copied()
        .flatten()
        .ok_or_else(|| err(format!("layer {il} reads no compressed group")))
}

fn hc_shape(hp: &Hparams, hy: &HyperWeights) -> hc_kernel::HcShape {
    hc_kernel::HcShape {
        streams: hy.spec.streams,
        dim: hp.n_embd,
        sinkhorn_iters: hy.spec.sinkhorn_iters,
        hc_eps: hy.spec.eps,
        norm_eps: hp.eps as f32,
    }
}

#[cfg(test)]
thread_local! {
    static QDQ_OFF: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Whether models loaded on this thread simulate no quantization.
#[cfg(test)]
fn qdq_disabled() -> bool {
    QDQ_OFF.get()
}

#[cfg(not(test))]
fn qdq_disabled() -> bool {
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
