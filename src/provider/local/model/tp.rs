//! Tensor parallelism over GPU shards: how one model is split across
//! `world = 1 + shards.len()` devices, and the all-reduce that joins the
//! halves of every layer.
//!
//! This module holds the parts of the split that do not touch a weight or a
//! tensor of the forward pass: the rules a model must meet
//! ([`validate`]), which heads, rows and columns a shard owns ([`Tp`],
//! [`Heads`], [`Cut`]), the byte arithmetic of a column cut
//! ([`copy_columns`], [`column_dtype`]) and the reduction ([`Collective`]).
//! [`gpu`](super::gpu) applies them: it reads each shard's weights, runs
//! the layers and calls [`Collective::all_reduce_sum`] twice per layer.
//!
//! # The split
//!
//! The split is Megatron/vLLM style. Every shard holds the whole residual
//! stream and runs a slice of each layer:
//!
//! - Attention: shard `s` computes the query heads
//!   `[s * n_head / world, (s + 1) * n_head / world)` (its rows of `attn_q`,
//!   or of the three parts of a fused `attn_qkv`), the key/value heads they
//!   read, attention over its own K/V pools, and the output projection over
//!   its heads' columns of `attn_output`. That product is a partial sum of
//!   the layer's `[t, n_embd]` output.
//! - KV-head replication: when `n_kv_head < world` (`world % n_kv_head
//!   == 0`, e.g. Gemma 4 E2B's single KV head on two shards) each shard keeps
//!   a full copy of the key/value head its query heads share: shard `s`
//!   serves KV head `s * n_kv_head / world`, and the `n_head / world` query
//!   heads of the shard all map to it. With `n_kv_head % world == 0` the KV
//!   heads divide like the query heads. The K/V pools of a shard hold its
//!   own KV heads only, so the pools of all shards together exceed the
//!   single-device pool by the replication factor
//!   (`GpuTransformer::kv_pool_bytes` sums them).
//! - Dense FFN: `ffn_gate` and `ffn_up` rows `[s * ff / world ..)` (a
//!   fused `ffn_up` holds `[gate | up]`: both halves are cut) and the
//!   matching `ffn_down` columns; the product is a partial sum again.
//! - Mixture of experts: every shard holds all experts, each sliced to
//!   `ff_exp / world` units (gate and up rows, down columns), and the shared
//!   expert is sliced the same way. The router is replicated, so every shard
//!   routes every token to the same experts and the MoE kernels keep indexing
//!   experts by id. The routing-weighted sum over experts is linear in the
//!   down projection, so the shards' partial outputs add up to the single
//!   device's result (the shared expert's per-token sigmoid gate is
//!   replicated too). The shards route from bit-identical inputs, so they
//!   agree as long as their devices run the router matmul identically
//!   (devices of one kind do; a near tie between two experts could otherwise
//!   route differently on a differently tuned device).
//! - Replicated: norms, the per-head Q/K norms, the router, the shared
//!   expert's gate vector, the per-layer-embedding block and layer output
//!   scale of Gemma 4, the RoPE tables, and the bias of `attn_output` (added
//!   once after the reduction). Q/K/V biases are cut with their rows. Only
//!   shard 0 holds the LM head; the final norm and the head run on its copy
//!   of the residual stream.
//!
//! A layer runs as: every shard's attention half; all-reduce; every
//! shard's `attn_output` bias, post-attention norm and residual add; every
//! shard's FFN half; all-reduce; every shard's post-FFN norm, residual
//! add, per-layer-embedding block and output scale. With one shard the
//! reductions are the identity.
//!
//! # Rules
//!
//! [`validate`] runs before any weight is read, for `world > 1` only, and
//! fails with [`CucaError::Config`] (`tensor parallelism: <rule> (<numbers>)`)
//! unless, for every layer:
//!
//! - `n_head % world == 0`;
//! - `n_kv_head % world == 0` or `world % n_kv_head == 0` (replication);
//! - `(n_head / world) * head_dim` is a multiple of 32 (the columns a shard
//!   reads of `attn_output` must start on a Q4_0/Q8_0 block);
//! - the dense FFN width, the expert FFN width and the shared expert width
//!   are multiples of `32 * world`;
//!
//! and the model has no recurrent (delta-net) layers, no latent attention
//! (one shared latent K/V row per position, up-projections not sliced), no
//! full-width Q/K norm (OLMo 2, OLMoE: the norm spans all heads, so a shard
//! cannot compute it alone) and no tiered or host experts. A drafter is not
//! supported either (the engine rejects it before the load).
//!
//! # Column cuts of other dtypes
//!
//! A column range of a weight is a per-row byte sub-slice when both bounds
//! lie on the dtype's block boundaries (every 32 columns for Q4_0, Q4_1,
//! Q5_x, Q8_x; every column for F32/F16/BF16; every 256 for the K-quants).
//! A cut that splits a block (a K-quant `ffn_down` whose slice is not a
//! multiple of 256 wide) dequantizes the rows, cuts the f32 values and
//! quantizes them to Q8_0 (f32 for a weight the loader keeps dense): the
//! same values the unsplit loader would hold for those columns, since Q8_0
//! blocks are 32 columns wide and the cut lies on 32-column bounds.
//!
//! # All-reduce
//!
//! [`Collective::all_reduce_sum`] returns, for the partial `[t, n_embd]`
//! outputs of every shard (one tensor per device, in shard order), the sum on
//! every device.
//!
//! - CUDA (every device a distinct CUDA device, Linux): burn 0.22's
//!   `all_reduce` over a `DistributedContext` that the model holds for its
//!   lifetime (NCCL through CubeCL), in place on each device.
//! - Everything else (Vulkan, Metal, ROCm, a repeated device such as
//!   `vulkan:0+vulkan:0`, mixed backends): host-staged. Every part is read
//!   back, summed in f32 in shard order `0, 1, .., world - 1` (a fixed order,
//!   so two runs agree bit for bit) and uploaded to every device. That costs
//!   two host round trips per layer; speed is not the goal of this path.
//!
//! The single-device matmul sums the whole inner dimension in one kernel and
//! a sharded layer sums two half-width partials, so the logits agree to
//! float rounding, not bit for bit.

use std::ops::Range;

use crate::provider::local::gguf;
#[cfg(feature = "provider-local-cuda")]
use burn::tensor::distributed::{
    CollectiveTensor, DistributedConfig, DistributedContext, ReduceOperation, all_reduce,
};
use burn::tensor::{Device, Tensor, TensorData};
use candle_core::quantized::GgmlDType;

use super::gpu::{push_rows, read_f32};
use super::layout::{Ffn, LayerLayout, Qkv, name};
use super::params::Hparams;
use crate::error::CucaError;
use crate::provider::local::{ExpertPlacement, LocalDevice};

/// Elements of a Q4_0/Q8_0 block: every cut a shard makes of a quantized
/// weight starts on one.
const BLOCK: usize = 32;

/// The `tensor parallelism: <rule> (<numbers>)` configuration error.
fn rule(rule: &str, numbers: impl std::fmt::Display) -> CucaError {
    CucaError::Config(format!("tensor parallelism: {rule} ({numbers})"))
}

/// Check that `width` splits across `world` shards in multiples of
/// [`BLOCK`].
fn check_width(what: &str, il: usize, width: usize, world: usize) -> Result<(), CucaError> {
    if width.is_multiple_of(world) && (width / world).is_multiple_of(BLOCK) {
        return Ok(());
    }
    Err(rule(
        &format!("the {what} width does not split into multiples of {BLOCK} per shard"),
        format!("layer {il}: {width}, {world} shards"),
    ))
}

/// Fail unless the model `hp` (tensors indexed by `content`) can be split
/// across `world` shards, placing its experts as `experts` says; see the
/// module docs for the rules. Reads no weight. A single shard always passes.
///
/// # Errors
///
/// [`CucaError::Config`] naming the rule and the numbers it failed on;
/// [`CucaError::Provider`] for a tensor the layout needs and the file lacks.
pub(super) fn validate(
    content: &gguf::Content,
    hp: &Hparams,
    world: usize,
    experts: &ExpertPlacement,
) -> Result<(), CucaError> {
    if world < 2 {
        return Ok(());
    }
    if let Some(il) = hp.layers.iter().position(|l| l.delta().is_some()) {
        return Err(rule(
            "recurrent (delta-net) layers cannot be split",
            format!("layer {il}"),
        ));
    }
    if let Some(il) = hp.layers.iter().position(|l| l.mla.is_some()) {
        return Err(rule(
            "multi-head latent attention layers cannot be split (one shared latent K/V row per position; the per-head up-projections are not sliced)",
            format!("layer {il}, architecture {}", hp.arch.name()),
        ));
    }
    if hp.behavior.hyper_connections {
        return Err(rule(
            "hyper-connection residual streams, compressed sparse attention, attention sinks, the grouped output projection and engram lookups cannot be split",
            format!("architecture {}", hp.arch.name()),
        ));
    }
    if hp.behavior.qk_norm_full {
        return Err(rule(
            "a full-width Q/K norm cannot be split across heads",
            format!("architecture {}", hp.arch.name()),
        ));
    }
    if matches!(experts, ExpertPlacement::Host { .. }) {
        return Err(rule(
            "host experts cannot be split across shards",
            "ExpertPlacement::Host",
        ));
    }
    if matches!(experts, ExpertPlacement::Tiered(_)) {
        return Err(rule(
            "tiered experts cannot be split across shards",
            "ExpertPlacement::Tiered",
        ));
    }
    for (il, spec) in hp.layers.iter().enumerate() {
        let (nh, nkv, d) = (spec.n_head, spec.n_kv_head, spec.head_dim);
        if !nh.is_multiple_of(world) {
            return Err(rule(
                "the attention heads do not divide across the shards",
                format!("layer {il}: {nh} heads, {world} shards"),
            ));
        }
        if !(nkv.is_multiple_of(world) || world.is_multiple_of(nkv)) {
            return Err(rule(
                "the key/value heads neither divide across nor replicate over the shards",
                format!("layer {il}: {nkv} KV heads, {world} shards"),
            ));
        }
        let width = nh / world * d;
        if !width.is_multiple_of(BLOCK) {
            return Err(rule(
                &format!("a shard's query width is not a multiple of {BLOCK}"),
                format!("layer {il}: {} heads x {d} = {width}", nh / world),
            ));
        }
        match LayerLayout::read(hp, il, spec, content)?.ffn {
            Ffn::Dense { ff, .. } => check_width("dense FFN", il, ff, world)?,
            Ffn::Moe { ff, shared_ff, .. } => {
                check_width("expert FFN", il, ff, world)?;
                if let Some(shared) = shared_ff {
                    check_width("shared expert FFN", il, shared, world)?;
                }
            }
        }
    }
    Ok(())
}

/// One shard's place in a tensor-parallel group of `world` shards.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Tp {
    pub(super) world: usize,
    pub(super) shard: usize,
}

/// The query heads of one shard and the key/value heads they read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Heads {
    pub(super) query: Range<usize>,
    pub(super) kv: Range<usize>,
}

impl Tp {
    /// The whole model on one device.
    pub(super) const SOLO: Tp = Tp { world: 1, shard: 0 };

    /// Whether the model is split at all.
    pub(super) fn is_split(self) -> bool {
        self.world > 1
    }

    /// The units `[shard * n / world, (shard + 1) * n / world)` of a
    /// dimension of `n` units this shard owns (`n` a multiple of `world`).
    pub(super) fn part(self, n: usize) -> Range<usize> {
        n * self.shard / self.world..n * (self.shard + 1) / self.world
    }

    /// The query heads of this shard and the key/value heads they read: the
    /// shard's share of the KV heads when they divide across the shards,
    /// else (`world % n_kv_head == 0`) the one KV head
    /// `shard * n_kv_head / world` that all its query heads map to.
    pub(super) fn heads(self, n_head: usize, n_kv_head: usize) -> Heads {
        let kv = if n_kv_head.is_multiple_of(self.world) {
            self.part(n_kv_head)
        } else {
            let head = self.shard * n_kv_head / self.world;
            head..head + 1
        };
        Heads {
            query: self.part(n_head),
            kv,
        }
    }

    /// `hp` as one shard sees it: the head counts of every layer cut to the
    /// shard's heads (`n_head / world`, `max(1, n_kv_head / world)`), the
    /// rest unchanged.
    pub(super) fn local_hparams(self, hp: &Hparams) -> Hparams {
        let mut local = hp.clone();
        if self.is_split() {
            for spec in &mut local.layers {
                spec.n_head /= self.world;
                spec.n_kv_head = (spec.n_kv_head / self.world).max(1);
            }
        }
        local
    }
}

impl Heads {
    /// The rows of a query projection that carry these heads: `per_head`
    /// rows each (`head_dim`, or twice that for a gated attention's
    /// `[query | gate]` packing).
    pub(super) fn query_rows(&self, per_head: usize) -> Range<usize> {
        self.query.start * per_head..self.query.end * per_head
    }

    /// The rows of a key or value projection that carry the KV heads.
    pub(super) fn kv_rows(&self, head_dim: usize) -> Range<usize> {
        self.kv.start * head_dim..self.kv.end * head_dim
    }

    /// The rows of a fused `[q | k | v]` projection (`q_width` and
    /// `kv_width` rows in the file) a shard reads, in the order it keeps
    /// them: its query rows, then its key rows, then its value rows.
    pub(super) fn fused_rows(
        &self,
        head_dim: usize,
        q_width: usize,
        kv_width: usize,
    ) -> Vec<Range<usize>> {
        let kv = self.kv_rows(head_dim);
        vec![
            self.query_rows(head_dim),
            q_width + kv.start..q_width + kv.end,
            q_width + kv_width + kv.start..q_width + kv_width + kv.end,
        ]
    }
}

/// The rows of a fused `[gate | up]` projection of `ff`-wide halves a shard
/// reads, given its `part` of `ff`.
pub(super) fn fused_halves(part: &Range<usize>, ff: usize) -> Vec<Range<usize>> {
    vec![part.clone(), ff + part.start..ff + part.end]
}

/// The cut of the `[rows, cols]` weight `part` (a `name::*` part: `attn_q`,
/// `ffn_down_shexp`, ..) of layer `il` that shard `tp` reads, the whole
/// matrix for the replicated parts (and for a single shard). `hp` is the
/// model's own (unsplit) hyperparameters and `layout` the layer's.
pub(super) fn part_cut(
    hp: &Hparams,
    layout: &LayerLayout,
    il: usize,
    part: &str,
    (rows, cols): (usize, usize),
    tp: Tp,
) -> Cut {
    let whole = Cut::whole(rows, cols);
    let Some(spec) = hp.layers.get(il).filter(|_| tp.is_split()) else {
        return whole;
    };
    let d = spec.head_dim;
    let heads = tp.heads(spec.n_head, spec.n_kv_head);
    match part {
        name::ATTN_Q => Cut::rows_only(heads.query_rows(rows / spec.n_head.max(1)), cols),
        // A gate projection of its own has the query's rows.
        name::ATTN_GATE if hp.behavior.attn_gate_tensor => {
            Cut::rows_only(heads.query_rows(rows / spec.n_head.max(1)), cols)
        }
        name::ATTN_K | name::ATTN_V => Cut::rows_only(heads.kv_rows(d), cols),
        name::ATTN_QKV if layout.qkv == Qkv::Fused => Cut::of_rows(
            heads.fused_rows(d, spec.n_head * d, spec.n_kv_head * d),
            0..cols,
        ),
        name::ATTN_OUTPUT => Cut::cols_only(rows, heads.query_rows(d)),
        name::FFN_GATE | name::FFN_GATE_SHEXP | name::FFN_UP_SHEXP => {
            Cut::rows_only(tp.part(rows), cols)
        }
        name::FFN_UP => match layout.ffn {
            Ffn::Dense { ff, fused: true } => Cut::of_rows(fused_halves(&tp.part(ff), ff), 0..cols),
            _ => Cut::rows_only(tp.part(rows), cols),
        },
        name::FFN_DOWN | name::FFN_DOWN_SHEXP => Cut::cols_only(rows, tp.part(cols)),
        _ => whole,
    }
}

/// The layer and part of a per-layer weight's tensor name
/// (`blk.{layer}.{part}.weight`); `None` for any other name.
#[cfg(test)]
pub(super) fn block_part(tensor_name: &str) -> Option<(usize, &str)> {
    let rest = tensor_name.strip_prefix("blk.")?.strip_suffix(".weight")?;
    let (il, part) = rest.split_once('.')?;
    Some((il.parse().ok()?, part))
}

/// The part of a stored `[rows, cols]` matrix one shard keeps: some row
/// ranges (kept in this order, one after the other) and a column range.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Cut {
    pub(super) rows: Vec<Range<usize>>,
    pub(super) cols: Range<usize>,
}

impl Cut {
    /// The whole `[rows, cols]` matrix.
    pub(super) fn whole(rows: usize, cols: usize) -> Self {
        Self {
            rows: std::iter::once(0..rows).collect(),
            cols: 0..cols,
        }
    }

    /// Row ranges `rows` (empty ranges dropped, a range that continues the
    /// one before merged into it, so a cut of every row is one range) of
    /// the columns `cols`.
    pub(super) fn of_rows(
        rows: impl IntoIterator<Item = Range<usize>>,
        cols: Range<usize>,
    ) -> Self {
        let mut merged: Vec<Range<usize>> = Vec::new();
        for r in rows.into_iter().filter(|r| !r.is_empty()) {
            push_rows(&mut merged, r);
        }
        Self { rows: merged, cols }
    }

    /// Every column of the row range `rows` of a matrix `cols` wide.
    pub(super) fn rows_only(rows: Range<usize>, cols: usize) -> Self {
        Self::of_rows([rows], 0..cols)
    }

    /// Every row of a matrix `rows` tall, the columns `cols`.
    pub(super) fn cols_only(rows: usize, cols: Range<usize>) -> Self {
        Self::of_rows(std::iter::once(0..rows), cols)
    }

    /// Whether the cut is the whole `[rows, cols]` matrix.
    pub(super) fn is_whole(&self, rows: usize, cols: usize) -> bool {
        self.cols == (0..cols) && matches!(self.rows.as_slice(), [r] if *r == (0..rows))
    }

    /// Rows the cut keeps.
    pub(super) fn n_rows(&self) -> usize {
        self.rows.iter().map(ExactSizeIterator::len).sum()
    }
}

/// Whether both bounds of `cols` lie on block boundaries of `dtype`, so the
/// columns are a byte sub-slice of every row of a `row_len`-wide matrix.
fn on_blocks(dtype: GgmlDType, row_len: usize, cols: &Range<usize>) -> bool {
    let block = dtype.block_size();
    row_len.is_multiple_of(block)
        && cols.start.is_multiple_of(block)
        && cols.end.is_multiple_of(block)
}

/// The dtype the columns `cols` of a `row_len`-wide matrix of `dtype` are
/// held in: `dtype` itself when the cut is every column or lies on block
/// boundaries (a byte copy, [`copy_columns`]), else the rows are dequantized
/// and cut, and quantized to Q8_0.
pub(super) fn column_dtype(dtype: GgmlDType, row_len: usize, cols: &Range<usize>) -> GgmlDType {
    if *cols == (0..row_len) || on_blocks(dtype, row_len, cols) {
        dtype
    } else {
        GgmlDType::Q8_0
    }
}

/// The bytes of the columns `cols` of every row of the matrix `raw` (rows
/// `row_len` wide, stored as `dtype`), one row after the other; `None` when
/// a bound does not lie on a block boundary of `dtype` or `raw` is not whole
/// rows.
pub(super) fn copy_columns(
    dtype: GgmlDType,
    raw: &[u8],
    row_len: usize,
    cols: &Range<usize>,
) -> Option<Vec<u8>> {
    if !on_blocks(dtype, row_len, cols) || cols.end > row_len {
        return None;
    }
    let (block, size) = (dtype.block_size(), dtype.type_size());
    let row_bytes = row_len / block * size;
    if row_bytes == 0 {
        return raw.is_empty().then(Vec::new);
    }
    if !raw.len().is_multiple_of(row_bytes) {
        return None;
    }
    let (from, to) = (cols.start / block * size, cols.end / block * size);
    let mut out = Vec::with_capacity(raw.len() / row_bytes * (to - from));
    for row in raw.chunks_exact(row_bytes) {
        out.extend_from_slice(&row[from..to]);
    }
    Some(out)
}

/// The sum of the partial outputs of every shard, on every shard's device;
/// see the module docs for the two ways it is done.
pub(super) struct Collective {
    #[cfg(feature = "provider-local-cuda")]
    nccl: Option<Nccl>,
}

/// The devices of an NCCL group and the context that keeps its
/// communication server alive.
#[cfg(feature = "provider-local-cuda")]
struct Nccl {
    devices: Vec<Device>,
    _context: DistributedContext,
}

#[cfg(feature = "provider-local-cuda")]
impl Nccl {
    fn open(devices: &[Device]) -> Self {
        let config = DistributedConfig {
            all_reduce_op: ReduceOperation::Sum,
        };
        Self {
            devices: devices.to_vec(),
            _context: DistributedContext::init(devices.to_vec(), config),
        }
    }

    /// `all_reduce` the part of every device (in place on each), then wait
    /// for every device's collective.
    fn all_reduce_sum(&self, parts: Vec<Tensor<2>>) -> Vec<Tensor<2>> {
        let pending: Vec<CollectiveTensor<2>> = parts
            .into_iter()
            .map(|part| all_reduce(part, ReduceOperation::Sum, self.devices.clone()))
            .collect();
        pending.into_iter().map(CollectiveTensor::resolve).collect()
    }
}

impl Collective {
    /// The collective of the devices `locals` (opened as `devices`, shard 0
    /// first). NCCL is used only for several distinct CUDA devices on
    /// Linux, the one platform whose NCCL CubeCL links; anything else
    /// reduces through the host.
    pub(super) fn new(locals: &[LocalDevice], devices: &[Device]) -> Self {
        #[cfg(feature = "provider-local-cuda")]
        {
            let nccl = cfg!(target_os = "linux")
                && locals.len() > 1
                && locals
                    .iter()
                    .enumerate()
                    .all(|(i, d)| matches!(d, LocalDevice::Cuda(_)) && !locals[..i].contains(d));
            Self {
                nccl: nccl.then(|| Nccl::open(devices)),
            }
        }
        #[cfg(not(feature = "provider-local-cuda"))]
        {
            let _ = (locals, devices);
            Self {}
        }
    }

    /// Stop the communication server of an NCCL group (a no-op for the
    /// host-staged reduction); the model does this before it returns its
    /// device memory.
    pub(super) fn close(&mut self) {
        #[cfg(feature = "provider-local-cuda")]
        {
            self.nccl = None;
        }
    }

    /// Whether the reduction runs on NCCL rather than through the host.
    #[cfg(test)]
    pub(super) fn uses_nccl(&self) -> bool {
        #[cfg(feature = "provider-local-cuda")]
        {
            self.nccl.is_some()
        }
        #[cfg(not(feature = "provider-local-cuda"))]
        {
            false
        }
    }

    /// The elementwise sum of `parts` (the partial `[t, n_embd]` outputs of
    /// the shards, in shard order, each on its shard's device) on every
    /// shard's device: tensor `s` of the result lives where `parts[s]` did.
    ///
    /// # Errors
    ///
    /// [`CucaError::Provider`] when the parts differ in shape or a device
    /// readback fails.
    pub(super) fn all_reduce_sum(
        &self,
        parts: Vec<Tensor<2>>,
    ) -> Result<Vec<Tensor<2>>, CucaError> {
        #[cfg(feature = "provider-local-cuda")]
        if let Some(nccl) = &self.nccl {
            return Ok(nccl.all_reduce_sum(parts));
        }
        host_all_reduce_sum(parts)
    }
}

/// [`Collective::all_reduce_sum`] through the host: read every part back,
/// add them in f32 in shard order and upload the sum to every device.
fn host_all_reduce_sum(parts: Vec<Tensor<2>>) -> Result<Vec<Tensor<2>>, CucaError> {
    if parts.len() < 2 {
        return Ok(parts);
    }
    let devices: Vec<Device> = parts.iter().map(Tensor::device).collect();
    let shape = parts.first().map(Tensor::dims).unwrap_or_default();
    let mut sum: Vec<f32> = Vec::new();
    for (s, part) in parts.into_iter().enumerate() {
        if part.dims() != shape {
            return Err(super::err(format!(
                "tensor parallelism: shard {s} returned a {:?} partial output, shard 0 a {shape:?} one",
                part.dims()
            )));
        }
        let values = read_f32(part, "a partial layer output")?;
        if s == 0 {
            sum = values;
        } else {
            for (a, b) in sum.iter_mut().zip(&values) {
                *a += b;
            }
        }
    }
    let last = devices.len() - 1;
    let mut reduced = Vec::with_capacity(devices.len());
    for (s, device) in devices.iter().enumerate() {
        let values = if s == last {
            std::mem::take(&mut sum)
        } else {
            sum.clone()
        };
        reduced.push(Tensor::from_data(TensorData::new(values, shape), device));
    }
    Ok(reduced)
}

#[cfg(test)]
mod tests {

    use super::super::read_hparams;
    use super::super::tiny::{TinySpec, write_spec};
    use super::*;

    fn tp(world: usize, shard: usize) -> Tp {
        Tp { world, shard }
    }

    /// The KV heads of a shard divide like the query heads or, with fewer
    /// KV heads than shards, one head is kept whole by every shard that
    /// reads it; whichever way, every query head of the shard finds its
    /// own GQA group's KV head at the index the shard's own head counts
    /// compute.
    #[test]
    fn heads_divide_or_replicate_the_kv_heads() {
        assert_eq!(
            tp(2, 0).heads(4, 2),
            Heads {
                query: 0..2,
                kv: 0..1
            }
        );
        assert_eq!(
            tp(2, 1).heads(4, 2),
            Heads {
                query: 2..4,
                kv: 1..2
            }
        );
        // One KV head, two shards: each shard keeps it.
        for s in 0..2 {
            assert_eq!(tp(2, s).heads(4, 1).kv, 0..1);
        }
        // Two KV heads over four shards: shard `s` serves head `s * 2 / 4`.
        for s in 0..4 {
            let heads = tp(4, s).heads(8, 2);
            assert_eq!(heads.query, 2 * s..2 * s + 2);
            assert_eq!(heads.kv, s / 2..s / 2 + 1);
        }
        for (n_head, n_kv, world) in [
            (4, 2, 2),
            (4, 1, 2),
            (8, 2, 4),
            (8, 1, 4),
            (8, 4, 2),
            (12, 4, 4),
            (6, 3, 3),
            (6, 1, 3),
        ] {
            for shard in 0..world {
                let heads = tp(world, shard).heads(n_head, n_kv);
                let local_kv = (n_kv / world).max(1);
                assert_eq!(heads.kv.len(), local_kv, "{n_head}/{n_kv}/{world}");
                assert_eq!(heads.query.len(), n_head / world);
                for q in heads.query.clone() {
                    // GQA: query head `q` reads KV head `q * n_kv / n_head`.
                    let global = q * n_kv / n_head;
                    assert!(heads.kv.contains(&global), "{n_head}/{n_kv}/{world}: {q}");
                    // The shard's own forward pass maps its local query head to
                    // its local KV head by the same rule.
                    let local = (q - heads.query.start) * local_kv / (n_head / world);
                    assert_eq!(local, global - heads.kv.start, "{n_head}/{n_kv}/{world}");
                }
            }
        }
    }

    #[test]
    fn local_hparams_cut_the_head_counts() {
        let model = write_spec(&TinySpec::llama());
        let content = gguf::Content::open(&model.path).unwrap();
        let hp = read_hparams(&content).unwrap();
        let (n_head, n_kv) = (hp.layers[0].n_head, hp.layers[0].n_kv_head);
        let local = tp(2, 1).local_hparams(&hp);
        assert_eq!(local.layers[0].n_head, n_head / 2);
        assert_eq!(local.layers[0].n_kv_head, n_kv / 2);
        assert_eq!(local.layers[0].head_dim, hp.layers[0].head_dim);
        assert_eq!(local.n_embd, hp.n_embd);
        // One KV head stays one.
        let one = tp(4, 0).local_hparams(&hp);
        assert_eq!(one.layers[0].n_kv_head, 1);
        // A single shard changes nothing.
        let solo = Tp::SOLO.local_hparams(&hp);
        assert_eq!(solo.layers[0].n_head, n_head);
        assert_eq!(solo.layers[0].n_kv_head, n_kv);
    }

    /// The rows a shard keeps of a fused `[q | k | v]` or `[gate | up]`
    /// projection, a gated query packing and an `attn_output` column cut,
    /// with the numbers worked out by hand.
    #[test]
    fn fused_and_gated_rows_land_on_the_shards_heads() {
        // Fused `phi3` layout: 4 query and 4 KV heads of 32, `[q | k | v]` has 384
        // rows (q 0..128, k 128..256, v 256..384).
        let heads = tp(2, 1).heads(4, 4);
        assert_eq!(
            heads.fused_rows(32, 128, 128),
            vec![64..128, 192..256, 320..384]
        );
        let first = tp(2, 0).heads(4, 4);
        assert_eq!(
            Cut::of_rows(first.fused_rows(32, 128, 128), 0..64).rows,
            vec![0..64, 128..192, 256..320]
        );
        // One KV head: both shards read the same k and v rows.
        for s in 0..2 {
            let heads = tp(2, s).heads(4, 1);
            let rows = heads.fused_rows(16, 64, 16);
            assert_eq!(rows, vec![32 * s..32 * (s + 1), 64..80, 80..96]);
        }
        // Everything kept is one range, whatever the parts.
        let whole = Tp::SOLO.heads(4, 4);
        assert_eq!(
            Cut::of_rows(whole.fused_rows(32, 128, 128), 0..64),
            Cut::whole(384, 64)
        );
        // A gated attention packs `2 * d` rows per head.
        assert_eq!(tp(2, 1).heads(4, 2).query_rows(64), 128..256);
        // Fused `[gate | up]` of 128-wide halves.
        assert_eq!(
            fused_halves(&tp(2, 1).part(128), 128),
            vec![64..128, 192..256]
        );
        assert_eq!(
            Cut::of_rows(fused_halves(&Tp::SOLO.part(128), 128), 0..64),
            Cut::whole(256, 64)
        );
        // The output projection keeps its heads' columns of every row.
        assert_eq!(
            Cut::cols_only(64, tp(2, 1).heads(4, 4).query_rows(32)),
            Cut {
                rows: std::iter::once(0..64).collect(),
                cols: 64..128
            }
        );
        // `of_rows` drops empty ranges and merges adjacent ones.
        assert_eq!(
            Cut::of_rows([0..4, 4..4, 4..8, 10..12], 0..2).rows,
            vec![0..8, 10..12]
        );
    }

    #[test]
    fn block_part_reads_the_layer_and_part() {
        assert_eq!(block_part("blk.3.ffn_up.weight"), Some((3, "ffn_up")));
        assert_eq!(
            block_part("blk.12.ffn_gate_shexp.weight"),
            Some((12, "ffn_gate_shexp"))
        );
        assert_eq!(block_part("blk.3.attn_q.bias"), None);
        assert_eq!(block_part("token_embd.weight"), None);
        assert_eq!(block_part("blk.x.attn_q.weight"), None);
    }

    /// For every sliced 2-D weight of several tiny models the cuts of all
    /// shards tile the matrix exactly once (a replicated KV head: once per
    /// shard that serves it), and a replicated weight (the router, ..) is
    /// kept whole. The tiny models cover separate and fused QKV, fused
    /// gate-up, one KV head, biases and experts' shared tensors.
    #[test]
    fn shard_cuts_tile_every_weight() {
        let specs = [
            TinySpec {
                ff: 128,
                ..TinySpec::llama()
            },
            TinySpec {
                ff: 128,
                ..TinySpec::phi3()
            },
            TinySpec {
                ff: 128,
                ..TinySpec::gemma3()
            },
            TinySpec {
                ff: 128,
                ..TinySpec::qwen2()
            },
            TinySpec::qwen3moe(),
        ];
        for spec in specs {
            let model = write_spec(&spec);
            let content = gguf::Content::open(&model.path).unwrap();
            let hp = read_hparams(&content).unwrap();
            for world in [2usize, 4] {
                if validate(&content, &hp, world, &ExpertPlacement::Resident).is_err() {
                    continue;
                }
                let mut seen = 0;
                for (tensor, info) in &content.tensor_infos {
                    let Some((il, part)) = block_part(tensor) else {
                        continue;
                    };
                    let [rows, cols] = *info.shape.dims() else {
                        continue;
                    };
                    let layout = LayerLayout::read(&hp, il, &hp.layers[il], &content).unwrap();
                    let n_kv = hp.layers[il].n_kv_head;
                    let want = if matches!(part, name::ATTN_K | name::ATTN_V) {
                        (world / n_kv).max(1)
                    } else {
                        1
                    };
                    let mut covered = vec![0usize; rows * cols];
                    for shard in 0..world {
                        let cut = part_cut(&hp, &layout, il, part, (rows, cols), tp(world, shard));
                        for r in &cut.rows {
                            for row in r.clone() {
                                for col in cut.cols.clone() {
                                    covered[row * cols + col] += 1;
                                }
                            }
                        }
                    }
                    let sliced = covered.iter().all(|c| *c == want);
                    let whole = covered.iter().all(|c| *c == world);
                    assert!(
                        sliced || whole,
                        "{} world {world} {tensor}: coverage {:?}",
                        spec.arch,
                        covered.iter().copied().max()
                    );
                    // Only the replicated parts are kept whole by every shard.
                    let replicated = !matches!(
                        part,
                        name::ATTN_Q
                            | name::ATTN_GATE
                            | name::ATTN_K
                            | name::ATTN_V
                            | name::ATTN_QKV
                            | name::ATTN_OUTPUT
                            | name::FFN_GATE
                            | name::FFN_UP
                            | name::FFN_DOWN
                            | name::FFN_GATE_SHEXP
                            | name::FFN_UP_SHEXP
                            | name::FFN_DOWN_SHEXP
                    );
                    assert_eq!(whole && !sliced, replicated, "{tensor}");
                    seen += 1;
                }
                assert!(seen > 0, "{}: no per-layer weights", spec.arch);
            }
        }
    }

    /// Each rule of [`validate`] rejects a model that breaks it.
    #[test]
    fn validate_names_the_rule_and_the_numbers() {
        let rejected = |spec: TinySpec, world: usize, rule: &str| {
            let model = write_spec(&spec);
            let content = gguf::Content::open(&model.path).unwrap();
            let hp = read_hparams(&content).unwrap();
            let e = validate(&content, &hp, world, &ExpertPlacement::Resident).unwrap_err();
            let CucaError::Config(message) = e else {
                panic!("{rule}: not a Config error");
            };
            assert!(
                message.starts_with("tensor parallelism: ") && message.contains(rule),
                "{message}"
            );
            message
        };
        let llama = TinySpec {
            ff: 128,
            ..TinySpec::llama()
        };
        let message = rejected(llama.clone(), 3, "attention heads do not divide");
        assert!(message.contains("layer 0: 4 heads, 3 shards"), "{message}");
        rejected(
            TinySpec {
                n_head: 6,
                n_kv_head: 3,
                ..llama.clone()
            },
            2,
            "key/value heads",
        );
        rejected(
            TinySpec {
                n_head: 2,
                n_kv_head: 2,
                ..llama.clone()
            },
            2,
            "query width",
        );
        rejected(TinySpec::llama(), 2, "dense FFN width");
        rejected(TinySpec::olmo2(), 2, "full-width Q/K norm");
        {
            let model = super::super::tiny::write_deepseek41(
                &super::super::tiny::TinyDeepseek41::default(),
            );
            let content = gguf::Content::open(&model.path).unwrap();
            let hp = read_hparams(&content).unwrap();
            let e = validate(&content, &hp, 2, &ExpertPlacement::Resident).unwrap_err();
            let message = e.to_string();
            assert!(
                message.contains("tensor parallelism: hyper-connection residual streams")
                    && message.contains("deepseek41"),
                "{message}"
            );
        }
        // One shard and a model that splits pass.
        for (spec, world) in [
            (llama, 2),
            (TinySpec::qwen3moe(), 2),
            (TinySpec::llama(), 1),
        ] {
            let model = write_spec(&spec);
            let content = gguf::Content::open(&model.path).unwrap();
            let hp = read_hparams(&content).unwrap();
            validate(&content, &hp, world, &ExpertPlacement::Resident).unwrap();
        }
    }
}
