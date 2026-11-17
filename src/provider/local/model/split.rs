//! Layer offload: some transformer layers on a GPU and the rest on the CPU
//! (llama.cpp's `-ngl`), for a model that does not fit the device.
//!
//! `LocalModelConfig::gpu_layers = Some(n)` puts `n` layers on the GPU:
//!
//! - Which layers: the last `n`, as llama.cpp does. The CPU runs the layers
//!   `0..n_layer - n` and the GPU the layers `n_layer - n..n_layer`. A
//!   boundary costs one residual stream per pass wherever it is, and the tail
//!   is where the head is, so the logits come off the device that already
//!   holds the last layer.
//! - The embeddings are always on the host: both engines look a token's row up
//!   in the file and hand the row on (the GPU engine uploads it), so the first
//!   layers' input needs no device copy of the table. The Gemma 4
//!   per-layer-embedding inputs are computed on the host by the CPU half for
//!   every layer, and the GPU half receives them with the hidden state.
//! - The head (final norm, LM head, logit scale and soft-cap) is on the GPU
//!   with the last layer. With `n = 0` nothing is on the GPU (the model loads
//!   as a plain CPU model and no device opens); with `n = n_layer` the model is
//!   a plain GPU model. llama.cpp counts the output layer as one more
//!   offloadable layer and keeps it on the CPU at `n = n_layer`; here the head
//!   always follows the last layer.
//!
//! # How a pass runs
//!
//! [`SplitModel::forward_batch`] runs the CPU half ([`Transformer`] over the
//! layers `0..first`, a [`Part`]: embeddings, per-layer inputs and the first
//! layers, no LM head) over the rows of every sequence of the batch, then
//! hands the residual stream `[rows, n_embd]` (and the per-layer inputs) to
//! the GPU half ([`GpuTransformer`] over `first..n_layer`: the last layers and
//! the head) in a [`Carry`]. The hidden state crosses the boundary once per
//! pass: one host read of the CPU half's output and one upload; nothing else
//! moves between the halves.
//!
//! The paged K/V cache is one [`KvLayout`](super::paged::KvLayout): both
//! halves build the same groups and block counts from the same
//! hyperparameters, the engine allocates one set of blocks per sequence and
//! passes the same block tables to both, and each layer's pool lives on its
//! layer's device (the CPU half allocates pools for layers `0..first` only,
//! the GPU half for the layers from `first`). The scheduler sees one
//! [`Model`](super::Model): its interface is unchanged. A hybrid model keeps
//! each delta-net layer's recurrent state on its layer's device; a snapshot
//! is the pair of the halves' snapshots.
//!
//! A layer that reads another layer's K/V (Gemma 4's shared-K/V layers)
//! can sit on the other side of the boundary from its source: the source's
//! pool then exists on both devices. The CPU half, which computes the source,
//! returns the rows it cached for it in every pass ([`MirrorRows`], read back
//! from its own pool through the block tables, `[n_kv_head, rows, head_dim]`
//! keys and values per sequence), and the GPU half writes them into a mirror
//! pool of its own under the same tables ([`mirrors`] lists the layers). They
//! ride the same host-to-device crossing as the hidden state. The mirror
//! pools are not counted in `kv_cache_bytes`, which counts each layer once.
//!
//! A K/V transfer ([`transfer`](super::transfer)) exports each layer's rows
//! from the half that owns the layer, merged in layout order (the mirror
//! pools copy the CPU half's, so they are not exported), and imports a
//! mirrored layer's rows into both the CPU pool and the GPU half's mirror
//! pool. A recurrent-state transfer lists the CPU half's delta-net layers,
//! then the GPU half's.
//!
//! # Stages
//!
//! The two halves are the two-stage case of a model run as a chain of stages,
//! each an engine over a [`LayerRange`] with the [`Part`] that says which K/V
//! rows cross its edges: `forward_stage(batch, input)` takes the [`Carry`] of
//! the stage before (none for the stage that embeds) and returns the [`Carry`]
//! for the stage after, or, from the stage that ends at the last layer, the
//! logits ([`StageOut`]). A middle stage forwards the K/V rows it received
//! for layers a later stage reads ([`passes`]). The stages need not be on one
//! machine: [`chain`](super::chain) runs the first ones on other machines.
//! [`legal_cuts`] lists where a model can be cut.
//!
//! # Experts
//!
//! With `ExpertPlacement::Tiered` the CPU half's MoE layers use the RAM
//! tier (they are computed on the CPU, with `ram` bytes of experts resident)
//! and the GPU half's the device tier, backed by a RAM tier of its own and
//! the file. The `ram` budget is shared out between the halves in
//! proportion to their MoE layer counts, `device` is the GPU half's alone.
//! The two stores share one set of counters and one heat table, so
//! `LocalModel::expert_tier_stats` and `expert_heat` describe the whole
//! model. Resident experts live where their layer runs.
//!
//! # Not supported
//!
//! Tensor parallelism (`shards`) and a drafter are rejected by
//! `LocalModel::load` with a [`CucaError::Config`].

use std::sync::Arc;

use super::cpu::Forward;
use super::engram::RowSource;
use super::params::Hparams;
use super::tier::TierShared;
use crate::error::CucaError;

/// Where the layers of an offloaded model run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Placement {
    /// `gpu_layers == 0`: every layer on the CPU, no device opens.
    Cpu,
    /// `gpu_layers == n_layer`: every layer on the GPU.
    Gpu,
    /// The layers `first..n_layer` on the GPU, `0..first` on the CPU.
    Split { first: usize },
}

/// Where `gpu_layers` layers of the model `hp` go.
///
/// # Errors
///
/// [`CucaError::Config`] for more layers than the model has.
pub(crate) fn plan(hp: &Hparams, gpu_layers: usize) -> Result<Placement, CucaError> {
    let n_layer = hp.n_layer;
    if gpu_layers > n_layer {
        return Err(CucaError::Config(format!(
            "gpu_layers {gpu_layers} exceeds the {n_layer} layers of `{}`",
            hp.arch.name()
        )));
    }
    if gpu_layers > 0 && gpu_layers < n_layer && !legal_cut(hp, n_layer - gpu_layers) {
        let legal = legal_gpu_layers(hp);
        return Err(CucaError::Config(format!(
            "gpu_layers {gpu_layers} cuts `{}` between layers that share compressed rows or indexer picks; legal values: {}",
            hp.arch.name(),
            legal
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        )));
    }
    Ok(if gpu_layers == 0 {
        Placement::Cpu
    } else if gpu_layers == n_layer {
        Placement::Gpu
    } else {
        Placement::Split {
            first: n_layer - gpu_layers,
        }
    })
}

/// Whether the layers `first..` of `hp` can run apart from the layers
/// before them. A layer with compressed sparse attention reads the compressed
/// rows of its cache source layer and the picks of its index source layer
/// (and, when it narrows its scoring to candidate blocks, those the candidate
/// source layer published), none of which crosses the boundary: every source
/// a layer from `first` on reads must itself be from `first` on. Every other
/// model can be cut anywhere.
pub(crate) fn legal_cut(hp: &Hparams, first: usize) -> bool {
    let candidate_source = hp
        .csa
        .as_ref()
        .and_then(|g| g.indexer.candidates)
        .map(|(src, _, _)| src);
    hp.layers.iter().skip(first).all(|l| {
        l.csa.is_none_or(|c| {
            c.kv_source >= first
                && c.index_source >= first
                && (c.candidates != super::csa::Candidates::Consumer
                    || candidate_source.is_none_or(|src| src >= first))
        })
    })
}

/// The `gpu_layers` values [`plan`] accepts for `hp`, ascending: 0, the
/// layer count and every cut [`legal_cut`] allows.
pub(crate) fn legal_gpu_layers(hp: &Hparams) -> Vec<usize> {
    let n = hp.n_layer;
    let mut legal: Vec<usize> = std::iter::once(0)
        .chain(
            (1..n)
                .filter(|&first| legal_cut(hp, first))
                .rev()
                .map(|first| n - first),
        )
        .chain(std::iter::once(n))
        .collect();
    legal.sort_unstable();
    legal.dedup();
    legal
}

/// Every `first` that [`legal_cut`] allows, ascending: where a stage of a
/// model can begin after another. Layer `0` and `n_layer` are no cuts.
pub(crate) fn legal_cuts(hp: &Hparams) -> Vec<usize> {
    (1..hp.n_layer)
        .filter(|&first| legal_cut(hp, first))
        .collect()
}

/// The layers `first..end` of a model: one engine's share of it. `first == 0`
/// embeds the tokens, `end == n_layer` holds the final norm and the head.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LayerRange {
    pub(crate) first: usize,
    pub(crate) end: usize,
}

impl LayerRange {
    /// Every layer of `hp`.
    pub(crate) fn whole(hp: &Hparams) -> Self {
        Self {
            first: 0,
            end: hp.n_layer,
        }
    }

    /// Whether layer `il` is in the range.
    pub(crate) fn contains(self, il: usize) -> bool {
        (self.first..self.end).contains(&il)
    }

    /// Layers in the range.
    pub(crate) fn len(self) -> usize {
        self.end - self.first
    }
}

/// What an engine that runs only some layers of a model needs besides the
/// range: how it shares its counters with the other engines of the model and
/// which K/V rows cross its edges.
pub(crate) struct Part {
    pub(crate) range: LayerRange,
    /// The expert-tier counters and heat shared with the other engines'
    /// expert stores, so that one set of statistics covers the whole model.
    pub(crate) shared: Option<Arc<TierShared>>,
    /// The K/V-owning layers in the range that a layer after it reads the
    /// K/V of ([`exports`]): the pass returns their new rows.
    pub(crate) export: Vec<usize>,
    /// The K/V-owning layers before the range that a layer in it reads
    /// ([`mirrors`]): the engine keeps a pool for each, filled from the rows
    /// the previous stage returned.
    pub(crate) mirror: Vec<usize>,
    /// The Engram table the other engines share (`None`: this engine opens
    /// its own, or the model has none), so that its row cache and counters
    /// are single.
    pub(crate) engram: Option<Arc<dyn RowSource>>,
}

/// What a stage hands to the stage after it: the residual stream and, for
/// Gemma 4, the per-layer inputs every layer reads.
pub(crate) struct Carry {
    /// `[rows, n_embd]` row-major (`[rows, streams * n_embd]`, the
    /// hyper-connection streams, for a model that has them).
    pub(crate) hidden: Vec<f32>,
    /// `[rows, n_layer, ple_dim]` row-major, `(ctx + tok) / sqrt(2)`: the
    /// first stage computes it and every later stage passes it on.
    pub(crate) per_layer: Option<Vec<f32>>,
    /// `[rows, streams]`: the pre weights the layers after the stage start
    /// from (hyper-connection models).
    pub(crate) pre: Option<Vec<f32>>,
    /// The rows the pass wrote into the cache of every layer of
    /// [`Part::export`], and the rows of [`passes`] the stage received.
    pub(crate) kv: Vec<MirrorRows>,
}

/// What one forward pass of a stage returns.
pub(crate) enum StageOut {
    /// The stage is not the last: what the next one starts from.
    Carry(Carry),
    /// The stage holds the head: the logits and hidden rows of every sequence.
    Done(Vec<Forward>),
}

/// The K/V-owning layers in `range` that a layer at or after `range.end`
/// reads the K/V of (the shared-K/V layers of Gemma 4 on a later stage),
/// ascending.
pub(crate) fn exports(hp: &Hparams, range: LayerRange) -> Vec<usize> {
    let mut sources: Vec<usize> = hp
        .layers
        .iter()
        .enumerate()
        .filter(|(il, spec)| *il >= range.end && range.contains(spec.kv_source))
        .map(|(_, spec)| spec.kv_source)
        .collect();
    sources.sort_unstable();
    sources.dedup();
    sources
}

/// The K/V-owning layers before `range.first` whose cache a layer in `range`
/// reads, ascending.
pub(crate) fn mirrors(hp: &Hparams, range: LayerRange) -> Vec<usize> {
    let mut sources: Vec<usize> = hp
        .layers
        .iter()
        .enumerate()
        .filter(|(il, spec)| range.contains(*il) && spec.kv_source < range.first)
        .map(|(_, spec)| spec.kv_source)
        .collect();
    sources.sort_unstable();
    sources.dedup();
    sources
}

/// The K/V-owning layers before `range.first` whose cache a layer at or after
/// `range.end` reads, ascending: a stage in the middle of the model forwards
/// the rows it received for them unchanged.
pub(crate) fn passes(hp: &Hparams, range: LayerRange) -> Vec<usize> {
    let mut sources: Vec<usize> = hp
        .layers
        .iter()
        .enumerate()
        .filter(|(il, spec)| *il >= range.end && spec.kv_source < range.first)
        .map(|(_, spec)| spec.kv_source)
        .collect();
    sources.sort_unstable();
    sources.dedup();
    sources
}

/// The K/V-owning layers before `first` whose cache a layer from `first` on
/// reads (the shared-K/V layers of Gemma 4 that sit on the other side of the
/// boundary from their source), ascending.
#[cfg(test)]
pub(crate) fn mirrored(hp: &Hparams, first: usize) -> Vec<usize> {
    mirrors(
        hp,
        LayerRange {
            first,
            end: hp.n_layer,
        },
    )
}

/// The rows a layer cached in one pass for the layers of other stages that
/// read its K/V ([`exports`]).
pub(crate) struct MirrorRows {
    /// The layer that owns the cache.
    pub(crate) layer: usize,
    /// Per sequence of the batch, in order: the new keys and values,
    /// `[n_kv_head, rows, head_dim]` row-major.
    pub(crate) seqs: Vec<(Vec<f32>, Vec<f32>)>,
}

#[cfg(feature = "provider-local-gpu")]
pub(crate) use model::{SplitModel, SplitSnapshot};

#[cfg(feature = "provider-local-gpu")]
mod model {

    use std::sync::Arc;

    use crate::provider::local::gguf;

    use super::super::cpu::{Forward, Transformer};
    use super::super::delta::DeltaState;
    use super::super::gpu::{GpuSnapshot, GpuTransformer};
    use super::super::paged::{BlockTable, KvLayout, Sizing};
    use super::super::tier::TierShared;
    use super::super::transfer::{DeltaRows, PoolRows, entries};
    use super::super::{Batch, err, read_hparams};
    use super::{Hparams, LayerRange, Part, StageOut, exports, mirrors};
    use crate::error::CucaError;
    use crate::provider::local::{EngramCache, ExpertPlacement, ExpertTiers, LocalDevice};

    /// A model whose first layers run on the CPU and whose last layers (and
    /// head) run on a GPU; see the module docs.
    pub(crate) struct SplitModel {
        front: Transformer,
        back: GpuTransformer,
        /// The first layer on the GPU.
        first: usize,
        n_layer: usize,
        tiers: Option<Arc<TierShared>>,
        /// The model keeps a compressor state per sequence slot in each half.
        stateful: bool,
    }

    /// A recurrent-state snapshot of both halves.
    pub(crate) struct SplitSnapshot {
        front: Vec<DeltaState>,
        back: GpuSnapshot,
    }

    impl SplitSnapshot {
        /// Bytes the snapshot holds on the host and the device.
        pub(crate) fn bytes(&self) -> usize {
            self.front.iter().map(DeltaState::bytes).sum::<usize>() + self.back.bytes()
        }
    }

    /// How `experts` is shared between the halves of the model `hp` split at
    /// `first`: the placement of each half and the counters they share.
    ///
    /// # Errors
    ///
    /// [`CucaError::Config`] for tiers on a model without experts, or without
    /// a RAM tier while MoE layers stay on the CPU.
    fn place_experts(
        hp: &Hparams,
        experts: &ExpertPlacement,
        first: usize,
    ) -> Result<(ExpertPlacement, ExpertPlacement, Option<Arc<TierShared>>), CucaError> {
        // Host experts belong to the GPU half's MoE layers; the CPU half
        // already keeps its experts in RAM.
        if let ExpertPlacement::Host { .. } = experts {
            return Ok((ExpertPlacement::Resident, experts.clone(), None));
        }
        let ExpertPlacement::Tiered(tiers) = experts else {
            return Ok((ExpertPlacement::Resident, ExpertPlacement::Resident, None));
        };
        let spec = hp.moe.ok_or_else(|| {
            CucaError::Config(format!(
                "expert tiers need a mixture-of-experts model; `{}` has no experts",
                hp.arch.name()
            ))
        })?;
        let moe = |il: usize| il >= hp.leading_dense;
        let on_cpu = (0..first).filter(|&il| moe(il)).count() as u64;
        let on_gpu = (first..hp.n_layer).filter(|&il| moe(il)).count() as u64;
        if on_cpu > 0 && tiers.ram_bytes == 0 {
            return Err(CucaError::Config(
                "experts.ram_bytes must be greater than 0 with gpu_layers: the layers kept on the CPU read their experts through the RAM tier".into(),
            ));
        }
        // The RAM budget follows the layers that read it.
        let total = (on_cpu + on_gpu).max(1);
        let ram_cpu = if on_gpu == 0 {
            tiers.ram_bytes
        } else {
            (u128::from(tiers.ram_bytes) * u128::from(on_cpu) / u128::from(total)) as u64
        };
        let half = |ram_bytes: u64, device_bytes: u64, layers: u64| {
            if layers == 0 {
                ExpertPlacement::Resident
            } else {
                ExpertPlacement::Tiered(ExpertTiers {
                    ram_bytes,
                    device_bytes,
                    loader_threads: tiers.loader_threads,
                    warm_from: tiers.warm_from.clone(),
                })
            }
        };
        let cpu = half(ram_cpu, 0, on_cpu);
        let gpu = half(tiers.ram_bytes - ram_cpu, tiers.device_bytes, on_gpu);
        Ok((
            cpu,
            gpu,
            Some(Arc::new(TierShared::new(hp.n_layer, spec.n_expert))),
        ))
    }

    impl SplitModel {
        /// Load the first `first` layers on the CPU and the rest on `device`.
        ///
        /// # Errors
        ///
        /// As [`Transformer::load_range`] and [`GpuTransformer::load_range`].
        pub(crate) fn load(
            content: &gguf::Content,
            context_length: usize,
            device: LocalDevice,
            experts: &ExpertPlacement,
            sizing: &Sizing,
            first: usize,
            engram: &EngramCache,
        ) -> Result<Self, CucaError> {
            Self::load_with(
                content,
                context_length,
                device,
                experts,
                sizing,
                first,
                false,
                engram,
            )
        }

        /// [`SplitModel::load`], with the CPU half's matmul weights held as
        /// f32 when `dequantize` (the reference of the tests).
        #[expect(
            clippy::too_many_arguments,
            reason = "the load's independent inputs; a struct would only rename them"
        )]
        pub(crate) fn load_with(
            content: &gguf::Content,
            context_length: usize,
            device: LocalDevice,
            experts: &ExpertPlacement,
            sizing: &Sizing,
            first: usize,
            dequantize: bool,
            engram: &EngramCache,
        ) -> Result<Self, CucaError> {
            let hp = read_hparams(content)?;
            let (cpu_experts, gpu_experts, shared) = place_experts(&hp, experts, first)?;
            let (head, tail) = (
                LayerRange {
                    first: 0,
                    end: first,
                },
                LayerRange {
                    first,
                    end: hp.n_layer,
                },
            );
            // One table for both halves: one row cache, one set of counters.
            let engram_rows = super::super::engram::open_table(content, &hp, engram)?;
            let front = Transformer::load_range(
                content,
                context_length,
                &cpu_experts,
                sizing,
                &Part {
                    range: head,
                    shared: shared.clone(),
                    export: exports(&hp, head),
                    mirror: Vec::new(),
                    engram: engram_rows.clone(),
                },
                dequantize,
            )?;
            let back = GpuTransformer::load_range(
                content,
                context_length,
                device,
                &gpu_experts,
                sizing,
                &Part {
                    range: tail,
                    shared: shared.clone(),
                    export: Vec::new(),
                    mirror: mirrors(&hp, tail),
                    engram: engram_rows,
                },
            )?;
            Ok(Self {
                front,
                back,
                first,
                n_layer: hp.n_layer,
                tiers: shared,
                stateful: hp.behavior.hyper_connections,
            })
        }

        /// Layers on the GPU.
        pub(crate) fn gpu_layers(&self) -> usize {
            self.n_layer - self.first
        }

        /// Layers in all.
        pub(crate) fn n_layer(&self) -> usize {
            self.n_layer
        }

        /// Counters and heat of the expert tiers of both halves.
        pub(crate) fn expert_tiers(&self) -> Option<Arc<TierShared>> {
            self.tiers.clone()
        }

        /// [`Model::forward_batch`](super::super::Model::forward_batch): the
        /// CPU half's layers, then the hidden state to the GPU half's.
        ///
        /// # Errors
        ///
        /// As the halves' passes; a failed GPU half empties the CPU half's
        /// recurrent state of every sequence of the step, as the GPU half
        /// does its own.
        pub(crate) fn forward_batch(
            &mut self,
            batch: &Batch<'_>,
        ) -> Result<Vec<Forward>, CucaError> {
            let StageOut::Carry(carry) = self.front.forward_stage(batch, None)? else {
                return Err(err("the CPU half returned logits"));
            };
            let result = match self.back.forward_stage(batch, Some(carry)) {
                Ok(StageOut::Done(outs)) => Ok(outs),
                Ok(StageOut::Carry(_)) => Err(err("the GPU half returned no logits")),
                Err(e) => Err(e),
            };
            if result.is_err() {
                for row in &batch.rows {
                    self.front.reset_slot(row.state.0);
                }
            }
            result
        }

        /// The architecture.
        pub(crate) fn arch(&self) -> super::super::arch::Arch {
            self.back.arch()
        }

        /// The block pools' layout (shared by both halves).
        pub(crate) fn kv_layout(&self) -> &KvLayout {
            self.back.kv_layout()
        }

        /// Start reading the engram rows of the next `tokens` of a sequence.
        /// Both halves share one table and each asks for the rows of the
        /// Engram layers in its own range, so nothing is read twice.
        pub(crate) fn engram_prefetch(&self, history: &[u32], tokens: &[u32]) {
            self.front.engram_prefetch(history, tokens);
            self.back.engram_prefetch(history, tokens);
        }

        /// The counters of the engram table both halves read.
        pub(crate) fn engram_counters(
            &self,
        ) -> Option<std::sync::Arc<super::super::engram_io::EngramCounters>> {
            self.front.engram_counters()
        }

        /// Bytes of K/V data the pools hold, on both devices.
        pub(crate) fn kv_pool_bytes(&self) -> u64 {
            self.front.kv_pool_bytes()
        }

        /// Whether the model has recurrent layers.
        pub(crate) fn is_hybrid(&self) -> bool {
            self.back.is_hybrid()
        }

        /// Most rows of one `Want::All` call after which every intermediate
        /// length can still be returned to: what both halves can do.
        pub(crate) fn rollback_rows(&self) -> usize {
            self.front.rollback_rows().min(self.back.rollback_rows())
        }

        /// Take the recurrent state of `slot` of both halves back to at most
        /// `len` positions; returns the position both sit at. The halves
        /// retain different positions (the CPU one the rows of the last
        /// verification call, the GPU one only the end), so the request
        /// settles on the position both can hold.
        pub(crate) fn rewind(&mut self, slot: usize, len: usize) -> usize {
            if !self.is_hybrid() && !self.stateful {
                return len;
            }
            let mut at = len;
            // Each half only ever moves to a smaller position, and 0 is
            // always possible: the loop ends at the first position both hold.
            loop {
                let back = self.back.rewind(slot, at);
                let front = self.front.rewind(slot, back);
                if front == back {
                    return back;
                }
                at = front;
            }
        }

        /// A copy of the recurrent state of `slot` of both halves.
        pub(crate) fn snapshot(&self, slot: usize) -> Option<SplitSnapshot> {
            Some(SplitSnapshot {
                front: self.front.snapshot(slot)?,
                back: self.back.snapshot(slot)?,
            })
        }

        /// Make `snapshot` the recurrent state of `slot`, at position `len`.
        ///
        /// # Errors
        ///
        /// As the halves' `restore`.
        pub(crate) fn restore(
            &mut self,
            slot: usize,
            snapshot: &SplitSnapshot,
            len: usize,
        ) -> Result<(), CucaError> {
            self.front.restore(slot, &snapshot.front, len)?;
            self.back.restore(slot, &snapshot.back, len)
        }

        /// The zero state at position 0 for `slot`.
        pub(crate) fn reset_slot(&mut self, slot: usize) {
            self.front.reset_slot(slot);
            self.back.reset_slot(slot);
        }

        /// See [`Model::export_rows`](super::super::Model::export_rows): the
        /// CPU half's entries (the layers before the cut) and the GPU half's
        /// (the layers from it on), merged in layout order. The GPU half does
        /// not export its mirror pools, which copy the CPU half's.
        ///
        /// # Errors
        ///
        /// As the halves' `export_rows`.
        pub(crate) fn export_rows(
            &self,
            tables: &[BlockTable],
            from: usize,
            to: usize,
            end: usize,
        ) -> Result<Vec<PoolRows>, CucaError> {
            let layout = self.kv_layout();
            let mut front = self
                .front
                .export_rows(tables, from, to, end)?
                .into_iter()
                .peekable();
            let mut back = self
                .back
                .export_rows(tables, from, to, end)?
                .into_iter()
                .peekable();
            let mut out = Vec::with_capacity(front.len() + back.len());
            for (gi, il) in entries(layout) {
                let compressed = layout.groups[gi].compressed;
                let half = if il < self.first {
                    &mut front
                } else {
                    &mut back
                };
                if let Some(rows) = half.next_if(|r| r.layer == il && r.compressed == compressed) {
                    out.push(rows);
                }
            }
            Ok(out)
        }

        /// See [`Model::import_rows`](super::super::Model::import_rows): the
        /// CPU half writes the entries of its layers, the GPU half those of
        /// its own and the mirror pools of the CPU layers its layers read
        /// (a mirrored layer's entry reaches both), and the indexes written
        /// by either are returned. A rejected transfer may leave rows in
        /// blocks the caller has not named yet.
        ///
        /// # Errors
        ///
        /// As the halves' `import_rows`.
        pub(crate) fn import_rows(
            &mut self,
            tables: &[BlockTable],
            from: usize,
            to: usize,
            end: usize,
            rows: &[PoolRows],
        ) -> Result<Vec<usize>, CucaError> {
            let mut written = self.front.import_rows(tables, from, to, end, rows)?;
            written.extend(self.back.import_rows(tables, from, to, end, rows)?);
            Ok(written)
        }

        /// The state of every delta-net layer of `snapshot`, ascending: the
        /// CPU half's layers, then the GPU half's.
        ///
        /// # Errors
        ///
        /// As the halves' `snapshot_export`.
        pub(crate) fn snapshot_export(
            &self,
            snapshot: &SplitSnapshot,
        ) -> Result<Vec<DeltaRows>, CucaError> {
            let mut rows = self.front.snapshot_export(&snapshot.front)?;
            rows.extend(self.back.snapshot_export(&snapshot.back)?);
            Ok(rows)
        }

        /// A snapshot of both halves holding `rows`, one per delta-net layer,
        /// ascending.
        ///
        /// # Errors
        ///
        /// [`CucaError::Provider`] when `rows` do not fit the recurrent layers
        /// of the halves, or as the halves' `snapshot_import`.
        pub(crate) fn snapshot_import(
            &self,
            mut rows: Vec<DeltaRows>,
        ) -> Result<SplitSnapshot, CucaError> {
            let on_cpu = self.back.delta_layers_before_first();
            if rows.len() < on_cpu {
                return Err(err(
                    "the snapshot does not fit this model's recurrent state",
                ));
            }
            let on_gpu = rows.split_off(on_cpu);
            Ok(SplitSnapshot {
                front: self.front.snapshot_import(rows)?,
                back: self.back.snapshot_import(on_gpu)?,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::cpu::{Hidden, Transformer};
    use super::super::tiny::{TinyOptions, TinySpec, VOCAB, write_spec, write_tiny_model_with};
    use super::super::{Batch, SeqRows, Want};
    use super::*;
    use crate::provider::local::gguf;
    use crate::provider::local::{CarryDtype, stageframe};

    fn hparams(model: &super::super::tiny::TinyModel) -> Hparams {
        let content = gguf::Content::open(&model.path).unwrap();
        super::super::read_hparams(&content).unwrap()
    }

    /// Every `gpu_layers` from 0 to the layer count places the model; one
    /// more fails the load naming the numbers.
    #[test]
    fn gpu_layers_is_validated_against_the_layer_count() {
        let hp = hparams(&write_spec(&TinySpec::llama()));
        let n = hp.n_layer;
        assert!(n >= 2, "the tiny llama has several layers");
        assert_eq!(plan(&hp, 0).unwrap(), Placement::Cpu);
        assert_eq!(plan(&hp, n).unwrap(), Placement::Gpu);
        for k in 1..n {
            assert_eq!(plan(&hp, k).unwrap(), Placement::Split { first: n - k });
        }
        for over in [n + 1, usize::MAX] {
            let e = plan(&hp, over).unwrap_err();
            assert!(
                matches!(&e, CucaError::Config(m) if m.contains(&format!("gpu_layers {over} exceeds the {n} layers of `llama`"))),
                "{e:?}"
            );
        }
    }

    /// `Model::load` checks `gpu_layers` once the file is read and before a
    /// device is touched: too many layers, a CPU device and shards fail with a
    /// configuration error naming the cause, and `Some(0)` is a plain CPU
    /// model.
    #[test]
    fn load_validates_gpu_layers_before_a_device_opens() {
        use super::super::Model;
        use super::super::paged::Sizing;
        use crate::provider::local::{ExpertPlacement, GpuAdapter, LocalDevice};

        let model = write_spec(&TinySpec::llama());
        let content = gguf::Content::open(&model.path).unwrap();
        let n = super::super::read_hparams(&content).unwrap().n_layer;
        let vulkan = LocalDevice::Vulkan(GpuAdapter::Default);
        let load = |device: LocalDevice, shards: &[LocalDevice], gpu_layers: usize| {
            Model::load(
                &content,
                32,
                device,
                shards,
                &ExpertPlacement::Resident,
                &Sizing::single(32),
                Some(gpu_layers),
                &crate::provider::local::EngramCache::default(),
            )
        };
        let over = format!("gpu_layers {} exceeds the {n} layers", n + 1);
        for (result, what) in [
            (load(vulkan, &[], n + 1), over.as_str()),
            (load(LocalDevice::Cpu, &[], 1), "gpu_layers needs a GPU"),
            (load(vulkan, &[vulkan], 1), "tensor parallelism"),
        ] {
            match result {
                Err(CucaError::Config(m)) => assert!(m.contains(what), "{m}"),
                Err(other) => panic!("{what}: {other:?}"),
                Ok(_) => panic!("{what}: loaded"),
            }
        }
        // Nothing on the GPU: the CPU model, whatever the device.
        assert!(matches!(load(vulkan, &[], 0), Ok(Model::Cpu(_))));
    }

    /// The layers whose cache crosses the boundary are the K/V sources of
    /// the shared layers on the far side: none for a model without shared
    /// layers, the last sliding and the last full-attention owner for the
    /// tiny Gemma 4 whenever a sharing layer is on the GPU and its source
    /// is not.
    #[test]
    fn mirrored_layers_are_the_kv_sources_across_the_boundary() {
        let llama = hparams(&write_spec(&TinySpec::llama()));
        for first in 1..llama.n_layer {
            assert!(mirrored(&llama, first).is_empty(), "first {first}");
        }
        let gemma = hparams(&write_tiny_model_with(TinyOptions::default()));
        let sharing: Vec<usize> = (0..gemma.n_layer)
            .filter(|&il| gemma.layers[il].kv_source != il)
            .collect();
        assert!(!sharing.is_empty(), "the tiny Gemma 4 shares K/V layers");
        let mut crossing = 0;
        for first in 1..gemma.n_layer {
            let mirror = mirrored(&gemma, first);
            for &source in &mirror {
                assert!(source < first, "first {first}: {source} is on the CPU");
                assert!(gemma.layers[source].owns_kv(source));
                assert!(
                    sharing
                        .iter()
                        .any(|&il| il >= first && gemma.layers[il].kv_source == source),
                    "first {first}: no GPU layer reads {source}"
                );
            }
            // Every GPU layer reading a CPU layer is covered.
            for &il in sharing.iter().filter(|&&il| il >= first) {
                let source = gemma.layers[il].kv_source;
                assert_eq!(source < first, mirror.contains(&source), "first {first}");
            }
            crossing += usize::from(!mirror.is_empty());
        }
        assert!(crossing > 0, "some split crosses a shared cache");
    }

    /// What a stage reads of the layers before it and what a later stage
    /// reads of it, for every cut of the tiny Gemma 4 (whose shared-K/V
    /// layers read the K/V of earlier ones): `exports` of a range are the
    /// `mirrors` of the layers after it, `passes` are the sources a stage
    /// forwards untouched.
    #[test]
    fn stage_sets_follow_the_kv_sources() {
        let hp = hparams(&write_tiny_model_with(TinyOptions::default()));
        let n = hp.n_layer;
        let mut crossing = 0;
        for a in 1..n {
            for b in a + 1..=n {
                let middle = LayerRange { first: a, end: b };
                let head = LayerRange { first: 0, end: a };
                for il in 0..n {
                    let source = hp.layers[il].kv_source;
                    let reads_before = il >= b && source < a;
                    assert_eq!(
                        passes(&hp, middle).contains(&source) && il >= b,
                        reads_before,
                        "{a}..{b}: layer {il}"
                    );
                }
                let mirror = mirrors(&hp, middle);
                assert!(mirror.iter().all(|&s| s < a));
                assert!(passes(&hp, middle).iter().all(|&s| s < a));
                let from_head = exports(&hp, head);
                assert!(from_head.iter().all(|&s| hp.layers[s].owns_kv(s)));
                // What the head exports is read by the later layers only.
                assert!(
                    from_head
                        .iter()
                        .all(|&s| { (a..n).any(|il| hp.layers[il].kv_source == s) })
                );
                crossing += passes(&hp, middle).len();
            }
        }
        assert!(crossing > 0, "some stage forwards rows it does not read");
        for first in 1..n {
            let tail = LayerRange { first, end: n };
            assert_eq!(mirrors(&hp, tail), mirrored(&hp, first));
            assert_eq!(
                mirrors(&hp, tail),
                exports(
                    &hp,
                    LayerRange {
                        first: 0,
                        end: first
                    }
                )
            );
            assert!(
                passes(&hp, tail).is_empty(),
                "the last stage forwards nothing"
            );
        }
        assert_eq!(legal_cuts(&hp), (1..n).collect::<Vec<_>>());
    }

    /// A stage that reads the K/V of an earlier layer (or forwards it) fails
    /// on a carry without those rows instead of reading a pool nobody wrote.
    #[test]
    fn a_carry_without_the_rows_a_stage_needs_is_refused() {
        use super::super::StateSlot;
        use super::super::paged::{KvPools, Sizing};
        use crate::provider::local::ExpertPlacement;
        let file = write_tiny_model_with(TinyOptions::default());
        let content = gguf::Content::open(&file.path).unwrap();
        let hp = super::super::read_hparams(&content).unwrap();
        let n = hp.n_layer;
        let first = (1..n)
            .find(|&f| !mirrored(&hp, f).is_empty())
            .expect("a cut that crosses a shared cache");
        let sizing = Sizing {
            max_sequences: 1,
            kv_tokens: 64,
            max_rows: 16,
        };
        let load = |range: LayerRange| {
            let part = Part {
                range,
                shared: None,
                export: exports(&hp, range),
                mirror: mirrors(&hp, range),
                engram: None,
            };
            Transformer::load_range(
                &content,
                32,
                &ExpertPlacement::Resident,
                &sizing,
                &part,
                false,
            )
            .unwrap()
        };
        let mut head = load(LayerRange {
            first: 0,
            end: first,
        });
        let mut tail = load(LayerRange { first, end: n });
        let mut pools = KvPools::new(head.kv_layout());
        let mut seq = pools.seq();
        pools.reserve(&mut seq, 6).unwrap();
        let tokens = [3u32, 5, 7, 1, 9, 2];
        let batch = Batch {
            rows: vec![SeqRows {
                history: &[],
                tokens: &tokens,
                start_pos: 0,
                tables: &seq.tables,
                state: StateSlot(0),
                want: Want::Last,
                hidden: Hidden::None,
            }],
            exact: true,
        };
        let carry = |head: &mut Transformer| match head.forward_stage(&batch, None).unwrap() {
            StageOut::Carry(c) => c,
            StageOut::Done(_) => panic!("the head stage holds no logits"),
        };
        let mut bare = carry(&mut head);
        assert!(!bare.kv.is_empty(), "the cut crosses a shared cache");
        bare.kv.clear();
        let Err(e) = tail.forward_stage(&batch, Some(bare)) else {
            panic!("a carry without the rows was accepted");
        };
        assert!(e.to_string().contains("holds no K/V rows for layer"), "{e}");
        // The complete carry runs.
        let whole = carry(&mut head);
        assert!(matches!(
            tail.forward_stage(&batch, Some(whole)).unwrap(),
            StageOut::Done(_)
        ));
    }

    type Rows = Vec<(Option<Vec<f32>>, Option<Vec<f32>>)>;

    /// The steps the stage tests run over two sequences: a prefill, decode
    /// rows, a verification-shaped call (`Want::All` over a few rows) and
    /// every `Hidden` the stages can carry.
    fn stage_steps() -> Vec<Vec<(usize, usize, Want, Hidden)>> {
        vec![
            vec![
                (0, 13, Want::None, Hidden::None),
                (1, 6, Want::Last, Hidden::PostNorm),
            ],
            vec![
                (0, 7, Want::Last, Hidden::PostNormAll),
                (1, 1, Want::Last, Hidden::None),
            ],
            vec![
                (0, 1, Want::Last, Hidden::PostNorm),
                (1, 4, Want::All, Hidden::PostNorm),
            ],
            vec![
                (0, 5, Want::All, Hidden::PostNormAll),
                (1, 1, Want::Last, Hidden::PostNormAll),
            ],
            vec![(1, 2, Want::None, Hidden::PostNormAll)],
        ]
    }

    /// Run the steps through `models`: one model is a whole pass, several are
    /// chained stages, each handing its carry on. Returns the logits and the
    /// hidden rows of every sequence of every step.
    fn drive_stages(models: &mut [Transformer], exact: bool) -> Vec<Rows> {
        drive_stages_as(models, exact, CarryDtype::F32)
    }

    /// [`drive_stages`] with every carry written to the wire in `dtype` and
    /// read back, as a worker and its coordinator do.
    fn drive_stages_as(models: &mut [Transformer], exact: bool, dtype: CarryDtype) -> Vec<Rows> {
        use super::super::StateSlot;
        use super::super::paged::KvPools;
        let layout = models[0].kv_layout().clone();
        let mut pools = KvPools::new(&layout);
        let mut seqs: Vec<_> = (0..2).map(|_| pools.seq()).collect();
        let mut ids: Vec<Vec<u32>> = vec![Vec::new(); 2];
        let mut out = Vec::new();
        for step in stage_steps() {
            let tokens: Vec<Vec<u32>> = step
                .iter()
                .map(|&(i, n, ..)| {
                    (ids[i].len()..ids[i].len() + n)
                        .map(|p| ((p * 5 + 3 + i) % VOCAB) as u32)
                        .collect()
                })
                .collect();
            for (&(i, ..), tokens) in step.iter().zip(&tokens) {
                pools
                    .reserve(&mut seqs[i], ids[i].len() + tokens.len())
                    .unwrap();
            }
            let batch = Batch {
                rows: step
                    .iter()
                    .zip(&tokens)
                    .map(|(&(i, _, want, hidden), tokens)| SeqRows {
                        history: &ids[i],
                        tokens,
                        start_pos: ids[i].len(),
                        tables: &seqs[i].tables,
                        state: StateSlot(i),
                        want,
                        hidden,
                    })
                    .collect(),
                exact,
            };
            let forwards = if let [whole] = models {
                whole.forward_batch(&batch).unwrap()
            } else {
                let mut carry = None;
                let mut done = None;
                for stage in models.iter_mut() {
                    match stage.forward_stage(&batch, carry.take()).unwrap() {
                        StageOut::Carry(c) => {
                            let bytes = stageframe::carry_bytes_as(&c, dtype).unwrap();
                            carry = Some(stageframe::decode_carry_as(&bytes).unwrap().1);
                        }
                        StageOut::Done(f) => done = Some(f),
                    }
                }
                done.expect("the last stage answers with logits")
            };
            out.push(forwards.into_iter().map(|f| (f.logits, f.hidden)).collect());
            drop(batch);
            for (&(i, ..), tokens) in step.iter().zip(&tokens) {
                ids[i].extend_from_slice(tokens);
            }
        }
        out
    }

    /// `model` cut into stages equals the whole model bit for bit: every
    /// legal two-way cut and every legal three-way cut, in exact and in
    /// batch-shaped arithmetic.
    fn check_stages(model: &super::super::tiny::TinyModel, context: usize, what: &str) {
        check_stages_as(model, context, what, CarryDtype::F32, 0.0);
    }

    /// [`check_stages`] with the carries written in `dtype`: for `F32` the
    /// stages equal the whole model bit for bit; for another type the
    /// logits stay within `tol` of the largest logit (relative) and differ
    /// from the whole model's somewhere, as the rounding must show. Returns
    /// the worst relative difference seen.
    fn check_stages_as(
        model: &super::super::tiny::TinyModel,
        context: usize,
        what: &str,
        dtype: CarryDtype,
        tol: f32,
    ) -> f32 {
        use super::super::paged::Sizing;
        use crate::provider::local::{EngramCache, ExpertPlacement};
        let content = gguf::Content::open(&model.path).unwrap();
        let hp = super::super::read_hparams(&content).unwrap();
        let sizing = Sizing {
            max_sequences: 2,
            kv_tokens: 2 * context,
            max_rows: 16,
        };
        let load_range = |range: LayerRange| {
            let part = Part {
                range,
                shared: None,
                export: exports(&hp, range),
                mirror: mirrors(&hp, range),
                engram: None,
            };
            Transformer::load_range(
                &content,
                context,
                &ExpertPlacement::Resident,
                &sizing,
                &part,
                false,
            )
            .unwrap()
        };
        let whole = || {
            Transformer::load_placed(
                &content,
                context,
                &ExpertPlacement::Resident,
                &sizing,
                &EngramCache::default(),
            )
            .unwrap()
        };
        let n = hp.n_layer;
        let cuts = legal_cuts(&hp);
        assert!(!cuts.is_empty(), "{what}: no legal cut");
        let mut plans: Vec<Vec<usize>> = cuts.iter().map(|&c| vec![0, c, n]).collect();
        for (i, &a) in cuts.iter().enumerate() {
            plans.extend(cuts[i + 1..].iter().map(|&b| vec![0, a, b, n]));
        }
        let mut worst = 0f32;
        for exact in [true, false] {
            let want = drive_stages(&mut [whole()], exact);
            assert!(
                want.iter().flatten().any(|(logits, _)| logits.is_some()),
                "{what}: the reference produces logits"
            );
            for plan in &plans {
                let mut stages: Vec<Transformer> = plan
                    .windows(2)
                    .map(|w| {
                        load_range(LayerRange {
                            first: w[0],
                            end: w[1],
                        })
                    })
                    .collect();
                let have = drive_stages_as(&mut stages, exact, dtype);
                if dtype == CarryDtype::F32 {
                    assert!(have == want, "{what}: stages {plan:?} (exact {exact})");
                    continue;
                }
                let (mut diff, mut scale) = (0f32, 0f32);
                for ((h, _), (w, _)) in have.iter().flatten().zip(want.iter().flatten()) {
                    let (Some(h), Some(w)) = (h, w) else {
                        assert_eq!(
                            h.is_some(),
                            w.is_some(),
                            "{what}: the same rows have logits"
                        );
                        continue;
                    };
                    assert_eq!(h.len(), w.len());
                    for (a, b) in h.iter().zip(w) {
                        diff = diff.max((a - b).abs());
                        scale = scale.max(b.abs());
                    }
                }
                worst = worst.max(diff / scale);
                assert!(
                    diff <= scale * tol,
                    "{what}: stages {plan:?} (exact {exact}) in {dtype:?}: |dlogit| {diff} of {scale}"
                );
            }
        }
        worst
    }

    /// Stages whose carries travel in `bf16` or `f16` stay near the whole
    /// model: the logits differ (the rounding is real) by a fraction of the
    /// largest logit well under the scale of the logits themselves, on every
    /// architecture the stage tests cover, including the per-layer inputs,
    /// shared K/V rows and hyper-connection streams a carry holds.
    ///
    /// The bound is loose on purpose. The tiny models are quantized, and the
    /// engine quantizes activations to 8 bits for its matmuls, so any change
    /// of the carry, down to one bit of one value, re-rounds later layers and
    /// moves the logits by about the same amount whatever the carry type (the
    /// tiny random models, with their sharp logits, see up to a quarter of
    /// the largest logit). What the bound catches is a carry that is wrong,
    /// not one that is imprecise: by hand, a decoder that drops the
    /// per-layer part, zeroes the pre part, swaps the mirrored K and V,
    /// zeroes a 64th of the hidden values or scales them by 1.01 each fail it.
    #[test]
    fn reduced_carries_stay_near_the_whole_model() {
        use super::super::tiny::{
            CONTEXT, TinyDeepseek41, TinyGemma4Moe, TinyQwen35, write_deepseek41, write_gemma4_moe,
            write_qwen35,
        };
        for (dtype, tol) in [(CarryDtype::Bf16, 0.5), (CarryDtype::F16, 0.5)] {
            let mut worst = 0f32;
            for spec in TinySpec::all() {
                worst = worst.max(check_stages_as(
                    &write_spec(&spec),
                    CONTEXT,
                    spec.arch,
                    dtype,
                    tol,
                ));
            }
            worst = worst.max(check_stages_as(
                &write_tiny_model_with(TinyOptions::default()),
                CONTEXT,
                "gemma4",
                dtype,
                tol,
            ));
            worst = worst.max(check_stages_as(
                &write_gemma4_moe(&TinyGemma4Moe::default()),
                CONTEXT,
                "gemma4 moe",
                dtype,
                tol,
            ));
            worst = worst.max(check_stages_as(
                &write_qwen35(&TinyQwen35::default()),
                CONTEXT,
                "qwen35",
                dtype,
                tol,
            ));
            worst = worst.max(check_stages_as(
                &write_deepseek41(&TinyDeepseek41::default()),
                32,
                "deepseek41",
                dtype,
                tol,
            ));
            assert!(worst > 0.0, "{dtype:?}: rounding left no trace");
        }
    }

    /// A model cut into stages at any legal layer boundary computes what the
    /// whole model does: the stages hand over the residual stream, the
    /// per-layer inputs, the hyper-connection streams and the K/V rows of the
    /// shared layers, and each holds only its layers' state.
    #[test]
    fn cpu_stages_equal_the_whole_model_at_every_cut() {
        use super::super::tiny::{
            CONTEXT, TinyDeepseek41, TinyGemma4Moe, TinyMistral4, TinyQwen35, write_deepseek41,
            write_gemma4_moe, write_mistral4, write_qwen35,
        };
        for spec in TinySpec::all() {
            check_stages(&write_spec(&spec), CONTEXT, spec.arch);
        }
        check_stages(
            &write_tiny_model_with(TinyOptions::default()),
            CONTEXT,
            "gemma4",
        );
        check_stages(
            &write_gemma4_moe(&TinyGemma4Moe::default()),
            CONTEXT,
            "gemma4 moe",
        );
        check_stages(&write_qwen35(&TinyQwen35::default()), CONTEXT, "qwen35");
        check_stages(
            &write_mistral4(&TinyMistral4::default()),
            CONTEXT,
            "mistral4",
        );
        for candidates in [Some(3), None] {
            check_stages(
                &write_deepseek41(&TinyDeepseek41 {
                    candidates,
                    ..TinyDeepseek41::default()
                }),
                32,
                "deepseek41",
            );
        }
    }
}
