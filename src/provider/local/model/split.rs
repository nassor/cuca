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
//! [`SplitModel::forward_batch`] runs the CPU half ([`Transformer`] with a
//! [`Front`]: embeddings, per-layer inputs and the first layers, no LM head)
//! over the rows of every sequence of the batch, then hands the residual
//! stream `[rows, n_embd]` (and the per-layer inputs) to the GPU half
//! ([`GpuTransformer`] with a [`Back`]: the last layers and the head). The
//! hidden state crosses the boundary once per pass: one host read of the
//! CPU half's output and one upload; nothing else moves between the halves.
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
//! pool of its own under the same tables ([`mirrored`] lists the layers). They
//! ride the same host-to-device crossing as the hidden state. The mirror
//! pools are not counted in `kv_cache_bytes`, which counts each layer once.
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

use super::params::Hparams;
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

/// The K/V-owning layers before `first` whose cache a layer from `first` on
/// reads (the shared-K/V layers of Gemma 4 that sit on the other side of the
/// boundary from their source), ascending.
#[cfg(any(feature = "provider-local-gpu", test))]
pub(crate) fn mirrored(hp: &Hparams, first: usize) -> Vec<usize> {
    let mut sources: Vec<usize> = hp
        .layers
        .iter()
        .enumerate()
        .filter(|(il, spec)| *il >= first && spec.kv_source < first)
        .map(|(_, spec)| spec.kv_source)
        .collect();
    sources.sort_unstable();
    sources.dedup();
    sources
}

/// The rows a CPU layer cached in one pass for the GPU layers that read its
/// K/V ([`mirrored`]).
#[cfg(feature = "provider-local-gpu")]
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
    use std::path::Path;
    use std::sync::Arc;

    use candle_core::quantized::gguf_file;

    use super::super::cpu::{Forward, Front, Transformer};
    use super::super::delta::DeltaState;
    use super::super::gpu::{Back, GpuSnapshot, GpuTransformer};
    use super::super::paged::{KvLayout, Sizing};
    use super::super::tier::TierShared;
    use super::super::{Batch, read_hparams};
    use super::{Hparams, mirrored};
    use crate::error::CucaError;
    use crate::provider::local::{ExpertPlacement, ExpertTiers, LocalDevice};

    /// A model whose first layers run on the CPU and whose last layers (and
    /// head) run on a GPU; see the module docs.
    pub(crate) struct SplitModel {
        front: Transformer,
        back: GpuTransformer,
        /// The first layer on the GPU.
        first: usize,
        n_layer: usize,
        tiers: Option<Arc<TierShared>>,
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
        /// As [`Transformer::load_front`] and [`GpuTransformer::load_back`].
        pub(crate) fn load(
            content: &gguf_file::Content,
            path: &Path,
            context_length: usize,
            device: LocalDevice,
            experts: &ExpertPlacement,
            sizing: &Sizing,
            first: usize,
        ) -> Result<Self, CucaError> {
            Self::load_with(
                content,
                path,
                context_length,
                device,
                experts,
                sizing,
                first,
                false,
            )
        }

        /// [`SplitModel::load`], with the CPU half's matmul weights held as
        /// f32 when `dequantize` (the reference of the tests).
        #[expect(
            clippy::too_many_arguments,
            reason = "the file, the placement and the split point are independent inputs"
        )]
        pub(crate) fn load_with(
            content: &gguf_file::Content,
            path: &Path,
            context_length: usize,
            device: LocalDevice,
            experts: &ExpertPlacement,
            sizing: &Sizing,
            first: usize,
            dequantize: bool,
        ) -> Result<Self, CucaError> {
            let hp = read_hparams(content)?;
            let (cpu_experts, gpu_experts, shared) = place_experts(&hp, experts, first)?;
            let mirror = mirrored(&hp, first);
            let front = Transformer::load_front(
                content,
                path,
                context_length,
                &cpu_experts,
                sizing,
                &Front {
                    layers: first,
                    shared: shared.clone(),
                    export: mirror.clone(),
                },
                dequantize,
            )?;
            let back = GpuTransformer::load_back(
                content,
                path,
                context_length,
                device,
                &gpu_experts,
                sizing,
                &Back {
                    first,
                    shared: shared.clone(),
                    mirror,
                },
            )?;
            Ok(Self {
                front,
                back,
                first,
                n_layer: hp.n_layer,
                tiers: shared,
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
            let out = self.front.forward_front(batch)?;
            let result = self
                .back
                .forward_back(batch, out.hidden, out.per_layer, &out.kv);
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
            if !self.is_hybrid() {
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
    }
}

#[cfg(test)]
mod tests {
    use super::super::tiny::{TinyOptions, TinySpec, write_spec, write_tiny_model_with};
    use super::*;

    fn hparams(model: &super::super::tiny::TinyModel) -> Hparams {
        let mut reader = std::io::BufReader::new(std::fs::File::open(&model.path).unwrap());
        let content = candle_core::quantized::gguf_file::Content::read(&mut reader).unwrap();
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
        let mut reader = std::io::BufReader::new(std::fs::File::open(&model.path).unwrap());
        let content = candle_core::quantized::gguf_file::Content::read(&mut reader).unwrap();
        let n = super::super::read_hparams(&content).unwrap().n_layer;
        let vulkan = LocalDevice::Vulkan(GpuAdapter::Default);
        let load = |device: LocalDevice, shards: &[LocalDevice], gpu_layers: usize| {
            Model::load(
                &content,
                &model.path,
                32,
                device,
                shards,
                &ExpertPlacement::Resident,
                &Sizing::single(32),
                Some(gpu_layers),
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
}
