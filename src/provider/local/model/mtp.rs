//! Multi-token-prediction (MTP) drafting: a separate Gemma 4 assistant file,
//! or the prediction head inside the target's own file (Qwen 3.5).
//!
//! # Separate drafter file
//!
//! The `gemma4-assistant` checkpoint (Google's "MTP drafter", published as a
//! separate ~170 MB GGUF next to each Gemma 4 model) is a four-layer,
//! query-only decoder of width 256. It owns no K/V projections: every layer
//! attends to the K/V cache of the *target* model, the sliding-window layers
//! to the cache the target's last sliding layer reads and the full-attention
//! layer to the cache the target's last full layer reads (exactly the caches
//! Gemma 4's own shared-K/V layers read), so drafting costs no extra cache
//! memory. One drafting step takes
//!
//! - the target's final-norm hidden state at the last cached position
//!   (`n_embd_out` = the target's embedding width) and
//! - the embedding of the token that follows it (scaled as the target's first
//!   layer sees it),
//!
//! concatenated in that order (embedding first) into `nextn.pre_projection`
//! (`2 * 1536 -> 256`), runs the four layers with its query at the next
//! position, applies its final norm, and returns
//!
//! - logits over the vocabulary from its own tied embedding table (`256 x
//!   262144`) and
//! - `nextn.post_projection` of the normed output (`256 -> 1536`): the hidden
//!   state that feeds the next drafting step, so a chain of tokens is
//!   drafted without running the target.
//!
//! [`Drafter::draft`] runs such a chain. Verification (one batched target
//! forward over the pending token and the drafts, longest-prefix acceptance,
//! cache truncation) lives in the engine, which keeps the output equal to the
//! target's own (see `engine.rs`).
//!
//! # Memory
//!
//! The drafter's F16 weights are held as f32 (about twice the file: 340 MB
//! for the E2B drafter) and its K/V is the target's cache, shared by
//! reference. Drafting is available on the CPU device only.
//!
//! # In-trunk head (`DrafterSource::InTrunk`)
//!
//! A Qwen 3.5 `*-MTP` file embeds one prediction block after its trunk (see
//! [`super::cpu::NextnHead`] for its semantics, cache layout and memory). It
//! has no file of its own and no K/V sharing: it keeps a K/V cache of its own
//! over every trunk position, so the engine feeds it the trunk's final-norm
//! hidden rows as they appear ([`Drafter::observe`], after every prefill
//! chunk and every decode round, drafted or not) and tells it when the trunk
//! changed tokens ([`Drafter::truncate`], [`Drafter::reset`]). Drafting
//! itself ([`Drafter::draft`]) has the same contract as the file drafter,
//! except that the logits it hands to `pick` cover only the first token ids
//! of the vocabulary (a draft step ranks a 65,536-id prefix; an index is
//! still a token id).
//! The head runs on the CPU only (the trunk it belongs to does not run on a
//! GPU) and only proposes: verification keeps the output equal to plain
//! decoding whatever the head computes.
//!
//! The cache stays consistent with the trunk by one rule: the head holds a
//! pair `(h[j], t[j + 1])` only while both its hidden row and its token are
//! the ones the trunk now has. The engine truncates the drafter to the
//! shared prefix of the cached ids and the new prompt before prefilling, the
//! replayed rows (whatever length the trunk could keep) are observed from the
//! position the trunk resumes at, and a failed pass resets it.

use std::path::Path;

use candle_core::quantized::gguf_file;

use super::arch::Arch;
use super::cpu::{ExtKv, NextnHead, Transformer};
#[cfg(feature = "provider-local-gpu")]
use super::gpu::GpuDrafter;
use super::{Model, err};
use crate::error::CucaError;

/// A loaded drafter bound to one target model, on the target's device.
// One `Drafter` exists per engine, so the size gap between the variants costs
// nothing worth a box.
#[allow(clippy::large_enum_variant)]
pub(crate) enum Drafter {
    /// A separate `gemma4-assistant` drafter on the CPU, sharing the target's
    /// K/V cache.
    Cpu { net: Transformer, ext: ExtKv },
    #[cfg(feature = "provider-local-gpu")]
    Gpu(GpuDrafter),
    /// The prediction head embedded in the target's own file (CPU only).
    InTrunk(NextnHead),
}

impl Drafter {
    /// Load the drafter at `path` (index `content`) for `target`.
    ///
    /// # Errors
    ///
    /// [`CucaError::Provider`] when `target` is not a Gemma 4 model, the
    /// hidden widths or vocabularies differ, or the file is not a drafter.
    pub(crate) fn open(
        content: &gguf_file::Content,
        path: &Path,
        target: &Model,
        context_length: usize,
    ) -> Result<Self, CucaError> {
        if target.arch() != Arch::Gemma4 {
            return Err(err(format!(
                "the `gemma4-assistant` drafter pairs with a `gemma4` model, not `{}`",
                target.arch().name()
            )));
        }
        match target {
            Model::Cpu(target) => {
                let net = Transformer::load_drafter(content, path, context_length)?;
                if net.n_embd_out() != target.n_embd() {
                    return Err(err(format!(
                        "drafter hidden width {} differs from the target's {}: it was trained for another model size",
                        net.n_embd_out(),
                        target.n_embd()
                    )));
                }
                if net.vocab() != target.vocab() {
                    return Err(err(format!(
                        "drafter vocabulary {} differs from the target's {}",
                        net.vocab(),
                        target.vocab()
                    )));
                }
                Ok(Self::Cpu {
                    ext: target.ext_kv()?,
                    net,
                })
            }
            #[cfg(feature = "provider-local-gpu")]
            Model::Gpu(target) => Ok(Self::Gpu(GpuDrafter::open(
                content,
                path,
                target,
                context_length,
            )?)),
        }
    }

    /// The prediction head embedded in `target`'s own file (`content` is its
    /// index, `path` its location).
    ///
    /// # Errors
    ///
    /// [`CucaError::Provider`] when `target` runs on a GPU, or its file has
    /// no prediction block or lacks one of the block's tensors.
    pub(crate) fn open_in_trunk(
        content: &gguf_file::Content,
        path: &Path,
        target: &Model,
    ) -> Result<Self, CucaError> {
        match target {
            Model::Cpu(target) => Ok(Self::InTrunk(target.load_head(content, path)?)),
            #[cfg(feature = "provider-local-gpu")]
            Model::Gpu(_) => Err(err(
                "the in-trunk prediction head runs on the CPU only: use `LocalDevice::Cpu`",
            )),
        }
    }

    /// Whether the drafter needs the trunk's hidden row of every position
    /// ([`Drafter::observe`]), not just the last one.
    pub(crate) fn observes(&self) -> bool {
        matches!(self, Self::InTrunk(_))
    }

    /// Tell the drafter the trunk ran the tokens `tokens[i]` at positions
    /// `pos + i` and produced the final-norm hidden rows `hidden` (one per
    /// token, in order). A no-op for the file drafters, which read the
    /// target's cache and need only the last hidden row at draft time.
    ///
    /// # Errors
    ///
    /// [`CucaError::Provider`] for a failed head forward or a `target` on
    /// another device than the drafter.
    pub(crate) fn observe(
        &mut self,
        target: &Model,
        pos: usize,
        tokens: &[u32],
        hidden: &[f32],
    ) -> Result<(), CucaError> {
        match (self, target) {
            (Self::InTrunk(head), Model::Cpu(target)) => {
                target.head_observe(head, pos, tokens, hidden)
            }
            #[cfg(feature = "provider-local-gpu")]
            (Self::InTrunk(_), _) => {
                Err(err("the drafter and its target run on different devices"))
            }
            _ => Ok(()),
        }
    }

    /// The first `shared` tokens of the trunk's sequence stay and any later
    /// one may change (a new prompt that shares `shared` tokens with the
    /// cached ids): forget what depended on the rest. A no-op for the file
    /// drafters.
    pub(crate) fn truncate(&mut self, shared: usize) {
        if let Self::InTrunk(head) = self {
            head.truncate(shared);
        }
    }

    /// Forget everything the drafter learned from the trunk (the trunk's
    /// cache was emptied). A no-op for the file drafters.
    pub(crate) fn reset(&mut self) {
        if let Self::InTrunk(head) = self {
            head.reset();
        }
    }

    /// Draft `k` tokens after the pending token `first`.
    ///
    /// `hidden` is the target's final-norm hidden state at its last cached
    /// position, `len` the number of cached positions (`first` sits at
    /// position `len` and is not cached). Each step's logits go to `pick`,
    /// which chooses (and may record) the drafted token.
    ///
    /// # Errors
    ///
    /// [`CucaError::Provider`] for a forward-pass failure or a `target` on
    /// another device than the drafter.
    pub(crate) fn draft(
        &self,
        target: &Model,
        hidden: &[f32],
        first: u32,
        len: usize,
        k: usize,
        pick: &mut dyn FnMut(&mut [f32]) -> u32,
    ) -> Result<Vec<u32>, CucaError> {
        match (self, target) {
            (Self::Cpu { net, ext }, Model::Cpu(target)) => {
                let mut drafts = Vec::with_capacity(k);
                let mut hidden = hidden.to_vec();
                let mut token = first;
                for _ in 0..k {
                    let mut input = target.embedding_row(token)?;
                    input.extend_from_slice(&hidden);
                    let (mut logits, next_hidden) = net.draft(ext, len, &input, len)?;
                    token = pick(&mut logits);
                    drafts.push(token);
                    hidden = next_hidden;
                }
                Ok(drafts)
            }
            (Self::InTrunk(head), Model::Cpu(target)) => {
                target.head_draft(head, hidden, first, len, k, pick)
            }
            #[cfg(feature = "provider-local-gpu")]
            (Self::Gpu(drafter), Model::Gpu(target)) => {
                drafter.draft(target, hidden, first, len, k, pick)
            }
            #[cfg(feature = "provider-local-gpu")]
            _ => Err(err("the drafter and its target run on different devices")),
        }
    }
}
