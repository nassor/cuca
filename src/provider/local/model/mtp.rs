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
//! target's own (see `engine`).
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
//! chunk and every decode round that drafts; a plain round only defers them,
//! see *Lazy upkeep*) and tells it when the trunk
//! changed tokens ([`Drafter::truncate`], [`Drafter::reset`]). Drafting
//! itself ([`Drafter::draft`]) has the same contract as the file drafter,
//! except that the logits it hands to `pick` cover only the first token ids
//! of the vocabulary (a draft step ranks a 65,536-id prefix; an index is
//! still a token id).
//! The head runs on the CPU only (a hybrid trunk can run on a GPU, but the
//! head is not implemented there) and only proposes: verification keeps the output equal to plain
//! decoding whatever the head computes.
//!
//! The cache stays consistent with the trunk by one rule: the head holds a
//! pair `(h[j], t[j + 1])` only while both its hidden row and its token are
//! the ones the trunk now has. The engine truncates the drafter to the
//! shared prefix of the cached ids and the new prompt before prefilling, the
//! replayed rows (whatever length the trunk could keep) are observed from the
//! position the trunk resumes at, and a failed pass resets it.
//!
//! # EAGLE drafter (`mistral4-eagle`)
//!
//! Mistral's EAGLE head for Mistral Small 4 is a separate GGUF (converted from
//! the published `mistralai/Mistral-Small-4-119B-2603-eagle`) of two
//! latent-attention layers with a dense FFN. It has no embedding and no LM
//! head: it reads the target's, and a draft step ranks the whole vocabulary.
//! Like the in-trunk head it keeps a K/V cache of its own over every trunk
//! position, is fed by [`Drafter::observe`] with the trunk's final-norm
//! hidden rows, follows [`Drafter::truncate`] and [`Drafter::reset`] by the
//! same pair rule, serves one sequence at a time, and runs on the CPU only.
//! See [`super::cpu::EagleHead`] for the semantics (embedding first, then the
//! hidden row; RoPE position of the hidden row), the cost of committing a
//! pair, and the memory.
//!
//! # DFlash block drafter (`dflash` file)
//!
//! A DFlash drafter ([`super::cpu::dflash::DflashNet`]) proposes a whole block of
//! tokens in one pass from the target's per-layer hidden features instead of
//! chaining one step per token. It observes every position like the in-trunk
//! head, but the rows it takes are the residual streams after chosen target
//! layers ([`Drafter::feature_rows`], `Hidden::Layers`), not the final-norm
//! row, and one [`Drafter::draft`] call costs one block pass whatever the
//! number of drafts kept ([`Drafter::block`]: the engine's cost model prices
//! it per round). CPU only; the same truncate, reset and observe rules keep
//! its context cache consistent with the trunk's.
//!
//! # Lazy upkeep
//!
//! The three drafters above take every committed position into a cache of
//! their own: 4.4 ms of a 90 ms token for the EAGLE drafter on Mistral Small
//! 4, 3.5 ms of 288 ms for the DFlash drafter on Muse Glimmer, 0.09 ms of
//! 17.3 ms for the Qwen 3.5 head. A round that drafts needs the cache, a
//! plain round does not, so the engine hands a plain round's rows over with
//! [`Drafter::defer`] instead of [`Drafter::observe`]: the drafter keeps the
//! tokens and rows (a [`Lag`], at most [`LAG_ROWS`] positions and
//! [`LAG_BYTES`] bytes; at the bound it catches up first) and the range
//! of positions it has not seen, and [`Drafter::catch_up`] takes them into
//! the cache in chunks of the trunk's prefill size before the next round
//! that drafts. A row's cache entry does not depend on the rows around it
//! (the CPU kernels are row-exact), so the cache after catching up is what
//! observing each round as it came leaves, bit for bit, and so are the
//! proposals: lazy upkeep moves cost, never output. A truncation drops the
//! held rows the new prompt does not share, a replay at an earlier position
//! replaces the ones it covers and is observed together with the rest, and a
//! reset forgets them.

use std::path::Path;

use candle_core::quantized::gguf_file;

use super::arch::Arch;
use super::cpu::dflash::DflashNet;
use super::cpu::{EagleHead, Hidden, NextnHead, Transformer};
#[cfg(feature = "provider-local-gpu")]
use super::gpu::GpuDrafter;
use super::paged::BlockTable;
use super::{Model, err, prefill_chunk};
use crate::error::CucaError;

/// Most positions a lazy drafter holds unobserved ([`Lag`]).
const LAG_ROWS: usize = 1024;

/// Most bytes of trunk rows a lazy drafter holds unobserved: 16 MiB is
/// 1,024 rows of the EAGLE drafter's 4,096 floats and 126 feature rows of
/// the DFlash drafter on Muse Glimmer 30B (5 x 6,656 floats). A drafter that
/// would exceed either bound observes what it holds first, so the bound
/// costs a batched pass, never a proposal.
const LAG_BYTES: usize = 16 << 20;

/// The trunk rows a drafter was told about while the engine ran plain rounds
/// ([`Drafter::defer`]) and has not taken into its own cache yet: `tokens[i]`
/// sits at position `pos + i` and `rows` holds its `width` floats of trunk
/// output (final-norm hidden row, or the feature rows of a block drafter).
/// At most [`LAG_ROWS`] rows and [`LAG_BYTES`] bytes are held.
pub(crate) struct Lag {
    /// Rows per catch-up call: the trunk's prefill chunk, which bounds the
    /// transient size of a batched pass.
    chunk: usize,
    /// Rows held at most, before the byte bound ([`LAG_ROWS`]; lowered by
    /// tests).
    max_rows: usize,
    pos: usize,
    width: usize,
    tokens: Vec<u32>,
    rows: Vec<f32>,
    /// Test-only: [`Drafter::defer`] observes at once.
    #[cfg(test)]
    eager: bool,
    /// Test-only: drafts made while rows were held back, which the engine
    /// must never do ([`Drafter::catch_up`] comes first).
    #[cfg(test)]
    stale_drafts: std::cell::Cell<usize>,
}

impl Lag {
    fn new(context_length: usize) -> Self {
        Self {
            chunk: prefill_chunk(context_length),
            max_rows: LAG_ROWS,
            pos: 0,
            width: 0,
            tokens: Vec::new(),
            rows: Vec::new(),
            #[cfg(test)]
            eager: false,
            #[cfg(test)]
            stale_drafts: std::cell::Cell::new(0),
        }
    }

    fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Rows held at most, for the rows held now: [`LAG_ROWS`] and, whatever
    /// their width, [`LAG_BYTES`] (at least one row).
    fn cap(&self) -> usize {
        let row_bytes = self.width * size_of::<f32>();
        self.max_rows.min(LAG_BYTES / row_bytes.max(1)).max(1)
    }

    fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// The position after the last held row.
    fn end(&self) -> usize {
        self.pos + self.tokens.len()
    }

    /// Drop the rows at positions of at least `from`.
    fn trim_from(&mut self, from: usize) {
        let keep = from.saturating_sub(self.pos).min(self.tokens.len());
        self.tokens.truncate(keep);
        self.rows.truncate(keep * self.width);
    }

    /// Append the rows of the tokens `tokens` at positions `pos..`; they must
    /// follow the held rows (`pos == self.end()`) or the lag must be empty.
    fn push(&mut self, pos: usize, tokens: &[u32], rows: &[f32]) -> Result<(), CucaError> {
        if tokens.is_empty() || !rows.len().is_multiple_of(tokens.len()) {
            return Err(err(format!(
                "{} trunk floats for {} tokens: not a whole row each",
                rows.len(),
                tokens.len()
            )));
        }
        let width = rows.len() / tokens.len();
        if self.is_empty() {
            self.pos = pos;
            self.width = width;
        } else if width != self.width || pos != self.end() {
            return Err(err(format!(
                "rows of {width} floats at position {pos} do not follow the {} held of {} floats up to position {}",
                self.len(),
                self.width,
                self.end()
            )));
        }
        self.tokens.extend_from_slice(tokens);
        self.rows.extend_from_slice(rows);
        Ok(())
    }

    /// Hand over everything held: the first position, the tokens and their rows.
    fn take(&mut self) -> (usize, Vec<u32>, Vec<f32>) {
        (
            self.pos,
            std::mem::take(&mut self.tokens),
            std::mem::take(&mut self.rows),
        )
    }
}

/// A loaded drafter bound to one target model, on the target's device.
// One `Drafter` exists per engine, so the size gap between the variants costs
// nothing worth a box.
#[allow(clippy::large_enum_variant)]
pub(crate) enum Drafter {
    /// A separate `gemma4-assistant` drafter on the CPU, reading the target's
    /// paged K/V cache through the sequence's block tables.
    Cpu { net: Transformer },
    #[cfg(feature = "provider-local-gpu")]
    Gpu(GpuDrafter),
    /// The prediction head embedded in the target's own file (CPU only).
    InTrunk(NextnHead, Lag),
    /// The EAGLE drafter of a Mistral Small 4 model, a separate file with a
    /// K/V cache of its own (CPU only).
    Eagle(EagleHead, Lag),
    /// A DFlash block drafter in its own file, driven by the target's
    /// per-layer features (CPU only).
    Dflash(DflashNet, Lag),
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
                // The target must have the caches a draft reads.
                target.ext_kv(&vec![Vec::new(); target.kv_layout().groups.len()])?;
                Ok(Self::Cpu { net })
            }
            #[cfg(feature = "provider-local-gpu")]
            Model::Gpu(target) => Ok(Self::Gpu(GpuDrafter::open(
                content,
                path,
                target,
                context_length,
            )?)),
            #[cfg(feature = "provider-local-gpu")]
            Model::Split(_) => Err(CucaError::Config(
                "a drafter cannot be combined with `gpu_layers`".into(),
            )),
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
            Model::Cpu(target) => Ok(Self::InTrunk(
                target.load_head(content, path)?,
                Lag::new(target.context_length()),
            )),
            #[cfg(feature = "provider-local-gpu")]
            Model::Gpu(_) => Err(err(
                "the in-trunk prediction head runs on the CPU only: use `LocalDevice::Cpu`",
            )),
            #[cfg(feature = "provider-local-gpu")]
            Model::Split(_) => Err(CucaError::Config(
                "the in-trunk prediction head runs on the CPU only: it cannot be combined with `gpu_layers`".into(),
            )),
        }
    }

    /// The EAGLE drafter in the GGUF file at `path` (index `content`) for
    /// `target`.
    ///
    /// # Errors
    ///
    /// [`CucaError::Provider`] when `target` is not a `mistral4` model on the
    /// CPU, the file is not an EAGLE drafter, or it was trained for another
    /// model (see [`Transformer::load_eagle`]).
    pub(crate) fn open_eagle(
        content: &gguf_file::Content,
        path: &Path,
        target: &Model,
    ) -> Result<Self, CucaError> {
        if target.arch() != Arch::Mistral4 {
            return Err(err(format!(
                "the `mistral4-eagle` drafter pairs with a `mistral4` model, not `{}`",
                target.arch().name()
            )));
        }
        match target {
            Model::Cpu(target) => Ok(Self::Eagle(
                target.load_eagle(content, path)?,
                Lag::new(target.context_length()),
            )),
            #[cfg(feature = "provider-local-gpu")]
            Model::Gpu(_) => Err(err(
                "the EAGLE drafter runs on the CPU only: use `LocalDevice::Cpu`",
            )),
            #[cfg(feature = "provider-local-gpu")]
            Model::Split(_) => Err(CucaError::Config(
                "the EAGLE drafter runs on the CPU only: it cannot be combined with `gpu_layers`"
                    .into(),
            )),
        }
    }

    /// The DFlash block drafter at `path` (index `content`) for `target`.
    ///
    /// # Errors
    ///
    /// [`CucaError::Provider`] when `target` runs on a GPU or the file is not
    /// a plain DFlash drafter this target can feed (see
    /// [`DflashNet::open`](super::cpu::dflash::DflashNet)).
    pub(crate) fn open_dflash(
        content: &gguf_file::Content,
        path: &Path,
        target: &Model,
        context_length: usize,
    ) -> Result<Self, CucaError> {
        match target {
            Model::Cpu(target) => Ok(Self::Dflash(
                DflashNet::open(content, path, target, context_length)?,
                Lag::new(context_length),
            )),
            #[cfg(feature = "provider-local-gpu")]
            Model::Gpu(_) => Err(err(
                "the DFlash drafter runs on the CPU only: use `LocalDevice::Cpu`",
            )),
            #[cfg(feature = "provider-local-gpu")]
            Model::Split(_) => Err(CucaError::Config(
                "the DFlash drafter runs on the CPU only: it cannot be combined with `gpu_layers`"
                    .into(),
            )),
        }
    }

    /// Whether the drafter needs the trunk's hidden row of every position
    /// ([`Drafter::observe`]), not just the last one.
    pub(crate) fn observes(&self) -> bool {
        matches!(self, Self::InTrunk(..) | Self::Eagle(..) | Self::Dflash(..))
    }

    /// The unobserved rows of a drafter that defers its upkeep, `None` for
    /// the others.
    fn lag(&mut self) -> Option<&mut Lag> {
        match self {
            Self::InTrunk(_, lag) | Self::Eagle(_, lag) | Self::Dflash(_, lag) => Some(lag),
            _ => None,
        }
    }

    /// Tell the drafter the trunk ran the tokens `tokens[i]` at positions
    /// `pos + i` and produced the final-norm hidden rows `hidden` (one per
    /// token, in order). A no-op for the file drafters, which read the
    /// target's cache and need only the last hidden row at draft time.
    ///
    /// Rows deferred earlier ([`Drafter::defer`]) that this call replaces
    /// (positions of at least `pos`) are dropped, and the ones that lead up
    /// to it are observed together with it in batched passes.
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
        self.reconcile(target, pos)?;
        match self.lag() {
            Some(lag) if !lag.is_empty() => {
                lag.push(pos, tokens, hidden)?;
                self.catch_up(target)
            }
            _ => self.observe_now(target, pos, tokens, hidden),
        }
    }

    /// Like [`Drafter::observe`] for rounds that do not draft: a drafter with
    /// a per-token upkeep (every one that observes) only records the rows and
    /// the range of positions it has not seen, and takes them into its cache
    /// in one batched pass ([`Drafter::catch_up`]) when the engine drafts
    /// again, so plain decoding does not pay for a drafter it does not use.
    /// At most [`LAG_ROWS`] rows wait: past it the drafter catches up first.
    /// A no-op for the file drafters.
    ///
    /// # Errors
    ///
    /// As [`Drafter::observe`].
    pub(crate) fn defer(
        &mut self,
        target: &Model,
        pos: usize,
        tokens: &[u32],
        hidden: &[f32],
    ) -> Result<(), CucaError> {
        #[cfg(test)]
        if self.lag().is_some_and(|lag| lag.eager) {
            return self.observe(target, pos, tokens, hidden);
        }
        self.reconcile(target, pos)?;
        if self
            .lag()
            .is_some_and(|lag| lag.len() + tokens.len() > lag.cap())
        {
            self.catch_up(target)?;
        }
        match self.lag() {
            Some(lag) => lag.push(pos, tokens, hidden),
            None => Ok(()),
        }
    }

    /// Observe every row [`Drafter::defer`] holds, in chunks of the trunk's
    /// prefill size: the drafter's cache is then what observing each round
    /// as it came would have left, bit for bit (a row's result does not
    /// depend on the rows around it). A no-op with nothing held.
    ///
    /// # Errors
    ///
    /// As [`Drafter::observe`]. The held rows are gone either way: a drafter
    /// that failed is reset by the engine, and a gap in its cache costs
    /// proposal quality, never output.
    pub(crate) fn catch_up(&mut self, target: &Model) -> Result<(), CucaError> {
        let Some(lag) = self.lag() else {
            return Ok(());
        };
        if lag.is_empty() {
            return Ok(());
        }
        let (chunk, width) = (lag.chunk.max(1), lag.width);
        let (pos, tokens, rows) = lag.take();
        for (i, ids) in tokens.chunks(chunk).enumerate() {
            let at = i * chunk;
            self.observe_now(
                target,
                pos + at,
                ids,
                &rows[at * width..(at + ids.len()) * width],
            )?;
        }
        Ok(())
    }

    /// Rows [`Drafter::defer`] holds back (tests).
    #[cfg(test)]
    pub(crate) fn pending(&self) -> usize {
        match self {
            Self::InTrunk(_, lag) | Self::Eagle(_, lag) | Self::Dflash(_, lag) => lag.len(),
            _ => 0,
        }
    }

    /// Drafts made while rows were held back (tests): always zero in an
    /// engine, which catches up before it drafts.
    #[cfg(test)]
    pub(crate) fn stale_drafts(&self) -> usize {
        match self {
            Self::InTrunk(_, lag) | Self::Eagle(_, lag) | Self::Dflash(_, lag) => {
                lag.stale_drafts.get()
            }
            _ => 0,
        }
    }

    /// Lower the number of rows [`Drafter::defer`] holds back (tests).
    #[cfg(test)]
    pub(crate) fn set_lag_cap(&mut self, cap: usize) {
        if let Some(lag) = self.lag() {
            lag.max_rows = cap;
        }
    }

    /// Make [`Drafter::defer`] observe at once, as before lazy upkeep (tests:
    /// the reference a lazy drafter must propose like).
    #[cfg(test)]
    pub(crate) fn set_eager(&mut self, eager: bool) {
        if let Some(lag) = self.lag() {
            lag.eager = eager;
        }
    }

    /// Make the held rows ones that lead up to `pos`: drop those a call at
    /// `pos` replaces, and observe the rest when they do not end there.
    fn reconcile(&mut self, target: &Model, pos: usize) -> Result<(), CucaError> {
        let Some(lag) = self.lag() else {
            return Ok(());
        };
        lag.trim_from(pos);
        if !lag.is_empty() && lag.end() != pos {
            self.catch_up(target)?;
        }
        Ok(())
    }

    /// [`Drafter::observe`] for the rows of one call, as they are.
    fn observe_now(
        &mut self,
        target: &Model,
        pos: usize,
        tokens: &[u32],
        hidden: &[f32],
    ) -> Result<(), CucaError> {
        match (self, target) {
            (Self::InTrunk(head, _), Model::Cpu(target)) => {
                target.head_observe(head, pos, tokens, hidden)
            }
            (Self::Eagle(head, _), Model::Cpu(target)) => {
                target.eagle_observe(head, pos, tokens, hidden)
            }
            (Self::Dflash(net, _), Model::Cpu(_)) => {
                if hidden.len() != tokens.len() * net.feature_width() {
                    return Err(err(format!(
                        "DFlash observe: {} tokens with {} feature floats, expected {} each",
                        tokens.len(),
                        hidden.len(),
                        net.feature_width()
                    )));
                }
                net.observe(pos, hidden)
            }
            #[cfg(feature = "provider-local-gpu")]
            (Self::InTrunk(..) | Self::Dflash(..) | Self::Eagle(..), _) => {
                Err(err("the drafter and its target run on different devices"))
            }
            _ => Ok(()),
        }
    }

    /// The per-layer feature rows a block drafter reads from the trunk
    /// instead of the final-norm hidden row (`None` for the others): ask the
    /// trunk's forward passes for exactly this ([`Hidden::Layers`]), then
    /// hand the rows to [`Drafter::observe`] as `hidden`.
    pub(crate) fn feature_rows(&self) -> Option<Hidden> {
        match self {
            Self::Dflash(net, _) => Some(Hidden::Layers(net.features())),
            _ => None,
        }
    }

    /// Whether one [`Drafter::draft`] call costs one pass whatever the
    /// number of drafts it returns (a block drafter), not one pass per
    /// draft: the cost model prices such a drafter per round.
    pub(crate) fn block(&self) -> bool {
        matches!(self, Self::Dflash(..))
    }

    /// The first `shared` tokens of the trunk's sequence stay and any later
    /// one may change (a new prompt that shares `shared` tokens with the
    /// cached ids): forget what depended on the rest, the deferred rows
    /// included. A no-op for the file drafters.
    pub(crate) fn truncate(&mut self, shared: usize) {
        match self {
            Self::InTrunk(head, lag) => {
                lag.trim_from(shared);
                head.truncate(shared);
            }
            Self::Eagle(head, lag) => {
                lag.trim_from(shared);
                head.truncate(shared);
            }
            Self::Dflash(net, lag) => {
                lag.trim_from(shared);
                net.truncate(shared);
            }
            _ => {}
        }
    }

    /// Forget everything the drafter learned from the trunk (the trunk's
    /// cache was emptied), the deferred rows included. A no-op for the file
    /// drafters.
    pub(crate) fn reset(&mut self) {
        self.truncate(0);
    }

    /// Draft at most `k` tokens after the pending token `first` of the
    /// sequence whose block tables (one per K/V group of the target) are
    /// `tables`.
    ///
    /// `hidden` is the target's final-norm hidden state at its last cached
    /// position, `len` the number of cached positions (`first` sits at
    /// position `len` and is not cached). Each step's logits go to `pick`,
    /// which chooses the token or ends the chain with `None`; the returned
    /// drafts are the tokens `pick` accepted, at most `k`. A drafter that
    /// defers its upkeep ([`Drafter::defer`]) proposes from whatever cache it
    /// holds: [`Drafter::catch_up`] first.
    ///
    /// # Errors
    ///
    /// [`CucaError::Provider`] for a forward-pass failure or a `target` on
    /// another device than the drafter.
    #[expect(
        clippy::too_many_arguments,
        reason = "one drafting call's inputs; a struct would only rename them"
    )]
    pub(crate) fn draft(
        &self,
        target: &Model,
        tables: &[BlockTable],
        hidden: &[f32],
        first: u32,
        len: usize,
        k: usize,
        pick: &mut dyn FnMut(&mut [f32]) -> Option<u32>,
    ) -> Result<Vec<u32>, CucaError> {
        #[cfg(test)]
        if let Self::InTrunk(_, lag) | Self::Eagle(_, lag) | Self::Dflash(_, lag) = self {
            lag.stale_drafts
                .set(lag.stale_drafts.get() + usize::from(!lag.is_empty()));
        }
        match (self, target) {
            (Self::Cpu { net }, Model::Cpu(target)) => {
                let ext = target.ext_kv(tables)?;
                let mut drafts = Vec::with_capacity(k);
                let mut hidden = hidden.to_vec();
                let mut token = first;
                for _ in 0..k {
                    let mut input = target.embedding_row(token)?;
                    input.extend_from_slice(&hidden);
                    let (mut logits, next_hidden) = net.draft(&ext, len, &input, len)?;
                    let Some(next) = pick(&mut logits) else { break };
                    token = next;
                    drafts.push(token);
                    hidden = next_hidden;
                }
                Ok(drafts)
            }
            (Self::InTrunk(head, _), Model::Cpu(target)) => {
                target.head_draft(head, hidden, first, len, k, pick)
            }
            (Self::Eagle(head, _), Model::Cpu(target)) => {
                target.eagle_draft(head, hidden, first, len, k, pick)
            }
            (Self::Dflash(net, _), Model::Cpu(target)) => net.draft(target, first, len, k, pick),
            #[cfg(feature = "provider-local-gpu")]
            (Self::Gpu(drafter), Model::Gpu(target)) => {
                drafter.draft(target, tables, hidden, first, len, k, pick)
            }
            #[cfg(feature = "provider-local-gpu")]
            _ => Err(err("the drafter and its target run on different devices")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `n` rows of three floats.
    fn rows(n: usize) -> Vec<f32> {
        (0..n * 3).map(|i| i as f32).collect()
    }

    /// A lag holding one row of `width` floats.
    fn lag_of_width(width: usize) -> Lag {
        let mut lag = Lag::new(2048);
        lag.push(0, &[1], &vec![0.0; width]).unwrap();
        lag
    }

    /// The held rows are contiguous, of one width, and trim from a position.
    #[test]
    fn the_lag_holds_contiguous_rows_and_trims_from_a_position() {
        let mut lag = Lag::new(2048);
        lag.push(5, &[1, 2], &rows(2)).unwrap();
        lag.push(7, &[3], &rows(1)).unwrap();
        assert_eq!((lag.len(), lag.end()), (3, 8));
        // A row that does not follow, one of another width and a call that
        // is not whole rows are refused, and leave the lag as it was.
        assert!(lag.push(9, &[4], &rows(1)).is_err());
        assert!(lag.push(8, &[4], &[0.0; 2]).is_err());
        assert!(lag.push(8, &[4, 5], &rows(1)).is_err());
        assert!(lag.push(8, &[], &[]).is_err());
        assert_eq!((lag.len(), lag.end()), (3, 8));

        lag.trim_from(9);
        assert_eq!(lag.len(), 3, "past the end: nothing goes");
        lag.trim_from(7);
        assert_eq!((lag.len(), lag.end(), lag.rows.len()), (2, 7, 6));
        // Below the first held position everything goes, and a new range
        // may start anywhere.
        lag.trim_from(3);
        assert!(lag.is_empty());
        lag.push(40, &[9], &rows(1)).unwrap();
        let (pos, tokens, held) = lag.take();
        assert_eq!((pos, tokens, held.len()), (40, vec![9], 3));
        assert!(lag.is_empty());
    }

    /// At most 1,024 rows and 16 MiB are held, whatever the row width.
    #[test]
    fn the_lag_is_bounded_by_rows_and_by_bytes() {
        // EAGLE on Mistral Small 4 and the Qwen 3.5 0.8B head: the rows bound.
        assert_eq!(lag_of_width(4096).cap(), 1024);
        assert_eq!(lag_of_width(1024).cap(), 1024);
        // DFlash on Muse Glimmer 30B: five feature rows of 6,656 floats.
        assert_eq!(lag_of_width(5 * 6656).cap(), 126);
        // A row wider than the bound still counts as one.
        assert_eq!(lag_of_width(8 << 20).cap(), 1);
    }
}
