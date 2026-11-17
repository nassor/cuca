//! The engine thread's state: model, K/V block pools and the step scheduler
//! behind every request.
//!
//! [`Engine::run_loop`] serves the jobs of the channel `LocalModel` owns on the
//! dedicated engine thread. A job is a pre-tokenized prompt plus the channel
//! its output blocks go to; the [`scheduler`] runs several at once, one
//! forward pass per step over the rows of all of them:
//!
//! 1. **Prefix reuse.** K/V lives in blocks of 16 positions named by the
//!    token ids they hold ([`KvPools`]), shared by every sequence. A finished
//!    (or preempted, or cancelled) sequence's full blocks stay in the pool
//!    (least recently used evicted first); a prompt, as soon as it is
//!    admitted, looks up the longest cached prefix of whole blocks (always
//!    leaving the last prompt token to prefill, since its logits start
//!    generation, so the trailing partial block, at most 15 tokens, is
//!    recomputed), so a conversation that appends a turn pays only for the
//!    new tokens, and so does a request that repeats the prompt of one still
//!    running. The reuse is reported as the response's [`PromptCacheUsage`]
//!    (`read_tokens` reused, `write_tokens` prefilled). A hybrid model's
//!    recurrent state is not positional: the engine keeps a snapshot of it at
//!    a block boundary of every prompt it prefills (the end of its last
//!    complete message) and resumes only from prefixes it has one for. An
//!    in-trunk head, an EAGLE, a DFlash or a DSpark drafter keeps the pairs
//!    (positions) the shared tokens still determine (`Drafter::truncate`,
//!    which also drops the rows it holds back) and re-observes the replayed
//!    rows; it keeps one such cache per state slot (`Engine::drafters`, forks
//!    of one set of weights), and a prompt takes the free slot whose cache
//!    shares the longest prefix with it.
//! 2. **Prefill** in chunks of [`prefill_chunk`] tokens, interleaved with the
//!    decode rows of the other sequences within the step's row budget; only
//!    the last chunk computes logits. An in-trunk head, an EAGLE, a DFlash or
//!    a DSpark drafter takes the hidden rows of every position
//!    (`Drafter::observe`).
//! 3. **Decode**: sample, stop on an end-of-generation id, feed the id through
//!    the [`OutputParser`], cut its text at the first stop string of the job
//!    (`StopMatcher`; thinking and tool calls pass through, and text a stop
//!    string might still begin is held back) and queue every resulting block
//!    for the stream (`try_send`: a slow consumer stalls its own sequence
//!    only), then run the model one token further (a round: draft, verify in
//!    the step's shared batch, roll back the rejected positions, hand the
//!    kept rows to the sequence's in-trunk head, EAGLE, DFlash or DSpark
//!    drafter: observed at once after a round that drafted, held back
//!    (`Drafter::defer`) after one that did not, until a round drafts
//!    again). Generation stops at the
//!    end-of-generation id, at a stop string, at `max_tokens`, or when the
//!    context is full; held text is released at every end but a stop string.
//! 4. **Usage**: exact prompt and completion token counts are recorded in the
//!    response metadata before the stream ends.
//!
//! A forward-pass error fails every running request (the shared K/V is no
//! longer trusted), resets the pools and is delivered as the stream's last
//! item; a panic is caught, reported the same way, and the engine keeps
//! serving. A request whose consumer has gone away leaves the cache
//! consistent with the ids it processed, so its work is reusable.

mod scheduler;
mod stage;
mod stop;

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use crate::provider::local::gguf;
use tokio::sync::mpsc;

pub(crate) use self::scheduler::ExportReady;
use self::scheduler::{Export, Import, Sequence, SnapshotStore, StepBufs};
use super::chat::{ChatFormat, ChatTemplate, expand_template_name};
use super::model::Model;
use super::model::arch::Arch;
use super::model::err as local_err;
use super::model::mtp::Drafter;
use super::model::paged::{KvLayout, KvPools, Sizing};
#[cfg(test)]
use super::model::prefill_chunk;
use super::model::split::LayerRange;
use super::sampler::{SplitMix64, random_seed};
use super::sampling::{FileSampling, Sampling};
use super::tokenizer::Tokenizer;
use super::{LocalDevice, LocalModelConfig, MAX_DRAFT_TOKENS};
use crate::client::ResponseMetadataHandle;
use crate::error::CucaError;
use crate::types::MessageContentBlock;

/// A [`DrafterSource`](super::DrafterSource) with its file resolved: the path
/// of the drafter GGUF, or the head inside the main model's own file.
pub(crate) enum ResolvedDrafter {
    File(PathBuf),
    InTrunk,
}

/// One generation request for the engine thread.
pub(crate) struct Job {
    /// The tokenized prompt, `0 < len < context_length`.
    pub(crate) prompt_ids: Vec<u32>,
    /// Whether the prompt ends inside an open reasoning channel.
    pub(crate) start_in_thinking: bool,
    /// Whether the request enabled thinking (reasoning tokens are reported).
    pub(crate) enable_thinking: bool,
    /// Per-request temperature override.
    pub(crate) temperature: Option<f32>,
    /// Per-request cap on completion tokens.
    pub(crate) max_tokens: Option<u32>,
    /// Per-request nucleus mass in `(0, 1]`, overriding the load-time one.
    pub(crate) top_p: Option<f32>,
    /// Per-request top-k (`0` disables it), overriding the load-time one.
    pub(crate) top_k: Option<u32>,
    /// Per-request seed: restarts the generator for this request only.
    pub(crate) seed: Option<u64>,
    /// Stop strings: generation ends before the first occurrence of any.
    pub(crate) stop: Vec<String>,
    /// Where output blocks (or the failure) go; dropping it ends the stream.
    pub(crate) events: mpsc::Sender<Result<MessageContentBlock, CucaError>>,
    /// Receives the prompt-cache and token usage of this job.
    pub(crate) metadata: ResponseMetadataHandle,
    /// When the request was dispatched: its wait for admission starts here.
    pub(crate) queued_at: std::time::Instant,
}

/// What the engine thread receives: a request, or a call to run on the
/// engine itself (the K/V transfer and stage endpoints read and change
/// engine state, which only this thread may touch). Both wait in the same
/// queue: a call behind a full queue of requests waits like one.
pub(crate) enum Msg {
    Job(Job),
    Call(Box<dyn FnOnce(&mut Engine) + Send + 'static>),
}

/// The resolved [`BatchingConfig`](super::BatchingConfig): what the scheduler
/// runs with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Batching {
    /// Sequences running at once (state slots of a hybrid model).
    pub(crate) max_sequences: usize,
    /// Rows of one forward step across all sequences.
    pub(crate) max_step_tokens: usize,
    /// Most tokens of one sequence's prompt in one step
    /// ([`super::model::prefill_rows`]).
    pub(crate) prefill_chunk: usize,
    /// Register finished and full blocks in the prefix store.
    pub(crate) prefix_cache: bool,
}

/// The model plus the state that survives between jobs.
pub(crate) struct Engine {
    model: Model,
    tokenizer: Arc<Tokenizer>,
    format: ChatFormat,
    sampling: Sampling,
    /// The K/V block pools and the prefix store over them, shared by every
    /// running sequence.
    pools: KvPools,
    context_length: usize,
    batching: Batching,
    /// Requests not admitted yet (FIFO; a preempted one is at the front).
    waiting: VecDeque<Sequence>,
    /// Admitted sequences, oldest first.
    running: Vec<Sequence>,
    /// Finished sequences whose output the consumer has not taken yet.
    draining: Vec<Sequence>,
    /// The model's free recurrent-state slots.
    free_slots: Vec<usize>,
    /// Recurrent-state snapshots of block-aligned prefixes (hybrid models).
    snapshots: SnapshotStore,
    /// The ids of the last sequence each state slot served, for a drafter
    /// that observes every position: the slot's drafter cache holds pairs
    /// for them.
    drafter_ids: Vec<Vec<u32>>,
    /// Sampling generator when no fixed seed is configured.
    rng: SplitMix64,
    /// Generator of tool-call ids.
    id_rng: SplitMix64,
    /// The multi-token-prediction drafters: none, one shared assistant
    /// (reads the target's K/V, holds no state), or one drafter per state
    /// slot (an observing drafter keeps a cache per sequence).
    drafters: Vec<Drafter>,
    /// Most tokens drafted per verification round.
    draft_tokens: usize,
    /// Speculation counters shared with `LocalModel::draft_stats`.
    stats: Arc<DraftCounters>,
    /// Scheduler counters shared with `LocalModel::serving_stats`.
    serving: Arc<ServingCounters>,
    /// Cost-aware on/off switch of speculation per concurrency class
    /// ([`spec_class`]), kept across requests.
    governor: [Governor; SPEC_CLASSES],
    /// Measured costs that set each round's draft length per concurrency
    /// class, kept across requests.
    cost: [CostModel; SPEC_CLASSES],
    /// Scratch the scheduler refills every step.
    bufs: StepBufs,
    /// Size in bytes of the GGUF file the model was read from (`0` for an
    /// engine not built by [`Engine::open`]).
    file_bytes: u64,
    /// What identifies the model's file and cache layout to a peer that
    /// transfers K/V ([`model_fingerprint`]); `0` for an engine not built by
    /// [`Engine::open`].
    fingerprint: u64,
    /// Most K/V exports (running and leased) at once; `None` when the model
    /// does not export ([`LocalModelConfig::kv_export`]).
    max_exports: Option<usize>,
    /// Prefixes whose blocks a peer is reading.
    exports: Vec<Export>,
    /// Prefixes being imported from a peer.
    imports: Vec<Import>,
    /// The id of the next export or import.
    next_transfer: u64,
    /// What the model was sized for, as a layer-stage peer must agree on it.
    sizing: Sizing,
    /// The state of a layer-stage worker ([`LocalModelConfig::stage`]);
    /// `None` for every other engine.
    stage: Option<stage::StageHost>,
    /// The ids the last sequence that ended held (empty after a failure).
    #[cfg(test)]
    last_ids: Vec<u32>,
}

/// Exponential moving average of a measurement (a throughput, a duration, a
/// count).
#[derive(Debug, Clone, Copy, Default)]
struct Ema {
    value: f64,
    /// Samples recorded, the first (a warm-up: kernel compilation, autotune)
    /// excluded.
    n: u32,
    skipped: bool,
}

impl Ema {
    const ALPHA: f64 = 0.3;

    fn record(&mut self, value: f64) {
        if !self.skipped {
            self.skipped = true;
            return;
        }
        self.value = if self.n == 0 {
            value
        } else {
            self.value + Self::ALPHA * (value - self.value)
        };
        self.n += 1;
    }
}

/// Per-round decay of the accepted and judged draft counts behind
/// [`CostModel::block_acceptance`]: about the last twenty rounds count.
const ACCEPT_DECAY: f64 = 0.95;

/// Cost model of a speculative round: verification seconds by row count,
/// drafter-step seconds, and how the drafter's confidence calibrates against
/// realised acceptance. It sets how many drafts a round verifies
/// ([`take_draft`]); the [`Governor`] decides whether a round drafts at all.
#[derive(Debug, Clone, Default)]
struct CostModel {
    /// Seconds of one verification forward over `n` rows at index `n - 1`
    /// (`n` in `1..=MAX_DRAFT_TOKENS + 1`), rollback and head upkeep included.
    verify: [Ema; MAX_DRAFT_TOKENS + 1],
    /// Seconds of one drafter step.
    draft: Ema,
    /// Predicted (the drafter's chained confidence summed over the verified
    /// drafts, uncalibrated) and realised accepted drafts per round; their
    /// ratio calibrates the confidence.
    predicted: Ema,
    accepted: Ema,
    /// Drafts accepted and drafts judged (an accepted one or the first
    /// rejected one of a round), summed over recent rounds with a decay of
    /// [`ACCEPT_DECAY`] per round.
    accept_hits: f64,
    accept_trials: f64,
    /// Test-only: record nothing, so every proposal is drafted.
    #[cfg(test)]
    frozen: bool,
}

impl CostModel {
    /// Estimated seconds of one verification over `n` rows (`n >= 1`).
    ///
    /// A measured count returns its average; otherwise the nearest measured
    /// counts are interpolated linearly, extrapolated upward from the two
    /// largest with a slope floored at 0, or held flat from the nearest one.
    /// With nothing measured every count costs the same (`1.0`), so the
    /// first rounds draft the full cap and fill the table.
    fn verify_secs(&self, n: usize) -> f64 {
        let i = n.saturating_sub(1).min(MAX_DRAFT_TOKENS);
        let measured = |j: usize| (self.verify[j].n > 0).then_some(self.verify[j].value);
        if let Some(v) = measured(i) {
            return v;
        }
        let below = (0..i).rev().find_map(|j| measured(j).map(|v| (j, v)));
        let above = (i + 1..=MAX_DRAFT_TOKENS).find_map(|j| measured(j).map(|v| (j, v)));
        match (below, above) {
            (Some((a, va)), Some((b, vb))) => va + (vb - va) * (i - a) as f64 / (b - a) as f64,
            (Some((b, vb)), None) => match (0..b).rev().find_map(|j| measured(j).map(|v| (j, v))) {
                Some((a, va)) => vb + ((vb - va) / (b - a) as f64).max(0.0) * (i - b) as f64,
                None => vb,
            },
            (None, Some((_, vb))) => vb,
            (None, None) => 1.0,
        }
    }

    /// Estimated seconds of one drafter step (0 before any measurement).
    fn draft_secs(&self) -> f64 {
        self.draft.value
    }

    /// Factor turning the drafter's chained confidence into a predicted
    /// acceptance: realised over predicted accepted drafts per round.
    fn calibration(&self) -> f64 {
        if self.predicted.n == 0 || self.predicted.value <= 0.0 {
            1.0
        } else {
            (self.accepted.value / self.predicted.value).clamp(0.25, 4.0)
        }
    }

    /// Predicted acceptance of draft `j` (1-based) of a block drafter's
    /// round: `rho^j`, `rho` the observed fraction of judged drafts that were
    /// accepted (1 before any round, so the first rounds draft the full cap).
    /// A block drafter's own confidence is no guide: its mask rows are
    /// deliberately flat while the drafts are right nine times in ten.
    fn block_acceptance(&self, j: usize) -> f64 {
        let rho = if self.accept_trials > 0.0 {
            self.accept_hits / self.accept_trials
        } else {
            1.0
        };
        rho.powi(i32::try_from(j).unwrap_or(i32::MAX))
    }

    /// Record a finished round: a verification over `rows` rows took
    /// `verify_secs`; `draft_steps` drafter steps took `draft_secs`; the
    /// verified drafts carried `predicted` summed confidence and `accepted`
    /// of them were accepted.
    fn record_round(
        &mut self,
        rows: usize,
        verify_secs: f64,
        draft_steps: usize,
        draft_secs: f64,
        predicted: f64,
        accepted: usize,
    ) {
        #[cfg(test)]
        if self.frozen {
            return;
        }
        if let Some(slot) = self.verify.get_mut(rows.wrapping_sub(1)) {
            slot.record(verify_secs);
        }
        if draft_steps > 0 {
            self.draft.record(draft_secs / draft_steps as f64);
        }
        // Only a round that verified drafts says how the confidence
        // calibrates.
        if rows > 1 {
            self.predicted.record(predicted);
            self.accepted.record(accepted as f64);
            // A draft behind a rejected one is never judged: the trials of
            // a round are the accepted drafts plus the first rejected one.
            let drafts = rows - 1;
            self.accept_hits = self.accept_hits * ACCEPT_DECAY + accepted as f64;
            self.accept_trials =
                self.accept_trials * ACCEPT_DECAY + (accepted + 1).min(drafts) as f64;
        }
    }
}

#[cfg(test)]
impl CostModel {
    /// A cost model that never learns: verification costs the same for
    /// every row count and drafting is free, so every round drafts the full
    /// cap whatever the drafter's confidence and the tiny model's timings.
    fn always_drafting() -> Self {
        Self {
            frozen: true,
            ..Self::default()
        }
    }
}

/// Whether draft `j` (1-based) is worth verifying: `a_j` is its predicted
/// acceptance, `e_prev` the expected tokens of the round without it
/// (`1 + a_1 + .. + a_{j-1}`), `t_j` and `t_next` the verification seconds
/// over `j` and `j + 1` rows and `d` the seconds of one drafter step. True
/// when adding the draft raises the round's expected tokens per second:
/// `(e_prev + a_j) / (t_next + j d) > e_prev / (t_j + (j - 1) d)`.
fn take_draft(a_j: f64, e_prev: f64, j: usize, t_j: f64, t_next: f64, d: f64) -> bool {
    a_j * (t_j + (j - 1) as f64 * d) > e_prev * (t_next - t_j + d)
}

/// [`take_draft`] for a block drafter, whose forward pass costs `f` seconds
/// per round however many drafts the round keeps: the expected-tokens-per-
/// second comparison `(e_prev + a_j) / (t_next + f) > e_prev / (t_j + f)`
/// reduces to `a_j (t_j + f) > e_prev (t_next - t_j)`. Adding a draft then
/// costs only the extra verified row.
fn take_block_draft(a_j: f64, e_prev: f64, t_j: f64, t_next: f64, f: f64) -> bool {
    a_j * (t_j + f) > e_prev * (t_next - t_j)
}

/// Decides, round by round, whether to draft: speculation runs only while its
/// measured throughput (tokens per second of a draft-plus-verify round,
/// acceptance and device costs included) is at least `MARGIN` (2 %) above
/// that of plain decoding, so the worst case is plain decoding plus the cost
/// of occasional probes.
///
/// It measures plain decoding first (two rounds after a discarded warm-up),
/// then speculation. While plain decoding is faster it re-probes with two
/// speculative rounds after `probe_every` plain rounds, doubling the interval
/// (up to `MAX_PROBE_EVERY`) after every failed probe, so a text that never
/// benefits costs about `2 / probe_every` of the speculative round cost; while
/// speculation is faster it refreshes the plain estimate with two plain rounds
/// every `REFRESH_EVERY` rounds.
#[derive(Debug, Clone, Default)]
struct Governor {
    plain: Ema,
    spec: Ema,
    /// Rounds left in a forced probe or refresh, and the mode it forces.
    forced: u32,
    forced_spec: bool,
    /// Rounds since the last probe (plain mode) or refresh (speculative mode).
    since: u32,
    probe_every: u32,
}

impl Governor {
    const MIN_SAMPLES: u32 = 2;
    const PROBE_EVERY: u32 = 32;
    const MAX_PROBE_EVERY: u32 = 512;
    const REFRESH_EVERY: u32 = 64;
    /// Speculation must beat plain decoding by this factor to stay on.
    const MARGIN: f64 = 1.02;
    /// Slowdown factor beyond which an early speculative round counts as a
    /// compile, not a measurement.
    const OUTLIER: f64 = 6.0;

    /// Whether the next round should draft.
    fn want_spec(&mut self) -> bool {
        if self.forced > 0 {
            self.forced -= 1;
            return self.forced_spec;
        }
        if self.plain.n < Self::MIN_SAMPLES + u32::from(!self.plain.skipped) {
            return false;
        }
        if self.spec.n < Self::MIN_SAMPLES {
            return true;
        }
        let interval = self.probe_every.max(Self::PROBE_EVERY);
        self.since += 1;
        if self.spec.value >= self.plain.value * Self::MARGIN {
            if self.since >= Self::REFRESH_EVERY {
                self.since = 0;
                self.forced = 1;
                self.forced_spec = false;
                return false;
            }
            self.probe_every = 0;
            true
        } else if self.since >= interval {
            self.since = 0;
            self.forced = 1;
            self.forced_spec = true;
            true
        } else {
            false
        }
    }

    /// Record a finished round: `tokens` advanced in `secs`.
    fn record(&mut self, spec: bool, tokens: usize, secs: f64) {
        if secs <= 0.0 {
            return;
        }
        let tps = tokens as f64 / secs;
        let was_probe = spec && self.spec.value < self.plain.value * Self::MARGIN;
        if spec {
            // A round more than `OUTLIER` times slower than plain decoding is
            // a first-shape compile (GPU autotune, shader build), not a cost
            // the next rounds will pay: keep it out of the estimate.
            if self.plain.n >= Self::MIN_SAMPLES
                && tps * Self::OUTLIER < self.plain.value
                && self.spec.n < Self::MIN_SAMPLES
            {
                return;
            }
            self.spec.record(tps);
        } else {
            self.plain.record(tps);
        }
        // A probe that fails doubles the interval before the next one.
        if was_probe
            && self.forced == 0
            && self.spec.n >= Self::MIN_SAMPLES
            && self.spec.value < self.plain.value * Self::MARGIN
        {
            self.probe_every =
                (self.probe_every.max(Self::PROBE_EVERY) * 2).min(Self::MAX_PROBE_EVERY);
        }
    }
}

#[cfg(test)]
impl Governor {
    /// A governor that keeps speculating (as if speculation measured far
    /// faster than plain decoding), so tests exercise the drafting path
    /// whatever the tiny model's timings.
    fn always_speculating() -> Self {
        Self {
            plain: Ema {
                value: 1.0,
                n: Self::MIN_SAMPLES,
                skipped: true,
            },
            spec: Ema {
                value: f64::MAX,
                n: Self::MIN_SAMPLES,
                skipped: true,
            },
            ..Self::default()
        }
    }
}

/// Concurrency classes of the speculation [`Governor`] and [`CostModel`]: a
/// round's cost and the benefit of drafting depend on how many sequences
/// share the step, so each class measures its own.
pub(super) const SPEC_CLASSES: usize = 7;

/// The class of a step with `decoding` decoding sequences: 1, 2, 3 to 4, 5 to
/// 8, 9 to 16, 17 to 32, and 33 or more (at most 64) sequences.
pub(super) fn spec_class(decoding: usize) -> usize {
    ((usize::BITS - decoding.max(1).saturating_sub(1).leading_zeros()) as usize)
        .min(SPEC_CLASSES - 1)
}

/// Speculation counters, updated by the engine thread and read by
/// `LocalModel::draft_stats`.
#[derive(Debug, Default)]
pub(crate) struct DraftCounters {
    rounds: AtomicU64,
    drafted: AtomicU64,
    accepted: AtomicU64,
}

impl DraftCounters {
    pub(crate) fn snapshot(&self) -> super::DraftStats {
        super::DraftStats {
            rounds: self.rounds.load(Ordering::Relaxed),
            drafted: self.drafted.load(Ordering::Relaxed),
            accepted: self.accepted.load(Ordering::Relaxed),
        }
    }
}

/// Scheduler counters, shared with `LocalModel::serving_stats`: gauges the
/// engine refreshes every turn, and totals.
#[derive(Debug, Default)]
pub(crate) struct ServingCounters {
    running: AtomicUsize,
    waiting: AtomicUsize,
    free_blocks: AtomicUsize,
    cached_blocks: AtomicUsize,
    used_blocks: AtomicUsize,
    total_blocks: AtomicUsize,
    max_sequences: AtomicUsize,
    snapshot_bytes: AtomicU64,
    prefix_hit_tokens: AtomicU64,
    preemptions: AtomicU64,
    steps: AtomicU64,
    batched_rows: AtomicU64,
    prefill_rows: AtomicU64,
    generated_tokens: AtomicU64,
    admissions: AtomicU64,
    queue_wait_ns: AtomicU64,
    step_ns: AtomicU64,
    last_step_ns: AtomicU64,
    last_step_rows: AtomicUsize,
}

impl ServingCounters {
    pub(crate) fn snapshot(&self) -> super::ServingStats {
        super::ServingStats {
            running: self.running.load(Ordering::Relaxed),
            waiting: self.waiting.load(Ordering::Relaxed),
            max_sequences: self.max_sequences.load(Ordering::Relaxed),
            free_blocks: self.free_blocks.load(Ordering::Relaxed),
            cached_blocks: self.cached_blocks.load(Ordering::Relaxed),
            used_blocks: self.used_blocks.load(Ordering::Relaxed),
            total_blocks: self.total_blocks.load(Ordering::Relaxed),
            snapshot_bytes: self.snapshot_bytes.load(Ordering::Relaxed),
            prefix_hit_tokens: self.prefix_hit_tokens.load(Ordering::Relaxed),
            preemptions: self.preemptions.load(Ordering::Relaxed),
            steps: self.steps.load(Ordering::Relaxed),
            batched_rows: self.batched_rows.load(Ordering::Relaxed),
            prefill_rows: self.prefill_rows.load(Ordering::Relaxed),
            generated_tokens: self.generated_tokens.load(Ordering::Relaxed),
            admissions: self.admissions.load(Ordering::Relaxed),
            queue_wait_ns: self.queue_wait_ns.load(Ordering::Relaxed),
            step_ns: self.step_ns.load(Ordering::Relaxed),
            last_step_ns: self.last_step_ns.load(Ordering::Relaxed),
            last_step_rows: self.last_step_rows.load(Ordering::Relaxed),
        }
    }
}

impl Engine {
    /// Read the GGUF file at `path` and build the engine, the tokenizer and
    /// the chat template from it.
    ///
    /// The model lands on `config.device`: candle on the CPU, or burn on a
    /// GPU. Either way it is owned by the calling (engine) thread for its
    /// whole life; burn's device streams are per thread, so every forward
    /// pass runs on the thread that loaded the weights.
    ///
    /// `drafter` selects the multi-token-prediction drafter: a separate
    /// file, or the head inside `path` itself ([`ResolvedDrafter`]).
    ///
    /// # Errors
    ///
    /// [`CucaError::Io`] when the file cannot be opened or read,
    /// [`CucaError::Config`] when `config.context_length` exceeds the model
    /// maximum or `config.device` is compiled out, and
    /// [`CucaError::Provider`] for a file that is not a GGUF file (or split
    /// set) the reader can parse, an unsupported checkpoint (wrong architecture, a
    /// weight dtype the GPU engine does not read), a missing tensor or
    /// metadata key, a drafter that does not match the model (an in-trunk
    /// head the file does not carry, or one asked for on a GPU), or an
    /// unusable tokenizer. `chat_template`, when given, replaces the GGUF's
    /// template for everything derived from it; a template that does not
    /// compile or render is a [`CucaError::Config`].
    pub(crate) fn open(
        path: &Path,
        drafter: Option<&ResolvedDrafter>,
        stats: Arc<DraftCounters>,
        config: &LocalModelConfig,
        chat_template: Option<&str>,
    ) -> Result<(Self, Arc<Tokenizer>, ChatTemplate, ChatFormat), CucaError> {
        let content = open_gguf(path, &config.shard_dirs)?;

        let hp = super::model::read_hparams(&content)?;
        let context_length = super::resolve_context_length(config.context_length, hp.n_ctx_train);
        // A separate drafter file is indexed before the model loads: its
        // architecture says which drafter to open, and an EAGLE drafter runs
        // on the CPU only.
        let file_drafter = match drafter {
            Some(ResolvedDrafter::File(dp)) => Some(open_gguf(dp, &config.shard_dirs)?),
            _ => None,
        };
        let file_arch = file_drafter.as_ref().map_or("", architecture_of);
        if file_arch == Arch::Mistral4Eagle.name() && config.device != LocalDevice::Cpu {
            return Err(CucaError::Config(format!(
                "the EAGLE drafter runs on the CPU only, not on {:?}",
                config.device
            )));
        }
        stage::check_stages(&hp, config, file_arch)?;
        // Tiered experts make a prefill step's cost the experts it touches, so
        // a model that can run big steps does (`prefill_rows`).
        let tiered = matches!(config.experts, super::ExpertPlacement::Tiered(_));
        let chunk = super::model::prefill_rows(&hp, context_length, tiered);
        let (batching, sizing) = resolve_batching_with(config, context_length, chunk)?;
        let file_bytes = content
            .files()
            .iter()
            .map(|f| std::fs::metadata(&f.path).map_or(0, |m| m.len()))
            .sum();
        let fingerprint_of = |tokenizer: &Tokenizer| {
            model_fingerprint(
                &content,
                file_bytes,
                hp.n_layer,
                tokenizer.vocab_size(),
                &KvLayout::sized(&hp, context_length, &sizing),
            )
        };
        // A coordinator names the file to its workers (and a worker checks
        // it) before any weights load; every other model reads the
        // tokenizer after its drafter, as it always did.
        let staged = config.stage.is_some() || !config.remote_stages.is_empty();
        let early_tokenizer = if staged {
            Some(Arc::new(Tokenizer::from_gguf(&content.metadata)?))
        } else {
            None
        };
        let early_fingerprint = early_tokenizer.as_deref().map_or(0, &fingerprint_of);
        let model = match (&config.stage, config.remote_stages.is_empty()) {
            (Some(stage), _) => Model::load_stage(
                &content,
                context_length,
                config.device,
                &config.experts,
                &sizing,
                &config.engram,
                LayerRange {
                    first: stage.first,
                    end: stage.first + stage.layers,
                },
            )?,
            (None, false) => Model::load_chain(
                &content,
                context_length,
                config.device,
                &config.experts,
                &sizing,
                &config.engram,
                &config.remote_stages,
                drafter.is_some(),
                early_fingerprint,
            )?,
            (None, true) => Model::load(
                &content,
                context_length,
                config.device,
                &config.shards,
                &config.experts,
                &sizing,
                config.gpu_layers,
                &config.engram,
            )?,
        };
        let drafter = drafter
            .map(|source| match (source, &file_drafter) {
                (ResolvedDrafter::File(_), Some(dc)) => match file_arch {
                    super::model::cpu::dflash::ARCH
                        if super::model::cpu::dspark::is_dspark(&dc.metadata) =>
                    {
                        Drafter::open_dspark(dc, &model, context_length)
                    }
                    super::model::cpu::dflash::ARCH => {
                        Drafter::open_dflash(dc, &model, context_length)
                    }
                    a if a == Arch::Mistral4Eagle.name() => Drafter::open_eagle(dc, &model),
                    _ => Drafter::open(dc, &model, context_length),
                },
                (ResolvedDrafter::File(dp), None) => Err(local_err(format!(
                    "the drafter file `{}` was not indexed",
                    dp.display()
                ))),
                (ResolvedDrafter::InTrunk, _) => Drafter::open_in_trunk(&content, &model),
            })
            .transpose()?;
        let drafters = match drafter {
            None => Vec::new(),
            Some(d) => drafters_per_slot(d, batching.max_sequences)?,
        };
        let tokenizer = match early_tokenizer {
            Some(tokenizer) => tokenizer,
            None => Arc::new(Tokenizer::from_gguf(&content.metadata)?),
        };
        let fingerprint = if staged {
            early_fingerprint
        } else {
            fingerprint_of(&tokenizer)
        };
        let (format, template) = load_chat(&content, &tokenizer, chat_template)?;

        let mut engine = Self::new(
            model,
            Arc::clone(&tokenizer),
            format.clone(),
            Sampling::resolve(
                &config.sampling,
                &FileSampling::from_metadata(&content.metadata),
            ),
            context_length,
            batching,
            drafters,
            config.draft_tokens,
            stats,
        );
        engine.file_bytes = file_bytes;
        engine.fingerprint = fingerprint;
        engine.sizing = sizing;
        engine.stage = config.stage.map(|s| {
            stage::StageHost::new(
                LayerRange {
                    first: s.first,
                    end: s.first + s.layers,
                },
                sizing.max_sequences,
            )
        });
        engine.max_exports = config.kv_export.as_ref().map(|c| c.max_exports);
        Ok((engine, tokenizer, template, format))
    }

    /// An engine over a loaded `model` with nothing queued.
    #[expect(
        clippy::too_many_arguments,
        reason = "the engine's parts; a struct would only rename them"
    )]
    fn new(
        model: Model,
        tokenizer: Arc<Tokenizer>,
        format: ChatFormat,
        sampling: Sampling,
        context_length: usize,
        batching: Batching,
        drafters: Vec<Drafter>,
        draft_tokens: usize,
        stats: Arc<DraftCounters>,
    ) -> Self {
        let pools = KvPools::new(model.kv_layout());
        let serving = Arc::<ServingCounters>::default();
        serving
            .max_sequences
            .store(batching.max_sequences, Ordering::Relaxed);
        let engine = Self {
            model,
            tokenizer,
            format,
            sampling,
            pools,
            context_length,
            batching,
            waiting: VecDeque::new(),
            running: Vec::new(),
            draining: Vec::new(),
            free_slots: (0..batching.max_sequences).rev().collect(),
            snapshots: SnapshotStore::new(batching.max_sequences + 4),
            drafter_ids: vec![Vec::new(); batching.max_sequences],
            rng: SplitMix64(random_seed()),
            id_rng: SplitMix64(random_seed()),
            drafters,
            draft_tokens,
            bufs: StepBufs::default(),
            file_bytes: 0,
            fingerprint: 0,
            max_exports: None,
            exports: Vec::new(),
            imports: Vec::new(),
            next_transfer: 0,
            sizing: Sizing::single(context_length),
            stage: None,
            stats,
            serving,
            governor: std::array::from_fn(|_| Governor::default()),
            cost: std::array::from_fn(|_| CostModel::default()),
            #[cfg(test)]
            last_ids: Vec::new(),
        };
        // The pool sizes are known before the first request.
        engine.publish_gauges();
        engine
    }

    /// The scheduler counters `LocalModel::serving_stats` reads.
    pub(crate) fn serving_counters(&self) -> Arc<ServingCounters> {
        Arc::clone(&self.serving)
    }

    /// The load-time sampling parameters after the config/file/built-in
    /// precedence.
    pub(crate) fn sampling(&self) -> Sampling {
        self.sampling
    }

    /// The context length the engine runs with: the configured value, or the
    /// default resolved against the model.
    pub(crate) fn context_length(&self) -> usize {
        self.context_length
    }

    /// Size in bytes of the GGUF file the model was read from.
    pub(crate) fn file_bytes(&self) -> u64 {
        self.file_bytes
    }

    /// K/V cache bytes of the loaded model at its context length.
    pub(crate) fn kv_cache_bytes(&self) -> u64 {
        self.model.kv_pool_bytes()
    }

    /// Layers on the GPU and in all when layers are offloaded.
    pub(crate) fn layer_split(&self) -> Option<(usize, usize)> {
        self.model.layer_split()
    }

    /// Counters and heat of the expert tiers; `None` unless the model loaded
    /// with `ExpertPlacement::Tiered`.
    pub(crate) fn expert_tiers(&self) -> Option<Arc<super::model::tier::TierShared>> {
        self.model.expert_tiers()
    }

    /// The counters of the Engram table's row cache; `None` for a model
    /// without an Engram table.
    pub(crate) fn engram_counters(&self) -> Option<Arc<super::model::engram_io::EngramCounters>> {
        self.model.engram_counters()
    }

    /// The call counters of the remote stages (empty unless this engine
    /// coordinates a chain), with their names.
    pub(crate) fn stage_counters(&self) -> Vec<(String, Arc<super::model::chain::StageCounters>)> {
        self.model.stage_counters()
    }

    /// What identifies this model's file and cache layout to a peer, see
    /// [`model_fingerprint`].
    pub(crate) fn fingerprint(&self) -> u64 {
        self.fingerprint
    }
}

/// The index of the GGUF file (or split set) at `path`, whose missing parts
/// are also looked for in `extra_dirs`.
fn open_gguf(path: &Path, extra_dirs: &[PathBuf]) -> Result<gguf::Content, CucaError> {
    gguf::Content::open_with(path, extra_dirs).map_err(|e| match e {
        // A read failure of the file itself; anything else is a header the
        // reader cannot interpret.
        candle_core::Error::Io(_) => CucaError::Io {
            message: format!("reading GGUF `{}`: {e}", path.display()),
        },
        other => local_err(format!(
            "`{}` is not a supported GGUF file: {other}",
            path.display()
        )),
    })
}

/// The chat format and template of a loaded model: those of `chat_template`
/// when given, else the ones the GGUF embeds.
fn load_chat(
    content: &gguf::Content,
    tokenizer: &Tokenizer,
    chat_template: Option<&str>,
) -> Result<(ChatFormat, ChatTemplate), CucaError> {
    const TEMPLATE_KEY: &str = "tokenizer.chat_template";
    let bos = tokenizer.bos_token().to_string();
    let eos = tokenizer.eos_token().to_string();
    match chat_template {
        Some(source) => {
            let source = expand_template_name(source)
                .map_err(|e| CucaError::Config(format!("chat_template override: {e}")))?;
            let template = ChatTemplate::checked(source.to_string(), bos, eos)?;
            Ok((
                ChatFormat::detect(&content.metadata, tokenizer, &source)?,
                template,
            ))
        }
        None => {
            let source = content
                .metadata
                .get(TEMPLATE_KEY)
                .ok_or_else(|| local_err(format!("GGUF metadata `{TEMPLATE_KEY}`: missing")))?
                .to_string()
                .map_err(|e| local_err(format!("GGUF metadata `{TEMPLATE_KEY}`: {e}")))?
                .clone();
            // llama.cpp's converter can write a template *name*.
            let source = expand_template_name(&source)
                .map_err(|e| local_err(format!("GGUF metadata `{TEMPLATE_KEY}`: {e}")))?
                .into_owned();
            let format = ChatFormat::detect(&content.metadata, tokenizer, &source)?;
            Ok((format, ChatTemplate::new(source, bos, eos)?))
        }
    }
}

/// Resolve `config.batching` against the model's context length: the
/// scheduler's limits and the sizing of the K/V pools and state slots, with
/// the default prefill chunk of the context length (tests).
///
/// # Errors
///
/// [`CucaError::Config`] for a `kv_tokens` below one context plus a block.
#[cfg(test)]
fn resolve_batching(
    config: &LocalModelConfig,
    context_length: usize,
) -> Result<(Batching, Sizing), CucaError> {
    resolve_batching_with(config, context_length, prefill_chunk(context_length))
}

/// Positions beyond one context that the default K/V pool of a model on a GPU
/// holds for cached prefixes. On the CPU the default is twice the context;
/// device memory is scarcer than RAM, and at the default 8192 positions the
/// second context would be 1.6 GB of f32 K/V for Qwen3-30B-A3B, the
/// difference between fitting a 16 GB card beside its 10 GB of weights and
/// not.
const GPU_PREFIX_SLACK: usize = 4096;

/// The K/V pool size of `batching.kv_tokens = None` for a model of
/// `context_length` positions on `device`: twice the context on the CPU, a
/// context plus at most [`GPU_PREFIX_SLACK`] positions on a GPU.
fn default_kv_tokens(context_length: usize, device: LocalDevice) -> usize {
    if device == LocalDevice::Cpu {
        2 * context_length
    } else {
        context_length + context_length.min(GPU_PREFIX_SLACK)
    }
}

/// [`resolve_batching`] for a model that prefills `chunk` tokens of one prompt
/// per step ([`super::model::prefill_rows`]).
///
/// # Errors
///
/// [`CucaError::Config`] for a `kv_tokens` below one context plus a block.
fn resolve_batching_with(
    config: &LocalModelConfig,
    context_length: usize,
    chunk: usize,
) -> Result<(Batching, Sizing), CucaError> {
    use super::model::paged::BLOCK_SIZE;
    let b = &config.batching;
    let kv_tokens = match b.kv_tokens {
        None => default_kv_tokens(context_length, config.device),
        Some(n) if n >= context_length + BLOCK_SIZE => n.div_ceil(BLOCK_SIZE) * BLOCK_SIZE,
        Some(n) => {
            return Err(CucaError::Config(format!(
                "batching.kv_tokens must be at least context_length + {BLOCK_SIZE} ({}), got {n}",
                context_length + BLOCK_SIZE
            )));
        }
    };
    let max_sequences = b.max_sequences;
    let max_step_tokens = b
        .max_step_tokens
        .unwrap_or_else(|| chunk.max(max_sequences));
    let sizing = Sizing {
        max_sequences,
        kv_tokens,
        max_rows: max_step_tokens
            .min(chunk)
            .max(super::model::cpu::MAX_VERIFY_ROWS),
    };
    let batching = Batching {
        max_sequences,
        max_step_tokens,
        prefill_chunk: chunk,
        prefix_cache: b.prefix_cache,
    };
    Ok((batching, sizing))
}

/// The `general.architecture` of a GGUF file (empty when absent).
fn architecture_of(content: &gguf::Content) -> &str {
    content
        .metadata
        .get("general.architecture")
        .and_then(|v| v.to_string().ok())
        .map_or("", String::as_str)
}

/// FNV-1a over 64 bits.
struct Fnv(u64);

impl Fnv {
    const BASIS: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0100_0000_01b3;

    fn new() -> Self {
        Self(Self::BASIS)
    }

    fn bytes(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0 ^ u64::from(b)).wrapping_mul(Self::PRIME);
        }
    }

    fn u64(&mut self, v: u64) {
        self.bytes(&v.to_le_bytes());
    }

    /// A string with its length, so neighbouring strings cannot run together.
    fn str(&mut self, s: &str) {
        self.u64(s.len() as u64);
        self.bytes(s.as_bytes());
    }
}

/// What two nodes must share to exchange K/V rows: a 64-bit FNV-1a hash of
/// the wire version, the architecture, the bytes of the model files, the
/// layer count, the vocabulary size, every tensor's name, shape and type
/// (sorted by name) and every K/V group of the cache layout (window, stride,
/// kind, layers). The pool sizes are left out: nodes may size `kv_tokens`
/// differently.
fn model_fingerprint(
    content: &gguf::Content,
    file_bytes: u64,
    n_layer: usize,
    vocab: usize,
    layout: &super::model::paged::KvLayout,
) -> u64 {
    let mut h = Fnv::new();
    h.u64(u64::from(super::wire::WIRE_VERSION));
    h.str(architecture_of(content));
    h.u64(file_bytes);
    h.u64(n_layer as u64);
    h.u64(vocab as u64);
    let mut names: Vec<&String> = content.tensor_infos.keys().collect();
    names.sort();
    for name in names {
        let info = &content.tensor_infos[name];
        h.str(name);
        for &d in info.shape.dims() {
            h.u64(d as u64);
        }
        h.str(&format!("{:?}", info.dtype));
    }
    for g in &layout.groups {
        h.u64(g.window.map_or(0, |w| w as u64 + 1));
        h.u64(g.stride as u64);
        h.u64(u64::from(g.compressed));
        h.u64(g.layers.len() as u64);
        for &l in &g.layers {
            h.u64(l as u64);
        }
    }
    h.0
}

/// One drafter per state slot: an assistant reads the target's K/V and holds
/// no state, so one serves every slot; a drafter that observes every position
/// keeps a cache per sequence, so `max_sequences` of them (`drafter` and its
/// forks, which share its weights) serve the slots by index.
///
/// # Errors
///
/// [`CucaError::Provider`] when a fork cannot allocate its cache.
fn drafters_per_slot(drafter: Drafter, max_sequences: usize) -> Result<Vec<Drafter>, CucaError> {
    if !drafter.observes() {
        return Ok(vec![drafter]);
    }
    let forks = (1..max_sequences)
        .map(|_| drafter.fork())
        .collect::<Result<Vec<_>, _>>()?;
    let mut all = Vec::with_capacity(max_sequences);
    all.push(drafter);
    all.extend(forks);
    Ok(all)
}

fn saturating_u32(n: usize) -> u32 {
    u32::try_from(n).unwrap_or(u32::MAX)
}

fn saturating_u64(n: u128) -> u64 {
    u64::try_from(n).unwrap_or(u64::MAX)
}

#[cfg(test)]
impl Engine {
    /// Install `drafter` the way [`Engine::open`] does: one shared
    /// assistant, or one drafter per state slot.
    fn set_drafter(&mut self, drafter: Drafter) {
        self.drafters = drafters_per_slot(drafter, self.batching.max_sequences).unwrap();
    }

    /// Make every concurrency class draft the full cap whatever the tiny
    /// drafter's confidence and the tiny model's timings.
    fn force_speculation(&mut self) {
        self.governor = std::array::from_fn(|_| Governor::always_speculating());
        self.cost = std::array::from_fn(|_| CostModel::always_drafting());
    }

    /// An engine that exports up to `max` prefixes at once, as a configured
    /// `kv_export` does.
    pub(crate) fn with_exports(mut self, max: usize) -> Self {
        self.max_exports = Some(max);
        self
    }
}

/// Fixtures shared by the engine tests and the `LocalModel` tests.
#[cfg(test)]
pub(crate) mod testing {
    use super::*;
    use crate::provider::local::model::chain::{StageCounters, StageTransport};
    use crate::provider::local::model::tiny::{TinyModel, VOCAB, load_sized};

    /// Vocabulary of the tiny model: ids 1 (`<eos>`), 3 (`<turn|>`) and 4
    /// (`<|tool_response>`) end generation; 9..16 are one-letter pieces.
    pub(crate) fn tiny_tokenizer() -> Tokenizer {
        let mut pieces: Vec<String> = [
            "<pad>",
            "<eos>",
            "<bos>",
            "<turn|>",
            "<|tool_response>",
            "<|channel>",
            "<channel|>",
            "<|tool_call>",
            "<tool_call|>",
        ]
        .iter()
        .map(|s| (*s).to_string())
        .collect();
        pieces.extend(["a", "b", "c", "d", "e", "f", "g"].map(String::from));
        assert_eq!(pieces.len(), VOCAB);
        let types = (0..VOCAB).map(|i| if i < 9 { 3 } else { 1 }).collect();
        Tokenizer::new(pieces, types, &[]).unwrap()
    }

    /// The output format of the tiny vocabulary: ids 1, 3 and 4 end generation,
    /// with a reasoning channel and the custom tool-call syntax.
    pub(crate) fn tiny_format(tokenizer: &Tokenizer) -> ChatFormat {
        ChatFormat::detect(
            &std::collections::HashMap::from([(
                "tokenizer.ggml.eos_token_id".to_string(),
                gguf::Value::U32(1),
            )]),
            tokenizer,
            "<|channel>thought\n <|tool_call>call:f{}<tool_call|>",
        )
        .unwrap()
    }

    /// An engine over the tiny model, with greedy or seeded sampling.
    pub(crate) fn engine(
        model: &TinyModel,
        context_length: usize,
        temperature: f32,
        seed: Option<u64>,
    ) -> Engine {
        engine_sized(model, context_length, temperature, seed, 1, None)
    }

    /// [`engine`] serving `max_sequences` sequences over `kv_tokens` pooled
    /// positions (`None`: twice the context), with a step of `max_step`
    /// rows (`None`: the default).
    pub(crate) fn engine_sized(
        model: &TinyModel,
        context_length: usize,
        temperature: f32,
        seed: Option<u64>,
        max_sequences: usize,
        kv_tokens: Option<usize>,
    ) -> Engine {
        engine_with(
            model,
            context_length,
            temperature,
            seed,
            max_sequences,
            kv_tokens,
            None,
        )
    }

    /// [`engine_sized`] with an explicit step size.
    pub(crate) fn engine_with(
        model: &TinyModel,
        context_length: usize,
        temperature: f32,
        seed: Option<u64>,
        max_sequences: usize,
        kv_tokens: Option<usize>,
        max_step: Option<usize>,
    ) -> Engine {
        let (sizing, max_step_tokens) =
            tiny_sizing(context_length, max_sequences, kv_tokens, max_step);
        let model = Model::Cpu(load_sized(model, context_length, &sizing));
        engine_over(
            model,
            context_length,
            temperature,
            seed,
            sizing,
            max_step_tokens,
        )
    }

    /// The sizing an engine over the tiny model gets, and its step size.
    pub(crate) fn tiny_sizing(
        context_length: usize,
        max_sequences: usize,
        kv_tokens: Option<usize>,
        max_step: Option<usize>,
    ) -> (Sizing, usize) {
        let max_step_tokens =
            max_step.unwrap_or_else(|| prefill_chunk(context_length).max(max_sequences));
        let sizing = Sizing {
            max_sequences,
            kv_tokens: kv_tokens.unwrap_or(2 * context_length),
            max_rows: max_step_tokens
                .min(prefill_chunk(context_length))
                .max(crate::provider::local::model::cpu::MAX_VERIFY_ROWS),
        };
        (sizing, max_step_tokens)
    }

    /// An engine over the loaded `model`, which was sized with `sizing`.
    pub(crate) fn engine_over(
        model: Model,
        context_length: usize,
        temperature: f32,
        seed: Option<u64>,
        sizing: Sizing,
        max_step_tokens: usize,
    ) -> Engine {
        let tokenizer = Arc::new(tiny_tokenizer());
        let mut e = Engine::new(
            model,
            Arc::clone(&tokenizer),
            tiny_format(&tokenizer),
            Sampling {
                temperature,
                top_k: 64,
                top_p: 0.95,
                seed,
            },
            context_length,
            Batching {
                max_sequences: sizing.max_sequences,
                max_step_tokens,
                prefill_chunk: prefill_chunk(context_length),
                prefix_cache: true,
            },
            Vec::new(),
            0,
            Arc::default(),
        );
        e.sizing = sizing;
        e
    }

    /// A layer-stage worker over the layers `range` of the tiny model,
    /// sized as [`engine_sized`] sizes `max_sequences` sequences.
    pub(crate) fn stage_worker(
        file: &TinyModel,
        context_length: usize,
        range: LayerRange,
        max_sequences: usize,
    ) -> Engine {
        let (sizing, max_step_tokens) = tiny_sizing(context_length, max_sequences, None, None);
        let content = gguf::Content::open(&file.path).unwrap();
        let model = Model::load_stage(
            &content,
            context_length,
            LocalDevice::Cpu,
            &super::super::ExpertPlacement::Resident,
            &sizing,
            &super::super::EngramCache::default(),
            range,
        )
        .unwrap();
        let mut e = engine_over(model, context_length, 0.0, None, sizing, max_step_tokens);
        e.stage = Some(stage::StageHost::new(range, max_sequences));
        e
    }

    /// An engine whose first layers run on the workers behind `links`
    /// (consecutive ranges from layer 0, in order) and whose last layers run
    /// here.
    pub(crate) fn chain_engine(
        file: &TinyModel,
        context_length: usize,
        links: Vec<(LayerRange, Box<dyn StageTransport>)>,
        temperature: f32,
        seed: Option<u64>,
        max_sequences: usize,
    ) -> Engine {
        let (sizing, max_step_tokens) = tiny_sizing(context_length, max_sequences, None, None);
        let content = gguf::Content::open(&file.path).unwrap();
        let model = Model::chain_of(
            &content,
            context_length,
            LocalDevice::Cpu,
            &super::super::ExpertPlacement::Resident,
            &sizing,
            &super::super::EngramCache::default(),
            links,
            true,
            0,
        )
        .unwrap();
        engine_over(
            model,
            context_length,
            temperature,
            seed,
            sizing,
            max_step_tokens,
        )
    }

    /// [`chain_engine`] drafting with the Gemma 4 assistant in `drafter`,
    /// `draft_tokens` per round, and a governor and cost model that always
    /// draft the full cap.
    pub(crate) fn chain_engine_with_drafter(
        file: &TinyModel,
        drafter: &TinyModel,
        context_length: usize,
        links: Vec<(LayerRange, Box<dyn StageTransport>)>,
        draft_tokens: usize,
    ) -> Engine {
        let mut e = chain_engine(file, context_length, links, 0.0, None, 1);
        let content = gguf::Content::open(&drafter.path).unwrap();
        e.set_drafter(Drafter::open(&content, &e.model, context_length).unwrap());
        e.draft_tokens = draft_tokens;
        e.force_speculation();
        e
    }

    /// A frame the worker ran and what it answered, shared with a test.
    pub(crate) type KeptCall = Arc<std::sync::Mutex<Option<(Vec<u8>, Vec<u8>)>>>;

    /// A transport that answers with a worker engine of this process. It
    /// records the op of every frame and fails the call number `fail_at`
    /// (counting from 0) once; at the call number `restart_at` the worker is
    /// first opened by another session, as if it restarted (what it held is
    /// gone), and the call then meets it. At the call number `lose_reply_at`
    /// (or at every call, with `lose_every_reply`) the worker runs the frame
    /// and its answer is lost on the way, so the transport sends the same
    /// frame again, once, as `HttpTransport` does.
    pub(crate) struct Loopback {
        pub(crate) worker: Arc<std::sync::Mutex<Engine>>,
        /// The op byte of every frame sent, in order, a frame sent again
        /// counted again.
        pub(crate) ops: Arc<std::sync::Mutex<Vec<u8>>>,
        /// Calls made so far.
        pub(crate) calls: Arc<AtomicUsize>,
        /// The call to fail; `usize::MAX` for none.
        pub(crate) fail_at: Arc<AtomicUsize>,
        /// The call after which the worker has restarted; `usize::MAX` for
        /// none.
        pub(crate) restart_at: Arc<AtomicUsize>,
        /// The call whose first answer is lost; `usize::MAX` for none.
        pub(crate) lose_reply_at: Arc<AtomicUsize>,
        /// Lose the first answer of every call.
        pub(crate) lose_every_reply: Arc<std::sync::atomic::AtomicBool>,
        /// The last frame the worker ran and what it answered, and the same
        /// for the last forward frame.
        pub(crate) last: KeptCall,
        pub(crate) last_forward: KeptCall,
        /// How many restores, snapshots and Engram prefetches the forward
        /// frames asked for, one count per call (a frame sent again does not
        /// count twice).
        pub(crate) asked: Arc<std::sync::Mutex<(usize, usize, usize)>>,
        stats: Arc<StageCounters>,
        /// The number type of the carries (`F32` unless a test sets it).
        pub(crate) carry: super::super::CarryDtype,
    }

    impl Loopback {
        pub(crate) fn new(worker: Engine) -> Self {
            Self {
                worker: Arc::new(std::sync::Mutex::new(worker)),
                ops: Arc::default(),
                calls: Arc::default(),
                fail_at: Arc::new(AtomicUsize::new(usize::MAX)),
                restart_at: Arc::new(AtomicUsize::new(usize::MAX)),
                lose_reply_at: Arc::new(AtomicUsize::new(usize::MAX)),
                lose_every_reply: Arc::default(),
                last: Arc::default(),
                last_forward: Arc::default(),
                asked: Arc::default(),
                stats: Arc::default(),
                carry: super::super::CarryDtype::F32,
            }
        }
    }

    impl StageTransport for Loopback {
        fn call(&self, frame: Vec<u8>) -> Result<Vec<u8>, CucaError> {
            if let Ok(crate::provider::local::stageframe::Call::Forward(call)) =
                crate::provider::local::stageframe::decode(&frame)
            {
                let mut asked = self
                    .asked
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                for s in &call.seqs {
                    asked.0 += usize::from(s.restore.is_some());
                    asked.1 += usize::from(s.snapshot.is_some());
                    asked.2 += usize::from(!s.prefetch.is_empty());
                }
            }
            let n = self.calls.fetch_add(1, Ordering::SeqCst);
            self.ops
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push(frame.first().copied().unwrap_or(0));
            if n == self.fail_at.load(Ordering::SeqCst) {
                self.stats.record_error(frame.len());
                return Err(local_err("stage loopback: the connection dropped"));
            }
            let mut worker = self
                .worker
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if n == self.restart_at.load(Ordering::SeqCst) {
                let open = worker.foreign_open(0x5EED);
                worker.stage_call(&open)?;
            }
            let started = std::time::Instant::now();
            let mut result = worker.stage_call(&frame);
            let lost = n == self.lose_reply_at.load(Ordering::SeqCst)
                || self.lose_every_reply.load(Ordering::SeqCst);
            if lost && result.is_ok() {
                self.stats.record_retry(frame.len());
                self.ops
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .push(frame.first().copied().unwrap_or(0));
                result = worker.stage_call(&frame);
            }
            let reply = result.inspect_err(|_| {
                self.stats.record_error(frame.len());
            })?;
            let wall = started.elapsed();
            self.stats.record(wall, wall, frame.len(), reply.len());
            let kept = Some((frame.clone(), reply.to_vec()));
            if frame.first() == Some(&crate::provider::local::stageframe::OP_FORWARD) {
                *self
                    .last_forward
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner) = kept.clone();
            }
            *self
                .last
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = kept;
            Ok(reply.to_vec())
        }

        fn name(&self) -> String {
            "stage loopback".to_owned()
        }

        fn counters(&self) -> Arc<StageCounters> {
            Arc::clone(&self.stats)
        }

        fn carry(&self) -> super::super::CarryDtype {
            self.carry
        }
    }

    /// [`engine`] with a drafter proposing `draft_tokens` per round, over
    /// `max_sequences` slots.
    pub(crate) fn engine_with_drafter(
        model: &TinyModel,
        drafter: &TinyModel,
        context_length: usize,
        temperature: f32,
        seed: Option<u64>,
        draft_tokens: usize,
        max_sequences: usize,
    ) -> Engine {
        let mut e = engine_sized(
            model,
            context_length,
            temperature,
            seed,
            max_sequences,
            None,
        );
        let content = gguf::Content::open(&drafter.path).unwrap();
        e.set_drafter(Drafter::open(&content, &e.model, context_length).unwrap());
        e.draft_tokens = draft_tokens;
        e
    }

    /// [`engine`] drafting with the prediction head inside `model`'s own
    /// file (a tiny `qwen35`), `draft_tokens` per round, and a governor and
    /// cost model that always draft the full cap (the tiny head's
    /// confidence would end most chains early).
    pub(crate) fn engine_in_trunk(
        model: &TinyModel,
        context_length: usize,
        temperature: f32,
        seed: Option<u64>,
        draft_tokens: usize,
        max_sequences: usize,
    ) -> Engine {
        let mut e = engine_sized(
            model,
            context_length,
            temperature,
            seed,
            max_sequences,
            None,
        );
        let content = gguf::Content::open(&model.path).unwrap();
        e.set_drafter(Drafter::open_in_trunk(&content, &e.model).unwrap());
        e.draft_tokens = draft_tokens;
        e.force_speculation();
        e
    }

    /// [`engine`] drafting with the EAGLE drafter in the file `eagle` for the
    /// tiny `mistral4` `model`, `draft_tokens` per round, and a governor and
    /// cost model that always draft the full cap.
    pub(crate) fn engine_with_eagle(
        model: &TinyModel,
        eagle: &TinyModel,
        context_length: usize,
        temperature: f32,
        seed: Option<u64>,
        draft_tokens: usize,
        max_sequences: usize,
    ) -> Engine {
        let mut e = engine_sized(
            model,
            context_length,
            temperature,
            seed,
            max_sequences,
            None,
        );
        let content = gguf::Content::open(&eagle.path).unwrap();
        e.set_drafter(Drafter::open_eagle(&content, &e.model).unwrap());
        e.draft_tokens = draft_tokens;
        e.force_speculation();
        e
    }

    /// [`engine`] drafting with a DFlash block drafter, `draft_tokens` per
    /// round, and a governor and cost model that always draft the full cap
    /// (the tiny drafter's confidence would end most chains early).
    pub(crate) fn engine_with_dflash(
        model: &TinyModel,
        drafter: &TinyModel,
        context_length: usize,
        temperature: f32,
        seed: Option<u64>,
        draft_tokens: usize,
        max_sequences: usize,
    ) -> Engine {
        let mut e = engine_sized(
            model,
            context_length,
            temperature,
            seed,
            max_sequences,
            None,
        );
        let content = gguf::Content::open(&drafter.path).unwrap();
        e.set_drafter(Drafter::open_dflash(&content, &e.model, context_length).unwrap());
        e.draft_tokens = draft_tokens;
        e.force_speculation();
        e
    }

    /// [`engine`] drafting with a DSpark block drafter, `draft_tokens` per
    /// round, and a governor and cost model that always draft the full cap.
    pub(crate) fn engine_with_dspark(
        model: &TinyModel,
        drafter: &TinyModel,
        context_length: usize,
        temperature: f32,
        seed: Option<u64>,
        draft_tokens: usize,
        max_sequences: usize,
    ) -> Engine {
        let mut e = engine_sized(
            model,
            context_length,
            temperature,
            seed,
            max_sequences,
            None,
        );
        let content = gguf::Content::open(&drafter.path).unwrap();
        e.set_drafter(Drafter::open_dspark(&content, &e.model, context_length).unwrap());
        e.draft_tokens = draft_tokens;
        e.force_speculation();
        e
    }

    /// A tiny model file (the Gemma 4 fixture, a context of 32) for a test
    /// outside this tier that serves it through [`tiny_local_model`].
    #[cfg(feature = "server")]
    pub(crate) fn tiny_file() -> TinyModel {
        crate::provider::local::model::tiny::write_tiny_model()
    }

    /// A template that renders the first message's content after `<bos>`.
    pub(crate) const TINY_TEMPLATE: &str = "{{ bos_token }}{{ messages[0]['content'] }}";

    /// An engine over the tiny model file at `path`, which stays the
    /// caller's: the file is not deleted when the engine goes. Build it on
    /// the engine thread, from the path.
    pub(crate) fn engine_at(path: &Path, context_length: usize) -> Engine {
        let tiny = TinyModel {
            path: path.to_path_buf(),
        };
        let engine = engine(&tiny, context_length, 0.0, None);
        // `tiny` would delete the file on drop; the caller still owns it.
        std::mem::forget(tiny);
        engine
    }

    /// [`stage_worker`] over the tiny model file at `path`, which stays the
    /// caller's (build it on the engine thread, from the path).
    pub(crate) fn stage_worker_at(
        path: &Path,
        context_length: usize,
        range: LayerRange,
        max_sequences: usize,
    ) -> Engine {
        let tiny = TinyModel {
            path: path.to_path_buf(),
        };
        let engine = stage_worker(&tiny, context_length, range, max_sequences);
        // `tiny` would delete the file on drop; the caller still owns it.
        std::mem::forget(tiny);
        engine
    }

    /// A tiny Llama model file with `context` positions, for the server's
    /// layer-stage tests (any cut of it is legal).
    #[cfg(feature = "server")]
    pub(crate) fn tiny_llama_file(context: usize) -> TinyModel {
        use crate::provider::local::model::tiny::{TinySpec, write_spec};
        write_spec(&TinySpec {
            context,
            ..TinySpec::llama()
        })
    }

    /// [`stage_worker_at`] over the layers `first..end`.
    #[cfg(feature = "server")]
    pub(crate) fn stage_worker_layers_at(
        path: &Path,
        context_length: usize,
        first: usize,
        end: usize,
    ) -> Engine {
        stage_worker_at(path, context_length, LayerRange { first, end }, 1)
    }

    /// A chain engine over the tiny model file at `path` whose remote stages
    /// are the workers `stages` name, each reached over HTTP through the
    /// production transport, holding `layers` layers in order from layer 0.
    /// Build it on the engine thread, which the transport's runtime needs.
    #[cfg(feature = "server")]
    pub(crate) fn http_chain_at(
        path: &Path,
        context_length: usize,
        stages: &[super::super::RemoteStage],
    ) -> Engine {
        use crate::provider::local::model::chain::HttpTransport;
        let mut first = 0;
        let links: Vec<(LayerRange, Box<dyn StageTransport>)> = stages
            .iter()
            .map(|stage| {
                let range = LayerRange {
                    first,
                    end: first + stage.layers,
                };
                first = range.end;
                let transport = HttpTransport::new(stage).expect("the transport builds");
                (range, Box::new(transport) as Box<dyn StageTransport>)
            })
            .collect();
        let tiny = TinyModel {
            path: path.to_path_buf(),
        };
        let engine = chain_engine(&tiny, context_length, links, 0.0, None, 1);
        // `tiny` would delete the file on drop; the caller still owns it.
        std::mem::forget(tiny);
        engine
    }

    /// A ready [`LocalModel`] over the engine `build` makes from the tiny
    /// model file at `path` (on the engine thread), whose context is
    /// `context_length`, with the tiny tokenizer and [`TINY_TEMPLATE`].
    pub(crate) async fn tiny_local_model(
        path: PathBuf,
        context_length: usize,
        build: impl FnOnce(&Path) -> Engine + Send + 'static,
    ) -> super::super::LocalModel {
        super::super::LocalModel::start(Arc::default(), move || {
            let tokenizer = Arc::new(tiny_tokenizer());
            let template = ChatTemplate::new(
                TINY_TEMPLATE.to_string(),
                "<bos>".to_string(),
                "<eos>".to_string(),
            )?;
            let engine = build(&path);
            assert_eq!(
                engine.context_length(),
                context_length,
                "the engine's context"
            );
            let format = tiny_format(&tokenizer);
            Ok((engine, tokenizer, template, format))
        })
        .await
        .expect("tiny engine starts")
    }

    /// A job with every optional setting at its default.
    pub(crate) fn job(
        prompt: &[u32],
        max_tokens: Option<u32>,
        events: mpsc::Sender<Result<MessageContentBlock, CucaError>>,
        metadata: ResponseMetadataHandle,
    ) -> Job {
        Job {
            prompt_ids: prompt.to_vec(),
            start_in_thinking: false,
            enable_thinking: false,
            temperature: None,
            max_tokens,
            top_p: None,
            top_k: None,
            seed: None,
            stop: Vec::new(),
            events,
            metadata,
            queued_at: std::time::Instant::now(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::testing::{
        engine, engine_in_trunk, engine_sized, engine_with_dflash, engine_with_drafter,
        engine_with_dspark, engine_with_eagle,
    };
    use super::*;
    use crate::provider::local::model::tiny::{
        TinyDeepseek41, write_deepseek41, write_tiny_drafter, write_tiny_model,
    };
    use crate::provider::local::model::tiny::{TinyDflash, write_tiny_dflash};
    use crate::provider::local::model::tiny::{TinyQwen35, TinySpec, write_qwen35, write_spec};
    use crate::provider::local::stageframe;
    use crate::request::PromptCacheUsage;

    struct Outcome {
        items: Vec<Result<MessageContentBlock, CucaError>>,
        metadata: crate::client::ResponseMetadata,
    }

    fn run(e: &mut Engine, prompt: &[u32], max_tokens: Option<u32>) -> Outcome {
        run_with(e, prompt, max_tokens, |_| {})
    }

    /// [`run`] with `edit` applied to the job first (sampling overrides, stop
    /// strings).
    fn run_with(
        e: &mut Engine,
        prompt: &[u32],
        max_tokens: Option<u32>,
        edit: impl FnOnce(&mut Job),
    ) -> Outcome {
        let (tx, mut rx) = mpsc::channel(1024);
        let metadata = ResponseMetadataHandle::empty();
        let mut job = super::testing::job(prompt, max_tokens, tx, metadata.clone());
        edit(&mut job);
        e.run(job);
        let mut items = Vec::new();
        while let Some(item) = rx.blocking_recv() {
            items.push(item);
        }
        Outcome {
            items,
            metadata: metadata.take(),
        }
    }

    /// The text of every `Text` block, concatenated.
    fn text(outcome: &Outcome) -> String {
        outcome
            .items
            .iter()
            .filter_map(|i| match i {
                Ok(MessageContentBlock::Text(t)) => Some(t.as_str()),
                _ => None,
            })
            .collect()
    }

    /// The completion tokens a request reported.
    fn completion_tokens(out: &Outcome) -> Option<u32> {
        out.metadata.token_usage.map(|u| u.completion_tokens)
    }

    /// Speculation must not change what a request replies: the same text and
    /// the same completion tokens.
    fn assert_same_reply(got: &Outcome, expected: &Outcome, what: &str) {
        assert_eq!(text(got), text(expected), "{what}");
        assert_eq!(
            completion_tokens(got),
            completion_tokens(expected),
            "{what}"
        );
    }

    /// A fixed seed reproduces a sampled reply, its usage and the draft
    /// counters, and every round of it drafts.
    fn seeded_run_is_reproducible(make: impl Fn() -> Engine) {
        let run_once = || {
            let mut e = make();
            let out = run(&mut e, &PROMPT, None);
            (text(&out), completion_tokens(&out), e.stats.snapshot())
        };
        let (a, n, stats_a) = run_once();
        let (b, m, stats_b) = run_once();
        assert_eq!((a, n, stats_a), (b, m, stats_b));
        assert!(n.is_some());
        assert!(stats_a.drafted > 0, "a seeded run always drafts");
    }

    const PROMPT: [u32; 6] = [2, 9, 10, 11, 12, 13];

    #[test]
    fn reports_exact_usage_and_respects_max_tokens() {
        let model = write_tiny_model();
        let mut e = engine(&model, 32, 0.0, None);
        let out = run(&mut e, &PROMPT, Some(5));
        // Random weights may emit a malformed tool call; that is a parse
        // error item, not an engine failure, and usage is still reported.
        let usage = out.metadata.token_usage.expect("usage reported");
        assert_eq!(usage.prompt_tokens, 6);
        assert!(usage.completion_tokens <= 5);
        assert_eq!(usage.reasoning_tokens, None);
        assert_eq!(
            out.metadata.prompt_cache_usage,
            Some(PromptCacheUsage {
                read_tokens: 0,
                write_tokens: 6
            })
        );
    }

    #[test]
    fn context_limit_caps_the_completion() {
        let model = write_tiny_model();
        let mut e = engine(&model, 8, 0.0, None);
        let out = run(&mut e, &PROMPT, None);
        let usage = out.metadata.token_usage.expect("usage reported");
        // 8 positions, 6 of them prompt.
        assert!(usage.completion_tokens <= 2, "{usage:?}");
    }

    #[test]
    fn prefix_reuse_reads_the_shared_prefix() {
        let model = write_spec(&TinySpec {
            context: 96,
            ..TinySpec::llama()
        });
        let mut e = engine(&model, 96, 0.0, None);
        let prompt = long_prompt(40);
        run(&mut e, &prompt, Some(3));

        // Same prompt again: whole blocks below the last token are reused,
        // 32 of 40 (the trailing partial block is computed again).
        let again = run(&mut e, &prompt, Some(3));
        assert_eq!(
            again.metadata.prompt_cache_usage,
            Some(PromptCacheUsage {
                read_tokens: 32,
                write_tokens: 8
            })
        );

        // Longer prompt: the old prompt's two full blocks are reused.
        let mut longer = prompt.clone();
        longer.extend([14, 15]);
        let out = run(&mut e, &longer, Some(3));
        let cache = out.metadata.prompt_cache_usage.unwrap();
        assert_eq!(cache.read_tokens, 32);
        assert_eq!(cache.write_tokens, 10);

        // Diverging prompt: only the whole blocks of the common head (20
        // tokens: one block) are reused.
        let mut diverging = prompt[..20].to_vec();
        diverging.extend([15, 14, 15, 14, 15, 14]);
        let out = run(&mut e, &diverging, Some(3));
        let cache = out.metadata.prompt_cache_usage.unwrap();
        assert_eq!(cache.read_tokens, 16);
        assert_eq!(cache.write_tokens, 10);

        // Shorter than a block: nothing to reuse.
        let out = run(&mut e, &prompt[..10], Some(3));
        assert_eq!(out.metadata.prompt_cache_usage.unwrap().read_tokens, 0);
    }

    /// A hyper-connection model with compressed attention keeps its
    /// compressed rows in blocks that span 32 tokens: a prefix is reused in
    /// multiples of 32, and the reply after a hit equals the cold one.
    #[test]
    fn a_compressed_attention_model_reuses_prefixes_in_whole_compressed_blocks() {
        let model = write_deepseek41(&TinyDeepseek41::default());
        let mut e = engine(&model, 96, 0.0, None);
        let prompt = long_prompt(70);
        let cold = run(&mut e, &prompt, Some(6));
        let again = run(&mut e, &prompt, Some(6));
        assert_eq!(
            again.metadata.prompt_cache_usage,
            Some(PromptCacheUsage {
                read_tokens: 64,
                write_tokens: 6
            })
        );
        assert_eq!(text(&again), text(&cold));
        assert!(!text(&cold).is_empty());

        // A prompt that shares 40 tokens reuses the 32 the compressed blocks
        // cover, and replies as a cold engine does.
        let mut diverging = prompt[..40].to_vec();
        diverging.extend([15, 14, 15, 14, 15, 14, 15]);
        let warm = run(&mut e, &diverging, Some(6));
        assert_eq!(warm.metadata.prompt_cache_usage.unwrap().read_tokens, 32);
        let mut fresh = engine(&model, 96, 0.0, None);
        assert_eq!(text(&run(&mut fresh, &diverging, Some(6))), text(&warm));
    }

    #[test]
    fn a_stop_string_cuts_the_reply_and_usage_is_still_reported() {
        let model = write_tiny_model();
        let mut e = engine(&model, 32, 0.0, None);
        let plain = text(&run(&mut e, &PROMPT, Some(8)));
        assert!(plain.len() >= 3, "tiny model reply too short: {plain:?}");

        // Stop at the second character of the reply: what precedes its first
        // occurrence is all that is emitted.
        let stop = plain[1..2].to_string();
        let cut = &plain[..plain.find(&stop).unwrap()];
        let out = run_with(&mut e, &PROMPT, Some(8), |job| {
            job.stop = vec![stop.clone()]
        });
        assert_eq!(text(&out), cut);
        assert!(out.metadata.token_usage.is_some());

        // Stop strings that never occur change nothing, and the held tail is
        // flushed at the end.
        let out = run_with(&mut e, &PROMPT, Some(8), |job| {
            job.stop = vec!["zzz".to_string(), format!("{}!", &plain[plain.len() - 1..])];
        });
        assert_eq!(text(&out), plain);
    }

    #[test]
    fn a_request_seed_restarts_the_generator_for_that_request() {
        let model = write_tiny_model();
        let mut e = engine(&model, 32, 1.0, None);
        let first = text(&run_with(&mut e, &PROMPT, Some(6), |job| {
            job.seed = Some(7)
        }));
        // Another request in between moves the engine's own generator.
        run(&mut e, &PROMPT, Some(6));
        let second = text(&run_with(&mut e, &PROMPT, Some(6), |job| {
            job.seed = Some(7)
        }));
        assert_eq!(first, second);
    }

    #[test]
    fn request_top_k_one_is_greedy() {
        let model = write_tiny_model();
        let mut greedy = engine(&model, 32, 0.0, None);
        let want = text(&run(&mut greedy, &PROMPT, Some(6)));
        let mut e = engine(&model, 32, 1.0, None);
        let got = text(&run_with(&mut e, &PROMPT, Some(6), |job| {
            job.top_k = Some(1)
        }));
        assert_eq!(got, want);
    }

    #[test]
    fn warm_cache_generates_what_a_cold_engine_generates() {
        let model = write_tiny_model();
        let mut warm = engine(&model, 32, 0.0, None);
        let cold = run(&mut engine(&model, 32, 0.0, None), &PROMPT, Some(4));
        // An unrelated job in between, then the same prompt: the cache is
        // truncated to the shared head and re-prefilled.
        run(&mut warm, &[2, 15, 14, 13, 12, 11, 10], Some(4));
        let rerun = run(&mut warm, &PROMPT, Some(4));
        assert_eq!(text(&cold), text(&rerun));
        assert_eq!(
            cold.metadata.token_usage.unwrap().completion_tokens,
            rerun.metadata.token_usage.unwrap().completion_tokens
        );
    }

    #[test]
    fn fixed_seed_reproduces_sampling() {
        let model = write_tiny_model();
        let a = run(&mut engine(&model, 32, 1.0, Some(5)), &PROMPT, Some(6));
        let b = run(&mut engine(&model, 32, 1.0, Some(5)), &PROMPT, Some(6));
        assert_eq!(text(&a), text(&b));
        assert_eq!(
            a.metadata.token_usage.unwrap().completion_tokens,
            b.metadata.token_usage.unwrap().completion_tokens
        );
    }

    #[test]
    fn forward_failure_is_reported_and_resets_the_cache() {
        let model = write_tiny_model();
        let mut e = engine(&model, 32, 0.0, None);
        run(&mut e, &PROMPT, Some(2));
        // Token 99 is outside the 16-entry vocabulary.
        let out = run(&mut e, &[2, 9, 99], Some(2));
        assert_eq!(out.items.len(), 1);
        assert!(matches!(&out.items[0], Err(CucaError::Provider { .. })));
        assert!(e.last_ids.is_empty());
        // The engine still serves the next job from a cold cache.
        let ok = run(&mut e, &PROMPT, Some(2));
        assert!(ok.metadata.token_usage.is_some());
        assert_eq!(ok.metadata.prompt_cache_usage.unwrap().read_tokens, 0);
    }

    /// `n` token ids of the tiny vocabulary, none of them the end-of-turn id.
    fn long_prompt(n: usize) -> Vec<u32> {
        (0..n).map(|i| 9 + (i * 3 % 7) as u32).collect()
    }

    /// Blocks some sequence still references, over every pool.
    fn used_blocks(e: &Engine) -> usize {
        e.pools.stats().iter().map(|s| s.used).sum()
    }

    #[test]
    fn dropped_consumer_stops_generation_without_corrupting_state() {
        // 40 tokens (no end-of-turn id among them): two full blocks.
        let long = long_prompt(40);
        let model = write_spec(&TinySpec {
            context: 96,
            ..TinySpec::llama()
        });
        let mut e = engine(&model, 96, 0.0, None);
        let (tx, mut rx) = mpsc::channel(1);
        e.submit(super::testing::job(
            &long,
            Some(10),
            tx,
            ResponseMetadataHandle::empty(),
        ));
        // The consumer takes the first block and goes away.
        while !e.idle() {
            e.turn();
            if rx.try_recv().is_ok() {
                break;
            }
        }
        drop(rx);
        e.run_until_idle();
        // The sequence ended with its job: its full blocks are in the prefix
        // store and nothing is referenced; the next job reads them back.
        assert_eq!(used_blocks(&e), 0);
        let out = run(&mut e, &long, Some(2));
        assert!(out.metadata.token_usage.is_some());
        // Whole blocks only: 39 shared positions read 32.
        assert_eq!(out.metadata.prompt_cache_usage.unwrap().read_tokens, 32);
    }

    #[test]
    fn rejects_prompts_that_do_not_fit() {
        let model = write_tiny_model();
        let mut e = engine(&model, 8, 0.0, None);
        let out = run(&mut e, &[2, 9, 10, 11, 12, 13, 14, 15], Some(1));
        assert_eq!(out.items.len(), 1);
        assert!(matches!(&out.items[0], Err(CucaError::Provider { .. })));
        let out = run(&mut e, &[], Some(1));
        assert!(matches!(&out.items[0], Err(CucaError::Provider { .. })));
    }

    /// A GGUF header declaring one tensor of type 31 (a removed repacked
    /// type the engine does not know): the header opens, and the tensor that is asked for
    /// names its type instead of the whole file failing to parse.
    #[test]
    fn an_unknown_tensor_dtype_opens_and_names_the_tensor_on_use() {
        let mut bytes = b"GGUF".to_vec();
        bytes.extend(3u32.to_le_bytes()); // version
        bytes.extend(1u64.to_le_bytes()); // tensors
        bytes.extend(0u64.to_le_bytes()); // metadata entries
        bytes.extend(1u64.to_le_bytes()); // name length
        bytes.push(b'x');
        bytes.extend(1u32.to_le_bytes()); // dimensions
        bytes.extend(32u64.to_le_bytes());
        bytes.extend(31u32.to_le_bytes()); // removed Q4_0_4_4
        bytes.extend(0u64.to_le_bytes()); // offset
        bytes.resize(bytes.len().next_multiple_of(32) + 64, 0);
        let path = std::env::temp_dir().join(format!("cuca-iq4-{}.gguf", std::process::id()));
        std::fs::write(&path, &bytes).unwrap();

        let content = open_gguf(&path, &[]).unwrap();
        let msg = content.reader().bytes("x").unwrap_err().to_string();
        assert!(msg.contains("`x`") && msg.contains("ggml type 31"), "{msg}");
        let config = LocalModelConfig::default();
        let result = Engine::open(&path, None, Arc::default(), &config, None);
        let _ = std::fs::remove_file(&path);
        assert!(
            matches!(result, Err(CucaError::Provider { .. })),
            "a file without a model must not load"
        );
    }

    /// Simulated rounds: plain costs 1 s per token; a speculative round costs
    /// `spec_cost` s and yields `spec_tokens` tokens. Returns (tokens, seconds,
    /// speculative rounds).
    fn simulate(spec_tokens: usize, spec_cost: f64, rounds: usize) -> (usize, f64, usize) {
        let mut g = Governor::default();
        let (mut tokens, mut secs, mut spec_rounds) = (0, 0.0, 0);
        for _ in 0..rounds {
            if g.want_spec() {
                g.record(true, spec_tokens, spec_cost);
                tokens += spec_tokens;
                secs += spec_cost;
                spec_rounds += 1;
            } else {
                g.record(false, 1, 1.0);
                tokens += 1;
                secs += 1.0;
            }
        }
        (tokens, secs, spec_rounds)
    }

    #[test]
    fn governor_keeps_profitable_speculation_on() {
        let (tokens, secs, spec_rounds) = simulate(3, 1.5, 1000);
        assert!(spec_rounds > 900, "{spec_rounds}");
        assert!(tokens as f64 / secs > 1.8, "{}", tokens as f64 / secs);
    }

    #[test]
    fn governor_bounds_the_cost_of_unprofitable_speculation() {
        // 1.4 tokens for 2.5 s: speculation would run at 0.56x.
        let (tokens, secs, spec_rounds) = simulate(1, 2.5, 3000);
        let speed = tokens as f64 / secs;
        assert!(
            speed > 0.96,
            "worst case stays near plain decoding: {speed}"
        );
        assert!(spec_rounds < 120, "{spec_rounds}");
    }

    #[test]
    fn governor_recovers_when_the_text_becomes_predictable() {
        let mut g = Governor::default();
        let mut spec_late = 0;
        for round in 0..3000 {
            // Unpredictable for 1500 rounds, then code-like.
            let (tokens, cost) = if round < 1500 { (1, 2.5) } else { (4, 1.6) };
            if g.want_spec() {
                g.record(true, tokens, cost);
                if round >= 2000 {
                    spec_late += 1;
                }
            } else {
                g.record(false, 1, 1.0);
            }
        }
        assert!(spec_late > 800, "{spec_late}");
    }

    /// The verification-cost table answers every row count: flat before any
    /// measurement, from the one measured count, then interpolated between
    /// and extrapolated beyond the measured counts (never downward).
    #[test]
    fn cost_model_interpolates_and_bootstraps() {
        let close = |a: f64, b: f64| (a - b).abs() < 1e-12;
        let mut c = CostModel::default();
        for n in 1..=MAX_DRAFT_TOKENS + 1 {
            assert_eq!(c.verify_secs(n), 1.0);
        }
        // The first sample of a row count is a warm-up and is skipped.
        c.record_round(1, 0.5, 0, 0.0, 0.0, 0);
        assert_eq!(c.verify_secs(3), 1.0);
        c.record_round(1, 0.010, 0, 0.0, 0.0, 0);
        for n in 1..=MAX_DRAFT_TOKENS + 1 {
            assert!(close(c.verify_secs(n), 0.010), "{n}");
        }
        let mut rising = c.clone();
        rising.record_round(4, 0.5, 0, 0.0, 0.0, 0);
        rising.record_round(4, 0.016, 0, 0.0, 0.0, 0);
        assert!(close(rising.verify_secs(2), 0.012));
        assert!(close(rising.verify_secs(6), 0.020));
        let mut falling = c.clone();
        falling.record_round(4, 0.5, 0, 0.0, 0.0, 0);
        falling.record_round(4, 0.008, 0, 0.0, 0.0, 0);
        assert!(close(falling.verify_secs(6), 0.008));

        // Calibration: realised over predicted accepted drafts, bounded.
        assert_eq!(c.calibration(), 1.0);
        c.record_round(3, 0.1, 2, 0.002, 2.0, 1);
        c.record_round(3, 0.1, 2, 0.002, 2.0, 1);
        assert!(close(c.calibration(), 0.5));
        assert!(close(c.draft_secs(), 0.001));
        let mut eager = CostModel::default();
        eager.record_round(2, 0.1, 1, 0.001, 0.01, 1);
        eager.record_round(2, 0.1, 1, 0.001, 0.01, 1);
        assert_eq!(eager.calibration(), 4.0);
    }

    /// A block drafter's draft `j` is predicted to be accepted with the
    /// observed per-draft acceptance to the power `j`; a draft behind a
    /// rejected one is not judged, and before any round every draft counts
    /// as accepted (the first rounds draft the full cap).
    #[test]
    fn block_acceptance_follows_the_judged_drafts() {
        let close = |a: f64, b: f64| (a - b).abs() < 1e-12;
        let mut c = CostModel::default();
        assert_eq!(c.block_acceptance(7), 1.0);
        // Three drafts, all accepted: three judged, three hits.
        c.record_round(4, 0.1, 1, 0.01, 0.0, 3);
        assert!(close(c.block_acceptance(1), 1.0));
        // Seven drafts, the first rejected: one judged (the second to
        // seventh are not), no hit.
        c.record_round(8, 0.1, 1, 0.01, 0.0, 0);
        let rho = 3.0 * ACCEPT_DECAY / (3.0 * ACCEPT_DECAY + 1.0);
        assert!(close(c.block_acceptance(1), rho));
        assert!(close(c.block_acceptance(3), rho.powi(3)));
        // Two drafts, one accepted: two judged, one hit.
        c.record_round(3, 0.1, 1, 0.01, 0.0, 1);
        let hits = 3.0 * ACCEPT_DECAY * ACCEPT_DECAY + ACCEPT_DECAY * 0.0 + 1.0;
        let trials = 3.0 * ACCEPT_DECAY * ACCEPT_DECAY + ACCEPT_DECAY + 2.0;
        assert!(close(c.block_acceptance(2), (hits / trials).powi(2)));
        // A plain round (one row) says nothing about drafts.
        let before = c.block_acceptance(2);
        c.record_round(1, 0.1, 0, 0.0, 0.0, 0);
        assert_eq!(c.block_acceptance(2), before);
    }

    /// With an affine verification cost the chain stops at the first draft
    /// whose chained confidence no longer pays for its row.
    #[test]
    fn draft_rule_stops_when_confidence_drops() {
        let t = |n: usize| 0.010 + 0.002 * (n - 1) as f64;
        let d = 0.001;
        let (mut chained, mut expected, mut taken) = (1.0, 1.0, 0);
        for (i, q) in [0.9, 0.9, 0.2, 0.9].into_iter().enumerate() {
            let j = i + 1;
            chained *= q;
            let a = f64::min(chained, 1.0);
            if !take_draft(a, expected, j, t(j), t(j + 1), d) {
                break;
            }
            expected += a;
            taken += 1;
        }
        assert_eq!(taken, 2);
    }

    /// Speculative decoding with a drafter must not change greedy output: the
    /// same tokens, the same usage and the same cache bookkeeping as plain
    /// decoding, whatever the draft length and however many drafts the target
    /// rejects.
    #[test]
    fn drafter_leaves_greedy_output_unchanged() {
        let model = write_tiny_model();
        let drafter = write_tiny_drafter();
        let prompts: [&[u32]; 4] = [
            &PROMPT,
            &[2, 9, 9, 9, 10],
            &[2, 13, 12, 11, 10, 9, 13],
            &[2, 11],
        ];
        let mut drafted = 0;
        for k in [1usize, 2, 4, 8] {
            for prompt in prompts {
                let mut plain = engine(&model, 32, 0.0, None);
                let mut spec = engine_with_drafter(&model, &drafter, 32, 0.0, None, k, 1);
                let expected = run(&mut plain, prompt, None);
                let got = run(&mut spec, prompt, None);
                let what = format!("k {k}, prompt {prompt:?}");
                assert_same_reply(&got, &expected, &what);
                assert_eq!(got.items.len(), expected.items.len(), "{what}");
                // The finished sequence left no block referenced, and its ids
                // equal plain decoding's: the K/V of rejected drafts never
                // leaks into the bookkeeping.
                assert_eq!(used_blocks(&spec), 0);
                assert_eq!(spec.last_ids, plain.last_ids, "k {k}, prompt {prompt:?}");
                drafted += spec.stats.snapshot().drafted;
            }
        }
        assert!(drafted > 0, "the drafter proposed tokens");
    }

    #[test]
    fn drafter_respects_max_tokens_and_the_context_limit() {
        let model = write_tiny_model();
        let drafter = write_tiny_drafter();
        for max in [1u32, 2, 3, 7] {
            let mut plain = engine(&model, 32, 0.0, None);
            let mut spec = engine_with_drafter(&model, &drafter, 32, 0.0, None, 4, 1);
            let a = run(&mut plain, &PROMPT, Some(max));
            let b = run(&mut spec, &PROMPT, Some(max));
            assert_same_reply(&a, &b, &format!("max {max}"));
        }
        // A context that fills up mid-round.
        let mut plain = engine(&model, 12, 0.0, None);
        let mut spec = engine_with_drafter(&model, &drafter, 12, 0.0, None, 8, 1);
        let a = run(&mut plain, &PROMPT, None);
        let b = run(&mut spec, &PROMPT, None);
        assert_eq!(text(&a), text(&b));
    }

    /// With sampling, speculative decoding follows the target distribution
    /// (checked in the sampler tests); here it must run, report usage and
    /// stay reproducible for a fixed seed.
    #[test]
    fn drafter_with_sampling_is_reproducible_for_a_seed() {
        let model = write_tiny_model();
        let drafter = write_tiny_drafter();
        let run_once = || {
            let mut e = engine_with_drafter(&model, &drafter, 32, 1.0, Some(5), 3, 1);
            let out = run(&mut e, &PROMPT, None);
            (
                text(&out),
                out.metadata.token_usage.map(|u| u.completion_tokens),
            )
        };
        let (a, n) = run_once();
        let (b, m) = run_once();
        assert_eq!((a, n), (b, m));
        assert!(n.is_some());
    }

    #[test]
    fn a_drafter_cannot_be_a_main_model_and_checks_its_target() {
        use crate::provider::local::model::cpu::Transformer;
        let model = write_tiny_model();
        let drafter = write_tiny_drafter();
        let content = gguf::Content::open(&drafter.path).unwrap();
        let e = Transformer::load(&content, 32).err().unwrap();
        assert!(e.to_string().contains("drafter"), "{e}");
        let content = gguf::Content::open(&model.path).unwrap();
        let e = Transformer::load_drafter(&content, 32).err().unwrap();
        assert!(e.to_string().contains("not a drafter"), "{e}");
    }

    #[test]
    fn batching_resolves_against_the_context_length() {
        // `props` are the properties of a KDL `batching` node.
        let config = |props: &str| {
            crate::config::load_kdl_str::<LocalModelConfig>(&format!("batching {props}")).unwrap()
        };
        let (batching, sizing) = resolve_batching(&config(""), 1000).unwrap();
        assert_eq!(
            batching,
            Batching {
                max_sequences: 4,
                max_step_tokens: 256,
                prefill_chunk: 256,
                prefix_cache: true
            }
        );
        assert_eq!(sizing.kv_tokens, 2000, "twice the context by default");
        // A GPU keeps one context and at most 4096 positions of prefixes.
        let gpu = LocalDevice::Vulkan(super::super::GpuAdapter::Default);
        assert_eq!(default_kv_tokens(1000, gpu), 2000);
        assert_eq!(default_kv_tokens(8192, gpu), 12288);
        assert_eq!(default_kv_tokens(32768, gpu), 36864);
        assert_eq!(default_kv_tokens(8192, LocalDevice::Cpu), 16384);
        assert_eq!(sizing.max_sequences, 4);
        // An explicit pool rounds up to blocks and holds at least a context
        // and a block.
        let (_, sizing) = resolve_batching(&config("kv_tokens=1016"), 1000).unwrap();
        assert_eq!(sizing.kv_tokens, 1024);
        let err = resolve_batching(&config("kv_tokens=1015"), 1000).unwrap_err();
        assert!(
            matches!(&err, CucaError::Config(m) if m.contains("kv_tokens")),
            "{err}"
        );
        // A very long context shrinks the prefill chunk below the sequences.
        let (batching, _) = resolve_batching(&config("max_sequences=64"), 131_072).unwrap();
        assert_eq!(batching.max_step_tokens, 64);
        // A drafter that observes every position keeps the configured
        // sequences (it forks one cache per state slot).
        let (batching, sizing) = resolve_batching(&config("max_sequences=8"), 1000).unwrap();
        assert_eq!((batching.max_sequences, sizing.max_sequences), (8, 8));
        let (batching, _) =
            resolve_batching(&config("prefix_cache=#false max_step_tokens=32"), 1000).unwrap();
        assert_eq!(
            (batching.max_step_tokens, batching.prefix_cache),
            (32, false)
        );
    }

    /// What a request reported it reused and prefilled.
    fn usage(out: &Outcome) -> (u32, u32) {
        let u = out.metadata.prompt_cache_usage.expect("cache usage");
        (u.read_tokens, u.write_tokens)
    }

    /// A tiny hybrid model that loads with a context of 128.
    fn hybrid_model() -> crate::provider::local::model::tiny::TinyModel {
        hybrid(TinyQwen35::default())
    }

    /// `base` with a context of 128.
    fn hybrid(base: TinyQwen35) -> crate::provider::local::model::tiny::TinyModel {
        write_qwen35(&TinyQwen35 {
            meta: vec![("context_length".into(), gguf::Value::U32(128))],
            ..base
        })
    }

    /// A hybrid model mixing the weight dtypes of the row-batched kernel,
    /// with 256-value-type FFN down projections (trunk and head).
    fn mixed_qwen35() -> TinyQwen35 {
        TinyQwen35 {
            ff: 256,
            mixed: true,
            ..TinyQwen35::default()
        }
    }

    #[test]
    fn snapshot_position_follows_the_last_end_of_turn_rounded_to_blocks() {
        let model = hybrid_model();
        let e = engine(&model, 128, 0.0, None);
        // Id 3 ends a turn: the snapshot goes right after the last one,
        // before the generation prompt, on a block boundary.
        let mut prompt = long_prompt(40);
        prompt[19] = 3;
        assert_eq!(e.snapshot_position(&prompt), 16);
        assert_eq!(e.snapshot_position(&long_prompt(40)), 32, "none: the end");
        // Never at the last token: its logits start generation.
        assert_eq!(e.snapshot_position(&long_prompt(32)), 16);
        assert_eq!(e.snapshot_position(&long_prompt(16)), 0);
        assert_eq!(e.snapshot_position(&[3]), 0);
    }

    /// Hybrid (recurrent-state) model: a prefix hit needs the snapshot of
    /// its position, so the engine reuses the nearest block boundary it has
    /// one for and reports what it really reused; the output stays exactly
    /// what a cold engine produces.
    #[test]
    fn hybrid_prefix_hit_uses_the_snapshot_and_matches_cold() {
        let model = hybrid_model();
        let mut warm = engine(&model, 128, 0.0, None);
        assert!(warm.model.is_hybrid());
        let mut first = long_prompt(40);
        first[19] = 3;
        let a = run(&mut warm, &first, Some(4));
        assert_eq!(usage(&a), (0, 40));
        assert_eq!(warm.snapshots.len(), 1);

        // The same conversation continues: 30 shared tokens make one shared
        // block, and the snapshot sits exactly there.
        let mut second = first[..30].to_vec();
        second.extend([14, 15, 13, 12, 11, 10, 9, 12, 13, 14]);
        let b = run(&mut warm, &second, Some(5));
        assert_eq!(usage(&b), (16, 24));
        let cold = run(&mut engine(&model, 128, 0.0, None), &second, Some(5));
        assert_eq!(usage(&cold), (0, 40));
        assert_eq!(text(&b), text(&cold));
        assert_eq!(
            b.metadata.token_usage.unwrap().completion_tokens,
            cold.metadata.token_usage.unwrap().completion_tokens
        );

        // The first prompt again, extended: two blocks are cached but the
        // model holds a state only at 16.
        let mut extended = first.clone();
        extended.extend([15, 14, 13]);
        let c = run(&mut warm, &extended, Some(3));
        assert_eq!(usage(&c), (16, 27));
        let cold = run(&mut engine(&model, 128, 0.0, None), &extended, Some(3));
        assert_eq!(text(&c), text(&cold));

        // An unrelated prompt restores nothing.
        let unrelated: Vec<u32> = (0..30).map(|i| 15 - (i % 5) as u32).collect();
        let d = run(&mut warm, &unrelated, Some(2));
        assert_eq!(usage(&d), (0, 30));

        // A raw prompt (no end-of-turn token) snapshots at its last block
        // boundary below its end, so an extension reuses it.
        let mut e = engine(&model, 128, 0.0, None);
        let raw = long_prompt(37);
        run(&mut e, &raw, Some(3));
        let mut longer = raw.clone();
        longer.extend([14, 15]);
        let out = run(&mut e, &longer, Some(3));
        assert_eq!(usage(&out), (32, 7));
        assert_eq!(
            text(&out),
            text(&run(&mut engine(&model, 128, 0.0, None), &longer, Some(3)))
        );
    }

    #[test]
    fn hybrid_engine_recovers_from_a_failed_request() {
        let model = write_qwen35(&TinyQwen35::default());
        let mut e = engine(&model, 32, 0.0, None);
        let prompt = [2u32, 9, 10, 3, 11, 12, 13];
        let before = run(&mut e, &prompt, Some(4));
        // Token 99 is outside the vocabulary: the request fails and the
        // cache, state included, is reset.
        let out = run(&mut e, &[2, 9, 10, 3, 99], Some(2));
        assert!(matches!(&out.items[0], Err(CucaError::Provider { .. })));
        assert!(e.last_ids.is_empty());
        assert_eq!(used_blocks(&e), 0);
        let after = run(&mut e, &prompt, Some(4));
        assert_eq!(usage(&after), (0, 7));
        assert_eq!(text(&after), text(&before));
    }

    /// The committed pairs of the in-trunk head of `e`.
    fn head_len(e: &Engine) -> usize {
        match e.drafters.as_slice() {
            [drafter @ Drafter::InTrunk(head, _), ..] => head.len() + drafter.pending(),
            _ => panic!("the engine has no in-trunk head"),
        }
    }

    /// Drafting with the head inside the file must not change greedy output:
    /// the same tokens, the same usage and the same cache bookkeeping as
    /// plain decoding, whatever the draft length and however many drafts the
    /// trunk rejects; and the head stays one pair behind the trunk's cache.
    #[test]
    fn in_trunk_head_leaves_greedy_output_unchanged() {
        in_trunk_head_is_exact(&write_qwen35(&TinyQwen35::default()), true);
        // The mixed-dtype model's head is random noise that never agrees with
        // the trunk: every draft is verified in a batched pass and rolled
        // back, and the output is still the plain output.
        in_trunk_head_is_exact(&write_qwen35(&mixed_qwen35()), false);
    }

    /// `accepts`: some drafts must be accepted and rolled forward.
    fn in_trunk_head_is_exact(
        model: &crate::provider::local::model::tiny::TinyModel,
        accepts: bool,
    ) {
        let prompts: [&[u32]; 5] = [
            &PROMPT,
            &[2, 9, 9, 9, 10],
            &[2, 13, 12, 11, 10, 9, 13],
            &[2, 11],
            &[2, 12, 10, 14, 13, 11, 9, 15, 10, 12, 14, 11],
        ];
        let (mut drafted, mut accepted) = (0, 0);
        for k in [1usize, 2, 4, 8] {
            for prompt in prompts {
                let mut plain = engine(model, 32, 0.0, None);
                let mut spec = engine_in_trunk(model, 32, 0.0, None, k, 1);
                let expected = run(&mut plain, prompt, None);
                let got = run(&mut spec, prompt, None);
                let what = format!("k {k}, prompt {prompt:?}");
                assert_same_reply(&got, &expected, &what);
                assert_eq!(got.items.len(), expected.items.len(), "{what}");
                assert_eq!(spec.last_ids, plain.last_ids, "k {k}, prompt {prompt:?}");
                assert_eq!(
                    head_len(&spec),
                    spec.last_ids.len() - 1,
                    "k {k}, prompt {prompt:?}"
                );
                let stats = spec.stats.snapshot();
                drafted += stats.drafted;
                accepted += stats.accepted;
            }
        }
        assert!(drafted > 0, "the head proposed tokens");
        if accepts {
            assert!(accepted > 0, "some drafts were accepted and rolled forward");
        }
    }

    /// A token budget that runs out inside a speculative round caches exactly
    /// the ids plain decoding does: the draft that spends the last token is
    /// emitted but, as in plain decoding, never forwarded.
    #[test]
    fn a_budget_ending_inside_a_round_caches_what_plain_decoding_does() {
        let (model, eagle) = (mistral4(16), eagle());
        let mut spent = 0;
        let prompts: Vec<Vec<u32>> = (0..40usize)
            .map(|s| {
                let n = 1 + s % 5;
                let mut p = vec![2u32];
                p.extend((0..n).map(|i| 9 + ((i * 7 + s * 3 + i * s) % 7) as u32));
                p
            })
            .collect();
        for prompt in &prompts {
            for max in 1..=10u32 {
                for k in [1usize, 2, 4, 8] {
                    let mut plain = engine(&model, 16, 0.0, None);
                    let mut spec = engine_with_eagle(&model, &eagle, 16, 0.0, None, k, 1);
                    let a = run(&mut plain, prompt, Some(max));
                    let b = run(&mut spec, prompt, Some(max));
                    let what = format!("k {k}, prompt {prompt:?}, max {max}");
                    assert_same_reply(&a, &b, &what);
                    assert_eq!(spec.last_ids, plain.last_ids, "{what}");
                    assert_eq!(eagle_len(&spec), spec.last_ids.len() - 1, "{what}");
                    spent += usize::from(completion_tokens(&b) == Some(max));
                }
            }
        }
        assert!(spent > 0, "some reply spent its whole budget");
    }

    /// A stop string that completes inside a speculative round ends the reply
    /// and caches exactly what plain decoding does: the draft that completes
    /// it is emitted but never forwarded.
    #[test]
    fn a_stop_string_ending_inside_a_round_caches_what_plain_decoding_does() {
        let (model, eagle) = (mistral4(16), eagle());
        let mut stopped = 0;
        for s in 0..24usize {
            let n = 1 + s % 5;
            let mut prompt = vec![2u32];
            prompt.extend((0..n).map(|i| 9 + ((i * 7 + s * 3 + i * s) % 7) as u32));
            let free = text(&run(&mut engine(&model, 16, 0.0, None), &prompt, Some(10)));
            let chars: Vec<char> = free.chars().collect();
            for at in 0..chars.len().saturating_sub(1) {
                let stop: String = chars[at..at + 2].iter().collect();
                for k in [1usize, 2, 4, 8] {
                    let mut plain = engine(&model, 16, 0.0, None);
                    let mut spec = engine_with_eagle(&model, &eagle, 16, 0.0, None, k, 1);
                    let with_stop = |e: &mut Engine| {
                        run_with(e, &prompt, Some(10), |job| job.stop = vec![stop.clone()])
                    };
                    let a = with_stop(&mut plain);
                    let b = with_stop(&mut spec);
                    let what = format!("k {k}, prompt {prompt:?}, stop {stop:?}");
                    assert_same_reply(&a, &b, &what);
                    assert_eq!(spec.last_ids, plain.last_ids, "{what}");
                    assert_eq!(eagle_len(&spec), spec.last_ids.len() - 1, "{what}");
                    stopped += usize::from(text(&a).len() < free.len());
                }
            }
        }
        assert!(stopped > 0, "some reply ended on its stop string");
    }

    #[test]
    fn in_trunk_head_respects_max_tokens_and_the_context_limit() {
        let model = write_qwen35(&TinyQwen35::default());
        for max in [1u32, 2, 3, 7] {
            let mut plain = engine(&model, 32, 0.0, None);
            let mut spec = engine_in_trunk(&model, 32, 0.0, None, 4, 1);
            let a = run(&mut plain, &PROMPT, Some(max));
            let b = run(&mut spec, &PROMPT, Some(max));
            assert_same_reply(&a, &b, &format!("max {max}"));
            assert_eq!(spec.last_ids, plain.last_ids, "max {max}");
            assert_eq!(head_len(&spec), spec.last_ids.len() - 1, "max {max}");
        }
        // A context that fills up mid-round, for several sizes and prompts:
        // the completion stops at the room the context leaves.
        let mut filled = 0;
        for ctx in [7usize, 8, 9, 10, 12, 16] {
            for prompt in [
                &PROMPT[..],
                &[2u32, 9, 9, 9, 10][..],
                &[2u32, 11][..],
                &[2u32, 12, 10, 14][..],
                &[2u32, 13, 12, 11, 10][..],
            ] {
                let mut plain = engine(&model, ctx, 0.0, None);
                let mut spec = engine_in_trunk(&model, ctx, 0.0, None, 8, 1);
                let a = run(&mut plain, prompt, None);
                let b = run(&mut spec, prompt, None);
                assert_eq!(text(&a), text(&b), "ctx {ctx}, prompt {prompt:?}");
                assert_eq!(
                    spec.last_ids, plain.last_ids,
                    "ctx {ctx}, prompt {prompt:?}"
                );
                assert_eq!(head_len(&spec), spec.last_ids.len() - 1, "ctx {ctx}");
                filled +=
                    usize::from(completion_tokens(&b) == u32::try_from(ctx - prompt.len()).ok());
            }
        }
        assert!(filled > 0, "some run stopped at the context limit");
    }

    /// Warm prefix reuse with the head equals a cold run, bit for bit: the
    /// trunk resumes from the snapshot of a shared block boundary, the head
    /// keeps what the shared tokens still determine and re-observes the
    /// replayed rows.
    #[test]
    fn in_trunk_head_warm_prefix_reuse_equals_cold() {
        for base in [TinyQwen35::default(), mixed_qwen35()] {
            in_trunk_head_warm_prefix_reuse_is_exact(&hybrid(base));
        }
    }

    fn in_trunk_head_warm_prefix_reuse_is_exact(
        model: &crate::provider::local::model::tiny::TinyModel,
    ) {
        let mut first = long_prompt(40);
        first[19] = 3;
        let mut second = first[..30].to_vec();
        second.extend([14, 15, 13, 12, 11, 10, 9, 12, 13, 14]);
        let mut warm = engine_in_trunk(model, 128, 0.0, None, 3, 1);
        let a = run(&mut warm, &first, Some(6));
        assert_eq!(usage(&a), (0, 40));
        assert_eq!(head_len(&warm), warm.last_ids.len() - 1);

        // Resumes from the snapshot at 16: nothing is missing in the head.
        let b = run(&mut warm, &second, Some(6));
        assert_eq!(usage(&b), (16, 24));
        let cold = run(
            &mut engine_in_trunk(model, 128, 0.0, None, 3, 1),
            &second,
            Some(6),
        );
        let plain = run(&mut engine(model, 128, 0.0, None), &second, Some(6));
        assert_eq!(text(&b), text(&cold));
        assert_eq!(text(&b), text(&plain));
        assert_eq!(head_len(&warm), warm.last_ids.len() - 1);

        // The same prompt again goes back to the snapshot too.
        let c = run(&mut warm, &second, Some(6));
        assert_eq!(usage(&c), (16, 24));
        assert_eq!(text(&c), text(&plain));
        assert_eq!(head_len(&warm), warm.last_ids.len() - 1);

        // The cached ids plus more tokens: two blocks are cached, the model
        // holds a state at 16, and the head's tail bridges the request
        // boundary.
        let mut extended = warm.last_ids.clone();
        extended.extend([15, 14]);
        let d = run(&mut warm, &extended, Some(4));
        assert_eq!(usage(&d), (16, u32::try_from(extended.len() - 16).unwrap()));
        let expected = run(&mut engine(model, 128, 0.0, None), &extended, Some(4));
        assert_eq!(text(&d), text(&expected));
        assert_eq!(head_len(&warm), warm.last_ids.len() - 1);

        // A prompt that diverges inside the second block keeps the first.
        let mut diverging = first[..18].to_vec();
        diverging.extend([15, 14, 13, 12, 11, 10, 9, 15, 14, 13, 12, 11]);
        let e = run(&mut warm, &diverging, Some(5));
        assert_eq!(usage(&e), (16, 14));
        let expected = run(&mut engine(model, 128, 0.0, None), &diverging, Some(5));
        assert_eq!(text(&e), text(&expected));
        assert_eq!(head_len(&warm), warm.last_ids.len() - 1);
    }

    /// A failed request resets the head with the trunk, and the engine keeps
    /// serving.
    #[test]
    fn in_trunk_head_recovers_from_a_failed_request() {
        let model = write_qwen35(&TinyQwen35::default());
        let mut e = engine_in_trunk(&model, 32, 0.0, None, 3, 1);
        let prompt = [2u32, 9, 10, 3, 11, 12, 13];
        let before = run(&mut e, &prompt, Some(4));
        let out = run(&mut e, &[2, 9, 10, 3, 99], Some(2));
        assert!(matches!(&out.items[0], Err(CucaError::Provider { .. })));
        assert!(e.last_ids.is_empty());
        assert_eq!(head_len(&e), 0);
        let after = run(&mut e, &prompt, Some(4));
        assert_eq!(usage(&after), (0, 7));
        assert_eq!(text(&after), text(&before));
        assert_eq!(head_len(&e), e.last_ids.len() - 1);
    }

    #[test]
    fn in_trunk_head_with_sampling_is_reproducible_for_a_seed() {
        let model = write_qwen35(&TinyQwen35::default());
        seeded_run_is_reproducible(|| engine_in_trunk(&model, 32, 1.0, Some(5), 3, 1));
    }

    /// The default context length is the model's maximum when that is below
    /// 8192 (the tiny model trains for 32 positions) and loads; an explicit
    /// value above the maximum stays a loud error.
    #[test]
    fn default_context_length_resolves_to_the_model_maximum() {
        let model = write_tiny_model();
        let content = gguf::Content::open(&model.path).unwrap();
        let hp = super::super::model::read_hparams(&content).unwrap();
        let resolved = super::super::resolve_context_length(None, hp.n_ctx_train);
        assert_eq!(resolved, 32);
        let device = super::super::LocalDevice::Cpu;
        let resident = super::super::ExpertPlacement::Resident;
        assert!(
            Model::load(
                &content,
                resolved,
                device,
                &[],
                &resident,
                &Sizing::single(resolved),
                None,
                &super::super::EngramCache::default(),
            )
            .is_ok()
        );
        let e = Model::load(
            &content,
            33,
            device,
            &[],
            &resident,
            &Sizing::single(33),
            None,
            &super::super::EngramCache::default(),
        )
        .err();
        assert!(
            e.is_some_and(|e| e.to_string().contains("exceeds the model maximum")),
            "an explicit value above the maximum fails loudly"
        );
    }

    /// `DrafterSource::InTrunk` on a model whose file has no prediction block
    /// (or whose architecture embeds none) says so, before anything else.
    #[test]
    fn in_trunk_head_needs_a_prediction_block() {
        let config = LocalModelConfig {
            context_length: Some(32),
            drafter: Some(super::super::DrafterSource::InTrunk),
            ..LocalModelConfig::default()
        };
        let gemma = write_tiny_model();
        let llama = write_spec(&TinySpec::llama());
        let bare = write_qwen35(&TinyQwen35 {
            nextn_layers: 0,
            ..TinyQwen35::default()
        });
        for (what, model) in [("gemma4", &gemma), ("llama", &llama), ("qwen35", &bare)] {
            let result = Engine::open(
                &model.path,
                Some(&ResolvedDrafter::InTrunk),
                Arc::default(),
                &config,
                None,
            );
            let e = match result {
                Err(e) => e,
                Ok(_) => panic!("{what}: opened without a prediction head"),
            };
            assert!(matches!(e, CucaError::Provider { .. }), "{what}: {e}");
            assert!(
                e.to_string()
                    .contains("no in-trunk multi-token-prediction head"),
                "{what}: {e}"
            );
        }
    }

    /// `DrafterSource::InTrunk` on a `qwen35moe` model fails with
    /// [`CucaError::Config`] naming the mixture-of-experts FFN; without a
    /// drafter the same file opens and generates.
    #[test]
    fn in_trunk_head_on_a_moe_model_fails_loudly_and_plain_decoding_works() {
        let model = write_qwen35(&TinyQwen35 {
            moe: Some(crate::provider::local::model::tiny::TinyMoe {
                n_expert: 8,
                n_used: 2,
                ff: 64,
                shared_ff: 32,
            }),
            ..TinyQwen35::default()
        });
        let config = LocalModelConfig {
            context_length: Some(32),
            drafter: Some(super::super::DrafterSource::InTrunk),
            ..LocalModelConfig::default()
        };
        let result = Engine::open(
            &model.path,
            Some(&ResolvedDrafter::InTrunk),
            Arc::default(),
            &config,
            None,
        );
        let e = match result {
            Err(e) => e,
            Ok(_) => panic!("opened an in-trunk head over experts"),
        };
        assert!(
            matches!(&e, CucaError::Config(m) if m.contains("mixture-of-experts")),
            "{e}"
        );
        let mut plain = engine(&model, 32, 0.0, None);
        let out = run(&mut plain, &PROMPT, Some(4));
        let usage = out.metadata.token_usage.expect("usage reported");
        assert_eq!(usage.prompt_tokens, 6);
        assert!(usage.completion_tokens <= 4);
    }

    // ---- the EAGLE drafter ----

    /// A tiny `mistral4` target (latent attention over a dense FFN in its
    /// first layer and experts after it) with a context of `context`.
    fn mistral4(context: usize) -> crate::provider::local::model::tiny::TinyModel {
        use crate::provider::local::model::tiny::{TinyMistral4, write_mistral4};
        write_mistral4(&TinyMistral4 {
            dense_layers: 1,
            meta: vec![("context_length".into(), gguf::Value::U32(context as u32))],
            ..TinyMistral4::default()
        })
    }

    fn eagle() -> crate::provider::local::model::tiny::TinyModel {
        use crate::provider::local::model::tiny::{TinyEagle, write_tiny_eagle};
        write_tiny_eagle(&TinyEagle::default())
    }

    /// The committed pairs of the EAGLE drafter of `e`.
    fn eagle_len(e: &Engine) -> usize {
        match e.drafters.as_slice() {
            [drafter @ Drafter::Eagle(head, _), ..] => head.len() + drafter.pending(),
            _ => panic!("the engine has no EAGLE drafter"),
        }
    }

    /// The EAGLE drafter must not change greedy output: the same tokens, the
    /// same usage and the same cache bookkeeping as plain decoding, whatever
    /// the draft length and however many drafts the target rejects; and the
    /// drafter stays one pair behind the target's cache.
    #[test]
    fn eagle_leaves_greedy_output_unchanged() {
        let (model, eagle) = (mistral4(32), eagle());
        let prompts: [&[u32]; 5] = [
            &PROMPT,
            &[2, 9, 9, 9, 10],
            &[2, 13, 12, 11, 10, 9, 13],
            &[2, 11],
            &[2, 12, 10, 14, 13, 11, 9, 15, 10, 12, 14, 11],
        ];
        let (mut drafted, mut accepted) = (0, 0);
        for k in [1usize, 2, 4, 8] {
            for prompt in prompts {
                let mut plain = engine(&model, 32, 0.0, None);
                let mut spec = engine_with_eagle(&model, &eagle, 32, 0.0, None, k, 1);
                let expected = run(&mut plain, prompt, None);
                let got = run(&mut spec, prompt, None);
                let what = format!("k {k}, prompt {prompt:?}");
                assert_same_reply(&got, &expected, &what);
                assert_eq!(got.items.len(), expected.items.len(), "{what}");
                assert_eq!(spec.last_ids, plain.last_ids, "k {k}, prompt {prompt:?}");
                assert_eq!(
                    eagle_len(&spec),
                    spec.last_ids.len() - 1,
                    "k {k}, prompt {prompt:?}"
                );
                let stats = spec.stats.snapshot();
                drafted += stats.drafted;
                accepted += stats.accepted;
            }
        }
        assert!(drafted > 0, "the drafter proposed tokens");
        assert!(accepted > 0, "some drafts were accepted and rolled forward");
    }

    #[test]
    fn eagle_respects_max_tokens_and_the_context_limit() {
        let (model, eagle) = (mistral4(32), eagle());
        for max in [1u32, 2, 3, 7] {
            let mut plain = engine(&model, 32, 0.0, None);
            let mut spec = engine_with_eagle(&model, &eagle, 32, 0.0, None, 4, 1);
            let a = run(&mut plain, &PROMPT, Some(max));
            let b = run(&mut spec, &PROMPT, Some(max));
            assert_same_reply(&a, &b, &format!("max {max}"));
            assert_eq!(eagle_len(&spec), spec.last_ids.len() - 1, "max {max}");
        }
        // A context that fills up mid-round, for several sizes and prompts:
        // the completion stops at the room the context leaves.
        let mut filled = 0;
        for ctx in [7usize, 8, 9, 10, 12, 16] {
            for prompt in [
                &PROMPT[..],
                &[2u32, 9, 9, 9, 10][..],
                &[2u32, 11][..],
                &[2u32, 12, 10, 14][..],
                &[2u32, 13, 12, 11, 10][..],
            ] {
                let mut plain = engine(&model, ctx, 0.0, None);
                let mut spec = engine_with_eagle(&model, &eagle, ctx, 0.0, None, 8, 1);
                let a = run(&mut plain, prompt, None);
                let b = run(&mut spec, prompt, None);
                assert_eq!(text(&a), text(&b), "ctx {ctx}, prompt {prompt:?}");
                assert_eq!(
                    spec.last_ids, plain.last_ids,
                    "ctx {ctx}, prompt {prompt:?}"
                );
                assert_eq!(eagle_len(&spec), spec.last_ids.len() - 1, "ctx {ctx}");
                filled +=
                    usize::from(completion_tokens(&b) == u32::try_from(ctx - prompt.len()).ok());
            }
        }
        assert!(filled > 0, "some run stopped at the context limit");
    }

    /// Warm prefix reuse with the EAGLE drafter equals a cold run, bit for
    /// bit: the target resumes from the shared blocks, the drafter keeps what
    /// the shared tokens still determine and re-observes the replayed rows,
    /// and its K/V ends up the cold drafter's.
    #[test]
    fn eagle_warm_prefix_reuse_equals_cold() {
        let (model, eagle) = (mistral4(128), eagle());
        let mut first = long_prompt(40);
        first[19] = 3;
        let mut second = first[..30].to_vec();
        second.extend([14, 15, 13, 12, 11, 10, 9, 12, 13, 14]);
        let kv_of = |e: &Engine| match e.drafters.as_slice() {
            [Drafter::Eagle(head, _), ..] => head.kv_prefix(),
            _ => panic!("the engine has no EAGLE drafter"),
        };
        let mut warm = engine_with_eagle(&model, &eagle, 128, 0.0, None, 3, 1);
        let a = run(&mut warm, &first, Some(6));
        assert_eq!(usage(&a), (0, 40));
        assert_eq!(eagle_len(&warm), warm.last_ids.len() - 1);

        // Resumes from the first shared block: 16 of the 30 shared tokens.
        let b = run(&mut warm, &second, Some(6));
        assert_eq!(usage(&b), (16, 24));
        let mut cold_engine = engine_with_eagle(&model, &eagle, 128, 0.0, None, 3, 1);
        let cold = run(&mut cold_engine, &second, Some(6));
        let plain = run(&mut engine(&model, 128, 0.0, None), &second, Some(6));
        assert_eq!(text(&b), text(&cold));
        assert_eq!(text(&b), text(&plain));
        assert_eq!(eagle_len(&warm), warm.last_ids.len() - 1);
        assert_eq!(warm.last_ids, cold_engine.last_ids);
        assert_eq!(kv_of(&warm), kv_of(&cold_engine));

        // The same prompt again reads both full blocks below its last token.
        let c = run(&mut warm, &second, Some(6));
        assert_eq!(usage(&c), (32, 8));
        assert_eq!(text(&c), text(&plain));
        assert_eq!(eagle_len(&warm), warm.last_ids.len() - 1);
        assert_eq!(kv_of(&warm), kv_of(&cold_engine));

        // The cached ids plus more tokens: the drafter's tail bridges the
        // request boundary.
        let mut extended = warm.last_ids.clone();
        extended.extend([15, 14]);
        let d = run(&mut warm, &extended, Some(4));
        assert_eq!(usage(&d).0 % 16, 0);
        assert!(usage(&d).0 >= 32, "{:?}", usage(&d));
        let expected = run(&mut engine(&model, 128, 0.0, None), &extended, Some(4));
        assert_eq!(text(&d), text(&expected));
        assert_eq!(eagle_len(&warm), warm.last_ids.len() - 1);

        // A prompt that diverges inside the second block keeps the first.
        let mut diverging = first[..18].to_vec();
        diverging.extend([15, 14, 13, 12, 11, 10, 9, 15, 14, 13, 12, 11]);
        let e = run(&mut warm, &diverging, Some(5));
        assert_eq!(usage(&e), (16, 14));
        let expected = run(&mut engine(&model, 128, 0.0, None), &diverging, Some(5));
        assert_eq!(text(&e), text(&expected));
        assert_eq!(eagle_len(&warm), warm.last_ids.len() - 1);
    }

    /// A failed request resets the drafter with the target, and the engine
    /// keeps serving.
    #[test]
    fn eagle_recovers_from_a_failed_request() {
        let (model, eagle) = (mistral4(32), eagle());
        let mut e = engine_with_eagle(&model, &eagle, 32, 0.0, None, 3, 1);
        let prompt = [2u32, 9, 10, 3, 11, 12, 13];
        let before = run(&mut e, &prompt, Some(4));
        let out = run(&mut e, &[2, 9, 10, 3, 99], Some(2));
        assert!(matches!(&out.items[0], Err(CucaError::Provider { .. })));
        assert!(e.last_ids.is_empty());
        assert_eq!(eagle_len(&e), 0);
        let after = run(&mut e, &prompt, Some(4));
        assert_eq!(usage(&after), (0, 7));
        assert_eq!(text(&after), text(&before));
        assert_eq!(eagle_len(&e), e.last_ids.len() - 1);
    }

    #[test]
    fn eagle_with_sampling_is_reproducible_for_a_seed() {
        let (model, eagle) = (mistral4(32), eagle());
        seeded_run_is_reproducible(|| engine_with_eagle(&model, &eagle, 32, 1.0, Some(5), 3, 1));
    }

    /// The engine recognises an EAGLE file by its architecture, keeps one
    /// drafter cache per sequence with it, pairs it with a `mistral4` target
    /// only, and refuses a GPU before loading anything.
    #[test]
    fn the_eagle_drafter_serves_mistral4_sequences_on_the_cpu() {
        use crate::provider::local::model::mtp::Drafter;
        let (model, eagle) = (mistral4(32), eagle());
        let header =
            |m: &crate::provider::local::model::tiny::TinyModel| open_gguf(&m.path, &[]).unwrap();
        assert_eq!(architecture_of(&header(&eagle)), Arch::Mistral4Eagle.name());

        // The configured sequences all run: the drafter forks one cache per
        // state slot instead of pinning the engine to one sequence.
        let config = LocalModelConfig {
            batching: super::super::BatchingConfig {
                max_sequences: 4,
                ..Default::default()
            },
            ..LocalModelConfig::default()
        };
        assert_eq!(resolve_batching(&config, 32).unwrap().0.max_sequences, 4);
        let e = engine_with_eagle(&model, &eagle, 32, 0.0, None, 3, 4);
        assert_eq!(e.drafters.len(), 4);
        assert!(e.drafters.iter().all(Drafter::observes));

        // Another target model is refused, loudly.
        let gemma = engine(&write_tiny_model(), 32, 0.0, None);
        let content = header(&eagle);
        let e = Drafter::open_eagle(&content, &gemma.model)
            .err()
            .unwrap()
            .to_string();
        assert!(
            e.contains("pairs with a `mistral4` model, not `gemma4`"),
            "{e}"
        );

        // A GPU fails before the target is read any further.
        let gpu = LocalModelConfig {
            device: LocalDevice::Vulkan(super::super::GpuAdapter::Default),
            ..LocalModelConfig::default()
        };
        let result = Engine::open(
            &model.path,
            Some(&ResolvedDrafter::File(eagle.path.clone())),
            Arc::default(),
            &gpu,
            None,
        );
        let e = match result {
            Err(e) => e,
            Ok(_) => panic!("opened an EAGLE drafter on a GPU"),
        };
        assert!(
            matches!(&e, CucaError::Config(m) if m.contains("EAGLE drafter runs on the CPU only")),
            "{e}"
        );
    }

    // ---- DFlash block drafter ----------------------------------------------

    /// A tiny `muse-glimmer` target that loads with `context` positions.
    fn muse_target(context: usize) -> crate::provider::local::model::tiny::TinyModel {
        write_spec(&TinySpec {
            context,
            ..TinySpec::muse_glimmer()
        })
    }

    /// The positions `lo..hi` the DFlash drafter of `e` holds context for.
    fn dflash_held(e: &Engine) -> (usize, usize) {
        match e.drafters.as_slice() {
            [drafter @ Drafter::Dflash(net, _), ..] => {
                let (lo, hi) = net.held();
                (lo, hi + drafter.pending())
            }
            _ => panic!("the engine has no DFlash drafter"),
        }
    }

    /// Drafting with a DFlash block drafter must not change greedy output:
    /// the same tokens, the same usage and the same cache bookkeeping as
    /// plain decoding, whatever the block length, the draft cap and however
    /// many drafts the target rejects; and the drafter's context ends where
    /// the trunk's cache does.
    #[test]
    fn dflash_drafter_leaves_greedy_output_unchanged() {
        let model = muse_target(32);
        let prompts: [&[u32]; 5] = [
            &PROMPT,
            &[2, 9, 9, 9, 10],
            &[2, 13, 12, 11, 10, 9, 13],
            &[2, 11],
            &[2, 12, 10, 14, 13, 11, 9, 15, 10, 12, 14, 11],
        ];
        let (mut drafted, mut accepted) = (0, 0);
        for (block, window) in [
            (2usize, Some(6usize)),
            (5, Some(6)),
            (8, Some(4)),
            (8, None),
            (16, Some(6)),
        ] {
            let drafter = write_tiny_dflash(&TinyDflash {
                block,
                window,
                ..TinyDflash::default()
            });
            for k in [1usize, 2, 4, 15] {
                for prompt in prompts {
                    let mut plain = engine(&model, 32, 0.0, None);
                    let mut spec = engine_with_dflash(&model, &drafter, 32, 0.0, None, k, 1);
                    let expected = run(&mut plain, prompt, None);
                    let got = run(&mut spec, prompt, None);
                    let what = format!("block {block} window {window:?} k {k} prompt {prompt:?}");
                    assert_same_reply(&got, &expected, &what);
                    assert_eq!(got.items.len(), expected.items.len(), "{what}");
                    assert_eq!(spec.last_ids, plain.last_ids, "{what}");
                    assert_eq!(dflash_held(&spec).1, spec.last_ids.len(), "{what}");
                    let stats = spec.stats.snapshot();
                    assert!(stats.drafted <= stats.rounds * (block as u64 - 1), "{what}");
                    drafted += stats.drafted;
                    accepted += stats.accepted;
                }
            }
        }
        assert!(drafted > 0, "the drafter proposed tokens");
        assert!(accepted > 0, "some drafts were accepted and rolled forward");
    }

    #[test]
    fn dflash_drafter_respects_max_tokens_and_the_context_limit() {
        let model = muse_target(32);
        let drafter = write_tiny_dflash(&TinyDflash::default());
        for max in [1u32, 2, 3, 7] {
            let mut plain = engine(&model, 32, 0.0, None);
            let mut spec = engine_with_dflash(&model, &drafter, 32, 0.0, None, 4, 1);
            let a = run(&mut plain, &PROMPT, Some(max));
            let b = run(&mut spec, &PROMPT, Some(max));
            assert_same_reply(&a, &b, &format!("max {max}"));
            assert_eq!(dflash_held(&spec).1, spec.last_ids.len(), "max {max}");
        }
        // A context that fills up mid-round, for several sizes and prompts.
        let mut filled = 0;
        for ctx in [7usize, 8, 9, 10, 12, 16] {
            let model = muse_target(ctx.max(8));
            for prompt in [
                &PROMPT[..],
                &[2u32, 9, 9, 9, 10][..],
                &[2u32, 11][..],
                &[2u32, 12, 10, 14][..],
                &[2u32, 13, 12, 11, 10][..],
            ] {
                let mut plain = engine(&model, ctx, 0.0, None);
                let mut spec = engine_with_dflash(&model, &drafter, ctx, 0.0, None, 8, 1);
                let a = run(&mut plain, prompt, None);
                let b = run(&mut spec, prompt, None);
                assert_eq!(text(&a), text(&b), "ctx {ctx}, prompt {prompt:?}");
                assert_eq!(
                    spec.last_ids, plain.last_ids,
                    "ctx {ctx}, prompt {prompt:?}"
                );
                assert_eq!(dflash_held(&spec).1, spec.last_ids.len(), "ctx {ctx}");
                filled +=
                    usize::from(completion_tokens(&b) == u32::try_from(ctx - prompt.len()).ok());
            }
        }
        assert!(filled > 0, "some run stopped at the context limit");
    }

    /// Warm prefix reuse with the drafter equals a cold run: the trunk
    /// resumes from the shared blocks, the drafter keeps the positions the
    /// shared tokens still determine and observes the replayed rows.
    #[test]
    fn dflash_warm_prefix_reuse_equals_cold() {
        let model = muse_target(128);
        let drafter = write_tiny_dflash(&TinyDflash::default());
        let first = long_prompt(40);
        let mut second = first[..30].to_vec();
        second.extend([14, 15, 13, 12, 11, 10, 9, 12, 13, 14]);
        let mut warm = engine_with_dflash(&model, &drafter, 128, 0.0, None, 3, 1);
        let a = run(&mut warm, &first, Some(6));
        assert_eq!(usage(&a), (0, 40));
        assert_eq!(dflash_held(&warm).1, warm.last_ids.len());

        // The first 30 tokens are shared: one whole block is reused.
        let b = run(&mut warm, &second, Some(6));
        assert_eq!(usage(&b), (16, 24));
        let cold = run(
            &mut engine_with_dflash(&model, &drafter, 128, 0.0, None, 3, 1),
            &second,
            Some(6),
        );
        let plain = run(&mut engine(&model, 128, 0.0, None), &second, Some(6));
        assert_eq!(text(&b), text(&cold));
        assert_eq!(text(&b), text(&plain));
        assert_eq!(dflash_held(&warm).1, warm.last_ids.len());

        // The same prompt again reuses its two whole blocks.
        let c = run(&mut warm, &second, Some(6));
        assert_eq!(usage(&c), (32, 8));
        assert_eq!(text(&c), text(&plain));
        assert_eq!(dflash_held(&warm).1, warm.last_ids.len());

        // The cached ids plus more tokens.
        let mut extended = warm.last_ids.clone();
        extended.extend([15, 14]);
        let d = run(&mut warm, &extended, Some(4));
        let expected = run(&mut engine(&model, 128, 0.0, None), &extended, Some(4));
        assert_eq!(text(&d), text(&expected));
        assert_eq!(dflash_held(&warm).1, warm.last_ids.len());

        // A prompt that diverges inside the second block keeps the first.
        let mut diverging = first[..18].to_vec();
        diverging.extend([15, 14, 13, 12, 11, 10, 9, 15, 14, 13, 12, 11]);
        let e = run(&mut warm, &diverging, Some(5));
        assert_eq!(usage(&e), (16, 14));
        let expected = run(&mut engine(&model, 128, 0.0, None), &diverging, Some(5));
        assert_eq!(text(&e), text(&expected));
        assert_eq!(dflash_held(&warm).1, warm.last_ids.len());
    }

    /// A failed request resets the drafter with the trunk, and the engine
    /// keeps serving.
    #[test]
    fn dflash_drafter_recovers_from_a_failed_request() {
        let model = muse_target(32);
        let drafter = write_tiny_dflash(&TinyDflash::default());
        let mut e = engine_with_dflash(&model, &drafter, 32, 0.0, None, 3, 1);
        let prompt = [2u32, 9, 10, 3, 11, 12, 13];
        let before = run(&mut e, &prompt, Some(4));
        let out = run(&mut e, &[2, 9, 10, 3, 99], Some(2));
        assert!(matches!(&out.items[0], Err(CucaError::Provider { .. })));
        assert!(e.last_ids.is_empty());
        assert_eq!(dflash_held(&e), (0, 0));
        let after = run(&mut e, &prompt, Some(4));
        assert_eq!(usage(&after), (0, 7));
        assert_eq!(text(&after), text(&before));
        assert_eq!(dflash_held(&e).1, e.last_ids.len());
    }

    #[test]
    fn dflash_drafter_with_sampling_is_reproducible_for_a_seed() {
        let model = muse_target(32);
        let drafter = write_tiny_dflash(&TinyDflash::default());
        seeded_run_is_reproducible(|| engine_with_dflash(&model, &drafter, 32, 1.0, Some(5), 3, 1));
    }

    /// The measured cost model and governor (no test freeze) price a block
    /// drafter per round: the output is still plain greedy's, and the
    /// drafter's cost is recorded once per round, not once per draft.
    #[test]
    fn dflash_drafter_with_the_measured_cost_model_matches_plain() {
        let model = muse_target(128);
        let drafter = write_tiny_dflash(&TinyDflash::default());
        let prompt = long_prompt(20);
        let mut plain = engine(&model, 128, 0.0, None);
        let mut spec = engine_with_dflash(&model, &drafter, 128, 0.0, None, 7, 1);
        spec.governor = std::array::from_fn(|_| Governor::default());
        spec.cost = std::array::from_fn(|_| CostModel::default());
        for _ in 0..6 {
            let a = run(&mut plain, &prompt, Some(24));
            let b = run(&mut spec, &prompt, Some(24));
            assert_eq!(text(&a), text(&b));
        }
        // Drafted rounds were measured: one pass each (not one per draft).
        assert!(spec.cost[0].draft.n > 0, "the cost model measured drafting");
        assert!(spec.stats.snapshot().drafted > 0);
    }

    /// A block drafter's drafts are free once the pass is paid for: the
    /// marginal rule keeps a draft while its acceptance beats the extra
    /// verified row, and the flat pass cost only weighs the whole round.
    #[test]
    fn a_block_draft_is_priced_per_round() {
        // Free rows: every draft with any chance of acceptance is kept.
        assert!(take_block_draft(0.01, 1.0, 1.0, 1.0, 0.0));
        assert!(!take_block_draft(0.0, 1.0, 1.0, 1.0, 0.0));
        // A row costing as much as the whole round: keep it only when its
        // acceptance is above the expected tokens of the round so far.
        assert!(!take_block_draft(0.5, 1.0, 1.0, 2.0, 0.0));
        assert!(take_block_draft(1.1, 1.0, 1.0, 2.0, 0.0));
        // A pass that dominates the round makes extra rows cheap in
        // relative terms: the same draft flips to kept.
        assert!(take_block_draft(0.5, 1.0, 1.0, 2.0, 10.0));
        // The chained rule with the same numbers charges the pass per draft.
        assert!(!take_draft(0.5, 1.0, 2, 1.0, 2.0, 10.0));
    }

    /// The positions the DSpark drafter of `e` holds context for.
    fn dspark_held(e: &Engine) -> (usize, usize) {
        match e.drafters.as_slice() {
            [drafter @ Drafter::Dspark(net, _), ..] => {
                let (lo, hi) = net.held();
                (lo, hi + drafter.pending())
            }
            _ => panic!("the engine has no DSpark drafter"),
        }
    }

    fn ds41() -> crate::provider::local::model::tiny::TinyModel {
        use crate::provider::local::model::tiny::{TinyDeepseek41, write_deepseek41};
        write_deepseek41(&TinyDeepseek41 {
            mxfp4: true,
            ..TinyDeepseek41::default()
        })
    }

    #[test]
    fn dspark_drafter_leaves_greedy_output_unchanged() {
        use crate::provider::local::model::tiny::{TinyDspark, write_tiny_dspark};
        let model = ds41();
        let prompts: [&[u32]; 5] = [
            &PROMPT,
            &[2, 9, 9, 9, 10],
            &[2, 13, 12, 11, 10, 9, 13],
            &[2, 11],
            &[2, 12, 10, 14, 13, 11, 9, 15, 10, 12, 14, 11],
        ];
        let (mut drafted, mut accepted) = (0, 0);
        for mxfp4 in [true, false] {
            let drafter = write_tiny_dspark(&TinyDspark {
                mxfp4,
                ..TinyDspark::default()
            });
            for k in [1usize, 2, 4, 5, 16] {
                for prompt in prompts {
                    let mut plain = engine(&model, 96, 0.0, None);
                    let mut spec = engine_with_dspark(&model, &drafter, 96, 0.0, None, k, 1);
                    let expected = run(&mut plain, prompt, None);
                    let got = run(&mut spec, prompt, None);
                    let what = format!("mxfp4 {mxfp4} k {k} prompt {prompt:?}");
                    assert_same_reply(&got, &expected, &what);
                    assert_eq!(got.items.len(), expected.items.len(), "{what}");
                    assert_eq!(spec.last_ids, plain.last_ids, "{what}");
                    assert_eq!(dspark_held(&spec).1, spec.last_ids.len(), "{what}");
                    let stats = spec.stats.snapshot();
                    assert!(stats.drafted <= stats.rounds * 5, "{what}");
                    drafted += stats.drafted;
                    accepted += stats.accepted;
                }
            }
        }
        assert!(drafted > 0, "the drafter proposed tokens");
        eprintln!("DSPARK|drafted {drafted}|accepted {accepted}");
    }

    #[test]
    fn dspark_drafter_respects_max_tokens_and_the_context_limit() {
        use crate::provider::local::model::tiny::{TinyDspark, write_tiny_dspark};
        let model = ds41();
        let drafter = write_tiny_dspark(&TinyDspark::default());
        for max in [1u32, 2, 3, 7] {
            let mut plain = engine(&model, 96, 0.0, None);
            let mut spec = engine_with_dspark(&model, &drafter, 96, 0.0, None, 4, 1);
            let a = run(&mut plain, &PROMPT, Some(max));
            let b = run(&mut spec, &PROMPT, Some(max));
            assert_same_reply(&a, &b, &format!("max {max}"));
            assert_eq!(dspark_held(&spec).1, spec.last_ids.len(), "max {max}");
        }
        for ctx in [12usize, 14, 20] {
            let mut plain = engine(&model, ctx, 0.0, None);
            let mut spec = engine_with_dspark(&model, &drafter, ctx, 0.0, None, 5, 1);
            let a = run(&mut plain, &PROMPT, None);
            let b = run(&mut spec, &PROMPT, None);
            assert_same_reply(&a, &b, &format!("ctx {ctx}"));
        }
    }

    #[test]
    fn dspark_warm_prefix_reuse_equals_cold() {
        use crate::provider::local::model::tiny::{TinyDspark, write_tiny_dspark};
        let model = ds41();
        let drafter = write_tiny_dspark(&TinyDspark::default());
        let first = long_prompt(40);
        let mut second = first[..36].to_vec();
        second.extend_from_slice(&[10, 11, 12, 13, 14, 15]);
        let mut warm = engine_with_dspark(&model, &drafter, 96, 0.0, None, 4, 1);
        run(&mut warm, &first, Some(8));
        assert_eq!(dspark_held(&warm).1, warm.last_ids.len());
        let w = run(&mut warm, &second, Some(10));
        assert_eq!(dspark_held(&warm).1, warm.last_ids.len());
        let c = run(&mut engine(&model, 96, 0.0, None), &second, Some(10));
        assert_same_reply(&w, &c, "warm == cold");
    }

    #[test]
    fn dspark_drafter_recovers_from_a_failed_request() {
        use crate::provider::local::model::tiny::{TinyDspark, write_tiny_dspark};
        let model = ds41();
        let drafter = write_tiny_dspark(&TinyDspark::default());
        let mut e = engine_with_dspark(&model, &drafter, 96, 0.0, None, 3, 1);
        let prompt = [2u32, 9, 10, 3, 11, 12, 13];
        let before = run(&mut e, &prompt, Some(4));
        let out = run(&mut e, &[2, 9, 10, 3, 99], Some(2));
        assert!(matches!(&out.items[0], Err(CucaError::Provider { .. })));
        assert!(e.last_ids.is_empty());
        assert_eq!(dspark_held(&e), (0, 0));
        let after = run(&mut e, &prompt, Some(4));
        assert_eq!(text(&after), text(&before));
        assert_eq!(dspark_held(&e).1, e.last_ids.len());
    }

    #[test]
    fn dspark_drafter_with_sampling_is_reproducible_for_a_seed() {
        use crate::provider::local::model::tiny::{TinyDspark, write_tiny_dspark};
        let model = ds41();
        let drafter = write_tiny_dspark(&TinyDspark::default());
        // The random target ends many sampled replies at once: reproduce
        // several seeds and require that some of them drafted.
        let mut drafted = 0;
        for seed in 1..25u64 {
            let run_once = || {
                let mut e = engine_with_dspark(&model, &drafter, 96, 1.0, Some(seed), 3, 1);
                let out = run(&mut e, &long_prompt(12), None);
                (text(&out), completion_tokens(&out), e.stats.snapshot())
            };
            let (a, b) = (run_once(), run_once());
            assert_eq!(a, b, "seed {seed}");
            drafted += a.2.drafted;
        }
        assert!(drafted > 0, "a seeded run always drafts");
    }

    /// The measured cost model and governor price the block drafter per round
    /// like DFlash's: the output is still plain greedy's.
    #[test]
    fn dspark_drafter_with_the_measured_cost_model_matches_plain() {
        use crate::provider::local::model::tiny::{TinyDspark, write_tiny_dspark};
        let model = ds41();
        let drafter = write_tiny_dspark(&TinyDspark::default());
        let prompt = long_prompt(20);
        let mut plain = engine(&model, 96, 0.0, None);
        let mut spec = engine_with_dspark(&model, &drafter, 96, 0.0, None, 5, 1);
        spec.governor = std::array::from_fn(|_| Governor::default());
        spec.cost = std::array::from_fn(|_| CostModel::default());
        for _ in 0..6 {
            let a = run(&mut plain, &prompt, Some(24));
            let b = run(&mut spec, &prompt, Some(24));
            assert_eq!(text(&a), text(&b));
        }
        assert!(spec.cost[0].draft.n > 0, "the cost model measured drafting");
        assert!(spec.stats.snapshot().drafted > 0);
    }

    /// A DSpark file is a `dflash` file, told apart by its stream count, and
    /// observes every position like a DFlash one.
    #[test]
    fn a_dspark_file_is_recognised_by_its_hyper_connections() {
        use crate::provider::local::model::cpu::dspark::is_dspark;
        use crate::provider::local::model::tiny::{TinyDspark, write_tiny_dspark};
        let dspark = write_tiny_dspark(&TinyDspark::default());
        let content = gguf::Content::open(&dspark.path).unwrap();
        assert!(is_dspark(&content.metadata));
        assert_eq!(architecture_of(&content), "dflash");
        let dflash = write_tiny_dflash(&TinyDflash::default());
        let content = gguf::Content::open(&dflash.path).unwrap();
        assert!(!is_dspark(&content.metadata));
    }

    /// A `dflash` file keeps the configured batch size (its drafter forks one
    /// cache per state slot); `Engine::open` dispatches a drafter file on its
    /// architecture, and a `dflash` file is refused as a main model. (The
    /// tiny GGUFs carry no tokenizer, so what is checked is everything `open`
    /// does before it.)
    #[test]
    fn a_dflash_file_is_dispatched_by_its_architecture() {
        let config = LocalModelConfig {
            context_length: Some(32),
            batching: super::super::BatchingConfig {
                max_sequences: 4,
                ..Default::default()
            },
            draft_tokens: 4,
            ..LocalModelConfig::default()
        };
        let (batching, sizing) = resolve_batching(&config, 32).unwrap();
        assert_eq!((batching.max_sequences, sizing.max_sequences), (4, 4));

        let model = muse_target(32);
        let open = |drafter: &crate::provider::local::model::tiny::TinyModel| {
            Engine::open(
                &model.path,
                Some(&ResolvedDrafter::File(drafter.path.clone())),
                Arc::default(),
                &config,
                None,
            )
            .err()
            .expect("the tiny model has no tokenizer")
            .to_string()
        };
        // A fitting drafter loads: `open` gets as far as the tokenizer.
        let e = open(&write_tiny_dflash(&TinyDflash::default()));
        assert!(e.contains("tokenizer.ggml.model"), "{e}");
        // A drafter that reads a layer the target lacks fails while loading.
        let e = open(&write_tiny_dflash(&TinyDflash {
            target_ids: vec![0, 9],
            ..TinyDflash::default()
        }));
        assert!(e.contains("layer 9"), "{e}");

        // The file is a drafter, not a model.
        let drafter = write_tiny_dflash(&TinyDflash::default());
        let err = Engine::open(&drafter.path, None, Arc::default(), &config, None)
            .err()
            .expect("a drafter is not a main model");
        assert!(err.to_string().contains("not a standalone model"), "{err}");
    }

    // ---- lazy drafter upkeep ------------------------------------------------

    /// The committed pair count and latent K/V rows of the EAGLE drafter of
    /// `e`: what the drafter has taken into its cache, not what it holds back.
    fn eagle_cache(e: &Engine) -> (usize, Vec<Vec<f32>>) {
        match e.drafters.as_slice() {
            [Drafter::Eagle(head, _), ..] => (head.len(), head.kv_prefix()),
            _ => panic!("the engine has no EAGLE drafter"),
        }
    }

    /// Rows the drafter of `e` holds back.
    fn pending(e: &Engine) -> usize {
        e.drafters[0].pending()
    }

    /// Make the drafter of `e` take the rows it holds back into its cache.
    fn catch_up(e: &mut Engine) {
        let drafter = &mut e.drafters[0];
        drafter.catch_up(&e.model).unwrap();
    }

    /// Rows handed to a drafter round by round (`defer`) and taken in one
    /// pass (`catch_up`) leave its cache as observing them as they came
    /// would, bit for bit, whatever the chunking and the cap that forces a
    /// pass early; the held rows follow truncation, replay and reset.
    #[test]
    fn eagle_deferred_rows_catch_up_to_the_eager_cache() {
        use crate::provider::local::model::tiny::{Lcg, N_EMBD};
        let (model, eagle) = (mistral4(128), eagle());
        let n = 40;
        let tokens: Vec<u32> = (0..n).map(|i| (i * 5 % 13 + 3) as u32).collect();
        let hidden = Lcg(9).values(n * N_EMBD, 1.0);
        let rows = |a: usize, b: usize| (&tokens[a..b], &hidden[a * N_EMBD..b * N_EMBD]);
        let fresh = || engine_with_eagle(&model, &eagle, 128, 0.0, None, 3, 1);

        let mut eager = fresh();
        let (t, h) = rows(0, n);
        eager.drafters[0].observe(&eager.model, 0, t, h).unwrap();
        let want = eagle_cache(&eager);
        assert_eq!(want.0, n - 1);

        // A prompt of four rows, then rounds of one or three rows.
        for cap in [None, Some(7), Some(1)] {
            let mut lazy = fresh();
            let d = &mut lazy.drafters[0];
            if let Some(cap) = cap {
                d.set_lag_cap(cap);
            }
            let (t, h) = rows(0, 4);
            d.observe(&lazy.model, 0, t, h).unwrap();
            let mut at = 4;
            while at < n {
                let step = (if at % 5 == 0 { 3 } else { 1 }).min(n - at);
                let (t, h) = rows(at, at + step);
                d.defer(&lazy.model, at, t, h).unwrap();
                at += step;
                assert!(d.pending() <= cap.unwrap_or(1024).max(3), "cap {cap:?}");
            }
            if cap.is_none() {
                assert_eq!(d.pending(), n - 4);
                assert_eq!(eagle_cache(&lazy).0, 3, "only the prompt was observed");
            }
            catch_up(&mut lazy);
            assert_eq!(pending(&lazy), 0);
            assert!(eagle_cache(&lazy) == want, "cap {cap:?}");
            // Nothing is left to take.
            catch_up(&mut lazy);
            assert!(eagle_cache(&lazy) == want, "cap {cap:?}");
        }

        // Truncation drops the held rows the new prompt does not share, and
        // the replay from an earlier position replaces the ones it covers:
        // the result is a cold drafter over the final sequence.
        let mut lazy = fresh();
        let d = &mut lazy.drafters[0];
        let (t, h) = rows(0, 4);
        d.observe(&lazy.model, 0, t, h).unwrap();
        for at in 4..30 {
            let (t, h) = rows(at, at + 1);
            d.defer(&lazy.model, at, t, h).unwrap();
        }
        assert_eq!(d.pending(), 26);
        d.truncate(24);
        assert_eq!(d.pending(), 20, "positions 24 and up are gone");
        let replay_tokens: Vec<u32> = (16..28).map(|i| (i * 3 % 11 + 2) as u32).collect();
        let replay_hidden = Lcg(21).values(12 * N_EMBD, 1.0);
        d.observe(&lazy.model, 16, &replay_tokens, &replay_hidden)
            .unwrap();
        assert_eq!(d.pending(), 0, "the replay took everything in one go");
        let mut cold = fresh();
        let mut seq_tokens = tokens[..16].to_vec();
        seq_tokens.extend_from_slice(&replay_tokens);
        let mut seq_hidden = hidden[..16 * N_EMBD].to_vec();
        seq_hidden.extend_from_slice(&replay_hidden);
        cold.drafters[0]
            .observe(&cold.model, 0, &seq_tokens, &seq_hidden)
            .unwrap();
        assert!(eagle_cache(&lazy) == eagle_cache(&cold));

        // A round that does not follow the held rows takes them first.
        let mut gap = fresh();
        let d = &mut gap.drafters[0];
        let (t, h) = rows(0, 4);
        d.observe(&gap.model, 0, t, h).unwrap();
        let (t, h) = rows(4, 12);
        d.defer(&gap.model, 4, t, h).unwrap();
        let (t, h) = rows(30, 31);
        d.defer(&gap.model, 30, t, h).unwrap();
        assert_eq!(d.pending(), 1);
        catch_up(&mut gap);
        let mut skipped = fresh();
        let d = &mut skipped.drafters[0];
        let (t, h) = rows(0, 12);
        d.observe(&skipped.model, 0, t, h).unwrap();
        let (t, h) = rows(30, 31);
        d.observe(&skipped.model, 30, t, h).unwrap();
        assert!(eagle_cache(&gap) == eagle_cache(&skipped));

        // Truncating below the held rows empties them, and a reset forgets
        // them with everything else.
        let mut gone = fresh();
        let d = &mut gone.drafters[0];
        let (t, h) = rows(0, 4);
        d.observe(&gone.model, 0, t, h).unwrap();
        let (t, h) = rows(4, 9);
        d.defer(&gone.model, 4, t, h).unwrap();
        d.truncate(3);
        assert_eq!(d.pending(), 0);
        let (t, h) = rows(4, 9);
        d.defer(&gone.model, 4, t, h).unwrap();
        assert_eq!(d.pending(), 5);
        d.reset();
        assert_eq!(d.pending(), 0);
        assert_eq!(eagle_cache(&gone).0, 0);
    }

    /// The committed pairs and K/V rows of the in-trunk head of `e`.
    fn head_cache(e: &Engine) -> (usize, Vec<Vec<f32>>) {
        match e.drafters.as_slice() {
            [Drafter::InTrunk(head, _), ..] => (head.len(), head.kv_prefix()),
            _ => panic!("the engine has no in-trunk head"),
        }
    }

    /// The positions the DFlash drafter of `e` holds context for (`hi`) and
    /// its whole ring: the first held position and every K and V row.
    fn dflash_cache(e: &Engine) -> (usize, Vec<Vec<f32>>) {
        match e.drafters.as_slice() {
            [Drafter::Dflash(net, _), ..] => {
                let (lo, hi, mut rows, v) = net.snapshot();
                rows.extend(v);
                rows.push(vec![lo as f32]);
                (hi, rows)
            }
            _ => panic!("the engine has no DFlash drafter"),
        }
    }

    /// A drafter with a per-token upkeep, as the lazy-upkeep test drives it.
    struct Upkeep<'a> {
        name: &'static str,
        /// An engine over the same target that never drafts.
        plain: Box<dyn Fn() -> Engine + 'a>,
        /// An engine that drafts three tokens in every round.
        spec: Box<dyn Fn() -> Engine + 'a>,
        /// The positions the drafter has taken into its cache, and the cache.
        cache: fn(&Engine) -> (usize, Vec<Vec<f32>>),
        /// How many positions of the cached ids the drafter lacks once it
        /// has taken everything in: a drafter of pairs holds one fewer, the
        /// last row waits for its successor token.
        behind: usize,
    }

    /// An engine that runs plain rounds for a while and then drafts decodes
    /// the plain text and leaves the drafter's cache what an engine that
    /// drafted all along leaves, bit for bit: the plain rounds hand the
    /// drafter their rows without taking them in (`pending`), and the first
    /// round that drafts takes them in one pass. A following request that
    /// replays the cached ids starts from the rows still held.
    #[test]
    fn plain_rounds_defer_the_upkeep_and_the_next_draft_catches_up() {
        let (mistral, eagle) = (mistral4(128), eagle());
        let qwen = hybrid(TinyQwen35::default());
        let (muse, dflash) = (muse_target(128), write_tiny_dflash(&TinyDflash::default()));
        let cases = [
            Upkeep {
                name: "EAGLE",
                plain: Box::new(|| engine(&mistral, 128, 0.0, None)),
                spec: Box::new(|| engine_with_eagle(&mistral, &eagle, 128, 0.0, None, 3, 1)),
                cache: eagle_cache,
                behind: 1,
            },
            Upkeep {
                name: "in-trunk head",
                plain: Box::new(|| engine(&qwen, 128, 0.0, None)),
                spec: Box::new(|| engine_in_trunk(&qwen, 128, 0.0, None, 3, 1)),
                cache: head_cache,
                behind: 1,
            },
            Upkeep {
                name: "DFlash",
                plain: Box::new(|| engine(&muse, 128, 0.0, None)),
                spec: Box::new(|| engine_with_dflash(&muse, &dflash, 128, 0.0, None, 3, 1)),
                cache: dflash_cache,
                behind: 0,
            },
        ];
        for case in &cases {
            lazy_upkeep_is_exact(case);
        }
    }

    fn lazy_upkeep_is_exact(case: &Upkeep<'_>) {
        let name = case.name;
        let prompt = long_prompt(20);
        let mut plain = (case.plain)();
        let expected = run(&mut plain, &prompt, Some(48));

        let mut eager = (case.spec)();
        let a = run(&mut eager, &prompt, Some(48));
        assert_eq!(text(&a), text(&expected), "{name}");
        assert_eq!(
            pending(&eager),
            0,
            "{name}: a drafting engine observes at once"
        );
        let eager_cache = (case.cache)(&eager);
        assert_eq!(eager_cache.0, eager.last_ids.len() - case.behind, "{name}");

        let mut drafted_after_plain = 0;
        for forced in [0u32, 1, 3, 10, 30, 1000] {
            let what = format!("{name}, {forced} plain rounds");
            let mut lazy = (case.spec)();
            lazy.governor[0] = Governor {
                forced,
                forced_spec: false,
                ..Governor::always_speculating()
            };
            let b = run(&mut lazy, &prompt, Some(48));
            assert_eq!(text(&b), text(&expected), "{what}");
            assert_eq!(lazy.last_ids, plain.last_ids, "{what}");
            assert_eq!(
                lazy.drafters[0].stale_drafts(),
                0,
                "{what}: a round drafted before the drafter caught up"
            );
            assert_eq!(
                (case.cache)(&lazy).0 + pending(&lazy),
                lazy.last_ids.len() - case.behind,
                "{what}"
            );
            if forced >= 48 {
                // Every round was plain: the drafter saw the prompt only.
                assert_eq!(lazy.stats.snapshot().drafted, 0, "{what}");
                assert_eq!(pending(&lazy), lazy.last_ids.len() - prompt.len(), "{what}");
                assert_eq!((case.cache)(&lazy).0, prompt.len() - case.behind, "{what}");
            } else {
                drafted_after_plain += lazy.stats.snapshot().drafted;
            }
            catch_up(&mut lazy);
            assert_eq!(pending(&lazy), 0, "{what}");
            assert!(
                (case.cache)(&lazy) == eager_cache,
                "{what}: the cache differs from the eager one"
            );

            // The same schedule with the upkeep done round by round: when
            // the drafter drafts, it holds the cache it would hold, so it
            // proposes what the lazy one proposes and the rounds are the
            // same ones.
            let mut reference = (case.spec)();
            reference.drafters[0].set_eager(true);
            reference.governor[0] = Governor {
                forced,
                forced_spec: false,
                ..Governor::always_speculating()
            };
            let r = run(&mut reference, &prompt, Some(48));
            assert_eq!(text(&r), text(&expected), "{what}");
            assert_eq!(pending(&reference), 0, "{what}");
            assert_eq!(
                reference.stats.snapshot(),
                lazy.stats.snapshot(),
                "{what}: the proposals differ from the eager drafter's"
            );
        }
        assert!(
            drafted_after_plain > 0,
            "{name}: the rounds after the plain ones draft"
        );

        // The next request replays the cached ids from the rows still held:
        // the drafter ends as one that never deferred.
        let mut lazy = (case.spec)();
        lazy.governor[0] = Governor {
            forced: 1000,
            forced_spec: false,
            ..Governor::always_speculating()
        };
        run(&mut lazy, &prompt, Some(48));
        assert!(pending(&lazy) > 0, "{name}");
        lazy.governor[0] = Governor::always_speculating();
        let mut extended = lazy.last_ids.clone();
        extended.extend([15, 14, 13]);
        let c = run(&mut lazy, &extended, Some(20));
        let expected = run(&mut plain, &extended, Some(20));
        assert_eq!(text(&c), text(&expected), "{name}");
        let d = run(&mut eager, &extended, Some(20));
        assert_eq!(text(&d), text(&expected), "{name}");
        catch_up(&mut lazy);
        assert!((case.cache)(&lazy) == (case.cache)(&eager), "{name}");
    }

    // ---- K/V transfer ------------------------------------------------------

    /// Export the prefix of `prompt` (from `from`) out of `p`: the lease, every
    /// frame the peer reads (the rows, the snapshot of a hybrid model, the
    /// end) and the receiver whose drop ends the lease.
    fn export_frames(
        p: &mut Engine,
        prompt: &[u32],
        from: usize,
    ) -> (
        ExportReady,
        Vec<Vec<u8>>,
        tokio::sync::oneshot::Receiver<()>,
    ) {
        use tokio::sync::oneshot::{self, error::TryRecvError};
        let (alive_tx, alive_rx) = oneshot::channel::<()>();
        let (reply_tx, mut reply_rx) = oneshot::channel();
        p.export_begin(prompt.to_vec(), from, alive_tx, reply_tx)
            .unwrap();
        let mut turns = 0;
        let ready: ExportReady = loop {
            match reply_rx.try_recv() {
                Ok(reply) => break reply.unwrap(),
                Err(TryRecvError::Empty) => {
                    turns += 1;
                    assert!(turns < 10_000, "the export never finished");
                    p.turn();
                }
                Err(e) => panic!("the export's reply: {e}"),
            }
        };
        let mut frames = Vec::new();
        let mut a = ready.from;
        while a < ready.to {
            let b = (a + ready.chunk).min(ready.to);
            frames.push(p.export_rows(ready.id, a, b).unwrap());
            a = b;
        }
        if ready.snapshot {
            frames.push(p.export_snapshot(ready.id).unwrap());
        }
        frames.push(crate::provider::local::kvframe::end_frame());
        (ready, frames, alive_rx)
    }

    /// Engine D imports what engine P exports for `prompt` and then serves
    /// the prompt: it reads the whole imported prefix, re-exports exactly the
    /// rows it took in, and replies what a fresh engine replies.
    fn imported_prefix_equals_local_prefill(
        name: &str,
        model: &crate::provider::local::model::tiny::TinyModel,
        ctx: usize,
        len: usize,
    ) {
        let prompt = long_prompt(len);
        let make = || engine_sized(model, ctx, 0.0, None, 2, None).with_exports(2);
        let mut fresh = make();
        let want = run(&mut fresh, &prompt, Some(10));

        let mut p = make();
        let (ready, frames, p_alive) = export_frames(&mut p, &prompt, 0);
        assert!(ready.to > 0 && ready.to < prompt.len(), "{name}: {ready:?}");

        let mut d = make();
        let (d_alive, _d_keep) = tokio::sync::oneshot::channel::<()>();
        let id = d
            .import_begin(prompt.clone(), 0, ready.to, d_alive)
            .unwrap();
        for (i, frame) in frames.iter().enumerate() {
            let ended = d.import_frame(id, frame).unwrap();
            assert_eq!(ended, i + 1 == frames.len(), "{name}: frame {i}");
        }
        assert!(d.imports.is_empty(), "{name}");
        assert_eq!(used_blocks(&d), 0, "{name}: the import is filed, not held");
        assert_eq!(d.kv_hit(&prompt), ready.to, "{name}");

        // What D holds is what P sent, row for row (a window group sends only
        // its window).
        let (again, again_frames, d_alive_again) = export_frames(&mut d, &prompt, 0);
        assert_eq!((again.from, again.to), (ready.from, ready.to), "{name}");
        assert!(again_frames == frames, "{name}: D re-exports other rows");
        drop(d_alive_again);
        d.turn();
        assert!(d.exports.is_empty(), "{name}");

        let got = run(&mut d, &prompt, Some(10));
        assert_eq!(
            usage(&got),
            (ready.to as u32, (prompt.len() - ready.to) as u32),
            "{name}: D read the imported prefix"
        );
        assert_eq!(text(&got), text(&want), "{name}");
        assert_eq!(d.last_ids, fresh.last_ids, "{name}");

        // The lease ends with its handle, its blocks cached.
        assert_eq!(p.exports.len(), 1, "{name}");
        drop(p_alive);
        p.turn();
        assert!(p.exports.is_empty(), "{name}");
        assert_eq!(used_blocks(&p), 0, "{name}");
    }

    #[test]
    fn an_imported_prefix_decodes_like_a_local_prefill() {
        use crate::provider::local::model::tiny::{TinyDeepseek41, write_deepseek41};
        for spec in TinySpec::all() {
            let name = spec.arch;
            let model = write_spec(&TinySpec {
                context: 128,
                ..spec
            });
            imported_prefix_equals_local_prefill(name, &model, 128, 100);
        }
        imported_prefix_equals_local_prefill("gemma4", &write_tiny_model(), 32, 28);
        imported_prefix_equals_local_prefill("qwen35", &hybrid_model(), 128, 100);
        imported_prefix_equals_local_prefill("mistral4", &mistral4(128), 128, 100);
        let ds41 = write_deepseek41(&TinyDeepseek41 {
            meta: vec![("context_length".into(), gguf::Value::U32(128))],
            ..TinyDeepseek41::default()
        });
        imported_prefix_equals_local_prefill("deepseek41", &ds41, 128, 100);
    }

    /// A second import carries only what the first left: `from` is what the
    /// store holds, and the prefix ends where the whole prompt's does.
    #[test]
    fn an_import_starts_after_the_prefix_the_store_holds() {
        let model = write_spec(&TinySpec {
            context: 160,
            ..TinySpec::llama()
        });
        let make = || engine_sized(&model, 160, 0.0, None, 2, None).with_exports(2);
        let long = long_prompt(130);
        let mut fresh = make();
        let want = run(&mut fresh, &long, Some(8));

        let mut p = make();
        let mut d = make();
        // First 64 tokens of the prompt, then the whole prompt.
        let first = &long[..65];
        for (prompt, expect_from) in [(first, 0), (&long[..], 64)] {
            let from = d.kv_hit(prompt);
            assert_eq!(from, expect_from);
            let (ready, frames, _alive) = export_frames(&mut p, prompt, from);
            assert_eq!(ready.from, from);
            let (d_alive, _d_keep) = tokio::sync::oneshot::channel::<()>();
            let id = d
                .import_begin(prompt.to_vec(), from, ready.to, d_alive)
                .unwrap();
            for frame in &frames {
                d.import_frame(id, frame).unwrap();
            }
        }
        let got = run(&mut d, &long, Some(8));
        assert_eq!(usage(&got).0, 128);
        assert_eq!(text(&got), text(&want));
    }

    #[test]
    fn an_export_is_refused_when_it_cannot_run() {
        use tokio::sync::oneshot;
        let model = write_spec(&TinySpec {
            context: 128,
            ..TinySpec::llama()
        });
        let prompt = long_prompt(100);
        let try_export = |e: &mut Engine, ids: Vec<u32>, from: usize| {
            let (alive, _alive_rx) = oneshot::channel();
            let (reply, _reply_rx) = oneshot::channel();
            e.export_begin(ids, from, alive, reply)
        };
        // Not configured.
        let mut plain = engine_sized(&model, 128, 0.0, None, 2, None);
        let err = try_export(&mut plain, prompt.clone(), 0).unwrap_err();
        assert!(
            matches!(&err, CucaError::Config(m) if m.contains("kv_export")),
            "{err}"
        );
        let mut e = engine_sized(&model, 128, 0.0, None, 2, None).with_exports(1);
        // The prompt ends at 96; an unaligned or empty range, a prompt the
        // context cannot hold, and an empty one.
        for from in [8, 96, 200] {
            let err = try_export(&mut e, prompt.clone(), from).unwrap_err();
            assert!(
                err.to_string().contains("nothing to export"),
                "{from}: {err}"
            );
        }
        let err = try_export(&mut e, long_prompt(128), 0).unwrap_err();
        assert!(err.to_string().contains("context_length"), "{err}");
        let err = try_export(&mut e, Vec::new(), 0).unwrap_err();
        assert!(err.to_string().contains("context_length"), "{err}");
        // One slot: the first export runs, the second is refused.
        let (alive, _keep_alive) = oneshot::channel();
        let (reply, _keep_reply) = oneshot::channel();
        e.export_begin(prompt.clone(), 0, alive, reply).unwrap();
        let err = try_export(&mut e, prompt, 0).unwrap_err();
        assert!(err.to_string().contains("busy"), "{err}");
    }

    #[test]
    fn a_bad_frame_ends_an_import_and_gives_its_blocks_back() {
        use crate::provider::local::kvframe;
        let model = write_spec(&TinySpec {
            context: 128,
            ..TinySpec::llama()
        });
        let prompt = long_prompt(100);
        let mut p = engine_sized(&model, 128, 0.0, None, 2, None).with_exports(1);
        let (ready, frames, _alive) = export_frames(&mut p, &prompt, 0);
        let mut d = engine_sized(&model, 128, 0.0, None, 2, None);
        let begin = |d: &mut Engine| {
            let (alive, keep) = tokio::sync::oneshot::channel();
            let id = d.import_begin(prompt.clone(), 0, ready.to, alive).unwrap();
            (id, keep)
        };
        // An end before the rows.
        let (id, _keep) = begin(&mut d);
        assert!(used_blocks(&d) > 0, "the import holds its blocks");
        let err = d.import_frame(id, &kvframe::end_frame()).unwrap_err();
        assert!(err.to_string().contains("ended at position 0"), "{err}");
        assert!(d.imports.is_empty());
        assert_eq!(used_blocks(&d), 0);
        // Rows twice: the second does not follow the first.
        let (id, _keep) = begin(&mut d);
        assert!(!d.import_frame(id, &frames[0]).unwrap());
        let err = d.import_frame(id, &frames[0]).unwrap_err();
        assert!(err.to_string().contains("do not follow"), "{err}");
        assert_eq!(used_blocks(&d), 0);
        // Nothing was filed: a request reads nothing.
        assert_eq!(d.kv_hit(&prompt), 0);
        // A frame of an import that is gone, and a malformed one.
        assert!(d.import_frame(id, &frames[0]).is_err());
        let (id, _keep) = begin(&mut d);
        assert!(d.import_frame(id, &[2, 1, 2]).is_err());
        assert_eq!(used_blocks(&d), 0);
        // A dropped handle gives the blocks back at the next turn.
        let (_id, keep) = begin(&mut d);
        assert!(used_blocks(&d) > 0);
        drop(keep);
        d.turn();
        assert!(d.imports.is_empty());
        assert_eq!(used_blocks(&d), 0);
    }

    /// The fingerprint follows the file and the cache layout: the same file
    /// twice gives the same value, another model or a changed layout another.
    #[test]
    fn the_fingerprint_follows_the_file_and_the_cache_layout() {
        let fingerprint =
            |model: &crate::provider::local::model::tiny::TinyModel,
             edit: &dyn Fn(&mut crate::provider::local::model::paged::KvLayout)| {
                let content = gguf::Content::open(&model.path).unwrap();
                let hp = super::super::model::read_hparams(&content).unwrap();
                let e = engine(model, 32, 0.0, None);
                let mut layout = e.model.kv_layout().clone();
                edit(&mut layout);
                let bytes = std::fs::metadata(&model.path).unwrap().len();
                model_fingerprint(&content, bytes, hp.n_layer, 16, &layout)
            };
        let llama = write_spec(&TinySpec::llama());
        let gemma3 = write_spec(&TinySpec::gemma3());
        let none = |_: &mut crate::provider::local::model::paged::KvLayout| {};
        let base = fingerprint(&llama, &none);
        assert_eq!(base, fingerprint(&llama, &none));
        assert_ne!(base, fingerprint(&gemma3, &none));
        // Another group layout of the same file (window, stride, layers).
        assert_ne!(
            base,
            fingerprint(&llama, &|l| l.groups[0].window = Some(64))
        );
        assert_ne!(base, fingerprint(&llama, &|l| l.groups[0].stride = 2));
        assert_ne!(
            base,
            fingerprint(&llama, &|l| l.groups[0]
                .layers
                .pop()
                .map(drop)
                .unwrap_or(()))
        );
        // The pool sizes are not part of it.
        assert_eq!(
            base,
            fingerprint(&llama, &|l| l.blocks.iter_mut().for_each(|b| *b += 3))
        );
    }

    /// A snapshot that does not fit the model fails a hybrid model's import
    /// when it arrives: nothing is filed, and the blocks are given back.
    #[test]
    fn a_snapshot_that_does_not_fit_fails_the_import() {
        use crate::provider::local::kvframe;
        use crate::provider::local::model::transfer::DeltaRows;
        let model = hybrid_model();
        let prompt = long_prompt(100);
        let mut p = engine_sized(&model, 128, 0.0, None, 2, None).with_exports(1);
        let (ready, frames, _alive) = export_frames(&mut p, &prompt, 0);
        assert!(ready.snapshot);
        let snapshot = frames
            .iter()
            .find(|f| kvframe::parse(f).is_ok_and(|(kind, _)| kind == kvframe::SNAPSHOT))
            .expect("a hybrid export carries its snapshot");
        let (_, payload) = kvframe::parse(snapshot).unwrap();
        let layers = kvframe::decode_snapshot(payload).unwrap();
        assert!(!layers.is_empty());

        let mut d = engine_sized(&model, 128, 0.0, None, 2, None);
        let (alive, _keep) = tokio::sync::oneshot::channel::<()>();
        let id = d.import_begin(prompt.clone(), 0, ready.to, alive).unwrap();
        for frame in frames
            .iter()
            .take_while(|f| f.as_slice() != snapshot.as_slice())
        {
            assert!(!d.import_frame(id, frame).unwrap());
        }
        // Right count of layers, wrong sizes.
        let wrong: Vec<DeltaRows> = layers
            .iter()
            .map(|_| DeltaRows {
                conv: vec![0.0; 3],
                s: vec![0.0; 5],
            })
            .collect();
        let err = d
            .import_frame(id, &kvframe::snapshot_frame(&wrong))
            .unwrap_err();
        assert!(err.to_string().contains("does not fit"), "{err}");
        assert!(d.imports.is_empty());
        assert_eq!(used_blocks(&d), 0);
        assert_eq!(d.kv_hit(&prompt), 0, "nothing was filed");
    }

    /// A call that panics fails the running sequences and resets the pools
    /// like a step that panics, and the engine keeps serving.
    #[test]
    fn a_panicking_call_does_not_end_the_engine() {
        let model = write_spec(&TinySpec {
            context: 128,
            ..TinySpec::llama()
        });
        let mut e = engine_sized(&model, 128, 0.0, None, 2, None);
        let prompt = long_prompt(20);
        let want = run(&mut e, &prompt, Some(6));
        e.receive(Msg::Call(Box::new(|_| panic!("a call that panics"))));
        let again = run(&mut e, &prompt, Some(6));
        assert_eq!(text(&again), text(&want));
        // The pools were reset: the prompt is no longer cached.
        assert_eq!(usage(&again).0, 0);
    }

    /// The handles on the loopback transports of a chain engine, one per
    /// remote stage.
    #[derive(Default)]
    struct Links {
        ops: Vec<Arc<std::sync::Mutex<Vec<u8>>>>,
        calls: Vec<Arc<AtomicUsize>>,
        fail_at: Vec<Arc<AtomicUsize>>,
        restart_at: Vec<Arc<AtomicUsize>>,
        lose_reply_at: Vec<Arc<AtomicUsize>>,
        lose_every_reply: Vec<Arc<std::sync::atomic::AtomicBool>>,
        workers: Vec<Arc<std::sync::Mutex<Engine>>>,
        last: Vec<super::testing::KeptCall>,
        last_forward: Vec<super::testing::KeptCall>,
        asked: Vec<Arc<std::sync::Mutex<(usize, usize, usize)>>>,
    }

    /// A chain engine over `file` whose remote stages hold the layers up to
    /// each of `bounds` in turn (a worker engine of this process behind
    /// each), and the layers after the last here.
    fn chain_over(
        file: &crate::provider::local::model::tiny::TinyModel,
        context: usize,
        bounds: &[usize],
    ) -> (Engine, Links) {
        chain_over_as(
            file,
            context,
            bounds,
            crate::provider::local::CarryDtype::F32,
        )
    }

    /// [`chain_over`] with every stage's carries in `dtype`.
    fn chain_over_as(
        file: &crate::provider::local::model::tiny::TinyModel,
        context: usize,
        bounds: &[usize],
        dtype: crate::provider::local::CarryDtype,
    ) -> (Engine, Links) {
        use super::testing::{Loopback, chain_engine, stage_worker};
        use crate::provider::local::model::chain::StageTransport;
        let mut first = 0;
        let mut links = Vec::new();
        let mut handles = Links::default();
        for &end in bounds {
            let range = LayerRange { first, end };
            let mut transport = Loopback::new(stage_worker(file, context, range, 1));
            transport.carry = dtype;
            handles.ops.push(Arc::clone(&transport.ops));
            handles.calls.push(Arc::clone(&transport.calls));
            handles.fail_at.push(Arc::clone(&transport.fail_at));
            handles.restart_at.push(Arc::clone(&transport.restart_at));
            handles
                .lose_reply_at
                .push(Arc::clone(&transport.lose_reply_at));
            handles
                .lose_every_reply
                .push(Arc::clone(&transport.lose_every_reply));
            handles.workers.push(Arc::clone(&transport.worker));
            handles.last.push(Arc::clone(&transport.last));
            handles
                .last_forward
                .push(Arc::clone(&transport.last_forward));
            handles.asked.push(Arc::clone(&transport.asked));
            links.push((range, Box::new(transport) as Box<dyn StageTransport>));
            first = end;
        }
        (chain_engine(file, context, links, 0.0, None, 1), handles)
    }

    /// The layer count and the legal stage boundaries of `file`.
    fn cuts_of(file: &crate::provider::local::model::tiny::TinyModel) -> (usize, Vec<usize>) {
        let content = gguf::Content::open(&file.path).unwrap();
        let hp = crate::provider::local::model::read_hparams(&content).unwrap();
        (
            hp.n_layer,
            crate::provider::local::model::split::legal_cuts(&hp),
        )
    }

    /// A chain over `bounds` answers as the whole model does, for every
    /// prompt of `prompts`, in the same order: the same text, the same usage
    /// and cache hits.
    fn chain_equals_whole(
        what: &str,
        file: &crate::provider::local::model::tiny::TinyModel,
        context: usize,
        bounds: &[usize],
        prompts: &[Vec<u32>],
    ) {
        chain_equals_whole_with(what, file, context, bounds, prompts, false);
    }

    /// [`chain_equals_whole`], with `lose_replies`: the transports lose the
    /// first answer of every call after the load (the worker ran the frame
    /// and the answer never came back) and send the frame again.
    fn chain_equals_whole_with(
        what: &str,
        file: &crate::provider::local::model::tiny::TinyModel,
        context: usize,
        bounds: &[usize],
        prompts: &[Vec<u32>],
        lose_replies: bool,
    ) {
        let (mut chain, links) = chain_over(file, context, bounds);
        for flag in &links.lose_every_reply {
            flag.store(lose_replies, Ordering::SeqCst);
        }
        let mut whole = engine(file, context, 0.0, None);
        for (i, prompt) in prompts.iter().enumerate() {
            let got = run(&mut chain, prompt, Some(8));
            let want = run(&mut whole, prompt, Some(8));
            let label = format!("{what}, stages {bounds:?}, prompt {i}");
            assert!(got.items.iter().all(Result::is_ok), "{label}");
            assert!(!text(&want).is_empty(), "{label}: the reply is empty");
            assert_same_reply(&got, &want, &label);
            assert_eq!(usage(&got), usage(&want), "{label}");
        }
        assert_eq!(used_blocks(&chain), 0, "{what}");
        if lose_replies {
            for ((name, counters), calls) in chain.stage_counters().iter().zip(&links.calls) {
                let s = counters.snapshot(name.clone());
                // Every call but the `open` of the load was sent twice.
                let made = calls.load(Ordering::SeqCst) - 1;
                assert!(made > 1, "{what}: the stage served calls");
                assert_eq!(s.retries as usize, made, "{what}");
                assert_eq!(s.errors, 0, "{what}");
            }
        }
    }

    /// A call whose answer is lost and sent again runs once on the worker,
    /// so a chain whose every call is sent twice still answers as the whole
    /// model does: attention-only, Gemma 4 (shared K/V rows), the hybrid
    /// Qwen 3.5 (the restores and snapshots ride in the forward calls) and the
    /// hyper-connection DeepSeek-V4.1 (whose compressor state a second run
    /// of a pass would refuse at an odd position).
    #[test]
    fn a_lost_answer_leaves_the_chain_equal_to_the_whole_model() {
        let llama = write_spec(&TinySpec {
            context: 96,
            ..TinySpec::llama()
        });
        let (n, cuts) = cuts_of(&llama);
        let prompts = [long_prompt(40), long_prompt(33), long_prompt(25)];
        chain_equals_whole_with("llama", &llama, 96, &[cuts[0]], &prompts, true);
        if n >= 3 {
            let bounds = [cuts[0], cuts[1]];
            chain_equals_whole_with("llama", &llama, 96, &bounds, &prompts, true);
        }

        let gemma = write_tiny_model();
        let (_, cuts) = cuts_of(&gemma);
        let prompts = [PROMPT.to_vec(), vec![2, 9, 9, 9, 10]];
        chain_equals_whole_with("gemma4", &gemma, 32, &[cuts[0]], &prompts, true);

        let qwen = hybrid_model();
        let (_, cuts) = cuts_of(&qwen);
        let mut first = long_prompt(40);
        first[19] = 3;
        let mut second = first[..30].to_vec();
        second.extend([14, 15, 13, 12, 11, 10, 9, 12, 13, 14]);
        let mut extended = first.clone();
        extended.extend([15, 14, 13]);
        let prompts = [first, second, extended];
        chain_equals_whole_with("qwen35", &qwen, 128, &[cuts[0]], &prompts, true);

        let deepseek = write_deepseek41(&TinyDeepseek41::default());
        let (_, cuts) = cuts_of(&deepseek);
        let prompts = [long_prompt(70), long_prompt(70)];
        for &cut in &cuts {
            chain_equals_whole_with("deepseek41", &deepseek, 96, &[cut], &prompts, true);
        }
    }

    /// A step costs one call per remote stage: the restore a hybrid model's
    /// sequence starts from, the snapshot it takes at a block boundary and
    /// the Engram rows read ahead ride in the forward call. With chunks of
    /// 16 tokens these cost extra calls of their own before (Qwen 3.5, a
    /// 40-token prompt: 7 calls for 5 steps; DeepSeek-V4.1, 70 tokens: 12
    /// calls for 8 steps).
    #[test]
    fn a_step_costs_one_call_per_stage() {
        use stageframe::{OP_FORWARD, OP_OPEN};
        let qwen = hybrid_model();
        let deepseek = write_deepseek41(&TinyDeepseek41::default());
        let mut first = long_prompt(40);
        first[19] = 3;
        let mut second = first[..30].to_vec();
        second.extend([14, 15, 13, 12, 11, 10, 9, 12, 13, 14]);
        let mut extended = first.clone();
        extended.extend([15, 14, 13]);
        for (what, model, context, prompts) in [
            ("qwen35", &qwen, 128, vec![first, second, extended]),
            (
                "deepseek41",
                &deepseek,
                96,
                vec![long_prompt(70), long_prompt(70)],
            ),
        ] {
            let (_, cuts) = cuts_of(model);
            // The first stage must hold an Engram layer (1 and 4).
            let cut = *cuts.iter().find(|&&c| c > 1).unwrap_or(&cuts[0]);
            let (mut chain, links) = chain_over(model, context, &[cut]);
            chain.batching.prefill_chunk = 16;
            let mut whole = engine(model, context, 0.0, None);
            whole.batching.prefill_chunk = 16;
            for (i, prompt) in prompts.iter().enumerate() {
                let label = format!("{what}, prompt {i}");
                let calls = links.calls[0].load(Ordering::SeqCst);
                let steps = chain.serving.snapshot().steps;
                let got = run(&mut chain, prompt, Some(8));
                let want = run(&mut whole, prompt, Some(8));
                assert_same_reply(&got, &want, &label);
                assert_eq!(usage(&got), usage(&want), "{label}");
                let calls = (links.calls[0].load(Ordering::SeqCst) - calls) as u64;
                let steps = chain.serving.snapshot().steps - steps;
                assert!(steps > 1, "{label}: the prompt takes several steps");
                assert_eq!(calls, steps, "{label}: one call per step");
            }
            let ops = links.ops[0].lock().unwrap().clone();
            assert_eq!(ops[0], OP_OPEN, "{what}");
            assert!(
                ops[1..].iter().all(|&op| op == OP_FORWARD),
                "{what}: no call but forward after the open"
            );
            let (restores, snapshots, prefetches) = *links.asked[0].lock().unwrap();
            if what == "qwen35" {
                assert_eq!(restores, prompts.len(), "a restore per admission");
                assert!(snapshots >= 1, "a snapshot at a block boundary");
                assert!(chain.snapshots.len() > 0, "the chain kept it");
                assert_eq!(prefetches, 0);
            } else {
                assert!(prefetches > 0, "Engram rows read ahead");
                assert_eq!((restores, snapshots), (0, 0));
            }
        }
    }

    /// The worker answers a call it already ran from memory: the reply it
    /// gave, byte for byte, and no state moves. A pass sent again under a
    /// new number is a different matter: an attention-only model repeats it,
    /// a hybrid model refuses it (its recurrent state is past the pass's
    /// start), and a hyper-connection model repeats it or refuses it (its
    /// compressor state is past an odd start), so only the numbering makes
    /// a repeat succeed everywhere. A call no newer than the last, or
    /// numbered 0, is refused.
    #[test]
    fn a_worker_answers_a_repeated_call_from_memory() {
        use stageframe::{Call, decode, forward_frame};
        let llama = write_spec(&TinySpec {
            context: 96,
            ..TinySpec::llama()
        });
        let hybrid = hybrid_model();
        let deepseek = write_deepseek41(&TinyDeepseek41::default());
        for (what, model, context) in [
            ("llama", &llama, 96),
            ("qwen35", &hybrid, 128),
            ("deepseek41", &deepseek, 96),
        ] {
            let (_, cuts) = cuts_of(model);
            let (mut chain, links) = chain_over(model, context, &[cuts[0]]);
            let out = run(&mut chain, &long_prompt(40), Some(4));
            assert!(out.items.iter().all(Result::is_ok), "{what}");
            let mut worker = links.workers[0].lock().unwrap();

            // Whatever the last call was, a repeat of it is answered alike.
            let (frame, reply) = links.last[0].lock().unwrap().clone().unwrap();
            let again = worker.stage_call(&frame).unwrap();
            assert_eq!(&again[..], &reply[..], "{what}: the repeat is replayed");

            // The last pass, sent as a call of its own, is not a repeat.
            let (frame, reply) = links.last_forward[0].lock().unwrap().clone().unwrap();
            let Ok(Call::Forward(mut call)) = decode(&frame) else {
                panic!("{what}: not a forward frame");
            };
            let last_seq = call.seq;
            call.seq = 1_000_000;
            let rerun = worker.stage_call(&forward_frame(&call).unwrap());
            match what {
                // Attention only: positional K/V rows, the same pass again.
                "llama" => assert_eq!(&rerun.unwrap()[..], &reply[..], "{what}"),
                "qwen35" => {
                    let e = rerun.unwrap_err().to_string();
                    assert!(e.contains("recurrent state's position"), "{what}: {e}");
                }
                // Never a different answer: the same one, or a refusal.
                _ => match rerun {
                    Ok(r) => assert_eq!(&r[..], &reply[..], "{what}"),
                    Err(e) => {
                        let e = e.to_string();
                        assert!(e.contains("compressor state"), "{what}: {e}");
                    }
                },
            }

            // Older and zero numbers are refused.
            for seq in [0, last_seq - 1] {
                call.seq = seq;
                let e = worker.stage_call(&forward_frame(&call).unwrap());
                let e = e.unwrap_err().to_string();
                assert!(e.contains("not newer"), "{what}, call {seq}: {e}");
            }
        }
    }

    /// A chain counts every call of each remote stage: as many as the
    /// transport saw, with the bytes both ways and the compute time, and
    /// the model without remote stages counts none.
    #[test]
    fn a_chain_counts_the_calls_of_each_remote_stage() {
        let llama = write_spec(&TinySpec {
            context: 96,
            ..TinySpec::llama()
        });
        let (n, cuts) = cuts_of(&llama);
        let bounds: Vec<usize> = if n >= 3 {
            vec![cuts[0], cuts[1]]
        } else {
            vec![cuts[0]]
        };
        let (mut chain, handles) = chain_over(&llama, 96, &bounds);
        let _ = run(&mut chain, &long_prompt(40), Some(6));
        let stages = chain.stage_counters();
        assert_eq!(stages.len(), bounds.len());
        for ((name, counters), seen) in stages.iter().zip(&handles.calls) {
            let s = counters.snapshot(name.clone());
            assert_eq!(name, "stage loopback");
            assert_eq!(s.calls as usize, seen.load(Ordering::SeqCst));
            assert!(s.calls > 1, "an open and at least one pass");
            assert_eq!(s.errors, 0);
            assert!(s.sent_bytes > 0 && s.received_bytes > 0);
            assert!(s.compute_ns > 0 && s.compute_ns <= s.wall_ns);
        }
        assert!(engine(&llama, 96, 0.0, None).stage_counters().is_empty());
    }

    /// Carries in `bf16` or `f16` cross the loopback stages in fewer bytes,
    /// every request completes (the open names the type, the worker answers
    /// in it), and for a hybrid model the snapshots and restores that ride
    /// in the frames still work across turns.
    #[test]
    fn a_chain_over_reduced_carries_runs_in_fewer_bytes() {
        use crate::provider::local::CarryDtype;
        let llama = write_spec(&TinySpec {
            context: 96,
            ..TinySpec::llama()
        });
        let qwen = hybrid_model();
        for (what, file, context) in [("llama", &llama, 96), ("qwen35", &qwen, 128)] {
            let (_, cuts) = cuts_of(file);
            let mut per_call = Vec::new();
            for dtype in [CarryDtype::F32, CarryDtype::Bf16, CarryDtype::F16] {
                let (mut chain, _) = chain_over_as(file, context, &[cuts[0]], dtype);
                // Two turns of the same prompt: the second hits the prefix
                // store (a restore from a snapshot on the hybrid model).
                for _ in 0..2 {
                    let got = run(&mut chain, &long_prompt(40), Some(6));
                    assert!(got.items.iter().all(Result::is_ok), "{what} {dtype:?}");
                    assert!(!text(&got).is_empty(), "{what} {dtype:?}");
                }
                let counters = chain.stage_counters();
                let s = counters[0].1.snapshot(counters[0].0.clone());
                assert_eq!(s.errors, 0, "{what} {dtype:?}");
                per_call.push((s.sent_bytes + s.received_bytes) as f64 / s.calls as f64);
            }
            // Decode calls are mostly headers, prefill calls mostly floats;
            // the mean per call falls clearly under the f32 chain's.
            assert!(
                per_call[1] < 0.9 * per_call[0] && per_call[2] < 0.9 * per_call[0],
                "{what}: bytes per call {per_call:?}"
            );
        }
    }

    /// `spec` over a chain drafting with `open`, `k` tokens a round, the
    /// governor and cost model forced to always draft.
    fn chain_drafting(
        file: &crate::provider::local::model::tiny::TinyModel,
        drafter_file: &crate::provider::local::model::tiny::TinyModel,
        context: usize,
        bounds: &[usize],
        k: usize,
        open: impl FnOnce(&gguf::Content, &Model) -> Result<Drafter, CucaError>,
    ) -> Result<(Engine, Links), CucaError> {
        let (mut chain, links) = chain_over(file, context, bounds);
        let content = gguf::Content::open(&drafter_file.path).unwrap();
        chain.set_drafter(open(&content, &chain.model)?);
        chain.draft_tokens = k;
        chain.force_speculation();
        Ok((chain, links))
    }

    /// Speculation on a chain: the EAGLE head of a tiny `mistral4` and the
    /// in-trunk head of a tiny attention-only `qwen35` draft against the
    /// coordinator's last stage, and so does the in-trunk head of the hybrid
    /// `qwen35` (one worker, then two), whose workers roll the recurrent
    /// state of rejected drafts back with the next call. One verification
    /// pass crosses the network for `k + 1` rows, and the reply is plain
    /// greedy decoding's, with and without lost answers.
    #[test]
    fn speculation_on_a_chain_equals_plain_greedy() {
        use crate::provider::local::model::tiny::{TinyQwen35, write_qwen35};
        let mistral = mistral4(32);
        let eagle_file = eagle();
        let attention = write_qwen35(&TinyQwen35 {
            interval: 1,
            ..TinyQwen35::default()
        });
        let prompts: [&[u32]; 5] = [
            &PROMPT,
            &[2, 9, 9, 9, 10],
            &[2, 13, 12, 11, 10, 9, 13],
            &[2, 11],
            &[2, 12, 10, 14, 13, 11, 9, 15, 10, 12, 14, 11],
        ];
        let hybrid = write_qwen35(&TinyQwen35::default());
        for what in [
            "eagle",
            "in-trunk",
            "in-trunk hybrid",
            "in-trunk hybrid, 2 workers",
        ] {
            let (file, drafter_file, workers) = match what {
                "eagle" => (&mistral, &eagle_file, 1),
                "in-trunk" => (&attention, &attention, 1),
                "in-trunk hybrid" => (&hybrid, &hybrid, 1),
                _ => (&hybrid, &hybrid, 2),
            };
            let (_, cuts) = cuts_of(file);
            let bounds = &cuts[..workers];
            let (mut drafted, mut accepted, mut single_accepted) = (0, 0, 0);
            for lose in [false, true] {
                for k in [1usize, 2, 4, 8] {
                    for prompt in prompts {
                        let mut plain = engine(file, 32, 0.0, None);
                        // The same drafter on one node: what a chain must
                        // draft and accept, row for row.
                        let mut single = if what == "eagle" {
                            engine_with_eagle(file, drafter_file, 32, 0.0, None, k, 1)
                        } else {
                            engine_in_trunk(file, 32, 0.0, None, k, 1)
                        };
                        let (mut spec, links) =
                            chain_drafting(file, drafter_file, 32, bounds, k, |c, m| {
                                if what == "eagle" {
                                    Drafter::open_eagle(c, m)
                                } else {
                                    Drafter::open_in_trunk(c, m)
                                }
                            })
                            .unwrap();
                        links.lose_every_reply[0].store(lose, Ordering::SeqCst);
                        let expected = run(&mut plain, prompt, None);
                        let alone = run(&mut single, prompt, None);
                        let got = run(&mut spec, prompt, None);
                        let label = format!("{what}, k {k}, lose {lose}, prompt {prompt:?}");
                        assert_same_reply(&got, &expected, &label);
                        assert_same_reply(&alone, &expected, &label);
                        assert_eq!(got.items.len(), expected.items.len(), "{label}");
                        assert_eq!(spec.last_ids, plain.last_ids, "{label}");
                        // The drafter saw every row the chain's last stage
                        // produced (a head one pair behind the cache).
                        let held = match what {
                            "eagle" => eagle_len(&spec),
                            _ => head_len(&spec),
                        };
                        assert_eq!(held, spec.last_ids.len() - 1, "{label}");
                        let (stats, stats_alone) = (spec.stats.snapshot(), single.stats.snapshot());
                        assert_eq!(
                            (stats.drafted, stats.accepted),
                            (stats_alone.drafted, stats_alone.accepted),
                            "{label}: the chain drafts as one node does"
                        );
                        drafted += stats.drafted;
                        accepted += stats.accepted;
                        single_accepted += stats_alone.accepted;
                        // A round verifies up to `k + 1` rows in one call:
                        // fewer calls than tokens when drafts are accepted.
                        if !lose && stats.accepted > 0 {
                            let calls = links.calls[0].load(Ordering::SeqCst);
                            assert!(
                                calls < spec.last_ids.len(),
                                "{label}: {calls} calls for {} tokens",
                                spec.last_ids.len()
                            );
                        }
                    }
                }
            }
            assert!(drafted > 0, "{what}: the drafter proposed tokens");
            // The tiny in-trunk head may agree with the trunk less than the
            // EAGLE fixture does: the chain must accept exactly what one
            // node does, and EAGLE must accept some.
            assert_eq!(accepted, single_accepted, "{what}");
            assert!(
                accepted > 0 || what == "in-trunk",
                "{what}: some drafts were accepted"
            );
        }
    }

    /// A stage that fails in the middle of a speculative round fails that
    /// request and no other: the engine resets the drafter and the stages
    /// (EAGLE on `mistral4`, the in-trunk head on the hybrid `qwen35`, whose
    /// workers also lose a pending rewind), and the next request is plain
    /// greedy decoding's.
    #[test]
    fn a_failure_in_a_speculative_round_fails_one_request() {
        use crate::provider::local::model::tiny::{TinyQwen35, write_qwen35};
        let (mistral, eagle_file) = (mistral4(32), eagle());
        let hybrid = write_qwen35(&TinyQwen35::default());
        for what in ["eagle", "in-trunk hybrid"] {
            let (model, drafter_file) = match what {
                "eagle" => (&mistral, &eagle_file),
                _ => (&hybrid, &hybrid),
            };
            let (_, cuts) = cuts_of(model);
            let mut plain = engine(model, 32, 0.0, None);
            let expected = run(&mut plain, &PROMPT, None);
            // The hybrid chain accepts drafts and needs fewer calls.
            let fails: &[usize] = if what == "eagle" {
                &[3, 4, 5, 6]
            } else {
                &[2, 3, 4, 5]
            };
            for &fail_at in fails {
                let (mut spec, links) =
                    chain_drafting(model, drafter_file, 32, &[cuts[0]], 4, |c, m| {
                        if what == "eagle" {
                            Drafter::open_eagle(c, m)
                        } else {
                            Drafter::open_in_trunk(c, m)
                        }
                    })
                    .unwrap();
                // The open took call 0; fail a forward call of the first
                // request.
                links.fail_at[0].store(fail_at, Ordering::SeqCst);
                let hit = run(&mut spec, &PROMPT, None);
                let label = format!("{what}, call {fail_at}");
                assert!(hit.items.iter().any(Result::is_err), "{label} fails");
                let again = run(&mut spec, &PROMPT, None);
                assert!(again.items.iter().all(Result::is_ok), "{label}");
                assert_same_reply(&again, &expected, &label);
                assert_eq!(used_blocks(&spec), 0, "{label}");
            }
        }
    }

    /// A chain refuses a head drafter when a stage answers that it can roll
    /// back one row only (a stage on a device with no verification record),
    /// and the block drafters, which read the hidden rows of layers a stage
    /// call does not return.
    #[test]
    fn a_chain_refuses_the_drafters_it_cannot_serve() {
        use crate::provider::local::model::tiny::{TinyQwen35, write_qwen35};
        let hybrid = write_qwen35(&TinyQwen35::default());
        let (_, cuts) = cuts_of(&hybrid);
        let (mut chain, _) = chain_over(&hybrid, 32, &[cuts[0]]);
        let content = gguf::Content::open(&hybrid.path).unwrap();
        // CPU workers answer that they can roll a verification round back.
        Drafter::open_in_trunk(&content, &chain.model).unwrap();
        let Model::Chain(m) = &mut chain.model else {
            panic!("not a chain");
        };
        m.limit_rollback(1);
        assert_eq!(chain.model.rollback_rows(), 1);
        let Err(e) = Drafter::open_in_trunk(&content, &chain.model) else {
            panic!("a chain that cannot roll back took the in-trunk head");
        };
        assert!(
            matches!(&e, CucaError::Config(m) if m.contains("keeps no verification record")),
            "{e:?}"
        );
        let Err(e) = Drafter::open_dflash(&content, &chain.model, 32) else {
            panic!("a chain took a DFlash drafter");
        };
        assert!(
            matches!(&e, CucaError::Config(m) if m.contains("reads the hidden rows of several layers")),
            "{e:?}"
        );
    }

    /// A verification round on a chain over recurrent or compressor state
    /// rolls back to every position of the call: the local stage at once,
    /// the workers with the next call's `rewind`, and the logits that follow
    /// equal the whole model's, bit for bit.
    #[test]
    fn a_chain_rolls_back_a_verification_round_like_the_whole_model() {
        use crate::provider::local::model::cpu::Hidden;
        use crate::provider::local::model::paged::KvPools;
        use crate::provider::local::model::tiny::{TinyDeepseek41, write_deepseek41};
        use crate::provider::local::model::{Batch, SeqRows, StateSlot, Want};
        fn step(
            model: &mut Model,
            pools: &mut KvPools,
            seq: &mut crate::provider::local::model::paged::SeqBlocks,
            ids: &[u32],
            start: usize,
            want: Want,
        ) -> Vec<f32> {
            pools.reserve(seq, start + ids.len()).unwrap();
            let history = ids_before(start);
            let batch = Batch {
                rows: vec![SeqRows {
                    history: &history,
                    tokens: ids,
                    start_pos: start,
                    tables: &seq.tables,
                    state: StateSlot(0),
                    want,
                    hidden: Hidden::None,
                }],
                exact: true,
            };
            let out = model.forward_batch(&batch).unwrap();
            out[0].logits.clone().unwrap_or_default()
        }
        // The token at position `i`.
        fn token(i: usize) -> u32 {
            ((i * 5 + 3) % 14 + 1) as u32
        }
        fn ids_before(n: usize) -> Vec<u32> {
            (0..n).map(token).collect()
        }
        let hybrid = hybrid_model();
        let hyper = write_deepseek41(&TinyDeepseek41::default());
        for (what, file, context) in [("qwen35", &hybrid, 128), ("deepseek41", &hyper, 32)] {
            let (_, cuts) = cuts_of(file);
            // The call at positions 6..11 verifies five rows; keep 6..=10
            // positions of it (11 is the whole call: nothing to undo).
            for keep in 6..=10usize {
                let mut outputs = Vec::new();
                for chained in [false, true] {
                    let mut e = if chained {
                        chain_over(file, context, &cuts[..1]).0
                    } else {
                        engine(file, context, 0.0, None)
                    };
                    let layout = e.model.kv_layout().clone();
                    let mut pools = KvPools::new(&layout);
                    let mut seq = pools.seq();
                    let all: Vec<u32> = (0..14).map(token).collect();
                    step(&mut e.model, &mut pools, &mut seq, &all[..6], 0, Want::Last);
                    step(
                        &mut e.model,
                        &mut pools,
                        &mut seq,
                        &all[6..11],
                        6,
                        Want::All,
                    );
                    assert_eq!(
                        e.model.rewind(StateSlot(0), keep),
                        keep,
                        "{what}, keep {keep}, chained {chained}"
                    );
                    outputs.push(step(
                        &mut e.model,
                        &mut pools,
                        &mut seq,
                        &all[keep..keep + 2],
                        keep,
                        Want::Last,
                    ));
                }
                assert!(!outputs[0].is_empty(), "{what}: logits");
                assert!(
                    outputs[0] == outputs[1],
                    "{what}, keep {keep}: the chain's logits equal the whole model's"
                );
            }
        }
    }

    /// A model whose first layers run on workers answers what the whole model
    /// does: attention-only (one and two remote stages, a prefix hit on the
    /// second turn), Gemma 4 (shared K/V rows cross the cut; the assistant
    /// drafts against the last stage), the hybrid Qwen 3.5 (a second turn
    /// restores the recurrent state from snapshots the workers took) and the
    /// hyper-connection DeepSeek-V4.1.
    #[test]
    fn a_chain_over_loopback_stages_equals_the_whole_model() {
        let llama = write_spec(&TinySpec {
            context: 96,
            ..TinySpec::llama()
        });
        let (n, cuts) = cuts_of(&llama);
        assert!(n >= 2);
        let prompts = [long_prompt(40), long_prompt(33), long_prompt(25)];
        chain_equals_whole("llama", &llama, 96, &[cuts[0]], &prompts);
        if n >= 3 {
            chain_equals_whole("llama", &llama, 96, &[cuts[0], cuts[1]], &prompts);
        }

        // Gemma 4: every cut, the shared layers' K/V rows crossing it.
        let gemma = write_tiny_model();
        let (_, cuts) = cuts_of(&gemma);
        let prompts = [PROMPT.to_vec(), vec![2, 9, 9, 9, 10]];
        for &cut in &cuts {
            chain_equals_whole("gemma4", &gemma, 32, &[cut], &prompts);
        }
        let drafter = write_tiny_drafter();
        for k in [1usize, 4] {
            let (n, cuts) = cuts_of(&gemma);
            // The assistant reads the K/V of the last sliding and the last
            // full layer's sources: the last stage must hold both, so the
            // first cut leaves it the most layers.
            let first = cuts[0];
            assert!(first < n);
            let range = LayerRange {
                first: 0,
                end: first,
            };
            let link: Box<dyn crate::provider::local::model::chain::StageTransport> = Box::new(
                super::testing::Loopback::new(super::testing::stage_worker(&gemma, 32, range, 1)),
            );
            let mut spec = super::testing::chain_engine_with_drafter(
                &gemma,
                &drafter,
                32,
                vec![(range, link)],
                k,
            );
            let mut plain = engine(&gemma, 32, 0.0, None);
            for prompt in &prompts {
                let got = run(&mut spec, prompt, None);
                let want = run(&mut plain, prompt, None);
                assert_same_reply(&got, &want, &format!("gemma4 assistant, k {k}"));
            }
            assert!(
                spec.stats.snapshot().drafted > 0,
                "the assistant drafted against the last stage"
            );
        }

        // Qwen 3.5: the snapshots a restore reads are the workers' as well.
        let qwen = hybrid_model();
        let (_, cuts) = cuts_of(&qwen);
        let mut first = long_prompt(40);
        first[19] = 3;
        let mut second = first[..30].to_vec();
        second.extend([14, 15, 13, 12, 11, 10, 9, 12, 13, 14]);
        let prompts = [first.clone(), second, {
            let mut extended = first;
            extended.extend([15, 14, 13]);
            extended
        }];
        let (mut chain, _) = chain_over(&qwen, 128, &[cuts[0]]);
        let mut whole = engine(&qwen, 128, 0.0, None);
        for (i, prompt) in prompts.iter().enumerate() {
            let got = run(&mut chain, prompt, Some(5));
            let want = run(&mut whole, prompt, Some(5));
            assert_same_reply(&got, &want, &format!("qwen35, turn {i}"));
            assert_eq!(usage(&got), usage(&want), "qwen35, turn {i}");
            if i > 0 {
                assert!(usage(&got).0 > 0, "turn {i} restored a snapshot");
            }
        }
        assert!(chain.snapshots.len() > 0, "the chain kept snapshots");

        // DeepSeek-V4.1: hyper-connection streams cross the cut.
        let deepseek = write_deepseek41(&TinyDeepseek41::default());
        let (_, cuts) = cuts_of(&deepseek);
        let prompts = [long_prompt(70), long_prompt(70)];
        for &cut in &cuts {
            chain_equals_whole("deepseek41", &deepseek, 96, &[cut], &prompts);
        }
    }

    /// A stage that fails a call fails the step that made it, and the engine
    /// fails the running request and empties its pools like any failed pass;
    /// the next request finds the stages opened again under a new session and
    /// answers as a fresh engine does.
    #[test]
    fn a_failed_stage_fails_the_step_and_the_next_request_reopens() {
        use stageframe::{OP_FORWARD, OP_OPEN};
        for model in [
            write_spec(&TinySpec {
                context: 96,
                ..TinySpec::llama()
            }),
            hybrid_model(),
        ] {
            let (_, cuts) = cuts_of(&model);
            let prompt = long_prompt(40);
            let want = run(&mut engine(&model, 96, 0.0, None), &prompt, Some(8));
            let opened = |links: &Links| {
                links.ops[0]
                    .lock()
                    .unwrap()
                    .iter()
                    .filter(|&&op| op == OP_OPEN)
                    .count()
            };
            // The stage fails the first, the second and the third call of
            // the request (each a forward pass).
            for offset in 0..3 {
                let (mut chain, links) = chain_over(&model, 96, &[cuts[0]]);
                assert_eq!(opened(&links), 1, "one session at load");
                let at = links.calls[0].load(Ordering::SeqCst) + offset;
                links.fail_at[0].store(at, Ordering::SeqCst);
                let failed = run(&mut chain, &prompt, Some(8));
                let reached = links.calls[0].load(Ordering::SeqCst) > at;
                assert!(reached || offset == 2, "the request makes {offset} calls");
                if reached {
                    assert!(
                        failed.items.iter().any(Result::is_err),
                        "offset {offset}: the request fails"
                    );
                    // The running request is told why at its next pass, not
                    // by a position error from a worker that was reopened
                    // under it.
                    assert!(
                        failed.items.iter().any(
                            |i| matches!(i, Err(e) if e.to_string().contains("connection dropped"))
                        ),
                        "offset {offset}: the stage's error reaches the request"
                    );
                    assert_eq!(used_blocks(&chain), 0, "the pools were reset");
                    assert_eq!(chain.snapshots.len(), 0);
                }

                let again = run(&mut chain, &prompt, Some(8));
                assert!(again.items.iter().all(Result::is_ok), "offset {offset}");
                assert_same_reply(
                    &again,
                    &want,
                    &format!("offset {offset}, after the failure"),
                );
                if reached {
                    assert_eq!(usage(&again).0, 0, "nothing is cached after a failure");
                    assert_eq!(
                        opened(&links),
                        2,
                        "the stage was opened once more (offset {offset})"
                    );
                    let ops = links.ops[0].lock().unwrap();
                    let last_open = ops.iter().rposition(|&op| op == OP_OPEN).unwrap();
                    assert!(
                        ops[last_open..].contains(&OP_FORWARD),
                        "the stage serves passes after the reopen"
                    );
                }
            }
        }
    }

    /// A worker another session opened under a running request (what a
    /// restarted worker looks like to its coordinator: it forgot the
    /// session and what it held) fails the request with a provider error
    /// that names the stage and says the sequence's state is lost, not with
    /// the worker's configuration error; the next request opens the stages
    /// again and answers as a fresh engine does.
    #[test]
    fn a_worker_that_restarts_under_a_request_fails_it_as_a_provider_error() {
        use stageframe::OP_OPEN;
        for model in [
            write_spec(&TinySpec {
                context: 96,
                ..TinySpec::llama()
            }),
            hybrid_model(),
        ] {
            let (_, cuts) = cuts_of(&model);
            let prompt = long_prompt(40);
            let want = run(&mut engine(&model, 96, 0.0, None), &prompt, Some(8));
            // The first, the second and the third call of the request (each
            // a forward pass).
            for offset in 0..3 {
                let (mut chain, links) = chain_over(&model, 96, &[cuts[0]]);
                let at = links.calls[0].load(Ordering::SeqCst) + offset;
                links.restart_at[0].store(at, Ordering::SeqCst);
                let failed = run(&mut chain, &prompt, Some(8));
                let reached = links.calls[0].load(Ordering::SeqCst) > at;
                assert!(reached || offset == 2, "the request makes {offset} calls");
                if reached {
                    let errors: Vec<&CucaError> = failed
                        .items
                        .iter()
                        .filter_map(|i| i.as_ref().err())
                        .collect();
                    assert!(!errors.is_empty(), "offset {offset}: the request fails");
                    for e in errors {
                        assert!(
                            matches!(e, CucaError::Provider { .. }),
                            "offset {offset}: a provider error, got {e:?}"
                        );
                        let text = e.to_string();
                        assert!(
                            text.contains(
                                "stage loopback restarted and lost this sequence's state"
                            ),
                            "offset {offset}: {text}"
                        );
                        assert!(!text.contains("belongs to another session"), "{text}");
                    }
                    assert_eq!(used_blocks(&chain), 0, "the pools were reset");
                }
                let again = run(&mut chain, &prompt, Some(8));
                assert!(again.items.iter().all(Result::is_ok), "offset {offset}");
                assert_same_reply(
                    &again,
                    &want,
                    &format!("offset {offset}, after the restart"),
                );
                if reached {
                    let ops = links.ops[0].lock().unwrap();
                    let opens = ops.iter().filter(|&&op| op == OP_OPEN).count();
                    assert_eq!(opens, 2, "the stage was opened once more (offset {offset})");
                }
            }
        }
    }

    /// A coordinator whose worker refuses its `open` (here: another sizing)
    /// fails the load with the worker's configuration error, named as the
    /// worker's: only the calls of an open session turn a mismatch into a
    /// restart.
    #[test]
    fn a_worker_that_does_not_match_fails_the_load_as_a_configuration_error() {
        use super::testing::{Loopback, stage_worker, tiny_sizing};
        use crate::provider::local::model::chain::StageTransport;
        let file = write_spec(&TinySpec {
            context: 96,
            ..TinySpec::llama()
        });
        let (_, cuts) = cuts_of(&file);
        let range = LayerRange {
            first: 0,
            end: cuts[0],
        };
        let link: Box<dyn StageTransport> =
            Box::new(Loopback::new(stage_worker(&file, 96, range, 1)));
        let content = gguf::Content::open(&file.path).unwrap();
        let (sizing, _) = tiny_sizing(96, 2, None, None);
        let Err(e) = Model::chain_of(
            &content,
            96,
            LocalDevice::Cpu,
            &crate::provider::local::ExpertPlacement::Resident,
            &sizing,
            &crate::provider::local::EngramCache::default(),
            vec![(range, link)],
            false,
            0,
        ) else {
            panic!("the chain loaded against a worker of another sizing");
        };
        assert!(
            matches!(&e, CucaError::Config(m) if m.contains("max_sequences is 1 here and 2 at the caller")),
            "{e:?}"
        );
    }

    /// An `open` frame for the worker `e` that matches it, to alter.
    fn open_for(e: &Engine, first: usize, end: usize) -> stageframe::Open {
        stageframe::Open {
            fingerprint: e.fingerprint,
            first: first as u32,
            end: end as u32,
            context_length: e.context_length as u64,
            max_sequences: e.sizing.max_sequences as u64,
            kv_tokens: e.sizing.kv_tokens as u64,
            max_rows: e.sizing.max_rows as u64,
            session: 5,
            carry: crate::provider::local::CarryDtype::F32,
        }
    }

    /// A worker opens only for a caller that holds the same file, wants its
    /// layers and sized its pools as it did, names each difference, and
    /// serves nobody but the session it opened.
    #[test]
    fn a_stage_worker_checks_the_caller_and_the_session() {
        use crate::provider::local::model::Want;
        use crate::provider::local::model::cpu::Hidden;
        use stageframe::{ForwardCall, Open, StageSeq, forward_frame, open_frame};
        // A forward frame of `session` and call number `seq` over `slot`,
        // asking for a snapshot or not; it is refused or fails before its
        // pass has anything to run over.
        let forward = |session: u64, seq: u64, slot: u32, snapshot: Option<u64>| {
            forward_frame(&ForwardCall {
                session,
                seq,
                drops: Vec::new(),
                exact: true,
                seqs: vec![StageSeq {
                    slot,
                    start_pos: 0,
                    hist_from: 0,
                    hist: Vec::new(),
                    tokens: vec![1, 2],
                    want: Want::None,
                    hidden: Hidden::None,
                    tables: Vec::new(),
                    restore: None,
                    rewind: None,
                    snapshot,
                    prefetch: Vec::new(),
                }],
                carry: None,
                carry_dtype: crate::provider::local::CarryDtype::F32,
            })
            .unwrap()
        };
        let file = write_spec(&TinySpec {
            context: 96,
            ..TinySpec::llama()
        });
        let mut worker =
            super::testing::stage_worker(&file, 96, LayerRange { first: 0, end: 1 }, 1);
        assert!(worker.is_stage());
        let good = open_for(&worker, 0, 1);
        let refused = |worker: &mut Engine, open: Open| match worker.stage_call(&open_frame(&open))
        {
            Err(CucaError::Config(m)) => m,
            other => panic!("expected a mismatch, got {other:?}"),
        };
        let cases: [(Open, &str); 7] = [
            (
                Open {
                    first: 1,
                    end: 2,
                    ..good
                },
                "this stage holds layers 0..1, the caller wants 1..2",
            ),
            (
                Open { end: 2, ..good },
                "this stage holds layers 0..1, the caller wants 0..2",
            ),
            (
                Open {
                    context_length: 64,
                    ..good
                },
                "context_length is 96 here and 64 at the caller",
            ),
            (
                Open {
                    max_sequences: 3,
                    ..good
                },
                "max_sequences is 1 here and 3 at the caller",
            ),
            (
                Open {
                    kv_tokens: good.kv_tokens + 16,
                    ..good
                },
                "kv_tokens is",
            ),
            (
                Open {
                    max_rows: good.max_rows + 1,
                    ..good
                },
                "max_rows is",
            ),
            (
                Open {
                    fingerprint: 77,
                    ..good
                },
                "the caller's model is not the same file",
            ),
        ];
        for (open, want) in cases {
            let message = refused(&mut worker, open);
            assert!(message.contains(want), "{message}");
        }

        // Nothing opened yet: no session matches.
        let m = match worker.stage_call(&forward(5, 1, 0, None)) {
            Err(CucaError::Config(m)) => m,
            other => panic!("{other:?}"),
        };
        assert!(m.contains("another session"), "{m}");
        worker.stage_call(&open_frame(&good)).unwrap();
        // Another session's calls are refused; the opened one's are served
        // (they get as far as their own errors).
        let m = match worker.stage_call(&forward(6, 1, 0, None)) {
            Err(CucaError::Config(m)) => m,
            other => panic!("{other:?}"),
        };
        assert!(m.contains("another session"), "{m}");
        // A slot the worker does not have, and a snapshot of a model with
        // no recurrent state: provider errors, from the opened session.
        for (slot, snapshot, what) in [
            (9, None, "out of range"),
            (0, Some(1), "no recurrent state"),
        ] {
            match worker.stage_call(&forward(5, 1, slot, snapshot)) {
                Err(CucaError::Provider { message, .. }) => {
                    assert!(message.contains(what), "{message}");
                }
                other => panic!("{other:?}"),
            }
        }
        // A frame that is not one.
        assert!(worker.stage_call(&[]).is_err());
        assert!(worker.stage_call(&[99, 0, 0]).is_err());

        // An engine that is not a stage answers no stage call.
        let mut plain = engine(&file, 96, 0.0, None);
        assert!(!plain.is_stage());
        assert!(matches!(
            plain.stage_call(&open_frame(&good)),
            Err(CucaError::Config(m)) if m.contains("not a layer stage")
        ));
    }

    /// The stage boundaries and drafters a stage configuration may use are
    /// checked against the model: layers that are no cut, a worker that
    /// would hold the head, remote stages that leave no layer, and drafter
    /// files that cannot pair with a chain.
    #[test]
    fn stage_configuration_is_checked_against_the_model() {
        use crate::provider::local::{RemoteStage, StageRange};
        let hp_of = |file: &crate::provider::local::model::tiny::TinyModel| {
            let content = gguf::Content::open(&file.path).unwrap();
            crate::provider::local::model::read_hparams(&content).unwrap()
        };
        let remote = |layers: usize| RemoteStage {
            url: "http://127.0.0.1:1".into(),
            layers,
            ..RemoteStage::default()
        };
        let check = |hp: &_, config: LocalModelConfig, drafter: &str| {
            stage::check_stages(hp, &config, drafter)
        };
        let message = |r: Result<(), CucaError>| match r {
            Err(CucaError::Config(m)) => m,
            other => panic!("expected a configuration error, got {other:?}"),
        };
        let llama = hp_of(&write_spec(&TinySpec::llama()));
        let n = llama.n_layer;
        let stage = |first, layers| LocalModelConfig {
            stage: Some(StageRange { first, layers }),
            ..LocalModelConfig::default()
        };
        let remotes = |layers: &[usize]| LocalModelConfig {
            remote_stages: layers.iter().map(|&l| remote(l)).collect(),
            ..LocalModelConfig::default()
        };
        check(&llama, stage(0, 1), "").unwrap();
        check(&llama, remotes(&[1]), "").unwrap();
        check(&llama, LocalModelConfig::default(), "").unwrap();
        let m = message(check(&llama, stage(0, n), ""));
        assert!(
            m.contains(&format!(
                "stage 0..{n} must end before the last layer of `llama` ({n} layers)"
            )),
            "{m}"
        );
        let m = message(check(&llama, remotes(&[n]), ""));
        assert!(m.contains(&format!("the remote stages hold {n} of the {n} layers of `llama`: the last stage needs at least one")), "{m}");
        // The EAGLE head pairs with a chain (checked when the drafter opens,
        // where the workers' rollback is known); the block drafters read the
        // hidden rows of several layers, which a stage call does not return.
        check(&llama, remotes(&[1]), "mistral4-eagle").unwrap();
        let m = message(check(&llama, remotes(&[1]), "dflash"));
        assert!(
            m.contains("the dflash drafter reads the hidden rows of several layers of the target")
                && m.contains("cannot be combined with `remote_stage`"),
            "{m}"
        );

        // A boundary inside the compressed attention's reach is no cut.
        let ds = hp_of(&write_deepseek41(&TinyDeepseek41::default()));
        let legal = crate::provider::local::model::split::legal_cuts(&ds);
        let bad = (1..ds.n_layer)
            .find(|b| !legal.contains(b))
            .expect("an illegal cut");
        let m = message(check(&ds, remotes(&[bad]), ""));
        assert!(
            m.contains(&format!(
                "a stage boundary at layer {bad} cuts `deepseek41`"
            )),
            "{m}"
        );
        assert!(m.contains("legal boundaries: "), "{m}");
        let m = message(check(&ds, stage(0, bad), ""));
        assert!(
            m.contains(&format!("a stage boundary at layer {bad}")),
            "{m}"
        );
        check(&ds, remotes(&[legal[0]]), "").unwrap();
    }
}
