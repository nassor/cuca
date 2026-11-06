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
//!    in-trunk drafter keeps the pairs the shared tokens still determine
//!    (`Drafter::truncate`) and re-observes the replayed rows; it serves one
//!    sequence at a time.
//! 2. **Prefill** in chunks of [`prefill_chunk`] tokens, interleaved with the
//!    decode rows of the other sequences within the step's row budget; only
//!    the last chunk computes logits. An in-trunk head takes the final-norm
//!    hidden row of every position (`Drafter::observe`).
//! 3. **Decode**: sample, stop on an end-of-generation id, feed the id through
//!    the [`OutputParser`], cut its text at the first stop string of the job
//!    (`StopMatcher`; thinking and tool calls pass through, and text a stop
//!    string might still begin is held back) and queue every resulting block
//!    for the stream (`try_send`: a slow consumer stalls its own sequence
//!    only), then run the model one token further (a round: for a lone
//!    sequence draft, verify one batch, roll back the rejected positions, let
//!    the in-trunk head observe the kept rows). Generation stops at the
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
mod stop;

use std::collections::VecDeque;
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use candle_core::quantized::gguf_file;
use tokio::sync::mpsc;

use self::scheduler::{Sequence, SnapshotStore};
use super::chat::{ChatFormat, ChatTemplate};
use super::model::mtp::Drafter;
use super::model::paged::{KvPools, Sizing};
use super::model::{Model, prefill_chunk};
use super::sampler::{SplitMix64, random_seed};
use super::tokenizer::Tokenizer;
use super::{LocalModelConfig, MAX_DRAFT_TOKENS, SamplingConfig};
use crate::client::ResponseMetadataHandle;
use crate::error::CucaError;
use crate::types::{MessageContentBlock, ProviderEndpoint};

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
}

/// The resolved [`BatchingConfig`](super::BatchingConfig): what the scheduler
/// runs with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Batching {
    /// Sequences running at once (state slots of a hybrid model).
    pub(crate) max_sequences: usize,
    /// Rows of one forward step across all sequences.
    pub(crate) max_step_tokens: usize,
    /// Register finished and full blocks in the prefix store.
    pub(crate) prefix_cache: bool,
}

/// The model plus the state that survives between jobs.
pub(crate) struct Engine {
    model: Model,
    tokenizer: Arc<Tokenizer>,
    format: ChatFormat,
    sampling: SamplingConfig,
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
    /// The ids the in-trunk drafter's cache holds pairs for.
    drafter_ids: Vec<u32>,
    /// Sampling generator when no fixed seed is configured.
    rng: SplitMix64,
    /// Generator of tool-call ids.
    id_rng: SplitMix64,
    /// The multi-token-prediction drafter, when one is configured.
    drafter: Option<Drafter>,
    /// Most tokens drafted per verification round.
    draft_tokens: usize,
    /// K/V cache bytes of the loaded model at its context length.
    kv_cache_bytes: u64,
    /// Speculation counters shared with `LocalModel::draft_stats`.
    stats: Arc<DraftCounters>,
    /// Scheduler counters shared with `LocalModel::serving_stats`.
    serving: Arc<ServingCounters>,
    /// Cost-aware on/off switch of speculation, kept across requests.
    governor: Governor,
    /// Measured costs that set each round's draft length, kept across
    /// requests.
    cost: CostModel,
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

/// Decides, round by round, whether to draft: speculation runs only while its
/// measured throughput (tokens per second of a draft-plus-verify round,
/// acceptance and device costs included) is at least that of plain decoding,
/// so the worst case is plain decoding plus the cost of occasional probes.
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
    snapshot_bytes: AtomicU64,
    prefix_hit_tokens: AtomicU64,
    preemptions: AtomicU64,
    steps: AtomicU64,
    batched_rows: AtomicU64,
}

impl ServingCounters {
    pub(crate) fn snapshot(&self) -> super::ServingStats {
        super::ServingStats {
            running: self.running.load(Ordering::Relaxed),
            waiting: self.waiting.load(Ordering::Relaxed),
            free_blocks: self.free_blocks.load(Ordering::Relaxed),
            cached_blocks: self.cached_blocks.load(Ordering::Relaxed),
            snapshot_bytes: self.snapshot_bytes.load(Ordering::Relaxed),
            prefix_hit_tokens: self.prefix_hit_tokens.load(Ordering::Relaxed),
            preemptions: self.preemptions.load(Ordering::Relaxed),
            steps: self.steps.load(Ordering::Relaxed),
            batched_rows: self.batched_rows.load(Ordering::Relaxed),
        }
    }
}

fn local_err(message: impl Into<String>) -> CucaError {
    CucaError::provider(ProviderEndpoint::Local, message)
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
    /// [`CucaError::Provider`] for a file that is not a GGUF file candle can
    /// parse (for example an unsupported tensor dtype such as an IQ
    /// quantization), an unsupported checkpoint (wrong architecture, a
    /// weight dtype the GPU engine does not read), a missing tensor or
    /// metadata key, a drafter that does not match the model (an in-trunk
    /// head the file does not carry, or one asked for on a GPU), or an
    /// unusable tokenizer.
    pub(crate) fn open(
        path: &Path,
        drafter: Option<&ResolvedDrafter>,
        stats: Arc<DraftCounters>,
        config: &LocalModelConfig,
    ) -> Result<(Self, Arc<Tokenizer>, ChatTemplate, ChatFormat), CucaError> {
        let io_err = |e: &dyn std::fmt::Display| CucaError::Io {
            message: format!("reading GGUF `{}`: {e}", path.display()),
        };
        let mut reader = BufReader::new(File::open(path).map_err(|e| io_err(&e))?);
        let content = gguf_file::Content::read(&mut reader).map_err(|e| match e {
            // A read failure of the file itself; anything else is a header
            // candle cannot interpret (a tensor dtype it does not implement,
            // such as an IQ quantization, is the usual cause).
            candle_core::Error::Io(_) => io_err(&e),
            other => local_err(format!(
                "`{}` is not a supported GGUF file: {other}",
                path.display()
            )),
        })?;
        drop(reader);

        let hp = super::model::read_hparams(&content)?;
        let context_length = super::resolve_context_length(config.context_length, hp.n_ctx_train);
        // An in-trunk drafter observes every position of its one sequence.
        let in_trunk = matches!(drafter, Some(ResolvedDrafter::InTrunk));
        let (batching, sizing) = resolve_batching(config, context_length, in_trunk)?;
        let model = Model::load(
            &content,
            path,
            context_length,
            config.device,
            &config.shards,
            &config.experts,
            &sizing,
        )?;
        let drafter = drafter
            .map(|source| -> Result<Drafter, CucaError> {
                match source {
                    ResolvedDrafter::File(dp) => {
                        let mut reader =
                            BufReader::new(File::open(dp).map_err(|e| CucaError::Io {
                                message: format!("reading GGUF `{}`: {e}", dp.display()),
                            })?);
                        let dc = gguf_file::Content::read(&mut reader).map_err(|e| {
                            local_err(format!(
                                "`{}` is not a supported GGUF file: {e}",
                                dp.display()
                            ))
                        })?;
                        Drafter::open(&dc, dp, &model, context_length)
                    }
                    ResolvedDrafter::InTrunk => Drafter::open_in_trunk(&content, path, &model),
                }
            })
            .transpose()?;
        let tokenizer = Arc::new(Tokenizer::from_gguf(&content.metadata)?);

        const TEMPLATE_KEY: &str = "tokenizer.chat_template";
        let source = content
            .metadata
            .get(TEMPLATE_KEY)
            .ok_or_else(|| local_err(format!("GGUF metadata `{TEMPLATE_KEY}`: missing")))?
            .to_string()
            .map_err(|e| local_err(format!("GGUF metadata `{TEMPLATE_KEY}`: {e}")))?
            .clone();
        let format = ChatFormat::detect(&content.metadata, &tokenizer, &source)?;
        let template = ChatTemplate::new(
            source,
            tokenizer.bos_token().to_string(),
            tokenizer.eos_token().to_string(),
        )?;

        let engine = Self::new(
            model,
            Arc::clone(&tokenizer),
            format.clone(),
            config.sampling,
            context_length,
            batching,
            drafter,
            config.draft_tokens,
            stats,
        );
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
        sampling: SamplingConfig,
        context_length: usize,
        batching: Batching,
        drafter: Option<Drafter>,
        draft_tokens: usize,
        stats: Arc<DraftCounters>,
    ) -> Self {
        let pools = KvPools::new(model.kv_layout());
        let kv_cache_bytes = model.kv_pool_bytes();
        Self {
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
            drafter_ids: Vec::new(),
            rng: SplitMix64(random_seed()),
            id_rng: SplitMix64(random_seed()),
            drafter,
            draft_tokens,
            kv_cache_bytes,
            stats,
            serving: Arc::default(),
            governor: Governor::default(),
            cost: CostModel::default(),
            #[cfg(test)]
            last_ids: Vec::new(),
        }
    }

    /// The scheduler counters `LocalModel::serving_stats` reads.
    pub(crate) fn serving_counters(&self) -> Arc<ServingCounters> {
        Arc::clone(&self.serving)
    }

    /// The context length the engine runs with: the configured value, or the
    /// default resolved against the model.
    pub(crate) fn context_length(&self) -> usize {
        self.context_length
    }

    /// K/V cache bytes of the loaded model at its context length.
    pub(crate) fn kv_cache_bytes(&self) -> u64 {
        self.kv_cache_bytes
    }

    /// Counters and heat of the expert tiers; `None` unless the model loaded
    /// with `ExpertPlacement::Tiered`.
    pub(crate) fn expert_tiers(&self) -> Option<Arc<super::model::tier::TierShared>> {
        self.model.expert_tiers()
    }
}

/// Resolve `config.batching` against the model's context length: the
/// scheduler's limits and the sizing of the K/V pools and state slots.
///
/// # Errors
///
/// [`CucaError::Config`] for a `kv_tokens` below one context plus a block.
fn resolve_batching(
    config: &LocalModelConfig,
    context_length: usize,
    in_trunk: bool,
) -> Result<(Batching, Sizing), CucaError> {
    use super::model::paged::BLOCK_SIZE;
    let b = &config.batching;
    let kv_tokens = match b.kv_tokens {
        None => 2 * context_length,
        Some(n) if n >= context_length + BLOCK_SIZE => n.div_ceil(BLOCK_SIZE) * BLOCK_SIZE,
        Some(n) => {
            return Err(CucaError::Config(format!(
                "batching.kv_tokens must be at least context_length + {BLOCK_SIZE} ({}), got {n}",
                context_length + BLOCK_SIZE
            )));
        }
    };
    // The in-trunk head observes every position in order: one sequence.
    let max_sequences = if in_trunk { 1 } else { b.max_sequences };
    let max_step_tokens = b
        .max_step_tokens
        .unwrap_or_else(|| prefill_chunk(context_length).max(max_sequences));
    let sizing = Sizing {
        max_sequences,
        kv_tokens,
        max_rows: max_step_tokens
            .min(prefill_chunk(context_length))
            .max(super::model::cpu::MAX_VERIFY_ROWS),
    };
    let batching = Batching {
        max_sequences,
        max_step_tokens,
        prefix_cache: b.prefix_cache,
    };
    Ok((batching, sizing))
}

fn saturating_u32(n: usize) -> u32 {
    u32::try_from(n).unwrap_or(u32::MAX)
}

/// Fixtures shared by the engine tests and the `LocalModel` tests.
#[cfg(test)]
pub(crate) mod testing {
    use super::*;
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
                gguf_file::Value::U32(1),
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
        let tokenizer = Arc::new(tiny_tokenizer());
        let max_step_tokens =
            max_step.unwrap_or_else(|| prefill_chunk(context_length).max(max_sequences));
        let sizing = Sizing {
            max_sequences,
            kv_tokens: kv_tokens.unwrap_or(2 * context_length),
            max_rows: max_step_tokens
                .min(prefill_chunk(context_length))
                .max(crate::provider::local::model::cpu::MAX_VERIFY_ROWS),
        };
        let model = Model::Cpu(load_sized(model, context_length, &sizing));
        Engine::new(
            model,
            Arc::clone(&tokenizer),
            tiny_format(&tokenizer),
            SamplingConfig {
                temperature,
                top_k: 64,
                top_p: 0.95,
                seed,
            },
            context_length,
            Batching {
                max_sequences,
                max_step_tokens,
                prefix_cache: true,
            },
            None,
            0,
            Arc::default(),
        )
    }

    /// [`engine`] with a drafter proposing `draft_tokens` per round.
    pub(crate) fn engine_with_drafter(
        model: &TinyModel,
        drafter: &TinyModel,
        context_length: usize,
        temperature: f32,
        seed: Option<u64>,
        draft_tokens: usize,
    ) -> Engine {
        let mut e = engine(model, context_length, temperature, seed);
        let content =
            gguf_file::Content::read(&mut BufReader::new(File::open(&drafter.path).unwrap()))
                .unwrap();
        e.drafter = Some(Drafter::open(&content, &drafter.path, &e.model, context_length).unwrap());
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
    ) -> Engine {
        let mut e = engine(model, context_length, temperature, seed);
        let content =
            gguf_file::Content::read(&mut BufReader::new(File::open(&model.path).unwrap()))
                .unwrap();
        e.drafter = Some(Drafter::open_in_trunk(&content, &model.path, &e.model).unwrap());
        e.draft_tokens = draft_tokens;
        e.governor = Governor::always_speculating();
        e.cost = CostModel::always_drafting();
        e
    }
}

#[cfg(test)]
mod tests {
    use super::testing::{engine, engine_in_trunk, engine_with_drafter};
    use super::*;
    use crate::provider::local::model::tiny::{TinyQwen35, TinySpec, write_qwen35, write_spec};
    use crate::provider::local::model::tiny::{write_tiny_drafter, write_tiny_model};
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
        let mut job = Job {
            prompt_ids: prompt.to_vec(),
            start_in_thinking: false,
            enable_thinking: false,
            temperature: None,
            max_tokens,
            top_p: None,
            top_k: None,
            seed: None,
            stop: Vec::new(),
            events: tx,
            metadata: metadata.clone(),
        };
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
        e.submit(Job {
            prompt_ids: long.clone(),
            start_in_thinking: false,
            enable_thinking: false,
            temperature: None,
            max_tokens: Some(10),
            top_p: None,
            top_k: None,
            seed: None,
            stop: Vec::new(),
            events: tx,
            metadata: ResponseMetadataHandle::empty(),
        });
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

    /// A GGUF header declaring one tensor of dtype 20 (IQ4_NL), which candle
    /// cannot read: the load must name an unsupported file, not an I/O error.
    #[test]
    fn open_reports_an_unreadable_tensor_dtype_as_unsupported() {
        let mut bytes = b"GGUF".to_vec();
        bytes.extend(3u32.to_le_bytes()); // version
        bytes.extend(1u64.to_le_bytes()); // tensors
        bytes.extend(0u64.to_le_bytes()); // metadata entries
        bytes.extend(1u64.to_le_bytes()); // name length
        bytes.push(b'x');
        bytes.extend(1u32.to_le_bytes()); // dimensions
        bytes.extend(32u64.to_le_bytes());
        bytes.extend(20u32.to_le_bytes()); // IQ4_NL
        bytes.extend(0u64.to_le_bytes()); // offset
        bytes.resize(bytes.len().next_multiple_of(32) + 64, 0);
        let path = std::env::temp_dir().join(format!("cuca-iq4-{}.gguf", std::process::id()));
        std::fs::write(&path, &bytes).unwrap();

        let config = LocalModelConfig::default();
        let result = Engine::open(&path, None, Arc::default(), &config);
        let _ = std::fs::remove_file(&path);
        let err = match result {
            Err(e) => e,
            Ok(_) => panic!("an IQ4_NL tensor must not load"),
        };
        assert!(
            matches!(&err, CucaError::Provider { message, .. } if message.contains("not a supported GGUF file")),
            "{err:?}"
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
                let mut spec = engine_with_drafter(&model, &drafter, 32, 0.0, None, k);
                let expected = run(&mut plain, prompt, None);
                let got = run(&mut spec, prompt, None);
                assert_eq!(text(&got), text(&expected), "k {k}, prompt {prompt:?}");
                assert_eq!(
                    got.items.len(),
                    expected.items.len(),
                    "k {k}, prompt {prompt:?}"
                );
                assert_eq!(
                    got.metadata.token_usage.map(|u| u.completion_tokens),
                    expected.metadata.token_usage.map(|u| u.completion_tokens),
                    "k {k}, prompt {prompt:?}"
                );
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
            let mut spec = engine_with_drafter(&model, &drafter, 32, 0.0, None, 4);
            let a = run(&mut plain, &PROMPT, Some(max));
            let b = run(&mut spec, &PROMPT, Some(max));
            assert_eq!(text(&a), text(&b), "max {max}");
            assert_eq!(
                a.metadata.token_usage.map(|u| u.completion_tokens),
                b.metadata.token_usage.map(|u| u.completion_tokens),
                "max {max}"
            );
        }
        // A context that fills up mid-round.
        let mut plain = engine(&model, 12, 0.0, None);
        let mut spec = engine_with_drafter(&model, &drafter, 12, 0.0, None, 8);
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
            let mut e = engine_with_drafter(&model, &drafter, 32, 1.0, Some(5), 3);
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
        let mut reader = BufReader::new(File::open(&drafter.path).unwrap());
        let content = gguf_file::Content::read(&mut reader).unwrap();
        let e = Transformer::load(&content, &drafter.path, 32)
            .err()
            .unwrap();
        assert!(e.to_string().contains("drafter"), "{e}");
        let mut reader = BufReader::new(File::open(&model.path).unwrap());
        let content = gguf_file::Content::read(&mut reader).unwrap();
        let e = Transformer::load_drafter(&content, &model.path, 32)
            .err()
            .unwrap();
        assert!(e.to_string().contains("not a drafter"), "{e}");
    }

    #[test]
    fn batching_resolves_against_the_context_length() {
        let config = |text: &str| LocalModelConfig {
            batching: text.parse().unwrap(),
            ..LocalModelConfig::default()
        };
        let (batching, sizing) = resolve_batching(&config(""), 1000, false).unwrap();
        assert_eq!(
            batching,
            Batching {
                max_sequences: 4,
                max_step_tokens: 256,
                prefix_cache: true
            }
        );
        assert_eq!(sizing.kv_tokens, 2000, "twice the context by default");
        assert_eq!(sizing.max_sequences, 4);
        // An explicit pool rounds up to blocks and holds at least a context
        // and a block.
        let (_, sizing) = resolve_batching(&config("kv=1016"), 1000, false).unwrap();
        assert_eq!(sizing.kv_tokens, 1024);
        let err = resolve_batching(&config("kv=1015"), 1000, false).unwrap_err();
        assert!(
            matches!(&err, CucaError::Config(m) if m.contains("kv_tokens")),
            "{err}"
        );
        // A very long context shrinks the prefill chunk below the sequences.
        let (batching, _) = resolve_batching(&config("seqs=64"), 131_072, false).unwrap();
        assert_eq!(batching.max_step_tokens, 64);
        // The in-trunk head serves one sequence.
        let (batching, sizing) = resolve_batching(&config("seqs=8"), 1000, true).unwrap();
        assert_eq!((batching.max_sequences, sizing.max_sequences), (1, 1));
        let (batching, _) = resolve_batching(&config("prefix=off,step=32"), 1000, false).unwrap();
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
        write_qwen35(&TinyQwen35 {
            meta: vec![("context_length".into(), gguf_file::Value::U32(128))],
            ..TinyQwen35::default()
        })
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

    /// A hybrid model's verification window bounds the drafts per round.
    #[test]
    fn hybrid_models_bound_the_verification_rows() {
        let model = write_qwen35(&TinyQwen35::default());
        let e = engine(&model, 32, 0.0, None);
        assert_eq!(e.model.rollback_rows(), 17);
        let plain = write_tiny_model();
        assert_eq!(
            engine(&plain, 32, 0.0, None).model.rollback_rows(),
            usize::MAX
        );
    }

    /// The committed pairs of the in-trunk head of `e`.
    fn head_len(e: &Engine) -> usize {
        match &e.drafter {
            Some(Drafter::InTrunk(head)) => head.len(),
            _ => panic!("the engine has no in-trunk head"),
        }
    }

    /// Drafting with the head inside the file must not change greedy output:
    /// the same tokens, the same usage and the same cache bookkeeping as
    /// plain decoding, whatever the draft length and however many drafts the
    /// trunk rejects; and the head stays one pair behind the trunk's cache.
    #[test]
    fn in_trunk_head_leaves_greedy_output_unchanged() {
        let model = write_qwen35(&TinyQwen35::default());
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
                let mut spec = engine_in_trunk(&model, 32, 0.0, None, k);
                let expected = run(&mut plain, prompt, None);
                let got = run(&mut spec, prompt, None);
                assert_eq!(text(&got), text(&expected), "k {k}, prompt {prompt:?}");
                assert_eq!(
                    got.items.len(),
                    expected.items.len(),
                    "k {k}, prompt {prompt:?}"
                );
                assert_eq!(
                    got.metadata.token_usage.map(|u| u.completion_tokens),
                    expected.metadata.token_usage.map(|u| u.completion_tokens),
                    "k {k}, prompt {prompt:?}"
                );
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
        assert!(accepted > 0, "some drafts were accepted and rolled forward");
    }

    #[test]
    fn in_trunk_head_respects_max_tokens_and_the_context_limit() {
        let model = write_qwen35(&TinyQwen35::default());
        for max in [1u32, 2, 3, 7] {
            let mut plain = engine(&model, 32, 0.0, None);
            let mut spec = engine_in_trunk(&model, 32, 0.0, None, 4);
            let a = run(&mut plain, &PROMPT, Some(max));
            let b = run(&mut spec, &PROMPT, Some(max));
            assert_eq!(text(&a), text(&b), "max {max}");
            assert_eq!(
                a.metadata.token_usage.map(|u| u.completion_tokens),
                b.metadata.token_usage.map(|u| u.completion_tokens),
                "max {max}"
            );
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
                let mut spec = engine_in_trunk(&model, ctx, 0.0, None, 8);
                let a = run(&mut plain, prompt, None);
                let b = run(&mut spec, prompt, None);
                assert_eq!(text(&a), text(&b), "ctx {ctx}, prompt {prompt:?}");
                // A budget that ends inside a round leaves the last accepted
                // draft cached, which plain decoding never forwards.
                assert!(
                    spec.last_ids.starts_with(&plain.last_ids),
                    "ctx {ctx}, prompt {prompt:?}"
                );
                assert_eq!(head_len(&spec), spec.last_ids.len() - 1, "ctx {ctx}");
                let tokens = b.metadata.token_usage.map(|u| u.completion_tokens);
                filled += usize::from(tokens == u32::try_from(ctx - prompt.len()).ok());
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
        let model = hybrid_model();
        let mut first = long_prompt(40);
        first[19] = 3;
        let mut second = first[..30].to_vec();
        second.extend([14, 15, 13, 12, 11, 10, 9, 12, 13, 14]);
        let mut warm = engine_in_trunk(&model, 128, 0.0, None, 3);
        let a = run(&mut warm, &first, Some(6));
        assert_eq!(usage(&a), (0, 40));
        assert_eq!(head_len(&warm), warm.last_ids.len() - 1);

        // Resumes from the snapshot at 16: nothing is missing in the head.
        let b = run(&mut warm, &second, Some(6));
        assert_eq!(usage(&b), (16, 24));
        let cold = run(
            &mut engine_in_trunk(&model, 128, 0.0, None, 3),
            &second,
            Some(6),
        );
        let plain = run(&mut engine(&model, 128, 0.0, None), &second, Some(6));
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
        let expected = run(&mut engine(&model, 128, 0.0, None), &extended, Some(4));
        assert_eq!(text(&d), text(&expected));
        assert_eq!(head_len(&warm), warm.last_ids.len() - 1);

        // A prompt that diverges inside the second block keeps the first.
        let mut diverging = first[..18].to_vec();
        diverging.extend([15, 14, 13, 12, 11, 10, 9, 15, 14, 13, 12, 11]);
        let e = run(&mut warm, &diverging, Some(5));
        assert_eq!(usage(&e), (16, 14));
        let expected = run(&mut engine(&model, 128, 0.0, None), &diverging, Some(5));
        assert_eq!(text(&e), text(&expected));
        assert_eq!(head_len(&warm), warm.last_ids.len() - 1);
    }

    /// A failed request resets the head with the trunk, and the engine keeps
    /// serving.
    #[test]
    fn in_trunk_head_recovers_from_a_failed_request() {
        let model = write_qwen35(&TinyQwen35::default());
        let mut e = engine_in_trunk(&model, 32, 0.0, None, 3);
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
        let run_once = || {
            let mut e = engine_in_trunk(&model, 32, 1.0, Some(5), 3);
            let out = run(&mut e, &PROMPT, None);
            (
                text(&out),
                out.metadata.token_usage.map(|u| u.completion_tokens),
                e.stats.snapshot(),
            )
        };
        let (a, n, stats_a) = run_once();
        let (b, m, stats_b) = run_once();
        assert_eq!((a, n, stats_a), (b, m, stats_b));
        assert!(n.is_some());
        assert!(stats_a.drafted > 0, "a seeded run always drafts");
    }

    /// The default context length is the model's maximum when that is below
    /// 8192 (the tiny model trains for 32 positions) and loads; an explicit
    /// value above the maximum stays a loud error.
    #[test]
    fn default_context_length_resolves_to_the_model_maximum() {
        let model = write_tiny_model();
        let mut reader = BufReader::new(File::open(&model.path).unwrap());
        let content = gguf_file::Content::read(&mut reader).unwrap();
        let hp = super::super::model::read_hparams(&content).unwrap();
        let resolved = super::super::resolve_context_length(None, hp.n_ctx_train);
        assert_eq!(resolved, 32);
        let device = super::super::LocalDevice::Cpu;
        let resident = super::super::ExpertPlacement::Resident;
        assert!(
            Model::load(
                &content,
                &model.path,
                resolved,
                device,
                &[],
                &resident,
                &Sizing::single(resolved)
            )
            .is_ok()
        );
        let e = Model::load(
            &content,
            &model.path,
            33,
            device,
            &[],
            &resident,
            &Sizing::single(33),
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
}
