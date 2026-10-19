//! The engine thread's state: model, K/V cache bookkeeping and the generation
//! loop behind one request.
//!
//! [`Engine::run`] executes one [`Job`] at a time on the dedicated engine
//! thread (jobs queue FIFO in the channel owned by `LocalModel`). A job is a
//! pre-tokenized prompt plus the channel its output blocks go to:
//!
//! 1. **Prefix reuse.** The engine remembers the ids whose K/V fill the model
//!    cache. A new prompt keeps the longest common prefix (always leaving the
//!    last prompt token to prefill, since its logits start generation) and the
//!    cache is truncated to it, so a conversation that appends a turn pays only
//!    for the new tokens. The reuse is reported as the response's
//!    [`PromptCacheUsage`] (`read_tokens` reused, `write_tokens` prefilled).
//!    An in-trunk drafter keeps the pairs the shared tokens still determine
//!    (`Drafter::truncate`) and re-observes the replayed rows.
//! 2. **Prefill** in chunks of [`prefill_chunk`] tokens; only the last chunk
//!    computes logits. The loop notices a dropped stream between chunks. An
//!    in-trunk head takes the final-norm hidden row of every position
//!    (`Drafter::observe`).
//! 3. **Decode**: sample, stop on an end-of-generation id, feed the id through
//!    the [`OutputParser`] and forward every resulting block to the stream with
//!    a blocking send (a slow consumer backpressures generation), then run the
//!    model one token further (a round: draft, verify one batch, roll back the
//!    rejected positions, let the in-trunk head observe the kept rows).
//!    Generation stops at the end-of-generation id, at `max_tokens`, or when
//!    the context is full.
//! 4. **Usage**: exact prompt and completion token counts are recorded in the
//!    response metadata before the stream ends.
//!
//! A forward-pass error resets the cache and is delivered as the stream's last
//! item; a panic inside a job is caught, reported the same way, and the engine
//! keeps serving later jobs. A job whose consumer has gone away leaves the
//! cache consistent with the ids it processed, so its work is reusable.

use std::fs::File;
use std::io::BufReader;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use candle_core::quantized::gguf_file;
use tokio::sync::mpsc;

use super::chat::{ChatFormat, ChatTemplate};
use super::model::cpu::Hidden;
use super::model::mtp::Drafter;
use super::model::{Model, Want, prefill_chunk};
use super::output::OutputParser;
use super::sampler::{Dist, Scratch, SplitMix64, greedy, random_seed, sample};
use super::tokenizer::Tokenizer;
use super::{LocalModelConfig, SamplingConfig};
use crate::client::ResponseMetadataHandle;
use crate::error::CucaError;
use crate::request::{PromptCacheUsage, TokenUsage};
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
    /// Where output blocks (or the failure) go; dropping it ends the stream.
    pub(crate) events: mpsc::Sender<Result<MessageContentBlock, CucaError>>,
    /// Receives the prompt-cache and token usage of this job.
    pub(crate) metadata: ResponseMetadataHandle,
}

/// The model plus the state that survives between jobs.
pub(crate) struct Engine {
    model: Model,
    tokenizer: Arc<Tokenizer>,
    format: ChatFormat,
    sampling: SamplingConfig,
    /// Ids whose K/V fill the model cache, in order; its length always equals
    /// the model's cache length.
    cached: Vec<u32>,
    context_length: usize,
    /// Sampling generator when no fixed seed is configured.
    rng: SplitMix64,
    /// Generator of tool-call ids.
    id_rng: SplitMix64,
    /// The multi-token-prediction drafter, when one is configured.
    drafter: Option<Drafter>,
    /// Tokens drafted per verification round.
    draft_tokens: usize,
    /// K/V cache bytes per context position of the loaded model.
    kv_bytes_per_position: usize,
    /// Speculation counters shared with `LocalModel::draft_stats`.
    stats: Arc<DraftCounters>,
    /// Cost-aware on/off switch of speculation, kept across requests.
    governor: Governor,
}

/// Exponential moving average of a throughput (tokens per second).
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

    fn record(&mut self, tokens: usize, secs: f64) {
        if !self.skipped {
            self.skipped = true;
            return;
        }
        if secs <= 0.0 {
            return;
        }
        let tps = tokens as f64 / secs;
        self.value = if self.n == 0 {
            tps
        } else {
            self.value + Self::ALPHA * (tps - self.value)
        };
        self.n += 1;
    }
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
        let was_probe = spec && self.spec.value < self.plain.value * Self::MARGIN;
        if spec {
            // A round more than `OUTLIER` times slower than plain decoding is
            // a first-shape compile (GPU autotune, shader build), not a cost
            // the next rounds will pay: keep it out of the estimate.
            if self.plain.n >= Self::MIN_SAMPLES
                && secs > 0.0
                && tokens as f64 / secs * Self::OUTLIER < self.plain.value
                && self.spec.n < Self::MIN_SAMPLES
            {
                return;
            }
            self.spec.record(tokens, secs);
        } else {
            self.plain.record(tokens, secs);
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
    /// quantization), an unsupported checkpoint (wrong architecture, MoE, a
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
        let kv_bytes_per_position = hp.kv_bytes_per_position();
        let context_length = super::resolve_context_length(config.context_length, hp.n_ctx_train);
        let model = Model::load(&content, path, context_length, config.device)?;
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

        let engine = Self {
            model,
            tokenizer: Arc::clone(&tokenizer),
            format: format.clone(),
            sampling: config.sampling,
            cached: Vec::new(),
            context_length,
            rng: SplitMix64(random_seed()),
            id_rng: SplitMix64(random_seed()),
            drafter,
            draft_tokens: config.draft_tokens,
            kv_bytes_per_position,
            stats,
            governor: Governor::default(),
        };
        Ok((engine, tokenizer, template, format))
    }

    /// The context length the engine runs with: the configured value, or the
    /// default resolved against the model.
    pub(crate) fn context_length(&self) -> usize {
        self.context_length
    }

    /// K/V cache bytes per context position of the loaded model.
    pub(crate) fn kv_bytes_per_position(&self) -> usize {
        self.kv_bytes_per_position
    }

    /// Run one job to completion on the calling (engine) thread.
    ///
    /// A panic inside the job is caught: the consumer gets a provider error as
    /// the last stream item and the cache is reset, so later jobs still work.
    pub(crate) fn run(&mut self, job: Job) {
        let events = job.events.clone();
        if catch_unwind(AssertUnwindSafe(|| self.generate(job))).is_err() {
            self.reset();
            let _ = events.blocking_send(Err(local_err("local engine panicked; KV cache reset")));
        }
    }

    /// Forget every cached position.
    fn reset(&mut self) {
        self.model.truncate_cache(0);
        self.cached.clear();
        if let Some(drafter) = &mut self.drafter {
            drafter.reset();
        }
    }

    /// Where a hybrid model checkpoints its recurrent state while prefilling
    /// `prompt_ids`, so the next request can restart from its shared prefix.
    ///
    /// The next request of a conversation repeats this prompt only up to the
    /// end of its last complete message: what follows is the generation
    /// prompt (an assistant header, maybe an empty reasoning block), which
    /// templates rewrite once the answer is part of the history. A checkpoint
    /// at the very end of the prompt would sit past the shared prefix and
    /// restore nothing. The end of the last message is the position after the
    /// last end-of-turn token; a prompt without one (a raw completion) is
    /// expected to be extended, so its end serves.
    fn checkpoint_position(&self, prompt_ids: &[u32]) -> usize {
        prompt_ids
            .iter()
            .rposition(|&id| self.format.is_eog(id))
            .map_or(prompt_ids.len(), |i| i + 1)
    }

    fn generate(&mut self, job: Job) {
        let Job {
            prompt_ids,
            start_in_thinking,
            enable_thinking,
            temperature,
            max_tokens,
            events,
            metadata,
        } = job;
        let prompt_len = prompt_ids.len();
        if prompt_len == 0 || prompt_len >= self.context_length {
            let _ = events.blocking_send(Err(local_err(format!(
                "prompt has {prompt_len} tokens; context_length is {}",
                self.context_length
            ))));
            return;
        }

        // 1. Prefix reuse: the last prompt token is always prefilled, because
        // its logits start generation. A hybrid model (recurrent state) can
        // return only to positions it holds a state for, so it may keep less
        // than the shared prefix; what it keeps is what was really reused.
        let shared = self
            .cached
            .iter()
            .zip(&prompt_ids)
            .take_while(|(a, b)| a == b)
            .count();
        let reuse = if shared == prompt_len {
            shared - 1
        } else {
            shared
        };
        let reuse = self.model.truncate_cache(reuse);
        self.cached.truncate(reuse);
        // An in-trunk head keeps what still holds for the shared tokens; the
        // rows replayed below replace the rest.
        if let Some(drafter) = &mut self.drafter {
            drafter.truncate(shared);
        }
        let checkpoint_at = self
            .model
            .is_hybrid()
            .then(|| self.checkpoint_position(&prompt_ids))
            .filter(|&at| at > reuse);

        // 2. Prefill in bounded chunks; only the last one computes logits (and,
        // with a drafter, the hidden state the first draft starts from). A
        // drafter that observes the trunk (the in-trunk head) takes the
        // hidden rows of every chunk.
        let observes = self.drafter.as_ref().is_some_and(Drafter::observes);
        let tap = if self.drafter.is_some() {
            Hidden::PostNorm
        } else {
            Hidden::None
        };
        let chunk = prefill_chunk(self.context_length);
        let mut logits = None;
        let mut hidden: Option<Vec<f32>> = None;
        let mut pos = reuse;
        while pos < prompt_len {
            let mut end = (pos + chunk).min(prompt_len);
            // A chunk ends where the checkpoint goes.
            if let Some(at) = checkpoint_at.filter(|&at| pos < at && at < end) {
                end = at;
            }
            let last = end == prompt_len;
            let want = if last { Want::Last } else { Want::None };
            let tap = if observes {
                Hidden::PostNormAll
            } else if last {
                tap
            } else {
                Hidden::None
            };
            let forward = match self
                .model
                .forward_ext(&prompt_ids[pos..end], pos, want, tap, true)
            {
                Ok(f) => f,
                Err(e) => {
                    let _ = events.blocking_send(Err(e));
                    self.reset();
                    return;
                }
            };
            if observes
                && let (Some(drafter), Some(rows)) = (&mut self.drafter, forward.hidden.as_deref())
                && let Err(e) = drafter.observe(&self.model, pos, &prompt_ids[pos..end], rows)
            {
                let _ = events.blocking_send(Err(e));
                self.reset();
                return;
            }
            logits = forward.logits;
            // Only the last row seeds the first draft.
            hidden = forward
                .hidden
                .map(|h| if observes { last_row(h, end - pos) } else { h });
            self.cached.extend_from_slice(&prompt_ids[pos..end]);
            pos = end;
            if checkpoint_at == Some(pos) {
                self.model.checkpoint();
            }
            if !last && events.is_closed() {
                return;
            }
        }
        let Some(mut logits) = logits else {
            let _ = events.blocking_send(Err(local_err("prefill produced no logits")));
            self.reset();
            return;
        };

        // 3. Prompt-cache usage: what was reused and what was prefilled.
        metadata.set_prompt_cache_usage(PromptCacheUsage {
            read_tokens: saturating_u32(reuse),
            write_tokens: saturating_u32(prompt_len - reuse),
        });

        // 4. Generation setup.
        let temperature = temperature.unwrap_or(self.sampling.temperature);
        let (top_k, top_p) = (self.sampling.top_k, self.sampling.top_p);
        let seed_fixed = self.sampling.seed.is_some();
        let mut seeded = self.sampling.seed.map(SplitMix64);
        let rng = seeded.as_mut().unwrap_or(&mut self.rng);
        let budget = max_tokens
            .map_or(usize::MAX, |n| n as usize)
            .min(self.context_length - prompt_len);
        let tokenizer = Arc::clone(&self.tokenizer);
        let mut parser = OutputParser::new(&self.format, start_in_thinking);
        let id_rng = &mut self.id_rng;
        let mut next_call_id = || format!("call_{:016x}", id_rng.next_u64());
        let mut items = Vec::new();
        let mut completion = 0usize;
        let mut scratch = Scratch::default();
        // The pending token followed by the drafts, refilled every round.
        let mut batch: Vec<u32> = Vec::new();
        let greedy_mode = temperature <= 0.0;
        // The token a rejected speculative draft was resampled to: the next
        // token, taken instead of sampling the verified logits.
        let mut forced: Option<u32> = None;

        // 5. Decode loop. Each round takes the pending token from the logits
        // after the last cached position, optionally drafts a few tokens
        // after it, runs the target once over the pending token and the
        // drafts, and keeps the longest prefix the target agrees with.
        'decode: while completion < budget {
            let id = match forced.take() {
                Some(id) => id,
                None => sample(&mut logits, temperature, top_k, top_p, rng, &mut scratch),
            };
            if self.format.is_eog(id) {
                break;
            }
            completion += 1;
            parser.push(id, &tokenizer, &mut next_call_id, &mut items);
            for item in items.drain(..) {
                if events.blocking_send(item).is_err() {
                    return;
                }
            }
            if completion == budget || self.cached.len() == self.context_length {
                break;
            }

            // Draft: `id` sits at position `cached.len()` and is not cached.
            let room = self.context_length - self.cached.len() - 1;
            // Draft only while the governor measures speculation as faster
            // than plain decoding (a fixed k keeps the batch shape, and so the
            // GPU kernels, the same from round to round).
            let round_start = std::time::Instant::now();
            // A seeded sampling run must reproduce its tokens, and a timing
            // dependent on/off decision would change how many random numbers
            // are drawn: it always drafts.
            let reproducible = !greedy_mode && seed_fixed;
            let spec_round = self.drafter.is_some() && (reproducible || self.governor.want_spec());
            let k = if spec_round {
                // A hybrid model can roll back only the rows of one bounded
                // verification call (`Model::rollback_rows`).
                self.draft_tokens
                    .min(budget - completion)
                    .min(room)
                    .min(self.model.rollback_rows().saturating_sub(1))
            } else {
                0
            };
            let mut drafts: Vec<u32> = Vec::new();
            let mut q_dists: Vec<Dist> = Vec::new();
            if k > 0
                && let (Some(drafter), Some(h)) = (&self.drafter, hidden.as_deref())
            {
                let mut pick = |l: &mut [f32]| -> u32 {
                    if greedy_mode {
                        greedy(l)
                    } else {
                        let q = Dist::warp(l, temperature, top_k, top_p, &mut scratch);
                        let d = q.draw(rng);
                        q_dists.push(q);
                        d
                    }
                };
                match drafter.draft(&self.model, h, id, self.cached.len(), k, &mut pick) {
                    Ok(d) => drafts = d,
                    Err(e) => {
                        let _ = events.blocking_send(Err(e));
                        self.reset();
                        return;
                    }
                }
            }

            // Verify: one forward over the pending token and the drafts.
            batch.clear();
            batch.push(id);
            batch.extend_from_slice(&drafts);
            let want = if drafts.is_empty() {
                Want::Last
            } else {
                Want::All
            };
            let start = self.cached.len();
            let forward = match self.model.forward_ext(&batch, start, want, tap, true) {
                Ok(f) => f,
                Err(e) => {
                    let _ = events.blocking_send(Err(e));
                    self.reset();
                    return;
                }
            };
            let Some(mut rows) = forward.logits else {
                let _ = events.blocking_send(Err(local_err("decode step produced no logits")));
                self.reset();
                return;
            };
            // `Want::Last` returns one row, `Want::All` one per batch token.
            let vocab = if drafts.is_empty() {
                rows.len()
            } else {
                rows.len() / batch.len()
            };

            // Accept the longest agreeing prefix of the drafts.
            let mut accepted = 0usize;
            for (j, &d) in drafts.iter().enumerate() {
                let row = &mut rows[j * vocab..(j + 1) * vocab];
                let agrees = if greedy_mode {
                    greedy(row) == d
                } else {
                    let p = Dist::warp(row, temperature, top_k, top_p, &mut scratch);
                    let q = &q_dists[j];
                    let ratio = p.prob(d) / q.prob(d);
                    if rng.next_f32() < ratio.min(1.0) {
                        true
                    } else {
                        forced = Some(match p.residual(q) {
                            Some(r) => r.draw(rng),
                            None => p.draw(rng),
                        });
                        false
                    }
                };
                if !agrees {
                    break;
                }
                accepted += 1;
            }
            if !drafts.is_empty() {
                self.stats.rounds.fetch_add(1, Ordering::Relaxed);
                self.stats
                    .drafted
                    .fetch_add(drafts.len() as u64, Ordering::Relaxed);
                self.stats
                    .accepted
                    .fetch_add(accepted as u64, Ordering::Relaxed);
            }

            // Cache `id` and the accepted drafts; the state for the next round
            // is the verified row after the last of them.
            let keep = start + 1 + accepted;
            if self.model.truncate_cache(keep) != keep {
                let _ = events.blocking_send(Err(local_err(format!(
                    "the model cannot roll its cache back to {keep} positions"
                ))));
                self.reset();
                return;
            }
            self.cached.push(id);
            self.cached.extend_from_slice(&drafts[..accepted]);
            // The in-trunk head takes the hidden rows of the tokens kept
            // (`id` and the accepted drafts; rejected drafts' rows are
            // dropped with them), in plain rounds too, so it never falls
            // behind the trunk.
            if observes
                && let (Some(drafter), Some(h)) = (&mut self.drafter, forward.hidden.as_deref())
            {
                let n = h.len() / batch.len();
                if let Err(e) = drafter.observe(
                    &self.model,
                    start,
                    &batch[..=accepted],
                    &h[..(accepted + 1) * n],
                ) {
                    let _ = events.blocking_send(Err(e));
                    self.reset();
                    return;
                }
            }
            // The verified row after the last kept token seeds the next round:
            // moved to the front of its buffer, not copied into a new one.
            rows.copy_within(accepted * vocab..(accepted + 1) * vocab, 0);
            rows.truncate(vocab);
            logits = rows;
            hidden = forward.hidden.map(|mut h| {
                let n = h.len() / batch.len();
                h.copy_within(accepted * n..(accepted + 1) * n, 0);
                h.truncate(n);
                h
            });

            // The measured round includes the cache rollback and the head's
            // upkeep: what speculation really costs per round.
            if self.drafter.is_some() && (!drafts.is_empty() || !spec_round) {
                self.governor.record(
                    !drafts.is_empty(),
                    1 + accepted,
                    round_start.elapsed().as_secs_f64(),
                );
            }

            // Emit the accepted drafts like sampled tokens.
            for &d in &drafts[..accepted] {
                if self.format.is_eog(d) {
                    break 'decode;
                }
                completion += 1;
                parser.push(d, &tokenizer, &mut next_call_id, &mut items);
                for item in items.drain(..) {
                    if events.blocking_send(item).is_err() {
                        return;
                    }
                }
                if completion == budget {
                    break 'decode;
                }
            }
        }

        // 6. Flush whatever the parser still holds.
        parser.finish(&mut next_call_id, &mut items);
        for item in items.drain(..) {
            if events.blocking_send(item).is_err() {
                return;
            }
        }

        // 7. Usage, then drop `events` to end the stream.
        metadata.set_token_usage(TokenUsage {
            prompt_tokens: saturating_u32(prompt_len),
            completion_tokens: saturating_u32(completion),
            reasoning_tokens: enable_thinking.then(|| parser.reasoning_tokens()),
        });
    }
}

fn saturating_u32(n: usize) -> u32 {
    u32::try_from(n).unwrap_or(u32::MAX)
}

/// The last of the `rows` equal-width rows held in `h`, kept in the same
/// buffer.
fn last_row(mut h: Vec<f32>, rows: usize) -> Vec<f32> {
    if rows > 1 {
        let width = h.len() / rows;
        h.drain(..h.len() - width);
    }
    h
}

/// Fixtures shared by the engine tests and the `LocalModel` tests.
#[cfg(test)]
pub(crate) mod testing {
    use super::*;
    use crate::provider::local::model::tiny::{TinyModel, VOCAB, load};

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
        let tokenizer = Arc::new(tiny_tokenizer());
        Engine {
            model: Model::Cpu(load(model, context_length)),
            format: tiny_format(&tokenizer),
            tokenizer,
            sampling: SamplingConfig {
                temperature,
                top_k: 64,
                top_p: 0.95,
                seed,
            },
            cached: Vec::new(),
            context_length,
            rng: SplitMix64(1),
            id_rng: SplitMix64(2),
            drafter: None,
            draft_tokens: 0,
            kv_bytes_per_position: 0,
            stats: Arc::default(),
            governor: Governor::default(),
        }
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
    /// file (a tiny `qwen35`), `draft_tokens` per round, and a governor that
    /// always drafts.
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
        e
    }
}

#[cfg(test)]
mod tests {
    use super::testing::{engine, engine_in_trunk, engine_with_drafter};
    use super::*;
    use crate::provider::local::model::tiny::{TinyQwen35, TinySpec, write_qwen35, write_spec};
    use crate::provider::local::model::tiny::{write_tiny_drafter, write_tiny_model};

    struct Outcome {
        items: Vec<Result<MessageContentBlock, CucaError>>,
        metadata: crate::client::ResponseMetadata,
    }

    fn run(e: &mut Engine, prompt: &[u32], max_tokens: Option<u32>) -> Outcome {
        let (tx, mut rx) = mpsc::channel(1024);
        let metadata = ResponseMetadataHandle::empty();
        e.run(Job {
            prompt_ids: prompt.to_vec(),
            start_in_thinking: false,
            enable_thinking: false,
            temperature: None,
            max_tokens,
            events: tx,
            metadata: metadata.clone(),
        });
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
        let model = write_tiny_model();
        let mut e = engine(&model, 32, 0.0, None);
        run(&mut e, &PROMPT, Some(3));

        // Same prompt again: everything but the last token is reused.
        let again = run(&mut e, &PROMPT, Some(3));
        assert_eq!(
            again.metadata.prompt_cache_usage,
            Some(PromptCacheUsage {
                read_tokens: 5,
                write_tokens: 1
            })
        );

        // Longer prompt: at least the old prompt is reused.
        let mut longer = PROMPT.to_vec();
        longer.extend([14, 15]);
        let out = run(&mut e, &longer, Some(3));
        let cache = out.metadata.prompt_cache_usage.unwrap();
        assert!(cache.read_tokens >= 6, "{cache:?}");
        assert_eq!(cache.read_tokens + cache.write_tokens, 8);

        // Diverging prompt: only the common head is reused.
        let out = run(&mut e, &[2, 9, 10, 15, 14], Some(3));
        let cache = out.metadata.prompt_cache_usage.unwrap();
        assert_eq!(cache.read_tokens, 3);
        assert_eq!(cache.write_tokens, 2);
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
        assert!(e.cached.is_empty());
        // The engine still serves the next job from a cold cache.
        let ok = run(&mut e, &PROMPT, Some(2));
        assert!(ok.metadata.token_usage.is_some());
        assert_eq!(ok.metadata.prompt_cache_usage.unwrap().read_tokens, 0);
    }

    #[test]
    fn dropped_consumer_stops_generation_without_corrupting_state() {
        let model = write_tiny_model();
        let mut e = engine(&model, 32, 0.0, None);
        let (tx, rx) = mpsc::channel(1);
        drop(rx);
        e.run(Job {
            prompt_ids: PROMPT.to_vec(),
            start_in_thinking: false,
            enable_thinking: false,
            temperature: None,
            max_tokens: Some(10),
            events: tx,
            metadata: ResponseMetadataHandle::empty(),
        });
        // `cached` mirrors the model cache, so the next job reuses it.
        assert_eq!(e.cached.len(), e.model.cache_len());
        let out = run(&mut e, &PROMPT, Some(2));
        assert!(out.metadata.token_usage.is_some());
        assert_eq!(out.metadata.prompt_cache_usage.unwrap().read_tokens, 5);
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
                assert_eq!(spec.cached.len(), spec.model.cache_len());
                // The cached ids equal plain decoding's: the K/V of rejected
                // drafts never leaks into the bookkeeping.
                assert_eq!(spec.cached, plain.cached, "k {k}, prompt {prompt:?}");
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

    /// What a request reported it reused and prefilled.
    fn usage(out: &Outcome) -> (u32, u32) {
        let u = out.metadata.prompt_cache_usage.expect("cache usage");
        (u.read_tokens, u.write_tokens)
    }

    /// Hybrid (recurrent-state) model: the engine reuses only what the model
    /// can return to, and reports what it really reused; the output stays
    /// exactly what a cold engine produces.
    #[test]
    fn hybrid_prefix_reuse_restarts_from_the_checkpoint_and_matches_cold() {
        let model = write_qwen35(&TinyQwen35::default());
        // Id 3 ends a turn (an end-of-generation token): the checkpoint goes
        // right after it, before the generation prompt.
        let first = [2u32, 9, 10, 3, 11, 12, 13];
        let mut warm = engine(&model, 32, 0.0, None);
        assert!(warm.model.is_hybrid());
        let a = run(&mut warm, &first, Some(4));
        assert_eq!(usage(&a), (0, 7));

        // The same conversation continues: the prompt shares the first five
        // tokens, but the model holds a state only at 4 (the checkpoint) and
        // at its end, so 4 are reused and 5 prefilled. The result is the cold
        // run's, bit for bit.
        let second = [2u32, 9, 10, 3, 11, 14, 15, 13, 12];
        let b = run(&mut warm, &second, Some(6));
        assert_eq!(usage(&b), (4, 5));
        let cold = run(&mut engine(&model, 32, 0.0, None), &second, Some(6));
        assert_eq!(usage(&cold), (0, 9));
        assert_eq!(text(&b), text(&cold));
        assert_eq!(
            b.metadata.token_usage.unwrap().completion_tokens,
            cold.metadata.token_usage.unwrap().completion_tokens
        );
        // `cached` mirrors the model cache whatever was restored.
        assert_eq!(warm.cached.len(), warm.model.cache_len());

        // The same prompt again: the last token is always prefilled, and the
        // model can only go back to its checkpoint, not to position 8.
        let c = run(&mut warm, &second, Some(6));
        assert_eq!(usage(&c), (4, 5));
        assert_eq!(text(&c), text(&cold));

        // An unrelated prompt (shares one token): nothing to restore.
        let d = run(&mut warm, &[2, 15, 14, 13, 12, 11, 10], Some(4));
        assert_eq!(usage(&d), (0, 7));
        let e = run(&mut warm, &first, Some(4));
        assert_eq!(usage(&e), (0, 7));
        assert_eq!(text(&e), text(&a));
    }

    #[test]
    fn hybrid_checkpoint_follows_the_last_end_of_turn_and_is_replaced_per_request() {
        let model = write_qwen35(&TinyQwen35::default());
        let mut e = engine(&model, 32, 0.0, None);
        // The last end-of-turn token (3) is at index 7: checkpoint at 8.
        let long = [2u32, 9, 3, 10, 11, 12, 13, 3, 14, 15];
        assert_eq!(e.checkpoint_position(&long), 8);
        assert_eq!(e.checkpoint_position(&[2, 9, 10]), 3, "none: the end");
        assert_eq!(e.checkpoint_position(&[3]), 1);
        run(&mut e, &long, Some(3));
        // Shares nine tokens: restarts at the checkpoint, 8.
        let mut next = long[..9].to_vec();
        next.extend([13, 12]);
        let out = run(&mut e, &next, Some(3));
        assert_eq!(usage(&out), (8, 3));
        assert_eq!(
            text(&out),
            text(&run(&mut engine(&model, 32, 0.0, None), &next, Some(3)))
        );
        // This request checkpointed at 8 again (its own last end of turn
        // is the same token): a third one sharing 10 tokens still gets 8.
        let mut third = next[..10].to_vec();
        third.extend([11, 10]);
        assert_eq!(usage(&run(&mut e, &third, Some(3))), (8, 4));

        // A raw prompt (no end-of-turn token) is checkpointed at its end,
        // so an extension reuses all of it.
        let mut e = engine(&model, 32, 0.0, None);
        let raw = [2u32, 9, 10, 11, 12, 13];
        run(&mut e, &raw, Some(3));
        let mut longer = raw.to_vec();
        longer.extend([14, 15]);
        let out = run(&mut e, &longer, Some(3));
        assert_eq!(usage(&out), (6, 2));
        assert_eq!(
            text(&out),
            text(&run(&mut engine(&model, 32, 0.0, None), &longer, Some(3)))
        );

        // The cached ids plus more tokens (an agent loop that appends): the
        // live state is the end, all of it is reused.
        let mut extended = e.cached.clone();
        let kept = extended.len();
        extended.extend([15, 14]);
        let out = run(&mut e, &extended, Some(2));
        assert_eq!(usage(&out), (u32::try_from(kept).unwrap(), 2));
        assert_eq!(
            text(&out),
            text(&run(&mut engine(&model, 32, 0.0, None), &extended, Some(2)))
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
        assert!(e.cached.is_empty());
        assert_eq!(e.model.cache_len(), 0);
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
                assert_eq!(spec.cached.len(), spec.model.cache_len());
                assert_eq!(spec.cached, plain.cached, "k {k}, prompt {prompt:?}");
                assert_eq!(
                    head_len(&spec),
                    spec.cached.len() - 1,
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
            assert_eq!(head_len(&spec), spec.cached.len() - 1, "max {max}");
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
                    spec.cached.starts_with(&plain.cached),
                    "ctx {ctx}, prompt {prompt:?}"
                );
                assert_eq!(head_len(&spec), spec.cached.len() - 1, "ctx {ctx}");
                let tokens = b.metadata.token_usage.map(|u| u.completion_tokens);
                filled += usize::from(tokens == u32::try_from(ctx - prompt.len()).ok());
            }
        }
        assert!(filled > 0, "some run stopped at the context limit");
    }

    /// Warm prefix reuse with the head equals a cold run, bit for bit: the
    /// trunk restarts from its checkpoint, the head keeps what the shared
    /// tokens still determine and re-observes the replayed rows.
    #[test]
    fn in_trunk_head_warm_prefix_reuse_equals_cold() {
        let model = write_qwen35(&TinyQwen35::default());
        let first = [2u32, 9, 10, 3, 11, 12, 13];
        let second = [2u32, 9, 10, 3, 11, 14, 15, 13, 12];
        let mut warm = engine_in_trunk(&model, 32, 0.0, None, 3);
        let a = run(&mut warm, &first, Some(6));
        assert_eq!(usage(&a), (0, 7));
        assert_eq!(head_len(&warm), warm.cached.len() - 1);

        // Restarts from the checkpoint at 4: nothing is missing in the head.
        let b = run(&mut warm, &second, Some(6));
        assert_eq!(usage(&b), (4, 5));
        let cold = run(
            &mut engine_in_trunk(&model, 32, 0.0, None, 3),
            &second,
            Some(6),
        );
        let plain = run(&mut engine(&model, 32, 0.0, None), &second, Some(6));
        assert_eq!(text(&b), text(&cold));
        assert_eq!(text(&b), text(&plain));
        assert_eq!(head_len(&warm), warm.cached.len() - 1);

        // The same prompt again goes back to the checkpoint too.
        let c = run(&mut warm, &second, Some(6));
        assert_eq!(usage(&c), (4, 5));
        assert_eq!(text(&c), text(&plain));
        assert_eq!(head_len(&warm), warm.cached.len() - 1);

        // The cached ids plus more tokens: the live state is the end, and
        // the head's tail bridges the request boundary.
        let mut extended = warm.cached.clone();
        extended.extend([15, 14]);
        let d = run(&mut warm, &extended, Some(4));
        assert_eq!(usage(&d).0 as usize, extended.len() - 2);
        let expected = run(&mut engine(&model, 32, 0.0, None), &extended, Some(4));
        assert_eq!(text(&d), text(&expected));
        assert_eq!(head_len(&warm), warm.cached.len() - 1);

        // A prompt that diverges right at the checkpoint leaves the pair
        // whose hidden row the trunk no longer has out: one position fewer,
        // the output unchanged.
        let diverging = [2u32, 9, 10, 3, 15, 14, 13];
        let e = run(&mut warm, &diverging, Some(5));
        assert_eq!(usage(&e), (4, 3));
        let expected = run(&mut engine(&model, 32, 0.0, None), &diverging, Some(5));
        assert_eq!(text(&e), text(&expected));
        assert!(head_len(&warm) < warm.cached.len() - 1);
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
        assert!(e.cached.is_empty());
        assert_eq!(head_len(&e), 0);
        let after = run(&mut e, &prompt, Some(4));
        assert_eq!(usage(&after), (0, 7));
        assert_eq!(text(&after), text(&before));
        assert_eq!(head_len(&e), e.cached.len() - 1);
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
        assert!(Model::load(&content, &model.path, resolved, device).is_ok());
        let e = Model::load(&content, &model.path, 33, device).err();
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
}
