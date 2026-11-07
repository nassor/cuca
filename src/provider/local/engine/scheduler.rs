//! The step scheduler: continuous batching over one shared K/V block pool.
//!
//! The engine thread keeps three lists of [`Sequence`]s: `waiting` (requests
//! not admitted yet, FIFO; a preempted one goes back to the front), `running`
//! (admitted, in admission order: the first is the oldest) and `draining`
//! (finished; only their output is still being delivered). One
//! [`Engine::turn`] is:
//!
//! 1. **Cancel** the sequences whose stream was dropped; their work stays in
//!    the prefix store.
//! 2. **Deliver** every sequence's pending output with `try_send`. A consumer
//!    that cannot keep up stalls its own sequence (it is skipped while its
//!    output is pending), never the others.
//! 3. **Admit** waiting requests while fewer than `max_sequences` run and the
//!    full-attention pool can hold their prompts next to what the running
//!    prompts still have to prefill. A prompt starts from the longest prefix
//!    the store holds ([`Engine::prefix_for`]).
//! 4. **Step**, in three parts:
//!    1. *Plan*: one forward pass over a row budget of `max_step_tokens`.
//!       Each decoding sequence with no output pending, oldest first, samples
//!       its next token from the logits the last pass left (or takes the one a
//!       rejected draft resampled), runs it through the sequence's output
//!       parser and stop matcher into its pending output, and takes one row
//!       (a lone sequence may verify drafted tokens, see [`Engine::draft`]); a
//!       reply that ends with the token leaves the plan. Then the prefilling
//!       sequences, in admission order, take chunks of their prompts with the
//!       rows left.
//!    2. **Deliver** again, before the forward pass: the tokens sampled while
//!       planning reach their streams while the pass runs, not after it.
//!    3. *Forward and apply*: run the pass over every planned row, then
//!       record its results per sequence (prefill progress, accepted draft
//!       tokens, the logits the next plan samples from) and retire the
//!       sequences that ended.
//! 5. **Deliver** a last time, so what the step produced goes out now, not
//!    after the next turn's cancel and deliver.
//!
//! A decoding sequence that needs a new block when none is left makes room by
//! preempting the youngest running sequence: its blocks go back to the
//! prefix store, it re-enters the queue with everything it had cached as its
//! prompt, and prefills again (mostly from the blocks it just released). The
//! oldest running sequence is never preempted, so it always finishes.

use std::collections::VecDeque;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::Ordering;
use std::time::Instant;

use tokio::sync::mpsc::{self, error::TrySendError};

use super::{Engine, Job, local_err, saturating_u32, take_block_draft, take_draft};
use crate::client::ResponseMetadataHandle;
use crate::error::CucaError;
use crate::provider::local::model::cpu::Hidden;
use crate::provider::local::model::mtp::Drafter;
use crate::provider::local::model::paged::{BLOCK_SIZE, BlockKey, SeqBlocks, block_keys};
use crate::provider::local::model::{Batch, RecurrentSnapshot, SeqRows, StateSlot, Want};
use crate::provider::local::output::OutputParser;
use crate::provider::local::sampler::{
    Dist, Scratch, SplitMix64, greedy, greedy_with_prob, sample,
};
use crate::request::{PromptCacheUsage, TokenUsage};
use crate::types::MessageContentBlock;

/// One output item of a stream.
pub(super) type Item = Result<MessageContentBlock, CucaError>;

/// Where a sequence is in its life.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    /// Queued, holding no blocks and no state slot.
    Waiting,
    /// Its prompt is being fed to the model in chunks.
    Prefill,
    /// Its prompt is cached and `logits` hold the distribution of the next
    /// token.
    Decode,
    /// Finished this step (successfully or not); it leaves `running` at the
    /// end of the step.
    Ended { failed: bool },
}

/// Cuts a reply's text at the request's stop strings.
struct Stopper {
    stop: super::stop::StopMatcher,
    /// A stop string has matched: nothing follows it.
    stopped: bool,
}

impl Stopper {
    /// Move `items` (parser output) to `out` in order. Text goes through the
    /// stop matcher; any other block first releases the text held back (it
    /// came earlier) and then passes through. Items after a stop match are
    /// dropped.
    fn push(&mut self, items: &mut Vec<Item>, out: &mut VecDeque<Item>) {
        for item in items.drain(..) {
            if self.stopped {
                continue;
            }
            if self.stop.is_passthrough() {
                out.push_back(item);
                continue;
            }
            match item {
                Ok(MessageContentBlock::Text(text)) => {
                    let cut = self.stop.push(&text);
                    if !cut.emit.is_empty() {
                        out.push_back(Ok(MessageContentBlock::Text(cut.emit)));
                    }
                    self.stopped = cut.done;
                }
                other => {
                    let held = self.stop.flush();
                    if !held.is_empty() {
                        out.push_back(Ok(MessageContentBlock::Text(held)));
                    }
                    out.push_back(other);
                }
            }
        }
    }

    /// Release the text still held back at an end that is not a stop match.
    fn finish(&mut self, out: &mut VecDeque<Item>) {
        let held = self.stop.flush();
        if !held.is_empty() {
            out.push_back(Ok(MessageContentBlock::Text(held)));
        }
    }
}

/// One request being served: its tokens, K/V blocks, sampling state and
/// output.
pub(super) struct Sequence {
    events: mpsc::Sender<Item>,
    metadata: ResponseMetadataHandle,
    phase: Phase,
    /// The tokens to prefill: the request's prompt, or, after a preemption,
    /// everything the sequence had cached.
    prompt: Vec<u32>,
    /// Length of the request's prompt (the usage it reports).
    prompt_len: usize,
    /// The tokens whose K/V the blocks hold, at positions `0..`: the prefix
    /// of `prompt` prefilled so far while prefilling.
    ids: Vec<u32>,
    blocks: SeqBlocks,
    /// The model's recurrent-state slot while running.
    slot: Option<usize>,
    /// A hybrid model's snapshot position (a block boundary) the prefill
    /// takes.
    snapshot_at: Option<usize>,
    /// Whether the prompt-cache usage was reported (once, at the first
    /// admission).
    usage_reported: bool,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    /// A request or load-time seed: this sequence draws from its own
    /// generator.
    seeded: Option<SplitMix64>,
    scratch: Scratch,
    parser: OutputParser,
    stopper: Stopper,
    enable_thinking: bool,
    /// Tokens generated so far, and the most this request may generate.
    completion: usize,
    budget: usize,
    /// The distribution of the next token (decode), and the final-norm hidden
    /// row it came from (a drafter's start).
    logits: Vec<f32>,
    hidden: Option<Vec<f32>>,
    /// The token a rejected speculative draft was resampled to: the next
    /// token, taken instead of sampling the verified logits.
    forced: Option<u32>,
    /// Output the consumer has not taken yet.
    pending: VecDeque<Item>,
}

impl Sequence {
    pub(super) fn new(engine: &Engine, job: Job) -> Self {
        let Job {
            prompt_ids,
            start_in_thinking,
            enable_thinking,
            temperature,
            max_tokens,
            top_p,
            top_k,
            seed,
            stop,
            events,
            metadata,
        } = job;
        let prompt_len = prompt_ids.len();
        let sampling = &engine.sampling;
        // A request seed wins over the load-time one; either restarts the
        // generator for this request alone.
        let seed = seed.or(sampling.seed);
        Self {
            events,
            metadata,
            phase: Phase::Waiting,
            prompt: prompt_ids,
            prompt_len,
            ids: Vec::new(),
            blocks: engine.pools.seq(),
            slot: None,
            snapshot_at: None,
            usage_reported: false,
            temperature: temperature.unwrap_or(sampling.temperature),
            top_k: top_k.map_or(sampling.top_k, |k| k as usize),
            top_p: top_p.unwrap_or(sampling.top_p),
            seeded: seed.map(SplitMix64),
            scratch: Scratch::default(),
            parser: OutputParser::new(&engine.format, start_in_thinking),
            stopper: Stopper {
                stop: super::stop::StopMatcher::new(&stop),
                stopped: false,
            },
            enable_thinking,
            completion: 0,
            budget: max_tokens
                .map_or(usize::MAX, |n| n as usize)
                .min(engine.context_length.saturating_sub(prompt_len)),
            logits: Vec::new(),
            hidden: None,
            forced: None,
            pending: VecDeque::new(),
        }
    }

    /// Whether the request draws reproducible tokens (sampling with a fixed
    /// seed).
    fn reproducible(&self) -> bool {
        self.temperature > 0.0 && self.seeded.is_some()
    }
}

/// The hybrid-model prefix snapshots the engine keeps: the recurrent state
/// after a block-aligned prefix, named by the key of the prefix's last block
/// (which names the whole prefix). Bounded; the least recently used goes
/// first.
pub(super) struct SnapshotStore {
    entries: Vec<(BlockKey, RecurrentSnapshot, u64)>,
    cap: usize,
    clock: u64,
}

impl SnapshotStore {
    pub(super) fn new(cap: usize) -> Self {
        Self {
            entries: Vec::new(),
            cap: cap.max(1),
            clock: 0,
        }
    }

    pub(super) fn contains(&self, key: &BlockKey) -> bool {
        self.entries.iter().any(|(k, _, _)| k == key)
    }

    fn get(&mut self, key: &BlockKey) -> Option<&RecurrentSnapshot> {
        self.clock += 1;
        let clock = self.clock;
        self.entries
            .iter_mut()
            .find(|(k, _, _)| k == key)
            .map(|(_, s, stamp)| {
                *stamp = clock;
                &*s
            })
    }

    fn insert(&mut self, key: BlockKey, snapshot: RecurrentSnapshot) {
        self.clock += 1;
        if let Some(entry) = self.entries.iter_mut().find(|(k, _, _)| *k == key) {
            *entry = (key, snapshot, self.clock);
            return;
        }
        if self.entries.len() >= self.cap
            && let Some(oldest) = self
                .entries
                .iter()
                .enumerate()
                .min_by_key(|(_, (_, _, stamp))| *stamp)
                .map(|(i, _)| i)
        {
            self.entries.swap_remove(oldest);
        }
        self.entries.push((key, snapshot, self.clock));
    }

    pub(super) fn clear(&mut self) {
        self.entries.clear();
    }

    /// Bytes the snapshots hold.
    pub(super) fn bytes(&self) -> usize {
        self.entries.iter().map(|(_, s, _)| s.bytes()).sum()
    }

    #[cfg(test)]
    pub(super) fn len(&self) -> usize {
        self.entries.len()
    }
}

/// What planning a step decided for one sequence.
struct Plan {
    /// Index into `running`.
    seq: usize,
    start: usize,
    tokens: Vec<u32>,
    want: Want,
    hidden: Hidden,
    kind: Kind,
}

enum Kind {
    /// A prompt chunk; `last` ends the prompt.
    Prefill { last: bool },
    /// The pending token (first of `tokens`) and the drafts after it.
    Decode(Round),
}

/// The speculative round of a decoding sequence: its drafts and what the
/// cost model needs to learn from it.
struct Round {
    q_dists: Vec<Dist>,
    spec_round: bool,
    draft_steps: usize,
    /// Drafter passes the round cost: one per step for a chained drafter,
    /// one for the whole block of a block drafter (zero when it did not run).
    draft_passes: usize,
    draft_secs: f64,
    predicted: f64,
    started: Instant,
}

impl Engine {
    /// Queue a request. A prompt the context cannot hold fails at once.
    pub(super) fn submit(&mut self, job: Job) {
        let prompt_len = job.prompt_ids.len();
        if prompt_len == 0 || prompt_len >= self.context_length {
            let _ = job.events.try_send(Err(local_err(format!(
                "prompt has {prompt_len} tokens; context_length is {}",
                self.context_length
            ))));
            return;
        }
        let seq = Sequence::new(self, job);
        self.waiting.push_back(seq);
    }

    /// Whether nothing is queued, running or still being delivered.
    pub(super) fn idle(&self) -> bool {
        self.running.is_empty() && self.waiting.is_empty() && self.draining.is_empty()
    }

    /// Requests waiting for a slot.
    pub(super) fn queued(&self) -> usize {
        self.waiting.len()
    }

    /// One turn of the scheduler (see the module docs): returns whether
    /// anything happened. A panic inside is caught: every running sequence
    /// gets a provider error and the pools are reset, so later requests
    /// still work.
    pub(super) fn turn(&mut self) -> bool {
        let progressed = match catch_unwind(AssertUnwindSafe(|| self.turn_inner())) {
            Ok(progressed) => progressed,
            Err(_) => {
                self.fail_running(&local_err("local engine panicked; KV cache reset"));
                true
            }
        };
        self.publish_gauges();
        progressed
    }

    fn turn_inner(&mut self) -> bool {
        let mut progressed = self.cancel_closed();
        progressed |= self.deliver();
        progressed |= self.admit();
        progressed |= self.step();
        // What the step produced goes out now, not after the next step.
        progressed |= self.deliver();
        progressed
    }

    /// Run one request to completion (alone, or with whatever else is
    /// queued).
    #[cfg(test)]
    pub(super) fn run(&mut self, job: Job) {
        self.submit(job);
        self.run_until_idle();
    }

    /// Turn until nothing is left.
    #[cfg(test)]
    pub(super) fn run_until_idle(&mut self) {
        while !self.idle() {
            if !self.turn() {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        }
    }

    /// The engine thread's loop: take requests from `jobs` and turn the
    /// scheduler until the channel is closed and every request is done.
    pub(crate) fn run_loop(&mut self, jobs: &mut mpsc::Receiver<Job>) {
        let mut open = true;
        loop {
            if self.idle() {
                if !open {
                    return;
                }
                match jobs.blocking_recv() {
                    Some(job) => self.submit(job),
                    None => return,
                }
            }
            // The channel's bound is the queue's bound: take no more while
            // enough wait.
            while open && self.queued() < super::super::JOB_QUEUE {
                match jobs.try_recv() {
                    Ok(job) => self.submit(job),
                    Err(mpsc::error::TryRecvError::Empty) => break,
                    Err(mpsc::error::TryRecvError::Disconnected) => open = false,
                }
            }
            if !self.turn() && !self.idle() {
                // Everything left waits for a consumer to take its output.
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        }
    }

    fn publish_gauges(&self) {
        let s = &self.serving;
        let pool = self.pools.full_stats();
        s.running.store(self.running.len(), Ordering::Relaxed);
        s.waiting.store(self.waiting.len(), Ordering::Relaxed);
        s.free_blocks.store(pool.free, Ordering::Relaxed);
        s.cached_blocks.store(pool.cached, Ordering::Relaxed);
        s.snapshot_bytes
            .store(self.snapshots.bytes() as u64, Ordering::Relaxed);
    }

    /// Where a hybrid model snapshots its recurrent state while prefilling
    /// `prompt`, so a later request can restart from its shared prefix.
    ///
    /// The next request of a conversation repeats this prompt only up to the
    /// end of its last complete message: what follows is the generation
    /// prompt (an assistant header, maybe an empty reasoning block), which
    /// templates rewrite once the answer is part of the history. A snapshot
    /// at the very end of the prompt would sit past the shared prefix and
    /// restore nothing. The end of the last message is the position after the
    /// last end-of-turn token; a prompt without one (a raw completion) is
    /// expected to be extended, so its end serves. The position rounds down
    /// to a block boundary (the prefix store shares whole blocks) and stays
    /// below the last prompt token (its logits start generation).
    pub(super) fn snapshot_position(&self, prompt: &[u32]) -> usize {
        let at = prompt
            .iter()
            .rposition(|&id| self.format.is_eog(id))
            .map_or(prompt.len(), |i| i + 1);
        at.min(prompt.len().saturating_sub(1)) / BLOCK_SIZE * BLOCK_SIZE
    }

    // ---- cancellation and delivery ----

    /// Drop every sequence whose consumer has gone away. A running one files
    /// its blocks in the prefix store first: its work is reusable.
    fn cancel_closed(&mut self) -> bool {
        let before = self.running.len() + self.waiting.len() + self.draining.len();
        self.waiting.retain(|s| !s.events.is_closed());
        self.draining.retain(|s| !s.events.is_closed());
        let mut i = 0;
        while i < self.running.len() {
            if self.running[i].events.is_closed() {
                let mut seq = self.running.remove(i);
                self.release_seq(&mut seq, true);
            } else {
                i += 1;
            }
        }
        before != self.running.len() + self.waiting.len() + self.draining.len()
    }

    /// Hand every sequence's pending output to its stream without waiting.
    /// A sequence whose stream is full keeps the rest for the next turn; a
    /// drained finished sequence is dropped, which ends its stream.
    fn deliver(&mut self) -> bool {
        fn flush(seq: &mut Sequence) -> bool {
            let mut sent = false;
            while let Some(item) = seq.pending.pop_front() {
                match seq.events.try_send(item) {
                    Ok(()) => sent = true,
                    Err(TrySendError::Full(item)) => {
                        seq.pending.push_front(item);
                        break;
                    }
                    Err(TrySendError::Closed(_)) => {
                        seq.pending.clear();
                        break;
                    }
                }
            }
            sent
        }
        let mut progressed = false;
        for seq in self.running.iter_mut().chain(self.draining.iter_mut()) {
            progressed |= flush(seq);
        }
        let before = self.draining.len();
        self.draining.retain(|s| !s.pending.is_empty());
        progressed || before != self.draining.len()
    }

    // ---- admission ----

    /// Blocks the full-attention pool still owes the prompts being
    /// prefilled.
    fn committed(&self) -> usize {
        self.running
            .iter()
            .filter(|s| s.phase == Phase::Prefill)
            .map(|s| {
                (s.prompt.len() + 1)
                    .div_ceil(BLOCK_SIZE)
                    .saturating_sub(self.pools.full_blocks(&s.blocks))
            })
            .sum()
    }

    /// The longest prefix of `prompt` the store can resume from, with a
    /// reference to its blocks: whole blocks of the full-attention pool (and
    /// the sliding windows before its end); a hybrid model also needs the
    /// recurrent-state snapshot taken at exactly that position, so the prefix
    /// shortens to the nearest block boundary that has one; an in-trunk
    /// drafter's cache must hold the same tokens, which bounds it by what the
    /// head last saw.
    fn prefix_for(&mut self, prompt: &[u32]) -> SeqBlocks {
        if !self.batching.prefix_cache {
            return self.pools.seq();
        }
        let mut blocks = self.pools.lookup(prompt).into_seq();
        if self.drafter.as_ref().is_some_and(Drafter::observes) {
            let shared = self
                .drafter_ids
                .iter()
                .zip(prompt)
                .take_while(|(a, b)| a == b)
                .count();
            self.pools
                .truncate(&mut blocks, shared / BLOCK_SIZE * BLOCK_SIZE);
        }
        if self.model.is_hybrid() {
            let keys = block_keys(prompt, blocks.len / BLOCK_SIZE);
            let at = (1..=keys.len())
                .rev()
                .find(|&n| self.snapshots.contains(&keys[n - 1]))
                .unwrap_or(0);
            self.pools.truncate(&mut blocks, at * BLOCK_SIZE);
            if blocks.len != at * BLOCK_SIZE {
                self.pools.truncate(&mut blocks, 0);
            }
        }
        blocks
    }

    /// Start waiting requests while there is a slot and room for their
    /// prompts (FIFO: a request that does not fit yet holds the ones behind
    /// it).
    fn admit(&mut self) -> bool {
        let mut admitted = false;
        while self.running.len() < self.batching.max_sequences
            && let Some(prompt_len) = self.waiting.front().map(|s| s.prompt.len())
        {
            let prompt = std::mem::take(&mut self.waiting[0].prompt);
            let mut blocks = self.prefix_for(&prompt);
            let need = (prompt_len + 1)
                .div_ceil(BLOCK_SIZE)
                .saturating_sub(self.pools.full_blocks(&blocks));
            if self.pools.available() < self.committed() + need {
                self.pools.release(&mut blocks);
                self.waiting[0].prompt = prompt;
                break;
            }
            let Some(mut seq) = self.waiting.pop_front() else {
                break;
            };
            let Some(slot) = self.free_slots.pop() else {
                self.pools.release(&mut blocks);
                seq.prompt = prompt;
                self.waiting.push_front(seq);
                break;
            };
            let reuse = blocks.len;
            // A hybrid model resumes from the snapshot of exactly `reuse`.
            let restored = if reuse == 0 {
                self.model.restore(StateSlot(slot), None, 0)
            } else if self.model.is_hybrid() {
                let key = block_keys(&prompt, reuse / BLOCK_SIZE)
                    .pop()
                    .and_then(|k| self.snapshots.get(&k).map(|s| (k, s)));
                match key {
                    Some((_, snapshot)) => {
                        self.model.restore(StateSlot(slot), Some(snapshot), reuse)
                    }
                    None => Err(local_err("the prefix has no recurrent-state snapshot")),
                }
            } else {
                Ok(())
            };
            if let Err(e) = restored {
                self.free_slots.push(slot);
                self.pools.release(&mut blocks);
                seq.pending.push_back(Err(e));
                seq.phase = Phase::Ended { failed: true };
                self.draining.push(seq);
                admitted = true;
                continue;
            }
            // An in-trunk head keeps the pairs the shared tokens still
            // determine, `reuse` and more; the replayed rows replace the
            // rest.
            if let Some(drafter) = &mut self.drafter {
                let shared = self
                    .drafter_ids
                    .iter()
                    .zip(&prompt)
                    .take_while(|(a, b)| a == b)
                    .count();
                drafter.truncate(shared);
            }
            self.serving
                .prefix_hit_tokens
                .fetch_add(reuse as u64, Ordering::Relaxed);
            if !seq.usage_reported {
                seq.usage_reported = true;
                seq.metadata.set_prompt_cache_usage(PromptCacheUsage {
                    read_tokens: saturating_u32(reuse),
                    write_tokens: saturating_u32(prompt_len - reuse),
                });
            }
            seq.snapshot_at = (self.model.is_hybrid() && self.batching.prefix_cache)
                .then(|| self.snapshot_position(&prompt))
                .filter(|&at| at > reuse);
            seq.ids = prompt[..reuse].to_vec();
            seq.prompt = prompt;
            seq.blocks = blocks;
            seq.slot = Some(slot);
            seq.phase = Phase::Prefill;
            self.running.push(seq);
            admitted = true;
        }
        admitted
    }

    // ---- ending a sequence ----

    /// Give back what `seq` holds: its blocks (filed in the prefix store when
    /// `share`) and its state slot.
    fn release_seq(&mut self, seq: &mut Sequence, share: bool) {
        if share && self.batching.prefix_cache {
            self.pools.finish(&mut seq.blocks, &seq.ids);
        } else {
            self.pools.release(&mut seq.blocks);
        }
        if let Some(slot) = seq.slot.take() {
            self.free_slots.push(slot);
        }
    }

    /// Send the end of `seq`'s reply: what the parser and the stop matcher
    /// still hold, unless a stop string ended it, and the usage.
    fn finish_reply(&mut self, seq: &mut Sequence) {
        if !seq.stopper.stopped {
            let mut items = Vec::new();
            let id_rng = &mut self.id_rng;
            let mut next_call_id = || format!("call_{:016x}", id_rng.next_u64());
            seq.parser.finish(&mut next_call_id, &mut items);
            seq.stopper.push(&mut items, &mut seq.pending);
            seq.stopper.finish(&mut seq.pending);
        }
        seq.metadata.set_token_usage(TokenUsage {
            prompt_tokens: saturating_u32(seq.prompt_len),
            completion_tokens: saturating_u32(seq.completion),
            reasoning_tokens: seq.enable_thinking.then(|| seq.parser.reasoning_tokens()),
        });
    }

    /// Move the sequences that ended this step from `running` to
    /// `draining`.
    fn retire_ended(&mut self) {
        let mut i = 0;
        while i < self.running.len() {
            let Phase::Ended { failed } = self.running[i].phase else {
                i += 1;
                continue;
            };
            let mut seq = self.running.remove(i);
            #[cfg(test)]
            {
                self.last_ids = if failed { Vec::new() } else { seq.ids.clone() };
            }
            if !failed {
                self.finish_reply(&mut seq);
            }
            self.release_seq(&mut seq, !failed);
            self.draining.push(seq);
        }
    }

    /// End `seq` with `error` as the last item of its stream.
    fn fail(&mut self, i: usize, error: CucaError) {
        let seq = &mut self.running[i];
        seq.pending.push_back(Err(error));
        seq.phase = Phase::Ended { failed: true };
    }

    /// Every running sequence ends with `error`, and the pools, the
    /// snapshots and the drafter forget everything: after a failed pass the
    /// K/V of the running sequences is not trusted.
    fn fail_running(&mut self, error: &CucaError) {
        for seq in &mut self.running {
            seq.pending.push_back(Err(error.clone()));
            seq.phase = Phase::Ended { failed: true };
        }
        self.retire_ended();
        // Waiting sequences hold no blocks; a failed pass can leave anything
        // in the pools.
        self.pools.reset();
        self.snapshots.clear();
        self.drafter_ids.clear();
        if let Some(drafter) = &mut self.drafter {
            drafter.reset();
        }
    }

    /// Put the running sequence `i` back at the front of the queue and give
    /// its blocks back to the prefix store.
    fn preempt(&mut self, i: usize) {
        let mut seq = self.running.remove(i);
        self.release_seq(&mut seq, true);
        if seq.phase == Phase::Decode {
            // Everything it cached is the prompt now; its logits come back
            // with the last chunk.
            seq.prompt.clone_from(&seq.ids);
        }
        seq.ids.clear();
        seq.logits = Vec::new();
        seq.hidden = None;
        seq.phase = Phase::Waiting;
        self.serving.preemptions.fetch_add(1, Ordering::Relaxed);
        self.waiting.push_front(seq);
    }

    /// Make sure running sequence `i` has the block its next row goes in,
    /// preempting younger sequences for it, or itself when it is the
    /// youngest. `Ok(false)`: it was preempted. `Err`: nothing can be freed.
    fn ensure_block(&mut self, i: usize) -> Result<bool, CucaError> {
        loop {
            let seq = &mut self.running[i];
            let upto = seq.ids.len() + 1;
            if self.pools.reserve(&mut seq.blocks, upto).is_ok() {
                return Ok(true);
            }
            let youngest = self.running.len() - 1;
            if youngest > i {
                self.preempt(youngest);
            } else if i > 0 {
                self.preempt(i);
                return Ok(false);
            } else {
                return Err(local_err(
                    "the K/V block pool is exhausted by a single sequence",
                ));
            }
        }
    }

    // ---- one step ----

    /// Plan, run and apply one forward step; returns whether work was done.
    fn step(&mut self) -> bool {
        let mut progressed = false;
        let observes = self.drafter.as_ref().is_some_and(Drafter::observes);
        let budget = self.batching.max_step_tokens;
        let mut plans: Vec<Plan> = Vec::new();
        let mut used = 0usize;

        // Decoding sequences, oldest first: one row each (a lone sequence
        // may draft more).
        let mut i = 0;
        while i < self.running.len() {
            if self.running[i].phase != Phase::Decode
                || !self.running[i].pending.is_empty()
                || used >= budget
            {
                i += 1;
                continue;
            }
            match self.ensure_block(i) {
                Ok(true) => {}
                Ok(false) => {
                    progressed = true;
                    continue;
                }
                Err(e) => {
                    self.fail(i, e);
                    progressed = true;
                    i += 1;
                    continue;
                }
            }
            if let Some(plan) = self.plan_decode(i, budget - used) {
                used += plan.tokens.len();
                plans.push(plan);
            } else {
                progressed = true;
            }
            i += 1;
        }

        // Prefilling sequences, in admission order, share what is left.
        let chunk = crate::provider::local::model::prefill_chunk(self.context_length);
        let tap = self.drafter.is_some() && self.running.len() == 1;
        let features = self.drafter.as_ref().and_then(Drafter::feature_rows);
        for i in 0..self.running.len() {
            let seq = &self.running[i];
            if seq.phase != Phase::Prefill {
                continue;
            }
            let room = budget.saturating_sub(used);
            if room == 0 {
                break;
            }
            let pos = seq.ids.len();
            let mut n = (seq.prompt.len() - pos).min(chunk).min(room);
            // A chunk ends where the snapshot goes.
            if let Some(at) = seq.snapshot_at.filter(|&at| pos < at && at < pos + n) {
                n = at - pos;
            }
            let end = pos + n;
            let last = end == seq.prompt.len();
            let tokens = seq.prompt[pos..end].to_vec();
            if let Err(e) = self.pools.reserve(&mut self.running[i].blocks, end) {
                self.fail(i, e);
                progressed = true;
                continue;
            }
            used += n;
            plans.push(Plan {
                seq: i,
                start: pos,
                tokens,
                want: if last { Want::Last } else { Want::None },
                hidden: if let Some(rows) = features {
                    rows
                } else if observes {
                    Hidden::PostNormAll
                } else if last && tap {
                    Hidden::PostNorm
                } else {
                    Hidden::None
                },
                kind: Kind::Prefill { last },
            });
        }

        if plans.is_empty() {
            self.retire_ended();
            return progressed;
        }

        // The tokens planned above go out before the forward pass that
        // computes the next ones, as a sequence's first token would with a
        // blocking send: a step costs tens of milliseconds.
        self.deliver();
        let started = Instant::now();
        let outs = {
            let batch = Batch {
                rows: plans
                    .iter()
                    .map(|p| {
                        let seq = &self.running[p.seq];
                        SeqRows {
                            tokens: &p.tokens,
                            start_pos: p.start,
                            tables: &seq.blocks.tables,
                            state: StateSlot(seq.slot.unwrap_or(0)),
                            want: p.want,
                            hidden: p.hidden,
                        }
                    })
                    .collect(),
                exact: true,
            };
            self.model.forward_batch(&batch)
        };
        let outs = match outs {
            Ok(outs) => outs,
            Err(e) => {
                self.fail_running(&e);
                return true;
            }
        };
        self.serving.steps.fetch_add(1, Ordering::Relaxed);
        self.serving
            .batched_rows
            .fetch_add(used as u64, Ordering::Relaxed);

        for (plan, out) in plans.into_iter().zip(outs) {
            match plan.kind {
                Kind::Prefill { last } => self.apply_prefill(
                    plan.seq,
                    plan.start,
                    &plan.tokens,
                    last,
                    out.logits,
                    out.hidden,
                ),
                Kind::Decode(round) => self.apply_decode(
                    plan.seq,
                    plan.start,
                    plan.tokens,
                    plan.want == Want::All,
                    round,
                    out.logits,
                    out.hidden,
                    started,
                ),
            }
        }
        if observes && let Some(seq) = self.running.first() {
            self.drafter_ids.clone_from(&seq.ids);
        }
        self.retire_ended();
        true
    }

    /// Take the next token of decoding sequence `i` (sampled, or the one a
    /// rejected draft resampled), emit it, and plan its row: the token and,
    /// for a lone sequence, the tokens the drafter proposes after it. `None`
    /// when the reply ended with the token (the sequence is marked ended).
    fn plan_decode(&mut self, i: usize, room_rows: usize) -> Option<Plan> {
        let round_start = Instant::now();
        let ctx = self.context_length;
        let seq = &mut self.running[i];
        let shared_rng = &mut self.rng;
        let rng = seq.seeded.as_mut().unwrap_or(shared_rng);
        let id = match seq.forced.take() {
            Some(id) => id,
            None => sample(
                &mut seq.logits,
                seq.temperature,
                seq.top_k,
                seq.top_p,
                rng,
                &mut seq.scratch,
            ),
        };
        if self.format.is_eog(id) {
            seq.phase = Phase::Ended { failed: false };
            return None;
        }
        seq.completion += 1;
        {
            let mut items = Vec::new();
            let id_rng = &mut self.id_rng;
            let mut next_call_id = || format!("call_{:016x}", id_rng.next_u64());
            seq.parser
                .push(id, &self.tokenizer, &mut next_call_id, &mut items);
            seq.stopper.push(&mut items, &mut seq.pending);
        }
        if seq.stopper.stopped || seq.completion == seq.budget || seq.ids.len() == ctx {
            seq.phase = Phase::Ended { failed: false };
            return None;
        }
        let observes = self.drafter.as_ref().is_some_and(Drafter::observes);
        let alone = self.running.len() == 1 && self.waiting.is_empty();
        let tap = observes || (self.drafter.is_some() && self.running.len() == 1);
        let features = self.drafter.as_ref().and_then(Drafter::feature_rows);
        let seq = &mut self.running[i];
        let mut tokens = vec![id];
        let mut round = Round {
            q_dists: Vec::new(),
            spec_round: false,
            draft_steps: 0,
            draft_passes: 0,
            draft_secs: 0.0,
            predicted: 0.0,
            started: round_start,
        };

        // Draft: `id` sits at position `ids.len()` and is not cached. Draft
        // only while the governor measures speculation as faster than plain
        // decoding, and a seeded sampling run must reproduce its tokens: a
        // timing dependent on/off decision or draft length would change how
        // many random numbers are drawn, so it always drafts the full cap.
        if alone && self.drafter.is_some() {
            let reproducible = seq.reproducible();
            round.spec_round = reproducible || self.governor.want_spec();
            let room = ctx - seq.ids.len() - 1;
            // A hybrid model can roll back only the rows of one bounded
            // verification call (`Model::rollback_rows`).
            let k = if round.spec_round {
                self.draft_tokens
                    .min(seq.budget - seq.completion)
                    .min(room)
                    .min(self.model.rollback_rows().saturating_sub(1))
                    .min(room_rows.saturating_sub(1))
            } else {
                0
            };
            if k > 0
                && let (Some(drafter), Some(h)) = (&self.drafter, seq.hidden.as_deref())
            {
                let greedy_mode = seq.temperature <= 0.0;
                let (temperature, top_k, top_p) = (seq.temperature, seq.top_k, seq.top_p);
                let shared_rng = &mut self.rng;
                let rng = seq.seeded.as_mut().unwrap_or(shared_rng);
                // Draft `j` is kept while its predicted acceptance (the
                // drafter's chained confidence, calibrated) makes the round's
                // expected tokens per second rise (`take_draft`); `k` is the
                // cap.
                let adaptive = !reproducible;
                let cost = &self.cost;
                let block = drafter.block();
                let kappa = cost.calibration();
                let step_secs = cost.draft_secs();
                let mut chained = 1.0f64;
                let mut expected = 1.0f64;
                let Round {
                    q_dists,
                    draft_steps,
                    predicted,
                    ..
                } = &mut round;
                let scratch = &mut seq.scratch;
                let mut pick = |l: &mut [f32]| -> Option<u32> {
                    *draft_steps += 1;
                    let j = *draft_steps;
                    let (d, q) = if greedy_mode {
                        greedy_with_prob(l)
                    } else {
                        let dist = Dist::warp(l, temperature, top_k, top_p, scratch);
                        let d = dist.draw(rng);
                        let q = dist.prob(d);
                        q_dists.push(dist);
                        (d, q)
                    };
                    chained *= f64::from(q);
                    // The first draft of a speculative round is always
                    // verified: the governor already decided that the round
                    // drafts, and its acceptance keeps the calibration live.
                    if adaptive {
                        let a = (kappa * chained).min(1.0);
                        let (t_j, t_next) = (cost.verify_secs(j), cost.verify_secs(j + 1));
                        let keep = if block {
                            take_block_draft(a, expected, t_j, t_next, step_secs)
                        } else {
                            take_draft(a, expected, j, t_j, t_next, step_secs)
                        };
                        if j > 1 && !keep {
                            return None;
                        }
                        expected += a;
                    }
                    *predicted += chained;
                    Some(d)
                };
                let draft_start = Instant::now();
                match drafter.draft(
                    &self.model,
                    &seq.blocks.tables,
                    h,
                    id,
                    seq.ids.len(),
                    k,
                    &mut pick,
                ) {
                    Ok(d) => tokens.extend_from_slice(&d),
                    Err(e) => {
                        self.fail(i, e);
                        return None;
                    }
                }
                round.draft_secs = draft_start.elapsed().as_secs_f64();
                // A block drafter runs one pass however many drafts `pick`
                // took, and none when it had nothing to condition on.
                round.draft_passes = if block {
                    usize::from(round.draft_steps > 0)
                } else {
                    round.draft_steps
                };
            }
        }

        let seq = &mut self.running[i];
        let drafted = tokens.len() > 1;
        let start = seq.ids.len();
        if let Err(e) = self.pools.reserve(&mut seq.blocks, start + tokens.len()) {
            self.fail(i, e);
            return None;
        }
        Some(Plan {
            seq: i,
            start,
            want: if drafted { Want::All } else { Want::Last },
            hidden: if let Some(rows) = features {
                rows
            } else if tap {
                Hidden::PostNorm
            } else {
                Hidden::None
            },
            tokens,
            kind: Kind::Decode(round),
        })
    }

    /// Record what a prefill chunk computed.
    fn apply_prefill(
        &mut self,
        i: usize,
        pos: usize,
        tokens: &[u32],
        last: bool,
        logits: Option<Vec<f32>>,
        hidden: Option<Vec<f32>>,
    ) {
        let observes = self.drafter.as_ref().is_some_and(Drafter::observes);
        let share = self.batching.prefix_cache;
        let end = pos + tokens.len();
        if observes
            && let (Some(drafter), Some(rows)) = (&mut self.drafter, hidden.as_deref())
            && let Err(e) = drafter.observe(&self.model, pos, tokens, rows)
        {
            self.fail(i, e);
            return;
        }
        let seq = &mut self.running[i];
        seq.ids.extend_from_slice(tokens);
        seq.blocks.len = end;
        // Sliding groups give back what left the window, filing it in the
        // store on the way; every full block is named from now on, so a
        // prompt that arrives while this one still runs can share them.
        let ids = share.then_some(seq.ids.as_slice());
        self.pools.release_window(&mut seq.blocks, end, ids);
        if share {
            self.pools.publish(&mut seq.blocks, &seq.ids);
        }
        if seq.snapshot_at == Some(end)
            && let Some(slot) = seq.slot
            && let Some(snapshot) = self.model.snapshot(StateSlot(slot))
            && let Some(key) = block_keys(&seq.ids, end / BLOCK_SIZE).pop()
        {
            self.snapshots.insert(key, snapshot);
        }
        if !last {
            return;
        }
        let Some(logits) = logits else {
            self.fail(i, local_err("prefill produced no logits"));
            return;
        };
        let seq = &mut self.running[i];
        seq.logits = logits;
        // Only the last row seeds the first draft.
        seq.hidden = hidden.map(|h| {
            if observes {
                last_row(h, tokens.len())
            } else {
                h
            }
        });
        seq.phase = if seq.budget == 0 {
            Phase::Ended { failed: false }
        } else {
            Phase::Decode
        };
    }

    /// Accept what the verification pass of a decode round computed.
    #[expect(
        clippy::too_many_arguments,
        reason = "one round's plan and result; a struct would only rename them"
    )]
    fn apply_decode(
        &mut self,
        i: usize,
        start: usize,
        batch: Vec<u32>,
        all_rows: bool,
        round: Round,
        logits: Option<Vec<f32>>,
        hidden: Option<Vec<f32>>,
        verify_start: Instant,
    ) {
        let observes = self.drafter.as_ref().is_some_and(Drafter::observes);
        let share = self.batching.prefix_cache;
        let Some(mut rows) = logits else {
            self.fail(i, local_err("decode step produced no logits"));
            return;
        };
        let drafts = &batch[1..];
        // `Want::Last` returns one row, `Want::All` one per batch token.
        let vocab = if all_rows {
            rows.len() / batch.len()
        } else {
            rows.len()
        };
        let seq = &mut self.running[i];
        let greedy_mode = seq.temperature <= 0.0;
        let (temperature, top_k, top_p) = (seq.temperature, seq.top_k, seq.top_p);
        let shared_rng = &mut self.rng;
        let rng = seq.seeded.as_mut().unwrap_or(shared_rng);

        // Accept the longest agreeing prefix of the drafts.
        let mut accepted = 0usize;
        for (j, &d) in drafts.iter().enumerate() {
            let row = &mut rows[j * vocab..(j + 1) * vocab];
            let agrees = if greedy_mode {
                greedy(row) == d
            } else {
                let p = Dist::warp(row, temperature, top_k, top_p, &mut seq.scratch);
                let q = &round.q_dists[j];
                let ratio = p.prob(d) / q.prob(d);
                if rng.next_f32() < ratio.min(1.0) {
                    true
                } else {
                    seq.forced = Some(match p.residual(q) {
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
            let stats = &self.stats;
            stats.rounds.fetch_add(1, Ordering::Relaxed);
            stats
                .drafted
                .fetch_add(drafts.len() as u64, Ordering::Relaxed);
            stats.accepted.fetch_add(accepted as u64, Ordering::Relaxed);
        }

        // Emit the accepted drafts like sampled tokens. A draft that ends the
        // reply (end of generation, a stop string, the token budget) ends it
        // here and is not cached: plain decoding samples that id and stops
        // without forwarding it, so the cached ids, the K/V rows and the
        // published prefix blocks end before it.
        let mut cached = accepted;
        let mut ended = false;
        for (j, &d) in drafts[..accepted].iter().enumerate() {
            if self.format.is_eog(d) {
                ended = true;
                cached = j;
                break;
            }
            seq.completion += 1;
            let mut items = Vec::new();
            let id_rng = &mut self.id_rng;
            let mut next_call_id = || format!("call_{:016x}", id_rng.next_u64());
            seq.parser
                .push(d, &self.tokenizer, &mut next_call_id, &mut items);
            seq.stopper.push(&mut items, &mut seq.pending);
            if seq.stopper.stopped || seq.completion == seq.budget {
                ended = true;
                cached = j;
                break;
            }
        }

        // Cache the pending token and the accepted drafts that did not end
        // the reply; the state for the next round is the verified row after
        // the last of them. The rows of the rejected drafts leave with their
        // tail blocks.
        let keep = start + 1 + cached;
        seq.blocks.len = start + batch.len();
        let slot = StateSlot(seq.slot.unwrap_or(0));
        if self.model.rewind(slot, keep) != keep
            || self.pools.truncate(&mut seq.blocks, keep) != keep
        {
            self.fail(
                i,
                local_err(format!(
                    "the model cannot roll its cache back to {keep} positions"
                )),
            );
            return;
        }
        let seq = &mut self.running[i];
        seq.ids.push(batch[0]);
        seq.ids.extend_from_slice(&drafts[..cached]);
        let len = seq.blocks.len;
        let ids = share.then_some(seq.ids.as_slice());
        self.pools.release_window(&mut seq.blocks, len, ids);
        // The in-trunk head takes the hidden rows of the tokens kept (the
        // pending token and the accepted drafts; rejected drafts' rows are
        // dropped with them), in plain rounds too, so it never falls behind
        // the trunk.
        if observes && let (Some(drafter), Some(h)) = (&mut self.drafter, hidden.as_deref()) {
            let n = h.len() / batch.len();
            if let Err(e) = drafter.observe(
                &self.model,
                start,
                &batch[..=cached],
                &h[..(cached + 1) * n],
            ) {
                self.fail(i, e);
                return;
            }
        }
        let seq = &mut self.running[i];
        // The verified row after the last kept token seeds the next round:
        // moved to the front of its buffer, not copied into a new one.
        rows.copy_within(cached * vocab..(cached + 1) * vocab, 0);
        rows.truncate(vocab);
        seq.logits = rows;
        seq.hidden = hidden.map(|mut h| {
            let n = h.len() / batch.len();
            h.copy_within(cached * n..(cached + 1) * n, 0);
            h.truncate(n);
            h
        });

        // The measured round includes the cache rollback and the head's
        // upkeep: what speculation really costs per round.
        if self.drafter.is_some() && self.running.len() == 1 {
            let verify_secs = verify_start.elapsed().as_secs_f64();
            self.cost.record_round(
                batch.len(),
                verify_secs,
                round.draft_passes,
                round.draft_secs,
                round.predicted,
                accepted,
            );
            if !drafts.is_empty() || !round.spec_round {
                self.governor.record(
                    !drafts.is_empty(),
                    1 + accepted,
                    round.started.elapsed().as_secs_f64(),
                );
            }
        }
        if ended {
            self.running[i].phase = Phase::Ended { failed: false };
        }
    }
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

#[cfg(test)]
mod tests {
    use super::super::testing::{engine_sized, engine_with};
    use super::*;
    use crate::provider::local::model::tiny::{
        TinyModel, TinyQwen35, TinySpec, write_qwen35, write_spec,
    };

    /// `n` token ids of the tiny vocabulary, none of them an end-of-turn id.
    fn prompt(n: usize, salt: usize) -> Vec<u32> {
        (0..n).map(|i| 9 + ((i + salt) * 3 % 7) as u32).collect()
    }

    /// One request of a test: the stream, and what the engine reported.
    struct Client {
        rx: mpsc::Receiver<Item>,
        metadata: ResponseMetadataHandle,
        items: Vec<Item>,
    }

    impl Client {
        /// Take what the stream holds now; whether it has ended.
        fn poll(&mut self) -> bool {
            loop {
                match self.rx.try_recv() {
                    Ok(item) => self.items.push(item),
                    Err(mpsc::error::TryRecvError::Empty) => return false,
                    Err(mpsc::error::TryRecvError::Disconnected) => return true,
                }
            }
        }

        fn text(&self) -> String {
            self.items
                .iter()
                .filter_map(|i| match i {
                    Ok(MessageContentBlock::Text(t)) => Some(t.as_str()),
                    _ => None,
                })
                .collect()
        }

        /// (prompt tokens, completion tokens) of the finished request.
        fn usage(&self) -> (u32, u32) {
            let u = self.metadata.take().token_usage.expect("usage reported");
            (u.prompt_tokens, u.completion_tokens)
        }
    }

    fn submit(e: &mut Engine, prompt: &[u32], max_tokens: u32, capacity: usize) -> Client {
        let (tx, rx) = mpsc::channel(capacity);
        let metadata = ResponseMetadataHandle::empty();
        e.submit(Job {
            prompt_ids: prompt.to_vec(),
            start_in_thinking: false,
            enable_thinking: false,
            temperature: None,
            max_tokens: Some(max_tokens),
            top_p: None,
            top_k: None,
            seed: None,
            stop: Vec::new(),
            events: tx,
            metadata: metadata.clone(),
        });
        Client {
            rx,
            metadata,
            items: Vec::new(),
        }
    }

    /// Turn until nothing is left, then take every stream.
    fn finish(e: &mut Engine, clients: &mut [Client]) {
        e.run_until_idle();
        for c in clients {
            assert!(c.poll(), "the stream ended");
        }
    }

    /// What one request produces alone on a fresh engine.
    fn alone(model: &TinyModel, ctx: usize, prompt: &[u32], max_tokens: u32) -> (String, u32) {
        let mut e = engine_sized(model, ctx, 0.0, None, 1, None);
        let mut c = [submit(&mut e, prompt, max_tokens, 1024)];
        finish(&mut e, &mut c);
        (c[0].text(), c[0].usage().1)
    }

    /// `n` prompts of 20 tokens whose greedy reply fills `tokens` tokens (the
    /// random weights often end a reply early).
    fn long_runners(model: &TinyModel, n: usize, tokens: u32) -> Vec<Vec<u32>> {
        let found: Vec<Vec<u32>> = (0..40)
            .map(|salt| prompt(20, salt))
            .filter(|p| alone(model, 96, p, tokens).1 == tokens)
            .take(n)
            .collect();
        assert_eq!(found.len(), n, "prompts whose replies fill {tokens} tokens");
        found
    }

    fn llama96() -> TinyModel {
        write_spec(&TinySpec {
            context: 96,
            ..TinySpec::llama()
        })
    }

    #[test]
    fn two_sequences_decode_in_one_step() {
        let model = llama96();
        let prompts = [prompt(20, 0), prompt(26, 3)];
        let mut e = engine_sized(&model, 96, 0.0, None, 2, None);
        let mut clients: Vec<Client> = prompts
            .iter()
            .map(|p| submit(&mut e, p, 12, 1024))
            .collect();
        finish(&mut e, &mut clients);
        for (c, p) in clients.iter().zip(&prompts) {
            let (text, completion) = alone(&model, 96, p, 12);
            assert_eq!(c.text(), text, "greedy output does not depend on the batch");
            assert_eq!(c.usage().1, completion);
        }
        let stats = e.serving.snapshot();
        assert!(
            stats.batched_rows > stats.steps,
            "steps carried several rows: {stats:?}"
        );
        assert_eq!((stats.running, stats.waiting), (0, 0));
    }

    #[test]
    fn prefill_chunks_interleave_with_decode() {
        let model = llama96();
        let short = prompt(5, 1);
        let long = prompt(60, 2);
        // A step of 8 rows: the long prompt prefills over many steps next
        // to the short request's decode rows.
        let mut e = engine_with(&model, 96, 0.0, None, 2, None, Some(8));
        let mut clients = [
            submit(&mut e, &short, 30, 1024),
            submit(&mut e, &long, 6, 1024),
        ];
        let mut mixed = false;
        while !e.idle() {
            e.turn();
            let phases: Vec<_> = e.running.iter().map(|s| (s.phase, s.ids.len())).collect();
            mixed |= matches!(
                phases.as_slice(),
                [(Phase::Decode, _), (Phase::Prefill, n)] if *n > 0 && *n < 60
            );
        }
        assert!(
            mixed,
            "a decoding and a half-prefilled sequence ran together"
        );
        for c in &mut clients {
            assert!(c.poll());
        }
        assert_eq!(clients[0].text(), alone(&model, 96, &short, 30).0);
        assert_eq!(clients[1].text(), alone(&model, 96, &long, 6).0);
        // No step exceeded its row budget.
        let stats = e.serving.snapshot();
        assert!(stats.batched_rows <= 8 * stats.steps, "{stats:?}");
    }

    #[test]
    fn slow_consumer_does_not_stall_others() {
        let model = llama96();
        let runners = long_runners(&model, 2, 40);
        let (slow_prompt, fast_prompt) = (&runners[0], &runners[1]);
        let mut e = engine_sized(&model, 96, 0.0, None, 2, None);
        // The slow stream holds one block and nobody reads it.
        let mut slow = submit(&mut e, slow_prompt, 40, 1);
        let mut fast = submit(&mut e, fast_prompt, 10, 1024);
        let mut fast_done = false;
        for _ in 0..500 {
            e.turn();
            fast_done = fast.poll();
            if fast_done {
                break;
            }
        }
        assert!(fast_done, "the other request finished");
        assert_eq!(fast.text(), alone(&model, 96, fast_prompt, 10).0);
        assert!(
            !e.idle(),
            "the stalled request is still waiting for its consumer"
        );
        assert!(e.running.len() + e.draining.len() > 0);
        // The consumer wakes up: everything it missed is delivered.
        let mut done = false;
        for _ in 0..5000 {
            e.turn();
            done = slow.poll();
            if done {
                break;
            }
        }
        assert!(done);
        assert_eq!(slow.text(), alone(&model, 96, slow_prompt, 40).0);
    }

    #[test]
    fn cancel_releases_blocks() {
        let model = llama96();
        let mut e = engine_sized(&model, 96, 0.0, None, 2, None);
        let runners = long_runners(&model, 2, 40);
        let a = submit(&mut e, &runners[0], 40, 1);
        let mut b = submit(&mut e, &runners[1], 40, 1024);
        // Let both start, then the first consumer goes away.
        for _ in 0..4 {
            e.turn();
        }
        assert_eq!(e.running.len(), 2);
        drop(a);
        e.turn();
        assert_eq!(e.running.len(), 1, "the cancelled sequence left");
        e.run_until_idle();
        assert!(b.poll());
        assert_eq!(e.pools.stats().iter().map(|s| s.used).sum::<usize>(), 0);
        // A request that is cancelled before it was admitted never runs.
        let waiting = submit(&mut e, &prompt(30, 1), 4, 4);
        drop(waiting);
        e.turn();
        assert!(e.idle());
    }

    /// Two sequences need more blocks than the pool has: the younger one is
    /// preempted, and both still produce what they produce alone.
    #[test]
    fn preemption_recovers_through_prefix_store() {
        let model = llama96();
        // A pool of 7 blocks (112 positions) against two sequences that
        // each grow to 80 positions.
        let long = long_runners(&model, 2, 60);
        let mut e = engine_sized(&model, 96, 0.0, None, 2, Some(112));
        let mut clients: Vec<Client> = long.iter().map(|p| submit(&mut e, p, 60, 1024)).collect();
        finish(&mut e, &mut clients);
        for (c, p) in clients.iter().zip(&long) {
            assert_eq!(
                c.text(),
                alone(&model, 96, p, 60).0,
                "preemption is invisible"
            );
            assert_eq!(c.usage(), (20, 60));
        }
        let stats = e.serving.snapshot();
        assert!(stats.preemptions >= 1, "{stats:?}");
        assert_eq!(e.pools.stats().iter().map(|s| s.used).sum::<usize>(), 0);
        assert!(
            stats.prefix_hit_tokens > 0,
            "the preempted sequence resumed from its own blocks: {stats:?}"
        );
    }

    #[test]
    fn requests_queue_when_the_pool_is_full_and_run_in_order() {
        let model = llama96();
        // 7 blocks hold one 60-token prompt (4 blocks) and not a second.
        let prompts = [prompt(60, 0), prompt(60, 2), prompt(10, 5)];
        let mut e = engine_sized(&model, 96, 0.0, None, 3, Some(112));
        let mut clients: Vec<Client> = prompts.iter().map(|p| submit(&mut e, p, 4, 1024)).collect();
        e.turn();
        assert_eq!(e.running.len(), 1, "the second prompt does not fit yet");
        assert_eq!(e.waiting.len(), 2);
        finish(&mut e, &mut clients);
        for (c, p) in clients.iter().zip(&prompts) {
            assert_eq!(c.text(), alone(&model, 96, p, 4).0);
        }
    }

    /// Every architecture serves three concurrent requests exactly as it
    /// serves each alone (the CPU rows are bit-identical).
    #[test]
    fn batched_equals_solo_for_every_architecture() {
        let mut models: Vec<(&str, TinyModel)> = TinySpec::all()
            .into_iter()
            .map(|s| (s.arch, write_spec(&TinySpec { context: 96, ..s })))
            .collect();
        models.push((
            "qwen35",
            write_qwen35(&TinyQwen35 {
                meta: vec![(
                    "context_length".into(),
                    candle_core::quantized::gguf_file::Value::U32(96),
                )],
                ..TinyQwen35::default()
            }),
        ));
        for (name, model) in &models {
            let prompts = [prompt(18, 0), prompt(35, 2), prompt(9, 4)];
            let mut e = engine_sized(model, 96, 0.0, None, 3, None);
            let mut clients: Vec<Client> =
                prompts.iter().map(|p| submit(&mut e, p, 8, 1024)).collect();
            finish(&mut e, &mut clients);
            for (c, p) in clients.iter().zip(&prompts) {
                assert_eq!(c.text(), alone(model, 96, p, 8).0, "{name}");
            }
            let stats = e.serving.snapshot();
            assert!(stats.batched_rows > stats.steps, "{name}: {stats:?}");
        }
    }

    /// A hybrid model serves requests that share a prefix: the first
    /// prefills, the others resume from its blocks and its snapshot.
    #[test]
    fn concurrent_hybrid_requests_share_a_prefix() {
        let model = write_qwen35(&TinyQwen35 {
            meta: vec![(
                "context_length".into(),
                candle_core::quantized::gguf_file::Value::U32(96),
            )],
            ..TinyQwen35::default()
        });
        let mut base = prompt(40, 0);
        base[19] = 3;
        let mut second = base[..35].to_vec();
        second.extend([14, 15, 13, 12, 11]);
        let mut e = engine_sized(&model, 96, 0.0, None, 2, None);
        let mut first = submit(&mut e, &base, 5, 1024);
        // The first request prefills its prompt (publishing its blocks and
        // the snapshot) before the second arrives.
        while e.running.first().is_none_or(|s| s.phase != Phase::Decode) {
            e.turn();
        }
        let mut next = submit(&mut e, &second, 5, 1024);
        e.run_until_idle();
        assert!(first.poll() && next.poll());
        let hit = next.metadata.take().prompt_cache_usage.unwrap();
        assert_eq!(hit.read_tokens, 16, "the snapshot sits at the end of turn");
        assert_eq!(next.text(), alone(&model, 96, &second, 5).0);
    }
}
