//! Deterministic session replay (`service-replay`).
//!
//! [`SessionReplay`] re-materializes a recorded trajectory as the same
//! [`AgentResponseStream`] of [`MessageContentBlock`]s a live provider turn
//! produces, with no network call and no provider dispatch: regression
//! fixtures, offline eval, and single-stepping the exact block sequence a bug
//! reproduced on. An explicit-call service, never a
//! [`CucaPlugin`](crate::plugin::CucaPlugin) ([`crate::services`] owns that
//! contract) — replay *drives* sessions instead of observing a live one, and no
//! hook signature can return a stream.
//!
//! # Entry points
//!
//! Every one is a plain synchronous method call:
//!
//! 1. [`SessionReplay::new`] / [`SessionReplay::with_config`] over an
//!    `Arc<dyn SessionBackend>`, obtained from
//!    [`SessionLogPlugin::backend()`](crate::SessionLogPlugin::backend) or
//!    [`FileBackend::new`](crate::FileBackend::new).
//! 2. [`SessionReplay::load`], [`SessionReplay::load_prefix`], or
//!    [`SessionReplay::load_at_point`] to read and segment a trajectory.
//! 3. [`ReplayTrajectory::stream_turn`] / [`ReplayTrajectory::into_stream`] and
//!    [`ReplayTurn::stream`] / [`ReplayTurn::into_stream`] to materialize a
//!    stream the caller polls exactly like a provider stream.
//! 4. [`ReplayTurn::response`] to rebuild the aggregated
//!    [`UnifiedResponse`] shape for consumers written against
//!    `on_response_complete`'s argument type.
//!
//! # Plugin-tier edge
//!
//! `service-replay = ["plugin-session-log"]`. Replay reads through
//! [`SessionBackend`], the whole read surface it needs; the session-log module
//! stays ignorant that replay exists, so the dependency direction is one-way.
//!
//! # Turn segmentation
//!
//! `SessionLogPlugin::on_response_complete` always appends
//! [`SessionEvent::Latency`] and then [`SessionEvent::TokenUsage`], so that pair
//! is the turn terminator: a turn closes once both have been seen, which for a
//! trajectory this crate recorded is the `TokenUsage` record. Records after the
//! last terminator form a final turn with [`ReplayTurn::completion`] `None` and
//! [`ReplayTurn::is_complete`] `false` (an interrupted generation, or a foreign
//! store that writes no terminator); they are never merged into the previous
//! turn and never dropped.
//!
//! # Determinism contract
//!
//! Order-deterministic, not time-deterministic. Record timestamps are
//! wall-clock ([`SessionRecord::new`]), so nothing here reproduces the original
//! pacing: the replay stream's `poll_next` never sleeps, never returns
//! `Poll::Pending`, and never yields `Err`, because every failure is raised at
//! load time. A materialized stream is therefore guaranteed to run to
//! completion, and two loads of the same trajectory produce identical block
//! sequences. The recorded latency is exposed as data on
//! [`ReplayCompletion::duration_ms`]; nothing waits on it.
//!
//! # Fidelity gaps
//!
//! Two things the recording never held, so replay cannot invent them:
//!
//! - [`MessageContentBlock::ImageBase64`] is never recorded (the session-log
//!   hook maps it to no event), so no image block can ever be replayed.
//! - [`SessionEvent::ToolResult`]'s `stdout`, `stderr`, and `exit_code` have no
//!   representation in [`MessageContentBlock::ToolResult`], so they are absent
//!   from the block stream.
//!
//! Nothing is synthesized to paper over either gap. Callers needing the
//! diagnostics read the raw records through [`SessionBackend::replay`].
//!
//! # Bounds
//!
//! [`ReplayConfig::max_records`] and [`ReplayConfig::max_turn_blocks`]
//! **refuse** with [`PluginError::Validation`] rather than truncate: a silently
//! shortened trajectory is a wrong fixture and a wrong eval. The *pre-read*
//! bound belongs to the backend
//! ([`InMemoryBackend::max_records`](crate::InMemoryBackend::max_records), or
//! disk for [`FileBackend`](crate::FileBackend)), because
//! [`SessionBackend::replay`] returns a whole `Vec<SessionRecord>` and has no
//! ranged read; `load*` consumes that `Vec` by value and drops it, so the
//! steady state is one trajectory rather than a trajectory plus its source
//! records. [`ReplayUsage`] is the O(1) gauge, computed once at load, and
//! [`ReplayConfig::warn_fraction`] drives [`ReplayUsage::near_cap`].
//! [`SessionReplay`] itself holds an `Arc` and a `Copy` config: no collection,
//! no per-session bookkeeping, no growth with traffic.
//!
//! # Scope
//!
//! Replayed blocks do **not** pass the plugin pipeline: `on_stream_chunk` and
//! `on_response_complete` never see them. `CucaClient::generate_stream` always
//! continues to provider dispatch, so feeding a caller-supplied stream through
//! the pipeline would need a new core `CucaClient` entry point, which is
//! deliberately not added here. Writing to a trajectory likewise stays with
//! `SessionStorePlugin`; this module only reads.

use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use futures_core::Stream;

use crate::error::{CucaError, PluginError};
use crate::plugins::session_log::SessionBackend;
use crate::request::{AgentResponseStream, UnifiedResponse};
use crate::session::{SessionEvent, SessionRecord};
use crate::types::{MessageContentBlock, ProviderEndpoint, UnifiedMessage};

/// Validated bounds for one [`SessionReplay`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReplayConfig {
    /// Maximum records a single load may retain; must be non-zero.
    pub max_records: usize,
    /// Maximum content blocks a single turn may retain; must be non-zero.
    pub max_turn_blocks: usize,
    /// Retained-record fraction of `max_records` at which
    /// [`ReplayUsage::near_cap`] flips; `None` disables the flag. When set,
    /// must lie in `(0.0, 1.0]`.
    pub warn_fraction: Option<f32>,
}

impl ReplayConfig {
    /// Cap used by [`Self::default`]; mirrors
    /// [`InMemoryBackend::DEFAULT_MAX_RECORDS`](crate::InMemoryBackend::DEFAULT_MAX_RECORDS).
    pub const DEFAULT_MAX_RECORDS: usize = 65_536;
    /// Per-turn block cap used by [`Self::default`].
    pub const DEFAULT_MAX_TURN_BLOCKS: usize = 4_096;

    /// Rejecting constructor.
    ///
    /// # Errors
    ///
    /// [`PluginError::Validation`] when either cap is zero or `warn_fraction`
    /// is outside `(0.0, 1.0]`.
    pub fn new(
        max_records: usize,
        max_turn_blocks: usize,
        warn_fraction: Option<f32>,
    ) -> Result<Self, PluginError> {
        let config = Self {
            max_records,
            max_turn_blocks,
            warn_fraction,
        };
        config.validate()?;
        Ok(config)
    }

    /// Re-check an assembled config (public fields allow struct literals).
    ///
    /// # Errors
    ///
    /// Same conditions as [`Self::new`].
    pub fn validate(&self) -> Result<(), PluginError> {
        if self.max_records == 0 {
            return Err(PluginError::Validation {
                schema: "max_records".to_string(),
                message: "max_records must be non-zero".to_string(),
            });
        }
        if self.max_turn_blocks == 0 {
            return Err(PluginError::Validation {
                schema: "max_turn_blocks".to_string(),
                message: "max_turn_blocks must be non-zero".to_string(),
            });
        }
        if let Some(fraction) = self.warn_fraction
            && !(fraction.is_finite() && fraction > 0.0 && fraction <= 1.0)
        {
            return Err(PluginError::Validation {
                schema: "warn_fraction".to_string(),
                message: format!("warn_fraction must lie in (0.0, 1.0], got {fraction}"),
            });
        }
        Ok(())
    }
}

impl Default for ReplayConfig {
    fn default() -> Self {
        Self {
            max_records: Self::DEFAULT_MAX_RECORDS,
            max_turn_blocks: Self::DEFAULT_MAX_TURN_BLOCKS,
            warn_fraction: Some(0.9),
        }
    }
}

/// Retained-size gauge for one [`ReplayTrajectory`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReplayUsage {
    /// Records retained by this load.
    pub records: usize,
    /// The configured `max_records` the load was checked against.
    pub max_records: usize,
    /// Content blocks retained across all turns.
    pub blocks: usize,
    /// Turns segmented.
    pub turns: usize,
    /// `records >= warn_fraction * max_records`; always `false` when
    /// `warn_fraction` is `None`.
    pub near_cap: bool,
}

/// Terminal accounting of a recorded turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReplayCompletion {
    /// From [`SessionEvent::Latency`].
    pub duration_ms: u64,
    /// From [`SessionEvent::TokenUsage`].
    pub prompt_tokens: u32,
    /// From [`SessionEvent::TokenUsage`].
    pub completion_tokens: u32,
}

/// A recorded event that is neither prompt, message, block, nor accounting.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReplayNote {
    /// From [`SessionEvent::ModelSwap`].
    ModelSwap {
        /// Model swapped away from.
        from: String,
        /// Model swapped to.
        to: String,
        /// Why the swap fired.
        reason: String,
    },
    /// From [`SessionEvent::Fork`].
    Fork {
        /// `point_id()` the branch was taken at.
        from_point: String,
        /// The new session id.
        to_session: String,
    },
}

/// Explicit-call replay capability over a [`SessionBackend`].
///
/// Not a [`CucaPlugin`](crate::plugin::CucaPlugin): see this module's header.
/// Holds one `Arc` and one `Copy` config, so it does not grow with traffic.
pub struct SessionReplay {
    backend: Arc<dyn SessionBackend>,
    config: ReplayConfig,
}

impl SessionReplay {
    /// Replay through `backend` with [`ReplayConfig::default`].
    pub fn new(backend: Arc<dyn SessionBackend>) -> Self {
        Self {
            backend,
            config: ReplayConfig::default(),
        }
    }

    /// Replay through `backend` with caller-supplied bounds.
    ///
    /// # Errors
    ///
    /// [`PluginError::Validation`] from [`ReplayConfig::validate`].
    pub fn with_config(
        backend: Arc<dyn SessionBackend>,
        config: ReplayConfig,
    ) -> Result<Self, PluginError> {
        config.validate()?;
        Ok(Self { backend, config })
    }

    /// The configured bounds.
    pub fn config(&self) -> &ReplayConfig {
        &self.config
    }

    /// Load a whole recorded session in append order.
    ///
    /// # Errors
    ///
    /// Propagates [`SessionBackend::replay`]'s errors, and returns
    /// [`PluginError::Validation`] when the trajectory exceeds a configured
    /// cap or its sequences are not strictly increasing.
    pub fn load(&self, session_id: &str) -> Result<ReplayTrajectory, PluginError> {
        let records = self.backend.replay(session_id)?;
        self.assemble(session_id.to_string(), records)
    }

    /// Fork-point load: retain only records with `sequence <= upto_sequence`.
    ///
    /// # Errors
    ///
    /// As [`Self::load`], plus [`PluginError::Validation`] when the session
    /// has no record at or below `upto_sequence`.
    pub fn load_prefix(
        &self,
        session_id: &str,
        upto_sequence: u64,
    ) -> Result<ReplayTrajectory, PluginError> {
        // Consumed by value: the retained records are moved into the
        // trajectory, and the source `Vec` is dropped at the end of this call.
        let retained: Vec<SessionRecord> = self
            .backend
            .replay(session_id)?
            .into_iter()
            .filter(|record| record.sequence <= upto_sequence)
            .collect();
        if retained.is_empty() {
            return Err(PluginError::Validation {
                schema: "point_id".to_string(),
                message: format!(
                    "no record at or below sequence {upto_sequence} in session `{session_id}`"
                ),
            });
        }
        self.assemble(session_id.to_string(), retained)
    }

    /// Fork-point load addressed by [`SessionRecord::point_id`]
    /// (`"{session_id}:{sequence}"`), the same string
    /// [`SessionStorePlugin::fork_session`](crate::plugin::SessionStorePlugin::fork_session)
    /// takes.
    ///
    /// # Errors
    ///
    /// As [`Self::load_prefix`], plus [`PluginError::Validation`] when
    /// `point_id` is malformed or names no recorded position.
    pub fn load_at_point(&self, point_id: &str) -> Result<ReplayTrajectory, PluginError> {
        // A session id may itself contain `:` (fork ids do), so the sequence is
        // the tail after the LAST separator.
        let malformed = |detail: &str| PluginError::Validation {
            schema: "point_id".to_string(),
            message: format!("malformed point_id `{point_id}`: {detail}"),
        };
        let (session_id, sequence) = point_id
            .rsplit_once(':')
            .ok_or_else(|| malformed("expected the form `{session_id}:{sequence}`"))?;
        if session_id.is_empty() {
            return Err(malformed("the session id before the last `:` is empty"));
        }
        let sequence: u64 = sequence
            .parse()
            .map_err(|_| malformed("the sequence after the last `:` is not a u64"))?;
        self.load_prefix(session_id, sequence)
    }

    /// Validate the retained records and segment them into turns.
    ///
    /// Order: cap first (a refusal must not depend on how far segmentation
    /// got), then strict ordering, then segmentation. The `Vec` is consumed by
    /// value, so every `String`, block, and `serde_json::Value` is moved.
    fn assemble(
        &self,
        session_id: String,
        records: Vec<SessionRecord>,
    ) -> Result<ReplayTrajectory, PluginError> {
        if records.len() > self.config.max_records {
            return Err(PluginError::Validation {
                schema: "max_records".to_string(),
                message: format!(
                    "session `{session_id}` retains {} records, over the configured max_records \
                     of {}; replay refuses rather than truncating a trajectory, since a shortened \
                     one is a wrong fixture: raise the bound with ReplayConfig::new",
                    records.len(),
                    self.config.max_records
                ),
            });
        }
        for pair in records.windows(2) {
            if pair[1].sequence <= pair[0].sequence {
                return Err(PluginError::Validation {
                    schema: "sequence".to_string(),
                    message: format!(
                        "session `{session_id}` records are not strictly increasing: sequence {} \
                         follows {}; replay refuses an ambiguous order rather than guessing it",
                        pair[1].sequence, pair[0].sequence
                    ),
                });
            }
        }

        let records_retained = records.len();
        let mut turns: Vec<ReplayTurn> = Vec::new();
        let mut current: Option<TurnBuilder> = None;

        for record in records {
            let SessionRecord {
                sequence, event, ..
            } = record;
            let mut turn = current.take().unwrap_or_else(|| TurnBuilder::new(sequence));
            turn.last_sequence = sequence;

            // Exhaustive on purpose (no `_ =>` arm): a new `SessionEvent`
            // variant must fail to compile here rather than drop out of replay.
            match event {
                SessionEvent::SystemPrompt { text } => turn.system_prompts.push(text),
                SessionEvent::Message { role, content } => turn.messages.push(UnifiedMessage {
                    role,
                    content,
                    name: None,
                    tool_call_id: None,
                }),
                SessionEvent::Reasoning {
                    reasoning,
                    signature,
                } => turn.blocks.push(MessageContentBlock::Thinking {
                    reasoning,
                    signature,
                }),
                SessionEvent::Output { text } => turn.blocks.push(MessageContentBlock::Text(text)),
                SessionEvent::ToolCall {
                    id,
                    name,
                    arguments,
                } => turn.blocks.push(MessageContentBlock::ToolCall {
                    id,
                    name,
                    arguments,
                }),
                // `stdout`/`stderr`/`exit_code` have no home in the unified
                // block; the fidelity gap is documented, not synthesized over.
                SessionEvent::ToolResult {
                    tool_call_id,
                    output,
                    stdout: _,
                    stderr: _,
                    exit_code: _,
                } => turn.blocks.push(MessageContentBlock::ToolResult {
                    tool_call_id,
                    output,
                }),
                SessionEvent::Latency { duration_ms } => {
                    turn.pending_duration_ms = Some(duration_ms);
                }
                SessionEvent::TokenUsage {
                    prompt_tokens,
                    completion_tokens,
                } => turn.pending_tokens = Some((prompt_tokens, completion_tokens)),
                SessionEvent::ModelSwap { from, to, reason } => {
                    turn.notes.push(ReplayNote::ModelSwap { from, to, reason });
                }
                SessionEvent::Fork {
                    from_point,
                    to_session,
                } => turn.notes.push(ReplayNote::Fork {
                    from_point,
                    to_session,
                }),
            }

            // The terminator is the record completing the Latency+TokenUsage
            // pair; for a trajectory this crate recorded that is the
            // `TokenUsage` record, since the hook appends them in that order.
            if turn.is_terminated() {
                turns.push(turn.finish(&session_id, self.config.max_turn_blocks)?);
            } else {
                current = Some(turn);
            }
        }
        // Trailing records without a terminator: their own incomplete turn.
        if let Some(turn) = current {
            turns.push(turn.finish(&session_id, self.config.max_turn_blocks)?);
        }

        let blocks = turns.iter().map(|turn| turn.blocks.len()).sum();
        let usage = ReplayUsage {
            records: records_retained,
            max_records: self.config.max_records,
            blocks,
            turns: turns.len(),
            near_cap: self.config.warn_fraction.is_some_and(|fraction| {
                records_retained as f32 >= fraction * self.config.max_records as f32
            }),
        };
        Ok(ReplayTrajectory {
            session_id,
            turns,
            usage,
        })
    }
}

/// Accumulator for one turn's records, before its bounds are checked.
struct TurnBuilder {
    system_prompts: Vec<String>,
    messages: Vec<UnifiedMessage>,
    blocks: Vec<MessageContentBlock>,
    notes: Vec<ReplayNote>,
    pending_duration_ms: Option<u64>,
    pending_tokens: Option<(u32, u32)>,
    first_sequence: u64,
    last_sequence: u64,
}

impl TurnBuilder {
    fn new(first_sequence: u64) -> Self {
        Self {
            system_prompts: Vec::new(),
            messages: Vec::new(),
            blocks: Vec::new(),
            notes: Vec::new(),
            pending_duration_ms: None,
            pending_tokens: None,
            first_sequence,
            last_sequence: first_sequence,
        }
    }

    /// Whether both halves of the terminator pair have arrived.
    fn is_terminated(&self) -> bool {
        self.pending_duration_ms.is_some() && self.pending_tokens.is_some()
    }

    /// Check the per-turn block bound and seal the turn.
    fn finish(self, session_id: &str, max_turn_blocks: usize) -> Result<ReplayTurn, PluginError> {
        if self.blocks.len() > max_turn_blocks {
            return Err(PluginError::Validation {
                schema: "max_turn_blocks".to_string(),
                message: format!(
                    "the turn covering sequences {}..={} of session `{session_id}` holds {} \
                     blocks, over the configured max_turn_blocks of {max_turn_blocks}; replay \
                     refuses rather than truncating a turn: raise the bound with \
                     ReplayConfig::new",
                    self.first_sequence,
                    self.last_sequence,
                    self.blocks.len()
                ),
            });
        }
        let completion = match (self.pending_duration_ms, self.pending_tokens) {
            (Some(duration_ms), Some((prompt_tokens, completion_tokens))) => {
                Some(ReplayCompletion {
                    duration_ms,
                    prompt_tokens,
                    completion_tokens,
                })
            }
            _ => None,
        };
        Ok(ReplayTurn {
            system_prompts: self.system_prompts,
            messages: self.messages,
            blocks: self.blocks,
            completion,
            notes: self.notes,
            first_sequence: self.first_sequence,
            last_sequence: self.last_sequence,
        })
    }
}

/// One loaded trajectory, segmented into turns.
#[derive(Debug)]
pub struct ReplayTrajectory {
    session_id: String,
    turns: Vec<ReplayTurn>,
    usage: ReplayUsage,
}

impl ReplayTrajectory {
    /// The session this trajectory was loaded from.
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// Number of segmented turns.
    pub fn len(&self) -> usize {
        self.turns.len()
    }

    /// Whether the trajectory holds no turns.
    pub fn is_empty(&self) -> bool {
        self.turns.is_empty()
    }

    /// The retained-size gauge.
    pub fn usage(&self) -> ReplayUsage {
        self.usage
    }

    /// The segmented turns, in record order.
    pub fn turns(&self) -> &[ReplayTurn] {
        &self.turns
    }

    /// One turn by index.
    pub fn turn(&self, index: usize) -> Option<&ReplayTurn> {
        self.turns.get(index)
    }

    /// Stream one turn's blocks; clones that turn's blocks so the trajectory
    /// stays replayable (which regression fixtures need, so the copy is a
    /// chosen cost). Use [`ReplayTurn::into_stream`] to move them instead.
    ///
    /// # Errors
    ///
    /// [`PluginError::Validation`] when `index` names no turn (an empty
    /// trajectory therefore refuses rather than yielding zero blocks).
    pub fn stream_turn(&self, index: usize) -> Result<AgentResponseStream, PluginError> {
        let turn = self
            .turns
            .get(index)
            .ok_or_else(|| PluginError::Validation {
                schema: "turn_index".to_string(),
                message: format!(
                    "turn index {index} is out of range: session `{}` replayed {} turns",
                    self.session_id,
                    self.turns.len()
                ),
            })?;
        Ok(turn.stream())
    }

    /// Stream every turn's blocks concatenated in record order, moving the
    /// blocks out (no clone).
    ///
    /// # Errors
    ///
    /// [`PluginError::Validation`] when the trajectory holds no turns.
    pub fn into_stream(self) -> Result<AgentResponseStream, PluginError> {
        if self.turns.is_empty() {
            return Err(PluginError::Validation {
                schema: "session_id".to_string(),
                message: format!(
                    "session `{}` replayed no turns; replay refuses rather than handing back a \
                     zero-block stream",
                    self.session_id
                ),
            });
        }
        let blocks: Vec<MessageContentBlock> = self
            .turns
            .into_iter()
            .flat_map(|turn| turn.blocks)
            .collect();
        Ok(Box::pin(ReplayStream {
            blocks: blocks.into_iter(),
        }))
    }
}

/// One recorded generation: the inputs recorded before it, the blocks it
/// produced, its terminal accounting, and any non-content annotations.
#[derive(Debug)]
pub struct ReplayTurn {
    system_prompts: Vec<String>,
    messages: Vec<UnifiedMessage>,
    blocks: Vec<MessageContentBlock>,
    completion: Option<ReplayCompletion>,
    notes: Vec<ReplayNote>,
    first_sequence: u64,
    last_sequence: u64,
}

impl ReplayTurn {
    /// System instructions recorded before this turn.
    pub fn system_prompts(&self) -> &[String] {
        &self.system_prompts
    }

    /// Conversation messages recorded before this turn.
    pub fn messages(&self) -> &[UnifiedMessage] {
        &self.messages
    }

    /// The blocks this turn produced, in record order.
    ///
    /// Two documented fidelity gaps apply: no
    /// [`MessageContentBlock::ImageBase64`] is ever recorded, so none is ever
    /// replayed, and [`SessionEvent::ToolResult`]'s `stdout`/`stderr`/
    /// `exit_code` have no representation in
    /// [`MessageContentBlock::ToolResult`], so they are absent here. Callers
    /// needing them read raw records via [`SessionBackend::replay`].
    pub fn blocks(&self) -> &[MessageContentBlock] {
        &self.blocks
    }

    /// Terminal accounting; `None` for a turn recorded without a
    /// `Latency`/`TokenUsage` terminator (an interrupted generation).
    pub fn completion(&self) -> Option<&ReplayCompletion> {
        self.completion.as_ref()
    }

    /// Non-content annotations (model swaps, fork audit records).
    pub fn notes(&self) -> &[ReplayNote] {
        &self.notes
    }

    /// Inclusive record-sequence range this turn covers.
    pub fn sequence_range(&self) -> (u64, u64) {
        (self.first_sequence, self.last_sequence)
    }

    /// Whether the turn carries its terminal accounting.
    pub fn is_complete(&self) -> bool {
        self.completion.is_some()
    }

    /// Rebuild the aggregated response shape from the recorded blocks and
    /// accounting. `model`/`provider` are caller-supplied: the trajectory does
    /// not record them (no [`SessionEvent`] carries them except
    /// [`SessionEvent::ModelSwap`], which describes a swap rather than the
    /// serving model).
    ///
    /// `finish_reason` is always `None` and `prompt_cache_usage` always `None`:
    /// neither is recorded. A turn with no terminator
    /// ([`Self::is_complete`] `false`) reports zero tokens and zero duration.
    pub fn response(
        &self,
        model: impl Into<String>,
        provider: ProviderEndpoint,
    ) -> UnifiedResponse {
        UnifiedResponse {
            model: model.into(),
            provider,
            duration_secs: self
                .completion
                .map_or(0.0, |c| c.duration_ms as f64 / 1000.0),
            prompt_tokens: self.completion.map_or(0, |c| c.prompt_tokens),
            completion_tokens: self.completion.map_or(0, |c| c.completion_tokens),
            finish_reason: None,
            content: self.blocks.clone(),
            prompt_cache_usage: None,
        }
    }

    /// Stream this turn's blocks (clones them, so the turn stays replayable).
    pub fn stream(&self) -> AgentResponseStream {
        Box::pin(ReplayStream {
            blocks: self.blocks.clone().into_iter(),
        })
    }

    /// Stream this turn's blocks, moving them out.
    pub fn into_stream(self) -> AgentResponseStream {
        Box::pin(ReplayStream {
            blocks: self.blocks.into_iter(),
        })
    }
}

/// The replay stream: a finite, already-resolved block sequence.
///
/// `poll_next` never returns `Pending`, never sleeps, and never yields `Err`
/// (every failure is raised at load time), so it needs no runtime and no waker.
/// The iterator is bounded by the turn or trajectory it was built from, which
/// the load-time caps already checked, and is drained monotonically.
struct ReplayStream {
    blocks: std::vec::IntoIter<MessageContentBlock>,
}

impl Stream for ReplayStream {
    type Item = Result<MessageContentBlock, CucaError>;

    fn poll_next(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Poll::Ready(self.blocks.next().map(Ok))
    }
}

#[cfg(all(test, feature = "service-replay"))]
mod tests {
    use super::*;
    use crate::plugins::session_log::InMemoryBackend;
    use crate::types::MessageRole;

    const SESSION: &str = "s";

    /// A backend holding `events` appended to [`SESSION`] at sequences `0..n`.
    fn backend_with(events: Vec<SessionEvent>) -> Arc<dyn SessionBackend> {
        let records = events
            .into_iter()
            .enumerate()
            .map(|(i, event)| SessionRecord::at(SESSION, i as u64, 1_000 + i as u64, event))
            .collect();
        backend_with_records(records)
    }

    /// A backend holding `records` verbatim, in the order given.
    fn backend_with_records(records: Vec<SessionRecord>) -> Arc<dyn SessionBackend> {
        let backend = InMemoryBackend::new();
        for record in &records {
            backend.append(record).expect("append must succeed");
        }
        Arc::new(backend)
    }

    fn output(text: &str) -> SessionEvent {
        SessionEvent::Output {
            text: text.to_string(),
        }
    }

    /// The `Latency` + `TokenUsage` pair `on_response_complete` appends.
    fn terminator(duration_ms: u64, prompt: u32, completion: u32) -> Vec<SessionEvent> {
        vec![
            SessionEvent::Latency { duration_ms },
            SessionEvent::TokenUsage {
                prompt_tokens: prompt,
                completion_tokens: completion,
            },
        ]
    }

    fn text(t: &str) -> MessageContentBlock {
        MessageContentBlock::Text(t.to_string())
    }

    /// The one turn of a single-generation fixture, plus its trajectory.
    fn one_generation() -> ReplayTrajectory {
        let mut events = vec![
            SessionEvent::SystemPrompt {
                text: "be concise".to_string(),
            },
            SessionEvent::Message {
                role: MessageRole::User,
                content: vec![text("hi")],
            },
            output("a"),
        ];
        events.extend(terminator(1_500, 10, 5));
        SessionReplay::new(backend_with(events))
            .load(SESSION)
            .expect("load must succeed")
    }

    /// Drain a replay stream with no runtime at all.
    ///
    /// `ReplayStream::poll_next` is documented never to return `Pending`, so a
    /// no-op waker is sufficient; the `Pending` arm below is the assertion
    /// enforcing that half of the determinism contract.
    fn drain_now(mut stream: AgentResponseStream) -> Vec<MessageContentBlock> {
        let mut cx = Context::from_waker(std::task::Waker::noop());
        let mut blocks = Vec::new();
        loop {
            match stream.as_mut().poll_next(&mut cx) {
                Poll::Ready(Some(item)) => {
                    blocks.push(item.expect("a replayed item is never an error"));
                }
                Poll::Ready(None) => return blocks,
                Poll::Pending => panic!("a replay stream must never return Pending"),
            }
        }
    }

    /// The schema of a `Validation` error, or a panic naming what came instead.
    fn validation_schema(err: &PluginError) -> &str {
        match err {
            PluginError::Validation { schema, .. } => schema,
            other => panic!("expected PluginError::Validation, got {other:?}"),
        }
    }

    /// The error of a refused call whose `Ok` type is not `Debug`, so
    /// `expect_err` cannot be used on it (`SessionReplay` holds an `Arc<dyn
    /// SessionBackend>`; `AgentResponseStream` is a boxed trait object).
    fn refusal<T>(result: Result<T, PluginError>, what: &str) -> PluginError {
        match result {
            Err(err) => err,
            Ok(_) => panic!("{what} must be refused"),
        }
    }

    #[test]
    fn config_rejects_zero_max_records() {
        let err = ReplayConfig::new(0, 16, None).expect_err("zero max_records must be refused");
        assert_eq!(validation_schema(&err), "max_records");
    }

    #[test]
    fn config_rejects_zero_max_turn_blocks() {
        let err = ReplayConfig::new(16, 0, None).expect_err("zero max_turn_blocks must be refused");
        assert_eq!(validation_schema(&err), "max_turn_blocks");
    }

    #[test]
    fn config_rejects_warn_fraction_outside_unit_range() {
        for fraction in [0.0, 1.5, f32::NAN, -0.5, f32::INFINITY] {
            let err = ReplayConfig::new(16, 16, Some(fraction))
                .expect_err("warn_fraction outside (0.0, 1.0] must be refused");
            assert_eq!(validation_schema(&err), "warn_fraction", "for {fraction}");
        }
        // The interval is closed at 1.0: a full-cap warning is legal.
        assert!(ReplayConfig::new(16, 16, Some(1.0)).is_ok());
        assert!(ReplayConfig::new(16, 16, None).is_ok());
    }

    #[test]
    fn with_config_propagates_config_validation() {
        let backend: Arc<dyn SessionBackend> = Arc::new(InMemoryBackend::new());
        let err = refusal(
            SessionReplay::with_config(
                backend,
                ReplayConfig {
                    max_records: 0,
                    ..ReplayConfig::default()
                },
            ),
            "an invalid config",
        );
        assert_eq!(validation_schema(&err), "max_records");
    }

    #[test]
    fn single_generation_segments_into_one_turn_with_prompt_message_and_completion() {
        let trajectory = one_generation();
        assert_eq!(trajectory.session_id(), SESSION);
        assert_eq!(trajectory.len(), 1);
        assert!(!trajectory.is_empty());

        let turn = trajectory.turn(0).expect("the single turn must be there");
        assert_eq!(turn.system_prompts(), ["be concise".to_string()]);
        assert_eq!(turn.messages().len(), 1);
        assert_eq!(turn.messages()[0].role, MessageRole::User);
        assert_eq!(turn.messages()[0].content, vec![text("hi")]);
        assert_eq!(turn.blocks(), [text("a")]);
        assert_eq!(
            turn.completion(),
            Some(&ReplayCompletion {
                duration_ms: 1_500,
                prompt_tokens: 10,
                completion_tokens: 5,
            })
        );
        assert!(turn.is_complete());
        assert_eq!(turn.sequence_range(), (0, 4));
        assert!(turn.notes().is_empty());
    }

    #[test]
    fn two_generations_segment_at_the_latency_token_usage_terminator() {
        let mut events = vec![output("a")];
        events.extend(terminator(100, 1, 2));
        events.push(output("b"));
        events.extend(terminator(200, 3, 4));

        let trajectory = SessionReplay::new(backend_with(events))
            .load(SESSION)
            .expect("load must succeed");
        assert_eq!(trajectory.len(), 2);
        assert_eq!(trajectory.turns()[0].blocks(), [text("a")]);
        assert_eq!(trajectory.turns()[0].sequence_range(), (0, 2));
        assert_eq!(trajectory.turns()[1].blocks(), [text("b")]);
        assert_eq!(trajectory.turns()[1].sequence_range(), (3, 5));
        assert!(trajectory.turns().iter().all(ReplayTurn::is_complete));
        assert_eq!(
            trajectory.turns()[1].completion(),
            Some(&ReplayCompletion {
                duration_ms: 200,
                prompt_tokens: 3,
                completion_tokens: 4,
            })
        );
    }

    #[test]
    fn trailing_records_without_terminator_form_an_incomplete_turn() {
        let mut events = vec![output("a")];
        events.extend(terminator(100, 1, 2));
        events.push(output("b"));

        let trajectory = SessionReplay::new(backend_with(events))
            .load(SESSION)
            .expect("load must succeed");
        assert_eq!(trajectory.len(), 2, "trailing records are their own turn");
        assert!(trajectory.turns()[0].is_complete());

        let trailing = &trajectory.turns()[1];
        assert_eq!(trailing.blocks(), [text("b")]);
        assert_eq!(trailing.completion(), None);
        assert!(!trailing.is_complete());
        assert_eq!(trailing.sequence_range(), (3, 3));
    }

    #[test]
    fn content_events_invert_on_stream_chunk_mapping() {
        let arguments = serde_json::json!({ "q": "x" });
        let events = vec![
            SessionEvent::Reasoning {
                reasoning: "think".to_string(),
                signature: Some("sig".to_string()),
            },
            output("said"),
            SessionEvent::ToolCall {
                id: "call-1".to_string(),
                name: "search".to_string(),
                arguments: arguments.clone(),
            },
            SessionEvent::ToolResult {
                tool_call_id: "call-1".to_string(),
                output: "found".to_string(),
                stdout: None,
                stderr: None,
                exit_code: None,
            },
        ];

        let trajectory = SessionReplay::new(backend_with(events))
            .load(SESSION)
            .expect("load must succeed");
        assert_eq!(trajectory.len(), 1);
        assert_eq!(
            trajectory.turns()[0].blocks(),
            [
                MessageContentBlock::Thinking {
                    reasoning: "think".to_string(),
                    signature: Some("sig".to_string()),
                },
                text("said"),
                MessageContentBlock::ToolCall {
                    id: "call-1".to_string(),
                    name: "search".to_string(),
                    arguments,
                },
                MessageContentBlock::ToolResult {
                    tool_call_id: "call-1".to_string(),
                    output: "found".to_string(),
                },
            ]
        );
    }

    #[test]
    fn tool_result_diagnostics_are_absent_from_the_block_but_the_turn_still_streams() {
        let events = vec![SessionEvent::ToolResult {
            tool_call_id: "call-1".to_string(),
            output: "found".to_string(),
            stdout: Some("out".to_string()),
            stderr: Some("err".to_string()),
            exit_code: Some(0),
        }];

        let trajectory = SessionReplay::new(backend_with(events))
            .load(SESSION)
            .expect("load must succeed");
        assert_eq!(
            trajectory.turns()[0].blocks(),
            [MessageContentBlock::ToolResult {
                tool_call_id: "call-1".to_string(),
                output: "found".to_string(),
            }],
            "the block carries only the call id and the output"
        );
        assert_eq!(
            drain_now(trajectory.stream_turn(0).expect("turn 0 must stream")),
            vec![MessageContentBlock::ToolResult {
                tool_call_id: "call-1".to_string(),
                output: "found".to_string(),
            }],
            "the dropped diagnostics must not stop the turn from streaming"
        );
    }

    #[test]
    fn model_swap_and_fork_records_land_in_notes_and_never_in_blocks() {
        let mut events = vec![
            output("a"),
            SessionEvent::ModelSwap {
                from: "fast".to_string(),
                to: "slow".to_string(),
                reason: "latency_threshold".to_string(),
            },
            SessionEvent::Fork {
                from_point: "s:0".to_string(),
                to_session: "branch".to_string(),
            },
        ];
        events.extend(terminator(10, 1, 1));

        let trajectory = SessionReplay::new(backend_with(events))
            .load(SESSION)
            .expect("load must succeed");
        let turn = &trajectory.turns()[0];
        assert_eq!(turn.blocks(), [text("a")], "notes are never blocks");
        assert_eq!(
            turn.notes(),
            [
                ReplayNote::ModelSwap {
                    from: "fast".to_string(),
                    to: "slow".to_string(),
                    reason: "latency_threshold".to_string(),
                },
                ReplayNote::Fork {
                    from_point: "s:0".to_string(),
                    to_session: "branch".to_string(),
                },
            ]
        );
        assert_eq!(trajectory.usage().blocks, 1);
    }

    #[test]
    fn trajectory_over_max_records_is_refused_not_truncated() {
        let backend = backend_with(vec![output("a"), output("b"), output("c")]);
        let replay = SessionReplay::with_config(
            backend,
            ReplayConfig::new(2, 16, None).expect("config must build"),
        )
        .expect("replay must build");

        let err = replay
            .load(SESSION)
            .expect_err("an over-cap trajectory must be refused");
        assert_eq!(validation_schema(&err), "max_records");
        let message = err.to_string();
        assert!(
            message.contains("retains 3 records") && message.contains("max_records of 2"),
            "the refusal must name the count and the cap: {message}"
        );
        assert!(
            message.contains(SESSION) && message.contains("ReplayConfig::new"),
            "the refusal must name the session and the remedy: {message}"
        );
    }

    #[test]
    fn turn_over_max_turn_blocks_is_refused() {
        let backend = backend_with(vec![output("a"), output("b")]);
        let replay = SessionReplay::with_config(
            backend,
            ReplayConfig::new(16, 1, None).expect("config must build"),
        )
        .expect("replay must build");

        let err = replay
            .load(SESSION)
            .expect_err("an over-cap turn must be refused");
        assert_eq!(validation_schema(&err), "max_turn_blocks");
        assert!(
            err.to_string().contains("sequences 0..=1"),
            "the refusal must name the turn's sequence range: {err}"
        );
    }

    #[test]
    fn duplicate_sequence_is_refused() {
        let backend = backend_with_records(vec![
            SessionRecord::at(SESSION, 0, 1, output("a")),
            SessionRecord::at(SESSION, 1, 2, output("b")),
            SessionRecord::at(SESSION, 1, 3, output("c")),
        ]);
        let err = SessionReplay::new(backend)
            .load(SESSION)
            .expect_err("a duplicate sequence must be refused");
        assert_eq!(validation_schema(&err), "sequence");
    }

    #[test]
    fn regressing_sequence_is_refused() {
        let backend = backend_with_records(vec![
            SessionRecord::at(SESSION, 0, 1, output("a")),
            SessionRecord::at(SESSION, 2, 2, output("b")),
            SessionRecord::at(SESSION, 1, 3, output("c")),
        ]);
        let err = SessionReplay::new(backend)
            .load(SESSION)
            .expect_err("a regressing sequence must be refused");
        assert_eq!(validation_schema(&err), "sequence");
    }

    #[test]
    fn sequence_gap_is_tolerated() {
        let backend = backend_with_records(vec![
            SessionRecord::at(SESSION, 0, 1, output("a")),
            SessionRecord::at(SESSION, 5, 2, SessionEvent::Latency { duration_ms: 7 }),
            SessionRecord::at(
                SESSION,
                9,
                3,
                SessionEvent::TokenUsage {
                    prompt_tokens: 1,
                    completion_tokens: 2,
                },
            ),
        ]);
        let trajectory = SessionReplay::new(backend)
            .load(SESSION)
            .expect("gaps are tolerated: only ambiguous order is refused");
        assert_eq!(trajectory.len(), 1);
        assert_eq!(trajectory.turns()[0].sequence_range(), (0, 9));
        assert!(trajectory.turns()[0].is_complete());
    }

    /// Five `Output` records at sequences 0..=4.
    fn five_outputs() -> Arc<dyn SessionBackend> {
        backend_with(vec![
            output("a"),
            output("b"),
            output("c"),
            output("d"),
            output("e"),
        ])
    }

    #[test]
    fn load_prefix_retains_only_sequences_up_to_the_bound() {
        let replay = SessionReplay::new(five_outputs());
        let trajectory = replay
            .load_prefix(SESSION, 2)
            .expect("load_prefix must succeed");
        assert_eq!(trajectory.usage().records, 3);
        assert_eq!(
            trajectory.turns()[0].blocks(),
            [text("a"), text("b"), text("c")]
        );
        assert_eq!(trajectory.turns()[0].sequence_range(), (0, 2));
        assert_eq!(
            replay
                .load(SESSION)
                .expect("load must succeed")
                .usage()
                .records,
            5,
            "the whole-session load is unaffected"
        );
    }

    #[test]
    fn load_at_point_matches_load_prefix_for_the_same_position() {
        let replay = SessionReplay::new(five_outputs());
        let by_point = replay
            .load_at_point(&format!("{SESSION}:2"))
            .expect("load_at_point must succeed");
        let by_prefix = replay
            .load_prefix(SESSION, 2)
            .expect("load_prefix must succeed");
        assert_eq!(by_point.session_id(), by_prefix.session_id());
        assert_eq!(by_point.usage(), by_prefix.usage());
        assert_eq!(
            by_point.turns()[0].blocks(),
            by_prefix.turns()[0].blocks(),
            "the two addressings must agree block for block"
        );
    }

    #[test]
    fn malformed_point_id_is_refused() {
        let replay = SessionReplay::new(five_outputs());
        for point_id in ["nocolon", "s:abc", ":0", "s:"] {
            let err = match replay.load_at_point(point_id) {
                Err(err) => err,
                Ok(loaded) => panic!(
                    "`{point_id}` must be refused, loaded session `{}` instead",
                    loaded.session_id()
                ),
            };
            assert_eq!(validation_schema(&err), "point_id", "for `{point_id}`");
        }
    }

    #[test]
    fn point_id_naming_no_record_is_refused() {
        // Records start at sequence 3, so nothing sits at or below 1.
        let backend = backend_with_records(vec![
            SessionRecord::at(SESSION, 3, 1, output("a")),
            SessionRecord::at(SESSION, 4, 2, output("b")),
        ]);
        let replay = SessionReplay::new(backend);

        let err = replay
            .load_at_point(&format!("{SESSION}:1"))
            .expect_err("a point_id naming no recorded position must be refused");
        assert_eq!(validation_schema(&err), "point_id");

        let err = replay
            .load_prefix(SESSION, 1)
            .expect_err("an upto_sequence naming no recorded position must be refused");
        assert_eq!(validation_schema(&err), "point_id");
    }

    #[tokio::test]
    async fn stream_turn_yields_recorded_blocks_in_order_then_none() {
        use tokio_stream::StreamExt;

        let trajectory = SessionReplay::new(backend_with(vec![output("a"), output("b")]))
            .load(SESSION)
            .expect("load must succeed");
        let mut stream = trajectory.stream_turn(0).expect("turn 0 must stream");

        let mut seen = Vec::new();
        while let Some(item) = stream.next().await {
            seen.push(item.expect("a replayed item is never an error"));
        }
        assert_eq!(seen, vec![text("a"), text("b")]);
        assert!(
            stream.next().await.is_none(),
            "an exhausted replay stream stays None"
        );
        assert!(stream.next().await.is_none());
    }

    #[tokio::test]
    async fn into_stream_concatenates_turns_in_record_order() {
        use tokio_stream::StreamExt;

        let mut events = vec![output("a")];
        events.extend(terminator(1, 1, 1));
        events.push(output("b"));
        events.extend(terminator(2, 2, 2));
        events.push(output("c"));

        let trajectory = SessionReplay::new(backend_with(events))
            .load(SESSION)
            .expect("load must succeed");
        assert_eq!(trajectory.len(), 3);

        let mut stream = trajectory
            .into_stream()
            .expect("a non-empty replay streams");
        let mut seen = Vec::new();
        while let Some(item) = stream.next().await {
            seen.push(item.expect("a replayed item is never an error"));
        }
        assert_eq!(seen, vec![text("a"), text("b"), text("c")]);
    }

    #[test]
    fn empty_session_loads_empty_and_refuses_to_stream() {
        let replay = SessionReplay::new(Arc::new(InMemoryBackend::new()));
        let trajectory = replay
            .load("never-recorded")
            .expect("an unknown session loads as empty");
        assert!(trajectory.is_empty());
        assert_eq!(trajectory.len(), 0);
        assert_eq!(
            trajectory.usage(),
            ReplayUsage {
                records: 0,
                max_records: ReplayConfig::DEFAULT_MAX_RECORDS,
                blocks: 0,
                turns: 0,
                near_cap: false,
            }
        );

        let err = refusal(
            trajectory.stream_turn(0),
            "every turn index of an empty trajectory",
        );
        assert_eq!(validation_schema(&err), "turn_index");

        let err = refusal(
            trajectory.into_stream(),
            "a whole-session stream of an empty trajectory",
        );
        assert_eq!(validation_schema(&err), "session_id");
        assert!(err.to_string().contains("never-recorded"), "{err}");
    }

    #[test]
    fn out_of_range_turn_index_is_refused() {
        let trajectory = one_generation();
        assert!(trajectory.turn(0).is_some());
        assert!(trajectory.turn(1).is_none());

        let err = refusal(trajectory.stream_turn(1), "an out-of-range turn index");
        assert_eq!(validation_schema(&err), "turn_index");
        assert!(
            err.to_string().contains("replayed 1 turns"),
            "the refusal must name the available range: {err}"
        );
    }

    #[test]
    fn two_loads_produce_identical_block_sequences() {
        let mut events = vec![output("a"), output("b")];
        events.extend(terminator(5, 1, 2));
        let replay = SessionReplay::new(backend_with(events));

        let first = replay.load(SESSION).expect("first load must succeed");
        let second = replay.load(SESSION).expect("second load must succeed");
        assert_eq!(first.usage(), second.usage());
        assert_eq!(
            drain_now(first.into_stream().expect("first replay streams")),
            drain_now(second.into_stream().expect("second replay streams")),
        );
    }

    #[test]
    fn usage_reports_records_blocks_turns_and_near_cap_at_warn_fraction() {
        let events = || {
            let mut events = vec![output("a"), output("b")];
            events.extend(terminator(1, 1, 1));
            events.push(output("c"));
            events.push(output("d"));
            events.extend(terminator(2, 2, 2));
            events
        };

        // 8 records, 4 blocks, 2 turns; 8 >= 0.9 * 8, so the flag is set.
        let at_cap = SessionReplay::with_config(
            backend_with(events()),
            ReplayConfig::new(8, 16, Some(0.9)).expect("config must build"),
        )
        .expect("replay must build")
        .load(SESSION)
        .expect("load must succeed");
        assert_eq!(
            at_cap.usage(),
            ReplayUsage {
                records: 8,
                max_records: 8,
                blocks: 4,
                turns: 2,
                near_cap: true,
            }
        );

        // Same trajectory, a far higher cap: 8 < 0.9 * 100.
        let roomy = SessionReplay::with_config(
            backend_with(events()),
            ReplayConfig::new(100, 16, Some(0.9)).expect("config must build"),
        )
        .expect("replay must build")
        .load(SESSION)
        .expect("load must succeed");
        assert!(!roomy.usage().near_cap);
        assert_eq!(roomy.usage().records, 8);

        // `None` disables the flag even at the cap.
        let disabled = SessionReplay::with_config(
            backend_with(events()),
            ReplayConfig::new(8, 16, None).expect("config must build"),
        )
        .expect("replay must build")
        .load(SESSION)
        .expect("load must succeed");
        assert!(!disabled.usage().near_cap);
    }

    #[test]
    fn stream_turn_leaves_the_trajectory_replayable() {
        let trajectory = SessionReplay::new(backend_with(vec![output("a"), output("b")]))
            .load(SESSION)
            .expect("load must succeed");

        let first = drain_now(trajectory.stream_turn(0).expect("turn 0 must stream"));
        let second = drain_now(trajectory.stream_turn(0).expect("turn 0 must stream again"));
        assert_eq!(first, vec![text("a"), text("b")]);
        assert_eq!(first, second, "the clone leaves the turn intact");
        assert_eq!(
            trajectory.turns()[0].blocks(),
            [text("a"), text("b")],
            "the retained blocks are unchanged after streaming"
        );
        assert_eq!(trajectory.usage().blocks, 2);
    }

    #[test]
    fn turn_response_carries_recorded_tokens_and_latency() {
        let trajectory = one_generation();
        let response = trajectory.turns()[0].response("m", ProviderEndpoint::LlamaCpp);
        assert_eq!(response.model, "m");
        assert_eq!(response.provider, ProviderEndpoint::LlamaCpp);
        assert_eq!(response.prompt_tokens, 10);
        assert_eq!(response.completion_tokens, 5);
        assert!(
            (response.duration_secs - 1.5).abs() < 1e-9,
            "1500 ms must report as 1.5 s, got {}",
            response.duration_secs
        );
        assert_eq!(response.content, vec![text("a")]);
        assert_eq!(
            response.finish_reason, None,
            "no SessionEvent records a stop reason"
        );
        assert_eq!(response.prompt_cache_usage, None);

        // An interrupted turn reports zeros rather than inventing accounting.
        let incomplete = SessionReplay::new(backend_with(vec![output("x")]))
            .load(SESSION)
            .expect("load must succeed");
        let response = incomplete.turns()[0].response("m", ProviderEndpoint::LlamaCpp);
        assert_eq!(response.prompt_tokens, 0);
        assert_eq!(response.completion_tokens, 0);
        assert_eq!(response.duration_secs, 0.0);
    }
}
