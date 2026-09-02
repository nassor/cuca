//! Bounded in-process vector recall of offloaded turns (`service-vector-store`).
//!
//! [`InMemoryVectorStore`] implements the memory plugin's [`VectorStore`]
//! offload seam, keeps the offloaded turns in a capped arena of unit-normalized
//! embeddings, and answers top-k similarity queries that the caller injects
//! back into the next prompt.
//!
//! # Hard dependency on `plugin-memory`
//!
//! The store *is* the seam's implementation, so this feature enables the plugin
//! that declares it: `service-vector-store = ["plugin-memory", "dep:wide"]`. It
//! is wired through
//! [`MemoryPlugin::with_extensions`](crate::plugins::memory::MemoryPlugin::with_extensions),
//! the only constructor that accepts a store. The dependency is one-way:
//! `plugin-memory` never references this module.
//!
//! # Explicit-call contract
//!
//! An explicit-call service, never a [`crate::plugin::CucaPlugin`]
//! ([`crate::services`] owns that contract), so registering it on the builder is
//! a compile error rather than an inert no-op. Two entry directions:
//!
//! - **Writes** arrive through
//!   [`CompactionStrategy::Offload`](crate::plugins::memory::CompactionStrategy::Offload),
//!   which calls [`VectorStore::store_turns`] with the fixed `"cuca-memory"`
//!   session hint.
//! - **Reads** are direct calls: [`InMemoryVectorStore::retrieve`],
//!   [`InMemoryVectorStore::retrieve_in`], and
//!   [`InMemoryVectorStore::retrieve_embedding`].
//!
//! Every public method on the store returns [`VectorStoreError`]; the
//! [`VectorStore`] seam impl converts it into [`PluginError`] at the
//! boundary, because that signature belongs to the memory plugin.
//!
//! # Mandatory hand-off
//!
//! A [`RetrievalReport`] is inert. Recall reaches the model only when the
//! application applies it with [`RetrievalReport::inject`]; dropping the report
//! discards the recall. Injection is idempotent through the
//! [`RECALL_RENDER_MARKER`] prefix (an existing recall message is replaced, not
//! appended to), and [`RecallInjection`] names exactly what the call did, so a
//! no-op is never silent.
//!
//! # Synchronous embedding
//!
//! `store_turns` runs inside the synchronous
//! [`CucaPlugin::on_request`](crate::plugin::CucaPlugin::on_request) hook, so
//! [`Embedder`] is a caller-supplied *synchronous* bridge, exactly like
//! [`Summarizer`](crate::plugins::memory::Summarizer). No implementation may
//! await, and the crate ships no embeddings route of its own: no provider
//! adapter exposes one, and a caller wanting HTTP embeddings owns that blocking
//! decision.
//!
//! # Bounds
//!
//! **Cap.** [`VectorStoreConfig`] validates three non-zero bounds at
//! construction: `max_entries` (live turns), `dimensions` (exact embedding
//! width), and `max_entry_bytes` (largest accepted turn), and rejects a
//! `max_entries × dimensions` arena that would overflow `usize` in floats or
//! in bytes. Resident size is therefore bounded by
//! `max_entries × (max_entry_bytes + dimensions × 4 + session_hint + constant metadata)`.
//!
//! **At the cap.** Deterministic FIFO: the oldest entry is evicted and its
//! arena slot is reused in place. The arena is allocated to its cap of
//! `max_entries × dimensions` floats at construction and never reallocates, so
//! that resident bound holds from the first insert with no transient
//! reallocation peak. A batch longer than `max_entries` keeps the newest
//! `max_entries` turns and counts the rest as evictions; only that surviving
//! tail is embedded, so the dropped head never pays for an [`Embedder`] call.
//! Nothing is ever truncated: a turn over `max_entry_bytes`, an embedding of
//! the wrong width, and a non-finite component are all rejected
//! ([`VectorStoreError::EntryTooLarge`] and [`VectorStoreError::Embedding`],
//! which the seam converts into [`PluginError::Validation`]), and the
//! rejection aborts the whole batch — the byte check covers every turn,
//! including the head the cap would drop — so the memory plugin's
//! all-or-nothing offload puts every turn back where it was and records the
//! error in
//! [`CompressionReport::last_error`](crate::plugins::memory::CompressionReport::last_error).
//!
//! **Usage reading.** [`InMemoryVectorStore::usage`] is O(1) and reports
//! entries, capacity, fill fraction, cumulative evictions, and the zero-norm
//! count; [`VectorStoreConfig::with_warn_fraction`] arms the
//! [`VectorStoreUsage::near_cap`] flag. There is no TTL: this is conversation
//! history rather than a cache, and FIFO needs no clock, which also keeps the
//! store free of wall-clock reads.
//!
//! # Retrieval performance
//!
//! Every embedding is unit-normalized once at insert, so cosine similarity is a
//! single dot product per entry at query time. Vectors live in one contiguous
//! `Vec<f32>` slot arena (`slot s` occupies `arena[s * d .. (s + 1) * d]`)
//! rather than in per-entry allocations, so a scan walks memory linearly.
//!
//! The kernel is `wide::f32x8` with four independent accumulators (32 lanes per
//! iteration) to hide FMA latency, then an 8-lane loop, then a scalar tail;
//! nested `as_chunks` splits make every SIMD load exactly eight wide, which
//! removes both bounds checks and the zero-padding branch of `From<&[f32]>`.
//! SIMD selection is compile-time only (`wide` has no runtime dispatch): a
//! baseline `x86-64` build lowers `f32x8` to two `f32x4` halves, and
//! `RUSTFLAGS="-C target-cpu=native"` (or `-C target-feature=+avx,+fma`)
//! unlocks the 256-bit single-rounding `vfmadd` path.
//!
//! Top-k is exact and O(n + k log k): `select_nth_unstable_by` partitions the
//! scored vector, then only the k survivors are sorted. Per query the store
//! allocates one embedding `Vec` (skipped by
//! [`InMemoryVectorStore::retrieve_embedding`]), one scored `Vec` of 16-byte
//! tuples, and k message clones. Nothing else.
//!
//! Scores and ranking are bit-deterministic for a given build: lane order and
//! accumulator order are fixed and the horizontal sum is a fixed-order tree
//! rather than `f32x8::reduce_add`, whose addition order is documented as
//! unspecified. Across builds only exact ties are guaranteed stable — they
//! break on the monotonic insertion sequence — because an FMA build and a
//! non-FMA build round the products differently, so a near-tie may reorder.
//! The scan is exact, not approximate: the bounded `n` sits far below the
//! crossover where a graph index pays off, and an RNG-driven approximate index
//! would trade the determinism contract away for nothing.
//!
//! # Text projection
//!
//! An entry is embedded from a space-joined projection of its blocks:
//! `Text` verbatim, `ToolCall` contributes its `name`, `ToolResult` contributes
//! its `output`. `Thinking` and `ImageBase64` are excluded: offloaded history is
//! mostly tool traffic, so a text-only projection would embed the empty string
//! for most turns, while reasoning text and base64 image bytes are noise in a
//! similarity query. A caller who disagrees pre-clamps the turns before offload.
//!
//! # Ordering against compaction
//!
//! The documented order is: [`MemoryPlugin::compress`](crate::plugins::memory::MemoryPlugin::compress)
//! out of band, then `retrieve` + [`RetrievalReport::inject`], then dispatch.
//! Recall is a caller-driven step, so unlike memory's graph injection it cannot
//! run after compaction inside the hook, and memory's
//! `DropObservations`/`SlidingWindow` strategies may drop an injected System
//! message. Injection also changes the digest
//! [`digest_request`](crate::services::prompt_cache::digest_request) computes,
//! because that digest covers the full message list.
//!
//! An injected recall message is at most `k × max_entry_bytes` of projected
//! text plus the one-line header, so a caller sizing `k` bounds how much
//! recall can grow the prompt.
//!
//! # Sensitive data
//!
//! Entries are full-fidelity turns: system prompts, user messages, tool
//! arguments and results. The store is process-local by construction. It has no
//! serde derives, no snapshot, no `CucaExport` section, and writes no file, so
//! nothing here can leak a turn to disk on CUCA's behalf.

use std::collections::VecDeque;
// Formatting straight into the output `String` instead of through a temporary
// one. `write!` on a `String` cannot fail, so the `Result` is discarded rather
// than unwrapped (this crate never panics outside tests).
use std::fmt::Write as _;
use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};

use wide::f32x8;

use crate::error::PluginError;
use crate::plugins::memory::VectorStore;
use crate::request::UnifiedRequest;
use crate::types::{MessageContentBlock, MessageRole, UnifiedMessage};

/// Prefix that makes recall injection idempotent: [`RetrievalReport::inject`]
/// scans for it and replaces the message it finds instead of appending a
/// second one.
pub const RECALL_RENDER_MARKER: &str = "CUCA recall:";

/// The `schema` field of the [`PluginError::Validation`] that
/// [`VectorStoreError`] converts into.
const VALIDATION_SCHEMA: &str = "vector-store";

/// Everything this module can refuse to do.
///
/// The store's own surface returns this type; the [`VectorStore`] seam impl
/// converts it into [`PluginError`], because that signature belongs to the
/// memory plugin rather than to this service.
#[derive(Debug, Clone)]
pub enum VectorStoreError {
    /// Invalid configuration: a zero bound, a warn fraction outside
    /// `(0.0, 1.0]`, an arena size that overflows `usize`, or a zero `k`.
    Config(String),
    /// An embedding the store cannot use: the wrong width, or a non-finite
    /// component or squared norm.
    Embedding(String),
    /// A turn larger than [`VectorStoreConfig::max_entry_bytes`]; rejected,
    /// never truncated.
    EntryTooLarge {
        /// Measured payload bytes of the offending turn.
        bytes: usize,
        /// The configured cap it exceeded.
        max_entry_bytes: usize,
    },
    /// The state lock was poisoned by a panic in another thread; reported
    /// rather than papered over with an empty store.
    Poisoned,
    /// Measuring a tool call's arguments failed (unreachable for a
    /// `serde_json::Value`, but library code never unwraps).
    Json(String),
    /// The caller's [`Embedder`] failed; carried through unchanged.
    Embedder(PluginError),
}

impl std::fmt::Display for VectorStoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VectorStoreError::Config(msg) => write!(f, "vector store configuration error: {msg}"),
            VectorStoreError::Embedding(msg) => write!(f, "vector store embedding error: {msg}"),
            VectorStoreError::EntryTooLarge {
                bytes,
                max_entry_bytes,
            } => write!(
                f,
                "vector store turn of {bytes} bytes exceeds max_entry_bytes {max_entry_bytes}"
            ),
            VectorStoreError::Poisoned => write!(f, "vector store state lock is poisoned"),
            VectorStoreError::Json(msg) => write!(f, "vector store JSON error: {msg}"),
            VectorStoreError::Embedder(err) => write!(f, "vector store embedder failed: {err}"),
        }
    }
}

impl std::error::Error for VectorStoreError {}

/// Lossy interop edge into the crate's plugin error contract.
///
/// [`VectorStore::store_turns`] carries the memory plugin's signature, so the
/// seam converts: a rejected turn or embedding becomes
/// [`PluginError::Validation`] under the `"vector-store"` schema, the caller's
/// own [`Embedder`] failure passes through untouched, and everything else
/// becomes [`PluginError::Internal`] carrying the
/// [`std::fmt::Display`] text. Callers that need the typed
/// `bytes`/`max_entry_bytes` match on [`VectorStoreError`] before converting.
impl From<VectorStoreError> for PluginError {
    fn from(error: VectorStoreError) -> Self {
        match error {
            VectorStoreError::Embedder(inner) => inner,
            VectorStoreError::Embedding(message) => PluginError::Validation {
                schema: VALIDATION_SCHEMA.to_string(),
                message,
            },
            oversized @ VectorStoreError::EntryTooLarge { .. } => PluginError::Validation {
                schema: VALIDATION_SCHEMA.to_string(),
                message: oversized.to_string(),
            },
            other => PluginError::Internal(other.to_string()),
        }
    }
}

/// Extension seam: text to dense vector.
///
/// Synchronous by construction: the in-crate caller of
/// [`VectorStore::store_turns`] is `CompactionStrategy::Offload`, running
/// inside the synchronous `CucaPlugin::on_request` hook, so no implementation
/// may await.
pub trait Embedder: Send + Sync {
    /// Embed `text` into a dense vector of the store's configured width.
    ///
    /// # Errors
    ///
    /// Any [`PluginError`] the backend produces. The store never rewrites it:
    /// [`VectorStore::store_turns`] returns it as is, and a query path carries
    /// it in [`VectorStoreError::Embedder`], which converts back to the same
    /// [`PluginError`].
    fn embed(&self, text: &str) -> Result<Vec<f32>, PluginError>;
}

/// Validated bounds for an [`InMemoryVectorStore`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VectorStoreConfig {
    /// Maximum live entries; non-zero. FIFO eviction at the cap.
    pub max_entries: usize,
    /// Exact embedding width; non-zero. Every vector must be this wide.
    pub dimensions: usize,
    /// Largest accepted turn, in payload bytes; non-zero. A larger turn is
    /// rejected, never truncated.
    pub max_entry_bytes: usize,
    /// Fill fraction at which [`VectorStoreUsage::near_cap`] flips; `None`
    /// leaves it permanently `false`.
    pub warn_fraction: Option<f32>,
}

impl VectorStoreConfig {
    /// Build a validated configuration with no near-cap warning.
    ///
    /// # Errors
    ///
    /// [`VectorStoreError::Config`] when any of the three bounds is zero, or
    /// when the resulting arena overflows `usize`.
    pub fn new(
        max_entries: usize,
        dimensions: usize,
        max_entry_bytes: usize,
    ) -> Result<Self, VectorStoreError> {
        let config = Self {
            max_entries,
            dimensions,
            max_entry_bytes,
            warn_fraction: None,
        };
        config.validate()?;
        Ok(config)
    }

    /// Arm the near-cap warning at `fraction` of [`Self::max_entries`].
    ///
    /// # Errors
    ///
    /// [`VectorStoreError::Config`] when `fraction` is outside `(0.0, 1.0]`
    /// or not finite.
    pub fn with_warn_fraction(mut self, fraction: f32) -> Result<Self, VectorStoreError> {
        self.warn_fraction = Some(fraction);
        self.validate()?;
        Ok(self)
    }

    /// Reject bounds that cannot be honored.
    ///
    /// Every field is `pub`, so a struct literal reaches
    /// [`InMemoryVectorStore::new`] without passing through [`Self::new`] or
    /// [`Self::with_warn_fraction`]: this is the one check both paths share,
    /// and it is what makes the arena allocation in
    /// [`InMemoryVectorStore::new`] unable to overflow.
    fn validate(&self) -> Result<(), VectorStoreError> {
        if self.max_entries == 0 {
            return Err(config_error("max_entries must be non-zero"));
        }
        if self.dimensions == 0 {
            return Err(config_error("dimensions must be non-zero"));
        }
        if self.max_entry_bytes == 0 {
            return Err(config_error("max_entry_bytes must be non-zero"));
        }
        if let Some(fraction) = self.warn_fraction
            && (!fraction.is_finite() || fraction <= 0.0 || fraction > 1.0)
        {
            return Err(VectorStoreError::Config(format!(
                "warn_fraction must be in (0.0, 1.0], got {fraction}"
            )));
        }
        let floats = self
            .max_entries
            .checked_mul(self.dimensions)
            .ok_or_else(|| {
                VectorStoreError::Config(format!(
                    "max_entries {} × dimensions {} overflows usize",
                    self.max_entries, self.dimensions
                ))
            })?;
        if floats.checked_mul(size_of::<f32>()).is_none() {
            return Err(VectorStoreError::Config(format!(
                "an arena of {floats} floats overflows usize in bytes"
            )));
        }
        Ok(())
    }
}

fn config_error(message: &str) -> VectorStoreError {
    VectorStoreError::Config(message.to_string())
}

/// O(1) size reading of an [`InMemoryVectorStore`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VectorStoreUsage {
    /// Live entries.
    pub entries: usize,
    /// Configured [`VectorStoreConfig::max_entries`].
    pub capacity: usize,
    /// `entries / capacity`.
    pub fraction: f32,
    /// Whether `fraction` reached [`VectorStoreConfig::warn_fraction`]; always
    /// `false` when no warning fraction is configured.
    pub near_cap: bool,
    /// Cumulative entries dropped by capacity eviction.
    pub evicted_entries: u64,
    /// Live entries whose embedding had a zero norm; they score `0.0` against
    /// every query.
    pub zero_norm_entries: usize,
}

/// One hit of a [`RetrievalReport`].
#[derive(Debug, Clone, PartialEq)]
pub struct RetrievedTurn {
    /// The stored turn, verbatim.
    pub message: UnifiedMessage,
    /// Cosine similarity in `[-1.0, 1.0]`; exactly `0.0` when either side has
    /// a zero norm, including a squared norm that underflowed, since such a
    /// vector is stored as zeros.
    pub score: f32,
    /// Monotonic insertion order; the tie-break for equal scores.
    pub sequence: u64,
    /// The session hint the turn was stored under.
    pub session_hint: String,
}

/// The result of one similarity query.
#[derive(Debug, Clone, PartialEq)]
pub struct RetrievalReport {
    /// Hits, best first: score descending, then newest sequence first. The
    /// order is bit-deterministic for a given build; across builds only exact
    /// ties are guaranteed stable, because FMA and non-FMA lowerings round
    /// differently (see the [module docs](crate::services::vector_store)).
    pub turns: Vec<RetrievedTurn>,
    /// Entries actually scored, after the session-hint filter.
    pub scanned: usize,
}

/// What [`RetrievalReport::inject`] did to the request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecallInjection {
    /// A recall message was added; none was present.
    Inserted,
    /// An existing recall message was replaced.
    Replaced,
    /// The report was empty and an existing recall message was removed.
    Removed,
    /// The report was empty and no recall message was present.
    Absent,
}

impl RetrievalReport {
    /// Render the hits as the body of a recall system message.
    ///
    /// `None` for an empty report. The text is byte-deterministic for a given
    /// report: line one is [`RECALL_RENDER_MARKER`] plus the hit count, then
    /// one `N. [score] projection` line per hit in report order.
    pub fn render(&self) -> Option<String> {
        if self.turns.is_empty() {
            return None;
        }
        let mut out = format!(
            "{RECALL_RENDER_MARKER} {} offloaded turn(s), best first",
            self.turns.len()
        );
        for (rank, turn) in self.turns.iter().enumerate() {
            out.push('\n');
            let _ = write!(
                out,
                "{}. [{:.4}] {}",
                rank + 1,
                turn.score,
                text_projection(&turn.message)
            );
        }
        Some(out)
    }

    /// Apply the recall to `request`, replacing any recall this report's
    /// marker already owns.
    ///
    /// A non-empty report is rendered into a System message placed immediately
    /// before the most recent User message (appended when the request has no
    /// User message), so recall stays adjacent to the newest question across
    /// turns. An empty report only removes.
    pub fn inject(&self, request: &mut UnifiedRequest) -> RecallInjection {
        let existing = request.messages.iter().position(is_recall_message);
        let had_existing = existing.is_some();
        if let Some(index) = existing {
            request.messages.remove(index);
        }
        let Some(text) = self.render() else {
            return if had_existing {
                RecallInjection::Removed
            } else {
                RecallInjection::Absent
            };
        };
        let message = UnifiedMessage::system(text);
        match request
            .messages
            .iter()
            .rposition(|m| m.role == MessageRole::User)
        {
            Some(index) => request.messages.insert(index, message),
            None => request.messages.push(message),
        }
        if had_existing {
            RecallInjection::Replaced
        } else {
            RecallInjection::Inserted
        }
    }
}

/// Whether `message` is a previously injected recall message.
fn is_recall_message(message: &UnifiedMessage) -> bool {
    message.role == MessageRole::System
        && matches!(
            message.content.first(),
            Some(MessageContentBlock::Text(text)) if text.starts_with(RECALL_RENDER_MARKER)
        )
}

/// One stored turn and the arena slot holding its unit vector.
struct StoredTurn {
    message: UnifiedMessage,
    slot: usize,
    zero_norm: bool,
    sequence: u64,
    /// Shared with every other entry of the same `store_turns` call: the hint
    /// is one allocation per batch, and a refcount bump per entry.
    session_hint: Arc<str>,
}

/// Everything behind the store's single `RwLock`: queries share the read
/// lock, and only a commit takes the write lock.
struct StoreState {
    /// Oldest first; the front is the eviction victim.
    entries: VecDeque<StoredTurn>,
    /// Slot arena: `slot s` is `arena[s * dimensions .. (s + 1) * dimensions]`.
    /// Allocated to `max_entries * dimensions` at construction; never grows.
    arena: Vec<f32>,
    next_sequence: u64,
    evicted: u64,
    zero_norm: usize,
}

/// Bounded in-process vector store behind the memory plugin's offload seam.
///
/// See the [module docs](crate::services::vector_store) for the bounds story,
/// the retrieval performance contract, and the mandatory
/// [`RetrievalReport::inject`] hand-off.
pub struct InMemoryVectorStore {
    config: VectorStoreConfig,
    embedder: Arc<dyn Embedder>,
    state: RwLock<StoreState>,
}

impl InMemoryVectorStore {
    /// Build a store over `config`, embedding through `embedder`.
    ///
    /// Both collections are sized to the cap here: the entry deque to
    /// `max_entries`, and the slot arena to `max_entries × dimensions` floats.
    /// Neither ever reallocates afterwards, so the resident bound in the
    /// [module docs](crate::services::vector_store) holds with no transient
    /// peak.
    ///
    /// # Errors
    ///
    /// [`VectorStoreError::Config`] when any configured bound is zero, the
    /// warn fraction is out of range, or the arena size overflows `usize`.
    pub fn new(
        config: VectorStoreConfig,
        embedder: Arc<dyn Embedder>,
    ) -> Result<Self, VectorStoreError> {
        config.validate()?;
        Ok(Self {
            config,
            embedder,
            state: RwLock::new(StoreState {
                entries: VecDeque::with_capacity(config.max_entries),
                // `validate` rejected an overflowing product, so the multiply
                // below cannot wrap.
                arena: Vec::with_capacity(config.max_entries * config.dimensions),
                next_sequence: 0,
                evicted: 0,
                zero_norm: 0,
            }),
        })
    }

    /// Take the shared read lock, mapping a poisoned lock to an error rather
    /// than reporting an empty store.
    fn read_state(&self) -> Result<RwLockReadGuard<'_, StoreState>, VectorStoreError> {
        self.state.read().map_err(|_| VectorStoreError::Poisoned)
    }

    /// Take the exclusive write lock, with the same poison-to-error mapping.
    fn write_state(&self) -> Result<RwLockWriteGuard<'_, StoreState>, VectorStoreError> {
        self.state.write().map_err(|_| VectorStoreError::Poisoned)
    }

    /// O(1) size reading.
    ///
    /// # Errors
    ///
    /// [`VectorStoreError::Poisoned`] when the state lock is poisoned.
    pub fn usage(&self) -> Result<VectorStoreUsage, VectorStoreError> {
        let state = self.read_state()?;
        let entries = state.entries.len();
        let capacity = self.config.max_entries;
        let fraction = entries as f32 / capacity as f32;
        Ok(VectorStoreUsage {
            entries,
            capacity,
            fraction,
            near_cap: self
                .config
                .warn_fraction
                .is_some_and(|warn| fraction >= warn),
            evicted_entries: state.evicted,
            zero_norm_entries: state.zero_norm,
        })
    }

    /// Live entry count.
    ///
    /// # Errors
    ///
    /// [`VectorStoreError::Poisoned`] when the state lock is poisoned.
    pub fn len(&self) -> Result<usize, VectorStoreError> {
        Ok(self.read_state()?.entries.len())
    }

    /// Whether the store holds no entries.
    ///
    /// # Errors
    ///
    /// [`VectorStoreError::Poisoned`] when the state lock is poisoned.
    pub fn is_empty(&self) -> Result<bool, VectorStoreError> {
        Ok(self.read_state()?.entries.is_empty())
    }

    /// Configured [`VectorStoreConfig::max_entries`]; takes no lock.
    pub fn capacity(&self) -> usize {
        self.config.max_entries
    }

    /// Top-`k` hits for `query` across every session hint.
    ///
    /// # Errors
    ///
    /// [`VectorStoreError::Config`] for `k == 0`;
    /// [`VectorStoreError::Poisoned`] for a poisoned state lock;
    /// [`VectorStoreError::Embedding`] when the embedded query has the wrong
    /// width or a non-finite component; [`VectorStoreError::Embedder`]
    /// carrying whatever the [`Embedder`] returned, unchanged.
    pub fn retrieve(&self, query: &str, k: usize) -> Result<RetrievalReport, VectorStoreError> {
        let mut vector = self.embed_query(query, k)?;
        normalize_in_place(&mut vector)?;
        self.scan(None, &vector, k)
    }

    /// Top-`k` hits for `query` among entries stored under `session_hint`.
    ///
    /// # Errors
    ///
    /// Same as [`Self::retrieve`].
    pub fn retrieve_in(
        &self,
        session_hint: &str,
        query: &str,
        k: usize,
    ) -> Result<RetrievalReport, VectorStoreError> {
        let mut vector = self.embed_query(query, k)?;
        normalize_in_place(&mut vector)?;
        self.scan(Some(session_hint), &vector, k)
    }

    /// Top-`k` hits for a caller-computed `query` vector; `None` scans every
    /// session hint.
    ///
    /// The vector need not be normalized: it is validated and unit-normalized
    /// into a local copy, leaving the caller's slice untouched.
    ///
    /// # Errors
    ///
    /// [`VectorStoreError::Config`] for `k == 0`;
    /// [`VectorStoreError::Poisoned`] for a poisoned state lock;
    /// [`VectorStoreError::Embedding`] when `query` has the wrong width or a
    /// non-finite component.
    pub fn retrieve_embedding(
        &self,
        session_hint: Option<&str>,
        query: &[f32],
        k: usize,
    ) -> Result<RetrievalReport, VectorStoreError> {
        self.check_k(k)?;
        self.check_width(query.len())?;
        let mut vector = query.to_vec();
        normalize_in_place(&mut vector)?;
        self.scan(session_hint, &vector, k)
    }

    /// Embed a query string, validating `k` first so a zero `k` never pays for
    /// an embedding call.
    fn embed_query(&self, query: &str, k: usize) -> Result<Vec<f32>, VectorStoreError> {
        self.check_k(k)?;
        let vector = self
            .embedder
            .embed(query)
            .map_err(VectorStoreError::Embedder)?;
        self.check_width(vector.len())?;
        Ok(vector)
    }

    fn check_k(&self, k: usize) -> Result<(), VectorStoreError> {
        if k == 0 {
            return Err(config_error("k must be non-zero"));
        }
        Ok(())
    }

    fn check_width(&self, width: usize) -> Result<(), VectorStoreError> {
        if width != self.config.dimensions {
            return Err(VectorStoreError::Embedding(format!(
                "embedding must have {} dimensions, got {width}",
                self.config.dimensions
            )));
        }
        Ok(())
    }

    /// Score every matching entry against a unit `query` and rank the top `k`.
    fn scan(
        &self,
        session_hint: Option<&str>,
        query: &[f32],
        k: usize,
    ) -> Result<RetrievalReport, VectorStoreError> {
        let dimensions = self.config.dimensions;
        let state = self.read_state()?;
        // (score, sequence, index): 16 bytes, one allocation per query.
        let mut scored: Vec<(f32, u64, usize)> = Vec::with_capacity(state.entries.len());
        for (index, entry) in state.entries.iter().enumerate() {
            // `Arc<str>` derefs to `str`, so the filter compares bytes and
            // allocates nothing.
            if let Some(hint) = session_hint
                && &*entry.session_hint != hint
            {
                continue;
            }
            let start = entry.slot * dimensions;
            // A zero-norm entry is stored as zeros, so it scores 0.0 with no
            // branch in the hot loop.
            let score = dot(query, &state.arena[start..start + dimensions]);
            scored.push((score, entry.sequence, index));
        }
        let scanned = scored.len();
        let rank =
            |a: &(f32, u64, usize), b: &(f32, u64, usize)| b.0.total_cmp(&a.0).then(b.1.cmp(&a.1));
        if scored.len() > k {
            // O(n) partition, then sort only the survivors: O(n + k log k).
            scored.select_nth_unstable_by(k - 1, rank);
            scored.truncate(k);
        }
        scored.sort_unstable_by(rank);
        let turns = scored
            .into_iter()
            .filter_map(|(score, sequence, index)| {
                let entry = state.entries.get(index)?;
                Some(RetrievedTurn {
                    message: entry.message.clone(),
                    score,
                    sequence,
                    session_hint: entry.session_hint.to_string(),
                })
            })
            .collect();
        Ok(RetrievalReport { turns, scanned })
    }
}

impl VectorStore for InMemoryVectorStore {
    /// Persist `turns` under `session_hint`, all or nothing.
    ///
    /// Staging runs outside the lock: every turn is measured, and the newest
    /// `max_entries` turns are projected, embedded, width-checked, and
    /// normalized into one contiguous scratch buffer, so a slow [`Embedder`]
    /// never blocks a concurrent query, concurrent queries share the read
    /// lock, and a rejected turn leaves the store byte-identical. The commit
    /// phase is infallible.
    ///
    /// A batch longer than `max_entries` cannot survive the commit whole, so
    /// only the surviving tail is embedded; the dropped head is still
    /// size-checked, because the rejection contract covers the batch, and it
    /// counts as [`VectorStoreUsage::evicted_entries`].
    ///
    /// # Errors
    ///
    /// [`PluginError::Validation`] when a turn exceeds
    /// [`VectorStoreConfig::max_entry_bytes`], or an embedding has the wrong
    /// width or a non-finite component; [`PluginError::Internal`] when the
    /// state lock is poisoned; whatever the [`Embedder`] returned, unchanged.
    /// The internal [`VectorStoreError`]s convert through [`From`], because
    /// this signature belongs to the memory plugin.
    fn store_turns(&self, session_hint: &str, turns: &[UnifiedMessage]) -> Result<(), PluginError> {
        if turns.is_empty() {
            return Ok(());
        }
        // The byte-size check is the rejection contract, so it covers every
        // turn, including the head the cap drops below.
        for turn in turns {
            let bytes = entry_bytes(turn)?;
            if bytes > self.config.max_entry_bytes {
                return Err(VectorStoreError::EntryTooLarge {
                    bytes,
                    max_entry_bytes: self.config.max_entry_bytes,
                }
                .into());
            }
        }

        // Only the newest `max_entries` turns can survive the commit, so only
        // they are embedded; `kept.len() <= max_entries`, whose product with
        // `dimensions` the config already proved not to overflow.
        let skipped = turns.len().saturating_sub(self.config.max_entries);
        let kept = &turns[skipped..];
        let dimensions = self.config.dimensions;
        let mut staged: Vec<f32> = Vec::with_capacity(kept.len() * dimensions);
        let mut zero_norms: Vec<bool> = Vec::with_capacity(kept.len());
        for turn in kept {
            let mut vector = self.embedder.embed(&text_projection(turn))?;
            self.check_width(vector.len())?;
            zero_norms.push(!normalize_in_place(&mut vector)?);
            staged.extend_from_slice(&vector);
        }
        // One allocation per batch; each entry keeps a refcount, not a copy.
        let hint: Arc<str> = Arc::from(session_hint);

        let mut state = self.write_state()?;
        // The head never reached the arena, but the cap is what dropped it, so
        // it is an eviction like any other.
        state.evicted += skipped as u64;
        for ((index, turn), zero_norm) in kept.iter().enumerate().zip(zero_norms) {
            let row = &staged[index * dimensions..(index + 1) * dimensions];
            let victim = if state.entries.len() >= self.config.max_entries {
                state.entries.pop_front()
            } else {
                None
            };
            let slot = match victim {
                Some(victim) => {
                    state.evicted += 1;
                    if victim.zero_norm {
                        state.zero_norm -= 1;
                    }
                    let start = victim.slot * dimensions;
                    state.arena[start..start + dimensions].copy_from_slice(row);
                    victim.slot
                }
                None => {
                    // Fresh slot: extending the arena into its reserved
                    // capacity is also the row write.
                    let slot = state.arena.len() / dimensions;
                    state.arena.extend_from_slice(row);
                    slot
                }
            };
            if zero_norm {
                state.zero_norm += 1;
            }
            let sequence = state.next_sequence;
            state.next_sequence += 1;
            state.entries.push_back(StoredTurn {
                message: turn.clone(),
                slot,
                zero_norm,
                sequence,
                session_hint: Arc::clone(&hint),
            });
        }
        Ok(())
    }
}

/// The text an entry is embedded from: `Text` verbatim, `ToolCall` names, and
/// `ToolResult` outputs, space-joined in block order.
///
/// `Thinking` and `ImageBase64` are excluded; see the
/// [module docs](crate::services::vector_store).
fn text_projection(message: &UnifiedMessage) -> String {
    let mut bytes = 0usize;
    let mut parts = 0usize;
    for block in &message.content {
        if let Some(part) = projected_part(block) {
            bytes += part.len();
            parts += 1;
        }
    }
    let mut out = String::with_capacity(bytes + parts.saturating_sub(1));
    let mut first = true;
    for block in &message.content {
        if let Some(part) = projected_part(block) {
            if !first {
                out.push(' ');
            }
            first = false;
            out.push_str(part);
        }
    }
    out
}

/// The projected slice of one block, or `None` when the block is excluded.
fn projected_part(block: &MessageContentBlock) -> Option<&str> {
    match block {
        MessageContentBlock::Text(text) => Some(text.as_str()),
        MessageContentBlock::ToolCall { name, .. } => Some(name.as_str()),
        MessageContentBlock::ToolResult { output, .. } => Some(output.as_str()),
        MessageContentBlock::Thinking { .. } | MessageContentBlock::ImageBase64 { .. } => None,
    }
}

/// Counts bytes written without keeping them, so tool-call arguments can be
/// measured as serialized JSON with no intermediate `String`.
struct CountingWriter(usize);

impl std::io::Write for CountingWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0 += buf.len();
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// Payload bytes of a turn, measured without allocating.
///
/// Counts every block variant plus the message's `name`/`tool_call_id`
/// annotations; tool-call arguments count as their serialized JSON length.
///
/// # Errors
///
/// [`VectorStoreError::Json`] when `serde_json` cannot serialize a tool call's
/// arguments (unreachable for a `Value`, but library code never unwraps).
fn entry_bytes(message: &UnifiedMessage) -> Result<usize, VectorStoreError> {
    let mut total = message.name.as_ref().map_or(0, String::len)
        + message.tool_call_id.as_ref().map_or(0, String::len);
    for block in &message.content {
        total += match block {
            MessageContentBlock::Text(text) => text.len(),
            MessageContentBlock::ImageBase64 { media_type, data } => media_type.len() + data.len(),
            MessageContentBlock::Thinking {
                reasoning,
                signature,
            } => reasoning.len() + signature.as_ref().map_or(0, String::len),
            MessageContentBlock::ToolCall {
                id,
                name,
                arguments,
            } => {
                let mut counter = CountingWriter(0);
                serde_json::to_writer(&mut counter, arguments).map_err(|e| {
                    VectorStoreError::Json(format!("could not measure tool call arguments: {e}"))
                })?;
                id.len() + name.len() + counter.0
            }
            MessageContentBlock::ToolResult {
                tool_call_id,
                output,
            } => tool_call_id.len() + output.len(),
        };
    }
    Ok(total)
}

/// Dot product of two equal-length slices, `wide::f32x8` with four
/// accumulators.
///
/// 32 lanes per iteration hides FMA latency. Nested `as_chunks` splits hand
/// the wide loop `&[[f32; 8]; 4]` groups and the narrow loop `&[f32; 8]`
/// groups, so every SIMD load goes through the array `From` impl: no bounds
/// checks, and never the zero-padding branch of `From<&[f32]>`. The horizontal
/// sum is a fixed-order tree, not `f32x8::reduce_add`, whose order is
/// documented as unspecified.
#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc0 = f32x8::splat(0.0);
    let mut acc1 = f32x8::splat(0.0);
    let mut acc2 = f32x8::splat(0.0);
    let mut acc3 = f32x8::splat(0.0);
    let (a8, a_tail) = a.as_chunks::<8>();
    let (b8, b_tail) = b.as_chunks::<8>();
    let (a32, a8_rest) = a8.as_chunks::<4>();
    let (b32, b8_rest) = b8.as_chunks::<4>();
    for (ca, cb) in a32.iter().zip(b32) {
        acc0 = f32x8::from(ca[0]).mul_add(f32x8::from(cb[0]), acc0);
        acc1 = f32x8::from(ca[1]).mul_add(f32x8::from(cb[1]), acc1);
        acc2 = f32x8::from(ca[2]).mul_add(f32x8::from(cb[2]), acc2);
        acc3 = f32x8::from(ca[3]).mul_add(f32x8::from(cb[3]), acc3);
    }
    let mut acc = (acc0 + acc1) + (acc2 + acc3);
    for (ca, cb) in a8_rest.iter().zip(b8_rest) {
        acc = f32x8::from(*ca).mul_add(f32x8::from(*cb), acc);
    }
    let lanes = acc.to_array();
    let mut sum = ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3]))
        + ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]));
    for (x, y) in a_tail.iter().zip(b_tail) {
        sum += x * y;
    }
    sum
}

/// Scale `v` to unit length in place; `Ok(false)` means a zero norm, and `v`
/// has been filled with zeros.
///
/// A squared norm that underflows to zero counts as a zero norm: the entry is
/// kept and flagged, and zeroing it is what makes "scores exactly `0.0`
/// against every query" true even for components whose squares underflow
/// (anything below roughly `3.7e-23`).
///
/// # Errors
///
/// [`VectorStoreError::Embedding`] when a component or the squared norm is not
/// finite.
fn normalize_in_place(v: &mut [f32]) -> Result<bool, VectorStoreError> {
    for (index, x) in v.iter().enumerate() {
        if !x.is_finite() {
            return Err(VectorStoreError::Embedding(format!(
                "embedding component {index} must be finite, got {x}"
            )));
        }
    }
    let square = dot(v, v);
    if !square.is_finite() {
        return Err(VectorStoreError::Embedding(format!(
            "embedding squared norm must be finite, got {square}"
        )));
    }
    if square == 0.0 {
        // Non-zero components can still square to zero, so store the zeros
        // the score contract promises rather than the original vector.
        v.fill(0.0);
        return Ok(false);
    }
    let inverse = 1.0 / square.sqrt();
    for x in v.iter_mut() {
        *x *= inverse;
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Maps a fixed word set to unit basis vectors and sums the matches, so
    /// every expected score is exact and immune to FMA rounding differences.
    struct FixedEmbedder {
        dimensions: usize,
        scale: f32,
    }

    impl FixedEmbedder {
        fn new(dimensions: usize) -> Arc<Self> {
            Arc::new(Self {
                dimensions,
                scale: 1.0,
            })
        }

        fn scaled(dimensions: usize, scale: f32) -> Arc<Self> {
            Arc::new(Self { dimensions, scale })
        }
    }

    impl Embedder for FixedEmbedder {
        fn embed(&self, text: &str) -> Result<Vec<f32>, PluginError> {
            let mut vector = vec![0.0f32; self.dimensions];
            for word in text.split_whitespace() {
                let axis = match word {
                    "alpha" => 0,
                    "beta" => 1,
                    "gamma" => 2,
                    "delta" => 3,
                    _ => continue,
                };
                if axis < self.dimensions {
                    vector[axis] += self.scale;
                }
            }
            Ok(vector)
        }
    }

    /// Hands back the same vector for every text; the shortest way to stage a
    /// wrong width or a non-finite component.
    struct CannedEmbedder(Vec<f32>);

    impl Embedder for CannedEmbedder {
        fn embed(&self, _text: &str) -> Result<Vec<f32>, PluginError> {
            Ok(self.0.clone())
        }
    }

    struct FailingEmbedder;

    impl Embedder for FailingEmbedder {
        fn embed(&self, _text: &str) -> Result<Vec<f32>, PluginError> {
            Err(PluginError::Internal("embedder down".to_string()))
        }
    }

    /// Records every text it was asked to embed.
    struct RecordingEmbedder {
        inner: FixedEmbedder,
        seen: Mutex<Vec<String>>,
    }

    impl RecordingEmbedder {
        fn new(dimensions: usize) -> Arc<Self> {
            Arc::new(Self {
                inner: FixedEmbedder {
                    dimensions,
                    scale: 1.0,
                },
                seen: Mutex::new(Vec::new()),
            })
        }
    }

    impl Embedder for RecordingEmbedder {
        fn embed(&self, text: &str) -> Result<Vec<f32>, PluginError> {
            self.seen.lock().unwrap().push(text.to_string());
            self.inner.embed(text)
        }
    }

    fn config(max_entries: usize, dimensions: usize) -> VectorStoreConfig {
        VectorStoreConfig::new(max_entries, dimensions, 64 * 1024).expect("config must build")
    }

    fn store_with(embedder: Arc<dyn Embedder>, max_entries: usize) -> InMemoryVectorStore {
        InMemoryVectorStore::new(config(max_entries, 4), embedder).expect("store must build")
    }

    fn store(max_entries: usize) -> InMemoryVectorStore {
        store_with(FixedEmbedder::new(4), max_entries)
    }

    fn user(text: &str) -> UnifiedMessage {
        UnifiedMessage::user(text)
    }

    fn text_of(message: &UnifiedMessage) -> String {
        text_projection(message)
    }

    /// Deterministic filler shared with the benchmark and the kernel test.
    fn sample(index: usize) -> f32 {
        ((index * 31 + 7) % 17) as f32 / 17.0 - 0.5
    }

    #[test]
    fn config_rejects_zero_max_entries_dimensions_and_entry_bytes() {
        for (entries, dimensions, bytes) in [(0, 8, 32), (4, 0, 32), (4, 8, 0)] {
            let error = VectorStoreConfig::new(entries, dimensions, bytes)
                .expect_err("a zero bound must be rejected");
            assert!(
                matches!(error, VectorStoreError::Config(_)),
                "expected Config, got {error:?}"
            );
        }
        assert!(VectorStoreConfig::new(1, 1, 1).is_ok());
    }

    #[test]
    fn config_rejects_an_arena_whose_float_or_byte_size_overflows() {
        for (entries, dimensions) in [(usize::MAX, 8), (usize::MAX / 2, 4), (usize::MAX / 2, 1)] {
            let error = VectorStoreConfig::new(entries, dimensions, 1)
                .expect_err("an arena that cannot be addressed must be rejected");
            assert!(
                matches!(error, VectorStoreError::Config(_)),
                "expected Config for {entries} x {dimensions}, got {error:?}"
            );
        }
        // The same bounds through a struct literal still reach the
        // constructor, where pre-sizing the arena would otherwise panic.
        let bypassed = VectorStoreConfig {
            max_entries: usize::MAX,
            dimensions: 8,
            max_entry_bytes: 1,
            warn_fraction: None,
        };
        assert!(matches!(
            InMemoryVectorStore::new(bypassed, FixedEmbedder::new(8)),
            Err(VectorStoreError::Config(_))
        ));
    }

    #[test]
    fn config_rejects_warn_fraction_outside_zero_to_one_and_non_finite() {
        let base = config(4, 4);
        for bad in [0.0, -0.1, 1.01, f32::NAN, f32::INFINITY] {
            let error = base
                .with_warn_fraction(bad)
                .expect_err("an out-of-range warn fraction must be rejected");
            match error {
                VectorStoreError::Config(message) => assert!(
                    message.starts_with("warn_fraction must be in (0.0, 1.0], got"),
                    "unexpected message: {message}"
                ),
                other => panic!("expected Config, got {other:?}"),
            }
        }
        assert_eq!(
            base.with_warn_fraction(1.0)
                .expect("1.0 is in range")
                .warn_fraction,
            Some(1.0)
        );
    }

    #[test]
    fn validate_rejects_a_struct_literal_config_that_bypassed_with_warn_fraction() {
        for bad in [5.0, f32::NAN] {
            let bypassed = VectorStoreConfig {
                max_entries: 4,
                dimensions: 4,
                max_entry_bytes: 64,
                warn_fraction: Some(bad),
            };
            assert!(
                matches!(
                    InMemoryVectorStore::new(bypassed, FixedEmbedder::new(4)),
                    Err(VectorStoreError::Config(_))
                ),
                "a struct literal must not smuggle warn_fraction {bad} past validation"
            );
        }
    }

    #[test]
    fn vector_store_errors_convert_into_the_plugin_error_contract() {
        let bad_config: PluginError = VectorStoreError::Config("bad".to_string()).into();
        assert!(matches!(bad_config, PluginError::Internal(_)));
        let poisoned: PluginError = VectorStoreError::Poisoned.into();
        assert!(matches!(poisoned, PluginError::Internal(_)));
        let embedding: PluginError = VectorStoreError::Embedding("narrow".to_string()).into();
        assert!(
            matches!(embedding, PluginError::Validation { schema, message }
                if schema == VALIDATION_SCHEMA && message == "narrow")
        );
        let oversized: PluginError = VectorStoreError::EntryTooLarge {
            bytes: 9,
            max_entry_bytes: 4,
        }
        .into();
        assert!(
            matches!(oversized, PluginError::Validation { ref message, .. }
                if message.contains("max_entry_bytes"))
        );
        // The caller's own failure is carried, not re-wrapped.
        let inner = PluginError::Internal("embedder down".to_string());
        assert_eq!(
            PluginError::from(VectorStoreError::Embedder(inner.clone())).to_string(),
            inner.to_string()
        );
    }

    #[test]
    fn text_projection_includes_text_tool_call_names_and_tool_result_outputs() {
        let message = UnifiedMessage {
            role: MessageRole::Assistant,
            content: vec![
                MessageContentBlock::Text("head".to_string()),
                MessageContentBlock::ToolCall {
                    id: "call-1".to_string(),
                    name: "read_file".to_string(),
                    arguments: serde_json::json!({ "path": "/tmp/x" }),
                },
                MessageContentBlock::ToolResult {
                    tool_call_id: "call-1".to_string(),
                    output: "tail".to_string(),
                },
            ],
            name: None,
            tool_call_id: None,
        };
        assert_eq!(text_of(&message), "head read_file tail");
    }

    #[test]
    fn text_projection_skips_thinking_and_image_blocks() {
        let message = UnifiedMessage {
            role: MessageRole::Assistant,
            content: vec![
                MessageContentBlock::Thinking {
                    reasoning: "secret reasoning".to_string(),
                    signature: Some("sig".to_string()),
                },
                MessageContentBlock::ImageBase64 {
                    media_type: "image/png".to_string(),
                    data: "AAAA".to_string(),
                },
                MessageContentBlock::Text("visible".to_string()),
            ],
            name: None,
            tool_call_id: None,
        };
        assert_eq!(text_of(&message), "visible");
    }

    #[test]
    fn entry_bytes_counts_text_image_thinking_tool_call_and_tool_result_payloads() {
        let arguments = serde_json::json!({ "path": "/tmp/x" });
        let serialized = serde_json::to_string(&arguments).expect("value must serialize");
        let message = UnifiedMessage {
            role: MessageRole::Tool,
            content: vec![
                MessageContentBlock::Text("abcd".to_string()),
                MessageContentBlock::ImageBase64 {
                    media_type: "image/png".to_string(),
                    data: "AAAA".to_string(),
                },
                MessageContentBlock::Thinking {
                    reasoning: "why".to_string(),
                    signature: Some("sg".to_string()),
                },
                MessageContentBlock::ToolCall {
                    id: "c1".to_string(),
                    name: "read_file".to_string(),
                    arguments,
                },
                MessageContentBlock::ToolResult {
                    tool_call_id: "c1".to_string(),
                    output: "out".to_string(),
                },
            ],
            name: Some("agent".to_string()),
            tool_call_id: Some("c1".to_string()),
        };
        let expected = "agent".len()
            + "c1".len()
            + "abcd".len()
            + "image/png".len()
            + "AAAA".len()
            + "why".len()
            + "sg".len()
            + "c1".len()
            + "read_file".len()
            + serialized.len()
            + "c1".len()
            + "out".len();
        assert_eq!(entry_bytes(&message).expect("measurable"), expected);
    }

    #[test]
    fn dot_kernel_matches_scalar_reference_across_remainder_widths() {
        for width in [1usize, 7, 8, 9, 31, 32, 33, 100, 384] {
            let a: Vec<f32> = (0..width).map(sample).collect();
            let b: Vec<f32> = (0..width).map(|i| sample(i + 5)).collect();
            let reference: f64 = a
                .iter()
                .zip(&b)
                .map(|(x, y)| f64::from(*x) * f64::from(*y))
                .sum();
            let actual = f64::from(dot(&a, &b));
            let tolerance = reference.abs().max(1.0) * 1e-4;
            assert!(
                (actual - reference).abs() <= tolerance,
                "width {width}: kernel {actual} vs reference {reference}"
            );
        }
    }

    #[test]
    fn normalization_makes_scaled_embeddings_score_identically() {
        let plain = store_with(FixedEmbedder::new(4), 4);
        let scaled = store_with(FixedEmbedder::scaled(4, 3.0), 4);
        for store in [&plain, &scaled] {
            store
                .store_turns("hint", &[user("alpha")])
                .expect("store must accept the turn");
        }
        let a = plain.retrieve("alpha", 1).expect("query must run");
        let b = scaled.retrieve("alpha", 1).expect("query must run");
        assert_eq!(a.turns[0].score, b.turns[0].score);
        assert!((a.turns[0].score - 1.0).abs() < 1e-6, "{a:?}");
    }

    #[test]
    fn store_turns_embeds_the_text_projection_of_every_turn() {
        let embedder = RecordingEmbedder::new(4);
        let store = store_with(Arc::clone(&embedder) as Arc<dyn Embedder>, 4);
        let turn = UnifiedMessage {
            role: MessageRole::Tool,
            content: vec![
                MessageContentBlock::Text("alpha".to_string()),
                MessageContentBlock::ToolResult {
                    tool_call_id: "c1".to_string(),
                    output: "beta".to_string(),
                },
            ],
            name: None,
            tool_call_id: None,
        };
        store
            .store_turns("hint", &[user("gamma"), turn])
            .expect("store must accept the batch");
        assert_eq!(
            *embedder.seen.lock().unwrap(),
            vec!["gamma".to_string(), "alpha beta".to_string()]
        );
    }

    #[test]
    fn store_turns_records_the_session_hint_on_every_entry() {
        let store = store(4);
        store
            .store_turns("session-a", &[user("alpha"), user("beta")])
            .expect("store must accept the batch");
        let report = store.retrieve("alpha beta", 2).expect("query must run");
        assert!(
            report
                .turns
                .iter()
                .all(|turn| turn.session_hint == "session-a"),
            "{report:?}"
        );
    }

    #[test]
    fn store_turns_rejects_an_embedding_of_the_wrong_width() {
        let store = store_with(Arc::new(CannedEmbedder(vec![0.0; 3])), 4);
        let error = store
            .store_turns("hint", &[user("alpha")])
            .expect_err("a 3-wide embedding must be rejected by a 4-wide store");
        assert!(
            matches!(error, PluginError::Validation { ref schema, .. } if schema == "vector-store"),
            "{error:?}"
        );
        assert_eq!(store.len().expect("len must read"), 0);
    }

    #[test]
    fn store_turns_rejects_an_embedding_with_a_non_finite_component() {
        let store = store_with(Arc::new(CannedEmbedder(vec![f32::NAN, 0.0, 0.0, 0.0])), 4);
        let error = store
            .store_turns("hint", &[user("alpha")])
            .expect_err("a NaN component must be rejected");
        assert!(
            matches!(error, PluginError::Validation { ref schema, .. } if schema == "vector-store"),
            "{error:?}"
        );
    }

    #[test]
    fn store_turns_rejects_a_turn_exceeding_max_entry_bytes() {
        let tight = VectorStoreConfig::new(4, 4, 8).expect("config must build");
        let store =
            InMemoryVectorStore::new(tight, FixedEmbedder::new(4)).expect("store must build");
        let error = store
            .store_turns("hint", &[user("alpha beta gamma delta")])
            .expect_err("an oversized turn must be rejected");
        match error {
            PluginError::Validation { schema, message } => {
                assert_eq!(schema, "vector-store");
                assert!(message.contains("max_entry_bytes"), "{message}");
            }
            other => panic!("expected Validation, got {other:?}"),
        }
        assert_eq!(store.len().expect("len must read"), 0);
    }

    #[test]
    fn store_turns_propagates_the_embedder_error_unchanged() {
        let store = store_with(Arc::new(FailingEmbedder), 4);
        let error = store
            .store_turns("hint", &[user("alpha")])
            .expect_err("the embedder failure must surface");
        match error {
            PluginError::Internal(message) => assert_eq!(message, "embedder down"),
            other => panic!("expected the embedder's own error, got {other:?}"),
        }
    }

    #[test]
    fn store_turns_stores_nothing_when_any_turn_in_the_batch_is_rejected() {
        let tight = VectorStoreConfig::new(8, 4, 12).expect("config must build");
        let store =
            InMemoryVectorStore::new(tight, FixedEmbedder::new(4)).expect("store must build");
        store
            .store_turns("hint", &[user("alpha")])
            .expect("the small turn must be accepted");
        let error = store
            .store_turns(
                "hint",
                &[user("beta"), user("alpha beta gamma delta epsilon")],
            )
            .expect_err("the batch must be rejected");
        assert!(matches!(error, PluginError::Validation { .. }), "{error:?}");
        assert_eq!(
            store.len().expect("len must read"),
            1,
            "the accepted first batch must be the only content"
        );
    }

    #[test]
    fn store_evicts_oldest_first_at_capacity_and_counts_the_eviction() {
        let store = store(2);
        store
            .store_turns("hint", &[user("alpha"), user("beta")])
            .expect("store must accept the batch");
        store
            .store_turns("hint", &[user("gamma")])
            .expect("store must accept the turn");
        let usage = store.usage().expect("usage must read");
        assert_eq!(usage.entries, 2);
        assert_eq!(usage.evicted_entries, 1);
        let report = store.retrieve("alpha", 2).expect("query must run");
        assert_eq!(report.scanned, 2);
        assert!(
            report
                .turns
                .iter()
                .all(|turn| text_of(&turn.message) != "alpha"),
            "the oldest turn must be gone: {report:?}"
        );
    }

    #[test]
    fn a_batch_larger_than_capacity_keeps_the_newest_entries_and_counts_the_rest_evicted() {
        let store = store(2);
        store
            .store_turns(
                "hint",
                &[user("alpha"), user("beta"), user("gamma"), user("delta")],
            )
            .expect("store must accept the batch");
        let usage = store.usage().expect("usage must read");
        assert_eq!(usage.entries, 2);
        assert_eq!(usage.evicted_entries, 2);
        let report = store.retrieve("gamma delta", 2).expect("query must run");
        let kept: Vec<String> = report
            .turns
            .iter()
            .map(|turn| text_of(&turn.message))
            .collect();
        assert!(kept.contains(&"gamma".to_string()) && kept.contains(&"delta".to_string()));
    }

    #[test]
    fn a_batch_larger_than_capacity_never_embeds_the_dropped_head() {
        let embedder = RecordingEmbedder::new(4);
        let store = store_with(Arc::clone(&embedder) as Arc<dyn Embedder>, 2);
        store
            .store_turns(
                "hint",
                &[user("alpha"), user("beta"), user("gamma"), user("delta")],
            )
            .expect("store must accept the batch");
        assert_eq!(
            *embedder
                .seen
                .lock()
                .expect("the recorder must not be poisoned"),
            vec!["gamma".to_string(), "delta".to_string()],
            "only the surviving tail may reach the embedder"
        );
        let usage = store.usage().expect("usage must read");
        assert_eq!((usage.entries, usage.evicted_entries), (2, 2));
    }

    #[test]
    fn evicted_slots_are_reused_without_growing_the_arena() {
        let store = store(3);
        // The arena is allocated to its cap in `new`, so any reallocation
        // would move it: the address is the cheapest proof none happened.
        let address = store
            .state
            .read()
            .expect("state must lock")
            .arena
            .as_ptr()
            .addr();
        for word in ["alpha", "beta", "gamma", "delta", "alpha", "beta", "gamma"] {
            store
                .store_turns("hint", &[user(word)])
                .expect("store must accept the turn");
        }
        let state = store.state.read().expect("state must lock");
        assert_eq!(state.arena.len(), 3 * 4, "the arena must stay at the cap");
        assert_eq!(
            state.arena.as_ptr().addr(),
            address,
            "the arena must never reallocate"
        );
        assert_eq!(state.entries.len(), 3);
    }

    #[test]
    fn usage_reports_entries_capacity_fraction_and_flips_near_cap_at_warn_fraction() {
        let config = VectorStoreConfig::new(4, 4, 4096)
            .expect("config must build")
            .with_warn_fraction(0.75)
            .expect("warn fraction must be accepted");
        let store =
            InMemoryVectorStore::new(config, FixedEmbedder::new(4)).expect("store must build");
        store
            .store_turns("hint", &[user("alpha"), user("beta")])
            .expect("store must accept the batch");
        let usage = store.usage().expect("usage must read");
        assert_eq!((usage.entries, usage.capacity), (2, 4));
        assert!((usage.fraction - 0.5).abs() < f32::EPSILON);
        assert!(!usage.near_cap);

        store
            .store_turns("hint", &[user("gamma")])
            .expect("store must accept the turn");
        assert!(store.usage().expect("usage must read").near_cap);
    }

    #[test]
    fn usage_counts_zero_norm_entries_and_retrieve_scores_them_zero() {
        let store = store(4);
        store
            .store_turns("hint", &[user("unknown words only"), user("alpha")])
            .expect("store must accept the batch");
        assert_eq!(store.usage().expect("usage must read").zero_norm_entries, 1);
        let report = store.retrieve("alpha", 2).expect("query must run");
        assert_eq!(report.turns[0].score, 1.0);
        assert_eq!(report.turns[1].score, 0.0);
    }

    #[test]
    fn an_underflowing_norm_is_stored_as_zeros_and_scores_exactly_zero() {
        // Every component squares to 1e-50, which underflows f32 to zero, so
        // the entry has a zero norm despite non-zero components.
        let underflowing = store_with(Arc::new(CannedEmbedder(vec![1e-25; 4])), 4);
        underflowing
            .store_turns("hint", &[user("alpha")])
            .expect("store must accept the turn");
        assert_eq!(
            underflowing
                .usage()
                .expect("usage must read")
                .zero_norm_entries,
            1
        );
        let report = underflowing
            .retrieve_embedding(None, &[1.0, 0.0, 0.0, 0.0], 1)
            .expect("query must run");
        assert_eq!(
            report.turns[0].score, 0.0,
            "a zeroed entry must score exactly 0.0: {report:?}"
        );

        // The same underflow on the query side zeroes every score.
        let live = store(4);
        live.store_turns("hint", &[user("alpha"), user("beta")])
            .expect("store must accept the batch");
        let report = live
            .retrieve_embedding(None, &[1e-25; 4], 2)
            .expect("query must run");
        assert!(
            report.turns.iter().all(|turn| turn.score == 0.0),
            "{report:?}"
        );
    }

    #[test]
    fn retrieve_ranks_hits_by_cosine_similarity_descending() {
        let store = store(4);
        store
            .store_turns(
                "hint",
                &[user("alpha alpha"), user("beta"), user("alpha beta")],
            )
            .expect("store must accept the batch");
        let report = store.retrieve("alpha", 3).expect("query must run");
        assert_eq!(report.scanned, 3);
        let ranked: Vec<String> = report
            .turns
            .iter()
            .map(|turn| text_of(&turn.message))
            .collect();
        assert_eq!(
            ranked,
            vec![
                "alpha alpha".to_string(),
                "alpha beta".to_string(),
                "beta".to_string()
            ]
        );
        assert_eq!(report.turns[0].score, 1.0);
        assert_eq!(report.turns[2].score, 0.0);
    }

    #[test]
    fn retrieve_breaks_equal_scores_by_newest_sequence() {
        let store = store(4);
        store
            .store_turns("hint", &[user("alpha"), user("alpha")])
            .expect("store must accept the batch");
        let report = store.retrieve("alpha", 2).expect("query must run");
        assert_eq!(report.turns[0].score, report.turns[1].score);
        assert!(
            report.turns[0].sequence > report.turns[1].sequence,
            "the newer entry must rank first: {report:?}"
        );
    }

    #[test]
    fn retrieve_truncates_to_k_and_reports_the_scanned_count() {
        let store = store(8);
        store
            .store_turns(
                "hint",
                &[user("alpha"), user("beta"), user("gamma"), user("delta")],
            )
            .expect("store must accept the batch");
        let report = store.retrieve("alpha beta", 2).expect("query must run");
        assert_eq!(report.turns.len(), 2);
        assert_eq!(report.scanned, 4);
    }

    #[test]
    fn retrieve_rejects_zero_k() {
        let store = store(4);
        let error = store
            .retrieve("alpha", 0)
            .expect_err("a zero k must be rejected");
        assert!(matches!(error, VectorStoreError::Config(_)), "{error:?}");
        assert!(matches!(
            store.retrieve_embedding(None, &[1.0, 0.0, 0.0, 0.0], 0),
            Err(VectorStoreError::Config(_))
        ));
    }

    #[test]
    fn retrieve_embedding_rejects_a_wrong_width_or_non_finite_query() {
        let store = store(4);
        assert!(matches!(
            store.retrieve_embedding(None, &[1.0, 0.0], 1),
            Err(VectorStoreError::Embedding(_))
        ));
        assert!(matches!(
            store.retrieve_embedding(None, &[f32::INFINITY, 0.0, 0.0, 0.0], 1),
            Err(VectorStoreError::Embedding(_))
        ));
    }

    #[test]
    fn retrieve_in_scans_only_the_named_session_hint() {
        let store = store(8);
        store
            .store_turns("session-a", &[user("alpha")])
            .expect("store must accept the turn");
        store
            .store_turns("session-b", &[user("alpha")])
            .expect("store must accept the turn");
        let scoped = store
            .retrieve_in("session-b", "alpha", 8)
            .expect("query must run");
        assert_eq!(scoped.scanned, 1);
        assert_eq!(scoped.turns.len(), 1);
        assert_eq!(scoped.turns[0].session_hint, "session-b");
        assert_eq!(
            store.retrieve("alpha", 8).expect("query must run").scanned,
            2
        );
        assert_eq!(
            store
                .retrieve_in("absent", "alpha", 8)
                .expect("query must run")
                .turns
                .len(),
            0
        );
    }

    #[test]
    fn retrieve_on_an_empty_store_returns_an_empty_report_not_an_error() {
        let store = store(4);
        let report = store.retrieve("alpha", 4).expect("query must run");
        assert_eq!(report.scanned, 0);
        assert!(report.turns.is_empty());
        assert!(store.is_empty().expect("is_empty must read"));
        assert_eq!(store.capacity(), 4);
    }

    #[test]
    fn a_zero_norm_query_scores_every_entry_zero_and_ranks_by_recency() {
        let store = store(4);
        store
            .store_turns("hint", &[user("alpha"), user("beta"), user("gamma")])
            .expect("store must accept the batch");
        let report = store
            .retrieve("no known words here", 3)
            .expect("query must run");
        assert!(
            report.turns.iter().all(|turn| turn.score == 0.0),
            "{report:?}"
        );
        let sequences: Vec<u64> = report.turns.iter().map(|turn| turn.sequence).collect();
        assert_eq!(sequences, vec![2, 1, 0]);
    }

    #[test]
    fn render_is_byte_deterministic_and_none_for_an_empty_report() {
        let store = store(4);
        store
            .store_turns("hint", &[user("alpha"), user("beta")])
            .expect("store must accept the batch");
        let report = store.retrieve("alpha", 2).expect("query must run");
        let rendered = report.render().expect("a non-empty report must render");
        assert_eq!(
            rendered,
            "CUCA recall: 2 offloaded turn(s), best first\n1. [1.0000] alpha\n2. [0.0000] beta"
        );
        assert_eq!(report.render(), Some(rendered));
        assert_eq!(
            RetrievalReport {
                turns: Vec::new(),
                scanned: 0
            }
            .render(),
            None
        );
    }

    #[test]
    fn inject_inserts_before_the_most_recent_user_message_and_appends_without_one() {
        let store = store(4);
        store
            .store_turns("hint", &[user("alpha")])
            .expect("store must accept the turn");
        let report = store.retrieve("alpha", 1).expect("query must run");

        let mut request = UnifiedRequest::new("m")
            .add_system_message("instructions")
            .add_user_message("first")
            .add_user_message("latest");
        assert_eq!(report.inject(&mut request), RecallInjection::Inserted);
        assert!(is_recall_message(&request.messages[2]));
        assert_eq!(request.messages[3].role, MessageRole::User);
        assert_eq!(text_of(&request.messages[3]), "latest");

        let mut without_user = UnifiedRequest::new("m").add_system_message("instructions");
        assert_eq!(report.inject(&mut without_user), RecallInjection::Inserted);
        assert!(is_recall_message(
            without_user.messages.last().expect("a message must exist")
        ));
    }

    #[test]
    fn inject_replaces_an_existing_recall_message_and_reports_replaced() {
        let store = store(4);
        store
            .store_turns("hint", &[user("alpha"), user("beta")])
            .expect("store must accept the batch");
        let first = store.retrieve("alpha", 1).expect("query must run");
        let second = store.retrieve("beta", 1).expect("query must run");

        let mut request = UnifiedRequest::new("m").add_user_message("question");
        assert_eq!(first.inject(&mut request), RecallInjection::Inserted);
        assert_eq!(second.inject(&mut request), RecallInjection::Replaced);
        assert_eq!(
            request
                .messages
                .iter()
                .filter(|m| is_recall_message(m))
                .count(),
            1
        );
        assert_eq!(request.messages.len(), 2);
        assert!(text_of(&request.messages[0]).contains("beta"));
    }

    #[test]
    fn inject_removes_or_reports_absent_for_an_empty_report() {
        let empty = RetrievalReport {
            turns: Vec::new(),
            scanned: 0,
        };
        let mut request = UnifiedRequest::new("m").add_user_message("question");
        assert_eq!(empty.inject(&mut request), RecallInjection::Absent);
        assert_eq!(request.messages.len(), 1);

        let store = store(4);
        store
            .store_turns("hint", &[user("alpha")])
            .expect("store must accept the turn");
        let filled = store.retrieve("alpha", 1).expect("query must run");
        assert_eq!(filled.inject(&mut request), RecallInjection::Inserted);
        assert_eq!(empty.inject(&mut request), RecallInjection::Removed);
        assert_eq!(request.messages.len(), 1);
    }

    #[test]
    fn a_poisoned_state_lock_surfaces_an_error_from_every_accessor() {
        let store = Arc::new(store(4));
        let poisoned = Arc::new(AtomicUsize::new(0));
        let handle = std::thread::spawn({
            let store = Arc::clone(&store);
            let poisoned = Arc::clone(&poisoned);
            move || {
                let _guard = store.state.write().expect("first lock must succeed");
                poisoned.fetch_add(1, Ordering::SeqCst);
                panic!("poison the state lock on purpose");
            }
        });
        assert!(handle.join().is_err(), "the helper thread must panic");
        assert_eq!(poisoned.load(Ordering::SeqCst), 1);

        // A poisoned lock refuses on both halves rather than reporting an
        // empty store.
        assert!(matches!(store.usage(), Err(VectorStoreError::Poisoned)));
        assert!(matches!(store.len(), Err(VectorStoreError::Poisoned)));
        assert!(matches!(store.is_empty(), Err(VectorStoreError::Poisoned)));
        assert!(matches!(
            store.retrieve_embedding(None, &[1.0, 0.0, 0.0, 0.0], 1),
            Err(VectorStoreError::Poisoned)
        ));
        // The seam keeps the plugin contract: `Poisoned` converts to
        // `PluginError::Internal`.
        assert!(matches!(
            store.store_turns("hint", &[user("alpha")]),
            Err(PluginError::Internal(_))
        ));
    }
}
