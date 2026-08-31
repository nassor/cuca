//! Context-window memory management via a configurable compaction menu.
//!
//! [`MemoryPlugin`] monitors the request prompt with fast tiktoken-rs token
//! counters and, when a configured size trigger fires, runs an ordered pipeline
//! of compaction strategies: cheap zero-LLM passes first (deduplicate file
//! reads, clear old tool results, clamp oversized parts, sliding window), then
//! a summarization tier, with escalation that re-measures tokens after each
//! pass and stops once under budget. A strategy whose extension seam is absent
//! no-ops, and a strategy error is recorded and falls through to the next one.
//!
//! # Synchronous hooks
//!
//! [`CucaPlugin::on_request`] is synchronous: no `await` anywhere in the plugin
//! hooks, so all in-hook compaction is zero-LLM. The [`Summarizer`] and
//! [`VectorStore`] traits are caller-supplied synchronous bridges, and
//! [`MemoryPlugin::compress`] is the public out-of-band entry point
//! (application-driven "compact now"), callable directly without a request.
//!
//! # Size triggers and budget resolution
//!
//! Compression is triggered by one of three size knobs on [`MemoryConfig`]:
//! [`MemoryConfig::max_messages`] (message count), [`MemoryConfig::max_tokens`]
//! (absolute token count), or [`MemoryConfig::max_fraction`] (fraction of the
//! resolved context window). `max_messages` takes precedence when combined with
//! a token budget; `max_tokens` and `max_fraction` are mutually exclusive
//! (rejected at construction). The window is resolved per model via
//! [`ContextWindowResolver`] when present, else the configured
//! [`MemoryConfig::context_window_tokens`] fallback. `on_request` also hands a
//! [`ContextUsage`] reading to every [`ContextUsageObserver`] on each request
//! and can inject a one-shot near-limit warning when usage crosses
//! [`MemoryConfig::warn_fraction`] (idempotent via a marker prefix).
//!
//! # Token counting
//!
//! [`MemoryPlugin::count_tokens`] is deliberately an approximation: it sums the
//! tiktoken-rs token counts of `"<role> <text content>"` for every message,
//! where text content is the concatenation of the message's `Text` blocks. It
//! ignores image/tool metadata and does not model the provider's per-message
//! framing tokens (e.g. the `<|im_start|>` delimiter count tiktoken-rs applies
//! for chat completions), and it excludes `max_tokens`/`temperature`. The same
//! approximation feeds the `max_tokens`/`max_fraction` triggers, the warning
//! check, and the observer readings. [`CompactionStrategy::ClampOversizedMessages`]
//! is the exception: it counts each individual block (Text, ToolResult output,
//! Thinking reasoning, stringified ToolCall arguments) with its own
//! `encode_ordinary` call, so a pass that only clamps non-Text blocks may not
//! move `tokens_before`/`tokens_after`.
//!
//! # Pairing safety
//!
//! Tool results must not outlive their calls: providers reject a result whose
//! call was removed. Every strategy that removes messages
//! ([`CompactionStrategy::Offload`], [`CompactionStrategy::Summarize`],
//! [`CompactionStrategy::SlidingWindow`], [`CompactionStrategy::DropTurns`])
//! pairing-closes its drop set first: for every dropped `ToolCall` id, the
//! message carrying its `ToolResult` (a `Tool`-role message with that
//! `tool_call_id`, or any message holding a `ToolResult` block with that id) is
//! added to the drop set too. Results always follow their calls, so this is the
//! only extension direction needed. The most recent user message is never added
//! (the never-remove invariant outranks the closure), so a result that rides
//! inside it survives with its message.
//!
//! # Extension seams
//!
//! The [`Summarizer`], [`VectorStore`], [`ContextWindowResolver`], and
//! [`ContextUsageObserver`] traits are the integration points for real
//! summarization models, vector/embedding stores, window registries, and
//! reporting gauges; tests ship fake implementations. `MemoryPlugin` holds them
//! as `Option<Arc<dyn …>>` / `Vec<Arc<dyn …>>` so a plain
//! [`MemoryPlugin::new`] works with no extensions and degrades to the drop-only
//! tail of the menu.
//!
//! # Graph memory
//!
//! The plugin also owns an in-memory working graph ([`MemoryGraph`], in
//! [`graph`]) as explicit, machine-readable long-term context: nodes with
//! labels/properties and directed weighted relationships. When
//! [`MemoryConfig::graph_context`] is set, `on_request` renders the graph via
//! [`MemoryGraph::render`] into a system message placed right after the first
//! System message. Injection is idempotent: the message starts with the
//! [`graph::GRAPH_RENDER_MARKER`] prefix, so a later `on_request` replaces it
//! in place instead of appending, and the message is removed once the graph
//! becomes empty. The step runs after compression, so compaction strategies
//! (which may drop System messages after the first) cannot delete it, and the
//! graph message's bounded token cost is not counted against the request that
//! triggered compression. Graph access is `Mutex`-guarded and merging happens
//! under a single lock hold.
//!
//! The working graph is the one piece of plugin state with no internal cap:
//! it grows only through explicit caller calls
//! ([`MemoryPlugin::merge_graph`], [`MemoryPlugin::replace_graph`],
//! [`MemoryPlugin::replace_snapshot`]), never from a hook, so the caller
//! owns its bound and decides what to drop. Per-request cost stays bounded
//! independently: [`GraphContextConfig::max_nodes`] and
//! [`GraphContextConfig::max_relationships`] cap what
//! [`MemoryGraph::render`] injects, and [`MemoryGraph::len`] plus
//! [`MemoryGraph::relationship_count`] are the O(1) size readings.

use std::collections::HashSet;
use std::sync::{Arc, Mutex};

use crate::error::PluginError;
use crate::plugin::CucaPlugin;
use crate::request::UnifiedRequest;
use crate::types::{MessageContentBlock, MessageRole, UnifiedMessage};

pub mod graph;
pub use graph::{
    GraphDirection, GraphNode, GraphRelationship, GraphSnapshot, MemoryGraph, MergePolicy,
    MergeReport,
};

use graph::GRAPH_RENDER_MARKER;

/// Fixed session hint used when offloading turns to a [`VectorStore`].
///
/// The hint is opaque storage context (namespace/collection key) rather than a
/// real session id; CUCA's session model is separate and the plugin does not
/// participate in it, so a stable constant keeps repeated offloads colocated
/// without plumbing an id through the plugin.
const SESSION_HINT: &str = "cuca-memory";

/// Prefix that makes the near-limit warning injection idempotent: `on_request`
/// scans for it and never injects a second warning while one is present.
const WARNING_MARKER: &str = "CUCA context warning:";

/// One configurable compaction step. Order in [`MemoryConfig::strategies`] is
/// execution order; a strategy whose extension seam is absent no-ops.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompactionStrategy {
    /// Move oldest turns to the [`VectorStore`] (no-op without a store).
    Offload { turns: usize },
    /// Replace oldest turns with one summary [`UnifiedMessage`] (no-op without
    /// a [`Summarizer`]).
    Summarize { turns: usize },
    /// Blank the output of file-read tool results superseded by a newer read
    /// of the same file.
    DeduplicateFileReads { tool_name: String },
    /// Blank old `ToolResult` outputs, keeping the last `keep_pairs` pairs.
    ClearToolResults { keep_pairs: usize },
    /// Head/tail-truncate any single block whose token count exceeds
    /// `max_part_tokens`.
    ClampOversizedMessages { max_part_tokens: u32 },
    /// Drop the oldest whole messages down to a tail of `keep_messages`.
    SlidingWindow { keep_messages: usize },
    /// Drop redundant System messages after the first, up to
    /// [`MemoryConfig::max_drop_system_observations`].
    DropObservations,
    /// Drop the oldest eligible turns (pairing-closed), keeping the most
    /// recent user message.
    DropTurns,
}

/// Resolve a model id to its context window in tokens. Returning `None` defers
/// to [`MemoryConfig::context_window_tokens`].
pub trait ContextWindowResolver: Send + Sync {
    /// The context window for `model`, or `None` to use the configured fallback.
    fn resolve_window(&self, model: &str) -> Option<u32>;
}

/// One context-usage reading handed to observers on every request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContextUsage {
    /// Token count of the request messages (same approximation as
    /// [`MemoryPlugin::count_tokens`]).
    pub used_tokens: u32,
    /// Context window in tokens: the resolver hit, or the configured fallback.
    pub window_tokens: u32,
    /// `false` when `window_tokens` is the configured fallback, not a resolver
    /// hit.
    pub resolved: bool,
}

/// Observes usage without editing history (reporting/UI gauge seam).
pub trait ContextUsageObserver: Send + Sync {
    /// Handed the usage reading of every request; an `Err` aborts the request.
    fn observe(&self, usage: &ContextUsage) -> Result<(), PluginError>;
}

/// Configuration for rendering the working graph into requests as context.
///
/// When [`MemoryConfig::graph_context`] is `Some`, `on_request` renders the
/// graph via [`MemoryGraph::render`] into a system message (see the module
/// docs' "Graph memory" section).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct GraphContextConfig {
    /// Maximum nodes rendered into the context message per request.
    pub max_nodes: usize,
    /// Maximum relationships rendered into the context message per request.
    pub max_relationships: usize,
}

impl Default for GraphContextConfig {
    /// 64 nodes / 128 relationships: small enough to inject into every
    /// request, large enough for a useful neighborhood.
    fn default() -> Self {
        Self {
            max_nodes: 64,
            max_relationships: 128,
        }
    }
}

/// Configuration for the context-memory plugin.
pub struct MemoryConfig {
    /// tiktoken-rs encoder name, e.g. `"cl100k_base"` (a base encoder name) or
    /// a model name like `"gpt-4o"` that maps to a tokenizer.
    pub encoder_name: String,
    /// Fallback/override context window in tokens; used when no resolver hit
    /// exists and for the `max_fraction`/`warn_fraction` math.
    pub context_window_tokens: u32,
    /// Optional per-model context-window resolver; `None` defers to
    /// [`Self::context_window_tokens`].
    pub context_window_resolver: Option<Arc<dyn ContextWindowResolver>>,
    /// Message-count trigger: compress when the message list is longer than
    /// this. Takes precedence over token budgets when combined.
    pub max_messages: Option<usize>,
    /// Absolute token-count trigger: compress when the used token count
    /// reaches this. Mutually exclusive with [`Self::max_fraction`].
    pub max_tokens: Option<u32>,
    /// Window-fraction trigger: compress when usage reaches this fraction of
    /// the resolved window.
    pub max_fraction: Option<f32>,
    /// Near-limit warning trigger: inject a one-shot warning system message
    /// when usage reaches this fraction of the resolved window. `None` disables.
    pub warn_fraction: Option<f32>,
    /// Observers handed a [`ContextUsage`] reading on every request.
    pub observers: Vec<Arc<dyn ContextUsageObserver>>,
    /// Ordered compaction pipeline; execution order, escalation order.
    pub strategies: Vec<CompactionStrategy>,
    /// How many oldest turns [`CompactionStrategy::DropTurns`] removes per pass.
    pub offload_turns: usize,
    /// How many redundant System messages after the first
    /// [`CompactionStrategy::DropObservations`] drops.
    pub max_drop_system_observations: usize,
    /// Optional graph-context injection: when `Some`, `on_request` renders
    /// the working graph into a system message placed right after the first
    /// System message (idempotent, replaced in place, removed when the graph
    /// is empty); `None` disables. See the module docs' "Graph memory"
    /// section.
    pub graph_context: Option<GraphContextConfig>,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            encoder_name: "cl100k_base".to_string(),
            context_window_tokens: 128_000,
            context_window_resolver: None,
            max_messages: None,
            max_tokens: None,
            max_fraction: Some(0.8),
            warn_fraction: None,
            observers: Vec::new(),
            strategies: vec![
                CompactionStrategy::Offload { turns: 10 },
                CompactionStrategy::Summarize { turns: 10 },
                CompactionStrategy::DeduplicateFileReads {
                    tool_name: "read_file".to_string(),
                },
                CompactionStrategy::ClearToolResults { keep_pairs: 3 },
                CompactionStrategy::ClampOversizedMessages {
                    max_part_tokens: 4096,
                },
                CompactionStrategy::SlidingWindow { keep_messages: 40 },
                CompactionStrategy::DropObservations,
                CompactionStrategy::DropTurns,
            ],
            offload_turns: 10,
            max_drop_system_observations: 3,
            graph_context: None,
        }
    }
}

impl MemoryConfig {
    /// Reject trigger combinations that cannot be honored.
    fn validate(&self) -> Result<(), PluginError> {
        if self.max_tokens.is_some() && self.max_fraction.is_some() {
            return Err(PluginError::Internal(
                "max_tokens and max_fraction are mutually exclusive".to_string(),
            ));
        }
        Ok(())
    }
}

/// Extension seam: semantic summarization.
///
/// The caller supplies a backend that condenses a group of turns into a single
/// summary string, which the plugin substitutes for the group when no vector
/// store is present.
pub trait Summarizer: Send + Sync {
    /// Condense `turns` (oldest first) into a single summary string.
    fn summarize(&self, turns: &[UnifiedMessage]) -> String;
}

/// Extension seam: offload storage.
///
/// The caller supplies a backend (e.g. a vector store) that persists historical
/// turns removed from the live prompt so they are not lost.
pub trait VectorStore: Send + Sync {
    /// Persist `turns` under `session_hint`; an `Err` aborts the offload.
    fn store_turns(&self, session_hint: &str, turns: &[UnifiedMessage]) -> Result<(), PluginError>;
}

/// The active compression trigger for a model, resolved from [`MemoryConfig`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Budget {
    /// Message-count cap: compress when the message list is longer than `n`.
    Messages(usize),
    /// Token cap: compress when the used token count reaches `n`.
    Tokens(u32),
    /// No trigger: out-of-band [`MemoryPlugin::compress`] runs regardless.
    Unlimited,
}

/// The outcome of a [`MemoryPlugin::compress`] pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionAction {
    /// Oldest turns were moved to the [`VectorStore`].
    Offloaded,
    /// Oldest turns were replaced by a single summary user message.
    Summarized,
    /// Superseded file-read tool results had their outputs blanked.
    DeduplicatedFileReads,
    /// Old `ToolResult` outputs were blanked, keeping the newest pairs.
    ClearedToolResults,
    /// Oversized content blocks were head/tail-truncated in place.
    ClampedParts,
    /// The message list was trimmed to a sliding-window tail.
    Slid,
    /// Redundant system observations were dropped.
    DroppedObservations,
    /// The oldest user/assistant turns were dropped.
    DroppedTurns,
}

/// Result of a [`MemoryPlugin::compress`] pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompressionReport {
    /// Token count of the message list before compression.
    pub tokens_before: u32,
    /// Token count of the message list after compression.
    pub tokens_after: u32,
    /// Which strategy actions ran, in pipeline order. Empty = no-op pass.
    pub actions: Vec<CompressionAction>,
    /// `Display` text of the last strategy error, if any; compression falls
    /// back to the next strategy rather than aborting.
    pub last_error: Option<String>,
}

/// Counts accepted by a successful
/// [`MemoryPlugin::replace_snapshot`] import.
///
/// A graph import is a wholesale replacement, so these are the complete node
/// and relationship counts of the graph now live in the plugin.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct GraphImportReport {
    /// Nodes in the imported graph.
    pub nodes: usize,
    /// Relationships in the imported graph.
    pub relationships: usize,
}

/// Context-compression / working-memory plugin.
///
/// Holds a shared tiktoken encoder behind a `Mutex` (tiktoken's `CoreBPE` is
/// `Send + Sync` in 0.6, but the mutex serializes access so the counter can be
/// shared safely across `await` points) plus optional extension backends, the
/// configured strategy menu, and an in-memory working graph behind a `Mutex`
/// that `on_request` can render into requests when configured.
pub struct MemoryPlugin {
    config: MemoryConfig,
    summarizer: Option<Arc<dyn Summarizer>>,
    store: Option<Arc<dyn VectorStore>>,
    encoder: Mutex<tiktoken_rs::CoreBPE>,
    /// In-memory working graph; empty by default. `Mutex`-guarded so the
    /// plugin stays `Send + Sync` and graph mutations serialize; `on_request`
    /// renders it when [`MemoryConfig::graph_context`] is set.
    graph: Mutex<MemoryGraph>,
}

impl MemoryPlugin {
    /// Build a plugin with no extensions (drop-only compaction tail) and the
    /// encoder named in `config.encoder_name`.
    ///
    /// # Errors
    ///
    /// Returns [`PluginError::Internal`] when the config is invalid (both
    /// `max_tokens` and `max_fraction` set) or the encoder cannot be loaded.
    pub fn new(config: MemoryConfig) -> Result<Self, PluginError> {
        config.validate()?;
        let encoder = crate::tokenize::load_encoder(&config.encoder_name)?;
        Ok(Self {
            config,
            summarizer: None,
            store: None,
            encoder: Mutex::new(encoder),
            graph: Mutex::new(MemoryGraph::new()),
        })
    }

    /// Build a plugin wired to a summarizer and a vector store.
    ///
    /// # Errors
    ///
    /// Same validation and encoder-loading failures as [`Self::new`].
    pub fn with_extensions(
        config: MemoryConfig,
        summarizer: Arc<dyn Summarizer>,
        store: Arc<dyn VectorStore>,
    ) -> Result<Self, PluginError> {
        config.validate()?;
        let encoder = crate::tokenize::load_encoder(&config.encoder_name)?;
        Ok(Self {
            config,
            summarizer: Some(summarizer),
            store: Some(store),
            encoder: Mutex::new(encoder),
            graph: Mutex::new(MemoryGraph::new()),
        })
    }

    /// Approximate token count of a message list.
    ///
    /// Sums the tiktoken-rs (`encode_ordinary`) counts of `"<role> <text>"` per
    /// message; see the [module docs](crate::plugins::memory) for the
    /// approximation caveats. Returns `Err` if the encoder lock is poisoned.
    ///
    /// # Errors
    ///
    /// Returns [`PluginError::Internal`] when the tiktoken encoder lock is
    /// poisoned.
    pub fn count_tokens(&self, messages: &[UnifiedMessage]) -> Result<u32, PluginError> {
        let encoder = self
            .encoder
            .lock()
            .map_err(|e| PluginError::Internal(format!("tiktoken encoder lock poisoned: {e}")))?;
        let mut total = 0usize;
        for msg in messages {
            let text = joined_text(msg);
            // One allocation per message.
            let serialized = format!("{} {}", role_label(msg.role), text);
            total += encoder.encode_ordinary(&serialized).len();
        }
        Ok(total as u32)
    }

    /// Resolve the context window for `model`: the resolver's hit, else the
    /// configured fallback. Returns `(window, resolved)` where `resolved` is
    /// false when the fallback was used.
    pub fn resolved_window(&self, model: &str) -> (u32, bool) {
        if let Some(resolver) = &self.config.context_window_resolver
            && let Some(window) = resolver.resolve_window(model)
        {
            return (window, true);
        }
        (self.config.context_window_tokens, false)
    }

    /// The compression trigger budget for `model`.
    ///
    /// Precedence: [`MemoryConfig::max_messages`] first (a message-count cap is
    /// a hard structural limit and the cheaper check, so it intentionally wins
    /// over a token budget when both are set), then
    /// [`MemoryConfig::max_tokens`], then [`MemoryConfig::max_fraction`]
    /// applied to the resolved window, else [`Budget::Unlimited`].
    pub fn budget(&self, model: &str) -> Budget {
        if let Some(n) = self.config.max_messages {
            return Budget::Messages(n);
        }
        if let Some(n) = self.config.max_tokens {
            return Budget::Tokens(n);
        }
        if let Some(f) = self.config.max_fraction {
            let (window, _) = self.resolved_window(model);
            return Budget::Tokens((window as f32 * f) as u32);
        }
        Budget::Unlimited
    }

    pub fn over_budget(&self, messages: &[UnifiedMessage], usage: u32, budget: &Budget) -> bool {
        match budget {
            Budget::Messages(n) => messages.len() > *n,
            Budget::Tokens(n) => usage >= *n,
            Budget::Unlimited => false,
        }
    }

    /// Locked access to the working graph.
    ///
    /// # Errors
    ///
    /// Returns [`PluginError::Internal`] when the graph lock is poisoned.
    pub fn graph(&self) -> Result<std::sync::MutexGuard<'_, MemoryGraph>, PluginError> {
        self.graph
            .lock()
            .map_err(|e| PluginError::Internal(format!("memory graph lock poisoned: {e}")))
    }

    /// Merge `other` into the working graph under a single lock hold.
    ///
    /// The graph-core merge moves nodes and relationships out of `other` (no
    /// clones), pre-reserves capacity, and resolves relationship-id collisions
    /// by deterministic renaming; the merge never drops data. The `Mutex` is
    /// held for the whole merge, so concurrent requests observe either the
    /// pre- or post-merge graph, never a partial one.
    ///
    /// # Errors
    ///
    /// Returns [`PluginError::Internal`] when the graph lock is poisoned.
    pub fn merge_graph(
        &self,
        other: MemoryGraph,
        policy: MergePolicy,
    ) -> Result<MergeReport, PluginError> {
        self.graph().map(|mut guard| guard.merge(other, policy))
    }

    /// Replace the working graph wholesale under a single lock hold.
    ///
    /// # Errors
    ///
    /// Returns [`PluginError::Internal`] when the graph lock is poisoned.
    pub fn replace_graph(&self, graph: MemoryGraph) -> Result<(), PluginError> {
        let mut guard = self.graph()?;
        *guard = graph;
        Ok(())
    }

    /// Export the complete working graph as a deterministic
    /// [`GraphSnapshot`].
    ///
    /// The lock is held for the full clone-and-sort ([`MemoryGraph::snapshot`]
    /// sorts nodes and relationships while called under this method's guard);
    /// there is no unlocked window before the sorted snapshot is returned.
    /// The live graph is not modified.
    ///
    /// **Sensitive full-fidelity export:** `cuca-export` intentionally
    /// includes the complete memory graph and local-cache request/response
    /// values. It may contain confidential system prompts, user messages,
    /// tool arguments and results, base64 image data, model output,
    /// signatures, and graph properties. Treat the JSON as sensitive data; do
    /// not log or publish it. CUCA does not encrypt, redact, or write it. The
    /// caller owns access control, encryption, storage, and deletion.
    ///
    /// # Errors
    ///
    /// Returns [`PluginError::Internal`] when the graph lock is poisoned.
    pub fn snapshot(&self) -> Result<GraphSnapshot, PluginError> {
        self.graph().map(|guard| guard.snapshot())
    }

    /// Import `snapshot` as the working graph: validate first, then swap.
    ///
    /// The snapshot is validated and reconstructed by
    /// [`MemoryGraph::from_snapshot`] *before* the live replacement lock is
    /// acquired, so a rejected snapshot never touches the working graph and a
    /// concurrent request never observes a partially imported graph. The
    /// import is a wholesale replacement, not a merge (see
    /// [`Self::merge_graph`] for merging).
    ///
    /// # Errors
    ///
    /// Returns [`PluginError::Internal`] when the snapshot is invalid
    /// (duplicate node or relationship id, non-finite relationship weight, or
    /// a relationship endpoint absent from the snapshot's nodes) or when the
    /// graph lock is poisoned.
    pub fn replace_snapshot(
        &self,
        snapshot: GraphSnapshot,
    ) -> Result<GraphImportReport, PluginError> {
        let staged = Self::stage_snapshot(snapshot)?;
        self.commit_staged_graph(staged)
    }

    /// Validate `snapshot` into a staged graph without locking or mutating
    /// anything.
    ///
    /// Staging seam for the combined export coordinator: it validates every
    /// component before any component commits, and the staged graph exposes
    /// only [`MemoryGraph`]'s public API, never the plugin's or the graph's
    /// private collections.
    ///
    /// # Errors
    ///
    /// Returns [`PluginError::Internal`] when the snapshot is invalid.
    pub(crate) fn stage_snapshot(snapshot: GraphSnapshot) -> Result<MemoryGraph, PluginError> {
        MemoryGraph::from_snapshot(snapshot)
    }

    /// Commit an already staged graph under a single lock hold and report the
    /// imported counts.
    ///
    /// Commit seam for the combined export coordinator: staging can fail
    /// freely, but this step performs exactly one [`Self::replace_graph`] and
    /// cannot reject the graph. Counts are read before the swap so the lock is
    /// held only for the replacement itself.
    ///
    /// # Errors
    ///
    /// Returns [`PluginError::Internal`] when the graph lock is poisoned.
    pub(crate) fn commit_staged_graph(
        &self,
        staged: MemoryGraph,
    ) -> Result<GraphImportReport, PluginError> {
        let report = GraphImportReport {
            nodes: staged.len(),
            relationships: staged.relationship_count(),
        };
        self.replace_graph(staged)?;
        Ok(report)
    }

    /// Compress `messages` in place by running the configured strategy pipeline.
    ///
    /// Public out-of-band entry point: no budget is known here, so the whole
    /// [`MemoryConfig::strategies`] list runs exactly once. The in-request path
    /// ([`CucaPlugin::on_request`]) uses the budgeted [`Self::compress_inner`]
    /// instead, which stops early once the trigger no longer fires and
    /// escalates across passes. The most recent user message and the first
    /// System message survive every strategy.
    ///
    /// # Errors
    ///
    /// Returns [`PluginError::Internal`] when the tiktoken encoder lock is
    /// poisoned; per-strategy failures are recorded in [`CompressionReport::last_error`]
    /// and do not abort the pipeline.
    pub fn compress(
        &self,
        messages: &mut Vec<UnifiedMessage>,
    ) -> Result<CompressionReport, PluginError> {
        self.compress_inner(messages, &Budget::Unlimited)
    }

    /// Tiered pipeline body shared by [`Self::compress`] and `on_request`.
    ///
    /// With [`Budget::Unlimited`] (out-of-band) one full pass runs. With a real
    /// budget the pass loop re-measures tokens after each pass and stops once
    /// under budget, when a pass made no progress, or after 4 passes.
    fn compress_inner(
        &self,
        messages: &mut Vec<UnifiedMessage>,
        budget: &Budget,
    ) -> Result<CompressionReport, PluginError> {
        let tokens_before = self.count_tokens(messages)?;
        let mut actions: Vec<CompressionAction> = Vec::new();
        let mut last_error: Option<String> = None;
        let limited = !matches!(budget, Budget::Unlimited);
        let mut pass = 0usize;
        loop {
            pass += 1;
            let before_len = messages.len();
            let pass_actions = self.run_pass(messages, budget, &mut last_error)?;
            let progress = !pass_actions.is_empty() || messages.len() != before_len;
            actions.extend(pass_actions);
            if !limited {
                break;
            }
            let used = self.count_tokens(messages)?;
            if !self.over_budget(messages, used, budget) {
                break;
            }
            if pass >= 4 || !progress {
                break;
            }
        }
        let tokens_after = self.count_tokens(messages)?;
        Ok(CompressionReport {
            tokens_before,
            tokens_after,
            actions,
            last_error,
        })
    }

    /// One pass over the strategy menu in configured order.
    ///
    /// A strategy whose seam is absent no-ops; an `Err` is recorded as
    /// `last_error` and the pass continues (fallback semantics). After each
    /// strategy that acted, a limited budget re-measures and breaks the pass
    /// early once the trigger no longer fires.
    fn run_pass(
        &self,
        messages: &mut Vec<UnifiedMessage>,
        budget: &Budget,
        last_error: &mut Option<String>,
    ) -> Result<Vec<CompressionAction>, PluginError> {
        let mut actions = Vec::new();
        for strategy in &self.config.strategies {
            match self.run_strategy(strategy, messages) {
                Ok(Some(action)) => {
                    actions.push(action);
                    if !matches!(budget, Budget::Unlimited) {
                        let used = self.count_tokens(messages)?;
                        if !self.over_budget(messages, used, budget) {
                            break;
                        }
                    }
                }
                Ok(None) => {}
                Err(e) => {
                    *last_error = Some(e.to_string());
                }
            }
        }
        Ok(actions)
    }

    /// Dispatch one strategy; `Ok(None)` means the strategy no-op'd.
    fn run_strategy(
        &self,
        strategy: &CompactionStrategy,
        messages: &mut Vec<UnifiedMessage>,
    ) -> Result<Option<CompressionAction>, PluginError> {
        match strategy {
            CompactionStrategy::Offload { turns } => self.strategy_offload(*turns, messages),
            CompactionStrategy::Summarize { turns } => self.strategy_summarize(*turns, messages),
            CompactionStrategy::DeduplicateFileReads { tool_name } => {
                self.strategy_dedup_file_reads(tool_name, messages)
            }
            CompactionStrategy::ClearToolResults { keep_pairs } => {
                self.strategy_clear_tool_results(*keep_pairs, messages)
            }
            CompactionStrategy::ClampOversizedMessages { max_part_tokens } => {
                self.strategy_clamp_oversized(*max_part_tokens, messages)
            }
            CompactionStrategy::SlidingWindow { keep_messages } => {
                self.strategy_sliding_window(*keep_messages, messages)
            }
            CompactionStrategy::DropObservations => self.strategy_drop_observations(messages),
            CompactionStrategy::DropTurns => self.strategy_drop_turns(messages),
        }
    }

    /// Move the oldest removable turns to the vector store and remove them.
    fn strategy_offload(
        &self,
        turns: usize,
        messages: &mut Vec<UnifiedMessage>,
    ) -> Result<Option<CompressionAction>, PluginError> {
        let store = match &self.store {
            Some(store) => store,
            None => return Ok(None),
        };
        let idxs = removable_turn_indices(messages, turns);
        if idxs.is_empty() {
            return Ok(None);
        }
        let drop_set: HashSet<usize> = idxs.into_iter().collect();
        let closed = pairing_close(messages, &drop_set);
        let mut closed_idxs: Vec<usize> = closed.into_iter().collect();
        closed_idxs.sort_unstable();
        // Take the closed turns by value: `Vec::remove` already yields the
        // owned message, so cloning first would allocate every offloaded
        // message twice. Back-to-front removal keeps the ascending indices
        // valid; the reverse restores the oldest-first order the store wants.
        let mut turns: Vec<UnifiedMessage> = closed_idxs
            .iter()
            .rev()
            .map(|&i| messages.remove(i))
            .collect();
        turns.reverse();
        if let Err(error) = store.store_turns(SESSION_HINT, &turns) {
            // Offload stays all-or-nothing: a failed store must not drop
            // history, so the turns go back at their original indices
            // (ascending inserts restore every position exactly).
            for (&i, turn) in closed_idxs.iter().zip(turns) {
                messages.insert(i, turn);
            }
            return Err(error);
        }
        Ok(Some(CompressionAction::Offloaded))
    }

    /// Replace the oldest removable turns with one summary user message.
    fn strategy_summarize(
        &self,
        turns: usize,
        messages: &mut Vec<UnifiedMessage>,
    ) -> Result<Option<CompressionAction>, PluginError> {
        let summarizer = match &self.summarizer {
            Some(summarizer) => summarizer,
            None => return Ok(None),
        };
        let idxs = removable_turn_indices(messages, turns);
        if idxs.is_empty() {
            return Ok(None);
        }
        let drop_set: HashSet<usize> = idxs.into_iter().collect();
        let closed = pairing_close(messages, &drop_set);
        let mut closed_idxs: Vec<usize> = closed.into_iter().collect();
        closed_idxs.sort_unstable();
        // Same ownership transfer as `strategy_offload`: the summarizer only
        // borrows the turns, and they are being removed anyway, so cloning
        // them would allocate each message twice. `summarize` is infallible,
        // so no rollback is needed here.
        let mut turns: Vec<UnifiedMessage> = closed_idxs
            .iter()
            .rev()
            .map(|&i| messages.remove(i))
            .collect();
        turns.reverse();
        let summary = summarizer.summarize(&turns);
        messages.insert(closed_idxs[0], UnifiedMessage::user(summary));
        Ok(Some(CompressionAction::Summarized))
    }

    /// Blank the outputs of file-read results superseded by a newer read.
    fn strategy_dedup_file_reads(
        &self,
        tool_name: &str,
        messages: &mut [UnifiedMessage],
    ) -> Result<Option<CompressionAction>, PluginError> {
        let mut seen: HashSet<(String, String)> = HashSet::new();
        let mut blanked = false;
        // Newest to oldest: the first (newest) read of a path records it; older
        // reads of a path read again later have their superseded output blanked.
        // A read without a result block present is skipped (nothing to supersede).
        for i in (0..messages.len()).rev() {
            let calls: Vec<(String, String)> = messages[i]
                .content
                .iter()
                .filter_map(|block| match block {
                    MessageContentBlock::ToolCall {
                        id,
                        name,
                        arguments,
                    } if name == tool_name => arguments
                        .get("path")
                        .and_then(|v| v.as_str())
                        .map(|path| (id.clone(), path.to_string())),
                    _ => None,
                })
                .collect();
            for (call_id, path) in calls {
                let key = (tool_name.to_string(), path);
                if let Some(location) = find_result_location(messages, &call_id)
                    && !seen.insert(key)
                {
                    blanked |= blank_result(messages, location);
                }
            }
        }
        Ok(if blanked {
            Some(CompressionAction::DeduplicatedFileReads)
        } else {
            None
        })
    }

    /// Blank old tool-result outputs, keeping the newest `keep_pairs` pairs.
    fn strategy_clear_tool_results(
        &self,
        keep_pairs: usize,
        messages: &mut [UnifiedMessage],
    ) -> Result<Option<CompressionAction>, PluginError> {
        let units = collect_result_units(messages);
        if units.len() <= keep_pairs {
            return Ok(None);
        }
        let keep_from = units.len() - keep_pairs;
        let mut blanked = false;
        for &(msg_idx, block_idx) in units.iter().take(keep_from) {
            blanked |= blank_result(messages, (msg_idx, block_idx));
        }
        Ok(if blanked {
            Some(CompressionAction::ClearedToolResults)
        } else {
            None
        })
    }

    /// Head/tail-truncate any single block whose token count exceeds the cap.
    fn strategy_clamp_oversized(
        &self,
        max_part_tokens: u32,
        messages: &mut [UnifiedMessage],
    ) -> Result<Option<CompressionAction>, PluginError> {
        let encoder = self
            .encoder
            .lock()
            .map_err(|e| PluginError::Internal(format!("tiktoken encoder lock poisoned: {e}")))?;
        let mut clamped = false;
        for msg in messages.iter_mut() {
            for block in &mut msg.content {
                match block {
                    MessageContentBlock::Text(t) => {
                        if encoder.encode_ordinary(t).len() as u32 > max_part_tokens {
                            *t = clamp_oversized(t);
                            clamped = true;
                        }
                    }
                    MessageContentBlock::ToolResult { output, .. } => {
                        if encoder.encode_ordinary(output).len() as u32 > max_part_tokens {
                            *output = clamp_oversized(output);
                            clamped = true;
                        }
                    }
                    MessageContentBlock::Thinking { reasoning, .. } => {
                        if encoder.encode_ordinary(reasoning).len() as u32 > max_part_tokens {
                            *reasoning = clamp_oversized(reasoning);
                            clamped = true;
                        }
                    }
                    MessageContentBlock::ToolCall { arguments, .. } => {
                        // Only stringified JSON arguments; object-valued
                        // arguments are small by construction and untouched.
                        if let serde_json::Value::String(s) = arguments
                            && encoder.encode_ordinary(s).len() as u32 > max_part_tokens
                        {
                            *s = clamp_oversized(s);
                            clamped = true;
                        }
                    }
                    MessageContentBlock::ImageBase64 { .. } => {}
                }
            }
        }
        Ok(if clamped {
            Some(CompressionAction::ClampedParts)
        } else {
            None
        })
    }

    /// Trim the message list to a sliding-window tail.
    fn strategy_sliding_window(
        &self,
        keep_messages: usize,
        messages: &mut Vec<UnifiedMessage>,
    ) -> Result<Option<CompressionAction>, PluginError> {
        if messages.len() <= keep_messages {
            return Ok(None);
        }
        let drop_count = messages.len() - keep_messages;
        let first_system = messages.iter().position(|m| m.role == MessageRole::System);
        let recent_user = messages.iter().rposition(|m| m.role == MessageRole::User);
        let mut drop_set: HashSet<usize> = HashSet::new();
        for i in 0..messages.len() {
            if drop_set.len() >= drop_count {
                break;
            }
            if Some(i) == first_system || Some(i) == recent_user {
                continue;
            }
            drop_set.insert(i);
        }
        let closed = pairing_close(messages, &drop_set);
        let mut closed_idxs: Vec<usize> = closed.into_iter().collect();
        closed_idxs.sort_unstable();
        remove_indices(messages, &closed_idxs);
        Ok(Some(CompressionAction::Slid))
    }

    /// Drop redundant System messages after the first, keeping the primary.
    fn strategy_drop_observations(
        &self,
        messages: &mut Vec<UnifiedMessage>,
    ) -> Result<Option<CompressionAction>, PluginError> {
        let mut dropped = 0usize;
        let mut seen_primary = false;
        let mut to_remove = Vec::new();
        for (i, msg) in messages.iter().enumerate() {
            if msg.role != MessageRole::System {
                continue;
            }
            if !seen_primary {
                // First system message is the primary instruction; always kept.
                seen_primary = true;
                continue;
            }
            if dropped >= self.config.max_drop_system_observations {
                break;
            }
            to_remove.push(i);
            dropped += 1;
        }
        if to_remove.is_empty() {
            return Ok(None);
        }
        remove_indices(messages, &to_remove);
        Ok(Some(CompressionAction::DroppedObservations))
    }

    /// Drop the oldest eligible turns, keeping the most recent user message.
    fn strategy_drop_turns(
        &self,
        messages: &mut Vec<UnifiedMessage>,
    ) -> Result<Option<CompressionAction>, PluginError> {
        let idxs = removable_turn_indices(messages, self.config.offload_turns);
        if idxs.is_empty() {
            return Ok(None);
        }
        let drop_set: HashSet<usize> = idxs.into_iter().collect();
        let closed = pairing_close(messages, &drop_set);
        let mut closed_idxs: Vec<usize> = closed.into_iter().collect();
        closed_idxs.sort_unstable();
        remove_indices(messages, &closed_idxs);
        Ok(Some(CompressionAction::DroppedTurns))
    }
}

impl CucaPlugin for MemoryPlugin {
    fn name(&self) -> &'static str {
        "context-memory"
    }

    fn on_request(&self, req: &mut UnifiedRequest) -> Result<(), PluginError> {
        let used = self.count_tokens(&req.messages)?;
        let (window, resolved) = self.resolved_window(&req.model);
        let usage = ContextUsage {
            used_tokens: used,
            window_tokens: window,
            resolved,
        };
        for observer in &self.config.observers {
            observer.observe(&usage)?;
        }
        // The marker scan avoids `joined_text`: the marker is injected as a
        // single `Text` block, so scanning blocks in place skips the
        // `Vec<&str>` plus joined `String` that `joined_text` would allocate
        // for every message on every request.
        if let Some(warn_fraction) = self.config.warn_fraction
            && (used as f32 / window as f32) >= warn_fraction
            && !req.messages.iter().any(|m| {
                m.content.iter().any(
                    |block| matches!(block, MessageContentBlock::Text(text) if text.contains(WARNING_MARKER)),
                )
            })
        {
            let percent = used as f32 / window as f32 * 100.0;
            req.messages.push(UnifiedMessage::system(format!(
                "{WARNING_MARKER} The conversation is at {percent:.0}% of the \
                 {window}-token context window; wrap up soon.",
            )));
        }
        let budget = self.budget(&req.model);
        if self.over_budget(&req.messages, used, &budget) {
            self.compress_inner(&mut req.messages, &budget)?;
        }
        // Graph context, after compression: see the module docs' "Graph
        // memory" section.
        if let Some(cfg) = &self.config.graph_context {
            let render = {
                let guard = self.graph()?;
                if guard.is_empty() {
                    None
                } else {
                    Some(guard.render(cfg.max_nodes, cfg.max_relationships))
                }
            };
            let existing = req.messages.iter().position(|m| {
                m.content
                    .iter()
                    .find_map(|block| match block {
                        MessageContentBlock::Text(text) => Some(text.as_str()),
                        _ => None,
                    })
                    .is_some_and(|text| text.starts_with(GRAPH_RENDER_MARKER))
            });
            match (render, existing) {
                (Some(text), Some(pos)) => {
                    req.messages[pos] = UnifiedMessage::system(text);
                }
                (Some(text), None) => {
                    let insert_at = req
                        .messages
                        .iter()
                        .position(|m| m.role == MessageRole::System)
                        .map_or(0, |i| i + 1);
                    req.messages.insert(insert_at, UnifiedMessage::system(text));
                }
                (None, Some(pos)) => {
                    req.messages.remove(pos);
                }
                (None, None) => {}
            }
        }
        Ok(())
    }
}

/// Human-readable label for a message role, used in the token serialization.
fn role_label(role: MessageRole) -> &'static str {
    match role {
        MessageRole::System => "system",
        MessageRole::User => "user",
        MessageRole::Assistant => "assistant",
        MessageRole::Tool => "tool",
    }
}

/// Concatenate the `Text` content blocks of a message.
///
/// Image and tool metadata are excluded from the estimate to keep the counter
/// fast; this is a documented approximation.
fn joined_text(msg: &UnifiedMessage) -> String {
    let mut parts = Vec::new();
    for block in &msg.content {
        if let MessageContentBlock::Text(t) = block {
            parts.push(t.as_str());
        }
    }
    parts.join(" ")
}

/// Indices (ascending) of the oldest removable turn-like messages.
///
/// A removable turn is any non-`System` message other than the most recent
/// `User` message (which the plugin must never remove). Capped at `cap` from the
/// front of the list, so the returned indices are the oldest eligible turns.
fn removable_turn_indices(messages: &[UnifiedMessage], cap: usize) -> Vec<usize> {
    let recent_user = messages.iter().rposition(|m| m.role == MessageRole::User);
    let mut out = Vec::new();
    for (i, msg) in messages.iter().enumerate() {
        if out.len() >= cap {
            break;
        }
        if msg.role == MessageRole::System {
            continue;
        }
        if Some(i) == recent_user {
            continue;
        }
        out.push(i);
    }
    out
}

/// Remove messages by ascending index; removing in reverse keeps earlier
/// indices valid.
fn remove_indices(messages: &mut Vec<UnifiedMessage>, idxs: &[usize]) {
    for &i in idxs.iter().rev() {
        messages.remove(i);
    }
}

/// Pairing-closure of a drop set: add every message that carries the result of
/// a dropped tool call.
///
/// Dropping a call while keeping its result orphans the result, which providers
/// reject. Results always follow their calls in the conversation, so only this
/// one extension direction is possible. The most recent user message is never
/// added: the never-remove invariant outranks the closure, so a result that
/// rides inside it survives with its message.
fn pairing_close(messages: &[UnifiedMessage], drop_set: &HashSet<usize>) -> HashSet<usize> {
    let mut call_ids: HashSet<String> = HashSet::new();
    for &i in drop_set {
        for block in &messages[i].content {
            if let MessageContentBlock::ToolCall { id, .. } = block {
                call_ids.insert(id.clone());
            }
        }
    }
    if call_ids.is_empty() {
        return drop_set.clone();
    }
    let recent_user = messages.iter().rposition(|m| m.role == MessageRole::User);
    let mut closed = drop_set.clone();
    for (i, msg) in messages.iter().enumerate() {
        if closed.contains(&i) || Some(i) == recent_user {
            continue;
        }
        let carries_result = (msg.role == MessageRole::Tool
            && msg
                .tool_call_id
                .as_deref()
                .is_some_and(|id| call_ids.contains(id)))
            || msg.content.iter().any(|block| {
                matches!(
                    block,
                    MessageContentBlock::ToolResult { tool_call_id, .. }
                        if call_ids.contains(tool_call_id.as_str())
                )
            });
        if carries_result {
            closed.insert(i);
        }
    }
    closed
}

/// One blankable tool result in message order: `(message index, block index)`.
/// `block` is `None` for a `Tool`-role message whose output lives in its `Text`
/// blocks; `Some(j)` is a `ToolResult` block at `content[j]` (may ride in any
/// message).
fn collect_result_units(messages: &[UnifiedMessage]) -> Vec<(usize, Option<usize>)> {
    let mut units = Vec::new();
    for (i, msg) in messages.iter().enumerate() {
        if msg.role == MessageRole::Tool && msg.tool_call_id.is_some() {
            units.push((i, None));
        }
        for (j, block) in msg.content.iter().enumerate() {
            if matches!(block, MessageContentBlock::ToolResult { .. }) {
                units.push((i, Some(j)));
            }
        }
    }
    units
}

/// Locate the output of a tool result for `call_id` as `(message, block)`;
/// `block` is `None` when the output lives in a `Tool`-role message's `Text`
/// blocks.
fn find_result_location(
    messages: &[UnifiedMessage],
    call_id: &str,
) -> Option<(usize, Option<usize>)> {
    for (i, msg) in messages.iter().enumerate() {
        if msg.role == MessageRole::Tool && msg.tool_call_id.as_deref() == Some(call_id) {
            if msg
                .content
                .iter()
                .any(|b| matches!(b, MessageContentBlock::Text(_)))
            {
                return Some((i, None));
            }
            return None;
        }
        for (j, block) in msg.content.iter().enumerate() {
            if let MessageContentBlock::ToolResult { tool_call_id, .. } = block
                && tool_call_id.as_str() == call_id
            {
                return Some((i, Some(j)));
            }
        }
    }
    None
}

/// Blank a located result output in place; returns whether anything was
/// blanked. `block` is `None` for a `Tool`-role message (its `Text` blocks
/// carry the output).
fn blank_result(messages: &mut [UnifiedMessage], location: (usize, Option<usize>)) -> bool {
    let (i, block_idx) = location;
    match block_idx {
        Some(j) => {
            if let MessageContentBlock::ToolResult { output, .. } = &mut messages[i].content[j] {
                output.clear();
                true
            } else {
                false
            }
        }
        None => {
            let mut blanked = false;
            for block in &mut messages[i].content {
                if let MessageContentBlock::Text(t) = block {
                    t.clear();
                    blanked = true;
                }
            }
            blanked
        }
    }
}

/// Head/tail-truncate an oversized string: keep roughly half the characters
/// total (a quarter from each end), joined by a truncation marker. Counting
/// characters, not tokens, so the cut never splits a UTF-8 scalar.
fn clamp_oversized(s: &str) -> String {
    const MARKER: &str = "\n…[truncated]…\n";
    let total = s.chars().count();
    // Guard against the marker dominating a degenerate input; token counts far
    // above any real cap guarantee far more than 32 characters in practice.
    if total <= MARKER.chars().count() * 2 {
        return MARKER.to_string();
    }
    let keep_each_side = total / 4;
    let head: String = s.chars().take(keep_each_side).collect();
    let tail: String = s.chars().skip(total - keep_each_side).collect();
    format!("{head}{MARKER}{tail}")
}

#[cfg(all(test, feature = "plugin-memory"))]
mod tests {
    use super::*;
    use serde_json::json;

    /// Fake vector store that records everything handed to it.
    struct FakeStore(Mutex<Vec<UnifiedMessage>>);

    impl VectorStore for FakeStore {
        fn store_turns(
            &self,
            session_hint: &str,
            turns: &[UnifiedMessage],
        ) -> Result<(), PluginError> {
            assert_eq!(session_hint, SESSION_HINT);
            self.0.lock().unwrap().extend_from_slice(turns);
            Ok(())
        }
    }

    /// Fake vector store whose `store_turns` always fails (fallback testing).
    struct FailingStore;

    impl VectorStore for FailingStore {
        fn store_turns(
            &self,
            _session_hint: &str,
            _turns: &[UnifiedMessage],
        ) -> Result<(), PluginError> {
            Err(PluginError::Internal("store failure".to_string()))
        }
    }

    /// Fake summarizer that always emits a fixed marker string.
    struct FakeSummarizer;

    impl Summarizer for FakeSummarizer {
        fn summarize(&self, _turns: &[UnifiedMessage]) -> String {
            "SUMMARY".to_string()
        }
    }

    /// Fake observer that records every usage reading.
    struct FakeObserver(Mutex<Vec<ContextUsage>>);

    impl ContextUsageObserver for FakeObserver {
        fn observe(&self, usage: &ContextUsage) -> Result<(), PluginError> {
            self.0.lock().unwrap().push(*usage);
            Ok(())
        }
    }

    /// Fake resolver with a hit for one model name.
    struct FakeResolver;

    impl ContextWindowResolver for FakeResolver {
        fn resolve_window(&self, model: &str) -> Option<u32> {
            if model == "big-model" {
                Some(1_000_000)
            } else {
                None
            }
        }
    }

    /// Fake resolver with a fixed tiny window for every model.
    struct TinyResolver;

    impl ContextWindowResolver for TinyResolver {
        fn resolve_window(&self, _model: &str) -> Option<u32> {
            Some(100)
        }
    }

    /// Assistant message whose content is a single tool call.
    fn assistant_with_call(id: &str, name: &str, arguments: serde_json::Value) -> UnifiedMessage {
        UnifiedMessage {
            role: MessageRole::Assistant,
            content: vec![MessageContentBlock::ToolCall {
                id: id.to_string(),
                name: name.to_string(),
                arguments,
            }],
            name: None,
            tool_call_id: None,
        }
    }

    /// Tool-role message carrying a tool result (OpenAI wire style).
    fn tool_result_message(call_id: &str, output: &str) -> UnifiedMessage {
        UnifiedMessage {
            role: MessageRole::Tool,
            content: vec![MessageContentBlock::Text(output.to_string())],
            name: None,
            tool_call_id: Some(call_id.to_string()),
        }
    }

    /// User message carrying a tool-result block (Anthropic wire style).
    fn user_with_tool_result(call_id: &str, output: &str) -> UnifiedMessage {
        UnifiedMessage {
            role: MessageRole::User,
            content: vec![MessageContentBlock::ToolResult {
                tool_call_id: call_id.to_string(),
                output: output.to_string(),
            }],
            name: None,
            tool_call_id: None,
        }
    }

    #[test]
    fn count_tokens_counts_and_grows() {
        let plugin = MemoryPlugin::new(MemoryConfig::default()).unwrap();
        let short = vec![UnifiedMessage::user("Hello, world")];
        let n = plugin.count_tokens(&short).unwrap();
        assert!(
            n > 0,
            "a non-empty message must count more than zero tokens"
        );

        let longer = vec![UnifiedMessage::user(
            "Hello, world. This is a considerably longer piece of text that should push the \
             token count well above the short message.",
        )];
        assert!(
            plugin.count_tokens(&longer).unwrap() > n,
            "adding text must grow the token count"
        );

        // Deterministic for identical input.
        assert_eq!(plugin.count_tokens(&short).unwrap(), n);
    }

    #[test]
    fn on_request_compresses_above_threshold_and_untouched_below() {
        let hot = MemoryPlugin::new(MemoryConfig {
            context_window_tokens: 100,
            max_fraction: Some(0.8),
            ..Default::default()
        })
        .unwrap();
        let mut req = UnifiedRequest::new("gpt-4o");
        req.messages
            .push(UnifiedMessage::system("You are a helpful assistant."));
        for i in 0..8 {
            req.messages.push(UnifiedMessage::user(format!(
                "Turn number {i}: several full sentences of conversational content that consume \
                 tokens and drive the total well past the eighty-token threshold."
            )));
        }
        let before = req.messages.len();
        hot.on_request(&mut req).unwrap();
        assert!(
            req.messages.len() < before,
            "at/over threshold, on_request must compress the messages"
        );

        // Below threshold: nothing changes.
        let cool = MemoryPlugin::new(MemoryConfig {
            context_window_tokens: 10_000,
            max_fraction: Some(0.8),
            ..Default::default()
        })
        .unwrap();
        let mut req = UnifiedRequest::new("gpt-4o");
        req.messages.push(UnifiedMessage::user("hi"));
        let snapshot = req.messages.clone();
        cool.on_request(&mut req).unwrap();
        assert_eq!(
            req.messages, snapshot,
            "below threshold, messages are untouched"
        );
    }

    #[test]
    fn compress_offloads_oldest_turns_to_store() {
        let store = Arc::new(FakeStore(Mutex::new(Vec::new())));
        let plugin = MemoryPlugin::with_extensions(
            MemoryConfig::default(),
            Arc::new(FakeSummarizer),
            store.clone(),
        )
        .unwrap();
        let mut messages = vec![
            UnifiedMessage::system("system prompt"),
            UnifiedMessage::user("oldest user turn"),
            UnifiedMessage::assistant("oldest assistant turn"),
            UnifiedMessage::user("most recent user"),
        ];
        let n = messages.len();
        let report = plugin.compress(&mut messages).unwrap();
        assert!(report.actions.contains(&CompressionAction::Offloaded));
        // The two oldest non-system turns (excluding the most recent user) were
        // removed from the live list and recorded by the store.
        assert_eq!(messages.len(), n - 2);
        assert_eq!(messages.last().unwrap().role, MessageRole::User);
        let stored = store.0.lock().unwrap();
        assert_eq!(stored.len(), 2);
        assert_eq!(stored[0].role, MessageRole::User);
        assert_eq!(stored[1].role, MessageRole::Assistant);
    }

    #[test]
    fn compress_summarizes_turn_group_into_one_user_message() {
        // A summarize-only menu isolates the strategy's effect: the summary
        // replaces the turn group and survives the pass.
        let plugin = MemoryPlugin {
            config: MemoryConfig {
                strategies: vec![CompactionStrategy::Summarize { turns: 10 }],
                ..Default::default()
            },
            summarizer: Some(Arc::new(FakeSummarizer)),
            store: None,
            encoder: Mutex::new(crate::tokenize::load_encoder("cl100k_base").unwrap()),
            graph: Mutex::new(MemoryGraph::new()),
        };
        let mut messages = vec![
            UnifiedMessage::system("system prompt"),
            UnifiedMessage::user("oldest user turn"),
            UnifiedMessage::assistant("oldest assistant turn"),
            UnifiedMessage::user("most recent user"),
        ];
        let n = messages.len();
        let report = plugin.compress(&mut messages).unwrap();
        assert!(report.actions.contains(&CompressionAction::Summarized));
        // Two-turn group replaced by a single user message: n - 2 + 1 = n - 1.
        assert_eq!(messages.len(), n - 1);
        let summary_msg = &messages[1];
        assert_eq!(summary_msg.role, MessageRole::User);
        assert_eq!(
            joined_text(summary_msg),
            "SUMMARY",
            "the summary text replaces the turn group"
        );
        assert_eq!(messages.last().unwrap().role, MessageRole::User);
    }

    #[test]
    fn compress_drops_redundant_observations_then_oldest_turns() {
        let plugin = MemoryPlugin::new(MemoryConfig::default()).unwrap();

        // Redundant system messages are dropped; the primary one survives.
        let mut messages = vec![
            UnifiedMessage::system("primary instruction"),
            UnifiedMessage::system("redundant observation 1"),
            UnifiedMessage::system("redundant observation 2"),
            UnifiedMessage::user("latest user"),
        ];
        let report = plugin.compress(&mut messages).unwrap();
        assert!(
            report
                .actions
                .contains(&CompressionAction::DroppedObservations)
        );
        assert_eq!(
            messages
                .iter()
                .filter(|m| m.role == MessageRole::System)
                .count(),
            1,
            "only the primary system message remains"
        );
        assert_eq!(messages.last().unwrap().role, MessageRole::User);

        // The most recent user message survives every compression pass; the
        // menu converges to a no-op terminal state (system + most recent user).
        let mut messages = vec![
            UnifiedMessage::system("system prompt"),
            UnifiedMessage::user("old user turn"),
            UnifiedMessage::assistant("old assistant turn"),
            UnifiedMessage::user("most recent user"),
        ];
        let mut passes = 0usize;
        loop {
            let report = plugin.compress(&mut messages).unwrap();
            if report.actions.is_empty() {
                break;
            }
            assert_eq!(messages.last().unwrap().role, MessageRole::User);
            passes += 1;
            assert!(
                passes <= 3,
                "the menu should converge to a no-op pass quickly"
            );
        }
        assert!(passes >= 1, "the drop tail must make at least one pass");
        // Terminal state: only the system prompt and the most recent user.
        assert_eq!(messages.len(), 2);
    }

    #[test]
    fn default_config_validates() {
        let config = MemoryConfig::default();
        assert_eq!(config.max_fraction, Some(0.8));
        assert_eq!(
            config.strategies.len(),
            8,
            "the default menu has 8 strategies"
        );
        // No panic on construction with the default config.
        let _ = MemoryPlugin::new(config).unwrap();
    }

    #[test]
    fn max_tokens_and_max_fraction_conflict_is_rejected() {
        let config = MemoryConfig {
            max_tokens: Some(100),
            max_fraction: Some(0.5),
            ..Default::default()
        };
        let err = MemoryPlugin::new(config)
            .err()
            .expect("conflicting triggers must be rejected at construction");
        assert!(matches!(err, PluginError::Internal(_)));
    }

    #[test]
    fn clear_tool_results_blanks_old_pairs_keeps_last() {
        let plugin = MemoryPlugin::new(MemoryConfig {
            strategies: vec![CompactionStrategy::ClearToolResults { keep_pairs: 2 }],
            ..Default::default()
        })
        .unwrap();
        let mut messages = vec![
            UnifiedMessage::user("old user turn"),
            user_with_tool_result("c1", "first result"),
            user_with_tool_result("c2", "second result"),
            user_with_tool_result("c3", "third result"),
            UnifiedMessage::user("latest user"),
        ];
        let report = plugin.compress(&mut messages).unwrap();
        assert!(
            report
                .actions
                .contains(&CompressionAction::ClearedToolResults)
        );
        assert_eq!(messages.len(), 5, "blanking must not remove messages");
        let output_of = |m: &UnifiedMessage| match &m.content[0] {
            MessageContentBlock::ToolResult { output, .. } => output.clone(),
            _ => panic!("expected a ToolResult block"),
        };
        assert_eq!(output_of(&messages[1]), "", "the oldest pair is blanked");
        assert_eq!(output_of(&messages[2]), "second result");
        assert_eq!(output_of(&messages[3]), "third result");
        // Pairing structure (tool_call_id) is intact.
        assert!(matches!(
            messages[1].content[0],
            MessageContentBlock::ToolResult { tool_call_id: ref id, .. } if id == "c1"
        ));
    }

    #[test]
    fn deduplicate_file_reads_blanks_superseded_outputs() {
        let plugin = MemoryPlugin::new(MemoryConfig {
            strategies: vec![CompactionStrategy::DeduplicateFileReads {
                tool_name: "read_file".to_string(),
            }],
            ..Default::default()
        })
        .unwrap();
        let mut messages = vec![
            UnifiedMessage::system("primary instruction"),
            assistant_with_call("c1", "read_file", json!({ "path": "/etc/hosts" })),
            tool_result_message("c1", "first read content"),
            assistant_with_call("c2", "read_file", json!({ "path": "/etc/hosts" })),
            tool_result_message("c2", "fresh read content"),
            assistant_with_call("c3", "read_file", json!({ "path": "/etc/passwd" })),
            tool_result_message("c3", "passwd content"),
            UnifiedMessage::user("latest user"),
        ];
        let report = plugin.compress(&mut messages).unwrap();
        assert!(
            report
                .actions
                .contains(&CompressionAction::DeduplicatedFileReads)
        );
        assert_eq!(messages.len(), 8, "blanking must not remove messages");
        assert_eq!(
            joined_text(&messages[2]),
            "",
            "the older read of the same file is blanked"
        );
        assert_eq!(joined_text(&messages[4]), "fresh read content");
        assert_eq!(
            joined_text(&messages[6]),
            "passwd content",
            "a different path is untouched"
        );
    }

    #[test]
    fn clamp_truncates_oversized_blocks_and_keeps_structure() {
        let plugin = MemoryPlugin::new(MemoryConfig {
            strategies: vec![CompactionStrategy::ClampOversizedMessages {
                max_part_tokens: 100,
            }],
            ..Default::default()
        })
        .unwrap();
        let big_text = "lorem ipsum dolor sit amet ".repeat(2_000);
        let mut messages = vec![
            UnifiedMessage::system("primary instruction"),
            UnifiedMessage::user(big_text.clone()),
            UnifiedMessage::user("short text"),
            user_with_tool_result("c1", &big_text),
            UnifiedMessage {
                role: MessageRole::Assistant,
                content: vec![MessageContentBlock::Thinking {
                    reasoning: big_text.clone(),
                    signature: None,
                }],
                name: None,
                tool_call_id: None,
            },
            UnifiedMessage::user("latest user"),
        ];
        let marker = "\n…[truncated]…\n";
        let report = plugin.compress(&mut messages).unwrap();
        assert!(report.actions.contains(&CompressionAction::ClampedParts));
        assert_eq!(messages.len(), 6, "clamping edits blocks, never messages");
        assert!(joined_text(&messages[1]).contains(marker));
        assert!(joined_text(&messages[1]).len() < big_text.len());
        assert_eq!(
            joined_text(&messages[2]),
            "short text",
            "blocks under the cap are untouched"
        );
        match &messages[3].content[0] {
            MessageContentBlock::ToolResult { output, .. } => {
                assert!(output.contains(marker));
                assert!(output.len() < big_text.len());
            }
            _ => panic!("expected a ToolResult block"),
        }
        match &messages[4].content[0] {
            MessageContentBlock::Thinking {
                reasoning,
                signature,
            } => {
                assert!(reasoning.contains(marker));
                assert!(reasoning.len() < big_text.len());
                assert!(signature.is_none(), "the signature survives clamping");
            }
            _ => panic!("expected a Thinking block"),
        }
    }

    #[test]
    fn sliding_window_trims_to_tail_and_preserves_pairing() {
        let plugin = MemoryPlugin::new(MemoryConfig {
            strategies: vec![CompactionStrategy::SlidingWindow { keep_messages: 10 }],
            ..Default::default()
        })
        .unwrap();
        let mut messages = vec![UnifiedMessage::system("primary instruction")];
        for i in 0..49 {
            if i % 2 == 0 {
                messages.push(UnifiedMessage::user(format!("user turn {i}")));
            } else {
                messages.push(UnifiedMessage::assistant(format!("assistant turn {i}")));
            }
        }
        assert_eq!(messages.len(), 50);
        let report = plugin.compress(&mut messages).unwrap();
        assert!(report.actions.contains(&CompressionAction::Slid));
        assert_eq!(messages.len(), 10, "50 messages trim to a tail of 10");
        assert_eq!(
            messages[0].role,
            MessageRole::System,
            "the first system message survives"
        );
        assert_eq!(
            joined_text(messages.last().unwrap()),
            "user turn 48",
            "the most recent user message survives"
        );
    }

    #[test]
    fn sliding_window_does_not_orphan_tool_results() {
        let plugin = MemoryPlugin::new(MemoryConfig {
            strategies: vec![CompactionStrategy::SlidingWindow { keep_messages: 6 }],
            ..Default::default()
        })
        .unwrap();
        let mut messages = vec![
            UnifiedMessage::system("primary instruction"),
            UnifiedMessage::user("u1"),
            UnifiedMessage::assistant("a1"),
            UnifiedMessage::user("u2"),
            assistant_with_call("c1", "read_file", json!({ "path": "/x" })),
            UnifiedMessage::user("u3"),
            UnifiedMessage::assistant("a3"),
            tool_result_message("c1", "result text"),
            UnifiedMessage::user("u4"),
            UnifiedMessage::assistant("a4"),
            UnifiedMessage::assistant("a5"),
            UnifiedMessage::assistant("a6"),
        ];
        assert_eq!(messages.len(), 12);
        let report = plugin.compress(&mut messages).unwrap();
        assert!(report.actions.contains(&CompressionAction::Slid));
        // Drop set {1..=6} plus the paired result message at index 7: 5 remain.
        assert_eq!(messages.len(), 5);
        assert_eq!(messages[0].role, MessageRole::System);
        assert_eq!(joined_text(&messages[1]), "u4", "most recent user survives");
        assert!(
            !messages
                .iter()
                .any(|m| m.tool_call_id.as_deref() == Some("c1")),
            "the result message must not outlive its call"
        );
        assert!(
            !messages.iter().any(|m| {
                m.content.iter().any(|b| {
                    matches!(
                        b,
                        MessageContentBlock::ToolCall { id, .. } if id == "c1"
                    )
                })
            }),
            "the call message must be gone"
        );
    }

    #[test]
    fn warn_injection_is_idempotent_and_triggered_by_fraction() {
        let plugin = MemoryPlugin::new(MemoryConfig {
            context_window_tokens: 100,
            warn_fraction: Some(0.5),
            max_fraction: None,
            ..Default::default()
        })
        .unwrap();
        let mut req = UnifiedRequest::new("gpt-4o");
        req.messages
            .push(UnifiedMessage::system("You are a helpful assistant."));
        for i in 0..8 {
            req.messages.push(UnifiedMessage::user(format!(
                "Turn number {i}: several full sentences of conversational content that consume \
                 tokens and drive the total well past the fifty-token mark for this tiny window."
            )));
        }
        plugin.on_request(&mut req).unwrap();
        assert_eq!(
            req.messages
                .iter()
                .filter(|m| joined_text(m).contains(WARNING_MARKER))
                .count(),
            1,
            "the first over-fraction request injects exactly one warning"
        );
        plugin.on_request(&mut req).unwrap();
        assert_eq!(
            req.messages
                .iter()
                .filter(|m| joined_text(m).contains(WARNING_MARKER))
                .count(),
            1,
            "the marker scan prevents a second injection"
        );
    }

    #[test]
    fn observer_receives_usage_every_request() {
        let observer = Arc::new(FakeObserver(Mutex::new(Vec::new())));
        let plugin = MemoryPlugin::new(MemoryConfig {
            observers: vec![observer.clone()],
            ..Default::default()
        })
        .unwrap();
        let mut req = UnifiedRequest::new("gpt-4o");
        req.messages.push(UnifiedMessage::user("hi"));
        plugin.on_request(&mut req).unwrap();
        plugin.on_request(&mut req).unwrap();
        let readings = observer.0.lock().unwrap();
        assert_eq!(readings.len(), 2, "one reading per request");
        for usage in readings.iter() {
            assert!(usage.used_tokens > 0);
            assert_eq!(usage.window_tokens, 128_000);
            assert!(!usage.resolved, "no resolver: the fallback window is used");
        }
    }

    #[test]
    fn resolver_overrides_fallback_window() {
        let observer = Arc::new(FakeObserver(Mutex::new(Vec::new())));
        let plugin = MemoryPlugin::new(MemoryConfig {
            context_window_resolver: Some(Arc::new(FakeResolver)),
            observers: vec![observer.clone()],
            ..Default::default()
        })
        .unwrap();
        let mut req = UnifiedRequest::new("big-model");
        req.messages.push(UnifiedMessage::user("hi"));
        plugin.on_request(&mut req).unwrap();
        req.model = "small-model".to_string();
        plugin.on_request(&mut req).unwrap();
        let readings = observer.0.lock().unwrap();
        assert_eq!(readings.len(), 2);
        assert_eq!(readings[0].window_tokens, 1_000_000);
        assert!(readings[0].resolved);
        assert_eq!(readings[1].window_tokens, 128_000);
        assert!(!readings[1].resolved, "unresolved models fall back");
    }

    #[test]
    fn fraction_budget_uses_resolved_window() {
        let plugin = MemoryPlugin::new(MemoryConfig {
            context_window_resolver: Some(Arc::new(TinyResolver)),
            max_fraction: Some(0.5),
            max_tokens: None,
            max_messages: None,
            ..Default::default()
        })
        .unwrap();

        // Resolved window 100 x 0.5 = 50-token budget: this list is over it.
        let mut hot = UnifiedRequest::new("m");
        hot.messages
            .push(UnifiedMessage::system("You are a helpful assistant."));
        for i in 0..8 {
            hot.messages.push(UnifiedMessage::user(format!(
                "Turn number {i}: several full sentences of conversational content that consume \
                 tokens and drive the total well past the fifty-token mark for this tiny window."
            )));
        }
        let before = hot.messages.len();
        plugin.on_request(&mut hot).unwrap();
        assert!(
            hot.messages.len() < before,
            "over the fraction budget, on_request compresses"
        );

        // Under the 50-token budget: untouched.
        let mut cool = UnifiedRequest::new("m");
        cool.messages.push(UnifiedMessage::user("hi"));
        let snapshot = cool.messages.clone();
        plugin.on_request(&mut cool).unwrap();
        assert_eq!(
            cool.messages, snapshot,
            "under the fraction budget, messages are untouched"
        );
    }

    #[test]
    fn offload_error_falls_back_to_next_strategy() {
        let plugin = MemoryPlugin::with_extensions(
            MemoryConfig {
                strategies: vec![
                    CompactionStrategy::Offload { turns: 10 },
                    CompactionStrategy::Summarize { turns: 10 },
                ],
                ..Default::default()
            },
            Arc::new(FakeSummarizer),
            Arc::new(FailingStore),
        )
        .unwrap();
        let mut messages = vec![
            UnifiedMessage::system("system prompt"),
            UnifiedMessage::user("oldest user turn"),
            UnifiedMessage::assistant("oldest assistant turn"),
            UnifiedMessage::user("most recent user"),
        ];
        let report = plugin.compress(&mut messages).unwrap();
        assert!(
            report.actions.contains(&CompressionAction::Summarized),
            "after an offload error the pipeline falls through to summarize"
        );
        assert!(
            report.last_error.is_some(),
            "the offload failure is recorded"
        );
        assert_eq!(messages.len(), 3, "summarize still replaced the turn group");
    }

    #[test]
    fn failed_offload_restores_every_turn_it_took() {
        // Offload alone: nothing downstream can rewrite the list, so the
        // messages observed afterwards are exactly what the rollback restored.
        let plugin = MemoryPlugin::with_extensions(
            MemoryConfig {
                strategies: vec![CompactionStrategy::Offload { turns: 10 }],
                ..Default::default()
            },
            Arc::new(FakeSummarizer),
            Arc::new(FailingStore),
        )
        .unwrap();
        let mut messages = vec![
            UnifiedMessage::system("system prompt"),
            UnifiedMessage::user("oldest user turn"),
            UnifiedMessage::assistant("oldest assistant turn"),
            UnifiedMessage::user("most recent user"),
        ];
        let before = messages.clone();

        let report = plugin.compress(&mut messages).unwrap();

        assert!(
            report.last_error.is_some(),
            "the offload failure is recorded"
        );
        assert_eq!(
            messages, before,
            "a failed offload drops no history and preserves order"
        );
    }

    #[test]
    fn pairing_close_extends_drop_set() {
        let messages = vec![
            assistant_with_call("c1", "read_file", json!({ "path": "/x" })),
            tool_result_message("c1", "out"),
        ];
        let drop_set: HashSet<usize> = HashSet::from([0]);
        let closed = pairing_close(&messages, &drop_set);
        assert_eq!(
            closed,
            HashSet::from([0, 1]),
            "a dropped call pulls in its result"
        );
        // The closure is one-directional: dropping a result does not pull the call.
        let result_only: HashSet<usize> = HashSet::from([1]);
        assert_eq!(pairing_close(&messages, &result_only), HashSet::from([1]));
    }

    #[test]
    fn compress_out_of_band_runs_full_list_without_budget() {
        let plugin = MemoryPlugin::new(MemoryConfig::default()).unwrap();
        let mut messages = vec![
            UnifiedMessage::system("primary instruction"),
            assistant_with_call("c1", "read_file", json!({ "path": "/x" })),
            tool_result_message("c1", "result output"),
            UnifiedMessage::user("u1"),
            UnifiedMessage::assistant("a1"),
            UnifiedMessage::user("u2"),
        ];
        let report = plugin.compress(&mut messages).unwrap();
        assert!(
            !report.actions.is_empty(),
            "the full strategy list runs once without a budget"
        );
        assert_eq!(
            messages.last().unwrap().role,
            MessageRole::User,
            "the most recent user message survives"
        );
        assert!(
            !messages
                .iter()
                .any(|m| m.tool_call_id.as_deref() == Some("c1")),
            "the paired tool result is removed with its call"
        );
    }

    #[test]
    fn name_and_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<MemoryPlugin>();
        assert_eq!(
            MemoryPlugin::new(MemoryConfig::default()).unwrap().name(),
            "context-memory"
        );
    }

    #[test]
    fn memory_plugin_graph_defaults_empty() {
        let plugin = MemoryPlugin::new(MemoryConfig::default()).unwrap();
        assert!(
            plugin.graph().unwrap().is_empty(),
            "a freshly built plugin starts with an empty graph"
        );
        let store = Arc::new(FakeStore(Mutex::new(Vec::new())));
        let plugin =
            MemoryPlugin::with_extensions(MemoryConfig::default(), Arc::new(FakeSummarizer), store)
                .unwrap();
        assert!(plugin.graph().unwrap().is_empty());
    }

    #[test]
    fn merge_graph_merges_under_single_lock() {
        let plugin = MemoryPlugin::new(MemoryConfig::default()).unwrap();
        let mut incoming = MemoryGraph::new();
        incoming.upsert_node(GraphNode {
            id: "a".into(),
            labels: vec!["person".into()],
            properties: serde_json::Map::new(),
        });
        incoming.upsert_node(GraphNode {
            id: "b".into(),
            labels: Vec::new(),
            properties: serde_json::Map::new(),
        });
        incoming
            .add_relationship(GraphRelationship {
                id: "r".into(),
                from: "a".into(),
                to: "b".into(),
                kind: "knows".into(),
                weight: 1.0,
                properties: serde_json::Map::new(),
            })
            .unwrap();
        let report = plugin.merge_graph(incoming, MergePolicy::Keep).unwrap();
        assert_eq!(report.nodes_added, 2);
        assert_eq!(report.relationships_added, 1);
        let guard = plugin.graph().unwrap();
        assert_eq!(guard.len(), 2);
        assert_eq!(guard.relationship_count(), 1);
        assert!(guard.relationship("r").is_some());
    }

    #[test]
    fn replace_graph_swaps_working_graph() {
        let plugin = MemoryPlugin::new(MemoryConfig::default()).unwrap();
        let mut g = MemoryGraph::new();
        g.upsert_node(GraphNode {
            id: "a".into(),
            labels: Vec::new(),
            properties: serde_json::Map::new(),
        });
        plugin.replace_graph(g).unwrap();
        assert_eq!(plugin.graph().unwrap().len(), 1);
        plugin.replace_graph(MemoryGraph::new()).unwrap();
        assert!(plugin.graph().unwrap().is_empty());
    }

    /// Staging validates off the lock: a rejected snapshot never reaches the
    /// live graph, and a staged graph is not live until it is committed.
    #[test]
    fn stage_snapshot_validates_without_touching_the_live_graph() {
        let plugin = MemoryPlugin::new(MemoryConfig::default()).unwrap();
        plugin
            .replace_snapshot(GraphSnapshot {
                nodes: vec![GraphNode {
                    id: "sentinel".into(),
                    labels: Vec::new(),
                    properties: serde_json::Map::new(),
                }],
                relationships: Vec::new(),
            })
            .unwrap();
        let before = plugin.snapshot().unwrap();

        let bad = GraphSnapshot {
            nodes: vec![GraphNode {
                id: "x".into(),
                labels: Vec::new(),
                properties: serde_json::Map::new(),
            }],
            relationships: vec![GraphRelationship {
                id: "r".into(),
                from: "x".into(),
                to: "ghost".into(),
                kind: "knows".into(),
                weight: 1.0,
                properties: serde_json::Map::new(),
            }],
        };
        assert!(MemoryPlugin::stage_snapshot(bad).is_err());
        assert_eq!(plugin.snapshot().unwrap(), before);

        // A successfully staged graph is inert until committed, and staging can
        // run while the live graph lock is held by someone else.
        let guard = plugin.graph().unwrap();
        let staged = MemoryPlugin::stage_snapshot(GraphSnapshot {
            nodes: vec![GraphNode {
                id: "fresh".into(),
                labels: Vec::new(),
                properties: serde_json::Map::new(),
            }],
            relationships: Vec::new(),
        })
        .unwrap();
        assert_eq!(staged.len(), 1);
        assert!(
            guard.node("sentinel").is_some(),
            "live graph still the old one"
        );
        drop(guard);

        let report = plugin.commit_staged_graph(staged).unwrap();
        assert_eq!(
            report,
            GraphImportReport {
                nodes: 1,
                relationships: 0
            }
        );
        assert!(plugin.graph().unwrap().node("fresh").is_some());
        assert!(plugin.graph().unwrap().node("sentinel").is_none());
    }

    /// Build a plugin with graph-context injection enabled and one node in the
    /// graph.
    fn plugin_with_graph_context(config: MemoryConfig) -> MemoryPlugin {
        let plugin = MemoryPlugin::new(config).unwrap();
        plugin.graph().unwrap().upsert_node(GraphNode {
            id: "alice".into(),
            labels: vec!["person".into()],
            properties: serde_json::Map::new(),
        });
        plugin
    }

    /// Count of graph-context messages in `messages` (marker-prefix scan).
    fn marker_count(messages: &[UnifiedMessage]) -> usize {
        messages
            .iter()
            .filter(|m| joined_text(m).starts_with(GRAPH_RENDER_MARKER))
            .count()
    }

    #[test]
    fn graph_context_injected_after_first_system_idempotently() {
        let plugin = plugin_with_graph_context(MemoryConfig {
            graph_context: Some(GraphContextConfig::default()),
            ..Default::default()
        });
        let mut req = UnifiedRequest::new("gpt-4o");
        req.messages
            .push(UnifiedMessage::system("primary instruction"));
        req.messages.push(UnifiedMessage::user("hi"));
        plugin.on_request(&mut req).unwrap();
        assert_eq!(marker_count(&req.messages), 1, "exactly one graph message");
        assert_eq!(req.messages[0].role, MessageRole::System);
        assert!(
            joined_text(&req.messages[1]).starts_with(GRAPH_RENDER_MARKER),
            "the graph message sits right after the first System message"
        );
        // Second request: replaced in place, never duplicated.
        plugin.on_request(&mut req).unwrap();
        assert_eq!(marker_count(&req.messages), 1);
        // Changing the graph updates the message content in place.
        plugin.graph().unwrap().upsert_node(GraphNode {
            id: "bob".into(),
            labels: Vec::new(),
            properties: serde_json::Map::new(),
        });
        plugin.on_request(&mut req).unwrap();
        assert_eq!(marker_count(&req.messages), 1);
        assert!(
            joined_text(&req.messages[1]).contains("bob"),
            "the message content reflects the updated graph"
        );
    }

    #[test]
    fn graph_context_removed_when_graph_emptied() {
        let plugin = plugin_with_graph_context(MemoryConfig {
            graph_context: Some(GraphContextConfig::default()),
            ..Default::default()
        });
        let mut req = UnifiedRequest::new("gpt-4o");
        req.messages
            .push(UnifiedMessage::system("primary instruction"));
        plugin.on_request(&mut req).unwrap();
        assert_eq!(marker_count(&req.messages), 1);
        plugin.replace_graph(MemoryGraph::new()).unwrap();
        plugin.on_request(&mut req).unwrap();
        assert_eq!(
            marker_count(&req.messages),
            0,
            "a stale graph message is removed once the graph is empty"
        );
    }

    #[test]
    fn graph_context_absent_when_disabled_or_empty() {
        // Disabled (default config) with a non-empty graph: no injection.
        let plugin = plugin_with_graph_context(MemoryConfig::default());
        let mut req = UnifiedRequest::new("gpt-4o");
        req.messages
            .push(UnifiedMessage::system("primary instruction"));
        plugin.on_request(&mut req).unwrap();
        assert_eq!(
            marker_count(&req.messages),
            0,
            "graph_context None disables injection"
        );
        // Enabled with an empty graph: no injection, no churn.
        let plugin = MemoryPlugin::new(MemoryConfig {
            graph_context: Some(GraphContextConfig::default()),
            ..Default::default()
        })
        .unwrap();
        let mut req = UnifiedRequest::new("gpt-4o");
        req.messages
            .push(UnifiedMessage::system("primary instruction"));
        plugin.on_request(&mut req).unwrap();
        plugin.on_request(&mut req).unwrap();
        assert_eq!(marker_count(&req.messages), 0);
    }

    #[test]
    fn graph_context_respects_rendering_limits() {
        let plugin = plugin_with_graph_context(MemoryConfig {
            graph_context: Some(GraphContextConfig {
                max_nodes: 1,
                max_relationships: 1,
            }),
            ..Default::default()
        });
        plugin.graph().unwrap().upsert_node(GraphNode {
            id: "bob".into(),
            labels: Vec::new(),
            properties: serde_json::Map::new(),
        });
        let mut req = UnifiedRequest::new("gpt-4o");
        req.messages
            .push(UnifiedMessage::system("primary instruction"));
        plugin.on_request(&mut req).unwrap();
        let text = joined_text(&req.messages[1]);
        assert!(text.starts_with(GRAPH_RENDER_MARKER));
        assert!(
            text.contains("... 1 more nodes omitted"),
            "the injected render honors max_nodes"
        );
        assert!(!text.contains("node bob"), "the second node is omitted");
    }

    #[test]
    fn graph_context_survives_compression() {
        let plugin = plugin_with_graph_context(MemoryConfig {
            context_window_tokens: 100,
            max_fraction: Some(0.8),
            graph_context: Some(GraphContextConfig::default()),
            ..Default::default()
        });
        let mut req = UnifiedRequest::new("gpt-4o");
        req.messages
            .push(UnifiedMessage::system("You are a helpful assistant."));
        for i in 0..8 {
            req.messages.push(UnifiedMessage::user(format!(
                "Turn number {i}: several full sentences of conversational content that consume \
                 tokens and drive the total well past the eighty-token threshold."
            )));
        }
        let before = req.messages.len();
        plugin.on_request(&mut req).unwrap();
        assert!(
            req.messages.len() < before,
            "over the budget, on_request still compresses"
        );
        assert_eq!(
            marker_count(&req.messages),
            1,
            "the graph context message survives compression"
        );
    }
}
