//! Feature-gated plugin implementations.
//!
//! Every capability behind a `plugin-*` feature lives in one submodule here;
//! each submodule's `//!` header is the authority on its hooks, its
//! bounded-growth policy, and any documented cross-plugin edge. Nothing in
//! this module is compiled unless its feature is enabled.

/// OpenTelemetry observability hooks for the request pipeline.
#[cfg(feature = "plugin-telemetry")]
pub mod telemetry;

/// JSON Schema output guardrails with bounded retry injection.
#[cfg(feature = "plugin-guardrails")]
pub mod guardrails;

/// Context-window compaction plus the working memory graph.
#[cfg(feature = "plugin-memory")]
pub mod memory;

/// Schema-validated entity/relationship extraction into a `MemoryGraph` delta
/// the caller merges into the memory plugin's working graph.
#[cfg(feature = "plugin-entity-extraction")]
pub mod entity_extraction;

/// Model Context Protocol client: remote tool discovery and invocation.
#[cfg(feature = "plugin-mcp")]
pub mod mcp;

/// WebAssembly code-execution sandbox with fuel, memory, and time bounds.
#[cfg(feature = "plugin-sandbox")]
pub mod sandbox;

/// Child subagent delegation with optional Git worktree isolation.
#[cfg(feature = "plugin-subagent")]
pub mod subagent;

/// Human-in-the-loop approval interceptors for high-risk tool calls.
#[cfg(feature = "plugin-hitl")]
pub mod hitl;
/// Append-only session trajectory store and forking.
#[cfg(feature = "plugin-session-log")]
pub mod session_log;
/// Reusable agent skills: SKILL.md instructions discoverable via the
/// `skill` / `skill_read` / `skill_search` tools (agentskills.io).
#[cfg(feature = "plugin-skills")]
pub mod skills;
/// Provider-backed web search and page extraction tools.
#[cfg(feature = "plugin-web-search")]
pub mod web_search;

/// Client-owned local response cache with deterministic effective-request
/// digesting and bounded LRU/TTL eviction.
#[cfg(feature = "plugin-prompt-cache")]
pub mod prompt_cache;

/// Client-side outbound throttle: an integer token bucket over request rate
/// plus a hard cap on concurrently in-flight turns.
#[cfg(feature = "plugin-rate-limit")]
pub mod rate_limit;

/// Outbound PII/secret scrubbing over every text-bearing request field.
#[cfg(feature = "plugin-redaction")]
pub mod redaction;

/// Cumulative token/currency ledger with a hard budget cap enforced before
/// provider dispatch.
#[cfg(feature = "plugin-cost")]
pub mod cost;
