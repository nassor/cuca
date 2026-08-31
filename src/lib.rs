//! CUCA: Compact Universal Client for Agents.
//! Unified, zero-default, everything-is-a-plugin async LLM client (see dev-docs spec).

#![forbid(unsafe_code)]
#[cfg(not(any(
    feature = "provider-openai",
    feature = "provider-anthropic",
    feature = "provider-deepseek",
    feature = "provider-gemini",
    feature = "provider-llamacpp",
    feature = "provider-vllm",
    feature = "provider-lmstudio",
)))]
compile_error!("CUCA requires one provider-* feature; enable the adapter for the backend you use.");

/// Core unified wire types shared by every provider adapter.
pub mod types;

/// Client-facing and plugin-facing error types.
pub mod error;

/// Normalized request/response contracts for the unified abstraction.
pub mod request;

/// Append-only session audit-trail model.
pub mod session;

/// Zero-allocation SSE stream parser engine.
pub mod sse;

/// Plugin trait layer: `CucaPlugin` and `SessionStorePlugin`.
pub mod plugin;

/// Feature-gated plugin implementations; one submodule per `plugin-*` feature.
pub mod plugins;

/// Client core: builder, client, and the plugin-instrumented stream pipeline.
pub mod client;

/// Provider adapter layer: per-provider dispatch implementations.
pub(crate) mod provider;

/// Canonical JSON encoding for `serde_json::Value` leaves carried inside
/// postcard-encoded storage records and digest input.
#[cfg(any(feature = "plugin-session-log", feature = "plugin-prompt-cache"))]
pub(crate) mod canonical;

/// Versioned canonical `cuca-export` envelope for memory-graph and
/// local-response-cache state.
#[cfg(any(feature = "plugin-memory", feature = "plugin-prompt-cache"))]
pub mod export;

/// Speculative fast/slow model pairing and deterministic complexity routing.
#[cfg(feature = "plugin-speculative")]
pub mod orchestrator;
pub use crate::client::{CucaClient, CucaClientBuilder};
pub use crate::error::{CucaError, PluginError};
#[cfg(any(feature = "plugin-memory", feature = "plugin-prompt-cache"))]
pub use crate::export::{
    CUCA_EXPORT_FORMAT, CUCA_EXPORT_VERSION, CucaExport, CucaExportError, CucaImportReport,
    GraphExportSection, PromptCacheExportSection,
};
#[cfg(feature = "plugin-entity-extraction")]
pub use crate::plugins::entity_extraction::{
    CandidateEntity, CandidateRelationship, EntityExtractionCandidate, EntityExtractionModel,
    EntityExtractionPlugin, EntityExtractionReport, EntityExtractionSchema, EntityReference,
    EntityTable, PropertyColumn, PropertyType, RelationshipTable,
};

#[cfg(feature = "plugin-speculative")]
pub use crate::orchestrator::{
    ClientPool, Complexity, ComplexityEvaluator, DraftValidator, JsonToolDraftValidator,
    ModelOrchestrator, SwappableModelPair, TurnExecutor,
};
#[cfg(feature = "plugin-guardrails")]
pub use crate::plugins::guardrails::JsonGuardrailPlugin;
#[cfg(feature = "plugin-hitl")]
pub use crate::plugins::hitl::{
    ApprovalChannel, ApprovalDecision, ApprovalRequest, HitlPlugin, OneshotApprovalChannel, Risk,
};
#[cfg(feature = "plugin-mcp")]
pub use crate::plugins::mcp::{McpPlugin, McpTransport};
#[cfg(feature = "plugin-memory")]
pub use crate::plugins::memory::{
    Budget, CompactionStrategy, CompressionAction, CompressionReport, ContextUsage,
    ContextUsageObserver, ContextWindowResolver, GraphContextConfig, GraphDirection,
    GraphImportReport, GraphNode, GraphRelationship, GraphSnapshot, MemoryConfig, MemoryGraph,
    MemoryPlugin, MergePolicy, MergeReport, Summarizer, VectorStore,
};
#[cfg(feature = "plugin-prompt-cache")]
pub use crate::plugins::prompt_cache::{
    PromptCache, PromptCacheConfig, PromptCacheEntry, PromptCacheError, PromptCacheImportReport,
    PromptCacheSnapshot,
};
#[cfg(feature = "plugin-sandbox")]
pub use crate::plugins::sandbox::{SandboxConfig, SandboxPlugin, SandboxResult};
#[cfg(feature = "plugin-session-log")]
pub use crate::plugins::session_log::{
    FileBackend, InMemoryBackend, SessionBackend, SessionLogPlugin,
};
#[cfg(feature = "plugin-skills")]
pub use crate::plugins::skills::{Skill, SkillsConfig, SkillsPlugin};
#[cfg(feature = "plugin-subagent")]
pub use crate::plugins::subagent::{
    SubagentPlugin, SubagentResult, SubagentRunner, SubagentSpec, WorktreeConfig,
};
#[cfg(feature = "plugin-telemetry")]
pub use crate::plugins::telemetry::OpenTelemetryPlugin;
#[cfg(feature = "plugin-web-search")]
pub use crate::plugins::web_search::{
    SearchResult, WebSearchConfig, WebSearchPlugin, WebSearchProvider,
};
pub use crate::request::{
    AgentResponseStream, PromptCacheBreakpoint, PromptCacheDirective, PromptCacheUsage,
    ThinkingConfig, ThinkingEffort, ThinkingParams, UnifiedRequest, UnifiedResponse,
};
pub use crate::session::{SessionEvent, SessionRecord};
