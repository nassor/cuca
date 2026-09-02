//! CUCA: Compact Universal Client for Agents.
//! Unified, zero-default async LLM client: a pipeline-plugin tier and an
//! explicit-call service tier over one provider abstraction.

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

/// Feature-gated explicit-call capabilities; one submodule per `service-*`
/// feature. Services are driven by direct method calls and never implement
/// `CucaPlugin`.
pub mod services;

/// Client core: builder, client, and the plugin-instrumented stream pipeline.
pub mod client;

/// Provider adapter layer: per-provider dispatch implementations.
pub(crate) mod provider;

/// Canonical JSON encoding for `serde_json::Value` leaves carried inside
/// postcard-encoded storage records and digest input.
#[cfg(any(feature = "plugin-session-log", feature = "service-prompt-cache"))]
pub(crate) mod canonical;

/// tiktoken-rs encoder resolution shared by the token-counting plugins.
#[cfg(any(feature = "plugin-memory", feature = "plugin-cost"))]
pub(crate) mod tokenize;

/// Versioned canonical `cuca-export` envelope for memory-graph and
/// local-response-cache state.
#[cfg(any(feature = "plugin-memory", feature = "service-prompt-cache"))]
pub mod export;

/// The cost ledger to OpenTelemetry bridge: a ready-made `CostObserver` that
/// records each reading to the caller's meter provider.
#[cfg(all(feature = "plugin-cost", feature = "plugin-telemetry"))]
pub mod cost_otel;

pub use crate::client::{CucaClient, CucaClientBuilder};
#[cfg(all(feature = "plugin-cost", feature = "plugin-telemetry"))]
pub use crate::cost_otel::OtelCostObserver;
pub use crate::error::{CucaError, PluginError};
#[cfg(any(feature = "plugin-memory", feature = "service-prompt-cache"))]
pub use crate::export::{
    CUCA_EXPORT_FORMAT, CUCA_EXPORT_VERSION, CucaExport, CucaExportError, CucaImportReport,
    GraphExportSection, PromptCacheExportSection,
};
#[cfg(feature = "plugin-cost")]
pub use crate::plugins::cost::{
    CostConfig, CostEntry, CostObserver, CostPlugin, CostUsage, ModelRates, PricingResolver,
    PricingTable, UnpricedModelPolicy,
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
#[cfg(feature = "plugin-redaction")]
pub use crate::plugins::redaction::{Redacted, RedactionConfig, RedactionPlugin, RedactionRule};
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
#[cfg(feature = "service-entity-extraction")]
pub use crate::services::entity_extraction::{
    CandidateEntity, CandidateRelationship, EntityExtractionCandidate, EntityExtractionModel,
    EntityExtractionReport, EntityExtractionSchema, EntityExtractor, EntityReference, EntityTable,
    PropertyColumn, PropertyType, RelationshipTable,
};
#[cfg(feature = "service-speculative")]
pub use crate::services::orchestrator::{
    ClientPool, Complexity, ComplexityEvaluator, DraftValidator, JsonToolDraftValidator,
    ModelOrchestrator, SwappableModelPair, TurnExecutor,
};
#[cfg(feature = "service-prompt-cache")]
pub use crate::services::prompt_cache::{
    PromptCache, PromptCacheConfig, PromptCacheEntry, PromptCacheError, PromptCacheImportReport,
    PromptCacheSnapshot,
};
#[cfg(feature = "service-rate-limit")]
pub use crate::services::rate_limit::{
    RateLimitConfig, RateLimitError, RateLimitObserver, RateLimitPermit, RateLimitUsage,
    RateLimiter,
};
#[cfg(feature = "service-replay")]
pub use crate::services::replay::{
    ReplayCompletion, ReplayConfig, ReplayNote, ReplayTrajectory, ReplayTurn, ReplayUsage,
    SessionReplay,
};
#[cfg(feature = "service-vector-store")]
pub use crate::services::vector_store::{
    Embedder, InMemoryVectorStore, RECALL_RENDER_MARKER, RecallInjection, RetrievalReport,
    RetrievedTurn, VectorStoreConfig, VectorStoreError, VectorStoreUsage,
};
pub use crate::session::{SessionEvent, SessionRecord};
