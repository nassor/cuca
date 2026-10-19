+++
title = "The public API surface"
description = "Every public module and every top-level re-export from src/lib.rs, with the feature gating each one."
template = "page.html"
weight = 4
+++

# The public API surface

<dl class="page-facts">
<dt>In one line</dt>
<dd>Nine always-present modules, two gated modules, and 148 re-exported items at the crate root, grouped by the feature that gates them</dd>
<dt>You need</dt>
<dd>Nothing; this page mirrors <code>src/lib.rs</code></dd>
<dt>Read this if</dt>
<dd>You are looking for the import path of a type, or which feature makes it exist</dd>
</dl>

`src/lib.rs` opens with `#![forbid(unsafe_code)]`, so no `unsafe` block compiles anywhere in the crate. A build with no provider feature stops at a `compile_error!`; see [The feature matrix](@/reference/features.md).

## Modules

| Module | Visibility | Gate | Holds |
|---|---|---|---|
| `types` | `pub` | none | the unified wire types shared by every provider adapter |
| `error` | `pub` | none | `CucaError` and `PluginError` |
| `request` | `pub` | none | the normalized request and response contracts |
| `session` | `pub` | none | the append-only session audit-trail model |
| `sse` | `pub` | none | the SSE stream parser |
| `plugin` | `pub` | none | the `CucaPlugin` and `SessionStorePlugin` traits |
| `plugins` | `pub` | none | plugin implementations, one gated submodule per `plugin-*` feature |
| `services` | `pub` | none | service implementations, one gated submodule per `service-*` feature |
| `client` | `pub` | none | the builder, the client and the plugin-instrumented stream |
| `export` | `pub` | `plugin-memory` or `service-prompt-cache` | the versioned `cuca-export` envelope |
| `cost_otel` | `pub` | `plugin-cost` and `plugin-telemetry` | the cost ledger to OpenTelemetry bridge |
| `provider` | `pub(crate)` | none | the provider adapters; not part of the public surface |
| `canonical` | `pub(crate)` | `plugin-session-log` or `service-prompt-cache` | canonical JSON encoding for storage records and digest input |
| `tokenize` | `pub(crate)` | `plugin-memory` or `plugin-cost` | tiktoken encoder resolution |

## Ungated re-exports

Available in every build that compiles. There are 17.

| Group | Items |
|---|---|
| Client | `CucaClient`, `CucaClientBuilder`, `ResponseHandle` |
| Errors | `CucaError`, `PluginError` |
| Request and response | `AgentResponseStream`, `PromptCacheBreakpoint`, `PromptCacheDirective`, `PromptCacheUsage`, `ThinkingConfig`, `ThinkingEffort`, `ThinkingParams`, `TokenUsage`, `UnifiedRequest`, `UnifiedResponse` |
| Session | `SessionEvent`, `SessionRecord` |

`PromptCacheBreakpoint`, `PromptCacheDirective` and `PromptCacheUsage` are ungated although `service-prompt-cache` is gated. They are fields of `UnifiedRequest` and `UnifiedResponse`, so they exist wherever those do. `TokenUsage` is ungated for the same reason.

## Items outside the crate root

| Path | Items |
|---|---|
| `cuca::types` | `MessageContentBlock`, `MessageRole`, `ProviderEndpoint`, `ToolDefinition`, `UnifiedMessage` |
| `cuca::sse` | `SseEvent`, `SseStreamParser` |
| `cuca::plugin` | `CucaPlugin`, `SessionStorePlugin` |
| `cuca::services::orchestrator` | `PoolTurnExecutor`, which is `pub` but not re-exported at the root |

`LlamaCppConfig` and `OAuthPkceConfig` live in the `pub(crate)` `provider` module and have no public path.

## Feature-gated re-exports

There are 131, in 21 groups.

| Gate | Items |
|---|---|
| `provider-local` | `DownloadProgress`, `DownloadProgressCallback`, `DraftStats`, `DrafterSource`, `GpuAdapter`, `LocalCapabilities`, `LocalDevice`, `LocalModel`, `LocalModelConfig`, `ModelPreset`, `ModelSource`, `SamplingConfig` |
| `plugin-cost` | `CostConfig`, `CostEntry`, `CostObserver`, `CostPlugin`, `CostUsage`, `ModelRates`, `PricingResolver`, `PricingTable`, `UnpricedModelPolicy` |
| `plugin-guardrails` | `JsonGuardrailPlugin` |
| `plugin-hitl` | `ApprovalChannel`, `ApprovalDecision`, `ApprovalRequest`, `HitlPlugin`, `OneshotApprovalChannel`, `Risk` |
| `plugin-mcp` | `McpPlugin`, `McpTransport` |
| `plugin-memory` | `Budget`, `CompactionStrategy`, `CompressionAction`, `CompressionReport`, `ContextUsage`, `ContextUsageObserver`, `ContextWindowResolver`, `GRAPH_RENDER_MARKER`, `GraphContextConfig`, `GraphDirection`, `GraphImportReport`, `GraphNode`, `GraphRelationship`, `GraphSnapshot`, `MemoryConfig`, `MemoryGraph`, `MemoryPlugin`, `MergePolicy`, `MergeReport`, `Summarizer`, `VectorStore` |
| `plugin-redaction` | `Redacted`, `RedactionConfig`, `RedactionPlugin`, `RedactionRule` |
| `plugin-sandbox` | `SandboxConfig`, `SandboxPlugin`, `SandboxResult` |
| `plugin-session-log` | `FileBackend`, `InMemoryBackend`, `SessionBackend`, `SessionLogPlugin` |
| `plugin-skills` | `Skill`, `SkillsConfig`, `SkillsPlugin` |
| `plugin-subagent` | `SubagentPlugin`, `SubagentResult`, `SubagentRunner`, `SubagentSpec`, `WorktreeConfig` |
| `plugin-telemetry` | `OpenTelemetryPlugin` |
| `plugin-web-search` | `SearchResult`, `WebSearchConfig`, `WebSearchPlugin`, `WebSearchProvider` |
| `service-entity-extraction` | `CandidateEntity`, `CandidateRelationship`, `EntityExtractionCandidate`, `EntityExtractionModel`, `EntityExtractionReport`, `EntityExtractionSchema`, `EntityExtractor`, `EntityReference`, `EntityTable`, `PropertyColumn`, `PropertyType`, `RelationshipTable` |
| `service-prompt-cache` | `PromptCache`, `PromptCacheConfig`, `PromptCacheEntry`, `PromptCacheError`, `PromptCacheImportReport`, `PromptCacheSnapshot` |
| `service-rate-limit` | `RateLimitConfig`, `RateLimitError`, `RateLimitObserver`, `RateLimitPermit`, `RateLimitUsage`, `RateLimiter` |
| `service-replay` | `ReplayCompletion`, `ReplayConfig`, `ReplayNote`, `ReplayTrajectory`, `ReplayTurn`, `ReplayUsage`, `SessionReplay` |
| `service-speculative` | `ClientPool`, `Complexity`, `ComplexityEvaluator`, `DraftValidator`, `JsonToolDraftValidator`, `ModelOrchestrator`, `SwappableModelPair`, `TurnExecutor` |
| `service-vector-store` | `Embedder`, `InMemoryVectorStore`, `RECALL_RENDER_MARKER`, `RecallInjection`, `RetrievalReport`, `RetrievedTurn`, `VectorStoreConfig`, `VectorStoreError`, `VectorStoreUsage` |
| `plugin-memory` or `service-prompt-cache` | `CUCA_EXPORT_FORMAT`, `CUCA_EXPORT_VERSION`, `CucaExport`, `CucaExportError`, `CucaImportReport`, `GraphExportSection`, `PromptCacheExportSection` |
| `plugin-cost` and `plugin-telemetry` | `OtelCostObserver` |

### Export envelope constants

| Constant | Type | Value |
|---|---|---|
| `CUCA_EXPORT_FORMAT` | `&str` | `cuca-export` |
| `CUCA_EXPORT_VERSION` | `u32` | `1` |

## Client surface

`CucaClientBuilder` methods, in declaration order:

| Method | Gate |
|---|---|
| `new` | none |
| `with_llamacpp_config` | `provider-llamacpp` |
| `with_local_model` | `provider-local` |
| `with_provider` | none |
| `with_base_url` | none |
| `with_api_key` | none |
| `with_bearer_token` | `provider-anthropic` |
| `with_anthropic_oauth` | `provider-anthropic` |
| `register_plugin` | none |
| `with_orchestrator` | `service-speculative` |
| `with_prompt_cache_config` | `service-prompt-cache` |
| `with_prompt_cache_service` | `service-prompt-cache` |
| `build` | none |

- `with_provider` is required. `build` fails with `CucaError::Config` without it.
- With `ProviderEndpoint::Local`, `build` also fails with `CucaError::Config` unless `with_local_model` attached a model.
- `with_bearer_token` and `with_api_key` are mutually exclusive at dispatch time. When both are set, the bearer token wins.
- `with_prompt_cache_service` takes precedence over `with_prompt_cache_config`. A validation error in the config surfaces from `build`.
- `with_base_url` and `with_api_key` configure this client, not the tier clients of a `ModelOrchestrator`.

`CucaClient` methods, in declaration order:

| Method | Gate |
|---|---|
| `builder` | none |
| `selected_provider` | none |
| `base_url` | none |
| `api_key` | none |
| `bearer_token` | `provider-anthropic` |
| `oauth_config` | `provider-anthropic` |
| `llamacpp_config` | `provider-llamacpp` |
| `local_model` | `provider-local` |
| `orchestrator` | `service-speculative` |
| `prompt_cache` | `service-prompt-cache` |
| `prompt_cache_snapshot` | `service-prompt-cache` |
| `replace_prompt_cache_snapshot` | `service-prompt-cache` |
| `http_client` | any of the seven HTTP `provider-*` features |
| `plugins` | none |
| `generate_stream` | none |
| `generate_stream_with_response` | none |

`generate_stream_with_response` returns the stream and a `ResponseHandle`. After the stream yields `None`, `ResponseHandle::take` returns the aggregated `UnifiedResponse` of that call.

## The re-export contract is tested

`tests/public_exports.rs` imports from `cuca` by name and constructs the types it names, so a removed or renamed re-export fails compilation. Its own gate is `any(provider-openai, provider-llamacpp)`. Its nested modules are gated on `service-prompt-cache`, `service-rate-limit`, `plugin-redaction`, `plugin-memory`, `plugin-cost`, the combination of cost and telemetry, either of memory or prompt cache, `service-replay` and `service-vector-store`.
