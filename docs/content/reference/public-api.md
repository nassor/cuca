+++
title = "The public API surface"
description = "Every public module and every top-level re-export from src/lib.rs, with the feature gating each one."
template = "page.html"
weight = 4
+++

# The public API surface

<dl class="page-facts">
<dt>In one line</dt>
<dd>Twelve modules and roughly ninety re-exported items, grouped by the feature that gates them</dd>
<dt>You need</dt>
<dd>Nothing; this page mirrors <code>src/lib.rs</code></dd>
<dt>Read this if</dt>
<dd>You are looking for the import path of a type, or which feature makes it exist</dd>
</dl>

`src/lib.rs` opens with `#![forbid(unsafe_code)]`, so no `unsafe` block compiles
anywhere in the crate.

## Modules

| Module | Visibility | Gate | Purpose |
|---|---|---|---|
| `types` | `pub` | none | core unified wire types shared by every provider adapter |
| `error` | `pub` | none | client-facing and plugin-facing error types |
| `request` | `pub` | none | normalized request and response contracts |
| `session` | `pub` | none | append-only session audit-trail model |
| `sse` | `pub` | none | the SSE stream parser engine |
| `plugin` | `pub` | none | the plugin trait layer |
| `plugins` | `pub` | none | plugin implementations, one gated submodule per `plugin-*` feature |
| `services` | `pub` | none | service implementations, one gated submodule per `service-*` feature |
| `client` | `pub` | none | the builder, the client, and the plugin-instrumented stream pipeline |
| `provider` | `pub(crate)` | none | provider adapter layer; not part of the public surface |
| `export` | `pub` | `plugin-memory` or `service-prompt-cache` | the versioned `cuca-export` envelope |
| `cost_otel` | `pub` | `plugin-cost` and `plugin-telemetry` | the cost ledger to OpenTelemetry bridge |

## Ungated re-exports

Available in every build that compiles at all.

| Group | Items |
|---|---|
| Client | `CucaClient`, `CucaClientBuilder` |
| Errors | `CucaError`, `PluginError` |
| Request and response | `AgentResponseStream`, `PromptCacheBreakpoint`, `PromptCacheDirective`, `PromptCacheUsage`, `ThinkingConfig`, `ThinkingEffort`, `ThinkingParams`, `UnifiedRequest`, `UnifiedResponse` |
| Session | `SessionEvent`, `SessionRecord` |

`PromptCacheBreakpoint`, `PromptCacheDirective` and `PromptCacheUsage` are
ungated even though `service-prompt-cache` is not: they are fields of
`UnifiedRequest` and `UnifiedResponse`, so they exist wherever those do. The
cache itself is gated.

Types reachable through `cuca::types` rather than the crate root:
`MessageContentBlock`, `MessageRole`, `ProviderEndpoint`, `ToolDefinition`,
`UnifiedMessage`. Types reachable through `cuca::sse`: `SseEvent`,
`SseStreamParser`. Through `cuca::plugin`: `CucaPlugin`,
`SessionStorePlugin`.

## Feature-gated re-exports

| Gate | Items |
|---|---|
| `plugin-cost` | `CostConfig`, `CostEntry`, `CostObserver`, `CostPlugin`, `CostUsage`, `ModelRates`, `PricingResolver`, `PricingTable`, `UnpricedModelPolicy` |
| `plugin-guardrails` | `JsonGuardrailPlugin` |
| `plugin-hitl` | `ApprovalChannel`, `ApprovalDecision`, `ApprovalRequest`, `HitlPlugin`, `OneshotApprovalChannel`, `Risk` |
| `plugin-mcp` | `McpPlugin`, `McpTransport` |
| `plugin-memory` | `Budget`, `CompactionStrategy`, `CompressionAction`, `CompressionReport`, `ContextUsage`, `ContextUsageObserver`, `ContextWindowResolver`, `GraphContextConfig`, `GraphDirection`, `GraphImportReport`, `GraphNode`, `GraphRelationship`, `GraphSnapshot`, `MemoryConfig`, `MemoryGraph`, `MemoryPlugin`, `MergePolicy`, `MergeReport`, `Summarizer`, `VectorStore` |
| `service-entity-extraction` | `CandidateEntity`, `CandidateRelationship`, `EntityExtractionCandidate`, `EntityExtractionModel`, `EntityExtractor`, `EntityExtractionReport`, `EntityExtractionSchema`, `EntityReference`, `EntityTable`, `PropertyColumn`, `PropertyType`, `RelationshipTable` |
| `service-prompt-cache` | `PromptCache`, `PromptCacheConfig`, `PromptCacheEntry`, `PromptCacheError`, `PromptCacheImportReport`, `PromptCacheSnapshot` |
| `plugin-sandbox` | `SandboxConfig`, `SandboxPlugin`, `SandboxResult` |
| `plugin-session-log` | `FileBackend`, `InMemoryBackend`, `SessionBackend`, `SessionLogPlugin` |
| `plugin-skills` | `Skill`, `SkillsConfig`, `SkillsPlugin` |
| `service-speculative` | `ClientPool`, `Complexity`, `ComplexityEvaluator`, `DraftValidator`, `JsonToolDraftValidator`, `ModelOrchestrator`, `SwappableModelPair`, `TurnExecutor` |
| `plugin-subagent` | `SubagentPlugin`, `SubagentResult`, `SubagentRunner`, `SubagentSpec`, `WorktreeConfig` |
| `plugin-telemetry` | `OpenTelemetryPlugin` |
| `plugin-web-search` | `SearchResult`, `WebSearchConfig`, `WebSearchPlugin`, `WebSearchProvider` |
| `service-rate-limit` | `RateLimiter`, `RateLimitConfig`, `RateLimitPermit`, `RateLimitUsage`, `RateLimitObserver`, `RateLimitError` |
| `plugin-memory` or `service-prompt-cache` | `CUCA_EXPORT_FORMAT`, `CUCA_EXPORT_VERSION`, `CucaExport`, `CucaExportError`, `CucaImportReport`, `GraphExportSection`, `PromptCacheExportSection` |
| `plugin-cost` and `plugin-telemetry` | `OtelCostObserver` |

`PoolTurnExecutor` is `pub` inside `crate::services::orchestrator` but is not
re-exported at the crate root.

## The export envelope constants

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

`with_provider` is required; `build` fails without it. `with_bearer_token` and
`with_api_key` are mutually exclusive at dispatch time, and the bearer token
wins. `with_prompt_cache_service` takes precedence over
`with_prompt_cache_config`.

`CucaClient` methods:

| Method | Gate |
|---|---|
| `builder` | none |
| `selected_provider` | none |
| `base_url` | none |
| `api_key` | none |
| `plugins` | none |
| `generate_stream` | none |
| `bearer_token` | `provider-anthropic` |
| `oauth_config` | `provider-anthropic` |
| `llamacpp_config` | `provider-llamacpp` |
| `orchestrator` | `service-speculative` |
| `prompt_cache` | `service-prompt-cache` |
| `prompt_cache_snapshot` | `service-prompt-cache` |
| `replace_prompt_cache_snapshot` | `service-prompt-cache` |
| `http_client` | any of the seven `provider-*` features |

## The re-export contract is tested

`tests/public_exports.rs` imports from `cuca` by name and constructs the
types it names, so a removed or renamed re-export fails compilation rather than
silently disappearing. Its own gate is
`any(provider-openai, provider-llamacpp)`, with nested modules gated on
`service-prompt-cache`, `plugin-memory`, `plugin-cost`, the two combinations of
prompt-cache and memory, and the combination of cost and telemetry.

Next page: [What CUCA is](@/_index.md), back at the top.
