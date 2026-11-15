+++
title = "The public API surface"
description = "Every public module and every item re-exported at the crate root, grouped by the feature that turns it on, plus the request, client, config, server and local model types."
template = "page.html"
weight = 4
+++

# The public API surface

<dl class="page-facts">
<dt>In one line</dt>
<dd>Ten modules that are always present, three that need a feature, and 203 items re-exported at the crate root: 17 always, 186 behind a feature</dd>
<dt>You need</dt>
<dd>Nothing; this page mirrors <code>src/lib.rs</code></dd>
<dt>Read this if</dt>
<dd>You want the import path of a type, or the feature that makes it exist</dd>
</dl>

`src/lib.rs` starts with `#![forbid(unsafe_code)]`, so no `unsafe` block compiles anywhere in the crate. A build with no provider feature stops with a `compile_error!`; see [The feature matrix](@/reference/features.md).

## Modules

| Module | Needs | Holds |
|---|---|---|
| `types` | nothing | the wire types shared by every provider: `MessageContentBlock`, `MessageRole`, `ProviderEndpoint`, `ToolDefinition`, `UnifiedMessage` |
| `error` | nothing | `CucaError` and `PluginError`; see [Error types](@/reference/errors.md) |
| `request` | nothing | the normalized request and response types |
| `session` | nothing | the append-only session record types |
| `sse` | nothing | the SSE parser: `SseEvent`, `SseStreamParser` |
| `plugin` | nothing | the `CucaPlugin` and `SessionStorePlugin` traits |
| `plugins` | nothing | plugin implementations; each submodule needs its `plugin-*` feature |
| `services` | nothing | service implementations; each submodule needs its `service-*` feature |
| `config` | nothing; the KDL reader needs `config-kdl` | field adapters for settings structs, and the KDL reader; see [Configuration files](@/reference/environment.md) |
| `client` | nothing | the builder, the client and the plugin-instrumented stream |
| `export` | `plugin-memory` or `service-prompt-cache` | the versioned `cuca-export` envelope |
| `cost_otel` | `plugin-cost` and `plugin-telemetry` | the cost ledger to OpenTelemetry bridge |
| `tls` | any HTTP provider, `provider-local` or `plugin-web-search` | `ensure_crypto_provider`, which installs `ring` as the process-default rustls provider unless one is set |
| `sysmon` | `sysmon` (`server` turns it on) | `SystemSampler` and its snapshot types `SystemSnapshot`, `CpuStats`, `MemoryStats`, `DiskStats`, `DiskIo`, `GpuStats`, `ProcessStats` |
| `server` | `server` | `cuca serve`, `cuca balance` and `cuca tui`, and `CucaConfig`; see [Server](@/server/_index.md) |

The provider adapters are in a private module. Use them through `ProviderEndpoint` and `CucaClientBuilder`.

## Items always available

17 items are re-exported at the crate root in every build.

| Group | Items |
|---|---|
| Client | `CucaClient`, `CucaClientBuilder`, `ResponseHandle` |
| Errors | `CucaError`, `PluginError` |
| Request and response | `AgentResponseStream`, `PromptCacheBreakpoint`, `PromptCacheDirective`, `PromptCacheUsage`, `ThinkingConfig`, `ThinkingEffort`, `ThinkingParams`, `TokenUsage`, `UnifiedRequest`, `UnifiedResponse` |
| Session | `SessionEvent`, `SessionRecord` |

`PromptCacheBreakpoint`, `PromptCacheDirective` and `PromptCacheUsage` need no feature because they are fields of `UnifiedRequest` and `UnifiedResponse`.

These items are reached by path, not from the root:

| Path | Items |
|---|---|
| `cuca::types` | `MessageContentBlock`, `MessageRole`, `ProviderEndpoint`, `ToolDefinition`, `UnifiedMessage` |
| `cuca::sse` | `SseEvent`, `SseStreamParser` |
| `cuca::plugin` | `CucaPlugin`, `SessionStorePlugin` |
| `cuca::services::orchestrator` | `PoolTurnExecutor` |

## Items behind a feature

186 items in 22 groups, re-exported at the crate root.

### Providers

| Needs | Items |
|---|---|
| `provider-local` | `BatchingConfig`, `ConvertConfig`, `DownloadProgress`, `DownloadProgressCallback`, `DraftStats`, `DrafterSource`, `EngramCache`, `EngramQuant`, `EngramStats`, `ExpertHeat`, `ExpertPlacement`, `ExpertTierStats`, `ExpertTiers`, `GenerationReport`, `GpuAdapter`, `LocalCapabilities`, `LocalDevice`, `LocalGeneration`, `LocalModel`, `LocalModelConfig`, `ModelPart`, `ModelPreset`, `ModelSource`, `ModelTemplate`, `PresetPart`, `PresetTemplate`, `SamplingConfig`, `ServingStats`, `WeightQuant` |

### Plugins

| Needs | Items |
|---|---|
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

### Services

| Needs | Items |
|---|---|
| `service-entity-extraction` | `CandidateEntity`, `CandidateRelationship`, `EntityExtractionCandidate`, `EntityExtractionModel`, `EntityExtractionReport`, `EntityExtractionSchema`, `EntityExtractor`, `EntityReference`, `EntityTable`, `PropertyColumn`, `PropertyType`, `RelationshipTable` |
| `service-prompt-cache` | `PromptCache`, `PromptCacheConfig`, `PromptCacheEntry`, `PromptCacheError`, `PromptCacheImportReport`, `PromptCacheSnapshot` |
| `service-rate-limit` | `RateLimitConfig`, `RateLimitError`, `RateLimitObserver`, `RateLimitPermit`, `RateLimitUsage`, `RateLimiter` |
| `service-replay` | `ReplayCompletion`, `ReplayConfig`, `ReplayNote`, `ReplayTrajectory`, `ReplayTurn`, `ReplayUsage`, `SessionReplay` |
| `service-scheduler` | `Backend`, `BackendClient`, `BackendId`, `BackendRegistry`, `BackendSpec`, `BucketSnapshot`, `CallMeta`, `CallOutcome`, `CallReport`, `CallSlot`, `ChangeReason`, `ClientFactory`, `Clock`, `CompletedCall`, `Completion`, `Direction`, `DispatchError`, `DispatchGate`, `Dispatcher`, `DispatcherConfig`, `KindTotals`, `KindTotalsEntry`, `Lane`, `LevelChange`, `Load`, `Metric`, `OpenGate`, `Priority`, `RunningCall`, `Scheduler`, `SchedulerConfig`, `SchedulerError`, `SchedulerPool`, `SchedulerSnapshot`, `SystemClock`, `TunerPhase`, `TunerSnapshot`, `WaitingCall` |
| `service-speculative` | `ClientPool`, `Complexity`, `ComplexityEvaluator`, `DraftValidator`, `JsonToolDraftValidator`, `ModelOrchestrator`, `SwappableModelPair`, `TurnExecutor` |
| `service-vector-store` | `Embedder`, `InMemoryVectorStore`, `RECALL_RENDER_MARKER`, `RecallInjection`, `RetrievalReport`, `RetrievedTurn`, `VectorStoreConfig`, `VectorStoreError`, `VectorStoreUsage` |

### Combinations

| Needs | Items |
|---|---|
| `plugin-memory` or `service-prompt-cache` | `CUCA_EXPORT_FORMAT`, `CUCA_EXPORT_VERSION`, `CucaExport`, `CucaExportError`, `CucaImportReport`, `GraphExportSection`, `PromptCacheExportSection` |
| `plugin-cost` and `plugin-telemetry` | `OtelCostObserver` |

The export envelope constants are `CUCA_EXPORT_FORMAT` (`"cuca-export"`) and `CUCA_EXPORT_VERSION` (`1`).

## Request type

`UnifiedRequest` is one request for every provider. Its public fields are `model`, `provider`, `messages`, `temperature`, `max_tokens`, `top_p`, `top_k`, `seed`, `stop`, `stream`, `thinking`, `tools` and `prompt_cache`.

| Field | Type | Default from `new` |
|---|---|---|
| `temperature` | `Option<f32>` | `None` |
| `max_tokens` | `Option<u32>` | `None` |
| `top_p` | `Option<f32>` | `None` |
| `top_k` | `Option<u32>` | `None`; `0` turns the cut off |
| `seed` | `Option<u64>` | `None` |
| `stop` | `Vec<String>` | empty |
| `stream` | `bool` | `true` |
| `provider` | `ProviderEndpoint` | `Custom("")`; `generate_stream` sets it |

Methods, in declaration order: `new`, `add_tool`, `with_prompt_cache`, `add_system_message`, `add_user_message`, `add_message`, `set_temperature`, `set_max_tokens`, `set_top_p`, `set_top_k`, `set_seed`, `add_stop`, `with_stream`, `with_thinking`, `enable_thinking`, `disable_thinking`. `message_count` reads the message list. `add_stop` adds one entry per call.

A backend that cannot honor a field fails the request with `CucaError::Config`. `top_k` fails on OpenAI, on the DeepSeek native route and on `Custom`. `seed` fails on Anthropic and on DeepSeek's Anthropic bridge. See [Error types](@/reference/errors.md#config-messages).

`ProviderEndpoint` has nine variants: `LlamaCpp`, `Vllm`, `LmStudio`, `OpenAi` (the default), `Anthropic`, `GoogleGemini`, `DeepSeek`, `Local` and `Custom(String)`.

## Client types

`CucaClientBuilder` methods, in declaration order:

| Method | Needs |
|---|---|
| `new` | nothing |
| `with_llamacpp_config` | `provider-llamacpp` |
| `with_local_model` | `provider-local` |
| `with_provider` | nothing |
| `with_base_url` | nothing |
| `with_api_key` | nothing |
| `with_bearer_token` | `provider-anthropic` |
| `with_anthropic_oauth` | `provider-anthropic` |
| `register_plugin` | nothing |
| `with_orchestrator` | `service-speculative` |
| `with_prompt_cache_config` | `service-prompt-cache` |
| `with_prompt_cache_service` | `service-prompt-cache` |
| `build` | nothing |

- `with_provider` is required. `build` fails with `CucaError::Config` without it.
- With `ProviderEndpoint::Local`, `build` also fails unless `with_local_model` attached a model.
- When both are set, `with_bearer_token` wins over `with_api_key`.
- `with_prompt_cache_service` wins over `with_prompt_cache_config`. A validation error in the config surfaces from `build`.
- `with_base_url` and `with_api_key` configure this client, not the tier clients of a `ModelOrchestrator`.

`CucaClient` methods, in declaration order:

| Method | Needs |
|---|---|
| `builder` | nothing |
| `selected_provider` | nothing |
| `base_url` | nothing |
| `api_key` | nothing |
| `bearer_token` | `provider-anthropic` |
| `oauth_config` | `provider-anthropic` |
| `llamacpp_config` | `provider-llamacpp` |
| `local_model` | `provider-local` |
| `orchestrator` | `service-speculative` |
| `prompt_cache` | `service-prompt-cache` |
| `prompt_cache_snapshot` | `service-prompt-cache` |
| `replace_prompt_cache_snapshot` | `service-prompt-cache` |
| `http_client` | any of the seven HTTP `provider-*` features |
| `plugins` | nothing |
| `generate_stream` | nothing |
| `generate_stream_with_response` | nothing |

`generate_stream_with_response` returns the stream and a `ResponseHandle`. After the stream yields `None`, `ResponseHandle::take` returns the full `UnifiedResponse` of that call.

## The config module

`cuca::config` is not re-exported at the crate root.

| Item | Needs | What it is |
|---|---|---|
| `ConfigValue` | nothing | `serde_json::Value`, the tree a KDL document becomes |
| `one_or_many` | nothing | a `deserialize_with` adapter: a list written once or several times |
| `from_str` | nothing | a `deserialize_with` adapter: a string parsed with `FromStr` |
| `size` | nothing | a `deserialize_with` adapter: a byte count (an integer, `512m` or `4g`) |
| `duration` | nothing | a `deserialize_with` adapter: a duration (`500ms`, `60s`, `5m`, `1h`) |
| `from_kdl_str`, `from_kdl_str_with_vars` | `config-kdl` | KDL text to a `ConfigValue`, without or with placeholder substitution |
| `load_kdl_str`, `load_kdl_file` | `config-kdl` | read a whole file into any `DeserializeOwned` type; every error is `CucaError::Config` |

## Server types

With `server`, `cuca::server` exports `CucaConfig`, `LogConfig`, `LogFormat`, `ServerConfig`, `ModelSpec`, `BalancerConfig`, `ServerError`, `MAX_MODELS` (16), `serve`, `balance` and `init_logging`.

| Type | What it holds |
|---|---|
| `CucaConfig` | fields `log`, `serve`, `balance`; methods `load`, `from_kdl_str`, `into_serve`, `into_balance`; unknown keys are rejected |
| `LogConfig` | `format` (`LogFormat::Pretty` by default, or `Json`) and `level` (`info` by default) |
| `ServerConfig` | `listen`, `models`, `api_key`, `stats_api`, `latency_window`; `new`, `validate`, `DEFAULT_LISTEN`, `DEFAULT_LATENCY_WINDOW` |
| `ModelSpec` | `name` and `config` (a `LocalModelConfig`); `new`, `source_label`, `device_label` |
| `BalancerConfig` | `listen`, `upstreams`, `upstream_api_key`, `api_key`, `policy`, `health_interval`, `latency_window`, `stats_api` |

## Local model types

With `provider-local`, the root re-exports are the whole public path to the engine. Each page of [Local (in-process)](@/providers/local/_index.md) uses these types. The meaning of each configuration field is on [Configuration](@/providers/local/configuration.md). Every type that a `LocalModelConfig` holds implements `Deserialize`, so the KDL keys are the field names; see [The model keys](@/reference/environment.md#the-model-keys).

| Type | What it holds |
|---|---|
| `LocalModelConfig` | fields `source`, `cache_dir`, `hf_token`, `context_length`, `sampling`, `chat_template`, `chat_template_file`, `device`, `drafter`, `draft_tokens`, `on_download_progress`, `experts`, `convert`, `batching`, `shards`, `gpu_layers`, `engram`, `shard_dirs` |
| `LocalModel` | `load`, `generate`, `capabilities`, `sampling`, `context_length`, `kv_cache_bytes`, `weights_file_bytes`, `layer_split`, `draft_stats`, `serving_stats`, `expert_tier_stats`, `expert_heat`, `engram_stats` |
| `LocalGeneration`, `GenerationReport` | the stream of one `LocalModel::generate` request (no client, no plugin hooks) and its `report()`: `usage`, `prompt_cache`, `finish_reason` (`stop` or `length`), `queue_secs` |
| `ModelSource` | variants `HuggingFace { repo, revision, file, sha256, parts, template }` and `File(PathBuf)` name a GGUF file; `Checkpoint { repo, revision }` and `Dir(PathBuf)` name a Hugging Face safetensors checkpoint, converted once to a cached GGUF (see [Model sources](@/providers/local/model-sources.md)); also `PRESETS`, `preset`, `default_model`, `hugging_face` and `FromStr` |
| `ModelPreset` | `name`, `repo`, `revision`, `file`, `size_bytes`, `architecture`, `drafter`, `parts`, `template`; method `source` |
| `ModelPart`, `PresetPart` | one part of a split GGUF: `file` and `sha256` (the digest is optional in `ModelPart`); `PresetPart` also has `size_bytes` |
| `ModelTemplate`, `PresetTemplate` | a chat template file: `repo`, `revision`, `file`, `sha256` (optional in `ModelTemplate`) |
| `DrafterSource` | `File(ModelSource)` or `InTrunk`; `From<ModelSource>` |
| `ConvertConfig`, `WeightQuant`, `EngramQuant` | `ConvertConfig { quant, engram_quant }`; `WeightQuant` is `F16`, `Q8_0` (default) or `Q4_0`; `EngramQuant` is `Q8_0` (default), `Q4_0` or `Bf16` and applies to DeepSeek-V4.1 only. Both have `name` and `FromStr`. A non-default `engram_quant` adds `-engram-<name>` to the cached file name |
| `EngramCache` | `ram_bytes` (4 GiB default, 1 MiB to 1 TiB) and `reader_threads` (16 default, 1 to 64); `validate` |
| `BatchingConfig` | `max_sequences` (4), `kv_tokens`, `max_step_tokens`, `prefix_cache` (true) |
| `ExpertPlacement`, `ExpertTiers` | `Resident` (default), `Tiered(ExpertTiers)`, `Host { layers }` |
| `SamplingConfig` | `temperature`, `top_k`, `top_p`, `seed`, all `Option`; `DEFAULT_TEMPERATURE` (1.0), `DEFAULT_TOP_K` (64), `DEFAULT_TOP_P` (0.95) |
| `LocalDevice`, `GpuAdapter` | `Cpu`, `Vulkan(GpuAdapter)`, `Metal(GpuAdapter)`, `Cuda(usize)`, `Rocm(usize)`; `GpuAdapter` is `Default`, `Discrete(usize)` or `Integrated(usize)`; `required_feature`, `FromStr` and `Display` (its inverse) |
| `LocalCapabilities`, `DraftStats`, `ServingStats`, `ExpertTierStats`, `ExpertHeat` | capability and statistics reports; `DraftStats::acceptance_rate` |
| `EngramStats` | `lookups`, `cache_hits`, `disk_reads`, `bytes_read`, `resident_rows`, `resident_bytes`, `capacity_rows`; `hit_rate`. `LocalModel::engram_stats` is `None` for a model without an Engram table |
| `DownloadProgress`, `DownloadProgressCallback` | `downloaded_bytes`, `total_bytes`; `DownloadProgressCallback::new` |
