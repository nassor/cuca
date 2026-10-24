+++
title = "Plugins"
description = "Twelve plugin features, all hook plugins, with no dependency between any of them."
template = "section.html"
sort_by = "weight"

[extra]
kicker = "Reference"
+++

<dl class="page-facts">
<dt>In one line</dt>
<dd>Twelve <code>plugin-*</code> features; each one implements <code>CucaPlugin</code> and no plugin depends on another.</dd>
<dt>You need</dt>
<dd>The plugin features you want, named explicitly; none is enabled by default.</dd>
<dt>Read this if</dt>
<dd>You need a plugin's feature flag, entry type, hooks, config defaults or caps.</dd>
</dl>

## The twelve features

Every plugin attaches with `CucaClientBuilder::register_plugin(Arc<dyn CucaPlugin>)`. The last column lists the hooks where the plugin acts; no bundled plugin overrides `execute_local_tool`.

| Feature | Entry type | Hooks that act |
|---|---|---|
| [`plugin-mcp`](@/plugins/mcp.md) | `McpPlugin` | `on_stream_chunk` |
| [`plugin-sandbox`](@/plugins/sandbox.md) | `SandboxPlugin` | `on_stream_chunk` |
| [`plugin-memory`](@/plugins/memory.md) | `MemoryPlugin` | `on_request` |
| [`plugin-guardrails`](@/plugins/guardrails.md) | `JsonGuardrailPlugin` | `on_stream_chunk` |
| [`plugin-subagent`](@/plugins/subagent.md) | `SubagentPlugin` | `on_stream_chunk` |
| [`plugin-hitl`](@/plugins/hitl.md) | `HitlPlugin` | `on_stream_chunk` |
| [`plugin-web-search`](@/plugins/web-search.md) | `WebSearchPlugin` | `on_stream_chunk` |
| [`plugin-skills`](@/plugins/skills.md) | `SkillsPlugin` | `on_request`, `on_stream_chunk` |
| [`plugin-telemetry`](@/plugins/telemetry.md) | `OpenTelemetryPlugin` | `on_request`, `on_stream_chunk`, `on_response_complete` |
| [`plugin-session-log`](@/plugins/session-log.md) | `SessionLogPlugin` | `on_request`, `on_stream_chunk`, `on_response_complete` |
| [`plugin-cost`](@/plugins/cost.md) | `CostPlugin` | `on_request`, `on_response_complete` |
| [`plugin-redaction`](@/plugins/redaction.md) | `RedactionPlugin` | `on_request` |

`plugin-session-log` also implements `SessionStorePlugin`, which adds `append_log`, `replay_session` and `fork_session`.

### Headings on every plugin page

Each plugin page answers the same questions under the same headings, in this order:

1. Feature and entry type
2. Hooks
3. Configuration
4. Caps and at-cap policy
5. Errors
6. Example
7. Related pages

## Hook order and error handling

Plugins run in registration order at every hook site. A plugin must not require a position relative to another plugin.

| Hook | Fires | On the first error |
|---|---|---|
| `on_request` | Once per turn, before dispatch; may mutate the request. | The turn fails before dispatch; later plugins' hooks do not run. |
| `execute_local_tool` | Per `ToolCall` block; the first plugin to return `Some` wins. | The stream yields the error in place of the block. |
| `on_stream_chunk` | Once per block; may mutate the block. | The stream yields the error in place of the block; the block is neither accumulated nor token counted. |
| `on_response_complete` | Once, when the stream ends. | Not propagated; the next plugin's hook still runs. |

- A value returned from `execute_local_tool` must be a `ToolResult` whose `tool_call_id` equals the incoming `ToolCall` id. A mismatch is `PluginError::Validation` with schema `local tool result`.
- A `plugin-guardrails` build logs a failed `on_response_complete` through `tracing` under target `cuca::client`; other builds drop it.
- A prompt cache hit under `service-prompt-cache` replays the stored blocks, skips `execute_local_tool` and `on_stream_chunk`, and still runs every `on_response_complete`.
- Recoverable conditions belong in the block itself: guardrails, MCP, sandbox and the other tool plugins replace a `ToolCall` with a `ToolResult` instead of returning an error.

## Caps and at-cap policies

Every plugin page states its caps as bound, policy at the bound and usage gauge, in that order. The reasoning behind refusing rather than evicting is [Memory discipline](@/concepts/memory-discipline.md).

| Plugin | Bound | At the bound | Usage gauge |
|---|---|---|---|
| `plugin-guardrails` | `MAX_TRACKED_CALLS`, 4096 | Evict the oldest, in insertion order. | `tracked_calls()` |
| `plugin-subagent` | `DEFAULT_MAX_PENDING`, 1024 | Refuse the spawn. | `pending_len()` |
| `plugin-subagent` | `MAX_SPAWN_LOG`, 4096 | Evict the oldest. | `spawns()` length |
| `plugin-hitl` | `DEFAULT_MAX_AUDIT_ENTRIES`, 65536 | Refuse the gated call. | `audit_len()` |
| `plugin-session-log` | `InMemoryBackend::DEFAULT_MAX_RECORDS`, 65536 | Refuse the append or fork. | `len()` against `max_records()` |
| `plugin-cost` | `CostConfig::max_tracked_models`, default 64 | Fold into one overflow bucket; totals stay exact. | `usage()`, `breakdown()` |
| `plugin-memory` | Message list: the active `Budget` | Run the `strategies` pipeline. | `ContextUsage` readings, `count_tokens()` |
| `plugin-memory` | Working graph: none | The caller owns the bound. | `snapshot()`, `MemoryGraph::len()` |
| `plugin-redaction` | See the list below | Reject at construction, or refuse the request. | `rule_count()`, `match_cap()` |
| `plugin-sandbox` | See the list below | Trap that call. | `last_diagnostic()` |
| `plugin-mcp`, `plugin-skills`, `plugin-telemetry`, `plugin-web-search` | None: nothing grows with traffic | Not applicable | None |

`plugin-redaction` bounds:

- `RedactionConfig::MAX_RULES`, 256 rules per policy.
- `MAX_PATTERN_BYTES`, 512 per literal or prefix; `MAX_KIND_BYTES`, 32 per `kind`.
- `max_matches_per_text`, default 1024, at most `MAX_MATCHES_PER_TEXT` (4096). A string over it fails with `PluginError::HookFailure` instead of being truncated.
- `MAX_JSON_DEPTH`, 64 levels of `ToolCall` arguments, also refused with `HookFailure`.
- An empty or over-cap policy is rejected at construction.

`plugin-sandbox` bounds, per call:

- `max_memory_bytes`, 64 MiB.
- `max_instructions`, 1000000.
- `timeout_ms`, 5000.
- Guest output, 8 MiB, not configurable.
- Nothing accumulates across calls, so no gauge is needed.

## No edges between plugins

The plugin tier is flat. `Cargo.toml` declares no feature edge between plugins, and no plugin references another in code or at runtime.

- Every `plugin-*` feature depends on core only and builds alone on any one provider.
- A [service](@/services/_index.md) may depend on a plugin: `service-entity-extraction` and `service-vector-store` require `plugin-memory`, and `service-replay` requires `plugin-session-log`. A plugin never depends on, or names, a service.
- Services never implement `CucaPlugin`; they are explicit-call, and registering one with `register_plugin` is a compile error.
- The CI job `plugin_layering` greps every file under `src/plugins/` for `service-`, for `crate::services`, and for `plugins::` paths to `prompt_cache`, `entity_extraction`, `replay` and `rate_limit`. A `cfg(all(...))` reverse edge compiles in both the solo and all-features builds, so only that inspection catches it.

[Plugins and services](@/concepts/plugin-layering.md) explains the layering.

## Shared optional dependencies

Eight optional crates are enabled by more than one feature. A shared crate is a shared dependency, not a feature edge between plugins.

| Crate | Enabled by |
|---|---|
| `reqwest` with `rustls` | The seven HTTP `provider-*` features, `provider-local` and `plugin-web-search` |
| `tokio` | `provider-local`, `plugin-mcp`, `plugin-subagent`, `plugin-hitl`, `plugin-web-search`, `service-speculative`, `service-rate-limit`, `service-scheduler` |
| `tokio-stream` | The seven HTTP `provider-*` features and `service-speculative` |
| `tracing` | `plugin-guardrails`, `plugin-telemetry`, `plugin-redaction`, `service-scheduler` |
| `base64` | `provider-anthropic`, `plugin-sandbox` |
| `sha2` | `provider-anthropic`, `provider-local`, `service-prompt-cache` |
| `postcard` | `plugin-session-log`, `service-prompt-cache` |
| `tiktoken-rs` | `plugin-memory`, `plugin-cost` |

`plugin-web-search` enables `reqwest/rustls` itself, so it runs HTTPS searches in a build where no provider feature contributes TLS.

## Related pages

- [The feature matrix](@/reference/features.md): every feature and its dependencies.
- [Error types](@/reference/errors.md): `PluginError` and `CucaError`.
- [Write a custom plugin](@/guides/custom-plugin.md): implement `CucaPlugin` yourself.

## Per-plugin pages
