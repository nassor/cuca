+++
title = "Plugins"
description = "Seventeen plugin features across three attachment points: twelve hook plugins, four explicit-call capabilities, one pipeline replacement."
template = "section.html"
sort_by = "weight"

[extra]
kicker = "Reference"
+++

<dl class="page-facts">
<dt>In one line</dt>
<dd>Seventeen <code>plugin-*</code> features; twelve implement <code>CucaPlugin</code>, four are called directly, one replaces the dispatch stage</dd>
<dt>You need</dt>
<dd>The plugin features you want, named explicitly; none is enabled by default</dd>
<dt>Read this if</dt>
<dd>You need a plugin's feature flag, entry type, hooks, config defaults or caps</dd>
</dl>

## The seventeen features

| Feature | Entry type | Attachment | Hooks overridden |
|---|---|---|---|
| [`plugin-mcp`](@/plugins/mcp.md) | `McpPlugin` | `register_plugin` | `on_request`, `on_stream_chunk`, `on_response_complete` |
| [`plugin-sandbox`](@/plugins/sandbox.md) | `SandboxPlugin` | `register_plugin` | `on_request`, `on_stream_chunk`, `on_response_complete` |
| [`plugin-memory`](@/plugins/memory.md) | `MemoryPlugin` | `register_plugin` | `on_request` |
| [`plugin-entity-extraction`](@/plugins/entity-extraction.md) | `EntityExtractionPlugin` | direct calls only | none; not a `CucaPlugin` |
| [`plugin-guardrails`](@/plugins/guardrails.md) | `JsonGuardrailPlugin` | `register_plugin` | `on_stream_chunk`, `on_response_complete` |
| [`plugin-subagent`](@/plugins/subagent.md) | `SubagentPlugin` | `register_plugin` | `on_request`, `on_stream_chunk`, `on_response_complete` |
| [`plugin-hitl`](@/plugins/hitl.md) | `HitlPlugin` | `register_plugin` | `on_stream_chunk` |
| [`plugin-web-search`](@/plugins/web-search.md) | `WebSearchPlugin` | `register_plugin` | `on_stream_chunk` |
| [`plugin-skills`](@/plugins/skills.md) | `SkillsPlugin` | `register_plugin` | `on_request`, `on_stream_chunk` |
| [`plugin-telemetry`](@/plugins/telemetry.md) | `OpenTelemetryPlugin` | `register_plugin` | `on_request`, `on_stream_chunk`, `on_response_complete` |
| [`plugin-speculative`](@/plugins/speculative.md) | `ModelOrchestrator` | `with_orchestrator` | none; replaces the dispatch stage |
| [`plugin-session-log`](@/plugins/session-log.md) | `SessionLogPlugin` | `register_plugin` | `on_request`, `on_stream_chunk`, `on_response_complete` |
| [`plugin-replay`](@/plugins/replay.md) | `SessionReplay` | direct calls only | none; not a `CucaPlugin` |
| [`plugin-prompt-cache`](@/plugins/prompt-cache.md) | `PromptCache` | `with_prompt_cache_config` or `with_prompt_cache_service` | none; not a `CucaPlugin` |
| [`plugin-cost`](@/plugins/cost.md) | `CostPlugin` | `register_plugin` | `on_request`, `on_response_complete` |
| [`plugin-rate-limit`](@/plugins/rate-limit.md) | `RateLimiter` | direct calls only | none; not a `CucaPlugin` |
| [`plugin-redaction`](@/plugins/redaction.md) | `RedactionPlugin` | `register_plugin` | `on_request` |

`plugin-session-log` also implements `SessionStorePlugin`, which adds
`append_log`, `replay_session` and `fork_session`.

`plugin-speculative` has no submodule under `src/plugins/`. It is declared as
`pub mod orchestrator` in `src/lib.rs` and lives entirely in
`src/orchestrator.rs`.

## Hook order and error handling

Plugins run in registration order at every hook site. What a plugin's answer
does differs per hook:

| Hook | Fires | On the first error |
|---|---|---|
| `on_request` | once per turn, before dispatch, may mutate the request | the turn fails before dispatch |
| `execute_local_tool` | per `ToolCall` block; the first plugin to return a value wins | the stream fails |
| `on_stream_chunk` | once per block, may mutate the block | the stream yields the error; the block is neither accumulated nor token counted |
| `on_response_complete` | exactly once, when the stream ends | logged, not propagated |

A value returned from `execute_local_tool` must be a `ToolResult` whose
`tool_call_id` equals the incoming `ToolCall` id. A mismatch is
`PluginError::Validation` with schema `local tool result`.

## Caps and at-cap policies

Every plugin page states its caps as bound, at-cap policy, usage gauge, in that
order. The consolidated view, and the reasoning behind refusing rather than
evicting, is [Memory discipline](@/concepts/memory-discipline.md).

| Plugin | Bound | At-cap policy | Usage gauge |
|---|---|---|---|
| `plugin-guardrails` | `MAX_TRACKED_CALLS`, `4096` | evict oldest, insertion order | `tracked_calls()` |
| `plugin-subagent` | `DEFAULT_MAX_PENDING`, `1024` | refuse the spawn | `pending_len()` |
| `plugin-subagent` | `MAX_SPAWN_LOG`, `4096` | evict oldest | length of `spawns()` |
| `plugin-hitl` | `DEFAULT_MAX_AUDIT_ENTRIES`, `65_536` | refuse the gated call | `audit_len()` |
| `plugin-session-log` | `InMemoryBackend::DEFAULT_MAX_RECORDS`, `65_536` | refuse the append or fork | `len()`, against `max_records()` |
| `plugin-prompt-cache` | `PromptCacheConfig::capacity`, caller-set | evict least recently used | `len()`, against `capacity()` |
| `plugin-cost` | `CostConfig::max_tracked_models`, caller-set (default 64) | fold into one overflow bucket; totals stay exact | `usage()`, `breakdown()` |
| `plugin-rate-limit` | `RateLimitConfig::max_concurrent`, caller-set | callers wait for a free permit | `RateLimitUsage::in_flight` |
| `plugin-rate-limit` | `RateLimitConfig::max_waiters`, caller-set | reject the acquire (`QueueFull`), never queue further | `RateLimitUsage::waiting` |
| `plugin-redaction` | `RedactionConfig::MAX_RULES`, `256` rules, each pattern `MAX_PATTERN_BYTES` (512), each `kind` `MAX_KIND_BYTES` (32); per-call match buffer capped at `max_matches_per_text` (≤ `MAX_MATCHES_PER_TEXT`, 4096) | reject an over-cap or empty policy at construction; refuse (`PluginError::HookFailure`) a string over the per-call match cap instead of truncating | `rule_count()`, `match_cap()` |
| `plugin-sandbox` | per call: 64 MiB memory, 1000000 instructions, 5000 ms, 8 MiB output | trap that call | none; nothing accumulates |
| `plugin-mcp`, `plugin-skills`, `plugin-telemetry`, `plugin-web-search` | no traffic-growing structure | not applicable | none |
| `plugin-memory` | the working graph has no internal cap | the caller owns the bound | `snapshot()` |
| `plugin-speculative` | `ClientPool` is uncapped, bounded by configuration | no eviction | `ClientPool::len()` |

## The cross-plugin edges

Plugins are peers. `Cargo.toml` declares exactly two feature edges between
them, and both point one way:

```toml,name=The two plugin-to-plugin feature edges in Cargo.toml
plugin-entity-extraction = ["plugin-memory"]
plugin-replay = ["plugin-session-log"]
```

`plugin-memory` and `plugin-session-log` compile and run with no knowledge
that their derived peer exists. CI enforces the direction with three greps:
nothing under `src/plugins/memory/` may reference entity extraction, nothing
in `src/plugins/session_log.rs` may reference replay, and no file under
`src/plugins/` may gate on `plugin-speculative` or import `crate::orchestrator`.

## Shared optional dependencies

Six optional crates are enabled by more than one feature, deliberately rather
than accidentally:

| Crate | Enabled by |
|---|---|
| `reqwest` with `rustls` | all seven `provider-*` features, and `plugin-web-search` |
| `tokio` | `plugin-mcp`, `plugin-subagent`, `plugin-hitl`, `plugin-web-search`, `plugin-speculative`, `plugin-rate-limit` |
| `tracing` | `plugin-guardrails`, `plugin-telemetry`, `plugin-redaction` |
| `base64` | `provider-anthropic`, `plugin-sandbox` |
| `sha2` | `provider-anthropic`, `plugin-prompt-cache` |
| `tiktoken-rs` | `plugin-memory`, `plugin-cost` |

`plugin-web-search` enabling `reqwest/rustls` itself is what lets it run HTTPS
searches in a build with no provider feature contributing TLS.

## Per-plugin pages
