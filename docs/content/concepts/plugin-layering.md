+++
title = "Plugins and services"
description = "Two tiers and three attachment mechanisms: every plugin hooks the pipeline, every service is called directly or replaces the dispatch stage."
template = "page.html"
weight = 3
+++

# Plugins and services

<dl class="page-facts">
<dt>In one line</dt>
<dd>A plugin observes the request and stream pipeline; a service is called directly. Attachment follows from whether the pipeline can drive the capability without being told to</dd>
<dt>You need</dt>
<dd>Nothing running</dd>
<dt>Read this if</dt>
<dd>You are adding a capability and need its tier and attachment point, or you want to know why <code>register_plugin</code> refuses a service</dd>
</dl>

CUCA has nineteen non-provider features: twelve plugins and seven services. Every plugin implements `CucaPlugin` and registers on the builder. A service never implements `CucaPlugin`, so registering one is a compile error. Six services are called directly, and one replaces the dispatch stage.

Tier and attachment are separate questions. The tier decides the feature namespace and the dependency rules. The attachment is how the client drives the capability.

## Tier and attachment

| Tier | Attachment | Members |
|---|---|---|
| Plugin | hook, through `register_plugin` | `plugin-mcp`, `plugin-sandbox`, `plugin-memory`, `plugin-guardrails`, `plugin-subagent`, `plugin-hitl`, `plugin-web-search`, `plugin-skills`, `plugin-telemetry`, `plugin-session-log`, `plugin-cost`, `plugin-redaction` |
| Service | explicit call | `service-prompt-cache`, `service-entity-extraction`, `service-replay`, `service-rate-limit`, `service-vector-store`, `service-scheduler` |
| Service | dispatch replacement, through `with_orchestrator` | `service-speculative` |

- A hook fires at a fixed point in every turn, in registration order. The pipeline drives a plugin.
- An explicit-call service runs when the caller decides, and the caller applies the result. The pipeline has no moment that calls it.
- The orchestrator owns the dispatch stage, so the pipeline hosts nothing there.

`plugin-session-log` also implements `SessionStorePlugin`, which extends `CucaPlugin` with `append_log`, `replay_session` and `fork_session`.

## The layers

<div class="dgm">
<div class="dgm-scroll">
<svg viewBox="0 0 660 150" role="img" aria-labelledby="layer-title layer-desc">
  <title id="layer-title">Dependency direction between services, plugins and core</title>
  <desc id="layer-desc">Three blocks, left to right: seven services, twelve plugins, and core. An arrow runs from services to plugins, labeled as declared features only, and an arrow runs from plugins to core, labeled as core only. No arrow runs between plugins or from a plugin back to a service.</desc>
  <defs>
    <marker id="layer-arrow" viewBox="0 0 8 8" refX="7" refY="4"
            markerWidth="6" markerHeight="6" orient="auto">
      <path d="M0 0.6 L7 4 L0 7.4 z" fill="var(--muted-foreground)"/>
    </marker>
  </defs>
  <text class="t-title" x="6" y="20">Dependencies point left to right only</text>
  <rect class="blk blk-ctl" x="6" y="44" width="188" height="66" rx="6"/>
  <text class="t-sm t-mid t-ctl" x="100" y="68">services (7)</text>
  <text class="t-sm t-mid" x="100" y="84">explicit call</text>
  <text class="t-sm t-mid" x="100" y="98">or dispatch replacement</text>
  <rect class="blk blk-ctl" x="236" y="44" width="188" height="66" rx="6"/>
  <text class="t-sm t-mid t-ctl" x="330" y="68">plugins (12)</text>
  <text class="t-sm t-mid" x="330" y="84">flat tier</text>
  <text class="t-sm t-mid" x="330" y="98">no plugin needs another</text>
  <rect class="blk blk-data" x="466" y="44" width="188" height="66" rx="6"/>
  <text class="t-sm t-mid t-data" x="560" y="68">core</text>
  <text class="t-sm t-mid" x="560" y="84">types, request, error, sse,</text>
  <text class="t-sm t-mid" x="560" y="98">plugin, client, provider</text>
  <path class="arw arw-ctl" d="M196 77 H232" marker-end="url(#layer-arrow)"/>
  <path class="arw arw-data" d="M426 77 H462" marker-end="url(#layer-arrow)"/>
  <text class="t-sm t-mid" x="214" y="132">declared features only</text>
  <text class="t-sm t-mid" x="444" y="132">core only</text>
</svg>
</div>
<p class="dgm-cap"><b>Core sits below every plugin and every service.</b> Services may depend on the plugin features they declare, and nothing depends upward.</p>
</div>

## The hook contract

`CucaPlugin` has five methods. `name` is required. The other four have default bodies that return success and ignore their inputs.

```rust,name=The CucaPlugin trait surface
fn name(&self) -> &'static str;
fn on_request(&self, req: &mut UnifiedRequest) -> Result<(), PluginError>;
fn execute_local_tool(&self, call: &MessageContentBlock)
    -> Result<Option<MessageContentBlock>, PluginError>;
fn on_stream_chunk(&self, chunk: &mut MessageContentBlock) -> Result<(), PluginError>;
fn on_response_complete(&self, res: &UnifiedResponse) -> Result<(), PluginError>;
```

A plugin implements only the hooks it uses. `Send + Sync` is a supertrait so the plugin list can cross `await` points in the async pipeline.

Order is registration order at every hook site. What differs is what a plugin's answer does:

- `on_request` runs for every plugin in order, and each may mutate the request. The first error becomes `CucaError::Plugin` and ends the turn before dispatch.
- `execute_local_tool` runs only for a `ToolCall` block. The first plugin to return `Some` wins and the rest are skipped. The block must be a `ToolResult` whose `tool_call_id` matches the call, or the stream yields a `PluginError::Validation` error.
- `on_stream_chunk` runs for every plugin in order, per block. The first error is yielded in place of the block, which is neither accumulated nor token counted, and polling continues.
- `on_response_complete` runs once when the stream ends, for every plugin. An error never reaches the consumer and never stops the other plugins' hooks.

The last hook is the one place the pipeline swallows an error. The response has already been delivered, so a failing telemetry export must not break a turn the caller consumed. The client reports the error through `tracing` when `plugin-guardrails` is compiled in and drops it otherwise.

## Why a service is not a plugin

A service could carry a `CucaPlugin` impl with empty hook bodies and a comment saying "call the methods directly". That would compile, register and do nothing. An inert registration is a bug that looks like configuration, so services do not implement the trait.

- `RateLimiter` paces a caller, and `on_request` is synchronous, so a hook could only reject a request. A permit taken in a hook and released in `on_response_complete` would leak on a dispatch error, an early stream drop, or the orchestrator's unwrapped stream, none of which reaches a terminal hook. `RateLimitPermit` is an RAII value that releases on every exit.
- `PromptCache` must look up after `on_request` has finished mutating the request, because the digest covers the request that is actually sent. A hit skips provider dispatch, `execute_local_tool` and `on_stream_chunk`, and still runs `on_response_complete` once. No hook fires at the lookup position, so `CucaClient` calls the cache itself.
- `EntityExtractor` produces a graph delta that has no effect until the application merges it into a `MemoryPlugin` with `merge_graph`. No moment in a turn makes applying the delta obviously right, so the caller decides. The feature declares `service-entity-extraction = ["plugin-memory"]` because the delta is a `MemoryGraph`.
- `SessionReplay` drives a recorded session. A turn has no live request to mutate and no arriving chunk to annotate, and no hook signature returns a stream. Its work is caller-triggered backend reads through `SessionReplay::load` and its streaming methods. The feature declares `service-replay = ["plugin-session-log"]` because it reads a trajectory through the `SessionBackend` seam.
- `Dispatcher` waits for a slot, and a hook cannot wait. A queue that orders calls from many callers needs `.await` on the caller's task, so the capability is a method the caller awaits.

## The orchestrator replaces dispatch

`ModelOrchestrator` does not instrument a turn. It decides which model runs it. When one is attached, `generate_stream` runs the `on_request` hooks and the prompt cache lookup, then hands the turn to `execute_adaptive_turn` instead of a provider.

`OrchestratorStream` carries its own validation, fallback and session events. The top-level `on_stream_chunk` and `on_response_complete` hooks do not run over an orchestrator turn, except when a prompt cache is configured and misses, because the cache write needs a completion point.

The tier executors run through pooled `CucaClient` instances built without `with_orchestrator`. Their `generate_stream` calls go straight to a provider and cannot re-enter the orchestrator. [Speculative fast/slow routing](@/concepts/speculative-routing.md) covers the routing itself.

## What the tiers forbid

Core sits below every plugin and every service. The plugin tier is flat: no plugin depends on another plugin in Cargo features, in `#[cfg]` and import edges, or at runtime. Services may depend on the plugin features they declare. `Cargo.toml` has three hard edges, all from a service to a plugin.

```toml,name=The three hard service feature edges in Cargo.toml
service-entity-extraction = ["plugin-memory"]
service-replay = ["plugin-session-log"]
service-vector-store = ["plugin-memory", "dep:wide"]
```

- `plugin-memory` and `plugin-session-log` build and work with no knowledge that a dependent service exists.
- `service-speculative` has no Cargo edge to `plugin-session-log`. `ModelOrchestrator::with_session_store` exists only when `plugin-session-log` is also enabled, and then records `SessionEvent::ModelSwap`.
- `service-scheduler` declares no plugin dependency; it owns `tokio` and `tracing` itself.
- `service-prompt-cache` and `service-rate-limit` declare no plugin dependency.

A plugin never names a service in any form: no `use crate::services`, no `cfg(feature = "service-...")`, no runtime lookup. The CI `plugin_layering` job greps `src/plugins/` for `service-`, `crate::services`, and `plugins::` paths ending in `prompt_cache`, `entity_extraction`, `replay` or `rate_limit`. A `cfg(all(...))` reverse edge compiles in both the solo and all-features builds, so only inspection catches it. The `plugin_solo` and `service_solo` jobs build each feature alone on `provider-openai`, which catches undeclared forward edges.

Next page: [Speculative fast/slow routing](@/concepts/speculative-routing.md).
