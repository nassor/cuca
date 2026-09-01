+++
title = "Plugins and services"
description = "Two tiers and three attachment mechanisms: every plugin hooks the pipeline, every service is called directly or replaces the dispatch stage."
template = "page.html"
weight = 3
+++

# Plugins and services

<dl class="page-facts">
<dt>In one line</dt>
<dd>A plugin observes the request/stream pipeline; a service is called directly. Attachment follows from whether the pipeline can drive the capability without being told to</dd>
<dt>You need</dt>
<dd>Nothing running</dd>
<dt>Read this if</dt>
<dd>You are adding a capability and need to know which tier and attachment point it belongs to, or why <code>register_plugin</code> refuses one of the seventeen</dd>
</dl>

Seventeen non-provider features: twelve plugins, five services. Every plugin
implements `CucaPlugin` and registers on the builder. Every service is a
compile error if you try: four call their capability directly, and one
replaces the dispatch stage outright.

That is not an inconsistency, it is the architecture. A plugin implements
`CucaPlugin` and observes the pipeline; a service is an explicit-call,
client-level capability driven by direct method calls and never implements
`CucaPlugin`. Tier and attachment mechanism are independent questions: tier is
the feature namespace and the dependency rules, attachment is how the client
drives the capability.

## Tier and attachment

| Tier | Mechanism | The pipeline can | Members |
|---|---|---|---|
| Plugin | hook (`register_plugin`) | drive it: hooks fire at fixed points, in registration order | `plugin-mcp`, `plugin-sandbox`, `plugin-memory`, `plugin-guardrails`, `plugin-subagent`, `plugin-hitl`, `plugin-web-search`, `plugin-skills`, `plugin-telemetry`, `plugin-session-log`, `plugin-cost`, `plugin-redaction` |
| Service | explicit call | not drive it: the caller decides when and applies the result | `service-prompt-cache`, `service-entity-extraction`, `service-replay`, `service-rate-limit` |
| Service | pipeline replacement (`with_orchestrator`) | not host it: it owns the dispatch stage | `service-speculative` |

`plugin-session-log` sits in the first row twice: it implements `CucaPlugin`
for the hooks and `SessionStorePlugin` for the append, replay and fork calls
`service-speculative`'s orchestrator makes.

## The hook contract

`CucaPlugin` has five methods. One is required, four have default bodies that
return success and ignore their inputs:

```rust,name=The CucaPlugin trait surface
fn name(&self) -> &'static str;
fn on_request(&self, req: &mut UnifiedRequest) -> Result<(), PluginError>;
fn execute_local_tool(&self, call: &MessageContentBlock)
    -> Result<Option<MessageContentBlock>, PluginError>;
fn on_stream_chunk(&self, chunk: &mut MessageContentBlock) -> Result<(), PluginError>;
fn on_response_complete(&self, res: &UnifiedResponse) -> Result<(), PluginError>;
```

Defaulted bodies mean a plugin implements only the hooks it uses, and most
implement one or two. `Send + Sync` is a supertrait so the plugin list can cross
`await` points in the async pipeline.

Order is registration order, and it is the same order at every hook site. What
differs is what a plugin's answer does:

- `on_request`: runs for every plugin in order, and each may mutate the request.
  The first error short-circuits the whole turn before dispatch.
- `execute_local_tool`: runs only for a `ToolCall` block. The first plugin to
  return a value wins and the rest are skipped. The returned block must be a
  `ToolResult` whose `tool_call_id` matches the call, or the stream fails.
- `on_stream_chunk`: runs for every plugin in order, per block. The first error
  wins, and the failed block is neither accumulated nor token-counted.
- `on_response_complete`: runs exactly once when the stream ends, for every
  plugin. Errors here are logged, not propagated, because the response has
  already been delivered and there is nothing left to fail.

The asymmetry in that last line is the only place the pipeline swallows an
error, and it is deliberate: a telemetry export that fails must not retroactively
break a turn the caller already consumed.

## Why they may not be plugins

`RateLimiter`, `PromptCache`, `EntityExtractor`, and `SessionReplay` do not
implement `CucaPlugin`, and the compiler therefore rejects `register_plugin` on
them. That refusal is the point.

`RateLimiter` cannot be a plugin because `on_request` is synchronous, so a hook
could only reject a request, never pace it, and pacing is the entire
capability. Acquiring a concurrency permit in a hook and releasing it in
`on_response_complete` would also leak the permit on a dispatch error, an early
stream drop, or the orchestrator's unwrapped stream, three real exits that
never reach a terminal hook. An RAII permit whose `Drop` runs on every one of
those exits is what closes the gap.

`PromptCache` has to run after `on_request` has finished mutating the request,
because the digest must cover the request that will actually be sent. It also
has to bypass most of the rest of the pipeline on a hit: provider dispatch,
`execute_local_tool` and `on_stream_chunk` are all skipped, while
`on_response_complete` still fires once so a session log or a metrics exporter
records the replayed turn. No hook fires in the lookup position, so
`CucaClient` calls the cache itself.

`EntityExtractor` produces a graph delta that has no effect until the
application merges it into a `MemoryPlugin`. There is no moment in a turn where
"apply this delta" is the obviously correct action, so the decision stays with
the caller. It also declares a hard dependency on `plugin-memory`
(`service-entity-extraction = ["plugin-memory"]`): the delta it produces is a
`MemoryGraph`, so enabling extraction enables memory with it.

`SessionReplay` refuses for a third reason: it drives a recorded session
instead of observing a live one. There is no live request to mutate
(`on_request`), no arriving chunk to annotate (`on_stream_chunk`), and no hook
signature can return a stream, which is the entire shape of what replay
produces. Its work is caller-triggered backend reads, at a moment with no
relation to any request in flight, so `SessionReplay::load` and its streaming
methods are called directly instead. Its own hard dependency,
`service-replay = ["plugin-session-log"]`, is why the feature lives in the
service tier: it reads a recorded trajectory through the session log's
`SessionBackend` seam.

All four could have been given a `CucaPlugin` impl with empty hook bodies and a
comment saying "call the methods directly". That would compile, register
successfully, and do nothing, which is worse than not compiling. An inert
registration is a bug that looks like configuration.

## Why the orchestrator is a service, not a plugin

`ModelOrchestrator` does not instrument a turn, it decides which model runs it.
When one is attached, `generate_stream` runs the `on_request` hooks and then
hands the whole turn to `execute_adaptive_turn` instead of dispatching to a
provider.

The recursion guard is worth naming. The orchestrator's tier executors run
through pooled `CucaClient` instances built without `with_orchestrator`, so
their own `generate_stream` calls dispatch straight to a provider and cannot
re-enter the orchestrator. See
[Speculative fast/slow routing](@/concepts/speculative-routing.md).

## What the tiers forbid

Core sits below every plugin and every service. The plugin tier is flat: no
plugin depends on another plugin, in Cargo features, in `#[cfg]`/import edges,
or at runtime. Services may depend on the plugin features they declare, and
`Cargo.toml` has exactly two hard edges, both service to plugin:

```toml,name=The two hard service feature edges in Cargo.toml
service-entity-extraction = ["plugin-memory"]
service-replay = ["plugin-session-log"]
```

`plugin-memory` and `plugin-session-log` compile and work with no knowledge
that a dependent service exists. `service-speculative` adds a third,
documented-optional edge to `plugin-session-log`:
`ModelOrchestrator::with_session_store` is compiled out of existence without
it, and records `SessionEvent::ModelSwap` with it. `service-prompt-cache` and
`service-rate-limit` declare no plugin dependency at all.

A plugin must never name a service, in any form: no `use crate::services::…`,
no `cfg(feature = "service-…")`, no runtime lookup. CI's `plugin_layering` job
greps every directory under `src/plugins/` for a `service-` feature name, a
`crate::services` path, or one of the `plugins::` paths the five moved modules
left behind, since a `cfg(all(…))` reverse edge would otherwise compile in both
the solo and all-features builds with nothing to catch it.

Next page: [Speculative fast/slow routing](@/concepts/speculative-routing.md).
