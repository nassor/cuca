+++
title = "Everything is a plugin"
description = "Three attachment points, one hook order, and why two capabilities are forbidden from implementing CucaPlugin at all."
template = "page.html"
weight = 3
+++

# Everything is a plugin

<dl class="page-facts">
<dt>In one line</dt>
<dd>A capability attaches in one of three ways, and which way is decided by whether the pipeline can drive it</dd>
<dt>You need</dt>
<dd>Nothing running</dd>
<dt>Read this if</dt>
<dd>You are adding a capability and need to know which attachment point it belongs to, or why <code>register_plugin</code> refuses one of the fourteen</dd>
</dl>

Fourteen `plugin-*` features. Eleven of them implement `CucaPlugin` and register on
the builder. Two of them are compile errors if you try. One replaces the
dispatch stage outright.

That is not an inconsistency, it is the layering. A capability's attachment
point follows from a single question: can the client's own pipeline call it at
the right moment without being told to?

## Three attachment points

| Tier | Mechanism | The pipeline can | Members |
|---|---|---|---|
| Hook plugin | `register_plugin(Arc<dyn CucaPlugin>)` | drive it: hooks fire at fixed points, in registration order | `plugin-mcp`, `plugin-sandbox`, `plugin-memory`, `plugin-guardrails`, `plugin-subagent`, `plugin-hitl`, `plugin-web-search`, `plugin-skills`, `plugin-telemetry`, `plugin-session-log`, `plugin-cost` |
| Explicit-call capability | direct method calls on the type | not drive it: the caller decides when and applies the result | `plugin-prompt-cache`, `plugin-entity-extraction` |
| Pipeline replacement | `with_orchestrator(ModelOrchestrator)` | not host it: it owns the dispatch stage | `plugin-speculative` |

`plugin-session-log` sits in the first tier twice: it implements `CucaPlugin`
for the hooks and `SessionStorePlugin` for the append, replay and fork calls the
orchestrator makes.

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

## Why two capabilities may not be plugins

`PromptCache` and `EntityExtractionPlugin` do not implement `CucaPlugin`, and the
compiler therefore rejects `register_plugin` on them. That refusal is the point.

A cache lookup has to happen after `on_request` has finished mutating the
request, because the digest must cover the request that will actually be sent.
It also has to bypass most of the rest of the pipeline on a hit: provider
dispatch, `execute_local_tool` and `on_stream_chunk` are all skipped, while
`on_response_complete` still fires once so a session log or a metrics exporter
records the replayed turn. No hook fires in the lookup position, so
`CucaClient` calls the cache itself.

Entity extraction produces a graph delta that has no effect until the
application merges it into a `MemoryPlugin`. There is no moment in a turn where
"apply this delta" is the obviously correct action, so the decision stays with
the caller.

Both could have been given a `CucaPlugin` impl with empty hook bodies and a
comment saying "call the methods directly". That would compile, register
successfully, and do nothing, which is worse than not compiling. An inert
registration is a bug that looks like configuration.

## Why the orchestrator is not a plugin either

`ModelOrchestrator` does not instrument a turn, it decides which model runs it.
When one is attached, `generate_stream` runs the `on_request` hooks and then
hands the whole turn to `execute_adaptive_turn` instead of dispatching to a
provider.

The recursion guard is worth naming. The orchestrator's tier executors run
through pooled `CucaClient` instances built without `with_orchestrator`, so
their own `generate_stream` calls dispatch straight to a provider and cannot
re-enter the orchestrator. See
[Speculative fast/slow routing](@/concepts/speculative-routing.md).

## What the layering forbids

Plugins are peers, not a dependency graph, and the build enforces that. There is
exactly one cross-plugin feature edge in `Cargo.toml`:

```toml,name=The only plugin-to-plugin feature edge in Cargo.toml
plugin-entity-extraction = ["plugin-memory"]
```

It points one way. `plugin-memory` compiles and works with no knowledge that
entity extraction exists, and CI greps `src/plugins/memory/` to keep it that
way. A second grep asserts that no file under `src/plugins/` gates on
`plugin-speculative` or imports `crate::orchestrator`, which is what keeps the
third tier from leaking into the first.

Next page: [Speculative fast/slow routing](@/concepts/speculative-routing.md).
