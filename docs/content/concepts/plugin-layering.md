+++
title = "Plugins and services"
description = "Two kinds of add-on: a plugin hooks into every turn, a service is something you call yourself or that picks the model."
template = "page.html"
weight = 3
+++

# Plugins and services

<dl class="page-facts">
<dt>In one line</dt>
<dd>A plugin watches the request and the stream through hooks. A service is called directly, or chooses the model. Which one a capability is depends on whether the pipeline can run it without being told to</dd>
<dt>You need</dt>
<dd>Nothing running</dd>
<dt>Read this if</dt>
<dd>You are adding a capability and need to know its kind, or you want to know why <code>register_plugin</code> refuses a service</dd>
</dl>

CUCA has nineteen add-on features besides the providers: twelve plugins and seven services. They solve different problems, so they attach to a turn in different ways.

- A **plugin** implements the `CucaPlugin` trait and registers on the client builder. CUCA calls its hooks at fixed points in every turn.
- A **service** never implements `CucaPlugin`, so registering one is a compile error. You call it yourself, or, for one service, it replaces the step that sends the request.

A hook is a method CUCA calls for you at a set moment, for example just before a request is sent.

## Which one is it?

<div class="dgm">
<div class="dgm-scroll">
<svg viewBox="0 0 660 232" role="img" aria-labelledby="kind-title kind-desc">
  <title id="kind-title">How a capability becomes a plugin or a service</title>
  <desc id="kind-desc">A question: can a fixed point in every turn drive the capability? If yes, it is a plugin that registers with register_plugin and runs through hooks. If no, and you decide when to call it, it is an explicit-call service. If no, and it chooses which model answers, it is the dispatch-replacing service, attached with with_orchestrator.</desc>
  <defs>
    <marker id="kind-arrow" viewBox="0 0 8 8" refX="7" refY="4"
            markerWidth="6" markerHeight="6" orient="auto">
      <path d="M0 0.6 L7 4 L0 7.4 z" fill="var(--muted-foreground)"/>
    </marker>
  </defs>
  <text class="t-title" x="6" y="20">Plugin or service</text>
  <rect class="blk" x="6" y="92" width="176" height="64" rx="6"/>
  <text class="t-sm t-mid" x="94" y="119">Can a fixed point in</text>
  <text class="t-sm t-mid" x="94" y="133">every turn drive it?</text>
  <rect class="blk blk-ctl" x="266" y="36" width="210" height="48" rx="6"/>
  <text class="t-lbl t-mid t-ctl" x="371" y="56">plugin</text>
  <text class="t-sm t-mid" x="371" y="72">register_plugin, hooks</text>
  <rect class="blk blk-ctl" x="266" y="100" width="210" height="48" rx="6"/>
  <text class="t-lbl t-mid t-ctl" x="371" y="120">service</text>
  <text class="t-sm t-mid" x="371" y="136">you call it when you decide</text>
  <rect class="blk blk-ctl" x="266" y="164" width="210" height="48" rx="6"/>
  <text class="t-lbl t-mid t-ctl" x="371" y="184">service</text>
  <text class="t-sm t-mid" x="371" y="200">with_orchestrator</text>
  <path class="arw" d="M184 124 C226 124 222 60 262 60" marker-end="url(#kind-arrow)"/>
  <path class="arw" d="M184 124 H262" marker-end="url(#kind-arrow)"/>
  <path class="arw" d="M184 124 C226 124 222 188 262 188" marker-end="url(#kind-arrow)"/>
  <text class="t-sm t-end" x="228" y="78">yes</text>
  <text class="t-sm t-end" x="244" y="118">no</text>
  <text class="t-sm t-end" x="228" y="182">no</text>
  <text class="t-sm" x="496" y="64">plugin-* (12 features)</text>
  <text class="t-sm" x="496" y="128">six service-* features</text>
  <text class="t-sm" x="496" y="192">service-speculative</text>
</svg>
</div>
<p class="dgm-cap"><b>The kind follows from how the capability is driven.</b> A hook can drive a plugin. A service needs a caller who decides when to run it, or it picks the model itself.</p>
</div>

The same split as a table:

| Kind | Attached by | Members |
|---|---|---|
| Plugin | `register_plugin`, hook | `plugin-mcp`, `plugin-sandbox`, `plugin-memory`, `plugin-guardrails`, `plugin-subagent`, `plugin-hitl`, `plugin-web-search`, `plugin-skills`, `plugin-telemetry`, `plugin-session-log`, `plugin-cost`, `plugin-redaction` |
| Service | explicit call | `service-prompt-cache`, `service-entity-extraction`, `service-replay`, `service-rate-limit`, `service-vector-store`, `service-scheduler` |
| Service | `with_orchestrator`, replaces dispatch | `service-speculative` |

`plugin-session-log` also implements `SessionStorePlugin`, which extends `CucaPlugin` with `append_log`, `replay_session` and `fork_session`.

## The hook contract

`CucaPlugin` has five methods. `name` is required. The other four have default bodies that do nothing and succeed, so a plugin writes only the hooks it uses.

```rust,name=The CucaPlugin trait
fn name(&self) -> &'static str;
fn on_request(&self, req: &mut UnifiedRequest) -> Result<(), PluginError>;
fn execute_local_tool(&self, call: &MessageContentBlock)
    -> Result<Option<MessageContentBlock>, PluginError>;
fn on_stream_chunk(&self, chunk: &mut MessageContentBlock) -> Result<(), PluginError>;
fn on_response_complete(&self, res: &UnifiedResponse) -> Result<(), PluginError>;
```

Plugins run in the order you registered them at every hook. What a plugin's answer does differs per hook:

| Hook | Runs | Errors |
|---|---|---|
| `on_request` | for every plugin, before the request is sent. Each may change the request. | The first error becomes `CucaError::Plugin` and ends the turn before anything is sent. |
| `execute_local_tool` | only for a `ToolCall` block. The first plugin that returns `Some` wins and the rest are skipped. | The result must be a `ToolResult` with the same `tool_call_id`, or the stream yields `PluginError::Validation`. |
| `on_stream_chunk` | for every plugin, once per block. Each may change the block. | The first error is yielded in place of the block. That block is not counted. Polling continues. |
| `on_response_complete` | once, for every plugin, when the stream ends. | The error is not shown to you and does not stop the other plugins. |

The last hook is the one place CUCA swallows an error. The answer has already reached you, so a failing telemetry export must not break a turn you already read. With `plugin-guardrails` compiled in, the error is reported through `tracing`. Otherwise it is dropped.

## Why a service is not a plugin

A service could carry an empty `CucaPlugin` impl and a comment saying "call the methods directly". It would compile, register and do nothing. An inert registration is a bug that looks like configuration, so services do not implement the trait. Each one needs something a hook cannot give:

- `RateLimiter` paces a caller. A permit taken in one hook and released in another would leak when the request fails or the stream is dropped early. `RateLimitPermit` releases itself on every exit.
- `PromptCache` must look up after every `on_request` hook has changed the request, because its key covers the request that is really sent. A hit skips sending, `execute_local_tool` and `on_stream_chunk`, and still runs `on_response_complete` once. No hook fires at that point, so `CucaClient` calls the cache itself.
- `EntityExtractor` produces a graph of facts that does nothing until you merge it into a `MemoryPlugin` with `merge_graph`. No moment in a turn makes the merge obviously right, so you decide.
- `SessionReplay` replays a recorded session. A replay has no live request to change and no arriving block to annotate.
- `Dispatcher` waits for a free slot, and a hook cannot wait. Waiting needs `.await` on the caller's task, so it is a method you await.

## The orchestrator replaces sending

`ModelOrchestrator` does not watch a turn. It decides which model runs it. With one attached, `generate_stream` runs the `on_request` hooks and the prompt cache lookup, then hands the turn to `execute_adaptive_turn` instead of a provider.

The orchestrator's stream does its own validation, fallback and session events. The top-level `on_stream_chunk` and `on_response_complete` hooks do not run over an orchestrator turn. The exception is a configured prompt cache that misses, because the cache write needs to know the turn ended. [Speculative fast/slow routing](@/concepts/speculative-routing.md) covers the routing itself.

## How the features depend on each other

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
<p class="dgm-cap"><b>Core sits below every plugin and every service.</b> A service may depend on the plugin features it declares. Nothing depends upward.</p>
</div>

Plugins do not depend on each other and never name a service, so you can turn on any one plugin alone. Three services need a plugin feature, and Cargo turns it on for you:

```toml,name=The three service to plugin edges in Cargo.toml
service-entity-extraction = ["plugin-memory"]
service-replay = ["plugin-session-log"]
service-vector-store = ["plugin-memory", "dep:wide"]
```

The other four services need no plugin. `ModelOrchestrator::with_session_store` exists only when `plugin-session-log` is also on, and then records `SessionEvent::ModelSwap`.

Next page: [Speculative fast/slow routing](@/concepts/speculative-routing.md).
