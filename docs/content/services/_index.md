+++
title = "Services"
description = "Seven optional features you call from your own code: rate limit, prompt cache, scheduler, speculative routing, replay, entity extraction and vector store."
template = "section.html"
sort_by = "weight"

[extra]
kicker = "Reference"
+++

<dl class="page-facts">
<dt>In one line</dt>
<dd>A service is a feature you call with plain method calls. It never runs by itself inside a request.</dd>
<dt>You need</dt>
<dd>The <code>service-*</code> features you want. None is on by default.</dd>
<dt>Read this if</dt>
<dd>You want to pace requests, cache replies, queue calls, route between models, replay a session, extract entities or recall old turns, and you need the feature name, entry type and limits.</dd>
</dl>

## Plugin or service

A plugin hooks into the request pipeline: CUCA calls its hooks on every request and every reply block, in order. A service sits outside the pipeline: your code creates it and calls its methods when it wants the result.

<div class="dgm">
<div class="dgm-scroll">
<svg viewBox="0 0 660 250" role="img" aria-labelledby="svc-hub-title svc-hub-desc">
  <title id="svc-hub-title">Services depend on a plugin in four cases and on nothing in three.</title>
  <desc id="svc-hub-desc">Two plugins sit on top: plugin-memory and plugin-session-log. Below them are the seven services. Entity extraction and vector store point up to plugin-memory. Replay points up to plugin-session-log. Speculative points up to plugin-session-log with an optional edge. Prompt cache, rate limit and scheduler have no arrow.</desc>
  <defs>
    <marker id="svc-hub-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M0 0.6 L7 4 L0 7.4 z" fill="var(--muted-foreground)"/>
    </marker>
  </defs>
  <text class="t-title" x="6" y="20">Plugins: hooks run on every request</text>
  <rect class="blk blk-ctl" x="40" y="34" width="120" height="40" rx="6"/>
  <text class="t-sm t-mid t-ctl" x="100" y="58">plugin-memory</text>
  <rect class="blk blk-ctl" x="219" y="34" width="130" height="40" rx="6"/>
  <text class="t-sm t-mid t-ctl" x="284" y="58">plugin-session-log</text>
  <text class="t-sm" x="400" y="50">Your code never calls a plugin's</text>
  <text class="t-sm" x="400" y="64">hooks. CUCA runs them for you.</text>
  <path class="arw" d="M54 164 V80" marker-end="url(#svc-hub-arrow)"/>
  <path class="arw" d="M146 164 V80" marker-end="url(#svc-hub-arrow)"/>
  <path class="arw" d="M238 164 V80" marker-end="url(#svc-hub-arrow)"/>
  <path class="arw arw-ctl" d="M330 164 V80" marker-end="url(#svc-hub-arrow)"/>
  <text class="t-sm" x="338" y="126">optional</text>
  <text class="t-title" x="6" y="150">Services: your code calls them</text>
  <rect class="blk" x="13" y="164" width="82" height="44" rx="6"/>
  <text class="t-sm t-mid" x="54" y="183">entity-</text>
  <text class="t-sm t-mid" x="54" y="196">extraction</text>
  <rect class="blk" x="105" y="164" width="82" height="44" rx="6"/>
  <text class="t-sm t-mid" x="146" y="183">vector-</text>
  <text class="t-sm t-mid" x="146" y="196">store</text>
  <rect class="blk" x="197" y="164" width="82" height="44" rx="6"/>
  <text class="t-sm t-mid" x="238" y="190">replay</text>
  <rect class="blk" x="289" y="164" width="82" height="44" rx="6"/>
  <text class="t-sm t-mid" x="330" y="190">speculative</text>
  <rect class="blk" x="381" y="164" width="82" height="44" rx="6"/>
  <text class="t-sm t-mid" x="422" y="183">prompt-</text>
  <text class="t-sm t-mid" x="422" y="196">cache</text>
  <rect class="blk" x="473" y="164" width="82" height="44" rx="6"/>
  <text class="t-sm t-mid" x="514" y="183">rate-</text>
  <text class="t-sm t-mid" x="514" y="196">limit</text>
  <rect class="blk" x="565" y="164" width="82" height="44" rx="6"/>
  <text class="t-sm t-mid" x="606" y="190">scheduler</text>
  <path class="ln" d="M381 222 H647"/>
  <text class="t-sm t-mid" x="514" y="240">no plugin needed</text>
</svg>
</div>
<p class="dgm-cap"><b>Four services build on a plugin, three stand alone.</b> An arrow means the service needs that plugin's feature. The speculative arrow is optional: it only adds a session log record of each model swap.</p>
</div>

[Plugins and services](@/concepts/plugin-layering.md) explains why the two tiers are separate.

## Turn on a service

Name each service feature when you add the crate. A service needs one `provider-*` feature to talk to a model.

```bash,name=Runs the same on all three platforms
cargo add cuca --features "provider-llamacpp,service-rate-limit"
```

Passing a service to `register_plugin` is a compile error. Each service builds alone with one provider feature.

## The seven services

- [Entity extraction](@/services/entity-extraction.md): turn text into a checked graph of entities. Feature `service-entity-extraction`, entry type `EntityExtractor`, needs `plugin-memory`.
- [Speculative](@/services/speculative.md): answer easy turns on a fast model and escalate to a slow one. Feature `service-speculative`, entry type `ModelOrchestrator`, optionally uses `plugin-session-log`.
- [Prompt cache](@/services/prompt-cache.md): skip the provider for a request you already answered. Feature `service-prompt-cache`, entry type `PromptCache`, needs no plugin.
- [Replay](@/services/replay.md): play a recorded session back with no provider call. Feature `service-replay`, entry type `SessionReplay`, needs `plugin-session-log`.
- [Rate limit](@/services/rate-limit.md): stay inside a provider's request quota. Feature `service-rate-limit`, entry type `RateLimiter`, needs no plugin.
- [Vector store](@/services/vector-store.md): recall old turns by similarity. Feature `service-vector-store`, entry type `InMemoryVectorStore`, needs `plugin-memory`.
- [Scheduler](@/services/scheduler.md): order many callers over one or more model backends. Feature `service-scheduler`, entry type `Dispatcher`, needs no plugin.

## Rules every service follows

- A service never implements `CucaPlugin`, and no plugin names a service.
- A service may depend on a plugin feature. It declares that in `Cargo.toml` as `service-<name> = ["plugin-<other>"]`.
- No service depends on another service.
- Data crosses between tiers through a value you pass in, such as an `Arc<dyn Trait>`, or through a step your application performs.
- Every service has bounds, and each page lists them with the policy at the bound. [Memory discipline](@/concepts/memory-discipline.md) shows them all in one place.

## Feature edges to the plugin tier

`Cargo.toml` declares three edges from a service to a plugin:

```toml,name=Every service-to-plugin feature edge in Cargo.toml
service-entity-extraction = ["plugin-memory"]
service-replay = ["plugin-session-log"]
service-vector-store = ["plugin-memory", "dep:wide"]
```

- Entity extraction needs `plugin-memory` because its result is a `MemoryGraph` that you merge into the memory plugin's graph.
- Replay needs `plugin-session-log` because it reads a recorded session through the session log's `SessionBackend`.
- Vector store needs `plugin-memory` because it implements the memory plugin's `VectorStore` offload trait.
- Speculative has no feature edge. `ModelOrchestrator::with_session_store` exists only when `plugin-session-log` is also on.
- Prompt cache, rate limit and scheduler need no plugin.
