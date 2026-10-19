+++
title = "Services"
description = "Six explicit-call service features that never implement CucaPlugin, and the plugin feature each one depends on."
template = "section.html"
sort_by = "weight"

[extra]
kicker = "Reference"
+++

<dl class="page-facts">
<dt>In one line</dt>
<dd>Six <code>service-*</code> features. Each is driven by direct method calls or a client builder attachment, never by plugin hooks.</dd>
<dt>You need</dt>
<dd>The service features you want, named explicitly. None is enabled by default.</dd>
<dt>Read this if</dt>
<dd>You need a service's feature flag, entry type, plugin dependency, configuration, caps or errors.</dd>
</dl>

## The six features

| Feature | Entry type | Required plugin feature |
|---|---|---|
| [`service-entity-extraction`](@/services/entity-extraction.md) | `EntityExtractor` | `plugin-memory`, hard |
| [`service-speculative`](@/services/speculative.md) | `ModelOrchestrator` | `plugin-session-log`, documented-optional |
| [`service-prompt-cache`](@/services/prompt-cache.md) | `PromptCache` | none |
| [`service-replay`](@/services/replay.md) | `SessionReplay` | `plugin-session-log`, hard |
| [`service-rate-limit`](@/services/rate-limit.md) | `RateLimiter` | none |
| [`service-vector-store`](@/services/vector-store.md) | `InMemoryVectorStore` | `plugin-memory`, hard |

## Service tier rules

- A service is an explicit-call, client-level capability. The caller drives it with method calls, or attaches it to a client builder (`with_orchestrator`, `with_prompt_cache_config`, `with_prompt_cache_service`).
- A service never implements `CucaPlugin`. Passing one to `register_plugin` is a compile error, not an inert registration.
- A service may depend on core and on the plugin features it declares in `Cargo.toml`. A plugin never names a service: no `use`, no `cfg(feature = "service-...")`, no feature edge, no runtime lookup.
- Every code edge from a service to a plugin is declared as `service-<name> = ["plugin-<other>"]`.
- No service depends on another service.
- Plugins and services do not look each other up at runtime. Data crosses tiers through a caller-injected `Arc<dyn Trait>` or an explicit hand-off by the application.
- Each service feature builds alone on one provider. The `service_solo` CI job checks it.

[Plugins and services](@/concepts/plugin-layering.md) explains why the tiers are shaped this way.

## Feature edges to the plugin tier

`Cargo.toml` declares three hard edges, all from a service to a plugin:

```toml,name=Every service-to-plugin feature edge in Cargo.toml
service-entity-extraction = ["plugin-memory"]
service-replay = ["plugin-session-log"]
service-vector-store = ["plugin-memory", "dep:wide"]
```

- `entity-extraction` to `memory`, hard: the extraction delta is a `MemoryGraph` that the caller merges into the memory plugin's working graph.
- `replay` to `session-log`, hard: replay reads a recorded trajectory through the session log's `SessionBackend` seam.
- `vector-store` to `memory`, hard: the store implements the memory plugin's `VectorStore` offload seam, wired through `MemoryPlugin::with_extensions`, and hands recalled turns back through `RetrievalReport::inject`.
- `speculative` to `session-log`, documented-optional: it declares no feature edge. `ModelOrchestrator::with_session_store` exists only when `plugin-session-log` is also enabled, and records `SessionEvent::ModelSwap`.
- `prompt-cache` and `rate-limit` declare no plugin dependency.

## Page layout

Every service page uses the same headings, in this order:

1. Feature and entry type
2. Entry points
3. Configuration
4. Caps and at-cap policy
5. Errors
6. Example
7. Related pages

The consolidated view of every cap, across plugins and services, is [Memory discipline](@/concepts/memory-discipline.md).
