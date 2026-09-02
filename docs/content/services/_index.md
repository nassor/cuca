+++
title = "Services"
description = "Six service features: explicit-call, client-level capabilities plus the pipeline-replacing orchestrator, and how each depends on the plugin tier."
template = "section.html"
sort_by = "weight"

[extra]
kicker = "Reference"
+++

<dl class="page-facts">
<dt>In one line</dt>
<dd>Six <code>service-*</code> features; five are called directly, one replaces the dispatch stage</dd>
<dt>You need</dt>
<dd>The service features you want, named explicitly; none is enabled by default</dd>
<dt>Read this if</dt>
<dd>You need a service's feature flag, entry type, plugin dependency, or caps</dd>
</dl>

## The six features

| Feature | Entry type | Attachment | Plugin dependency |
|---|---|---|---|
| [`service-entity-extraction`](@/services/entity-extraction.md) | `EntityExtractor` | direct calls only | `plugin-memory`, hard |
| [`service-prompt-cache`](@/services/prompt-cache.md) | `PromptCache` | `with_prompt_cache_config` or `with_prompt_cache_service` | none |
| [`service-replay`](@/services/replay.md) | `SessionReplay` | direct calls only | `plugin-session-log`, hard |
| [`service-speculative`](@/services/speculative.md) | `ModelOrchestrator` | `with_orchestrator` | `plugin-session-log`, documented-optional |
| [`service-rate-limit`](@/services/rate-limit.md) | `RateLimiter` | direct calls only | none |
| [`service-vector-store`](@/services/vector-store.md) | `InMemoryVectorStore` | direct calls plus the `with_extensions` hand-off | `plugin-memory`, hard |

The mechanical difference between explicit-call and pipeline-replacement is in
[Plugins and services](@/concepts/plugin-layering.md).

## Depending on the plugin tier

A service may depend on the plugin features it declares in `Cargo.toml`; a
plugin must never depend on, or name, a service. `Cargo.toml` carries exactly
three hard edges and one documented-optional edge, all service to plugin:

```toml,name=Every service-to-plugin feature edge in Cargo.toml
service-entity-extraction = ["plugin-memory"]
service-replay = ["plugin-session-log"]
service-vector-store = ["plugin-memory", "dep:wide"]
```

- `entity-extraction → memory` (hard): the extraction delta is a `MemoryGraph`
  the caller merges into the memory plugin's working graph.
- `replay → session-log` (hard): replay reads a recorded trajectory through the
  session log's `SessionBackend` seam.
- `vector-store → memory` (hard): the store implements the memory plugin's
  `VectorStore` offload seam, wired through `MemoryPlugin::with_extensions`.
- `speculative → session-log` (documented-optional): `ModelOrchestrator::with_session_store`
  is compiled out of existence without it, and records `SessionEvent::ModelSwap`
  with it.
- `prompt-cache` and `rate-limit` declare no plugin dependency at all.

A service-to-service edge does not exist and would require re-tiering the
shared part downward first.

## Caps and at-cap policies

Every service page states its caps as bound, at-cap policy, usage gauge, in
that order. The consolidated view, and the reasoning behind refusing rather
than evicting, is [Memory discipline](@/concepts/memory-discipline.md).

| Service | Bound | At-cap policy | Usage gauge |
|---|---|---|---|
| `service-prompt-cache` | `PromptCacheConfig::capacity`, caller-set | evict least recently used | `len()`, against `capacity()` |
| `service-replay` | `ReplayConfig::max_records`, default `65536` | refuse, never truncate | `usage().records` against `usage().max_records` |
| `service-replay` | `ReplayConfig::max_turn_blocks`, default `4096` | refuse | `usage().blocks` |
| `service-entity-extraction` | no growth cap | each call produces one standalone report | not applicable |
| `service-speculative` | `ClientPool` is uncapped, bounded by configuration | no eviction | `ClientPool::len()` |
| `service-rate-limit` | `RateLimitConfig::max_concurrent`, caller-set | callers wait for a free permit | `RateLimitUsage::in_flight` |
| `service-rate-limit` | `RateLimitConfig::max_waiters`, caller-set | reject the acquire (`QueueFull`), never queue further | `RateLimitUsage::waiting` |
| `service-vector-store` | `VectorStoreConfig::max_entries`, caller-set | evict oldest first, reusing its arena slot | `usage().entries` against `usage().capacity` |
| `service-vector-store` | `VectorStoreConfig::max_entry_bytes`, caller-set | reject the turn, never truncate | `VectorStoreError::EntryTooLarge`, converted to `PluginError::Validation` at the seam and recorded in `CompressionReport::last_error` |

## Per-service pages
