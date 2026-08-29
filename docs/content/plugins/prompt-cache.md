+++
title = "Prompt cache"
description = "The client-owned local response cache: the digest key, the TTL and LRU cap, and the export snapshot."
template = "page.html"
weight = 13
+++

# Prompt cache

<dl class="page-facts">
<dt>In one line</dt>
<dd>Caches complete UnifiedRequest to UnifiedResponse pairs on the client, keyed by a digest of the effective request.</dd>
<dt>You need</dt>
<dd>The <code>plugin-prompt-cache</code> feature.</dd>
<dt>Read this if</dt>
<dd>You are configuring a client-level cache or exporting and importing its state.</dd>
</dl>

## Entry types

`PromptCache`, `PromptCacheConfig`, `PromptCacheEntry`, `PromptCacheSnapshot`, `PromptCacheImportReport`, `PromptCacheError`.

## Not a `CucaPlugin`

`PromptCache` does not implement `CucaPlugin`. Registering it with `register_plugin` is a compile error, not an inert no-op. It is a plain client-level service, attached with `CucaClientBuilder::with_prompt_cache_config(config)` or `with_prompt_cache_service(service)`, and wired directly into `CucaClient::generate_stream`: a lookup runs after every `on_request` hook and before provider dispatch, and a miss is written back after a fully successful stream.

## Key

The lookup key is the lowercase SHA-256 hex digest of the canonicalized JSON of the effective request, the request exactly as it will cross the wire after provider selection and every `on_request` hook. Canonicalization sorts object keys recursively; array order and scalar values are preserved.

## Capacity

`PromptCacheConfig::new(capacity, ttl)` is the only constructor; there is no default configuration, and it rejects a zero `capacity` or a zero `ttl` with `PromptCacheError::Config`.

| | |
|---|---|
| Bound | `PromptCacheConfig::capacity` entries, each expiring `ttl` after insertion |
| At-cap policy | Deterministic LRU eviction: the least recently used entry is evicted to make room |
| Usage gauge | `PromptCache::len()` against `capacity()`; `len()` is an upper bound that may include not-yet-pruned expired entries, `snapshot()` gives the exact live set |

## Export

`CucaClient::prompt_cache_snapshot()` returns a `PromptCacheSnapshot` of every live entry, sorted by key. `CucaClient::replace_prompt_cache_snapshot(snapshot)` validates the snapshot in full, then atomically replaces the cache's state, returning a `PromptCacheImportReport` of imported, expired, and capacity-evicted entry counts.
