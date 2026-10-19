+++
title = "Prompt cache"
description = "The client-owned local response cache: the digest key, the TTL and LRU cap, and the export snapshot."
template = "page.html"
weight = 3
+++

# Prompt cache

<dl class="page-facts">
<dt>In one line</dt>
<dd>Caches complete <code>UnifiedRequest</code> to <code>UnifiedResponse</code> pairs on the client, keyed by a SHA-256 digest of the effective request.</dd>
<dt>You need</dt>
<dd>The <code>service-prompt-cache</code> feature.</dd>
<dt>Read this if</dt>
<dd>You configure a client-level cache, read its usage, or export and import its state.</dd>
</dl>

## Feature and entry type

| Item | Value |
|---|---|
| Feature | `service-prompt-cache` |
| Own dependencies | `dep:sha2`, `dep:postcard` |
| Declared plugin edge | none |
| Entry type | `PromptCache` |
| Module | `cuca::services::prompt_cache` |
| Attachment | `CucaClientBuilder::with_prompt_cache_config(config)` or `with_prompt_cache_service(service)` |

`PromptCache` never implements `CucaPlugin`. Passing it to `register_plugin` is a compile error.

Supporting types: `PromptCacheConfig`, `PromptCacheEntry`, `PromptCacheSnapshot`, `PromptCacheImportReport`, `PromptCacheError`, and the function `digest_request`.

## Entry points

On the builder:

- `with_prompt_cache_config(config)` gives the client a private cache that `build()` constructs from a validated `PromptCacheConfig`.
- `with_prompt_cache_service(service)` attaches a caller-owned `Arc<PromptCache>`, for example one shared between clients. It takes precedence when both are set.

On the client, which runs the cache automatically:

- `generate_stream` computes the digest after every `on_request` hook and before provider dispatch, then looks the digest up.
- A miss runs the provider and writes one entry after a fully successful stream. A stream that ends in an error, or is dropped before its last block, writes nothing.
- A failed write never fails the stream, which has already finished.
- `prompt_cache()` returns `Option<Arc<PromptCache>>`.
- `prompt_cache_snapshot()` and `replace_prompt_cache_snapshot(snapshot)` export and import the cache state.

Direct use of `PromptCache`:

- `PromptCache::new(config)` builds an empty cache on the wall clock.
- `digest_request(&request)` returns the lowercase SHA-256 hex key.
- `lookup(key)` returns `Option<PromptCacheEntry>`.
- `insert(request, response)` digests the request and stores the pair.
- `len()`, `is_empty()` and `capacity()` read usage.
- `snapshot()` and `replace_snapshot(snapshot)` export and import state.

### Hand-off the caller performs

A caller using `PromptCache` directly computes the key itself, after provider selection and every `on_request` hook. A hook that changes the request changes the digest. For example, `plugin-memory` graph and warning injections change what `digest_request` returns afterward.

On a miss, the caller completes the entry with `insert` once the response finishes. A client with a cache attached does this itself.

### Key

The key is the lowercase SHA-256 hex digest of the effective request, the request as it will cross the wire.

- The hashed bytes are the postcard encoding of a borrowed view of the whole `UnifiedRequest`, written straight into SHA-256 with no input buffer.
- Every `serde_json::Value` leaf is encoded as canonical JSON text, with object keys sorted recursively and array order preserved.
- The view destructures the request exhaustively, so a new `UnifiedRequest` field fails to compile until it is part of the key.
- Client credentials, bearer tokens, base URLs and HTTP clients live on `CucaClient` and are not part of the key.
- The digest input format is not stable across crate versions. A snapshot from another version never produces a hit and ages out by TTL and LRU.

### Behavior on a hit

- The stored `content` blocks replay in order.
- Provider dispatch, `execute_local_tool` and `on_stream_chunk` are skipped.
- `on_request` ran before the lookup, in registration order.
- Every registered plugin's `on_response_complete` runs once, against the stored `UnifiedResponse` with `duration_secs` replaced by the replay's elapsed time.
- `model`, `provider`, `content`, `prompt_tokens`, `completion_tokens`, `finish_reason`, `prompt_cache_usage` and `usage` keep their stored values.
- A hit writes nothing back. A lookup refreshes the entry's recency but not its expiry.
- A hit returns before a speculative orchestrator is reached.

Every terminal hook still runs on a hit, so each turn appends its `Latency` and `TokenUsage` records to a session log and `plugin-telemetry` records the replay latency. The `plugin-cost` ledger counts a replayed response again, so it reads as gross spend before the cache. No per block work repeats: `service-entity-extraction` is a separate explicit-call service, and `plugin-memory` has no completion hook.

## Configuration

There is no default configuration. `PromptCacheConfig::new(capacity, ttl)` is the only validated constructor.

| Key | Type | Rule |
|---|---|---|
| `capacity` | `usize` | Maximum live entries; non-zero. |
| `ttl` | `Duration` | Lifetime of an entry from insertion; positive. |

- The fields are public, and `PromptCache::new` does not re-check them, so build the config with `PromptCacheConfig::new`.
- `PromptCache::new` is infallible today; it returns `Result` so a later construction check stays compatible.
- `with_prompt_cache_config` is ignored when `with_prompt_cache_service` is also called.
- `PromptCacheEntry` fields: `key`, `request`, `response`, `stored_at_unix_ms`, `expires_at_unix_ms`, and `lru_rank` (`0` is least recently used at export time).

## Caps and at-cap policy

- Bound: `PromptCacheConfig::capacity` entries, each expiring `ttl` after insertion.
- At-cap policy: deterministic LRU eviction. `insert` evicts the least recently used entry (rank `0`) until the cache fits. Recency order is an oldest first map keyed by a monotonic touch counter, never hash map order, so eviction is reproducible. A touch is O(log n).
- Expiry: `insert` prunes expired entries first, and `lookup` removes an expired entry and reports a miss. Inserting under an existing digest refreshes its timestamps and recency without growing the cache.
- Usage gauge: `PromptCache::len()` against `capacity()`. `len()` is an upper bound because it counts expired entries not yet pruned. `snapshot()` returns the exact live set and clones every entry.

### Snapshot export and import

- `snapshot()` returns a `PromptCacheSnapshot` of every live entry, sorted by key, with ranks renumbered contiguously from `0`.
- `replace_snapshot(snapshot)` validates every entry before taking any lock, then atomically replaces the cache state. One failure rejects the whole import with no state change.
- Entries that already expired are skipped. If more live entries remain than `capacity`, only the newest (highest rank) are kept.
- Installed ranks are renumbered from `0`.
- The returned `PromptCacheImportReport` has `imported_entries`, `expired_entries` and `capacity_evictions`.
- A snapshot holds complete request and response values: system prompts, user messages, tool arguments and results, base64 images and model output. Treat it as sensitive. CUCA does not encrypt, redact or write it, and the caller owns access control, storage and deletion.

## Errors

`PromptCacheError` has four variants:

- `Config(String)` for a zero `capacity` or a zero `ttl` in `PromptCacheConfig::new`.
- `Validation { field, message }` for a non-finite `temperature` in `digest_request`, or for an imported entry with a malformed key, a key that does not match the digest of its request, `stored_at_unix_ms` not below `expires_at_unix_ms`, a duplicate key, a duplicate `lru_rank`, or ranks that are not contiguous from `0`.
- `Json(String)` when the request cannot be encoded for the digest, or when an import cannot digest an entry's request.
- `Lock(String)` when the state mutex is poisoned.

On the client these map to `CucaError`: `PromptCacheError::Json` becomes `CucaError::Json`, and every other variant becomes `CucaError::Config`. `prompt_cache_snapshot` and `replace_prompt_cache_snapshot` also return `CucaError::Config("no prompt cache configured")` on a client without a cache. `CucaClientBuilder::build` reports a cache construction error the same way. See [Error types](@/reference/errors.md).

## Example

```rust,name=Configure a cache, repeat a turn, export the state
let client = CucaClient::builder()
    .with_provider(ProviderEndpoint::LlamaCpp)
    .with_base_url("http://127.0.0.1:1234/v1")
    .with_prompt_cache_config(PromptCacheConfig::new(64, Duration::from_secs(300))?)
    .build()?;
let cache = client.prompt_cache().expect("configured on the builder");

let key = digest_request(&request)?;                 // cuca::services::prompt_cache
let first = client.generate_stream(request.clone()).await?;   // miss: dispatches, then writes
let second = client.generate_stream(request).await?;          // hit: replays, no socket
println!("{}/{} entries", cache.len()?, cache.capacity());

let snapshot = client.prompt_cache_snapshot()?;
```

`examples/prompt_cache.rs` is the full program. It runs a turn, repeats it to hit the cache, changes one word to miss, then imports the snapshot into a second client whose base URL is a closed port. The hit still answers there, and an uncached turn fails on the connection the hit never opened. It needs a `llama-server` on port 1234. `CUCA_BASE_URL` and `CUCA_MODEL` retarget it.

```bash,name=Runs the same on all three platforms
cargo run --example prompt_cache --features "provider-llamacpp service-prompt-cache"
```

## Related pages

- [Memory discipline](@/concepts/memory-discipline.md) lists the cache bound beside every other cap.
- [Speculative](@/services/speculative.md) describes the orchestrator path, where a cache miss is instrumented.
- [Session log](@/plugins/session-log.md) records the completion events a hit still produces.
- [Services overview](@/services/_index.md) lists all six services.
