+++
title = "Prompt cache"
description = "Keep replies on the client and answer an identical request without calling the provider."
template = "page.html"
weight = 3
+++

# Prompt cache

<dl class="page-facts">
<dt>In one line</dt>
<dd>Stores each finished reply on the client, keyed by a SHA-256 hash of the request, and serves an identical request from memory.</dd>
<dt>You need</dt>
<dd>The <code>service-prompt-cache</code> feature, plus one <code>provider-*</code> feature.</dd>
<dt>Read this if</dt>
<dd>You send the same request more than once, or you want to save and restore the cache.</dd>
</dl>

## What it does

Before CUCA sends a request, it hashes the request and looks the hash up. On a hit it replays the stored reply and sends nothing. On a miss it calls the provider, and after the stream finishes without error it stores the reply.

<div class="dgm">
<div class="dgm-scroll">
<svg viewBox="0 0 660 210" role="img" aria-labelledby="pc-flow-title pc-flow-desc">
  <title id="pc-flow-title">A request is hashed after the plugin hooks, then served from the cache on a hit or sent to the provider on a miss.</title>
  <desc id="pc-flow-desc">The request passes the on_request hooks, then is hashed into a key. The cache lookup has two outcomes. On a hit the stored blocks replay and no provider is called. On a miss the provider runs, the reply streams to you, and a successful stream is stored under the key.</desc>
  <defs>
    <marker id="pc-flow-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M0 0.6 L7 4 L0 7.4 z" fill="var(--muted-foreground)"/>
    </marker>
  </defs>
  <text class="t-title" x="6" y="20">CucaClient::generate_stream with a cache</text>
  <rect class="blk blk-ctl" x="6" y="76" width="118" height="50" rx="6"/>
  <text class="t-sm t-mid t-ctl" x="65" y="98">on_request</text>
  <text class="t-sm t-mid" x="65" y="113">hooks run first</text>
  <path class="arw arw-ctl" d="M126 101 H150" marker-end="url(#pc-flow-arrow)"/>
  <rect class="blk" x="152" y="76" width="118" height="50" rx="6"/>
  <text class="t-sm t-mid" x="211" y="98">hash request</text>
  <text class="t-sm t-mid" x="211" y="113">SHA-256 key</text>
  <path class="arw" d="M272 101 H296" marker-end="url(#pc-flow-arrow)"/>
  <rect class="blk" x="298" y="76" width="100" height="50" rx="6"/>
  <text class="t-sm t-mid" x="348" y="98">lookup</text>
  <text class="t-sm t-mid" x="348" y="113">key in cache</text>
  <path class="arw arw-data" d="M398 90 L440 52" marker-end="url(#pc-flow-arrow)"/>
  <text class="t-sm" x="402" y="60">hit</text>
  <rect class="blk blk-data" x="442" y="34" width="212" height="40" rx="6"/>
  <text class="t-sm t-mid t-data" x="548" y="58">replay stored blocks, no call</text>
  <path class="arw arw-bnd" d="M398 112 L440 150" marker-end="url(#pc-flow-arrow)"/>
  <text class="t-sm" x="402" y="148">miss</text>
  <rect class="blk blk-bnd" x="442" y="128" width="212" height="40" rx="6"/>
  <text class="t-sm t-mid t-bnd" x="548" y="152">provider call, then store</text>
  <text class="t-sm" x="6" y="196">Only a stream that ends without error is stored. A dropped or failed stream writes nothing.</text>
</svg>
</div>
<p class="dgm-cap"><b>The key covers the request after every hook.</b> A hook that changes the request changes the key, so the hash is taken last.</p>
</div>

## Try it

This program hashes requests and needs no server. The same request always gives the same key, and any change gives a new one.

```rust,name=See what the cache key depends on
use cuca::UnifiedRequest;
use cuca::services::prompt_cache::digest_request;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = UnifiedRequest::new("any-model").add_user_message("hi");
    let b = UnifiedRequest::new("any-model").add_user_message("hi!");
    let key = digest_request(&a)?;
    println!("key length: {}", key.len());
    println!("same request, same key: {}", key == digest_request(&a.clone())?);
    println!("changed request, same key: {}", key == digest_request(&b)?);
    Ok(())
}
```

```text,name=Expected output
key length: 64
same request, same key: true
changed request, same key: false
```

To turn the cache on, give the client builder a config. This one keeps up to 64 replies for 300 seconds each.

```rust,name=Configure a cache, repeat a turn, export the state
let client = CucaClient::builder()
    .with_provider(ProviderEndpoint::LlamaCpp)
    .with_base_url("http://127.0.0.1:1234/v1")
    .with_prompt_cache_config(PromptCacheConfig::new(64, Duration::from_secs(300))?)
    .build()?;
let cache = client.prompt_cache().expect("configured on the builder");

let first = client.generate_stream(request.clone()).await?;   // miss: calls the provider, then stores
let second = client.generate_stream(request).await?;          // hit: replays, opens no socket
println!("{}/{} entries", cache.len()?, cache.capacity());    // 1/64 entries once the first stream has ended

let snapshot = client.prompt_cache_snapshot()?;
```

`examples/prompt_cache.rs` is the full program. It runs a turn, repeats it to hit the cache, changes one word to miss, then imports the snapshot into a second client whose server address is a closed port. The hit still answers there, and an uncached turn fails on the connection that the hit never opened. It needs a `llama-server` on port 1234. The flags `--base-url` and `--model`, written after `--`, point it at any OpenAI-compatible server.

```bash,name=Runs the same on all three platforms
cargo run --example prompt_cache --features "provider-llamacpp service-prompt-cache"
```

## Feature and entry type

| Item | Value |
|---|---|
| Feature | `service-prompt-cache` |
| Own dependencies | `dep:sha2`, `dep:postcard` |
| Plugin it needs | none |
| Entry type | `PromptCache` |
| Module | `cuca::services::prompt_cache` |
| Attach with | `CucaClientBuilder::with_prompt_cache_config(config)` or `with_prompt_cache_service(service)` |

Supporting types: `PromptCacheConfig`, `PromptCacheEntry`, `PromptCacheSnapshot`, `PromptCacheImportReport`, `PromptCacheError`, and the function `digest_request`.

## Entry points

On the builder:

- `with_prompt_cache_config(config)` gives the client its own cache, built by `build()` from a validated `PromptCacheConfig`.
- `with_prompt_cache_service(service)` attaches an `Arc<PromptCache>` you own, for example one shared between clients. It wins when both are set.

On the client, which runs the cache for you:

- `generate_stream` hashes the request after every `on_request` hook and before the provider call, then looks the hash up.
- A miss calls the provider and writes one entry after a stream that finished without error. A stream that ends in an error, or is dropped before its last block, writes nothing.
- A failed write never fails the stream, which has already finished.
- `prompt_cache()` returns `Option<Arc<PromptCache>>`.
- `prompt_cache_snapshot()` and `replace_prompt_cache_snapshot(snapshot)` export and import the cache.

Using `PromptCache` directly:

| Method | What it does |
|---|---|
| `PromptCache::new(config)` | builds an empty cache on the wall clock |
| `digest_request(&request)` | returns the lowercase SHA-256 hex key |
| `lookup(key)` | returns `Option<PromptCacheEntry>` |
| `insert(request, response)` | hashes the request and stores the pair |
| `len()`, `is_empty()`, `capacity()` | read usage |
| `snapshot()`, `replace_snapshot(snapshot)` | export and import state |

A caller who uses `PromptCache` directly computes the key itself, after provider selection and every `on_request` hook. A hook that changes the request changes the key. For example, `plugin-memory` graph and warning injections change what `digest_request` returns. On a miss, call `insert` once the response finishes. A client with a cache does all of this itself.

### What goes into the key

The key is the lowercase SHA-256 hex hash of the request as it will go over the wire.

- It covers the whole `UnifiedRequest`. Every JSON value is written with object keys sorted and array order kept.
- A new `UnifiedRequest` field fails to compile until it is part of the key.
- Credentials, bearer tokens, base URLs and HTTP clients live on `CucaClient` and are not part of the key.
- The hash input format can change between crate versions. A snapshot from another version never hits and ages out by TTL and LRU.

### What a hit does

- The stored `content` blocks replay in order.
- The provider call, `execute_local_tool` and `on_stream_chunk` are skipped.
- `on_request` already ran before the lookup, in registration order.
- Every registered plugin's `on_response_complete` runs once, on the stored `UnifiedResponse`, with `duration_secs` replaced by the replay time.
- `model`, `provider`, `content`, `prompt_tokens`, `completion_tokens`, `finish_reason`, `prompt_cache_usage` and `usage` keep their stored values.
- A hit writes nothing back. It refreshes the entry's recency, not its expiry.
- A hit returns before a [speculative](@/services/speculative.md) orchestrator is reached.

Because every end-of-reply hook still runs, a session log gets its `Latency` and `TokenUsage` records for each turn, and `plugin-telemetry` records the replay time. The `plugin-cost` ledger counts a replayed reply again, so it shows spend as if there were no cache.

## Configuration

There is no default configuration. `PromptCacheConfig::new(capacity, ttl)` is the only checked constructor.

| Key | Type | Rule |
|---|---|---|
| `capacity` | `usize` | The most live entries. Non-zero. |
| `ttl` | `Duration` | How long an entry lives after insertion. Positive. |

- The fields are public and `PromptCache::new` does not check them again, so build the config with `PromptCacheConfig::new`.
- `PromptCache::new` cannot fail today. It returns `Result` so that a later check keeps the same signature.
- `with_prompt_cache_config` is ignored when `with_prompt_cache_service` is also called.
- `PromptCacheEntry` has these fields: `key`, `request`, `response`, `stored_at_unix_ms`, `expires_at_unix_ms` and `lru_rank` (`0` is the least recently used at export time).

## Caps and at-cap policy

- Bound: `capacity` entries, each expiring `ttl` after insertion.
- At the cap: least recently used eviction. `insert` evicts the oldest-used entry (rank `0`) until the cache fits. The order is tracked with a counter, not hash map order, so eviction is repeatable.
- Expiry: `insert` removes expired entries first, and `lookup` removes an expired entry and reports a miss. Inserting under an existing key refreshes its times and recency without growing the cache.
- Gauge: `PromptCache::len()` against `capacity()`. `len()` is an upper bound, because it counts expired entries that are not yet removed. `snapshot()` returns the exact live set and clones every entry.

### Save and restore the cache

- `snapshot()` returns a `PromptCacheSnapshot` of every live entry, sorted by key, with ranks numbered from `0`.
- `replace_snapshot(snapshot)` checks every entry first, then swaps the cache state in one step. One bad entry rejects the whole import and changes nothing.
- Entries that already expired are skipped. If more live entries remain than `capacity`, only the newest (highest rank) are kept, and ranks are renumbered from `0`.
- The `PromptCacheImportReport` has `imported_entries`, `expired_entries` and `capacity_evictions`.
- A snapshot holds full requests and replies: system prompts, user messages, tool arguments and results, base64 images and model output. Treat it as sensitive. CUCA does not encrypt, redact or write it. You own access control, storage and deletion.

## Errors

`PromptCacheError` has four variants:

- `Config(String)`: `capacity` or `ttl` is zero in `PromptCacheConfig::new`.
- `Validation { field, message }`: `digest_request` got a non-finite `temperature`, or an imported entry has a malformed key, a key that does not match the hash of its request, `stored_at_unix_ms` not below `expires_at_unix_ms`, a duplicate key, a duplicate `lru_rank`, or ranks that do not run from `0` without gaps.
- `Json(String)`: the request could not be encoded for hashing, or an import could not hash an entry's request.
- `Lock(String)`: the state mutex was poisoned.

On the client these become `CucaError`: `PromptCacheError::Json` becomes `CucaError::Json`, and every other variant becomes `CucaError::Config`. `prompt_cache_snapshot` and `replace_prompt_cache_snapshot` also return `CucaError::Config("no prompt cache configured")` on a client without a cache. `CucaClientBuilder::build` reports a cache construction error the same way. See [Error types](@/reference/errors.md).

## Related pages

- [Memory discipline](@/concepts/memory-discipline.md) lists the cache bound next to every other cap.
- [Speculative](@/services/speculative.md) describes the orchestrator path, where a cache miss still runs the end-of-reply hooks.
- [Session log](@/plugins/session-log.md) records the events a hit still produces.
- [Services](@/services/_index.md) lists all seven services.
