+++
title = "Rate limit"
description = "Pace your own requests so many tasks stay inside a provider's quota."
template = "page.html"
weight = 5
+++

# Rate limit

<dl class="page-facts">
<dt>In one line</dt>
<dd>Lets only a set number of requests start per time window, and only a set number run at once, so a fan-out of turns stays inside a provider's quota.</dd>
<dt>You need</dt>
<dd>The <code>service-rate-limit</code> feature, plus one <code>provider-*</code> feature for real turns.</dd>
<dt>Read this if</dt>
<dd>You call <code>CucaClient::generate_stream</code> from many tasks against one rate-limited provider.</dd>
</dl>

## What it does

A `RateLimiter` has two limits. A token bucket allows `max_requests` new requests per `interval`. A slot count allows `max_concurrent` requests at the same time. Before each turn you call `acquire()`. It waits until a slot and a token are free, then hands you a permit. You hold the permit until the stream is done.

<div class="dgm">
<div class="dgm-scroll">
<svg viewBox="0 0 660 190" role="img" aria-labelledby="rl-flow-title rl-flow-desc">
  <title id="rl-flow-title">acquire checks the waiting line, waits for a slot, waits for a token, and returns a permit you hold during the turn.</title>
  <desc id="rl-flow-desc">Five boxes in a row. The caller calls acquire. The first check refuses with QueueFull if max_waiters callers are already waiting. Then the call waits for a free concurrency slot, then for a token from the bucket. Then the caller runs generate_stream while holding the permit. If the wait passes max_wait, acquire returns Exhausted.</desc>
  <defs>
    <marker id="rl-flow-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M0 0.6 L7 4 L0 7.4 z" fill="var(--muted-foreground)"/>
    </marker>
  </defs>
  <text class="t-title" x="6" y="20">RateLimiter::acquire</text>
  <rect class="blk" x="6" y="44" width="112" height="50" rx="6"/>
  <text class="t-sm t-mid" x="62" y="66">your task</text>
  <text class="t-sm t-mid" x="62" y="81">acquire().await</text>
  <path class="arw" d="M120 69 H140" marker-end="url(#rl-flow-arrow)"/>
  <rect class="blk blk-ctl" x="142" y="44" width="112" height="50" rx="6"/>
  <text class="t-sm t-mid t-ctl" x="198" y="66">line check</text>
  <text class="t-sm t-mid" x="198" y="81">max_waiters</text>
  <path class="arw" d="M256 69 H276" marker-end="url(#rl-flow-arrow)"/>
  <rect class="blk blk-ctl" x="278" y="44" width="112" height="50" rx="6"/>
  <text class="t-sm t-mid t-ctl" x="334" y="66">wait for slot</text>
  <text class="t-sm t-mid" x="334" y="81">max_concurrent</text>
  <path class="arw" d="M392 69 H412" marker-end="url(#rl-flow-arrow)"/>
  <rect class="blk blk-ctl" x="414" y="44" width="112" height="50" rx="6"/>
  <text class="t-sm t-mid t-ctl" x="470" y="66">wait for token</text>
  <text class="t-sm t-mid" x="470" y="81">bucket, burst</text>
  <path class="arw arw-data" d="M528 69 H548" marker-end="url(#rl-flow-arrow)"/>
  <rect class="blk blk-data" x="550" y="44" width="104" height="50" rx="6"/>
  <text class="t-sm t-mid t-data" x="602" y="66">run the turn</text>
  <text class="t-sm t-mid" x="602" y="81">hold permit</text>
  <text class="t-sm" x="142" y="116">line full: QueueFull</text>
  <text class="t-sm" x="278" y="132">no slot or token before max_wait: Exhausted</text>
  <text class="t-sm" x="6" y="164">Drop the permit after the stream ends: that frees the slot. The token stays spent.</text>
  <text class="t-sm" x="6" y="180">The bucket refills max_requests tokens every interval, up to burst.</text>
</svg>
</div>
<p class="dgm-cap"><b>Three limits guard one call.</b> The waiting line is capped, the slots are capped, and the bucket paces new starts.</p>
</div>

## Try it

This program takes one permit and reads the limiter's gauges. It needs no server.

```rust,name=Take a permit and read the gauges
use std::time::Duration;

use cuca::{RateLimitConfig, RateLimiter};

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 10 requests per 60 seconds, 2 turns at once, up to 8 callers waiting.
    let limiter = RateLimiter::new(RateLimitConfig::new(10, Duration::from_secs(60), 2, 8)?)?;

    let permit = limiter.acquire().await?;
    let usage = limiter.usage()?;
    println!(
        "tokens={} in_flight={} waiting={}",
        usage.available_tokens, usage.in_flight, usage.waiting
    );
    drop(permit);
    Ok(())
}
```

```text,name=Expected output
tokens=9 in_flight=1 waiting=0
```

To pace real turns, share the limiter in an `Arc` and hold the permit across the whole turn:

```rust,name=Hold one permit across a whole turn
let permit = limiter.acquire().await?;               // 1. wait for a slot and a token
let mut stream = client.generate_stream(request).await?;   // 2. start the turn
while let Some(block) = stream.next().await {        // 3. drain the stream
    block?;
}
drop(permit);                                        // 4. free the slot
```

`examples/rate_limit.rs` fans six turns out over one shared limiter sized at one request per 45 seconds. It prints each admission time and `usage()` reading, the peak `in_flight` and `waiting`, and the `retry_after` that `try_acquire` reports once the bucket is empty. It needs a `llama-server` on port 1234. The flags `--base-url` and `--model`, written after `--`, point it at any OpenAI-compatible server.

```bash,name=Runs the same on all three platforms
cargo run --example rate_limit --features "provider-llamacpp service-rate-limit"
```

## Feature and entry type

| Item | Value |
|---|---|
| Feature flag | `service-rate-limit = ["dep:tokio"]` |
| Plugin it needs | none |
| Entry type | `RateLimiter` |
| Other public types | `RateLimitConfig`, `RateLimitPermit`, `RateLimitUsage`, `RateLimitObserver`, `RateLimitError` |

A plugin hook cannot do this job. `CucaPlugin::on_request` is synchronous, so it can reject a request but cannot make one wait. A permit taken in a hook also leaks when `generate_stream` returns `Err`, when the consumer drops the stream early, or when the stream is never wrapped by the client's top-level hooks.

## Entry points

| Method | What it does |
|---|---|
| `RateLimiter::new(config)` | builds a limiter with a full bucket (`burst` tokens) and every slot free |
| `acquire().await` | waits for a slot, then for a token, for at most `max_wait` |
| `try_acquire()` | takes a slot and a token now, or fails at once. It never waits. |
| `usage()` | reads a `RateLimitUsage` |
| `config()` | returns the validated `RateLimitConfig` |

- `acquire` takes the slot first and the token second. A caller that gives up waiting for a slot has spent no quota.
- A token is spent and never returned. Dropping a `RateLimitPermit` frees its slot only.
- `acquire` needs a Tokio runtime, because it uses `tokio::time` and a `tokio::sync::Semaphore`. `try_acquire` and `usage` need neither, so a non-Tokio caller can use them.
- Share one `RateLimiter` in an `Arc` across every task that calls the same throttled provider. `RateLimiter` and `RateLimitPermit` are `Send + Sync`.
- The limiter uses a monotonic clock, so a system clock jump cannot create a burst. The bucket lock is never held across an `.await`.

### Not covered

The crate has no 429 or `Retry-After` feedback. A non-2xx reply becomes `CucaError::Http { status, body }` inside the provider adapter, before any stream or hook exists. The limiter never learns that an admitted request was rejected upstream, and it has no `penalize()` call. After a 429, widen `interval` or lower `max_requests`, and rebuild the limiter. `RateLimiter` never fails a turn by itself. For reject-only behavior, call `try_acquire`.

## Configuration

`RateLimitConfig::new(max_requests, interval, max_concurrent, max_waiters)` is the only base constructor. `with_burst`, `with_max_wait` and `with_warn_fraction` change a defaulted field, and `with_observer` adds an observer. Each `with_*` builder except `with_observer` returns `Result` and checks again. `validate()` re-checks every bound, since the fields are public and a struct literal can skip `new`.

| Key | Type | Default | Valid values |
|---|---|---|---|
| `max_requests` | `u32` | required | non-zero |
| `interval` | `Duration` | required | non-zero |
| `burst` | `u32` | `max_requests` | non-zero |
| `max_concurrent` | `usize` | required | non-zero, at most `Semaphore::MAX_PERMITS` |
| `max_waiters` | `usize` | required | non-zero |
| `max_wait` | `Duration` | `interval` | non-zero |
| `warn_fraction` | `Option<f32>` | `None` | `None`, or finite and in `(0.0, 1.0]` |
| `observers` | `Vec<Arc<dyn RateLimitObserver>>` | empty | none |

- `max_requests` is the number of tokens added per `interval`.
- `burst` is the bucket size, the most requests that can start at once.
- `max_waiters` is the most callers that may be inside `acquire` at once.
- `max_wait` is the time one `acquire` may wait.
- `warn_fraction`: once `waiting / max_waiters` reaches it, every observer receives a `RateLimitUsage`. The check runs after the caller has a slot and before a token is spent.
- `observers` is the list of `RateLimitObserver` implementations. `observe` returns `Result<(), PluginError>`, and an `Err` aborts that `acquire`.
- The bucket counts whole milliseconds, so an `interval` under one millisecond rounds up to one millisecond.

## Caps and at-cap policy

| Bound | At the cap | Gauge |
|---|---|---|
| token bucket, at most `burst * 1_000` thousandths of a token | saturates: a long idle time never saves credit past `burst` | `RateLimitUsage::available_tokens` |
| `max_concurrent` slots | callers wait for a free slot | `RateLimitUsage::in_flight` |
| `max_waiters` callers inside `acquire` | rejects, never evicts: an over-cap `acquire` returns `QueueFull` at once, before it waits | `RateLimitUsage::waiting` |
| `max_wait` per `acquire` | the call returns `Exhausted` when the time is up | none |

- Slots in use plus callers waiting never exceed `max_concurrent + max_waiters`.
- The bucket is two integers and never grows. The limiter keeps no request copy, key, log or map.
- `observers` is sized by you at construction and does not grow with traffic.

## Errors

`RateLimitError` is the error type of every method. It implements `std::error::Error`.

- `Config(String)`: a bound was rejected by `RateLimitConfig::new`, a `with_*` call, `validate()` or `RateLimiter::new`.
- `QueueFull { waiters, max_waiters }`: `acquire` found the waiting line at `max_waiters` and refused to queue.
- `Busy { in_flight, max_concurrent }`: `try_acquire` found no free slot.
- `Exhausted { retry_after }`: `try_acquire` found an empty bucket, or `acquire` ran out of `max_wait` while waiting for a slot or a token. `retry_after` is the time until the next whole token. It is zero when a full bucket was never the obstacle.
- `Observer(PluginError)`: a `RateLimitObserver` refused an `acquire` that crossed `warn_fraction`. The slot already taken is freed before the error returns.
- `Lock(String)`: the bucket lock was poisoned. The limiter refuses instead of admitting from a half-updated bucket.

`Exhausted.retry_after` is the only value meant to drive a retry, and you decide whether to retry. `From<RateLimitError> for PluginError` loses detail:

- `Config` becomes `PluginError::Validation` with schema `rate-limit-config`.
- `Observer(e)` becomes `e` unchanged.
- `QueueFull`, `Busy`, `Exhausted` and `Lock` become `PluginError::Internal`, with their numbers only in the message.

If you need the typed `retry_after`, match on `RateLimitError` before you convert.

## Related pages

- [Services](@/services/_index.md) lists every service and its plugin dependency.
- [Scheduler](@/services/scheduler.md) orders many callers over several backends, where this page paces one caller's requests.
- [Plugins and services](@/concepts/plugin-layering.md) explains the explicit-call tier.
- [Memory discipline](@/concepts/memory-discipline.md) explains the bounded-by-default rule.
- [Errors](@/reference/errors.md) lists the crate's error types.
