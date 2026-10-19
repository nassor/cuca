+++
title = "Rate limit"
description = "The client-side outbound throttle: an integer token bucket over request rate, a concurrency cap, and the permit that holds a slot for a whole turn."
template = "page.html"
weight = 5
+++

# Rate limit

<dl class="page-facts">
<dt>In one line</dt>
<dd>An integer token bucket over request rate plus a hard concurrency cap, so a caller fanning out over <code>generate_stream</code> stays inside a provider's quota.</dd>
<dt>You need</dt>
<dd>The <code>service-rate-limit</code> feature, plus one <code>provider-*</code> feature.</dd>
<dt>Read this if</dt>
<dd>You are calling <code>RateLimiter::acquire</code> or <code>try_acquire</code> around <code>CucaClient::generate_stream</code>.</dd>
</dl>

## Feature and entry type

| Item | Value |
|---|---|
| Feature flag | `service-rate-limit = ["dep:tokio"]` |
| Declared plugin edge | none |
| Entry type | `RateLimiter` |
| Other public types | `RateLimitConfig`, `RateLimitPermit`, `RateLimitUsage`, `RateLimitObserver`, `RateLimitError` |

`RateLimiter` is an explicit-call service. It never implements `CucaPlugin`, so passing it to `register_plugin` is a compile error. A hook cannot carry the capability: `CucaPlugin::on_request` is synchronous, so it can reject a request but never pace one. A permit taken in a hook also leaks when `generate_stream` returns `Err`, when the consumer drops the stream early, or when the stream is never wrapped by the top-level instrumentation.

## Entry points

| Method | Behavior |
|---|---|
| `RateLimiter::new(config)` | builds a limiter with a full bucket (`burst` tokens) and every slot free |
| `acquire().await` | waits for a concurrency slot, then for a token, for at most `max_wait` |
| `try_acquire()` | takes a slot and a token immediately, or fails at once; no waiting |
| `usage()` | reads a `RateLimitUsage` gauge |
| `config()` | returns the validated `RateLimitConfig` |

- `acquire` takes the slot first and the token second. A caller that gives up waiting for a slot has spent no quota.
- A token is spent and never returned. Dropping a `RateLimitPermit` returns its slot only.
- `acquire` needs a Tokio reactor, because it uses `tokio::time` and a `tokio::sync::Semaphore`. `try_acquire` and `usage` need neither, so a non-Tokio caller can use them.
- Share one `RateLimiter` behind an `Arc` across every task that dispatches to the same throttled provider. `RateLimiter` and `RateLimitPermit` are `Send + Sync`.
- The limiter measures time with a monotonic clock, so a system clock jump cannot mint a burst. The bucket mutex is never held across an `.await`.

### Hand-off: hold the permit for the whole turn

The caller performs this sequence for every turn:

1. `acquire()` the permit.
2. Call `generate_stream`.
3. Drain the stream.
4. Drop the permit.

Dropping the permit before the stream is drained under-counts concurrency. `usage().in_flight` makes that mistake observable.

### Not covered

The crate has no 429 or `Retry-After` feedback loop. A non-2xx response becomes `CucaError::Http { status, body }` inside the provider adapter and propagates before any stream or hook exists. No adapter keeps the response headers. The limiter never learns that an admitted request was rejected upstream, and it has no `penalize()` call. After a 429, widen `interval` or lower `max_requests` and rebuild the limiter. `RateLimiter` never fails a turn by itself; a caller that wants reject-only behavior calls `try_acquire`.

## Configuration

`RateLimitConfig::new(max_requests, interval, max_concurrent, max_waiters)` is the only base constructor. `with_burst`, `with_max_wait` and `with_warn_fraction` override a defaulted field, and `with_observer` appends an observer. Each `with_*` builder except `with_observer` returns `Result` and re-validates. `validate()` re-checks every bound, since the fields are public and a struct literal can bypass `new`.

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
- `burst` is the bucket capacity, the largest instantaneous burst.
- `max_waiters` is the most callers that may be inside `acquire` at once.
- `max_wait` is the wait budget of one `acquire`.
- `warn_fraction` makes every observer receive a `RateLimitUsage` once `waiting / max_waiters` reaches it. The check runs after the caller has a slot and before a token is spent.
- `observers` is the list of `RateLimitObserver` implementations. `observe` returns `Result<(), PluginError>`, and an `Err` aborts that `acquire`.
- The bucket counts in whole milliseconds, so an `interval` under one millisecond rounds up to one millisecond.

## Caps and at-cap policy

| Bound | At-cap policy | Usage gauge |
|---|---|---|
| token bucket, at most `burst * 1_000` thousandths of a token | saturates; a long idle period never banks credit past `burst` | `RateLimitUsage::available_tokens` |
| `max_concurrent` permits | callers wait for a free slot | `RateLimitUsage::in_flight` |
| `max_waiters` callers inside `acquire` | rejects, never evicts: an over-cap `acquire` returns `QueueFull` at once, before it parks | `RateLimitUsage::waiting` |
| `max_wait` per `acquire` | the call returns `Exhausted` when the budget elapses | none |

- Slots in flight plus callers waiting never exceed `max_concurrent + max_waiters`.
- The bucket is two integers and never grows. The limiter keeps no request clone, key, log or map.
- `RateLimitConfig::observers` is sized by the caller at construction and does not grow with traffic.

## Errors

`RateLimitError` is the error type of every method. It implements `std::error::Error`.

- `Config(String)`: a bound was rejected by `RateLimitConfig::new`, a `with_*` call, `validate()` or `RateLimiter::new`.
- `QueueFull { waiters, max_waiters }`: `acquire` found the waiter queue at `max_waiters` and refused instead of queueing.
- `Busy { in_flight, max_concurrent }`: `try_acquire` found no free slot.
- `Exhausted { retry_after }`: `try_acquire` found an empty bucket, or `acquire` ran out of `max_wait` while waiting for a slot or a token. `retry_after` is the time until the next whole token, and it is zero when a full bucket was never the obstacle.
- `Observer(PluginError)`: a `RateLimitObserver` refused an `acquire` that crossed `warn_fraction`. The slot already taken is released before the error returns.
- `Lock(String)`: the bucket mutex was poisoned. The limiter fails closed instead of admitting from a half-updated bucket.

`Exhausted.retry_after` is the only value meant to drive a retry, and the caller decides whether to retry. `From<RateLimitError> for PluginError` is lossy:

- `Config` becomes `PluginError::Validation` with schema `rate-limit-config`.
- `Observer(e)` becomes `e` unchanged.
- `QueueFull`, `Busy`, `Exhausted` and `Lock` become `PluginError::Internal`, with their numbers kept only in the message.

A caller that needs the typed `retry_after` matches on `RateLimitError` before converting.

## Example

One limiter paces turns at 10 requests per 60 seconds, with 2 turns in flight and up to 8 callers waiting. The program needs a `llama-server` on port 1234 with a model loaded.

```rust,name=Hold one permit across a whole turn
use std::sync::Arc;
use std::time::Duration;

use cuca::types::ProviderEndpoint;
use cuca::{CucaClient, RateLimitConfig, RateLimiter, UnifiedRequest};
use tokio_stream::StreamExt;

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let limiter = Arc::new(RateLimiter::new(RateLimitConfig::new(
        10,
        Duration::from_secs(60),
        2,
        8,
    )?)?);
    let client = CucaClient::builder()
        .with_provider(ProviderEndpoint::LlamaCpp)
        .with_base_url("http://127.0.0.1:1234/v1")
        .build()?;

    let permit = limiter.acquire().await?;
    let usage = limiter.usage()?;
    println!(
        "tokens={} in_flight={} waiting={}",
        usage.available_tokens, usage.in_flight, usage.waiting
    );
    let request = UnifiedRequest::new("any-model").add_user_message("Reply with: ok");
    let mut stream = client.generate_stream(request).await?;
    while let Some(block) = stream.next().await {
        block?;
    }
    drop(permit);
    Ok(())
}
```

```text,name=Expected output
tokens=9 in_flight=1 waiting=0
```

`examples/rate_limit.rs` fans six turns out over one shared limiter sized at one request per 45 seconds. It prints each admission time and `usage()` reading, the peak `in_flight` and `waiting`, and the `retry_after` that `try_acquire` reports once the bucket is empty. It needs a `llama-server` on port 1234 with the demo model loaded; `CUCA_BASE_URL` and `CUCA_MODEL` point it at any OpenAI-compatible server.

```bash,name=Runs the same on all three platforms
cargo run --example rate_limit --features "provider-llamacpp service-rate-limit"
```

## Related pages

- [Services](@/services/_index.md) lists every service and its plugin edge.
- [Plugins and services](@/concepts/plugin-layering.md) explains the explicit-call tier.
- [Memory discipline](@/concepts/memory-discipline.md) explains the bounded-by-default rule.
- [Errors](@/reference/errors.md) lists the crate's error types.
