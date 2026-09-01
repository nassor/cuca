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
<dd>The <code>service-rate-limit</code> feature.</dd>
<dt>Read this if</dt>
<dd>You are calling <code>RateLimiter::acquire</code> or <code>try_acquire</code> around <code>CucaClient::generate_stream</code>.</dd>
</dl>

`RateLimiter` paces outbound turns from the caller's side: an integer token bucket over request rate plus a hard cap on turns in flight. `acquire()` waits for a concurrency slot and then a token, up to `RateLimitConfig::max_wait`; `try_acquire()` takes both or fails at once, with no waiting and no Tokio reactor; `usage()` reads a cheap `RateLimitUsage` gauge. Reach for it when a fan-out over `CucaClient::generate_stream` has to stay inside a provider's quota.

```rust,name=Pace one turn through a shared limiter
use std::sync::Arc;
use std::time::Duration;

use cuca::types::ProviderEndpoint;
use cuca::{CucaClient, RateLimitConfig, RateLimiter, UnifiedRequest};
use tokio_stream::StreamExt;

let limiter = Arc::new(RateLimiter::new(
    RateLimitConfig::new(60, Duration::from_secs(60), 4, 128)?,
)?);
let client = CucaClient::builder()
    .with_provider(ProviderEndpoint::LlamaCpp)
    .with_base_url("http://127.0.0.1:1234/v1")
    .build()?;

// Hold the permit for the whole turn: dispatch and drain.
let permit = limiter.acquire().await?;
let usage = limiter.usage()?;
println!(
    "tokens={} in_flight={} waiting={}",
    usage.available_tokens, usage.in_flight, usage.waiting
);

let mut stream = client
    .generate_stream(UnifiedRequest::new("google/gemma-4-e4b").add_user_message("Say ok."))
    .await?;
while let Some(block) = stream.next().await {
    let _ = block?;
}
drop(permit);
```

```text,name=Expected output with the permit still held
tokens=59 in_flight=1 waiting=0
```

## Entry types

`RateLimiter`, `RateLimitConfig`, `RateLimitPermit`, `RateLimitUsage`, `RateLimitObserver`, `RateLimitError`.

## Config

`RateLimitConfig::new(max_requests, interval, max_concurrent, max_waiters)` is the only base constructor; every bound must be non-zero, or it returns `RateLimitError::Config`. `with_burst`, `with_max_wait`, and `with_warn_fraction` override the defaulted fields, and `with_observer` appends a `RateLimitObserver`. `validate()` re-checks every bound, since the fields are public and a struct literal can bypass `new`.

| Field | Meaning | Default |
|---|---|---|
| `max_requests` | Tokens replenished per `interval`; must be non-zero | required |
| `interval` | Refill window; must be non-zero | required |
| `burst` | Bucket capacity, the largest instantaneous burst; must be non-zero | `max_requests` |
| `max_concurrent` | Hard cap on concurrently held permits; must be non-zero | required |
| `max_waiters` | Hard cap on callers parked in `acquire`; must be non-zero | required |
| `max_wait` | Per-acquire wait budget; must be non-zero | `interval` |
| `warn_fraction` | Warn every registered observer once `waiting / max_waiters` reaches this fraction; must be in `(0.0, 1.0]` when set | `None` (disabled) |
| `observers` | `RateLimitObserver`s notified when `warn_fraction` is crossed | empty |

## The acquire, dispatch, drain, drop hand-off

Hold the permit for the whole turn, not just the call that starts it. Dropping it before the stream is drained under-counts concurrency, because the slot is what actually comes back on drop; the token stays spent.

`acquire` waits for a slot before it waits for a token, on purpose: a caller that gives up waiting for a slot has spent no quota, and the number of parked callers is already bounded by `max_concurrent + max_waiters` together. Share one `RateLimiter` behind an `Arc` across every task dispatching to the same throttled provider; both `RateLimiter` and `RateLimitPermit` are `Send + Sync`.

## Try it

`examples/rate_limit.rs` runs the hand-off for real: six turns share one client and one `Arc<RateLimiter>` sized at three requests per two seconds with `max_concurrent = 2` and `max_waiters = 8`. Each turn prints its admission time and a `usage()` reading, a peak `in_flight`/`waiting` line follows the fan-out, and the run ends by draining the bucket with `try_acquire` until it reports `Exhausted` with a `retry_after`.

```bash,name=Runs the same on all three platforms
cargo run --example rate_limit --features "provider-llamacpp service-rate-limit"
```

## Errors

| Variant | When | Carries |
|---|---|---|
| `Config` | A bound rejected at `RateLimitConfig::new`, a `with_*` call, `validate()`, or `RateLimiter::new` | the invalid bound, as a message |
| `QueueFull` | `acquire`, when the waiter queue is already at `max_waiters`: the call is refused instead of queued | `waiters`, `max_waiters` |
| `Busy` | `try_acquire` only, when no concurrency slot is free right now | `in_flight`, `max_concurrent` |
| `Exhausted` | `try_acquire` when the bucket is empty, or `acquire` when the wait budget elapses before a token frees up | `retry_after`, the time until the next whole token |
| `Observer` | A `RateLimitObserver` refused an acquire crossing `warn_fraction`; the already-taken slot is released before the error returns | the wrapped `PluginError` |
| `Lock` | The bucket mutex was poisoned by a panicking lock holder; the limiter fails closed rather than admit from a half-updated bucket | a message |

`Exhausted.retry_after` is the one actionable value the crate reports; nothing else in `RateLimitError` is meant to drive a retry loop, because retrying is the caller's decision, not the limiter's. `From<RateLimitError> for PluginError` is a lossy interop edge for code that already threads `PluginError`: `Config` becomes `PluginError::Validation`, `Observer(e)` round-trips `e` unchanged, and the three throttle variants collapse into `PluginError::Internal` with their numbers kept only in the message. A caller that needs the typed `retry_after` matches on `RateLimitError` before converting.

## Capacity

| Structure | Bound | At-cap policy | Usage gauge |
|---|---|---|---|
| Token bucket | Two integers, capped at `burst * 1_000` thousandths | Saturates; a long idle period never accumulates unbounded credit | `RateLimitUsage::available_tokens` |
| Concurrency permits | `max_concurrent`, validated non-zero | Callers wait for a free slot | `RateLimitUsage::in_flight` |
| Waiter queue | `max_waiters`, validated non-zero | Reject, never evict: an over-cap `acquire` returns `QueueFull` immediately, before it ever parks | `RateLimitUsage::waiting` |
| `RateLimitConfig::observers` | Caller-sized at construction | Fixed; not traffic-growing | n/a |

The limiter retains no per-request data: no request clone, no key, no log, no map. Only concurrent callers make it grow, and both axes of that are capped above.

## If you want automatic rejection

`RateLimiter` never fails a turn by itself; a caller decides what to do with `Busy`, `QueueFull`, or `Exhausted`. Wanting automatic, hook-based rejection instead of pacing is a legitimate but different capability, and it is ten lines around `try_acquire`, not a crate feature:

```rust,name=A reject-only wrapper around try_acquire
struct RejectOnBusy(Arc<RateLimiter>);

impl CucaPlugin for RejectOnBusy {
    fn name(&self) -> &'static str {
        "reject-on-busy"
    }

    fn on_request(&self, _req: &mut UnifiedRequest) -> Result<(), PluginError> {
        self.0.try_acquire().map(drop).map_err(PluginError::from)
    }
}
```

This spends a token and immediately drops the permit, so it never holds a concurrency slot across the turn; it can only fail a request, never pace one. That is exactly why `RateLimiter` itself does not ship this shape: a partial hook implementation that silently dropped the concurrency half of the capability would be exactly the silent no-op [Plugins and services](@/concepts/plugin-layering.md) rules out.

## No server feedback

There is no 429 or `Retry-After` seam in this crate to plug into: a non-2xx response becomes `CucaError::Http { status, body }` inside the provider's dispatch future and is propagated by `?` before any stream or plugin hook exists, and no provider adapter keeps the response's header map around afterward. `RateLimiter` therefore never learns that a request it admitted was itself rejected upstream.

If a turn does come back `CucaError::Http { status: 429, .. }`, the limiter's own bounds are simply stale for that provider. There is no `penalize()` call to drain the bucket in response; widen `interval` or lower `max_requests` on the `RateLimitConfig` and rebuild the limiter instead.

## Runtime coupling

`acquire` needs a Tokio reactor: it calls `tokio::time::sleep` and waits on a `tokio::sync::Semaphore`. `try_acquire` and `usage` need neither, so they also work from a non-Tokio caller.
