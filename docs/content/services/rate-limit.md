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

```rust,name=Pace a fan-out of turns through one shared limiter
use std::io::{Write, stdout};
use std::sync::Arc;
use std::time::{Duration, Instant};

use cuca::types::{MessageContentBlock, ProviderEndpoint};
use cuca::{CucaClient, RateLimitConfig, RateLimitError, RateLimiter, UnifiedRequest};
use tokio_stream::StreamExt;

/// Turns fanned out concurrently after the first sequential one.
const FANNED_OUT_TURNS: usize = 5;

/// One labelled gauge reading.
fn print_usage(label: &str, limiter: &RateLimiter) {
    match limiter.usage() {
        Ok(usage) => println!(
            "  {label:<22} tokens={} in_flight={} waiting={}",
            usage.available_tokens, usage.in_flight, usage.waiting
        ),
        Err(error) => println!("  {label:<22} usage unavailable: {error}"),
    }
}

/// A short turn. The reply is one word: the pacing is the point, and a
/// reasoning model needs a token budget big enough to finish thinking and
/// still say it.
fn request(model: &str, turn: usize) -> UnifiedRequest {
    UnifiedRequest::new(model)
        .add_system_message("You are concise.")
        .add_user_message(format!("Reply with exactly: turn {turn}"))
        .set_max_tokens(96)
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Base URL and model come from the environment so the example runs
    // against any OpenAI-compatible server; the defaults target a local
    // llama.cpp server (see the module docs for the override recipe).
    let base_url =
        std::env::var("CUCA_BASE_URL").unwrap_or_else(|_| "http://127.0.0.1:1234/v1".to_string());
    let model = std::env::var("CUCA_MODEL").unwrap_or_else(|_| "google/gemma-4-e4b".to_string());

    // Stage 1: the bounds. Every one of them is validated at construction, so
    // a zero rate or a zero cap fails here instead of being clamped.
    let limiter = Arc::new(RateLimiter::new(
        RateLimitConfig::new(1, Duration::from_secs(45), 1, 8)?
            .with_max_wait(Duration::from_secs(600))?,
    )?);
    let config = limiter.config();
    println!(
        "Limiter: {} requests / {:?}, burst {}, {} concurrent, {} queued, {:?} wait budget",
        config.max_requests,
        config.interval,
        config.burst,
        config.max_concurrent,
        config.max_waiters,
        config.max_wait
    );
    print_usage("before any turn", &limiter);

    // Stage 2: one client for every turn. The limiter stays a separate object:
    // it has no hooks, so `register_plugin(limiter)` would not compile.
    let client = Arc::new(
        CucaClient::builder()
            .with_provider(ProviderEndpoint::LlamaCpp)
            .with_base_url(base_url.clone())
            .build()?,
    );

    let started = Instant::now();

    // Stage 3: one turn end to end, which doubles as the reachability check.
    // The permit is taken before the dispatch and dropped after the drain.
    let permit = limiter.acquire().await?;
    let mut stream = match client.generate_stream(request(&model, 0)).await {
        Ok(stream) => stream,
        Err(error) => {
            println!("\nNo server answered at {base_url}: {error}");
            println!("Start llama-server there, or set CUCA_BASE_URL, then run this again.");
            return Ok(());
        }
    };
    print!("\n  turn 0 replies: ");
    let mut thinking = 0usize;
    while let Some(chunk) = stream.next().await {
        match chunk {
            Ok(MessageContentBlock::Text(text)) => {
                print!("{text}");
                stdout().flush()?;
            }
            Ok(MessageContentBlock::Thinking { .. }) => thinking += 1,
            Ok(_) => {}
            Err(error) => {
                print!("[the stream ended early: {error}]");
                break;
            }
        }
    }
    println!(" (+{thinking} thinking blocks)");
    drop(permit);

    // Stage 4: the remaining turns, all dispatched at once. The limiter is
    // what decides when each of them actually reaches the provider.
    println!("\nFanning out {FANNED_OUT_TURNS} turns over the same limiter");
    let mut handles = Vec::with_capacity(FANNED_OUT_TURNS);
    for turn in 1..=FANNED_OUT_TURNS {
        let client = Arc::clone(&client);
        let limiter = Arc::clone(&limiter);
        let model = model.clone();
        handles.push(tokio::spawn(async move {
            let permit = match limiter.acquire().await {
                Ok(permit) => permit,
                // QueueFull past `max_waiters`, Exhausted past `max_wait`.
                Err(error) => {
                    println!("  turn {turn}: refused, {error}");
                    return;
                }
            };
            let admitted_ms = started.elapsed().as_millis();
            let reading = match limiter.usage() {
                Ok(usage) => format!(
                    "tokens={} in_flight={} waiting={}",
                    usage.available_tokens, usage.in_flight, usage.waiting
                ),
                Err(error) => format!("usage unavailable: {error}"),
            };

            let mut reply = String::new();
            let mut thinking = 0usize;
            match client.generate_stream(request(&model, turn)).await {
                Ok(mut stream) => {
                    while let Some(chunk) = stream.next().await {
                        match chunk {
                            Ok(MessageContentBlock::Text(text)) => reply.push_str(&text),
                            Ok(MessageContentBlock::Thinking { .. }) => thinking += 1,
                            Ok(_) => {}
                            Err(error) => {
                                println!("  turn {turn}: the stream ended early, {error}");
                                break;
                            }
                        }
                    }
                }
                Err(error) => println!("  turn {turn}: dispatch failed, {error}"),
            }
            println!(
                "  turn {turn}: admitted at {admitted_ms} ms ({reading}), replied {:?} \
                 (+{thinking} thinking) at {} ms",
                reply.trim(),
                started.elapsed().as_millis()
            );
            // Explicit for the demo: the slot comes back here, not when the
            // stream ended.
            drop(permit);
        }));
    }

    // Stage 5: the gauge, sampled while the fan-out runs. The peak is the
    // observable proof that the caps held.
    let mut peak_in_flight = 0;
    let mut peak_waiting = 0;
    loop {
        if let Ok(usage) = limiter.usage() {
            peak_in_flight = peak_in_flight.max(usage.in_flight);
            peak_waiting = peak_waiting.max(usage.waiting);
        }
        if handles.iter().all(tokio::task::JoinHandle::is_finished) {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    for handle in handles {
        handle.await?;
    }
    println!(
        "\nPeak while the fan-out ran: in_flight={peak_in_flight} (cap {}), \
         waiting={peak_waiting} (cap {})",
        limiter.config().max_concurrent,
        limiter.config().max_waiters
    );
    println!(
        "{} turns took {} ms through a {}-per-{:?} bucket",
        FANNED_OUT_TURNS + 1,
        started.elapsed().as_millis(),
        limiter.config().max_requests,
        limiter.config().interval
    );

    // Stage 6: the non-blocking path. `try_acquire` never waits: it spends a
    // token when one is there and otherwise reports how long until the next
    // one, without ever reaching the provider.
    println!("\nDraining the bucket with try_acquire");
    loop {
        match limiter.try_acquire() {
            Ok(permit) => {
                println!("  admitted: one token spent, nothing dispatched");
                drop(permit);
            }
            Err(RateLimitError::Exhausted { retry_after }) => {
                println!(
                    "  refused: the bucket is empty, retry after {} ms",
                    retry_after.as_millis()
                );
                break;
            }
            // Busy (every slot held) or Lock (poisoned bucket); neither can
            // happen here, since each permit above is dropped immediately.
            Err(error) => {
                println!("  refused: {error}");
                break;
            }
        }
    }
    print_usage("after the demo", &limiter);
    Ok(())
}
```

```text,name=Expected output
Limiter: 1 requests / 45s, burst 1, 1 concurrent, 8 queued, 600s wait budget
  before any turn        tokens=1 in_flight=0 waiting=0

  turn 0 replies: turn 0 (+49 thinking blocks)

Fanning out 5 turns over the same limiter
  turn 1: admitted at 45030 ms (tokens=0 in_flight=1 waiting=4), replied "turn 1" (+59 thinking) at 75968 ms
  turn 2: admitted at 90054 ms (tokens=0 in_flight=1 waiting=3), replied "turn 2" (+53 thinking) at 126285 ms
  turn 3: admitted at 135061 ms (tokens=0 in_flight=1 waiting=2), replied "turn 3" (+58 thinking) at 166749 ms
  turn 4: admitted at 180070 ms (tokens=0 in_flight=1 waiting=1), replied "turn 4" (+50 thinking) at 211396 ms
  turn 5: admitted at 225078 ms (tokens=0 in_flight=1 waiting=0), replied "turn 5" (+46 thinking) at 255461 ms

Peak while the fan-out ran: in_flight=1 (cap 1), waiting=5 (cap 8)
6 turns took 255479 ms through a 1-per-45s bucket

Draining the bucket with try_acquire
  refused: the bucket is empty, retry after 14625 ms
  after the demo         tokens=0 in_flight=0 waiting=0
```

`google/gemma-4-12b-qat` produced that run: the replies and the thinking-block counts are the model's, while the 45 second admission schedule is the bucket's and holds for any model.

## Try it

`examples/rate_limit.rs` is the program above. Six turns share one client and one `Arc<RateLimiter>` sized at one request per 45 seconds, with `max_concurrent = 1`, `max_waiters = 8`, and a 600 second wait budget. The bucket refills more slowly than the model answers, so the admission timestamps are the refill schedule. Each fanned-out turn prints its admission time and a `usage()` reading, a peak `in_flight`/`waiting` line follows the fan-out, and the run ends by draining the bucket with `try_acquire` until it reports `Exhausted` with a `retry_after`. It needs a `llama-server` on port 1234 with the demo model loaded; `CUCA_BASE_URL` and `CUCA_MODEL` retarget it at any OpenAI-compatible server.

```bash,name=Runs the same on all three platforms
cargo run --example rate_limit --features "provider-llamacpp service-rate-limit"
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

## Errors

| Variant | When | Carries |
|---|---|---|
| `Config` | A bound rejected at `RateLimitConfig::new`, a `with_*` call, `validate()`, or `RateLimiter::new` | the invalid bound, as a message |
| `QueueFull` | `acquire`, when the waiter queue is already at `max_waiters`: the call is refused instead of queued | `waiters`, `max_waiters` |
| `Busy` | `try_acquire` only, when no concurrency slot is free right now | `in_flight`, `max_concurrent` |
| `Exhausted` | `try_acquire` when the bucket is empty, or `acquire` when the wait budget elapses, whether it was waiting for a slot or for a token | `retry_after`, the time until the next whole token, which is zero when a full bucket was never the obstacle |
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
