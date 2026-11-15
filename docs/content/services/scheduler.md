+++
title = "Scheduler"
description = "Queue calls by priority in front of one or more model backends, pick the least busy backend, and fail over when one is down."
template = "page.html"
weight = 7
+++

# Scheduler

<dl class="page-facts">
<dt>In one line</dt>
<dd>Puts a priority queue in front of each model backend. A <code>Dispatcher</code> sends every call to the least busy backend, fails over when a backend is unreachable, and tunes how many calls each backend runs at once.</dd>
<dt>You need</dt>
<dd>The <code>service-scheduler</code> feature, a Tokio runtime, and one <code>provider-*</code> feature.</dd>
<dt>Read this if</dt>
<dd>Several tasks share one or more model backends, and you want order, a limit on parallel calls, and failover instead of a free-for-all.</dd>
</dl>

## What it does

A backend is one model server. A server can run only a few calls well at once, so extra calls must wait. The scheduler decides who waits and who goes next. Each call has a priority. A low priority call gains value while it waits, so it is never starved.

<div class="dgm">
<div class="dgm-scroll">
<svg viewBox="0 0 660 210" role="img" aria-labelledby="sc-flow-title sc-flow-desc">
  <title id="sc-flow-title">A call is placed on the least busy backend, waits in that backend's priority queue, runs in a slot, and returns a Completion. An unreachable backend sends the call to another.</title>
  <desc id="sc-flow-desc">Five boxes in a row: the call with its metadata, placement on the backend with the lightest load, the backend's priority queue with aging, a run slot limited by the backend's ceiling, and the Completion. An arrow from the run slot back to placement shows failover to another backend when the first one is unreachable before any stream starts.</desc>
  <defs>
    <marker id="sc-flow-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M0 0.6 L7 4 L0 7.4 z" fill="var(--muted-foreground)"/>
    </marker>
  </defs>
  <text class="t-title" x="6" y="20">Dispatcher::call</text>
  <rect class="blk" x="6" y="44" width="118" height="50" rx="6"/>
  <text class="t-sm t-mid" x="65" y="66">your call</text>
  <text class="t-sm t-mid" x="65" y="81">request, meta</text>
  <path class="arw" d="M126 69 H146" marker-end="url(#sc-flow-arrow)"/>
  <rect class="blk blk-ctl" x="148" y="44" width="128" height="50" rx="6"/>
  <text class="t-sm t-mid t-ctl" x="212" y="66">place</text>
  <text class="t-sm t-mid" x="212" y="81">lightest load</text>
  <path class="arw" d="M278 69 H298" marker-end="url(#sc-flow-arrow)"/>
  <rect class="blk blk-ctl" x="300" y="44" width="128" height="50" rx="6"/>
  <text class="t-sm t-mid t-ctl" x="364" y="66">queue</text>
  <text class="t-sm t-mid" x="364" y="81">priority + aging</text>
  <path class="arw" d="M430 69 H450" marker-end="url(#sc-flow-arrow)"/>
  <rect class="blk blk-bnd" x="452" y="44" width="100" height="50" rx="6"/>
  <text class="t-sm t-mid t-bnd" x="502" y="66">run slot</text>
  <text class="t-sm t-mid" x="502" y="81">up to ceiling</text>
  <path class="arw arw-data" d="M554 69 H574" marker-end="url(#sc-flow-arrow)"/>
  <rect class="blk blk-data" x="576" y="44" width="78" height="50" rx="6"/>
  <text class="t-sm t-mid t-data" x="615" y="66">reply</text>
  <text class="t-sm t-mid" x="615" y="81">Completion</text>
  <path class="arw arw-bnd" d="M502 96 V140 H212 V96" marker-end="url(#sc-flow-arrow)"/>
  <text class="t-sm" x="225" y="156">unreachable before any stream: try another backend</text>
  <text class="t-sm" x="6" y="176">Each backend has its own queue, its own ceiling and its own circuit breaker.</text>
  <text class="t-sm" x="6" y="192">Any other error returns to you as DispatchError::Call, with no failover.</text>
</svg>
</div>
<p class="dgm-cap"><b>Placement picks the backend, the queue picks the call.</b> A transport error opens that backend's breaker for <code>unreachable_cooldown</code>, and the call goes once to each other backend that still accepts calls.</p>
</div>

## Try it

This program sends one call through a dispatcher with one backend. It needs a `llama-server` or LM Studio on port 1234 with a model loaded. Change the base URL and model id to match your server.

```rust,name=Send one call through a dispatcher
use std::sync::Arc;

use cuca::request::UnifiedRequest;
use cuca::types::ProviderEndpoint;
use cuca::{
    BackendClient, BackendId, BackendRegistry, BackendSpec, CallMeta, CucaClient, DispatchGate,
    Dispatcher, DispatcherConfig, OpenGate, Priority, SchedulerConfig,
};

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let base_url = "http://127.0.0.1:1234/v1";

    // The registry owns one scheduler and one circuit breaker per backend.
    // OpenGate never pauses a queue and never marks a task stale.
    let gate: Arc<dyn DispatchGate> = Arc::new(OpenGate::default());
    let mut registry = BackendRegistry::new(gate, Arc::new(tokio::sync::watch::Sender::new(0)));

    let client = CucaClient::builder()
        .with_provider(ProviderEndpoint::LlamaCpp)
        .with_base_url(base_url)
        .build()?;
    let spec = BackendSpec {
        id: BackendId::parse(base_url),
        client: BackendClient::Ready(Arc::new(client)),
        model: None,
        thinking: None,
    };
    // At most 2 calls at once on this backend.
    let dispatcher = Dispatcher::new(
        &mut registry,
        vec![spec],
        DispatcherConfig::new(SchedulerConfig::new(2)?),
    )?;

    let request = UnifiedRequest::new("any-model")
        .add_user_message("Reply with exactly: ok")
        .set_max_tokens(96);
    let done = dispatcher
        .call(request, CallMeta::new("demo", Priority::Medium, "first call"))
        .await?;
    println!("{} (level {})", done.text.trim(), done.level);
    Ok(())
}
```

```text,name=Expected output
ok (level 1)
```

The reply text depends on the model. The level does not: every backend starts at one call at a time and raises it only after it measures a gain.

`examples/scheduler.rs` sends three calls of three priorities through one backend and prints the scheduler's list of completed calls. The flags `--base-url` and `--model`, written after `--`, point it at any OpenAI-compatible server.

```bash,name=Runs the same on all three platforms
cargo run --example scheduler --features "provider-llamacpp service-scheduler"
```

The milliseconds and replies change from run to run. The order does not: the call that arrives first runs first, then the high priority call, then the medium one.

```text,name=Example output
Dispatcher over http://127.0.0.1:1234/v1: ceiling 2, one call at a time to start

  low    replied "low" (level 1, waited 0 ms, ran 386 ms)
  medium replied "medium" (level 1, waited 705 ms, ran 302 ms)
  high   replied "high" (level 1, waited 385 ms, ran 319 ms)

Completed calls, newest first
  http://loopback:1234/v1 medium medium level=1 waited_ms=705 run_ms=302
  http://loopback:1234/v1 high   high   level=1 waited_ms=385 run_ms=319
  http://loopback:1234/v1 low    low    level=1 waited_ms=0 run_ms=386
```

## Feature and entry type

| Item | Value |
|---|---|
| Feature flag | `service-scheduler = ["dep:tokio", "dep:tracing"]` |
| Plugin it needs | none |
| Entry type | `Dispatcher` |
| Other public types | `BackendRegistry`, `BackendSpec`, `BackendClient`, `ClientFactory`, `Backend`, `DispatcherConfig`, `Completion`, `DispatchError`, `Scheduler`, `SchedulerConfig`, `SchedulerError`, `SchedulerPool`, `CallMeta`, `CallSlot`, `CallReport`, `CallOutcome`, `Priority`, `Lane`, `BackendId`, `DispatchGate`, `OpenGate`, `Clock`, `SystemClock`, `Load`, `SchedulerSnapshot`, and the snapshot and tuner value types |

A plugin hook cannot do this job. `CucaPlugin` hooks are synchronous, and a queue has to wait. The service uses `tracing` itself and does not depend on `plugin-telemetry`.

## Entry points

| Method | What it does |
|---|---|
| `BackendRegistry::new(gate, changes)` | builds the registry that owns one `Scheduler` per backend id, for the life of the process |
| `Dispatcher::new(registry, specs, config)` | joins the backends in `specs` to the registry and builds the dispatcher |
| `Dispatcher::call(request, meta).await` | places the call, waits for a slot, streams the reply and returns a `Completion` |
| `Dispatcher::set_backends(registry, specs)` | swaps the backend list. On an error the old list stays. |
| `Dispatcher::rebalance(interval)` | moves waiting calls to a lighter backend, at most once per `interval` |
| `Dispatcher::backends()`, `ceiling()`, `allowed_backends()` | lists the backends, sums their ceilings, and reads the bit mask of the current list |
| `Scheduler::acquire(meta).await` | the same queue without a dispatcher: enqueue, wait for a turn, return a `CallSlot` |
| `Scheduler::snapshot()` | reads a `SchedulerSnapshot`: running, waiting and recent calls, buckets, totals and the tuner |

### Describe a call

A call carries a `CallMeta`: a free `kind` label, a `Priority`, a `Lane`, a `subject`, and optional `reference`, `task` and `reruns`.

- `Priority` is `Low`, `Medium`, `High` or `Urgent`, with starting values 1, 10, 100 and 10 000. A call with `reruns` above zero runs as `Urgent`.
- A waiting call gains one point of value for every call that completes ahead of it, so a low priority call is never starved. The highest value runs first, and the lowest id breaks a tie.
- `Lane::Reserved` calls get a reserved lane. From `reserved_lane_from_level` running slots upward, bucket 0 serves the best reserved call first and a shared call only when no reserved call waits, and the other buckets never serve a reserved call. Below that level the lanes behave alike. `Scheduler::hold_reserved_for` holds reserved calls back for a time, and `clear_reserved_hold` ends the hold.

### Pause and cancel

A `DispatchGate` lets you pause every queue and cancel a waiting call whose inputs changed.

- `paused` pauses dispatch. Only urgent calls still go.
- `is_stale` cancels a waiting call. It surfaces as `InputsChanged`.
- `OpenGate` never pauses and never reports a stale task.

### Choose backends

A `BackendSpec` names a `BackendId`, a `BackendClient`, and optionally a model tag and a `ThinkingConfig` that overwrite the request's own.

- A `Ready` client is used as given.
- A `Lazy` client is a `ClientFactory` that builds the `CucaClient` once, before the first slot is reserved and outside the call deadline. A failed build opens that backend's circuit breaker.
- `Dispatcher::call` places a call on the backend with the lightest `Load` among those whose breaker admits it.
- A `CucaError::Transport` before any stream exists opens that breaker for `unreachable_cooldown`, and the call goes once to each other backend that still admits it. Any other error, including one in the middle of a stream, returns as `DispatchError::Call` with no failover.

### How parallelism is tuned

- Each backend starts at one call at a time.
- The tuner tries one level up or down and keeps a step up only if throughput rises by at least 10 percent. A failed call during a trial reverts it.
- The first trial is due 15 minutes after start. The gap between trials doubles up to 24 hours.
- `SchedulerSnapshot` reports the tuner's phase, history and reasons.

### Tracing

The service emits two `tracing` debug spans under the target `cuca::scheduler`: `scheduler.wait` around the wait for a slot and `scheduler.stream` around the reply. It opens no span for the whole call. That is yours.

### Your duties

`Dispatcher::call` finishes its own `CallSlot`. If you use `Scheduler::acquire` directly, you must call `CallSlot::finish` with a `CallReport`. A dropped slot counts as `Cancelled` and teaches the tuner nothing.

You also drive these:

- Call `Dispatcher::rebalance` on your own timer.
- Call `notify_waiters()` on the gate's `wake` notifier when a pause begins or ends, or when a task goes stale.

The service starts no timer of its own, except the sleeper behind `hold_reserved_for`.

### Not covered

The service never retries a failed call, and it has no 429 or `Retry-After` feedback.

## Configuration

`SchedulerConfig::new(ceiling)` and `DispatcherConfig::new(scheduler)` are the base constructors. `with_max_waiting`, `with_timeout` and `with_unreachable_cooldown` change a defaulted field. `SchedulerConfig::validate()` re-checks every bound, since the fields are public.

| Key | Type | Default | Valid values |
|---|---|---|---|
| `SchedulerConfig::ceiling` | `u32` | required | 1 to 64 |
| `SchedulerConfig::max_waiting` | `usize` | `256` | non-zero |
| `SchedulerConfig::reserved_lane_from_level` | `u32` | `4` | non-zero |
| `DispatcherConfig::scheduler` | `SchedulerConfig` | required | a validated config |
| `DispatcherConfig::timeout` | `Option<Duration>` | `None` | any. `None` waits with no deadline. |
| `DispatcherConfig::unreachable_cooldown` | `Duration` | 20 seconds | non-zero |

- `ceiling` is the most calls one backend runs at once. `Dispatcher::ceiling` sums it over the backends.
- `max_waiting` is the most calls one backend queues. The check is made before a call is queued.
- `timeout` limits one call from the slot to the end of the stream.
- `unreachable_cooldown` is how long a tripped breaker keeps a backend out of placement.

## Caps and at-cap policy

| Bound | At the cap | Gauge |
|---|---|---|
| `max_waiting` calls per backend | refuses, never evicts: an over-cap `enqueue` returns `QueueFull` | the waiting list in `SchedulerSnapshot` |
| 32 distinct call kinds in the totals | the 33rd and later kinds are counted under `"other"` | `SchedulerSnapshot::totals` |
| 64 recent completed calls | the oldest entry is dropped | the recent list in `SchedulerSnapshot` |
| tuner: 128 samples and 32 history entries | the oldest entry is dropped | `TunerSnapshot` |
| 64 backends per `SchedulerPool` | joining a 65th backend returns `DispatchError::Config` | `Dispatcher::backends` |

A backend is never removed from a registry. If you list a backend id again, you get its existing scheduler, breaker and built client back.

## Errors

`SchedulerError` is the error type of `SchedulerConfig` and `Scheduler`. `DispatchError` is the error type of `Dispatcher` and `DispatcherConfig`. Both implement `std::error::Error` and `Clone`. `DispatchError::source()` returns the inner `CucaError` for `Call` and `None` otherwise.

- `SchedulerError::Config(String)`: a bound was rejected by `SchedulerConfig::new`, a `with_*` call or `validate()`.
- `SchedulerError::QueueFull { backend, max_waiting }`: `enqueue` found the queue at `max_waiting`.
- `SchedulerError::InputsChanged`: the gate reported the waiting task stale.
- `DispatchError::Config(String)`: a bad configuration, a backend listed twice, or more than 64 backends.
- `DispatchError::QueueFull { backend }`: the chosen backend's queue was full.
- `DispatchError::Unreachable { detail }`: every admitting backend failed to connect, a lazy client failed to build, no breaker admitted a call, or the list was empty.
- `DispatchError::Timeout { seconds }`: the call ran past `timeout`, counted from the moment it left the queue. No breaker opens and no failover follows.
- `DispatchError::Empty`: the reply held no text and no reasoning.
- `DispatchError::ReasoningOnly { reasoning_chars }`: the reply held reasoning and no text.
- `DispatchError::InputsChanged`: the gate cancelled the call.
- `DispatchError::Call(CucaError)`: the provider failed with anything other than a transport error before the stream, or an error item arrived mid-stream.

`From<SchedulerError> for DispatchError` keeps the variant. Neither type converts to `CucaError` or `PluginError`.

## Related pages

- [Services](@/services/_index.md) lists every service and its plugin dependency.
- [Plugins and services](@/concepts/plugin-layering.md) explains the explicit-call tier.
- [Memory discipline](@/concepts/memory-discipline.md) explains the bounded-by-default rule.
- [Rate limit](@/services/rate-limit.md) paces one caller's requests instead of ordering many callers.
- [Errors](@/reference/errors.md) lists the crate's error types.
