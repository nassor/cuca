+++
title = "Scheduler"
description = "The per-backend priority scheduler and the dispatcher over several backends: placement, aging, a reserved lane, an adaptive parallelism tuner, failover and a circuit breaker."
template = "page.html"
weight = 7
+++

# Scheduler

<dl class="page-facts">
<dt>In one line</dt>
<dd>A priority queue with aging in front of each backend, and a <code>Dispatcher</code> that places every call on the least loaded backend, fails over when one is unreachable, and tunes each backend's parallelism.</dd>
<dt>You need</dt>
<dd>The <code>service-scheduler</code> feature, a Tokio runtime, plus one <code>provider-*</code> feature.</dd>
<dt>Read this if</dt>
<dd>Several callers share one or more model backends and you want ordering, a parallelism limit and failover instead of a free for all.</dd>
</dl>

## Feature and entry type

| Item | Value |
|---|---|
| Feature flag | `service-scheduler = ["dep:tokio", "dep:tracing"]` |
| Declared plugin edge | none |
| Entry type | `Dispatcher` |
| Other public types | `BackendRegistry`, `BackendSpec`, `BackendClient`, `ClientFactory`, `Backend`, `DispatcherConfig`, `Completion`, `DispatchError`, `Scheduler`, `SchedulerConfig`, `SchedulerError`, `SchedulerPool`, `CallMeta`, `CallSlot`, `CallReport`, `CallOutcome`, `Priority`, `Lane`, `BackendId`, `DispatchGate`, `OpenGate`, `Clock`, `SystemClock`, `Load`, `SchedulerSnapshot` and the snapshot and tuner value types |

`Dispatcher` is an explicit-call service. It never implements `CucaPlugin`. A hook cannot carry the capability: `CucaPlugin` hooks are synchronous, and a queue has to wait. The service owns `tracing` itself and does not depend on `plugin-telemetry`.

## Entry points

| Method | Behavior |
|---|---|
| `BackendRegistry::new(gate, changes)` | builds the registry that owns one `Scheduler` per backend id, for the life of the process |
| `Dispatcher::new(registry, specs, config)` | joins the backends in `specs` to the registry and builds the dispatcher |
| `Dispatcher::call(request, meta).await` | places the call, waits for a slot, streams the reply and returns a `Completion` |
| `Dispatcher::set_backends(registry, specs)` | swaps the backend list; on an error the old list stays |
| `Dispatcher::rebalance(interval)` | moves waiting calls to a lighter backend, at most once per `interval` |
| `Dispatcher::backends()`, `ceiling()` and `allowed_backends()` | lists the backends, sums their parallelism ceilings, and reads the bit mask of the current list |
| `Scheduler::acquire(meta).await` | the same queue without a dispatcher: enqueue, wait for placement, return a `CallSlot` |
| `Scheduler::snapshot()` | reads a `SchedulerSnapshot`: running, waiting and recent calls, buckets, totals and the tuner |

- A call carries a `CallMeta`: a free `kind` label, a `Priority`, a `Lane`, a `subject`, and optional `reference`, `task` and `reruns`. `Priority` is `Low`, `Medium`, `High` or `Urgent`, with starting values 1, 10, 100 and 10 000. A call with `reruns` above zero runs as `Urgent`.
- A waiting call gains one point of value for every call that completes ahead of it, so a low priority call is never starved. The highest value runs first, and the lowest id breaks a tie.
- `Lane::Reserved` calls get a reserved lane. From `reserved_lane_from_level` running slots upward, bucket 0 serves the best reserved call first and a shared call only when no reserved call waits, and the other buckets never serve a reserved call. Below that level the lanes behave alike. `Scheduler::hold_reserved_for` holds reserved calls back for a duration, and `clear_reserved_hold` ends the hold.
- A `DispatchGate` lets the caller pause every queue (`paused`, which leaves only urgent calls dispatching) and cancel a waiting call whose inputs changed (`is_stale`, which surfaces as `InputsChanged`). `OpenGate` never pauses and never reports a stale task.
- A `BackendSpec` names a `BackendId`, a `BackendClient` and optionally a model tag and a `ThinkingConfig` that overwrite the request's own. A `Ready` client is used as given. A `Lazy` client is a `ClientFactory` that builds the `CucaClient` once, before the first slot is reserved and outside the call deadline. A failed build opens that backend's circuit breaker.
- `Dispatcher::call` places a call on the backend with the lightest `Load` among those whose breaker admits. A `CucaError::Transport` before any stream exists opens that breaker for `unreachable_cooldown`, and the call goes once to each other backend that still admits. Any other error, including one in the middle of a stream, returns as `DispatchError::Call` with no failover.
- Each backend starts at one call at a time. The tuner earns every extra level by measuring a trial one level up or down against the current level, and keeps a step up only on a gain of at least 10 percent in throughput. The first experiment is due 15 minutes after boot, the interval doubles up to 24 hours, and a failed call during a trial reverts it. `SchedulerSnapshot` reports the tuner's phase, history and reasons.
- The service emits two `tracing` debug spans under the target `cuca::scheduler`: `scheduler.wait` around the wait for a slot and `scheduler.stream` around the reply. It opens no call level span; the caller owns that.

### Hand-off: finish every slot, drive the rebalance

`Dispatcher::call` finishes its own `CallSlot`. A caller that uses `Scheduler::acquire` directly must call `CallSlot::finish` with a `CallReport`; a dropped slot counts as `Cancelled` and teaches the tuner nothing. The caller also calls `Dispatcher::rebalance` on its own beat, and calls `notify_waiters()` on the gate's `wake` notifier when a pause begins or ends or a task goes stale. The service starts no timer of its own except the sleeper behind `hold_reserved_for`.

### Not covered

The service never retries a failed call, and it has no 429 or `Retry-After` feedback.

## Configuration

`SchedulerConfig::new(ceiling)` and `DispatcherConfig::new(scheduler)` are the base constructors. `with_max_waiting`, `with_timeout` and `with_unreachable_cooldown` override a defaulted field. `SchedulerConfig::validate()` re-checks every bound, since the fields are public.

| Key | Type | Default | Valid values |
|---|---|---|---|
| `SchedulerConfig::ceiling` | `u32` | required | 1 to 64 |
| `SchedulerConfig::max_waiting` | `usize` | `256` | non-zero |
| `SchedulerConfig::reserved_lane_from_level` | `u32` | `4` | non-zero |
| `DispatcherConfig::scheduler` | `SchedulerConfig` | required | a validated config |
| `DispatcherConfig::timeout` | `Option<Duration>` | `None` | any; `None` waits without a deadline |
| `DispatcherConfig::unreachable_cooldown` | `Duration` | 20 seconds | non-zero |

- `ceiling` is the most calls one backend runs at once. `Dispatcher::ceiling` sums it over the backends.
- `max_waiting` is the most calls one backend queues. The check is made before a call is queued.
- `timeout` bounds one call from the slot to the end of the stream.
- `unreachable_cooldown` is how long a tripped breaker keeps a backend out of placement.

## Caps and at-cap policy

| Bound | At-cap policy | Gauge |
|---|---|---|
| `max_waiting` calls per backend | refuses, never evicts: an over-cap `enqueue` returns `QueueFull` | `SchedulerSnapshot` waiting list |
| 32 distinct call kinds in the totals | the 33rd and later kinds are counted under `"other"` | `SchedulerSnapshot::totals` |
| 64 recent completed calls | the oldest entry is dropped | `SchedulerSnapshot` recent list |
| tuner: 128 samples and 32 history entries | the oldest entry is dropped | `TunerSnapshot` |
| 64 backends per `SchedulerPool` | joining a 65th backend returns `DispatchError::Config` | `Dispatcher::backends` |

- A backend is never removed from a registry. A backend id listed again gets its existing scheduler, breaker and built client back.

## Errors

`SchedulerError` is the error type of `SchedulerConfig` and `Scheduler`. `DispatchError` is the error type of `Dispatcher` and `DispatcherConfig`. Both implement `std::error::Error` and `Clone`. `DispatchError::source()` returns the inner `CucaError` for `Call` and `None` otherwise.

- `SchedulerError::Config(String)`: a bound was rejected by `SchedulerConfig::new`, a `with_*` call or `validate()`.
- `SchedulerError::QueueFull { backend, max_waiting }`: `enqueue` found the queue at `max_waiting`.
- `SchedulerError::InputsChanged`: the gate reported the waiting task stale.
- `DispatchError::Config(String)`: a bad configuration, a backend listed twice, or more than 64 backends.
- `DispatchError::QueueFull { backend }`: the chosen backend's queue was full.
- `DispatchError::Unreachable { detail }`: every admitting backend failed to connect, a lazy client failed to build, no breaker admitted, or the list was empty.
- `DispatchError::Timeout { seconds }`: the call ran past `timeout`, counted from the moment it left the queue. No breaker opens and no failover follows.
- `DispatchError::Empty`: the reply held no text and no reasoning.
- `DispatchError::ReasoningOnly { reasoning_chars }`: the reply held reasoning and no text.
- `DispatchError::InputsChanged`: the gate cancelled the call.
- `DispatchError::Call(CucaError)`: the provider failed with anything other than a transport error before the stream, or an error item arrived mid-stream.

`From<SchedulerError> for DispatchError` keeps the variant. Neither type converts to `CucaError` or `PluginError`.

## Example

The program needs a `llama-server` or LM Studio on port 1234 with a model loaded. `CUCA_BASE_URL` and `CUCA_MODEL` point it at any OpenAI-compatible server. It builds a registry and a dispatcher over one backend, sends three calls of three priorities, and prints the `recent` list of the snapshot.

```bash,name=Linux and macOS
cargo run --example scheduler --features provider-llamacpp,service-scheduler
```

```powershell,name=Windows (PowerShell)
cargo run --example scheduler --features provider-llamacpp,service-scheduler
```

Output of a recorded run against LM Studio serving `google/gemma-4-e2b`. The replies and the milliseconds vary with the model and the machine. The order does not: the first call to arrive runs first, then the high call, then the medium one.

```text
Dispatcher over http://127.0.0.1:1234/v1: ceiling 2, one call at a time to start

  low    replied "low" (level 1, waited 0 ms, ran 386 ms)
  medium replied "medium" (level 1, waited 705 ms, ran 302 ms)
  high   replied "high" (level 1, waited 385 ms, ran 319 ms)

Completed calls, newest first
  http://loopback:1234/v1 medium medium level=1 waited_ms=705 run_ms=302
  http://loopback:1234/v1 high   high   level=1 waited_ms=385 run_ms=319
  http://loopback:1234/v1 low    low    level=1 waited_ms=0 run_ms=386
```

## Related pages

- [Services](@/services/_index.md) lists every service and its plugin edge.
- [Plugins and services](@/concepts/plugin-layering.md) explains the explicit-call tier.
- [Memory discipline](@/concepts/memory-discipline.md) explains the bounded-by-default rule.
- [Rate limit](@/services/rate-limit.md) paces one caller's requests instead of ordering many callers.
- [Errors](@/reference/errors.md) lists the crate's error types.
