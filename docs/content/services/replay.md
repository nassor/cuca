+++
title = "Replay"
description = "Deterministic re-materialization of a recorded session trajectory as an AgentResponseStream, with no provider dispatch."
template = "page.html"
weight = 4
+++

# Replay

<dl class="page-facts">
<dt>In one line</dt>
<dd>Re-materializes a recorded session trajectory as the same AgentResponseStream a live provider turn produces, with no network call.</dd>
<dt>You need</dt>
<dd>The <code>service-replay</code> feature, which enables <code>plugin-session-log</code>, plus one <code>provider-*</code> feature.</dd>
<dt>Read this if</dt>
<dd>You are replaying a recorded turn for a regression fixture, an offline eval, or a fork-point comparison.</dd>
</dl>

## Feature and entry type

| Item | Value |
|---|---|
| Feature flag | `service-replay` |
| Declared plugin edge | `plugin-session-log`, hard: `service-replay = ["plugin-session-log"]` |
| Entry type | `SessionReplay` |
| Other public types | `ReplayConfig`, `ReplayTrajectory`, `ReplayTurn`, `ReplayUsage`, `ReplayCompletion`, `ReplayNote` |

`SessionReplay` is an explicit-call service. It never implements `CucaPlugin`, so passing it to `register_plugin` is a compile error. The session-log module does not name replay, so the dependency runs one way. Replay reads a recording through the `SessionBackend` seam and never writes; writing a trajectory stays with `SessionStorePlugin`.

## Entry points

Every entry point is a synchronous method call.

### Bind a backend

- `SessionReplay::new(backend)` binds an `Arc<dyn SessionBackend>` with the default `ReplayConfig`.
- `SessionReplay::with_config(backend, config)` validates `config` and returns `Result<SessionReplay, PluginError>`.
- The backend is `SessionLogPlugin::backend()` for a registered log, or `FileBackend::new(dir)` for a directory of `.cslog` files.

### Load a trajectory

| Method | Retains |
|---|---|
| `load(session_id)` | the whole recorded session, in append order |
| `load_prefix(session_id, upto_sequence)` | records with `sequence <= upto_sequence` |
| `load_at_point(point_id)` | the same prefix, addressed as `"{session_id}:{sequence}"`, the string `SessionStorePlugin::fork_session` takes |

- Every load reads through `SessionBackend::replay`, which has no ranged read. A fork-point load therefore reads the whole session before it filters.
- A session id the backend does not know loads as an empty trajectory with `InMemoryBackend` and `FileBackend`. `load_prefix` and `load_at_point` refuse it because no record matches.
- `load_at_point` splits at the last `:`, since a fork session id can itself contain `:`.

### Segmentation into turns

- A turn closes on the `Latency` then `TokenUsage` pair that `SessionLogPlugin::on_response_complete` always appends.
- Records after the last pair form a final turn with `completion() == None` and `is_complete() == false`. They are never merged into the previous turn and never dropped.
- Each `SessionEvent` lands in the turn it belongs to:
  - `SystemPrompt` goes to `ReplayTurn::system_prompts`.
  - `Message` goes to `ReplayTurn::messages`.
  - `Reasoning`, `Output`, `ToolCall` and `ToolResult` become blocks (`Thinking`, `Text`, `ToolCall`, `ToolResult`).
  - `Latency` and `TokenUsage` fill `ReplayTurn::completion`, a `ReplayCompletion` with `duration_ms`, `prompt_tokens` and `completion_tokens`.
  - `ModelSwap` and `Fork` go to `ReplayTurn::notes` as `ReplayNote::ModelSwap { from, to, reason }` and `ReplayNote::Fork { from_point, to_session }`.

### Read and stream a turn

| Method | Blocks | Errors on |
|---|---|---|
| `ReplayTrajectory::stream_turn(index)` | clones one turn's blocks; the trajectory stays replayable | an index that names no turn |
| `ReplayTrajectory::into_stream()` | moves every turn's blocks, concatenated in record order | a trajectory with no turns |
| `ReplayTurn::stream()` | clones this turn's blocks | never |
| `ReplayTurn::into_stream()` | moves this turn's blocks | never |

- `ReplayTrajectory` also exposes `session_id()`, `len()`, `is_empty()`, `usage()`, `turns()` and `turn(index)`.
- `ReplayTurn` also exposes `blocks()`, `completion()`, `notes()`, `sequence_range()` and `is_complete()`.
- `ReplayTurn::response(model, provider)` rebuilds the aggregated `UnifiedResponse` for code written against `on_response_complete`. The caller supplies `model` and `provider` because no record holds them. `finish_reason`, `prompt_cache_usage` and `usage` are always `None`, and a turn with no terminator reports zero tokens and zero duration.
- The stream is order-deterministic, not time-deterministic. Its `poll_next` never sleeps, never returns `Poll::Pending` and never yields `Err`, so two loads of one trajectory give identical block sequences. The recorded latency is data in `ReplayCompletion::duration_ms`.

### Hand-off: replayed blocks skip the plugin pipeline

Replay streams and responses go straight to the caller. No registered plugin's `on_stream_chunk` or `on_response_complete` runs on them, because no `CucaClient::generate_stream` call routes them. A consumer that needs those hooks calls them itself.

### Fidelity gaps

Replay cannot return data the recording never held, and nothing is synthesized in its place.

- `MessageContentBlock::ImageBase64` is never recorded, because `SessionLogPlugin::on_stream_chunk` maps it to no event.
- `SessionEvent::ToolResult` carries `stdout`, `stderr` and `exit_code`, which `MessageContentBlock::ToolResult` has no field for. A caller that needs them reads the raw records with `SessionBackend::replay`.

## Configuration

`ReplayConfig::new(max_records, max_turn_blocks, warn_fraction)` is the validating constructor. The fields are public, so `validate()` re-checks a struct literal, and `SessionReplay::with_config` calls it.

| Key | Default | Valid values |
|---|---|---|
| `max_records` | `65536` (`ReplayConfig::DEFAULT_MAX_RECORDS`) | non-zero |
| `max_turn_blocks` | `4096` (`ReplayConfig::DEFAULT_MAX_TURN_BLOCKS`) | non-zero |
| `warn_fraction` | `Some(0.9)` | `None`, or finite and in `(0.0, 1.0]` |

- `max_records` is the most records one load may retain.
- `max_turn_blocks` is the most content blocks one turn may retain.
- `warn_fraction` is the fraction of `max_records` at which `ReplayUsage::near_cap` turns `true`; `None` keeps it `false`.

## Caps and at-cap policy

| Bound | At-cap policy | Usage gauge |
|---|---|---|
| `max_records`, records one load retains | refuse with `PluginError::Validation`, never truncate | `usage().records` against `usage().max_records` |
| `max_turn_blocks`, blocks one turn retains | refuse with `PluginError::Validation`, never truncate | `usage().blocks`, the total across all turns |

- A silently shortened trajectory is a wrong fixture and a wrong eval, so replay refuses.
- `ReplayUsage` is computed once at load and holds `records`, `max_records`, `blocks`, `turns` and `near_cap`. A loaded trajectory never grows, so the warning is a field to read and no message is injected.
- The pre-read bound belongs to the backend. `SessionBackend::replay` returns the whole `Vec<SessionRecord>` before `ReplayConfig` filters it, so `InMemoryBackend::DEFAULT_MAX_RECORDS` caps an in-memory session and disk caps a `FileBackend` session.
- `load*` moves that `Vec` into the trajectory, so the steady state is one trajectory and not a trajectory plus its source records.
- `SessionReplay` holds one `Arc` and one `Copy` config, so it does not grow with traffic.

## Errors

Replay has no error type of its own. Every method returns `PluginError`, and each refusal is `PluginError::Validation { schema, message }`.

| `schema` | Raised by | When |
|---|---|---|
| `max_records` | `ReplayConfig::new`, `validate`, `with_config`, `load*` | the cap is zero, or the retained records exceed it |
| `max_turn_blocks` | `ReplayConfig::new`, `validate`, `with_config`, `load*` | the cap is zero, or one turn's blocks exceed it |
| `warn_fraction` | `ReplayConfig::new`, `validate`, `with_config` | the value is not finite or lies outside `(0.0, 1.0]` |
| `sequence` | `load*` | record sequences are not strictly increasing |
| `point_id` | `load_prefix`, `load_at_point` | no record sits at or below the sequence; or the id has no `:`, an empty session id, or a sequence that is not a `u64` |
| `turn_index` | `ReplayTrajectory::stream_turn` | the index names no turn, including every index on an empty trajectory |
| `session_id` | `ReplayTrajectory::into_stream` | the trajectory holds no turns |

- `SessionBackend::replay` errors pass through `load*` unchanged.
- A materialized stream never yields `Err`, because every failure is raised at load time.

## Example

This program replays turn 0 of session `demo` from a directory of `.cslog` files that a `SessionLogPlugin` wrote earlier. It sends no request, so it needs no server.

```rust,name=Replay one recorded turn with no provider
use std::sync::Arc;

use cuca::{FileBackend, SessionReplay};
use tokio_stream::StreamExt;

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let replay = SessionReplay::new(Arc::new(FileBackend::new("sessions")?));
    let trajectory = replay.load("demo")?;
    let mut stream = trajectory.stream_turn(0)?;
    while let Some(block) = stream.next().await {
        println!("{:?}", block?);
    }
    Ok(())
}
```

`examples/replay.rs` records one live turn through a `SessionLogPlugin` over a temp-directory `FileBackend`, drops the client, then replays from a fresh `FileBackend` over the same directory. It also rebuilds the `UnifiedResponse` and replays a fork-point prefix. It needs a `llama-server` on port 1234 with the demo model loaded; `CUCA_BASE_URL` and `CUCA_MODEL` point it at any OpenAI-compatible server.

```bash,name=Runs the same on all three platforms
cargo run --example replay --features "provider-llamacpp service-replay"
```

## Related pages

- [Session log](@/plugins/session-log.md) records the trajectories replay reads.
- [Services](@/services/_index.md) lists every service and its plugin edge.
- [Plugins and services](@/concepts/plugin-layering.md) explains the explicit-call tier.
- [Memory discipline](@/concepts/memory-discipline.md) explains the bounded-by-default rule.
