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
<dd>The <code>service-replay</code> feature, which pulls in <code>plugin-session-log</code>.</dd>
<dt>Read this if</dt>
<dd>You are replaying a recorded turn for a regression fixture, an offline eval, or a fork-point comparison.</dd>
</dl>

## Entry types

`SessionReplay`, `ReplayConfig`, `ReplayTrajectory`, `ReplayTurn`, `ReplayUsage`, `ReplayCompletion`, `ReplayNote`.

## Not a `CucaPlugin`

`SessionReplay` does not implement `CucaPlugin`. Registering it with `register_plugin` is a compile error, not an inert no-op, the same rule that governs `PromptCache`: replay drives a session instead of observing one, there is no live request or chunk to attach to, and no hook signature can return a stream. `SessionReplay::new(backend)` or `with_config(backend, config)` binds an `Arc<dyn SessionBackend>`, taken from `SessionLogPlugin::backend()` for an already-registered log or `FileBackend::new(dir)` for a directory of `.cslog` files.

## Loading a trajectory

| Method | Retains |
|---|---|
| `load(session_id)` | the whole recorded session, in append order |
| `load_prefix(session_id, upto_sequence)` | records with `sequence <= upto_sequence` |
| `load_at_point(point_id)` | the same fork point, addressed by `"{session_id}:{sequence}"`, the string `SessionStorePlugin::fork_session` takes |

Every load reads through `SessionBackend::replay`, so a fork-point load loads the whole trajectory before filtering; there is no ranged read on the backend. Each `SessionEvent` maps onto the trajectory it produced: `SystemPrompt` into `ReplayTurn::system_prompts`, `Message` into `ReplayTurn::messages`, `Reasoning`/`Output`/`ToolCall`/`ToolResult` into blocks, `Latency`/`TokenUsage` into `ReplayTurn::completion`, and `ModelSwap`/`Fork` into `ReplayTurn::notes`.

## Turn segmentation

A turn closes on the `Latency` then `TokenUsage` pair `SessionLogPlugin::on_response_complete` always appends. Records after the last such pair form a final turn with `completion() == None` and `is_complete() == false`, rather than being merged into the previous turn or dropped.

## Streaming a trajectory

| Method | Blocks | Errors on |
|---|---|---|
| `ReplayTrajectory::stream_turn(index)` | clones one turn's blocks; the trajectory stays replayable | `index` naming no turn, including every index on an empty trajectory |
| `ReplayTrajectory::into_stream()` | moves every turn's blocks, concatenated in record order | an empty trajectory |
| `ReplayTurn::stream()` | clones this turn's blocks | never |
| `ReplayTurn::into_stream()` | moves this turn's blocks | never |

The returned `AgentResponseStream` never yields `Err`: every failure is raised at load time, so a materialized stream is guaranteed to run to completion. `ReplayTurn::response(model, provider)` rebuilds the aggregated `UnifiedResponse` shape from the recorded blocks and completion, for callers written against `on_response_complete`'s argument type; `model` and `provider` are caller-supplied because the trajectory does not record them.

## Scope: replayed blocks skip the plugin pipeline

`ReplayTrajectory`/`ReplayTurn` streams and responses are handed back directly. No registered plugin's `on_stream_chunk` or `on_response_complete` runs against them, because there is no `CucaClient::generate_stream` call routing a caller-supplied stream through `PluginStream`. A consumer that needs those hooks to fire calls them itself.

## Fidelity gaps

Two gaps are not errors, because nothing failed: the recording never held the data.

- `MessageContentBlock::ImageBase64` is never recorded. `SessionLogPlugin::on_stream_chunk` maps it to no event, so replay can never emit an image block.
- `SessionEvent::ToolResult`'s `stdout`, `stderr`, and `exit_code` have no field on `MessageContentBlock::ToolResult`, so they are absent from every replayed block stream. A caller that needs them reads the raw records through `SessionBackend::replay` directly.

## Capacity

| | |
|---|---|
| Bound | `ReplayConfig::max_records`, default `65536`, on records retained by one load |
| At-cap policy | Refuse, `PluginError::Validation`, never truncate: a silently shortened trajectory is a wrong fixture and a wrong eval |
| Usage gauge | `ReplayTrajectory::usage().records` against `usage().max_records` |

| | |
|---|---|
| Bound | `ReplayConfig::max_turn_blocks`, default `4096`, on blocks retained by one turn |
| At-cap policy | Refuse, `PluginError::Validation`; one pathological turn cannot be materialized |
| Usage gauge | `ReplayTrajectory::usage().blocks` |

`ReplayConfig::warn_fraction` (default `Some(0.9)`) sets the retained-record fraction of `max_records` at which `ReplayUsage::near_cap` flips; `None` disables the flag. A trajectory is loaded once and never grows afterward, so the flag is a caller-readable field on `ReplayUsage`, not an injected warning message. `ReplayConfig::new` rejects a zero cap or a `warn_fraction` outside `(0.0, 1.0]`.

The pre-read bound belongs to the backend, not to replay: `SessionBackend::replay` returns the whole `Vec<SessionRecord>` before `ReplayConfig` filters it, so an in-memory session is already capped by `InMemoryBackend::DEFAULT_MAX_RECORDS` and a `FileBackend` session by disk. `load*` consumes that `Vec` by value, moving every field into the trajectory, so the steady state is one trajectory, not a trajectory plus its source records.
