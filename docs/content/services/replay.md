+++
title = "Replay"
description = "Play a recorded session back as the same stream a live turn gives, with no provider call."
template = "page.html"
weight = 4
+++

# Replay

<dl class="page-facts">
<dt>In one line</dt>
<dd>Reads a recorded session and gives you the same <code>AgentResponseStream</code> a live turn produced, with no network call.</dd>
<dt>You need</dt>
<dd>The <code>service-replay</code> feature, which turns on <code>plugin-session-log</code>, plus one <code>provider-*</code> feature. A recorded session to read.</dd>
<dt>Read this if</dt>
<dd>You want a regression fixture, an offline eval, or a comparison from a fork point.</dd>
</dl>

## What it does

The [session log](@/plugins/session-log.md) plugin records every turn as a list of numbered records. `SessionReplay` reads those records back, groups them into turns, and turns a turn into a stream. The stream yields the same blocks in the same order. It never sleeps and never fails.

<div class="dgm">
<div class="dgm-scroll">
<svg viewBox="0 0 660 130" role="img" aria-labelledby="rp-flow-title rp-flow-desc">
  <title id="rp-flow-title">The session log writes records, and replay reads them back into turns and streams.</title>
  <desc id="rp-flow-desc">Four boxes in a row: a live turn that the session log plugin records, the session backend that stores the records, SessionReplay that loads them into a trajectory of turns, and the replayed stream. Replay only reads.</desc>
  <defs>
    <marker id="rp-flow-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M0 0.6 L7 4 L0 7.4 z" fill="var(--muted-foreground)"/>
    </marker>
  </defs>
  <text class="t-title" x="6" y="20">Record once, replay many times</text>
  <rect class="blk blk-ctl" x="6" y="40" width="130" height="50" rx="6"/>
  <text class="t-sm t-mid t-ctl" x="71" y="62">live turn</text>
  <text class="t-sm t-mid" x="71" y="77">session log plugin</text>
  <path class="arw arw-ctl" d="M138 65 H162" marker-end="url(#rp-flow-arrow)"/>
  <rect class="blk" x="164" y="40" width="130" height="50" rx="6"/>
  <text class="t-sm t-mid" x="229" y="62">SessionBackend</text>
  <text class="t-sm t-mid" x="229" y="77">numbered records</text>
  <path class="arw" d="M296 65 H320" marker-end="url(#rp-flow-arrow)"/>
  <rect class="blk blk-ctl" x="322" y="40" width="150" height="50" rx="6"/>
  <text class="t-sm t-mid t-ctl" x="397" y="62">SessionReplay::load</text>
  <text class="t-sm t-mid" x="397" y="77">trajectory of turns</text>
  <path class="arw arw-data" d="M474 65 H498" marker-end="url(#rp-flow-arrow)"/>
  <rect class="blk blk-data" x="500" y="40" width="154" height="50" rx="6"/>
  <text class="t-sm t-mid t-data" x="577" y="62">stream_turn(0)</text>
  <text class="t-sm t-mid" x="577" y="77">same blocks, same order</text>
  <text class="t-sm" x="6" y="116">Replay only reads. Writing a session stays with the session log plugin.</text>
</svg>
</div>
<p class="dgm-cap"><b>Replay needs no provider.</b> It reads from the backend and builds the stream locally, so it works offline.</p>
</div>

## Try it

This program replays turn 0 of the session `demo` from a directory of `.cslog` files that a `SessionLogPlugin` wrote earlier. It sends no request.

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

If the recorded reply was the single word `ok`, with no reasoning, the output is:

```text,name=Expected output
Text("ok")
```

`examples/replay.rs` records one live turn, drops the client, then replays from a fresh `FileBackend` over the same directory. It also rebuilds the `UnifiedResponse` and replays a fork-point prefix. It needs a `llama-server` on port 1234. The flags `--base-url` and `--model`, written after `--`, point it at any OpenAI-compatible server.

```bash,name=Runs the same on all three platforms
cargo run --example replay --features "provider-llamacpp service-replay"
```

## Feature and entry type

| Item | Value |
|---|---|
| Feature flag | `service-replay` |
| Plugin it needs | `plugin-session-log`: `service-replay = ["plugin-session-log"]` |
| Entry type | `SessionReplay` |
| Other public types | `ReplayConfig`, `ReplayTrajectory`, `ReplayTurn`, `ReplayUsage`, `ReplayCompletion`, `ReplayNote` |

## Entry points

Every entry point is a plain synchronous method call.

### Bind a backend

- `SessionReplay::new(backend)` binds an `Arc<dyn SessionBackend>` with the default `ReplayConfig`.
- `SessionReplay::with_config(backend, config)` validates `config` and returns `Result<SessionReplay, PluginError>`.
- The backend is `SessionLogPlugin::backend()` for a registered log, or `FileBackend::new(dir)` for a directory of `.cslog` files.

### Load a trajectory

| Method | What you get |
|---|---|
| `load(session_id)` | the whole recorded session, in order |
| `load_prefix(session_id, upto_sequence)` | the records with `sequence <= upto_sequence` |
| `load_at_point(point_id)` | the same prefix, addressed as `"{session_id}:{sequence}"`, the string `SessionStorePlugin::fork_session` takes |

- A load reads the whole session through `SessionBackend::replay`, then filters. There is no ranged read.
- A session id the backend does not know loads as an empty trajectory with `InMemoryBackend` and `FileBackend`. `load_prefix` and `load_at_point` refuse it, because no record matches.
- `load_at_point` splits at the last `:`, since a fork session id can itself contain `:`.

### How records become turns

- A turn ends at the `Latency` then `TokenUsage` pair that `SessionLogPlugin::on_response_complete` always appends.
- Records after the last pair form a final turn with `completion() == None` and `is_complete() == false`. They are kept, not merged into the previous turn and not dropped.
- Each `SessionEvent` lands in the turn it belongs to:
  - `SystemPrompt` goes to `ReplayTurn::system_prompts`.
  - `Message` goes to `ReplayTurn::messages`.
  - `Reasoning`, `Output`, `ToolCall` and `ToolResult` become blocks (`Thinking`, `Text`, `ToolCall`, `ToolResult`).
  - `Latency` and `TokenUsage` fill `ReplayTurn::completion`, a `ReplayCompletion` with `duration_ms`, `prompt_tokens` and `completion_tokens`.
  - `ModelSwap` and `Fork` go to `ReplayTurn::notes` as `ReplayNote::ModelSwap { from, to, reason }` and `ReplayNote::Fork { from_point, to_session }`.

### Read and stream a turn

| Method | Blocks | Errors on |
|---|---|---|
| `ReplayTrajectory::stream_turn(index)` | clones one turn's blocks, so the trajectory stays replayable | an index that names no turn |
| `ReplayTrajectory::into_stream()` | moves every turn's blocks, joined in record order | a trajectory with no turns |
| `ReplayTurn::stream()` | clones this turn's blocks | never |
| `ReplayTurn::into_stream()` | moves this turn's blocks | never |

- `ReplayTrajectory` also has `session_id()`, `len()`, `is_empty()`, `usage()`, `turns()` and `turn(index)`.
- `ReplayTurn` also has `blocks()`, `completion()`, `notes()`, `sequence_range()` and `is_complete()`.
- `ReplayTurn::response(model, provider)` rebuilds the `UnifiedResponse` for code written against `on_response_complete`. You pass `model` and `provider`, because no record holds them. `finish_reason`, `prompt_cache_usage` and `usage` are always `None`. A turn with no end pair reports zero tokens and zero duration.
- The stream is the same on every run, but it does not keep the original timing. It never sleeps, never returns `Poll::Pending` and never yields `Err`. The recorded time is data in `ReplayCompletion::duration_ms`.

### Replayed blocks skip the plugins

Replay streams and responses go straight to you. No registered plugin's `on_stream_chunk` or `on_response_complete` runs on them, because no `CucaClient::generate_stream` call carries them. Call those hooks yourself if you need them.

### What a recording cannot give back

Replay never makes up data the recording does not hold.

- `MessageContentBlock::ImageBase64` is never recorded, so replay never emits one.
- `SessionEvent::ToolResult` carries `stdout`, `stderr` and `exit_code`, and `MessageContentBlock::ToolResult` has no field for them. Read the raw records with `SessionBackend::replay` to get them.

## Configuration

`ReplayConfig::new(max_records, max_turn_blocks, warn_fraction)` is the checked constructor. The fields are public, so `validate()` re-checks a struct you built by hand, and `SessionReplay::with_config` calls it.

| Key | Default | Valid values |
|---|---|---|
| `max_records` | `65536` (`ReplayConfig::DEFAULT_MAX_RECORDS`) | non-zero |
| `max_turn_blocks` | `4096` (`ReplayConfig::DEFAULT_MAX_TURN_BLOCKS`) | non-zero |
| `warn_fraction` | `Some(0.9)` | `None`, or finite and in `(0.0, 1.0]` |

- `max_records` is the most records one load may keep.
- `max_turn_blocks` is the most content blocks one turn may keep.
- `warn_fraction` is the share of `max_records` at which `ReplayUsage::near_cap` turns `true`. `None` keeps it `false`.

## Caps and at-cap policy

| Bound | At the cap | Gauge |
|---|---|---|
| `max_records`: records one load keeps | refuses with `PluginError::Validation`, never cuts | `usage().records` against `usage().max_records` |
| `max_turn_blocks`: blocks one turn keeps | refuses with `PluginError::Validation`, never cuts | `usage().blocks`, the total across all turns |

- A silently shortened trajectory would be a wrong fixture and a wrong eval, so replay refuses.
- `ReplayUsage` is computed once at load. It holds `records`, `max_records`, `blocks`, `turns` and `near_cap`. A loaded trajectory never grows, so the warning is a field you read and no message is injected.
- The limit before the read belongs to the backend. `SessionBackend::replay` returns the whole `Vec<SessionRecord>` first. `InMemoryBackend::DEFAULT_MAX_RECORDS` caps an in-memory session, and the disk caps a `FileBackend` session.
- `SessionReplay` holds one `Arc` and a small config, so it does not grow with traffic.

## Errors

Replay has no error type of its own. Every method returns `PluginError`, and each refusal is `PluginError::Validation { schema, message }`.

| `schema` | Raised by | When |
|---|---|---|
| `max_records` | `ReplayConfig::new`, `validate`, `with_config`, `load*` | the cap is zero, or the retained records exceed it |
| `max_turn_blocks` | `ReplayConfig::new`, `validate`, `with_config`, `load*` | the cap is zero, or one turn's blocks exceed it |
| `warn_fraction` | `ReplayConfig::new`, `validate`, `with_config` | the value is not finite or lies outside `(0.0, 1.0]` |
| `sequence` | `load*` | record sequences are not strictly increasing |
| `point_id` | `load_prefix`, `load_at_point` | no record sits at or below the sequence, or the id has no `:`, an empty session id, or a sequence that is not a `u64` |
| `turn_index` | `ReplayTrajectory::stream_turn` | the index names no turn, including every index on an empty trajectory |
| `session_id` | `ReplayTrajectory::into_stream` | the trajectory holds no turns |

- `SessionBackend::replay` errors pass through `load*` unchanged.
- A replay stream never yields `Err`, because every failure is raised at load time.

## Related pages

- [Session log](@/plugins/session-log.md) records the sessions replay reads.
- [Services](@/services/_index.md) lists every service and its plugin dependency.
- [Plugins and services](@/concepts/plugin-layering.md) explains the explicit-call tier.
- [Memory discipline](@/concepts/memory-discipline.md) explains the bounded-by-default rule.
