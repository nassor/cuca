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

`SessionReplay` reads a recorded session back through the session log's `SessionBackend` seam and re-materializes it as the same `AgentResponseStream` a live turn produces. `SessionReplay::new(backend)` binds an `Arc<dyn SessionBackend>` with the default `ReplayConfig`, and `with_config(backend, config)` sets the caps instead. That backend is `SessionLogPlugin::backend()` for an already-registered log, or `FileBackend::new(dir)` for a directory of `.cslog` files. Reach for it when a recorded turn has to drive a regression fixture, an offline eval, or a fork-point comparison.

```rust,name=Record one live turn then replay it with no provider
use std::sync::Arc;

use cuca::plugin::CucaPlugin;
use cuca::types::{MessageContentBlock, ProviderEndpoint};
use cuca::{CucaClient, FileBackend, SessionLogPlugin, SessionReplay, UnifiedRequest};
use tokio_stream::StreamExt;

/// The session the recorded turn is written under.
const SESSION: &str = "demo";

/// Removes the recording directory on drop, so the demo leaves no files behind.
struct TempDir(std::path::PathBuf);

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// A short rendering of one block, for printing.
///
/// The `ImageBase64` arm is unreachable on a replay path: the session log maps
/// image blocks to no event, so no trajectory can ever carry one. It is spelled
/// out anyway because the same helper prints the live stream, which can.
fn describe(block: &MessageContentBlock) -> String {
    match block {
        MessageContentBlock::Text(text) => format!("Text({text:?})"),
        MessageContentBlock::Thinking { reasoning, .. } => {
            format!("Thinking({} chars)", reasoning.chars().count())
        }
        MessageContentBlock::ToolCall { id, name, .. } => format!("ToolCall({name}, id={id})"),
        MessageContentBlock::ToolResult { tool_call_id, .. } => {
            format!("ToolResult(for {tool_call_id})")
        }
        MessageContentBlock::ImageBase64 { media_type, data } => {
            format!("ImageBase64({media_type}, {} base64 chars)", data.len())
        }
    }
}

/// Per-kind counts of a block sequence, in first-seen order.
///
/// A reasoning model emits one `Thinking` block per token, so printing every
/// block twice (once live, once replayed) would bury the comparison the demo
/// is about.
fn kinds(blocks: &[String]) -> String {
    let mut counts: Vec<(&str, usize)> = Vec::new();
    for block in blocks {
        let kind = block.split('(').next().unwrap_or(block);
        match counts.iter_mut().find(|(seen, _)| *seen == kind) {
            Some((_, count)) => *count += 1,
            None => counts.push((kind, 1)),
        }
    }
    counts
        .iter()
        .map(|(kind, count)| format!("{count} {kind}"))
        .collect::<Vec<_>>()
        .join(", ")
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Base URL and model come from the environment so the example runs
    // against any OpenAI-compatible server; the defaults target a local
    // llama.cpp server (see the module docs for the override recipe).
    let base_url =
        std::env::var("CUCA_BASE_URL").unwrap_or_else(|_| "http://127.0.0.1:1234/v1".to_string());
    let model = std::env::var("CUCA_MODEL").unwrap_or_else(|_| "google/gemma-4-e4b".to_string());

    // The trajectory goes to disk, so stage 2 can reopen it from scratch.
    let dir =
        TempDir(std::env::temp_dir().join(format!("cuca-replay-example-{}", std::process::id())));
    let plugin = Arc::new(
        SessionLogPlugin::new(Arc::new(FileBackend::new(&dir.0)?)).with_session_id(SESSION),
    );

    // Stage 1: one recorded turn. This is the only provider dispatch here.
    println!("Live turn (the only provider dispatch in this program)");
    let client = CucaClient::builder()
        .with_provider(ProviderEndpoint::LlamaCpp)
        .with_base_url(base_url.clone())
        .register_plugin(Arc::clone(&plugin) as Arc<dyn CucaPlugin>)
        .build()?;
    let request = UnifiedRequest::new(&model)
        .add_system_message("You are concise.")
        .add_user_message("Reply with the single word: ok")
        .set_max_tokens(128);
    let started = std::time::Instant::now();
    let mut stream = match client.generate_stream(request).await {
        Ok(stream) => stream,
        Err(error) => {
            println!("\nNo server answered at {base_url}: {error}");
            println!("Start llama-server there, or set CUCA_BASE_URL, then run this again.");
            return Ok(());
        }
    };
    let mut live: Vec<String> = Vec::new();
    while let Some(chunk) = stream.next().await {
        match chunk {
            Ok(block) => live.push(describe(&block)),
            Err(error) => {
                println!("  the stream ended early: {error}");
                break;
            }
        }
    }
    println!(
        "  {} blocks in {:.2} s: {}",
        live.len(),
        started.elapsed().as_secs_f64(),
        kinds(&live)
    );
    println!(
        "  last block: {}",
        live.last().map_or("none", String::as_str)
    );

    // Stage 2: no client, no provider, no cache. The dispatch path is *gone*
    // before a single block is replayed; a fresh backend over the same
    // directory proves the records came off disk. A caller that still holds the
    // store can use `SessionReplay::new(Arc::clone(plugin.backend()))` instead.
    drop(stream);
    drop(client);
    drop(plugin);
    println!("\nClient dropped. Replaying from {}", dir.0.display());
    let replay = SessionReplay::new(Arc::new(FileBackend::new(&dir.0)?));
    let trajectory = replay.load(SESSION)?;
    let usage = trajectory.usage();
    println!(
        "  session `{}`: {} turn(s), {} block(s), {} records of {} (near cap: {})",
        trajectory.session_id(),
        trajectory.len(),
        usage.blocks,
        usage.records,
        usage.max_records,
        usage.near_cap
    );

    let turn = match trajectory.turn(0) {
        Some(turn) => turn,
        None => {
            println!("  the trajectory recorded no turn; nothing to replay");
            return Ok(());
        }
    };
    let (first, last) = turn.sequence_range();
    println!(
        "  turn 0 covers sequences {first}..={last}, complete: {}",
        turn.is_complete()
    );
    for prompt in turn.system_prompts() {
        println!("    system prompt: {prompt:?}");
    }
    for message in turn.messages() {
        println!(
            "    message: {:?} with {} block(s)",
            message.role,
            message.content.len()
        );
    }
    match turn.completion() {
        Some(completion) => println!(
            "    accounting: {} ms, {} prompt tokens, {} completion tokens",
            completion.duration_ms, completion.prompt_tokens, completion.completion_tokens
        ),
        None => println!("    accounting: none (an interrupted generation)"),
    }
    for note in turn.notes() {
        println!("    note: {note:?}");
    }

    // `stream_turn` clones the turn's blocks, so the trajectory stays
    // replayable; `into_stream` moves them out instead.
    let mut replayed: Vec<String> = Vec::new();
    let mut stream = trajectory.stream_turn(0)?;
    while let Some(item) = stream.next().await {
        // A replayed stream never yields Err: every failure was raised by
        // `load` above, so a materialized stream runs to completion.
        match item {
            Ok(block) => replayed.push(describe(&block)),
            Err(error) => println!("  unreachable: {error}"),
        }
    }
    println!("  replayed {} blocks: {}", replayed.len(), kinds(&replayed));
    println!(
        "  last block: {}",
        replayed.last().map_or("none", String::as_str)
    );
    // The claim the demo exists to check: same blocks, same order, no
    // provider. `zip` would hide a length difference, so the lengths are
    // compared too.
    match replayed
        .iter()
        .zip(&live)
        .position(|(after, before)| after != before)
    {
        None if replayed.len() == live.len() => {
            println!("  identical to the live stream, block for block: true")
        }
        None => println!(
            "  block counts differ: {} live, {} replayed",
            live.len(),
            replayed.len()
        ),
        Some(index) => println!(
            "  first divergence at block {index}: live {}, replayed {}",
            live[index], replayed[index]
        ),
    }

    // Stage 3: the aggregated shape, for consumers written against
    // `on_response_complete`'s argument. The model and provider are explicit:
    // no SessionEvent records them.
    let response = trajectory
        .turn(0)
        .expect("turn 0 was already loaded above")
        .response(&model, ProviderEndpoint::LlamaCpp);
    println!(
        "\nRebuilt UnifiedResponse: model={} provider={:?}",
        response.model, response.provider
    );
    println!(
        "  duration={:.3} s prompt={} completion={} content={} blocks",
        response.duration_secs,
        response.prompt_tokens,
        response.completion_tokens,
        response.content.len()
    );

    // Stage 4: fork-point replay. The same trajectory, cut at a recorded
    // position addressed exactly as `fork_session` addresses it.
    let point_id = format!("{SESSION}:2");
    println!("\nFork-point replay at `{point_id}`");
    let prefix = replay.load_at_point(&point_id)?;
    println!(
        "  {} of {} records retained, {} turn(s), {} block(s)",
        prefix.usage().records,
        usage.records,
        prefix.len(),
        prefix.usage().blocks
    );

    Ok(())
}
```

```text,name=Expected output
Live turn (the only provider dispatch in this program)
  58 blocks in 34.17 s: 57 Thinking, 1 Text
  last block: Text("ok")

Client dropped. Replaying from /tmp/cuca-replay-example-3199010
  session `demo`: 1 turn(s), 58 block(s), 62 records of 65536 (near cap: false)
  turn 0 covers sequences 0..=61, complete: true
    system prompt: "You are concise."
    message: User with 1 block(s)
    accounting: 34166 ms, 0 prompt tokens, 58 completion tokens
  replayed 58 blocks: 57 Thinking, 1 Text
  last block: Text("ok")
  identical to the live stream, block for block: true

Rebuilt UnifiedResponse: model=google/gemma-4-12b-qat provider=LlamaCpp
  duration=34.166 s prompt=0 completion=58 content=58 blocks

Fork-point replay at `demo:2`
  3 of 62 records retained, 1 turn(s), 1 block(s)
```

`google/gemma-4-12b-qat` produced that run: the block and record counts are the model's, since one `Thinking` block per reasoning token becomes one more record. The `identical to the live stream` line is not: replay is order-deterministic for any model.

## Try it

`examples/replay.rs` is the program above. It records one live turn through a `SessionLogPlugin` over a temp-directory `FileBackend`, drops the client, opens a fresh `FileBackend` over the same directory, and replays the trajectory with no provider in sight; it then rebuilds the aggregated `UnifiedResponse` and replays a fork-point prefix. It needs a `llama-server` on port 1234 with the demo model loaded; `CUCA_BASE_URL` and `CUCA_MODEL` retarget it at any OpenAI-compatible server.

```bash,name=Runs the same on all three platforms
cargo run --example replay --features "provider-llamacpp service-replay"
```

## Entry types

`SessionReplay`, `ReplayConfig`, `ReplayTrajectory`, `ReplayTurn`, `ReplayUsage`, `ReplayCompletion`, `ReplayNote`.

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
