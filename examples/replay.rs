//! Record one live turn, then re-materialize it with no provider in sight.
//!
//! Stage 1 registers `SessionLogPlugin` over a temp-directory `FileBackend` and
//! streams one real llama.cpp turn through it, so the trajectory lands on disk
//! as framed records. Stage 2 **drops the client**, opens a brand-new
//! `FileBackend` over the same directory, and replays the session through
//! `SessionReplay`: the block sequence comes back byte for byte with no
//! network call and no provider dispatch, because by then no `CucaClient`
//! exists at all. Stage 3 rebuilds the aggregated `UnifiedResponse`, and
//! stage 4 replays a fork-point prefix of the same trajectory.
//!
//! # Prerequisites
//!
//! - A checkout of this repository (the example builds from this crate).
//! - A running [llama.cpp](https://github.com/ggml-org/llama.cpp) server
//!   (`llama-server`) on port 1234 with the demo model loaded.
//!
//! # Run
//!
//! ```sh
//! cargo run --example replay --features provider-llamacpp,service-replay
//! ```
//!
//! # Configuration
//!
//! Both values default to a local llama.cpp server; override them to target
//! any OpenAI-compatible server:
//!
//! - `CUCA_BASE_URL`: server base URL, defaults to `http://127.0.0.1:1234/v1`.
//! - `CUCA_MODEL`: upstream model id, defaults to `google/gemma-4-e4b`.
//!
//! Example: `CUCA_BASE_URL=http://127.0.0.1:8000/v1 CUCA_MODEL=<server-model-id> cargo run --example replay --features provider-llamacpp,service-replay`
//!
//! # Output
//!
//! One run against `google/gemma-4-12b-qat` on llama.cpp:
//!
//! ```text
//! Live turn (the only provider dispatch in this program)
//!   58 blocks in 34.17 s: 57 Thinking, 1 Text
//!   last block: Text("ok")
//!
//! Client dropped. Replaying from /tmp/cuca-replay-example-3199010
//!   session `demo`: 1 turn(s), 58 block(s), 62 records of 65536 (near cap: false)
//!   turn 0 covers sequences 0..=61, complete: true
//!     system prompt: "You are concise."
//!     message: User with 1 block(s)
//!     accounting: 34166 ms, 0 prompt tokens, 58 completion tokens
//!   replayed 58 blocks: 57 Thinking, 1 Text
//!   last block: Text("ok")
//!   identical to the live stream, block for block: true
//!
//! Rebuilt UnifiedResponse: model=google/gemma-4-12b-qat provider=LlamaCpp
//!   duration=34.166 s prompt=0 completion=58 content=58 blocks
//!
//! Fork-point replay at `demo:2`
//!   3 of 62 records retained, 1 turn(s), 1 block(s)
//! ```
//!
//! The `identical to the live stream` line is the demo: 58 blocks came back in
//! the same order with no `CucaClient` alive to dispatch them.
//!
//! Block counts and record counts depend on the model: a reasoning-heavy reply
//! adds `Thinking` blocks, and each recorded block is one more record. The
//! accounting line is exactly what the provider reported and what
//! `on_response_complete` therefore recorded, so a server that reports no token
//! usage replays as zeros rather than as an invented estimate.
//!
//! Two loads of the same trajectory always produce the same block sequence:
//! replay is order-deterministic. It is not time-deterministic, and nothing
//! here waits on the recorded latency; `ReplayCompletion::duration_ms` is data.
//!
//! With no server on the base URL, the program prints one line naming the
//! address and exits successfully.
//!
//! # Why a service and not a plugin hook?
//!
//! `SessionReplay` is a service, not a `CucaPlugin`, so it is constructed and
//! called directly rather than registered on the builder. Replay drives
//! sessions instead of observing one: there is no live request to mutate, no
//! arriving chunk to annotate, and no hook signature that can return a stream.
//! Stage 2 below is the reason the rule exists: a "plugin" that only ever runs
//! when the caller explicitly asks would be a permanent no-op in the pipeline.
//!
//! Two documented fidelity gaps also show up here: `ImageBase64` blocks are
//! never recorded, so replay can never emit one, and a `ToolResult`'s
//! `stdout`/`stderr`/`exit_code` have no home in the unified block, so they are
//! absent from the replayed stream.

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
