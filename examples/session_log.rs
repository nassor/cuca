//! Record two live turns into an append-only trajectory, then fork it.
//!
//! Two `SessionLogPlugin` instances are registered on one client, one over the
//! capped `InMemoryBackend` and one over the append-only `FileBackend`, so both
//! record the same two real turns. The demo prints the record kinds each turn
//! appended, the in-memory usage gauge against its cap, and the bytes the file
//! backend wrote. It then forks the trajectory at a historical point and shows
//! the branch's prefix plus the `Fork` audit record the original gained.
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
//! cargo run --example session_log --features provider-llamacpp,plugin-session-log
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
//! The file backend writes into a per-process directory under the OS temp
//! directory and removes it before exiting.
//!
//! Example: `CUCA_BASE_URL=http://127.0.0.1:8000/v1 CUCA_MODEL=<server-model-id> cargo run --example session_log --features provider-llamacpp,plugin-session-log`
//!
//! # Output
//!
//! From one run against `google/gemma-4-12b-qat` on llama.cpp:
//!
//! ```text
//! Session "demo", two backends recording the same turns
//!   memory: InMemoryBackend, cap 65536 records
//!   file:   /tmp/cuca-session-log-3178600/demo.cslog
//!
//! Turn 1: "The vault is in Lisbon. Reply with: noted"
//!   reply: noted
//!   appended 78 records: SystemPrompt 1, Message 1, Reasoning 72, Output 2, Latency 1, TokenUsage 1
//!   gauge: 78/65536 records in memory, 1647 bytes on disk
//!
//! Turn 2: "Where is the vault? Answer in one word."
//!   reply: Lisbon
//!   appended 135 records: SystemPrompt 1, Message 2, Reasoning 128, Output 2, Latency 1, TokenUsage 1
//!   gauge: 213/65536 records in memory, 4547 bytes on disk
//!
//! Fork at demo:1, the first Message record
//!   new session: demo:fork:demo:1:0
//!   branch: 2 records: SystemPrompt 1, Message 1
//!   original tail: Fork { from_point: "demo:1", to_session: "demo:fork:demo:1:0" }
//!   original: 214 records, one more than before the fork
//!
//! The file backend replays the same trajectory: 213 records
//! ```
//!
//! One `Reasoning` record per streamed `Thinking` block is why the counts run
//! into the hundreds: a reasoning model emits one such block per token, and
//! every inbound block is recorded verbatim. Turn 2 appends a second
//! `SystemPrompt` because only user and assistant messages are deduplicated by
//! position; a system message is recorded on every request it appears in. Its
//! two `Message` records are the assistant reply from turn 1 and the new user
//! question.
//!
//! The counts, the byte totals, and the directory's process id depend on the
//! run. The record kinds, the branch prefix, and the audit record do not.
//!
//! With no server on the base URL, the program prints one line naming the
//! address and exits successfully.
//!
//! # Why the fork adds a record to the original
//!
//! A trajectory is append-only: nothing is rewritten or removed, so branching
//! cannot be recorded by editing history. The branch gets the prefix of the
//! original up to and including the fork point, relabelled with the new session
//! id, and the original gains one `Fork` record naming both ends. That is also
//! why the in-memory backend refuses to append at its cap instead of evicting:
//! dropping a record would silently corrupt both replay and fork.

use std::collections::HashMap;
use std::sync::Arc;

use cuca::plugin::{CucaPlugin, SessionStorePlugin};
use cuca::session::{SessionEvent, SessionRecord};
use cuca::types::{MessageContentBlock, ProviderEndpoint, UnifiedMessage};
use cuca::{
    AgentResponseStream, CucaClient, FileBackend, InMemoryBackend, SessionBackend,
    SessionLogPlugin, UnifiedRequest,
};
use tokio_stream::StreamExt;

/// Per-turn completion cap. A reasoning model spends most of it on `Thinking`
/// blocks, so a smaller budget returns an empty reply.
const MAX_TOKENS: u32 = 512;

/// The session the hooks write to, in both backends.
const SESSION: &str = "demo";

/// Two turns: the first states a fact, the second asks for it back.
const PROMPTS: [&str; 2] = [
    "The vault is in Lisbon. Reply with: noted",
    "Where is the vault? Answer in one word.",
];

/// The record kind, as the store's own event variants name them.
fn event_kind(event: &SessionEvent) -> &'static str {
    match event {
        SessionEvent::SystemPrompt { .. } => "SystemPrompt",
        SessionEvent::Message { .. } => "Message",
        SessionEvent::Reasoning { .. } => "Reasoning",
        SessionEvent::Output { .. } => "Output",
        SessionEvent::ToolCall { .. } => "ToolCall",
        SessionEvent::ToolResult { .. } => "ToolResult",
        SessionEvent::ModelSwap { .. } => "ModelSwap",
        SessionEvent::Latency { .. } => "Latency",
        SessionEvent::TokenUsage { .. } => "TokenUsage",
        SessionEvent::Fork { .. } => "Fork",
    }
}

/// Per-kind tally of `records`, in append order of first appearance.
///
/// A tally, not a listing: one `Reasoning` record per reasoning token would
/// otherwise be the entire output.
fn tally(records: &[SessionRecord]) -> String {
    let mut order: Vec<&'static str> = Vec::new();
    let mut counts: HashMap<&'static str, usize> = HashMap::new();
    for record in records {
        let kind = event_kind(&record.event);
        let count = counts.entry(kind).or_insert(0);
        if *count == 0 {
            order.push(kind);
        }
        *count += 1;
    }
    order
        .into_iter()
        .map(|kind| format!("{kind} {}", counts[kind]))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Drain a turn into its text, dropping the `Thinking` blocks: the store
/// already records every one of them as a `Reasoning` record.
async fn drain(mut stream: AgentResponseStream) -> String {
    let mut text = String::new();
    while let Some(chunk) = stream.next().await {
        match chunk {
            Ok(MessageContentBlock::Text(chunk_text)) => text.push_str(&chunk_text),
            Ok(_) => {}
            Err(error) => {
                println!("  the stream ended early: {error}");
                break;
            }
        }
    }
    text
}

/// Bytes the file backend has written for `SESSION`.
fn file_bytes(dir: &std::path::Path) -> u64 {
    std::fs::metadata(dir.join(format!("{SESSION}.cslog")))
        .map(|meta| meta.len())
        .unwrap_or(0)
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Base URL and model come from the environment so the example runs
    // against any OpenAI-compatible server; the defaults target a local
    // llama.cpp server (see the module docs for the override recipe).
    let base_url =
        std::env::var("CUCA_BASE_URL").unwrap_or_else(|_| "http://127.0.0.1:1234/v1".to_string());
    let model = std::env::var("CUCA_MODEL").unwrap_or_else(|_| "google/gemma-4-e4b".to_string());

    // One directory per process: the file backend opens with `append(true)` and
    // never truncates, so a shared directory would keep the previous run's
    // frames and the counts below would grow every time.
    let dir = std::env::temp_dir().join(format!("cuca-session-log-{}", std::process::id()));
    let memory_backend = Arc::new(InMemoryBackend::new());
    let memory_log = Arc::new(
        SessionLogPlugin::new(Arc::clone(&memory_backend) as Arc<dyn SessionBackend>)
            .with_session_id(SESSION),
    );
    let file_log = Arc::new(
        SessionLogPlugin::new(Arc::new(FileBackend::new(dir.clone())?)).with_session_id(SESSION),
    );
    let client = CucaClient::builder()
        .with_provider(ProviderEndpoint::LlamaCpp)
        .with_base_url(base_url.clone())
        .register_plugin(Arc::clone(&memory_log) as Arc<dyn CucaPlugin>)
        .register_plugin(Arc::clone(&file_log) as Arc<dyn CucaPlugin>)
        .build()?;

    println!("Session {SESSION:?}, two backends recording the same turns");
    println!(
        "  memory: InMemoryBackend, cap {} records",
        InMemoryBackend::DEFAULT_MAX_RECORDS
    );
    println!(
        "  file:   {}",
        dir.join(format!("{SESSION}.cslog")).display()
    );

    let mut messages = vec![UnifiedMessage::system("You are concise.")];
    let mut recorded = 0usize;
    for (index, prompt) in PROMPTS.iter().enumerate() {
        messages.push(UnifiedMessage::user(*prompt));
        println!("\nTurn {}: {prompt:?}", index + 1);
        let mut request = UnifiedRequest::new(&model).set_max_tokens(MAX_TOKENS);
        request.messages = messages.clone();
        let stream = match client.generate_stream(request).await {
            Ok(stream) => stream,
            Err(error) => {
                println!("\nNo server answered at {base_url}: {error}");
                println!("Start llama-server there, or set CUCA_BASE_URL, then run this again.");
                std::fs::remove_dir_all(&dir).ok();
                return Ok(());
            }
        };
        let reply = drain(stream).await;
        println!("  reply: {}", reply.trim());
        messages.push(UnifiedMessage::assistant(reply.trim()));

        let records = memory_log.replay_session(SESSION)?;
        let appended = &records[recorded..];
        println!("  appended {} records: {}", appended.len(), tally(appended));
        recorded = records.len();
        println!(
            "  gauge: {}/{} records in memory, {} bytes on disk",
            memory_backend.len(),
            InMemoryBackend::DEFAULT_MAX_RECORDS,
            file_bytes(&dir)
        );
    }

    // Fork from a historical point rather than the tail: the branch is the
    // prefix up to and including that record, which is the whole point of
    // forking an append-only log.
    let records = memory_log.replay_session(SESSION)?;
    let point = records
        .iter()
        .find(|record| matches!(record.event, SessionEvent::Message { .. }))
        .map(SessionRecord::point_id)
        .expect("the first turn recorded a Message");
    println!("\nFork at {point}, the first Message record");
    let branch = memory_log.fork_session(SESSION, &point)?;
    println!("  new session: {branch}");
    let branch_records = memory_log.replay_session(&branch)?;
    println!(
        "  branch: {} records: {}",
        branch_records.len(),
        tally(&branch_records)
    );
    let original = memory_log.replay_session(SESSION)?;
    if let Some(SessionEvent::Fork {
        from_point,
        to_session,
    }) = original.last().map(|record| &record.event)
    {
        println!(
            "  original tail: Fork {{ from_point: {from_point:?}, to_session: {to_session:?} }}"
        );
    }
    println!(
        "  original: {} records, one more than before the fork",
        original.len()
    );

    println!(
        "\nThe file backend replays the same trajectory: {} records",
        file_log.replay_session(SESSION)?.len()
    );
    std::fs::remove_dir_all(&dir).ok();
    Ok(())
}
