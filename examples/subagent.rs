//! Fan a task out to a child agent mid-stream, then collect its summary.
//!
//! The parent model calls `spawn_subagent`, and the plugin answers immediately
//! with the new child's id: the spawn is non-blocking, and the pending gauge
//! moves to one while the child is still running. A caller-supplied
//! `SubagentRunner` executes that child as its own live turn. A second parent
//! turn calls `collect_subagent`, which drains the pending registry back to
//! zero and hands the child's summary to the parent conversation.
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
//! cargo run --example subagent --features provider-llamacpp,plugin-subagent
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
//! Example: `CUCA_BASE_URL=http://127.0.0.1:8000/v1 CUCA_MODEL=<server-model-id> cargo run --example subagent --features provider-llamacpp,plugin-subagent`
//!
//! # Output
//!
//! From one run against `google/gemma-4-12b-qat` on llama.cpp:
//!
//! ```text
//! Pending gauge: 0 of 1024 children awaiting collection
//!
//! Turn 1: the parent delegates, and spawning does not block
//!   child id from the tool result: sub-0
//!   pending 1 of 1024, spawns so far 1
//!   thinking blocks: 121
//!
//! The child runs its own turn while the parent waits
//!   the child delivered its result
//!
//! Turn 2: the parent collects the summary through the same pipeline
//!   collected summary: "The largest moon of Saturn is Titan."
//!   pending 0 of 1024, spawns so far 1
//!   thinking blocks: 83
//!
//! Spawn log, 1 entr(ies)
//!   parent session "unset", worktree "none"
//! ```
//!
//! The child's summary and the block counts depend on the model. The ids and
//! the gauge do not: ids are `sub-<n>` from a per-plugin counter, so the first
//! child is always `sub-0`, and pending goes one up on spawn and one down on
//! collect.
//!
//! With no server on the base URL, the program prints one line naming the
//! address and exits successfully.
//!
//! # Why spawn and collect are split
//!
//! `spawn_subagent` registers a receiver and fires the child on a background
//! task, so the parent stream keeps producing blocks while the child works.
//! Only `collect` blocks, and it blocks the calling thread with a std
//! `recv()`, which parks that one thread and carries no runtime guard. On a
//! current-thread runtime the parent's thread is also the child's only
//! executor, so collecting before the child has delivered would park the
//! executor the child needs; the wait loop below is what keeps the two apart.

use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use cuca::plugin::CucaPlugin;
use cuca::types::{MessageContentBlock, ProviderEndpoint, ToolDefinition};
use cuca::{
    AgentResponseStream, CucaClient, SubagentPlugin, SubagentResult, SubagentRunner, SubagentSpec,
    UnifiedRequest,
};
use serde_json::json;
use tokio_stream::StreamExt;

/// Runs one child as a real llama.cpp turn and summarizes what it said.
///
/// The plugin owns the fan-out plumbing and nothing else: this seam is where a
/// deployment wires its own agent process or CLI. There is deliberately no
/// fallback summary, so a broken child reads as `exit_ok: false` with the
/// reason instead of a plausible-looking string.
struct LiveRunner {
    client: Arc<CucaClient>,
    model: String,
    /// Opened once the parent's stream has drained.
    ///
    /// `spawn_subagent` fires the child from inside `on_stream_chunk`, so the
    /// parent's stream is still open at that moment. A local server with one
    /// resident model serves one request at a time, and two turns racing it
    /// would say nothing about the plugin, so the child waits for the parent to
    /// finish. `Notify::notify_one` stores the permit, so the order of the two
    /// is irrelevant.
    gate: Arc<tokio::sync::Notify>,
    /// Flipped when the child's result has been delivered; the parent must not
    /// call `collect` before this, see the pump loop below.
    done: Arc<AtomicBool>,
}

impl SubagentRunner for LiveRunner {
    fn spawn(&self, spec: SubagentSpec) -> Pin<Box<dyn Future<Output = SubagentResult> + Send>> {
        let client = Arc::clone(&self.client);
        let model = self.model.clone();
        let done = Arc::clone(&self.done);
        let gate = Arc::clone(&self.gate);
        Box::pin(async move {
            gate.notified().await;
            let request = UnifiedRequest::new(&model)
                .set_max_tokens(320)
                .add_system_message("Answer in one short sentence.")
                .add_user_message(&spec.task);
            let mut summary = String::new();
            let mut failure: Option<String> = None;
            match client.generate_stream(request).await {
                Ok(mut stream) => {
                    while let Some(chunk) = stream.next().await {
                        match chunk {
                            Ok(MessageContentBlock::Text(text)) => summary.push_str(&text),
                            Ok(_) => {}
                            Err(error) => {
                                failure = Some(format!("stream error: {error}"));
                                break;
                            }
                        }
                    }
                }
                Err(error) => failure = Some(format!("generate_stream failed: {error}")),
            }
            if failure.is_none() && summary.trim().is_empty() {
                failure = Some("the child emitted no text".to_string());
            }
            let exit_ok = failure.is_none();
            done.store(true, Ordering::SeqCst);
            SubagentResult {
                subagent_id: spec.name,
                summary: failure.unwrap_or_else(|| summary.trim().to_string()),
                worktree_path: None,
                exit_ok,
            }
        })
    }
}

/// The two tools the plugin answers.
fn subagent_tools() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition {
            name: "spawn_subagent".to_string(),
            description: "Delegate a task to a child agent and get its id back.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "name": { "type": "string" },
                    "task": { "type": "string", "description": "what the child must do" },
                },
                "required": ["task"],
            }),
        },
        ToolDefinition {
            name: "collect_subagent".to_string(),
            description: "Fetch a spawned child's summary by its id.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": { "subagent_id": { "type": "string" } },
                "required": ["subagent_id"],
            }),
        },
    ]
}

/// Drain a turn into its text, the tool results the consumer received, and the
/// thinking-block count.
///
/// A reasoning model emits one `Thinking` block per token, so printing every
/// block buries the lines this demo is about. The count stays in the output
/// because it is the honest shape of a live turn.
async fn drain(mut stream: AgentResponseStream) -> (String, Vec<(String, String)>, usize) {
    let mut text = String::new();
    let mut results = Vec::new();
    let mut thinking = 0usize;
    while let Some(chunk) = stream.next().await {
        match chunk {
            Ok(MessageContentBlock::Text(chunk_text)) => text.push_str(&chunk_text),
            Ok(MessageContentBlock::Thinking { .. }) => thinking += 1,
            Ok(MessageContentBlock::ToolResult {
                tool_call_id,
                output,
            }) => results.push((tool_call_id, output)),
            Ok(_) => {}
            Err(error) => {
                println!("  the stream ended early: {error}");
                break;
            }
        }
    }
    (text, results, thinking)
}

/// A parent turn that offers both subagent tools.
fn tooled(model: &str, prompt: &str) -> UnifiedRequest {
    let mut request = UnifiedRequest::new(model)
        .set_max_tokens(256)
        .add_user_message(prompt);
    for tool in subagent_tools() {
        request = request.add_tool(tool);
    }
    request
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Base URL and model come from the environment so the example runs
    // against any OpenAI-compatible server; the defaults target a local
    // llama.cpp server (see the module docs for the override recipe).
    let base_url =
        std::env::var("CUCA_BASE_URL").unwrap_or_else(|_| "http://127.0.0.1:1234/v1".to_string());
    let model = std::env::var("CUCA_MODEL").unwrap_or_else(|_| "google/gemma-4-e4b".to_string());

    // The child's client carries no plugins: a child that could spawn children
    // is a fan-out with no bottom.
    let child_client = Arc::new(
        CucaClient::builder()
            .with_provider(ProviderEndpoint::LlamaCpp)
            .with_base_url(base_url.clone())
            .build()?,
    );
    let done = Arc::new(AtomicBool::new(false));
    let gate = Arc::new(tokio::sync::Notify::new());
    let subagent = Arc::new(SubagentPlugin::new(Arc::new(LiveRunner {
        client: child_client,
        model: model.clone(),
        gate: Arc::clone(&gate),
        done: Arc::clone(&done),
    })));
    let client = CucaClient::builder()
        .with_provider(ProviderEndpoint::LlamaCpp)
        .with_base_url(base_url.clone())
        .register_plugin(Arc::clone(&subagent) as Arc<dyn CucaPlugin>)
        .build()?;

    println!(
        "Pending gauge: {} of {} children awaiting collection",
        subagent.pending_len(),
        subagent.max_pending()
    );

    println!("\nTurn 1: the parent delegates, and spawning does not block");
    let stream = match client
        .generate_stream(tooled(
            &model,
            "Delegate this to a child agent with spawn_subagent: name the largest moon of Saturn.",
        ))
        .await
    {
        Ok(stream) => stream,
        Err(error) => {
            println!("\nNo server answered at {base_url}: {error}");
            println!("Start llama-server there, or set CUCA_BASE_URL, then run this again.");
            return Ok(());
        }
    };
    let (reply, results, thinking) = drain(stream).await;
    let Some((_, child_id)) = results.into_iter().next() else {
        println!("  the parent answered without delegating: {reply:?}");
        return Ok(());
    };
    println!("  child id from the tool result: {child_id}");
    println!(
        "  pending {} of {}, spawns so far {}",
        subagent.pending_len(),
        subagent.max_pending(),
        subagent.spawn_count()
    );
    println!("  thinking blocks: {thinking}");

    // `collect` blocks the calling thread, and on a current-thread runtime that
    // thread is the only executor the child has. Waiting for the delivery here
    // is what keeps the collect turn below from parking the child forever.
    println!("\nThe child runs its own turn while the parent waits");
    gate.notify_one();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(600);
    while !done.load(Ordering::SeqCst) {
        if tokio::time::Instant::now() >= deadline {
            println!("  the child did not finish within 600s");
            return Ok(());
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    println!("  the child delivered its result");

    println!("\nTurn 2: the parent collects the summary through the same pipeline");
    let (reply, results, thinking) = drain(
        client
            .generate_stream(tooled(
                &model,
                &format!(
                    "Call collect_subagent with subagent_id {child_id} and quote its summary."
                ),
            ))
            .await?,
    )
    .await;
    match results.first() {
        Some((_, summary)) => println!("  collected summary: {summary:?}"),
        None => println!("  the parent answered without collecting: {reply:?}"),
    }
    println!(
        "  pending {} of {}, spawns so far {}",
        subagent.pending_len(),
        subagent.max_pending(),
        subagent.spawn_count()
    );
    println!("  thinking blocks: {thinking}");

    println!("\nSpawn log, {} entr(ies)", subagent.spawns().len());
    for (session_id, worktree) in subagent.spawns() {
        println!(
            "  parent session {:?}, worktree {:?}",
            session_id.unwrap_or_else(|| "unset".to_string()),
            worktree.unwrap_or_else(|| "none".to_string())
        );
    }

    Ok(())
}
