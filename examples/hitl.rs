//! Gate a destructive tool call behind an approval policy, then read the audit
//! trail.
//!
//! One `ApprovalChannel` rules on both stages: it approves a write aimed at the
//! `scratch` notebook and denies the same tool aimed at `audit`. Stage 1 is
//! what approval means, the block streams through untouched and the
//! application is what executes it. Stage 2 is what denial means, the plugin
//! replaces the call with a denial `ToolResult`, nothing runs, and the model
//! still gets an answer it can act on. The audit log at the end carries one
//! entry per gated call, with the approver's identity.
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
//! cargo run --example hitl --features provider-llamacpp,plugin-hitl
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
//! Example: `CUCA_BASE_URL=http://127.0.0.1:8000/v1 CUCA_MODEL=<server-model-id> cargo run --example hitl --features provider-llamacpp,plugin-hitl`
//!
//! # Output
//!
//! From one run against `google/gemma-4-12b-qat` on llama.cpp:
//!
//! ```text
//! Stage 1: a call the policy approves
//!   gate: file_write write_note {"notebook":"scratch","text":"ship it"} -> Approved
//!   delivered: [ToolCall { id: "AIG5T6wty1u1kN5zRNPtMxvOjy8hTYpo", name: "write_note", arguments: Object {"notebook": String("scratch"), "text": String("ship it")} }]
//!   app executed: appended note 1 to notebook "scratch"
//!   reply: OK. I've saved the note "ship it" in your scratch notebook.
//!   thinking blocks: 73 then 0
//!
//! Stage 2: the same tool, a notebook the policy refuses
//!   gate: file_write write_note {"notebook":"audit","text":"ship it"} -> Denied
//!   delivered: [ToolResult { tool_call_id: "POLkDMwHOUZcg35grlKbmhCSxvOQ3SoT", output: "denied by approver" }]
//!   reply: I'm sorry, but I was unable to save the note to the audit notebook because it was denied by the approver.
//!   thinking blocks: 55 then 250
//!
//! Audit log, 2 of 65536 entries
//!   approved file_write   approver ops-oncall
//!   denied   file_write   approver ops-oncall
//!   notebook "scratch" holds ["ship it"]
//! ```
//!
//! The replies, the call ids and the block counts depend on the model. The two
//! rulings do not: `write_note` matches the file-write keyword group, so both
//! calls are `Risk::High`, and the policy reads the notebook name out of
//! `ApprovalRequest::detail`.
//!
//! With no server on the base URL, the program prints one line naming the
//! address and exits successfully.
//!
//! # Why the approval channel blocks
//!
//! `ApprovalChannel::request_approval` parks the calling thread, which is the
//! whole mechanism: the pipeline pauses at a gated call until an approver
//! decides, so the tool cannot run while the answer is still in flight. The
//! same design makes the failure mode a policy question rather than a race. An
//! implementation that cannot reach an approver must return `Denied`, so a lost
//! round trip refuses the call instead of letting it through.

use std::sync::{Arc, Mutex};

use cuca::plugin::CucaPlugin;
use cuca::types::{
    MessageContentBlock, MessageRole, ProviderEndpoint, ToolDefinition, UnifiedMessage,
};
use cuca::{
    AgentResponseStream, ApprovalChannel, ApprovalDecision, ApprovalRequest, CucaClient, CucaError,
    HitlPlugin, PluginError, UnifiedRequest,
};
use serde_json::{Value, json};
use tokio_stream::StreamExt;

/// The one notebook this policy is willing to let the model write into.
const OPEN_NOTEBOOK: &str = "scratch";

/// One approval policy, two outcomes.
///
/// `ApprovalRequest::detail` is the tool name plus its JSON arguments, which is
/// everything an approver needs to rule on the call, so the policy needs no
/// access to the request or the stream.
struct NotebookPolicy;

impl ApprovalChannel for NotebookPolicy {
    fn request_approval(&self, req: &ApprovalRequest) -> ApprovalDecision {
        let decision = if req.detail.contains(OPEN_NOTEBOOK) {
            ApprovalDecision::Approved
        } else {
            ApprovalDecision::Denied
        };
        println!("  gate: {} {} -> {decision:?}", req.action, req.detail);
        decision
    }

    fn approver_id(&self) -> Option<String> {
        Some("ops-oncall".to_string())
    }
}

/// Keeps the tool call the model issued.
///
/// `on_stream_chunk` hooks run in registration order over one shared block, so
/// a recorder registered *before* `HitlPlugin` sees the call as the model wrote
/// it, which a denied call otherwise erases.
#[derive(Default)]
struct CallRecorder {
    calls: Mutex<Vec<MessageContentBlock>>,
}

impl CallRecorder {
    fn take(&self) -> Vec<MessageContentBlock> {
        std::mem::take(&mut self.calls.lock().unwrap_or_else(|p| p.into_inner()))
    }
}

impl CucaPlugin for CallRecorder {
    fn name(&self) -> &'static str {
        "call-recorder"
    }

    fn on_stream_chunk(&self, chunk: &mut MessageContentBlock) -> Result<(), PluginError> {
        if matches!(chunk, MessageContentBlock::ToolCall { .. }) {
            self.calls
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .push(chunk.clone());
        }
        Ok(())
    }
}

/// The tool the model is offered. `classify_tool_call` reads the name only, and
/// `write_note` matches the file-write keyword group, so every call is gated.
fn write_note_tool() -> ToolDefinition {
    ToolDefinition {
        name: "write_note".to_string(),
        description: "Append one note to a named notebook.".to_string(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "notebook": { "type": "string", "description": "notebook name" },
                "text": { "type": "string", "description": "the note to append" },
            },
            "required": ["notebook", "text"],
        }),
    }
}

/// Drain a turn into its text, the tool blocks the consumer received, and the
/// thinking-block count.
///
/// A reasoning model emits one `Thinking` block per token, so printing every
/// block buries the lines this demo is about. The count stays in the output
/// because it is the honest shape of a live turn.
async fn drain(mut stream: AgentResponseStream) -> (String, Vec<MessageContentBlock>, usize) {
    let mut text = String::new();
    let mut tools = Vec::new();
    let mut thinking = 0usize;
    while let Some(chunk) = stream.next().await {
        match chunk {
            Ok(MessageContentBlock::Text(chunk_text)) => text.push_str(&chunk_text),
            Ok(MessageContentBlock::Thinking { .. }) => thinking += 1,
            Ok(block) => tools.push(block),
            Err(error) => {
                println!("  the stream ended early: {error}");
                break;
            }
        }
    }
    (text, tools, thinking)
}

/// The follow-up turn: the same prompt, the call the model made, and the result
/// the gate produced or permitted.
fn follow_up(
    model: &str,
    prompt: &str,
    call: MessageContentBlock,
    output: String,
) -> UnifiedRequest {
    let call_id = match &call {
        MessageContentBlock::ToolCall { id, .. } => id.clone(),
        _ => String::new(),
    };
    UnifiedRequest::new(model)
        // A reasoning model spends the token budget on thinking first, so a
        // tight cap can end the turn before any text is emitted.
        .set_max_tokens(320)
        .add_user_message(prompt)
        .add_message(UnifiedMessage {
            role: MessageRole::Assistant,
            content: vec![call],
            name: None,
            tool_call_id: None,
        })
        .add_message(UnifiedMessage {
            role: MessageRole::Tool,
            content: vec![MessageContentBlock::ToolResult {
                tool_call_id: call_id.clone(),
                output,
            }],
            name: None,
            tool_call_id: Some(call_id),
        })
}

/// The application-side executor: what an approved call is allowed to do.
fn append_note(notebook: &Mutex<Vec<String>>, arguments: &Value) -> String {
    let name = arguments
        .get("notebook")
        .and_then(Value::as_str)
        .unwrap_or("");
    let text = arguments.get("text").and_then(Value::as_str).unwrap_or("");
    let mut notes = notebook.lock().unwrap_or_else(|p| p.into_inner());
    notes.push(text.to_string());
    format!("appended note {} to notebook {name:?}", notes.len())
}

/// One gated turn: the model is given the tool, the gate rules on the call it
/// makes, and the returned blocks show which way the ruling went.
async fn gated_turn(
    client: &CucaClient,
    model: &str,
    prompt: &str,
) -> Result<(String, Vec<MessageContentBlock>, usize), CucaError> {
    let request = UnifiedRequest::new(model)
        .set_max_tokens(160)
        .add_user_message(prompt)
        .add_tool(write_note_tool());
    Ok(drain(client.generate_stream(request).await?).await)
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Base URL and model come from the environment so the example runs
    // against any OpenAI-compatible server; the defaults target a local
    // llama.cpp server (see the module docs for the override recipe).
    let base_url =
        std::env::var("CUCA_BASE_URL").unwrap_or_else(|_| "http://127.0.0.1:1234/v1".to_string());
    let model = std::env::var("CUCA_MODEL").unwrap_or_else(|_| "google/gemma-4-e4b".to_string());

    let recorder = Arc::new(CallRecorder::default());
    let hitl = Arc::new(HitlPlugin::new(Arc::new(NotebookPolicy)));
    let client = CucaClient::builder()
        .with_provider(ProviderEndpoint::LlamaCpp)
        .with_base_url(base_url.clone())
        .register_plugin(Arc::clone(&recorder) as Arc<dyn CucaPlugin>)
        .register_plugin(Arc::clone(&hitl) as Arc<dyn CucaPlugin>)
        .build()?;
    let notebook: Mutex<Vec<String>> = Mutex::new(Vec::new());

    let approved_prompt = format!("Save the note \"ship it\" in the {OPEN_NOTEBOOK} notebook.");
    println!("Stage 1: a call the policy approves");
    let (reply, blocks, thinking) = match gated_turn(&client, &model, &approved_prompt).await {
        Ok(turn) => turn,
        Err(error) => {
            println!("\nNo server answered at {base_url}: {error}");
            println!("Start llama-server there, or set CUCA_BASE_URL, then run this again.");
            return Ok(());
        }
    };
    let call = recorder.take().into_iter().next();
    let Some(call @ MessageContentBlock::ToolCall { .. }) = call else {
        println!("  the model answered without calling the tool: {reply:?}");
        return Ok(());
    };
    println!("  delivered: {blocks:?}");
    // An approved call streams through untouched, so executing it is the
    // application's job: the gate decided that it may run, not that it ran.
    let MessageContentBlock::ToolCall { arguments, .. } = &call else {
        unreachable!("matched above")
    };
    let output = append_note(&notebook, arguments);
    println!("  app executed: {output}");
    let (reply, _, follow_thinking) = drain(
        client
            .generate_stream(follow_up(&model, &approved_prompt, call, output))
            .await?,
    )
    .await;
    println!("  reply: {}", reply.trim());
    println!("  thinking blocks: {thinking} then {follow_thinking}");

    let denied_prompt = "Save the note \"ship it\" in the audit notebook.";
    println!("\nStage 2: the same tool, a notebook the policy refuses");
    let (reply, blocks, thinking) = gated_turn(&client, &model, denied_prompt).await?;
    let call = recorder.take().into_iter().next();
    let Some(call @ MessageContentBlock::ToolCall { .. }) = call else {
        println!("  the model answered without calling the tool: {reply:?}");
        return Ok(());
    };
    println!("  delivered: {blocks:?}");
    // The denial is already a ToolResult, so nothing executed and the model
    // still gets an answer it can act on.
    let denial = blocks
        .iter()
        .find_map(|block| match block {
            MessageContentBlock::ToolResult { output, .. } => Some(output.clone()),
            _ => None,
        })
        .unwrap_or_default();
    let (reply, _, follow_thinking) = drain(
        client
            .generate_stream(follow_up(&model, denied_prompt, call, denial))
            .await?,
    )
    .await;
    println!("  reply: {}", reply.trim());
    println!("  thinking blocks: {thinking} then {follow_thinking}");

    println!(
        "\nAudit log, {} of {} entries",
        hitl.audit_len(),
        hitl.max_audit_entries()
    );
    for entry in hitl.audit_log() {
        println!(
            "  {:8} {:12} approver {}",
            entry.status,
            entry.action_requested,
            entry
                .approver_id
                .unwrap_or_else(|| "unattributed".to_string())
        );
    }
    println!(
        "  notebook {OPEN_NOTEBOOK:?} holds {:?}",
        notebook.lock().unwrap_or_else(|p| p.into_inner())
    );

    Ok(())
}
