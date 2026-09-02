//! Give a model a library of `SKILL.md` instructions it can discover and read.
//!
//! The demo writes one skill and one bundled reference file into a temporary
//! directory, loads them with `SkillsPlugin::from_dir`, and lets the model find
//! its own way: `on_request` injects a catalog naming the three tools and every
//! loaded skill, the model calls `skill` to read the instructions, those
//! instructions send it to `skill_read` for the conversion factor, and the last
//! turn, which offers no tools at all, answers from what it gathered. The
//! directory is removed when the program ends.
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
//! cargo run --example skills --features provider-llamacpp,plugin-skills
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
//! Example: `CUCA_BASE_URL=http://127.0.0.1:8000/v1 CUCA_MODEL=<server-model-id> cargo run --example skills --features provider-llamacpp,plugin-skills`
//!
//! # Output
//!
//! From one run against `google/gemma-4-12b-qat` on llama.cpp:
//!
//! ```text
//! Loaded from /tmp/cuca-skills-3284142
//!   unit-convert: Convert between metric and imperial units using exact factors (1 reference file(s))
//!
//! The model reaches for the skill out of the injected catalog
//!   catalog in the outbound request, last line: - unit-convert: Convert between metric and imperial units using exact factors
//!   skill returned: {"description":"Convert between metric and imperial units using exact factors","instructions":"# Unit conversion\n\nRead `factors.md` for the exact factor, multiply, and answer with the number\nfollowed by the unit and nothing else.","name":"unit-convert","references":["factors.md"]}
//!   thinking blocks: 138
//!
//! The model reaches for the reference file
//!   catalog in the outbound request, last line: - unit-convert: Convert between metric and imperial units using exact factors
//!   skill_read returned: 1 mile = 1.609344 kilometers
//! 1 pound = 0.45359237 kilograms
//!
//!   thinking blocks: 122
//!
//! The answer, from the instructions and the factor
//!   reply: 19.312128 kilometers
//!   thinking blocks: 369
//! ```
//!
//! The temporary path carries the process id, and the reply wording and block
//! counts depend on the model; `12 * 1.609344` does not.
//!
//! With no server on the base URL, the program prints one line naming the
//! address and exits successfully.
//!
//! # Why the catalog and the tools are separate things
//!
//! The injected catalog is prose: it tells the model which skills exist and
//! which tools reach them. It is not what makes a call possible. The wire
//! `tools` array is, which is why `skill_tools` below declares the two tools
//! this demo uses even though the catalog already names all three. A tool
//! result that fails, an unknown skill or a missing reference, comes back as
//! error text inside the `ToolResult` rather than failing the hook, so the
//! model can correct itself in the same conversation.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use cuca::plugin::CucaPlugin;
use cuca::types::{
    MessageContentBlock, MessageRole, ProviderEndpoint, ToolDefinition, UnifiedMessage,
};
use cuca::{
    AgentResponseStream, CucaClient, PluginError, SkillsPlugin, UnifiedRequest, UnifiedResponse,
};
use serde_json::json;
use tokio_stream::StreamExt;

/// The question, whose answer needs both the skill and its reference file.
const QUESTION: &str = "Load the unit-convert skill, then tell me how many kilometers 12 miles is.";

/// The `SKILL.md` the demo writes, in the agentskills.io frontmatter shape the
/// plugin parses: a leading `---`, `name:` and `description:`, a closing `---`,
/// then the instruction body. `license:` is parsed and ignored.
const SKILL_MD: &str = "---\n\
name: unit-convert\n\
description: Convert between metric and imperial units using exact factors\n\
license: Apache-2.0\n\
---\n\
\n\
# Unit conversion\n\
\n\
Read `factors.md` for the exact factor, multiply, and answer with the number\n\
followed by the unit and nothing else.\n";

/// One bundled reference file, reachable through the `skill_read` tool.
const FACTORS_MD: &str = "1 mile = 1.609344 kilometers\n1 pound = 0.45359237 kilograms\n";

/// Removes the demo skills directory even when a turn returns early.
struct SkillsDir(PathBuf);

impl SkillsDir {
    /// Lay out `<tmp>/cuca-skills-<pid>/unit-convert/{SKILL.md,references/}`:
    /// the plugin scans direct subdirectories, so the skill name is a directory.
    fn create() -> std::io::Result<Self> {
        let root = std::env::temp_dir().join(format!("cuca-skills-{}", std::process::id()));
        let skill = root.join("unit-convert");
        std::fs::create_dir_all(skill.join("references"))?;
        std::fs::write(skill.join("SKILL.md"), SKILL_MD)?;
        std::fs::write(skill.join("references").join("factors.md"), FACTORS_MD)?;
        Ok(Self(root))
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for SkillsDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Keeps the catalog `on_request` injected and the tool call the model issued.
///
/// Hooks run in registration order over one shared request and one shared
/// block, so this plugin sees the injected catalog only when it is registered
/// after `SkillsPlugin`, and sees the model's `ToolCall` only because the
/// skills plugin has not replaced it yet.
#[derive(Default)]
struct Recorder {
    catalog: Mutex<Option<String>>,
    call: Mutex<Option<MessageContentBlock>>,
}

impl Recorder {
    fn take_call(&self) -> Option<MessageContentBlock> {
        self.call.lock().unwrap_or_else(|p| p.into_inner()).take()
    }
}

impl CucaPlugin for Recorder {
    fn name(&self) -> &'static str {
        "recorder"
    }

    fn on_request(&self, req: &mut UnifiedRequest) -> Result<(), PluginError> {
        *self.catalog.lock().unwrap_or_else(|p| p.into_inner()) = req
            .messages
            .iter()
            .filter(|message| message.role == MessageRole::System)
            .flat_map(|message| message.content.iter())
            .filter_map(|block| match block {
                MessageContentBlock::Text(text) => Some(text.clone()),
                _ => None,
            })
            .next_back();
        Ok(())
    }

    fn on_stream_chunk(&self, chunk: &mut MessageContentBlock) -> Result<(), PluginError> {
        if matches!(chunk, MessageContentBlock::ToolCall { .. }) {
            *self.call.lock().unwrap_or_else(|p| p.into_inner()) = Some(chunk.clone());
        }
        Ok(())
    }

    fn on_response_complete(&self, _res: &UnifiedResponse) -> Result<(), PluginError> {
        Ok(())
    }
}

/// Two of the three tools the plugin answers, declared so the model can call
/// them.
///
/// The injected catalog tells the model the tools exist; the wire `tools` array
/// is what lets it emit a call for one.
fn skill_tools() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition {
            name: "skill".to_string(),
            description: "Load a skill's full instructions by name.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": { "name": { "type": "string" } },
                "required": ["name"],
            }),
        },
        ToolDefinition {
            name: "skill_read".to_string(),
            description: "Read a reference file bundled with a skill.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "skill": { "type": "string" },
                    "file": { "type": "string" },
                },
                "required": ["skill", "file"],
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

/// A turn over `history`, with the skill tools offered when `with_tools`.
fn turn(model: &str, history: &[UnifiedMessage], with_tools: bool) -> UnifiedRequest {
    let mut request = UnifiedRequest::new(model)
        // A reasoning model spends the token budget on thinking first, so a
        // tight cap can end the turn before any text is emitted.
        .set_max_tokens(768);
    request.messages = history.to_vec();
    if with_tools {
        for tool in skill_tools() {
            request = request.add_tool(tool);
        }
    }
    request
}

/// Append the call the model made and the result the plugin produced, so the
/// next turn sees the exchange as the conversation it was.
fn extend(history: &mut Vec<UnifiedMessage>, call: MessageContentBlock, output: &str) {
    let call_id = match &call {
        MessageContentBlock::ToolCall { id, .. } => id.clone(),
        _ => String::new(),
    };
    history.push(UnifiedMessage {
        role: MessageRole::Assistant,
        content: vec![call],
        name: None,
        tool_call_id: None,
    });
    history.push(UnifiedMessage {
        role: MessageRole::Tool,
        content: vec![MessageContentBlock::ToolResult {
            tool_call_id: call_id.clone(),
            output: output.to_string(),
        }],
        name: None,
        tool_call_id: Some(call_id),
    });
}

/// Name of the tool a recorded call invoked.
fn tool_name(call: &MessageContentBlock) -> &str {
    match call {
        MessageContentBlock::ToolCall { name, .. } => name,
        _ => "",
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Base URL and model come from the environment so the example runs
    // against any OpenAI-compatible server; the defaults target a local
    // llama.cpp server (see the module docs for the override recipe).
    let base_url =
        std::env::var("CUCA_BASE_URL").unwrap_or_else(|_| "http://127.0.0.1:1234/v1".to_string());
    let model = std::env::var("CUCA_MODEL").unwrap_or_else(|_| "google/gemma-4-e4b".to_string());

    let dir = SkillsDir::create()?;
    let skills = Arc::new(SkillsPlugin::from_dir(dir.path())?);
    println!("Loaded from {}", dir.path().display());
    for skill in skills.list_skills() {
        println!(
            "  {}: {} ({} reference file(s))",
            skill.name,
            skill.description,
            skill.references.len()
        );
    }

    let recorder = Arc::new(Recorder::default());
    let client = CucaClient::builder()
        .with_provider(ProviderEndpoint::LlamaCpp)
        .with_base_url(base_url.clone())
        // The same recorder sits on both sides of the skills plugin, because
        // its two hooks need opposite positions: the chunk hook must run
        // before the plugin replaces the model's `ToolCall`, and the request
        // hook must run after the plugin has appended its catalog.
        .register_plugin(Arc::clone(&recorder) as Arc<dyn CucaPlugin>)
        .register_plugin(Arc::clone(&skills) as Arc<dyn CucaPlugin>)
        .register_plugin(Arc::clone(&recorder) as Arc<dyn CucaPlugin>)
        .build()?;

    let mut history = vec![UnifiedMessage::user(QUESTION)];

    // Two tool-using turns: the model discovers the skill, then the reference
    // file the skill's own instructions point it at.
    for stage in [
        "the skill out of the injected catalog",
        "the reference file",
    ] {
        println!("\nThe model reaches for {stage}");
        let stream = match client.generate_stream(turn(&model, &history, true)).await {
            Ok(stream) => stream,
            Err(error) => {
                println!("\nNo server answered at {base_url}: {error}");
                println!("Start llama-server there, or set CUCA_BASE_URL, then run this again.");
                return Ok(());
            }
        };
        let (reply, results, thinking) = drain(stream).await;
        if let Some(catalog) = recorder
            .catalog
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .as_deref()
            .and_then(|catalog| catalog.lines().next_back())
        {
            println!("  catalog in the outbound request, last line: {catalog}");
        }
        let (Some(call), Some((_, output))) = (recorder.take_call(), results.into_iter().next())
        else {
            println!("  the model answered without calling a skill tool: {reply:?}");
            return Ok(());
        };
        println!("  {} returned: {output}", tool_name(&call));
        println!("  thinking blocks: {thinking}");
        extend(&mut history, call, &output);
    }

    // The final turn offers no tools: everything the skill asked for is in the
    // transcript, so the only thing left is the answer.
    println!("\nThe answer, from the instructions and the factor");
    let (reply, _, thinking) = drain(
        client
            .generate_stream(turn(&model, &history, false))
            .await?,
    )
    .await;
    println!("  reply: {}", reply.trim());
    println!("  thinking blocks: {thinking}");

    Ok(())
}
