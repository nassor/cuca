//! Let a model run a real WebAssembly guest, under hard resource limits.
//!
//! The model is offered one tool, `run_code`, and names a published module plus
//! the input to feed it. A caller-side plugin binds the module bytes into the
//! call, then `SandboxPlugin` compiles and runs the guest in a fresh wasmtime
//! store and hands back whatever the guest wrote through `write_out`. A second
//! turn answers from that output, and one last line runs a guest that never
//! returns, so the fuel budget is the thing that stops it.
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
//! cargo run --example sandbox --features provider-llamacpp,plugin-sandbox
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
//! Example: `CUCA_BASE_URL=http://127.0.0.1:8000/v1 CUCA_MODEL=<server-model-id> cargo run --example sandbox --features provider-llamacpp,plugin-sandbox`
//!
//! # Output
//!
//! From one run against `google/gemma-4-12b-qat` on llama.cpp:
//!
//! ```text
//! Limits per call: 67108864 bytes of memory, 1000000 instructions of fuel, 5000 ms
//!
//! Turn 1: the model calls the tool, wasmtime runs the guest
//!   guest wrote: "acuc"
//!   ran in 0 ms using 65536 bytes of linear memory
//!   thinking blocks: 59
//!
//! Turn 2: the same question, with the guest's output in the prompt
//!   reply: Reversing the word "cuca" produces **acuc**.
//!   thinking blocks: 47
//!
//! A guest that never returns
//!   internal plugin error: fuel exhausted: instruction budget exceeded
//! ```
//!
//! The reply wording and the block counts depend on the model, and the
//! execution time depends on the machine. The rest does not: the guest reverses
//! its input, one memory page is 65536 bytes, and a guest that loops forever
//! always ends on the fuel budget.
//!
//! With no server on the base URL, the program prints one line naming the
//! address and exits successfully.
//!
//! # Why the module comes from the host
//!
//! `run_code` carries the module in its own arguments, base64 encoded, because
//! the plugin's contract is "run exactly these bytes" and nothing else. That
//! leaves the question of who produces the bytes, and it is not the model: no
//! model reproduces a compiled module verbatim, and one that could would be
//! choosing the code that runs. The application publishes the modules and the
//! model chooses among them by name, which is also why the guest ABI is two
//! exports and one import: a guest with no imports beyond `write_out` cannot
//! reach the host at all.

use std::sync::Arc;

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD;
use cuca::plugin::CucaPlugin;
use cuca::types::{
    MessageContentBlock, MessageRole, ProviderEndpoint, ToolDefinition, UnifiedMessage,
};
use cuca::{
    AgentResponseStream, CucaClient, PluginError, SandboxConfig, SandboxPlugin, UnifiedRequest,
};
use serde_json::{Value, json};
use tokio_stream::StreamExt;

/// The guest the model gets to run: reverse the input bytes.
///
/// The ABI is the whole contract. The host pre-loads the input at scratch
/// offset 1024 and calls `run(1024, len)`; this module writes the reversed
/// bytes at 4096 and hands that region to `write_out`, then returns 0 for
/// success. Nothing else is imported, so the guest cannot reach the host.
const REVERSE_WAT: &str = r#"
(module
  (import "env" "write_out" (func $write_out (param i32 i32)))
  (memory (export "memory") 1)
  (func (export "run") (param $ptr i32) (param $len i32) (result i32)
    (local $i i32)
    (block $done
      (loop $next
        (br_if $done (i32.ge_s (local.get $i) (local.get $len)))
        (i32.store8
          (i32.add (i32.const 4096) (local.get $i))
          (i32.load8_u
            (i32.sub
              (i32.add (local.get $ptr) (i32.sub (local.get $len) (i32.const 1)))
              (local.get $i))))
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $next)))
    (call $write_out (i32.const 4096) (local.get $len))
    (i32.const 0)))
"#;

/// A guest that never returns, so the fuel budget has to stop it.
const SPIN_WAT: &str = r#"
(module
  (func (export "run") (param i32) (param i32) (result i32)
    (loop $forever (br $forever))
    (i32.const 0)))
"#;

/// The module name the demo publishes to the model.
const MODULE: &str = "reverse";

/// Fills the module bytes into the call before the sandbox reads them.
///
/// `SandboxPlugin` expects `arguments.wasm` to be the module itself, base64
/// encoded. No model can be asked to reproduce a module byte for byte, so the
/// application owns the code and the model only names it and supplies the
/// input. `on_stream_chunk` hooks run in registration order over one shared
/// block, so registering this before the sandbox is what makes the swap work.
struct ModuleBinder;

impl CucaPlugin for ModuleBinder {
    fn name(&self) -> &'static str {
        "module-binder"
    }

    fn on_stream_chunk(&self, chunk: &mut MessageContentBlock) -> Result<(), PluginError> {
        let MessageContentBlock::ToolCall {
            id,
            name,
            arguments,
        } = chunk
        else {
            return Ok(());
        };
        if name != "run_code" {
            return Ok(());
        }
        let requested = arguments
            .get("module")
            .and_then(Value::as_str)
            .unwrap_or("");
        if requested != MODULE {
            *chunk = MessageContentBlock::ToolResult {
                tool_call_id: std::mem::take(id),
                output: format!("no module named {requested:?}; this host publishes {MODULE:?}"),
            };
            return Ok(());
        }
        if let Some(object) = arguments.as_object_mut() {
            object.insert(
                "wasm".to_string(),
                Value::String(STANDARD.encode(REVERSE_WAT)),
            );
        }
        Ok(())
    }
}

/// The tool the model is offered: a module name and the input to feed it.
fn run_code_tool() -> ToolDefinition {
    ToolDefinition {
        name: "run_code".to_string(),
        description: "Run a sandboxed WebAssembly module over an input string.".to_string(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "module": { "type": "string", "enum": [MODULE] },
                "input": { "type": "string", "description": "the text to feed the module" },
            },
            "required": ["module", "input"],
        }),
    }
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

/// The follow-up turn: the same prompt plus what the guest wrote out.
fn with_result(
    model: &str,
    prompt: &str,
    call_id: &str,
    input: &str,
    output: &str,
) -> UnifiedRequest {
    UnifiedRequest::new(model)
        // A reasoning model spends the token budget on thinking first, so a
        // tight cap can end the turn before any text is emitted.
        .set_max_tokens(320)
        .add_user_message(prompt)
        .add_message(UnifiedMessage {
            role: MessageRole::Assistant,
            content: vec![MessageContentBlock::ToolCall {
                id: call_id.to_string(),
                name: "run_code".to_string(),
                arguments: json!({ "module": MODULE, "input": input }),
            }],
            name: None,
            tool_call_id: None,
        })
        .add_message(UnifiedMessage {
            role: MessageRole::Tool,
            content: vec![MessageContentBlock::ToolResult {
                tool_call_id: call_id.to_string(),
                output: output.to_string(),
            }],
            name: None,
            tool_call_id: Some(call_id.to_string()),
        })
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Base URL and model come from the environment so the example runs
    // against any OpenAI-compatible server; the defaults target a local
    // llama.cpp server (see the module docs for the override recipe).
    let base_url =
        std::env::var("CUCA_BASE_URL").unwrap_or_else(|_| "http://127.0.0.1:1234/v1".to_string());
    let model = std::env::var("CUCA_MODEL").unwrap_or_else(|_| "google/gemma-4-e4b".to_string());

    let config = SandboxConfig::default();
    println!(
        "Limits per call: {} bytes of memory, {} instructions of fuel, {} ms",
        config.max_memory_bytes, config.max_instructions, config.timeout_ms
    );
    let sandbox = Arc::new(SandboxPlugin::new(config));
    let client = CucaClient::builder()
        .with_provider(ProviderEndpoint::LlamaCpp)
        .with_base_url(base_url.clone())
        .register_plugin(Arc::new(ModuleBinder) as Arc<dyn CucaPlugin>)
        .register_plugin(Arc::clone(&sandbox) as Arc<dyn CucaPlugin>)
        .build()?;

    let prompt = "Use the reverse module on the word cuca, then tell me what it produced.";
    println!("\nTurn 1: the model calls the tool, wasmtime runs the guest");
    let stream = match client
        .generate_stream(
            UnifiedRequest::new(&model)
                .set_max_tokens(256)
                .add_user_message(prompt)
                .add_tool(run_code_tool()),
        )
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
    let Some((call_id, output)) = results.into_iter().next() else {
        println!("  the model answered without calling the tool: {reply:?}");
        return Ok(());
    };
    println!("  guest wrote: {output:?}");
    if let Some((elapsed_ms, memory_bytes)) = sandbox.last_diagnostic() {
        println!("  ran in {elapsed_ms} ms using {memory_bytes} bytes of linear memory");
    }
    println!("  thinking blocks: {thinking}");

    println!("\nTurn 2: the same question, with the guest's output in the prompt");
    let (reply, _, thinking) = drain(
        client
            .generate_stream(with_result(&model, prompt, &call_id, "cuca", &output))
            .await?,
    )
    .await;
    println!("  reply: {}", reply.trim());
    println!("  thinking blocks: {thinking}");

    // The bound is the point of the sandbox, so it gets one line: a guest that
    // never returns is stopped by the fuel budget, not by the host waiting.
    println!("\nA guest that never returns");
    match sandbox.run(SPIN_WAT.as_bytes(), b"") {
        Ok(_) => println!("  it returned, which the fuel budget should have prevented"),
        Err(error) => println!("  {error}"),
    }

    Ok(())
}
