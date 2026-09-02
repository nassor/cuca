//! Expose an MCP server's tools to a model, and route its calls back.
//!
//! `McpPlugin` connects over stdio to a server that is this same example binary
//! re-executed with `--mcp-add-server`, so the demo needs nothing installed and
//! still speaks JSON-RPC over a real child process's pipes. Discovery runs at
//! connect time, the discovered tool is published to the model, and the call
//! the model makes is executed by the child mid-stream and delivered as a
//! `ToolResult`. A second turn answers from it.
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
//! cargo run --example mcp --features provider-llamacpp,plugin-mcp
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
//! Example: `CUCA_BASE_URL=http://127.0.0.1:8000/v1 CUCA_MODEL=<server-model-id> cargo run --example mcp --features provider-llamacpp,plugin-mcp`
//!
//! # Output
//!
//! From one run against `google/gemma-4-12b-qat` on llama.cpp:
//!
//! ```text
//! Discovered over stdio, from tools/list:
//!   add: Add two numbers and return their sum
//!
//! Turn 1: the model calls the MCP tool, the child process answers
//!   tool result from the MCP server: "42"
//!   thinking blocks: 78
//!
//! Turn 2: the same question, with the server's answer in the prompt
//!   reply: The sum of 21 and 21 is 42.
//!   thinking blocks: 44
//! ```
//!
//! The reply wording and the block counts depend on the model. The discovered
//! tool, its description and the sum do not: they come from the child process.
//!
//! With no server on the base URL, the program prints one line naming the
//! address and exits successfully.
//!
//! # Why the connector never rewrites the prompt
//!
//! `on_request` is a no-op, so publishing the discovered tools is the caller's
//! job, as the loop over `McpPlugin::tools` below does. A connector that
//! injected them would decide for the application which of a server's tools a
//! given turn may use, and would collide with every other source of tools in
//! the same request. The stream hook is the opposite: it claims a call only
//! when the name is in the discovered set, and leaves every other tool call
//! untouched.

use std::sync::Arc;

use cuca::plugin::CucaPlugin;
use cuca::types::{
    MessageContentBlock, MessageRole, ProviderEndpoint, ToolDefinition, UnifiedMessage,
};
use cuca::{AgentResponseStream, CucaClient, McpPlugin, McpTransport, UnifiedRequest};
use serde_json::json;
use tokio_stream::StreamExt;

/// Argument that turns this binary into the MCP server instead of the demo.
const SERVER_FLAG: &str = "--mcp-add-server";

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

/// The follow-up turn: the same prompt plus the sum the MCP server computed.
fn with_result(model: &str, prompt: &str, call_id: &str, output: &str) -> UnifiedRequest {
    UnifiedRequest::new(model)
        // A reasoning model spends the token budget on thinking first, so a
        // tight cap can end the turn before any text is emitted.
        .set_max_tokens(320)
        .add_user_message(prompt)
        .add_message(UnifiedMessage {
            role: MessageRole::Assistant,
            content: vec![MessageContentBlock::ToolCall {
                id: call_id.to_string(),
                name: "add".to_string(),
                arguments: json!({ "a": 21, "b": 21 }),
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

/// The flag check must run before any runtime exists: [`add_server::run`]
/// builds its own and `block_on` panics inside another runtime, so `main`
/// stays synchronous and starts the demo's runtime only on the demo path.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    if std::env::args_os().any(|arg| arg == SERVER_FLAG) {
        add_server::run();
    }
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?
        .block_on(demo())
}

async fn demo() -> Result<(), Box<dyn std::error::Error>> {
    // Base URL and model come from the environment so the example runs
    // against any OpenAI-compatible server; the defaults target a local
    // llama.cpp server (see the module docs for the override recipe).
    let base_url =
        std::env::var("CUCA_BASE_URL").unwrap_or_else(|_| "http://127.0.0.1:1234/v1".to_string());
    let model = std::env::var("CUCA_MODEL").unwrap_or_else(|_| "google/gemma-4-e4b".to_string());

    // The MCP server is this same binary, re-executed with the flag above, so
    // the demo needs no third-party server on PATH. The plugin still speaks
    // JSON-RPC over a real child process's stdio pipes.
    let exe = std::env::current_exe()?;
    let mcp = Arc::new(
        McpPlugin::connect(McpTransport::Stdio {
            command: exe.to_string_lossy().into_owned(),
            args: vec![SERVER_FLAG.to_string()],
        })
        .await?,
    );

    // The connector never rewrites prompts, so publishing the discovered tools
    // to the model is the caller's job.
    let mut request = UnifiedRequest::new(&model).set_max_tokens(256);
    println!("Discovered over stdio, from tools/list:");
    for tool in mcp.tools() {
        println!(
            "  {}: {}",
            tool.name,
            tool.description.as_deref().unwrap_or("(no description)")
        );
        request = request.add_tool(ToolDefinition {
            name: tool.name.to_string(),
            description: tool.description.as_deref().unwrap_or_default().to_string(),
            input_schema: serde_json::Value::Object((*tool.input_schema).clone()),
        });
    }

    let client = CucaClient::builder()
        .with_provider(ProviderEndpoint::LlamaCpp)
        .with_base_url(base_url.clone())
        .register_plugin(Arc::clone(&mcp) as Arc<dyn CucaPlugin>)
        .build()?;

    let prompt = "Add 21 and 21 with the add tool, then tell me the sum.";
    println!("\nTurn 1: the model calls the MCP tool, the child process answers");
    let stream = match client
        .generate_stream(request.add_user_message(prompt))
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
    println!("  tool result from the MCP server: {output:?}");
    println!("  thinking blocks: {thinking}");

    println!("\nTurn 2: the same question, with the server's answer in the prompt");
    let (reply, _, thinking) = drain(
        client
            .generate_stream(with_result(&model, prompt, &call_id, &output))
            .await?,
    )
    .await;
    println!("  reply: {}", reply.trim());
    println!("  thinking blocks: {thinking}");

    Ok(())
}

/// The MCP server this binary becomes when re-executed with [`SERVER_FLAG`].
///
/// One tool, `add`, served over stdio. The rmcp server framework answers
/// `server/discover`, `tools/list` and `ping` itself, so only the tool is
/// defined here. The process exits when stdin reaches EOF, which happens when
/// the plugin is dropped.
mod add_server {
    use rmcp::handler::server::wrapper::Parameters;
    use rmcp::schemars::JsonSchema;
    use rmcp::serde::Deserialize;
    use rmcp::{ServiceExt, tool, tool_router};

    /// Arguments of the `add` tool: the two summands.
    #[derive(Deserialize, JsonSchema)]
    struct AddArgs {
        a: f64,
        b: f64,
    }

    struct AddServer;

    #[tool_router(server_handler)]
    impl AddServer {
        #[tool(description = "Add two numbers and return their sum")]
        async fn add(&self, params: Parameters<AddArgs>) -> String {
            (params.0.a + params.0.b).to_string()
        }
    }

    /// Serve `add` over stdio on a runtime of its own, then exit. Never returns.
    pub(crate) fn run() -> ! {
        let runtime = match tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
        {
            Ok(runtime) => runtime,
            Err(error) => {
                eprintln!("mcp add server: failed to build runtime: {error}");
                std::process::exit(1);
            }
        };
        runtime.block_on(async {
            let service = match AddServer.serve(rmcp::transport::stdio()).await {
                Ok(service) => service,
                Err(error) => {
                    eprintln!("mcp add server: failed to serve: {error}");
                    std::process::exit(1);
                }
            };
            if let Err(error) = service.waiting().await {
                eprintln!("mcp add server: serve loop failed: {error}");
                std::process::exit(1);
            }
        });
        std::process::exit(0);
    }
}
