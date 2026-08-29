//! Integration tests for the MCP plugin (`plugin-mcp`).
//!
//! # Sync hook vs. current-thread runtimes
//!
//! The sync hook (`CucaPlugin::on_stream_chunk`) blocks the pipeline until the
//! tool call completes, which is pause semantics by design, and is safe on
//! current-thread runtimes: the plugin's worker thread constructs the child
//! process *inside its own tokio runtime*, so the child's stdio fds bind to
//! the worker's I/O driver (tokio registers them with the runtime current at
//! spawn). The hook's blocking std `recv()` therefore parks only the calling
//! thread; no other driver has to keep running for the pipe I/O to complete.
//! These tests exercise the hook directly on the `#[tokio::test]` current-
//! thread runtime against a real stdio child process.
//!
//! # In-test rmcp echo server (test-binary re-execution)
//!
//! The MCP echo server is no longer an external script fixture: each test spawns
//! **this test binary itself** ([`std::env::current_exe`]) with the single
//! argument `--mcp-echo-server`, and a [`ctor`] constructor runs *before*
//! libtest's `main`, spots the flag via `args_os` (no UTF-8 panics), and
//! serves the rmcp stdio echo server defined in [`mcp_echo_server`]. The
//! child runs its own current-thread tokio runtime, built inside the
//! interceptor with I/O and timers enabled (`enable_all`, the same flavor
//! the plugin's worker uses), blocks on the rmcp serve future, and
//! `std::process::exit(0)`s once stdin reaches EOF — the same stdio shutdown
//! rule the script fixture followed. The plugin still talks to a genuine
//! child process over real pipes, so the worker-thread / sync-hook /
//! io-driver binding rationale above is unchanged.

#![cfg(all(feature = "provider-llamacpp", feature = "plugin-mcp"))]

mod common;

use std::sync::Arc;

use cuca::plugin::CucaPlugin;
use cuca::types::MessageContentBlock;
use cuca::{McpPlugin, McpTransport};
use serde_json::json;

/// Pre-main interceptor: when this test binary is re-executed with
/// `--mcp-echo-server` (spawned by [`connect_echo_plugin`]), serve the rmcp
/// stdio echo server instead of running the test suite. `run()` never
/// returns: it exits the process once the stdio connection closes.
#[ctor::ctor(unsafe)]
fn intercept_mcp_echo_server() {
    if std::env::args_os().any(|arg| arg == "--mcp-echo-server") {
        mcp_echo_server::run();
    }
}

/// Connect to the in-test rmcp echo server over a real stdio child process:
/// the child is this test binary re-executed with `--mcp-echo-server`, which
/// [`intercept_mcp_echo_server`] turns into a pre-main rmcp stdio server.
async fn connect_echo_plugin() -> McpPlugin {
    let exe = std::env::current_exe().expect("current test binary path");
    McpPlugin::connect(McpTransport::Stdio {
        command: exe.to_string_lossy().into_owned(),
        args: vec!["--mcp-echo-server".to_string()],
    })
    .await
    .unwrap_or_else(|e| panic!("McpPlugin::connect to the in-test echo server failed: {e}"))
}

#[tokio::test]
async fn connect_discovers_echo_tool() {
    let plugin = connect_echo_plugin().await;
    let tools = plugin.tools();
    assert!(
        tools.iter().any(|tool| tool.name.as_ref() == "echo"),
        "expected the echo tool in {tools:?}"
    );
}

#[tokio::test]
async fn call_tool_echo_round_trips_child_process() {
    let plugin = connect_echo_plugin().await;
    // Real round trip: plugin worker -> child process stdin -> child stdout ->
    // worker -> caller.
    let output = plugin
        .call_tool("echo", json!({ "text": "hi" }))
        .await
        .expect("call_tool must succeed");
    assert!(
        output.contains("hi"),
        "echo output must contain the echoed text, got {output:?}"
    );
}

#[tokio::test]
async fn stream_chunk_routes_known_tool_and_passes_unknown() {
    let plugin = connect_echo_plugin().await;
    assert!(
        plugin
            .tools()
            .iter()
            .any(|tool| tool.name.as_ref() == "echo"),
        "expected the echo tool"
    );

    // The hook runs directly on this current-thread test runtime. Its blocking
    // recv parks the calling thread while the plugin's worker thread runs the
    // MCP exchange; the child's stdio fds are owned by the worker's driver
    // (transport construction happens inside the worker's runtime), so the
    // pipe I/O completes and the hook returns.
    let mut call = MessageContentBlock::ToolCall {
        id: "call_1".to_string(),
        name: "echo".to_string(),
        arguments: json!({ "text": "hi" }),
    };
    plugin
        .on_stream_chunk(&mut call)
        .expect("hook must succeed");
    match call {
        MessageContentBlock::ToolResult {
            tool_call_id,
            output,
        } => {
            assert_eq!(tool_call_id, "call_1");
            assert!(output.contains("hi"), "unexpected output: {output}");
        }
        other => panic!("expected ToolResult, got {other:?}"),
    }

    // An unknown tool name passes through untouched.
    let mut foreign = MessageContentBlock::ToolCall {
        id: "call_2".to_string(),
        name: "not_an_mcp_tool".to_string(),
        arguments: json!({ "q": 1 }),
    };
    plugin
        .on_stream_chunk(&mut foreign)
        .expect("hook must succeed");
    assert!(
        matches!(&foreign, MessageContentBlock::ToolCall { name, .. } if name == "not_an_mcp_tool"),
        "unknown tool must pass through untouched: {foreign:?}"
    );
}

/// The live round trip must exercise the plugin's real machinery, not merely
/// coexist with it. An injector registered *before* the plugin rewrites the
/// first live model chunk into an `echo` tool call, so the MCP plugin runs a
/// genuine child-process exchange mid-stream and the block the consumer
/// receives is the echoed `ToolResult`.
#[tokio::test]
async fn live_stream_chunk_executes_the_real_mcp_tool() {
    // The gate probes the server with its own runtime, which must never run
    // inside a tokio runtime, so resolve it on a plain OS thread first.
    if let Err(reason) = common::require_live_server() {
        eprintln!("SKIP: llama.cpp not reachable: {reason}");
        return;
    }
    let model = common::live_model();
    let payload = "cuca-live-echo";
    let injector = Arc::new(common::ToolCallInjector::new(
        "live-echo-1",
        "echo",
        json!({ "text": payload }),
    ));
    let plugin = connect_echo_plugin().await;
    let client = common::client_with_plugins(vec![
        injector.clone() as Arc<dyn CucaPlugin>,
        Arc::new(plugin) as Arc<dyn CucaPlugin>,
    ]);
    let stream = client
        .generate_stream(common::live_request(
            "Reply with the single word: ok",
            &model,
        ))
        .await
        .expect("generate_stream must succeed");
    let blocks = common::drain_timeout(stream, 60).await;

    assert!(
        injector.injected(),
        "the live turn produced no model chunk to convert, so nothing was \
         exercised; got {blocks:?}"
    );
    let output = common::tool_result_output(&blocks, injector.call_id()).unwrap_or_else(|| {
        panic!("the MCP plugin must replace the injected call with a ToolResult, got {blocks:?}")
    });
    assert!(
        output.contains(payload),
        "the ToolResult must carry the child process's echoed payload, got {output:?}"
    );
    assert!(
        !blocks
            .iter()
            .any(|b| matches!(b, MessageContentBlock::ToolCall { name, .. } if name == "echo")),
        "the plugin must consume the echo call, not pass it through: {blocks:?}"
    );
}

/// The in-test MCP stdio echo server.
///
/// Replicates the `echo` tool of the former external script fixture: one tool
/// named `echo`, description "Echo the given text back verbatim", inputSchema
/// `{ type: "object", properties: { text: { type: "string" } }, required:
/// ["text"] }`. The rmcp server framework answers `server/discover`,
/// `tools/list`, `ping`, and method-not-found itself; only the tool is defined
/// here. Served over stdio; the process exits when stdin reaches EOF.
mod mcp_echo_server {
    use rmcp::handler::server::wrapper::Parameters;
    use rmcp::schemars::JsonSchema;
    use rmcp::serde::Deserialize;
    use rmcp::{ServiceExt, tool, tool_router};

    /// Arguments of the `echo` tool: a single required string `text`.
    #[derive(Deserialize, JsonSchema)]
    struct EchoArgs {
        text: String,
    }

    /// The echo server service: one tool, no other capabilities.
    struct EchoServer;

    #[tool_router(server_handler)]
    impl EchoServer {
        #[tool(description = "Echo the given text back verbatim")]
        async fn echo(&self, params: Parameters<EchoArgs>) -> String {
            params.0.text
        }
    }

    /// Serve the echo server over stdio on a current-thread runtime and exit
    /// the process when the connection closes (stdin EOF). Never returns.
    pub(crate) fn run() -> ! {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap_or_else(|err| {
                eprintln!("mcp echo server: failed to build runtime: {err}");
                std::process::exit(1);
            });
        runtime.block_on(async {
            let service = EchoServer
                .serve(rmcp::transport::stdio())
                .await
                .unwrap_or_else(|err| {
                    eprintln!("mcp echo server: failed to serve: {err}");
                    std::process::exit(1);
                });
            service.waiting().await.unwrap_or_else(|err| {
                eprintln!("mcp echo server: serve loop failed: {err}");
                std::process::exit(1);
            });
        });
        std::process::exit(0);
    }
}
