//! Shared harness for the live llama.cpp integration tests.
//!
//! Targets llama-server's OpenAI-compatible route (default
//! `http://127.0.0.1:1234/v1`, no API key). llama-server's own default port is
//! 8080, so the suite expects it started with `--port 1234`, or `CUCA_BASE_URL`
//! pointed at wherever it listens. Server-dependent tests skip when the server
//! is unreachable unless `CUCA_REQUIRE_LIVE=1` is set, which turns the skip
//! into a panic.
//!
//! `#![allow(dead_code)]`: each test binary compiles this module standalone, and
//! no single binary uses every helper, so shared harness helpers would
//! otherwise trip `-D warnings` in CI.
#![allow(dead_code)]

use std::any::Any;
use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, SystemTime};

use cuca::plugin::CucaPlugin;
use cuca::request::AgentResponseStream;
use cuca::types::{MessageContentBlock, ProviderEndpoint};
use cuca::{CucaClient, CucaClientBuilder, PluginError, UnifiedRequest};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio_stream::StreamExt;

/// Live entity-extraction adapter shared by the entity-extraction suite and
/// the cross-capability combination suite.
#[cfg(feature = "service-entity-extraction")]
pub mod extraction;

/// Test plugin that rewrites the **first** model-generated chunk of a stream
/// (`Text` or `Thinking`) into a synthetic `ToolCall`.
///
/// Registered *before* a tool-executing plugin, this turns an ordinary live
/// model turn into a real exercise of that plugin's machinery: `on_stream_chunk`
/// hooks run in registration order over one shared block (`src/client.rs`), so
/// the target plugin sees a `ToolCall` it owns and the block the consumer
/// finally receives is whatever the target produced. Without it, a live
/// "reply with ok" turn never emits a tool call, and a tool plugin registered
/// on the client is never engaged at all.
///
/// `Thinking` counts as a trigger on purpose: which of the two a small model
/// emits first is a token-budget accident, irrelevant to whether the target
/// plugin executes, and keying on `Text` alone makes every test using this
/// fixture fragile against a reasoning-heavy reply.
pub struct ToolCallInjector {
    call_id: String,
    tool: String,
    arguments: serde_json::Value,
    injected: AtomicBool,
}

impl ToolCallInjector {
    pub fn new(
        call_id: impl Into<String>,
        tool: impl Into<String>,
        arguments: serde_json::Value,
    ) -> Self {
        Self {
            call_id: call_id.into(),
            tool: tool.into(),
            arguments,
            injected: AtomicBool::new(false),
        }
    }

    /// The synthetic tool call's id, for matching the resulting `ToolResult`.
    pub fn call_id(&self) -> &str {
        &self.call_id
    }

    /// True once a chunk has actually been rewritten. Tests assert this so that
    /// "the live model emitted nothing" fails loudly instead of vacuously
    /// passing.
    pub fn injected(&self) -> bool {
        self.injected.load(Ordering::SeqCst)
    }
}

impl CucaPlugin for ToolCallInjector {
    fn name(&self) -> &'static str {
        "test-tool-call-injector"
    }

    fn on_stream_chunk(&self, chunk: &mut MessageContentBlock) -> Result<(), PluginError> {
        if matches!(
            chunk,
            MessageContentBlock::Text(_) | MessageContentBlock::Thinking { .. }
        ) && !self.injected.swap(true, Ordering::SeqCst)
        {
            *chunk = MessageContentBlock::ToolCall {
                id: self.call_id.clone(),
                name: self.tool.clone(),
                arguments: self.arguments.clone(),
            };
        }
        Ok(())
    }
}

/// The output of the delivered `ToolResult` for `call_id`, if the stream
/// carried one.
pub fn tool_result_output(blocks: &[MessageContentBlock], call_id: &str) -> Option<String> {
    blocks.iter().find_map(|block| match block {
        MessageContentBlock::ToolResult {
            tool_call_id,
            output,
        } if tool_call_id == call_id => Some(output.clone()),
        _ => None,
    })
}

/// The llama-server endpoint the suite targets, unless `CUCA_BASE_URL`
/// overrides it.
pub const DEFAULT_BASE_URL: &str = "http://127.0.0.1:1234/v1";

/// Effective base URL: `$CUCA_BASE_URL` or [`DEFAULT_BASE_URL`].
pub fn base_url() -> String {
    std::env::var("CUCA_BASE_URL").unwrap_or_else(|_| DEFAULT_BASE_URL.to_string())
}

/// Probe the server: `GET {base}/models` with a 2s timeout.
///
/// Ok(model ids) when the server answers; Err(reason) otherwise.
pub fn server_probe() -> Result<Vec<String>, String> {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| format!("failed to build probe runtime: {e}"))?;
    rt.block_on(async {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(2))
            .build()
            .map_err(|e| format!("failed to build probe client: {e}"))?;
        let url = format!("{}/models", base_url());
        let resp = http
            .get(&url)
            .send()
            .await
            .map_err(|e| format!("GET {url} failed: {e}"))?;
        let body: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| format!("GET {url} bad body: {e}"))?;
        let ids: Vec<String> = body
            .get("data")
            .and_then(serde_json::Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|m| m.get("id").and_then(serde_json::Value::as_str))
            .map(str::to_owned)
            .collect();
        if ids.is_empty() {
            return Err(format!("GET {url} returned no model ids"));
        }
        Ok(ids)
    })
}

/// Gate a server-dependent test: Ok(()) to proceed, Err(reason) to skip.
/// With `CUCA_REQUIRE_LIVE=1` an unreachable server panics instead.
pub fn require_server() -> Result<(), String> {
    match server_probe() {
        Ok(_) => Ok(()),
        Err(reason) => {
            if std::env::var("CUCA_REQUIRE_LIVE").is_ok_and(|v| v == "1") {
                panic!("CUCA_REQUIRE_LIVE=1 but llama.cpp is unreachable: {reason}");
            }
            Err(reason)
        }
    }
}

/// Model id under test: `$CUCA_MODEL`, else the first id the server reports.
pub fn model_name() -> String {
    if let Ok(name) = std::env::var("CUCA_MODEL")
        && !name.is_empty()
    {
        return name;
    }
    server_probe()
        .ok()
        .and_then(|ids| ids.into_iter().next())
        .unwrap_or_else(|| {
            panic!(
                "no model id: set CUCA_MODEL or start llama-server at {}",
                base_url()
            )
        })
}

/// Builder preset for a llama.cpp client at `base`, with no API key.
///
/// Split out of [`client_with_plugins`] because the combination suites must
/// attach a prompt cache or an orchestrator, and point at a mock-server
/// address, before `build()`.
///
/// Call sites pass a `/v1`-suffixed base because the chat route appends the
/// suffix when it is missing, which would move a mock server's request path.
pub fn llamacpp_builder(base: impl Into<String>) -> CucaClientBuilder {
    CucaClient::builder()
        .with_provider(ProviderEndpoint::LlamaCpp)
        .with_base_url(base)
}

/// llama.cpp client with the given plugins registered (no API key).
pub fn client_with_plugins(plugins: Vec<Arc<dyn CucaPlugin>>) -> CucaClient {
    let mut builder = llamacpp_builder(base_url());
    for plugin in plugins {
        builder = builder.register_plugin(plugin);
    }
    builder
        .build()
        .expect("llama.cpp client build must succeed")
}

/// llama.cpp client with no plugins.
pub fn client() -> CucaClient {
    client_with_plugins(Vec::new())
}

/// Drain a stream, panicking with context on timeout or stream errors.
pub async fn drain_timeout(mut stream: AgentResponseStream, secs: u64) -> Vec<MessageContentBlock> {
    let mut blocks = Vec::new();
    tokio::time::timeout(Duration::from_secs(secs), async {
        while let Some(item) = stream.next().await {
            match item {
                Ok(block) => blocks.push(block),
                Err(err) => panic!("stream error: {err}"),
            }
        }
    })
    .await
    .unwrap_or_else(|_| panic!("stream did not finish within {secs}s"));
    blocks
}

/// Concatenated `Text` content of a drained block sequence.
pub fn text_of(blocks: &[MessageContentBlock]) -> String {
    blocks
        .iter()
        .filter_map(|block| match block {
            MessageContentBlock::Text(text) => Some(text.as_str()),
            _ => None,
        })
        .collect()
}

/// Wall-clock milliseconds since the UNIX epoch.
pub fn now_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Spawn an ephemeral loopback OpenAI-compatible SSE server that counts one
/// dispatch per accepted connection and answers every request with the same
/// canned single-`Text`-block stream (`text`, `finish_reason: "stop"`,
/// `[DONE]`).
///
/// Returns the address it listens on; the spawned task is aborted when the
/// test's tokio runtime is torn down, so no explicit shutdown is needed. This
/// is the deterministic counterpart to the live server: dispatch counts, cache
/// hits, and hook-invocation counts cannot be asserted against a real model.
pub async fn spawn_counting_sse_server(
    dispatches: Arc<AtomicUsize>,
    text: &'static str,
) -> SocketAddr {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind ephemeral port");
    let addr = listener.local_addr().expect("local addr");
    tokio::spawn(async move {
        loop {
            let Ok((mut socket, _)) = listener.accept().await else {
                break;
            };
            dispatches.fetch_add(1, Ordering::SeqCst);
            // A short JSON chat-completion body from a one-line test message
            // always fits a single 4KiB read, matching the existing provider
            // end-to-end tests' single-`read` pattern.
            let mut buf = [0u8; 4096];
            let _ = socket.read(&mut buf).await;
            let mut response = String::from(
                "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\nconnection: close\r\n\r\n",
            );
            for frame in [
                format!(r#"data: {{"choices":[{{"delta":{{"content":"{text}"}}}}]}}"#),
                r#"data: {"choices":[{"delta":{},"finish_reason":"stop"}]}"#.to_string(),
                "data: [DONE]".to_string(),
            ] {
                let body = format!("{frame}\n\n");
                response.push_str(&format!("{:x}\r\n{body}\r\n", body.len()));
            }
            response.push_str("0\r\n\r\n");
            let _ = socket.write_all(response.as_bytes()).await;
            let _ = socket.shutdown().await;
        }
    });
    addr
}

/// Re-panic with the original message when a helper thread panicked, so
/// `CUCA_REQUIRE_LIVE=1` gates keep their documented panic semantics.
fn panic_message(payload: Box<dyn Any + Send>, fallback: &str) -> String {
    payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_else(|| fallback.to_string())
}

/// [`require_server`] run on a plain OS thread.
///
/// `server_probe` builds its own tokio runtime and `block_on` panics when a
/// runtime is already running, so the gate must never execute inside a
/// `#[tokio::test]` body; this variant keeps live tests on the default
/// current-thread flavor. A panic inside the thread (e.g. the
/// `CUCA_REQUIRE_LIVE=1` gate) is re-raised on the calling thread.
pub fn require_live_server() -> Result<(), String> {
    match std::thread::spawn(require_server).join() {
        Ok(result) => result,
        Err(payload) => panic!(
            "{}",
            panic_message(payload, "live server gate thread panicked")
        ),
    }
}

/// [`model_name`] resolved on a plain OS thread (see [`require_live_server`]).
pub fn live_model() -> String {
    match std::thread::spawn(model_name).join() {
        Ok(model) => model,
        Err(payload) => panic!("{}", panic_message(payload, "live model thread panicked")),
    }
}

/// A small, fast request built from an already resolved model id so no server
/// probe runs inside the tokio runtime.
///
/// Uses a 128-token cap: live tests assert a `Text` block, and small models
/// such as gemma 4 e4b can spend a large share of a tight budget on thinking
/// blocks before emitting text.
pub fn live_request(prompt: &str, model: &str) -> UnifiedRequest {
    UnifiedRequest::new(model)
        .add_system_message("You are concise.")
        .add_user_message(prompt)
        .set_max_tokens(128)
}
