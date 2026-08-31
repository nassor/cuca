//! MCP connector plugin.
//!
//! [`McpPlugin`] connects to a Model Context Protocol (MCP) server,
//! spawned as a child process over stdio or reached over Streamable HTTP,
//! discovers its tools (`tools/list`) and executes them as normalized
//! [`ToolCall`](crate::types::MessageContentBlock::ToolCall) →
//! [`ToolResult`](crate::types::MessageContentBlock::ToolResult) exchanges in
//! the stream pipeline.
//!
//! # Protocol: MCP 2026-07-28 (stateless)
//!
//! This plugin speaks **only** the stateless MCP protocol version `2026-07-28`
//! (MCP 2.0): there is no connection-setup phase and no shared connection
//! state. Every request is self-contained: the protocol version, client
//! identity, and client capabilities travel in the request's `_meta`
//! (`io.modelcontextprotocol/protocolVersion`,
//! `io.modelcontextprotocol/clientInfo`,
//! `io.modelcontextprotocol/clientCapabilities`), and the client probes the
//! server once with `server/discover` to pick a mutually supported protocol
//! version before listing tools. rmcp 3.x attaches that per-request `_meta`
//! automatically after discovery, so the plugin never constructs it by hand.
//! No legacy-protocol fallback exists: a server that answers `server/discover`
//! with anything but a `DiscoverResult` is an error, never a downgrade.
//!
//! # Bridge design: sync hook → async client
//!
//! [`CucaPlugin::on_stream_chunk`] is synchronous, but an MCP client is async.
//! Blocking the caller's executor to wait on the client would deadlock a
//! current-thread runtime, so [`McpPlugin`] instead owns a **dedicated worker
//! OS thread** ([`std::thread::spawn`]) with its own single-threaded tokio
//! runtime. The worker thread receives the [`McpTransport`] description and
//! **constructs the rmcp client transport inside its own `block_on`**: the
//! child process (`TokioChildProcess::new`) is spawned while the worker's
//! runtime is current. That placement is load-bearing: tokio registers a
//! child process's stdio fds with the I/O driver of the runtime that is
//! current at spawn time. Building the transport on the caller's runtime
//! would bind the pipes to the *caller's* driver, which the worker's epoll
//! never drives; the worker would park in `do_epoll_wait` forever as soon as
//! the caller blocked. Constructing inside the worker's own runtime binds the
//! fds to the worker's driver, the only driver the worker ever runs, so
//! the pipe I/O always progresses.
//!
//! After the transport is built, the worker runs the stateless discovery
//! (`server/discover` + `tools/list`, both inside the same `block_on`),
//! reports the tool list, and loops over a `std::sync::mpsc` channel of
//! tool-call requests (unbounded: senders never block). The sync hook sends a
//! request and waits on the reply with a plain std blocking `recv()`. That
//! block is safe even on a current-thread runtime: the calling thread may be
//! the runtime's only thread, but the client's I/O lives entirely on the
//! worker's driver, so no other driver has to keep running for the exchange
//! to complete. The wait uses std primitives, which carry no tokio runtime
//! guard (tokio's `blocking_send`/`blocking_recv` panic when called from
//! inside *any* runtime), so the hook may block even on a runtime worker
//! thread such as the pipeline's `poll_next`.
//!
//! Pause semantics: the stream pipeline **pauses** until the tool executes
//! (the model's stream stops producing blocks while the tool runs). That is by
//! design, not a caveat: the hook blocks whatever thread calls it, but the
//! worker owns its runtime exclusively, so the blocked thread never has to
//! make progress on it. Async callers (and tests) use
//! [`McpPlugin::call_tool`], which awaits the same channel via a
//! `spawn_blocking` wait instead of blocking the caller.
//!
//! # API mapping to rmcp 3.x
//!
//! [`ClientHandler`](rmcp::ClientHandler) is a *trait* the client service
//! implements ([`ClientInfo`] implements it out of the box), and the
//! connection is established by
//! [`serve_client_with_lifecycle`] with
//! [`ClientLifecycleMode::Discover`](rmcp::ClientLifecycleMode), which sends
//! the stateless `server/discover` probe, negotiates the protocol version, and
//! returns a `RunningService` that dereferences to the request
//! [`Peer`](rmcp::service::Peer). Discovery happens inside
//! [`McpPlugin::connect`] on the worker thread. rmcp 3.x also ships no
//! WebSocket client transport (its `ws` module is commented out), so
//! [`McpTransport::WebSocket`] resolves to [`PluginError::NotSupported`] while
//! `Stdio` and `StreamableHttp` are live.
//!
//! Tool calls map [`serde_json::Value`] arguments to rmcp's `JsonObject`
//! (an object → the map, `null` → absent, anything else → a `Validation`
//! error) and render the `CallToolResult` content blocks into the
//! `ToolResult`'s single string output (text blocks joined by newlines,
//! images/audio/resources as compact placeholders, error results prefixed,
//! `structuredContent` rendered as JSON when no content blocks are present).
//!
//! # Multi Round-Trip Requests (`resultType: "input_required"`)
//!
//! The 2026-07-28 spec lets a server answer `tools/call` with
//! `resultType: "input_required"` (MRTR, SEP-2322): the call is not finished
//! and the client must gather the requested inputs (elicitation, sampling,
//! roots) and retry with `inputResponses`/`requestState`. The connector has
//! **no elicitation UI** (it is a headless stream-pipeline plugin with no way
//! to prompt the user), so it deliberately does **not** drive MRTR rounds.
//! It uses rmcp's one-shot `call_tool_once` and surfaces any
//! `input_required` result (and the SEP-2663 `resultType: "task"` result,
//! which likewise cannot be completed synchronously) as a distinct
//! [`PluginError::NotSupported`] so the model sees a clear failure instead of
//! a fabricated "complete" result. The one-shot call also avoids rmcp's
//! auto-MRTR loop, whose default client handler would silently *decline*
//! elicitation requests, behavior the model could mistake for a real tool
//! outcome.

use std::collections::HashMap;
use std::sync::{Mutex, mpsc};

use rmcp::model::{
    CallToolRequestParams, CallToolResponse, CallToolResult, ClientInfo, ContentBlock,
    Implementation, ProtocolVersion, Tool,
};
use rmcp::serve_client_with_lifecycle;
use rmcp::service::{ClientLifecycleMode, RoleClient, RunningService};
use rmcp::transport::{
    ConfigureCommandExt, IntoTransport, StreamableHttpClientTransport, TokioChildProcess,
};
use serde_json::Value;
use tokio::sync::oneshot;

use crate::error::PluginError;
use crate::plugin::CucaPlugin;
use crate::request::{UnifiedRequest, UnifiedResponse};
use crate::types::MessageContentBlock;

/// A tool-call request handed from the plugin to the worker thread.
///
/// The reply travels back over the embedded std mpsc channel: the sync hook
/// blocks on it, async callers wait on a blocking-pool thread.
struct ToolRequest {
    name: String,
    arguments: Value,
    reply: mpsc::Sender<Result<String, PluginError>>,
}

/// A transport description for an MCP server connection.
///
/// [`McpPlugin::connect`] resolves each variant into the matching rmcp client
/// transport: [`Stdio`](Self::Stdio) spawns the server executable as a child
/// process speaking JSON-RPC over its stdio pipes
/// (`rmcp::transport::TokioChildProcess`), and
/// [`StreamableHttp`](Self::StreamableHttp) uses
/// `rmcp::transport::StreamableHttpClientTransport` (the 2026-07-28
/// Streamable HTTP binding; the old HTTP+SSE transport is gone).
/// [`WebSocket`](Self::WebSocket) has no rmcp 3.x client transport, so
/// connecting it yields [`PluginError::NotSupported`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum McpTransport {
    /// Spawn the server executable as a child process and speak MCP over
    /// stdio (the classic local-server layout, e.g. `github-mcp-server`).
    Stdio {
        /// The server executable to spawn.
        command: String,
        /// Extra arguments passed to the executable.
        args: Vec<String>,
    },
    /// A `ws://`/`wss://` endpoint.
    ///
    /// Not connectable with rmcp 3.x: no WebSocket client transport is
    /// implemented.
    WebSocket {
        /// The endpoint URL.
        url: String,
    },
    /// A Streamable HTTP endpoint (MCP 2026-07-28 binding).
    StreamableHttp {
        /// The MCP endpoint URL.
        url: String,
    },
}

impl McpTransport {
    /// Build a stdio transport with no extra arguments.
    pub fn stdio(command: impl Into<String>) -> Self {
        McpTransport::Stdio {
            command: command.into(),
            args: Vec::new(),
        }
    }
}

/// MCP connector plugin.
///
/// Connects to one MCP server, caches its discovered tools, and executes tool
/// calls from the stream hook. Tools are injected into the caller's tool set
/// via [`Self::tools`]; the plugin itself never rewrites prompts
/// ([`CucaPlugin::on_request`] is a no-op).
pub struct McpPlugin {
    /// Discovered tools, keyed by name. A `Mutex` lets the synchronous stream
    /// hook check membership without round-tripping to the worker thread.
    tools: Mutex<HashMap<String, Tool>>,
    /// Channel to the dedicated worker thread that owns the live client.
    worker: mpsc::Sender<ToolRequest>,
}

impl McpPlugin {
    /// Connect to an MCP server spawned as a child process over stdio.
    ///
    /// Convenience for [`Self::connect`] with [`McpTransport::stdio`]:
    ///
    /// ```ignore
    /// let plugin = McpPlugin::connect_stdio("github-mcp-server").await?;
    /// ```
    ///
    /// # Errors
    ///
    /// The [`Self::connect`] errors, for a stdio transport.
    pub async fn connect_stdio(command: impl Into<String>) -> Result<Self, PluginError> {
        Self::connect(McpTransport::stdio(command)).await
    }

    /// Spawns the worker thread, which constructs the rmcp client transport
    /// (the child process for [`McpTransport::Stdio`], the Streamable HTTP
    /// connection for [`McpTransport::StreamableHttp`]) **inside its own tokio
    /// runtime** and then runs the stateless protocol setup, the
    /// `server/discover` probe and pagination-aware `tools/list`, handled by
    /// rmcp's [`serve_client_with_lifecycle`] with
    /// [`ClientLifecycleMode::Discover`] plus `list_all_tools`, on the same
    /// runtime, after which it answers tool-call requests over the worker
    /// channel. Building the transport on the worker keeps the child's stdio
    /// fds bound to the worker's I/O driver, so the synchronous stream hook
    /// may block on a reply even on a current-thread runtime (see module
    /// docs). Returns once the tool list is cached; a failed setup (missing
    /// executable, protocol error, closed stream) is surfaced as an error.
    pub async fn connect(transport: McpTransport) -> Result<Self, PluginError> {
        // Each arm hands the worker a factory closure that builds the concrete
        // rmcp client transport inside the worker's own `block_on`, never on
        // the caller's runtime, whose driver would otherwise own the child's
        // stdio fds (see module docs for why that deadlocks the sync hook).
        match transport {
            McpTransport::Stdio { command, args } => {
                Self::connect_inner(move || async move {
                    let cmd = tokio::process::Command::new(&command).configure(|cmd| {
                        cmd.args(&args);
                    });
                    TokioChildProcess::new(cmd).map_err(|err| PluginError::Io(err.to_string()))
                })
                .await
            }
            McpTransport::StreamableHttp { url } => {
                Self::connect_inner(move || async move {
                    Ok(StreamableHttpClientTransport::from_uri(url))
                })
                .await
            }
            McpTransport::WebSocket { url } => Err(PluginError::NotSupported(format!(
                "WebSocket MCP transport is not implemented by rmcp 3.x; \
                 use Stdio or StreamableHttp (requested {url:?})"
            ))),
        }
    }

    /// Core connect: hand a transport-building factory to a fresh worker
    /// thread and wait for its tool discovery to land.
    ///
    /// The factory is an async closure that resolves to the rmcp client
    /// transport; it is executed inside the worker's own `block_on`, so the
    /// transport is constructed (child process spawned / Streamable HTTP
    /// transport built) while the worker's runtime is current. Generic over
    /// the rmcp transport so tests can inject an in-memory `tokio::io::duplex`
    /// pair via a factory that returns the prebuilt client end.
    async fn connect_inner<T, E, A, F, Fut>(factory: F) -> Result<Self, PluginError>
    where
        T: IntoTransport<RoleClient, E, A> + Send + 'static,
        E: std::error::Error + Send + Sync + 'static,
        F: FnOnce() -> Fut + Send + 'static,
        Fut: Future<Output = Result<T, PluginError>>,
    {
        let (init_tx, init_rx) = oneshot::channel::<Result<Vec<Tool>, PluginError>>();
        let (worker_tx, worker_rx) = mpsc::channel::<ToolRequest>();

        // The worker owns transport construction + the client on its own
        // thread + runtime; see the module docs for why the sync hook can then
        // block safely. Dropping `worker_tx` (with the plugin) closes
        // `worker_rx`, ending the loop.
        std::thread::Builder::new()
            .name("cuca-mcp-worker".to_owned())
            .spawn(move || {
                let rt = match tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                {
                    Ok(rt) => rt,
                    Err(err) => {
                        let _ = init_tx.send(Err(PluginError::Internal(format!(
                            "failed to build worker runtime: {err}"
                        ))));
                        return;
                    }
                };
                worker_run(&rt, factory, worker_rx, init_tx);
            })
            .map_err(|err| PluginError::Io(format!("failed to spawn mcp worker thread: {err}")))?;

        let tools = match init_rx.await {
            Ok(Ok(tools)) => tools,
            Ok(Err(err)) => return Err(err),
            Err(_recv) => {
                return Err(PluginError::Internal(
                    "mcp worker thread exited before completing the connection setup".into(),
                ));
            }
        };

        let registry: HashMap<String, Tool> = tools
            .iter()
            .map(|tool| (tool.name.to_string(), tool.clone()))
            .collect();
        Ok(McpPlugin {
            tools: Mutex::new(registry),
            worker: worker_tx,
        })
    }

    /// All discovered tools, sorted by name for deterministic iteration.
    pub fn tools(&self) -> Vec<Tool> {
        // A poisoned registry (a writer panicked mid-write) still holds a
        // consistent map; recover the guard rather than failing the call.
        let registry = self
            .tools
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut tools: Vec<Tool> = registry.values().cloned().collect();
        tools.sort_by(|a, b| a.name.cmp(&b.name));
        tools
    }

    /// Execute one tool call, awaiting the worker's reply.
    ///
    /// Async face for async callers and tests; the synchronous stream hook
    /// uses `call_tool_sync` instead. The reply is awaited on a
    /// blocking-pool thread because the channel is std mpsc (not awaitable).
    ///
    /// # Errors
    ///
    /// [`PluginError::Internal`] when the worker thread has exited or its
    /// reply channel closed; otherwise whatever the server's tool call
    /// returned, including [`PluginError::NotSupported`] for a result this
    /// adapter cannot represent (input-required or task results).
    pub async fn call_tool(&self, name: &str, arguments: Value) -> Result<String, PluginError> {
        let (reply_tx, reply_rx) = mpsc::channel();
        let request = ToolRequest {
            name: name.to_owned(),
            arguments,
            reply: reply_tx,
        };
        self.worker
            .send(request)
            .map_err(|_| PluginError::Internal("mcp worker task exited".into()))?;
        tokio::task::spawn_blocking(move || reply_rx.recv())
            .await
            .map_err(|_| PluginError::Internal("mcp reply wait task failed".into()))?
            .map_err(|_| PluginError::Internal("mcp worker dropped the reply channel".into()))?
    }

    /// Execute one tool call from the synchronous stream hook.
    ///
    /// Sends the request over the unbounded std channel (never blocks) and
    /// waits on the reply with a plain std `recv()`. Because the client never
    /// runs on the caller's executor, the wait parks only the calling thread,
    /// legal even on a runtime worker thread such as the pipeline's
    /// `poll_next` (std blocking carries no tokio runtime guard; see module
    /// docs). The stream pipeline pauses for the duration.
    fn call_tool_sync(&self, name: &str, arguments: Value) -> Result<String, PluginError> {
        let (reply_tx, reply_rx) = mpsc::channel();
        let request = ToolRequest {
            name: name.to_owned(),
            arguments,
            reply: reply_tx,
        };
        self.worker
            .send(request)
            .map_err(|_| PluginError::Internal("mcp worker task exited".into()))?;
        reply_rx
            .recv()
            .map_err(|_| PluginError::Internal("mcp worker dropped the reply channel".into()))?
    }
}

impl CucaPlugin for McpPlugin {
    /// Stable plugin name.
    fn name(&self) -> &'static str {
        "mcp-connector"
    }

    /// No-op: discovered tools are injected by the caller via
    /// [`McpPlugin::tools`]; the connector does not rewrite prompts.
    fn on_request(&self, _req: &mut UnifiedRequest) -> Result<(), PluginError> {
        Ok(())
    }

    /// Route `ToolCall` blocks whose tool name is in the discovered set.
    ///
    /// Executes the call through the sync bridge and replaces the block with
    /// a `ToolResult` carrying the tool's rendered text output (or the error
    /// text when the call fails), the same shape the guardrails plugin uses.
    /// Unknown tool names pass through untouched, so other plugins' tools and
    /// provider-native tools are unaffected.
    fn on_stream_chunk(&self, chunk: &mut MessageContentBlock) -> Result<(), PluginError> {
        let MessageContentBlock::ToolCall {
            id,
            name,
            arguments,
        } = chunk
        else {
            return Ok(());
        };
        // Membership check only; recover the registry even if poisoned.
        let known = self
            .tools
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .contains_key(name.as_str());
        if !known {
            return Ok(());
        }
        // `arguments` and `id` are moved out rather than cloned: `*chunk` is
        // overwritten on the next line, so the borrowed block's fields are
        // dead. Cloning `arguments` deep-copied the whole tool-call JSON tree
        // on every intercepted chunk.
        let output = match self.call_tool_sync(name, std::mem::take(arguments)) {
            Ok(output) => output,
            Err(err) => err.to_string(),
        };
        *chunk = MessageContentBlock::ToolResult {
            tool_call_id: std::mem::take(id),
            output,
        };
        Ok(())
    }

    /// No-op: the connector keeps no per-response state.
    fn on_response_complete(&self, _res: &UnifiedResponse) -> Result<(), PluginError> {
        Ok(())
    }
}

/// Worker entry, run on the dedicated thread.
///
/// Performs transport construction + stateless discovery inside one
/// `block_on`: the factory builds the rmcp client transport while the
/// worker's runtime is current (so a spawned child's stdio fds bind to the
/// worker's driver, and the Streamable HTTP transport's worker task is
/// spawned on it), then rmcp's `serve_client_with_lifecycle` runs the
/// `server/discover` probe in [`ClientLifecycleMode::Discover`] and
/// `list_all_tools` walks tools/list pagination: every request carrying its
/// protocol version / client identity / capabilities in `_meta`. Reports the
/// tool list over `init`, then serves tool-call requests until the plugin is
/// dropped (which closes `requests` and ends the loop). The runtime stays
/// alive across requests: each tool call is executed inside its own short
/// `block_on` that drives the serve-loop task, and the thread parks in the
/// std `recv()` in between, so the client can never run on a caller's
/// executor.
fn worker_run<T, E, A, F, Fut>(
    rt: &tokio::runtime::Runtime,
    factory: F,
    requests: mpsc::Receiver<ToolRequest>,
    init: oneshot::Sender<Result<Vec<Tool>, PluginError>>,
) where
    T: IntoTransport<RoleClient, E, A> + Send + 'static,
    E: std::error::Error + Send + Sync + 'static,
    F: FnOnce() -> Fut + Send + 'static,
    Fut: Future<Output = Result<T, PluginError>>,
{
    let setup: Result<(RunningService<RoleClient, ClientInfo>, Vec<Tool>), PluginError> = rt
        .block_on(async move {
            let client = factory().await?;
            let mut client_info = ClientInfo::default();
            client_info.protocol_version = ProtocolVersion::V_2026_07_28;
            client_info.client_info =
                Implementation::new("cuca-mcp-connector", env!("CARGO_PKG_VERSION"));
            let running = serve_client_with_lifecycle(
                client_info,
                client,
                ClientLifecycleMode::Discover {
                    preferred_versions: vec![ProtocolVersion::V_2026_07_28],
                },
            )
            .await
            .map_err(|err| PluginError::Internal(format!("mcp discovery failed: {err}")))?;
            let tools = running
                .list_all_tools()
                .await
                .map_err(|err| PluginError::Internal(format!("mcp tools/list failed: {err}")))?;
            Ok((running, tools))
        });

    let (running, tools) = match setup {
        Ok(pair) => pair,
        Err(err) => {
            let _ = init.send(Err(err));
            return;
        }
    };
    let _ = init.send(Ok(tools));

    while let Ok(request) = requests.recv() {
        let result = rt.block_on(execute_tool_call(&running, request.name, request.arguments));
        let _ = request.reply.send(result);
    }
}

/// Execute one tool call on the connected client and render its result.
///
/// Uses rmcp's one-shot `call_tool_once` (no MRTR auto-fulfilment): a
/// `resultType: "complete"` result is rendered; a `resultType:
/// "input_required"` result (the tool needs interactive input this headless
/// connector cannot elicit) or `resultType: "task"` (SEP-2663, a
/// task-backed call that must be polled) is surfaced as a distinct
/// [`PluginError::NotSupported`] instead of a fabricated tool output. See the
/// module docs for the MRTR rationale.
async fn execute_tool_call(
    running: &RunningService<RoleClient, ClientInfo>,
    name: String,
    arguments: Value,
) -> Result<String, PluginError> {
    // MCP `tools/call` arguments must be a JSON object; `null` means "no
    // arguments". Anything else is a malformed model output: surface it as a
    // validation error rather than guessing.
    let arguments = match arguments {
        Value::Object(map) => Some(map),
        Value::Null => None,
        other => {
            return Err(PluginError::Validation {
                schema: "tools/call arguments".to_owned(),
                message: format!(
                    "tool arguments must be a JSON object, got {}",
                    json_type_name(&other)
                ),
            });
        }
    };
    let request = match arguments {
        Some(map) => CallToolRequestParams::new(name.clone()).with_arguments(map),
        None => CallToolRequestParams::new(name.clone()),
    };
    let response = running
        .call_tool_once(request)
        .await
        .map_err(|err| PluginError::Internal(format!("mcp tool call failed: {err}")))?;
    match response {
        CallToolResponse::Complete(result) => Ok(render_tool_result(&result)),
        CallToolResponse::InputRequired(result) => Err(PluginError::NotSupported(format!(
            "tool \"{name}\" requires interactive input (MCP resultType \"input_required\", \
             {} input request(s)) but the mcp-connector has no elicitation UI, so the \
             multi round-trip request cannot be fulfilled",
            result
                .input_requests
                .as_ref()
                .map_or(0, |requests| requests.len()),
        ))),
        CallToolResponse::Task(_) => Err(PluginError::NotSupported(format!(
            "tool \"{name}\" returned a task (MCP SEP-2663 resultType \"task\") which the \
             mcp-connector does not poll",
        ))),
        // `CallToolResponse` is non-exhaustive (rmcp may add variants); a
        // response we do not recognize is a protocol-level surprise.
        _ => Err(PluginError::Internal(
            "mcp tool call returned an unrecognized response".into(),
        )),
    }
}

/// Render an MCP `CallToolResult` into the plugin's single-string output:
/// text blocks joined by newlines, images/audio and resources as compact
/// placeholders, `structuredContent` as JSON when no content blocks are
/// present. Error results are prefixed with `"tool error: "`.
fn render_tool_result(result: &CallToolResult) -> String {
    let mut parts = Vec::with_capacity(result.content.len());
    for content in &result.content {
        let piece = match content {
            ContentBlock::Text(text) => text.text.clone(),
            ContentBlock::Image(image) => format!(
                "[image: {} ({} base64 chars)]",
                image.mime_type,
                image.data.len()
            ),
            ContentBlock::Audio(audio) => format!(
                "[audio: {} ({} base64 chars)]",
                audio.mime_type,
                audio.data.len()
            ),
            ContentBlock::Resource(_) => "[resource content]".to_owned(),
            ContentBlock::ResourceLink(_) => "[resource link]".to_owned(),
            // `ContentBlock` is non-exhaustive (rmcp may add variants); render
            // unknown blocks as a compact placeholder rather than failing.
            _ => "[content block]".to_owned(),
        };
        parts.push(piece);
    }
    if parts.is_empty()
        && let Some(structured) = &result.structured_content
    {
        parts.push(structured.to_string());
    }
    let output = parts.join("\n");
    if result.is_error == Some(true) {
        format!("tool error: {output}")
    } else {
        output
    }
}

/// Short JSON type name for validation error messages.
fn json_type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

#[cfg(all(test, feature = "plugin-mcp"))]
mod tests {
    // In-memory server recipe (rmcp 3.x): rmcp ships no dedicated in-memory
    // test transport, so the suite pairs `tokio::io::duplex` byte streams: one
    // end feeds the plugin's worker client, the other a `ServerHandler`
    // served with `serve_server` on its own OS thread/runtime. JSON-RPC
    // framing is rmcp's `AsyncRwTransport` codec on both ends; no external
    // process, no network. The server speaks the stateless 2026-07-28
    // protocol: `serve_server` answers the client's `server/discover` probe
    // with the default `DiscoverResult` (all known protocol versions), and
    // the mock's `tools/list` / `tools/call` handlers serve the tools. The
    // mock server thread exits when the client end closes (the serve loop
    // hits EOF and `waiting()` returns).

    use std::collections::BTreeMap;
    use std::sync::Arc;
    use std::thread;

    use rmcp::handler::server::ServerHandler;
    use rmcp::model::{
        CallToolRequestParams, CallToolResponse, CallToolResult, ContentBlock, ElicitRequest,
        ElicitRequestParams, ErrorData, InputRequest, InputRequiredResult, ListToolsResult,
        PaginatedRequestParams, ServerCapabilities, ServerInfo, Tool,
    };
    use rmcp::service::{RequestContext, RoleServer, serve_server};
    use serde_json::{Value, json};
    use tokio::io::duplex;

    use super::{McpPlugin, McpTransport};
    use crate::error::PluginError;
    use crate::plugin::CucaPlugin;
    use crate::types::MessageContentBlock;

    /// The mock MCP server: two tools (`get_weather`, `search`).
    #[derive(Debug, Clone)]
    struct MockMcpServer {
        tools: Vec<Tool>,
    }

    impl MockMcpServer {
        fn new() -> Self {
            MockMcpServer {
                tools: vec![
                    Tool::new(
                        "get_weather",
                        "Current weather for a city",
                        tool_schema(json!({
                            "type": "object",
                            "properties": { "city": { "type": "string" } },
                            "required": ["city"],
                        })),
                    ),
                    Tool::new(
                        "search",
                        "Full-text search over documents",
                        tool_schema(json!({
                            "type": "object",
                            "properties": { "query": { "type": "string" } },
                            "required": ["query"],
                        })),
                    ),
                ],
            }
        }
    }

    /// Parse a JSON schema literal into rmcp's `Arc<JsonObject>` input schema.
    fn tool_schema(value: Value) -> Arc<serde_json::Map<String, Value>> {
        Arc::new(serde_json::from_value(value).expect("tool input schema must be a JSON object"))
    }

    impl ServerHandler for MockMcpServer {
        fn get_info(&self) -> ServerInfo {
            // `ServerInfo` is `#[non_exhaustive]`, so mutate the default.
            let mut info = ServerInfo::default();
            info.capabilities = ServerCapabilities::builder().enable_tools().build();
            info
        }

        fn list_tools(
            &self,
            _request: Option<PaginatedRequestParams>,
            _context: RequestContext<RoleServer>,
        ) -> impl Future<Output = Result<ListToolsResult, ErrorData>> + Send + '_ {
            std::future::ready(Ok(ListToolsResult::with_all_items(self.tools.clone())))
        }

        async fn call_tool(
            &self,
            request: CallToolRequestParams,
            _context: RequestContext<RoleServer>,
        ) -> Result<CallToolResponse, ErrorData> {
            let arg = |key: &str| -> String {
                request
                    .arguments
                    .as_ref()
                    .and_then(|args| args.get(key))
                    .and_then(Value::as_str)
                    .unwrap_or("unknown")
                    .to_owned()
            };
            // MRTR exercise path: `get_weather` with city "ELICIT" answers with
            // `resultType: "input_required"` carrying one elicitation request,
            // the 2026-07-28 Multi Round-Trip Requests contract. The connector
            // has no elicitation UI, so the client must surface a distinct
            // NotSupported error instead of a fabricated "complete" result.
            if request.name.as_ref() == "get_weather" && arg("city") == "ELICIT" {
                let mut input_requests = BTreeMap::new();
                input_requests.insert(
                    "city".to_owned(),
                    InputRequest::Elicitation(ElicitRequest::new(
                        ElicitRequestParams::UrlElicitationParams {
                            meta: None,
                            message: "Which city's weather?".to_owned(),
                            url: "https://example.com/authorize".to_owned(),
                            elicitation_id: "weather-auth-1".to_owned(),
                        },
                    )),
                );
                return Ok(CallToolResponse::InputRequired(
                    InputRequiredResult::from_input_requests(input_requests),
                ));
            }
            let output = match request.name.as_ref() {
                "get_weather" => {
                    format!("weather report for {}: sunny, 22C", arg("city"))
                }
                "search" => format!("search for {}: 3 results", arg("query")),
                other => {
                    return Ok(CallToolResponse::Complete(CallToolResult::error(vec![
                        ContentBlock::text(format!("unknown tool {other}")),
                    ])));
                }
            };
            Ok(CallToolResponse::Complete(CallToolResult::success(vec![
                ContentBlock::text(output),
            ])))
        }
    }

    /// Serve the mock handler on one end of a duplex pair, on its own OS
    /// thread with a dedicated current-thread runtime. The thread returns once
    /// the client end closes.
    fn spawn_mock_server(server_side: tokio::io::DuplexStream) -> thread::JoinHandle<()> {
        thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("build mock server runtime");
            rt.block_on(async move {
                let server = serve_server(MockMcpServer::new(), server_side)
                    .await
                    .expect("mock server setup");
                let _reason = server.waiting().await.expect("mock server serve loop");
            });
        })
    }

    /// Build a plugin connected to the mock server through an in-memory
    /// duplex; the factory handed to `connect_inner` returns the prebuilt
    /// duplex client end (memory-backed, no fds, so the driver-binding concern
    /// does not apply).
    async fn connect_to_mock() -> (McpPlugin, thread::JoinHandle<()>) {
        let (client_side, server_side) = duplex(1 << 16);
        let server = spawn_mock_server(server_side);
        let plugin = McpPlugin::connect_inner(move || async move { Ok(client_side) })
            .await
            .expect("plugin connect");
        (plugin, server)
    }

    #[test]
    fn stdio_constructor_builds_command_and_empty_args() {
        let transport = McpTransport::stdio("github-mcp-server");
        match transport {
            McpTransport::Stdio { command, args } => {
                assert_eq!(command, "github-mcp-server");
                assert!(args.is_empty(), "stdio must default to empty args");
            }
            other => panic!("expected Stdio, got {other:?}"),
        }
    }

    #[test]
    fn plugin_and_transport_are_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<McpPlugin>();
        assert_send_sync::<McpTransport>();
    }

    #[tokio::test]
    async fn connect_discovers_tools() {
        let (plugin, server) = connect_to_mock().await;

        let tools = plugin.tools();
        assert_eq!(tools.len(), 2);
        let names: Vec<&str> = tools.iter().map(|tool| tool.name.as_ref()).collect();
        assert_eq!(names, ["get_weather", "search"]);

        drop(plugin);
        server.join().expect("mock server thread must exit");
    }

    #[tokio::test]
    async fn stream_chunk_routes_known_tool_and_passes_unknown() {
        let (plugin, server) = connect_to_mock().await;

        // Known tool: replaced by a ToolResult carrying the mock's response.
        let mut call = MessageContentBlock::ToolCall {
            id: "call_1".to_owned(),
            name: "get_weather".to_owned(),
            arguments: json!({ "city": "Berlin" }),
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
                assert!(
                    output.contains("weather report for Berlin"),
                    "unexpected output: {output}"
                );
            }
            other => panic!("expected ToolResult, got {other:?}"),
        }

        // Unknown tool: passes through untouched.
        let mut foreign = MessageContentBlock::ToolCall {
            id: "call_2".to_owned(),
            name: "not_a_mcp_tool".to_owned(),
            arguments: json!({ "q": 1 }),
        };
        plugin
            .on_stream_chunk(&mut foreign)
            .expect("hook must succeed");
        match foreign {
            MessageContentBlock::ToolCall { ref name, .. } => {
                assert_eq!(name, "not_a_mcp_tool");
            }
            other => panic!("expected ToolCall passthrough, got {other:?}"),
        }

        drop(plugin);
        server.join().expect("mock server thread must exit");
    }

    #[tokio::test]
    async fn call_tool_direct_executes_via_worker() {
        let (plugin, server) = connect_to_mock().await;

        let output = plugin
            .call_tool("search", json!({ "query": "rmcp" }))
            .await
            .expect("call_tool must succeed");
        assert!(
            output.contains("search for rmcp"),
            "unexpected output: {output}"
        );

        drop(plugin);
        server.join().expect("mock server thread must exit");
    }
    #[tokio::test]
    async fn input_required_result_surfaces_not_supported() {
        let (plugin, server) = connect_to_mock().await;

        // The mock answers `get_weather` with city "ELICIT" using an MRTR
        // `resultType: "input_required"` result (one elicitation request). The
        // connector has no elicitation UI, so the call must surface a distinct
        // NotSupported error naming the result type and the input request
        // count, never a fabricated "complete" tool output.
        let result = plugin
            .call_tool("get_weather", json!({ "city": "ELICIT" }))
            .await;
        match result {
            Err(PluginError::NotSupported(message)) => {
                assert!(
                    message.contains("input_required"),
                    "message must name the result type: {message}"
                );
                assert!(
                    message.contains("1 input request(s)"),
                    "message must report the input request count: {message}"
                );
            }
            Err(other) => panic!("expected NotSupported, got {other:?}"),
            Ok(output) => {
                panic!("expected NotSupported, got a fabricated complete result: {output:?}");
            }
        }

        drop(plugin);
        server.join().expect("mock server thread must exit");
    }

    #[tokio::test]
    async fn websocket_transport_reports_not_supported() {
        let result = McpPlugin::connect(McpTransport::WebSocket {
            url: "ws://localhost:9999".to_owned(),
        })
        .await;
        match result {
            Err(PluginError::NotSupported(_)) => {}
            Err(other) => panic!("expected NotSupported, got {other:?}"),
            Ok(_) => panic!("websocket connect must not succeed"),
        }
    }
    #[tokio::test]
    async fn connect_stdio_documented_shape_compiles_and_errors_on_missing_binary() {
        // Compile-time proof of the documented `connect_stdio` shape
        // (`McpPlugin::connect_stdio("github-mcp-server").await?`): the
        // runtime half spawns a child process that cannot exist, so the
        // connection must fail with a spawn/IO error rather than hang.
        let result = McpPlugin::connect_stdio("definitely-not-a-real-mcp-server-binary")
            .await
            .map(|_| ());
        match result {
            Err(PluginError::Io(_)) => {}
            Err(other) => panic!("expected Io spawn error, got {other:?}"),
            Ok(()) => panic!("connect_stdio must fail for a missing binary"),
        }
    }
}
