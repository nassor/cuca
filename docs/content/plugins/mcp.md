+++
title = "MCP"
description = "The Model Context Protocol client plugin: transports, the stateless 2026-07-28 protocol, and tool call routing."
template = "page.html"
weight = 1
+++

# MCP

<dl class="page-facts">
<dt>In one line</dt>
<dd><code>McpPlugin</code> connects to one Model Context Protocol server, discovers its tools once, and executes matching <code>ToolCall</code> blocks as <code>ToolResult</code> blocks.</dd>
<dt>You need</dt>
<dd>The <code>plugin-mcp</code> feature and a reachable MCP server: a child process over stdio or a Streamable HTTP endpoint.</dd>
<dt>Read this if</dt>
<dd>You are registering <code>McpPlugin</code> or choosing an <code>McpTransport</code>.</dd>
</dl>

## Feature and entry type

| Item | Value |
|---|---|
| Feature flag | `plugin-mcp` (enables `dep:rmcp` and `dep:tokio`) |
| Plugin name | `"mcp-connector"` |
| Entry types | `McpPlugin`, `McpTransport` |
| Attachment | `CucaClientBuilder::register_plugin` |
| Needs | An MCP server speaking protocol version `2026-07-28` |

## Hooks

`McpPlugin` acts on `on_stream_chunk`. It does not override `execute_local_tool`.

| Hook | Behavior |
|---|---|
| `on_request` | No-op. The caller publishes the discovered tools through `McpPlugin::tools()`. |
| `on_stream_chunk` | Replaces a `ToolCall` whose name is a discovered tool with a `ToolResult`. Other blocks and unknown tool names pass through. |
| `on_response_complete` | No-op. |

- The replacement `ToolResult` carries the tool's rendered output, or the error text when the call fails. The hook itself returns `Ok(())`.
- The stream pauses while the tool runs: the hook blocks the calling thread until the server replies.
- `McpPlugin::call_tool(name, arguments)` is the async entry point. It awaits the same worker channel through `spawn_blocking`.

### Protocol

- The plugin speaks only MCP `2026-07-28`, the stateless protocol. It has no connection-setup phase and no legacy fallback.
- Every request carries its protocol version, client identity and client capabilities in `_meta`.
- Connecting probes the server once with `server/discover`, then lists every tool with pagination-aware `tools/list`.
- A `tools/call` answered with `resultType: "input_required"` (SEP-2322) or `resultType: "task"` (SEP-2663) is not driven further. It surfaces as `PluginError::NotSupported`.
- Arguments map to a JSON object: an object stays as is, `null` becomes no arguments, anything else is a validation error.
- Output rendering: text blocks join with newlines, images, audio and resources become compact placeholders, error results get a prefix, and `structuredContent` renders as JSON when no content blocks exist.

## Configuration

`McpPlugin` has no config struct and no keys with defaults. The `McpTransport` argument of `McpPlugin::connect` selects the connection.

| Variant | Fields | Connection |
|---|---|---|
| `Stdio` | `command`, `args` | Spawns the executable as a child process and speaks MCP over its stdio pipes. |
| `StreamableHttp` | `url` | Connects over Streamable HTTP, the `2026-07-28` binding. |
| `WebSocket` | `url` | Not connectable: resolves to `PluginError::NotSupported`. |

`McpPlugin::connect_stdio(command)` is shorthand for `McpPlugin::connect(McpTransport::stdio(command))`, a stdio transport with no arguments.

## Caps and at-cap policy

No cap: nothing grows with traffic.

- The tool map is filled once at connect time from `tools/list`.
- `McpPlugin::tools()` returns the full list sorted by name.
- One worker thread named `cuca-mcp-worker` runs every call over an unbounded `std::sync::mpsc` channel; each hook call blocks until its own reply arrives.
- No usage gauge exists.

## Errors

The hooks return no `PluginError`. A failed tool call becomes `ToolResult` text. `connect` and `call_tool` return these variants:

| Variant | When |
|---|---|
| `Io` | The stdio child process fails to spawn, or the worker thread fails to spawn. |
| `NotSupported` | `McpTransport::WebSocket` is used, or a tool call answers `input_required` or `task`. |
| `Validation` (schema `tools/call arguments`) | Tool arguments are neither a JSON object nor `null`. |
| `Internal` | The worker runtime fails to build, discovery or `tools/list` fails, a tool call fails, the worker exits or drops its reply channel, or the server returns an unrecognized response. |

## Example

`examples/mcp.rs` re-executes its own binary with `--mcp-add-server` as a stdio server with one `add` tool. The model calls `add`, the child computes the sum, and a second turn answers from it.

```rust,name=Register McpPlugin and publish its tools
let mcp = Arc::new(McpPlugin::connect_stdio("github-mcp-server").await?);

let mut request = UnifiedRequest::new(&model);
for tool in mcp.tools() {
    request = request.add_tool(ToolDefinition {
        name: tool.name.to_string(),
        description: tool.description.as_deref().unwrap_or_default().to_string(),
        input_schema: serde_json::Value::Object((*tool.input_schema).clone()),
    });
}

let client = CucaClient::builder()
    .with_provider(ProviderEndpoint::LlamaCpp)
    .with_base_url(base_url)
    .register_plugin(mcp as Arc<dyn CucaPlugin>)
    .build()?;
```

The example needs a `llama-server` on port 1234 with the demo model loaded. `CUCA_BASE_URL` and `CUCA_MODEL` retarget it at any OpenAI-compatible server.

```bash,name=Runs the same on all three platforms
cargo run --example mcp --features "provider-llamacpp plugin-mcp"
```

```text,name=Expected output
Discovered over stdio, from tools/list:
  add: Add two numbers and return their sum

Turn 1: the model calls the MCP tool, the child process answers
  tool result from the MCP server: "42"
  thinking blocks: 78

Turn 2: the same question, with the server's answer in the prompt
  reply: The sum of 21 and 21 is 42.
  thinking blocks: 44
```

The tool-call id, the reply wording and the thinking-block counts depend on the model. The output above came from `google/gemma-4-12b-qat`.

## Related pages

- [Plugins](@/plugins/_index.md): hook order, error handling and the plugin tier rules.
- [Error types](@/reference/errors.md): `PluginError` and `CucaError`.
- [Point the client at another OpenAI-compatible server](@/guides/other-openai-server.md): retarget the example.
