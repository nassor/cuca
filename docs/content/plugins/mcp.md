+++
title = "MCP"
description = "Connect to a Model Context Protocol server and let the model call its tools."
template = "page.html"
weight = 1
+++

# MCP

<dl class="page-facts">
<dt>In one line</dt>
<dd><code>McpPlugin</code> connects to one Model Context Protocol server, finds its tools, and runs the model's calls to them.</dd>
<dt>You need</dt>
<dd>The <code>plugin-mcp</code> feature and an MCP server: a program you start over stdio, or a Streamable HTTP address.</dd>
<dt>Read this if</dt>
<dd>You want the model to use tools that an MCP server provides.</dd>
</dl>

The Model Context Protocol (MCP) is a standard way for a program to offer tools to a model. This plugin is the client side. It connects once, lists the server's tools, and answers each matching `ToolCall` with a `ToolResult`.

## Register it

`connect_stdio` starts the server program and talks to it over its input and output pipes. Then you give the discovered tools to the request, so the model knows they exist.

```rust,name=Connect, publish the tools, register
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

Expected effect: when the model calls one of those tools, the stream contains a `ToolResult` with the server's answer instead of the `ToolCall`. The plugin does not add the tools to the request for you.

## Run the example

`examples/mcp.rs` starts its own binary as a small MCP server with one `add` tool. It needs a `llama-server` on port 1234 with a tool-calling model loaded. Put `--base-url` and `--model` after `--` to use another OpenAI-compatible server.

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

The reply wording and the thinking block counts depend on the model.

## Feature and entry type

| Item | Value |
|---|---|
| Feature flag | `plugin-mcp` |
| Plugin name | `"mcp-connector"` |
| Entry types | `McpPlugin`, `McpTransport` |
| Attachment | `CucaClientBuilder::register_plugin` |
| Needs | An MCP server that speaks protocol version `2026-07-28` |

## Hooks

`McpPlugin` acts on `on_stream_chunk` only.

| Hook | Behavior |
|---|---|
| `on_request` | Does nothing. Publish the tools yourself with `McpPlugin::tools()`. |
| `on_stream_chunk` | Replaces a `ToolCall` whose name is a discovered tool with a `ToolResult`. Other blocks and unknown tool names pass through. |
| `on_response_complete` | Does nothing. |

- The `ToolResult` holds the tool's output, or the error text if the call failed. The hook still returns `Ok(())`.
- The stream pauses while the tool runs.
- `McpPlugin::call_tool(name, arguments)` is the async way to call a tool directly.
- Arguments must be a JSON object. `null` means no arguments. Anything else is a validation error.
- Output text blocks join with newlines. Images, audio and resources become short placeholders. If there are no content blocks, `structuredContent` renders as JSON.

### Protocol limits

- The plugin speaks only MCP `2026-07-28`, which is stateless. There is no older-version fallback.
- A `tools/call` answered with `resultType: "input_required"` or `resultType: "task"` is not followed up. It fails with `PluginError::NotSupported`.

## Configuration

`McpPlugin` has no config struct. The `McpTransport` you pass to `McpPlugin::connect` picks the connection.

| Variant | Fields | Connection |
|---|---|---|
| `Stdio` | `command`, `args` | Starts the program and speaks MCP over its stdio pipes. |
| `StreamableHttp` | `url` | Connects over Streamable HTTP. |
| `WebSocket` | `url` | Not supported. Fails with `PluginError::NotSupported`. |

`McpPlugin::connect_stdio(command)` is shorthand for `McpPlugin::connect(McpTransport::stdio(command))`, with no arguments.

## Caps and at-cap policy

No cap. The tool list is read once at connect time, and nothing grows with traffic. `McpPlugin::tools()` returns the list sorted by name. There is no usage gauge.

## Errors

A failed tool call becomes `ToolResult` text, so the hooks return no error. `connect` and `call_tool` return these:

| Variant | When |
|---|---|
| `Io` | The server program or the worker thread fails to start. |
| `NotSupported` | You use `McpTransport::WebSocket`, or a tool call answers `input_required` or `task`. |
| `Validation` (schema `tools/call arguments`) | The arguments are neither a JSON object nor `null`. |
| `Internal` | Discovery, `tools/list` or a tool call fails, the worker exits, or the server sends a response the plugin does not recognize. |

## Related pages

- [Plugins](@/plugins/_index.md): the hooks, their order and the error rules.
- [Error types](@/reference/errors.md): `PluginError` and `CucaError`.
- [Point the client at another OpenAI-compatible server](@/guides/other-openai-server.md): retarget the example.
