+++
title = "MCP"
description = "The Model Context Protocol client plugin: transports, the stateless 2026-07-28 protocol, and tool call routing."
template = "page.html"
weight = 1
+++

# MCP

<dl class="page-facts">
<dt>In one line</dt>
<dd>Connects to one Model Context Protocol server, discovers its tools, and executes them as ToolCall to ToolResult exchanges.</dd>
<dt>You need</dt>
<dd>The <code>plugin-mcp</code> feature and a reachable MCP server (a spawned child process or a Streamable HTTP endpoint).</dd>
<dt>Read this if</dt>
<dd>You are registering <code>McpPlugin</code> or choosing an <code>McpTransport</code>.</dd>
</dl>

## Entry types

`McpPlugin`, `McpTransport`.

## `CucaPlugin`

`McpPlugin` implements `CucaPlugin` with the plugin name `"mcp-connector"`.

| Hook | Behavior |
|---|---|
| `on_request` | No-op. Discovered tools are injected into the caller's tool set through `McpPlugin::tools()`, not by rewriting the request. |
| `on_stream_chunk` | Routes `ToolCall` blocks whose name is a discovered tool to a `ToolResult` carrying the call's rendered output, or the error text on failure. Unknown tool names pass through untouched. |
| `on_response_complete` | No-op. |

## Transport

| Variant | Connection |
|---|---|
| `McpTransport::Stdio { command, args }` | Spawns the executable as a child process and speaks MCP over its stdio pipes |
| `McpTransport::StreamableHttp { url }` | Connects over Streamable HTTP, the MCP 2026-07-28 binding |
| `McpTransport::WebSocket { url }` | Not connectable; resolves to `PluginError::NotSupported`, there is no WebSocket client transport |

`McpPlugin::connect_stdio(command)` is shorthand for `McpPlugin::connect(McpTransport::stdio(command))`.

## Protocol

The plugin speaks only the stateless MCP protocol version `2026-07-28`. There is no connection-setup phase and no shared connection state; every request carries its protocol version, client identity, and client capabilities in `_meta`. Connecting probes the server once with `server/discover`, then lists every tool with pagination-aware `tools/list`.

A server answering `tools/call` with `resultType: "input_required"` (multi round trip requests, SEP-2322) or `resultType: "task"` (SEP-2663) is not driven further; both surface as `PluginError::NotSupported` instead of a fabricated result.

## Capacity

The discovered tool map is populated once at connect time from `tools/list`, not on each request, so it carries no traffic-growth cap. `McpPlugin::tools()` returns the full list, sorted by name.
