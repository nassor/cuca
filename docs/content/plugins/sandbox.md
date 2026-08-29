+++
title = "Sandbox"
description = "The WebAssembly code execution plugin: the guest ABI, resource limits, and the run_code and sandbox_exec tools."
template = "page.html"
weight = 2
+++

# Sandbox

<dl class="page-facts">
<dt>In one line</dt>
<dd>Runs model-generated WebAssembly modules in a confined wasmtime instance and returns the collected output as a ToolResult.</dd>
<dt>You need</dt>
<dd>The <code>plugin-sandbox</code> feature.</dd>
<dt>Read this if</dt>
<dd>You are registering <code>SandboxPlugin</code>, writing a guest module, or sizing its resource limits.</dd>
</dl>

## Entry types

`SandboxPlugin`, `SandboxConfig`, `SandboxResult`.

## `CucaPlugin`

`SandboxPlugin` implements `CucaPlugin` with the plugin name `"wasm-sandbox"`.

| Hook | Behavior |
|---|---|
| `on_request` | No-op. |
| `on_stream_chunk` | Routes `run_code` and `sandbox_exec` `ToolCall` blocks to a `ToolResult` carrying the guest's collected stdout, or the error text on failure. Unknown tool names pass through untouched. |
| `on_response_complete` | No-op. |

## Config

`SandboxConfig` defaults:

| Field | Default | Meaning |
|---|---|---|
| `max_memory_bytes` | 67108864 (64 MiB) | Linear memory cap per instance |
| `max_instructions` | 1000000 | Fuel budget per call; the guest traps on exhaustion |
| `timeout_ms` | 5000 | Wall-clock cap, enforced by epoch interruption |

## Guest ABI

- The guest module exports a linear memory named `memory` and a function `run(ptr: i32, len: i32) -> i32`.
- The guest may import `env.write_out(ptr: i32, len: i32)`.
- The host writes the raw input bytes into instance memory at a fixed scratch offset, 1024, before calling `run(1024, input.len())`.
- The guest calls `write_out` any number of times; each call appends `memory[ptr..ptr+len]` to the host's collected stdout.
- `run` returns `0` on success; any non-zero value is a guest-reported error.

## Capacity

Every call runs in a fresh store on a process-wide engine, so no state carries between calls. The one bound that spans a single call: guest output through `write_out` is capped at 8388608 bytes (8 MiB). At that cap, the call fails with `PluginError::Internal` carrying the trap message `write_out: output limit exceeded`. The usage gauge is `SandboxPlugin::last_diagnostic()`, which returns the `(execution_time_ms, memory_bytes_used)` of the most recent successful run.
