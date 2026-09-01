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

`SandboxPlugin` runs model-generated WebAssembly in a fresh, memory-confined wasmtime store per call: the guest exports `memory` and `run(ptr, len) -> i32`, may import `env.write_out`, and the plugin hands its collected output back as a `ToolResult` for `run_code`/`sandbox_exec` tool calls. Every call is bounded by `max_memory_bytes`, a fuel-based `max_instructions`, and a wall-clock `timeout_ms` enforced by epoch interruption. Reach for it to let a model execute short sandboxed logic instead of looping through JSON tool calls.

```rust,name=Run a guest module that echoes its input
use std::sync::Arc;

use cuca::plugin::CucaPlugin;
use cuca::types::ProviderEndpoint;
use cuca::{CucaClient, SandboxConfig, SandboxPlugin};

const ECHO_WAT: &str = r#"
(module
  (import "env" "write_out" (func $write_out (param i32 i32)))
  (memory (export "memory") 1)
  (func (export "run") (param $ptr i32) (param $len i32) (result i32)
    (call $write_out (local.get $ptr) (local.get $len))
    (i32.const 0)))
"#;

let sandbox = Arc::new(SandboxPlugin::new(SandboxConfig::default()));

let client = CucaClient::builder()
    .with_provider(ProviderEndpoint::LlamaCpp)
    .with_base_url("http://127.0.0.1:1234/v1")
    .register_plugin(Arc::clone(&sandbox) as Arc<dyn CucaPlugin>)
    .build()?;

let result = sandbox.run(ECHO_WAT.as_bytes(), b"hello sandbox")?;
```

```text,name=The guest wrote its input straight back out
result.stdout               b"hello sandbox"
result.memory_bytes_used    65536
```

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
