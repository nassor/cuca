+++
title = "Sandbox"
description = "Run model-written WebAssembly in a locked-down sandbox with memory, instruction and time limits."
template = "page.html"
weight = 2
+++

# Sandbox

<dl class="page-facts">
<dt>In one line</dt>
<dd><code>SandboxPlugin</code> runs a WebAssembly module the model wrote in a fresh, limited sandbox, and returns its output as a <code>ToolResult</code>.</dd>
<dt>You need</dt>
<dd>The <code>plugin-sandbox</code> feature. Nothing else runs; the engine is built in your process.</dd>
<dt>Read this if</dt>
<dd>You want the model to run code safely, or you are writing a guest module or setting its limits.</dd>
</dl>

WebAssembly (Wasm) is a compact code format that runs in an isolated virtual machine. The plugin uses the `wasmtime` engine. Each call gets its own memory, a fuel budget (a cap on instructions) and a time limit. Code that breaks a limit is stopped. It cannot touch your files or network.

## Register it

You can run a module directly, without a model:

```rust,name=Run a module and read the gauge
use cuca::{SandboxConfig, SandboxPlugin};

let sandbox = SandboxPlugin::new(SandboxConfig::default());
let result = sandbox.run(module_wat.as_bytes(), b"cuca")?;
println!("{:?}", String::from_utf8_lossy(&result.stdout));
println!("{:?}", sandbox.last_diagnostic());
```

To let the model call it, register the plugin with `CucaClientBuilder::register_plugin` and add a `run_code` `ToolDefinition` to your request. The plugin does not declare the tool for you. When the model calls `run_code` or `sandbox_exec` with a module, the stream carries a `ToolResult` with the output. A guest that runs out of fuel or time returns an error text instead.

## Run the example

`examples/sandbox.rs` publishes a `reverse` module and shows a guest that never returns being stopped. It needs a `llama-server` on port 1234 with a tool-calling model loaded. Put `--base-url` and `--model` after `--` to use another OpenAI-compatible server.

```bash,name=Runs the same on all three platforms
cargo run --example sandbox --features "provider-llamacpp plugin-sandbox"
```

```text,name=Expected output
Limits per call: 67108864 bytes of memory, 1000000 instructions of fuel, 5000 ms

Turn 1: the model calls the tool, wasmtime runs the guest
  guest wrote: "acuc"
  ran in 0 ms using 65536 bytes of linear memory
  thinking blocks: 59

Turn 2: the same question, with the guest's output in the prompt
  reply: Reversing the word "cuca" produces **acuc**.
  thinking blocks: 47

A guest that never returns
  internal plugin error: fuel exhausted: instruction budget exceeded
```

The reply wording, the run time and the thinking block counts depend on the model and the machine.

## Feature and entry type

| Item | Value |
|---|---|
| Feature flag | `plugin-sandbox` |
| Plugin name | `"wasm-sandbox"` |
| Entry types | `SandboxPlugin`, `SandboxConfig`, `SandboxResult` |
| Attachment | `CucaClientBuilder::register_plugin` |

## Hooks

`SandboxPlugin` acts on `on_stream_chunk` only.

| Hook | Behavior |
|---|---|
| `on_request` | Does nothing. |
| `on_stream_chunk` | Replaces a `run_code` or `sandbox_exec` `ToolCall` with a `ToolResult`. Other blocks and tool names pass through. |
| `on_response_complete` | Does nothing. |

Tool arguments:

- `arguments.wasm` is the module, base64 encoded. WAT text (the readable Wasm format) is accepted too.
- `arguments.input` is the input string. Missing means empty.

The `ToolResult` holds the guest's output as text, or the error text if the run failed. The stream pauses while the guest runs.

`SandboxPlugin::run(wasm, input)` is the direct entry point. It returns a `SandboxResult` with `stdout`, `execution_time_ms` and `memory_bytes_used`.

### Guest ABI

The guest is the module the model writes. It must follow this contract:

- Export a linear memory named `memory`.
- Export a function `run(ptr: i32, len: i32) -> i32`.
- Optionally import `env.write_out(ptr: i32, len: i32)`. Each call appends `memory[ptr..ptr+len]` to the output.
- The host writes the input bytes at offset 1024, then calls `run(1024, input.len())`.
- `run` returns `0` for success. Any other value is a guest error.
- A module with no `memory` export can run only with empty input.

Every call runs in a fresh store, so a failed run leaves nothing behind. The time limit starts after setup, so `timeout_ms` covers guest execution only.

## Configuration

`SandboxConfig` has three keys. `SandboxPlugin::new(config)` takes it.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `max_memory_bytes` | `usize` | 67108864 (64 MiB) | Memory limit per call. A denied `memory.grow` stops the call. |
| `max_instructions` | `u64` | 1000000 | Fuel per call. The call stops when it runs out. |
| `timeout_ms` | `u64` | 5000 | Wall-clock limit per call. |

## Caps and at-cap policy

Every call is bounded, and nothing builds up between calls.

| Limit | Value | At the limit |
|---|---|---|
| Memory | `max_memory_bytes`, 64 MiB | The call stops. |
| Instructions | `max_instructions`, 1000000 | The call stops with `fuel exhausted: instruction budget exceeded`. |
| Time | `timeout_ms`, 5000 | The call stops with `epoch deadline reached: execution timed out`. |
| Guest output | 8388608 bytes (8 MiB), fixed | `write_out` stops the call with `write_out: output limit exceeded`. |

A stopped call returns `PluginError::Internal` from `run`, and the hook turns it into `ToolResult` text.

Usage gauge: `SandboxPlugin::last_diagnostic()` returns `(execution_time_ms, memory_bytes_used)` of the last successful run. A failed run does not change it.

## Errors

`run` returns these. The hook turns any `run` failure into `ToolResult` text and returns `Ok(())`.

| Variant | When |
|---|---|
| `Validation` (schema `wasm`) | The module does not compile, has no `memory` export while input is not empty, the input does not fit at offset 1024, or the module has no `run(ptr, len) -> i32` export. |
| `Internal` | Setup or instantiation fails, a limit stops the call, or `run` returns a non-zero status (`guest run reported status N`). |

The hook itself returns `PluginError::Validation` with schema `run_code.wasm` or `run_code.input` when `arguments.wasm` is missing, is not a string or is not valid base64, or when `arguments.input` is not a string.

## Related pages

- [Plugins](@/plugins/_index.md): the hooks, their order and the error rules.
- [Error types](@/reference/errors.md): `PluginError` and `CucaError`.
- [Write a custom plugin](@/guides/custom-plugin.md): the hook contract the sandbox follows.
