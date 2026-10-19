+++
title = "Sandbox"
description = "The WebAssembly code execution plugin: the guest ABI, resource limits, and the run_code and sandbox_exec tools."
template = "page.html"
weight = 2
+++

# Sandbox

<dl class="page-facts">
<dt>In one line</dt>
<dd><code>SandboxPlugin</code> runs a model-generated WebAssembly module in a fresh, memory-confined wasmtime store and returns its collected output as a <code>ToolResult</code>.</dd>
<dt>You need</dt>
<dd>The <code>plugin-sandbox</code> feature.</dd>
<dt>Read this if</dt>
<dd>You are registering <code>SandboxPlugin</code>, writing a guest module, or sizing its resource limits.</dd>
</dl>

## Feature and entry type

| Item | Value |
|---|---|
| Feature flag | `plugin-sandbox` (enables `dep:wasmtime`, `dep:base64` and `wasmtime/wat`) |
| Plugin name | `"wasm-sandbox"` |
| Entry types | `SandboxPlugin`, `SandboxConfig`, `SandboxResult` |
| Attachment | `CucaClientBuilder::register_plugin` |
| Needs | Nothing running; the engine is built in process on first use |

## Hooks

`SandboxPlugin` acts on `on_stream_chunk`. It does not override `execute_local_tool`.

| Hook | Behavior |
|---|---|
| `on_request` | No-op. |
| `on_stream_chunk` | Replaces a `run_code` or `sandbox_exec` `ToolCall` with a `ToolResult`. Other blocks and other tool names pass through. |
| `on_response_complete` | No-op. |

- `arguments.wasm` is the module, base64 encoded. Because the engine enables wasmtime's `wat` feature, WAT text is accepted as well as binary.
- `arguments.input` is the raw UTF-8 input string. Absent means empty.
- The `ToolResult` carries the guest's collected stdout as lossy UTF-8, or the error text when execution fails.
- The module compiles and runs synchronously inside the hook, so the stream pauses while the guest runs.
- `SandboxPlugin::run(wasm, input)` is the direct entry point. It returns a `SandboxResult` with `stdout`, `execution_time_ms` and `memory_bytes_used`.

### Guest ABI

- The guest exports a linear memory named `memory` and a function `run(ptr: i32, len: i32) -> i32`.
- The guest may import `env.write_out(ptr: i32, len: i32)`.
- The host writes the raw input bytes into instance memory at the fixed scratch offset 1024, then calls `run(1024, input.len())`.
- Each `write_out` call appends `memory[ptr..ptr+len]` to the collected stdout.
- `run` returns `0` on success. Any non-zero value is a guest-reported error.
- A module with no exported `memory` can run only with empty input.

### Isolation

- Every call runs in a fresh store on a process-wide engine, so a trapped run leaves no state for the next one.
- A process-wide lock serializes the armed-deadline window of concurrent calls, so one call's timer cannot interrupt another call.
- The timer starts after instantiation and input setup, so `timeout_ms` bounds guest execution only.

## Configuration

`SandboxConfig` has three fields. `SandboxPlugin::new(config)` takes it.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `max_memory_bytes` | `usize` | 67108864 (64 MiB) | Linear memory cap per instance; a denied `memory.grow` traps. |
| `max_instructions` | `u64` | 1000000 | Fuel budget per call; the guest traps when it runs out. |
| `timeout_ms` | `u64` | 5000 | Wall-clock cap, enforced by epoch interruption. |

## Caps and at-cap policy

Every call is bounded. Nothing accumulates across calls.

| Bound | Value | At the bound |
|---|---|---|
| Memory | `max_memory_bytes`, 64 MiB | The call traps. |
| Instructions | `max_instructions`, 1000000 | The call traps with `fuel exhausted: instruction budget exceeded`. |
| Wall clock | `timeout_ms`, 5000 | The call traps with `epoch deadline reached: execution timed out`. |
| Guest output | 8388608 bytes (8 MiB), not configurable | `write_out` traps with `write_out: output limit exceeded`. |

- A trapped call returns `PluginError::Internal` from `run`; the hook turns it into `ToolResult` text.
- Usage gauge: `SandboxPlugin::last_diagnostic()` returns `(execution_time_ms, memory_bytes_used)` of the most recent successful run. A failed run does not overwrite it.
- `plugin-sandbox` does not emit the diagnostic as telemetry; read it through `last_diagnostic()`.

## Errors

`run` returns these variants. The hook converts any `run` failure into `ToolResult` text and returns `Ok(())`.

| Variant | When |
|---|---|
| `Validation` (schema `wasm`) | The module fails to compile, the module has no `memory` export while input is non-empty, the input does not fit at offset 1024, or the module does not export `run(ptr, len) -> i32`. |
| `Internal` | The engine or fuel setup fails, instantiation fails, a limit traps, or `run` returns a non-zero status (`guest run reported status N`). |

The hook returns `PluginError::Validation` itself, with schema `run_code.wasm` or `run_code.input`, when `arguments.wasm` is missing, is not a string or is not valid base64, or when `arguments.input` is not a string.

## Example

`examples/sandbox.rs` publishes a `reverse` module. A caller-side plugin binds the module bytes into the model's call, wasmtime runs the guest, and a final line shows a guest that never returns being stopped by the fuel budget.

```rust,name=Run a module directly and read the gauge
use cuca::{SandboxConfig, SandboxPlugin};

let sandbox = SandboxPlugin::new(SandboxConfig::default());
let result = sandbox.run(module_wat.as_bytes(), b"cuca")?;
println!("{:?}", String::from_utf8_lossy(&result.stdout));
println!("{:?}", sandbox.last_diagnostic());
```

The example needs a `llama-server` on port 1234 with the demo model loaded. `CUCA_BASE_URL` and `CUCA_MODEL` retarget it at any OpenAI-compatible server.

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

The reply wording, the execution time and the thinking-block counts depend on the model and the machine. The output above came from `google/gemma-4-12b-qat`.

## Related pages

- [Plugins](@/plugins/_index.md): hook order, error handling and the plugin tier rules.
- [Error types](@/reference/errors.md): `PluginError` and `CucaError`.
- [Write a custom plugin](@/guides/custom-plugin.md): the hook contract the sandbox follows.
