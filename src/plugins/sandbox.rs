//! Isolated WebAssembly code execution inside a memory-confined wasmtime
//! instance.
//!
//! [`SandboxPlugin`] runs model-generated WebAssembly in a fresh, confined
//! store per call. It replaces the iterative JSON tool-call loop with direct
//! code execution: the model emits a `run_code`/`sandbox_exec` tool call whose
//! arguments carry a base64-encoded `.wasm` module and an input string; the
//! plugin compiles the module, runs it under hard resource limits, and swaps
//! the block for a `ToolResult` carrying the module's collected output.
//!
//! # Guest ABI
//!
//! The guest module must export a linear memory named `memory` and a function
//! `run(ptr: i32, len: i32) -> i32`, and may import
//! `env.write_out(ptr: i32, len: i32)`:
//!
//! * **Input**: the host writes the raw input bytes into the instance memory
//!   at the fixed scratch offset [`INPUT_PTR`] (1024) *before* calling `run`,
//!   then calls `run(INPUT_PTR, input.len())`. The guest reads
//!   `memory[ptr..ptr+len]` for its input. A fixed host-chosen scratch address
//!   (rather than a guest `alloc` export) keeps the guest contract to exactly
//!   two exports plus one import and avoids trusting an untrusted allocator
//!   with pointer-returning calls.
//! * **Output**: the guest calls `write_out(ptr, len)` any number of times;
//!   each call appends `memory[ptr..ptr+len]` to the host's collected stdout.
//! * **Status**: `run` returns `0` on success; any non-zero value is a
//!   guest-reported error, surfaced as a [`PluginError`] by the host.
//!
//! # Resource controls
//!
//! Every call runs in a fresh [`Store`] on a process-wide engine (built once
//! behind a `LazyLock`), so a trapped run can never poison a later one:
//!
//! * **Instructions**: [`Config::consume_fuel`] + [`Store::set_fuel`] give the
//!   guest a hard fuel budget; exhaustion traps the instance.
//! * **Wall-clock time**: [`Config::epoch_interruption`] +
//!   [`Store::set_epoch_deadline`] arm an epoch deadline one tick out; a
//!   background thread sleeps `timeout_ms` and then fires
//!   [`Engine::increment_epoch`], trapping long-running guests at their next
//!   epoch check. The thread is dismissed and joined after every run.
//! * **Memory**: [`StoreLimitsBuilder::memory_size`] caps each linear memory
//!   in bytes; [`StoreLimitsBuilder::trap_on_grow_failure`] turns a denied
//!   `memory.grow` into a hard trap instead of a silent `-1` the guest might
//!   mis-handle.
//!
//! Wasmtime checks epoch deadlines while instantiating as well as while
//! executing guest code. [`run`](SandboxPlugin::run) therefore sets a fresh
//! deadline before instantiation and holds a process-wide lock through ticker
//! join: only one store can hold an armed deadline at any instant, so a ticker
//! can only ever interrupt its own run. The ticker itself starts only after
//! instantiation and input setup, so the timeout still bounds guest execution,
//! not host setup.
//!
//! Hitting any limit returns [`PluginError::Internal`] carrying the trap
//! message (fuel exhaustion and the epoch deadline are named explicitly).
//!
//! # Telemetry note
//!
//! `tracing` is not a dependency of this feature, so the spec's diagnostic
//! event (`execution_time_ms`, `memory_bytes_used`) is exposed through
//! [`SandboxPlugin::last_diagnostic`] rather than emitted as structured
//! telemetry. Emitting it belongs to `plugin-telemetry` when both features
//! are enabled; wiring that up is a known gap, deliberately not papered over
//! here.

use std::sync::mpsc;
use std::sync::{LazyLock, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD;
use wasmtime::{
    Caller, Config, Engine, Extern, Linker, Module, Store, StoreLimits, StoreLimitsBuilder, Trap,
};

use crate::error::PluginError;
use crate::plugin::CucaPlugin;
use crate::request::{UnifiedRequest, UnifiedResponse};
use crate::types::MessageContentBlock;

/// Fixed scratch offset at which the host pre-loads the input bytes.
///
/// The ABI (documented in the module docs) has no guest `alloc` export: the
/// host picks this low, fixed address, which leaves room for the module's
/// static data and needs only a bounds check: no guest cooperation.
const INPUT_PTR: usize = 1024;

/// Serializes the armed-deadline windows of concurrent [`SandboxPlugin::run`]
/// calls.
///
/// The engine's epoch counter is process-global, and `Engine::increment_epoch`
/// traps any store whose deadline has elapsed. Two concurrent runs would
/// therefore share one clock: the first run's ticker could trip the second
/// run's deadline mid-execution. Holding this lock from deadline arming
/// through ticker join guarantees at most one armed deadline exists at any
/// instant, so a ticker can only interrupt its own run. (The lock is taken
/// only for the execute phase, not compilation, so compile parallelism is
/// unaffected.)
static RUN_LOCK: Mutex<()> = Mutex::new(());

struct SandboxStoreData {
    limits: StoreLimits,
    stdout: Vec<u8>,
}

const MAX_OUTPUT_BYTES: usize = 8 * 1024 * 1024;

/// Configuration for the WebAssembly execution sandbox.
pub struct SandboxConfig {
    /// Maximum linear-memory size per instance, in bytes
    /// ([`StoreLimitsBuilder::memory_size`]).
    pub max_memory_bytes: usize,
    /// Fuel budget per call ([`Store::set_fuel`]); the guest traps when it
    /// runs out, bounding instruction count even for infinite loops.
    pub max_instructions: u64,
    /// Wall-clock execution cap in milliseconds, enforced by epoch
    /// interruption; a guest still running after this traps.
    pub timeout_ms: u64,
}

impl Default for SandboxConfig {
    fn default() -> Self {
        Self {
            max_memory_bytes: 64 * 1024 * 1024,
            max_instructions: 1_000_000,
            timeout_ms: 5_000,
        }
    }
}

/// The outcome of a successful [`SandboxPlugin::run`] call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SandboxResult {
    /// Bytes the guest appended via the `write_out` host import, in call order.
    pub stdout: Vec<u8>,
    /// Wall-clock time the guest ran, in milliseconds.
    pub execution_time_ms: u64,
    /// The instance's linear-memory size in bytes after the run (0 when the
    /// module exports no `memory`).
    pub memory_bytes_used: usize,
}

/// The WebAssembly execution sandbox plugin.
///
/// Holds the immutable [`SandboxConfig`] plus the most recent run's
/// diagnostic, shared behind a `Mutex` so the stream hook can be called from
/// any thread (the `CucaPlugin` supertrait requires `Send + Sync`).
pub struct SandboxPlugin {
    config: SandboxConfig,
    /// Most recent successful run's `(execution_time_ms, memory_bytes_used)`.
    last: Mutex<Option<(u64, usize)>>,
}

impl SandboxPlugin {
    /// Create a sandbox with the given resource configuration.
    pub fn new(config: SandboxConfig) -> Self {
        Self {
            config,
            last: Mutex::new(None),
        }
    }

    /// Compile and run `wasm` with `input` in a fresh, confined store.
    ///
    /// `wasm` is the raw module bytes; because the engine is built with the
    /// `wat` feature, WAT text is accepted as well as binary. `input` is the
    /// raw byte payload pre-loaded at [`INPUT_PTR`] per the module ABI. All
    /// three resource limits (fuel, wall-clock timeout, memory cap) are
    /// enforced per call; hitting any of them returns a [`PluginError`]
    /// carrying the trap message. A fresh store per call means a trapped or
    /// limit-hit run leaves no state behind for the next one.
    pub fn run(&self, wasm: &[u8], input: &[u8]) -> Result<SandboxResult, PluginError> {
        let engine = shared_engine().map_err(PluginError::Internal)?;
        let module = Module::new(engine, wasm).map_err(|e| PluginError::Validation {
            schema: "wasm".to_owned(),
            message: format!("module compilation failed: {e}"),
        })?;

        let mut store = Store::new(
            engine,
            SandboxStoreData {
                limits: StoreLimitsBuilder::new()
                    .memory_size(self.config.max_memory_bytes)
                    .trap_on_grow_failure(true)
                    .build(),
                stdout: Vec::new(),
            },
        );
        store.limiter(|data| &mut data.limits);
        store
            .set_fuel(self.config.max_instructions)
            .map_err(|e| PluginError::Internal(format!("failed to set fuel budget: {e}")))?;

        let mut linker = Linker::new(engine);
        linker
            .func_wrap("env", "write_out", write_out)
            .map_err(|e| {
                PluginError::Internal(format!("failed to define write_out import: {e}"))
            })?;

        // Epoch interruption applies during instantiation too. Arm a fresh
        // one-tick deadline while holding the global window lock, preventing a
        // prior run's epoch increment from interrupting this fresh store.
        let _run_guard = RUN_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        store.set_epoch_deadline(1);

        let instance = linker
            .instantiate(&mut store, &module)
            .map_err(|e| PluginError::Internal(format!("instantiation failed: {e}")))?;

        // Pre-load the input at the fixed scratch pointer. A module without an
        // exported memory cannot receive input, which is only an error when
        // input was actually provided (the resource-limit tests run with empty
        // input and no memory at all).
        let memory = instance.get_memory(&mut store, "memory");
        if !input.is_empty() {
            let memory = memory.ok_or_else(|| PluginError::Validation {
                schema: "wasm".to_owned(),
                message: "module has no exported `memory` to receive input".to_owned(),
            })?;
            let capacity = memory.data_size(&store);
            let end = INPUT_PTR
                .checked_add(input.len())
                .ok_or_else(|| PluginError::Internal("input pointer overflow".to_owned()))?;
            if end > capacity {
                return Err(PluginError::Validation {
                    schema: "wasm".to_owned(),
                    message: format!(
                        "input of {} bytes does not fit at scratch offset {INPUT_PTR} \
                         (module memory is {capacity} bytes)",
                        input.len()
                    ),
                });
            }
            memory.data_mut(&mut store)[INPUT_PTR..end].copy_from_slice(input);
        }

        let run = instance
            .get_typed_func::<(i32, i32), i32>(&mut store, "run")
            .map_err(|e| PluginError::Validation {
                schema: "wasm".to_owned(),
                message: format!("module does not export run(ptr, len) -> i32: {e}"),
            })?;

        // Instantiation and input setup completed before the ticker starts, so
        // the timeout below bounds only guest execution. The lock acquired
        // above remains held through ticker join, preventing cross-run epoch
        // interruption.
        let (done_tx, done_rx) = mpsc::channel::<()>();
        let ticker_engine = engine.clone();
        let timeout = self.config.timeout_ms;
        let ticker = thread::spawn(move || {
            if done_rx
                .recv_timeout(Duration::from_millis(timeout))
                .is_err()
            {
                ticker_engine.increment_epoch();
            }
        });

        let start = Instant::now();
        let status = run.call(&mut store, (INPUT_PTR as i32, input.len() as i32));
        let execution_time_ms = start.elapsed().as_millis() as u64;

        let _ = done_tx.send(());
        let _ = ticker.join();

        let status = status.map_err(|e| PluginError::Internal(trap_message(&e)))?;
        if status != 0 {
            return Err(PluginError::Internal(format!(
                "guest run reported status {status}"
            )));
        }

        let stdout = std::mem::take(&mut store.data_mut().stdout);
        let memory_bytes_used = memory.map(|m| m.data_size(&store)).unwrap_or(0);
        if let Ok(mut last) = self.last.lock() {
            *last = Some((execution_time_ms, memory_bytes_used));
        }
        Ok(SandboxResult {
            stdout,
            execution_time_ms,
            memory_bytes_used,
        })
    }

    /// The most recent successful run's `(execution_time_ms,
    /// memory_bytes_used)`.
    ///
    /// Runs that ended in a [`PluginError`] produce no metrics, so they do not
    /// overwrite the last successful diagnostic. Structured emission of this
    /// event is out of scope for this feature (no `tracing` dependency). See the module docs'
    /// telemetry note.
    pub fn last_diagnostic(&self) -> Option<(u64, usize)> {
        *self
            .last
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

/// The process-wide engine, configured once with fuel + epoch interruption.
///
/// Built lazily on first use; the `Result` lets an engine-construction
/// failure surface as a [`PluginError`] from `run` instead of panicking
/// during static initialization.
static ENGINE: LazyLock<Result<Engine, String>> = LazyLock::new(|| {
    let mut config = Config::new();
    config.consume_fuel(true).epoch_interruption(true);
    Engine::new(&config).map_err(|e| e.to_string())
});

/// Resolve the shared engine, propagating a one-time construction failure.
fn shared_engine() -> Result<&'static Engine, String> {
    ENGINE.as_ref().map_err(|e| e.clone())
}

fn write_out(mut caller: Caller<'_, SandboxStoreData>, ptr: i32, len: i32) -> wasmtime::Result<()> {
    let memory = caller
        .get_export("memory")
        .and_then(Extern::into_memory)
        .ok_or_else(|| wasmtime::Error::msg("write_out: module has no exported memory"))?;
    let start =
        usize::try_from(ptr).map_err(|_| wasmtime::Error::msg("write_out: negative pointer"))?;
    let count =
        usize::try_from(len).map_err(|_| wasmtime::Error::msg("write_out: negative length"))?;
    let end = start
        .checked_add(count)
        .ok_or_else(|| wasmtime::Error::msg("write_out: pointer overflow"))?;
    let bytes = memory
        .data(&caller)
        .get(start..end)
        .ok_or_else(|| wasmtime::Error::msg("write_out: memory range out of bounds"))?
        .to_vec();
    let output_end = caller
        .data()
        .stdout
        .len()
        .checked_add(bytes.len())
        .ok_or_else(|| wasmtime::Error::msg("write_out: output size overflow"))?;
    if output_end > MAX_OUTPUT_BYTES {
        return Err(wasmtime::Error::msg("write_out: output limit exceeded"));
    }
    caller.data_mut().stdout.extend_from_slice(&bytes);
    Ok(())
}

/// Render a wasmtime error into a sandbox diagnostic, naming the specific
/// resource limit that was hit when the error is a trap.
fn trap_message(error: &wasmtime::Error) -> String {
    match error.downcast_ref::<Trap>() {
        Some(Trap::OutOfFuel) => "fuel exhausted: instruction budget exceeded".to_owned(),
        Some(Trap::Interrupt) => "epoch deadline reached: execution timed out".to_owned(),
        Some(trap) => format!("wasm trap: {trap}"),
        // Alternate formatting walks the context chain, surfacing the root
        // cause (e.g. the resource limiter's "growing memory" message) past
        // the wasm-backtrace context wasmtime attaches.
        None => format!("sandbox execution failed: {error:#}"),
    }
}

impl CucaPlugin for SandboxPlugin {
    /// Stable plugin name: `"wasm-sandbox"` (spec-fixed).
    fn name(&self) -> &'static str {
        "wasm-sandbox"
    }

    /// No-op: the sandbox rewrites only streamed tool-call blocks.
    fn on_request(&self, _req: &mut UnifiedRequest) -> Result<(), PluginError> {
        Ok(())
    }

    /// Route `run_code`/`sandbox_exec` tool calls.
    ///
    /// `arguments.wasm` is a base64-encoded module and `arguments.input` the
    /// raw UTF-8 input string (absent means empty; any non-string is a
    /// validation error). The module is compiled and executed synchronously
    /// inside the hook, the same pause semantics the MCP connector uses, and
    /// the block is replaced with a `ToolResult` carrying the collected
    /// stdout as lossy UTF-8, or the error text when execution failed.
    /// Malformed payloads (undecodable base64, wrong field types) surface as
    /// [`PluginError::Validation`]; unknown tool names pass through untouched.
    fn on_stream_chunk(&self, chunk: &mut MessageContentBlock) -> Result<(), PluginError> {
        let MessageContentBlock::ToolCall {
            id,
            name,
            arguments,
        } = chunk
        else {
            return Ok(());
        };
        if name != "run_code" && name != "sandbox_exec" {
            return Ok(());
        }
        let wasm = STANDARD
            .decode(
                arguments
                    .get("wasm")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| PluginError::Validation {
                        schema: "run_code.wasm".to_owned(),
                        message: "`wasm` must be a base64-encoded string".to_owned(),
                    })?,
            )
            .map_err(|e| PluginError::Validation {
                schema: "run_code.wasm".to_owned(),
                message: format!("`wasm` is not valid base64: {e}"),
            })?;
        let input: &[u8] = match arguments.get("input") {
            None => &[],
            Some(value) => value
                .as_str()
                .ok_or_else(|| PluginError::Validation {
                    schema: "run_code.input".to_owned(),
                    message: "`input` must be a string".to_owned(),
                })?
                .as_bytes(),
        };
        let output = match self.run(&wasm, input) {
            Ok(result) => String::from_utf8_lossy(&result.stdout).into_owned(),
            Err(err) => err.to_string(),
        };
        // `id` is moved, not cloned: `*chunk` is overwritten by this very
        // assignment, so the borrowed block's id is dead.
        *chunk = MessageContentBlock::ToolResult {
            tool_call_id: std::mem::take(id),
            output,
        };
        Ok(())
    }

    /// No-op: the sandbox keeps no per-response state.
    fn on_response_complete(&self, _res: &UnifiedResponse) -> Result<(), PluginError> {
        Ok(())
    }
}

#[cfg(all(test, feature = "plugin-sandbox"))]
mod tests {
    use super::*;

    /// Guest that writes a fixed banner via `write_out`; run must ignore the
    /// pointer args (they address the empty input scratch region).
    const HELLO_WAT: &str = r#"
        (module
          (import "env" "write_out" (func $write_out (param i32 i32)))
          (memory (export "memory") 1)
          (data (i32.const 0) "hello from wasm")
          (func (export "run") (param $ptr i32) (param $len i32) (result i32)
            (call $write_out (i32.const 0) (i32.const 15))
            (i32.const 0)))
    "#;

    /// Guest that echoes its input region back through `write_out`: the ABI
    /// contract under test in one module.
    const ECHO_WAT: &str = r#"
        (module
          (import "env" "write_out" (func $write_out (param i32 i32)))
          (memory (export "memory") 1)
          (func (export "run") (param $ptr i32) (param $len i32) (result i32)
            (call $write_out (local.get $ptr) (local.get $len))
            (i32.const 0)))
    "#;

    /// Infinite loop: burns fuel forever, so a finite budget must trap it.
    const INFINITE_LOOP_WAT: &str = r#"
        (module
          (func (export "run") (param $ptr i32) (param $len i32) (result i32)
            (loop $l (br $l))
            (i32.const 0)))
    "#;

    /// Tight busy loop: must be cut off by the epoch-interruption timeout.
    const BUSY_LOOP_WAT: &str = r#"
        (module
          (func (export "run") (param $ptr i32) (param $len i32) (result i32)
            (local $i i32)
            (loop $l
              (local.set $i (i32.add (local.get $i) (i32.const 1)))
              (br $l))
            (i32.const 0)))
    "#;

    /// Declares a 6.4 MiB maximum memory but must be stopped at the 1 MiB
    /// store cap by `memory.grow`.
    const MEMORY_GROW_WAT: &str = r#"
        (module
          (memory (export "memory") 1 100)
          (func (export "run") (param $ptr i32) (param $len i32) (result i32)
            (drop (memory.grow (i32.const 20)))
            (i32.const 0)))
    "#;

    #[test]
    fn run_executes_hello_module() {
        let plugin = SandboxPlugin::new(SandboxConfig::default());
        let result = plugin.run(HELLO_WAT.as_bytes(), &[]).unwrap();
        // Wall-clock elapsed for a trivial module is normally 0 ms; the bound
        // just proves the timer ran rather than being vacuously unset.
        assert!(result.execution_time_ms < 10_000);
    }

    #[test]
    fn input_is_written_at_scratch_ptr() {
        let plugin = SandboxPlugin::new(SandboxConfig::default());
        let result = plugin.run(ECHO_WAT.as_bytes(), b"hi there").unwrap();
        assert_eq!(result.stdout, b"hi there");
    }

    #[test]
    fn fuel_exhaustion_traps() {
        let plugin = SandboxPlugin::new(SandboxConfig {
            max_instructions: 100,
            ..SandboxConfig::default()
        });
        let err = plugin.run(INFINITE_LOOP_WAT.as_bytes(), &[]).unwrap_err();
        match err {
            PluginError::Internal(message) => assert!(message.contains("fuel")),
            other => panic!("expected Internal fuel error, got {other:?}"),
        }
    }

    #[test]
    fn memory_limit_traps() {
        let plugin = SandboxPlugin::new(SandboxConfig {
            max_memory_bytes: 1024 * 1024,
            ..SandboxConfig::default()
        });
        let err = plugin.run(MEMORY_GROW_WAT.as_bytes(), &[]).unwrap_err();
        match err {
            PluginError::Internal(message) => {
                assert!(
                    message.contains("growing memory"),
                    "unexpected message: {message}"
                )
            }
            other => panic!("expected Internal memory error, got {other:?}"),
        }
    }

    #[test]
    fn timeout_interrupts() {
        let plugin = SandboxPlugin::new(SandboxConfig {
            timeout_ms: 1,
            // A large fuel budget ensures the 1 ms epoch deadline fires first;
            // with the default 1_000_000 the loop would exhaust fuel first.
            max_instructions: 1_000_000_000,
            ..SandboxConfig::default()
        });
        let err = plugin.run(BUSY_LOOP_WAT.as_bytes(), &[]).unwrap_err();
        match err {
            PluginError::Internal(message) => {
                assert!(
                    message.contains("timed out"),
                    "unexpected message: {message}"
                )
            }
            other => panic!("expected Internal timeout error, got {other:?}"),
        }
    }

    #[test]
    fn on_stream_chunk_routes_run_code_and_passes_unknown_through() {
        let plugin = SandboxPlugin::new(SandboxConfig::default());
        let wasm = STANDARD.encode(ECHO_WAT.as_bytes());
        let mut chunk = MessageContentBlock::ToolCall {
            id: "call-1".to_owned(),
            name: "run_code".to_owned(),
            arguments: serde_json::json!({ "wasm": wasm, "input": "hello" }),
        };
        plugin.on_stream_chunk(&mut chunk).unwrap();
        assert_eq!(
            chunk,
            MessageContentBlock::ToolResult {
                tool_call_id: "call-1".to_owned(),
                output: "hello".to_owned(),
            }
        );

        let mut other = MessageContentBlock::ToolCall {
            id: "call-2".to_owned(),
            name: "some_other_tool".to_owned(),
            arguments: serde_json::json!({}),
        };
        plugin.on_stream_chunk(&mut other).unwrap();
        assert!(
            matches!(other, MessageContentBlock::ToolCall { ref name, .. } if name == "some_other_tool")
        );
    }

    #[test]
    fn plugin_name_is_wasm_sandbox() {
        assert_eq!(
            SandboxPlugin::new(SandboxConfig::default()).name(),
            "wasm-sandbox"
        );
    }

    #[test]
    fn plugin_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<SandboxPlugin>();
    }
}
