//! Integration tests for the WebAssembly sandbox plugin (`plugin-sandbox`).
//!
//! The deterministic tests run the guest ABI through [`SandboxPlugin::run`]
//! (module exports `memory` + `run(ptr, len) -> i32`, imports
//! `env.write_out(ptr, len)`; the host pre-loads input at scratch offset 1024)
//! and route a `run_code` tool call through [`CucaPlugin::on_stream_chunk`];
//! the live test registers the plugin on a llama.cpp client and verifies a
//! real request still streams.
//!
//! `run` accepts WAT text as well as binary (the engine is built with the
//! `wat` feature), so the tests pass the WAT sources directly and only base64
//! encode them for the `on_stream_chunk` path.
#![cfg(all(feature = "provider-llamacpp", feature = "plugin-sandbox"))]

mod common;

use std::sync::Arc;

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD;
use cuca::plugin::CucaPlugin;
use cuca::types::MessageContentBlock;
use cuca::{PluginError, SandboxConfig, SandboxPlugin};
use serde_json::json;

/// Echoes its input region (host-loaded at scratch offset 1024) back through
/// `write_out` and returns 0.
const ECHO_WAT: &str = r#"
    (module
      (import "env" "write_out" (func $write_out (param i32 i32)))
      (memory (export "memory") 1)
      (func (export "run") (param $ptr i32) (param $len i32) (result i32)
        (call $write_out (local.get $ptr) (local.get $len))
        (i32.const 0)))
"#;

/// Infinite loop: burns fuel forever, so a finite fuel budget must trap it.
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

#[test]
fn run_echoes_input_and_reports_result() {
    let plugin = SandboxPlugin::new(SandboxConfig::default());
    let result = plugin
        .run(ECHO_WAT.as_bytes(), b"hello sandbox")
        .expect("echo module must run");
    assert_eq!(result.stdout, b"hello sandbox");
    assert!(
        result.memory_bytes_used > 0,
        "the module exports a linear memory"
    );
}

#[test]
fn fuel_exhaustion_is_a_plugin_error() {
    let plugin = SandboxPlugin::new(SandboxConfig {
        max_instructions: 100,
        ..SandboxConfig::default()
    });
    let err = plugin
        .run(INFINITE_LOOP_WAT.as_bytes(), &[])
        .expect_err("an infinite loop with a tiny fuel budget must trap");
    match err {
        PluginError::Internal(message) => assert!(message.contains("fuel"), "message: {message}"),
        other => panic!("expected Internal fuel error, got {other:?}"),
    }
}

#[test]
fn timeout_traps_a_busy_loop() {
    let plugin = SandboxPlugin::new(SandboxConfig {
        timeout_ms: 1,
        // A huge fuel budget ensures the 1 ms epoch deadline fires first.
        max_instructions: 1_000_000_000,
        ..SandboxConfig::default()
    });
    let err = plugin
        .run(BUSY_LOOP_WAT.as_bytes(), &[])
        .expect_err("a busy loop past the deadline must trap");
    match err {
        PluginError::Internal(message) => {
            assert!(message.contains("timed out"), "message: {message}")
        }
        other => panic!("expected Internal timeout error, got {other:?}"),
    }
}

#[test]
fn on_stream_chunk_routes_run_code_to_a_tool_result() {
    let plugin = SandboxPlugin::new(SandboxConfig::default());
    let wasm = STANDARD.encode(ECHO_WAT.as_bytes());
    let mut block = MessageContentBlock::ToolCall {
        id: "call-1".to_string(),
        name: "run_code".to_string(),
        arguments: json!({ "wasm": wasm, "input": "hello" }),
    };
    plugin
        .on_stream_chunk(&mut block)
        .expect("on_stream_chunk must return Ok(())");
    assert_eq!(
        block,
        MessageContentBlock::ToolResult {
            tool_call_id: "call-1".to_string(),
            output: "hello".to_string(),
        }
    );
}

/// The live round trip must run real wasm, not merely coexist with the plugin.
/// An injector registered *before* the plugin rewrites the first live model
/// chunk into a `run_code` call, so wasmtime compiles and executes the echo
/// module mid-stream and the consumer receives its output.
#[tokio::test]
async fn live_stream_chunk_runs_real_wasm_for_an_injected_run_code_call() {
    if let Err(reason) = common::require_live_server() {
        eprintln!("SKIP: llama.cpp not reachable: {reason}");
        return;
    }
    let payload = "cuca-live-wasm";
    let injector = Arc::new(common::ToolCallInjector::new(
        "live-wasm-1",
        "run_code",
        json!({ "wasm": STANDARD.encode(ECHO_WAT.as_bytes()), "input": payload }),
    ));
    let client = common::client_with_plugins(vec![
        injector.clone() as Arc<dyn CucaPlugin>,
        Arc::new(SandboxPlugin::new(SandboxConfig::default())) as Arc<dyn CucaPlugin>,
    ]);
    let request = common::live_request("Reply with the single word: ok", &common::live_model());
    let stream = client
        .generate_stream(request)
        .await
        .expect("generate_stream must start");
    let blocks = common::drain_timeout(stream, 60).await;

    assert!(
        injector.injected(),
        "the live turn produced no model chunk to convert, so nothing was \
         exercised; got {blocks:?}"
    );
    let output = common::tool_result_output(&blocks, injector.call_id()).unwrap_or_else(|| {
        panic!("the sandbox must replace the injected call with a ToolResult, got {blocks:?}")
    });
    assert_eq!(
        output, payload,
        "the guest module echoes its host-loaded input, so the delivered output \
         is proof the wasm actually ran"
    );
    assert!(
        !blocks
            .iter()
            .any(|b| matches!(b, MessageContentBlock::ToolCall { name, .. } if name == "run_code")),
        "the plugin must consume the run_code call, not pass it through: {blocks:?}"
    );
}
