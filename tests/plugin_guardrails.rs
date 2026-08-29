//! Integration tests for the JSON Schema guardrail plugin (`plugin-guardrails`).
//!
//! The deterministic tests drive [`CucaPlugin::on_stream_chunk`] with crafted
//! blocks to prove validation/re-injection semantics; the live test registers
//! the plugin on a llama.cpp client and verifies a real request still streams.
#![cfg(all(feature = "provider-llamacpp", feature = "plugin-guardrails"))]

mod common;

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use cuca::JsonGuardrailPlugin;
use cuca::PluginError;
use cuca::plugin::CucaPlugin;
use cuca::types::MessageContentBlock;
use serde_json::json;

/// Schema map: a `make_reservation` tool requiring a string `date`, plus the
/// reserved `"response"` key requiring a string `status`.
fn schemas() -> HashMap<String, serde_json::Value> {
    HashMap::from([
        (
            "make_reservation".to_string(),
            json!({
                "type": "object",
                "required": ["date"],
                "properties": { "date": { "type": "string" } }
            }),
        ),
        (
            "response".to_string(),
            json!({
                "type": "object",
                "required": ["status"],
                "properties": { "status": { "type": "string" } }
            }),
        ),
    ])
}

fn plugin() -> JsonGuardrailPlugin {
    JsonGuardrailPlugin::with_schemas(schemas(), 3).expect("schemas must compile")
}

#[test]
fn invalid_tool_call_is_replaced_with_tool_result_error() {
    let plugin = plugin();
    // `date` is required but missing from the arguments.
    let mut block = MessageContentBlock::ToolCall {
        id: "call-1".to_string(),
        name: "make_reservation".to_string(),
        arguments: json!({}),
    };
    plugin.on_stream_chunk(&mut block).unwrap();
    match block {
        MessageContentBlock::ToolResult {
            tool_call_id,
            output,
        } => {
            assert_eq!(tool_call_id, "call-1");
            assert!(
                output.contains("schema_validation_failed"),
                "expected schema_validation_failed in output, got: {output}"
            );
            assert!(output.contains("make_reservation"), "output: {output}");
        }
        other => panic!("expected ToolResult, got {other:?}"),
    }
}

#[test]
fn valid_tool_call_passes_through_untouched() {
    let plugin = plugin();
    let original = MessageContentBlock::ToolCall {
        id: "call-2".to_string(),
        name: "make_reservation".to_string(),
        arguments: json!({ "date": "2026-09-01" }),
    };
    let mut block = original.clone();
    plugin.on_stream_chunk(&mut block).unwrap();
    assert_eq!(
        block, original,
        "a schema-conforming ToolCall must pass through byte-for-byte unchanged"
    );
}

#[test]
fn tool_without_registered_schema_passes_through() {
    let plugin = plugin();
    let original = MessageContentBlock::ToolCall {
        id: "call-3".to_string(),
        name: "some_unknown_tool".to_string(),
        arguments: json!({ "anything": true }),
    };
    let mut block = original.clone();
    plugin.on_stream_chunk(&mut block).unwrap();
    assert_eq!(
        block, original,
        "an unregistered tool must pass through byte-for-byte unchanged"
    );
}

#[test]
fn invalid_response_json_text_is_replaced_with_error() {
    let plugin = plugin();
    // The reserved "response" schema requires a string `status`; this object
    // violates it and so must be replaced with a Text carrying the error.
    let mut block = MessageContentBlock::Text("{\"foo\": 1}".to_string());
    plugin.on_stream_chunk(&mut block).unwrap();
    match block {
        MessageContentBlock::Text(text) => {
            assert!(
                text.contains("schema_validation_failed"),
                "expected schema_validation_failed in text, got: {text}"
            );
            assert!(text.contains("response"), "text: {text}");
        }
        other => panic!("expected Text, got {other:?}"),
    }
}

#[test]
fn schema_conforming_response_text_passes_through() {
    let plugin = plugin();
    let mut block = MessageContentBlock::Text("{\"status\": \"confirmed\"}".to_string());
    plugin.on_stream_chunk(&mut block).unwrap();
    assert_eq!(
        block,
        MessageContentBlock::Text("{\"status\": \"confirmed\"}".to_string()),
        "a schema-conforming response object must pass through unchanged"
    );
}

/// Records every chunk it observes. Registered *after* the guardrail plugin
/// so `on_stream_chunk` hooks run in registration order (see
/// `src/client.rs`'s pipeline docs): every recorded chunk has already passed
/// through guardrails.
#[derive(Default)]
struct RecordingPlugin {
    chunks: Mutex<Vec<MessageContentBlock>>,
}

impl CucaPlugin for RecordingPlugin {
    fn name(&self) -> &'static str {
        "recording"
    }

    fn on_stream_chunk(&self, chunk: &mut MessageContentBlock) -> Result<(), PluginError> {
        self.chunks.lock().unwrap().push(chunk.clone());
        Ok(())
    }
}

#[tokio::test]
async fn live_request_streams_text_with_guardrails_registered() {
    if let Err(reason) = common::require_live_server() {
        eprintln!("SKIP: llama.cpp not reachable: {reason}");
        return;
    }
    let recorder = Arc::new(RecordingPlugin::default());
    let client = common::client_with_plugins(vec![
        Arc::new(plugin()) as Arc<dyn CucaPlugin>,
        recorder.clone() as Arc<dyn CucaPlugin>,
    ]);
    let request = common::live_request("Reply with the single word: ok", &common::live_model());
    let stream = client
        .generate_stream(request)
        .await
        .expect("generate_stream must start");
    let blocks = common::drain_timeout(stream, 60).await;
    assert!(
        blocks
            .iter()
            .any(|b| matches!(b, MessageContentBlock::Text(_))),
        "expected at least one Text block, got {blocks:?}"
    );

    // The recorder is registered after guardrails, so every chunk it saw has
    // already passed through `JsonGuardrailPlugin::on_stream_chunk`: guardrails
    // ran over every block, and the delivered stream is exactly what the
    // recorder observed, in the same order.
    let recorded = recorder.chunks.lock().unwrap();
    assert!(
        !recorded.is_empty(),
        "the recording plugin (registered after guardrails) must observe at least one stream chunk"
    );
    assert_eq!(
        *recorded, blocks,
        "the recorder's post-guardrails view must match the delivered stream exactly, in order"
    );
    // The reply text ("ok") is plain prose, not a `{`-prefixed JSON object, so
    // guardrails' response-schema check never applies to it (see
    // `src/plugins/guardrails.rs::on_stream_chunk`): schema-conforming text
    // must reach the recorder unmodified, never rewritten into a
    // `schema_validation_failed` error payload.
    for chunk in recorded.iter() {
        if let MessageContentBlock::Text(text) = chunk {
            assert!(
                !text.contains("schema_validation_failed"),
                "guardrails must not have rewritten a passthrough reply into a validation error: {text}"
            );
        }
    }
}
