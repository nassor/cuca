//! Integration tests for the skills plugin (`plugin-skills`).
//!
//! The deterministic tests drive the [`CucaPlugin`] hooks directly with crafted
//! requests and tool calls; the live test verifies the injected catalog does
//! not break a real request through a llama.cpp client.
#![cfg(all(feature = "provider-llamacpp", feature = "plugin-skills"))]

mod common;

use std::sync::{Arc, Mutex};

use cuca::plugin::CucaPlugin;
use cuca::types::{MessageContentBlock, MessageRole};
use cuca::{PluginError, Skill, SkillsPlugin, UnifiedRequest};
use serde_json::{Value, json};

fn plugin() -> SkillsPlugin {
    SkillsPlugin::inline(vec![
        Skill::inline(
            "math",
            "Arithmetic operations",
            "Add, subtract, multiply, divide.",
        ),
        Skill::inline(
            "web",
            "Fetch web pages",
            "Use HTTP requests to retrieve pages.",
        ),
    ])
}

#[test]
fn on_request_injects_a_catalog_system_message() {
    let plugin = plugin();
    let mut req = UnifiedRequest::new("test-model").add_user_message("hi");
    plugin
        .on_request(&mut req)
        .expect("on_request must return Ok(())");

    let text = req
        .messages
        .iter()
        .filter(|m| m.role == MessageRole::System)
        .flat_map(|m| m.content.iter())
        .filter_map(|b| match b {
            MessageContentBlock::Text(t) => Some(t.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n");

    assert!(
        text.contains("math") && text.contains("Arithmetic operations"),
        "the injected catalog must mention the skill name and description, got: {text}"
    );
}

#[test]
fn skill_tool_call_returns_json_for_known_and_error_for_unknown() {
    let plugin = plugin();

    // Known skill -> ToolResult whose output parses as JSON with the skill data.
    let mut known = MessageContentBlock::ToolCall {
        id: "c1".to_string(),
        name: "skill".to_string(),
        arguments: json!({ "name": "math" }),
    };
    plugin
        .on_stream_chunk(&mut known)
        .expect("on_stream_chunk must return Ok(())");
    match known {
        MessageContentBlock::ToolResult {
            tool_call_id,
            output,
        } => {
            assert_eq!(tool_call_id, "c1");
            let v: Value = serde_json::from_str(&output)
                .unwrap_or_else(|e| panic!("skill output must be JSON: {e} — got {output}"));
            assert_eq!(v["name"], "math");
            assert!(
                v["instructions"]
                    .as_str()
                    .unwrap()
                    .contains("Add, subtract"),
                "instructions: {v}"
            );
        }
        other => panic!("expected ToolResult, got {other:?}"),
    }

    // Unknown skill -> ToolResult with descriptive error text.
    let mut unknown = MessageContentBlock::ToolCall {
        id: "c2".to_string(),
        name: "skill".to_string(),
        arguments: json!({ "name": "no_such_skill" }),
    };
    plugin
        .on_stream_chunk(&mut unknown)
        .expect("on_stream_chunk must return Ok(())");
    match unknown {
        MessageContentBlock::ToolResult { output, .. } => {
            assert!(output.contains("unknown skill"), "output: {output}");
        }
        other => panic!("expected ToolResult, got {other:?}"),
    }
}

#[test]
fn skill_search_tool_call_lists_matching_skills() {
    let plugin = plugin();
    let mut block = MessageContentBlock::ToolCall {
        id: "c3".to_string(),
        name: "skill_search".to_string(),
        arguments: json!({ "query": "arithmetic" }),
    };
    plugin
        .on_stream_chunk(&mut block)
        .expect("on_stream_chunk must return Ok(())");
    match block {
        MessageContentBlock::ToolResult { output, .. } => {
            let v: Value = serde_json::from_str(&output)
                .unwrap_or_else(|e| panic!("search output must be JSON: {e} — got {output}"));
            let arr = v.as_array().expect("search output must be an array");
            assert!(
                arr.iter().any(|s| s["name"] == "math"),
                "search must surface the matching `math` skill, got: {arr:?}"
            );
        }
        other => panic!("expected ToolResult, got {other:?}"),
    }
}

/// Records the request as it exists after every earlier `on_request` hook, so a
/// live test can prove the injected catalog is in the request that actually
/// went out rather than only in a hand-built one.
#[derive(Default)]
struct RequestCapture {
    requests: Mutex<Vec<UnifiedRequest>>,
}

impl CucaPlugin for RequestCapture {
    fn name(&self) -> &'static str {
        "request-capture"
    }

    fn on_request(&self, req: &mut UnifiedRequest) -> Result<(), PluginError> {
        self.requests
            .lock()
            .expect("capture lock must not be poisoned")
            .push(req.clone());
        Ok(())
    }
}

/// The live round trip must show the plugin engaging on both of its seams: the
/// catalog `on_request` injects must be in the outbound request (observed by a
/// capture plugin registered after it), and an injected `skill` tool call must
/// be answered from the real catalog mid-stream.
#[tokio::test]
async fn live_request_carries_the_catalog_and_answers_a_skill_tool_call() {
    if let Err(reason) = common::require_live_server() {
        eprintln!("SKIP: llama.cpp not reachable: {reason}");
        return;
    }
    let injector = Arc::new(common::ToolCallInjector::new(
        "live-skill-1",
        "skill",
        json!({ "name": "math" }),
    ));
    let capture = Arc::new(RequestCapture::default());
    // Order matters on both seams: the injector must precede skills so skills
    // sees the rewritten chunk, and skills must precede the capture so the
    // capture observes the injected catalog.
    let client = common::client_with_plugins(vec![
        injector.clone() as Arc<dyn CucaPlugin>,
        Arc::new(plugin()) as Arc<dyn CucaPlugin>,
        capture.clone() as Arc<dyn CucaPlugin>,
    ]);
    let request = common::live_request("Reply with the single word: ok", &common::live_model());
    let stream = client
        .generate_stream(request)
        .await
        .expect("generate_stream must start");
    let blocks = common::drain_timeout(stream, 60).await;

    // Seam 1: the catalog reached the outbound request.
    let captured = capture
        .requests
        .lock()
        .expect("capture lock must not be poisoned");
    assert_eq!(captured.len(), 1, "exactly one request was sent");
    let system_text = captured[0]
        .messages
        .iter()
        .filter(|m| m.role == MessageRole::System)
        .flat_map(|m| m.content.iter())
        .filter_map(|b| match b {
            MessageContentBlock::Text(t) => Some(t.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        system_text.contains("math") && system_text.contains("Arithmetic operations"),
        "the outbound request must carry the injected catalog, got: {system_text}"
    );

    // Seam 2: the real catalog answered a tool call mid-live-stream.
    assert!(
        injector.injected(),
        "the live turn produced no model chunk to convert, so nothing was \
         exercised; got {blocks:?}"
    );
    let output = common::tool_result_output(&blocks, injector.call_id()).unwrap_or_else(|| {
        panic!("the skills plugin must answer the injected call, got {blocks:?}")
    });
    let value: Value = serde_json::from_str(&output)
        .unwrap_or_else(|e| panic!("skill output must be JSON: {e} — got {output}"));
    assert_eq!(value["name"], "math");
    assert!(
        value["instructions"]
            .as_str()
            .is_some_and(|text| text.contains("Add, subtract")),
        "the ToolResult must carry the real skill's instructions, got {value}"
    );
}
