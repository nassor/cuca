//! Integration tests for the speculative fast/slow orchestrator
//! (`plugin-speculative`).
//!
//! Deterministic tests cover the complexity router, the draft validator, and
//! the endpoint a pool-backed tier turn dispatches to; the live test runs a
//! full orchestrated turn (complexity routing, fast-tier draft, validation,
//! latency guard) against llama.cpp.

#![cfg(all(feature = "provider-llamacpp", feature = "plugin-speculative"))]

mod common;

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use cuca::types::{MessageContentBlock, MessageRole, ProviderEndpoint, UnifiedMessage};
use cuca::{
    ClientPool, Complexity, ComplexityEvaluator, DraftValidator, JsonToolDraftValidator,
    ModelOrchestrator, SwappableModelPair, UnifiedRequest,
};
use serde_json::json;

#[test]
fn complexity_routes_tiny_request_fast() {
    let evaluator = ComplexityEvaluator::default();
    let req = UnifiedRequest::new("m")
        .add_system_message("You are concise.")
        .add_user_message("Summarize the plan.");
    assert_eq!(evaluator.evaluate(&req), Complexity::Fast);
}

#[test]
fn complexity_routes_tool_messages_slow() {
    // The default slow_tool_call_depth is 1: a single tool round-trip tips the
    // request into the slow tier.
    let evaluator = ComplexityEvaluator::default();
    let req = UnifiedRequest::new("m").add_message(UnifiedMessage {
        role: MessageRole::Assistant,
        content: vec![MessageContentBlock::ToolCall {
            id: "call_1".into(),
            name: "read".into(),
            arguments: json!({ "path": "src/lib.rs" }),
        }],
        name: None,
        tool_call_id: None,
    });
    assert_eq!(evaluator.evaluate(&req), Complexity::Slow);
}

#[test]
fn complexity_routes_large_input_slow() {
    // 8000 chars estimate to 2000 input tokens, exactly the slow threshold.
    let evaluator = ComplexityEvaluator::default();
    let big = "a".repeat(8000);
    let req = UnifiedRequest::new("m").add_user_message(big);
    assert_eq!(evaluator.evaluate(&req), Complexity::Slow);
}

#[test]
fn complexity_routes_multi_file_context_slow() {
    // Three distinct path-like tokens hit the default multi-file threshold.
    let evaluator = ComplexityEvaluator::default();
    let req = UnifiedRequest::new("m").add_user_message("refactor src/a.rs src/b.rs src/c.rs");
    assert_eq!(evaluator.evaluate(&req), Complexity::Slow);
}

#[test]
fn validator_rejects_empty_tool_call_id() {
    let validator = JsonToolDraftValidator;
    let block = MessageContentBlock::ToolCall {
        id: String::new(),
        name: "read".into(),
        arguments: json!({}),
    };
    assert_eq!(
        validator.validate(&block),
        Err("tool call id must be non-empty".to_string()),
        "empty tool call id must be rejected with the id-specific message"
    );
}

#[test]
fn validator_rejects_unparseable_json_text_arguments() {
    let validator = JsonToolDraftValidator;
    let block = MessageContentBlock::ToolCall {
        id: "call_1".into(),
        name: "read".into(),
        arguments: serde_json::Value::String("{not valid json".into()),
    };
    let err = validator
        .validate(&block)
        .expect_err("string arguments that are not valid JSON must be rejected");
    assert!(
        err.starts_with("tool call arguments are not valid JSON: "),
        "unexpected rejection message: {err}"
    );
}

#[test]
fn validator_rejects_non_object_json_text() {
    let validator = JsonToolDraftValidator;
    let block = MessageContentBlock::Text("\"just a string\"".into());
    assert_eq!(
        validator.validate(&block),
        Err("text block is valid JSON but not a JSON object".to_string()),
        "text that parses as JSON but is not an object must be rejected with the object-specific message"
    );
}

#[test]
fn validator_accepts_valid_blocks() {
    let validator = JsonToolDraftValidator;
    let call = MessageContentBlock::ToolCall {
        id: "call_1".into(),
        name: "read".into(),
        arguments: json!({ "path": "src/lib.rs" }),
    };
    let call_before = call.clone();
    assert_eq!(validator.validate(&call), Ok(()));
    assert_eq!(
        call, call_before,
        "validate must not mutate the ToolCall block it inspects"
    );

    let prose = MessageContentBlock::Text("plain prose".into());
    let prose_before = prose.clone();
    assert_eq!(validator.validate(&prose), Ok(()));
    assert_eq!(
        prose, prose_before,
        "validate must not mutate the Text block it inspects"
    );

    let result = MessageContentBlock::ToolResult {
        tool_call_id: "call_1".into(),
        output: "42".into(),
    };
    let result_before = result.clone();
    assert_eq!(validator.validate(&result), Ok(()));
    assert_eq!(
        result, result_before,
        "validate must not mutate the ToolResult block it inspects"
    );
}

/// A pool-backed tier turn dispatches to the endpoint configured on the
/// orchestrator.
///
/// The enclosing client points at a closed port, so the mock server can only
/// be reached through the orchestrator's own endpoint.
#[tokio::test]
async fn orchestrated_turn_dispatches_to_the_endpoint_configured_on_the_orchestrator() {
    let dispatches = Arc::new(AtomicUsize::new(0));
    let addr = common::spawn_counting_sse_server(Arc::clone(&dispatches), "ok").await;
    let tier_base = format!("http://{addr}/v1");
    let config = SwappableModelPair {
        fast_provider: ProviderEndpoint::LlamaCpp,
        fast_model_id: "fast-tier-id".into(),
        slow_provider: ProviderEndpoint::LlamaCpp,
        slow_model_id: "slow-tier-id".into(),
        latency_threshold_ms: 60_000,
        fallback_on_tool_error: false,
    };
    let orchestrator = ModelOrchestrator::new(config, Arc::new(ClientPool::default()))
        .with_endpoint(tier_base, None);
    let client = common::llamacpp_builder("http://127.0.0.1:1/v1")
        .with_orchestrator(orchestrator)
        .build()
        .expect("client build must succeed");

    let blocks = common::drain_timeout(
        client
            .generate_stream(UnifiedRequest::new("ignored").add_user_message("hi"))
            .await
            .expect("orchestrated turn must start"),
        10,
    )
    .await;

    assert_eq!(common::text_of(&blocks), "ok");
    assert_eq!(
        dispatches.load(Ordering::SeqCst),
        1,
        "the fast tier must reach the mock exactly once"
    );
}

#[tokio::test]
async fn live_orchestrated_turn_yields_text() {
    // The gate probes the server with its own runtime, which must never run
    // inside a tokio runtime, so resolve it on a plain OS thread first.
    if let Err(reason) = common::require_live_server() {
        eprintln!("SKIP: llama.cpp not reachable: {reason}");
        return;
    }
    let model = common::live_model();
    let config = SwappableModelPair {
        fast_provider: ProviderEndpoint::LlamaCpp,
        fast_model_id: model.clone(),
        slow_provider: ProviderEndpoint::LlamaCpp,
        slow_model_id: model.clone(),
        latency_threshold_ms: 30_000,
        fallback_on_tool_error: false,
    };
    let pool = Arc::new(ClientPool::default());
    // The tier executors dispatch through pool clients, which take their
    // endpoint from the orchestrator, not from the enclosing client.
    let orchestrator = ModelOrchestrator::new(config, pool).with_endpoint(common::base_url(), None);

    let client = cuca::CucaClient::builder()
        .with_provider(ProviderEndpoint::LlamaCpp)
        .with_base_url(common::base_url())
        .with_orchestrator(orchestrator)
        .build()
        .expect("llama.cpp client build must succeed");

    let stream = client
        .generate_stream(common::live_request(
            "Reply with the single word: ok",
            &model,
        ))
        .await
        .expect("generate_stream must succeed");
    let blocks = common::drain_timeout(stream, 120).await;
    assert!(
        blocks
            .iter()
            .any(|b| matches!(b, MessageContentBlock::Text(_))),
        "expected at least one Text block from the orchestrated turn, got {blocks:?}"
    );
}
