//! Integration tests for the session-log plugin (`plugin-session-log`).
//!
//! The deterministic tests drive the [`CucaPlugin`] hooks and
//! [`SessionStorePlugin`] methods directly against an in-memory backend; the
//! live test registers the plugin on a llama.cpp client and replays the
//! records that a real request produced.
#![cfg(all(feature = "provider-llamacpp", feature = "plugin-session-log"))]

mod common;

use std::sync::Arc;

use cuca::plugin::{CucaPlugin, SessionStorePlugin};
use cuca::session::SessionEvent;
use cuca::types::{MessageContentBlock, ProviderEndpoint};
use cuca::{SessionLogPlugin, SessionRecord, UnifiedRequest, UnifiedResponse};

/// Human-readable kind of a session event, for assertions.
fn event_kind(e: &SessionEvent) -> &'static str {
    match e {
        SessionEvent::SystemPrompt { .. } => "SystemPrompt",
        SessionEvent::Message { .. } => "Message",
        SessionEvent::Reasoning { .. } => "Reasoning",
        SessionEvent::Output { .. } => "Output",
        SessionEvent::ToolCall { .. } => "ToolCall",
        SessionEvent::ToolResult { .. } => "ToolResult",
        SessionEvent::ModelSwap { .. } => "ModelSwap",
        SessionEvent::Latency { .. } => "Latency",
        SessionEvent::TokenUsage { .. } => "TokenUsage",
        SessionEvent::Fork { .. } => "Fork",
    }
}

#[test]
fn hooks_record_system_output_and_usage_events() {
    let plugin = Arc::new(SessionLogPlugin::new_in_memory());

    let mut req = UnifiedRequest::new("test-model").add_system_message("You are concise.");
    plugin
        .on_request(&mut req)
        .expect("on_request must return Ok(())");

    let mut chunk = MessageContentBlock::Text("hello".to_string());
    plugin
        .on_stream_chunk(&mut chunk)
        .expect("on_stream_chunk must return Ok(())");

    let res = UnifiedResponse {
        model: "test-model".into(),
        provider: ProviderEndpoint::LlamaCpp,
        duration_secs: 1.5,
        prompt_tokens: 10,
        completion_tokens: 5,
        finish_reason: Some("stop".into()),
        content: Vec::new(),
        prompt_cache_usage: None,
    };
    plugin
        .on_response_complete(&res)
        .expect("on_response_complete must return Ok(())");

    let records = plugin
        .replay_session("default")
        .expect("replay must succeed");
    let kinds: Vec<&str> = records.iter().map(|r| event_kind(&r.event)).collect();
    assert!(kinds.contains(&"SystemPrompt"), "records: {kinds:?}");
    assert!(kinds.contains(&"Output"), "records: {kinds:?}");
    assert!(kinds.contains(&"Latency"), "records: {kinds:?}");
    assert!(kinds.contains(&"TokenUsage"), "records: {kinds:?}");
}

#[test]
fn fork_session_branches_and_audits_the_original() {
    let plugin = Arc::new(SessionLogPlugin::new_in_memory());
    for i in 0..4 {
        plugin
            .append_log(
                "default",
                &SessionRecord::new(
                    "default",
                    SessionEvent::Output {
                        text: format!("o{i}"),
                    },
                ),
            )
            .expect("append must succeed");
    }

    let new_id = plugin
        .fork_session("default", "default:2")
        .expect("fork must succeed");
    assert!(new_id.starts_with("default:fork:default:2:"));

    // The branch contains the prefix up to and including the fork point.
    let branch = plugin.replay_session(&new_id).expect("replay must succeed");
    assert_eq!(branch.len(), 3);
    assert!(branch.iter().all(|r| r.session_id == new_id));

    // The original gains a Fork audit record at its tail.
    let original = plugin
        .replay_session("default")
        .expect("replay must succeed");
    assert!(
        matches!(
            original.last().map(|r| &r.event),
            Some(SessionEvent::Fork { .. })
        ),
        "expected a trailing Fork record, got: {original:?}"
    );
}

#[tokio::test]
async fn live_request_writes_replayable_session_records() {
    if let Err(reason) = common::require_live_server() {
        eprintln!("SKIP: llama.cpp not reachable: {reason}");
        return;
    }
    let plugin = Arc::new(SessionLogPlugin::new_in_memory());
    let client = common::client_with_plugins(vec![Arc::clone(&plugin) as Arc<dyn CucaPlugin>]);

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

    let records = plugin
        .replay_session("default")
        .expect("replay must succeed");
    let kinds: Vec<&str> = records.iter().map(|r| event_kind(&r.event)).collect();
    assert!(kinds.contains(&"SystemPrompt"), "records: {kinds:?}");
    assert!(kinds.contains(&"Output"), "records: {kinds:?}");
    assert!(kinds.contains(&"Latency"), "records: {kinds:?}");
    assert!(kinds.contains(&"TokenUsage"), "records: {kinds:?}");
    assert!(
        records.iter().all(|r| r.session_id == "default"),
        "every replayed record must belong to the \"default\" session, got: {records:?}"
    );
    assert!(
        records.windows(2).all(|w| w[0].sequence < w[1].sequence),
        "records must replay in strictly increasing sequence order, got: {records:?}"
    );
}
