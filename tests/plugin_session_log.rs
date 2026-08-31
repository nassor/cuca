//! Integration tests for the session-log plugin (`plugin-session-log`).
//!
//! The deterministic tests drive the [`CucaPlugin`] hooks and
//! [`SessionStorePlugin`] methods directly against an in-memory backend and a
//! temp-directory file backend; the live test registers the plugin on a
//! llama.cpp client and replays the records that a real request produced.
#![cfg(all(feature = "provider-llamacpp", feature = "plugin-session-log"))]

mod common;

use std::sync::Arc;

use cuca::plugin::{CucaPlugin, SessionStorePlugin};
use cuca::session::SessionEvent;
use cuca::types::{MessageContentBlock, ProviderEndpoint};
use cuca::{FileBackend, SessionLogPlugin, SessionRecord, UnifiedRequest, UnifiedResponse};

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

/// Removes the temp directory on drop so tests leave no files behind.
struct TestDir(std::path::PathBuf);

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn fresh_temp_dir(label: &str) -> TestDir {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    TestDir(std::env::temp_dir().join(format!(
        "cuca-session-log-it-{label}-{}-{nanos}",
        std::process::id()
    )))
}

#[test]
fn file_backend_persists_hook_records_across_plugin_instances() {
    let guard = fresh_temp_dir("persist");
    let dir = guard.0.clone();

    // First instance records a full interaction through the hooks.
    {
        let backend = Arc::new(FileBackend::new(&dir).expect("backend must open"));
        let plugin = Arc::new(SessionLogPlugin::new(backend).with_session_id("s1"));

        let mut req = UnifiedRequest::new("test-model").add_system_message("You are concise.");
        plugin
            .on_request(&mut req)
            .expect("on_request must return Ok(())");

        let mut call = MessageContentBlock::ToolCall {
            id: "call-1".to_string(),
            name: "search".to_string(),
            arguments: serde_json::json!({ "z": [1, 2], "a": null }),
        };
        plugin
            .on_stream_chunk(&mut call)
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
    }

    // A brand-new instance over the same directory replays what was written.
    let reopened = SessionLogPlugin::new(Arc::new(
        FileBackend::new(&dir).expect("backend must reopen"),
    ));
    let records = reopened.replay_session("s1").expect("replay must succeed");
    let kinds: Vec<&str> = records.iter().map(|r| event_kind(&r.event)).collect();
    assert!(kinds.contains(&"SystemPrompt"), "records: {kinds:?}");
    assert!(kinds.contains(&"ToolCall"), "records: {kinds:?}");
    assert!(kinds.contains(&"Latency"), "records: {kinds:?}");
    assert!(kinds.contains(&"TokenUsage"), "records: {kinds:?}");
    assert!(
        records.windows(2).all(|w| w[0].sequence < w[1].sequence),
        "records must replay in strictly increasing sequence order, got: {records:?}"
    );

    // The tool-call arguments survive the postcard round trip intact.
    let arguments = records
        .iter()
        .find_map(|r| match &r.event {
            SessionEvent::ToolCall { arguments, .. } => Some(arguments.clone()),
            _ => None,
        })
        .expect("a ToolCall record must be present");
    assert_eq!(arguments, serde_json::json!({ "a": null, "z": [1, 2] }));

    // Records land in a framed `.cslog` file, one per session.
    let bytes = std::fs::read(dir.join("s1.cslog")).expect("session file must exist");
    assert_eq!(&bytes[..8], b"CUCASLOG");
}

#[test]
fn file_backend_forks_a_session_on_disk() {
    let guard = fresh_temp_dir("fork");
    let backend = Arc::new(FileBackend::new(&guard.0).expect("backend must open"));
    let plugin = SessionLogPlugin::new(backend);

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
    let branch = plugin.replay_session(&new_id).expect("replay must succeed");
    assert_eq!(
        branch.len(),
        3,
        "branch is the prefix through the fork point"
    );
    assert!(branch.iter().all(|r| r.session_id == new_id));

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

#[test]
fn file_backend_reports_a_legacy_jsonl_session() {
    let guard = fresh_temp_dir("legacy");
    let dir = guard.0.clone();
    std::fs::create_dir_all(&dir).expect("dir must be created");
    std::fs::write(
        dir.join("old.jsonl"),
        b"{\"session_id\":\"old\",\"sequence\":0,\"timestamp_ms\":1,\
          \"event\":{\"type\":\"output\",\"text\":\"x\"}}\n",
    )
    .expect("fixture must be written");

    let plugin =
        SessionLogPlugin::new(Arc::new(FileBackend::new(&dir).expect("backend must open")));
    let err = plugin
        .replay_session("old")
        .expect_err("a legacy JSON-lines session must not replay as empty");
    assert!(format!("{err}").contains("old.jsonl"), "{err}");
}
