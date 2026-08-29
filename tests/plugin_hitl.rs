//! Integration tests for the human-in-the-loop approval plugin (`plugin-hitl`).
//!
//! Risk classification, approval pass-through/denial, the audit log, and the
//! oneshot channel seam are all proven deterministically; one live pipeline
//! smoke test runs a real request with an auto-approving channel registered.

#![cfg(all(feature = "provider-llamacpp", feature = "plugin-hitl"))]

mod common;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use cuca::plugin::CucaPlugin;
use cuca::plugins::hitl::classify_tool_call;
use cuca::types::MessageContentBlock;
use cuca::{
    ApprovalChannel, ApprovalDecision, ApprovalRequest, HitlPlugin, OneshotApprovalChannel, Risk,
};
use serde_json::json;

/// Channel that approves every request.
#[derive(Clone, Copy)]
struct AutoApprove;

impl ApprovalChannel for AutoApprove {
    fn request_approval(&self, _req: &ApprovalRequest) -> ApprovalDecision {
        ApprovalDecision::Approved
    }
}

/// Channel that denies every request.
#[derive(Clone, Copy)]
struct AutoDeny;

impl ApprovalChannel for AutoDeny {
    fn request_approval(&self, _req: &ApprovalRequest) -> ApprovalDecision {
        ApprovalDecision::Denied
    }
}

#[test]
fn classify_tool_call_table() {
    for high in [
        "shell",
        "exec",
        "bash",
        "write",
        "edit",
        "delete",
        "http_post",
        "api_write",
        "run_command",
        "terminal",
        "create_file",
    ] {
        assert_eq!(classify_tool_call(high), Risk::High, "{high}");
    }
    for low in [
        "read",
        "search",
        "web_search",
        "get_weather",
        "unrelated_tool",
    ] {
        assert_eq!(classify_tool_call(low), Risk::Low, "{low}");
    }
}

#[test]
fn approved_high_risk_passes_through() {
    let plugin = HitlPlugin::new(Arc::new(AutoApprove));
    let mut chunk = MessageContentBlock::ToolCall {
        id: "call_1".into(),
        name: "shell".into(),
        arguments: json!({ "cmd": "ls -la" }),
    };
    plugin
        .on_stream_chunk(&mut chunk)
        .expect("hook must succeed");
    // Approved calls stream through exactly as the model emitted them.
    assert!(
        matches!(&chunk, MessageContentBlock::ToolCall { name, .. } if name == "shell"),
        "approved call must pass through unchanged: {chunk:?}"
    );
    let audit = plugin.audit_log();
    assert_eq!(audit.len(), 1);
    assert_eq!(audit[0].status, "approved");
    assert_eq!(audit[0].action_requested, "shell_exec");
}

#[test]
fn denied_high_risk_replaced_with_tool_result() {
    let plugin = HitlPlugin::new(Arc::new(AutoDeny));
    let mut chunk = MessageContentBlock::ToolCall {
        id: "call_2".into(),
        name: "write".into(),
        arguments: json!({ "path": "/tmp/x", "content": "x" }),
    };
    plugin
        .on_stream_chunk(&mut chunk)
        .expect("hook must succeed");
    // Denied calls are replaced by a ToolResult carrying the denial text.
    match &chunk {
        MessageContentBlock::ToolResult {
            tool_call_id,
            output,
        } => {
            assert_eq!(tool_call_id, "call_2");
            assert!(output.contains("denied"), "output: {output}");
        }
        other => panic!("expected ToolResult, got {other:?}"),
    }
    let audit = plugin.audit_log();
    assert_eq!(audit.len(), 1);
    assert_eq!(audit[0].status, "denied");
    assert_eq!(audit[0].action_requested, "file_write");
}

#[tokio::test]
async fn oneshot_channel_driven_from_os_thread() {
    let (channel, sender) = OneshotApprovalChannel::new();
    let plugin = Arc::new(HitlPlugin::new(Arc::new(channel)));
    // Ordering witness: `sent` flips immediately before the decision is
    // dispatched, and the gate thread reads it only *after* `on_stream_chunk`
    // returned. A hook that did not block until the decision arrived would
    // return during the 50ms window below and observe `false`.
    let sent = Arc::new(AtomicBool::new(false));
    let witness = Arc::clone(&sent);
    let gate = Arc::clone(&plugin);
    // request_approval blocks on a tokio oneshot (blocking_lock +
    // blocking_recv), which panics inside a runtime worker, so the gate runs
    // on a dedicated OS thread and is answered from here.
    let handle = std::thread::spawn(move || {
        let mut chunk = MessageContentBlock::ToolCall {
            id: "call_4".into(),
            name: "exec".into(),
            arguments: json!({}),
        };
        gate.on_stream_chunk(&mut chunk).expect("hook must succeed");
        (chunk, witness.load(Ordering::SeqCst))
    });
    std::thread::sleep(std::time::Duration::from_millis(50));
    sent.store(true, Ordering::SeqCst);
    let _ = sender.send(ApprovalDecision::Approved);
    let (chunk, observed_send) = handle.join().expect("gate thread must not panic");

    assert!(
        observed_send,
        "the hook must block until the decision is dispatched; it returned before \
         the send"
    );
    assert!(
        matches!(&chunk, MessageContentBlock::ToolCall { name, .. } if name == "exec"),
        "approved oneshot call must pass through: {chunk:?}"
    );
    let audit = plugin.audit_log();
    assert_eq!(
        audit.len(),
        1,
        "the gated call must be audited exactly once"
    );
    assert_eq!(audit[0].status, "approved");
}

/// The live round trip must exercise the gate, not merely coexist with it. An
/// injector registered *before* the plugin rewrites the first live model chunk
/// into a high-risk `shell` call, so the real classifier and approval channel
/// decide mid-stream: the approving client passes the call through and audits
/// it, the denying client replaces the delivered block outright.
#[tokio::test]
async fn live_stream_chunk_gates_an_injected_high_risk_call() {
    if let Err(reason) = common::require_live_server() {
        eprintln!("SKIP: llama.cpp not reachable: {reason}");
        return;
    }
    let model = common::live_model();

    let injector = Arc::new(common::ToolCallInjector::new(
        "live-shell-1",
        "shell",
        json!({ "cmd": "ls -la" }),
    ));
    let approving = Arc::new(HitlPlugin::new(Arc::new(AutoApprove)));
    let client = common::client_with_plugins(vec![
        injector.clone() as Arc<dyn CucaPlugin>,
        approving.clone() as Arc<dyn CucaPlugin>,
    ]);
    let blocks = common::drain_timeout(
        client
            .generate_stream(common::live_request(
                "Reply with the single word: ok",
                &model,
            ))
            .await
            .expect("generate_stream must succeed"),
        60,
    )
    .await;

    assert!(
        injector.injected(),
        "the live turn produced no model chunk to convert, so nothing was \
         exercised; got {blocks:?}"
    );
    assert!(
        blocks.iter().any(
            |b| matches!(b, MessageContentBlock::ToolCall { id, name, .. }
                if id == "live-shell-1" && name == "shell")
        ),
        "an approved call streams through as the model emitted it: {blocks:?}"
    );
    let audit = approving.audit_log();
    assert_eq!(
        audit.len(),
        1,
        "the live high-risk call must be audited exactly once, got {audit:?}"
    );
    assert_eq!(audit[0].status, "approved");
    assert_eq!(
        audit[0].action_requested, "shell_exec",
        "the real classifier ran over the injected call"
    );

    // Same injected call, denying channel: the gate changes what the consumer
    // receives, so a plugin that stopped engaging is visible in the stream too.
    let injector = Arc::new(common::ToolCallInjector::new(
        "live-shell-2",
        "shell",
        json!({ "cmd": "ls -la" }),
    ));
    let denying = Arc::new(HitlPlugin::new(Arc::new(AutoDeny)));
    let client = common::client_with_plugins(vec![
        injector.clone() as Arc<dyn CucaPlugin>,
        denying.clone() as Arc<dyn CucaPlugin>,
    ]);
    let blocks = common::drain_timeout(
        client
            .generate_stream(common::live_request(
                "Reply with the single word: ok",
                &model,
            ))
            .await
            .expect("generate_stream must succeed"),
        60,
    )
    .await;

    assert!(injector.injected(), "no model chunk to convert: {blocks:?}");
    let output = common::tool_result_output(&blocks, "live-shell-2").unwrap_or_else(|| {
        panic!("a denied call must be replaced by a ToolResult, got {blocks:?}")
    });
    assert!(
        output.contains("denied"),
        "the denial text must reach the consumer, got {output:?}"
    );
    assert!(
        !blocks
            .iter()
            .any(|b| matches!(b, MessageContentBlock::ToolCall { name, .. } if name == "shell")),
        "a denied call must not stream through: {blocks:?}"
    );
    let audit = denying.audit_log();
    assert_eq!(audit.len(), 1);
    assert_eq!(audit[0].status, "denied");
}
