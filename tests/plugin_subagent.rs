//! Integration tests for the subagent delegation plugin (`plugin-subagent`).
//!
//! The plugin's fan-out runs each child on a background tokio task; on the
//! default current-thread test runtime the child only makes progress while the
//! test task awaits, so the spawn/collect tests pump the runtime until the
//! runner signals completion before calling `collect` (which would otherwise
//! block the only executor thread). One live test exercises a REAL runner that
//! performs a llama.cpp request through `common::client()`.

#![cfg(all(feature = "provider-llamacpp", feature = "plugin-subagent"))]

mod common;

use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use cuca::plugin::CucaPlugin;
use cuca::types::MessageContentBlock;
use cuca::{PluginError, SubagentPlugin, SubagentResult, SubagentRunner, SubagentSpec};
use serde_json::json;
use tokio_stream::StreamExt;

/// Runner that resolves every spawn to a fixed summary and flips `done` right
/// before returning, so the pump loop knows the mpsc delivery has landed.
struct CannedRunner {
    summary: String,
    done: Arc<AtomicBool>,
}

impl CannedRunner {
    fn new(summary: &str, done: Arc<AtomicBool>) -> Self {
        Self {
            summary: summary.to_string(),
            done,
        }
    }
}

impl SubagentRunner for CannedRunner {
    fn spawn(&self, _spec: SubagentSpec) -> Pin<Box<dyn Future<Output = SubagentResult> + Send>> {
        let summary = self.summary.clone();
        let done = self.done.clone();
        Box::pin(async move {
            done.store(true, Ordering::SeqCst);
            SubagentResult {
                subagent_id: "canned".to_string(),
                summary,
                worktree_path: None,
                exit_ok: true,
            }
        })
    }
}

/// Runner that performs a REAL llama.cpp request (trivial prompt), summarizing
/// the text the model emits. The model id is resolved by the test on a plain OS
/// thread and passed in, because the runner executes inside the tokio runtime
/// where a server probe would panic.
///
/// There is deliberately no fallback summary: a dispatch error, a stream error,
/// a stall, or an empty reply all surface as `exit_ok: false` with the reason as
/// the summary, so a broken live call fails the test instead of being masked by
/// a canned string that satisfies a non-empty assertion.
struct LiveRunner {
    model: String,
    done: Arc<AtomicBool>,
}

impl SubagentRunner for LiveRunner {
    fn spawn(&self, _spec: SubagentSpec) -> Pin<Box<dyn Future<Output = SubagentResult> + Send>> {
        let model = self.model.clone();
        let done = self.done.clone();
        Box::pin(async move {
            let mut summary = String::new();
            let mut failure: Option<String> = None;
            let client = common::client();
            let request = common::live_request("Reply with the single word: ok", &model);
            match client.generate_stream(request).await {
                Ok(mut stream) => loop {
                    match tokio::time::timeout(Duration::from_secs(60), stream.next()).await {
                        Ok(Some(Ok(MessageContentBlock::Text(text)))) => summary.push_str(&text),
                        Ok(Some(Ok(_))) => {}
                        Ok(Some(Err(error))) => {
                            failure = Some(format!("stream error: {error}"));
                            break;
                        }
                        Ok(None) => break,
                        Err(_) => {
                            failure = Some("stream stalled for 60s".to_string());
                            break;
                        }
                    }
                },
                Err(error) => failure = Some(format!("generate_stream failed: {error}")),
            }
            if failure.is_none() && summary.trim().is_empty() {
                failure = Some("the model emitted no text".to_string());
            }
            let exit_ok = failure.is_none();
            done.store(true, Ordering::SeqCst);
            SubagentResult {
                subagent_id: "live".to_string(),
                summary: failure.unwrap_or(summary),
                worktree_path: None,
                exit_ok,
            }
        })
    }
}

fn spec(task: &str) -> SubagentSpec {
    SubagentSpec {
        name: "child".to_string(),
        task: task.to_string(),
        tool_scope: vec!["read".to_string()],
        worktree: None,
        session_id: Some("sess-1".to_string()),
    }
}

/// Spawn a child and pump the runtime until the runner signals completion,
/// then collect. On the current-thread runtime `collect` blocks the executor,
/// so it must only run once the child's result is already delivered.
async fn spawn_and_collect(
    plugin: &SubagentPlugin,
    done: Arc<AtomicBool>,
    timeout: Duration,
) -> SubagentResult {
    let id = plugin
        .spawn_subagent(spec("summarize the docs"))
        .unwrap_or_else(|e| panic!("spawn_subagent failed: {e}"));
    let deadline = tokio::time::Instant::now() + timeout;
    while !done.load(Ordering::SeqCst) {
        assert!(
            tokio::time::Instant::now() < deadline,
            "child subagent never completed within {timeout:?}"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    plugin
        .collect(&id)
        .unwrap_or_else(|e| panic!("collect failed: {e}"))
}

#[tokio::test]
async fn canned_runner_spawn_collect_roundtrip() {
    let done = Arc::new(AtomicBool::new(false));
    let runner = CannedRunner::new("summarized!", done.clone());
    let plugin = SubagentPlugin::new(Arc::new(runner));

    let result = spawn_and_collect(&plugin, done, Duration::from_secs(10)).await;
    assert_eq!(result.summary, "summarized!");
    assert!(result.exit_ok);
    assert_eq!(plugin.spawn_count(), 1);

    // The diagnostic metric logged one (session_id, worktree_path) entry.
    let spawns = plugin.spawns();
    assert_eq!(spawns.len(), 1);
    assert_eq!(spawns[0].0.as_deref(), Some("sess-1"));
    assert_eq!(spawns[0].1, None);
}

#[tokio::test]
async fn real_runner_live_model_request() {
    if let Err(reason) = common::require_live_server() {
        eprintln!("SKIP: llama.cpp not reachable: {reason}");
        return;
    }
    let model = common::live_model();
    let done = Arc::new(AtomicBool::new(false));
    let plugin = SubagentPlugin::new(Arc::new(LiveRunner {
        model,
        done: done.clone(),
    }));

    let result = spawn_and_collect(&plugin, done, Duration::from_secs(120)).await;
    // `exit_ok` is the whole assertion: the runner sets it false and puts the
    // reason in `summary` for every failure mode, so a live call that did not
    // actually produce model text fails here with that reason.
    assert!(
        result.exit_ok,
        "the real runner's live request must succeed; reason: {}",
        result.summary
    );
    assert!(
        !result.summary.trim().is_empty(),
        "a successful run carries the model's own text"
    );
    assert_eq!(result.subagent_id, "live");
}

#[test]
fn malformed_spawn_call_reports_validation_error() {
    let plugin = SubagentPlugin::new(Arc::new(CannedRunner::new(
        "x",
        Arc::new(AtomicBool::new(false)),
    )));
    let mut call = MessageContentBlock::ToolCall {
        id: "call_1".to_string(),
        name: "spawn_subagent".to_string(),
        arguments: json!({}),
    };
    // Missing `task`: the hook surfaces PluginError::Validation and leaves the
    // block untouched, because no spec could be parsed from the arguments.
    let err = plugin
        .on_stream_chunk(&mut call)
        .expect_err("missing task must fail the hook");
    assert!(
        matches!(err, PluginError::Validation { .. }),
        "expected Validation, got {err:?}"
    );
    assert!(
        matches!(&call, MessageContentBlock::ToolCall { name, .. } if name == "spawn_subagent"),
        "block must stay a ToolCall: {call:?}"
    );
    assert_eq!(plugin.spawn_count(), 0);
}

#[test]
fn collect_unknown_id_rendered_as_tool_result() {
    let plugin = SubagentPlugin::new(Arc::new(CannedRunner::new(
        "x",
        Arc::new(AtomicBool::new(false)),
    )));
    // A collect for an unknown id cannot be answered; the error text becomes
    // the ToolResult output so the parent model can react to it.
    let mut call = MessageContentBlock::ToolCall {
        id: "call_2".to_string(),
        name: "collect_subagent".to_string(),
        arguments: json!({ "subagent_id": "sub-999" }),
    };
    plugin
        .on_stream_chunk(&mut call)
        .expect("hook must not fail on an unknown id");
    match call {
        MessageContentBlock::ToolResult { output, .. } => {
            assert!(
                output.contains("unknown subagent id"),
                "output must carry the error text, got {output:?}"
            );
        }
        other => panic!("expected ToolResult, got {other:?}"),
    }
}

/// The live round trip must actually spawn a child, not merely coexist with the
/// plugin. An injector registered *before* the plugin rewrites the first live
/// `Text` chunk into a `spawn_subagent` call, so the real fan-out runs
/// mid-stream: the delivered block carries the new child's id, the spawn log
/// records it, and the child's result is then collected off the same runtime.
#[tokio::test]
async fn live_stream_chunk_spawns_and_collects_a_real_child() {
    if let Err(reason) = common::require_live_server() {
        eprintln!("SKIP: llama.cpp not reachable: {reason}");
        return;
    }
    let model = common::live_model();
    let done = Arc::new(AtomicBool::new(false));
    let injector = Arc::new(common::ToolCallInjector::new(
        "live-spawn-1",
        "spawn_subagent",
        json!({
            "name": "child",
            "task": "summarize the docs",
            "tool_scope": ["read"],
            "session_id": "live-sess",
        }),
    ));
    let plugin = Arc::new(SubagentPlugin::new(Arc::new(CannedRunner::new(
        "summarized live!",
        done.clone(),
    ))));
    let client = common::client_with_plugins(vec![
        injector.clone() as Arc<dyn CucaPlugin>,
        plugin.clone() as Arc<dyn CucaPlugin>,
    ]);
    let stream = client
        .generate_stream(common::live_request(
            "Reply with the single word: ok",
            &model,
        ))
        .await
        .expect("generate_stream must succeed");
    let blocks = common::drain_timeout(stream, 60).await;

    assert!(
        injector.injected(),
        "the live turn produced no model chunk to convert, so nothing was \
         exercised; got {blocks:?}"
    );
    let child_id = common::tool_result_output(&blocks, injector.call_id()).unwrap_or_else(|| {
        panic!(
            "the spawn call must be replaced by a ToolResult carrying the child id, got {blocks:?}"
        )
    });
    assert!(!child_id.trim().is_empty(), "the child id must be real");
    assert_eq!(
        plugin.spawn_count(),
        1,
        "the live stream hook must have spawned exactly one child"
    );
    let spawns = plugin.spawns();
    assert_eq!(spawns.len(), 1);
    assert_eq!(
        spawns[0].0.as_deref(),
        Some("live-sess"),
        "the injected spec's session id must reach the spawn log"
    );
    assert!(
        !blocks.iter().any(
            |b| matches!(b, MessageContentBlock::ToolCall { name, .. } if name == "spawn_subagent")
        ),
        "the plugin must consume the spawn call, not pass it through: {blocks:?}"
    );

    // The child was spawned onto this runtime; pump until the runner signals
    // delivery, because `collect` blocks the only executor thread.
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    while !done.load(Ordering::SeqCst) {
        assert!(
            tokio::time::Instant::now() < deadline,
            "the child spawned by the live stream never completed"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    let result = plugin
        .collect(&child_id)
        .unwrap_or_else(|e| panic!("collecting the live-spawned child failed: {e}"));
    assert_eq!(result.summary, "summarized live!");
    assert!(result.exit_ok);
}
