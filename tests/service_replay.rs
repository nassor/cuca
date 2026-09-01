//! Integration tests for the deterministic replay capability (`service-replay`).
//!
//! The deterministic tests record a trajectory through the session-log plugin's
//! [`CucaPlugin`] hooks over a temp-directory `FileBackend`, then replay it
//! through [`SessionReplay`] and pin the block sequence, the fork-point prefix,
//! the fork audit note, and the loud refusal of a corrupt file. The live tests
//! record a real llama.cpp turn and assert the replayed stream and the rebuilt
//! `UnifiedResponse` match what the provider produced.
//!
//! `SessionReplay` is never registered on a client: it is a service, not a
//! plugin, and implements no hook, so every entry point below is a plain method
//! call (the tier contract lives in `src/services/mod.rs`).
#![cfg(all(feature = "provider-llamacpp", feature = "service-replay"))]

mod common;

use std::sync::Arc;

use cuca::plugin::{CucaPlugin, SessionStorePlugin};
use cuca::types::{MessageContentBlock, ProviderEndpoint};
use cuca::{
    FileBackend, PluginError, ReplayNote, SessionBackend, SessionEvent, SessionLogPlugin,
    SessionRecord, SessionReplay, UnifiedRequest, UnifiedResponse,
};

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
        "cuca-replay-it-{label}-{}-{nanos}",
        std::process::id()
    )))
}

/// The two blocks every deterministic recording below streams in.
fn recorded_blocks() -> Vec<MessageContentBlock> {
    vec![
        MessageContentBlock::Text("hello".to_string()),
        MessageContentBlock::ToolCall {
            id: "call-1".to_string(),
            name: "search".to_string(),
            arguments: serde_json::json!({ "q": "cuca" }),
        },
    ]
}

fn canned_response() -> UnifiedResponse {
    UnifiedResponse {
        model: "test-model".into(),
        provider: ProviderEndpoint::LlamaCpp,
        duration_secs: 1.5,
        prompt_tokens: 10,
        completion_tokens: 5,
        finish_reason: Some("stop".into()),
        content: Vec::new(),
        prompt_cache_usage: None,
    }
}

/// Drive one whole turn through the plugin's hooks: request, blocks, terminal
/// accounting. This is exactly what a live client does to the plugin.
fn record_one_turn(plugin: &SessionLogPlugin) {
    let mut req = UnifiedRequest::new("test-model")
        .add_system_message("You are concise.")
        .add_user_message("say hello");
    plugin
        .on_request(&mut req)
        .expect("on_request must return Ok(())");
    for mut block in recorded_blocks() {
        plugin
            .on_stream_chunk(&mut block)
            .expect("on_stream_chunk must return Ok(())");
    }
    plugin
        .on_response_complete(&canned_response())
        .expect("on_response_complete must return Ok(())");
}

/// A recorded turn replays block for block, with its prompts, messages, and
/// terminal accounting on the same turn.
#[test]
fn file_backend_round_trip_replays_the_recorded_turn() {
    let guard = fresh_temp_dir("round-trip");
    let backend = Arc::new(FileBackend::new(&guard.0).expect("backend must open"));
    let plugin = SessionLogPlugin::new(backend).with_session_id("s1");
    record_one_turn(&plugin);

    // The replay seam is the backend handle the store already exposes.
    let replay = SessionReplay::new(Arc::clone(plugin.backend()));
    let trajectory = replay.load("s1").expect("replay must load");

    assert_eq!(trajectory.session_id(), "s1");
    assert_eq!(trajectory.len(), 1, "one generation is one turn");
    let turn = trajectory.turn(0).expect("the single turn must be there");
    assert_eq!(
        turn.blocks(),
        recorded_blocks().as_slice(),
        "the replayed blocks must invert on_stream_chunk exactly"
    );
    assert_eq!(turn.system_prompts(), ["You are concise.".to_string()]);
    assert_eq!(turn.messages().len(), 1, "one user message was recorded");
    assert!(
        turn.is_complete(),
        "the recorded turn carries its terminator"
    );
    let completion = turn.completion().expect("a complete turn has accounting");
    assert_eq!(completion.duration_ms, 1_500);
    assert_eq!(completion.prompt_tokens, 10);
    assert_eq!(completion.completion_tokens, 5);
    assert!(turn.notes().is_empty(), "nothing to annotate here");

    let usage = trajectory.usage();
    assert_eq!(usage.turns, 1);
    assert_eq!(usage.blocks, 2);
    assert_eq!(
        usage.records, 6,
        "prompt + message + 2 blocks + the Latency/TokenUsage terminator pair"
    );
    assert!(!usage.near_cap, "6 records is nowhere near the default cap");
}

/// Durability plus determinism across process-level state: a brand-new backend
/// over the same directory replays the identical block sequence.
#[test]
fn a_fresh_file_backend_over_the_same_dir_replays_identically() {
    let guard = fresh_temp_dir("fresh");
    let dir = guard.0.clone();

    let first_blocks = {
        let plugin =
            SessionLogPlugin::new(Arc::new(FileBackend::new(&dir).expect("backend must open")))
                .with_session_id("s1");
        record_one_turn(&plugin);
        SessionReplay::new(Arc::clone(plugin.backend()))
            .load("s1")
            .expect("replay must load")
            .turn(0)
            .expect("the single turn must be there")
            .blocks()
            .to_vec()
    };

    // Nothing of the first backend, plugin, or trajectory survives here.
    let reopened: Arc<dyn SessionBackend> =
        Arc::new(FileBackend::new(&dir).expect("backend must reopen"));
    let trajectory = SessionReplay::new(reopened)
        .load("s1")
        .expect("replay must load from disk");
    assert_eq!(
        trajectory
            .turn(0)
            .expect("the single turn must be there")
            .blocks(),
        first_blocks.as_slice(),
        "two loads over the same file produce identical block sequences"
    );
    assert_eq!(trajectory.usage().blocks, 2);
}

/// A `load_prefix` at a fork point and the forked session itself replay the
/// same blocks: `FileBackend::fork` writes exactly that prefix under the new id.
#[test]
fn fork_point_prefix_matches_the_forked_session() {
    let guard = fresh_temp_dir("fork-prefix");
    let backend = Arc::new(FileBackend::new(&guard.0).expect("backend must open"));
    let plugin = SessionLogPlugin::new(Arc::clone(&backend) as Arc<dyn SessionBackend>);
    for i in 0..5 {
        plugin
            .append_log(
                "orig",
                &SessionRecord::new(
                    "orig",
                    SessionEvent::Output {
                        text: format!("o{i}"),
                    },
                ),
            )
            .expect("append must succeed");
    }

    let replay = SessionReplay::new(Arc::clone(&backend) as Arc<dyn SessionBackend>);
    let prefix = replay
        .load_prefix("orig", 2)
        .expect("the prefix must load")
        .turn(0)
        .expect("the trailing turn must be there")
        .blocks()
        .to_vec();
    assert_eq!(
        prefix,
        vec![
            MessageContentBlock::Text("o0".to_string()),
            MessageContentBlock::Text("o1".to_string()),
            MessageContentBlock::Text("o2".to_string()),
        ]
    );

    // The same position addressed as a point_id agrees with the prefix.
    assert_eq!(
        replay
            .load_at_point("orig:2")
            .expect("the point load must succeed")
            .turn(0)
            .expect("the trailing turn must be there")
            .blocks(),
        prefix.as_slice()
    );

    let branch_id = plugin
        .fork_session("orig", "orig:2")
        .expect("fork must succeed");
    let branch = replay.load(&branch_id).expect("the branch must load");
    assert_eq!(branch.session_id(), branch_id);
    assert_eq!(
        branch
            .turn(0)
            .expect("the trailing turn must be there")
            .blocks(),
        prefix.as_slice(),
        "the forked session replays the same blocks as the prefix load"
    );
}

/// The fork audit record on the original surfaces as a note, never as a block,
/// so a post-fork replay of the original has an unchanged block sequence.
#[test]
fn fork_audit_record_appears_as_a_note_on_the_original() {
    let guard = fresh_temp_dir("fork-note");
    let backend = Arc::new(FileBackend::new(&guard.0).expect("backend must open"));
    let plugin = SessionLogPlugin::new(Arc::clone(&backend) as Arc<dyn SessionBackend>)
        .with_session_id("orig");
    record_one_turn(&plugin);

    let replay = SessionReplay::new(Arc::clone(&backend) as Arc<dyn SessionBackend>);
    let before = replay.load("orig").expect("replay must load");
    assert_eq!(before.usage().blocks, 2);

    let branch_id = plugin
        .fork_session("orig", "orig:1")
        .expect("fork must succeed");

    let after = replay
        .load("orig")
        .expect("replay must load after the fork");
    assert_eq!(
        after.usage().blocks,
        2,
        "the audit record must not add a block"
    );
    let notes: Vec<&ReplayNote> = after.turns().iter().flat_map(|t| t.notes()).collect();
    assert_eq!(
        notes,
        vec![&ReplayNote::Fork {
            from_point: "orig:1".to_string(),
            to_session: branch_id,
        }],
        "the branch must stay visible as exactly one note"
    );
}

/// A torn frame is refused loudly with the backend's own `cslog` schema, never
/// replayed as a shortened trajectory.
#[test]
fn corrupt_cslog_is_refused_loudly() {
    let guard = fresh_temp_dir("corrupt");
    let dir = guard.0.clone();
    let plugin =
        SessionLogPlugin::new(Arc::new(FileBackend::new(&dir).expect("backend must open")))
            .with_session_id("s1");
    record_one_turn(&plugin);

    // Drop the final delimiter and one payload byte: a crash mid-write.
    let path = dir.join("s1.cslog");
    let bytes = std::fs::read(&path).expect("the session file must exist");
    std::fs::write(&path, &bytes[..bytes.len() - 2]).expect("the truncation must be written");

    let replay = SessionReplay::new(Arc::new(
        FileBackend::new(&dir).expect("backend must reopen"),
    ));
    let err = match replay.load("s1") {
        Err(err) => err,
        Ok(trajectory) => panic!(
            "a torn frame must be refused, replayed {} turns instead",
            trajectory.len()
        ),
    };
    match &err {
        PluginError::Validation { schema, message } => {
            assert_eq!(schema, "cslog");
            assert!(message.contains("record frame"), "{message}");
        }
        other => panic!("expected PluginError::Validation, got {other:?}"),
    }
}

/// A live turn's recorded trajectory replays to the same block sequence the
/// provider streamed, with no second dispatch.
#[tokio::test]
async fn live_stream_replays_to_the_same_blocks() {
    use tokio_stream::StreamExt;

    if let Err(reason) = common::require_live_server() {
        eprintln!("SKIP: llama.cpp not reachable: {reason}");
        return;
    }
    let plugin = Arc::new(SessionLogPlugin::new_in_memory().with_session_id("live"));
    let client = common::client_with_plugins(vec![Arc::clone(&plugin) as Arc<dyn CucaPlugin>]);

    let request = common::live_request("Reply with the single word: ok", &common::live_model());
    let stream = client
        .generate_stream(request)
        .await
        .expect("generate_stream must start");
    let live_blocks = common::drain_timeout(stream, 60).await;
    assert!(
        live_blocks
            .iter()
            .any(|b| matches!(b, MessageContentBlock::Text(_))),
        "expected at least one Text block, got {live_blocks:?}"
    );

    let trajectory = SessionReplay::new(Arc::clone(plugin.backend()))
        .load("live")
        .expect("the recorded session must replay");
    assert_eq!(trajectory.len(), 1, "one live turn is one recorded turn");

    let mut stream = trajectory
        .stream_turn(0)
        .expect("the recorded turn must stream");
    let mut replayed = Vec::new();
    while let Some(item) = stream.next().await {
        replayed.push(item.expect("a replayed item is never an error"));
    }
    // Only `ImageBase64` is unrecordable, and a text turn never emits one.
    assert_eq!(
        replayed, live_blocks,
        "the replay must reproduce the live block sequence exactly"
    );
}

/// `ReplayTurn::response` reports the token counts and latency
/// `on_response_complete` recorded for the live turn.
#[tokio::test]
async fn live_turn_response_matches_recorded_usage() {
    if let Err(reason) = common::require_live_server() {
        eprintln!("SKIP: llama.cpp not reachable: {reason}");
        return;
    }
    let plugin = Arc::new(SessionLogPlugin::new_in_memory().with_session_id("usage"));
    let client = common::client_with_plugins(vec![Arc::clone(&plugin) as Arc<dyn CucaPlugin>]);

    let model = common::live_model();
    let stream = client
        .generate_stream(common::live_request(
            "Reply with the single word: ok",
            &model,
        ))
        .await
        .expect("generate_stream must start");
    let live_blocks = common::drain_timeout(stream, 60).await;

    let trajectory = SessionReplay::new(Arc::clone(plugin.backend()))
        .load("usage")
        .expect("the recorded session must replay");
    let turn = trajectory.turn(0).expect("the live turn must be there");
    assert!(
        turn.is_complete(),
        "on_response_complete appended the terminator pair"
    );
    let completion = turn.completion().expect("a complete turn has accounting");

    let response = turn.response(&model, ProviderEndpoint::LlamaCpp);
    assert_eq!(response.model, model);
    assert_eq!(response.provider, ProviderEndpoint::LlamaCpp);
    assert_eq!(response.prompt_tokens, completion.prompt_tokens);
    assert_eq!(response.completion_tokens, completion.completion_tokens);
    assert!(
        response.content.len() == live_blocks.len(),
        "the rebuilt content must carry every recorded block"
    );
    assert_eq!(
        response.duration_secs,
        completion.duration_ms as f64 / 1000.0,
        "the rebuilt duration comes from the recorded Latency"
    );
    assert_eq!(
        response.content, live_blocks,
        "the rebuilt content is the recorded block sequence"
    );
    assert_eq!(
        response.finish_reason, None,
        "no SessionEvent records a stop reason"
    );
}
