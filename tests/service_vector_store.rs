//! Integration tests for bounded vector recall (`service-vector-store`).
//!
//! The deterministic tests drive the real seam rather than a fake: a
//! [`MemoryPlugin`] built with [`MemoryPlugin::with_extensions`] runs
//! `CompactionStrategy::Offload` out of band through
//! [`MemoryPlugin::compress`], the store receives the removed turns, and the
//! application completes the mandatory hand-off with
//! [`RetrievalReport::inject`]. They also pin the two failure contracts the
//! plugin and the store share: a rejected batch leaves the message list
//! byte-identical and lands in `CompressionReport::last_error`, and the
//! rejection is recoverable, because a later strategy clamps the offending turn
//! and the next pass offloads it. None of them touch the network.
//!
//! The live test proves route acceptance only: it offloads real history, asks
//! the store for the turns matching a follow-up question, injects the recall,
//! and dispatches through llama.cpp. It asserts that a `Text` block comes back,
//! never what the model said about it.
#![cfg(all(feature = "provider-llamacpp", feature = "service-vector-store"))]

mod common;

use std::sync::Arc;

use cuca::plugins::memory::VectorStore;
use cuca::types::{MessageContentBlock, MessageRole, UnifiedMessage};
use cuca::{
    CompactionStrategy, CompressionAction, Embedder, InMemoryVectorStore, MemoryConfig,
    MemoryPlugin, PluginError, RECALL_RENDER_MARKER, RecallInjection, Summarizer, UnifiedRequest,
    VectorStoreConfig,
};

/// Embedding width used throughout: wide enough for the hashing bag of words
/// to keep distinct sentences apart, narrow enough to stay cheap.
const DIMENSIONS: usize = 256;

/// The question every recall test asks; the answer is offloaded, not live.
const QUESTION: &str = "where does the deploy token live?";

/// The fact the recall must bring back.
const FACT: &str = "the deploy token lives in vault slot 7";

/// `with_extensions` is the only constructor that accepts a store, so a
/// store-only test still has to supply a summarizer. Every config below omits
/// `CompactionStrategy::Summarize`, so this is never called.
struct NoSummarizer;

impl Summarizer for NoSummarizer {
    fn summarize(&self, _turns: &[UnifiedMessage]) -> String {
        String::new()
    }
}

/// Deterministic hashing bag of words.
///
/// FNV-1a over lowercased ASCII alphanumeric tokens, bucketed into
/// [`DIMENSIONS`] with unit weights. `DefaultHasher` is deliberately avoided:
/// its seed is per-process random, which would make recall order differ between
/// runs of the same test.
struct HashEmbedder;

impl Embedder for HashEmbedder {
    fn embed(&self, text: &str) -> Result<Vec<f32>, PluginError> {
        let mut vector = vec![0.0f32; DIMENSIONS];
        for token in text
            .split(|c: char| !c.is_ascii_alphanumeric())
            .filter(|token| !token.is_empty())
        {
            let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
            for byte in token.bytes() {
                hash ^= u64::from(byte.to_ascii_lowercase());
                hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
            }
            vector[(hash % DIMENSIONS as u64) as usize] += 1.0;
        }
        Ok(vector)
    }
}

/// Always fails, so the offload rollback path is exercised end to end.
struct FailingEmbedder;

impl Embedder for FailingEmbedder {
    fn embed(&self, _text: &str) -> Result<Vec<f32>, PluginError> {
        Err(PluginError::Internal("embedder offline".to_string()))
    }
}

fn store_with(
    embedder: Arc<dyn Embedder>,
    max_entries: usize,
    max_entry_bytes: usize,
) -> Arc<InMemoryVectorStore> {
    let config = VectorStoreConfig::new(max_entries, DIMENSIONS, max_entry_bytes)
        .expect("config must build")
        .with_warn_fraction(0.8)
        .expect("warn fraction must be accepted");
    Arc::new(InMemoryVectorStore::new(config, embedder).expect("store must build"))
}

fn store(max_entries: usize) -> Arc<InMemoryVectorStore> {
    store_with(Arc::new(HashEmbedder), max_entries, 64 * 1024)
}

fn memory_with(
    store: Arc<InMemoryVectorStore>,
    strategies: Vec<CompactionStrategy>,
) -> MemoryPlugin {
    MemoryPlugin::with_extensions(
        MemoryConfig {
            strategies,
            ..Default::default()
        },
        Arc::new(NoSummarizer),
        store as Arc<dyn VectorStore>,
    )
    .expect("memory plugin must build")
}

/// One System message, six removable turns, and a final User question the
/// plugin must never remove.
fn history() -> Vec<UnifiedMessage> {
    vec![
        UnifiedMessage::system("You are concise."),
        UnifiedMessage::user(FACT),
        UnifiedMessage::assistant("Noted: the deploy token is in vault slot 7."),
        UnifiedMessage::user("the staging cluster is named borealis"),
        UnifiedMessage::assistant("Noted: staging is borealis."),
        UnifiedMessage::user("the on-call rotation starts on monday"),
        UnifiedMessage::assistant("Noted: on-call starts monday."),
        UnifiedMessage::user(QUESTION),
    ]
}

fn joined_text(message: &UnifiedMessage) -> String {
    message
        .content
        .iter()
        .filter_map(|block| match block {
            MessageContentBlock::Text(text) => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// The seam itself: `CompactionStrategy::Offload` removes the oldest turns from
/// the live prompt and the store is where they land.
#[test]
fn offload_writes_the_oldest_turns_into_the_store() {
    let store = store(64);
    let memory = memory_with(
        Arc::clone(&store),
        vec![CompactionStrategy::Offload { turns: 4 }],
    );
    let mut messages = history();
    let before = messages.len();

    let report = memory.compress(&mut messages).expect("compress must run");

    assert!(
        report.actions.contains(&CompressionAction::Offloaded),
        "the offload strategy must have acted: {report:?}"
    );
    assert_eq!(report.last_error, None);
    assert_eq!(messages.len(), before - 4);
    assert_eq!(store.len().expect("len must read"), 4);
    assert_eq!(messages[0].role, MessageRole::System);
    assert_eq!(
        joined_text(messages.last().expect("a turn survives")),
        QUESTION
    );
}

/// The whole hand-off: offloaded history is retrieved by similarity and reaches
/// the next request as a recall system message placed against the newest user
/// question.
#[test]
fn retrieved_offloaded_turn_reaches_the_next_request() {
    let store = store(64);
    let memory = memory_with(
        Arc::clone(&store),
        vec![CompactionStrategy::Offload { turns: 4 }],
    );
    let mut messages = history();
    memory.compress(&mut messages).expect("compress must run");
    assert!(
        !messages.iter().any(|m| joined_text(m) == FACT),
        "the fact must be out of the live prompt before recall"
    );

    let recall = store.retrieve(QUESTION, 2).expect("retrieval must run");
    assert_eq!(recall.scanned, 4);
    assert_eq!(
        joined_text(&recall.turns[0].message),
        FACT,
        "the matching turn must rank first: {recall:?}"
    );

    let mut request = UnifiedRequest::new("recall-model");
    request.messages = messages;
    assert_eq!(recall.inject(&mut request), RecallInjection::Inserted);

    let recall_index = request
        .messages
        .iter()
        .position(|m| joined_text(m).starts_with(RECALL_RENDER_MARKER))
        .expect("the recall message must be in the request");
    assert_eq!(request.messages[recall_index].role, MessageRole::System);
    assert!(
        joined_text(&request.messages[recall_index]).contains("vault slot 7"),
        "the recalled fact must be visible to the model: {:?}",
        joined_text(&request.messages[recall_index])
    );
    assert_eq!(
        joined_text(&request.messages[recall_index + 1]),
        QUESTION,
        "recall sits immediately before the newest user message"
    );
}

/// The offload contract is all-or-nothing in both directions: the plugin puts
/// every turn back and the store keeps nothing.
#[test]
fn a_failing_embedder_leaves_history_intact_and_records_the_error() {
    let store = store_with(Arc::new(FailingEmbedder), 64, 64 * 1024);
    let memory = memory_with(
        Arc::clone(&store),
        vec![CompactionStrategy::Offload { turns: 4 }],
    );
    let mut messages = history();
    let before = messages.clone();

    let report = memory.compress(&mut messages).expect("compress must run");

    assert!(
        !report.actions.contains(&CompressionAction::Offloaded),
        "a failed offload must not be reported as an action: {report:?}"
    );
    let error = report.last_error.expect("the failure must be recorded");
    assert!(error.contains("embedder offline"), "{error}");
    assert_eq!(messages, before, "history must be restored exactly");
    assert!(store.is_empty().expect("is_empty must read"));
}

/// A rejection is recoverable, not terminal: the clamp strategy later in the
/// same menu shrinks the offending turn, and the next pass offloads it.
#[test]
fn an_oversized_turn_is_rejected_then_offloaded_after_clamping() {
    let store = store_with(Arc::new(HashEmbedder), 64, 3_000);
    let memory = memory_with(
        Arc::clone(&store),
        vec![
            CompactionStrategy::Offload { turns: 4 },
            CompactionStrategy::ClampOversizedMessages {
                max_part_tokens: 64,
            },
        ],
    );
    let mut messages = history();
    messages.insert(
        1,
        UnifiedMessage {
            role: MessageRole::Tool,
            content: vec![MessageContentBlock::ToolResult {
                tool_call_id: "call-1".to_string(),
                output: "audit line ".repeat(400),
            }],
            name: None,
            tool_call_id: Some("call-1".to_string()),
        },
    );

    let first = memory.compress(&mut messages).expect("compress must run");
    assert!(
        first.last_error.is_some(),
        "the oversized turn must be rejected first: {first:?}"
    );
    assert!(
        first.actions.contains(&CompressionAction::ClampedParts),
        "the clamp strategy must still run after the rejection: {first:?}"
    );
    assert!(store.is_empty().expect("is_empty must read"));

    let second = memory.compress(&mut messages).expect("compress must run");
    assert_eq!(second.last_error, None, "the clamped turn now fits");
    assert!(second.actions.contains(&CompressionAction::Offloaded));
    assert_eq!(store.len().expect("len must read"), 4);
}

/// The cap is real and observable: offloading more turns than the store holds
/// evicts the oldest and reports it.
#[test]
fn capacity_eviction_is_visible_through_usage() {
    let store = store(2);
    let memory = memory_with(
        Arc::clone(&store),
        vec![CompactionStrategy::Offload { turns: 6 }],
    );
    let mut messages = history();

    memory.compress(&mut messages).expect("compress must run");

    let usage = store.usage().expect("usage must read");
    assert_eq!(usage.entries, usage.capacity);
    assert_eq!(usage.entries, 2);
    assert_eq!(usage.evicted_entries, 4);
    assert!(usage.near_cap, "a full store is past warn_fraction");
    assert!((usage.fraction - 1.0).abs() < f32::EPSILON);
}

/// Route acceptance: a request carrying an injected recall message is a
/// well-formed prompt for a real server. The assertion is that text comes back,
/// never what the text says.
#[tokio::test]
async fn live_recall_injection_is_accepted_by_llamacpp() {
    if let Err(reason) = common::require_live_server() {
        eprintln!("SKIP: llama.cpp not reachable: {reason}");
        return;
    }
    let model = common::live_model();
    let store = store(64);
    let memory = memory_with(
        Arc::clone(&store),
        vec![CompactionStrategy::Offload { turns: 4 }],
    );
    let mut messages = history();
    memory.compress(&mut messages).expect("compress must run");

    let recall = store.retrieve(QUESTION, 2).expect("retrieval must run");
    let mut request = common::live_request(QUESTION, &model);
    request.messages = messages;
    assert_eq!(recall.inject(&mut request), RecallInjection::Inserted);

    let blocks = common::drain_timeout(
        common::client()
            .generate_stream(request)
            .await
            .expect("generate_stream must start"),
        60,
    )
    .await;

    assert!(
        blocks
            .iter()
            .any(|block| matches!(block, MessageContentBlock::Text(_))),
        "the server must accept the recall-carrying prompt and answer: {blocks:?}"
    );
}
