//! Offload old turns into a bounded vector store, then recall them into the
//! next prompt.
//!
//! Stage 1 runs one real llama.cpp turn over a scripted history that contains a
//! distinctive fact. Stage 2 calls `MemoryPlugin::compress` out of band, so
//! `CompactionStrategy::Offload` removes the oldest turns from the live prompt
//! and hands them to `InMemoryVectorStore` through the `VectorStore` seam.
//! Stage 3 asks the store for the turns closest to a follow-up question, and
//! stage 4 performs the mandatory hand-off: `RetrievalReport::inject` puts the
//! recall back into the request as a system message, and the second live turn
//! answers a question whose evidence is no longer in the conversation.
//!
//! # Prerequisites
//!
//! - A checkout of this repository (the example builds from this crate).
//! - A running [llama.cpp](https://github.com/ggml-org/llama.cpp) server
//!   (`llama-server`) on port 1234 with the demo model loaded.
//!
//! # Run
//!
//! ```sh
//! cargo run --example vector_store --features provider-llamacpp,service-vector-store
//! ```
//!
//! # Configuration
//!
//! Both values default to a local llama.cpp server; override them to target
//! any OpenAI-compatible server:
//!
//! - `CUCA_BASE_URL`: server base URL, defaults to `http://127.0.0.1:1234/v1`.
//! - `CUCA_MODEL`: upstream model id, defaults to `google/gemma-4-e4b`.
//!
//! Example: `CUCA_BASE_URL=http://127.0.0.1:8000/v1 CUCA_MODEL=<server-model-id> cargo run --example vector_store --features provider-llamacpp,service-vector-store`
//!
//! # Output
//!
//! The shape, from one run against `google/gemma-4-12b-qat` on llama.cpp:
//!
//! ```text
//! Stage 1: one live turn over a scripted 8-message history
//!   reply: Vault slot 7.
//!   blocks: 5 text, 97 thinking
//!
//! Stage 2: compaction out of band (MemoryPlugin::compress)
//!   actions: [Offloaded]
//!   messages: 8 -> 2, tokens: 71 -> 13
//!   store: 6/64 entries, fraction 0.09, near cap false, evicted 0
//!
//! Stage 3: recall for "where does the deploy token live?"
//!   scanned 6 entries
//!   [0.4330] the deploy token lives in vault slot 7
//!   [0.4082] Noted: the deploy token is in vault slot 7.
//!
//! Stage 4: the follow-up turn, with the recall injected
//!   injection: Inserted
//!   prompt now carries: "CUCA recall: 2 offloaded turn(s), best first"
//!   reply: It lives in vault slot 7.
//!   blocks: 8 text, 114 thinking
//! ```
//!
//! Stage 4 is the point of the demo: by then the live prompt holds two
//! messages, so the only place the model can read the vault slot is the
//! injected recall.
//!
//! The replies and the block counts depend on the model. The scores depend on
//! the embedder. The offload counts depend on neither: `Offload { turns: 6 }`
//! over this history always moves six turns, because only the first System
//! message and the most recent User message are protected, which leaves two
//! messages in the live prompt.
//!
//! With no server on the base URL, the program prints one line naming the
//! address and exits successfully.
//!
//! # Why a service and not a plugin hook?
//!
//! `InMemoryVectorStore` is a service, not a `CucaPlugin`. Its write half is
//! already driven by a hook the memory plugin owns, and its read half has no
//! hook to live in: recall is a decision about *this* request that the
//! application makes, and no hook signature can return a ranked set of turns to
//! the caller. Stage 4 is the reason the rule exists. `RetrievalReport` is inert
//! until `inject` is called, so a "plugin" wrapper would either do nothing or
//! silently rewrite every prompt.
//!
//! The embedder is caller-supplied and synchronous for the same structural
//! reason: `store_turns` runs inside the synchronous `on_request` hook. The
//! `HashEmbedder` below is a deterministic bag of words, good enough to rank a
//! scripted history and dependency-free; a real deployment substitutes an
//! embedding model behind the same one-method trait.

use std::sync::Arc;

use cuca::plugins::memory::VectorStore;
use cuca::types::{MessageContentBlock, ProviderEndpoint, UnifiedMessage};
use cuca::{
    AgentResponseStream, CompactionStrategy, CucaClient, Embedder, InMemoryVectorStore,
    MemoryConfig, MemoryPlugin, PluginError, RECALL_RENDER_MARKER, Summarizer, UnifiedRequest,
    VectorStoreConfig,
};
use tokio_stream::StreamExt;

/// Embedding width; wide enough that a hashing bag of words keeps these
/// sentences apart, small enough to stay a toy.
const DIMENSIONS: usize = 256;

/// The follow-up question, whose answer is only in the offloaded turns.
const QUESTION: &str = "where does the deploy token live?";

/// Deterministic hashing bag of words: FNV-1a over lowercased ASCII
/// alphanumeric tokens, bucketed into [`DIMENSIONS`] with unit weights.
///
/// `DefaultHasher` is deliberately avoided: its seed is randomized per process,
/// so the same history would rank differently on every run.
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

/// `MemoryPlugin::with_extensions` is the only constructor that accepts a
/// store, so a store-only demo still supplies a summarizer. The strategy list
/// below omits `Summarize`, so this is never called.
struct NoSummarizer;

impl Summarizer for NoSummarizer {
    fn summarize(&self, _turns: &[UnifiedMessage]) -> String {
        String::new()
    }
}

/// Concatenated text of a message, for printing.
fn text_of(message: &UnifiedMessage) -> String {
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

/// Drain a turn into its text plus per-kind block counts.
///
/// A reasoning model emits one `Thinking` block per token, so printing every
/// block buries the four lines this demo is about. The counts stay in the
/// output because they are the honest shape of a live turn.
async fn drain(mut stream: AgentResponseStream) -> (String, usize, usize) {
    let mut text = String::new();
    let mut text_blocks = 0usize;
    let mut thinking_blocks = 0usize;
    while let Some(chunk) = stream.next().await {
        match chunk {
            Ok(MessageContentBlock::Text(chunk_text)) => {
                text_blocks += 1;
                text.push_str(&chunk_text);
            }
            Ok(MessageContentBlock::Thinking { .. }) => thinking_blocks += 1,
            Ok(_) => {}
            Err(error) => {
                println!("  the stream ended early: {error}");
                break;
            }
        }
    }
    (text, text_blocks, thinking_blocks)
}

/// The scripted history: one System message, six removable turns, and the
/// follow-up question the plugin must never remove.
fn history() -> Vec<UnifiedMessage> {
    vec![
        UnifiedMessage::system("You are concise."),
        UnifiedMessage::user("the deploy token lives in vault slot 7"),
        UnifiedMessage::assistant("Noted: the deploy token is in vault slot 7."),
        UnifiedMessage::user("the staging cluster is named borealis"),
        UnifiedMessage::assistant("Noted: staging is borealis."),
        UnifiedMessage::user("the on-call rotation starts on monday"),
        UnifiedMessage::assistant("Noted: on-call starts monday."),
        UnifiedMessage::user(QUESTION),
    ]
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Base URL and model come from the environment so the example runs
    // against any OpenAI-compatible server; the defaults target a local
    // llama.cpp server (see the module docs for the override recipe).
    let base_url =
        std::env::var("CUCA_BASE_URL").unwrap_or_else(|_| "http://127.0.0.1:1234/v1".to_string());
    let model = std::env::var("CUCA_MODEL").unwrap_or_else(|_| "google/gemma-4-e4b".to_string());

    let store = Arc::new(InMemoryVectorStore::new(
        VectorStoreConfig::new(64, DIMENSIONS, 16 * 1024)?.with_warn_fraction(0.8)?,
        Arc::new(HashEmbedder),
    )?);
    let memory = MemoryPlugin::with_extensions(
        MemoryConfig {
            strategies: vec![CompactionStrategy::Offload { turns: 6 }],
            ..Default::default()
        },
        Arc::new(NoSummarizer),
        Arc::clone(&store) as Arc<dyn VectorStore>,
    )?;

    let client = CucaClient::builder()
        .with_provider(ProviderEndpoint::LlamaCpp)
        .with_base_url(base_url.clone())
        .build()?;

    // Stage 1: a real turn over the full history, so what gets offloaded is a
    // conversation the model actually saw.
    let mut messages = history();
    println!(
        "Stage 1: one live turn over a scripted {}-message history",
        messages.len()
    );
    let mut request = UnifiedRequest::new(&model).set_max_tokens(128);
    request.messages = messages.clone();
    let stream = match client.generate_stream(request).await {
        Ok(stream) => stream,
        Err(error) => {
            println!("\nNo server answered at {base_url}: {error}");
            println!("Start llama-server there, or set CUCA_BASE_URL, then run this again.");
            return Ok(());
        }
    };
    let (reply, text_blocks, thinking_blocks) = drain(stream).await;
    println!("  reply: {}", reply.trim());
    println!("  blocks: {text_blocks} text, {thinking_blocks} thinking");

    // Stage 2: compaction is an explicit, out-of-band call here. That is the
    // documented order for recall: compress, then retrieve and inject, then
    // dispatch. A compaction pass running *after* injection could drop the
    // recall message itself.
    println!("\nStage 2: compaction out of band (MemoryPlugin::compress)");
    let before = messages.len();
    let report = memory.compress(&mut messages)?;
    println!("  actions: {:?}", report.actions);
    println!(
        "  messages: {before} -> {}, tokens: {} -> {}",
        messages.len(),
        report.tokens_before,
        report.tokens_after
    );
    if let Some(error) = &report.last_error {
        println!("  a strategy reported: {error}");
    }
    let usage = store.usage()?;
    println!(
        "  store: {}/{} entries, fraction {:.2}, near cap {}, evicted {}",
        usage.entries, usage.capacity, usage.fraction, usage.near_cap, usage.evicted_entries
    );

    // Stage 3: the read half is a direct call, never a hook.
    println!("\nStage 3: recall for {QUESTION:?}");
    let recall = store.retrieve(QUESTION, 2)?;
    println!("  scanned {} entries", recall.scanned);
    for hit in &recall.turns {
        println!("  [{:.4}] {}", hit.score, text_of(&hit.message));
    }

    // Stage 4: the hand-off. Without this call the report is inert and the
    // recalled turns never reach the model.
    println!("\nStage 4: the follow-up turn, with the recall injected");
    let mut request = UnifiedRequest::new(&model).set_max_tokens(128);
    request.messages = messages;
    let injection = recall.inject(&mut request);
    println!("  injection: {injection:?}");
    if let Some(line) = request
        .messages
        .iter()
        .map(text_of)
        .find(|text| text.starts_with(RECALL_RENDER_MARKER))
        .and_then(|text| text.lines().next().map(str::to_string))
    {
        println!("  prompt now carries: {line:?}");
    }
    let (reply, text_blocks, thinking_blocks) = drain(client.generate_stream(request).await?).await;
    println!("  reply: {}", reply.trim());
    println!("  blocks: {text_blocks} text, {thinking_blocks} thinking");

    Ok(())
}
