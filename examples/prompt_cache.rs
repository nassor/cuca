//! Serve a repeated turn from the client cache, then hand the cache to a
//! second client that has no server to talk to.
//!
//! Stage 1 sends one turn through a client carrying a `PromptCache` and times
//! it. Stage 2 sends the byte-identical turn again: the digest matches, the
//! stored blocks replay, and the timing plus the hook counters show that no
//! provider was involved. Stage 3 changes one word, which changes the digest
//! and costs a second real dispatch. Stage 4 exports the cache with
//! `prompt_cache_snapshot`, imports it into a client whose base URL is a
//! closed port, and replays the first turn there: a hit answers, and a
//! never-cached turn fails on the connection the hit never opened.
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
//! cargo run --example prompt_cache --features provider-llamacpp,service-prompt-cache
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
//! Example: `CUCA_BASE_URL=http://127.0.0.1:8000/v1 CUCA_MODEL=<server-model-id> cargo run --example prompt_cache --features provider-llamacpp,service-prompt-cache`
//!
//! # Output
//!
//! One run against `google/gemma-4-12b-qat` on llama.cpp:
//!
//! ```text
//! Cache: 0/64 entries, ttl 300 s
//! Key of the turn about to run: 5fe40816c33c86f1...
//!
//! Stage 1: the first turn (a miss)
//!   reply: A tokenizer is a tool that breaks down text into smaller units, such as words, characters, or subwords, to make it processable by machine learning models.
//!   28564 ms, 33 text blocks, 291 thinking blocks
//!   hooks: on_request 1, on_stream_chunk 324, on_response_complete 1
//!   cache: 1/64 entries
//!
//! Stage 2: the identical turn (a hit)
//!   reply: A tokenizer is a tool that breaks down text into smaller units, such as words, characters, or subwords, to make it processable by machine learning models.
//!   0 ms, 33 text blocks, 291 thinking blocks
//!   hooks: on_request 2, on_stream_chunk 324, on_response_complete 2
//!   identical to stage 1: true
//!
//! Stage 3: one word changed, so the digest changes
//!   key: ef2c17e471bfb101...
//!   reply: An embedding is a numerical representation of data, such as words or images, in a high-dimensional vector space where similar items are positioned closer together.
//!   36273 ms, 2/64 entries
//!
//! Stage 4: the snapshot, imported into a client with no reachable server
//!   exported 2 entries, imported 2, expired 0, evicted 0
//!   base URL: http://127.0.0.1:1/v1
//!   replayed stage 1 in 0 ms: A tokenizer is a tool that breaks down text into smaller units, such as words, characters, or subwords, to make it processable by machine learning models.
//!   an uncached turn on that client: transport failure: error sending request for url (http://127.0.0.1:1/v1/chat/completions)
//! ```
//!
//! Stage 4 is the point of the demo. The second client cannot reach any
//! server, so the only thing that can answer is the imported cache, and the
//! uncached turn shows what happens when the digest misses.
//!
//! The replies, the block counts, and the two dispatch timings depend on the
//! model. Everything else does not: a hit is always served without a socket,
//! `on_stream_chunk` never advances on a hit, and `on_response_complete` runs
//! once per turn either way.
//!
//! With no server on the base URL, the program prints one line naming the
//! address and exits successfully.
//!
//! # Why a service and not a plugin hook?
//!
//! `PromptCache` is a service, not a `CucaPlugin`. Its lookup has to happen
//! after every `on_request` hook has finished mutating the request, since the
//! digest covers the effective request, and it has to be able to *replace*
//! the provider dispatch. No hook signature can do either: `on_request`
//! returns `Result<(), PluginError>`, so it can refuse a turn but never
//! answer one. The cache is wired on the builder and read by
//! `CucaClient::generate_stream` itself.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use cuca::plugin::CucaPlugin;
use cuca::services::prompt_cache::digest_request;
use cuca::types::{MessageContentBlock, ProviderEndpoint};
use cuca::{
    AgentResponseStream, CucaClient, PluginError, PromptCacheConfig, UnifiedRequest,
    UnifiedResponse,
};
use tokio_stream::StreamExt;

/// A base URL nothing listens on, so stage 4 can only be answered by the
/// imported cache.
const CLOSED_PORT: &str = "http://127.0.0.1:1/v1";

/// Counts the three hooks a turn can run, so a hit and a miss can be told
/// apart from the caller's side.
#[derive(Default)]
struct HookCounters {
    requests: AtomicUsize,
    chunks: AtomicUsize,
    completions: AtomicUsize,
}

impl CucaPlugin for HookCounters {
    fn name(&self) -> &'static str {
        "hook-counters"
    }

    fn on_request(&self, _request: &mut UnifiedRequest) -> Result<(), PluginError> {
        self.requests.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    fn on_stream_chunk(&self, _chunk: &mut MessageContentBlock) -> Result<(), PluginError> {
        self.chunks.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    fn on_response_complete(&self, _response: &UnifiedResponse) -> Result<(), PluginError> {
        self.completions.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
}

impl HookCounters {
    fn reading(&self) -> String {
        format!(
            "on_request {}, on_stream_chunk {}, on_response_complete {}",
            self.requests.load(Ordering::Relaxed),
            self.chunks.load(Ordering::Relaxed),
            self.completions.load(Ordering::Relaxed)
        )
    }
}

/// The turn that gets repeated. `temperature` is pinned because a cache is
/// only interesting when the same question really is the same request.
fn question(model: &str, topic: &str) -> UnifiedRequest {
    UnifiedRequest::new(model)
        .add_system_message("You are concise. Answer in one sentence.")
        .add_user_message(format!("What is {topic} in one sentence?"))
        .set_temperature(0.0)
        .set_max_tokens(512)
}

/// Drain a turn into its text plus per-kind block counts.
///
/// A reasoning model emits one `Thinking` block per token, so printing every
/// block buries the lines this demo is about. The counts stay in the output
/// because a hit replays the stored blocks and the two counts have to match.
async fn drain(mut stream: AgentResponseStream) -> (String, usize, usize) {
    let mut text = String::new();
    let (mut text_blocks, mut thinking_blocks) = (0usize, 0usize);
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
    (text.trim().to_string(), text_blocks, thinking_blocks)
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Base URL and model come from the environment so the example runs
    // against any OpenAI-compatible server; the defaults target a local
    // llama.cpp server (see the module docs for the override recipe).
    let base_url =
        std::env::var("CUCA_BASE_URL").unwrap_or_else(|_| "http://127.0.0.1:1234/v1".to_string());
    let model = std::env::var("CUCA_MODEL").unwrap_or_else(|_| "google/gemma-4-e4b".to_string());

    let counters = Arc::new(HookCounters::default());
    let client = CucaClient::builder()
        .with_provider(ProviderEndpoint::LlamaCpp)
        .with_base_url(base_url.clone())
        .with_prompt_cache_config(PromptCacheConfig::new(64, Duration::from_secs(300))?)
        .register_plugin(Arc::clone(&counters) as Arc<dyn CucaPlugin>)
        .build()?;
    let cache = client.prompt_cache().expect("configured on the builder");
    println!(
        "Cache: {}/{} entries, ttl 300 s",
        cache.len()?,
        cache.capacity()
    );

    // The key the client will compute for itself. `on_request` runs before the
    // lookup, so a hook that mutates the request changes this digest; the
    // counter plugin above mutates nothing.
    let first = question(&model, "a tokenizer");
    println!(
        "Key of the turn about to run: {:.16}...",
        digest_request(&first)?
    );

    println!("\nStage 1: the first turn (a miss)");
    let started = Instant::now();
    let stream = match client.generate_stream(first.clone()).await {
        Ok(stream) => stream,
        Err(error) => {
            println!("\nNo server answered at {base_url}: {error}");
            println!("Start llama-server there, or set CUCA_BASE_URL, then run this again.");
            return Ok(());
        }
    };
    let miss = drain(stream).await;
    println!("  reply: {}", miss.0);
    println!(
        "  {} ms, {} text blocks, {} thinking blocks",
        started.elapsed().as_millis(),
        miss.1,
        miss.2
    );
    println!("  hooks: {}", counters.reading());
    println!("  cache: {}/{} entries", cache.len()?, cache.capacity());

    // Stage 2: the write-back happened when the stage 1 stream reached its
    // end, so this lookup hits. A stream dropped before its last block writes
    // nothing.
    println!("\nStage 2: the identical turn (a hit)");
    let started = Instant::now();
    let hit = drain(client.generate_stream(first.clone()).await?).await;
    println!("  reply: {}", hit.0);
    println!(
        "  {} ms, {} text blocks, {} thinking blocks",
        started.elapsed().as_millis(),
        hit.1,
        hit.2
    );
    println!("  hooks: {}", counters.reading());
    println!("  identical to stage 1: {}", hit == miss);

    println!("\nStage 3: one word changed, so the digest changes");
    let second = question(&model, "an embedding");
    println!("  key: {:.16}...", digest_request(&second)?);
    let started = Instant::now();
    let (reply, _, _) = drain(client.generate_stream(second).await?).await;
    println!("  reply: {reply}");
    println!(
        "  {} ms, {}/{} entries",
        started.elapsed().as_millis(),
        cache.len()?,
        cache.capacity()
    );

    // Stage 4: the snapshot is the whole cache, so a client that cannot reach
    // any server still answers every turn the snapshot covers.
    println!("\nStage 4: the snapshot, imported into a client with no reachable server");
    let snapshot = client.prompt_cache_snapshot()?;
    let exported = snapshot.entries.len();
    let offline = CucaClient::builder()
        .with_provider(ProviderEndpoint::LlamaCpp)
        .with_base_url(CLOSED_PORT)
        .with_prompt_cache_config(PromptCacheConfig::new(64, Duration::from_secs(300))?)
        .build()?;
    let report = offline.replace_prompt_cache_snapshot(snapshot)?;
    println!(
        "  exported {exported} entries, imported {}, expired {}, evicted {}",
        report.imported_entries, report.expired_entries, report.capacity_evictions
    );
    println!("  base URL: {CLOSED_PORT}");
    let started = Instant::now();
    let (reply, _, _) = drain(offline.generate_stream(first).await?).await;
    println!(
        "  replayed stage 1 in {} ms: {reply}",
        started.elapsed().as_millis()
    );
    match offline.generate_stream(question(&model, "a logit")).await {
        Ok(_) => println!("  an uncached turn on that client: unexpectedly dispatched"),
        Err(error) => println!("  an uncached turn on that client: {error}"),
    }

    Ok(())
}
