//! Watch a live conversation cross a compaction trigger inside `on_request`.
//!
//! A `MemoryPlugin` is registered on the client with a six-message trigger, a
//! deliberately tiny context window, and a `warn_fraction` of `0.2`. Four real
//! turns then stream through it. A `ContextUsageObserver` prints the gauge the
//! hook hands it on every request, a second plugin registered after memory
//! prints the prompt memory left behind, and the last turn shows both edits the
//! hook makes: the one-shot near-limit warning, and the compaction that trims
//! the message list back under the trigger.
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
//! cargo run --example memory --features provider-llamacpp,plugin-memory
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
//! Example: `CUCA_BASE_URL=http://127.0.0.1:8000/v1 CUCA_MODEL=<server-model-id> cargo run --example memory --features provider-llamacpp,plugin-memory`
//!
//! # Output
//!
//! From one run against `google/gemma-4-12b-qat` on llama.cpp:
//!
//! ```text
//! Trigger: max_messages 6, window 256 tokens, warn at 20%
//! Strategy: SlidingWindow { keep_messages: 4 }
//! Prompt shape letters: S system, U user, A assistant
//!
//! Turn 1: "The vault is in Lisbon. Reply with: noted"
//!   gauge: 16/256 tokens (6%), window resolved false
//!   prompt: 2 messages in, 2 out (SU), warning false
//!   reply: noted
//!
//! Turn 2: "The key rotates on friday. Reply with: noted"
//!   gauge: 29/256 tokens (11%), window resolved false
//!   prompt: 4 messages in, 4 out (SUAU), warning false
//!   reply: noted
//!
//! Turn 3: "The owner is ops@example.com. Reply with: noted"
//!   gauge: 43/256 tokens (17%), window resolved false
//!   prompt: 6 messages in, 6 out (SUAUAU), warning false
//!   reply: noted
//!
//! Turn 4: "Where is the vault? Answer in one word."
//!   gauge: 56/256 tokens (22%), window resolved false
//!   prompt: 8 messages in, 4 out (SAUS), warning true, compacted
//!   reply: Unknown.
//! ```
//!
//! Turn 4 is both edits at once. The gauge crosses `0.2`, so the hook appends
//! one warning system message, taking the prompt to nine messages; the
//! nine-message prompt is over the six-message trigger, so `SlidingWindow`
//! trims it to four. The two messages the plugin never removes are still
//! there, first system and most recent user, which is why the shape is `SAUS`.
//!
//! `Unknown.` is the honest consequence: the message that said Lisbon is one of
//! the four the window dropped, so the model cannot read it any more. Keeping
//! dropped turns reachable is the vector store's offload seam, not compaction's
//! job.
//!
//! The replies and the exact token counts depend on the model. The message
//! counts do not: the trigger, the strategy, and the never-remove invariant are
//! all deterministic.
//!
//! With no server on the base URL, the program prints one line naming the
//! address and exits successfully.
//!
//! # Why the counts are read from a second plugin
//!
//! `on_request` compacts the request in place and returns nothing: the
//! `CompressionReport` it builds internally is only returned by the out of band
//! `MemoryPlugin::compress`. Hooks run in registration order over one shared
//! request, so the honest way to see what the hook did to a dispatched turn is
//! a plugin registered after it. The gauge is the other half: the plugin hands
//! every `ContextUsageObserver` a `ContextUsage` reading on every request,
//! before any compaction, which is the reading a dashboard would show.

use std::sync::{Arc, Mutex};

use cuca::plugin::CucaPlugin;
use cuca::types::{MessageContentBlock, MessageRole, ProviderEndpoint, UnifiedMessage};
use cuca::{
    AgentResponseStream, CompactionStrategy, ContextUsage, ContextUsageObserver, CucaClient,
    MemoryConfig, MemoryPlugin, PluginError, UnifiedRequest,
};
use tokio_stream::StreamExt;

/// Per-turn completion cap. A reasoning model spends most of it on `Thinking`
/// blocks, so a smaller budget returns an empty reply.
const MAX_TOKENS: u32 = 512;

/// Compaction trigger: compress once the prompt is longer than this.
const MAX_MESSAGES: usize = 6;

/// Context window the fractions are measured against. The real window of the
/// demo model is far larger; 256 tokens is what makes a four-turn conversation
/// cross `warn_fraction` instead of a four-hundred-turn one. A deployment sets
/// the model's own window here, or supplies a `ContextWindowResolver`.
const WINDOW_TOKENS: u32 = 256;

/// Prompt for each turn. The first three seed facts, the fourth asks about the
/// oldest of them, after compaction has run.
const PROMPTS: [&str; 4] = [
    "The vault is in Lisbon. Reply with: noted",
    "The key rotates on friday. Reply with: noted",
    "The owner is ops@example.com. Reply with: noted",
    "Where is the vault? Answer in one word.",
];

/// The marker the near-limit warning starts with.
const WARNING_MARKER: &str = "CUCA context warning:";

/// The reporting gauge seam: handed one reading per request, before compaction.
#[derive(Default)]
struct UsageGauge {
    latest: Mutex<Option<ContextUsage>>,
}

impl UsageGauge {
    fn latest(&self) -> Option<ContextUsage> {
        self.latest.lock().ok().and_then(|latest| *latest)
    }
}

impl ContextUsageObserver for UsageGauge {
    fn observe(&self, usage: &ContextUsage) -> Result<(), PluginError> {
        *self
            .latest
            .lock()
            .map_err(|_| PluginError::Internal("gauge lock poisoned".into()))? = Some(*usage);
        Ok(())
    }
}

/// The prompt as the memory hook left it: message count, role shape, and
/// whether the near-limit warning is in there.
#[derive(Default)]
struct PromptRecorder {
    shape: Mutex<Option<(usize, String, bool)>>,
}

impl PromptRecorder {
    fn shape(&self) -> Option<(usize, String, bool)> {
        self.shape.lock().ok().and_then(|shape| shape.clone())
    }
}

impl CucaPlugin for PromptRecorder {
    fn name(&self) -> &'static str {
        "prompt-recorder"
    }

    fn on_request(&self, req: &mut UnifiedRequest) -> Result<(), PluginError> {
        let letters: String = req
            .messages
            .iter()
            .map(|message| match message.role {
                MessageRole::System => 'S',
                MessageRole::User => 'U',
                MessageRole::Assistant => 'A',
                MessageRole::Tool => 'T',
            })
            .collect();
        let warned = req
            .messages
            .iter()
            .flat_map(|message| message.content.iter())
            .any(|block| {
                matches!(block, MessageContentBlock::Text(text) if text.starts_with(WARNING_MARKER))
            });
        *self
            .shape
            .lock()
            .map_err(|_| PluginError::Internal("recorder lock poisoned".into()))? =
            Some((req.messages.len(), letters, warned));
        Ok(())
    }
}

/// Drain a turn into its text, dropping the `Thinking` blocks a reasoning model
/// emits one per token: printing them would bury the four lines per turn this
/// demo is about.
async fn drain(mut stream: AgentResponseStream) -> String {
    let mut text = String::new();
    while let Some(chunk) = stream.next().await {
        match chunk {
            Ok(MessageContentBlock::Text(chunk_text)) => text.push_str(&chunk_text),
            Ok(_) => {}
            Err(error) => {
                println!("  the stream ended early: {error}");
                break;
            }
        }
    }
    text
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Base URL and model come from the environment so the example runs
    // against any OpenAI-compatible server; the defaults target a local
    // llama.cpp server (see the module docs for the override recipe).
    let base_url =
        std::env::var("CUCA_BASE_URL").unwrap_or_else(|_| "http://127.0.0.1:1234/v1".to_string());
    let model = std::env::var("CUCA_MODEL").unwrap_or_else(|_| "google/gemma-4-e4b".to_string());

    let gauge = Arc::new(UsageGauge::default());
    let strategy = CompactionStrategy::SlidingWindow { keep_messages: 4 };
    // One strategy, not the default eight-step pipeline: every other step needs
    // an extension seam or tool traffic this demo has none of, and would no-op.
    let memory = Arc::new(MemoryPlugin::new(MemoryConfig {
        context_window_tokens: WINDOW_TOKENS,
        max_messages: Some(MAX_MESSAGES),
        warn_fraction: Some(0.2),
        observers: vec![Arc::clone(&gauge) as Arc<dyn ContextUsageObserver>],
        strategies: vec![strategy.clone()],
        ..Default::default()
    })?);
    let recorder = Arc::new(PromptRecorder::default());
    let client = CucaClient::builder()
        .with_provider(ProviderEndpoint::LlamaCpp)
        .with_base_url(base_url.clone())
        .register_plugin(Arc::clone(&memory) as Arc<dyn CucaPlugin>)
        .register_plugin(Arc::clone(&recorder) as Arc<dyn CucaPlugin>)
        .build()?;

    println!("Trigger: max_messages {MAX_MESSAGES}, window {WINDOW_TOKENS} tokens, warn at 20%");
    println!("Strategy: {strategy:?}");
    println!("Prompt shape letters: S system, U user, A assistant");

    let mut messages = vec![UnifiedMessage::system("You are concise.")];
    for (index, prompt) in PROMPTS.iter().enumerate() {
        messages.push(UnifiedMessage::user(*prompt));
        println!("\nTurn {}: {prompt:?}", index + 1);
        let sent = messages.len();
        let mut request = UnifiedRequest::new(&model).set_max_tokens(MAX_TOKENS);
        request.messages = messages.clone();
        let stream = match client.generate_stream(request).await {
            Ok(stream) => stream,
            Err(error) => {
                println!("\nNo server answered at {base_url}: {error}");
                println!("Start llama-server there, or set CUCA_BASE_URL, then run this again.");
                return Ok(());
            }
        };
        if let Some(usage) = gauge.latest() {
            println!(
                "  gauge: {}/{} tokens ({:.0}%), window resolved {}",
                usage.used_tokens,
                usage.window_tokens,
                f64::from(usage.used_tokens) / f64::from(usage.window_tokens) * 100.0,
                usage.resolved
            );
        }
        if let Some((count, letters, warned)) = recorder.shape() {
            let note = if count < sent { ", compacted" } else { "" };
            println!(
                "  prompt: {sent} messages in, {count} out ({letters}), warning {warned}{note}"
            );
        }
        let reply = drain(stream).await;
        println!("  reply: {}", reply.trim());
        // The conversation the caller owns keeps growing: compaction rewrites
        // the request the provider sees, never the caller's own history.
        messages.push(UnifiedMessage::assistant(reply.trim()));
    }

    Ok(())
}
