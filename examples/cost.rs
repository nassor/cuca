//! Charge live turns against a price table, then let the budget cap refuse one.
//!
//! A `CostPlugin` is registered on the client with rates for the demo model, a
//! 160-token cumulative cap, and a `warn_fraction` of `0.3`.
//! Real turns then stream through it: `on_request` charges the prompt estimate
//! and enforces the cap before dispatch, `on_response_complete` charges the
//! completion estimate, and the ledger is printed after every turn. The cap is
//! crossed while the conversation is still running, so the last turn is refused
//! before it reaches the provider and the ledger keeps nothing for it.
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
//! cargo run --example cost --features provider-llamacpp,plugin-cost
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
//! Example: `CUCA_BASE_URL=http://127.0.0.1:8000/v1 CUCA_MODEL=<server-model-id> cargo run --example cost --features provider-llamacpp,plugin-cost`
//!
//! # Output
//!
//! From one run against `google/gemma-4-12b-qat` on llama.cpp:
//!
//! ```text
//! Budget: 160 tokens, against a first prompt estimated at 18 tokens
//! Rates: 3000000 micros/Mtok in, 15000000 micros/Mtok out
//!
//! Turn 1: "Name one bird. Reply with the name only."
//!   reply: Eagle
//!   blocks: 67 thinking
//!   ledger: 18 prompt + 71 completion = 89/160 tokens, 1119 micros, turns 1
//!   near cap: true
//!
//! Turn 2: "Name one fish. Reply with the name only."
//!   reply: Salmon
//!   blocks: 46 thinking
//!   ledger: 52 prompt + 119 completion = 171/160 tokens, 1941 micros, turns 2
//!   near cap: true
//!   warning injected into this prompt: "CUCA cost warning: This client has used 77% of its budget cap; wrap up soon."
//!
//! Turn 3: "Name one insect. Reply with the name only."
//!   refused before dispatch
//!   plugin "cost-accounting" at stage "request": token budget exceeded: this turn would reach 221 of 160 tokens
//!   ledger after the refusal: 171/160 tokens, turns 2
//!
//! Per-model breakdown
//!   google/gemma-4-12b-qat  52 prompt + 119 completion, 1941 micros, turns 2
//!
//! After reset(): 0 tokens, turns 0, cap still Some(160)
//! ```
//!
//! The committed total ends at `171/160`, past its own cap, and that is the
//! design: `on_request` gates the turn on the projected *prompt* total, then
//! `on_response_complete` charges the completion once the stream is over. A cap
//! therefore bounds what is dispatched, never what is billed for a turn already
//! in flight. The refusal itself charges nothing, which is why the ledger reads
//! the same before and after it.
//!
//! Which turn the cap refuses depends on the model: every charged token is a
//! tiktoken estimate of what the model actually emitted, and a reasoning model
//! spends most of its completion budget on `Thinking` blocks. The budget, the
//! rates, and the refusal being charge-free do not depend on the model.
//!
//! With no server on the base URL, the program prints one line naming the
//! address and exits successfully.
//!
//! # Why the cap is projected, not detected
//!
//! `on_request` charges the prompt estimate and compares the *projected* total
//! against the cap, so a budget is never crossed rather than merely reported
//! afterwards. That is also why the refused turn commits nothing: the check
//! runs before the charge, and no terminal hook fires on a turn that never
//! dispatched.

use std::sync::{Arc, Mutex};

use cuca::plugin::CucaPlugin;
use cuca::types::{MessageContentBlock, ProviderEndpoint, UnifiedMessage};
use cuca::{
    AgentResponseStream, CostConfig, CostPlugin, CucaClient, CucaError, ModelRates, PluginError,
    PricingTable, UnifiedRequest,
};
use tokio_stream::StreamExt;

/// Per-turn completion cap. A reasoning model spends most of it on `Thinking`
/// blocks, so a smaller budget returns an empty reply.
const MAX_TOKENS: u32 = 512;

/// Cumulative token cap, deliberately small: a four-turn conversation crosses
/// it, which is the refusal this demo exists to show. `on_request` enforces it
/// against the projected total, so the crossing turn never dispatches.
const BUDGET: u64 = 160;

/// One prompt per turn, over a growing conversation, so each turn's prompt
/// estimate is larger than the last.
const PROMPTS: [&str; 4] = [
    "Name one bird. Reply with the name only.",
    "Name one fish. Reply with the name only.",
    "Name one insect. Reply with the name only.",
    "Name one tree. Reply with the name only.",
];

/// The marker every near-cap warning starts with.
const WARNING_MARKER: &str = "CUCA cost warning:";

/// Rates in micro-units of the caller's currency per million tokens: US$3.00 in
/// and US$15.00 out, if that currency is USD. The crate never names one.
fn rates() -> ModelRates {
    ModelRates {
        input_micros_per_mtok: 3_000_000,
        output_micros_per_mtok: 15_000_000,
        ..Default::default()
    }
}

/// Reports the near-cap warning the cost plugin injects.
///
/// `on_request` hooks run in registration order over one shared request, so a
/// plugin registered after the cost plugin sees its injection. Nothing outside
/// the pipeline can: the request is moved into the provider adapter.
#[derive(Default)]
struct WarningWatcher {
    seen: Mutex<Option<String>>,
}

impl WarningWatcher {
    /// The warning present on the most recent request, if the plugin injected
    /// one.
    fn seen(&self) -> Option<String> {
        self.seen.lock().ok().and_then(|seen| seen.clone())
    }
}

impl CucaPlugin for WarningWatcher {
    fn name(&self) -> &'static str {
        "cost-warning-watcher"
    }

    fn on_request(&self, req: &mut UnifiedRequest) -> Result<(), PluginError> {
        let warning = req
            .messages
            .iter()
            .flat_map(|message| message.content.iter())
            .find_map(|block| match block {
                MessageContentBlock::Text(text) if text.starts_with(WARNING_MARKER) => {
                    Some(text.clone())
                }
                _ => None,
            });
        *self
            .seen
            .lock()
            .map_err(|_| PluginError::Internal("watcher lock poisoned".into()))? = warning;
        Ok(())
    }
}

/// Drain a turn into its text plus the thinking-block count.
///
/// A reasoning model emits one `Thinking` block per token, so printing every
/// block would bury the ledger lines this demo is about.
async fn drain(mut stream: AgentResponseStream) -> (String, usize) {
    let mut text = String::new();
    let mut thinking_blocks = 0usize;
    while let Some(chunk) = stream.next().await {
        match chunk {
            Ok(MessageContentBlock::Text(chunk_text)) => text.push_str(&chunk_text),
            Ok(MessageContentBlock::Thinking { .. }) => thinking_blocks += 1,
            Ok(_) => {}
            Err(error) => {
                println!("  the stream ended early: {error}");
                break;
            }
        }
    }
    (text, thinking_blocks)
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Base URL and model come from the environment so the example runs
    // against any OpenAI-compatible server; the defaults target a local
    // llama.cpp server (see the module docs for the override recipe).
    let base_url =
        std::env::var("CUCA_BASE_URL").unwrap_or_else(|_| "http://127.0.0.1:1234/v1".to_string());
    let model = std::env::var("CUCA_MODEL").unwrap_or_else(|_| "google/gemma-4-e4b".to_string());

    let mut messages = vec![
        UnifiedMessage::system("You are concise."),
        UnifiedMessage::user(PROMPTS[0]),
    ];

    // `estimate_request_tokens` runs the hooks' own estimator with no client
    // in play, so the cap above can be chosen against a measured turn instead
    // of a guess.
    let probe = CostPlugin::new(CostConfig::default())?;
    let mut first = UnifiedRequest::new(&model).set_max_tokens(MAX_TOKENS);
    first.messages = messages.clone();
    println!(
        "Budget: {BUDGET} tokens, against a first prompt estimated at {} tokens",
        probe.estimate_request_tokens(&first)?
    );
    let rates = rates();
    println!(
        "Rates: {} micros/Mtok in, {} micros/Mtok out",
        rates.input_micros_per_mtok, rates.output_micros_per_mtok
    );

    let cost = Arc::new(CostPlugin::new(CostConfig {
        pricing: PricingTable::new().with_model(model.clone(), rates),
        max_total_tokens: Some(BUDGET),
        warn_fraction: Some(0.3),
        ..Default::default()
    })?);
    let watcher = Arc::new(WarningWatcher::default());
    let client = CucaClient::builder()
        .with_provider(ProviderEndpoint::LlamaCpp)
        .with_base_url(base_url.clone())
        .register_plugin(Arc::clone(&cost) as Arc<dyn CucaPlugin>)
        .register_plugin(Arc::clone(&watcher) as Arc<dyn CucaPlugin>)
        .build()?;

    for (index, prompt) in PROMPTS.iter().enumerate() {
        if index > 0 {
            messages.push(UnifiedMessage::user(*prompt));
        }
        println!("\nTurn {}: {prompt:?}", index + 1);
        let mut request = UnifiedRequest::new(&model).set_max_tokens(MAX_TOKENS);
        request.messages = messages.clone();
        let stream = match client.generate_stream(request).await {
            Ok(stream) => stream,
            Err(CucaError::Plugin(PluginError::HookFailure {
                plugin,
                stage,
                message,
            })) => {
                println!("  refused before dispatch");
                println!("  plugin {plugin:?} at stage {stage:?}: {message}");
                let usage = cost.usage()?;
                println!(
                    "  ledger after the refusal: {}/{BUDGET} tokens, turns {}",
                    usage.total_tokens(),
                    usage.turns
                );
                break;
            }
            Err(error) => {
                println!("\nNo server answered at {base_url}: {error}");
                println!("Start llama-server there, or set CUCA_BASE_URL, then run this again.");
                return Ok(());
            }
        };
        let (reply, thinking_blocks) = drain(stream).await;
        println!("  reply: {}", reply.trim());
        println!("  blocks: {thinking_blocks} thinking");
        messages.push(UnifiedMessage::assistant(reply.trim()));

        let usage = cost.usage()?;
        println!(
            "  ledger: {} prompt + {} completion = {}/{BUDGET} tokens, {} micros, turns {}",
            usage.prompt_tokens,
            usage.completion_tokens,
            usage.total_tokens(),
            usage.spent_micros,
            usage.turns
        );
        println!("  near cap: {}", usage.near_cap);
        if let Some(warning) = watcher.seen() {
            println!("  warning injected into this prompt: {warning:?}");
        }
    }

    println!("\nPer-model breakdown");
    for (charged_model, entry) in cost.breakdown()? {
        println!(
            "  {charged_model}  {} prompt + {} completion, {} micros, turns {}",
            entry.prompt_tokens, entry.completion_tokens, entry.spent_micros, entry.turns
        );
    }

    // A billing-period rollover zeroes the ledger and leaves the caps in place.
    cost.reset()?;
    let usage = cost.usage()?;
    println!(
        "\nAfter reset(): {} tokens, turns {}, cap still {:?}",
        usage.total_tokens(),
        usage.turns,
        usage.max_total_tokens
    );
    Ok(())
}
