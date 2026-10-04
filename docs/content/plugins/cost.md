+++
title = "Cost accounting"
description = "The token and currency ledger plugin: budget caps, per-model breakdown, and the tiktoken estimate it prices from."
template = "page.html"
weight = 14
+++

# Cost accounting

<dl class="page-facts">
<dt>In one line</dt>
<dd>Estimates prompt and completion tokens with tiktoken, prices them against a caller-supplied table, and refuses a turn that would cross a configured budget cap.</dd>
<dt>You need</dt>
<dd>The <code>plugin-cost</code> feature.</dd>
<dt>Read this if</dt>
<dd>You are registering <code>CostPlugin</code>, setting a budget cap, or reading its per-model spend.</dd>
</dl>

`CostPlugin` estimates prompt and completion tokens with tiktoken, prices them against a caller-supplied `PricingTable`, and refuses a turn in `on_request` once the projected spend would cross `max_total_tokens` or `max_total_micros`. Every reading is an estimate, made before the turn is dispatched so the cap can refuse it; provider-reported counts arrive only afterwards, in `UnifiedResponse::usage`, and only from adapters that receive them, so this plugin is the crate's only budget-enforcing ledger. Reach for it to enforce a budget cap or to read per-model spend without an upstream billing API.

```rust,name=Charge live turns then let the budget refuse one
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
```

```text,name=Expected output
Budget: 160 tokens, against a first prompt estimated at 18 tokens
Rates: 3000000 micros/Mtok in, 15000000 micros/Mtok out

Turn 1: "Name one bird. Reply with the name only."
  reply: Eagle
  blocks: 67 thinking
  ledger: 18 prompt + 71 completion = 89/160 tokens, 1119 micros, turns 1
  near cap: true

Turn 2: "Name one fish. Reply with the name only."
  reply: Salmon
  blocks: 46 thinking
  ledger: 52 prompt + 119 completion = 171/160 tokens, 1941 micros, turns 2
  near cap: true
  warning injected into this prompt: "CUCA cost warning: This client has used 77% of its budget cap; wrap up soon."

Turn 3: "Name one insect. Reply with the name only."
  refused before dispatch
  plugin "cost-accounting" at stage "request": token budget exceeded: this turn would reach 221 of 160 tokens
  ledger after the refusal: 171/160 tokens, turns 2

Per-model breakdown
  google/gemma-4-12b-qat  52 prompt + 119 completion, 1941 micros, turns 2

After reset(): 0 tokens, turns 0, cap still Some(160)
```

## Try it

`examples/cost.rs` is the program above. It prices the demo model, caps the ledger at 160 tokens, and dispatches a growing conversation until the cap refuses a turn: turn 2 crosses `warn_fraction` and carries the injected `CUCA cost warning:` system message, turn 3 is refused in `on_request` and charges nothing.

The committed total ends past its own cap, at `171/160`, because `on_request` gates on the projected prompt total and `on_response_complete` charges the completion afterwards. A cap bounds what is dispatched, not what a turn already in flight bills. Which turn crosses it depends on the model: every figure is a tiktoken estimate of what `google/gemma-4-12b-qat` actually emitted, most of it `Thinking` blocks.

```bash,name=Runs the same on all three platforms
cargo run --example cost --features "provider-llamacpp plugin-cost"
```

## Entry types

`CostPlugin`, `CostConfig`, `CostUsage`, `CostEntry`, `CostObserver`, `PricingTable`, `PricingResolver`, `ModelRates`, `UnpricedModelPolicy`.

## `CucaPlugin`

`CostPlugin` implements `CucaPlugin` with the plugin name `"cost-accounting"` and attaches via `register_plugin`, like any other hook plugin. It overrides `on_request` and `on_response_complete`; `execute_local_tool` and `on_stream_chunk` use the trait defaults, because the plugin owns no tool and a budget cannot abort a turn mid-stream. `CostPlugin::new(config)` validates `CostConfig` and loads the tiktoken encoder named by `encoder_name`, returning `PluginError::Internal` for either failure.

Every reading is an estimate. `CostPlugin` charges its own tiktoken counts and never reads `UnifiedResponse::prompt_tokens`, `completion_tokens` or `usage`, which carry provider-reported counts only when the adapter received them (the OpenAI-compatible adapters do; elsewhere `prompt_tokens` is `0` and `completion_tokens` counts blocks). `prompt_cache_usage`, populated by the Anthropic adapter only, is the only correction `CostPlugin` applies against its own tiktoken count.

## Config

`CostConfig` fields and their defaults:

| Field | Default |
|---|---|
| `encoder_name` | `"cl100k_base"` |
| `pricing` | empty `PricingTable` |
| `pricing_resolver` | `None` |
| `max_total_tokens` | `None` |
| `max_total_micros` | `None` |
| `warn_fraction` | `None` |
| `max_tracked_models` | `64` |
| `on_unpriced_model` | `UnpricedModelPolicy::CountTokensOnly` |
| `observers` | empty |

`max_total_tokens` and `max_total_micros` each disable enforcement on that axis when `None`; `Some(0)` is rejected. `warn_fraction` must fall in `(0.0, 1.0]` and requires at least one cap set. `UnpricedModelPolicy::Reject` with an empty `pricing` table and no `pricing_resolver` is rejected, since it would refuse every turn. Rates in `ModelRates` and spend in `CostUsage` are micro-units of a caller-defined currency per million tokens; the crate never names a currency and never converts.

## Hooks

`on_request` estimates the turn's prompt tokens, including every `req.tools` schema, prices them through `rates_for` (the resolver first, then `pricing`), and projects the post-charge totals against `max_total_tokens` and `max_total_micros`. Either projected total crossing its cap returns `PluginError::HookFailure` and charges nothing. A `warn_fraction` crossing injects a one-shot system message starting `CUCA cost warning:`, the same marker scheme `plugin-memory` uses for its own warning. The charge then commits and every `CostObserver` receives the fresh `CostUsage`; an observer `Err` aborts the turn.

`on_response_complete` estimates completion tokens from `res.content`, re-prices the prompt portion at the cache read and write rates when `res.prompt_cache_usage` is present, commits the model's bucket, and observes again. An observer `Err` here is logged and never surfaces; a poisoned ledger lock instead surfaces on the plugin's next `on_request`.

A client-level cache hit (`service-prompt-cache`) still runs `on_response_complete` against the replayed response, so a cached turn is still charged: the ledger reads as gross, pre-cache spend.

## Caps

| | |
|---|---|
| Bound | `CostConfig::max_tracked_models` per-model entries, caller-set, default `64` |
| At-cap policy | A turn for an untracked model folds into one reserved overflow bucket and increments `CostUsage::overflow_turns`; no eviction, and `usage()` totals stay exact, only per-model attribution degrades |
| Usage gauge | `CostPlugin::usage()` for totals, `CostPlugin::breakdown()` (bounded by `max_tracked_models + 1`) for the per-model slice |

`CostConfig::pricing` and `CostConfig::observers` are caller-owned and fixed at construction; neither grows from traffic.

## Accessors

`CostPlugin::usage()` returns a `Copy` `CostUsage` reading under one lock. `CostPlugin::breakdown()` returns the per-model entries as a `Vec<(String, CostEntry)>`, sorted by model id. `CostPlugin::reset()` zeroes the ledger for a billing-period rollover, leaving configuration untouched. `CostPlugin::estimate_request_tokens(req)` runs the same estimator the hooks use, callable before a client exists.

## OpenTelemetry bridge

`OtelCostObserver` is a `CostObserver` the crate ships, compiled only when `plugin-cost` and `plugin-telemetry` are both enabled. Put one in `CostConfig::observers` and every reading reaches a caller-supplied meter provider; its `observe` is infallible, so it never aborts a turn. It lives in core, at `cuca::cost_otel`, because neither plugin may name the other. Its instruments are listed under [Telemetry](@/plugins/telemetry.md).

`examples/cost_otel.rs` runs that bridge: one priced live turn, the ledger read three ways, and the gauges read back through an in-memory exporter.

```bash,name=Runs the same on all three platforms
cargo run --example cost_otel --features "provider-llamacpp plugin-cost plugin-telemetry"
```
