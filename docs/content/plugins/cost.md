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

`CostPlugin` estimates prompt and completion tokens with tiktoken, prices them against a caller-supplied `PricingTable`, and refuses a turn in `on_request` once the projected spend would cross `max_total_tokens` or `max_total_micros`. Every reading is an estimate: no provider adapter parses an upstream `usage` object into `UnifiedResponse::prompt_tokens`, so this plugin is the crate's only token and spend ledger. Reach for it to enforce a budget cap or to read per-model spend without an upstream billing API.

```rust,name=Charge a turn against a priced model
use std::sync::Arc;

use cuca::plugin::CucaPlugin;
use cuca::types::ProviderEndpoint;
use cuca::{CostConfig, CostPlugin, CucaClient, ModelRates, PricingTable, UnifiedRequest};

let cost = Arc::new(CostPlugin::new(CostConfig {
    pricing: PricingTable::new().with_model(
        "google/gemma-4-e4b",
        ModelRates {
            input_micros_per_mtok: 3_000_000,
            output_micros_per_mtok: 15_000_000,
            ..Default::default()
        },
    ),
    max_total_tokens: Some(50_000),
    ..Default::default()
})?);

let client = CucaClient::builder()
    .with_provider(ProviderEndpoint::LlamaCpp)
    .with_base_url("http://127.0.0.1:1234/v1")
    .register_plugin(Arc::clone(&cost) as Arc<dyn CucaPlugin>)
    .build()?;

let mut req = UnifiedRequest::new("google/gemma-4-e4b")
    .add_system_message("You are concise.")
    .add_user_message("Explain CUCA in one sentence.");
cost.on_request(&mut req)?;
let usage = cost.usage()?;
```

```text,name=One estimated turn against the rate table above
usage.prompt_tokens   16
usage.spent_micros    48
```

## Entry types

`CostPlugin`, `CostConfig`, `CostUsage`, `CostEntry`, `CostObserver`, `PricingTable`, `PricingResolver`, `ModelRates`, `UnpricedModelPolicy`.

## `CucaPlugin`

`CostPlugin` implements `CucaPlugin` with the plugin name `"cost-accounting"` and attaches via `register_plugin`, like any other hook plugin. It overrides `on_request` and `on_response_complete`; `execute_local_tool` and `on_stream_chunk` use the trait defaults, because the plugin owns no tool and a budget cannot abort a turn mid-stream. `CostPlugin::new(config)` validates `CostConfig` and loads the tiktoken encoder named by `encoder_name`, returning `PluginError::Internal` for either failure.

Every reading is an estimate. `UnifiedResponse::prompt_tokens` is always `0` and `completion_tokens` counts blocks, not tokens; no provider adapter parses an upstream `usage` object into either field. `prompt_cache_usage`, populated by the Anthropic adapter only, is the sole provider-reported token data in the crate, and the only correction `CostPlugin` applies against its own tiktoken count.

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
