+++
title = "Cost accounting"
description = "Estimate tokens and spend for each turn, and refuse turns that would pass a budget you set."
template = "page.html"
weight = 14
+++

# Cost accounting

<dl class="page-facts">
<dt>In one line</dt>
<dd><code>CostPlugin</code> estimates the tokens of each turn, prices them with rates you supply, and refuses a turn that would pass a token or money budget.</dd>
<dt>You need</dt>
<dd>The <code>plugin-cost</code> feature. Without prices, it counts tokens only.</dd>
<dt>Read this if</dt>
<dd>You want a hard spending limit on a client, or a running total of tokens and cost per model.</dd>
</dl>

A token is a small piece of text that a model reads or writes. Providers bill by tokens. The plugin counts tokens itself with tiktoken (a tokenizer library), so every figure is an estimate and may differ a little from the provider's bill.

## Register it

```rust,name=Cap the ledger and price one model
let cost = Arc::new(CostPlugin::new(CostConfig {
    pricing: PricingTable::new().with_model(
        model.clone(),
        ModelRates {
            input_micros_per_mtok: 3_000_000,
            output_micros_per_mtok: 15_000_000,
            ..Default::default()
        },
    ),
    max_total_tokens: Some(160),
    warn_fraction: Some(0.3),
    ..Default::default()
})?);
let client = CucaClient::builder()
    .with_provider(ProviderEndpoint::LlamaCpp)
    .with_base_url(base_url)
    .register_plugin(Arc::clone(&cost) as Arc<dyn CucaPlugin>)
    .build()?;
```

Expected effect: the ledger adds up each turn. A turn whose prompt would take the total past 160 tokens is refused before it is sent, and costs nothing. Read the totals with `cost.usage()` and `cost.breakdown()`.

## How a turn is charged

<div class="dgm">
<div class="dgm-scroll">
<svg viewBox="0 0 660 176" role="img" aria-labelledby="cost-title cost-desc">
  <title id="cost-title">The prompt is checked against the caps before sending, and the completion is charged after.</title>
  <desc id="cost-desc">In on_request the plugin estimates the prompt tokens and projects the totals. If a cap would be passed, the turn is refused and nothing is charged. Otherwise the prompt is charged and the request is sent. In on_response_complete the completion tokens are estimated and charged, so the total can end past a cap.</desc>
  <defs>
    <marker id="cost-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M0 0.6 L7 4 L0 7.4 z" fill="var(--muted-foreground)"/>
    </marker>
  </defs>
  <text class="t-title" x="6" y="20">CostPlugin across one turn</text>
  <rect class="blk blk-ctl" x="5" y="40" width="140" height="50" rx="6"/>
  <text class="t-lbl t-mid t-ctl" x="75" y="62">on_request</text>
  <text class="t-sm t-mid" x="75" y="78">estimate the prompt</text>
  <rect class="blk blk-ctl" x="185" y="40" width="130" height="50" rx="6"/>
  <text class="t-lbl t-mid t-ctl" x="250" y="62">over a cap?</text>
  <text class="t-sm t-mid" x="250" y="78">tokens or money</text>
  <rect class="blk blk-ctl" x="355" y="40" width="120" height="50" rx="6"/>
  <text class="t-lbl t-mid t-ctl" x="415" y="62">charge prompt</text>
  <text class="t-sm t-mid" x="415" y="78">then send</text>
  <rect class="blk blk-ctl" x="515" y="40" width="140" height="50" rx="6"/>
  <text class="t-sm t-mid t-ctl" x="585" y="62">on_response_complete</text>
  <text class="t-sm t-mid" x="585" y="78">charge completion</text>
  <rect class="blk blk-data" x="185" y="122" width="290" height="40" rx="6"/>
  <text class="t-sm t-mid" x="330" y="146">yes: refuse the turn, charge nothing</text>
  <path class="arw arw-ctl" d="M146 65 H183" marker-end="url(#cost-arrow)"/>
  <path class="arw arw-ctl" d="M316 65 H353" marker-end="url(#cost-arrow)"/>
  <text class="t-sm t-mid" x="335" y="58">no</text>
  <path class="arw arw-ctl" d="M476 65 H513" marker-end="url(#cost-arrow)"/>
  <path class="arw arw-data" d="M250 92 V120" marker-end="url(#cost-arrow)"/>
</svg>
</div>
<p class="dgm-cap"><b>A cap stops what is sent, not what a turn already in flight bills.</b> The completion is charged after the reply, so the committed total can end past a cap. The next turn is then refused.</p>
</div>

## Run the example

`examples/cost.rs` prices the demo model, caps the ledger at 160 tokens, and sends a growing conversation until the cap refuses a turn. It needs a `llama-server` on port 1234. Put `--base-url` and `--model` after `--` to use another OpenAI-compatible server.

```bash,name=Runs the same on all three platforms
cargo run --example cost --features "provider-llamacpp plugin-cost"
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

Turn 2 crosses `warn_fraction` and carries the warning. Turn 3 is refused and charges nothing. The total ends at `171/160` because the completion is charged after the check. Which turn crosses the cap depends on the model.

## Feature and entry type

| Item | Value |
|---|---|
| Feature flag | `plugin-cost` |
| Entry type | `CostPlugin`, built with `CostPlugin::new(CostConfig)` |
| Related types | `CostConfig`, `CostUsage`, `CostEntry`, `CostObserver`, `PricingTable`, `PricingResolver`, `ModelRates`, `UnpricedModelPolicy` |
| Plugin name | `"cost-accounting"` |
| Attachment | `register_plugin` |
| Needs | A `PricingTable` or `PricingResolver` to price spend. Without one, turns are counted in tokens only. |
| Telemetry bridge | `OtelCostObserver`, available when `plugin-telemetry` is also on (see [Telemetry](@/plugins/telemetry.md)) |

## Hooks

`CostPlugin` acts on `on_request` and `on_response_complete`. It does not act on `on_stream_chunk`, because a budget cannot stop a turn that is already streaming.

- `on_request` estimates the prompt tokens, prices them at the input rate, and projects the new totals against `max_total_tokens` and `max_total_micros`. A projected total over either cap refuses the turn and charges nothing.
- After the check, `on_request` records the charge. If `near_cap` is true and no warning is present yet, it adds a system message starting `CUCA cost warning:`. Then every `CostObserver` gets the new `CostUsage`.
- `on_response_complete` estimates the completion tokens from the response content, prices them at the output rate, records them under the model, and calls the observers again.
- If the response has `prompt_cache_usage`, cached read tokens are refunded at the input rate and charged at the cache read rate. Cache write tokens are charged at the cache write rate.

### What is counted

Every figure is a tiktoken estimate. The plugin does not read the token counts a provider reports, because the prompt charge is made before the provider has counted anything, and one estimator keeps providers consistent. The one provider figure it uses is `prompt_cache_usage`, which only the Anthropic adapter fills.

- The prompt estimate counts each message's role label, every `Text` block, every `Thinking.reasoning`, every `ToolResult.output`, every `ToolCall` name with its JSON arguments, and every tool definition (name, description, input schema).
- It skips `ImageBase64` blocks, per-message overhead tokens, and settings like `max_tokens`. Each skipped image adds 1 to `CostUsage::untokenized_image_blocks`.
- The completion estimate counts `Text`, `Thinking.reasoning`, and `ToolCall` name plus arguments. `ToolResult` and `ImageBase64` blocks add nothing.

### Edge cases

- A turn that fails after `on_request` keeps its prompt charge, because no later hook runs. A budget errs on the safe side.
- With a client-level cache hit (`service-prompt-cache`), `on_response_complete` still runs on the replayed response, so a cached turn is charged. The ledger shows spend before caching.
- The near-cap warning changes the request, so it changes the key a response cache computes afterward.

### Reading the ledger

- `CostPlugin::usage()` returns a `Copy` `CostUsage`.
- `CostPlugin::breakdown()` returns `Vec<(String, CostEntry)>`, sorted by model id.
- `CostPlugin::reset()` zeroes the ledger, for example at the start of a billing period. The configuration stays.
- `CostPlugin::estimate_request_tokens(req)` runs the same estimator, before a client exists.
- `CostPlugin::rates_for(model)` returns the resolver's rates first, then the `pricing` table's.

## Configuration

`CostConfig::default()` carries the defaults below. `CostPlugin::new` checks the config and loads the encoder.

| Key | Type | Default |
|---|---|---|
| `encoder_name` | `String` | `"cl100k_base"` |
| `pricing` | `PricingTable` | empty |
| `pricing_resolver` | `Option<Arc<dyn PricingResolver>>` | `None` |
| `max_total_tokens` | `Option<u64>` | `None` |
| `max_total_micros` | `Option<u64>` | `None` |
| `warn_fraction` | `Option<f32>` | `None` |
| `max_tracked_models` | `usize` | `64` |
| `on_unpriced_model` | `UnpricedModelPolicy` | `CountTokensOnly` |
| `observers` | `Vec<Arc<dyn CostObserver>>` | empty |

- `encoder_name` is a tiktoken-rs base encoder (`r50k_base`, `p50k_base`, `p50k_edit`, `cl100k_base`, `o200k_base`) or a model name such as `"gpt-4o"`.
- `pricing` maps a model id to `ModelRates`. `PricingTable::new().with_model(id, rates)` builds one. A repeated id replaces its rates.
- `pricing_resolver` supplies live rates and is asked before `pricing`. Returning `None` falls back to the table.
- `max_total_tokens` and `max_total_micros` cap total spend. `None` turns that cap off. `Some(0)` is rejected.
- `warn_fraction` must be above `0.0` and at most `1.0`, and needs at least one cap. `near_cap` is true when the tightest cap's used share reaches it.
- `on_unpriced_model` is `Reject` to refuse a turn whose model has no rates, or `CountTokensOnly` to count tokens, charge no money, and add 1 to `CostUsage::unpriced_turns`.
- `observers` are fixed when the plugin is built.

### Rates and currency

`ModelRates` has four fields, each a `u64` that defaults to `0`: `input_micros_per_mtok`, `output_micros_per_mtok`, `cache_read_micros_per_mtok` and `cache_write_micros_per_mtok`. The cache read rate replaces the input rate for cached tokens. The cache write rate is charged on top of the input rate.

Rates and spend are micro-units of a currency you choose, per million tokens. The crate has no prices, names no currency and does not convert. For US$3.00 per million tokens, pass `3_000_000`. Prices use integer math, so rounding errors do not build up.

### Observers

`CostObserver::observe(&CostUsage)` runs after every `on_request` charge and every `on_response_complete` commit.

- An `Err` from the `on_request` call stops the turn and undoes its charge.
- An `Err` from the `on_response_complete` call is logged by the client, and the charge stays.

## Caps and at-cap policy

| Structure | Limit | At the limit | Usage gauge |
|---|---|---|---|
| Total tokens | `max_total_tokens`, set by you, default none | `on_request` refuses the turn before it is sent and charges nothing. | `CostPlugin::usage()`, with `near_cap` |
| Total spend | `max_total_micros`, set by you, default none | Same refusal. | Same |
| Per-model entries | `max_tracked_models`, default `64` | A turn for a new model goes into one overflow entry and adds 1 to `CostUsage::overflow_turns`. Nothing is dropped, and `usage()` totals stay exact. Only the per-model split gets coarser. | `CostPlugin::breakdown()`, at most `max_tracked_models + 1` entries |

A cap checks the projected prompt total in `on_request`. The completion is charged afterward, so the final total can pass the cap.

`CostConfig::pricing` and `CostConfig::observers` belong to you and do not grow with traffic.

## Errors

| Variant | When |
|---|---|
| `Internal` | `CostPlugin::new` finds `max_tracked_models == 0`, a `Some(0)` cap, a `warn_fraction` that is not finite, at most `0.0` or above `1.0`, or a `warn_fraction` with no cap. |
| `Internal` | `CostPlugin::new` finds `UnpricedModelPolicy::Reject` with an empty `pricing` table and no `pricing_resolver`, which would refuse every turn. |
| `Internal` | The tiktoken encoder named by `encoder_name` cannot be loaded. |
| `Internal` | The encoder or ledger lock is poisoned, or `ToolCall` arguments cannot be serialized. |
| `HookFailure`, plugin `"cost-accounting"`, stage `"request"` | The projected token total passes `max_total_tokens` (`token budget exceeded: ...`). |
| `HookFailure`, same plugin and stage | The projected spend passes `max_total_micros` (`currency budget exceeded: ...`). |
| `HookFailure`, same plugin and stage | The model has no rates and `on_unpriced_model` is `Reject` (`no rates for model '...'`). |

A `CostObserver` error in `on_request` is returned as the observer produced it.

## Related pages

- [Telemetry](@/plugins/telemetry.md): the `OtelCostObserver` gauges. `examples/cost_otel.rs` shows them.
- [Memory discipline](@/concepts/memory-discipline.md): why a cap refuses a turn and the overflow entry never drops data.
- [Prompt cache](@/services/prompt-cache.md): a cache hit still charges the ledger.
