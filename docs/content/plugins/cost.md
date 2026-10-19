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

## Feature and entry type

| Item | Value |
|---|---|
| Feature flag | `plugin-cost` (enables `tiktoken-rs`, which `plugin-memory` also uses; this is a shared dependency, not a feature edge) |
| Entry type | `CostPlugin`, built with `CostPlugin::new(CostConfig)` |
| Related types | `CostConfig`, `CostUsage`, `CostEntry`, `CostObserver`, `PricingTable`, `PricingResolver`, `ModelRates`, `UnpricedModelPolicy` |
| Plugin name | `"cost-accounting"` |
| Attachment | `register_plugin` |
| Needs | A `PricingTable` or `PricingResolver` to price spend; without one, turns are counted in tokens only |
| Telemetry bridge | `OtelCostObserver`, compiled when `plugin-telemetry` is also enabled (see [Telemetry](@/plugins/telemetry.md)) |

## Hooks

`CostPlugin` overrides `on_request` and `on_response_complete`. `execute_local_tool` and `on_stream_chunk` use the trait defaults: the plugin owns no tool, and a budget cannot abort a turn mid-stream.

- `on_request` estimates the prompt tokens, prices them at the input rate, and projects the post-charge totals against `max_total_tokens` and `max_total_micros`. A projected total over either cap refuses the turn and charges nothing.
- After the cap check, `on_request` commits the charge and, when `near_cap` is true and no warning is present, appends a system message starting `CUCA cost warning:`. Then every `CostObserver` receives the new `CostUsage`.
- `on_response_complete` estimates the completion tokens from `res.content`, prices them at the output rate, commits the model's bucket, and calls the observers again.
- When `res.prompt_cache_usage` is present, `on_response_complete` refunds the cached read tokens at the input rate and charges them at the cache read rate. It also charges the cache write tokens at the cache write rate.

### What the ledger counts

Every figure is a tiktoken estimate. `CostPlugin` never reads `UnifiedResponse::prompt_tokens`, `completion_tokens`, or `usage`, because the prompt charge must be made before the provider has counted anything and one estimator keeps every provider consistent. `prompt_cache_usage`, filled by the Anthropic adapter only, is the only provider figure it applies.

- The prompt estimate encodes each message's role label, every `Text` block, every `Thinking.reasoning`, every `ToolResult.output`, every `ToolCall` name with its JSON arguments, and every tool definition in `req.tools` (name, description, input schema).
- The prompt estimate skips `ImageBase64` blocks, per-message framing tokens, and `max_tokens` and `temperature`. Each skipped image increments `CostUsage::untokenized_image_blocks`.
- The completion estimate encodes `Text`, `Thinking.reasoning`, and `ToolCall` name plus arguments. `ToolResult` and `ImageBase64` blocks add nothing.

### Ledger edge cases

- A turn that fails after `on_request`, from a provider error or a later plugin's `on_request` error, keeps its prompt charge, because no terminal hook fires. A budget cap fails safe.
- A client-level cache hit (`service-prompt-cache`) still runs `on_response_complete` on the replayed response, so a cached turn is charged. The ledger reads as gross spend before caching.
- The near-cap warning changes the request, so it changes the digest a client-level response cache computes afterward. Neither plugin requires a registration position.

### Accessors

- `CostPlugin::usage()` returns a `Copy` `CostUsage` reading under one lock.
- `CostPlugin::breakdown()` returns `Vec<(String, CostEntry)>`, sorted by model id.
- `CostPlugin::reset()` zeroes the ledger for a billing-period rollover and leaves the configuration untouched.
- `CostPlugin::estimate_request_tokens(req)` runs the hooks' own estimator, callable before a client exists.
- `CostPlugin::rates_for(model)` returns the resolver's rates first, then the `pricing` table's.

## Configuration

`CostConfig::default()` carries the defaults below. `CostPlugin::new` validates the config and loads the encoder.

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
- `pricing` maps a model id to `ModelRates`. `PricingTable::new().with_model(id, rates)` builds one, and a repeated id replaces its rates.
- `pricing_resolver` supplies live rates and is consulted before `pricing`. Returning `None` falls through to the table.
- `max_total_tokens` and `max_total_micros` cap cumulative spend. `None` disables enforcement on that axis, and `Some(0)` is rejected.
- `warn_fraction` must be in `(0.0, 1.0]` and needs at least one cap. `near_cap` is true when the tightest cap's used fraction reaches it.
- `on_unpriced_model` is `Reject` to refuse a turn whose model has no rates, or `CountTokensOnly` to charge tokens, charge no currency, and increment `CostUsage::unpriced_turns`.
- `observers` are fixed at construction.

### Rates and currency

`ModelRates` holds four fields, each a `u64` that defaults to `0`: `input_micros_per_mtok`, `output_micros_per_mtok`, `cache_read_micros_per_mtok`, and `cache_write_micros_per_mtok`. The cache read rate replaces the input rate for cached tokens. The cache write rate is charged on top of the input rate.

Rates and spend are micro-units of a caller-defined currency per million tokens. The crate ships no prices, names no currency, and never converts. Pass `3_000_000` for US$3.00 per million tokens if the currency is USD. Prices use integer math with `u128` intermediates, so rounding does not accumulate.

### Observers

`CostObserver::observe(&CostUsage)` runs after every `on_request` charge and every `on_response_complete` commit.

- An `Err` from the `on_request` call aborts the turn and rolls its charge back.
- An `Err` from the `on_response_complete` call is logged by the client, and the commit stands.

## Caps and at-cap policy

| Structure | Bound | At-cap policy | Usage gauge |
|---|---|---|---|
| Cumulative tokens | `max_total_tokens`, caller-set, default none | `on_request` refuses the turn before dispatch and charges nothing | `CostPlugin::usage()`, with `near_cap` |
| Cumulative spend | `max_total_micros`, caller-set, default none | Same refusal | Same |
| Per-model entries | `max_tracked_models`, default `64` | A turn for an untracked model folds into one reserved overflow bucket and increments `CostUsage::overflow_turns`. Nothing is evicted, and `usage()` totals stay exact; only per-model attribution degrades | `CostPlugin::breakdown()`, at most `max_tracked_models + 1` entries |

A cap gates the projected prompt total in `on_request`. The completion is charged afterward, so the committed total can end past its cap. A cap bounds what is dispatched, not what a turn already in flight bills.

`CostConfig::pricing` and `CostConfig::observers` are caller-owned and never grow from traffic.

## Errors

| Variant | When |
|---|---|
| `Internal` | `CostPlugin::new` finds `max_tracked_models == 0`, a `Some(0)` cap, a `warn_fraction` that is non-finite, at most `0.0`, or above `1.0`, or a `warn_fraction` with no cap |
| `Internal` | `CostPlugin::new` finds `UnpricedModelPolicy::Reject` with an empty `pricing` table and no `pricing_resolver`, which would refuse every turn |
| `Internal` | The tiktoken encoder named by `encoder_name` cannot be loaded |
| `Internal` | The encoder or ledger lock is poisoned, or `ToolCall` arguments cannot be serialized |
| `HookFailure`, plugin `"cost-accounting"`, stage `"request"` | The projected token total passes `max_total_tokens` (`token budget exceeded: ...`) |
| `HookFailure`, same plugin and stage | The projected spend passes `max_total_micros` (`currency budget exceeded: ...`) |
| `HookFailure`, same plugin and stage | The model has no rates and `on_unpriced_model` is `Reject` (`no rates for model '...'`) |

A `CostObserver` error in `on_request` is returned as the observer produced it.

## Example

`examples/cost.rs` prices the demo model, caps the ledger at 160 tokens, and dispatches a growing conversation until the cap refuses a turn.

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

```bash,name=Runs the same on all three platforms
cargo run --example cost --features "provider-llamacpp plugin-cost"
```

It needs a `llama-server` on port 1234. `CUCA_BASE_URL` and `CUCA_MODEL` retarget it at another OpenAI-compatible server.

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

Turn 2 crosses `warn_fraction` and carries the injected warning. Turn 3 is refused in `on_request` and charges nothing. The total ends at `171/160` because the completion is charged after the gate. Which turn crosses the cap depends on the model: every figure is a tiktoken estimate of what `google/gemma-4-12b-qat` emitted, most of it `Thinking` blocks.

`examples/cost_otel.rs` runs the telemetry bridge: one priced live turn, the ledger read three ways, and the gauges read back through an in-memory exporter.

```bash,name=Runs the same on all three platforms
cargo run --example cost_otel --features "provider-llamacpp plugin-cost plugin-telemetry"
```

## Related pages

- [Telemetry](@/plugins/telemetry.md): the `OtelCostObserver` gauges.
- [Memory discipline](@/concepts/memory-discipline.md): why a cap refuses a turn and the overflow bucket never evicts.
- [Prompt cache](@/services/prompt-cache.md): a cache hit still charges the ledger.
