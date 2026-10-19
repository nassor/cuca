+++
title = "Telemetry"
description = "The OpenTelemetry observability plugin: its three instruments, the meter name, and the tracing log events."
template = "page.html"
weight = 10
+++

# Telemetry

<dl class="page-facts">
<dt>In one line</dt>
<dd>Records request, token, and latency metrics on a caller-supplied OpenTelemetry meter provider, and emits structured logs.</dd>
<dt>You need</dt>
<dd>The <code>plugin-telemetry</code> feature and a <code>&amp;dyn opentelemetry::metrics::MeterProvider</code>.</dd>
<dt>Read this if</dt>
<dd>You are registering <code>OpenTelemetryPlugin</code> or wiring its metrics into an exporter.</dd>
</dl>

## Feature and entry type

| Item | Value |
|---|---|
| Feature flag | `plugin-telemetry` (enables `opentelemetry`, `opentelemetry_sdk`, `tracing`) |
| Entry type | `OpenTelemetryPlugin`, built with `OpenTelemetryPlugin::new(&dyn MeterProvider)` |
| Plugin name | `"opentelemetry-observability"` |
| Attachment | `register_plugin` |
| Needs | A meter provider the caller owns; the plugin installs no global provider |
| Cost bridge | `OtelCostObserver`, compiled when `plugin-cost` and `plugin-telemetry` are both enabled |

## Hooks

`OpenTelemetryPlugin` overrides `on_request`, `on_stream_chunk`, and `on_response_complete`. `execute_local_tool` uses the trait default.

- `on_request` adds 1 to `cuca_requests_total` with the attributes `model` and `provider`, then logs the dispatch.
- `on_stream_chunk` adds 1 to `cuca_streamed_tokens_total` with no attributes, once per streamed content block. It is an approximation of a token count, because a block can span several tokens.
- `on_response_complete` records `UnifiedResponse::duration_secs` into `cuca_request_duration_seconds` with the attribute `status="success"`, then logs the completion.

### Instruments

The plugin creates all three once in `new`, from the meter named `"cuca_client"`.

| Instrument | Kind | Attributes |
|---|---|---|
| `cuca_requests_total` | `Counter<u64>` | `model`, `provider` |
| `cuca_streamed_tokens_total` | `Counter<u64>` | none |
| `cuca_request_duration_seconds` | `Histogram<f64>` | `status` |

### Logs

`tracing::info!` events under the target `cuca::telemetry`:

- On request dispatch: `model` and `provider`.
- On response completion: `duration`, `prompt_tokens`, and `completion_tokens`.

A no-op meter provider yields a plugin that only emits these logs.

### Cost bridge

`OtelCostObserver` implements `CostObserver` and records the cost ledger to the same `"cuca_client"` meter. It is built with `OtelCostObserver::new(&dyn MeterProvider)` and attached through `CostConfig::observers` (see [Cost accounting](@/plugins/cost.md)). It lives in core, at `cuca::cost_otel`, because neither plugin may name the other.

Its ten instruments are `Gauge<u64>`, recorded with no attributes after every `on_request` charge and every `on_response_complete` commit, one per `CostUsage` field. They are gauges because each reading is a cumulative snapshot and the observer seam carries no deltas.

| Instrument | `CostUsage` field |
|---|---|
| `cuca_cost_spent_micros` | `spent_micros` |
| `cuca_cost_prompt_tokens` | `prompt_tokens` |
| `cuca_cost_completion_tokens` | `completion_tokens` |
| `cuca_cost_cache_read_tokens` | `cache_read_tokens` |
| `cuca_cost_cache_write_tokens` | `cache_write_tokens` |
| `cuca_cost_turns` | `turns` |
| `cuca_cost_unpriced_turns` | `unpriced_turns` |
| `cuca_cost_overflow_turns` | `overflow_turns` |
| `cuca_cost_untokenized_image_blocks` | `untokenized_image_blocks` |
| `cuca_cost_near_cap` | `near_cap`, recorded as `1` or `0` |

The two configured caps are not recorded: they are construction-time constants.

## Configuration

`OpenTelemetryPlugin::new` takes only the meter provider. It has no config struct and no keys.

| Fixed value | Setting |
|---|---|
| Meter name | `"cuca_client"`, shared with `OtelCostObserver` |
| Log target | `cuca::telemetry` |

Export, aggregation, and temporality belong to the caller's meter provider and reader.

## Caps and at-cap policy

No cap: the plugin holds three instrument handles and nothing grows with traffic. Accumulation is the caller's meter provider's job. `OtelCostObserver` holds ten gauge handles, no state, and takes no lock of its own. There is no usage gauge.

## Errors

None. Every hook returns `Ok(())`, and `OtelCostObserver::observe` always returns `Ok(())`, so neither aborts a turn.

## Example

`examples/telemetry.rs` registers the plugin on an in-memory exporter, runs two live turns, and prints every exported series after one `force_flush`.

```rust,name=Register the plugin on an in-memory exporter
let exporter = InMemoryMetricExporter::default();
let meter_provider = SdkMeterProvider::builder()
    .with_reader(PeriodicReader::builder(exporter.clone()).build())
    .build();
let otel = Arc::new(OpenTelemetryPlugin::new(&meter_provider));
let client = CucaClient::builder()
    .with_provider(ProviderEndpoint::LlamaCpp)
    .with_base_url(base_url)
    .register_plugin(Arc::clone(&otel) as Arc<dyn CucaPlugin>)
    .build()?;
```

```bash,name=Runs the same on all three platforms
cargo run --example telemetry --features "provider-llamacpp plugin-telemetry"
```

It needs a `llama-server` on port 1234. `CUCA_BASE_URL` and `CUCA_MODEL` retarget it at another OpenAI-compatible server.

```text,name=Expected output
Meter "cuca_client" on an SdkMeterProvider reading into InMemoryMetricExporter

Turn 1: "Name one bird. Reply with the name only."
  reply: Sparrow
  blocks: 2 text, 78 thinking

Turn 2: "Name one fish. Reply with the name only."
  reply: Salmon
  blocks: 1 text, 72 thinking

Exported after force_flush, scope "cuca_client"
  cuca_requests_total  Sum<u64>  Total LLM requests dispatched by CUCA
    2  model="google/gemma-4-12b-qat" provider="LlamaCpp"
  cuca_request_duration_seconds  Histogram<f64>  Latency distribution of LLM execution turns
    count 2, sum 57.034, min 25.250, max 31.784  status="success"
  cuca_streamed_tokens_total  Sum<u64>  Total streaming tokens processed across sessions
    153  no attributes
```

`cuca_streamed_tokens_total` reaches 153 because it counts every streamed block, `Thinking` blocks included: `2 + 78 + 1 + 72`. The counts and durations come from `google/gemma-4-12b-qat` on one machine and change with both. The series names, kinds, and attribute keys do not.

`examples/cost_otel.rs` registers `CostPlugin`, `OpenTelemetryPlugin`, and `OtelCostObserver` on one meter provider and prints all ten gauges after a live priced turn.

```bash,name=Runs the same on all three platforms
cargo run --example cost_otel --features "provider-llamacpp plugin-cost plugin-telemetry"
```

## Related pages

- [Cost accounting](@/plugins/cost.md): the ledger that `OtelCostObserver` reports.
- [Memory discipline](@/concepts/memory-discipline.md): why capped structures expose a usage reading and leave alerting to the caller's meter.
- [Plugin layering](@/concepts/plugin-layering.md): why the bridge lives in core.
