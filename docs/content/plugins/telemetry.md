+++
title = "Telemetry"
description = "Report request, token and latency metrics through OpenTelemetry, and log each request with tracing."
template = "page.html"
weight = 10
+++

# Telemetry

<dl class="page-facts">
<dt>In one line</dt>
<dd><code>OpenTelemetryPlugin</code> records request counts, streamed block counts and request duration on an OpenTelemetry meter that you own, and writes a log line for each request.</dd>
<dt>You need</dt>
<dd>The <code>plugin-telemetry</code> feature and an OpenTelemetry meter provider (a <code>&amp;dyn opentelemetry::metrics::MeterProvider</code>).</dd>
<dt>Read this if</dt>
<dd>You want to see how your client behaves in a metrics dashboard such as Prometheus or Grafana.</dd>
</dl>

OpenTelemetry is a standard way to collect metrics. A meter provider is the object that receives measurements and sends them to an exporter. You create and configure it. The plugin installs no global provider.

## Register it

This example reads the metrics back from an in-memory exporter. In production, use the exporter for your backend.

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

Expected effect: after each turn, three series appear on the meter named `"cuca_client"`: `cuca_requests_total`, `cuca_streamed_tokens_total` and `cuca_request_duration_seconds`.

## Run the example

`examples/telemetry.rs` runs two turns and prints every exported series after one `force_flush`. It needs a `llama-server` on port 1234. Put `--base-url` and `--model` after `--` to use another OpenAI-compatible server.

```bash,name=Runs the same on all three platforms
cargo run --example telemetry --features "provider-llamacpp plugin-telemetry"
```

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

`cuca_streamed_tokens_total` is 153 because it counts every streamed block, `Thinking` blocks included: `2 + 78 + 1 + 72`. Counts and durations depend on the model and the machine. The series names, kinds and attribute keys do not.

## Feature and entry type

| Item | Value |
|---|---|
| Feature flag | `plugin-telemetry` |
| Entry type | `OpenTelemetryPlugin`, built with `OpenTelemetryPlugin::new(&dyn MeterProvider)` |
| Plugin name | `"opentelemetry-observability"` |
| Attachment | `register_plugin` |
| Needs | A meter provider you own |
| Cost bridge | `OtelCostObserver`, available when `plugin-cost` and `plugin-telemetry` are both on |

## Hooks

`OpenTelemetryPlugin` acts on `on_request`, `on_stream_chunk` and `on_response_complete`.

- `on_request` adds 1 to `cuca_requests_total` with the attributes `model` and `provider`, then logs the request.
- `on_stream_chunk` adds 1 to `cuca_streamed_tokens_total` for each streamed block, with no attributes. A block can hold several tokens, so this is an approximation of a token count.
- `on_response_complete` records `UnifiedResponse::duration_secs` into `cuca_request_duration_seconds` with the attribute `status="success"`, then logs the result.

### Instruments

An instrument is one named metric. The plugin creates all three once, from the meter named `"cuca_client"`.

| Instrument | Kind | Attributes |
|---|---|---|
| `cuca_requests_total` | `Counter<u64>` | `model`, `provider` |
| `cuca_streamed_tokens_total` | `Counter<u64>` | none |
| `cuca_request_duration_seconds` | `Histogram<f64>` | `status` |

### Logs

`tracing::info!` events under the target `cuca::telemetry`:

- When a request is sent: `model` and `provider`.
- When a response completes: `duration`, `prompt_tokens` and `completion_tokens`.

With a no-op meter provider, the plugin only writes these logs.

### Cost bridge

`OtelCostObserver` reports the [cost ledger](@/plugins/cost.md) to the same `"cuca_client"` meter. Build it with `OtelCostObserver::new(&dyn MeterProvider)` and attach it through `CostConfig::observers`. It lives in the core crate, so neither plugin needs to name the other.

It creates ten `Gauge<u64>` instruments, one per `CostUsage` field. Each is recorded, with no attributes, after every `on_request` charge and every `on_response_complete` commit. They are gauges because each value is a running total.

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

The two budget caps are not recorded. They are fixed when the cost plugin is built.

`examples/cost_otel.rs` registers `CostPlugin`, `OpenTelemetryPlugin` and `OtelCostObserver` on one meter provider and prints all ten gauges after a priced turn.

```bash,name=Runs the same on all three platforms
cargo run --example cost_otel --features "provider-llamacpp plugin-cost plugin-telemetry"
```

## Configuration

`OpenTelemetryPlugin::new` takes only the meter provider. It has no config struct and no keys.

| Fixed value | Setting |
|---|---|
| Meter name | `"cuca_client"`, shared with `OtelCostObserver` |
| Log target | `cuca::telemetry` |

Exporting, aggregation and temporality are set on your meter provider and reader.

## Caps and at-cap policy

No cap. The plugin holds three instrument handles, and nothing grows with traffic. Your meter provider accumulates the data. `OtelCostObserver` holds ten gauge handles and no state. There is no usage gauge.

## Errors

None. Every hook returns `Ok(())`, and `OtelCostObserver::observe` always returns `Ok(())`, so neither can stop a turn.

## Related pages

- [Cost accounting](@/plugins/cost.md): the ledger that `OtelCostObserver` reports.
- [Memory discipline](@/concepts/memory-discipline.md): why capped structures expose a usage reading and leave alerting to your meter.
- [Plugin layering](@/concepts/plugin-layering.md): why the bridge lives in the core crate.
