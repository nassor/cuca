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

## Entry types

`OpenTelemetryPlugin`.

## `CucaPlugin`

`OpenTelemetryPlugin` implements `CucaPlugin` with the plugin name `"opentelemetry-observability"`. It overrides `on_request`, `on_stream_chunk`, and `on_response_complete`.

## Instruments

Created once from the meter named `"cuca_client"`:

| Instrument | Kind | Recorded |
|---|---|---|
| `cuca_requests_total` | `Counter<u64>` | Incremented by 1 in `on_request`, with `model` and `provider` attributes |
| `cuca_streamed_tokens_total` | `Counter<u64>` | Incremented by 1 in `on_stream_chunk`, once per streamed content block |
| `cuca_request_duration_seconds` | `Histogram<f64>` | Recorded in `on_response_complete` with `res.duration_secs` |

The token counter is a coarse approximation: one increment per streamed block, not a real token count.

## Logs

`tracing::info!` events under target `cuca::telemetry` on request dispatch (`model`, `provider`) and on response completion (`duration`, `prompt_tokens`, `completion_tokens`).

## Capacity

No growth cap. Accumulation and export are the caller's OpenTelemetry meter provider's responsibility; no default meter provider is installed by this plugin.
