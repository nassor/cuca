//! Read the three CUCA instruments out of an in-memory OpenTelemetry exporter.
//!
//! An `SdkMeterProvider` is built over `InMemoryMetricExporter`, handed to
//! `OpenTelemetryPlugin`, and the plugin is registered on the client. Two real
//! turns then stream through it, so the request counter, the streamed-block
//! counter, and the latency histogram all move for real. One `force_flush`
//! later the demo prints every exported series with its data points, its
//! attributes, and the histogram's own count and sum.
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
//! cargo run --example telemetry --features provider-llamacpp,plugin-telemetry
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
//! Example: `CUCA_BASE_URL=http://127.0.0.1:8000/v1 CUCA_MODEL=<server-model-id> cargo run --example telemetry --features provider-llamacpp,plugin-telemetry`
//!
//! # Output
//!
//! From one run against `google/gemma-4-12b-qat` on llama.cpp:
//!
//! ```text
//! Meter "cuca_client" on an SdkMeterProvider reading into InMemoryMetricExporter
//!
//! Turn 1: "Name one bird. Reply with the name only."
//!   reply: Sparrow
//!   blocks: 2 text, 78 thinking
//!
//! Turn 2: "Name one fish. Reply with the name only."
//!   reply: Salmon
//!   blocks: 1 text, 72 thinking
//!
//! Exported after force_flush, scope "cuca_client"
//!   cuca_requests_total  Sum<u64>  Total LLM requests dispatched by CUCA
//!     2  model="google/gemma-4-12b-qat" provider="LlamaCpp"
//!   cuca_request_duration_seconds  Histogram<f64>  Latency distribution of LLM execution turns
//!     count 2, sum 57.034, min 25.250, max 31.784  status="success"
//!   cuca_streamed_tokens_total  Sum<u64>  Total streaming tokens processed across sessions
//!     153  no attributes
//! ```
//!
//! The counters are the honest shape of the instruments. `cuca_requests_total`
//! carries one data point per model and provider pair. `cuca_streamed_tokens_total`
//! counts one per streamed content block with no attributes at all, so its 153
//! is exactly the block counts printed above, `2 + 78 + 1 + 72`, and not a token
//! count. The numbers depend on the model and the machine. The series names,
//! their kinds, and their attribute keys do not.
//!
//! With no server on the base URL, the program prints one line naming the
//! address and exits successfully.
//!
//! # Why the exporter is the caller's
//!
//! `OpenTelemetryPlugin::new` takes a `&dyn MeterProvider` and installs nothing
//! global, so the plugin composes with whatever pipeline an application already
//! runs. This demo hands it the in-memory exporter because that is the one
//! pipeline whose output can be printed to a terminal; a deployment passes the
//! same handle its OTLP pipeline is built from, and reads the same three series
//! on the collector side.

use std::sync::Arc;

use cuca::plugin::CucaPlugin;
use cuca::types::{MessageContentBlock, ProviderEndpoint};
use cuca::{AgentResponseStream, CucaClient, OpenTelemetryPlugin, UnifiedRequest};
use opentelemetry_sdk::metrics::data::{AggregatedMetrics, MetricData, ResourceMetrics};
use opentelemetry_sdk::metrics::{InMemoryMetricExporter, PeriodicReader, SdkMeterProvider};
use tokio_stream::StreamExt;

/// Per-turn completion cap. A reasoning model spends most of it on `Thinking`
/// blocks, so a smaller budget returns an empty reply.
const MAX_TOKENS: u32 = 512;

/// Two turns, so the request counter and the histogram both hold more than one
/// point.
const PROMPTS: [&str; 2] = [
    "Name one bird. Reply with the name only.",
    "Name one fish. Reply with the name only.",
];

/// Attribute set of one data point, as `key="value"` pairs.
fn attributes<'a>(mut attrs: impl Iterator<Item = &'a opentelemetry::KeyValue>) -> String {
    let mut out = String::new();
    for attr in attrs.by_ref() {
        if !out.is_empty() {
            out.push(' ');
        }
        out.push_str(&format!("{}={:?}", attr.key.as_str(), attr.value.as_str()));
    }
    if out.is_empty() {
        out.push_str("no attributes");
    }
    out
}

/// Print every exported series: name, kind, description, then its data points.
///
/// The two instrument shapes this plugin creates are a `Sum<u64>` (both
/// counters, under cumulative temporality, so one flush carries running
/// totals) and a `Histogram<f64>`.
fn print_series(metrics: &[ResourceMetrics]) {
    for scope in metrics.iter().flat_map(ResourceMetrics::scope_metrics) {
        println!(
            "\nExported after force_flush, scope {:?}",
            scope.scope().name()
        );
        for metric in scope.metrics() {
            match metric.data() {
                AggregatedMetrics::U64(MetricData::Sum(sum)) => {
                    println!("  {}  Sum<u64>  {}", metric.name(), metric.description());
                    for point in sum.data_points() {
                        println!("    {}  {}", point.value(), attributes(point.attributes()));
                    }
                }
                AggregatedMetrics::F64(MetricData::Histogram(histogram)) => {
                    println!(
                        "  {}  Histogram<f64>  {}",
                        metric.name(),
                        metric.description()
                    );
                    for point in histogram.data_points() {
                        println!(
                            "    count {}, sum {:.3}, min {:.3}, max {:.3}  {}",
                            point.count(),
                            point.sum(),
                            point.min().unwrap_or(0.0),
                            point.max().unwrap_or(0.0),
                            attributes(point.attributes())
                        );
                    }
                }
                other => println!("  {}  {other:?}", metric.name()),
            }
        }
    }
}

/// Drain a turn into its text plus per-kind block counts.
///
/// The block counts are what `cuca_streamed_tokens_total` counts: one
/// increment per streamed block, `Thinking` blocks included.
async fn drain(mut stream: AgentResponseStream) -> (String, usize, usize) {
    let mut text = String::new();
    let mut text_blocks = 0usize;
    let mut thinking_blocks = 0usize;
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
    (text, text_blocks, thinking_blocks)
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Base URL and model come from the environment so the example runs
    // against any OpenAI-compatible server; the defaults target a local
    // llama.cpp server (see the module docs for the override recipe).
    let base_url =
        std::env::var("CUCA_BASE_URL").unwrap_or_else(|_| "http://127.0.0.1:1234/v1".to_string());
    let model = std::env::var("CUCA_MODEL").unwrap_or_else(|_| "google/gemma-4-e4b".to_string());

    // The plain `PeriodicReader` exports from a dedicated OS thread, so
    // `force_flush` below cannot deadlock this current-thread runtime.
    let exporter = InMemoryMetricExporter::default();
    let meter_provider = SdkMeterProvider::builder()
        .with_reader(PeriodicReader::builder(exporter.clone()).build())
        .build();
    let otel = Arc::new(OpenTelemetryPlugin::new(&meter_provider));
    let client = CucaClient::builder()
        .with_provider(ProviderEndpoint::LlamaCpp)
        .with_base_url(base_url.clone())
        .register_plugin(Arc::clone(&otel) as Arc<dyn CucaPlugin>)
        .build()?;

    println!("Meter \"cuca_client\" on an SdkMeterProvider reading into InMemoryMetricExporter");

    for (index, prompt) in PROMPTS.iter().enumerate() {
        println!("\nTurn {}: {prompt:?}", index + 1);
        let request = UnifiedRequest::new(&model)
            .add_system_message("You are concise.")
            .add_user_message(*prompt)
            .set_max_tokens(MAX_TOKENS);
        let stream = match client.generate_stream(request).await {
            Ok(stream) => stream,
            Err(error) => {
                println!("\nNo server answered at {base_url}: {error}");
                println!("Start llama-server there, or set CUCA_BASE_URL, then run this again.");
                return Ok(());
            }
        };
        let (reply, text_blocks, thinking_blocks) = drain(stream).await;
        println!("  reply: {}", reply.trim());
        println!("  blocks: {text_blocks} text, {thinking_blocks} thinking");
    }

    meter_provider.force_flush()?;
    print_series(&exporter.get_finished_metrics()?);
    Ok(())
}
