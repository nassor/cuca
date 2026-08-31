//! Price one live turn and export the cost ledger to OpenTelemetry.
//!
//! The demo registers two plugins on one meter provider: `CostPlugin` with
//! [`cuca::OtelCostObserver`] attached, and `OpenTelemetryPlugin`. The
//! observer is the core bridge between the two features; neither plugin knows
//! the other exists.
//!
//! # Prerequisites
//!
//! - A checkout of this repository (the example depends on `cuca-core` by path).
//! - A running [llama.cpp](https://github.com/ggml-org/llama.cpp) server
//!   (`llama-server`) on port 1234 with the demo model loaded.
//!
//! # Run
//!
//! ```sh
//! cargo run --example cost_otel --features provider-llamacpp,plugin-cost,plugin-telemetry
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
//! Example: `CUCA_BASE_URL=http://127.0.0.1:8000/v1 CUCA_MODEL=<server-model-id> cargo run --example cost_otel --features provider-llamacpp,plugin-cost,plugin-telemetry`
//!
//! # Output
//!
//! The reply prints as text chunks. Three blocks follow it: the `CostUsage`
//! reading, the per-model breakdown, and the ten gauges the bridge recorded,
//! read back through an in-memory exporter. Two of those ten lines, with the
//! numbers a short prompt produced:
//!
//! ```text
//! Exported OTel gauges
//!   cuca_cost_prompt_tokens             = 16
//!   cuca_cost_spent_micros              = 258
//! ```
//!
//! With no server on the base URL, the program prints one line naming the
//! address and exits successfully.
//!
//! # Prices are yours
//!
//! `PricingTable` below carries a made-up rate, in micro-units of a
//! caller-defined currency per million tokens. The crate ships no vendor
//! rates and never names a currency: replace the rate with your own before
//! reading any number here as money.

use std::io::{Write, stdout};
use std::sync::Arc;

use cuca::plugin::CucaPlugin;
use cuca::types::{MessageContentBlock, ProviderEndpoint};
use cuca::{
    CostConfig, CostPlugin, CucaClient, ModelRates, OpenTelemetryPlugin, OtelCostObserver,
    PricingTable, UnifiedRequest,
};
use opentelemetry_sdk::metrics::data::{AggregatedMetrics, MetricData};
use opentelemetry_sdk::metrics::{InMemoryMetricExporter, PeriodicReader, SdkMeterProvider};
use tokio_stream::StreamExt;

/// Every gauge the bridge records, read back from the exporter and sorted by
/// name so the printout is stable.
fn exported_cost_gauges(exporter: &InMemoryMetricExporter) -> Vec<(String, u64)> {
    let batches = exporter.get_finished_metrics().unwrap_or_default();
    let mut gauges: Vec<(String, u64)> = batches
        .iter()
        .flat_map(|rm| rm.scope_metrics())
        .flat_map(|sm| sm.metrics())
        .filter(|metric| metric.name().starts_with("cuca_cost_"))
        .filter_map(|metric| match metric.data() {
            AggregatedMetrics::U64(MetricData::Gauge(gauge)) => gauge
                .data_points()
                .next()
                .map(|dp| (metric.name().to_string(), dp.value())),
            _ => None,
        })
        .collect();
    gauges.sort_unstable();
    gauges
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Base URL and model come from the environment so the example runs
    // against any OpenAI-compatible server; the defaults target a local
    // llama.cpp server (see the module docs for the override recipe).
    let base_url =
        std::env::var("CUCA_BASE_URL").unwrap_or_else(|_| "http://127.0.0.1:1234/v1".to_string());
    let model = std::env::var("CUCA_MODEL").unwrap_or_else(|_| "google/gemma-4-e4b".to_string());

    // Stage 1: the caller's meter pipeline. An in-memory exporter keeps the
    // demo self-contained; a real deployment swaps in OTLP and never reads the
    // batch back by hand. The crate installs no global provider.
    let exporter = InMemoryMetricExporter::default();
    let meter_provider = SdkMeterProvider::builder()
        .with_reader(PeriodicReader::builder(exporter.clone()).build())
        .build();

    // Stage 2: the ledger. The rate is per million tokens, in micro-units of
    // whichever currency the caller means. `warn_fraction` needs a cap, and
    // injects one near-limit system message once the tighter cap is 80% used.
    let cost = Arc::new(CostPlugin::new(CostConfig {
        pricing: PricingTable::new().with_model(
            &model,
            ModelRates {
                input_micros_per_mtok: 3_000_000,
                output_micros_per_mtok: 15_000_000,
                ..Default::default()
            },
        ),
        max_total_tokens: Some(50_000),
        warn_fraction: Some(0.8),
        // The bridge: every reading reaches the meter provider above, with no
        // observer of the caller's own.
        observers: vec![Arc::new(OtelCostObserver::new(&meter_provider))],
        ..Default::default()
    })?);

    // Stage 3: both plugins on that one provider. `OpenTelemetryPlugin` adds
    // the request counter, the token counter, and the latency histogram under
    // the same `cuca_client` meter the bridge uses.
    let client = CucaClient::builder()
        .with_provider(ProviderEndpoint::LlamaCpp)
        .with_base_url(base_url.clone())
        .register_plugin(Arc::clone(&cost) as Arc<dyn CucaPlugin>)
        .register_plugin(Arc::new(OpenTelemetryPlugin::new(&meter_provider)))
        .build()?;

    let request = UnifiedRequest::new(&model)
        .add_system_message("You are concise.")
        .add_user_message("Explain CUCA in one sentence.");

    // Stage 4: one turn. A refused connection is the expected outcome with no
    // server up, so it reports the address rather than failing the process.
    let mut stream = match client.generate_stream(request).await {
        Ok(stream) => stream,
        Err(error) => {
            println!("No server answered at {base_url}: {error}");
            println!("Start llama-server there, or set CUCA_BASE_URL, then run this again.");
            return Ok(());
        }
    };
    while let Some(chunk) = stream.next().await {
        match chunk {
            Ok(MessageContentBlock::Text(text)) => {
                print!("{text}");
                stdout().flush()?;
            }
            Ok(_) => {}
            Err(error) => {
                println!("\nThe stream ended early: {error}");
                break;
            }
        }
    }
    println!();

    // Stage 5: the same numbers from three places. `usage()` and `breakdown()`
    // read the ledger directly; the gauges are what the bridge recorded on the
    // way through.
    let usage = cost.usage()?;
    println!("\nCost ledger");
    println!("  turns                     {}", usage.turns);
    println!("  prompt tokens             {}", usage.prompt_tokens);
    println!("  completion tokens         {}", usage.completion_tokens);
    println!("  spent (micro-units)       {}", usage.spent_micros);
    println!("  unpriced turns            {}", usage.unpriced_turns);
    println!("  near cap                  {}", usage.near_cap);

    println!("\nPer-model breakdown");
    for (model_id, entry) in cost.breakdown()? {
        println!(
            "  {model_id}: prompt={} completion={} spent={}",
            entry.prompt_tokens, entry.completion_tokens, entry.spent_micros
        );
    }

    // A real pipeline exports on the reader's own schedule; this flush is what
    // makes the values readable inside one short program.
    meter_provider.force_flush()?;
    println!("\nExported OTel gauges");
    for (name, value) in exported_cost_gauges(&exporter) {
        println!("  {name:<36}= {value}");
    }
    Ok(())
}
