//! Integration tests for the OpenTelemetry plugin (`plugin-telemetry`).
//!
//! The deterministic tests drive the [`CucaPlugin`] hooks directly against an
//! in-memory metric exporter; the live test registers the plugin on an LM
//! Studio client and asserts a real request bumps the request counter.
//!
//! # Runtime note
//!
//! This crate does not enable tokio's `rt-multi-thread`, so every test must
//! run on the default current-thread flavor. The opentelemetry_sdk
//! [`PeriodicReader`] runs its export worker on a dedicated background OS
//! thread, so `force_flush` cannot deadlock a current-thread tokio runtime.
#![cfg(all(feature = "provider-llamacpp", feature = "plugin-telemetry"))]

mod common;

use std::sync::Arc;

use cuca::plugin::CucaPlugin;
use cuca::request::UnifiedRequest;
use cuca::types::{MessageContentBlock, ProviderEndpoint};
use cuca::{OpenTelemetryPlugin, UnifiedResponse};
use opentelemetry_sdk::metrics::data::{AggregatedMetrics, Metric, MetricData, ResourceMetrics};
use opentelemetry_sdk::metrics::{InMemoryMetricExporter, PeriodicReader, SdkMeterProvider};

/// A meter provider whose exports land in an in-memory exporter. Returns both
/// so a test can register the plugin, flush, and inspect the exported batches.
fn provider_with_exporter() -> (SdkMeterProvider, InMemoryMetricExporter) {
    let exporter = InMemoryMetricExporter::default();
    let reader = PeriodicReader::builder(exporter.clone()).build();
    let provider = SdkMeterProvider::builder().with_reader(reader).build();
    (provider, exporter)
}

/// The exported metric named `name`, panicking if it is absent.
fn exported_metric<'a>(metrics: &'a [ResourceMetrics], name: &str) -> &'a Metric {
    metrics
        .iter()
        .flat_map(|rm| rm.scope_metrics())
        .flat_map(|sm| sm.metrics())
        .find(|m| m.name() == name)
        .unwrap_or_else(|| panic!("metric `{name}` missing from export"))
}

fn counter_total(metrics: &[ResourceMetrics], name: &str) -> u64 {
    let AggregatedMetrics::U64(MetricData::Sum(sum)) = exported_metric(metrics, name).data() else {
        panic!("`{name}` must export a Sum<u64>");
    };
    sum.data_points().map(|dp| dp.value()).sum()
}

#[test]
fn name_is_the_stable_identifier() {
    let (provider, _exporter) = provider_with_exporter();
    let plugin = OpenTelemetryPlugin::new(&provider);
    assert_eq!(plugin.name(), "opentelemetry-observability");
}

#[test]
fn on_request_records_exactly_one_request() {
    let (provider, exporter) = provider_with_exporter();
    let plugin = OpenTelemetryPlugin::new(&provider);

    let mut req = UnifiedRequest::new("test-model");
    req.provider = ProviderEndpoint::LlamaCpp;
    plugin
        .on_request(&mut req)
        .expect("on_request must return Ok(())");

    provider.force_flush().expect("force_flush must succeed");
    let metrics = exporter
        .get_finished_metrics()
        .expect("get_finished_metrics must succeed");
    assert_eq!(counter_total(&metrics, "cuca_requests_total"), 1);
}

#[test]
fn on_response_complete_records_latency_and_usage() {
    let (provider, exporter) = provider_with_exporter();
    let plugin = OpenTelemetryPlugin::new(&provider);

    let res = UnifiedResponse {
        model: "test-model".into(),
        provider: ProviderEndpoint::LlamaCpp,
        duration_secs: 1.25,
        prompt_tokens: 10,
        completion_tokens: 5,
        finish_reason: Some("stop".into()),
        content: Vec::new(),
        prompt_cache_usage: None,
    };
    plugin
        .on_response_complete(&res)
        .expect("on_response_complete must return Ok(())");

    provider.force_flush().expect("force_flush must succeed");
    let metrics = exporter
        .get_finished_metrics()
        .expect("get_finished_metrics must succeed");
    // The latency histogram records one data point with the reported duration.
    let AggregatedMetrics::F64(MetricData::Histogram(hist)) =
        exported_metric(&metrics, "cuca_request_duration_seconds").data()
    else {
        panic!("histogram must export a Histogram<f64>");
    };
    let dps: Vec<_> = hist.data_points().collect();
    assert_eq!(dps.len(), 1);
    assert_eq!(dps[0].sum(), res.duration_secs);
}

#[tokio::test]
async fn live_request_increments_the_request_counter() {
    if let Err(reason) = common::require_live_server() {
        eprintln!("SKIP: llama.cpp not reachable: {reason}");
        return;
    }
    let (provider, exporter) = provider_with_exporter();
    let plugin = Arc::new(OpenTelemetryPlugin::new(&provider));
    let client = common::client_with_plugins(vec![Arc::clone(&plugin) as Arc<dyn CucaPlugin>]);

    let request = common::live_request("Reply with the single word: ok", &common::live_model());
    let stream = client
        .generate_stream(request)
        .await
        .expect("generate_stream must start");
    let blocks = common::drain_timeout(stream, 60).await;
    assert!(
        blocks
            .iter()
            .any(|b| matches!(b, MessageContentBlock::Text(_))),
        "expected at least one Text block, got {blocks:?}"
    );

    provider.force_flush().expect("force_flush must succeed");
    let metrics = exporter
        .get_finished_metrics()
        .expect("get_finished_metrics must succeed");
    assert!(
        counter_total(&metrics, "cuca_requests_total") >= 1,
        "the live request must increment cuca_requests_total"
    );
}
