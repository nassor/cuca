//! OpenTelemetry metrics and structured-log plugin.
//!
//! [`OpenTelemetryPlugin`] wires the [`CucaPlugin`] hooks to three OTel
//! instruments on a caller-supplied meter provider: a request counter, a
//! latency histogram, and a streamed-token counter. It emits `tracing` logs
//! on request dispatch and response completion. No default meter provider is
//! installed here: callers pass their own `&MeterProvider`, so the plugin
//! composes with any exporter pipeline.

use crate::error::PluginError;
use crate::plugin::CucaPlugin;
use crate::request::{UnifiedRequest, UnifiedResponse};
use crate::types::MessageContentBlock;

/// OTel-instrumented plugin recording request/token counters and a latency
/// histogram, plus structured `tracing` logs on dispatch and completion.
///
/// `Send + Sync` via the [`CucaPlugin`] supertrait, so instances can be shared
/// as `Arc<dyn CucaPlugin>` across `await` points in the client pipeline.
#[cfg(feature = "plugin-telemetry")]
pub struct OpenTelemetryPlugin {
    /// Meter the three instruments below were built from.
    ///
    /// Retained so the plugin owns the meter handle for its whole lifetime;
    /// the hooks only touch the instruments.
    #[expect(
        dead_code,
        reason = "retained handle; the hooks read the instruments, never the meter"
    )]
    meter: opentelemetry::metrics::Meter,
    /// Monotonic counter of dispatched unified LLM requests.
    request_counter: opentelemetry::metrics::Counter<u64>,
    /// Latency distribution of LLM execution turns.
    latency_histogram: opentelemetry::metrics::Histogram<f64>,
    /// Monotonic counter of streamed tokens processed across sessions.
    token_counter: opentelemetry::metrics::Counter<u64>,
}

impl OpenTelemetryPlugin {
    /// Create the plugin from a caller-supplied meter provider.
    ///
    /// Instruments are created once here under the `"cuca_client"` meter; a
    /// no-op provider yields a plugin that only emits the `tracing` logs.
    pub fn new(meter_provider: &dyn opentelemetry::metrics::MeterProvider) -> Self {
        let meter = meter_provider.meter("cuca_client");
        let request_counter = meter
            .u64_counter("cuca_requests_total")
            .with_description("Total LLM requests dispatched by CUCA")
            .build();
        let latency_histogram = meter
            .f64_histogram("cuca_request_duration_seconds")
            .with_description("Latency distribution of LLM execution turns")
            .build();
        let token_counter = meter
            .u64_counter("cuca_streamed_tokens_total")
            .with_description("Total streaming tokens processed across sessions")
            .build();
        Self {
            meter,
            request_counter,
            latency_histogram,
            token_counter,
        }
    }
}

impl CucaPlugin for OpenTelemetryPlugin {
    fn name(&self) -> &'static str {
        "opentelemetry-observability"
    }

    fn on_request(&self, req: &mut UnifiedRequest) -> Result<(), PluginError> {
        self.request_counter.add(
            1,
            &[
                opentelemetry::KeyValue::new("model", req.model.clone()),
                opentelemetry::KeyValue::new("provider", format!("{:?}", req.provider)),
            ],
        );
        tracing::info!(
            target: "cuca::telemetry",
            model = %req.model,
            provider = ?req.provider,
            "Dispatched unified LLM request"
        );
        Ok(())
    }

    fn on_stream_chunk(&self, _chunk: &mut MessageContentBlock) -> Result<(), PluginError> {
        // One token per streamed content block is a coarse approximation of
        // the true token count: a block may span many tokens, or a partial one
        // mid-stream.
        self.token_counter.add(1, &[]);
        Ok(())
    }

    fn on_response_complete(&self, res: &UnifiedResponse) -> Result<(), PluginError> {
        self.latency_histogram.record(
            res.duration_secs,
            &[opentelemetry::KeyValue::new("status", "success")],
        );
        tracing::info!(
            target: "cuca::telemetry",
            duration = res.duration_secs,
            prompt_tokens = res.prompt_tokens,
            completion_tokens = res.completion_tokens,
            "Completed LLM response turn"
        );
        Ok(())
    }
}

#[cfg(all(test, feature = "plugin-telemetry"))]
mod tests {
    // In-memory reading recipe (opentelemetry_sdk 0.32): the in-memory exporter
    // lives at `metrics::InMemoryMetricExporter` (behind the `testing`
    // feature), and the plain `PeriodicReader::builder(exporter)` runs its
    // export loop on a dedicated OS thread, no tokio runtime argument. We
    // record, `force_flush()`, and read `exporter.get_finished_metrics()`.
    // Default Cumulative temporality means the single flushed batch carries
    // running totals.

    use opentelemetry::Value;
    use opentelemetry_sdk::metrics::data::{
        AggregatedMetrics, Metric, MetricData, ResourceMetrics,
    };
    use opentelemetry_sdk::metrics::{InMemoryMetricExporter, PeriodicReader, SdkMeterProvider};

    use crate::plugin::CucaPlugin;
    use crate::request::{UnifiedRequest, UnifiedResponse};
    use crate::types::{MessageContentBlock, ProviderEndpoint};

    use super::OpenTelemetryPlugin;

    /// Build a meter provider whose exports land in an in-memory exporter,
    /// returning both so tests can flush and inspect the exported batches.
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

    // Current-thread flavor: the SDK 0.32 `PeriodicReader` exports on its own
    // OS thread, so `force_flush` cannot deadlock a single-thread tokio
    // runtime (this crate never enables `rt-multi-thread`).
    #[tokio::test]
    async fn on_request_increments_request_counter_with_model_and_provider() {
        let (provider, exporter) = provider_with_exporter();
        let plugin = OpenTelemetryPlugin::new(&provider);

        let mut req = UnifiedRequest::new("gpt-4o");
        req.provider = ProviderEndpoint::OpenAi;
        plugin
            .on_request(&mut req)
            .expect("on_request must return Ok(())");

        provider.force_flush().expect("force_flush must succeed");
        let metrics = exporter
            .get_finished_metrics()
            .expect("get_finished_metrics must succeed");

        let AggregatedMetrics::U64(MetricData::Sum(sum)) =
            exported_metric(&metrics, "cuca_requests_total").data()
        else {
            panic!("counter must export a Sum<u64>");
        };
        let dps: Vec<_> = sum.data_points().collect();
        assert_eq!(dps.len(), 1);
        let dp = dps[0];
        assert_eq!(dp.value(), 1);
        assert!(
            dp.attributes()
                .any(|kv| kv.key.as_str() == "model" && kv.value == Value::from("gpt-4o"))
        );
        assert!(
            dp.attributes()
                .any(|kv| kv.key.as_str() == "provider" && kv.value == Value::from("OpenAi"))
        );
    }

    #[tokio::test]
    async fn on_stream_chunk_increments_token_counter_per_call() {
        let (provider, exporter) = provider_with_exporter();
        let plugin = OpenTelemetryPlugin::new(&provider);

        let mut block = MessageContentBlock::Text("hello".into());
        plugin
            .on_stream_chunk(&mut block)
            .expect("on_stream_chunk must return Ok(())");
        plugin
            .on_stream_chunk(&mut block)
            .expect("on_stream_chunk must return Ok(())");

        provider.force_flush().expect("force_flush must succeed");
        let metrics = exporter
            .get_finished_metrics()
            .expect("get_finished_metrics must succeed");

        let AggregatedMetrics::U64(MetricData::Sum(sum)) =
            exported_metric(&metrics, "cuca_streamed_tokens_total").data()
        else {
            panic!("counter must export a Sum<u64>");
        };
        let dps: Vec<_> = sum.data_points().collect();
        assert_eq!(dps.len(), 1);
        assert_eq!(dps[0].value(), 2);
    }

    #[tokio::test]
    async fn on_response_complete_records_duration_histogram_with_status() {
        let (provider, exporter) = provider_with_exporter();
        let plugin = OpenTelemetryPlugin::new(&provider);

        let res = UnifiedResponse {
            model: "gpt-4o".into(),
            provider: ProviderEndpoint::OpenAi,
            duration_secs: 1.25,
            prompt_tokens: 12,
            completion_tokens: 34,
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

        let AggregatedMetrics::F64(MetricData::Histogram(hist)) =
            exported_metric(&metrics, "cuca_request_duration_seconds").data()
        else {
            panic!("histogram must export a Histogram<f64>");
        };
        let dps: Vec<_> = hist.data_points().collect();
        assert_eq!(dps.len(), 1);
        let dp = dps[0];
        assert_eq!(dp.count(), 1);
        assert_eq!(dp.sum(), res.duration_secs);
        assert!(
            dp.attributes()
                .any(|kv| kv.key.as_str() == "status" && kv.value == Value::from("success"))
        );
    }

    #[test]
    fn name_is_stable_and_plugin_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<OpenTelemetryPlugin>();

        let provider = SdkMeterProvider::builder().build();
        let plugin = OpenTelemetryPlugin::new(&provider);
        assert_eq!(plugin.name(), "opentelemetry-observability");
    }
}
