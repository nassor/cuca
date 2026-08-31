//! The cost ledger to OpenTelemetry bridge.
//!
//! [`OtelCostObserver`] implements
//! [`CostObserver`](crate::plugins::cost::CostObserver) by recording each
//! [`CostUsage`](crate::plugins::cost::CostUsage) reading to gauges on a
//! caller-supplied meter provider. Put one in
//! [`CostConfig::observers`](crate::plugins::cost::CostConfig::observers) and
//! the ledger reaches the same pipeline
//! [`OpenTelemetryPlugin`](crate::plugins::telemetry::OpenTelemetryPlugin)
//! feeds, under the same `"cuca_client"` meter, without the caller writing an
//! observer. No global meter provider is installed here either.
//!
//! # Why the bridge lives in core
//!
//! It joins two plugins, so it belongs to neither. A `use` of the telemetry
//! plugin inside `src/plugins/cost.rs`, or the reverse, is an undeclared
//! plugin-to-plugin code edge, and a runtime peer lookup is forbidden outright.
//! Core may `#[cfg]`-reference any plugin, so the bridge sits here under
//! `cfg(all(feature = "plugin-cost", feature = "plugin-telemetry"))`, beside
//! `CucaExport::from_live` in `src/export.rs`, the crate's other
//! two-plugin workflow. Either feature alone compiles the module and its root
//! re-export out of existence.
//!
//! # Why gauges and not counters
//!
//! `CostUsage` is a cumulative snapshot: every field is a running total since
//! construction or the last `CostPlugin::reset`. An OTel `Counter` takes a
//! delta, and the seam hands out no deltas. Deriving one would mean holding
//! the previous reading and subtracting, which has no correct answer here: the
//! ledger is observed twice per turn (the `on_request` charge, then the
//! `on_response_complete` commit), an observer error rolls a charge back, and
//! `reset` zeroes the totals. A synchronous `Gauge` records a last value,
//! which is exactly what a reading is. An observable gauge is not available
//! either: its callback would have to hold the plugin, and the plugin owns its
//! observers.
//!
//! One gauge per field, rather than one gauge with a discriminating attribute:
//! every `record` call then passes an empty attribute slice, so no attribute
//! set is built, allocated, or hashed per reading. `near_cap` records as `1`
//! or `0` because OTel carries no boolean measurement. The two configured caps
//! in the reading are not recorded: they are construction-time constants, not
//! traffic.
//!
//! # Cost of one reading
//!
//! Ten `Gauge::record` calls with an empty attribute slice, and nothing else.
//! The observer holds no state and takes no lock of its own; the only
//! synchronization is whatever the SDK's instrument does internally. Recording
//! is infallible, so [`OtelCostObserver::observe`] always returns `Ok(())`: an
//! observer error aborts the turn it was handed, and this one never produces
//! any.

use opentelemetry::metrics::{Gauge, MeterProvider};

use crate::error::PluginError;
use crate::plugins::cost::{CostObserver, CostUsage};

/// Records every cost-ledger reading to OpenTelemetry gauges.
///
/// Built once from the caller's meter provider and attached to
/// [`CostConfig::observers`](crate::plugins::cost::CostConfig::observers).
/// The instruments, all `Gauge<u64>` under the meter named `"cuca_client"`:
///
/// | Instrument | [`CostUsage`] field |
/// |---|---|
/// | `cuca_cost_spent_micros` | `spent_micros` |
/// | `cuca_cost_prompt_tokens` | `prompt_tokens` |
/// | `cuca_cost_completion_tokens` | `completion_tokens` |
/// | `cuca_cost_cache_read_tokens` | `cache_read_tokens` |
/// | `cuca_cost_cache_write_tokens` | `cache_write_tokens` |
/// | `cuca_cost_turns` | `turns` |
/// | `cuca_cost_unpriced_turns` | `unpriced_turns` |
/// | `cuca_cost_overflow_turns` | `overflow_turns` |
/// | `cuca_cost_untokenized_image_blocks` | `untokenized_image_blocks` |
/// | `cuca_cost_near_cap` | `near_cap`, as `1` or `0` |
///
/// Each is recorded with no attributes. A no-op meter provider yields an
/// observer that records nowhere.
#[derive(Debug)]
pub struct OtelCostObserver {
    spent_micros: Gauge<u64>,
    prompt_tokens: Gauge<u64>,
    completion_tokens: Gauge<u64>,
    cache_read_tokens: Gauge<u64>,
    cache_write_tokens: Gauge<u64>,
    turns: Gauge<u64>,
    unpriced_turns: Gauge<u64>,
    overflow_turns: Gauge<u64>,
    untokenized_image_blocks: Gauge<u64>,
    near_cap: Gauge<u64>,
}

impl OtelCostObserver {
    /// Create the observer from a caller-supplied meter provider.
    ///
    /// The ten instruments are created once here, under the `"cuca_client"`
    /// meter that [`OpenTelemetryPlugin`](crate::plugins::telemetry::OpenTelemetryPlugin)
    /// also uses, so both land in one instrumentation scope.
    pub fn new(meter_provider: &dyn MeterProvider) -> Self {
        let meter = meter_provider.meter("cuca_client");
        Self {
            spent_micros: meter
                .u64_gauge("cuca_cost_spent_micros")
                .with_description(
                    "Cumulative spend in micro-units of the caller's currency, as of the last \
                     ledger reading",
                )
                .build(),
            prompt_tokens: meter
                .u64_gauge("cuca_cost_prompt_tokens")
                .with_description("Cumulative estimated prompt tokens charged to the ledger")
                .build(),
            completion_tokens: meter
                .u64_gauge("cuca_cost_completion_tokens")
                .with_description("Cumulative estimated completion tokens charged to the ledger")
                .build(),
            cache_read_tokens: meter
                .u64_gauge("cuca_cost_cache_read_tokens")
                .with_description(
                    "Cumulative provider-reported prompt tokens served from the provider cache",
                )
                .build(),
            cache_write_tokens: meter
                .u64_gauge("cuca_cost_cache_write_tokens")
                .with_description(
                    "Cumulative provider-reported prompt tokens written to the provider cache",
                )
                .build(),
            turns: meter
                .u64_gauge("cuca_cost_turns")
                .with_description("Turns committed to the ledger")
                .build(),
            unpriced_turns: meter
                .u64_gauge("cuca_cost_unpriced_turns")
                .with_description("Turns charged in tokens only, because the model had no rates")
                .build(),
            overflow_turns: meter
                .u64_gauge("cuca_cost_overflow_turns")
                .with_description(
                    "Turns folded into the reserved overflow bucket at max_tracked_models",
                )
                .build(),
            untokenized_image_blocks: meter
                .u64_gauge("cuca_cost_untokenized_image_blocks")
                .with_description("Prompt image blocks the token estimator skipped")
                .build(),
            near_cap: meter
                .u64_gauge("cuca_cost_near_cap")
                .with_description(
                    "1 when the reading meets the configured warn_fraction of the tightest cap, \
                     0 otherwise",
                )
                .build(),
        }
    }
}

impl CostObserver for OtelCostObserver {
    /// Record the reading to the ten gauges.
    ///
    /// # Errors
    ///
    /// Never. OTel recording is infallible, so this is always `Ok(())` and no
    /// reading can abort the turn it belongs to.
    fn observe(&self, usage: &CostUsage) -> Result<(), PluginError> {
        self.spent_micros.record(usage.spent_micros, &[]);
        self.prompt_tokens.record(usage.prompt_tokens, &[]);
        self.completion_tokens.record(usage.completion_tokens, &[]);
        self.cache_read_tokens.record(usage.cache_read_tokens, &[]);
        self.cache_write_tokens
            .record(usage.cache_write_tokens, &[]);
        self.turns.record(usage.turns, &[]);
        self.unpriced_turns.record(usage.unpriced_turns, &[]);
        self.overflow_turns.record(usage.overflow_turns, &[]);
        self.untokenized_image_blocks
            .record(usage.untokenized_image_blocks, &[]);
        self.near_cap.record(u64::from(usage.near_cap), &[]);
        Ok(())
    }
}

#[cfg(all(test, feature = "plugin-cost", feature = "plugin-telemetry"))]
mod tests {
    // Same in-memory reading recipe the telemetry plugin's tests use
    // (opentelemetry_sdk 0.32): `PeriodicReader` exports on its own OS thread,
    // so `force_flush` needs no runtime, and the default Cumulative
    // temporality means one flushed batch carries the last gauge value.

    use std::sync::Arc;

    use opentelemetry_sdk::metrics::data::{
        AggregatedMetrics, Metric, MetricData, ResourceMetrics,
    };
    use opentelemetry_sdk::metrics::{InMemoryMetricExporter, PeriodicReader, SdkMeterProvider};

    use super::OtelCostObserver;
    use crate::plugin::CucaPlugin;
    use crate::plugins::cost::{
        CostConfig, CostPlugin, ModelRates, PricingTable, UnpricedModelPolicy,
    };
    use crate::request::{UnifiedRequest, UnifiedResponse};
    use crate::types::{MessageContentBlock, ProviderEndpoint};

    const MODEL: &str = "bridge-model";

    /// A meter provider whose exports land in an in-memory exporter, returning
    /// both so a test can record, flush, and read the batch back.
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

    /// The single data point of the `u64` gauge named `name`.
    fn gauge_value(metrics: &[ResourceMetrics], name: &str) -> u64 {
        let AggregatedMetrics::U64(MetricData::Gauge(gauge)) =
            exported_metric(metrics, name).data()
        else {
            panic!("`{name}` must export a Gauge<u64>");
        };
        let points: Vec<_> = gauge.data_points().collect();
        assert_eq!(
            points.len(),
            1,
            "`{name}` must carry one unattributed point"
        );
        assert_eq!(
            points[0].attributes().count(),
            0,
            "`{name}` must be recorded with no attributes"
        );
        points[0].value()
    }

    fn plugin_with_bridge(provider: &SdkMeterProvider, config: CostConfig) -> CostPlugin {
        CostPlugin::new(CostConfig {
            observers: vec![Arc::new(OtelCostObserver::new(provider))],
            ..config
        })
        .expect("cost plugin must build")
    }

    fn request() -> UnifiedRequest {
        UnifiedRequest::new(MODEL)
            .add_system_message("You are concise.")
            .add_user_message("hello there")
    }

    fn response(text: &str) -> UnifiedResponse {
        UnifiedResponse {
            model: MODEL.to_string(),
            provider: ProviderEndpoint::LlamaCpp,
            duration_secs: 0.5,
            prompt_tokens: 0,
            completion_tokens: 1,
            finish_reason: Some("stop".to_string()),
            content: vec![MessageContentBlock::Text(text.to_string())],
            prompt_cache_usage: None,
        }
    }

    fn priced() -> CostConfig {
        CostConfig {
            pricing: PricingTable::new().with_model(
                MODEL,
                ModelRates {
                    input_micros_per_mtok: 3_000_000,
                    output_micros_per_mtok: 15_000_000,
                    ..Default::default()
                },
            ),
            ..Default::default()
        }
    }

    #[test]
    fn a_committed_turn_exports_the_ledger_numbers() {
        let (provider, exporter) = provider_with_exporter();
        let plugin = plugin_with_bridge(&provider, priced());

        plugin
            .on_request(&mut request())
            .expect("on_request must charge");
        plugin
            .on_response_complete(&response("a reply with several words in it"))
            .expect("on_response_complete must commit");

        provider.force_flush().expect("force_flush must succeed");
        let metrics = exporter
            .get_finished_metrics()
            .expect("get_finished_metrics must succeed");
        let usage = plugin.usage().expect("ledger lock must not be poisoned");

        assert!(usage.prompt_tokens > 0 && usage.completion_tokens > 0);
        assert!(usage.spent_micros > 0);
        assert_eq!(
            gauge_value(&metrics, "cuca_cost_prompt_tokens"),
            usage.prompt_tokens
        );
        assert_eq!(
            gauge_value(&metrics, "cuca_cost_completion_tokens"),
            usage.completion_tokens
        );
        assert_eq!(
            gauge_value(&metrics, "cuca_cost_spent_micros"),
            usage.spent_micros
        );
        assert_eq!(gauge_value(&metrics, "cuca_cost_turns"), 1);
        assert_eq!(gauge_value(&metrics, "cuca_cost_cache_read_tokens"), 0);
        assert_eq!(gauge_value(&metrics, "cuca_cost_cache_write_tokens"), 0);
        assert_eq!(gauge_value(&metrics, "cuca_cost_unpriced_turns"), 0);
        assert_eq!(gauge_value(&metrics, "cuca_cost_overflow_turns"), 0);
        assert_eq!(
            gauge_value(&metrics, "cuca_cost_untokenized_image_blocks"),
            0
        );
        assert_eq!(gauge_value(&metrics, "cuca_cost_near_cap"), 0);
    }

    /// The gauge carries the latest reading, not a sum of the readings: two
    /// turns leave the ledger's own total, which is what a snapshot means.
    #[test]
    fn the_gauge_holds_the_last_reading_rather_than_a_sum() {
        let (provider, exporter) = provider_with_exporter();
        let plugin = plugin_with_bridge(&provider, priced());

        plugin.on_request(&mut request()).expect("first charge");
        plugin
            .on_response_complete(&response("first"))
            .expect("first commit");
        let after_one = plugin.usage().expect("ledger read").prompt_tokens;

        plugin.on_request(&mut request()).expect("second charge");
        plugin
            .on_response_complete(&response("second"))
            .expect("second commit");

        provider.force_flush().expect("force_flush must succeed");
        let metrics = exporter
            .get_finished_metrics()
            .expect("get_finished_metrics must succeed");

        assert_eq!(
            gauge_value(&metrics, "cuca_cost_prompt_tokens"),
            after_one * 2,
            "the ledger total after two identical turns"
        );
        assert_eq!(gauge_value(&metrics, "cuca_cost_turns"), 2);
    }

    /// The counters that only a degraded turn moves: an unpriced model and an
    /// image block the estimator skips.
    #[test]
    fn unpriced_turns_and_skipped_image_blocks_reach_their_gauges() {
        let (provider, exporter) = provider_with_exporter();
        let plugin = plugin_with_bridge(
            &provider,
            CostConfig {
                on_unpriced_model: UnpricedModelPolicy::CountTokensOnly,
                ..Default::default()
            },
        );

        let mut req = request();
        req.messages[1]
            .content
            .push(MessageContentBlock::ImageBase64 {
                media_type: "image/png".to_string(),
                data: "AAAA".to_string(),
            });
        plugin.on_request(&mut req).expect("on_request must charge");
        plugin
            .on_response_complete(&response("reply"))
            .expect("on_response_complete must commit");

        provider.force_flush().expect("force_flush must succeed");
        let metrics = exporter
            .get_finished_metrics()
            .expect("get_finished_metrics must succeed");

        assert_eq!(gauge_value(&metrics, "cuca_cost_spent_micros"), 0);
        assert_eq!(gauge_value(&metrics, "cuca_cost_unpriced_turns"), 1);
        assert_eq!(
            gauge_value(&metrics, "cuca_cost_untokenized_image_blocks"),
            1
        );
    }

    /// `near_cap` is the only boolean in the reading; it records as `1`.
    #[test]
    fn near_cap_records_as_one_once_the_warn_fraction_is_met() {
        let (provider, exporter) = provider_with_exporter();
        let plugin = plugin_with_bridge(
            &provider,
            CostConfig {
                max_total_tokens: Some(1_000),
                warn_fraction: Some(0.001),
                ..priced()
            },
        );

        plugin
            .on_request(&mut request())
            .expect("on_request must charge without crossing the cap");

        provider.force_flush().expect("force_flush must succeed");
        let metrics = exporter
            .get_finished_metrics()
            .expect("get_finished_metrics must succeed");

        assert!(
            plugin
                .usage()
                .expect("ledger lock must not be poisoned")
                .near_cap
        );
        assert_eq!(gauge_value(&metrics, "cuca_cost_near_cap"), 1);
    }
}
