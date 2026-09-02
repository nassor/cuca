//! Public-API surface tests: every type a caller needs must be reachable from
//! the `cuca` crate root under its own feature gate, never through a
//! private or nested module path.
//!
//! The imports below are the contract. Each test only constructs values, so a
//! rename or a missing/incorrectly gated re-export fails to compile.
#![cfg(any(feature = "provider-openai", feature = "provider-llamacpp"))]

use cuca::{PromptCacheBreakpoint, PromptCacheDirective, PromptCacheUsage, UnifiedRequest};

#[test]
fn request_prompt_cache_types_are_root_exported_unconditionally() {
    let request = UnifiedRequest::new("gpt-4o")
        .add_user_message("hi")
        .with_prompt_cache(PromptCacheDirective::Ephemeral {
            breakpoints: vec![PromptCacheBreakpoint {
                message_index: 0,
                block_index: 0,
            }],
        });

    assert!(matches!(
        request.prompt_cache,
        PromptCacheDirective::Ephemeral { .. }
    ));
    assert_eq!(
        UnifiedRequest::new("gpt-4o").prompt_cache,
        PromptCacheDirective::Disabled
    );

    let usage = PromptCacheUsage {
        read_tokens: 4,
        write_tokens: 2,
    };
    assert_eq!(usage.read_tokens + usage.write_tokens, 6);
}

#[cfg(feature = "service-prompt-cache")]
mod prompt_cache_surface {
    use std::sync::Arc;
    use std::time::Duration;

    use cuca::{
        CucaClient, PromptCache, PromptCacheConfig, PromptCacheEntry, PromptCacheError,
        PromptCacheImportReport, PromptCacheSnapshot,
    };

    #[test]
    fn cache_config_and_service_are_root_exported() {
        let config = PromptCacheConfig::new(4, Duration::from_secs(60))
            .expect("a valid configuration must build");
        assert_eq!(config.capacity, 4);

        let cache = PromptCache::new(config).expect("cache must build");
        let snapshot: PromptCacheSnapshot = cache.snapshot().expect("snapshot must succeed");
        assert!(snapshot.entries.is_empty());

        let report: PromptCacheImportReport = cache
            .replace_snapshot(PromptCacheSnapshot::default())
            .expect("an empty import must succeed");
        assert_eq!(report.imported_entries, 0);
        assert_eq!(report.expired_entries, 0);
        assert_eq!(report.capacity_evictions, 0);
    }

    #[test]
    fn cache_errors_are_structured_and_root_exported() {
        let err = PromptCacheConfig::new(0, Duration::from_secs(60))
            .expect_err("zero capacity is invalid");
        assert!(matches!(err, PromptCacheError::Config(_)));
        assert!(!err.to_string().is_empty());
    }

    /// The cache entry DTO is nameable by callers building or inspecting a
    /// snapshot.
    #[test]
    fn cache_entry_dto_is_root_exported() {
        fn entry_keys(snapshot: &PromptCacheSnapshot) -> Vec<&str> {
            snapshot
                .entries
                .iter()
                .map(|entry: &PromptCacheEntry| entry.key.as_str())
                .collect()
        }
        assert!(entry_keys(&PromptCacheSnapshot::default()).is_empty());
    }

    /// The client seam is reachable without naming a private module.
    #[test]
    fn client_cache_seam_is_root_exported() {
        let config = PromptCacheConfig::new(2, Duration::from_secs(30)).unwrap();
        let client = CucaClient::builder()
            .with_provider(cuca::types::ProviderEndpoint::OpenAi)
            .with_api_key("sk-test")
            .with_prompt_cache_config(config)
            .build()
            .expect("client must build");
        assert!(client.prompt_cache().is_some());
        assert!(
            client
                .prompt_cache_snapshot()
                .expect("snapshot must succeed")
                .entries
                .is_empty()
        );

        let shared: Arc<PromptCache> = client.prompt_cache().unwrap();
        let service_client = CucaClient::builder()
            .with_provider(cuca::types::ProviderEndpoint::OpenAi)
            .with_api_key("sk-test")
            .with_prompt_cache_service(shared)
            .build()
            .expect("client must build");
        let report = service_client
            .replace_prompt_cache_snapshot(PromptCacheSnapshot::default())
            .expect("an empty import must succeed");
        assert_eq!(report.imported_entries, 0);
    }
}

#[cfg(feature = "service-rate-limit")]
mod rate_limit_surface {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    use cuca::{
        PluginError, RateLimitConfig, RateLimitError, RateLimitObserver, RateLimitPermit,
        RateLimitUsage, RateLimiter,
    };

    /// A caller-supplied observer, named through the root re-export only.
    #[derive(Default)]
    struct CountingObserver {
        readings: AtomicUsize,
    }

    impl RateLimitObserver for CountingObserver {
        fn observe(&self, usage: &RateLimitUsage) -> Result<(), PluginError> {
            assert!(usage.waiting >= 1);
            self.readings.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }

    #[test]
    fn limiter_config_permit_and_usage_are_root_exported() {
        let observer = Arc::new(CountingObserver::default());
        let config = RateLimitConfig::new(4, Duration::from_secs(1), 2, 8)
            .expect("a valid configuration must build")
            .with_burst(6)
            .expect("a non-zero burst must be accepted")
            .with_warn_fraction(0.75)
            .expect("a warn fraction inside (0.0, 1.0] must be accepted")
            .with_observer(Arc::clone(&observer) as Arc<dyn RateLimitObserver>);
        assert_eq!(config.burst, 6);

        let limiter = RateLimiter::new(config).expect("limiter must build");
        assert_eq!(limiter.config().max_concurrent, 2);

        let permit: RateLimitPermit = limiter.try_acquire().expect("a full bucket admits");
        let usage: RateLimitUsage = limiter.usage().expect("usage must read");
        assert_eq!(usage.available_tokens, 5);
        assert_eq!(usage.in_flight, 1);
        assert_eq!(usage.waiting, 0);
        drop(permit);
        assert_eq!(limiter.usage().expect("usage must read").in_flight, 0);
    }

    #[test]
    fn limiter_errors_are_structured_and_root_exported() {
        let err = RateLimitConfig::new(0, Duration::from_secs(1), 1, 1)
            .expect_err("a zero rate is invalid");
        assert!(matches!(err, RateLimitError::Config(_)));
        assert!(!err.to_string().is_empty());
        assert!(matches!(
            PluginError::from(err),
            PluginError::Validation { .. }
        ));
    }
}

#[cfg(feature = "plugin-redaction")]
mod redaction_surface {
    use std::borrow::Cow;
    use std::sync::Arc;

    use cuca::plugin::CucaPlugin;
    use cuca::{Redacted, RedactionConfig, RedactionPlugin, RedactionRule};

    #[test]
    fn redaction_policy_plugin_and_result_are_root_exported() {
        let config = RedactionConfig::new(vec![
            RedactionRule::Literal {
                kind: "api-key".to_string(),
                value: "sk-live-4242".to_string(),
            },
            RedactionRule::Prefixed {
                kind: "token".to_string(),
                prefix: "sk-".to_string(),
                min_len: 8,
                max_len: 64,
            },
            RedactionRule::EmailLike {
                kind: "email".to_string(),
            },
            RedactionRule::DigitRun {
                kind: "card".to_string(),
                min_digits: 13,
                max_digits: 19,
            },
        ])
        .expect("a valid policy must build");
        assert_eq!(config.max_matches_per_text, 1024);
        assert!(config.scrub_tool_definitions);

        let plugin = RedactionPlugin::new(config).expect("plugin must build");
        assert_eq!(plugin.rule_count(), 4);
        assert_eq!(plugin.match_cap(), 1024);

        let clean: Redacted<'_> = plugin
            .scrub_str("nothing here")
            .expect("scrub must succeed");
        assert!(matches!(clean.text, Cow::Borrowed(_)));
        assert_eq!(clean.count, 0);

        let dirty = plugin
            .scrub_str("key sk-live-4242 here")
            .expect("scrub must succeed");
        assert_eq!(dirty.text, "key [REDACTED:api-key] here");
        assert_eq!(dirty.count, 1);

        let shared: Arc<dyn CucaPlugin> = Arc::new(plugin);
        assert_eq!(shared.name(), "redaction");
    }

    #[test]
    fn redaction_config_rejection_is_root_exported() {
        let err = RedactionConfig::new(Vec::new()).expect_err("an empty policy is invalid");
        assert!(matches!(err, cuca::PluginError::Validation { .. }));
        assert!(!err.to_string().is_empty());
    }
}

#[cfg(feature = "plugin-memory")]
mod memory_surface {
    use cuca::{
        GRAPH_RENDER_MARKER, GraphImportReport, GraphNode, GraphRelationship, GraphSnapshot,
        MemoryConfig, MemoryGraph, MemoryPlugin,
    };

    #[test]
    fn graph_snapshot_dtos_are_root_exported() {
        let snapshot = GraphSnapshot {
            nodes: vec![GraphNode {
                id: "a".into(),
                labels: vec!["person".into()],
                properties: serde_json::Map::new(),
            }],
            relationships: vec![GraphRelationship {
                id: "r".into(),
                from: "a".into(),
                to: "a".into(),
                kind: "self".into(),
                weight: 1.0,
                properties: serde_json::Map::new(),
            }],
        };

        let graph = MemoryGraph::from_snapshot(snapshot.clone()).expect("valid snapshot");
        assert_eq!(graph.snapshot(), snapshot);

        let plugin = MemoryPlugin::new(MemoryConfig::default()).expect("plugin must build");
        let report: GraphImportReport = plugin
            .replace_snapshot(snapshot.clone())
            .expect("import must succeed");
        assert_eq!(report.nodes, 1);
        assert_eq!(report.relationships, 1);
        assert_eq!(plugin.snapshot().expect("snapshot must succeed"), snapshot);
    }

    #[test]
    fn graph_render_marker_is_root_exported() {
        let mut graph = MemoryGraph::new();
        graph.upsert_node(GraphNode {
            id: "a".into(),
            labels: Vec::new(),
            properties: serde_json::Map::new(),
        });

        assert!(graph.render(8, 8).starts_with(GRAPH_RENDER_MARKER));
    }
}

#[cfg(feature = "plugin-cost")]
mod cost_surface {
    use std::sync::Arc;

    use cuca::{
        CostConfig, CostEntry, CostObserver, CostPlugin, CostUsage, ModelRates, PluginError,
        PricingResolver, PricingTable, UnpricedModelPolicy,
    };

    /// A caller-side live rate source.
    struct FixedRates;

    impl PricingResolver for FixedRates {
        fn resolve_rates(&self, _model: &str) -> Option<ModelRates> {
            Some(ModelRates {
                input_micros_per_mtok: 1,
                output_micros_per_mtok: 2,
                cache_read_micros_per_mtok: 3,
                cache_write_micros_per_mtok: 4,
            })
        }
    }

    /// A caller-side reporting gauge.
    struct Recorder;

    impl CostObserver for Recorder {
        fn observe(&self, _usage: &CostUsage) -> Result<(), PluginError> {
            Ok(())
        }
    }

    #[test]
    fn cost_config_plugin_and_seams_are_root_exported() {
        let pricing = PricingTable::new().with_model(
            "gpt-4o",
            ModelRates {
                input_micros_per_mtok: 2_500_000,
                output_micros_per_mtok: 10_000_000,
                cache_read_micros_per_mtok: 1_250_000,
                cache_write_micros_per_mtok: 0,
            },
        );
        assert_eq!(pricing.len(), 1);
        assert!(!pricing.is_empty());

        let plugin = CostPlugin::new(CostConfig {
            pricing,
            pricing_resolver: Some(Arc::new(FixedRates)),
            max_total_tokens: Some(1_000_000),
            max_total_micros: Some(50_000_000),
            warn_fraction: Some(0.8),
            observers: vec![Arc::new(Recorder)],
            ..Default::default()
        })
        .expect("plugin must build");

        let usage: CostUsage = plugin.usage().expect("usage read must succeed");
        assert_eq!(usage.total_tokens(), 0);
        assert_eq!(usage.max_total_tokens, Some(1_000_000));

        let breakdown: Vec<(String, CostEntry)> =
            plugin.breakdown().expect("breakdown read must succeed");
        assert!(breakdown.is_empty());

        assert!(matches!(
            UnpricedModelPolicy::Reject,
            UnpricedModelPolicy::Reject
        ));
    }
}

#[cfg(all(feature = "plugin-cost", feature = "plugin-telemetry"))]
mod cost_otel_bridge_surface {
    use std::sync::Arc;

    use cuca::{CostConfig, CostObserver, CostPlugin, CostUsage, OtelCostObserver};
    use opentelemetry_sdk::metrics::SdkMeterProvider;

    /// The bridge is reachable from the crate root, builds from a plain
    /// `&dyn MeterProvider`, and drops straight into `CostConfig::observers`.
    #[test]
    fn otel_cost_observer_is_root_exported_under_both_features() {
        let provider = SdkMeterProvider::builder().build();
        let observer = OtelCostObserver::new(&provider);

        let usage: CostUsage = CostPlugin::new(CostConfig::default())
            .expect("plugin must build")
            .usage()
            .expect("usage read must succeed");
        observer.observe(&usage).expect("recording is infallible");

        let plugin = CostPlugin::new(CostConfig {
            observers: vec![Arc::new(OtelCostObserver::new(&provider))],
            ..Default::default()
        })
        .expect("plugin must build with the bridge attached");
        assert_eq!(plugin.usage().expect("usage read must succeed").turns, 0);
    }
}

#[cfg(any(feature = "plugin-memory", feature = "service-prompt-cache"))]
mod export_surface {
    use cuca::{
        CUCA_EXPORT_FORMAT, CUCA_EXPORT_VERSION, CucaExport, CucaExportError, GraphExportSection,
        PromptCacheExportSection,
    };

    fn empty_envelope() -> CucaExport {
        CucaExport::new(
            GraphExportSection {
                nodes: Vec::new(),
                relationships: Vec::new(),
            },
            PromptCacheExportSection {
                entries: Vec::new(),
            },
        )
    }

    #[test]
    fn envelope_types_are_root_exported() {
        let export = empty_envelope();
        assert_eq!(export.format, CUCA_EXPORT_FORMAT);
        assert_eq!(export.version, CUCA_EXPORT_VERSION);

        let bytes = export.to_json_bytes().expect("encode must succeed");
        assert_eq!(
            CucaExport::from_json_slice(&bytes).expect("decode must succeed"),
            export
        );
    }

    #[test]
    fn envelope_errors_are_structured_and_root_exported() {
        let err = CucaExport::from_json_slice(b"{}").expect_err("an empty document is invalid");
        assert!(matches!(err, CucaExportError::Json { .. }));
        assert!(!err.to_string().is_empty());
    }

    /// The combined coordinator report is nameable wherever both components
    /// are compiled.
    #[cfg(all(feature = "plugin-memory", feature = "service-prompt-cache"))]
    #[test]
    fn combined_import_report_is_root_exported() {
        use std::time::Duration;

        use cuca::{CucaImportReport, MemoryConfig, MemoryPlugin, PromptCache, PromptCacheConfig};

        let memory = MemoryPlugin::new(MemoryConfig::default()).expect("plugin must build");
        let cache = PromptCache::new(
            PromptCacheConfig::new(2, Duration::from_secs(60)).expect("config must build"),
        )
        .expect("cache must build");

        let report: CucaImportReport = empty_envelope()
            .import_into(&memory, &cache, 1_000)
            .expect("an empty combined import must succeed");
        assert_eq!(
            report,
            CucaImportReport {
                graph_nodes: 0,
                graph_relationships: 0,
                imported_cache_entries: 0,
                expired_cache_entries: 0,
                capacity_evictions: 0,
            }
        );
    }
}

#[cfg(feature = "service-replay")]
mod replay_surface {
    use std::sync::Arc;

    use cuca::{
        AgentResponseStream, InMemoryBackend, ReplayCompletion, ReplayConfig, ReplayNote,
        ReplayTrajectory, ReplayTurn, ReplayUsage, SessionEvent, SessionRecord, SessionReplay,
    };

    #[test]
    fn replay_config_and_service_are_root_exported() {
        let config = ReplayConfig::new(64, 8, Some(0.5)).expect("a valid configuration must build");
        assert_eq!(config.max_records, 64);
        assert_eq!(config.max_turn_blocks, 8);

        let replay = SessionReplay::with_config(Arc::new(InMemoryBackend::new()), config)
            .expect("replay must build");
        assert_eq!(replay.config(), &config);
        assert!(
            replay
                .load("never-recorded")
                .expect("an unknown session loads as empty")
                .is_empty()
        );
    }

    /// The trajectory DTOs are nameable by callers inspecting a replay.
    #[test]
    fn replay_trajectory_types_are_root_exported() {
        let backend = InMemoryBackend::new();
        for (sequence, event) in [
            SessionEvent::Output {
                text: "hi".to_string(),
            },
            SessionEvent::ModelSwap {
                from: "fast".to_string(),
                to: "slow".to_string(),
                reason: "latency_threshold".to_string(),
            },
            SessionEvent::Latency { duration_ms: 250 },
            SessionEvent::TokenUsage {
                prompt_tokens: 4,
                completion_tokens: 2,
            },
        ]
        .into_iter()
        .enumerate()
        {
            cuca::SessionBackend::append(
                &backend,
                &SessionRecord::at("s", sequence as u64, 1, event),
            )
            .expect("append must succeed");
        }

        let trajectory: ReplayTrajectory = SessionReplay::new(Arc::new(backend))
            .load("s")
            .expect("replay must load");
        let usage: ReplayUsage = trajectory.usage();
        assert_eq!(usage.turns, 1);

        let turn: &ReplayTurn = trajectory.turn(0).expect("the single turn must be there");
        let completion: &ReplayCompletion = turn.completion().expect("the turn is complete");
        assert_eq!(completion.duration_ms, 250);

        match turn.notes() {
            [ReplayNote::ModelSwap { from, to, reason }] => {
                assert_eq!(
                    (from.as_str(), to.as_str(), reason.as_str()),
                    ("fast", "slow", "latency_threshold")
                );
            }
            other => panic!("expected exactly one ReplayNote::ModelSwap, got {other:?}"),
        }
    }

    /// A replayed stream *is* the public stream contract: the alias accepts it
    /// with no adapter, which is what makes replay a drop-in for a provider
    /// turn.
    #[test]
    fn replay_stream_is_an_agent_response_stream() {
        let backend = InMemoryBackend::new();
        cuca::SessionBackend::append(
            &backend,
            &SessionRecord::at(
                "s",
                0,
                1,
                SessionEvent::Output {
                    text: "hi".to_string(),
                },
            ),
        )
        .expect("append must succeed");

        let trajectory = SessionReplay::new(Arc::new(backend))
            .load("s")
            .expect("replay must load");
        let _stream: AgentResponseStream = trajectory
            .into_stream()
            .expect("a non-empty replay streams");
    }
}

#[cfg(feature = "service-vector-store")]
mod vector_store_surface {
    use std::sync::Arc;

    use cuca::{
        Embedder, InMemoryVectorStore, PluginError, RECALL_RENDER_MARKER, RecallInjection,
        RetrievalReport, RetrievedTurn, UnifiedRequest, VectorStoreConfig, VectorStoreError,
        VectorStoreUsage,
    };

    /// A basis embedder: `"alpha"` is e0, everything else is the zero vector.
    struct FixedEmbedder;

    impl Embedder for FixedEmbedder {
        fn embed(&self, text: &str) -> Result<Vec<f32>, PluginError> {
            let mut vector = vec![0.0f32; 8];
            if text.contains("alpha") {
                vector[0] = 1.0;
            }
            Ok(vector)
        }
    }

    /// The store, its config, its usage reading, and the recall DTOs are all
    /// nameable from the crate root, and the concrete store coerces to the
    /// `VectorStore` seam the memory plugin declares.
    #[test]
    fn vector_store_types_are_root_exported() {
        let config = VectorStoreConfig::new(4, 8, 4096)
            .expect("a valid configuration must build")
            .with_warn_fraction(0.8)
            .expect("a valid warn fraction must be accepted");
        assert_eq!(config.max_entries, 4);

        let store = Arc::new(
            InMemoryVectorStore::new(config, Arc::new(FixedEmbedder)).expect("store must build"),
        );
        let seam: Arc<dyn cuca::VectorStore> = Arc::clone(&store) as Arc<dyn cuca::VectorStore>;
        seam.store_turns(
            "surface",
            &[cuca::types::UnifiedMessage::user("alpha token")],
        )
        .expect("the seam must accept the turn");

        let usage: VectorStoreUsage = store.usage().expect("usage must read");
        assert_eq!((usage.entries, usage.capacity), (1, 4));
        assert_eq!(store.capacity(), 4);

        let report: RetrievalReport = store.retrieve("alpha", 2).expect("query must run");
        assert_eq!(report.scanned, 1);
        let hit: &RetrievedTurn = report.turns.first().expect("one hit must come back");
        assert_eq!(hit.session_hint, "surface");

        let mut request = UnifiedRequest::new("gpt-4o").add_user_message("follow-up");
        assert_eq!(report.inject(&mut request), RecallInjection::Inserted);
        assert!(
            request
                .messages
                .iter()
                .any(|m| format!("{m:?}").contains(RECALL_RENDER_MARKER))
        );
    }

    #[test]
    fn vector_store_config_errors_are_root_exported() {
        let error =
            VectorStoreConfig::new(0, 8, 4096).expect_err("zero max_entries must be rejected");
        assert!(matches!(error, VectorStoreError::Config(_)));
        assert!(!error.to_string().is_empty());
    }
}
