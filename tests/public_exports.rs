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

#[cfg(feature = "plugin-prompt-cache")]
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

#[cfg(feature = "plugin-memory")]
mod memory_surface {
    use cuca::{
        GraphImportReport, GraphNode, GraphRelationship, GraphSnapshot, MemoryConfig, MemoryGraph,
        MemoryPlugin,
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
}

#[cfg(any(feature = "plugin-memory", feature = "plugin-prompt-cache"))]
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
    #[cfg(all(feature = "plugin-memory", feature = "plugin-prompt-cache"))]
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
