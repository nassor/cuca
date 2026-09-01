//! The versioned canonical `cuca-export` envelope.
//!
//! [`CucaExport`] is the single bundle DTO that carries a memory-graph
//! snapshot and a local-response-cache snapshot as one versioned document. It
//! is a pure in-memory value: [`CucaExport::to_json_bytes`] and
//! [`CucaExport::from_json_slice`] are the only serialization helpers, and
//! neither accepts a path, a file, or a writer. Persistence, transport,
//! encryption, and access control belong to the caller.
//!
//! # Wire shape
//!
//! ```json
//! {
//!   "format": "cuca-export",
//!   "version": 1,
//!   "graph": { "nodes": [], "relationships": [] },
//!   "prompt_cache": { "entries": [] }
//! }
//! ```
//!
//! `format` and `version` are required and fixed
//! ([`CUCA_EXPORT_FORMAT`]/[`CUCA_EXPORT_VERSION`]); version 1 has no
//! implicit migration, so an unknown format or version is rejected before any
//! payload is applied. The section names and shapes are fixed too: unknown
//! top-level fields, missing fields, and malformed sections all fail to
//! decode.
//!
//! # Canonical bytes
//!
//! Encoding emits compact UTF-8 with object keys sorted recursively at every
//! level, so neither map insertion order nor input key order can change the
//! output. Graph arrays are already sorted by id by
//! [`crate::plugins::memory::MemoryGraph::snapshot`] and cache entries by key,
//! so equivalent state always produces identical bytes. Component values are
//! never normalized, redacted, or rewritten by this format.
//!
//! # Compiled-out components
//!
//! A build without `plugin-memory` (or without `service-prompt-cache`) still
//! emits that component's section with the same wire shape and empty arrays,
//! and refuses to import non-empty data for it: the data is rejected as
//! [`CucaExportError::Unsupported`] rather than silently discarded.

use serde::{Deserialize, Serialize};

/// The only accepted value of [`CucaExport::format`].
pub const CUCA_EXPORT_FORMAT: &str = "cuca-export";

/// The only accepted value of [`CucaExport::version`]. Version 1 has no
/// implicit migration path.
pub const CUCA_EXPORT_VERSION: u32 = 1;

/// Failure modes of the export envelope.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CucaExportError {
    /// The bytes were not valid JSON for the envelope schema (malformed JSON,
    /// missing field, unknown field, or wrong type).
    Json {
        /// Decoder detail.
        message: String,
    },
    /// A structural check failed for a named component and field.
    Validation {
        /// `"envelope"`, `"graph"`, or `"prompt_cache"`.
        component: &'static str,
        /// The field that failed the check.
        field: &'static str,
        /// Human-readable validation detail.
        message: String,
    },
    /// The document carries data for a component this build compiled out.
    Unsupported {
        /// `"graph"` or `"prompt_cache"`.
        component: &'static str,
    },
    /// Live state could not be read or replaced (e.g. a poisoned lock).
    State {
        /// Underlying state error detail.
        message: String,
    },
}

impl std::fmt::Display for CucaExportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CucaExportError::Json { message } => write!(f, "cuca-export JSON error: {message}"),
            CucaExportError::Validation {
                component,
                field,
                message,
            } => write!(
                f,
                "cuca-export validation failed for {component}.{field}: {message}"
            ),
            CucaExportError::Unsupported { component } => write!(
                f,
                "cuca-export carries {component} data, but this build compiled that component out"
            ),
            CucaExportError::State { message } => write!(f, "cuca-export state error: {message}"),
        }
    }
}

impl std::error::Error for CucaExportError {}

/// The `graph` section: [`crate::plugins::memory::GraphSnapshot`] when
/// `plugin-memory` is enabled.
#[cfg(feature = "plugin-memory")]
pub type GraphExportSection = crate::plugins::memory::GraphSnapshot;

/// The `graph` section in a build without `plugin-memory`: the same wire
/// shape, but no typed graph values.
///
/// Both arrays must stay empty. A document with graph data is rejected as
/// [`CucaExportError::Unsupported`] on both encode and decode, so a build that
/// cannot import graph state can never silently discard it either.
#[cfg(not(feature = "plugin-memory"))]
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GraphExportSection {
    /// Must be empty in this build.
    pub nodes: Vec<serde_json::Value>,
    /// Must be empty in this build.
    pub relationships: Vec<serde_json::Value>,
}

/// The `prompt_cache` section:
/// [`crate::services::prompt_cache::PromptCacheSnapshot`] when
/// `service-prompt-cache` is enabled.
#[cfg(feature = "service-prompt-cache")]
pub type PromptCacheExportSection = crate::services::prompt_cache::PromptCacheSnapshot;

/// The `prompt_cache` section in a build without `service-prompt-cache`: the
/// same wire shape, but no typed cache entries.
///
/// `entries` must stay empty. A document with cache data is rejected as
/// [`CucaExportError::Unsupported`] on both encode and decode.
#[cfg(not(feature = "service-prompt-cache"))]
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PromptCacheExportSection {
    /// Must be empty in this build.
    pub entries: Vec<serde_json::Value>,
}

/// A versioned bundle of exportable CUCA state.
///
/// Build one with [`CucaExport::new`], which fixes `format` and `version`;
/// encode with [`CucaExport::to_json_bytes`] and decode with
/// [`CucaExport::from_json_slice`]. There is no file, path, or writer API:
/// the envelope only converts between itself and JSON bytes held in memory.
///
/// **Sensitive full-fidelity export:** `cuca-export` intentionally includes
/// the complete memory graph and local-cache request/response values. It may
/// contain confidential system prompts, user messages, tool arguments and
/// results, base64 image data, model output, signatures, and graph properties.
/// Treat the JSON as sensitive data; do not log or publish it. CUCA does not
/// encrypt, redact, or write it. The caller owns access control, encryption,
/// storage, and deletion.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CucaExport {
    /// Always [`CUCA_EXPORT_FORMAT`].
    pub format: String,
    /// Always [`CUCA_EXPORT_VERSION`].
    pub version: u32,
    /// Memory-graph state.
    pub graph: GraphExportSection,
    /// Local-response-cache state.
    pub prompt_cache: PromptCacheExportSection,
}

impl CucaExport {
    /// Bundle two component sections under the fixed format and version.
    pub fn new(graph: GraphExportSection, prompt_cache: PromptCacheExportSection) -> Self {
        Self {
            format: CUCA_EXPORT_FORMAT.to_string(),
            version: CUCA_EXPORT_VERSION,
            graph,
            prompt_cache,
        }
    }

    /// Check the format, the version, and every compiled-out section.
    ///
    /// Runs before encoding and immediately after decoding, so no caller can
    /// observe or apply an envelope this build does not fully support.
    ///
    /// # Errors
    ///
    /// [`CucaExportError::Validation`] for a wrong `format` or unknown
    /// `version`; [`CucaExportError::Unsupported`] when a section carries data
    /// for a component this build compiled out.
    pub fn validate(&self) -> Result<(), CucaExportError> {
        if self.format != CUCA_EXPORT_FORMAT {
            return Err(CucaExportError::Validation {
                component: "envelope",
                field: "format",
                message: format!(
                    "expected format '{CUCA_EXPORT_FORMAT}', got '{}'",
                    self.format
                ),
            });
        }
        if self.version != CUCA_EXPORT_VERSION {
            return Err(CucaExportError::Validation {
                component: "envelope",
                field: "version",
                message: format!(
                    "expected version {CUCA_EXPORT_VERSION}, got {}",
                    self.version
                ),
            });
        }
        #[cfg(not(feature = "plugin-memory"))]
        if !self.graph.nodes.is_empty() || !self.graph.relationships.is_empty() {
            return Err(CucaExportError::Unsupported { component: "graph" });
        }
        #[cfg(not(feature = "service-prompt-cache"))]
        if !self.prompt_cache.entries.is_empty() {
            return Err(CucaExportError::Unsupported {
                component: "prompt_cache",
            });
        }
        Ok(())
    }

    /// Encode to canonical compact UTF-8 JSON bytes: object keys sorted
    /// recursively, no insignificant whitespace, no I/O.
    ///
    /// # Errors
    ///
    /// Whatever [`Self::validate`] returns, or [`CucaExportError::Json`] if
    /// serialization fails.
    pub fn to_json_bytes(&self) -> Result<Vec<u8>, CucaExportError> {
        self.validate()?;
        let value = serde_json::to_value(self).map_err(|e| CucaExportError::Json {
            message: e.to_string(),
        })?;
        serde_json::to_vec(&canonical_value(value)).map_err(|e| CucaExportError::Json {
            message: e.to_string(),
        })
    }

    /// Decode and validate JSON bytes held in memory.
    ///
    /// # Errors
    ///
    /// [`CucaExportError::Json`] for malformed JSON, a missing field, an
    /// unknown field, or a wrong type; otherwise whatever [`Self::validate`]
    /// returns.
    pub fn from_json_slice(bytes: &[u8]) -> Result<Self, CucaExportError> {
        let export: Self = serde_json::from_slice(bytes).map_err(|e| CucaExportError::Json {
            message: e.to_string(),
        })?;
        export.validate()?;
        Ok(export)
    }
}

/// Counts accepted by a successful [`CucaExport::import_into`].
///
/// Graph counts are the complete replaced state; cache counts split the
/// snapshot's entries into the ones installed, the ones skipped because they
/// had already expired at the import instant, and the ones dropped only
/// because the destination cache's capacity is smaller than the snapshot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CucaImportReport {
    /// Nodes in the imported graph.
    pub graph_nodes: usize,
    /// Relationships in the imported graph.
    pub graph_relationships: usize,
    /// Cache entries actually installed.
    pub imported_cache_entries: usize,
    /// Cache entries skipped because they had already expired.
    pub expired_cache_entries: usize,
    /// Live cache entries dropped to fit the destination capacity.
    pub capacity_evictions: usize,
}

#[cfg(all(feature = "plugin-memory", feature = "service-prompt-cache"))]
impl CucaExport {
    /// Build the combined envelope from live component state.
    ///
    /// Each component is exported through its own accessor
    /// ([`crate::plugins::memory::MemoryPlugin::snapshot`] and
    /// [`crate::services::prompt_cache::PromptCache::snapshot`]), so each holds
    /// its lock only long enough to clone its own state. Expired cache entries
    /// are pruned from the export by the cache itself.
    ///
    /// See the [`CucaExport`] type docs for the sensitive-data warning.
    ///
    /// # Errors
    ///
    /// [`CucaExportError::State`] when a component's lock is poisoned or its
    /// export fails.
    pub fn from_live(
        memory: &crate::plugins::memory::MemoryPlugin,
        cache: &crate::services::prompt_cache::PromptCache,
    ) -> Result<Self, CucaExportError> {
        let graph = memory.snapshot().map_err(|e| CucaExportError::State {
            message: e.to_string(),
        })?;
        let prompt_cache = cache.snapshot().map_err(|e| CucaExportError::State {
            message: e.to_string(),
        })?;
        Ok(Self::new(graph, prompt_cache))
    }

    /// Apply this envelope to live component state as one transaction.
    ///
    /// Phase 1 validates the envelope, then stages **both** components
    /// completely: the graph through [`crate::plugins::memory::MemoryPlugin`]'s
    /// staging seam (duplicate ids, non-finite weights, endpoint existence,
    /// full adjacency rebuild)
    /// and the cache through its staging seam (key shape, digest match,
    /// timestamp order, rank uniqueness, duplicate keys, including duplicates
    /// among already-expired entries, then expiration filtering and capacity
    /// trimming against `now_unix_ms`). No live lock is acquired in this
    /// phase, so every validation failure returns before anything is swapped.
    ///
    /// Phase 2 commits the staged values in fixed graph-then-cache order, one
    /// lock hold each. A commit can only fail on a poisoned mutex; the graph
    /// is swapped first, so a poisoned cache mutex is reported as
    /// [`CucaExportError::State`] after the graph has already been replaced.
    /// A poisoned lock already means a previous holder panicked mid-update.
    ///
    /// The import is a wholesale replacement of both components, never a
    /// merge.
    ///
    /// # Errors
    ///
    /// [`CucaExportError::Validation`] for a wrong format/version
    /// (`component: "envelope"`), a rejected graph (`component: "graph"`), or
    /// a rejected cache snapshot (`component: "prompt_cache"`);
    /// [`CucaExportError::Json`] when digesting a cache entry's request fails;
    /// [`CucaExportError::State`] when a component lock is poisoned.
    pub fn import_into(
        &self,
        memory: &crate::plugins::memory::MemoryPlugin,
        cache: &crate::services::prompt_cache::PromptCache,
        now_unix_ms: u64,
    ) -> Result<CucaImportReport, CucaExportError> {
        use crate::plugins::memory::MemoryPlugin;

        // --- phase 1: validate and stage everything, touching nothing ---
        self.validate()?;
        let staged_graph = MemoryPlugin::stage_snapshot(self.graph.clone()).map_err(|e| {
            CucaExportError::Validation {
                component: "graph",
                field: "snapshot",
                message: e.to_string(),
            }
        })?;
        let staged_cache = cache
            .stage_snapshot(self.prompt_cache.clone(), now_unix_ms)
            .map_err(cache_stage_error)?;

        // --- phase 2: commit in fixed graph-then-cache order ---
        let graph_report =
            memory
                .commit_staged_graph(staged_graph)
                .map_err(|e| CucaExportError::State {
                    message: e.to_string(),
                })?;
        let cache_report =
            cache
                .commit_staged(staged_cache)
                .map_err(|e| CucaExportError::State {
                    message: e.to_string(),
                })?;

        Ok(CucaImportReport {
            graph_nodes: graph_report.nodes,
            graph_relationships: graph_report.relationships,
            imported_cache_entries: cache_report.imported_entries,
            expired_cache_entries: cache_report.expired_entries,
            capacity_evictions: cache_report.capacity_evictions,
        })
    }
}

/// Map a cache staging failure onto the envelope's error contract.
///
/// The cache reports a dynamic field path (e.g. `entries[<key>].lru_rank`),
/// which is folded into the message because [`CucaExportError::Validation`]
/// carries a static field name.
#[cfg(all(feature = "plugin-memory", feature = "service-prompt-cache"))]
fn cache_stage_error(error: crate::services::prompt_cache::PromptCacheError) -> CucaExportError {
    use crate::services::prompt_cache::PromptCacheError;

    match error {
        PromptCacheError::Validation { field, message } => CucaExportError::Validation {
            component: "prompt_cache",
            field: "entries",
            message: format!("{field}: {message}"),
        },
        PromptCacheError::Config(message) => CucaExportError::Validation {
            component: "prompt_cache",
            field: "config",
            message,
        },
        PromptCacheError::Json(message) => CucaExportError::Json { message },
        PromptCacheError::Lock(message) => CucaExportError::State { message },
    }
}

/// Rebuild `value` with every object's keys in sorted order.
///
/// `serde_json::Map` is `BTreeMap`-backed unless `preserve_order` is enabled
/// somewhere in the dependency graph; sorting explicitly makes canonical bytes
/// independent of that build detail and of any map's insertion order.
fn canonical_value(value: serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Object(map) => {
            let mut pairs: Vec<(String, serde_json::Value)> = map.into_iter().collect();
            pairs.sort_by(|(a, _), (b, _)| a.cmp(b));
            serde_json::Value::Object(
                pairs
                    .into_iter()
                    .map(|(k, v)| (k, canonical_value(v)))
                    .collect(),
            )
        }
        serde_json::Value::Array(items) => {
            serde_json::Value::Array(items.into_iter().map(canonical_value).collect())
        }
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The canonical serialization of a fully empty export, keys sorted at
    /// every level and no insignificant whitespace.
    const EMPTY_CANONICAL: &str = concat!(
        r#"{"format":"cuca-export","#,
        r#""graph":{"nodes":[],"relationships":[]},"#,
        r#""prompt_cache":{"entries":[]},"#,
        r#""version":1}"#
    );
    /// An empty section of each kind, spelled so it compiles whether the
    /// component's typed DTO or its compiled-out wire DTO is in scope.
    fn empty_graph_section() -> GraphExportSection {
        GraphExportSection {
            nodes: Vec::new(),
            relationships: Vec::new(),
        }
    }

    fn empty_cache_section() -> PromptCacheExportSection {
        PromptCacheExportSection {
            entries: Vec::new(),
        }
    }

    fn empty_export() -> CucaExport {
        CucaExport::new(empty_graph_section(), empty_cache_section())
    }

    #[test]
    fn new_uses_the_fixed_format_and_version() {
        let export = empty_export();
        assert_eq!(export.format, CUCA_EXPORT_FORMAT);
        assert_eq!(export.format, "cuca-export");
        assert_eq!(export.version, CUCA_EXPORT_VERSION);
        assert_eq!(export.version, 1);
    }

    #[test]
    fn to_json_bytes_emits_exactly_the_four_wire_keys() {
        let bytes = empty_export()
            .to_json_bytes()
            .expect("empty export encodes");
        let text = std::str::from_utf8(&bytes).expect("canonical JSON is UTF-8");
        assert_eq!(text, EMPTY_CANONICAL, "canonical compact wire shape");

        let value: serde_json::Value = serde_json::from_str(text).unwrap();
        let obj = value.as_object().expect("envelope is an object");
        let mut keys: Vec<&str> = obj.keys().map(String::as_str).collect();
        keys.sort_unstable();
        assert_eq!(keys, vec!["format", "graph", "prompt_cache", "version"]);
        let mut graph_keys: Vec<&str> = obj["graph"]
            .as_object()
            .expect("graph is an object")
            .keys()
            .map(String::as_str)
            .collect();
        graph_keys.sort_unstable();
        assert_eq!(graph_keys, vec!["nodes", "relationships"]);
        let cache_keys: Vec<&str> = obj["prompt_cache"]
            .as_object()
            .expect("prompt_cache is an object")
            .keys()
            .map(String::as_str)
            .collect();
        assert_eq!(cache_keys, vec!["entries"]);
    }

    /// Input key order never reaches the output: envelopes parsed from
    /// differently ordered JSON re-serialize to identical canonical bytes.
    #[test]
    fn canonical_bytes_are_independent_of_input_key_order() {
        let scrambled = concat!(
            r#"{"version":1,"#,
            r#""prompt_cache":{"entries":[]},"#,
            r#""graph":{"relationships":[],"nodes":[]},"#,
            r#""format":"cuca-export"}"#
        );
        let a = CucaExport::from_json_slice(EMPTY_CANONICAL.as_bytes()).expect("canonical parses");
        let b = CucaExport::from_json_slice(scrambled.as_bytes()).expect("scrambled parses");
        assert_eq!(a, b, "key order is not part of the value");
        assert_eq!(a.to_json_bytes().unwrap(), b.to_json_bytes().unwrap());
        assert_eq!(
            std::str::from_utf8(&b.to_json_bytes().unwrap()).unwrap(),
            EMPTY_CANONICAL
        );
    }

    #[test]
    fn from_json_slice_rejects_unknown_top_level_fields() {
        let json = concat!(
            r#"{"format":"cuca-export","version":1,"#,
            r#""graph":{"nodes":[],"relationships":[]},"#,
            r#""prompt_cache":{"entries":[]},"extra":true}"#
        );
        let err = CucaExport::from_json_slice(json.as_bytes()).unwrap_err();
        assert!(
            matches!(err, CucaExportError::Json { .. }),
            "unknown top-level fields are a decode failure, got {err:?}"
        );
    }

    #[test]
    fn from_json_slice_rejects_missing_required_fields() {
        for json in [
            r#"{"version":1,"graph":{"nodes":[],"relationships":[]},"prompt_cache":{"entries":[]}}"#,
            r#"{"format":"cuca-export","graph":{"nodes":[],"relationships":[]},"prompt_cache":{"entries":[]}}"#,
            r#"{"format":"cuca-export","version":1,"prompt_cache":{"entries":[]}}"#,
            r#"{"format":"cuca-export","version":1,"graph":{"nodes":[],"relationships":[]}}"#,
            r#"{}"#,
        ] {
            let err = CucaExport::from_json_slice(json.as_bytes()).unwrap_err();
            assert!(
                matches!(err, CucaExportError::Json { .. }),
                "missing fields must fail decoding for {json}, got {err:?}"
            );
        }
    }

    #[test]
    fn from_json_slice_rejects_a_wrong_format() {
        let json = concat!(
            r#"{"format":"cuca-dump","version":1,"#,
            r#""graph":{"nodes":[],"relationships":[]},"prompt_cache":{"entries":[]}}"#
        );
        let err = CucaExport::from_json_slice(json.as_bytes()).unwrap_err();
        assert_eq!(
            err,
            CucaExportError::Validation {
                component: "envelope",
                field: "format",
                message: "expected format 'cuca-export', got 'cuca-dump'".to_string(),
            }
        );
    }

    #[test]
    fn from_json_slice_rejects_an_unknown_version() {
        for bad in ["0", "2", "4294967295"] {
            let json = format!(
                concat!(
                    r#"{{"format":"cuca-export","version":{},"#,
                    r#""graph":{{"nodes":[],"relationships":[]}},"prompt_cache":{{"entries":[]}}}}"#
                ),
                bad
            );
            let err = CucaExport::from_json_slice(json.as_bytes()).unwrap_err();
            assert!(
                matches!(
                    err,
                    CucaExportError::Validation {
                        component: "envelope",
                        field: "version",
                        ..
                    }
                ),
                "version {bad} must be rejected, got {err:?}"
            );
        }
    }

    #[test]
    fn from_json_slice_rejects_malformed_section_shapes() {
        for json in [
            // graph is not an object
            r#"{"format":"cuca-export","version":1,"graph":[],"prompt_cache":{"entries":[]}}"#,
            // prompt_cache is not an object
            r#"{"format":"cuca-export","version":1,"graph":{"nodes":[],"relationships":[]},"prompt_cache":7}"#,
            // entries is not an array
            r#"{"format":"cuca-export","version":1,"graph":{"nodes":[],"relationships":[]},"prompt_cache":{"entries":{}}}"#,
            // graph section missing a required array
            r#"{"format":"cuca-export","version":1,"graph":{"nodes":[]},"prompt_cache":{"entries":[]}}"#,
            // version is not a number
            r#"{"format":"cuca-export","version":"1","graph":{"nodes":[],"relationships":[]},"prompt_cache":{"entries":[]}}"#,
            // not JSON at all
            r#"not json"#,
        ] {
            let err = CucaExport::from_json_slice(json.as_bytes()).unwrap_err();
            assert!(
                matches!(err, CucaExportError::Json { .. }),
                "malformed shape must fail decoding for {json}, got {err:?}"
            );
        }
    }

    #[test]
    fn error_display_is_informative() {
        assert!(
            CucaExportError::Unsupported { component: "graph" }
                .to_string()
                .contains("graph")
        );
        assert!(
            CucaExportError::Json {
                message: "boom".to_string()
            }
            .to_string()
            .contains("boom")
        );
    }

    // --- compiled-out sections -------------------------------------------

    #[cfg(not(feature = "plugin-memory"))]
    #[test]
    fn non_empty_graph_section_is_unsupported_without_plugin_memory() {
        for json in [
            concat!(
                r#"{"format":"cuca-export","version":1,"#,
                r#""graph":{"nodes":[{"id":"a","labels":[],"properties":{}}],"relationships":[]},"#,
                r#""prompt_cache":{"entries":[]}}"#
            ),
            concat!(
                r#"{"format":"cuca-export","version":1,"#,
                r#""graph":{"nodes":[],"relationships":[{"id":"r"}]},"#,
                r#""prompt_cache":{"entries":[]}}"#
            ),
        ] {
            let err = CucaExport::from_json_slice(json.as_bytes()).unwrap_err();
            assert_eq!(
                err,
                CucaExportError::Unsupported { component: "graph" },
                "compiled-out graph data must be rejected, never discarded"
            );
        }
    }

    /// A hand-constructed non-empty compiled-out section cannot be emitted
    /// either: encoding rejects it instead of writing data the build cannot
    /// import back.
    #[cfg(not(feature = "plugin-memory"))]
    #[test]
    fn encoding_a_non_empty_compiled_out_graph_section_is_unsupported() {
        let mut export = empty_export();
        export.graph.nodes.push(serde_json::json!({"id": "a"}));
        assert_eq!(
            export.to_json_bytes().unwrap_err(),
            CucaExportError::Unsupported { component: "graph" }
        );
    }

    #[cfg(not(feature = "service-prompt-cache"))]
    #[test]
    fn non_empty_cache_section_is_unsupported_without_service_prompt_cache() {
        let json = concat!(
            r#"{"format":"cuca-export","version":1,"#,
            r#""graph":{"nodes":[],"relationships":[]},"#,
            r#""prompt_cache":{"entries":[{"key":"deadbeef"}]}}"#
        );
        let err = CucaExport::from_json_slice(json.as_bytes()).unwrap_err();
        assert_eq!(
            err,
            CucaExportError::Unsupported {
                component: "prompt_cache"
            }
        );
    }

    #[cfg(not(feature = "service-prompt-cache"))]
    #[test]
    fn encoding_a_non_empty_compiled_out_cache_section_is_unsupported() {
        let mut export = empty_export();
        export
            .prompt_cache
            .entries
            .push(serde_json::json!({"key": "deadbeef"}));
        assert_eq!(
            export.to_json_bytes().unwrap_err(),
            CucaExportError::Unsupported {
                component: "prompt_cache"
            }
        );
    }

    // --- typed sections under their feature gates ------------------------

    #[cfg(feature = "plugin-memory")]
    #[test]
    fn graph_section_round_trips_labels_and_nested_properties() {
        use crate::plugins::memory::{GraphNode, GraphRelationship, GraphSnapshot};

        let mut properties = serde_json::Map::new();
        properties.insert("z".to_string(), serde_json::json!({"b": 1, "a": [2, 3]}));
        properties.insert("a".to_string(), serde_json::Value::Null);
        let export = CucaExport::new(
            GraphSnapshot {
                nodes: vec![GraphNode {
                    id: "alice".to_string(),
                    labels: vec!["person".to_string(), "author".to_string()],
                    properties: properties.clone(),
                }],
                relationships: vec![GraphRelationship {
                    id: "r1".to_string(),
                    from: "alice".to_string(),
                    to: "alice".to_string(),
                    kind: "knows".to_string(),
                    weight: 1.5,
                    properties: serde_json::Map::new(),
                }],
            },
            empty_cache_section(),
        );

        let bytes = export.to_json_bytes().expect("graph export encodes");
        let back = CucaExport::from_json_slice(&bytes).expect("graph export decodes");
        assert_eq!(back, export, "graph values survive unchanged");
        assert_eq!(back.graph.nodes[0].labels, vec!["person", "author"]);
        assert_eq!(back.graph.nodes[0].properties, properties);
        assert_eq!(back.graph.relationships[0].weight, 1.5);
        // Nested object keys are sorted recursively, so the emitted bytes are
        // identical regardless of how the property map was built.
        let text = std::str::from_utf8(&bytes).unwrap();
        assert!(
            text.contains(r#""properties":{"a":null,"z":{"a":[2,3],"b":1}}"#),
            "nested keys must be sorted: {text}"
        );
    }

    #[cfg(feature = "service-prompt-cache")]
    #[test]
    fn cache_section_round_trips_full_fidelity_entries() {
        use crate::request::{UnifiedRequest, UnifiedResponse};
        use crate::services::prompt_cache::{PromptCacheEntry, PromptCacheSnapshot};
        use crate::types::{MessageContentBlock, ProviderEndpoint};

        let request = UnifiedRequest::new("gpt-4o")
            .add_system_message("be concise")
            .add_user_message("hello");
        let response = UnifiedResponse {
            model: "gpt-4o".to_string(),
            provider: ProviderEndpoint::OpenAi,
            duration_secs: 1.0,
            prompt_tokens: 10,
            completion_tokens: 5,
            finish_reason: Some("stop".to_string()),
            content: vec![MessageContentBlock::Text("ok".to_string())],
            prompt_cache_usage: None,
        };
        let entry = PromptCacheEntry {
            key: "a".repeat(64),
            request,
            response,
            stored_at_unix_ms: 1_000,
            expires_at_unix_ms: 61_000,
            lru_rank: 0,
        };
        let export = CucaExport::new(
            empty_graph_section(),
            PromptCacheSnapshot {
                entries: vec![entry.clone()],
            },
        );

        let bytes = export.to_json_bytes().expect("cache export encodes");
        let back = CucaExport::from_json_slice(&bytes).expect("cache export decodes");
        assert_eq!(back, export, "cache entries survive unchanged");
        let round_tripped = &back.prompt_cache.entries[0];
        assert_eq!(round_tripped.key, entry.key);
        assert_eq!(round_tripped.request, entry.request);
        assert_eq!(round_tripped.response, entry.response);
        assert_eq!(round_tripped.stored_at_unix_ms, 1_000);
        assert_eq!(round_tripped.expires_at_unix_ms, 61_000);
        assert_eq!(round_tripped.lru_rank, 0);
    }
}

#[cfg(all(test, feature = "plugin-memory", feature = "service-prompt-cache"))]
mod coordinator_tests {
    use std::time::Duration;

    use super::*;
    use crate::plugins::memory::{
        GraphNode, GraphRelationship, GraphSnapshot, MemoryConfig, MemoryPlugin,
    };
    use crate::request::{UnifiedRequest, UnifiedResponse};
    use crate::services::prompt_cache::{
        PromptCache, PromptCacheConfig, PromptCacheEntry, PromptCacheSnapshot, digest_request,
    };
    use crate::types::{MessageContentBlock, ProviderEndpoint};

    const NOW: u64 = 10_000;

    fn node(id: &str) -> GraphNode {
        GraphNode {
            id: id.to_string(),
            labels: Vec::new(),
            properties: serde_json::Map::new(),
        }
    }

    fn rel(id: &str, from: &str, to: &str, weight: f64) -> GraphRelationship {
        GraphRelationship {
            id: id.to_string(),
            from: from.to_string(),
            to: to.to_string(),
            kind: "knows".to_string(),
            weight,
            properties: serde_json::Map::new(),
        }
    }

    fn response() -> UnifiedResponse {
        UnifiedResponse {
            model: "gpt-4o".to_string(),
            provider: ProviderEndpoint::OpenAi,
            duration_secs: 1.0,
            prompt_tokens: 10,
            completion_tokens: 5,
            finish_reason: Some("stop".to_string()),
            content: vec![MessageContentBlock::Text("ok".to_string())],
            prompt_cache_usage: None,
        }
    }

    /// A structurally valid entry for `prompt`, live from `NOW` unless
    /// `expires_at` says otherwise.
    fn entry(prompt: &str, rank: usize, stored_at: u64, expires_at: u64) -> PromptCacheEntry {
        let request = UnifiedRequest::new("gpt-4o").add_user_message(prompt);
        PromptCacheEntry {
            key: digest_request(&request).expect("digest"),
            request,
            response: response(),
            stored_at_unix_ms: stored_at,
            expires_at_unix_ms: expires_at,
            lru_rank: rank,
        }
    }

    fn live_entry(prompt: &str, rank: usize) -> PromptCacheEntry {
        entry(prompt, rank, NOW - 1_000, NOW + 60_000)
    }

    fn expired_entry(prompt: &str, rank: usize) -> PromptCacheEntry {
        entry(prompt, rank, NOW - 5_000, NOW - 1)
    }

    /// Sentinel live state: a two-node graph with one relationship, and a
    /// cache holding one long-lived entry.
    fn sentinels(capacity: usize) -> (MemoryPlugin, PromptCache) {
        let memory = MemoryPlugin::new(MemoryConfig::default()).expect("plugin builds");
        memory
            .replace_snapshot(GraphSnapshot {
                nodes: vec![node("sentinel"), node("other")],
                relationships: vec![rel("keep", "sentinel", "other", 1.0)],
            })
            .expect("sentinel graph imports");
        let cache = PromptCache::new(
            PromptCacheConfig::new(capacity, Duration::from_secs(3_600)).expect("config"),
        )
        .expect("cache builds");
        cache
            .insert(
                UnifiedRequest::new("gpt-4o").add_user_message("sentinel cache entry"),
                response(),
            )
            .expect("sentinel insert");
        (memory, cache)
    }

    fn envelope(graph: GraphSnapshot, cache: PromptCacheSnapshot) -> CucaExport {
        CucaExport::new(graph, cache)
    }

    fn valid_graph() -> GraphSnapshot {
        GraphSnapshot {
            nodes: vec![node("dave"), node("erin")],
            relationships: vec![rel("r9", "dave", "erin", -0.5)],
        }
    }

    #[test]
    fn from_live_builds_the_combined_envelope() {
        let (memory, cache) = sentinels(4);
        let export = CucaExport::from_live(&memory, &cache).expect("live export succeeds");

        assert_eq!(export.format, CUCA_EXPORT_FORMAT);
        assert_eq!(export.version, CUCA_EXPORT_VERSION);
        assert_eq!(
            export
                .graph
                .nodes
                .iter()
                .map(|n| n.id.as_str())
                .collect::<Vec<_>>(),
            vec!["other", "sentinel"]
        );
        assert_eq!(export.graph.relationships.len(), 1);
        assert_eq!(export.prompt_cache.entries.len(), 1);
        // The envelope it produces is canonical and re-importable.
        let bytes = export.to_json_bytes().expect("encodes");
        assert_eq!(CucaExport::from_json_slice(&bytes).unwrap(), export);
    }

    #[test]
    fn import_into_replaces_both_components_and_reports_counts() {
        let (memory, cache) = sentinels(4);
        let export = envelope(
            valid_graph(),
            PromptCacheSnapshot {
                entries: vec![
                    live_entry("first", 0),
                    expired_entry("second", 1),
                    live_entry("third", 2),
                ],
            },
        );

        let report = export
            .import_into(&memory, &cache, NOW)
            .expect("a valid combined import succeeds");
        assert_eq!(
            report,
            CucaImportReport {
                graph_nodes: 2,
                graph_relationships: 1,
                imported_cache_entries: 2,
                expired_cache_entries: 1,
                capacity_evictions: 0,
            }
        );

        // Graph replaced wholesale.
        let graph = memory.snapshot().expect("graph snapshot");
        assert_eq!(graph, valid_graph());
        assert!(graph.nodes.iter().all(|n| n.id != "sentinel"));
        // Cache replaced wholesale: the sentinel entry is gone, the expired
        // snapshot entry was skipped, ranks renumbered from 0.
        let cache_state = cache.snapshot_at(NOW).expect("cache snapshot");
        assert_eq!(cache_state.entries.len(), 2);
        let mut ranks: Vec<usize> = cache_state.entries.iter().map(|e| e.lru_rank).collect();
        ranks.sort_unstable();
        assert_eq!(ranks, vec![0, 1]);
        let sentinel_key =
            digest_request(&UnifiedRequest::new("gpt-4o").add_user_message("sentinel cache entry"))
                .unwrap();
        assert!(cache_state.entries.iter().all(|e| e.key != sentinel_key));
    }

    #[test]
    fn import_into_reports_capacity_evictions() {
        let (memory, cache) = sentinels(1);
        let export = envelope(
            valid_graph(),
            PromptCacheSnapshot {
                entries: vec![live_entry("first", 0), live_entry("second", 1)],
            },
        );

        let report = export
            .import_into(&memory, &cache, NOW)
            .expect("import succeeds within destination capacity");
        assert_eq!(
            report,
            CucaImportReport {
                graph_nodes: 2,
                graph_relationships: 1,
                imported_cache_entries: 1,
                expired_cache_entries: 0,
                capacity_evictions: 1,
            }
        );
        let entries = cache.snapshot_at(NOW).unwrap().entries;
        assert_eq!(
            entries.len(),
            1,
            "only the destination capacity is retained"
        );
        // The most recently used entry (highest incoming rank) survives.
        let newest =
            digest_request(&UnifiedRequest::new("gpt-4o").add_user_message("second")).unwrap();
        assert_eq!(entries[0].key, newest);
    }

    #[test]
    fn from_live_to_json_bytes_from_json_slice_import_into_round_trips_a_resident_expired_entry() {
        // Regression test for a bug where `PromptCache::snapshot_at` derived
        // exported `lru_rank` values from positions in the UNPRUNED
        // `lru_order` while filtering expired entries out of the exported
        // set, producing non-contiguous ranks that `import_into`'s cache
        // staging then rejected. This exercises the real
        // `PromptCache::snapshot` entry point (via `from_live`), not the
        // `snapshot_at` test seam, and the full coordinator round trip.
        let memory = MemoryPlugin::new(MemoryConfig::default()).expect("plugin builds");
        let cache = PromptCache::new(
            PromptCacheConfig::new(5, Duration::from_secs(3_600)).expect("config"),
        )
        .expect("cache builds");

        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_millis() as u64;
        let about_to_expire = entry("resident-expired", 0, now_ms, now_ms + 50);
        let far_from_expiry = entry("resident-live", 1, now_ms, now_ms + 60_000);
        // Seed directly (bypassing `insert`'s own clock reads) so both
        // entries are physically resident with contiguous incoming ranks and
        // both live at seed time.
        cache
            .replace_snapshot_at(
                PromptCacheSnapshot {
                    entries: vec![about_to_expire.clone(), far_from_expiry.clone()],
                },
                now_ms,
            )
            .expect("seed import succeeds: both entries are live at seed time");

        // Let real wall-clock time pass the short-lived entry's expiry
        // without any `insert`/`lookup` call, so it stays physically
        // resident in the cache's internal state instead of being pruned.
        std::thread::sleep(Duration::from_millis(300));

        let export = CucaExport::from_live(&memory, &cache).expect("live export succeeds");
        assert_eq!(
            export.prompt_cache.entries.len(),
            1,
            "the resident-but-expired entry must be excluded from the export"
        );
        assert_eq!(export.prompt_cache.entries[0].key, far_from_expiry.key);

        let bytes = export.to_json_bytes().expect("encodes");
        let decoded = CucaExport::from_json_slice(&bytes).expect("decodes");

        let import_now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_millis() as u64;
        let report = decoded
            .import_into(&memory, &cache, import_now)
            .expect("a live export must always re-import cleanly");
        assert_eq!(report.imported_cache_entries, 1);
        assert_eq!(report.expired_cache_entries, 0);
        assert_eq!(report.capacity_evictions, 0);

        let final_state = cache.snapshot_at(import_now).unwrap();
        assert_eq!(final_state.entries.len(), 1);
        assert_eq!(final_state.entries[0].key, far_from_expiry.key);
        assert_eq!(final_state.entries[0].lru_rank, 0);
    }

    /// Every rejected import leaves BOTH components byte-identical: phase 1
    /// validates graph and cache completely before phase 2 swaps anything.
    #[test]
    fn every_invalid_import_leaves_both_components_untouched() {
        let (memory, cache) = sentinels(4);
        let graph_before = memory.snapshot().expect("graph snapshot");
        let cache_before = cache.snapshot_at(NOW).expect("cache snapshot");

        let bad_version = {
            let mut export = envelope(valid_graph(), PromptCacheSnapshot::default());
            export.version = 2;
            export
        };
        let bad_format = {
            let mut export = envelope(valid_graph(), PromptCacheSnapshot::default());
            export.format = "cuca-dump".to_string();
            export
        };
        let duplicate_node = envelope(
            GraphSnapshot {
                nodes: vec![node("dup"), node("dup")],
                relationships: Vec::new(),
            },
            PromptCacheSnapshot::default(),
        );
        let duplicate_relationship = envelope(
            GraphSnapshot {
                nodes: vec![node("a"), node("b")],
                relationships: vec![rel("same", "a", "b", 1.0), rel("same", "b", "a", 1.0)],
            },
            PromptCacheSnapshot::default(),
        );
        let missing_endpoint = envelope(
            GraphSnapshot {
                nodes: vec![node("a")],
                relationships: vec![rel("r", "a", "ghost", 1.0)],
            },
            PromptCacheSnapshot::default(),
        );
        let non_finite_weight = envelope(
            GraphSnapshot {
                nodes: vec![node("a"), node("b")],
                relationships: vec![rel("r", "a", "b", f64::INFINITY)],
            },
            PromptCacheSnapshot::default(),
        );
        let duplicate_cache_key = envelope(
            valid_graph(),
            PromptCacheSnapshot {
                entries: vec![live_entry("same", 0), live_entry("same", 1)],
            },
        );
        // Duplicate detection must run before expiration filtering, so two
        // already-expired copies of one key are still rejected.
        let duplicate_expired_key = envelope(
            valid_graph(),
            PromptCacheSnapshot {
                entries: vec![expired_entry("same", 0), expired_entry("same", 1)],
            },
        );
        let malformed_key = envelope(
            valid_graph(),
            PromptCacheSnapshot {
                entries: vec![PromptCacheEntry {
                    key: "NOT-A-DIGEST".to_string(),
                    ..live_entry("first", 0)
                }],
            },
        );
        let digest_mismatch = envelope(
            valid_graph(),
            PromptCacheSnapshot {
                entries: vec![PromptCacheEntry {
                    key: "a".repeat(64),
                    ..live_entry("first", 0)
                }],
            },
        );
        let invalid_timestamp = envelope(
            valid_graph(),
            PromptCacheSnapshot {
                entries: vec![entry("first", 0, NOW + 60_000, NOW + 60_000)],
            },
        );
        let invalid_rank = envelope(
            valid_graph(),
            PromptCacheSnapshot {
                entries: vec![live_entry("first", 7)],
            },
        );

        let cases: Vec<(&str, CucaExport, CucaExportError)> = vec![
            (
                "bad version",
                bad_version,
                CucaExportError::Validation {
                    component: "envelope",
                    field: "version",
                    message: "expected version 1, got 2".to_string(),
                },
            ),
            (
                "bad format",
                bad_format,
                CucaExportError::Validation {
                    component: "envelope",
                    field: "format",
                    message: "expected format 'cuca-export', got 'cuca-dump'".to_string(),
                },
            ),
        ];
        for (label, export, expected) in cases {
            assert_eq!(
                export.import_into(&memory, &cache, NOW).unwrap_err(),
                expected,
                "{label} must be rejected with the envelope error"
            );
            assert_eq!(memory.snapshot().unwrap(), graph_before, "{label}: graph");
            assert_eq!(
                cache.snapshot_at(NOW).unwrap(),
                cache_before,
                "{label}: cache"
            );
        }

        for (label, export) in [
            ("duplicate node id", duplicate_node),
            ("duplicate relationship id", duplicate_relationship),
            ("missing endpoint", missing_endpoint),
            ("non-finite weight", non_finite_weight),
        ] {
            let err = export.import_into(&memory, &cache, NOW).unwrap_err();
            assert!(
                matches!(
                    err,
                    CucaExportError::Validation {
                        component: "graph",
                        ..
                    }
                ),
                "{label} must be a graph validation error, got {err:?}"
            );
            assert_eq!(memory.snapshot().unwrap(), graph_before, "{label}: graph");
            assert_eq!(
                cache.snapshot_at(NOW).unwrap(),
                cache_before,
                "{label}: cache"
            );
        }

        for (label, export) in [
            ("duplicate cache key", duplicate_cache_key),
            ("duplicate expired cache key", duplicate_expired_key),
            ("malformed key", malformed_key),
            ("digest mismatch", digest_mismatch),
            ("invalid timestamp", invalid_timestamp),
            ("invalid rank", invalid_rank),
        ] {
            let err = export.import_into(&memory, &cache, NOW).unwrap_err();
            assert!(
                matches!(
                    err,
                    CucaExportError::Validation {
                        component: "prompt_cache",
                        ..
                    }
                ),
                "{label} must be a cache validation error, got {err:?}"
            );
            assert_eq!(memory.snapshot().unwrap(), graph_before, "{label}: graph");
            assert_eq!(
                cache.snapshot_at(NOW).unwrap(),
                cache_before,
                "{label}: cache"
            );
        }
    }

    /// A bad cache section must not let a valid graph section through: phase 1
    /// stages both components before phase 2 commits either.
    #[test]
    fn a_bad_cache_section_blocks_the_graph_swap() {
        let (memory, cache) = sentinels(4);
        let graph_before = memory.snapshot().unwrap();
        let export = envelope(
            valid_graph(),
            PromptCacheSnapshot {
                entries: vec![live_entry("first", 3)],
            },
        );
        assert!(export.import_into(&memory, &cache, NOW).is_err());
        assert_eq!(
            memory.snapshot().unwrap(),
            graph_before,
            "the graph must not swap when the cache section is invalid"
        );
    }

    /// A full cycle: export live state, import it back, and observe identical
    /// state and a report that accounts for every entry.
    #[test]
    fn live_export_import_round_trip_is_idempotent() {
        let (memory, cache) = sentinels(4);
        let export = CucaExport::from_live(&memory, &cache).unwrap();
        let bytes = export.to_json_bytes().unwrap();
        let decoded = CucaExport::from_json_slice(&bytes).unwrap();

        let report = decoded
            .import_into(&memory, &cache, NOW)
            .expect("re-importing live state succeeds");
        assert_eq!(report.graph_nodes, 2);
        assert_eq!(report.graph_relationships, 1);
        assert_eq!(report.imported_cache_entries, 1);
        assert_eq!(report.expired_cache_entries, 0);
        assert_eq!(report.capacity_evictions, 0);
        assert_eq!(memory.snapshot().unwrap(), export.graph);
        assert_eq!(cache.snapshot_at(NOW).unwrap(), export.prompt_cache);
    }
}
