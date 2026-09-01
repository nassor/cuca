//! Integration tests for schema-guided entity extraction
//! (`service-entity-extraction`).
//!
//! The deterministic tests drive the explicit-call contract with canned
//! [`EntityExtractionModel`] adapters: `extract` hands the caller's source and
//! the extractor's own schema to the model, propagates the model's own error
//! unchanged, and returns a delta that is inert until the application applies
//! it to a [`MemoryPlugin`] (the mandatory hand-off documented in
//! `src/services/entity_extraction.rs`). None of them touch the network.
//!
//! The live test wires the shared adapter in `tests/common/extraction.rs` to
//! llama.cpp through this crate's own `CucaClient`, asks the served model for
//! a small JSON extraction, maps the reply into an
//! [`EntityExtractionCandidate`], and validates it through the extractor. The
//! adapter only ever proposes rows that satisfy
//! [`common::extraction::org_schema`]: it dedups by identity and drops
//! non-string values. An `Err` from `extract` *after* a candidate was built
//! therefore means the validation contract broke, not that the model was
//! unhelpful. A model that never produces parseable JSON yields no candidate
//! at all and is reported as a model-quality skip.
#![cfg(all(feature = "provider-llamacpp", feature = "service-entity-extraction"))]

mod common;

use std::pin::Pin;
use std::sync::Mutex;

use common::extraction::{LiveExtractionModel, SOURCE, org_extractor, pair_candidate};
use cuca::{
    EntityExtractionCandidate, EntityExtractionModel, EntityExtractionSchema, MemoryConfig,
    MemoryPlugin, MergePolicy, PluginError,
};

fn memory() -> MemoryPlugin {
    MemoryPlugin::new(MemoryConfig::default()).expect("memory plugin must build")
}

/// A model that answers with a fixed result and records what it was asked.
struct CannedModel {
    result: Result<EntityExtractionCandidate, PluginError>,
    /// `(source, schema name)` per call.
    seen: Mutex<Vec<(String, String)>>,
}

impl CannedModel {
    fn ok(candidate: EntityExtractionCandidate) -> Self {
        Self {
            result: Ok(candidate),
            seen: Mutex::new(Vec::new()),
        }
    }

    fn err(error: PluginError) -> Self {
        Self {
            result: Err(error),
            seen: Mutex::new(Vec::new()),
        }
    }
}

impl EntityExtractionModel for CannedModel {
    fn extract<'a>(
        &'a self,
        source: &'a str,
        schema: &'a EntityExtractionSchema,
    ) -> Pin<Box<dyn Future<Output = Result<EntityExtractionCandidate, PluginError>> + Send + 'a>>
    {
        self.seen
            .lock()
            .expect("seen lock must not be poisoned")
            .push((source.to_owned(), schema.name.clone()));
        let result = self.result.clone();
        Box::pin(async move { result })
    }
}

#[tokio::test]
async fn extract_hands_the_source_and_the_extractors_schema_to_the_model() {
    let model = CannedModel::ok(pair_candidate("Ada", "Analytical Engines"));
    let report = org_extractor()
        .extract("Ada works at Analytical Engines.", &model)
        .await
        .expect("a schema-conformant candidate must be accepted");

    assert_eq!(report.nodes_accepted, 2);
    assert_eq!(
        model
            .seen
            .lock()
            .expect("seen lock must not be poisoned")
            .as_slice(),
        &[(
            "Ada works at Analytical Engines.".to_string(),
            "org-chart".to_string()
        )],
        "extract must pass the caller's source and the extractor's own schema, once"
    );
}

/// The report is a standalone delta: the extraction step never touches a
/// `MemoryPlugin`, and the delta only reaches the working graph when the
/// application merges it.
#[tokio::test]
async fn report_delta_is_inert_until_the_caller_merges_it() {
    let memory = memory();
    let model = CannedModel::ok(pair_candidate("Ada", "Analytical Engines"));
    let report = org_extractor()
        .extract("Ada works at Analytical Engines.", &model)
        .await
        .expect("candidate must be accepted");

    assert_eq!(report.nodes_accepted, 2, "one person plus one company");
    assert_eq!(report.relationships_accepted, 1, "one works_at edge");
    assert_eq!(
        report.nodes_accepted,
        report.delta.len(),
        "reported node count must equal the delta it describes"
    );
    assert_eq!(
        report.relationships_accepted,
        report.delta.relationship_count(),
        "reported relationship count must equal the delta it describes"
    );

    let delta_snapshot = report.delta.snapshot();
    assert!(
        memory
            .snapshot()
            .expect("graph lock must not be poisoned")
            .nodes
            .is_empty(),
        "extraction must not mutate memory state before the hand-off"
    );

    let merge = memory
        .merge_graph(report.delta, MergePolicy::Keep)
        .expect("merge must not fail");
    assert_eq!(merge.nodes_added, 2);
    assert_eq!(merge.relationships_added, 1);
    assert_eq!(merge.relationships_renamed, 0);
    assert_eq!(
        memory.snapshot().expect("graph lock must not be poisoned"),
        delta_snapshot,
        "the merged graph must be exactly the delta the report reported"
    );

    let edge = &delta_snapshot.relationships[0];
    assert_eq!(edge.kind, "works_at", "kind comes from the schema table");
    assert_eq!(edge.weight, 1.0, "extraction edges carry unit weight");
    let node_ids: Vec<&str> = delta_snapshot
        .nodes
        .iter()
        .map(|node| node.id.as_str())
        .collect();
    assert!(
        node_ids.contains(&edge.from.as_str()) && node_ids.contains(&edge.to.as_str()),
        "both endpoints must be nodes of the same delta, got {node_ids:?} for {edge:?}"
    );
    for node in &delta_snapshot.nodes {
        let expected = if node.properties["name"] == serde_json::json!("Ada") {
            "person"
        } else {
            "company"
        };
        assert_eq!(
            node.labels,
            vec![expected.to_string()],
            "labels come from the entity table, not the model"
        );
    }
}

/// Derived node ids are a pure function of table plus identity columns:
/// re-validating the same candidate yields byte-identical state, a different
/// identity yields a different id, and the same identity in a different table
/// stays distinct.
#[test]
fn derived_node_ids_are_stable_and_scoped_to_table_and_identity() {
    let extractor = org_extractor();
    let first = extractor
        .validate_candidate(pair_candidate("Ada", "Analytical Engines"))
        .expect("candidate must be accepted");
    let again = extractor
        .validate_candidate(pair_candidate("Ada", "Analytical Engines"))
        .expect("candidate must be accepted");
    assert_eq!(
        first.delta.snapshot(),
        again.delta.snapshot(),
        "identical candidates must derive identical graph state"
    );

    let renamed = extractor
        .validate_candidate(pair_candidate("Grace", "Analytical Engines"))
        .expect("candidate must be accepted");
    let ada = first.delta.snapshot();
    let grace = renamed.delta.snapshot();
    assert_ne!(
        ada.nodes[0].id, grace.nodes[0].id,
        "a different identity must derive a different node id"
    );
    assert_eq!(
        ada.nodes.iter().filter(|n| n.labels == ["company"]).count(),
        1
    );
    let company_id = |snapshot: &cuca::GraphSnapshot| {
        snapshot
            .nodes
            .iter()
            .find(|n| n.labels == ["company"])
            .expect("company node must exist")
            .id
            .clone()
    };
    assert_eq!(
        company_id(&ada),
        company_id(&grace),
        "an unchanged identity must derive the same node id across extractions"
    );

    // Same identity value, different table: the ids must not collide.
    let homonym = extractor
        .validate_candidate(pair_candidate("Ada", "Ada"))
        .expect("candidate must be accepted");
    let homonym = homonym.delta.snapshot();
    assert_eq!(homonym.nodes.len(), 2, "person and company stay separate");
    assert_ne!(
        homonym.nodes[0].id, homonym.nodes[1].id,
        "identity is scoped by table, got {homonym:?}"
    );
}

/// Applying the same extraction twice is node-idempotent but edge-additive
/// under `merge_graph`: the colliding relationship id is deterministically
/// renamed rather than dropped. A caller that wants exactly-once edges uses
/// `replace_graph`.
#[test]
fn re_merging_an_extraction_keeps_nodes_and_renames_the_duplicate_edge() {
    let memory = memory();
    let extractor = org_extractor();
    let delta = || {
        extractor
            .validate_candidate(pair_candidate("Ada", "Analytical Engines"))
            .expect("candidate must be accepted")
            .delta
    };

    let first = memory
        .merge_graph(delta(), MergePolicy::Keep)
        .expect("first merge must not fail");
    assert_eq!(first.nodes_added, 2);
    assert_eq!(first.relationships_added, 1);

    let second = memory
        .merge_graph(delta(), MergePolicy::Keep)
        .expect("second merge must not fail");
    assert_eq!(second.nodes_added, 0, "identities already present");
    assert_eq!(
        second.nodes_kept, 2,
        "Keep policy retains the existing rows"
    );
    assert_eq!(
        second.relationships_renamed, 1,
        "the colliding edge id must be renamed, never dropped"
    );
    assert_eq!(second.relationships_added, 0);

    let snapshot = memory.snapshot().expect("graph lock must not be poisoned");
    assert_eq!(snapshot.nodes.len(), 2, "no node duplication");
    assert_eq!(
        snapshot.relationships.len(),
        2,
        "re-merging accumulates a parallel edge"
    );

    memory
        .replace_graph(delta())
        .expect("replace must not fail");
    let snapshot = memory.snapshot().expect("graph lock must not be poisoned");
    assert_eq!(snapshot.nodes.len(), 2);
    assert_eq!(
        snapshot.relationships.len(),
        1,
        "replace_graph collapses the extraction back to exactly-once edges"
    );
}

/// A candidate that violates the schema is rejected as
/// [`PluginError::Validation`] naming the schema, and no partial delta or
/// memory mutation survives.
#[tokio::test]
async fn extract_rejects_a_broken_candidate_and_leaves_memory_untouched() {
    let memory = memory();
    let mut broken = pair_candidate("Ada", "Analytical Engines");
    broken.entities[0]
        .properties
        .insert("salary".into(), serde_json::json!(1));
    let model = CannedModel::ok(broken);

    let error = org_extractor()
        .extract("Ada works at Analytical Engines.", &model)
        .await
        .expect_err("an undeclared property on a strict table must be rejected");
    match error {
        PluginError::Validation { schema, message } => {
            assert_eq!(schema, "org-chart", "the error must name the schema");
            assert!(
                message.contains("salary"),
                "the message must name the offending property, got: {message}"
            );
        }
        other => panic!("expected PluginError::Validation, got {other:?}"),
    }
    assert!(
        memory
            .snapshot()
            .expect("graph lock must not be poisoned")
            .nodes
            .is_empty(),
        "a rejected extraction must leave no state behind"
    );

    // The same schema still accepts a well-formed candidate afterwards: the
    // rejection is per-candidate, not a poisoned extractor.
    let good = CannedModel::ok(pair_candidate("Ada", "Analytical Engines"));
    let report = org_extractor()
        .extract("Ada works at Analytical Engines.", &good)
        .await
        .expect("a valid candidate must still be accepted after a rejection");
    assert_eq!(report.nodes_accepted, 2);
}

/// A relationship whose endpoint is not among the accepted entities is
/// rejected outright rather than yielding an edgeless partial delta.
#[tokio::test]
async fn extract_rejects_a_relationship_with_an_unextracted_endpoint() {
    let mut candidate = pair_candidate("Ada", "Analytical Engines");
    candidate.entities.pop();
    let model = CannedModel::ok(candidate);

    let error = org_extractor()
        .extract("Ada works at Analytical Engines.", &model)
        .await
        .expect_err("a dangling endpoint must be rejected");
    assert!(
        matches!(&error, PluginError::Validation { schema, .. } if schema == "org-chart"),
        "expected a schema-named validation error, got {error:?}"
    );
}

/// The model's own failure surfaces unchanged: `extract` never repackages a
/// transport or adapter error as a validation error.
#[tokio::test]
async fn extract_propagates_the_models_error_unchanged() {
    let model = CannedModel::err(PluginError::Internal("adapter transport failed".into()));
    let error = org_extractor()
        .extract("anything", &model)
        .await
        .expect_err("the model's error must propagate");
    match error {
        PluginError::Internal(message) => assert_eq!(message, "adapter transport failed"),
        other => panic!("expected the model's own PluginError::Internal, got {other:?}"),
    }
}

/// End-to-end: llama.cpp produces an extraction, the extractor validates it into
/// a schema-typed delta, and the application hands the delta to a
/// `MemoryPlugin`.
#[tokio::test]
async fn live_extraction_validates_into_a_delta_and_merges_into_memory() {
    if let Err(reason) = common::require_live_server() {
        eprintln!("SKIP: llama.cpp not reachable: {reason}");
        return;
    }
    let model = LiveExtractionModel::new(common::live_model());
    let extractor = org_extractor();
    let memory = memory();

    let report = match extractor.extract(SOURCE, &model).await {
        Ok(report) => report,
        Err(error) if model.produced_no_candidate() => {
            eprintln!(
                "SKIP: the served model produced no parseable extraction in {} attempts \
                 ({error}); raw replies:\n{}",
                model.attempts(),
                model.diagnostics()
            );
            return;
        }
        Err(error) => panic!(
            "the extractor rejected an adapter-built, schema-conformant candidate: {error:?}\n\
             candidate: {:?}\nraw replies:\n{}",
            model.candidate(),
            model.diagnostics()
        ),
    };

    // Contract, not model quality: whatever the model said, the accepted delta
    // is schema-typed, internally consistent, and reported exactly.
    assert!(
        report.nodes_accepted > 0,
        "a produced candidate must yield at least one node"
    );
    assert_eq!(report.nodes_accepted, report.delta.len());
    assert_eq!(
        report.relationships_accepted,
        report.delta.relationship_count()
    );

    let snapshot = report.delta.snapshot();
    let node_ids: Vec<&str> = snapshot.nodes.iter().map(|n| n.id.as_str()).collect();
    for node in &snapshot.nodes {
        assert!(
            node.id.starts_with("entity:"),
            "node ids are derived, not model-supplied, got {:?}",
            node.id
        );
        assert!(
            node.labels == ["person"] || node.labels == ["company"],
            "labels come from the schema table, got {:?}",
            node.labels
        );
        assert!(
            node.properties.contains_key("name"),
            "the identity column must be present on {:?}",
            node.id
        );
        for key in node.properties.keys() {
            assert!(
                key == "name" || (node.labels == ["person"] && key == "title"),
                "a strict table must not carry undeclared property {key:?}"
            );
        }
    }
    for edge in &snapshot.relationships {
        assert_eq!(edge.kind, "works_at");
        assert!(
            node_ids.contains(&edge.from.as_str()) && node_ids.contains(&edge.to.as_str()),
            "endpoints must resolve inside the delta, got {edge:?} against {node_ids:?}"
        );
        let from = snapshot
            .nodes
            .iter()
            .find(|n| n.id == edge.from)
            .expect("from endpoint");
        let to = snapshot
            .nodes
            .iter()
            .find(|n| n.id == edge.to)
            .expect("to endpoint");
        assert_eq!(from.labels, ["person"], "works_at starts at a person");
        assert_eq!(to.labels, ["company"], "works_at ends at a company");
    }

    // Re-validating the model's own candidate is deterministic.
    let candidate = model.candidate().expect("a candidate was produced");
    assert_eq!(
        extractor
            .validate_candidate(candidate)
            .expect("re-validation must succeed")
            .delta
            .snapshot(),
        snapshot,
        "validation of one candidate must be deterministic"
    );

    // The hand-off: nothing reached memory until the merge.
    assert!(
        memory
            .snapshot()
            .expect("graph lock must not be poisoned")
            .nodes
            .is_empty(),
        "live extraction must not mutate memory by itself"
    );
    let expected_nodes = report.nodes_accepted;
    let expected_edges = report.relationships_accepted;
    let merge = memory
        .merge_graph(report.delta, MergePolicy::Keep)
        .expect("merge must not fail");
    assert_eq!(merge.nodes_added, expected_nodes, "every delta node lands");
    assert_eq!(
        merge.relationships_added + merge.relationships_renamed,
        expected_edges,
        "no delta relationship is dropped"
    );
    assert_eq!(
        memory.snapshot().expect("graph lock must not be poisoned"),
        snapshot,
        "the merged graph must equal the delta on an empty memory"
    );

    // Diagnostic only: name grounding is model quality, never a contract.
    let grounded = snapshot.nodes.iter().filter(|node| {
        node.properties["name"]
            .as_str()
            .is_some_and(|name| SOURCE.contains(name))
    });
    eprintln!(
        "live extraction: {} nodes, {} relationships, {} name(s) verbatim from the source",
        snapshot.nodes.len(),
        snapshot.relationships.len(),
        grounded.count()
    );
}
