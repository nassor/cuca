//! Integration tests for the context-memory plugin (`plugin-memory`).
//!
//! The deterministic tests drive the public [`CucaPlugin`] hooks and
//! [`MemoryPlugin`] entry points directly with crafted requests; the live test
//! registers the plugin on a llama.cpp client and observes a real request.
#![cfg(all(feature = "provider-llamacpp", feature = "plugin-memory"))]

mod common;

use std::sync::{Arc, Mutex};

use cuca::plugin::CucaPlugin;
use cuca::types::{MessageContentBlock, MessageRole, UnifiedMessage};
use cuca::{
    CompactionStrategy, CompressionAction, ContextUsage, ContextUsageObserver, GraphContextConfig,
    GraphImportReport, GraphNode, GraphRelationship, GraphSnapshot, MemoryConfig, MemoryGraph,
    MemoryPlugin, MergePolicy, PluginError, UnifiedRequest,
};

/// An observer that records every usage reading it is handed, for assertions.
#[derive(Default)]
struct RecordingObserver {
    readings: Mutex<Vec<ContextUsage>>,
}

impl ContextUsageObserver for RecordingObserver {
    fn observe(&self, usage: &ContextUsage) -> Result<(), PluginError> {
        self.readings.lock().unwrap().push(*usage);
        Ok(())
    }
}

/// A config that triggers compression after two messages using the drop-only
/// tail of the menu (no extension seams needed).
fn compact_config() -> MemoryConfig {
    MemoryConfig {
        max_messages: Some(2),
        strategies: vec![CompactionStrategy::DropTurns],
        ..MemoryConfig::default()
    }
}

#[test]
fn count_tokens_is_positive_for_a_small_conversation() {
    let plugin = MemoryPlugin::new(compact_config()).expect("plugin must build");
    let messages = vec![
        UnifiedMessage::system("You are concise."),
        UnifiedMessage::user("Hello there, world!"),
    ];
    // Upper bound: BPE tokenization never produces more tokens than input
    // characters for ASCII text, so the count must stay well under the raw
    // character total across both messages.
    let total_chars: usize = messages
        .iter()
        .flat_map(|m| m.content.iter())
        .map(|b| match b {
            MessageContentBlock::Text(t) => t.chars().count(),
            _ => 0,
        })
        .sum();
    let tokens = plugin
        .count_tokens(&messages)
        .expect("counting must not fail");
    assert!(
        tokens > 0 && usize::try_from(tokens).unwrap() < total_chars,
        "expected a token count between 0 and {total_chars} chars, got {tokens}"
    );
}

#[test]
fn compress_drops_oldest_turns_and_reports_the_action() {
    let plugin = MemoryPlugin::new(compact_config()).expect("plugin must build");
    let mut messages = vec![
        UnifiedMessage::system("primary instruction"),
        UnifiedMessage::user("oldest user turn"),
        UnifiedMessage::assistant("assistant reply"),
        UnifiedMessage::user("recent user turn"),
        UnifiedMessage::assistant("latest reply"),
    ];
    let before = messages.len();
    let report = plugin
        .compress(&mut messages)
        .expect("compress must not fail");

    assert!(
        report.actions.contains(&CompressionAction::DroppedTurns),
        "expected DropTurns to act, got actions {:?}",
        report.actions
    );
    assert!(
        messages.len() < before,
        "message count must drop, was {before} now {}",
        messages.len()
    );
    // The never-remove invariants survive: the first system message and the
    // most recent user message must still be present.
    assert_eq!(messages[0].role, MessageRole::System);
    assert_eq!(
        messages.last().map(|m| &m.role),
        Some(&MessageRole::User),
        "the most recent user message must survive"
    );
}

#[test]
fn on_request_fires_observers_with_usage_readings() {
    let observer = Arc::new(RecordingObserver::default());
    let plugin = MemoryPlugin::new(MemoryConfig {
        observers: vec![observer.clone()],
        ..compact_config()
    })
    .expect("plugin must build");

    let mut req = UnifiedRequest::new("test-model").add_user_message("hi");
    plugin
        .on_request(&mut req)
        .expect("on_request must return Ok(())");
    plugin
        .on_request(&mut req)
        .expect("on_request must return Ok(())");

    let readings = observer.readings.lock().unwrap();
    assert_eq!(readings.len(), 2, "one usage reading per request");
    for usage in readings.iter() {
        assert!(usage.used_tokens > 0, "used tokens must be positive");
        assert!(usage.window_tokens > 0, "window tokens must be positive");
    }
}

#[test]
fn merge_graph_merges_public_graphs() {
    let mut first = MemoryGraph::new();
    first.upsert_node(GraphNode {
        id: "alice".into(),
        labels: vec!["person".into()],
        properties: serde_json::Map::new(),
    });
    first.upsert_node(GraphNode {
        id: "bob".into(),
        labels: Vec::new(),
        properties: serde_json::Map::new(),
    });
    first
        .add_relationship(GraphRelationship {
            id: "r".into(),
            from: "alice".into(),
            to: "bob".into(),
            kind: "knows".into(),
            weight: 1.0,
            properties: serde_json::Map::new(),
        })
        .expect("relationship endpoints exist in the first graph");

    let mut second = MemoryGraph::new();
    second.upsert_node(GraphNode {
        id: "carol".into(),
        labels: Vec::new(),
        properties: serde_json::Map::new(),
    });
    second.upsert_node(GraphNode {
        id: "dave".into(),
        labels: Vec::new(),
        properties: serde_json::Map::new(),
    });
    second
        .add_relationship(GraphRelationship {
            id: "r".into(),
            from: "carol".into(),
            to: "dave".into(),
            kind: "knows".into(),
            weight: 2.0,
            properties: serde_json::Map::new(),
        })
        .expect("relationship endpoints exist in the second graph");

    let plugin = MemoryPlugin::new(MemoryConfig::default()).expect("plugin must build");
    plugin
        .merge_graph(first, MergePolicy::Keep)
        .expect("first merge must not fail");
    let report = plugin
        .merge_graph(second, MergePolicy::Keep)
        .expect("merge must not fail");

    assert_eq!(report.nodes_added, 2, "both second-graph nodes are new");
    assert_eq!(
        report.relationships_renamed, 1,
        "the colliding relationship id 'r' must be renamed"
    );

    let graph = plugin.graph().expect("graph lock must not be poisoned");
    assert_eq!(graph.len(), 4, "nodes from both graphs survive the merge");
    assert_eq!(graph.relationship_count(), 2, "no relationship is dropped");
    let rendered = graph.render(8, 16);
    for id in ["alice", "bob", "carol", "dave"] {
        assert!(
            rendered.contains(&format!("node {id}:")),
            "render must list node '{id}', got:\n{rendered}"
        );
    }
}

#[test]
fn graph_accessor_exposes_live_state() {
    let plugin = MemoryPlugin::new(MemoryConfig::default()).expect("plugin must build");
    {
        let mut graph = plugin.graph().expect("graph lock must not be poisoned");
        graph.upsert_node(GraphNode {
            id: "alice".into(),
            labels: vec!["person".into()],
            properties: serde_json::Map::new(),
        });
        graph.upsert_node(GraphNode {
            id: "bob".into(),
            labels: Vec::new(),
            properties: serde_json::Map::new(),
        });
        graph
            .add_relationship(GraphRelationship {
                id: "alice-knows-bob".into(),
                from: "alice".into(),
                to: "bob".into(),
                kind: "knows".into(),
                weight: 1.0,
                properties: serde_json::Map::new(),
            })
            .expect("relationship endpoints exist");
    }

    let graph = plugin.graph().expect("graph lock must not be poisoned");
    assert_eq!(graph.len(), 2, "both upserted nodes are visible");
    assert_eq!(
        graph.relationship_count(),
        1,
        "the added relationship is visible"
    );
    let alice = graph.node("alice").expect("node lookup must hit");
    assert_eq!(alice.id, "alice");
    assert_eq!(alice.labels, vec!["person"]);
}

#[test]
fn graph_context_config_defaults() {
    let default = GraphContextConfig::default();
    assert_eq!(default.max_nodes, 64);
    assert_eq!(default.max_relationships, 128);
    assert_eq!(
        MemoryConfig::default().graph_context,
        None,
        "graph injection is opt-in"
    );
}

/// A node with no labels or properties.
fn plain_node(id: &str) -> GraphNode {
    GraphNode {
        id: id.into(),
        labels: Vec::new(),
        properties: serde_json::Map::new(),
    }
}

fn plain_rel(id: &str, from: &str, to: &str, weight: f64) -> GraphRelationship {
    GraphRelationship {
        id: id.into(),
        from: from.into(),
        to: to.into(),
        kind: "knows".into(),
        weight,
        properties: serde_json::Map::new(),
    }
}

/// A plugin whose graph holds `alice -[knows]-> bob` and an isolated `carol`.
fn plugin_with_sentinel_graph() -> MemoryPlugin {
    let plugin = MemoryPlugin::new(MemoryConfig::default()).expect("plugin must build");
    {
        let mut graph = plugin.graph().expect("graph lock must not be poisoned");
        graph.upsert_node(GraphNode {
            id: "alice".into(),
            labels: vec!["person".into()],
            properties: serde_json::Map::new(),
        });
        graph.upsert_node(plain_node("bob"));
        graph.upsert_node(plain_node("carol"));
        graph
            .add_relationship(plain_rel("r1", "alice", "bob", 1.5))
            .expect("sentinel endpoints exist");
    }
    plugin
}

#[test]
fn snapshot_exposes_deterministic_complete_state() {
    let plugin = plugin_with_sentinel_graph();
    let snapshot = plugin.snapshot().expect("graph lock must not be poisoned");

    assert_eq!(
        snapshot
            .nodes
            .iter()
            .map(|n| n.id.as_str())
            .collect::<Vec<_>>(),
        vec!["alice", "bob", "carol"],
        "nodes are complete and sorted by id"
    );
    assert_eq!(
        snapshot
            .relationships
            .iter()
            .map(|r| r.id.as_str())
            .collect::<Vec<_>>(),
        vec!["r1"]
    );
    assert_eq!(snapshot.nodes[0].labels, vec!["person"]);
    assert_eq!(snapshot.relationships[0].weight, 1.5);
    // Repeated exports of an unchanged graph are byte-identical.
    let again = plugin.snapshot().expect("graph lock must not be poisoned");
    assert_eq!(
        serde_json::to_string(&snapshot).unwrap(),
        serde_json::to_string(&again).unwrap()
    );
    // Exporting does not disturb the live graph.
    assert_eq!(plugin.graph().unwrap().len(), 3);
}

#[test]
fn replace_snapshot_replaces_live_graph_and_reports_counts() {
    let plugin = plugin_with_sentinel_graph();
    let replacement = GraphSnapshot {
        nodes: vec![plain_node("dave"), plain_node("erin")],
        relationships: vec![plain_rel("r9", "dave", "erin", -0.5)],
    };

    let report: GraphImportReport = plugin
        .replace_snapshot(replacement.clone())
        .expect("a valid snapshot must import");
    assert_eq!(report.nodes, 2);
    assert_eq!(report.relationships, 1);

    let graph = plugin.graph().expect("graph lock must not be poisoned");
    assert_eq!(graph.len(), 2, "import is a replacement, not a merge");
    assert_eq!(graph.relationship_count(), 1);
    for gone in ["alice", "bob", "carol"] {
        assert!(
            graph.node(gone).is_none(),
            "pre-import node '{gone}' must disappear"
        );
    }
    assert!(graph.node("dave").is_some());
    assert_eq!(graph.relationship("r9").unwrap().weight, -0.5);
    drop(graph);
    assert_eq!(
        plugin.snapshot().unwrap(),
        replacement,
        "the imported snapshot round-trips through the plugin"
    );
}

/// Every invalid snapshot is rejected before the live graph is touched: the
/// sentinel graph is byte-identical after each failure.
#[test]
fn replace_snapshot_rejects_invalid_snapshots_and_keeps_sentinel_state() {
    let plugin = plugin_with_sentinel_graph();
    let before = plugin.snapshot().expect("graph lock must not be poisoned");

    let invalid = vec![
        GraphSnapshot {
            nodes: vec![plain_node("dup"), plain_node("dup")],
            relationships: Vec::new(),
        },
        GraphSnapshot {
            nodes: vec![plain_node("x"), plain_node("y")],
            relationships: vec![
                plain_rel("same", "x", "y", 1.0),
                plain_rel("same", "y", "x", 1.0),
            ],
        },
        GraphSnapshot {
            nodes: vec![plain_node("x")],
            relationships: vec![plain_rel("r", "x", "ghost", 1.0)],
        },
        GraphSnapshot {
            nodes: vec![plain_node("x"), plain_node("y")],
            relationships: vec![plain_rel("r", "x", "y", f64::NAN)],
        },
    ];
    for snapshot in invalid {
        let err = plugin
            .replace_snapshot(snapshot)
            .expect_err("an invalid snapshot must be rejected");
        assert!(
            matches!(err, PluginError::Internal(_)),
            "expected an internal validation error, got {err:?}"
        );
        assert_eq!(
            plugin.snapshot().unwrap(),
            before,
            "the live graph must be untouched after a rejected import"
        );
    }

    let graph = plugin.graph().expect("graph lock must not be poisoned");
    assert_eq!(graph.len(), 3);
    assert_eq!(graph.relationship_count(), 1);
    assert!(graph.node("alice").is_some());
    assert!(graph.relationship("r1").is_some());
}

#[test]
fn replace_snapshot_accepts_an_empty_snapshot() {
    let plugin = plugin_with_sentinel_graph();
    let report = plugin
        .replace_snapshot(GraphSnapshot::default())
        .expect("an empty snapshot is a valid replacement");
    assert_eq!(report.nodes, 0);
    assert_eq!(report.relationships, 0);
    assert!(plugin.graph().unwrap().is_empty());
}

/// The graph seam and the snapshot seam agree: a snapshot taken after a merge
/// reflects the merged state, and merging still works after an import.
#[test]
fn snapshot_and_merge_seams_compose() {
    let plugin = plugin_with_sentinel_graph();
    plugin
        .replace_snapshot(GraphSnapshot {
            nodes: vec![plain_node("dave")],
            relationships: Vec::new(),
        })
        .expect("import must succeed");

    let mut incoming = MemoryGraph::new();
    incoming.upsert_node(plain_node("erin"));
    let report = plugin
        .merge_graph(incoming, MergePolicy::Keep)
        .expect("merge must not fail");
    assert_eq!(report.nodes_added, 1);

    let snapshot = plugin.snapshot().expect("graph lock must not be poisoned");
    assert_eq!(
        snapshot
            .nodes
            .iter()
            .map(|n| n.id.as_str())
            .collect::<Vec<_>>(),
        vec!["dave", "erin"]
    );
}

#[tokio::test]
async fn live_request_records_usage_and_yields_text() {
    if let Err(reason) = common::require_live_server() {
        eprintln!("SKIP: llama.cpp not reachable: {reason}");
        return;
    }
    let model = common::live_model();
    let observer = Arc::new(RecordingObserver::default());
    let plugin = MemoryPlugin::new(MemoryConfig {
        observers: vec![observer.clone()],
        ..compact_config()
    })
    .expect("plugin must build");
    let client = common::client_with_plugins(vec![Arc::new(plugin) as Arc<dyn CucaPlugin>]);

    // Several user messages push the request over the `max_messages: 2`
    // trigger. The final message is a directive so the small model finishes its
    // reasoning and emits a Text block (it otherwise spends its whole budget on
    // `Thinking` blocks); a slightly higher cap leaves room for that answer.
    let request = UnifiedRequest::new(model)
        .add_system_message("You are concise.")
        .add_user_message("First turn.")
        .add_user_message("Second turn.")
        .add_user_message("Reply with the single word: ok")
        .set_max_tokens(128);
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
    let readings = observer.readings.lock().unwrap();
    assert_eq!(
        readings.len(),
        1,
        "exactly one usage reading must be recorded for the single dispatched request, got {readings:?}"
    );
    let usage = readings[0];
    assert!(
        usage.used_tokens > 0,
        "used tokens must be positive, got {usage:?}"
    );
    assert!(
        usage.window_tokens > 0,
        "window tokens must be positive, got {usage:?}"
    );
    assert!(
        usage.used_tokens <= usage.window_tokens,
        "used tokens must not exceed the window, got {usage:?}"
    );
}
