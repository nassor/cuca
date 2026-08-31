//! In-memory graph memory for the CUCA memory plugin.
//!
//! [`MemoryGraph`] is a small, in-process graph store: nodes with labels and
//! properties, directed weighted relationships with properties, a bidirectional
//! adjacency index, and a deterministic whole-graph merge. It is the graph
//! representation behind the plugin's optional graph-context injection: agents
//! receive an explicit rendering of the current graph (nodes and
//! relationships) as context, so they can reason over explicit connections
//! rather than flat text.
//!
//! The design adapts the feature set of an embedded in-process graph database
//! reference (weighted directed edges, neighbor traversal, bounded BFS
//! traversal, subgraph extraction, upsert semantics, node/relationship
//! property maps, fast whole-graph merge) to CUCA's constraints: the graph
//! core uses only `std` collections (`HashMap`/`Vec`) plus `serde_json`, which
//! is already an unconditional core dependency. There is no disk, no external
//! store, and no new cargo feature.
//!
//! # Data structures
//!
//! [`MemoryGraph`] keeps six collections: a node store keyed by node id, a
//! relationship store keyed by relationship id, two adjacency indexes
//! (`outgoing`/`incoming`) mapping a node id to the ids of its incident
//! relationships, and two position indexes mapping relationship ids to their
//! slots in those adjacency vectors. All fields are private; mutation happens
//! only through the methods, which maintain the invariants below.
//!
//! # Invariants
//!
//! 1. **Endpoint validity**: every relationship's `from`/`to` names an
//!    existing node.
//! 2. **Relationship-id uniqueness**: no two relationships share an id; an
//!    insert collision is an upsert, and a merge collision is resolved by
//!    deterministic renaming; data is never dropped.
//! 3. **Adjacency completeness**: the adjacency and position indexes mirror
//!    the relationship store exactly.
//! 4. **Merge never loses data**: after a merge every incoming node and
//!    relationship is present, under its original or renamed id.
//! 5. **Determinism**: no observable behavior depends on `HashMap` iteration
//!    order: every query result is by-id or sorted, and [`MemoryGraph::render`]
//!    output is byte-deterministic for a given graph.
//!
//! # Complexity
//!
//! `upsert_node`/`node`/`relationship`/`add_relationship`/`remove_relationship`
//! are amortized O(1). `remove_node` is O(deg) for the incident relationships.
//! `neighbors` is O(deg log deg) (sort + dedup), `traverse` is O(V + E + S)
//! over the visited region, where S is the sum of per-level sorting costs;
//! `is_connected` is O(V + E) over the visited region. `subgraph` is O(V_c +
//! E_c + E + N log N + R log R), where V_c/E_c are the closure's visited
//! nodes/adjacencies, E is the full relationship store scanned for induced
//! edges, and N/R are selected nodes/relationships. `render` is O(V log(k + 1)
//! + E log(r + 1) + k log(k + 1) + r log(r + 1)), where k/r are the selected
//!   node/relationship counts (bounded by the limits, with full sorting when a
//!   limit is not smaller than its collection). `merge` is O(n + m + c log c)
//!   where c is the number of colliding relationship ids (see
//!   [`MemoryGraph::merge`]).
//!
//! # Example
//!
//! ```text
//! let mut graph = MemoryGraph::new();
//! graph.upsert_node(GraphNode { id: "alice".into(), labels: vec!["person".into()], properties: Map::new() });
//! graph.upsert_node(GraphNode { id: "bob".into(), labels: vec!["person".into()], properties: Map::new() });
//! graph.add_relationship(GraphRelationship {
//!     id: "r1".into(), from: "alice".into(), to: "bob".into(),
//!     kind: "knows".into(), weight: 1.0, properties: Map::new(),
//! }).unwrap();
//! let context = graph.render(16, 32); // deterministic LLM-readable listing
//! ```
//!
//! The plugin (see `crate::plugins::memory`) owns a `Mutex<MemoryGraph>` and
//! renders it into requests when configured.

use std::collections::{BinaryHeap, HashMap, HashSet};
// Formatting straight into the output `String` instead of through a temporary
// one. `write!` on a `String` cannot fail, so the `Result` is discarded rather
// than unwrapped (this crate never panics outside tests).
use std::fmt::Write as _;

use serde::{Deserialize, Serialize};

use crate::error::PluginError;

/// Prefix of the first line of [`MemoryGraph::render`] output. The memory
/// plugin uses it as the idempotency marker when injecting the rendered graph
/// into requests as a system message.
pub const GRAPH_RENDER_MARKER: &str = "CUCA graph memory:";

/// A node in the graph: an id, zero or more labels, and a property map.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GraphNode {
    /// Globally unique node id (within a graph).
    pub id: String,
    /// Categorical labels, e.g. `["person", "author"]`.
    pub labels: Vec<String>,
    /// Arbitrary key/value properties.
    pub properties: serde_json::Map<String, serde_json::Value>,
}

/// A directed, weighted relationship between two nodes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GraphRelationship {
    /// Globally unique relationship id (within a graph).
    pub id: String,
    /// Id of the source node; must exist when the relationship is added.
    pub from: String,
    /// Id of the target node; must exist when the relationship is added.
    pub to: String,
    /// Relationship type, e.g. `"knows"`.
    pub kind: String,
    /// Edge weight; unvalidated (NaN allowed), rendered via `Display`.
    pub weight: f64,
    /// Arbitrary key/value properties.
    pub properties: serde_json::Map<String, serde_json::Value>,
}

/// Direction used by neighbor and traversal queries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GraphDirection {
    /// Follow `from -> to` edges out of a node.
    Outgoing,
    /// Follow edges into a node (`to == node`).
    Incoming,
    /// Both directions.
    Any,
}

/// Node-id collision policy for [`MemoryGraph::merge`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MergePolicy {
    /// Existing node wins unchanged; the incoming node is dropped, but its
    /// incident relationships still merge (they attach to the kept node, whose
    /// id is the same).
    Keep,
    /// The incoming node replaces the existing one (labels + properties).
    Overwrite,
}

/// Outcome of [`MemoryGraph::merge`].
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct MergeReport {
    /// Incoming node ids that did not exist in self.
    pub nodes_added: usize,
    /// Incoming node ids that replaced an existing node (Overwrite policy).
    pub nodes_overwritten: usize,
    /// Incoming node ids that were dropped in favor of the existing node (Keep
    /// policy).
    pub nodes_kept: usize,
    /// Incoming relationships whose id did not collide.
    pub relationships_added: usize,
    /// Incoming relationships whose id collided and was deterministically
    /// renamed.
    pub relationships_renamed: usize,
}

/// A complete, deterministic, lossless snapshot of a [`MemoryGraph`].
///
/// Contains exactly the graph's nodes and relationships: no adjacency
/// vectors, position maps, capacities, traversal caches, or other derived
/// indexes. [`MemoryGraph::snapshot`] sorts nodes by [`GraphNode::id`] and
/// relationships by [`GraphRelationship::id`], so equivalent graphs built in
/// different insertion orders serialize to identical bytes.
/// [`MemoryGraph::from_snapshot`] validates and rebuilds a graph from one.
///
/// **Sensitive full-fidelity export:** `cuca-export` intentionally includes
/// the complete memory graph and local-cache request/response values. It may
/// contain confidential system prompts, user messages, tool arguments and
/// results, base64 image data, model output, signatures, and graph
/// properties. Treat the JSON as sensitive data; do not log or publish it.
/// CUCA does not encrypt, redact, or write it. The caller owns access
/// control, encryption, storage, and deletion.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GraphSnapshot {
    /// All nodes, sorted by [`GraphNode::id`].
    pub nodes: Vec<GraphNode>,
    /// All relationships, sorted by [`GraphRelationship::id`].
    pub relationships: Vec<GraphRelationship>,
}

/// The in-memory graph: node store, relationship store, and bidirectional
/// adjacency index.
///
/// All fields are private; mutation happens only through the methods, which
/// maintain the invariants in the [module docs](self). All public read paths
/// are deterministic (by-id lookups or sorted output).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct MemoryGraph {
    nodes: HashMap<String, GraphNode>,
    relationships: HashMap<String, GraphRelationship>,
    /// node id -> relationship ids with `from == node`.
    outgoing: HashMap<String, Vec<String>>,
    /// node id -> relationship ids with `to == node`.
    incoming: HashMap<String, Vec<String>>,
    /// relationship id -> position in its outgoing adjacency list.
    outgoing_positions: HashMap<String, usize>,
    /// relationship id -> position in its incoming adjacency list.
    incoming_positions: HashMap<String, usize>,
}

impl MemoryGraph {
    /// Empty graph; allocates no heap (all six collections have capacity 0).
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            relationships: HashMap::new(),
            outgoing: HashMap::new(),
            incoming: HashMap::new(),
            outgoing_positions: HashMap::new(),
            incoming_positions: HashMap::new(),
        }
    }

    /// Empty graph with reserved capacity for `nodes` nodes and
    /// `relationships` relationships (all six collections are reserved).
    pub fn with_capacity(nodes: usize, relationships: usize) -> Self {
        Self {
            nodes: HashMap::with_capacity(nodes),
            relationships: HashMap::with_capacity(relationships),
            outgoing: HashMap::with_capacity(nodes),
            incoming: HashMap::with_capacity(nodes),
            outgoing_positions: HashMap::with_capacity(relationships),
            incoming_positions: HashMap::with_capacity(relationships),
        }
    }

    /// Reserve capacity for more nodes and relationships (all six
    /// collections).
    pub fn reserve(&mut self, nodes: usize, relationships: usize) {
        self.nodes.reserve(nodes);
        self.relationships.reserve(relationships);
        self.outgoing.reserve(nodes);
        self.incoming.reserve(nodes);
        self.outgoing_positions.reserve(relationships);
        self.incoming_positions.reserve(relationships);
    }

    /// Insert `node`, or replace an existing node with the same id.
    ///
    /// Upsert semantics: a replacement overwrites labels AND properties
    /// wholesale. Returns `true` if the id was newly inserted, `false` if an
    /// existing node was replaced.
    pub fn upsert_node(&mut self, node: GraphNode) -> bool {
        self.nodes.insert(node.id.clone(), node).is_none()
    }

    /// Add `rel`, or replace an existing relationship with the same id.
    ///
    /// Upsert by id: a replaced relationship is re-indexed in the adjacency
    /// even when its `from`/`to` changed. Both endpoints must already exist as
    /// nodes. Add nodes before their relationships.
    ///
    /// # Errors
    ///
    /// Returns [`PluginError::Internal`] naming the missing endpoint id(s)
    /// when `rel.from` or `rel.to` is not a node in the graph.
    pub fn add_relationship(&mut self, rel: GraphRelationship) -> Result<(), PluginError> {
        if !self.nodes.contains_key(&rel.from) {
            return Err(PluginError::Internal(format!(
                "graph relationship '{}' references unknown node '{}' (from)",
                rel.id, rel.from
            )));
        }
        if !self.nodes.contains_key(&rel.to) {
            return Err(PluginError::Internal(format!(
                "graph relationship '{}' references unknown node '{}' (to)",
                rel.id, rel.to
            )));
        }
        let id = rel.id.clone();
        self.remove_relationship(&id);
        let from = rel.from.clone();
        let to = rel.to.clone();
        self.relationships.insert(id.clone(), rel);
        let outgoing = self.outgoing.entry(from).or_default();
        self.outgoing_positions.insert(id.clone(), outgoing.len());
        outgoing.push(id.clone());
        let incoming = self.incoming.entry(to).or_default();
        self.incoming_positions.insert(id.clone(), incoming.len());
        incoming.push(id);
        Ok(())
    }

    /// The node with `id`, if present.
    pub fn node(&self, id: &str) -> Option<&GraphNode> {
        self.nodes.get(id)
    }

    /// The relationship with `id`, if present.
    pub fn relationship(&self, id: &str) -> Option<&GraphRelationship> {
        self.relationships.get(id)
    }

    /// Iterate all nodes. Order is unspecified; use [`Self::render`] or
    /// [`Self::subgraph`] for deterministic output.
    pub fn nodes(&self) -> impl Iterator<Item = &GraphNode> {
        self.nodes.values()
    }

    /// Iterate all relationships. Order is unspecified.
    pub fn relationships(&self) -> impl Iterator<Item = &GraphRelationship> {
        self.relationships.values()
    }

    /// Number of nodes.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Number of relationships.
    pub fn relationship_count(&self) -> usize {
        self.relationships.len()
    }

    /// Whether the graph has no nodes (and therefore no relationships).
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Complete deterministic snapshot: every node and relationship, cloned
    /// from the stores and sorted by id.
    ///
    /// Emits no derived indexes or capacities, so equivalent graphs built in
    /// different insertion orders produce identical [`GraphSnapshot`] values
    /// (and identical canonical JSON). O(V log V + E log E).
    ///
    /// See [`GraphSnapshot`] for the sensitive-data warning.
    pub fn snapshot(&self) -> GraphSnapshot {
        let mut nodes: Vec<GraphNode> = self.nodes.values().cloned().collect();
        nodes.sort_unstable_by(|a, b| a.id.cmp(&b.id));
        let mut relationships: Vec<GraphRelationship> =
            self.relationships.values().cloned().collect();
        relationships.sort_unstable_by(|a, b| a.id.cmp(&b.id));
        GraphSnapshot {
            nodes,
            relationships,
        }
    }

    /// Build a graph from a complete [`GraphSnapshot`], validating before
    /// mutating anything.
    ///
    /// Import validation is deliberately stricter than [`Self::upsert_node`]
    /// and [`Self::add_relationship`], which upsert by id: a snapshot is a
    /// complete state, so duplicates are a defect rather than an implicit
    /// overwrite. All checks run before any insertion, and the reconstructed
    /// graph is returned only after all six collections are rebuilt through
    /// [`Self::add_relationship`], which rechecks the endpoint invariant.
    /// Nothing the caller owns, including a live graph awaiting replacement,
    /// is touched on any error.
    ///
    /// Import is a replacement, not a merge: relationship ids are preserved,
    /// so parallel edges stay separate and self-loops stay self-loops.
    ///
    /// # Errors
    ///
    /// Returns [`PluginError::Internal`] naming the offending id when the
    /// snapshot has a duplicate node id, a duplicate relationship id, a
    /// relationship whose `weight` is not finite, or a relationship whose
    /// `from`/`to` is absent from the snapshot's node ids.
    pub fn from_snapshot(snapshot: GraphSnapshot) -> Result<Self, PluginError> {
        let GraphSnapshot {
            nodes,
            relationships,
        } = snapshot;

        let mut node_ids: HashSet<&str> = HashSet::with_capacity(nodes.len());
        for node in &nodes {
            if !node_ids.insert(node.id.as_str()) {
                return Err(PluginError::Internal(format!(
                    "graph snapshot has duplicate node id '{}'",
                    node.id
                )));
            }
        }
        let mut rel_ids: HashSet<&str> = HashSet::with_capacity(relationships.len());
        for rel in &relationships {
            if !rel_ids.insert(rel.id.as_str()) {
                return Err(PluginError::Internal(format!(
                    "graph snapshot has duplicate relationship id '{}'",
                    rel.id
                )));
            }
            if !rel.weight.is_finite() {
                return Err(PluginError::Internal(format!(
                    "graph snapshot relationship '{}' has non-finite weight {}",
                    rel.id, rel.weight
                )));
            }
            if !node_ids.contains(rel.from.as_str()) {
                return Err(PluginError::Internal(format!(
                    "graph snapshot relationship '{}' references unknown node '{}' (from)",
                    rel.id, rel.from
                )));
            }
            if !node_ids.contains(rel.to.as_str()) {
                return Err(PluginError::Internal(format!(
                    "graph snapshot relationship '{}' references unknown node '{}' (to)",
                    rel.id, rel.to
                )));
            }
        }

        let mut graph = Self::with_capacity(nodes.len(), relationships.len());
        for node in nodes {
            graph.upsert_node(node);
        }
        for rel in relationships {
            graph.add_relationship(rel)?;
        }
        Ok(graph)
    }

    /// Remove `id` and cascade-remove every incident relationship (both
    /// directions; a self-loop is counted once). Returns `true` if the node
    /// existed.
    pub fn remove_node(&mut self, id: &str) -> bool {
        if self.nodes.remove(id).is_none() {
            return false;
        }
        let mut incident: HashSet<String> = HashSet::new();
        if let Some(list) = self.outgoing.get(id) {
            incident.extend(list.iter().cloned());
        }
        if let Some(list) = self.incoming.get(id) {
            incident.extend(list.iter().cloned());
        }
        for rid in incident {
            self.remove_relationship(&rid);
        }
        self.outgoing.remove(id);
        self.incoming.remove(id);
        true
    }

    /// Remove the relationship with `id` from the store and both adjacency
    /// lists. Returns `true` if it existed.
    pub fn remove_relationship(&mut self, id: &str) -> bool {
        let Some(rel) = self.relationships.remove(id) else {
            return false;
        };
        if let Some(list) = self.outgoing.get_mut(&rel.from) {
            let index = self
                .outgoing_positions
                .get(id)
                .copied()
                .filter(|&index| list.get(index).is_some_and(|rid| rid == id))
                .or_else(|| list.iter().position(|rid| rid == id));
            if let Some(index) = index {
                self.outgoing_positions.remove(id);
                let moved_id = list.swap_remove(index);
                if moved_id != id {
                    self.outgoing_positions.insert(moved_id, index);
                }
            }
        }
        if let Some(list) = self.incoming.get_mut(&rel.to) {
            let index = self
                .incoming_positions
                .get(id)
                .copied()
                .filter(|&index| list.get(index).is_some_and(|rid| rid == id))
                .or_else(|| list.iter().position(|rid| rid == id));
            if let Some(index) = index {
                self.incoming_positions.remove(id);
                let moved_id = list.swap_remove(index);
                if moved_id != id {
                    self.incoming_positions.insert(moved_id, index);
                }
            }
        }
        true
    }

    /// Neighbor node ids of `id` in `direction`, sorted by id and deduplicated
    /// (parallel relationships between the same pair collapse to one entry).
    /// A missing node yields an empty Vec.
    pub fn neighbors(&self, id: &str, direction: GraphDirection) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        match direction {
            GraphDirection::Outgoing => {
                if let Some(rels) = self.outgoing.get(id) {
                    for rid in rels {
                        if let Some(rel) = self.relationships.get(rid) {
                            out.push(rel.to.clone());
                        }
                    }
                }
            }
            GraphDirection::Incoming => {
                if let Some(rels) = self.incoming.get(id) {
                    for rid in rels {
                        if let Some(rel) = self.relationships.get(rid) {
                            out.push(rel.from.clone());
                        }
                    }
                }
            }
            GraphDirection::Any => {
                if let Some(rels) = self.outgoing.get(id) {
                    for rid in rels {
                        if let Some(rel) = self.relationships.get(rid) {
                            out.push(rel.to.clone());
                        }
                    }
                }
                if let Some(rels) = self.incoming.get(id) {
                    for rid in rels {
                        if let Some(rel) = self.relationships.get(rid) {
                            out.push(rel.from.clone());
                        }
                    }
                }
            }
        }
        out.sort_unstable();
        out.dedup();
        out
    }

    /// Directed reachability: is there a path from `from` to `to` following
    /// outgoing edges (BFS; cycles safe)? A node reaches itself; a missing
    /// endpoint yields `false`.
    pub fn is_connected(&self, from: &str, to: &str) -> bool {
        if from == to {
            return self.nodes.contains_key(from);
        }
        if !self.nodes.contains_key(from) || !self.nodes.contains_key(to) {
            return false;
        }
        let mut visited: HashSet<String> = HashSet::new();
        let mut frontier: Vec<String> = vec![from.to_string()];
        visited.insert(from.to_string());
        while !frontier.is_empty() {
            let mut next: Vec<String> = Vec::new();
            for node in &frontier {
                if let Some(rels) = self.outgoing.get(node) {
                    for rid in rels {
                        if let Some(rel) = self.relationships.get(rid) {
                            if rel.to == to {
                                return true;
                            }
                            if visited.insert(rel.to.clone()) {
                                next.push(rel.to.clone());
                            }
                        }
                    }
                }
            }
            frontier = next;
        }
        false
    }

    /// Bounded BFS from `start`, at most `max_depth` hops, in `direction`.
    ///
    /// Returns the visited node ids including `start`, in BFS order with each
    /// level sorted by id. `max_depth == 0` yields `[start]`. A missing start
    /// yields an empty Vec.
    pub fn traverse(
        &self,
        start: &str,
        max_depth: usize,
        direction: GraphDirection,
    ) -> Vec<String> {
        if !self.nodes.contains_key(start) {
            return Vec::new();
        }
        let mut visited: HashSet<String> = HashSet::new();
        let mut out: Vec<String> = vec![start.to_string()];
        visited.insert(start.to_string());
        let mut frontier: Vec<String> = vec![start.to_string()];
        for _ in 0..max_depth {
            let mut next: Vec<String> = Vec::new();
            for node in &frontier {
                let mut enqueue = |rels: Option<&Vec<String>>, incoming: bool| {
                    if let Some(rels) = rels {
                        for rid in rels {
                            if let Some(rel) = self.relationships.get(rid) {
                                let neighbor = if incoming { &rel.from } else { &rel.to };
                                if visited.insert(neighbor.clone()) {
                                    next.push(neighbor.clone());
                                }
                            }
                        }
                    }
                };
                match direction {
                    GraphDirection::Outgoing => enqueue(self.outgoing.get(node), false),
                    GraphDirection::Incoming => enqueue(self.incoming.get(node), true),
                    GraphDirection::Any => {
                        enqueue(self.outgoing.get(node), false);
                        enqueue(self.incoming.get(node), true);
                    }
                }
            }
            next.sort_unstable();
            out.extend(next.iter().cloned());
            frontier = next;
        }
        out
    }

    /// Extract the induced subgraph on the BFS closure of `roots` (missing
    /// roots are skipped).
    ///
    /// The result's nodes are every node within `max_depth` hops of any root
    /// (undirected, [`GraphDirection::Any`]); its relationships are every
    /// relationship whose `from` AND `to` are both in the node set, so
    /// cross-links not on the BFS tree are preserved. Relationship ids are
    /// preserved unchanged (ids are unique within the parent, hence within the
    /// subgraph). Clones nodes and relationships.
    pub fn subgraph(&self, roots: &[String], max_depth: usize) -> MemoryGraph {
        let mut ids: HashSet<String> = HashSet::new();
        let mut frontier: Vec<String> = Vec::new();
        for root in roots {
            if self.nodes.contains_key(root) && ids.insert(root.clone()) {
                frontier.push(root.clone());
            }
        }
        for _ in 0..max_depth {
            let mut next: Vec<String> = Vec::new();
            for node in &frontier {
                let mut enqueue = |rels: Option<&Vec<String>>, incoming: bool| {
                    if let Some(rels) = rels {
                        for rid in rels {
                            if let Some(rel) = self.relationships.get(rid) {
                                let neighbor = if incoming { &rel.from } else { &rel.to };
                                if ids.insert(neighbor.clone()) {
                                    next.push(neighbor.clone());
                                }
                            }
                        }
                    }
                };
                enqueue(self.outgoing.get(node), false);
                enqueue(self.incoming.get(node), true);
            }
            frontier = next;
        }
        let mut out = MemoryGraph::with_capacity(ids.len(), ids.len().saturating_mul(2));
        let mut sorted_ids: Vec<&String> = ids.iter().collect();
        sorted_ids.sort();
        for id in sorted_ids {
            if let Some(node) = self.nodes.get(id) {
                out.nodes.insert(id.clone(), node.clone());
            }
        }
        let mut rel_ids: Vec<&String> = self
            .relationships
            .keys()
            .filter(|rid| {
                let rel = &self.relationships[*rid];
                ids.contains(&rel.from) && ids.contains(&rel.to)
            })
            .collect();
        rel_ids.sort();
        for rid in rel_ids {
            if let Some(rel) = self.relationships.get(rid) {
                let id = rid.clone();
                let from = rel.from.clone();
                let to = rel.to.clone();
                out.relationships.insert(id.clone(), rel.clone());
                let outgoing = out.outgoing.entry(from).or_default();
                out.outgoing_positions.insert(id.clone(), outgoing.len());
                outgoing.push(id.clone());
                let incoming = out.incoming.entry(to).or_default();
                out.incoming_positions.insert(id.clone(), incoming.len());
                incoming.push(id);
            }
        }
        out
    }

    /// Merge `other` into self in place, consuming `other` (its nodes and
    /// relationships are moved, not cloned).
    ///
    /// Node ids are unioned per `policy`. Relationship ids are never dropped:
    /// an id that collides with an existing relationship is deterministically
    /// renamed to `{id}-2`, `{id}-3`, … such that it also avoids every
    /// incoming id that survives the merge. The report's counters sum to the
    /// incoming sizes (see [`MergeReport`]).
    ///
    /// # Complexity
    ///
    /// Two passes, one over the node set and one over the relationship set,
    /// each element moved exactly once: `O(n + m + c log c)` with `c` the
    /// number of colliding relationship ids (sorted for a deterministic rename
    /// mapping). On top of that, the rename probe set is built from every
    /// pre-merge relationship id in *both* graphs, so it costs
    /// `O(|self.relationships| + m)`: a term in the receiving graph's
    /// existing size, paid even when nothing collides. Capacity is
    /// pre-reserved, so no rehash occurs.
    pub fn merge(&mut self, other: MemoryGraph, policy: MergePolicy) -> MergeReport {
        let mut report = MergeReport::default();

        self.nodes.reserve(other.nodes.len());
        self.relationships.reserve(other.relationships.len());
        self.outgoing.reserve(other.outgoing.len());
        self.incoming.reserve(other.incoming.len());
        self.outgoing_positions.reserve(other.relationships.len());
        self.incoming_positions.reserve(other.relationships.len());

        // Pass 1: nodes union per policy, order-independent.
        for (id, node) in other.nodes {
            match self.nodes.entry(id) {
                std::collections::hash_map::Entry::Vacant(entry) => {
                    entry.insert(node);
                    report.nodes_added += 1;
                }
                std::collections::hash_map::Entry::Occupied(mut entry) => match policy {
                    MergePolicy::Keep => {
                        report.nodes_kept += 1;
                    }
                    MergePolicy::Overwrite => {
                        entry.insert(node);
                        report.nodes_overwritten += 1;
                    }
                },
            }
        }

        // Pass 2a: deterministic rename resolution for relationship-id
        // collisions. `taken` is seeded with EVERY pre-merge relationship id
        // from both graphs: a rename must also avoid every incoming id that
        // survives the merge, otherwise it could land on a surviving incoming
        // id and silently overwrite it (duplicate-id bug). Only the colliding
        // subset is sorted (typically empty), and collisions resolve against
        // `taken` + `assigned` in that canonical order, so the mapping is
        // deterministic regardless of HashMap iteration order.
        let mut collisions: Vec<&String> = other
            .relationships
            .keys()
            .filter(|k| self.relationships.contains_key(*k))
            .collect();
        collisions.sort();
        let mut taken: HashSet<&str> = self.relationships.keys().map(String::as_str).collect();
        taken.extend(other.relationships.keys().map(String::as_str));
        let mut assigned: HashSet<String> = HashSet::with_capacity(collisions.len());
        let mut rename: HashMap<String, String> = HashMap::with_capacity(collisions.len());
        for orig in collisions {
            let mut n: u32 = 1;
            let candidate = loop {
                n += 1;
                let candidate = format!("{orig}-{n}");
                if !taken.contains(candidate.as_str()) && !assigned.contains(&candidate) {
                    break candidate;
                }
            };
            assigned.insert(candidate.clone());
            rename.insert(orig.clone(), candidate);
        }

        // Pass 2b: relationships, moved in. Endpoints exist by invariant 1
        // plus the node union never dropping ids.
        for (orig_id, mut rel) in other.relationships {
            let final_id = match rename.get(&orig_id) {
                Some(new_id) => {
                    rel.id = new_id.clone();
                    report.relationships_renamed += 1;
                    rel.id.clone()
                }
                None => {
                    report.relationships_added += 1;
                    orig_id
                }
            };
            let from = rel.from.clone();
            let to = rel.to.clone();
            self.relationships.insert(final_id.clone(), rel);
            let outgoing = self.outgoing.entry(from).or_default();
            self.outgoing_positions
                .insert(final_id.clone(), outgoing.len());
            outgoing.push(final_id.clone());
            let incoming = self.incoming.entry(to).or_default();
            self.incoming_positions
                .insert(final_id.clone(), incoming.len());
            incoming.push(final_id);
        }

        report
    }

    /// Deterministic, compact, LLM-readable listing of the graph.
    ///
    /// Format: the first line is
    /// exactly `CUCA graph memory: {n} nodes, {m} relationships` (prefix
    /// [`GRAPH_RENDER_MARKER`]); then node lines sorted by id (up to
    /// `max_nodes`), each `node {id}:` with ` labels=[a, b]` and ` props={json}`
    /// when non-empty; then relationship lines sorted by id (up to
    /// `max_relationships`), each
    /// `rel {id}: {from} -[{kind}]-> {to} weight={w}` plus ` props={json}` when
    /// non-empty. Truncated categories end with an omission line
    /// (`... {n} more nodes omitted` / `... {n} more relationships omitted`).
    /// Property keys serialize in sorted order (`serde_json::Map` is
    /// `BTreeMap`-backed without `preserve_order`), so identical graphs render
    /// byte-identical output. Never fails.
    pub fn render(&self, max_nodes: usize, max_relationships: usize) -> String {
        let mut out = String::new();
        let _ = writeln!(
            out,
            "{GRAPH_RENDER_MARKER} {} nodes, {} relationships",
            self.nodes.len(),
            self.relationships.len()
        );
        let node_count = self.nodes.len();
        let node_ids = bounded_sorted_ids(self.nodes.keys(), max_nodes, node_count);
        for id in &node_ids {
            let node = &self.nodes[*id];
            out.push_str("node ");
            out.push_str(id);
            out.push(':');
            if !node.labels.is_empty() {
                out.push_str(" labels=[");
                for (i, label) in node.labels.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    out.push_str(label);
                }
                out.push(']');
            }
            if !node.properties.is_empty() {
                out.push_str(" props=");
                out.push_str(&props_json(&node.properties));
            }
            out.push('\n');
        }
        if node_count > max_nodes {
            let _ = writeln!(out, "... {} more nodes omitted", node_count - max_nodes);
        }
        let relationship_count = self.relationships.len();
        let rel_ids = bounded_sorted_ids(
            self.relationships.keys(),
            max_relationships,
            relationship_count,
        );
        for rid in &rel_ids {
            let rel = &self.relationships[*rid];
            out.push_str("rel ");
            out.push_str(rid);
            out.push_str(": ");
            out.push_str(&rel.from);
            out.push_str(" -[");
            out.push_str(&rel.kind);
            out.push_str("]-> ");
            out.push_str(&rel.to);
            out.push_str(" weight=");
            let _ = write!(out, "{}", rel.weight);
            if !rel.properties.is_empty() {
                out.push_str(" props=");
                out.push_str(&props_json(&rel.properties));
            }
            out.push('\n');
        }
        if relationship_count > max_relationships {
            let _ = writeln!(
                out,
                "... {} more relationships omitted",
                relationship_count - max_relationships
            );
        }
        out
    }
}

/// Serialize a property map for rendering.
///
/// Infallible in practice (`serde_json::Map<String, Value>` always
/// serializes); the fallback exists only to honor the no-panic rule.
fn props_json(props: &serde_json::Map<String, serde_json::Value>) -> String {
    serde_json::to_string(props).unwrap_or_else(|_| "{}".to_string())
}

fn bounded_sorted_ids<'a>(
    ids: impl Iterator<Item = &'a String>,
    max: usize,
    total: usize,
) -> Vec<&'a String> {
    if max >= total {
        let mut selected: Vec<&String> = ids.collect();
        selected.sort_unstable();
        return selected;
    }
    if max == 0 {
        return Vec::new();
    }
    let mut selected: BinaryHeap<&String> = BinaryHeap::with_capacity(max);
    for id in ids {
        if selected.len() < max {
            selected.push(id);
        } else if let Some(&largest) = selected.peek()
            && id < largest
        {
            selected.pop();
            selected.push(id);
        }
    }
    let mut selected = selected.into_vec();
    selected.sort_unstable();
    selected
}

#[cfg(all(test, feature = "plugin-memory"))]
mod tests {
    use super::*;
    use serde_json::json;

    fn node(id: &str) -> GraphNode {
        GraphNode {
            id: id.to_string(),
            labels: Vec::new(),
            properties: serde_json::Map::new(),
        }
    }

    fn rel(id: &str, from: &str, to: &str, kind: &str, weight: f64) -> GraphRelationship {
        GraphRelationship {
            id: id.to_string(),
            from: from.to_string(),
            to: to.to_string(),
            kind: kind.to_string(),
            weight,
            properties: serde_json::Map::new(),
        }
    }

    fn props(pairs: &[(&str, serde_json::Value)]) -> serde_json::Map<String, serde_json::Value> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect()
    }

    /// Property-style check of invariant 1: every relationship endpoint names
    /// an existing node.
    fn assert_endpoint_invariant(g: &MemoryGraph) {
        for r in g.relationships.values() {
            assert!(
                g.nodes.contains_key(&r.from),
                "relationship '{}' from '{}' has no node",
                r.id,
                r.from
            );
            assert!(
                g.nodes.contains_key(&r.to),
                "relationship '{}' to '{}' has no node",
                r.id,
                r.to
            );
        }
    }

    #[test]
    fn new_is_empty_with_zero_capacity() {
        let g = MemoryGraph::new();
        assert!(g.is_empty());
        assert_eq!(g.len(), 0);
        assert_eq!(g.relationship_count(), 0);
        assert_eq!(g.nodes.capacity(), 0);
        assert_eq!(g.relationships.capacity(), 0);
        assert_eq!(g.outgoing.capacity(), 0);
        assert_eq!(g.incoming.capacity(), 0);
    }

    #[test]
    fn with_capacity_reserves_collections() {
        let g = MemoryGraph::with_capacity(16, 32);
        assert!(g.nodes.capacity() >= 16);
        assert!(g.relationships.capacity() >= 32);
        assert!(g.outgoing.capacity() >= 16);
        assert!(g.incoming.capacity() >= 16);
        assert!(g.is_empty());
    }

    #[test]
    fn default_matches_new() {
        assert_eq!(MemoryGraph::default(), MemoryGraph::new());
        assert!(MemoryGraph::default().is_empty());
    }

    #[test]
    fn upsert_node_inserts_then_replaces() {
        let mut g = MemoryGraph::new();
        assert!(g.upsert_node(GraphNode {
            id: "a".into(),
            labels: vec!["person".into()],
            properties: props(&[("name", json!("Ada"))]),
        }));
        assert!(!g.upsert_node(GraphNode {
            id: "a".into(),
            labels: vec!["engineer".into()],
            properties: props(&[("age", json!(36))]),
        }));
        assert_eq!(g.len(), 1, "replacement keeps one node");
        let n = g.node("a").unwrap();
        assert_eq!(n.labels, vec!["engineer"]);
        assert_eq!(n.properties.get("age"), Some(&json!(36)));
        assert!(
            n.properties.get("name").is_none(),
            "properties are replaced wholesale"
        );
    }

    #[test]
    fn add_relationship_validates_endpoints() {
        let mut g = MemoryGraph::new();
        g.upsert_node(node("a"));
        g.upsert_node(node("b"));
        assert!(matches!(
            g.add_relationship(rel("r", "a", "zz", "k", 1.0)),
            Err(PluginError::Internal(_))
        ));
        assert!(matches!(
            g.add_relationship(rel("r", "zz", "b", "k", 1.0)),
            Err(PluginError::Internal(_))
        ));
        assert!(matches!(
            g.add_relationship(rel("r", "zz", "yy", "k", 1.0)),
            Err(PluginError::Internal(_))
        ));
        assert!(g.add_relationship(rel("r", "a", "b", "k", 1.0)).is_ok());
        assert!(g.relationship("r").is_some());
    }

    #[test]
    fn add_relationship_upserts_by_id_and_reindexes_adjacency() {
        let mut g = MemoryGraph::new();
        for id in ["a", "b", "c"] {
            g.upsert_node(node(id));
        }
        g.add_relationship(rel("r", "a", "b", "knows", 1.0))
            .unwrap();
        // Same id, different endpoints: adjacency must be rebuilt.
        g.add_relationship(rel("r", "a", "c", "knows", 2.0))
            .unwrap();
        assert_eq!(g.relationship_count(), 1, "upsert keeps one relationship");
        assert_eq!(
            g.neighbors("a", GraphDirection::Outgoing),
            vec!["c".to_string()]
        );
        assert!(g.neighbors("b", GraphDirection::Incoming).is_empty());
        assert_eq!(
            g.neighbors("c", GraphDirection::Incoming),
            vec!["a".to_string()]
        );
    }

    #[test]
    fn node_and_relationship_lookup() {
        let mut g = MemoryGraph::new();
        g.upsert_node(node("a"));
        g.upsert_node(node("b"));
        g.add_relationship(rel("r", "a", "b", "k", 1.0)).unwrap();
        assert!(g.node("a").is_some());
        assert!(g.node("zz").is_none());
        assert!(g.relationship("r").is_some());
        assert!(g.relationship("zz").is_none());
    }

    #[test]
    fn neighbors_respects_direction_and_dedups() {
        let mut g = MemoryGraph::new();
        for id in ["a", "b", "c"] {
            g.upsert_node(node(id));
        }
        // Two parallel a -> b edges: the neighbor b must appear once.
        g.add_relationship(rel("r1", "a", "b", "k", 1.0)).unwrap();
        g.add_relationship(rel("r2", "a", "b", "k", 2.0)).unwrap();
        g.add_relationship(rel("r3", "b", "c", "k", 1.0)).unwrap();
        assert_eq!(
            g.neighbors("a", GraphDirection::Outgoing),
            vec!["b".to_string()]
        );
        assert_eq!(
            g.neighbors("b", GraphDirection::Incoming),
            vec!["a".to_string()]
        );
        assert_eq!(
            g.neighbors("b", GraphDirection::Any),
            vec!["a".to_string(), "c".to_string()]
        );
        assert!(g.neighbors("zz", GraphDirection::Any).is_empty());
    }

    #[test]
    fn remove_relationship_cleans_adjacency() {
        let mut g = MemoryGraph::new();
        for id in ["a", "b", "c"] {
            g.upsert_node(node(id));
        }
        g.add_relationship(rel("r1", "a", "b", "k", 1.0)).unwrap();
        g.add_relationship(rel("r2", "b", "c", "k", 1.0)).unwrap();
        assert!(g.remove_relationship("r1"));
        assert!(!g.remove_relationship("zz"), "missing id returns false");
        assert!(g.relationship("r1").is_none());
        assert!(g.neighbors("a", GraphDirection::Outgoing).is_empty());
        assert!(g.neighbors("b", GraphDirection::Incoming).is_empty());
        assert_eq!(
            g.neighbors("b", GraphDirection::Outgoing),
            vec!["c".to_string()]
        );
        assert!(
            g.relationship("r2").is_some(),
            "unrelated relationship survives"
        );
    }

    #[test]
    fn remove_node_cascades_incident_relationships() {
        let mut g = MemoryGraph::new();
        for id in ["a", "b", "c"] {
            g.upsert_node(node(id));
        }
        g.add_relationship(rel("r1", "a", "b", "k", 1.0)).unwrap();
        g.add_relationship(rel("r2", "c", "a", "k", 1.0)).unwrap();
        g.add_relationship(rel("r3", "b", "b", "loop", 1.0))
            .unwrap();
        assert!(g.remove_node("b"));
        assert!(!g.remove_node("zz"), "missing id returns false");
        assert!(g.node("b").is_none());
        assert!(
            g.relationship("r1").is_none(),
            "outgoing relationship cascaded"
        );
        assert!(
            g.relationship("r3").is_none(),
            "self-loop cascaded exactly once"
        );
        assert!(
            g.relationship("r2").is_some(),
            "unrelated relationship survives"
        );
        assert!(g.neighbors("a", GraphDirection::Outgoing).is_empty());
        assert_eq!(
            g.neighbors("a", GraphDirection::Incoming),
            vec!["c".to_string()]
        );
        assert_eq!(
            g.neighbors("c", GraphDirection::Outgoing),
            vec!["a".to_string()]
        );
        assert_endpoint_invariant(&g);
    }

    #[test]
    fn traverse_bounds_depth_and_orders_levels() {
        let mut g = MemoryGraph::new();
        for id in ["a", "b", "c", "d"] {
            g.upsert_node(node(id));
        }
        g.add_relationship(rel("r1", "a", "b", "k", 1.0)).unwrap();
        g.add_relationship(rel("r2", "a", "c", "k", 1.0)).unwrap();
        g.add_relationship(rel("r3", "b", "d", "k", 1.0)).unwrap();
        assert_eq!(
            g.traverse("a", 0, GraphDirection::Outgoing),
            vec!["a".to_string()]
        );
        assert_eq!(
            g.traverse("a", 1, GraphDirection::Outgoing),
            vec!["a".to_string(), "b".to_string(), "c".to_string()]
        );
        assert_eq!(
            g.traverse("a", 2, GraphDirection::Outgoing),
            vec![
                "a".to_string(),
                "b".to_string(),
                "c".to_string(),
                "d".to_string()
            ]
        );
        assert!(g.traverse("zz", 2, GraphDirection::Outgoing).is_empty());
    }

    #[test]
    fn traverse_respects_direction() {
        let mut g = MemoryGraph::new();
        for id in ["a", "b", "c"] {
            g.upsert_node(node(id));
        }
        g.add_relationship(rel("r1", "a", "b", "k", 1.0)).unwrap();
        g.add_relationship(rel("r2", "c", "a", "k", 1.0)).unwrap();
        assert_eq!(
            g.traverse("a", 1, GraphDirection::Outgoing),
            vec!["a".to_string(), "b".to_string()]
        );
        assert_eq!(
            g.traverse("a", 1, GraphDirection::Incoming),
            vec!["a".to_string(), "c".to_string()]
        );
        assert_eq!(
            g.traverse("a", 1, GraphDirection::Any),
            vec!["a".to_string(), "b".to_string(), "c".to_string()]
        );
    }

    #[test]
    fn traverse_handles_cycles() {
        let mut g = MemoryGraph::new();
        for id in ["a", "b", "c"] {
            g.upsert_node(node(id));
        }
        g.add_relationship(rel("r1", "a", "b", "k", 1.0)).unwrap();
        g.add_relationship(rel("r2", "b", "a", "k", 1.0)).unwrap();
        g.add_relationship(rel("r3", "b", "c", "k", 1.0)).unwrap();
        assert_eq!(
            g.traverse("a", 5, GraphDirection::Outgoing),
            vec!["a".to_string(), "b".to_string(), "c".to_string()],
            "a cycle terminates and visits each node once"
        );
    }

    #[test]
    fn is_connected_reports_directed_reachability() {
        let mut g = MemoryGraph::new();
        for id in ["a", "b", "c"] {
            g.upsert_node(node(id));
        }
        g.add_relationship(rel("r1", "a", "b", "k", 1.0)).unwrap();
        g.add_relationship(rel("r2", "b", "c", "k", 1.0)).unwrap();
        assert!(g.is_connected("a", "c"), "multi-hop reachability");
        assert!(g.is_connected("a", "b"));
        assert!(!g.is_connected("c", "a"), "reachability is directed");
        assert!(g.is_connected("a", "a"), "a node reaches itself");
        assert!(!g.is_connected("zz", "a"));
        assert!(!g.is_connected("a", "zz"));
    }

    #[test]
    fn subgraph_extracts_induced_neighborhood() {
        let mut g = MemoryGraph::new();
        for id in ["a", "b", "c", "d"] {
            g.upsert_node(node(id));
        }
        g.add_relationship(rel("r1", "a", "b", "k", 1.0)).unwrap();
        g.add_relationship(rel("r2", "b", "c", "k", 1.0)).unwrap();
        g.add_relationship(rel("r3", "c", "d", "k", 1.0)).unwrap();
        g.add_relationship(rel("r4", "a", "c", "cross", 1.0))
            .unwrap();
        let sg = g.subgraph(&["a".to_string()], 1);
        let mut sg_nodes: Vec<&String> = sg.nodes.keys().collect();
        sg_nodes.sort();
        assert_eq!(
            sg_nodes,
            vec!["a", "b", "c"],
            "closure at depth 1 excludes d"
        );
        let mut sg_rels: Vec<&String> = sg.relationships.keys().collect();
        sg_rels.sort();
        assert_eq!(
            sg_rels,
            vec!["r1", "r2", "r4"],
            "cross-link r4 is preserved; r3 leaves the closure and is excluded"
        );
        assert_endpoint_invariant(&sg);
        // Depth 2 reaches d through a -> c -> d, pulling r3 in.
        let sg2 = g.subgraph(&["a".to_string()], 2);
        let mut sg2_nodes: Vec<&String> = sg2.nodes.keys().collect();
        sg2_nodes.sort();
        assert_eq!(sg2_nodes, vec!["a", "b", "c", "d"]);
        assert_eq!(sg2.relationship_count(), 4);
        assert_endpoint_invariant(&sg2);
        assert!(
            g.subgraph(&[], 5).is_empty(),
            "no roots yield an empty subgraph"
        );
    }

    #[test]
    fn subgraph_skips_missing_roots() {
        let mut g = MemoryGraph::new();
        g.upsert_node(node("a"));
        g.upsert_node(node("b"));
        g.add_relationship(rel("r1", "a", "b", "k", 1.0)).unwrap();
        let sg = g.subgraph(&["zz".to_string(), "a".to_string()], 1);
        let mut sg_nodes: Vec<&String> = sg.nodes.keys().collect();
        sg_nodes.sort();
        assert_eq!(sg_nodes, vec!["a", "b"], "the missing root is skipped");
    }

    #[test]
    fn merge_unions_disjoint_graphs() {
        let mut g1 = MemoryGraph::new();
        g1.upsert_node(node("a"));
        g1.upsert_node(node("b"));
        g1.add_relationship(rel("r1", "a", "b", "k", 1.0)).unwrap();
        let mut g2 = MemoryGraph::new();
        g2.upsert_node(node("c"));
        g2.upsert_node(node("d"));
        g2.add_relationship(rel("r2", "c", "d", "k", 1.0)).unwrap();
        let report = g1.merge(g2, MergePolicy::Keep);
        assert_eq!(report.nodes_added, 2);
        assert_eq!(report.nodes_overwritten, 0);
        assert_eq!(report.nodes_kept, 0);
        assert_eq!(report.relationships_added, 1);
        assert_eq!(report.relationships_renamed, 0);
        assert_eq!(g1.len(), 4);
        assert_eq!(g1.relationship_count(), 2);
        assert!(g1.node("a").is_some() && g1.node("c").is_some());
        assert!(g1.relationship("r1").is_some() && g1.relationship("r2").is_some());
        assert_endpoint_invariant(&g1);
    }

    #[test]
    fn merge_keep_policy_preserves_existing_node() {
        let mut g1 = MemoryGraph::new();
        g1.upsert_node(GraphNode {
            id: "a".into(),
            labels: vec!["original".into()],
            properties: props(&[("name", json!("Ada"))]),
        });
        let mut g2 = MemoryGraph::new();
        g2.upsert_node(GraphNode {
            id: "a".into(),
            labels: vec!["incoming".into()],
            properties: props(&[("name", json!("Grace"))]),
        });
        g2.upsert_node(node("b"));
        g2.add_relationship(rel("r1", "a", "b", "k", 1.0)).unwrap();
        let report = g1.merge(g2, MergePolicy::Keep);
        assert_eq!(report.nodes_kept, 1);
        let a = g1.node("a").unwrap();
        assert_eq!(
            a.labels,
            vec!["original"],
            "Keep leaves the existing node untouched"
        );
        assert_eq!(a.properties.get("name"), Some(&json!("Ada")));
        assert!(
            g1.relationship("r1").is_some(),
            "incoming relationships still merge"
        );
        assert!(g1.node("b").is_some());
    }

    #[test]
    fn merge_overwrite_policy_replaces_node() {
        let mut g1 = MemoryGraph::new();
        g1.upsert_node(GraphNode {
            id: "a".into(),
            labels: vec!["original".into()],
            properties: props(&[("name", json!("Ada"))]),
        });
        let mut g2 = MemoryGraph::new();
        g2.upsert_node(GraphNode {
            id: "a".into(),
            labels: vec!["incoming".into()],
            properties: props(&[("name", json!("Grace"))]),
        });
        let report = g1.merge(g2, MergePolicy::Overwrite);
        assert_eq!(report.nodes_overwritten, 1);
        let a = g1.node("a").unwrap();
        assert_eq!(
            a.labels,
            vec!["incoming"],
            "Overwrite replaces labels and properties"
        );
        assert_eq!(a.properties.get("name"), Some(&json!("Grace")));
    }

    #[test]
    fn merge_renames_relationship_id_collisions_deterministically() {
        // Simple collision: incoming `r` becomes `r-2`; the original survives.
        let mut g1 = MemoryGraph::new();
        g1.upsert_node(node("a"));
        g1.upsert_node(node("b"));
        g1.add_relationship(rel("r", "a", "b", "k", 1.0)).unwrap();
        let mut g2 = MemoryGraph::new();
        g2.upsert_node(node("a"));
        g2.upsert_node(node("b"));
        g2.add_relationship(rel("r", "a", "b", "k", 2.0)).unwrap();
        let report = g1.merge(g2, MergePolicy::Keep);
        assert_eq!(report.relationships_renamed, 1);
        assert_eq!(report.relationships_added, 0);
        assert_eq!(g1.relationship_count(), 2);
        assert_eq!(
            g1.relationship("r").unwrap().weight,
            1.0,
            "original kept under its id"
        );
        assert_eq!(
            g1.relationship("r-2").unwrap().weight,
            2.0,
            "incoming renamed"
        );

        // Chain: self already has `r-2`, so incoming `r` becomes `r-3`.
        let mut g1 = MemoryGraph::new();
        g1.upsert_node(node("a"));
        g1.upsert_node(node("b"));
        g1.add_relationship(rel("r", "a", "b", "k", 1.0)).unwrap();
        g1.add_relationship(rel("r-2", "a", "b", "k", 1.5)).unwrap();
        let mut g2 = MemoryGraph::new();
        g2.upsert_node(node("a"));
        g2.upsert_node(node("b"));
        g2.add_relationship(rel("r", "a", "b", "k", 2.0)).unwrap();
        let report = g1.merge(g2, MergePolicy::Keep);
        assert_eq!(report.relationships_renamed, 1);
        assert_eq!(g1.relationship("r").unwrap().weight, 1.0);
        assert_eq!(g1.relationship("r-2").unwrap().weight, 1.5);
        assert_eq!(g1.relationship("r-3").unwrap().weight, 2.0);
        assert_eq!(g1.relationship_count(), 3);
    }

    #[test]
    fn merge_rename_collision_between_incoming_relationships() {
        // Both incoming `x` and `x-2` collide: sorted-origin processing fixes
        // the mapping: incoming `x` -> `x-3`, incoming `x-2` -> `x-2-2`.
        let mut g1 = MemoryGraph::new();
        g1.upsert_node(node("a"));
        g1.upsert_node(node("b"));
        g1.add_relationship(rel("x", "a", "b", "k", 1.0)).unwrap();
        g1.add_relationship(rel("x-2", "a", "b", "k", 1.5)).unwrap();
        let mut g2 = MemoryGraph::new();
        g2.upsert_node(node("a"));
        g2.upsert_node(node("b"));
        g2.add_relationship(rel("x", "a", "b", "k", 2.0)).unwrap();
        g2.add_relationship(rel("x-2", "a", "b", "k", 2.5)).unwrap();
        let report = g1.merge(g2, MergePolicy::Keep);
        assert_eq!(report.relationships_renamed, 2);
        assert_eq!(report.relationships_added, 0);
        assert_eq!(g1.relationship_count(), 4);
        assert_eq!(g1.relationship("x").unwrap().weight, 1.0);
        assert_eq!(g1.relationship("x-2").unwrap().weight, 1.5);
        assert_eq!(
            g1.relationship("x-3").unwrap().weight,
            2.0,
            "incoming x renamed to x-3"
        );
        assert_eq!(
            g1.relationship("x-2-2").unwrap().weight,
            2.5,
            "incoming x-2 renamed to x-2-2 (the suffix appends to the full original id)"
        );
    }

    #[test]
    fn merge_rename_avoids_surviving_incoming_ids() {
        // Duplicate-id bug case: self has `x`; incoming has `x` (collides) and
        // `x-2` (survives). The rename must not land on the surviving `x-2`.
        let mut g1 = MemoryGraph::new();
        g1.upsert_node(node("a"));
        g1.upsert_node(node("b"));
        g1.add_relationship(rel("x", "a", "b", "k", 1.0)).unwrap();
        let mut g2 = MemoryGraph::new();
        g2.upsert_node(node("a"));
        g2.upsert_node(node("b"));
        g2.add_relationship(rel("x", "a", "b", "k", 2.0)).unwrap();
        g2.add_relationship(rel("x-2", "a", "b", "k", 2.5)).unwrap();
        let report = g1.merge(g2, MergePolicy::Keep);
        assert_eq!(report.relationships_renamed, 1);
        assert_eq!(report.relationships_added, 1);
        assert_eq!(
            g1.relationship_count(),
            3,
            "no relationship is silently lost"
        );
        assert_eq!(g1.relationship("x").unwrap().weight, 1.0);
        assert_eq!(
            g1.relationship("x-2").unwrap().weight,
            2.5,
            "surviving incoming id untouched"
        );
        assert_eq!(
            g1.relationship("x-3").unwrap().weight,
            2.0,
            "renamed away from x-2"
        );
    }

    #[test]
    fn merge_never_loses_relationships() {
        let mut g1 = MemoryGraph::new();
        g1.upsert_node(node("a"));
        g1.upsert_node(node("b"));
        g1.add_relationship(rel("r1", "a", "b", "k", 1.0)).unwrap();
        g1.add_relationship(rel("r2", "b", "a", "k", 1.0)).unwrap();
        let mut g2 = MemoryGraph::new();
        g2.upsert_node(node("a"));
        g2.upsert_node(node("b"));
        g2.add_relationship(rel("r2", "b", "a", "k", 2.0)).unwrap();
        g2.add_relationship(rel("r3", "a", "b", "k", 3.0)).unwrap();
        g2.add_relationship(rel("r4", "b", "a", "k", 4.0)).unwrap();
        let before = g1.relationship_count();
        let other_count = 3;
        g1.merge(g2, MergePolicy::Keep);
        assert_eq!(
            g1.relationship_count(),
            before + other_count,
            "merge never loses relationships"
        );
        assert!(g1.relationship("r1").is_some());
        assert!(g1.relationship("r2").is_some(), "self's r2 kept");
        assert!(g1.relationship("r2-2").is_some(), "incoming r2 renamed");
        assert!(g1.relationship("r3").is_some());
        assert!(g1.relationship("r4").is_some());
        assert_endpoint_invariant(&g1);
    }

    #[test]
    fn merge_report_counts_are_exact() {
        let mut g1 = MemoryGraph::new();
        for id in ["a", "b", "c"] {
            g1.upsert_node(node(id));
        }
        g1.add_relationship(rel("r1", "a", "b", "k", 1.0)).unwrap();
        g1.add_relationship(rel("r2", "b", "c", "k", 1.0)).unwrap();
        let mut g2 = MemoryGraph::new();
        g2.upsert_node(node("a"));
        g2.upsert_node(node("b"));
        g2.upsert_node(node("d"));
        g2.add_relationship(rel("r2", "b", "d", "k", 2.0)).unwrap();
        g2.add_relationship(rel("r3", "d", "a", "k", 2.0)).unwrap();
        let report = g1.merge(g2, MergePolicy::Keep);
        assert_eq!(report.nodes_added, 1, "d added");
        assert_eq!(report.nodes_kept, 2, "a and b kept");
        assert_eq!(report.nodes_overwritten, 0);
        assert_eq!(report.relationships_added, 1, "r3 added");
        assert_eq!(report.relationships_renamed, 1, "r2 renamed");
        assert_eq!(
            report.nodes_added + report.nodes_overwritten + report.nodes_kept,
            3
        );
        assert_eq!(report.relationships_added + report.relationships_renamed, 2);
        assert_endpoint_invariant(&g1);
    }

    #[test]
    fn merge_preserves_endpoint_invariant() {
        let mut g1 = MemoryGraph::new();
        g1.upsert_node(node("a"));
        g1.upsert_node(node("b"));
        g1.add_relationship(rel("r1", "a", "b", "k", 1.0)).unwrap();
        let mut g2 = MemoryGraph::new();
        g2.upsert_node(node("b"));
        g2.upsert_node(node("c"));
        g2.add_relationship(rel("r2", "b", "c", "k", 2.0)).unwrap();
        g1.merge(g2, MergePolicy::Keep);
        assert_endpoint_invariant(&g1);
        let mut g3 = MemoryGraph::new();
        g3.upsert_node(node("c"));
        g3.upsert_node(node("d"));
        g3.add_relationship(rel("r3", "c", "d", "k", 3.0)).unwrap();
        g1.merge(g3, MergePolicy::Overwrite);
        assert_endpoint_invariant(&g1);
    }

    #[test]
    fn merge_deterministic_across_input_history() {
        // Identical content built in different insertion orders must merge to
        // equal graphs with equal reports.
        let mut g1a = MemoryGraph::new();
        g1a.upsert_node(node("a"));
        g1a.upsert_node(node("b"));
        g1a.upsert_node(node("c"));
        g1a.add_relationship(rel("x", "a", "b", "k", 1.0)).unwrap();
        g1a.add_relationship(rel("y", "b", "c", "k", 1.0)).unwrap();
        let mut g1b = MemoryGraph::new();
        g1b.upsert_node(node("c"));
        g1b.upsert_node(node("a"));
        g1b.upsert_node(node("b"));
        g1b.add_relationship(rel("y", "b", "c", "k", 1.0)).unwrap();
        g1b.add_relationship(rel("x", "a", "b", "k", 1.0)).unwrap();

        let mut g2a = MemoryGraph::new();
        g2a.upsert_node(node("d"));
        g2a.upsert_node(node("e"));
        g2a.add_relationship(rel("z", "d", "e", "k", 2.0)).unwrap();
        g2a.add_relationship(rel("w", "e", "d", "k", 2.0)).unwrap();
        let mut g2b = MemoryGraph::new();
        g2b.upsert_node(node("e"));
        g2b.upsert_node(node("d"));
        g2b.add_relationship(rel("w", "e", "d", "k", 2.0)).unwrap();
        g2b.add_relationship(rel("z", "d", "e", "k", 2.0)).unwrap();

        let report_a = g1a.merge(g2a, MergePolicy::Keep);
        let report_b = g1b.merge(g2b, MergePolicy::Keep);
        assert_eq!(report_a, report_b);
        assert_eq!(g1a, g1b, "merge result is independent of insertion order");
    }

    #[test]
    fn render_is_deterministic_and_sorted() {
        let mut g = MemoryGraph::new();
        g.upsert_node(GraphNode {
            id: "b".into(),
            labels: vec!["person".into()],
            properties: props(&[("name", json!("Bob"))]),
        });
        g.upsert_node(node("a"));
        g.add_relationship(rel("r1", "b", "a", "works_on", 1.0))
            .unwrap();
        g.add_relationship(GraphRelationship {
            id: "r2".into(),
            from: "a".into(),
            to: "b".into(),
            kind: "knows".into(),
            weight: 0.5,
            properties: props(&[("since", json!(2020))]),
        })
        .unwrap();
        let expected = "CUCA graph memory: 2 nodes, 2 relationships\n\
                        node a:\n\
                        node b: labels=[person] props={\"name\":\"Bob\"}\n\
                        rel r1: b -[works_on]-> a weight=1\n\
                        rel r2: a -[knows]-> b weight=0.5 props={\"since\":2020}\n";
        let rendered = g.render(16, 32);
        assert_eq!(
            rendered, expected,
            "exact byte format per MemoryGraph::render"
        );
        assert_eq!(g.render(16, 32), rendered, "byte-identical across calls");
    }

    #[test]
    fn render_truncates_to_maxes_with_omission_lines() {
        let mut g = MemoryGraph::new();
        g.upsert_node(GraphNode {
            id: "b".into(),
            labels: vec!["person".into()],
            properties: props(&[("name", json!("Bob"))]),
        });
        g.upsert_node(node("a"));
        g.add_relationship(rel("r1", "b", "a", "works_on", 1.0))
            .unwrap();
        g.add_relationship(rel("r2", "a", "b", "knows", 0.5))
            .unwrap();
        let truncated = "CUCA graph memory: 2 nodes, 2 relationships\n\
                         node a:\n\
                         ... 1 more nodes omitted\n\
                         rel r1: b -[works_on]-> a weight=1\n\
                         ... 1 more relationships omitted\n";
        assert_eq!(g.render(1, 1), truncated);
        let zeroed = "CUCA graph memory: 2 nodes, 2 relationships\n\
                      ... 2 more nodes omitted\n\
                      ... 2 more relationships omitted\n";
        assert_eq!(g.render(0, 0), zeroed);
    }

    #[test]
    fn serde_round_trip_wire_types() {
        let node = GraphNode {
            id: "a".into(),
            labels: vec!["person".into()],
            properties: props(&[("name", json!("Ada"))]),
        };
        let node_json = serde_json::to_string(&node).unwrap();
        assert_eq!(
            node_json, r#"{"id":"a","labels":["person"],"properties":{"name":"Ada"}}"#,
            "stable JSON form"
        );
        let back: GraphNode = serde_json::from_str(&node_json).unwrap();
        assert_eq!(back, node);

        let rel = GraphRelationship {
            id: "r".into(),
            from: "a".into(),
            to: "b".into(),
            kind: "knows".into(),
            weight: 0.5,
            properties: props(&[("since", json!(2020))]),
        };
        let rel_json = serde_json::to_string(&rel).unwrap();
        assert_eq!(
            rel_json,
            r#"{"id":"r","from":"a","to":"b","kind":"knows","weight":0.5,"properties":{"since":2020}}"#
        );
        let back: GraphRelationship = serde_json::from_str(&rel_json).unwrap();
        assert_eq!(back, rel);

        assert_eq!(
            serde_json::to_string(&GraphDirection::Outgoing).unwrap(),
            r#""outgoing""#
        );
        let dir: GraphDirection = serde_json::from_str(r#""any""#).unwrap();
        assert_eq!(dir, GraphDirection::Any);

        assert_eq!(
            serde_json::to_string(&MergePolicy::Keep).unwrap(),
            r#""keep""#
        );
        let policy: MergePolicy = serde_json::from_str(r#""overwrite""#).unwrap();
        assert_eq!(policy, MergePolicy::Overwrite);

        let report = MergeReport {
            nodes_added: 2,
            nodes_overwritten: 0,
            nodes_kept: 1,
            relationships_added: 3,
            relationships_renamed: 1,
        };
        let report_json = serde_json::to_string(&report).unwrap();
        let back: MergeReport = serde_json::from_str(&report_json).unwrap();
        assert_eq!(back, report);
    }

    #[test]
    fn graph_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<MemoryGraph>();
    }

    /// Equivalent graphs built in different insertion orders must serialize to
    /// identical canonical JSON: nodes and relationships sorted by id, and no
    /// derived index fields.
    #[test]
    fn snapshot_is_deterministic_across_insertion_orders() {
        let labeled = |id: &str, labels: &[&str], p: &[(&str, serde_json::Value)]| GraphNode {
            id: id.to_string(),
            labels: labels.iter().map(|l| l.to_string()).collect(),
            properties: props(p),
        };
        // Note the label order below is deliberately *not* sorted: label order
        // is part of the node value and must survive the snapshot unchanged.
        let nodes = vec![
            labeled(
                "alice",
                &["person", "author"],
                &[("meta", json!({"z": [1, 2, {"deep": true}], "a": null}))],
            ),
            labeled("bob", &["person"], &[("age", json!(41))]),
            labeled("carol", &[], &[]),
        ];
        let rels = vec![
            rel("r1", "alice", "bob", "knows", 1.5),
            rel("r2", "bob", "alice", "knows", -0.25),
            rel("r3", "carol", "carol", "self", 0.0),
        ];

        let mut forward = MemoryGraph::new();
        for n in &nodes {
            forward.upsert_node(n.clone());
        }
        for r in &rels {
            forward.add_relationship(r.clone()).unwrap();
        }

        let mut reverse = MemoryGraph::new();
        for n in nodes.iter().rev() {
            reverse.upsert_node(n.clone());
        }
        for r in rels.iter().rev() {
            reverse.add_relationship(r.clone()).unwrap();
        }

        let a = serde_json::to_string(&forward.snapshot()).unwrap();
        let b = serde_json::to_string(&reverse.snapshot()).unwrap();
        assert_eq!(a, b, "snapshot JSON must not depend on insertion order");

        let value: serde_json::Value = serde_json::from_str(&a).unwrap();
        let obj = value.as_object().unwrap();
        let mut keys: Vec<&String> = obj.keys().collect();
        keys.sort_unstable();
        assert_eq!(
            keys,
            vec!["nodes", "relationships"],
            "snapshot must expose exactly two fields and no derived indexes"
        );

        let snapshot = forward.snapshot();
        assert_eq!(
            snapshot
                .nodes
                .iter()
                .map(|n| n.id.as_str())
                .collect::<Vec<_>>(),
            vec!["alice", "bob", "carol"]
        );
        assert_eq!(
            snapshot
                .relationships
                .iter()
                .map(|r| r.id.as_str())
                .collect::<Vec<_>>(),
            vec!["r1", "r2", "r3"]
        );
        // Values are the existing serde values, labels in their original order.
        assert_eq!(snapshot.nodes[0], nodes[0]);
        assert_eq!(snapshot.nodes[0].labels, vec!["person", "author"]);
        assert_eq!(snapshot.relationships[2], rels[2]);
    }

    #[test]
    fn snapshot_of_empty_graph_has_empty_collections() {
        let snapshot = MemoryGraph::new().snapshot();
        assert!(snapshot.nodes.is_empty());
        assert!(snapshot.relationships.is_empty());
        assert_eq!(
            serde_json::to_string(&snapshot).unwrap(),
            r#"{"nodes":[],"relationships":[]}"#
        );
    }

    #[test]
    fn graph_snapshot_deserialize_rejects_unknown_field() {
        let mut value = serde_json::to_value(GraphSnapshot::default()).unwrap();
        value
            .as_object_mut()
            .unwrap()
            .insert("bogus".to_string(), json!(true));
        let result: Result<GraphSnapshot, _> = serde_json::from_value(value);
        assert!(
            result.is_err(),
            "an unknown field must be rejected, not silently ignored"
        );
    }

    /// A snapshot round trip is lossless for every field, keeps parallel edges
    /// separate (relationship ids are preserved) and keeps a self-loop a
    /// self-loop (both endpoints are preserved).
    #[test]
    fn from_snapshot_round_trip_is_lossless() {
        let mut graph = MemoryGraph::new();
        graph.upsert_node(GraphNode {
            id: "alice".to_string(),
            labels: vec!["person".to_string(), "author".to_string()],
            properties: props(&[("meta", json!({"tags": ["x", "y"]}))]),
        });
        graph.upsert_node(node("bob"));
        graph.upsert_node(node("carol"));
        // Parallel edges alice -> bob with distinct ids and distinct payloads.
        let mut parallel_a = rel("r1", "alice", "bob", "knows", 1.5);
        parallel_a.properties = props(&[("since", json!(2020))]);
        let mut parallel_b = rel("r2", "alice", "bob", "follows", -2.25);
        parallel_b.properties = props(&[("since", json!(2024))]);
        graph.add_relationship(parallel_a).unwrap();
        graph.add_relationship(parallel_b).unwrap();
        graph
            .add_relationship(rel("r3", "carol", "carol", "self", 0.0))
            .unwrap();

        let snapshot = graph.snapshot();
        let rebuilt = MemoryGraph::from_snapshot(snapshot.clone()).unwrap();

        assert_eq!(rebuilt.len(), graph.len());
        assert_eq!(rebuilt.relationship_count(), graph.relationship_count());
        for n in &snapshot.nodes {
            let got = rebuilt.node(&n.id).expect("node present");
            assert_eq!(got, n, "node '{}' must round-trip every field", n.id);
        }
        for r in &snapshot.relationships {
            let got = rebuilt.relationship(&r.id).expect("relationship present");
            assert_eq!(got.id, r.id);
            assert_eq!(got.from, r.from);
            assert_eq!(got.to, r.to);
            assert_eq!(got.kind, r.kind);
            assert_eq!(got.weight, r.weight);
            assert_eq!(got.properties, r.properties);
        }
        // Parallel edges stayed separate; the self-loop stayed a self-loop.
        assert_eq!(rebuilt.relationship("r1").unwrap().kind, "knows");
        assert_eq!(rebuilt.relationship("r2").unwrap().kind, "follows");
        let loop_rel = rebuilt.relationship("r3").unwrap();
        assert_eq!(loop_rel.from, loop_rel.to);
        // Adjacency (all six collections) was rebuilt, not just the stores.
        assert_eq!(
            rebuilt.neighbors("alice", GraphDirection::Outgoing),
            vec!["bob".to_string()]
        );
        assert_eq!(
            rebuilt.neighbors("bob", GraphDirection::Any),
            vec!["alice".to_string()]
        );
        assert_eq!(
            rebuilt.neighbors("carol", GraphDirection::Any),
            vec!["carol".to_string()]
        );
        assert_endpoint_invariant(&rebuilt);
        assert_eq!(rebuilt.snapshot(), snapshot);
        assert_eq!(rebuilt, graph);
    }

    #[test]
    fn from_snapshot_rejects_duplicate_node_ids() {
        let snapshot = GraphSnapshot {
            nodes: vec![node("alice"), node("alice")],
            relationships: Vec::new(),
        };
        let err = MemoryGraph::from_snapshot(snapshot).unwrap_err();
        assert!(
            format!("{err}").contains("alice"),
            "error must name the duplicate node id: {err}"
        );
    }

    #[test]
    fn from_snapshot_rejects_duplicate_relationship_ids() {
        let snapshot = GraphSnapshot {
            nodes: vec![node("alice"), node("bob")],
            relationships: vec![
                rel("r1", "alice", "bob", "knows", 1.0),
                rel("r1", "bob", "alice", "knows", 1.0),
            ],
        };
        let err = MemoryGraph::from_snapshot(snapshot).unwrap_err();
        assert!(
            format!("{err}").contains("r1"),
            "error must name the duplicate relationship id: {err}"
        );
    }

    #[test]
    fn from_snapshot_rejects_missing_endpoints() {
        for (missing, r) in [
            ("ghost", rel("r1", "ghost", "bob", "knows", 1.0)),
            ("phantom", rel("r1", "bob", "phantom", "knows", 1.0)),
        ] {
            let snapshot = GraphSnapshot {
                nodes: vec![node("bob")],
                relationships: vec![r],
            };
            let err = MemoryGraph::from_snapshot(snapshot).unwrap_err();
            assert!(
                format!("{err}").contains(missing),
                "error must name the missing endpoint '{missing}': {err}"
            );
        }
    }

    #[test]
    fn from_snapshot_rejects_non_finite_weights() {
        for weight in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let snapshot = GraphSnapshot {
                nodes: vec![node("alice"), node("bob")],
                relationships: vec![rel("r1", "alice", "bob", "knows", weight)],
            };
            let err = MemoryGraph::from_snapshot(snapshot).unwrap_err();
            assert!(
                format!("{err}").contains("r1"),
                "error must name the offending relationship for weight {weight}: {err}"
            );
        }
    }

    /// Every rejection is total: no graph is returned, and neither the live
    /// graph a caller might replace nor its snapshot is disturbed.
    #[test]
    fn from_snapshot_errors_leave_the_live_graph_untouched() {
        let mut live = MemoryGraph::new();
        live.upsert_node(node("sentinel"));
        live.upsert_node(node("other"));
        live.add_relationship(rel("keep", "sentinel", "other", "knows", 1.0))
            .unwrap();
        let before = live.snapshot();

        let invalid = [
            GraphSnapshot {
                nodes: vec![node("dup"), node("dup")],
                relationships: Vec::new(),
            },
            GraphSnapshot {
                nodes: vec![node("a"), node("b")],
                relationships: vec![
                    rel("r1", "a", "b", "knows", 1.0),
                    rel("r1", "a", "b", "knows", 2.0),
                ],
            },
            GraphSnapshot {
                nodes: vec![node("a")],
                relationships: vec![rel("r1", "a", "missing", "knows", 1.0)],
            },
            GraphSnapshot {
                nodes: vec![node("a"), node("b")],
                relationships: vec![rel("r1", "a", "b", "knows", f64::NAN)],
            },
        ];
        for snapshot in invalid {
            assert!(MemoryGraph::from_snapshot(snapshot).is_err());
            assert_eq!(live.snapshot(), before);
        }
        assert_eq!(live.len(), 2);
        assert_eq!(live.relationship_count(), 1);
        assert!(live.relationship("keep").is_some());
    }

    /// Import is a wholesale replacement, never a merge: a valid snapshot
    /// reconstructs exactly its own contents.
    #[test]
    fn from_snapshot_replaces_rather_than_merges() {
        let snapshot = GraphSnapshot {
            nodes: vec![node("new")],
            relationships: Vec::new(),
        };
        let rebuilt = MemoryGraph::from_snapshot(snapshot).unwrap();
        assert_eq!(rebuilt.len(), 1);
        assert!(rebuilt.node("new").is_some());
        assert_eq!(rebuilt.relationship_count(), 0);
    }

    #[test]
    fn from_snapshot_accepts_empty_snapshot() {
        let rebuilt = MemoryGraph::from_snapshot(GraphSnapshot::default()).unwrap();
        assert!(rebuilt.is_empty());
        assert_eq!(rebuilt, MemoryGraph::new());
    }
}
