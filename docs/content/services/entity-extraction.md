+++
title = "Entity extraction"
description = "Schema-guided entity and relationship extraction into a graph delta for the memory plugin's working graph."
template = "page.html"
weight = 1
+++

# Entity extraction

<dl class="page-facts">
<dt>In one line</dt>
<dd>Validates model-produced entity and relationship candidates against a declared schema and builds a graph delta.</dd>
<dt>You need</dt>
<dd>The <code>service-entity-extraction</code> feature, which enables <code>plugin-memory</code>.</dd>
<dt>Read this if</dt>
<dd>You are calling <code>EntityExtractor::extract</code> or applying its report to a working memory graph.</dd>
</dl>

`EntityExtractor` validates model-produced entity and relationship rows against a schema you declare up front, then builds the accepted rows into a graph delta. It is driven by direct calls: `extract(source, model)` asks an `EntityExtractionModel` for a candidate and validates it, and `validate_candidate(candidate)` validates a candidate you already hold. Both return an `EntityExtractionReport { delta, nodes_accepted, relationships_accepted }` whose `delta` is a standalone `MemoryGraph`, never a mutation of live plugin state. Reach for it when a model should populate a [`MemoryPlugin`](@/plugins/memory.md) working graph under a declared contract instead of free-form text.

```rust,name=Validate a candidate and merge its delta
use cuca::{
    CandidateEntity, EntityExtractionCandidate, EntityExtractionSchema, EntityExtractor,
    EntityTable, MemoryConfig, MemoryPlugin, MergePolicy, PropertyColumn, PropertyType,
};

let extractor = EntityExtractor::new(EntityExtractionSchema {
    name: "contacts".into(),
    entities: vec![EntityTable {
        name: "person".into(),
        labels: vec!["Person".into()],
        identity_columns: vec!["email".into()],
        columns: vec![PropertyColumn {
            name: "email".into(),
            property_type: PropertyType::String,
            required: true,
        }],
        allow_model_properties: false,
    }],
    relationships: vec![],
})?;

let report = extractor.validate_candidate(EntityExtractionCandidate {
    entities: vec![CandidateEntity {
        table: "person".into(),
        properties: [("email".to_string(), serde_json::json!("ada@example.com"))]
            .into_iter()
            .collect(),
    }],
    relationships: vec![],
})?;
println!(
    "{} node(s), {} relationship(s)",
    report.nodes_accepted, report.relationships_accepted
);

// The delta is inert until it is applied, and dropping the report discards
// the extraction. `MemoryPlugin::replace_graph` swaps the graph wholesale.
let memory = MemoryPlugin::new(MemoryConfig::default())?;
let merged = memory.merge_graph(report.delta, MergePolicy::Overwrite)?;
println!("{} node(s) added", merged.nodes_added);
```

```text,name=Expected output
1 node(s), 0 relationship(s)
1 node(s) added
```

## Feature edge

`service-entity-extraction = ["plugin-memory"]` in `Cargo.toml`: enabling this feature enables `plugin-memory` with it. This is one of the crate's three hard service feature edges.

## Entry types

`EntityExtractor`, `EntityExtractionSchema`, `EntityTable`, `RelationshipTable`, `PropertyColumn`, `PropertyType`, `EntityReference`, `CandidateEntity`, `CandidateRelationship`, `EntityExtractionCandidate`, `EntityExtractionReport`, `EntityExtractionModel`.

## Schema

`EntityExtractionSchema` declares:

- `entities: Vec<EntityTable>`, each with a `name`, `labels` copied onto every produced node, `identity_columns` that form the node's identity and hence its graph id, declared `columns: Vec<PropertyColumn>`, and `allow_model_properties`.
- `relationships: Vec<RelationshipTable>`, each with a `name`, an edge `kind`, `from_table` and `to_table` endpoint constraints, declared `columns`, and `allow_model_properties`.

Table names are unique across entities and relationships. Each `PropertyColumn` declares a `name`, a `property_type` (`String`, `Boolean`, `Integer`, `Number`, `Array`, `Object`, or `Null`), and whether it is `required`.

## Validation

Validation is total: an unknown table, a missing required property, a type mismatch, an undeclared property on a table with `allow_model_properties = false`, or a relationship endpoint that no accepted entity satisfies all return `PluginError::Validation`. Nothing is accepted partially.

## Capacity

No growth cap. Each call produces one standalone report; nothing is retained between calls.
