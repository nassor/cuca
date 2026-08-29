+++
title = "Entity extraction"
description = "Schema-guided entity and relationship extraction into a graph delta for the memory plugin's working graph."
template = "page.html"
weight = 4
+++

# Entity extraction

<dl class="page-facts">
<dt>In one line</dt>
<dd>Validates model-produced entity and relationship candidates against a declared schema and builds a graph delta.</dd>
<dt>You need</dt>
<dd>The <code>plugin-entity-extraction</code> feature, which enables <code>plugin-memory</code>.</dd>
<dt>Read this if</dt>
<dd>You are calling <code>EntityExtractionPlugin::extract</code> or applying its report to a working memory graph.</dd>
</dl>

## Feature edge

`plugin-entity-extraction = ["plugin-memory"]` in `Cargo.toml`: enabling this feature enables `plugin-memory` with it. This is the crate's one cross-plugin feature edge.

## Entry types

`EntityExtractionPlugin`, `EntityExtractionSchema`, `EntityTable`, `RelationshipTable`, `PropertyColumn`, `PropertyType`, `EntityReference`, `CandidateEntity`, `CandidateRelationship`, `EntityExtractionCandidate`, `EntityExtractionReport`, `EntityExtractionModel`.

## Not a `CucaPlugin`

`EntityExtractionPlugin` does not implement `CucaPlugin`. It has no request or stream hooks, so registering it with `register_plugin` is a compile error, not an inert no-op. It is driven by direct calls:

- `EntityExtractionPlugin::extract(source, model)` asks an `EntityExtractionModel` for a candidate, then validates it.
- `EntityExtractionPlugin::validate_candidate(candidate)` validates a candidate the caller already has.

Both return an `EntityExtractionReport { delta, nodes_accepted, relationships_accepted }`. `delta` is a standalone `MemoryGraph`, not a mutation of any live plugin state; the extraction step never touches a [`MemoryPlugin`](@/plugins/memory.md). Applying the delta is the caller's job, through `MemoryPlugin::merge_graph` with a chosen `MergePolicy`, or `MemoryPlugin::replace_graph` for a wholesale replacement. Dropping the report discards the extraction.

## Schema

`EntityExtractionSchema` declares:

- `entities: Vec<EntityTable>`, each with a `name`, `labels` copied onto every produced node, `identity_columns` that form the node's identity and hence its graph id, declared `columns: Vec<PropertyColumn>`, and `allow_model_properties`.
- `relationships: Vec<RelationshipTable>`, each with a `name`, an edge `kind`, `from_table` and `to_table` endpoint constraints, declared `columns`, and `allow_model_properties`.

Table names are unique across entities and relationships. Each `PropertyColumn` declares a `name`, a `property_type` (`String`, `Boolean`, `Integer`, `Number`, `Array`, `Object`, or `Null`), and whether it is `required`.

## Validation

Validation is total: an unknown table, a missing required property, a type mismatch, an undeclared property on a table with `allow_model_properties = false`, or a relationship endpoint that no accepted entity satisfies all return `PluginError::Validation`. Nothing is accepted partially.

## Capacity

No growth cap. Each call produces one standalone report; nothing is retained between calls.
