+++
title = "Entity extraction"
description = "Schema-guided entity and relationship extraction into a graph delta for the memory plugin's working graph."
template = "page.html"
weight = 1
+++

# Entity extraction

<dl class="page-facts">
<dt>In one line</dt>
<dd>Validates model-produced entity and relationship candidates against a declared schema and builds a standalone graph delta.</dd>
<dt>You need</dt>
<dd>The <code>service-entity-extraction</code> feature, which enables <code>plugin-memory</code>.</dd>
<dt>Read this if</dt>
<dd>You call <code>EntityExtractor::extract</code> or apply its report to a <a href="@/plugins/memory.md">memory</a> working graph.</dd>
</dl>

## Feature and entry type

| Item | Value |
|---|---|
| Feature | `service-entity-extraction` |
| Declared plugin edge | `plugin-memory`, hard: `service-entity-extraction = ["plugin-memory"]` |
| Own dependencies | none |
| Entry type | `EntityExtractor` |
| Module | `cuca::services::entity_extraction` |
| Model boundary | `EntityExtractionModel`, implemented by the caller |

`EntityExtractor` never implements `CucaPlugin`. Passing it to `register_plugin` is a compile error.

The delta is a `MemoryGraph`, which is why the feature needs `plugin-memory`. The edge is one way: `plugin-memory` never references this module.

Supporting types: `EntityExtractionSchema`, `EntityTable`, `RelationshipTable`, `PropertyColumn`, `PropertyType`, `EntityReference`, `CandidateEntity`, `CandidateRelationship`, `EntityExtractionCandidate`, `EntityExtractionReport`.

## Entry points

- `EntityExtractor::new(schema)` validates and retains the schema. It returns `Result<EntityExtractor, PluginError>`.
- `extract(source, model).await` asks the `EntityExtractionModel` for an unvalidated `EntityExtractionCandidate`, then runs `validate_candidate` on it.
- `validate_candidate(candidate)` validates a candidate the caller already holds.
- `EntityExtractionModel::extract(source, schema)` is the trait method the caller implements. It returns a boxed future of `Result<EntityExtractionCandidate, PluginError>`, so a transport or decode failure maps to the matching `PluginError`.

Both extractor methods return `EntityExtractionReport`:

- `delta` is a standalone `MemoryGraph`.
- `nodes_accepted` and `relationships_accepted` count the delta's nodes and relationships after duplicates coalesce.

### Hand-off the caller performs

The extractor never touches a `MemoryPlugin`. The delta has no effect until the application applies it, and dropping the report discards the extraction.

- `MemoryPlugin::merge_graph(delta, MergePolicy)` accumulates the delta into the working graph.
- `MemoryPlugin::replace_graph(graph)` replaces the working graph wholesale.

### Derived ids

- A node id is `entity:` followed by the length prefixed table name and a canonical encoding of the identity columns, so the same extraction always yields the same ids.
- A relationship id folds in the table name, both endpoint ids and the relationship's own properties.
- Every relationship has weight `1.0` and the `kind` of its table.
- Identical duplicate candidates count once.

## Configuration

`EntityExtractor` has no config struct and no defaults. Its only input is the `EntityExtractionSchema`, and every field is required.

| Key | Type | Rule |
|---|---|---|
| `EntityExtractionSchema::name` | `String` | Not blank; reported as `schema` in every `PluginError::Validation`. |
| `EntityExtractionSchema::entities` | `Vec<EntityTable>` | Table names are unique across entities and relationships. |
| `EntityExtractionSchema::relationships` | `Vec<RelationshipTable>` | Each endpoint names a declared entity table. |
| `EntityTable::name` | `String` | The name a model uses in `CandidateEntity::table`. |
| `EntityTable::labels` | `Vec<String>` | Copied onto every node the table produces. |
| `EntityTable::identity_columns` | `Vec<String>` | Non-empty, no duplicates, each a `required` column; forms the node id. |
| `EntityTable::columns` | `Vec<PropertyColumn>` | Declared properties. |
| `EntityTable::allow_model_properties` | `bool` | `true` accepts undeclared properties; `false` rejects them. |
| `RelationshipTable::name` | `String` | The name a model uses in `CandidateRelationship::table`. |
| `RelationshipTable::kind` | `String` | Edge kind copied onto every edge the table produces. |
| `RelationshipTable::from_table`<br>`RelationshipTable::to_table` | `String` | Entity tables the endpoints must reference. |
| `RelationshipTable::columns`<br>`RelationshipTable::allow_model_properties` | `Vec<PropertyColumn>`<br>`bool` | Same rules as on `EntityTable`. |
| `PropertyColumn::name` | `String` | The property key in the candidate's property map. |
| `PropertyColumn::property_type` | `PropertyType` | `String`, `Boolean`, `Integer`, `Number`, `Array`, `Object` or `Null`. |
| `PropertyColumn::required` | `bool` | The property must be present on every row of the table. |

- `PropertyType::Integer` accepts a JSON number with an exact integer representation; `Number` accepts any JSON number.
- `PropertyType::Array` and `Object` do not constrain element or member shapes.
- A relationship endpoint is an `EntityReference` holding a `table` and exactly that table's identity columns.

## Caps and at-cap policy

No cap: the extractor retains only its immutable schema, and nothing grows with traffic. Each call returns one standalone report, and the size of a report follows the candidate it validates.

## Errors

The extractor raises one error, `PluginError::Validation { schema, message }`. Validation is total: nothing is accepted partially.

`EntityExtractor::new` raises it for:

- a blank schema name;
- a duplicate table name;
- an entity table with no identity column;
- an identity column that is missing, duplicated or not `required`;
- a duplicate or malformed property column;
- a relationship endpoint naming an undeclared entity table.

`validate_candidate` raises it for:

- an unknown entity or relationship table;
- a missing required property or a missing identity property;
- a property whose JSON shape does not match its column;
- an undeclared property on a table with `allow_model_properties = false`;
- a relationship endpoint whose table differs from the declared one, or that does not carry exactly its table's identity columns;
- an endpoint that no accepted entity of the same candidate satisfies;
- two conflicting candidates that share one derived id.

`extract` also returns whatever the `EntityExtractionModel` returns, which is implementation defined. The error variants are listed in [Error types](@/reference/errors.md).

## Example

This excerpt extracts a candidate, validates it, and performs the hand-off.

```rust,name=Extract, validate and merge into memory
let extractor = EntityExtractor::new(schema)?;           // schema: EntityExtractionSchema
let report = extractor.extract(source, &model).await?;   // model: impl EntityExtractionModel
let memory = MemoryPlugin::new(MemoryConfig::default())?;
memory.merge_graph(report.delta, MergePolicy::Overwrite)?;
```

`examples/entity_extraction.rs` is the full program with an org chart schema. It extracts through a live model, merges the delta, renders the graph into a second request and asks a question only the graph can answer. It needs a `llama-server` on port 1234. `CUCA_BASE_URL` and `CUCA_MODEL` retarget it at any OpenAI-compatible server.

```bash,name=Runs the same on all three platforms
cargo run --example entity_extraction --features "provider-llamacpp service-entity-extraction"
```

## Related pages

- [Memory](@/plugins/memory.md) is the plugin whose working graph receives the delta.
- [Plugins and services](@/concepts/plugin-layering.md) explains why the edge is one way.
- [Services overview](@/services/_index.md) lists all six services.
