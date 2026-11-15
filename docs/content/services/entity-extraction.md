+++
title = "Entity extraction"
description = "Check entities and relationships a model found in text against a schema, and get a graph you can merge into memory."
template = "page.html"
weight = 1
+++

# Entity extraction

<dl class="page-facts">
<dt>In one line</dt>
<dd>Checks the entities and relationships a model found in text against a schema you declare, and returns them as a graph.</dd>
<dt>You need</dt>
<dd>The <code>service-entity-extraction</code> feature, which turns on <code>plugin-memory</code>.</dd>
<dt>Read this if</dt>
<dd>You want typed facts from prose, such as "Ada works at Analytical Engines", and you want to put them in the <a href="@/plugins/memory.md">memory</a> graph.</dd>
</dl>

## What it does

A model reads text and proposes entities (people, companies) and relationships between them. `EntityExtractor` checks every proposal against a schema you wrote. If everything is valid, you get a graph. If anything is invalid, you get an error and nothing is kept.

<div class="dgm">
<div class="dgm-scroll">
<svg viewBox="0 0 660 130" role="img" aria-labelledby="ee-flow-title ee-flow-desc">
  <title id="ee-flow-title">Text goes through a model, then a schema check, then a graph delta that your code merges into memory.</title>
  <desc id="ee-flow-desc">Four boxes in a row: source text, your EntityExtractionModel, the EntityExtractor schema check, and the MemoryGraph delta. A final arrow shows your code calling merge_graph on the memory plugin.</desc>
  <defs>
    <marker id="ee-flow-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M0 0.6 L7 4 L0 7.4 z" fill="var(--muted-foreground)"/>
    </marker>
  </defs>
  <text class="t-title" x="6" y="20">EntityExtractor::extract</text>
  <rect class="blk blk-data" x="6" y="40" width="110" height="50" rx="6"/>
  <text class="t-sm t-mid t-data" x="61" y="69">source text</text>
  <path class="arw arw-data" d="M118 65 H142" marker-end="url(#ee-flow-arrow)"/>
  <rect class="blk blk-bnd" x="144" y="40" width="130" height="50" rx="6"/>
  <text class="t-sm t-mid t-bnd" x="209" y="62">your model</text>
  <text class="t-sm t-mid" x="209" y="77">proposes rows</text>
  <path class="arw arw-data" d="M276 65 H300" marker-end="url(#ee-flow-arrow)"/>
  <rect class="blk blk-ctl" x="302" y="40" width="130" height="50" rx="6"/>
  <text class="t-sm t-mid t-ctl" x="367" y="62">schema check</text>
  <text class="t-sm t-mid" x="367" y="77">all or nothing</text>
  <path class="arw arw-data" d="M434 65 H458" marker-end="url(#ee-flow-arrow)"/>
  <rect class="blk blk-data" x="460" y="40" width="94" height="50" rx="6"/>
  <text class="t-sm t-mid t-data" x="507" y="62">report</text>
  <text class="t-sm t-mid" x="507" y="77">graph delta</text>
  <path class="arw" d="M556 65 H580" marker-end="url(#ee-flow-arrow)"/>
  <rect class="blk" x="582" y="40" width="72" height="50" rx="6"/>
  <text class="t-sm t-mid" x="618" y="62">your code</text>
  <text class="t-sm t-mid" x="618" y="77">merges it</text>
  <text class="t-sm" x="6" y="116">The extractor never touches the memory plugin. Dropping the report throws the result away.</text>
</svg>
</div>
<p class="dgm-cap"><b>The delta does nothing until you merge it.</b> The extractor returns a standalone graph, and your code decides when and how to apply it.</p>
</div>

## Try it

This program checks a hand-written candidate against a small schema. It needs no model and no server.

```rust,name=Validate one candidate against a schema
use cuca::{
    CandidateEntity, CandidateRelationship, EntityExtractionCandidate, EntityExtractionSchema,
    EntityExtractor, EntityReference, EntityTable, PropertyColumn, PropertyType,
    RelationshipTable,
};
use serde_json::{Map, Value, json};

fn name_column() -> PropertyColumn {
    PropertyColumn { name: "name".into(), property_type: PropertyType::String, required: true }
}

fn name_map(name: &str) -> Map<String, Value> {
    let mut map = Map::new();
    map.insert("name".into(), json!(name));
    map
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let table = |name: &str| EntityTable {
        name: name.into(),
        labels: vec![name.into()],
        identity_columns: vec!["name".into()],
        columns: vec![name_column()],
        allow_model_properties: false,
    };
    let extractor = EntityExtractor::new(EntityExtractionSchema {
        name: "org-chart".into(),
        entities: vec![table("person"), table("company")],
        relationships: vec![RelationshipTable {
            name: "works_at".into(),
            kind: "works_at".into(),
            from_table: "person".into(),
            to_table: "company".into(),
            columns: vec![],
            allow_model_properties: false,
        }],
    })?;

    let report = extractor.validate_candidate(EntityExtractionCandidate {
        entities: vec![
            CandidateEntity { table: "person".into(), properties: name_map("Ada") },
            CandidateEntity { table: "company".into(), properties: name_map("Analytical Engines") },
        ],
        relationships: vec![CandidateRelationship {
            table: "works_at".into(),
            from: EntityReference { table: "person".into(), identity: name_map("Ada") },
            to: EntityReference { table: "company".into(), identity: name_map("Analytical Engines") },
            properties: Map::new(),
        }],
    })?;
    println!("nodes {}, relationships {}", report.nodes_accepted, report.relationships_accepted);
    Ok(())
}
```

```text,name=Expected output
nodes 2, relationships 1
```

To merge the result into memory, add one call:

```rust,name=Merge the delta into the memory graph
let memory = MemoryPlugin::new(MemoryConfig::default())?;
memory.merge_graph(report.delta, MergePolicy::Overwrite)?;
```

To let a model do the proposing, call `extractor.extract(source, &model).await?`. It asks your `EntityExtractionModel` for a candidate, then runs the same check.

`examples/entity_extraction.rs` is the full program. It extracts an org chart through a live model, merges the graph, and asks a question that only the graph can answer. It needs a `llama-server` on port 1234. The flags `--base-url` and `--model`, written after `--`, point it at any OpenAI-compatible server.

```bash,name=Runs the same on all three platforms
cargo run --example entity_extraction --features "provider-llamacpp service-entity-extraction"
```

## Feature and entry type

| Item | Value |
|---|---|
| Feature | `service-entity-extraction` |
| Plugin it needs | `plugin-memory`: `service-entity-extraction = ["plugin-memory"]` |
| Own dependencies | none |
| Entry type | `EntityExtractor` |
| Module | `cuca::services::entity_extraction` |
| Model trait | `EntityExtractionModel`, implemented by you |

Supporting types: `EntityExtractionSchema`, `EntityTable`, `RelationshipTable`, `PropertyColumn`, `PropertyType`, `EntityReference`, `CandidateEntity`, `CandidateRelationship`, `EntityExtractionCandidate`, `EntityExtractionReport`.

## Entry points

| Method | What it does |
|---|---|
| `EntityExtractor::new(schema)` | Validates and keeps the schema. Returns `Result<EntityExtractor, PluginError>`. |
| `extract(source, model).await` | Asks the model for a candidate, then validates it. |
| `validate_candidate(candidate)` | Validates a candidate you already hold. |
| `EntityExtractionModel::extract(source, schema)` | The trait method you implement. It returns a boxed future of `Result<EntityExtractionCandidate, PluginError>`. |

Both extractor methods return an `EntityExtractionReport`:

- `delta` is a standalone `MemoryGraph`.
- `nodes_accepted` and `relationships_accepted` count the nodes and relationships in the delta after duplicates merge into one.

### Apply the delta yourself

- `MemoryPlugin::merge_graph(delta, MergePolicy)` adds the delta to the working graph.
- `MemoryPlugin::replace_graph(graph)` replaces the working graph.

### How ids are built

- A node id is `entity:` followed by the table name and the identity columns, encoded so that the same input always gives the same id.
- A relationship id combines the table name, both endpoint ids and the relationship's own properties.
- Every relationship has weight `1.0` and the `kind` of its table.
- Identical duplicate candidates count once.

## Configuration

`EntityExtractor` has no config struct and no defaults. Its only input is the schema, and every field is required.

- `EntityExtractionSchema::name` (`String`): not blank. It appears as `schema` in every `PluginError::Validation`.
- `EntityExtractionSchema::entities` (`Vec<EntityTable>`): table names are unique across entities and relationships.
- `EntityExtractionSchema::relationships` (`Vec<RelationshipTable>`): each endpoint names a declared entity table.
- `EntityTable::name` (`String`): the name a model uses in `CandidateEntity::table`.
- `EntityTable::labels` (`Vec<String>`): copied onto every node the table produces.
- `EntityTable::identity_columns` (`Vec<String>`): non-empty, no duplicates, each a `required` column. They form the node id.
- `EntityTable::columns` (`Vec<PropertyColumn>`): the declared properties.
- `EntityTable::allow_model_properties` (`bool`): `true` accepts undeclared properties. `false` rejects them.
- `RelationshipTable::name` (`String`): the name a model uses in `CandidateRelationship::table`.
- `RelationshipTable::kind` (`String`): the edge kind copied onto every edge the table produces.
- `RelationshipTable::from_table` and `RelationshipTable::to_table` (`String`): the entity tables the endpoints must reference.
- `RelationshipTable::columns` (`Vec<PropertyColumn>`) and `RelationshipTable::allow_model_properties` (`bool`): the same rules as on `EntityTable`.
- `PropertyColumn::name` (`String`): the property key in the candidate's property map.
- `PropertyColumn::property_type` (`PropertyType`): `String`, `Boolean`, `Integer`, `Number`, `Array`, `Object` or `Null`.
- `PropertyColumn::required` (`bool`): the property must be present on every row of the table.
- `PropertyType::Integer` accepts a JSON number with an exact integer value. `Number` accepts any JSON number.
- `PropertyType::Array` and `Object` do not check what is inside.
- A relationship endpoint is an `EntityReference`: a `table` and exactly that table's identity columns.

## Caps and at-cap policy

There is no cap. The extractor keeps only its schema, so nothing grows with traffic. Each call returns one standalone report, sized by the candidate it checks.

## Errors

The extractor raises one error: `PluginError::Validation { schema, message }`. Validation is all or nothing.

`EntityExtractor::new` raises it for:

- a blank schema name;
- a duplicate table name;
- an entity table with no identity column;
- an identity column that is missing, duplicated or not `required`;
- a duplicate or malformed property column;
- a relationship endpoint that names an undeclared entity table.

`validate_candidate` raises it for:

- an unknown entity or relationship table;
- a missing required property or a missing identity property;
- a property whose JSON type does not match its column;
- an undeclared property on a table with `allow_model_properties = false`;
- a relationship endpoint whose table differs from the declared one, or that does not carry exactly its table's identity columns;
- an endpoint that no accepted entity in the same candidate matches;
- two different candidates that produce the same id.

`extract` also returns whatever your `EntityExtractionModel` returns. [Error types](@/reference/errors.md) lists the variants.

## Related pages

- [Memory](@/plugins/memory.md) is the plugin whose graph receives the delta.
- [Plugins and services](@/concepts/plugin-layering.md) explains why the dependency runs one way.
- [Services](@/services/_index.md) lists all seven services.
