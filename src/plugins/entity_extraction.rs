//! Schema-guided entity and relationship extraction (`plugin-entity-extraction`).
//!
//! [`EntityExtractionPlugin`] validates model-produced entity candidates
//! against an immutable [`EntityExtractionSchema`] (entity tables,
//! relationship tables, declared property columns, identity columns) and turns
//! the accepted candidate into a graph delta. Validation is total: an unknown
//! table, a missing required property, a type mismatch, an undeclared property
//! on a table with `allow_model_properties = false`, or a relationship endpoint
//! that no accepted entity satisfies all return [`PluginError::Validation`]
//! instead of a partially accepted delta.
//!
//! # Hard dependency on `plugin-memory`
//!
//! The delta is a [`MemoryGraph`] built from [`GraphNode`]/[`GraphRelationship`]
//! values, so this feature enables `plugin-memory`
//! (`plugin-entity-extraction = ["plugin-memory"]`). The dependency is one-way:
//! `plugin-memory` never references this module.
//!
//! # Explicit-call contract
//!
//! [`EntityExtractionPlugin`] is not a [`crate::plugin::CucaPlugin`]: it has no
//! request/stream hooks, so it cannot be registered on the client builder (as
//! with `PromptCache`, the compiler rejects the attempt rather than accepting
//! an inert registration). Extraction is driven by direct calls:
//! [`EntityExtractionPlugin::extract`] asks an [`EntityExtractionModel`] for a
//! candidate and validates it, and
//! [`EntityExtractionPlugin::validate_candidate`] validates a candidate the
//! caller already has.
//!
//! # Mandatory hand-off
//!
//! Both entry points return an [`EntityExtractionReport`] whose
//! [`EntityExtractionReport::delta`] is a standalone graph, **not** a mutation
//! of any live plugin state: the extraction step never touches a
//! [`crate::plugins::memory::MemoryPlugin`]. The delta has no effect until the
//! application applies it, via
//! [`crate::plugins::memory::MemoryPlugin::merge_graph`] with the desired
//! [`crate::plugins::memory::MergePolicy`] (accumulating extraction into the
//! working graph) or
//! [`crate::plugins::memory::MemoryPlugin::replace_graph`] (wholesale
//! replacement). Dropping the report discards the extraction.

use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::pin::Pin;

use crate::error::PluginError;
use crate::plugins::memory::{GraphNode, GraphRelationship, MemoryGraph};

/// Declarative contract for extracting entities and relationships from source text.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct EntityExtractionSchema {
    /// Schema identifier, reported as the `schema` field of every
    /// [`PluginError::Validation`] this module raises. Must not be blank.
    pub name: String,
    /// Entity tables; table names are unique across entities *and*
    /// relationships.
    pub entities: Vec<EntityTable>,
    /// Relationship tables; each endpoint must name a declared entity table.
    pub relationships: Vec<RelationshipTable>,
}

/// Schema for one entity table and the graph nodes it produces.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct EntityTable {
    /// Table name the model uses in [`CandidateEntity::table`].
    pub name: String,
    /// Labels copied onto every [`GraphNode`] this table produces.
    pub labels: Vec<String>,
    /// Columns forming the node identity, and hence its graph id. Must be
    /// non-empty, free of duplicates, and every entry must be a `required`
    /// column of [`Self::columns`].
    pub identity_columns: Vec<String>,
    /// Declared properties, with the JSON shape each one must have.
    pub columns: Vec<PropertyColumn>,
    /// Whether properties absent from [`Self::columns`] are accepted (`true`)
    /// or rejected as unknown (`false`).
    pub allow_model_properties: bool,
}

/// Schema for one directed relationship table and the graph edges it produces.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct RelationshipTable {
    /// Table name the model uses in [`CandidateRelationship::table`].
    pub name: String,
    /// Edge kind copied onto every [`GraphRelationship`] this table produces.
    pub kind: String,
    /// Entity table the `from` endpoint must reference.
    pub from_table: String,
    /// Entity table the `to` endpoint must reference.
    pub to_table: String,
    /// Declared properties, with the JSON shape each one must have.
    pub columns: Vec<PropertyColumn>,
    /// Whether properties absent from [`Self::columns`] are accepted (`true`)
    /// or rejected as unknown (`false`).
    pub allow_model_properties: bool,
}

/// One declared JSON property in an entity or relationship table.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct PropertyColumn {
    /// Property key as it appears in the candidate's property map.
    pub name: String,
    /// JSON shape the value must have.
    pub property_type: PropertyType,
    /// Whether the property must be present on every row of the table.
    pub required: bool,
}

/// JSON value shape allowed for a declared property.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum PropertyType {
    /// JSON string.
    String,
    /// JSON boolean.
    Boolean,
    /// JSON number with an exact integer representation.
    Integer,
    /// Any JSON number.
    Number,
    /// JSON array (element shapes are not constrained).
    Array,
    /// JSON object (member shapes are not constrained).
    Object,
    /// JSON null.
    Null,
}

/// An entity endpoint identified by its table's declared identity columns.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct EntityReference {
    /// Entity table the endpoint points at; must equal the relationship
    /// table's declared `from_table`/`to_table`.
    pub table: String,
    /// Exactly the table's `identity_columns` and their values: no more, no
    /// fewer.
    pub identity: serde_json::Map<String, serde_json::Value>,
}

/// A model-proposed entity row.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CandidateEntity {
    /// Name of the [`EntityTable`] this row belongs to.
    pub table: String,
    /// Row properties, validated against that table's columns.
    pub properties: serde_json::Map<String, serde_json::Value>,
}

/// A model-proposed directed relationship row.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CandidateRelationship {
    /// Name of the [`RelationshipTable`] this row belongs to.
    pub table: String,
    /// Source endpoint; must resolve to an entity accepted from the same
    /// candidate.
    pub from: EntityReference,
    /// Target endpoint; must resolve to an entity accepted from the same
    /// candidate.
    pub to: EntityReference,
    /// Row properties, validated against that table's columns.
    pub properties: serde_json::Map<String, serde_json::Value>,
}

/// A complete model-proposed graph extraction.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct EntityExtractionCandidate {
    /// Proposed entity rows; relationship endpoints may only reference these.
    pub entities: Vec<CandidateEntity>,
    /// Proposed relationship rows.
    pub relationships: Vec<CandidateRelationship>,
}

/// The graph delta accepted from one extraction candidate.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct EntityExtractionReport {
    /// Nodes in [`Self::delta`] (post-coalescing, so duplicates of one
    /// identity count once).
    pub nodes_accepted: usize,
    /// Relationships in [`Self::delta`] (post-coalescing).
    pub relationships_accepted: usize,
    /// The standalone graph delta. Applying it is the caller's job: see the
    /// module's "Mandatory hand-off" section.
    #[serde(
        serialize_with = "serialize_graph_delta",
        deserialize_with = "deserialize_graph_delta"
    )]
    pub delta: MemoryGraph,
}

/// Model boundary for explicitly extracting entity candidates from source text.
pub trait EntityExtractionModel: Send + Sync {
    /// Extract a candidate from `source` under `schema`.
    ///
    /// The candidate is unvalidated: [`EntityExtractionPlugin`] enforces
    /// `schema` afterwards, so an implementation may return whatever the model
    /// produced.
    ///
    /// # Errors
    ///
    /// Implementation-defined: a transport, decode, or model-refusal failure
    /// returns the matching [`PluginError`].
    fn extract<'a>(
        &'a self,
        source: &'a str,
        schema: &'a EntityExtractionSchema,
    ) -> Pin<Box<dyn Future<Output = Result<EntityExtractionCandidate, PluginError>> + Send + 'a>>;
}

/// Explicit entity-extraction capability backed by an immutable validated
/// schema. Not a `CucaPlugin`: call [`Self::extract`] /
/// [`Self::validate_candidate`] and apply the report's delta to a
/// `MemoryPlugin` (see the module docs).
#[derive(Debug, Clone, PartialEq)]
pub struct EntityExtractionPlugin {
    schema: EntityExtractionSchema,
}

impl EntityExtractionPlugin {
    /// Validate and retain an extraction schema.
    ///
    /// # Errors
    ///
    /// [`PluginError::Validation`] for a blank schema name, a duplicate table
    /// name, an entity table with no identity column, an identity column that
    /// is missing or not `required`, a duplicate or malformed property column,
    /// or a relationship endpoint naming an undeclared entity table.
    pub fn new(schema: EntityExtractionSchema) -> Result<Self, PluginError> {
        validate_schema(&schema)?;
        Ok(Self { schema })
    }

    /// Request and validate an entity extraction candidate from `model`.
    ///
    /// # Errors
    ///
    /// Whatever `model` returns, or the [`Self::validate_candidate`] errors for
    /// the candidate it produced.
    pub async fn extract(
        &self,
        source: &str,
        model: &dyn EntityExtractionModel,
    ) -> Result<EntityExtractionReport, PluginError> {
        let candidate = model.extract(source, &self.schema).await?;
        self.validate_candidate(candidate)
    }

    /// Validate a model-produced candidate and build its graph delta.
    ///
    /// # Errors
    ///
    /// [`PluginError::Validation`] for an unknown table, a missing required
    /// property, a property whose JSON shape does not match its column, an
    /// undeclared property on a table with `allow_model_properties = false`, a
    /// relationship endpoint that does not carry exactly its table's identity
    /// columns, or an endpoint no accepted entity of this candidate satisfies.
    /// Validation is total: nothing is accepted partially.
    pub fn validate_candidate(
        &self,
        candidate: EntityExtractionCandidate,
    ) -> Result<EntityExtractionReport, PluginError> {
        let mut nodes = HashMap::with_capacity(candidate.entities.len());
        for entity in candidate.entities {
            let table = self
                .schema
                .entities
                .iter()
                .find(|table| table.name == entity.table)
                .ok_or_else(|| {
                    validation_error(
                        &self.schema,
                        format!("unknown entity table `{}`", entity.table),
                    )
                })?;
            validate_properties(
                &self.schema,
                &table.name,
                &table.columns,
                table.allow_model_properties,
                &entity.properties,
            )?;
            let identity = entity_identity(&self.schema, table, &entity.properties)?;
            let node = GraphNode {
                id: entity_id(&table.name, &identity),
                labels: table.labels.clone(),
                properties: entity.properties,
            };
            coalesce(&self.schema, &mut nodes, node, "entity")?;
        }

        let mut relationships = HashMap::with_capacity(candidate.relationships.len());
        for relationship in candidate.relationships {
            let table = self
                .schema
                .relationships
                .iter()
                .find(|table| table.name == relationship.table)
                .ok_or_else(|| {
                    validation_error(
                        &self.schema,
                        format!("unknown relationship table `{}`", relationship.table),
                    )
                })?;
            validate_properties(
                &self.schema,
                &table.name,
                &table.columns,
                table.allow_model_properties,
                &relationship.properties,
            )?;
            let from = reference_id(&self.schema, &relationship.from, &table.from_table, "from")?;
            let to = reference_id(&self.schema, &relationship.to, &table.to_table, "to")?;
            if !nodes.contains_key(&from) || !nodes.contains_key(&to) {
                return Err(validation_error(
                    &self.schema,
                    format!(
                        "relationship table `{}` references an entity absent from the candidate",
                        table.name
                    ),
                ));
            }
            let relationship = GraphRelationship {
                id: relationship_id(&table.name, &from, &to, &relationship.properties),
                from,
                to,
                kind: table.kind.clone(),
                weight: 1.0,
                properties: relationship.properties,
            };
            coalesce(
                &self.schema,
                &mut relationships,
                relationship,
                "relationship",
            )?;
        }

        let mut nodes: Vec<_> = nodes.into_values().collect();
        nodes.sort_by(|left, right| left.id.cmp(&right.id));
        let mut relationships: Vec<_> = relationships.into_values().collect();
        relationships.sort_by(|left, right| left.id.cmp(&right.id));

        let mut delta = MemoryGraph::with_capacity(nodes.len(), relationships.len());
        for node in nodes {
            delta.upsert_node(node);
        }
        for relationship in relationships {
            delta.add_relationship(relationship).map_err(|error| {
                validation_error(
                    &self.schema,
                    format!("could not build graph delta: {error:?}"),
                )
            })?;
        }

        Ok(EntityExtractionReport {
            nodes_accepted: delta.len(),
            relationships_accepted: delta.relationship_count(),
            delta,
        })
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
struct GraphDeltaWire {
    nodes: Vec<GraphNode>,
    relationships: Vec<GraphRelationship>,
}

fn serialize_graph_delta<S>(graph: &MemoryGraph, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    let mut delta = GraphDeltaWire {
        nodes: graph.nodes().cloned().collect(),
        relationships: graph.relationships().cloned().collect(),
    };
    delta.nodes.sort_by(|left, right| left.id.cmp(&right.id));
    delta
        .relationships
        .sort_by(|left, right| left.id.cmp(&right.id));
    serde::Serialize::serialize(&delta, serializer)
}

fn deserialize_graph_delta<'de, D>(deserializer: D) -> Result<MemoryGraph, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let mut delta: GraphDeltaWire = serde::Deserialize::deserialize(deserializer)?;
    delta.nodes.sort_by(|left, right| left.id.cmp(&right.id));
    delta
        .relationships
        .sort_by(|left, right| left.id.cmp(&right.id));

    let mut graph = MemoryGraph::with_capacity(delta.nodes.len(), delta.relationships.len());
    for node in delta.nodes {
        graph.upsert_node(node);
    }
    for relationship in delta.relationships {
        graph
            .add_relationship(relationship)
            .map_err(<D::Error as serde::de::Error>::custom)?;
    }
    Ok(graph)
}

fn validate_schema(schema: &EntityExtractionSchema) -> Result<(), PluginError> {
    if schema.name.trim().is_empty() {
        return Err(validation_error(schema, "schema name must not be empty"));
    }

    let mut table_names = HashSet::new();
    for entity in &schema.entities {
        validate_table_name(schema, "entity", &entity.name, &mut table_names)?;
        validate_columns(schema, &entity.name, &entity.columns)?;

        if entity.identity_columns.is_empty() {
            return Err(validation_error(
                schema,
                format!(
                    "entity table `{}` must declare an identity column",
                    entity.name
                ),
            ));
        }

        let mut identity_names = HashSet::new();
        for identity_name in &entity.identity_columns {
            if !identity_names.insert(identity_name.as_str()) {
                return Err(validation_error(
                    schema,
                    format!(
                        "entity table `{}` declares identity column `{identity_name}` more than once",
                        entity.name
                    ),
                ));
            }

            match entity
                .columns
                .iter()
                .find(|column| column.name == *identity_name)
            {
                Some(column) if column.required => {}
                Some(_) => {
                    return Err(validation_error(
                        schema,
                        format!(
                            "identity column `{identity_name}` in entity table `{}` must be required",
                            entity.name
                        ),
                    ));
                }
                None => {
                    return Err(validation_error(
                        schema,
                        format!(
                            "identity column `{identity_name}` is not declared by entity table `{}`",
                            entity.name
                        ),
                    ));
                }
            }
        }
    }

    for relationship in &schema.relationships {
        validate_table_name(schema, "relationship", &relationship.name, &mut table_names)?;
        validate_columns(schema, &relationship.name, &relationship.columns)?;
    }

    let entity_names: HashSet<&str> = schema
        .entities
        .iter()
        .map(|entity| entity.name.as_str())
        .collect();
    for relationship in &schema.relationships {
        for (endpoint, table_name) in [
            ("from_table", relationship.from_table.as_str()),
            ("to_table", relationship.to_table.as_str()),
        ] {
            if !entity_names.contains(table_name) {
                return Err(validation_error(
                    schema,
                    format!(
                        "relationship table `{}` has unknown {endpoint} `{table_name}`",
                        relationship.name
                    ),
                ));
            }
        }
    }

    Ok(())
}
fn validate_properties(
    schema: &EntityExtractionSchema,
    table_name: &str,
    columns: &[PropertyColumn],
    allow_model_properties: bool,
    properties: &serde_json::Map<String, serde_json::Value>,
) -> Result<(), PluginError> {
    for (name, value) in properties {
        match columns.iter().find(|column| column.name == *name) {
            Some(column) if property_type_matches(&column.property_type, value) => {}
            Some(column) => {
                return Err(validation_error(
                    schema,
                    format!(
                        "property `{name}` in table `{table_name}` must be {:?}",
                        column.property_type
                    ),
                ));
            }
            None if !allow_model_properties => {
                return Err(validation_error(
                    schema,
                    format!("unknown property `{name}` in table `{table_name}`"),
                ));
            }
            None => {}
        }
    }

    for column in columns {
        if column.required && !properties.contains_key(&column.name) {
            return Err(validation_error(
                schema,
                format!(
                    "missing required property `{}` in table `{table_name}`",
                    column.name
                ),
            ));
        }
    }
    Ok(())
}

fn property_type_matches(property_type: &PropertyType, value: &serde_json::Value) -> bool {
    match property_type {
        PropertyType::String => value.is_string(),
        PropertyType::Boolean => value.is_boolean(),
        PropertyType::Integer => value.as_i64().is_some() || value.as_u64().is_some(),
        PropertyType::Number => value.is_number(),
        PropertyType::Array => value.is_array(),
        PropertyType::Object => value.is_object(),
        PropertyType::Null => value.is_null(),
    }
}

fn entity_identity(
    schema: &EntityExtractionSchema,
    table: &EntityTable,
    properties: &serde_json::Map<String, serde_json::Value>,
) -> Result<serde_json::Map<String, serde_json::Value>, PluginError> {
    let mut identity = serde_json::Map::new();
    for name in &table.identity_columns {
        let value = properties.get(name).ok_or_else(|| {
            validation_error(
                schema,
                format!(
                    "missing identity property `{name}` in entity table `{}`",
                    table.name
                ),
            )
        })?;
        identity.insert(name.clone(), value.clone());
    }
    Ok(identity)
}

fn reference_id(
    schema: &EntityExtractionSchema,
    reference: &EntityReference,
    expected_table: &str,
    endpoint: &str,
) -> Result<String, PluginError> {
    if reference.table != expected_table {
        return Err(validation_error(
            schema,
            format!(
                "relationship {endpoint} endpoint table `{}` does not match declared table `{expected_table}`",
                reference.table
            ),
        ));
    }
    let table = schema
        .entities
        .iter()
        .find(|table| table.name == reference.table)
        .ok_or_else(|| {
            validation_error(
                schema,
                format!("unknown entity table `{}`", reference.table),
            )
        })?;
    if reference.identity.len() != table.identity_columns.len()
        || table
            .identity_columns
            .iter()
            .any(|name| !reference.identity.contains_key(name))
    {
        return Err(validation_error(
            schema,
            format!(
                "relationship {endpoint} endpoint for entity table `{}` must contain exactly its identity columns",
                table.name
            ),
        ));
    }
    for name in &table.identity_columns {
        let column = table
            .columns
            .iter()
            .find(|column| column.name == *name)
            .ok_or_else(|| {
                validation_error(
                    schema,
                    format!(
                        "identity column `{name}` is not declared by entity table `{}`",
                        table.name
                    ),
                )
            })?;
        let value = reference.identity.get(name).ok_or_else(|| {
            validation_error(
                schema,
                format!(
                    "missing identity property `{name}` in entity table `{}`",
                    table.name
                ),
            )
        })?;
        if !property_type_matches(&column.property_type, value) {
            return Err(validation_error(
                schema,
                format!(
                    "identity property `{name}` in entity table `{}` must be {:?}",
                    table.name, column.property_type
                ),
            ));
        }
    }
    Ok(entity_id(&table.name, &reference.identity))
}

fn coalesce<T: PartialEq + HasId>(
    schema: &EntityExtractionSchema,
    values: &mut HashMap<String, T>,
    value: T,
    kind: &str,
) -> Result<(), PluginError> {
    let id = value.id().to_owned();
    match values.get(&id) {
        Some(existing) if existing != &value => Err(validation_error(
            schema,
            format!("conflicting {kind} candidates share derived id `{id}`"),
        )),
        Some(_) => Ok(()),
        None => {
            values.insert(id, value);
            Ok(())
        }
    }
}

trait HasId {
    fn id(&self) -> &str;
}

impl HasId for GraphNode {
    fn id(&self) -> &str {
        &self.id
    }
}

impl HasId for GraphRelationship {
    fn id(&self) -> &str {
        &self.id
    }
}

fn entity_id(table: &str, identity: &serde_json::Map<String, serde_json::Value>) -> String {
    let mut id = String::from("entity:");
    append_length_prefixed(&mut id, table);
    id.push(':');
    append_canonical_map(&mut id, identity);
    id
}

fn relationship_id(
    table: &str,
    from: &str,
    to: &str,
    properties: &serde_json::Map<String, serde_json::Value>,
) -> String {
    let mut id = String::from("relationship:");
    append_length_prefixed(&mut id, table);
    append_length_prefixed(&mut id, from);
    append_length_prefixed(&mut id, to);
    append_canonical_map(&mut id, properties);
    id
}

fn append_length_prefixed(out: &mut String, value: &str) {
    out.push_str(&value.len().to_string());
    out.push(':');
    out.push_str(value);
}

fn append_canonical_value(out: &mut String, value: &serde_json::Value) {
    match value {
        serde_json::Value::Null => out.push_str("null"),
        serde_json::Value::Bool(value) => out.push_str(if *value { "true" } else { "false" }),
        serde_json::Value::Number(value) => {
            out.push_str("number:");
            append_length_prefixed(out, &value.to_string());
        }
        serde_json::Value::String(value) => {
            out.push_str("string:");
            append_length_prefixed(out, value);
        }
        serde_json::Value::Array(values) => {
            out.push_str("array:");
            out.push_str(&values.len().to_string());
            out.push(':');
            for value in values {
                append_canonical_value(out, value);
            }
        }
        serde_json::Value::Object(values) => append_canonical_map(out, values),
    }
}

fn append_canonical_map(out: &mut String, values: &serde_json::Map<String, serde_json::Value>) {
    out.push_str("object:");
    out.push_str(&values.len().to_string());
    out.push(':');
    let mut entries: Vec<_> = values.iter().collect();
    entries.sort_unstable_by(|left, right| left.0.cmp(right.0));
    for (key, value) in entries {
        append_length_prefixed(out, key);
        append_canonical_value(out, value);
    }
}

fn validate_table_name<'a>(
    schema: &EntityExtractionSchema,
    table_kind: &str,
    name: &'a str,
    names: &mut HashSet<&'a str>,
) -> Result<(), PluginError> {
    if name.trim().is_empty() {
        return Err(validation_error(
            schema,
            format!("{table_kind} table name must not be empty"),
        ));
    }
    if !names.insert(name) {
        return Err(validation_error(
            schema,
            format!("duplicate table name `{name}`"),
        ));
    }
    Ok(())
}

fn validate_columns(
    schema: &EntityExtractionSchema,
    table_name: &str,
    columns: &[PropertyColumn],
) -> Result<(), PluginError> {
    let mut column_names = HashSet::new();
    for column in columns {
        if column.name.trim().is_empty() {
            return Err(validation_error(
                schema,
                format!("table `{table_name}` has an empty column name"),
            ));
        }
        if !column_names.insert(column.name.as_str()) {
            return Err(validation_error(
                schema,
                format!(
                    "table `{table_name}` has duplicate column `{}`",
                    column.name
                ),
            ));
        }
    }
    Ok(())
}

fn validation_error(schema: &EntityExtractionSchema, message: impl Into<String>) -> PluginError {
    PluginError::Validation {
        schema: schema.name.clone(),
        message: message.into(),
    }
}

#[cfg(all(test, feature = "plugin-entity-extraction"))]
mod tests {
    use super::*;
    use std::future::Future;
    use std::pin::Pin;

    use crate::plugins::memory::{MemoryConfig, MemoryPlugin, MergePolicy};

    fn contacts_schema() -> EntityExtractionSchema {
        EntityExtractionSchema {
            name: "contacts".into(),
            entities: vec![EntityTable {
                name: "person".into(),
                labels: vec!["Person".into()],
                identity_columns: vec!["email".into()],
                columns: vec![
                    PropertyColumn {
                        name: "email".into(),
                        property_type: PropertyType::String,
                        required: true,
                    },
                    PropertyColumn {
                        name: "name".into(),
                        property_type: PropertyType::String,
                        required: false,
                    },
                ],
                allow_model_properties: false,
            }],
            relationships: vec![],
        }
    }

    fn properties(
        entries: &[(&str, serde_json::Value)],
    ) -> serde_json::Map<String, serde_json::Value> {
        entries
            .iter()
            .map(|(name, value)| ((*name).into(), value.clone()))
            .collect()
    }

    fn work_schema() -> EntityExtractionSchema {
        let mut schema = contacts_schema();
        schema.entities.push(EntityTable {
            name: "company".into(),
            labels: vec!["Company".into()],
            identity_columns: vec!["domain".into()],
            columns: vec![PropertyColumn {
                name: "domain".into(),
                property_type: PropertyType::String,
                required: true,
            }],
            allow_model_properties: false,
        });
        schema.relationships.push(RelationshipTable {
            name: "works_for".into(),
            kind: "WORKS_FOR".into(),
            from_table: "person".into(),
            to_table: "company".into(),
            columns: vec![PropertyColumn {
                name: "role".into(),
                property_type: PropertyType::String,
                required: false,
            }],
            allow_model_properties: false,
        });
        schema
    }

    fn works_for_candidate() -> EntityExtractionCandidate {
        EntityExtractionCandidate {
            entities: vec![
                CandidateEntity {
                    table: "person".into(),
                    properties: properties(&[
                        ("email", serde_json::json!("ada@example.com")),
                        ("name", serde_json::json!("Ada")),
                    ]),
                },
                CandidateEntity {
                    table: "company".into(),
                    properties: properties(&[("domain", serde_json::json!("example.com"))]),
                },
            ],
            relationships: vec![CandidateRelationship {
                table: "works_for".into(),
                from: EntityReference {
                    table: "person".into(),
                    identity: properties(&[("email", serde_json::json!("ada@example.com"))]),
                },
                to: EntityReference {
                    table: "company".into(),
                    identity: properties(&[("domain", serde_json::json!("example.com"))]),
                },
                properties: properties(&[("role", serde_json::json!("engineer"))]),
            }],
        }
    }
    struct FixedModel(EntityExtractionCandidate);

    impl EntityExtractionModel for FixedModel {
        fn extract<'a>(
            &'a self,
            _source: &'a str,
            _schema: &'a EntityExtractionSchema,
        ) -> Pin<Box<dyn Future<Output = Result<EntityExtractionCandidate, PluginError>> + Send + 'a>>
        {
            Box::pin(async move { Ok(self.0.clone()) })
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn extracts_and_validates_a_supplied_model_candidate() {
        let candidate = works_for_candidate();
        let plugin = EntityExtractionPlugin::new(work_schema()).unwrap();

        let report = plugin
            .extract("Ada works for example.com.", &FixedModel(candidate.clone()))
            .await
            .unwrap();

        assert_eq!(report, plugin.validate_candidate(candidate).unwrap());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn invalid_model_extraction_leaves_memory_graph_unchanged() {
        let memory = MemoryPlugin::new(MemoryConfig::default()).unwrap();
        let initial = EntityExtractionPlugin::new(contacts_schema())
            .unwrap()
            .validate_candidate(EntityExtractionCandidate {
                entities: vec![CandidateEntity {
                    table: "person".into(),
                    properties: properties(&[("email", serde_json::json!("bea@example.com"))]),
                }],
                relationships: vec![],
            })
            .unwrap()
            .delta;
        memory.merge_graph(initial, MergePolicy::Keep).unwrap();
        let before = memory.graph().unwrap().clone();
        let invalid = EntityExtractionCandidate {
            entities: vec![CandidateEntity {
                table: "unknown".into(),
                properties: serde_json::Map::new(),
            }],
            relationships: vec![],
        };

        assert!(matches!(
            EntityExtractionPlugin::new(contacts_schema())
                .unwrap()
                .extract("Unknown entity.", &FixedModel(invalid))
                .await,
            Err(PluginError::Validation { .. })
        ));
        assert_eq!(*memory.graph().unwrap(), before);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn merges_valid_model_delta_into_memory_with_keep_policy() {
        let memory = MemoryPlugin::new(MemoryConfig::default()).unwrap();
        let report = EntityExtractionPlugin::new(work_schema())
            .unwrap()
            .extract(
                "Ada works for example.com.",
                &FixedModel(works_for_candidate()),
            )
            .await
            .unwrap();

        let merge = memory.merge_graph(report.delta, MergePolicy::Keep).unwrap();

        assert_eq!(merge.nodes_added, 2);
        assert_eq!(merge.relationships_added, 1);
        assert_eq!(memory.graph().unwrap().len(), 2);
        assert_eq!(memory.graph().unwrap().relationship_count(), 1);
    }

    #[test]
    fn produces_schema_typed_graph_delta_for_valid_candidate() {
        let candidate = works_for_candidate();
        let person_id = entity_id(
            "person",
            &properties(&[("email", serde_json::json!("ada@example.com"))]),
        );
        let company_id = entity_id("company", &candidate.entities[1].properties);
        let relationship_id = relationship_id(
            "works_for",
            &person_id,
            &company_id,
            &candidate.relationships[0].properties,
        );

        let report = EntityExtractionPlugin::new(work_schema())
            .unwrap()
            .validate_candidate(candidate)
            .unwrap();

        assert_eq!(report.nodes_accepted, 2);
        assert_eq!(report.relationships_accepted, 1);
        assert_eq!(
            report.delta.node(&person_id).unwrap().labels,
            vec!["Person"]
        );
        assert_eq!(
            report.delta.node(&person_id).unwrap().properties["name"],
            serde_json::json!("Ada")
        );
        let relationship = report.delta.relationship(&relationship_id).unwrap();
        assert_eq!(relationship.kind, "WORKS_FOR");
        assert_eq!(relationship.weight, 1.0);
    }

    #[test]
    fn report_round_trip_preserves_nonempty_graph_delta() {
        let report = EntityExtractionPlugin::new(work_schema())
            .unwrap()
            .validate_candidate(works_for_candidate())
            .unwrap();

        let encoded = serde_json::to_string(&report).unwrap();
        let decoded: EntityExtractionReport = serde_json::from_str(&encoded).unwrap();

        assert_eq!(decoded, report);
    }

    #[test]
    fn coalesces_equal_duplicate_entity_candidates() {
        let entity = CandidateEntity {
            table: "person".into(),
            properties: properties(&[("email", serde_json::json!("ada@example.com"))]),
        };
        let candidate = EntityExtractionCandidate {
            entities: vec![entity.clone(), entity],
            relationships: vec![],
        };

        let report = EntityExtractionPlugin::new(contacts_schema())
            .unwrap()
            .validate_candidate(candidate)
            .unwrap();

        assert_eq!(report.nodes_accepted, 1);
    }

    #[test]
    fn rejects_conflicting_entity_candidates_with_same_identity() {
        let candidate = EntityExtractionCandidate {
            entities: vec![
                CandidateEntity {
                    table: "person".into(),
                    properties: properties(&[
                        ("email", serde_json::json!("ada@example.com")),
                        ("name", serde_json::json!("Ada")),
                    ]),
                },
                CandidateEntity {
                    table: "person".into(),
                    properties: properties(&[
                        ("email", serde_json::json!("ada@example.com")),
                        ("name", serde_json::json!("Ada Lovelace")),
                    ]),
                },
            ],
            relationships: vec![],
        };

        assert!(matches!(
            EntityExtractionPlugin::new(contacts_schema())
                .unwrap()
                .validate_candidate(candidate),
            Err(PluginError::Validation { .. })
        ));
    }

    #[test]
    fn rejects_extra_properties_for_strict_tables() {
        let candidate = EntityExtractionCandidate {
            entities: vec![CandidateEntity {
                table: "person".into(),
                properties: properties(&[
                    ("email", serde_json::json!("ada@example.com")),
                    ("nickname", serde_json::json!("ada")),
                ]),
            }],
            relationships: vec![],
        };

        assert!(matches!(
            EntityExtractionPlugin::new(contacts_schema())
                .unwrap()
                .validate_candidate(candidate),
            Err(PluginError::Validation { .. })
        ));
    }

    #[test]
    fn retains_extra_properties_for_permissive_tables() {
        let mut schema = contacts_schema();
        schema.entities[0].allow_model_properties = true;
        let candidate = EntityExtractionCandidate {
            entities: vec![CandidateEntity {
                table: "person".into(),
                properties: properties(&[
                    ("email", serde_json::json!("ada@example.com")),
                    ("nickname", serde_json::json!("ada")),
                ]),
            }],
            relationships: vec![],
        };

        let report = EntityExtractionPlugin::new(schema)
            .unwrap()
            .validate_candidate(candidate)
            .unwrap();
        assert_eq!(
            report.delta.nodes().next().unwrap().properties["nickname"],
            "ada"
        );
    }

    #[test]
    fn rejects_incorrect_declared_json_type() {
        let candidate = EntityExtractionCandidate {
            entities: vec![CandidateEntity {
                table: "person".into(),
                properties: properties(&[("email", serde_json::json!(42))]),
            }],
            relationships: vec![],
        };

        assert!(matches!(
            EntityExtractionPlugin::new(contacts_schema())
                .unwrap()
                .validate_candidate(candidate),
            Err(PluginError::Validation { .. })
        ));
    }

    #[test]
    fn rejects_entity_missing_its_identity() {
        let candidate = EntityExtractionCandidate {
            entities: vec![CandidateEntity {
                table: "person".into(),
                properties: properties(&[("name", serde_json::json!("Ada"))]),
            }],
            relationships: vec![],
        };

        assert!(matches!(
            EntityExtractionPlugin::new(contacts_schema())
                .unwrap()
                .validate_candidate(candidate),
            Err(PluginError::Validation { .. })
        ));
    }

    #[test]
    fn rejects_unknown_entity_table_candidate() {
        let candidate = EntityExtractionCandidate {
            entities: vec![CandidateEntity {
                table: "unknown".into(),
                properties: serde_json::Map::new(),
            }],
            relationships: vec![],
        };

        assert!(matches!(
            EntityExtractionPlugin::new(contacts_schema())
                .unwrap()
                .validate_candidate(candidate),
            Err(PluginError::Validation { .. })
        ));
    }

    #[test]
    fn rejects_unknown_relationship_table_candidate() {
        let candidate = EntityExtractionCandidate {
            entities: vec![],
            relationships: vec![CandidateRelationship {
                table: "unknown".into(),
                from: EntityReference {
                    table: "person".into(),
                    identity: properties(&[("email", serde_json::json!("ada@example.com"))]),
                },
                to: EntityReference {
                    table: "person".into(),
                    identity: properties(&[("email", serde_json::json!("bea@example.com"))]),
                },
                properties: serde_json::Map::new(),
            }],
        };

        assert!(matches!(
            EntityExtractionPlugin::new(contacts_schema())
                .unwrap()
                .validate_candidate(candidate),
            Err(PluginError::Validation { .. })
        ));
    }

    #[test]
    fn rejects_relationship_endpoints_with_wrong_declared_tables() {
        let mut candidate = works_for_candidate();
        candidate.relationships[0].from.table = "company".into();
        candidate.relationships[0].from.identity =
            properties(&[("domain", serde_json::json!("example.com"))]);

        assert!(matches!(
            EntityExtractionPlugin::new(work_schema())
                .unwrap()
                .validate_candidate(candidate),
            Err(PluginError::Validation { .. })
        ));
    }

    #[test]
    fn rejects_identity_column_that_is_not_declared_required() {
        let schema = EntityExtractionSchema {
            name: "contacts".into(),
            entities: vec![EntityTable {
                name: "person".into(),
                labels: vec!["Person".into()],
                identity_columns: vec!["email".into()],
                columns: vec![PropertyColumn {
                    name: "email".into(),
                    property_type: PropertyType::String,
                    required: false,
                }],
                allow_model_properties: false,
            }],
            relationships: vec![],
        };

        assert!(matches!(
            EntityExtractionPlugin::new(schema),
            Err(PluginError::Validation { .. })
        ));
    }

    #[test]
    fn rejects_relationship_with_unknown_endpoint_table() {
        let mut schema = contacts_schema();
        schema.relationships.push(RelationshipTable {
            name: "works_for".into(),
            kind: "WORKS_FOR".into(),
            from_table: "person".into(),
            to_table: "company".into(),
            columns: vec![],
            allow_model_properties: false,
        });

        assert!(matches!(
            EntityExtractionPlugin::new(schema),
            Err(PluginError::Validation { .. })
        ));
    }
}
