//! Live entity-extraction adapter and its schema, shared by
//! `tests/service_entity_extraction.rs` and `tests/plugin_combinations.rs`.
//!
//! Integration test binaries can only share code through the `common` module
//! tree, and the adapter is the one non-trivial piece both the per-service
//! suite and the extraction→memory→prompt combination test need.
//!
//! Gated on `service-entity-extraction` (which enables `plugin-memory` via
//! Cargo), so the rest of `common` stays feature-neutral.

use std::pin::Pin;
use std::sync::Mutex;
use std::time::Duration;

use cuca::types::MessageContentBlock;
use cuca::{
    CandidateEntity, CandidateRelationship, CucaClient, EntityExtractionCandidate,
    EntityExtractionModel, EntityExtractionSchema, EntityExtractor, EntityReference, EntityTable,
    PluginError, PropertyColumn, PropertyType, RelationshipTable, UnifiedRequest,
};
use tokio_stream::StreamExt;

/// The schema under test: `person -[works_at]-> company`, both identified by
/// `name`, with an optional non-identity `title` column on `person`. Both
/// tables are strict (`allow_model_properties: false`), so any undeclared
/// property the model invents is rejected rather than absorbed.
pub fn org_schema() -> EntityExtractionSchema {
    EntityExtractionSchema {
        name: "org-chart".into(),
        entities: vec![
            EntityTable {
                name: "person".into(),
                labels: vec!["person".into()],
                identity_columns: vec!["name".into()],
                columns: vec![
                    PropertyColumn {
                        name: "name".into(),
                        property_type: PropertyType::String,
                        required: true,
                    },
                    PropertyColumn {
                        name: "title".into(),
                        property_type: PropertyType::String,
                        required: false,
                    },
                ],
                allow_model_properties: false,
            },
            EntityTable {
                name: "company".into(),
                labels: vec!["company".into()],
                identity_columns: vec!["name".into()],
                columns: vec![PropertyColumn {
                    name: "name".into(),
                    property_type: PropertyType::String,
                    required: true,
                }],
                allow_model_properties: false,
            },
        ],
        relationships: vec![RelationshipTable {
            name: "works_at".into(),
            kind: "works_at".into(),
            from_table: "person".into(),
            to_table: "company".into(),
            columns: vec![],
            allow_model_properties: false,
        }],
    }
}

/// [`org_schema`] wrapped in a validated [`EntityExtractor`].
pub fn org_extractor() -> EntityExtractor {
    EntityExtractor::new(org_schema()).expect("org-chart schema must validate")
}

/// The single-column identity map both entity tables use.
pub fn identity(name: &str) -> serde_json::Map<String, serde_json::Value> {
    let mut map = serde_json::Map::new();
    map.insert("name".into(), serde_json::json!(name));
    map
}

/// One `person -[works_at]-> company` pair, both identified by `name`.
pub fn pair_candidate(person: &str, company: &str) -> EntityExtractionCandidate {
    EntityExtractionCandidate {
        entities: vec![
            CandidateEntity {
                table: "person".into(),
                properties: identity(person),
            },
            CandidateEntity {
                table: "company".into(),
                properties: identity(company),
            },
        ],
        relationships: vec![CandidateRelationship {
            table: "works_at".into(),
            from: EntityReference {
                table: "person".into(),
                identity: identity(person),
            },
            to: EntityReference {
                table: "company".into(),
                identity: identity(company),
            },
            properties: serde_json::Map::new(),
        }],
    }
}

/// The extraction the adapter asks the served model for: a flat people list.
/// Keeping the wire shape independent of (and much smaller than) the schema is
/// deliberate: small local models are unreliable at deep nested JSON, and
/// mapping a flat reply into schema rows is exactly the application-side work
/// the explicit-call contract expects of the adapter.
const WIRE_SHAPE: &str = r#"{"people":[{"name":"...","employer":"...","title":"..."}]}"#;

/// Source text for the live extraction.
pub const SOURCE: &str = "Ada Lovelace is a mathematician at Analytical Engines. \
Grace Hopper is a rear admiral at Naval Systems. \
Analytical Engines and Naval Systems are separate companies.";

/// A real [`EntityExtractionModel`] backed by llama.cpp through this crate's
/// own client.
///
/// The adapter never proposes a row the schema cannot accept: it drops
/// non-string values, ignores blank names, dedups people and companies by
/// identity (so no two candidate rows can share a derived id with different
/// properties), and caps the row count. Consequently a validation failure
/// after a candidate was produced is a real contract break, which is what
/// [`Self::produced_no_candidate`] lets a test distinguish.
pub struct LiveExtractionModel {
    client: CucaClient,
    model: String,
    attempts: usize,
    /// The last candidate the adapter managed to build, if any.
    candidate: Mutex<Option<EntityExtractionCandidate>>,
    /// Raw model replies, for failure diagnostics.
    replies: Mutex<Vec<String>>,
}

impl LiveExtractionModel {
    pub fn new(model: String) -> Self {
        Self {
            client: super::client_with_plugins(Vec::new()),
            model,
            attempts: 2,
            candidate: Mutex::new(None),
            replies: Mutex::new(Vec::new()),
        }
    }

    /// One live round trip; returns the concatenated `Text` content.
    async fn ask(&self, source: &str, attempt: usize) -> Result<String, PluginError> {
        let insistence = if attempt == 1 {
            ""
        } else {
            "Your previous answer was not valid JSON. Output the JSON object only. "
        };
        let request = UnifiedRequest::new(self.model.clone())
            .add_system_message(
                "You are a JSON extraction engine. You reply with a single JSON object and \
                 nothing else: no prose, no explanation, no markdown fences.",
            )
            .add_user_message(format!(
                "{insistence}Extract every person, the company they work at, and their job \
                 title from the text. Reply with exactly this shape:\n{WIRE_SHAPE}\nCopy names \
                 verbatim from the text. Omit a field you cannot fill.\n\nText:\n{source}"
            ))
            .set_temperature(0.0)
            .set_max_tokens(512);

        let mut stream =
            self.client.generate_stream(request).await.map_err(|e| {
                PluginError::Internal(format!("live extraction request failed: {e}"))
            })?;
        let mut text = String::new();
        let collected = tokio::time::timeout(Duration::from_secs(120), async {
            while let Some(item) = stream.next().await {
                match item {
                    Ok(MessageContentBlock::Text(chunk)) => text.push_str(&chunk),
                    Ok(_) => {}
                    Err(e) => {
                        return Err(PluginError::Internal(format!(
                            "live extraction stream error: {e}"
                        )));
                    }
                }
            }
            Ok(())
        })
        .await;
        match collected {
            Ok(result) => result?,
            Err(_) => {
                return Err(PluginError::Internal(
                    "live extraction stream did not finish within 120s".into(),
                ));
            }
        }
        Ok(text)
    }

    /// How many parse attempts one `extract` call makes.
    pub fn attempts(&self) -> usize {
        self.attempts
    }

    /// The last candidate the adapter built, if any.
    pub fn candidate(&self) -> Option<EntityExtractionCandidate> {
        self.candidate
            .lock()
            .expect("candidate lock must not be poisoned")
            .clone()
    }

    /// True when no attempt produced a parseable extraction, i.e. the failure
    /// is model quality rather than a broken validation contract.
    pub fn produced_no_candidate(&self) -> bool {
        self.candidate
            .lock()
            .expect("candidate lock must not be poisoned")
            .is_none()
    }

    /// Raw model replies, joined, for failure messages.
    pub fn diagnostics(&self) -> String {
        self.replies
            .lock()
            .expect("replies lock must not be poisoned")
            .join("\n---\n")
    }
}

impl EntityExtractionModel for LiveExtractionModel {
    fn extract<'a>(
        &'a self,
        source: &'a str,
        schema: &'a EntityExtractionSchema,
    ) -> Pin<Box<dyn Future<Output = Result<EntityExtractionCandidate, PluginError>> + Send + 'a>>
    {
        Box::pin(async move {
            for attempt in 1..=self.attempts {
                let reply = self.ask(source, attempt).await?;
                self.replies
                    .lock()
                    .expect("replies lock must not be poisoned")
                    .push(reply.clone());
                let rows = parse_people(&reply);
                if !rows.is_empty() {
                    let candidate = candidate_from_rows(&rows);
                    *self
                        .candidate
                        .lock()
                        .expect("candidate lock must not be poisoned") = Some(candidate.clone());
                    return Ok(candidate);
                }
            }
            Err(PluginError::Internal(format!(
                "model produced no parseable extraction for schema `{}` in {} attempts",
                schema.name, self.attempts
            )))
        })
    }
}

/// One extracted row: person name, optional employer, optional title.
type PersonRow = (String, Option<String>, Option<String>);

/// The first JSON value embedded in `text` that parses, tried as an object
/// first and then as an array (models wrap replies in prose or fences).
fn embedded_json(text: &str) -> Option<serde_json::Value> {
    for (open, close) in [('{', '}'), ('[', ']')] {
        let (Some(start), Some(end)) = (text.find(open), text.rfind(close)) else {
            continue;
        };
        if start >= end {
            continue;
        }
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(&text[start..=end]) {
            return Some(value);
        }
    }
    None
}

/// Map a model reply into rows, tolerating the shapes small models actually
/// emit: the requested `{"people": [...]}`, a bare array, and the common
/// `company`/`works_at` synonyms for `employer`.
fn parse_people(text: &str) -> Vec<PersonRow> {
    let Some(value) = embedded_json(text) else {
        return Vec::new();
    };
    let people = match value {
        serde_json::Value::Array(items) => items,
        serde_json::Value::Object(mut map) => match map.remove("people") {
            Some(serde_json::Value::Array(items)) => items,
            _ => return Vec::new(),
        },
        _ => return Vec::new(),
    };

    let string_field = |row: &serde_json::Value, keys: &[&str]| {
        keys.iter()
            .filter_map(|key| row.get(*key))
            .filter_map(serde_json::Value::as_str)
            .map(str::trim)
            .find(|value| !value.is_empty())
            .map(str::to_owned)
    };

    let mut rows: Vec<PersonRow> = Vec::new();
    for row in people.iter().take(8) {
        let Some(name) = string_field(row, &["name"]) else {
            continue;
        };
        // Dedup by identity: two rows sharing a derived node id but carrying
        // different properties would be a conflicting candidate, which is the
        // application's bug, not the plugin's.
        if rows.iter().any(|(existing, _, _)| *existing == name) {
            continue;
        }
        let employer = string_field(row, &["employer", "company", "works_at", "organization"]);
        let title = string_field(row, &["title", "role", "job_title"]);
        rows.push((name, employer, title));
    }
    rows
}

/// Build a schema-conformant candidate from extracted rows.
fn candidate_from_rows(rows: &[PersonRow]) -> EntityExtractionCandidate {
    let mut entities = Vec::with_capacity(rows.len() * 2);
    let mut relationships = Vec::new();
    let mut companies: Vec<&str> = Vec::new();

    for (name, employer, title) in rows {
        let mut properties = identity(name);
        if let Some(title) = title {
            properties.insert("title".into(), serde_json::json!(title));
        }
        entities.push(CandidateEntity {
            table: "person".into(),
            properties,
        });
        let Some(employer) = employer else {
            continue;
        };
        if !companies.contains(&employer.as_str()) {
            companies.push(employer);
            entities.push(CandidateEntity {
                table: "company".into(),
                properties: identity(employer),
            });
        }
        relationships.push(CandidateRelationship {
            table: "works_at".into(),
            from: EntityReference {
                table: "person".into(),
                identity: identity(name),
            },
            to: EntityReference {
                table: "company".into(),
                identity: identity(employer),
            },
            properties: serde_json::Map::new(),
        });
    }

    EntityExtractionCandidate {
        entities,
        relationships,
    }
}
