//! Extract a typed graph from prose, merge it into memory, and answer from
//! the merged graph.
//!
//! Stage 1 asks a live model for an org chart and validates its answer against
//! an `EntityExtractionSchema`, so the accepted delta is schema-typed rather
//! than whatever JSON the model felt like emitting. Stage 2 performs the
//! mandatory hand-off: the report's `delta` is inert until
//! `MemoryPlugin::merge_graph` applies it. Stage 3 renders the merged graph
//! into the next request and asks a question whose answer exists only in the
//! graph, since the source prose is never sent again.
//!
//! # Prerequisites
//!
//! - A checkout of this repository (the example builds from this crate).
//! - A running [llama.cpp](https://github.com/ggml-org/llama.cpp) server
//!   (`llama-server`) on port 1234 with the demo model loaded.
//!
//! # Run
//!
//! ```sh
//! cargo run --example entity_extraction --features provider-llamacpp,service-entity-extraction
//! ```
//!
//! # Configuration
//!
//! Both values default to a local llama.cpp server; override them to target
//! any OpenAI-compatible server:
//!
//! - `CUCA_BASE_URL`: server base URL, defaults to `http://127.0.0.1:1234/v1`.
//! - `CUCA_MODEL`: upstream model id, defaults to `google/gemma-4-e4b`.
//!
//! Example: `CUCA_BASE_URL=http://127.0.0.1:8000/v1 CUCA_MODEL=<server-model-id> cargo run --example entity_extraction --features provider-llamacpp,service-entity-extraction`
//!
//! # Output
//!
//! One run against `google/gemma-4-12b-qat` on llama.cpp:
//!
//! ```text
//! Schema `org-chart`: entity tables [person, company], relationship tables [works_at]
//!
//! Stage 1: one live extraction turn
//!   the model answered with 326 thinking blocks and 166 chars of JSON
//!   accepted 4 nodes, 2 relationships
//!   node entity:6:person:object:1:4:namestring:12:Grace Hopper [person] {"name":"Grace Hopper","title":"rear admiral"}
//!   node entity:6:person:object:1:4:namestring:12:Ada Lovelace [person] {"name":"Ada Lovelace","title":"mathematician"}
//!   node entity:7:company:object:1:4:namestring:13:Naval Systems [company] {"name":"Naval Systems"}
//!   node entity:7:company:object:1:4:namestring:18:Analytical Engines [company] {"name":"Analytical Engines"}
//!   edge entity:6:person:object:1:4:namestring:12:Grace Hopper -[works_at]-> entity:7:company:object:1:4:namestring:13:Naval Systems
//!   edge entity:6:person:object:1:4:namestring:12:Ada Lovelace -[works_at]-> entity:7:company:object:1:4:namestring:18:Analytical Engines
//!
//! Stage 2: the mandatory hand-off (MemoryPlugin::merge_graph)
//!   graph before the merge: 0 nodes, 0 relationships
//!   merge report: 4 added, 0 overwritten, 0 kept, 2 edges added, 0 renamed
//!   graph after the merge: 4 nodes, 2 relationships
//!
//! Stage 3: the graph rendered into the next request
//!   CUCA graph memory: 4 nodes, 2 relationships
//!   node entity:6:person:object:1:4:namestring:12:Ada Lovelace: labels=[person] props={"name":"Ada Lovelace","title":"mathematician"}
//!   node entity:6:person:object:1:4:namestring:12:Grace Hopper: labels=[person] props={"name":"Grace Hopper","title":"rear admiral"}
//!   node entity:7:company:object:1:4:namestring:13:Naval Systems: labels=[company] props={"name":"Naval Systems"}
//!   node entity:7:company:object:1:4:namestring:18:Analytical Engines: labels=[company] props={"name":"Analytical Engines"}
//!   rel relationship:8:works_at53:entity:6:person:object:1:4:namestring:12:Ada Lovelace60:entity:7:company:object:1:4:namestring:18:Analytical Enginesobject:0:: entity:6:person:object:1:4:namestring:12:Ada Lovelace -[works_at]-> entity:7:company:object:1:4:namestring:18:Analytical Engines weight=1
//!   rel relationship:8:works_at53:entity:6:person:object:1:4:namestring:12:Grace Hopper55:entity:7:company:object:1:4:namestring:13:Naval Systemsobject:0:: entity:6:person:object:1:4:namestring:12:Grace Hopper -[works_at]-> entity:7:company:object:1:4:namestring:13:Naval Systems weight=1
//!   asked: Which company does Grace Hopper work at, and what is her title?
//!   reply: Grace Hopper works at Naval Systems and her title is rear admiral.
//! ```
//!
//! Stage 3 is the point of the demo: the source prose is never in that
//! request, so the reply can only come from the rendered graph.
//!
//! The thinking-block count, the JSON length and the final reply depend on the
//! model. The node ids do not: they are a length-prefixed canonical encoding
//! of the table name plus the identity columns, so the same extraction always
//! produces the same ids, and a relationship id folds in both endpoints and
//! its own properties.
//!
//! With no server on the base URL, the program prints one line naming the
//! address and exits successfully.
//!
//! # Why a service and not a plugin hook?
//!
//! `EntityExtractor` is a service, not a `CucaPlugin`. Extraction is a turn of
//! its own against a model of the caller's choosing, and its product is a
//! graph, which no hook signature can return. The hand-off in stage 2 is the
//! reason: the delta is a standalone `MemoryGraph` that the application
//! merges, so a hook that silently merged it would take the merge policy away
//! from the caller.

use std::pin::Pin;

use cuca::plugin::CucaPlugin;
use cuca::types::{MessageContentBlock, ProviderEndpoint};
use cuca::{
    CandidateEntity, CandidateRelationship, CucaClient, EntityExtractionCandidate,
    EntityExtractionModel, EntityExtractionSchema, EntityExtractor, EntityReference, EntityTable,
    GRAPH_RENDER_MARKER, GraphContextConfig, MemoryConfig, MemoryPlugin, MergePolicy, PluginError,
    PropertyColumn, PropertyType, RelationshipTable, UnifiedRequest,
};
use tokio_stream::StreamExt;

/// The prose the extraction reads. It is never sent again after stage 1.
const SOURCE: &str = "Ada Lovelace is a mathematician at Analytical Engines. \
Grace Hopper is a rear admiral at Naval Systems. \
Analytical Engines and Naval Systems are separate companies.";

/// The flat shape the adapter asks for. Keeping the wire shape independent of
/// the schema is deliberate: local models are unreliable at deep nested JSON,
/// and mapping a flat reply into schema rows is the application-side work the
/// explicit-call contract expects.
const WIRE_SHAPE: &str = r#"{"people":[{"name":"...","employer":"...","title":"..."}]}"#;

/// `person -[works_at]-> company`, both identified by `name`. Both tables are
/// strict, so any property the model invents is rejected, not absorbed.
fn org_schema() -> EntityExtractionSchema {
    let required_name = PropertyColumn {
        name: "name".into(),
        property_type: PropertyType::String,
        required: true,
    };
    EntityExtractionSchema {
        name: "org-chart".into(),
        entities: vec![
            EntityTable {
                name: "person".into(),
                labels: vec!["person".into()],
                identity_columns: vec!["name".into()],
                columns: vec![
                    required_name.clone(),
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
                columns: vec![required_name],
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

/// The single-column identity map both entity tables use.
fn identity(name: &str) -> serde_json::Map<String, serde_json::Value> {
    let mut map = serde_json::Map::new();
    map.insert("name".into(), serde_json::json!(name));
    map
}

/// A real `EntityExtractionModel` backed by llama.cpp through this crate's own
/// client.
struct LiveModel {
    client: CucaClient,
    model: String,
    /// Thinking blocks the last call produced, counted rather than printed.
    thinking: std::sync::atomic::AtomicUsize,
}

impl EntityExtractionModel for LiveModel {
    fn extract<'a>(
        &'a self,
        source: &'a str,
        schema: &'a EntityExtractionSchema,
    ) -> Pin<Box<dyn Future<Output = Result<EntityExtractionCandidate, PluginError>> + Send + 'a>>
    {
        Box::pin(async move {
            let request = UnifiedRequest::new(&self.model)
                .add_system_message(
                    "You are a JSON extraction engine. You reply with a single JSON object and \
                     nothing else: no prose, no explanation, no markdown fences.",
                )
                .add_user_message(format!(
                    "Extract every person, the company they work at, and their job title from \
                     the text. Reply with exactly this shape:\n{WIRE_SHAPE}\nCopy names verbatim \
                     from the text.\n\nText:\n{source}"
                ))
                .set_temperature(0.0)
                .set_max_tokens(768);
            let mut stream = self
                .client
                .generate_stream(request)
                .await
                .map_err(|error| {
                    PluginError::Internal(format!("extraction turn failed: {error}"))
                })?;
            let mut reply = String::new();
            while let Some(chunk) = stream.next().await {
                match chunk {
                    Ok(MessageContentBlock::Text(text)) => reply.push_str(&text),
                    Ok(MessageContentBlock::Thinking { .. }) => {
                        self.thinking
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                    Ok(_) => {}
                    Err(error) => {
                        return Err(PluginError::Internal(format!("stream error: {error}")));
                    }
                }
            }
            let reply = reply.trim();
            println!(
                "  the model answered with {} thinking blocks and {} chars of JSON",
                self.thinking.load(std::sync::atomic::Ordering::Relaxed),
                reply.len()
            );
            candidate_from(reply).ok_or_else(|| {
                PluginError::Internal(format!(
                    "no parseable extraction for schema `{}` in: {reply}",
                    schema.name
                ))
            })
        })
    }
}

/// Map the model's flat reply into schema rows, or `None` when nothing parses.
///
/// The adapter never proposes a row the schema cannot accept: it drops
/// non-string values and dedups by identity, so a validation failure here
/// would be a real contract break rather than an unhelpful model.
fn candidate_from(reply: &str) -> Option<EntityExtractionCandidate> {
    let (start, end) = (reply.find('{')?, reply.rfind('}')?);
    let value: serde_json::Value = serde_json::from_str(reply.get(start..=end)?).ok()?;
    let mut candidate = EntityExtractionCandidate {
        entities: Vec::new(),
        relationships: Vec::new(),
    };
    let mut companies: Vec<String> = Vec::new();
    let mut people: Vec<String> = Vec::new();
    for row in value.get("people")?.as_array()?.iter().take(8) {
        let field = |key: &str| row.get(key).and_then(serde_json::Value::as_str);
        let Some(name) = field("name").map(str::trim).filter(|n| !n.is_empty()) else {
            continue;
        };
        if people.iter().any(|seen| seen == name) {
            continue;
        }
        people.push(name.to_string());
        let mut properties = identity(name);
        if let Some(title) = field("title").filter(|t| !t.is_empty()) {
            properties.insert("title".into(), serde_json::json!(title));
        }
        candidate.entities.push(CandidateEntity {
            table: "person".into(),
            properties,
        });
        let Some(employer) = field("employer").map(str::trim).filter(|e| !e.is_empty()) else {
            continue;
        };
        if !companies.iter().any(|seen| seen == employer) {
            companies.push(employer.to_string());
            candidate.entities.push(CandidateEntity {
                table: "company".into(),
                properties: identity(employer),
            });
        }
        candidate.relationships.push(CandidateRelationship {
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
    (!candidate.entities.is_empty()).then_some(candidate)
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Base URL and model come from the environment so the example runs
    // against any OpenAI-compatible server; the defaults target a local
    // llama.cpp server (see the module docs for the override recipe).
    let base_url =
        std::env::var("CUCA_BASE_URL").unwrap_or_else(|_| "http://127.0.0.1:1234/v1".to_string());
    let model = std::env::var("CUCA_MODEL").unwrap_or_else(|_| "google/gemma-4-e4b".to_string());

    let client = CucaClient::builder()
        .with_provider(ProviderEndpoint::LlamaCpp)
        .with_base_url(base_url.clone())
        .build()?;
    let extractor = EntityExtractor::new(org_schema())?;
    let schema = org_schema();
    println!(
        "Schema `{}`: entity tables [{}], relationship tables [{}]",
        schema.name,
        schema
            .entities
            .iter()
            .map(|table| table.name.as_str())
            .collect::<Vec<_>>()
            .join(", "),
        schema
            .relationships
            .iter()
            .map(|table| table.name.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    );

    println!("\nStage 1: one live extraction turn");
    let live = LiveModel {
        client,
        model: model.clone(),
        thinking: std::sync::atomic::AtomicUsize::new(0),
    };
    let report = match extractor.extract(SOURCE, &live).await {
        Ok(report) => report,
        Err(error) => {
            println!("\nNo usable extraction from {base_url}: {error}");
            println!("Start llama-server there, or set CUCA_BASE_URL, then run this again.");
            return Ok(());
        }
    };
    println!(
        "  accepted {} nodes, {} relationships",
        report.nodes_accepted, report.relationships_accepted
    );
    for node in report.delta.nodes() {
        let properties = serde_json::to_string(&node.properties)?;
        println!(
            "  node {} [{}] {properties}",
            node.id,
            node.labels.join(",")
        );
    }
    for edge in report.delta.relationships() {
        println!("  edge {} -[{}]-> {}", edge.from, edge.kind, edge.to);
    }

    // Stage 2: the hand-off. The extractor never touched a MemoryPlugin, so
    // without this call the whole extraction is discarded when the report
    // drops.
    println!("\nStage 2: the mandatory hand-off (MemoryPlugin::merge_graph)");
    let memory = MemoryPlugin::new(MemoryConfig {
        graph_context: Some(GraphContextConfig::default()),
        ..Default::default()
    })?;
    // One guard per reading: `graph()` hands back a `MutexGuard`, so two calls
    // inside one expression would hold two guards on the same mutex and
    // deadlock.
    let size = |label: &str| -> Result<(), PluginError> {
        let graph = memory.graph()?;
        println!(
            "  graph {label} the merge: {} nodes, {} relationships",
            graph.len(),
            graph.relationship_count()
        );
        Ok(())
    };
    size("before")?;
    let merge = memory.merge_graph(report.delta, MergePolicy::Overwrite)?;
    println!(
        "  merge report: {} added, {} overwritten, {} kept, {} edges added, {} renamed",
        merge.nodes_added,
        merge.nodes_overwritten,
        merge.nodes_kept,
        merge.relationships_added,
        merge.relationships_renamed
    );
    size("after")?;

    // Stage 3: `on_request` is called here rather than through a registered
    // plugin so the printed message is exactly the bytes that go out; a
    // registered `MemoryPlugin` performs the same injection inside
    // `generate_stream`.
    println!("\nStage 3: the graph rendered into the next request");
    let question = "Which company does Grace Hopper work at, and what is her title?";
    let mut request = UnifiedRequest::new(&model)
        .add_system_message("You are concise. Answer from the graph memory only.")
        .add_user_message(question)
        .set_max_tokens(512);
    memory.on_request(&mut request)?;
    for message in &request.messages {
        for block in &message.content {
            if let MessageContentBlock::Text(text) = block
                && text.starts_with(GRAPH_RENDER_MARKER)
            {
                for line in text.lines() {
                    println!("  {line}");
                }
            }
        }
    }
    println!("  asked: {question}");
    let client = CucaClient::builder()
        .with_provider(ProviderEndpoint::LlamaCpp)
        .with_base_url(base_url)
        .build()?;
    let mut stream = client.generate_stream(request).await?;
    let mut reply = String::new();
    while let Some(chunk) = stream.next().await {
        match chunk {
            Ok(MessageContentBlock::Text(text)) => reply.push_str(&text),
            Ok(_) => {}
            Err(error) => {
                println!("  the stream ended early: {error}");
                break;
            }
        }
    }
    println!("  reply: {}", reply.trim());

    Ok(())
}
