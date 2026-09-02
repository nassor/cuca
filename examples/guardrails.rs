//! Validate live tool calls against JSON Schemas and watch the diagnostic
//! replace one in the stream.
//!
//! One `book_flight` tool is offered to the model on three real turns, each
//! through a differently configured `JsonGuardrailPlugin`. The first guardrail
//! enforces the schema the tool advertises, so the model's call passes and
//! reaches the caller untouched. The second enforces a policy that also demands
//! a `passenger`, which no call to this tool can carry, so `on_stream_chunk`
//! replaces the `ToolCall` block with a `ToolResult` carrying the diagnostic the
//! model would read on its next turn. The third enforces the same policy with a
//! retry budget of zero, which is the `"guardrail_exhausted"` path. Every turn
//! prints the blocks that reached the caller, the tracked-call gauge, and the
//! retry event.
//!
//! # Prerequisites
//!
//! - A checkout of this repository (the example builds from this crate).
//! - A running [llama.cpp](https://github.com/ggml-org/llama.cpp) server
//!   (`llama-server`) on port 1234 with the demo model loaded. The model must
//!   support tool calling.
//!
//! # Run
//!
//! ```sh
//! cargo run --example guardrails --features provider-llamacpp,plugin-guardrails
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
//! Example: `CUCA_BASE_URL=http://127.0.0.1:8000/v1 CUCA_MODEL=<server-model-id> cargo run --example guardrails --features provider-llamacpp,plugin-guardrails`
//!
//! # Output
//!
//! From one run against `google/gemma-4-12b-qat` on llama.cpp:
//!
//! ```text
//! Tool book_flight advertises: {"additionalProperties":false,"properties":{"date":{"pattern":"^\\d{4}-\\d{2}-\\d{2}$","type":"string"},"destination":{"type":"string"}},"required":["destination","date"],"type":"object"}
//! Guardrail policy for the last two turns: {"properties":{"date":{"pattern":"^\\d{4}-\\d{2}-\\d{2}$","type":"string"},"destination":{"type":"string"},"passenger":{"type":"string"}},"required":["destination","date","passenger"],"type":"object"}
//!
//! Policy matches the tool, retry budget 3: "Book a flight to Lisbon for 2026-03-14 with the book_flight tool."
//!   thinking blocks: 117
//!   ToolCall call="0UNMlIsYtMPi2o1N6r910nE645OFrWuD" name="book_flight" passed the schema
//!     {"date":"2026-03-14","destination":"Lisbon"}
//!   tracked_calls: 0
//!   last_retry_event: none, nothing was rewritten
//!
//! Policy demands a passenger, retry budget 3: "Book a flight to Porto for 2026-04-02 with the book_flight tool."
//!   thinking blocks: 87
//!   ToolResult call="bgzkJZOykk62A9zcSTMnE3X7YHgXQEhe"
//!     {"error":"schema_validation_failed","issues":["\"passenger\" is a required property"],"tool":"book_flight"}
//!   tracked_calls: 1
//!   last_retry_event: schema "book_flight", error "schema_validation_failed", attempt 1
//!
//! Same policy, retry budget 0: "Book a flight to Faro for 2026-05-20 with the book_flight tool."
//!   thinking blocks: 107
//!   ToolResult call="g1zw5RS7PUVPuVM5o4xM7or2ZerArLPu"
//!     {"error":"guardrail_exhausted","issues":["\"passenger\" is a required property"],"tool":"book_flight"}
//!   tracked_calls: 1
//!   last_retry_event: schema "book_flight", error "guardrail_exhausted", attempt 1
//! ```
//!
//! The first turn is the pass-through: `tracked_calls` stays `0`, because the
//! plugin only tracks a call id once that call has failed. The second is the
//! retry path: the attempt count is inside the budget, so the diagnostic says
//! `schema_validation_failed` and the model is expected to correct itself on the
//! next turn. The third is the bound: a zero retry budget means the first
//! failure is already past it, so the plugin stops asking and says
//! `guardrail_exhausted`. Both diagnostics carry the same `issues`, so the
//! `error` field is the only thing the budget changes.
//!
//! The tool-call ids and the thinking counts depend on the model, as does
//! whether it emits a valid ISO date at all. The diagnostic shapes, the attempt
//! counts, and the gauge do not.
//!
//! With no server on the base URL, the program prints one line naming the
//! address and exits successfully.
//!
//! # Why a diagnostic block and not an error
//!
//! `on_stream_chunk` returning `Err` would drop the block and leave the caller
//! nothing to feed back, so the plugin rewrites the block instead: a
//! `ToolResult` for the same call id, carrying the schema issues as JSON. That
//! is a message the model can read on the next turn, which is what makes the
//! correction loop self-contained. The bound exists because the loop would
//! otherwise never end: a model that cannot satisfy the schema would be asked
//! forever.

use std::collections::HashMap;
use std::sync::Arc;

use cuca::plugin::CucaPlugin;
use cuca::types::{MessageContentBlock, ProviderEndpoint, ToolDefinition};
use cuca::{CucaClient, JsonGuardrailPlugin, UnifiedRequest};
use serde_json::json;
use tokio_stream::StreamExt;

/// Completion cap for a tool-calling turn. The reasoning blocks a 12b model
/// emits before the call count against it, so a smaller budget returns the
/// thinking and no call.
const MAX_TOKENS: u32 = 512;

/// The tool the model is offered, in every turn. Its `input_schema` is the
/// contract the model sees; which schema a guardrail enforces is a separate
/// caller decision, which is what the later turns exercise.
fn book_flight() -> ToolDefinition {
    ToolDefinition {
        name: "book_flight".to_string(),
        description: "Book one flight to a destination on a calendar date.".to_string(),
        input_schema: tool_schema(),
    }
}

/// The tool's own contract: a destination and an ISO calendar date.
fn tool_schema() -> serde_json::Value {
    json!({
        "type": "object",
        "required": ["destination", "date"],
        "properties": {
            "destination": { "type": "string" },
            "date": { "type": "string", "pattern": "^\\d{4}-\\d{2}-\\d{2}$" }
        },
        "additionalProperties": false
    })
}

/// A policy stricter than the tool advertises: it also demands a `passenger`,
/// which the tool schema forbids. Schema drift between a tool definition and
/// the guardrail policy is the failure this plugin exists to catch, and no
/// model output can satisfy both.
fn enforced_schema() -> serde_json::Value {
    json!({
        "type": "object",
        "required": ["destination", "date", "passenger"],
        "properties": {
            "destination": { "type": "string" },
            "date": { "type": "string", "pattern": "^\\d{4}-\\d{2}-\\d{2}$" },
            "passenger": { "type": "string" }
        }
    })
}

/// Dispatch one turn and print every block the guardrail let through.
///
/// The `ToolResult` blocks here were `ToolCall` blocks until `on_stream_chunk`
/// replaced them: the caller never sees the invalid call.
async fn turn(
    base_url: &str,
    client: &CucaClient,
    guardrails: &JsonGuardrailPlugin,
    model: &str,
    label: &str,
    prompt: &str,
) -> Result<bool, Box<dyn std::error::Error>> {
    println!("\n{label}: {prompt:?}");
    let request = UnifiedRequest::new(model)
        .add_user_message(prompt)
        .add_tool(book_flight())
        .set_max_tokens(MAX_TOKENS);
    let mut stream = match client.generate_stream(request).await {
        Ok(stream) => stream,
        Err(error) => {
            println!("\nNo server answered at {base_url}: {error}");
            println!("Start llama-server there, or set CUCA_BASE_URL, then run this again.");
            return Ok(false);
        }
    };
    let mut thinking_blocks = 0usize;
    let mut text = String::new();
    while let Some(chunk) = stream.next().await {
        match chunk {
            Ok(MessageContentBlock::Thinking { .. }) => thinking_blocks += 1,
            Ok(MessageContentBlock::Text(chunk_text)) => text.push_str(&chunk_text),
            Ok(MessageContentBlock::ToolCall {
                id,
                name,
                arguments,
            }) => {
                println!("  thinking blocks: {thinking_blocks}");
                println!("  ToolCall call={id:?} name={name:?} passed the schema");
                println!("    {arguments}");
            }
            Ok(MessageContentBlock::ToolResult {
                tool_call_id,
                output,
            }) => {
                println!("  thinking blocks: {thinking_blocks}");
                println!("  ToolResult call={tool_call_id:?}");
                println!("    {output}");
            }
            Ok(_) => {}
            Err(error) => {
                println!("  the stream ended early: {error}");
                break;
            }
        }
    }
    if !text.trim().is_empty() {
        println!("  text: {}", text.trim());
    }
    println!("  tracked_calls: {}", guardrails.tracked_calls());
    match guardrails.last_retry_event() {
        Some((schema, error, attempt)) => {
            println!("  last_retry_event: schema {schema:?}, error {error:?}, attempt {attempt}")
        }
        None => println!("  last_retry_event: none, nothing was rewritten"),
    }
    Ok(true)
}

/// A client with one guardrail plugin registered.
fn client_with(
    base_url: &str,
    guardrails: &Arc<JsonGuardrailPlugin>,
) -> Result<CucaClient, Box<dyn std::error::Error>> {
    Ok(CucaClient::builder()
        .with_provider(ProviderEndpoint::LlamaCpp)
        .with_base_url(base_url.to_string())
        .register_plugin(Arc::clone(guardrails) as Arc<dyn CucaPlugin>)
        .build()?)
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Base URL and model come from the environment so the example runs
    // against any OpenAI-compatible server; the defaults target a local
    // llama.cpp server (see the module docs for the override recipe).
    let base_url =
        std::env::var("CUCA_BASE_URL").unwrap_or_else(|_| "http://127.0.0.1:1234/v1".to_string());
    let model = std::env::var("CUCA_MODEL").unwrap_or_else(|_| "google/gemma-4-e4b".to_string());

    println!("Tool book_flight advertises: {}", tool_schema());
    println!(
        "Guardrail policy for the last two turns: {}",
        enforced_schema()
    );

    // Three plugins, not one with three schemas: the retry budget is per
    // plugin, so a policy and its budget travel together.
    let matching = Arc::new(JsonGuardrailPlugin::with_schemas(
        HashMap::from([("book_flight".to_string(), tool_schema())]),
        3,
    )?);
    let reachable = turn(
        &base_url,
        &client_with(&base_url, &matching)?,
        &matching,
        &model,
        "Policy matches the tool, retry budget 3",
        "Book a flight to Lisbon for 2026-03-14 with the book_flight tool.",
    )
    .await?;
    if !reachable {
        return Ok(());
    }

    let retrying = Arc::new(JsonGuardrailPlugin::with_schemas(
        HashMap::from([("book_flight".to_string(), enforced_schema())]),
        3,
    )?);
    turn(
        &base_url,
        &client_with(&base_url, &retrying)?,
        &retrying,
        &model,
        "Policy demands a passenger, retry budget 3",
        "Book a flight to Porto for 2026-04-02 with the book_flight tool.",
    )
    .await?;

    let exhausted = Arc::new(JsonGuardrailPlugin::with_schemas(
        HashMap::from([("book_flight".to_string(), enforced_schema())]),
        0,
    )?);
    turn(
        &base_url,
        &client_with(&base_url, &exhausted)?,
        &exhausted,
        &model,
        "Same policy, retry budget 0",
        "Book a flight to Faro for 2026-05-20 with the book_flight tool.",
    )
    .await?;
    Ok(())
}
