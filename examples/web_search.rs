//! Resolve a model's `web_search` call against a live search backend.
//!
//! The model is given one tool, calls it, and the plugin performs the HTTPS
//! request on a short-lived thread while the stream pauses; the normalized
//! result comes back as a `ToolResult` and a second turn answers from it. Every
//! backend the plugin speaks to is a paid API, so the demo needs a key in
//! `CUCA_SEARCH_API_KEY` and prints one line and exits without one.
//!
//! # Prerequisites
//!
//! - A checkout of this repository (the example builds from this crate).
//! - A running [llama.cpp](https://github.com/ggml-org/llama.cpp) server
//!   (`llama-server`) on port 1234 with the demo model loaded.
//! - A [Firecrawl](https://firecrawl.dev) API key.
//!
//! # Run
//!
//! ```sh
//! cargo run --example web_search --features provider-llamacpp,plugin-web-search
//! ```
//!
//! # Configuration
//!
//! The first two values default to a local llama.cpp server; override them to
//! target any OpenAI-compatible server:
//!
//! - `CUCA_BASE_URL`: server base URL, defaults to `http://127.0.0.1:1234/v1`.
//! - `CUCA_MODEL`: upstream model id, defaults to `google/gemma-4-e4b`.
//! - `CUCA_SEARCH_API_KEY`: Firecrawl API key. Required; without it the program
//!   prints one line and exits successfully.
//! - `CUCA_SEARCH_BASE_URL`: search endpoint override, defaults to
//!   `https://api.firecrawl.dev`.
//!
//! # Output
//!
//! From one run against `google/gemma-4-12b-qat` on llama.cpp, with a
//! deliberately invalid key, which is the only key that was available:
//!
//! ```text
//! Backend Firecrawl, at most 3 results per search
//!
//! Turn 1: the model searches, and the pipeline pauses for the HTTP call
//!   tool result: internal plugin error: web search returned HTTP 401 Unauthorized: {"success":false,"error":"Unauthorized: Invalid token"}
//!   thinking blocks: 60
//!
//! No results, so there is no second turn
//!   a valid key makes this a JSON array of {title, url, snippet} objects
//! ```
//!
//! Every line above is from a real run: a real turn, a real tool call, and a
//! real HTTPS request to `https://api.firecrawl.dev/v1/search`. With a valid
//! key the `ToolResult` instead carries a JSON array of at most `max_results`
//! `SearchResult` objects and the second turn answers from them; that output
//! has not been captured here, because no Firecrawl key was available.
//!
//! With no server on the base URL, the program prints one line naming the
//! address and exits successfully.
//!
//! # Why a failed search is not an error
//!
//! The 401 above did not fail the hook. Transport failures, non-2xx responses
//! and argument-validation failures all land inside the `ToolResult` as text,
//! because the model is the one that can react: it can rephrase the query, try
//! another tool, or tell the user the search is unavailable. A hook that
//! errored instead would take the whole stream down over a rate limit.

use std::sync::Arc;

use cuca::plugin::CucaPlugin;
use cuca::types::{
    MessageContentBlock, MessageRole, ProviderEndpoint, ToolDefinition, UnifiedMessage,
};
use cuca::{
    AgentResponseStream, CucaClient, UnifiedRequest, WebSearchConfig, WebSearchPlugin,
    WebSearchProvider,
};
use serde_json::json;
use tokio_stream::StreamExt;

/// The question the model is asked, and the query it should search for.
const QUESTION: &str =
    "Search the web for the Rust 1.98 release announcement and tell me one thing it shipped.";

/// The tool the model is offered. The plugin claims this exact name.
fn web_search_tool() -> ToolDefinition {
    ToolDefinition {
        name: "web_search".to_string(),
        description: "Search the live web and return titles, URLs and snippets.".to_string(),
        input_schema: json!({
            "type": "object",
            "properties": { "query": { "type": "string" } },
            "required": ["query"],
        }),
    }
}

/// Drain a turn into its text, the tool results the consumer received, and the
/// thinking-block count.
///
/// A reasoning model emits one `Thinking` block per token, so printing every
/// block buries the lines this demo is about. The count stays in the output
/// because it is the honest shape of a live turn.
async fn drain(mut stream: AgentResponseStream) -> (String, Vec<(String, String)>, usize) {
    let mut text = String::new();
    let mut results = Vec::new();
    let mut thinking = 0usize;
    while let Some(chunk) = stream.next().await {
        match chunk {
            Ok(MessageContentBlock::Text(chunk_text)) => text.push_str(&chunk_text),
            Ok(MessageContentBlock::Thinking { .. }) => thinking += 1,
            Ok(MessageContentBlock::ToolResult {
                tool_call_id,
                output,
            }) => results.push((tool_call_id, output)),
            Ok(_) => {}
            Err(error) => {
                println!("  the stream ended early: {error}");
                break;
            }
        }
    }
    (text, results, thinking)
}

/// The follow-up turn: the same question plus whatever the backend returned.
fn with_result(model: &str, call_id: &str, query: &str, output: &str) -> UnifiedRequest {
    UnifiedRequest::new(model)
        // A reasoning model spends the token budget on thinking first, so a
        // tight cap can end the turn before any text is emitted.
        .set_max_tokens(768)
        .add_user_message(QUESTION)
        .add_message(UnifiedMessage {
            role: MessageRole::Assistant,
            content: vec![MessageContentBlock::ToolCall {
                id: call_id.to_string(),
                name: "web_search".to_string(),
                arguments: json!({ "query": query }),
            }],
            name: None,
            tool_call_id: None,
        })
        .add_message(UnifiedMessage {
            role: MessageRole::Tool,
            content: vec![MessageContentBlock::ToolResult {
                tool_call_id: call_id.to_string(),
                output: output.to_string(),
            }],
            name: None,
            tool_call_id: Some(call_id.to_string()),
        })
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Base URL and model come from the environment so the example runs
    // against any OpenAI-compatible server; the defaults target a local
    // llama.cpp server (see the module docs for the override recipe).
    let base_url =
        std::env::var("CUCA_BASE_URL").unwrap_or_else(|_| "http://127.0.0.1:1234/v1".to_string());
    let model = std::env::var("CUCA_MODEL").unwrap_or_else(|_| "google/gemma-4-e4b".to_string());

    // Every backend the plugin speaks to is a paid search API, so there is no
    // keyless path to demonstrate.
    let Ok(api_key) = std::env::var("CUCA_SEARCH_API_KEY") else {
        println!("Set CUCA_SEARCH_API_KEY to a Firecrawl key and run this again.");
        return Ok(());
    };
    let config = WebSearchConfig {
        provider: WebSearchProvider::Firecrawl,
        api_key,
        base_url: std::env::var("CUCA_SEARCH_BASE_URL").ok(),
        max_results: 3,
    };
    println!(
        "Backend {:?}, at most {} results per search",
        config.provider, config.max_results
    );
    let web_search = Arc::new(WebSearchPlugin::new(config));
    let client = CucaClient::builder()
        .with_provider(ProviderEndpoint::LlamaCpp)
        .with_base_url(base_url.clone())
        .register_plugin(Arc::clone(&web_search) as Arc<dyn CucaPlugin>)
        .build()?;

    println!("\nTurn 1: the model searches, and the pipeline pauses for the HTTP call");
    let stream = match client
        .generate_stream(
            UnifiedRequest::new(&model)
                .set_max_tokens(256)
                .add_user_message(QUESTION)
                .add_tool(web_search_tool()),
        )
        .await
    {
        Ok(stream) => stream,
        Err(error) => {
            println!("\nNo server answered at {base_url}: {error}");
            println!("Start llama-server there, or set CUCA_BASE_URL, then run this again.");
            return Ok(());
        }
    };
    let (reply, results, thinking) = drain(stream).await;
    let Some((call_id, output)) = results.into_iter().next() else {
        println!("  the model answered without searching: {reply:?}");
        return Ok(());
    };
    // A transport or HTTP error lands inside the ToolResult rather than failing
    // the hook, so the model can react to it in the conversation.
    println!("  tool result: {output}");
    println!("  thinking blocks: {thinking}");

    // A failed search has nothing for the model to answer from, and asking it
    // to try burns the whole token budget on deliberation about the error.
    if !output.starts_with('[') {
        println!("\nNo results, so there is no second turn");
        println!("  a valid key makes this a JSON array of {{title, url, snippet}} objects");
        return Ok(());
    }

    println!("\nTurn 2: the same question, with the search result in the prompt");
    let (reply, _, thinking) = drain(
        client
            .generate_stream(with_result(&model, &call_id, "Rust 1.98 release", &output))
            .await?,
    )
    .await;
    println!("  reply: {}", reply.trim());
    println!("  thinking blocks: {thinking}");

    Ok(())
}
