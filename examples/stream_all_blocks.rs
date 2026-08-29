//! Stream a reply and print every normalized block type.
//!
//! # Prerequisites
//!
//! - A checkout of this repository (the example depends on `cuca-core` by path).
//! - A running [llama.cpp](https://github.com/ggml-org/llama.cpp) server
//!   (`llama-server`) on port 1234 with the demo model loaded.
//!
//! # Run
//!
//! ```sh
//! cargo run --example stream_all_blocks --features provider-llamacpp
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
//! Example: `CUCA_BASE_URL=http://127.0.0.1:8000/v1 CUCA_MODEL=<server-model-id> cargo run --example stream_all_blocks --features provider-llamacpp`
//!
//! # Output
//!
//! Text blocks print to stdout as they arrive; every other block type and
//! every stream error print to stderr:
//!
//! - `Thinking` prints as `[reasoning] …` (Gemma 4 E4B emits reasoning
//!   depending on the server and request).
//! - `ToolCall` prints as `[tool call] <name> <args>`.
//! - `ImageBase64` and `ToolResult` print a one-line note.
//! - An `Err` prints as `[error] …` and stops the drain.
//!
//! # Why a match over the stream?
//!
//! `generate_stream` yields the full normalized contract: every provider's
//! chunks arrive as one of the five `MessageContentBlock` variants. Matching on
//! all of them is the canonical way to consume the stream.

use std::io::{Write, stdout};

use cuca::types::{MessageContentBlock, ProviderEndpoint};
use cuca::{CucaClient, UnifiedRequest};
use tokio_stream::StreamExt;

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Base URL and model come from the environment so the example runs
    // against any OpenAI-compatible server; the defaults target a local
    // llama.cpp server (see the module docs for the override recipe).
    let base_url =
        std::env::var("CUCA_BASE_URL").unwrap_or_else(|_| "http://127.0.0.1:1234/v1".to_string());
    let model = std::env::var("CUCA_MODEL").unwrap_or_else(|_| "google/gemma-4-e4b".to_string());

    // Stage 1: build the client. The llama.cpp adapter (feature
    // `provider-llamacpp`) defaults to its chat route and to port 8080, so
    // the base URL above is passed explicitly to reach port 1234; no API key.
    let client = CucaClient::builder()
        .with_provider(ProviderEndpoint::LlamaCpp)
        .with_base_url(base_url)
        .build()?;

    // Stage 2: build the request.
    let request = UnifiedRequest::new(model)
        .add_system_message("You are concise.")
        .add_user_message("List two short facts about CUCA.");

    // Stage 3: start the stream.
    let mut stream = client.generate_stream(request).await?;

    // Stage 4: match on every block variant. Text goes to stdout; everything
    // else is annotated on stderr so the text reply stays clean.
    while let Some(chunk) = stream.next().await {
        match chunk {
            Ok(MessageContentBlock::Text(text)) => {
                print!("{text}");
                stdout().flush()?;
            }
            Ok(MessageContentBlock::Thinking { reasoning, .. }) => {
                eprintln!("[reasoning] {reasoning}");
            }
            Ok(MessageContentBlock::ToolCall {
                id,
                name,
                arguments,
            }) => {
                eprintln!("[tool call] {name} {arguments} (id: {id})");
            }
            Ok(MessageContentBlock::ImageBase64 { media_type, .. }) => {
                eprintln!("[image] {media_type} (base64 data omitted)");
            }
            Ok(MessageContentBlock::ToolResult { tool_call_id, .. }) => {
                eprintln!("[tool result] for tool call {tool_call_id}");
            }
            Err(err) => {
                eprintln!("[error] {err}");
                break;
            }
        }
    }
    Ok(())
}
