//! Stream a reply and print every normalized block type.
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
//! - `Thinking` prints as `[reasoning] …`, one line per streamed block.
//! - `ToolCall` prints as `[tool call] <name> <args> (id: <id>)`.
//! - `ImageBase64` and `ToolResult` print a one-line note.
//! - An `Err` prints as `[error] …` and stops the drain.
//!
//! Expect that stderr stream to be long. One run against
//! `google/gemma-4-12b-qat` on llama.cpp wrote 1506 `[reasoning]` lines, one
//! per reasoning token, for two lines of answer on stdout:
//!
//! ```text
//! 1. In Romanian folklore, the **Cuca** is a legendary bogeyman used to frighten children into behaving.
//! 2. It is traditionally depicted as a witch-like creature with a long nose and long hair.
//! ```
//!
//! That volume is the point of the example rather than noise: it shows every
//! block the provider actually sent. Redirect stderr to keep the answer alone,
//! `cargo run --example stream_all_blocks --features provider-llamacpp 2>/dev/null`
//! on Linux and macOS. The counts and the answer depend on the model, and which
//! tags appear at all depends on what the server emits: a plain prose reply
//! from a non-reasoning model is all `Text`.
//!
//! With no server on the base URL, the program prints one line naming the
//! address and exits successfully.
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
        .with_base_url(base_url.clone())
        .build()?;

    // Stage 2: build the request.
    let request = UnifiedRequest::new(model)
        .add_system_message("You are concise.")
        .add_user_message("List two short facts about CUCA.");

    // Stage 3: start the stream.
    let mut stream = match client.generate_stream(request).await {
        Ok(stream) => stream,
        Err(error) => {
            println!("No server answered at {base_url}: {error}");
            println!("Start llama-server there, or set CUCA_BASE_URL, then run this again.");
            return Ok(());
        }
    };

    // Stage 4: match on every block variant, keeping the text reply clean of
    // the other annotations.
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
