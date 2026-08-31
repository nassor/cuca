//! Stream a plain-text reply from a local llama.cpp Gemma 4 E4B server.
//!
//! This is the README quick-start demo: one client, one request, one stream.
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
//! cargo run --example llamacpp_gemma --features provider-llamacpp
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
//! Example: `CUCA_BASE_URL=http://127.0.0.1:8000/v1 CUCA_MODEL=<server-model-id> cargo run --example llamacpp_gemma --features provider-llamacpp`
//!
//! # Output
//!
//! The model's reply prints as text chunks arrive. The program exits when the
//! stream ends; if the llama.cpp server is not running, it fails with a
//! connection error.
//!
//! # Why the llama.cpp adapter?
//!
//! `llama-server` exposes two API styles: an OpenAI-compatible
//! `/v1/chat/completions` route and a native `/completion` route with
//! raw-token frames. `provider-llamacpp` defaults to the chat route, reusing
//! the same OpenAI-compatible translator as the other adapters. The adapter's
//! own default base URL is `http://127.0.0.1:8080`, so this demo passes an
//! explicit `with_base_url` to reach the server on port 1234 instead. No API
//! key is needed: the `Authorization` header is only sent when one is
//! configured.

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

    // Stage 1: build the client. The base URL override and no-API-key
    // behavior are explained in the module docs.
    let client = CucaClient::builder()
        .with_provider(ProviderEndpoint::LlamaCpp)
        .with_base_url(base_url)
        .build()?;

    // Stage 2: build the request.
    let request = UnifiedRequest::new(model)
        .add_system_message("You are concise.")
        .add_user_message("Explain CUCA in one sentence.");

    // Stage 3: generate_stream starts the request and returns the normalized
    // `AgentResponseStream`, which yields `Result<MessageContentBlock,
    // CucaError>` items.
    let mut stream = client.generate_stream(request).await?;

    // Stage 4: drain the stream and print each text block as it arrives.
    // Flushing after every chunk makes the streaming visible even when piped.
    while let Some(chunk) = stream.next().await {
        if let Ok(MessageContentBlock::Text(text)) = chunk {
            print!("{text}");
            stdout().flush()?;
        }
    }
    Ok(())
}
