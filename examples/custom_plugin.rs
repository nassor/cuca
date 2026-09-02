//! Stream a reply through a small custom [`cuca::plugin::CucaPlugin`].
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
//! cargo run --example custom_plugin --features provider-llamacpp
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
//! Example: `CUCA_BASE_URL=http://127.0.0.1:8000/v1 CUCA_MODEL=<server-model-id> cargo run --example custom_plugin --features provider-llamacpp`
//!
//! # Output
//!
//! The reply prints as text chunks, then the plugin's summary line. From one
//! run against `google/gemma-4-12b-qat` on llama.cpp:
//!
//! ```text
//! CUCA most commonly refers to either the Credit Union of Central Alabama or the Center for Urban Community Action, depending on the context.
//! [example-block-counter] model=google/gemma-4-12b-qat duration=81.89s completion_tokens=1991 blocks=1991
//! ```
//!
//! The summary line carries its own leading newline because the hook runs
//! inside the caller's final `next()` poll, before the drain loop returns.
//!
//! `completion_tokens` equals the block count here: the client counts one token
//! per `Text`, `Thinking`, and `ToolCall` block. Both reach 1991 because this
//! demo sets no `max_tokens` and a reasoning model emits one `Thinking` block
//! per reasoning token. The reply, the counts, and the duration all depend on
//! the model.
//!
//! # The plugin pipeline
//!
//! Every request flows through the same stages:
//!
//! 1. `on_request`: before dispatch; may mutate the request.
//! 2. Provider dispatch: the adapter streams normalized blocks.
//! 3. `on_stream_chunk`: once per block, in registration order.
//! 4. `on_response_complete`: exactly once, when the stream ends.
//!
//! This example implements the last two hooks: it counts blocks as they stream
//! and reports the aggregated [`cuca::UnifiedResponse`] at the end.

use std::io::{Write, stdout};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use cuca::plugin::CucaPlugin;
use cuca::types::{MessageContentBlock, ProviderEndpoint};
use cuca::{CucaClient, PluginError, UnifiedRequest, UnifiedResponse};
use tokio_stream::StreamExt;

/// A [`CucaPlugin`] that counts streamed blocks and reports a summary.
#[derive(Default)]
struct BlockCounterPlugin {
    blocks: AtomicUsize,
}

impl CucaPlugin for BlockCounterPlugin {
    fn name(&self) -> &'static str {
        "example-block-counter"
    }

    fn on_stream_chunk(&self, _chunk: &mut MessageContentBlock) -> Result<(), PluginError> {
        self.blocks.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    fn on_response_complete(&self, res: &UnifiedResponse) -> Result<(), PluginError> {
        // The hook fires inside the caller's final `next()` poll, so the
        // leading newline is what separates this line from the streamed reply.
        println!(
            "\n[example-block-counter] model={} duration={:.2}s completion_tokens={} blocks={}",
            res.model,
            res.duration_secs,
            res.completion_tokens,
            self.blocks.load(Ordering::Relaxed),
        );
        Ok(())
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Base URL and model come from the environment so the example runs
    // against any OpenAI-compatible server; the defaults target a local
    // llama.cpp server (see the module docs for the override recipe).
    let base_url =
        std::env::var("CUCA_BASE_URL").unwrap_or_else(|_| "http://127.0.0.1:1234/v1".to_string());
    let model = std::env::var("CUCA_MODEL").unwrap_or_else(|_| "google/gemma-4-e4b".to_string());

    // Stage 1: build the client and register the plugin. Plugins run in
    // registration order; each is held as an `Arc<dyn CucaPlugin>`.
    let client = CucaClient::builder()
        .with_provider(ProviderEndpoint::LlamaCpp)
        .with_base_url(base_url.clone())
        .register_plugin(Arc::new(BlockCounterPlugin::default()))
        .build()?;

    // Stage 2: build the request.
    let request = UnifiedRequest::new(model)
        .add_system_message("You are concise.")
        .add_user_message("Explain CUCA in one sentence.");

    // Stage 3: start the stream.
    let mut stream = match client.generate_stream(request).await {
        Ok(stream) => stream,
        Err(error) => {
            println!("No server answered at {base_url}: {error}");
            println!("Start llama-server there, or set CUCA_BASE_URL, then run this again.");
            return Ok(());
        }
    };

    // Stage 4: drain the stream; the plugin's summary prints from
    // `on_response_complete` once the stream ends.
    while let Some(chunk) = stream.next().await {
        if let Ok(MessageContentBlock::Text(text)) = chunk {
            print!("{text}");
            stdout().flush()?;
        }
    }
    Ok(())
}
