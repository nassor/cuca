+++
title = "LM Studio"
description = "The LM Studio adapter: local default endpoint, optional bearer auth, and the shared OpenAI-compatible request and stream translation."
template = "page.html"
weight = 7
+++

# LM Studio

<dl class="page-facts">
<dt>In one line</dt>
<dd>Dispatches unified requests to LM Studio's local OpenAI-compatible chat completions route.</dd>
<dt>You need</dt>
<dd>The <code>provider-lmstudio</code> feature and a running LM Studio server. An API key is optional.</dd>
<dt>Read this if</dt>
<dd>You are routing requests through <code>ProviderEndpoint::LmStudio</code>.</dd>
</dl>

The smallest streaming turn: the `LmStudio` variant against the default local
endpoint; no API key. The model id is the one the server reports under
`GET {base}/models`.

Add the crate with `cargo add cuca --features provider-lmstudio`, `cargo add tokio --features rt,macros`, and `cargo add tokio-stream`.

```rust,name=A first stream through the LM Studio adapter
use std::io::{Write, stdout};

use cuca::types::{MessageContentBlock, ProviderEndpoint};
use cuca::{CucaClient, UnifiedRequest};
use tokio_stream::StreamExt;

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let base_url =
        std::env::var("CUCA_BASE_URL").unwrap_or_else(|_| "http://127.0.0.1:1234/v1".to_string());
    let model = std::env::var("CUCA_MODEL").unwrap_or_else(|_| "google/gemma-4-e4b".to_string());

    let client = CucaClient::builder()
        .with_provider(ProviderEndpoint::LmStudio)
        .with_base_url(base_url)
        .build()?;

    let request = UnifiedRequest::new(model)
        .add_system_message("You are concise.")
        .add_user_message("Say hello.")
        .set_max_tokens(128);
    let mut stream = client.generate_stream(request).await?;

    let mut text_blocks = 0usize;
    let mut thinking_blocks = 0usize;
    while let Some(block) = stream.next().await {
        match block? {
            MessageContentBlock::Text(text) => {
                print!("{text}");
                stdout().flush()?;
                text_blocks += 1;
            }
            MessageContentBlock::Thinking { .. } => thinking_blocks += 1,
            _ => {}
        }
    }
    println!("\nblocks: {text_blocks} text, {thinking_blocks} thinking");
    Ok(())
}
```

```text,name=Expected output; exact wording varies by model
Hello.
blocks: 2 text, 67 thinking
```

`google/gemma-4-12b-qat`, served by a local server on port 1234, produced this reply.

## Endpoint

| Fact | Value |
|---|---|
| Feature flag | `provider-lmstudio` |
| `ProviderEndpoint` variant | `LmStudio` |
| Default base URL | `http://127.0.0.1:1234/v1`, used when the client's base URL is empty |
| Route | `POST {base_url}/chat/completions` |

## Authentication

`Authorization: Bearer <api_key>` when an API key is configured; LM Studio treats it as an optional header, not a required credential.

## Shared adapter

This provider shares the [OpenAI](@/providers/openai.md)-compatible request builder and SSE translator in full. LM Studio can emit `reasoning_content` for reasoning models, which flows into `Thinking` blocks through the same translation. See the OpenAI page for the request body shape, the tool-call accumulation rules, and the thinking effort mapping.

## Base URL deviation

The adapter's default, `http://127.0.0.1:1234/v1`, carries the `/v1` suffix the shared adapter appends `/chat/completions` to. The design specification's provider table instead lists the bare host `http://127.0.0.1:1234` without `/v1`, unlike its OpenAI and DeepSeek rows, which do include `/v1`. The code default here includes `/v1`.
