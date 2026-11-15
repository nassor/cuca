+++
title = "LM Studio"
description = "The LM Studio adapter: a first request, the local default endpoint, optional bearer auth, and the shared OpenAI-compatible translation."
template = "page.html"
weight = 7
+++

# LM Studio

<dl class="page-facts">
<dt>In one line</dt>
<dd>Sends unified requests to LM Studio's local OpenAI-compatible chat completions route.</dd>
<dt>You need</dt>
<dd>The <code>provider-lmstudio</code> feature and a running LM Studio server. An API key is optional.</dd>
<dt>Read this if</dt>
<dd>You route requests through <code>ProviderEndpoint::LmStudio</code>.</dd>
</dl>

## Send a first request

Start the LM Studio local server first. The model id is the one the server reports under `GET {base}/models`. Change `base_url` and `model` to match your server.

```bash,name=Runs the same on all three platforms
cargo add cuca --path /path/to/cuca --features provider-lmstudio
cargo add tokio --features rt,macros
cargo add tokio-stream
```

```rust,name=A first stream through the LM Studio adapter
use std::io::{Write, stdout};

use cuca::types::{MessageContentBlock, ProviderEndpoint};
use cuca::{CucaClient, UnifiedRequest};
use tokio_stream::StreamExt;

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let base_url = "http://127.0.0.1:1234/v1";
    let model = "google/gemma-4-e4b";

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

```bash,name=Runs the same on all three platforms
cargo run
```

```text,name=Expected shape; the wording and the block counts vary by model
Hello.
blocks: 2 text, 67 thinking
```

A model that reasons first streams thinking blocks, then text blocks.

## What you need

| Fact | Value |
|---|---|
| Feature flag | `provider-lmstudio` |
| `ProviderEndpoint` variant | `LmStudio` |
| Default base URL | `http://127.0.0.1:1234/v1`, used when the client's base URL is empty |
| Route | `POST {base_url}/chat/completions` |
| Auth | `Authorization: Bearer <api_key>`, only when a key is set |

The default base URL ends in `/v1`. A base URL you set is used as given, so it must end in `/v1` too when the server serves under it. The adapter trims a trailing `/` and appends only `/chat/completions`.

## How it differs from OpenAI

This adapter runs the code on the [OpenAI](@/providers/openai.md) page unchanged. The differences:

- The default base URL is `http://127.0.0.1:1234/v1`.
- The API key is optional. With no key, no auth header is sent and dispatch does not fail.

## What is shared with OpenAI

| Topic | Rule |
|---|---|
| Request | [OpenAI request translation](@/providers/openai.md#request-translation). The model is the request's model. `top_p`, `top_k`, `seed` and `stop` go out under those names, and no sampling field is rejected |
| Stream | [OpenAI response stream](@/providers/openai.md#response-stream). LM Studio sends `reasoning_content` for reasoning models, and those deltas become `Thinking` blocks |
| Token usage | [OpenAI token usage](@/providers/openai.md#token-usage). LM Studio honours `stream_options.include_usage`, so `UnifiedResponse::usage` carries the server's own counts, with `reasoning_tokens` when the server reports them |
| Thinking | [OpenAI thinking](@/providers/openai.md#thinking). `req.thinking` maps to `reasoning_effort` |
| Limits | [OpenAI limits](@/providers/openai.md#limits). There is no adapter cap and no request timeout |

## Errors

| Condition | Error |
|---|---|
| Feature not compiled | `CucaError::ProviderNotEnabled("provider-lmstudio")` |
| Non-2xx response | `CucaError::Http { status, body }` with the captured body |
| Connection or body-read failure | `CucaError::Transport { message }` |
| Frame is not valid JSON | `CucaError::Json` |
| Frame carries `{"error": {"message": ...}}` | `CucaError::Provider` with `provider: OpenAi` |
| Frame bytes are not valid UTF-8 | `CucaError::SseParse` |

The provider name in a frame error is `OpenAi`, because the translator is shared. A malformed frame or an error frame ends the stream after the error.
