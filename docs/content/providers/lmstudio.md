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
<dd>You route requests through <code>ProviderEndpoint::LmStudio</code>.</dd>
</dl>

This adapter runs the code on the [OpenAI](@/providers/openai.md) page unchanged. Its own differences:

- The default base URL is `http://127.0.0.1:1234/v1`.
- The API key is optional. A key is sent as a bearer token only when one is set.

Everything else on this page points to the matching section of the OpenAI page.

## Endpoint

| Fact | Value |
|---|---|
| Feature flag | `provider-lmstudio` |
| `ProviderEndpoint` variant | `LmStudio` |
| Default base URL | `http://127.0.0.1:1234/v1`, used when the client's base URL is empty |
| Route | `POST {base_url}/chat/completions` |

The default base URL carries `/v1`. Any base URL you set is used as given, so it must carry `/v1` too when the server serves under it. The adapter appends only `/chat/completions` and trims a trailing `/` first.

## Authentication

- `Authorization: Bearer <api_key>` is sent when the client has an API key.
- With no key, no auth header is sent and dispatch does not fail.

## Request translation

Same as [OpenAI request translation](@/providers/openai.md#request-translation). The model is the request's model.

## Response stream

Same as [OpenAI response stream](@/providers/openai.md#response-stream). LM Studio emits `reasoning_content` for reasoning models, and those deltas become `Thinking` blocks.

## Token usage

Same as [OpenAI token usage](@/providers/openai.md#token-usage). LM Studio honours `stream_options.include_usage`, so `UnifiedResponse::usage` carries the server's own counts, with `reasoning_tokens` when the server reports them.

## Thinking

Same as [OpenAI thinking](@/providers/openai.md#thinking): `req.thinking` maps to `reasoning_effort`.

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

## Limits

Same as [OpenAI limits](@/providers/openai.md#limits). There is no adapter cap and no request timeout.

## Example

The smallest streaming turn: the `LmStudio` variant against a local server, with no API key. The model id is the one the server reports under `GET {base}/models`.

```bash,name=Runs the same on all three platforms
cargo add cuca --path /path/to/cuca --features provider-lmstudio
cargo add tokio --features rt,macros
cargo add tokio-stream
```

```bash,name=Linux and macOS: choose the server and model
export CUCA_BASE_URL="http://127.0.0.1:1234/v1"
export CUCA_MODEL="google/gemma-4-e4b"
```

```powershell,name=Windows (PowerShell): choose the server and model
$env:CUCA_BASE_URL = "http://127.0.0.1:1234/v1"
$env:CUCA_MODEL = "google/gemma-4-e4b"
```

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

```bash,name=Runs the same on all three platforms
cargo run
```

The `provider-lmstudio` feature comes from the `cargo add` line above.

```text,name=Expected output; exact wording varies by model
Hello.
blocks: 2 text, 67 thinking
```

`google/gemma-4-12b-qat`, served by a local server on port 1234, produced this reply.
