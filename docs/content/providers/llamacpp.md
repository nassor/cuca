+++
title = "llama.cpp"
description = "The llama.cpp adapter's two server routes: the OpenAI-compatible chat route and the native completion route."
template = "page.html"
weight = 5
+++

# llama.cpp

<dl class="page-facts">
<dt>In one line</dt>
<dd>Dispatches unified requests to a llama.cpp server (llama-server) over either its OpenAI-compatible chat route or its native completion route.</dd>
<dt>You need</dt>
<dd>The <code>provider-llamacpp</code> feature and a running llama-server instance. No API key is required.</dd>
<dt>Read this if</dt>
<dd>You are routing requests through <code>ProviderEndpoint::LlamaCpp</code>, or choosing between the chat and completion routes.</dd>
</dl>

The smallest streaming turn: the `LlamaCpp` variant and an explicit base URL
for a server on port 1234; no API key. The adapter's own default is port 8080.

Add the crate with `cargo add cuca --features provider-llamacpp`, `cargo add tokio --features rt,macros`, and `cargo add tokio-stream`.

```rust,name=A first stream through the llama.cpp adapter
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
        .with_provider(ProviderEndpoint::LlamaCpp)
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
blocks: 2 text, 18 thinking
```

`google/gemma-4-12b-qat`, served by a local server on port 1234, produced this reply.

## Endpoint

| Fact | Value |
|---|---|
| Feature flag | `provider-llamacpp` |
| `ProviderEndpoint` variant | `LlamaCpp` |
| Default base URL | `http://127.0.0.1:8080`, used when both `LlamaCppConfig::base_url` and the client's base URL are empty |

The default carries no `/v1` suffix.

## `LlamaCppConfig`

| Field | Meaning |
|---|---|
| `base_url` | Overrides the client base URL for this dispatch; empty defers to it |
| `api_key` | Sent as `Authorization: Bearer` when set; falls back to the client's API key, then to no header |
| `model` | Overrides the request's model in the outgoing body when non-empty |
| `n_threads` | `n_threads` body parameter; omitted when unset |
| `n_gpu_layers` | `n_gpu_layers` body parameter; omitted when unset |
| `flash_attn` | `flash_attn` body parameter |
| `route` | `LlamaRoute::Chat` or `LlamaRoute::Completion`; default `Chat` |

## `LlamaRoute::Chat`

Reuses the [OpenAI](@/providers/openai.md)-compatible adapter's request body and SSE translation. The dispatch appends `/v1` to the resolved base URL when it is not already present, then posts `POST {base}/v1/chat/completions`. Thinking maps through the same `reasoning_effort` translation as OpenAI.

## `LlamaRoute::Completion`

The native route, `POST {base}/completion`, with no `/v1` suffix. It has no tool-call protocol and no thinking knob; `req.thinking` is silently ignored.

- `stream` is always `true`.
- `n_predict` carries `max_tokens`, or 128 when the request leaves it unset.
- `n_threads`, `n_gpu_layers`, and `flash_attn` appear in the body only when configured on `LlamaCppConfig`.
- The prompt is assembled from the conversation: `System` message text becomes bare leading context; `User` and `Assistant` messages are marked with `### User:` and `### Assistant:` prefixes. Only `Text` blocks contribute; images, reasoning, and tool blocks are dropped, and `Tool`-role messages are skipped entirely.
- Response frames are one token per frame, `{"content": "tok", "stop": false}`, terminated by `{"content": "", "stop": true}`. An `error` field on a frame surfaces as `CucaError::Provider`.

## See also

[vLLM](@/providers/vllm.md) and [LM Studio](@/providers/lmstudio.md) also speak the OpenAI-compatible chat route.
