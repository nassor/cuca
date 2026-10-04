+++
title = "OpenAI"
description = "The OpenAI chat completions adapter: endpoint, auth, routes, and the shared OpenAI-compatible request and stream translation."
template = "page.html"
weight = 1
+++

# OpenAI

<dl class="page-facts">
<dt>In one line</dt>
<dd>Dispatches unified requests to OpenAI's chat completions API and defines the OpenAI-compatible wire translation four other adapters reuse.</dd>
<dt>You need</dt>
<dd>The <code>provider-openai</code> feature and an API key.</dd>
<dt>Read this if</dt>
<dd>You are routing requests through <code>ProviderEndpoint::OpenAi</code>, or you landed here from another OpenAI-compatible provider's page.</dd>
</dl>

The smallest streaming turn: the `OpenAi` variant, an API key, and every other
builder default.

Add the crate with `cargo add cuca --features provider-openai`, `cargo add tokio --features rt,macros`, and `cargo add tokio-stream`.

```rust,name=A first stream through the OpenAI adapter
use std::io::{Write, stdout};

use cuca::types::{MessageContentBlock, ProviderEndpoint};
use cuca::{CucaClient, UnifiedRequest};
use tokio_stream::StreamExt;

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = CucaClient::builder()
        .with_provider(ProviderEndpoint::OpenAi)
        .with_api_key(std::env::var("OPENAI_API_KEY")?)
        .build()?;

    let request = UnifiedRequest::new("gpt-4o-mini")
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

```text,name=Expected shape; not captured from a live run
Hello! How can I help you today?
blocks: 1 text, 0 thinking
```

## Endpoint

| Fact | Value |
|---|---|
| Feature flag | `provider-openai` |
| `ProviderEndpoint` variant | `OpenAi` |
| Default base URL | `https://api.openai.com/v1`, used when the client's base URL is empty |
| Route | `POST {base_url}/chat/completions` |

The default base URL already carries the `/v1` suffix; the adapter never appends one.

## Authentication

`Authorization: Bearer <api_key>` when an API key is configured on the client. No header is sent when none is configured.

## Shared OpenAI-compatible adapter

This adapter's request builder and SSE translator are shared by four other endpoints: `provider-vllm`, `provider-lmstudio`, DeepSeek's native route, and llama.cpp's chat route. Their pages link back here for the shared behavior below rather than restate it.

Request body, from a `UnifiedRequest`:

- `stream` is always `true`, and `stream_options` is always `{"include_usage": true}`.
- `temperature` and `max_tokens` are included only when set on the request.
- `tools` carries every `UnifiedRequest::tools` entry as `{"type": "function", "function": {name, description, parameters}}`, with `parameters` holding the definition's `input_schema`; the key is omitted when the request declares no tool, and no `tool_choice` is sent, so the server's own selection default applies.
- A message's `Text` blocks are joined with a newline into plain string content.
- A message carrying any `ImageBase64` block becomes a content array of `{type: "text"}` and `{type: "image_url"}` parts instead of a plain string.
- A `Thinking` block becomes `reasoning_content` on an assistant message carrying exactly one thinking block; elsewhere it is dropped.
- `ToolCall` blocks become the assistant `tool_calls` array with arguments stringified; a message whose only blocks are tool calls carries `content: null`.
- `ToolResult` becomes a `role: "tool"` message with `tool_call_id` and the output as `content`.

Response frames arrive as `choices[0].delta.{content, reasoning_content, tool_calls}`, terminated by `data: [DONE]`. Tool calls are accumulated by their frame `index` across multiple deltas and flushed as complete `ToolCall` blocks on `finish_reason` or `[DONE]`.

## Token usage

Because the request sets `stream_options.include_usage`, the last frame before `[DONE]` carries a `usage` object: most servers send it as one more frame with an empty `choices` array, DeepSeek on the final `finish_reason` frame. The `usage` object never becomes a block. Its `prompt_tokens` and `completion_tokens` become `UnifiedResponse::prompt_tokens` and `completion_tokens`, and the whole object becomes `UnifiedResponse::usage` as a `TokenUsage`, with `completion_tokens_details.reasoning_tokens` as `reasoning_tokens` when the server breaks it out. Plugins see it in `on_response_complete`; a caller reads it per call with `CucaClient::generate_stream_with_response`:

```rust,name=Read one call's token usage
let (mut stream, response) = client
    .generate_stream_with_response(UnifiedRequest::new("gpt-4o-mini").add_user_message("Say hello."))
    .await?;
while let Some(block) = stream.next().await {
    let _ = block?;
}
if let Some(usage) = response.take().and_then(|res| res.usage) {
    println!("{} prompt + {} completion tokens", usage.prompt_tokens, usage.completion_tokens);
}
```

A server that sends no `usage` frame leaves `usage` at `None`, and the response keeps the fallback counts: `completion_tokens` one per `Text`, `Thinking` and `ToolCall` block, `prompt_tokens` `0`.

## Thinking effort

`req.thinking` maps to the `reasoning_effort` request field: a `ThinkingParams::OpenAi` override wins, otherwise the unified effort maps as below, otherwise `reasoning_effort` defaults to `"medium"`. A disabled `ThinkingConfig` omits the key.

| Unified `ThinkingEffort` | `reasoning_effort` |
|---|---|
| `Minimal` | `"minimal"` |
| `Low` | `"low"` |
| `Medium` | `"medium"` |
| `High` | `"high"` |
| `XHigh` | `"high"` (no native extra-high value) |

## See also

[Anthropic](@/providers/anthropic.md), [Google Gemini](@/providers/gemini.md), [DeepSeek](@/providers/deepseek.md), [llama.cpp](@/providers/llamacpp.md), [vLLM](@/providers/vllm.md), and [LM Studio](@/providers/lmstudio.md).
