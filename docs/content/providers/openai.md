+++
title = "OpenAI"
description = "The OpenAI chat completions adapter: endpoint, auth, and the shared OpenAI-compatible request and stream translation."
template = "page.html"
weight = 1
+++

# OpenAI

<dl class="page-facts">
<dt>In one line</dt>
<dd>Dispatches unified requests to OpenAI's chat completions API and defines the OpenAI-compatible wire translation that four other adapters reuse.</dd>
<dt>You need</dt>
<dd>The <code>provider-openai</code> feature and an API key.</dd>
<dt>Read this if</dt>
<dd>You route requests through <code>ProviderEndpoint::OpenAi</code>, or you came from the <a href="@/providers/deepseek.md">DeepSeek</a>, <a href="@/providers/llamacpp.md">llama.cpp</a>, <a href="@/providers/vllm.md">vLLM</a> or <a href="@/providers/lmstudio.md">LM Studio</a> page for the shared translation.</dd>
</dl>

This adapter owns the translation described below. The default base URL is `https://api.openai.com/v1` and the key is sent as a bearer token. [DeepSeek](@/providers/deepseek.md) (native route), [llama.cpp](@/providers/llamacpp.md) (chat route), [vLLM](@/providers/vllm.md) and [LM Studio](@/providers/lmstudio.md) run the same code and state only their differences.

## Endpoint

| Fact | Value |
|---|---|
| Feature flag | `provider-openai` |
| `ProviderEndpoint` variant | `OpenAi` |
| Default base URL | `https://api.openai.com/v1`, used when the client's base URL is empty |
| Route | `POST {base_url}/chat/completions` |

The default base URL already carries `/v1`. The adapter appends only `/chat/completions` and trims a trailing `/` from the base URL first.

## Authentication

- `Authorization: Bearer <api_key>` is sent when the client has an API key.
- No auth header is sent when no key is set, and dispatch does not fail.

## Request translation

The request body is built from a `UnifiedRequest`:

- `model` is the request's model; `messages` is one entry per unified message.
- `stream` is always `true`, and `stream_options` is always `{"include_usage": true}`.
- `temperature` and `max_tokens` are included only when set on the request.
- `tools` carries every `UnifiedRequest::tools` entry as `{"type": "function", "function": {name, description, parameters}}`, with `parameters` holding the definition's `input_schema`. The key is omitted when the request declares no tool.
- No `tool_choice` is sent, so the server's own selection default applies.
- A message's `Text` blocks are joined with a newline into plain string content.
- A message with any `ImageBase64` block becomes a content array of `{type: "text"}` and `{type: "image_url"}` parts. `Thinking`, `ToolCall` and `ToolResult` blocks have no part in that array and are dropped from it.
- A `Thinking` block becomes `reasoning_content` on an assistant message that carries exactly one thinking block. In any other position it is dropped.
- `ToolCall` blocks become the assistant `tool_calls` array with arguments stringified. A message whose only blocks are tool calls carries `content: null`.
- A `ToolResult` becomes a `role: "tool"` message with `tool_call_id` and the output as `content`.

## Response stream

- Frames arrive as `choices[0].delta.{content, reasoning_content, tool_calls}`, terminated by `data: [DONE]`.
- A non-empty `content` delta becomes `MessageContentBlock::Text`.
- A non-empty `reasoning_content` delta becomes `MessageContentBlock::Thinking` with no signature.
- Tool calls accumulate by the frame's `index` across deltas. A call is emitted as a `ToolCall` block once its arguments parse as JSON, or at a non-null `finish_reason` or `[DONE]`, whichever comes first.
- A fragment that never parses as JSON is emitted with `null` arguments instead of being dropped.
- Without `[DONE]`, the stream ends when the byte stream ends, after any flushed tool calls.

## Token usage

Because the request sets `stream_options.include_usage`, the last frame before `[DONE]` carries a `usage` object. Most servers send it as an extra frame with an empty `choices` array. DeepSeek sends it on the final `finish_reason` frame.

- The `usage` object never becomes a block; the last one seen wins.
- `prompt_tokens` and `completion_tokens` become `UnifiedResponse::prompt_tokens` and `completion_tokens`.
- The whole object becomes `UnifiedResponse::usage` as a `TokenUsage`, with `completion_tokens_details.reasoning_tokens` as `reasoning_tokens` when the server reports it.
- A `usage` object missing either counter is ignored.
- With no usable `usage` frame, `usage` is `None`, `completion_tokens` counts one per `Text`, `Thinking` and `ToolCall` block, and `prompt_tokens` is `0`.

Plugins see the usage in `on_response_complete`. A caller reads it per call with `CucaClient::generate_stream_with_response`:

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

## Thinking

`req.thinking` maps to the `reasoning_effort` request field.

- A `ThinkingParams::OpenAi` override wins.
- Otherwise the unified effort maps as in the table below.
- With no effort set, `reasoning_effort` is `"medium"`.
- A disabled `ThinkingConfig` omits the key. With no `thinking` on the request, the key is also omitted.

| Unified `ThinkingEffort` | `reasoning_effort` |
|---|---|
| `Minimal` | `"minimal"` |
| `Low` | `"low"` |
| `Medium` | `"medium"` |
| `High` | `"high"` |
| `XHigh` | `"high"` (no native extra-high value) |

## Errors

| Condition | Error |
|---|---|
| Feature not compiled | `CucaError::ProviderNotEnabled("provider-openai")` |
| Non-2xx response | `CucaError::Http { status, body }` with the captured body |
| Connection or body-read failure | `CucaError::Transport { message }` |
| Frame is not valid JSON | `CucaError::Json`; the stream ends after it |
| Frame carries `{"error": {"message": ...}}` | `CucaError::Provider` with `provider: OpenAi`; the stream ends after it |
| Frame bytes are not valid UTF-8 | `CucaError::SseParse` |

The frame-level `Provider` error names `OpenAi` on every adapter that shares this translation, including vLLM, LM Studio, DeepSeek native and llama.cpp chat.

## Limits

- No cap on request or response size, and the client sets no request timeout.
- The SSE parser keeps one reusable buffer that starts at 8192 bytes. Its high-water mark is the largest single frame plus one partial line, not the stream length.
- In-flight tool calls hold their argument fragments until each call completes or `finish_reason` or `[DONE]` flushes it. Nothing else in the translator grows with the stream.
- `max_tokens` has no default here; the server's default applies when the request leaves it unset.

## Example

The smallest streaming turn: the `OpenAi` variant, an API key, and every other builder default.

```bash,name=Runs the same on all three platforms
cargo add cuca --path /path/to/cuca --features provider-openai
cargo add tokio --features rt,macros
cargo add tokio-stream
```

```bash,name=Linux and macOS: set the key
export OPENAI_API_KEY="sk-..."
```

```powershell,name=Windows (PowerShell): set the key
$env:OPENAI_API_KEY = "sk-..."
```

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

```bash,name=Runs the same on all three platforms
cargo run
```

The `provider-openai` feature comes from the `cargo add` line above.

```text,name=Expected shape; not captured from a live run
Hello! How can I help you today?
blocks: 1 text, 0 thinking
```
