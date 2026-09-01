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

```rust,name=A first stream through the OpenAI adapter
use cuca::types::{MessageContentBlock, ProviderEndpoint};
use cuca::{CucaClient, UnifiedRequest};
use tokio_stream::StreamExt;

let client = CucaClient::builder()
    .with_provider(ProviderEndpoint::OpenAi)
    .with_api_key(std::env::var("OPENAI_API_KEY")?)
    .build()?;

let mut stream = client
    .generate_stream(UnifiedRequest::new("gpt-4o-mini").add_user_message("Say hello."))
    .await?;
while let Some(block) = stream.next().await {
    if let MessageContentBlock::Text(text) = block? {
        print!("{text}");
    }
}
```

```text,name=Expected output; exact wording varies by model
Hello! How can I help you today?
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

- `stream` is always `true`.
- `temperature` and `max_tokens` are included only when set on the request.
- A message's `Text` blocks are joined with a newline into plain string content.
- A message carrying any `ImageBase64` block becomes a content array of `{type: "text"}` and `{type: "image_url"}` parts instead of a plain string.
- A `Thinking` block becomes `reasoning_content` on an assistant message carrying exactly one thinking block; elsewhere it is dropped.
- `ToolCall` blocks become the assistant `tool_calls` array with arguments stringified; a message whose only blocks are tool calls carries `content: null`.
- `ToolResult` becomes a `role: "tool"` message with `tool_call_id` and the output as `content`.

Response frames arrive as `choices[0].delta.{content, reasoning_content, tool_calls}`, terminated by `data: [DONE]`. Tool calls are accumulated by their frame `index` across multiple deltas and flushed as complete `ToolCall` blocks on `finish_reason` or `[DONE]`.

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
