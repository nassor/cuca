+++
title = "OpenAI"
description = "The OpenAI chat completions adapter: a first request, endpoint, auth, and the OpenAI-compatible translation that four other adapters share."
template = "page.html"
weight = 1
+++

# OpenAI

<dl class="page-facts">
<dt>In one line</dt>
<dd>Sends unified requests to OpenAI's chat completions API. Its translation is shared by the DeepSeek native route, llama.cpp chat, vLLM and LM Studio.</dd>
<dt>You need</dt>
<dd>The <code>provider-openai</code> feature and an API key.</dd>
<dt>Read this if</dt>
<dd>You route requests through <code>ProviderEndpoint::OpenAi</code>, or you came from the <a href="@/providers/deepseek.md">DeepSeek</a>, <a href="@/providers/llamacpp.md">llama.cpp</a>, <a href="@/providers/vllm.md">vLLM</a> or <a href="@/providers/lmstudio.md">LM Studio</a> page for the shared translation.</dd>
</dl>

## Send a first request

Add the crate with the feature, then set your key.

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

Put this in `src/main.rs`. It prints each text block as it arrives, then counts the blocks.

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

```text,name=Expected shape; not captured from a live run
Hello! How can I help you today?
blocks: 1 text, 0 thinking
```

A text block is one piece of the reply. A thinking block is one piece of the model's reasoning, which some models stream before the answer.

## What you need

| Fact | Value |
|---|---|
| Feature flag | `provider-openai` |
| `ProviderEndpoint` variant | `OpenAi` |
| Default base URL | `https://api.openai.com/v1`, used when the client's base URL is empty |
| Route | `POST {base_url}/chat/completions` |
| Auth | `Authorization: Bearer <api_key>` |

- The default base URL already ends in `/v1`. The adapter trims a trailing `/` and appends only `/chat/completions`.
- A key is sent as a bearer token when the client has one. With no key, no auth header is sent and dispatch does not fail.

## How the translation works

This page owns the translation. [DeepSeek](@/providers/deepseek.md) (native route), [llama.cpp](@/providers/llamacpp.md) (chat route), [vLLM](@/providers/vllm.md) and [LM Studio](@/providers/lmstudio.md) run the same code and list only their differences.

<div class="dgm">
<div class="dgm-scroll">
<svg viewBox="0 0 660 130" role="img" aria-labelledby="oai-title oai-desc">
  <title id="oai-title">The OpenAI-compatible translation in four steps</title>
  <desc id="oai-desc">A UnifiedRequest becomes a chat completions JSON body. It crosses the HTTP boundary as a POST. The server answers with SSE frames of delta fields. The adapter turns each frame into MessageContentBlock values and keeps the last usage object.</desc>
  <defs>
    <marker id="oai-arrow" viewBox="0 0 8 8" refX="7" refY="4"
            markerWidth="6" markerHeight="6" orient="auto">
      <path d="M0 0.6 L7 4 L0 7.4 z" fill="var(--muted-foreground)"/>
    </marker>
  </defs>
  <text class="t-title" x="6" y="20">Request out, blocks back</text>
  <rect class="blk blk-data" x="6" y="44" width="130" height="60" rx="6"/>
  <text class="t-lbl t-mid t-data" x="71" y="68">UnifiedRequest</text>
  <text class="t-sm t-mid" x="71" y="86">messages, tools</text>
  <rect class="blk blk-data" x="172" y="44" width="130" height="60" rx="6"/>
  <text class="t-lbl t-mid t-data" x="237" y="68">JSON body</text>
  <text class="t-sm t-mid" x="237" y="86">stream: true</text>
  <rect class="blk blk-bnd" x="358" y="44" width="130" height="60" rx="6"/>
  <text class="t-lbl t-mid t-bnd" x="423" y="68">SSE frames</text>
  <text class="t-sm t-mid" x="423" y="86">choices[0].delta</text>
  <rect class="blk blk-data" x="524" y="44" width="130" height="60" rx="6"/>
  <text class="t-lbl t-mid t-data" x="589" y="68">Content blocks</text>
  <text class="t-sm t-mid" x="589" y="86">Text, Thinking, …</text>
  <path class="arw arw-data" d="M138 74 H168" marker-end="url(#oai-arrow)"/>
  <path class="arw arw-bnd" d="M304 74 H354" marker-end="url(#oai-arrow)"/>
  <path class="arw arw-data" d="M490 74 H520" marker-end="url(#oai-arrow)"/>
  <text class="t-ax" x="306" y="62">POST</text>
</svg>
</div>
<p class="dgm-cap"><b>The adapter translates twice, once in each direction.</b> The request becomes one JSON body. Each streamed frame becomes zero or more content blocks, and the last <code>usage</code> object becomes the token counts.</p>
</div>

## Request translation

- `model` is the request's model. `messages` has one entry per unified message.
- `stream` is always `true`, and `stream_options` is always `{"include_usage": true}`.
- `temperature`, `max_tokens`, `top_p` and `seed` are sent only when set. `stop` (an array of strings) is sent only when non-empty.
- `max_tokens` has no default here. The server's default applies when you leave it unset.
- OpenAI's chat API has no `top_k`. A request that sets it fails with `CucaError::Config` and the message ``"`top_k` is not supported by OpenAI"``.
- `tools` carries each `UnifiedRequest::tools` entry as `{"type": "function", "function": {name, description, parameters}}`. `parameters` holds the definition's `input_schema`. The key is omitted when the request has no tools.
- No `tool_choice` is sent, so the server's own default applies.

How message content maps:

- The `Text` blocks of a message are joined with a newline into one string.
- A message with an `ImageBase64` block becomes an array of `{type: "text"}` and `{type: "image_url"}` parts. `Thinking`, `ToolCall` and `ToolResult` blocks are left out of that array.
- A `Thinking` block becomes `reasoning_content` on an assistant message that has exactly one thinking block. In any other position it is dropped.
- `ToolCall` blocks become the assistant `tool_calls` array, with arguments as a JSON string. A message made only of tool calls has `content: null`.
- A `ToolResult` becomes a `role: "tool"` message with `tool_call_id` and the output as `content`.

## Response stream

- Each frame carries `choices[0].delta.{content, reasoning_content, tool_calls}`. The stream ends at `data: [DONE]`.
- A non-empty `content` delta becomes `MessageContentBlock::Text`.
- A non-empty `reasoning_content` delta becomes `MessageContentBlock::Thinking`, with no signature.
- Tool calls arrive in fragments. They are collected by the frame's `index`. A call is emitted as a `ToolCall` block once its arguments parse as JSON, or at a non-null `finish_reason` or `[DONE]`, whichever comes first.
- A fragment that never parses as JSON is emitted with `null` arguments. It is not dropped.
- Without `[DONE]`, the stream ends when the connection ends, after any pending tool calls are flushed.

## Token usage

The request sets `stream_options.include_usage`, so the last frame before `[DONE]` carries a `usage` object. Most servers send it as an extra frame with an empty `choices` array. DeepSeek sends it on the final `finish_reason` frame.

- The `usage` object never becomes a block. The last one seen wins.
- `prompt_tokens` and `completion_tokens` become `UnifiedResponse::prompt_tokens` and `completion_tokens`.
- The whole object becomes `UnifiedResponse::usage` as a `TokenUsage`. `reasoning_tokens` is set from `completion_tokens_details.reasoning_tokens` when the server reports it.
- A `usage` object missing either counter is ignored.
- With no usable `usage`, `usage` is `None`, `completion_tokens` counts one per `Text`, `Thinking` and `ToolCall` block, and `prompt_tokens` is `0`.

Plugins see the usage in `on_response_complete`. You read it per call with `CucaClient::generate_stream_with_response`:

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
- With a `thinking` config but no effort, `reasoning_effort` is `"medium"`.
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
| Frame is not valid JSON | `CucaError::Json`. The stream ends after it |
| Frame carries `{"error": {"message": ...}}` | `CucaError::Provider` with `provider: OpenAi`. The stream ends after it |
| Frame bytes are not valid UTF-8 | `CucaError::SseParse` |

The frame-level `Provider` error names `OpenAi` on every adapter that shares this translation: vLLM, LM Studio, DeepSeek native and llama.cpp chat.

## Limits

- No cap on request or response size. The client sets no request timeout.
- The SSE parser keeps one reusable buffer that starts at 8192 bytes. It grows to the largest single frame plus one partial line, not to the stream length.
- Pending tool calls hold their argument fragments until each call completes or `finish_reason` or `[DONE]` flushes it. Nothing else in the translator grows with the stream.
