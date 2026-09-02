+++
title = "Google Gemini"
description = "The Gemini streamGenerateContent adapter: endpoint, required auth, thinking levels, and its no-DONE-marker stream ending."
template = "page.html"
weight = 3
+++

# Google Gemini

<dl class="page-facts">
<dt>In one line</dt>
<dd>Dispatches unified requests to Google's streamGenerateContent endpoint.</dd>
<dt>You need</dt>
<dd>The <code>provider-gemini</code> feature and an API key. The key is required; there is no keyless mode.</dd>
<dt>Read this if</dt>
<dd>You are routing requests through <code>ProviderEndpoint::GoogleGemini</code> or handling tool calls against Gemini.</dd>
</dl>

The smallest streaming turn: the `GoogleGemini` variant and the required API
key.

Add the crate with `cargo add cuca --features provider-gemini`, `cargo add tokio --features rt,macros`, and `cargo add tokio-stream`.

```rust,name=A first stream through the Gemini adapter
use std::io::{Write, stdout};

use cuca::types::{MessageContentBlock, ProviderEndpoint};
use cuca::{CucaClient, UnifiedRequest};
use tokio_stream::StreamExt;

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = CucaClient::builder()
        .with_provider(ProviderEndpoint::GoogleGemini)
        .with_api_key(std::env::var("GEMINI_API_KEY")?)
        .build()?;

    let request = UnifiedRequest::new("gemini-2.5-flash")
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
| Feature flag | `provider-gemini` |
| `ProviderEndpoint` variant | `GoogleGemini` |
| Default base URL | `https://generativelanguage.googleapis.com`, used when the client's base URL is empty |
| Route | `POST {base_url}/v1beta/models/{model}:streamGenerateContent?alt=sse` |

The base URL carries no `/v1` suffix. The `/v1beta` API version segment is part of the route, appended by the adapter, not the base URL.

## Authentication

`x-goog-api-key` header, required on every request. Dispatch fails with `CucaError::Config("gemini requires an api key")` when no key is configured.

## Request body

- `system_instruction: { parts: [{ text }] }` from every `Text` block of every `System` message, joined with a newline; omitted with no system messages. Non-text blocks in system messages are dropped.
- `generationConfig` carries `temperature` and `maxOutputTokens` only when set; the key is omitted entirely when neither is set.
- `contents` holds one entry per non-system message: role `"user"` for `User` and `Tool` messages, `"model"` for `Assistant` messages.
- `tools` holds one entry whose `functionDeclarations` array carries every `UnifiedRequest::tools` definition as `{name, description, parameters}`, with `parameters` holding the definition's `input_schema`; the key is omitted when the request declares no tool, and no `toolConfig` is sent, so the API's own selection default applies.

## Tool calls

Gemini's `functionCall` parts carry no call id, so the unified `ToolCall::id` is dropped on the wire. A `Tool`-role `ToolResult` message becomes a `"user"`-role turn carrying a `functionResponse` part; the function name comes from the message's `name` annotation, falling back to the block's `tool_call_id` when `name` is unset.

## Thinking

A `thinkingConfig` key is emitted when `req.thinking` is set: `includeThoughts: false` when disabled; otherwise `thinkingBudget` (params override), else `thinkingLevel` (params override, then the unified effort map below), else `includeThoughts: true`.

| Unified `ThinkingEffort` | `thinkingLevel` |
|---|---|
| `Minimal` | `"LOW"` |
| `Low` | `"LOW"` |
| `Medium` | `"MEDIUM"` |
| `High` | `"HIGH"` |
| `XHigh` | `"HIGH"` |

## Streaming end

Gemini sends no `[DONE]` marker. The final frame carries only `usageMetadata`, which translates to no blocks, and the stream ends when the underlying byte stream ends.

## Out of scope

Non-streaming `generateContent` batching, and Google auth beyond the `x-goog-api-key` header. OAuth for GCP service accounts is not implemented.
