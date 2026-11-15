+++
title = "Google Gemini"
description = "The Gemini streamGenerateContent adapter: a first request, the required API key, request and stream translation, and thinking levels."
template = "page.html"
weight = 3
+++

# Google Gemini

<dl class="page-facts">
<dt>In one line</dt>
<dd>Sends unified requests to Google's <code>streamGenerateContent</code> endpoint with an <code>x-goog-api-key</code> header.</dd>
<dt>You need</dt>
<dd>The <code>provider-gemini</code> feature and an API key. There is no keyless mode.</dd>
<dt>Read this if</dt>
<dd>You route requests through <code>ProviderEndpoint::GoogleGemini</code>, handle tool calls against Gemini, or set Gemini thinking budgets and levels.</dd>
</dl>

## Send a first request

```bash,name=Runs the same on all three platforms
cargo add cuca --path /path/to/cuca --features provider-gemini
cargo add tokio --features rt,macros
cargo add tokio-stream
```

```bash,name=Linux and macOS: set the key
export GEMINI_API_KEY="..."
```

```powershell,name=Windows (PowerShell): set the key
$env:GEMINI_API_KEY = "..."
```

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

```bash,name=Runs the same on all three platforms
cargo run
```

```text,name=Expected shape; not captured from a live run
Hello! How can I help you today?
blocks: 1 text, 0 thinking
```

Gemini models may stream thinking blocks before the answer, so the thinking count can be above zero.

## What you need

| Fact | Value |
|---|---|
| Feature flag | `provider-gemini` |
| `ProviderEndpoint` variant | `GoogleGemini` |
| Default base URL | `https://generativelanguage.googleapis.com`, used when the client's base URL is empty |
| Route | `POST {base_url}/v1beta/models/{model}:streamGenerateContent?alt=sse` |
| Auth | `x-goog-api-key: <api_key>`, required |

The base URL has no `/v1` suffix. The adapter trims a trailing `/`, then appends the `/v1beta` segment, the model id and the method.

## How it differs from OpenAI

This adapter speaks Gemini's own `contents` and `parts` protocol. Against the [OpenAI](@/providers/openai.md) page it differs in four ways:

- The API key is required and travels in `x-goog-api-key`.
- The model id is part of the URL path, not the body.
- The stream has no end marker. It ends when the connection ends.
- Token usage is not read.

## Authentication

- `x-goog-api-key: <api_key>` is sent on every request.
- With no API key, dispatch fails with `CucaError::Config("gemini requires an api key")`.
- Google OAuth for service accounts is not implemented.

## Request translation

- `system_instruction` is `{parts: [{text}]}`, built from every `Text` block of every `System` message, joined with a newline. Non-text system blocks are dropped. With no system messages the key is omitted.
- `generationConfig` carries `temperature`, `maxOutputTokens`, `topP`, `topK`, `seed` and `stopSequences`, each only when set or non-empty. The whole key is omitted when none is set.
- `maxOutputTokens` is sent only when the request sets `max_tokens`. There is no adapter default.
- `contents` has one entry per non-system message. The role is `"user"` for `User` and `Tool` messages and `"model"` for `Assistant` messages.
- `tools` is one entry whose `functionDeclarations` array holds every `UnifiedRequest::tools` definition as `{name, description, parameters}`. `parameters` holds the `input_schema`. The key is omitted when the request has no tools. No `toolConfig` is sent.

How content blocks map to `parts`:

| Unified block | Gemini part |
|---|---|
| `Text` | `{text}` |
| `ImageBase64` | `{inline_data: {mime_type, data}}` |
| `Thinking` | `{thought: true, text}`. The signature is dropped |
| `ToolCall` | `{functionCall: {name, args}}`. The call id is dropped |
| `ToolResult` | `{functionResponse: {name, response: {output}}}` |

Gemini's `functionResponse` names the function, not the call. `name` is the message's `name` annotation, falling back to the block's `tool_call_id` when `name` is unset.

## Response stream

Each frame's `candidates[0].content.parts` maps to blocks, in order:

- A part with `thought: true` becomes `MessageContentBlock::Thinking`, with no signature.
- A part with `text` becomes `MessageContentBlock::Text`.
- A part with `functionCall` becomes `MessageContentBlock::ToolCall`. The arguments are the `args` object as sent, or `null` when `args` is missing. The `id` is read when the frame has one and is an empty string otherwise.
- A frame without a candidate or parts, such as the final `usageMetadata` frame, yields no blocks.
- Gemini sends no `[DONE]` marker. The stream ends when the connection ends.

## Token usage

- The adapter ignores `usageMetadata`. `UnifiedResponse::usage` is `None`, `prompt_tokens` is `0`, and `completion_tokens` counts one per `Text`, `Thinking` and `ToolCall` block.
- No prompt cache counters are read.

## Thinking

A `thinkingConfig` key is sent when `req.thinking` is set. The first matching rule applies:

1. A disabled config sends `{"includeThoughts": false}`.
2. `ThinkingParams::Gemini { thinking_budget: Some(n), .. }` sends `{"thinkingBudget": n}`.
3. `ThinkingParams::Gemini { thinking_level: Some(level), .. }` sends `{"thinkingLevel": level}`, with the string as given.
4. A unified effort sends `{"thinkingLevel": ...}` from the table below.
5. Otherwise it sends `{"includeThoughts": true}`.

| Unified `ThinkingEffort` | `thinkingLevel` |
|---|---|
| `Minimal` | `"LOW"` |
| `Low` | `"LOW"` |
| `Medium` | `"MEDIUM"` |
| `High` | `"HIGH"` |
| `XHigh` | `"HIGH"` |

## Errors

| Condition | Error |
|---|---|
| Feature not compiled | `CucaError::ProviderNotEnabled("provider-gemini")` |
| No API key | `CucaError::Config` |
| Non-2xx response | `CucaError::Http { status, body }` with the captured body |
| Connection or body-read failure | `CucaError::Transport { message }` |
| Frame is not valid JSON | `CucaError::Json` |
| Frame carries `{"error": {"message": ...}}` | `CucaError::Provider` with `provider: GoogleGemini` |
| Frame bytes are not valid UTF-8 | `CucaError::SseParse` |

A malformed frame or an error frame ends the stream after the error.

## Limits

- No cap on request or response size. The client sets no request timeout.
- The SSE parser keeps one reusable buffer that starts at 8192 bytes. It grows to the largest single frame plus one partial line.
- Non-streaming `generateContent` is not implemented.
