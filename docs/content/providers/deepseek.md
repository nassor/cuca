+++
title = "DeepSeek"
description = "The DeepSeek adapter's two wire protocols: the native OpenAI-compatible route and the Anthropic Messages bridge, chosen by base URL."
template = "page.html"
weight = 4
+++

# DeepSeek

<dl class="page-facts">
<dt>In one line</dt>
<dd>Dispatches unified requests to DeepSeek over the OpenAI-compatible route or the Anthropic Messages bridge, chosen by the base URL.</dd>
<dt>You need</dt>
<dd>The <code>provider-deepseek</code> feature and an API key. The bridge route requires the key.</dd>
<dt>Read this if</dt>
<dd>You route requests through <code>ProviderEndpoint::DeepSeek</code> on either route, or need the bridge's model id and prompt cache rules.</dd>
</dl>

The adapter reuses two other adapters instead of translating on its own. The native route runs the code on the [OpenAI](@/providers/openai.md) page, and the bridge runs the code on the [Anthropic](@/providers/anthropic.md) page. Its own differences:

- A base URL containing `api.deepseek.com/anthropic`, or ending in `/anthropic`, selects the bridge. Every other base URL selects the native route.
- Native thinking is a `thinking` mode object with no effort level.
- The bridge translates Claude-style model ids and always disables prompt caching.

## Endpoint

| Fact | Native route | Bridge route |
|---|---|---|
| Feature flag | `provider-deepseek` | `provider-deepseek` |
| `ProviderEndpoint` variant | `DeepSeek` | `DeepSeek` |
| Default base URL | `https://api.deepseek.com/v1`, used when the client's base URL is empty | none; set `https://api.deepseek.com/anthropic` or another URL ending in `/anthropic` |
| Route | `POST {base_url}/chat/completions` | `POST {base_url}/messages` |
| Wire protocol | OpenAI-compatible | Anthropic Messages |

The native route trims a trailing `/` and appends `/chat/completions`. The bridge uses the base URL as given, with no `/v1` handling.

## Authentication

| | Native route | Bridge route |
|---|---|---|
| Header | `Authorization: Bearer <api_key>` | `x-api-key: <api_key>` |
| Without a key | no header is sent | `CucaError::Config("deepseek anthropic bridge requires an api key")` |
| Other headers | none | `anthropic-version: 2023-06-01` |

The bridge ignores `CucaClientBuilder::with_bearer_token`. It sends no `anthropic-beta` header.

## Request translation

Native route: the body is the one described under [OpenAI request translation](@/providers/openai.md#request-translation), with the request's model unchanged.

Bridge route: the body is the one described under [Anthropic request translation](@/providers/anthropic.md#request-translation), after two changes to the request:

- The model id is mapped: `claude-opus` becomes `deepseek-v4-pro`, and `claude-sonnet` and `claude-haiku` become `deepseek-v4-flash`. Any other id passes through unchanged.
- `PromptCacheDirective` is set to `Disabled`, whatever the caller requested. No bridge request carries a prompt cache beta header, a `cache_control` block or a breakpoint check.

## Response stream

- Native route: the stream is the one described under [OpenAI response stream](@/providers/openai.md#response-stream). `reasoning_content` deltas become `Thinking` blocks.
- Bridge route: the stream is the one described under [Anthropic response stream](@/providers/anthropic.md#response-stream).

## Token usage

- Native route: DeepSeek sends the `usage` object on the final `finish_reason` frame, and the shared translator reads it. The rules are on the [OpenAI](@/providers/openai.md#token-usage) page.
- Bridge route: no token counts are read, as on the [Anthropic](@/providers/anthropic.md#token-usage) page. `usage` is `None`, `prompt_tokens` is `0`, and `completion_tokens` counts one per `Text`, `Thinking` and `ToolCall` block.
- The bridge still reads prompt cache counters from `message_start` when the server sends them.

## Thinking

- Native route: `req.thinking` becomes a top-level `thinking` object. An enabled config sends `{"type": "enabled"}` and a disabled one sends `{"type": "disabled"}`. With no `thinking` on the request, the key is omitted. No `reasoning_effort` is sent, and the effort level and `ThinkingParams` are ignored.
- Bridge route: `req.thinking` follows the budget and adaptive rules in the [Anthropic thinking table](@/providers/anthropic.md#thinking).

## Errors

| Condition | Error |
|---|---|
| Feature not compiled | `CucaError::ProviderNotEnabled("provider-deepseek")` |
| Bridge route with no API key | `CucaError::Config` |
| Non-2xx response | `CucaError::Http { status, body }` with the captured body |
| Connection or body-read failure | `CucaError::Transport { message }` |
| Native frame is not valid JSON | `CucaError::Json` |
| Native frame carries `{"error": {"message": ...}}` | `CucaError::Provider` with `provider: OpenAi` |
| Bridge `error` event | `CucaError::Provider` with `provider: Anthropic` |
| Frame bytes are not valid UTF-8 | `CucaError::SseParse` |

The provider name in a stream error comes from the shared translator: `OpenAi` on the native route and `Anthropic` on the bridge, never `DeepSeek`.

## Limits

- No cap on request or response size, and the client sets no request timeout.
- The SSE parser keeps one reusable buffer that starts at 8192 bytes. Its high-water mark is the largest single frame plus one partial line.
- Native route: in-flight tool calls hold their argument fragments until each call completes. Bridge route: one thinking text, signature or tool call input is buffered at a time.
- The bridge sends `max_tokens` 1024 when the request leaves it unset. The native route sends none.

## Example

The smallest streaming turn: the `DeepSeek` variant and an API key. The default base URL selects the native route.

```bash,name=Runs the same on all three platforms
cargo add cuca --path /path/to/cuca --features provider-deepseek
cargo add tokio --features rt,macros
cargo add tokio-stream
```

```bash,name=Linux and macOS: set the key
export DEEPSEEK_API_KEY="sk-..."
```

```powershell,name=Windows (PowerShell): set the key
$env:DEEPSEEK_API_KEY = "sk-..."
```

```rust,name=A first stream through the DeepSeek adapter
use std::io::{Write, stdout};

use cuca::types::{MessageContentBlock, ProviderEndpoint};
use cuca::{CucaClient, UnifiedRequest};
use tokio_stream::StreamExt;

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = CucaClient::builder()
        .with_provider(ProviderEndpoint::DeepSeek)
        .with_api_key(std::env::var("DEEPSEEK_API_KEY")?)
        .build()?;

    let request = UnifiedRequest::new("deepseek-v4-flash")
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

The `provider-deepseek` feature comes from the `cargo add` line above.

To use the bridge, add `.with_base_url("https://api.deepseek.com/anthropic")` to the builder.

## Related pages

- [Prompt cache](@/services/prompt-cache.md) is the client-owned local response cache, a separate feature from the provider-side prompt caching the bridge disables.
