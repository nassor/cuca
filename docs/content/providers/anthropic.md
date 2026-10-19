+++
title = "Anthropic"
description = "The Anthropic Messages API adapter: auth modes, request and stream translation, thinking modes and prompt caching."
template = "page.html"
weight = 2
+++

# Anthropic

<dl class="page-facts">
<dt>In one line</dt>
<dd>Dispatches unified requests to the Anthropic Messages API with an <code>x-api-key</code> header or an OAuth bearer token.</dd>
<dt>You need</dt>
<dd>The <code>provider-anthropic</code> feature and either an API key or a bearer token.</dd>
<dt>Read this if</dt>
<dd>You route requests through <code>ProviderEndpoint::Anthropic</code>, use extended thinking or prompt caching, or want the Messages translation that <a href="@/providers/deepseek.md">DeepSeek</a> reuses for its bridge.</dd>
</dl>

This adapter speaks the Anthropic Messages protocol. Against the [OpenAI](@/providers/openai.md) page it differs in four ways:

- The key goes in `x-api-key`, and every request carries `anthropic-version: 2023-06-01`.
- `max_tokens` is always sent and defaults to 1024.
- The stream has typed events, and a thinking block carries a signature.
- Token counts are not read; only prompt cache counters are.

## Endpoint

| Fact | Value |
|---|---|
| Feature flag | `provider-anthropic` |
| `ProviderEndpoint` variant | `Anthropic` |
| Default base URL | `https://api.anthropic.com/v1`, used when the client's base URL is empty |
| Route | `POST {base_url}/messages` |

The default base URL carries `/v1`. The adapter appends only `/messages` and trims a trailing `/` first.

## Authentication

Exactly one credential is sent.

| Source | Header |
|---|---|
| `CucaClientBuilder::with_bearer_token` | `Authorization: Bearer <token>` |
| `CucaClientBuilder::with_api_key` | `x-api-key: <key>` |

- When both are set, the bearer token is used.
- Every request also carries `anthropic-version: 2023-06-01`.
- Dispatch sends at most one `anthropic-beta` header, `prompt-caching-2024-07-31`; see [Prompt caching](#prompt-caching).
- With neither credential, dispatch fails with `CucaError::Config("anthropic requires an api key or bearer token")`.

## Request translation

The request body is built from a `UnifiedRequest`:

- `model` is the request's model; `stream` is always `true`.
- `max_tokens` is required by the API and defaults to 1024 when the request leaves it unset.
- `temperature` is included only when set.
- `system` is every `Text` block of every `System` message joined with a newline, as one string. With a prompt cache breakpoint on a system block it is an array of `{type: "text"}` blocks instead. Non-text system blocks are dropped, and the key is omitted with no system messages.
- `messages` carries user and assistant turns only. Tool results ride inside a user message as `tool_result` blocks, and standalone `Tool`-role messages are skipped.
- `tools` carries every `UnifiedRequest::tools` entry as `{name, description, input_schema}`. The key is omitted when the request declares no tool, and no `tool_choice` is sent.
- Content blocks map as follows:
  - `Text` becomes `{type: "text"}`.
  - `ImageBase64` becomes an `image` block with a `base64` source.
  - `Thinking` becomes a `thinking` block, with `signature` only when the block has one.
  - `ToolCall` becomes `tool_use` with `input` holding the arguments.
  - `ToolResult` becomes `tool_result` with `tool_use_id` and `content`.

## Response stream

The SSE parser yields typed events, and the translator keeps at most one content block in flight.

- A `text_delta` becomes `MessageContentBlock::Text` immediately.
- `thinking_delta` and `signature_delta` accumulate and are emitted as one `Thinking` block, with its signature, at `content_block_stop`.
- `input_json_delta` fragments accumulate and are emitted as one `ToolCall` block at `content_block_stop`. Fragments that do not parse as JSON give `null` arguments.
- `message_start`, `message_delta` and `ping` produce no block. Unknown event types are ignored.
- `message_stop` ends the stream. The stream also ends when the byte stream ends.

## Token usage

- The adapter reads no `input_tokens` or `output_tokens`. `UnifiedResponse::usage` is `None`, `prompt_tokens` is `0`, and `completion_tokens` counts one per `Text`, `Thinking` and `ToolCall` block.
- Prompt cache counters are read from `message_start`: `usage.cache_read_input_tokens` becomes `PromptCacheUsage::read_tokens` and `usage.cache_creation_input_tokens` becomes `write_tokens`.
- A counter present alone pairs with zero. With neither counter, `UnifiedResponse::prompt_cache_usage` is `None`.

## Thinking

A top-level `thinking` key is sent when `req.thinking` is set and enabled. A disabled or unset config sends no key.

- Budget mode sends `{"type": "enabled", "budget_tokens": N}`. `N` is `ThinkingParams::Anthropic { budget_tokens: Some(n), .. }` when set, otherwise the map below.
- Adaptive mode applies when `ThinkingParams::Anthropic { adaptive: true, .. }` is set. It sends `{"type": "adaptive"}` plus `"effort"` when the unified effort is set.

| Unified `ThinkingEffort` | Budget mode `budget_tokens` | Adaptive mode `effort` |
|---|---|---|
| `Minimal` | 1024 | `"low"` |
| `Low` | 2048 | `"low"` |
| `Medium` | 8192 | `"medium"` |
| `High` | 16384 | `"high"` |
| `XHigh` | 16384 (same as `High`) | `"xhigh"` |
| unset | 10000 | key omitted |

## Errors

| Condition | Error |
|---|---|
| Feature not compiled | `CucaError::ProviderNotEnabled("provider-anthropic")` |
| No API key and no bearer token | `CucaError::Config` |
| Key, token or beta value is not a valid header value | `CucaError::Config` |
| Invalid prompt cache breakpoint | `CucaError::Config`; see [Prompt caching](#prompt-caching) |
| Non-2xx response | `CucaError::Http { status, body }` with the captured body |
| Connection or body-read failure | `CucaError::Transport { message }` |
| `content_block_*` or `error` event is not valid JSON | `CucaError::Json` |
| `error` event | `CucaError::Provider` with `provider: Anthropic` and the event's message |
| Frame bytes are not valid UTF-8 | `CucaError::SseParse` |

A malformed frame or an `error` event ends the stream after the error.

## Limits

- No cap on request or response size, and the client sets no request timeout.
- The SSE parser keeps one reusable buffer that starts at 8192 bytes. Its high-water mark is the largest single frame plus one partial line.
- One thinking text, one signature and one tool call input are buffered at a time, each until its `content_block_stop`.
- `max_tokens` defaults to 1024. Nothing caps a value the request sets.

## Prompt caching

`PromptCacheDirective::Ephemeral` breakpoints add `"cache_control": {"type": "ephemeral"}` to the marked content blocks.

- The `anthropic-beta: prompt-caching-2024-07-31` header is sent exactly when the request's selected provider is Anthropic and the directive has at least one breakpoint.
- A breakpoint on a system block makes `system` an array of text blocks instead of one string.
- Each breakpoint is a `(message_index, block_index)` pair. A duplicate pair, an index out of range, a non-text system block, or a standalone `Tool`-role message is a `CucaError::Config`.
- The DeepSeek bridge forces the directive to `Disabled`; see [DeepSeek](@/providers/deepseek.md).

## OAuth 2.0 PKCE helpers

The `provider-anthropic` feature adds the pieces of the RFC 7636 authorization-code flow:

- `OAuthPkceConfig` holds `client_id`, `authorize_url`, `token_url` and `scopes`.
- `generate_pkce_pair()` returns a code verifier of 64 random bytes, base64url encoded without padding (86 characters), and the challenge `base64url(SHA-256(verifier))`. It fails with `CucaError::Io` when the system has no randomness source.
- `authorization_url(config, challenge, state)` builds the URL with `response_type=code`, `client_id`, `code_challenge`, `code_challenge_method=S256`, `state`, and `scope` (scopes joined by `+`) when scopes are set.
- `exchange_code(config, code, verifier, redirect_uri)` posts `grant_type=authorization_code` to the token endpoint and returns the `access_token`. It fails with `CucaError::Http` on a non-2xx response and `CucaError::Provider` when the response has no `access_token`.

These items live in the crate-private `provider` module, and the crate root does not re-export them. `CucaClientBuilder::with_anthropic_oauth` stores an `OAuthPkceConfig` that `CucaClient::oauth_config` returns. Dispatch does not read it and nothing in the crate calls `exchange_code`. A token obtained by any means is passed to `CucaClientBuilder::with_bearer_token`.

## Example

The smallest streaming turn: the `Anthropic` variant and an API key. A bearer token through `with_bearer_token` is the other auth mode.

```bash,name=Runs the same on all three platforms
cargo add cuca --path /path/to/cuca --features provider-anthropic
cargo add tokio --features rt,macros
cargo add tokio-stream
```

```bash,name=Linux and macOS: set the key
export ANTHROPIC_API_KEY="sk-ant-..."
```

```powershell,name=Windows (PowerShell): set the key
$env:ANTHROPIC_API_KEY = "sk-ant-..."
```

```rust,name=A first stream through the Anthropic adapter
use std::io::{Write, stdout};

use cuca::types::{MessageContentBlock, ProviderEndpoint};
use cuca::{CucaClient, UnifiedRequest};
use tokio_stream::StreamExt;

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = CucaClient::builder()
        .with_provider(ProviderEndpoint::Anthropic)
        .with_api_key(std::env::var("ANTHROPIC_API_KEY")?)
        .build()?;

    let request = UnifiedRequest::new("claude-sonnet-4-0")
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

The `provider-anthropic` feature comes from the `cargo add` line above.
