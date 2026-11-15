+++
title = "Anthropic"
description = "The Anthropic Messages API adapter: a first request, auth modes, request and stream translation, thinking modes and prompt caching."
template = "page.html"
weight = 2
+++

# Anthropic

<dl class="page-facts">
<dt>In one line</dt>
<dd>Sends unified requests to the Anthropic Messages API with an <code>x-api-key</code> header or an OAuth bearer token.</dd>
<dt>You need</dt>
<dd>The <code>provider-anthropic</code> feature and either an API key or a bearer token.</dd>
<dt>Read this if</dt>
<dd>You route requests through <code>ProviderEndpoint::Anthropic</code>, use extended thinking or prompt caching, or want the Messages translation that <a href="@/providers/deepseek.md">DeepSeek</a> reuses for its bridge.</dd>
</dl>

## Send a first request

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

```text,name=Expected shape; not captured from a live run
Hello! How can I help you today?
blocks: 1 text, 0 thinking
```

To authenticate with a bearer token instead of a key, call `with_bearer_token` in place of `with_api_key`.

## What you need

| Fact | Value |
|---|---|
| Feature flag | `provider-anthropic` |
| `ProviderEndpoint` variant | `Anthropic` |
| Default base URL | `https://api.anthropic.com/v1`, used when the client's base URL is empty |
| Route | `POST {base_url}/messages` |
| Auth | `x-api-key`, or `Authorization: Bearer` for a token |

The default base URL ends in `/v1`. The adapter trims a trailing `/` and appends only `/messages`.

## How it differs from OpenAI

This adapter speaks the Anthropic Messages protocol. Against the [OpenAI](@/providers/openai.md) page it differs in four ways:

- The key goes in `x-api-key`, and every request carries `anthropic-version: 2023-06-01`.
- `max_tokens` is always sent and defaults to 1024.
- The stream has typed events, and a thinking block carries a signature.
- Token counts are not read. Only prompt cache counters are.

## Authentication

Exactly one credential is sent.

| Source | Header |
|---|---|
| `CucaClientBuilder::with_bearer_token` | `Authorization: Bearer <token>` |
| `CucaClientBuilder::with_api_key` | `x-api-key: <key>` |

- When both are set, the bearer token is used.
- Every request also carries `anthropic-version: 2023-06-01`.
- A request sends at most one `anthropic-beta` header, `prompt-caching-2024-07-31`. See [Prompt caching](#prompt-caching).
- With neither credential, dispatch fails with `CucaError::Config("anthropic requires an api key or bearer token")`.

## Request translation

- `model` is the request's model. `stream` is always `true`.
- `max_tokens` is required by the API. It defaults to 1024 when the request leaves it unset. Nothing caps a value you set.
- `temperature`, `top_p` and `top_k` are sent only when set. `UnifiedRequest::stop` becomes `stop_sequences`, only when non-empty.
- The API has no seed. A request that sets `seed` fails with `CucaError::Config` and the message ``"`seed` is not supported by Anthropic"``. The check runs first, before anything else is built.
- `system` is every `Text` block of every `System` message, joined with a newline, as one string. With a prompt cache breakpoint on a system block it is an array of `{type: "text"}` blocks instead. Non-text system blocks are dropped. With no system messages the key is omitted.
- `messages` has user and assistant turns only. Tool results ride inside a user message as `tool_result` blocks. Standalone `Tool`-role messages are skipped.
- `tools` carries each `UnifiedRequest::tools` entry as `{name, description, input_schema}`. The key is omitted when the request has no tools. No `tool_choice` is sent.

How content blocks map:

| Unified block | Anthropic block |
|---|---|
| `Text` | `{type: "text"}` |
| `ImageBase64` | `image` with a `base64` source |
| `Thinking` | `thinking`, with `signature` only when the block has one |
| `ToolCall` | `tool_use`, with `input` holding the arguments |
| `ToolResult` | `tool_result`, with `tool_use_id` and `content` |

## Response stream

The stream is made of typed events. At most one content block is in flight at a time.

<div class="dgm">
<div class="dgm-scroll">
<svg viewBox="0 0 660 150" role="img" aria-labelledby="ant-title ant-desc">
  <title id="ant-title">How Anthropic stream events become content blocks</title>
  <desc id="ant-desc">A text_delta event becomes a Text block right away. thinking_delta and signature_delta events are collected and become one Thinking block at content_block_stop. input_json_delta events are collected and become one ToolCall block at content_block_stop. message_stop ends the stream.</desc>
  <defs>
    <marker id="ant-arrow" viewBox="0 0 8 8" refX="7" refY="4"
            markerWidth="6" markerHeight="6" orient="auto">
      <path d="M0 0.6 L7 4 L0 7.4 z" fill="var(--muted-foreground)"/>
    </marker>
  </defs>
  <text class="t-title" x="6" y="20">Stream events to blocks</text>
  <rect class="blk blk-bnd" x="6" y="38" width="236" height="30" rx="6"/>
  <text class="t-sm t-mid t-bnd" x="124" y="57">text_delta</text>
  <rect class="blk blk-bnd" x="6" y="76" width="236" height="30" rx="6"/>
  <text class="t-sm t-mid t-bnd" x="124" y="95">thinking_delta, signature_delta</text>
  <rect class="blk blk-bnd" x="6" y="114" width="236" height="30" rx="6"/>
  <text class="t-sm t-mid t-bnd" x="124" y="133">input_json_delta</text>
  <rect class="blk blk-data" x="454" y="38" width="200" height="30" rx="6"/>
  <text class="t-sm t-mid t-data" x="554" y="57">Text</text>
  <rect class="blk blk-data" x="454" y="76" width="200" height="30" rx="6"/>
  <text class="t-sm t-mid t-data" x="554" y="95">Thinking, with signature</text>
  <rect class="blk blk-data" x="454" y="114" width="200" height="30" rx="6"/>
  <text class="t-sm t-mid t-data" x="554" y="133">ToolCall</text>
  <path class="arw arw-data" d="M244 53 H450" marker-end="url(#ant-arrow)"/>
  <text class="t-ax t-mid" x="330" y="49">at once</text>
  <path class="arw arw-data" d="M244 91 H450" marker-end="url(#ant-arrow)"/>
  <text class="t-ax t-mid" x="330" y="87">at content_block_stop</text>
  <path class="arw arw-data" d="M244 129 H450" marker-end="url(#ant-arrow)"/>
  <text class="t-ax t-mid" x="330" y="125">at content_block_stop</text>
</svg>
</div>
<p class="dgm-cap"><b>Only text is emitted at once.</b> Thinking and tool input are collected until the block stops, because the signature and the JSON arguments are incomplete before then.</p>
</div>

- `text_delta` becomes `MessageContentBlock::Text` immediately.
- `thinking_delta` and `signature_delta` are collected and emitted as one `Thinking` block, with its signature, at `content_block_stop`.
- `input_json_delta` fragments are collected and emitted as one `ToolCall` block at `content_block_stop`. Fragments that do not parse as JSON give `null` arguments.
- `message_start`, `message_delta` and `ping` produce no block. Unknown event types are ignored.
- `message_stop` ends the stream. The stream also ends when the connection ends.

## Token usage

- The adapter reads no `input_tokens` or `output_tokens`. `UnifiedResponse::usage` is `None`, `prompt_tokens` is `0`, and `completion_tokens` counts one per `Text`, `Thinking` and `ToolCall` block.
- Prompt cache counters are read from `message_start`. `usage.cache_read_input_tokens` becomes `PromptCacheUsage::read_tokens`, and `usage.cache_creation_input_tokens` becomes `write_tokens`.
- A counter present alone pairs with zero. With neither counter, `UnifiedResponse::prompt_cache_usage` is `None`.

## Thinking

A top-level `thinking` key is sent when `req.thinking` is set and enabled. A disabled or unset config sends no key.

- Budget mode sends `{"type": "enabled", "budget_tokens": N}`. `N` is `ThinkingParams::Anthropic { budget_tokens: Some(n), .. }` when set, otherwise the value from the table below.
- Adaptive mode applies when `ThinkingParams::Anthropic { adaptive: true, .. }` is set. It sends `{"type": "adaptive"}` plus `"effort"` when the unified effort is set.

| Unified `ThinkingEffort` | Budget mode `budget_tokens` | Adaptive mode `effort` |
|---|---|---|
| `Minimal` | 1024 | `"low"` |
| `Low` | 2048 | `"low"` |
| `Medium` | 8192 | `"medium"` |
| `High` | 16384 | `"high"` |
| `XHigh` | 16384 (same as `High`) | `"xhigh"` |
| unset | 10000 | key omitted |

## Prompt caching

Prompt caching lets Anthropic reuse the start of a long prompt across calls. You mark where the reusable part ends with a breakpoint. `PromptCacheDirective::Ephemeral` breakpoints add `"cache_control": {"type": "ephemeral"}` to the marked content blocks.

- The `anthropic-beta: prompt-caching-2024-07-31` header is sent exactly when the request's selected provider is Anthropic and the directive has at least one breakpoint.
- A breakpoint on a system block makes `system` an array of text blocks instead of one string.
- Each breakpoint is a `(message_index, block_index)` pair. A duplicate pair, an index out of range, a non-text system block, or a standalone `Tool`-role message is a `CucaError::Config`.
- The DeepSeek bridge forces the directive to `Disabled`. See [DeepSeek](@/providers/deepseek.md).

## Errors

| Condition | Error |
|---|---|
| Feature not compiled | `CucaError::ProviderNotEnabled("provider-anthropic")` |
| No API key and no bearer token | `CucaError::Config` |
| Key, token or beta value is not a valid header value | `CucaError::Config` |
| Invalid prompt cache breakpoint | `CucaError::Config`. See [Prompt caching](#prompt-caching) |
| Non-2xx response | `CucaError::Http { status, body }` with the captured body |
| Connection or body-read failure | `CucaError::Transport { message }` |
| `content_block_*` or `error` event is not valid JSON | `CucaError::Json` |
| `error` event | `CucaError::Provider` with `provider: Anthropic` and the event's message |
| Frame bytes are not valid UTF-8 | `CucaError::SseParse` |

A malformed frame or an `error` event ends the stream after the error.

## Limits

- No cap on request or response size. The client sets no request timeout.
- The SSE parser keeps one reusable buffer that starts at 8192 bytes. It grows to the largest single frame plus one partial line.
- One thinking text, one signature and one tool call input are buffered at a time, each until its `content_block_stop`.
- `max_tokens` defaults to 1024.

## OAuth 2.0 PKCE helpers

PKCE is the OAuth sign-in flow for apps that cannot keep a secret. `provider-anthropic` includes the pieces of the RFC 7636 authorization-code flow:

- `OAuthPkceConfig` holds `client_id`, `authorize_url`, `token_url` and `scopes`.
- `generate_pkce_pair()` returns a code verifier of 64 random bytes, base64url encoded without padding (86 characters), and the challenge `base64url(SHA-256(verifier))`. It fails with `CucaError::Io` when the system has no randomness source.
- `authorization_url(config, challenge, state)` builds the URL with `response_type=code`, `client_id`, `code_challenge`, `code_challenge_method=S256`, `state`, and `scope` (scopes joined by `+`) when scopes are set.
- `exchange_code(config, code, verifier, redirect_uri)` posts `grant_type=authorization_code` to the token endpoint and returns the `access_token`. It fails with `CucaError::Http` on a non-2xx response and `CucaError::Provider` when the response has no `access_token`.

These items live in the crate-private `provider` module, and the crate root does not re-export them. `CucaClientBuilder::with_anthropic_oauth` stores an `OAuthPkceConfig` that `CucaClient::oauth_config` returns. Dispatch does not read it, and nothing in the crate calls `exchange_code`. A token obtained by any means goes to `CucaClientBuilder::with_bearer_token`.
