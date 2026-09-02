+++
title = "Anthropic"
description = "The Anthropic Messages API adapter: both auth modes, the OAuth 2.0 PKCE surface, thinking modes, and prompt caching."
template = "page.html"
weight = 2
+++

# Anthropic

<dl class="page-facts">
<dt>In one line</dt>
<dd>Dispatches unified requests to the Anthropic Messages API, with static API key or OAuth PKCE bearer auth.</dd>
<dt>You need</dt>
<dd>The <code>provider-anthropic</code> feature and either an API key or an OAuth-issued bearer token.</dd>
<dt>Read this if</dt>
<dd>You are routing requests through <code>ProviderEndpoint::Anthropic</code>, using extended thinking, or driving the OAuth authorization flow.</dd>
</dl>

The smallest streaming turn: the `Anthropic` variant and an API key. A bearer
token via `with_bearer_token` is the other auth mode; see Authentication.

Add the crate with `cargo add cuca --features provider-anthropic`, `cargo add tokio --features rt,macros`, and `cargo add tokio-stream`.

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

```text,name=Expected shape; not captured from a live run
Hello! How can I help you today?
blocks: 1 text, 0 thinking
```

## Endpoint

| Fact | Value |
|---|---|
| Feature flag | `provider-anthropic` |
| `ProviderEndpoint` variant | `Anthropic` |
| Default base URL | `https://api.anthropic.com/v1`, used when the client's base URL is empty |
| Route | `POST {base_url}/messages` |

## Authentication

Exactly one of two modes is sent:

| Mode | Header | Source |
|---|---|---|
| `AnthropicAuth::ApiKey` | `x-api-key` | `CucaClientBuilder::with_api_key` |
| `AnthropicAuth::Bearer` | `Authorization: Bearer` | an OAuth PKCE access token, `CucaClientBuilder::with_bearer_token` |

When both a bearer token and an API key are configured, the bearer token is used. Dispatch fails with `CucaError::Config` when neither is set. Every request also carries `anthropic-version: 2023-06-01` and one `anthropic-beta` header per required beta.

## OAuth 2.0 PKCE

The adapter implements the full authorization-code-with-PKCE flow (RFC 7636), gated to `provider-anthropic`:

- `OAuthPkceConfig`: client id, authorization and token endpoints, and requested scopes.
- `generate_pkce_pair()`: 64 random bytes, base64url encoded without padding (86 characters), as the code verifier; the code challenge is `base64url(SHA-256(verifier))`.
- `authorization_url(config, challenge, state)`: builds the authorization URL with `response_type=code`, `client_id`, `code_challenge`, `code_challenge_method=S256`, `state`, and `scope` when scopes are non-empty.
- `exchange_code(config, code, verifier, redirect_uri)`: exchanges the authorization code for an access token via `grant_type=authorization_code` at the token endpoint.

The resulting access token is passed to `CucaClientBuilder::with_bearer_token`, configured through `CucaClientBuilder::with_anthropic_oauth`.

## Request body

- `max_tokens` is required by the API and defaults to 1024 when the unified request leaves it unset.
- System message `Text` blocks are joined with a newline into the top-level `system` string; non-text system content is dropped; the key is omitted with no system messages.
- `messages` carries user and assistant turns only. Tool results ride inside a user message as `tool_result` blocks; standalone `Tool`-role messages are skipped.
- `tools` carries every `UnifiedRequest::tools` entry as `{name, description, input_schema}`; the key is omitted when the request declares no tool, and no `tool_choice` is sent, so the API's own selection default applies.

## Thinking

A top-level `thinking` key is emitted when `req.thinking` is set. Adaptive mode (`ThinkingParams::Anthropic { adaptive: true, .. }`) sends `{"type": "adaptive", "effort": ...}`; otherwise budget mode sends `{"type": "enabled", "budget_tokens": N}`, using the params override when present or the map below.

| Unified `ThinkingEffort` | Budget mode `budget_tokens` | Adaptive mode `effort` |
|---|---|---|
| `Minimal` | 1024 | `"low"` |
| `Low` | 2048 | `"low"` |
| `Medium` | 8192 | `"medium"` |
| `High` | 16384 | `"high"` |
| `XHigh` | 16384 (shares `High`'s budget) | `"xhigh"` |
| unset | 10000 | not applicable |

## Prompt caching

`PromptCacheDirective::Ephemeral` breakpoints add `"cache_control": {"type": "ephemeral"}` to the marked content blocks. The `anthropic-beta: prompt-caching-2024-07-31` header is sent exactly when the request's selected provider is Anthropic and the directive carries at least one breakpoint. Provider-reported cache usage is read from the `message_start` frame's `usage.cache_read_input_tokens` and `usage.cache_creation_input_tokens` fields into `PromptCacheUsage`.

## See also

[DeepSeek](@/providers/deepseek.md) reuses this adapter's Messages API translation for its Anthropic bridge.
