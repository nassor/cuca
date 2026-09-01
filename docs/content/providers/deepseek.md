+++
title = "DeepSeek"
description = "The DeepSeek adapter's two wire protocols: the native OpenAI-compatible route and the Anthropic Messages API bridge."
template = "page.html"
weight = 4
+++

# DeepSeek

<dl class="page-facts">
<dt>In one line</dt>
<dd>Dispatches unified requests to DeepSeek over one of two wire protocols, chosen by the configured base URL.</dd>
<dt>You need</dt>
<dd>The <code>provider-deepseek</code> feature and an API key.</dd>
<dt>Read this if</dt>
<dd>You are routing requests through <code>ProviderEndpoint::DeepSeek</code>, on either its native route or the Anthropic bridge.</dd>
</dl>

The smallest streaming turn: the `DeepSeek` variant and an API key. The default
base URL selects the native route.

```rust,name=A first stream through the DeepSeek adapter
use cuca::types::{MessageContentBlock, ProviderEndpoint};
use cuca::{CucaClient, UnifiedRequest};
use tokio_stream::StreamExt;

let client = CucaClient::builder()
    .with_provider(ProviderEndpoint::DeepSeek)
    .with_api_key(std::env::var("DEEPSEEK_API_KEY")?)
    .build()?;

let mut stream = client
    .generate_stream(UnifiedRequest::new("deepseek-v4-flash").add_user_message("Say hello."))
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
| Feature flag | `provider-deepseek` |
| `ProviderEndpoint` variant | `DeepSeek` |
| Default base URL | `https://api.deepseek.com/v1`, used when the client's base URL is empty |

## Two wire protocols

The base URL decides which protocol a request speaks. A base URL containing `api.deepseek.com/anthropic`, or ending in `/anthropic`, selects the bridge; every other base URL, including the default, selects the native route.

| Route | Base URL | Wire protocol | Route posted |
|---|---|---|---|
| Native | `https://api.deepseek.com/v1` (default) | Shares the [OpenAI](@/providers/openai.md)-compatible adapter | `POST {base_url}/chat/completions` |
| Anthropic bridge | `https://api.deepseek.com/anthropic`, or any base URL ending in `/anthropic` | Shares the [Anthropic](@/providers/anthropic.md) Messages API adapter | `POST {base_url}/messages` |

## Native route

Requests and responses follow the shared OpenAI-compatible contract; see the OpenAI page for the full translation. DeepSeek's own difference: `req.thinking` becomes a top-level `thinking` mode object, `{"type": "enabled"}` or `{"type": "disabled"}`, instead of a `reasoning_effort` string. There is no effort knob on this object. `reasoning_content` in the response flows into `Thinking` blocks through the shared translator.

## Anthropic bridge

Requests follow the Anthropic Messages API translation; see the Anthropic page for the full request shape and thinking modes. Authentication is `x-api-key`, required: dispatch fails with `CucaError::Config("deepseek anthropic bridge requires an api key")` when no key is configured.

Two bridge-specific transforms are applied before the Anthropic body is built:

- The model id is translated from Claude-style names to DeepSeek ids: `claude-opus` becomes `deepseek-v4-pro`; `claude-sonnet` and `claude-haiku` become `deepseek-v4-flash`; any other model id passes through unchanged.
- `PromptCacheDirective` is forced to `Disabled` regardless of what the caller requested, so no bridge request can carry a prompt-cache beta header, `cache_control` block, or breakpoint.

## See also

[Prompt cache](@/services/prompt-cache.md) for the client-owned local response cache, unaffected by the bridge's provider-side prompt-cache restriction.
