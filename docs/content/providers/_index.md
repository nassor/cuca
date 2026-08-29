+++
title = "Providers"
description = "Seven provider adapters, three wire protocols, one dispatch match. Default base URLs, auth headers and routes for all of them."
template = "section.html"
sort_by = "weight"

[extra]
kicker = "Reference"
+++

<dl class="page-facts">
<dt>In one line</dt>
<dd>Seven <code>provider-*</code> features, each owning one <code>dispatch_*</code> method and one <code>ProviderEndpoint</code> variant</dd>
<dt>You need</dt>
<dd>At least one <code>provider-*</code> feature enabled; the crate does not compile without one</dd>
<dt>Read this if</dt>
<dd>You need a base URL, an auth header, a route, or the exact behavior one adapter does not share with the others</dd>
</dl>

## The seven adapters

| Feature | `ProviderEndpoint` | Default base URL | Auth header | Route |
|---|---|---|---|---|
| [`provider-openai`](@/providers/openai.md) | `OpenAi` | `https://api.openai.com/v1` | `Authorization: Bearer`, when a key is set | `POST {base}/chat/completions` |
| [`provider-anthropic`](@/providers/anthropic.md) | `Anthropic` | `https://api.anthropic.com/v1` | `x-api-key` or `Authorization: Bearer`, plus `anthropic-version: 2023-06-01` | `POST {base}/messages` |
| [`provider-gemini`](@/providers/gemini.md) | `GoogleGemini` | `https://generativelanguage.googleapis.com` | `x-goog-api-key`, required | `POST {base}/v1beta/models/{model}:streamGenerateContent?alt=sse` |
| [`provider-deepseek`](@/providers/deepseek.md) | `DeepSeek` | `https://api.deepseek.com/v1` | `Authorization: Bearer` on the native route, `x-api-key` on the bridge route | `POST {base}/chat/completions`, or `POST {base}/messages` on the bridge |
| [`provider-llamacpp`](@/providers/llamacpp.md) | `LlamaCpp` | `http://127.0.0.1:8080` | `Authorization: Bearer`, when a key is set | `POST {base}/v1/chat/completions`, or `POST {base}/completion` on the native route |
| [`provider-vllm`](@/providers/vllm.md) | `Vllm` | `http://127.0.0.1:8000/v1` | `Authorization: Bearer`, optional | `POST {base}/chat/completions` |
| [`provider-lmstudio`](@/providers/lmstudio.md) | `LmStudio` | `http://127.0.0.1:1234/v1` | `Authorization: Bearer`, optional | `POST {base}/chat/completions` |

Every feature in the table enables `reqwest` with its `rustls` backend and
`tokio-stream`. `provider-anthropic` additionally enables `sha2`, `getrandom`
and `base64` for its OAuth PKCE surface. The full dependency mapping is
[The feature matrix](@/reference/features.md).

## Three wire protocols, two shared modules

Five of the seven speak the OpenAI-compatible `/chat/completions` protocol and
share one module, `src/provider/openai_compat.rs`: OpenAI, vLLM, LM Studio,
DeepSeek's native route, and llama.cpp's chat route. Two speak the Anthropic
Messages protocol and share `src/provider/anthropic.rs`: Anthropic, and
DeepSeek's bridge route.

Gemini shares neither. llama.cpp's native `/completion` route shares neither.

| Module | Serves | Frame terminator |
|---|---|---|
| `openai_compat.rs` | `provider-openai`, `provider-vllm`, `provider-lmstudio`, `provider-deepseek` native, `provider-llamacpp` chat | `data: [DONE]` |
| `anthropic.rs` | `provider-anthropic`, `provider-deepseek` bridge | `message_stop` event |
| `gemini.rs` | `provider-gemini` | none; the byte stream ends |
| `llamacpp.rs` native route | `provider-llamacpp` with `LlamaRoute::Completion` | a frame with `"stop": true` |

## Dispatch

There is no provider trait. Each adapter module contributes an inherent
`impl CucaClient` block holding one method:

```rust,name=The shape every adapter implements
pub(crate) async fn dispatch_<provider>(
    &self,
    req: UnifiedRequest,
) -> Result<ProviderDispatch, CucaError>
```

`CucaClient::generate_stream` selects among them with a `match` on
`self.selected_provider`. Each arm is gated on its own feature, and every arm
has a mirrored `#[cfg(not(feature = ...))]` counterpart returning
`CucaError::ProviderNotEnabled` with the missing flag name.

## The eighth variant

`ProviderEndpoint` has an eighth variant, `Custom(String)`, which is not gated
by any feature and has no adapter. Dispatching it always returns:

```text,name=CucaError::Config from the Custom arm
configuration error: custom endpoints require a registered adapter
```

The default value of `ProviderEndpoint` is `OpenAi`. `UnifiedRequest::new` sets
the field to `Custom(String::new())`, which `generate_stream` then overwrites
with the client's own selected provider before any hook runs.

## No provider reads the environment

No file under `src/provider/` reads an environment variable. Base URLs, API keys
and bearer tokens reach an adapter only through `CucaClientBuilder`. The three
`CUCA_*` variables that exist belong to the examples and the test harness, and
are listed in [Environment variables](@/reference/environment.md).

## Per-provider pages
