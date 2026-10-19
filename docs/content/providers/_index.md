+++
title = "Providers"
description = "The eight provider adapters, seven over HTTP and one in-process, with each one's feature flag, default base URL and auth header."
template = "section.html"
sort_by = "weight"

[extra]
kicker = "Reference"
+++

<dl class="page-facts">
<dt>In one line</dt>
<dd>Eight <code>provider-*</code> features, each owning one <code>dispatch_*</code> method and one <code>ProviderEndpoint</code> variant</dd>
<dt>You need</dt>
<dd>At least one <code>provider-*</code> feature enabled; the crate does not compile without one</dd>
<dt>Read this if</dt>
<dd>You need a base URL, an auth header, a route, or the exact behavior one adapter does not share with the others</dd>
</dl>

## Compare the adapters

| Feature | `ProviderEndpoint` | Default base URL | Auth header |
|---|---|---|---|
| [`provider-openai`](@/providers/openai.md) | `OpenAi` | `https://api.openai.com/v1` | `Authorization: Bearer`, when a key is set |
| [`provider-anthropic`](@/providers/anthropic.md) | `Anthropic` | `https://api.anthropic.com/v1` | `x-api-key`, or `Authorization: Bearer` for a bearer token |
| [`provider-gemini`](@/providers/gemini.md) | `GoogleGemini` | `https://generativelanguage.googleapis.com` | `x-goog-api-key`, required |
| [`provider-deepseek`](@/providers/deepseek.md) | `DeepSeek` | `https://api.deepseek.com/v1` | `Authorization: Bearer` (native), `x-api-key` (bridge) |
| [`provider-llamacpp`](@/providers/llamacpp.md) | `LlamaCpp` | `http://127.0.0.1:8080` | `Authorization: Bearer`, when a key is set |
| [`provider-vllm`](@/providers/vllm.md) | `Vllm` | `http://127.0.0.1:8000/v1` | `Authorization: Bearer`, when a key is set |
| [`provider-lmstudio`](@/providers/lmstudio.md) | `LmStudio` | `http://127.0.0.1:1234/v1` | `Authorization: Bearer`, when a key is set |
| [`provider-local`](@/providers/local/_index.md) | `Local` | none, runs in your process | none |

- The default base URL applies when the client's base URL is empty. Set another with `CucaClientBuilder::with_base_url`.
- Each provider page lists the route, the request and stream translation, thinking, errors and limits under the same headings.
- Every HTTP feature enables `reqwest` with its `rustls` backend and `tokio-stream`. `provider-anthropic` also enables `sha2`, `getrandom` and `base64`.
- [The feature matrix](@/reference/features.md) maps every feature to its dependencies.

## Wire formats

The seven HTTP adapters speak four wire formats and share two translation modules.

| Wire format | Adapters | Stream ends at |
|---|---|---|
| OpenAI-compatible `/chat/completions` | OpenAI, vLLM, LM Studio, DeepSeek native, llama.cpp chat | `data: [DONE]` |
| Anthropic Messages | Anthropic, DeepSeek bridge | `message_stop` event |
| Gemini `streamGenerateContent` | Gemini | the end of the byte stream |
| llama.cpp native `/completion` | llama.cpp with `LlamaRoute::Completion` | the end of the byte stream |

- `src/provider/openai_compat.rs` holds the OpenAI-compatible translation, documented on the [OpenAI](@/providers/openai.md) page.
- `src/provider/anthropic.rs` holds the Anthropic translation, documented on the [Anthropic](@/providers/anthropic.md) page.
- Gemini and the llama.cpp native route each have their own translator.

## The in-process adapter

`provider-local` runs a GGUF language model inside your process and exposes no HTTP endpoint. It has no base URL, auth header or route.

- `ProviderEndpoint::Local` selects it, and `CucaClientBuilder::with_local_model` attaches the model. `build` fails when the selected provider is `Local` and no model is attached.
- The CPU is the default device. `provider-local-vulkan`, `provider-local-metal`, `provider-local-cuda` and `provider-local-rocm` add a GPU engine that `LocalModelConfig::device` selects.
- It does not use `tokio-stream`. It uses `reqwest` only for the one-time model download.
- The engine is pure Rust. The `candle` dependency tree builds one C crate, `onig_sys`.

See [Local (in-process)](@/providers/local/_index.md) for configuration and behavior, and [Local models and formats](@/providers/local/models.md) for the supported architectures and presets.

## Dispatch

There is no provider trait. Each adapter module contributes an inherent `impl CucaClient` block with one method:

```rust,name=The shape every adapter implements
pub(crate) async fn dispatch_<provider>(
    &self,
    req: UnifiedRequest,
) -> Result<ProviderDispatch, CucaError>
```

`CucaClient::generate_stream` sets `request.provider` to the client's selected provider, runs the `on_request` hooks, then selects an adapter with a `match` on `self.selected_provider`. Each arm is gated on its feature. A mirrored `#[cfg(not(feature = ...))]` arm returns `CucaError::ProviderNotEnabled` with the missing flag name.

## Custom endpoints

`ProviderEndpoint` has nine variants: the eight above and `Custom(String)`. `Custom` has no feature and no adapter. Dispatching it returns `CucaError::Config`:

```text,name=CucaError::Config from the Custom arm
configuration error: custom endpoints require a registered adapter
```

- `ProviderEndpoint::default()` is `OpenAi`.
- `UnifiedRequest::new` sets the request's provider to `Custom(String::new())`, and `generate_stream` overwrites it with the client's provider before any hook runs.

## Environment variables

None of the seven HTTP adapters reads an environment variable. Base URLs, API keys and bearer tokens reach them only through `CucaClientBuilder`. The `CUCA_*` variables that belong to the examples and the test harness are listed in [Environment variables](@/reference/environment.md).

`provider-local` reads two variables as fallbacks:

- `CUCA_CACHE_DIR`, then the platform's cache variables, when `LocalModelConfig::cache_dir` is `None`.
- `HF_TOKEN` when `LocalModelConfig::hf_token` is `None`.

Setting the config fields keeps it from reading either.

## Related pages

- [Unified request](@/concepts/unified-request.md) explains the request every adapter translates.
- [Use another OpenAI-compatible server](@/guides/other-openai-server.md) shows how to point an adapter at a server not listed here.
