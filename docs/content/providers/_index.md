+++
title = "Providers"
description = "The eight provider adapters, seven over HTTP and one in-process: pick one, set its feature flag, base URL and key, and send the same request to any of them."
template = "section.html"
sort_by = "weight"

[extra]
kicker = "Reference"
+++

<dl class="page-facts">
<dt>In one line</dt>
<dd>Eight <code>provider-*</code> features. Each owns one adapter that turns a <code>UnifiedRequest</code> into that provider's wire format.</dd>
<dt>You need</dt>
<dd>At least one <code>provider-*</code> feature enabled. The crate does not compile without one.</dd>
<dt>Read this if</dt>
<dd>You need a base URL, an auth header, a route, or the exact behavior one adapter does not share with the others</dd>
</dl>

A provider is the service that runs the model. CUCA has one adapter per provider. You write the request once as a `UnifiedRequest`, pick a provider on the client, and the adapter does the rest.

## What you need

- One `provider-*` feature in `Cargo.toml`, for example `provider-openai`.
- For a hosted provider, an API key. A local server (llama.cpp, vLLM, LM Studio) needs none unless you turned auth on.
- A base URL only when you do not use the default. The table below lists the defaults.

## Send a first request

This client talks to OpenAI. To use another provider, change the `ProviderEndpoint` variant, the key and the model id.

```rust,name=One request through any adapter
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
        .add_user_message("Say hello.")
        .set_max_tokens(128);
    let mut stream = client.generate_stream(request).await?;

    while let Some(block) = stream.next().await {
        if let MessageContentBlock::Text(text) = block? {
            print!("{text}");
            stdout().flush()?;
        }
    }
    println!();
    Ok(())
}
```

```text,name=Expected shape; not captured from a live run
Hello! How can I help you today?
```

The reply arrives as a stream. Each `Text` block is one piece of it, printed as it lands. Each provider page below has the same example with its own variant and model id.

## How a request reaches a provider

`CucaClient::generate_stream` sets the request's provider to the client's, runs your plugin hooks, then calls the matching adapter. The seven HTTP adapters share four translators, because several providers speak the same wire format.

<div class="dgm">
<div class="dgm-scroll">
<svg viewBox="0 0 660 266" role="img" aria-labelledby="prov-title prov-desc">
  <title id="prov-title">One unified request fans out to the provider adapters</title>
  <desc id="prov-desc">A UnifiedRequest goes to the provider-local engine, which runs in your process, or across the HTTP boundary to one of four translators: OpenAI-compatible chat for five providers, Anthropic Messages for Anthropic and the DeepSeek bridge, Gemini streamGenerateContent, and the llama.cpp native completion route.</desc>
  <defs>
    <marker id="prov-arrow" viewBox="0 0 8 8" refX="7" refY="4"
            markerWidth="6" markerHeight="6" orient="auto">
      <path d="M0 0.6 L7 4 L0 7.4 z" fill="var(--muted-foreground)"/>
    </marker>
  </defs>
  <text class="t-title" x="6" y="20">CucaClient::generate_stream</text>
  <path class="seam" d="M250 30 V252"/>
  <text class="t-ax t-bnd" x="256" y="24">HTTP BOUNDARY</text>
  <rect class="blk blk-data" x="6" y="90" width="130" height="60" rx="6"/>
  <text class="t-lbl t-mid t-data" x="71" y="115">UnifiedRequest</text>
  <text class="t-sm t-mid" x="71" y="132">one request type</text>
  <rect class="blk" x="6" y="196" width="130" height="50" rx="6"/>
  <text class="t-lbl t-mid" x="71" y="217">provider-local</text>
  <text class="t-sm t-mid" x="71" y="233">in your process</text>
  <rect class="blk blk-bnd" x="300" y="34" width="354" height="44" rx="6"/>
  <text class="t-lbl t-mid t-bnd" x="477" y="53">OpenAI-compatible chat</text>
  <text class="t-sm t-mid" x="477" y="69">OpenAI, vLLM, LM Studio, DeepSeek, llama.cpp</text>
  <rect class="blk blk-bnd" x="300" y="88" width="354" height="44" rx="6"/>
  <text class="t-lbl t-mid t-bnd" x="477" y="107">Anthropic Messages</text>
  <text class="t-sm t-mid" x="477" y="123">Anthropic, DeepSeek bridge</text>
  <rect class="blk blk-bnd" x="300" y="142" width="354" height="44" rx="6"/>
  <text class="t-lbl t-mid t-bnd" x="477" y="161">Gemini streamGenerateContent</text>
  <text class="t-sm t-mid" x="477" y="177">Gemini</text>
  <rect class="blk blk-bnd" x="300" y="196" width="354" height="44" rx="6"/>
  <text class="t-lbl t-mid t-bnd" x="477" y="215">llama.cpp /completion</text>
  <text class="t-sm t-mid" x="477" y="231">llama.cpp with LlamaRoute::Completion</text>
  <path class="arw arw-bnd" d="M138 112 L296 58" marker-end="url(#prov-arrow)"/>
  <path class="arw arw-bnd" d="M138 118 L296 110" marker-end="url(#prov-arrow)"/>
  <path class="arw arw-bnd" d="M138 124 L296 164" marker-end="url(#prov-arrow)"/>
  <path class="arw arw-bnd" d="M138 130 L296 218" marker-end="url(#prov-arrow)"/>
  <path class="arw arw-data" d="M71 152 V194" marker-end="url(#prov-arrow)"/>
</svg>
</div>
<p class="dgm-cap"><b>Only the adapters cross the network.</b> The four HTTP translators sit across the violet boundary. <code>provider-local</code> runs the model in your process and sends nothing.</p>
</div>

## Compare the adapters

- [OpenAI](@/providers/openai.md) (`OpenAi`): feature `provider-openai`, default base URL `https://api.openai.com/v1`. Auth header `Authorization: Bearer`, when a key is set. Rejects `top_k`.
- [Anthropic](@/providers/anthropic.md) (`Anthropic`): feature `provider-anthropic`, default base URL `https://api.anthropic.com/v1`. Auth header `x-api-key`, or `Authorization: Bearer` for a bearer token. Rejects `seed`. `max_tokens` defaults to 1024. Needs a key or token.
- [Google Gemini](@/providers/gemini.md) (`GoogleGemini`): feature `provider-gemini`, default base URL `https://generativelanguage.googleapis.com`. Auth header `x-goog-api-key`, required. Needs a key. No token counts.
- [DeepSeek](@/providers/deepseek.md) (`DeepSeek`): feature `provider-deepseek`, default base URL `https://api.deepseek.com/v1`. Auth header `Authorization: Bearer` (native) or `x-api-key` (bridge). The native route rejects `top_k`. The bridge route rejects `seed` and needs a key.
- [llama.cpp](@/providers/llamacpp.md) (`LlamaCpp`): feature `provider-llamacpp`, default base URL `http://127.0.0.1:8080`. Auth header `Authorization: Bearer`, when a key is set. Rejects nothing. The `Completion` route has no tools and no thinking.
- [vLLM](@/providers/vllm.md) (`Vllm`): feature `provider-vllm`, default base URL `http://127.0.0.1:8000/v1`. Auth header `Authorization: Bearer`, when a key is set. Rejects nothing.
- [LM Studio](@/providers/lmstudio.md) (`LmStudio`): feature `provider-lmstudio`, default base URL `http://127.0.0.1:1234/v1`. Auth header `Authorization: Bearer`, when a key is set. Rejects nothing.
- [Local](@/providers/local/_index.md) (`Local`): feature `provider-local`. No base URL, it runs in your process, and no auth header. Needs a model attached to the client.

These notes apply to every adapter above:

- The default base URL applies when the client's base URL is empty. Set another with `CucaClientBuilder::with_base_url`.
- Each provider page has the same sections: what you need, the endpoint, the request and stream translation, thinking, errors and limits.
- Every HTTP feature enables `reqwest` with its `rustls` backend and `tokio-stream`. `provider-anthropic` also enables `sha2`, `getrandom` and `base64`.
- [The feature matrix](@/reference/features.md) maps every feature to its dependencies.

## Wire formats

A wire format is the JSON shape one provider's HTTP API expects. The seven HTTP adapters speak four of them and share two translation modules. Replies stream back as server-sent events (SSE): the server keeps the connection open and sends one `data:` line per piece of the reply.

| Wire format | Adapters | Stream ends at |
|---|---|---|
| OpenAI-compatible `/chat/completions` | OpenAI, vLLM, LM Studio, DeepSeek native, llama.cpp chat | `data: [DONE]` |
| Anthropic Messages | Anthropic, DeepSeek bridge | `message_stop` event |
| Gemini `streamGenerateContent` | Gemini | the end of the byte stream |
| llama.cpp native `/completion` | llama.cpp with `LlamaRoute::Completion` | the end of the byte stream |

- The [OpenAI](@/providers/openai.md) page documents the OpenAI-compatible translation. The pages for DeepSeek, llama.cpp, vLLM and LM Studio list only their differences.
- The [Anthropic](@/providers/anthropic.md) page documents the Anthropic translation. DeepSeek's bridge reuses it.
- Gemini and the llama.cpp native route each have their own translator.

## Sampling fields

`top_p`, `top_k`, `seed` and `stop` are optional on `UnifiedRequest`. Each adapter sends a set field under the wire name below, or rejects it with `CucaError::Config` before any request goes out. A field left unset is not sent.

| Adapter | `top_p` | `top_k` | `seed` | `stop` |
|---|---|---|---|---|
| OpenAI | `top_p` | rejected | `seed` | `stop` |
| Anthropic | `top_p` | `top_k` | rejected | `stop_sequences` |
| Gemini | `topP` | `topK` | `seed` | `stopSequences` |
| DeepSeek native | `top_p` | rejected | `seed` | `stop` |
| DeepSeek bridge | `top_p` | `top_k` | rejected | `stop_sequences` |
| llama.cpp chat and `/completion` | `top_p` | `top_k` | `seed` | `stop` |
| vLLM | `top_p` | `top_k` | `seed` | `stop` |
| LM Studio | `top_p` | `top_k` | `seed` | `stop` |
| Local | per request override | per request override | per request override | per request override |

Gemini nests all four fields under `generationConfig`, for example `generationConfig.topP`.

- `stop` is sent only when non-empty.
- The rejection messages are ``"`top_k` is not supported by OpenAI"``, ``"`top_k` is not supported by DeepSeek"`` and ``"`seed` is not supported by Anthropic"``. A `top_k` on any other endpoint that reaches the OpenAI-compatible builder reads ``"`top_k` is not supported by this endpoint"``.
- `provider-local` accepts all four and checks `top_p` in `(0, 1]`. See [Requests, thinking and tool calls](@/providers/local/requests.md).

## The in-process adapter

`provider-local` runs a language model inside your process, from a GGUF file or a Hugging Face safetensors checkpoint that CUCA converts once to a cached GGUF. It has no base URL, auth header or route, and it exposes no HTTP endpoint.

- `ProviderEndpoint::Local` selects it, and `CucaClientBuilder::with_local_model` attaches the model. `build` fails when the selected provider is `Local` and no model is attached.
- The CPU is the default device. `provider-local-vulkan`, `provider-local-metal`, `provider-local-cuda` and `provider-local-rocm` add a GPU engine that `LocalModelConfig::device` selects.
- It does not use `tokio-stream`. It uses `reqwest` only for the one-time model download.
- The engine is pure Rust. The `candle` dependency tree builds one C crate, `onig_sys`.

See [Local (in-process)](@/providers/local/_index.md) for a first run, [Choose a model](@/providers/local/models.md) for the supported architectures and presets, and [Configuration](@/providers/local/configuration.md) for every setting.

## Dispatch

There is no provider trait. Each adapter module adds one method to `CucaClient`:

```rust,name=The shape every adapter implements
pub(crate) async fn dispatch_<provider>(
    &self,
    req: UnifiedRequest,
) -> Result<ProviderDispatch, CucaError>
```

`CucaClient::generate_stream` picks the adapter with a `match` on the client's selected provider. Each arm is gated on its feature. When the feature is off, the arm returns `CucaError::ProviderNotEnabled` with the missing flag name.

## Custom endpoints

`ProviderEndpoint` has nine variants: the eight above and `Custom(String)`. `Custom` has no feature and no adapter. Dispatching it returns `CucaError::Config`:

```text,name=CucaError::Config from the Custom arm
configuration error: custom endpoints require a registered adapter
```

- `ProviderEndpoint::default()` is `OpenAi`.
- `UnifiedRequest::new` sets the request's provider to `Custom(String::new())`, and `generate_stream` replaces it with the client's provider before any hook runs.

## Configuration

None of the seven HTTP adapters reads an environment variable. Base URLs, API keys and bearer tokens reach them only through `CucaClientBuilder`. The examples take their base URL and model as flags, and the tests read `cuca-test.kdl`. Both are on [Configuration files](@/reference/environment.md).

`provider-local` reads no environment variable for its settings either. A KDL file fills the same fields as `LocalModelConfig`. See [Configuration files](@/reference/environment.md) and the local [Configuration](@/providers/local/configuration.md) reference.

## Related pages

- [Unified request](@/concepts/unified-request.md) explains the request every adapter translates.
- [Use another OpenAI-compatible server](@/guides/other-openai-server.md) shows how to point an adapter at a server not listed here.
