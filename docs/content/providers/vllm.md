+++
title = "vLLM"
description = "The vLLM adapter: local default endpoint, optional bearer auth, and the shared OpenAI-compatible request and stream translation."
template = "page.html"
weight = 6
+++

# vLLM

<dl class="page-facts">
<dt>In one line</dt>
<dd>Dispatches unified requests to a vLLM server's OpenAI-compatible chat completions route.</dd>
<dt>You need</dt>
<dd>The <code>provider-vllm</code> feature and a running vLLM server. An API key is optional.</dd>
<dt>Read this if</dt>
<dd>You are routing requests through <code>ProviderEndpoint::Vllm</code>.</dd>
</dl>

The smallest streaming turn: the `Vllm` variant against the default local
endpoint; no API key. The model id is the one the server reports under
`GET {base}/models`.

```rust,name=A first stream through the vLLM adapter
use cuca::types::{MessageContentBlock, ProviderEndpoint};
use cuca::{CucaClient, UnifiedRequest};
use tokio_stream::StreamExt;

let client = CucaClient::builder()
    .with_provider(ProviderEndpoint::Vllm)
    .build()?;

let mut stream = client
    .generate_stream(UnifiedRequest::new("google/gemma-4-e4b").add_user_message("Say hello."))
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
| Feature flag | `provider-vllm` |
| `ProviderEndpoint` variant | `Vllm` |
| Default base URL | `http://127.0.0.1:8000/v1`, used when the client's base URL is empty |
| Route | `POST {base_url}/chat/completions` |

## Authentication

`Authorization: Bearer <api_key>` when an API key is configured; vLLM treats it as an optional header, not a required credential.

## Shared adapter

This provider shares the [OpenAI](@/providers/openai.md)-compatible request builder and SSE translator in full. `reasoning_content` in the response flows into `Thinking` blocks through the same translation. See the OpenAI page for the request body shape, the tool-call accumulation rules, and the thinking effort mapping.

## Base URL deviation

The adapter's default, `http://127.0.0.1:8000/v1`, carries the `/v1` suffix the shared adapter appends `/chat/completions` to. The design specification's provider table instead lists the bare host `http://127.0.0.1:8000` without `/v1`, unlike its OpenAI and DeepSeek rows, which do include `/v1`. The code default here includes `/v1`.
