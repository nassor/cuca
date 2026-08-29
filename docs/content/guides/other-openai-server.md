+++
title = "Point the client at another OpenAI-compatible server"
description = "Pick the provider feature that matches the protocol, override the base URL, confirm the model id."
template = "page.html"
weight = 1
+++

# Point the client at another OpenAI-compatible server

<dl class="page-facts">
<dt>In one line</dt>
<dd>Choose the provider feature by protocol, not by vendor name, then override the base URL on the builder</dd>
<dt>You need</dt>
<dd>A running server that mounts <code>/chat/completions</code>, and its base URL</dd>
<dt>Read this if</dt>
<dd>Your backend is Ollama, a self-hosted vLLM, LM Studio, a corporate gateway, or anything else that speaks the OpenAI protocol</dd>
</dl>

Five of the seven adapters share one wire protocol, so switching between servers
that speak it is a feature flag and a base URL. Nothing about the request,
the stream, or your consuming code changes.

## Step 1: decide which feature your target speaks

The choice is about protocol and defaults, not branding. All five options below
post the same body to the same route.

| Your server | Feature | Why this one |
|---|---|---|
| A hosted OpenAI-compatible API | `provider-openai` | its default base URL is the public API, and no local port is assumed |
| vLLM | `provider-vllm` | default `http://127.0.0.1:8000/v1`, and the key is treated as optional |
| LM Studio | `provider-lmstudio` | default `http://127.0.0.1:1234/v1`, key optional |
| `llama-server` from llama.cpp | `provider-llamacpp` | default `http://127.0.0.1:8080`, and the chat route appends `/v1` for you |
| Anything else OpenAI-compatible, including Ollama and gateways | any of the above | pick by which default is closest; you are overriding the base URL regardless |

If your backend does not speak the OpenAI protocol, this guide is the wrong one.
Anthropic and Gemini have their own adapters, listed in
[Providers](@/providers/_index.md).

## Step 2: find out whether your base URL needs `/v1`

The shared adapter builds the request URL by appending to whatever you give it:

```text,name=How the route is assembled
{base_url with trailing slash trimmed}/chat/completions
```

So the base URL must already include every path segment before
`/chat/completions`. Most OpenAI-compatible servers mount it under `/v1`, which
means the base URL ends in `/v1`.

`provider-llamacpp` is the one exception: on its chat route it appends `/v1`
itself when the resolved base URL does not already end in it. Passing
`http://127.0.0.1:1234` and `http://127.0.0.1:1234/v1` both work there, and only
there.

## Step 3: build the client

```rust,name=The two calls that matter
use cuca::{CucaClient, ProviderEndpoint};

let client = CucaClient::builder()
    .with_provider(ProviderEndpoint::Vllm)
    .with_base_url("http://10.0.0.7:8000/v1")
    .build()?;
```

If your server wants a key, add `.with_api_key("...")`, which becomes an
`Authorization: Bearer` header. If it does not, leave it out: the header is only
attached when a key is configured, so an unauthenticated local server needs no
placeholder value.

`with_provider` is not optional. `build()` fails without it:

```text,name=CucaError::Config when with_provider was never called
configuration error: no provider selected; call with_provider before build
```

## Step 4: confirm the model id

The request carries a model id, and the server decides whether it recognises it.
Ask the server rather than guessing.

Linux and macOS:

```bash
curl http://10.0.0.7:8000/v1/models
```

Windows (PowerShell), where `curl` is an alias for `Invoke-WebRequest`:

```powershell
curl.exe http://10.0.0.7:8000/v1/models
```

Use one of the `id` values from the response in `UnifiedRequest::new`.

## Step 5: verify before wiring it into your code

The bundled examples read `CUCA_BASE_URL` and `CUCA_MODEL`, so they double as a
connectivity check for any OpenAI-compatible endpoint.

Linux and macOS:

```bash
CUCA_BASE_URL=http://10.0.0.7:8000/v1 CUCA_MODEL=your-model-id \
  cargo run --example llamacpp_gemma --features provider-llamacpp
```

Windows (PowerShell), where assignments are separate statements joined by `;`:

```powershell
$env:CUCA_BASE_URL = "http://10.0.0.7:8000/v1"; $env:CUCA_MODEL = "your-model-id"; cargo run --example llamacpp_gemma --features provider-llamacpp
```

Text on stdout means the endpoint, the model id and the protocol all agree. A
`CucaError::Http` with a status and a body means you reached a server and it
refused; the body carries its reason. A `CucaError::Transport` means nothing
answered.

Those two variables affect the examples and the test harness only. No code under
`src/` reads any environment variable, so your own client passes its base URL to
the builder.

## If you were reaching for `ProviderEndpoint::Custom`

It exists, it takes a `String`, and dispatching it always fails:

```text,name=CucaError::Config from the Custom arm
configuration error: custom endpoints require a registered adapter
```

`Custom` is a label for a gateway, not a protocol. Selecting the feature whose
protocol your gateway actually speaks and overriding the base URL is the working
route, which is steps 1 through 3 above.

Next page: [Write a custom plugin](@/guides/custom-plugin.md).
