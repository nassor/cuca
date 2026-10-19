+++
title = "Point the client at another OpenAI-compatible server"
description = "Pick the provider feature that matches the protocol, override the base URL, confirm the model id."
template = "page.html"
weight = 1
+++

# Point the client at another OpenAI-compatible server

<dl class="page-facts">
<dt>In one line</dt>
<dd>Enable the provider feature whose protocol your server speaks, then set the base URL on the builder</dd>
<dt>You need</dt>
<dd>A running server that mounts <code>/chat/completions</code>, and its base URL</dd>
<dt>Read this if</dt>
<dd>Your backend is a self-hosted vLLM or LM Studio, a corporate gateway, or any other server that speaks the OpenAI chat protocol</dd>
</dl>

Goal: stream one reply from a server that is not the bundled llama.cpp default, with no change to your request or consuming code.

Five of the seven HTTP adapters share one wire protocol, so the switch is a feature flag and a base URL. Anthropic and Gemini speak their own protocols; see [Providers](@/providers/_index.md).

## Step 1: choose the feature

Pick by the protocol and default URL your server matches. All five features post the same request body to `{base_url}/chat/completions`.

| Your server | Feature | Default base URL |
|---|---|---|
| A hosted OpenAI-compatible API | `provider-openai` | `https://api.openai.com/v1` |
| DeepSeek native API | `provider-deepseek` | `https://api.deepseek.com/v1` |
| vLLM | `provider-vllm` | `http://127.0.0.1:8000/v1` |
| LM Studio | `provider-lmstudio` | `http://127.0.0.1:1234/v1` |
| `llama-server` from llama.cpp | `provider-llamacpp` | `http://127.0.0.1:8080` |

For any other server or gateway, use the feature with the closest default. You override the base URL in step 2 anyway.

Add the crate with the feature. The crate has no default provider, so the feature is required.

Linux and macOS:

```bash
cargo add cuca --path /path/to/cuca --features provider-vllm
```

Windows (PowerShell):

```powershell
cargo add cuca --path C:\path\to\cuca --features provider-vllm
```

Result: `cargo add` prints `Adding cuca (local) to dependencies`, then the feature list with `+ provider-vllm` enabled. With the feature off, `generate_stream` fails with `provider feature not enabled: provider-vllm`.

## Step 2: set the base URL and find the model id

For `provider-openai`, `provider-deepseek`, `provider-vllm` and `provider-lmstudio`, the adapter builds the route by appending to the base URL:

```text,name=How the route is assembled
{base_url with trailing slash trimmed}/chat/completions
```

The base URL must already include every path segment before `/chat/completions`. Most servers mount the route under `/v1`, so the base URL ends in `/v1`.

`provider-llamacpp` is the exception. On its chat route it appends `/v1` when the base URL does not already end in it, so `http://127.0.0.1:1234` and `http://127.0.0.1:1234/v1` both work there.

A DeepSeek base URL that contains `api.deepseek.com/anthropic` or ends in `/anthropic` selects the Anthropic bridge instead, and that route requires an API key.

Ask the server which model ids it serves.

Linux and macOS:

```bash
curl http://10.0.0.7:8000/v1/models
```

Windows (PowerShell), where `curl` is an alias for `Invoke-WebRequest`:

```powershell
curl.exe http://10.0.0.7:8000/v1/models
```

Result: a JSON list. Each `id` value is a model id you can put in `UnifiedRequest::new`.

## Step 3: build the client

```rust,name=The two builder calls that matter
use cuca::CucaClient;
use cuca::types::ProviderEndpoint;

let client = CucaClient::builder()
    .with_provider(ProviderEndpoint::Vllm)
    .with_base_url("http://10.0.0.7:8000/v1")
    .build()?;
```

If the server wants a key, add `.with_api_key("...")`. It becomes an `Authorization: Bearer` header. With no key the header is not sent, so an unauthenticated local server needs no placeholder.

`with_provider` is required. Without it `build()` fails:

```text,name=CucaError::Config when with_provider was never called
configuration error: no provider selected; call with_provider before build
```

Result: `build()` returns `Ok(client)`.

## Step 4: verify with a bundled example

The examples read `CUCA_BASE_URL` and `CUCA_MODEL`, so they check any OpenAI-compatible endpoint. They always use the llama.cpp chat route, which appends `/v1` to a base URL that does not end in it. A server that mounts `/chat/completions` somewhere other than `/v1` cannot be checked this way; run your own client from step 3 instead.

Linux and macOS:

```bash
CUCA_BASE_URL=http://10.0.0.7:8000/v1 CUCA_MODEL=your-model-id \
  cargo run --example llamacpp_gemma --features provider-llamacpp
```

Windows (PowerShell), where each assignment is a statement and `;` joins them:

```powershell
$env:CUCA_BASE_URL = "http://10.0.0.7:8000/v1"; $env:CUCA_MODEL = "your-model-id"; cargo run --example llamacpp_gemma --features provider-llamacpp
```

The `$env:` assignments last for the PowerShell session. Clear them with `Remove-Item Env:CUCA_BASE_URL, Env:CUCA_MODEL`.

Result: the reply text prints on stdout. That means the endpoint, the model id and the protocol agree. A reasoning model prints nothing while it thinks, because the example prints `Text` blocks only.

If `generate_stream` fails, the example prints `No server answered at <base URL>: <error>` and exits with code 0. The error text distinguishes the causes:

- `HTTP error <status>: <body>` means a server answered and refused; the body carries its reason.
- `transport failure: ...` means nothing answered at that address.

The two variables affect the examples and the test harness only. A client you write passes its base URL to the builder; the variables are listed in [Environment variables](@/reference/environment.md).

## If you used ProviderEndpoint::Custom

`Custom` takes a `String`, and dispatching it always fails:

```text,name=CucaError::Config from the Custom arm
configuration error: custom endpoints require a registered adapter
```

Select the feature whose protocol your gateway speaks and set the base URL, as in steps 1 to 3.

Next page: [Write a custom plugin](@/guides/custom-plugin.md).
