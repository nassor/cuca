+++
title = "Point the client at another OpenAI-compatible server"
description = "Pick the provider feature that matches the protocol, set the base URL, and confirm the model id."
template = "page.html"
weight = 1
+++

# Point the client at another OpenAI-compatible server

<dl class="page-facts">
<dt>In one line</dt>
<dd>Enable the provider feature whose protocol your server speaks, then set the base URL on the builder</dd>
<dt>You need</dt>
<dd>A running server that has a <code>/chat/completions</code> route, and its base URL (the address up to, not including, <code>/chat/completions</code>)</dd>
<dt>Read this if</dt>
<dd>Your backend is a self-hosted vLLM or LM Studio, a company gateway, or any other server that speaks the OpenAI chat protocol</dd>
</dl>

Goal: stream one reply from a server that is not the default llama.cpp one, without changing your request or the code that reads the reply.

Five of the seven HTTP adapters share one wire protocol, so the switch is a Cargo feature and a base URL. Anthropic and Gemini speak their own protocols. See [Providers](@/providers/_index.md).

## Step 1: choose the feature

Pick the row that matches your server. All five features send the same request body to `{base_url}/chat/completions`.

| Your server | Feature | Default base URL |
|---|---|---|
| A hosted OpenAI-compatible API | `provider-openai` | `https://api.openai.com/v1` |
| DeepSeek native API | `provider-deepseek` | `https://api.deepseek.com/v1` |
| vLLM | `provider-vllm` | `http://127.0.0.1:8000/v1` |
| LM Studio | `provider-lmstudio` | `http://127.0.0.1:1234/v1` |
| `llama-server` from llama.cpp | `provider-llamacpp` | `http://127.0.0.1:8080` |

For any other server or gateway, use the feature with the closest default. Step 2 overrides the base URL anyway.

Add the crate with the feature. The crate has no default provider, so the feature is required.

```bash,name=Linux and macOS
cargo add cuca --path /path/to/cuca --features provider-vllm
```

```powershell,name=Windows (PowerShell)
cargo add cuca --path C:\path\to\cuca --features provider-vllm
```

Result: `cargo add` prints `Adding cuca (local) to dependencies`, then the feature list with `+ provider-vllm` enabled. With the feature off, `generate_stream` fails with `provider feature not enabled: provider-vllm`.

## Step 2: set the base URL and find the model id

For `provider-openai`, `provider-deepseek`, `provider-vllm` and `provider-lmstudio`, the adapter builds the route like this:

```text,name=How the route is assembled
{base_url with trailing slash trimmed}/chat/completions
```

The base URL must already include every path segment before `/chat/completions`. Most servers mount the route under `/v1`, so the base URL ends in `/v1`.

`provider-llamacpp` is the exception. It adds `/v1` when the base URL does not already end in it, so `http://127.0.0.1:1234` and `http://127.0.0.1:1234/v1` both work there.

A DeepSeek base URL that contains `api.deepseek.com/anthropic`, or ends in `/anthropic`, selects the Anthropic bridge instead. That route requires an API key.

Ask the server which model ids it serves:

```bash,name=Linux and macOS
curl http://10.0.0.7:8000/v1/models
```

```powershell,name=Windows (PowerShell)
curl.exe http://10.0.0.7:8000/v1/models
```

Result: a JSON list. Each `id` value is a model id you can pass to `UnifiedRequest::new`.

## Step 3: build the client

```rust,name=The two builder calls that matter
use cuca::CucaClient;
use cuca::types::ProviderEndpoint;

let client = CucaClient::builder()
    .with_provider(ProviderEndpoint::Vllm)
    .with_base_url("http://10.0.0.7:8000/v1")
    .build()?;
```

If the server wants a key, add `.with_api_key("...")`. It is sent as an `Authorization: Bearer` header. With no key the header is not sent, so a local server without authentication needs no placeholder.

`with_provider` is required. Without it, `build()` fails:

```text,name=CucaError::Config when with_provider was never called
configuration error: no provider selected; call with_provider before build
```

Result: `build()` returns `Ok(client)`.

`top_k` is sent by `provider-vllm`, `provider-lmstudio` and `provider-llamacpp`. With `provider-openai` or the DeepSeek native route it fails with `CucaError::Config`. A gateway that accepts `top_k` therefore needs one of the other three features. `top_p`, `seed` and `stop` go out on all five.

## Step 4: check with a bundled example

The examples take `--base-url` and `--model`, so they can check any OpenAI-compatible endpoint. They always use the llama.cpp chat route, which adds `/v1` to a base URL that does not end in it. A server that mounts `/chat/completions` somewhere other than `/v1` cannot be checked this way. Run your own client from step 3 instead.

```bash,name=Runs the same on all three platforms
cargo run --example llamacpp_gemma --features provider-llamacpp -- --base-url http://10.0.0.7:8000/v1 --model your-model-id
```

Result: the reply text prints. That means the address, the model id and the protocol agree. A reasoning model prints nothing while it thinks, because the example prints `Text` blocks only.

If the call fails, the example prints `No server answered at <base URL>: <error>` and exits with code 0. The error text tells you the cause:

- `HTTP error <status>: <body>`: a server answered and refused. The body has its reason.
- `transport failure: ...`: nothing answered at that address.

The two flags change the examples only. A client you write passes its base URL to the builder. See [Configuration files](@/reference/environment.md).

## If you used ProviderEndpoint::Custom

`Custom` takes a `String`, and using it always fails:

```text,name=CucaError::Config from the Custom arm
configuration error: custom endpoints require a registered adapter
```

Select the feature whose protocol your gateway speaks and set the base URL, as in steps 1 to 3.

Next page: [Write a custom plugin](@/guides/custom-plugin.md).
