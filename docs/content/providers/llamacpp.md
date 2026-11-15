+++
title = "llama.cpp"
description = "The llama.cpp adapter: a first request, the OpenAI-compatible chat route, the native completion route, and the LlamaCppConfig that selects between them."
template = "page.html"
weight = 5
+++

# llama.cpp

<dl class="page-facts">
<dt>In one line</dt>
<dd>Sends unified requests to a <code>llama-server</code> over its OpenAI-compatible chat route or its native completion route.</dd>
<dt>You need</dt>
<dd>The <code>provider-llamacpp</code> feature and a running <code>llama-server</code>. No API key is required.</dd>
<dt>Read this if</dt>
<dd>You route requests through <code>ProviderEndpoint::LlamaCpp</code>, or choose between the chat and completion routes.</dd>
</dl>

## Send a first request

Start a `llama-server` first. The example uses its chat route on port 1234. Change `base_url` and `model` to match your server.

```bash,name=Runs the same on all three platforms
cargo add cuca --path /path/to/cuca --features provider-llamacpp
cargo add tokio --features rt,macros
cargo add tokio-stream
```

```rust,name=A first stream through the llama.cpp adapter
use std::io::{Write, stdout};

use cuca::types::{MessageContentBlock, ProviderEndpoint};
use cuca::{CucaClient, UnifiedRequest};
use tokio_stream::StreamExt;

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let base_url = "http://127.0.0.1:1234/v1";
    let model = "google/gemma-4-e4b";

    let client = CucaClient::builder()
        .with_provider(ProviderEndpoint::LlamaCpp)
        .with_base_url(base_url)
        .build()?;

    let request = UnifiedRequest::new(model)
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

```text,name=Expected shape; the wording and the block counts vary by model
Hello.
blocks: 2 text, 18 thinking
```

A model that reasons first streams thinking blocks, then text blocks.

## What you need

| Fact | Chat route (default) | Completion route |
|---|---|---|
| Feature flag | `provider-llamacpp` | `provider-llamacpp` |
| `ProviderEndpoint` variant | `LlamaCpp` | `LlamaCpp` |
| Default base URL | `http://127.0.0.1:8080` | `http://127.0.0.1:8080` |
| Route | `POST {base}/v1/chat/completions` | `POST {base}/completion` |
| Wire protocol | OpenAI-compatible | native, one token per frame |
| Auth | `Authorization: Bearer`, only when a key is set | same |

- The base URL is the first non-empty of `LlamaCppConfig::base_url`, the client's base URL, then the default.
- The chat route trims a trailing `/` and appends `/v1`, unless the base URL already ends in `/v1`.
- The completion route trims a trailing `/` and appends `/completion`, with no `/v1` handling.

## How it differs from OpenAI

The default route, `Chat`, runs the code on the [OpenAI](@/providers/openai.md) page. The differences:

- The default base URL is `http://127.0.0.1:8080`, and the adapter adds `/v1` for the chat route.
- No API key is needed. A key is sent as a bearer token only when one is set. `LlamaCppConfig::api_key` wins over the client's API key.
- `LlamaCppConfig` selects the route and carries `llama-server` body parameters.
- The `Completion` route is a separate protocol with its own body, prompt format and frames.

<div class="dgm">
<div class="dgm-scroll">
<svg viewBox="0 0 660 170" role="img" aria-labelledby="llc-title llc-desc">
  <title id="llc-title">The two llama.cpp routes</title>
  <desc id="llc-desc">LlamaCppConfig::route selects the route. Chat sends an OpenAI-compatible request to /v1/chat/completions with tools, thinking and token usage. Completion sends a single prompt string to /completion and reads text only.</desc>
  <defs>
    <marker id="llc-arrow" viewBox="0 0 8 8" refX="7" refY="4"
            markerWidth="6" markerHeight="6" orient="auto">
      <path d="M0 0.6 L7 4 L0 7.4 z" fill="var(--muted-foreground)"/>
    </marker>
  </defs>
  <text class="t-title" x="6" y="20">LlamaCppConfig::route</text>
  <rect class="blk blk-ctl" x="6" y="66" width="130" height="50" rx="6"/>
  <text class="t-lbl t-mid t-ctl" x="71" y="88">route</text>
  <text class="t-sm t-mid" x="71" y="104">Chat or Completion</text>
  <rect class="blk blk-bnd" x="250" y="38" width="404" height="52" rx="6"/>
  <text class="t-lbl t-mid t-bnd" x="452" y="59">Chat (default): /v1/chat/completions</text>
  <text class="t-sm t-mid" x="452" y="76">OpenAI translation. Tools, thinking, token usage</text>
  <rect class="blk blk-bnd" x="250" y="108" width="404" height="52" rx="6"/>
  <text class="t-lbl t-mid t-bnd" x="452" y="129">Completion: /completion</text>
  <text class="t-sm t-mid" x="452" y="146">One prompt string. Text only. No tools, no usage</text>
  <path class="arw arw-bnd" d="M138 84 L246 64" marker-end="url(#llc-arrow)"/>
  <path class="arw arw-bnd" d="M138 98 L246 134" marker-end="url(#llc-arrow)"/>
</svg>
</div>
<p class="dgm-cap"><b>Chat is the full route.</b> Use <code>Completion</code> only when you want the server's native prompt handling and need no tools, thinking or token counts.</p>
</div>

## Configuration

`CucaClientBuilder::with_llamacpp_config` sets a `LlamaCppConfig`. With none set, the defaults below apply.

| Field | Type | Default | Used by |
|---|---|---|---|
| `base_url` | `String` | empty, defers to the client | both routes |
| `api_key` | `Option<String>` | `None`, defers to the client | both routes |
| `model` | `String` | empty, defers to the request's model | both routes |
| `route` | `LlamaRoute` | `Chat` | selects the route |
| `n_threads` | `Option<u32>` | `None` | `Completion` only |
| `n_gpu_layers` | `Option<u32>` | `None` | `Completion` only |
| `flash_attn` | `bool` | `false` | `Completion` only |

`LlamaCppConfig` and `LlamaRoute` are defined in the crate-private `provider` module, and the crate root does not re-export them.

## Request translation

Chat route: the body is the one under [OpenAI request translation](@/providers/openai.md#request-translation). The model is `LlamaCppConfig::model` when non-empty, otherwise the request's model. The three `llama-server` parameters (`n_threads`, `n_gpu_layers`, `flash_attn`) are not sent. `top_p`, `top_k`, `seed` and `stop` go out under those names, and no sampling field is rejected.

Completion route: the body is a JSON object with these keys:

- `prompt` is built from the conversation. `System` message text comes first as bare context. `User` and `Assistant` messages get `### User:` and `### Assistant:` prefixes.
- Only `Text` blocks count. Images, thinking and tool blocks are dropped, and `Tool`-role messages are skipped.
- `stream` is always `true`.
- `temperature`, `top_p`, `top_k` and `seed` are sent only when set. `stop` is sent only when non-empty. No sampling field is rejected.
- `n_predict` carries `max_tokens`, or 128 when the request leaves it unset.
- `n_threads` and `n_gpu_layers` are sent only when configured, and `flash_attn` only when `true`.
- `model` is `LlamaCppConfig::model`, or the request's model when that is empty. `llama-server` ignores it.

The completion route has no tool protocol, so `UnifiedRequest::tools` is not sent.

## Response stream

- Chat route: the stream is the one under [OpenAI response stream](@/providers/openai.md#response-stream), ending at `data: [DONE]`.
- Completion route: each frame is `{"content": "tok", "stop": false}`. A non-empty `content` becomes `MessageContentBlock::Text`. A frame with empty or missing `content`, including the final `{"content": "", "stop": true}`, produces no block.
- The completion route emits only `Text` blocks. The stream ends when the connection ends.

## Token usage

- Chat route: the rules on the [OpenAI](@/providers/openai.md#token-usage) page apply, because the request sets `stream_options.include_usage` and `llama-server` honours it.
- Completion route: no token counts are read. `UnifiedResponse::usage` is `None`, `prompt_tokens` is `0`, and `completion_tokens` counts one per `Text` block.

## Thinking

- Chat route: `req.thinking` maps to `reasoning_effort` as on the [OpenAI](@/providers/openai.md#thinking) page, and `reasoning_content` deltas become `Thinking` blocks.
- Completion route: `req.thinking` is ignored, and no `Thinking` block is produced.

## Errors

| Condition | Error |
|---|---|
| Feature not compiled | `CucaError::ProviderNotEnabled("provider-llamacpp")` |
| Non-2xx response | `CucaError::Http { status, body }` with the captured body |
| Connection or body-read failure | `CucaError::Transport { message }` |
| Frame is not valid JSON | `CucaError::Json` |
| Chat frame carries `{"error": {"message": ...}}` | `CucaError::Provider` with `provider: OpenAi` |
| Completion frame has an `error` field, as a string or `{"message": ...}` | `CucaError::Provider` with `provider: LlamaCpp` |
| Frame bytes are not valid UTF-8 | `CucaError::SseParse` |

A malformed frame or an error frame ends the stream after the error.

## Limits

- No cap on request or response size. The client sets no request timeout.
- The SSE parser keeps one reusable buffer that starts at 8192 bytes. It grows to the largest single frame plus one partial line.
- Completion route: `n_predict` defaults to 128 when the request leaves `max_tokens` unset. Chat route: `max_tokens` is sent only when the request sets it.
- Chat route: pending tool calls hold their argument fragments until each call completes.

## Related pages

- [vLLM](@/providers/vllm.md) and [LM Studio](@/providers/lmstudio.md) also speak the OpenAI-compatible chat route.
