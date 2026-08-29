+++
title = "llama.cpp"
description = "The llama.cpp adapter's two server routes: the OpenAI-compatible chat route and the native completion route."
template = "page.html"
weight = 5
+++

# llama.cpp

<dl class="page-facts">
<dt>In one line</dt>
<dd>Dispatches unified requests to a llama.cpp server (llama-server) over either its OpenAI-compatible chat route or its native completion route.</dd>
<dt>You need</dt>
<dd>The <code>provider-llamacpp</code> feature and a running llama-server instance. No API key is required.</dd>
<dt>Read this if</dt>
<dd>You are routing requests through <code>ProviderEndpoint::LlamaCpp</code>, or choosing between the chat and completion routes.</dd>
</dl>

## Endpoint

| Fact | Value |
|---|---|
| Feature flag | `provider-llamacpp` |
| `ProviderEndpoint` variant | `LlamaCpp` |
| Default base URL | `http://127.0.0.1:8080`, used when both `LlamaCppConfig::base_url` and the client's base URL are empty |

The default carries no `/v1` suffix.

## `LlamaCppConfig`

| Field | Meaning |
|---|---|
| `base_url` | Overrides the client base URL for this dispatch; empty defers to it |
| `api_key` | Sent as `Authorization: Bearer` when set; falls back to the client's API key, then to no header |
| `model` | Overrides the request's model in the outgoing body when non-empty |
| `n_threads` | `n_threads` body parameter; omitted when unset |
| `n_gpu_layers` | `n_gpu_layers` body parameter; omitted when unset |
| `flash_attn` | `flash_attn` body parameter |
| `route` | `LlamaRoute::Chat` or `LlamaRoute::Completion`; default `Chat` |

## `LlamaRoute::Chat`

Reuses the [OpenAI](@/providers/openai.md)-compatible adapter's request body and SSE translation. The dispatch appends `/v1` to the resolved base URL when it is not already present, then posts `POST {base}/v1/chat/completions`. Thinking maps through the same `reasoning_effort` translation as OpenAI.

## `LlamaRoute::Completion`

The native route, `POST {base}/completion`, with no `/v1` suffix. It has no tool-call protocol and no thinking knob; `req.thinking` is silently ignored.

- `stream` is always `true`.
- `n_predict` carries `max_tokens`, or 128 when the request leaves it unset.
- `n_threads`, `n_gpu_layers`, and `flash_attn` appear in the body only when configured on `LlamaCppConfig`.
- The prompt is assembled from the conversation: `System` message text becomes bare leading context; `User` and `Assistant` messages are marked with `### User:` and `### Assistant:` prefixes. Only `Text` blocks contribute; images, reasoning, and tool blocks are dropped, and `Tool`-role messages are skipped entirely.
- Response frames are one token per frame, `{"content": "tok", "stop": false}`, terminated by `{"content": "", "stop": true}`. An `error` field on a frame surfaces as `CucaError::Provider`.

## See also

[vLLM](@/providers/vllm.md) and [LM Studio](@/providers/lmstudio.md) also speak the OpenAI-compatible chat route.
