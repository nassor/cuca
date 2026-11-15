+++
title = "The unified request and stream"
description = "Why one request type and five block types are enough for every backend, and where each provider's differences live."
template = "page.html"
weight = 1
+++

# The unified request and stream

<dl class="page-facts">
<dt>In one line</dt>
<dd>The unified types are the smallest shape every backend can agree on, and every difference between backends lives in the provider adapter</dd>
<dt>You need</dt>
<dd>Nothing running</dd>
<dt>Read this if</dt>
<dd>You wonder why the request has the fields it has, or where a provider quirk belongs</dd>
</dl>

Every LLM service has its own HTTP API. They differ in the address you call, the shape of the request body, the way you sign in, the format of the streamed answer and how tool calls work. Moving from one service to another would normally mean rewriting all of it.

CUCA hides this behind two types. You build a `UnifiedRequest`. You read a stream of `MessageContentBlock` values. Switching providers is a feature flag and a base URL, which is the address of the server.

## Four wire shapes

A wire shape is the format a provider speaks over HTTP. The seven HTTP providers use four shapes. The eighth provider, `provider-local`, runs the model inside your process and has no wire.

<div class="dgm">
<div class="dgm-scroll">
<svg viewBox="0 0 660 262" role="img" aria-labelledby="wire-title wire-desc">
  <title id="wire-title">One request type, four wire shapes</title>
  <desc id="wire-desc">A UnifiedRequest goes to the provider adapter, which writes one of four wire shapes. The chat completions route serves OpenAI, vLLM, LM Studio, DeepSeek native and llama.cpp chat. The messages route serves Anthropic and DeepSeek bridge. The streamGenerateContent route serves Gemini. The completion route serves llama.cpp native.</desc>
  <defs>
    <marker id="wire-arrow" viewBox="0 0 8 8" refX="7" refY="4"
            markerWidth="6" markerHeight="6" orient="auto">
      <path d="M0 0.6 L7 4 L0 7.4 z" fill="var(--muted-foreground)"/>
    </marker>
  </defs>
  <text class="t-title" x="6" y="20">The provider adapter picks the wire shape</text>
  <rect class="blk" x="6" y="102" width="120" height="80" rx="6"/>
  <text class="t-sm t-mid" x="66" y="138">Unified</text>
  <text class="t-sm t-mid" x="66" y="152">Request</text>
  <rect class="blk blk-bnd" x="180" y="36" width="150" height="44" rx="6"/>
  <text class="t-sm t-mid t-bnd" x="255" y="62">/chat/completions</text>
  <rect class="blk blk-bnd" x="180" y="92" width="150" height="44" rx="6"/>
  <text class="t-sm t-mid t-bnd" x="255" y="118">/messages</text>
  <rect class="blk blk-bnd" x="180" y="148" width="150" height="44" rx="6"/>
  <text class="t-sm t-mid t-bnd" x="255" y="174">streamGenerateContent</text>
  <rect class="blk blk-bnd" x="180" y="204" width="150" height="44" rx="6"/>
  <text class="t-sm t-mid t-bnd" x="255" y="230">/completion</text>
  <path class="arw" d="M128 142 C154 142 152 58 176 58" marker-end="url(#wire-arrow)"/>
  <path class="arw" d="M128 142 C154 142 152 114 176 114" marker-end="url(#wire-arrow)"/>
  <path class="arw" d="M128 142 C154 142 152 170 176 170" marker-end="url(#wire-arrow)"/>
  <path class="arw" d="M128 142 C154 142 152 226 176 226" marker-end="url(#wire-arrow)"/>
  <text class="t-sm" x="354" y="55">OpenAI, vLLM, LM Studio,</text>
  <text class="t-sm" x="354" y="69">DeepSeek native, llama.cpp chat</text>
  <text class="t-sm" x="354" y="111">Anthropic, DeepSeek bridge</text>
  <text class="t-sm" x="354" y="167">Gemini</text>
  <text class="t-sm" x="354" y="223">llama.cpp native</text>
</svg>
</div>
<p class="dgm-cap"><b>Your code builds one request.</b> The adapter for the chosen provider turns it into one of these four formats, so your code never sees them.</p>
</div>

## One turn

A turn is one request and its whole answer. It passes six stages.

<div class="dgm">
<div class="dgm-scroll">
<svg viewBox="0 0 660 124" role="img" aria-labelledby="pipe-title pipe-desc">
  <title id="pipe-title">The six stages of one CUCA turn</title>
  <desc id="pipe-desc">Six stages, left to right. A UnifiedRequest passes through the on_request hook of every registered plugin. The selected provider adapter builds the vendor body and posts it over HTTP. The reply bytes feed SseStreamParser, which emits protocol level SSE events. The same provider adapter translates those events into MessageContentBlock values. Each block then passes through every plugin's on_stream_chunk hook before reaching the caller.</desc>
  <defs>
    <marker id="pipe-arrow" viewBox="0 0 8 8" refX="7" refY="4"
            markerWidth="6" markerHeight="6" orient="auto">
      <path d="M0 0.6 L7 4 L0 7.4 z" fill="var(--muted-foreground)"/>
    </marker>
  </defs>
  <text class="t-title" x="6" y="20">One turn through CucaClient::generate_stream</text>
  <rect class="blk" x="6" y="44" width="88" height="50" rx="6"/>
  <text class="t-sm t-mid" x="50" y="67">Unified</text>
  <text class="t-sm t-mid" x="50" y="81">Request</text>
  <rect class="blk blk-ctl" x="118" y="44" width="88" height="50" rx="6"/>
  <text class="t-sm t-mid t-ctl" x="162" y="67">on_request</text>
  <text class="t-sm t-mid" x="162" y="81">hooks</text>
  <rect class="blk blk-bnd" x="230" y="44" width="88" height="50" rx="6"/>
  <text class="t-sm t-mid t-bnd" x="274" y="67">provider</text>
  <text class="t-sm t-mid" x="274" y="81">adapter</text>
  <rect class="blk blk-data" x="342" y="44" width="88" height="50" rx="6"/>
  <text class="t-sm t-mid t-data" x="386" y="67">SseStream</text>
  <text class="t-sm t-mid" x="386" y="81">Parser</text>
  <rect class="blk blk-bnd" x="454" y="44" width="88" height="50" rx="6"/>
  <text class="t-sm t-mid t-bnd" x="498" y="67">adapter</text>
  <text class="t-sm t-mid" x="498" y="81">translate</text>
  <rect class="blk blk-ctl" x="566" y="44" width="88" height="50" rx="6"/>
  <text class="t-sm t-mid t-ctl" x="610" y="67">on_stream</text>
  <text class="t-sm t-mid" x="610" y="81">_chunk</text>
  <path class="arw" d="M96 69 H114" marker-end="url(#pipe-arrow)"/>
  <path class="arw arw-bnd" d="M208 69 H226" marker-end="url(#pipe-arrow)"/>
  <path class="arw arw-data" d="M320 69 H338" marker-end="url(#pipe-arrow)"/>
  <path class="arw arw-data" d="M432 69 H450" marker-end="url(#pipe-arrow)"/>
  <path class="arw arw-ctl" d="M544 69 H562" marker-end="url(#pipe-arrow)"/>
  <text class="t-ax" x="6" y="114">stages 3 and 5 are the provider adapter; the wire format never leaves it</text>
</svg>
</div>
<p class="dgm-cap"><b>Stages 3 and 5 are the only places vendor JSON exists.</b> Stage 4 reads the stream format and knows no vendor. Stages 2 and 6 see only the unified types.</p>
</div>

1. You pass a `UnifiedRequest` to `generate_stream`.
2. Each plugin's `on_request` hook may change it. See [Plugins and services](@/concepts/plugin-layering.md).
3. The adapter builds the vendor's request body and sends it.
4. The reply arrives as SSE (Server-Sent Events), a text format for streaming over HTTP. [The SSE parser](@/concepts/sse-parser.md) cuts it into events.
5. The adapter turns each event into a `MessageContentBlock`.
6. Each plugin's `on_stream_chunk` hook sees the block, and then you do.

`generate_stream` sets `request.provider` to the client's provider before any hook runs. You cannot mis-route a request by setting the field, and a plugin always sees the real endpoint there.

## What the request carries

`UnifiedRequest` has thirteen fields:

| Field | Meaning |
|---|---|
| `model` | The model name the server knows. |
| `provider` | Set by `generate_stream`. |
| `messages` | The conversation, in order. |
| `temperature` | Higher values give more varied text. `Option`. |
| `max_tokens` | Cap on the length of the answer, in tokens. `Option`. |
| `top_p` | Nucleus sampling: keep the likeliest tokens whose probabilities add up to this value, in `(0, 1]`. `Option`. |
| `top_k` | Keep only the `k` likeliest tokens; `0` is off. `Option`. |
| `seed` | Makes sampling repeatable. `Option`. |
| `stop` | Strings that end the answer before their first occurrence. The stop text is not emitted. An empty list means none. |
| `stream` | Whether the answer streams. On by default. |
| `thinking` | The reasoning effort. `Option`. |
| `tools` | Tool definitions the model may call. |
| `prompt_cache` | Prompt cache behavior. Disabled by default. |

An unset `Option` means the adapter leaves the key out of the request body. It never invents a default. An OpenAI-compatible body has no `temperature` or `max_tokens` unless you set them.

Set the sampling fields with `set_top_p`, `set_top_k`, `set_seed` and `add_stop`.

A provider that has no equivalent for a field rejects the request with `CucaError::Config`. It does not drop the value silently:

- Anthropic and DeepSeek's bridge route reject `seed`.
- OpenAI, DeepSeek's native route and custom endpoints reject `top_k`.
- Gemini sends all four inside `generationConfig`. Anthropic sends `stop` as `stop_sequences`.

The [provider hub](@/providers/_index.md) lists every adapter. Anthropic's API requires `max_tokens`, so that adapter sends `1024` when you leave it unset.

The request has no free-form field for a vendor setting. A map of pass-through options would be easy to extend and would end portability the first time someone used it. The one vendor override is `ThinkingParams`, a closed enum with variants for Anthropic, Gemini and OpenAI-compatible reasoning.

## Thinking: one level, several translations

Some models reason before they answer. `ThinkingEffort` has five levels. Each provider gets its closest native control, and where a provider has fewer steps, levels merge.

| Effort | OpenAI | `budget_tokens` | Adaptive | Gemini |
|---|---|---|---|---|
| `Minimal` | `"minimal"` | `1024` | `"low"` | `"LOW"` |
| `Low` | `"low"` | `2048` | `"low"` | `"LOW"` |
| `Medium` | `"medium"` | `8192` | `"medium"` | `"MEDIUM"` |
| `High` | `"high"` | `16384` | `"high"` | `"HIGH"` |
| `XHigh` | `"high"` | `16384` | `"xhigh"` | `"HIGH"` |

The columns are, in order: the OpenAI-compatible `reasoning_effort`, the Anthropic `budget_tokens`, the Anthropic adaptive `effort`, and the Gemini `thinkingLevel`.

- DeepSeek's native route has no effort setting. It takes `{"type":"enabled"}` or `{"type":"disabled"}` and drops the level.
- llama.cpp's chat route uses the OpenAI-compatible column. Its native `/completion` route has no reasoning control and ignores `thinking`.
- `ThinkingParams` carries a native value, and a set override wins over the unified level. Using it costs portability.

## Five block types

`MessageContentBlock` has five variants:

- `Text`: words of the answer.
- `ImageBase64`: an image, encoded as base64 text.
- `Thinking`: the model's reasoning.
- `ToolCall`: the model asks you to run a tool.
- `ToolResult`: your answer to a tool call.

Every adapter emits only these, so code written against one backend works against the others. A block serializes as `{"type": ..., "value": ...}`.

## The Custom provider

`ProviderEndpoint` has nine variants. Eight have adapters: `LlamaCpp`, `Vllm`, `LmStudio`, `OpenAi`, `Anthropic`, `GoogleGemini`, `DeepSeek` and `Local`. The ninth is `Custom(String)`, and using it always fails:

```text,name=CucaError::Config from a Custom endpoint
configuration error: custom endpoints require a registered adapter
```

CUCA does not guess that an unknown gateway is OpenAI-compatible. A wrong guess sends a malformed request to a real endpoint and returns a confusing error. If you know the gateway's protocol, pick that protocol's feature and override the base URL, as in [Point the client at another OpenAI-compatible server](@/guides/other-openai-server.md).

## Where a provider quirk goes

A quirk is a change in one adapter, never a new field on `UnifiedRequest`. Vendors are known only inside `src/provider/`.

- Gemini's tool results name the function, not the call id. CUCA matches a `ToolResult` by `UnifiedMessage::name` when it is set, and by the call id otherwise.
- DeepSeek's bridge route rewrites `claude-opus` to `deepseek-v4-pro`, rewrites `claude-sonnet` and `claude-haiku` to `deepseek-v4-flash`, and forces prompt caching off.
- llama.cpp's native route builds one prompt string with `### User:` and `### Assistant:` markers, because that route has no message list.

Each provider's page under [Providers](@/providers/_index.md) lists its own quirks.

Next page: [The SSE parser](@/concepts/sse-parser.md).
