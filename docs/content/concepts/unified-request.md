+++
title = "The unified request and stream"
description = "Why one request type and five block types are enough for every backend, and where per-provider difference is allowed to live."
template = "page.html"
weight = 1
+++

# The unified request and stream

<dl class="page-facts">
<dt>In one line</dt>
<dd>The unified types are the narrowest shape every backend can agree on, and every remaining difference lives in the provider adapter</dd>
<dt>You need</dt>
<dd>Nothing running</dd>
<dt>Read this if</dt>
<dd>You are wondering why the request has the fields it has, or where a provider quirk belongs</dd>
</dl>

The seven HTTP backends use four wire shapes. OpenAI, vLLM, LM Studio, DeepSeek's native route and llama.cpp's chat route post `/chat/completions`. Anthropic and DeepSeek's bridge route post `/messages`. Gemini posts `streamGenerateContent`. llama.cpp's native route posts `/completion` with raw token frames. The eighth provider runs a model in process and has no wire.

Switching between two of them would otherwise mean rewriting the request body, the auth header, the frame reader and the tool-call round trip. `UnifiedRequest` and `MessageContentBlock` make that switch a feature flag and a base URL.

## One turn

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
<p class="dgm-cap"><b>Stages 3 and 5 are the only places vendor JSON exists.</b> Stage 4 is protocol work with no vendor knowledge, and stages 2 and 6 see normalized types only.</p>
</div>

`generate_stream` overwrites `request.provider` with the client's selected provider before any hook runs. Setting the field on a `UnifiedRequest` therefore cannot mis-route it, and a plugin reading `req.provider` in `on_request` always sees the real endpoint. `UnifiedRequest::new` starts with `ProviderEndpoint::Custom("")` for that reason.

## What the request carries

`UnifiedRequest` has nine fields: `model`, `provider`, `messages`, `temperature`, `max_tokens`, `stream`, `thinking`, `tools` and `prompt_cache`. Three are `Option`: `temperature`, `max_tokens` and `thinking`. An unset `Option` means the adapter omits the key instead of inventing a default, so an OpenAI-compatible body has no `temperature` or `max_tokens` unless the caller set them.

Anthropic is the exception because its API requires `max_tokens`. That adapter sends `1024` when the field is unset, since the alternative is refusing a request the caller considers complete.

The request has no field for a vendor-specific knob, and the type is built around that constraint. A `HashMap<String, Value>` of pass-through options would make `UnifiedRequest` easy to extend and would end its portability the first time anyone used it. The one vendor override is `ThinkingParams`, a closed enum with variants for Anthropic, Gemini and OpenAI-compatible reasoning, plus `None`.

## Thinking: one level, several translations

`ThinkingEffort` has five levels. Each provider gets its closest native control, and where a provider has fewer gradations the levels collapse.

| Effort | OpenAI-compatible `reasoning_effort` | Anthropic `budget_tokens` | Anthropic adaptive `effort` | Gemini `thinkingLevel` |
|---|---|---|---|---|
| `Minimal` | `"minimal"` | `1024` | `"low"` | `"LOW"` |
| `Low` | `"low"` | `2048` | `"low"` | `"LOW"` |
| `Medium` | `"medium"` | `8192` | `"medium"` | `"MEDIUM"` |
| `High` | `"high"` | `16384` | `"high"` | `"HIGH"` |
| `XHigh` | `"high"` | `16384` | `"xhigh"` | `"HIGH"` |

- DeepSeek's native route has no effort knob. It takes a mode object, `{"type":"enabled"}` or `{"type":"disabled"}`, and drops the effort level.
- llama.cpp's chat route uses the OpenAI-compatible column. Its native `/completion` route has no reasoning control and ignores `thinking`.
- `ThinkingParams` carries a native value for a caller who needs it, and a set override wins over the unified effort. Using it costs portability.

## Five block types

`MessageContentBlock` has `Text`, `ImageBase64`, `Thinking`, `ToolCall` and `ToolResult`. Every adapter's translator emits only these, so code written against one backend behaves the same against the others.

The serialization is adjacently tagged with `#[serde(tag = "type", content = "value")]`. Internal tagging cannot represent a newtype variant, and `Text(String)` is one. Session logging serializes `Text` blocks, so every variant has to round-trip.

## The Custom provider

`ProviderEndpoint` has nine variants. Eight have adapters: `LlamaCpp`, `Vllm`, `LmStudio`, `OpenAi`, `Anthropic`, `GoogleGemini`, `DeepSeek` and `Local`. The ninth is `Custom(String)`, and dispatching it always fails.

```text,name=CucaError::Config from the Custom dispatch arm
configuration error: custom endpoints require a registered adapter
```

The crate does not assume an unknown gateway is OpenAI-compatible. A wrong guess sends a malformed request to a real endpoint and returns a confusing error from the far side. A caller who knows the gateway's protocol selects that protocol's feature and overrides the base URL, as in [Point the client at another OpenAI-compatible server](@/guides/other-openai-server.md).

## Where a provider quirk goes

Normalized types know nothing about vendors, and vendors are known only inside `src/provider/`. A quirk is therefore an adapter change, never a new field on `UnifiedRequest`.

- Gemini's `functionResponse` names the function, not the call id, so a `ToolResult` is matched by `UnifiedMessage::name` when set, with the call id as a fallback.
- DeepSeek's bridge route rewrites `claude-opus` to `deepseek-v4-pro`, `claude-sonnet` and `claude-haiku` to `deepseek-v4-flash`, and forces prompt caching off.
- llama.cpp's native route builds one prompt string with `### User:` and `### Assistant:` markers, because that route has no message array.

None of this is visible in the unified types. Each provider's page under [Providers](@/providers/_index.md) states its own.

Next page: [The SSE parser](@/concepts/sse-parser.md).
