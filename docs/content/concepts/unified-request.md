+++
title = "The unified request and stream"
description = "Why one request type and five block types are enough for seven backends, and where per-provider difference is allowed to live."
template = "page.html"
weight = 1
+++

# The unified request and stream

<dl class="page-facts">
<dt>In one line</dt>
<dd>The unified types are the narrowest shape all seven backends can agree on, and every remaining difference is pushed into the adapter</dd>
<dt>You need</dt>
<dd>Nothing running. This page is read away from the keyboard</dd>
<dt>Read this if</dt>
<dd>You are wondering why the request has the fields it has, or where a provider quirk is supposed to go</dd>
</dl>

Seven backends speak four wire protocols between them. OpenAI, vLLM, LM Studio,
DeepSeek's native route and llama.cpp's chat route all post
`/chat/completions`. Anthropic and DeepSeek's bridge route post `/messages`.
Gemini posts `streamGenerateContent`. llama.cpp's native route posts
`/completion` with raw token frames.

A caller who wanted to switch between two of those would otherwise rewrite the
request body, the auth header, the streaming frame reader and the tool-call
round trip. `UnifiedRequest` and `MessageContentBlock` exist so that switch is a
feature flag and a base URL.

## One turn, six stages

<div class="dgm">
<div class="dgm-scroll">
<svg viewBox="0 0 660 124" role="img" aria-labelledby="pipe-title pipe-desc">
  <title id="pipe-title">The six stages of one CUCA turn</title>
  <desc id="pipe-desc">Six stages, left to right. A UnifiedRequest passes through the
  on_request hook of every registered plugin. The selected provider adapter builds the
  vendor body and posts it over HTTP. The reply bytes feed SseStreamParser, which emits
  protocol-level SSE events. The same provider adapter translates those events into
  MessageContentBlock values. Each block then passes through every plugin's
  on_stream_chunk hook before reaching the caller.</desc>
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
  <text class="t-ax" x="6" y="114">crate boundary at stage 3 and 5; the wire format never leaves the adapter</text>
</svg>
</div>
<p class="dgm-cap"><b>Stages 3 and 5 are the only places vendor JSON exists.</b>
Stage 4 is protocol work with no vendor knowledge, and stages 2 and 6 see
normalized types only.</p>
</div>

Stage 1 is not quite the request the caller built. `generate_stream` overwrites
`request.provider` with the client's own selected provider before any hook runs,
so setting the field on `UnifiedRequest` cannot mis-route a request. A plugin
reading `req.provider` in `on_request` therefore always sees the truth.

## What the request carries, and what it deliberately does not

`UnifiedRequest` has nine fields: `model`, `provider`, `messages`,
`temperature`, `max_tokens`, `stream`, `thinking`, `tools`, `prompt_cache`. Four
of them are `Option`, and an unset `Option` means the adapter omits the key
rather than inventing a default. `temperature` and `max_tokens` are absent from
an OpenAI-compatible body unless the caller set them.

The exception is Anthropic, where `max_tokens` is required by the API. That
adapter substitutes `1024` when the field is unset, because the alternative is
refusing to send a request the caller considers complete.

There is no field for a vendor-specific knob, and that is the constraint the
type is built around. A `HashMap<String, Value>` of pass-through options would
make `UnifiedRequest` trivially extensible and simultaneously destroy the reason
it exists: a request would stop being portable the moment anyone used it. The
one place a vendor override is admitted is `ThinkingParams`, and it is admitted
as a closed enum with one variant per provider, not an open map.

## Thinking: one level, five translations

`ThinkingEffort` has five levels. Each provider gets the closest native control,
and where a provider has fewer gradations than five, levels collapse. The
collapse is visible rather than hidden:

| Effort | OpenAI-compatible `reasoning_effort` | Anthropic `budget_tokens` | Anthropic adaptive `effort` | Gemini `thinkingLevel` |
|---|---|---|---|---|
| `Minimal` | `"minimal"` | `1024` | `"low"` | `"LOW"` |
| `Low` | `"low"` | `2048` | `"low"` | `"LOW"` |
| `Medium` | `"medium"` | `8192` | `"medium"` | `"MEDIUM"` |
| `High` | `"high"` | `16384` | `"high"` | `"HIGH"` |
| `XHigh` | `"high"` | `16384` | `"xhigh"` | `"HIGH"` |

DeepSeek has no effort knob at all: its native route takes a mode object,
`{"type":"enabled"}` or `{"type":"disabled"}`, and the effort level is dropped.
llama.cpp's chat route inherits the OpenAI-compatible column; its native
`/completion` route has no reasoning control and ignores `thinking` entirely.

`ThinkingParams` is the escape hatch for callers who need the native value
rather than the mapped one, and a set override wins over the unified effort.
Reaching for it costs portability, which is exactly the trade it represents.

## Five block types, not four and not twelve

`MessageContentBlock` has `Text`, `ImageBase64`, `Thinking`, `ToolCall` and
`ToolResult`. Every adapter's translator emits only these, so consuming code
written against one backend compiles and behaves against the others.

The serialization is adjacently tagged, `#[serde(tag = "type", content = "value")]`,
which is unusual enough to be worth naming. Internal tagging cannot represent a
newtype variant, and `Text(String)` is a newtype variant. The alternative was
turning `Text` into `Text { text: String }` to make internal tagging work, which
would have added a field name to the single most-used variant in the crate for
the benefit of a serialization detail. The tagging changed instead.

Blocks round-trip through `plugin-session-log`, so this is not a cosmetic
choice: it is what makes a recorded session replayable.

## The eighth provider variant

`ProviderEndpoint` has eight variants, but only seven have adapters. The eighth
is `Custom(String)`, and dispatching it always fails:

```text,name=CucaError::Config from the Custom dispatch arm
configuration error: custom endpoints require a registered adapter
```

An unknown gateway could plausibly be assumed OpenAI-compatible, since most
are. The crate refuses to assume, because a wrong guess produces a malformed
request against a real endpoint and a confusing error from the far side. A
caller who knows their gateway speaks a given protocol says so by selecting that
protocol's feature and overriding the base URL, which is
[Point the client at another OpenAI-compatible server](@/guides/other-openai-server.md).

## Where a provider quirk goes

The rule the layout enforces: normalized types know nothing about vendors, and
vendors are known only inside `src/provider/`. A quirk is therefore always an
adapter change, never a new field on `UnifiedRequest`.

Gemini drops tool-call ids because its wire format has no field for them, and
matches results back by function name. DeepSeek's bridge route rewrites
`claude-opus` to `deepseek-v4-pro` and forces prompt caching off. llama.cpp's
native route assembles a single prompt string with `### User:` markers because
it has no message array. None of that is visible in the unified types, and all
of it is stated on the provider's own page under
[Providers](@/providers/_index.md).

Next page: [The SSE parser](@/concepts/sse-parser.md), which is stage 4 of the
diagram above and the one stage with no vendor knowledge at all.
