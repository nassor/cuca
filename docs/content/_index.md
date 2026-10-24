+++
title = "What CUCA is"
description = "An asynchronous Rust client for LLM backends: one unified request, one typed block stream, eight provider adapters, twelve plugins and seven services."
template = "section.html"
sort_by = "weight"

[extra]
kicker = "Compact Universal Client for Agents"
+++

<dl class="page-facts">
<dt>In one line</dt>
<dd>One <code>UnifiedRequest</code> goes in, one <code>AgentResponseStream</code> of typed blocks comes out, whichever backend answers</dd>
<dt>You need</dt>
<dd>Rust 1.98 or newer, and a checkout of the repository</dd>
<dt>Read this if</dt>
<dd>You are deciding whether <code>cuca</code> fits, or looking for where a fact lives on this site</dd>
</dl>

`cuca` is a Rust library crate for talking to language models. It builds a request, sends it to one of eight providers, and gives the caller a stream of normalized blocks. Seven providers are HTTP servers and one runs a GGUF model inside your process. Everything beyond that core sits behind a Cargo feature.

## The shape of a turn

A caller builds a `UnifiedRequest`, passes it to `CucaClient::generate_stream`, and polls the returned `AgentResponseStream`. Plugins can change the request on the way out and each block on the way back.

<div class="dgm">
<div class="dgm-scroll">
<svg viewBox="0 0 660 124" role="img" aria-labelledby="home-title home-desc">
  <title id="home-title">One turn through CucaClient::generate_stream</title>
  <desc id="home-desc">Five stages, left to right. A UnifiedRequest passes through the on_request hook of every registered plugin. The provider adapter sends it to the backend and translates the reply. The adapter produces a stream of MessageContentBlock values. Each block passes through every plugin's on_stream_chunk hook before it reaches the caller. When the stream ends, every plugin's on_response_complete hook runs once.</desc>
  <defs>
    <marker id="home-arrow" viewBox="0 0 8 8" refX="7" refY="4"
            markerWidth="6" markerHeight="6" orient="auto">
      <path d="M0 0.6 L7 4 L0 7.4 z" fill="var(--muted-foreground)"/>
    </marker>
  </defs>
  <text class="t-title" x="6" y="20">One turn through CucaClient::generate_stream</text>
  <rect class="blk" x="6" y="44" width="104" height="50" rx="6"/>
  <text class="t-sm t-mid" x="58" y="67">Unified</text>
  <text class="t-sm t-mid" x="58" y="81">Request</text>
  <rect class="blk blk-ctl" x="142" y="44" width="104" height="50" rx="6"/>
  <text class="t-sm t-mid t-ctl" x="194" y="67">plugins</text>
  <text class="t-sm t-mid" x="194" y="81">on_request</text>
  <rect class="blk blk-bnd" x="278" y="44" width="104" height="50" rx="6"/>
  <text class="t-sm t-mid t-bnd" x="330" y="67">provider</text>
  <text class="t-sm t-mid" x="330" y="81">adapter</text>
  <rect class="blk blk-data" x="414" y="44" width="104" height="50" rx="6"/>
  <text class="t-sm t-mid t-data" x="466" y="67">MessageContent</text>
  <text class="t-sm t-mid" x="466" y="81">Block stream</text>
  <rect class="blk blk-ctl" x="550" y="44" width="104" height="50" rx="6"/>
  <text class="t-sm t-mid t-ctl" x="602" y="67">plugins</text>
  <text class="t-sm t-mid" x="602" y="81">on_stream_chunk</text>
  <path class="arw" d="M112 69 H138" marker-end="url(#home-arrow)"/>
  <path class="arw arw-bnd" d="M248 69 H274" marker-end="url(#home-arrow)"/>
  <path class="arw arw-data" d="M384 69 H410" marker-end="url(#home-arrow)"/>
  <path class="arw arw-ctl" d="M520 69 H546" marker-end="url(#home-arrow)"/>
  <text class="t-ax" x="6" y="114">on_response_complete runs once when the stream ends</text>
</svg>
</div>
<p class="dgm-cap"><b>Vendor protocols exist only inside the provider adapter.</b> The caller and the plugins see the same types for every backend.</p>
</div>

Each stream item is a `Result<MessageContentBlock, CucaError>`. `MessageContentBlock` has five variants: `Text`, `ImageBase64`, `Thinking`, `ToolCall` and `ToolResult`.

```rust,name=src/request.rs
pub type AgentResponseStream =
    Pin<Box<dyn Stream<Item = Result<MessageContentBlock, CucaError>> + Send>>;
```

The same five variants arrive from Anthropic's Messages API, Gemini's `streamGenerateContent`, a llama.cpp server on loopback, or the in-process model. Why the request has the fields it has is on [The unified request and stream](@/concepts/unified-request.md).

## Nothing compiles by default

The crate sets `default = []`. Its eight provider adapters each sit behind a `provider-*` feature, and there are twelve `plugin-*` features and seven `service-*` features. `src/lib.rs` stops the build with a `compile_error!` until at least one `provider-*` feature is on.

```text,name=src/lib.rs
CUCA requires one provider-* feature; enable the adapter for the backend you use.
```

No backend and no dependency is implicit. `reqwest`, `tokio`, `wasmtime`, `rmcp`, `tiktoken-rs`, `jsonschema` and `opentelemetry` are optional and enabled only by the feature that needs them. The feature grid and the edges between features are on [The feature matrix](@/reference/features.md).

## Where to go

- [Stream a first reply](@/quick-start/first-stream.md): a local llama.cpp server, one cargo command, text on stdout.
- [The unified request and stream](@/concepts/unified-request.md): the design behind the request and block types.
- [Providers](@/providers/_index.md): eight adapters, each with its default base URL, auth header, route and behavior.
- [Plugins](@/plugins/_index.md): twelve pipeline capabilities, each with its feature flag, entry type, hooks, config defaults and caps.
- [Services](@/services/_index.md): seven explicit-call capabilities, driven by direct method calls instead of hooks.
- [How-to guides](@/guides/other-openai-server.md): point the client at another server, write a plugin, run the live suite.
- [Reference](@/reference/features.md): the feature matrix, error types, environment variables and the public API surface.
- Rustdoc holds signatures, field docs and generic bounds for each item.

Next page: [Stream a first reply](@/quick-start/first-stream.md).
