+++
title = "What CUCA is"
description = "A Rust crate for language models: a client for seven LLM APIs, and a local engine that runs a model in your program, as an OpenAI-compatible server, or behind a load balancer."
template = "section.html"
sort_by = "weight"

[extra]
kicker = "Compact Universal Client for Agents"
+++

<dl class="page-facts">
<dt>In one line</dt>
<dd>A client for LLM APIs, and a local engine you run as a library, a model server or a load balancer</dd>
<dt>You need</dt>
<dd>Rust 1.98 or newer, and a checkout of the repository</dd>
<dt>Read this if</dt>
<dd>You are deciding whether <code>cuca</code> fits, or looking for where a fact lives on this site</dd>
</dl>

`cuca` is a Rust crate, plus the `cuca` binary, for working with language models. It talks to LLM APIs over HTTP, and it runs models itself. Everything sits behind a Cargo feature.

## Two ways to use CUCA

<div class="dgm">
<div class="dgm-scroll">
<svg viewBox="10 12 660 290" role="img" aria-labelledby="use-title use-desc">
  <title id="use-title">Two ways to use CUCA: a client for LLM APIs, and a local engine with three modes</title>
  <desc id="use-desc">Top row: your application holds a CucaClient, which sends requests over HTTP to an LLM API: OpenAI, Anthropic, Gemini, DeepSeek, llama.cpp, vLLM or LM Studio. Middle row: the same application can instead run the local engine, provider-local, in its own process on the CPU, a GPU or both; this is library mode. Bottom row: HTTP clients that speak the OpenAI API call cuca balance, which forwards each request to a cuca serve instance, which runs the local engine; these are balancer mode and server mode.</desc>
  <defs>
    <marker id="use-arrow" viewBox="0 0 8 8" refX="7" refY="4"
            markerWidth="6" markerHeight="6" orient="auto">
      <path d="M0 0.6 L7 4 L0 7.4 z" fill="var(--muted-foreground)"/>
    </marker>
  </defs>
  <text class="t-title" x="20" y="32">Two ways to use CUCA</text>
  <rect class="blk" x="20" y="56" width="150" height="56" rx="6"/>
  <text class="t-lbl t-mid" x="95" y="80">your application</text>
  <text class="t-sm t-mid" x="95" y="97">CucaClient</text>
  <rect class="blk blk-bnd" x="450" y="48" width="190" height="80" rx="6"/>
  <text class="t-lbl t-mid t-bnd" x="545" y="70">LLM APIs</text>
  <text class="t-sm t-mid" x="545" y="88">OpenAI, Anthropic,</text>
  <text class="t-sm t-mid" x="545" y="103">Gemini, DeepSeek,</text>
  <text class="t-sm t-mid" x="545" y="118">llama.cpp, vLLM, LM Studio</text>
  <path class="arw arw-bnd" d="M172 84 H444" marker-end="url(#use-arrow)"/>
  <text class="t-sm t-mid t-bnd" x="308" y="76">HTTP</text>
  <text class="t-ax t-mid" x="308" y="100">client for LLM APIs</text>
  <rect class="blk blk-data" x="450" y="152" width="190" height="56" rx="6"/>
  <text class="t-lbl t-mid t-data" x="545" y="176">local engine</text>
  <text class="t-sm t-mid" x="545" y="193">provider-local, CPU and GPU</text>
  <path class="arw arw-data" d="M95 114 V180 H444" marker-end="url(#use-arrow)"/>
  <text class="t-sm t-mid t-data" x="270" y="172">in process</text>
  <text class="t-ax t-mid" x="270" y="196">library mode</text>
  <rect class="blk" x="20" y="236" width="150" height="50" rx="6"/>
  <text class="t-lbl t-mid" x="95" y="257">HTTP clients</text>
  <text class="t-sm t-mid" x="95" y="273">OpenAI API</text>
  <rect class="blk blk-data" x="196" y="236" width="104" height="50" rx="6"/>
  <text class="t-lbl t-mid t-data" x="248" y="257">cuca balance</text>
  <text class="t-sm t-mid" x="248" y="273">balancer mode</text>
  <rect class="blk blk-data" x="326" y="236" width="110" height="50" rx="6"/>
  <text class="t-lbl t-mid t-data" x="381" y="257">cuca serve</text>
  <text class="t-sm t-mid" x="381" y="273">server mode</text>
  <path class="arw arw-data" d="M172 261 H192" marker-end="url(#use-arrow)"/>
  <path class="arw arw-data" d="M302 261 H322" marker-end="url(#use-arrow)"/>
  <path class="arw arw-data" d="M438 261 H545 V212" marker-end="url(#use-arrow)"/>
</svg>
</div>
<p class="dgm-cap"><b>One engine, three ways to reach it.</b> Your program calls it in process; any OpenAI client reaches it through <code>cuca serve</code>, alone or behind <code>cuca balance</code>.</p>
</div>

### Client for LLM APIs

One `UnifiedRequest` goes to OpenAI, Anthropic, Gemini, DeepSeek, llama.cpp, vLLM or LM Studio, and one stream of typed blocks comes back. Plugins watch the request and the stream, and services such as the prompt cache are called directly. Start with [Stream a first reply](@/quick-start/first-stream.md), then [Providers](@/providers/_index.md).

### Local engine

`provider-local` runs a GGUF model in Rust on the CPU, a GPU or both. One model can span VRAM, RAM and the SSD: layers split between the GPU and the CPU, and a mixture of experts keeps its experts on disk behind RAM and VRAM caches. See [Hybrid placement](@/providers/local/placement.md). It has three modes:

- Library mode: the model runs inside your program behind the same `CucaClient`. See [Local (in-process)](@/providers/local/_index.md).
- Server mode: `cuca serve` answers OpenAI-compatible chat requests over HTTP, with metrics and a stats API. See [Serve a model over HTTP](@/guides/serve-a-model.md) and [Model server](@/server/model-server.md).
- Balancer mode: `cuca balance` spreads requests over several `cuca serve` instances and keeps a conversation on the server that caches its prefix. See [Balancer](@/server/balancer.md).

## One turn through the client

A caller builds a `UnifiedRequest`, passes it to `CucaClient::generate_stream`, and polls the returned `AgentResponseStream`. Plugins can change the request on the way out and each block on the way back. In library mode the provider adapter is the local engine.

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

The crate sets `default = []`. Its eight provider adapters each sit behind a `provider-*` feature, and there are twelve `plugin-*` features and seven `service-*` features. `server` builds the `cuca` binary for server and balancer mode, and `server-tui` adds its terminal UI. `src/lib.rs` stops the build with a `compile_error!` until at least one `provider-*` feature is on; `server` turns on `provider-local`, so it builds alone.

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
- [Server](@/server/_index.md): `cuca serve`, `cuca balance` and `cuca tui`, an OpenAI-compatible endpoint over the in-process engine with metrics and a live terminal view.
- [Reference](@/reference/features.md): the feature matrix, error types, the configuration files and the public API surface.
- Rustdoc holds signatures, field docs and generic bounds for each item.

Next page: [Stream a first reply](@/quick-start/first-stream.md).
