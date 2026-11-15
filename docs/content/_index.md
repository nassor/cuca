+++
title = "What CUCA is"
description = "A Rust library that sends one kind of request to many LLM APIs, and runs language models inside your own program."
template = "section.html"
sort_by = "weight"

[extra]
kicker = "Compact Universal Client for Agents"
+++

<dl class="page-facts">
<dt>In one line</dt>
<dd>A Rust library, plus a <code>cuca</code> command, that talks to language models over the network or runs them itself</dd>
<dt>You need</dt>
<dd>Rust 1.98 or newer, and a checkout of the repository</dd>
<dt>Read this if</dt>
<dd>You are deciding whether CUCA fits, or you want to know where to start</dd>
</dl>

`cuca` is a Rust crate (a Rust library package) and a command-line program for working with large language models, or LLMs. An LLM is the kind of model behind a chat assistant: you send it a conversation and it writes text back.

CUCA does two jobs:

- It sends your request to an LLM service over HTTP, such as OpenAI or Anthropic, and gives you the answer in one shape whichever service replied.
- It runs a model itself, on your own CPU or GPU, so no outside service is involved.

## Who it is for

- Rust developers who build an app or an agent (a program that lets a model call tools) on top of an LLM.
- Teams that want to switch between LLM services without rewriting their code.
- Anyone who wants to run an open model on their own machine, inside an app or as a server.

## Start here

1. [Stream a first reply](@/quick-start/first-stream.md). About 15 minutes. You start a local model server and watch text arrive in your terminal.
2. [After the first run](@/quick-start/after-the-first-run.md). Short recipes for your own project: all block types, a plugin, thinking.
3. [Providers](@/providers/_index.md). Pick the service you will really use, or run a model yourself with [Local (in-process)](@/providers/local/_index.md).
4. [The unified request and stream](@/concepts/unified-request.md). Read this when you want to know why the request looks the way it does.

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
<p class="dgm-cap"><b>One engine, three ways to reach it.</b> Your program calls it in process. Any OpenAI client reaches it through <code>cuca serve</code>, alone or behind <code>cuca balance</code>.</p>
</div>

### Call an LLM API

You build one `UnifiedRequest` and CUCA sends it to OpenAI, Anthropic, Gemini, DeepSeek, llama.cpp, vLLM or LM Studio. A stream of typed blocks comes back. The service that answers is called a provider, and CUCA's code for one provider is its adapter.

Two optional extras sit around the request. Plugins watch every request and every block of the answer, for example to log, check or redact them. Services are tools you call yourself, such as a prompt cache. Both are covered in [Plugins and services](@/concepts/plugin-layering.md).

### Run a model yourself

`provider-local` runs a language model in Rust, on the CPU, a GPU or both. The model is a GGUF file, the single-file format llama.cpp made common, or a Hugging Face safetensors checkpoint that CUCA converts once to a cached GGUF. A model that is bigger than your GPU memory can still run: CUCA keeps some of it in RAM or on the SSD. See [Model files and downloads](@/providers/local/model-sources.md) and [Fit a large model](@/providers/local/placement.md).

The local engine has three modes:

- Library mode: the model runs inside your program, behind the same `CucaClient`. See [Local (in-process)](@/providers/local/_index.md).
- Server mode: `cuca serve` answers OpenAI-compatible chat requests over HTTP, with metrics and a stats API. See [Serve a model over HTTP](@/guides/serve-a-model.md) and [Model server](@/server/model-server.md).
- Balancer mode: `cuca balance` spreads requests over several `cuca serve` instances. See [Balancer](@/server/balancer.md).

## One turn through the client

You build a `UnifiedRequest`, pass it to `CucaClient::generate_stream`, and read the stream it returns. Plugins can change the request on the way out and each block on the way back.

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
<p class="dgm-cap"><b>Vendor formats exist only inside the provider adapter.</b> Your code and the plugins see the same types for every backend.</p>
</div>

Each item in the stream is a `MessageContentBlock`, or an error. A block is one of five kinds:

- `Text`: words of the answer.
- `Thinking`: the model's reasoning, when it shows any.
- `ToolCall`: the model asks your program to run a tool.
- `ToolResult`: the result of a tool, sent back to the model.
- `ImageBase64`: an image.

The same five kinds arrive from every backend, including the in-process model. [The unified request and stream](@/concepts/unified-request.md) explains why.

## Nothing compiles by default

CUCA uses Cargo features, which are switches that turn parts of a crate on. None is on by default, so you pick only what you need.

- Eight provider adapters, each behind a `provider-*` feature: seven for HTTP services and `provider-local` for the in-process engine.
- Twelve plugins, each behind a `plugin-*` feature.
- Seven services, each behind a `service-*` feature.
- `server` builds the `cuca` program for server and balancer mode, and `server-tui` adds its terminal view.

A build with no `provider-*` feature stops with this message:

```text,name=Message from a build with no provider feature
CUCA requires one provider-* feature; enable the adapter for the backend you use.
```

`server` turns on `provider-local`, so it builds alone. The feature grid is on [The feature matrix](@/reference/features.md).

## Where to go

- [Stream a first reply](@/quick-start/first-stream.md): a local llama.cpp server, one cargo command, text in your terminal.
- [The unified request and stream](@/concepts/unified-request.md): the design behind the request and block types.
- [Providers](@/providers/_index.md): eight adapters, each with its default base URL, auth header and behavior.
- [Plugins](@/plugins/_index.md): twelve capabilities that watch the request and the stream.
- [Services](@/services/_index.md): seven capabilities you call directly.
- [How-to guides](@/guides/other-openai-server.md): point the client at another server, write a plugin, serve a model, run the live tests.
- [Server](@/server/_index.md): `cuca serve`, `cuca balance` and `cuca tui`.
- [Reference](@/reference/features.md): the feature matrix, error types, configuration files and the public API.
- Rustdoc holds signatures, field docs and generic bounds for each item.

Next page: [Stream a first reply](@/quick-start/first-stream.md).
