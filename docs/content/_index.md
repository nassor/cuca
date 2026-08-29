+++
title = "What CUCA is"
description = "An asynchronous Rust client for LLM backends: one unified request, one typed block stream, seven provider adapters, thirteen plugins."
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
<dd>You are deciding whether <code>cuca-core</code> fits, or looking for where a fact lives on this site</dd>
</dl>

`cuca-core` is one library crate. It builds a request, dispatches it to a model
backend, parses the Server-Sent Events the backend streams back, and hands the
caller a stream of normalized blocks. Every capability beyond that sits behind a
Cargo feature.

## The shape of a turn

A caller constructs a `UnifiedRequest`, hands it to `CucaClient::generate_stream`,
and polls the returned `AgentResponseStream`. Each item is a
`Result<MessageContentBlock, CucaError>`, and `MessageContentBlock` has exactly
five variants: `Text`, `ImageBase64`, `Thinking`, `ToolCall`, `ToolResult`.

```rust,name=src/request.rs lines 350 to 351
pub type AgentResponseStream =
    Pin<Box<dyn Stream<Item = Result<MessageContentBlock, CucaError>> + Send>>;
```

Those same five variants arrive whether the answer came from Anthropic's
Messages API, Gemini's `streamGenerateContent`, or a llama.cpp server on
loopback. What each provider actually does with the request is
[The unified request and stream](@/concepts/unified-request.md).

## Nothing compiles by default

`default = []`. The crate declares seven `provider-*` features and thirteen
`plugin-*` features, and `src/lib.rs` stops the build with a `compile_error!`
until at least one `provider-*` feature is enabled:

```text,name=src/lib.rs line 14
CUCA requires one provider-* feature; enable the adapter for the backend you use.
```

There is no implicit backend and no implicit dependency. `reqwest`, `tokio`,
`wasmtime`, `rmcp`, `tiktoken-rs`, `jsonschema` and `opentelemetry` are all
optional, pulled in only by the feature that needs them. The grid of features,
their dependencies and the single cross-plugin edge is
[The feature matrix](@/reference/features.md).

## Where things are

- [Stream a first reply](@/quick-start/first-stream.md): a local llama.cpp
  server, one cargo command, text on stdout.
- [The unified request and stream](@/concepts/unified-request.md): why the
  request has the fields it has, and where per-provider difference is allowed to
  live.
- [Providers](@/providers/_index.md): seven adapters, each with its default base
  URL, auth header, route and behavior.
- [Plugins](@/plugins/_index.md): thirteen capabilities, each with its feature
  flag, entry type, hooks, config defaults and caps.
- [How-to guides](@/guides/other-openai-server.md): point the client somewhere
  else, write a plugin, run the live suite.
- [Reference](@/reference/features.md): the feature matrix, the error types, the
  environment variables, the public API surface.

## What this site does not hold

Rustdoc is the item-level reference: signatures, field docs and exact generic
bounds live next to the code, where the compiler keeps them honest. This site
holds what rustdoc cannot: the reading order, the cross-provider comparison
tables, and the reasoning behind the design.

Next page: [Stream a first reply](@/quick-start/first-stream.md).
