+++
title = "After the first run"
description = "See all five block types, add a plugin, depend on the crate from your own project, and turn thinking on."
template = "page.html"
weight = 2
+++

# After the first run

<dl class="page-facts">
<dt>In one line</dt>
<dd>Short recipes that turn the demo into the start of your own client</dd>
<dt>You need</dt>
<dd><a href="@/quick-start/first-stream.md">Stream a first reply</a> working, with <code>llama-server</code> still up on port 1234</dd>
<dt>Read this if</dt>
<dd>The demo ran and you want to know what to change first</dd>
</dl>

Each section stands alone. Take the one that matches what you need next.

## See every block type

The first demo prints `Text` blocks and drops the rest. The `stream_all_blocks` example handles all five kinds of `MessageContentBlock`, which is the shape real code has:

1. Run it from the `cuca` checkout.

   ```bash,name=Runs the same on all three platforms
   cargo run --example stream_all_blocks --features provider-llamacpp
   ```

2. Read the two outputs. `Text` goes to stdout, which is the normal terminal output. Everything else goes to stderr, the error channel, with a tag in front:

   ```text,name=Tags the example writes to stderr
   [reasoning] ...
   [tool call] <name> <arguments> (id: <id>)
   [image] <media_type> (base64 data omitted)
   [tool result] for tool call <tool_call_id>
   [error] <error text>
   ```

An `[error]` line stops the example. Which tags appear depends on the server and the model. A plain answer is all `Text`. A reasoning model also writes many `[reasoning]` lines before its answer.

## Add a plugin

A plugin is code that watches every request and every block of the answer. You write one with a single `impl` block. The `custom_plugin` example counts blocks as they stream and prints a summary when the stream ends:

1. Run it.

   ```bash,name=Runs the same on all three platforms
   cargo run --example custom_plugin --features provider-llamacpp
   ```

2. After the reply, look for the summary line. The values depend on your model.

   ```text,name=Summary line format
   [example-block-counter] model=<model> duration=<seconds>s completion_tokens=<n> blocks=<n>
   ```

`completion_tokens` comes from the server's own count. A token is a small piece of text, often part of a word, and models read and write text in tokens. llama-server reports its count in a final usage frame. A server that reports no usage leaves the prompt count at `0`, and `completion_tokens` counts the `Text`, `Thinking` and `ToolCall` blocks.

To write your own plugin, see [Write a custom plugin](@/guides/custom-plugin.md). To learn what a plugin is, see [Plugins and services](@/concepts/plugin-layering.md).

## Depend on the crate from your own project

The crate is named `cuca` and is not on crates.io. You point Cargo at your checkout by path and pick its features yourself:

```bash,name=Runs the same on all three platforms
cargo add cuca --path /path/to/cuca --features provider-llamacpp
```

On Windows, write the path as `C:\path\to\cuca`.

Swap `provider-llamacpp` for the adapter that fits your backend. There are eight: seven for HTTP services and the in-process `provider-local`. [Providers](@/providers/_index.md) lists what each one talks to. Add `plugin-*` features the same way, one for each capability you want compiled in.

## Turn thinking on

Some models can reason before they answer. `UnifiedRequest::thinking` is `None` by default, and a provider with no reasoning control ignores it. CUCA has one effort level and maps it to each provider's own control:

```rust,name=One effort level for every provider
use cuca::{ThinkingEffort, UnifiedRequest};

let request = UnifiedRequest::new(model)
    .add_user_message("Explain the trade-off.")
    .enable_thinking(Some(ThinkingEffort::Medium));
```

`Medium` becomes `reasoning_effort: "medium"` on OpenAI-compatible routes, `budget_tokens: 8192` on Anthropic, and `thinkingLevel: "MEDIUM"` on Gemini. The full table is in [The unified request and stream](@/concepts/unified-request.md).

## Point somewhere other than llama.cpp

Any OpenAI-compatible server, including Ollama, vLLM and LM Studio, needs a different base URL and the right provider feature. The steps are in [Point the client at another OpenAI-compatible server](@/guides/other-openai-server.md).

## Run a model without a server

`provider-local` runs a model inside your program, so there is no `llama-server` to start. See [Local (in-process)](@/providers/local/_index.md).

## Run the tests

The unit tests need no server:

```bash,name=Runs the same on all three platforms
cargo test --no-default-features --features provider-openai
```

The integration suite needs a running server and skips what it cannot reach. [Run the live integration suite](@/guides/integration-tests.md) has its features, its `cuca-test.kdl` config file, and how to make a skip fail.

Next page: [The unified request and stream](@/concepts/unified-request.md).
