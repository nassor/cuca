+++
title = "After the first run"
description = "See all five block types, add a plugin, depend on the crate from your own project, and turn thinking on."
template = "page.html"
weight = 2
+++

# After the first run

<dl class="page-facts">
<dt>In one line</dt>
<dd>Four changes that turn the demo into the start of your own client</dd>
<dt>You need</dt>
<dd><a href="@/quick-start/first-stream.md">Stream a first reply</a> working, with <code>llama-server</code> still up on port 1234</dd>
<dt>Read this if</dt>
<dd>The demo ran and you want to know what to change first</dd>
</dl>

Each section below stands alone. Take the one that matches what you need next.

## See every block type

`llamacpp_gemma` prints `Text` blocks and drops the rest. `stream_all_blocks`
matches on all five `MessageContentBlock` variants, which is the shape real
consuming code has:

```bash,name=Runs the same on all three platforms
cargo run --example stream_all_blocks --features provider-llamacpp
```

`Text` goes to stdout. Everything else goes to stderr, tagged:

```text,name=stderr tags the example writes
[reasoning] ...
[tool call] <name> <arguments> (id: <id>)
[image] <media_type> (base64 data omitted)
[tool result] for tool call <tool_call_id>
[error] <CucaError Display text>
```

An `[error]` line stops the drain. Which tags appear depends on the server and
the model: a plain prose reply is all `Text`, while a reasoning model buries it.
One run against `google/gemma-4-12b-qat` wrote 1506 `[reasoning]` lines to
stderr for two lines of answer on stdout:

```text,name=stdout from that run
1. In Romanian folklore, the **Cuca** is a legendary bogeyman used to frighten children into behaving.
2. It is traditionally depicted as a witch-like creature with a long nose and long hair.
```

## Add a plugin

Plugins are the crate's only extension point, and a plugin is one `impl` block.
`custom_plugin` counts blocks as they stream and reports the aggregated response
when the stream ends:

```bash,name=Runs the same on all three platforms
cargo run --example custom_plugin --features provider-llamacpp
```

```text,name=The summary line from one run against gemma-4-12b-qat
CUCA most commonly refers to either the Credit Union of Central Alabama or the Center for Urban Community Action, depending on the context.
[example-block-counter] model=google/gemma-4-12b-qat duration=81.89s completion_tokens=1991 blocks=1991
```

The two counts agree because the client counts one token per `Text`, `Thinking`
and `ToolCall` block, and both reach 1991 because the example sets no
`max_tokens` and this model reasons before it answers. `prompt_tokens` stays
`0`: no adapter populates it.

To write your own, see [Write a custom plugin](@/guides/custom-plugin.md).

## Depend on the crate from your own project

The crate is named `cuca` and is not on crates.io, so a dependant names the
checkout by path and picks its features explicitly:

```bash,name=Runs the same on all three platforms
cargo add cuca --path /path/to/cuca --features provider-llamacpp
```

Swap `provider-llamacpp` for whichever adapter speaks to your backend. The seven
choices and what each one talks to are in [Providers](@/providers/_index.md).
Add `plugin-*` features the same way, one per capability you want compiled in.

## Turn thinking on

`UnifiedRequest::thinking` is `None` by default, and providers with no reasoning
knob ignore it either way. One unified effort level maps onto each provider's
native control:

```rust,name=One effort level for five providers
use cuca::{ThinkingEffort, UnifiedRequest};

let request = UnifiedRequest::new(model)
    .add_user_message("Explain the trade-off.")
    .enable_thinking(Some(ThinkingEffort::Medium));
```

`Medium` becomes `reasoning_effort: "medium"` on OpenAI-compatible routes,
`budget_tokens: 8192` on Anthropic's budget mode, `thinkingLevel: "MEDIUM"` on
Gemini. The full mapping table, including where two effort levels collapse onto
one provider value, is in
[The unified request and stream](@/concepts/unified-request.md).

## Point somewhere other than llama.cpp

Any OpenAI-compatible server, including Ollama, vLLM and LM Studio, is a base
URL change plus the right provider feature. That is its own guide:
[Point the client at another OpenAI-compatible server](@/guides/other-openai-server.md).

## Run the tests

The unit tests need no server:

```bash,name=Runs the same on all three platforms
cargo test --no-default-features --features provider-openai
```

The integration suite is server-dependent and skips what it cannot reach. Its
feature list, its three environment variables and how to make a skip fail
loudly are in
[Run the live integration suite](@/guides/integration-tests.md).

Next page: [The unified request and stream](@/concepts/unified-request.md),
which explains why the request has the fields it has.
