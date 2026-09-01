+++
title = "Write a custom plugin"
description = "Pick the hook, implement CucaPlugin with interior mutability, register it on the builder, watch it fire."
template = "page.html"
weight = 2
+++

# Write a custom plugin

<dl class="page-facts">
<dt>In one line</dt>
<dd>Implement <code>name</code> plus the one or two hooks you need, then pass an <code>Arc</code> of it to <code>register_plugin</code></dd>
<dt>You need</dt>
<dd>A working stream, from <a href="@/quick-start/first-stream.md">Stream a first reply</a></dd>
<dt>Read this if</dt>
<dd>You want to observe, rewrite or intercept part of a turn without forking the crate</dd>
</dl>

Five methods on one trait. Four have default bodies, so a useful plugin is
usually thirty lines.

## Step 1: pick the hook

Choose by when your code needs to run, not by what it does.

| You want to | Hook | Signature detail |
|---|---|---|
| change the request before it is sent | `on_request` | `&mut UnifiedRequest`; the first error aborts the turn before dispatch |
| answer a tool call yourself, locally | `execute_local_tool` | returns `Option<MessageContentBlock>`; the first plugin to return a value wins |
| see or rewrite every block as it arrives | `on_stream_chunk` | `&mut MessageContentBlock`; the first error fails the stream |
| act once on the finished turn | `on_response_complete` | `&UnifiedResponse`; errors here are logged, not propagated |

If none of those moments is right for your capability, it probably should not be
a `CucaPlugin` at all. Three of the crate's own sixteen features are not, and the
reasoning is in [Everything is a plugin](@/concepts/plugin-layering.md).

## Step 2: note that every hook takes `&self`

This is the one thing that catches people out. Hooks are `&self`, not
`&mut self`, because the plugin list is shared across `await` points as
`Arc<dyn CucaPlugin>`. State therefore needs interior mutability: an
`AtomicUsize` for a counter, a `Mutex` for anything larger.

## Step 3: implement the trait

The bundled example counts blocks and reports the aggregate. It overrides two
hooks and takes the defaults for the rest:

```rust,name=The whole plugin from examples/custom_plugin.rs
#[derive(Default)]
struct BlockCounterPlugin {
    blocks: AtomicUsize,
}

impl CucaPlugin for BlockCounterPlugin {
    fn name(&self) -> &'static str {
        "example-block-counter"
    }

    fn on_stream_chunk(&self, _chunk: &mut MessageContentBlock) -> Result<(), PluginError> {
        self.blocks.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    fn on_response_complete(&self, res: &UnifiedResponse) -> Result<(), PluginError> {
        println!(
            "[example-block-counter] model={} duration={:.2}s completion_tokens={} blocks={}",
            res.model,
            res.duration_secs,
            res.completion_tokens,
            self.blocks.load(Ordering::Relaxed),
        );
        Ok(())
    }
}
```

`name` is the only required method. It returns `&'static str` and appears in the
log line the client writes when an `on_response_complete` hook fails, so make it
identifiable.

## Step 4: register it

```rust,name=Registration order is hook order
let client = CucaClient::builder()
    .with_provider(ProviderEndpoint::LlamaCpp)
    .with_base_url(base_url)
    .register_plugin(Arc::new(BlockCounterPlugin::default()))
    .build()?;
```

Call `register_plugin` once per plugin. They run in the order you registered
them, at every hook site, and `client.plugins()` returns them in that order.

## Step 5: run it

```bash,name=Runs the same on all three platforms
cargo run --example custom_plugin --features provider-llamacpp
```

```text,name=The summary line; values will differ
[example-block-counter] model=google/gemma-4-e4b duration=2.31s completion_tokens=23 blocks=23
```

Both counts agree for a text-only reply, because the client counts one token per
`Text`, `Thinking` and `ToolCall` block. They diverge as soon as the reply
carries an `ImageBase64` or `ToolResult` block, which count as blocks but not as
tokens.

## If your plugin answers tool calls

`execute_local_tool` has one rule the compiler cannot enforce. Given a
`ToolCall` with id `X`, a returned value must be a `ToolResult` whose
`tool_call_id` is `X`. Anything else fails the stream:

```text,name=PluginError::Validation on a mismatched result
validation failed for schema local tool result: local tool executor must return a ToolResult for the input ToolCall id
```

Return `Ok(None)` for a tool call you do not handle, so the next plugin gets its
turn.

## If your plugin holds state that grows

Anything whose size follows traffic needs a bound, a stated policy for reaching
it, and a way to read current usage. That is a hard rule in this crate, not a
suggestion, and the six existing examples of it plus the reasoning behind
refusing rather than evicting are in
[Memory discipline](@/concepts/memory-discipline.md).

## If your plugin stores session records

Implement `SessionStorePlugin` on top of `CucaPlugin`. It adds `append_log`,
`replay_session` and `fork_session`, none of which has a default body, and it is
what the speculative orchestrator calls to record a model swap. The bundled
implementation is [`plugin-session-log`](@/plugins/session-log.md).

Next page: [Run the live integration suite](@/guides/integration-tests.md).
