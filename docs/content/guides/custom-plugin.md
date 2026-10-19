+++
title = "Write a custom plugin"
description = "Pick the hook, implement CucaPlugin with interior mutability, register it on the builder, watch it fire."
template = "page.html"
weight = 2
+++

# Write a custom plugin

<dl class="page-facts">
<dt>In one line</dt>
<dd>Implement <code>name</code> plus the hooks you need, then pass an <code>Arc</code> of the plugin to <code>register_plugin</code></dd>
<dt>You need</dt>
<dd>A working stream, from <a href="@/quick-start/first-stream.md">Stream a first reply</a></dd>
<dt>Read this if</dt>
<dd>You want to observe, rewrite or intercept part of a turn without forking the crate</dd>
</dl>

Goal: run your own code at a chosen point of every turn. The trait has five methods and only `name` is required, so a useful plugin is about thirty lines.

## Step 1: pick the hook

Choose by when your code must run.

| You want to | Hook | Receives | If it returns `Err` |
|---|---|---|---|
| Change the request before it is sent | `on_request` | `&mut UnifiedRequest` | `generate_stream` fails before dispatch; later plugins do not run |
| Answer a tool call yourself | `execute_local_tool` | `&MessageContentBlock` (a `ToolCall`) | the stream yields `CucaError::Plugin` in place of the block and keeps polling |
| See or rewrite each block | `on_stream_chunk` | `&mut MessageContentBlock` | the stream yields `CucaError::Plugin` in place of the block, which is not accumulated; polling continues |
| Act once on the finished turn | `on_response_complete` | `&UnifiedResponse` | logged and dropped; the remaining plugins still run |

`execute_local_tool` returns `Option<MessageContentBlock>`, and the first plugin to return `Some` wins.

If no hook fits your capability, it belongs in the service tier, which is driven by direct method calls. See [Plugins and services](@/concepts/plugin-layering.md).

Result: you know which of the four hooks you override.

## Step 2: keep state behind `&self`

Every hook takes `&self`, because the client shares the plugin list as `Arc<dyn CucaPlugin>` across `await` points. Mutable state needs interior mutability: an `AtomicUsize` for a counter, a `Mutex` for anything larger.

Result: your struct holds its state in atomics or locks, and the trait bound `Send + Sync` holds.

## Step 3: implement the trait

The bundled example overrides two hooks and takes the defaults for the rest. It counts streamed blocks and prints the aggregate.

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
        // The hook fires inside the caller's final `next()` poll, so the
        // leading newline is what separates this line from the streamed reply.
        println!(
            "\n[example-block-counter] model={} duration={:.2}s completion_tokens={} blocks={}",
            res.model,
            res.duration_secs,
            res.completion_tokens,
            self.blocks.load(Ordering::Relaxed),
        );
        Ok(())
    }
}
```

`name` returns a `&'static str`. The client puts it on the `tracing` error event it emits when an `on_response_complete` hook fails, and only under `plugin-guardrails`. Make it unique and recognizable.

Result: the plugin compiles against `cuca::plugin::CucaPlugin`.

## Step 4: register it

```rust,name=Registration order is hook order
let client = CucaClient::builder()
    .with_provider(ProviderEndpoint::LlamaCpp)
    .with_base_url(base_url.clone())
    .register_plugin(Arc::new(BlockCounterPlugin::default()))
    .build()?;
```

Call `register_plugin` once per plugin. Plugins run in registration order at every hook site, and `client.plugins()` returns them in that order.

Result: `client.plugins().len()` is the number of plugins you registered.

## Step 5: run it

Start `llama-server` on port 1234 as in [Stream a first reply](@/quick-start/first-stream.md), then run the example.

```bash,name=Runs the same on all three platforms
cargo run --example custom_plugin --features provider-llamacpp
```

```text,name=Expected output (one run against gemma-4-12b-qat)
CUCA most commonly refers to either the Credit Union of Central Alabama or the Center for Urban Community Action, depending on the context.
[example-block-counter] model=google/gemma-4-12b-qat duration=81.89s completion_tokens=1991 blocks=1991
```

Result: the reply streams, then one summary line prints. The reply text, the counts and the duration depend on the model. This model reasons before it answers and the example sets no `max_tokens`, so both counts are large.

Reading the summary:

- `completion_tokens` is the server's count from its final usage frame.
- `blocks` counts every block your hook saw, and a text delta carries about one token, so the two roughly agree for a text-only reply.
- They diverge once the reply carries an `ImageBase64` or `ToolResult` block, which count as blocks but not as tokens.
- A server that reports no usage leaves `completion_tokens` at one per `Text`, `Thinking` and `ToolCall` block, and `UnifiedResponse::usage` at `None`.

## If your plugin answers tool calls

For a `ToolCall` with id `X`, a value returned from `execute_local_tool` must be a `ToolResult` whose `tool_call_id` is `X`. Any other block fails the call:

```text,name=CucaError::Plugin carrying PluginError::Validation on a mismatched result
plugin error: validation failed for schema local tool result: local tool executor must return a ToolResult for the input ToolCall id
```

Return `Ok(None)` for a tool call you do not handle, so the next plugin gets its turn.

## If your plugin holds state that grows

State whose size follows traffic needs a bound, a stated policy at the bound, and a way to read current usage. The existing structures and the reasoning for each policy are in [Memory discipline](@/concepts/memory-discipline.md).

## If your plugin stores session records

Implement `SessionStorePlugin` on top of `CucaPlugin`. It adds `append_log`, `replay_session` and `fork_session`, none with a default body. `ModelOrchestrator::with_session_store` takes an `Arc<dyn SessionStorePlugin>` to record model swaps. The bundled implementation is [Session log](@/plugins/session-log.md).

## If the prompt cache is on

A prompt-cache hit replays the stored blocks without calling `execute_local_tool` or `on_stream_chunk`. `on_request` still runs, and `on_response_complete` runs once. See [Prompt cache](@/services/prompt-cache.md).

Next page: [Run the live integration suite](@/guides/integration-tests.md).
