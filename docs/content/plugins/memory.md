+++
title = "Memory"
description = "Shrink long conversations to fit a budget, and keep a small working knowledge graph."
template = "page.html"
weight = 3
+++

# Memory

<dl class="page-facts">
<dt>In one line</dt>
<dd><code>MemoryPlugin</code> shrinks a request's message list when it grows past a size you set, and can add a small knowledge graph to each request as context.</dd>
<dt>You need</dt>
<dd>The <code>plugin-memory</code> feature. Nothing else runs.</dd>
<dt>Read this if</dt>
<dd>Your conversations grow past the model's context window, or you want to feed known facts into every request.</dd>
</dl>

A model reads only so many tokens at once. A token is a small piece of text, roughly a short word or part of one. The context window is that limit. A long chat eventually does not fit.

`MemoryPlugin` counts the tokens of each request. When the count crosses your trigger, it runs a list of compaction strategies (ways to drop or shorten old messages) until the request fits.

## Register it

```rust,name=Compact a conversation once it crosses the trigger
let memory = Arc::new(MemoryPlugin::new(MemoryConfig {
    context_window_tokens: 256,
    max_messages: Some(6),
    warn_fraction: Some(0.2),
    observers: vec![gauge as Arc<dyn ContextUsageObserver>],
    strategies: vec![CompactionStrategy::SlidingWindow { keep_messages: 4 }],
    ..Default::default()
})?);
let client = CucaClient::builder()
    .with_provider(ProviderEndpoint::LlamaCpp)
    .with_base_url(base_url)
    .register_plugin(memory as Arc<dyn CucaPlugin>)
    .build()?;
```

Expected effect: a request with more than 6 messages is cut to the last 4. The first system message and the newest user message are never removed. Once usage reaches 20% of the window, one warning message is added to the request.

## Run the example

`examples/memory.rs` runs four turns with the config above. It needs a `llama-server` on port 1234 with the demo model loaded. Put `--base-url` and `--model` after `--` to use another OpenAI-compatible server.

```bash,name=Runs the same on all three platforms
cargo run --example memory --features "provider-llamacpp plugin-memory"
```

```text,name=Expected output
Trigger: max_messages 6, window 256 tokens, warn at 20%
Strategy: SlidingWindow { keep_messages: 4 }
Prompt shape letters: S system, U user, A assistant

Turn 1: "The vault is in Lisbon. Reply with: noted"
  gauge: 16/256 tokens (6%), window resolved false
  prompt: 2 messages in, 2 out (SU), warning false
  reply: noted

Turn 2: "The key rotates on friday. Reply with: noted"
  gauge: 29/256 tokens (11%), window resolved false
  prompt: 4 messages in, 4 out (SUAU), warning false
  reply: noted

Turn 3: "The owner is ops@example.com. Reply with: noted"
  gauge: 43/256 tokens (17%), window resolved false
  prompt: 6 messages in, 6 out (SUAUAU), warning false
  reply: noted

Turn 4: "Where is the vault? Answer in one word."
  gauge: 56/256 tokens (22%), window resolved false
  prompt: 8 messages in, 4 out (SAUS), warning true, compacted
  reply: Unknown.
```

In turn 4 the prompt has 8 messages plus the warning, which is over the trigger of 6. `SlidingWindow` trims it to 4. The model no longer sees the first turns, so it answers `Unknown.` Moving old turns into a store you can search later is the job of the [vector store](@/services/vector-store.md). Replies and token counts depend on the model. The message counts do not.

## Feature and entry type

| Item | Value |
|---|---|
| Feature flag | `plugin-memory` |
| Plugin name | `"context-memory"` |
| Entry types | `MemoryPlugin`, `MemoryConfig`, `MemoryGraph` |
| Attachment | `CucaClientBuilder::register_plugin` |
| Optional | A `Summarizer` (for `Summarize`) and a `VectorStore` (for `Offload`), attached with `MemoryPlugin::with_extensions(config, summarizer, store)` |

Supporting types:

- Compaction: `CompactionStrategy`, `CompressionAction`, `CompressionReport`, `Budget`.
- Usage reporting: `ContextUsage`, `ContextUsageObserver`, `ContextWindowResolver`.
- Extension points: `Summarizer`, `VectorStore`.
- Graph: `GraphContextConfig`, `GraphNode`, `GraphRelationship`, `GraphDirection`, `MergePolicy`, `MergeReport`, `GraphSnapshot`, `GraphImportReport`.

## Hooks

`MemoryPlugin` acts on `on_request` only. On every request it does these steps in order:

1. Counts the request's tokens and gives a `ContextUsage` reading to every `ContextUsageObserver`. An observer error fails the request.
2. If `warn_fraction` is set and usage reaches it, adds one system message starting `CUCA context warning:`. It adds it once.
3. Works out the `Budget`. If the request is over it, runs the compaction pipeline.
4. If `graph_context` is set, renders the working graph into a system message right after the first system message.

### Triggers

`max_messages`, `max_tokens` and `max_fraction` pick the trigger.

- `max_messages` triggers when the list is longer than the value. It wins if combined with a token trigger.
- `max_tokens` triggers when usage reaches it.
- `max_fraction` is a share of the context window. The default is `Some(0.8)`.
- `max_tokens` and `max_fraction` cannot both be set. Setting both is rejected at construction. To use `max_tokens`, also set `max_fraction: None`.
- The window comes from `ContextWindowResolver` if one is set and returns a value. Otherwise it is `context_window_tokens`.
- With no trigger, the budget is `Budget::Unlimited` and nothing is compacted.

### Compaction pipeline

<div class="dgm">
<div class="dgm-scroll">
<svg viewBox="0 0 660 190" role="img" aria-labelledby="mem-title mem-desc">
  <title id="mem-title">The compaction loop that runs when a request is over budget.</title>
  <desc id="mem-desc">The request is counted and compared to the budget. If it fits, the graph step runs and the request is sent. If it is over budget, the next strategy runs, the request is measured again, and the loop repeats until it fits, no strategy makes progress, or four passes are done.</desc>
  <defs>
    <marker id="mem-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M0 0.6 L7 4 L0 7.4 z" fill="var(--muted-foreground)"/>
    </marker>
  </defs>
  <text class="t-title" x="6" y="20">on_request in MemoryPlugin</text>
  <rect class="blk blk-data" x="5" y="38" width="130" height="50" rx="6"/>
  <text class="t-lbl t-mid" x="70" y="60">request</text>
  <text class="t-sm t-mid" x="70" y="76">count tokens</text>
  <rect class="blk blk-ctl" x="205" y="38" width="140" height="50" rx="6"/>
  <text class="t-lbl t-mid t-ctl" x="275" y="60">over budget?</text>
  <text class="t-sm t-mid" x="275" y="76">compare to Budget</text>
  <rect class="blk blk-data" x="475" y="38" width="180" height="50" rx="6"/>
  <text class="t-lbl t-mid" x="565" y="60">graph step</text>
  <text class="t-sm t-mid" x="565" y="76">then send the request</text>
  <rect class="blk blk-ctl" x="185" y="128" width="180" height="50" rx="6"/>
  <text class="t-lbl t-mid t-ctl" x="275" y="150">run next strategy</text>
  <text class="t-sm t-mid" x="275" y="166">list order, 4 passes at most</text>
  <path class="arw arw-data" d="M136 63 H203" marker-end="url(#mem-arrow)"/>
  <path class="arw arw-data" d="M346 63 H473" marker-end="url(#mem-arrow)"/>
  <text class="t-sm t-mid" x="410" y="56">no, it fits</text>
  <path class="arw arw-ctl" d="M255 90 V126" marker-end="url(#mem-arrow)"/>
  <text class="t-sm t-end" x="247" y="112">yes</text>
  <path class="arw arw-ctl" d="M295 126 V90" marker-end="url(#mem-arrow)"/>
  <text class="t-sm" x="303" y="112">measure again</text>
</svg>
</div>
<p class="dgm-cap"><b>Strategies run one at a time until the request fits.</b> The loop stops when the request is under budget, when a pass makes no progress, or after 4 passes. The graph step runs last, so compaction never removes the graph message.</p>
</div>

- A strategy that needs a `Summarizer` or `VectorStore` that is not attached does nothing. A strategy that fails is noted in the report, and the pipeline goes on.
- The newest user message and the first system message survive every strategy.
- A strategy that removes messages also removes the `ToolResult` of each dropped `ToolCall`, so no result is left without its call.
- `MemoryPlugin::compress(&mut messages)` runs the whole `strategies` list once with no budget and returns a `CompressionReport`.
- `MemoryPlugin::count_tokens()` is an estimate. It counts the `Text` blocks of each message with tiktoken-rs and adds no per-message overhead.

| Strategy | Effect |
|---|---|
| `Offload { turns }` | Moves the oldest turns to the `VectorStore`. Does nothing without a store. |
| `Summarize { turns }` | Replaces the oldest turns with one summary user message. Does nothing without a `Summarizer`. |
| `DeduplicateFileReads { tool_name }` | Blanks file-read results that a newer read of the same file replaced. |
| `ClearToolResults { keep_pairs }` | Blanks old `ToolResult` outputs and keeps the newest pairs. |
| `ClampOversizedMessages { max_part_tokens }` | Cuts the middle out of any block over the token limit, keeping its head and tail. |
| `SlidingWindow { keep_messages }` | Drops the oldest messages and keeps a tail of `keep_messages`. |
| `DropObservations` | Drops extra system messages after the first, up to `max_drop_system_observations`. |
| `DropTurns` | Drops the oldest `offload_turns` turns that can be removed. |

### Working graph

The plugin owns one `MemoryGraph`: nodes with labels and properties, and directed weighted relationships between them.

- Change it with `merge_graph(other, policy)`, `replace_graph(graph)` or `replace_snapshot(snapshot)`. Read it with `snapshot()` and `graph()`.
- The rendered message starts with `CUCA graph memory:`. A later request replaces it in place. It is removed when the graph is empty.
- The graph step runs after compaction, so compaction cannot delete the graph message.

## Configuration

`MemoryConfig` implements `Default`. `MemoryPlugin::new(config)` checks it.

| Key | Type | Default |
|---|---|---|
| `encoder_name` | `String` | `"cl100k_base"` |
| `context_window_tokens` | `u32` | `128000` |
| `context_window_resolver` | `Option<Arc<dyn ContextWindowResolver>>` | `None` |
| `max_messages` | `Option<usize>` | `None` |
| `max_tokens` | `Option<u32>` | `None` |
| `max_fraction` | `Option<f32>` | `Some(0.8)` |
| `warn_fraction` | `Option<f32>` | `None` |
| `observers` | `Vec<Arc<dyn ContextUsageObserver>>` | empty |
| `offload_turns` | `usize` | `10` |
| `max_drop_system_observations` | `usize` | `3` |
| `graph_context` | `Option<GraphContextConfig>` | `None` |
| `strategies` | `Vec<CompactionStrategy>` | the eight strategies below |

- `encoder_name` is a tiktoken-rs encoder name, or a model name such as `"gpt-4o"`.
- `context_window_tokens` is the fallback window. `max_fraction` and `warn_fraction` are shares of it.
- `offload_turns` is how many turns `DropTurns` removes per pass.
- `GraphContextConfig` defaults: `max_nodes` 64, `max_relationships` 128.

Default `strategies`, in order: `Offload { turns: 10 }`, `Summarize { turns: 10 }`, `DeduplicateFileReads { tool_name: "read_file" }`, `ClearToolResults { keep_pairs: 3 }`, `ClampOversizedMessages { max_part_tokens: 4096 }`, `SlidingWindow { keep_messages: 40 }`, `DropObservations`, `DropTurns`.

## Caps and at-cap policy

Message list:

| Item | Value |
|---|---|
| Limit | The active `Budget`, from `max_messages`, `max_tokens` or `max_fraction`. |
| At the limit | The `strategies` run in order until the request fits, at most 4 passes. |
| Usage gauge | `ContextUsageObserver` readings on every request. `MemoryPlugin::count_tokens()` on demand. |

Working graph:

| Item | Value |
|---|---|
| Limit on storage | None. The graph grows only when you call `merge_graph`, `replace_graph` or `replace_snapshot`, so you own the bound. |
| Limit per request | `GraphContextConfig::max_nodes` (64) and `max_relationships` (128). |
| At the per-request limit | `MemoryGraph::render` picks a subset for that request. The stored graph is unchanged. |
| Usage gauge | `MemoryGraph::len()` (nodes) and `MemoryGraph::relationship_count()`. |

`warn_fraction` adds a one-time near-limit warning before the limit is reached.

## Errors

Every error `MemoryPlugin` raises is `PluginError::Internal`.

- `MemoryPlugin::new` and `with_extensions` fail when both `max_tokens` and `max_fraction` are set, or when the encoder cannot load.
- `count_tokens`, `compress`, `graph`, `merge_graph`, `replace_graph`, `snapshot` and `replace_snapshot` fail when the encoder or graph lock is poisoned.
- `replace_snapshot` fails on an invalid snapshot: a duplicate node or relationship id, a weight that is not finite, or an endpoint that is not in the nodes. The working graph is not changed.
- `MemoryGraph::add_relationship` fails when `from` or `to` names an unknown node.
- `on_request` fails the request, before it is sent, if token counting fails or a `ContextUsageObserver` returns an error.
- A strategy error never fails the request. It is saved in `CompressionReport::last_error` and the pipeline goes on. A failed `Offload` puts the turns back where they were.

## Related pages

- [Plugins](@/plugins/_index.md): the hooks, their order and the error rules.
- [Memory discipline](@/concepts/memory-discipline.md): why every growing structure has a cap.
- [Entity extraction](@/services/entity-extraction.md): produces graph updates you apply with `merge_graph` or `replace_graph`.
- [Vector store](@/services/vector-store.md): implements the `VectorStore` that `Offload` writes to.
