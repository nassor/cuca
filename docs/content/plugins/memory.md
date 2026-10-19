+++
title = "Memory"
description = "The context compaction and working memory graph plugin: triggers, the compaction pipeline, and graph context injection."
template = "page.html"
weight = 3
+++

# Memory

<dl class="page-facts">
<dt>In one line</dt>
<dd><code>MemoryPlugin</code> compacts the request's message list when it crosses a size trigger, and renders an in-memory working graph into requests as context.</dd>
<dt>You need</dt>
<dd>The <code>plugin-memory</code> feature.</dd>
<dt>Read this if</dt>
<dd>You are registering <code>MemoryPlugin</code>, tuning compaction, or working with the <code>MemoryGraph</code>.</dd>
</dl>

## Feature and entry type

| Item | Value |
|---|---|
| Feature flag | `plugin-memory` (enables `dep:tiktoken-rs`) |
| Plugin name | `"context-memory"` |
| Entry types | `MemoryPlugin`, `MemoryConfig`, `MemoryGraph` |
| Attachment | `CucaClientBuilder::register_plugin` |
| Needs | Nothing running; an optional `Summarizer` and `VectorStore` enable two strategies |

Supporting types:

- Compaction: `CompactionStrategy`, `CompressionAction`, `CompressionReport`, `Budget`.
- Usage reporting: `ContextUsage`, `ContextUsageObserver`, `ContextWindowResolver`.
- Extension seams: `Summarizer`, `VectorStore`.
- Graph: `GraphContextConfig`, `GraphNode`, `GraphRelationship`, `GraphDirection`, `MergePolicy`, `MergeReport`, `GraphSnapshot`, `GraphImportReport`.

## Hooks

`MemoryPlugin` acts on `on_request` only. `on_stream_chunk` and `on_response_complete` use the trait defaults.

On every request, `on_request` runs these steps in order:

1. Count the request's tokens and hand a `ContextUsage` reading to every `ContextUsageObserver`. An observer error fails the request.
2. When `warn_fraction` is set and usage reaches it, append one system message that starts with `CUCA context warning:`. The marker makes the warning one-shot.
3. Resolve the `Budget`. When the request is over it, run the compaction pipeline.
4. When `graph_context` is set, render the working graph into a system message right after the first system message.

### Triggers

- `max_messages`, `max_tokens` and `max_fraction` select the trigger. `max_messages` wins when combined with either token budget.
- `max_tokens` and `max_fraction` are mutually exclusive; setting both is rejected at construction. The default `max_fraction` is `Some(0.8)`, so a config that sets `max_tokens` must also set `max_fraction: None`.
- `max_messages` triggers when the list is longer than the value. `max_tokens` triggers when usage reaches it. `max_fraction` multiplies the resolved context window.
- The window comes from `ContextWindowResolver` when one is set and returns a value, otherwise from `context_window_tokens`.
- With no trigger set, the budget is `Budget::Unlimited` and `on_request` compacts nothing.

### Compaction pipeline

- Strategies run in the order of `strategies`. A strategy whose extension seam is absent no-ops, and a strategy error is recorded and the pipeline continues.
- With a trigger, the pipeline re-measures after each acting strategy and stops once under budget. It runs at most 4 passes and stops early when a pass makes no progress.
- The most recent user message and the first system message survive every strategy.
- Every strategy that removes messages also removes the `ToolResult` message of each dropped `ToolCall`, so no result outlives its call.
- `MemoryPlugin::compress(&mut messages)` is the out-of-band entry point. It runs the whole `strategies` list once with no budget and returns a `CompressionReport`.
- `MemoryPlugin::count_tokens()` approximates: it sums tiktoken-rs counts of `"<role> <text>"` per message, over `Text` blocks only, and models no per-message framing tokens.

| Strategy | Effect |
|---|---|
| `Offload { turns }` | Moves the oldest turns to the `VectorStore`; no-op without a store. |
| `Summarize { turns }` | Replaces the oldest turns with one summary user message; no-op without a `Summarizer`. |
| `DeduplicateFileReads { tool_name }` | Blanks file-read results superseded by a newer read of the same file. |
| `ClearToolResults { keep_pairs }` | Blanks old `ToolResult` outputs, keeping the newest pairs. |
| `ClampOversizedMessages { max_part_tokens }` | Head and tail truncates any block over the token limit. |
| `SlidingWindow { keep_messages }` | Drops the oldest messages down to a tail of `keep_messages`. |
| `DropObservations` | Drops redundant system messages after the first, up to `max_drop_system_observations`. |
| `DropTurns` | Drops the oldest `offload_turns` removable turns. |

### Working graph

- The plugin owns one `MemoryGraph` behind a mutex: nodes with labels and properties, and directed weighted relationships.
- `merge_graph(other, policy)`, `replace_graph(graph)`, `replace_snapshot(snapshot)`, `snapshot()` and `graph()` access it. Merging happens under one lock hold.
- The rendered message starts with the marker `CUCA graph memory:`. A later request replaces it in place, and it is removed once the graph is empty.
- The graph step runs after compaction, so compaction cannot delete the graph message, and its token cost is not counted against the request that triggered compaction.

## Configuration

`MemoryConfig` implements `Default`; `MemoryPlugin::new(config)` validates it.

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
| `strategies` | `Vec<CompactionStrategy>` | the eight-step pipeline below |

- `encoder_name` is a tiktoken-rs encoder name or a model name such as `"gpt-4o"`.
- `context_window_tokens` is the fallback window, and the base for `max_fraction` and `warn_fraction`.
- `offload_turns` is how many turns `DropTurns` removes per pass.
- `GraphContextConfig` defaults: `max_nodes` 64, `max_relationships` 128.

Default `strategies`, in order: `Offload { turns: 10 }`, `Summarize { turns: 10 }`, `DeduplicateFileReads { tool_name: "read_file" }`, `ClearToolResults { keep_pairs: 3 }`, `ClampOversizedMessages { max_part_tokens: 4096 }`, `SlidingWindow { keep_messages: 40 }`, `DropObservations`, `DropTurns`.

`MemoryPlugin::with_extensions(config, summarizer, store)` attaches the `Summarizer` and `VectorStore` that `Summarize` and `Offload` need.

## Caps and at-cap policy

Message list:

| Item | Value |
|---|---|
| Bound | The active `Budget`, resolved from `max_messages`, `max_tokens` or `max_fraction` against the resolved window. |
| At the bound | The `strategies` pipeline runs in order until the request fits, at most 4 passes. |
| Usage gauge | `ContextUsageObserver` readings on every request; `MemoryPlugin::count_tokens()` on demand. |

Working graph:

| Item | Value |
|---|---|
| Bound on storage | None. The graph grows only through explicit `merge_graph`, `replace_graph` and `replace_snapshot` calls, never from a hook, so the caller owns the bound. |
| Bound per request | `GraphContextConfig::max_nodes` (64) and `max_relationships` (128). |
| At the per-request bound | `MemoryGraph::render` selects a bounded subset for that request; the stored graph is unchanged. |
| Usage gauge | `MemoryGraph::len()` (nodes) and `MemoryGraph::relationship_count()`. |

`warn_fraction` injects the one-shot near-limit warning before the bound is reached.

## Errors

Every error `MemoryPlugin` itself raises is `PluginError::Internal`.

- `MemoryPlugin::new` and `with_extensions` fail when both `max_tokens` and `max_fraction` are set, or when the encoder cannot load.
- `count_tokens`, `compress`, `graph`, `merge_graph`, `replace_graph`, `snapshot` and `replace_snapshot` fail when the encoder or graph mutex is poisoned.
- `replace_snapshot` fails on an invalid snapshot: a duplicate node or relationship id, a non-finite weight, or an endpoint missing from the nodes. The working graph is untouched.
- `MemoryGraph::add_relationship` fails when `from` or `to` names an unknown node.
- `on_request` returns the token-count failure and any error a `ContextUsageObserver` returns. Either fails the request before dispatch.
- A strategy error never fails the request. It lands in `CompressionReport::last_error` and the pipeline continues. A failed `Offload` returns the turns to their original positions.

## Example

`examples/memory.rs` runs four turns with `max_messages: Some(6)`, a 256 token window, `warn_fraction: Some(0.2)` and a single `SlidingWindow { keep_messages: 4 }` strategy.

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

Turn 4 carries both edits. The gauge crosses `warn_fraction`, so one warning message is appended, and the nine message prompt exceeds the six message trigger, so `SlidingWindow` trims it to four. The surviving shape is `SAUS` because the first system message and the most recent user message are never removed. The reply `Unknown.` follows from the dropped turns: reaching them again is the offload seam of the [vector store](@/services/vector-store.md).

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

The replies and token counts depend on the model; the message counts do not. The output above came from `google/gemma-4-12b-qat`.

## Related pages

- [Plugins](@/plugins/_index.md): hook order, error handling and the plugin tier rules.
- [Memory discipline](@/concepts/memory-discipline.md): why every growing structure carries a cap.
- [Entity extraction](@/services/entity-extraction.md): produces graph deltas applied through `merge_graph` or `replace_graph`.
- [Vector store](@/services/vector-store.md): implements the `VectorStore` seam that `Offload` writes to.
