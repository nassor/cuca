+++
title = "Memory"
description = "The context compaction and working memory graph plugin: triggers, the compaction pipeline, and graph context injection."
template = "page.html"
weight = 3
+++

# Memory

<dl class="page-facts">
<dt>In one line</dt>
<dd>Compacts the request's message list when it crosses a configured size trigger, and can render an in-memory working graph into requests as context.</dd>
<dt>You need</dt>
<dd>The <code>plugin-memory</code> feature.</dd>
<dt>Read this if</dt>
<dd>You are registering <code>MemoryPlugin</code>, tuning compaction, or working with the <code>MemoryGraph</code>.</dd>
</dl>

`MemoryPlugin` keeps a conversation's message list inside a configured size budget: it counts tokens with tiktoken-rs and, once `max_messages`, `max_tokens`, or `max_fraction` triggers, runs an ordered pipeline of compaction strategies until the request fits again. It also owns an optional in-memory working graph that `on_request` renders into every request as durable, structured context. Reach for it once a long conversation risks crossing the model's context window, or when the agent needs graph-shaped long-term memory beyond plain summarization.

```rust,name=Compact once the message count crosses a trigger
use std::sync::Arc;

use cuca::plugin::CucaPlugin;
use cuca::types::ProviderEndpoint;
use cuca::{CucaClient, MemoryConfig, MemoryPlugin, UnifiedRequest};

let memory = Arc::new(MemoryPlugin::new(MemoryConfig {
    max_messages: Some(4),
    ..Default::default()
})?);

let client = CucaClient::builder()
    .with_provider(ProviderEndpoint::LlamaCpp)
    .with_base_url("http://127.0.0.1:1234/v1")
    .register_plugin(Arc::clone(&memory) as Arc<dyn CucaPlugin>)
    .build()?;

let mut req = UnifiedRequest::new("google/gemma-4-e4b")
    .add_system_message("You are concise.")
    .add_user_message("turn 1")
    .add_user_message("turn 2")
    .add_user_message("turn 3")
    .add_user_message("turn 4")
    .add_user_message("turn 5");
memory.on_request(&mut req)?;
```

```text,name=Six messages over the trigger compact to two
System  "You are concise."
User    "turn 5"
```

## Entry types

`MemoryPlugin`, `MemoryConfig`, `MemoryGraph`, `GraphContextConfig`, `GraphNode`, `GraphRelationship`, `GraphDirection`, `MergePolicy`, `MergeReport`, `GraphSnapshot`, `Budget`, `CompactionStrategy`, `CompressionAction`, `CompressionReport`, `ContextUsage`, `ContextUsageObserver`, `ContextWindowResolver`, `Summarizer`, `VectorStore`, `GraphImportReport`.

## `CucaPlugin`

`MemoryPlugin` implements `CucaPlugin` with the plugin name `"context-memory"`. It overrides `on_request` only; `on_stream_chunk` and `on_response_complete` use the trait defaults.

## Config

`MemoryConfig` defaults:

| Field | Default |
|---|---|
| `encoder_name` | `"cl100k_base"` |
| `context_window_tokens` | 128000 |
| `context_window_resolver` | `None` |
| `max_messages` | `None` |
| `max_tokens` | `None` |
| `max_fraction` | `Some(0.8)` |
| `warn_fraction` | `None` |
| `observers` | empty |
| `offload_turns` | 10 |
| `max_drop_system_observations` | 3 |
| `graph_context` | `None` |
| `strategies` | see below |

`max_tokens` and `max_fraction` are mutually exclusive; setting both is rejected at construction. `max_messages`, when set, takes precedence over either token budget.

Default `strategies` pipeline, in order: `Offload { turns: 10 }`, `Summarize { turns: 10 }`, `DeduplicateFileReads { tool_name: "read_file" }`, `ClearToolResults { keep_pairs: 3 }`, `ClampOversizedMessages { max_part_tokens: 4096 }`, `SlidingWindow { keep_messages: 40 }`, `DropObservations`, `DropTurns`.

## Capacity, message list

| | |
|---|---|
| Bound | The active `Budget`, resolved from `max_messages`, `max_tokens`, or `max_fraction` against the resolved context window |
| At-cap policy | The ordered `strategies` pipeline runs in sequence; a strategy with no extension seam configured no-ops |
| Usage gauge | `MemoryPlugin::count_tokens()` |

## Capacity, working graph

The `MemoryGraph` itself carries no internal cap; it grows only through explicit calls to `MemoryPlugin::merge_graph`, `replace_graph`, or `replace_snapshot`, never from a hook.

| | |
|---|---|
| Bound (per-request render) | `GraphContextConfig::max_nodes` (default 64) and `max_relationships` (default 128) |
| At-cap policy | `MemoryGraph::render` selects the bounded subset for that request; the stored graph is unaffected |
| Usage gauge | `MemoryGraph::len()` (nodes) and `MemoryGraph::relationship_count()` |

When `MemoryConfig::graph_context` is set, `on_request` renders the graph into a system message placed right after the first system message. The message is idempotent: it starts with a fixed marker, so a later request replaces it in place rather than appending, and it is removed once the graph is empty.

## See also

[Entity extraction](@/services/entity-extraction.md) produces graph deltas applied to this plugin's working graph through `merge_graph` or `replace_graph`.

[Vector store](@/services/vector-store.md) implements the `VectorStore` offload seam `CompactionStrategy::Offload` writes to, wired through `MemoryPlugin::with_extensions`.
