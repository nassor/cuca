+++
title = "Vector store"
description = "Keep old conversation turns as embeddings in memory and recall the closest ones into the next request."
template = "page.html"
weight = 6
+++

# Vector store

<dl class="page-facts">
<dt>In one line</dt>
<dd>Stores turns that were removed from the prompt as embeddings (lists of numbers that capture meaning) in a capped in-memory store, and finds the closest turns to a question.</dd>
<dt>You need</dt>
<dd>The <code>service-vector-store</code> feature, which turns on <code>plugin-memory</code>, plus one <code>provider-*</code> feature. An embedding function you write.</dd>
<dt>Read this if</dt>
<dd>You compact long conversations and want the model to recall details from the removed turns, or you are sizing the store.</dd>
</dl>

## What it does

A long conversation outgrows the prompt. The [memory](@/plugins/memory.md) plugin can move old turns out of the prompt (the `Offload` strategy). `InMemoryVectorStore` keeps those turns. Later you ask it for the turns closest in meaning to a question and put them back into the next request.

<div class="dgm">
<div class="dgm-scroll">
<svg viewBox="0 0 660 200" role="img" aria-labelledby="vs-flow-title vs-flow-desc">
  <title id="vs-flow-title">Offloaded turns are embedded and stored. Before the next request you retrieve the closest turns and inject them.</title>
  <desc id="vs-flow-desc">Write path on top: the memory plugin's Offload strategy sends turns to your Embedder, which turns each into a vector, and the vector store keeps them in a capped arena. Read path below: you call retrieve with a question and k, which scans the store and returns the top k turns, then inject writes them into the request as one system message.</desc>
  <defs>
    <marker id="vs-flow-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M0 0.6 L7 4 L0 7.4 z" fill="var(--muted-foreground)"/>
    </marker>
  </defs>
  <text class="t-title" x="6" y="20">Write path: runs inside the memory plugin</text>
  <rect class="blk blk-ctl" x="6" y="36" width="140" height="50" rx="6"/>
  <text class="t-sm t-mid t-ctl" x="76" y="58">compress</text>
  <text class="t-sm t-mid" x="76" y="73">Offload strategy</text>
  <path class="arw arw-ctl" d="M148 61 H174" marker-end="url(#vs-flow-arrow)"/>
  <rect class="blk blk-bnd" x="176" y="36" width="130" height="50" rx="6"/>
  <text class="t-sm t-mid t-bnd" x="241" y="58">Embedder</text>
  <text class="t-sm t-mid" x="241" y="73">your embed()</text>
  <path class="arw arw-data" d="M308 61 H334" marker-end="url(#vs-flow-arrow)"/>
  <rect class="blk blk-data" x="336" y="36" width="150" height="50" rx="6"/>
  <text class="t-sm t-mid t-data" x="411" y="58">InMemoryVectorStore</text>
  <text class="t-sm t-mid" x="411" y="73">capped, oldest out</text>
  <text class="t-sm" x="6" y="116">Read path: you call it before the next request</text>
  <path class="arw arw-data" d="M411 88 V132" marker-end="url(#vs-flow-arrow)"/>
  <rect class="blk blk-data" x="336" y="134" width="150" height="50" rx="6"/>
  <text class="t-sm t-mid t-data" x="411" y="156">retrieve(q, k)</text>
  <text class="t-sm t-mid" x="411" y="171">top k by similarity</text>
  <path class="arw arw-data" d="M488 159 H506" marker-end="url(#vs-flow-arrow)"/>
  <rect class="blk blk-ctl" x="508" y="134" width="146" height="50" rx="6"/>
  <text class="t-sm t-mid t-ctl" x="581" y="156">inject(&amp;mut req)</text>
  <text class="t-sm t-mid" x="581" y="171">one system message</text>
</svg>
</div>
<p class="dgm-cap"><b>Recall is a step you run.</b> The write path happens inside the memory plugin's hook. The read path does not, so you call <code>retrieve</code> and <code>inject</code> before you send the request.</p>
</div>

## Try it

This program offloads four turns, finds the two closest to a question, and injects them into the next request. The embedder counts hashed words, so the scores are the same on every run. In a real app you put an embedding model behind the same one-method trait. The program sends no request and needs no server.

```rust,name=Offload history then recall it into the next request
use std::sync::Arc;

use cuca::plugins::memory::VectorStore;
use cuca::types::{MessageContentBlock, UnifiedMessage};
use cuca::{
    CompactionStrategy, Embedder, InMemoryVectorStore, MemoryConfig, MemoryPlugin, PluginError,
    RECALL_RENDER_MARKER, Summarizer, UnifiedRequest, VectorStoreConfig,
};

const DIMENSIONS: usize = 64;
const QUESTION: &str = "where does the deploy token live?";

// FNV-1a bag of words: the same vector on every run.
struct HashEmbedder;

impl Embedder for HashEmbedder {
    fn embed(&self, text: &str) -> Result<Vec<f32>, PluginError> {
        let mut vector = vec![0.0f32; DIMENSIONS];
        for token in text
            .split(|c: char| !c.is_ascii_alphanumeric())
            .filter(|token| !token.is_empty())
        {
            let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
            for byte in token.bytes() {
                hash ^= u64::from(byte.to_ascii_lowercase());
                hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
            }
            vector[(hash % DIMENSIONS as u64) as usize] += 1.0;
        }
        Ok(vector)
    }
}

// `Summarize` is not in the strategy list below, so this is never called.
struct NoSummarizer;

impl Summarizer for NoSummarizer {
    fn summarize(&self, _turns: &[UnifiedMessage]) -> String {
        String::new()
    }
}

fn first_text(message: &UnifiedMessage) -> &str {
    match message.content.first() {
        Some(MessageContentBlock::Text(text)) => text,
        _ => "",
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let store = Arc::new(InMemoryVectorStore::new(
        VectorStoreConfig::new(64, DIMENSIONS, 16 * 1024)?,
        Arc::new(HashEmbedder),
    )?);
    let memory = MemoryPlugin::with_extensions(
        MemoryConfig {
            strategies: vec![CompactionStrategy::Offload { turns: 4 }],
            ..Default::default()
        },
        Arc::new(NoSummarizer),
        Arc::clone(&store) as Arc<dyn VectorStore>,
    )?;

    let mut messages = vec![
        UnifiedMessage::system("You are concise."),
        UnifiedMessage::user("the deploy token lives in vault slot 7"),
        UnifiedMessage::assistant("Noted: the deploy token is in vault slot 7."),
        UnifiedMessage::user("the staging cluster is named borealis"),
        UnifiedMessage::assistant("Noted: staging is borealis."),
        UnifiedMessage::user(QUESTION),
    ];

    // Compress keeps the first System message and the most recent User
    // message, so the four turns between them leave the prompt for the store.
    let before = messages.len();
    let report = memory.compress(&mut messages)?;
    println!("actions: {:?}, messages: {before} -> {}", report.actions, messages.len());

    let usage = store.usage()?;
    println!("store: {}/{} entries, evicted {}", usage.entries, usage.capacity, usage.evicted_entries);

    let recall = store.retrieve(QUESTION, 2)?;
    println!("scanned {} entries", recall.scanned);
    for hit in &recall.turns {
        println!("[{:.4}] {}", hit.score, first_text(&hit.message));
    }

    let mut request = UnifiedRequest::new("any-model");
    request.messages = messages;
    println!("injection: {:?}", recall.inject(&mut request));

    // Exactly what the model reads: the recall block, marker line first.
    if let Some(text) = request
        .messages
        .iter()
        .map(first_text)
        .find(|text| text.starts_with(RECALL_RENDER_MARKER))
    {
        println!("{text}");
    }
    Ok(())
}
```

```text,name=Expected output
actions: [Offloaded], messages: 6 -> 2
store: 4/64 entries, evicted 0
scanned 4 entries
[0.5774] the deploy token lives in vault slot 7
[0.5443] Noted: the deploy token is in vault slot 7.
injection: Inserted
CUCA recall: 2 offloaded turn(s), best first
1. [0.5774] the deploy token lives in vault slot 7
2. [0.5443] Noted: the deploy token is in vault slot 7.
```

`examples/vector_store.rs` runs the same order against a live model in four stages: one turn over a scripted eight-message history, `MemoryPlugin::compress` out of band, a recall for the same question, then `inject` and a second turn that answers from the recall alone. It needs a `llama-server` on port 1234. The flags `--base-url` and `--model`, written after `--`, point it at any OpenAI-compatible server.

```bash,name=Runs the same on all three platforms
cargo run --example vector_store --features "provider-llamacpp service-vector-store"
```

## Feature and entry type

| Item | Value |
|---|---|
| Feature flag | `service-vector-store = ["plugin-memory", "dep:wide"]` |
| Plugin it needs | `plugin-memory` |
| Entry type | `InMemoryVectorStore`, which implements the memory plugin's `VectorStore` trait |
| Other public types | `Embedder`, `VectorStoreConfig`, `VectorStoreUsage`, `VectorStoreError`, `RetrievalReport`, `RetrievedTurn`, `RecallInjection`, `RECALL_RENDER_MARKER` |

The store implements a trait that `plugin-memory` declares, so it cannot exist without that plugin. `plugin-memory` never names the store.

## Entry points

### Write path

- `MemoryPlugin::with_extensions(config, summarizer, store)` is the only constructor that accepts a store.
- `CompactionStrategy::Offload { turns }` writes to the store. It calls `VectorStore::store_turns` with the fixed session hint `"cuca-memory"`, so the turns that compaction removes from the prompt are embedded and stored.
- `store_turns` runs inside the synchronous `on_request` hook, so `Embedder` is a synchronous function you supply: `embed(&self, text) -> Result<Vec<f32>, PluginError>`. It must not await. No provider adapter exposes an embeddings route, so if you want HTTP embeddings, the blocking call is yours to make.
- Each entry is embedded from its blocks joined by spaces: `Text` as is, `ToolCall` as its `name`, and `ToolResult` as its `output`. `Thinking` and `ImageBase64` are left out.
- Every embedding is scaled to unit length once, at insert. A zero-length embedding is kept, counted in `VectorStoreUsage::zero_norm_entries`, and scores exactly `0.0`.

### Read path

| Method | Scans | Embeds the query |
|---|---|---|
| `retrieve(query, k)` | every session hint | yes |
| `retrieve_in(session_hint, query, k)` | one session hint | yes |
| `retrieve_embedding(session_hint, query, k)` | one hint, or all when `None` | no: you pass the vector |

- Each returns a `RetrievalReport` with the exact top k `turns` (best first) and `scanned`. Each `RetrievedTurn` has `message`, `score`, `sequence` and `session_hint`.
- The scan is exact, not approximate. Equal scores break ties by insertion order, newest first.
- `retrieve_embedding` accepts a vector that is not scaled to unit length, and leaves your slice untouched.
- Small reads are `usage()`, `len()`, `is_empty()` and `capacity()`.
- The scan uses SIMD chosen at compile time. A build with `-C target-cpu=native` is faster on a CPU with 256-bit vectors. Scores are the same for a given build, but two builds can differ in the last digits.

```bash,name=Linux and macOS: build with native SIMD
RUSTFLAGS="-C target-cpu=native" cargo build --release --features "provider-openai service-vector-store"
```

```powershell,name=Windows (PowerShell): build with native SIMD
$env:RUSTFLAGS = "-C target-cpu=native"
cargo build --release --features "provider-openai service-vector-store"
```

### Inject the recall

A `RetrievalReport` does nothing by itself. The model sees the recall only when you call `RetrievalReport::inject(&mut request)`.

- `inject` removes a system message whose first text block starts with `RECALL_RENDER_MARKER` (`"CUCA recall:"`), so calling it twice is safe.
- It puts the new recall in a system message right before the most recent user message, or at the end when the request has no user message.
- It returns a `RecallInjection`: `Inserted`, `Replaced`, `Removed` (an empty report removed an old recall) or `Absent` (nothing to remove).
- `RetrievalReport::render()` returns the message text, or `None` for an empty report. Line one is the marker plus the hit count. Then each hit is one `N. [score] projection` line.

### Order of steps

Compress, then retrieve and inject, then send.

- Recall is a step you run. It cannot run after compaction inside the hook, as the memory plugin's graph injection does. A later compaction pass can therefore drop the injected message through `DropObservations` or `SlidingWindow`.
- An injected recall changes the request, so it changes the [prompt cache](@/services/prompt-cache.md) key. An otherwise identical turn misses the cache.

## Configuration

`VectorStoreConfig::new(max_entries, dimensions, max_entry_bytes)` is the constructor, and `with_warn_fraction(fraction)` arms the near-cap flag. `InMemoryVectorStore::new(config, embedder)` checks again, since the fields are public.

| Key | Type | Default | Valid values |
|---|---|---|---|
| `max_entries` | `usize` | required | non-zero |
| `dimensions` | `usize` | required | non-zero |
| `max_entry_bytes` | `usize` | required | non-zero |
| `warn_fraction` | `Option<f32>` | `None` | `None`, or finite and in `(0.0, 1.0]` |

- `max_entries` is the number of turns kept.
- `dimensions` is the exact length every embedding must have.
- `max_entry_bytes` is the largest accepted turn, counted in payload bytes. The count includes every block type, the message `name` and `tool_call_id`, and a tool call's arguments as JSON text.
- `warn_fraction` is the fill level at which `VectorStoreUsage::near_cap` turns `true`. `None` keeps it `false`.
- `max_entries * dimensions`, in floats and in bytes, must not overflow `usize`.

## Caps and at-cap policy

| Bound | At the cap | Gauge |
|---|---|---|
| `max_entries` turns | evicts the oldest first (FIFO) and reuses its space | `usage().entries` against `usage().capacity`, plus `evicted_entries` |
| `max_entry_bytes` per turn | rejects the turn with `EntryTooLarge`, never cuts it | the error carries `bytes` and `max_entry_bytes` |

- `usage()` is O(1). `VectorStoreUsage` holds `entries`, `capacity`, `fraction`, `near_cap`, `evicted_entries` (running total) and `zero_norm_entries`.
- The storage is allocated to `max_entries` at construction and never reallocates.
- Memory use is at most `max_entries * (max_entry_bytes + dimensions * 4 + session hint + a small constant)`.
- A batch longer than `max_entries` keeps its newest `max_entries` turns and counts the rest as evictions. Only the kept turns are embedded. The dropped head is still size-checked.
- A batch is all or nothing. A rejected turn leaves the store unchanged.
- An injected recall adds at most `k` turns, each at most `max_entry_bytes`, plus one header line. Your choice of `k` bounds that growth.
- There is no time-to-live. The store holds conversation history, not a cache, so eviction needs no clock.
- Entries are full turns, including system prompts, tool arguments and tool results. The store lives in the process. It has no serde derives, no snapshot, no `cuca-export` section, and writes no file.

## Errors

The store's own methods return `VectorStoreError`, which implements `std::error::Error`.

- `Config(String)`: a bound is zero, `warn_fraction` is outside `(0.0, 1.0]`, the storage size overflows, or a query has `k == 0`.
- `Embedding(String)`: an embedding has the wrong length, a non-finite number, or a non-finite squared length.
- `EntryTooLarge { bytes, max_entry_bytes }`: a turn is over `max_entry_bytes`.
- `Embedder(PluginError)`: your `Embedder` failed. The error is carried unchanged.
- `Poisoned`: the state lock was poisoned by a panic in another thread.
- `Json(String)`: measuring a tool call's arguments failed. This cannot happen for a `serde_json::Value`.

Which methods return which variants:

- `retrieve` and `retrieve_in` return `Config`, `Embedder`, `Embedding` and `Poisoned`.
- `retrieve_embedding` returns the same set without `Embedder`, since it skips the embedder.
- `usage`, `len` and `is_empty` return `Poisoned`.

`VectorStore::store_turns` has the memory plugin's `PluginError` signature, so the errors convert:

- `EntryTooLarge` and `Embedding` become `PluginError::Validation` with schema `vector-store`.
- `Embedder(e)` becomes `e` unchanged.
- `Config`, `Poisoned` and `Json` become `PluginError::Internal`.

One rejection aborts the whole batch. `CompactionStrategy::Offload` then puts every turn back at its original index and records the converted error in `CompressionReport::last_error`. A later strategy such as `ClampOversizedMessages` can shrink the offending turn so a following pass succeeds.

## Related pages

- [Memory](@/plugins/memory.md) declares the `VectorStore` trait and the `Offload` strategy.
- [Prompt cache](@/services/prompt-cache.md) keys on the full message list, which includes an injected recall.
- [Services](@/services/_index.md) lists every service and its plugin dependency.
- [Plugins and services](@/concepts/plugin-layering.md) explains the explicit-call tier.
- [Memory discipline](@/concepts/memory-discipline.md) explains the bounded-by-default rule.
