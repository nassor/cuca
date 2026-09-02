+++
title = "Vector store"
description = "The bounded in-process vector store behind the memory plugin's offload seam: the config bounds, the SIMD scan, and the recall injection hand-off."
template = "page.html"
weight = 6
+++

# Vector store

<dl class="page-facts">
<dt>In one line</dt>
<dd>Keeps offloaded turns as unit-normalized embeddings in a capped arena and answers top-k similarity queries the caller injects into the next prompt.</dd>
<dt>You need</dt>
<dd>The <code>service-vector-store</code> feature, which enables <code>plugin-memory</code>.</dd>
<dt>Read this if</dt>
<dd>You are wiring recall of compacted history, or sizing the store's bounds.</dd>
</dl>

`InMemoryVectorStore` implements `VectorStore`, the offload seam declared by `plugin-memory`. `MemoryPlugin::with_extensions` is the only constructor that accepts one, and `CompactionStrategy::Offload` is what writes to it: the turns compaction removes from the live prompt are embedded and stored instead of discarded. Reading is a direct call, and the recalled turns reach the model only through `RetrievalReport::inject`.

```rust,name=Offload history, then recall it into the next request
use std::sync::Arc;

use cuca::plugins::memory::VectorStore;
use cuca::types::{MessageContentBlock, UnifiedMessage};
use cuca::{
    CompactionStrategy, Embedder, InMemoryVectorStore, MemoryConfig, MemoryPlugin, PluginError,
    RECALL_RENDER_MARKER, Summarizer, UnifiedRequest, VectorStoreConfig,
};

const DIMENSIONS: usize = 64;
const QUESTION: &str = "where does the deploy token live?";

// FNV-1a bag of words: deterministic, unlike a per-process reseeded `DefaultHasher`.
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

let mut request = UnifiedRequest::new("google/gemma-4-e4b");
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

## Try it

`examples/vector_store.rs` runs the same order against a live model, in four stages: one turn over a scripted 8-message history, `MemoryPlugin::compress` out of band moving six turns into the store, a recall for `"where does the deploy token live?"`, then `inject` plus a second turn that answers from the recall alone, since the live prompt holds two messages by then. It needs a `llama-server` on port 1234 with the demo model loaded; `CUCA_BASE_URL` and `CUCA_MODEL` retarget it at any OpenAI-compatible server.

```bash,name=Runs the same on all three platforms
cargo run --example vector_store --features "provider-llamacpp service-vector-store"
```

## Feature edge

```toml,name=Cargo.toml
service-vector-store = ["plugin-memory", "dep:wide"]
```

The edge is hard and one-way. The store is the seam's implementation, so it cannot exist without the plugin that declares the seam; `plugin-memory` never names this service, and nothing under `src/plugins/` may.

## Entry types

`Embedder`, `InMemoryVectorStore`, `RECALL_RENDER_MARKER`, `RecallInjection`, `RetrievalReport`, `RetrievedTurn`, `VectorStoreConfig`, `VectorStoreError`, `VectorStoreUsage`.

`Embedder` is the caller-supplied text-to-vector bridge, and it is synchronous: `store_turns` runs inside the synchronous `on_request` hook, so no implementation may await. No provider adapter exposes an embeddings route, so a caller wanting HTTP embeddings owns that blocking decision.

## Configuration

`VectorStoreConfig::new(max_entries, dimensions, max_entry_bytes)` is the only constructor, and every bound must be non-zero.

| Field | Meaning | Rejected with |
|---|---|---|
| `max_entries` | live turns retained | `VectorStoreError::Config` when zero |
| `dimensions` | exact embedding width every vector must have | `VectorStoreError::Config` when zero |
| `max_entry_bytes` | largest accepted turn, in payload bytes | `VectorStoreError::Config` when zero |
| `warn_fraction` | fill fraction that flips `VectorStoreUsage::near_cap` | `VectorStoreError::Config` outside `(0.0, 1.0]` |

`validate` also rejects a `max_entries * dimensions` arena, in floats or in bytes, that overflows `usize`, with the same `VectorStoreError::Config`.

At insert, a turn over `max_entry_bytes` is `VectorStoreError::EntryTooLarge`, and an embedding of the wrong width or a non-finite component is `VectorStoreError::Embedding`; a failed `Embedder` call is `VectorStoreError::Embedder`, and a poisoned state lock is `VectorStoreError::Poisoned`. `VectorStore::store_turns` carries the memory plugin's `PluginError` signature, so it converts each at the seam: `EntryTooLarge` and `Embedding` become `PluginError::Validation` with schema `vector-store`, `Embedder` passes through the caller's own error unchanged, and everything else becomes `PluginError::Internal`. One rejection aborts the whole batch, so `CompactionStrategy::Offload` puts every turn back at its original index and records the converted `PluginError` in `CompressionReport::last_error`. A later strategy such as `ClampOversizedMessages` can shrink the offending turn so a following pass succeeds.

## Retrieval

`retrieve` scans every session hint, `retrieve_in` scans one, and `retrieve_embedding` takes a caller-computed vector and skips the embedder. All three return an exact top-k `RetrievalReport`, never an approximation: the bounded entry count sits far below the size where a graph index pays off, and an RNG-driven approximate index would cost the determinism the ranking depends on.

`retrieve` and `retrieve_in` return `Result<RetrievalReport, VectorStoreError>`: `Config` for `k == 0`, `Embedder` for a failed embed call, `Embedding` for the wrong width or a non-finite component, and `Poisoned` for a poisoned state lock. `retrieve_embedding` returns the same four variants minus `Embedder`, since it skips the embedder.

Embeddings are unit-normalized once at insert, so cosine similarity is a single dot product per entry at query time, and every vector lives in one contiguous slot arena rather than a per-entry allocation. The kernel is `wide::f32x8` with four accumulators over 32 lanes per iteration, then an 8-lane loop, then a scalar tail. Ranking uses `select_nth_unstable_by` on the scored vector and sorts only the k survivors.

`wide` selects its implementation at compile time and has no runtime dispatch. A baseline `x86-64` build lowers `f32x8` to two `f32x4` halves; `+avx,+fma` unlocks the 256-bit single-rounding path.

Linux/macOS:

```
RUSTFLAGS="-C target-cpu=native" cargo build --features provider-openai,service-vector-store
```

Windows (PowerShell):

```
$env:RUSTFLAGS = "-C target-cpu=native"
cargo build --features provider-openai,service-vector-store
```

Scores and ranking are bit-deterministic for a given build: lane order and accumulator order are fixed, and the horizontal sum is a fixed-order tree rather than `f32x8::reduce_add`, whose addition order is unspecified. FMA and non-FMA builds can differ in a score's low bits, so across builds only exact ties are guaranteed stable, because they break on the monotonic insertion sequence, newest first. A zero-norm embedding is kept, counted in `VectorStoreUsage::zero_norm_entries`, and scores exactly `0.0`.

An entry is embedded from a space-joined projection of its blocks: `Text` verbatim, `ToolCall` contributes its `name`, `ToolResult` contributes its `output`. `Thinking` and `ImageBase64` are excluded.

## Capacity

| | |
|---|---|
| Bound | `max_entries` entries, each at most `max_entry_bytes`, each holding `dimensions` floats |
| At-cap policy | Deterministic FIFO: the oldest entry is evicted and its arena slot reused; the arena is allocated to `max_entries * dimensions` floats at construction and never reallocates |
| Usage gauge | `usage()` reports entries, capacity, fraction, `near_cap`, `evicted_entries`, and `zero_norm_entries`; `len()`, `is_empty()` and `capacity()` are the narrow reads |

Resident size is bounded by `max_entries * (max_entry_bytes + dimensions * 4 + session hint + constant metadata)`. There is no time-to-live: this is conversation history rather than a cache, so eviction needs no clock and the store reads none.

Entries are full-fidelity turns, including system prompts, tool arguments and tool results. The store is process-local: no serde derives, no snapshot, no `cuca-export` section, no file.

## Ordering

The order is compress, then retrieve and inject, then dispatch. Recall is a caller-driven step, so unlike the memory plugin's graph injection it cannot run after compaction inside the hook; a compaction pass running later can drop the injected system message through `DropObservations` or `SlidingWindow`. The recall system message carries at most `k` projected turns, each bounded by `max_entry_bytes`, so choosing `k` bounds the prompt growth the injection adds; a later compaction pass may still clamp or drop it.

Injection is idempotent. `inject` finds a system message whose first text block starts with `RECALL_RENDER_MARKER`, removes it, and places the new recall immediately before the most recent user message, appending when the request has no user message. `RecallInjection` reports which of `Inserted`, `Replaced`, `Removed` and `Absent` happened, so a report that changed nothing says so.

Recall is inside the prompt-cache key. `digest_request` hashes the full message list, so an injected recall changes the digest and an otherwise identical turn misses the cache.
