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
<dd>The <code>service-vector-store</code> feature, which enables <code>plugin-memory</code>, plus one <code>provider-*</code> feature.</dd>
<dt>Read this if</dt>
<dd>You are wiring recall of compacted history, or sizing the store's bounds.</dd>
</dl>

## Feature and entry type

| Item | Value |
|---|---|
| Feature flag | `service-vector-store = ["plugin-memory", "dep:wide"]` |
| Declared plugin edge | `plugin-memory`, hard |
| Entry type | `InMemoryVectorStore`, which implements the memory plugin's `VectorStore` seam |
| Other public types | `Embedder`, `VectorStoreConfig`, `VectorStoreUsage`, `VectorStoreError`, `RetrievalReport`, `RetrievedTurn`, `RecallInjection`, `RECALL_RENDER_MARKER` |

`InMemoryVectorStore` is an explicit-call service. It never implements `CucaPlugin`, so passing it to `register_plugin` is a compile error. The edge is one-way. The store implements a seam that `plugin-memory` declares, so it cannot exist without that plugin. `plugin-memory` never names the service, and nothing under `src/plugins/` may.

## Entry points

### Write path

- `MemoryPlugin::with_extensions(config, summarizer, store)` is the only constructor that accepts a store.
- `CompactionStrategy::Offload { turns }` writes to the store. It calls `VectorStore::store_turns` with the fixed session hint `"cuca-memory"`, so the turns compaction removes from the live prompt are embedded and stored.
- `store_turns` runs inside the synchronous `on_request` hook, so `Embedder` is a synchronous, caller-supplied bridge: `embed(&self, text) -> Result<Vec<f32>, PluginError>`. No implementation may await. No provider adapter exposes an embeddings route, so a caller that wants HTTP embeddings owns that blocking decision.
- An entry is embedded from a space-joined projection of its blocks: `Text` verbatim, `ToolCall` contributes its `name`, and `ToolResult` contributes its `output`. `Thinking` and `ImageBase64` are excluded.
- Every embedding is unit-normalized once at insert. A zero-norm embedding is kept, counted in `VectorStoreUsage::zero_norm_entries`, and scores exactly `0.0`.

### Read path

| Method | Scans | Embeds the query |
|---|---|---|
| `retrieve(query, k)` | every session hint | yes |
| `retrieve_in(session_hint, query, k)` | one session hint | yes |
| `retrieve_embedding(session_hint, query, k)` | one hint, or all when `None` | no; the caller passes the vector |

- Each returns an exact top-k `RetrievalReport` with `turns` (best first) and `scanned`. Each `RetrievedTurn` carries `message`, `score`, `sequence` and `session_hint`.
- The scan is exact and never approximate. The bounded entry count sits far below the size where a graph index pays off, and a randomized index would give up the deterministic ranking.
- `retrieve_embedding` accepts a vector that is not normalized and leaves the caller's slice untouched.
- Narrow reads are `usage()`, `len()`, `is_empty()` and `capacity()`.

### Hand-off: inject the recall

A `RetrievalReport` is inert. Recall reaches the model only when the application calls `RetrievalReport::inject(&mut request)`.

- `inject` removes a system message whose first text block starts with `RECALL_RENDER_MARKER` (`"CUCA recall:"`), so it is idempotent.
- It places the new recall as a system message immediately before the most recent user message, or appends it when the request has no user message.
- It returns a `RecallInjection`: `Inserted`, `Replaced`, `Removed` (an empty report removed an existing recall) or `Absent` (nothing to remove).
- `RetrievalReport::render()` returns the message text, or `None` for an empty report. Line one is the marker plus the hit count, then one `N. [score] projection` line per hit.

### Ordering against compaction

The order is compress, then retrieve and inject, then dispatch.

- Recall is a caller-driven step. It cannot run after compaction inside the hook as the memory plugin's graph injection does, so a later compaction pass can drop the injected message through `DropObservations` or `SlidingWindow`.
- `digest_request` hashes the full message list. An injected recall changes the digest, so an otherwise identical turn misses the prompt cache.

### Scan kernel

- Vectors live in one contiguous slot arena, so a scan walks memory linearly. Cosine similarity is one dot product per entry.
- The kernel is `wide::f32x8` with four accumulators (32 lanes per iteration), then an 8-lane loop, then a scalar tail. Ranking uses `select_nth_unstable_by`, then sorts only the k survivors.
- Each query allocates one embedding `Vec` (skipped by `retrieve_embedding`), one scored `Vec` of 16-byte tuples, and k message clones.
- `wide` selects its implementation at compile time and has no runtime dispatch. A baseline `x86-64` build lowers `f32x8` to two `f32x4` halves. `-C target-cpu=native`, or `+avx,+fma`, enables the 256-bit single-rounding path.
- Scores and ranking are bit-deterministic for a given build: lane order and accumulator order are fixed, and the horizontal sum is a fixed-order tree. FMA and non-FMA builds can differ in a score's low bits, so across builds only exact ties are stable. Ties break on the monotonic insertion sequence, newest first.

`benches/vector_store.rs` runs under Divan with `k = 8`. It pairs the store's `retrieve_embedding` against a plain scalar dot-product loop with the same ranking at 1024 entries of 384 floats, 8192 of 768, and 8192 of 1536. It also times `store_turns` for a batch of 64 turns of 768 floats.

```bash,name=Runs the same on all three platforms
cargo bench --features "provider-openai service-vector-store"
```

```bash,name=Linux and macOS: bench with native SIMD
RUSTFLAGS="-C target-cpu=native" cargo bench --features "provider-openai service-vector-store"
```

```powershell,name=Windows (PowerShell): bench with native SIMD
$env:RUSTFLAGS = "-C target-cpu=native"
cargo bench --features "provider-openai service-vector-store"
```

## Configuration

`VectorStoreConfig::new(max_entries, dimensions, max_entry_bytes)` is the constructor, and `with_warn_fraction(fraction)` arms the near-cap flag. `InMemoryVectorStore::new(config, embedder)` validates again, since the fields are public.

| Key | Type | Default | Valid values |
|---|---|---|---|
| `max_entries` | `usize` | required | non-zero |
| `dimensions` | `usize` | required | non-zero |
| `max_entry_bytes` | `usize` | required | non-zero |
| `warn_fraction` | `Option<f32>` | `None` | `None`, or finite and in `(0.0, 1.0]` |

- `max_entries` is the number of live turns retained.
- `dimensions` is the exact width every embedding must have.
- `max_entry_bytes` is the largest accepted turn, measured in payload bytes. The count includes every block variant, the message `name` and `tool_call_id`, and a tool call's arguments as serialized JSON.
- `warn_fraction` is the fill fraction at which `VectorStoreUsage::near_cap` turns `true`; `None` keeps it `false`.
- `max_entries * dimensions`, in floats and in bytes, must not overflow `usize`.

## Caps and at-cap policy

| Bound | At-cap policy | Usage gauge |
|---|---|---|
| `max_entries` live turns | evicts entries oldest first (FIFO) and reuses the freed arena slot | `usage().entries` against `usage().capacity`, plus `evicted_entries` |
| `max_entry_bytes` per turn | rejects the turn with `EntryTooLarge`, never truncates | the error carries `bytes` and `max_entry_bytes` |

- `usage()` is O(1). `VectorStoreUsage` holds `entries`, `capacity`, `fraction`, `near_cap`, `evicted_entries` (cumulative) and `zero_norm_entries`.
- The arena and the entry queue are allocated to `max_entries` at construction and never reallocate.
- Resident size is bounded by `max_entries * (max_entry_bytes + dimensions * 4 + session hint + constant metadata)`.
- A batch longer than `max_entries` keeps its newest `max_entries` turns and counts the rest as evictions. Only the surviving turns are embedded. The dropped head is still size-checked.
- A batch commits all or nothing. A rejected turn leaves the store unchanged.
- An injected recall adds at most `k` projected turns, each at most `max_entry_bytes`, plus one header line. The choice of `k` bounds that growth.
- There is no time-to-live. The store holds conversation history rather than a cache, so eviction needs no clock.
- Entries are full-fidelity turns, including system prompts, tool arguments and tool results. The store is process-local: it has no serde derives, no snapshot, no `cuca-export` section and writes no file.

## Errors

The store's own methods return `VectorStoreError`, which implements `std::error::Error`.

- `Config(String)`: a bound is zero, `warn_fraction` is outside `(0.0, 1.0]`, the arena size overflows, or a query has `k == 0`.
- `Embedding(String)`: an embedding has the wrong width, a non-finite component, or a non-finite squared norm.
- `EntryTooLarge { bytes, max_entry_bytes }`: a turn is over `max_entry_bytes`.
- `Embedder(PluginError)`: the caller's `Embedder` failed. The error is carried unchanged.
- `Poisoned`: the state lock was poisoned by a panic in another thread.
- `Json(String)`: measuring a tool call's arguments failed. This cannot happen for a `serde_json::Value`.

Which methods return which variants:

- `retrieve` and `retrieve_in` return `Config`, `Embedder`, `Embedding` and `Poisoned`.
- `retrieve_embedding` returns the same set without `Embedder`, since it skips the embedder.
- `usage`, `len` and `is_empty` return `Poisoned`.

`VectorStore::store_turns` has the memory plugin's `PluginError` signature, so the seam converts:

- `EntryTooLarge` and `Embedding` become `PluginError::Validation` with schema `vector-store`.
- `Embedder(e)` becomes `e` unchanged.
- `Config`, `Poisoned` and `Json` become `PluginError::Internal`.

One rejection aborts the whole batch. `CompactionStrategy::Offload` then puts every turn back at its original index and records the converted error in `CompressionReport::last_error`. A later strategy such as `ClampOversizedMessages` can shrink the offending turn so a following pass succeeds.

## Example

The program below offloads four turns, retrieves the two closest to a question, and injects them into the next request. The embedder is a deterministic bag of words, so the scores repeat on every run. A real deployment puts an embedding model behind the same one-method trait. The program sends no request, so it needs no server.

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

`examples/vector_store.rs` runs the same order against a live model in four stages: one turn over a scripted eight-message history, `MemoryPlugin::compress` out of band, a recall for the same question, then `inject` plus a second turn that answers from the recall alone. It needs a `llama-server` on port 1234 with the demo model loaded; `CUCA_BASE_URL` and `CUCA_MODEL` point it at any OpenAI-compatible server.

```bash,name=Runs the same on all three platforms
cargo run --example vector_store --features "provider-llamacpp service-vector-store"
```

## Related pages

- [Memory](@/plugins/memory.md) declares the `VectorStore` seam and the `Offload` strategy.
- [Prompt cache](@/services/prompt-cache.md) keys on the full message list, which includes an injected recall.
- [Services](@/services/_index.md) lists every service and its plugin edge.
- [Plugins and services](@/concepts/plugin-layering.md) explains the explicit-call tier.
- [Memory discipline](@/concepts/memory-discipline.md) explains the bounded-by-default rule.
