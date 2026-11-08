+++
title = "Memory discipline"
description = "Every structure that grows with traffic ships capped, and whether it evicts or refuses at the cap depends on whether the data is an optimization or a record."
template = "page.html"
weight = 5
+++

# Memory discipline

<dl class="page-facts">
<dt>In one line</dt>
<dd>A structure that grows with traffic carries a bound, a stated policy at the bound, and a way to read current usage</dd>
<dt>You need</dt>
<dd>Nothing running</dd>
<dt>Read this if</dt>
<dd>You are adding state to a plugin, or you want to know which structures refuse work instead of dropping data</dd>
</dl>

A library that streams for hours inside another program's process cannot leak. Every collection whose size depends on traffic ships with a cap, and each cap has a stated policy and a usage reading.

The policies differ. Some structures evict the oldest entry, some refuse the call, and one folds overflow into a single bucket. The difference follows from what the data is.

## Evict or refuse

| Structure | Bound | At the cap | Usage reading |
|---|---|---|---|
| Guardrails attempt counters | `MAX_TRACKED_CALLS`, `4096` | evict the oldest tracked id | `tracked_calls()` |
| Subagent spawn log | `MAX_SPAWN_LOG`, `4096` | evict the oldest entry | length of `spawns()` |
| Prompt cache entries | `PromptCacheConfig::capacity`, caller set, non-zero | evict the least recently used entry | `len()` against `capacity()` |
| Vector store entries | `VectorStoreConfig::max_entries`, caller set, non-zero | evict the oldest entry (FIFO) | `usage()` |
| Subagent pending registry | `DEFAULT_MAX_PENDING`, `1024` | refuse the spawn with `PluginError::Internal` | `pending_len()` |
| Human approval audit log | `DEFAULT_MAX_AUDIT_ENTRIES`, `65_536` | refuse the gated call with `PluginError::Internal` | `audit_len()` |
| In-memory session records | `DEFAULT_MAX_RECORDS`, `65_536` | refuse the append or fork with `PluginError::Internal` | `len()` against `max_records()` |
| Replay trajectory | `ReplayConfig::max_records` `65_536`<br>`ReplayConfig::max_turn_blocks` `4_096` | refuse the load with `PluginError::Validation` | `usage()` |
| Rate limiter queue | `RateLimitConfig::max_waiters`, caller set, non-zero | refuse the acquire with `RateLimitError::QueueFull` | `usage()` |
| Scheduler queue | `SchedulerConfig::max_waiting`, default `256`, non-zero | refuse the call with `SchedulerError::QueueFull` | `SchedulerSnapshot` |
| Cost per-model breakdown | `CostConfig::max_tracked_models`, default `64` | fold into one overflow bucket | `usage()`, `breakdown()` |
| Scheduler call kinds | `MAX_KINDS`, `32` | fold into the kind `"other"` | `SchedulerSnapshot::totals` |

The split is one question: if this entry disappears, does anything become wrong, or only slower?

- A guardrail attempt counter is a retry budget. Losing an old one gives a long-abandoned tool call a fresh budget it never uses.
- A prompt cache entry is a saved round trip. Evicting it costs one request.
- A vector store entry is conversation history offloaded from the context window. The store evicts the oldest entry at the cap and counts evictions in `usage()`. It never truncates an entry.
- An approval ruling is the record of who allowed what. Evicting the oldest would erase the start of the audit trail while the log still looked healthy, so the plugin refuses the gated call. A call whose ruling cannot be recorded does not proceed.
- Session records are append-only, and replay and fork both need every record. Evicting one would corrupt a fork that has not happened yet.
- A replayed trajectory that was truncated would be a wrong fixture, so replay refuses.
- The subagent pending registry holds live receivers for running child processes. Dropping the oldest would abandon a running child's result while it keeps consuming resources.
- A rate limiter queue refuses at `max_waiters` because queueing past the bound would grow the semaphore's own waiter list.
- A scheduler queue refuses at `max_waiting` because the caller can route the call to another backend or back off, and a silently dropped call would never answer.
- The cost breakdown folds because totals stay exact. Only per-model attribution degrades, and `CostUsage::overflow_turns` counts the folded turns.
- The scheduler's kind totals fold the 33rd and later kinds into `"other"`, so the call counts stay exact and only the per-kind attribution degrades.

Refusing is the less convenient policy. The design chooses it where eviction would lose a record or abandon live work.

## Context window compaction

`plugin-memory` bounds the request itself. A size trigger fires on one of three knobs: `max_messages`, `max_tokens` or `max_fraction` of the context window. It then runs an ordered list of `CompactionStrategy` values until the request fits. Examples are `Offload` to a caller-supplied `VectorStore`, `Summarize`, `ClearToolResults`, `ClampOversizedMessages`, `SlidingWindow` and `DropTurns`; [Memory](@/plugins/memory.md) lists the full set and its defaults.

Memory reports usage on every request. A `ContextUsageObserver` receives a `ContextUsage` reading, and `MemoryConfig::warn_fraction` injects a one-shot near-limit warning, idempotent through a marker scan. Alerting and export belong on the caller-supplied OpenTelemetry meter of `plugin-telemetry`. No plugin installs a global provider.

## The local engine

`provider-local` preallocates its K/V memory and bounds its queues. Each cap below has a policy at the bound and a usage reading; [Limits and errors](@/providers/local/limits-and-errors.md) holds the full table.

| Structure | Bound | At the cap | Usage reading |
|---|---|---|---|
| K/V block pool | `BatchingConfig::kv_tokens` positions per full-attention group, default twice `context_length`, blocks of 16 | the least recently used unreferenced block is evicted; a running request that needs a block preempts the youngest one | `LocalModel::kv_cache_bytes()`, `serving_stats()` |
| Prefix store | the unreferenced full 16-position blocks of the same pool, an LRU | evict the least recently used block, first when an allocation finds no free block | `ServingStats` (free and cached blocks, prefix hits) |
| Recurrent snapshots (hybrid models) | `max_sequences + 4` | evict the least recently used snapshot | `ServingStats` |
| Running sequences | `BatchingConfig::max_sequences`, 1 to 64, default 4 | later requests wait | `ServingStats` |
| Waiting jobs | 16 behind the running ones | `generate_stream` waits for a slot | `ServingStats` |
| Output channel | 64 blocks per request | that request is skipped until its consumer reads | none |
| Step rows | `BatchingConfig::max_step_tokens`, at most 2048, default the prefill chunk | decode rows come first, prompts prefill in the rest | none |
| Tiered experts, RAM | `ExpertTiers::ram_bytes` | evict the coolest expert the pass no longer needs | `LocalModel::expert_tier_stats()` |
| Tiered experts, device | `ExpertTiers::device_bytes`, GPU only | same policy | `LocalModel::expert_tier_stats()` |

The K/V pool never grows. Only the oldest request alone in a pool smaller than it needs fails, with a `CucaError::Provider`. The queues apply backpressure and refuse nothing, which suits a single in-process caller that can wait. See [Local configuration](@/providers/local/configuration.md) and [Expert tiers](@/providers/local/expert-tiers.md).

## The server

`cuca serve` and `cuca balance` keep every traffic-driven structure fixed in size. The tables are on [Stats API and metrics](@/server/stats-and-metrics.md#bounds) and [Balancer](@/server/balancer.md#bounds).

| Structure | Bound | At the cap | Usage reading |
|---|---|---|---|
| Latency window | 60 s and 4,096 records (balancer: 1,024 per upstream) | the oldest record drops | `latency` in the stats snapshot |
| Sample rings | 300 one-second samples per model and for the machine | the oldest sample drops | `StatsSnapshot` rates |
| Metrics | fixed atomics; labels are the model, status code, upstream and GPU | no series is added at run time | `/metrics` |
| Request body | 32 MiB | 413 `request_too_large` | none |
| Chat stream buffer | 32 events per request | the producer waits for the client | `cuca_response_stall_seconds_total` |
| Stats streams | 64 open | 429 `too_many_streams` | none |
| Models, upstreams | 16 per server, 256 per balancer | startup fails | `models`, `upstreams` |

## Per-call limits need no usage reading

`plugin-sandbox` has four limits, all per call:

- `max_memory_bytes`, default 64 MiB.
- `max_instructions`, default `1_000_000`.
- `timeout_ms`, default `5000`.
- An 8 MiB cap on the output a guest writes.

Each run builds a fresh wasmtime `Store`, and exceeding a limit traps that call and nothing else. `plugin-redaction` has the same shape: `max_matches_per_text`, default `1024`, caps matches in one string, and a string past it fails the hook instead of being partly scrubbed.

Nothing accumulates across calls, so there is no cumulative reading. A cap that resets every call has no usage to report.

## Structures with no cap

An uncapped structure is legitimate only when something other than traffic bounds it.

- The working memory graph grows only through explicit caller calls: `merge_graph`, `replace_graph` and `replace_snapshot`. No hook adds to it, and nothing the model streams can enlarge it. A cap would mean the crate silently discarding data the application handed it, so the caller owns the bound. What the graph injects into a request is capped separately by `GraphContextConfig::max_nodes` and `max_relationships`.
- The orchestrator's `ClientPool` holds one client per distinct provider and base URL a caller configures. Configuration sets its size, and `ClientPool::len` reports it. A caller that mints a base URL per request owns that bound.
- The SSE parser's buffer is bounded by the peer. Its high-water mark is the largest single frame plus one partial line, and a server that never ends a frame makes it grow. [The SSE parser](@/concepts/sse-parser.md) gives the reasoning.
- The session log plugin's next-sequence map holds one entry per distinct session id. Evicting an entry would restart that session's numbering at zero and corrupt replay and fork, so a caller that mints unbounded session ids owns that bound. The trajectory itself lives in the backend, which has its own cap or writes to disk through `FileBackend`.

## Bound, policy, usage

Each capped structure is described by those three facts, in that order. A bound alone says nothing about what happens at the limit. A bound and a policy still do not say how close the structure is to it. A reader who has all three can decide whether to raise the cap.

Next page: [Providers](@/providers/_index.md).
