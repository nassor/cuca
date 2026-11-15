+++
title = "Memory discipline"
description = "Every structure that grows with traffic has a cap. At the cap it evicts, refuses or folds, depending on whether its data is an optimization or a record."
template = "page.html"
weight = 5
+++

# Memory discipline

<dl class="page-facts">
<dt>In one line</dt>
<dd>A structure that grows with traffic has a bound, a stated policy at the bound, and a way to read its current use</dd>
<dt>You need</dt>
<dd>Nothing running</dd>
<dt>Read this if</dt>
<dd>You are adding state to a plugin, or you want to know which structures refuse work instead of dropping data</dd>
</dl>

A library can run for hours inside someone else's program, so it must not leak memory. Every collection in CUCA whose size depends on traffic has a cap. Each cap states what happens at the limit and how to read the current use.

At the limit, a structure does one of three things:

- **Evict**: drop the oldest or least recently used entry to make room.
- **Refuse**: reject the new work with an error.
- **Fold**: put the overflow into one shared bucket.

## Evict, refuse or fold

<div class="dgm">
<div class="dgm-scroll">
<svg viewBox="0 0 660 232" role="img" aria-labelledby="mem-title mem-desc">
  <title id="mem-title">What a capped structure does at its limit</title>
  <desc id="mem-desc">A structure that reaches its cap does one of three things. If losing an entry only costs time, it evicts the oldest or least recently used entry, as the prompt cache and the K/V block pool do. If losing an entry would lose a record or abandon live work, it refuses the new call, as the audit log and the session records do. If only the per-item detail is lost and the totals stay exact, it folds the overflow into one bucket, as the cost breakdown does.</desc>
  <text class="t-title" x="6" y="20">The cap policy follows from what the data is</text>
  <defs>
    <marker id="mem-arrow" viewBox="0 0 8 8" refX="7" refY="4"
            markerWidth="6" markerHeight="6" orient="auto">
      <path d="M0 0.6 L7 4 L0 7.4 z" fill="var(--muted-foreground)"/>
    </marker>
  </defs>
  <rect class="blk" x="6" y="97" width="150" height="64" rx="6"/>
  <text class="t-sm t-mid" x="81" y="125">If a structure</text>
  <text class="t-sm t-mid" x="81" y="139">reaches its cap</text>
  <rect class="blk blk-ctl" x="240" y="34" width="230" height="58" rx="6"/>
  <text class="t-lbl t-mid t-ctl" x="355" y="52">evict</text>
  <text class="t-sm t-mid" x="355" y="68">if losing it only costs time</text>
  <text class="t-sm t-mid" x="355" y="82">the oldest or least used goes</text>
  <rect class="blk blk-ctl" x="240" y="100" width="230" height="58" rx="6"/>
  <text class="t-lbl t-mid t-ctl" x="355" y="118">refuse</text>
  <text class="t-sm t-mid" x="355" y="134">if losing it loses a record</text>
  <text class="t-sm t-mid" x="355" y="148">or abandons live work</text>
  <rect class="blk blk-ctl" x="240" y="166" width="230" height="58" rx="6"/>
  <text class="t-lbl t-mid t-ctl" x="355" y="184">fold</text>
  <text class="t-sm t-mid" x="355" y="200">if only the detail is lost</text>
  <text class="t-sm t-mid" x="355" y="214">the totals stay exact</text>
  <path class="arw arw-ctl" d="M158 129 C196 129 196 63 236 63" marker-end="url(#mem-arrow)"/>
  <path class="arw arw-ctl" d="M158 129 H236" marker-end="url(#mem-arrow)"/>
  <path class="arw arw-ctl" d="M158 129 C196 129 196 195 236 195" marker-end="url(#mem-arrow)"/>
  <text class="t-sm" x="490" y="62">prompt cache, K/V blocks</text>
  <text class="t-sm" x="490" y="128">audit log, session records</text>
  <text class="t-sm" x="490" y="194">cost per model, kinds</text>
</svg>
</div>
<p class="dgm-cap"><b>The question is what the entry is worth.</b> A cache entry saves time, so dropping it is safe. A record is the thing itself, so CUCA refuses new work before it loses one.</p>
</div>

## Library structures

Each bullet names a capped structure in the plugins and services, its bound, what happens at the cap, and how you read its current use.

- Guardrails attempt counters: `MAX_TRACKED_CALLS`, `4096`. At the cap the oldest tracked id is evicted. Read it with `tracked_calls()`.
- Subagent spawn log: `MAX_SPAWN_LOG`, `4096`. At the cap the oldest entry is evicted. Read it with the length of `spawns()`.
- Prompt cache entries: `PromptCacheConfig::capacity`, caller set, non-zero. At the cap the least recently used entry is evicted. Read it with `len()` against `capacity()`.
- Vector store entries: `VectorStoreConfig::max_entries`, caller set, non-zero. At the cap the oldest entry is evicted (first in, first out). Read it with `usage()`.
- Subagent pending registry: `DEFAULT_MAX_PENDING`, `1024`. At the cap the spawn is refused with `PluginError::Internal`. Read it with `pending_len()`.
- Human approval audit log: `DEFAULT_MAX_AUDIT_ENTRIES`, `65_536`. At the cap the gated call is refused with `PluginError::Internal`. Read it with `audit_len()`.
- In-memory session records: `DEFAULT_MAX_RECORDS`, `65_536`. At the cap the append or fork is refused with `PluginError::Internal`. Read it with `len()` against `max_records()`.
- Replay trajectory: `ReplayConfig::max_records` `65_536` and `ReplayConfig::max_turn_blocks` `4_096`. At the cap the load is refused with `PluginError::Validation`. Read it with `usage()`.
- Rate limiter queue: `RateLimitConfig::max_waiters`, caller set, non-zero. At the cap the acquire is refused with `RateLimitError::QueueFull`. Read it with `usage()`.
- Scheduler queue: `SchedulerConfig::max_waiting`, default `256`, non-zero. At the cap the call is refused with `SchedulerError::QueueFull`. Read it with `SchedulerSnapshot`.
- Cost per-model breakdown: `CostConfig::max_tracked_models`, default `64`. At the cap extra models fold into one overflow bucket. Read it with `usage()` and `breakdown()`.
- Scheduler call kinds: `MAX_KINDS`, `32`. At the cap extra kinds fold into the kind `"other"`. Read it with `SchedulerSnapshot::totals`.

Why each one picks its policy:

- A guardrail attempt counter is a retry budget. Losing an old one only gives an abandoned tool call a fresh budget it never uses.
- A prompt cache entry is a saved round trip. Evicting it costs one request.
- A vector store entry is conversation history moved out of the context window. The store evicts the oldest entry and counts evictions in `usage()`. It never cuts an entry short.
- An approval ruling is the record of who allowed what. Evicting the oldest would erase the start of the audit trail while the log still looked healthy. So the plugin refuses the gated call, and a call whose ruling cannot be recorded does not run.
- Session records only grow, and replay and fork both need every record. Evicting one would corrupt a fork that has not happened yet.
- A replay that was cut short would be a wrong test fixture, so replay refuses.
- The subagent pending registry holds the receivers of running child processes. Dropping the oldest would lose the result of a child that is still running.
- The rate limiter and scheduler queues refuse when full. You can send the call to another backend or back off, and a silently dropped call would never get an answer.
- The cost breakdown and the scheduler's call kinds fold. The totals stay exact. Only the split by model or by kind gets less detailed. `CostUsage::overflow_turns` counts the folded turns.

Refusing is the less convenient policy. CUCA picks it where eviction would lose a record or abandon live work.

## Context window compaction

The context window is the amount of text a model can read in one request. `plugin-memory` keeps the request inside it. It starts when one of three limits is reached: `max_messages`, `max_tokens` or `max_fraction` of the context window. It then runs a list of `CompactionStrategy` steps, in order, until the request fits. Examples are `Offload` to a `VectorStore` you supply, `Summarize`, `ClearToolResults`, `ClampOversizedMessages`, `SlidingWindow` and `DropTurns`. [Memory](@/plugins/memory.md) lists all of them and their defaults.

Memory reports its use on every request. A `ContextUsageObserver` receives a `ContextUsage` reading. `MemoryConfig::warn_fraction` adds a one-time near-limit warning to the request. Alerts and export belong on the OpenTelemetry meter you give `plugin-telemetry`. No plugin installs a global provider.

## The local engine

`provider-local` allocates its K/V memory up front and bounds its queues. The K/V cache is the memory where a model keeps what it has already read, so it does not recompute it for each new token. [Limits and errors](@/providers/local/limits-and-errors.md) has the full table.

- K/V block pool: `BatchingConfig::kv_tokens` positions per full-attention group, default twice `context_length`, in blocks of 16. At the cap the least recently used unreferenced block is evicted. A running request that needs a block preempts the youngest one. Read it with `LocalModel::kv_cache_bytes()` and `serving_stats()`.
- Prefix store: the unreferenced full 16-position blocks of the same pool, least recently used first. At the cap the least recently used block is evicted, first when an allocation finds no free block. Read it with `ServingStats` (free and cached blocks, prefix hits).
- Recurrent snapshots (hybrid models): `max_sequences + 4`. At the cap the least recently used snapshot is evicted. Read it with `ServingStats`.
- Running sequences: `BatchingConfig::max_sequences`, 1 to 64, default 4. At the cap later requests wait. Read it with `ServingStats`.
- Waiting jobs: 16 behind the running ones. At the cap `generate_stream` waits for a slot. Read it with `ServingStats`.
- Output channel: 64 blocks per request. At the cap that request is skipped until its consumer reads. No usage reading.
- Step rows: `BatchingConfig::max_step_tokens`, at most 2048, default the prefill chunk. At the cap decode rows come first and prompts prefill in the rest. No usage reading.
- Tiered experts, RAM: `ExpertTiers::ram_bytes`. At the cap the coolest expert the pass no longer needs is evicted. Read it with `LocalModel::expert_tier_stats()`.
- Tiered experts, device: `ExpertTiers::device_bytes`, GPU only. Same policy as the RAM tier. Read it with `LocalModel::expert_tier_stats()`.
- Engram row cache (DeepSeek-V4.1): `EngramCache::ram_bytes`, 1 MiB to 1 TiB, default 4 GiB. A cached row counts its bytes plus 48 bytes of bookkeeping. At the cap the least recently used row is evicted, and a row evicted before its pass uses it is read again. Read it with `LocalModel::engram_stats()`.

The K/V pool never grows. Only the oldest request, alone in a pool smaller than it needs, fails with a `CucaError::Provider`. The queues make callers wait and refuse nothing, which suits one in-process caller that can wait. See [Local configuration](@/providers/local/configuration.md) and [Tiered experts](@/providers/local/expert-tiers.md).

The Engram table of a DeepSeek-V4.1 model stays in the model file. Reader threads (`EngramCache::reader_threads`, 1 to 64, default 16) fetch the distinct rows of each pass. The cache only decides how many rows are read twice.

Converting a checkpoint to GGUF is bounded too. It streams one tensor at a time. DeepSeek-V4.1 streams blocks of rows through two buffers of at most 32 MiB of `f32`. See [Model files and downloads](@/providers/local/model-sources.md).

## The server

`cuca serve` and `cuca balance` keep every structure that grows with traffic at a fixed size. The tables are on [Stats API and metrics](@/server/stats-and-metrics.md) and [Balancer](@/server/balancer.md).

- Latency window: 60 s and 4,096 records (balancer: 1,024 per upstream). At the cap the oldest record drops. Read it with `latency` in the stats snapshot.
- Sample rings: 300 one-second samples per model and for the machine. At the cap the oldest sample drops. Read it with the `StatsSnapshot` rates.
- Metrics: fixed counters. Labels are the model, status code, upstream and GPU. No series is added at run time. Read them at `/metrics`.
- Request body: 32 MiB. Above it the server answers 413 `request_too_large`.
- Chat stream buffer: 32 events per request. At the cap the producer waits for the client. Read it with `cuca_response_stall_seconds_total`.
- Stats streams: 64 open. At the cap the server answers 429 `too_many_streams`.
- Models, upstreams: 16 per server, 256 per balancer. Above that, startup fails. Read them with `models` and `upstreams`.

## Per-call limits

`plugin-sandbox` runs a WebAssembly guest, a small program in a sandbox, and has four limits. Each applies to one call:

- `max_memory_bytes`, default 64 MiB.
- `max_instructions`, default `1_000_000`.
- `timeout_ms`, default `5000`.
- An 8 MiB cap on the output a guest writes.

Each run starts with a fresh store, and a call that exceeds a limit traps and nothing else is affected. `plugin-redaction` works the same way: `max_matches_per_text`, default `1024`, caps matches in one string, and a string past it fails the hook instead of being half scrubbed.

Nothing builds up across calls, so there is no usage reading. A cap that resets on every call has no use to report.

## Structures with no cap

A structure may go uncapped only when something other than traffic sets its size.

- The working memory graph grows only through your calls: `merge_graph`, `replace_graph` and `replace_snapshot`. No hook adds to it, and nothing the model streams can enlarge it. A cap would mean CUCA silently discarding data you handed it, so you own the bound. What the graph adds to a request is capped separately by `GraphContextConfig::max_nodes` and `max_relationships`.
- The orchestrator's `ClientPool` holds one client per distinct provider and base URL you configure. Your configuration sets its size, and `ClientPool::len` reports it. If you create a new base URL for every request, you own that bound.
- The SSE parser's buffer is bounded by the server. A server that never ends a frame makes it grow. [The SSE parser](@/concepts/sse-parser.md) explains why.
- The session log plugin keeps one next-sequence number per session id. Evicting one would restart that session's numbering at zero and corrupt replay and fork. If you create unbounded session ids, you own that bound. The trajectory itself lives in the backend, which has its own cap or writes to disk through `FileBackend`.

## Bound, policy, usage

Every capped structure is described by three facts, in this order. A bound alone says nothing about what happens at the limit. A bound and a policy do not say how close the structure is to it. With all three, you can decide whether to raise the cap.

Next page: [Providers](@/providers/_index.md).
