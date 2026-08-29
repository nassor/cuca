+++
title = "Memory discipline"
description = "Every structure that grows with traffic ships capped, and whether it evicts or refuses at the cap depends on whether the data is an optimization or a record."
template = "page.html"
weight = 5
+++

# Memory discipline

<dl class="page-facts">
<dt>In one line</dt>
<dd>A structure that grows with traffic carries a bound, a stated policy for reaching it, and a way to read current usage</dd>
<dt>You need</dt>
<dd>Nothing running</dd>
<dt>Read this if</dt>
<dd>You are adding state to a plugin, or you want to know which structures can refuse work rather than drop data</dd>
</dl>

A library that streams for hours inside somebody else's process cannot leak. So
every collection whose size is a function of traffic ships with a cap, and
nothing in the crate accumulates silently.

The interesting part is not the caps. It is that they do not all behave the same
way at the limit, and the difference is not an inconsistency.

## Evict or refuse

Two policies, chosen per structure:

| Structure | Bound | At-cap policy | Usage gauge |
|---|---|---|---|
| Guardrails attempt counters | `MAX_TRACKED_CALLS`, `4096` | evict the oldest tracked id, in insertion order | `tracked_calls()` |
| Subagent spawn log | `MAX_SPAWN_LOG`, `4096` | evict the oldest entry | length of `spawns()` |
| Prompt cache entries | `PromptCacheConfig::capacity`, caller-set, non-zero | evict the least recently used entry | `len()`, against `capacity()` |
| Subagent pending registry | `DEFAULT_MAX_PENDING`, `1024` | refuse the spawn with `PluginError::Internal` | `pending_len()` |
| Human-approval audit log | `DEFAULT_MAX_AUDIT_ENTRIES`, `65_536` | refuse the gated call with `PluginError::Internal` | `audit_len()` |
| In-memory session records | `DEFAULT_MAX_RECORDS`, `65_536` | refuse the append or fork with `PluginError::Internal` | `len()`, against `max_records()` |

The split is one question: if this entry disappears, does anything become
wrong, or only slower?

A guardrail attempt counter is a retry budget. Losing an old one means a
long-abandoned tool call gets a fresh budget it will never use. A prompt cache
entry is a saved round trip; evicting it costs a request. Both are
optimizations, and dropping the oldest is free.

An approval ruling is not an optimization. It is the record of who allowed what.
Evicting the oldest ruling would quietly erase the beginning of the audit trail,
and the log would still look healthy. So the plugin refuses the gated call
instead: a call whose ruling cannot be recorded does not proceed. Session
records are the same argument, one level down. They are append-only by design,
and replay and fork both depend on no record having gone missing, so evicting
would corrupt a fork that has not happened yet.

The subagent pending registry refuses for a third reason. Its entries are live
receivers for running child processes. Dropping the oldest receiver would not
lose a record, it would abandon a running child's result while the child keeps
consuming resources.

Refusing is the less convenient policy, and it is chosen exactly where
convenience would be dishonest.

## Per-call limits need no gauge

`plugin-sandbox` is bounded four ways at once: `max_memory_bytes` at 64 MiB,
`max_instructions` at 1000000, `timeout_ms` at 5000, and an 8 MiB cap on the
output a guest may write. All four are per call, enforced by a fresh wasmtime
`Store` on every run, and exceeding any of them traps that call and nothing
else.

Nothing accumulates across calls, so there is no cumulative reading to expose.
That is why the sandbox appears in no gauge column: a cap that resets every call
has no usage to report.

## Three structures with no cap, and why each one is allowed

The rule has exceptions, and an exception is only legitimate when its bound
comes from somewhere other than traffic.

**The working memory graph** grows only through explicit caller calls such as
`merge_graph` and `replace_graph`, never from a hook. Nothing the model streams
can make it bigger. A cap here would mean the crate silently discarding
application data it was handed on purpose, so the caller owns the bound and
decides what to drop. Separately, the graph's rendered injection into a request
is bounded, by `GraphContextConfig`, because that part is traffic-facing.

**The orchestrator's client pool** holds one client per distinct provider and
base URL a caller has configured. Its size is a function of configuration, not
of how many turns run. `ClientPool::len` reports it.

**The SSE parser's buffer** is bounded by the peer rather than by the crate. Its
high-water mark is the largest single frame plus one partial line, and the one
case that grows without limit is a server that opens a frame and never
terminates it. The reasoning is in [The SSE parser](@/concepts/sse-parser.md).

The session-log plugin's own sequence bookkeeping is a fourth, narrower case: one
entry per distinct session id, uncapped because evicting a session's counter
would corrupt the sequence numbers of records still being written to it.

## Why the order is always bound, policy, gauge

Every capped structure on this site states those three things, in that order, on
the page that owns it. The order is not stylistic. A bound alone tells a reader
nothing about what happens at the limit, and a bound plus a policy still leaves
them unable to tell how close they are. A reader who knows all three can decide
whether to raise the cap, and a reader missing the third cannot.

Where any of the three is absent from a page, that is a gap in the docs, not a
structure without one.

Next page: [Providers](@/providers/_index.md), which is where the reference half
of this site starts.
