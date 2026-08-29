+++
title = "Session log"
description = "The append-only session trajectory plugin: the SessionEvent model, the two storage backends, and forking."
template = "page.html"
weight = 12
+++

# Session log

<dl class="page-facts">
<dt>In one line</dt>
<dd>Records every request, streamed block, and completion as an append-only SessionEvent, and supports replay and forking.</dd>
<dt>You need</dt>
<dd>The <code>plugin-session-log</code> feature.</dd>
<dt>Read this if</dt>
<dd>You are registering <code>SessionLogPlugin</code>, choosing a <code>SessionBackend</code>, or forking a session.</dd>
</dl>

## Entry types

`SessionLogPlugin`, `SessionBackend`, `InMemoryBackend`, `JsonFileBackend`.

## `CucaPlugin` and `SessionStorePlugin`

`SessionLogPlugin` implements `CucaPlugin` with the plugin name `"session-log"`, and also `SessionStorePlugin` (`append_log`, `replay_session`, `fork_session`). It overrides `on_request`, `on_stream_chunk`, and `on_response_complete`, writing to the session named by `SessionLogPlugin::with_session_id` (default `"default"`).

## `SessionEvent`

`SessionEvent` variants: `SystemPrompt { text }`, `Message { role, content }`, `Reasoning { reasoning, signature }`, `Output { text }`, `ToolCall { id, name, arguments }`, `ToolResult { tool_call_id, output, stdout, stderr, exit_code }`, `ModelSwap { from, to, reason }`, `Latency { duration_ms }`, `TokenUsage { prompt_tokens, completion_tokens }`, `Fork { from_point, to_session }`.

Each stored `SessionRecord` carries `session_id`, a 0-based `sequence` assigned by the store on append, a `timestamp_ms`, and the `event`. `SessionRecord::point_id()` returns `"{session_id}:{sequence}"`, the string `fork_session` takes as `point_id`.

## Backends

| Backend | Storage | Growth |
|---|---|---|
| `InMemoryBackend` | `HashMap<session_id, Vec<SessionRecord>>` | Capped, see below |
| `JsonFileBackend` | One file per session, `{dir}/{session_id}.jsonl`, one JSON record per line, opened with `append(true)` | Disk-bound |

`JsonFileBackend` rejects session ids containing `/` or `\` with `PluginError::Validation` rather than mapping them into a subdirectory.

## Capacity

| | |
|---|---|
| Bound | `InMemoryBackend::DEFAULT_MAX_RECORDS`, 65536 records in total across sessions |
| At-cap policy | `append` and `fork` fail rather than evict; this is an audit log, and dropping a record would corrupt replay and forking |
| Usage gauge | `InMemoryBackend::len()` against `max_records()` |

`SessionLogPlugin` also keeps two small per-session bookkeeping maps (next sequence, recorded message count); these are deliberately uncapped, since evicting an entry would restart that session's sequence numbering.

## Forking

`fork_session(session_id, point_id)` branches from any historical point, producing a new session whose trajectory is the prefix of the original up to and including that point. The original session gains a `SessionEvent::Fork` record for auditability.
