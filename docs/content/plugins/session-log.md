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

`SessionLogPlugin` records every request, streamed block, and completion as an append-only `SessionEvent`, writing to the session named by `with_session_id` (default `"default"`). `InMemoryBackend` (capped) and `FileBackend` (COBS-framed, disk-bound) are the two storage seams, and `fork_session` branches a new session from any historical point. Reach for it to build a replayable audit trail, or to fork a conversation at a known point.

```rust,name=Record a turn and replay it back
use std::sync::Arc;

use cuca::plugin::{CucaPlugin, SessionStorePlugin};
use cuca::types::{MessageContentBlock, ProviderEndpoint};
use cuca::{CucaClient, SessionLogPlugin, UnifiedRequest};

let session_log = Arc::new(SessionLogPlugin::new_in_memory());

let client = CucaClient::builder()
    .with_provider(ProviderEndpoint::LlamaCpp)
    .with_base_url("http://127.0.0.1:1234/v1")
    .register_plugin(Arc::clone(&session_log) as Arc<dyn CucaPlugin>)
    .build()?;

let mut req = UnifiedRequest::new("google/gemma-4-e4b").add_system_message("You are concise.");
session_log.on_request(&mut req)?;
let mut chunk = MessageContentBlock::Text("hello".to_string());
session_log.on_stream_chunk(&mut chunk)?;
let records = session_log.replay_session("default")?;
```

```text,name=One SystemPrompt record plus one Output record
records.len()               2
records[0].point_id()       "default:0"
records[1].point_id()       "default:1"
```

## Entry types

`SessionLogPlugin`, `SessionBackend`, `InMemoryBackend`, `FileBackend`.

## `CucaPlugin` and `SessionStorePlugin`

`SessionLogPlugin` implements `CucaPlugin` with the plugin name `"session-log"`, and also `SessionStorePlugin` (`append_log`, `replay_session`, `fork_session`). It overrides `on_request`, `on_stream_chunk`, and `on_response_complete`, writing to the session named by `SessionLogPlugin::with_session_id` (default `"default"`).

## `SessionEvent`

`SessionEvent` variants: `SystemPrompt { text }`, `Message { role, content }`, `Reasoning { reasoning, signature }`, `Output { text }`, `ToolCall { id, name, arguments }`, `ToolResult { tool_call_id, output, stdout, stderr, exit_code }`, `ModelSwap { from, to, reason }`, `Latency { duration_ms }`, `TokenUsage { prompt_tokens, completion_tokens }`, `Fork { from_point, to_session }`.

Each stored `SessionRecord` carries `session_id`, a 0-based `sequence` assigned by the store on append, a `timestamp_ms`, and the `event`. `SessionRecord::point_id()` returns `"{session_id}:{sequence}"`, the string `fork_session` takes as `point_id`.

## Backends

| Backend | Storage | Growth |
|---|---|---|
| `InMemoryBackend` | `HashMap<session_id, Vec<SessionRecord>>` | Capped, see below |
| `FileBackend` | One file per session, `{dir}/{session_id}.cslog`, one COBS-framed postcard record per append, opened with `append(true)` | Disk-bound |

Each `.cslog` file starts with the 8-byte magic `CUCASLOG` and a one-byte format version, written once when the file is created. `replay` rejects an unknown magic or version with `PluginError::Validation`, and rejects a session whose only file is a legacy `{session_id}.jsonl` rather than replaying it as empty.

`FileBackend` rejects session ids containing `/` or `\` with `PluginError::Validation` rather than mapping them into a subdirectory.

## Capacity

| | |
|---|---|
| Bound | `InMemoryBackend::DEFAULT_MAX_RECORDS`, 65536 records in total across sessions |
| At-cap policy | `append` and `fork` fail rather than evict; this is an audit log, and dropping a record would corrupt replay and forking |
| Usage gauge | `InMemoryBackend::len()` against `max_records()` |

`FileBackend` frames every record through one reusable buffer.

| | |
|---|---|
| Bound | 64 KiB of retained scratch capacity |
| At-cap policy | The buffer is released and shrunk back to the bound after any larger record; no record is refused or truncated |
| Usage gauge | `FileBackend::scratch_capacity()` |

`SessionLogPlugin` also keeps two small per-session bookkeeping maps (next sequence, recorded message count); these are deliberately uncapped, since evicting an entry would restart that session's sequence numbering.

## Forking

`fork_session(session_id, point_id)` branches from any historical point, producing a new session whose trajectory is the prefix of the original up to and including that point. The original session gains a `SessionEvent::Fork` record for auditability.
