+++
title = "Session log"
description = "Record every request, streamed block and completion in an append-only log that you can replay and fork."
template = "page.html"
weight = 12
+++

# Session log

<dl class="page-facts">
<dt>In one line</dt>
<dd><code>SessionLogPlugin</code> writes each request, streamed block and completion to an append-only log, so you can replay a session or fork it from any point.</dd>
<dt>You need</dt>
<dd>The <code>plugin-session-log</code> feature.</dd>
<dt>Read this if</dt>
<dd>You want a record of what the model saw and said, or you want to branch a conversation from an earlier point.</dd>
</dl>

Append-only means the plugin only adds records and never edits or removes one. The log can live in memory (`InMemoryBackend`) or in files on disk (`FileBackend`).

## Register it

`SessionLogPlugin::new_in_memory()` builds a plugin over a fresh in-memory log.

```rust,name=Record to memory and to disk, then fork
let memory_backend = Arc::new(InMemoryBackend::new());
let memory_log = Arc::new(
    SessionLogPlugin::new(Arc::clone(&memory_backend) as Arc<dyn SessionBackend>)
        .with_session_id("demo"),
);
let file_log = Arc::new(
    SessionLogPlugin::new(Arc::new(FileBackend::new(dir.clone())?)).with_session_id("demo"),
);
let client = CucaClient::builder()
    .with_provider(ProviderEndpoint::LlamaCpp)
    .with_base_url(base_url)
    .register_plugin(Arc::clone(&memory_log) as Arc<dyn CucaPlugin>)
    .register_plugin(Arc::clone(&file_log) as Arc<dyn CucaPlugin>)
    .build()?;

// After the turns: branch at the first Message record.
let branch = memory_log.fork_session("demo", &point)?;
```

Expected effect: both logs hold the same records for session `demo`. The file log is `{dir}/demo.cslog`. `fork_session` makes a new session that copies the records up to the point you name.

## Run the example

`examples/session_log.rs` records two turns to memory and to disk, then forks the in-memory session at its first `Message` record. It needs a `llama-server` on port 1234. Put `--base-url` and `--model` after `--` to use another OpenAI-compatible server.

```bash,name=Runs the same on all three platforms
cargo run --example session_log --features "provider-llamacpp plugin-session-log"
```

```text,name=Expected output
Session "demo", two backends recording the same turns
  memory: InMemoryBackend, cap 65536 records
  file:   /tmp/cuca-session-log-3178600/demo.cslog

Turn 1: "The vault is in Lisbon. Reply with: noted"
  reply: noted
  appended 78 records: SystemPrompt 1, Message 1, Reasoning 72, Output 2, Latency 1, TokenUsage 1
  gauge: 78/65536 records in memory, 1647 bytes on disk

Turn 2: "Where is the vault? Answer in one word."
  reply: Lisbon
  appended 135 records: SystemPrompt 1, Message 2, Reasoning 128, Output 2, Latency 1, TokenUsage 1
  gauge: 213/65536 records in memory, 4547 bytes on disk

Fork at demo:1, the first Message record
  new session: demo:fork:demo:1:0
  branch: 2 records: SystemPrompt 1, Message 1
  original tail: Fork { from_point: "demo:1", to_session: "demo:fork:demo:1:0" }
  original: 214 records, one more than before the fork

The file backend replays the same trajectory: 213 records
```

There is one `Reasoning` record per streamed `Thinking` block. Turn 2 adds a second `SystemPrompt` because only user and assistant messages are matched by position. Counts and byte sizes depend on the model. The record kinds and the fork result do not.

## Feature and entry type

| Item | Value |
|---|---|
| Feature flag | `plugin-session-log` |
| Entry types | `SessionLogPlugin`, `SessionBackend`, `InMemoryBackend`, `FileBackend` |
| Plugin name | `"session-log"` |
| Attachment | `register_plugin` |
| Also implements | `SessionStorePlugin`: `append_log`, `replay_session`, `fork_session` |
| Needs | A `SessionBackend`. `SessionLogPlugin::new_in_memory()` builds one over a fresh `InMemoryBackend`. |

## Hooks

`SessionLogPlugin` acts on `on_request`, `on_stream_chunk` and `on_response_complete`. Every hook writes to the plugin's session id.

- `on_request` adds a `SystemPrompt` for every `System` message, on every request. It adds a `Message` for a `User` or `Assistant` message only the first time that message position appears. `Tool` messages are not recorded. It records the request as earlier plugins left it.
- `on_stream_chunk` adds one record per block: `Thinking` becomes `Reasoning`, `Text` becomes `Output`, `ToolCall` becomes `ToolCall`, and `ToolResult` becomes `ToolResult` with `stdout`, `stderr` and `exit_code` set to `None`. `ImageBase64` blocks are not recorded. Blocks are recorded as they are.
- `on_response_complete` adds `Latency` (`duration_ms`) and then `TokenUsage`.
- `ModelSwap` records come only from `append_log`, which the holder of the store handle calls. The hooks never write one.

The plugin numbers every record. Backends store records as given, except for the `Fork` record a backend adds during a fork, numbered at the tail of the original session.

### Session events

| Variant | Fields |
|---|---|
| `SystemPrompt` | `text` |
| `Message` | `role`, `content` |
| `Reasoning` | `reasoning`, `signature` |
| `Output` | `text` |
| `ToolCall` | `id`, `name`, `arguments` |
| `ToolResult` | `tool_call_id`, `output`, `stdout`, `stderr`, `exit_code` |
| `ModelSwap` | `from`, `to`, `reason` |
| `Latency` | `duration_ms` |
| `TokenUsage` | `prompt_tokens`, `completion_tokens` |
| `Fork` | `from_point`, `to_session` |

Each stored `SessionRecord` has `session_id`, a `sequence` that starts at 0, a `timestamp_ms` and the `event`. `SessionRecord::point_id()` returns `"{session_id}:{sequence}"`. That string is what `fork_session` takes as `point_id`.

### Forking

`fork_session(session_id, point_id)` branches from any earlier point. The new session holds the original's records up to and including that point, relabelled with the new session id. The original gets one `SessionEvent::Fork` record.

The new id is `{session_id}:fork:{point_id}:{n}`, where `n` counts forks on that backend.

## Configuration

- `SessionLogPlugin::with_session_id(id)`: takes `impl Into<String>`, default `"default"`. The session the hooks write to.
- `InMemoryBackend::with_max_records(n)`: takes a `usize`, default `InMemoryBackend::DEFAULT_MAX_RECORDS` (65536). Limit on records across all sessions. Zero is rejected.
- `FileBackend::new(dir)`: takes `impl Into<PathBuf>`, required. Folder for the session files. Created if missing.

### Backends

- `InMemoryBackend` keeps records in memory. Nothing is saved to disk.
- `FileBackend` keeps one file per session, `{dir}/{session_id}.cslog`, with one record added per append. It never truncates or rewrites a record.

### File format

- A `.cslog` file starts with the 8-byte marker `CUCASLOG` and a one-byte format version, `1`. They are written once, when the file is created.
- A missing file replays as an empty session.
- `FileBackend` rejects a session id that contains `/` or `\`.

## Caps and at-cap policy

| Structure | Limit | At the limit | Usage gauge |
|---|---|---|---|
| `InMemoryBackend` records | `max_records`, default 65536 records across all sessions | `append` and `fork` fail with `PluginError::Internal`. Nothing is dropped, because dropping a record would break replay and forking. | `InMemoryBackend::len()` against `max_records()` |
| `FileBackend` log | Disk space | No limit inside the process. | None |
| `FileBackend` write buffer | 64 KiB kept between writes | The buffer shrinks back to 64 KiB after a larger record. No record is refused or cut. | `FileBackend::scratch_capacity()` |
| `SessionLogPlugin` bookkeeping | None: one small entry per session id | Never dropped, because dropping one would restart that session's numbering. If you create unlimited session ids, you own that bound. | None |

`fork` on `InMemoryBackend` checks the limit first. It needs room for the copied records plus the `Fork` record, and fails as a whole if there is not enough.

## Errors

`SessionLogPlugin` returns the `PluginError` its backend returns. A hook passes on an append failure. A failure in `on_response_complete` is logged by the client and does not stop the turn.

| Variant | When |
|---|---|
| `Validation`, schema `max_records` | `InMemoryBackend::with_max_records(0)` |
| `Internal` | `InMemoryBackend` is full on `append` or `fork` |
| `Validation`, schema `point_id` | `fork` names a `point_id` that no record in the session has |
| `Validation`, schema `session_id` | `FileBackend` gets a session id with `/` or `\`, including the derived fork id |
| `Validation`, schema `cslog` | A `.cslog` file is shorter than its header, lacks the `CUCASLOG` marker, has an unknown format version, or has a record that cannot be decoded |
| `Validation`, schema `cslog` | A session has a `{session_id}.jsonl` file and no `.cslog` file. It is rejected, not replayed as empty. |
| `Io` | `FileBackend::new` cannot create the folder, or a file read, write or record encode fails |

## Related pages

- [Memory discipline](@/concepts/memory-discipline.md): why the in-memory log refuses at its limit instead of dropping records.
- [Replay](@/services/replay.md): the service that reads a log back through `SessionBackend`.
- [Redaction](@/plugins/redaction.md): registration order decides whether the log stores scrubbed or raw content.
