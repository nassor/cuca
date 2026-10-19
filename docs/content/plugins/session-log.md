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

## Feature and entry type

| Item | Value |
|---|---|
| Feature flag | `plugin-session-log` (enables `postcard`) |
| Entry types | `SessionLogPlugin`, `SessionBackend`, `InMemoryBackend`, `FileBackend` |
| Plugin name | `"session-log"` |
| Attachment | `register_plugin` |
| Also implements | `SessionStorePlugin`: `append_log`, `replay_session`, `fork_session` |
| Needs | A `SessionBackend`; `SessionLogPlugin::new_in_memory()` builds one over a fresh `InMemoryBackend` |

## Hooks

`SessionLogPlugin` overrides `on_request`, `on_stream_chunk`, and `on_response_complete`. `execute_local_tool` uses the trait default. Every hook writes to the plugin's session id.

- `on_request` appends a `SystemPrompt` for every `System` message on every request. It appends a `Message` for a `User` or `Assistant` message only the first time that message position appears. `Tool` messages are not recorded. It records the request as earlier `on_request` hooks left it.
- `on_stream_chunk` appends one record per block: `Thinking` becomes `Reasoning`, `Text` becomes `Output`, `ToolCall` becomes `ToolCall`, and `ToolResult` becomes `ToolResult` with `stdout`, `stderr`, and `exit_code` set to `None`. `ImageBase64` blocks are not recorded. Blocks are recorded verbatim.
- `on_response_complete` appends `Latency` (`duration_ms`) and `TokenUsage`, in that order.
- `ModelSwap` records arrive through `append_log`, called by whoever holds the store handle. The plugin's own hooks never write one.

The plugin assigns every sequence number. Backends store records as given, with one exception: the `Fork` audit record a backend appends during `fork`, sequenced at the original session's tail.

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

Each stored `SessionRecord` carries `session_id`, a 0-based `sequence`, a `timestamp_ms`, and the `event`. `SessionRecord::point_id()` returns `"{session_id}:{sequence}"`, the string `fork_session` takes as `point_id`.

### Forking

`fork_session(session_id, point_id)` branches from any historical point. The new session holds the original's prefix up to and including that point, relabelled with the new session id. The original gains one `SessionEvent::Fork` record, and the plugin then realigns that session's next sequence number to its replayed length.

The new id is `{session_id}:fork:{point_id}:{n}`, where `n` counts forks on that backend instance.

## Configuration

| Key | Type | Default | Meaning |
|---|---|---|---|
| `SessionLogPlugin::with_session_id(id)` | `impl Into<String>` | `"default"` | Session the hooks write to |
| `InMemoryBackend::with_max_records(n)` | `usize` | `InMemoryBackend::DEFAULT_MAX_RECORDS` (65536) | Total-record cap across sessions; zero is rejected |
| `FileBackend::new(dir)` | `impl Into<PathBuf>` | required | Directory for the session files; created if missing |

### Backends

- `InMemoryBackend` stores a `HashMap<session_id, Vec<SessionRecord>>`. It is not persisted.
- `FileBackend` stores one file per session, `{dir}/{session_id}.cslog`, with one COBS-framed postcard record per append. It opens files with `append(true)` and never truncates or rewrites a frame.

### File format

- Each `.cslog` file starts with the 8-byte magic `CUCASLOG` and a one-byte format version, `1`, written once when the file is created.
- A missing file replays as an empty session.
- `FileBackend` rejects a session id containing `/` or `\` rather than mapping it into a subdirectory.

## Caps and at-cap policy

| Structure | Bound | At-cap policy | Usage gauge |
|---|---|---|---|
| `InMemoryBackend` records | `max_records`, default 65536 records in total across sessions | `append` and `fork` fail with `PluginError::Internal` and evict nothing, because dropping a record would corrupt replay and forking | `InMemoryBackend::len()` against `max_records()` |
| `FileBackend` trajectory | Disk | No in-process bound | None |
| `FileBackend` framing scratch | 64 KiB of retained capacity | The buffer is cleared and shrunk back to 64 KiB after any larger record; no record is refused or truncated | `FileBackend::scratch_capacity()` |
| `SessionLogPlugin` bookkeeping | None: one small entry per distinct session id | Not evicted, because evicting an entry would restart that session's sequence numbering. A caller that mints unbounded session ids owns that bound | None |

`fork` on `InMemoryBackend` checks the cap before it changes anything. It needs room for the branch prefix plus the `Fork` audit record, and fails whole when that does not fit.

## Errors

`SessionLogPlugin` returns the `PluginError` its backend returns, and a hook propagates an append failure. A failure in `on_response_complete` is logged by the client and does not abort the turn.

| Variant | When |
|---|---|
| `Validation`, schema `max_records` | `InMemoryBackend::with_max_records(0)` |
| `Internal` | `InMemoryBackend` is full on `append` or `fork` |
| `Validation`, schema `point_id` | `fork` names a `point_id` that no record in the session carries |
| `Validation`, schema `session_id` | `FileBackend` gets a session id containing `/` or `\`, including the derived fork id |
| `Validation`, schema `cslog` | A `.cslog` file is shorter than its header, lacks the `CUCASLOG` magic, has an unknown format version, or holds an undecodable frame |
| `Validation`, schema `cslog` | A session has a `{session_id}.jsonl` file and no `.cslog` file; it is rejected, not replayed as empty |
| `Io` | `FileBackend::new` cannot create the directory, or a file read, write, or frame serialization fails |

## Example

`examples/session_log.rs` records the same two live turns through two plugin instances, one over `InMemoryBackend` and one over `FileBackend`, then forks the in-memory session at its first `Message` record.

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

```bash,name=Runs the same on all three platforms
cargo run --example session_log --features "provider-llamacpp plugin-session-log"
```

It needs a `llama-server` on port 1234. `CUCA_BASE_URL` and `CUCA_MODEL` retarget it at another OpenAI-compatible server.

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

The `Reasoning` records are one per streamed `Thinking` block. Turn 2 appends a second `SystemPrompt` because only user and assistant messages are deduplicated by position. The counts and byte totals come from `google/gemma-4-12b-qat` and change with the model. The record kinds, the branch prefix, and the audit record do not.

## Related pages

- [Memory discipline](@/concepts/memory-discipline.md): why the in-memory backend refuses at its cap instead of evicting.
- [Replay](@/services/replay.md): the service that reads a trajectory back through `SessionBackend`.
- [Redaction](@/plugins/redaction.md): registration order decides whether the log stores scrubbed or raw content.
