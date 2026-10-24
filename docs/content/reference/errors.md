+++
title = "Error types"
description = "Every variant of CucaError, PluginError, PromptCacheError, RateLimitError, SchedulerError, DispatchError, CucaExportError and VectorStoreError, with its Display text and its conversions."
template = "page.html"
weight = 2
+++

# Error types

<dl class="page-facts">
<dt>In one line</dt>
<dd>Two always-present enums, <code>CucaError</code> and <code>PluginError</code>, plus one per feature that owns its own failure mode</dd>
<dt>You need</dt>
<dd>Nothing; <code>error</code> is an unconditional module</dd>
<dt>Read this if</dt>
<dd>You are matching on a failure, or reading a message and looking for its source</dd>
</dl>

Each section below gives the feature gate, the variants with their fields and `Display` text, the source, and the conversions. Every enum derives `Debug` and `Clone`; `CucaExportError` and `SchedulerError` also derive `PartialEq` and `Eq`.

## `CucaError`

Gate: none. Nine variants, none feature-gated.

| Variant | Fields | `Display` |
|---|---|---|
| `Transport` | `message: String` | `transport failure: {message}` |
| `Http` | `status: u16`, `body: String` | `HTTP error {status}: {body}` |
| `SseParse` | `String` | `SSE parse failure: {msg}` |
| `Json` | `message: String` | `JSON error: {message}` |
| `Provider` | `provider: ProviderEndpoint`, `message: String` | `provider {provider} failed: {message}` |
| `ProviderNotEnabled` | `&'static str` | `provider feature not enabled: {flag}` |
| `Plugin` | `PluginError` | `plugin error: {e}` |
| `Config` | `String` | `configuration error: {msg}` |
| `Io` | `message: String` | `I/O error: {message}` |

`CucaError::provider(endpoint, msg)` builds the `Provider` variant.

The `provider` field formats through the `Display` of `ProviderEndpoint`: `llamacpp`, `vllm`, `lmstudio`, `openai`, `anthropic`, `gemini`, `deepseek`, `local`, and the inner string for `Custom`. These differ from its serde names.

### Where each variant is raised

- `Transport`: a `reqwest` failure (connection, TLS, body read), through `From<reqwest::Error>`; also the `provider-local` model download.
- `Http`: a non-2xx response from any HTTP adapter, with the body captured; the Anthropic OAuth token exchange; the `provider-local` download, where the body is cut to 512 characters.
- `SseParse`: `SseStreamParser` meets a completed frame whose `data`, `event` or `id` bytes are not valid UTF-8.
- `Json`: `From<serde_json::Error>`; a malformed provider payload; a `PromptCacheError::Json` mapped by the client.
- `Provider`: a provider error payload (OpenAI-style `error` body, Anthropic `error` event, Gemini or llama.cpp `error` field); an Anthropic token response with no `access_token`; and every `provider-local` failure on an unsupported or malformed GGUF file, a prompt that does not fit the context, a failed forward pass or a download checksum mismatch.
- `ProviderNotEnabled`: `generate_stream` is called for an endpoint whose feature is not compiled. The payload is the flag, such as `provider-openai`.
- `Plugin`: an `on_request` hook fails, or a stream hook fails on a block.
- `Config`: see the next section.
- `Io`: `From<std::io::Error>`; a `provider-local` model file that is missing or unreadable; the local engine thread cannot start; Anthropic PKCE cannot read system randomness.

### `source()`

`source()` returns `Some` for `Plugin` only. Every other variant carries a formatted string and returns `None`.

### `Config` messages

Builder and dispatch:

| Message | Condition |
|---|---|
| `no provider selected; call with_provider before build` | `CucaClientBuilder::build` with no provider set |
| `ProviderEndpoint::Local requires CucaClientBuilder::with_local_model` | `Local` selected with no model attached, at `build` or dispatch |
| `custom endpoints require a registered adapter` | dispatching `ProviderEndpoint::Custom` |
| `anthropic requires an api key or bearer token` | Anthropic dispatch with neither configured |
| `gemini requires an api key` | Gemini dispatch with no key |
| `deepseek anthropic bridge requires an api key` | the DeepSeek bridge route with no key |
| `no prompt cache configured` | `prompt_cache_snapshot` or `replace_prompt_cache_snapshot` with no cache |
| `client pool lock poisoned` | `ClientPool::get_or_create` on a poisoned mutex |

A `PromptCacheError` other than `Json` reaches the caller as `Config` carrying that error's `Display` text.

Anthropic request building:

- `anthropic api key is not a valid header value` and `anthropic bearer token is not a valid header value` report a credential that cannot be a header.
- `invalid anthropic-beta header value: {beta}` reports a beta string that cannot be a header.
- A prompt-cache breakpoint is refused with `prompt-cache breakpoint ...` when its `message_index` or `block_index` is out of range, when it targets a standalone tool message or a non-text system block, or when two breakpoints name the same block.

`provider-local` configuration, raised by `LocalModel::load` and the local dispatch:

- `cannot resolve a model cache directory; set LocalModelConfig::cache_dir or CUCA_CACHE_DIR`.
- `invalid model {label} ...` for a repository, revision or file that is not a relative `/`-separated path of plain segments; `model path ... has no parent directory`.
- `unknown model ...: expected a preset (...) or repo:revision:file` and `unknown local device ...`.
- ``device {device:?} requires the `{feature}` cargo feature`` for a GPU device compiled out; `device ... is not available`.
- `context_length must be at least 1`, and `context_length {n} exceeds the model maximum {max}`.
- `sampling.temperature must be a finite value >= 0`, `sampling.top_p must be in (0, 1]`, and `temperature must be a finite value >= 0` for a request temperature.
- `DrafterSource::InTrunk runs on the CPU only`, and `draft_tokens must be in 1..=16` when a drafter is set.

## `PluginError`

Gate: none. Five variants, none feature-gated.

| Variant | Fields | `Display` |
|---|---|---|
| `HookFailure` | `plugin: &'static str`, `stage: &'static str`, `message: String` | `plugin {plugin} failed at stage {stage}: {message}` |
| `Validation` | `schema: String`, `message: String` | `validation failed for schema {schema}: {message}` |
| `NotSupported` | `String` | `not supported: {msg}` |
| `Io` | `String` | `plugin I/O error: {msg}` |
| `Internal` | `String` | `internal plugin error: {msg}` |

`PluginError::hook(plugin, stage, msg)` builds `HookFailure`. `impl std::error::Error for PluginError` overrides nothing, so `source()` is always `None`.

### Where each variant is raised

- `HookFailure`: `plugin-cost` refuses a request in `on_request` (stage `request`) for an unpriced model under `UnpricedModelPolicy::Reject` or when a budget cap would be crossed; `plugin-redaction` (stage `request`) refuses a string with more matches than its cap and a `ToolCall` argument nested past `RedactionConfig::MAX_JSON_DEPTH`.
- `Validation`: guardrail schema failures; malformed tool arguments (sandbox, skills, subagent, MCP); a zero cap passed to a `with_max_*` constructor; an `execute_local_tool` result whose id does not match its call; a `.cslog` file with an unknown magic, an unknown format version or a torn record frame; a session whose only file is a legacy `.jsonl`; an unknown `point_id`; a session id with a path separator; invalid redaction config; a wasm module that fails to compile.
- `NotSupported`: `McpTransport::WebSocket`; an MCP input-required or task result; `WebSearchPlugin::extract_page` on a non-Firecrawl provider; an unknown subagent id; a failed worktree preparation.
- `Io`: file and directory access in `plugin-skills`, `FileBackend` and the guardrails schema file; record framing failures in `FileBackend`; MCP child process and worker thread start; every `From<std::io::Error>` conversion.
- `Internal`: a full cap that refuses instead of evicting (HITL audit log, in-memory session log, subagent pending registry); a closed channel; a non-2xx reply or request failure inside a plugin; a panicked worker thread; a poisoned lock; invalid `CostConfig` or `MemoryConfig`; an invalid graph snapshot; a sandbox trap (fuel, memory, deadline).

## Conversions

| Impl | Result | Gate |
|---|---|---|
| `From<PluginError> for CucaError` | `CucaError::Plugin(e)` | none |
| `From<serde_json::Error> for CucaError` | `CucaError::Json { message }` | none |
| `From<std::io::Error> for CucaError` | `CucaError::Io { message }` | none |
| `From<std::io::Error> for PluginError` | `PluginError::Io(msg)` | none |
| `From<reqwest::Error> for CucaError` | `CucaError::Transport { message }` | any of the eight `provider-*` features |
| `From<RateLimitError> for PluginError` | see `RateLimitError` | `service-rate-limit` |
| `From<VectorStoreError> for PluginError` | see `VectorStoreError` | `service-vector-store` |
| `From<SchedulerError> for DispatchError` | `Config`, `QueueFull` and `InputsChanged` keep their variant | `service-scheduler` |

The `reqwest` conversion formats through `Display`, because `reqwest::Error` is not `Clone` and `CucaError` is. The other `Io`, `Json` and `Transport` conversions format the same way.

## `PromptCacheError`

Gate: `service-prompt-cache`. Four variants. It is not a `CucaError` variant and has no `From` impl into one. `CucaClient` maps it: `Json` becomes `CucaError::Json`, every other variant becomes `CucaError::Config`.

| Variant | Fields | `Display` | Condition |
|---|---|---|---|
| `Config` | `String` | `prompt cache configuration error: {msg}` | zero `capacity` or zero `ttl` in `PromptCacheConfig::new` |
| `Validation` | `field: String`, `message: String` | `prompt cache validation failed for {field}: {message}` | a malformed snapshot entry (key shape, digest mismatch, timestamp order, duplicate key or rank, non-contiguous ranks), or a non-finite temperature |
| `Json` | `String` | `prompt cache JSON error: {msg}` | encoding a request for its digest failed |
| `Lock` | `String` | `prompt cache lock error: {msg}` | a poisoned state mutex |

## `RateLimitError`

Gate: `service-rate-limit`. Six variants. `From<RateLimitError> for PluginError` maps `Config` to `Validation` under schema `rate-limit-config`, returns an `Observer` error unchanged, and maps the rest to `Internal` with the `Display` text.

| Variant | Fields | `Display` | Condition |
|---|---|---|---|
| `Config` | `String` | `rate limit configuration error: {msg}` | a zero rate, interval, burst, concurrency, queue bound or wait budget; a `warn_fraction` outside `(0.0, 1.0]`; `max_concurrent` above the semaphore maximum |
| `QueueFull` | `waiters: usize`, `max_waiters: usize` | `rate limit queue is full: {waiters} of {max_waiters} waiters are already parked` | `acquire` with the waiter queue at `max_waiters` |
| `Busy` | `in_flight: usize`, `max_concurrent: usize` | `rate limit concurrency cap reached: {in_flight} of {max_concurrent} slots in flight` | `try_acquire` with no free slot |
| `Exhausted` | `retry_after: Duration` | `rate limit token bucket is exhausted; retry after {ms} ms` | no token within the wait budget, or an empty bucket in `try_acquire` |
| `Observer` | `PluginError` | `rate limit observer refused the acquire: {err}` | a `RateLimitObserver` returned an error; the slot is released |
| `Lock` | `String` | `rate limit lock error: {msg}` | a poisoned bucket mutex |

## `SchedulerError`

Gate: `service-scheduler`. Three variants. It is not a `CucaError` variant and has no `From` impl into one. It implements `std::error::Error` with no `source()`.

| Variant | Fields | `Display` | Condition |
|---|---|---|---|
| `Config` | `String` | `scheduler config: {message}` | a `ceiling` of 0 or above 64, a zero `max_waiting`, or a zero `reserved_lane_from_level` |
| `QueueFull` | `backend: BackendId`, `max_waiting: usize` | `backend {backend} already has {max_waiting} calls waiting` | `Scheduler::enqueue` with the queue at `max_waiting`; the call is not queued |
| `InputsChanged` | none | `the call's inputs changed while it waited` | the `DispatchGate` reported the waiting call's task stale; the call left the queue unsent |

## `DispatchError`

Gate: `service-scheduler`. Eight variants. It is not a `CucaError` variant and has no `From` impl into one. `From<SchedulerError> for DispatchError` maps `Config`, `QueueFull` and `InputsChanged` to the variant of the same name. `source()` returns the inner error for `Call` and `None` for the rest.

| Variant | Fields | `Display` | Condition |
|---|---|---|---|
| `Config` | `String` | `dispatcher configuration error: {message}` | an invalid `SchedulerConfig`, a zero `unreachable_cooldown`, a backend listed twice in one list, or more than 64 backends in one registry |
| `QueueFull` | `backend: BackendId` | `the queue of backend {backend} is full` | the chosen backend already holds `max_waiting` calls |
| `Unreachable` | `detail: String` | `no backend is reachable: {detail}` | every admitting backend failed to connect, a lazy client failed to build, every breaker is open, or the list is empty |
| `Timeout` | `seconds: u64` | `the call timed out after {seconds}s` | the call ran past `DispatcherConfig::timeout` |
| `Empty` | none | `the backend returned no text` | the stream ended with no text and no reasoning |
| `ReasoningOnly` | `reasoning_chars: usize` | `the backend returned only reasoning ({reasoning_chars} characters) and no text` | the stream held reasoning and no text |
| `InputsChanged` | none | `the call's inputs changed while it waited` | the `DispatchGate` reported the task stale |
| `Call` | `CucaError` | `the backend call failed: {error}` | any other provider error, or an error item in the middle of the stream; no failover |

## `CucaExportError`

Gate: `plugin-memory` or `service-prompt-cache`. Four variants.

| Variant | Fields | `Display` | Condition |
|---|---|---|---|
| `Json` | `message: String` | `cuca-export JSON error: {message}` | malformed JSON, a missing or unknown field, or a wrong type |
| `Validation` | `component: &'static str`, `field: &'static str`, `message: String` | `cuca-export validation failed for {component}.{field}: {message}` | a section failed its own check; `component` is `envelope`, `graph` or `prompt_cache` |
| `Unsupported` | `component: &'static str` | `cuca-export carries {component} data, but this build compiled that component out` | the document carries data for a compiled-out component (`graph` or `prompt_cache`) |
| `State` | `message: String` | `cuca-export state error: {message}` | live state could not be read or replaced, such as a poisoned lock |

## `VectorStoreError`

Gate: `service-vector-store`. Six variants. It is not a `CucaError` variant. `VectorStore::store_turns` must return `PluginError`, so `From<VectorStoreError> for PluginError` converts: `Embedder(e)` becomes `e` unchanged, `Embedding` and `EntryTooLarge` become `PluginError::Validation` under schema `vector-store`, and every other variant becomes `PluginError::Internal` with the `Display` text.

| Variant | Fields | `Display` | Condition |
|---|---|---|---|
| `Config` | `String` | `vector store configuration error: {msg}` | a zero bound, a `warn_fraction` outside `(0.0, 1.0]`, an arena that overflows `usize`, or a zero `k` |
| `Embedding` | `String` | `vector store embedding error: {msg}` | a wrong-width embedding, or a non-finite component or squared norm |
| `EntryTooLarge` | `bytes: usize`, `max_entry_bytes: usize` | `vector store turn of {bytes} bytes exceeds max_entry_bytes {max_entry_bytes}` | a turn over `VectorStoreConfig::max_entry_bytes`; rejected, never truncated |
| `Poisoned` | none | `vector store state lock is poisoned` | the state lock was poisoned by a panic on another thread |
| `Json` | `String` | `vector store JSON error: {msg}` | measuring a tool call's arguments failed; unreachable for a `serde_json::Value` |
| `Embedder` | `PluginError` | `vector store embedder failed: {err}` | the caller's `Embedder` failed; carried through unchanged |

## Errors the stream swallows

- An error from an `on_response_complete` hook is logged and dropped, because the response has already been delivered. The log is a `tracing::error!` event at target `cuca::client` with the plugin name, and it exists only when `plugin-guardrails` is enabled. Without that feature the error is dropped silently.
- A prompt-cache write after a successful stream is advisory. Its failure never fails the turn.
