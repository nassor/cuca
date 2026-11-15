+++
title = "Error types"
description = "Every variant of CucaError, PluginError and the per-feature error enums: when it happens, what the message looks like and what to do."
template = "page.html"
weight = 2
+++

# Error types

<dl class="page-facts">
<dt>In one line</dt>
<dd>Two error enums are always present, <code>CucaError</code> (the client) and <code>PluginError</code> (plugin hooks); six more belong to features</dd>
<dt>You need</dt>
<dd>Nothing; the <code>error</code> module is always compiled</dd>
<dt>Read this if</dt>
<dd>You are matching on a failure, or you have an error message and want to know what caused it</dd>
</dl>

Every enum derives `Debug` and `Clone`. `CucaExportError` and `SchedulerError` also derive `PartialEq` and `Eq`. Most errors carry text, not the original error, so they can be cloned and sent through a response stream.

## Which error do I get

<div class="dgm">
<div class="dgm-scroll">
<svg viewBox="0 0 660 190" role="img" aria-labelledby="errx-title errx-desc">
  <title id="errx-title">Where each error type surfaces.</title>
  <desc id="errx-desc">A call to CucaClient returns CucaError. A plugin hook returns PluginError, which the client wraps as CucaError::Plugin. Feature errors such as PromptCacheError and RateLimitError come from their own services; some convert to PluginError or CucaError.</desc>
  <defs>
    <marker id="errx-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M0 0.6 L7 4 L0 7.4 z" fill="var(--muted-foreground)"/>
    </marker>
  </defs>
  <text class="t-title" x="6" y="20">What your code receives</text>
  <rect class="blk blk-ctl" x="6" y="44" width="170" height="56" rx="6"/>
  <text class="t-lbl t-ctl" x="18" y="66">PluginError</text>
  <text class="t-sm" x="18" y="84">from a plugin hook</text>
  <rect class="blk blk-data" x="246" y="44" width="170" height="56" rx="6"/>
  <text class="t-lbl t-data" x="258" y="66">CucaError</text>
  <text class="t-sm" x="258" y="84">from CucaClient</text>
  <rect class="blk" x="476" y="44" width="178" height="56" rx="6"/>
  <text class="t-lbl" x="488" y="66">your code</text>
  <text class="t-sm" x="488" y="84">match on the variant</text>
  <path class="arw" d="M178 72 H244" marker-end="url(#errx-arrow)"/>
  <path class="arw" d="M418 72 H474" marker-end="url(#errx-arrow)"/>
  <text class="t-sm t-mid" x="211" y="64">Plugin(e)</text>
  <rect class="blk" x="6" y="128" width="410" height="48" rx="6"/>
  <text class="t-lbl" x="18" y="148">PromptCacheError, RateLimitError, VectorStoreError</text>
  <text class="t-sm" x="18" y="164">SchedulerError, DispatchError, CucaExportError</text>
  <path class="arw" d="M330 126 V102" marker-end="url(#errx-arrow)"/>
</svg>
</div>
<p class="dgm-cap"><b>Client calls return <code>CucaError</code>.</b> A failing plugin hook becomes <code>CucaError::Plugin</code>. Service errors are returned by the service you call. The client turns a prompt cache error into <code>CucaError::Config</code> or <code>CucaError::Json</code>.</p>
</div>

## `CucaError`

Nine variants, none behind a feature. `CucaError::provider(endpoint, msg)` builds a `Provider` error.

- **`Transport { message }`**: `transport failure: {message}`.
  - Happens: a connection, TLS or body-read failure; the `provider-local` model download, including a stall of 60 seconds.
  - Do: check the base URL, the network and that the server is running.
- **`Http { status, body }`**: `HTTP error {status}: {body}`.
  - Happens: any adapter received a non-2xx response; the Anthropic OAuth token exchange; the model download (the body is cut to 512 characters).
  - Do: read `status` and `body`. 401 and 403 usually mean a wrong key; 404 a wrong base URL or model id; 429 a rate limit.
- **`SseParse(msg)`**: `SSE parse failure: {msg}`.
  - Happens: a frame has `data`, `event` or `id` bytes that are not valid UTF-8.
  - Do: the server is not sending a valid event stream. Check the base URL points at the right route.
- **`Json { message }`**: `JSON error: {message}`.
  - Happens: a response payload is not the JSON the adapter expects; a prompt cache JSON error.
  - Do: check that the `ProviderEndpoint` and the base URL match the server you call.
- **`Provider { provider, message }`**: `provider {provider} failed: {message}`.
  - Happens: the provider reported an error (an OpenAI-style `error` body, an Anthropic `error` event, a Gemini or llama.cpp `error` field); an Anthropic token response with no `access_token`; and most `provider-local` failures (see below).
  - Do: read `message`. It is the provider's own text.
- **`ProviderNotEnabled(flag)`**: `provider feature not enabled: {flag}`.
  - Happens: the request targets an endpoint whose feature is not compiled. `flag` is the feature, for example `provider-openai`.
  - Do: enable that feature.
- **`Plugin(PluginError)`**: `plugin error: {e}`.
  - Happens: an `on_request` hook fails, or a stream hook fails on a block.
  - Do: read the inner `PluginError` below.
- **`Config(msg)`**: `configuration error: {msg}`.
  - Happens: invalid builder or configuration state, listed in the next sections.
  - Do: fix the setting the message names. The request never reached the network.
- **`Io { message }`**: `I/O error: {message}`.
  - Happens: a file or subprocess failure; a model file or chat template file that is missing or unreadable; the local engine thread cannot start; Anthropic PKCE cannot read system randomness.
  - Do: check the path and its permissions.

The `provider` field prints as `llamacpp`, `vllm`, `lmstudio`, `openai`, `anthropic`, `gemini`, `deepseek`, `local`, or the inner text of `Custom`.

Which `provider-local` failures are `Provider` errors: an unsupported or malformed GGUF file or safetensors checkpoint; a request the engine cannot take (an `ImageBase64` block, tool arguments that are not a JSON object, a chat template that does not render); a prompt that does not fit the context; a failed forward pass; a drafter that does not match its target; a download checksum mismatch. [Limits and errors](@/local/limits-and-errors.md) lists each cause and fix.

`source()` returns `Some` for `Plugin` only. Every other variant returns `None`.

## Config messages

### Builder and dispatch

- `no provider selected; call with_provider before build`: `CucaClientBuilder::build` with no provider. Call `with_provider`.
- `ProviderEndpoint::Local requires CucaClientBuilder::with_local_model`: `Local` selected with no model attached. Call `with_local_model`.
- `custom endpoints require a registered adapter`: dispatch to `ProviderEndpoint::Custom`. Use a built-in endpoint.
- `anthropic requires an api key or bearer token`: Anthropic dispatch with neither. Call `with_api_key` or `with_bearer_token`.
- `gemini requires an api key`: Gemini dispatch with no key. Call `with_api_key`.
- `deepseek anthropic bridge requires an api key`: the DeepSeek bridge route with no key. Call `with_api_key`.
- `no prompt cache configured`: `prompt_cache_snapshot` or `replace_prompt_cache_snapshot` with no cache. Configure a prompt cache first.
- `client pool lock poisoned`: `ClientPool::get_or_create` after a thread panicked. Recreate the pool.

A `PromptCacheError` other than `Json` reaches the caller as `Config` with that error's text.

### Request fields a backend cannot honor

The check runs when the request body is built, before any network call.

| Message | Cause |
|---|---|
| `` `top_k` is not supported by OpenAI `` | `top_k` is set on the OpenAI endpoint |
| `` `top_k` is not supported by DeepSeek `` | `top_k` is set on DeepSeek's native route |
| `` `top_k` is not supported by this endpoint `` | `top_k` is set on another endpoint that uses the OpenAI-compatible builder, such as `Custom` |
| `` `seed` is not supported by Anthropic `` | `seed` is set on Anthropic or on DeepSeek's Anthropic bridge |

`top_k` works on llama.cpp, vLLM, LM Studio, Anthropic, Gemini and the DeepSeek Anthropic bridge. `seed` works everywhere else. `top_p` and `stop` work everywhere.

### Anthropic request building

- `anthropic api key is not a valid header value` and `anthropic bearer token is not a valid header value`: the credential contains characters a header cannot hold.
- `invalid anthropic-beta header value: {beta}`: the beta string cannot be a header.
- `prompt-cache breakpoint ...`: a breakpoint `message_index` or `block_index` is out of range, targets a standalone tool message or a non-text system block, or two breakpoints name the same block.

### Local model settings

`LocalModel::load` and the local dispatch check the model settings and report the first problem. The messages are in [Limits and errors](@/local/limits-and-errors.md); the valid ranges are in [Configuration](@/local/configuration.md). The groups are:

- the cache directory cannot be found: set `LocalModelConfig::cache_dir`.
- an unknown model, preset or device string; a GPU device whose Cargo feature is off (`device ... requires the ... cargo feature`) or that is not present.
- `context_length` below 1 or above the model maximum; `sampling.temperature` or `sampling.top_p` out of range, in the configuration or in a request.
- `shards` (tensor parallelism), `gpu_layers`, `batching`, `drafter`, `draft_tokens`, `experts` and `engram` values out of range or combined in a way that is not allowed.
- a split GGUF whose parts are not named or ordered as listed; a chat template file that is not UTF-8.
- a DeepSeek-V4.1 conversion whose output disk is too small.

### KDL settings files

With `config-kdl`, the KDL reader raises `Config` for a settings file. A message about a file starts with its path, for example ``reading `cuca.kdl`: ...``.

| Message starts with | Cause | Fix |
|---|---|---|
| `parsing KDL: <line>:<column>: ...` | the text is not valid KDL, repeats a property, or uses `#null` or a type annotation | fix the line it names |
| `configuration: ...` | the document does not match the settings; an unknown key is named | fix or remove the key |
| `variables block must be name/string pairs: ...` | a `variables` entry is not a string, or its name has characters outside `A-Z`, `a-z`, `0-9`, `_` | rename or quote it |
| a `${NAME}` message | the name is neither in `variables` nor in the environment and has no default; a `${` is not closed; two variables refer to each other | declare it, set it, or write `${NAME:-default}` |
| `` expected a byte count such as `4294967296`, `512m` or `4g` `` | a size value is malformed | write digits with an optional `k`, `m` or `g` |
| `` expected a duration such as `500ms`, `60s`, `5m` or `1h` `` | a duration value is malformed | write digits with `ms`, `s`, `m` or `h` |

An unreadable `LocalModelConfig::chat_template_file` raises `CucaError::Io` with ``reading chat template `{path}`: {error}``.

## `PluginError`

Five variants, none behind a feature. `PluginError::hook(plugin, stage, msg)` builds `HookFailure`. `source()` is always `None`.

- **`HookFailure { plugin, stage, message }`**: `plugin {plugin} failed at stage {stage}: {message}`.
  - Happens: `plugin-cost` refuses a request (stage `request`) for an unpriced model under `UnpricedModelPolicy::Reject`, or when a budget cap would be crossed; `plugin-redaction` refuses a string with more matches than its cap, or a `ToolCall` argument nested deeper than `RedactionConfig::MAX_JSON_DEPTH`.
  - Do: raise the budget, add a price for the model, or shorten the input.
- **`Validation { schema, message }`**: `validation failed for schema {schema}: {message}`.
  - Happens: a guardrail schema check fails; tool arguments are malformed (sandbox, skills, subagent, MCP); a `with_max_*` constructor gets a zero cap; an `execute_local_tool` result does not match its call id; a `.cslog` file has an unknown magic, an unknown version or a torn record; a session has only a legacy `.jsonl` file; a `point_id` is unknown; a session id contains a path separator; the redaction configuration is invalid; a wasm module does not compile.
  - Do: fix the input the `schema` names. The model's bad output is sent back to it in the stream.
- **`NotSupported(msg)`**: `not supported: {msg}`.
  - Happens: `McpTransport::WebSocket`; an MCP input-required or task result; `WebSearchPlugin::extract_page` on a provider other than Firecrawl; an unknown subagent id; a failed worktree preparation.
  - Do: use a supported transport or provider.
- **`Io(msg)`**: `plugin I/O error: {msg}`.
  - Happens: file or directory access in `plugin-skills`, `FileBackend` or the guardrails schema file; a record framing failure in `FileBackend`; the MCP child process or worker thread cannot start.
  - Do: check paths and permissions.
- **`Internal(msg)`**: `internal plugin error: {msg}`.
  - Happens: a full cap that refuses instead of evicting (HITL audit log, in-memory session log, subagent pending registry); a closed channel; a non-2xx reply or request failure inside a plugin; a panicked worker thread or poisoned lock; an invalid `CostConfig`, `MemoryConfig` or graph snapshot; a sandbox trap (fuel, memory or deadline).
  - Do: read the message; it names the cap or the limit.

## Conversions

| Impl | Result | Needs |
|---|---|---|
| `From<PluginError> for CucaError` | `CucaError::Plugin(e)` | nothing |
| `From<serde_json::Error> for CucaError` | `CucaError::Json` | nothing |
| `From<std::io::Error> for CucaError` | `CucaError::Io` | nothing |
| `From<std::io::Error> for PluginError` | `PluginError::Io` | nothing |
| `From<reqwest::Error> for CucaError` | `CucaError::Transport` | any `provider-*` feature |
| `From<RateLimitError> for PluginError` | see `RateLimitError` | `service-rate-limit` |
| `From<VectorStoreError> for PluginError` | see `VectorStoreError` | `service-vector-store` |
| `From<SchedulerError> for DispatchError` | `Config`, `QueueFull` and `InputsChanged` keep their name | `service-scheduler` |

## `PromptCacheError`

Needs `service-prompt-cache`. It is not a `CucaError` variant. The client maps `Json` to `CucaError::Json` and every other variant to `CucaError::Config`.

- `Config(msg)`, displays `prompt cache configuration error: {msg}`: `PromptCacheConfig::new` gets a zero `capacity` or `ttl`.
- `Validation { field, message }`, displays `prompt cache validation failed for {field}: {message}`: a snapshot entry is malformed (key shape, digest mismatch, timestamp order, duplicate key or rank, ranks not contiguous), or a temperature is not finite.
- `Json(msg)`, displays `prompt cache JSON error: {msg}`: encoding a request for its digest failed.
- `Lock(msg)`, displays `prompt cache lock error: {msg}`: a thread panicked while holding the state lock.

## `RateLimitError`

Needs `service-rate-limit`. `From<RateLimitError> for PluginError` maps `Config` to `Validation` (schema `rate-limit-config`), returns an `Observer` error unchanged and maps the rest to `Internal`.

- `Config(msg)`, displays `rate limit configuration error: {msg}`: a rate, interval, burst, concurrency, queue bound or wait budget is zero; `warn_fraction` is outside `(0.0, 1.0]`; `max_concurrent` is above the semaphore maximum.
- `QueueFull { waiters, max_waiters }`, displays `rate limit queue is full: {waiters} of {max_waiters} waiters are already parked`: `acquire` while the waiter queue is full.
- `Busy { in_flight, max_concurrent }`, displays `rate limit concurrency cap reached: {in_flight} of {max_concurrent} slots in flight`: `try_acquire` with no free slot.
- `Exhausted { retry_after }`, displays `rate limit token bucket is exhausted; retry after {ms} ms`: no token within the wait budget, or an empty bucket in `try_acquire`.
- `Observer(PluginError)`, displays `rate limit observer refused the acquire: {err}`: a `RateLimitObserver` returned an error; the slot is released.
- `Lock(msg)`, displays `rate limit lock error: {msg}`: a thread panicked while holding the bucket lock.

## `SchedulerError`

Needs `service-scheduler`. It is not a `CucaError` variant. It implements `std::error::Error` with no `source()`.

- `Config(message)`, displays `scheduler config: {message}`: `ceiling` is 0 or above 64, `max_waiting` is 0, or `reserved_lane_from_level` is 0.
- `QueueFull { backend, max_waiting }`, displays `backend {backend} already has {max_waiting} calls waiting`: `Scheduler::enqueue` with the queue full; the call is not queued.
- `InputsChanged`, displays `the call's inputs changed while it waited`: the `DispatchGate` reported the waiting call stale; it left the queue unsent.

## `DispatchError`

Needs `service-scheduler`. It is not a `CucaError` variant. `source()` returns the inner error for `Call` and `None` for the rest.

- `Config(message)`, displays `dispatcher configuration error: {message}`: an invalid `SchedulerConfig`, a zero `unreachable_cooldown`, a backend listed twice in one list, or more than 64 backends in one registry.
- `QueueFull { backend }`, displays `the queue of backend {backend} is full`: the chosen backend already holds `max_waiting` calls.
- `Unreachable { detail }`, displays `no backend is reachable: {detail}`: every admitting backend failed to connect, a lazy client failed to build, every breaker is open, or the list is empty.
- `Timeout { seconds }`, displays `the call timed out after {seconds}s`: the call ran longer than `DispatcherConfig::timeout`.
- `Empty`, displays `the backend returned no text`: the stream ended with no text and no reasoning.
- `ReasoningOnly { reasoning_chars }`, displays `the backend returned only reasoning ({reasoning_chars} characters) and no text`: the stream held reasoning and no text.
- `InputsChanged`, displays `the call's inputs changed while it waited`: the `DispatchGate` reported the task stale.
- `Call(CucaError)`, displays `the backend call failed: {error}`: any other provider error, or an error in the middle of the stream; no failover.

## `CucaExportError`

Needs `plugin-memory` or `service-prompt-cache`.

- `Json { message }`, displays `cuca-export JSON error: {message}`: malformed JSON, a missing or unknown field, or a wrong type.
- `Validation { component, field, message }`, displays `cuca-export validation failed for {component}.{field}: {message}`: a section failed its own check; `component` is `envelope`, `graph` or `prompt_cache`.
- `Unsupported { component }`, displays `cuca-export carries {component} data, but this build compiled that component out`: the document has `graph` or `prompt_cache` data and that feature is off.
- `State { message }`, displays `cuca-export state error: {message}`: live state could not be read or replaced, such as after a poisoned lock.

## `VectorStoreError`

Needs `service-vector-store`. It is not a `CucaError` variant. `From<VectorStoreError> for PluginError` turns `Embedder(e)` into `e` unchanged, `Embedding` and `EntryTooLarge` into `PluginError::Validation` (schema `vector-store`), and every other variant into `PluginError::Internal`.

- `Config(msg)`, displays `vector store configuration error: {msg}`: a bound is zero, `warn_fraction` is outside `(0.0, 1.0]`, an arena overflows `usize`, or `k` is zero.
- `Embedding(msg)`, displays `vector store embedding error: {msg}`: an embedding has the wrong width, or a component or squared norm is not finite.
- `EntryTooLarge { bytes, max_entry_bytes }`, displays `vector store turn of {bytes} bytes exceeds max_entry_bytes {max_entry_bytes}`: a turn is over `VectorStoreConfig::max_entry_bytes`; it is rejected, never cut.
- `Poisoned`, displays `vector store state lock is poisoned`: a thread panicked while holding the state lock.
- `Json(msg)`, displays `vector store JSON error: {msg}`: measuring a tool call's arguments failed; this cannot happen for a `serde_json::Value`.
- `Embedder(PluginError)`, displays `vector store embedder failed: {err}`: your `Embedder` failed; the error is passed through.

## Errors the stream drops

- An error from an `on_response_complete` hook is dropped, because the response was already delivered. With `plugin-guardrails` on, it is first logged as a `tracing::error!` event at target `cuca::client`, with the plugin name.
- A prompt cache write after a successful stream is advisory. If it fails, the turn still succeeds.
