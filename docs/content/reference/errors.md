+++
title = "Error types"
description = "Every variant of CucaError, PluginError, PromptCacheError and CucaExportError, with its Display text and its conversions."
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

## `CucaError`

Nine variants. `#[derive(Debug, Clone)]`. No variant is feature-gated.

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

The `provider` field in `Provider` formats through `ProviderEndpoint`'s
`Display`, whose labels differ from its serde names: `llamacpp`, `vllm`,
`lmstudio`, `openai`, `anthropic`, `gemini`, `deepseek`, and the inner string for
`Custom`.

`CucaError::provider(endpoint, msg)` constructs the `Provider` variant.

### `source()`

`impl std::error::Error for CucaError` returns `Some` from `source()` for the
`Plugin` variant only. Every other variant returns `None`: they carry formatted
strings rather than a typed cause.

### Config messages the crate emits verbatim

| Message | Condition |
|---|---|
| `no provider selected; call with_provider before build` | `CucaClientBuilder::build` with no provider set |
| `custom endpoints require a registered adapter` | dispatching `ProviderEndpoint::Custom` |
| `anthropic requires an api key or bearer token` | `dispatch_anthropic` with neither configured |
| `gemini requires an api key` | `dispatch_gemini` with no key |
| `deepseek anthropic bridge requires an api key` | the DeepSeek bridge route with no key |
| `no prompt cache configured` | `prompt_cache_snapshot` or `replace_prompt_cache_snapshot` with no cache |
| `client pool lock poisoned` | `ClientPool::get_or_create` on a poisoned mutex |

## `PluginError`

Five variants. `#[derive(Debug, Clone)]`. No variant is feature-gated.

| Variant | Fields | `Display` |
|---|---|---|
| `HookFailure` | `plugin: &'static str`, `stage: &'static str`, `message: String` | `plugin {plugin} failed at stage {stage}: {message}` |
| `Validation` | `schema: String`, `message: String` | `validation failed for schema {schema}: {message}` |
| `NotSupported` | `String` | `not supported: {msg}` |
| `Io` | `String` | `plugin I/O error: {msg}` |
| `Internal` | `String` | `internal plugin error: {msg}` |

`PluginError::hook(plugin, stage, msg)` constructs the `HookFailure` variant.

`impl std::error::Error for PluginError` overrides nothing, so `source()` is
always `None`.

### Where each variant comes from

| Variant | Raised by |
|---|---|
| `Validation` | schema validation, malformed tool arguments, a zero cap passed to a `with_max_*` constructor, a `execute_local_tool` result whose id does not match its call |
| `NotSupported` | `McpTransport::WebSocket`, `WebSearchPlugin::extract_page` on a non-Firecrawl provider, an unknown subagent id, a failed worktree preparation |
| `Io` | file and directory access in `plugin-skills` and `JsonFileBackend`, and every `From<std::io::Error>` conversion |
| `Internal` | a full cap that refuses rather than evicts, a closed channel, a non-2xx HTTP reply inside a plugin, a panicked worker thread |
| `HookFailure` | available through the `hook` constructor; no bundled plugin raises it |

## Conversions

| Impl | Result | Gate |
|---|---|---|
| `From<PluginError> for CucaError` | `CucaError::Plugin(e)` | none |
| `From<serde_json::Error> for CucaError` | `CucaError::Json { message }` | none |
| `From<std::io::Error> for CucaError` | `CucaError::Io { message }` | none |
| `From<std::io::Error> for PluginError` | `PluginError::Io(msg)` | none |
| `From<reqwest::Error> for CucaError` | `CucaError::Transport { message }` | any of the seven `provider-*` features |

The `reqwest` conversion is the only feature-gated item in `src/error.rs`. It
formats through `Display` because `reqwest::Error` is not `Clone` and
`CucaError` is.

## `PromptCacheError`

Four variants, gated on `plugin-prompt-cache`. Not a `CucaError` variant, and
not convertible to one by a `From` impl: `CucaClient` maps it at the seam,
sending `Json` to `CucaError::Json` and everything else to `CucaError::Config`.

| Variant | Fields | Condition |
|---|---|---|
| `Config` | `String` | `capacity == 0` or a zero `ttl` passed to `PromptCacheConfig::new` |
| `Validation` | `field: String`, `message: String` | a malformed snapshot entry: bad key shape, digest mismatch, timestamp order, duplicate key or rank, non-finite temperature |
| `Json` | `String` | serialization failure while digesting a request |
| `Lock` | `String` | a poisoned state mutex |

## `CucaExportError`

Four variants, gated on `plugin-memory` or `plugin-prompt-cache`.

| Variant | Fields | Condition |
|---|---|---|
| `Json` | `message: String` | encoding or decoding the envelope failed |
| `Validation` | `component: &'static str`, `field: &'static str`, `message: String` | a section failed its own validation |
| `Unsupported` | `component: &'static str` | the envelope carries non-empty data for a component this build compiled out |
| `State` | `message: String` | a staged import could not be committed |

## Errors the stream swallows

One place in the pipeline does not propagate. An error returned from an
`on_response_complete` hook is logged and discarded, because the response has
already been delivered to the caller. Under `plugin-guardrails` the log goes
through `tracing::error!` at target `cuca::client` with the plugin name
attached; without it, the error is dropped.

A prompt-cache write after a successful stream is advisory in the same way: its
failure never fails the turn it belongs to.

Next page: [Environment variables](@/reference/environment.md).
