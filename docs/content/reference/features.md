+++
title = "The feature matrix"
description = "Every Cargo feature, what each one enables, the optional dependencies they own, and the gate that fails a build with no provider."
template = "page.html"
weight = 1
+++

# The feature matrix

<dl class="page-facts">
<dt>In one line</dt>
<dd>Twenty-one features, all opt-in: seven providers, fourteen plugins, one cross-plugin edge</dd>
<dt>You need</dt>
<dd>At least one <code>provider-*</code> feature; the build stops without one</dd>
<dt>Read this if</dt>
<dd>You are choosing a feature list, or tracing which crate a feature pulls in</dd>
</dl>

## The provider gate

`default = []`. A build with no provider feature stops at a `compile_error!` in
`src/lib.rs`:

```text,name=The compile error verbatim
CUCA requires one provider-* feature; enable the adapter for the backend you use.
```

The gate is `#[cfg(not(any(...)))]` over all seven provider features. CI asserts
that `cargo check --no-default-features` fails.

## Provider features

| Feature | Enables |
|---|---|
| `provider-openai` | `dep:reqwest`, `dep:tokio-stream`, `reqwest/rustls` |
| `provider-anthropic` | `dep:reqwest`, `dep:tokio-stream`, `dep:sha2`, `dep:getrandom`, `dep:base64`, `reqwest/rustls` |
| `provider-gemini` | `dep:reqwest`, `dep:tokio-stream`, `reqwest/rustls` |
| `provider-deepseek` | `dep:reqwest`, `dep:tokio-stream`, `reqwest/rustls` |
| `provider-llamacpp` | `dep:reqwest`, `dep:tokio-stream`, `reqwest/rustls` |
| `provider-vllm` | `dep:reqwest`, `dep:tokio-stream`, `reqwest/rustls` |
| `provider-lmstudio` | `dep:reqwest`, `dep:tokio-stream`, `reqwest/rustls` |

## Plugin features

| Feature | Enables |
|---|---|
| `plugin-mcp` | `dep:rmcp`, `dep:tokio` |
| `plugin-sandbox` | `dep:wasmtime`, `dep:base64`, `wasmtime/wat` |
| `plugin-memory` | `dep:tiktoken-rs` |
| `plugin-entity-extraction` | `plugin-memory` |
| `plugin-guardrails` | `dep:jsonschema`, `dep:tracing` |
| `plugin-subagent` | `dep:tokio` |
| `plugin-hitl` | `dep:tokio` |
| `plugin-web-search` | `dep:reqwest`, `dep:tokio`, `reqwest/rustls` |
| `plugin-skills` | nothing |
| `plugin-telemetry` | `dep:opentelemetry`, `dep:opentelemetry_sdk`, `dep:tracing` |
| `plugin-speculative` | `dep:tokio`, `dep:tokio-stream` |
| `plugin-session-log` | `dep:postcard` |
| `plugin-prompt-cache` | `dep:sha2`, `dep:postcard` |
| `plugin-cost` | `dep:tiktoken-rs` |

`plugin-entity-extraction = ["plugin-memory"]` is the only feature edge between
two plugins.

## Unconditional dependencies

Five crates compile in every build, regardless of features. They carry the public
wire types, the SSE parser and the response stream contract.

| Crate | Version requirement | Features |
|---|---|---|
| `bytes` | `1` | default |
| `futures-core` | `0.3` | default |
| `memchr` | `2` | default |
| `serde` | `1` | `derive` |
| `serde_json` | `1` | default |

## Optional dependencies

| Crate | Version requirement | Declared features | Owned by |
|---|---|---|---|
| `reqwest` | `0.13` | `json`, `stream`, `form`; `default-features = false` | all seven `provider-*`, `plugin-web-search` |
| `tokio` | `1` | `rt`, `time`, `macros`, `sync`, `process`; `default-features = false` | `plugin-mcp`, `plugin-subagent`, `plugin-hitl`, `plugin-web-search`, `plugin-speculative` |
| `tokio-stream` | `0.1` | default | all seven `provider-*`, `plugin-speculative` |
| `sha2` | `0.11` | default | `provider-anthropic`, `plugin-prompt-cache` |
| `getrandom` | `0.4` | default | `provider-anthropic` |
| `base64` | `0.23` | none | `provider-anthropic`, `plugin-sandbox` |
| `rmcp` | `3` | `client`, `transport-child-process`, `transport-streamable-http-client-reqwest` | `plugin-mcp` |
| `wasmtime` | `48` | default, plus `wat` | `plugin-sandbox` |
| `tiktoken-rs` | `0.12` | default | `plugin-memory`, `plugin-cost` |
| `jsonschema` | `0.52` | `default-features = false` | `plugin-guardrails` |
| `opentelemetry` | `0.32` | default | `plugin-telemetry` |
| `opentelemetry_sdk` | `0.32` | `metrics`, `testing` | `plugin-telemetry` |
| `tracing` | `0.1` | default | `plugin-guardrails`, `plugin-telemetry` |
| `postcard` | `1` | `use-std`; `default-features = false` | `plugin-session-log`, `plugin-prompt-cache` |

`reqwest` is declared with `default-features = false`, so TLS arrives only
through the `reqwest/rustls` entry that every provider feature and
`plugin-web-search` carries. `jsonschema` is declared with
`default-features = false` because guardrails compiles inline schemas only and
needs no `$ref` retrieval. `postcard` is declared with
`default-features = false` because its default `heapless-cas` feature pulls in
`heapless`, which this crate never uses; `use-std` supplies the std buffer and
COBS helpers.

## Feature-conditional modules

| Module | Gate |
|---|---|
| `export` | `plugin-memory` or `plugin-prompt-cache` |
| `orchestrator` | `plugin-speculative` |

Every other public module compiles unconditionally: `types`, `error`, `request`,
`session`, `sse`, `plugin`, `plugins`, `client`. The `plugins` module is always
present; its submodules are individually gated, so it is empty in a build with no
plugin feature.

`plugin-speculative` has no submodule under `src/plugins/`. Its entire
implementation is `src/orchestrator.rs`.

## Crate metadata

| Field | Value |
|---|---|
| Package name | `cuca-core` |
| Version | `0.1.0` |
| Edition | `2024` |
| `rust-version` | `1.98` |
| Pinned toolchain | `1.98.0`, from `rust-toolchain.toml` |
| License | `Apache-2.0` |
| `[lib] doctest` | `false` |
| Binary targets | none |

`doctest = false` because the doctest pass performs a plain library build that
the provider gate rejects, and the crate has no doctests.

## Example targets

All four require `provider-llamacpp`; `cost_otel` needs two plugin features on
top, because the bridge it demonstrates is gated on both.

| Example | Path | Additional required features |
|---|---|---|
| `llamacpp_gemma` | `examples/llamacpp_gemma.rs` | none |
| `stream_all_blocks` | `examples/stream_all_blocks.rs` | none |
| `custom_plugin` | `examples/custom_plugin.rs` | none |
| `cost_otel` | `examples/cost_otel.rs` | `plugin-cost`, `plugin-telemetry` |

## Feature combinations CI verifies

| Job | Feature list |
|---|---|
| `clippy`, `test` | `--no-default-features --features provider-openai`, and `--all-features` |
| `no_provider` | `--no-default-features`, asserted to fail |
| `doc` | `--all-features` |
| `plugin_solo` | `--no-default-features --features provider-openai,<plugin>`, once per each of the fourteen plugin features |
| `plugin_layering` | greps asserting `src/plugins/memory/` never references entity extraction, and no file under `src/plugins/` gates on `plugin-speculative` or imports `crate::orchestrator` |

Next page: [Error types](@/reference/errors.md).
