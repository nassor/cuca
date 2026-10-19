+++
title = "The feature matrix"
description = "Every Cargo feature, what each one enables, the optional dependencies they own, and the gate that fails a build with no provider."
template = "page.html"
weight = 1
+++

# The feature matrix

<dl class="page-facts">
<dt>In one line</dt>
<dd>Thirty-one features, all opt-in: eight providers, five <code>provider-local</code> GPU features, twelve plugins, six services</dd>
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

The gate is `#[cfg(not(any(...)))]` over all eight provider features. CI asserts
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
| `provider-local` | `dep:candle-core`, `dep:candle-nn`, `dep:half`, `dep:minijinja`, `dep:minijinja-contrib`, `dep:reqwest`, `dep:sha2`, `dep:tokio`, `reqwest/rustls` |
| `provider-local-gpu` | `provider-local`, `dep:burn` |
| `provider-local-vulkan` | `provider-local-gpu`, `burn/vulkan` |
| `provider-local-metal` | `provider-local-gpu`, `burn/metal` |
| `provider-local-cuda` | `provider-local-gpu`, `burn/cuda` |
| `provider-local-rocm` | `provider-local-gpu`, `burn/rocm` |

`provider-local-gpu` is the shared GPU engine and selects no backend by itself;
each backend feature enables it. None of the five is a provider: the provider gate
counts only the eight `provider-*` adapters, and `provider-local` alone still runs
the model on the CPU. `LocalDevice` and `GpuAdapter` exist in every `provider-local`
build; selecting a backend whose feature is off fails `LocalModel::load` with a
`CucaError::Config`.

## Plugin features

| Feature | Enables |
|---|---|
| `plugin-mcp` | `dep:rmcp`, `dep:tokio` |
| `plugin-sandbox` | `dep:wasmtime`, `dep:base64`, `wasmtime/wat` |
| `plugin-memory` | `dep:tiktoken-rs` |
| `plugin-guardrails` | `dep:jsonschema`, `dep:tracing` |
| `plugin-subagent` | `dep:tokio` |
| `plugin-hitl` | `dep:tokio` |
| `plugin-web-search` | `dep:reqwest`, `dep:tokio`, `reqwest/rustls` |
| `plugin-skills` | nothing |
| `plugin-telemetry` | `dep:opentelemetry`, `dep:opentelemetry_sdk`, `dep:tracing` |
| `plugin-session-log` | `dep:postcard` |
| `plugin-cost` | `dep:tiktoken-rs` |
| `plugin-redaction` | `dep:tracing` |

No plugin depends on another plugin: the tier is flat.

## Service features

| Feature | Enables |
|---|---|
| `service-entity-extraction` | `plugin-memory` |
| `service-speculative` | `dep:tokio`, `dep:tokio-stream` |
| `service-replay` | `plugin-session-log` |
| `service-prompt-cache` | `dep:sha2`, `dep:postcard` |
| `service-rate-limit` | `dep:tokio` |
| `service-vector-store` | `plugin-memory`, `dep:wide` |

`service-entity-extraction = ["plugin-memory"]`,
`service-replay = ["plugin-session-log"]` and
`service-vector-store = ["plugin-memory", "dep:wide"]` are the three hard feature edges,
all service to plugin. `service-speculative` adds a fourth,
documented-optional edge to `plugin-session-log`. `service-prompt-cache` and
`service-rate-limit` declare no plugin dependency at all. A plugin must never
depend on, or name, a service.

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
| `reqwest` | `0.13` | `json`, `stream`, `form`; `default-features = false` | all eight `provider-*`, `plugin-web-search` |
| `tokio` | `1` | `rt`, `time`, `macros`, `sync`, `process`; `default-features = false` | `provider-local`, `plugin-mcp`, `plugin-subagent`, `plugin-hitl`, `plugin-web-search`, `service-speculative`, `service-rate-limit` |
| `tokio-stream` | `0.1` | default | the seven HTTP `provider-*` (not `provider-local`), `service-speculative` |
| `sha2` | `0.11` | default | `provider-anthropic`, `provider-local`, `service-prompt-cache` |
| `getrandom` | `0.4` | default | `provider-anthropic` |
| `base64` | `0.23` | none | `provider-anthropic`, `plugin-sandbox` |
| `rmcp` | `3` | `client`, `transport-child-process`, `transport-streamable-http-client-reqwest` | `plugin-mcp` |
| `wasmtime` | `48` | default, plus `wat` | `plugin-sandbox` |
| `tiktoken-rs` | `0.12` | default | `plugin-memory`, `plugin-cost` |
| `jsonschema` | `0.52` | `default-features = false` | `plugin-guardrails` |
| `opentelemetry` | `0.32` | default | `plugin-telemetry` |
| `opentelemetry_sdk` | `0.32` | `metrics`, `testing` | `plugin-telemetry` |
| `tracing` | `0.1` | default | `plugin-guardrails`, `plugin-telemetry`, `plugin-redaction` |
| `postcard` | `1` | `use-std`; `default-features = false` | `plugin-session-log`, `service-prompt-cache` |
| `wide` | `1.7` | default | `service-vector-store` |
| `candle-core` | `0.11.0` | `default-features = false` | `provider-local` |
| `candle-nn` | `0.11.0` | `default-features = false` | `provider-local` |
| `half` | `2` | default | `provider-local` |
| `minijinja` | `2.24` | `loader` | `provider-local` |
| `minijinja-contrib` | `2.24` | `pycompat` | `provider-local` |
| `burn` | `0.22` | `std`, `fusion`, `autotune`; `default-features = false` | `provider-local-gpu` (backends: `burn/vulkan`, `burn/metal`, `burn/cuda`, `burn/rocm`) |

`reqwest` is declared with `default-features = false`, so TLS arrives only
through the `reqwest/rustls` entry that every provider feature and
`plugin-web-search` carries. `candle-core` hard-depends on `tokenizers[onig]`,
so `provider-local` also compiles C (`onig_sys`). `burn` is declared with
`default-features = false` because its defaults enable the optimizer and autodiff,
which inference never uses. The graphics backends behind `provider-local-vulkan` and
`provider-local-metal` are Rust (wgpu: SPIR-V, MSL), but every GPU feature compiles C:
CubeCL's runtime pulls an embedded store (`turso`) that builds C through `cc`, and on
Windows a dependency's build script runs the SDK resource compiler (`rc.exe`; `windres`
on the GNU target). `provider-local-cuda` and `provider-local-rocm` pull
cubecl's LLVM backend: its build script, `tracel-llvm-bundler`, downloads a prebuilt
LLVM on the first build (network needed once, then cached) and links it statically,
and ROCm (HIP bindings) is Linux-only. docs.rs builds have no network, so the
`[package.metadata.docs.rs]` feature list names every feature except those two.
`jsonschema` is declared with
`default-features = false` because guardrails compiles inline schemas only and
needs no `$ref` retrieval. `postcard` is declared with
`default-features = false` because its default `heapless-cas` feature pulls in
`heapless`, which this crate never uses; `use-std` supplies the std buffer and
COBS helpers.

## Feature-conditional modules

| Module | Gate |
|---|---|
| `export` | `plugin-memory` or `service-prompt-cache` |
| `services` | any of the six `service-*` features |

Every other public module compiles unconditionally: `types`, `error`, `request`,
`session`, `sse`, `plugin`, `plugins`, `client`. The `plugins` module is always
present; its submodules are individually gated, so it is empty in a build with no
plugin feature.

`services` is empty in a build with no service feature enabled, the same way
`plugins` is empty with no plugin feature enabled.

## Crate metadata

| Field | Value |
|---|---|
| Package name | `cuca` |
| Version | `0.5.0` |
| Edition | `2024` |
| `rust-version` | `1.98` |
| Pinned toolchain | `1.98.0`, from `rust-toolchain.toml` |
| License | `Apache-2.0` |
| `[lib] doctest` | `false` |
| Binary targets | none |

`doctest = false` because the doctest pass performs a plain library build that
the provider gate rejects, and the crate has no doctests.

## Example targets

All six require `provider-llamacpp`; `cost_otel`, `rate_limit`, and `redaction` each
need one or more plugin or service features on top, because what they demonstrate is gated on them.

| Example | Path | Additional required features |
|---|---|---|
| `llamacpp_gemma` | `examples/llamacpp_gemma.rs` | none |
| `stream_all_blocks` | `examples/stream_all_blocks.rs` | none |
| `custom_plugin` | `examples/custom_plugin.rs` | none |
| `cost_otel` | `examples/cost_otel.rs` | `plugin-cost`, `plugin-telemetry` |
| `rate_limit` | `examples/rate_limit.rs` | `service-rate-limit` |
| `redaction` | `examples/redaction.rs` | `plugin-redaction` |

## Feature combinations CI verifies

| Job | Feature list |
|---|---|
| `clippy`, `test` | `--no-default-features --features provider-openai`, and `--no-default-features --features $ALL_NO_LLVM` |
| `no_provider` | `--no-default-features`, asserted to fail |
| `doc` | `--no-default-features --features $ALL_NO_LLVM` |
| `local_gpu_llvm` | `--all-targets --no-default-features --features provider-local-cuda,provider-local-rocm`, `cargo check` on ubuntu; the only job that downloads the LLVM bundle |
| `plugin_solo` | `--no-default-features --features provider-openai,<plugin>`, once per each of the twelve plugin features |
| `service_solo` | `--no-default-features --features provider-openai,<service>`, once per each of the six service features |
| `plugin_layering` | greps asserting no file under `src/plugins/` names a `service-` feature, references `crate::services`, or imports one of the `plugins::` paths the five moved modules left behind |

`ALL_NO_LLVM` is a top-level `env` entry in `.github/workflows/ci.yml` (the same
list as the docs.rs feature set): every feature except `provider-local-cuda` and
`provider-local-rocm`. The run steps expand it as `$ALL_NO_LLVM`, and it replaces
`--all-features` in the `clippy`, `test` and `doc` jobs.

Next page: [Error types](@/reference/errors.md).
