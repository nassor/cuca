+++
title = "The feature matrix"
description = "Every Cargo feature, what each one enables, the optional dependencies they own, and the gate that fails a build with no provider."
template = "page.html"
weight = 1
+++

# The feature matrix

<dl class="page-facts">
<dt>In one line</dt>
<dd>Thirty-one opt-in features: eight providers, five <code>provider-local</code> GPU features, twelve plugins, six services</dd>
<dt>You need</dt>
<dd>At least one <code>provider-*</code> feature; the build stops without one</dd>
<dt>Read this if</dt>
<dd>You are choosing a feature list, or tracing which crate a feature pulls in</dd>
</dl>

`default = []`. Every feature below is off until you enable it. The source of truth is the `[features]` table in `Cargo.toml`.

## The provider gate

A build with none of the eight provider features stops at a `compile_error!` in `src/lib.rs`:

```text,name=The compile error verbatim
CUCA requires one provider-* feature; enable the adapter for the backend you use.
```

The gate is a `#[cfg(not(any(...)))]` over `provider-openai`, `provider-anthropic`, `provider-deepseek`, `provider-gemini`, `provider-llamacpp`, `provider-vllm`, `provider-lmstudio` and `provider-local`. The five GPU features are not providers and do not satisfy it. CI asserts that `cargo check --no-default-features` fails.

## Provider features

Eight features: seven HTTP adapters and one in-process engine. Each `dep:` entry is an optional dependency the feature turns on.

| Feature | Enables |
|---|---|
| `provider-openai` | `dep:reqwest`, `dep:tokio-stream`, `reqwest/rustls` |
| `provider-anthropic` | `dep:reqwest`, `dep:tokio-stream`, `dep:sha2`, `dep:getrandom`, `dep:base64`, `reqwest/rustls` |
| `provider-gemini` | `dep:reqwest`, `dep:tokio-stream`, `reqwest/rustls` |
| `provider-deepseek` | `dep:reqwest`, `dep:tokio-stream`, `reqwest/rustls` |
| `provider-llamacpp` | `dep:reqwest`, `dep:tokio-stream`, `reqwest/rustls` |
| `provider-vllm` | `dep:reqwest`, `dep:tokio-stream`, `reqwest/rustls` |
| `provider-lmstudio` | `dep:reqwest`, `dep:tokio-stream`, `reqwest/rustls` |
| `provider-local` | `dep:candle-core`, `dep:candle-nn`, `dep:half`, `dep:minijinja`, `dep:minijinja-contrib`, `dep:reqwest`, `dep:sha2`, `dep:tokio`, `dep:unicode-general-category`, `reqwest/rustls` |

`provider-local` does not use `tokio-stream`: its stream is a plain mpsc receiver. It uses `reqwest` only for the one-time model download.

## Local GPU features

Five features extend `provider-local`. None is a provider.

| Feature | Enables |
|---|---|
| `provider-local-gpu` | `provider-local`, `dep:burn` |
| `provider-local-vulkan` | `provider-local-gpu`, `burn/vulkan` |
| `provider-local-metal` | `provider-local-gpu`, `burn/metal` |
| `provider-local-cuda` | `provider-local-gpu`, `burn/cuda` |
| `provider-local-rocm` | `provider-local-gpu`, `burn/rocm` |

- `provider-local-gpu` is the shared GPU engine. It selects no backend by itself; each backend feature enables it.
- `provider-local` alone runs the model on the CPU.
- `LocalDevice` and `GpuAdapter` exist in every `provider-local` build. Selecting a backend whose feature is off makes `LocalModel::load` fail with `CucaError::Config`.
- `provider-local-cuda` and `provider-local-rocm` download a prebuilt LLVM on the first build. ROCm builds on Linux only.

## Plugin features

Twelve features. Each plugin implements `CucaPlugin` and runs on the request pipeline.

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

No plugin feature enables another plugin feature, and none names a service: the tier is flat.

## Service features

Six features. A service is driven by direct method calls and never implements `CucaPlugin`.

| Feature | Enables |
|---|---|
| `service-entity-extraction` | `plugin-memory` |
| `service-speculative` | `dep:tokio`, `dep:tokio-stream` |
| `service-replay` | `plugin-session-log` |
| `service-prompt-cache` | `dep:sha2`, `dep:postcard` |
| `service-rate-limit` | `dep:tokio` |
| `service-vector-store` | `plugin-memory`, `dep:wide` |

## Feature edges

Three hard edges, all service to plugin:

- `service-entity-extraction = ["plugin-memory"]`
- `service-replay = ["plugin-session-log"]`
- `service-vector-store = ["plugin-memory", "dep:wide"]`

One optional edge, not a Cargo feature: `ModelOrchestrator::with_session_store` exists only when `service-speculative` and `plugin-session-log` are both enabled.

`service-prompt-cache` and `service-rate-limit` declare no plugin dependency. A plugin never depends on, or names, a service.

## Unconditional dependencies

Five crates compile in every build. They carry the public wire types, the SSE parser and the response stream contract.

| Crate | Version | Features |
|---|---|---|
| `bytes` | `1` | default |
| `futures-core` | `0.3` | default |
| `memchr` | `2` | default |
| `serde` | `1` | `derive` |
| `serde_json` | `1` | default |

## Optional dependencies

| Crate | Version | Declared features | Owned by |
|---|---|---|---|
| `reqwest` | `0.13` | `json`, `stream`, `form`; no defaults | the eight `provider-*`, `plugin-web-search` |
| `tokio` | `1` | `rt`, `time`, `macros`, `sync`, `process`; no defaults | `provider-local`, `plugin-mcp`, `plugin-subagent`, `plugin-hitl`, `plugin-web-search`, `service-speculative`, `service-rate-limit` |
| `tokio-stream` | `0.1` | default | the seven HTTP `provider-*`, `service-speculative` |
| `sha2` | `0.11` | default | `provider-anthropic`, `provider-local`, `service-prompt-cache` |
| `getrandom` | `0.4` | default | `provider-anthropic` |
| `base64` | `0.23` | none | `provider-anthropic`, `plugin-sandbox` |
| `rmcp` | `3` | `client`, `transport-child-process`, `transport-streamable-http-client-reqwest` | `plugin-mcp` |
| `wasmtime` | `48` | default, plus `wat` | `plugin-sandbox` |
| `tiktoken-rs` | `0.12` | default | `plugin-memory`, `plugin-cost` |
| `jsonschema` | `0.52` | no defaults | `plugin-guardrails` |
| `opentelemetry` | `0.32` | default | `plugin-telemetry` |
| `opentelemetry_sdk` | `0.32` | `metrics`, `testing` | `plugin-telemetry` |
| `tracing` | `0.1` | default | `plugin-guardrails`, `plugin-telemetry`, `plugin-redaction` |
| `postcard` | `1` | `use-std`; no defaults | `plugin-session-log`, `service-prompt-cache` |
| `wide` | `1.7` | default | `service-vector-store` |
| `candle-core` | `0.11.0` | no defaults | `provider-local` |
| `candle-nn` | `0.11.0` | no defaults | `provider-local` |
| `half` | `2` | default | `provider-local` |
| `minijinja` | `2.24` | `loader` | `provider-local` |
| `minijinja-contrib` | `2.24` | `pycompat` | `provider-local` |
| `unicode-general-category` | `1` | default | `provider-local` (the byte-level BPE pre-tokenizers) |
| `burn` | `0.22` | `std`, `fusion`, `autotune`; no defaults | `provider-local-gpu` (backends `burn/vulkan`, `burn/metal`, `burn/cuda`, `burn/rocm`) |

### Why some defaults are off

- `reqwest` has no defaults, so TLS arrives only through the `reqwest/rustls` entry that every `provider-*` feature and `plugin-web-search` carries.
- `jsonschema` has no defaults because guardrails compiles inline schemas only and needs no `$ref` retrieval.
- `postcard` has no defaults because its default `heapless-cas` feature pulls in `heapless`, which the crate never uses. `use-std` supplies the std buffer and COBS helpers.
- `burn` has no defaults because they enable the optimizer and autodiff, which inference never uses.

### C compilers and downloads

- `candle-core` hard-depends on `tokenizers[onig]`, so `provider-local` compiles C through `onig_sys` and needs a C compiler.
- The Vulkan and Metal backends are Rust (wgpu: SPIR-V, MSL). Every GPU feature still compiles C, because CubeCL's runtime pulls an embedded store (`turso`) that builds C through `cc`.
- On Windows, a dependency's build script runs the SDK resource compiler (`rc.exe`; `windres` on the GNU target) for the GPU features.
- `provider-local-cuda` and `provider-local-rocm` pull CubeCL's LLVM backend. Its build script, `tracel-llvm-bundler`, downloads a prebuilt LLVM on the first build (network needed once, then cached) and links it statically.
- docs.rs builds have no network, so the `[package.metadata.docs.rs]` feature list names every feature except those two.

## Feature-conditional modules

| Module | Gate |
|---|---|
| `export` | `plugin-memory` or `service-prompt-cache` |
| `cost_otel` | `plugin-cost` and `plugin-telemetry` |

Every other public module compiles unconditionally: `types`, `error`, `request`, `session`, `sse`, `plugin`, `plugins`, `services`, `client`. The `plugins` and `services` modules are always present. Their submodules are individually gated, so each is empty when none of its features is on. The `provider` module is `pub(crate)`.

## Crate metadata

| Field | Value |
|---|---|
| Package name | `cuca` |
| Version | `0.5.1` |
| Edition | `2024` |
| `rust-version` | `1.98` |
| Pinned toolchain | `1.98.0`, from `rust-toolchain.toml` |
| License | `Apache-2.0` |
| `[lib] doctest` | `false` |
| Binary targets | none |

`doctest = false` because the doctest pass does a plain library build that the provider gate rejects, and the crate has no doctests.

## Example targets

Twenty-three examples. All but `local_model` require `provider-llamacpp`. Run one with `cargo run --example <name> --features <list>`.

| Example | Required features |
|---|---|
| `llamacpp_gemma`, `stream_all_blocks`, `custom_plugin` | `provider-llamacpp` |
| `local_model` | `provider-local` |
| `cost` | `provider-llamacpp`, `plugin-cost` |
| `cost_otel` | `provider-llamacpp`, `plugin-cost`, `plugin-telemetry` |
| `entity_extraction` | `provider-llamacpp`, `service-entity-extraction` |
| `guardrails` | `provider-llamacpp`, `plugin-guardrails` |
| `hitl` | `provider-llamacpp`, `plugin-hitl` |
| `mcp` | `provider-llamacpp`, `plugin-mcp` |
| `memory` | `provider-llamacpp`, `plugin-memory` |
| `prompt_cache` | `provider-llamacpp`, `service-prompt-cache` |
| `rate_limit` | `provider-llamacpp`, `service-rate-limit` |
| `redaction` | `provider-llamacpp`, `plugin-redaction` |
| `replay` | `provider-llamacpp`, `service-replay` |
| `sandbox` | `provider-llamacpp`, `plugin-sandbox` |
| `session_log` | `provider-llamacpp`, `plugin-session-log` |
| `skills` | `provider-llamacpp`, `plugin-skills` |
| `speculative` | `provider-llamacpp`, `service-speculative` |
| `subagent` | `provider-llamacpp`, `plugin-subagent` |
| `telemetry` | `provider-llamacpp`, `plugin-telemetry` |
| `vector_store` | `provider-llamacpp`, `service-vector-store` |
| `web_search` | `provider-llamacpp`, `plugin-web-search` |

Examples live in `examples/<name>.rs`.

## Bench target

`benches/vector_store.rs` is a Divan benchmark with `harness = false`. It requires `provider-openai` and `service-vector-store`. No CI job runs it.

## Feature combinations CI verifies

| Job | Feature list |
|---|---|
| `clippy`, `test` | `--no-default-features --features provider-openai`, and `--no-default-features --features $ALL_NO_LLVM` |
| `no_provider` | `--no-default-features`, asserted to fail |
| `doc` | `--no-default-features --features $ALL_NO_LLVM` |
| `local_gpu_llvm` | `--all-targets --no-default-features --features provider-local-cuda,provider-local-rocm`, `cargo check` on Ubuntu; the only job that downloads the LLVM bundle |
| `plugin_solo` | `--no-default-features --features provider-openai,<plugin>`, once per plugin feature (twelve) |
| `service_solo` | `--no-default-features --features provider-openai,<service>`, once per service feature (six) |
| `plugin_layering` | greps asserting no file under `src/plugins/` names a `service-` feature, a `crate::services` path, or a `plugins::` path for `prompt_cache`, `entity_extraction`, `replay` or `rate_limit` |

`ALL_NO_LLVM` is a top-level `env` entry in `.github/workflows/ci.yml`, the same list as the docs.rs feature set: every feature except `provider-local-cuda` and `provider-local-rocm`. The run steps expand it as `$ALL_NO_LLVM`.
