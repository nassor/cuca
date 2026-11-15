+++
title = "The feature matrix"
description = "Every Cargo feature of cuca: what it is for, what it turns on, which recipe to copy, and the C compiler and download notes."
template = "page.html"
weight = 1
+++

# The feature matrix

<dl class="page-facts">
<dt>In one line</dt>
<dd>Thirty-five opt-in features: eight providers, five local GPU features, twelve plugins, seven services, the <code>config-kdl</code> reader and two server features</dd>
<dt>You need</dt>
<dd>At least one <code>provider-*</code> feature; the build stops without one</dd>
<dt>Read this if</dt>
<dd>You are choosing a feature list, or you want to know what a feature pulls into your build</dd>
</dl>

Every feature is off until you turn it on (`default = []`). A feature adds code and dependencies only when it is on, so a small feature list means a small, fast build. The `[features]` table in `Cargo.toml` is the source of truth.

## Which features do I need

Pick the row that matches what you want to do. Add plugin and service features to any row.

| I want to | Features |
|---|---|
| Stream from a `llama-server` (the quick start) | `provider-llamacpp` |
| Call OpenAI, Anthropic, Gemini or DeepSeek | `provider-openai`, `provider-anthropic`, `provider-gemini` or `provider-deepseek` |
| Call vLLM or LM Studio | `provider-vllm` or `provider-lmstudio` |
| Run a language model inside my program, on the CPU | `provider-local` |
| Run it on a GPU | `provider-local-vulkan` (Windows, Linux), `provider-local-metal` (macOS), `provider-local-cuda` or `provider-local-rocm` |
| Serve the local model over an OpenAI-compatible HTTP API | `server` plus a GPU feature if you want one, for example `server,provider-local-vulkan` |
| Add a terminal dashboard for the server | `server-tui` |
| Read a KDL settings file in my own program | `config-kdl` |
| Let the model call tools through MCP | any provider plus `plugin-mcp` |

In `Cargo.toml`:

```toml,name=A provider and two plugins
[dependencies]
cuca = { version = "0.8", default-features = false, features = ["provider-openai", "plugin-memory", "plugin-guardrails"] }
```

In a cargo command, name the features every time:

```bash,name=Runs the same on all three platforms
cargo run --release --example local_model --features provider-local-vulkan
```

## The provider gate

With no provider feature on, the build stops with this message:

```text,name=The compile error
CUCA requires one provider-* feature; enable the adapter for the backend you use.
```

Eight features satisfy the gate: `provider-openai`, `provider-anthropic`, `provider-deepseek`, `provider-gemini`, `provider-llamacpp`, `provider-vllm`, `provider-lmstudio` and `provider-local`. The five GPU features enable `provider-local`, so they satisfy it too. `server` and `server-tui` also enable `provider-local`.

## How the features relate

<div class="dgm">
<div class="dgm-scroll">
<svg viewBox="0 0 660 296" role="img" aria-labelledby="featx-title featx-desc">
  <title id="featx-title">The feature tiers of cuca and which feature turns on which.</title>
  <desc id="featx-desc">Top row: the core, twelve plugins and seven services. Three services turn on a plugin. Middle row: seven HTTP providers, the in-process provider-local, and the GPU features that turn on provider-local. Bottom row: provider-local turns on config-kdl, and the server features turn on provider-local and config-kdl.</desc>
  <defs>
    <marker id="featx-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M0 0.6 L7 4 L0 7.4 z" fill="var(--muted-foreground)"/>
    </marker>
  </defs>
  <text class="t-title" x="6" y="20">Arrows read "turns on"</text>
  <rect class="blk" x="6" y="36" width="170" height="62" rx="6"/>
  <text class="t-lbl" x="18" y="58">core</text>
  <text class="t-sm" x="18" y="74">no feature needed</text>
  <text class="t-sm" x="18" y="88">types, client, SSE</text>
  <rect class="blk blk-ctl" x="236" y="36" width="170" height="62" rx="6"/>
  <text class="t-lbl t-ctl" x="248" y="58">plugin-* (12)</text>
  <text class="t-sm" x="248" y="74">hooks on every request</text>
  <text class="t-sm" x="248" y="88">no plugin edges</text>
  <rect class="blk blk-ctl" x="466" y="36" width="170" height="62" rx="6"/>
  <text class="t-lbl t-ctl" x="478" y="58">service-* (7)</text>
  <text class="t-sm" x="478" y="74">called directly</text>
  <text class="t-sm" x="478" y="88">3 turn on a plugin</text>
  <path class="arw" d="M464 67 H408" marker-end="url(#featx-arrow)"/>
  <rect class="blk blk-bnd" x="6" y="126" width="170" height="62" rx="6"/>
  <text class="t-lbl t-bnd" x="18" y="148">7 HTTP providers</text>
  <text class="t-sm" x="18" y="164">openai, anthropic, ...</text>
  <text class="t-sm" x="18" y="178">reqwest with rustls</text>
  <rect class="blk blk-bnd" x="236" y="126" width="170" height="62" rx="6"/>
  <text class="t-lbl t-bnd" x="248" y="148">provider-local</text>
  <text class="t-sm" x="248" y="164">in-process, CPU</text>
  <text class="t-sm" x="248" y="178">turns on config-kdl</text>
  <rect class="blk blk-bnd" x="466" y="126" width="170" height="62" rx="6"/>
  <text class="t-lbl t-bnd" x="478" y="148">provider-local-gpu</text>
  <text class="t-sm" x="478" y="164">+ vulkan, metal,</text>
  <text class="t-sm" x="478" y="178">cuda, rocm (5 total)</text>
  <path class="arw" d="M464 157 H408" marker-end="url(#featx-arrow)"/>
  <rect class="blk" x="236" y="216" width="170" height="62" rx="6"/>
  <text class="t-lbl" x="248" y="238">config-kdl</text>
  <text class="t-sm" x="248" y="254">KDL file reader</text>
  <rect class="blk" x="466" y="216" width="170" height="62" rx="6"/>
  <text class="t-lbl" x="478" y="238">server, server-tui</text>
  <text class="t-sm" x="478" y="254">turn on provider-local</text>
  <text class="t-sm" x="478" y="268">and config-kdl</text>
  <path class="arw" d="M321 190 V214" marker-end="url(#featx-arrow)"/>
  <path class="arw" d="M464 247 H408" marker-end="url(#featx-arrow)"/>
</svg>
</div>
<p class="dgm-cap"><b>Features stack in one direction.</b> A service may turn on a plugin, and a plugin never names a service. Plugins never turn on other plugins, and services never turn on other services.</p>
</div>

The three service to plugin edges are `service-entity-extraction` and `service-vector-store` turning on `plugin-memory`, and `service-replay` turning on `plugin-session-log`. `ModelOrchestrator::with_session_store` exists only when `service-speculative` and `plugin-session-log` are both on. That link is not a Cargo feature edge.

## Provider features

Eight features: seven HTTP adapters and one in-process engine. [Providers](@/providers/_index.md) has one page for each.

| Feature | What it is for | Pulls in |
|---|---|---|
| `provider-openai` | OpenAI Chat Completions | `reqwest`, `tokio-stream` |
| `provider-anthropic` | Anthropic Messages, with API key, bearer token or OAuth PKCE | `reqwest`, `tokio-stream`, `sha2`, `getrandom`, `base64` |
| `provider-gemini` | Google Gemini | `reqwest`, `tokio-stream` |
| `provider-deepseek` | DeepSeek, native and Anthropic bridge routes | `reqwest`, `tokio-stream` |
| `provider-llamacpp` | `llama-server` | `reqwest`, `tokio-stream` |
| `provider-vllm` | vLLM | `reqwest`, `tokio-stream` |
| `provider-lmstudio` | LM Studio | `reqwest`, `tokio-stream` |
| `provider-local` | Inference inside your process on the CPU, from a GGUF file or a Hugging Face safetensors checkpoint, no HTTP and no llama.cpp | `config-kdl`, `candle-core`, `candle-nn`, `half`, `minijinja`, `minijinja-contrib`, `reqwest`, `sha2`, `sysinfo`, `tokio`, `tokenizers`, `unicode-general-category`, `wide` |

Every `reqwest` entry turns on its `rustls` TLS backend, so no feature needs OpenSSL. `provider-local` uses `reqwest` only to download model files, `sha2` to check them, and `sysinfo` to check free disk space when it converts a checkpoint.

## Local GPU features

Five features extend `provider-local`. None is a provider on its own. See [Run on a GPU](@/providers/local/gpu.md).

| Feature | What it is for | Pulls in |
|---|---|---|
| `provider-local-gpu` | the shared GPU engine; picks no backend by itself | `provider-local`, `burn`, `burn-cubecl`, `cubecl` |
| `provider-local-vulkan` | Vulkan GPUs (Windows, Linux) | `provider-local-gpu`, `burn/vulkan` |
| `provider-local-metal` | Apple GPUs | `provider-local-gpu`, `burn/metal` |
| `provider-local-cuda` | NVIDIA GPUs through CUDA | `provider-local-gpu`, `burn/cuda` |
| `provider-local-rocm` | AMD GPUs through ROCm (Linux, Windows) | `provider-local-gpu`, `burn/rocm` |

- `LocalDevice` and `GpuAdapter` exist in every `provider-local` build. Choosing a backend whose feature is off makes `LocalModel::load` fail with `CucaError::Config`.
- A GPU feature alone does not move the model: set `device` in the model configuration. See [Configuration](@/providers/local/configuration.md).

## Plugin features

Twelve features. A plugin implements `CucaPlugin` and runs as a hook on every request. [Plugins](@/plugins/_index.md) has one page for each.

| Feature | What it is for | Pulls in |
|---|---|---|
| `plugin-mcp` | call tools on MCP servers over a child process or Streamable HTTP | `rmcp`, `tokio` |
| `plugin-sandbox` | run WebAssembly tool code with a fuel, memory and time limit | `wasmtime`, `base64` |
| `plugin-memory` | keep long chats inside the context window, plus a memory graph | `tiktoken-rs` |
| `plugin-guardrails` | check tool-call arguments against a JSON schema | `jsonschema`, `tracing` |
| `plugin-subagent` | let the model start a sub-agent as a tool call | `tokio` |
| `plugin-hitl` | ask a human to approve risky tool calls | `tokio` |
| `plugin-web-search` | web search and page extraction as tools | `reqwest`, `tokio` |
| `plugin-skills` | reusable agent skills from `SKILL.md` files | nothing |
| `plugin-telemetry` | OpenTelemetry traces and metrics | `opentelemetry`, `opentelemetry_sdk`, `tracing` |
| `plugin-session-log` | an append-only log of every session | `postcard` |
| `plugin-cost` | token and currency ledger with a hard budget cap | `tiktoken-rs` |
| `plugin-redaction` | scrub PII and secrets from outgoing requests | `tracing` |

## Service features

Seven features. A service is a type you call yourself; it never implements `CucaPlugin`. [Services](@/services/_index.md) has one page for each.

| Feature | What it is for | Pulls in |
|---|---|---|
| `service-entity-extraction` | pull entities and relationships out of text into a memory graph | `plugin-memory` |
| `service-speculative` | a fast model answers first and a slow model checks | `tokio`, `tokio-stream` |
| `service-replay` | replay a recorded session as a response stream | `plugin-session-log` |
| `service-prompt-cache` | a local response cache keyed by the request | `sha2`, `postcard` |
| `service-rate-limit` | a token bucket plus a concurrency cap | `tokio` |
| `service-vector-store` | an in-memory vector store for recalled turns | `plugin-memory`, `wide` |
| `service-scheduler` | a priority scheduler and dispatcher over several backends | `tokio`, `tracing` |

## Configuration and server features

| Feature | What it is for | Pulls in |
|---|---|---|
| `config-kdl` | read a settings file in KDL: `from_kdl_str`, `from_kdl_str_with_vars`, `load_kdl_str`, `load_kdl_file` | `kdl` |
| `server` | the `cuca` binary: `cuca serve` and `cuca balance`; an OpenAI-compatible HTTP API, stats and metrics | `provider-local`, `config-kdl`, `axum`, `clap`, `tracing`, `tracing-subscriber`, `sysinfo`, `nvml-wrapper`, `getrandom`, `tokio-stream`; on Windows also `wmi` and `winreg` |
| `server-tui` | `cuca tui`, a live terminal dashboard | `server`, `ratatui`, `crossterm` |

The `cuca::config` helpers `one_or_many`, `from_str`, `size`, `duration` and `ConfigValue` compile in every build. [Configuration files](@/reference/environment.md) describes the KDL reader. [Server](@/server/_index.md) describes the binary.

## Build notes

- `provider-local` compiles some C code, so it needs a C compiler. Every GPU feature also compiles C.
- On Windows, the GPU features run the SDK resource compiler (`rc.exe`, or `windres` on the GNU target) in a build script.
- `provider-local-cuda` and `provider-local-rocm` download a prebuilt LLVM on the first build, so they need network access once. The download is cached. `provider-local-vulkan` and `provider-local-metal` need no download.
- docs.rs builds have no network, so the docs.rs feature list names every feature except those two: thirty-three of the thirty-five.
- Publishing and CI use the same list, `ALL_NO_LLVM`. Do not build with `--all-features`; on Windows it fails to link.
- `provider-local` turns on `serde_json/preserve_order`, so JSON maps keep the order the caller wrote them. Chat templates see that order.

## Modules that depend on features

| Module | Present when |
|---|---|
| `export` | `plugin-memory` or `service-prompt-cache` |
| `cost_otel` | `plugin-cost` and `plugin-telemetry` |
| `server` | `server` |

Every other public module compiles in every build. `plugins` and `services` always exist, and each of their submodules is empty unless its feature is on. See [The public API surface](@/reference/public-api.md).

## Always-on dependencies

Five crates compile in every build. They carry the wire types, the SSE parser and the response stream.

| Crate | Features |
|---|---|
| `bytes` | default |
| `futures-core` | default |
| `memchr` | default |
| `serde` | `derive` |
| `serde_json` | default |

## Examples

There are 24 examples in `examples/`. Run one with `cargo run --example <name> --features <list>`. All but `local_model` need `provider-llamacpp`, because they stream from a local `llama-server`.

| Example | Features |
|---|---|
| `llamacpp_gemma`, `stream_all_blocks`, `custom_plugin` | `provider-llamacpp` |
| `local_model` | `provider-local` |
| `cost`, `cost_otel` | `provider-llamacpp`, `plugin-cost`; `cost_otel` also `plugin-telemetry` |
| `guardrails`, `hitl`, `mcp`, `memory`, `redaction`, `sandbox`, `session_log`, `skills`, `subagent`, `telemetry`, `web_search` | `provider-llamacpp` plus the plugin feature with the same name, such as `plugin-hitl` for `hitl` (`session_log` uses `plugin-session-log`, `web_search` uses `plugin-web-search`) |
| `entity_extraction`, `prompt_cache`, `rate_limit`, `replay`, `scheduler`, `speculative`, `vector_store` | `provider-llamacpp` plus the service feature with the same name, such as `service-rate-limit` for `rate_limit` |

`benches/vector_store.rs` is a Divan benchmark. It needs `provider-openai` and `service-vector-store`.

## Crate facts

| Field | Value |
|---|---|
| Package | `cuca`, version `0.8.0` |
| Edition | `2024` |
| Minimum Rust | `1.98`; `rust-toolchain.toml` pins `1.98.0` |
| License | `Apache-2.0` |
| Binary | `cuca` (`src/bin/cuca.rs`), needs `server` |
