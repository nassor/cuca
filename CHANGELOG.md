# Changelog

## Unreleased

### Breaking

- `LocalModelConfig` has a new field, `device: LocalDevice` (default
  `LocalDevice::Cpu`). Code that builds `LocalModelConfig` with an exhaustive
  struct literal, without `..Default::default()`, must set it.

### Added

- `provider-local` runs on a GPU through burn 0.22 / CubeCL. Five new cargo
  features: `provider-local-gpu` (the shared GPU engine; it selects no backend
  alone) and `provider-local-vulkan`, `provider-local-metal`,
  `provider-local-cuda` and `provider-local-rocm`, each of which enables
  `provider-local-gpu` plus the matching burn backend. The graphics backends
  (wgpu: SPIR-V for Vulkan, MSL for Metal) are Rust, but every GPU feature
  compiles C: CubeCL's runtime pulls an embedded store (`turso`) that builds C
  through `cc`, and on Windows a dependency's build script runs the SDK
  resource compiler (`rc.exe`; `windres` on the GNU target). Metal runs on
  Apple only but compiles everywhere.
  CUDA and ROCm pull cubecl's LLVM backend: its build script downloads a
  prebuilt LLVM on the first build (network needed once, cached) and links it
  statically, and ROCm is Linux-only. The CPU path is unchanged (candle 0.11,
  same API and throughput), and nothing selects a GPU automatically.
- `LocalDevice` (`Cpu` default, `Vulkan(GpuAdapter)`, `Metal(GpuAdapter)`,
  `Cuda(usize)`, `Rocm(usize)`) and `GpuAdapter` (`Default`,
  `Discrete(usize)`, `Integrated(usize)`), re-exported from the crate root
  under `provider-local`. `LocalDevice::required_feature` names the cargo
  feature a device needs, and `LocalDevice` parses from `cpu`, `vulkan`,
  `vulkan:<n>`, `vulkan:integrated:<n>`, the same three forms for `metal`, and
  `cuda`, `cuda:<n>`, `rocm`, `rocm:<n>` (ASCII case-insensitive). Select one
  with `LocalModelConfig::device`. Every variant exists in every build:
  selecting a GPU whose feature is off fails `LocalModel::load` with a
  `CucaError::Config` before any thread starts.
- A GPU that cannot be opened (no adapter, missing driver) fails
  `LocalModel::load` with a `CucaError::Provider`. The engine thread now
  catches a panic during load and reports it as
  `local engine failed to initialize: <message>` instead of the opaque
  "local engine thread exited during load".
- GPU weight dtypes: matmul weights must be Q4_0, Q8_0, F32, F16 or BF16.
  Q4_0 and Q8_0 are repacked at load into burn's block-quantized layout and
  stay quantized in device memory; float weights are uploaded as f32. Any other
  dtype (K-quants, Q4_1, Q5_x) fails the load with an error naming the tensor;
  the CPU still reads those. All four published presets qualify. Device memory
  is about the file size minus the per-layer embedding table (still read from
  the file per token on the host), except `per_layer_model_proj`, which is held
  as f32 (92 MB for E2B), plus the K/V cache at the CPU formula.
- GPU activations stay in f32, while candle's CPU quantized matmul rounds
  activations to Q8_0, so GPU and CPU logits differ slightly; the greedy
  outputs of the live suite are identical. On Windows with an AMD RX 9070 XT
  over Vulkan, all five live `tests/provider_local.rs` scenarios pass. Metal,
  CUDA and ROCm are compile-checked only.
- `CUCA_LOCAL_DEVICE` (`LocalDevice` syntax) is read by `tests/provider_local.rs`
  and `examples/local_gemma.rs`, not by the library, to set
  `LocalModelConfig::device`; the example prints the device it loaded on. The
  unit test `gpu_tiny_model_matches_cpu` runs only when it names a GPU.
- docs.rs and CI feature lists: `[package.metadata.docs.rs]` lists every feature
  explicitly instead of `all-features`, omitting `provider-local-cuda` and
  `provider-local-rocm` (docs.rs builds have no network for the LLVM download).
  CI uses the same list (`ALL_NO_LLVM`) in place of `--all-features` for
  clippy, test and doc, and a new job, `local_gpu_llvm`, checks the CUDA and
  ROCm features on Linux.

## 0.5.0

### Breaking

- `ProviderEndpoint` has a new variant, `Local`. An exhaustive `match` on
  `ProviderEndpoint` must add an arm for it.

### Added

- `provider-local`: Gemma 4 (E2B and E4B, Q4_0 or Q8_0 presets; any dense
  `gemma4` GGUF loads) runs in-process on the CPU through
  candle 0.11, with no HTTP endpoint and no llama.cpp. The engine code is pure
  Rust with no FFI written by cuca; candle's dependency tree builds one C crate
  (`onig_sys`, via `tokenizers`), so a C compiler is required.
  `LocalModel::load(LocalModelConfig)` downloads the pinned model into a cache
  directory (resumable, SHA-256-verified) or opens a GGUF file; attach it with
  `CucaClientBuilder::with_local_model` and select `ProviderEndpoint::Local`.
  New public items: `LocalModel`, `LocalModelConfig`, `ModelSource`,
  `Gemma4Variant`, `Gemma4Quant`, `SamplingConfig`, `DownloadProgress`,
  `DownloadProgressCallback`, `CucaClient::local_model`. The engine is a single
  slot with a bounded queue, reuses the K/V cache across requests that share a
  prefix, is text-only, and reports exact token usage. See
  `examples/local_gemma.rs` and `tests/provider_local.rs`; build with
  `RUSTFLAGS="-C target-cpu=native"` for candle's SIMD kernels.
- `provider-local` model selection: `ModelSource::gemma4_it(variant, quant)`,
  `gemma4_e2b_it(quant)` and `gemma4_e4b_it(quant)` build verified presets
  (`ggml-org` repositories, pinned commit and SHA-256), `gemma4_e2b_it_q4_0()`
  stays the default, `ModelSource::hugging_face(repo, revision, file)` names any
  other file, and `ModelSource` parses from a preset name
  (`ModelSource::PRESET_NAMES`) or `repo:revision:file`. The loader derives
  every dimension from the GGUF metadata and tensor shapes, supports
  grouped-query attention (E4B has two KV heads), and reads the per-layer and
  token embedding rows in F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0 and the
  K-quants. MoE, per-layer KV head counts, differing key and value lengths and
  IQ-family quantizations fail to load with a `CucaError::Provider`. The K/V
  cache costs `context_length * 2 * 4 * sum(n_kv_head * head_dim)` bytes over
  the K/V-owning layers (302 MB at 8192 for E2B, 939 MB for E4B). The live
  tests and the example read `CUCA_LOCAL_MODEL` (preset name or
  `repo:revision:file`) next to `CUCA_LOCAL_MODEL_PATH`.

## 0.4.0

### Breaking

- `UnifiedResponse` has a new public field, `usage: Option<TokenUsage>`. Code
  that builds a `UnifiedResponse` with a struct literal must add `usage: None`.
  The field is `#[serde(default)]`, so serialized 0.3 responses still
  deserialize.
- `UnifiedResponse::completion_tokens` and `prompt_tokens` change meaning on
  the OpenAI-compatible adapters (OpenAI, DeepSeek, vLLM, LM Studio and the
  llama.cpp chat route). When the server reports usage, they are the provider's
  own token counts: `completion_tokens` is no longer one per `Text`, `Thinking`
  or `ToolCall` block, and `prompt_tokens` is no longer `0`. Without a
  reported usage, and on Gemini, Anthropic, the llama.cpp native `/completion`
  route and the speculative orchestrator, the block-count fallback stays and
  `usage` is `None`. `plugin-telemetry` and `plugin-session-log` now record the
  real counts; `CostPlugin` still uses its tiktoken estimates.

### Added

- Every OpenAI-compatible request body carries
  `"stream_options": {"include_usage": true}`, and `ChatCompletionTranslator`
  reads `usage` (including `completion_tokens_details.reasoning_tokens`) from
  whichever frame carries it. `ChatCompletionTranslator::take_usage` exposes
  it.
- `TokenUsage { prompt_tokens, completion_tokens, reasoning_tokens }`,
  exported from the crate root.
- `CucaClient::generate_stream_with_response` returns the block stream plus a
  `ResponseHandle`; `ResponseHandle::take` yields the call's terminal
  `UnifiedResponse` once the stream has ended. Unlike a plugin's
  `on_response_complete`, the handle belongs to one call, so concurrent calls
  stay apart.
- `UnifiedRequest::tools` reaches the wire in every provider adapter
  (OpenAI-compatible `tools`, Anthropic `tools`, Gemini
  `functionDeclarations`), without a tool-choice override.

### Fixed

- `ChatCompletionTranslator` no longer flushes a tool call on a
  `"finish_reason": null` frame, which emitted calls with `null` arguments.
- A usage frame that also drains a flushed tool call still records its usage.
