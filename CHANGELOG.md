# Changelog

## Unreleased

### Breaking

- `LocalModelConfig` has three new fields: `device: LocalDevice` (default
  `LocalDevice::Cpu`), `drafter: Option<DrafterSource>` (default `None`) and
  `draft_tokens: usize` (default 3). Code that builds `LocalModelConfig` with
  an exhaustive struct literal, without `..Default::default()`, must set them.
- `LocalModelConfig::context_length` is now `Option<usize>` (default `None`),
  not `usize`: `None` resolves to the smaller of 8192 and the model's own maximum,
  so models trained on fewer positions (OLMo 2, TinyLlama) load without
  configuration. Wrap an explicit value in `Some(..)`; an explicit value above the
  model's maximum still fails to load.
- `provider-local` is no longer Gemma-only, and its model selection API moved
  from Gemma-specific constructors to a preset table. Removed: `Gemma4Variant`,
  `Gemma4Quant`, `ModelSource::gemma4_it`, `gemma4_e2b_it`, `gemma4_e4b_it`,
  `gemma4_e2b_it_q4_0` and `ModelSource::PRESET_NAMES`. Use
  `ModelSource::PRESETS` (a slice of `ModelPreset`), `ModelSource::preset(name)`,
  `ModelPreset::source()`, `ModelSource::default_model()` (Gemma 4 E2B Q4_0, the
  `LocalModelConfig` default), or parse a name:
  `"gemma4-e4b-q4_0".parse::<ModelSource>()?`. The four Gemma 4 preset names are
  unchanged.
- The example `local_gemma` is now `local_model`
  (`cargo run --example local_model --features provider-local`;
  `examples/local_gemma.rs` is `examples/local_model.rs`).
- A GGUF whose architecture is not in the supported table, or whose
  `tokenizer.ggml.model` or `tokenizer.ggml.pre` the engine does not implement,
  still fails to load, but the error now names the value and lists what is
  supported. A mixture-of-experts checkpoint no longer fails by itself: the
  `granitemoe` architecture runs.

### Added

- `provider-local` is a generic GGUF engine, as llama.cpp's loader is:
  `general.architecture` selects the behaviour from one table, and every
  dimension still comes from the file. Supported: `gemma4`, `gemma3`, `gemma2`,
  `llama`, `qwen2`, `qwen3`, `phi3` (Phi 3.5 and 4-mini), `smollm3`, `granite`,
  `granitemoe` (a top-k mixture of experts), `olmo2`, `ernie4_5` and `qwen35`, plus
  the `gemma4-assistant` drafter. The forward
  pass gained per-layer head counts, fused QKV and gate/up projections, QKV
  biases, QK-norm, post-norms, sliding-window patterns, NORM and NEOX RoPE with
  partial rotary dimensions, linear, YaRN and LongRoPE scaling, NoPE layers,
  embedding, residual and logit scales, and expert routing, on the CPU and on
  every GPU backend. Qwen 3.5 (`qwen35`: gated delta-net layers with a
  recurrent state plus gated full attention, and an optional embedded
  multi-token-prediction block) runs on the CPU and is rejected on a GPU device
  with an error naming the cause; its recurrent state is checkpointed while
  prefilling (one checkpoint slot, about 61 MB with the verification copies for
  the 0.8B model) so prefix reuse and speculative-verification rollback stay
  exact, and its reference arithmetic matches HuggingFace within 5e-5 on the logits.
- Qwen 3.5 writes tool calls in the XML style
  (`<tool_call><function=name><parameter=key>value</parameter></function></tool_call>`);
  the output parser detects it from the template and reads each value as JSON when
  it is valid JSON and as a string otherwise.
- Tokenizers, in Rust and built from the GGUF: SentencePiece (`llama`),
  byte-level BPE (`gpt2`) with hand-written linear pre-tokenizers for the
  `tokenizer.ggml.pre` values of the Llama 3, Qwen, SmolLM, GPT-2, StarCoder,
  GPT-4o and DeepSeek families (a new dependency of `provider-local`:
  `unicode-general-category`, pure Rust), and Gemma 4's space-escaped scheme.
  Special tokens are cut out of the text as llama.cpp does, BOS/EOS follow the
  model's flags without doubling, and Phi-3-family `rstrip` is honoured. On the
  catalog models the token ids equal the reference tokenizers'.
- Chat templates render the GGUF's own Jinja text with no model-specific code:
  Python string methods, `raise_exception`, `strftime_now`, a `tojson` and a
  `string` filter with Python's formatting, `{% generation %}` tags, and a
  `tools` variable that exists only when the request declares tools.
- Tool declarations reach the template in OpenAI key order (`type`, then `function`
  with `name`, `description`, `parameters`) and `tojson` keeps key order, so the
  tool block of a prompt equals what Python renders; the order inside a tool's JSON
  schema is that of the request's `serde_json` map (sorted unless `preserve_order`).
- Reasoning and tool calls follow the model's template: reasoning blocks
  (`<|channel>thought`, `<think>`) become `Thinking` blocks and `reasoning_tokens`
  counts them; tool calls are parsed from Gemma 4's custom syntax, tagged JSON
  (Qwen, SmolLM, Granite), or bare JSON and `<|python_tag|>` (Llama 3.x), and the
  end-of-generation ids come from the metadata and the vocabulary. Markers are
  matched on text, so multi-token spellings work.
- Gemma 2 (attention and final logit soft-caps, alternating sliding window), OLMo 2
  (no pre-norms, full-width Q/K norm), dense Granite and ERNIE 4.5 run through the
  same architecture table as the others, on the CPU and on every GPU backend; on
  the six models checked against HuggingFace (Gemma 2 2B, OLMo 2 1B, Granite 3.3
  2B, ERNIE 4.5 0.3B, SmolLM 2 360M, TinyLlama 1.1B) the reference arithmetic
  matches exactly. ERNIE has no preset: the GGUFs that exist omit
  `tokenizer.ggml.add_space_prefix`, which changes its answers (see the docs).
  Chat templates that read the `thinking` variable (Granite 3.3) now report
  switchable reasoning and receive it from the request.
- A verified preset table, `ModelSource::PRESETS` (`ModelPreset`): Gemma 4 E2B and
  E4B, Gemma 2 2B, Gemma 3 270M and 1B, Qwen 3 0.6B and 1.7B, Qwen 2.5 0.5B,
  DeepSeek-R1-Distill-Qwen 1.5B, Qwen 3.5 0.8B, Llama 3.2 1B, TinyLlama 1.1B, SmolLM 2 360M and
  SmolLM 3 3B, Phi 3.5 mini, Phi 4 mini (two quantizations), OLMo 2 1B, Granite
  3.3 2B and Granite 3.1 MoE 1B, each pinned to a commit and a SHA-256, named
  `family-size-quant`. `LocalModel::kv_cache_bytes` reports the cache size, and `LocalModel::capabilities`
  returns `LocalCapabilities { reasoning, can_disable_thinking, tools }` read from
  the model's template and output conventions.
- An opt-in multi-token-prediction drafter for self-speculative decoding:
  `LocalModelConfig::drafter` (an `Option<DrafterSource>`: `File(ModelSource)`, such
  as the `gemma4-e2b-mtp-f16` preset for a Gemma 4 E2B target, or `InTrunk`, the
  head embedded in a Qwen 3.5 `*-MTP` file, CPU only; a `ModelSource` converts with
  `.into()`), `draft_tokens`, and `LocalModel::draft_stats` returning `DraftStats`
  (with `acceptance_rate`). Greedy output equals plain decoding
  (`tests/provider_local_mtp.rs` checks it; `examples/local_model.rs` reads
  `CUCA_LOCAL_DRAFTER`, a preset or `in-trunk`, and `CUCA_LOCAL_DRAFTER_PATH`).
  Measured on the CPU at three drafts per round: Qwen 3.5 0.8B Q8_0 with its
  in-trunk head 1.62x on code (72 % of the drafted tokens accepted), 1.18x on a list
  and 0.98x on prose; the Gemma 4 file drafter 1.17x, 1.37x and 1.00x on the same
  three kinds of text.
- On Vulkan, files that mix quantization schemes (Llama 3.2 and Phi 3.5 are Q4_0
  files with Q4_1 layers and a Q6_K head) no longer fail at a matmul launch: the
  loader holds the weights of the scheme with fewer bytes as exact f32 where burn's
  autotune could replay one scheme's kernel on the other, a few hundred MB more
  device memory on such files.
- `tests/provider_local.rs` is a multi-model matrix selected by `CUCA_LOCAL_MODEL`
  or `CUCA_LOCAL_MODEL_PATH`: what runs follows the loaded model's template, with
  exact usage counts, thinking on and off, tool round trips, a long-context
  needle and prefix reuse. `CUCA_LOCAL_REF_DIR` adds a prompt-token check against
  HuggingFace oracle dumps. The docs have a new page, "Local models and
  formats", with the preset catalog, measured K/V cost and the limits found.
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
- GPU weight dtypes: Q4_0 and Q8_0 matmul weights are repacked at load into
  burn's block-quantized layout and stay quantized in device memory; F32, F16 and
  BF16 weights are uploaded as f32. Any other dtype (Q4_1, Q5_x, Q8_1 and the K-quants)
  is dequantized at load and requantized to Q8_0, costing about its file size or
  less on the device, with a rounding error of at most half a Q8_0 step per
  element. Device memory
  is about the file size minus the per-layer embedding table of Gemma 4 (still
  read from the file per token on the host), plus the K/V cache at the CPU
  formula.
- GPU activations stay in f32, while candle's CPU quantized matmul rounds
  activations to Q8_0, so GPU and CPU logits differ slightly; the greedy
  outputs of the live suite are identical. On Windows with an AMD RX 9070 XT
  over Vulkan, the live `tests/provider_local.rs` matrix passes for Gemma 4 E2B
  Q4_0, Llama 3.2 1B, Phi 3.5 mini, Gemma 2 2B, OLMo 2 1B and Granite 3.3 2B. Metal,
  CUDA and ROCm are compile-checked only.
- `CUCA_LOCAL_DEVICE` (`LocalDevice` syntax) is read by `tests/provider_local.rs`
  and `examples/local_model.rs`, not by the library, to set
  `LocalModelConfig::device`; the example prints the device it loaded on. The
  GPU parity unit tests (`gpu_matches_cpu_*`, `gpu_drafter_matches_cpu_drafter`)
  run only when it names a GPU.
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
