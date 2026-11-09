# Changelog

## Unreleased

### Added

- `provider-local`: DeepSeek-V4.1 (`deepseek41`, DeepSeek-V4.1-Flash-1017 with FP8 dense and FP4
  expert weights) on the CPU engine and on one GPU. The generic loader reads it
  through `Behavior` flags: hyper-connection residual streams, latent attention with a per-head sink
  and a sliding window, compressed sparse attention (compressors, an indexer with hierarchical
  candidate blocks, cache source layers, FP8 and FP4 simulated quantization of the cached rows,
  grouped output projection), per-layer Engram token embeddings, a `sqrtsoftplus` router with a
  selection bias and SwiGLU clamps, and MXFP4 experts through the row-batched kernel
  (`model/qmm.rs`). The K/V pools gain compressed groups whose blocks span `16 * ratio` tokens. The
  GPU engine runs it with CubeCL kernels (`hc_kernel.rs`, `csa_kernel.rs`, `topk_kernel.rs`,
  `qdq_kernel.rs`, MXFP4 experts in `moe_kernel.rs`); experts can be host-tiered
  (`ExpertPlacement::Tiered` with `device_bytes = 0`), `gpu_layers` is accepted at cuts that keep
  every cache source with its readers, and tensor parallelism is rejected. The CPU engine is checked
  against a torch oracle on a tiny release-format checkpoint, and the GPU engine against the CPU
  engine on tiny synthetic models. The real DeepSeek-V4.1-Flash-1017 (default-precision set, experts
  tiered from an HDD-class volume) matches a torch streaming oracle on 16 prompt tokens in one pass,
  on the CPU (top-1 equal at 15 of 16 positions with the quantizers off, KL 8.9e-8 at the last
  position) and on Vulkan (same top-1 count, KL 1.9e-8), answers `Paris` and `72` to short chat
  prompts on both, and passes eight scenarios of the live suite on the CPU. It decodes at about 0.2
  tokens/s on either engine because the experts stream from disk.
- `provider-local`: `EngramCache` (`LocalModelConfig::engram`, parsed from `ram=<size>,threads=<n>`)
  bounds the Engram reader. The table (209 GB at the default Q8_0) stays in the GGUF; `threads`
  reader threads (1 to 64, default 16) fetch the distinct rows of each pass, and an LRU cache of
  `ram` bytes (1 MiB to 1 TiB, default 4 GiB, 48 bytes of bookkeeping per row counted) evicts the
  least recently used row at the bound. `LocalModel::engram_stats` returns `EngramStats` (lookups,
  cache hits, disk reads, bytes read, resident rows and bytes, capacity). Both engines prefetch the
  rows of a pass at its entry and of the next prefill chunk, and a model split by `gpu_layers` opens
  one table, one cache and one set of counters for both halves.
- `provider-local`: the DSpark block drafter of DeepSeek-V4.1 (a `dflash` GGUF with
  `dflash.hyper_connection.count`): stages of the target's own block kind, drafting a block of
  5 tokens per round with a Markov head and a confidence head, through `DrafterSource::File`.
  Speculative output equals plain greedy output. On the real target (CPU, one prompt) 5 of 15
  drafts were accepted and there was no speedup, since decoding is disk bound.
- `provider-local`: the GGUF reader is the crate's own (`src/provider/local/gguf/`). It reads the
  dtypes MXFP4, IQ4_NL and I8 beside candle's, keeps an unknown dtype id until its tensor is asked
  for, and opens a split set (`<base>-00001-of-0000N.gguf`) natively. Row decoding handles F32, F16,
  BF16, the legacy quantizations, IQ4_NL, MXFP4 and I8 in place.
- `provider-local`: the DeepSeek-V4.1 checkpoint converts once, in pure Rust, to a cached split set
  (`model-<quant>-00001-of-00003.gguf` holds the metadata, `-00002-` the backbone, `-00003-` the
  Engram table; two parts for a model without Engram layers) and its DSpark stages to
  `dspark-model-<quant>.gguf`. Matrices follow `quant` (dense FP8 becomes Q8_0 at the default), FP4
  experts are repacked to native MXFP4, `quant = f16` dequantizes exactly, and hyper-connection,
  router, sink and norm tensors stay
  F32. `ConvertConfig` gains `engram_quant` (`EngramQuant`: `Q8_0` by default, `Q4_0`, `Bf16`); a
  non-default value adds `-engram-<name>` to the cached file name. The conversion checks the free
  space of the output volume first and fails with a `CucaError::Config` naming the sizes. The
  converter is checked against the tiny torch oracle (bit-exact tensors and metadata) and its
  output on the full release checkpoint against the published ggml-org headers.
- `provider-local`: the builtin chat template `deepseek-v41`, DeepSeek's DSML tool-call syntax
  (`ToolSyntax::Dsml`, parsed by `parse_dsml_calls`), thinking on by default with `reasoning_budget`
  (`low` for a minimal or low effort, `high` for medium and above), the pre-tokenizers `joyai-llm`
  and `deepseek-v3` (a three-stage split chain), and JSON maps that keep the caller's key order
  (`serde_json/preserve_order` under `provider-local`).
- `provider-local` depends on `tokenizers` 0.22 (`onig`, already compiled for candle) for the
  converter's Engram token-map normalizers only, and on `sysinfo` for the converter's free-space
  check. The engine's tokenizer stays hand-written.
- `server` feature and the `cuca` binary (`src/bin/cuca.rs`, `required-features = ["server"]`) with
  `cuca serve`: an OpenAI-compatible HTTP server over `provider-local`, as vLLM and llama.cpp's
  server speak it. It loads one or more models concurrently (`--model [name=]source`, a preset, a
  GGUF path or Hugging Face file, a checkpoint, `--config` TOML `[[model]]` tables, global
  `--device`, `--context-length`, `--batching`, `--experts`, `--gpu-layers`, `--cache-dir`) and serves
  `GET /health` (503 until every model is ready), `/version`, `/v1/models` (with `max_model_len`),
  `POST /v1/chat/completions` (text, reasoning, tools, SSE with `include_usage`, exact usage with
  cached prompt tokens, `finish_reason` `stop`, `length` or `tool_calls`) and `/metrics`.
  Parameters the engine cannot honour (`n` above 1, `logprobs`, penalties, `logit_bias`,
  `response_format`, image parts) are a 400 naming the field; `/v1/completions` is not served.
  `--api-key` (or `CUCA_API_KEY`) requires a bearer key on every route but `/health`. Every request has an
  `x-request-id`, a tracing span and an access log line (`--log-format pretty|json`, `--log-level`).
  A client disconnect stops the generation. Library entry points are in `cuca::server`
  (`serve`, `ServerConfig`, `ModelSpec`, `ServerError`).
- `cuca balance`: a load balancer over several `cuca serve` instances. Routing policies are
  `prefix-affinity` (default: rendezvous hashing of the model and the prompt prefix, with a fall back
  to the least-loaded upstream above `2 * min_score + 2`), `least-loaded` and `round-robin`. Upstreams
  are polled for `/health`, `/v1/models` and the stats API; a connect error or a 5xx before the first
  byte ejects an upstream at once; requests retry at most twice on a connect error or a 503 before any
  body byte. Responses stream through unchanged, a client disconnect reaches the upstream, and the
  balancer serves `/health`, `/v1/models` (union), `/metrics` and its own stats API.
- `cuca tui` (`server-tui` feature, `ratatui` and `crossterm`): a live terminal view of a server or a
  balancer with resources, latency percentiles, the ranked reasons for the latency and a model or
  upstream list; Enter drills into an upstream's own view.
- Stats API (`--stats-api`): `GET /cuca/v1/stats` and `GET /cuca/v1/stats/stream` (SSE, one snapshot per
  second, `?interval_ms=` 250 to 10000) serve a `StatsSnapshot` (`cuca::server::stats`, serde
  types shared by the balancer and the TUI): node identity, per-model engine state, K/V pool, rates,
  memory, windowed latency percentiles, requests, the machine, ranked `Diagnosis` entries with
  evidence, and for a balancer the upstreams with their nested snapshots. Diagnosis causes cover
  queue saturation, K/V pressure, preemptions, prefill, batching, step time growth, CPU, memory
  paging, VRAM, GPU, disk reads and slow clients, and for a balancer no healthy upstream, a down,
  saturated or slow upstream and load imbalance.
- `GET /metrics`: hand-written Prometheus text with per-model request counters by status, TTFT,
  inter-token, end to end and queue histograms, token counters, engine gauges and counters, and the
  machine's CPU, memory, swap and paging, disk and GPU series; the balancer exports its own
  `cuca_balancer_*` series.
- System sampler (`cuca::server::sysmon`): one thread samples CPU, RAM, swap and paging rates, disks,
  this process and GPUs every second (`sysinfo`; WMI and the registry on Windows; `sysfs`, `fdinfo`
  and NVML on Linux; `ioreg` on macOS). Only the Windows sampler has run; the Linux and macOS
  samplers are compile-checked.
- `provider-local`: `LocalModel::generate` runs one request without a `CucaClient` and returns a
  `LocalGeneration` stream whose `report()` is a `GenerationReport` (exact `usage`, `prompt_cache`,
  `finish_reason` `stop` or `length`, `queue_secs`). `LocalModel::context_length` and
  `weights_file_bytes` are new accessors.
- `ServingStats` gains `max_sequences`, `used_blocks`, `total_blocks`, `prefill_rows`,
  `generated_tokens`, `admissions`, `queue_wait_ns`, `step_ns`, `last_step_ns` and `last_step_rows`,
  all atomics on the engine thread.

### Changed

- `provider-local`: a split GGUF set is no longer merged. `ModelSource::File` naming any part opens
  the set from its parts, which are looked for next to the named file and in
  `LocalModelConfig::shard_dirs` (new), and a Hugging Face source whose file is a split name
  downloads every part into the cache and keeps them there. A split model needs its own size on
  disk and no more.
- `provider-local`: the row-batched quantized matmul (`model/qmm.rs`) decodes four weight rows once
  into `i16` lane pairs and multiply-adds them against every activation tile through `wide`
  (`provider-local` now owns that dependency), and quantizes the activations over the pool. Its
  results are unchanged bit for bit and a test checks them against the scalar definition. A
  2,014-token prompt on the real DeepSeek-V4.1-Flash-1017 from an SSD prefills in 147.4 s against
  276.1 s on the CPU; one MXFP4 expert matrix of 41 rows takes 0.84 ms against 4.96 ms.
- `provider-local`: the expert loader reads missing experts that are adjacent in the file as one run
  of up to 4 per matrix (layers that store their experts expert after expert; the bytes, counters and
  budgets are the single reads'). Cold expert reads from the SSD rise from 3.41 to 3.50 GB/s.

## 0.7.0 - 2026-11-07

### Added

- `UnifiedRequest` has the per request fields `top_p`, `top_k`, `seed` and `stop` (`set_top_p`,
  `set_top_k`, `set_seed`, `add_stop`). Every provider sends them under its own keys: `top_p`,
  `seed` and `stop` on OpenAI-compatible routes (plus `top_k` on llama.cpp, vLLM and LM Studio),
  `top_p`, `top_k` and `stop_sequences` on Anthropic, `topP`, `topK`, `seed` and `stopSequences` in
  Gemini's `generationConfig`, and all four on llama.cpp's native route and in the local engine. A
  backend without an equivalent fails with `CucaError::Config`: `top_k` on OpenAI, DeepSeek's native
  route and custom endpoints, `seed` on Anthropic and DeepSeek's Anthropic bridge. Breaking: a
  `UnifiedRequest { .. }` struct literal must add the four fields (serde defaults keep stored JSON
  loading), and the `PromptCache` digest includes them, so every key changes and importing a
  snapshot exported by 0.6.2 fails with `PromptCacheError::Validation`.
- `provider-local`: Hugging Face checkpoints load without a GGUF. `ModelSource::Checkpoint { repo,
  revision }` (parsed from `repo:revision`) and `ModelSource::Dir(path)` are converted once, in
  process and in pure Rust, to a GGUF cached under `<cache>/converted/`, by a port of llama.cpp's
  `convert_hf_to_gguf.py` that covers every engine architecture. `LocalModelConfig::convert`
  (`ConvertConfig { quant: WeightQuant }`: `F16`, `Q8_0` by default, `Q4_0`) picks the precision,
  and `on_download_progress` reports the tensors written. Breaking: `ModelSource` gains the variants
  `Checkpoint` and `Dir`.
- `provider-local`: split GGUF files (`llama-gguf-split`, `<base>-00001-of-0000N.gguf`) load.
  `ModelSource::File` naming any part merges the set into one GGUF under `<cache>/merged/`, and a
  Hugging Face source whose file is a split name downloads and verifies every part, merges them
  into `<base>.gguf` and deletes the parts; the merge needs about twice the model's size free and
  `on_download_progress` reports it. Breaking: `ModelSource::HuggingFace` gains `parts:
  Vec<ModelPart>` and `ModelPreset` gains `parts: &[PresetPart]`, set for
  `mistral-small-4-119b-q4_0`.
- `provider-local-gpu`: tensor parallelism. `LocalModelConfig::shards` (or
  `LocalModelConfig::set_devices("cuda:0+cuda:1")`) splits a model over `1 + shards.len()` GPUs of
  one backend: attention heads and FFN widths are divided evenly, each device holds its slice of the
  weights and of the K/V pools, and the partial outputs are summed twice per layer. The sum runs on
  NCCL for distinct CUDA devices on Linux (compile-checked, not run) and host-staged in a fixed
  order everywhere else, so runs are deterministic, and a device may repeat (`vulkan:0+vulkan:0`).
  The load fails with `CucaError::Config` for a drafter, tiered or host experts, delta-net layers
  (`qwen35`, `qwen35moe`), latent attention (`mistral4`), full-width Q/K norms (`olmo2`, `olmoe`)
  and models whose head count or widths do not divide.
- `provider-local-gpu`: `LocalModelConfig::gpu_layers` (llama.cpp's `-ngl`) runs the last `n`
  layers and the LM head on the GPU and the first layers and the embeddings on the CPU, so the
  hidden state crosses once per forward pass. Each layer's K/V pool and recurrent state live on its
  layer's device, tiered experts split between the two halves, and `LocalModel::layer_split()`
  returns `(layers on the GPU, layers in all)`. The load fails with `CucaError::Config` on the CPU
  device or combined with `shards` or a drafter.
- `provider-local-gpu`: `ExpertPlacement::Host { layers }` (`host`, `host:layers=<n>`; llama.cpp's
  `--cpu-moe` and `--n-cpu-moe`) keeps the routed experts of the MoE layers in host RAM and computes
  them on the CPU while the GPU runs the rest, with one readback and one upload per MoE layer per
  pass. It needs a GPU device and a mixture of experts model and does not combine with tiered
  experts or `shards`. Breaking: `ExpertPlacement` gains the variant `Host`.
- `provider-local`: Muse Glimmer 30B (`muse-glimmer`, preset `muse-glimmer-30b-q4_0`, Q4_0 from
  `bartowski/Muse-Glimmer-30B-GGUF`) on the CPU and the GPU engine: a dense decoder with sliding
  window and full attention layers and a sigmoid attention gate per layer. The output parser reads
  its channel protocol (a message addressed to `self` becomes `Thinking`, one addressed to `user`
  text, any other recipient a tool call), and `ThinkingEffort` selects the template's
  `reasoning_strength`; the model cannot switch reasoning off, so `can_disable_thinking` is false.
  The converter reads `MuseGlimmerForConditionalGeneration` checkpoints.
- `provider-local`: Ministral 3 and Devstral Small 2 (`mistral3`; presets
  `ministral3-3b-reasoning-q4_0` and `ministral3-3b-instruct-q4_0`, Q4_0 from bartowski) on the CPU
  and the GPU engine: the `llama` graph with YaRN and the Llama 4 query scaling
  (`attention.temperature_scale`, over the YaRN original context length). The converter reads
  `Ministral3ForCausalLM` and `Mistral3ForConditionalGeneration` with a `ministral3` or `mistral`
  text model (Mistral Small 3.1 and 3.2, Magistral 2509) and writes `mistral3`.
- `provider-local`: Mistral Small 4 (`mistral4`, preset `mistral-small-4-119b-q4_0`, bartowski's
  two part Q4_0) on the CPU and the GPU engine, with resident, tiered or (on a GPU) host experts. It
  uses multi-head latent attention (one `kv_lora + rot` row of 320 floats per position and layer,
  1,280 bytes, the value being a prefix of the key row), fused gate/up experts, an ungated shared
  expert, `expert_weights_scale` and leading dense blocks. Tensor parallelism rejects it.
- `provider-local`: the output parser reads Mistral's reasoning markers `[THINK]` and `[/THINK]` and
  its tool syntax `[TOOL_CALLS]name[ARGS]{json}`, detected from the chat template and the
  vocabulary (Mistral Small 3.2 and 4, Devstral, Ministral 3). A call has no closing marker and ends
  at the next opener or at the end of the reply, and `[CALL_ID]id` becomes the call's id. A template
  that reads `reasoning_effort` (`none` or `high`) is driven by the request's thinking setting.
- `provider-local`: the checkpoint converter reads Mistral Small 4 (`Mistral3ForConditionalGeneration`
  with a `mistral4` text model, written in llama.cpp's `mistral4` layout) and `MistralForCausalLM`
  (as `llama`). `F8_E4M3` weights are dequantized while converting (times their `*_scale_inv`, per
  tensor or per expert) and written at the chosen precision; block scaled FP8 is refused up front.
  The `tokenizer.ggml.pre` name follows llama.cpp's probe-text hash table when the engine implements
  it (`pixtral` for Mistral Small 4).
- `provider-local`: the `tekken` pre-tokenizer (Mistral Nemo, Pixtral, Mistral Small 3 and 4); the
  name `pixtral` selects it, with `ignore_merges` and a default BOS. Checkpoints that ship only
  `tekken.json` (Mistral Small 3.0 to 3.2, Magistral, Devstral Small) convert as llama.cpp's
  `MistralVocab` path does: the ranked byte vocabulary becomes a `gpt2` vocabulary with the `tekken`
  pre-tokenizer and rebuilt merges, and the chat template is the one llama.cpp bundles for tokenizer
  version v11 or v13, or the name `mistral-v7-tekken` for v7. Any other tokenizer version is
  refused.
- `provider-local`: `LocalModelConfig::chat_template` replaces the GGUF's chat template (llama.cpp's
  `--chat-template-file`, vLLM's `--chat-template`); the reasoning markers, tool syntax and end of
  turn tokens follow it, and a template that does not compile or cannot render a one message
  conversation fails `LocalModel::load` with `CucaError::Config`. A preset pins one with
  `ModelPreset::template` (`PresetTemplate { repo, revision, file, sha256 }`, downloaded and verified
  like a model file): `ministral3-3b-reasoning-q4_0`, `ministral3-3b-instruct-q4_0` and
  `muse-glimmer-30b-q4_0` pin the `chat_template.jinja` of the model's own repository, and
  `CUCA_LOCAL_CHAT_TEMPLATE=<path>` sets one in `examples/local_model.rs`. Breaking:
  `ModelSource::HuggingFace` gains `template: Option<Box<ModelTemplate>>` and `ModelPreset` gains
  `template`.
- `provider-local`: llama.cpp's named chat templates `mistral-v7` and `mistral-v7-tekken` (what its
  converter writes for Mistral Small 3.0 and 3.1 and Devstral Small 2505) expand to Jinja at load,
  from `tokenizer.chat_template` or `LocalModelConfig::chat_template`, and render as llama.cpp's
  `llm_chat_apply_template` does (system message in place, no BOS, no generation prompt, tools
  ignored). Any other bare name is a load error that lists the supported ones.
- `provider-local`: the model file's sampling defaults apply. The GGUF keys `general.sampling.temp`,
  `general.sampling.top_k` and `general.sampling.top_p` are read at load, and the converter writes
  them from `generation_config.json`. Each value resolves from the request, then `SamplingConfig`,
  then the file, then the built-in default, and `LocalModel::sampling()` reports the result.
  Breaking: `SamplingConfig::temperature`, `top_k` and `top_p` are `Option` (`None` defers to the
  file), and the built-in values are the constants `SamplingConfig::DEFAULT_TEMPERATURE` (1.0),
  `DEFAULT_TOP_K` (64) and `DEFAULT_TOP_P` (0.95).
- `provider-local`: the Mistral Small 4 EAGLE drafter (`mistralai/Mistral-Small-4-119B-2603-eagle`,
  architecture `mistral4-eagle`) as `DrafterSource::File` with a checkpoint source
  (`mistralai/Mistral-Small-4-119B-2603-eagle:<revision>`). The converter turns its Mistral-format
  files (`params.json`, FP8 `consolidated.safetensors`) into a GGUF at `convert.quant`; the drafter
  is two latent attention layers over a dense FFN that use the target's embedding and LM head, with
  a K/V cache of its own. It pairs with a `mistral4` target on the CPU only and serves one sequence
  at a time.
- `provider-local`: a DFlash block drafter as `DrafterSource::File`: the preset
  `muse-glimmer-30b-dflash-q4_0` (llama.cpp's `dflash-Muse-Glimmer-30B-Q4_0.gguf`), another `dflash`
  GGUF, or a checkpoint such as `meta-models/Muse-Glimmer-30B-assistant` converted at
  `convert.quant`. It proposes up to 15 tokens per round in one pass from the target's hidden
  features at five layers, keeps a sliding window K/V ring of its own, and runs on the CPU only with
  one sequence at a time.
- `provider-local`: preset `r1-distill-llama-8b-q4_0` (`unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF`,
  Q4_0, `llama` architecture).

### Changed

- `provider-local`: the K/V cache is paged. Blocks of 16 positions live in per layer group pools,
  and a prefix store names full blocks by a hash chain over their token ids, so a finished
  request's blocks stay cached (least recently used evicted first) and an unrelated request does
  not evict a prefix. A hybrid model (Qwen 3.5) resumes only from a block boundary that has a
  recurrent state snapshot, taken below the last end of turn token of each prompt. Breaking:
  `prompt_cache_usage.read_tokens` is a multiple of 16 (the trailing partial block, at most 15
  tokens, is recomputed); the pool of a full attention layer group holds twice `context_length`
  positions, so `LocalModel::kv_cache_bytes` grows (Gemma 4 E2B at 8192: 481 MB); and CPU attention
  sums in a different order than candle's matmul attention, so a greedy reply can differ from 0.6.2
  where two logits are close.
- `provider-local`: continuous batching. The engine is a step scheduler: up to
  `BatchingConfig::max_sequences` requests (default 4) run at once, every step is one forward pass
  over a decode row per running sequence plus prompt chunks within `max_step_tokens` rows, and all
  sequences share one K/V block pool (`kv_tokens`, default twice `context_length`, so more
  sequences do not grow it). A request that needs a block when the pool is full preempts the
  youngest running one, which resumes from the prefix store, and a slow consumer stalls only its own
  request. `BatchingConfig` parses from `seqs=4,kv=16384,step=256,prefix=off`
  (`CUCA_LOCAL_BATCHING` in `examples/local_model.rs`) and `LocalModel::serving_stats` reports the
  scheduler; a request speculates only while it runs alone, and an in-trunk, EAGLE or DFlash drafter
  limits the engine to one sequence. On the CPU a sequence's output equals what it produces alone,
  bit for bit.
- `provider-local`: `LocalModelConfig` has the new public fields `chat_template`, `convert`,
  `batching`, `shards` and `gpu_layers`, and the crate root exports `BatchingConfig`,
  `ConvertConfig`, `ModelPart`, `ModelTemplate`, `PresetPart`, `PresetTemplate`, `ServingStats` and
  `WeightQuant`. Breaking: code that builds `LocalModelConfig` with a struct literal, without
  `..Default::default()`, must set the new fields.
- `provider-local`: `rope.scaling.yarn_log_multiplier` sets the YaRN cos/sin magnitude for every
  architecture, as llama.cpp does (`get_mscale(f, 1) / get_mscale(f, m)`); a file without the key
  keeps ggml's `1 + 0.1 ln f` boost. A `llama` file that carries the key (Devstral 2 123B) gets a
  different magnitude, so its logits change.
- `provider-local`: the batched CPU quantized matmul (`model/qmm.rs`) serves Q4_0, Q4_1, Q5_0, Q5_1
  and Q8_0 weights, and a row's result is bit-identical alone or in any batch. Q4_1, Q5_0 and Q5_1
  run 2 to 4 times faster than candle's per row kernel from 4 rows on (measured on a Ryzen 9
  9950X3D2). K-quants and every other dtype stay on candle's per row kernel, which is also row
  independent.
- `provider-local-gpu`: dense weights (attention projections, dense FFNs, the LM head) are
  multiplied by two CubeCL kernels of this crate: a matrix-vector kernel with f32 accumulation for
  up to 64 input rows (decode, batched decode, verification, prompt tails) and a dequantizing slice
  kernel that feeds burn's float matmul on longer prefill chunks. Q4_0, Q4_1 and Q8_0 weights are
  held as they are and other quantized dtypes are requantized to Q8_0; burn's quantized matmul is
  not used. A weight larger than 256 MiB on the device is split into row chunks.
- `provider-local`: checkpoint conversion encodes `q4_0`, `q8_0` and `f16` tensors on every core and
  reads the next block while the current one is written, with identical output bytes.
- `provider-local`: the `olmoe-1b-7b-q4_0` preset pins `allenai/OLMoE-1B-7B-0125-Instruct-GGUF`
  (Q4_0, revision `2ac1d273`). Breaking: it downloads a different file than 0.6.2's 0924
  `bartowski` one, so replies can change.

### Removed

Models first released before 2025 and the engine code only they needed.

- `provider-local` presets `gemma2-2b-q4_0`, `qwen2.5-0.5b-q8_0`, `llama3.2-1b-q4_0`,
  `phi3.5-mini-q4_0`, `smollm2-360m-q8_0`, `tinyllama-1.1b-q4_0` and `granite3.1-moe-1b-q4_0` (all
  first released between December 2023 and December 2024). Any GGUF of those models still loads
  when its architecture is supported (`llama`, `qwen2`, `phi3`).
- The `gemma2` and `granitemoe` architectures: a GGUF with that `general.architecture` fails with
  the unknown architecture error, and the checkpoint classes `Gemma2ForCausalLM` and
  `GraniteMoeForCausalLM` fail with the no converter error, both naming the value. The Granite MoE
  `block_sparse_moe` expert tensors leave the converter with them.
- Attention logit soft-capping (the `{arch}.attn_logit_softcapping` key): only Gemma 2 used it. The
  final logit soft-cap of Gemma 4 stays.
- The Llama 3 tool call syntax (`ToolSyntax::LlamaJson`: a bare `{"name": ..., "parameters": ...}`
  reply and the `<|python_tag|>` body that became a `python` call); only Llama 3.2 used it. A model
  with such a template answers in plain text. The tagged JSON syntax still reads `parameters` as an
  alias of `arguments`.
- The pre-tokenizer names `smollm`, `megrez`, `stablelm2`, `phi-2`, `mpt`, `jais`, `jina-es`,
  `jina-de`, `jina-v2-es`, `jina-v2-de`, `codeshell` and `minerva-7b`, whose families were first
  released before 2025. `unsupported pre-tokenizer` names them.
- The converter's Phi 3 SentencePiece vocabulary reader (a Phi 3 or 3.5 SentencePiece checkpoint
  fails with `CucaError::Config`; Phi 4 mini, which has a GPT2 tokenizer, converts), its SmolLM
  `add_bos_token` override and the tokenizer's special token `lstrip` and `rstrip` rule, which only
  Phi 3 SentencePiece files used.

### Fixed

- `provider-local`: a speculative round whose accepted draft ends the reply (end of generation, a
  stop string or the token budget) does not cache that draft. The accepted drafts are emitted like
  sampled tokens before the round is cached, so the K/V rows, the cached ids and the published
  prefix blocks end where plain decoding's do.

## 0.6.2

### Changed

- `provider-local` (GPU): mixture-of-experts layers (qwen3moe, olmoe,
  granitemoe, qwen35moe) run in fused CubeCL kernels instead of one burn
  matmul triple per selected expert (`moe_kernel.rs`, safe Rust, attached
  through burn's `backend_extension`; new direct dependencies `cubecl` and
  `burn-cubecl`, burn features `extension` and `cubecl`). Routing (softmax
  and top-k) runs on the device and a resident layer never reads it back,
  so a resident MoE token has no per-layer host wait; a tiered layer reads
  back only the router output and loads missing experts into a device slot
  pool, computing in groups that do not change a bit of the result. Q4_1
  experts are read as Q4_1 (the 5 Q4_1 down layers of Qwen3.6-35B-A3B were
  held as f32, 4.2 MB per expert); `plan_dense` no longer sees expert
  tensors. On a RX 9070 XT, against the previous commit: OLMoE resident
  ~15 -> ~44 tok/s, Qwen3.6-35B-A3B at `tiered:device=4g,ram=12g` 3.7/4.5 ->
  6.5/8.3 tok/s cold/warm (a long reply 5.9 -> 10.1), at `device=6g,ram=12g`
  5.3 -> 9.6, Qwen3-30B-A3B at `device=6g,ram=12g` 2.7/2.9 -> 4.9/6.1. A
  tiered load now agrees with the resident one to float rounding instead of
  bit for bit on a GPU (tiered placements still agree among themselves bit
  for bit).

- `provider-local` (GPU): faster decode of mixture-of-experts and hybrid
  models. The gated delta-net step queues about half the operations (one
  product-and-sum convolution for a single token, one normalisation of the
  query and key heads, value heads broadcasting their key head instead of a
  repeat, `ssm_beta` and `ssm_alpha` as one f32 matmul), a single token sums
  its experts' outputs without a zero tensor, and an expert uploaded from the
  RAM tier shares the tier's bytes instead of copying them. On a RX 9070 XT,
  Qwen3.6-35B-A3B at `tiered:device=4g,ram=12g` went from 2.9 cold / 3.3 warm
  to 3.7 / 4.5 tok/s (a long reply at `device=6g,ram=12g`: 4.6-4.8 to 5.3),
  Qwen3-30B-A3B at `device=6g,ram=12g` from 2.3 / 2.5 to 2.7 / 2.9. Decode is
  bound by the host: one readback of the router logits per layer costs about
  1 ms however little it carries. Output is unchanged up to float rounding.

### Added

- `provider-local`: architecture `qwen35moe` (Qwen3.6-35B-A3B): the `qwen35`
  hybrid stack with a mixture of experts in every layer (softmax router,
  renormalised top-k) plus a shared expert scaled by a sigmoid gate. It runs
  on the CPU and on a GPU, with resident or tiered experts; the prediction block in the file
  is ignored, and `DrafterSource::InTrunk` on it fails with
  `CucaError::Config`. Preset `qwen3.6-35b-a3b-q4_0`.
- `provider-local-gpu`: gated delta-net (hybrid `qwen35` and `qwen35moe`)
  layers, gated attention and the shared expert run on the GPU. The
  recurrent state is kept on the device with a free checkpoint and the CPU's
  `truncate_cache` fallback (no verify record). The `qwen35` GPU load error
  is gone.

### Fixed

- `provider-local`: gated delta-net layers with fewer key heads than value
  heads (`ssm.group_count` < `ssm.time_step_rank`) read key head `h %
  group_count` for value head `h`, the tiled order llama.cpp's converter
  writes; they used the HuggingFace grouped order before. Qwen3.5 0.8B has
  equal head counts and is unaffected.
- `provider-local-gpu`: `plan_dense` weighs every expert of a stacked expert
  tensor, so a MoE model whose Q8 weights outweigh one expert of each Q4
  tensor converts those Q8 weights to f32 instead of its Q4 experts (the
  Qwen3.6-35B-A3B experts would have cost 7 times the device memory).

## 0.6.1

### Added

- `provider-local`: tiered expert placement for mixture-of-experts models
  (`LocalModelConfig::experts`, `ExpertPlacement::{Resident, Tiered}`,
  `ExpertTiers`). Routed experts stay in the GGUF file and load on demand
  through bounded RAM and (on a GPU) device caches with learned heat; output
  equals the resident mode on the same device. `LocalModel::expert_tier_stats`
  and `LocalModel::expert_heat` (`ExpertTierStats`, `ExpertHeat`, reusable via
  `ExpertTiers::warm_from`).
- `provider-local`: architectures `qwen3moe` and `olmoe`, presets
  `qwen3-30b-a3b-q4_0` and `olmoe-1b-7b-q4_0`.

### Fixed

- GPU: an expert with one row in a multi-row pass no longer loses the Vulkan
  device; dropping a GPU model returns its device memory to the driver.
- CI: the test job builds without debug info so it fits the runner's disk.

## 0.6.0

### Added

- `service-scheduler` feature (owns `tokio` and `tracing`): a per-backend call
  scheduler and a dispatcher over several backends, in
  `services::scheduler`. It is an explicit-call service, so it never
  implements `CucaPlugin`.
- `Scheduler`, `SchedulerConfig`, `SchedulerError`: a priority queue per
  backend with aging (a waiting call gains one point per completed call),
  `Priority` (`Low`, `Medium`, `High`, `Urgent`), a reserved `Lane` served
  first from a configurable running level, `max_waiting` refusal with
  `SchedulerError::QueueFull`, and `SchedulerSnapshot` with running, waiting
  and recent calls, buckets and per-kind totals (at most 32 kinds, the rest
  counted as `"other"`).
- Adaptive parallelism tuner per backend, reported by `TunerSnapshot`,
  `TunerPhase`, `LevelChange`, `ChangeReason`, `Direction` and `Metric`.
- `CallMeta`, `CallSlot`, `CallReport`, `CallOutcome`: the call description
  and the slot a caller must finish.
- `DispatchGate` and `OpenGate`: a caller-owned pause and stale-task check
  shared by every queue. `Clock` and `SystemClock` make time injectable.
- `Dispatcher`, `DispatcherConfig`, `DispatchError`, `Completion`:
  least-load placement over `BackendRegistry` backends, failover when a
  backend is unreachable, a circuit breaker with an unreachable cooldown,
  periodic `rebalance` of waiting calls, `set_backends` to swap the list, and
  a per-call timeout.
- `BackendSpec`, `BackendClient`, `ClientFactory`, `Backend`: a backend holds
  a ready `CucaClient` or a factory that builds it once before the first
  slot is reserved, with an optional model tag and thinking override.
- `tracing` debug spans `scheduler.wait` and `scheduler.stream` under the
  target `cuca::scheduler`.
- `examples/scheduler.rs`, `tests/service_scheduler.rs` and the Scheduler
  page of the documentation site.

### Changed

- Local provider: a speculative round always verifies its first draft and
  adds each further one only while the drafter's calibrated confidence makes
  it pay for its measured verification cost. `LocalModelConfig::draft_tokens`
  is now the cap, default 5 (was 3). A seeded sampling run still drafts the
  full cap.
- Local provider: sliding-window layers hold at most `window - 1 + 1280`
  positions (CPU and GPU) and are compacted before they overflow; a
  truncation below a compacted layer's floor empties the cache.
  `LocalModel::kv_cache_bytes` reports the bounded size (Gemma 4 E2B at 8192
  positions: 145 MB, was 302 MB).
- Local provider: row-exact attention runs on candle's worker pool instead
  of spawning threads per call, and `top_k = 0` sampling no longer sorts the
  vocabulary.

## 0.5.2

### Changed

- Rust best-practices and performance review round: fewer clones and allocations in providers, the local engine, plugins and services.
- Prompt-cache LRU eviction is now O(log n).
- Lint suppressions moved to `#[expect]` with cleanups.

## 0.5.1

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
