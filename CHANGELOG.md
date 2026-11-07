# Changelog

## Unreleased

### Added

- `provider-local`: `LocalModelConfig::chat_template` replaces the GGUF's chat template (llama.cpp's `--chat-template-file`, vLLM's `--chat-template`); the reasoning markers, tool syntax, end-of-turn tokens and the `enable_thinking`/`reasoning_effort`/`reasoning_strength` variables follow it, and a template that does not compile or cannot render a one-message conversation fails `LocalModel::load` with `CucaError::Config`. A preset can pin one: `ModelPreset::template` (`PresetTemplate { repo, revision, file, sha256 }`, downloaded and verified like a model file; `ModelSource::HuggingFace` gains `template: Option<Box<ModelTemplate>>`, a breaking change for code that builds the variant). `ministral3-3b-reasoning-q4_0` pins `chat_template.jinja` of `mistralai/Ministral-3-3B-Reasoning-2512` at `4a36357c` (every published GGUF of that model embeds the older December 2025 template) and `muse-glimmer-30b-q4_0` pins the one of `meta-models/Muse-Glimmer-30B` at `cec94fcf` (the embedded one differs when a system prompt mentions "Reasoning effort"). `CUCA_LOCAL_CHAT_TEMPLATE=<path>` sets it in the live tests and `examples/local_model.rs`. The Ministral template renders byte for byte like Python jinja2 on 44 cases.
- `provider-local`: the model file's sampling defaults apply. The GGUF keys `general.sampling.temp`, `top_k` and `top_p` (llama.cpp's names) are read at load, and the converter writes `general.sampling.*` from `generation_config.json` exactly as llama.cpp's does (identical to the pinned converter on a Mistral Small 3.2 truncation). Precedence per value: request, then `SamplingConfig`, then the file, then the built-in default. `SamplingConfig`'s `temperature`, `top_k` and `top_p` are now `Option` (`None` = not set; the constants `DEFAULT_TEMPERATURE`, `DEFAULT_TOP_K`, `DEFAULT_TOP_P` hold the built-in values) and `LocalModel::sampling()` reports the resolved values. `min_p` is not added: no supported model's generation config sets it.
- `provider-local`: checkpoints that ship only `tekken.json` (Mistral Small 3.1/3.2, Magistral 2506-2509, Devstral Small 2505/2507) convert, as llama.cpp's `MistralVocab` path does: the 1,000 special tokens and the ranked byte vocabulary become a `gpt2` vocabulary with the `tekken` pre-tokenizer, the BPE merges are rebuilt from the ranks, and the chat template is the community one llama.cpp bundles for tokenizer version v11 or v13 (v1/v3/v7, whose template llama.cpp only names, and other versions are refused). Level 0 against llama.cpp `67107ac3` on a 2-layer Mistral Small 3.2 truncation: 0 differences in 35 keys and 21 tensors; the converted vocabulary encodes 1,536 texts to the same ids as `mistral_common`.
- `provider-local`: Ministral 3 and Devstral Small 2 (`mistral3`, presets `ministral3-3b-reasoning-q4_0` and `ministral3-3b-instruct-q4_0`, bartowski's Q4_0, 2.05 GB each) on the CPU and the GPU engine, by table flags only: the `llama` graph plus the Llama 4 query scaling (`attention.temperature_scale`, length = the YaRN original context; `Hparams::attn_temp`, shared with `mistral4`). `rope.scaling.yarn_log_multiplier` now sets the YaRN cos/sin magnitude for every architecture (`get_mscale(f, 1) / get_mscale(f, m)`, llama.cpp's rule; before, such a `llama` file, e.g. Devstral 2 123B, got ggml's `1 + 0.1 ln f` boost). The converter reads `Ministral3ForCausalLM` and `Mistral3ForConditionalGeneration` with a `ministral3` text model (as `mistral3`, with the log multiplier and temperature written as llama.cpp does: zero-problem diff on 2 layers of Ministral-3-3B-Reasoning) and with a `mistral` text model (Mistral Small 3.1/3.2, Magistral 2509; also `mistral3`, as llama.cpp's converter writes them; their published GGUFs say `llama` and load as before). Verified: the converted 2-layer cut against `Ministral3ForCausalLM` in f32 at 17,000 tokens (2.8e-5, greedy equal; a reference without the query scaling differs by 0.5 from 16,384), a tiny model through the CPU/GPU comparison and the batching invariants, and the live CPU matrix of the preset (reasoning and the rest pass; the two tool scenarios are known limits shared with HuggingFace f32) next to the Instruct preset passing all eight scenarios including the `[TOOL_CALLS]` round trip.
- `provider-local`: Mistral Small 4 (`mistral4`, preset `mistral-small-4-119b-q4_0`, bartowski's two-part Q4_0, 67.71 GB) on the CPU, resident or with tiered experts. Multi-head latent attention: the per-head keys and values are never built; the key up-projection is absorbed into the query and the K/V cache is one `kv_lora + rot` row (320 floats) per position and layer, 1,280 bytes, 46,080 over the 36 layers (755 MB for the default pool; 1.18 MB per position for the expanded heads), the value being the first 256 columns of the key row (`KvLayer` holds no value pool). YaRN keeps its magnitude scale out of cos/sin and puts `mscale^2` into the softmax scale as llama.cpp's `deepseek2` graph does; the Llama 4 per-position query scaling (`1 + 0.1 ln(1 + floor(pos / 8192))`) uses the file's `attention.temperature_scale`/`temperature_length` or, absent, the checkpoint's own 0.1 and 8,192 (which llama.cpp does not apply). The mixture of experts takes a fused `ffn_gate_up_exps` tensor (tiers read it as two strided halves: `ExpertSlice::stride`), an ungated shared expert, `expert_weights_scale` (`MoeSpec::scale`, applied by the router for every architecture) and leading dense blocks (`leading_dense_block_count`). Verified: an f64 implementation of transformers' expanded attention, the MoE FFN, fused-equals-split and tiered-equals-resident on tiny models; the real model's first two layers against `Mistral4ForCausalLM` in f32 (maximum absolute logit difference 4.1e-5 at 300 tokens and 7.0e-5 at 8,400, greedy equal; transformers' own float32 rotary angles show 1.1e-3 at 8k, the reference was run with float64 angles; switching the reference's query scaling off makes the positions from 8,192 differ by 0.25 to 0.45); the converter's output for the FP8 checkpoint and its tiny synthetic FP8 checkpoint with a dense first layer against transformers; and the live CPU matrix (all eight scenarios pass resident, 10.5 to 11.3 tokens per second decoding, 26 to 32 prefilling, 209 s to load, and with `tiered:ram=48g`), plus `provider_local_tiers` at `ram=48g` and `ram=8g` (replies equal the resident ones). The GPU engine rejects the architecture with a `CucaError::Config` naming latent attention: a known gap.
- `provider-local`: the chat side of Mistral Small 4 (and Devstral, Ministral 3, Mistral Small 3.2). The reasoning markers `[THINK]` ... `[/THINK]` are detected from the template and the vocabulary (an opener inside a reasoning block stays reasoning text), and a new tool syntax reads `[TOOL_CALLS]name[ARGS]{json}` once per call with no closing marker (a call ends at the next opener or at the end of the reply; Mistral Small 3.2's `[CALL_ID]id` becomes the call's id). A template that reads `reasoning_effort` (`none` or `high`) is driven by the request's thinking setting. The parser follows llama.cpp's Ministral 3, Devstral and Mistral Small 3.2 fixtures, and the Mistral Small 4 and Small 3.2 chat templates render byte for byte like Python `jinja2` (transformers' environment) on 42 and 29 cases, error cases included. `testdata/chat_golden_new.json` now has LF line endings.
- `provider-local`: the checkpoint converter handles Mistral Small 4 (`Mistral3ForConditionalGeneration` with a `mistral4` text model) and `MistralForCausalLM` (Mistral Small 3 and Nemo style checkpoints, as `llama`). `F8_E4M3` weights are dequantized while converting (multiplied by their `*_scale_inv`, per tensor or per expert; activation scales dropped; block-scaled FP8 refused up front) and written at the chosen precision. Mistral Small 4 is written in llama.cpp's `mistral4` layout (latent attention with `kv_b_proj` split into `attn_k_b` and `attn_v_b`, stacked fused `ffn_gate_up_exps`, shared experts, the `mistral4.*` keys); the engine runs that architecture on the CPU (see the `mistral4` entry). The `tokenizer.ggml.pre` name from llama.cpp's probe-text hash table now wins over the `tokenizer.json` structure when the engine implements it (Mistral Small 4: `pixtral`), and a repository that ships only `tekken.json` is refused by name. Every tensor and key equals llama.cpp's converter output at F16 on a synthetic FP8 checkpoint, the first two layers of the real FP8 Mistral Small 4 shards and two layers of Mistral Small 24B 2501.
- `provider-local`: Muse Glimmer 30B (`muse-glimmer`, preset `muse-glimmer-30b-q4_0`, bartowski's imatrix Q4_0 file, which embeds the earlier chat template: it differs from the current one only when the caller's system prompt mentions "Reasoning effort"). A dense decoder with a `[sliding, sliding, sliding, full]` pattern (RoPE on the sliding layers only, with the one `rope.freq_base`; the full layers apply none), a separate sigmoid attention-gate projection per layer, a scale-less per-head Q/K norm (the file's Q norm carries the constant scale), post-attention and post-FFN norms with epsilon 1e-8, a scale-less norm on the embeddings and a logit multiplier before the soft-cap. The behaviour is table-driven (`Behavior` flags `attn_gate_tensor`, `nope_full_layers`, `swa_base_from_freq_base`, `embd_norm`, `post_norm_eps`, `logit_multiplier`) on the CPU and the GPU engine (the gate is cut by heads under tensor parallelism). The converter reads `MuseGlimmerForConditionalGeneration` checkpoints (text tower; the vision tensors are skipped) and its output is bit-identical to llama.cpp's for a tiny random model. The model speaks a channel protocol: the output parser reads messages addressed to `self` as `Thinking`, to `user` as text and to any other recipient as a tool call (`<atem:function_calls>`), and `<|eom|>` ends a message, not the turn. The request's `ThinkingEffort` maps to the template's `reasoning_strength` (a model that reads `reasoning_effort` gets `none` or `high`); the model cannot switch reasoning off, so `can_disable_thinking` is false. Verified: the tiny model against `transformers` f32 (maximum absolute logit difference 2.6e-4, greedy equal), a 4-layer cut of the real checkpoint with a 2,600-token prompt (8.4e-5, greedy equal), 744 tokenizer cases and 25 chat-template renders equal to the reference, llama.cpp's Muse parser fixtures as unit tests, and the live CPU suite (every applicable scenario passes; about 3.3 tokens per second decoding on 16 cores). The 30B preset has not run on a GPU (16.27 GB of weights); the tiny-model GPU tests pass on Vulkan.
- `provider-local`: Hugging Face checkpoints load without a GGUF. `ModelSource::Checkpoint { repo, revision }` (parses from `repo:revision`) and `ModelSource::Dir(path)` are converted once, in process and in pure Rust, to a cached GGUF (`<cache>/converted/<repo>/<revision>/model-<quant>.gguf`), which the existing loader, tokenizer and chat template read unchanged. `LocalModelConfig::convert` (`ConvertConfig { quant: WeightQuant }`, `F16`, `Q8_0` default or `Q4_0`) picks the precision. The converter ports llama.cpp's `convert_hf_to_gguf.py` for all sixteen engine architectures and the Gemma 4 assistant, and streams tensor by tensor (a block of at most 32 MiB of `f32` in flight). Its output was compared with the real script's, key by key and tensor by tensor: identical F16 files for thirteen real checkpoints and four synthetic ones, identical Q8_0 for SmolLM 2. `ModelSource::from_str` gains the `repo:revision` form; `LocalModelConfig::on_download_progress` also reports conversion progress (tensors written out of the tensor count).
- `provider-local`: new preset `r1-distill-llama-8b-q4_0`
  (`unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF`, Q4_0, 4.68 GB), so the `llama`
  architecture keeps a live test; its embedded chat template equals the
  Hugging Face one (the bartowski file's lacks the trailing `<think>`).
- `provider-local`: split GGUF files (`llama-gguf-split`, `<base>-00001-of-0000N.gguf`) load. `ModelSource::File` naming any part merges the set into one cached GGUF under `<cache>/merged/`; a Hugging Face source whose file is a split name downloads and verifies every part, merges them into `<base>.gguf` in the repository's cache directory and deletes the parts. `ModelSource::HuggingFace` gains `parts: Vec<ModelPart>`, `ModelPreset` gains `parts: &[PresetPart]` (empty for every current preset), and `on_download_progress` reports the merge. A merge needs about twice the model's size free and a deleted merged file means a re-download; see the model sources page.
- `provider-local`: the `tekken` pre-tokenizer (Mistral Nemo, Pixtral, Mistral Small 3 and 4), built from the GPT-4o word rule with optional contraction suffixes and a digit-group size as parameters, and the checkpoint converter's structural name for the Tekken regex. Equal ids to Hugging Face `tokenizers` (Mistral Small 4) and `mistral_common` (Mistral Small 3.2) on 4,537 and 1,536 texts.

### Changed

- `provider-local` live suite: the two tool scenarios of a model whose template forces reasoning decode with the model's own sampling defaults (`LocalModel::sampling()`) and the fixed request seed 1 instead of greedily. Greedy decoding of `ministral3-3b-reasoning-q4_0` loops in its reasoning for the whole budget (7,965 tokens at 8,192 context too; HuggingFace f32 the same), while at the model card's temperature 0.7 it calls the tool in 66 to 137 tokens (HuggingFace bf16: 94 to 149); the template, the budget and the parser were ruled out. Both tool scenarios now pass for that preset and its known limit is removed; `muse-glimmer-30b-q4_0` (pinned repository template) passes all eight scenarios too.
- `provider-local`: the documented thinking behaviour of `LocalModel` and the requests page is corrected: effort levels reach templates that read `reasoning_strength` (Muse Glimmer: `low` to `xhigh`) and the two-mode `reasoning_effort` (`none`/`high`) is driven by the enabled flag.
- `provider-local`: the pre-tokenizer name `pixtral` now selects the Tekken expression (llama.cpp's mapping, with `ignore_merges` and a default BOS) instead of the Llama 3 expression. No preset used it.
- `provider-local`: the K/V cache is paged. K/V lives in blocks of 16
  positions in per-layer-group pools (`model/paged.rs`), a sequence holds one
  block table per group, and a prefix store names full blocks by a hash chain
  over their token ids, so a finished request's blocks stay cached (least
  recently used evicted first) and an unrelated request no longer evicts a
  prefix. `prompt_cache_usage.read_tokens` is now a multiple of 16 (the
  trailing partial block, at most 15 tokens, is recomputed). The sliding-window
  ring buffer and its compaction are gone: a sliding block is released once it
  leaves the window. The pool holds twice `context_length` positions, so
  `LocalModel::kv_cache_bytes` grows (Gemma 4 E2B at 8192: 145 MB to 422 MB;
  full-attention models double). On the CPU attention reads the blocks in
  place with an online-softmax kernel per (row, head): decode is faster
  (Gemma 4 E2B 20.9 to 25 tok/s, Qwen3-0.6B 27 to 50 at 1,100 tokens), a row
  is bit-identical alone or in a batch, and warm equals cold, chunked equals
  token by token and speculative equals plain bit for bit as before; the
  rounding differs from candle's matmul attention, so a greedy reply can
  differ from 0.6.x where two logits are within about 0.1. On a GPU a decode
  step reads the blocks with a split (flash-decoding) CubeCL kernel
  (`paged_kernel.rs`) and a prefill gathers the blocks it needs.
- `provider-local`: continuous batching. The engine is a step scheduler
  (`engine/scheduler.rs`): up to `LocalModelConfig::batching.max_sequences`
  requests (default 4) run at once, and every step is one forward pass
  (`Model::forward_batch`) over a decode row of each running sequence plus
  chunks of the prompts being prefilled, within `max_step_tokens` rows: the
  dense layers read each weight once per step, attention, the K/V block
  writes and a hybrid model's recurrent scan stay per sequence (per-sequence
  state slots). All sequences share the one K/V block pool
  (`batching.kv_tokens`, default twice `context_length`, the memory knob:
  more sequences do not grow it); a request that needs a block when the pool
  is full preempts the youngest running one, which resumes from the blocks it
  left in the prefix store, and blocks of a prompt are named as soon as they
  are filled, so a request with the same prompt as a running one reuses them.
  A slow consumer stalls only its own request (`try_send`, no blocking send
  in the engine), and a dropped stream releases its blocks at once.
  `LocalModel::serving_stats` reports the scheduler (running and waiting
  requests, free and cached blocks, snapshot bytes, prefix hits, preemptions,
  rows per step). `BatchingConfig` parses from `seqs=4,kv=16384,step=256,
  prefix=off` (`CUCA_LOCAL_BATCHING` in the example and the live tests). On
  the CPU a sequence's output equals what it produces alone, bit for bit;
  with four sequences decoding 96 tokens each the aggregate rate rose from
  24.3 to 50.1 tok/s on Gemma 4 E2B Q4_0 (2.07x), 65.5 to 110.8 on
  Qwen3-0.6B Q8_0 (1.69x) and 53.3 to 95.1 on Qwen3.5-0.8B Q8_0 (1.78x); on
  Vulkan 11.3 to 21.7 (1.93x) and 27.4 to 43.8 tok/s (1.60x). A lone
  sequence decodes as fast as before (Qwen3-0.6B 51 tok/s on both). A hybrid
  model (Qwen 3.5) no longer keeps its last sequence and one checkpoint: the
  prefix store keeps a recurrent-state snapshot at the block boundary below
  the last end-of-turn token of every prompt it prefills
  (`Model::snapshot`/`restore`; `read_tokens` is that boundary at most), and
  `Model::checkpoint` is gone. An in-trunk drafter serves one sequence at a
  time and a file drafter speculates only while one request runs alone.
- `provider-local` (GPU): tensor parallelism. `LocalModelConfig::shards`
  (and `LocalModelConfig::set_devices("cuda:0+cuda:1")`, the syntax of
  `CUCA_LOCAL_DEVICE` in the example and the live tests; the example's
  `loaded on` line lists the devices) splits a model over `1 + shards.len()`
  GPUs of one backend: attention heads and FFN widths are split evenly (a mixture
  of experts inside every expert, K/V heads replicated when a model has fewer
  than there are devices, as Gemma 4 E2B's one), every device holds its slice
  of the weights and of the K/V pools, and the partial outputs are summed twice
  per layer, by burn's NCCL `all_reduce` on CUDA (Linux, distinct devices;
  compile-checked, not run) and host-staged in a fixed order everywhere else,
  so runs are deterministic. A device may repeat (`vulkan:0+vulkan:0`): the
  same path on one GPU. Rejected at load with `tensor parallelism: ...`: a
  drafter, tiered experts, delta-net layers (Qwen 3.5), full-width Q/K norms
  (OLMo 2, OLMoE) and models whose heads or widths do not divide. On
  `vulkan:0+vulkan:0` the live matrix of Gemma 4 E2B Q4_0 and Qwen3-0.6B Q8_0
  gives the single-device replies (15 of 15 each), at about a third of the
  speed (the host-staged reduction: two readbacks per layer), and continuous
  batching works on it (8 of 8 concurrent replies equal to the serial ones).
- `provider-local`: the `olmoe-1b-7b-q4_0` preset now pins
  `allenai/OLMoE-1B-7B-0125-Instruct-GGUF` (Q4_0, revision `2ac1d273`, 3.93 GB,
  published by the model's lab) instead of the September 2024
  `bartowski/OLMoE-1B-7B-0924-Instruct-GGUF`; same architecture and sizes.

### Removed

Models first released before 2025 and the engine code only they needed.

- `provider-local` presets `gemma2-2b-q4_0`, `qwen2.5-0.5b-q8_0`,
  `llama3.2-1b-q4_0`, `phi3.5-mini-q4_0`, `smollm2-360m-q8_0`,
  `tinyllama-1.1b-q4_0` and `granite3.1-moe-1b-q4_0` (all first released
  between December 2023 and December 2024). Any GGUF of those models still
  loads when its architecture is supported (`llama`, `qwen2`, `phi3`).
- The `gemma2` and `granitemoe` architectures: a GGUF with that
  `general.architecture` now fails with the unknown-architecture error, and
  the checkpoint classes `Gemma2ForCausalLM` and `GraniteMoeForCausalLM` fail
  with the no-converter error, both naming the value. No official model
  released since January 2025 uses `granitemoe` except a safety classifier,
  and `gemma2` only TxGemma.
- Attention logit soft-capping (the `{arch}.attn_logit_softcapping` key and
  the `softcap` argument of the CPU attention, the paged attention and the GPU
  decode kernel): only Gemma 2 used it. The final logit soft-cap of Gemma 4
  stays.
- The Llama 3 tool-call syntax (`ToolSyntax::LlamaJson`: a bare
  `{"name": ..., "parameters": ...}` reply and the `<|python_tag|>` body that
  became a `python` call); only Llama 3.2 used it. A model with such a
  template now answers in plain text. The tagged-JSON syntax still reads
  `parameters` as an alias of `arguments`.
- The pre-tokenizer names `smollm`, `megrez`, `stablelm2`, `phi-2`, `mpt`,
  `jais`, `jina-es`, `jina-de`, `jina-v2-es`, `jina-v2-de`, `codeshell` and
  `minerva-7b`, whose families were first released before 2025.
  `unsupported pre-tokenizer` names them now.
- The Granite MoE `block_sparse_moe` expert tensors in the checkpoint
  converter and the `TinySpec::gemma2` and `TinySpec::granitemoe` test
  fixtures; the SmolLM 2, TinyLlama, Gemma 2 and Granite 3.1 MoE converter
  oracle checkpoints and the fixtures, goldens, templates and known limits of
  the removed presets. The converter oracle now covers
  `unsloth/gemma-3-1b-it` (the ungated mirror of `google/gemma-3-1b-it`, SentencePiece),
  `allenai/OLMoE-1B-7B-0125-Instruct` and `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` (a 2025 `LlamaForCausalLM`).

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
