+++
title = "Local models and formats"
description = "What provider-local loads from a GGUF file: architectures, weight formats, tokenizers, chat templates, reasoning and tool-call parsing, the preset catalog, and what each was verified to do."
template = "page.html"
weight = 5
+++

# Local models and formats

<dl class="page-facts">
<dt>In one line</dt>
<dd>The engine loads a GGUF file the way llama.cpp does: the architecture, the tokenizer and the chat template all come from the file, and the stream has the same <code>Text</code>, <code>Thinking</code> and <code>ToolCall</code> blocks whatever the model.</dd>
<dt>You need</dt>
<dd>The <code>provider-local</code> feature. Loading, configuration, devices and errors are on <a href="@/providers/local/_index.md">Local (in-process)</a>.</dd>
<dt>Read this if</dt>
<dd>You are choosing a model, loading a GGUF that is not a preset, or checking why a model's reply is parsed the way it is.</dd>
</dl>

Nothing here is specific to one model family. The loader reads every dimension from the file's metadata and tensor shapes, selects the forward-pass behavior from `general.architecture`, builds the tokenizer from `tokenizer.ggml.*`, and renders prompts with the Jinja chat template stored in `tokenizer.chat_template`. llama.cpp is the design reference; nothing of it is linked or called.

## What the engine reads

| Fact | Value |
|---|---|
| Feature flag | `provider-local` (CPU); GPU features are on [Local (in-process)](@/providers/local/gpu.md#devices-and-gpu-features) |
| Entry types | `ModelSource`, `ModelPreset`, `LocalModelConfig::source` |
| File format | GGUF, or a Hugging Face safetensors checkpoint converted to GGUF once ([Checkpoints](#checkpoints)) |
| Selects the forward pass | `general.architecture` |
| Selects the tokenizer | `tokenizer.ggml.model` and `tokenizer.ggml.pre` |
| Supplies the prompt format | `tokenizer.chat_template`, required |
| Selects end-of-generation tokens | `tokenizer.ggml.eos_token_id`, `eot_token_id`, `eom_token_id` and known terminator tokens in the vocabulary |

## Architectures

`general.architecture` selects the behavior. An architecture outside this list fails the load with a `CucaError::Provider` that names it.

| Architecture | Model families | Devices |
|---|---|---|
| `gemma4` | Gemma 4 (E2B, E4B) | CPU, GPU |
| `gemma3` | Gemma 3 (270M, 1B) | CPU, GPU |
| `llama` | Llama-family models published as `llama` (for example DeepSeek-R1-Distill-Llama, Mistral-style models) | CPU, GPU |
| `qwen2` | DeepSeek-R1-Distill-Qwen and other Qwen 2.5-family models | CPU, GPU |
| `qwen3` | Qwen 3 | CPU, GPU |
| `qwen35` | Qwen 3.5 (0.8B) | CPU, GPU (tiny-model tests only on the GPU) |
| `qwen35moe` | Qwen 3.6 MoE (Qwen3.6-35B-A3B) | CPU, GPU |
| `phi3` | Phi 4 mini | CPU, GPU |
| `smollm3` | SmolLM 3 | CPU, GPU |
| `granite` | Granite 3.3 (dense) | CPU, GPU |
| `qwen3moe` | Qwen 3 MoE (Qwen3-30B-A3B) | CPU, GPU |
| `olmoe` | OLMoE (OLMoE-1B-7B-0125) | CPU, GPU |
| `olmo2` | OLMo 2 | CPU, GPU |
| `ernie4_5` | ERNIE 4.5 (dense) | CPU, GPU |
| `muse-glimmer` | Muse Glimmer 30B | CPU; GPU on tiny models only (the 30B model does not fit a 16 GB card) |
| `mistral3` | Ministral 3 (3B, 8B, 14B), Devstral Small 2 | CPU, GPU (tiny-model tests only on the GPU) |
| `mistral4` | Mistral Small 4 (119B, 128 experts) | CPU and GPU (not tensor parallel) |
| `gemma4-assistant` | the Gemma 4 multi-token-prediction drafter | only through `LocalModelConfig::drafter` |

What each architecture adds to the common forward pass:

- `gemma4`: sliding and full layers with their own head sizes and RoPE bases, shared K/V layers (the last `shared_kv_layers` layers read an earlier layer's cache), per-layer embeddings read from the file per token, a weightless value norm, a per-layer output scale, and a final logit soft-cap.
- `gemma3`: sliding and full layers (every sixth layer is full), QK-norm, post-attention and post-FFN norms.
- `llama`: the Llama 3 RoPE scaling that the file carries as `rope_freqs.weight`, optional biases, a tied or separate output head.
- `qwen2`: Q, K and V biases.
- `qwen3`: QK-norm, and a head size that may differ from the model width.
- `phi3`: fused attention and gate/up projections, LongRoPE with short and long factors, partial rotary dimensions.
- `smollm3`: Llama-style layers where every fourth layer applies no RoPE.
- `granite`: embedding, attention, residual and logit scales.
- `qwen3moe`: Qwen 3 attention (a Q/K RMSNorm per head, NEOX RoPE) and a top-k mixture of experts whose selected weights are renormalised to sum to one.
- `olmoe`: pre-norms, a Q/K RMSNorm that spans the whole projection, NEOX RoPE and a top-k mixture of experts whose selected weights are not renormalised.
- `olmo2`: no pre-norms, only post-attention and post-FFN norms, and a Q/K RMSNorm that spans the whole projection.
- `ernie4_5`: Llama-style layers with a head size larger than the width divided by the head count.
- `muse-glimmer`: a dense decoder with a `[sliding, sliding, sliding, full]` pattern from `attention.sliding_window_pattern`. Sliding layers apply `NORM` RoPE with `rope.freq_base` (the file carries no `rope.freq_base_swa`) and full layers apply no RoPE at all. Every layer has a separate `attn_gate` projection of the normed input whose sigmoid multiplies the attention output before `attn_output`, a scale-less per-head RMS norm on Q and K (the file's `attn_q_norm` is the constant `qk_scale_factor`, 3.87, and `attn_k_norm` is ones), and Gemma-style post-attention and post-FFN norms whose epsilon is 1e-8 where the other norms use `attention.layer_norm_rms_epsilon`. The embeddings go through a scale-less RMSNorm after the lookup (no `sqrt(n_embd)` scale), and the logits are multiplied by `logit_scale` (0.196, a multiplier, not Granite's divisor) before the final soft-cap of 20.
- `mistral3`: Llama layers (NORM RoPE, a tied or separate head, `attention.key_length` honoured) with YaRN whose magnitude follows `rope.scaling.yarn_log_multiplier` and the Llama 4 query scaling: after RoPE the query of position `p` is multiplied by `1 + attention.temperature_scale * ln(1 + floor(p / original_context_length))`, which is 1 inside the original context (16,384 for Ministral 3) and applies to every layer. Without `temperature_scale` (or at 0) there is no scaling.
- `mistral4`: multi-head latent attention, a mixture of experts and Mistral's YaRN; see [Latent attention models](#latent-attention-models).
- `qwen35`: a hybrid stack; see [Hybrid models](#hybrid-models).
- `qwen35moe`: the `qwen35` hybrid stack with a mixture of experts in every layer: a softmax over all experts, the top k, the selected weights renormalised to sum to one, plus a shared expert (a dense gated FFN every token runs) whose output is scaled by `sigmoid(ffn_gate_inp_shexp . x)` and added to the routed output. The routed experts honor `ExpertPlacement`; the shared expert and the router stay resident.
- `gemma4-assistant`: a four-layer, query-only decoder that reads the K/V cache of a `gemma4` target. It pairs only with a `gemma4` model of the same hidden width and vocabulary.

The common forward pass covers grouped-query attention, causal and sliding-window masks, `NORM` and `NEOX` RoPE (partial rotary, linear and YaRN scaling, per-pair factors), RMSNorm, SiLU and GELU gated FFNs, tied and separate output heads, and mixture-of-experts routing. Optional tensors (biases, QK-norm, fused projections, a separate `output.weight`) are detected by their presence in the file.

Not supported: other recurrent architectures such as Mamba and RWKV. Their GGUF fails with ``unsupported GGUF architecture `NAME`; supported: gemma4, gemma3, llama, qwen2, qwen3, phi3, smollm3, granite, qwen3moe, olmoe, qwen35, qwen35moe, olmo2, ernie4_5, muse-glimmer, mistral4, gemma4-assistant``.

### Latent attention models

Mistral Small 4 (`mistral4`, 36 layers, hidden 4,096, 32 heads) replaces the per-head K and V projections by a compressed latent. A layer computes `q = q_b(norm(q_a(x)))` (a rank of 1,024; every head is `nope 64 + rot 64` wide) and `kv = kv_a(x)`: its first 256 values, normed, are the latent `c` shared by every head, and its last 64 are the rotary key shared by every head too. In the reference form head `h` reads keys `K_h c` and values `V_h c`, and a position's score is `q_nope . (K_h c) + q_rot . k_rot`. The engine never builds those per-head keys and values: it absorbs `K_h` into the query (`q_abs = K_h^T q_nope`, 256 values), so the score is `[q_abs | q_rot] . [c | k_rot]` against one cached row of 320 floats per position and layer, and the weighted sum of the values is `V_h (sum_j p_j c_j)`, taken over the first 256 columns of the same cached rows. The K/V cache is therefore 1,280 bytes per position and layer (46,080 bytes over the 36 layers, 755 MB for the pool of a default 8,192-position context), against 1.18 MB per position for the expanded heads. The `attn_k_b` and `attn_v_b` tensors are held as f32 (216 MB in all) and applied one row at a time, so a row is bit-identical alone or in a batch, as for the other attention kernels.

- YaRN magnitude, for every architecture: with `rope.scaling.yarn_log_multiplier` (`mscale_all_dim` in the checkpoint) `m` other than 0 the cosine and sine tables are scaled by `get_mscale(factor, 1) / get_mscale(factor, m)` with `get_mscale(f, x) = 1 + 0.1 x ln f` (1 for `m = 1`, the Ministral 3 and Devstral 2 value); without it the table keeps ggml's `1 + 0.1 ln f` boost. `mistral4` stores `0.1 * mscale_all_dim` and puts the magnitude into the softmax scale instead (below).
- Rotary embedding: adjacent pairs of the 64 rotary dimensions of every query head and of the shared key, YaRN (factor 128 over 8,192, `beta_fast` 32, `beta_slow` 1). The cosine and sine tables carry no magnitude scale; the YaRN `mscale` (`1 + 0.1 ln 128`, from `rope.scaling.yarn_log_multiplier`) enters twice into the softmax scale instead, `mscale^2 / sqrt(128)` as in llama.cpp's `deepseek2` graph and transformers' `Mistral4Attention`.
- Llama 4 query scaling: the whole query of position `p` is multiplied by `1 + 0.1 ln(1 + floor(p / 8192))`, which is 1 for the first 8,192 positions. A GGUF may carry `attention.temperature_scale` and `attention.temperature_length` (zero switches the scaling off); without them the engine applies the checkpoint's own configuration, 0.1 and 8,192, which llama.cpp's converter did not write for this model (so llama.cpp runs it unscaled past 8,192 positions).
- Mixture of experts: a softmax over all 128 experts, the top 4, their weights renormalised and multiplied by `expert_weights_scale`, plus one shared expert that is added as it is, without a gate. The gate and up matrices of the experts are one tensor (`ffn_gate_up_exps`: for every expert its gate rows, then its up rows). Experts may be tiered (see [Tiered expert placement](@/providers/local/expert-tiers.md)); the fused tensor is read as two strided halves, and the router, the shared expert and the attention stay resident.
- Leading dense blocks (`leading_dense_block_count`, as in DeepSeek models) run a dense gated FFN in those layers. Not supported: grouped or sigmoid expert routing, a model without the low-rank query (`attention.q_lora_rank`), and the GPU engine: loading on a GPU fails with a `CucaError::Config` that names latent attention.

### Hybrid models

Qwen 3.5 mixes gated delta-net layers with gated full-attention layers. A delta-net layer keeps a causal convolution and a recurrent state per sequence instead of a K/V cache. Every `full_attention_interval`-th layer (every fourth in the 0.8B model) is gated full attention with partial NEOX RoPE. A `*-MTP` file adds a trailing prediction block that `DrafterSource::InTrunk` drafts with. When a layer has fewer key heads than value heads (`ssm.group_count` below `ssm.time_step_rank`, as in `qwen35moe`: 16 and 32), value head `h` reads key head `h % ssm.group_count`: llama.cpp's converter permutes the HuggingFace value heads into that tiled order and the engine reads the file the same way.

- Memory for the 0.8B model: 24,576 bytes of K/V per position (6 attention layers with 2 K/V heads of 256), plus a recurrent state of 20.2 MB that does not grow with the context.
- The state cannot be truncated. Prefix reuse resumes from a snapshot of the state that the engine takes while prefilling, at the block boundary of 16 positions below the last end-of-turn token of the prompt (the end of the last complete message). The next turn of a conversation re-prefills the few tokens after that point, and its output equals a cold run's. The prefix store keeps at most `max_sequences + 4` snapshots, least recently used out first.
- The prompt cache usage a request reports counts what was reused.
- Each running sequence owns a state slot, so `max_sequences` slots exist; the in-trunk drafter serves one sequence.
- Rolling back a speculative verification keeps a copy of the state from before the call plus the per-row inputs of the recurrence. A verification call covers at most 17 rows (the pending token and 16 drafts). For the 0.8B model that is about 40 MB per slot (live state and pre-call copy) plus 445 KB per verified row, and 20.2 MB per snapshot in the prefix store, none of it growing with the context.
- On a GPU a hybrid model keeps the same state, snapshots and cache fallback (see [Hybrid models on a GPU](@/providers/local/gpu.md#verified-and-compile-checked)); it keeps no verify record, since the in-trunk head and every drafter are CPU only or pair with `gemma4`. The delta-net recurrence runs token by token on the device, so a hybrid model decodes more slowly on the GPU than on the CPU today (Qwen3.6-35B-A3B with tiered experts: about 3 to 4 tokens per second on Vulkan against 7 to 17 on the CPU).
- The trailing prediction block of a `qwen35moe` file is not loaded (the trunk ignores it). `DrafterSource::InTrunk` on such a file fails with a `CucaError::Config` that names the mixture-of-experts FFN: the in-trunk head supports a dense block only.

## Checkpoints

A safetensors checkpoint of any architecture in the table above loads through `ModelSource::Checkpoint` or `ModelSource::Dir` (see [Model sources and presets](@/providers/local/model-sources.md#checkpoints)): it is converted once, in process, to a cached GGUF at `f16`, `q8_0` (default) or `q4_0`, and the engine then reads that file exactly as it reads any other. The `architectures[0]` of `config.json` selects the converter:

| `architectures[0]` | Engine architecture |
|---|---|
| `LlamaForCausalLM`, `MistralForCausalLM` | `llama` |
| `SmolLM3ForCausalLM` | `smollm3` |
| `GraniteForCausalLM` | `granite` |
| `Qwen2ForCausalLM`, `Qwen3ForCausalLM`, `Qwen3MoeForCausalLM` | `qwen2`, `qwen3`, `qwen3moe` |
| `Qwen3_5ForCausalLM`, `Qwen3_5ForConditionalGeneration` | `qwen35` |
| `Qwen3_5MoeForCausalLM`, `Qwen3_5MoeForConditionalGeneration` | `qwen35moe` |
| `Phi3ForCausalLM` | `phi3` |
| `Gemma3ForCausalLM`, `Gemma3ForConditionalGeneration` | `gemma3` |
| `Gemma4ForCausalLM`, `Gemma4ForConditionalGeneration`, `Gemma4AssistantForCausalLM` | `gemma4`, `gemma4-assistant` |
| `Olmo2ForCausalLM`, `Olmo3ForCausalLM` (without sliding layers), `OlmoeForCausalLM` | `olmo2`, `olmoe` |
| `Ernie4_5ForCausalLM` | `ernie4_5` |
| `Ministral3ForCausalLM`, `Mistral3ForConditionalGeneration` whose `text_config.model_type` is `ministral3` or `mistral` (Ministral 3, Devstral Small 2, Mistral Small 3.1/3.2, Magistral 2509; text tower, FP8 weights dequantized) | `mistral3` (YaRN log multiplier and attention temperature written as llama.cpp does) |
| `MuseGlimmerForConditionalGeneration` (text tower; the vision tensors are skipped) | `muse-glimmer` |
| `Mistral3ForConditionalGeneration` whose `text_config.model_type` is `mistral4` (Mistral Small 4; text tower, FP8 weights dequantized) | `mistral4` (written as llama.cpp does) |

Any other architecture fails with `` architecture `X` has no checkpoint converter; convert it to GGUF with llama.cpp ``. The weights, the metadata, the tokenizer (including `tokenizer.ggml.pre`, found from `tokenizer.json` and fingerprinted with llama.cpp's probe-text hash table) and the chat template are those llama.cpp's `convert_hf_to_gguf.py` writes: that script (commit `67107ac34f62882d8b53ea41ad68010ea635b9be`) was run on the same checkpoints and its GGUF files compared key by key and tensor by tensor with the converter's, with no difference but the float-rounding ones listed under [Checkpoints](@/providers/local/model-sources.md#checkpoints). Converted `f16` checkpoints also match the transformers oracle (top-1 equal, top-10 overlap at least 9 of 10, logits within 2 % relative L2) in `src/provider/local/convert/oracle.rs`, and answer the oracle's chat turn in `tests/provider_local_convert.rs`. All eight pinned repositories (OLMo 2, Granite 3.3, ERNIE 4.5, SmolLM 3, Gemma 3 1B through the ungated `unsloth/gemma-3-1b-it` mirror, OLMoE-1B-7B-0125-Instruct, DeepSeek-R1-Distill-Llama-8B as the 2025 `LlamaForCausalLM`, Qwen 3.5) passed: top-1 equal, top-10 overlap 10 of 10, greedy ids equal for all 24 (8 or 4 on the long prompts) steps, and every checkpoint answers the oracle's chat turn.

The live matrix (`tests/provider_local.rs`, CPU, release) on converted checkpoints gives the same PASS/SKIP/KNOWN-LIMIT categories as the preset GGUF of the same model for Qwen 3 0.6B, Phi 4 mini and Gemma 4 E2B (converted at Q4_0: `token_embd` at Q8_0 and `per_layer_model_proj` at F16, as the published file has them). The one scenario that differs is a model effect, not a conversion one: Gemma 3 1B converted at Q8_0 writes the count in words (`One, Two, ...`) where the Q4_0 preset writes digits, so `stop_string_cuts_the_reply` fails; converted at `q4_0` it passes like the preset. The multi-token-prediction drafter check (`tests/provider_local_mtp.rs`) passes with a converted Gemma 4 E2B target and a converted `google/gemma-4-E2B-it-assistant` (output identical to plain decoding, 1.17x at 3 drafts), and the Qwen 3 0.6B matrix passes on Vulkan.

## Weight formats

| Dtype | CPU | GPU |
|---|---|---|
| Q4_0, Q8_0 | stays quantized; cuca's batch-invariant quantized matmul kernel | repacked at load into the kernels' own block-major words; stays quantized |
| F32, F16, BF16 | held as f32 | uploaded as f32 (4 bytes per element) |
| Q4_1 | stays quantized; candle's quantized matmul | repacked like Q4_0, with the f16 minimum of each block |
| Q5_0, Q5_1, K-quants | stays quantized; candle's quantized matmul | dequantized row by row at load, requantized to Q8_0 |
| Q8_1 | not covered by a test | dequantized row by row at load, requantized to Q8_0 |
| IQ family | rejected | rejected |

- Norm vectors, biases and scalars are f32 whatever their stored dtype. Resident weights are about the file size.
- The CPU reads any dtype candle's quantized matmul reads: F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0 and the K-quants (Q2_K to Q6_K). An IQ-family tensor makes the load fail with `` `PATH` is not a supported GGUF file: ... ``.
- A GPU requantizes every quantized dtype other than Q4_0, Q4_1 and Q8_0 to Q8_0. Each element moves by at most half a Q8_0 step. That is a smaller error than the one a Q4_K, Q5_K or Q6_K file already carries, but GPU logits then differ from the CPU's by more than float rounding.
- A GPU file that mixes Q4_0, Q4_1 and Q8_0 weights, as the Q4_0 presets with a Q8_0 embedding do, keeps every weight in its own format, one copy each: the dense kernels read the format of each weight, so no weight is held as f32 to keep schemes apart.
- A weight over 256 MiB on the GPU is split into row chunks.
- Measured on Vulkan with `phi4-mini-q4_k_m` (AMD RX 9070 XT, Windows): the top-1 token equals the CPU's and the relative logit error against the dequantized CPU reference is 1.8e-2.
- Mixture of experts: by default expert tensors are split once at load into one quantized 2-D weight per expert, so memory is the file size. `LocalModelConfig::experts` can leave them in the file instead; see [Tiered expert placement](@/providers/local/expert-tiers.md).

## Tokenizers

`tokenizer.ggml.model` selects the scheme. All of it is Rust and is built once at load:

- `llama` is SentencePiece: score-ordered merging of UTF-8 characters, `<0xHH>` byte fallback, and the optional leading space of `tokenizer.ggml.add_space_prefix` (default true when the key is absent).
- `gpt2` is byte-level BPE: the text is cut into words by the pre-tokenizer that `tokenizer.ggml.pre` names, each word is mapped through the GPT-2 byte alphabet and merged by rank. The pre-tokenizers are hand-written linear splitters over Unicode general categories, not regular expressions, so their cost grows linearly with the text.
- `gemma4` (also `gpt2` with `pre` set to `gemma4`) replaces spaces with `▁`, never merges across a newline boundary, and falls back to `<0xHH>` byte tokens.

Any other `tokenizer.ggml.model` fails the load with ``unsupported tokenizer model `NAME` ``. The tokenizer each preset family uses:

| Family | `tokenizer.ggml.model` | `tokenizer.ggml.pre` |
|---|---|---|
| Gemma 4 | `gemma4` | none |
| Gemma 3, ERNIE 4.5 | `llama` | `default` |
| Qwen 3 | `gpt2` | `qwen2` |
| Qwen 3.5 | `gpt2` | `qwen35` |
| DeepSeek-R1 distill (Qwen) | `gpt2` | `deepseek-r1-qwen` |
| DeepSeek-R1 distill (Llama) | `gpt2` | `llama-bpe` |
| SmolLM 3 | `gpt2` | `smaug-bpe` |
| Phi 4 mini | `gpt2` | `gpt-4o` |
| Granite 3.3 | `gpt2` | `refact` |
| OLMo 2 | `gpt2` | `dbrx` |
| OLMoE | `gpt2` | `olmo` |
| Muse Glimmer | `gpt2` | `llama4` |

### Pre-tokenizers

A `gpt2` model with no `pre`, or `default`, uses the default expression. SentencePiece models use no pre-tokenizer. The implemented values, by expression:

- `llama3`, `llama-v3`, `llama-bpe`, `falcon3`, `falcon-h1`, `midm-2.0`, `lfm2`, `jina-v5-nano`: the Llama 3 expression. Whole words found in the vocabulary skip the merges, and a BOS token is added by default.
- `smaug-bpe`, `dbrx`: the Llama 3 expression without that shortcut.
- `qwen2`, `deepseek-r1-qwen`, `kormo`, `f2llmv2`, `hunyuan`, `solar-open`: the Qwen 2 expression.
- `qwen35`: the Qwen 2 expression with combining marks.
- `gpt-2`, `olmo`, `trillion`, `granite-docling`, `exaone4`, `mellum`, `modern-bert`, `gigachat`, `a.x-4.0`: the GPT-2 expression.
- `starcoder`, `refact`, `command-r`, `exaone`, `mellum2`: digits are split first, then the GPT-2 expression runs on the fragments between them.
- `gpt-4o`, `llama4`, `kanana2`, `talkie`, `minimax-m2`: the GPT-4o expression, with case classes from real Unicode categories.
- `tekken`, `pixtral`: the Tekken expression of Mistral Nemo, Pixtral, Mistral Small 3 and Mistral Small 4, as llama.cpp maps both names (`LLAMA_VOCAB_PRE_TYPE_TEKKEN`). It is the GPT-4o expression without the contraction suffix on words and with single digits (`\p{N}`, not groups of three). Whole words found in the vocabulary skip the merges, and a BOS token is added by default. Before this release `pixtral` read as the Llama 3 expression, which is what the Hugging Face `tokenizer.json` of Pixtral and of Mistral Small 3.x declares; the Tekken expression is what the models were trained with (their `tekken.json`), and on a 1,536-text corpus the two differ on 13 texts (non-ASCII letters, mostly). A checkpoint whose `tokenizer.json` carries the Tekken regex and `ignore_merges` converts to `tekken`; the probe-text fingerprints of llama.cpp's `tekken` and `pixtral` tokenizers are in the converter's table.
- `gemma4`: the space-escaped scheme above.

Any other `pre` fails the load with ``unsupported pre-tokenizer `NAME` ``. The names of families first released before 2025 (`smollm`, `megrez`, `stablelm2`, `phi-2`, `mpt`, `jais`, `jina-es`, `jina-de`, `jina-v2-es`, `jina-v2-de`, `codeshell`, `minerva-7b`) are not implemented.

### Special tokens and BOS

- Special tokens (control, user-defined and unknown types) are cut out of the text longest-first before the scheme runs, so `<|im_start|>` in a rendered prompt is one token.
- The prompt gets a BOS token when `tokenizer.ggml.add_bos_token` (or the scheme's default) says so, and never a second one when the rendered template already starts with it.
- For a SentencePiece model with `add_space_prefix`, text after a special token such as `</s>` gets a leading space token, as in llama.cpp. HuggingFace's tokenizer does not add it, so such a model's prompt has one more token than the reference tokenizer's (ERNIE 4.5's GGUF files that omit the key; the default stays because llama.cpp applies it).

The Tekken rule is verified by `tekken_words_match_the_reference_regex` (`src/provider/local/tokenizer/pretok.rs`): the pieces of 336 texts (ASCII, digits, CJK, emoji, accented mixed case, code, whitespace runs, 300 seeded random ones) equal those of Hugging Face `tokenizers`' regex splitter with Mistral Small 4's `tokenizer.json` regex. With real vocabularies, the env-gated `tokenizer_corpus_matches_reference_ids` (`CUCA_TOK_GGUF`, `CUCA_TOK_CASES`) gave identical ids for 4,537 texts (those plus every one of the model's 1,000 special tokens alone and in context) against Mistral Small 4's `tokenizer.json`, and for 1,536 texts against Mistral Small 3.2's `tekken.json` read by `mistral_common`, through the headers of the published GGUFs (pre-tokenizer `pixtral` and `tekken`).

Verified by the tokenizer golden test in `src/provider/local/tokenizer/golden.rs`: for 7 models (Qwen 3, Gemma 3 270M and 1B, SmolLM 3, Phi 4 mini, R1 distill, Qwen 3.5) the ids of 16 texts and 3 chat prompts equal the HuggingFace `tokenizer.json` ids. The test reads cached model files and prints `SKIP:` for a missing one. For Muse Glimmer the ids of a corpus of 744 texts (ASCII, digits, contractions, mixed-case non-ASCII, CJK, emoji, special tokens and fuzzed strings) equal those of the HuggingFace `tokenizers` library, both through a tiny GGUF and through the real preset file's tokenizer (`tokenizer_corpus_matches_reference_ids`, driven by `CUCA_TOK_GGUF` and `CUCA_TOK_CASES`); HuggingFace's `ignore_merges` flag, which the model's `tokenizer.json` sets and the engine's rule does not, has no observable effect: all 400,000 texts made of one vocabulary entry, alone and after a space, encode identically with it on and off and equal the engine's ids. Where llama.cpp and HuggingFace differ, the expected ids follow llama.cpp, with one exception that follows HuggingFace: the GPT-4o case classes.

## Chat templates

The template is the Jinja text stored in the GGUF, compiled with minijinja in the environment HuggingFace's `apply_chat_template` uses: blocks trimmed, leading block whitespace stripped, and Python string methods such as `.strip()`, `.startswith()` and `.split()` available. The template text is not modified (`LocalModelConfig::chat_template`, or the file a preset pins, replaces it as a whole; see [Chat templates](@/providers/local/model-sources.md#chat-templates)), except that HuggingFace's generation and end-generation tags are rewritten to a no-op block.

Variables the template sees:

- `messages`: the request's messages in OpenAI shape.
- `tools`: defined only when the request declares at least one tool, because templates test `tools is not none`. Each tool is `{"type": "function", "function": {"name", "description", "parameters"}}`.
- `add_generation_prompt`: always true.
- `enable_thinking` and `thinking`: both true when the request enables thinking, otherwise both false.
- `preserve_thinking`: always false.
- `reasoning_strength` and `reasoning_effort`: defined only for a template that reads them; see [Reasoning](#reasoning).
- `bos_token` and `eos_token`: the token texts, or empty.

Helpers the templates rely on:

- `raise_exception(message)` surfaces as a render error.
- `strftime_now(format)` returns the current UTC time.
- The `tojson` filter lays JSON out like Python's `json.dumps` (`", "` and `": "` separators, with `indent`, `sort_keys` and `ensure_ascii`). It keeps the order of an object's keys, as Python does.
- The `string` filter renders like Python's `str()`, so a dict prints as `{'a': 1}`.

Tool arguments stay JSON objects, as templates expect. The key order inside a tool's JSON schema (`parameters`) is the order of the request's `serde_json` map: sorted by key, unless your application enables the `preserve_order` feature of `serde_json`, in which case it is the order you wrote. Arguments of tool calls in the history are sorted the same way.

Verified by the unit test `renders_match_the_reference_jinja_environment` in `src/provider/local/chat/golden.rs`. For 9 templates (Gemma 3 1B and 270M, Gemma 4, Granite 3.3, Phi 4 mini, Qwen 3, Qwen 3.5, the R1 Qwen distill, SmolLM 3) and 7 cases (a plain turn, a system prompt, a multi-turn history, thinking on and off, tools declared, a tool call with its result) the rendered text equals what Python `jinja2` renders with `trim_blocks`, `lstrip_blocks`, a `json.dumps(ensure_ascii=False)` `tojson` and `strftime_now` pinned to 2026-10-18 UTC. The fixtures are in `src/provider/local/testdata/`. Muse Glimmer's template is tested in `src/provider/local/chat/golden_new.rs` (same test name): 25 cases, with every reasoning strength among them, whose bytes equal what `transformers`' `apply_chat_template` renders (`testdata/chat_golden_new.json`). The same file tests Mistral Small 4 (42 cases: tool calls with mapping, string and empty-string arguments, reasoning as a top-level field and as content blocks, consecutive messages of one role, list contents, `reasoning_effort` `none` and `high`, and five the template rejects) and Mistral Small 3.2's embedded template (29 cases, with its call ids and alternation rule); every render equals the Python result, errors included. The templates added later (Muse Glimmer's current one, Mistral Small 4, Mistral Small 3.2 and the current Ministral 3 Reasoning one) are checked the same way in `chat/golden_new.rs`; the Ministral one on 44 cases (system prompt or none, tools, tool calls and results, reasoning in the history, multi-turn, content lists, and the cases that must raise), byte for byte against Python jinja2 in transformers 5.20.

## Reasoning

When the template writes a reasoning block, the engine finds its markers from the vocabulary and the template text:

- `<|channel>thought` ... `<channel|>`: Gemma 4's channel. The header word `thought` and the whitespace after the opening marker are dropped.
- `<think>` ... `</think>`: Qwen 3, Qwen 3.5, SmolLM 3, Granite 3.3 and the R1 distills.
- `[THINK]` ... `[/THINK]`: Mistral Small 4 (and Ministral 3). Both are single control tokens and the template must mention `[THINK]`. The prompt ends after `[/INST]`, so the model opens the block itself, and a `[TOOL_CALLS]` written inside it stays reasoning text.
- Messages addressed to `self`: Muse Glimmer's channel protocol, below.

The text between the markers arrives as `Thinking` blocks. If the rendered prompt ends inside an open block, generation starts in thinking. Markers are matched on text, so a marker that the vocabulary spells as several tokens works like a single control token. A close marker with no open marker is dropped.

Whether a request can switch thinking off follows the template:

- Gemma 4, Qwen 3, Qwen 3.5 and SmolLM 3 read `enable_thinking`.
- Muse Glimmer reads `reasoning_strength` (`low`, `medium`, `high` or `xhigh`, default `high`). The request's `ThinkingEffort` maps to it (`Minimal` and `Low` to `low`, then `medium`, `high`, `xhigh`); a request that disables thinking gets `low`, the least reasoning the model has, so it cannot switch it off (`can_disable_thinking` is false). `enable_thinking(None)` and no thinking configuration leave the template's `high`. The variable is defined only for a template that reads it.
- A template that reads `reasoning_effort` (a two-mode model: Mistral Small 4, `none` or `high`, which raises on any other value) is a switch like `enable_thinking`: an enabled request renders `high` whatever its effort level, anything else `none`.
- Granite 3.3 reads `thinking`, which the request sets to the same value.
- The DeepSeek-R1 distills read neither and reason on every request.

`reasoning_tokens` in the usage counts the tokens inside reasoning blocks, markers included, and is reported only when the request enabled thinking. `LocalModel::capabilities()` reports `reasoning` and `can_disable_thinking` from the same detection.

## Tool calls

The template says how the model was taught to call tools, and the output parser follows it. The engine buffers a call whole and parses it into a `ToolCall` block. Declared tools reach a model only through its template, so a model whose template has no `tools` variable never sees them.

Syntaxes, in detection order:

- Channel protocol ("ATEM"), Muse Glimmer: a turn is a sequence of messages `<|start|>assistant to=RECIPIENT<|message|>BODY END`, where `END` is `<|eom|>` when more messages follow and `<|eot|>` (an end-of-generation token) at the end of the turn. The prompt ends with `<|start|>assistant`, so generation starts at ` to=...`. A message addressed to `self` is reasoning (`Thinking` blocks), one to `user` (or without a recipient) is answer text, and any other recipient is a tool call whose body is `<atem:function_calls>`, one `<atem:invoke name="NAME">` per call and one `<atem:parameter name="KEY">VALUE</atem:parameter>` per argument. `<|eom|>` ends a message and generation continues; it is not an end-of-generation token. Markup quoted inside an answer or inside the reasoning stays text. A value is read as JSON when it is a JSON number, boolean, `null`, list or object without surrounding whitespace and as the raw string otherwise (a string that looks like one of these reads back as it; the format has that ambiguity). Detected when the template contains `<atem:function_calls>` and `<|eom|>` and the vocabulary has `<|start|>`, `<|message|>`, `<|eom|>` and `<|eot|>`. The fixtures of llama.cpp's `tests/test-chat.cpp` for this format (a plain answer, a first-message call, inform-then-act, quoted markup, reasoning containing markup) are unit tests of `output/atem.rs`.
- Mistral, Mistral Small 4, Devstral and Ministral 3: `[TOOL_CALLS]name[ARGS]{json}` once per call, with no closing marker: a call ends at the next `[TOOL_CALLS]` or at the end of the reply, and the arguments are one JSON object. Mistral Small 3.2 writes `[TOOL_CALLS]name[CALL_ID]id[ARGS]{json}` and its template requires 9-character ids, so the id the model wrote becomes the call's id (any other format takes engine-made `call_N` ids). Detected when the template contains `[TOOL_CALLS]` and `[ARGS]` and the vocabulary has both tokens (`[CALL_ID]` likewise). A malformed call is reported as `malformed tool call: <body>`, and the next call is still read.
- Gemma 4 custom: `<|tool_call>call:name{key:<|"|>value<|"|>}<tool_call|>` between control tokens. Detected when the template contains `<|tool_call>` and the vocabulary has both markers.
- XML, Qwen 3.5: `<tool_call>` ... `</tool_call>` around `<function=name>` and one `<parameter=key>` element per argument, with the value between a newline after the opening tag and a newline before the closing one. Detected when the template contains `<tool_call>`, `</tool_call>`, `<function=` and `<parameter=`.
- Tagged JSON, Qwen 3, SmolLM 3: `<tool_call>` ... `</tool_call>` around one call object or a list of them. Detected when the template contains `<tool_call>`; without `</tool_call>` the call runs to the end of the reply.
- Tagged JSON with an alternate opener, Granite 3.3: `<|tool_call|>`, or the plain text `<tool_call>` that the model writes instead, followed by a JSON list to the end of the reply. A closing `<|/tool_call|>` is read only when the template contains it.
- None, Gemma 3, Phi 4 mini, OLMo 2, ERNIE 4.5 and the R1 distills: the output is plain text and `LocalCapabilities::tools` is false.

JSON call objects are read in the shapes models write them:

- The name comes from `name`, from a string `function` (`{"function": "get_weather", "arguments": {...}}`), or from the nested OpenAI shape (`{"type": "function", "function": {"name": ..., "arguments": ...}}`).
- The arguments come from `arguments` or `parameters`, in the nested object first and then beside it, as an object, `null`, or a string holding an object. A missing `arguments` means no arguments.
- An empty list, a missing name, or arguments that are not an object are a malformed call.

In the XML syntax the template writes a string argument as raw text and any other value (number, boolean, object, array) as JSON. The parser reads a value as JSON when it is valid JSON and as a string otherwise. A string argument that is itself valid JSON, such as `42` or `true`, therefore comes back as that JSON value.

A malformed call is reported as a `CucaError::Provider` stream item, `malformed tool call: <body>`, and the stream continues. Tool call ids are generated by the engine.

Generation stops at an end-of-generation token: the EOS, EOT and EOM ids of the metadata plus these vocabulary tokens when present: `<|eot_id|>`, `<|eom_id|>`, `<|im_end|>`, `<|end|>`, `<end_of_turn>`, `<turn|>`, `<|tool_response>`, `<|endoftext|>`, `<|end_of_text|>`, `<EOT>`, `_<EOT>`, `[EOT]`, the DeepSeek end-of-sentence token and `<end_of_utterance>`. A model with none of these and no metadata id fails to load.

### Tool calls of a reasoning model

A model that reasons on every request can fail to reach its tool call when it decodes greedily. Ministral 3 3B Reasoning was recorded as a known limit on both tool scenarios for that reason; the cause was settled by elimination on the matrix prompt (`What's the weather in Paris right now? Use the tool.`, `get_weather` declared, 227 prompt tokens):

- The chat template is not the cause: the December 2025 template embedded in the GGUFs and the repository's current one (`4a36357c`) render the identical prompt for this request, and the preset now uses the current one anyway.
- The token budget is not the cause: greedy decoding with a budget of 8,192 tokens (7,965 after the prompt) reasons the whole time (31,967 characters, `Alternatively, perhaps ... I'll simulate the call`), never closes `[/THINK]` and never writes `[TOOL_CALLS]`. With the 2,048-token budget of the suite it is the same, and HuggingFace f32 on the original BF16 weights did the same for 2,100 greedy tokens.
- The parser is not the cause: sampled replies are `[THINK]...[/THINK][TOOL_CALLS]get_weather[ARGS]{"city": "Paris", ...}` and arrive as a `Thinking` block and a `ToolCall` (`mistral_think_then_call` in `output/mod.rs`; llama.cpp's Ministral 3 parser reads the same shape).
- Greedy decoding is the cause. At the model card's sampling (temperature 0.7, top-p 0.95), seeds 1 to 3 close `[/THINK]` and call the tool after 76, 137 and 66 completion tokens in the engine, and after 94, 132 and 149 new tokens in HuggingFace bf16 on the same prompt text.

The live suite therefore decodes the two tool scenarios of a model whose template forces reasoning (`Capabilities::always_thinks`) with the model's own sampling defaults (`LocalModel::sampling()`: the GGUF's `general.sampling.*` keys, else the built-in temperature 1.0, `top_k` 64, `top_p` 0.95) and the fixed request seed 1, so the run stays reproducible; it prints `MATRIX|<scenario>|DECODING|...`. Every other scenario stays greedy. Ministral 3 3B Reasoning passes both tool scenarios that way. An application that wants tool calls from such a model should sample (its model card recommends temperature 0.7) rather than set `temperature` to 0.

## Presets

`ModelSource::PRESETS` has 24 entries, each pinned to a repository commit and a SHA-256. The size is the file. The K/V column is `LocalModel::kv_cache_bytes` at the default `context_length` (8,192 positions, or the model's own maximum when smaller), in decimal units: the block pools, which hold twice `context_length` positions. Weights are resident at about the file size on top of it, except for a mixture-of-experts model on tiered placement.

| Preset | Architecture | Repository | Size | K/V |
|---|---|---|---|---|
| `gemma4-e2b-q4_0` (default) | `gemma4` | `ggml-org/gemma-4-E2B-it-GGUF` | 2.84 GB | 422 MB |
| `gemma4-e2b-q8_0` | `gemma4` | `ggml-org/gemma-4-E2B-it-GGUF` | 4.97 GB | 422 MB |
| `gemma4-e4b-q4_0` | `gemma4` | `ggml-org/gemma-4-E4B-it-GGUF` | 4.59 GB | 1.27 GB |
| `gemma4-e4b-q8_0` | `gemma4` | `ggml-org/gemma-4-E4B-it-GGUF` | 8.03 GB | 1.27 GB |
| `gemma3-270m-q8_0` | `gemma3` | `ggml-org/gemma-3-270m-it-GGUF` | 0.29 GB | 377 MB |
| `gemma3-1b-q4_0` | `gemma3` | `unsloth/gemma-3-1b-it-GGUF` | 0.72 GB | 539 MB |
| `qwen3-0.6b-q8_0` | `qwen3` | `Qwen/Qwen3-0.6B-GGUF` | 0.64 GB | 3.76 GB |
| `qwen3-1.7b-q8_0` | `qwen3` | `Qwen/Qwen3-1.7B-GGUF` | 1.83 GB | 3.76 GB |
| `r1-distill-qwen-1.5b-q4_0` | `qwen2` | `bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF` | 1.07 GB | 940 MB |
| `r1-distill-llama-8b-q4_0` | `llama` | `unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF` | 4.68 GB | 4.29 GB |
| `smollm3-3b-q4_0` | `smollm3` | `bartowski/HuggingFaceTB_SmolLM3-3B-GGUF` | 1.81 GB | 2.42 GB |
| `phi4-mini-q4_0` | `phi3` | `bartowski/microsoft_Phi-4-mini-instruct-GGUF` | 2.33 GB | 4.29 GB |
| `phi4-mini-q4_k_m` | `phi3` | `unsloth/Phi-4-mini-instruct-GGUF` | 2.49 GB | 4.29 GB |
| `granite3.3-2b-q4_0` | `granite` | `ibm-granite/granite-3.3-2b-instruct-GGUF` | 1.45 GB | 2.68 GB |
| `olmo2-1b-q4_0` | `olmo2` | `allenai/OLMo-2-0425-1B-Instruct-GGUF` | 0.89 GB | 2.15 GB (4,096 maximum) |
| `qwen3-30b-a3b-q4_0` | `qwen3moe` | `unsloth/Qwen3-30B-A3B-GGUF` | 17.38 GB | 3.22 GB |
| `olmoe-1b-7b-q4_0` | `olmoe` | `allenai/OLMoE-1B-7B-0125-Instruct-GGUF` | 3.93 GB | 2.15 GB (4,096 maximum) |
| `qwen3.5-0.8b-q8_0` | `qwen35` | `unsloth/Qwen3.5-0.8B-MTP-GGUF` | 0.83 GB | 403 MB, plus a 20.2 MB recurrent state |
| `qwen3.6-35b-a3b-q4_0` | `qwen35moe` | `bartowski/Qwen_Qwen3.6-35B-A3B-GGUF` | 20.84 GB | 671 MB, plus a 65.9 MB recurrent state |
| `muse-glimmer-30b-q4_0` | `muse-glimmer` | `bartowski/Muse-Glimmer-30B-GGUF` | 16.27 GB | 1.74 GB |
| `ministral3-3b-instruct-q4_0` | `mistral3` | `bartowski/mistralai_Ministral-3-3B-Instruct-2512-GGUF` | 2.05 GB | 3.49 GB |
| `ministral3-3b-reasoning-q4_0` | `mistral3` | `bartowski/mistralai_Ministral-3-3B-Reasoning-2512-GGUF` | 2.05 GB | 3.49 GB |
| `mistral-small-4-119b-q4_0` | `mistral4` | `bartowski/mistralai_Mistral-Small-4-119B-2603-GGUF` | 67.71 GB (two parts, merged at load) | 755 MB |
| `gemma4-e2b-mtp-f16` | `gemma4-assistant` | `unsloth/gemma-4-E2B-it-GGUF` | 0.17 GB | none: a drafter, reads the target's cache |

- The K/V cost depends on the model more than on its size: a model with many K/V heads pays far more per position. The pool holds twice `context_length` positions (see [Local limits and errors](@/providers/local/limits-and-errors.md#limits)), and a sliding-window group (the Gemma presets) holds its window plus about 270 positions per running sequence plus cached prefixes. Qwen 3 0.6B needs 3.76 GB of cache at 8,192 positions for a 0.64 GB file. Lower `context_length` to shrink it.
- A model trained on fewer positions loads at its maximum by default (OLMo 2 and OLMoE: 4,096). An explicit larger value fails with `context_length N exceeds the model maximum M`.
- `gemma4-e2b-mtp-f16` has `drafter: true`: it loads only through `LocalModelConfig::drafter` and cannot be the main model.
- `qwen3.5-0.8b-q8_0` carries the in-trunk prediction head, so `DrafterSource::InTrunk` works with it.
- `qwen3.6-35b-a3b-q4_0` has 256 experts per layer in 40 layers, 8 used, so every layer is MoE; resident it holds about 18.9 GiB of RAM, with `tiered:ram=4g` about 6 GiB and with `tiered:ram=600m` (the minimum, 8 experts per layer) about 2.4 GiB. Its file carries a prediction block that the engine ignores; see [Hybrid models](#hybrid-models).
- `muse-glimmer-30b-q4_0` is a 30B dense model (52 layers, hidden 6,656, 32 query and 2 K/V heads of 128). Its K/V is small for its size: only the 13 full layers cache the whole context, the other 39 keep a 2,048-position window. The file is bartowski's imatrix Q4_0: Q4_0 except `output.weight` and `attn_q` in 26 layers (Q6_K) and `ffn_down` in 6 layers (Q4_1), which run on candle's kernels instead of the row-batched ones; that costs about 6% of prefill speed against a pure Q4_0 conversion (104.5 s against 98.1 s for a 1,156-token prompt). It embeds the earlier Muse chat template, and the preset pins the repository's current `chat_template.jinja` (`meta-models/Muse-Glimmer-30B` at `cec94fcf`) in its place (the only published Q4_0 or Q8_0 whose embedded template equals the current one is ggml-org's Q8_0, 29.6 GB, twice the memory and about half the CPU speed). The earlier template renders the same bytes as the current one in 23 of 25 reference cases and differs only when the caller's system prompt itself mentions `Reasoning effort`: the current template rewrites that to `Reasoning strength`, this one keeps the caller's line and appends its own `Reasoning strength` directive after it. Provenance was checked against Meta's weights: for the first four layers of the original checkpoint the file's tensors differ from the converter's F16 output by relative L2 errors of 8.2e-2 to 9.1e-2 (Q4_0), 7.4e-2 (Q4_1) and 1.8e-2 (Q6_K, `attn_q` and `output`), the size of the quantization, and its norms, including the constant Q/K norms, are equal. On the CPU of the verification machine (16 cores, release build for the host) it decodes 3.3 to 3.4 tokens per second and prefills a 1,156-token prompt in 104.5 s (11.1 tokens per second); a warm prefix is reused (1.4 s to the first token). It always reasons (see [Reasoning](#reasoning)), so give requests a generous `max_tokens`. It does not fit a 16 GB GPU; see [GPU](@/providers/local/gpu.md).
- `ministral3-3b-instruct-q4_0` is Ministral 3 3B Instruct, the same architecture without reasoning, as bartowski's imatrix Q4_0. Every published GGUF of it embeds the December 2025 chat template (an earlier default system prompt, without the current tool-calling paragraph), so the preset pins the repository's current `chat_template.jinja` (`mistralai/Ministral-3-3B-Instruct-2512` at `b35d4dfe`) in its place; it renders like Python jinja2 on 44 cases. It is the preset that passes the Mistral `[TOOL_CALLS]name[ARGS]json` round trip live.
- `ministral3-3b-reasoning-q4_0` is Ministral 3 3B Reasoning (26 layers, 8 K/V heads of 128, a tied head) as bartowski's imatrix Q4_0 (mistralai publishes no Q4_0 of it). Every published GGUF of this model embeds the December 2025 chat template, which is not the repository's current one (`chat_template.jinja` at `4a36357c`, "update to match mistral-common": it merges consecutive messages, writes `reasoning`/`reasoning_content` back as `[THINK]` blocks and accepts system content lists), so the preset pins the repository's file and uses it in place of the embedded one (see [Chat templates](@/providers/local/model-sources.md#chat-templates)). The two render the same prompt for a plain turn, tools and a tool round trip, but not for reasoning in the history. It is the live test of the Mistral chat side (`[THINK]`, `[TOOL_CALLS]name[ARGS]json`) at a size a routine run affords: 0.9 s to load, 21.8 tokens per second decoding on the CPU. It always reasons, and it needs sampling to call a tool: decoded greedily it loops in its reasoning for the whole budget (see [Tool calls of a reasoning model](#tool-calls-of-a-reasoning-model) below), so the live tool scenarios run it with the model's sampling defaults and a fixed seed, and pass.
- `mistral-small-4-119b-q4_0` is bartowski's imatrix Q4_0 of Mistral Small 4 (36 layers, 128 experts per layer, 4 used, plus one shared expert; about 6 B parameters are read per token). The file is published as two parts (39.92 and 27.79 GB) that the loader downloads, verifies by SHA-256, merges into one 67.71 GB cached GGUF and deletes (about twice the file's size is free during the merge); most weights are Q4_0, the experts' `ffn_down` is Q4_1 in 4 layers, `attn_kv_a_mqa` and the shared experts are Q8_0 and `output` is Q6_K. Resident it holds 67.7 GB of RAM plus a 755 MB K/V pool at the default 8,192 positions (the cache is one latent row per position: 1,280 bytes per layer, see [Latent attention models](#latent-attention-models)); loading takes 209 s measured by the live suite (`MATRIX|load_seconds`; the loader builds 36 x 128 quantized expert tensors and splits the fused gate/up tensor, so it is not drive-bound). With `CUCA_LOCAL_EXPERTS=tiered:ram=48g` the experts live in a 51 GB RAM tier and the rest is read from the file, and the minimum budget is 4 experts per layer (about 2 GB). On the CPU of the verification machine (16 cores, release build for the host) the resident model decodes 10.5 to 11.3 tokens per second over two runs and prefills a 1,116-token prompt in 34.8 s and 43.6 s (26 to 32 tokens per second); a warm prefix is reused (0.5 s to the first token). It reasons by default (`[THINK]`) and can switch reasoning off. It has no GPU path: see [GPU](@/providers/local/gpu.md).
- Presets pin both the revision and the SHA-256; see [Model sources and presets](@/providers/local/model-sources.md#model-sources-and-presets) for the source types.

## Verification

### What was run

Three kinds of evidence back this page, from strongest to weakest:

- Unit tests in `src/provider/local/` run in CI on tiny random GGUF files and on committed fixtures: chat template renders, tokenizer pre-splitters, dequantization against candle, and the cross-architecture forward-pass tests (`TinySpec::all()`: `llama`, `qwen2`, `qwen3`, `phi3`, `smollm3`, `qwen3moe`, `olmoe`, `gemma3`, `granite`, `olmo2`, `ernie4_5`, `muse-glimmer`; `gemma4`, `qwen35`, `qwen35moe` and `mistral4` have their own builders; `mistral4` is checked against an f64 implementation of the transformers (expanded) latent attention and of the MoE FFN, and against the fused/split, dense-first and tiered variants of the same weights).
- The live suite `tests/provider_local.rs` runs one real preset per invocation. CI compiles it and every test prints a skip line, so the live results below come from recorded local runs, not from CI.
- Oracle runs against HuggingFace `transformers` (f32, CPU) use dumps under `.omp/local-models/ref/`, which is gitignored. Those results are reproducible only with that tooling, not from a clean checkout.

Per architecture:

- Verified in the live suite on the CPU: `gemma4`, `gemma3`, `qwen2` (R1 distill), `qwen3`, `qwen35`, `qwen35moe`, `phi3` (Phi 4 mini), `smollm3`, `granite`, `qwen3moe`, `olmoe`, `olmo2`, `muse-glimmer`, `mistral4`. `llama` runs live through `r1-distill-llama-8b-q4_0` and is also covered by tiny-model tests and by the converter oracle on the same model's checkpoint (see [Checkpoints](@/providers/local/model-sources.md#checkpoints)).
- Also verified in the live suite on Vulkan (AMD RX 9070 XT, Windows): `gemma4` (E2B Q4_0), `olmo2`, `granite` (Granite 3.3 2B), `qwen3moe` (Qwen3-30B-A3B with tiered experts).
- `ministral3-3b-reasoning-q4_0` in the live suite: the Ministral 3 3B Instruct Q4_0 (not a preset) passes all eight scenarios, tool round trips included; the Reasoning model passes all of them, the tool scenarios sampled (below).
- `ernie4_5`: tiny-model tests and a recorded oracle run only. It has no preset and has not run in the live suite.
- `qwen35moe` against HuggingFace `transformers` in f32 (`Qwen3_5MoeForCausalLM`, GGUF dequantized by `gguf`, routed experts dequantized per use): a 4-layer cut of the real Qwen3.6-35B-A3B file matches with the engine's dequantized weights to 1e-5 on the top-10 logits, top-1 equal and greedy output equal for 24 of 24 tokens on two prompts and 8 of 8 on a 321-token prompt. The full 40-layer file under the production Q4_0 path keeps the top-1 token on all three prompts, with a maximum absolute difference of 0.13 to 0.62 on the top-k logits. The oracle answers `The capital of France is **Paris**.`. Reading the value heads in grouped (HuggingFace) order instead of tiled order breaks the 4-layer match (top-10 overlap 0 to 2 of 10), which is how the tiled order was confirmed.
- `olmoe` against HuggingFace `transformers` in f32: the engine with dequantized weights matches the reference logits to a maximum absolute difference of 2e-5 and picks the same greedy tokens. `qwen3moe`: a 4-layer cut of the real Qwen3-30B-A3B weights matches to 1e-5, and the production Q4_0 engine's top-1 token differs from the oracle's only where two candidates are within a margin of 0.073. On Vulkan, both reproduced their reference replies with tiered experts (see [Tiered expert placement](@/providers/local/expert-tiers.md#output-guarantee)).
- `gemma4-assistant`: verified by `tests/provider_local_mtp.rs` as a drafter, and by tiny-model tests on the CPU and the GPU (the GPU ones only when `CUCA_LOCAL_DEVICE` names a GPU).
- Compile-checked only: every GPU backend other than Vulkan; see [Verified and compile-checked](@/providers/local/gpu.md#verified-and-compile-checked).

### Live results per preset

The suite ran each preset that has a model of its own on the CPU, in a release build for the host CPU, one model per run. Scenarios: a factual answer with exact usage, a prompt of about 1,300 tokens with a code word to retrieve, prefix reuse, and, where the template supports them, thinking on and off and a tool round trip (a `get_weather` call for Paris, then an answer from the result).

| Preset | Reasoning | Tool calls | Result |
|---|---|---|---|
| `gemma4-e2b-q4_0`, `gemma4-e2b-q8_0`, `gemma4-e4b-q4_0` | on and off | yes | all scenarios pass |
| `gemma4-e4b-q8_0` | on and off | yes | all pass except thinking, a limit of the model |
| `gemma3-270m-q8_0` | no | no syntax | long-context retrieval is a known limit; the rest pass |
| `gemma3-1b-q4_0` | no | no syntax | all applicable scenarios pass |
| `qwen3-0.6b-q8_0`, `qwen3-1.7b-q8_0` | on and off | yes | all scenarios pass |
| `smollm3-3b-q4_0` | on and off | yes | all seven run; thinking and the tool answer leg are known limits |
| `phi4-mini-q4_0`, `phi4-mini-q4_k_m` | no | no syntax | all applicable scenarios pass |
| `r1-distill-qwen-1.5b-q4_0` | always | no syntax | thinking and prefix reuse pass; long-context retrieval is a known limit |
| `r1-distill-llama-8b-q4_0` | always | no syntax | all applicable scenarios pass (the oracle prompt count has no dump) |
| `qwen3-30b-a3b-q4_0` | on and off | yes | all seven pass |
| `olmoe-1b-7b-q4_0` | no | no syntax | greedy text and usage, long-context retrieval and prefix reuse pass; the thinking and tool scenarios skip |
| `qwen3.5-0.8b-q8_0` | on and off | yes, XML | all seven pass on the CPU; not run on Vulkan |
| `qwen3.6-35b-a3b-q4_0` | on and off | yes, XML | all seven pass on the CPU, resident and with `tiered:ram=4g`, and on Vulkan with `tiered:device=4g,ram=12g` and `tiered:device=6g,ram=12g` |
| `olmo2-1b-q4_0` | no | no syntax | all applicable scenarios pass, on the CPU and on Vulkan |
| `ministral3-3b-instruct-q4_0` | no | yes, `[TOOL_CALLS]name[ARGS]json` | all eight scenarios pass on the CPU with the pinned repository template (the thinking scenarios skip: no reasoning in the template; the whole matrix: 57 s, 26 s load including the template download) |
| `ministral3-3b-reasoning-q4_0` | always (`[THINK]`) | yes, `[TOOL_CALLS]name[ARGS]json` | all eight scenarios pass on the CPU (thinking-off skips; the two tool scenarios decode with the model's sampling defaults and a fixed seed, below; the whole matrix: 185 s) |
| `mistral-small-4-119b-q4_0` | on and off (`reasoning_effort`) | yes, `[TOOL_CALLS]name[ARGS]json` | all eight scenarios pass on the CPU, resident (the whole matrix: 417 s) and with `tiered:ram=48g`; the oracle-prompt count skips (no dump) |
| `muse-glimmer-30b-q4_0` | always (`reasoning_strength`) | yes, channel protocol | all pass on the CPU with the pinned repository template (the two tool scenarios decode with the model's sampling defaults and a fixed seed; thinking-off skips, because the model cannot switch reasoning off; the whole matrix: 808 s, load 128 s) |
| `granite3.3-2b-q4_0` | switched on by the request | yes | all pass except long-context retrieval, a known limit |

The Vulkan engine ran the same suite for `gemma4-e2b-q4_0`, `olmo2-1b-q4_0` and `granite3.3-2b-q4_0`, with the same results as on the CPU.

### Known limits

A known limit is reported as `MATRIX|<scenario>|KNOWN-LIMIT` by the suite, with its evidence. The entries live in `known_limit` in `tests/provider_local.rs`.

- `gemma3-270m-q8_0` answers with its end-of-turn token at once on prompts of 750 and 1,110 tokens, while a 570-token prompt still retrieves the code word. The HuggingFace f32 model does the same, so this is the model.
- `gemma4-e4b-q8_0` answers `72` at once, without a reasoning block, to `What is 9 * 8?` with thinking on. This file derives from the base `google/gemma-4-E4B-it` checkpoint, and its Q4_0 sibling from the QAT checkpoint. At the first position `7` scores 28.77 against 28.37 for `<|channel>` in HuggingFace f32 and bf16, and 28.81 against 28.34 for the file's dequantized weights. The QAT-derived Q4_0 file opens the channel (29.28 against 28.91). On the harder question `What is 17 * 23? Think carefully.` the Q8_0 file does reason.
- `r1-distill-qwen-1.5b-q4_0` is marginal at retrieving a code word from a 1,100-token prompt. HuggingFace f32 answers `PELCAN` (misspelled). The Q4_0 model either answers or reasons for more than 4,096 tokens, depending on how its greedy trajectory falls on near-ties.
- `granite3.3-2b-q4_0` answers `PEICAN` where HuggingFace f32 answers `PELICAN`, on the 1,300-token retrieval prompt. HuggingFace on the same file's dequantized weights, and the engine's exact arithmetic, also give `PEICAN`: quantization flips a near-tie at the second token.
- `smollm3-3b-q4_0`, two scenarios that are quantized-arithmetic near-ties, not properties of the model. The engine's exact arithmetic (dequantized f32, no Q8_0 activation rounding) equals the dequantized reference run in HuggingFace. The production path rounds activations to Q8_0 and flips a close logit:
  - Thinking on `What is 9 * 8?`: exact arithmetic closes `</think>` at step 1,709 (HuggingFace f32 on the original weights: 535). The production path flips a 0.12-logit tie at the 154th reasoning token, then reasons for 3,070 tokens without concluding.
  - The tool round trip's answer leg with the tool still declared: exact arithmetic gives `The` 18.966 against the tool-call token 18.793 and answers. The production path flips it (18.663 against 19.237) and calls the tool again.
  - Give reasoning models a generous `max_tokens`.
- ERNIE 4.5 0.3B (`ernie4_5`, a pretrained model, not an instruct one) runs and matches the oracle numerically, but has no preset. Every Q8_0 and Q4_0 GGUF of it that was found (29 of 283 files in 54 Hugging Face repositories) lacks `tokenizer.ggml.add_space_prefix`, so llama.cpp's SentencePiece default (a space in front of the text after a special token) applies, and the model answers the capital question with `France` where the HuggingFace tokenizer gives `Paris`. Current llama.cpp conversions write that key as `false`.

### Numerical parity with HuggingFace

Method: HuggingFace `transformers`, f32, CPU, against the engine's CPU path, on prompts of 24 chat tokens, a 700-token prompt and, for Phi, a prompt of about 4,200 tokens.

| Path compared | Result |
|---|---|
| Reference arithmetic (weights dequantized to f32) against `transformers`, the first catalog models that are still supported (Gemma 3, Gemma 4, Qwen 3, R1 distills, SmolLM 3, Phi 4 mini) | top-1 token and top-10 set equal, maximum absolute logit difference below 5e-4, greedy output equal for 24 of 24 tokens on the chat prompt and 8 of 8 on the 700-token prompt (bounds recorded over a larger run that included models since removed); the Phi 4,200-token prompt exercises LongRoPE's long factors and matches |
| The same, OLMo 2, Granite 3.3, ERNIE 4.5 | maximum absolute logit difference 0.000 on the top-k, greedy output equal on every prompt |
| Production quantized CPU path against `transformers`, the same models | top-1 token kept on every prompt case; maximum absolute logit difference at most 1.13 (0.05 to 1.13 over the larger run), the cost of quantized weights |
| The same, the later models still supported | top-1 token kept on every prompt case |
| Muse Glimmer (`muse-glimmer`), tiny random model with every norm randomized, reference arithmetic | converted GGUF bit-identical to llama.cpp's `convert_hf_to_gguf.py` output (59 of 59 tensors); maximum absolute logit difference 2.6e-4 at all 48 positions, greedy output equal for 16 of 16 tokens |
| The same, a 4-layer cut of the real checkpoint converted to F16 (prompt of 2,600 tokens, past the 2,048 window) | maximum absolute logit difference 8.4e-5 at ten positions including 2,046 to 2,050, greedy output equal for 8 of 8 tokens |
| Ministral 3 (`mistral3`), the first 2 layers of Ministral-3-3B-Reasoning-2512 converted by this crate (F16) against `Ministral3ForCausalLM` in f32 | 300 tokens: 2.3e-5; 17,000 tokens (positions 16,382 to 16,386 cross the 16,384 original context where the query scaling starts): 2.8e-5; greedy equal. With the reference's `llama_4_scaling_beta` set to 0 the positions below 16,384 still agree and those from 16,384 differ by 0.42 to 0.52 |
| Mistral Small 4 (`mistral4`), the published Q4_0 GGUF cut to its first 2 layers (35 tensors), reference arithmetic against `Mistral4ForCausalLM` in f32 on the same dequantized tensors | 300-token prompt: maximum absolute logit difference 4.1e-5, greedy output equal for 8 of 8 tokens |
| The same, an 8,400-token prompt (positions 2,046 to 2,050 and 8,190 to 8,194 included; the Llama 4 query scaling is active from 8,192) | 7.0e-5, greedy equal. The reference computes its rotary angles in float64: transformers multiplies the position by the inverse frequency in float32, an angle error that grows with the position (5e-4 rad at 8k) and, against its own angles, shows as 1.1e-3 at 8,191. With the reference's query scaling switched off the positions below 8,192 still agree (5.4e-5) and those from 8,192 differ by 0.25 to 0.45 with another greedy output, so the check is sensitive to the scaling |
| The same cut converted from the FP8 checkpoint by this crate (F16), against `Mistral4ForCausalLM` on that GGUF's dequantized tensors | 300 tokens: 3.3e-5; 8,400 tokens: 7.7e-5; greedy equal |
| The same converted cut against `Mistral4ForCausalLM` on the FP8 checkpoint itself (every weight `fp8 * scale_inv` in f32) | 1.2e-3 to 4.5e-3 on both prompts, greedy equal for 8 of 8 tokens. The gap is the F16 rounding of the dequantized weights (an FP8 value times a bf16 scale has up to 12 significant bits, F16 keeps 11), not the engine: the previous row isolates it |
| Synthetic FP8 checkpoint with a dense first layer and two MoE layers (`first_k_dense_replace` 1), converted to F16 | 2.1e-4 at all 28 positions, greedy equal |
| Qwen 3.5 0.8B Q8_0, reference arithmetic | maximum absolute logit difference below 5e-5; the production path keeps the top-1 token |
| Qwen 3.6 35B-A3B Q4_0 (`qwen35moe`), 4-layer cut, reference arithmetic | maximum absolute logit difference 1e-5, greedy output equal for 24 of 24 and 8 of 8 tokens |
| The same file, 40 layers, production path | top-1 token kept on all three prompts; maximum absolute logit difference 0.13 to 0.62 |
| GPU (Vulkan) against the CPU dequantized reference | relative error 2e-6 to 2e-5, greedy token ids equal, on Qwen 3 0.6B and 1.7B, Gemma 3 270M and 1B, SmolLM 3, Gemma 4 E2B and the R1 distill |

On the CPU, every Q4_0 and Q8_0 matrix multiplication runs through cuca's own batch-invariant quantized kernel, written in safe Rust; other dtypes use candle. Prefill is row-exact: the logits of a prompt row are bit-identical whether the prompt is prefilled in one chunk, in several, or decoded one token at a time, so reusing a cached prefix never changes the output.

Since the paged K/V cache, attention reads the K/V blocks in place with an online-softmax kernel (per query row and head, so a row is bit-identical alone or in a batch) instead of candle's matmul attention. Its float rounding differs slightly from 0.6.x: logits move by about 0.04 (median) and up to 0.7 on the models measured, because the quantized matmuls round activations to Q8_0, so a greedy reply can differ from 0.6.x's where two logits are within about 0.1 of each other. In the 4-model matrix every first difference sat at a margin of 0.13 or less; OLMoE (tiered) replies were identical.

Measured prefill speed with that kernel (1,300-token prompt in chunks of 256, release build with `-C target-cpu=native`, 32-core CPU): `gemma4-e2b-q4_0` 86.5 tokens/s, `qwen3-0.6b-q8_0` 249.6.

## Using another GGUF

Any file whose architecture is in the table and whose tokenizer is one of the schemes above loads through `ModelSource::File(path)` or `ModelSource::hugging_face(repo, revision, file)`. The CPU reads every dtype candle reads. A model the engine accepts but that never ran through the live suite is unverified. The first check is that its rendered prompt and token ids equal its reference tokenizer's, which is how the catalog models were checked.

## Errors

Loading a file the engine cannot run fails with a `CucaError::Provider` that names the cause:

- An unknown architecture: ``unsupported GGUF architecture `NAME`; supported: ...``.
- An unknown tokenizer model or pre-tokenizer: ``unsupported tokenizer model `NAME` `` or ``unsupported pre-tokenizer `NAME` ``.
- A file candle cannot parse, such as one with an IQ-family tensor: ``` `PATH` is not a supported GGUF file: ... ```.
- A missing tensor or metadata key, including `tokenizer.chat_template`: ``GGUF metadata `KEY`: missing``.
- A tensor with an unexpected shape, or a row length that is not a multiple of its dtype's block size.
- A checkpoint whose value length differs from its key length (``GGUF metadata `ARCH.attention.value_length`: ... only equal key and value lengths are supported``), or with odd or zero attention key lengths.
- No end-of-generation token: the metadata names none and the vocabulary holds no known terminator.
- `DrafterSource::InTrunk` on a file without a prediction head: ``this `ARCH` file has no in-trunk multi-token-prediction head ...``.
- A `gemma4-assistant` drafter with a target that is not `gemma4`, or whose hidden width or vocabulary differs.

Device, configuration and download errors are on [Local (in-process)](@/providers/local/limits-and-errors.md#errors).
