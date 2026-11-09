+++
title = "Local models and formats"
description = "The GGUF architectures provider-local runs and what each adds to the common forward pass, the preset catalog, the weight formats and their kernels, and what a GGUF that is not a preset must satisfy."
template = "page.html"
weight = 6
+++

# Local models and formats

<dl class="page-facts">
<dt>In one line</dt>
<dd>The engine loads a GGUF file the way llama.cpp does: the architecture, the tokenizer and the chat template all come from the file, and the stream has the same <code>Text</code>, <code>Thinking</code> and <code>ToolCall</code> blocks whatever the model.</dd>
<dt>You need</dt>
<dd>The <code>provider-local</code> feature. Loading, configuration and errors are on <a href="@/providers/local/_index.md">Local (in-process)</a>.</dd>
<dt>Read this if</dt>
<dd>You are choosing a model, loading a GGUF that is not a preset, or checking which architecture and weight format a file needs.</dd>
</dl>

The loader reads every dimension from the file's metadata and tensor shapes, selects the forward-pass behavior from `general.architecture`, builds the tokenizer from `tokenizer.ggml.*` ([Tokenizers](@/providers/local/tokenizers.md)), and renders prompts with the Jinja template in `tokenizer.chat_template` ([Chat templates](@/providers/local/chat-templates.md)). llama.cpp is the design reference; none of it is linked or called. What was run on which model is on [Local verification](@/providers/local/verification.md).

## Architectures

`general.architecture` selects the behavior from one table in `src/provider/local/model/arch.rs`. The table has 20 entries: 18 models and 2 drafters. An architecture outside it fails the load with a `CucaError::Provider` that names it and lists them all:

```text,name=Expected error for an unknown architecture
unsupported GGUF architecture `NAME`; supported: gemma4, gemma3, llama, qwen2, qwen3, phi3, smollm3, granite, qwen3moe, olmoe, qwen35, qwen35moe, olmo2, ernie4_5, muse-glimmer, mistral3, mistral4, gemma4-assistant, mistral4-eagle, deepseek41
```

| Architecture | Model families | Notes |
|---|---|---|
| `gemma4` | Gemma 4 (E2B, E4B) | |
| `gemma3` | Gemma 3 (270M, 1B) | |
| `llama` | Llama-family models published as `llama` (for example DeepSeek-R1-Distill-Llama, Mistral-style models) | |
| `qwen2` | DeepSeek-R1-Distill-Qwen and other Qwen 2.5-family models | |
| `qwen3` | Qwen 3 | |
| `qwen35` | Qwen 3.5 (0.8B) | hybrid, see [Hybrid models](#hybrid-models) |
| `qwen35moe` | Qwen 3.6 MoE (Qwen3.6-35B-A3B) | hybrid, mixture of experts |
| `phi3` | Phi 4 mini | |
| `smollm3` | SmolLM 3 | |
| `granite` | Granite 3.3 (dense) | |
| `qwen3moe` | Qwen 3 MoE (Qwen3-30B-A3B) | mixture of experts |
| `olmoe` | OLMoE (OLMoE-1B-7B-0125) | mixture of experts |
| `olmo2` | OLMo 2 | |
| `ernie4_5` | ERNIE 4.5 (dense) | no preset |
| `muse-glimmer` | Muse Glimmer 30B | |
| `mistral3` | Ministral 3 (3B, 8B, 14B), Devstral Small 2 | |
| `mistral4` | Mistral Small 4 (119B, 128 experts) | latent attention, see [Latent attention](#latent-attention); not tensor parallel |
| `gemma4-assistant` | the Gemma 4 multi-token-prediction drafter | only through `LocalModelConfig::drafter` |
| `mistral4-eagle` | the Mistral Small 4 EAGLE drafter (`mistralai/Mistral-Small-4-119B-2603-eagle`) | only through `LocalModelConfig::drafter`; CPU only |
| `deepseek41` | DeepSeek-V4.1 Flash | hyper-connections and compressed sparse attention, see [DeepSeek-V4.1](#deepseek-v4-1); not tensor parallel |

The `dflash` architecture is not in the table above: it loads only through `LocalModelConfig::drafter`, as the DFlash drafter of Muse Glimmer on the CPU or, with `dflash.hyper_connection.count` in the file, as the DSpark drafter of `deepseek41`; as a main model it fails with an error that says so. A drafter architecture as a main model fails the same way. Drafters, their pairings and limits are on [Speculative decoding](@/providers/local/drafting.md).

Every architecture except the drafters runs on the CPU and on a GPU; the device limits and what was run are on [Run on a GPU](@/providers/local/gpu.md).

What each architecture adds to the common forward pass:

- `gemma4`: sliding and full layers with their own head sizes and RoPE bases, shared K/V layers (the last `shared_kv_layers` layers read an earlier layer's cache), per-layer embeddings, a weightless value norm, a per-layer output scale, and a final logit soft-cap. The per-layer embedding table is never loaded: each token's row is read from the GGUF file.
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
- `muse-glimmer`: a dense decoder with a `[sliding, sliding, sliding, full]` pattern from `attention.sliding_window_pattern`. Sliding layers apply `NORM` RoPE with `rope.freq_base` and full layers apply no RoPE. Every layer has a separate `attn_gate` projection of the normed input whose sigmoid multiplies the attention output before `attn_output`, a scale-less per-head RMS norm on Q and K, and Gemma-style post-attention and post-FFN norms with an epsilon of 1e-8 where the other norms use `attention.layer_norm_rms_epsilon`. The embeddings go through a scale-less RMSNorm after the lookup (no `sqrt(n_embd)` scale), and the logits are multiplied by `logit_scale` before the final soft-cap.
- `mistral3`: Llama layers (NORM RoPE, a tied or separate head, `attention.key_length` honoured) with YaRN whose magnitude follows `rope.scaling.yarn_log_multiplier` and the Llama 4 query scaling. After RoPE the query of position `p` is multiplied by `1 + attention.temperature_scale * ln(1 + floor(p / original_context_length))`, which is 1 inside the original context and applies to every layer. Without `temperature_scale` (or at 0) there is no scaling.
- `mistral4`: multi-head latent attention, a mixture of experts and Mistral's YaRN; see [Latent attention](#latent-attention).
- `qwen35` and `qwen35moe`: a hybrid stack; see [Hybrid models](#hybrid-models). `qwen35moe` has a mixture of experts in every layer: a softmax over all experts, the top k, the selected weights renormalised to sum to one, plus a shared expert (a dense gated FFN every token runs) whose output is scaled by `sigmoid(ffn_gate_inp_shexp . x)` and added to the routed output. The routed experts honor `ExpertPlacement`; the shared expert and the router stay resident.
- `gemma4-assistant`: a four-layer, query-only decoder that reads the K/V cache of a `gemma4` target. It pairs only with a `gemma4` model of the same hidden width and vocabulary.
- `mistral4-eagle`: two latent-attention layers over a dense FFN with a K/V cache of its own, sharing the target's embedding and LM head.
- `deepseek41`: hyper-connection residual streams, latent attention with a sliding window and compressed sparse attention, per-layer Engram table lookups and a `sqrtsoftplus`-routed mixture of experts; see [DeepSeek-V4.1](#deepseek-v4-1).

The common forward pass covers grouped-query attention, causal and sliding-window masks, `NORM` and `NEOX` RoPE (partial rotary, linear and YaRN scaling, per-pair factors), RMSNorm, SiLU and GELU gated FFNs, tied and separate output heads, and mixture-of-experts routing. Optional tensors (biases, QK-norm, fused projections, a separate `output.weight`) are detected by their presence in the file.

### DeepSeek-V4.1

`deepseek41` runs on the CPU and on a GPU (see [Run on a GPU](@/providers/local/gpu.md#deepseek-v4-1-on-a-gpu)). The real DeepSeek-V4.1-Flash-1017 has run in the engine on the CPU and on Vulkan: it passes eight live scenarios on the CPU and matches the torch oracle, at about 0.2 tokens/s with its experts streaming from an HDD-class volume; the measurements, with their method, are on [Verification](@/providers/local/verification.md#deepseek-v4-1-real-checkpoint-on-the-cpu). Its drafter is the DSpark block drafter of the same checkpoint, a `dflash` file that runs on the CPU only (see [Speculative decoding](@/providers/local/drafting.md)). A checkpoint is converted once to a split GGUF set; see [Local checkpoint conversion](@/providers/local/checkpoints.md#deepseek-v4-1).

- The residual stream is `hyper_connection.count` copies of the hidden state per token. Each attention and FFN block reads a weighted sum of the copies and writes back through coefficients predicted from the stream, normalised by `hyper_connection.sinkhorn_iterations` Sinkhorn passes.
- Attention has low-rank queries, one shared vector per position as key and value (its last `rope.dimension_count` values rotate and the output is rotated back), a sink logit per head and a grouped low-rank output projection. Every layer attends to the last `attention.sliding_window` positions.
- A layer with a non-zero entry in `attention.compress_ratios` (1 or 2) also attends to up to `attention.indexer.top_k` compressed rows. Row `j` pools the tokens `j * ratio` to `(j + 1) * ratio - 1` and is visible to a query once its last token has passed. An indexer scores the visible rows and keeps the best. `attention.kv_source_layers` compute the rows, `attention.indexer.source_layers` the picks, and the layers after a source reuse them. With `attention.indexer.candidate_source_layer` the later scoring layers look only inside the blocks that layer liked best.
- The cached window keys, the compressed rows, the indexer's vectors and the attention output carry the quantization the model trained with: FP8 for the window keys and the output, FP4 for the rows and the indexer.
- Engram layers (`ple.layers`) hash each token with the tokens before it into rows of `per_layer_token_embd.weight`, which gate a residual add. The table is never loaded into memory: a pool of reader threads reads the distinct rows of a pass from the file, and a row cache bounded by `LocalModelConfig::engram` keeps the most recently used ones (see [Engram cache](@/providers/local/configuration.md#engram-cache)).
- The router scores experts with `sqrt(softplus(logit))` (`expert_gating_func` 4), selects with the `exp_probs_b` bias but weights with the unbiased scores, and clamps the SwiGLU activations of the routed and the shared expert (`swiglu_clamp_exp`, `swiglu_clamp_shexp`). Experts stored as MXFP4 run through the row-batched kernel with Q8_0 activations.
- A window group holds 16 positions per block like every K/V layer. A compressed group holds one block per `16 * ratio` tokens, so `prompt_cache_usage.read_tokens` is a multiple of 32. A compressor keeps a half-finished pair of tokens between calls; a rollback within the last call restores it, and the scheduler limits a speculative round to 17 rows.

### Latent attention

`mistral4` replaces the per-head K and V projections by a compressed latent. For Mistral Small 4 (36 layers, hidden 4,096, 32 heads) a layer computes `q = q_b(norm(q_a(x)))` (a rank of 1,024; every head is `nope 64 + rot 64` wide) and `kv = kv_a(x)`: its first 256 values, normed, are the latent `c` shared by every head, and its last 64 are the rotary key shared by every head too.

In the reference form head `h` reads keys `K_h c` and values `V_h c`, and a position's score is `q_nope . (K_h c) + q_rot . k_rot`. The engine never builds those per-head keys and values. It absorbs `K_h` into the query (`q_abs = K_h^T q_nope`, 256 values), so the score is `[q_abs | q_rot] . [c | k_rot]` against one cached row of 320 floats per position and layer, and the weighted sum of the values is `V_h (sum_j p_j c_j)`, taken over the first 256 columns of the same cached rows.

The K/V cache is therefore 1,280 bytes per position and layer (46,080 bytes over the 36 layers, 755 MB for the pool of a default 8,192-position context), against 1.18 MB per position for the expanded heads. The `attn_k_b` and `attn_v_b` tensors are held as f32 (216 MB in all) and applied one row at a time, so a row is bit-identical alone or in a batch.

- Rotary embedding: adjacent pairs of the 64 rotary dimensions of every query head and of the shared key, YaRN (factor 128 over 8,192, `beta_fast` 32, `beta_slow` 1). The cosine and sine tables carry no magnitude scale; the YaRN `mscale` (`1 + 0.1 ln 128`, from `rope.scaling.yarn_log_multiplier`) enters twice into the softmax scale instead, `mscale^2 / sqrt(128)` as in llama.cpp's `deepseek2` graph and transformers' `Mistral4Attention`.
- YaRN magnitude for the other architectures: with `rope.scaling.yarn_log_multiplier` (`mscale_all_dim` in the checkpoint) `m` other than 0, the cosine and sine tables are scaled by `get_mscale(factor, 1) / get_mscale(factor, m)` with `get_mscale(f, x) = 1 + 0.1 x ln f`. Without the key the table keeps ggml's `1 + 0.1 ln f` boost.
- Llama 4 query scaling: the whole query of position `p` is multiplied by `1 + 0.1 ln(1 + floor(p / 8192))`, which is 1 for the first 8,192 positions. A GGUF may carry `attention.temperature_scale` and `attention.temperature_length` (zero switches the scaling off). Without them the engine applies the checkpoint's own configuration, 0.1 and 8,192, which llama.cpp's converter does not write for this model, so llama.cpp runs it unscaled past 8,192 positions.
- Mixture of experts: a softmax over all 128 experts, the top 4, their weights renormalised and multiplied by `expert_weights_scale`, plus one shared expert that is added as it is, without a gate. The gate and up matrices of the experts are one tensor (`ffn_gate_up_exps`: for every expert its gate rows, then its up rows). Experts may be tiered (see [Tiered expert placement](@/providers/local/expert-tiers.md)); the fused tensor is read as two strided halves, and the router, the shared expert and the attention stay resident.
- Leading dense blocks (`leading_dense_block_count`, as in DeepSeek models) run a dense gated FFN in those layers.
- Not supported: grouped or sigmoid expert routing, and a model without the low-rank query (`attention.q_lora_rank`).
- On a GPU the layer caches one `[latent | rotary key]` row per position with no value pool. It does not split over `shards`.

### Hybrid models

Qwen 3.5 mixes gated delta-net layers with gated full-attention layers. A delta-net layer keeps a causal convolution and a recurrent state per sequence instead of a K/V cache. Every `full_attention_interval`-th layer (every fourth in the 0.8B model) is gated full attention with partial NEOX RoPE. When a layer has fewer key heads than value heads (`ssm.group_count` below `ssm.time_step_rank`, as in `qwen35moe`: 16 and 32), value head `h` reads key head `h % ssm.group_count`, the tiled order llama.cpp's converter writes.

- A `*-MTP` file carries a trailing prediction block that `DrafterSource::InTrunk` drafts with. The trunk loaders ignore it. The block of a `qwen35moe` file is not loaded: `DrafterSource::InTrunk` on it fails with a `CucaError::Config` that names the mixture-of-experts FFN.
- Memory for the 0.8B model: 24,576 bytes of K/V per position (6 attention layers with 2 K/V heads of 256), plus a recurrent state of 20.2 MB per sequence that does not grow with the context.
- The state cannot be truncated. Prefix reuse resumes from a snapshot of the state that the engine takes while prefilling, at the block boundary of 16 positions below the last end-of-turn token of the prompt (the end of the last complete message). The next turn of a conversation re-prefills the few tokens after that point, and its output equals a cold run's. The prefix store keeps at most `max_sequences + 4` snapshots, least recently used out first.
- Each running sequence owns a state slot, so `max_sequences` slots exist. The in-trunk drafter serves one sequence.
- Rolling back a speculative verification keeps a copy of the state from before the call plus the per-row inputs of the recurrence. A verification call covers at most 17 rows (the pending token and 16 drafts). For the 0.8B model that is about 40 MB per slot (live state and pre-call copy) plus 445 KB per verified row, and 20.2 MB per snapshot in the prefix store, none of it growing with the context.
- On a GPU a hybrid model keeps the same state, snapshots and cache fallback. It keeps no verify record, since the in-trunk head is CPU only. See [Run on a GPU](@/providers/local/gpu.md#verified-and-compile-checked).

## Presets

`ModelSource::PRESETS` has 25 entries: 23 models and 2 drafters. Each is pinned to a repository commit and a SHA-256; the entry points are on [Model sources](@/providers/local/model-sources.md#presets). Sizes are the file in decimal units. The K/V column is `LocalModel::kv_cache_bytes` at the default `context_length` (8,192 positions, or the model's own maximum when smaller) with the default batching (4 sequences, steps of 256 rows): the block pools, which hold twice `context_length` positions. Each value is computed from the GGUF header of the pinned file with the pool layout code (`KvLayout::pool_bytes`), without loading the model. Weights are resident at about the file size on top of it, except for a mixture-of-experts model on tiered placement.

| Preset | Architecture | Repository | Size | K/V |
|---|---|---|---|---|
| `gemma4-e2b-q4_0` (default) | `gemma4` | `ggml-org/gemma-4-E2B-it-GGUF` | 2.84 GB | 481 MB |
| `gemma4-e2b-q8_0` | `gemma4` | `ggml-org/gemma-4-E2B-it-GGUF` | 4.97 GB | 481 MB |
| `gemma4-e4b-q4_0` | `gemma4` | `ggml-org/gemma-4-E4B-it-GGUF` | 4.59 GB | 1.47 GB |
| `gemma4-e4b-q8_0` | `gemma4` | `ggml-org/gemma-4-E4B-it-GGUF` | 8.03 GB | 1.47 GB |
| `gemma3-270m-q8_0` | `gemma3` | `ggml-org/gemma-3-270m-it-GGUF` | 0.29 GB | 451 MB |
| `gemma3-1b-q4_0` | `gemma3` | `unsloth/gemma-3-1b-it-GGUF` | 0.72 GB | 647 MB |
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
| `ministral3-3b-reasoning-q4_0` | `mistral3` | `bartowski/mistralai_Ministral-3-3B-Reasoning-2512-GGUF` | 2.05 GB | 3.49 GB |
| `ministral3-3b-instruct-q4_0` | `mistral3` | `bartowski/mistralai_Ministral-3-3B-Instruct-2512-GGUF` | 2.05 GB | 3.49 GB |
| `mistral-small-4-119b-q4_0` | `mistral4` | `bartowski/mistralai_Mistral-Small-4-119B-2603-GGUF` | 67.71 GB (two parts) | 755 MB |
| `gemma4-e2b-mtp-f16` | `gemma4-assistant` | `unsloth/gemma-4-E2B-it-GGUF` | 0.17 GB | none: a drafter, reads the target's cache |
| `muse-glimmer-30b-dflash-q4_0` | `dflash` | `ggml-org/Muse-Glimmer-30B-GGUF` | 1.45 GB | none from the pool: a drafter with its own 2,048-position K/V ring |

- The K/V cost depends on the model more than on its size: a model with many K/V heads pays far more per position. Qwen 3 0.6B needs 3.76 GB of cache at 8,192 positions for a 0.64 GB file. A sliding-window group (the Gemma presets) holds `ceil((window - 1 + rows) / 16) + 2` blocks of 16 positions per running sequence plus cached prefixes. The recurrent-state sizes of the two hybrid presets (20.2 MB and 65.9 MB per sequence) are recorded figures, not recomputed from the header. Lower `context_length` to shrink the cache; the pool rules are on [Local limits and errors](@/providers/local/limits-and-errors.md#limits).
- A model trained on fewer positions loads at its maximum by default (OLMo 2 and OLMoE: 4,096). An explicit larger value fails with `context_length N exceeds the model maximum M`.
- `gemma4-e2b-mtp-f16` and `muse-glimmer-30b-dflash-q4_0` have `drafter: true`: they load only through `LocalModelConfig::drafter`. The first pairs with the Gemma 4 E2B presets, the second with `muse-glimmer-30b-q4_0` on the CPU and proposes a block of up to 15 tokens per pass.
- `qwen3.5-0.8b-q8_0` carries the in-trunk prediction head, so `DrafterSource::InTrunk` works with it.
- `qwen3.6-35b-a3b-q4_0` has 40 layers of 256 experts, 8 used, so every layer is a mixture of experts. Its file carries a prediction block that the engine ignores. RAM budgets for tiered placement are on [Tiered expert placement](@/providers/local/expert-tiers.md).
- `muse-glimmer-30b-q4_0` is a dense model (52 layers, hidden 6,656, 32 query and 2 K/V heads of 128). Only its 13 full layers cache the whole context; the other 39 keep a 2,048-position window. The file is bartowski's imatrix Q4_0: Q4_0 with a Q6_K `output.weight` and `attn_q` in 26 layers (candle's kernel serves K-quants) and a Q4_1 `ffn_down` in 6. It always reasons, so give requests a generous `max_tokens`. The preset pins the repository's chat template.
- `ministral3-3b-reasoning-q4_0` (26 layers, 8 K/V heads of 128, a tied head) is bartowski's imatrix Q4_0, and mistralai publishes no Q4_0 of it. It always reasons (`[THINK]`) and needs sampling to call a tool; see [Known limits](@/providers/local/verification.md#known-limits). The preset pins the repository's chat template.
- `ministral3-3b-instruct-q4_0` is the same architecture without reasoning, also bartowski's imatrix Q4_0, with its template pinned.
- `mistral-small-4-119b-q4_0` is bartowski's imatrix Q4_0 of Mistral Small 4 (36 layers, 128 experts per layer, 4 used, plus one shared expert). The file is published as two parts (39.92 and 27.79 GB) that the loader downloads and verifies by SHA-256, then opens as one set (see [Split GGUF files](@/providers/local/model-sources.md#split-gguf-files)). The weights are Q4_0 with Q4_1 and Q5_0 in places, Q8_0 for the shared experts and the latent projection, and a Q6_K `output`. Resident it holds 67.7 GB of RAM plus the 755 MB K/V pool. It reasons by default (`[THINK]`) and can switch reasoning off.
- Presets pin both the revision and the SHA-256. DeepSeek-V4.1 has no preset: it loads as a checkpoint, see [Local checkpoint conversion](@/providers/local/checkpoints.md#deepseek-v4-1).

## Weight formats

| Dtype | CPU | GPU |
|---|---|---|
| Q4_0, Q8_0 | stays quantized; cuca's batch-invariant quantized matmul kernel | repacked at load into the kernels' own block-major words; stays quantized |
| Q4_1 | stays quantized; cuca's batch-invariant kernel | repacked like Q4_0, with the f16 minimum of each block |
| Q5_0, Q5_1 | stays quantized; cuca's batch-invariant kernel | dequantized row by row at load, requantized to Q8_0 |
| K-quants (Q2_K to Q6_K) | stays quantized; candle's quantized matmul, one activation row at a time | dequantized row by row at load, requantized to Q8_0 |
| Q8_1 | stays quantized; candle's quantized matmul (no test covers it) | dequantized row by row at load, requantized to Q8_0 |
| F32, F16, BF16 | held as f32 | uploaded as f32 (4 bytes per element) |
| MXFP4 | stays quantized; cuca's batch-invariant kernel | stays quantized in the expert kernels |
| IQ4_NL, I8 | read row by row as an embedding or per-layer table only; a weight matrix fails the load | the same |
| Other IQ-family types | the file opens and the load fails when such a tensor is requested | the same |

- Norm vectors, biases and scalars are f32 whatever their stored dtype. Resident weights are about the file size.
- A tensor whose type the engine cannot run makes the load fail with ``tensor `NAME` has type TYPE, which this engine cannot run`` or `` `PATH` is not a supported GGUF file: ... ``.
- A GPU requantizes every quantized dtype other than Q4_0, Q4_1, Q8_0 and MXFP4 to Q8_0. Each element moves by at most half a Q8_0 step, a smaller error than the one a Q4_K, Q5_K or Q6_K file already carries, but GPU logits then differ from the CPU's by more than float rounding.
- A GPU file that mixes Q4_0, Q4_1 and Q8_0 weights, as the Q4_0 presets with a Q8_0 embedding do, keeps every weight in its own format, one copy each. A weight over 256 MiB on the GPU is split into row chunks.
- Mixture of experts: expert tensors are split once at load into one quantized 2-D weight per expert, so memory is the file size. `LocalModelConfig::experts` can leave them in the file instead; see [Tiered expert placement](@/providers/local/expert-tiers.md).

### CPU kernels

Every CPU matrix multiplication of a Q4_0, Q4_1, Q5_0, Q5_1 or Q8_0 weight, mixture-of-experts and tiered experts included, runs through cuca's own batch-invariant quantized kernel in `src/provider/local/model/qmm.rs`, written in safe Rust. Each dtype is paired with the activation format ggml pairs it with (Q8_0 for Q4_0, Q5_0 and Q8_0, Q8_1 for Q4_1 and Q5_1), so a quantized file loses no more accuracy than under llama.cpp's CPU backend. Other dtypes, the K-quants among them, use candle's kernel, whose result for a row also depends on that row alone.

A dtype is never dispatched to different kernels by batch size, because a row's bits must not depend on it. Prefill is row-exact: the logits of a prompt row are bit-identical whether the prompt is prefilled in one chunk, in several, or decoded one token at a time, so reusing a cached prefix never changes the output on the CPU. Attention reads the K/V blocks in place with an online-softmax kernel per query row and head, so a row is bit-identical alone or in a batch as well.

MXFP4 weights run through the same `qmm.rs` kernel on the CPU, with Q8_0 activations.

The K-quants stay on candle on purpose. Measured on a Ryzen 9 9950X3D2 (3D V-cache L3, release build for the host CPU), candle's K-quant prefill matched Q4_0's (`phi4-mini-q4_k_m` against `phi4-mini-q4_0`), a row-batched safe-Rust K-quant kernel was 2 to 3 times slower than candle's at one row, and the Q4_1, Q5_0 and Q5_1 kernels are 2 to 4 times faster than candle's from four rows on.

## Loading a GGUF that is not a preset

Any file whose architecture is in the table and whose tokenizer is one of the supported schemes loads through `ModelSource::File(path)` or `ModelSource::hugging_face(repo, revision, file)`. A split set (`<base>-00001-of-0000N.gguf`) loads from any of its parts. The CPU reads F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, the K-quants and MXFP4.

The load fails with a `CucaError::Provider` that names the cause for:

- an unknown architecture, tokenizer model or pre-tokenizer;
- a tensor candle cannot read and no row reader covers, such as an IQ-family one other than IQ4_NL;
- a missing tensor or metadata key, `tokenizer.chat_template` included;
- a tensor with an unexpected shape, or a row length that is not a multiple of its dtype's block size;
- a value length that differs from the key length (``GGUF metadata `ARCH.attention.value_length`: ... only equal key and value lengths are supported``), or odd or zero attention key lengths;
- no end-of-generation token: the metadata names none and the vocabulary holds no known terminator;
- `DrafterSource::InTrunk` on a file without a prediction head, or a `gemma4-assistant` drafter with a target that is not `gemma4` or whose hidden width or vocabulary differs.

The device, configuration and download errors are on [Local limits and errors](@/providers/local/limits-and-errors.md#errors).

A model the engine accepts but that never ran through the live suite is unverified. The first check is that its rendered prompt and token ids equal its reference tokenizer's.
