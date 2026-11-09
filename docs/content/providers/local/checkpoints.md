+++
title = "Local checkpoint conversion"
description = "How a Hugging Face safetensors checkpoint is converted in process to a cached GGUF: the supported checkpoint classes, the weight precisions, the files it reads, FP8 and tekken.json handling, the DeepSeek-V4.1 split set, and what is not converted."
template = "page.html"
weight = 5
+++

# Local checkpoint conversion

<dl class="page-facts">
<dt>In one line</dt>
<dd><code>ModelSource::Checkpoint</code> and <code>ModelSource::Dir</code> convert a safetensors checkpoint once, in pure Rust, to the GGUF llama.cpp's converter would write, then load it like any other file.</dd>
<dt>You need</dt>
<dd>The <code>provider-local</code> feature. Network access once for a <code>Checkpoint</code>, free disk for the converted file.</dd>
<dt>Read this if</dt>
<dd>You have a Hugging Face checkpoint but no GGUF, or a conversion fails and you need its rules.</dd>
</dl>

## Load a checkpoint

```rust,name=Load a safetensors checkpoint
use cuca::{ConvertConfig, LocalModel, LocalModelConfig, ModelSource, WeightQuant};

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = LocalModel::load(LocalModelConfig {
        // `repo:revision`; a local directory is `ModelSource::Dir(path)`.
        source: "Qwen/Qwen3-0.6B:c1899de289a04d12100db370d81485cdf75e47ca".parse()?,
        convert: ConvertConfig { quant: WeightQuant::Q8_0, ..Default::default() },
        ..Default::default()
    })
    .await?;
    println!("{model:?}");
    Ok(())
}
```

A checkpoint is a directory with `config.json`, `model.safetensors` or the shards listed by `model.safetensors.index.json`, and the tokenizer files. `ModelSource::Checkpoint` downloads it pinned to `revision`. Source forms are on [Model sources](@/providers/local/model-sources.md#model-sources).

## Supported checkpoints

`architectures[0]` of `config.json` selects the converter. Every converter writes an architecture of [Local models and formats](@/providers/local/models.md#architectures).

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
| `Gemma4ForCausalLM`, `Gemma4ForConditionalGeneration` | `gemma4` |
| `Gemma4AssistantForCausalLM` | `gemma4-assistant`, a drafter |
| `Olmo2ForCausalLM`, `Olmo3ForCausalLM` (without sliding layers) | `olmo2` |
| `OlmoeForCausalLM` | `olmoe` |
| `Ernie4_5ForCausalLM` | `ernie4_5` |
| `MuseGlimmerForConditionalGeneration` (text tower; vision tensors are skipped) | `muse-glimmer` |
| `Ministral3ForCausalLM`, or `Mistral3ForConditionalGeneration` whose `text_config.model_type` is `ministral3` or `mistral` (text tower, FP8 weights dequantized) | `mistral3` |
| `Mistral3ForConditionalGeneration` whose `text_config.model_type` is `mistral4` (text tower, FP8 weights dequantized) | `mistral4` |
| `MuseGlimmerAssistantModel` | the DFlash drafter, `dflash` |
| `DeepseekV41ForCausalLM` (text tower; vision and aligner tensors are skipped) | `deepseek41`, and its DSpark drafter as a `dflash` file |
| `EagleMistralLarge3ForCausalLM`, read from `params.json` and `consolidated.safetensors` | `mistral4-eagle`, a drafter |

`mistral3` covers Ministral 3, Devstral Small 2 and, through `Mistral3ForConditionalGeneration` with a `mistral` text model, Mistral Small 3.1 and 3.2 and Magistral 2509. Any other class fails with `` architecture `X` has no checkpoint converter; convert it to GGUF with llama.cpp `` before a shard is downloaded. A `Mistral3ForConditionalGeneration` with any other text model fails naming it.

The drafter rows are loaded through `LocalModelConfig::drafter`; their pairings are on [Speculative decoding](@/providers/local/drafting.md).

## Files and cache

- `Checkpoint` downloads `config.json` first (or `params.json` for the EAGLE drafter), then the optional `generation_config.json`, `tekken.json`, `tokenizer.json`, `tokenizer_config.json`, `tokenizer.model`, `added_tokens.json`, `chat_template.jinja`, `chat_template.json` and `model.safetensors.index.json` (a missing one is skipped), then the shards. Progress and token use follow [Download and cache](@/providers/local/model-sources.md#download-and-cache).
- The chat template comes from `chat_template.jinja`, else `chat_template.json`, else `tokenizer_config.json`. A checkpoint with none fails with `checkpoint has no chat template`.
- The output is `<cache>/converted/<repo with / as -->/<revision>/model-<quant>.gguf`, or `<cache>/converted/dir-<key>/model-<quant>.gguf` for a directory. A non-default `ConvertConfig::engram_quant` adds `-engram-<name>` to the file stem. The key is the first 16 hex digits of a SHA-256 over the canonical path and the name, length and modification time of `config.json`, `params.json` and every `*.safetensors` file, so an edited checkpoint converts again. DeepSeek-V4.1 writes a split set next to that path, see [DeepSeek-V4.1](#deepseek-v4-1).
- The file is written to `<file>.partial` under an exclusive `<file>.lock` and renamed into place. An existing file is reused without touching the network or the shards.
- `on_download_progress` reports the conversion: `downloaded_bytes` is the number of tensors written so far and `total_bytes` the tensor count.
- Memory is bounded by two blocks of one tensor (at most 32 MiB of `f32` each, one read ahead while the other is encoded) plus its quantized form, whatever the model's size. The header is written first and the tensors stream behind it. The crate forbids `unsafe`, so the shards are read with positional reads, not memory-mapped.
- Before the rename, the metadata is loaded through the engine's own hyperparameter and tokenizer readers; a missing key fails the conversion naming it. The two drafter architectures that carry no vocabulary skip the tokenizer check.

## Weight precision

`ConvertConfig::quant` is `f16`, `q8_0` (default) or `q4_0`, and each precision is converted and cached separately.

- Every 2-D weight matrix whose rows are a multiple of 32 long (attention and FFN projections, expert stacks, the embeddings, the output head) is stored at the chosen precision. Norms, biases, scales, RoPE factors, the router and the recurrent-layer parameters stay `f32`.
- At `q4_0` the token embedding and the output head stay at Q8_0, and Gemma 4's `per_layer_model_proj` at F16, as in the published Gemma 4 E2B Q4_0 file.
- `ConvertConfig::engram_quant` is `q8_0` (default), `q4_0` or `bf16` and sets the storage type of DeepSeek-V4.1's Engram table; every other architecture ignores it.

## Sampling defaults and tokenizer

- The conversion writes `general.sampling.*` from `generation_config.json`, as llama.cpp's converter does; see [SamplingConfig](@/providers/local/configuration.md#samplingconfig).
- `tokenizer.ggml.pre` is the name llama.cpp's probe-text hash table gives when the engine implements it, else the name read from the structure of `tokenizer.json`. Mistral Small 4's Tekken expression is written as `pixtral`, as llama.cpp does. The names the engine implements are on [Local tokenizers](@/providers/local/tokenizers.md#pre-tokenizers).

## FP8 checkpoints

FP8 (`F8_E4M3`) checkpoints are dequantized while converting. Each weight is multiplied by its `<name>_scale_inv` companion, one scale for the tensor or one per expert for a stacked `[E, rows, cols]` expert tensor, then written at the chosen precision, never as FP8. The `*activation_scale` tensors are dropped. A Mistral-format checkpoint carries per-tensor `qscale_weight` scales instead.

Mistral Small 4 converts to llama.cpp's `mistral4` layout:

- multi-head latent attention stored as one key/value head (`attn_q_a`, `attn_q_b`, `attn_kv_a_mqa`, the per-head `attn_k_b` transposed and `attn_v_b` split out of `kv_b_proj`);
- experts kept stacked with gate and up fused (`ffn_gate_up_exps`);
- the router, the shared expert and every `mistral4.*` key llama.cpp writes.

## DeepSeek-V4.1

`DeepseekV41ForCausalLM` is the release format of `deepseek-ai/DeepSeek-V4.1-Flash-1017`: FP8 dense weights on a 32 by 32 `ue8m0` scale grid and FP4 routed experts, 510 GB of shards. It has its own converter, which writes the layout llama.cpp's `deepseek41` loader reads. A `weight_block_size` other than 32 by 32 fails the conversion.

```rust,name=Load DeepSeek-V4.1 with its drafter
use cuca::{DrafterSource, LocalDevice, LocalModel, LocalModelConfig, ModelSource};

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let source: ModelSource =
        "deepseek-ai/DeepSeek-V4.1-Flash-1017:1bae0e1088993b9b82de08f59de5be5a34e42170".parse()?;
    let model = LocalModel::load(LocalModelConfig {
        // The same checkpoint converts to the DSpark drafter's file as well.
        drafter: Some(DrafterSource::File(source.clone())),
        source,
        device: LocalDevice::Cpu,
        ..Default::default()
    })
    .await?;
    println!("{model:?}");
    Ok(())
}
```

The first load downloads the shards and converts them; a later load opens the cached set. The weights are about 298 GB, so a machine with less memory places the routed experts with `ExpertPlacement::Tiered` (see [Tiered expert placement](@/providers/local/expert-tiers.md)). The DSpark drafter runs on the CPU only; see [Speculative decoding](@/providers/local/drafting.md).

The conversion writes these files into the directory of the nominal path `model-<quant>.gguf`:

- `model-<quant>-00001-of-00003.gguf` holds the metadata and no tensor, `-00002-` every tensor except the Engram table, and `-00003-` the Engram table `per_layer_token_embd.weight`. A model with no Engram layers writes two parts.
- Parts 2 and 3 are renamed into place before part 1, so a present part 1 marks a finished set, which a later load reuses without the network or the shards.
- `dspark-model-<quant>.gguf` is the DSpark drafter, a `dflash` file, written when a `DrafterSource::File` names the same checkpoint source.
- The set opens natively, so its parts can move to different volumes; see [Split GGUF files](@/providers/local/model-sources.md#split-gguf-files).

The storage type of each tensor class:

| Checkpoint tensors | GGUF type |
|---|---|
| FP8 dense weights, the embedding and the output head | the `quant` precision, `q8_0` by default; the embedding and the head stay Q8_0 at `q4_0` |
| FP4 routed experts | MXFP4, repacked block for block; `f16` at `quant = f16`, dequantized exactly |
| compressor and indexer key and gate projections | F16 |
| hyper-connection, router, attention sink and norm tensors, `engram_gate` | F32 |
| the Engram tables, joined into one tensor | `engram_quant` |
| `conf_proj.weight` of the drafter | a 1-D vector |

FP8 weights are dequantized exactly from their scale grid before they are stored. The vision tower, the aligner and the `exp_probs_b_vl` tensors are not converted.

- Sizes at the default precision: the backbone part is 297.96 GB, the Engram table 208.90 GB at `q8_0` (about 110.6 GB at `q4_0`, about 393.2 GB at `bf16`) and the drafter 7.91 GB.
- Before it writes, the converter checks the free space of the output volume against the tensor bytes plus 256 MiB. Too little space fails with a `CucaError::Config` that names the sizes and suggests another cache directory, a smaller `quant` or `engram_quant: q4_0`. A volume whose free space cannot be read skips the check.
- Memory is bounded by block buffers, never a whole tensor: the 196.6 billion values of the Engram tables stream through. A release build converting this checkpoint at the default precision peaked at about 220 MB of resident memory.
- The conversion is bound by disk throughput. The same run wrote the 507 GB set in about 2 h 44 min on an HDD-class volume at about 52 MB/s.
- `on_download_progress` reports bytes written out of the bytes to write.

## Tekken-only checkpoints

Mistral Small 3.1 and 3.2, Magistral 2506 to 2509 and Devstral Small 2505 and 2507 publish their shards with `tekken.json` and no `tokenizer.json`. For `llama`, `smollm3`, `granite` and `mistral3` such a directory converts the way llama.cpp's `LlamaModel.set_vocab` does (`_set_vocab_mistral`), in `src/provider/local/convert/tekken.rs`:

- The vocabulary is `default_vocab_size` entries: `default_num_special_tokens` special tokens (the file's `special_tokens`, completed with `<SPECIAL_{i}>`) written as control tokens, then the first `default_vocab_size - default_num_special_tokens` ranked byte strings (`token_bytes`, base64) shown in GPT-2's byte alphabet. Scores are 0, the tokenizer is `gpt2` with the `tekken` pre-tokenizer, `<s>`, `</s>`, `<unk>` and `<pad>` give the BOS, EOS, unknown and padding ids, `add_bos_token` is true and `add_eos_token` false.
- The merges are rebuilt from the ranks: from rank 256 up, every split of the token into two ranked halves is a merge, ordered by the halves' ranks, and all merges by the merged token's rank.
- The chat template is the community one llama.cpp bundles for the file's tokenizer `version`: v11 the Mistral Small 3.2 template and v13 the Devstral Small 2507 one (copies are in `src/provider/local/convert/templates/`; the checkpoint's own `chat_template.jinja` is not read on this path, as in llama.cpp). For v7 (Mistral Small 3.0 and 3.1, Devstral Small 2505; such a `tekken.json` lists no special tokens, so `mistral_common`'s 20 default ones apply) the converter writes the name `mistral-v7-tekken`, which the engine expands at load; see [Named templates](@/providers/local/chat-templates.md#named-templates).
- Versions v1 and v3 fail the conversion, and so does any other version, with a `CucaError::Config` that names the version. A vocabulary that splits numbers into right-aligned groups of three (`tekken-num3`) is refused by name.

A tekken-only directory of any other architecture fails, asking for the repository's `tokenizer.json`.

## Not converted

Each of these fails with a `CucaError::Config` that names the cause:

- mixture-of-experts Gemma 4, and OLMo 3 sliding-window layers;
- Qwen 3 MoE with `norm_topk_prob = false`, shared experts or `mlp_only_layers`, and OLMoE with `norm_topk_prob = true`, because the engine fixes each model's expert weight rule;
- an explicit YaRN `attention_factor`, and a rope scaling type the engine does not implement;
- block-scaled FP8 (`quantization_config.weight_block_size`), except DeepSeek-V4.1's 32 by 32 grid;
- a pre-tokenizer the engine does not implement;
- a Muse Glimmer checkpoint with rope scaling, attention biases or a `post_norm_eps` other than 1e-8.

The Gemma 4 assistant (`google/gemma-4-E2B-it-assistant`) converts without a vocabulary, as a drafter needs none.

## Equivalence with llama.cpp and the oracle

The converter is a port of llama.cpp's `convert_hf_to_gguf.py` at commit `67107ac34f62882d8b53ea41ad68010ea635b9be`.

- The script was run on the same checkpoints at F16 and its GGUF files compared key by key and tensor by tensor with the converter's. They are identical for Gemma 3, Gemma 4, the Gemma 4 assistant (except the tokenizer it does not carry), OLMo 2, Granite 3.3, ERNIE 4.5, SmolLM 3, Phi 4 mini, a `Qwen2ForCausalLM` checkpoint, the first two layers of Ministral 3 3B Reasoning and of Mistral Small 3.1 24B, and synthetic Qwen 3 MoE, OLMoE, Llama with attention biases, Qwen 3.5 and Qwen 3.5 MoE checkpoints.
- The differences left are one-ulp `f32` rounding in two generated tensors (Llama 3 `rope_freqs`, Qwen 3.5 `ssm_a`), a tied `lm_head` that is not written again, and the Qwen 3.5 prediction-block count, which is read from the tensors present instead of `config.json`.
- Mistral Small 4 was compared the same way with every key and tensor equal at F16, tokenizer arrays and chat template included, on a synthetic FP8 checkpoint (stacked FP8 experts with per-expert and per-tensor scales, activation-scale tensors, vision tensors, one dense layer), on the first two layers of the real `mistralai/Mistral-Small-4-119B-2603` FP8 shards and, for `MistralForCausalLM`, on the first two layers of `mistralai/Mistral-Small-24B-Instruct-2501`.
- The tekken-only path was compared on 2-layer truncations of `mistralai/Mistral-Small-3.2-24B-Instruct-2506` (v11) and `mistralai/Mistral-Small-3.1-24B-Instruct-2503` (v7, template name) with the real `tekken.json` and `generation_config.json` and no `tokenizer.json`: every metadata key, the `general.sampling.*` ones and the chat template included, and every tensor are identical.
- The comparison scripts are not in the repository.
- Converted `f16` checkpoints also match the transformers oracle in `convert::oracle` and `tests/provider_local_convert.rs`: top-1 equal, top-10 overlap of at least 9 of 10, logits within 2 % relative L2. All eight pinned repositories pass (OLMo 2, Granite 3.3, ERNIE 4.5, SmolLM 3, Gemma 3 1B through the `unsloth/gemma-3-1b-it` mirror, OLMoE-1B-7B-0125-Instruct, DeepSeek-R1-Distill-Llama-8B, Qwen 3.5 0.8B). Both tests need the oracle dumps (`CUCA_LOCAL_REF_DIR`) and print `SKIP:` without them. Results on a live matrix are on [Local verification](@/providers/local/verification.md#converted-checkpoints).
- DeepSeek-V4.1 is checked against a tiny release-format oracle checkpoint written by the torch reference (`CUCA_LOCAL_DS41_REF`): converted at `f16`, the main and the drafter files hold exactly the oracle's float tensors and metadata, and the default output keeps native MXFP4 experts. These tests print `SKIP:` without the variable.
- A conversion of the real 510 GB checkpoint was compared at the header level with the files `ggml-org` publishes: all 62 metadata keys are identical, and 1003 of the 1043 published tensor names match with identical shapes. The other 40 are the `exp_probs_b_vl` tensors the converter skips. The remaining differences are the documented storage types.
