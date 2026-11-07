+++
title = "Local model sources and presets"
description = "ModelSource variants, the pinned preset catalog entry points, how a Hugging Face file is downloaded, verified and cached, and how a safetensors checkpoint is converted to a cached GGUF."
template = "page.html"
weight = 4
+++

# Local model sources and presets

<dl class="page-facts">
<dt>In one line</dt>
<dd>A model comes from a GGUF file on disk, a Hugging Face GGUF file (pinned by revision and SHA-256 for the presets), or a safetensors checkpoint that is converted to a GGUF once; downloads resume, verify and cache.</dd>
<dt>You need</dt>
<dd>The <code>provider-local</code> feature. Network access once, for a Hugging Face source.</dd>
<dt>Read this if</dt>
<dd>You are choosing where the model file comes from, or debugging a download or the cache.</dd>
</dl>

## Model sources and presets

`LocalModelConfig::source` is a `ModelSource`:

- `ModelSource::File(path)` is a GGUF file already on disk; nothing is downloaded. A file named like a split part (`<base>-00001-of-0000N.gguf`) is merged with its siblings, see [Split GGUF files](#split-gguf-files).
- `ModelSource::HuggingFace { repo, revision, file, sha256, parts, template }` is a file on Hugging Face. `sha256: None` skips the checksum. `parts` is empty for a single file; for a split GGUF it lists every part with its digest (see [Split GGUF files](#split-gguf-files)). `template` is `None`, or a boxed `ModelTemplate { repo, revision, file, sha256 }`: a chat template file that replaces the GGUF's (see [Chat templates](#chat-templates)).
- `ModelSource::hugging_face(repo, revision, file)` builds the same value with `sha256: None`, so the download is not verified (`parts` is empty). Use a commit hash as the revision, because the cache is keyed by it and a moving ref such as `main` pins a stale file.
- `ModelSource::default_model()` is the preset `gemma4-e2b-q4_0`.
- `ModelSource::Checkpoint { repo, revision }` is a Hugging Face safetensors checkpoint, and `ModelSource::Dir(path)` a checkpoint directory on disk. Both are converted to a cached GGUF (see [Checkpoints](#checkpoints)).
- `"name".parse::<ModelSource>()?` accepts a preset name (ASCII case-insensitive), `repo:revision:file` (a GGUF, none of the three parts contains `:`), or `repo:revision` (a checkpoint: `repo` is `owner/name`). The result of `repo:revision:file` has no checksum; a checkpoint is pinned by its revision only. Anything else fails with `` unknown model `…`: expected a preset (…), `repo:revision:file` (a GGUF), or `repo:revision` (a checkpoint) ``.

`ModelSource::PRESETS` has 25 entries, each pinned to one Hugging Face commit and one SHA-256. A `ModelPreset` has `name`, `repo`, `revision`, `file`, `sha256`, `size_bytes`, `architecture`, `drafter`, `parts` (empty unless the model is a split GGUF; a `PresetPart` is `file`, `sha256`, `size_bytes`) and `template` (`None`, or a `PresetTemplate` of `repo`, `revision`, `file`, `sha256`: see [Chat templates](#chat-templates)). `preset.source()` returns its pinned `ModelSource`, and `ModelSource::preset(name)` finds one by name. A preset with `drafter: true` is used through `LocalModelConfig::drafter` and cannot be loaded as the main model. The table with sizes and K/V cost is on [Local models and formats](@/providers/local/models.md#presets).

```rust,name=List the presets
use cuca::ModelSource;

fn main() {
    for preset in ModelSource::PRESETS {
        println!(
            "{:<28} {:<17} {:>14} bytes  {}{}",
            preset.name,
            preset.architecture,
            preset.size_bytes,
            preset.repo,
            if preset.drafter { "  (drafter)" } else { "" },
        );
    }
}
```

```rust,name=Selecting a different model
use cuca::{LocalModel, LocalModelConfig, ModelSource};

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Qwen3 0.6B at Q8_0 instead of the default; the rest of the program is the
    // same as above.
    let model = LocalModel::load(LocalModelConfig {
        source: "qwen3-0.6b-q8_0".parse::<ModelSource>()?,
        ..Default::default()
    })
    .await?;
    println!("{model:?}");
    Ok(())
}
```

A file of an architecture the engine does not know fails to load with a `CucaError::Provider` that names the architecture and lists the supported ones. The chat template and the tokenizer come from the file; a tokenizer model the engine does not implement fails the same way.

## Chat templates

The chat template normally comes from the GGUF (`tokenizer.chat_template`). Two things replace it, as llama.cpp's `--chat-template-file` and vLLM's `--chat-template` do:

- `LocalModelConfig::chat_template: Option<String>`, the template text (the live tests and `examples/local_model.rs` read it from the file `CUCA_LOCAL_CHAT_TEMPLATE` names). It wins over everything else.
- A template a source pins: `ModelSource::HuggingFace::template`, which a preset sets through `ModelPreset::template`. The file is downloaded once into the cache like a model file (`<cache>/huggingface/<repo>/<revision>/<file>`), verified against its SHA-256, and read at load.

The replacement is used for everything derived from the template: the reasoning markers and whether a request can switch reasoning off, the tool-call syntax, the end-of-turn tokens, and the `enable_thinking`, `reasoning_effort` and `reasoning_strength` variables. The replacement text may also be one of llama.cpp's template *names* ([Named templates](#named-templates)). A template that does not compile, or cannot render a one-message conversation, fails the load with `CucaError::Config` (`chat_template override does not compile: …` or `… does not render a one-message conversation: …`), not the first request.

Published GGUFs often embed an older template than their original repository. `ministral3-3b-reasoning-q4_0` pins `chat_template.jinja` of `mistralai/Ministral-3-3B-Reasoning-2512` at revision `4a36357c` and `ministral3-3b-instruct-q4_0` the one of `mistralai/Ministral-3-3B-Instruct-2512` at `b35d4dfe` ("update to match mistral-common", 2026-07-15), because every GGUF of those models (mistralai, ggml-org, lmstudio-community, unsloth, bartowski) embeds the December 2025 template instead. `muse-glimmer-30b-q4_0` (bartowski's Q4_0, which embeds the earlier template, differing when a system prompt mentions "Reasoning effort") pins `chat_template.jinja` of `meta-models/Muse-Glimmer-30B` at revision `cec94fcf`. The one published Q4_0 or Q8_0 whose embedded template equals it is ggml-org's Q8_0, which needs twice the memory and decodes at about half the speed on the CPU, so the preset keeps the Q4_0 file.

### Named templates

llama.cpp's converter writes a template *name* instead of Jinja for some Mistral checkpoints, and llama.cpp renders it in C++ (`llm_chat_apply_template`). A `tokenizer.chat_template` (or a `chat_template` override) that is exactly one of the names this engine knows is expanded to the equivalent Jinja text at load, before anything is derived from the template (`src/provider/local/chat/builtin.rs`):

| Name | Used by | Renders |
|---|---|---|
| `mistral-v7-tekken` | Mistral Small 3.0 and 3.1, Devstral Small 2505 (tekken v7) | `[SYSTEM_PROMPT]{system}[/SYSTEM_PROMPT]`, `[INST]{user}[/INST]`, `{assistant}</s>` |
| `mistral-v7` | the SentencePiece variant | the same with one space after `[SYSTEM_PROMPT]` and `[INST]` and before each assistant turn |

The port follows llama.cpp's branch exactly: a system message is written where it stands (not merged into the first user turn), every role other than system and user (assistant, tool) is written as an assistant turn, no BOS is written (the tokenizer adds it), the generation prompt adds nothing, and declared tools are ignored, so the model never sees them. Verified byte for byte against llama.cpp's own expected output for the v7 case of `tests/test-chat-template.cpp` and against `mistral_common`'s Tekken v7 rendering of six conversations. The table holds only names a 2025 or later model uses: `mistral-v1`, `mistral-v3`, `mistral-v3-tekken` and llama.cpp's other built-in names belong to older models, and any other bare name (letters, digits, `.`, `-`, `_` only) fails the load loudly, naming the supported ones, instead of rendering the name as the prompt.

## Checkpoints

A Hugging Face checkpoint (`config.json`, `model.safetensors` or the shards listed by `model.safetensors.index.json`, and the tokenizer files) loads without a prior GGUF conversion:

```rust,name=Load a safetensors checkpoint
use cuca::{ConvertConfig, LocalModel, LocalModelConfig, ModelSource, WeightQuant};

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = LocalModel::load(LocalModelConfig {
        // `repo:revision`; a local directory is `ModelSource::Dir(path)`.
        source: "Qwen/Qwen3-0.6B:c1899de289a04d12100db370d81485cdf75e47ca".parse()?,
        convert: ConvertConfig { quant: WeightQuant::Q8_0 },
        ..Default::default()
    })
    .await?;
    println!("{model:?}");
    Ok(())
}
```

- The files are downloaded like any other source: `config.json` first, then the optional `generation_config.json`, `tekken.json`, `tokenizer.json`, `tokenizer_config.json`, `tokenizer.model`, `added_tokens.json`, `chat_template.jinja` and `model.safetensors.index.json` (a missing one is skipped), then the shards. An architecture without a converter fails with `` architecture `X` has no checkpoint converter; convert it to GGUF with llama.cpp `` before any shard is fetched.
- The conversion writes `<cache>/converted/<repo with / as -->/<revision>/model-<quant>.gguf` (`dir-<key>/model-<quant>.gguf` for a directory, where the key is the first 16 hex digits of a SHA-256 over the canonical path and each `config.json` and `*.safetensors` file's name, length and modification time, so an edited checkpoint converts again). `<quant>` is `f16`, `q8_0` or `q4_0` (`ConvertConfig::quant`, default `q8_0`; each is cached separately). The output is written to `<file>.partial` under an exclusive `<file>.lock` and renamed into place; an existing file is reused without touching the network or the shards.
- Every 2-D weight matrix whose rows are a multiple of 32 long (attention and FFN projections, expert stacks, the embeddings, the output head) is stored at the chosen precision; norms, biases, scales, RoPE factors, the router and the recurrent-layer parameters stay `f32`. At `q4_0` the token embedding and the output head are kept at Q8_0 and Gemma 4's `per_layer_model_proj` at F16, as in the published Gemma 4 E2B Q4_0 file (llama.cpp's Q4_0 mixes keep the embeddings above 4 bits too).
- Memory is bounded by one block of one tensor (at most 32 MiB of `f32`) plus its quantized form, whatever the model's size: the header is written first and the tensors stream behind it. The crate forbids `unsafe`, so the shards are read with positional reads, not memory-mapped.
- `on_download_progress` also reports the conversion: `downloaded_bytes` is the number of tensors written so far and `total_bytes` the tensor count.
- Before the file is renamed into place, its metadata is loaded through the engine's own hyperparameter and tokenizer readers; a missing key fails the conversion naming it. A checkpoint without a chat template fails with `checkpoint has no chat template` (the template is read from `chat_template.jinja`, `chat_template.json` or `tokenizer_config.json`).
- Not converted: mixture-of-experts Gemma 4, OLMo 3 sliding-window layers, Qwen 3 MoE with `norm_topk_prob = false`, OLMoE with `norm_topk_prob = true`, an explicit YaRN `attention_factor`, a pre-tokenizer the engine does not implement, block-scaled FP8 (`quantization_config.weight_block_size`), a vocabulary that ships only as `tekken.json` for an architecture other than the Llama family or at a tokenizer version other than v11 and v13 (see [Tekken-only checkpoints](#tekken-only-checkpoints)); each fails with a `CucaError::Config` that names the cause. The Gemma 4 assistant (`google/gemma-4-E2B-it-assistant`) converts without a vocabulary, as a drafter needs none.
- FP8 (`F8_E4M3`) checkpoints, such as Mistral Small 4's, are dequantized while converting: each weight is multiplied by its `<name>_scale_inv` companion (one scale for the tensor, or one per expert for a stacked `[E, rows, cols]` expert tensor) and written at the chosen precision, never as FP8; the `*activation_scale` tensors are dropped. Mistral Small 4 (`Mistral3ForConditionalGeneration` with a `mistral4` text model) converts to llama.cpp's `mistral4` layout: multi-head latent attention stored as one key/value head (`attn_q_a`, `attn_q_b`, `attn_kv_a_mqa`, the per-head `attn_k_b` transposed and `attn_v_b` split out of `kv_b_proj`), experts kept stacked with gate and up fused (`ffn_gate_up_exps`), the router, the shared expert and every `mistral4.*` key llama.cpp writes. `tokenizer.ggml.pre` is the name llama.cpp's probe-text hash table gives when the engine implements it, before the name read from the `tokenizer.json` structure: Mistral Small 4's Tekken expression is written as `pixtral`, as llama.cpp does. The engine runs that file on the CPU (see [Latent attention models](@/providers/local/models.md#latent-attention-models)); its metadata passes the engine's own reader before the file is written, and the file is checked against llama.cpp's output.

### Tekken-only checkpoints

Mistral Small 3.1 and 3.2, Magistral 2506-2509 and Devstral Small 2505/2507 publish their Hugging Face shards with `tekken.json` and no `tokenizer.json`. For the Llama family, a directory like that converts the way llama.cpp's `LlamaModel.set_vocab` does (`_set_vocab_mistral`, gguf-py `MistralVocab`), from `src/provider/local/convert/tekken.rs`:

- The vocabulary is `default_vocab_size` entries: `default_num_special_tokens` special tokens (the file's `special_tokens`, completed with `<SPECIAL_{i}>`), written as control tokens, then the first `default_vocab_size - default_num_special_tokens` ranked byte strings (`token_bytes`, base64) shown in GPT-2's byte alphabet. Scores are 0, the tokenizer is `gpt2` with the `tekken` pre-tokenizer, `<s>`, `</s>`, `<unk>` and `<pad>` give the BOS, EOS, unknown and padding ids, `add_bos_token` is true and `add_eos_token` false.
- The merges are rebuilt from the ranks: from rank 256 up, every split of the token into two ranked halves is a merge, ordered by the halves' ranks, and all merges by the merged token's rank.
- The chat template is the community one llama.cpp bundles for the file's tokenizer `version`: v11 the Mistral Small 3.2 template, v13 the Devstral Small 2507 one (copies are in `src/provider/local/convert/templates/`; the checkpoint's own `chat_template.jinja` is not read on this path, as in llama.cpp). For v7 (Mistral Small 3.0 and 3.1, Devstral Small 2505; such a `tekken.json` lists no special tokens, so mistral_common's 20 default ones apply) llama.cpp writes the *name* `mistral-v7-tekken`, and so does this converter: the engine expands that name to Jinja at load (see [Named templates](#named-templates)). For v1 and v3 (pre-2025 formats) llama.cpp writes names this engine does not port, and it refuses every other version; both fail the conversion with a `CucaError::Config` that names the version. A file that splits numbers into right-aligned groups of three (`tekken-num3`) is refused too: no supported model uses it.
- Verified against llama.cpp at the pinned commit on 2-layer truncations of `mistralai/Mistral-Small-3.2-24B-Instruct-2506` (v11) and `mistralai/Mistral-Small-3.1-24B-Instruct-2503` (v7, template name) with the real `tekken.json` and `generation_config.json` and no `tokenizer.json` (21 tensors, F16): every metadata key, including the `general.sampling.*` ones and the chat template, and every tensor are identical (0 differences in both). The converted vocabulary encodes 1,536 texts to the same ids as `mistral_common` (`tokenizer_corpus_matches_reference_ids`, `CUCA_TOK_GGUF` and `CUCA_TOK_CASES`).

The conversion also writes `general.sampling.*` from `generation_config.json`, as llama.cpp's converter does (see [Local configuration](@/providers/local/configuration.md)).

The conversion is a port of llama.cpp's `convert_hf_to_gguf.py` (commit `67107ac34f62882d8b53ea41ad68010ea635b9be`) and was verified against it: the script was run on the same checkpoints and the two GGUF files compared key by key and tensor by tensor. The F16 files were identical for Gemma 3, Gemma 4 (and the Gemma 4 assistant, but for the tokenizer it does not carry), OLMo 2, Granite 3.3, ERNIE 4.5, SmolLM 3, Phi 4 mini, a `Qwen2ForCausalLM` checkpoint and the first two layers of Ministral 3 3B Reasoning (`mistral3`) and of Mistral Small 3.1 24B (`Mistral3ForConditionalGeneration` with a `mistral` text model, written as `mistral3`), and for synthetic Qwen 3 MoE, OLMoE, Llama with attention biases, Qwen 3.5 and Qwen 3.5 MoE checkpoints. The remaining differences are one-ulp `f32` rounding in two generated tensors (Llama 3 `rope_freqs`, Qwen 3.5 `ssm_a`), a tied `lm_head` that is not written again, and the Qwen 3.5 prediction-block count, which is read from the tensors present instead of `config.json`. Mistral Small 4 was compared the same way, with every tensor and every key equal at F16 (the tokenizer arrays and the chat template included) on a synthetic FP8 checkpoint (stacked FP8 experts with per-expert and per-tensor scales, activation-scale tensors and vision tensors present, one dense layer), on the first two layers of the real `mistralai/Mistral-Small-4-119B-2603` FP8 shards and, for `MistralForCausalLM`, on the first two layers of `mistralai/Mistral-Small-24B-Instruct-2501`; the converted vocabularies give the same ids as Hugging Face `tokenizers` (4,537 texts) and `mistral_common` (1,536 texts).

## Split GGUF files

`llama-gguf-split` cuts a model of more than about 50 GB into `<base>-00001-of-0000N.gguf` files. Each part is a GGUF with `split.no` (from 0), `split.count` and `split.tensors.count`; the first part holds the metadata (and in some sets no tensor at all), the others hold tensors. The engine reads one file, so a set is merged once, like `llama-gguf-split --merge`:

- The merged file has the metadata of part 1 without the `split.*` keys, every tensor of every part with recomputed offsets, and the tensor data copied part by part (never more than 8 MiB in memory). Tensor types the engine cannot load (the IQ family, MXFP4) are copied as they are; the loader rejects them as for any file.
- A Hugging Face source whose `file` is a split name (a preset or `repo:revision:dir/base-00001-of-0000N.gguf`) downloads every part, merges them into `<cache>/huggingface/<repo>/<revision>/<dir>/<base>.gguf` and deletes the parts, so the merged file is the cache entry. A preset pins every part's SHA-256 and size (`ModelPreset::parts`), and each part is verified as it downloads; a source from a string verifies none.
- `ModelSource::File(path)` naming any part merges the parts in the same directory into `<cache>/merged/<key>/<base>.gguf`. The key covers the directory and the name, length and modification time of every part, so a changed part merges again. The parts stay where they are.
- The merge writes `<merged>.partial` and renames it under `<merged>.lock`, as the converter does, so concurrent loaders merge once. A Hugging Face source holds the lock from the first download to the end of the merge.
- Disk use: while merging, the parts and the merged file coexist, so a download needs about twice the model's size free (a 67.7 GB Q4_0 needs about 135 GB) until the parts are deleted; afterwards the cache holds the merged file only. A merged file that is deleted means a new download of every part. Parts that failed verification or merging stay for a retry (a verified part is not downloaded again).
- `on_download_progress` reports each part's download, then the merge as bytes copied out of the bytes to copy.
- A missing part (`CucaError::Io` naming it), a part whose `split.count`, `split.no` or `split.tensors.count` disagree with the set, a file that is not a split part, or a tensor that appears in two parts (`CucaError::Config` naming the file) fail the load; nothing is left at the merged path.

## Download and cache

A Hugging Face source is downloaded once from `https://huggingface.co/{repo}/resolve/{revision}/{file}` into the cache directory.

- The cache root is `LocalModelConfig::cache_dir`, else `CUCA_CACHE_DIR`, else the platform directory. Empty environment values count as unset.
- The platform directory is `%LOCALAPPDATA%\cuca` on Windows (else `%USERPROFILE%\AppData\Local\cuca`), `$HOME/Library/Caches/cuca` on macOS, and `$XDG_CACHE_HOME/cuca` on Linux (else `$HOME/.cache/cuca`).
- The file lands at `<cache>/huggingface/<repo with / as -->/<revision>/<file>`.
- An interrupted download resumes from the `<file>.partial` file with an HTTP `Range` request, and restarts from byte 0 once per call if the server ignores or rejects the range.
- The SHA-256 is computed while streaming, covering any resumed bytes. A mismatch deletes the partial file and fails the load.
- The verified file is renamed into place atomically, so a final file that exists is complete.
- An exclusive lock file, `<file>.lock`, serializes concurrent loaders in this process or another. The empty file stays in the cache on purpose: deleting it after unlock would race with a process still blocked on it.
- Memory use during the download is bounded by a 1 MiB write buffer and one network chunk.
- `LocalModelConfig::on_download_progress` takes a `DownloadProgressCallback` that receives `DownloadProgress { downloaded_bytes, total_bytes }` at the start, about every mebibyte and at the end. `total_bytes` is `None` when the server reports no size. The callback runs on the engine thread during `load`, so keep it cheap. A checkpoint conversion reports tensors converted out of the tensor count in the same fields, and a split merge reports bytes copied out of the bytes to copy.
- `hf_token`, or `HF_TOKEN` when it is `None`, is sent as a bearer token; the pinned preset files download anonymously.
- The download uses a private HTTP client and runtime, separate from the `CucaClient` one.
