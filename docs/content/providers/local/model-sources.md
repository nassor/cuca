+++
title = "Model files and downloads"
description = "Where a local model file comes from: preset names, Hugging Face files, files on disk and converted checkpoints, plus the download cache, split GGUF files and checkpoint conversion."
template = "page.html"
weight = 3
+++

# Model files and downloads

<dl class="page-facts">
<dt>In one line</dt>
<dd><code>source</code> says where the model comes from: a preset name, a GGUF file on Hugging Face or on your disk, or a safetensors checkpoint that CUCA converts once to a cached GGUF. Downloads resume, verify and cache.</dd>
<dt>You need</dt>
<dd>The <code>provider-local</code> feature. Network access once, for a Hugging Face source.</dd>
<dt>Read this if</dt>
<dd>You are choosing where the model file comes from, or debugging a download, a split set, the cache or a conversion.</dd>
</dl>

A language model comes as a GGUF file, which holds its weights, tokenizer and chat template, or as a Hugging Face safetensors checkpoint, which CUCA converts once to a cached GGUF. Every source form ends at a model file on disk, and `LocalModel::load` reads that file. This page lists the forms and what happens to each.

<div class="dgm">
<div class="dgm-scroll">
<svg viewBox="0 0 660 262" role="img" aria-labelledby="src-title src-desc">
  <title id="src-title">Four source forms resolve to a model file that the model loads</title>
  <desc id="src-desc">Four source forms on the left. A preset name and a repo:revision:file string go through a download step that resumes, checks the SHA-256 and writes into the cache. A file= path goes straight to the model file. A repo:revision checkpoint, or a dir= checkpoint directory, goes through a one-time conversion that downloads the shards first and writes the model file into the cache. All three paths end at one model file, or one split set, that LocalModel::load reads.</desc>
  <defs>
    <marker id="src-arrow" viewBox="0 0 8 8" refX="7" refY="4"
            markerWidth="6" markerHeight="6" orient="auto">
      <path d="M0 0.6 L7 4 L0 7.4 z" fill="var(--muted-foreground)"/>
    </marker>
  </defs>
  <text class="t-title" x="6" y="20">From a source string to a loaded model</text>
  <rect class="blk" x="6" y="40" width="186" height="44" rx="6"/>
  <text class="t-sm" x="16" y="59">"qwen3-0.6b-q8_0"</text>
  <text class="t-sm" x="16" y="74">a preset name</text>
  <rect class="blk" x="6" y="96" width="186" height="44" rx="6"/>
  <text class="t-sm" x="16" y="115">"repo:revision:file"</text>
  <text class="t-sm" x="16" y="130">a GGUF on Hugging Face</text>
  <rect class="blk" x="6" y="152" width="186" height="44" rx="6"/>
  <text class="t-sm" x="16" y="171">file="/models/x.gguf"</text>
  <text class="t-sm" x="16" y="186">a GGUF on your disk</text>
  <rect class="blk" x="6" y="208" width="186" height="44" rx="6"/>
  <text class="t-sm" x="16" y="227">"repo:revision"</text>
  <text class="t-sm" x="16" y="242">a checkpoint, or dir=</text>
  <rect class="blk blk-bnd" x="250" y="40" width="170" height="100" rx="6"/>
  <text class="t-lbl t-mid t-bnd" x="335" y="70">download once</text>
  <text class="t-sm t-mid" x="335" y="90">resumes, checks SHA-256</text>
  <text class="t-sm t-mid" x="335" y="106">cache/huggingface/</text>
  <rect class="blk blk-ctl" x="250" y="208" width="170" height="44" rx="6"/>
  <text class="t-lbl t-mid t-ctl" x="335" y="227">convert once</text>
  <text class="t-sm t-mid" x="335" y="242">cache/converted/</text>
  <rect class="blk blk-data" x="470" y="66" width="184" height="176" rx="6"/>
  <text class="t-lbl t-mid t-data" x="562" y="140">model file on disk</text>
  <text class="t-sm t-mid" x="562" y="158">one file or a split set</text>
  <text class="t-sm t-mid" x="562" y="174">then LocalModel::load</text>
  <path class="arw" d="M194 62 H246" marker-end="url(#src-arrow)"/>
  <path class="arw" d="M194 118 H246" marker-end="url(#src-arrow)"/>
  <path class="arw" d="M194 230 H246" marker-end="url(#src-arrow)"/>
  <path class="arw arw-data" d="M194 174 H466" marker-end="url(#src-arrow)"/>
  <path class="arw arw-data" d="M422 90 H466" marker-end="url(#src-arrow)"/>
  <path class="arw arw-data" d="M422 230 H466" marker-end="url(#src-arrow)"/>
</svg>
</div>
<p class="dgm-cap"><b>Only a file= path skips the cache.</b> A downloaded file and a converted checkpoint land in the cache directory, so the second load makes no network call.</p>
</div>

## The source forms

`LocalModelConfig::source` is a `ModelSource`. In KDL a string is a preset name, `repo:revision:file` or `repo:revision`. Everything else is a property or a child node.

Each form is shown as its KDL and its Rust spelling.

- `source "qwen3-0.6b-q8_0"`<br>`"qwen3-0.6b-q8_0".parse::<ModelSource>()?` or `ModelSource::preset(name)`: a preset. It downloads the pinned file and checks its SHA-256.
- `source "repo:revision:file"`<br>`ModelSource::hugging_face(repo, revision, file)`: a GGUF on Hugging Face, with no checksum.
- `source file="/models/x.gguf"`<br>`ModelSource::File(path)`: a GGUF already on disk. Nothing downloads, and any part of a split set opens the whole set.
- `source "repo:revision"` or `source { checkpoint repo="Qwen/Qwen3-0.6B" revision="c1899de"; }`<br>`ModelSource::Checkpoint { repo, revision }`: a safetensors checkpoint on Hugging Face, downloaded and converted once.
- `source dir="/ckpt"`<br>`ModelSource::Dir(path)`: a checkpoint directory on disk, converted once.
- `source { hugging_face repo="org/r" revision="abc" file="m.gguf" sha256="..." { part file="..." sha256="..."; template repo="..." revision="..." file="chat_template.jinja"; } }`<br>`ModelSource::HuggingFace { repo, revision, file, sha256, parts, template }`: a GGUF on Hugging Face with a checksum, the parts of a split file, or a replacement chat template.

- `ModelSource::default_model()` is the preset `gemma4-e2b-q4_0`.
- A string never names a path. Write a file as `file=` and a directory as `dir=`.
- In `repo:revision:file`, none of the three parts contains `:`. In `repo:revision`, `repo` is `owner/name`.
- Use a commit hash as the revision. The cache is keyed by it, so a moving name such as `main` pins a stale file.
- A source with `sha256: None` skips the checksum, and so does every source parsed from a string except a preset.
- `template` is a chat template file that replaces the model's own. See [Requests, thinking and tool calls](@/providers/local/requests.md).

A string that is none of the three forms fails with a `CucaError::Config`:

```text,name=Error for an unknown source string
unknown model `NAME`: expected a preset (gemma4-e2b-q4_0, ...), `repo:revision:file` (a GGUF), or `repo:revision` (a checkpoint)
```

The message lists every preset name where this page shows `...`.

## Presets

`ModelSource::PRESETS` has 26 entries: 24 models and 2 drafters. Each is pinned to one Hugging Face commit and one SHA-256. The names, sizes and memory need are on [Choose a model](@/providers/local/models.md).

A `ModelPreset` has these fields:

- `name`, `repo`, `revision`, `file`, `sha256`, `size_bytes` and `architecture`, the file's `general.architecture`.
- `drafter`: `true` for a drafter, which loads only through `LocalModelConfig::drafter`.
- `parts`: empty unless the model is a split GGUF. A `PresetPart` is `file`, `sha256` and `size_bytes`. For a split preset, `file` and `sha256` are the first part's, and `size_bytes` is the sum of the parts.
- `template`: `None`, or a `PresetTemplate` of `repo`, `revision`, `file` and `sha256`, the pinned chat template.

`preset.source()` returns the pinned `ModelSource`, and `ModelSource::preset(name)` finds a preset by name.

```rust,name=List the presets (provider-local)
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

```text,name=Expected output (first line of 26)
gemma4-e2b-q4_0              gemma4                2841481184 bytes  ggml-org/gemma-4-E2B-it-GGUF
```

```rust,name=Pick a different preset (provider-local)
use cuca::{LocalModel, LocalModelConfig, ModelSource};

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Qwen 3 0.6B at Q8_0 instead of the default.
    let model = LocalModel::load(LocalModelConfig {
        source: "qwen3-0.6b-q8_0".parse::<ModelSource>()?,
        ..Default::default()
    })
    .await?;
    println!("{}", model.context_length());
    Ok(())
}
```

```text,name=Expected output
8192
```

The first run downloads 0.64 GB. The number is the default context length, 8,192 tokens.

## Download and cache

A Hugging Face source downloads once from `https://huggingface.co/{repo}/resolve/{revision}/{file}` into the cache directory.

- The cache root is `cache_dir` (KDL `cache_dir "D:/cuca-cache"`). With `None`, the default is the platform directory in the table below. Empty environment values count as unset. No `CUCA_*` variable exists.
- The file lands at `<cache>/huggingface/<repo with / as -->/<revision>/<file>`.
- An interrupted download resumes from `<file>.partial` with an HTTP `Range` request. If the server ignores the range, the download restarts from byte 0 once.
- The SHA-256 is computed while streaming, resumed bytes included. A mismatch deletes the partial file and fails the load.
- The finished file is renamed into place in one step, so a final file that exists is complete.
- A lock file, `<file>.lock`, makes two loaders in one process or in two processes take turns. The empty file stays in the cache.
- A download fails when no data arrives for 60 seconds. Connecting times out after 30 seconds.
- `hf_token` is sent as a bearer token. `None` or an empty string downloads anonymously, as the preset files do. A KDL file can read it from the environment with `hf_token "${HF_TOKEN:-}"`; see [Configuration](@/providers/local/configuration.md).

| System | Default cache directory |
|---|---|
| Windows | `%LOCALAPPDATA%\cuca`, else `%USERPROFILE%\AppData\Local\cuca` |
| macOS | `$HOME/Library/Caches/cuca` |
| Linux | `$XDG_CACHE_HOME/cuca`, else `$HOME/.cache/cuca` |

`on_download_progress` takes a `DownloadProgressCallback` that receives `DownloadProgress { downloaded_bytes, total_bytes }` at the start, about every mebibyte and at the end. `total_bytes` is `None` when the server reports no size. The callback runs on the engine thread during `load`, so keep it cheap.

```rust,name=Print the progress (provider-local)
use cuca::{DownloadProgressCallback, LocalModelConfig};

fn config() -> LocalModelConfig {
    LocalModelConfig {
        on_download_progress: Some(DownloadProgressCallback::new(|p| {
            eprint!("\r{} / {:?} bytes", p.downloaded_bytes, p.total_bytes);
        })),
        ..Default::default()
    }
}
```

A checkpoint conversion reports through the same callback: the tensors converted so far out of the tensor count. A DeepSeek-V4.1 conversion reports bytes written out of bytes to write.

## Split GGUF files

`llama-gguf-split` cuts a large model into `<base>-00001-of-0000N.gguf` files. The first part holds the metadata and the others hold tensors. CUCA opens the set as one file and reads each part in place, so nothing is merged or copied.

- A `file=` source naming any part opens the set. The other parts are looked for next to that file, then in each directory of `shard_dirs`. The parts of one model may sit on different volumes.
- A Hugging Face source whose file is a split name downloads every part into `<cache>/huggingface/<repo>/<revision>/<dir>/` and opens the set from the first part. The set needs its own size on disk.
- A preset pins every part's SHA-256 and size, and each part is verified as it downloads. A verified part is not downloaded again. `mistral-small-4-119b-q4_0` is the one split preset.
- A source with no `parts` verifies only the first part, and only when `sha256` is set. A source parsed from a string verifies none.
- `on_download_progress` reports the parts one after another.
- A missing part, a part whose `split.count` or `split.no` disagrees with its name, a set with the wrong tensor count and a tensor that appears in two parts all fail the load. The message names the file.

```kdl,name=A split set on two volumes
source file="D:/models/big-00001-of-00003.gguf"
shard_dirs "E:/parts" "F:/parts"
```

## Convert a safetensors checkpoint

A checkpoint is the model as its authors publish it: `config.json`, `model.safetensors` or several shards, and the tokenizer files. When no GGUF exists, `ModelSource::Checkpoint` and `ModelSource::Dir` convert the checkpoint once, in pure Rust, then load the result like any file.

```rust,name=Load a checkpoint (provider-local)
use cuca::{ConvertConfig, LocalModel, LocalModelConfig, WeightQuant};

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = LocalModel::load(LocalModelConfig {
        // `repo:revision`; a local directory is `ModelSource::Dir(path)`.
        source: "Qwen/Qwen3-0.6B:c1899de289a04d12100db370d81485cdf75e47ca".parse()?,
        convert: ConvertConfig { quant: WeightQuant::Q8_0, ..Default::default() },
        ..Default::default()
    })
    .await?;
    println!("{}", model.context_length());
    Ok(())
}
```

```text,name=Expected output
8192
```

The first load downloads the shards and converts them. Later loads open the cached file. The KDL form is `source "Qwen/Qwen3-0.6B:<revision>"` with `convert quant="q8_0"`.

### Supported checkpoints

`architectures[0]` in `config.json` selects the converter.

| `architectures[0]` | Becomes |
|---|---|
| `LlamaForCausalLM`, `MistralForCausalLM` | `llama` |
| `SmolLM3ForCausalLM` | `smollm3` |
| `GraniteForCausalLM` | `granite` |
| `Qwen2ForCausalLM`, `Qwen3ForCausalLM`, `Qwen3MoeForCausalLM` | `qwen2`, `qwen3`, `qwen3moe` |
| `Qwen3_5ForCausalLM`, `Qwen3_5ForConditionalGeneration` | `qwen35` |
| `Qwen3_5MoeForCausalLM`, `Qwen3_5MoeForConditionalGeneration` | `qwen35moe` |
| `Phi3ForCausalLM` | `phi3` |
| `Gemma3ForCausalLM`, `Gemma3ForConditionalGeneration` | `gemma3` |
| `Gemma4ForCausalLM`, `Gemma4ForConditionalGeneration` | `gemma4` (dense models) |
| `Gemma4AssistantForCausalLM` | `gemma4-assistant`, a drafter |
| `Olmo2ForCausalLM`, `Olmo3ForCausalLM` (without sliding layers) | `olmo2` |
| `OlmoeForCausalLM` | `olmoe` |
| `Ernie4_5ForCausalLM` | `ernie4_5` |
| `MuseGlimmerForConditionalGeneration` (text part only) | `muse-glimmer` |
| `MuseGlimmerAssistantModel` | `dflash`, the Muse Glimmer drafter |
| `Ministral3ForCausalLM`, or `Mistral3ForConditionalGeneration` whose `text_config.model_type` is `ministral3` or `mistral` | `mistral3` |
| `Mistral3ForConditionalGeneration` whose `text_config.model_type` is `mistral4` | `mistral4` |
| `EagleMistralLarge3ForCausalLM` (read from `params.json` and `consolidated.safetensors`) | `mistral4-eagle`, a drafter |
| `DeepseekV41ForCausalLM` (text part only) | `deepseek41` and its drafter; see [Run DeepSeek-V4.1](@/providers/local/deepseek-v41.md) |

- `mistral3` covers Ministral 3, Devstral Small 2, Mistral Small 3.1 and 3.2, and Magistral 2509. Mistral Small 4 is `mistral4`.
- Any other class fails with ``architecture `X` has no checkpoint converter; convert it to GGUF with llama.cpp`` before a shard downloads. A `Mistral3ForConditionalGeneration` with another text model fails naming it.
- Drafter pairings are on [Speculative decoding](@/providers/local/drafting.md).

### Precision

`convert quant` is `f16`, `q8_0` (the default) or `q4_0`. Each precision converts and caches separately. KDL: `convert quant="q4_0"`.

- Every 2-D weight whose rows are a multiple of 32 long is stored at the chosen precision. Norms, biases, scales, the router and the recurrent-layer parameters stay in 32-bit floats.
- At `q4_0` the token embedding and the output head stay at Q8_0.
- FP8 checkpoints are converted from their scales and stored at the chosen precision, never as FP8.
- `convert engram_quant` is `q8_0` (the default), `q4_0` or `bf16`. It sets the storage type of DeepSeek-V4.1's Engram table, and every other architecture ignores it.

### Output and cache

- The output is `<cache>/converted/<repo with / as -->/<revision>/model-<quant>.gguf`, or `<cache>/converted/dir-<key>/model-<quant>.gguf` for a directory. A non-default `engram_quant` adds `-engram-<name>` to the file name.
- The directory key covers the path and the name, size and modification time of `config.json`, `params.json` and every `*.safetensors` file, so an edited checkpoint converts again.
- An existing output is reused without touching the network or the shards. A lock file keeps two loaders from converting at once.
- A checkpoint source downloads `config.json` first, then the optional tokenizer and template files that exist, then the shards.
- The chat template comes from `chat_template.jinja`, else `chat_template.json`, else `tokenizer_config.json`. A checkpoint with none fails with `checkpoint has no chat template`.
- Sampling defaults in `generation_config.json` become the GGUF's `general.sampling.*` keys, see [Configuration](@/providers/local/configuration.md).
- Memory stays small whatever the model size. Free disk must hold the converted file.

### Not converted

Each of these fails with a `CucaError::Config` that names the cause:

- the Gemma 4 mixture of experts, and OLMo 3 sliding-window layers;
- Qwen 3 MoE with `norm_topk_prob = false`, shared experts or `mlp_only_layers`, and OLMoE with `norm_topk_prob = true`;
- an explicit YaRN `attention_factor`, and a rope scaling type the engine does not run;
- block-scaled FP8 (`quantization_config.weight_block_size`), except DeepSeek-V4.1's 32 by 32 grid;
- a pre-tokenizer the engine does not implement, see [Choose a model](@/providers/local/models.md);
- a Muse Glimmer checkpoint with rope scaling, attention biases or a `post_norm_eps` other than 1e-8.

### Checkpoints with only tekken.json

Mistral Small 3.1 and 3.2, Magistral 2506 to 2509 and Devstral Small 2505 and 2507 publish `tekken.json` and no `tokenizer.json`. For `llama`, `smollm3`, `granite` and `mistral3` the converter builds the tokenizer and the chat template from `tekken.json`; tokenizer versions v7, v11 and v13 work, and any other version fails with a `CucaError::Config` that names it. A tekken-only directory of any other architecture fails and asks for the repository's `tokenizer.json`.
