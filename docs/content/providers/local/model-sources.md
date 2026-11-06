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

- `ModelSource::File(path)` is a GGUF file already on disk; nothing is downloaded.
- `ModelSource::HuggingFace { repo, revision, file, sha256 }` is a file on Hugging Face. `sha256: None` skips the checksum.
- `ModelSource::hugging_face(repo, revision, file)` builds the same value with `sha256: None`, so the download is not verified. Use a commit hash as the revision, because the cache is keyed by it and a moving ref such as `main` pins a stale file.
- `ModelSource::default_model()` is the preset `gemma4-e2b-q4_0`.
- `ModelSource::Checkpoint { repo, revision }` is a Hugging Face safetensors checkpoint, and `ModelSource::Dir(path)` a checkpoint directory on disk. Both are converted to a cached GGUF (see [Checkpoints](#checkpoints)).
- `"name".parse::<ModelSource>()?` accepts a preset name (ASCII case-insensitive), `repo:revision:file` (a GGUF, none of the three parts contains `:`), or `repo:revision` (a checkpoint: `repo` is `owner/name`). The result of `repo:revision:file` has no checksum; a checkpoint is pinned by its revision only. Anything else fails with `` unknown model `…`: expected a preset (…), `repo:revision:file` (a GGUF), or `repo:revision` (a checkpoint) ``.

`ModelSource::PRESETS` has 25 entries, each pinned to one Hugging Face commit and one SHA-256. A `ModelPreset` has `name`, `repo`, `revision`, `file`, `sha256`, `size_bytes`, `architecture` and `drafter`. `preset.source()` returns its pinned `ModelSource`, and `ModelSource::preset(name)` finds one by name. A preset with `drafter: true` is used through `LocalModelConfig::drafter` and cannot be loaded as the main model. The table with sizes and K/V cost is on [Local models and formats](@/providers/local/models.md#presets).

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

- The files are downloaded like any other source: `config.json` first, then the optional `generation_config.json`, `tokenizer.json`, `tokenizer_config.json`, `tokenizer.model`, `added_tokens.json`, `chat_template.jinja` and `model.safetensors.index.json` (a missing one is skipped), then the shards. An architecture without a converter fails with `` architecture `X` has no checkpoint converter; convert it to GGUF with llama.cpp `` before any shard is fetched.
- The conversion writes `<cache>/converted/<repo with / as -->/<revision>/model-<quant>.gguf` (`dir-<key>/model-<quant>.gguf` for a directory, where the key is the first 16 hex digits of a SHA-256 over the canonical path and each `config.json` and `*.safetensors` file's name, length and modification time, so an edited checkpoint converts again). `<quant>` is `f16`, `q8_0` or `q4_0` (`ConvertConfig::quant`, default `q8_0`; each is cached separately). The output is written to `<file>.partial` under an exclusive `<file>.lock` and renamed into place; an existing file is reused without touching the network or the shards.
- Every 2-D weight matrix whose rows are a multiple of 32 long (attention and FFN projections, expert stacks, the embeddings, the output head) is stored at the chosen precision; norms, biases, scales, RoPE factors, the router and the recurrent-layer parameters stay `f32`. At `q4_0` the token embedding and the output head are kept at Q8_0 and Gemma 4's `per_layer_model_proj` at F16, as in the published Gemma 4 E2B Q4_0 file (llama.cpp's Q4_0 mixes keep the embeddings above 4 bits too).
- Memory is bounded by one block of one tensor (at most 32 MiB of `f32`) plus its quantized form, whatever the model's size: the header is written first and the tensors stream behind it. The crate forbids `unsafe`, so the shards are read with positional reads, not memory-mapped.
- `on_download_progress` also reports the conversion: `downloaded_bytes` is the number of tensors written so far and `total_bytes` the tensor count.
- Before the file is renamed into place, its metadata is loaded through the engine's own hyperparameter and tokenizer readers; a missing key fails the conversion naming it. A checkpoint without a chat template fails with `checkpoint has no chat template`.
- Not converted: mixture-of-experts Gemma 4, OLMo 3 sliding-window layers, Qwen 3 MoE with `norm_topk_prob = false`, OLMoE with `norm_topk_prob = true`, Granite MoE shared experts, an explicit YaRN `attention_factor`, a pre-tokenizer the engine does not implement; each fails with a `CucaError::Config` that names the cause. The Gemma 4 assistant (`google/gemma-4-E2B-it-assistant`) converts without a vocabulary, as a drafter needs none.

The conversion is a port of llama.cpp's `convert_hf_to_gguf.py` (commit `67107ac34f62882d8b53ea41ad68010ea635b9be`) and was verified against it: the script was run on the same checkpoints and the two GGUF files compared key by key and tensor by tensor. The F16 files were identical for SmolLM 2, TinyLlama, Gemma 2, Gemma 3, Gemma 4, OLMo 2, Granite 3.3, Granite 3.1 MoE, ERNIE 4.5, SmolLM 3, Qwen 2.5, Phi 3.5 and Phi 4 mini, and for synthetic Qwen 3 MoE, OLMoE, Qwen 3.5 and Qwen 3.5 MoE checkpoints; Q8_0 matched byte for byte for SmolLM 2. The remaining differences are one-ulp `f32` rounding in two generated tensors (Llama 3 `rope_freqs`, Qwen 3.5 `ssm_a`), a tied `lm_head` that is not written again, and the Qwen 3.5 prediction-block count, which is read from the tensors present instead of `config.json`.

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
- `LocalModelConfig::on_download_progress` takes a `DownloadProgressCallback` that receives `DownloadProgress { downloaded_bytes, total_bytes }` at the start, about every mebibyte and at the end. `total_bytes` is `None` when the server reports no size. The callback runs on the engine thread during `load`, so keep it cheap. A checkpoint conversion reports tensors converted out of the tensor count in the same fields.
- `hf_token`, or `HF_TOKEN` when it is `None`, is sent as a bearer token; the pinned preset files download anonymously.
- The download uses a private HTTP client and runtime, separate from the `CucaClient` one.
