+++
title = "Local model sources and presets"
description = "ModelSource variants, the pinned preset entry points, how a Hugging Face file is downloaded, verified and cached, and how a split GGUF is merged."
template = "page.html"
weight = 4
+++

# Local model sources and presets

<dl class="page-facts">
<dt>In one line</dt>
<dd>A model comes from a GGUF file on disk, a Hugging Face GGUF file (pinned by revision and SHA-256 for the presets), or a safetensors checkpoint converted to a GGUF once; downloads resume, verify and cache.</dd>
<dt>You need</dt>
<dd>The <code>provider-local</code> feature. Network access once, for a Hugging Face source.</dd>
<dt>Read this if</dt>
<dd>You are choosing where the model file comes from, or debugging a download, a merge or the cache.</dd>
</dl>

## Model sources

`LocalModelConfig::source` is a `ModelSource`:

- `ModelSource::File(path)` is a GGUF file already on disk; nothing is downloaded. A file named like a split part (`<base>-00001-of-0000N.gguf`) is merged with its siblings, see [Split GGUF files](#split-gguf-files).
- `ModelSource::HuggingFace { repo, revision, file, sha256, parts, template }` is a file on Hugging Face. `sha256: None` skips the checksum. `parts` is empty for a single file and lists every part with its digest for a split GGUF. `template` is `None` or a boxed `ModelTemplate { repo, revision, file, sha256 }`, a chat template file that replaces the GGUF's; see [Chat templates](@/providers/local/chat-templates.md#where-the-template-comes-from).
- `ModelSource::hugging_face(repo, revision, file)` builds the same value with `sha256: None`, `parts` empty and no `template`, so the download is not verified. Use a commit hash as the revision: the cache is keyed by it, and a moving ref such as `main` pins a stale file.
- `ModelSource::Checkpoint { repo, revision }` is a Hugging Face safetensors checkpoint and `ModelSource::Dir(path)` a checkpoint directory on disk. Both are converted to a cached GGUF; see [Checkpoints](@/providers/local/checkpoints.md).
- `ModelSource::default_model()` is the preset `gemma4-e2b-q4_0`.

`"text".parse::<ModelSource>()` accepts three forms:

- a preset name, ASCII case-insensitive;
- `repo:revision:file`, a GGUF with no checksum, where none of the three parts contains `:`;
- `repo:revision`, a checkpoint, where `repo` is `owner/name`; it is pinned by its revision only.

Anything else fails with `` unknown model `…`: expected a preset (…), `repo:revision:file` (a GGUF), or `repo:revision` (a checkpoint) `` and the list of preset names.

A file of an architecture the engine does not know, or a tokenizer it does not implement, fails to load with a `CucaError::Provider` naming the cause; see [Local models and formats](@/providers/local/models.md#loading-a-gguf-that-is-not-a-preset).

## Presets

`ModelSource::PRESETS` has 25 entries: 23 models and 2 drafters. Each is pinned to one Hugging Face commit and one SHA-256. The names, sizes and K/V cost are on [Local models and formats](@/providers/local/models.md#presets).

A `ModelPreset` has these fields:

- `name`, `repo`, `revision`, `file`, `sha256`, `size_bytes` and `architecture`, the file's `general.architecture`.
- `drafter`: `true` for a drafter, which is used through `LocalModelConfig::drafter` and cannot be loaded as the main model.
- `parts`: empty unless the model is a split GGUF; a `PresetPart` is `file`, `sha256` and `size_bytes`. For a split preset `file` and `sha256` are those of the first part and `size_bytes` is the sum of the parts.
- `template`: `None`, or a `PresetTemplate` of `repo`, `revision`, `file` and `sha256`, the pinned chat template.

`preset.source()` returns the pinned `ModelSource`, and `ModelSource::preset(name)` finds a preset by name.

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
    // Qwen3 0.6B at Q8_0 instead of the default.
    let model = LocalModel::load(LocalModelConfig {
        source: "qwen3-0.6b-q8_0".parse::<ModelSource>()?,
        ..Default::default()
    })
    .await?;
    println!("{model:?}");
    Ok(())
}
```

## Download and cache

A Hugging Face source is downloaded once from `https://huggingface.co/{repo}/resolve/{revision}/{file}` into the cache directory.

- The cache root is `LocalModelConfig::cache_dir`, else `CUCA_CACHE_DIR`, else the platform directory. Empty environment values count as unset.
- The platform directory is `%LOCALAPPDATA%\cuca` on Windows (else `%USERPROFILE%\AppData\Local\cuca`), `$HOME/Library/Caches/cuca` on macOS, and `$XDG_CACHE_HOME/cuca` on Linux (else `$HOME/.cache/cuca`).
- The file lands at `<cache>/huggingface/<repo with / as -->/<revision>/<file>`.
- An interrupted download resumes from the `<file>.partial` file with an HTTP `Range` request, and restarts from byte 0 once per call if the server ignores or rejects the range.
- The SHA-256 is computed while streaming, covering any resumed bytes. A mismatch deletes the partial file and fails the load.
- The verified file is renamed into place atomically, so a final file that exists is complete.
- An exclusive lock file, `<file>.lock`, serializes concurrent loaders in this process or another. The empty file stays in the cache: deleting it after unlock would race with a process still blocked on it.
- Memory use during the download is bounded by a 1 MiB write buffer and one network chunk. A download fails when no chunk arrives for 60 seconds.
- `LocalModelConfig::on_download_progress` takes a `DownloadProgressCallback` that receives `DownloadProgress { downloaded_bytes, total_bytes }` at the start, about every mebibyte and at the end. `total_bytes` is `None` when the server reports no size. The callback runs on the engine thread during `load`, so keep it cheap. A checkpoint conversion reports tensors converted out of the tensor count in the same fields, and a split merge reports bytes copied out of the bytes to copy.
- `hf_token`, or `HF_TOKEN` when it is `None`, is sent as a bearer token. The pinned preset files download anonymously.
- The download uses a private HTTP client and runtime with a 30 second connect timeout, separate from the `CucaClient` one.

## Split GGUF files

`llama-gguf-split` cuts a large model into `<base>-00001-of-0000N.gguf` files. Each part is a GGUF with `split.no` (from 0), `split.count` and `split.tensors.count`; the first part holds the metadata, the others hold tensors. The engine reads one file, so a set is merged once, like `llama-gguf-split --merge`:

- The merged file has the metadata of part 1 without the `split.*` keys, every tensor of every part with recomputed offsets, and the tensor data copied part by part, never more than 8 MiB in memory. Tensor types the engine cannot load (the IQ family, MXFP4) are copied as they are; the loader rejects them as for any file.
- A Hugging Face source whose `file` is a split name (a preset, or `repo:revision:dir/base-00001-of-0000N.gguf`) downloads every part, merges them into `<cache>/huggingface/<repo>/<revision>/<dir>/<base>.gguf` and deletes the parts, so the merged file is the cache entry. A preset pins every part's SHA-256 and size, and each part is verified as it downloads. A source with no `parts` verifies only the first part, and only when `sha256` is set; one parsed from a string verifies none.
- `ModelSource::File(path)` naming any part merges the parts in the same directory into `<cache>/merged/<key>/<base>.gguf`. The key covers the directory and the name, length and modification time of every part, so a changed part merges again. The parts stay where they are.
- The merge writes `<merged>.partial` and renames it under `<merged>.lock`, so concurrent loaders merge once. A Hugging Face source holds the lock from the first download to the end of the merge.
- While merging, the parts and the merged file coexist, so a download needs about twice the model's size free until the parts are deleted. Afterwards the cache holds the merged file only, and deleting it means a new download of every part. Parts that failed verification or merging stay for a retry, and a verified part is not downloaded again.
- `on_download_progress` reports each part's download, then the merge as bytes copied out of the bytes to copy.
- A missing part (`CucaError::Io` naming it), a part whose `split.count`, `split.no` or `split.tensors.count` disagree with the set, a file that is not a split part, or a tensor that appears in two parts (`CucaError::Config` naming the file) fail the load. Nothing is left at the merged path.
