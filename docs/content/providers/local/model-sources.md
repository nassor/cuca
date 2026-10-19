+++
title = "Local model sources and presets"
description = "ModelSource variants, the pinned preset catalog entry points, and how a Hugging Face file is downloaded, verified and cached."
template = "page.html"
weight = 4
+++

# Local model sources and presets

<dl class="page-facts">
<dt>In one line</dt>
<dd>A model comes from a file on disk or a Hugging Face file, pinned by revision and SHA-256 for the presets; the download resumes, verifies and caches.</dd>
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
- `"name".parse::<ModelSource>()?` accepts a preset name (ASCII case-insensitive) or `repo:revision:file`, where none of the three parts contains `:`. The result of `repo:revision:file` has no checksum.

`ModelSource::PRESETS` has 23 entries, each pinned to one Hugging Face commit and one SHA-256. A `ModelPreset` has `name`, `repo`, `revision`, `file`, `sha256`, `size_bytes`, `architecture` and `drafter`. `preset.source()` returns its pinned `ModelSource`, and `ModelSource::preset(name)` finds one by name. A preset with `drafter: true` is used through `LocalModelConfig::drafter` and cannot be loaded as the main model. The table with sizes and K/V cost is on [Local models and formats](@/providers/local/models.md#presets).

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
- `LocalModelConfig::on_download_progress` takes a `DownloadProgressCallback` that receives `DownloadProgress { downloaded_bytes, total_bytes }` at the start, about every mebibyte and at the end. `total_bytes` is `None` when the server reports no size. The callback runs on the engine thread during `load`, so keep it cheap.
- `hf_token`, or `HF_TOKEN` when it is `None`, is sent as a bearer token; the pinned preset files download anonymously.
- The download uses a private HTTP client and runtime, separate from the `CucaClient` one.
