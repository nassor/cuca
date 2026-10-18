+++
title = "Local (in-process)"
description = "The provider-local adapter: Gemma 4 (E2B, E4B, any dense gemma4 GGUF) on the CPU inside your process, engine code in pure Rust. Model presets, source and download, configuration, concurrency, memory and behavior."
template = "page.html"
weight = 8
+++

# Local (in-process)

<dl class="page-facts">
<dt>In one line</dt>
<dd>Runs a dense Gemma 4 GGUF (E2B or E4B, Q4_0 or Q8_0 presets; any <code>gemma4</code> file loads) on the CPU inside your process, through candle, with no server and no HTTP endpoint.</dd>
<dt>You need</dt>
<dd>The <code>provider-local</code> feature, disk for the one-time download (2.84 GB for the default E2B Q4_0, up to 8.03 GB for E4B Q8_0) or a GGUF file you already have, and RAM of roughly 2 GB (E2B Q4_0) to 6 GB (E4B Q8_0) including the K/V cache.</dd>
<dt>Read this if</dt>
<dd>You are routing requests through <code>ProviderEndpoint::Local</code>, or want a model with no external service to run.</dd>
</dl>

This is the only provider that does not speak HTTP. The engine code is pure Rust:
no llama.cpp, no FFI written by cuca, and no HTTP endpoint. `LocalModel::load` resolves the
GGUF file, loads the weights, the tokenizer and the chat template, and starts one
dedicated engine thread. `CucaClientBuilder::with_local_model` attaches the model
to a client built for `ProviderEndpoint::Local`, and `generate_stream` routes
requests to that engine. The output is the usual stream of `Text`, `Thinking`
and `ToolCall` blocks.

Add the crate with `cargo add cuca --features provider-local`, `cargo add tokio --features rt,macros`, and `cargo add tokio-stream`.

```rust,name=A first stream through the local engine
use std::io::{Write, stdout};

use cuca::types::{MessageContentBlock, ProviderEndpoint};
use cuca::{CucaClient, LocalModel, LocalModelConfig, UnifiedRequest};
use tokio_stream::StreamExt;

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Downloads the pinned Gemma 4 E2B Q4_0 file on first use, then loads it.
    let model = LocalModel::load(LocalModelConfig::default()).await?;

    let client = CucaClient::builder()
        .with_provider(ProviderEndpoint::Local)
        .with_local_model(model)
        .build()?;

    let request = UnifiedRequest::new("gemma-4-it")
        .add_system_message("You are concise.")
        .add_user_message("Explain CUCA in one sentence.")
        .set_max_tokens(256);
    let mut stream = client.generate_stream(request).await?;
    while let Some(block) = stream.next().await {
        if let MessageContentBlock::Text(text) = block? {
            print!("{text}");
            stdout().flush()?;
        }
    }
    println!();
    Ok(())
}
```

The reply depends on the model and the sampling seed. The runnable version is
`examples/local_gemma.rs`; it also prints download progress on stderr and prints
`Thinking` blocks, and reads `CUCA_LOCAL_MODEL` (a preset name or
`repo:revision:file`) or `CUCA_LOCAL_MODEL_PATH` to run another model. Run it
built for the host CPU, because candle selects its quantized SIMD kernels at
compile time and a generic build is several times slower:

```bash
RUSTFLAGS="-C target-cpu=native" cargo run --release --example local_gemma --features provider-local
CUCA_LOCAL_MODEL=gemma4-e4b-q4_0 RUSTFLAGS="-C target-cpu=native" cargo run --release --example local_gemma --features provider-local
```

## Facts

| Fact | Value |
|---|---|
| Feature flag | `provider-local` |
| `ProviderEndpoint` variant | `Local` |
| Attached with | `CucaClientBuilder::with_local_model`; `build` fails without it |
| Network use | only the one-time model download from Hugging Face |
| Dependencies it owns | `candle-core` and `candle-nn` (0.11), `half`, `minijinja`, `minijinja-contrib`; it reuses `reqwest`, `sha2` and `tokio`, and does not use `tokio-stream` |

candle-core 0.11 hard-depends on `tokenizers[onig]`, so the dependency tree
compiles C (`onig_sys`); the feature needs a C compiler.
cuca never calls that crate; the engine code is pure Rust.

## Choosing a model

Any dense `gemma4` GGUF works. The loader reads every dimension from the file's
metadata and tensor shapes (layer count, residual width, query and KV head
counts, head dimensions, per-layer FFN width, vocabulary, per-layer-embedding
width), so no checkpoint size is built in. It accepts the tensor dtypes candle
can read for matrix weights (F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0 and
the K-quants Q2_K to Q6_K), and the token and per-layer embedding tables in any
of those. It rejects, with a `CucaError::Provider` that names the cause, another
architecture, a Mixture-of-Experts checkpoint (`gemma4.expert_count` above 0),
per-layer KV head counts that differ, key and value lengths that differ, and
IQ-family quantizations (candle has no kernels for them).

`LocalModelConfig::source` is a `ModelSource`. The presets are pinned to one
Hugging Face commit and one SHA-256 each, and downloaded from the `ggml-org`
repositories:

| Preset (`str::parse` name) | Constructor | File | Size | RAM at 8192 context |
|---|---|---|---|---|
| `gemma4-e2b-q4_0` (default) | `ModelSource::gemma4_e2b_it_q4_0()` | `ggml-org/gemma-4-E2B-it-GGUF` `gemma-4-E2B-it-Q4_0.gguf` | 2.84 GB | about 2 GB |
| `gemma4-e2b-q8_0` | `ModelSource::gemma4_e2b_it(Gemma4Quant::Q8_0)` | `ggml-org/gemma-4-E2B-it-GGUF` `gemma-4-E2B-it-Q8_0.gguf` | 4.97 GB | about 3 GB |
| `gemma4-e4b-q4_0` | `ModelSource::gemma4_e4b_it(Gemma4Quant::Q4_0)` | `ggml-org/gemma-4-E4B-it-GGUF` `gemma-4-E4B-it-Q4_0.gguf` | 4.59 GB | about 4 GB |
| `gemma4-e4b-q8_0` | `ModelSource::gemma4_e4b_it(Gemma4Quant::Q8_0)` | `ggml-org/gemma-4-E4B-it-GGUF` `gemma-4-E4B-it-Q8_0.gguf` | 8.03 GB | about 6 GB |

The RAM column is an estimate: the file size minus the per-layer embedding table
(read from disk per token, see *Concurrency and memory*) plus the K/V cache at
the default 8192 context, rounded up; a longer `context_length` adds its K/V
cache at the rate in that section. The E2B Q4_0 and E4B Q4_0 presets are run
end to end by the live test suite; E2B Q8_0 and E4B Q8_0 were checked against
their GGUF headers (same architecture and tensor layout, different dtypes) and
the row decoder is unit-tested for every dtype. The BF16 files of both
repositories (9.3 GB and 15 GB) also parse, but candle dequantizes BF16 weights
to f32 at load, so they are not realistic for a CPU process.

Select a preset with `ModelSource::gemma4_it(Gemma4Variant::E4b, Gemma4Quant::Q4_0)`,
the shortcuts above, or by name: `ModelSource` implements `FromStr`, so
`"gemma4-e4b-q4_0".parse::<ModelSource>()?` works (ASCII case-insensitive;
`ModelSource::PRESET_NAMES` lists the names). For any other file:

- `ModelSource::hugging_face(repo, revision, file)` for a file on Hugging Face,
  for example an `unsloth/gemma-4-E2B-it-GGUF` quantization. It has no SHA-256,
  so the download is not verified; the revision should be a commit hash, since
  the cache is keyed by it. Parsing `repo:revision:file` builds the same value.
- `ModelSource::HuggingFace { repo, revision, file, sha256 }` to pin a digest
  yourself; `sha256: None` skips the check.
- `ModelSource::File(path)` for a GGUF file already on disk; nothing is
  downloaded.

```rust,name=Selecting a different model
use cuca::{Gemma4Quant, LocalModel, LocalModelConfig, ModelSource};

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Gemma 4 E4B at Q4_0 instead of the E2B default; the rest of the program
    // is the same as above.
    let model = LocalModel::load(LocalModelConfig {
        source: ModelSource::gemma4_e4b_it(Gemma4Quant::Q4_0),
        ..Default::default()
    })
    .await?;
    println!("{model:?}");
    Ok(())
}
```

The engine reads the `gemma4` GGUF layout and a `gemma4` tokenizer; a file with a
different tokenizer model fails to load with a provider error. The chat template
comes from the file itself.

## Download and cache

A Hugging Face source is downloaded once into the cache directory, which
`LocalModelConfig::cache_dir` sets. When it is `None`, the library looks at the
`CUCA_CACHE_DIR` environment variable, then the platform cache directory. The
file lands at `<cache>/huggingface/<repo with / as -->/<revision>/<file>`.

- An interrupted download resumes from the partial file with an HTTP `Range`
  request, and restarts from byte 0 if the server ignores the range.
- The SHA-256 is computed while streaming, covering any resumed bytes. A mismatch
  deletes the partial file and fails the load.
- The verified file is renamed into place atomically, so a final file that exists
  is complete.
- A lock file, `<file>.lock`, serialises concurrent loaders. The empty file is
  intentionally left in the cache: deleting it after unlock would race with a
  process still blocked on it.
- `LocalModelConfig::on_download_progress` takes a `DownloadProgressCallback`
  that receives `DownloadProgress { downloaded_bytes, total_bytes }` at the
  start, about every megabyte and at the end. It runs on the engine thread during
  `load`, so keep it cheap.
- `LocalModelConfig::hf_token`, or the `HF_TOKEN` environment variable when it is
  `None`, is sent as a bearer token; anonymous access works for the pinned file.

## Configuration

| Field | Default | Meaning |
|---|---|---|
| `source` | `ModelSource::gemma4_e2b_it_q4_0()` | where the GGUF file comes from |
| `cache_dir` | `None` | download cache; `None` resolves `CUCA_CACHE_DIR`, then the platform directory |
| `hf_token` | `None` | Hugging Face token; `None` falls back to `HF_TOKEN` |
| `context_length` | `8192` | maximum prompt plus completion tokens per request, and the K/V cache size |
| `sampling` | `SamplingConfig::default()` | default sampling parameters |
| `on_download_progress` | `None` | download progress observer |

`SamplingConfig` defaults to `temperature` 1.0, `top_k` 64, `top_p` 0.95 and no
fixed seed. With `seed: Some(n)`, every request restarts the generator from the
seed, so equal prompts sample equal tokens. `temperature` of `0.0` is greedy.

`LocalModel::load` returns `CucaError::Config` for an invalid configuration (a
zero or oversized `context_length`, a non-finite `temperature`, or a `top_p`
outside `(0, 1]`), `CucaError::Io` for a missing or unreadable file,
`CucaError::Transport` or `CucaError::Http` for download failures, and
`CucaError::Provider` for an unsupported or malformed GGUF file (another
architecture, MoE, per-layer KV head counts, a tensor dtype candle cannot read,
a missing tensor or metadata key) or a failed checksum.

## Concurrency and memory

- **Single slot.** The engine runs requests one at a time in FIFO order, with at
  most 16 queued behind the running one. Past that, `generate_stream` waits for a
  free queue slot: backpressure, not an error. A running request streams through
  a bounded 64-block channel, so a slow consumer paces generation, and dropping
  the stream stops it.
- **K/V cache.** It is preallocated for `context_length` positions and costs
  `context_length * 2 * 4 * sum(n_kv_head * head_dim)` bytes, summed over the
  layers that own K/V (the last `shared_kv_layers` layers read an earlier
  layer's cache), with each layer's `head_dim` taken from its attention type.
  For E2B that is 12 sliding layers of head dimension 256 plus 3 full layers of
  512, one KV head: `context_length * 36,864` bytes, 302 MB at the default 8192.
  E4B has 20 sliding and 4 full owners with two KV heads:
  `context_length * 114,688` bytes, 939 MB at 8192. It is reused
  across requests: a prompt that shares a prefix with the previous one only
  prefills the new tokens, reported as the response's `prompt_cache_usage`.
- **Weights.** They stay quantized. The per-layer embedding table
  (`per_layer_token_embd`: 1.32 GB in E2B Q4_0, 1.59 GB in E4B Q4_0) is not
  loaded: one row per input token (5,040 bytes in E2B Q4_0) is read from the GGUF
  file, usually from the OS page cache. Resident weights are therefore about the
  file size minus that table, 1.5 GB of the 2.84 GB E2B Q4_0 file.
- **Cloning.** A `LocalModel` is cheap to clone, and every clone shares the one
  engine thread. The thread ends when the last clone is dropped.

## Behavior

- Thinking is binary: a request with thinking enabled opens the model's
  reasoning channel, and its effort level is ignored. Reasoning arrives as
  `Thinking` blocks.
- The engine is text-only: a message with an `ImageBase64` block is rejected with
  a provider error.
- `UnifiedRequest::model` is not used for routing, since one client holds one
  model. `temperature` and `max_tokens` apply per request; `top_k`, `top_p` and
  `seed` come from `SamplingConfig`. Generation also stops at the context limit.
- Tool definitions reach the model through its own chat template, and tool calls
  come back as `ToolCall` blocks.
- Token usage in the response is exact, not estimated.

## Testing it

`tests/provider_local.rs` runs the real model and needs no server. Point
`CUCA_LOCAL_MODEL_PATH` at a GGUF file, set `CUCA_LOCAL_MODEL` to a preset name
or `repo:revision:file` to download that model into the cache, or set
`CUCA_LOCAL_DOWNLOAD=1` to download the default (E2B Q4_0). `CUCA_LOCAL_MODEL_PATH`
wins when several are set. With none set, each test prints
`SKIP: local model not available: ...` and passes, unless `CUCA_REQUIRE_LIVE=1`
turns the skip into a panic.

```bash
CUCA_LOCAL_DOWNLOAD=1 RUSTFLAGS="-C target-cpu=native" cargo test --release --features provider-local --test provider_local
CUCA_LOCAL_MODEL=gemma4-e4b-q4_0 RUSTFLAGS="-C target-cpu=native" cargo test --release --features provider-local --test provider_local
```
