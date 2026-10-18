+++
title = "Local (in-process)"
description = "The provider-local adapter: Gemma 4 E2B on the CPU inside your process, in pure Rust. Model source and download, configuration, concurrency, memory and behavior."
template = "page.html"
weight = 8
+++

# Local (in-process)

<dl class="page-facts">
<dt>In one line</dt>
<dd>Runs Gemma 4 E2B (Q4_0 GGUF) on the CPU inside your process, through candle, with no server and no HTTP endpoint.</dd>
<dt>You need</dt>
<dd>The <code>provider-local</code> feature, about 2.8 GB of disk for the one-time download (or a GGUF file you already have), and roughly 1.5 GB of RAM plus the K/V cache.</dd>
<dt>Read this if</dt>
<dd>You are routing requests through <code>ProviderEndpoint::Local</code>, or want a model with no external service to run.</dd>
</dl>

This is the only provider that does not speak HTTP. The engine is 100% Rust:
no llama.cpp, no FFI, and no HTTP endpoint. `LocalModel::load` resolves the
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

    let request = UnifiedRequest::new("gemma-4-e2b-it")
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
`Thinking` blocks. Run it built for the host CPU, because candle selects its
Q4_0 and Q8_0 SIMD kernels at compile time and a generic build is several times
slower:

```bash
RUSTFLAGS="-C target-cpu=native" cargo run --release --example local_gemma --features provider-local
```

## Facts

| Fact | Value |
|---|---|
| Feature flag | `provider-local` |
| `ProviderEndpoint` variant | `Local` |
| Attached with | `CucaClientBuilder::with_local_model`; `build` fails without it |
| Network use | only the one-time model download from Hugging Face |
| Dependencies it owns | `candle-core` and `candle-nn` (0.9.x), `half`, `minijinja`, `minijinja-contrib`; it reuses `reqwest`, `sha2` and `tokio`, and does not use `tokio-stream` |

candle is pinned to 0.9.x because 0.10 and 0.11 hard-depend on
`tokenizers[onig]`, which compiles C.

## Choosing the model file

`LocalModelConfig::source` is a `ModelSource`:

- `ModelSource::gemma4_e2b_it_q4_0()`, the default: the `gemma-4-E2B-it-Q4_0.gguf`
  file from `ggml-org/gemma-4-E2B-it-GGUF`, pinned to one commit and one SHA-256
  (2.84 GB).
- `ModelSource::HuggingFace { repo, revision, file, sha256 }` for another file;
  `sha256: None` skips the check.
- `ModelSource::File(path)` for a GGUF file already on disk; nothing is
  downloaded.

The engine reads the `gemma4` GGUF layout and a `gemma4` tokenizer; a file with a
different tokenizer model fails to load with a provider error.

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
`CucaError::Provider` for an unsupported or malformed GGUF file or a failed
checksum.

## Concurrency and memory

- **Single slot.** The engine runs requests one at a time in FIFO order, with at
  most 16 queued behind the running one. Past that, `generate_stream` waits for a
  free queue slot: backpressure, not an error. A running request streams through
  a bounded 64-block channel, so a slow consumer paces generation, and dropping
  the stream stops it.
- **K/V cache.** It is preallocated for `context_length` positions and costs
  `context_length * 36,864` bytes, 302 MB at the default 8192. It is reused
  across requests: a prompt that shares a prefix with the previous one only
  prefills the new tokens, reported as the response's `prompt_cache_usage`.
- **Weights.** They stay quantized, about 1.5 GB resident. The 1.32 GB per-layer
  embedding table is not loaded: one 5,040-byte row per input token is read from
  the GGUF file, usually from the OS page cache.
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
`CUCA_LOCAL_MODEL_PATH` at a GGUF file, or set `CUCA_LOCAL_DOWNLOAD=1` to download
the pinned file into the cache. With neither set, each test prints
`SKIP: local model not available: ...` and passes, unless `CUCA_REQUIRE_LIVE=1`
turns the skip into a panic.

```bash
CUCA_LOCAL_DOWNLOAD=1 RUSTFLAGS="-C target-cpu=native" cargo test --release --features provider-local --test provider_local
```
