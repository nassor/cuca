+++
title = "Local (in-process)"
description = "The provider-local adapter: Gemma 4 (E2B, E4B, any dense gemma4 GGUF) on the CPU or a GPU (Vulkan, Metal, CUDA, ROCm) inside your process, engine code in pure Rust. Model presets, source and download, configuration, running on a GPU, concurrency, memory and behavior."
template = "page.html"
weight = 8
+++

# Local (in-process)

<dl class="page-facts">
<dt>In one line</dt>
<dd>Runs a dense Gemma 4 GGUF (E2B or E4B, Q4_0 or Q8_0 presets; any <code>gemma4</code> file loads) on the CPU or a GPU inside your process, with no server and no HTTP endpoint. The CPU path uses candle; the GPU path uses burn and CubeCL.</dd>
<dt>You need</dt>
<dd>The <code>provider-local</code> feature (CPU), or one of <code>provider-local-vulkan</code>, <code>provider-local-metal</code>, <code>provider-local-cuda</code> and <code>provider-local-rocm</code> to run on a GPU; disk for the one-time download (2.84 GB for the default E2B Q4_0, up to 8.03 GB for E4B Q8_0) or a GGUF file you already have; and memory of roughly 2 GB (E2B Q4_0) to 6 GB (E4B Q8_0) including the K/V cache, in RAM on the CPU or in device memory on a GPU.</dd>
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
| Feature flag | `provider-local`; GPU: `provider-local-vulkan`, `provider-local-metal`, `provider-local-cuda`, `provider-local-rocm` (each enables `provider-local-gpu`, the shared GPU engine) |
| Device | `LocalModelConfig::device`; `LocalDevice::Cpu` by default, no automatic selection |
| `ProviderEndpoint` variant | `Local` |
| Attached with | `CucaClientBuilder::with_local_model`; `build` fails without it |
| Network use | only the one-time model download from Hugging Face |
| Dependencies it owns | `candle-core` and `candle-nn` (0.11), `half`, `minijinja`, `minijinja-contrib`; with a GPU feature also `burn` 0.22 (`std`, `fusion`, `autotune`); it reuses `reqwest`, `sha2` and `tokio`, and does not use `tokio-stream` |

candle-core 0.11 hard-depends on `tokenizers[onig]`, so the dependency tree
compiles C (`onig_sys`); the feature needs a C compiler.
cuca never calls that crate; the engine code is pure Rust.

Every GPU feature compiles C and needs a C compiler on top of the one `onig_sys`
already requires: CubeCL's runtime pulls an embedded store (`turso`) that builds C,
and on Windows a dependency's build script runs the SDK resource compiler (`rc.exe`;
`windres` on the GNU target). `provider-local-cuda` and `provider-local-rocm`
additionally link a prebuilt LLVM; see *Running on a GPU*.

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
IQ-family quantizations (candle has no kernels for them). These are the CPU rules;
a GPU device accepts fewer matrix-weight dtypes (see *Running on a GPU*).

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
| `device` | `LocalDevice::Cpu` | where the model runs; a GPU needs its cargo feature, see *Running on a GPU* |
| `on_download_progress` | `None` | download progress observer |

`SamplingConfig` defaults to `temperature` 1.0, `top_k` 64, `top_p` 0.95 and no
fixed seed. With `seed: Some(n)`, every request restarts the generator from the
seed, so equal prompts sample equal tokens. `temperature` of `0.0` is greedy.

`LocalModel::load` returns `CucaError::Config` for an invalid configuration (a
zero or oversized `context_length`, a non-finite `temperature`, a `top_p`
outside `(0, 1]`, or a `device` whose cargo feature is not compiled in),
`CucaError::Io` for a missing or unreadable file,
`CucaError::Transport` or `CucaError::Http` for download failures, and
`CucaError::Provider` for an unsupported or malformed GGUF file (another
architecture, MoE, per-layer KV head counts, a tensor dtype candle cannot read,
a missing tensor or metadata key), a failed checksum, or a GPU that cannot be
opened or a weight dtype the GPU engine does not take.

## Running on a GPU

The model runs on the CPU unless `LocalModelConfig::device` says otherwise. A GPU
device runs the forward pass through burn 0.22 and CubeCL, with the same math as
the CPU engine; the CPU path (candle 0.11, same API, same throughput) is
unchanged. The CPU stays the default: nothing selects a GPU for you, so callers
opt in. The engine thread owns the device.

| Feature | Backend | Build |
|---|---|---|
| `provider-local-gpu` | the shared GPU engine; selects no backend by itself | enabled by each feature below |
| `provider-local-vulkan` | Vulkan | wgpu with SPIR-V shaders (Rust); C compiler needed, plus `rc.exe` on Windows |
| `provider-local-metal` | Metal | wgpu with MSL shaders (Rust); C compiler needed, plus `rc.exe` on Windows; runs on Apple only, compiles everywhere |
| `provider-local-cuda` | CUDA | links a prebuilt LLVM, which a build script downloads on the first build (network needed once, then cached) |
| `provider-local-rocm` | ROCm | the same LLVM download; ROCm (HIP bindings) is Linux-only |

Each backend feature enables `provider-local` and `provider-local-gpu`, so
`cargo add cuca --features provider-local-vulkan` is enough. The GPU drivers
themselves are system libraries loaded at runtime (the Vulkan loader, for
example); the engine code stays pure Rust, with no HTTP endpoint and no llama.cpp.

```rust,name=Running on a GPU
use cuca::{GpuAdapter, LocalDevice, LocalModel, LocalModelConfig};

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Same model and the same API as on the CPU; only the device changes.
    let model = LocalModel::load(LocalModelConfig {
        device: LocalDevice::Vulkan(GpuAdapter::Default),
        ..Default::default()
    })
    .await?;
    println!("{model:?}");
    Ok(())
}
```

`LocalDevice` is `Cpu` (the default), `Vulkan(GpuAdapter)`, `Metal(GpuAdapter)`,
`Cuda(usize)` or `Rocm(usize)`; `GpuAdapter` is `Default`, `Discrete(usize)` or
`Integrated(usize)`. It also parses from text (ASCII case-insensitive), so
`"vulkan:1".parse::<LocalDevice>()?` works:

| Text | Device |
|---|---|
| `cpu` | `Cpu` |
| `vulkan`, `metal` | `Vulkan(GpuAdapter::Default)`, `Metal(GpuAdapter::Default)` |
| `vulkan:<n>`, `metal:<n>` | the n-th discrete GPU: `Discrete(n)` |
| `vulkan:integrated:<n>`, `metal:integrated:<n>` | the n-th integrated GPU: `Integrated(n)` |
| `cuda`, `cuda:<n>` | `Cuda(n)`; bare `cuda` is device 0 |
| `rocm`, `rocm:<n>` | `Rocm(n)`; bare `rocm` is device 0 |

Any other text is a `CucaError::Config` ("unknown local device ...").
`LocalDevice::required_feature` returns the cargo feature a device needs
(`None` for `Cpu`).

- **Memory.** Device memory is about the file size minus the per-layer embedding
  table. That table is never uploaded: its rows are still read from the GGUF file
  per token on the host, as on the CPU. The token embedding is uploaded once as the
  tied LM head, while embedding lookups read rows from the file on the host.
  `per_layer_model_proj` is BF16 in the files and is held as f32 (92 MB for E2B).
  The K/V cache lives in device memory with the same formula as on the CPU (see
  *Concurrency and memory*; 302 MB at 8192 for E2B), and the RoPE tables are on
  the device. E2B Q4_0 is about 1.5 GB of weights plus the K/V cache.
- **Weight dtypes.** Matmul weights must be Q4_0, Q8_0, F32, F16 or BF16. Q4_0
  and Q8_0 are repacked at load into burn's block-quantized layout and stay
  quantized in device memory (dequantized inside the matmul kernel); float
  weights are uploaded as f32. Any other dtype (K-quants, Q4_1, Q5_x) fails the
  load with an error naming the tensor; the CPU reads those. All four published
  presets qualify.
- **Errors.** Every `LocalDevice` variant exists in every build. Selecting a GPU
  whose feature is off fails `LocalModel::load` with a `CucaError::Config`, for
  example ``device Vulkan(Default) requires the `provider-local-vulkan` cargo
  feature``, before any thread starts. A GPU that cannot be opened (no adapter, a
  missing driver) fails with a `CucaError::Provider`, either
  `device ...: <wgpu error>` or, for a backend that panics during setup,
  `local engine failed to initialize: <message>`: the engine thread catches
  panics during load.
- **Numerics.** GPU kernels keep activations in f32, while candle's CPU quantized
  matmul rounds activations to Q8_0, so GPU and CPU logits differ slightly. On the
  live test suite the greedy outputs are identical.
- **Speed.** One machine: Windows, an AMD RX 9070 XT over Vulkan, E2B Q4_0, a
  release build, steady state after the autotune warm-up, measured about 80 to 99
  tokens/s of prefill and about 26 to 28 tokens/s of decode. The CPU path on the
  same machine (`target-cpu=native`) measured about 70 to 85 and 27 to 29: on a
  fast desktop CPU the GPU path is at parity, not ahead, and it frees the CPU. The
  first forward pass of each new shape is slow: CubeCL autotune and shader
  compilation take seconds.
- **What was run.** Vulkan was exercised on that machine, where all five live
  scenarios of `tests/provider_local.rs` pass with the same outputs as on the CPU.
  Metal, CUDA and ROCm are compile-checked only.

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
- **GPU.** On a GPU device the weights and the K/V cache are in device memory
  instead of RAM, at the sizes described in *Running on a GPU*; the engine and
  its queue behave the same.
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

Set `CUCA_LOCAL_DEVICE` (the `LocalDevice` text form above) to run the same
scenarios on a GPU. The variable is read only by this test file and by
`examples/local_gemma.rs`, not by the library, and a GPU device needs its feature:

```bash
CUCA_LOCAL_DEVICE=vulkan CUCA_LOCAL_DOWNLOAD=1 cargo test --release --features provider-local-vulkan --test provider_local
CUCA_LOCAL_DEVICE=vulkan cargo run --release --example local_gemma --features provider-local-vulkan
```

The example prints `loaded on <device>`. The unit test `gpu_tiny_model_matches_cpu`
compares the GPU engine to a dequantized-weight CPU reference on a tiny random
model. It runs only when `CUCA_LOCAL_DEVICE` names a GPU, and otherwise prints
`SKIP: set CUCA_LOCAL_DEVICE to a GPU device`:

```bash
CUCA_LOCAL_DEVICE=vulkan cargo test --release --no-default-features --features provider-local-vulkan --lib gpu_tiny_model_matches_cpu
```
