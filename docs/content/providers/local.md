+++
title = "Local (in-process)"
description = "The provider-local adapter: any supported GGUF language model (Gemma 2, 3 and 4, Qwen 2.5, 3 and 3.5, Llama 3.2, TinyLlama, SmolLM 2 and 3, Phi 3.5 and 4, OLMo 2, DeepSeek-R1 distills, Granite 3 and 3.3) on the CPU or a GPU (Vulkan, Metal, CUDA, ROCm) inside your process, engine code in pure Rust. Presets, source and download, configuration, MTP drafting, running on a GPU, concurrency, memory and behavior."
template = "page.html"
weight = 8
+++

# Local (in-process)

<dl class="page-facts">
<dt>In one line</dt>
<dd>Runs a GGUF language model of a supported architecture (see <a href="@/providers/local-models.md">Local models and formats</a>) on the CPU or a GPU inside your process, with no server and no HTTP endpoint. The CPU path uses candle; the GPU path uses burn and CubeCL.</dd>
<dt>You need</dt>
<dd>The <code>provider-local</code> feature (CPU), or one of <code>provider-local-vulkan</code>, <code>provider-local-metal</code>, <code>provider-local-cuda</code> and <code>provider-local-rocm</code> to run on a GPU; disk for the one-time download (0.29 GB for the smallest model preset up to 8.03 GB for the largest, 2.84 GB for the default) or a GGUF file you already have; and memory of about the file size plus the K/V cache, in RAM on the CPU or in device memory on a GPU.</dd>
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

Which models run, how their tokenizers and chat templates are handled, and what
each preset was verified to do is on [Local models and formats](@/providers/local-models.md).

Add the crate with `cargo add cuca --features provider-local`, `cargo add tokio --features rt,macros`, and `cargo add tokio-stream`.

```rust,name=A first stream through the local engine
use std::io::{Write, stdout};

use cuca::types::{MessageContentBlock, ProviderEndpoint};
use cuca::{CucaClient, LocalModel, LocalModelConfig, UnifiedRequest};
use tokio_stream::StreamExt;

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Downloads the pinned default model (Gemma 4 E2B Q4_0) on first use, then
    // loads it.
    let model = LocalModel::load(LocalModelConfig::default()).await?;

    let client = CucaClient::builder()
        .with_provider(ProviderEndpoint::Local)
        .with_local_model(model)
        .build()?;

    // The request's model name is not used for routing: one client holds one model.
    let request = UnifiedRequest::new("local")
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
`examples/local_model.rs`; it also prints download progress on stderr and prints
`Thinking` blocks, and reads `CUCA_LOCAL_MODEL` (a preset name or
`repo:revision:file`) or `CUCA_LOCAL_MODEL_PATH` to run another model, and
`CUCA_LOCAL_DRAFTER` / `CUCA_LOCAL_DRAFTER_PATH` to add a drafter. Run it
built for the host CPU, because candle selects its quantized SIMD kernels at
compile time and a generic build is several times slower:

```bash
RUSTFLAGS="-C target-cpu=native" cargo run --release --example local_model --features provider-local
CUCA_LOCAL_MODEL=qwen3-0.6b-q8_0 RUSTFLAGS="-C target-cpu=native" cargo run --release --example local_model --features provider-local
```

## Facts

| Fact | Value |
|---|---|
| Feature flag | `provider-local`; GPU: `provider-local-vulkan`, `provider-local-metal`, `provider-local-cuda`, `provider-local-rocm` (each enables `provider-local-gpu`, the shared GPU engine) |
| Device | `LocalModelConfig::device`; `LocalDevice::Cpu` by default, no automatic selection |
| `ProviderEndpoint` variant | `Local` |
| Attached with | `CucaClientBuilder::with_local_model`; `build` fails without it |
| Network use | only the one-time model download from Hugging Face |
| Dependencies it owns | `candle-core` and `candle-nn` (0.11), `half`, `minijinja`, `minijinja-contrib`, `unicode-general-category`; with a GPU feature also `burn` 0.22 (`std`, `fusion`, `autotune`); it reuses `reqwest`, `sha2` and `tokio`, and does not use `tokio-stream` |

candle-core 0.11 hard-depends on `tokenizers[onig]`, so the dependency tree
compiles C (`onig_sys`); the feature needs a C compiler.
cuca never calls that crate: the tokenizers, the chat templates and the engine
code are pure Rust.

Every GPU feature compiles C and needs a C compiler on top of the one `onig_sys`
already requires: CubeCL's runtime pulls an embedded store (`turso`) that builds C,
and on Windows a dependency's build script runs the SDK resource compiler (`rc.exe`;
`windres` on the GNU target). `provider-local-cuda` and `provider-local-rocm`
additionally link a prebuilt LLVM; see *Running on a GPU*.

## Choosing a model

`LocalModelConfig::source` is a `ModelSource`. The default is the preset
`gemma4-e2b-q4_0`; `ModelSource::PRESETS` lists every verified preset, each pinned
to one Hugging Face commit and one SHA-256:

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

The table of presets, with sizes, K/V cache cost and what each was verified to do,
is on [Local models and formats](@/providers/local-models.md#presets).
A `ModelPreset` has `name`, `repo`, `revision`, `file`, `sha256`, `size_bytes`,
`architecture` and `drafter`; `preset.source()` is its pinned `ModelSource`, and
`ModelSource::preset(name)` finds one by name (ASCII case-insensitive).

`ModelSource` also implements `FromStr`, so `"qwen3-0.6b-q8_0".parse::<ModelSource>()?`
works. The error for an unknown name lists the preset names. For any other file:

- `ModelSource::hugging_face(repo, revision, file)` for a file on Hugging Face. It
  has no SHA-256, so the download is not verified; the revision should be a commit
  hash, since the cache is keyed by it. Parsing `repo:revision:file` builds the same
  value.
- `ModelSource::HuggingFace { repo, revision, file, sha256 }` to pin a digest
  yourself; `sha256: None` skips the check.
- `ModelSource::File(path)` for a GGUF file already on disk; nothing is
  downloaded.

```rust,name=Selecting a different model
use cuca::{LocalModel, LocalModelConfig, ModelSource};

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Qwen3 0.6B at Q8_0 instead of the Gemma 4 default; the rest of the program
    // is the same as above.
    let model = LocalModel::load(LocalModelConfig {
        source: "qwen3-0.6b-q8_0".parse::<ModelSource>()?,
        ..Default::default()
    })
    .await?;
    println!("{model:?}");
    Ok(())
}
```

A file of an architecture the engine does not know fails to load with a
`CucaError::Provider` that names the architecture and lists the supported ones.
The chat template and the tokenizer come from the file itself; a tokenizer
model the engine does not implement fails the same way.

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
  `None`, is sent as a bearer token; anonymous access works for the pinned files.

## Configuration

| Field | Default | Meaning |
|---|---|---|
| `source` | `ModelSource::default_model()` (preset `gemma4-e2b-q4_0`) | where the GGUF file comes from |
| `cache_dir` | `None` | download cache; `None` resolves `CUCA_CACHE_DIR`, then the platform directory |
| `hf_token` | `None` | Hugging Face token; `None` falls back to `HF_TOKEN` |
| `context_length` | `None` | maximum prompt plus completion tokens per request, and the K/V cache size (`Option<usize>`); `None` resolves to the smaller of 8192 and the model's own maximum, an explicit value must be at least 1 and at most the model's own maximum |
| `sampling` | `SamplingConfig::default()` | default sampling parameters |
| `device` | `LocalDevice::Cpu` | where the model runs; a GPU needs its cargo feature, see *Running on a GPU* |
| `drafter` | `None` | optional multi-token-prediction drafter, an `Option<DrafterSource>`, see *Speculative decoding with a drafter* |
| `draft_tokens` | `3` | tokens the drafter proposes per round, `1..=16`; ignored without a drafter |
| `on_download_progress` | `None` | download progress observer |

`SamplingConfig` defaults to `temperature` 1.0, `top_k` 64, `top_p` 0.95 and no
fixed seed. With `seed: Some(n)`, every request restarts the generator from the
seed, so equal prompts sample equal tokens. `temperature` of `0.0` is greedy.

`LocalModel::load` returns `CucaError::Config` for an invalid configuration (a
zero or oversized `context_length`, a non-finite `temperature`, a `top_p`
outside `(0, 1]`, a `draft_tokens` outside `1..=16` with a drafter, or a `device`
whose cargo feature is not compiled in), `CucaError::Io` for a missing or
unreadable file, `CucaError::Transport` or `CucaError::Http` for download
failures, and `CucaError::Provider` for an unsupported or malformed GGUF file
(another architecture or tokenizer, a tensor dtype candle cannot read, a missing
tensor or metadata key), a failed checksum, or a GPU that cannot be opened.

`LocalModel::kv_cache_bytes` reports the K/V cache size of the loaded model, so
you can check a `context_length` against your memory before sending a request.
`LocalModel::capabilities` returns `LocalCapabilities { reasoning,
can_disable_thinking, tools }`, derived from the model's own chat template and
output conventions: whether the model writes a reasoning block, whether a request
can switch it off (the template reads `enable_thinking`), and whether it has a tool
syntax the output parser understands. Use it to decide, per model, whether to
request thinking or declare tools.

## Speculative decoding with a drafter

A model can run with a multi-token-prediction drafter: a Gemma 4 E2B target with
its assistant, the `gemma4-e2b-mtp-f16` preset (a 170 MB GGUF, architecture
`gemma4-assistant`; a small decoder that reads the target's K/V cache, so it adds no
cache memory), or a Qwen 3.5 model with the head in its own file (below). Each round it proposes up to `draft_tokens` tokens and the target verifies
them in one batched forward pass; `draft_tokens` is a maximum, adapted as the
request runs: the length grows by one after a round the target fully accepts, falls
to the accepted count plus one after a partial round, and after two rounds with no
accepted draft the engine stops drafting for eight steps and restarts at one. At
`temperature` 0.0 a draft is accepted when it equals the target's greedy token,
and the greedy output is identical to a run without the drafter
(`tests/provider_local_mtp.rs` compares them). Above 0.0 a draft is accepted with
probability `min(1, p/q)`, where `p` and `q` are the target's and the drafter's
distributions after temperature, `top_k` and `top_p`, and a rejected draft is
replaced by a token drawn from the residual distribution (the speculative
rejection-sampling rule). A drafter preset cannot be loaded as the main model.
A separate drafter file runs on the CPU and on a GPU device.

`LocalModelConfig::drafter` is an `Option<DrafterSource>`:

- `DrafterSource::File(ModelSource)` is a separate drafter GGUF, resolved and
  downloaded like any model source (the preset above). A `ModelSource` converts with
  `.into()`.
- `DrafterSource::InTrunk` is the prediction head embedded in the main model's own
  GGUF file, for a main model whose file carries one: nothing more is downloaded,
  the head keeps a K/V cache of its own, and it runs on the CPU only (a GPU device
  with `InTrunk` is a `CucaError::Config`). A main model whose file has no such
  head fails to load with an error that says so. Qwen 3.5 files that carry the
  head (for example `unsloth/Qwen3.5-0.8B-MTP-GGUF`, the `qwen3.5-0.8b-q8_0`
  preset) draft with it: the head is one gated-attention block after the trunk,
  with a K/V cache of its own (4 KB per context position for the 0.8B model), fed
  the trunk's last hidden state and the next token. Greedy output equals plain
  decoding, as with a separate file; a hybrid model rolls back at most 16 drafts
  per verification call.

```rust,name=A Gemma 4 target with its drafter
use cuca::{LocalModel, LocalModelConfig, ModelSource};

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = LocalModel::load(LocalModelConfig {
        source: "gemma4-e2b-q4_0".parse::<ModelSource>()?,
        drafter: Some("gemma4-e2b-mtp-f16".parse::<ModelSource>()?.into()),
        draft_tokens: 3,
        ..Default::default()
    })
    .await?;
    // ... send requests; afterwards:
    println!("{:?}", model.draft_stats());
    Ok(())
}
```

`LocalModel::draft_stats` returns `DraftStats { rounds, drafted, accepted }`,
summed over every request, and `DraftStats::acceptance_rate` the accepted
fraction (`None` before the first round). A longer round wins more on
predictable text and wastes more verification on unpredictable text, so
`draft_tokens` is worth tuning on your own prompts.
`examples/local_model.rs` reads `CUCA_LOCAL_DRAFTER` (a preset name, or `in-trunk`)
or `CUCA_LOCAL_DRAFTER_PATH`
and prints the counters at the end, and `tests/provider_local_mtp.rs` compares a
run with and without the drafter.

A governor that weighs the cost of drafting against what it saves backs off to
plain decoding when drafting does not pay. Measured against plain decoding on the
same device (Gemma 4 E2B with `gemma4-e2b-mtp-f16`; a 32-core CPU, an AMD RX 9070
XT over Vulkan, Windows): on the CPU the speedup was 1.17x on code, 1.37x on a
list and 1.00x on prose. On Vulkan it was 0.88x to 1.01x with the governor; with
drafting forced on it was 0.97x, 0.71x and 0.57x on the same three texts. Decode
on Vulkan is bound by the host today, so drafting does not pay there: use the
drafter on the CPU, and expect gains only on predictable text.

With the in-trunk head of Qwen 3.5 0.8B Q8_0 on the CPU at `draft_tokens: 3`, the
speedup was 1.62x on code (72 % of the drafted tokens accepted), 1.18x on a list
and 0.98x on prose. Prose is the case the governor exists for: its text is hard
to predict, so drafting stays mostly off and the run stays near plain decoding.
A draft step ranks only the first 65,536 of the model's 248,320 token ids, which
makes it about five times cheaper than ranking the whole vocabulary for a few
percent fewer accepted tokens per round.

## Running on a GPU

The model runs on the CPU unless `LocalModelConfig::device` says otherwise. A GPU
device runs the forward pass through burn 0.22 and CubeCL, with the same math as
the CPU engine; the CPU path (candle 0.11, same API) is unchanged. The CPU stays
the default: nothing selects a GPU for you, so callers opt in. The engine thread
owns the device. Every architecture the engine supports runs through the same
GPU forward pass.

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

- **Memory.** Device memory is about the file size, minus any table the engine
  reads from the file instead of uploading (Gemma 4's per-layer embedding table,
  read per token on the host as on the CPU), plus the K/V cache. The token
  embedding is uploaded once as the LM head when it is tied to it; embedding
  lookups read rows from the file on the host. The K/V cache lives in device
  memory with the same formula as on the CPU (see *Concurrency and memory*), and
  the RoPE tables are on the device. A weight over 1 GiB on the device is split
  into row chunks.
- **Weight dtypes.** Q4_0 and Q8_0 matmul weights are repacked at load into
  burn's block-quantized layout and stay quantized in device memory (dequantized
  inside the matmul kernel); F32, F16 and BF16 weights are uploaded as f32. A file
  that mixes the two schemes (a Q4_0 file with a few Q4_1 layers and a Q6_K head,
  which the engine keeps as Q8_0, as the Llama 3.2 and Phi 3.5 files are) would make burn's Vulkan matmul autotune replay a kernel
  tuned for one scheme on the other and fail at launch, so the loader holds the
  weights of the scheme with fewer bytes as f32 where they could share a tuning
  key with the other scheme. That is exact (an f32 copy of the file's dequantized
  values, no requantization) and costs a few hundred MB more device memory on such
  files; a file of one scheme is untouched.
  Every other dtype (Q4_1, Q5_0, Q5_1, Q8_1 and the K-quants, for example the
  Phi-4-mini Q4_K_M preset) is dequantized row by row at load and requantized to
  Q8_0, so it costs about its file size or less on the device instead of four bytes
  per element, at the price of a rounding error of at most half a Q8_0 step per
  element; logits then differ from the CPU's more than float rounding alone would
  make them. Measured on Vulkan with the Phi-4-mini Q4_K_M preset: the top-1 token
  equals the CPU's, the relative logit error is 1.8e-2, prefill runs at 97
  tokens/s and decode at 42 tokens/s.
- **Hybrid models.** Qwen 3.5 (gated delta-net layers with a recurrent state) is
  CPU only. Loading it on a GPU device fails with a `CucaError::Provider`:
  `recurrent (gated delta-net) layers are not implemented on GPU devices; use
  LocalDevice::Cpu`.
- **Failures while serving.** A request that fails on the device ends its stream
  with a `CucaError::Provider`; a panic inside the backend is caught and reported as
  `local engine panicked; KV cache reset`, the cache is reset and the engine thread
  keeps running, so the next request is attempted on the same device. The engine
  does not detect a lost device or rebuild it, so requests on a dead device keep
  failing the same way.
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
  live Gemma 4 E2B Q4_0 suite the greedy outputs were identical.
- **Gemma 4 E2B Q4_0 on Vulkan** with a 1,300-token prompt measured 73 tokens/s of
  prefill and 36 tokens/s of decode; the first round is slower while CubeCL
  autotunes and compiles shaders.
- **What was run.** The live matrix of `tests/provider_local.rs` ran on Vulkan on
  that machine for Gemma 4 E2B Q4_0, Llama 3.2 1B, Phi 3.5 mini, Gemma 2 2B, OLMo 2
  1B and Granite 3.3 2B: every applicable scenario passed, with the same known
  limits as on the CPU (see [Local models and formats](@/providers/local-models.md)).
  The other models ran on the CPU. Metal, CUDA and ROCm are compile-checked only.

## Concurrency and memory

- **Single slot.** The engine runs requests one at a time in FIFO order, with at
  most 16 queued behind the running one. Past that, `generate_stream` waits for a
  free queue slot: backpressure, not an error. A running request streams through
  a bounded 64-block channel, so a slow consumer paces generation, and dropping
  the stream stops it.
- **K/V cache.** It is preallocated for `context_length` positions and costs
  `context_length * 2 * 4 * sum(n_kv_head * head_dim)` bytes (f32 keys and
  values), summed over the layers that own K/V, with each layer's own head count
  and head dimension (Gemma 4's last `shared_kv_layers` layers read an earlier
  layer's cache and own none). It is reused across requests: a prompt that shares
  a prefix with the previous one only prefills the new tokens, reported as the
  response's `prompt_cache_usage`. The cost varies a lot between models: at the
  default 8192 positions it is 302 MB for Gemma 4 E2B and 940 MB for E4B, but
  6.44 GB for Phi-3.5-mini, which has 32 K/V heads. The table of presets lists
  the measured value for each.
- **Recurrent state (Qwen 3.5).** A hybrid model also keeps a recurrent state
  that does not grow with the context: 20.2 MB for Qwen 3.5 0.8B, next to 24,576
  bytes of K/V per position. Prefix reuse restarts from one checkpoint of that
  state (one slot: a new checkpoint replaces the old one), and rolling back a
  speculative verification keeps a copy of the state before the call: about 61 MB
  in all, plus 445 KB per verified row. See
  [Local models and formats](@/providers/local-models.md#architectures).
- **Weights.** They stay quantized and are consumed by the engine's own quantized matmul kernel (Q4_0 and Q8_0) or candle's (other dtypes);
  norm vectors, biases and float-typed weights are held as f32. Resident weights
  are about the file size. Two tables are not held: Gemma 4's per-layer embedding
  table (`per_layer_token_embd`: 1.32 GB in E2B Q4_0, 1.59 GB in E4B Q4_0) is not
  loaded, and one row per input token (5,040 bytes in E2B Q4_0) is read from the
  GGUF file, usually from the OS page cache, so E2B Q4_0 holds 1.5 GB of its
  2.84 GB file.
- **Mixture of experts.** Expert tensors are split once at load into one
  quantized 2-D weight per expert, so a MoE model costs its file size, and the
  transient memory of a forward pass does not depend on the expert count.
- **Prefill.** It runs in chunks (at most 256 tokens, fewer for a long context),
  which bounds the transient attention score matrix.
- **GPU.** On a GPU device the weights and the K/V cache are in device memory
  instead of RAM, at the sizes described in *Running on a GPU*; the engine and
  its queue behave the same.
- **Cloning.** A `LocalModel` is cheap to clone, and every clone shares the one
  engine thread. The thread ends when the last clone is dropped.

## Behavior

- Thinking is binary: a request with thinking enabled sets `enable_thinking` (and `thinking`, which Granite 3.3's template reads) for
  the model's chat template, and its effort level is ignored. Reasoning arrives
  as `Thinking` blocks, and `reasoning_tokens` is reported in the usage only when
  the request enabled thinking. A model whose template has no switch (the
  DeepSeek-R1 distills) reasons on every request.
- Tools are declared to the model through its own chat template, which only sees
  a `tools` variable when the request has tools; tool calls come back as
  `ToolCall` blocks whose ids the engine generates. How a call is recognised
  depends on the model's family; see [Local models and formats](@/providers/local-models.md#tool-calls).
- The engine is text-only: a message with an `ImageBase64` block is rejected with
  a provider error.
- `UnifiedRequest::model` is not used for routing, since one client holds one
  model. `temperature` and `max_tokens` apply per request; `top_k`, `top_p` and
  `seed` come from `SamplingConfig`. Generation also stops at the model's
  end-of-generation token and at the context limit.
- Token usage in the response is exact, not estimated: the prompt tokens equal
  the prefix-reuse read plus write counts, and a `max_tokens` cap is met to the
  token.

## Testing it

`tests/provider_local.rs` runs the real model and needs no server. Point
`CUCA_LOCAL_MODEL_PATH` at a GGUF file, set `CUCA_LOCAL_MODEL` to a preset name
or `repo:revision:file` to download that model into the cache, or set
`CUCA_LOCAL_DOWNLOAD=1` to download the default (E2B Q4_0). `CUCA_LOCAL_MODEL_PATH`
wins when several are set. With none set, each test prints
`SKIP: local model not available: ...` and passes, unless `CUCA_REQUIRE_LIVE=1`
turns the skip into a panic.

The suite is a matrix: one model per run, and each scenario applies to a model
only when `LocalModel::capabilities()` says it can do the thing: it reasons, it
can switch reasoning off, it has a tool syntax. A scenario that does not apply
prints `MATRIX|<scenario>|SKIP|<reason>`; one that applies and fails fails the
test; a scenario that is a documented limit of the model prints
`MATRIX|<scenario>|KNOWN-LIMIT` (or `SKIP|KNOWN-LIMIT: ...`) with its evidence. The
scenarios are:

- a factual greedy answer, with exact usage (prompt counts equal the
  prefix-reuse counts, a `max_tokens` cap is met to the token, and, when the
  HuggingFace oracle dumps are given in `CUCA_LOCAL_REF_DIR`, the prompt token
  count equals the oracle's);
- thinking on (`Thinking` blocks precede the answer, `reasoning_tokens` is
  reported) and thinking off (no `Thinking` blocks);
- a tool round trip, with the tool declared on both requests and again with no
  tool declared on the follow-up;
- a prompt of about 1,300 tokens with a code word to retrieve;
- prefix reuse: the warm prefix is read from the cache, an unrelated request
  shrinks it, and the warm and the cold greedy output are identical.

```bash
CUCA_LOCAL_DOWNLOAD=1 RUSTFLAGS="-C target-cpu=native" cargo test --release --features provider-local --test provider_local -- --test-threads=1 --nocapture
CUCA_LOCAL_MODEL=qwen3-0.6b-q8_0 RUSTFLAGS="-C target-cpu=native" cargo test --release --features provider-local --test provider_local -- --test-threads=1 --nocapture
```

Set `CUCA_LOCAL_DEVICE` (the `LocalDevice` text form above) to run the same
scenarios on a GPU. The variable is read only by the tests and by
`examples/local_model.rs`, not by the library, and a GPU device needs its feature:

```bash
CUCA_LOCAL_DEVICE=vulkan CUCA_LOCAL_DOWNLOAD=1 cargo test --release --features provider-local-vulkan --test provider_local
CUCA_LOCAL_DEVICE=vulkan cargo run --release --example local_model --features provider-local-vulkan
```

The example prints `loaded on <device>`. The unit tests `gpu_matches_cpu_on_the_tiny_gemma4`,
`gpu_matches_cpu_on_every_architecture` and `gpu_matches_cpu_on_other_weight_dtypes`
compare the GPU engine to a CPU reference on tiny random models (and
`gpu_drafter_matches_cpu_drafter` does the same for the drafter). They run only
when `CUCA_LOCAL_DEVICE` names a GPU, and otherwise print
`SKIP: set CUCA_LOCAL_DEVICE to a GPU device`:

```bash
CUCA_LOCAL_DEVICE=vulkan cargo test --release --no-default-features --features provider-local-vulkan --lib gpu_
```

`tests/provider_local_mtp.rs` runs a target alone and with its drafter (a Gemma 4
target with its assistant file, or a Qwen 3.5 `*-MTP` file with `in-trunk`) and
checks that the greedy output is identical. It needs both files
(`CUCA_LOCAL_MODEL` or `CUCA_LOCAL_MODEL_PATH`, and `CUCA_LOCAL_DRAFTER`,
`CUCA_LOCAL_DRAFTER=in-trunk` or `CUCA_LOCAL_DRAFTER_PATH`); `CUCA_LOCAL_DRAFT_TOKENS`
may be a comma-separated list to run one comparison per value.
