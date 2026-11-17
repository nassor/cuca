+++
title = "Run on a GPU"
description = "Pick the GPU feature, set device in the config, run the local model on Vulkan, Metal, CUDA or ROCm, and learn what the GPU runs, how much video memory it needs and what speed to expect."
template = "page.html"
aliases = ["providers/local/gpu/"]
weight = 6
+++

# Run on a GPU

<dl class="page-facts">
<dt>In one line</dt>
<dd>Set <code>device</code> in the model config to move the model onto your GPU. Nothing picks a GPU for you, and the CPU stays the default.</dd>
<dt>You need</dt>
<dd>One GPU feature (<code>provider-local-vulkan</code>, <code>provider-local-metal</code>, <code>provider-local-cuda</code> or <code>provider-local-rocm</code>) and the normal driver for your GPU.</dd>
<dt>Read this if</dt>
<dd>You want the local model on a GPU, or you want to know what the GPU does and does not run. For a model that is bigger than your GPU memory, go to <a href="@/local/placement.md">Fit a large model</a>.</dd>
</dl>

A GPU runs the model's math much faster than a CPU, because it does many small calculations at once. The model lives in the GPU's own memory, called VRAM. This page shows how to turn a GPU on and what to expect. The GPU path is built on burn and CubeCL, two Rust GPU libraries.

## Run a model on a GPU

### 1. Pick the feature for your GPU

A cargo feature is a build option. Each GPU brand has its own feature:

| Your GPU | Feature | Status | Extra build needs |
|---|---|---|---|
| A GPU with a Vulkan driver (AMD, NVIDIA, Intel) | `provider-local-vulkan` | Verified: run by hand on Windows with an AMD RX 9070 XT | a C compiler; `rc.exe` on Windows |
| Apple GPU | `provider-local-metal` | Compile-checked only, never run on Apple hardware | a C compiler; runs on Apple only |
| NVIDIA GPU | `provider-local-cuda` | Compile-checked only | a prebuilt LLVM, downloaded on the first build |
| AMD GPU | `provider-local-rocm` | Ran on Windows; compile-checked on Linux | the same LLVM download |

If you are unsure, choose Vulkan. `provider-local-gpu` is the shared engine behind all four and selects no backend by itself. The GPU driver is a system library that loads when the program starts, for example the Vulkan loader. CI has no GPU, so every GPU run is manual. See [Support status and known limits](@/local/verification.md) for what ran. Build tools for each platform are on [Build requirements](@/local/build-and-tests.md).

### 2. Write the config

Create `vulkan.kdl`. With no `source` line the example loads its default model, Gemma 4 E2B Q4_0, and downloads it on the first run:

```kdl,name=vulkan.kdl
device "vulkan"
```

### 3. Run it

```bash,name=Runs the same on all three platforms
cargo run --release --example local_model --features provider-local-vulkan -- --config vulkan.kdl
```

The example prints the load time, the device and the adapter it resolved to on stderr, then streams the reply:

```text,name=Expected on stderr
load: <seconds> s
loaded on vulkan
adapter vulkan = default adapter: <the name of your GPU>
```

The load also names the adapter it chose. A machine can hold several (a discrete card and the processor's integrated one), and `device "vulkan"` picks one for you. As each GPU device opens, before the weights upload, the engine logs one `info` line through `tracing`, so a load that then fails for lack of memory still says which adapter it was on:

```text,name=Expected log line
gpu adapter: vulkan:0 = discrete #0: AMD Radeon AI PRO R9700
```

The text after `=` says what the index counted. `vulkan:0` is the first discrete adapter in the order the graphics API lists them, `vulkan:integrated:0` the first integrated one, and plain `vulkan` the adapter wgpu prefers (`default adapter`; `default adapter, CPU` when no card stands behind it, as with the llvmpipe software rasterizer). The graphics backend does not report the type of a default adapter, so only a numbered index says discrete or integrated. `cuda:1` and `rocm:1` are the driver's device 1. `cuca run` prints the line with the rest of its log, a library user installs a `tracing` subscriber, and `LocalModel::gpus()` returns the same as `GpuInfo` values. The server's `model ready` line lists the adapters, and the [stats snapshot](@/server/stats-and-metrics.md) carries the names in `models[].memory.device_name`.

The first request is slow. The GPU compiles and tunes its kernels (small GPU programs) the first time the model sees each new input shape. Later requests are fast.

### 4. Use it from Rust

The same model and API as on the CPU. Only the device changes:

```rust,name=Running on a GPU (provider-local-vulkan)
use cuca::{GpuAdapter, LocalDevice, LocalModel, LocalModelConfig};

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = LocalModel::load(LocalModelConfig {
        device: LocalDevice::Vulkan(GpuAdapter::Default),
        ..Default::default()
    })
    .await?;
    println!("{model:?}");
    Ok(())
}
```

## Device strings

`device` is a string in KDL and a `LocalDevice` in Rust. The text is not case sensitive:

| Text | Meaning | Rust value |
|---|---|---|
| `cpu` | the CPU, the default | `LocalDevice::Cpu` |
| `vulkan`, `metal` | the default GPU of that backend | `Vulkan(GpuAdapter::Default)`, `Metal(GpuAdapter::Default)` |
| `vulkan:<n>`, `metal:<n>` | the n-th discrete GPU | `Vulkan(GpuAdapter::Discrete(n))` |
| `vulkan:integrated:<n>`, `metal:integrated:<n>` | the n-th integrated GPU (the one inside the CPU) | `Vulkan(GpuAdapter::Integrated(n))` |
| `cuda`, `cuda:<n>` | CUDA device n; bare `cuda` is device 0 | `Cuda(n)` |
| `rocm`, `rocm:<n>` | ROCm device n; bare `rocm` is device 0 | `Rocm(n)` |

Two errors to know:

- Any other text fails with a `CucaError::Config`: ``unknown local device `...`; expected cpu, vulkan[:<n>|:integrated:<n>], metal[...], cuda[:<n>] or rocm[:<n>]``.
- A device whose feature is off fails the load: ``device Vulkan(Default) requires the `provider-local-vulkan` cargo feature``. Rebuild with that feature.

Other load errors are on [Limits and errors](@/local/limits-and-errors.md).

## What runs on the GPU

- Every supported model architecture runs on the GPU, except the Gemma 4 mixture of experts (26B-A4B). It runs on the CPU only, and a GPU load of it fails with a `CucaError::Provider`.
- The Gemma 4 assistant drafter runs on the GPU. The Qwen 3.5 in-trunk head, EAGLE, DFlash and DSpark drafters are CPU only. See [Speculative decoding](@/local/drafting.md).
- A model larger than the card splits between the GPU and the CPU. See [Fit a large model](@/local/placement.md).
- DeepSeek-V4.1 on a GPU has its own steps and limits: see [Run DeepSeek-V4.1](@/local/deepseek-v41.md).
- The GPU keeps activations in f32, so its logits differ slightly from the CPU's. A batch of rows equals single-token steps only to float rounding, so a speculative greedy run on a GPU can differ from plain decoding where two choices are that close.

## GPU memory

How a weight is stored decides how much VRAM it takes.

- Quantized weights stay in the file's own blocks: the legacy formats, the K-quants behind files such as Q4_K_M, the I-quants, ternary and FP4 formats. One copy sits in VRAM, about the file size. Only Q8_1 becomes Q8_0 at load. See [Weight formats](@/local/models.md#weight-formats).
- F32, F16 and BF16 weights are held as f32, so an F16 or BF16 model takes twice its file size.
- The K/V cache (the attention memory of past tokens, see [How it works](@/local/how-it-works.md)) lives in VRAM as f32. By default it holds `context_length` plus at most 4,096 positions (12,288 at an 8,192 context), where the CPU holds twice `context_length`: VRAM is the scarcer memory. The [presets table](@/local/models.md#the-presets) gives each preset's cache on both, and [When to change kv_tokens](@/local/configuration.md#when-to-change-kv-tokens) says when to set it yourself.
- Dropping the last clone of a `LocalModel` returns its VRAM to the driver, so another model can load in the same process.
- Every new batch size compiles kernels once per process, so the first request of each size is slower.
- If the model does not fit, the load fails with `the device failed while the model loaded (most likely out of memory: ...)`. Lower `context_length` or `batching kv_tokens`, or use `gpu_layers` or expert placement from [Fit a large model](@/local/placement.md).

## Several GPUs

Tensor parallelism splits every layer over several GPUs. In `device` and `shards`, one string names each GPU:

```kdl,name=two-gpus.kdl
device "vulkan:0"
shards "vulkan:1"
```

The model then runs on `1 + shards.len()` devices. Each GPU holds its slice of the weights and its own K/V cache. The GPUs add their partial results twice per layer.

<div class="dgm">
<div class="dgm-scroll">
<svg viewBox="0 0 660 176" role="img" aria-labelledby="shard-title shard-desc">
  <title id="shard-title">One layer split over two GPUs</title>
  <desc id="shard-desc">The hidden state goes to both GPUs. GPU 0 and GPU 1 each compute half of the attention heads and half of the feed-forward width. A box then adds the two halves. This happens twice per layer, after attention and after the feed-forward block. The sum goes to the next layer.</desc>
  <defs>
    <marker id="shard-arrow" viewBox="0 0 8 8" refX="7" refY="4"
            markerWidth="6" markerHeight="6" orient="auto">
      <path d="M0 0.6 L7 4 L0 7.4 z" fill="var(--muted-foreground)"/>
    </marker>
  </defs>
  <text class="t-title" x="6" y="20">Tensor parallelism: one layer on two GPUs</text>
  <rect class="blk blk-data" x="6" y="60" width="110" height="50" rx="6"/>
  <text class="t-sm t-mid t-data" x="61" y="82">hidden state</text>
  <text class="t-sm t-mid" x="61" y="96">on every GPU</text>
  <rect class="blk blk-ctl" x="170" y="36" width="170" height="46" rx="6"/>
  <text class="t-sm t-mid t-ctl" x="255" y="55">GPU 0</text>
  <text class="t-sm t-mid" x="255" y="69">half the heads and FFN</text>
  <rect class="blk blk-ctl" x="170" y="100" width="170" height="46" rx="6"/>
  <text class="t-sm t-mid t-ctl" x="255" y="119">GPU 1</text>
  <text class="t-sm t-mid" x="255" y="133">half the heads and FFN</text>
  <rect class="blk" x="400" y="60" width="130" height="50" rx="6"/>
  <text class="t-sm t-mid" x="465" y="82">add the halves</text>
  <text class="t-sm t-mid" x="465" y="96">twice per layer</text>
  <rect class="blk" x="580" y="60" width="74" height="50" rx="6"/>
  <text class="t-sm t-mid" x="617" y="89">next layer</text>
  <path class="arw arw-data" d="M118 85 H142 V59 H166" marker-end="url(#shard-arrow)"/>
  <path class="arw arw-data" d="M118 85 H142 V123 H166" marker-end="url(#shard-arrow)"/>
  <path class="arw arw-data" d="M342 59 H370 V85 H396" marker-end="url(#shard-arrow)"/>
  <path class="arw arw-data" d="M342 123 H370 V85 H396" marker-end="url(#shard-arrow)"/>
  <path class="arw arw-data" d="M532 85 H576" marker-end="url(#shard-arrow)"/>
  <text class="t-ax" x="6" y="168">FFN is the feed-forward block of a layer. Each GPU keeps its own slice of the weights.</text>
</svg>
</div>
<p class="dgm-cap"><b>Every layer is cut in half, and the halves are added back together.</b> The sum goes through the CPU unless the GPUs are distinct NVIDIA cards on Linux, which use NCCL.</p>
</div>

- Use it only when the model does not fit one GPU. If the model fits one card, a split decode is slower, because the sum makes two round trips through host memory per layer.
- A split model agrees with the one-GPU model to float rounding, not bit for bit. Batching and prefix reuse work as on one GPU.
- One GPU used twice (`device "vulkan:0"` with `shards "vulkan:0"`) gave the single-GPU replies at about a third of the speed. Two different GPUs and the NCCL path have not been run.
- The load fails with a `CucaError::Config` (`tensor parallelism: <rule> (<numbers>)`) for a CPU device, a shard of another backend, a `drafter`, `gpu_layers` and tiered or host experts.
- A few models cannot be split: Qwen 3.5 and its mixture of experts variant (gated delta-net layers), `mistral4` (latent attention), `deepseek41`, OLMo 2 and OLMoE (full-width Q/K norm). A head count that does not divide by the number of GPUs, and layer widths that do not divide into multiples of 32 per GPU, are rejected too.

## ROCm on Windows

`provider-local-rocm` builds and runs on Windows on AMD RDNA4 cards. It was run on an RX 9070 XT (gfx1201). The AMD display driver does not ship the HIP runtime that CubeCL loads (`amdhip64.dll` and `hiprtc.dll`), so you take it from AMD's ROCm 10.1.0 release. The Windows tarball extracts without administrator rights. The steps on AMD's install page use `setx ... /M`, which needs them, so the steps below set per-session variables instead.

The tarball is per GPU family. `gfx120X-all` covers the RDNA4 cards gfx1200 and gfx1201. Another card needs its own family, which was not tried.

### 1. Install the HIP runtime

```powershell,name=Windows (PowerShell): install the HIP runtime without administrator rights
$root = "C:\TheRock"
$tar = "therock-dist-windows-gfx120X-all-10.1.0.tar.gz"
New-Item -ItemType Directory -Force "$root\build" | Out-Null
curl.exe -L -o "$root\$tar" "https://stable.repo.amd.com/rocm/core/tarball/$tar"
tar -xzf "$root\$tar" -C "$root\build" --strip-components=1
# The five DLLs CubeCL needs, in a folder of their own, two under the names it asks for.
$dll = "$root\cuca-dlls"
New-Item -ItemType Directory -Force $dll | Out-Null
Copy-Item "$root\build\bin\amdhip64_7.dll" "$dll\amdhip64.dll"
Copy-Item "$root\build\bin\hiprtc0716.dll" "$dll\hiprtc.dll"
foreach ($n in "hiprtc0716.dll", "hiprtc-builtins0716.dll", "amd_comgr.dll", "rocm_kpack.dll") {
    Copy-Item "$root\build\bin\$n" "$dll\$n"
}
& "$root\build\bin\hipInfo.exe" | Select-String "^device#|^Name|gcnArchName"
Test-Path "$root\build\lib\llvm\amdgcn\bitcode\ocml.bc"
```

The download is 2.24 GB. `hipInfo` lists every GPU as `device# N` with its name, and `Test-Path` prints `True`. On a PC with a Ryzen integrated GPU, `device# 0` is `AMD Radeon(TM) Graphics` (`gfx1036`) and `device# 1` is the discrete card, `AMD Radeon RX 9070 XT` (`gfx1201`). Note the number of your card.

### 2. Write the config

```kdl,name=rocm.kdl
source "qwen3-0.6b-q8_0"
device "rocm:1"
```

### 3. Run it

```powershell,name=Windows (PowerShell): run on ROCm
$dll = "C:\TheRock\cuca-dlls"
$env:PATH = "$dll;" + $env:PATH
$env:HIP_PATH = $dll
$env:HIP_DEVICE_LIB_PATH = "C:\TheRock\build\lib\llvm\amdgcn\bitcode"
$env:RUSTFLAGS = "-C target-cpu=native"
cargo run --release --no-default-features --features provider-local-rocm --example local_model -- --config rocm.kdl --prompt "Explain in two sentences what a mixture of experts model is." --thinking off
```

The run prints `loaded on rocm:1` and then the reply. The `$env:` values last for one shell, so set them again in a new one.

### 4. Three rules

- Keep the SDK's own `bin` folder off `PATH`, for the build and for the run. It holds `hipconfig.exe`. The build script of `cubecl-hip-sys` runs it to choose HIP bindings, and the tarball's HIP version is newer than the bindings the crate bundles. With the folder on `PATH`, the build of `cubecl-hip` fails with many unresolved type errors (`hipGraph_t`, `hipStream_t` and similar).
- Put the DLL folder on `PATH`. `amdhip64.dll` imports `rocm_kpack.dll` from its own folder, and the loader does not look there. Without it, the load fails with `device Rocm(1): 0 such device(s) found`. That message names `ROCM_PATH`, `HIP_PATH`, the `PATH` rule and `HIP_DEVICE_LIB_PATH`.
- Use `rocm:1` when the PC has an integrated GPU. HIP numbers the integrated GPU 0 and the discrete card 1. With one GPU, `rocm` (device 0) is enough.

Known limit: on ROCm the first request of a process can take minutes while kernels compile, and some runs decode much slower than Vulkan. If you see that, run on Vulkan.

## What to expect

Method: a prompt of about 1,100 tokens and 96 generated tokens, prefix cache off, Vulkan on an AMD RX 9070 XT (16 GB) under Windows, against a Ryzen 9 desktop CPU, release build with native CPU flags. Prefill is the time to read the prompt, until the first token. Decode is tokens per second after that. Figures are rounded, and the first round (kernel compile) is left out.

| Model | Where it runs | Decode, tokens/s | Prefill |
|---|---|---|---|
| `gemma4-e2b-q4_0` | GPU | 48 to 57 | 1 to 4 s |
| `gemma4-e2b-q4_0` | CPU | 27.5 | 11 s |
| `phi4-mini-q4_k_m` | GPU (measured when K-quant weights became Q8_0 at load) | 52 to 59 | 2 to 4 s |
| `phi4-mini-q4_k_m` | CPU | 19.6 | 10 s |
| `muse-glimmer-30b-q4_0` (16 GB file, bigger than the card) | CPU | 3.5 | 81 s |
| `muse-glimmer-30b-q4_0` | GPU, `gpu_layers 38` of 52 | 8.3 | 39 s |

- The GPU shortens every prefill.
- A small model that fits the card decodes about 1.7 to 3 times faster on the GPU.
- The Muse row splits the model between GPU and CPU. How to choose `gpu_layers` is on [Fit a large model](@/local/placement.md).
