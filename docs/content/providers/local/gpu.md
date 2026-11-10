+++
title = "Local engine on a GPU"
description = "LocalDevice, the GPU cargo features, tensor parallelism, layer offload, host experts, device memory, measured speed and what ran on which backend."
template = "page.html"
weight = 11
+++

# Local engine on a GPU

<dl class="page-facts">
<dt>In one line</dt>
<dd><code>LocalModelConfig::device</code> moves the forward pass to Vulkan, Metal, CUDA or ROCm; nothing selects a GPU for you.</dd>
<dt>You need</dt>
<dd>One of <code>provider-local-vulkan</code>, <code>provider-local-metal</code>, <code>provider-local-cuda</code> or <code>provider-local-rocm</code>, and the system GPU driver.</dd>
<dt>Read this if</dt>
<dd>You want the engine on a GPU, or need the device text form, the memory cost, the split options or the verification status.</dd>
</dl>

## Devices and GPU features

The model runs on the CPU unless `LocalModelConfig::device` selects a GPU. A GPU device runs the forward pass through burn 0.22 and CubeCL, and the engine thread owns the device. Every supported architecture runs through the GPU forward pass. The in-trunk head, the EAGLE drafter, the DFlash drafter and the DSpark drafter are CPU only; the Gemma 4 assistant runs on the GPU.

| Feature | Backend | Extra build needs |
|---|---|---|
| `provider-local-gpu` | the shared GPU engine | selects no backend by itself |
| `provider-local-vulkan` | Vulkan through wgpu, SPIR-V shaders | C compiler; `rc.exe` on Windows |
| `provider-local-metal` | Metal through wgpu, MSL shaders | C compiler; `rc.exe` on Windows; runs on Apple only |
| `provider-local-cuda` | CUDA | prebuilt LLVM, downloaded on the first build |
| `provider-local-rocm` | ROCm (HIP) | the same LLVM download; Linux and Windows |

The GPU drivers are system libraries loaded at runtime, the Vulkan loader for example. `Cargo.toml` leaves `provider-local-cuda` and `provider-local-rocm` out of the docs.rs feature list because docs.rs builds have no network for the LLVM download.

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

The example `examples/local_model.rs` reads the device from `CUCA_LOCAL_DEVICE` and prints `loaded on Vulkan(Default)` on stderr once the model is on the device.

```bash,name=Linux and macOS: run the example on a GPU
export CUCA_LOCAL_DEVICE="vulkan"
```

```powershell,name=Windows (PowerShell): run the example on a GPU
$env:CUCA_LOCAL_DEVICE = "vulkan"
```

```bash,name=Runs the same on all three platforms
cargo run --release --example local_model --features provider-local-vulkan
```

## Device selection

`LocalDevice` is `Cpu` (the default), `Vulkan(GpuAdapter)`, `Metal(GpuAdapter)`, `Cuda(usize)` or `Rocm(usize)`. `GpuAdapter` is `Default`, `Discrete(usize)` or `Integrated(usize)`. Every variant exists in every build. `LocalDevice::required_feature` returns the cargo feature a device needs, `None` for `Cpu`.

`LocalDevice` parses from text, ASCII case-insensitive:

| Text | Device |
|---|---|
| `cpu` | `Cpu` |
| `vulkan`, `metal` | `Vulkan(GpuAdapter::Default)`, `Metal(GpuAdapter::Default)` |
| `vulkan:<n>`, `metal:<n>` | the n-th discrete GPU, `Discrete(n)` |
| `vulkan:integrated:<n>`, `metal:integrated:<n>` | the n-th integrated GPU, `Integrated(n)` |
| `cuda`, `cuda:<n>` | `Cuda(n)`; bare `cuda` is device 0 |
| `rocm`, `rocm:<n>` | `Rocm(n)`; bare `rocm` is device 0 |

Any other text is a `CucaError::Config`: ``unknown local device `...`; expected cpu, vulkan[:<n>|:integrated:<n>], metal[...], cuda[:<n>] or rocm[:<n>]``. A device whose feature is off fails the load with ``device Vulkan(Default) requires the `provider-local-vulkan` cargo feature``. The other load errors are on [Local limits and errors](@/providers/local/limits-and-errors.md#errors).

## Tensor parallelism

`LocalModelConfig::shards` splits one model over several GPUs. The model runs on `1 + shards.len()` devices, `device` being the first. `LocalModelConfig::set_devices("vulkan:0+vulkan:1")` fills both fields from one `+` separated list, which is the text `CUCA_LOCAL_DEVICE` carries in the tests and the example. A device may repeat: `vulkan:0+vulkan:0` splits the model in two on one GPU. That gives no speedup and holds the replicated parts twice, but it runs the whole path.

- Each layer's query heads are split evenly, with the K/V heads they read. A model with fewer K/V heads than devices keeps a copy of each K/V head on every device that needs it (Gemma 4 E2B has one K/V head, so with two devices each holds it whole). The FFN width is split evenly. In a mixture of experts every device holds all experts, each cut to its share of the width, with the router replicated; the shared expert is cut the same way.
- Each device holds its slice of the weights and its own K/V pools. Norms, embeddings, Gemma 4's per-layer inputs and the RoPE tables are replicated. Only the first device holds the LM head and runs the final norm. `LocalModel::kv_cache_bytes` sums the pools of all devices, which exceeds the single-device number when K/V heads are replicated.
- The partial outputs are summed twice per layer, after the attention output projection and after the FFN down projection. On CUDA under Linux with every device a distinct GPU the sum is burn's NCCL `all_reduce`. Everywhere else it is host staged: every partial is read back, added in f32 in device order and uploaded to every device. The order is fixed, so repeated runs give the same bits. The host path makes two round trips per layer, so a split decode is slower than one device when the model fits on one. Tensor parallelism is for a model that does not fit one device.
- A split model agrees with the single-device GPU model to float rounding, not bit for bit: two half-width partial sums add in another order than one full-width matmul.
- Continuous batching and prefix reuse work as on one device.

The load fails with `CucaError::Config` (`tensor parallelism: <rule> (<numbers>)`) for a CPU device or a shard of another backend than `device`, a `drafter`, and `ExpertPlacement::Tiered` or `Host`. Before any weight is read it also rejects a model that cannot be split:

- a gated delta-net layer (Qwen 3.5 and its mixture of experts variant);
- multi-head latent attention (`mistral4`);
- hyper-connections with compressed sparse attention (`deepseek41`);
- a full-width Q/K norm (OLMo 2, OLMoE);
- an `n_head` not divisible by the device count, or K/V heads that neither divide nor replicate;
- a query width per device that is not a multiple of 32, or a dense FFN, expert or shared expert width that does not divide into multiples of 32 per device.

## Device memory

- Q4_0, Q4_1 and Q8_0 matmul weights are repacked at load and stay quantized on the device, one copy per weight.
- F32, F16 and BF16 matmul weights are held as f32 (4 bytes per element, so an F16 or BF16 weight doubles). Norm vectors, biases and scalars are f32.
- MXFP4 experts stay in the file's format in the expert kernels. Every other quantized dtype (Q5_0, Q5_1, Q8_1 and the K-quants Q2_K to Q6_K) is requantized to Q8_0 at load, about 1.06 bytes per element. A 3.8 B parameter Q4_K_M file needs about 4 GB on the device rather than 15 GB. Each element moves by at most half a Q8_0 step, `block_max / 254`, and Q8_0 resolves a block with more levels than Q4_K, Q5_K or Q6_K do, so the extra error is below the one the file already carries.
- A weight over 256 MiB is split into row chunks, so no weight buffer exceeds 256 MiB and no buffer at all exceeds 1 GiB. The chunk size keeps burn's memory pages small, which lets a nearly full card place the next page.
- The token embedding is uploaded once, as the LM head, only when it is tied to it. Embedding lookups read rows from the file on the host, and Gemma 4's per-layer embedding table is never uploaded.
- The K/V cache lives on the device as the same block pools as on the CPU: f32, twice `context_length` positions by default, in blocks of 16. The RoPE tables are on the device.
- A mixture of experts keeps its experts in the MoE kernels' layout: the file's bytes for Q4_0, Q4_1, Q8_0 and MXFP4 experts, Q8_0 for other quantized dtypes, f32 for float dtypes. `ExpertPlacement::Tiered` reads no expert at load; see [Tiered expert placement](@/providers/local/expert-tiers.md).
- Dropping the last clone of a `LocalModel` on a GPU returns its device memory to the driver, so another model can load in the same process.
- The first forward pass of each new input shape is slow while CubeCL autotunes and compiles kernels. Continuous batching adds shapes: every distinct number of rows in a step pays this once per process. A current figure for that first request is not recorded.

Dtype handling on the CPU is on [Weight formats](@/providers/local/models.md#weight-formats).

## Kernels

- Dense weights (attention projections, dense FFNs, the LM head) run through two CubeCL kernels of this crate. Up to 64 input rows (decode, batched decode, verification, prompt tails) one matrix-vector kernel dots the blocks with the input in f32, with several threads per row for weights that have few rows. More rows (prefill) dequantize slices of the weight to f32 and use burn's float matmul. burn's quantized matmul is not used by any weight.
- A mixture of experts runs four kernel dispatches per layer: softmax and top-k of the router logits, `act(gate . x) * (up . x)` for every (token, expert) pair, the down projections weighted by the routing, and a sum over each token's experts in expert order. The up and down kernels split the blocks of each output row over 8 lanes of a cube (a row's result does not depend on the other rows of the launch). A resident layer never reads the router output back, so a resident token queues all its layers without waiting for the host. A tiered layer reads back the router output (ids and weights, a few hundred floats) to find the experts its pool lacks.
- Attention over one query row (a decode step, every drafter layer) reads the paged K/V in place with one kernel, a single launch per layer when every sequence in the step decodes one row. Several rows (prefill, verification) gather the visible blocks and use burn's matmul attention with a mask. A `mistral4` layer caches one shared `[latent 256 | rotary 64]` row per position and no value pool: the key up-projection is absorbed into the query and the value up-projection applies to the attended latent, so no per-head K or V is built.
- A gated delta-net layer (Qwen 3.5) runs its recurrence token by token on the device. Its recurrent state is a set of immutable device tensors, so a snapshot costs nothing. The GPU keeps no verify record for a hybrid model, and no drafter pairs with one on a GPU.
- On Vulkan, single-token decode is bound by the host: each queued operation costs about 10 microseconds of host time.

## DeepSeek-V4.1 on a GPU

`deepseek41` runs on one GPU through the kernels of `hc_kernel.rs`, `csa_kernel.rs`, `topk_kernel.rs` and `qdq_kernel.rs` and the MoE kernels of `moe_kernel.rs`.

- The residual stream is one `[rows, streams * n_embd]` tensor. One kernel per site computes the mixing coefficients of every token (the matrix-vector product with `hc_*_fn`, the sigmoids, the row softmax and the Sinkhorn passes), others collapse a stream to a sub-block input and expand the output back.
- Each layer owns a window pool and, for a compressed-row owner, a compressed pool, both 2-D `[blocks * 16, width]` tensors addressed through the sequence's block tables. One kernel writes the new rows in place, one rotates the tail of every head, one pools the pairs of a ratio-2 compressor, and one attends over the window and the picked compressed rows with the head's sink, split over key chunks and merged in chunk order. A row's result does not depend on the other rows of the launch.
- The indexer scores every visible compressed row on the device and selects the best rows with a radix threshold select (ties go to the lower row), so a decode step reads nothing back for it. The candidate blocks of the source layer stay on the device for the layers that narrow their scoring.
- The simulated quantization (FP8 window keys and output, FP4 compressed rows and indexer vectors) runs in elementwise kernels.
- The compressor's half-finished pair stays in each sequence slot as device tensors: the last rows of the previous call, staged during a pass and kept when it succeeds, so a rollback within the last call restores it.
- MXFP4 experts keep their file blocks (16 nibble bytes and an E8M0 scale byte per 32 values, 17 bytes) in the expert pool: one expert of the real model takes 18.8 MB, so a layer's 384 experts (7.2 GB) need tiers. The router scores `sqrt(softplus)`, adds the selection bias for the choice only, and the SwiGLU clamp applies in the up kernel; the shared expert uses burn operations.
- Engram rows are read on the host from the GGUF file (`EngramTable`, bounded by `LocalModelConfig::engram`, see [Local limits and errors](@/providers/local/limits-and-errors.md#engram-row-cache)) and uploaded per engram layer. The rows of a pass are requested at its entry and the rows of the next prefill chunk while a chunk runs, and `LocalModel::engram_stats()` reports the counters. A model split by `gpu_layers` opens one table for both halves.
- A pool must fit one device buffer (1 GiB). A compressed row is 640 floats, so a pool holds about 419,000 rows: `kv_tokens` up to about 419,000 for a ratio-1 owner and 838,000 for a ratio-2 one. A load whose pools do not fit fails naming `kv_tokens`.
- The window pools take 80 KB per cached position at the real shape (40 layers of 512 floats), for the window and one call's rows of each running sequence plus `kv_tokens - context_length` positions kept for prefix reuse. With the default `kv_tokens` of twice the context that is 2.6 GB at 32k context and 10 GB at 128k; `kv_tokens` equal to the context keeps them at about 0.35 GB per sequence.
- A tiered model prefills in steps of 4096 rows. The indexer scores them in tiles of at most 64 MB of scores, and a launch of the sparse attention whose partial results would pass 512 MiB (4096 rows at the real shape) takes the fused path, which is slower and gives the same values.
- Measured on the real checkpoint (RX 9070 XT, 16 GB): 10.0 GB of device memory after the load with host-tiered experts, 12.0 GB at the peak of a chat request, 13.0 GB for a 16-token pass with every position's logits. With `gpu_layers = 20` and a 3 GiB device tier the peak is 10.5 GB. The expert RAM tier (85 GiB in the runs) is host memory.
- With the set on an NVMe SSD (same card, `tiered:ram=90g,threads=32`, host-tiered): the process peaks at 11.7 GiB of device memory on a chat request and 12.5 GiB on a 1,752-token prompt, with 90 GiB of RSS at most. A first token after a 24-token prompt takes 26.7 s against 15.7 to 17.6 s on the CPU (the first pass of a Vulkan process is slower than later ones: 22 s against 10 s for the 16-token oracle prompt) and decode runs at 1.96 tokens/s; a 1,752-token prompt prefills at 11.8 tokens/s. In a separate 38-token run the first compilation and tuning of the kernels cost about 8 s once per process (7.3 s in the device readbacks of the first prefill pass, 1.2 s in the first one-row pass, against 0.08 s in later one-row passes), which is the likely source of the gap to the CPU's first token. The same request sent again in that process answered after 1.39 s, with its prompt prefix cached. With `gpu_layers = 20` the peak is 10.2 GiB at a 3 GiB device tier and 11.8 GiB at 6 GiB, and decode is 0.94 and 1.08 tokens/s; a device tier of 8 GiB runs out of device memory. Details are on [Local verification and known limits](@/providers/local/verification.md#deepseek-v4-1-real-checkpoint-on-an-ssd).
- Tensor parallelism is rejected (`tensor parallelism: hyper-connection residual streams ...`).
- `gpu_layers` is accepted only for cuts that keep every cache source with the layers that read it: each layer with compressed attention reads the compressed rows of its cache source layer, the picks of its index source layer and, when it narrows its scoring, the candidate blocks of the candidate source layer. The other values fail with `` gpu_layers N cuts `deepseek41` between layers that share compressed rows or indexer picks; legal values: ... ``. At the real layout (compressed rows owned by layers 2, 8, 14 and 20, layers 24 to 36 narrowing their scoring to the candidates of layer 20) the legal values are 20, 26, 32, 38, 39 and 40 of 40, and 0. The CPU half hands the GPU half the stream and the pre weights.
- Experts: `ExpertPlacement::Host` and `ExpertPlacement::Tiered` with `device_bytes = 0` (host-tiered: the experts stay in the file behind a RAM tier and the CPU computes the ones a pass needs) keep the attention, the router, the shared experts and the head on the GPU. A device tier works for a model whose experts fit it.

## Layer offload (`gpu_layers`)

`LocalModelConfig::gpu_layers = Some(n)` is llama.cpp's `-ngl`, for a model that does not fit the device. The last `n` layers run on the GPU together with the final norm and the LM head. The first layers, the token embeddings and Gemma 4's per-layer embeddings stay on the CPU, so the residual stream crosses to the device once per forward pass. `Some(0)` loads a plain CPU model and opens no device. `Some(n_layer)` loads a plain GPU model. `LocalModel::layer_split()` returns `(layers on the GPU, layers in all)`, and `None` for a model on one device.

- Each layer's K/V pool lives on its layer's device under the same block tables, and so does a hybrid model's recurrent state (a snapshot is the pair of the halves'). The batching scheduler is unchanged.
- A Gemma 4 layer that reads another layer's K/V from the other side of the boundary gets its source's new rows from the CPU half in the same crossing, into a mirror pool on the device that `kv_cache_bytes` does not count.
- With tiered experts the CPU layers read the RAM tier and the GPU layers the device tier, which has a RAM tier of its own. The `ram` budget is split by MoE layer count and `device` belongs to the GPU half.
- It fails with `CucaError::Config` for a CPU device, for `shards` or a `drafter`, and, once the file header is read, for a count above the layer count (`` gpu_layers N exceeds the M layers of `<architecture>` ``).

```bash,name=Linux and macOS: 17 of 35 layers of Gemma 4 E2B on the GPU
export CUCA_LOCAL_DEVICE="vulkan"
export CUCA_LOCAL_GPU_LAYERS="17"
cargo run --release --example local_model --features provider-local-vulkan
```

```powershell,name=Windows (PowerShell): 17 of 35 layers of Gemma 4 E2B on the GPU
$env:CUCA_LOCAL_DEVICE = "vulkan"
$env:CUCA_LOCAL_GPU_LAYERS = "17"
cargo run --release --example local_model --features provider-local-vulkan
```

```text,name=Expected line on stderr
loaded on Vulkan(Default): 17 of 35 layers on the GPU (layers 18 to 34), layers 0 to 17 and the embeddings on the CPU
```

## Host experts

`ExpertPlacement::Host { layers }` (`host`, `host:layers=20`) is llama.cpp's `--cpu-moe` and `--n-cpu-moe`. The routed experts stay in host RAM and the CPU computes them with the CPU engine's expert code, while attention, the router, the shared expert, the norms and the head run on the GPU. Per MoE layer the GPU routes, one readback brings the normed rows and the routing to the host, the CPU runs every selected expert over its own tokens in ascending expert order and sums them, and one upload returns the sum, which the GPU adds to the shared expert.

- `layers: None` hosts every MoE layer. `Some(n)` hosts the first `n`, and the other layers keep resident experts on the device.
- Every hosted expert is read into RAM at load.
- It composes with batching and with `gpu_layers`: the GPU layers host their experts and the CPU layers already compute theirs from RAM.
- It fails with `CucaError::Config` on a CPU device, with `shards`, and on a model without experts.

## ROCm on Windows

`provider-local-rocm` builds and runs on Windows on the RX 9070 XT (gfx1201). CubeCL loads `amdhip64.dll` and `hiprtc.dll` by those names at run time and compiles its kernels with its own LLVM, which reads the AMD device libraries (`ocml.bc`) from `HIP_DEVICE_LIB_PATH`. The Adrenalin driver (32.0.32015.12012) ships `amdhip64_7.dll` in `System32` and no `hiprtc`, so the HIP runtime has to come from AMD's ROCm 10.1.0 release.

The Windows tarball of that release needs no administrator rights to extract. The environment steps on AMD's install page use `setx ... /M`, which does need them; the per-session variables below replace those steps. The tarball is per GPU family, `gfx120X-all` covering gfx1200 and gfx1201, the RDNA4 cards; another card needs its own family, which was not tried.

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

The download (2.24 GB) took 29 s and the extraction 14 s. `hipInfo` lists `device# 0` as `AMD Radeon(TM) Graphics` (`gfx1036`, the Ryzen's integrated GPU) and `device# 1` as `AMD Radeon RX 9070 XT` (`gfx1201`), and `Test-Path` prints `True`. The DLLs carry the version in their names (`amdhip64_7.dll`, `hiprtc0716.dll`) in this release, and the tarball has no `amdhip64.dll` or `hiprtc.dll`, hence the copies.

```powershell,name=Windows (PowerShell): run on ROCm
$dll = "C:\TheRock\cuca-dlls"
$env:PATH = "$dll;" + $env:PATH
$env:HIP_PATH = $dll
$env:HIP_DEVICE_LIB_PATH = "C:\TheRock\build\lib\llvm\amdgcn\bitcode"
$env:CUCA_LOCAL_DEVICE = "rocm:1"
$env:CUCA_LOCAL_MODEL = "qwen3-0.6b-q8_0"
$env:CUCA_LOCAL_PROMPT = "Explain in two sentences what a mixture of experts model is."
$env:CUCA_LOCAL_THINKING = "off"
$env:RUSTFLAGS = "-C target-cpu=native"
cargo run --release --no-default-features --features provider-local-rocm --example local_model
```

The run prints `loaded on Rocm(1)` and then the reply. The variables are per shell: set them again in a new one.

- Keep the SDK's own `bin` folder off `PATH`, for the build as well as the run. It holds `hipconfig.exe`, which the build script of `cubecl-hip-sys` runs to pick its bindings by HIP version; that crate (7.14.6085001) bundles bindings up to HIP 7.14.60850 and the tarball reports 7.16.26385, so with the folder on `PATH` the build of `cubecl-hip` fails with 87 errors (unresolved `hipGraph_t`, `hipStream_t`, `hipError_t` and the like). Without `hipconfig` on `PATH` it takes its newest bundled bindings, and the 7.16 DLLs run them: every result below was obtained that way.
- The DLL folder must be on `PATH`, because `amdhip64.dll` imports `rocm_kpack.dll` from its own folder and the loader does not search there. Without it, or without the libraries, the load fails with `device Rocm(1): 0 such device(s) found`, whose message names `ROCM_PATH`, `HIP_PATH`, the `PATH` requirement and `HIP_DEVICE_LIB_PATH`.
- HIP numbers the integrated GPU 0 and the RX 9070 XT 1, so the device is `rocm:1`.
- The `rocm-sdk-core` 10.1.0 wheel (AMD's page gives `python -m pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ "rocm[libraries,device-all]==10.1.0"`; the test machine had it installed in Python's `site-packages`) holds the same files in `_rocm_sdk_core`. The copies and the `PATH` rule are the same, and the runs of the `qwen3-0.6b-q8_0` reply, the `deepseek41` generation and the production-path test below were made with its DLLs; the test suites, the oracle checks and the speed test with the tarball's. The nightly 7.14 wheel of the `gfx120X-all` index also ran the `qwen3-0.6b-q8_0` request with the same reply.
- Kernels allocate their outputs contiguously. HIP pads the rows of an ordinary tensor to the device's pitch, which the kernels' linear indexing ignores; before that change 21 of the 37 `model::gpu` tests failed with wrong values.
- `CUCA_LOCAL_DEVICE=rocm:1 cargo test --release --no-default-features --features provider-local-rocm --lib model::gpu -- --test-threads=1` passes 38 tests in 553 s (tarball DLLs); four of them, the real-model tests, return early without `CUCA_LOCAL_DS41_REAL`. `gpu_moe_one_row_experts_match_single_tokens` failed in 2 of 3 runs before its router was changed: the router rows came in groups of identical weights, so eight experts had exactly equal logits, and the last bit of a four-row and a one-row matmul (the kernel an autotune picks differs from run to run) decided which of them took the eighth slot, giving a different expert in the two passes. The rows are now scaled apart and the test passed in 4 of 4 runs.
- `qwen3-0.6b-q8_0`, thinking off, 23 tokens: the reply is byte for byte the Vulkan reply in every run. With the kernels compiled, decode runs at 27 to 29 tokens/s (24.3 on Vulkan).
- The oracle check of `deepseek41` (`real_model_on_the_gpu_matches_the_streaming_oracle_with_quantization_{off,on}`, host-tiered `ram=85g`, the 16-token prompt): quantizers off, top-1 equal at 15 of 16 positions, smallest top-10 overlap 8, KL 1.78e-7 at the last position, the pass 294 s; on, 13 of 16, overlap 6, KL 5.29e-8, 539 s. `real_model_on_the_gpu_production_paths_match_the_oracle` (`CUCA_LOCAL_DS41_REAL_PATHS=1`) runs the prompt as an inexact prefill, as a shorter prefill then one token per step, exactly, as last-position logits whole and in chunks of 4, with the context, sequence count and step rows of the engine (`CUCA_LOCAL_DS41_REAL_CTX=8192`, `_SEQS=4`, `_ROWS=4096`, `_QDQ=1`): every path passes the oracle's checks and the last-position paths pick the same top-1 token as the exact one.
- `deepseek41` from `examples/local_model.rs`, host-tiered `tiered:ram=85g,threads=32`, greedy: `What is the capital of France? Answer in one word.` with thinking off answers `Paris` (21 prompt tokens, 1 completion token), and `What is 9 * 8?` with thinking on reasons `We need answer simple. 9*8=72. Need concise.` and answers `72` (42 prompt tokens, 17 completion tokens), as on Vulkan. Peak device memory 10.0 GiB (11.7 GiB on Vulkan).
- With the experts in the RAM tier (`real_model_on_the_gpu_speed`, the 16-token prompt): ROCm takes 177.1 s for the cold prompt, 1.15 s warm (14.0 tokens/s), 3.45 s for the first decode step and a median of 0.202 s for a warm one (4.94 tokens/s); Vulkan takes 19.7 s, 1.08 s (14.8 tokens/s), 2.04 s and 0.192 s (5.20 tokens/s). Once the kernels are compiled the two backends run at the same speed.

### Open: the first block of a ROCm process

The first block of a ROCm process takes far longer than Vulkan's and varies from run to run; the cause is not found.

- `qwen3-0.6b-q8_0`, the same request in nine processes (two with the settings named below): first block after 63.7 to 362.7 s (63.7, 66.8, 84.6, 105.2, 126.5, 136.5, 233.8, 268.5, 362.7), then 28.4 or 29.3 tokens/s in two runs and 0.2 to 0.65 tokens/s in five (the rest not recorded). On Vulkan it is 8.4 s and 24.3 tokens/s.
- `deepseek41` with the 21-token prompt: first block after 282.6 and 683.8 s in two runs, 30.1 s on Vulkan with the same expert reads; the 42-token request 403.4 s, then 0.065 tokens/s. The disk is not the limit: reading the model file with a 16 MB buffer between the runs gave 3.2 and 3.4 GB/s.
- With `CUBECL_DEBUG_OPTION=profile` the first block of the `qwen3-0.6b-q8_0` request takes 9.4 s and decode 8.8 tokens/s; with `CUBECL_DEBUG_LOG=<file>` 9.8 and 9.9 s and 8.7 and 8.8 tokens/s (81 kernel compilations in the logged run). `CUBECL_AUTOTUNE_LEVEL=minimal` changed nothing (136.5 s), and CubeCL's persistent compilation cache (`[compilation] cache = true` in a `cubecl.toml`) did not help (66.8 s, then 362.7 s). Neither did a single stream (`[streaming] policy = "single"` and `max_streams = 1`): 437.4, 32.4 and 299.6 s in three runs.

## Measured speed

Machine: Ryzen 9 9950X3D2, RX 9070 XT with 15.9 GiB, Windows, release build with native CPU flags. Test: `perf_long_prompt` in `tests/provider_local.rs`, a prompt of about 1,100 tokens (1,108 for Gemma 4, 1,102 for Phi 4 mini) and 96 generated tokens, with the prefix cache bypassed so each round prefills the whole prompt. 14.6 GiB of the card was free before each run. Prefill is the time to the first token in rounds 1 and 2; round 0 compiles kernels. Decode is tokens per second after the first token. Ranges are the spread over the measured rounds. Adapter memory is the peak of the run.

| Model and mode | Decode, tokens/s | Prefill | Adapter memory, GiB |
|---|---|---|---|
| `mistral-small-4-119b-q4_0`, CPU only | 11.2 | 34.4 s (32 tokens/s) | 0 |
| the same, GPU with `host` experts | 8.6 to 8.8 | 22.3 to 23.9 s (47 to 50 tokens/s) | 6.6 |
| `qwen3.6-35b-a3b-q4_0`, CPU only | 18.8 to 18.9 | 23.8 to 24.4 s | 0 |
| the same, GPU with `host` experts | 10.1 to 10.2 | 19.6 to 21.8 s | 5.9 |
| the same, GPU with `tiered:device=6g,ram=12g` | 10.6 to 10.9 | 12.2 to 14.4 s | 10.6 |
| `muse-glimmer-30b-q4_0`, CPU only | 3.5 | 79.8 to 83.0 s | 0 |
| the same, `gpu_layers = 26` of 52 | 5.4 to 5.5 | 48 to 71 s | 13.5 |
| the same, `gpu_layers = 34` of 52 | 7.3 | 43.2 to 46.7 s | 13.5 |
| the same, `gpu_layers = 36` of 52 | 7.7 | 39.9 to 43.8 s | 14.1 |
| the same, `gpu_layers = 38` of 52 | 8.3 to 8.4 | 37.5 to 40.7 s | 14.2 |
| `gemma4-e2b-q4_0`, GPU only | 47.7 to 57.0 | 1.2 to 3.7 s | 5.4 |
| the same, `gpu_layers = 17` of 35 | 40.6 to 41.1 | 5.4 to 7.0 s | 4.3 |
| the same, CPU only | 27.4 to 27.6 | 10.7 s | 0 |
| `phi4-mini-q4_k_m`, GPU only (K-quant layers requantized to Q8_0) | 51.7 to 59.1 | 1.8 to 3.6 s | 11.1 |
| the same, CPU only | 19.5 to 19.7 | 9.5 s | 0 |

The rows for Mistral Small 4, Qwen3.6, and Muse at 26 and 34 layers ran with weight chunks of 1 GiB; the current chunk is 256 MiB and these rows were not repeated. The other GPU rows ran with 256 MiB chunks.

- The GPU wins every prefill. It wins decode on Muse 30B (2.4 times at 38 layers, and its prefill is 2.1 times faster) and on Gemma 4 E2B (1.7 to 2.1 times). It loses decode on the two large mixture-of-experts models, where the CPU engine computes the experts at full memory bandwidth and the GPU path pays a launch and a readback per MoE layer.
- Muse 30B decode speeds up with every layer moved to the GPU, by about 5.3 ms per layer between 26 and 38 layers (the CPU half sets the rate). 38 of 52 layers is the most that fits this card: 40, 42 and 44 fail with `out of device memory allocating 269484032 bytes`. 39 was not tried.
- On Gemma 4 E2B half the layers on the GPU are faster than the CPU alone and slower than the GPU alone: one hidden state crosses the boundary per pass and the CPU half still pays its per layer cost.

## Numerics

- GPU kernels keep activations in f32, while the CPU's quantized matmul rounds activations to Q8_0, so GPU and CPU logits differ slightly.
- A batched GPU pass equals single-token steps only up to float rounding, where the CPU is bit exact. A speculative greedy run on a GPU can therefore differ from plain decoding where two logits are that close.
- Phi 4 mini Q4_K_M on Vulkan (`phi4-mini-q4_k_m`, RX 9070 XT): the top-1 token equals the CPU's, and the relative logit error against the dequantized CPU reference is 1.8e-2, from the requantization to Q8_0.
- Against a dequantized CPU reference on tiny random models the tests accept 1e-3 absolute and 1e-2 relative for every architecture, and 1e-4 for `mistral4`, layer offload, host experts and sharded dense models (1e-3 for sharded mixtures of experts).
- Tiered and resident experts agree to float rounding on a GPU, and tiered placements agree with each other bit for bit; see [Output guarantee](@/providers/local/expert-tiers.md#output-guarantee).

## Verified and compile-checked

The GPU unit tests in `src/provider/local/model/gpu.rs` run only when `CUCA_LOCAL_DEVICE` names a GPU (`vulkan`, or `a+b` for two devices); otherwise they print `SKIP: set CUCA_LOCAL_DEVICE to a GPU device`. The live suites run on the device that variable names, the CPU by default. CI has no GPU, so every GPU run is manual, on the development machine: Vulkan on the RX 9070 XT under Windows. The repository records no per commit result for them.

Live runs on Vulkan, with the suite `tests/provider_local.rs` unless noted:

- Gemma 4 E2B Q4_0, OLMo 2 1B and Granite 3.3 2B passed every applicable scenario, with the same known limits as on the CPU.
- Phi 4 mini Q4_K_M passed (9 tests; the reasoning and tool scenarios skip because its template has neither), and so did Gemma 3 1B Q4_0.
- Qwen3-30B-A3B with `tiered:device=6g,ram=12g`, and Qwen3.6-35B-A3B (`qwen35moe`, hybrid) with `tiered:device=4g,ram=12g` and `tiered:device=6g,ram=12g`, passed every applicable scenario. `tests/provider_local_tiers.rs` reproduced the reference replies for OLMoE 1B-7B, Qwen3-30B-A3B and Qwen3.6-35B-A3B.
- Muse Glimmer 30B passed with `gpu_layers` of 34, 36 and 38 of 52, with the CPU run's categories (9 tests; thinking off skips for this model). The 16.03 GB file does not fit the card whole.
- Mistral Small 4 with `host` experts passed with the same categories as the CPU run.
- Tensor parallelism on `vulkan:0+vulkan:0` gave the single-device replies for Gemma 4 E2B Q4_0 and Qwen3-0.6B Q8_0 (15 of 15 each), at about a third of the speed because of the host-staged sum, and 8 concurrent requests gave the replies of the serial runs.
- The Gemma 4 assistant drafter produced output identical to plain decoding; see [Speculative decoding](@/providers/local/drafting.md#verified).

Unit tests on tiny random models:

- `gpu_matches_cpu_on_the_tiny_gemma4` and `gpu_matches_cpu_on_every_architecture` cover every `TinySpec` (Muse Glimmer included), the tiny `mistral4` models and the tiny `qwen35` and `qwen35moe` hybrid models, through prefill, decode, chunked prefill, the snapshot round trip and the cache fallback. `gpu_matches_cpu_on_other_weight_dtypes`, `gpu_paged_latent_decode_kernel_matches_cpu`, `dense_kernels_match_the_host_reference` and `moe_kernels_match_host_reference` cover the dtypes and the kernels. `gpu_moe_one_row_experts_match_single_tokens` runs a Qwen3-30B-A3B shaped layer over several rows and compares each with its single-token pass.
- `gpu_batched_matches_gpu_solo_on_every_architecture` runs three sequences in batched steps (a joint prefill, decode rows, a sequence sitting one step out, a prefill chunk next to decode rows, and a snapshot restored into another slot) and compares each with a solo run.
- `gpu_tiered_experts_match_gpu_resident`, `gpu_host_experts_match_gpu_resident`, `gpu_offload_matches_single_device_at_every_split` and `gpu_offload_serves_again_after_a_rejected_mirror_write` cover the placements.
- `gpu_sharded_matches_single_device`, `gpu_all_reduce_adds_in_shard_order` and `gpu_sharded_rejects_unsplittable_heads` cover tensor parallelism, and `gpu_drafter_matches_cpu_drafter` the Gemma 4 assistant.
- For DeepSeek-V4.1 the tests build tiny `deepseek41` models (Q8_0 experts, MXFP4 experts, and an indexer without candidate blocks): `gpu_matches_cpu_on_every_deepseek41_variant` against the CPU engine, `gpu_hyper_batched_and_prefix_reuse_match_solo`, `gpu_engram_prefetch_reads_each_row_once_and_changes_no_bit`, `gpu_offload_of_a_hyper_model_matches_single_device_at_every_legal_cut` for `gpu_layers`, and the tiered and host placements (`gpu_tiered_experts_match_gpu_resident`, `gpu_host_experts_match_gpu_resident`, `gpu_host_tiered_experts_match_gpu_host_experts`). `gpu_sharded_rejects_unsplittable_heads` includes the tensor parallelism rejection. The model-level comparisons run with the simulated quantization switched off on both sides; the quantization kernels have their own tests (`power_of_two_scales_are_bit_exact`, `e4m3_scale_matches_up_to_rare_division_steps`, `rows_do_not_depend_on_the_batch`).

Compile-checked only:

- Metal: compiled in the CI jobs on Linux that use the `ALL_NO_LLVM` feature list. No run on Apple hardware.
- CUDA and ROCm: the CI job `local_gpu_llvm` runs `cargo check` for both. ROCm also ran on Windows, see [ROCm on Windows](#rocm-on-windows). The NCCL all-reduce of tensor parallelism has never run.

Not run:

- Tensor parallelism over two distinct GPUs, and Muse Glimmer with `shards`.
- Models on a GPU other than those listed above. The `mistral3` architecture ran only as a tiny model, and the Ministral 3 presets have not run on a GPU. The remaining presets ran on the CPU.
- DeepSeek-V4.1 on a GPU faster than the CPU engine by more than about 1.2 times: the experts run on the CPU (see [DeepSeek-V4.1 on a GPU](#deepseek-v4-1-on-a-gpu)), so a decode step with the experts in RAM takes 0.41 s against 0.48 s, and a cold one is bound by the volume holding the experts.
