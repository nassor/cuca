+++
title = "Run the local engine on a GPU"
description = "LocalDevice, the GPU cargo features, device memory, numerics and what was verified on Vulkan, Metal, CUDA and ROCm."
template = "page.html"
weight = 7
+++

# Run the local engine on a GPU

<dl class="page-facts">
<dt>In one line</dt>
<dd><code>LocalModelConfig::device</code> moves the forward pass to Vulkan, Metal, CUDA or ROCm; nothing selects a GPU for you.</dd>
<dt>You need</dt>
<dd>One of <code>provider-local-vulkan</code>, <code>provider-local-metal</code>, <code>provider-local-cuda</code> or <code>provider-local-rocm</code>, and the system GPU driver.</dd>
<dt>Read this if</dt>
<dd>You want the engine on a GPU, or need the device text form, the memory cost or the verification status.</dd>
</dl>

## Devices and GPU features

The model runs on the CPU unless `LocalModelConfig::device` selects a GPU, and nothing selects a GPU for you. A GPU device runs the forward pass through burn 0.22 and CubeCL with the same math as the CPU engine. The engine thread owns the device. Every supported architecture except `qwen35` runs through the GPU forward pass.

| Feature | Backend | Extra build needs |
|---|---|---|
| `provider-local-gpu` | the shared GPU engine | selects no backend by itself |
| `provider-local-vulkan` | Vulkan through wgpu, SPIR-V shaders | C compiler; `rc.exe` on Windows |
| `provider-local-metal` | Metal through wgpu, MSL shaders | C compiler; `rc.exe` on Windows; runs on Apple only |
| `provider-local-cuda` | CUDA | prebuilt LLVM, downloaded on the first build |
| `provider-local-rocm` | ROCm (HIP) | the same LLVM download; Linux only |

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

```bash,name=Linux and macOS: run the example on a GPU
export CUCA_LOCAL_DEVICE="vulkan"
```

```powershell,name=Windows (PowerShell): run the example on a GPU
$env:CUCA_LOCAL_DEVICE = "vulkan"
```

```bash,name=Runs the same on all three platforms
cargo run --release --example local_model --features provider-local-vulkan
```

### Device selection

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

Any other text is a `CucaError::Config`: ``unknown local device `...` ``.

### Device memory

- Device memory is about the file size, minus any table the engine reads from the file instead of uploading (Gemma 4's per-layer embedding table), plus the K/V cache.
- The token embedding is uploaded once, as the LM head, only when it is tied to it. Embedding lookups read rows from the file on the host.
- The K/V cache lives in device memory with the same formula as on the CPU. The RoPE tables are on the device.
- A weight over 1 GiB is split into row chunks, because wgpu cannot allocate a single buffer that large.
- Dtype handling is in [Weight formats](@/providers/local/models.md#weight-formats).
- The first forward pass of each new input shape is slow while CubeCL autotunes and compiles kernels.

### Numerics

- GPU kernels keep activations in f32, while the CPU's quantized matmul rounds activations to Q8_0, so GPU and CPU logits differ slightly.
- A batched GPU pass equals single-token steps only up to float rounding. A speculative greedy run on a GPU can differ from plain decoding where two logits are that close.
- Measured on Vulkan with Phi 4 mini Q4_K_M (`phi4-mini-q4_k_m`, AMD RX 9070 XT, Windows): the top-1 token equals the CPU's, the relative logit error against the dequantized CPU reference is 1.8e-2, prefill runs at 97 tokens/s and decode at 42 tokens/s.
- Measured on Vulkan with Gemma 4 E2B Q4_0 (same machine, 1,300-token prompt, release build): prefill 73 tokens/s, decode 36 tokens/s.

### Verified and compile-checked

- Verified on Vulkan (AMD RX 9070 XT, Windows): the live suite `tests/provider_local.rs` ran for Gemma 4 E2B Q4_0, Llama 3.2 1B, Phi 3.5 mini, Gemma 2 2B, OLMo 2 1B and Granite 3.3 2B. Every applicable scenario passed, with the same known limits as on the CPU. The other presets ran on the CPU.
- The unit tests `gpu_matches_cpu_on_the_tiny_gemma4`, `gpu_matches_cpu_on_every_architecture` and `gpu_matches_cpu_on_other_weight_dtypes` compare the GPU engine with a CPU reference on tiny random models, and `gpu_drafter_matches_cpu_drafter` does the same for the drafter. They run only when `CUCA_LOCAL_DEVICE` names a GPU; otherwise they print `SKIP: set CUCA_LOCAL_DEVICE to a GPU device`.
- Compile-checked only: Metal (builds in the CI jobs that use `ALL_NO_LLVM`, which run on Linux), CUDA and ROCm (the CI job `local_gpu_llvm` runs `cargo check` for both).
