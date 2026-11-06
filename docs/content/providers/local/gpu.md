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

The model runs on the CPU unless `LocalModelConfig::device` selects a GPU, and nothing selects a GPU for you. A GPU device runs the forward pass through burn 0.22 and CubeCL with the same math as the CPU engine. The engine thread owns the device. Every supported architecture runs through the GPU forward pass; only the in-trunk prediction head of a `*-MTP` file is CPU only.

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
- A mixture-of-experts model can keep its experts in the GGUF file with `ExpertPlacement::Tiered`: `device_bytes` bounds the device memory they use and an optional RAM tier (`ram_bytes`) holds packed copies. See [Tiered expert placement](@/providers/local/expert-tiers.md).
- The token embedding is uploaded once, as the LM head, only when it is tied to it. Embedding lookups read rows from the file on the host.
- The K/V cache lives in device memory as the same block pools as on the CPU (twice `context_length` positions, in blocks of 16); the decode step reads it in place with a CubeCL kernel and a prefill gathers the blocks it needs. The RoPE tables are on the device.
- A weight over 1 GiB is split into row chunks, because wgpu cannot allocate a single buffer that large.
- Dtype handling is in [Weight formats](@/providers/local/models.md#weight-formats).
- Dropping the last clone of a `LocalModel` on a GPU returns its device memory to the driver, so another model can load in the same process.
- The first forward pass of each new input shape is slow while CubeCL autotunes and compiles kernels. Continuous batching adds shapes: every distinct number of rows in a step (one to `max_sequences` decode rows, and each prefill chunk size) pays this once per process. Measured on Gemma 4 E2B Q4_0 on Vulkan: the first 1,108-token prompt takes 37 to 42 s to its first token, the first warm request (a 4-row chunk) about 2 s, and every request after that 0.06 s, the same with and without batching.

### Numerics

- GPU kernels keep activations in f32, while the CPU's quantized matmul rounds activations to Q8_0, so GPU and CPU logits differ slightly.
- A batched GPU pass equals single-token steps only up to float rounding. A speculative greedy run on a GPU can differ from plain decoding where two logits are that close.
- Measured on Vulkan with Phi 4 mini Q4_K_M (`phi4-mini-q4_k_m`, AMD RX 9070 XT, Windows): the top-1 token equals the CPU's, the relative logit error against the dequantized CPU reference is 1.8e-2, prefill runs at 97 tokens/s and decode at 42 tokens/s.
- Measured on Vulkan with Gemma 4 E2B Q4_0 (same machine, 1,300-token prompt, release build): prefill 73 tokens/s, decode 36 tokens/s.

### Verified and compile-checked

- Verified on Vulkan (AMD RX 9070 XT, Windows): the live suite `tests/provider_local.rs` ran for Gemma 4 E2B Q4_0, Llama 3.2 1B, Phi 3.5 mini, Gemma 2 2B, OLMo 2 1B and Granite 3.3 2B, for Qwen3-30B-A3B with tiered experts (`tiered:device=6g,ram=12g`) and for Qwen3.6-35B-A3B (`qwen35moe`, hybrid) with tiered experts at `tiered:device=4g,ram=12g` and `tiered:device=6g,ram=12g`. Every applicable scenario passed, with the same known limits as on the CPU. The other presets ran on the CPU.
- The unit tests `gpu_matches_cpu_on_the_tiny_gemma4`, `gpu_matches_cpu_on_every_architecture` (every `TinySpec`, and the tiny `qwen35` and `qwen35moe` hybrid models, through prefill, decode, chunked prefill, the snapshot round trip and the cache fallback) and `gpu_matches_cpu_on_other_weight_dtypes` compare the GPU engine with a CPU reference on tiny random models, and `gpu_drafter_matches_cpu_drafter` does the same for the drafter. `moe_kernels_match_host_reference` checks the MoE kernels against a host evaluation, and `gpu_moe_one_row_experts_match_single_tokens` runs a Qwen3-30B-A3B-shaped MoE layer over a pass of several rows and compares every row with its single-token pass. They run only when `CUCA_LOCAL_DEVICE` names a GPU; otherwise they print `SKIP: set CUCA_LOCAL_DEVICE to a GPU device`.
- Continuous batching on a GPU: `gpu_batched_matches_gpu_solo_on_every_architecture` runs three sequences (a joint prefill, decode rows, a sequence sitting one step out, a prefill chunk next to decode rows, and a snapshot of a hybrid model restored into another slot) in batched steps and compares each with a solo run. On a GPU the rows of a batched step agree with solo steps up to float rounding, not bit for bit as on the CPU; when several sequences each decode one row, one attention launch per layer serves all of them.
- Tiered experts on a GPU: `gpu_tiered_experts_match_gpu_resident` checks them on tiny models, and `tests/provider_local_tiers.rs` reproduced the reference replies on Vulkan for OLMoE 1B-7B and Qwen3-30B-A3B; see [Output guarantee](@/providers/local/expert-tiers.md#output-guarantee).
- Hybrid models on a GPU: a gated delta-net layer runs its recurrence token by token on the device (a convolution over the history, then per value head a decay, a read, a write and a read of an `[head_k_dim, head_v_dim]` state; the key heads are tiled across the value heads). The recurrent state is a set of immutable device tensors, so a snapshot costs nothing; `truncate_cache` retains the end of the cache, else 0 but the GPU keeps no verify record, because no drafter pairs with a hybrid model there. A single token queues about half the operations of a straight transcription of the recurrence (the host spends about 10 µs per queued operation, and decode is host-bound: one router readback per MoE layer costs about 1 ms whatever it carries, and each submission of 32 kernel launches costs the host a few hundred µs). Quantized matmuls stay fast only while burn fuses them with their elementwise neighbours; a slice of a matmul output (stacking gate and up into one matrix and narrowing the result) or a cached transposed weight view made the autotuner pick a kernel 15 times slower. A scan waits for the device every 16 tokens so a 256-token prefill chunk does not queue hundreds of megabytes of temporaries. The shared expert of `qwen35moe` runs as a dense gated FFN scaled by a sigmoid gate computed from an elementwise product and a row sum (no matmul).
- Mixture-of-experts layers run in CubeCL kernels of this crate (`src/provider/local/model/moe_kernel.rs`, written in safe Rust with the checked `#[cube(launch)]` launch and attached through burn's `backend_extension`, whose `Fusion` glue queues them as opaque operations), because burn cannot gather the weights of a data-dependent subset of quantized experts (`q_select` dequantizes). Each layer keeps its experts as slots of one `u32` tensor (the pool) in the kernels' own layout: Q4_0, Q4_1, Q8_0 and f32 are read as they are (as many bytes as the file), every other dtype is requantized to Q8_0. Four dispatches replace the per-expert matmul triples: softmax and top-k of the router logits (a cube per token), `act(gate . x) * (up . x)` for every (token, expert) pair, the routing-weighted down projections into a per-pair scratch, and a sum over each token's experts in expert order. A resident layer never reads the router output back, so a resident MoE token queues all its layers without a host wait; a tiered layer reads back the router output (ids and weights, a few hundred floats) to find the experts its pool lacks, writes them into free or victim slots with a one-slot kernel and computes in groups of at most `device_slots` experts, which changes no bit of the result. Because the kernels do not go through burn's matmul autotune, the Q4/Q8 key hazard does not apply to experts: `plan_dense` ignores the stacked expert tensors, and the 5 Q4_1 down layers of `qwen3.6-35b-a3b-q4_0` no longer cost 4.2 MB per expert as f32. The router matmul, the shared expert and the other layers stay burn operations. The test `moe_kernels_match_host_reference` compares every kernel and the grouping against a host evaluation on tiny shapes.
- Compile-checked only: Metal (builds in the CI jobs that use `ALL_NO_LLVM`, which run on Linux), CUDA and ROCm (the CI job `local_gpu_llvm` runs `cargo check` for both).
