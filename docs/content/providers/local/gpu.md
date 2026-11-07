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

### Tensor parallelism

`LocalModelConfig::shards` splits one model over several GPUs, as vLLM's tensor parallelism does. The model runs on `1 + shards.len()` devices (`device` is the first); the tests and the example read a `+`-separated list from `CUCA_LOCAL_DEVICE` (`cuda:0+cuda:1`, `vulkan:0+vulkan:1`), and `LocalModelConfig::set_devices` parses the same text. A device may repeat: `vulkan:0+vulkan:0` splits the model in two on one GPU. That gives no speedup, and holds the replicated parts twice, but runs the whole path, which is how it is tested on one GPU.

- Each layer's attention heads are split evenly, with the K/V heads they read. A model with fewer K/V heads than devices keeps a copy of each K/V head on every device that needs it (Gemma 4 E2B has one K/V head, so with two devices each holds it whole). The FFN width is split evenly, a mixture of experts inside every expert (each device holds all experts, each cut to its share of the width, with the router replicated) and the shared expert the same way.
- Each device holds its slice of the weights and its own K/V pools. Norms, embeddings, Gemma 4's per-layer inputs and the RoPE tables are replicated; only the first device holds the LM head and runs the final norm. `LocalModel::kv_cache_bytes` sums the pools of all devices, which exceeds the single-device number when K/V heads are replicated.
- The partial outputs are summed twice per layer, after the attention output projection and after the FFN down projection. On CUDA (Linux, every device a distinct GPU) the sum is burn's NCCL `all_reduce`; everywhere else it is host-staged: every partial is read back, added in f32 in device order and uploaded to every device. The order is fixed, so repeated runs give the same bits. The host round trip is two per layer and makes a split decode slower than one device when the model fits on one; the point is a model that does not.
- A split model agrees with the single-device GPU model to float rounding, not bit for bit (the two half-width partial sums add in another order than one full-width matmul).
- `LocalModel::load` rejects, with `CucaError::Config` (`tensor parallelism: <rule> (<numbers>)`): a CPU device or shards of another backend than `device`, a `drafter`, `ExpertPlacement::Tiered`, and a model that cannot be split: `n_head` not divisible by the device count, K/V heads that neither divide nor replicate, a query width per device that is not a multiple of 32, an FFN (or expert) width that does not divide into multiples of 32 per device, a gated delta-net layer (Qwen 3.5 and its MoE variant) or a full-width Q/K norm (OLMo 2, OLMoE).
- Continuous batching and prefix reuse work as on one device.

### Device memory

- Device memory is about the file size, minus any table the engine reads from the file instead of uploading (Gemma 4's per-layer embedding table), plus the K/V cache.
- A mixture-of-experts model can keep its experts in the GGUF file with `ExpertPlacement::Tiered`: `device_bytes` bounds the device memory they use and an optional RAM tier (`ram_bytes`) holds packed copies. See [Tiered expert placement](@/providers/local/expert-tiers.md).
- The token embedding is uploaded once, as the LM head, only when it is tied to it. Embedding lookups read rows from the file on the host.
- The K/V cache lives in device memory as the same block pools as on the CPU (twice `context_length` positions, in blocks of 16); the decode step reads it in place with a CubeCL kernel and a prefill gathers the blocks it needs. The RoPE tables are on the device.
- A weight over 1 GiB is split into row chunks, because wgpu cannot allocate a single buffer that large.
- Dtype handling is in [Weight formats](@/providers/local/models.md#weight-formats).
- Dropping the last clone of a `LocalModel` on a GPU returns its device memory to the driver, so another model can load in the same process.
- The first forward pass of each new input shape is slow while CubeCL autotunes and compiles kernels. Continuous batching adds shapes: every distinct number of rows in a step (one to `max_sequences` decode rows, and each prefill chunk size) pays this once per process. Measured on Gemma 4 E2B Q4_0 on Vulkan: the first 1,108-token prompt takes 37 to 42 s to its first token, the first warm request (a 4-row chunk) about 2 s, and every request after that 0.06 s, the same with and without batching.

### Layer offload (`gpu_layers`)

`LocalModelConfig::gpu_layers = Some(n)` (`CUCA_LOCAL_GPU_LAYERS=n` in the tests and the example) is llama.cpp's `-ngl`, for a model that does not fit the device. The **last** `n` layers run on the GPU together with the final norm and the LM head; the first layers, the token embeddings and Gemma 4's per-layer embeddings stay on the CPU, so the residual stream crosses to the device once per forward pass. `Some(0)` loads a plain CPU model (no device opens) and `Some(n_layer)` a plain GPU model; more than the layer count fails with a `CucaError::Config`. The head always follows the last layer (llama.cpp counts it as one more layer and keeps it on the CPU at `n = n_layer`).

Each layer's K/V pool lives on its layer's device under the same block tables, a hybrid model's recurrent state likewise (a snapshot is the pair of the halves'), and the batching scheduler is unchanged. A Gemma 4 layer that reads another layer's K/V from the other side of the boundary gets the new rows of its source from the CPU half in the same crossing (a mirror pool on the device, not counted in `kv_cache_bytes`). With tiered experts the CPU layers read the RAM tier and the GPU layers the device tier; the `ram` budget is split by MoE layer count. Tensor parallelism and a drafter are rejected with a `CucaError::Config`. Verified by `gpu_offload_matches_single_device_at_every_split` (every architecture, every split point, 1e-4 against the GPU-only load, the CPU half held dequantized) and `gpu_offload_serves_again_after_a_rejected_mirror_write`.

### Host experts (`CUCA_LOCAL_EXPERTS=host`)

`ExpertPlacement::Host { layers }` (`host`, `host:layers=20`) is llama.cpp's `--cpu-moe` / `--n-cpu-moe`: the routed experts stay in host RAM and the CPU computes them with the CPU engine's expert code, while attention, the router, the shared expert, the norms and the head run on the GPU. Per MoE layer the GPU routes, one readback brings the normed rows and the routing to the host, the CPU runs every selected expert over its own tokens in ascending expert order and sums them, and one upload returns the sum, which the GPU adds to the shared expert. `layers: None` hosts every MoE layer, `Some(n)` the first `n` (the other layers keep resident experts on the device). It needs a GPU device and a model with experts, reads every hosted expert into RAM at load, composes with batching and with `gpu_layers` (the GPU layers host theirs; the CPU layers already compute theirs from RAM), and cannot be combined with `tiered` or `shards`. Verified by `gpu_host_experts_match_gpu_resident` (1e-4 against resident experts).

Measured on this machine (Ryzen 9 9950X3D2, RX 9070 XT, Windows, native release build, `perf_long_prompt`, a 1,100-token prompt and 96 generated tokens, runs back to back on a quiet machine with 14-14.7 GiB of the 15.9 GiB card free):

| Model and mode | Decode, tokens/s | Prefill of the whole prompt, no prefix hit (after the first run) | Adapter memory in use at the peak, GiB (1.2-1.9 before the run) |
|---|---|---|---|
| `mistral-small-4-119b-q4_0`, CPU only | 11.2-11.4 | 33.5-33.8 s (33 tokens/s) | 0 |
| the same, GPU with `host` experts | 8.0-8.3 | 22.0-28.3 s (39-51 tokens/s) | 9.9 |
| `qwen3.6-35b-a3b-q4_0`, CPU only | 18.2-18.8 | (first run 24.3 s) | 0 |
| the same, GPU with `host` experts | 9.3-9.9 | (first run 50.7 s) | 10.7 |
| the same, GPU with `tiered:device=6g,ram=12g` | 11.0-11.2 | (first run 33.4 s) | 14.0 |
| `muse-glimmer-30b-q4_0`, CPU only | 3.6 | | 0 |
| the same, `gpu_layers = 26` of 52 | 3.0-3.1 | | 14.4 |
| `gemma4-e2b-q4_0`, GPU only | 30.6 | | 6.0 |
| the same, `gpu_layers = 17` of 35 | 22.3 | | 5.2 |
| the same, CPU only | 27.6 | | 0 |

The GPU modes therefore win only the Mistral Small 4 prefill; they lose decode everywhere on this machine, and none of them is a speedup for decode:

- **Host experts.** A decode layer pass costs about 1.4 ms of readback (it waits for the layer's queued device work: attention, router and shared expert are launch-bound), 1.23 ms of CPU experts and 0.01 ms of upload (`CUCA_LOCAL_PROFILE_HOST=1` prints these), about 3 ms per MoE layer, 100 ms over the 36 layers, where the CPU engine runs a whole token in 89 ms. The launch latency and the readback of every layer are a floor the GPU cannot hide. Overlapping the shared expert with the CPU experts by forcing a submit (`Device::flush`) made decode 20% slower (6.4 tokens/s) and was not kept. Prefill is compute-bound and the GPU wins it. The matrix of `mistral-small-4-119b-q4_0` with host experts passes with the same categories as the CPU run.
- **Layer offload of Muse 30B.** At 26 layers a decode step costs 138.5 ms on the CPU half (5.3 ms per layer, the CPU-only rate) and 202 ms on the GPU half (26 layers, the head and the readback), 7.8 ms per layer where a Q4_0 layer of 290 MB should need about 0.6 ms of device bandwidth. The kernel profile (`CUBECL_DEBUG_OPTION=profile`) shows the quantized matmuls of this file running as a dequantize kernel followed by a dense matmul (about 200-270 us per matmul plus a 74 us dequantize), and `CUCA_LOCAL_PROFILE_SPLIT=1` shows that 14 of the 209 weights from layer 26 on are held as f32 by the mixed-scheme rule (1.42 GiB where 0.38 GiB would be quantized). The device half is the bottleneck; a native fused quantized matrix-vector kernel for dense layers would be the fix and is not implemented. The full matrix at 26 layers passes with the CPU run's categories.
- **Layer offload of Gemma 4 E2B.** Half the layers on the GPU costs 27% of full-GPU decode, and the CPU alone is faster than either on this machine.

### Numerics

- GPU kernels keep activations in f32, while the CPU's quantized matmul rounds activations to Q8_0, so GPU and CPU logits differ slightly.
- A batched GPU pass equals single-token steps only up to float rounding. A speculative greedy run on a GPU can differ from plain decoding where two logits are that close.
- Measured on Vulkan with Phi 4 mini Q4_K_M (`phi4-mini-q4_k_m`, AMD RX 9070 XT, Windows): the top-1 token equals the CPU's, the relative logit error against the dequantized CPU reference is 1.8e-2, prefill runs at 97 tokens/s and decode at 42 tokens/s.
- Measured on Vulkan with Gemma 4 E2B Q4_0 (same machine, 1,300-token prompt, release build): prefill 73 tokens/s, decode 36 tokens/s.

### Verified and compile-checked

- Verified on Vulkan (AMD RX 9070 XT, Windows): the live suite `tests/provider_local.rs` ran for Gemma 4 E2B Q4_0, OLMo 2 1B and Granite 3.3 2B, for Qwen3-30B-A3B with tiered experts (`tiered:device=6g,ram=12g`) and for Qwen3.6-35B-A3B (`qwen35moe`, hybrid) with tiered experts at `tiered:device=4g,ram=12g` and `tiered:device=6g,ram=12g`. Every applicable scenario passed, with the same known limits as on the CPU. The other presets ran on the CPU.
- The unit tests `gpu_matches_cpu_on_the_tiny_gemma4`, `gpu_matches_cpu_on_every_architecture` (every `TinySpec`, `muse-glimmer` included, and the tiny `qwen35` and `qwen35moe` hybrid models, through prefill, decode, chunked prefill, the snapshot round trip and the cache fallback) and `gpu_matches_cpu_on_other_weight_dtypes` compare the GPU engine with a CPU reference on tiny random models, and `gpu_drafter_matches_cpu_drafter` does the same for the drafter. `moe_kernels_match_host_reference` checks the MoE kernels against a host evaluation, and `gpu_moe_one_row_experts_match_single_tokens` runs a Qwen3-30B-A3B-shaped MoE layer over a pass of several rows and compares every row with its single-token pass. They run only when `CUCA_LOCAL_DEVICE` names a GPU; otherwise they print `SKIP: set CUCA_LOCAL_DEVICE to a GPU device`.
- `muse-glimmer` on a GPU: verified on tiny random models only (32 `model::gpu` tests pass on Vulkan, among them the every-architecture comparison with the CPU). The 30B preset (16.03 GB of weights) does not fit the 16 GB card of the verification machine, so it has not run on a GPU; it should load on a device with room for the weights plus the K/V cache, or split over GPUs with `shards`, and that is untested.
- `mistral3` on a GPU: the dense Llama math plus a per-row query scale (one `[1, 1, t, 1]` tensor per pass, built only when a position is past the original context); verified on the tiny `mistral3` model, whose positions cross the scaling length, by the every-architecture comparison with the CPU. The 3B preset has not been run on a GPU.
- `mistral4` (Mistral Small 4) runs on a GPU: multi-head latent attention keeps one shared `[latent 256 | rotary 64]` row per position (a pool of `[blocks, 1, 16, 320]`, no value pool: the value is the first 256 columns of the key row), the per-head key up-projection is absorbed into the query by one batched matmul and the value up-projection is applied to the attended latent afterwards, so no per-head K or V is ever built. A single query row runs the paged decode kernel with a 320-wide key and a 256-wide value over the one shared row; several rows gather the visible rows and use burn's matmul attention. The fused `ffn_gate_up_exps` tensor, the ungated shared expert, `expert_weights_scale` (in the route kernel), YaRN and the Llama 4 query scaling are implemented. Verified by the every-architecture, batched, tiered and layer-offload unit tests (tiny models, 1e-4 against the dequantized CPU reference) and `gpu_paged_latent_decode_kernel_matches_cpu`. Tensor parallelism rejects it with a `CucaError::Config`.
- Continuous batching on a GPU: `gpu_batched_matches_gpu_solo_on_every_architecture` runs three sequences (a joint prefill, decode rows, a sequence sitting one step out, a prefill chunk next to decode rows, and a snapshot of a hybrid model restored into another slot) in batched steps and compares each with a solo run. On a GPU the rows of a batched step agree with solo steps up to float rounding, not bit for bit as on the CPU; when several sequences each decode one row, one attention launch per layer serves all of them.
- Tiered experts on a GPU: `gpu_tiered_experts_match_gpu_resident` checks them on tiny models, and `tests/provider_local_tiers.rs` reproduced the reference replies on Vulkan for OLMoE 1B-7B and Qwen3-30B-A3B; see [Output guarantee](@/providers/local/expert-tiers.md#output-guarantee).
- Hybrid models on a GPU: a gated delta-net layer runs its recurrence token by token on the device (a convolution over the history, then per value head a decay, a read, a write and a read of an `[head_k_dim, head_v_dim]` state; the key heads are tiled across the value heads). The recurrent state is a set of immutable device tensors, so a snapshot costs nothing; `truncate_cache` retains the end of the cache, else 0 but the GPU keeps no verify record, because no drafter pairs with a hybrid model there. A single token queues about half the operations of a straight transcription of the recurrence (the host spends about 10 µs per queued operation, and decode is host-bound: one router readback per MoE layer costs about 1 ms whatever it carries, and each submission of 32 kernel launches costs the host a few hundred µs). Quantized matmuls stay fast only while burn fuses them with their elementwise neighbours; a slice of a matmul output (stacking gate and up into one matrix and narrowing the result) or a cached transposed weight view made the autotuner pick a kernel 15 times slower. A scan waits for the device every 16 tokens so a 256-token prefill chunk does not queue hundreds of megabytes of temporaries. The shared expert of `qwen35moe` runs as a dense gated FFN scaled by a sigmoid gate computed from an elementwise product and a row sum (no matmul).
- Mixture-of-experts layers run in CubeCL kernels of this crate (`src/provider/local/model/moe_kernel.rs`, written in safe Rust with the checked `#[cube(launch)]` launch and attached through burn's `backend_extension`, whose `Fusion` glue queues them as opaque operations), because burn cannot gather the weights of a data-dependent subset of quantized experts (`q_select` dequantizes). Each layer keeps its experts as slots of one `u32` tensor (the pool) in the kernels' own layout: Q4_0, Q4_1, Q8_0 and f32 are read as they are (as many bytes as the file), every other dtype is requantized to Q8_0. Four dispatches replace the per-expert matmul triples: softmax and top-k of the router logits (a cube per token), `act(gate . x) * (up . x)` for every (token, expert) pair, the routing-weighted down projections into a per-pair scratch, and a sum over each token's experts in expert order. A resident layer never reads the router output back, so a resident MoE token queues all its layers without a host wait; a tiered layer reads back the router output (ids and weights, a few hundred floats) to find the experts its pool lacks, writes them into free or victim slots with a one-slot kernel and computes in groups of at most `device_slots` experts, which changes no bit of the result. Because the kernels do not go through burn's matmul autotune, the Q4/Q8 key hazard does not apply to experts: `plan_dense` ignores the stacked expert tensors, and the 5 Q4_1 down layers of `qwen3.6-35b-a3b-q4_0` no longer cost 4.2 MB per expert as f32. The router matmul, the shared expert and the other layers stay burn operations. The test `moe_kernels_match_host_reference` compares every kernel and the grouping against a host evaluation on tiny shapes.
- Compile-checked only: Metal (builds in the CI jobs that use `ALL_NO_LLVM`, which run on Linux), CUDA and ROCm (the CI job `local_gpu_llvm` runs `cargo check` for both).
