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

The model runs on the CPU unless `LocalModelConfig::device` selects a GPU. A GPU device runs the forward pass through burn 0.22 and CubeCL, and the engine thread owns the device. Every supported architecture runs through the GPU forward pass. The in-trunk head, the EAGLE drafter and the DFlash drafter are CPU only; the Gemma 4 assistant runs on the GPU.

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
- a full-width Q/K norm (OLMo 2, OLMoE);
- an `n_head` not divisible by the device count, or K/V heads that neither divide nor replicate;
- a query width per device that is not a multiple of 32, or a dense FFN, expert or shared expert width that does not divide into multiples of 32 per device.

## Device memory

- Q4_0, Q4_1 and Q8_0 matmul weights are repacked at load and stay quantized on the device, one copy per weight.
- F32, F16 and BF16 matmul weights are held as f32 (4 bytes per element, so an F16 or BF16 weight doubles). Norm vectors, biases and scalars are f32.
- Every other quantized dtype (Q5_0, Q5_1, Q8_1 and the K-quants Q2_K to Q6_K) is requantized to Q8_0 at load, about 1.06 bytes per element. A 3.8 B parameter Q4_K_M file needs about 4 GB on the device rather than 15 GB. Each element moves by at most half a Q8_0 step, `block_max / 254`, and Q8_0 resolves a block with more levels than Q4_K, Q5_K or Q6_K do, so the extra error is below the one the file already carries.
- A weight over 256 MiB is split into row chunks, so no weight buffer exceeds 256 MiB and no buffer at all exceeds 1 GiB. The chunk size keeps burn's memory pages small, which lets a nearly full card place the next page.
- The token embedding is uploaded once, as the LM head, only when it is tied to it. Embedding lookups read rows from the file on the host, and Gemma 4's per-layer embedding table is never uploaded.
- The K/V cache lives on the device as the same block pools as on the CPU: f32, twice `context_length` positions by default, in blocks of 16. The RoPE tables are on the device.
- A mixture of experts keeps its experts in the MoE kernels' layout: the file's bytes for Q4_0, Q4_1 and Q8_0 experts, Q8_0 for other quantized dtypes, f32 for float dtypes. `ExpertPlacement::Tiered` reads no expert at load; see [Tiered expert placement](@/providers/local/expert-tiers.md).
- Dropping the last clone of a `LocalModel` on a GPU returns its device memory to the driver, so another model can load in the same process.
- The first forward pass of each new input shape is slow while CubeCL autotunes and compiles kernels. Continuous batching adds shapes: every distinct number of rows in a step pays this once per process. A current figure for that first request is not recorded.

Dtype handling on the CPU is on [Weight formats](@/providers/local/models.md#weight-formats).

## Kernels

- Dense weights (attention projections, dense FFNs, the LM head) run through two CubeCL kernels of this crate. Up to 64 input rows (decode, batched decode, verification, prompt tails) one matrix-vector kernel dots the blocks with the input in f32, with several threads per row for weights that have few rows. More rows (prefill) dequantize slices of the weight to f32 and use burn's float matmul. burn's quantized matmul is not used by any weight.
- A mixture of experts runs four kernel dispatches per layer: softmax and top-k of the router logits, `act(gate . x) * (up . x)` for every (token, expert) pair, the down projections weighted by the routing, and a sum over each token's experts in expert order. A resident layer never reads the router output back, so a resident token queues all its layers without waiting for the host. A tiered layer reads back the router output (ids and weights, a few hundred floats) to find the experts its pool lacks.
- Attention over one query row (a decode step, every drafter layer) reads the paged K/V in place with one kernel, a single launch per layer when every sequence in the step decodes one row. Several rows (prefill, verification) gather the visible blocks and use burn's matmul attention with a mask. A `mistral4` layer caches one shared `[latent 256 | rotary 64]` row per position and no value pool: the key up-projection is absorbed into the query and the value up-projection applies to the attended latent, so no per-head K or V is built.
- A gated delta-net layer (Qwen 3.5) runs its recurrence token by token on the device. Its recurrent state is a set of immutable device tensors, so a snapshot costs nothing. The GPU keeps no verify record for a hybrid model, and no drafter pairs with one on a GPU.
- On Vulkan, single-token decode is bound by the host: each queued operation costs about 10 microseconds of host time.

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

Compile-checked only:

- Metal: compiled in the CI jobs on Linux that use the `ALL_NO_LLVM` feature list. No run on Apple hardware.
- CUDA and ROCm: the CI job `local_gpu_llvm` runs `cargo check` for both. The NCCL all-reduce of tensor parallelism has never run.

Not run:

- Tensor parallelism over two distinct GPUs, and Muse Glimmer with `shards`.
- Models on a GPU other than those listed above. The `mistral3` architecture ran only as a tiny model, and the Ministral 3 presets have not run on a GPU. The remaining presets ran on the CPU.
