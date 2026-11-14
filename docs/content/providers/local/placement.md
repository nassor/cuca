+++
title = "Hybrid placement"
description = "How one local model spreads over VRAM, RAM and the SSD, and over the GPU and the CPU: the keys that place each part, the combinations that load, and measured choices."
template = "page.html"
weight = 11
+++

# Hybrid placement

<dl class="page-facts">
<dt>In one line</dt>
<dd><code>device</code>, <code>gpu_layers</code>, <code>experts</code> and <code>engram</code> decide which parts of a model sit in VRAM, in RAM or only in the GGUF file on disk, and whether the GPU or the CPU computes them.</dd>
<dt>You need</dt>
<dd><code>provider-local</code> for the CPU placements, a GPU feature such as <code>provider-local-vulkan</code> for the others, and a mixture-of-experts model for the <code>experts</code> placements.</dd>
<dt>Read this if</dt>
<dd>A model does not fit your GPU, or does not fit your RAM, and you are choosing how to split it.</dd>
</dl>

The engine has two processors and three places to keep a weight. The GPU computes from VRAM, the CPU computes from RAM, and the GGUF file on disk holds every tensor. A resident part is read into its memory at load; a part behind a cache is read from the file while the model runs.

## Where each part lives

<div class="dgm">
<div class="dgm-scroll">
<svg viewBox="10 12 660 296" role="img" aria-labelledby="hyb-title hyb-desc">
  <title id="hyb-title">One local model spread over VRAM, RAM and the SSD, computed by the GPU and the CPU</title>
  <desc id="hyb-desc">Two processors sit above two memories: the GPU above VRAM and the CPU above RAM. The GGUF file on the SSD sits beside them with no processor. VRAM holds the last gpu_layers layers, attention, the router and the head, the expert device tier, and the K/V of the GPU layers. RAM holds the first layers and the embeddings, host experts, the expert RAM tier, the Engram row cache, and the K/V of the CPU layers. The file holds every tensor, the experts no tier holds, the Engram table, and split parts on any volume. Reader threads read from the file into RAM, and uploads move experts from RAM to VRAM. The hidden state crosses between the GPU and the CPU at the gpu_layers cut and for host experts.</desc>
  <defs>
    <marker id="hyb-arrow" viewBox="0 0 8 8" refX="7" refY="4"
            markerWidth="6" markerHeight="6" orient="auto">
      <path d="M0 0.6 L7 4 L0 7.4 z" fill="var(--muted-foreground)"/>
    </marker>
  </defs>
  <text class="t-title" x="20" y="32">One model over VRAM, RAM and the SSD</text>
  <rect class="blk blk-ctl" x="20" y="52" width="170" height="44" rx="6"/>
  <text class="t-lbl t-mid t-ctl" x="105" y="71">GPU</text>
  <text class="t-sm t-mid" x="105" y="87">Vulkan, Metal, CUDA, ROCm</text>
  <rect class="blk blk-ctl" x="245" y="52" width="170" height="44" rx="6"/>
  <text class="t-lbl t-mid t-ctl" x="330" y="71">CPU</text>
  <text class="t-sm t-mid" x="330" y="87">the default device</text>
  <path class="arw arw-data" d="M192 68 H241" marker-end="url(#hyb-arrow)"/>
  <path class="arw arw-data" d="M243 80 H194" marker-end="url(#hyb-arrow)"/>
  <text class="t-ax t-mid t-data" x="217" y="114">hidden state</text>
  <path class="arw" d="M105 98 V124" marker-end="url(#hyb-arrow)"/>
  <path class="arw" d="M330 98 V124" marker-end="url(#hyb-arrow)"/>
  <rect class="blk blk-data" x="20" y="128" width="170" height="146" rx="6"/>
  <text class="t-lbl t-mid t-data" x="105" y="150">VRAM</text>
  <text class="t-ax t-mid" x="105" y="166">device memory</text>
  <text class="t-sm t-mid" x="105" y="190">last gpu_layers layers</text>
  <text class="t-sm t-mid" x="105" y="207">attention, router, head</text>
  <text class="t-sm t-mid" x="105" y="224">expert device tier</text>
  <text class="t-sm t-mid" x="105" y="241">K/V of the GPU layers</text>
  <rect class="blk blk-data" x="245" y="128" width="170" height="146" rx="6"/>
  <text class="t-lbl t-mid t-data" x="330" y="150">RAM</text>
  <text class="t-ax t-mid" x="330" y="166">host memory</text>
  <text class="t-sm t-mid" x="330" y="190">first layers, embeddings</text>
  <text class="t-sm t-mid" x="330" y="207">host experts</text>
  <text class="t-sm t-mid" x="330" y="224">expert RAM tier</text>
  <text class="t-sm t-mid" x="330" y="241">Engram row cache</text>
  <text class="t-sm t-mid" x="330" y="258">K/V of the CPU layers</text>
  <rect class="blk" x="470" y="128" width="170" height="146" rx="6"/>
  <text class="t-lbl t-mid" x="555" y="150">SSD</text>
  <text class="t-ax t-mid" x="555" y="166">the GGUF file</text>
  <text class="t-sm t-mid" x="555" y="190">every tensor</text>
  <text class="t-sm t-mid" x="555" y="207">experts no tier holds</text>
  <text class="t-sm t-mid" x="555" y="224">Engram table</text>
  <text class="t-sm t-mid" x="555" y="241">split parts, any volume</text>
  <path class="arw arw-data" d="M468 200 H419" marker-end="url(#hyb-arrow)"/>
  <text class="t-ax t-mid t-data" x="442" y="192">reads</text>
  <path class="arw arw-data" d="M243 200 H194" marker-end="url(#hyb-arrow)"/>
  <text class="t-ax t-mid t-data" x="217" y="192">uploads</text>
  <text class="t-ax" x="20" y="300">device, gpu_layers, experts and engram choose where each part lives</text>
</svg>
</div>
<p class="dgm-cap"><b>The GPU computes from VRAM and the CPU from RAM.</b> Reader threads fill RAM from the file, uploads fill VRAM from RAM, and the hidden state crosses between the processors where a model is split.</p>
</div>

- Tiered experts and Engram rows are read by reader threads (`loader_threads`, `reader_threads`, 1 to 64 each) with positional reads on the file. On a GPU the expert loaders also pack each expert into the kernels' layout, so the engine thread only uploads it.
- The GPU engine reads token embedding rows from the file on the host. Gemma 4's per-layer embedding table and DeepSeek-V4.1's Engram table stay in the file on every device.
- Under `gpu_layers` the hidden state crosses to the GPU once per forward pass. Host experts cost one readback and one upload per MoE layer. Tiered experts on a GPU read back each layer's router output, a few hundred floats, to find the experts the device pool lacks.
- A split GGUF set opens in place: its parts may sit on different volumes (`shard_dirs`), and nothing is merged or copied. See [Split GGUF files](@/providers/local/model-sources.md#split-gguf-files).

## The placements

Each row is one placement, named by its `LocalModelConfig` keys; a GPU row also sets `device "vulkan"` or another GPU. A part in VRAM runs on the GPU; a part in RAM runs on the CPU unless the cell says it waits for upload.

| Placement | VRAM | RAM | Read from the file while running |
|---|---|---|---|
| CPU, the default | nothing | every weight, the K/V | nothing |
| CPU<br>`tiered ram_bytes` | nothing | dense weights, the K/V, the hottest experts | the other experts |
| GPU | every layer, the head, the K/V | nothing | token embedding rows |
| GPU<br>`gpu_layers` | the last layers, the head, their K/V | the first layers, the embeddings, their K/V | nothing |
| GPU<br>`host` | attention, router, shared expert, head, the K/V | the routed experts | token embedding rows |
| GPU<br>`host layers=n` | the same, plus the experts of the later MoE layers | the routed experts of the first `n` MoE layers | token embedding rows |
| GPU<br>`tiered device_bytes`<br>`ram_bytes` | dense weights, the K/V, the hottest experts | the next hottest experts, waiting for upload | the other experts, token embedding rows |
| GPU<br>`tiered ram_bytes` | dense weights, the K/V | the hottest experts | the other experts, token embedding rows |
| GPU<br>`gpu_layers`<br>`tiered device_bytes`<br>`ram_bytes` | the last layers, their device tier | the first layers, both halves' RAM tiers | the other experts |
| GPUs<br>`shards` | a slice of every layer on each GPU, each with its own K/V | nothing | token embedding rows |

- A `tiered` placement on a GPU with `device_bytes` at 0 is host-tiered: the CPU computes the experts from the RAM tier while the rest runs on the GPU.
- Under `gpu_layers` with tiered experts the CPU layers read the RAM tier and the GPU layers their device tier, which has a RAM tier of its own. `ram_bytes` is split between the halves by MoE layer count, and `device_bytes` belongs to the GPU half.
- `engram ram_bytes="4g"` bounds the Engram row cache of a model with an Engram table on every device; see [Engram cache](@/providers/local/configuration.md#engram-cache).
- The budgets, slots and eviction of the expert tiers are on [Tiered expert placement](@/providers/local/expert-tiers.md); `gpu_layers`, host experts and `shards` are on [Local engine on a GPU](@/providers/local/gpu.md).

## Combinations that load

`LocalModel::load` rejects the others with `CucaError::Config`, naming the rule.

- `gpu_layers` needs a GPU `device` and rejects `shards` and a `drafter`.
- `experts { host; }` needs a GPU `device`. It composes with `gpu_layers` and batching, not with `shards`.
- Host-tiered experts serve a model on one device, so `gpu_layers` with tiered experts needs `device_bytes` above 0.
- `shards` rejects tiered and host experts, `gpu_layers` and a `drafter`.
- On the CPU, `tiered` needs `ram_bytes` above 0 and `device_bytes` at 0.
- The Gemma 4 mixture of experts (26B-A4B) runs on the CPU engine only, with resident or RAM-tiered experts.
- `deepseek41` takes `gpu_layers` only at cuts that keep each compressed cache source with its readers: 0, 20, 26, 32, 38, 39 and 40 of 40.

On the CPU, tiered experts give the resident load's output bit for bit. A placement that uses a GPU agrees with the single-device GPU model to float rounding.

## Measured choices

Machine: Ryzen 9 9950X3D2 with 125 GB of RAM, RX 9070 XT with 15.9 GiB under Windows, Vulkan, release build with native CPU flags. Test: `perf_long_prompt` in `tests/provider_local.rs`, a prompt of about 1,100 tokens and 96 generated tokens, with the prefix cache bypassed. VRAM is the adapter's peak. The Qwen3.6 and Mistral Small 4 rows ran with 1 GiB weight chunks, the others with the current 256 MiB.

| Model and placement | Decode, tokens/s | Prefill | VRAM, GiB |
|---|---|---|---|
| `muse-glimmer-30b-q4_0` (16.27 GB, dense), CPU only | 3.5 | 79.8 to 83.0 s | 0 |
| the same, `gpu_layers 38` of 52 | 8.3 to 8.4 | 37.5 to 40.7 s | 14.2 |
| `qwen3.6-35b-a3b-q4_0` (20.84 GB, experts), CPU only | 18.8 to 18.9 | 23.8 to 24.4 s | 0 |
| the same, `experts { host; }` | 10.1 to 10.2 | 19.6 to 21.8 s | 5.9 |
| the same, `experts { tiered device_bytes="6g" ram_bytes="12g"; }` | 10.6 to 10.9 | 12.2 to 14.4 s | 10.6 |
| `mistral-small-4-119b-q4_0` (67.71 GB, experts), CPU only | 11.2 | 34.4 s | 0 |
| the same, `experts { host; }` | 8.6 to 8.8 | 22.3 to 23.9 s | 6.6 |

- The GPU shortens every prefill.
- A dense model larger than the card decodes faster with every layer moved to the GPU: Muse 30B gains about 5.3 ms per token for each layer between 26 and 38. At 40 layers it runs out of device memory.
- A mixture of experts that fits in RAM decodes faster on the CPU alone, which computes the experts at full memory bandwidth while the GPU path pays a launch and a readback per MoE layer. A GPU placement pays off there for long prompts.

DeepSeek-V4.1 is larger than RAM: the converted set is about 507 GB, and the runs read it from an NVMe SSD with `experts { tiered ram_bytes="90g" loader_threads=32; }`. On the CPU alone a 24-token prompt decodes 200 tokens at 2.15 tokens/s and a 2,014-token prompt prefills at 13.7 tokens/s. With attention on the GPU (host-tiered, 12.5 GiB of VRAM at the peak) it decodes at 1.96 tokens/s and prefills a 1,752-token prompt at 11.8 tokens/s. Moving 20 of its 40 layers to the GPU with a 3 or 6 GiB device tier is slower, 0.94 and 1.08 tokens/s. Method and every run: [DeepSeek-V4.1 real checkpoint on an SSD](@/providers/local/verification.md#deepseek-v4-1-real-checkpoint-on-an-ssd).

## Examples

A dense model larger than the card, with 38 of its 52 layers on the GPU:

```kdl,name=muse-offload.kdl
source "muse-glimmer-30b-q4_0"
device "vulkan"
gpu_layers 38
```

A mixture of experts larger than the card, with experts in VRAM, RAM and the file:

```kdl,name=qwen-moe.kdl
source "qwen3.6-35b-a3b-q4_0"
device "vulkan"
experts { tiered device_bytes="4g" ram_bytes="12g"; }
```

```bash,name=Runs the same on all three platforms
cargo run --release --example local_model --features provider-local-vulkan -- --config qwen-moe.kdl
```

A mixture of experts larger than the RAM you give it, on the CPU alone, with 48 GiB of experts in RAM and the rest read from the file (2.1 tokens/s warm in the tiers test, against 8.7 with all 67.7 GB resident):

```kdl,name=mistral-tiered.kdl
source "mistral-small-4-119b-q4_0"
experts { tiered ram_bytes="48g"; }
```

```bash,name=Linux and macOS
RUSTFLAGS="-C target-cpu=native" cargo run --release --example local_model --features provider-local -- --config mistral-tiered.kdl
```

```powershell,name=Windows (PowerShell)
$env:RUSTFLAGS = "-C target-cpu=native"
cargo run --release --example local_model --features provider-local -- --config mistral-tiered.kdl
```

The example prints the split on stderr (`loaded on Vulkan(Default): 38 of 52 layers on the GPU ...` for the first file) and, with tiered experts, a line that starts with `expert tiers: ExpertTierStats {` after the reply.

`cuca serve` takes the same keys in a `model` node:

```kdl,name=cuca.kdl
serve {
    model "qwen-moe" {
        source "qwen3.6-35b-a3b-q4_0"
        device "vulkan"
        experts { tiered device_bytes="4g" ram_bytes="12g"; }
    }
}
```

In Rust the keys are the fields of `LocalModelConfig`:

```rust,name=The qwen-moe placement in library mode (provider-local-vulkan)
use cuca::{
    ExpertPlacement, ExpertTiers, GpuAdapter, LocalDevice, LocalModel, LocalModelConfig,
    ModelSource,
};

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = LocalModel::load(LocalModelConfig {
        source: "qwen3.6-35b-a3b-q4_0".parse::<ModelSource>()?,
        device: LocalDevice::Vulkan(GpuAdapter::Default),
        experts: ExpertPlacement::Tiered(ExpertTiers {
            device_bytes: 4 << 30,
            ram_bytes: 12 << 30,
            ..ExpertTiers::default()
        }),
        ..Default::default()
    })
    .await?;
    println!("{:?}", model.expert_tier_stats());
    Ok(())
}
```
