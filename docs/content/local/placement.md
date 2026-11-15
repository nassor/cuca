+++
title = "Fit a large model"
description = "Choose how to run a model that is bigger than your GPU memory or your RAM: layer offload, host experts, tiered experts, with a decision chart, config examples and measured speeds."
template = "page.html"
aliases = ["providers/local/placement/"]
weight = 7
+++

# Fit a large model

<dl class="page-facts">
<dt>In one line</dt>
<dd>Three config keys, <code>gpu_layers</code>, <code>experts</code> and <code>device</code>, decide which parts of a model sit in GPU memory, in RAM or only in the model file on disk.</dd>
<dt>You need</dt>
<dd><code>provider-local</code> for CPU placements, a GPU feature such as <code>provider-local-vulkan</code> for the others, and a mixture-of-experts model for the <code>experts</code> keys.</dd>
<dt>Read this if</dt>
<dd>A model does not fit your GPU memory, or does not fit your RAM, and you need to choose how to split it.</dd>
</dl>

A model has to live somewhere while it runs. The GPU computes from its own memory (VRAM). The CPU computes from RAM. The model file on the SSD holds every weight, but reading from it while the model runs is slow. When the model is bigger than VRAM, you split it between these places.

Two terms help here:

- A **dense** model uses all of its weights for every token.
- A **mixture of experts** (MoE) model has many small expert blocks in each layer, and every token uses only a few of them. That makes it easy to keep most experts out of fast memory. See [How it works](@/local/how-it-works.md).

## Pick a placement

Follow the chart from the top. The first question you can answer "yes" to is your placement.

<div class="dgm">
<div class="dgm-scroll">
<svg viewBox="0 0 660 296" role="img" aria-labelledby="pick-title pick-desc">
  <title id="pick-title">Which placement fits which model</title>
  <desc id="pick-desc">Four questions in order. If the model fits in VRAM, run it on the GPU with the device key. If not and it is dense, use gpu_layers to put the last layers on the GPU. If it is a mixture of experts that fits in RAM, use host experts, or tiered experts with a device tier. If it is a mixture of experts bigger than RAM, use tiered experts with a RAM budget so the rest is read from the SSD.</desc>
  <defs>
    <marker id="pick-arrow" viewBox="0 0 8 8" refX="7" refY="4"
            markerWidth="6" markerHeight="6" orient="auto">
      <path d="M0 0.6 L7 4 L0 7.4 z" fill="var(--muted-foreground)"/>
    </marker>
  </defs>
  <text class="t-title" x="6" y="20">Which placement for which model</text>
  <rect class="blk" x="6" y="40" width="230" height="44" rx="6"/>
  <text class="t-lbl t-mid" x="121" y="67">Fits in VRAM?</text>
  <rect class="blk blk-ctl" x="330" y="40" width="324" height="44" rx="6"/>
  <text class="t-lbl t-mid t-ctl" x="492" y="58">Run it on the GPU</text>
  <text class="t-sm t-mid" x="492" y="74">device "vulkan"</text>
  <path class="arw" d="M238 62 H326" marker-end="url(#pick-arrow)"/>
  <text class="t-ax t-mid" x="282" y="54">yes</text>
  <path class="arw" d="M121 86 V104" marker-end="url(#pick-arrow)"/>
  <text class="t-ax" x="130" y="99">no</text>
  <rect class="blk" x="6" y="106" width="230" height="44" rx="6"/>
  <text class="t-lbl t-mid" x="121" y="133">Dense, bigger than VRAM?</text>
  <rect class="blk blk-ctl" x="330" y="106" width="324" height="44" rx="6"/>
  <text class="t-lbl t-mid t-ctl" x="492" y="124">Layer offload</text>
  <text class="t-sm t-mid" x="492" y="140">gpu_layers 38</text>
  <path class="arw" d="M238 128 H326" marker-end="url(#pick-arrow)"/>
  <text class="t-ax t-mid" x="282" y="120">yes</text>
  <path class="arw" d="M121 152 V170" marker-end="url(#pick-arrow)"/>
  <text class="t-ax" x="130" y="165">no</text>
  <rect class="blk" x="6" y="172" width="230" height="44" rx="6"/>
  <text class="t-lbl t-mid" x="121" y="199">MoE, fits in RAM?</text>
  <rect class="blk blk-ctl" x="330" y="172" width="324" height="44" rx="6"/>
  <text class="t-lbl t-mid t-ctl" x="492" y="190">Host or tiered experts</text>
  <text class="t-sm t-mid" x="492" y="206">experts { host; } or a device tier</text>
  <path class="arw" d="M238 194 H326" marker-end="url(#pick-arrow)"/>
  <text class="t-ax t-mid" x="282" y="186">yes</text>
  <path class="arw" d="M121 218 V236" marker-end="url(#pick-arrow)"/>
  <text class="t-ax" x="130" y="231">no</text>
  <rect class="blk" x="6" y="238" width="230" height="44" rx="6"/>
  <text class="t-lbl t-mid" x="121" y="265">MoE, bigger than RAM?</text>
  <rect class="blk blk-ctl" x="330" y="238" width="324" height="44" rx="6"/>
  <text class="t-lbl t-mid t-ctl" x="492" y="256">Tiered experts on the SSD</text>
  <text class="t-sm t-mid" x="492" y="272">experts { tiered ram_bytes="48g"; }</text>
  <path class="arw" d="M238 260 H326" marker-end="url(#pick-arrow)"/>
  <text class="t-ax t-mid" x="282" y="252">yes</text>
</svg>
</div>
<p class="dgm-cap"><b>Fit the model, then split what does not fit.</b> A dense model splits by layer. A mixture of experts splits by expert, because only a few experts work on each token.</p>
</div>

## Where each part lives

<div class="dgm">
<div class="dgm-scroll">
<svg viewBox="10 12 660 296" role="img" aria-labelledby="hyb-title hyb-desc">
  <title id="hyb-title">One local model spread over VRAM, RAM and the SSD, computed by the GPU and the CPU</title>
  <desc id="hyb-desc">Two processors sit above two memories: the GPU above VRAM and the CPU above RAM. The model file on the SSD sits beside them with no processor. VRAM holds the last gpu_layers layers, attention, the router and the head, the expert device tier, and the K/V of the GPU layers. RAM holds the first layers and the embeddings, host experts, the expert RAM tier, and the K/V of the CPU layers. The file holds every tensor, the experts no tier holds, and split parts on any volume. Reader threads read from the file into RAM, and uploads move experts from RAM to VRAM. The hidden state crosses between the GPU and the CPU at the gpu_layers cut and for host experts.</desc>
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
  <text class="t-ax t-mid" x="105" y="166">GPU memory</text>
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
  <text class="t-sm t-mid" x="330" y="241">K/V of the CPU layers</text>
  <rect class="blk" x="470" y="128" width="170" height="146" rx="6"/>
  <text class="t-lbl t-mid" x="555" y="150">SSD</text>
  <text class="t-ax t-mid" x="555" y="166">the model file</text>
  <text class="t-sm t-mid" x="555" y="190">every tensor</text>
  <text class="t-sm t-mid" x="555" y="207">experts no tier holds</text>
  <text class="t-sm t-mid" x="555" y="224">split parts, any volume</text>
  <path class="arw arw-data" d="M468 200 H419" marker-end="url(#hyb-arrow)"/>
  <text class="t-ax t-mid t-data" x="442" y="192">reads</text>
  <path class="arw arw-data" d="M243 200 H194" marker-end="url(#hyb-arrow)"/>
  <text class="t-ax t-mid t-data" x="217" y="192">uploads</text>
  <text class="t-ax" x="20" y="300">device, gpu_layers and experts choose where each part lives</text>
</svg>
</div>
<p class="dgm-cap"><b>The GPU computes from VRAM and the CPU from RAM.</b> Reader threads fill RAM from the file, uploads fill VRAM from RAM, and the hidden state crosses between the processors where a model is split.</p>
</div>

The K/V cache is the model's memory of the tokens it has already read (see [How it works](@/local/how-it-works.md)). It sits with the layers that use it.

| Placement | In VRAM | In RAM | Read from the file while running |
|---|---|---|---|
| CPU, the default | nothing | every weight, the K/V | nothing |
| CPU, `tiered ram_bytes` | nothing | dense weights, the K/V, the hottest experts | the other experts |
| GPU | every layer, the head, the K/V | nothing | token embedding rows |
| GPU, `gpu_layers` | the last layers, the head, their K/V | the first layers, the embeddings, their K/V | nothing |
| GPU, `host` | attention, router, shared expert, head, the K/V | the routed experts | token embedding rows |
| GPU, `host layers=n` | the same, plus the experts of the later MoE layers | the routed experts of the first `n` MoE layers | token embedding rows |
| GPU, `tiered device_bytes` and `ram_bytes` | dense weights, the K/V, the hottest experts | the next hottest experts | the other experts |
| GPU, `tiered ram_bytes` only | dense weights, the K/V | the hottest experts | the other experts |
| GPU, `gpu_layers` and `tiered` | the last layers, their device tier | the first layers, both RAM tiers | the other experts |
| Several GPUs, `shards` | a slice of every layer on each GPU | nothing | token embedding rows |

A `tiered` placement on a GPU with `device_bytes` at 0 is called host-tiered. The CPU computes the experts from the RAM tier while the rest runs on the GPU. A split GGUF set (a model stored as several part files) opens in place, and its parts may sit on different volumes. See [Model files and downloads](@/local/model-sources.md).

## Layer offload

Use this for a dense model that is bigger than VRAM. It is llama.cpp's `-ngl`. The last `n` layers, the final norm and the output head run on the GPU. The first layers and the token embeddings stay on the CPU. The hidden state (the values passed from layer to layer) crosses to the GPU once per forward pass.

1. Write the config. This runs 17 of the 35 layers of Gemma 4 E2B on the GPU:

   ```kdl,name=gemma4-offload.kdl
   device "vulkan"
   gpu_layers 17
   ```

2. Run it:

   ```bash,name=Runs the same on all three platforms
   cargo run --release --example local_model --features provider-local-vulkan -- --config gemma4-offload.kdl
   ```

3. Check the split on stderr:

   ```text,name=Expected line on stderr
   loaded on vulkan: 17 of 35 layers on the GPU (layers 18 to 34), layers 0 to 17 and the embeddings on the CPU
   ```

To choose `n`, start high and lower it until the load works. Each layer you move to the GPU makes decoding faster. If `n` is too high, the load or the first request can fail when VRAM runs out (the error text comes from the GPU backend): lower `n` and try again.

- `gpu_layers 0` loads a plain CPU model and opens no device. `gpu_layers` equal to the layer count loads a plain GPU model.
- `LocalModel::layer_split()` returns `(layers on the GPU, layers in all)`, and `None` for a model on one device.
- It fails with a `CucaError::Config` for a CPU device, for `shards` or a `drafter`, and for a count above the layer count (``gpu_layers N exceeds the M layers of `<architecture>` ``).

## Host experts

Use this for a mixture of experts that fits in RAM but not in VRAM. It is llama.cpp's `--cpu-moe` and `--n-cpu-moe`. The routed experts stay in RAM, and the CPU computes them. Attention, the router (the part that picks the experts for each token), the shared expert and the head run on the GPU. For each MoE layer, the GPU routes, the CPU runs the chosen experts, and the sum goes back to the GPU.

```kdl,name=host-experts.kdl (write one of the two)
experts { host; }
experts { host layers=20; }
```

- `host` keeps every MoE layer's experts in RAM. `host layers=20` does so for the first 20 MoE layers, and the other layers keep their experts in VRAM.
- Every hosted expert is read into RAM at load, so your RAM must hold them.
- It works with batching and with `gpu_layers`.
- It fails with a `CucaError::Config` on a CPU device, with `shards`, and on a model without experts.

## Tiered experts

Use this when a mixture of experts is bigger than the RAM you want to spend, or to cache the hottest experts in VRAM. The experts stay in the file. Two caches, one in VRAM and one in RAM, keep the experts the model uses most. The rest are read from the SSD when a token needs them. The budgets, slots and counters are on [Tiered experts](@/local/expert-tiers.md).

A mixture of experts larger than the card, with experts in VRAM, RAM and the file:

```kdl,name=qwen-moe.kdl
source "qwen3.6-35b-a3b-q4_0"
device "vulkan"
experts { tiered device_bytes="4g" ram_bytes="12g"; }
```

```bash,name=Runs the same on all three platforms
cargo run --release --example local_model --features provider-local-vulkan -- --config qwen-moe.kdl
```

The example prints the split on stderr, and after the reply a line that starts with `expert tiers: ExpertTierStats {`.

A mixture of experts larger than the RAM you give it, on the CPU alone. This keeps 48 GiB of experts in RAM and reads the rest from the file:

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

## Combinations that load

`LocalModel::load` rejects every other combination with a `CucaError::Config` that names the rule.

- `gpu_layers` needs a GPU `device`, and rejects `shards` and a `drafter`.
- `experts { host; }` needs a GPU `device`. It works with `gpu_layers` and batching, and not with `shards`.
- `gpu_layers` with tiered experts needs `device_bytes` above 0, because host-tiered experts serve a model on one device.
- `shards` rejects tiered and host experts, `gpu_layers` and a `drafter`.
- On the CPU, `tiered` needs `ram_bytes` above 0 and `device_bytes` at 0.
- The Gemma 4 mixture of experts (26B-A4B) runs on the CPU only, with resident or RAM-tiered experts.
- `deepseek41` takes `gpu_layers` at only a few values. See [Run DeepSeek-V4.1](@/local/deepseek-v41.md).

Placement changes speed, not the answer. On the CPU, tiered experts give the resident load's output bit for bit. A placement that uses a GPU agrees with the one-GPU model to float rounding.

## Use a placement in a model server and Rust

The `serve` section of a `cuca run` file takes the same keys in a `model` node:

```kdl,name=cuca.kdl
serve {
    model "qwen-moe" {
        source "qwen3.6-35b-a3b-q4_0"
        device "vulkan"
        experts { tiered device_bytes="4g" ram_bytes="12g"; }
    }
}
```

In Rust the keys are fields of `LocalModelConfig`. `gpu_layers: Some(38)` is layer offload, and `ExpertPlacement::Host { layers: None }` is `experts { host; }`:

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

## What to expect

Method: a prompt of about 1,100 tokens and 96 generated tokens, prefix cache off. Vulkan on an AMD RX 9070 XT (16 GB) under Windows, with a Ryzen 9 desktop CPU and 125 GB of RAM, release build with native CPU flags. VRAM is the peak of the run. Prefill is the time to read the prompt.

| Model and placement | Decode, tokens/s | Prefill | VRAM, GiB |
|---|---|---|---|
| `muse-glimmer-30b-q4_0` (16 GB, dense), CPU only | 3.5 | 81 s | 0 |
| the same, `gpu_layers 38` of 52 | 8.3 | 39 s | 14.2 |
| `qwen3.6-35b-a3b-q4_0` (21 GB, MoE), CPU only | 18.9 | 24 s | 0 |
| the same, `experts { host; }` | 10.2 | 21 s | 5.9 |
| the same, `tiered device_bytes="6g" ram_bytes="12g"` | 10.8 | 13 s | 10.6 |
| `mistral-small-4-119b-q4_0` (68 GB, MoE), CPU only | 11.2 | 34 s | 0 |
| the same, `experts { host; }` | 8.7 | 23 s | 6.6 |

- The GPU shortens every prefill.
- A dense model bigger than the card decodes faster with every layer you move to the GPU. Muse 30B is 2.4 times faster at 38 layers. It does not fit at 40.
- A mixture of experts that fits in RAM decodes faster on the CPU alone, because the CPU computes the experts at full memory bandwidth and the GPU pays a launch and a copy per MoE layer. A GPU placement pays off there for long prompts.
- In a separate CPU test (three short replies), Mistral Small 4 with `experts { tiered ram_bytes="48g"; }` ran at 2.1 tokens/s once warm, against 8.7 tokens/s with all 68 GB in RAM. Reading experts from the SSD is the price of fitting the model.
- DeepSeek-V4.1 is about 507 GB, bigger than most RAM. See [Run DeepSeek-V4.1](@/local/deepseek-v41.md).
