+++
title = "Tiered experts"
description = "Run a mixture-of-experts model with its experts on disk and a RAM and VRAM cache you size: the config keys, how big the budget must be, warm starts, counters, errors and measured speed."
template = "page.html"
aliases = ["providers/local/expert-tiers/"]
weight = 8
+++

# Tiered experts

<dl class="page-facts">
<dt>In one line</dt>
<dd><code>ExpertPlacement::Tiered</code> keeps a mixture-of-experts model's experts in the model file and caches the most used ones in RAM, and on a GPU in VRAM, within byte budgets you choose. The output stays the same.</dd>
<dt>You need</dt>
<dd>The <code>provider-local</code> feature, or a GPU feature such as <code>provider-local-vulkan</code>, and a model with experts.</dd>
<dt>Read this if</dt>
<dd>A mixture-of-experts model is bigger than the RAM or VRAM you want to spend on it, or you want the cache counters and the heat export.</dd>
</dl>

A mixture-of-experts model has many small expert blocks in every layer, and each token uses only a few of them. Loading the whole file puts every expert in memory, even though most sit unused. Tiered placement keeps the experts in the file on disk and caches the ones the model uses most. The dense parts (attention, norms, routers, embeddings and the output head) stay in memory.

How tiers combine with `gpu_layers` and host experts is on [Fit a large model](@/local/placement.md).

## Try it

This loads OLMoE 1B-7B with room for 600 MiB of experts in RAM. The rest of the model stays on disk.

```kdl,name=olmoe-tiered.kdl
source "olmoe-1b-7b-q4_0"
experts { tiered ram_bytes="600m"; }
```

```bash,name=Linux and macOS: the example with tiered experts
RUSTFLAGS="-C target-cpu=native" cargo run --release --example local_model --features provider-local -- --config olmoe-tiered.kdl
```

```powershell,name=Windows (PowerShell): the example with tiered experts
$env:RUSTFLAGS = "-C target-cpu=native"
cargo run --release --example local_model --features provider-local -- --config olmoe-tiered.kdl
```

After the reply, the example prints the cache counters on stderr:

```text,name=Expected on stderr after the reply
expert tiers: ExpertTierStats { lookups: <n>, device_hits: 0, ram_hits: <n>, disk_reads: <n>, bytes_read: <n>, ram_resident_bytes: <n>, device_resident_bytes: 0 }
```

`disk_reads` is above 0 once the first token has been routed. Each field is explained in [Counters](#counters).

Supported models are the mixture-of-experts architectures `qwen3moe`, `olmoe`, `qwen35moe`, `mistral4`, `deepseek41`, and `gemma4` with experts (the Gemma 4 26B-A4B, CPU only).

## How it works

<div class="dgm">
<div class="dgm-scroll">
<svg viewBox="0 0 660 146" role="img" aria-labelledby="tier-title tier-desc">
  <title id="tier-title">Where an expert travels from the file to the compute</title>
  <desc id="tier-desc">Five boxes, left to right. The model file on disk holds every expert. Loader threads read a missing expert with positional reads. The RAM tier holds built experts. The device tier holds uploaded experts and exists on GPU devices only. Expert compute runs the routed experts of a layer in ascending order. On the CPU a second arrow carries experts from the RAM tier straight to compute.</desc>
  <defs>
    <marker id="tier-arrow" viewBox="0 0 8 8" refX="7" refY="4"
            markerWidth="6" markerHeight="6" orient="auto">
      <path d="M0 0.6 L7 4 L0 7.4 z" fill="var(--muted-foreground)"/>
    </marker>
  </defs>
  <text class="t-title" x="6" y="20">Experts on demand</text>
  <rect class="blk" x="6" y="44" width="104" height="50" rx="6"/>
  <text class="t-sm t-mid" x="58" y="67">model file</text>
  <text class="t-sm t-mid" x="58" y="81">all experts</text>
  <rect class="blk blk-ctl" x="142" y="44" width="104" height="50" rx="6"/>
  <text class="t-sm t-mid t-ctl" x="194" y="67">loader threads</text>
  <text class="t-sm t-mid" x="194" y="81">1 to 64</text>
  <rect class="blk blk-data" x="278" y="44" width="104" height="50" rx="6"/>
  <text class="t-sm t-mid t-data" x="330" y="67">RAM tier</text>
  <text class="t-sm t-mid" x="330" y="81">ram_bytes</text>
  <rect class="blk blk-data" x="414" y="44" width="104" height="50" rx="6"/>
  <text class="t-sm t-mid t-data" x="466" y="67">device tier</text>
  <text class="t-sm t-mid" x="466" y="81">device_bytes</text>
  <rect class="blk" x="550" y="44" width="104" height="50" rx="6"/>
  <text class="t-sm t-mid" x="602" y="67">expert compute</text>
  <text class="t-sm t-mid" x="602" y="81">ascending order</text>
  <path class="arw" d="M112 69 H138" marker-end="url(#tier-arrow)"/>
  <path class="arw" d="M248 69 H274" marker-end="url(#tier-arrow)"/>
  <path class="arw arw-data" d="M384 69 H410" marker-end="url(#tier-arrow)"/>
  <path class="arw arw-data" d="M520 69 H546" marker-end="url(#tier-arrow)"/>
  <path class="arw arw-data" d="M330 96 V116 H602 V98" marker-end="url(#tier-arrow)"/>
  <text class="t-ax t-mid" x="466" y="134">CPU device: no device tier, the RAM tier feeds compute</text>
</svg>
</div>
<p class="dgm-cap"><b>The CPU has one cache, a GPU has two.</b> A GPU device with <code>ram_bytes = 0</code> reads every device miss from the file.</p>
</div>

- On the CPU, the RAM tier holds experts ready to compute. An expert costs its file bytes.
- On a GPU, the device tier holds the uploaded experts. The optional RAM tier holds host copies that upload without conversion.
- Loader threads read an expert's matrices from the file. On a GPU they also convert them to the layout the GPU kernels read, so the engine thread only uploads.
- Each expert has a heat counter that rises when the router picks it and halves after every 4,096 routed rows of its layer, so heat follows recent use. A full tier evicts the cached expert with the lowest heat that the running pass no longer needs. A prefill that scans many cold experts evicts them among themselves and leaves the hot ones in place.
- Placement changes speed only. On the CPU the output equals the resident load's bit for bit. On a GPU, tiered placements agree with each other bit for bit and with the resident load to float rounding.

## Configuration

`LocalModelConfig::experts` is an `ExpertPlacement`, `Resident` by default. `Tiered` carries an `ExpertTiers`:

| Field | Default | Rule |
|---|---|---|
| `ram_bytes` | `0` | Bytes of RAM for cached experts. On the CPU it must be above 0. On a GPU, `0` turns the RAM tier off, unless `device_bytes` is 0 too |
| `device_bytes` | `0` | Bytes of VRAM for cached experts. Above 0 gives a GPU a device tier. On a GPU, `0` with a `ram_bytes` above 0 is host-tiered (see below). Exactly 0 on the CPU |
| `loader_threads` | `4` | Disk reader threads, `1` to `64` |
| `warm_from` | `None` | An `ExpertHeat` from an earlier session. The hottest experts load at startup |

In KDL the placement is the `experts` node:

```kdl,name=The forms of experts (write one)
experts "resident"
experts { tiered ram_bytes="4g"; }
experts { tiered ram_bytes="12g" device_bytes="6g" loader_threads=8; }
experts { host; }
experts { host layers=4; }
```

- `ram_bytes` and `device_bytes` are an integer or a string with an optional `k`, `m` or `g` suffix. The suffixes are powers of 1024 and not case sensitive: `"4g"` is 4,294,967,296 bytes.
- `experts { tiered loader_threads=8; }` parses, but the load rejects it because it names no tier.
- `host` is a different placement: the GPU keeps attention and the CPU computes the experts, with no cache tiers. See [Fit a large model](@/local/placement.md).
- A GPU with `device_bytes` at 0 and a `ram_bytes` budget is host-tiered. The experts sit behind the RAM tier and the CPU computes them, while attention, the router, the shared expert and the head run on the GPU. It does not combine with `gpu_layers` or `shards`.
- An unknown key inside `experts` is an error naming it.

## How big a budget

A budget is divided into the same number of slots in every layer that has experts. A slot holds one expert:

```text,name=Slots per layer
slots = min(n_expert, budget / per_expert)
```

`per_expert` is the bytes one expert takes in that tier, summed over the layers that have experts. A tier must hold at least `n_used` experts in every layer (the number of experts one token uses), so the smallest budget is `n_used * per_expert`. A budget below that fails the load and prints the number to use.

- `olmoe-1b-7b-q4_0`: 64 experts per layer, 8 used per token. The smallest budget is `434m`. The suggested budget is `ram_bytes="600m"`, which gives 11 slots per layer.
- `qwen3-30b-a3b-q4_0`: 128 experts per layer, 8 used. The smallest budget is `977m`. The suggested budget is `ram_bytes="4g"`, which gives 33 slots per layer.
- `qwen3.6-35b-a3b-q4_0`: 256 experts per layer, 8 used. The smallest budget is `543m`. The suggested budget is `ram_bytes="4g"` or `device_bytes="4g"`, which gives 60 slots per layer.
- `gemma4-26b-a4b-q4_0`: 128 experts per layer, 8 used. The smallest budget is `766m`. The suggested budget is `ram_bytes="6g"`, which gives 64 slots per layer.
- `mistral-small-4-119b-q4_0`: 128 experts per layer, 4 used. The smallest budget is `2g`. The suggested budget is `ram_bytes="48g"`, which gives 100 slots per layer.

The shared expert and the router are dense weights and always stay in memory. A GPU layer's device pool is one buffer of at most 1 GiB, which caps the slots of a device tier. DeepSeek-V4.1 has its own sizes: see [Run DeepSeek-V4.1](@/local/deepseek-v41.md).

## Heat and warm start

Heat counts how often each expert was picked. A model starts cold, with empty caches. `LocalModel::expert_heat()` returns the learned counters as `Option<ExpertHeat>`. `ExpertHeat` implements serde's `Serialize` and `Deserialize`. Pass it back as `ExpertTiers::warm_from` and the next load starts warm: it seeds the counters and loads the hottest experts of every layer at startup, the device tier first and then the RAM tier. A heat file from a model of another shape fails the load.

```rust,name=Save the heat and start warm next time (provider-local)
use cuca::{ExpertHeat, ExpertPlacement, ExpertTiers, LocalModel, LocalModelConfig, ModelSource};

const HEAT_FILE: &str = "olmoe-heat.json";

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // The file is absent on the first run, so the model starts cold.
    let warm_from = std::fs::read_to_string(HEAT_FILE)
        .ok()
        .map(|text| serde_json::from_str::<ExpertHeat>(&text))
        .transpose()?;

    let model = LocalModel::load(LocalModelConfig {
        source: "olmoe-1b-7b-q4_0".parse::<ModelSource>()?,
        experts: ExpertPlacement::Tiered(ExpertTiers {
            ram_bytes: 600 << 20,
            warm_from,
            ..ExpertTiers::default()
        }),
        ..Default::default()
    })
    .await?;

    // ... serve requests through a CucaClient built with `with_local_model` ...

    if let Some(stats) = model.expert_tier_stats() {
        println!("hit rate {:?}, {} disk reads", stats.hit_rate(), stats.disk_reads);
    }
    if let Some(heat) = model.expert_heat() {
        std::fs::write(HEAT_FILE, serde_json::to_string(&heat)?)?;
    }
    Ok(())
}
```

The example needs `cargo add serde_json` and `cargo add tokio --features rt,macros` next to `cuca` with the `provider-local` feature. Heat lives in memory only: the library writes no file.

## Counters

`LocalModel::expert_tier_stats()` returns `Option<ExpertTierStats>`, `None` unless the model loaded with `Tiered`. The counters add up from load, except the two `resident` gauges.

| Field | Meaning |
|---|---|
| `lookups` | Expert computations the routers asked for, one per expert per layer pass |
| `device_hits` | Lookups served by the device tier (GPU only) |
| `ram_hits` | Lookups served by the RAM tier |
| `disk_reads` | Experts read from the file, loads of a warm start included |
| `bytes_read` | Bytes read from the file for experts |
| `ram_resident_bytes` | Bytes the RAM tier holds now |
| `device_resident_bytes` | Bytes the device tier holds now |

`ExpertTierStats::hit_rate()` is `(device_hits + ram_hits) / lookups`, and `None` before the first lookup.

## Errors

`LocalModel::load` fails with a `CucaError::Config`, before any thread starts, for:

- `experts.loader_threads must be in 1..=64, got N`.
- `experts.device_bytes is for GPU devices; set it to 0 on LocalDevice::Cpu`.
- `experts.ram_bytes must be greater than 0 on LocalDevice::Cpu`.
- A GPU with `device_bytes` and `ram_bytes` both 0: ``experts: on <device> set device_bytes (a device tier) or ram_bytes (host-tiered: the CPU computes the experts from the RAM tier)``. The device appears in its `Debug` text.

Once the file is open, it also fails with a `CucaError::Config` for:

- A budget below the smallest: ``expert RAM tier of 524288000 bytes holds 4 experts per layer; this model needs at least 8 (1023934464 bytes)``. This one came from `qwen3-30b-a3b-q4_0` at `ram_bytes="500m"`. A device tier prints ``expert device tier of ...`` instead. Use the number it prints.
- A `warm_from` of another shape: `experts.warm_from is heat for AxB experts; this model has CxD`, with layers by experts.
- A model without experts: ``expert tiers need a mixture-of-experts model; `llama` has no experts``.
- `gpu_layers` with host-tiered experts: `experts.device_bytes must be greater than 0 with gpu_layers: host-tiered experts serve a model that runs on one device`.

A KDL `experts` node with an unknown key or placement name is also a `CucaError::Config` that names it. An expert read that fails while generating, for example from a truncated file, fails the request with a `CucaError::Provider` item in the stream.

## What to expect

Tiered placement trades speed for memory. Method: the CPU of a Ryzen 9 desktop, release build with `RUSTFLAGS="-C target-cpu=native"`, three prompts of 48 greedy tokens, the model file in the OS page cache. Cold starts with empty caches. Warm loads with the heat the cold run exported. The hit rate is the share of lookups served from a cache.

| Model | Placement | Tokens per second | Hit rate |
|---|---|---|---|
| `olmoe-1b-7b-q4_0` | resident | 42 | not applicable |
| `olmoe-1b-7b-q4_0` | `ram_bytes="500m"`, 9 of 64 experts per layer | 10.0 cold, 10.1 warm | 0.38 cold, 0.39 warm |
| `qwen3-30b-a3b-q4_0` | resident, 17.4 GB of RAM | 12.3 | not applicable |
| `qwen3-30b-a3b-q4_0` | `ram_bytes="4g"`, 33 of 128 experts per layer | 6.7 cold, 6.9 warm | 0.64 cold, 0.69 warm |
| `mistral-small-4-119b-q4_0` | resident, 67.7 GB of RAM | 8.7 | not applicable |
| `mistral-small-4-119b-q4_0` | `ram_bytes="48g"`, 100 of 128 experts per layer | 0.8 cold, 2.1 warm | 0.82 cold, 0.96 warm |

- A bigger budget gives a higher hit rate and usually more speed. A warm start matters most when reading from disk is slow: Mistral Small 4 runs at 0.8 tokens/s cold and 2.1 warm.
- Memory drops a lot. For Qwen3.6-35B-A3B, one 1,024-token reply peaked at 18.8 GiB of RAM with resident experts, 5.95 GiB at `ram_bytes="4g"` and 2.43 GiB at `ram_bytes="600m"`. The model loaded in 8.3 s resident against 0.8 s tiered.
- Mistral Small 4 is 67.7 GB next to 125 GB of RAM, so only part of the file stayed in the page cache. Its cold start reads many experts from disk.

To run the tiers test yourself, see [Run the live integration suite](@/guides/integration-tests.md).
