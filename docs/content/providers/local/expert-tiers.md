+++
title = "Tiered expert placement"
description = "LocalModelConfig::experts runs a mixture-of-experts model with its experts in the GGUF file and a bounded RAM and device cache, with the same output as a resident load."
template = "page.html"
weight = 12
+++

# Tiered expert placement

<dl class="page-facts">
<dt>In one line</dt>
<dd><code>ExpertPlacement::Tiered</code> keeps a mixture-of-experts model's experts in the GGUF file and loads them on demand through caches of a byte size you choose, with the same output as the resident load.</dd>
<dt>You need</dt>
<dd>The <code>provider-local</code> feature, or a GPU feature such as <code>provider-local-vulkan</code>, and a model with experts (<code>qwen3moe</code>, <code>olmoe</code>, <code>qwen35moe</code> or <code>mistral4</code>).</dd>
<dt>Read this if</dt>
<dd>A mixture-of-experts model is larger than the RAM or device memory you want to spend on it, or you need the cache counters and the heat export.</dd>
</dl>

A token routes to `n_used` of a layer's `n_expert` experts, yet the default placement, `ExpertPlacement::Resident`, loads every expert at startup, so memory is the file size. Tiered placement keeps the dense weights (attention, norms, routers, embeddings, head) resident and leaves the experts in the file. Qwen3-30B-A3B is 17.38 GB, of which the expert tensors are 16.38 GB.

## Configuration

`LocalModelConfig::experts` is an `ExpertPlacement`, `Resident` by default. `Tiered` carries an `ExpertTiers`:

| Field | Default | Rule |
|---|---|---|
| `ram_bytes` | `0` | bytes of RAM for cached experts. On `LocalDevice::Cpu` it must be above 0. On a GPU device `0` disables the RAM tier |
| `device_bytes` | `0` | bytes of device memory for cached experts. Must be above 0 on a GPU device and exactly 0 on `LocalDevice::Cpu` |
| `loader_threads` | `4` | disk reader threads, `1..=64` |
| `warm_from` | `None` | an `ExpertHeat` from an earlier session; the hottest experts load at startup |

An `ExpertPlacement` also parses from text, ASCII case-insensitive, with `str::parse`:

- `resident`, or `tiered:` followed by comma separated `ram=<size>`, `device=<size>` and `threads=<n>` in any order, each at most once, with `ram` or `device` present.
- `<size>` is a byte count with an optional `k`, `m` or `g` suffix, powers of 1024: `4g` is 4,294,967,296 bytes and `1100m` is 1,153,433,600.
- `tiered:ram=4g` and `tiered:device=6g,ram=12g,threads=8` are valid. `tiered:threads=8` is not, because it names no tier.
- `host` and `host:layers=<n>` parse to `ExpertPlacement::Host`, the GPU placement with experts computed on the CPU. It is not a tier; see [Host experts](@/providers/local/gpu.md#host-experts).
- The parser checks the syntax only. The rules in the table apply at `LocalModel::load`.

## Slots and the minimum budget

A tier budget becomes the same number of slots in every layer that has experts:

```text
slots = min(n_expert, budget / per_expert)
```

`per_expert` is the bytes one expert occupies in that tier, summed over the layers that have experts. An expert is the gate, up and down matrices of one layer. The CPU's RAM tier counts an expert's file bytes. A GPU's RAM tier and device tier count its packed bytes, ready to upload.

A tier must hold `n_used` experts in every layer, so the minimum budget is `n_used * per_expert`.

| Model | Experts per layer, used | MoE layers | `per_expert`, bytes | Minimum, bytes | Slots per layer |
|---|---|---|---|---|---|
| `olmoe-1b-7b-q4_0` | 64, 8 | 16 | 56,885,248 | 455,081,984 | `ram=500m`: 9 |
| `qwen3-30b-a3b-q4_0` | 128, 8 | 48 | 127,991,808 | 1,023,934,464 | `ram=4g`: 33<br>`ram=1100m`: 9 |
| `qwen3.6-35b-a3b-q4_0` | 256, 8 | 40 | 71,106,560 | 568,852,480 | `ram=600m`: 8<br>`ram=4g`: 60<br>`device=2g`: 30<br>`device=4g`: 60<br>`device=6g`: 90 |
| `mistral-small-4-119b-q4_0` | 128, 4 | 36 | about 511.7 MB | about 2.05 GB | `ram=8g`: 16<br>`ram=48g`: 100 |

- In the Qwen3 models the down matrices are Q4_0 in most layers and Q4_1 in the rest, so `per_expert` is a sum over layers: for Qwen3-30B-A3B, 42 layers of 2,654,208 bytes and 6 of 2,752,512.
- The shared expert and the router are dense weights and stay resident.
- A GPU tier counts the MoE kernels' own layout, which holds as many bytes as the file for Q4_0, Q4_1 and Q8_0 experts (every other dtype is requantized to Q8_0, a float dtype held as f32; see [Weight formats](@/providers/local/models.md#weight-formats)). The figures above hold on a GPU too, except that a layer's device pool is one buffer of at most 1 GiB, which caps its slots.
- The minimum printed by a failed load is the number to use.

## Tiers

<div class="dgm">
<div class="dgm-scroll">
<svg viewBox="0 0 660 146" role="img" aria-labelledby="tier-title tier-desc">
  <title id="tier-title">Where an expert travels from the file to the compute</title>
  <desc id="tier-desc">Five boxes, left to right. The GGUF file on disk holds every expert. Loader threads read a missing expert with positional reads. The RAM tier holds built experts. The device tier holds uploaded experts and exists on GPU devices only. Expert compute runs the routed experts of a layer in ascending order. On the CPU a second arrow carries experts from the RAM tier straight to compute.</desc>
  <defs>
    <marker id="tier-arrow" viewBox="0 0 8 8" refX="7" refY="4"
            markerWidth="6" markerHeight="6" orient="auto">
      <path d="M0 0.6 L7 4 L0 7.4 z" fill="var(--muted-foreground)"/>
    </marker>
  </defs>
  <text class="t-title" x="6" y="20">Experts on demand</text>
  <rect class="blk" x="6" y="44" width="104" height="50" rx="6"/>
  <text class="t-sm t-mid" x="58" y="67">GGUF file</text>
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

- On `LocalDevice::Cpu` the RAM tier holds quantized matmul weights ready to compute. An expert costs its file bytes.
- On a GPU device the device tier holds the uploaded weights. The optional RAM tier holds packed host copies that upload without conversion.
- Loader threads read the three matrices of an expert with positional reads on one shared file handle, so they never move a cursor. On a GPU they also convert the matrices to the packed layout, so the engine thread only uploads. At most twice `loader_threads` reads are in flight.
- Reads in flight count against the RAM slots, and a tier never holds more experts than its slots. An expert a pass is computing may outlive its eviction for that pass: one on the CPU, and on a GPU up to `n_used` in a single-token pass, whose missing experts are all uploaded before it computes.
- The heat table costs `4 * n_layer * n_expert` bytes, fixed at load.

## Eviction

Each `(layer, expert)` pair has a heat counter. Every routing adds the expert's pick count to it. After every 4,096 routed rows of a layer, all of that layer's counts halve, so heat follows recent traffic.

A full tier evicts one expert to make room:

- The victim is the cached expert that the running layer pass no longer needs, with the lowest heat. Ties go to the least recently used, then to the lower expert id.
- If the pass still needs every cached expert, the one needed farthest ahead goes (the largest id, since a pass computes experts in ascending order), and only when it is needed after the expert being fetched.
- On a GPU, a pass whose experts outnumber the device slots runs in groups of at most `device_slots` experts: the pool's hits first, then the misses, written into free slots or the slots of victims outside the group. The kernels sum every group's results in expert order at the end, so grouping changes no bit.

A prefill that scans many cold experts evicts them among themselves and leaves the hot ones in place.

## Output guarantee

Placement decides speed only. An expert is built from the same file bytes as in the resident load, and a layer pass accumulates its experts in the same ascending order.

- On the CPU the output equals the resident load's bit for bit.
- On a GPU the kernels' result does not depend on how a pass groups its experts, so tiered placements agree with each other bit for bit. Against the resident load on the same device they agree to float rounding: the resident load never reads the router output back and a tiered one does at every layer, which splits burn's fused elementwise kernels at other places.

What ran:

- The CPU unit tests `tiered_experts_match_resident_bit_for_bit` and `tiered_fused_experts_match_resident_bit_for_bit`, and on a GPU (when `CUCA_LOCAL_DEVICE` names one) `gpu_tiered_experts_match_gpu_resident`, on tiny models.
- `tests/provider_local_tiers.rs`: on the CPU every run of the table below equaled the resident replies (for Mistral Small 4 at `ram=8g`, the `ram=48g` run, which equals them). On Vulkan (RX 9070 XT, Windows) it reproduced the reference replies for OLMoE 1B-7B (`tiered:device=1g,ram=0` against resident), Qwen3-30B-A3B (`tiered:device=6g,ram=12g` cold and warm against `tiered:device=2g,ram=0`) and Qwen3.6-35B-A3B (`tiered:device=4g,ram=12g` and `tiered:device=2g,ram=4g`). The live suite `tests/provider_local.rs` passed all seven scenarios for Qwen3-30B-A3B at `tiered:device=6g,ram=12g`.

## Statistics and heat

`LocalModel::expert_tier_stats()` returns `Option<ExpertTierStats>`, `None` unless the model loaded with `Tiered`. The counters are cumulative since load, except the two resident gauges.

| Field | Meaning |
|---|---|
| `lookups` | expert computations the routers asked for, one per expert per layer pass |
| `device_hits` | lookups served by the device tier, GPU only |
| `ram_hits` | lookups served by the RAM tier |
| `disk_reads` | experts read from the file, loads of a warm start included |
| `bytes_read` | bytes read from the file for experts |
| `ram_resident_bytes` | bytes the RAM tier holds now |
| `device_resident_bytes` | bytes the device tier holds now |

`ExpertTierStats::hit_rate()` is `(device_hits + ram_hits) / lookups`, `None` before the first lookup.

`LocalModel::expert_heat()` returns `Option<ExpertHeat>`: the learned counters, one per layer and expert. `ExpertHeat` implements serde's `Serialize` and `Deserialize`, and has `n_layer()`, `n_expert()` and `count(layer, expert)`. Pass it back as `ExpertTiers::warm_from` and the next load seeds the counters and loads the hottest experts of every layer at startup, the device tier first and then the RAM tier. A heat file from a model of another shape fails the load.

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

The example needs `cargo add serde_json` and `cargo add tokio --features rt,macros` next to `cuca` with the `provider-local` feature. Heat is in memory only: the library writes no file.

## Measured speed

Tiered placement trades speed for memory. Method: `tests/provider_local_tiers.rs`, three prompts of 48 greedy tokens each, the GGUF file in the OS page cache. Cold starts with empty caches. Warm loads with the heat the cold run exported.

CPU: Ryzen 9 9950X3D2 with 125 GB of RAM, release build with `RUSTFLAGS="-C target-cpu=native"`.

| Model | Placement | Tokens per second | Hit rate |
|---|---|---|---|
| `olmoe-1b-7b-q4_0` | resident | 42 | not applicable |
| `olmoe-1b-7b-q4_0` | `tiered:ram=500m`, 9 of 64 experts per layer | 10.0 cold, 10.1 warm | 0.381 cold, 0.385 warm |
| `qwen3-30b-a3b-q4_0` | resident, 17.4 GB of RAM | 12.3 | not applicable |
| `qwen3-30b-a3b-q4_0` | `tiered:ram=4g`, 33 of 128 experts per layer | 6.7 cold, 6.9 warm | 0.641 cold, 0.686 warm |
| `qwen3-30b-a3b-q4_0` | `tiered:ram=1100m`, 9 of 128 experts per layer | 6.2 cold, 5.3 warm | 0.381 cold, 0.383 warm |
| `qwen3.6-35b-a3b-q4_0` | resident, 18.9 GiB peak resident set | 14.7 and 14.9 | not applicable |
| `qwen3.6-35b-a3b-q4_0` | `tiered:ram=4g`, 60 of 256 experts per layer | 7.4 cold, 7.1 warm | 0.661 cold, 0.708 warm |
| `qwen3.6-35b-a3b-q4_0` | `tiered:ram=600m`, 8 of 256 experts per layer | 6.3 cold, 6.2 warm | 0.307 cold, 0.308 warm |
| `mistral-small-4-119b-q4_0` | resident, 67.7 GB of RAM | 8.7 | not applicable |
| `mistral-small-4-119b-q4_0` | `tiered:ram=48g`, 100 of 128 experts per layer | 0.8 cold, 2.1 warm | 0.824 cold, 0.958 warm |
| `mistral-small-4-119b-q4_0` | `tiered:ram=8g`, 16 of 128 experts per layer | 1.1 cold, 1.8 warm | 0.310 cold, 0.337 warm |

- Mistral Small 4 is 67.7 GB next to 125 GB of RAM, so only part of the file stayed in the page cache. One expert of all 36 layers is 511.7 MB. The cold run at `ram=48g` read 4,618 experts (65.7 GB) out of 26,205 lookups, and the cold run at `ram=8g` read 18,081 (257 GB).
- Qwen3.6-35B-A3B: the cold runs read 18,861 experts (33.6 GB) at `ram=4g` and 38,510 (68.5 GB) at `ram=600m`, out of 55,597 lookups, from a file that was in the OS page cache. A 1,024-token reply (`examples/local_model.rs`) peaked at 18.8 GiB of resident set with resident experts, 5.95 GiB at `ram=4g` and 2.43 GiB at `ram=600m`. The model loaded in 8.3 s resident against 0.8 s tiered.

Vulkan: RX 9070 XT with 15.9 GiB, Windows, release build. The rates include the kernel tuning that each new matrix shape pays on first use, which a short run does not amortize.

| Model | Placement | Tokens per second | Hit rate |
|---|---|---|---|
| `olmoe-1b-7b-q4_0` | resident | 8.3 | not applicable |
| `olmoe-1b-7b-q4_0` | `tiered:device=1g,ram=0` | 5.7 cold, 6.1 warm | 0.575 cold, 0.599 warm |
| `qwen3-30b-a3b-q4_0` | `tiered:device=2g,ram=0` | 1.9 | 0.501 |
| `qwen3-30b-a3b-q4_0` | `tiered:device=6g,ram=12g` | 4.9 cold, 6.1 warm | 0.911 cold, 0.983 warm |
| `qwen3.6-35b-a3b-q4_0` | `tiered:device=2g,ram=0` | 1.9 to 2.5 | 0.541 |
| `qwen3.6-35b-a3b-q4_0` | `tiered:device=2g,ram=4g` | 4.8 cold, 4.8 warm | 0.669 cold, 0.736 warm |
| `qwen3.6-35b-a3b-q4_0` | `tiered:device=4g,ram=12g` | 6.5 cold, 8.3 warm | 0.863 cold, 0.973 warm |

- In the `qwen3.6-35b-a3b-q4_0` run at `device=4g,ram=12g` the cold start read 7,627 experts from the file, 9,939 hits came from the RAM tier and 38,041 from the device tier, out of 55,607 lookups. One 1,024-token reply (`examples/local_model.rs`, `loaded on Vulkan(Default)`) ran at about 9.6 tokens per second with `tiered:device=6g,ram=12g` and 10.1 with `tiered:device=4g,ram=12g`.
- The desktop held 2.4 to 3.2 GiB of the card during these runs. Dedicated memory of the whole card, desktop included, peaked at 10.5 GiB over the full live suite at `device=4g,ram=12g` and at 14.6 GiB at `device=6g,ram=12g`.

## Errors

`LocalModel::load` fails with `CucaError::Config`, before any thread starts, for:

- `experts.loader_threads must be in 1..=64, got N`.
- `experts.device_bytes is for GPU devices; set it to 0 on LocalDevice::Cpu`.
- `experts.ram_bytes must be greater than 0 on LocalDevice::Cpu`.
- `experts.device_bytes must be greater than 0 on <device>` for a GPU device, with the device as `LocalDevice`'s `Debug` text.

At load, once the file is open, `LocalModel::load` also fails with `CucaError::Config` for:

- A budget below the minimum: ``expert RAM tier of 524288000 bytes holds 4 experts per layer; this model needs at least 8 (1023934464 bytes)``, from `qwen3-30b-a3b-q4_0` at `ram=500m`. A device tier reads ``expert device tier of ...`` instead.
- `warm_from` of another shape: `experts.warm_from is heat for AxB experts; this model has CxD`, with layers by experts.
- A model without experts: ``expert tiers need a mixture-of-experts model; `llama` has no experts``.

`str::parse` on an unknown placement returns ``CucaError::Config("unknown expert placement `...`; expected resident, host[:layers=<n>] or tiered:ram=<size>[,device=<size>][,threads=<n>]")``.

An expert read that fails while generating (a truncated file, for example) fails the request with a `CucaError::Provider` item in the stream.

## Run it

The example reads `CUCA_LOCAL_EXPERTS` and prints the tier counters on stderr after the reply, as a line that starts with `expert tiers: ExpertTierStats {`. `disk_reads` is above 0 once the first token has routed.

```bash,name=Linux and macOS: the example with tiered experts
export RUSTFLAGS="-C target-cpu=native"
export CUCA_LOCAL_MODEL="olmoe-1b-7b-q4_0"
export CUCA_LOCAL_EXPERTS="tiered:ram=600m"
cargo run --release --example local_model --features provider-local
```

```powershell,name=Windows (PowerShell): the example with tiered experts
$env:RUSTFLAGS = "-C target-cpu=native"
$env:CUCA_LOCAL_MODEL = "olmoe-1b-7b-q4_0"
$env:CUCA_LOCAL_EXPERTS = "tiered:ram=600m"
cargo run --release --example local_model --features provider-local
```

The live test `tests/provider_local_tiers.rs` loads the reference placement, then the tiered one cold, then warm from the exported heat. It asserts equal greedy replies (three prompts, 48 tokens), resident bytes within the budgets, and that the warm start saves reads while generating. It drops each model and waits 5 seconds before the next load, so a GPU gives its memory back. It prints `TIERS|reference|...`, `TIERS|cold|...` and `TIERS|warm|...` lines with the tokens per second, the counters and the hit rate. Without a model or without `CUCA_LOCAL_EXPERTS` it prints `SKIP` and passes, `CUCA_REQUIRE_LIVE=1` turns the skip into a failure, and a `CUCA_LOCAL_EXPERTS` that is not tiered fails the test.

```bash,name=Linux and macOS: the tiers test
export RUSTFLAGS="-C target-cpu=native"
export CUCA_LOCAL_MODEL="olmoe-1b-7b-q4_0"
export CUCA_LOCAL_EXPERTS="tiered:ram=500m"
cargo test --release --features provider-local --test provider_local_tiers -- --nocapture
```

```powershell,name=Windows (PowerShell): the tiers test
$env:RUSTFLAGS = "-C target-cpu=native"
$env:CUCA_LOCAL_MODEL = "olmoe-1b-7b-q4_0"
$env:CUCA_LOCAL_EXPERTS = "tiered:ram=500m"
cargo test --release --features provider-local --test provider_local_tiers -- --nocapture
```

- `CUCA_LOCAL_EXPERTS_REF` sets the reference placement, `resident` by default. Set it to a small tiered placement when the resident model does not fit the device.
- `CUCA_LOCAL_DEVICE` selects one device. A GPU run needs its feature and a placement with `device=`, for example `tiered:device=6g,ram=12g` with `--features provider-local-vulkan`.
- `CUCA_LOCAL_GPU_LAYERS` sets `gpu_layers` for every phase, so the CPU layers use the RAM tier and the GPU layers the device tier.
- `tests/provider_local.rs` reads `CUCA_LOCAL_EXPERTS` too, so the whole scenario matrix runs on tiered experts. Unset means resident.

Other variables are in [Local configuration](@/providers/local/configuration.md#environment-variables).
