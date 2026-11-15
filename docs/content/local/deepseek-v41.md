+++
title = "Run DeepSeek-V4.1"
description = "Run DeepSeek-V4.1 Flash on one PC: the disk, RAM and SSD it needs, the config, the first-run conversion, the speed to expect, and the Engram cache, drafter and GPU settings."
template = "page.html"
aliases = ["providers/local/deepseek-v41/"]
weight = 10
+++

# Run DeepSeek-V4.1

<dl class="page-facts">
<dt>In one line</dt>
<dd>DeepSeek-V4.1 Flash is a 510 GB model. CUCA downloads and converts it once, then keeps its experts on your SSD behind a RAM cache, so it runs on one PC.</dd>
<dt>You need</dt>
<dd>The <code>provider-local</code> feature (or <code>provider-local-vulkan</code> for a GPU), about 1 TB of free disk, about 90 GB of RAM and, for a usable speed, an NVMe SSD. The commands run from a checkout of this repository.</dd>
<dt>Read this if</dt>
<dd>You want the largest model the engine supports to answer on your own machine.</dd>
</dl>

DeepSeek-V4.1 is a mixture of experts: each layer holds 384 small expert networks and a token uses 6 of them. The experts are most of the file. Keeping them all in RAM needs about 289 GB, so the config below leaves them on the SSD and caches the busy ones in RAM. That setting is called tiered experts.

## What you need

- Disk: about 1 TB free on the cache volume. The download is 510 GB of shards. The converted set that later runs read is about 507 GB. The shards stay in the cache after the conversion, and you may delete them: a finished set loads without them and without the network.
- RAM: about 90 GB for the expert cache. A smaller `ram_bytes` also works and reads more from disk. At 90g each layer caches 128 of its 384 experts, at 64g 91 and at 48g 68.
- Storage speed: the experts are read from disk while the model answers. An NVMe SSD gives about 2 tokens per second. A hard disk gives about 0.2.
- Processor: the CPU, or one GPU. A GPU gains little here, see [What to expect](#what-to-expect).
- Model source: DeepSeek-V4.1 has no preset. It loads as a checkpoint, a `repo:revision` string. See [Model files and downloads](@/local/model-sources.md).

## 1. Write the config

```kdl,name=deepseek.kdl
source "deepseek-ai/DeepSeek-V4.1-Flash-1017:1bae0e1088993b9b82de08f59de5be5a34e42170"
experts { tiered ram_bytes="90g" loader_threads=32; }
engram ram_bytes="4g" reader_threads=16
```

- `source` names the checkpoint and pins it to a commit. CUCA downloads it into the cache and converts it.
- `experts { tiered ... }` keeps the experts in the file. `ram_bytes` is the RAM cache for them, and `loader_threads` is how many threads read them from disk (1 to 64). The sizes take a `k`, `m` or `g` suffix.
- `engram` sets the cache for the Engram table, see [The Engram cache](#the-engram-cache). These are its defaults.
- To put the files on a bigger volume, add `cache_dir "D:/cuca-cache"` with your own path.

The same setup in Rust:

```rust,name=DeepSeek-V4.1 in code (provider-local)
use cuca::{EngramCache, ExpertPlacement, ExpertTiers, LocalModel, LocalModelConfig};

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = LocalModel::load(LocalModelConfig {
        source: "deepseek-ai/DeepSeek-V4.1-Flash-1017:1bae0e1088993b9b82de08f59de5be5a34e42170"
            .parse()?,
        experts: ExpertPlacement::Tiered(ExpertTiers {
            ram_bytes: 90 << 30,
            loader_threads: 32,
            ..ExpertTiers::default()
        }),
        engram: EngramCache { ram_bytes: 4 << 30, reader_threads: 16 },
        ..Default::default()
    })
    .await?;
    println!("{}", model.context_length());
    Ok(())
}
```

```text,name=Expected output
8192
```

## 2. Run it

<div class="dgm">
<div class="dgm-scroll">
<svg viewBox="0 0 660 124" role="img" aria-labelledby="dsf-title dsf-desc">
  <title id="dsf-title">The first run downloads and converts, every run loads and answers</title>
  <desc id="dsf-desc">Four boxes, left to right. Step 1 downloads 510 GB of shards into the cache. Step 2 converts them and writes the 507 GB set, limited by disk speed and taking hours. These two steps happen on the first run only. Step 3 opens the set and takes 6 to 11 seconds from an NVMe SSD. Step 4 answers, with the first token after about 20 to 30 seconds for a short request. These two steps happen on every run.</desc>
  <defs>
    <marker id="dsf-arrow" viewBox="0 0 8 8" refX="7" refY="4"
            markerWidth="6" markerHeight="6" orient="auto">
      <path d="M0 0.6 L7 4 L0 7.4 z" fill="var(--muted-foreground)"/>
    </marker>
  </defs>
  <text class="t-title" x="6" y="20">First run and later runs</text>
  <path class="ln" d="M6 54 V48 H312 V54"/>
  <text class="t-ax t-mid" x="159" y="42">FIRST RUN ONLY</text>
  <path class="ln" d="M342 54 V48 H648 V54"/>
  <text class="t-ax t-mid" x="495" y="42">EVERY RUN</text>
  <rect class="blk blk-bnd" x="6" y="62" width="138" height="52" rx="6"/>
  <text class="t-lbl t-mid t-bnd" x="75" y="81">1 download</text>
  <text class="t-sm t-mid" x="75" y="96">510 GB of shards</text>
  <text class="t-sm t-mid" x="75" y="108">into the cache</text>
  <rect class="blk blk-ctl" x="174" y="62" width="138" height="52" rx="6"/>
  <text class="t-lbl t-mid t-ctl" x="243" y="81">2 convert</text>
  <text class="t-sm t-mid" x="243" y="96">writes the 507 GB set</text>
  <text class="t-sm t-mid" x="243" y="108">disk bound, hours</text>
  <rect class="blk blk-data" x="342" y="62" width="138" height="52" rx="6"/>
  <text class="t-lbl t-mid t-data" x="411" y="81">3 load</text>
  <text class="t-sm t-mid" x="411" y="96">opens the set</text>
  <text class="t-sm t-mid" x="411" y="108">6 to 11 s on an SSD</text>
  <rect class="blk blk-data" x="510" y="62" width="138" height="52" rx="6"/>
  <text class="t-lbl t-mid t-data" x="579" y="81">4 answer</text>
  <text class="t-sm t-mid" x="579" y="96">first token after</text>
  <text class="t-sm t-mid" x="579" y="108">about 20 to 30 s</text>
  <path class="arw" d="M146 88 H170" marker-end="url(#dsf-arrow)"/>
  <path class="arw" d="M314 88 H338" marker-end="url(#dsf-arrow)"/>
  <path class="arw" d="M482 88 H506" marker-end="url(#dsf-arrow)"/>
</svg>
</div>
<p class="dgm-cap"><b>Only the first run is long.</b> The conversion is limited by disk speed: it took about 2 h 44 min on a hard disk that wrote about 52 MB/s. Later runs open the converted set.</p>
</div>

```bash,name=Linux and macOS: ask a quick question
RUSTFLAGS="-C target-cpu=native" cargo run --release --example local_model --features provider-local -- --config deepseek.kdl --prompt "What is the capital of France? Answer in one word." --thinking off --temperature 0
```

```powershell,name=Windows (PowerShell): ask a quick question
$env:RUSTFLAGS = "-C target-cpu=native"
cargo run --release --example local_model --features provider-local -- --config deepseek.kdl --prompt "What is the capital of France? Answer in one word." --thinking off --temperature 0
```

`--thinking off` skips the model's reasoning and `--temperature 0` makes the answer repeatable.

```text,name=Expected output (timings vary)
load: <seconds> s
loaded on cpu
Paris
timing: 21 prompt tokens, first block after <s> s (~<r> prompt tokens/s); 1 completion tokens (<n> blocks), then ...
```

- On the first run stderr shows a `<n> / Some(<total>) bytes` counter for each file it downloads, then one for the conversion, which counts bytes written out of bytes to write.
- A later run prints no counter. Loading takes 6 to 11 seconds from an NVMe SSD. On the CPU from a hard disk it takes 40 to 75 seconds.
- Leave `--thinking off` out and the model reasons first. For `What is 9 * 8?` it reasons briefly and answers `72`. Thinking, tools and the request fields are on [Requests, thinking and tool calls](@/local/requests.md).

## 3. Use a GPU (optional)

A GPU runs the attention, the router, the shared experts and the head. The CPU still computes the experts, so the speed is close to the CPU's.

```kdl,name=deepseek-gpu.kdl
source "deepseek-ai/DeepSeek-V4.1-Flash-1017:1bae0e1088993b9b82de08f59de5be5a34e42170"
device "vulkan"
experts { tiered ram_bytes="90g" loader_threads=32; }
engram ram_bytes="4g" reader_threads=16
```

```bash,name=Linux and macOS: run on Vulkan
RUSTFLAGS="-C target-cpu=native" cargo run --release --example local_model --features provider-local-vulkan -- --config deepseek-gpu.kdl --prompt "What is the capital of France? Answer in one word." --thinking off --temperature 0
```

```powershell,name=Windows (PowerShell): run on Vulkan
$env:RUSTFLAGS = "-C target-cpu=native"
cargo run --release --example local_model --features provider-local-vulkan -- --config deepseek-gpu.kdl --prompt "What is the capital of France? Answer in one word." --thinking off --temperature 0
```

The reply is `Paris` again, and the `loaded on` line names the device. This setup used 12.5 GiB of VRAM at its peak on a 16 GB card. It uses one GPU: a model split over several GPUs (`shards`) is rejected. GPU setup is on [Run on a GPU](@/local/gpu.md).

## Where the parts live

<div class="dgm">
<div class="dgm-scroll">
<svg viewBox="0 0 660 232" role="img" aria-labelledby="dsw-title dsw-desc">
  <title id="dsw-title">Where the parts of DeepSeek-V4.1 live while it runs</title>
  <desc id="dsw-desc">Five boxes. The SSD holds the converted set: part 1 with the metadata, part 2 of 298 GB with the dense weights and all experts, part 3 of 209 GB with the Engram table, and an optional 7.9 GB drafter file. Reader threads read experts and Engram rows from the SSD into RAM. RAM holds the expert tier of 90 GB, the Engram row cache of 4 GB, and the dense weights on a CPU run. The CPU computes every expert from RAM. An optional GPU runs the attention, the router, the shared experts and the head in about 12.5 GiB of VRAM and exchanges the hidden state with the CPU.</desc>
  <defs>
    <marker id="dsw-arrow" viewBox="0 0 8 8" refX="7" refY="4"
            markerWidth="6" markerHeight="6" orient="auto">
      <path d="M0 0.6 L7 4 L0 7.4 z" fill="var(--muted-foreground)"/>
    </marker>
  </defs>
  <text class="t-title" x="6" y="20">Disk, RAM and the processors</text>
  <rect class="blk" x="6" y="40" width="190" height="182" rx="6"/>
  <text class="t-lbl t-mid" x="101" y="62">SSD, the converted set</text>
  <text class="t-sm t-mid" x="101" y="86">part 1: metadata only</text>
  <text class="t-sm t-mid" x="101" y="108">part 2: 298 GB</text>
  <text class="t-sm t-mid" x="101" y="122">dense weights, experts</text>
  <text class="t-sm t-mid" x="101" y="146">part 3: 209 GB</text>
  <text class="t-sm t-mid" x="101" y="160">Engram table</text>
  <text class="t-sm t-mid" x="101" y="184">optional drafter: 7.9 GB</text>
  <rect class="blk blk-ctl" x="236" y="40" width="176" height="70" rx="6"/>
  <text class="t-lbl t-mid t-ctl" x="324" y="62">reader threads</text>
  <text class="t-sm t-mid" x="324" y="82">loader_threads: experts</text>
  <text class="t-sm t-mid" x="324" y="96">reader_threads: Engram</text>
  <rect class="blk blk-data" x="236" y="142" width="176" height="80" rx="6"/>
  <text class="t-lbl t-mid t-data" x="324" y="164">RAM</text>
  <text class="t-sm t-mid" x="324" y="184">expert tier, 90g</text>
  <text class="t-sm t-mid" x="324" y="198">Engram row cache, 4g</text>
  <text class="t-sm t-mid" x="324" y="212">dense weights, CPU run</text>
  <rect class="blk blk-ctl" x="470" y="40" width="184" height="56" rx="6"/>
  <text class="t-lbl t-mid t-ctl" x="562" y="62">CPU</text>
  <text class="t-sm t-mid" x="562" y="80">computes every expert</text>
  <rect class="blk blk-ctl" x="470" y="142" width="184" height="80" rx="6"/>
  <text class="t-lbl t-mid t-ctl" x="562" y="164">GPU, optional</text>
  <text class="t-sm t-mid" x="562" y="184">attention, router,</text>
  <text class="t-sm t-mid" x="562" y="198">shared experts, head</text>
  <text class="t-sm t-mid" x="562" y="212">12.5 GiB VRAM peak</text>
  <path class="arw arw-data" d="M198 75 H232" marker-end="url(#dsw-arrow)"/>
  <path class="arw arw-data" d="M324 112 V138" marker-end="url(#dsw-arrow)"/>
  <path class="arw arw-data" d="M414 170 H442 V68 H466" marker-end="url(#dsw-arrow)"/>
  <text class="t-ax t-end t-data" x="438" y="126">experts</text>
  <path class="arw arw-data" d="M520 98 V138" marker-end="url(#dsw-arrow)"/>
  <path class="arw arw-data" d="M548 138 V98" marker-end="url(#dsw-arrow)"/>
  <text class="t-ax t-data" x="562" y="124">hidden state</text>
</svg>
</div>
<p class="dgm-cap"><b>The SSD feeds RAM, and the CPU computes the experts from RAM.</b> A GPU takes the attention and the other dense parts. On a CPU-only run the dense weights sit in RAM too.</p>
</div>

## What conversion writes

The converted files go to `<cache>/converted/deepseek-ai--DeepSeek-V4.1-Flash-1017/<revision>/`:

- `model-q8_0-00001-of-00003.gguf` holds the metadata and no tensor. You name this file when you move the set.
- `model-q8_0-00002-of-00003.gguf` holds every tensor except the Engram table, 297.96 GB.
- `model-q8_0-00003-of-00003.gguf` holds the Engram table, 208.90 GB.
- `dspark-model-q8_0.gguf`, 7.91 GB, is the drafter's file. It is written only when a `drafter` names the same checkpoint, see [The DSpark drafter](#the-dspark-drafter).

Part 1 is written last, so a present part 1 means a finished set. A later load reuses the set with no network and no shards.

- The dense weights are stored at 8 bits (`q8_0`) by default and the experts keep their native 4-bit format (MXFP4). `convert quant` also takes `q4_0`, `f16` and the K-quant mixes `q4_k_s`, `q4_k_m`, `q5_k_m` and `q6_k`, which apply to the dense weights; the experts stay MXFP4 except at `f16`.
- `convert engram_quant="q4_0"` shrinks the Engram part to about 110.6 GB, so the set is about 409 GB. The default `q8_0` is 208.90 GB, and `bf16` is about 393.2 GB. A non-default value converts into its own files next to the default ones.
- The conversion checks the free space of the output volume first. It needs little RAM, about 220 MB at its peak.

If the set is on a hard disk, move it to the SSD and name part 1 with `source file=`. The other parts are looked for next to it, and `shard_dirs` names more folders if the parts sit on different volumes.

```bash,name=Linux and macOS: copy the converted set to the SSD
rsync -a /mnt/hdd/cuca-cache/converted/ /mnt/ssd/cuca-cache/converted/
```

```powershell,name=Windows (PowerShell): copy the converted set to the SSD
robocopy D:\cuca-cache\converted C:\cuca-cache\converted *.gguf /E /J /NP
```

```kdl,name=deepseek-ssd.kdl: open the set from the SSD
source file="C:/cuca-cache/converted/deepseek-ai--DeepSeek-V4.1-Flash-1017/1bae0e1088993b9b82de08f59de5be5a34e42170/model-q8_0-00001-of-00003.gguf"
experts { tiered ram_bytes="90g" loader_threads=32; }
engram ram_bytes="4g" reader_threads=16
```

On Linux and macOS write the same key with your path, for example `source file="/mnt/ssd/cuca-cache/converted/deepseek-ai--DeepSeek-V4.1-Flash-1017/1bae0e1088993b9b82de08f59de5be5a34e42170/model-q8_0-00001-of-00003.gguf"`.

## What to expect

| Setup | First token, short request | Decode | Reading a long prompt | Memory |
|---|---|---|---|---|
| CPU, NVMe SSD | about 21 s | 2.15 tokens/s | 13.7 tokens/s (2,014 tokens) | 98.9 GiB of RAM |
| Vulkan GPU, NVMe SSD | about 32 s | 1.96 tokens/s | 11.8 tokens/s (1,752 tokens) | 90.4 GiB of RAM, 12.5 GiB of VRAM |
| CPU, hard disk | about 6 min | 0.2 tokens/s | 0.1 to 0.2 tokens/s (42 tokens) | about 87 GB of RAM |

Method: the default-precision set with `experts { tiered ram_bytes="90g" loader_threads=32; }` (16 threads on the hard disk) and greedy decoding. Each process started with an empty RAM cache. The short request is 42 prompt tokens with thinking on. Decode is the speed after the first token, over 200 reply tokens (17 on the hard disk). Machine: Ryzen 9 9950X3D2 with 125 GiB of RAM, RX 9070 XT with 16 GB, Windows, a release build with `-C target-cpu=native`. The SSD reads up to 3.5 GB/s and the hard disk about 0.4 to 0.5 GB/s.

- Every token reads the experts that are not in the RAM cache from disk. The cache starts empty in every process, so the first requests are the slowest.
- With every needed expert already in RAM, decoding runs at about 2.4 to 3.2 tokens/s on the CPU and on Vulkan. That is the ceiling.
- A GPU gains at most about 1.2 times, because the CPU still computes the experts and the disk is the limit.
- 8 to 64 `loader_threads` gave the same speed within the run-to-run spread on the test SSD.
- A long prompt is limited by computing, not by the SSD, once the SSD is fast.

## The Engram cache

DeepSeek-V4.1 has a lookup table that the model reads by hashing each token with the tokens before it, the Engram table. It is 208.90 GB at `q8_0`. Each token reads 24 rows of 272 bytes per Engram layer. CUCA never loads the table into RAM. Reader threads read the rows a pass needs from the file, and a cache keeps the most recently used rows.

| Key | Default | Range |
|---|---|---|
| `engram ram_bytes` | 4 GiB | 1 MiB to 1 TiB |
| `engram reader_threads` | 16 | 1 to 64 |

Cache sizes of 1g, 4g and 16g gave the same first-token time within about 4 seconds, so the default is fine. A smaller cache gives the same output and only changes speed. `LocalModel::engram_stats()` returns the cache counters.

## The DSpark drafter

A drafter guesses the next few tokens, and the main model checks them in one pass. DeepSeek-V4.1's drafter is called DSpark. It is stored in the same checkpoint, so you name the same source:

```kdl,name=Add the DSpark drafter to deepseek.kdl
drafter "deepseek-ai/DeepSeek-V4.1-Flash-1017:1bae0e1088993b9b82de08f59de5be5a34e42170"
```

The first run then also writes `dspark-model-q8_0.gguf`. The output equals plain decoding, so the drafter changes speed only. While the disk is the limit it gives no speedup: 1.53 tokens/s with the drafter against 1.58 without (CPU, NVMe SSD, 48 tokens). Checking extra tokens reads extra experts from disk.

- It runs on the CPU only. A GPU `device` fails with ``the DSpark drafter runs on the CPU only: use `LocalDevice::Cpu` ``, and `gpu_layers` fails the load.
- It loads its own 7.2 GB of experts into RAM when it opens.
- It serves one request at a time. `draft_tokens` (default 5, 1 to 16) caps the tokens per round, and the real file's block is 5. More on drafters is on [Speculative decoding](@/local/drafting.md).

## Put layers on the GPU with gpu_layers

`gpu_layers 20` runs the last 20 of the 40 layers on the GPU and the rest on the CPU. DeepSeek-V4.1 accepts only these values: 0, 20, 26, 32, 38, 39 and 40. Layers that share compressed attention data must stay on one side, and any other value fails the load:

```text,name=Error for another gpu_layers value
gpu_layers N cuts `deepseek41` between layers that share compressed rows or indexer picks; legal values: ...
```

Tiered experts with `gpu_layers` need a device tier, so set `device_bytes` above 0:

```kdl,name=20 layers on the GPU
device "vulkan"
gpu_layers 20
experts { tiered device_bytes="3g" ram_bytes="85g" loader_threads=32; }
```

This runs slower than the default. With a 3 GiB device tier it decodes at 0.94 tokens/s and with 6 GiB at 1.08, against 1.96 tokens/s with all layers on the GPU. An 8g device tier with 20 layers fails with `out of device memory` on a 16 GB card. Use `gpu_layers` only when the default's 12.5 GiB of VRAM does not fit: 20 layers with a 3 GiB device tier peaked at 10.2 GiB.

## Common errors

- ``converting ... writes N GB into `DIR`, which has M GB free; free K GB more, choose another cache directory, ...``: set `cache_dir` to a bigger volume, or add `convert engram_quant="q4_0"`.
- ``expert RAM tier of N bytes holds K experts per layer; this model needs at least 6 (4512153600 bytes)``: raise `ram_bytes` to at least 4.5 GB.
- `experts.ram_bytes must be greater than 0 on LocalDevice::Cpu`: give the CPU a `ram_bytes`.
- `experts.device_bytes must be greater than 0 with gpu_layers: host-tiered experts serve a model that runs on one device`: set `device_bytes` with `gpu_layers`, or drop `gpu_layers`.
- ``gpu_layers N cuts `deepseek41` between layers ...``: use 0, 20, 26, 32, 38, 39 or 40.
- `out of device memory`: lower `device_bytes` or `gpu_layers`, or use host-tiered experts with no `gpu_layers`.
- ``the DSpark drafter runs on the CPU only: use `LocalDevice::Cpu` ``: remove `drafter`, or run on the CPU.
- `engram.reader_threads must be in 1..=64, got N`: use 1 to 64.
- `engram.ram_bytes must be in 1048576..=1099511627776 (1 MiB to 1 TiB), got N`: use a size in that range.

Other device, configuration and download errors are on [Limits and errors](@/local/limits-and-errors.md).
