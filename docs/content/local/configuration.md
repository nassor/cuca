+++
title = "Configuration"
description = "Every LocalModelConfig, SamplingConfig, BatchingConfig and EngramCache key with its default and range, the KDL forms, the cache directory and the local_model example flags."
template = "page.html"
aliases = ["providers/local/configuration/"]
weight = 5
+++

# Configuration

<dl class="page-facts">
<dt>In one line</dt>
<dd>One <code>LocalModelConfig</code> holds every setting of a local model. You fill it from a KDL file or from Rust, and both use the same names.</dd>
<dt>You need</dt>
<dd>The <code>provider-local</code> feature. Entry points are on <a href="@/local/_index.md">Local (in-process)</a>.</dd>
<dt>Read this if</dt>
<dd>You need a key, its default, its valid range, or its KDL spelling.</dd>
</dl>

## KDL configuration

A KDL file is the usual way to configure a model. Each key is the name of a `LocalModelConfig` field. Lines that start with `//` are comments, so the commented keys below are alternatives you can switch on.

```kdl,name=local_model.kdl
// Where the model comes from: a preset name, a model file, or a checkpoint.
source "qwen3-0.6b-q8_0"
// source file="/models/x.gguf"
// source dir="/ckpt"
// source "repo:revision:file"

// Where it runs. A GPU needs its cargo feature, for example provider-local-vulkan.
device "vulkan:0"
// shards "vulkan:1"                 // split the model over more GPUs
// gpu_layers 20                     // run only the last 20 layers on the GPU

context_length 8192                  // prompt plus reply tokens per request
cache_dir "D:/cuca-cache"            // downloads and conversions
hf_token "${HF_TOKEN:-}"             // empty means an anonymous download
// chat_template_file "templates/x.jinja"

sampling temperature=0.7 top_k=40 top_p=0.9 seed=7
batching max_sequences=8 kv_tokens=16384 max_step_tokens=256

// A mixture of experts model: where its experts live.
// experts "resident"
// experts { host; }
// experts { tiered ram_bytes="4g" device_bytes="6g" loader_threads=8; }

// Speculative decoding.
// drafter "in-trunk"
// draft_tokens 5

// DeepSeek-V4.1 and converted checkpoints.
// engram ram_bytes="2g" reader_threads=8
// convert quant="q4_0" engram_quant="q8_0"
// shard_dirs "E:/parts" "F:/parts"
```

Not every combination of keys loads. `LocalModel::load` rejects the impossible ones with a `CucaError::Config`; the list is in [Limits and errors](@/local/limits-and-errors.md#rejected-combinations).

Load the file with `cuca::config::load_kdl_file`. It needs the `config-kdl` feature, which `provider-local` turns on. An unknown key fails with an error naming it.

```rust,name=Load a configuration file
use std::path::Path;
use cuca::LocalModelConfig;

let config: LocalModelConfig = cuca::config::load_kdl_file(Path::new("local_model.kdl"))?;
let model = cuca::LocalModel::load(config).await?;
```

`load_kdl_str` does the same for text you already hold. The Rust form of the same settings fills the struct directly. Every field has a default, so you name only the ones you change.

```rust,name=The same settings in Rust
use cuca::{BatchingConfig, LocalModel, LocalModelConfig, SamplingConfig};

let config = LocalModelConfig {
    source: "qwen3-0.6b-q8_0".parse()?,
    device: "vulkan:0".parse()?,
    context_length: Some(8192),
    cache_dir: Some("D:/cuca-cache".into()),
    hf_token: std::env::var("HF_TOKEN").ok(),
    sampling: SamplingConfig {
        temperature: Some(0.7),
        top_k: Some(40),
        top_p: Some(0.9),
        seed: Some(7),
    },
    batching: BatchingConfig {
        max_sequences: 8,
        kv_tokens: Some(16384),
        max_step_tokens: Some(256),
        ..Default::default()
    },
    ..Default::default()
};
let model = LocalModel::load(config).await?;
```

The `cuca` binary reads the same keys inside the `model` nodes of its `serve` section; see [Model server](@/server/model-server.md).

### KDL spelling

- `source` is a string (a preset name, `"repo:revision:file"` or `"repo:revision"`), `source file="/models/x.gguf"`, `source dir="/ckpt"`, `source { checkpoint repo=".." revision=".."; }` or `source { hugging_face repo=".." revision=".." file=".." sha256=".." { part file=".." sha256=".."; template repo=".." revision=".." file=".."; } }`. The string form never touches the file system. The forms are explained on [Model files and downloads](@/local/model-sources.md).
- `device` is a string such as `"vulkan:0"`. A list of devices is `device "vulkan:0"` plus `shards "vulkan:1" "vulkan:2"`. `shard_dirs` takes one or several directories the same way.
- A byte count (`ram_bytes`, `device_bytes`, `engram ram_bytes`) is an integer or a string with a `k`, `m` or `g` suffix, such as `"512m"` or `"4g"`. The suffixes are powers of 1024 and not case sensitive.
- `${VAR}` and `${VAR:-default}` in the file are replaced before parsing, from a top-level `variables { NAME "value" }` block first, then from the process environment. `hf_token "${HF_TOKEN:-}"` reads the token from the environment and gives an empty token (an anonymous download) when it is unset. A `${VAR}` with no value and no default is a `CucaError::Config`. This is the only place cuca reads the environment for a setting.
- `on_download_progress` has no KDL form. Set it in Rust.
- A string that is no valid device, quantization or source is a `CucaError::Config`. For a source the message lists the preset names.

## LocalModelConfig

All fields are public. `LocalModel::load` checks the configuration before it starts the engine thread and returns `CucaError::Config` for a field outside its range.

| Field | Default | Sets |
|---|---|---|
| `source` | preset `gemma4-e2b-q4_0` | the model to load; see [Model files and downloads](@/local/model-sources.md) |
| `cache_dir` | platform cache directory | where downloads and conversions are kept; see [Cache directory and token](#cache-directory-and-token) |
| `hf_token` | none, anonymous | Hugging Face access token |
| `context_length` | the smaller of 8,192 and the model's maximum | prompt plus reply tokens per request; sizes the K/V pool |
| `sampling` | all fields unset | sampler defaults; see [SamplingConfig](#samplingconfig) |
| `chat_template` | none | template text or built-in name that replaces the model's own template; see [Replace the chat template](@/local/requests.md#replace-the-chat-template) |
| `chat_template_file` | none | path of a file holding that template, read at load |
| `device` | `cpu` | where the forward pass runs; see [Run on a GPU](@/local/gpu.md) |
| `shards` | none | more GPUs to split the model over (tensor parallelism) |
| `gpu_layers` | none | how many layers run on the GPU, the rest on the CPU |
| `experts` | `resident` | where a mixture of experts model keeps its experts |
| `drafter` | none | a draft model for speculative decoding; see [Speculative decoding](@/local/drafting.md) |
| `draft_tokens` | 5 | most tokens the drafter proposes per round, 1 to 16 |
| `convert` | `quant` `q8_0`, `engram_quant` `q8_0` | precision of a converted checkpoint |
| `batching` | see [BatchingConfig](#batchingconfig) | how concurrent requests share the engine |
| `engram` | 4 GiB, 16 threads | the row cache of an Engram table; see [Engram cache](#engram-cache) |
| `kv_export` | none | serve a prompt's K/V rows to a decode node; see below |
| `remote_stage` | none | a worker that runs the first layers of the model; one node per worker |
| `stage` | none | makes this model a worker that holds only some layers |
| `shard_dirs` | none | extra directories searched for the parts of a split GGUF |
| `on_download_progress` | none | download and conversion progress observer; Rust only |

- `context_length`: an explicit value must be at least 1 and at most the model's own maximum, or the load fails with `context_length N exceeds the model maximum M`. The K/V memory it costs depends on the model; see [Limits and errors](@/local/limits-and-errors.md#how-much-memory-a-model-needs).
- `device`: `cpu`, `vulkan`, `vulkan:<n>` (the n-th discrete GPU), `vulkan:integrated:<n>`, `metal` in the same three forms, `cuda` or `cuda:<n>`, `rocm` or `rocm:<n>`. A GPU needs its cargo feature, or the load fails with `CucaError::Config` naming the feature. See [Run on a GPU](@/local/gpu.md).
- `shards`: the model runs on `1 + shards.len()` devices, `device` being the first. Every device is a GPU of the same backend, there is no `drafter`, `experts` is `resident`, and the model must be splittable. See [Run on a GPU](@/local/gpu.md).
- `kv_export`: lets a prefill node hand a prompt's K/V rows to a decode node on another machine that runs the same model file. It takes `max_exports`, the exports served at once, 1 to 16, default 2; write it bare for the default. The rows of an export stay in the K/V pool until the peer has read them, so a small `max_exports` bounds the pool they can hold. It cannot be combined with `shards`. The server route and the decode side are on [Model server](@/server/model-server.md#k-v-transfer-between-servers). `LocalModel::kv_export` and `LocalModel::kv_import` are the library calls.
- `remote_stage`: one node per worker, in layer order from layer 0: `remote_stage url="http://box-a:8000" layers=14`. This model runs the layers left over and the output head on its `device`, so the workers' `layers` must sum to less than the model's layer count. Other keys: `model` (the worker's model name when its server serves several), `api_key` (sent as a bearer key; `Debug` prints it as `<redacted>`), `timeout` (one call, 1 s to 1 h, default `"600s"`) and `carry` (the number type the hidden states travel in, `"f32"` by default, or `"bf16"` or `"f16"`, see below). At most 16 stages, and `url` starts with `http://` or `https://`.
- `remote_stage carry`: `"f32"` is bit-exact, so the chain answers what one process does. `"bf16"` and `"f16"` send every float of the hidden states, per-layer inputs and mirrored K/V rows in two bytes, which halves those bytes (what a worker replies and what a later worker receives; the first worker receives no hidden states, and token ids, block tables and headers do not shrink), **but they are not bit-exact**: the values are rounded (8 and 11 significant bits), the logits differ from the whole model's, and a greedy reply can differ where two logits are close. With quantized weights the engine also quantizes activations to 8 bits for its matmuls, so any change to the carry, down to one bit, moves the logits by about the same amount: on Qwen3 0.6B and Qwen 3.5 0.8B (`q8_0` files, the only weights measured) `"f16"`, whose rounding is eight times finer, was not measurably closer to `"f32"` than `"bf16"`. With float weights (F16, BF16, F32 files) the matmuls run in float and `"f16"` should show its finer rounding; that was not measured. `"f16"` holds values up to 65504 only: a larger one fails the call (use `"bf16"`). Use them where the link is slow; on loopback the saving is not measurable. Each worker has its own `carry`; the coordinator's `open` tells the worker, which answers in the same type. The numbers are in the changelog.
- `stage`: `stage first=0 layers=14` makes the model a worker that holds those layers. It serves stage calls and no chat, and a model is either a worker or a coordinator, not both.
- Layer stages cannot be combined with `shards`, `gpu_layers` or `kv_export`. A worker takes no `drafter`. A coordinator takes the Gemma 4 assistant, the in-trunk head and the EAGLE drafter, with the drafter and the last stage on the CPU; the block drafters (DFlash, DSpark) are refused, and so is any drafter on a model whose layers hold recurrent or compressor state (Qwen 3.5, DeepSeek-V4.1), until the workers can roll that state back. See [Speculative decoding](@/local/drafting.md#layer-stages). The load fails with `CucaError::Config` when the workers hold every layer, or when a stage boundary cuts a model between layers that share compressed rows or indexer picks; the message lists the boundaries the model allows. See [Layer stages](@/server/model-server.md#layer-stages).
- `gpu_layers`: runs the last `n` layers and the output head on `device`, the rest on the CPU. It needs a GPU `device`, no `shards` and no `drafter`, and `n` may not exceed the model's layer count. `0` loads the model on the CPU. `LocalModel::layer_split()` reports the split. See [Fit a large model](@/local/placement.md).
- `experts`: matters only for a mixture of experts model. KDL forms are `experts "resident"`, `experts { host; }`, `experts { host layers=4; }` and `experts { tiered ram_bytes="4g"; }`. `host` needs a GPU. `tiered` needs a RAM budget on the CPU, and on a GPU a device budget, a RAM budget or both. See [Tiered experts](@/local/expert-tiers.md) and [Fit a large model](@/local/placement.md).
- `drafter`: `drafter "in-trunk"` uses the prediction head inside the model's own file and needs the CPU. Any source form of `source` names a separate drafter file. The pairings and limits are on [Speculative decoding](@/local/drafting.md).
- `chat_template` also replaces a template a preset pins, and it wins over `chat_template_file`.
- `convert` applies to checkpoint sources only: `source dir=".."`, `source { checkpoint ..; }` and the `"repo:revision"` string. `quant` is `f16`, `q8_0`, `q4_0`, `q4_k_s`, `q4_k_m`, `q5_k_m` or `q6_k`. `engram_quant` is `q8_0`, `q4_0` or `bf16` and applies to DeepSeek-V4.1's Engram table only. A non-default value converts and caches next to the default one. See [Model files and downloads](@/local/model-sources.md) and [Run DeepSeek-V4.1](@/local/deepseek-v41.md).
- `shard_dirs`: searched after the directory of the named first part, for a split GGUF whose parts sit on different volumes. See [Model files and downloads](@/local/model-sources.md).
- `Debug` for `LocalModelConfig` prints `hf_token` as `<redacted>`.

## SamplingConfig

`SamplingConfig` sets the sampler defaults for requests that do not set their own.

| Field | Default | Range |
|---|---|---|
| `temperature` | unset, built-in `1.0` | finite and at least 0; `0.0` is greedy |
| `top_k` | unset, built-in `64` | `0` turns the filter off |
| `top_p` | unset, built-in `0.95` | above 0 up to 1; `1.0` turns the filter off |
| `seed` | unset | any `u64` |

The fields are optional so that "set" and "default" differ: `top_k: Some(64)` beats a model file that says 20, and `None` does not. `temperature`, `top_k` and `top_p` each resolve on their own, taking the first value that is set.

<div class="dgm">
<div class="dgm-scroll">
<svg viewBox="0 0 660 156" role="img" aria-labelledby="samp-title samp-desc">
  <title id="samp-title">Where a sampling value comes from, highest priority first</title>
  <desc id="samp-desc">Four boxes from left to right. 1: the request, with temperature, top_k, top_p and seed. 2: the SamplingConfig in your configuration. 3: the model file's general.sampling keys for temperature, top_k and top_p. 4: the built-in defaults, temperature 1.0, top_k 64 and top_p 0.95. Each value takes the first one that is set. The seed comes only from the request or the configuration.</desc>
  <defs>
    <marker id="samp-arrow" viewBox="0 0 8 8" refX="7" refY="4"
            markerWidth="6" markerHeight="6" orient="auto">
      <path d="M0 0.6 L7 4 L0 7.4 z" fill="var(--muted-foreground)"/>
    </marker>
  </defs>
  <text class="t-title" x="10" y="20">Sampling precedence</text>
  <rect class="blk blk-data" x="10" y="38" width="136" height="80" rx="6"/>
  <text class="t-lbl t-mid t-data" x="78" y="60">1 Request</text>
  <text class="t-sm t-mid" x="78" y="80">temperature, top_k,</text>
  <text class="t-sm t-mid" x="78" y="94">top_p, seed</text>
  <path class="arw" d="M148 78 H176" marker-end="url(#samp-arrow)"/>
  <rect class="blk blk-ctl" x="178" y="38" width="136" height="80" rx="6"/>
  <text class="t-lbl t-mid t-ctl" x="246" y="60">2 SamplingConfig</text>
  <text class="t-sm t-mid" x="246" y="80">the sampling key</text>
  <text class="t-sm t-mid" x="246" y="94">of your config</text>
  <path class="arw" d="M316 78 H344" marker-end="url(#samp-arrow)"/>
  <rect class="blk" x="346" y="38" width="136" height="80" rx="6"/>
  <text class="t-lbl t-mid" x="414" y="60">3 Model file</text>
  <text class="t-sm t-mid" x="414" y="80">general.sampling.*</text>
  <text class="t-sm t-mid" x="414" y="94">temp, top_k, top_p</text>
  <path class="arw" d="M484 78 H512" marker-end="url(#samp-arrow)"/>
  <rect class="blk" x="514" y="38" width="136" height="80" rx="6"/>
  <text class="t-lbl t-mid" x="582" y="60">4 Built-in</text>
  <text class="t-sm t-mid" x="582" y="80">temperature 1.0</text>
  <text class="t-sm t-mid" x="582" y="94">top_k 64, top_p 0.95</text>
  <text class="t-ax" x="10" y="144">each value takes the first one that is set, left to right</text>
</svg>
</div>
<p class="dgm-cap"><b>The request wins, the built-in default comes last.</b> The seed has only two sources, the request and the configuration. <code>LocalModel::sampling()</code> returns the values the load resolved.</p>
</div>

- The file defaults are the GGUF keys `general.sampling.temp`, `general.sampling.top_k` and `general.sampling.top_p`, the names llama.cpp reads. The checkpoint converter writes them from `generation_config.json`.
- A file value the config would reject (a negative temperature, a `top_p` outside 0 to 1) is ignored. A `top_k` of 0 or less turns the filter off.
- The other `general.sampling.*` keys (`min_p`, repetition penalties, XTC, mirostat) name samplers the engine does not have, so they are not read.
- With `seed: Some(n)`, every request restarts the generator from `n`, so equal prompts sample equal tokens. A request `seed` does the same for that request alone and wins over the config seed. With no seed, the generator is seeded from the system once at load.
- The sampler applies temperature, top-k, softmax and top-p in that order, then draws one token. Temperature `0.0` takes the first maximum.

## BatchingConfig

`BatchingConfig` decides how many requests run at once and how big the shared K/V pool is. The K/V cache is the memory where the engine keeps each request's attention state so it does not recompute earlier tokens. In KDL: `batching max_sequences=8 kv_tokens=16384 max_step_tokens=256 prefix_cache=#false`, any subset of the keys.

| Field | Default | Sets |
|---|---|---|
| `max_sequences` | 4 | requests that run at once, 1 to 64 |
| `kv_tokens` | CPU: twice `context_length`<br>GPU: `context_length` plus at most 4,096 | positions in the shared K/V pool |
| `max_step_tokens` | the prefill chunk | rows of one forward step across all requests |
| `prefix_cache` | `true` | keep the K/V of finished requests for later prompts |

- `kv_tokens` is the K/V memory knob. An explicit value is at least `context_length + 16` and rounds up to whole blocks of 16. Raising `max_sequences` does not raise it. A pool too small for the running requests makes them wait or preempt each other. `LocalModel::kv_cache_bytes` reports the result.
- `max_step_tokens` is at least `max_sequences` and at most 8,192. The default prefill chunk is `2,097,152 / context_length`, clamped to 16 to 256: 256 at 8,192 positions. With tiered experts under DeepSeek-V4.1 it is 4,096.
- `prefix_cache: false` registers nothing in the prefix store, so every prompt prefills in full.
- A drafter that reads every position of its sequence (the in-trunk head, EAGLE, DFlash and DSpark) keeps one cache per request slot, so its cache memory grows with `max_sequences`. Every running request drafts in the shared step, except in a step where a request prefills.
- `LocalModel::serving_stats()` returns a `ServingStats` snapshot. The gauges are `running`, `waiting`, `max_sequences`, `free_blocks`, `cached_blocks`, `used_blocks`, `total_blocks` and `snapshot_bytes`. The totals since load are `prefix_hit_tokens`, `preemptions`, `steps`, `batched_rows`, `prefill_rows`, `generated_tokens`, `admissions`, `queue_wait_ns` and `step_ns`, and `last_step_ns` and `last_step_rows` describe the last pass. Reading them never waits for a step. Subtract two snapshots to get a rate.

### When to change kv_tokens

The pool holds the K/V of every running request plus the prompts of finished requests, kept for reuse. One request of the full `context_length` always fits. The pool's size never changes how fast a token is computed; a pool that runs short only makes the engine redo work. A GPU starts smaller than the CPU because device memory is the tighter limit: on a 16 GB card the CPU default would add 0.8 GB to the 2.42 GB pool of `qwen3-30b-a3b-iq2_xxs`, more than is left beside its weights.

Read two counters of `serving_stats()` while the model serves your real traffic:

- `preemptions` grows: several long requests ran at once and did not fit, so the youngest one was paused and reads its prompt again later. Raise `kv_tokens`, or lower `max_sequences`.
- `prefix_hit_tokens` grows little although prompts repeat (the same system prompt, tools or conversation sent again): their cached K/V was evicted for other prompts. Raise `kv_tokens`.

Lower `kv_tokens` or `context_length` when a GPU load fails with `the device failed while the model loaded (most likely out of memory: ...)`, or when RAM is short. A K/V cache costs the same per position on every device; the per-preset figures for both defaults are in the [presets table](@/local/models.md#the-presets).

```kdl,name=Pin the pool to the CPU default on a GPU
device "vulkan"
batching kv_tokens=16384
```

## Engram cache

Only DeepSeek-V4.1 has an Engram table. It is too large for memory, so the engine reads the rows it needs from the model file with a pool of reader threads and keeps the recently used rows in a cache. A model without an Engram table ignores this setting. In KDL: `engram ram_bytes="8g" reader_threads=32`, any subset.

| Field | Default | Range |
|---|---|---|
| `ram_bytes` | 4 GiB | 1 MiB to 1 TiB |
| `reader_threads` | 16 | 1 to 64 |

- A smaller cache is slower, never different: rows evicted before use are read again, and the output stays the same.
- The cache grows as it fills, so a short session does not pay for the bound.
- `LocalModel::engram_stats()` returns `EngramStats` (`lookups`, `cache_hits`, `disk_reads`, `bytes_read`, `resident_rows`, `resident_bytes`, `capacity_rows`, and `hit_rate()`), or `None` for a model without an Engram table.
- A value outside the range fails the load; the messages are on [Limits and errors](@/local/limits-and-errors.md#errors).

## Cache directory and token

- `cache_dir: None` is the platform cache directory: `%LOCALAPPDATA%\cuca` on Windows (`%USERPROFILE%\AppData\Local\cuca` when that variable is unset), `~/Library/Caches/cuca` on macOS, and `$XDG_CACHE_HOME/cuca` else `~/.cache/cuca` elsewhere. When no directory can be resolved, `load` fails with `CucaError::Config` asking for `cache_dir`.
- `hf_token: None` or an empty string downloads anonymously. The library reads no `HF_TOKEN` variable; a KDL file can take it from the environment with `hf_token "${HF_TOKEN:-}"`. A token is sent as a bearer token on downloads.

## The local_model example

`examples/local_model.rs` takes the request on the command line and everything about the model from a `LocalModelConfig` file. `examples/configs/local_model.kdl` lists every key, all but `source` commented out.

| Flag | Meaning |
|---|---|
| `--config <FILE>` | the KDL file; without it the default `LocalModelConfig` applies |
| `--prompt <TEXT>` | the user message; default `Explain CUCA in one sentence.` |
| `--max-tokens <N>` | `max_tokens`; default `1024` |
| `--temperature <T>` | sampling temperature; `0` is greedy; default the model's own |
| `--thinking off` | disables thinking; any other value or none leaves it enabled |

```bash,name=Linux and macOS
RUSTFLAGS="-C target-cpu=native" cargo run --release --example local_model --features provider-local -- --config examples/configs/local_model.kdl --prompt "Say hi"
```

```powershell,name=Windows (PowerShell)
$env:RUSTFLAGS = "-C target-cpu=native"
cargo run --release --example local_model --features provider-local -- --config examples/configs/local_model.kdl --prompt "Say hi"
```

On stderr the example prints `load: <seconds> s` and `loaded on <device>` before the reply, and a `timing:` line after it with the prompt tokens, the time to the first block and the tokens per second. The reply text goes to stdout and depends on the model.

To run the live tests against a model, see [Integration tests](@/guides/integration-tests.md).
