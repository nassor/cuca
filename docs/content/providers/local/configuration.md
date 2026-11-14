+++
title = "Local configuration"
description = "The LocalModelConfig, SamplingConfig and BatchingConfig fields with their defaults, valid ranges and KDL spelling, the cache directory and token rules, and the local_model example's flags."
template = "page.html"
weight = 1
+++

# Local configuration

<dl class="page-facts">
<dt>In one line</dt>
<dd><code>LocalModelConfig</code> holds the model source, cache, context length, sampling, device, drafter, expert placement, Engram cache, batching and offload; <code>SamplingConfig</code> and <code>BatchingConfig</code> hold the sampler and scheduler settings.</dd>
<dt>You need</dt>
<dd>The <code>provider-local</code> feature. Entry points are on <a href="@/providers/local/_index.md">Local (in-process)</a>.</dd>
<dt>Read this if</dt>
<dd>You need a field default, a valid range, or the KDL key of a field.</dd>
</dl>

## LocalModelConfig

All fields are public. `LocalModel::load` validates the configuration before it starts the engine thread and returns `CucaError::Config` for a field outside its range. The struct also deserializes from KDL with the field names as keys; see [KDL configuration](#kdl-configuration).

| Field | Default | Meaning |
|---|---|---|
| `source` | `ModelSource::default_model()` | where the GGUF file comes from; see [Model sources](@/providers/local/model-sources.md) |
| `cache_dir` | `None` | download cache; `None` is the platform cache directory; see [Cache directory and token](#cache-directory-and-token) |
| `hf_token` | `None` | Hugging Face token; `None` or an empty string downloads anonymously |
| `context_length` | `None` | prompt plus completion tokens per request; sizes the K/V pool |
| `sampling` | `SamplingConfig::default()` | sampler defaults you set; see [SamplingConfig](#samplingconfig) |
| `chat_template` | `None` | Jinja text that replaces the GGUF's chat template; see [Chat templates](@/providers/local/chat-templates.md#where-the-template-comes-from) |
| `chat_template_file` | `None` | path of a file holding that Jinja text, read at load |
| `device` | `LocalDevice::Cpu` | where the forward pass runs |
| `drafter` | `None` | `Option<DrafterSource>`: a draft model for speculative decoding |
| `draft_tokens` | `5` | most tokens a drafter proposes per round, `1..=16`; checked only with a drafter |
| `experts` | `ExpertPlacement::Resident` | where a mixture-of-experts model's experts live |
| `convert` | `ConvertConfig::default()` | `quant` and `engram_quant` of a converted checkpoint |
| `batching` | `BatchingConfig::default()` | how concurrent requests share the engine; see [BatchingConfig](#batchingconfig) |
| `shards` | `[]` | more GPUs to split the model over (tensor parallelism) |
| `gpu_layers` | `None` | layers to run on the GPU, the rest on the CPU |
| `engram` | `EngramCache::default()` | row cache and reader threads of an Engram table; see [Engram cache](#engram-cache) |
| `shard_dirs` | `[]` | extra directories searched for the parts of a split GGUF set |
| `on_download_progress` | `None` | download and conversion progress observer |

- `source` defaults to the preset `gemma4-e2b-q4_0`.
- `context_length: None` resolves to the smaller of 8,192 and the model's own maximum. An explicit value must be at least 1 and at most the model's maximum.
- `device` needs the matching cargo feature for a GPU, else `load` fails with `CucaError::Config` naming it. In KDL it is a string such as `device "vulkan:0"`; further GPUs go in `shards`. See [Devices and GPU features](@/providers/local/gpu.md#devices-and-gpu-features).
- `chat_template` also replaces the template a preset pins. Everything derived from the template follows it. `chat_template_file` is the same override read from a file: `chat_template` wins over it, and it wins over a preset's pinned template. A file that cannot be read fails `load` with `CucaError::Io` naming it. A template that does not compile, or cannot render a one-message conversation, fails `load` with `CucaError::Config`.
- `drafter` takes a `DrafterSource::File(ModelSource)` or `DrafterSource::InTrunk`; in KDL `drafter "in-trunk"` or a source in any form of `source`. The pairings and limits are on [Speculative decoding](@/providers/local/drafting.md). `DrafterSource::InTrunk` needs the CPU. For DeepSeek-V4.1 the `File` source is the model's own checkpoint source, which converts to the DSpark drafter's file; it runs on the CPU only.
- `experts` matters only for a model with experts. `Tiered` needs a RAM budget on the CPU; on a GPU it needs a device budget, a RAM budget or both, and a RAM budget alone is host-tiered: the CPU computes the experts from the RAM tier. `Host` needs a GPU. KDL forms are `experts "resident"`, `experts { host; }`, `experts { host layers=4; }` and `experts { tiered ram_bytes="4g"; }`, where a size is an integer or a string with a `k`, `m` or `g` suffix; see [Tiered expert placement](@/providers/local/expert-tiers.md) and [Run on a GPU](@/providers/local/gpu.md).
- `convert` applies to `ModelSource::Checkpoint` and `ModelSource::Dir` only. `WeightQuant` is `f16`, `q8_0` (default) or `q4_0`. `EngramQuant` is `q8_0` (default), `q4_0` or `bf16` and applies to DeepSeek-V4.1's Engram table only. A non-default `engram_quant` converts and caches next to the default one; see [DeepSeek-V4.1](@/providers/local/checkpoints.md#deepseek-v4-1).
- `shards` makes the model run on `1 + shards.len()` devices, `device` being the first. Every device is a GPU of one backend, there is no `drafter`, `experts` is `Resident`, and the model must be splittable; see [Tensor parallelism](@/providers/local/gpu.md#tensor-parallelism).
- `gpu_layers: Some(n)` runs the last `n` layers and the output head on `device` and the rest on the CPU. It needs a GPU `device`, no `shards` and no `drafter`; `n` may not exceed the model's layer count. `Some(0)` loads the model on the CPU, and `LocalModel::layer_split()` reports the split. See [Layer offload](@/providers/local/gpu.md#layer-offload-gpu-layers).
- `shard_dirs` is searched after the directory of the named first part, for `ModelSource::File` naming a split set whose parts sit on different volumes; see [Split GGUF files](@/providers/local/model-sources.md#split-gguf-files).
- `Debug` for `LocalModelConfig` prints `hf_token` as `<redacted>`.

## SamplingConfig

| Field | Default | Meaning |
|---|---|---|
| `temperature` | `None` (built-in `1.0`) | softmax temperature, finite and at least `0.0`; `Some(0.0)` is greedy |
| `top_k` | `None` (built-in `64`) | keeps the `top_k` most likely tokens; `Some(0)` disables the filter |
| `top_p` | `None` (built-in `0.95`) | nucleus threshold in `(0, 1]`; `Some(1.0)` disables the filter |
| `seed` | `None` | fixed generator seed |

The fields are `Option` so "set" and "default" can be told apart: `top_k: Some(64)` beats a model file that says 20, `None` does not.

- Each of temperature, `top_k` and `top_p` resolves on its own, highest first: the request value (`set_temperature`, `set_top_k`, `set_top_p`), the `SamplingConfig` value, the model file's default, the built-in default (`SamplingConfig::DEFAULT_TEMPERATURE`, `DEFAULT_TOP_K`, `DEFAULT_TOP_P`). `LocalModel::sampling()` returns the resolved values.
- The file defaults are the GGUF keys `general.sampling.temp`, `general.sampling.top_k` and `general.sampling.top_p`, the names llama.cpp reads. The checkpoint converter writes them from `generation_config.json`. A value the config would reject (a negative temperature, `top_p` outside `(0, 1]`) is ignored, and a `top_k` of zero or less disables the filter. The other `general.sampling.*` keys (`min_p`, repetition penalties, XTC, mirostat, `sequence`) name samplers the engine does not have and are not read.
- With `seed: Some(n)`, every request restarts the generator from `n`, so equal prompts sample equal tokens. A request `seed` does the same for that request alone and wins over the config seed. With `None` the generator is seeded from the system once at load.
- The sampler applies temperature, top-k, softmax and top-p in that order, then draws once. Temperature `0.0` takes the first maximum.

## BatchingConfig

In KDL: `batching max_sequences=8 kv_tokens=16384 max_step_tokens=256 prefix_cache=#false`, any subset of the properties.

| Field | Default | Meaning |
|---|---|---|
| `max_sequences` | `4` | requests that run at once, `1..=64` |
| `kv_tokens` | `None` | positions in the shared K/V pool |
| `max_step_tokens` | `None` | rows of one forward step across all requests |
| `prefix_cache` | `true` | keep the K/V of finished requests for later prompts |

- `kv_tokens: None` is twice `context_length`. An explicit value is at least `context_length + 16` and rounds up to whole blocks of 16. It is the K/V memory knob, reported by `LocalModel::kv_cache_bytes`: raising `max_sequences` does not raise it, and a pool too small for the running requests makes them wait or preempt each other. The cost per model is in [Local limits and errors](@/providers/local/limits-and-errors.md#limits).
- `max_step_tokens` is at least `max_sequences` and at most 8,192. `None` is the prefill chunk of the context length, `clamp(2,097,152 / context_length, 16, 256)`, so 256 at 8,192 positions. With tiered experts under compressed sparse attention (DeepSeek-V4.1) it is 4,096, because a larger step reads fewer expert bytes per prompt token.
- `prefix_cache: false` registers nothing in the prefix store, so every prompt prefills in full.
- A drafter that reads every position of its sequence (the in-trunk head, the EAGLE and the DFlash drafters) forces `max_sequences` to 1. A Gemma 4 assistant drafts only while one request runs alone.
- `LocalModel::serving_stats()` returns `ServingStats`. Gauges: `running`, `waiting`, `max_sequences`, `free_blocks`, `cached_blocks`, `used_blocks`, `total_blocks` and `snapshot_bytes`. Totals since load: `prefix_hit_tokens`, `preemptions`, `steps`, `batched_rows`, `prefill_rows` (the part of `batched_rows` that are prompt rows), `generated_tokens`, `admissions`, `queue_wait_ns` (the wait between dispatch and first admission, summed), `step_ns` (forward pass time, summed) and, for the last pass, `last_step_ns` and `last_step_rows`. `batched_rows > steps` means steps carried several rows. The counters are atomics the engine thread updates, so reading them never waits for a step.

## Engram cache

A model with an Engram table (DeepSeek-V4.1) reads rows of `per_layer_token_embd.weight` from the GGUF file. `EngramCache` sets how: a pool of reader threads reads the distinct rows of a forward pass sorted by file offset, and a row cache keeps the most recently used rows. In KDL: `engram ram_bytes="8g" reader_threads=32`, any subset; `ram_bytes` is an integer or a string with an optional `k`, `m` or `g` suffix (powers of 1024). The bounds and error messages are on [Local limits and errors](@/providers/local/limits-and-errors.md#engram-row-cache).

| Field | Default | Meaning |
|---|---|---|
| `ram_bytes` | 4 GiB | bytes of RAM for cached rows, `1 MiB..=1 TiB` |
| `reader_threads` | `16` | disk reader threads, `1..=64` |

- The cache holds rows as the file stores them and counts 48 bytes of overhead per row against `ram_bytes`. At the bound it evicts the least recently used row. Its memory is allocated as it fills, so a short session does not pay for the bound.
- A cache smaller than the distinct rows of one pass still gives correct output: rows evicted before use are read again. The values change speed, never the output.
- `LocalModel::engram_stats()` returns `EngramStats`: `lookups`, `cache_hits`, `disk_reads`, `bytes_read`, `resident_rows`, `resident_bytes` and `capacity_rows`, and `hit_rate()`. A model without an Engram table ignores the setting.

## KDL configuration

`cuca::config::load_kdl_file::<LocalModelConfig>(path)` reads a KDL file into a `LocalModelConfig`, and `load_kdl_str` does the same for text. Both need the `config-kdl` feature, which `provider-local` turns on. Every key is the field name; an unknown key fails with an error naming it.

```kdl,name=local_model.kdl
source "qwen3-0.6b-q8_0"
device "vulkan:0"
shards "vulkan:1"
gpu_layers 20
context_length 8192
cache_dir "D:/cuca-cache"
hf_token "${HF_TOKEN:-}"
chat_template_file "templates/x.jinja"
sampling temperature=0.7 top_k=40 top_p=0.9 seed=7
batching max_sequences=8 kv_tokens=16384 max_step_tokens=256 prefix_cache=#false
experts { tiered ram_bytes="4g" device_bytes="6g" loader_threads=8; }
engram ram_bytes="2g" reader_threads=8
convert quant="q4_0" engram_quant="q8_0"
drafter "in-trunk"
draft_tokens 5
```

`examples/configs/local_model.kdl` lists every key, all but `source` commented out.

```rust,name=Load a configuration file
use std::path::Path;
use cuca::LocalModelConfig;

let config: LocalModelConfig = cuca::config::load_kdl_file(Path::new("local_model.kdl"))?;
let model = cuca::LocalModel::load(config).await?;
```

- `source` is a string (a preset name, `"repo:revision:file"` or `"repo:revision"`), `source file="/models/x.gguf"`, `source dir="/ckpt"`, `source { checkpoint repo=".." revision=".."; }` or `source { hugging_face repo=".." revision=".." file=".." sha256=".." { part file=".." sha256=".."; template repo=".." revision=".." file=".."; } }`. The string form never touches the file system. The forms are on [Model sources](@/providers/local/model-sources.md).
- A list of devices is `device "vulkan:0"` plus `shards "vulkan:1" "vulkan:2"`. `shard_dirs` takes one or several directories the same way.
- A byte count (`ram_bytes`, `device_bytes`) is an integer or a string such as `"512m"` or `"4g"`.
- `${VAR}` and `${VAR:-default}` in the file text are replaced before parsing, from a top-level `variables { NAME "value" }` block first, then the process environment. `hf_token "${HF_TOKEN:-}"` reads the token from the environment and yields an empty token, an anonymous download, when it is unset. This is the only place cuca reads the environment for a setting.
- `on_download_progress` has no KDL form; set it in Rust.

The `cuca` binary reads the same keys inside the `model` nodes of its `serve` section; see [Model server](@/server/model-server.md).

## Cache directory and token

- `cache_dir: None` is the platform cache directory: `%LOCALAPPDATA%\cuca` on Windows (`%USERPROFILE%\AppData\Local\cuca` when that variable is unset), `~/Library/Caches/cuca` on macOS, `$XDG_CACHE_HOME/cuca` else `~/.cache/cuca` elsewhere. When no directory can be resolved, `load` fails with `CucaError::Config` asking for `cache_dir`.
- `hf_token: None` or an empty string downloads anonymously; the library reads no `HF_TOKEN` variable. A token is sent as a bearer token on downloads, and `Debug` prints it as `<redacted>`.

## The local_model example

`examples/local_model.rs` takes the request on the command line and everything about the model from a `LocalModelConfig` file.

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

The live tests read one `cuca-test.kdl` in the repository root; see [Build and test](@/providers/local/build-and-tests.md).
