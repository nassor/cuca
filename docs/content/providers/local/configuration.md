+++
title = "Local configuration"
description = "The LocalModelConfig, SamplingConfig and BatchingConfig fields with their defaults and valid ranges, and the environment variables the library and the example read."
template = "page.html"
weight = 1
+++

# Local configuration

<dl class="page-facts">
<dt>In one line</dt>
<dd><code>LocalModelConfig</code> holds the model source, cache, context length, sampling, device, drafter, expert placement, batching and offload; <code>SamplingConfig</code> and <code>BatchingConfig</code> hold the sampler and scheduler settings.</dd>
<dt>You need</dt>
<dd>The <code>provider-local</code> feature. Entry points are on <a href="@/providers/local/_index.md">Local (in-process)</a>.</dd>
<dt>Read this if</dt>
<dd>You need a field default, a valid range, or the name of an environment variable.</dd>
</dl>

## LocalModelConfig

All fields are public. `LocalModel::load` validates the configuration before it starts the engine thread and returns `CucaError::Config` for a field outside its range.

| Field | Default | Meaning |
|---|---|---|
| `source` | `ModelSource::default_model()` | where the GGUF file comes from; see [Model sources](@/providers/local/model-sources.md) |
| `cache_dir` | `None` | download cache; see [Download and cache](@/providers/local/model-sources.md#download-and-cache) |
| `hf_token` | `None` | Hugging Face token; `None` falls back to `HF_TOKEN` |
| `context_length` | `None` | prompt plus completion tokens per request; sizes the K/V pool |
| `sampling` | `SamplingConfig::default()` | sampler defaults you set; see [SamplingConfig](#samplingconfig) |
| `chat_template` | `None` | Jinja text that replaces the GGUF's chat template; see [Chat templates](@/providers/local/chat-templates.md#where-the-template-comes-from) |
| `device` | `LocalDevice::Cpu` | where the forward pass runs |
| `drafter` | `None` | `Option<DrafterSource>`: a draft model for speculative decoding |
| `draft_tokens` | `5` | most tokens a drafter proposes per round, `1..=16`; checked only with a drafter |
| `experts` | `ExpertPlacement::Resident` | where a mixture-of-experts model's experts live |
| `convert` | `ConvertConfig::default()` | weight precision of a converted checkpoint |
| `batching` | `BatchingConfig::default()` | how concurrent requests share the engine; see [BatchingConfig](#batchingconfig) |
| `shards` | `[]` | more GPUs to split the model over (tensor parallelism) |
| `gpu_layers` | `None` | layers to run on the GPU, the rest on the CPU |
| `on_download_progress` | `None` | download, conversion and merge progress observer |

- `source` defaults to the preset `gemma4-e2b-q4_0`.
- `context_length: None` resolves to the smaller of 8,192 and the model's own maximum. An explicit value must be at least 1 and at most the model's maximum.
- `device` needs the matching cargo feature for a GPU, else `load` fails with `CucaError::Config` naming it. `LocalModelConfig::set_devices("cuda:0+cuda:1")` fills `device` and `shards` from one `+`-separated text. See [Devices and GPU features](@/providers/local/gpu.md#devices-and-gpu-features).
- `chat_template` also replaces the template a preset pins. Everything derived from the template follows it. A template that does not compile, or cannot render a one-message conversation, fails `load` with `CucaError::Config`.
- `drafter` takes a `DrafterSource::File(ModelSource)` or `DrafterSource::InTrunk`; the pairings and limits are on [Speculative decoding](@/providers/local/drafting.md). `DrafterSource::InTrunk` needs the CPU.
- `experts` matters only for a model with experts. `Tiered` needs a RAM budget on the CPU and a device budget on a GPU. `Host` needs a GPU. Text forms are `resident`, `host`, `host:layers=<n>` and `tiered:ram=4g`; see [Tiered expert placement](@/providers/local/expert-tiers.md) and [Run on a GPU](@/providers/local/gpu.md).
- `convert` applies to `ModelSource::Checkpoint` and `ModelSource::Dir` only. `WeightQuant` is `f16`, `q8_0` (default) or `q4_0`.
- `shards` makes the model run on `1 + shards.len()` devices, `device` being the first. Every device is a GPU of one backend, there is no `drafter`, `experts` is `Resident`, and the model must be splittable; see [Tensor parallelism](@/providers/local/gpu.md#tensor-parallelism).
- `gpu_layers: Some(n)` runs the last `n` layers and the output head on `device` and the rest on the CPU. It needs a GPU `device`, no `shards` and no `drafter`; `n` may not exceed the model's layer count. `Some(0)` loads the model on the CPU, and `LocalModel::layer_split()` reports the split. See [Layer offload](@/providers/local/gpu.md#layer-offload-gpu-layers).
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

`BatchingConfig` parses from text such as `seqs=4,kv=16384,step=256,prefix=off`, any subset, with `prefix` accepting `on`, `off`, `true`, `false`, `1` and `0`.

| Field | Default | Meaning |
|---|---|---|
| `max_sequences` | `4` | requests that run at once, `1..=64` |
| `kv_tokens` | `None` | positions in the shared K/V pool |
| `max_step_tokens` | `None` | rows of one forward step across all requests |
| `prefix_cache` | `true` | keep the K/V of finished requests for later prompts |

- `kv_tokens: None` is twice `context_length`. An explicit value is at least `context_length + 16` and rounds up to whole blocks of 16. It is the K/V memory knob, reported by `LocalModel::kv_cache_bytes`: raising `max_sequences` does not raise it, and a pool too small for the running requests makes them wait or preempt each other. The cost per model is in [Local limits and errors](@/providers/local/limits-and-errors.md#limits).
- `max_step_tokens` is at least `max_sequences` and at most 2,048. `None` is the prefill chunk of the context length, `clamp(2,097,152 / context_length, 16, 256)`, so 256 at 8,192 positions.
- `prefix_cache: false` registers nothing in the prefix store, so every prompt prefills in full.
- A drafter that reads every position of its sequence (the in-trunk head, the EAGLE and the DFlash drafters) forces `max_sequences` to 1. A Gemma 4 assistant drafts only while one request runs alone.
- `LocalModel::serving_stats()` returns `ServingStats`: `running`, `waiting`, `free_blocks`, `cached_blocks`, `snapshot_bytes`, `prefix_hit_tokens`, `preemptions`, `steps` and `batched_rows`. `batched_rows > steps` means steps carried several rows.

## Environment variables

The library reads these:

| Variable | Meaning |
|---|---|
| `CUCA_CACHE_DIR` | download cache root when `cache_dir` is `None` |
| `HF_TOKEN` | Hugging Face token when `hf_token` is `None` |
| `LOCALAPPDATA`, `USERPROFILE` | platform cache root on Windows |
| `HOME` | platform cache root on macOS and Linux |
| `XDG_CACHE_HOME` | platform cache root on Linux |

`examples/local_model.rs` reads these; the library never does:

| Variable | Meaning |
|---|---|
| `CUCA_LOCAL_MODEL_PATH` | path of a GGUF file, or of a checkpoint directory (converted); wins over `CUCA_LOCAL_MODEL` |
| `CUCA_LOCAL_MODEL` | preset name, `repo:revision:file` (a GGUF) or `repo:revision` (a checkpoint, converted); default `gemma4-e2b-q4_0` |
| `CUCA_LOCAL_CONVERT_QUANT` | `f16`, `q8_0` or `q4_0`; default `q8_0` |
| `CUCA_LOCAL_DEVICE` | `LocalDevice` text, or a `+`-separated list for tensor parallelism; default `cpu` |
| `CUCA_LOCAL_GPU_LAYERS` | `gpu_layers` |
| `CUCA_LOCAL_DRAFTER` | drafter preset name, or `in-trunk` |
| `CUCA_LOCAL_DRAFTER_PATH` | path of a drafter GGUF file; wins over `CUCA_LOCAL_DRAFTER` |
| `CUCA_LOCAL_EXPERTS` | `ExpertPlacement` text, for example `tiered:ram=4g`; default `resident` |
| `CUCA_LOCAL_BATCHING` | `BatchingConfig` text |
| `CUCA_LOCAL_CHAT_TEMPLATE` | path of a `.jinja` file, read into `chat_template` |

The live tests read the same variables, parsed in `tests/local_env/mod.rs`, plus test-only ones. Every variable, with the files that read it, is on [Environment variables](@/reference/environment.md).
