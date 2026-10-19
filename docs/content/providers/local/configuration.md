+++
title = "Local configuration"
description = "The LocalModelConfig and SamplingConfig fields with their defaults, and the environment variables the library, the example and the tests read."
template = "page.html"
weight = 1
+++

# Local configuration

<dl class="page-facts">
<dt>In one line</dt>
<dd><code>LocalModelConfig</code> holds the model source, cache, context length, sampling, device and drafter; <code>SamplingConfig</code> holds the sampler defaults.</dd>
<dt>You need</dt>
<dd>The <code>provider-local</code> feature. Entry points are on <a href="@/providers/local/_index.md">Local (in-process)</a>.</dd>
<dt>Read this if</dt>
<dd>You need a field default, a valid range, or the name of an environment variable.</dd>
</dl>

## Configuration

`LocalModelConfig` fields, all public:

| Field | Default | Meaning |
|---|---|---|
| `source` | `ModelSource::default_model()` | where the GGUF file comes from |
| `cache_dir` | `None` | download cache; see [Download and cache](@/providers/local/model-sources.md#download-and-cache) |
| `hf_token` | `None` | Hugging Face token; `None` falls back to `HF_TOKEN` |
| `context_length` | `None` | prompt plus completion tokens per request, and the K/V cache size |
| `sampling` | `SamplingConfig::default()` | default sampling parameters |
| `device` | `LocalDevice::Cpu` | where the forward pass runs |
| `drafter` | `None` | optional multi-token-prediction drafter, an `Option<DrafterSource>` |
| `draft_tokens` | `3` | tokens the drafter proposes per round, `1..=16`; ignored without a drafter |
| `on_download_progress` | `None` | download progress observer |

- `source` defaults to the preset `gemma4-e2b-q4_0`.
- `context_length` of `None` resolves to the smaller of 8,192 and the model's own maximum, so a model trained for 4,096 positions loads without configuration. An explicit value must be at least 1 and at most the model's maximum.
- `device` needs the matching cargo feature for a GPU; see [Devices and GPU features](@/providers/local/gpu.md#devices-and-gpu-features).
- `Debug` for `LocalModelConfig` prints `hf_token` as `<redacted>`.

`SamplingConfig` fields:

| Field | Default | Meaning |
|---|---|---|
| `temperature` | `1.0` | softmax temperature, finite and at least `0.0`; `0.0` is greedy |
| `top_k` | `64` | keeps the `top_k` most likely tokens; `0` disables the filter |
| `top_p` | `0.95` | nucleus threshold in `(0, 1]`; `1.0` disables the filter |
| `seed` | `None` | fixed generator seed |

- A request's `temperature` overrides the config value for that request. `top_k`, `top_p` and `seed` apply to every request.
- With `seed: Some(n)`, every request restarts the generator from `n`, so equal prompts sample equal tokens. With `None`, the generator is seeded from the system once at load.
- The sampler applies temperature, top-k, softmax and top-p in that order, then draws once. Temperature `0.0` takes the first maximum.

### Environment variables

The library reads these:

| Variable | Meaning |
|---|---|
| `CUCA_CACHE_DIR` | download cache root when `cache_dir` is `None` |
| `HF_TOKEN` | Hugging Face token when `hf_token` is `None` |
| `LOCALAPPDATA`, `USERPROFILE` | platform cache root on Windows |
| `HOME` | platform cache root on macOS and Linux |
| `XDG_CACHE_HOME` | platform cache root on Linux |

The example `examples/local_model.rs` reads these; the library never does:

| Variable | Meaning |
|---|---|
| `CUCA_LOCAL_MODEL_PATH` | path of a GGUF file; wins over `CUCA_LOCAL_MODEL` |
| `CUCA_LOCAL_MODEL` | preset name or `repo:revision:file`; default `gemma4-e2b-q4_0` |
| `CUCA_LOCAL_DEVICE` | `LocalDevice` text form; default `cpu` |
| `CUCA_LOCAL_DRAFTER` | drafter preset name, or `in-trunk` |
| `CUCA_LOCAL_DRAFTER_PATH` | path of a drafter GGUF file |

The live tests read the same variables. Their other variables and run commands are in `AGENTS.md`.
