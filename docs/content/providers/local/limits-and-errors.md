+++
title = "Local limits and errors"
description = "The queue, channel, prompt, K/V cache and download caps with their at-cap behavior, the memory cost of a model, and every error the load and the stream return."
template = "page.html"
weight = 3
+++

# Local limits and errors

<dl class="page-facts">
<dt>In one line</dt>
<dd>A table of caps and their at-cap behavior, the memory formulas, and the <code>CucaError</code> variants the local provider returns.</dd>
<dt>You need</dt>
<dd>The <code>provider-local</code> feature.</dd>
<dt>Read this if</dt>
<dd>You are sizing <code>context_length</code> against memory, or matching an error message to its cause.</dd>
</dl>

## Limits

| Item | Cap | At the cap |
|---|---|---|
| Waiting requests | 16 behind the running one | `generate_stream` waits for a free slot; no error |
| Output channel | 64 blocks | the engine waits for the consumer, so a slow consumer paces generation |
| Prompt | fewer than `context_length` tokens | `CucaError::Provider`: `prompt has N tokens; context_length is M` |
| Completion | `max_tokens` and the remaining context | generation stops |
| K/V cache | `context_length` positions, preallocated | no growth; the cache is reused across requests |
| `draft_tokens` | `1..=16` with a drafter | `CucaError::Config` at load |
| Prefill chunk | 256 tokens at most | a longer prompt prefills in chunks |
| Download stall | 60 seconds without data | `CucaError::Transport`; the partial file stays for a later resume |
| Download connect | 30 seconds | `CucaError::Transport` |

- The engine is a single slot: it runs one request at a time, in arrival order.
- The prefill chunk is `2,097,152 / context_length`, clamped to between 16 and 256 tokens: 256 at 8,192 positions, 64 at 32,768 and 16 at 131,072. The chunk size bounds the transient attention score matrix.
- The K/V cache costs `context_length * 2 * 4 * sum(n_kv_head * head_dim)` bytes (f32 keys and values), summed over the layers that own K/V, with each layer's own head count and head dimension. `LocalModel::kv_cache_bytes` reports the total, so you can compare a `context_length` with your memory before the first request.
- K/V cost varies by model. At 8,192 positions it is 302 MB for Gemma 4 E2B and 6.44 GB for Phi 3.5 mini, which has 32 K/V heads; the preset table on [Local models and formats](@/providers/local/models.md#presets) lists the value for each preset.
- Weights stay quantized and are consumed by the engine's quantized matmul kernel (Q4_0 and Q8_0) or candle's (other dtypes). Norm vectors, biases and float weights are held as f32. Resident weights are about the file size.
- Gemma 4's per-layer embedding table (`per_layer_token_embd`: 1.32 GB in E2B Q4_0, 1.59 GB in E4B Q4_0) is not loaded. The engine reads one row per input token from the GGUF file, usually from the OS page cache, so E2B Q4_0 holds 1.5 GB of its 2.84 GB file.
- Mixture of experts: expert tensors are split once at load into one quantized 2-D weight per expert. Memory is the file size, and a pass computes only the experts its tokens route to.
- Hybrid models (Qwen 3.5) keep a recurrent state of fixed size next to the K/V cache. For the 0.8B model it is 20.2 MB, about 61 MB with the one checkpoint slot and the rollback copy, plus 445 KB per verified row. See [Architectures](@/providers/local/models.md#architectures).
- On a GPU the weights and the K/V cache are in device memory instead of RAM, at the sizes in [Devices and GPU features](@/providers/local/gpu.md#devices-and-gpu-features).

## Errors

`LocalModel::load` returns:

- `CucaError::Config` for an invalid configuration, before any thread starts:
  - `context_length` of 0 (`context_length must be at least 1`), or above the model's maximum (`context_length N exceeds the model maximum M`).
  - `temperature` not finite or below 0, or `top_p` outside `(0, 1]`.
  - `draft_tokens` outside `1..=16` with a drafter.
  - A `device` whose cargo feature is off, for example ``device Vulkan(Default) requires the `provider-local-vulkan` cargo feature``.
  - `DrafterSource::InTrunk` with a GPU device.
  - A repo, revision or file with an empty, `.`, `..` or `:` segment, a backslash or a leading `/`.
  - No resolvable cache directory: set `cache_dir` or `CUCA_CACHE_DIR`.
- `CucaError::Io` for a missing or unreadable model file, or when the engine thread cannot start.
- `CucaError::Transport` for a download that fails, stalls for 60 seconds, ends short of the reported size, or gets an unusable range response.
- `CucaError::Http` for a non-2xx download response, with the first 512 characters of the body.
- `CucaError::Provider` for the model file and the engine:
  - SHA-256 mismatch: `sha256 mismatch for FILE: expected X, got Y`; the partial file is deleted.
  - An unsupported GGUF: an unknown architecture, tokenizer model or pre-tokenizer, an IQ quantization, a missing tensor or metadata key (`tokenizer.chat_template` included), or no end-of-generation token. The message names the cause.
  - A drafter that does not match its target, or `InTrunk` on a file without a prediction head.
  - A chat template that does not compile: `chat template: ...`.
  - `qwen35` on a GPU device: `recurrent (gated delta-net) layers are not implemented on GPU devices; use LocalDevice::Cpu`.
  - A GPU that cannot be opened: `device ...: <wgpu error>`, or `local engine failed to initialize: <message>` when the backend panics during setup.
  - The engine thread exits during load: `local engine thread exited during load`.

`ModelSource::from_str` returns `CucaError::Config` for an unknown preset or a malformed `repo:revision:file`; the message lists the preset names. `LocalDevice::from_str` returns `CucaError::Config` for unknown text.

`CucaClientBuilder::build` and `generate_stream` return:

- `CucaError::Config` when `ProviderEndpoint::Local` has no model: `ProviderEndpoint::Local requires CucaClientBuilder::with_local_model`. The check runs at `build` and again at dispatch. A request `temperature` that is not finite or is below 0 is also `CucaError::Config`.
- `CucaError::Provider` for a request the engine cannot take:
  - An `ImageBase64` block: `the local engine is text-only; ImageBase64 blocks are not supported`.
  - Tool call `arguments` that are not a JSON object: ``tool call `NAME` arguments must be a JSON object``.
  - A template render failure: `chat template: ...`.
  - A prompt of `context_length` tokens or more: `prompt has N tokens; context_length is M`.
  - A stopped engine: `local engine has shut down`.

Failures while streaming arrive as the last stream item:

- A forward pass failure is a `CucaError::Provider` item, and the engine resets the K/V cache.
- A panic inside the engine is caught and reported as `local engine panicked; KV cache reset`. The engine thread keeps running, so the next request is attempted.
- A malformed tool call is a `CucaError::Provider` item, `malformed tool call: <body>`, and the stream continues.

The engine does not detect a lost GPU or rebuild it, so requests on a dead device keep failing the same way.
