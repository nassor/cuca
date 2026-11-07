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
| Running requests | `batching.max_sequences`, 1 to 64, default 4 (1 with an in-trunk drafter) | later requests wait in the queue |
| Waiting requests | 16 behind the running ones | `generate_stream` waits for a free slot; no error |
| Output channel | 64 blocks | that request is skipped until its consumer takes its output; the others keep running |
| Prompt | fewer than `context_length` tokens | `CucaError::Provider`: `prompt has N tokens; context_length is M` |
| Completion | `max_tokens`, the remaining context and any `stop` string | generation stops |
| K/V cache | `batching.kv_tokens` positions (twice `context_length` by default) in blocks of 16 per full-attention group, shared by all running sequences and the cached prefixes; a window plus about 270 positions per running sequence plus cached prefixes per sliding-window group; preallocated | no growth; unreferenced blocks are evicted least recently used; a running request that needs a block preempts the youngest one; the prefix cache is reused across requests |
| Step rows | `batching.max_step_tokens`, at most 2048, default the prefill chunk | the decode rows come first, prompts prefill in the rest |
| `draft_tokens` | `1..=16` with a drafter | `CucaError::Config` at load |
| Prefill chunk | 256 tokens at most | a longer prompt prefills in chunks |
| Download stall | 60 seconds without data | `CucaError::Transport`; the partial file stays for a later resume |
| Download connect | 30 seconds | `CucaError::Transport` |

- The engine is a step scheduler: requests start in arrival order while fewer than `max_sequences` run and the pool can hold their prompts, and every step runs one decode row per running request plus chunks of the prompts being prefilled in one forward pass. `batching.kv_tokens` is the shared pool for all of them, so raising `max_sequences` does not raise `kv_cache_bytes`; a smaller pool makes requests wait or preempt each other. A preempted request goes back to the front of the queue and prefills again, mostly from the blocks it left in the prefix store. An in-trunk drafter observes every position of its one sequence, so with it the engine runs one request at a time; a file drafter speculates only while one request runs alone.
- The prefill chunk is `2,097,152 / context_length`, clamped to between 16 and 256 tokens: 256 at 8,192 positions, 64 at 32,768 and 16 at 131,072. The chunk size bounds the transient attention score matrix.
- The K/V cache is paged: blocks of 16 positions, `2 * 4 * 16 * n_kv_head * head_dim` bytes per block and K/V-owning layer (f32 keys and values), with each layer's own head count and head dimension. Layers that share an attention window share block ids (a group). A full-attention group holds `batching.kv_tokens` positions (twice `context_length` by default) in blocks, shared by every running sequence, so a finished request's prefix stays cached while the next one runs. A sliding-window group holds `ceil((window - 1 + rows) / 16) + 2` blocks per running sequence (its window, one call's rows of at most 256, and two blocks of slack), plus up to `kv_tokens - context_length` positions of cached prefixes. `LocalModel::kv_cache_bytes` reports the total of all pools, so you can compare a `context_length` with your memory before the first request.
- A sliding-window block is released once every position in it is before the window of the next query; it stays cached, named by its token ids, until the pool needs it. A prefix hit needs the blocks of the window before its end, so a hit shortens by whole blocks until every group has them.
- Eviction is least recently used among blocks no sequence references; a block in use is never evicted, and a running request that cannot get a block preempts the youngest running request, and only the oldest one alone in a pool smaller than it needs fails with a `CucaError::Provider` (`the K/V block pool is exhausted by a single sequence`).
- K/V cost varies by model. At 8,192 positions it is 422 MB for Gemma 4 E2B and 3.76 GB for Qwen 3 0.6B, which has no sliding-window layers (twice what a pool of exactly `context_length` positions would cost); the preset table on [Local models and formats](@/providers/local/models.md#presets) lists the value for each preset.
- Weights stay quantized and are consumed by the engine's quantized matmul kernel (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0 and the K-quants) or candle's (any other dtype). Norm vectors, biases and float weights are held as f32. Resident weights are about the file size.
- Gemma 4's per-layer embedding table (`per_layer_token_embd`: 1.32 GB in E2B Q4_0, 1.59 GB in E4B Q4_0) is not loaded. The engine reads one row per input token from the GGUF file, usually from the OS page cache, so E2B Q4_0 holds 1.5 GB of its 2.84 GB file.
- Mixture of experts: by default expert tensors are split once at load into one quantized 2-D weight per expert. Memory is the file size, and a pass computes only the experts its tokens route to. With `ExpertPlacement::Tiered` the experts stay in the GGUF file and memory is the dense weights plus the RAM and device budgets you set; see [Tiered expert placement](@/providers/local/expert-tiers.md).
- Hybrid models (Qwen 3.5) keep a recurrent state of fixed size next to the K/V cache. For the 0.8B model it is 20.2 MB per running sequence (`max_sequences` slots), about 40 MB with the rollback copy, plus 445 KB per verified row, and one more 20.2 MB per snapshot in the prefix store (at most `max_sequences + 4`). See [Architectures](@/providers/local/models.md#architectures).
- On a GPU the weights and the K/V cache are in device memory instead of RAM, at the sizes in [Devices and GPU features](@/providers/local/gpu.md#devices-and-gpu-features).

## Errors

`LocalModel::load` returns:

- `CucaError::Config` for an invalid configuration, before any thread starts:
  - `context_length` of 0 (`context_length must be at least 1`), or above the model's maximum (`context_length N exceeds the model maximum M`).
  - `temperature` not finite or below 0, or `top_p` outside `(0, 1]`.
  - `draft_tokens` outside `1..=16` with a drafter.
  - `shards` with a CPU device, a different GPU backend than `device`, a `drafter` or `ExpertPlacement::Tiered` (`tensor parallelism: ...`), or a model that cannot be split over the shards (heads, K/V heads or widths that do not divide, a delta-net layer, a full-width Q/K norm), reported at load before any weight is read.
  - `batching.max_sequences` outside `1..=64`, `batching.max_step_tokens` outside `max_sequences..=2048`, or `batching.kv_tokens` below `context_length + 16`.
  - A `device` whose cargo feature is off, for example ``device Vulkan(Default) requires the `provider-local-vulkan` cargo feature``.
  - `DrafterSource::InTrunk` with a GPU device.
  - A repo, revision or file with an empty, `.`, `..` or `:` segment, a backslash or a leading `/`.
  - An `experts` placement outside its rules (`loader_threads` outside `1..=64`, `device_bytes` on the CPU, `ram_bytes` of 0 on the CPU, `device_bytes` of 0 on a GPU), a budget below the model's `n_used` experts per layer, `warm_from` heat of another shape, or `Tiered` on a model without experts. The messages are on [Tiered expert placement](@/providers/local/expert-tiers.md#errors).
  - No resolvable cache directory: set `cache_dir` or `CUCA_CACHE_DIR`.
- `CucaError::Io` for a missing or unreadable model file, or when the engine thread cannot start.
- `CucaError::Transport` for a download that fails, stalls for 60 seconds, ends short of the reported size, or gets an unusable range response.
- `CucaError::Http` for a non-2xx download response, with the first 512 characters of the body.
- `CucaError::Provider` for the model file and the engine:
  - SHA-256 mismatch: `sha256 mismatch for FILE: expected X, got Y`; the partial file is deleted.
  - An unsupported GGUF: an unknown architecture, tokenizer model or pre-tokenizer, an IQ quantization, a missing tensor or metadata key (`tokenizer.chat_template` included), or no end-of-generation token. The message names the cause.
  - A drafter that does not match its target, or `InTrunk` on a file without a prediction head.
  - A chat template that does not compile: `chat template: ...`.
  - A GPU that cannot be opened: `device ...: <wgpu error>`, or `local engine failed to initialize: <message>` when the backend panics during setup.
  - The engine thread exits during load: `local engine thread exited during load`.

`ModelSource::from_str` returns `CucaError::Config` for an unknown preset or a malformed `repo:revision:file`; the message lists the preset names. `LocalDevice::from_str` returns `CucaError::Config` for unknown text.

`CucaClientBuilder::build` and `generate_stream` return:

- `CucaError::Config` when `ProviderEndpoint::Local` has no model: `ProviderEndpoint::Local requires CucaClientBuilder::with_local_model`. The check runs at `build` and again at dispatch. A request `temperature` that is not finite or is below 0, or a request `top_p` outside `(0, 1]` (`top_p must be a finite value in (0, 1], got P`), is also `CucaError::Config`.
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
