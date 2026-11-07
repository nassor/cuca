+++
title = "Local limits and errors"
description = "The queue, channel, prompt, K/V cache and download caps with their at-cap behavior, the memory cost of a model, the rejected option combinations and every error the load and the stream return."
template = "page.html"
weight = 3
+++

# Local limits and errors

<dl class="page-facts">
<dt>In one line</dt>
<dd>The caps and what happens at each, the memory formulas, the option combinations that fail, and the <code>CucaError</code> variants the local provider returns.</dd>
<dt>You need</dt>
<dd>The <code>provider-local</code> feature.</dd>
<dt>Read this if</dt>
<dd>You are sizing <code>context_length</code> against memory, or matching an error message to its cause.</dd>
</dl>

## Limits

| Item | Cap | At the cap |
|---|---|---|
| Running requests | `batching.max_sequences`, 1 to 64, default 4; 1 with an in-trunk, EAGLE or DFlash drafter | later requests wait |
| Waiting requests | 16 in the engine queue, 16 more buffered in the job channel | `generate_stream` waits for room; no error |
| Output channel | 64 blocks per request | that request stalls until its consumer reads; the others keep running |
| Prompt | fewer than `context_length` tokens | `CucaError::Provider`: `prompt has N tokens; context_length is M` |
| Completion | `max_tokens`, the remaining context, any `stop` string | generation stops |
| K/V pool | `batching.kv_tokens` positions per full-attention group, preallocated | no growth: unreferenced blocks are evicted least recently used, then the youngest running request is preempted |
| Step rows | `batching.max_step_tokens`, from `max_sequences` to 2048, default the prefill chunk | decode rows come first, prompts prefill in the rest |
| Prefill chunk | `2,097,152 / context_length` tokens, clamped to 16 to 256 | a longer prompt prefills in chunks |
| `draft_tokens` | 1 to 16 with a drafter | `CucaError::Config` at load |
| Download stall | 60 seconds without data | `CucaError::Transport`; the partial file stays for a later resume |
| Download connect | 30 seconds | `CucaError::Transport` |

Requests start in arrival order while fewer than `max_sequences` run and the pool can hold their prompts. Every step is one forward pass over one decode row per running request plus chunks of the prompts being prefilled. A preempted request goes back to the front of the queue and prefills again, mostly from the blocks it left in the prefix store. The prefill chunk is 256 at 8,192 positions, 64 at 32,768 and 16 at 131,072; it bounds the transient attention score matrix.

A drafter that observes every position (the in-trunk head, EAGLE, DFlash) serves one request at a time. The Gemma 4 assistant speculates only while one request runs alone. See [Speculative decoding](@/providers/local/drafting.md#rounds-and-acceptance).

## Memory

- A K/V block holds 16 positions: `2 * 4 * 16 * n_kv_head * head_dim` bytes per K/V layer (f32 keys and values, each layer's own head count and dimension). A latent-attention layer (`mistral4`) stores one `kv_lora + rot` row per position and no value pool.
- Layers that share an attention window form a group and share block ids. A full-attention group holds `batching.kv_tokens` positions, twice `context_length` by default, shared by every running request and the cached prefixes. A sliding-window group holds `ceil((window - 1 + rows) / 16) + 2` blocks per running request, plus up to `kv_tokens - context_length` positions of cached prefixes, never more than the full group. `rows` is the largest call one request makes: at most the prefill chunk, at least 17.
- `LocalModel::kv_cache_bytes` reports the pools of all devices. At 8,192 positions it is 422 MB for Gemma 4 E2B and 3.76 GB for Qwen 3 0.6B, which has no sliding-window layers. The K/V column of the presets table on [Local models and formats](@/providers/local/models.md#presets) lists the value per preset.
- A sliding-window block is released once every position in it is before the window of the next query. It stays cached, named by its token ids, until the pool needs it. A prefix hit needs the blocks of the window before its end, so it shortens by whole blocks until every group has them.
- Only the oldest running request, alone in a pool smaller than it needs, fails: `CucaError::Provider`, `the K/V block pool is exhausted by a single sequence`.
- Weights stay quantized. On the CPU, Q4_0, Q4_1, Q5_0, Q5_1 and Q8_0 weights run through the engine's batch invariant kernel, and the K-quants (Q2_K to Q6_K) and every other dtype through candle's per row kernel. Float matmul weights, norm vectors and biases are held as f32. Resident weights are about the file size. The GPU layouts are on [Device memory](@/providers/local/gpu.md#device-memory), and the kernel choice is on [Weight formats](@/providers/local/models.md#weight-formats).
- Gemma 4's per-layer embedding table (`per_layer_token_embd`: 1.32 GB in E2B Q4_0, 1.59 GB in E4B Q4_0) is not loaded. The engine reads one row per input token from the GGUF file, usually from the OS page cache, so E2B Q4_0 holds 1.5 GB of its 2.84 GB file.
- A mixture-of-experts model splits its expert tensors once at load into one quantized 2-D weight per expert. Memory is the file size, and a pass computes only the experts its tokens route to. With `ExpertPlacement::Tiered` memory is the dense weights plus the RAM and device budgets; see [Tiered expert placement](@/providers/local/expert-tiers.md).
- A hybrid model (Qwen 3.5) keeps a recurrent state of fixed size per running request (about 20 MB for the 0.8B model, two copies while speculating) and one more per cached snapshot, at most `max_sequences + 4` of them. None of it is counted in `kv_cache_bytes`; `ServingStats::snapshot_bytes` reports the snapshots. See [Architectures](@/providers/local/models.md#architectures).
- With `gpu_layers`, the CPU half keeps its layers' weights and K/V pools in RAM and the GPU half its own in device memory. With `ExpertPlacement::Host`, every hosted expert is read into RAM at load, the file's size for those layers. The mirror pools of a Gemma 4 split across a shared K/V boundary are not counted in `kv_cache_bytes`. See [Layer offload](@/providers/local/gpu.md#layer-offload-gpu-layers).

## Rejected combinations

`LocalModel::load` returns `CucaError::Config` for each of these. `Checked` says whether the test runs before the engine thread starts (`config`) or once the file header is read (`header`).

| Combination | Checked | Message |
|---|---|---|
| `shards` with a CPU device | config | ``tensor parallelism: `device` and every shard must be a GPU device, got cpu`` |
| `shards` of another GPU backend than `device` | config | `tensor parallelism: every shard must use the same GPU backend as ...` |
| `shards` with a `drafter` | config | `tensor parallelism: a drafter is not supported with ...` |
| `shards` with `ExpertPlacement::Tiered` or `Host` | config | `tensor parallelism: tiered experts and host experts are not supported with ...` |
| `gpu_layers` with a CPU device | config | ``gpu_layers needs a GPU `device`, got cpu`` |
| `gpu_layers` with `shards` | config | ``gpu_layers cannot be combined with tensor parallelism (`shards`)`` |
| `gpu_layers` with a `drafter` | config | `` gpu_layers cannot be combined with a `drafter` `` |
| `ExpertPlacement::Host` with a CPU device | config | `ExpertPlacement::Host keeps the experts on the CPU for a GPU device; ...` |
| `DrafterSource::InTrunk` with a GPU device | config | `DrafterSource::InTrunk runs on the CPU only, not on ...` |
| `shards` with a hybrid, latent attention or full-width Q/K norm model, or heads and widths that do not divide | header, before any weight is read | `tensor parallelism: <rule> (<numbers>)` |
| `gpu_layers` above the model's layer count | header | `` gpu_layers N exceeds the M layers of `<architecture>` `` |
| `Host` or `Tiered` on a model without experts | header | ``host experts need a mixture-of-experts model; `<architecture>` has no experts``, or the same with `expert tiers need` |

`gpu_layers` combines with batching, with `ExpertPlacement::Tiered` and with `ExpertPlacement::Host`. `shards` combines with batching and prefix reuse. The EAGLE and DFlash drafters fail on a GPU device: EAGLE with `CucaError::Config` (`the EAGLE drafter runs on the CPU only, not on <device>`), DFlash with `CucaError::Provider` (`the DFlash drafter runs on the CPU only`).

## Errors

`LocalModel::load` returns:

- `CucaError::Config` for an invalid configuration, before any thread starts:
  - `context_length must be at least 1`.
  - `sampling.temperature` not finite or below 0, or `sampling.top_p` outside `(0, 1]`.
  - `draft_tokens` outside `1..=16` with a drafter (`draft_tokens must be in 1..=16, got N`).
  - `batching.max_sequences` outside `1..=64`, or `batching.max_step_tokens` outside `max_sequences..=2048`.
  - A `device` whose cargo feature is off, for example ``device Vulkan(Default) requires the `provider-local-vulkan` cargo feature``.
  - An `experts` placement outside its rules: `loader_threads` outside `1..=64`, `device_bytes` on the CPU, `ram_bytes` of 0 on the CPU, `device_bytes` of 0 on a GPU. The messages are on [Tiered expert placement](@/providers/local/expert-tiers.md#errors).
  - The combinations marked `config` above.
- `CucaError::Config` once the file header is read, on the engine thread:
  - `context_length N exceeds the model maximum M`.
  - `batching.kv_tokens must be at least context_length + 16 (N), got M`.
  - The combinations marked `header` above, an expert budget below the model's `n_used` experts per layer, `warm_from` heat of another shape ([Tiered expert placement](@/providers/local/expert-tiers.md#errors)), and `DrafterSource::InTrunk` on a mixture of experts file ([Speculative decoding](@/providers/local/drafting.md#qwen-3-5-head)).
  - A repo, revision or file with an empty, `.`, `..` or `:` segment, a backslash or a leading `/`.
  - No resolvable cache directory: set `cache_dir` or `CUCA_CACHE_DIR`.
  - A `chat_template` override that does not compile or does not render a one-message conversation (`chat_template override does not compile: ...`, `chat_template override does not render: ...`).
- `CucaError::Io` for a missing or unreadable model file, or when the engine thread cannot start.
- `CucaError::Transport` for a download that fails, stalls for 60 seconds, ends short of the reported size, or gets an unusable range response.
- `CucaError::Http` for a non-2xx download response, with the first 512 characters of the body.
- `CucaError::Provider` for the model file and the engine:
  - SHA-256 mismatch: `sha256 mismatch for FILE: expected X, got Y`; the partial file is deleted.
  - An unsupported GGUF: an unknown architecture, tokenizer model or pre-tokenizer, an IQ quantization, a missing tensor or metadata key, or no end-of-generation token. The message names the cause.
  - A GGUF without `tokenizer.chat_template` (``GGUF metadata `tokenizer.chat_template`: missing``), or one whose template does not compile (`chat template: ...`).
  - A drafter that does not match its target, or `DrafterSource::InTrunk` on a file without a prediction head. The messages are on [Speculative decoding](@/providers/local/drafting.md#drafters).
  - A GPU that cannot be opened: `device <device>: <wgpu error>` for Vulkan and Metal, `device Cuda(N): K such device(s) found (missing driver or GPU?)` for CUDA and ROCm, or `local engine failed to initialize: <message>` when the backend panics during setup.
  - The engine thread exits during load: `local engine thread exited during load`.

The parsers behind the environment variables return `CucaError::Config` for text they do not know: `ModelSource::from_str` (the message lists the preset names), `LocalDevice::from_str`, `ExpertPlacement::from_str`, `BatchingConfig::from_str` and `WeightQuant::from_str`.

`CucaClientBuilder::build` and `generate_stream` return:

- `CucaError::Config` when `ProviderEndpoint::Local` has no model: `ProviderEndpoint::Local requires CucaClientBuilder::with_local_model`. The check runs at `build` and again at dispatch. A request `temperature` that is not finite or is below 0, or a request `top_p` outside `(0, 1]` (`top_p must be a finite value in (0, 1], got P`), is also `CucaError::Config`.
- `CucaError::Provider` for a request the engine cannot take:
  - An `ImageBase64` block: `the local engine is text-only; ImageBase64 blocks are not supported`.
  - Tool call `arguments` that are not a JSON object: ``tool call `NAME` arguments must be a JSON object``.
  - A template render failure: `chat template: ...`.
  - A prompt of `context_length` tokens or more: `prompt has N tokens; context_length is M`.
  - A stopped engine: `local engine has shut down`.

Failures while streaming arrive as the last stream item:

- A forward pass failure is a `CucaError::Provider` item for every running request, and the engine resets the K/V pools.
- A panic inside the engine is caught and reported the same way as `local engine panicked; KV cache reset`. The engine thread keeps running, so the next request is attempted.
- A malformed tool call is a `CucaError::Provider` item, `malformed tool call: <body>`, and the stream continues.

The engine has no recovery for a lost GPU: requests on a dead device keep failing.
