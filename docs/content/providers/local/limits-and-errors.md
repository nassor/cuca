+++
title = "Limits and errors"
description = "The caps of the local engine and what happens at each, how much memory a model needs, the option combinations that fail to load, and the errors with their causes and fixes."
template = "page.html"
weight = 11
+++

# Limits and errors

<dl class="page-facts">
<dt>In one line</dt>
<dd>The caps and what happens at each, the memory a model needs, the option combinations that fail, and every <code>CucaError</code> the local provider returns, with a fix.</dd>
<dt>You need</dt>
<dd>The <code>provider-local</code> feature.</dd>
<dt>Read this if</dt>
<dd>You are sizing <code>context_length</code> against memory, or you have an error message and want its cause.</dd>
</dl>

## Limits

| Item | Cap | At the cap |
|---|---|---|
| Running requests | `batching.max_sequences`, 1 to 64, default 4; 1 with an in-trunk, EAGLE, DFlash or DSpark drafter | later requests wait |
| Waiting requests | 16 in the engine queue, 16 more buffered in the job channel | `generate_stream` waits for room; no error |
| Output channel | 64 blocks per request | that request pauses until you read; the others keep running |
| Prompt | fewer than `context_length` tokens | `CucaError::Provider`: `prompt has N tokens; context_length is M` |
| Reply | `max_tokens`, the remaining context, or any `stop` string | generation stops |
| K/V pool | `batching.kv_tokens` positions, allocated at load | no growth: cached blocks nobody uses are evicted, least recently used first, then the youngest running request is preempted |
| Step rows | `batching.max_step_tokens`, `max_sequences` to 8,192 | decode rows come first, prompts prefill in the rest |
| Prefill chunk | `2,097,152 / context_length` tokens, 16 to 256 | a longer prompt prefills in chunks |
| `draft_tokens` | 1 to 16 with a drafter | `CucaError::Config` at load outside the range |
| Engram row cache | `engram.ram_bytes`, 1 MiB to 1 TiB, default 4 GiB | no growth: the least recently used row goes first |
| Engram readers | `engram.reader_threads`, 1 to 64, default 16 | `CucaError::Config` at load outside the range |
| Download stall | 60 seconds without data | `CucaError::Transport`; the partial file stays for a later resume |
| Download connect | 30 seconds | `CucaError::Transport` |

- Requests start in arrival order while fewer than `max_sequences` run and the pool can hold their prompts. A preempted request goes back to the front of the queue and prefills again, mostly from blocks it left in the prefix cache.
- A prefill chunk is 256 tokens at 8,192 positions, 64 at 32,768 and 16 at 131,072.
- A drafter that reads every position of its sequence (the in-trunk head, EAGLE, DFlash, DSpark) serves one request at a time, and the Gemma 4 assistant speculates only while one request runs alone. See [Speculative decoding](@/providers/local/drafting.md).

## How much memory a model needs

A local model needs memory for its weights and for its K/V cache. The K/V cache is where the engine keeps each request's attention state, so it does not recompute earlier tokens. Both live in RAM on the CPU and in device memory on a GPU.

<div class="dgm">
<div class="dgm-scroll">
<svg viewBox="0 0 660 222" role="img" aria-labelledby="mem-title mem-desc">
  <title id="mem-title">What fills the memory of a local model</title>
  <desc id="mem-desc">A frame for RAM, or device memory on a GPU, holds three blocks: the weights, about the file size and kept quantized; the K/V cache, kv_tokens positions that LocalModel::kv_cache_bytes reports; and extras such as the recurrent state of a hybrid model and the experts of a DSpark drafter. Below the frame, a box says that Gemma 4's per-layer embeddings, the Engram table and experts outside a tier stay in the model file on disk and are read on demand.</desc>
  <defs>
    <marker id="mem-arrow" viewBox="0 0 8 8" refX="7" refY="4"
            markerWidth="6" markerHeight="6" orient="auto">
      <path d="M0 0.6 L7 4 L0 7.4 z" fill="var(--muted-foreground)"/>
    </marker>
  </defs>
  <text class="t-title" x="6" y="20">Memory of one local model</text>
  <rect class="blk" x="6" y="34" width="648" height="104" rx="6"/>
  <text class="t-ax" x="20" y="52">RAM, OR DEVICE MEMORY ON A GPU</text>
  <rect class="blk blk-data" x="20" y="62" width="210" height="64" rx="6"/>
  <text class="t-lbl t-mid t-data" x="125" y="82">Weights</text>
  <text class="t-sm t-mid" x="125" y="100">about the file size</text>
  <text class="t-sm t-mid" x="125" y="114">kept quantized</text>
  <rect class="blk blk-data" x="238" y="62" width="240" height="64" rx="6"/>
  <text class="t-lbl t-mid t-data" x="358" y="82">K/V cache</text>
  <text class="t-sm t-mid" x="358" y="100">kv_tokens positions</text>
  <text class="t-sm t-mid" x="358" y="114">LocalModel::kv_cache_bytes</text>
  <rect class="blk blk-data" x="486" y="62" width="154" height="64" rx="6"/>
  <text class="t-lbl t-mid t-data" x="563" y="82">Extras</text>
  <text class="t-sm t-mid" x="563" y="100">recurrent state,</text>
  <text class="t-sm t-mid" x="563" y="114">DSpark experts</text>
  <path class="arw arw-data" d="M330 182 V142" marker-end="url(#mem-arrow)"/>
  <text class="t-ax t-data" x="340" y="164">rows read on demand</text>
  <rect class="blk" x="6" y="184" width="648" height="30" rx="6"/>
  <text class="t-sm t-mid" x="330" y="203">Stays in the model file: Gemma 4 per-layer embeddings, the Engram table, experts outside a tier</text>
</svg>
</div>
<p class="dgm-cap"><b>Plan for the weights plus the K/V cache.</b> Lower <code>context_length</code> or <code>batching.kv_tokens</code> to shrink the cache. The weights are the file.</p>
</div>

- Weights stay quantized, so resident weights are about the file size. A 4.6 GB file needs about 4.6 GB. Gemma 4's per-layer embedding table (1.32 GB in E2B Q4_0) is not loaded, so E2B Q4_0 holds 1.5 GB of its 2.84 GB file. DeepSeek-V4.1's Engram table (208.9 GB at the default precision) is not loaded either; see [Engram cache](@/providers/local/configuration.md#engram-cache).
- Each cached position stores a key and a value of `n_kv_head * head_dim` numbers, as 4-byte floats, in every layer that keeps K/V. The pool holds `kv_tokens` positions for all running requests together, twice `context_length` by default. So a full-attention layer costs `kv_tokens * 2 * 4 * n_kv_head * head_dim` bytes, and the cache is the sum over layers. It does not grow with `max_sequences`.
- At 8,192 positions, Qwen 3 0.6B costs 458,752 bytes per context position, 3.76 GB in all, because it has many K/V heads for its size. Gemma 4 E2B costs 481 MB, because its sliding-window layers keep only a window per running request plus cached prefixes.
- The K/V column of the presets table on [Choose a model](@/providers/local/models.md) gives the cache for each preset at the default settings. `LocalModel::kv_cache_bytes` reports the cache of a loaded model.
- A hybrid model (Qwen 3.5) also keeps a recurrent state per running request, about 20 MB for the 0.8B model, and one more per cached snapshot. `kv_cache_bytes` does not count it; `ServingStats::snapshot_bytes` reports the snapshots.
- Resident experts of a mixture of experts model cost the file size. With tiered experts, memory is the dense weights plus the RAM and device budgets; see [Tiered experts](@/providers/local/expert-tiers.md).
- With `gpu_layers`, the CPU half keeps its weights and K/V in RAM and the GPU half keeps its own in device memory. With `experts { host; }`, every hosted expert is read into RAM at load. See [Fit a large model](@/providers/local/placement.md).
- A DSpark drafter reads its experts into RAM when it opens (7.2 GB of MXFP4 experts in a 7.91 GB file at the default precision). See [Speculative decoding](@/providers/local/drafting.md).
- Only the oldest running request, alone in a pool smaller than it needs, fails: `CucaError::Provider`, `the K/V block pool is exhausted by a single sequence`. Raise `kv_tokens` or lower `context_length`.

## Rejected combinations

`LocalModel::load` returns `CucaError::Config` for each of these. Some checks run from the configuration alone, before the engine thread starts. The others run once the file header is read.

Found from the configuration alone:

- `shards` with a CPU device: ``tensor parallelism: `device` and every shard must be a GPU device, got cpu``
- `shards` of another GPU backend than `device`: `tensor parallelism: every shard must use the same GPU backend as ...`
- `shards` with a `drafter`: `tensor parallelism: a drafter is not supported with ...`
- `shards` with `experts` set to `tiered` or `host`: `tensor parallelism: tiered experts and host experts are not supported with ...`
- `gpu_layers` with a CPU device: ``gpu_layers needs a GPU `device`, got cpu``
- `gpu_layers` with `shards`: ``gpu_layers cannot be combined with tensor parallelism (`shards`)``
- `gpu_layers` with a `drafter`: ``gpu_layers cannot be combined with a `drafter` ``
- `experts` `host` with a CPU device: `ExpertPlacement::Host keeps the experts on the CPU for a GPU device; ...`
- `drafter "in-trunk"` with a GPU device: `DrafterSource::InTrunk runs on the CPU only, not on ...`

Found once the file header is read:

- `shards` with a hybrid, latent attention, DeepSeek-V4.1 or full-width Q/K norm model, or heads and widths that do not divide: `tensor parallelism: <rule> (<numbers>)`
- `gpu_layers` above the model's layer count: ``gpu_layers N exceeds the M layers of `<architecture>` ``
- `gpu_layers` that cuts DeepSeek-V4.1 between layers that share compressed rows or indexer picks: ``gpu_layers N cuts `deepseek41` between layers that share compressed rows or indexer picks; legal values: ...``
- `host` or `tiered` experts on a model without experts: ``host experts need a mixture-of-experts model; `<architecture>` has no experts``, or the same with `expert tiers need`

`gpu_layers` combines with batching, with `tiered` and with `host`. `shards` combines with batching and prefix reuse.

The EAGLE, DFlash and DSpark drafters fail on a GPU device, each with its own message:

- EAGLE, `CucaError::Config`: `the EAGLE drafter runs on the CPU only, not on <device>`
- DFlash, `CucaError::Provider`: `the DFlash drafter runs on the CPU only`
- DSpark, `CucaError::Provider`: `the DSpark drafter runs on the CPU only: use LocalDevice::Cpu`

## Errors

`LocalModel::load` fails before it serves a request. A request fails when it is sent, or later as the last item of its stream.

### Configuration errors

`CucaError::Config`. Most come from the configuration alone, before any thread starts.

| Message | Cause and fix |
|---|---|
| `context_length must be at least 1` | `context_length` is 0. Use 1 or more, or leave it unset. |
| `context_length N exceeds the model maximum M` | The model was trained on fewer positions (OLMo 2 and OLMoE: 4,096). Lower it. |
| `batching.kv_tokens must be at least context_length + 16 (N), got M` | The pool cannot hold one full context. Raise `kv_tokens` or leave it unset. |
| `draft_tokens must be in 1..=16, got N` | Out of range with a drafter. Use 1 to 16. |
| `batching.max_sequences` or `batching.max_step_tokens` out of range | `max_sequences` is 1 to 64, and `max_step_tokens` is `max_sequences` to 8,192. |
| `sampling.temperature` or `sampling.top_p` out of range | `temperature` must be finite and at least 0. `top_p` must be above 0 and at most 1. |
| ``device Vulkan(Default) requires the `provider-local-vulkan` cargo feature`` | The device's feature is off. Build with the feature the message names. |
| `engram.ram_bytes must be in 1048576..=1099511627776 (1 MiB to 1 TiB), got N` | Use a size from 1 MiB to 1 TiB. |
| `engram.reader_threads must be in 1..=64, got N` | Use 1 to 64. |
| ``expected a byte count such as `4294967296`, `512m` or `4g` `` | A size string is not an integer with an optional `k`, `m` or `g`. |
| An `experts` placement outside its rules | `loader_threads` outside 1 to 64, `device_bytes` on the CPU, `ram_bytes` of 0 on the CPU, or both budgets 0 on a GPU. The messages are on [Tiered experts](@/providers/local/expert-tiers.md). |
| `chat_template override does not compile: ...` or `... does not render a one-message conversation: ...` | Your template is broken. See [Replace the chat template](@/providers/local/requests.md#replace-the-chat-template). |
| A `chat_template` that is a bare name the engine does not know | The message names the three built-in names. |
| No resolvable cache directory | Set `cache_dir`. |
| A repo, revision or file with an empty, `.`, `..` or `:` segment, a backslash or a leading `/` | Fix the name. |
| `ProviderEndpoint::Local requires CucaClientBuilder::with_local_model` | A client routes to the local provider with no model. Call `with_local_model`. The check runs at `build` and again at dispatch. |

The combinations in [Rejected combinations](#rejected-combinations) are also `CucaError::Config`. So are an expert budget below the model's experts per token, heat from another model shape in `warm_from`, and `drafter "in-trunk"` on a mixture of experts file.

A KDL file that does not parse is a `CucaError::Config` too: an unknown key (the message names it), a value of the wrong type, a `${VAR}` placeholder with no value and no default, or a string that is no valid source, device or quantization. The message for a source lists the preset names.

A checkpoint conversion that cannot run is a `CucaError::Config` that names the cause:

- ``checkpoint has no tensor `<name>` ``, a tensor of a type or shape the converter does not read, or a checkpoint tensor no rule accounts for.
- `converting <what> writes N GB into <dir>, which has M GB free; free K GB more, choose another cache directory, or convert with a smaller precision (...)`. A DeepSeek-V4.1 conversion checks the free space of the output volume before it writes anything.
- ``the checkpoint has no DSpark drafter (num_nextn_predict_layers and dspark_* in config.json)``, when you ask for a DSpark drafter from a checkpoint that has none.

### Model file and download errors

- `CucaError::Io`: the model file is missing or unreadable, `chat_template_file` is unreadable, or the engine thread cannot start. The message names the file.
- `CucaError::Transport`: a download failed, stalled for 60 seconds, ended short of the reported size, or got an unusable range response. Run again: the partial file stays and the download resumes.
- `CucaError::Http`: the server answered a download with a non-2xx status. The message holds the first 512 characters of the body. A gated or private repository needs `hf_token`.
- `sha256 mismatch for FILE: expected X, got Y`: `CucaError::Provider`. The file is corrupt or not the pinned one. The partial file is deleted; run again.
- ``unsupported GGUF architecture `<name>`; supported: ...``: `CucaError::Provider`. The engine does not load that architecture. The message lists the supported ones. See [Choose a model](@/providers/local/models.md).
- An unsupported tokenizer model, pre-tokenizer or tensor type, a missing tensor or metadata key, or no end-of-generation token: `CucaError::Provider`. The message names the cause. IQ4_NL and MXFP4 tensors load; the other IQ types do not.
- ``GGUF metadata `tokenizer.chat_template`: missing``: the model file carries no chat template. Set `chat_template` or `chat_template_file`.
- `chat template: ...`: the model's own template does not compile. Set `chat_template` or `chat_template_file`.
- ``part N of M of the split GGUF `<path>` is missing: `<name>` ``: a part of a split GGUF set is missing. The loader looks next to the named file, then in `shard_dirs`. Other messages for this set: `... is named part N but its split.no is M` and `the split GGUF ... declares N tensors but its parts hold M`.
- A drafter that does not match its target, or `drafter "in-trunk"` on a file with no prediction head: `CucaError::Provider`. See [Speculative decoding](@/providers/local/drafting.md).
- `deepseek41.attention.kv_source_layers` missing: a DeepSeek-V4.1 GGUF written in an older layout. Convert the checkpoint again.

### GPU errors

| Message | Cause and fix |
|---|---|
| `device <device>: <wgpu error>` | Vulkan or Metal could not open the adapter. Check the driver and the `vulkan:<n>` index. |
| `device Cuda(N): K such device(s) found (missing driver or GPU?)` | CUDA or ROCm found no device at that index. Check the driver and the index. |
| `local engine failed to initialize: <message>` | The GPU backend panicked during setup. The message carries the backend's text. |
| `local engine thread exited during load` | The engine thread stopped before the model was ready. Read the error printed before it. |

The Gemma 4 26B-A4B mixture of experts does not run on a GPU: a GPU load of it fails with a `CucaError::Provider` that says to use `LocalDevice::Cpu`.

### Request errors

`CucaClientBuilder::build` and `generate_stream` return:

- `CucaError::Config`: the request `temperature` is not finite or is below 0, or `top_p` is outside `(0, 1]` (`top_p must be a finite value in (0, 1], got P`).
- `the local engine is text-only; ImageBase64 blocks are not supported`: a message holds an image block. Remove it.
- ``tool call `NAME` arguments must be a JSON object``: a `ToolCall` in the history has arguments that are not an object.
- `chat template: ...`: the template failed to render this conversation. Some templates raise on a message order they do not allow.
- `prompt has N tokens; context_length is M`: the prompt does not fit. Shorten it, or raise `context_length`.
- `local engine has shut down`: the engine stopped. Load the model again.

### Stream errors

A failure while streaming arrives as the last stream item.

- A forward pass failure is a `CucaError::Provider` item for every running request, and the engine resets the K/V pools.
- A panic inside the engine is caught and reported as `local engine panicked; KV cache reset`. The engine thread keeps running, so the next request is attempted.
- A malformed tool call is a `CucaError::Provider` item, `malformed tool call: <body>`, and the stream continues.
- `the K/V block pool is exhausted by a single sequence` is described under [How much memory a model needs](#how-much-memory-a-model-needs).

The engine has no recovery for a lost GPU. Requests on a dead device keep failing, so restart the process.
