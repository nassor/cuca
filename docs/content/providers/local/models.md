+++
title = "Local models and formats"
description = "What provider-local loads from a GGUF file: architectures, weight formats, tokenizers, chat templates, reasoning and tool-call parsing, the preset catalog, and what each was verified to do."
template = "page.html"
weight = 5
+++

# Local models and formats

<dl class="page-facts">
<dt>In one line</dt>
<dd>The engine loads a GGUF file the way llama.cpp does: the architecture, the tokenizer and the chat template all come from the file, and the stream has the same <code>Text</code>, <code>Thinking</code> and <code>ToolCall</code> blocks whatever the model.</dd>
<dt>You need</dt>
<dd>The <code>provider-local</code> feature. Loading, configuration, devices and errors are on <a href="@/providers/local/_index.md">Local (in-process)</a>.</dd>
<dt>Read this if</dt>
<dd>You are choosing a model, loading a GGUF that is not a preset, or checking why a model's reply is parsed the way it is.</dd>
</dl>

Nothing here is specific to one model family. The loader reads every dimension from the file's metadata and tensor shapes, selects the forward-pass behavior from `general.architecture`, builds the tokenizer from `tokenizer.ggml.*`, and renders prompts with the Jinja chat template stored in `tokenizer.chat_template`. llama.cpp is the design reference; nothing of it is linked or called.

## What the engine reads

| Fact | Value |
|---|---|
| Feature flag | `provider-local` (CPU); GPU features are on [Local (in-process)](@/providers/local/gpu.md#devices-and-gpu-features) |
| Entry types | `ModelSource`, `ModelPreset`, `LocalModelConfig::source` |
| File format | GGUF |
| Selects the forward pass | `general.architecture` |
| Selects the tokenizer | `tokenizer.ggml.model` and `tokenizer.ggml.pre` |
| Supplies the prompt format | `tokenizer.chat_template`, required |
| Selects end-of-generation tokens | `tokenizer.ggml.eos_token_id`, `eot_token_id`, `eom_token_id` and known terminator tokens in the vocabulary |

## Architectures

`general.architecture` selects the behavior. An architecture outside this list fails the load with a `CucaError::Provider` that names it.

| Architecture | Model families | Devices |
|---|---|---|
| `gemma4` | Gemma 4 (E2B, E4B) | CPU, GPU |
| `gemma3` | Gemma 3 (270M, 1B) | CPU, GPU |
| `gemma2` | Gemma 2 (2B) | CPU, GPU |
| `llama` | Llama 3.2, SmolLM 2, TinyLlama; other models published as `llama` | CPU, GPU |
| `qwen2` | Qwen 2.5, DeepSeek-R1-Distill-Qwen | CPU, GPU |
| `qwen3` | Qwen 3 | CPU, GPU |
| `qwen35` | Qwen 3.5 (0.8B) | CPU only |
| `phi3` | Phi 3.5 mini, Phi 4 mini | CPU, GPU |
| `smollm3` | SmolLM 3 | CPU, GPU |
| `granite` | Granite 3.3 (dense) | CPU, GPU |
| `granitemoe` | Granite 3.1 MoE | CPU, GPU |
| `olmo2` | OLMo 2 | CPU, GPU |
| `ernie4_5` | ERNIE 4.5 (dense) | CPU, GPU |
| `gemma4-assistant` | the Gemma 4 multi-token-prediction drafter | only through `LocalModelConfig::drafter` |

What each architecture adds to the common forward pass:

- `gemma4`: sliding and full layers with their own head sizes and RoPE bases, shared K/V layers (the last `shared_kv_layers` layers read an earlier layer's cache), per-layer embeddings read from the file per token, a weightless value norm, a per-layer output scale, and a final logit soft-cap.
- `gemma3`: sliding and full layers (every sixth layer is full), QK-norm, post-attention and post-FFN norms.
- `gemma2`: alternating sliding and full layers (every other layer is full), an attention logit soft-cap and a final logit soft-cap, post-attention and post-FFN norms, no QK-norm.
- `llama`: the Llama 3 RoPE scaling that the file carries as `rope_freqs.weight`, optional biases, a tied or separate output head.
- `qwen2`: Q, K and V biases.
- `qwen3`: QK-norm, and a head size that may differ from the model width.
- `phi3`: fused attention and gate/up projections, LongRoPE with short and long factors, partial rotary dimensions.
- `smollm3`: Llama-style layers where every fourth layer applies no RoPE.
- `granite`: embedding, attention, residual and logit scales.
- `granitemoe`: the Granite scales and a top-k mixture of experts.
- `olmo2`: no pre-norms, only post-attention and post-FFN norms, and a Q/K RMSNorm that spans the whole projection.
- `ernie4_5`: Llama-style layers with a head size larger than the width divided by the head count.
- `qwen35`: a hybrid stack; see [Hybrid models](#hybrid-models).
- `gemma4-assistant`: a four-layer, query-only decoder that reads the K/V cache of a `gemma4` target. It pairs only with a `gemma4` model of the same hidden width and vocabulary.

The common forward pass covers grouped-query attention, causal and sliding-window masks, `NORM` and `NEOX` RoPE (partial rotary, linear and YaRN scaling, per-pair factors), RMSNorm, SiLU and GELU gated FFNs, tied and separate output heads, and mixture-of-experts routing. Optional tensors (biases, QK-norm, fused projections, a separate `output.weight`) are detected by their presence in the file.

Not supported: other recurrent architectures such as Mamba and RWKV. Their GGUF fails with ``unsupported GGUF architecture `NAME`; supported: gemma4, gemma3, gemma2, llama, qwen2, qwen3, phi3, smollm3, granite, granitemoe, qwen35, olmo2, ernie4_5, gemma4-assistant``.

### Hybrid models

Qwen 3.5 mixes gated delta-net layers with gated full-attention layers. A delta-net layer keeps a causal convolution and a recurrent state per sequence instead of a K/V cache. Every `full_attention_interval`-th layer (every fourth in the 0.8B model) is gated full attention with partial NEOX RoPE. A `*-MTP` file adds a trailing prediction block that `DrafterSource::InTrunk` drafts with.

- Memory for the 0.8B model: 24,576 bytes of K/V per position (6 attention layers with 2 K/V heads of 256), plus a recurrent state of 20.2 MB that does not grow with the context.
- The state cannot be truncated. Prefix reuse restarts from one checkpoint that the engine takes while prefilling, right after the last end-of-turn token of the prompt (the end of the last complete message). The next turn of a conversation re-prefills the few tokens after that point, and its output equals a cold run's.
- The prompt cache usage a request reports counts what was reused.
- There is one checkpoint slot: a new checkpoint replaces the one before it.
- Rolling back a speculative verification keeps a copy of the state from before the call plus the per-row inputs of the recurrence. A verification call covers at most 17 rows (the pending token and 16 drafts). For the 0.8B model the total is about 61 MB (live state, checkpoint and pre-call copy) plus 445 KB per verified row, none of it growing with the context.
- A hybrid model cannot run on a GPU device: the load fails with `recurrent (gated delta-net) layers are not implemented on GPU devices; use LocalDevice::Cpu`.

## Weight formats

| Dtype | CPU | GPU |
|---|---|---|
| Q4_0, Q8_0 | stays quantized; cuca's batch-invariant quantized matmul kernel | repacked at load into burn's block layout; stays quantized |
| F32, F16, BF16 | held as f32 | uploaded as f32 (4 bytes per element) |
| Q4_1, Q5_0, Q5_1, K-quants | stays quantized; candle's quantized matmul | dequantized row by row at load, requantized to Q8_0 |
| Q8_1 | not covered by a test | dequantized row by row at load, requantized to Q8_0 |
| IQ family | rejected | rejected |

- Norm vectors, biases and scalars are f32 whatever their stored dtype. Resident weights are about the file size.
- The CPU reads any dtype candle's quantized matmul reads: F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0 and the K-quants (Q2_K to Q6_K). An IQ-family tensor makes the load fail with `` `PATH` is not a supported GGUF file: ... ``.
- A GPU requantizes every dtype other than Q4_0, Q8_0, F32, F16 and BF16 to Q8_0. Each element moves by at most half a Q8_0 step. That is a smaller error than the one a Q4_K, Q5_K or Q6_K file already carries, but GPU logits then differ from the CPU's by more than float rounding.
- A GPU file that mixes Q4_0 and Q8_0 weights, as the Llama 3.2 and Phi 3.5 presets do, keeps the weights of the scheme with fewer bytes as f32 where the two have similar shapes. That is exact and costs extra device memory.
- A weight over 1 GiB on the GPU is split into row chunks.
- Measured on Vulkan with `phi4-mini-q4_k_m` (AMD RX 9070 XT, Windows): the top-1 token equals the CPU's and the relative logit error against the dequantized CPU reference is 1.8e-2.
- Mixture of experts: expert tensors are split once at load into one quantized 2-D weight per expert.

## Tokenizers

`tokenizer.ggml.model` selects the scheme. All of it is Rust and is built once at load:

- `llama` is SentencePiece: score-ordered merging of UTF-8 characters, `<0xHH>` byte fallback, and the optional leading space of `tokenizer.ggml.add_space_prefix` (default true when the key is absent).
- `gpt2` is byte-level BPE: the text is cut into words by the pre-tokenizer that `tokenizer.ggml.pre` names, each word is mapped through the GPT-2 byte alphabet and merged by rank. The pre-tokenizers are hand-written linear splitters over Unicode general categories, not regular expressions, so their cost grows linearly with the text.
- `gemma4` (also `gpt2` with `pre` set to `gemma4`) replaces spaces with `▁`, never merges across a newline boundary, and falls back to `<0xHH>` byte tokens.

Any other `tokenizer.ggml.model` fails the load with ``unsupported tokenizer model `NAME` ``. The tokenizer each preset family uses:

| Family | `tokenizer.ggml.model` | `tokenizer.ggml.pre` |
|---|---|---|
| Gemma 4 | `gemma4` | none |
| Gemma 3, Gemma 2, Phi 3.5, ERNIE 4.5 | `llama` | `default` |
| TinyLlama | `llama` | absent |
| Qwen 2.5, Qwen 3 | `gpt2` | `qwen2` |
| Qwen 3.5 | `gpt2` | `qwen35` |
| DeepSeek-R1 distill | `gpt2` | `deepseek-r1-qwen` |
| Llama 3.2 | `gpt2` | `llama-bpe` |
| SmolLM 3 | `gpt2` | `smaug-bpe` |
| Phi 4 mini | `gpt2` | `gpt-4o` |
| Granite 3.1 and 3.3 | `gpt2` | `refact` |
| OLMo 2 | `gpt2` | `dbrx` |
| SmolLM 2 | `gpt2` | `smollm` |

### Pre-tokenizers

A `gpt2` model with no `pre`, or `default`, uses the default expression. SentencePiece models use no pre-tokenizer. The implemented values, by expression:

- `llama3`, `llama-v3`, `llama-bpe`, `falcon3`, `falcon-h1`, `pixtral`, `midm-2.0`, `lfm2`, `jina-v5-nano`: the Llama 3 expression. Whole words found in the vocabulary skip the merges, and a BOS token is added by default.
- `smaug-bpe`, `dbrx`: the Llama 3 expression without that shortcut.
- `qwen2`, `deepseek-r1-qwen`, `kormo`, `f2llmv2`, `megrez`, `stablelm2`, `hunyuan`, `solar-open`: the Qwen 2 expression.
- `qwen35`: the Qwen 2 expression with combining marks.
- `gpt-2`, `phi-2`, `mpt`, `olmo`, `jais`, `trillion`, `granite-docling`, `exaone4`, `mellum`, `modern-bert`, `gigachat`, `a.x-4.0`, `jina-es`, `jina-de`, `jina-v2-es`, `jina-v2-de`: the GPT-2 expression.
- `starcoder`, `refact`, `command-r`, `smollm`, `codeshell`, `exaone`, `minerva-7b`, `mellum2`: digits are split first, then the GPT-2 expression runs on the fragments between them.
- `gpt-4o`, `llama4`, `kanana2`, `talkie`, `minimax-m2`: the GPT-4o expression, with case classes from real Unicode categories.
- `gemma4`: the space-escaped scheme above.

Any other `pre` fails the load with ``unsupported pre-tokenizer `NAME` ``.

### Special tokens and BOS

- Special tokens (control, user-defined and unknown types) are cut out of the text longest-first before the scheme runs, so `<|im_start|>` in a rendered prompt is one token.
- The prompt gets a BOS token when `tokenizer.ggml.add_bos_token` (or the scheme's default) says so, and never a second one when the rendered template already starts with it.
- Phi-3-family SentencePiece models strip the whitespace that follows a special token, as their reference tokenizer does. No other family does.
- For a SentencePiece model with `add_space_prefix`, text after a special token such as `</s>` gets a leading space token, as in llama.cpp. HuggingFace's tokenizer does not add it for TinyLlama's chat template, so TinyLlama's prompt has one more token.

Verified by the tokenizer golden test in `src/provider/local/tokenizer/golden.rs`: for 11 models (Qwen 3, Qwen 2.5, Llama 3.2, Gemma 3 270M and 1B, SmolLM 3, Phi 3.5, Phi 4 mini, R1 distill, Granite 3.1 MoE, Qwen 3.5) the ids of 16 texts and 3 chat prompts equal the HuggingFace `tokenizer.json` ids. The test reads cached model files and prints `SKIP:` for a missing one. Where llama.cpp and HuggingFace differ, the expected ids follow llama.cpp, with two exceptions that follow HuggingFace: Phi 3.5's `rstrip` on special tokens, and the GPT-4o case classes.

## Chat templates

The template is the Jinja text stored in the GGUF, compiled with minijinja in the environment HuggingFace's `apply_chat_template` uses: blocks trimmed, leading block whitespace stripped, and Python string methods such as `.strip()`, `.startswith()` and `.split()` available. The template text is not modified, except that HuggingFace's generation and end-generation tags are rewritten to a no-op block.

Variables the template sees:

- `messages`: the request's messages in OpenAI shape.
- `tools`: defined only when the request declares at least one tool, because templates test `tools is not none`. Each tool is `{"type": "function", "function": {"name", "description", "parameters"}}`.
- `add_generation_prompt`: always true.
- `enable_thinking` and `thinking`: both true when the request enables thinking, otherwise both false.
- `preserve_thinking`: always false.
- `bos_token` and `eos_token`: the token texts, or empty.

Helpers the templates rely on:

- `raise_exception(message)` surfaces as a render error.
- `strftime_now(format)` returns the current UTC time.
- The `tojson` filter lays JSON out like Python's `json.dumps` (`", "` and `": "` separators, with `indent`, `sort_keys` and `ensure_ascii`). It keeps the order of an object's keys, as Python does.
- The `string` filter renders like Python's `str()`, so a dict prints as `{'a': 1}`.

Tool arguments stay JSON objects, as templates expect. The key order inside a tool's JSON schema (`parameters`) is the order of the request's `serde_json` map: sorted by key, unless your application enables the `preserve_order` feature of `serde_json`, in which case it is the order you wrote. Arguments of tool calls in the history are sorted the same way.

Verified by the unit test `renders_match_the_reference_jinja_environment` in `src/provider/local/chat/golden.rs`. For 13 templates (Gemma 3 1B and 270M, Gemma 4, Granite 3.3, Granite 3.1 MoE, Llama 3.2, Phi 3.5, Phi 4 mini, Qwen 2.5, Qwen 3, Qwen 3.5, the R1 Qwen distill, SmolLM 3) and 7 cases (a plain turn, a system prompt, a multi-turn history, thinking on and off, tools declared, a tool call with its result) the rendered text equals what Python `jinja2` renders with `trim_blocks`, `lstrip_blocks`, a `json.dumps(ensure_ascii=False)` `tojson` and `strftime_now` pinned to 2026-10-18 UTC. The fixtures are in `src/provider/local/testdata/`.

## Reasoning

When the template writes a reasoning block, the engine finds its markers from the vocabulary and the template text:

- `<|channel>thought` ... `<channel|>`: Gemma 4's channel. The header word `thought` and the whitespace after the opening marker are dropped.
- `<think>` ... `</think>`: Qwen 3, Qwen 3.5, SmolLM 3, Granite 3.3 and the R1 distills.

The text between the markers arrives as `Thinking` blocks. If the rendered prompt ends inside an open block, generation starts in thinking. Markers are matched on text, so a marker that the vocabulary spells as several tokens works like a single control token. A close marker with no open marker is dropped.

Whether a request can switch thinking off follows the template:

- Gemma 4, Qwen 3, Qwen 3.5 and SmolLM 3 read `enable_thinking`.
- Granite 3.3 reads `thinking`, which the request sets to the same value.
- The DeepSeek-R1 distills read neither and reason on every request.

`reasoning_tokens` in the usage counts the tokens inside reasoning blocks, markers included, and is reported only when the request enabled thinking. `LocalModel::capabilities()` reports `reasoning` and `can_disable_thinking` from the same detection.

## Tool calls

The template says how the model was taught to call tools, and the output parser follows it. The engine buffers a call whole and parses it into a `ToolCall` block. Declared tools reach a model only through its template, so a model whose template has no `tools` variable never sees them.

Syntaxes, in detection order:

- Gemma 4 custom: `<|tool_call>call:name{key:<|"|>value<|"|>}<tool_call|>` between control tokens. Detected when the template contains `<|tool_call>` and the vocabulary has both markers.
- XML, Qwen 3.5: `<tool_call>` ... `</tool_call>` around `<function=name>` and one `<parameter=key>` element per argument, with the value between a newline after the opening tag and a newline before the closing one. Detected when the template contains `<tool_call>`, `</tool_call>`, `<function=` and `<parameter=`.
- Tagged JSON, Qwen 2.5, Qwen 3, SmolLM 3: `<tool_call>` ... `</tool_call>` around one call object or a list of them. Detected when the template contains `<tool_call>`; without `</tool_call>` the call runs to the end of the reply.
- Tagged JSON with an alternate opener, Granite 3: `<|tool_call|>`, or the plain text `<tool_call>` that the model writes instead, followed by a JSON list to the end of the reply. A closing `<|/tool_call|>` is read only when the template contains it.
- Bare JSON, Llama 3.x: a reply that is `{"name": ..., "parameters": ...}`, optionally after `<|python_tag|>`. Other `<|python_tag|>` bodies become a call to `python` with a `code` argument. Detected when the template contains `tool_calls` and `"parameters"`.
- None, Gemma 3, Gemma 2, Phi 3.5, Phi 4 mini, OLMo 2, ERNIE 4.5, SmolLM 2, TinyLlama and the R1 distills: the output is plain text and `LocalCapabilities::tools` is false.

JSON call objects are read in the shapes models write them:

- The name comes from `name`, from a string `function` (`{"function": "get_weather", "arguments": {...}}`), or from the nested OpenAI shape (`{"type": "function", "function": {"name": ..., "arguments": ...}}`).
- The arguments come from `arguments` or `parameters`, in the nested object first and then beside it, as an object, `null`, or a string holding an object. A missing `arguments` means no arguments, except in a bare-JSON reply, where it marks the reply as not a call.
- An empty list, a missing name, or arguments that are not an object are a malformed call.

In the XML syntax the template writes a string argument as raw text and any other value (number, boolean, object, array) as JSON. The parser reads a value as JSON when it is valid JSON and as a string otherwise. A string argument that is itself valid JSON, such as `42` or `true`, therefore comes back as that JSON value.

A malformed call is reported as a `CucaError::Provider` stream item, `malformed tool call: <body>`, and the stream continues. Tool call ids are generated by the engine.

Generation stops at an end-of-generation token: the EOS, EOT and EOM ids of the metadata plus these vocabulary tokens when present: `<|eot_id|>`, `<|eom_id|>`, `<|im_end|>`, `<|end|>`, `<end_of_turn>`, `<turn|>`, `<|tool_response>`, `<|endoftext|>`, `<|end_of_text|>`, `<EOT>`, `_<EOT>`, `[EOT]`, the DeepSeek end-of-sentence token and `<end_of_utterance>`. A model with none of these and no metadata id fails to load.

## Presets

`ModelSource::PRESETS` has 23 entries, each pinned to a repository commit and a SHA-256. The size is the file. The K/V column is `LocalModel::kv_cache_bytes` at the default `context_length` (8,192 positions, or the model's own maximum when smaller), in decimal units. Weights are resident at about the file size on top of it.

| Preset | Architecture | Repository | Size | K/V |
|---|---|---|---|---|
| `gemma4-e2b-q4_0` (default) | `gemma4` | `ggml-org/gemma-4-E2B-it-GGUF` | 2.84 GB | 302 MB |
| `gemma4-e2b-q8_0` | `gemma4` | `ggml-org/gemma-4-E2B-it-GGUF` | 4.97 GB | 302 MB |
| `gemma4-e4b-q4_0` | `gemma4` | `ggml-org/gemma-4-E4B-it-GGUF` | 4.59 GB | 940 MB |
| `gemma4-e4b-q8_0` | `gemma4` | `ggml-org/gemma-4-E4B-it-GGUF` | 8.03 GB | 940 MB |
| `gemma3-270m-q8_0` | `gemma3` | `ggml-org/gemma-3-270m-it-GGUF` | 0.29 GB | 302 MB |
| `gemma3-1b-q4_0` | `gemma3` | `unsloth/gemma-3-1b-it-GGUF` | 0.72 GB | 436 MB |
| `gemma2-2b-q4_0` | `gemma2` | `second-state/gemma-2-2b-it-GGUF` | 1.63 GB | 1.74 GB |
| `qwen3-0.6b-q8_0` | `qwen3` | `Qwen/Qwen3-0.6B-GGUF` | 0.64 GB | 1.88 GB |
| `qwen3-1.7b-q8_0` | `qwen3` | `Qwen/Qwen3-1.7B-GGUF` | 1.83 GB | 1.88 GB |
| `qwen2.5-0.5b-q8_0` | `qwen2` | `Qwen/Qwen2.5-0.5B-Instruct-GGUF` | 0.68 GB | 201 MB |
| `r1-distill-qwen-1.5b-q4_0` | `qwen2` | `bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF` | 1.07 GB | 470 MB |
| `llama3.2-1b-q4_0` | `llama` | `bartowski/Llama-3.2-1B-Instruct-GGUF` | 0.77 GB | 537 MB |
| `smollm3-3b-q4_0` | `smollm3` | `bartowski/HuggingFaceTB_SmolLM3-3B-GGUF` | 1.81 GB | 1.21 GB |
| `phi3.5-mini-q4_0` | `phi3` | `bartowski/Phi-3.5-mini-instruct-GGUF` | 2.18 GB | 6.44 GB |
| `phi4-mini-q4_0` | `phi3` | `bartowski/microsoft_Phi-4-mini-instruct-GGUF` | 2.33 GB | 2.15 GB |
| `phi4-mini-q4_k_m` | `phi3` | `unsloth/Phi-4-mini-instruct-GGUF` | 2.49 GB | 2.15 GB |
| `granite3.3-2b-q4_0` | `granite` | `ibm-granite/granite-3.3-2b-instruct-GGUF` | 1.45 GB | 1.34 GB |
| `olmo2-1b-q4_0` | `olmo2` | `allenai/OLMo-2-0425-1B-Instruct-GGUF` | 0.89 GB | 1.07 GB (4,096 maximum) |
| `smollm2-360m-q8_0` | `llama` | `bartowski/SmolLM2-360M-Instruct-GGUF` | 0.39 GB | 671 MB |
| `tinyllama-1.1b-q4_0` | `llama` | `TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF` | 0.64 GB | 92 MB (2,048 maximum) |
| `granite3.1-moe-1b-q4_0` | `granitemoe` | `bartowski/granite-3.1-1b-a400m-instruct-GGUF` | 0.77 GB | 805 MB |
| `qwen3.5-0.8b-q8_0` | `qwen35` | `unsloth/Qwen3.5-0.8B-MTP-GGUF` | 0.83 GB | 201 MB, plus a 20.2 MB recurrent state |
| `gemma4-e2b-mtp-f16` | `gemma4-assistant` | `unsloth/gemma-4-E2B-it-GGUF` | 0.17 GB | none: a drafter, reads the target's cache |

- The K/V cost depends on the model more than on its size: a model with many K/V heads pays far more per position. Phi 3.5 mini has 32 K/V heads and needs 6.44 GB of cache at 8,192 positions for a 2.18 GB file. Lower `context_length` to shrink it.
- A model trained on fewer positions loads at its maximum by default (OLMo 2: 4,096; TinyLlama: 2,048). An explicit larger value fails with `context_length N exceeds the model maximum M`.
- `gemma4-e2b-mtp-f16` has `drafter: true`: it loads only through `LocalModelConfig::drafter` and cannot be the main model.
- `qwen3.5-0.8b-q8_0` carries the in-trunk prediction head, so `DrafterSource::InTrunk` works with it.
- Presets pin both the revision and the SHA-256; see [Model sources and presets](@/providers/local/model-sources.md#model-sources-and-presets) for the source types.

## Verification

### What was run

Three kinds of evidence back this page, from strongest to weakest:

- Unit tests in `src/provider/local/` run in CI on tiny random GGUF files and on committed fixtures: chat template renders, tokenizer pre-splitters, dequantization against candle, and the cross-architecture forward-pass tests (`TinySpec::all()`: `llama`, `qwen2`, `qwen3`, `phi3`, `smollm3`, `granitemoe`, `gemma3`, `gemma2`, `granite`, `olmo2`, `ernie4_5`; `gemma4` and `qwen35` have their own builders).
- The live suite `tests/provider_local.rs` runs one real preset per invocation. CI compiles it and every test prints a skip line, so the live results below come from recorded local runs, not from CI.
- Oracle runs against HuggingFace `transformers` (f32, CPU) use dumps under `.omp/local-models/ref/`, which is gitignored. Those results are reproducible only with that tooling, not from a clean checkout.

Per architecture:

- Verified in the live suite on the CPU: `gemma4`, `gemma3`, `gemma2`, `llama`, `qwen2`, `qwen3`, `qwen35`, `phi3`, `smollm3`, `granite`, `granitemoe`, `olmo2`.
- Also verified in the live suite on Vulkan (AMD RX 9070 XT, Windows): `gemma4` (E2B Q4_0), `llama` (Llama 3.2 1B), `phi3` (Phi 3.5 mini), `gemma2`, `olmo2`, `granite` (Granite 3.3 2B).
- `ernie4_5`: tiny-model tests and a recorded oracle run only. It has no preset and has not run in the live suite.
- `gemma4-assistant`: verified by `tests/provider_local_mtp.rs` as a drafter, and by tiny-model tests on the CPU and the GPU (the GPU ones only when `CUCA_LOCAL_DEVICE` names a GPU).
- Compile-checked only: every GPU backend other than Vulkan; see [Verified and compile-checked](@/providers/local/gpu.md#verified-and-compile-checked).

### Live results per preset

The suite ran each preset that has a model of its own on the CPU, in a release build for the host CPU, one model per run. Scenarios: a factual answer with exact usage, a prompt of about 1,300 tokens with a code word to retrieve, prefix reuse, and, where the template supports them, thinking on and off and a tool round trip (a `get_weather` call for Paris, then an answer from the result).

| Preset | Reasoning | Tool calls | Result |
|---|---|---|---|
| `gemma4-e2b-q4_0`, `gemma4-e2b-q8_0`, `gemma4-e4b-q4_0` | on and off | yes | all scenarios pass |
| `gemma4-e4b-q8_0` | on and off | yes | all pass except thinking, a limit of the model |
| `gemma3-270m-q8_0` | no | no syntax | long-context retrieval is a known limit; the rest pass |
| `gemma3-1b-q4_0` | no | no syntax | all applicable scenarios pass |
| `qwen3-0.6b-q8_0`, `qwen3-1.7b-q8_0` | on and off | yes | all scenarios pass |
| `qwen2.5-0.5b-q8_0` | no | yes | all applicable scenarios pass |
| `llama3.2-1b-q4_0` | no | yes | the tool round trip is a known limit |
| `smollm3-3b-q4_0` | on and off | yes | all seven run; thinking and the tool answer leg are known limits |
| `phi3.5-mini-q4_0`, `phi4-mini-q4_0`, `phi4-mini-q4_k_m` | no | no syntax | all applicable scenarios pass |
| `r1-distill-qwen-1.5b-q4_0` | always | no syntax | thinking and prefix reuse pass; long-context retrieval is a known limit |
| `granite3.1-moe-1b-q4_0` | no | yes | all scenarios pass |
| `qwen3.5-0.8b-q8_0` | on and off | yes, XML | all seven pass, CPU only |
| `gemma2-2b-q4_0`, `olmo2-1b-q4_0` | no | no syntax | all applicable scenarios pass, on the CPU and on Vulkan |
| `granite3.3-2b-q4_0` | switched on by the request | yes | all pass except long-context retrieval, a known limit |
| `smollm2-360m-q8_0` | no | no syntax | all applicable scenarios pass |
| `tinyllama-1.1b-q4_0` | no | no syntax | all applicable scenarios pass; the oracle prompt count has one extra token, a known limit |

The Vulkan engine ran the same suite for `gemma4-e2b-q4_0`, `llama3.2-1b-q4_0`, `phi3.5-mini-q4_0`, `gemma2-2b-q4_0`, `olmo2-1b-q4_0` and `granite3.3-2b-q4_0`, with the same results as on the CPU.

### Known limits

A known limit is reported as `MATRIX|<scenario>|KNOWN-LIMIT` by the suite, with its evidence. The entries live in `known_limit` in `tests/provider_local.rs`.

- `gemma3-270m-q8_0` answers with its end-of-turn token at once on prompts of 750 and 1,110 tokens, while a 570-token prompt still retrieves the code word. The HuggingFace f32 model does the same, so this is the model.
- `gemma4-e4b-q8_0` answers `72` at once, without a reasoning block, to `What is 9 * 8?` with thinking on. This file derives from the base `google/gemma-4-E4B-it` checkpoint, and its Q4_0 sibling from the QAT checkpoint. At the first position `7` scores 28.77 against 28.37 for `<|channel>` in HuggingFace f32 and bf16, and 28.81 against 28.34 for the file's dequantized weights. The QAT-derived Q4_0 file opens the channel (29.28 against 28.91). On the harder question `What is 17 * 23? Think carefully.` the Q8_0 file does reason.
- `llama3.2-1b-q4_0` calls the tool again after the tool result when the tool is still declared on the follow-up request. HuggingFace f32 does the same on that prompt. With no tool declared on the follow-up it answers from the result.
- `r1-distill-qwen-1.5b-q4_0` is marginal at retrieving a code word from a 1,100-token prompt. HuggingFace f32 answers `PELCAN` (misspelled). The Q4_0 model either answers or reasons for more than 4,096 tokens, depending on how its greedy trajectory falls on near-ties.
- `granite3.3-2b-q4_0` answers `PEICAN` where HuggingFace f32 answers `PELICAN`, on the 1,300-token retrieval prompt. HuggingFace on the same file's dequantized weights, and the engine's exact arithmetic, also give `PEICAN`: quantization flips a near-tie at the second token.
- `tinyllama-1.1b-q4_0`: one prompt token more than HuggingFace's tokenizer, from the SentencePiece rule in [Special tokens and BOS](#special-tokens-and-bos).
- `smollm3-3b-q4_0`, two scenarios that are quantized-arithmetic near-ties, not properties of the model. The engine's exact arithmetic (dequantized f32, no Q8_0 activation rounding) equals the dequantized reference run in HuggingFace. The production path rounds activations to Q8_0 and flips a close logit:
  - Thinking on `What is 9 * 8?`: exact arithmetic closes `</think>` at step 1,709 (HuggingFace f32 on the original weights: 535). The production path flips a 0.12-logit tie at the 154th reasoning token, then reasons for 3,070 tokens without concluding.
  - The tool round trip's answer leg with the tool still declared: exact arithmetic gives `The` 18.966 against the tool-call token 18.793 and answers. The production path flips it (18.663 against 19.237) and calls the tool again.
  - Give reasoning models a generous `max_tokens`.
- ERNIE 4.5 0.3B (`ernie4_5`, a pretrained model, not an instruct one) runs and matches the oracle numerically, but has no preset. Every Q8_0 and Q4_0 GGUF of it that was found (29 of 283 files in 54 Hugging Face repositories) lacks `tokenizer.ggml.add_space_prefix`, so llama.cpp's SentencePiece default (a space in front of the text after a special token) applies, and the model answers the capital question with `France` where the HuggingFace tokenizer gives `Paris`. Current llama.cpp conversions write that key as `false`.

### Numerical parity with HuggingFace

Method: HuggingFace `transformers`, f32, CPU, against the engine's CPU path, on prompts of 24 chat tokens, a 700-token prompt and, for Phi and Gemma 2, a prompt of about 4,200 tokens.

| Path compared | Result |
|---|---|
| Reference arithmetic (weights dequantized to f32) against `transformers`, the 13 first catalog models | top-1 token and top-10 set equal, maximum absolute logit difference below 5e-4, greedy output equal for 24 of 24 tokens on the chat prompt and 8 of 8 on the 700-token prompt; the Phi 4,200-token prompt exercises LongRoPE's long factors and matches |
| The same, Gemma 2, OLMo 2, Granite 3.3, ERNIE 4.5, SmolLM 2, TinyLlama | maximum absolute logit difference 0.000 on the top-k, greedy output equal on every prompt, the 4,200-token Gemma 2 prompt included |
| Production quantized CPU path against `transformers`, the 13 first models | top-1 token kept on all 48 prompt cases; maximum absolute logit difference between 0.05 and 1.13, the cost of quantized weights |
| The same, the six later models | top-1 token kept on all 24 prompt cases |
| Qwen 3.5 0.8B Q8_0, reference arithmetic | maximum absolute logit difference below 5e-5; the production path keeps the top-1 token |
| GPU (Vulkan) against the CPU dequantized reference | relative error 2e-6 to 2e-5, greedy token ids equal, on Qwen 3 0.6B and 1.7B, Qwen 2.5 0.5B, Llama 3.2 1B, Gemma 3 270M and 1B, SmolLM 3, Phi 3.5, Granite 3.1 MoE, Gemma 4 E2B and the R1 distill |

On the CPU, every Q4_0 and Q8_0 matrix multiplication runs through cuca's own batch-invariant quantized kernel, written in safe Rust; other dtypes use candle. Prefill is row-exact: the logits of a prompt row are bit-identical whether the prompt is prefilled in one chunk, in several, or decoded one token at a time, so reusing a cached prefix never changes the output.

Measured prefill speed with that kernel (1,300-token prompt in chunks of 256, release build with `-C target-cpu=native`, 32-core CPU): `gemma4-e2b-q4_0` 86.5 tokens/s, `phi3.5-mini-q4_0` 76.8, `qwen3-0.6b-q8_0` 249.6.

## Using another GGUF

Any file whose architecture is in the table and whose tokenizer is one of the schemes above loads through `ModelSource::File(path)` or `ModelSource::hugging_face(repo, revision, file)`. The CPU reads every dtype candle reads. A model the engine accepts but that never ran through the live suite is unverified. The first check is that its rendered prompt and token ids equal its reference tokenizer's, which is how the catalog models were checked.

## Errors

Loading a file the engine cannot run fails with a `CucaError::Provider` that names the cause:

- An unknown architecture: ``unsupported GGUF architecture `NAME`; supported: ...``.
- An unknown tokenizer model or pre-tokenizer: ``unsupported tokenizer model `NAME` `` or ``unsupported pre-tokenizer `NAME` ``.
- A file candle cannot parse, such as one with an IQ-family tensor: ``` `PATH` is not a supported GGUF file: ... ```.
- A missing tensor or metadata key, including `tokenizer.chat_template`: ``GGUF metadata `KEY`: missing``.
- A tensor with an unexpected shape, or a row length that is not a multiple of its dtype's block size.
- A checkpoint whose value length differs from its key length (``GGUF metadata `ARCH.attention.value_length`: ... only equal key and value lengths are supported``), or with odd or zero attention key lengths.
- No end-of-generation token: the metadata names none and the vocabulary holds no known terminator.
- `DrafterSource::InTrunk` on a file without a prediction head: ``this `ARCH` file has no in-trunk multi-token-prediction head ...``.
- A `gemma4-assistant` drafter with a target that is not `gemma4`, or whose hidden width or vocabulary differs.

Device, configuration and download errors are on [Local (in-process)](@/providers/local/limits-and-errors.md#errors).
