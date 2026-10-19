+++
title = "Local models and formats"
description = "What provider-local runs and how: the supported architectures, the tokenizers (SentencePiece, byte-level BPE with the pre-tokenizers it needs, Gemma 4's), the model's own chat template, reasoning and tool-call formats, the verified preset catalog with sizes and K/V cost, and the limits that were measured."
template = "page.html"
weight = 9
+++

# Local models and formats

<dl class="page-facts">
<dt>In one line</dt>
<dd>The engine loads a GGUF file the way llama.cpp does: the architecture, the tokenizer and the chat template all come from the file, and the stream that comes back has the same <code>Text</code>, <code>Thinking</code> and <code>ToolCall</code> blocks whatever the model.</dd>
<dt>You need</dt>
<dd>The <code>provider-local</code> feature; see <a href="@/providers/local.md">Local (in-process)</a> for loading, configuration and the GPU.</dd>
<dt>Read this if</dt>
<dd>You are choosing a model, loading a GGUF that is not a preset, or wondering why a model's reply is parsed the way it is.</dd>
</dl>

No part of this is specific to one model family. The loader reads every dimension
from the file's metadata and tensor shapes, picks the forward-pass behaviour from
`general.architecture`, builds the tokenizer from `tokenizer.ggml.*`, and
renders prompts with the Jinja chat template stored in `tokenizer.chat_template`.
The design follows llama.cpp (its sources were the reference; nothing of it is
linked or called), and every model below was checked against HuggingFace
`transformers` where a reference could be produced.

## Architectures

`general.architecture` selects the behaviour. An architecture that is not in this
table fails the load with a `CucaError::Provider` that names it and lists the
supported ones.

| Architecture | Models | What it adds to the common forward pass |
|---|---|---|
| `gemma4` | Gemma 4 (E2B, E4B) | sliding and full layers with their own head sizes and RoPE bases, shared K/V layers, per-layer embeddings, weightless value norm, per-layer output scale, a final logit soft-cap |
| `gemma2` | Gemma 2 (2B) | alternating sliding-window and full layers, an attention logit soft-cap and a final logit soft-cap, post-attention and post-FFN norms, no QK-norm |
| `gemma3` | Gemma 3 (270M, 1B) | sliding and full layers (every sixth layer full), QK-norm, post-attention and post-FFN norms |
| `llama` | Llama 3.2, SmolLM 2, TinyLlama | the Llama 3 RoPE scaling that the file carries as `rope_freqs.weight`, optional biases, tied or separate output head |
| `olmo2` | OLMo 2 | no pre-norms: only post-attention and post-FFN norms, and a Q/K RMSNorm that spans the whole projection |
| `ernie4_5` | ERNIE 4.5 (dense) | Llama-style layers with a head size larger than the width divided by the heads |
| `granite` | Granite 3.3 (dense) | embedding, attention, residual and logit scales |
| `qwen2` | Qwen 2.5, DeepSeek-R1-Distill-Qwen | Q/K/V biases |
| `qwen3` | Qwen 3 | QK-norm, a head size that may differ from the model width |
| `phi3` | Phi 3.5, Phi 4-mini | fused attention and gate/up projections, LongRoPE with short and long factors, partial rotary dimensions |
| `smollm3` | SmolLM 3 | Llama-style layers with some layers that apply no RoPE |
| `granitemoe` | Granite 3 MoE | the Granite scales and a top-k mixture of experts |
| `qwen35` | Qwen 3.5 (0.8B) | a hybrid stack: gated delta-net layers (a causal convolution and a recurrent state per sequence instead of a K/V cache) with gated full attention in every fourth layer, partial rotary dimensions, and an optional trailing prediction block that `DrafterSource::InTrunk` drafts with; CPU only |
| `gemma4-assistant` | the Gemma 4 MTP drafter | loaded only through `LocalModelConfig::drafter` |

The common forward pass covers grouped-query attention, causal and
sliding-window masks, NORM and NEOX RoPE (partial rotary, linear and YaRN
scaling, per-pair factors), RMSNorm, SiLU and GELU gated FFNs, tied and separate
output heads and mixture-of-experts routing. Optional tensors (biases, QK-norm,
fused projections, a separate `output.weight`) are detected by their presence in
the file. A checkpoint whose value length differs from its key length is
rejected, naming the key. Matrix weights may be any dtype candle's quantized
matmul reads (F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0 and the K-quants);
IQ-family quantizations are rejected.

A hybrid model keeps a recurrent state next to the K/V cache of its attention
layers (for Qwen 3.5 0.8B: 24,576 bytes of K/V per position, and 20.2 MB of
state that does not grow with the context). The state cannot be truncated, so
prefix reuse restarts from a checkpoint the engine takes while prefilling a
prompt, right after its last end-of-turn token (the end of the last complete
message), instead of from the exact end of the shared prefix: the next turn of a
conversation re-prefills the few tokens after that point, and its output is
exactly what a cold run produces. The prompt-cache usage a request reports
counts what was really reused. There is one checkpoint slot (a new checkpoint
replaces the old one), and rolling back a speculative verification keeps a copy
of the state from before the call plus the per-row inputs of the recurrence: for
Qwen 3.5 0.8B about 61 MB in all (the live state, the checkpoint and the pre-call
copy) and 445 KB per verified row, none of it growing with the context. A hybrid
model cannot run on a GPU device: the load fails with `recurrent (gated delta-net) layers are not implemented on GPU
devices; use LocalDevice::Cpu`.

Not supported: other recurrent architectures (Mamba, RWKV). Loading their GGUF
fails with `unsupported GGUF architecture '...'; supported: gemma4, gemma3,
gemma2, llama, qwen2, qwen3, phi3, smollm3, granite, granitemoe, qwen35, olmo2,
ernie4_5, gemma4-assistant`.

## Tokenizers

`tokenizer.ggml.model` selects the scheme, all of it written in Rust and built
once at load:

- `llama` is SentencePiece: score-ordered merging of UTF-8 characters, `<0xHH>`
  byte fallback, and the optional leading space of `tokenizer.ggml.add_space_prefix`.
- `gpt2` is byte-level BPE: the text is cut into words by the pre-tokenizer
  that `tokenizer.ggml.pre` names, each word is mapped through the GPT-2 byte
  alphabet and merged by rank. The pre-tokenizers are hand-written linear
  splitters over Unicode general categories, not regular expressions, so
  their cost grows linearly with the text.
- `gemma4` (also `gpt2` with `pre = gemma4`) replaces spaces with `▁`, never
  merges across a newline boundary and falls back to `<0xHH>` byte tokens.

The `pre` values that are implemented, by family:

| `tokenizer.ggml.pre` | Used by |
|---|---|
| missing, `default` | the default expressions, for a `gpt2` model with no `pre` (SentencePiece models use no pre-tokenizer) |
| `llama3`, `llama-v3`, `llama-bpe`, `falcon3`, `falcon-h1`, `pixtral`, `midm-2.0`, `lfm2`, `jina-v5-nano` | Llama 3 family (whole words found in the vocabulary skip the merges) |
| `smaug-bpe`, `dbrx` | the Llama 3 expression without that shortcut (SmolLM 3) |
| `qwen2`, `deepseek-r1-qwen`, `kormo`, `f2llmv2`, `megrez`, `stablelm2`, `hunyuan`, `solar-open` | Qwen 2, 2.5 and 3, DeepSeek-R1 distills |
| `qwen35` | the Qwen 2 expression with combining marks |
| `gpt-2`, `phi-2`, `mpt`, `olmo`, `jais`, `trillion`, `granite-docling`, `exaone4`, `mellum`, `modern-bert`, `gigachat`, `a.x-4.0`, `jina-es`, `jina-de`, `jina-v2-es`, `jina-v2-de` | the GPT-2 expression |
| `starcoder`, `refact`, `command-r`, `smollm`, `codeshell`, `exaone`, `minerva-7b`, `mellum2` | digits split first, then the GPT-2 expression (Granite 3) |
| `gpt-4o`, `llama4`, `kanana2`, `talkie`, `minimax-m2` | the GPT-4o expression (Phi-4-mini) |
| `gemma4` | the space-escaped scheme above |

Any other `pre` value fails the load with `unsupported pre-tokenizer ...`, naming
it; so does any other `tokenizer.ggml.model`.

Special tokens (control, user-defined and unknown types) are cut out of the
text longest-first before the scheme runs, so `<|im_start|>` in a rendered
prompt is one token. The prompt gets a BOS token when the model's
`tokenizer.ggml.add_bos_token` (or the scheme's default) says so, but never a
second one when the rendered template already starts with it. Phi-3-family
SentencePiece models strip the whitespace that follows a special token, as
their reference tokenizer does. On the catalog models the token ids of a set of
texts and of the rendered chat prompt equal the HuggingFace tokenizers' ids.

## The chat template

The template is the Jinja text stored in the GGUF, compiled with minijinja in
the environment HuggingFace's `apply_chat_template` uses (blocks trimmed, Python
string methods such as `.strip()`, `.startswith()` and `.split()` available). It
sees `messages`, `bos_token`, `eos_token`, `add_generation_prompt` (always
true), `enable_thinking` and `thinking` (both true when the request enables
thinking) and `tools`.
`tools` is defined only when the request declares tools, since templates test
`tools is not none`. The helpers templates rely on exist: `raise_exception`,
`strftime_now` (UTC), a `tojson` filter laid out like Python's `json.dumps`
(including `indent`, `sort_keys` and `ensure_ascii`), a `string` filter that
renders like Python's `str()`, and HuggingFace's `{% generation %}` tags (which
produce no output). Tool arguments stay JSON objects, as templates expect.

`tojson` keeps the order of an object's keys, as Python does, and each declared
tool reaches the template as `{"type", "function": {"name", "description",
"parameters"}}`, the order OpenAI documents and Python callers write, so the tool
block of a prompt matches what the model saw in training. The key order inside a
tool's JSON schema (`parameters`) is the order of the request's `serde_json` map:
sorted by key, unless your application enables `serde_json`'s `preserve_order`
feature, in which case it is the order you wrote. Arguments of tool calls in the
history are sorted the same way.

Nothing about the prompt is hard-coded per model. For the catalog templates the
rendered text of seven cases (a plain turn, a system prompt, a multi-turn history,
thinking on and off, tools declared, and a tool call with its result) is
byte-identical to what HuggingFace's Python `jinja2` environment renders, for
thirteen models (Qwen 2.5, Qwen 3, Qwen 3.5, Llama 3.2, Granite 3.1 MoE and 3.3 and
SmolLM 3 among them).

## Reasoning

When the template writes a reasoning block, the engine finds it from the
vocabulary and the template text: Gemma 4's `<|channel>thought` ... `<channel|>`
channel, or `<think>` ... `</think>` tags. The text between the markers arrives
as `Thinking` blocks; a header word such as `thought` and the whitespace after a
marker are dropped. If the rendered prompt ends inside an open block (a
template that already opened it), generation starts in thinking. Markers are
matched on text, so a marker the vocabulary spells over several tokens works like
a single control token.

Whether thinking can be switched off follows the template: Gemma 4, Qwen 3, Qwen 3.5
and SmolLM 3 read `enable_thinking` (Granite 3.3 reads `thinking`, which the request
sets to the same value); the DeepSeek-R1 distills do not, and reason on
every request. `reasoning_tokens` in the usage counts the tokens inside reasoning
blocks, markers included, and is reported only when the request enabled thinking.

## Tool calls

The template says how the model was taught to call tools, and the output parser
follows it. The call is buffered whole and parsed into a `ToolCall` block; a
malformed one is reported as a `CucaError::Provider` quoting the raw text
(`malformed tool call: ...`) and the stream continues.

| Syntax | Models | What the parser reads |
|---|---|---|
| custom | Gemma 4 | `<\|tool_call>call:name{key:<\|"\|>value<\|"\|>}<tool_call\|>` between control tokens |
| tagged JSON | Qwen 2.5 and 3, SmolLM 3 | `<tool_call>` ... `</tool_call>` around one call object or a list of them |
| tagged JSON, no closing tag | Granite 3 | `<\|tool_call\|>` (or the plain text `<tool_call>` the model writes) followed by a JSON list, to the end of the reply |
| XML | Qwen 3.5 | `<tool_call>` ... `</tool_call>` around `<function=name>` and one `<parameter=key>` element per argument, the value between a newline after the opening tag and a newline before the closing one |
| bare JSON | Llama 3.x | a reply that is `{"name": ..., "parameters": ...}`, optionally after `<\|python_tag\|>`; other `<\|python_tag\|>` bodies become a call to `python` with a `code` argument |
| none | Gemma 3, Phi 3.5 and 4-mini, R1 distills | the text is plain text |

The JSON call objects (tagged and bare) are read in the shapes models write them:
the name from `name`, from a string `function` (`{"function": "get_weather",
"arguments": {...}}`) or from the nested OpenAI shape (`{"type": "function",
"function": {"name": ..., "arguments": ...}}`), and the arguments from
`arguments` or `parameters`, as an object or as a string holding one.

In the XML syntax the template writes a string argument as its raw text and any
other value (number, boolean, object, array) as JSON, so the parser reads a value
as JSON when it is valid JSON and as a string otherwise. A string argument that is
itself valid JSON (`42`, `true`) therefore comes back as that JSON value, an
ambiguity of the format.

Declared tools are shown to a model only through its template, so a model whose
template has no `tools` variable never sees them.

Generation stops at the model's end-of-generation tokens: the EOS, EOT and EOM
ids of the metadata plus the turn and sequence terminators the vocabulary
holds (`<|im_end|>`, `<|eot_id|>`, `<end_of_turn>`, `<|end|>`, `<turn|>`, ...).

## Presets

`ModelSource::PRESETS` has these entries, each pinned to a repository commit and
a SHA-256. The size is the file; the K/V column is
`LocalModel::kv_cache_bytes` at 8192 positions, the default `context_length`
capped at the model's own maximum, measured on the loaded model (decimal units);
the weights are resident at about the file size besides it.

| Preset | Architecture | File | Size | K/V at 8192 |
|---|---|---|---|---|
| `gemma4-e2b-q4_0` (default) | `gemma4` | `ggml-org/gemma-4-E2B-it-GGUF` | 2.84 GB | 302 MB |
| `gemma4-e2b-q8_0` | `gemma4` | `ggml-org/gemma-4-E2B-it-GGUF` | 4.97 GB | same as Q4_0 (same architecture) |
| `gemma4-e4b-q4_0` | `gemma4` | `ggml-org/gemma-4-E4B-it-GGUF` | 4.59 GB | 940 MB |
| `gemma4-e4b-q8_0` | `gemma4` | `ggml-org/gemma-4-E4B-it-GGUF` | 8.03 GB | same as Q4_0 (same architecture) |
| `gemma2-2b-q4_0` | `gemma2` | `second-state/gemma-2-2b-it-GGUF` | 1.63 GB | 1.74 GB |
| `gemma3-270m-q8_0` | `gemma3` | `ggml-org/gemma-3-270m-it-GGUF` | 0.29 GB | 302 MB |
| `gemma3-1b-q4_0` | `gemma3` | `unsloth/gemma-3-1b-it-GGUF` | 0.72 GB | 436 MB |
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
| `olmo2-1b-q4_0` | `olmo2` | `allenai/OLMo-2-0425-1B-Instruct-GGUF` | 0.89 GB | 1.07 GB at its 4,096 maximum |
| `smollm2-360m-q8_0` | `llama` | `bartowski/SmolLM2-360M-Instruct-GGUF` | 0.39 GB | 671 MB |
| `tinyllama-1.1b-q4_0` | `llama` | `TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF` | 0.64 GB | 92 MB at its 2,048 maximum |
| `granite3.1-moe-1b-q4_0` | `granitemoe` | `bartowski/granite-3.1-1b-a400m-instruct-GGUF` | 0.77 GB | 805 MB |
| `qwen3.5-0.8b-q8_0` | `qwen35` | `unsloth/Qwen3.5-0.8B-MTP-GGUF` | 0.83 GB | 201 MB, plus a 20.2 MB recurrent state |
| `gemma4-e2b-mtp-f16` | `gemma4-assistant` | `unsloth/gemma-4-E2B-it-GGUF` | 0.17 GB | none (drafter; reads the target's cache) |

The K/V cost depends on the model more than on its size: a model with many K/V
heads pays far more per position (Phi-3.5-mini, 32 K/V heads, needs 6.44 GB of
cache at 8192 positions for a 2.18 GB file). Lower `context_length` to shrink it.
`LocalModelConfig::context_length` defaults to the smaller of 8,192 and the
model's own maximum, so a model trained on fewer positions (OLMo 2: 4,096;
TinyLlama: 2,048) loads with its maximum; an explicit larger value fails with
`context_length N exceeds the model maximum M`.

The tokenizer each preset uses (`tokenizer.ggml.model` / `pre`): Gemma 4
`gemma4`; Gemma 3 and Phi 3.5 `llama` (SentencePiece); Qwen 2.5 and 3 `gpt2` /
`qwen2`; Qwen 3.5 `gpt2` / `qwen35`; the R1 distill `gpt2` / `deepseek-r1-qwen`; Llama 3.2 `gpt2` /
`llama-bpe`; SmolLM 3 `gpt2` / `smaug-bpe`; Phi-4-mini `gpt2` / `gpt-4o`;
Granite 3 `gpt2` / `refact`; Gemma 2 `llama`; OLMo 2 `gpt2` / `dbrx`;
SmolLM 2 `gpt2` / `smollm`; TinyLlama `llama`.

### What each preset was verified to do

The live suite (`tests/provider_local.rs`) ran every preset in the table that has
a model of its own on the CPU, in a release build for the host CPU, one model per
run: a factual answer with exact usage, a prompt of about 1,300 tokens with a code
word to retrieve, prefix reuse, and, where the template supports them, thinking on
and off and a tool round trip (a call to `get_weather` for Paris, then an answer
from the result). Every preset with a model of its own was run, the Gemma 4 E2B and
E4B ones included. The Vulkan engine ran the same suite for Gemma 4 E2B Q4_0, Llama
3.2 1B, Phi 3.5 mini, Gemma 2 2B, OLMo 2 1B and Granite 3.3 2B, with the same
results as on the CPU; Qwen 3.5 does not load on a GPU device (see above).

| Preset | Reasoning | Tool calls | Notes |
|---|---|---|---|
| `gemma4-e2b-q4_0`, `gemma4-e2b-q8_0`, `gemma4-e4b-q4_0` | on and off | yes | all scenarios pass |
| `gemma4-e4b-q8_0` | on and off | yes | all scenarios pass but the thinking one, a limit of the model (see below) |
| `gemma3-270m-q8_0` | no | no tool syntax | cannot retrieve a code word from 750 or more tokens (see below); the other scenarios pass |
| `gemma3-1b-q4_0` | no | no tool syntax | all applicable scenarios pass |
| `qwen3-0.6b-q8_0`, `qwen3-1.7b-q8_0` | on and off | yes | all scenarios pass |
| `qwen2.5-0.5b-q8_0` | no | yes | all applicable scenarios pass |
| `llama3.2-1b-q4_0` | no | yes | calls the tool again after the result when tools are still declared (see below) |
| `smollm3-3b-q4_0` | on and off | yes | all seven scenarios run; the thinking scenario and the tool round trip's answer leg are known limits of the quantized arithmetic, not of the model (see below) |
| `phi3.5-mini-q4_0` | no | no tool syntax | all applicable scenarios pass |
| `phi4-mini-q4_0`, `phi4-mini-q4_k_m` | no | no tool syntax | all applicable scenarios pass |
| `r1-distill-qwen-1.5b-q4_0` | always | no tool syntax | thinking and prefix reuse pass; the long-context retrieval is marginal (see below) |
| `granite3.1-moe-1b-q4_0` | no | yes | all scenarios pass; the 1B model may write `"function"` for `"name"`, and both are read |
| `qwen3.5-0.8b-q8_0` | on and off | yes (XML) | all seven scenarios pass, on the CPU only; with its in-trunk drafter the greedy output equals plain decoding |
| `gemma2-2b-q4_0` | no | no tool syntax | all applicable scenarios pass, on the CPU and on Vulkan |
| `olmo2-1b-q4_0` | no | no tool syntax | all applicable scenarios pass, on the CPU and on Vulkan |
| `granite3.3-2b-q4_0` | yes, switched on by the request | yes | all scenarios pass but the long-context retrieval, which is a limit of this file (see below) |
| `smollm2-360m-q8_0` | no | no tool syntax | all applicable scenarios pass |
| `tinyllama-1.1b-q4_0` | no | no tool syntax | all applicable scenarios pass; one prompt token more than HuggingFace's (see below) |

### Limits measured against the reference

- **Gemma 3 270M** answers with its end-of-turn token at once when the prompt is
  750 or 1,110 tokens long (a 570-token prompt still retrieves the code word).
  The HuggingFace f32 model does exactly the same on those prompts, so this is the
  model, not the engine; the live suite reports that scenario as a known limit for
  it.
- **Gemma 4 E4B Q8_0** answers `72` at once, without a reasoning block, when asked
  `What is 9 * 8?` with thinking on. The E4B Q8_0 and Q4_0 files come from different
  checkpoints: the Q8_0 file from the base `google/gemma-4-E4B-it` (its dequantized
  tensors are within 0.6 % of it), the Q4_0 file from the QAT checkpoint
  (`google/gemma-4-E4B-it-qat-q4_0-unquantized`). The base checkpoint itself skips the
  thought channel on that question in HuggingFace f32 and bf16 (at the first
  position `7` scores 28.77 against 28.37 for `<|channel>`), as the dequantized
  Q8_0 weights do (28.81 against 28.34), while the QAT-derived Q4_0 file opens the
  channel (29.28 against 28.91). On a harder question (`What is 17 * 23? Think
  carefully.`) the Q8_0 file does reason. The live suite reports that scenario as a
  known limit for it.
- **Llama 3.2 1B** answers a tool result by calling the tool again when the tool
  is still declared on the follow-up request; HuggingFace f32 does the same on the
  same prompt. With no tool declared on the follow-up it answers from the result.
- **DeepSeek-R1-Distill-Qwen 1.5B** is marginal at retrieving a code word from a
  1,100-token prompt: HuggingFace f32 answers `PELCAN` (misspelled), and the Q4_0
  model either answers or reasons for over 4,096 tokens depending on how its
  greedy trajectory falls on near-ties. The live suite reports that scenario as a
  known limit for it.
- **Granite 3.3 2B Q4_0** answers `PEICAN` where the HuggingFace f32 model
  answers `PELICAN`, on the 1,300-token retrieval prompt. HuggingFace
  `transformers` run on the same file's dequantized weights, and the engine's exact
  arithmetic, also give `PEICAN`: quantization flips a near-tie at the second token
  (`PE` then `IC` instead of `L`), so it is this Q4_0 file, and the live suite reports that scenario as a known limit for it.
- **ERNIE 4.5 0.3B** (`ernie4_5`, a pretrained model rather than an instruct
  one) runs and matches the oracle numerically, but there is no preset: every
  ungated Q8_0 and Q4_0 GGUF of it that was checked (281 files) lacks
  `tokenizer.ggml.add_space_prefix`, so llama.cpp's SentencePiece default (a space
  in front of the text after a special token) applies, and the model then answers
  the capital question with `France` where the HuggingFace tokenizer gives
  `Paris`. Current llama.cpp conversions write that key as `false`.
- **TinyLlama** and any SentencePiece model: following llama.cpp, the text after a
  special token such as `</s>` gets a leading space token, one more token than
  the HuggingFace tokenizer produces for TinyLlama's chat template.
- **SmolLM 3 3B Q4_0**, two scenarios, a different kind of limit from the ones above:
  quantized-arithmetic near-ties, not properties of the model. (1) Thinking on
  `What is 9 * 8?`: the engine's exact arithmetic (dequantized f32, no Q8_0
  activation rounding) equals the dequantized reference run in HuggingFace for all
  1,710 tokens and closes `</think>` at step 1,709 (HuggingFace f32 on the
  original weights: 535). The production path rounds activations to Q8_0, which
  flips a 0.12-logit tie at the 154th reasoning token (index 153 after `<think>` and
  the newline; index 155 of the generated ids); the run then reasons for 3,070 tokens
  without concluding and loops (`Wait, no, 9*8 is 72?`). (2) The tool round trip's
  answer leg with the tool still declared: the exact arithmetic gives `The` 18.966
  against the tool-call token 18.793 (margin 0.173), identical to the dequantized
  HuggingFace run (HuggingFace f32: a 2.06 margin) and answers from the result; the
  production path flips it (18.663 against 19.237) and calls the tool again. The
  live suite reports both as known limits. Give reasoning models a generous
  `max_tokens`.

## Numerical parity with HuggingFace

Measured against HuggingFace `transformers` (f32, CPU) on the 13 first catalog
models, and again on six later ones (Gemma 2, OLMo 2, Granite 3.3, ERNIE 4.5,
SmolLM 2, TinyLlama): the reference arithmetic matches exactly on
those six too (`max|Δlogit|` 0.000 on the top-k, greedy output equal over every
prompt, the 4,200-token Gemma 2 prompt included) and the production quantized path
keeps the top-1 token on all 24 prompt cases of those six; the Vulkan engine matches the CPU
on tiny models of every architecture, and six models pass the live suite on
Vulkan (see above):

| Path compared | Result |
|---|---|
| Reference arithmetic (weights dequantized to f32) against `transformers` | Matches on all 13 models: top-1 token and top-10 set equal, `max\|Δlogit\| < 5e-4`, greedy output equal for 24 of 24 tokens on the chat prompt and 8 of 8 on a 700-token prompt; the Phi long prompt of about 4,200 tokens exercises LongRoPE's long factors and matches |
| Production quantized path (the CPU engine as it ships) against `transformers` | Top-1 token kept on all 48 prompt cases of the 13 first models; `max\|Δlogit\|` between 0.05 and 1.13, the cost of quantized weights |
| Qwen 3.5 0.8B Q8_0, reference arithmetic against `transformers` | `max\|Δlogit\| < 5e-5`, and the production quantized path keeps the top-1 token |
| GPU (Vulkan) against the CPU dequantized reference | Relative error 2e-6 to 2e-5 on 10 models, greedy token ids equal |

On the CPU, every Q4_0 and Q8_0 matrix multiplication runs through cuca's own
batch-invariant quantized kernel, written in Rust; other dtypes use candle. The
prefill is row-exact: the logits of a prompt row are bit-identical whether the
prompt is prefilled in one chunk, in several, or decoded one token at a time,
so reusing a cached prefix never changes the output. Measured 1,300-token
prefill speed with the engine's own kernel: Gemma 4 E2B 86.5 tokens/s, Phi-3.5
mini 76.8, Qwen 3 0.6B 249.6.

## Using another GGUF

Any file whose architecture is in the table and whose tokenizer is one of the
schemes above loads through `ModelSource::File(path)` or
`ModelSource::hugging_face(repo, revision, file)`. The CPU reads every dtype
candle can; a model the engine accepts but that was never run through the live
suite is, for now, unverified: the first thing worth checking is that its
rendered prompt and token ids equal its reference tokenizer's, which is how the
catalog models were checked.
