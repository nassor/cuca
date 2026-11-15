+++
title = "Support status and known limits"
description = "Which model presets and devices were actually run, what each status means, and the known limits of individual models with the practical advice for each."
template = "page.html"
aliases = ["providers/local/verification/"]
weight = 12
+++

# Support status and known limits

<dl class="page-facts">
<dt>In one line</dt>
<dd>A model or device counts as verified only where a recorded run exercised it. Everything else is compile-checked or not run.</dd>
<dt>You need</dt>
<dd>Nothing running. To reproduce a result, see <a href="@/guides/integration-tests.md">Integration tests</a>.</dd>
<dt>Read this if</dt>
<dd>You want to know whether a model or a GPU was really run, or why a model answers badly in one situation.</dd>
</dl>

## Status words

The live test suite loads one real model and sends it a fixed set of requests. Where the model's template supports them, the requests are a factual answer with exact token counts, retrieval of a code word from a prompt of about 1,300 tokens, prefix reuse, a stop string, thinking on and off, and a tool round trip (a `get_weather` call for Paris, then an answer from the result). CI compiles the suite but has no models, so every result below is a recorded run on the development machine.

- `Passes`: every scenario that applies to the model passed.
- `Known limit`: everything passed except the scenario named under [Known limits](#known-limits).
- `Matches CPU`: on a GPU only. The GPU produced the CPU's greedy tokens in a comparison, without a full live run.
- `Not run`: no recorded run.
- `Not supported`: the load fails.

## Model presets

The names are the presets of [Choose a model](@/local/models.md). The GPU column is Vulkan on an AMD RX 9070 XT under Windows.

| Preset | CPU | GPU (Vulkan) | Notes |
|---|---|---|---|
| `gemma4-e2b-q4_0` | Passes | Passes | the default model |
| `gemma4-e2b-q8_0` | Passes | Not run | |
| `gemma4-e4b-q4_0` | Passes | Not run | |
| `gemma4-e4b-q8_0` | Known limit | Not run | can skip its reasoning block |
| `gemma4-26b-a4b-q4_0` | Passes | Not supported | CPU only, resident or tiered experts; a GPU load fails |
| `gemma3-270m-q8_0` | Known limit | Matches CPU | fails on long prompts; no thinking or tools |
| `gemma3-1b-q4_0` | Passes | Passes | no thinking or tools |
| `qwen3-0.6b-q8_0` | Passes | Matches CPU | |
| `qwen3-1.7b-q8_0` | Passes | Matches CPU | |
| `r1-distill-qwen-1.5b-q4_0` | Known limit | Matches CPU | always reasons; no tool calls |
| `r1-distill-llama-8b-q4_0` | Passes | Not run | always reasons; no tool calls |
| `smollm3-3b-q4_0` | Known limit | Matches CPU | thinking and the tool answer |
| `phi4-mini-q4_0` | Passes | Not run | no thinking or tools |
| `phi4-mini-q4_k_m` | Passes | Passes | no thinking or tools |
| `granite3.3-2b-q4_0` | Known limit | Known limit | long retrieval |
| `olmo2-1b-q4_0` | Passes | Passes | no thinking or tools |
| `qwen3-30b-a3b-q4_0` | Passes | Passes | Vulkan with tiered experts |
| `olmoe-1b-7b-q4_0` | Passes | Matches CPU | Vulkan with tiered experts; no thinking or tools |
| `qwen3.5-0.8b-q8_0` | Passes | Not run | XML tool calls |
| `qwen3.6-35b-a3b-q4_0` | Passes | Passes | CPU resident or tiered; Vulkan tiered |
| `muse-glimmer-30b-q4_0` | Passes | Passes | always reasons; Vulkan with `gpu_layers` 34, 36 and 38 |
| `ministral3-3b-reasoning-q4_0` | Passes | Not run | always reasons; sample to get tool calls |
| `ministral3-3b-instruct-q4_0` | Passes | Not run | no reasoning |
| `mistral-small-4-119b-q4_0` | Passes | Passes | CPU resident or tiered; Vulkan with `experts { host; }` |

The two drafter presets are on [Speculative decoding](@/local/drafting.md). DeepSeek-V4.1 has no preset: it has run from a converted checkpoint on the CPU and on Vulkan. See [Run DeepSeek-V4.1](@/local/deepseek-v41.md).

## Devices

| Device | Status | Notes |
|---|---|---|
| CPU | verified | every run above; the default device |
| Vulkan | verified | AMD RX 9070 XT, Windows |
| ROCm | ran on Windows | AMD RX 9070 XT (gfx1201); CI also compile-checks it |
| CUDA | compile-checked only | no run on an NVIDIA GPU |
| Metal | compile-checked only | no run on Apple hardware |
| Tensor parallelism | ran on one Vulkan GPU | the same device twice, at about a third of the speed; never run over two different GPUs, and the CUDA collective never ran |

See [Run on a GPU](@/local/gpu.md) for the features to build with and the Windows ROCm setup.

## Numerical accuracy

CPU output was compared with Hugging Face `transformers` in f32 on the same prompts and matches its top token. The quantized CPU path rounds activations to Q8_0, so a greedy reply can differ from a float reference where two candidates are within about 0.1 in score.

## Known limits

Each entry comes from the model itself or from quantization rounding. The live suite prints `KNOWN-LIMIT` for them.

- Gemma 3 270M (`gemma3-270m-q8_0`) ends its reply at once on prompts of 750 and 1,110 tokens, while a 570 token prompt still works. The Hugging Face f32 model does the same, so use a larger model for long prompts.
- Gemma 4 E4B Q8_0 (`gemma4-e4b-q8_0`) can answer a simple question such as `What is 9 * 8?` at once, with no reasoning block, even with thinking on. It does reason on a harder question such as `What is 17 * 23? Think carefully.`, and the Q4_0 file opens a reasoning block for the simple one, so use `gemma4-e4b-q4_0` when you need the reasoning.
- DeepSeek-R1 distill Qwen 1.5B (`r1-distill-qwen-1.5b-q4_0`) is marginal at retrieving a fact from a prompt of about 1,100 tokens. It can reason for more than 4,096 tokens, so give it a generous `max_tokens`.
- Granite 3.3 2B (`granite3.3-2b-q4_0`) can misspell a word it copies from a prompt of about 1,300 tokens, because quantization flips a close choice. Check exact quotations from long prompts.
- SmolLM 3 (`smollm3-3b-q4_0`) can reason for thousands of tokens without concluding when thinking is on, and can call a tool again instead of answering after the tool result while the tool is still declared. Give it a generous `max_tokens` and cap your tool loop.
- Ministral 3 Reasoning (`ministral3-3b-reasoning-q4_0`) always reasons. Decoded greedily it can spend its whole `max_tokens` budget reasoning and never write its tool call: in the test, 8,192 tokens of greedy reasoning never reached the call, and Hugging Face behaves the same. Sample instead of setting `temperature` to 0. At temperature 0.7 and `top_p` 0.95 it closes its reasoning and calls the tool within 137 tokens in the three seeds tried. The live suite samples for the tool scenarios of every model that always reasons, Muse Glimmer included, and these models need a generous `max_tokens`.
- ERNIE 4.5 (0.3B, a pretrained model) has no preset. Every Q8_0 and Q4_0 GGUF that was searched for lacks `tokenizer.ggml.add_space_prefix`, so a space lands after each special token and the model answers `France` where the Hugging Face tokenizer gives `Paris`. Check that metadata key in any ERNIE 4.5 file you load.

## Converted checkpoints

A model you convert from a checkpoint gives the same live results as the preset GGUF of the same model. This was checked for Qwen 3 0.6B, Phi 4 mini and Gemma 4 E2B. Gemma 3 1B converted at `q8_0` writes a count in words where the `q4_0` preset writes digits, so convert at `q4_0` to match the preset. The conversion itself is on [Model files and downloads](@/local/model-sources.md).
