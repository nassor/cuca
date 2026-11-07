+++
title = "Local verification and known limits"
description = "Which architectures and presets the provider-local tests exercised on which device, the live results per preset, the recorded known limits, and the numerical parity measurements against Hugging Face transformers."
template = "page.html"
weight = 9
+++

# Local verification and known limits

<dl class="page-facts">
<dt>In one line</dt>
<dd>An architecture or a preset counts as verified only where a test or a recorded run exercised it; everything else is compile-checked or unverified.</dd>
<dt>You need</dt>
<dd>Nothing running. The commands that reproduce a result are on <a href="@/providers/local/build-and-tests.md">Build requirements and tests</a>.</dd>
<dt>Read this if</dt>
<dd>You need to know whether a model or device was actually run, or why a live scenario is reported as a known limit.</dd>
</dl>

## Evidence

Three kinds of evidence back the claims on the local pages, from strongest to weakest:

- Unit tests in `src/provider/local/` run in CI on tiny random GGUF files and on committed fixtures: chat template renders, tokenizer pre-splitters, dequantization against candle, and the cross-architecture forward-pass tests. `TinySpec::all()` covers `llama`, `qwen2`, `qwen3`, `phi3`, `smollm3`, `qwen3moe`, `olmoe`, `gemma3`, `granite`, `olmo2`, `ernie4_5`, `muse-glimmer` and `mistral3`. `gemma4`, `qwen35`, `qwen35moe` and `mistral4` have their own builders. `mistral4` is checked against an f64 implementation of the transformers (expanded) latent attention and of the MoE FFN, and against the fused or split, dense-first and tiered variants of the same weights.
- The live suite `tests/provider_local.rs` runs one real model per invocation. CI compiles it and every test prints a skip line, so the live results below come from recorded local runs, not from CI.
- Oracle runs against Hugging Face `transformers` (f32, CPU) use dumps under `.omp/local-models/ref/`, which is gitignored. Those results are reproducible only with that tooling, not from a clean checkout.

## Architectures

- Verified in the live suite on the CPU: `gemma4`, `gemma3`, `qwen2` (R1 distill), `qwen3`, `qwen35`, `qwen35moe`, `phi3` (Phi 4 mini), `smollm3`, `granite`, `qwen3moe`, `olmoe`, `olmo2`, `muse-glimmer`, `mistral3`, `mistral4`. `llama` runs live through `r1-distill-llama-8b-q4_0` and is also covered by tiny-model tests and by the converter oracle on the same model's checkpoint.
- Also verified in the live suite on Vulkan (AMD RX 9070 XT, Windows): `gemma4` (E2B Q4_0), `olmo2`, `granite` (Granite 3.3 2B), `qwen3moe` (Qwen3-30B-A3B with tiered experts), `qwen35moe` (with tiered experts). `mistral3` has run on a GPU in the tiny-model tests only. The device matrix is on [Run on a GPU](@/providers/local/gpu.md#verified-and-compile-checked).
- `ernie4_5`: tiny-model tests and a recorded oracle run only. It has no preset and has not run in the live suite.
- `gemma4-assistant`: `tests/provider_local_mtp.rs` as a drafter, and tiny-model tests on the CPU and the GPU (the GPU ones only when `CUCA_LOCAL_DEVICE` names a GPU). The other drafters are on [Speculative decoding](@/providers/local/drafting.md#verified).
- Compile-checked only: every GPU backend other than Vulkan.

## Live results per preset

The suite ran each preset that has a model of its own on the CPU, in a release build for the host CPU, one model per run. The scenarios are a factual answer with exact usage, a prompt of about 1,300 tokens with a code word to retrieve, prefix reuse, a stop string, and, where the template supports them, thinking on and off and a tool round trip (a `get_weather` call for Paris, then an answer from the result).

| Preset | Reasoning | Tool calls | Result |
|---|---|---|---|
| `gemma4-e2b-q4_0`, `gemma4-e2b-q8_0`, `gemma4-e4b-q4_0` | on and off | yes | all scenarios pass |
| `gemma4-e4b-q8_0` | on and off | yes | all pass except thinking, a limit of the model |
| `gemma3-270m-q8_0` | no | no syntax | long-context retrieval is a known limit; the rest pass |
| `gemma3-1b-q4_0` | no | no syntax | all applicable scenarios pass |
| `qwen3-0.6b-q8_0`, `qwen3-1.7b-q8_0` | on and off | yes | all scenarios pass |
| `smollm3-3b-q4_0` | on and off | yes | thinking and the tool answer leg are known limits |
| `phi4-mini-q4_0`, `phi4-mini-q4_k_m` | no | no syntax | all applicable scenarios pass |
| `r1-distill-qwen-1.5b-q4_0` | always | no syntax | thinking and prefix reuse pass; long-context retrieval is a known limit |
| `r1-distill-llama-8b-q4_0` | always | no syntax | all applicable scenarios pass (the oracle prompt count has no dump) |
| `qwen3-30b-a3b-q4_0` | on and off | yes | all scenarios pass |
| `olmoe-1b-7b-q4_0` | no | no syntax | greedy text and usage, long-context retrieval and prefix reuse pass; the thinking and tool scenarios skip |
| `qwen3.5-0.8b-q8_0` | on and off | yes, XML | all scenarios pass on the CPU; not run on Vulkan |
| `qwen3.6-35b-a3b-q4_0` | on and off | yes, XML | all pass on the CPU, resident and with `tiered:ram=4g`, and on Vulkan with `tiered:device=4g,ram=12g` and `tiered:device=6g,ram=12g` |
| `olmo2-1b-q4_0` | no | no syntax | all applicable scenarios pass, on the CPU and on Vulkan |
| `ministral3-3b-instruct-q4_0` | no | yes, `[TOOL_CALLS]name[ARGS]json` | all pass on the CPU with the pinned template; the thinking scenarios skip, the template has no reasoning |
| `ministral3-3b-reasoning-q4_0` | always (`[THINK]`) | yes, `[TOOL_CALLS]name[ARGS]json` | all pass on the CPU; thinking-off skips, and the two tool scenarios decode with the model's sampling defaults and a fixed seed |
| `mistral-small-4-119b-q4_0` | on and off (`reasoning_effort`) | yes, `[TOOL_CALLS]name[ARGS]json` | all pass on the CPU, resident and with `tiered:ram=48g`; the oracle-prompt count skips (no dump) |
| `muse-glimmer-30b-q4_0` | always (`reasoning_strength`) | yes, channel protocol | all pass on the CPU with the pinned template; the two tool scenarios decode with the model's sampling defaults and a fixed seed, and thinking-off skips because the model cannot switch reasoning off |
| `granite3.3-2b-q4_0` | switched on by the request | yes | all pass except long-context retrieval, a known limit |

The Vulkan engine ran the same suite for `gemma4-e2b-q4_0`, `olmo2-1b-q4_0` and `granite3.3-2b-q4_0` with the same results as on the CPU.

## Known limits

The suite reports a known limit as `MATRIX|<scenario>|KNOWN-LIMIT` with its evidence. The entries live in `known_limit` in `tests/provider_local.rs`.

- `gemma3-270m-q8_0` answers with its end-of-turn token at once on prompts of 750 and 1,110 tokens, while a 570-token prompt still retrieves the code word. The Hugging Face f32 model does the same, so this is the model.
- `gemma4-e4b-q8_0` answers `72` at once, without a reasoning block, to `What is 9 * 8?` with thinking on. This file derives from the base `google/gemma-4-E4B-it` checkpoint, and its Q4_0 sibling from the QAT checkpoint. At the first position `7` scores 28.77 against 28.37 for `<|channel>` in Hugging Face f32 and bf16, and 28.81 against 28.34 for the file's dequantized weights. The QAT-derived Q4_0 file opens the channel (29.28 against 28.91). On the harder question `What is 17 * 23? Think carefully.` the Q8_0 file does reason.
- `r1-distill-qwen-1.5b-q4_0` is marginal at retrieving a code word from a 1,100-token prompt. Hugging Face f32 answers `PELCAN` (misspelled). The Q4_0 model either answers or reasons for more than 4,096 tokens, depending on how its greedy trajectory falls on near-ties.
- `granite3.3-2b-q4_0` answers `PEICAN` where Hugging Face f32 answers `PELICAN`, on the 1,300-token retrieval prompt. Hugging Face on the same file's dequantized weights, and the engine's exact arithmetic, also give `PEICAN`: quantization flips a near-tie at the second token.
- `smollm3-3b-q4_0` has two scenarios that are quantized-arithmetic near-ties, not properties of the model. The engine's exact arithmetic (dequantized f32, no Q8_0 activation rounding) equals the dequantized reference run in Hugging Face. The production path rounds activations to Q8_0 and flips a close logit:
  - Thinking on `What is 9 * 8?`: exact arithmetic closes `</think>` at step 1,709 (Hugging Face f32 on the original weights: 535). The production path flips a 0.12-logit tie at the 154th reasoning token, then reasons for 3,070 tokens without concluding.
  - The tool round trip's answer leg with the tool still declared: exact arithmetic gives `The` 18.966 against the tool-call token 18.793 and answers. The production path flips it (18.663 against 19.237) and calls the tool again.
  - Give reasoning models a generous `max_tokens`.
- `ernie4_5` (ERNIE 4.5 0.3B, a pretrained model, not an instruct one) runs and matches the oracle numerically, but has no preset. Every Q8_0 and Q4_0 GGUF of it that was searched for (29 of 283 files in 54 Hugging Face repositories) lacks `tokenizer.ggml.add_space_prefix`, so llama.cpp's SentencePiece default applies (a space in front of the text after a special token), and the model answers the capital question with `France` where the Hugging Face tokenizer gives `Paris`.

### Greedy decoding of a reasoning model

Ministral 3 3B Reasoning decoded greedily can reason for the whole token budget without writing its tool call. On the matrix prompt (`What's the weather in Paris right now? Use the tool.`, `get_weather` declared, 227 prompt tokens), a budget of 8,192 tokens (7,965 after the prompt) gives 31,967 characters of reasoning that never closes `[/THINK]` and never writes `[TOOL_CALLS]`. Hugging Face f32 on the original BF16 weights does the same for 2,100 greedy tokens.

- The chat template is not the cause: the embedded December 2025 template and the repository's current one render the identical prompt for this request.
- The parser is not the cause: sampled replies of the form `[THINK]...[/THINK][TOOL_CALLS]get_weather[ARGS]{"city": "Paris", ...}` arrive as a `Thinking` block and a `ToolCall` (`mistral_think_then_call` in `output/mod.rs`).
- At the model card's sampling (temperature 0.7, top-p 0.95), seeds 1 to 3 close `[/THINK]` and call the tool after 76, 137 and 66 completion tokens in the engine, and after 94, 132 and 149 new tokens in Hugging Face bf16.

The live suite therefore decodes the two tool scenarios of a model whose template forces reasoning (`Capabilities::always_thinks`) with the model's own sampling defaults (`LocalModel::sampling()`) and the fixed request seed 1, and prints `MATRIX|<scenario>|DECODING|...`. Every other scenario stays greedy. An application that wants tool calls from such a model should sample (its model card recommends temperature 0.7) rather than set `temperature` to 0.

## Converted checkpoints

The live matrix (CPU, release) on converted checkpoints gives the same categories as the preset GGUF of the same model for Qwen 3 0.6B, Phi 4 mini and Gemma 4 E2B (converted at Q4_0, with `token_embd` at Q8_0 and `per_layer_model_proj` at F16, as the published file has them). The one scenario that differs is a model effect: Gemma 3 1B converted at Q8_0 writes the count in words (`One, Two, ...`) where the Q4_0 preset writes digits, so `stop_string_cuts_the_reply` fails; converted at `q4_0` it passes like the preset.

The multi-token-prediction check in `tests/provider_local_mtp.rs` passes with a converted Gemma 4 E2B target and a converted `google/gemma-4-E2B-it-assistant`, with output identical to plain decoding. The Qwen 3 0.6B matrix passes on Vulkan. The comparison with llama.cpp's converter and the transformers oracle is on [Checkpoint conversion](@/providers/local/checkpoints.md).

## Numerical parity with Hugging Face

Method: Hugging Face `transformers` in f32 on the CPU against the engine's CPU path, on prompts of 24 chat tokens, a 700-token prompt and, for Phi, a prompt of about 4,200 tokens. Reference arithmetic means the engine with the GGUF weights dequantized to f32; the production path is the quantized CPU path.

| Path compared | Result |
|---|---|
| Reference arithmetic, Gemma 3, Gemma 4, Qwen 3, R1 distills, SmolLM 3, Phi 4 mini | top-1 token and top-10 set equal, maximum absolute logit difference below 5e-4, greedy output equal for 24 of 24 tokens on the chat prompt and 8 of 8 on the 700-token prompt; the Phi 4,200-token prompt exercises LongRoPE's long factors and matches |
| Reference arithmetic, OLMo 2, Granite 3.3, ERNIE 4.5 | maximum absolute logit difference 0.000 on the top-k, greedy output equal on every prompt |
| Reference arithmetic, OLMoE | maximum absolute logit difference 2e-5, same greedy tokens |
| Reference arithmetic, Qwen 3 MoE, a 4-layer cut of the real Qwen3-30B-A3B weights | maximum absolute logit difference 1e-5; the production Q4_0 engine's top-1 token differs from the oracle's only where two candidates are within a margin of 0.073 |
| Production quantized CPU path, the models above | top-1 token kept on every prompt case; maximum absolute logit difference at most 1.13, the cost of quantized weights |
| Muse Glimmer, tiny random model with every norm randomized | converted GGUF bit-identical to llama.cpp's `convert_hf_to_gguf.py` output (59 of 59 tensors); maximum absolute logit difference 2.6e-4 at all 48 positions, greedy output equal for 16 of 16 tokens |
| Muse Glimmer, a 4-layer cut of the real checkpoint converted to F16 (prompt of 2,600 tokens, past the 2,048 window) | maximum absolute logit difference 8.4e-5 at ten positions including 2,046 to 2,050, greedy output equal for 8 of 8 tokens |
| Ministral 3, the first 2 layers of Ministral-3-3B-Reasoning-2512 converted to F16, against `Ministral3ForCausalLM` | 300 tokens: 2.3e-5; 17,000 tokens (positions 16,382 to 16,386 cross the 16,384 original context where the query scaling starts): 2.8e-5; greedy equal. With the reference's `llama_4_scaling_beta` set to 0 the positions below 16,384 still agree and those from 16,384 differ by 0.42 to 0.52 |
| Mistral Small 4, the published Q4_0 GGUF cut to its first 2 layers (35 tensors), reference arithmetic against `Mistral4ForCausalLM` | 300-token prompt: 4.1e-5, greedy equal for 8 of 8 tokens; 8,400-token prompt (positions 2,046 to 2,050 and 8,190 to 8,194 included; the Llama 4 query scaling is active from 8,192): 7.0e-5, greedy equal |
| The same cut converted from the FP8 checkpoint (F16), against `Mistral4ForCausalLM` on that GGUF's dequantized tensors | 300 tokens: 3.3e-5; 8,400 tokens: 7.7e-5; greedy equal |
| The same converted cut against `Mistral4ForCausalLM` on the FP8 checkpoint itself (every weight `fp8 * scale_inv` in f32) | 1.2e-3 to 4.5e-3 on both prompts, greedy equal for 8 of 8 tokens |
| Synthetic FP8 checkpoint with a dense first layer and two MoE layers (`first_k_dense_replace` 1), converted to F16 | 2.1e-4 at all 28 positions, greedy equal |
| Qwen 3.5 0.8B Q8_0, reference arithmetic | maximum absolute logit difference below 5e-5; the production path keeps the top-1 token |
| Qwen 3.6 35B-A3B Q4_0 (`qwen35moe`), 4-layer cut, reference arithmetic | maximum absolute logit difference 1e-5, greedy output equal for 24 of 24 and 8 of 8 tokens |
| The same file, 40 layers, production path | top-1 token kept on all three prompts; maximum absolute logit difference 0.13 to 0.62; the reference answers `The capital of France is **Paris**.` |
| GPU (Vulkan) against the CPU dequantized reference | relative error 2e-6 to 2e-5, greedy token ids equal, on Qwen 3 0.6B and 1.7B, Gemma 3 270M and 1B, SmolLM 3, Gemma 4 E2B and the R1 distill |
| GPU (Vulkan, AMD RX 9070 XT, Windows) on `phi4-mini-q4_k_m`, against the dequantized CPU reference | top-1 token equal to the CPU's; relative logit error 1.8e-2, the Q8_0 requantization of the K-quants |

Notes on these rows:

- The 8,400-token Mistral Small 4 prompt differs from the reference's own angles by 1.1e-3 at position 8,191: transformers multiplies the position by the inverse frequency in float32, an angle error that grows with the position (5e-4 rad at 8k), where the engine computes the angles in float64. With the reference's query scaling switched off the positions below 8,192 still agree (5.4e-5) and those from 8,192 differ by 0.25 to 0.45 with another greedy output, so the check is sensitive to the scaling.
- The 1.2e-3 to 4.5e-3 gap against the FP8 checkpoint is the F16 rounding of the dequantized weights (an FP8 value times a bf16 scale has up to 12 significant bits, F16 keeps 11), not the engine: the previous row isolates it.
- For `qwen35moe`, reading the value heads in grouped (Hugging Face) order instead of tiled order breaks the 4-layer match (top-10 overlap 0 to 2 of 10), which is how the tiled order was confirmed.

The quantized matmuls round activations to Q8_0, so on the production path a greedy reply can differ from a float reference where two logits are within about 0.1 of each other. The kernels behind this are on [Local models and formats](@/providers/local/models.md#cpu-kernels).

## Measured on the CPU

Release build for the host CPU, on the verification machine (Ryzen 9 9950X3D2, 16 cores); ranges are over the recorded runs.

| Preset | Decode | Prefill | Other |
|---|---|---|---|
| `muse-glimmer-30b-q4_0` | 3.3 to 3.4 tokens per second | 1,156-token prompt in 104.5 s (11.1 tokens per second) | warm prefix: 1.4 s to the first token |
| `mistral-small-4-119b-q4_0`, resident | 10.5 to 11.7 tokens per second | 1,116-token prompt in 34.8 s and 43.6 s (26 to 32 tokens per second) | load 209 s (`MATRIX|load_seconds`); warm prefix: 0.5 s to the first token |
| `ministral3-3b-reasoning-q4_0` | 21.8 tokens per second | not recorded | load 0.9 s |

Drafter speedups, GPU speeds and tiered-expert speeds are on [Speculative decoding](@/providers/local/drafting.md), [Run on a GPU](@/providers/local/gpu.md) and [Tiered expert placement](@/providers/local/expert-tiers.md).
