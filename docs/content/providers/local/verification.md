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

- Unit tests in `src/provider/local/` run in CI on tiny random GGUF files and on committed fixtures: chat template renders, tokenizer pre-splitters, dequantization against candle, and the cross-architecture forward-pass tests. `TinySpec::all()` covers `llama`, `qwen2`, `qwen3`, `phi3`, `smollm3`, `qwen3moe`, `olmoe`, `gemma3`, `granite`, `olmo2`, `ernie4_5`, `muse-glimmer` and `mistral3`. `gemma4`, `qwen35`, `qwen35moe` and `mistral4` have their own builders. `mistral4` is checked against an f64 implementation of the transformers (expanded) latent attention and of the MoE FFN, and against the fused or split, dense-first and tiered variants of the same weights. `deepseek41` has its own builder (`TinyDeepseek41`, Q8_0 or MXFP4 experts, with and without candidate blocks): chunked, token by token, batched, rolled back, prefix-reused and tiered runs equal one pass bit for bit.
- The live suite `tests/provider_local.rs` runs one real model per invocation. CI compiles it and every test prints a skip line, so the live results below come from recorded local runs, not from CI.
- Oracle runs against Hugging Face `transformers` (f32, CPU) use dumps under `.omp/local-models/ref/`, which is gitignored. Those results are reproducible only with that tooling, not from a clean checkout.

## Architectures

- Verified in the live suite on the CPU: `gemma4`, `gemma3`, `qwen2` (R1 distill), `qwen3`, `qwen35`, `qwen35moe`, `phi3` (Phi 4 mini), `smollm3`, `granite`, `qwen3moe`, `olmoe`, `olmo2`, `muse-glimmer`, `mistral3`, `mistral4`. `llama` runs live through `r1-distill-llama-8b-q4_0` and is also covered by tiny-model tests and by the converter oracle on the same model's checkpoint.
- Also verified in the live suite on Vulkan (AMD RX 9070 XT, Windows): `gemma4` (E2B Q4_0), `gemma3` (1B Q4_0), `phi3` (Phi 4 mini Q4_K_M), `olmo2`, `granite` (Granite 3.3 2B), `qwen3moe` and `qwen35moe` (with tiered experts), `muse-glimmer` (with `gpu_layers` of 34, 36 and 38 of 52) and `mistral4` (with host experts). `mistral3` has run on a GPU in the tiny-model tests only. The device matrix, including tensor parallelism, is on [Run on a GPU](@/providers/local/gpu.md#verified-and-compile-checked).
- `ernie4_5`: tiny-model tests and a recorded oracle run only. It has no preset and has not run in the live suite.
- `deepseek41`: the engine is verified on tiny synthetic models and on the tiny release-format oracle, on the CPU and (against the CPU engine) on Vulkan, and the real DeepSeek-V4.1-Flash-1017 has run in the engine on the CPU and on Vulkan, from an HDD and from an SSD; see [DeepSeek-V4.1 against the torch reference](#deepseek-v4-1-against-the-torch-reference), [DeepSeek-V4.1 real checkpoint on the CPU](#deepseek-v4-1-real-checkpoint-on-the-cpu), [DeepSeek-V4.1 on the GPU](#deepseek-v4-1-on-the-gpu), [DeepSeek-V4.1 real checkpoint on the GPU](#deepseek-v4-1-real-checkpoint-on-the-gpu) and [DeepSeek-V4.1 real checkpoint on an SSD](#deepseek-v4-1-real-checkpoint-on-an-ssd). The converter and the real checkpoint are covered under [DeepSeek-V4.1 conversion and the real checkpoint](#deepseek-v4-1-conversion-and-the-real-checkpoint).
- The DSpark drafter of `deepseek41` (a `dflash` file): `model/cpu/dspark_tests.rs` compares the block's logits and confidence with the torch oracle and checks that chunked and replayed observation equal one pass, on the CPU with the tiny oracle checkpoint. With the real target it ran on the CPU from an HDD and from an SSD, see the real checkpoint sections.
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
| `muse-glimmer-30b-q4_0` | always (`reasoning_strength`) | yes, channel protocol | all pass on the CPU with the pinned template, and on Vulkan with `gpu_layers` 34, 36 and 38; the two tool scenarios decode with the model's sampling defaults and a fixed seed, and thinking-off skips because the model cannot switch reasoning off |
| `granite3.3-2b-q4_0` | switched on by the request | yes | all pass except long-context retrieval, a known limit |

The Vulkan engine ran the same suite for `gemma4-e2b-q4_0`, `gemma3-1b-q4_0`, `phi4-mini-q4_k_m`, `olmo2-1b-q4_0` and `granite3.3-2b-q4_0`, with the same results as on the CPU, and for `mistral-small-4-119b-q4_0` with host experts.

### DeepSeek-V4.1 against the torch reference

With `CUCA_LOCAL_DS41_REF` set to the directory `.omp/local-models/ref/ds41/make_ref.py` writes, the tests in `src/provider/local/model/cpu/hyper_tests.rs` convert its tiny release-format checkpoint at `f16` (every tensor, the MXFP4 experts included, exact) and compare the CPU engine with the reference's logits. Measured on the 40-token prompt: without the simulated quantization the largest logit difference is 6e-6, with it 3e-6 and the same greedy token at every position, and the Engram hash rows and the captured layer inputs match. Without the variable these tests print `SKIP:`.

### DeepSeek-V4.1 on the GPU

With `CUCA_LOCAL_DEVICE=vulkan`, `gpu_matches_cpu_on_every_deepseek41_variant` compares the engine on an RX 9070 XT with the CPU engine on the tiny model (Q8_0 experts, MXFP4 experts, no candidate blocks, the simulated quantization off on both sides) at 1e-3 absolute plus 1e-2 relative through a whole prefill, chunked prefill, decode and a rollback into the last call. `gpu_hyper_batched_and_prefix_reuse_match_solo` checks batched steps and a 64-token prefix hit against single sequences, `gpu_offload_of_a_hyper_model_matches_single_device_at_every_legal_cut` every legal `gpu_layers` cut, `gpu_tiered_experts_match_gpu_resident` and `gpu_host_tiered_experts_match_gpu_host_experts` the expert placements. Each kernel module tests against a host reference at tiny shapes and at the real model's: the indexer picks equal the host selection from the same scores at 130,000 visible rows, the attention over 64 heads of 512 matches within 1e-4, and MXFP4 expert slots match `dequant_row`. Single kernel launches on that GPU (host overhead included, warm): sparse attention for one row 0.12 to 0.14 ms, `hc_coefficients` 65 to 90 microseconds, the indexer for one row over 20,000 rows 1.5 ms and over 130,000 rows 2.7 ms, one MoE layer's MXFP4 experts (up, down and sum, queued back to back) 0.4 ms for one row (six experts), 1.2 ms for four rows and 8.8 ms for 33 rows, and 1.5 ms for one row when each layer reads back.

### DeepSeek-V4.1 real checkpoint on the CPU

The converted real `deepseek-ai/DeepSeek-V4.1-Flash-1017` ran in the CPU engine from the default-precision set (dense Q8_0, experts MXFP4, Engram table Q8_0), in a release build with `-C target-cpu=native`, with `tiered:ram=90g,threads=16` experts and an Engram cache of `ram=4g,threads=16`. The set sits on an HDD-class volume (D:) from which the experts stream at about 0.4 to 0.5 GB/s. Loading takes 40 to 75 s.

- Logit check against the torch streaming oracle (`model/cpu/real_tests.rs`, gated by `CUCA_LOCAL_DS41_REAL` and `CUCA_CACHE_DIR`, expert RAM from `CUCA_LOCAL_DS41_REAL_RAM` in GiB, default 90): 16 prompt ids in one pass. With the simulated quantization off, top-1 is equal at 15 of 16 positions, the smallest top-10 overlap is 9, the largest absolute logit difference at the last position is 2.003, KL(oracle, engine) there is 8.88e-8 and the top token is 51119, `Paris`. With it on, top-1 is equal at 14 of 16, the smallest overlap is 8, the largest difference 1.378 and the KL 1.44e-8. The test asserts a KL below 5e-3 off and below 5e-2 on. The pass takes about 165 s cold (about 32.7 GB of expert reads) and 13 s warm.
- `examples/local_model.rs`, greedy: thinking off, `What is the capital of France?` answers `Paris`. Thinking on, `What is 9 * 8?` reasons briefly and answers `72`: 42 prompt tokens, first token after 364.8 s, 17 completion tokens at 0.196 tokens/s, 7,592 expert lookups with 4,158 disk reads (78.2 GB), 2,816 Engram lookups with 64 hits, peak resident memory 86.9 GB. Prefill runs at about 0.1 to 0.2 prompt tokens/s and decode at about 0.18 to 0.2 tokens/s. Both are bound by reading the experts, about 5 s per token cold, because the RAM tier starts cold in every process.
- `tests/provider_local.rs` with `CUCA_LOCAL_MODEL=deepseek-v41-flash-1017` and `--test-threads=1`: eight scenarios pass with no known limit: `greedy_text_and_usage`, `thinking_precedes_answer`, `thinking_off_has_no_reasoning`, `stop_string_cuts_the_reply`, `tool_call_round_trip`, `tool_result_answer_without_tools`, `prefix_reuse_hits_and_matches_cold` and `long_context_needle`. `perf_long_prompt`, a seeded-sampling scenario and the rest of the multi-hour matrix were not run.
- DSpark drafter (`CUCA_LOCAL_DRAFTER` naming the same checkpoint, CPU): on the prompt `Explain in two sentences what a mixture of experts model is.` with thinking off, temperature 0 and 48 tokens, the speculative output equals plain greedy decoding. 5 of 15 drafted tokens were accepted (33 %) in 7 rounds. There is no speedup: plain decoding ran at 0.185 tokens/s and the drafter run at 0.172, because both are bound by the disk and verifying extra rows reads more experts (87.3 GB against 83.8 GB).

### DeepSeek-V4.1 real checkpoint on the GPU

The converted real checkpoint ran on the RX 9070 XT (16 GB, Vulkan) with the dense weights and attention on the device and the experts host-tiered (`tiered:ram=85g`, 85 GiB of expert RAM, default Engram cache).

- Oracle check, `real_model_on_the_gpu_matches_the_streaming_oracle_with_quantization_{off,on}` (the CPU check's prompt of 16 tokens, every position, one pass): with the simulated quantization off, top-1 equal at 15 of 16 positions (the miss is a 0.001 tie), minimum top-10 overlap 8, KL at the last position 1.9e-8 against 8.9e-8 on the CPU, `Paris` on top. With it on, KL 5.8e-8 against 1.4e-8 on the CPU. With the quantizers on, rounding decides which grid point an activation takes, so the engines differ at positions where the oracle's own top two are within 0.7 logits: the GPU flips positions 2, 12 and 14 (the CPU flips 2 and 14), and the GPU run is gated on the last position's overlap and on flips only where the oracle's lead is under 1.5. The KL between the GPU and the CPU at intermediate positions is up to 0.08 at positions 10 and 12, against 0.06 and 0.05 for the CPU's own ON against OFF difference there; at position 14 it is 0.41, where the GPU's own ON against OFF difference is 0.35 and the CPU's 0.08.
- Layer offload: with the last 20 layers on the GPU (`CUCA_LOCAL_DS41_REAL_GPU_LAYERS=20`, a 3 GiB device tier) the same check gives KL 7.2e-8 off and 4.1e-8 on, with the same top-1 and overlap figures.
- `examples/local_model.rs` with `CUCA_LOCAL_DEVICE=vulkan`, greedy: thinking off, `What is the capital of France?` answers `Paris` (16 prompt tokens, first token after 75 s, 1 completion token); thinking on, `What is 9 * 8?` reasons `We need answer simple. 9*8=72. Need concise.` and answers `72` (42 prompt tokens, first token after 256 s, 17 completion tokens at 0.19 tokens/s). Both runs are bound by reading the experts from the HDD-class volume: 78 GB read in the second. Loading takes 7 s. With 20 layers on the GPU the replies are the same, first token after 81 s and 282 s, 0.17 tokens/s.
- Memory: the process holds 10.0 GB of device memory after the load and peaks at 12.0 GB (the all-position logits check peaks at 13.0 GB); RSS follows the RAM tier, 73 GB at the end of the second run. With 20 layers on the GPU the peak is 10.5 GB of device memory and 78 GB RSS. Measured with the per-process `GPU Process Memory` performance counters.
- Speed with the experts in the RAM tier (`real_model_on_the_gpu_speed`, `CUCA_LOCAL_DS41_REAL_SPEED=1`): the 16-token prompt takes 2.9 s warm (5.5 tokens/s) against 3.3 s on the CPU engine, and one decode step 0.41 s (2.4 tokens/s) against 0.48 s (2.1 tokens/s). The CPU computes the experts in both, so the GPU only takes the attention, the router and the shared experts.

### DeepSeek-V4.1 real checkpoint on an SSD

The same default-precision set (514.8 GB: the backbone parts, the 209 GB Engram part and the DSpark file) ran from a 4 TB NVMe SSD (`C:`, Crucial CT4000P310) instead of the HDD (`D:`, WD102KFBX). `robocopy /J` copied it in 35 min 38 s (240.7 MB/s, bound by the HDD), the sizes match and 32 sampled 16 MiB ranges of every file hash equal. Machine: Ryzen 9 9950X3D2, 125 GiB RAM, RX 9070 XT, release build with `-C target-cpu=native` and `provider-local-vulkan`, `CUCA_LOCAL_MODEL_PATH` naming part 1 on `C:`. Every number below is one process started fresh, with greedy decoding, `ram=4g,threads=16` for the Engram cache unless noted, and the RAM tier cold: the experts are read unbuffered (`FILE_FLAG_NO_BUFFERING`), so no earlier run leaves them in the OS cache, but the Engram table is read buffered and Windows cannot drop its cache here, so Engram pages from earlier runs may have been cached. Peak RSS is `PeakWorkingSet`, peak device memory the per-process `GPU Process Memory` counter sampled about once a second, disk throughput the per-second delta of the system disk counters.

```bash,name=Linux and macOS: copy the set to the SSD and run the CPU request A
rsync -a /mnt/hdd/cuca-cache/converted/ /mnt/ssd/cuca-cache/converted/
export CUCA_LOCAL_MODEL_PATH="/mnt/ssd/cuca-cache/converted/deepseek-ai--DeepSeek-V4.1-Flash-1017/1bae0e1088993b9b82de08f59de5be5a34e42170/model-q8_0-00001-of-00003.gguf"
export CUCA_LOCAL_EXPERTS="tiered:ram=90g,threads=32"
export CUCA_LOCAL_ENGRAM="ram=4g,threads=16"
export CUCA_LOCAL_PROMPT="What is 9 * 8?"
export CUCA_LOCAL_TEMPERATURE="0"
RUSTFLAGS="-C target-cpu=native" cargo run --release --example local_model --features provider-local-vulkan
```

```powershell,name=Windows (PowerShell): copy the set to the SSD and run the CPU request A
robocopy D:\cuca-cache\converted C:\cuca-cache\converted *.gguf /E /J /NP
$env:CUCA_LOCAL_MODEL_PATH = "C:\cuca-cache\converted\deepseek-ai--DeepSeek-V4.1-Flash-1017\1bae0e1088993b9b82de08f59de5be5a34e42170\model-q8_0-00001-of-00003.gguf"
$env:CUCA_LOCAL_EXPERTS = "tiered:ram=90g,threads=32"
$env:CUCA_LOCAL_ENGRAM = "ram=4g,threads=16"
$env:CUCA_LOCAL_PROMPT = "What is 9 * 8?"
$env:CUCA_LOCAL_TEMPERATURE = "0"
$env:RUSTFLAGS = "-C target-cpu=native"
cargo run --release --example local_model --features provider-local-vulkan
```

- Drive ceiling (a `ReadFile` harness with `FILE_FLAG_NO_BUFFERING`, random aligned offsets in part 2): 3.47 GB/s for 6 MiB reads (one expert matrix) with 8, 16, 32 or 64 threads, 2.4 GB/s for 2 MiB, 4.07 GB/s for 18 MiB, 4.3 to 4.5 GB/s for 64 MiB and for one 256 MiB stream. The engine's expert reads peak at up to 3.35 GB/s in the per-second samples, and a cold 16-token pass reads 32.7 GB in 11.7 s (2.8 GB/s), so cold reads run near the drive's ceiling for reads of this size.
- Request A (`What is 9 * 8?`, thinking on, 42 prompt tokens, 17 completion tokens): 78.1 to 78.2 GB of expert reads in 4,156 to 4,162 reads for 7,583 to 7,592 lookups (45 % RAM hits on the CPU) and 2,816 Engram lookups (64 hits, 748,544 bytes read), on every placement.
- Request B (`Explain in detail how a mixture of experts language model routes tokens to experts.`, thinking off, 24 prompt tokens, 200 completion tokens): 50,435 lookups, 84 % RAM hits on the CPU, 154.6 GB read. Decode speed is after the first token; the CPU decode range is over three runs.
- Request C (a 9,511 character prompt of 1,752 tokens that ends with `In one word, what is the capital of France?`, thinking off): the reply is `Paris` on the CPU and on Vulkan.

| Placement | Load | Request A: first token, decode | Request B: first token, decode | Request C: first token, prefill | Peak RSS, device memory |
|---|---|---|---|---|---|
| CPU, `tiered:ram=90g,threads=32` | 6.0 s | 21.3 s, 1.46 tokens/s | 15.7 to 17.6 s, 1.30 to 1.83 tokens/s | 184.8 s, 9.5 tokens/s | 98.9 GiB |
| Vulkan, host-tiered `tiered:ram=90g,threads=32` (request A: `ram=85g`) | 8.5 to 10.7 s | 31.6 s, 1.58 tokens/s | 26.7 s, 1.96 tokens/s | 148.1 s, 11.8 tokens/s | 90.4 GiB, 12.5 GiB |
| Vulkan, `gpu_layers=20`, `tiered:device=3g,ram=85g,threads=32` | 7.9 to 8.1 s | 42.6 s, 0.68 tokens/s | 35.4 s, 0.94 tokens/s | not run | 90.6 GiB, 10.2 GiB |
| Vulkan, `gpu_layers=20`, `tiered:device=6g,ram=85g,threads=32` | 7.9 s | not run | 31.5 s, 1.08 tokens/s | not run | 90.6 GiB, 11.8 GiB |

- `device=8g` with `gpu_layers=20` fails with `out of device memory` during the first pass. With 20 layers on the GPU the device tier served 22 % of the lookups at 3 GiB and 29 % at 6 GiB (request B), and decode (0.94 and 1.08 tokens/s) is slower than with host-tiered experts on all 40 layers (1.96 tokens/s). The cause was not profiled.
- Loader threads (CPU, request A): 8 threads give a first token after 24.3 s and 1.35 tokens/s, 16 give 23.8 s and 1.44, 32 give 21.3 s and 1.46, 64 give 21.1 s and 1.56. The raw benchmark reaches the 6 MiB ceiling with 8 threads, and the differences above are within the spread of repeated runs (the three request B runs on the CPU span 1.30 to 1.83 tokens/s). The Engram cache at `ram=1g,threads=4`, `ram=4g,threads=16` and `ram=16g,threads=64` gives first tokens of 24.9 s, 21.3 s and 21.8 s: the table costs 272 bytes per row and about 48 lookups per generated or prompt token, so it does not move the result.
- Step size (CPU, request C): the default 4,096 rows per step (the prompt is one step) takes 184.8 s with 6,306 lookups and 114.8 GB read, `step=512` 194.5 s with 15,546 lookups and 125.0 GB read.
- Bound: the prefill of request C reads 114.8 GB at an average 0.6 GB/s while the process keeps about 11 CPU cores busy (sampled every 5 s), so a long prefill is bound by the CPU's expert matmuls, not by the drive. A decode step repeated at one position with every expert resident takes 0.38 s on the CPU and 0.32 s on Vulkan (below); request B decodes at 0.55 to 0.77 s per token on the CPU and 0.51 s on Vulkan, with 16 % of its lookups reading the disk.
- Oracle check on the SSD set, same 16-token prompt and gates as above: CPU, quantizers off, top-1 equal at 15 of 16 positions, minimum top-10 overlap 9, KL 8.88e-8 at the last position with `Paris` on top, the pass 12 s cold (32.7 GB read) after a 7 s load; on, 14 of 16, overlap 8, KL 1.44e-8, 11 s. Vulkan, host-tiered: off 15 of 16, overlap 8, KL 1.91e-8; on 13 of 16, overlap 6, KL 5.79e-8. Vulkan with `gpu_layers=20`: KL 7.18e-8 off and 4.08e-8 on. The first Vulkan pass of a process is slower than later ones: the quantizers-off test ran first and its pass took 22 s, the quantizers-on pass 10 s.
- Speed with the experts in the RAM tier (`CUCA_LOCAL_DS41_REAL_SPEED=1`, the 16-token prompt): CPU 11.7 s cold, 2.53 s warm (6.3 tokens/s), one decode step 0.377 s (2.65 tokens/s); Vulkan 12.1 to 14.4 s cold, 2.36 to 2.91 s warm (5.5 to 6.8 tokens/s), one decode step 0.32 to 0.41 s (2.4 to 3.2 tokens/s) over two runs.
- Replies: `Paris` on the CPU and with host-tiered Vulkan, `72` after the same reasoning on all three placements. With `gpu_layers=20` the thinking-off question answers `Paris.` (two completion tokens, the same in three runs), which differs from the HDD run above; the cause was not investigated.
- DSpark drafter (CPU, `Explain in two sentences what a mixture of experts model is.`, thinking off, 48 tokens): output equal to plain decoding, 23 of 37 drafted tokens accepted (62 %) in 22 rounds, 1.53 tokens/s against 1.58 plain, 89.0 GB read against 83.8 GB. There is no speedup.
- `tests/provider_local.rs` with `CUCA_LOCAL_MODEL=deepseek-v41-flash-1017`, `CUCA_LOCAL_EXPERTS=tiered:ram=90g,threads=32` and `--test-threads=1`: the eight scenarios and the checkpoint name test pass in 634 s for the whole suite; the oracle prompt count skips (no dump) and `perf_long_prompt` records nothing without `CUCA_LOCAL_PERF`.

### DeepSeek-V4.1 conversion and the real checkpoint

- Converter, tiny oracle: with `CUCA_LOCAL_DS41_REF` set, `convert/oracle_ds41.rs` converts the oracle's release-format checkpoint at `f16` and finds exactly its float tensors and metadata, for the main file and the drafter file, and checks that the default output keeps native MXFP4 experts and that the Engram table follows `engram_quant`. Without the variable the tests print `SKIP:`.
- Converter, real checkpoint: a release build converted the 510 GB `deepseek-ai/DeepSeek-V4.1-Flash-1017` checkpoint at the default precision into the three-part split set and the DSpark file. The header of the output matches the headers `ggml-org` publishes: 62 of 62 metadata keys identical, 1003 of 1043 tensor names with identical shapes, the other 40 being the `exp_probs_b_vl` tensors the converter skips, and only the documented storage types differing. Peak resident memory was about 220 MB. The run took about 2 h 44 min for the 507 GB set on an HDD-class volume at about 52 MB/s.
- Real checkpoint, torch reference: a pure-torch CPU oracle (`.omp/local-models/ref/deepseek-v41-real/`) ran the real model on the prompt `What is the capital of France? Answer in one word.`, and its top-1 token is `Paris`. The engine results against it are in the two real checkpoint sections above.

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
