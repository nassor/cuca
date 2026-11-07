+++
title = "Build requirements and tests"
description = "The dependencies provider-local owns, the C compiler and download needs of each feature, and the live test suites with their commands."
template = "page.html"
weight = 13
+++

# Build requirements and tests

<dl class="page-facts">
<dt>In one line</dt>
<dd><code>provider-local</code> needs a C compiler and a host CPU build; the GPU features add more; the live suites run one real model per invocation.</dd>
<dt>You need</dt>
<dd>The <code>provider-local</code> feature and a C compiler.</dd>
<dt>Read this if</dt>
<dd>Your build fails on a native dependency, or you are running the local live tests.</dd>
</dl>

## Build requirements

`provider-local` owns these dependencies: `candle-core` and `candle-nn` 0.11, `half`, `minijinja`, `minijinja-contrib` and `unicode-general-category`. It reuses `reqwest` for the download, `sha2` for the checksum and `tokio`. It does not use `tokio-stream`: the stream is a plain channel receiver. A GPU feature adds `burn` 0.22 (with `std`, `fusion`, `autotune`, `extension` and `cubecl`), `burn-cubecl` and `cubecl`.

- `provider-local` needs a C compiler. `candle-core` 0.11 depends on `tokenizers[onig]`, so the dependency tree compiles the C crate `onig_sys`. cuca never calls that crate: the tokenizers, the chat templates and the engine are Rust.
- Every GPU feature needs a C compiler as well, because CubeCL's runtime pulls an embedded store (`turso`) that builds C (`simsimd`, `aegis`). On Windows a dependency's build script runs the SDK resource compiler, `rc.exe` (`windres` on the GNU target).
- `provider-local-cuda` and `provider-local-rocm` download and link a prebuilt LLVM on the first build, which needs the network once. ROCm builds on Linux only. docs.rs and the CI all-features jobs leave both out; one CI job, `local_gpu_llvm`, checks them alone.
- The crate has no default provider, so every `cargo` command that builds the engine names its feature.
- candle selects its quantized SIMD kernels when it compiles: a generic build, without `-C target-cpu=native`, is several times slower.

```bash,name=Linux and macOS: build for the host CPU
export RUSTFLAGS="-C target-cpu=native"
```

```powershell,name=Windows (PowerShell): build for the host CPU
$env:RUSTFLAGS = "-C target-cpu=native"
```

## Live suites

Every suite runs one real model per invocation on the CPU by default and needs no server. With no model configured, each test prints `SKIP: local model not available: ...` and passes, unless `CUCA_REQUIRE_LIVE=1` turns the skip into a failure. The CI `test` job compiles the suites under the all-features set without CUDA and ROCm, so CI runs no model.

The model and device come from environment variables parsed in `tests/local_env/mod.rs`: `CUCA_LOCAL_MODEL_PATH`, `CUCA_LOCAL_MODEL` or `CUCA_LOCAL_DOWNLOAD=1` for the model, `CUCA_LOCAL_DEVICE` for the device (a GPU needs its feature). Every variable, with the files that read it, is on [Environment variables](@/reference/environment.md).

```bash,name=Linux and macOS: the live matrix on one preset
CUCA_LOCAL_MODEL=qwen3-0.6b-q8_0 cargo test --release --features provider-local --test provider_local -- --test-threads=1 --nocapture
```

```powershell,name=Windows (PowerShell): the live matrix on one preset
$env:CUCA_LOCAL_MODEL = "qwen3-0.6b-q8_0"
cargo test --release --features provider-local --test provider_local -- --test-threads=1 --nocapture
```

On a GPU, set `CUCA_LOCAL_DEVICE` (for example `vulkan`) and build with the matching feature, here `provider-local-vulkan`.

### provider_local.rs

The scenario matrix. A scenario applies to a model only when `LocalModel::capabilities()` says the model can do it, and every request is greedy, except the two tool scenarios of a model whose template forces reasoning. A scenario that does not apply prints `MATRIX|<scenario>|SKIP|<reason>`, a passing one `MATRIX|<scenario>|PASS`, and a documented limit of a model `MATRIX|<scenario>|KNOWN-LIMIT` with its evidence (see [Known limits](@/providers/local/verification.md#known-limits)).

- `greedy_text_and_usage`: a factual answer with exact usage. Prompt counts equal the prefix-reuse counts and a `max_tokens` cap is met to the token.
- `thinking_precedes_answer` and `thinking_off_has_no_reasoning`: `Thinking` blocks precede the answer when thinking is on and are absent when it is off.
- `stop_string_cuts_the_reply`: `stop = ["5"]` on a counting prompt ends the reply before the first `5`.
- `tool_call_round_trip` and `tool_result_answer_without_tools`: a `get_weather` call, then an answer from the result, with the tool declared on both requests and with no tool declared on the follow-up.
- `long_context_needle`: a prompt of 1,000 tokens or more with a code word to retrieve.
- `prefix_reuse_hits_and_matches_cold`: the warm prefix is read from the cache in whole blocks of 16, an unrelated request reuses less, and a repeat of the second turn reads at least as much and gives the identical greedy output.
- `perf_long_prompt`: runs only with `CUCA_LOCAL_PERF=1`. A prompt of about 1,100 tokens, three rounds, printing `PERF|model|round|prompt|cached|completion|first|decode tok/s` and asserting nothing; `CUCA_LOCAL_PERF_COLD=1` makes every round miss the prefix cache.

`CUCA_LOCAL_PRINT_REPLIES=1` prints every greedy reply as a `REPLY|...` line in `provider_local.rs`, `provider_local_tiers.rs` and `provider_local_mtp.rs`. The diff of two runs is the regression check for attention or K/V changes. Tool call ids are random, so mask them.

### The other suites

- `tests/provider_local_batch.rs` runs 8 concurrent clients on one model (`CUCA_LOCAL_BATCHING` sets the batching, `seqs=4` by default). On the CPU replies equal the serial ones, steps carry several rows, a second round reads the prefix store, and `BATCH|...` lines report aggregate throughput.
- `tests/provider_local_mtp.rs` is the drafter check described under [Verified](@/providers/local/drafting.md#verified).
- `tests/provider_local_tiers.rs` runs a mixture-of-experts model on a tiered placement and compares its greedy replies with a reference placement; see [Run it](@/providers/local/expert-tiers.md#run-it). It prints `TIERS|<phase>|<tok/s>|<stats>`.
- `tests/provider_local_convert.rs` and the unit test `convert::oracle` check converted checkpoints against the transformers oracle (`CUCA_LOCAL_REF_DIR`; each prints `SKIP:` without it); see [Checkpoint conversion](@/providers/local/checkpoints.md).

### Unit tests

Unit tests live next to the code under the same feature. The tokenizer and chat-template golden tests read cached model files and print `SKIP:` for a missing one. `CUCA_TOK_GGUF` and `CUCA_TOK_CASES` drive the tokenizer corpus test. The GPU unit tests in `model::gpu` run only when `CUCA_LOCAL_DEVICE` names a GPU. Which models and devices the suites covered is on [Local verification](@/providers/local/verification.md).
