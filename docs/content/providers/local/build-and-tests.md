+++
title = "Build requirements and tests"
description = "The dependencies provider-local owns, the C compiler and download needs of each feature, and the live test suites."
template = "page.html"
weight = 8
+++

# Build requirements and tests

<dl class="page-facts">
<dt>In one line</dt>
<dd><code>provider-local</code> needs a C compiler, the GPU features add more; the live suites run one real model per invocation.</dd>
<dt>You need</dt>
<dd>The <code>provider-local</code> feature and a C compiler.</dd>
<dt>Read this if</dt>
<dd>Your build fails on a native dependency, or you are running the local live tests.</dd>
</dl>

## Build requirements

`provider-local` owns these dependencies: `candle-core` and `candle-nn` 0.11, `half`, `minijinja`, `minijinja-contrib` and `unicode-general-category`. It reuses `reqwest` for the download, `sha2` for the checksum and `tokio`. It does not use `tokio-stream`: the stream is a plain channel receiver. A GPU feature adds `burn` 0.22 with `std`, `fusion` and `autotune`.

- `provider-local` needs a C compiler. `candle-core` 0.11 depends on `tokenizers[onig]`, so the dependency tree compiles the C crate `onig_sys`. cuca never calls that crate: the tokenizers, the chat templates and the engine code are Rust.
- Every GPU feature needs a C compiler as well, because CubeCL's runtime pulls an embedded store (`turso`) that builds C. On Windows a dependency's build script runs the SDK resource compiler, `rc.exe` (`windres` on the GNU target).
- `provider-local-cuda` and `provider-local-rocm` download and link a prebuilt LLVM on the first build, which needs the network once. ROCm builds on Linux only.
- The crate has no default provider, so every `cargo` command that builds the engine names its feature. candle selects its quantized SIMD kernels when it compiles: a generic build, without `RUSTFLAGS="-C target-cpu=native"`, is several times slower.

## Tests

`tests/provider_local.rs` runs one real model per invocation and needs no server. A scenario applies to a model only when `LocalModel::capabilities()` says the model can do it, and every request is greedy. A scenario that does not apply prints `MATRIX|<scenario>|SKIP|<reason>`, and a documented limit of a model prints `MATRIX|<scenario>|KNOWN-LIMIT` with its evidence. With no model configured, each test prints `SKIP: local model not available: ...` and passes. The CI `test` job compiles the suite under `ALL_NO_LLVM`, so CI exercises no model.

The scenarios:

- `greedy_text_and_usage`: a factual answer with exact usage. Prompt counts equal the prefix-reuse counts and a `max_tokens` cap is met to the token.
- `thinking_precedes_answer` and `thinking_off_has_no_reasoning`: `Thinking` blocks precede the answer when thinking is on, and are absent when it is off.
- `tool_call_round_trip` and `tool_result_answer_without_tools`: a `get_weather` call, then an answer from the result, with the tool declared on both requests and with no tool declared on the follow-up.
- `long_context_needle`: a prompt of 1,000 tokens or more with a code word to retrieve.
- `prefix_reuse_hits_and_matches_cold`: the warm prefix is read from the cache, an unrelated request shrinks it, and the warm and cold greedy outputs are identical.

`tests/provider_local_mtp.rs` is the drafter check described under [Verified](@/providers/local/drafting.md#verified). Which models and devices these suites covered is on [Local models and formats](@/providers/local/models.md#verification).
