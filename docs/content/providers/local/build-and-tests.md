+++
title = "Build requirements"
description = "What to install before you build provider-local and each GPU feature, why to build for the host CPU, and the one command that runs the live local tests."
template = "page.html"
weight = 13
+++

# Build requirements

<dl class="page-facts">
<dt>In one line</dt>
<dd><code>provider-local</code> needs a C compiler and a build for your own CPU. Each GPU feature adds a few more build needs, and the GPU driver is a system library the program loads when it runs.</dd>
<dt>You need</dt>
<dd>Rust 1.98, a C compiler and the <code>provider-local</code> feature.</dd>
<dt>Read this if</dt>
<dd>Your build fails on a native dependency, the engine runs slowly, or you want to run the local live tests.</dd>
</dl>

## What to install

| Feature | Needed at build time |
|---|---|
| `provider-local` | a C compiler |
| `provider-local-vulkan`<br>`provider-local-metal` | a C compiler; on Windows also the SDK resource compiler `rc.exe` (`windres` on the GNU target) |
| `provider-local-cuda`<br>`provider-local-rocm` | the same as Vulkan and Metal, and network access on the first build to download a prebuilt LLVM |

- `provider-local` needs a C compiler because its dependency tree builds the C crate `onig_sys`. The engine, the tokenizer and the chat templates are Rust.
- Every GPU feature also needs a C compiler, because the GPU runtime pulls in a store that builds C code. On Windows a dependency's build script runs `rc.exe`.
- `provider-local-cuda` and `provider-local-rocm` download and link a prebuilt LLVM on the first build. `provider-local-rocm` builds on Linux and on Windows.
- Do not build with `--all-features`. On Windows it fails to link. The feature set to use instead is on [The feature matrix](@/reference/features.md).
- The crate has no default provider, so every `cargo` command that builds the engine names its feature, for example `--features provider-local`.

## GPU drivers

A GPU build does not need the GPU or its driver. The driver is a system library that the program loads when it runs, for example the Vulkan loader. Install the normal driver for your GPU. On Windows, ROCm also needs AMD's HIP runtime files; the steps are on [Run on a GPU](@/providers/local/gpu.md).

A missing driver or GPU fails `LocalModel::load` with a `CucaError::Provider` such as `device Cuda(N): K such device(s) found (missing driver or GPU?)`. The messages are on [Limits and errors](@/providers/local/limits-and-errors.md).

## Build for your CPU

The CPU engine, built on the `candle` tensor library, picks its quantized SIMD kernels when it compiles. SIMD kernels handle many numbers per instruction. A generic build, without `-C target-cpu=native`, is several times slower. Set the flag before every `cargo build`, `cargo run` and `cargo test` of the engine.

```bash,name=Linux and macOS: build for the host CPU
export RUSTFLAGS="-C target-cpu=native"
```

```powershell,name=Windows (PowerShell): build for the host CPU
$env:RUSTFLAGS = "-C target-cpu=native"
```

The variable lasts for the current terminal session. A binary built this way runs only on CPUs that have the instructions of the machine that built it.

## Run the live tests

The live tests run one real model per invocation, on the CPU unless you choose a GPU, and need no server. They read one gitignored file, `cuca-test.kdl`, in the repository root. This file selects the model:

```kdl,name=cuca-test.kdl
local {
    model {
        source "qwen3-0.6b-q8_0"
    }
}
```

```bash,name=Runs the same on all three platforms
cargo test --release --features provider-local --test provider_local -- --test-threads=1 --nocapture
```

Result: the first run downloads the model, a 0.64 GB file. Then `cargo` reports `test result: ok` for `provider_local`. `--test-threads=1` runs one test at a time.

- With no `cuca-test.kdl`, or no model in it, each test prints `SKIP: local model not available: ...` and passes. `require_live #true` at the top of the file turns that skip into a failure.
- For a GPU, add `device "vulkan"` inside the `model` node and build with the matching feature, here `provider-local-vulkan`.
- `cuca-test.example.kdl`, next to the file, lists every key. The keys are on [Configuration files](@/reference/environment.md), and the steps for the llama.cpp suites are in [Run the live integration suites](@/guides/integration-tests.md).
