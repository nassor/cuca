+++
title = "Local (in-process)"
description = "Run a language model inside your own program, on the CPU or a GPU, with no server and no HTTP endpoint. Start here for a first run, then pick a model, a GPU or a configuration."
template = "section.html"
aliases = ["providers/local/"]
sort_by = "weight"

[extra]
kicker = "Reference"
+++

<dl class="page-facts">
<dt>In one line</dt>
<dd>A language model, a GGUF file or a Hugging Face safetensors checkpoint, runs inside your program, on the CPU or a GPU. There is no server and no HTTP endpoint.</dd>
<dt>You need</dt>
<dd>Rust 1.100, a C compiler and the <code>provider-local</code> feature. A GPU needs <code>provider-local-vulkan</code>, <code>provider-local-metal</code>, <code>provider-local-cuda</code> or <code>provider-local-rocm</code> instead. Disk for the model file: 2.84 GB for the default.</dd>
<dt>Read this if</dt>
<dd>You want a model that needs no external service, or you route requests through <code>ProviderEndpoint::Local</code>.</dd>
</dl>

`provider-local` runs a language model inside your own program. The model is a GGUF file, one file that holds the weights, the tokenizer and the chat template, or a Hugging Face safetensors checkpoint that CUCA converts once to a cached GGUF. See [Model files and downloads](@/local/model-sources.md). You load it from Rust, send it a request and read the reply as a stream.

The engine is written in Rust. It uses no llama.cpp and no FFI written by cuca. The [server](@/server/_index.md) runs the same engine behind an HTTP API.

<div class="dgm">
<div class="dgm-scroll">
<svg viewBox="0 0 660 200" role="img" aria-labelledby="loc-title loc-desc">
  <title id="loc-title">A local model runs inside your program</title>
  <desc id="loc-desc">Four boxes in a row, left to right: your program, the LocalModel handle, the engine thread named cuca-local-engine, and the CPU or a GPU. Requests go left to right and content blocks stream back. Under the engine thread sits the model file in the cache directory, which the engine thread loads. A dashed arrow from Hugging Face to the model file is the one-time download and the only network use.</desc>
  <defs>
    <marker id="loc-arrow" viewBox="0 0 8 8" refX="7" refY="4"
            markerWidth="6" markerHeight="6" orient="auto">
      <path d="M0 0.6 L7 4 L0 7.4 z" fill="var(--muted-foreground)"/>
    </marker>
  </defs>
  <text class="t-title" x="6" y="20">Everything runs inside your process</text>
  <rect class="blk" x="6" y="40" width="130" height="52" rx="6"/>
  <text class="t-sm t-mid" x="71" y="63">your program</text>
  <text class="t-sm t-mid" x="71" y="79">async Rust</text>
  <rect class="blk" x="176" y="40" width="130" height="52" rx="6"/>
  <text class="t-sm t-mid" x="241" y="63">LocalModel</text>
  <text class="t-sm t-mid" x="241" y="79">shared handle</text>
  <rect class="blk blk-ctl" x="346" y="40" width="130" height="52" rx="6"/>
  <text class="t-sm t-mid t-ctl" x="411" y="63">engine thread</text>
  <text class="t-sm t-mid" x="411" y="79">cuca-local-engine</text>
  <rect class="blk" x="516" y="40" width="130" height="52" rx="6"/>
  <text class="t-sm t-mid" x="581" y="63">CPU or a GPU</text>
  <text class="t-sm t-mid" x="581" y="79">runs the model</text>
  <path class="arw" d="M138 56 H174" marker-end="url(#loc-arrow)"/>
  <path class="arw arw-data" d="M174 78 H138" marker-end="url(#loc-arrow)"/>
  <path class="arw" d="M308 56 H344" marker-end="url(#loc-arrow)"/>
  <path class="arw arw-data" d="M344 78 H308" marker-end="url(#loc-arrow)"/>
  <path class="arw" d="M478 66 H514" marker-end="url(#loc-arrow)"/>
  <rect class="blk" x="346" y="128" width="130" height="44" rx="6"/>
  <text class="t-sm t-mid" x="411" y="147">model file</text>
  <text class="t-sm t-mid" x="411" y="161">in the cache dir</text>
  <path class="arw" d="M411 126 V94" marker-end="url(#loc-arrow)"/>
  <text class="t-ax" x="419" y="114">loads</text>
  <rect class="blk blk-bnd" x="176" y="128" width="130" height="44" rx="6"/>
  <text class="t-sm t-mid t-bnd" x="241" y="147">Hugging Face</text>
  <text class="t-sm t-mid" x="241" y="161">model host</text>
  <path class="arw arw-bnd" d="M308 150 H344" marker-end="url(#loc-arrow)"/>
  <text class="t-ax" x="6" y="192">dashed: the one-time download, the only network use</text>
</svg>
</div>
<p class="dgm-cap"><b>No server sits between your code and the model.</b> Your program calls a <code>LocalModel</code>, one engine thread runs the model on the CPU or a GPU, and content blocks stream back the same way. The network is used once, to download the file.</p>
</div>

New to the words on this page? Read [How local inference works](@/local/how-it-works.md) first or after the tutorial.

## Run your first local model

This tutorial runs the default model, `gemma4-e2b-q4_0`, from a new Rust crate. It ends with a reply printed in your terminal.

You need:

- Rust 1.100 or newer.
- A C compiler. See [Build requirements](@/local/build-and-tests.md) if the build fails on a native dependency.
- About 3 GB of free disk. The default model is a 2.84 GB download.
- About 2 GB of RAM for the default model: 1.5 GB of weights and 481 MB of K/V cache. The K/V cache is the memory a model keeps for the text it has already read. `LocalModel::kv_cache_bytes` reports its size.

### Step 1: add the crate

```bash,name=Runs the same on all three platforms
cargo add cuca --path /path/to/cuca --features provider-local
cargo add tokio --features rt,macros
cargo add tokio-stream
```

Result: `Cargo.toml` lists `cuca` with the `provider-local` feature.

### Step 2: build for your CPU

The CPU engine picks its fast code (SIMD kernels, which handle many numbers per instruction) when it compiles. A generic build is several times slower, so build for the machine you run on. The variable lasts for the current terminal session.

```bash,name=Linux and macOS: build for the host CPU
export RUSTFLAGS="-C target-cpu=native"
```

```powershell,name=Windows (PowerShell): build for the host CPU
$env:RUSTFLAGS = "-C target-cpu=native"
```

### Step 3: write the program

Put this in `src/main.rs`. It loads the default model, builds a client for `ProviderEndpoint::Local`, and prints the reply as it streams.

```rust,name=A first stream through the local engine
use std::io::{Write, stdout};

use cuca::types::{MessageContentBlock, ProviderEndpoint};
use cuca::{CucaClient, LocalModel, LocalModelConfig, UnifiedRequest};
use tokio_stream::StreamExt;

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Downloads the pinned default model (Gemma 4 E2B Q4_0) on first use, then
    // loads it.
    let model = LocalModel::load(LocalModelConfig::default()).await?;

    let client = CucaClient::builder()
        .with_provider(ProviderEndpoint::Local)
        .with_local_model(model)
        .build()?;

    // The request's model name is not used for routing: one client holds one model.
    let request = UnifiedRequest::new("local")
        .add_system_message("You are concise.")
        .add_user_message("Explain CUCA in one sentence.")
        .set_max_tokens(256);
    let mut stream = client.generate_stream(request).await?;
    while let Some(block) = stream.next().await {
        if let MessageContentBlock::Text(text) = block? {
            print!("{text}");
            stdout().flush()?;
        }
    }
    println!();
    Ok(())
}
```

### Step 4: run it

```bash,name=Runs the same on all three platforms
cargo run --release
```

Result: the first run downloads the default preset, a 2.84 GB model file, into the cache directory, checks its SHA-256 and loads it. Then one short answer prints, word by word as it is written. The words differ from run to run, because the reply depends on the model and the sampling seed. Later runs skip the download and only load the file.

### Step 5: try the example and another model

The repository has the same program with flags. Run it from a checkout of the repository:

```bash,name=Runs the same on all three platforms
cargo run --release --example local_model --features provider-local
```

```text,name=What the example prints (numbers differ on your machine)
<downloaded> / Some(<total>) bytes
load: <seconds> s
loaded on cpu
<the reply>
timing: <n> prompt tokens, first block after <s> s (~<r> prompt tokens/s); <n> completion tokens (<n> blocks), then <r> tokens/s after the first block
```

The download line appears on the first run only. Download progress, `load:`, `loaded on` and `timing:` go to stderr, and the reply goes to stdout.

To run another preset, put its name in a KDL file and pass the file with `--config`. A model that reasons prints its reasoning first, dimmed, after `[thinking]`. Add `--thinking off` to skip it.

```kdl,name=qwen.kdl
source "qwen3-0.6b-q8_0"
```

```bash,name=Runs the same on all three platforms
cargo run --release --example local_model --features provider-local -- --config qwen.kdl
```

Result: the example downloads the new file once, prints `loaded on cpu` and then the reply. `--prompt "Say hi"` changes the question. The other flags and every key of the file are on [Configuration](@/local/configuration.md).

## Where to go next

- Learn the words and what happens to a request: [How local inference works](@/local/how-it-works.md).
- Pick another model, or load your own GGUF file or safetensors checkpoint: [Choose a model](@/local/models.md). The presets range from 0.29 GB to 67.71 GB.
- Run on a GPU: [Run on a GPU](@/local/gpu.md).
- The model is too big for your memory: [Fit a large model](@/local/placement.md).
- Send tools, switch thinking on or off, or stop on a string: [Requests, thinking and tool calls](@/local/requests.md).
- Change a setting or look up a default: [Configuration](@/local/configuration.md).
- Match an error message to its cause: [Limits and errors](@/local/limits-and-errors.md).

## Features and endpoint

| Fact | Value |
|---|---|
| Feature flag | `provider-local` (CPU) |
| GPU feature flags | `provider-local-vulkan`<br>`provider-local-metal`<br>`provider-local-cuda`<br>`provider-local-rocm` |
| `ProviderEndpoint` variant | `Local` |
| Base URL, route, auth header | none: there is no HTTP endpoint |
| Attached with | `CucaClientBuilder::with_local_model`; `build` fails without it |
| Network use | the one-time model download from Hugging Face, nothing else |
| Models per client | one; `UnifiedRequest::model` is not used for routing |

Each GPU feature turns on `provider-local` and `provider-local-gpu`, the shared GPU engine, which picks no backend by itself. With the feature off, dispatching `ProviderEndpoint::Local` returns `CucaError::ProviderNotEnabled("provider-local")`.

There is no API key. The one credential is the optional Hugging Face token, `hf_token`, sent as a bearer token on downloads. It is anonymous when it is `None` or empty, and the pinned preset files download anonymously.

## API at a glance

- `LocalModel::load(LocalModelConfig)` is async. It finds or downloads the file, loads it and starts the engine thread, named `cuca-local-engine`. The download, the hashing and the weight parsing run on that thread, so the call never blocks your async runtime.
- `CucaClientBuilder::with_local_model(LocalModel)` attaches the model to a client. `CucaClient::local_model()` returns `Option<&LocalModel>`.
- `LocalModel` is cheap to clone. Every clone, and every client built from a clone, shares the one engine thread. The thread ends when the last clone is dropped.
- `LocalModel::generate(UnifiedRequest)` runs one request without a client. It returns a `LocalGeneration`, a stream of `Result<MessageContentBlock, CucaError>`. It keeps no copy of the blocks and runs no plugin hooks. The [server](@/server/_index.md) serves through this entry point.
- After the stream ends, `LocalGeneration::report()` returns a `GenerationReport`. It holds the exact `usage`, the prompt reuse as `prompt_cache`, the `finish_reason` and `queue_secs`, the wait from dispatch to first admission. `finish_reason` is `stop` for the end token or a stop string, and `length` when `max_tokens` or the context ran out. Dropping the `LocalGeneration` stops the request.
- `LocalModel::capabilities()` returns `LocalCapabilities { reasoning, can_disable_thinking, tools }`: whether the model writes reasoning, whether a request can switch it off, and whether it has a tool call format the engine reads. `LocalModel::sampling()` returns the resolved sampling defaults.
- `LocalModel::kv_cache_bytes()` returns the K/V cache size in bytes. `LocalModel::weights_file_bytes()` returns the size of the model file it runs from, every part of a split set added up. `LocalModel::context_length()` returns the resolved context length.
- The other accessors return counters and layout facts: `serving_stats()`, `draft_stats()`, `expert_tier_stats()`, `engram_stats()` and `layer_split()`. The last three return `None` when the model has no such part. The fields are on [Configuration](@/local/configuration.md) and [Limits and errors](@/local/limits-and-errors.md).
- `ModelSource::default_model()` and `ModelSource::preset(name)` pick a preset. `LocalModelConfig` also reads from a KDL file, with the field names as keys; see [Configuration](@/local/configuration.md).
