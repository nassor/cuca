+++
title = "Local (in-process)"
description = "The provider-local adapter: a GGUF language model runs inside your process on the CPU or a GPU, with no server and no HTTP endpoint."
template = "section.html"
sort_by = "weight"
weight = 8

[extra]
kicker = "Reference"
+++

<dl class="page-facts">
<dt>In one line</dt>
<dd>Runs a GGUF language model of a supported architecture (see <a href="@/providers/local/models.md">Local models and formats</a>) on the CPU or a GPU inside your process, with no server and no HTTP endpoint.</dd>
<dt>You need</dt>
<dd>The <code>provider-local</code> feature for the CPU, or <code>provider-local-vulkan</code>, <code>provider-local-metal</code>, <code>provider-local-cuda</code> or <code>provider-local-rocm</code> for a GPU. A GGUF file, or disk for the one-time download (0.29 GB to 8.03 GB across the presets, 2.84 GB for the default). Memory of about the file size plus the K/V cache.</dd>
<dt>Read this if</dt>
<dd>You are routing requests through <code>ProviderEndpoint::Local</code>, or you want a model that needs no external service.</dd>
</dl>

This is the only provider without an HTTP endpoint. `LocalModel::load` resolves the GGUF file, loads the weights, the tokenizer and the chat template, and starts one engine thread. `CucaClientBuilder::with_local_model` attaches the model to a client built for `ProviderEndpoint::Local`. The engine code is pure Rust: no llama.cpp and no FFI written by cuca.

## Endpoint

| Fact | Value |
|---|---|
| Feature flag | `provider-local` (CPU, candle 0.11) |
| GPU feature flags | `provider-local-vulkan`<br>`provider-local-metal`<br>`provider-local-cuda`<br>`provider-local-rocm` |
| `ProviderEndpoint` variant | `Local` |
| Base URL, route, auth header | none: there is no HTTP endpoint |
| Attached with | `CucaClientBuilder::with_local_model`; `build` fails without it |
| Network use | the one-time model download from Hugging Face, nothing else |
| Models per client | one; `UnifiedRequest::model` is not used for routing |

Each GPU feature enables `provider-local` and `provider-local-gpu`, the shared GPU engine, which selects no backend by itself. With the feature off, dispatching `ProviderEndpoint::Local` returns `CucaError::ProviderNotEnabled("provider-local")`.

## Authentication

There is no API key and no auth header. The one credential is the optional Hugging Face token (`hf_token`, else `HF_TOKEN`), sent as a bearer token on the model download. The pinned preset files download anonymously.

## Entry points

- `LocalModel::load(LocalModelConfig)` is async. It resolves or downloads the file, loads it and starts the thread named `cuca-local-engine`. The download, the hashing and the weight parsing run on that thread, so the call never blocks the async runtime.
- `CucaClientBuilder::with_local_model(LocalModel)` attaches the model. `CucaClient::local_model()` returns `Option<&LocalModel>`.
- `LocalModel` is cheap to clone. Every clone and every client built from a clone shares the one engine thread, which ends when the last clone is dropped.
- `LocalModel::kv_cache_bytes()` returns the K/V cache size in bytes. `LocalModel::capabilities()` returns `LocalCapabilities { reasoning, can_disable_thinking, tools }`. `LocalModel::draft_stats()` returns the drafter counters.
- `ModelSource::PRESETS`, `ModelSource::preset(name)`, `ModelSource::default_model()` and `ModelSource::hugging_face(repo, revision, file)` describe model sources. `ModelSource` and `LocalDevice` also implement `FromStr`.

<div class="dgm">
<div class="dgm-scroll">
<svg viewBox="0 0 660 124" role="img" aria-labelledby="loc-title loc-desc">
  <title id="loc-title">One request through the in-process engine</title>
  <desc id="loc-desc">Five stages, left to right. CucaClient::generate_stream receives the UnifiedRequest. The chat template and the tokenizer turn it into prompt token ids. The job queue holds up to 16 waiting requests. The engine thread runs one request at a time: prefix reuse, prefill and decode. A channel of up to 64 blocks carries Text, Thinking and ToolCall blocks back to the stream. Dropping the stream stops generation, and a full queue makes generate_stream wait.</desc>
  <defs>
    <marker id="loc-arrow" viewBox="0 0 8 8" refX="7" refY="4"
            markerWidth="6" markerHeight="6" orient="auto">
      <path d="M0 0.6 L7 4 L0 7.4 z" fill="var(--muted-foreground)"/>
    </marker>
  </defs>
  <text class="t-title" x="6" y="20">One request through the local engine</text>
  <rect class="blk" x="6" y="44" width="104" height="50" rx="6"/>
  <text class="t-sm t-mid" x="58" y="67">CucaClient</text>
  <text class="t-sm t-mid" x="58" y="81">generate_stream</text>
  <rect class="blk" x="142" y="44" width="104" height="50" rx="6"/>
  <text class="t-sm t-mid" x="194" y="67">chat template</text>
  <text class="t-sm t-mid" x="194" y="81">and tokenizer</text>
  <rect class="blk" x="278" y="44" width="104" height="50" rx="6"/>
  <text class="t-sm t-mid" x="330" y="67">job queue</text>
  <text class="t-sm t-mid" x="330" y="81">16 waiting</text>
  <rect class="blk" x="414" y="44" width="104" height="50" rx="6"/>
  <text class="t-sm t-mid" x="466" y="67">engine thread</text>
  <text class="t-sm t-mid" x="466" y="81">prefill, decode</text>
  <rect class="blk blk-data" x="550" y="44" width="104" height="50" rx="6"/>
  <text class="t-sm t-mid t-data" x="602" y="67">block channel</text>
  <text class="t-sm t-mid" x="602" y="81">64 blocks</text>
  <path class="arw" d="M112 69 H138" marker-end="url(#loc-arrow)"/>
  <path class="arw" d="M248 69 H274" marker-end="url(#loc-arrow)"/>
  <path class="arw" d="M384 69 H410" marker-end="url(#loc-arrow)"/>
  <path class="arw arw-data" d="M520 69 H546" marker-end="url(#loc-arrow)"/>
  <text class="t-ax" x="6" y="114">dropping the stream stops generation; a full queue makes generate_stream wait</text>
</svg>
</div>
<p class="dgm-cap"><b>One thread runs one request at a time.</b> Requests run in arrival order, and a slow consumer paces generation through the bounded channel.</p>
</div>

## Example

The smallest stream: the default model, the `Local` variant and every other builder default.

```bash,name=Runs the same on all three platforms
cargo add cuca --path /path/to/cuca --features provider-local
cargo add tokio --features rt,macros
cargo add tokio-stream
```

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

The reply depends on the model and the sampling seed. `examples/local_model.rs` is the runnable version: it prints download progress on stderr, prints `Thinking` blocks, and prints `loaded on <device>`. It reads the variables listed under [Environment variables](@/providers/local/configuration.md#environment-variables).

candle selects its quantized SIMD kernels when it compiles, so build for the host CPU. A generic build is several times slower.

```bash,name=Linux and macOS: build for the host CPU
export RUSTFLAGS="-C target-cpu=native"
```

```powershell,name=Windows (PowerShell): build for the host CPU
$env:RUSTFLAGS = "-C target-cpu=native"
```

```bash,name=Runs the same on all three platforms
cargo run --release --example local_model --features provider-local
```

To run another preset, set `CUCA_LOCAL_MODEL` first:

```bash,name=Linux and macOS: choose a preset
export CUCA_LOCAL_MODEL="qwen3-0.6b-q8_0"
```

```powershell,name=Windows (PowerShell): choose a preset
$env:CUCA_LOCAL_MODEL = "qwen3-0.6b-q8_0"
```

PowerShell keeps `$env:` values for the rest of the session; `Remove-Item Env:CUCA_LOCAL_MODEL` clears one.
