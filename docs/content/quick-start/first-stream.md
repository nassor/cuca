+++
title = "Stream a first reply"
description = "Start a local llama.cpp server, run the bundled example, and watch the model's reply appear in your terminal."
template = "page.html"
weight = 1
+++

# Stream a first reply

<dl class="page-facts">
<dt>In one line</dt>
<dd>Point <code>cuca</code> at <code>llama-server</code> on port 1234 and run <code>cargo run --example llamacpp_gemma --features provider-llamacpp</code></dd>
<dt>You need</dt>
<dd>Rust 1.100 or newer, <code>git</code>, and <code>llama-server</code> from llama.cpp with a GGUF build of Gemma 4 E4B</dd>
<dt>Read this if</dt>
<dd>You want a reply streaming out of the crate before reading anything else</dd>
</dl>

You will start a small model server on your machine, then run a CUCA example that asks it one question and prints the answer as it is written. Three steps, each ending in something you can check. About 15 minutes.

Every `cargo` command runs the same on Linux, macOS and Windows (PowerShell).

Four terms you will see:

- A **GGUF** file holds a model's weights in a single file. `llama-server` loads one.
- **llama.cpp** is an open source program that runs models. Its `llama-server` command serves a model over HTTP. Get it from [github.com/ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp).
- A **port** is the number after the colon in an address. This tutorial uses 1234.
- A **stream** is an answer that arrives in pieces while the model writes it, instead of all at once.

<div class="dgm">
<div class="dgm-scroll">
<svg viewBox="0 0 660 124" role="img" aria-labelledby="first-title first-desc">
  <title id="first-title">The two programs in this tutorial</title>
  <desc id="first-desc">Two terminals. In the second terminal, cargo run starts the llamacpp_gemma example, which sends a chat request over HTTP to port 1234. In the first terminal, llama-server holds the model and streams the reply back. The example prints the text to its own terminal as it arrives.</desc>
  <defs>
    <marker id="first-arrow" viewBox="0 0 8 8" refX="7" refY="4"
            markerWidth="6" markerHeight="6" orient="auto">
      <path d="M0 0.6 L7 4 L0 7.4 z" fill="var(--muted-foreground)"/>
    </marker>
  </defs>
  <text class="t-title" x="6" y="20">Two terminals, one HTTP connection</text>
  <rect class="blk" x="6" y="36" width="196" height="76" rx="6"/>
  <text class="t-lbl t-mid" x="104" y="58">terminal 2: cargo run</text>
  <text class="t-sm t-mid" x="104" y="76">example llamacpp_gemma</text>
  <text class="t-sm t-mid t-data" x="104" y="96">prints the text here</text>
  <rect class="blk blk-bnd" x="398" y="36" width="256" height="76" rx="6"/>
  <text class="t-lbl t-mid t-bnd" x="526" y="58">terminal 1: llama-server</text>
  <text class="t-sm t-mid" x="526" y="76">--port 1234</text>
  <text class="t-sm t-mid" x="526" y="96">model.gguf in memory</text>
  <path class="arw arw-bnd" d="M204 60 H394" marker-end="url(#first-arrow)"/>
  <text class="t-sm t-mid t-bnd" x="299" y="54">chat request</text>
  <path class="arw arw-data" d="M396 88 H206" marker-end="url(#first-arrow)"/>
  <text class="t-sm t-mid t-data" x="299" y="104">reply, as it is written</text>
</svg>
</div>
<p class="dgm-cap"><b>The example is a client.</b> It holds no model. <code>llama-server</code> holds the model, and the two talk over HTTP on port 1234.</p>
</div>

## Step 1: get the crate

The crate is named `cuca` and is not published to crates.io. You build it from a checkout.

```bash,name=Runs the same on all three platforms
git clone https://github.com/nassor/cuca
cd cuca
cargo --version
```

`cargo --version` must report 1.100.0 or newer. The repository pins `1.100.0` in `rust-toolchain.toml`, so a `rustup` toolchain installs it on the first cargo command in this directory.

The crate turns every provider off by default, so a build with no provider feature fails on purpose. Try it:

```bash,name=Runs the same on all three platforms
cargo check --no-default-features
```

```text,name=Expected output
error: CUCA requires one provider-* feature; enable the adapter for the backend you use.
```

A provider is the service that answers: OpenAI, Anthropic, a llama.cpp server and so on. A provider feature is the Cargo switch that compiles its adapter. Every command from here on names one.

## Step 2: start the model server

The example talks to `llama-server` through its OpenAI-compatible route, which is the chat HTTP API that OpenAI made common. `llama-server` listens on port 8080 by default. Start it on 1234, which is the port the example expects.

The `google/gemma-4-e4b` repository on Hugging Face holds safetensors files, not GGUF, so `-hf` cannot load it. Point `-m` at a GGUF file of the model, and give it the alias the example asks for:

```bash,name=Runs the same on all three platforms
llama-server -m /path/to/model.gguf --port 1234 --alias google/gemma-4-e4b
```

On Windows, write the path as `C:\path\to\model.gguf`.

`llama-server` stays in the foreground. Leave it running in its own terminal and check it from a second one.

Linux and macOS:

```bash
curl http://127.0.0.1:1234/v1/models
```

Windows (PowerShell), where `curl` is an alias for `Invoke-WebRequest`:

```powershell
curl.exe http://127.0.0.1:1234/v1/models
```

```json,name=Expected output (abridged)
{"object":"list","data":[{"id":"google/gemma-4-e4b","object":"model"}]}
```

The `id` is the model name a request must carry. If your alias differs, note it for step 3.

## Step 3: run the example

In the second terminal, from the `cuca` checkout:

```bash,name=Runs the same on all three platforms
cargo run --example llamacpp_gemma --features provider-llamacpp
```

The first run compiles the crate before it starts, so it waits a while. Then the model's answer prints piece by piece and the program exits when the answer ends. The wording is the model's own. This is one sample:

```text,name=Example output (your wording will differ)
CUCA is an acronym that has several different meanings depending on the context, such as a Clinical Unit Coordination Area in healthcare or a brand of beer.
```

A reasoning model thinks before it answers. The example drops those `Thinking` blocks, so expect a pause before the first word.

### If your port or model name differs

The example defaults to `http://127.0.0.1:1234/v1` and `google/gemma-4-e4b`. Change them with two flags:

```bash,name=Runs the same on all three platforms
cargo run --example llamacpp_gemma --features provider-llamacpp -- --base-url http://127.0.0.1:1234/v1 --model your-model-id
```

### If nothing answers

With no server at the base URL, the example prints this and exits without an error:

```text,name=Output when no server answers
No server answered at http://127.0.0.1:1234/v1: <error text>
Start llama-server there, or pass --base-url, then run this again.
```

## What just happened

The example is a short program. Its core:

```rust,name=The core of examples/llamacpp_gemma.rs
let client = CucaClient::builder()
    .with_provider(ProviderEndpoint::LlamaCpp)
    .with_base_url("http://127.0.0.1:1234/v1")
    .build()?;

let request = UnifiedRequest::new("google/gemma-4-e4b")
    .add_system_message("You are concise.")
    .add_user_message("Explain CUCA in one sentence.");

let mut stream = client.generate_stream(request).await?;
while let Some(chunk) = stream.next().await {
    if let Ok(MessageContentBlock::Text(text)) = chunk {
        print!("{text}");
    }
}
```

- `CucaClient` is the object you send requests through. `with_provider` picks the llama.cpp adapter.
- The adapter's own default address is `http://127.0.0.1:8080`, so the example sets `with_base_url` for port 1234.
- `UnifiedRequest` is one request type for every provider: a model name plus a system message and a user message.
- `generate_stream` returns the stream. Each item is a block of the answer. The example prints `Text` blocks and ignores the rest.
- No API key was set, and none was sent.

The same steps in a new crate of your own are in the README Quick start. [After the first run](@/quick-start/after-the-first-run.md) shows how to depend on `cuca` from your own project.

Next page: [After the first run](@/quick-start/after-the-first-run.md).
