+++
title = "Stream a first reply"
description = "Start a local llama.cpp server, run the bundled example, watch normalized text blocks arrive on stdout."
template = "page.html"
weight = 1
+++

# Stream a first reply

<dl class="page-facts">
<dt>In one line</dt>
<dd>Point <code>cuca-core</code> at <code>llama-server</code> on port 1234 and run <code>cargo run --example llamacpp_gemma</code></dd>
<dt>You need</dt>
<dd>Rust 1.98 or newer, <code>git</code>, and a GGUF build of Gemma 4 E4B for <code>llama-server</code></dd>
<dt>Read this if</dt>
<dd>You want a reply streaming out of the crate before reading anything else</dd>
</dl>

Three steps, each ending in output you can check. Every `cargo` command runs the
same on Linux, macOS and Windows (PowerShell).

## Step 1: get the crate

`cuca-core` is not published to crates.io, so the source comes from a checkout
and dependants reference it by path.

```bash,name=Runs the same on all three platforms
git clone https://github.com/nassor/cuca
cd cuca
cargo --version
```

`cargo --version` must report 1.98.0 or newer. `rust-toolchain.toml` pins
`1.98.0`, so a `rustup`-managed toolchain installs it on the first cargo
invocation in this directory.

Nothing builds yet, and that is deliberate. Ask cargo to build with no features
and it refuses:

```bash,name=Runs the same on all three platforms
cargo check --no-default-features
```

```text,name=Expected output
error: CUCA requires one provider-* feature; enable the adapter for the backend you use.
```

That message comes from a `compile_error!` in `src/lib.rs`. Every command from
here on names a provider feature.

## Step 2: start the model server

The demo talks to `llama-server` from llama.cpp over its OpenAI-compatible
route. `llama-server` listens on 8080 by default, so start it on 1234, which is
the port the example expects.

The `google/gemma-4-e4b` repository on Hugging Face publishes safetensors, not
GGUF, so `-hf` cannot load it. Point `-m` at a GGUF file for the model and give
it the alias the example asks for:

```bash,name=Runs the same on all three platforms
llama-server -m /path/to/model.gguf --port 1234 --alias google/gemma-4-e4b
```

`llama-server` stays in the foreground. Leave it running in its own terminal and
check it from a second one.

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

The `id` in that response is the model id the request will carry. If your alias
differs, note the value; step 3 shows how to pass it.

## Step 3: run the example

```bash,name=Runs the same on all three platforms
cargo run --example llamacpp_gemma --features provider-llamacpp
```

The reply prints incrementally: each `MessageContentBlock::Text` the stream
yields goes straight to stdout, flushed per block, so you see the answer being
written rather than appearing at once. The process exits when the stream ends.

If your model id or port differs from the defaults, the example reads two
environment variables. It defaults to `http://127.0.0.1:1234/v1` and
`google/gemma-4-e4b`.

Linux and macOS:

```bash
CUCA_BASE_URL=http://127.0.0.1:1234/v1 CUCA_MODEL=your-model-id \
  cargo run --example llamacpp_gemma --features provider-llamacpp
```

Windows (PowerShell), where environment assignments are separate statements and
`;` joins them:

```powershell
$env:CUCA_BASE_URL = "http://127.0.0.1:1234/v1"; $env:CUCA_MODEL = "your-model-id"; cargo run --example llamacpp_gemma --features provider-llamacpp
```

A connection error instead of text means `llama-server` is not answering at that
base URL. Re-run the `curl` check from step 2.

## What just happened

The example built a `CucaClient` with `ProviderEndpoint::LlamaCpp` and an
explicit base URL, constructed a `UnifiedRequest` with one system message and
one user message, and drained the stream `generate_stream` returned. The
llama.cpp adapter's own default base URL is `http://127.0.0.1:8080`, which is
why the example passes `with_base_url` rather than relying on the default.

No API key was configured, and none was sent: the `Authorization` header is
attached only when a key is set.

Next page: [After the first run](@/quick-start/after-the-first-run.md), which
adapts this into something closer to your own code.
