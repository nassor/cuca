+++
title = "Serve a model over HTTP"
description = "Build the cuca binary, serve a model with an OpenAI-compatible API, send a chat request and a stream, and watch it in the terminal UI."
template = "page.html"
weight = 4
+++

# Serve a model over HTTP

<dl class="page-facts">
<dt>In one line</dt>
<dd>Build <code>cuca</code> with the <code>server</code> feature, run <code>cuca run</code> on a file with a <code>serve</code> section, and call it with curl or any OpenAI client</dd>
<dt>You need</dt>
<dd>Rust 1.100 or newer, a checkout of this repository, a C compiler (the <code>provider-local</code> build needs one), and about 0.7 GB of free disk and RAM for the model used here</dd>
<dt>Read this if</dt>
<dd>You want a local endpoint for an editor, an agent framework or a test, with no llama.cpp or vLLM install</dd>
</dl>

Goal: a server on port 8000 that serves Qwen3-0.6B Q8_0, one reply from it, and a live view of it.

## Step 1: build the binary

Run this in the repository root. The CPU kernels choose their instructions when you build, so build for your own CPU.

```bash,name=Linux and macOS
RUSTFLAGS="-C target-cpu=native" cargo build --release --no-default-features --features server-tui --bin cuca
```

```powershell,name=Windows (PowerShell)
$env:RUSTFLAGS = "-C target-cpu=native"
cargo build --release --no-default-features --features server-tui --bin cuca
```

Result: the binary is `target/release/cuca` (`target\release\cuca.exe` on Windows). `server-tui` includes `server` and adds the terminal UI. Use `server` alone to leave the UI out.

## Step 2: write the configuration and start the server

Create `cuca.kdl` in the repository root. KDL is the small configuration language cuca reads. One `serve` section holds the models and the stats switch:

```kdl,name=cuca.kdl
serve {
    stats_api #true

    model {
        source "qwen3-0.6b-q8_0"
    }
}
```

The first start downloads the model (0.6 GB) into the cache folder. Start the server and leave it running:

```bash,name=Linux and macOS
./target/release/cuca run --config cuca.kdl
```

```powershell,name=Windows (PowerShell)
.\target\release\cuca.exe run --config cuca.kdl
```

Result: the log ends with `model ready` and `every model is ready; serving`.

If the start fails with an access error on the port, another program or a Windows port reservation holds it. Add `listen "127.0.0.1:18000"` to the `serve` section and use that port in the steps below.

## Step 3: check health and the model list

Run this from a second terminal.

```bash,name=Linux and macOS
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/models
```

```powershell,name=Windows (PowerShell)
curl.exe http://127.0.0.1:8000/health
curl.exe http://127.0.0.1:8000/v1/models
```

Result: `{"status":"ok"}`, then a list with one model, `qwen3-0.6b-q8_0`, whose `max_model_len` is 8192. While the model loads, `/health` answers 503.

## Step 4: send a chat request

```bash,name=Linux and macOS
curl http://127.0.0.1:8000/v1/chat/completions -H "content-type: application/json" -d '{"messages":[{"role":"user","content":"Say hi in three words."}],"max_tokens":60,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}'
```

PowerShell 5.1 strips the double quotes of an inline argument, so the body goes through a file:

```powershell,name=Windows (PowerShell)
Set-Content -Path body.json -Value '{"messages":[{"role":"user","content":"Say hi in three words."}],"max_tokens":60,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}'
curl.exe http://127.0.0.1:8000/v1/chat/completions -H "content-type: application/json" -d "@body.json"
```

```text,name=Expected output, trimmed (ids and counts differ)
{"id":"chatcmpl-...","object":"chat.completion","model":"qwen3-0.6b-q8_0","choices":[{"index":0,"message":{"role":"assistant","content":"Hello!"},"finish_reason":"stop"}],"usage":{"prompt_tokens":18,"completion_tokens":2,"total_tokens":20,"prompt_tokens_details":{"cached_tokens":0}}}
```

The `model` field is optional when one model is loaded. `enable_thinking: false` turns off Qwen3's reasoning. Without it, the model may spend tokens on `reasoning_content` before it answers.

## Step 5: stream the reply

Add `"stream":true`, and `-N` so curl prints each event as it arrives. An event is one `data:` line the server sends while the model is still writing.

```bash,name=Linux and macOS
curl -N http://127.0.0.1:8000/v1/chat/completions -H "content-type: application/json" -d '{"messages":[{"role":"user","content":"Count from 1 to 5."}],"max_tokens":40,"temperature":0,"stream":true,"stream_options":{"include_usage":true}}'
```

```powershell,name=Windows (PowerShell)
Set-Content -Path stream.json -Value '{"messages":[{"role":"user","content":"Count from 1 to 5."}],"max_tokens":40,"temperature":0,"stream":true,"stream_options":{"include_usage":true}}'
curl.exe -N http://127.0.0.1:8000/v1/chat/completions -H "content-type: application/json" -d "@stream.json"
```

```text,name=Expected output, shortened
data: {"object":"chat.completion.chunk",...}
data: {"object":"chat.completion.chunk",...}
...
data: {"object":"chat.completion.chunk","choices":[],"usage":{...}}
data: [DONE]
```

Result: one `data:` line per piece of text, a last content line with a `finish_reason`, a line with `"choices":[]` and the `usage`, and `data: [DONE]`.

## Step 6: watch it

Run the terminal UI in a third terminal. Repeat step 5 to see the load.

```bash,name=Linux and macOS
./target/release/cuca tui --url http://127.0.0.1:8000
```

```powershell,name=Windows (PowerShell)
.\target\release\cuca.exe tui --url http://127.0.0.1:8000
```

Result: panels for CPU, memory and GPUs, the latency percentiles, the reasons for the latency, and a row for `qwen3-0.6b-q8_0`. `q` quits.

Next page: [Model server](@/server/model-server.md) lists every option and the exact wire format, and [Balancer](@/server/balancer.md) spreads traffic over several model servers.
