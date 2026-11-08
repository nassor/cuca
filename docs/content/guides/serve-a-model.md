+++
title = "Serve a model over HTTP"
description = "Build the cuca binary, serve a model with an OpenAI-compatible API, send a chat completion and a stream, and watch it in the terminal UI."
template = "page.html"
weight = 4
+++

# Serve a model over HTTP

<dl class="page-facts">
<dt>In one line</dt>
<dd>Build <code>cuca</code> with the <code>server</code> feature, run <code>cuca serve</code>, and call it with curl or any OpenAI client</dd>
<dt>You need</dt>
<dd>Rust 1.98 or newer, a checkout of this repository, a C compiler (the <code>provider-local</code> build needs one), and about 0.7 GB of disk and RAM for the model used here</dd>
<dt>Read this if</dt>
<dd>You want a local endpoint for an editor, an agent framework or a test, with no llama.cpp or vLLM install</dd>
</dl>

Goal: a server on port 8000 serving Qwen3-0.6B Q8_0, one reply from it, and a live view of it.

## Step 1: build the binary

Run this in the repository root. The CPU kernels pick their instructions at compile time, so build for the host CPU.

Linux and macOS:

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release --no-default-features --features server-tui --bin cuca
```

Windows (PowerShell):

```powershell
$env:RUSTFLAGS = "-C target-cpu=native"
cargo build --release --no-default-features --features server-tui --bin cuca
```

Result: `target/release/cuca` (`target\release\cuca.exe` on Windows). `server-tui` includes `server`; use `server` alone to leave the terminal UI out.

## Step 2: start the server

The first start downloads the preset model (0.6 GB) into the cache directory. Leave the server running.

Linux and macOS:

```bash
./target/release/cuca serve --model qwen3-0.6b-q8_0 --stats-api
```

Windows (PowerShell):

```powershell
.\target\release\cuca.exe serve --model qwen3-0.6b-q8_0 --stats-api
```

Result: log lines end with `model ready` and `every model is ready; serving`. A model already in the cache loads in under a second on the machine this page was checked on. If the bind fails with an access error, another program or a Windows port reservation holds the port; pick another with `--listen 127.0.0.1:18000` and use it below.

## Step 3: check health and the model list

From a second terminal.

Linux and macOS:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/models
```

Windows (PowerShell):

```powershell
curl.exe http://127.0.0.1:8000/health
curl.exe http://127.0.0.1:8000/v1/models
```

Result: `{"status":"ok"}`, then a list with one model, `qwen3-0.6b-q8_0`, whose `max_model_len` is 8192. While the model loads, `/health` answers 503.

## Step 4: send a chat completion

Linux and macOS:

```bash
curl http://127.0.0.1:8000/v1/chat/completions -H "content-type: application/json" -d '{"messages":[{"role":"user","content":"Say hi in three words."}],"max_tokens":60,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}'
```

Windows (PowerShell). The body goes through a file, because PowerShell 5.1 strips the double quotes of an inline argument.

```powershell
Set-Content -Path body.json -Value '{"messages":[{"role":"user","content":"Say hi in three words."}],"max_tokens":60,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}'
curl.exe http://127.0.0.1:8000/v1/chat/completions -H "content-type: application/json" -d "@body.json"
```

```text,name=Expected output, trimmed (ids and times differ)
{"id":"chatcmpl-...","object":"chat.completion","model":"qwen3-0.6b-q8_0","choices":[{"index":0,"message":{"role":"assistant","content":"Hello!"},"finish_reason":"stop"}],"usage":{"prompt_tokens":18,"completion_tokens":2,"total_tokens":20,"prompt_tokens_details":{"cached_tokens":0}}}
```

The model name is optional with one model loaded. `chat_template_kwargs.enable_thinking` turns Qwen3's reasoning off; without it the template's default applies, and a request that enables reasoning spends tokens on `reasoning_content` first.

## Step 5: stream the reply

Add `"stream":true` and `-N`, so curl prints each event as it arrives.

Linux and macOS:

```bash
curl -N http://127.0.0.1:8000/v1/chat/completions -H "content-type: application/json" -d '{"messages":[{"role":"user","content":"Count from 1 to 5."}],"max_tokens":40,"temperature":0,"stream":true,"stream_options":{"include_usage":true}}'
```

Windows (PowerShell):

```powershell
Set-Content -Path stream.json -Value '{"messages":[{"role":"user","content":"Count from 1 to 5."}],"max_tokens":40,"temperature":0,"stream":true,"stream_options":{"include_usage":true}}'
curl.exe -N http://127.0.0.1:8000/v1/chat/completions -H "content-type: application/json" -d "@stream.json"
```

Result: lines of `data: {"object":"chat.completion.chunk",...}`, one per token, a last content chunk with a `finish_reason`, a chunk with `"choices":[]` and the `usage`, and `data: [DONE]`.

## Step 6: watch it

Run the terminal UI in a third terminal, and repeat step 5 to see the load.

Linux and macOS:

```bash
./target/release/cuca tui --url http://127.0.0.1:8000
```

Windows (PowerShell):

```powershell
.\target\release\cuca.exe tui --url http://127.0.0.1:8000
```

Result: panels for CPU, memory and GPUs, the latency percentiles, the reasons for the latency, and a row for `qwen3-0.6b-q8_0`. `q` quits.

Next page: [Model server](@/server/model-server.md) lists every option and the exact wire format, and [Balancer](@/server/balancer.md) spreads traffic over several servers.
