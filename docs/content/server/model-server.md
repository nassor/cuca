+++
title = "Model server"
description = "The model server: start it with cuca run, call it, and the configuration file, routes, OpenAI chat wire format, errors, authentication and limits."
template = "page.html"
weight = 1
+++

# Model server

<dl class="page-facts">
<dt>In one line</dt>
<dd><code>cuca run</code> on a file with a <code>serve</code> section loads local models and answers an OpenAI-compatible HTTP API with <code>/health</code>, <code>/v1/models</code>, <code>/v1/chat/completions</code>, <code>/metrics</code> and an opt-in stats API</dd>
<dt>You need</dt>
<dd>The <code>cuca</code> binary built with the <code>server</code> feature (see <a href="@/server/_index.md">Server</a>), and a model: a preset name, a GGUF file, a Hugging Face GGUF file or safetensors checkpoint, or a checkpoint directory</dd>
<dt>Read this if</dt>
<dd>You want to point an OpenAI client, an editor or a load balancer at a cuca engine and need the exact fields, codes and limits</dd>
</dl>

## Quick start

A language model is a GGUF file or a Hugging Face safetensors checkpoint. A *preset* is a short name for a GGUF file that cuca knows how to download. This example serves the preset `qwen3-0.6b-q8_0`, which is about 0.6 GB.

1. Write `cuca.kdl` in the folder where you run the command:

```kdl,name=cuca.kdl
serve {
    stats_api #true

    model {
        source "qwen3-0.6b-q8_0"
    }
}
```

2. Start the server. The first start downloads the model into the cache folder. Leave this terminal open.

```bash,name=Linux and macOS
cuca run --config cuca.kdl
```

```powershell,name=Windows (PowerShell)
cuca.exe run --config cuca.kdl
```

The log ends with `every model is ready; serving`.

3. In a second terminal, check that the server is up.

```bash,name=Linux and macOS
curl http://127.0.0.1:8000/health
```

```powershell,name=Windows (PowerShell)
curl.exe http://127.0.0.1:8000/health
```

```text,name=Expected output
{"status":"ok"}
```

While the model loads, `/health` answers 503 instead.

4. Send a chat request. `enable_thinking: false` switches off the model's reasoning, so the reply is short.

```bash,name=Linux and macOS
curl http://127.0.0.1:8000/v1/chat/completions -H "content-type: application/json" -d '{"messages":[{"role":"user","content":"Say hi in three words."}],"max_tokens":60,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}'
```

PowerShell 5.1 strips the double quotes of an inline argument, so put the body in a file:

```powershell,name=Windows (PowerShell)
Set-Content -Path body.json -Value '{"messages":[{"role":"user","content":"Say hi in three words."}],"max_tokens":60,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}'
curl.exe http://127.0.0.1:8000/v1/chat/completions -H "content-type: application/json" -d "@body.json"
```

```text,name=Expected output, trimmed (ids and counts differ)
{"id":"chatcmpl-...","object":"chat.completion","model":"qwen3-0.6b-q8_0","choices":[{"index":0,"message":{"role":"assistant","content":"Hello!"},"finish_reason":"stop"}],"usage":{"prompt_tokens":18,"completion_tokens":2,"total_tokens":20,"prompt_tokens_details":{"cached_tokens":0}}}
```

For a streamed reply and the terminal UI, follow [Serve a model over HTTP](@/guides/serve-a-model.md).

<div class="dgm">
<div class="dgm-scroll">
<svg viewBox="0 0 660 150" role="img" aria-labelledby="msv-title msv-desc">
  <title id="msv-title">What a model server does with one chat request</title>
  <desc id="msv-desc">A client sends a chat request over HTTP. The server checks the API key, finds the named model, and puts the request in that model's engine queue. The engine runs it in a batch with other requests and streams tokens back as server-sent events.</desc>
  <defs>
    <marker id="msv-arrow" viewBox="0 0 8 8" refX="7" refY="4"
            markerWidth="6" markerHeight="6" orient="auto">
      <path d="M0 0.6 L7 4 L0 7.4 z" fill="var(--muted-foreground)"/>
    </marker>
  </defs>
  <text class="t-title" x="6" y="20">POST /v1/chat/completions</text>
  <rect class="blk" x="6" y="44" width="96" height="56" rx="6"/>
  <text class="t-sm t-mid" x="54" y="68">Client</text>
  <text class="t-sm t-mid" x="54" y="82">curl or SDK</text>
  <rect class="blk blk-ctl" x="136" y="44" width="110" height="56" rx="6"/>
  <text class="t-sm t-mid t-ctl" x="191" y="68">Checks</text>
  <text class="t-sm t-mid" x="191" y="82">key, model name</text>
  <rect class="blk blk-ctl" x="280" y="44" width="110" height="56" rx="6"/>
  <text class="t-sm t-mid t-ctl" x="335" y="68">Queue</text>
  <text class="t-sm t-mid" x="335" y="82">16 can wait</text>
  <rect class="blk blk-data" x="424" y="44" width="110" height="56" rx="6"/>
  <text class="t-sm t-mid t-data" x="479" y="68">Engine</text>
  <text class="t-sm t-mid" x="479" y="82">batches requests</text>
  <rect class="blk blk-data" x="568" y="44" width="86" height="56" rx="6"/>
  <text class="t-sm t-mid t-data" x="611" y="68">SSE</text>
  <text class="t-sm t-mid" x="611" y="82">tokens back</text>
  <path class="arw" d="M104 72 H132" marker-end="url(#msv-arrow)"/>
  <path class="arw arw-ctl" d="M248 72 H276" marker-end="url(#msv-arrow)"/>
  <path class="arw arw-ctl" d="M392 72 H420" marker-end="url(#msv-arrow)"/>
  <path class="arw arw-data" d="M536 72 H564" marker-end="url(#msv-arrow)"/>
  <text class="t-ax" x="136" y="124">400, 401 and 404 are answered here</text>
  <text class="t-ax" x="424" y="124">a client that leaves stops it</text>
</svg>
</div>
<p class="dgm-cap"><b>Bad requests are refused before they reach the engine.</b> SSE (server-sent events) is a plain HTTP response that sends one <code>data:</code> line per chunk as the model writes it.</p>
</div>

## Command line

`cuca run [--config <FILE>]`, on a file with a `serve` section. The one option is `--config` (`-c`), the KDL file, which defaults to `cuca.kdl` in the working directory. KDL is a small configuration language; every other setting is a key of that file. A file with a `balance` section as well also starts the balancer; see [Balancer](@/server/balancer.md#serve-and-balance-in-one-process).

- Models load at the same time when the server starts.
- Until every model is ready, `/health` answers 503 and chat requests get a 503 `model_loading` error.
- A model that fails to load stops the server with exit status 1.
- A missing or invalid file stops the command with exit status 1 and a message that names the file and the key. An unknown key is an error.

## Configuration file

The `serve` section holds the server's keys and one `model` node per model. The `log` section applies to every section; see [logs](@/server/stats-and-metrics.md#logs). The whole file, including the `balance` section, is `examples/configs/cuca.kdl` in the repository. The syntax is on [Configuration files](@/reference/environment.md).

```kdl,name=Three models: CPU, GPU plus CPU, and a mixture of experts over VRAM, RAM and the SSD
variables {
    API_KEY "change-me"
}

log format="pretty" level="info"

serve {
    listen "127.0.0.1:8000"
    api_key "${API_KEY}"
    stats_api #true
    latency_window "60s"

    model "chat" {
        source "qwen3-0.6b-q8_0"
        batching max_sequences=8
    }
    // Layer offload: the last 20 layers on the GPU, the first ones on the CPU.
    model {
        source file="/models/gemma-4-E4B-it-Q4_0.gguf"
        device "vulkan:0"
        gpu_layers 20
    }
    // A mixture of experts larger than the GPU: 4 GiB of experts in VRAM,
    // 12 GiB in RAM, the rest read from the model file on demand.
    model "moe" {
        source "qwen3.6-35b-a3b-q4_0"
        device "vulkan:0"
        experts { tiered device_bytes="4g" ram_bytes="12g"; }
    }
}
```

| `serve` key | Default | Meaning |
|---|---|---|
| `listen` | `127.0.0.1:8000` | address of the HTTP listener |
| `api_key` | none | bearer key required on every route except `/health`; an empty key is an error |
| `stats_api` | `#false` | serve `/cuca/v1/stats` and `/cuca/v1/stats/stream` |
| `latency_window` | `"60s"` | the sliding window of the latency figures |
| `model` | none; at least one | one node per model, at most 16; see below |

A `model` node holds a [`LocalModelConfig`](@/local/configuration.md). Its keys are the field names. The common ones:

- `source`: required. What to load; see below.
- `device`: default `"cpu"`. One of `"cpu"`, `"vulkan:0"`, `"metal"`, `"cuda:0"` or `"rocm:0"`.
- `shards`: none by default. Further devices that split one model: `shards "vulkan:1"`.
- `context_length`: default the smaller of 8192 and the model's maximum. Tokens of prompt plus reply per request.
- `batching`: default `max_sequences=4`. Takes `max_sequences=4 kv_tokens=16384 max_step_tokens=256 prefix_cache=#false`.
- `experts`: default `"resident"`. Takes `"resident"`, `{ tiered ram_bytes="4g" device_bytes="6g" loader_threads=4 }` or `{ host layers=4 }`.
- `gpu_layers`: default all. Layers on the GPU, the rest on the CPU; see [Fit a large model](@/local/placement.md).
- `cache_dir`: default the platform cache directory. Where models download and convert.
- `chat_template_file`: default the model's template. A Jinja file that replaces the template.
- `convert`: default `quant="q8_0"`. Checkpoint conversion: `quant` is `f16`, `q8_0` or `q4_0`.
- `drafter`, `draft_tokens`: none by default. `"in-trunk"` or a source for a drafter file; tokens drafted per round, 1 to 16.

A device whose Cargo feature is off fails the load with a configuration error. The other keys (`sampling`, `engram`, `hf_token`, `shard_dirs`, `chat_template`) are listed on [Configuration files](@/reference/environment.md).

## Model sources and names

The argument of `model` is the name clients pass as `model`. The `source` is one of:

- a preset name, such as `source "qwen3-0.6b-q8_0"` (the presets are listed on [Choose a model](@/local/models.md));
- `source "repo:revision:file"`, a GGUF on Hugging Face;
- `source "repo:revision"`, a safetensors checkpoint, converted once to a cached GGUF;
- `source file="/models/x.gguf"`, a GGUF file, or `source dir="/ckpt"`, a checkpoint directory.

Without an argument the name is the preset name, else the file stem, the checkpoint's repository name or the directory name. Names are unique, and at most 16 models load per server. [Model files and downloads](@/local/model-sources.md) covers the cache and split files.

## Routes

| Route | Answer |
|---|---|
| `GET /health` | `200 {"status":"ok"}` once every model is ready, else `503` with an error body (`Loading model`, or the failed model's error) |
| `GET /version` | `{"version": "..."}` |
| `GET /v1/models` | the ready models: `id`, `object`, `created`, `owned_by` (`cuca`) and `max_model_len` |
| `POST /v1/chat/completions` | a chat completion, or an SSE stream with `"stream": true` |
| `GET /metrics` | Prometheus text, see [Stats API and metrics](@/server/stats-and-metrics.md) |
| `GET /cuca/v1/stats`, `GET /cuca/v1/stats/stream` | the stats API, only with `stats_api #true` |

`/v1/completions` is not served. Every request goes through the model's chat template, and the engine has no raw prompt entry.

## Chat completions

With one model loaded, any `model` value, or none, selects it, as llama.cpp's server does. With several, `model` must match a name exactly: a missing one is a 400 and an unknown one is a 404 `model_not_found`.

### Request fields

- `messages`: roles `system`, `developer` (treated as `system`), `user`, `assistant` and `tool`. `content` is a string or an array of `text` parts, joined with a newline. An assistant message may carry `tool_calls` and `reasoning_content` (or `reasoning`). A `tool` message needs `tool_call_id`.
- Sampling: `temperature`, `top_p`, `top_k`, `seed`, `stop` (a string or an array), and `max_completion_tokens`, which wins over `max_tokens`.
- `stream` and `stream_options.include_usage`.
- `tools` with `type: "function"`. `tool_choice` `none` drops the tools, a named function keeps only that one, and `auto` and `required` keep all of them. The engine cannot force a call.
- `reasoning_effort`: `none` turns thinking off. `minimal`, `low`, `medium`, `high` and `xhigh` turn it on at that level. `chat_template_kwargs.enable_thinking` set to `false` turns it off and wins over an effort.

The server answers a 400 that names the field when you send something the engine cannot honour:

- image or audio content parts;
- `n` other than 1, or `logprobs`;
- a non-zero `frequency_penalty` or `presence_penalty`, or a non-empty `logit_bias`;
- a `response_format` other than `text`;
- tool types other than `function`, or unknown roles.

Fields that do not change the output (`user`, `store`, `metadata`) are ignored.

### Response

A completion has one choice.

- `message.content` is the text.
- `message.reasoning_content` is the thinking text, when there is any.
- `message.tool_calls` holds the calls, with `arguments` as JSON text.
- `finish_reason` is `tool_calls` after a call, `length` when `max_tokens` or the context ran out, else `stop` (the end token or a stop string).

`usage` carries exact counts: `prompt_tokens`, `completion_tokens`, `total_tokens`, `prompt_tokens_details.cached_tokens` (prompt tokens served from the prefix cache, a multiple of 16) and `completion_tokens_details.reasoning_tokens` when the model reports it.

### Streaming

With `"stream": true` the reply is `text/event-stream`. Each event is `data: {chunk}` followed by a blank line. Chunks have object `chat.completion.chunk` and ids `chatcmpl-<random>`.

- The first chunk announces the role.
- Text arrives in `delta.content`, thinking in `delta.reasoning_content`, and a tool call in one `delta.tool_calls` entry with `index`, `id`, `type` and `function`.
- The last content chunk carries `finish_reason`.
- With `include_usage`, every chunk has `"usage": null`, and a final chunk with `"choices": []` carries the usage.
- The stream ends with `data: [DONE]`.

A generation that fails after the stream began sends an error object as one event, then `[DONE]`. A client that disconnects stops the generation: the engine ends the request at its next step and keeps its prefix blocks cached.

### Request ids and logs

Every response has an `x-request-id` header. An incoming one is kept when it is up to 128 printable ASCII characters. Otherwise the server makes one. Each request has a tracing span `http` and one `info` line when it finishes, described under [logs](@/server/stats-and-metrics.md#logs).

## Errors

Errors are `{"error":{"message","type","param","code"}}`.

| Status | `code` | When | Fix |
|---|---|---|---|
| 400 | `invalid_request` | a malformed body, a missing `model` with several models loaded, or an unsupported field (`param` names it) | correct the request |
| 400 | `context_length_exceeded` | the prompt has as many tokens as the context length or more | shorten the prompt, or raise `context_length` |
| 401 | `invalid_api_key` | a key is configured and the request lacks it or sends another | send `Authorization: Bearer <key>` |
| 404 | `model_not_found` | the `model` matches no loaded model | check `GET /v1/models` |
| 404 | `not_found` | no such route | check the path |
| 413 | `request_too_large` | the body exceeds 32 MiB | send less |
| 429 | `too_many_streams` | 64 stats streams are already open | close one |
| 500 | `internal_error` | the engine failed the generation | read the server log |
| 503 | `model_loading` | the model is still loading or failed to load | wait for `/health` to answer 200 |
| 503 | `engine_unavailable` | the engine thread has shut down | restart the server |

## Authentication

With `api_key` set in the `serve` section, every route except `/health` needs the header `Authorization: Bearer <key>`. A rejected request gets 401 and `WWW-Authenticate: Bearer`. `/health` stays open so health probes work.

## Limits

- The request body is limited to 32 MiB.
- A chat stream buffers 32 events between the engine and the socket. A client that reads slowly stalls only its own request, and the stall is counted in `cuca_response_stall_seconds_total`.
- `max_sequences` requests run at once and 16 more wait in the queue. A caller waits when the queue is full.
- The K/V pool (the memory that holds each running request's attention state) is fixed when the model loads. See [Memory discipline](@/concepts/memory-discipline.md).
- The latency window, the sample rings and the metric series have fixed sizes, listed under [Stats API and metrics](@/server/stats-and-metrics.md#bounds).
