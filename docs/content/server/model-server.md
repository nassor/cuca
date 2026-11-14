+++
title = "Model server"
description = "cuca serve: the configuration file, routes, the OpenAI chat completions wire format, errors, authentication and bounds."
template = "page.html"
weight = 1
+++

# Model server

<dl class="page-facts">
<dt>In one line</dt>
<dd><code>cuca serve</code> loads local models and answers an OpenAI-compatible HTTP API with <code>/health</code>, <code>/v1/models</code>, <code>/v1/chat/completions</code>, <code>/metrics</code> and an opt-in stats API</dd>
<dt>You need</dt>
<dd>The <code>server</code> feature, and a model: a preset name, a GGUF, a Hugging Face file or checkpoint, or a checkpoint directory</dd>
<dt>Read this if</dt>
<dd>You are pointing an OpenAI client, vLLM tooling or a load balancer at a cuca engine and need the exact fields, codes and limits</dd>
</dl>

## Command line

`cuca serve [--config <FILE>]`. The one option is `--config` (`-c`), the KDL file, default `cuca.kdl` in the working directory. Every other setting is a key of that file. Models load concurrently at startup. Until every model is ready, `/health` answers 503 and chat requests get a 503 `model_loading` error. A model that fails to load stops the server with exit status 1.

Linux and macOS:

```bash,name=Serve the models of cuca.kdl
cuca serve --config cuca.kdl
```

Windows (PowerShell):

```powershell,name=Serve the models of cuca.kdl
cuca.exe serve --config cuca.kdl
```

A missing or invalid file stops the command with exit status 1 and a message that names the file and the key. An unknown key is an error.

## Configuration file

The `serve` section holds the server's keys and one `model` node per model. The `log` section applies to every command; see [logs](@/server/stats-and-metrics.md#logs). The whole file, including the `balance` section, is `examples/configs/cuca.kdl` in the repository; the syntax is on [Configuration files](@/reference/environment.md).

```kdl,name=Three models: one on the CPU, one split between the GPU and the CPU, one mixture of experts over VRAM, RAM and the SSD
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
    // 12 GiB in RAM, the rest read from the GGUF file on demand.
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

A `model` node holds a [`LocalModelConfig`](@/providers/local/configuration.md): its keys are the field names. The common ones:

| `model` key | Default | Meaning |
|---|---|---|
| `source` | required | what to load; see below |
| `device` | `"cpu"` | `"cpu"`, `"vulkan:0"`, `"metal"`, `"cuda:0"`, `"rocm:0"` |
| `shards` | none | further devices that split one model: `shards "vulkan:1"` |
| `context_length` | the smaller of 8192 and the model's maximum | tokens of prompt plus completion per request |
| `batching` | `max_sequences=4` | `max_sequences=4 kv_tokens=16384 max_step_tokens=256 prefix_cache=#false` |
| `experts` | `"resident"` | `"resident"`, `{ tiered ram_bytes="4g" device_bytes="6g" loader_threads=4 }` or `{ host layers=4 }` |
| `gpu_layers` | all | layers on the GPU, the rest on the CPU; see [Hybrid placement](@/providers/local/placement.md) |
| `cache_dir` | the platform cache directory | where models download and convert |
| `chat_template_file` | the model's template | a Jinja file that replaces the template |
| `convert` | `quant="q8_0"` | checkpoint conversion: `quant` is `f16`, `q8_0` or `q4_0` |
| `drafter`, `draft_tokens` | none | `"in-trunk"` or a source for a drafter file; tokens drafted per round, 1 to 16 |

A device whose Cargo feature is off fails the load with a configuration error. The other keys (`sampling`, `engram`, `hf_token`, `shard_dirs`, `chat_template`) are listed on [Configuration files](@/reference/environment.md#the-model-keys).

## Model sources and names

The argument of `model` is the name clients pass as `model`. The `source` is one of:

- a preset name from [`ModelSource::PRESETS`](@/providers/local/model-sources.md), such as `source "qwen3-0.6b-q8_0"`;
- `source "repo:revision:file"`, a GGUF on Hugging Face;
- `source "repo:revision"`, a safetensors checkpoint, converted once to a cached GGUF;
- `source file="/models/x.gguf"`, a GGUF file, or `source dir="/ckpt"`, a checkpoint directory.

Without an argument the name is the preset name, else the file stem, the checkpoint's repository name or the directory name. Names are unique and at most 16 models load per server.

## Routes

| Route | Answer |
|---|---|
| `GET /health` | `200 {"status":"ok"}` once every model is ready, else `503` with an error body (`Loading model`, or the failed model's error) |
| `GET /version` | `{"version": "..."}` |
| `GET /v1/models` | the ready models: `id`, `object`, `created`, `owned_by` (`cuca`) and `max_model_len` |
| `POST /v1/chat/completions` | a chat completion, or an SSE stream with `"stream": true` |
| `GET /metrics` | Prometheus text, see [Stats API and metrics](@/server/stats-and-metrics.md) |
| `GET /cuca/v1/stats`, `GET /cuca/v1/stats/stream` | the stats API, only with `stats_api #true` |

`/v1/completions` is not served: the engine has no raw prompt entry, and every request goes through the model's chat template.

## Chat completions

With one model loaded, any `model` value or none selects it, as llama.cpp's server does. With several, `model` must match a name exactly: a missing one is a 400 and an unknown one is a 404 `model_not_found`.

### Request fields

- `messages`: roles `system`, `developer` (treated as `system`), `user`, `assistant` and `tool`. `content` is a string or an array of `text` parts, joined with a newline. An assistant message may carry `tool_calls` and `reasoning_content` (or `reasoning`). A `tool` message needs `tool_call_id`.
- Sampling: `temperature`, `top_p`, `top_k`, `seed`, `stop` (a string or an array), and `max_completion_tokens`, which wins over `max_tokens`.
- `stream` and `stream_options.include_usage`.
- `tools` with `type: "function"`. `tool_choice` `none` drops the tools, a named function keeps only that one, and `auto` and `required` keep all of them: the engine cannot force a call.
- `reasoning_effort`: `none` turns thinking off, `minimal`, `low`, `medium`, `high` and `xhigh` turn it on at that level. `chat_template_kwargs.enable_thinking` set to `false` turns it off and wins over an effort.

These are rejected with a 400 that names the field, because the engine cannot honour them: image or audio content parts, `n` other than 1, `logprobs`, a non-zero `frequency_penalty` or `presence_penalty`, a non-empty `logit_bias`, a `response_format` other than `text`, tool types other than `function`, and unknown roles. Fields that do not change the output (`user`, `store`, `metadata`) are ignored.

### Response

A completion has one choice. `message.content` is the text, `message.reasoning_content` the thinking text when there is any, and `message.tool_calls` the calls with `arguments` as JSON text. `finish_reason` is `tool_calls` after a call, else `length` when `max_tokens` or the context ran out, else `stop` (the end token or a stop string).

`usage` carries the engine's exact counts: `prompt_tokens`, `completion_tokens`, `total_tokens`, `prompt_tokens_details.cached_tokens` (prompt tokens served from the prefix cache, a multiple of 16) and `completion_tokens_details.reasoning_tokens` when the model reports it.

### Streaming

The reply is `text/event-stream`. Each event is `data: {chunk}` followed by a blank line, with chunks of object `chat.completion.chunk` and ids `chatcmpl-<random>`. The first chunk announces the role. Text arrives in `delta.content`, thinking in `delta.reasoning_content`, and a tool call in one `delta.tool_calls` entry with `index`, `id`, `type` and `function`. The last content chunk carries `finish_reason`. With `include_usage`, every chunk has `"usage": null` and a final chunk with `"choices": []` carries the usage. The stream ends with `data: [DONE]`.

A generation that fails after the stream began sends an error object as one event, then `[DONE]`. A client that disconnects stops the generation: the engine ends the request at its next step and keeps its prefix blocks cached.

### Request ids and logs

Every response has an `x-request-id` header. An incoming one is kept when it is up to 128 printable ASCII characters; otherwise the server makes one. Each request has a tracing span `http` and one `info` line when it finishes, described under [logs](@/server/stats-and-metrics.md#logs).

## Errors

Errors are `{"error":{"message","type","param","code"}}`.

| Status | `code` | When |
|---|---|---|
| 400 | `invalid_request` | a malformed body, a missing `model` with several models loaded, or an unsupported field (`param` names it) |
| 400 | `context_length_exceeded` | the prompt has as many tokens as the context length or more |
| 401 | `invalid_api_key` | a key is configured and the request lacks it or sends another |
| 404 | `model_not_found` | the `model` matches no loaded model |
| 404 | `not_found` | no such route |
| 413 | `request_too_large` | the body exceeds 32 MiB |
| 429 | `too_many_streams` | 64 stats streams are already open |
| 500 | `internal_error` | the engine failed the generation |
| 503 | `model_loading` | the model is still loading or failed to load |
| 503 | `engine_unavailable` | the engine thread has shut down |

## Authentication

With `api_key` set in the `serve` section, every route except `/health` needs `Authorization: Bearer <key>`; the comparison is constant time. A rejected request gets 401 and `WWW-Authenticate: Bearer`. `/health` stays open so probes work.

## Bounds

- The request body is limited to 32 MiB.
- A chat stream buffers 32 events between the engine and the socket. A client that reads slowly stalls only its own request, and the stall is counted in `cuca_response_stall_seconds_total`.
- The engine bounds requests: `max_sequences` run at once and 16 more queue, and a caller waits when the queue is full. The K/V pool is fixed at load. See [Memory discipline](@/concepts/memory-discipline.md).
- The latency window, the sample rings and the metric series have fixed sizes, listed under [Stats API and metrics](@/server/stats-and-metrics.md#bounds).
