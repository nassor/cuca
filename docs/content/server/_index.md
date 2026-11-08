+++
title = "Server"
description = "The cuca binary: an OpenAI-compatible model server over the in-process engine, a load balancer for several servers, a terminal UI, and the stats API and metrics they share."
template = "section.html"
sort_by = "weight"
weight = 9

[extra]
kicker = "Reference"
+++

<dl class="page-facts">
<dt>In one line</dt>
<dd><code>cuca serve</code> serves local models over an OpenAI-compatible HTTP API, <code>cuca balance</code> spreads requests over several servers, and <code>cuca tui</code> shows either one live.</dd>
<dt>You need</dt>
<dd>The <code>server</code> feature for <code>serve</code> and <code>balance</code>, <code>server-tui</code> for <code>tui</code>, and a GPU feature such as <code>provider-local-vulkan</code> to serve from a GPU.</dd>
<dt>Read this if</dt>
<dd>You want vLLM or llama.cpp style endpoints backed by the cuca engine, or the latency, resource and metric names a dashboard can read.</dd>
</dl>

The server is an application layer over [`provider-local`](@/providers/local/_index.md). It loads one or more GGUF or checkpoint models with `LocalModel::load` and answers `POST /v1/chat/completions` the way vLLM and llama.cpp's server do. One binary, `cuca`, carries three subcommands.

<div class="dgm">
<div class="dgm-scroll">
<svg viewBox="0 0 660 190" role="img" aria-labelledby="srv-title srv-desc">
  <title id="srv-title">Where the three cuca subcommands sit</title>
  <desc id="srv-desc">Clients send chat requests to cuca balance. The balancer forwards each request to one of two cuca serve instances, and each instance runs its models in the in-process engine. cuca tui reads the stats API of the balancer, and can read any server directly.</desc>
  <defs>
    <marker id="srv-arrow" viewBox="0 0 8 8" refX="7" refY="4"
            markerWidth="6" markerHeight="6" orient="auto">
      <path d="M0 0.6 L7 4 L0 7.4 z" fill="var(--muted-foreground)"/>
    </marker>
  </defs>
  <text class="t-title" x="6" y="20">cuca serve, balance and tui</text>
  <rect class="blk" x="6" y="62" width="100" height="50" rx="6"/>
  <text class="t-sm t-mid" x="56" y="85">Clients</text>
  <text class="t-sm t-mid" x="56" y="99">OpenAI API</text>
  <rect class="blk blk-data" x="184" y="62" width="130" height="50" rx="6"/>
  <text class="t-sm t-mid t-data" x="249" y="85">cuca balance</text>
  <text class="t-sm t-mid" x="249" y="99">routes and proxies</text>
  <rect class="blk blk-data" x="400" y="32" width="150" height="50" rx="6"/>
  <text class="t-sm t-mid t-data" x="475" y="55">cuca serve</text>
  <text class="t-sm t-mid" x="475" y="69">engine and models</text>
  <rect class="blk blk-data" x="400" y="102" width="150" height="50" rx="6"/>
  <text class="t-sm t-mid t-data" x="475" y="125">cuca serve</text>
  <text class="t-sm t-mid" x="475" y="139">engine and models</text>
  <rect class="blk blk-ctl" x="184" y="136" width="130" height="40" rx="6"/>
  <text class="t-sm t-mid t-ctl" x="249" y="160">cuca tui</text>
  <path class="arw arw-data" d="M108 87 H180" marker-end="url(#srv-arrow)"/>
  <path class="arw arw-data" d="M316 78 L396 58" marker-end="url(#srv-arrow)"/>
  <path class="arw arw-data" d="M316 96 L396 126" marker-end="url(#srv-arrow)"/>
  <path class="arw arw-ctl" d="M249 134 V114" marker-end="url(#srv-arrow)"/>
  <text class="t-ax" x="262" y="128">stats API</text>
</svg>
</div>
<p class="dgm-cap"><b>The stats API is the one interface the terminal UI reads.</b> A balancer serves the same API as a server, with each upstream's own snapshot nested inside.</p>
</div>

## Pages

- [Model server](@/server/model-server.md): the `serve` command, its options and config file, the routes and the OpenAI wire format.
- [Balancer](@/server/balancer.md): the `balance` command, routing policies, prefix affinity, health checks and retries.
- [Stats API and metrics](@/server/stats-and-metrics.md): the JSON snapshot, every Prometheus series, the latency diagnosis and the logs.
- [Terminal UI](@/server/tui.md): the `tui` command, its panels and keys, and which resource figures each platform reports.
- [Serve a model over HTTP](@/guides/serve-a-model.md): a how-to from build to a streamed reply.

## Features

| Feature | Enables |
|---|---|
| `server` | `provider-local`, the HTTP server, the balancer, the stats API, the metrics and the system sampler, and the `cuca` binary |
| `server-tui` | `server`, `ratatui` and `crossterm`, for `cuca tui` |

Both features build alone, with no other provider. The [feature matrix](@/reference/features.md) lists the dependencies they own. A GPU build adds the backend feature:

```bash,name=Runs the same on all three platforms
cargo build --release --no-default-features --features server,provider-local-vulkan --bin cuca
```

The CPU kernels select their instructions at compile time, so a CPU server wants `RUSTFLAGS="-C target-cpu=native"` (Windows PowerShell: `$env:RUSTFLAGS = "-C target-cpu=native"`), as [Build requirements and tests](@/providers/local/build-and-tests.md) explains.

## What was verified

- Run on Windows with the CPU engine and Qwen3-0.6B Q8_0 (release build, `target-cpu=native`): `serve` (non-streaming and streaming chat, tool calls, two concurrent streams, a client disconnect, a prompt over the context length), `balance` over two servers (prefix affinity, concurrent spread, an upstream killed mid-run), `/metrics` and the stats API.
- Run on Windows on a Vulkan GPU (Radeon RX 9070 XT, `server-tui,provider-local-vulkan` release build) with the same model: streaming and four concurrent requests through `cuca serve --device vulkan`. The stats showed `device: vulkan`, a 3.5 GiB K/V pool and 0.6 GB of weights, this process's VRAM at 5.1 GiB through WMI, GPU utilization of 43% under load, and about 150 tokens per second over four concurrent requests once the kernels were compiled. The first request on a GPU takes about 10 seconds while kernels compile.
- The terminal UI was exercised live in a Windows ConPTY against a server and a balancer: both views, the keys, a reconnect after the node restarted, and the terminal restored on exit. Tests also draw it on ratatui's test backend.
- Covered by unit and in-process tests: the wire mapping, the routing rules, the diagnosis rules, and the balancer against mock upstreams.
- Compile-checked only: the Linux and macOS system samplers, other GPU backends through the server (Metal, CUDA, ROCm), and NVML.
