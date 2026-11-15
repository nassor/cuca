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

The `cuca` program turns the local engine into a service. A *server* answers HTTP requests from any OpenAI client, and a *load balancer* shares requests between several servers. You choose a command:

| Command | What it does |
|---|---|
| `cuca serve` | loads one or more models and answers the OpenAI chat API, the way vLLM and llama.cpp's server do |
| `cuca balance` | sits in front of several `cuca serve` instances and sends each request to one of them |
| `cuca tui` | shows a running server or balancer live in your terminal |

All three read the same stats API (a JSON page of live numbers). To run your first server, follow [Serve a model over HTTP](@/guides/serve-a-model.md). The server is built on the [`provider-local`](@/providers/local/_index.md) engine.

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
<p class="dgm-cap"><b>The stats API is the one interface the terminal UI reads.</b> A balancer serves the same API as a server, with each upstream's own snapshot inside it. An upstream is one server behind the balancer.</p>
</div>

## Pages

- [Model server](@/server/model-server.md): write `cuca.kdl`, start `cuca serve`, call it. Then the keys, routes, errors and limits.
- [Balancer](@/server/balancer.md): put `cuca balance` in front of several servers. Then the routing rules, health checks and retries.
- [Stats API and metrics](@/server/stats-and-metrics.md): the JSON snapshot, every Prometheus series, the "why is it slow" list and the logs.
- [Terminal UI](@/server/tui.md): the `tui` command, its panels and keys, and which figures each platform reports.
- [Serve a model over HTTP](@/guides/serve-a-model.md): a step-by-step how-to from build to a streamed reply.

## Build the binary

| Feature | Gives you |
|---|---|
| `server` | the `cuca` binary with `serve` and `balance`, the stats API, `/metrics`, and the local engine |
| `server-tui` | everything in `server`, plus `cuca tui` |

Both features build with no other provider. The [feature matrix](@/reference/features.md) lists what each one pulls in. To serve from a GPU, add a backend feature such as `provider-local-vulkan`:

```bash,name=Runs the same on all three platforms
cargo build --release --no-default-features --features server,provider-local-vulkan --bin cuca
```

The CPU kernels choose their instructions when you build. A CPU server should build with `RUSTFLAGS="-C target-cpu=native"` (Windows PowerShell: `$env:RUSTFLAGS = "-C target-cpu=native"`). [Build requirements](@/providers/local/build-and-tests.md) explains why.

## What to expect

| Setup | Result |
|---|---|
| Qwen3-0.6B Q8_0 on a Vulkan GPU, four requests at once, Windows, release build | about 150 tokens per second in total |
| The same, first request after start | about 10 seconds longer, while the GPU kernels compile |

A token is a word piece, roughly four characters of English. Speed depends on your model, device and prompt, so measure on your own hardware.

## Platform support

- `serve`, `balance` and the terminal UI run on Windows, Linux and macOS.
- The CPU engine and the Vulkan GPU engine are tested through the server on Windows.
- The Linux and macOS resource readers, the Metal, CUDA and ROCm backends through the server, and NVIDIA readings are compile-checked only. They build, and nobody has run them against a live server yet.
