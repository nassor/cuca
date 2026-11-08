+++
title = "Terminal UI"
description = "cuca tui: a live terminal view of a server or a balancer, its panels and keys, and what each platform reports for CPU, memory, disk and GPU."
template = "page.html"
weight = 4
+++

# Terminal UI

<dl class="page-facts">
<dt>In one line</dt>
<dd><code>cuca tui</code> follows a node's stats stream and shows resources, latency, the ranked reasons for it, and the node's models or upstreams</dd>
<dt>You need</dt>
<dd>The <code>server-tui</code> feature, and a <code>cuca serve</code> or <code>cuca balance</code> started with <code>--stats-api</code></dd>
<dt>Read this if</dt>
<dd>You want to see why a server is slow while it runs, or need to know which resource figures a platform reports</dd>
</dl>

## Run it

`cuca tui [--url <url>] [--api-key <key>] [--interval-ms <ms>]`.

| Option | Default | Meaning |
|---|---|---|
| `--url` | `http://127.0.0.1:8000` | base URL of the server or balancer |
| `--api-key` | none; `CUCA_API_KEY` | bearer key, when the node has one |
| `--interval-ms` | `1000` | time between snapshots, at least 100; the node clamps it to 250 to 10000 |

Linux and macOS:

```bash,name=Watch a local server
cuca tui --url http://127.0.0.1:8000
```

Windows (PowerShell):

```powershell,name=Watch a local server
cuca.exe tui --url http://127.0.0.1:8000
```

The UI follows `GET /cuca/v1/stats/stream`. When the stream breaks it reconnects with a backoff of up to 15 seconds and keeps showing the last snapshot. It restores the terminal on exit, on error and on a panic.

## Layout

A terminal of 110 columns by 34 rows or more shows every panel. A smaller one shows the focused pane alone under a one-line summary of the worst diagnosis, so 80 by 24 stays usable. Below 40 by 10 the UI asks for a larger terminal.

- Header: who is watched, the connection state and the request counters.
- Resources: CPU with per-core usage, RAM, swap and paging rate, each GPU's utilization, VRAM and memory split, disk read and write rates, disk space, and this process. Rows drop from the bottom when the panel is short, keeping CPU, RAM and each GPU's utilization and VRAM longest.
- Latency: a table of the window's time to first token, inter-token latency, end to end and queue wait (percentiles, a p95 trend and the count), tokens per second and in-flight sparklines, and a p95 history chart when there is room. The "Why is it slow?" list shows the node's [diagnosis](@/server/stats-and-metrics.md#diagnosis) with severity badges and evidence.
- List: for a server, one row per model (state, device, running, waiting and slots, K/V usage, cached blocks, prefix hit ratio, generation and prompt tokens per second, step time, rows per step, preemptions per minute, weight and K/V sizes). For a balancer, one row per upstream (health, routed and error counts, latency, models, last error).

Colors: green is fine, yellow costs latency, red is a bottleneck.

## Keys

| Key | Action |
|---|---|
| `q`, `Esc` | back from a drilled upstream, otherwise quit |
| `Ctrl-C` | quit |
| `Tab`, `Shift-Tab` | next and previous pane |
| `Up`, `Down`, `k`, `j` | select a row; in the Latency pane, scroll the reasons |
| `Enter` | details of the selected model, or drill into the selected upstream |
| `p` | pause or resume updates |
| `?`, `h`, `F1` | help |

Drilling into an upstream shows that upstream's own snapshot, with its resources and models, from the balancer's stream.

## What each platform reports

The sampler runs one thread, `cuca-sysmon`, that probes every second and keeps the latest snapshot. A probe that fails leaves its figures unknown, shown as `n/a`, and never stops the sampler.

| Figure | Source |
|---|---|
| CPU, RAM, swap, disk space, this process | `sysinfo`, on every platform |
| Paging rates | Windows: WMI page counters, where page-ins include memory-mapped file reads. Linux: `/proc/vmstat`. macOS: unknown |
| Disk throughput | system-wide per-disk counters, or this process's own where the platform reports none |
| GPU on Windows | WMI performance counters (`GPUEngine`, `GPUAdapterMemory`, `GPUProcessMemory`) and the registry's adapter table, for every vendor |
| GPU on Linux | `sysfs` for amdgpu, i915 and xe, `/proc/self/fdinfo` for this process's VRAM, NVML for NVIDIA, loaded at run time |
| GPU on macOS | `ioreg -r -d 1 -c IOAccelerator`, at most every two seconds |

Not reported:

- Windows: GPU temperature and power, per-process thread count, load average.
- Linux: GPU utilization on i915 and xe, and VRAM size on Intel integrated GPUs.
- macOS: GPU memory size (unified memory, only the amount in use), temperature and power.
- Every platform: at most 16 GPUs and 64 disks are listed.

The Windows sampler was run here against a Radeon RX 9070 XT and an integrated Radeon, and reported utilization, VRAM total and used, this process's VRAM while serving on Vulkan, and shared GPU memory. The Linux and macOS samplers, and NVML, are compile-checked only. The UI was exercised live in a Windows ConPTY (server and balancer views, keys, reconnect, terminal restored on exit) and is also covered by tests that draw it on ratatui's test backend.
