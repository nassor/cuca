+++
title = "Stats API and metrics"
description = "The stats snapshot a server and a balancer serve, every Prometheus series, the latency diagnosis with its causes and thresholds, and the request logs."
template = "page.html"
weight = 3
+++

# Stats API and metrics

<dl class="page-facts">
<dt>In one line</dt>
<dd>One JSON snapshot per node (<code>/cuca/v1/stats</code>), a Prometheus endpoint (<code>/metrics</code>), a ranked list of reasons for the current latency, and one log line per request</dd>
<dt>You need</dt>
<dd>A running <code>cuca serve</code> or <code>cuca balance</code>. The stats routes need <code>--stats-api</code>; <code>/metrics</code> is always on</dd>
<dt>Read this if</dt>
<dd>You are building a dashboard or an alert, reading the TUI's figures, or deciding which threshold a diagnosis uses</dd>
</dl>

## Stats API

`GET /cuca/v1/stats` returns one `StatsSnapshot`. `GET /cuca/v1/stats/stream` is server-sent events with one snapshot per event, one per second by default; `?interval_ms=` is clamped to 250 to 10000. At most 64 streams are open at once. The types are `cuca::server::stats::StatsSnapshot` and the structs under it, plain serde data. Every struct deserializes with defaults, so a reader tolerates fields a newer cuca adds. Durations are seconds (`*_s`) or milliseconds (`*_ms`) as named, sizes are bytes, ratios run 0 to 1 and percentages 0 to 100.

| Field | Content |
|---|---|
| `node` | `id` (`host:port`), `role` (`server` or `balancer`), `version`, `listen`, `started_at` (RFC 3339 UTC), `started_at_ms`, `uptime_s` |
| `models[]` | one entry per model, below |
| `latency` | `window_s` and `ttft`, `itl`, `e2e`, `queue`, each `{p50, p95, p99, mean, count}` in seconds, over all models |
| `requests` | `in_flight`, `total`, `errors`, `rate_per_s` |
| `system` | the machine, below |
| `diagnosis[]` | ranked reasons for the latency, below |
| `upstreams[]` | empty on a server; on a balancer one entry per upstream |

A model entry has:

- `name`, `source`, `device`, `context_length`, `state` (`loading`, `ready` or `failed`), `error`, `load_progress`.
- `max_sequences`, `running`, `waiting`.
- `kv`: `total_blocks`, `free_blocks`, `cached_blocks`, `used_blocks`, `bytes`, `usage_ratio` (used over total; cached blocks are evictable and count as available).
- Window figures: `prefix_hit_ratio`, `preemptions_per_min`, `tokens_per_s` (`prompt` counts tokens the engine computed, `generation` all generated tokens), `steps_per_s`, `rows_per_s`, `mean_rows_per_step`, `prefill_row_share`, `mean_step_ms`, `mean_queue_ms`. `last_step_ms` is the last forward pass.
- `memory`: `weights_bytes` (the GGUF size), `kv_bytes`, `device`.
- `expert_tiers` (tiered experts only): `disk_reads_per_s`, `read_bytes_per_s`, `hit_ratio`, `ram_resident_bytes`, `device_resident_bytes`.
- `requests` and `latency` for that model.

Rates come from a ring of one-second samples of the engine's counters, 300 per model (five minutes): a rate is the difference between the newest sample and the oldest one within the last 10 seconds, and `preemptions_per_min` uses the last 60 seconds. The latency figures come from the finished requests of the sliding window, 60 seconds and at most 4,096 requests.

The system part, filled by the sampler described on the [terminal UI](@/server/tui.md#what-each-platform-reports) page, has `cpu` (`usage_pct`, `per_core_pct`, `cores`, `load_avg`, `brand`), `memory` (RAM, swap and the paging rates `swap_in_bytes_per_s` and `swap_out_bytes_per_s`, `None` where the OS reports none), `disks`, `disk_io`, `gpus[]` (`util_pct`, `vram_*`, `process_vram_bytes`, `shared_vram_used_bytes`, `temperature_c`, `power_w`, `source`, each `None` when unknown), `process` and `sampled_at_ms`.

```text,name=The shape of a snapshot from a CPU server (Windows, Qwen3-0.6B Q8_0), trimmed
{ "node": { "id": "NASSOR-DESKTOP:18000", "role": "server", "version": "0.7.0", ... },
  "models": [ { "name": "qwen3-0.6b-q8_0", "state": "ready", "device": "cpu",
                "context_length": 8192, "max_sequences": 4,
                "kv": { "total_blocks": 1024, "bytes": 3758096384, ... },
                "memory": { "weights_bytes": 639446688, "device": "cpu", ... }, ... } ],
  "latency": { "window_s": 60, "ttft": { ... }, ... },
  "requests": { ... }, "system": { ... }, "diagnosis": [], "upstreams": [] }
```

### Balancer snapshots

A balancer sets `role` to `balancer`. `upstreams[]` has one entry per upstream: `id`, `url`, `healthy`, `in_flight`, `routed_total`, `errors_total`, `latency` (measured at the balancer), `error` (the cause in a few words, such as `connection refused`, `timed out` or `HTTP 503`), `snapshot` and `snapshot_age_s`. The snapshot is the upstream's own latest one when it has the stats API; a down upstream keeps its last snapshot for context, and `snapshot_age_s` says how old it is. The balancer does not report a down upstream's old diagnoses, only `upstream_down`. In `latency`, `ttft` is the time to the first body byte and `itl` is the mean gap between SSE events as they pass through, measured from the first content event (the first event is the role prelude, sent before prefill) to the last. It is a balancer-side approximation of the engine's per-token gap: a chunk may carry several events, the usage chunk counts as one, `[DONE]` does not, and plain JSON replies have none. `models` is the union over the healthy upstreams, with counts and rates summed and means weighted by the rate they describe. `latency` and `requests` are the balancer's, `system` is the balancer host's, and `diagnosis` holds the balancer's causes followed by each healthy upstream's own, tagged with `upstream`.

## Diagnosis

`diagnosis[]` lists the reasons latency is what it is, most severe first, ties in rule order. Each entry has `severity` (`info`, `warn`, `critical`), `cause`, `model` or `upstream` when it concerns one, a `message`, and `evidence`, a map of the measurements that triggered it. A reader treats an unknown `cause` as `other`.

The rules below run over the latest figures. The resource rules (GPU, CPU, memory, disk reads) fire only while the server is busy: a model is running or waiting, a request is in flight, or a request finished in the window.

| Cause | Fires when | Severity |
|---|---|---|
| `model_loading` | a model is loading | info |
| `model_failed` | a model failed to load | critical |
| `queue_saturated` | `waiting` above 0 and `running` at `max_sequences` | warn; critical at `waiting` of twice `max_sequences` |
| `kv_pool_full` | K/V usage at 90% or more | warn; critical at 97% |
| `preemptions` | 1 or more preemptions per minute | warn; critical at 10 |
| `slow_consumer` | clients stall 20% or more of the response time, over at least 3 requests | warn |
| `prefill_bound` | mean prompt of 512 tokens or more, prefix hit ratio under 25%, median time to first token of 1 s or more | warn |
| `prefill_interference` | 50% or more of the rows are prompt rows, 2 or more running, steps of 50 ms or more | info |
| `large_batch` | 8 or more rows per step and steps of 100 ms or more | info |
| `step_time_growth` | the last 15 seconds average 1.5 times the step time of the 45 before them, and 20 ms more, with 5 steps in each | warn |
| `gpu_saturated` | a GPU at 95% utilization or more | warn |
| `vram_near_full` | a GPU's memory at 92% or more | warn; critical at 98% |
| `cpu_saturated` | the CPU at 90% or more, saying whether cuca or other processes use it | warn; critical at 97% |
| `memory_pressure` | available RAM under 10%, or paging in at 32 MiB/s or more while swap is in use; where the OS reports no paging rate, swap growth of 64 MiB over 30 s | warn; critical under 3% available |
| `disk_reads` | a tiered model reads experts at 20 MiB/s or more; without tiers, the process reads 50 MiB/s or more | warn |

The paging rate counts only while swap holds pages because Windows also counts reads of memory-mapped files as page-ins, so a cold model load alone does not raise `memory_pressure`.

A balancer adds:

| Cause | Fires when | Severity |
|---|---|---|
| `no_healthy_upstream` | no upstream is healthy | critical |
| `upstream_down` | an upstream is unhealthy | warn; critical when none is healthy |
| `upstream_saturated` | every healthy upstream, all with stats, has all slots busy and requests waiting | warn; critical at waiting of twice the slots |
| `load_imbalance` | under prefix affinity, with 2 or more healthy upstreams, one score is above `3 * min_score + 4` | warn |
| `upstream_slow` | with at least 5 requests on each of 2 or more upstreams, one's p95 time to first byte is above 3 times the others' mean and 0.5 s | warn |

The limits are the fields of `Thresholds` in `cuca::server::diagnosis` and `cuca::server::balancer::diagnosis`.

## Metrics

`GET /metrics` is the Prometheus text format 0.0.4. Histograms use fixed buckets: time to first token and end to end are vLLM's, and the queue histogram runs from 1 ms to 300 s. The `model` label has at most 16 values, `code` is one of 200, 400, 401, 404, 413, 499, 500, 502 and 503 (other codes count in their class), and `gpu` and `name` come from the sampler.

### Model server metrics

| Series | Type | Meaning |
|---|---|---|
| `cuca_uptime_seconds` | gauge | seconds since start |
| `cuca_http_requests_total{code}` | counter | responses, counted when headers go out |
| `cuca_rejected_requests_total{code}` | counter | chat requests refused before reaching a model |
| `cuca_model_ready{model}` | gauge | 1 when the model serves |
| `cuca_requests_total{model,code}` | counter | finished requests; 499 is a client that went away |
| `cuca_requests_in_flight{model}` | gauge | queued, prefilling or generating |
| `cuca_prompt_tokens_total{model}`, `cuca_prompt_tokens_cached_total{model}`, `cuca_generation_tokens_total{model}` | counter | tokens of finished requests, and the prompt tokens served from the prefix cache |
| `cuca_response_stall_seconds_total{model}` | counter | generation waiting for clients to read |
| `cuca_request_ttft_seconds{model}` | histogram | arrival to first token |
| `cuca_request_inter_token_seconds{model}` | histogram | mean gap between a request's tokens, one observation per request |
| `cuca_request_e2e_seconds{model}` | histogram | arrival to last token |
| `cuca_request_queue_seconds{model}` | histogram | arrival to admission by the engine |
| `cuca_requests_running{model}`, `cuca_requests_waiting{model}`, `cuca_max_sequences{model}` | gauge | engine slots |
| `cuca_kv_cache_usage_ratio{model}`, `cuca_kv_cache_blocks_total{model}`, `cuca_kv_cache_blocks_cached{model}`, `cuca_kv_cache_bytes{model}` | gauge | the K/V pool |
| `cuca_model_weights_bytes{model}` | gauge | the GGUF size |
| `cuca_prefix_cache_hit_tokens_total{model}` | counter | prompt tokens served from the prefix store |
| `cuca_preemptions_total{model}` | counter | sequences preempted for blocks |
| `cuca_engine_steps_total{model}` | counter | forward steps |
| `cuca_engine_rows_total{model,kind}` | counter | rows, `kind` `prefill` or `decode` |
| `cuca_engine_step_seconds_total{model}`, `cuca_engine_last_step_seconds{model}`, `cuca_engine_last_step_rows{model}` | counter, gauge | forward pass time and size |
| `cuca_engine_admissions_total{model}`, `cuca_engine_queue_wait_seconds_total{model}` | counter | admitted requests and their total wait |
| `cuca_engine_generated_tokens_total{model}` | counter | tokens the engine generated |

### System metrics

Both a server and a balancer expose these from their own host.

| Series | Meaning |
|---|---|
| `cuca_system_cpu_usage_percent`, `cuca_system_cpu_cores` | CPU load and logical cores |
| `cuca_system_memory_total_bytes`, `_used_bytes`, `_available_bytes` | RAM |
| `cuca_system_swap_total_bytes`, `_used_bytes` | swap or page file |
| `cuca_system_swap_in_bytes_per_second`, `_out_bytes_per_second` | paging rates; absent where the OS reports none |
| `cuca_system_disk_read_bytes_per_second`, `_write_bytes_per_second` | disk throughput, system-wide where the OS reports per-disk counters |
| `cuca_process_cpu_percent`, `cuca_process_resident_memory_bytes` | this process |
| `cuca_process_disk_read_bytes_per_second`, `_write_bytes_per_second` | this process |
| `cuca_gpu_utilization_percent{gpu,name}` | GPU utilization |
| `cuca_gpu_memory_total_bytes`, `_used_bytes`, `cuca_gpu_process_memory_bytes`, `cuca_gpu_shared_memory_used_bytes` | device memory, this process's part, and system memory the GPU uses |
| `cuca_gpu_temperature_celsius`, `cuca_gpu_power_watts` | where the platform reports them |

A GPU series is absent for a GPU that does not report that figure.

### Balancer metrics

| Series | Type | Meaning |
|---|---|---|
| `cuca_uptime_seconds`, `cuca_http_requests_total{code}`, `cuca_rejected_requests_total{code}` | | as on a server |
| `cuca_balancer_healthy_upstreams` | gauge | upstreams that take requests |
| `cuca_balancer_upstream_healthy{upstream}` | gauge | 1 when the upstream takes requests |
| `cuca_balancer_upstream_in_flight{upstream}` | gauge | requests outstanding to it |
| `cuca_balancer_routed_total{upstream}` | counter | requests routed to it |
| `cuca_balancer_requests_total{upstream,code}` | counter | its responses; 502 is a failed connection, 499 a client that went away |
| `cuca_balancer_request_ttft_seconds{upstream}` | histogram | arrival to the first body byte at the balancer |
| `cuca_balancer_request_e2e_seconds{upstream}` | histogram | arrival to the end of the body |
| `cuca_balancer_retries_total` | counter | attempts repeated on another upstream |
| `cuca_balancer_affinity_fallbacks_total` | counter | affinity targets passed over |
| `cuca_balancer_decisions_total{decision}` | counter | `affinity`, `fallback`, `least_loaded` or `round_robin`, per attempt |

The system metrics above follow.

## Logs

Logs go to standard output through `tracing`. `--log-format json` writes one JSON object per line. `--log-level` takes precedence over `RUST_LOG`.

Every request gets an `x-request-id` (an incoming one is kept) and an `http` span. The `cuca::access` target carries two kinds of line:

- When response headers go out: `status` and `latency_ms`, at `info` (at `debug` for `/health`, `/metrics` and the stats route).
- When a chat request finishes: `request_id`, `model`, `status`, `outcome`, `prompt_tokens`, `completion_tokens`, `cached_tokens`, `queue_ms`, `ttft_ms`, `itl_ms`, `e2e_ms` and `finish_reason`. A balancer's line has `upstream`, `decision`, `retries`, `status`, `ttft_ms` and `e2e_ms` inside a `proxy` span.

Model loads log their download, conversion or merge progress about every 5%, then one `model ready` line with the context length, K/V and weight sizes and `max_sequences`.

## Bounds

| What | Bound | At the bound |
|---|---|---|
| latency window | 60 s, 4,096 requests (a balancer: 1,024 per upstream) | the oldest record drops |
| sample rings | 300 samples per model and for the machine | the oldest sample drops |
| metric series | fixed atomics; labels are the model, the code, the upstream, the GPU | none grows with traffic |
| stats streams | 64 open | a 429 `too_many_streams` |
| models, upstreams | 16 per server, 256 per balancer | startup fails |
