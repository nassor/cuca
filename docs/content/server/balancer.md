+++
title = "Balancer"
description = "cuca balance: start it in front of several servers, then the routing policies, prefix affinity, health checks, retries and routes."
template = "page.html"
weight = 2
+++

# Balancer

<dl class="page-facts">
<dt>In one line</dt>
<dd><code>cuca balance</code> sends each chat request to one of several <code>cuca serve</code> instances, keeps a conversation on the instance that already cached its prompt, and streams the answer back unchanged</dd>
<dt>You need</dt>
<dd>The <code>cuca</code> binary with the <code>server</code> feature, and the base URLs of two or more running <code>cuca serve</code> instances. They need <code>stats_api #true</code> for load-aware routing</dd>
<dt>Read this if</dt>
<dd>One engine does not carry your traffic, or you need the routing, health and retry rules exactly</dd>
</dl>

## Quick start

This example runs two servers and one balancer on one machine. Each server loads its own copy of the model.

1. Write three files. The two servers differ only in the port.

```kdl,name=a.kdl
serve {
    listen "127.0.0.1:8001"
    stats_api #true
    model { source "qwen3-0.6b-q8_0" }
}
```

```kdl,name=b.kdl
serve {
    listen "127.0.0.1:8002"
    stats_api #true
    model { source "qwen3-0.6b-q8_0" }
}
```

```kdl,name=balancer.kdl
balance {
    listen "127.0.0.1:8080"
    upstreams "http://127.0.0.1:8001" "http://127.0.0.1:8002"
}
```

2. Start the two servers and the balancer, each in its own terminal. Start the balancer once both servers log `every model is ready; serving`.

```bash,name=Linux and macOS
cuca serve --config a.kdl
cuca serve --config b.kdl
cuca balance --config balancer.kdl
```

```powershell,name=Windows (PowerShell)
cuca.exe serve --config a.kdl
cuca.exe serve --config b.kdl
cuca.exe balance --config balancer.kdl
```

3. Ask the balancer for its health.

```bash,name=Linux and macOS
curl http://127.0.0.1:8080/health
```

```powershell,name=Windows (PowerShell)
curl.exe http://127.0.0.1:8080/health
```

```text,name=Expected output, trimmed
{"status":"ok","healthy":2,"total":2,"upstreams":[{"id":"127.0.0.1:8001","url":"http://127.0.0.1:8001","healthy":true,"error":null}, ...]}
```

4. Send the same chat request as in [Model server](@/server/model-server.md#quick-start), but to port 8080. The reply is identical in shape. The balancer passes it through.

## Command line

`cuca balance [--config <FILE>]`. The one option is `--config` (`-c`), the KDL file, which defaults to `cuca.kdl`. Every other setting is a key of the `balance` section. The `log` section applies as for `serve`.

A missing or invalid file stops the command with exit status 1 and a message that names the file and the key. An unknown key or an unknown policy is an error.

```kdl,name=A balancer over two servers, with keys
variables {
    API_KEY "change-me"
}

balance {
    listen "127.0.0.1:8080"
    upstreams "http://10.0.0.1:8000" "http://10.0.0.2:8000"
    upstream_api_key "${API_KEY}"
    policy "prefix-affinity"
    health_interval "1s"
    stats_api #true
}
```

| `balance` key | Default | Meaning |
|---|---|---|
| `listen` | `127.0.0.1:8000` | address of the HTTP listener |
| `upstreams` | none; at least one | base URLs of the servers, `http://host:port` or `https://...`; one node with several arguments, or several nodes; 256 at most |
| `upstream_api_key` | none | sent to every upstream as `Authorization: Bearer <key>` |
| `api_key` | none | bearer key clients must send on every route except `/health` |
| `stats_api` | `#false` | serve the stats API with `role: balancer` |
| `policy` | `"prefix-affinity"` | `"prefix-affinity"`, `"least-loaded"` or `"round-robin"` |
| `health_interval` | `"1s"` | poll interval per upstream, `"100ms"` to `"60s"` |
| `latency_window` | `"60s"` | the sliding window of the latency figures |

The file is `examples/configs/cuca.kdl` in the repository. The syntax is on [Configuration files](@/reference/environment.md).

## How a request is routed

<div class="dgm">
<div class="dgm-scroll">
<svg viewBox="0 0 660 214" role="img" aria-labelledby="bal-title bal-desc">
  <title id="bal-title">How the balancer picks an upstream for one chat request</title>
  <desc id="bal-desc">The balancer reads the model name and builds the candidate list: healthy upstreams that serve the model. It gives each candidate a load score. Under prefix affinity it hashes the model and the prompt prefix with each candidate and takes the highest weight. If that target's score is above 2 times the lowest score plus 2, it sends the request to the least loaded candidate instead. Least loaded and round robin skip the hash.</desc>
  <defs>
    <marker id="bal-arrow" viewBox="0 0 8 8" refX="7" refY="4"
            markerWidth="6" markerHeight="6" orient="auto">
      <path d="M0 0.6 L7 4 L0 7.4 z" fill="var(--muted-foreground)"/>
    </marker>
  </defs>
  <text class="t-title" x="6" y="20">One chat request, one upstream</text>
  <rect class="blk blk-ctl" x="6" y="40" width="124" height="60" rx="6"/>
  <text class="t-sm t-mid t-ctl" x="68" y="64">1. Candidates</text>
  <text class="t-sm t-mid" x="68" y="78">healthy and</text>
  <text class="t-sm t-mid" x="68" y="90">serve the model</text>
  <rect class="blk blk-ctl" x="164" y="40" width="124" height="60" rx="6"/>
  <text class="t-sm t-mid t-ctl" x="226" y="64">2. Load score</text>
  <text class="t-sm t-mid" x="226" y="78">in flight + queue</text>
  <text class="t-sm t-mid" x="226" y="90">+ running share</text>
  <rect class="blk blk-data" x="322" y="40" width="156" height="60" rx="6"/>
  <text class="t-sm t-mid t-data" x="400" y="64">3. Rendezvous hash</text>
  <text class="t-sm t-mid" x="400" y="78">model + prompt prefix</text>
  <text class="t-sm t-mid" x="400" y="90">highest weight wins</text>
  <rect class="blk blk-data" x="512" y="40" width="142" height="60" rx="6"/>
  <text class="t-sm t-mid t-data" x="583" y="64">4. Busy check</text>
  <text class="t-sm t-mid" x="583" y="78">score above</text>
  <text class="t-sm t-mid" x="583" y="90">2 * min + 2?</text>
  <rect class="blk blk-data" x="322" y="146" width="156" height="46" rx="6"/>
  <text class="t-sm t-mid t-data" x="400" y="166">Send to the target</text>
  <text class="t-sm t-mid" x="400" y="180">same prefix, same cache</text>
  <rect class="blk blk-data" x="512" y="146" width="142" height="46" rx="6"/>
  <text class="t-sm t-mid t-data" x="583" y="166">Least loaded</text>
  <text class="t-sm t-mid" x="583" y="180">counted as fallback</text>
  <path class="arw arw-ctl" d="M132 70 H160" marker-end="url(#bal-arrow)"/>
  <path class="arw arw-ctl" d="M290 70 H318" marker-end="url(#bal-arrow)"/>
  <path class="arw arw-data" d="M480 70 H508" marker-end="url(#bal-arrow)"/>
  <path class="arw arw-data" d="M545 102 V118 H430 V142" marker-end="url(#bal-arrow)"/>
  <path class="arw arw-data" d="M600 102 V142" marker-end="url(#bal-arrow)"/>
  <text class="t-ax" x="440" y="114">no</text>
  <text class="t-ax" x="606" y="124">yes</text>
  <text class="t-ax" x="6" y="136">least-loaded: steps 1, 2, lowest score</text>
  <text class="t-ax" x="6" y="152">round-robin: step 1, next in turn</text>
</svg>
</div>
<p class="dgm-cap"><b>Prefix affinity hashes first and checks load second.</b> The balancer reads only the <code>model</code> field and the start of the prompt, and forwards your original bytes. The score and the check are defined below.</p>
</div>

## Discovery and health

One task per upstream polls three routes: `GET /health`, `GET /v1/models` and, when the upstream has it, `GET /cuca/v1/stats`. Each poll has a timeout of twice the interval (1 to 5 seconds). A 404 on the stats route marks the upstream as having no stats API. That is not an error. The balancer keeps only the latest result: one model list and one snapshot per upstream.

An upstream is healthy when its last `/health` answered 200 within three intervals and the proxy has not ejected it since.

- A failed poll makes the upstream unhealthy. The delay before the next poll doubles, up to 10 seconds.
- A snapshot older than three intervals is not used for scoring.
- A connect error, or a 5xx status before the first body byte, ejects the upstream at once, until its next successful poll.
- An upstream restarted with `stats_api #true` is probed for the stats route again after it recovers.

## Choosing the model and the candidates

The candidates are the healthy upstreams whose last `/v1/models` lists the model.

- A request with no `model` works when the healthy upstreams all serve the same single model. Otherwise it is a 400 `invalid_request` with `param: "model"`.
- A model no upstream has reported is a 404 `model_not_found`, once every upstream has completed a poll. Before that, and for a model that only unhealthy upstreams list, the answer is a 503 `no_healthy_upstream`.

## Policies

Each candidate has a load score: the requests the balancer has outstanding to it, plus the requests it reports waiting for that model, plus its running fraction of `max_sequences`. The last term stays below one request and breaks ties. Without a snapshot only the first term counts.

| Policy | Picks |
|---|---|
| `least-loaded` | the lowest score. Equal scores take turns. |
| `round-robin` | the candidates in turn |
| `prefix-affinity` | the candidate that cached this conversation's prefix, unless it is much busier than the rest (below) |

### Prefix affinity

An engine caches the K/V blocks of the prompts it served, so a request that shares a prefix with an earlier one only has to process what follows. Affinity sends such requests to the same upstream.

The prefix key is the canonical text of the leading `system` and `developer` messages and the first `user` message, as `role`, text and a zero byte per message, cut to 4096 bytes at a character boundary. Later turns of a conversation share the first user message, so they share the key. A request with no user text has no key and goes least loaded.

The balancer hashes the model and the key together with each candidate's id (`host:port`) and picks the highest weight. This is rendezvous hashing. A weight depends only on the key and that candidate. When an upstream leaves, only the keys it held move, and when it returns they move back. The other upstreams keep their conversations and their caches.

The target is passed over when it is much busier than the rest. When its score is above `2 * min_score + 2`, where `min_score` is the lowest score among the candidates, the request goes to the least-loaded candidate instead. `cuca_balancer_affinity_fallbacks_total` counts these.

## Proxying

- The upstream's status, headers (minus hop-by-hop ones) and body stream back as they arrive, with no buffering.
- `x-request-id` is passed to the upstream and echoed to the client. The balancer makes one when the client sent none.
- An attempt is retried on the next candidate only when the request never reached an upstream (a connect error) or the upstream answered 503 before any body byte. At most 2 retries are made, each on an upstream not yet tried.
- When no candidate is left, the last 503 body passes through, or the balancer answers 503 `no_healthy_upstream` or 502 `upstream_error`.
- Other failures after the request was sent, and every other status, are not retried. A 5xx other than 503 also ejects the upstream and passes through.
- A client that disconnects drops the upstream response, which closes the upstream connection, so the upstream engine stops generating.
- When `upstream_api_key` is set, each upstream request carries it as a bearer token.

## Routes

| Route | Answer |
|---|---|
| `POST /v1/chat/completions` | routed to an upstream; the response streams through |
| `GET /health` | `200` with the counts and each upstream's health when at least one is healthy, else `503` |
| `GET /v1/models` | the models of the healthy upstreams, deduplicated, each with the `upstreams` that serve it |
| `GET /version` | the version, with `"role":"balancer"` |
| `GET /metrics` | the balancer's Prometheus series, see [Stats API and metrics](@/server/stats-and-metrics.md#balancer-metrics) |
| `GET /cuca/v1/stats`, `GET /cuca/v1/stats/stream` | the stats API, only with `stats_api #true` |

Other routes are not proxied. `/v1/completions` is absent for the reason given on the [model server](@/server/model-server.md#routes) page.

## Errors

| Status | `code` | When |
|---|---|---|
| 400 | `invalid_request` | a body that is not JSON, or a missing `model` that is ambiguous |
| 401 | `invalid_api_key` | `api_key` is set and the request lacks it |
| 404 | `model_not_found` | no upstream serves the model |
| 413 | `request_too_large` | the body exceeds 32 MiB |
| 429 | `too_many_streams` | 64 stats streams are already open |
| 502 | `upstream_error` | the connection to the upstream failed after retries, or broke before a response |
| 503 | `no_healthy_upstream` | no healthy upstream serves the model |

## Limits

- At most 256 upstreams.
- Per upstream: one stats snapshot, one model list, a latency ring of 1,024 records, and fixed counters and histograms.
- A request body is held only while the request is routed and retried. Response bodies are never buffered, except a 503 that will be retried.
- At most 64 stats streams are open at once.

## What was tested

Two `cuca serve` instances with `stats_api #true` (Qwen3-0.6B Q8_0, CPU, Windows) behind one balancer. A streamed completion went through. A repeated conversation prefix reached the same upstream, and the second request had `cached_tokens` above zero. Eight concurrent distinct requests split four and four. With one upstream killed, requests kept succeeding on the other and the stats reported `upstream_down`. The other policies, the affinity fallback and the API keys are covered by automated tests with mock upstreams, not by a live run.
