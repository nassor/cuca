+++
title = "Balancer"
description = "cuca balance: routing policies, prefix affinity with rendezvous hashing, health checks, ejection, retries and the balancer's routes."
template = "page.html"
weight = 2
+++

# Balancer

<dl class="page-facts">
<dt>In one line</dt>
<dd><code>cuca balance</code> sends each chat request to one of several <code>cuca serve</code> instances, keeps a conversation on the instance that caches its prefix, and streams the answer back unchanged</dd>
<dt>You need</dt>
<dd>The <code>server</code> feature and the base URLs of the servers. The servers need <code>--stats-api</code> for load-aware routing and for the per-upstream view</dd>
<dt>Read this if</dt>
<dd>One engine does not carry your traffic, or you need the routing rule, the health rules and the retry rules exactly</dd>
</dl>

## Command line

`cuca balance --upstream <url> [--upstream <url> ...] [options]`.

| Option | Default | Meaning |
|---|---|---|
| `--listen <addr>` | `127.0.0.1:8000` | address of the HTTP listener |
| `--upstream <url>` | none; at least one | base URL of a server, `http://host:port` or `https://...`; repeatable, 256 at most |
| `--upstream-api-key <key>` | none; `CUCA_UPSTREAM_API_KEY` | sent to every upstream as `Authorization: Bearer <key>` |
| `--api-key <key>` | none; `CUCA_API_KEY` | bearer key clients must send on every route except `/health` |
| `--stats-api` | off | serve the stats API with `role: balancer` |
| `--policy <p>` | `prefix-affinity` | `prefix-affinity`, `least-loaded` or `round-robin` |
| `--health-interval-ms <ms>` | `1000` | poll interval per upstream, 100 to 60000 |
| `--config <file>` | none | TOML file with a `[balancer]` table |
| `--log-format`, `--log-level` | as `serve` | |

The `[balancer]` table takes `listen`, `upstreams`, `upstream_api_key`, `api_key`, `stats_api`, `policy` and `health_interval_ms`. A command line value wins, and the upstream lists of both are joined, the command line's first. An unknown key is an error.

Linux and macOS:

```bash,name=A balancer over two servers
cuca balance --upstream http://10.0.0.1:8000 --upstream http://10.0.0.2:8000 --stats-api
```

Windows (PowerShell):

```powershell,name=A balancer over two servers
cuca.exe balance --upstream http://10.0.0.1:8000 --upstream http://10.0.0.2:8000 --stats-api
```

## Routes

| Route | Answer |
|---|---|
| `POST /v1/chat/completions` | routed to an upstream; the response streams through |
| `GET /health` | `200` with the counts and each upstream's health when at least one is healthy, else `503` |
| `GET /v1/models` | the models of the healthy upstreams, deduplicated, each with the `upstreams` that serve it |
| `GET /version` | the version, with `"role":"balancer"` |
| `GET /metrics` | the balancer's Prometheus series, see [Stats API and metrics](@/server/stats-and-metrics.md#balancer-metrics) |
| `GET /cuca/v1/stats`, `GET /cuca/v1/stats/stream` | the stats API, only with `--stats-api` |

Other routes are not proxied. `/v1/completions` is absent for the reason given on the [model server](@/server/model-server.md#routes) page.

## Discovery and health

One task per upstream polls `GET /health`, `GET /v1/models` and, when the upstream has it, `GET /cuca/v1/stats`, each with a timeout of twice the interval (1 to 5 seconds). A 404 on the stats route marks the upstream as having no stats API, which is not an error. The balancer keeps only the latest result: one model list and one snapshot per upstream.

An upstream is healthy when its last `/health` answered 200 within three intervals and the proxy has not ejected it since. A poll that fails makes it unhealthy, and the delay before the next poll doubles up to 10 seconds. A snapshot older than three intervals is not used for scoring.

Passive health acts at once. A connect error, or a 5xx status before the first body byte, ejects the upstream until its next successful poll. An upstream restarted with `--stats-api` is probed for the stats route again after it recovers.

## Choosing the model and the candidates

The balancer parses the body only to read `model` and the prompt prefix, and forwards the original bytes. The candidates are the healthy upstreams whose last `/v1/models` lists the model.

- A request with no `model` works when the healthy upstreams all serve the same single model. Otherwise it is a 400 `invalid_request` with `param: "model"`.
- A model no upstream has reported is a 404 `model_not_found`, once every upstream has completed a poll. Before that, and for a model that only unhealthy upstreams list, the answer is a 503 `no_healthy_upstream`.

## Policies

Each candidate has a load score: the requests the balancer has outstanding to it, plus the requests it reports waiting for that model, plus its running fraction of `max_sequences`. The last term stays below one request and breaks ties. Without a snapshot only the first term counts.

- `least-loaded`: the lowest score. Equal scores take turns.
- `round-robin`: the candidates in turn.
- `prefix-affinity`: see below.

### Prefix affinity

An engine caches the K/V blocks of the prompts it served, so a request that shares a prefix with an earlier one prefills only what follows. Affinity sends such requests to the same upstream.

The prefix key is the canonical text of the leading `system` and `developer` messages and the first `user` message, as `role`, text and a zero byte per message, cut to 4096 bytes at a character boundary. Later turns of a conversation share the first user message, so they share the key. A request with no user text has no key and goes least loaded.

The balancer hashes the model and the key with each candidate's id (`host:port`) and picks the highest weight: rendezvous hashing. A weight depends only on the key and that candidate, so when an upstream leaves, only the keys it held move, and when it returns they move back. The other upstreams keep their conversations and their caches.

The target is passed over when it is much busier than the rest: when its score is above `2 * min_score + 2`, where `min_score` is the lowest score among the candidates, the request goes to the least-loaded candidate instead. `cuca_balancer_affinity_fallbacks_total` counts these.

## Proxying

- The upstream's status, headers (minus hop-by-hop ones) and body stream back as they arrive, with no buffering. `x-request-id` is propagated to the upstream and echoed to the client, and one is made when the client sent none.
- An attempt is retried on the next candidate only when the request never reached an upstream (a connect error) or the upstream answered 503 before any body byte. At most 2 retries are made, each on an upstream not yet tried. When no candidate is left, the last 503 body passes through, or the balancer answers 503 `no_healthy_upstream` or 502 `upstream_error`.
- Other failures after the request was sent, and every other status, are not retried. A 5xx other than 503 also ejects the upstream and passes through.
- A client that disconnects drops the upstream response, which closes the upstream connection, so the upstream engine stops generating.
- When `--upstream-api-key` is set, each upstream request carries it as a bearer token.

### Errors

| Status | `code` | When |
|---|---|---|
| 400 | `invalid_request` | a body that is not JSON, or a missing `model` that is ambiguous |
| 401 | `invalid_api_key` | `--api-key` is set and the request lacks it |
| 404 | `model_not_found` | no upstream serves the model |
| 413 | `request_too_large` | the body exceeds 32 MiB |
| 429 | `too_many_streams` | 64 stats streams are already open |
| 502 | `upstream_error` | the connection to the upstream failed after retries, or broke before a response |
| 503 | `no_healthy_upstream` | no healthy upstream serves the model |

## Bounds

At most 256 upstreams. Per upstream: one stats snapshot, one model list, a latency ring of 1,024 records, and fixed counters and histograms. A request body is held only while the request is routed and retried. Response bodies are never buffered, except a 503 that will be retried. At most 64 stats streams are open at once.

## Verified

On Windows, with two `cuca serve --stats-api` instances (Qwen3-0.6B Q8_0, CPU) behind one balancer: a streamed completion through it, a repeated conversation prefix reaching the same upstream with `cached_tokens` above zero on the second request, eight concurrent distinct requests split four and four, and one upstream killed, after which requests kept succeeding on the other and the stats reported `upstream_down`. The other policies, the affinity fallback and the API keys are covered by unit tests and mock upstreams, not by a live run.
