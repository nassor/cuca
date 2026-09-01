+++
title = "Speculative"
description = "The fast/slow model orchestrator: complexity routing, the draft and fallback pipeline, and the client pool."
template = "page.html"
weight = 2
+++

# Speculative

<dl class="page-facts">
<dt>In one line</dt>
<dd>Routes a turn to a fast or slow model tier by complexity, drafts speculatively on the fast tier, and falls back to the slow tier on rejection or latency.</dd>
<dt>You need</dt>
<dd>The <code>service-speculative</code> feature and a <code>SwappableModelPair</code>.</dd>
<dt>Read this if</dt>
<dd>You are attaching a <code>ModelOrchestrator</code> to a client, or tuning its complexity thresholds.</dd>
</dl>

## Location

`service-speculative` is a service, not a plugin: its implementation lives in `src/services/orchestrator.rs`, gated behind the same feature flag.

## Entry types

`ModelOrchestrator`, `SwappableModelPair`, `ClientPool`, `Complexity`, `ComplexityEvaluator`, `DraftValidator`, `JsonToolDraftValidator`, `TurnExecutor`.

## Attaching

`ModelOrchestrator` is not a `CucaPlugin`. It is attached with `CucaClientBuilder::with_orchestrator(orchestrator)`. When one is attached, `CucaClient::generate_stream` runs `on_request` hooks as usual, then hands the whole turn to `ModelOrchestrator::execute_adaptive_turn` instead of dispatching to a provider adapter directly.

## `SwappableModelPair`

| Field | Meaning |
|---|---|
| `fast_provider`, `fast_model_id` | The low-latency routing tier |
| `slow_provider`, `slow_model_id` | The high-capacity tier |
| `latency_threshold_ms` | Milliseconds the fast tier has to produce its next block before the orchestrator may swap to the slow tier |
| `fallback_on_tool_error` | Whether a rejected draft block triggers a fallback to the slow tier |

## `ModelOrchestrator`

| Method | Effect |
|---|---|
| `new(config, pool)` | Default `ComplexityEvaluator` and `JsonToolDraftValidator`; both tiers start on their provider adapters' own default endpoints |
| `with_executors(config, pool, fast, slow)` | Injects tier executors directly |
| `with_endpoint(base_url, api_key)` | Points both pooled tier clients at `base_url`, with `api_key` as their credential when given; rebuilds the tier executors |
| `with_session_store(store, session_id)` | Gated on `plugin-session-log`; attaches a [session log](@/plugins/session-log.md) store for `SessionEvent::ModelSwap` records |
| `client_pool()` | Returns the shared `ClientPool` |
| `execute_adaptive_turn(request)` | Runs the three-stage pipeline and returns the block stream |

## Pipeline

1. Complexity routing: `Complexity::Slow` requests go straight to the slow tier with no draft phase.
2. Speculative draft: the fast tier streams; each block passes through `DraftValidator`, rejecting malformed tool calls, invalid JSON, or low confidence.
3. Fallback cascade: when `fallback_on_tool_error` is set, a rejection re-routes the turn to the slow tier with the captured error state, up to two cascades; exhaustion surfaces the last rejection as `CucaError::Provider`.

The fast tier gets `latency_threshold_ms` to produce its next block; once that deadline passes while the fast stream is pending, the orchestrator swaps to the slow tier at the next poll. Blocks already yielded stay delivered.

## `ComplexityEvaluator`

Default thresholds: `slow_tool_call_depth: 1`, `slow_input_tokens: 2000`, `slow_multi_file_threshold: 3`. A request meeting or exceeding any one of tool-call depth, approximate input token volume, or distinct file references routes to `Complexity::Slow`.

## Capacity

`ClientPool` caches one `CucaClient` per `(provider, base_url)` pair and is deliberately uncapped: its size is a property of the deployment, not of traffic. The usage gauge is `ClientPool::len()`.
