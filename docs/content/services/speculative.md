+++
title = "Speculative"
description = "The fast/slow model orchestrator: complexity routing, the draft and fallback pipeline, and the client pool."
template = "page.html"
weight = 2
+++

# Speculative

<dl class="page-facts">
<dt>In one line</dt>
<dd>Routes a turn to a fast or slow model tier by complexity, drafts on the fast tier, and falls back to the slow tier on a rejected block or a missed latency deadline.</dd>
<dt>You need</dt>
<dd>The <code>service-speculative</code> feature and a <code>SwappableModelPair</code>.</dd>
<dt>Read this if</dt>
<dd>You attach a <code>ModelOrchestrator</code> to a client or tune its complexity thresholds. For why the tiers escalate, read <a href="@/concepts/speculative-routing.md">Speculative fast/slow routing</a>.</dd>
</dl>

## Feature and entry type

| Item | Value |
|---|---|
| Feature | `service-speculative` |
| Own dependencies | `dep:tokio`, `dep:tokio-stream` |
| Declared plugin edge | none in `Cargo.toml`; documented-optional on `plugin-session-log` |
| Entry type | `ModelOrchestrator` |
| Module | `cuca::services::orchestrator` |
| Attachment | `CucaClientBuilder::with_orchestrator(orchestrator)` or direct calls |

`ModelOrchestrator` never implements `CucaPlugin`. Passing it to `register_plugin` is a compile error.

The documented-optional edge is `ModelOrchestrator::with_session_store`. It exists only when `plugin-session-log` is also enabled and is compiled out otherwise, so a swap is never silently dropped. The session log plugin does not know this service exists.

Supporting types: `SwappableModelPair`, `ClientPool`, `Complexity`, `ComplexityEvaluator`, `DraftValidator`, `JsonToolDraftValidator`, `TurnExecutor`, `PoolTurnExecutor`.

## Entry points

`ModelOrchestrator` constructors:

- `new(config, pool)` builds the default `ComplexityEvaluator`, the default `JsonToolDraftValidator` and pool-backed tier executors. Both tiers start on their provider adapter's default endpoint.
- `with_executors(config, pool, fast, slow)` injects `TurnExecutor`s directly. The validator and evaluator stay at their defaults, and no session store is attached.
- `with_endpoint(base_url, api_key)` points both pooled tiers at `base_url`, with `api_key` as the credential when given. It rebuilds the tier executors, so it replaces any injected by `with_executors`.
- `with_session_store(store, session_id)` attaches a [session log](@/plugins/session-log.md) store that receives `SessionEvent::ModelSwap` records. It requires `plugin-session-log`.

Running a turn:

- `execute_adaptive_turn(request).await` runs the three stage pipeline and returns an `AgentResponseStream`.
- `client_pool()` returns the shared `ClientPool`.

### Attaching to a client

With `CucaClientBuilder::with_orchestrator`, `CucaClient::generate_stream` runs the `on_request` hooks, then hands the turn to `execute_adaptive_turn` instead of a provider adapter.

- The orchestrator stream runs no top-level `on_stream_chunk` or `on_response_complete` hooks.
- Exception: with `service-prompt-cache` and a cache configured, a cache miss wraps the stream in the usual instrumentation, so the terminal hooks and the cache write run for that turn.
- A cache hit returns before the orchestrator is reached.
- `with_base_url` and `with_api_key` on the client configure that client, not the tier clients. The tier endpoint is `with_endpoint`.
- The pool clients are built without an orchestrator, so tier turns dispatch straight to providers and never re-enter it.

### Pipeline

1. Complexity routing. A `Complexity::Slow` request goes straight to the slow tier with no draft phase.
2. Speculative draft. The fast tier streams, and each block passes through `DraftValidator`. A rejected block is appended to a working copy of the request as a `ToolResult` message so the next tier sees what was refused and why.
3. Fallback cascade. With `fallback_on_tool_error` set, a rejection re-routes the remainder of the turn to the slow tier, up to two cascades. Blocks from the slow tier are validated too, so a second rejection spends the second cascade.

Blocks already yielded stay delivered when the turn swaps tier.

### Latency swap

The latency guard runs only when the fast stream reports `Pending`. At the first such poll after more than `latency_threshold_ms` has elapsed since the draft began, the turn swaps to the slow tier. A tier stream that is woken only when a block is ready never presents that poll, and the turn stays on the fast tier.

The latency swap does not depend on `fallback_on_tool_error`. Every swap appends a `ModelSwap` record when a session store is attached, with reason `latency_threshold` or `fallback_validation`. A failed append is written to stderr and does not fail the turn.

### Routing rules

`ComplexityEvaluator::evaluate` returns `Complexity::Slow` when any one indicator meets or exceeds its threshold, checked in this order:

- Tool call depth counts the messages that carry a `ToolCall` or `ToolResult` block.
- Input tokens are estimated as `chars / 4` per text, thinking or tool argument block (minimum 1), with 1 per tool result and 0 per image. No tokenizer is involved.
- File references count distinct path like tokens per message: tokens starting with `./` or `/`, or a `word.ext` token with a known source or config extension.

### Draft validation

`JsonToolDraftValidator` is the default `DraftValidator`:

- a `ToolCall` needs a non-empty `id` and `name`, and string `arguments` must parse as JSON;
- a `Text` block that parses as JSON must parse to an object, and any other text passes;
- every other block passes.

A custom `DraftValidator` returns `Err(reason)` to reject a block. Rejection of low confidence output is a supported use of that seam.

## Configuration

`SwappableModelPair` has no `Default`; every field is required.

| Key | Type | Meaning |
|---|---|---|
| `fast_provider`<br>`fast_model_id` | `ProviderEndpoint`<br>`String` | The low latency tier. |
| `slow_provider`<br>`slow_model_id` | `ProviderEndpoint`<br>`String` | The high capacity tier. |
| `latency_threshold_ms` | `u64` | Draft time after which the first `Pending` poll swaps to the slow tier. |
| `fallback_on_tool_error` | `bool` | Whether a rejected block re-routes to the slow tier. |

`ComplexityEvaluator` fields, all public:

| Key | Type | Default |
|---|---|---|
| `slow_tool_call_depth` | `usize` | `1` |
| `slow_input_tokens` | `usize` | `2000` |
| `slow_multi_file_threshold` | `usize` | `3` |

- The orchestrator endpoint defaults to an empty base URL and no key, which leaves each provider adapter on its own default endpoint.
- The evaluator and validator are fixed at `ComplexityEvaluator::default()` and `JsonToolDraftValidator` by every constructor; no builder method replaces them.

## Caps and at-cap policy

- Bound: `ClientPool` is uncapped. It holds one `CucaClient` per distinct `(provider, base_url)` pair, so its size follows the deployment, and repeated requests add no entries.
- At-cap policy: none, because no entry is ever evicted. A caller that creates a distinct base URL per request owns that bound and builds a client per tenant instead.
- Usage gauge: `ClientPool::len()` is O(1); `is_empty()` is the companion.
- Turn bound: at most 2 slow tier fallback cascades per turn (`FALLBACK_ATTEMPTS`).

## Errors

- `execute_adaptive_turn` returns the tier executor's `CucaError` when a tier fails to start.
- A rejected block with no cascade left, or with `fallback_on_tool_error` unset, ends the stream with `CucaError::Provider { provider, message }`. The message is the validator's reason, and `provider` is the tier that produced the block.
- `ClientPool::get_or_create` returns `CucaError::Config` when the pool lock is poisoned, plus whatever `CucaClientBuilder::build` rejects.
- A provider feature that is not compiled does not fail at build time. It surfaces as `CucaError::ProviderNotEnabled` at `generate_stream` time.
- `TurnExecutor::execute` needs a running tokio runtime, which `execute_adaptive_turn` always has.

The default validator's reasons are `tool call id must be non-empty`, `tool call name must be non-empty`, `tool call arguments are not valid JSON: ...` and `text block is valid JSON but not a JSON object`. Variants of `CucaError` are listed in [Error types](@/reference/errors.md).

## Example

```rust,name=Attach an orchestrator to a client
let config = SwappableModelPair {
    fast_provider: ProviderEndpoint::LlamaCpp,
    fast_model_id: "fast-tier-id".into(),
    slow_provider: ProviderEndpoint::LlamaCpp,
    slow_model_id: "slow-tier-id".into(),
    latency_threshold_ms: 60_000,
    fallback_on_tool_error: true,
};
let orchestrator = ModelOrchestrator::new(config, Arc::new(ClientPool::default()))
    .with_endpoint("http://127.0.0.1:1234/v1", None);
let client = CucaClient::builder()
    .with_provider(ProviderEndpoint::LlamaCpp)
    .with_orchestrator(orchestrator)
    .build()?;
```

`examples/speculative.rs` is the full program. It routes three requests with no network, serves a simple one from the fast tier, triggers a fallback cascade with a reply the default validator refuses, and sends a multi file request straight to the slow tier. Both tiers run through injected `TurnExecutor`s, so every dispatch prints. It needs a `llama-server` on port 1234 serving both tier models. `CUCA_BASE_URL`, `CUCA_FAST_MODEL` and `CUCA_SLOW_MODEL` retarget it.

```bash,name=Runs the same on all three platforms
cargo run --example speculative --features "provider-llamacpp service-speculative"
```

## Related pages

- [Speculative fast/slow routing](@/concepts/speculative-routing.md) explains why the tiers escalate.
- [Session log](@/plugins/session-log.md) records the `ModelSwap` events.
- [Prompt cache](@/services/prompt-cache.md) interacts with the orchestrator path.
- [Services overview](@/services/_index.md) lists all six services.
