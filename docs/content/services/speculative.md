+++
title = "Speculative"
description = "Answer easy turns on a fast model and escalate to a slow model when the draft is rejected or too late."
template = "page.html"
weight = 2
+++

# Speculative

<dl class="page-facts">
<dt>In one line</dt>
<dd>Sends each turn to a fast model or a slow model by how hard the request looks. The fast model drafts, and the slow model takes over if a block is rejected or the draft is too slow.</dd>
<dt>You need</dt>
<dd>The <code>service-speculative</code> feature, one <code>provider-*</code> feature, and two model ids: one fast, one slow.</dd>
<dt>Read this if</dt>
<dd>You attach a <code>ModelOrchestrator</code> to a client or tune when a request counts as hard. For the reasoning behind the design, read <a href="@/concepts/speculative-routing.md">Speculative fast/slow routing</a>.</dd>
</dl>

## What it does

A small model is quick and cheap but sometimes wrong. A large model is slow and strong. `ModelOrchestrator` uses both. It first judges the request. A hard request goes straight to the slow model. An easy one goes to the fast model, and every block of the draft is checked. If a block fails the check, or the draft takes too long, the rest of the turn moves to the slow model.

<div class="dgm">
<div class="dgm-scroll">
<svg viewBox="0 0 660 210" role="img" aria-labelledby="sp-flow-title sp-flow-desc">
  <title id="sp-flow-title">A request is judged hard or easy. Easy goes to the fast model, whose blocks are validated. Hard, or rejected, goes to the slow model.</title>
  <desc id="sp-flow-desc">Five boxes. The complexity evaluator sends a Fast request to the fast tier, which drafts a reply. A validator checks each block, and valid blocks stream to you. A Slow request skips the draft and goes straight to the slow tier. A rejected block, or a draft that runs past the latency threshold, also moves the turn to the slow tier, at most two times. The slow tier streams to you.</desc>
  <defs>
    <marker id="sp-flow-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M0 0.6 L7 4 L0 7.4 z" fill="var(--muted-foreground)"/>
    </marker>
  </defs>
  <text class="t-title" x="6" y="20">ModelOrchestrator::execute_adaptive_turn</text>
  <rect class="blk blk-ctl" x="6" y="50" width="110" height="50" rx="6"/>
  <text class="t-sm t-mid t-ctl" x="61" y="72">evaluate</text>
  <text class="t-sm t-mid" x="61" y="87">complexity</text>
  <path class="arw arw-ctl" d="M118 75 H178" marker-end="url(#sp-flow-arrow)"/>
  <text class="t-sm" x="132" y="68">Fast</text>
  <rect class="blk" x="180" y="50" width="120" height="50" rx="6"/>
  <text class="t-sm t-mid" x="240" y="72">fast tier</text>
  <text class="t-sm t-mid" x="240" y="87">drafts reply</text>
  <path class="arw" d="M302 75 H338" marker-end="url(#sp-flow-arrow)"/>
  <rect class="blk blk-ctl" x="340" y="50" width="110" height="50" rx="6"/>
  <text class="t-sm t-mid t-ctl" x="395" y="72">validate</text>
  <text class="t-sm t-mid" x="395" y="87">each block</text>
  <path class="arw arw-data" d="M452 75 H498" marker-end="url(#sp-flow-arrow)"/>
  <text class="t-sm" x="468" y="68">ok</text>
  <rect class="blk blk-data" x="500" y="50" width="154" height="50" rx="6"/>
  <text class="t-sm t-mid t-data" x="577" y="72">stream to you</text>
  <text class="t-sm t-mid" x="577" y="87">blocks delivered</text>
  <path class="arw" d="M61 102 V172 H338" marker-end="url(#sp-flow-arrow)"/>
  <text class="t-sm" x="70" y="166">Slow: skip the draft</text>
  <path class="arw" d="M395 102 V148" marker-end="url(#sp-flow-arrow)"/>
  <text class="t-sm" x="403" y="128">rejected or too slow</text>
  <rect class="blk" x="340" y="150" width="314" height="44" rx="6"/>
  <text class="t-sm t-mid" x="497" y="176">slow tier, at most 2 cascades per turn</text>
  <path class="arw arw-data" d="M580 148 V102" marker-end="url(#sp-flow-arrow)"/>
</svg>
</div>
<p class="dgm-cap"><b>Blocks already sent to you stay sent.</b> When a block is rejected, the slow tier gets the request again with a note of which block was refused and why.</p>
</div>

## Try it

This program judges two requests with the built-in evaluator. It needs no model and no server.

```rust,name=See which tier a request goes to
use cuca::{ComplexityEvaluator, UnifiedRequest};

fn main() {
    let evaluator = ComplexityEvaluator::default();

    let short = UnifiedRequest::new("any-model")
        .add_user_message("The build finished.");
    let files = UnifiedRequest::new("any-model").add_user_message(
        "Which of src/sse.rs, src/client.rs, src/request.rs and src/types.rs parses server-sent events?",
    );

    println!("short sentence: {:?}", evaluator.evaluate(&short));
    println!("four file references: {:?}", evaluator.evaluate(&files));
}
```

```text,name=Expected output
short sentence: Fast
four file references: Slow
```

To use both tiers on a client, build a `SwappableModelPair` and attach the orchestrator:

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

`examples/speculative.rs` is the full program. It routes three requests, serves a simple one from the fast tier, triggers a fallback with a reply the default validator refuses, and sends a multi-file request straight to the slow tier. It needs a `llama-server` on port 1234 serving both tier models. The flags `--base-url`, `--fast-model` and `--slow-model`, written after `--`, point it at any OpenAI-compatible server.

```bash,name=Runs the same on all three platforms
cargo run --example speculative --features "provider-llamacpp service-speculative"
```

## Feature and entry type

| Item | Value |
|---|---|
| Feature | `service-speculative` |
| Own dependencies | `dep:tokio`, `dep:tokio-stream` |
| Plugin it needs | none in `Cargo.toml`. Optional: `plugin-session-log` |
| Entry type | `ModelOrchestrator` |
| Module | `cuca::services::orchestrator` |
| Attach with | `CucaClientBuilder::with_orchestrator(orchestrator)`, or call it directly |

The optional link is `ModelOrchestrator::with_session_store`. It exists only when `plugin-session-log` is also on. It records each tier swap as a `SessionEvent::ModelSwap`.

Supporting types: `SwappableModelPair`, `ClientPool`, `Complexity`, `ComplexityEvaluator`, `DraftValidator`, `JsonToolDraftValidator`, `TurnExecutor`, `PoolTurnExecutor`.

## Entry points

`ModelOrchestrator` constructors:

| Constructor | What it does |
|---|---|
| `new(config, pool)` | builds the default `ComplexityEvaluator`, the default `JsonToolDraftValidator` and pool-backed tier executors. Both tiers start on their provider's default endpoint. |
| `with_executors(config, pool, fast, slow)` | injects your own `TurnExecutor` for each tier. The validator and evaluator stay at their defaults, and no session store is attached. |
| `with_endpoint(base_url, api_key)` | points both pooled tiers at `base_url`, with `api_key` when given. It rebuilds the tier executors, so it replaces any you injected with `with_executors`. |
| `with_session_store(store, session_id)` | attaches a [session log](@/plugins/session-log.md) store that receives `SessionEvent::ModelSwap` records. Needs `plugin-session-log`. |

Running a turn:

- `execute_adaptive_turn(request).await` runs the three steps below and returns an `AgentResponseStream`.
- `client_pool()` returns the shared `ClientPool`.

### Attached to a client

With `CucaClientBuilder::with_orchestrator`, `CucaClient::generate_stream` runs the `on_request` hooks, then hands the turn to `execute_adaptive_turn` instead of a provider adapter.

- The orchestrator stream runs no top-level `on_stream_chunk` or `on_response_complete` hooks.
- One exception: with `service-prompt-cache` and a cache configured, a cache miss wraps the stream in the usual hooks, so the end-of-reply hooks and the cache write run for that turn.
- A cache hit returns before the orchestrator is reached.
- `with_base_url` and `with_api_key` on the client set that client, not the tier clients. Set the tier endpoint with `with_endpoint`.
- The tier clients are built without an orchestrator, so a tier turn goes straight to its provider and never re-enters it.

### The three steps

1. Complexity routing. A `Complexity::Slow` request goes straight to the slow tier, with no draft.
2. Draft. The fast tier streams, and each block goes through `DraftValidator`. A rejected block is added to a working copy of the request as a `ToolResult` message, so the next tier sees what was refused and why.
3. Fallback. With `fallback_on_tool_error` set, a rejection moves the rest of the turn to the slow tier, at most two times. Blocks from the slow tier are validated too, so a second rejection uses the second fallback.

### Latency swap

The latency check runs only when the fast stream reports `Pending`. At the first such poll after `latency_threshold_ms` has passed since the draft began, the turn swaps to the slow tier. A tier stream that is only woken when a block is ready never offers that poll, and the turn stays on the fast tier.

The latency swap does not depend on `fallback_on_tool_error`. Every swap appends a `ModelSwap` record when a session store is attached, with the reason `latency_threshold` or `fallback_validation`. A failed append is written to stderr and does not fail the turn.

### How a request is judged

`ComplexityEvaluator::evaluate` returns `Complexity::Slow` when any one indicator reaches its threshold. It checks them in this order:

- Tool call depth: the number of messages that carry a `ToolCall` or `ToolResult` block.
- Input tokens, estimated as `chars / 4` for each text, thinking or tool argument block (at least 1). A tool result counts 1 and an image counts 0. No tokenizer is used.
- File references: the distinct path-like words in each message. A word is path-like if it starts with `./` or `/`, or is a `word.ext` token with a known source or config extension.

### How a draft is checked

`JsonToolDraftValidator` is the default `DraftValidator`:

- A `ToolCall` needs a non-empty `id` and `name`, and string `arguments` must parse as JSON.
- A `Text` block that parses as JSON must parse to an object. Any other text passes.
- Every other block passes.

Your own `DraftValidator` returns `Err(reason)` to reject a block. Rejecting low-confidence output is a supported use.

## Configuration

`SwappableModelPair` has no `Default`. Every field is required.

| Key | Type | Meaning |
|---|---|---|
| `fast_provider`<br>`fast_model_id` | `ProviderEndpoint`<br>`String` | The low-latency tier. |
| `slow_provider`<br>`slow_model_id` | `ProviderEndpoint`<br>`String` | The high-capacity tier. |
| `latency_threshold_ms` | `u64` | The draft time after which the first `Pending` poll swaps to the slow tier. |
| `fallback_on_tool_error` | `bool` | Whether a rejected block moves the turn to the slow tier. |

`ComplexityEvaluator` fields, all public:

| Key | Type | Default |
|---|---|---|
| `slow_tool_call_depth` | `usize` | `1` |
| `slow_input_tokens` | `usize` | `2000` |
| `slow_multi_file_threshold` | `usize` | `3` |

- The orchestrator endpoint starts with an empty base URL and no key, so each provider adapter uses its own default endpoint.
- Every constructor uses `ComplexityEvaluator::default()` and `JsonToolDraftValidator`. No builder method replaces them.

## Caps and at-cap policy

- Pool: `ClientPool` has no cap. It holds one `CucaClient` per distinct `(provider, base_url)` pair, so its size follows your deployment, and repeated requests add no entries. If you use a different base URL for every request, you own that growth: build one client per tenant instead.
- Gauge: `ClientPool::len()` is O(1). `is_empty()` is its pair.
- Turn bound: at most 2 slow-tier fallbacks per turn (`FALLBACK_ATTEMPTS`).

## Errors

- `execute_adaptive_turn` returns the tier executor's `CucaError` when a tier fails to start.
- A rejected block with no fallback left, or with `fallback_on_tool_error` unset, ends the stream with `CucaError::Provider { provider, message }`. The message is the validator's reason, and `provider` is the tier that produced the block.
- `ClientPool::get_or_create` returns `CucaError::Config` when the pool lock is poisoned, plus whatever `CucaClientBuilder::build` rejects.
- A provider feature that is not compiled does not fail at build time. It shows up as `CucaError::ProviderNotEnabled` when `generate_stream` runs.
- `TurnExecutor::execute` needs a running tokio runtime, which `execute_adaptive_turn` always has.

The default validator's reasons are `tool call id must be non-empty`, `tool call name must be non-empty`, `tool call arguments are not valid JSON: ...` and `text block is valid JSON but not a JSON object`. [Error types](@/reference/errors.md) lists the `CucaError` variants.

## Related pages

- [Speculative fast/slow routing](@/concepts/speculative-routing.md) explains why the tiers escalate.
- [Session log](@/plugins/session-log.md) records the `ModelSwap` events.
- [Prompt cache](@/services/prompt-cache.md) interacts with the orchestrator path.
- [Services](@/services/_index.md) lists all seven services.
