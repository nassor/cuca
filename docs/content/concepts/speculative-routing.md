+++
title = "Speculative fast/slow routing"
description = "A deterministic classifier picks the tier, a validator polices the draft, and a latency deadline escalates mid-stream."
template = "page.html"
weight = 4
+++

# Speculative fast/slow routing

<dl class="page-facts">
<dt>In one line</dt>
<dd>Route cheap turns to a fast model, and escalate to a slow one when the request looks hard, the draft looks wrong, or the fast tier stalls</dd>
<dt>You need</dt>
<dd>Nothing running</dd>
<dt>Read this if</dt>
<dd>You want to know what triggers an escalation, and what a caller sees when one happens mid-stream</dd>
</dl>

The orchestrator pairs two models. One is cheap and quick, the other is expensive and capable. It decides which one answers, and it can change its mind while the fast one is already streaming. The feature is `service-speculative`, and [Speculative](@/services/speculative.md) is its reference page.

## Three escalation triggers

<div class="dgm">
<div class="dgm-scroll">
<svg viewBox="0 0 660 190" role="img" aria-labelledby="spec-title spec-desc">
  <title id="spec-title">How a turn reaches the fast or the slow tier</title>
  <desc id="spec-desc">A request first meets the complexity evaluator. A request the evaluator calls Slow goes straight to the slow tier. Otherwise it goes to the fast tier, which streams a draft. From the fast tier a second path leads down to the slow tier, taken when the latency deadline passes or the draft validator rejects a block. Both tiers deliver into the same single stream the caller polls.</desc>
  <defs>
    <marker id="spec-arrow" viewBox="0 0 8 8" refX="7" refY="4"
            markerWidth="6" markerHeight="6" orient="auto">
      <path d="M0 0.6 L7 4 L0 7.4 z" fill="var(--muted-foreground)"/>
    </marker>
  </defs>
  <text class="t-title" x="6" y="20">execute_adaptive_turn</text>
  <rect class="blk" x="6" y="74" width="92" height="44" rx="6"/>
  <text class="t-sm t-mid" x="52" y="92">Unified</text>
  <text class="t-sm t-mid" x="52" y="106">Request</text>
  <rect class="blk blk-ctl" x="122" y="66" width="116" height="60" rx="6"/>
  <text class="t-sm t-mid t-ctl" x="180" y="90">Complexity</text>
  <text class="t-sm t-mid" x="180" y="104">Evaluator</text>
  <rect class="blk blk-data" x="272" y="30" width="120" height="44" rx="6"/>
  <text class="t-sm t-mid t-data" x="332" y="48">fast tier</text>
  <text class="t-sm t-mid" x="332" y="62">fast_model_id</text>
  <rect class="blk blk-bnd" x="272" y="122" width="120" height="44" rx="6"/>
  <text class="t-sm t-mid t-bnd" x="332" y="140">slow tier</text>
  <text class="t-sm t-mid" x="332" y="154">slow_model_id</text>
  <rect class="blk" x="470" y="74" width="112" height="44" rx="6"/>
  <text class="t-sm t-mid" x="526" y="92">one stream</text>
  <text class="t-sm t-mid" x="526" y="106">the caller polls</text>
  <path class="arw" d="M100 96 H118" marker-end="url(#spec-arrow)"/>
  <path class="arw arw-data" d="M240 84 C258 84 252 52 268 52" marker-end="url(#spec-arrow)"/>
  <path class="arw arw-bnd" d="M240 108 C258 108 252 144 268 144" marker-end="url(#spec-arrow)"/>
  <path class="arw arw-data" d="M394 52 C434 52 430 96 466 96" marker-end="url(#spec-arrow)"/>
  <path class="arw arw-bnd" d="M394 144 C434 144 430 96 466 96" marker-end="url(#spec-arrow)"/>
  <path class="arw arw-bnd" d="M332 78 V118" marker-end="url(#spec-arrow)"/>
  <text class="t-sm t-data" x="246" y="34">Fast</text>
  <text class="t-sm t-bnd" x="246" y="176">Slow</text>
  <text class="t-sm t-end" x="324" y="94">latency</text>
  <text class="t-sm t-end" x="324" y="108">or rejection</text>
</svg>
</div>
<p class="dgm-cap"><b>The evaluator runs once, before anything is sent.</b> The vertical path is checked while the fast tier streams, so an escalation can land after the fast tier has already yielded blocks.</p>
</div>

The first trigger runs before dispatch: the evaluator calls the request `Slow` and the fast tier never starts. The other two run mid-stream: the latency deadline passes, or the validator rejects a draft block.

## Classification is arithmetic

`ComplexityEvaluator::evaluate` reads three signals off the request. Any one at or above its threshold routes the turn to the slow tier.

| Signal | What it counts | Default |
|---|---|---|
| `slow_tool_call_depth` | messages holding at least one `ToolCall` or `ToolResult` block | `1` |
| `slow_input_tokens` | approximate tokens across all blocks | `2_000` |
| `slow_multi_file_threshold` | distinct path-like tokens in `Text` blocks, deduplicated per message, summed over messages | `3` |

- Token estimates use four characters per token, floored, with a minimum of 1 per block. `Text`, `Thinking` and serialized `ToolCall` arguments are counted by characters. A `ToolResult` counts as 1 and an `ImageBase64` block as 0.
- A tool depth of `1` means one tool round trip is enough. A turn that has used a tool has structure to maintain, and the fast tier is more likely to break it.
- A token is path-like if it starts with `./` or `/`, or if its last dot suffix is one of 33 known source and config extensions. The list misclassifies: prose mentioning `README.md` counts as a file reference, and a real path with an unlisted extension does not.

A model call to classify the request would be more accurate. It would also cost a model call, add latency to the decision meant to reduce latency, and make routing differ between runs. Arithmetic on the request is cheap, exact and reproducible in a test. The orchestrator uses `ComplexityEvaluator::default()`.

## The latency deadline

`latency_threshold_ms` is a deadline measured from the start of the fast tier's draft. The clock starts once and is never reset by a block arriving. The check runs when the fast stream returns `Pending`, and the comparison is strict: the swap fires when elapsed time is greater than the threshold.

Nothing cancels the fast tier mid-token and nothing retracts what it already sent. A caller can receive several fast-tier blocks and then, in the same stream, blocks from the slow tier. The slow tier continues from the working request, not from the partial answer. The stream stays one `AgentResponseStream`, and the tier change is not visible in the block types.

A fast tier that never returns `Pending` is never checked, so the deadline is a guard on stalls and not a cap on total time.

Buffering the fast output until the deadline and discarding it on a swap would turn every turn into request and response, which removes the reason to have a fast tier. Delivering what arrived is the trade the design makes. With a session store attached, each swap is recorded as `SessionEvent::ModelSwap` with the reason `latency_threshold` or `fallback_validation`.

## The validator checks structure

The orchestrator always uses `JsonToolDraftValidator`; it has no setter for another. The validator checks three things:

- A `ToolCall` must have a non-empty id and a non-empty name.
- A `ToolCall` whose arguments arrive as a JSON string must parse.
- A `Text` block that parses as JSON must be a JSON object, not a bare scalar.

It does not check that the tool exists, that the arguments match a schema, or that the answer is correct. Those belong to other tiers: `plugin-guardrails` validates against JSON Schema and `plugin-hitl` puts a human in the path. Tying escalation to policy would answer a policy failure with "ask a bigger model", which is the wrong remedy.

Validation applies to blocks from both tiers. When a block is rejected and `fallback_on_tool_error` is set, the orchestrator appends the rejection to the working request as a `ToolResult` message and runs the slow tier again. For a rejected `ToolCall` the result carries that call's id. For any other block the id is empty.

The budget is two cascades per turn, shared across both tiers. Once it is spent, or when `fallback_on_tool_error` is unset, the rejection reaches the caller as `CucaError::Provider` with the reason. Its `provider` is the fast tier's endpoint for a draft rejection and the slow tier's for a slow rejection.

## What has no default

`SwappableModelPair` has no `Default`. The caller names all six fields: `fast_provider`, `fast_model_id`, `slow_provider`, `slow_model_id`, `latency_threshold_ms` and `fallback_on_tool_error`. No default answers "which model is your fast one", and a wrong guess spends the caller's money on the wrong tier.

## The client pool

Each tier resolves to a `CucaClient` from a `ClientPool`, keyed by provider and base URL. The pool has no cap. It holds one entry per distinct endpoint a caller configures, which configuration sets and traffic does not. `ClientPool::len` reports the count. [Memory discipline](@/concepts/memory-discipline.md) explains the exception.

Next page: [Memory discipline](@/concepts/memory-discipline.md).
