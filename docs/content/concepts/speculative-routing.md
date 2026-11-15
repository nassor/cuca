+++
title = "Speculative fast/slow routing"
description = "Answer easy turns with a fast model, and move to a slow one when the request looks hard, the draft looks wrong, or the fast model stalls."
template = "page.html"
weight = 4
+++

# Speculative fast/slow routing

<dl class="page-facts">
<dt>In one line</dt>
<dd>Route cheap turns to a fast model, and escalate to a slow one when the request looks hard, the draft looks wrong, or the fast model stalls</dd>
<dt>You need</dt>
<dd>Nothing running</dd>
<dt>Read this if</dt>
<dd>You want to know what triggers an escalation, and what you see when one happens mid-stream</dd>
</dl>

Big models answer better and cost more time and money. Small models are quick and cheap, and they fail on hard tasks. Speculative routing uses both. A fast model starts the answer, and CUCA hands the turn to a slow model when it has a reason.

The orchestrator holds the pair. It decides which one answers, and it can change its mind while the fast one is already streaming. The feature is `service-speculative`, and [Speculative](@/services/speculative.md) is its reference page.

## Three reasons to escalate

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
<p class="dgm-cap"><b>The evaluator runs once, before anything is sent.</b> The path from the fast tier to the slow tier is checked while the fast tier streams, so a switch can happen after some fast blocks have already arrived.</p>
</div>

1. The evaluator reads the request and calls it `Slow`. The fast model never starts.
2. The fast model is too slow: the latency deadline passes.
3. The validator rejects a block of the fast model's draft.

The first happens before sending. The other two happen mid-stream.

## Classification is arithmetic

`ComplexityEvaluator::evaluate` counts three things in the request. If any one reaches its threshold, the turn goes to the slow tier.

| Signal | What it counts | Default |
|---|---|---|
| `slow_tool_call_depth` | messages holding at least one `ToolCall` or `ToolResult` block | `1` |
| `slow_input_tokens` | approximate tokens across all blocks | `2_000` |
| `slow_multi_file_threshold` | distinct path-like words in `Text` blocks, counted once per message and summed | `3` |

- A token is a small piece of text. CUCA estimates four characters per token, rounded down, with a minimum of 1 per block. `Text`, `Thinking` and `ToolCall` arguments count by characters. A `ToolResult` counts as 1 and an `ImageBase64` block as 0.
- A tool depth of `1` means one tool round trip is enough. A turn that used a tool has structure to keep intact, and the fast model is more likely to break it.
- A word is path-like if it starts with `./` or `/`, or if its last dot suffix is one of 33 known source and config extensions. The test is rough: prose that mentions `README.md` counts as a file, and a real path with an unlisted extension does not.

Asking a model to classify the request would be more accurate. It would also cost a model call, add delay to the decision meant to save delay, and give different routes on different runs. Arithmetic is cheap, exact and repeatable in a test. The orchestrator uses `ComplexityEvaluator::default()`.

## The latency deadline

`latency_threshold_ms` is a deadline measured from the start of the fast model's draft. The clock starts once and a new block does not reset it. CUCA checks it when the fast stream has nothing ready. The switch happens when elapsed time is greater than the threshold.

<div class="dgm">
<div class="dgm-scroll">
<svg viewBox="0 0 660 150" role="img" aria-labelledby="swap-title swap-desc">
  <title id="swap-title">A mid-stream switch from the fast tier to the slow tier</title>
  <desc id="swap-desc">A time axis runs left to right. First the fast tier's blocks arrive. At the deadline, or when the validator rejects a block, the stream switches. Then the slow tier's blocks arrive. All of them reach the caller through one stream.</desc>
  <text class="t-title" x="6" y="20">Time</text>
  <text class="t-sm t-data" x="6" y="46">fast tier blocks, already delivered</text>
  <text class="t-sm t-bnd" x="316" y="46">slow tier blocks, from the working request</text>
  <rect class="bar bar-data" x="6" y="54" width="294" height="22" rx="3"/>
  <rect class="bar bar-bnd" x="304" y="54" width="350" height="22" rx="3"/>
  <path class="mark" d="M302 30 V92"/>
  <text class="t-sm t-mid" x="302" y="108">deadline passed, or draft rejected</text>
  <path class="ln" d="M6 120 H654"/>
  <text class="t-ax" x="6" y="138">the caller reads one AgentResponseStream and sees no break</text>
</svg>
</div>
<p class="dgm-cap"><b>Nothing the fast tier sent is taken back.</b> You can receive several fast blocks, then slow blocks in the same stream.</p>
</div>

Nothing cancels the fast model mid-token and nothing retracts what it already sent. The slow model continues from the working request, not from the partial answer. The block types do not show which tier wrote a block.

A fast stream that never has to wait is never checked. The deadline guards against stalls. It does not cap total time.

Holding the fast output back until the deadline, then dropping it on a swap, would turn every turn into request and response. That removes the reason to have a fast tier. With a session store attached, each swap is recorded as `SessionEvent::ModelSwap` with the reason `latency_threshold` or `fallback_validation`.

## The validator checks structure

The orchestrator always uses `JsonToolDraftValidator`. It has no setter for another. It checks three things:

- A `ToolCall` must have a non-empty id and a non-empty name.
- A `ToolCall` whose arguments arrive as a JSON string must parse.
- A `Text` block that parses as JSON must be a JSON object, not a bare value.

It does not check that the tool exists, that the arguments match a schema, or that the answer is right. Those jobs belong to other features: `plugin-guardrails` checks against a JSON Schema, and `plugin-hitl` puts a human in the path. Answering a policy failure with "ask a bigger model" would be the wrong fix.

The validator checks blocks from both tiers. When it rejects a block and `fallback_on_tool_error` is set, the orchestrator adds the rejection to the working request as a `ToolResult` message and runs the slow tier again. For a rejected `ToolCall`, the result carries that call's id. For any other block the id is empty.

Each turn has a budget of two such retries, shared by both tiers. When the budget is spent, or `fallback_on_tool_error` is unset, you get `CucaError::Provider` with the reason. Its `provider` is the fast tier's endpoint for a draft rejection and the slow tier's for a slow rejection.

## What has no default

`SwappableModelPair` has no `Default`. You name all six fields: `fast_provider`, `fast_model_id`, `slow_provider`, `slow_model_id`, `latency_threshold_ms` and `fallback_on_tool_error`. No default can answer "which model is your fast one", and a wrong guess spends your money on the wrong tier.

## The client pool

Each tier gets a `CucaClient` from a `ClientPool`, keyed by provider and base URL. The pool has no size cap. It holds one client per distinct endpoint you configure, and `ClientPool::len` reports the count. [Memory discipline](@/concepts/memory-discipline.md) explains why this is allowed.

Next page: [Memory discipline](@/concepts/memory-discipline.md).
