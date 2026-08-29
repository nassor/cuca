+++
title = "Speculative fast/slow routing"
description = "A deterministic classifier picks the tier, a validator polices the draft, and a latency deadline escalates mid-stream."
template = "page.html"
weight = 4
+++

# Speculative fast/slow routing

<dl class="page-facts">
<dt>In one line</dt>
<dd>Route cheap turns to a fast model, escalate to a slow one when the request looks hard, the draft looks wrong, or the fast tier stalls</dd>
<dt>You need</dt>
<dd>Nothing running</dd>
<dt>Read this if</dt>
<dd>You want to know what triggers an escalation, and what a caller sees when one happens mid-stream</dd>
</dl>

Two models, paired. One is cheap and quick, one is expensive and capable. The
orchestrator decides which answers, and can change its mind while the cheap one
is already talking.

## The three escalation triggers

<div class="dgm">
<div class="dgm-scroll">
<svg viewBox="0 0 660 190" role="img" aria-labelledby="spec-title spec-desc">
  <title id="spec-title">How a turn reaches the fast or the slow tier</title>
  <desc id="spec-desc">A request first meets the complexity evaluator. A request the
  evaluator calls Slow goes straight to the slow tier. Otherwise it goes to the fast
  tier, which streams a draft. From the fast tier a second path leads down to the slow
  tier, taken when the latency deadline passes or the draft validator rejects a block.
  Both tiers deliver into the same single stream the caller polls.</desc>
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
<p class="dgm-cap"><b>The evaluator runs once, before anything is sent.</b> The
vertical path is checked on every poll of the fast tier's stream, so an
escalation can land after the fast tier has already yielded blocks.</p>
</div>

Trigger one runs before dispatch. Triggers two and three run mid-stream.

## Classification is arithmetic, not a model call

`ComplexityEvaluator::evaluate` reads three signals off the request and compares
each to a threshold. Any one of them at or above its threshold routes to the
slow tier:

| Signal | What it counts | Default threshold |
|---|---|---|
| `slow_tool_call_depth` | messages containing at least one `ToolCall` or `ToolResult` block | `1` |
| `slow_input_tokens` | approximate tokens across every block, at four characters per token | `2_000` |
| `slow_multi_file_threshold` | distinct path-like tokens in `Text` blocks, deduplicated per message | `3` |

A tool-call depth threshold of `1` means a single completed tool round trip is
enough. That is aggressive on purpose: a turn that has already used a tool is a
turn with structure to maintain, and the fast tier is the one more likely to
break it.

Path detection is a fixed list. A token counts if it starts with `./` or `/`, or
if its final dot-suffix is one of thirty-three known source extensions. That
list will misclassify. A prose sentence mentioning `README.md` counts as a file
reference, and a real path with an unlisted extension does not.

Asking a model to classify the request would be more accurate and would also
cost a model call, add latency to the decision meant to reduce latency, and make
routing nondeterministic across runs. Arithmetic on the request is cheap, exact,
and reproducible in a test. Three tests in the suite assert the boundary values
directly, including that an 8000-character message lands exactly on the 2000
token threshold.

## The latency deadline is checked, not enforced

`latency_threshold_ms` is the budget the fast tier gets to produce its next
block. The check happens when the fast stream returns `Pending`, and it is a
strict comparison: the swap fires once elapsed time is greater than the
threshold, not equal to it.

The consequence is worth stating plainly. Nothing cancels the fast tier
mid-token, and nothing retracts what it already sent. A caller can therefore
receive several fast-tier blocks and then, in the same stream, blocks from the
slow tier that continue from the original request rather than from the partial
answer. The stream stays a single continuous `AgentResponseStream`, and the tier
change is invisible in the block types.

The alternative would be buffering the fast tier's output until the deadline
passes, then discarding it on a swap. That converts streaming into
request-response for every turn, which defeats the reason a fast tier exists.
Delivering what arrived is the honest trade, and the two swap reasons are
recorded so the seam is auditable rather than silent.

## The validator polices structure, not meaning

The default `DraftValidator` is `JsonToolDraftValidator`, and it checks three
things: a `ToolCall` must have a non-empty id, a `ToolCall` whose arguments are
carried as a JSON text must parse, and a `Text` block that happens to be valid
JSON must be a JSON object rather than a bare scalar.

That is all. It does not check the tool exists, that the arguments match a
schema, or that the answer is correct. Semantic validation is a different job
with a different owner: `plugin-guardrails` validates against JSON Schema, and
`plugin-hitl` puts a human in the path. Overloading the draft validator with
either would tie escalation to policy, and a policy failure would then be
answered by "ask a bigger model", which is the wrong remedy.

When a block is rejected and `fallback_on_tool_error` is set, the orchestrator
appends the rejection to the working request as a tool result and re-routes to
the slow tier. That budget is two cascades. On exhaustion, or with
`fallback_on_tool_error` unset, the rejection surfaces as `CucaError::Provider`
carrying the reason.

## What has no default

`SwappableModelPair` has no `Default` impl. All six fields, including both model
ids, the latency threshold and the fallback switch, must be named by the caller.
There is no defensible default for "which model is your fast one", and a wrong
guess spends the caller's money on the wrong tier.

`ComplexityEvaluator` does have defaults, and the orchestrator uses them.

## What the client pool holds

Each tier resolves to a `CucaClient` from a `ClientPool`, keyed by provider and
base URL. The pool is deliberately uncapped: one entry per distinct endpoint a
caller configures, which is a small fixed number set by configuration rather
than by traffic. `ClientPool::len` reports the count. That reasoning, and the
rule it is an exception to, are in
[Memory discipline](@/concepts/memory-discipline.md).

Next page: [Memory discipline](@/concepts/memory-discipline.md).
